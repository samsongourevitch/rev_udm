from dataclasses import dataclass
import time

import hydra
import lightning as L
import torch

import ddm


@dataclass
class LossInfo:
    loss: torch.Tensor
    nlls: torch.Tensor
    num_tokens: torch.Tensor


_DIFFUSION_CLASS_NAMES = {
    "mdlm": "MDLM",
    "udlm": "UDLM",
    "reaudm": "ReAUDM",
    "audm": "AUDM",
    "max_coupling": "MaximalCoupling",
    "remdlm": "ReMDLM",
}


def resolve_diffusion_model(name):
    class_name = _DIFFUSION_CLASS_NAMES.get(name)
    if class_name is None:
        raise ValueError(f"Model not implemented: {name}")

    diffusion_model = getattr(ddm, class_name, None)
    if diffusion_model is None:
        raise ValueError(
            f"Configured diffusion algo '{name}' expects ddm.{class_name}, "
            "but that class is not available."
        )
    return diffusion_model


class LitDiffusionBase(L.LightningModule):
    """Shared Lightning utilities for token diffusion trainers."""

    def __init__(self):
        super().__init__()
        self.ema = None
        self._throughput_tokens = None
        self._throughput_start_time = None

    def _sample_t(self, batch_size, current_accumulation_step=None):
        if current_accumulation_step is not None:
            local_batch_size = batch_size
            num_nodes = int(self.trainer.num_nodes)
            num_devices = int(self.trainer.num_devices)
            accumulation_steps = int(self.trainer.accumulate_grad_batches)
            batch_size = (
                local_batch_size * num_nodes * num_devices * accumulation_steps
            )

        t = torch.rand(batch_size, device=self.device)
        if self.config.training.antithetic_sampling:
            offset = torch.arange(batch_size, device=self.device) / batch_size
            t = (t / batch_size + offset) % 1

        t = (
            1 - self.config.training.sampling_eps
        ) * t + self.config.training.sampling_eps

        if current_accumulation_step is not None:
            t = t.view(
                num_nodes,
                num_devices,
                accumulation_steps,
                local_batch_size,
            )
            t = t[
                int(self.trainer.node_rank),
                int(self.trainer.local_rank),
                int(current_accumulation_step),
            ]

        return t.unsqueeze(-1)

    def _current_accumulation_step(self, batch_idx):
        return batch_idx % self.trainer.accumulate_grad_batches

    def _model_prediction_type(self):
        return self.diffusion.model_prediction

    def _compute_loss_info(
        self,
        *,
        logits,
        x0,
        xt,
        alpha_t,
        dalpha_t,
        valid_tokens,
    ):
        loss_fn_name = getattr(self.config.training, "loss_fn", "elbo")
        if loss_fn_name == "elbo":
            per_token = self.diffusion.elbo_per_token(
                logits=logits,
                x0=x0,
                xt=xt,
                alpha_t=alpha_t,
                dalpha_t=dalpha_t,
            )
            nll_per_token = per_token
        elif loss_fn_name == "cross_entropy":
            per_token = ddm.cross_entropy_per_token(
                logits=logits,
                x0=x0,
                xt=xt,
                alpha_t=alpha_t,
                model_prediction_type=self._model_prediction_type(),
            )
            nll_per_token = self.diffusion.elbo_per_token(
                logits=logits,
                x0=x0,
                xt=xt,
                alpha_t=alpha_t,
                dalpha_t=dalpha_t,
            )
        else:
            raise ValueError(f"Unsupported training.loss_fn: {loss_fn_name}")

        valid_tokens = valid_tokens.to(per_token.dtype)
        loss_sum = (per_token * valid_tokens).sum()
        nlls = (nll_per_token * valid_tokens).sum()
        num_tokens = valid_tokens.sum().clamp_min(1)
        return LossInfo(loss=loss_sum / num_tokens, nlls=nlls, num_tokens=num_tokens)

    def _reset_throughput_window(self):
        self._throughput_tokens = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )
        self._throughput_start_time = time.perf_counter()

    def _log_training_throughput(self, num_tokens, batch_idx):
        if self._throughput_tokens is None:
            self._reset_throughput_window()
        self._throughput_tokens += num_tokens.detach().to(torch.float64)
        if (batch_idx + 1) % self.trainer.log_every_n_steps != 0:
            return

        elapsed = time.perf_counter() - self._throughput_start_time
        if elapsed > 0:
            self.log(
                "perf/tokens_per_sec",
                self._throughput_tokens / elapsed,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self._reset_throughput_window()

    def _log_training_losses(self, losses, *, prog_bar=False):
        self.log(
            "trainer/loss",
            losses.loss,
            on_step=True,
            on_epoch=False,
            prog_bar=prog_bar,
            sync_dist=True,
        )
        self.log(
            "train/nll",
            losses.nlls / losses.num_tokens,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _get_parameters(self):
        return self.diffusion.model.parameters()

    def _refresh_ema_from_parameters(self):
        if not self.ema:
            return
        self.ema.shadow_params = [
            p.clone().detach() for p in self._get_parameters() if p.requires_grad
        ]
        self.ema.collected_params = []
        if self.ema.num_updates is not None:
            self.ema.num_updates = 0

    def _warn_missing_ema_checkpoint(self, ckpt_path=None):
        location = f" in checkpoint: {ckpt_path}" if ckpt_path is not None else ""
        print(
            "EMA state not found"
            f"{location}; initializing EMA from the currently loaded parameters."
        )

    def _eval_mode(self):
        if self.ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        self.diffusion.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.diffusion.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler,
            optimizer=optimizer,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/nll",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if not self.ema:
            return
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._warn_missing_ema_checkpoint()
            self._refresh_ema_from_parameters()

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
