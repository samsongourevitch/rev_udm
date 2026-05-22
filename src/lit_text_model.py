import torch
from tqdm import tqdm

import hydra

import metrics as metrics
from models.dit import DIT
from models.ema import ExponentialMovingAverage
from ddm import (
    LinearSchedule,
    SigmaSchedule,
)
from lit_diffusion_base import LitDiffusionBase


class Backbone(torch.nn.Module):
    def __init__(self, backbone, time_conditioning, schedule):
        super().__init__()
        self.backbone = backbone
        self.time_conditioning = time_conditioning
        self.schedule = schedule

    def forward(self, x, t, conditioning_tokens=None):
        alpha, _ = self.schedule(t)
        alpha = alpha.flatten()
        if self.time_conditioning:
            sigma = -torch.log(alpha)
        else:
            sigma = torch.zeros_like(alpha)
        if conditioning_tokens is None:
            return self.backbone(x, sigma)
        return self.backbone(x, sigma, conditioning_tokens=conditioning_tokens)


class LitTextDDM(LitDiffusionBase):
    def __init__(
        self,
        config,
        diffusion_model,
        tokenizer,
        sampling_ancestral_step=None,
        sampling_ancestral_step_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["tokenizer"])

        if getattr(tokenizer, "mask_token_id", None) is None:
            tokenizer.mask_index = tokenizer.vocab_size
            vocab_size = tokenizer.vocab_size + 1
        else:
            tokenizer.mask_index = tokenizer.mask_token_id
            vocab_size = tokenizer.vocab_size

        dit = DIT(config, vocab_size=vocab_size)
        self.schedule = LinearSchedule(config.training.sampling_eps)
        sigma_config = getattr(config.algo, "sigma_config", {})
        self.sigma_schedule = SigmaSchedule(**sigma_config)
        wrapped_backbone = Backbone(
            backbone=dit,
            time_conditioning=config.algo.time_conditioning,
            schedule=self.schedule,
        )

        self.diffusion = diffusion_model(
            model=wrapped_backbone,
            schedule_fn=self.schedule,
            sigma_schedule_fn=self.sigma_schedule,
            config=config,
            vocab_size=tokenizer.vocab_size,
            mask_index=tokenizer.mask_index,
            **kwargs,
        )
        self.ignore_bos = getattr(config.algo, "ignore_bos", False)

        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema
            )
        else:
            self.ema = None
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._sampling_generator = None
        self._sampling_generator_seed = None
        self._sampling_generator_device = None
        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
        )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def _loss(self, input_ids, attention_mask, current_accumulation_step=None):
        t = self._sample_t(
            input_ids.shape[0],
            current_accumulation_step=current_accumulation_step,
        )
        alpha_t, dalpha_t = self.schedule(t)
        xt = self.diffusion.sample_forward(input_ids, t)
        logits = self.diffusion(xt=xt, t=t)
        valid_tokens = attention_mask
        if self.ignore_bos and valid_tokens.shape[1] > 0:
            valid_tokens[:, 0] = 0

        return self._compute_loss_info(
            logits=logits,
            x0=input_ids,
            xt=xt,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            valid_tokens=valid_tokens,
        )

    def training_step(self, batch, batch_idx):
        current_accumulation_step = self._current_accumulation_step(batch_idx)
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            current_accumulation_step=current_accumulation_step,
        )
        self.metrics.update_train(losses.nlls, 0.0, losses.num_tokens)
        self._log_training_throughput(losses.num_tokens, batch_idx)
        self._log_training_losses(losses)
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch["input_ids"], batch["attention_mask"])
        self.metrics.update_valid(losses.nlls, 0.0, losses.num_tokens)
        return losses.loss

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()

    def on_validation_epoch_end(self):
        for key, metric in self.metrics.valid_nlls.items():
            self.log(
                name=key,
                value=metric.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if (
            self.config.eval.compute_perplexity_on_sanity
            or not self.trainer.sanity_checking
        ) and self.config.eval.generate_samples:
            for _ in tqdm(range(self.config.sampling.num_sample_batches)):
                samples = self.generate_samples(
                    num_samples=self.config.loader.eval_batch_size,
                    nfes=self.config.sampling.nfes,
                )
                self.metrics.record_entropy(samples)
                text_samples = self.tokenizer.batch_decode(samples)
                if self.config.eval.compute_generative_perplexity:
                    self.metrics.record_generative_perplexity(
                        text_samples,
                        self.config.model.length,
                        device=self.device,
                    )
            if self.config.eval.compute_generative_perplexity:
                self.log(
                    "val/gen_ppl",
                    self.metrics.gen_ppl.compute(),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )
                self.log(
                    "val/sample_entropy",
                    self.metrics.sample_entropy.compute(),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )
        self._train_mode()

    def on_train_epoch_start(self):
        self.metrics.reset()
        self._reset_throughput_window()

    def _get_sampling_generator(self):
        generator_seed = self.config.sampling.get("generator_seed", None)
        if generator_seed is None:
            return None

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        if (
            self._sampling_generator is None
            or self._sampling_generator_seed != generator_seed
            or self._sampling_generator_device != generator_device
        ):
            self._sampling_generator = torch.Generator(device=generator_device)
            self._sampling_generator.manual_seed(int(generator_seed))
            self._sampling_generator_seed = generator_seed
            self._sampling_generator_device = generator_device

        return self._sampling_generator

    def generate_samples(
        self,
        num_samples,
        nfes=None,
        temperature=1.0,
        ancestral_step=None,
        ancestral_step_kwargs=None,
    ):
        if nfes is None:
            nfes = self.config.sampling.nfes

        if not ancestral_step_kwargs:
            ancestral_step_kwargs = {}
        return self.diffusion.sample(
            nfes=nfes,
            n_samples=num_samples,
            temperature=temperature,
            apply_temp_output=self.config.sampling.apply_temp_output,
            generator=self._get_sampling_generator(),
            ancestral_step=ancestral_step,
            **ancestral_step_kwargs,
        )

    def restore_model_and_sample(self, nfes, temperature):
        self._eval_mode()
        ancestral_step = None
        ancestral_step_kwargs = {}

        if self.config.sampling.corrector:
            ancestral_step = hydra.utils.get_method(
                str(self.config.sampling.corrector.ancestral_step)
            )
            ancestral_step_kwargs = {
                key: value
                for key, value in self.config.sampling.corrector.items()
                if key != "ancestral_step"
            }

        samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size,
            nfes=nfes,
            temperature=temperature,
            ancestral_step=ancestral_step,
            ancestral_step_kwargs=ancestral_step_kwargs,
        )
        self._train_mode()
        return samples

    def on_train_start(self):
        super().on_train_start()

        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        sampler_cls = (
            dataloader.FaultTolerantDistributedSampler
            if distributed
            else dataloader.RandomFaultTolerantSampler
        )

        if not hasattr(self.trainer.fit_loop, "_combined_loader"):
            return

        train_num_workers = self.config.loader.get(
            "train_num_workers", self.config.loader.num_workers
        )
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            persistent_workers = train_num_workers > 0
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)

            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (
                            self.fast_forward_batches * self.config.loader.batch_size
                        ),
                    }
                )

            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=train_num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=persistent_workers,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def load_backbone_from_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "backbone_state_dict" in checkpoint:
            state_dict = checkpoint["backbone_state_dict"]
            self.diffusion.model.backbone.load_state_dict(state_dict, strict=True)
            if self.ema:
                if "ema" in checkpoint:
                    self.ema.load_state_dict(checkpoint["ema"])
                else:
                    self._warn_missing_ema_checkpoint(ckpt_path)
                    self._refresh_ema_from_parameters()
            return

        state_dict = checkpoint.get("state_dict", checkpoint)
        prefixes = (
            "backbone.",
            "module.backbone.",
            "diffusion.model.backbone.",
            "module.diffusion.model.backbone.",
        )
        extracted = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    extracted[key[len(prefix) :]] = value
                    break

        if extracted:
            current_backbone_keys = set(
                self.diffusion.model.backbone.state_dict().keys()
            )
            matched_keys = current_backbone_keys.intersection(extracted.keys())
            if not matched_keys:
                raise RuntimeError(
                    f"No backbone parameters matched when loading checkpoint: {ckpt_path}"
                )
            self.diffusion.model.backbone.load_state_dict(extracted, strict=True)
        else:
            current_keys = set(self.state_dict().keys())
            matched_keys = current_keys.intersection(state_dict.keys())
            if not matched_keys:
                raise RuntimeError(
                    f"No model parameters matched when loading checkpoint: {ckpt_path}"
                )
            self.load_state_dict(state_dict, strict=False)

        if self.ema:
            if "ema" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema"])
            else:
                self._warn_missing_ema_checkpoint(ckpt_path)
                self._refresh_ema_from_parameters()
