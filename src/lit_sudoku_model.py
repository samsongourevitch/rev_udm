from pathlib import Path

import torch

from models.dit import DIT
from ddm import (
    LinearSchedule,
    SigmaSchedule,
)
from lit_diffusion_base import LitDiffusionBase
from utils.sudoku_utils import build_unit_indices, count_violations_batch


class SudokuDITBackbone(torch.nn.Module):
    def __init__(self, config, vocab_size, schedule):
        super().__init__()
        self.config = config
        self.schedule = schedule
        self.time_conditioning = bool(config.algo.time_conditioning)
        self.backbone = DIT(config, vocab_size=vocab_size)

    def forward(self, x, t, conditioning_tokens=None, attention_mask=None):
        if attention_mask is not None and not attention_mask.bool().all():
            raise ValueError(
                "SudokuDITBackbone does not support padded attention masks. "
                "Set data.cutoff_len to the exact encoded Sudoku length."
            )
        alpha, _ = self.schedule(t)
        alpha = alpha.flatten()
        if self.time_conditioning:
            sigma = -torch.log(alpha)
        else:
            sigma = torch.zeros_like(alpha)
        if conditioning_tokens is None:
            return self.backbone(x, sigma)
        return self.backbone(x, sigma, conditioning_tokens=conditioning_tokens)


class LitSudokuDDM(LitDiffusionBase):
    """Lightning trainer for conditional Sudoku discrete diffusion."""

    def __init__(self, config, tokenizer, diffusion_model, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["tokenizer"])

        if int(config.model.length) != int(config.data.cutoff_len):
            raise ValueError(
                f"model.length={config.model.length} must match "
                f"data.cutoff_len={config.data.cutoff_len}."
            )

        self.schedule = LinearSchedule(config.training.sampling_eps)
        self.sigma_schedule = SigmaSchedule(**getattr(config.algo, "sigma_config", {}))
        if config.model.type == "ddit":
            self.backbone = SudokuDITBackbone(
                config=config,
                vocab_size=tokenizer.vocab_size,
                schedule=self.schedule,
            )
        else:
            raise ValueError(f"Unsupported Sudoku model.type: {config.model.type}")
        self.diffusion = diffusion_model(
            model=self.backbone,
            schedule_fn=self.schedule,
            sigma_schedule_fn=self.sigma_schedule,
            config=config,
            vocab_size=tokenizer.vocab_size,
            mask_index=tokenizer.mask_token_id,
            **kwargs,
        )

        self.ema = None
        self._last_train_sample_log_step = None

        digit_lookup = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
        for token, idx in tokenizer._vocab_str_to_int.items():
            if token.isdigit():
                digit_lookup[idx] = int(token)
        self.register_buffer("digit_lookup", digit_lookup, persistent=False)

        row_idx, col_idx = build_unit_indices()
        self.register_buffer("row_idx", row_idx, persistent=False)
        self.register_buffer("col_idx", col_idx, persistent=False)

    def _maskable_mask(self, batch):
        return (~batch["src_mask"].bool()) & batch["attention_mask"].bool()

    def _condition_on_source(self, xt, x0, maskable_mask):
        conditioned = torch.where(maskable_mask, xt, x0)
        for attr in ("noise", "pi"):
            if hasattr(xt, attr):
                setattr(conditioned, attr, getattr(xt, attr))
        return conditioned

    def _diffusion_forward_kwargs(self, attention_mask):
        return {"attention_mask": attention_mask}

    def _loss(self, batch, current_accumulation_step=None):
        x0 = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        maskable_mask = self._maskable_mask(batch)

        t = self._sample_t(
            x0.shape[0],
            current_accumulation_step=current_accumulation_step,
        )
        alpha_t, dalpha_t = self.schedule(t)
        xt = self.diffusion.sample_forward(x0, t)
        xt = self._condition_on_source(xt, x0, maskable_mask)
        logits = self.diffusion(
            xt=xt,
            t=t,
            **self._diffusion_forward_kwargs(attention_mask),
        )
        return self._compute_loss_info(
            logits=logits,
            x0=x0,
            xt=xt,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            valid_tokens=maskable_mask,
        )

    def training_step(self, batch, batch_idx):
        current_accumulation_step = self._current_accumulation_step(batch_idx)
        losses = self._loss(
            batch,
            current_accumulation_step=current_accumulation_step,
        )
        self._log_training_throughput(losses.num_tokens, batch_idx)
        self._log_training_losses(losses, prog_bar=True)
        self._log_train_generated_samples(batch)
        return losses.loss

    def validation_step(self, batch, batch_idx):
        losses = self._loss(batch)
        batch_size = batch["input_ids"].shape[0]
        self.log(
            "val/nll",
            losses.nlls / losses.num_tokens,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/loss",
            losses.loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.config.eval.generate_samples:
            if self.config.eval.generate_greedy:
                greedy = self.generate(batch, mode="greedy")
                self._log_sample_metrics(batch, greedy, "greedy")
                self._log_generated_samples(batch, greedy, "greedy", "val", batch_idx)
            if self.config.eval.generate_random:
                random = self.generate(batch, mode="random")
                self._log_sample_metrics(batch, random, "random")
                self._log_generated_samples(batch, random, "random", "val", batch_idx)
        return losses.loss

    def on_train_epoch_start(self):
        self._reset_throughput_window()

    def _greedy_ancestral_step(self, xt, x0, alpha_t, alpha_s):
        if self.config.algo.name not in {"mdlm", "remdlm"}:
            raise ValueError(
                "Greedy Sudoku sampling is implemented only for mdlm/remdlm. "
                "Set eval.generate_greedy=false for this algorithm."
            )
        x0_token = x0.argmax(dim=-1, keepdim=True)
        greedy_x0 = torch.zeros_like(x0)
        greedy_x0.scatter_(-1, x0_token, 1.0)
        unnormalized_probs = (alpha_s - alpha_t) * greedy_x0
        mask_prob = (1 - alpha_s).to(unnormalized_probs.dtype)
        unnormalized_probs.scatter_add_(
            dim=-1,
            index=xt.unsqueeze(-1),
            src=mask_prob.view(-1, 1, 1).expand(xt.size(0), xt.size(1), 1),
        )
        return unnormalized_probs.argmax(dim=-1)

    @torch.no_grad()
    def generate(self, batch, mode):
        if mode not in {"greedy", "random"}:
            raise ValueError(f"Unsupported generation mode: {mode}")

        x0 = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        maskable_mask = self._maskable_mask(batch)
        xt = self.diffusion.sample_prior(n_samples=x0.shape[0])
        xt = self._condition_on_source(xt, x0, maskable_mask)

        nfes = int(self.config.sampling.nfes)
        timesteps = torch.linspace(1, 0, nfes + 1, device=self.device)[:-1]
        if timesteps.numel() == 1:
            dt = timesteps[0]
        else:
            dt = timesteps[0] - timesteps[1]

        for scalar_t in timesteps:
            scalar_s = (scalar_t - dt).clamp_min(0)
            t = scalar_t.expand(x0.shape[0], 1)
            s = scalar_s.expand(x0.shape[0], 1)
            alpha_t, _ = self.schedule(t)
            alpha_s, _ = self.schedule(s)
            alpha_t = torch.atleast_3d(alpha_t)
            alpha_s = torch.atleast_3d(alpha_s)
            probs = (
                self.diffusion(
                    xt=xt,
                    t=t,
                    **self._diffusion_forward_kwargs(attention_mask),
                    apply_temp_output=self.config.sampling.apply_temp_output,
                )
                .to(torch.float64)
                .exp()
            )
            probs = self.diffusion._truncate_prediction(probs)

            if mode == "greedy":
                xs = self._greedy_ancestral_step(xt, probs, alpha_t, alpha_s)
            else:
                xs = self.diffusion._ancestral_step(
                    xt=xt,
                    x0=probs,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    t=scalar_t,
                    s=scalar_s,
                )
            xt = self._condition_on_source(xs, x0, maskable_mask)
        return xt

    def _ids_to_grid(self, ids, src_mask):
        sudoku_tokens = self.digit_lookup[ids][~src_mask.bool()]
        sudoku_grid = sudoku_tokens.reshape(ids.shape[0], -1)[..., :-1]
        if sudoku_grid.shape[1] != 81:
            raise ValueError(
                f"Expected 81 decoded Sudoku cells, got {sudoku_grid.shape[1]}."
            )
        min_digit = sudoku_grid.min(dim=-1)[0].reshape(-1, 1)
        sudoku_grid = sudoku_grid.clamp(min=min_digit)
        one_hot = torch.nn.functional.one_hot(sudoku_grid.long(), num_classes=10)
        return one_hot[..., 1:].reshape(-1, 9, 9, 9)

    def _ids_to_cell_grid(self, ids, blank_zero=False):
        if ids.numel() != 81:
            raise ValueError(f"Expected 81 Sudoku cells, got {ids.numel()}.")
        cells = []
        invalid_count = 0
        for token_id in ids.detach().cpu().tolist():
            token = self.tokenizer._vocab_int_to_str.get(
                int(token_id),
                self.tokenizer.unk_token,
            )
            if token.isdigit():
                cells.append("." if blank_zero and token == "0" else token)
            else:
                cells.append("?")
                invalid_count += 1
        grid = "\n".join("".join(cells[start:start + 9]) for start in range(0, 81, 9))
        return grid, invalid_count

    def _should_log_generated_samples(self, split, batch_idx=None):
        if not self.trainer.is_global_zero:
            return False
        if split == "val":
            return (
                self.config.eval.get("log_samples", False)
                and batch_idx == 0
                and not self.trainer.sanity_checking
            )
        if split == "train":
            if not self.config.eval.get("log_train_samples", False):
                return False
            interval = int(self.config.eval.train_sample_log_interval)
            if interval <= 0:
                raise ValueError("eval.train_sample_log_interval must be positive.")
            step = int(self.global_step)
            if step % interval != 0:
                return False
            if self._last_train_sample_log_step == step:
                return False
            self._last_train_sample_log_step = step
            return True
        raise ValueError(f"Unsupported sample logging split: {split}")

    @torch.no_grad()
    def _generate_for_logging(self, batch, mode):
        was_training = self.diffusion.training
        self.diffusion.eval()
        try:
            return self.generate(batch, mode=mode)
        finally:
            if was_training:
                self.diffusion.train()

    def _log_train_generated_samples(self, batch):
        if not self.config.eval.generate_samples:
            return
        if not self._should_log_generated_samples("train"):
            return
        if self.config.eval.generate_greedy:
            greedy = self._generate_for_logging(batch, mode="greedy")
            self._log_generated_samples(batch, greedy, "greedy", "train", check=False)
        if self.config.eval.generate_random:
            random = self._generate_for_logging(batch, mode="random")
            self._log_generated_samples(batch, random, "random", "train", check=False)

    def _log_generated_samples(
        self,
        batch,
        preds,
        suffix,
        split,
        batch_idx=None,
        check=True,
    ):
        if check and not self._should_log_generated_samples(split, batch_idx):
            return

        source_mask = (
            batch["src_mask"].bool()
            & batch["attention_mask"].bool()
            & batch["input_ids"].ne(self.tokenizer.sep_token_id)
        )
        target_mask = (
            ~batch["src_mask"].bool()
            & batch["attention_mask"].bool()
            & batch["input_ids"].ne(self.tokenizer.eos_token_id)
        )
        n_samples = min(int(self.config.eval.num_sample_log), batch["input_ids"].shape[0])
        rows = []
        for sample_idx in range(n_samples):
            target_ids = batch["input_ids"][sample_idx][target_mask[sample_idx]]
            pred_ids = preds[sample_idx][target_mask[sample_idx]]
            quiz, _ = self._ids_to_cell_grid(
                batch["input_ids"][sample_idx][source_mask[sample_idx]],
                blank_zero=True,
            )
            target, _ = self._ids_to_cell_grid(target_ids)
            model_prediction, invalid_count = self._ids_to_cell_grid(pred_ids)
            cell_errors = int(pred_ids.ne(target_ids).sum().item())
            rows.append(
                [
                    suffix,
                    int(self.global_step),
                    sample_idx,
                    quiz,
                    target,
                    model_prediction,
                    cell_errors,
                    invalid_count,
                ]
            )

        columns = [
            "sampler",
            "global_step",
            "sample_idx",
            "quiz",
            "target",
            "model_prediction",
            "cell_errors",
            "invalid_model_prediction_cells",
        ]
        self._write_generated_samples(rows, columns, suffix, split)
        if self.logger is not None and hasattr(self.logger, "log_table"):
            self.logger.log_table(
                key=f"{split}/samples_{suffix}",
                columns=columns,
                data=rows,
                step=self.global_step,
            )

    def _write_generated_samples(self, rows, columns, suffix, split):
        sample_dir = Path(self.trainer.default_root_dir) / "generated_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"{split}_samples_step_{int(self.global_step):08d}_{suffix}.txt"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\t".join(columns) + "\n")
            for row in rows:
                handle.write("\t".join(str(value).replace("\n", "\\n") for value in row))
                handle.write("\n\n")
                handle.write(f"sample_idx: {row[2]}\n")
                handle.write("quiz:\n")
                handle.write(f"{row[3]}\n")
                handle.write("target:\n")
                handle.write(f"{row[4]}\n")
                handle.write("model_prediction:\n")
                handle.write(f"{row[5]}\n")
                handle.write(f"cell_errors: {row[6]}\n")
                handle.write(f"invalid_model_prediction_cells: {row[7]}\n")
                handle.write("\n")

    def _log_sample_metrics(self, batch, preds, suffix):
        maskable_mask = self._maskable_mask(batch)
        labels = batch["input_ids"]
        correct = ((preds == labels) | ~maskable_mask).all(dim=1).float().sum()
        total = torch.tensor(float(labels.shape[0]), device=self.device)
        grid = self._ids_to_grid(preds, batch["src_mask"])
        per_sample_violations = count_violations_batch(grid, self.row_idx, self.col_idx)
        violations = per_sample_violations.sum()
        solved = (per_sample_violations == 0).float().sum()

        self.log(
            f"val/acc_{suffix}",
            correct / total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        self.log(
            f"val/solve_rate_{suffix}",
            solved / total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        self.log(
            f"val/avg_violations_{suffix}",
            violations / total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )

    def _eval_mode(self):
        self.diffusion.eval()

    def _train_mode(self):
        self.diffusion.train()
