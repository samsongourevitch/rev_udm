from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import sample_categorical

NEG_INFINITY = -1_000_000.0


def _carry_over_unmasking(
    xt,
    logits,
    mask_index,
    other_value=NEG_INFINITY,
    selected_value=0.0,
):
    unmasked_indices = xt != mask_index
    frozen_logits = torch.full_like(logits, other_value)
    frozen_logits.scatter_(-1, xt.unsqueeze(-1), selected_value)
    return torch.where(unmasked_indices.unsqueeze(-1), frozen_logits, logits)


def to_loo_denoiser_logits(logits, xt, alpha_t, model_prediction_type):
    logits = logits.to(torch.float64)
    if model_prediction_type == "mean_loo":
        out = logits
    elif model_prediction_type == "mean":
        alpha_t = torch.atleast_3d(alpha_t).to(torch.float64)
        K = logits.shape[-1]
        delta = -torch.log1p(K * alpha_t / (1 - alpha_t))

        out = logits.clone()
        out.scatter_add_(-1, xt[..., None], delta.expand_as(xt[..., None]))
    else:
        raise ValueError(
            f"Unsupported model_prediction_type: {model_prediction_type}"
        )

    return out


def to_denoiser_logits(logits, xt, alpha_t, model_prediction_type):
    logits = logits.to(torch.float64)
    if model_prediction_type == "mean":
        out = logits
    elif model_prediction_type == "mean_loo":
        alpha_t = torch.atleast_3d(alpha_t).to(torch.float64)
        K = logits.shape[-1]
        delta = torch.log1p(K * alpha_t / (1 - alpha_t))

        out = logits.clone()
        out.scatter_add_(-1, xt[..., None], delta.expand_as(xt[..., None]))
    else:
        raise ValueError(
            f"Unsupported model_prediction_type: {model_prediction_type}"
        )

    return out


def cross_entropy_per_token(
    logits,
    x0,
    xt,
    alpha_t,
    model_prediction_type,
    **kwargs,
):
    if model_prediction_type == "mean_loo":
        logits = to_denoiser_logits(
            logits=logits,
            xt=xt,
            alpha_t=alpha_t,
            model_prediction_type="mean_loo",
        )
        logits = logits.log_softmax(dim=-1)
    return -torch.gather(input=logits, dim=-1, index=x0[:, :, None]).squeeze(-1)


def _sample_udlm_bridge_from_probs(
    xt,
    x0,
    alpha_t,
    alpha_s,
    vocab_size,
    generator=None,
):
    K = vocab_size
    x0 = x0.to(torch.float64)
    alpha_t = torch.as_tensor(alpha_t, device=x0.device, dtype=x0.dtype)
    alpha_s = torch.as_tensor(alpha_s, device=x0.device, dtype=x0.dtype)
    ratio = alpha_t / alpha_s
    diff_alpha = alpha_s - alpha_t
    x_xt = torch.gather(x0, -1, xt[..., None])
    num = diff_alpha * x0 + (1 - ratio) * (1 - alpha_s) / K
    num.scatter_add_(-1, xt[..., None], K * alpha_t * x_xt + (ratio - alpha_t))
    return sample_categorical(num, generator=generator)


class LinearSchedule(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps)
        return alpha_t, dalpha_t


class SigmaSchedule(torch.nn.Module):
    """
    ReMDM sigma schedules.

    Base schedules:
      - "zero" / "mdlm": sigma_t = 0
      - "constant": sigma_t = min(constant, sigma_max_t)
      - "cap": sigma_t = min(eta_cap, sigma_max_t)
      - "rescale": sigma_t = eta_rescale * sigma_max_t
      - "loop": sigma_t = loop_eta on the plateau t_off < t <= t_on, else 0

    Optional modifiers:
      - switch: sigma_t = 0 for t > t_switch
    """

    def __init__(
        self,
        schedule_type="constant",
        constant=0.0,
        eta_cap=1.0,
        eta_rescale=1.0,
        t_switch=None,
        t_on=None,
        t_off=None,
        alpha_on=None,
        loop_eta=None,
        eps=1e-3,
    ):
        super().__init__()
        self.schedule_type = str(schedule_type)
        self.constant = float(constant)
        self.eta_cap = float(eta_cap)
        self.eta_rescale = float(eta_rescale)
        self.t_switch = None if t_switch is None else float(t_switch)
        self.t_on = None if t_on is None else float(t_on)
        self.t_off = None if t_off is None else float(t_off)
        self.alpha_on = None if alpha_on is None else float(alpha_on)
        self.loop_eta = None if loop_eta is None else float(loop_eta)
        self.eps = float(eps)

    def sigma_max(self, alpha_t, alpha_s):
        ones = torch.ones_like(alpha_t)
        return torch.minimum(ones, (1.0 - alpha_s) / alpha_t.clamp_min(1e-12))

    def has_loop_schedule(self):
        return self.schedule_type == "loop"

    def _validate_loop_config(self):
        if not self.has_loop_schedule():
            return
        missing = [
            name
            for name, value in (
                ("t_on", self.t_on),
                ("t_off", self.t_off),
                ("alpha_on", self.alpha_on),
                ("loop_eta", self.loop_eta),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Loop schedule requires " + ", ".join(missing) + " to be set."
            )
        if not (0.0 < self.t_off < self.t_on <= 1.0):
            raise ValueError(
                f"Loop schedule expects 0 < t_off < t_on <= 1, got "
                f"t_off={self.t_off}, t_on={self.t_on}."
            )
        if not (0.0 < self.alpha_on < 1.0):
            raise ValueError(
                f"Loop schedule expects 0 < alpha_on < 1, got {self.alpha_on}."
            )
        if self.loop_eta < 0.0:
            raise ValueError(
                f"Loop schedule expects loop_eta >= 0, got {self.loop_eta}."
            )

    def loop_alphas(self, t, s):
        self._validate_loop_config()
        t = torch.as_tensor(t)
        s = torch.as_tensor(s, device=t.device, dtype=t.dtype)
        alpha_on = torch.as_tensor(self.alpha_on, device=t.device, dtype=t.dtype)
        t_on = torch.as_tensor(self.t_on, device=t.device, dtype=t.dtype)
        t_off = torch.as_tensor(self.t_off, device=t.device, dtype=t.dtype)

        alpha_t = torch.full_like(t, alpha_on)
        alpha_s = torch.full_like(s, alpha_on)

        high_region = t > t_on
        low_region = t <= t_off

        alpha_t_high = (1.0 - t) * alpha_on / (1.0 - t_on)
        alpha_s_high = (1.0 - s) * alpha_on / (1.0 - t_on)
        alpha_t_low = 1.0 - t * (1.0 - alpha_on) / t_off
        alpha_s_low = 1.0 - s * (1.0 - alpha_on) / t_off

        alpha_t = torch.where(high_region, alpha_t_high, alpha_t)
        alpha_s = torch.where(high_region, alpha_s_high, alpha_s)
        alpha_t = torch.where(low_region, alpha_t_low, alpha_t)
        alpha_s = torch.where(low_region, alpha_s_low, alpha_s)
        return alpha_t, alpha_s

    def _base_sigma(self, t, alpha_t, alpha_s):
        sigma_max = self.sigma_max(alpha_t, alpha_s)

        if self.schedule_type in {"zero", "mdlm"}:
            sigma = torch.zeros_like(sigma_max)

        elif self.schedule_type in {"constant", "remdm"}:
            sigma = torch.full_like(sigma_max, self.constant)

        elif self.schedule_type == "cap":
            sigma = torch.full_like(sigma_max, self.eta_cap)

        elif self.schedule_type == "rescale":
            sigma = self.eta_rescale * sigma_max

        elif self.schedule_type == "loop":
            self._validate_loop_config()
            sigma = torch.zeros_like(sigma_max)
            in_plateau = (t > self.t_off) & (t <= self.t_on)
            sigma = torch.where(
                in_plateau,
                torch.full_like(sigma_max, self.loop_eta),
                sigma,
            )

        else:
            raise ValueError(f"Unsupported sigma schedule type: {self.schedule_type}")

        return torch.minimum(sigma, sigma_max)

    def forward(self, t, alpha_t, alpha_s):
        sigma = self._base_sigma(t=t, alpha_t=alpha_t, alpha_s=alpha_s)

        # ReMDM-switch: only turn on remasking when t <= t_switch
        if self.t_switch is not None:
            sigma = torch.where(
                t <= self.t_switch,
                sigma,
                torch.zeros_like(sigma),
            )

        return torch.minimum(sigma, self.sigma_max(alpha_t, alpha_s))


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        model,
        schedule_fn,
        config,
        vocab_size,
        model_prediction="mean",
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.schedule = schedule_fn
        self.vocab_size = vocab_size
        self.model_prediction = model_prediction
        # Representation returned by forward(); subclasses override this when
        # _process_logits converts the raw model output to another parameterization.
        self.processed_prediction_type = model_prediction
        self.p_nucleus = float(getattr(config.sampling, "p_nucleus", 1.0))
        if not (0.0 <= self.p_nucleus <= 1.0):
            raise ValueError(
                f"sampling.p_nucleus must be in [0, 1], got {self.p_nucleus}"
            )

    def _get_logits(self, xt, t, **kwargs):
        conditioning_tokens = kwargs.get("conditioning_tokens", None)
        model_kwargs = {
            key: kwargs[key]
            for key in ("attention_mask", "labels", "drop_label")
            if key in kwargs
        }
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.model(
                x=xt,
                t=t,
                conditioning_tokens=conditioning_tokens,
                **model_kwargs,
            )
        return logits

    def _process_logits(self, logits, xt, alpha_t, **kwargs):
        return NotImplementedError

    def forward(self, xt, t, temperature=1.0, apply_temp_output=True, **kwargs):
        """
        returns log-normalized logits
        """
        logits = self._get_logits(xt, t, **kwargs)
        orig_dtype = logits.dtype

        alpha_t, _ = self.schedule(t)

        if apply_temp_output:
            logits = logits / temperature

        processed_logits = self._process_logits(
            logits=logits,
            xt=xt,
            alpha_t=alpha_t,
            **kwargs,
        )

        if not apply_temp_output:
            processed_logits = processed_logits / temperature

        return processed_logits.log_softmax(dim=-1).to(orig_dtype)

    def sample_forward(self, x, t, generator=None):
        return NotImplementedError

    def sample_prior(self, n_samples=1, generator=None):
        return NotImplementedError

    def _ancestral_step(self, xt, x0, alpha_t, alpha_s, generator=None, **kwargs):
        return NotImplementedError

    def _apply_p_nucleus(self, categorical_probs):
        if not (0.0 < self.p_nucleus < 1.0):
            return categorical_probs

        if (categorical_probs < 0).any():
            raise ValueError(
                "p-nucleus filtering expects nonnegative categorical weights."
            )

        total_mass = categorical_probs.sum(dim=-1)
        if (total_mass <= 0).any():
            raise ValueError(
                "p-nucleus filtering received a categorical distribution with zero total mass."
            )

        sorted_probs, sorted_indices = torch.sort(
            categorical_probs, descending=True, dim=-1
        )
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cum_probs > self.p_nucleus
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        return categorical_probs.masked_fill(indices_to_remove, 0.0)

    def _sample_categorical_with_p_nucleus(self, categorical_probs, generator=None):
        categorical_probs = self._apply_p_nucleus(categorical_probs)
        return sample_categorical(categorical_probs, generator=generator)

    def _truncate_prediction(self, categorical_probs):
        if not (0.0 < self.p_nucleus < 1.0):
            return categorical_probs

        truncated_probs = self._apply_p_nucleus(categorical_probs)
        tiny = torch.finfo(truncated_probs.dtype).tiny
        return truncated_probs / truncated_probs.sum(dim=-1, keepdim=True).clamp_min(
            tiny
        )

    @torch.no_grad()
    def sample(
        self,
        nfes,
        n_samples=1,
        temperature=1.0,
        ancestral_step=None,
        apply_temp_output=True,
        generator=None,
        **kwargs,
    ):
        """
        Implements (cached) ancestral sampler
        """
        device = next(self.model.parameters()).device
        corrector_steps = kwargs.get("corrector_steps", 0)
        if not corrector_steps:
            corrected_steps = 0
            n_steps = nfes
        else:
            corrected_steps = (nfes - 1) // (1 + corrector_steps)
            n_steps = nfes - corrected_steps * corrector_steps

        if n_steps < 1:
            raise ValueError(
                f"nfes must induce at least one ancestral step, got nfes={nfes}."
            )

        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)[:-1]
        # Handle the degenerate 1-step schedule (t=1 -> s=0).
        if timesteps.numel() == 1:
            dt = timesteps[0]
        else:
            dt = timesteps[0] - timesteps[1]

        time_conditioning = getattr(self.model, "time_conditioning", True)
        update_fn = ancestral_step or self._ancestral_step

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        xt = self.sample_prior(n_samples=n_samples, generator=generator)
        prev_xt = None
        for i, t in enumerate(tqdm(timesteps)):
            alpha_t, _ = self.schedule(t)
            alpha_s, _ = self.schedule(t - dt)
            alpha_t, alpha_s = torch.atleast_3d(alpha_t), torch.atleast_3d(alpha_s)

            if prev_xt is None or time_conditioning or not torch.equal(xt, prev_xt):
                logits = self.forward(
                    xt=xt,
                    t=t,
                    temperature=temperature,
                    apply_temp_output=apply_temp_output,
                    **kwargs,
                ).to(torch.float64)
                probs = self._truncate_prediction(logits.exp())

            prev_xt = xt
            step_kwargs = dict(kwargs)
            step_kwargs["corrector_steps"] = (
                corrector_steps if i < corrected_steps else 0
            )
            xs = update_fn(
                xt=xt,
                x0=probs,
                t=t,
                s=t - dt,
                alpha_t=alpha_t,
                alpha_s=alpha_s,
                diffusion=self,
                generator=generator,
                **step_kwargs,
            )
            xt = xs

        return xt

    def elbo_per_token(self, logits, x0, alpha_t, dalpha_t, **kwargs):
        return NotImplementedError

    def loss_per_token(self, logits, x0, xt, alpha_t, dalpha_t, loss_fn=None, **kwargs):
        fn = loss_fn or self.elbo_per_token
        return fn(
            logits=logits,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            xt=xt,
            **kwargs,
        )


class MDLM(DiscreteDiffusion):
    def __init__(self, model, schedule_fn, config, vocab_size, mask_index, **kwargs):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
            model_prediction="mean",
        )
        self.mask_index = mask_index

    def _process_logits(self, logits, xt, alpha_t, **kwargs):
        del alpha_t, kwargs
        logits[:, :, self.mask_index] += NEG_INFINITY
        return _carry_over_unmasking(
            xt=xt,
            logits=logits,
            mask_index=self.mask_index,
        )

    def sample_forward(self, x, t, generator=None):
        del generator
        alpha_t, _ = self.schedule(t)
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def sample_prior(self, n_samples=1, generator=None):
        del generator
        device = next(self.model.parameters()).device
        length = self.config.model.length
        return torch.full(
            (n_samples, length),
            fill_value=self.mask_index,
            dtype=torch.long,
            device=device,
        )

    def _ancestral_step(self, xt, x0, alpha_t, alpha_s, generator=None, **kwargs):
        unnormalized_probs = (alpha_s - alpha_t) * x0
        mask_prob = (1 - alpha_s).to(unnormalized_probs.dtype)
        unnormalized_probs.scatter_add_(
            dim=-1,
            index=xt.unsqueeze(-1),
            src=mask_prob.view(-1, 1, 1).expand(xt.size(0), xt.size(1), 1),
        )
        return sample_categorical(unnormalized_probs, generator=generator)

    def elbo_per_token(
        self, logits, x0, xt=None, alpha_t=None, dalpha_t=None, **kwargs
    ):
        weight = dalpha_t / (1 - alpha_t)
        log_p_theta = torch.gather(input=logits, dim=-1, index=x0[:, :, None]).squeeze(
            -1
        )
        return weight * log_p_theta


class ReMDLM(MDLM):
    def __init__(
        self,
        model,
        schedule_fn,
        sigma_schedule_fn,
        config,
        vocab_size,
        mask_index,
        **kwargs,
    ):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
            mask_index=mask_index,
            **kwargs,
        )
        self.sigma_schedule = sigma_schedule_fn

    def _ancestral_step(self, xt, x0, alpha_t, alpha_s, t, s, generator=None, **kwargs):
        """
        xt: current tokens at time t, shape [B, L]
        x0: model categorical probabilities/logits after exp, shape [B, L, V]
        """

        del kwargs
        sampled_x0 = sample_categorical(x0, generator=generator)

        if self.sigma_schedule.has_loop_schedule():
            alpha_t, alpha_s = self.sigma_schedule.loop_alphas(t=t, s=s)
        else:
            alpha_t, _ = self.schedule(t)
            alpha_s, _ = self.schedule(s)

        alpha_t = alpha_t.squeeze().to(dtype=x0.dtype)
        alpha_s = alpha_s.squeeze().to(dtype=x0.dtype)

        sigma_t = (
            self.sigma_schedule(
                t=t,
                alpha_t=alpha_t,
                alpha_s=alpha_s,
            )
            .squeeze()
            .to(dtype=x0.dtype)
        )

        prob_mask_if_masked = (1.0 - alpha_s - sigma_t * alpha_t) / (1.0 - alpha_t)

        is_masked_now = xt.eq(self.mask_index)
        u = torch.rand(xt.shape, device=xt.device, dtype=x0.dtype, generator=generator)

        # If xt is masked: sample either x0 or mask.
        xs_if_masked = torch.where(
            u < prob_mask_if_masked,
            torch.full_like(xt, self.mask_index),
            sampled_x0,
        )

        # If xt is unmasked: under carry-over parameterization, keep xt with prob 1-sigma_t, remask with sigma_t.
        xs_if_unmasked = torch.where(
            u < sigma_t,
            torch.full_like(xt, self.mask_index),
            xt,
        )

        xs = torch.where(is_masked_now, xs_if_masked, xs_if_unmasked)
        return xs


class AUDM(DiscreteDiffusion):
    def __init__(self, model, schedule_fn, config, vocab_size, **kwargs):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
        )
        self.model_prediction = "mean"

    def _get_logits(self, xt, t, **kwargs):
        noise = getattr(xt, "noise", None)
        if noise is None:
            raise ValueError("Missing conditioning noise on `xt.noise`.")

        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.model(
                x=xt,
                t=t,
                conditioning_tokens=noise,
                **kwargs,
            )
        return logits

    def _process_logits(
        self,
        logits,
        xt,
        alpha_t,
        **kwargs,
    ):
        del alpha_t, kwargs
        noise = xt.noise
        return _carry_over_unmasking(
            xt=xt,
            logits=logits,
            mask_index=noise,
        )

    def sample_forward(self, x, t, generator=None):
        alpha_t, _ = self.schedule(t)
        move_indices = (
            torch.rand(*x.shape, device=x.device, generator=generator) < 1 - alpha_t
        )
        noise = torch.randint(
            0,
            self.vocab_size,
            x.shape,
            device=x.device,
            generator=generator,
        )
        xt = torch.where(move_indices, noise, x)
        xt.noise = noise
        return xt

    def sample_prior(self, n_samples=1, generator=None):
        device = next(self.model.parameters()).device
        noise = torch.randint(
            0,
            self.vocab_size,
            (n_samples, self.config.model.length),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        xt = noise.clone()
        xt.noise = noise
        return xt

    def _ancestral_step(
        self,
        xt,
        x0,
        alpha_t,
        alpha_s,
        conditioning_tokens=None,
        generator=None,
        **kwargs,
    ):
        noise = conditioning_tokens
        if noise is None:
            noise = getattr(xt, "noise", None)
        if noise is None:
            raise ValueError("Missing conditioning noise on `xt.noise`.")
        unnormalized_probs = (alpha_s - alpha_t) * x0
        stay_prob = (1 - alpha_s).to(unnormalized_probs.dtype)
        unnormalized_probs.scatter_add_(
            dim=-1,
            index=xt.unsqueeze(-1),
            src=stay_prob.view(-1, 1, 1).expand(xt.size(0), xt.size(1), 1),
        )
        xt = sample_categorical(unnormalized_probs, generator=generator)
        xt.noise = noise
        return xt

    def elbo_per_token(
        self,
        logits,
        x0,
        xt,
        alpha_t,
        dalpha_t,
        conditioning_tokens=None,
        **kwargs,
    ):
        noise = conditioning_tokens
        if noise is None:
            noise = getattr(xt, "noise", None)
        if noise is None:
            raise ValueError("Missing conditioning noise on `xt.noise`.")
        probs = logits.exp().clamp_min(1e-12)
        p_xt = torch.gather(probs, -1, xt.unsqueeze(-1)).squeeze(-1)
        logp_x0 = torch.gather(logits, -1, x0.unsqueeze(-1)).squeeze(-1)

        active = (xt == noise).to(logits.dtype)
        x_neq_noise = x0 != noise
        mismatch_term = torch.where(
            x_neq_noise,
            1.0 + logp_x0,
            torch.zeros_like(logp_x0),
        )
        rate = (-dalpha_t) / (1.0 - alpha_t)
        return rate * active * (1.0 - p_xt - mismatch_term)


class ReAUDM(AUDM):
    """
    AUDM endpoint corruption with a standard UDLM reverse bridge on x.

    The denoiser is the AUDM posterior p_theta(x0 | x_t, u_t).
    The lifted reverse step samples x0 from that posterior, samples x_s from
    q_{s|0,t}(x_s | x0, x_t; pi), then samples u_s from its exact conditional
    law given (x0, x_s).
    """

    def __init__(self, model, schedule_fn, config, vocab_size, eps=1e-12, **kwargs):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
            **kwargs,
        )
        self.eps = float(eps)

    def _tokenwise_time(self, value, like, dtype):
        value = torch.as_tensor(value, device=like.device, dtype=dtype)
        while value.ndim > like.ndim and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim == 1 and value.shape[0] == like.shape[0]:
            value = value[:, None]
        return value.expand_as(like)

    def _sample_noise_given_x0_xs(self, x0, xs, alpha_s, generator=None):
        if self.vocab_size == 1:
            return torch.zeros_like(xs)

        dtype = torch.float64
        alpha_s = self._tokenwise_time(alpha_s, xs, dtype=dtype)
        same_as_clean = xs.eq(x0)

        keep_prob = 1.0 / (1.0 + (self.vocab_size - 1.0) * alpha_s)
        keep_xs = (
            torch.rand(xs.shape, device=xs.device, dtype=dtype, generator=generator)
            < keep_prob
        )

        other = torch.randint(
            0,
            self.vocab_size - 1,
            xs.shape,
            device=xs.device,
            generator=generator,
        )
        other = other + other.ge(xs).long()

        noise_if_same = torch.where(keep_xs, xs, other)
        return torch.where(same_as_clean, noise_if_same, xs)

    def _ancestral_step(
        self,
        xt,
        x0,
        alpha_t,
        alpha_s,
        conditioning_tokens=None,
        generator=None,
        **kwargs,
    ):
        del kwargs
        noise = conditioning_tokens
        if noise is None:
            noise = getattr(xt, "noise", None)
        if noise is None:
            raise ValueError("Missing conditioning noise on `xt.noise`.")

        x0 = x0.to(torch.float64)
        x0 = sample_categorical(x0, generator=generator)
        x0_bridge = F.one_hot(x0, num_classes=self.vocab_size).to(torch.float64)
        xs = _sample_udlm_bridge_from_probs(
            xt=xt,
            x0=x0_bridge,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            vocab_size=self.vocab_size,
            generator=generator,
        )
        noise_s = self._sample_noise_given_x0_xs(
            x0=x0,
            xs=xs,
            alpha_s=alpha_s,
            generator=generator,
        )
        xs.noise = noise_s
        return xs


class UDLM(DiscreteDiffusion):
    def __init__(
        self,
        model,
        schedule_fn,
        config,
        vocab_size,
        model_prediction="mean_loo",
        **kwargs,
    ):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
            model_prediction="mean_loo",
        )
        assert model_prediction in ["mean", "mean_loo"]
        # The network predicts this representation.
        self.model_prediction = model_prediction
        # UDLM.forward() always returns the LOO representation consumed by its ELBO.
        self.processed_prediction_type = "mean_loo"

    def _process_logits(self, logits, xt, alpha_t, **kwargs):
        return to_loo_denoiser_logits(
            logits=logits,
            xt=xt,
            alpha_t=alpha_t,
            model_prediction_type=self.model_prediction,
        )

    def sample_forward(self, x, t, generator=None):
        alpha_t, _ = self.schedule(t)
        move_indices = (
            torch.rand(*x.shape, device=x.device, generator=generator) < 1 - alpha_t
        )
        uniform_tensor = torch.randint(
            0,
            self.vocab_size,
            x.shape,
            device=x.device,
            generator=generator,
        )
        xt = torch.where(move_indices, uniform_tensor, x)
        return xt

    def sample_prior(self, n_samples=1, generator=None):
        device = next(self.model.parameters()).device
        return torch.randint(
            0,
            self.vocab_size,
            (n_samples, self.config.model.length),
            dtype=torch.long,
            device=device,
            generator=generator,
        )

    def elbo_per_token(self, logits, x0, xt, alpha_t, dalpha_t, **kwargs):
        x_reconst = logits.exp()
        x_bar_theta = (
            self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        )
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)

        const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = (
            (self.vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt
        ) * const.log()
        term2_theta = -term2_coefs * (
            x_bar_theta.log().sum(-1) - self.vocab_size * xbar_theta_xt.log()
        )
        term2_theta = (
            term2_theta
            - self.vocab_size
            * alpha_t
            / (1 - alpha_t)
            * (xbar_theta_x.log() - xbar_theta_xt.log())
            * x_neq_xt
        )
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)

        return diffusion_loss

    def _ancestral_step(self, xt, x0, alpha_t, alpha_s, generator=None, **kwargs):
        del kwargs
        return _sample_udlm_bridge_from_probs(
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            vocab_size=self.vocab_size,
            generator=generator,
        )


class MaximalCoupling(DiscreteDiffusion):
    def __init__(
        self,
        model,
        schedule_fn,
        config,
        model_prediction,
        vocab_size,
        **kwargs,
    ):
        super().__init__(
            model=model,
            schedule_fn=schedule_fn,
            config=config,
            vocab_size=vocab_size,
            model_prediction="mean",
        )
        assert model_prediction in ["mean", "mean_loo"]
        # The network predicts this representation.
        self.model_prediction = model_prediction
        # MaximalCoupling.forward() always returns ordinary denoiser logits.
        self.processed_prediction_type = "mean"

    def _process_logits(self, logits, xt, alpha_t, **kwargs):
        return to_denoiser_logits(
            logits=logits,
            xt=xt,
            alpha_t=alpha_t,
            model_prediction_type=self.model_prediction,
        )

    def sample_forward(self, x, t, generator=None):
        alpha_t, _ = self.schedule(t)
        move_indices = (
            torch.rand(*x.shape, device=x.device, generator=generator) < 1 - alpha_t
        )
        uniform_tensor = torch.randint(
            0,
            self.vocab_size,
            x.shape,
            device=x.device,
            generator=generator,
        )
        xt = torch.where(move_indices, uniform_tensor, x)
        return xt

    def sample_prior(self, n_samples=1, generator=None):
        device = next(self.model.parameters()).device
        return torch.randint(
            0,
            self.vocab_size,
            (n_samples, self.config.model.length),
            dtype=torch.long,
            device=device,
            generator=generator,
        )

    def _ancestral_step(self, xt, x0, alpha_t, alpha_s, generator=None, **kwargs):
        del kwargs
        denom = 1.0 - alpha_t
        w0 = (alpha_s - alpha_t) / denom
        wt = (1.0 - alpha_s) / denom

        q = x0 * w0
        xt = xt.unsqueeze(-1)
        q.scatter_add_(-1, xt, wt.expand_as(xt).to(torch.float64))
        # NOTE: not needed because sample_categorical uses Gumbel-max trick
        # q = q / q.sum(dim=-1, keepdim=True)
        return sample_categorical(q, generator=generator)

    def elbo_per_token(self, logits, x0, xt, alpha_t, dalpha_t, **kwargs):
        probs = logits.exp()
        probs = probs.clamp_min(1e-12)
        p_xt = torch.gather(probs, -1, xt.unsqueeze(-1)).squeeze(-1)
        logp_x0 = torch.gather(probs.log(), -1, x0.unsqueeze(-1)).squeeze(-1)

        r = (-dalpha_t) / (1.0 - alpha_t)
        x_neq = x0 != xt
        mismatch_term = torch.where(
            x_neq,
            1.0 + logp_x0,
            torch.zeros_like(logp_x0),
        )

        return r * (1.0 - p_xt - mismatch_term)
