from __future__ import annotations

import torch

from ddm import to_denoiser_logits, to_loo_denoiser_logits


@torch.no_grad()
def confidence_based_step(
    xt: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    *,
    diffusion,
    inference_steps: int | None = None,
    k_per_step: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Confidence-guided update with ancestral-step signature.

    - compute token confidence as (best_prob - current_token_prob)
    - select top-k positions with the lowest confidence
    - set those positions to argmax token ("fully denoise" selected coords)

    """
    B, L = xt.shape
    if L <= 0:
        return xt

    forbid_bos = bool(getattr(diffusion, "ignore_bos", False))

    valid_count = L - (1 if (forbid_bos and L > 0) else 0)
    if valid_count <= 0:
        return xt

    if k_per_step is None or int(k_per_step) <= 0:
        if int(inference_steps) < 1:
            raise ValueError("inference_steps must be >= 1 when provided.")
        # Ceil so auto schedule can cover the full sequence over all steps.
        k_eff = (valid_count + int(inference_steps) - 1) // int(inference_steps)
        k_eff = min(valid_count, max(1, k_eff))
    else:
        k_eff = min(valid_count, max(1, int(k_per_step)))

    x0_denoiser_logits = to_denoiser_logits(
        x0.log(),
        xt,
        alpha_t,
        diffusion.model_prediction,
    )

    best_prob, best_tok = x0_denoiser_logits.max(dim=-1)
    current_prob = x0_denoiser_logits.gather(-1, xt.unsqueeze(-1)).squeeze(-1)
    scores = best_prob - current_prob

    if forbid_bos and L > 0:
        scores = scores.clone()
        scores[:, 0] = -float("inf")

    pos = torch.topk(scores, k=k_eff, dim=-1, largest=True).indices
    batch = torch.arange(B, device=xt.device)[:, None].expand(-1, k_eff)

    updated = xt.clone()
    updated[batch, pos] = best_tok[batch, pos]
    return updated


@torch.no_grad()
def sample_paper_confidence(
    model,
    *,
    num_samples: int,
    inference_steps: int,
    temperature: float | None = None,
    k_per_step: int | None = None,
    forbid_bos: bool | None = None,
):
    """Compatibility wrapper using current `generate_samples` API."""

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    if inference_steps < 1:
        raise ValueError("inference_steps must be >= 1.")

    if temperature is None:
        temperature = float(model.config.sampling.temperature)

    return model.generate_samples(
        num_samples=int(num_samples),
        nfes=int(inference_steps),
        temperature=float(temperature),
        ancestral_step=confidence_based_step,
        ancestral_step_kwargs={
            "diffusion": model,
            "inference_steps": int(inference_steps),
            "k_per_step": k_per_step,
            "forbid_bos": forbid_bos,
        },
    )
