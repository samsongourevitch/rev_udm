import torch
import math

from ddm import to_loo_denoiser_logits


@torch.no_grad()
def _compute_pt_conditional(
    model,
    xt: torch.Tensor,
    t: torch.Tensor | float,
    pc_temperature: float,
) -> torch.Tensor:
    diffusion = model
    # NOTE: to fix
    del pc_temperature
    alpha_t, _ = diffusion.schedule(t)
    logits = diffusion.forward(xt=xt, t=t)
    logits_loo = to_loo_denoiser_logits(
        logits=logits,
        xt=xt,
        alpha_t=alpha_t,
        model_prediction_type=diffusion.model_prediction,
    )
    logits_loo = logits_loo.to(torch.float64)
    log_alpha = alpha_t.log()
    log_uniform = torch.log1p(-alpha_t) - math.log(logits_loo.shape[-1])

    return torch.logaddexp(
        log_alpha + logits_loo,
        log_uniform,
    )


def _pick_resample_positions(
    *,
    xt: torch.Tensor,
    cond_logits: torch.Tensor,
    selection: str,
    selection_tau: float,
    k_parallel: int,
    forbid_bos: bool,
) -> torch.Tensor:
    batch_size, seq_len = xt.shape
    device = xt.device

    valid = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if forbid_bos and seq_len > 0:
        valid[:, 0] = False

    valid_count = seq_len - (1 if (forbid_bos and seq_len > 0) else 0)
    if valid_count <= 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

    k_eff = min(max(1, int(k_parallel)), valid_count)

    if selection == "random":
        rand_scores = torch.rand(batch_size, seq_len, device=device)
        rand_scores = rand_scores.masked_fill(~valid, -1.0)
        return torch.topk(rand_scores, k=k_eff, largest=True).indices

    if selection_tau <= 0:
        raise ValueError(f"selection_tau must be > 0, got {selection_tau}")

    if selection == "lowest_maxprob":
        confidence = -cond_logits.max(dim=-1).values
    elif selection == "lowest_current_prob":
        confidence = -torch.gather(cond_logits, -1, xt.unsqueeze(-1)).squeeze(-1)
    elif selection == "lowest_log_margin":
        log_probs = cond_logits
        log_current = torch.gather(log_probs, -1, xt.unsqueeze(-1)).squeeze(-1)
        alt_log_probs = log_probs.clone()
        alt_log_probs.scatter_(-1, xt.unsqueeze(-1), float("-inf"))
        log_best_alt = alt_log_probs.max(dim=-1).values
        confidence = -(log_current - log_best_alt)
    else:
        raise ValueError(f"Unknown selection rule: {selection}")

    pos_logits = (confidence / float(selection_tau)).masked_fill(~valid, float("-inf"))
    # Gumbel-top-k: sampling without replacement from categorical weights.
    u = torch.rand_like(pos_logits).clamp_(1e-12, 1.0 - 1e-12)
    gumbel = -torch.log(-torch.log(u))
    keys = pos_logits + gumbel
    return torch.topk(keys, k=k_eff, largest=True).indices


@torch.no_grad()
def _gibbs_corrector(
    *,
    model,
    x: torch.Tensor,
    t: torch.Tensor | float,
    pc_temperature: float,
    steps: int,
    selection: str,
    selection_tau: float,
    k_parallel: int,
) -> tuple[torch.Tensor, list[float]]:
    diffusion = model

    if steps <= 0:
        return x, []

    batch_size, seq_len = x.shape
    forbid_bos = bool(getattr(diffusion, "ignore_bos", False))
    valid_count = seq_len - (1 if (forbid_bos and seq_len > 0) else 0)
    if valid_count <= 0:
        return x, []

    k_eff = min(max(1, int(k_parallel)), valid_count)
    row_idx = torch.arange(batch_size, device=x.device)[:, None].expand(-1, k_eff)

    for _ in range(steps):
        cond_logits = _compute_pt_conditional(
            model=diffusion,
            xt=x,
            t=t,
            pc_temperature=pc_temperature,
        )
        pos = _pick_resample_positions(
            xt=x,
            cond_logits=cond_logits,
            selection=selection,
            selection_tau=selection_tau,
            k_parallel=k_eff,
            forbid_bos=forbid_bos,
        )

        chosen_probs = cond_logits[row_idx, pos].exp()
        # current_tokens = x[row_idx, pos]
        # curr_tok_prob = chosen_probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(
        #     -1
        # )
        # selected_confidences.append(float(curr_tok_prob.mean().item()))

        sampled = torch.multinomial(
            input=chosen_probs.reshape(-1, chosen_probs.shape[-1]),
            num_samples=1,
        ).squeeze(-1)
        sampled = sampled.reshape(batch_size, k_eff)

        x = x.clone()
        x[row_idx, pos] = sampled

    return x  # , selected_confidences


@torch.no_grad()
def predictor_corrector_gibbs(
    xt: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    *,
    diffusion,
    pc_temperature: float,
    corrector_steps: int,
    k_parallel: int,
    selection: str,
    selection_tau: float,
    **kwargs,
) -> torch.Tensor:
    del kwargs

    x = diffusion._ancestral_step(
        xt=xt,
        x0=x0,
        alpha_t=alpha_t,
        alpha_s=alpha_s,
    )

    if corrector_steps <= 0:
        return x

    if s == 0.0:
        return x

    x = _gibbs_corrector(
        model=diffusion,
        x=x,
        t=s,
        pc_temperature=pc_temperature,
        steps=corrector_steps,
        selection=selection,
        selection_tau=selection_tau,
        k_parallel=k_parallel,
    )
    return x
