#!/usr/bin/env python3
"""Leave-one-out sensitivity probe for ancestral sampling."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import dataloader
import ddm
from lit_text_model import LitTextDDM
from utils import _register_resolvers

DEFAULT_CHECKPOINT_PATH = (
    "checkpoints"
    "udlm_elbo_mean_loo.ckpt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/loo_probe")
DEFAULT_HYDRA_OVERRIDES = ["wandb=null"]

DIFFUSION_MODEL_REGISTRY = {
    "udlm": ddm.UDLM,
    "max_coupling": ddm.MaximalCoupling,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Leave-one-out sensitivity probe for ancestral sampling."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="sample_eval")
    parser.add_argument(
        "--algo",
        default="udlm",
        choices=sorted(DIFFUSION_MODEL_REGISTRY),
    )
    parser.add_argument(
        "--model-prediction",
        choices=["mean", "mean_loo"],
        default=None,
    )
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data", default="openwebtext-split")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--nfes", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--positions-per-sample", type=int, default=5)
    parser.add_argument("--perturbation-trials", type=int, default=10)
    parser.add_argument(
        "--save-inputs",
        dest="save_inputs",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-inputs",
        dest="save_inputs",
        action="store_false",
    )
    parser.set_defaults(save_inputs=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _build_hydra_overrides(args):
    overrides = [
        f"seed={args.seed}",
        f"mode={args.mode}",
        f"algo={args.algo}",
        f"eval.checkpoint_path={json.dumps(args.checkpoint_path)}",
        f"data={args.data}",
        f"model.length={args.length}",
        f"sampling.nfes={args.nfes}",
        f"sampling.temperature={args.temperature}",
        f"loader.eval_batch_size={args.eval_batch_size}",
        f"++probe.positions_per_sample={args.positions_per_sample}",
        f"++probe.perturbation_trials={args.perturbation_trials}",
        f"++probe.save_inputs={str(args.save_inputs).lower()}",
        f"++probe.output_dir={json.dumps(str(args.output_dir))}",
        *DEFAULT_HYDRA_OVERRIDES,
    ]
    if args.model_prediction is not None:
        overrides.append(f"algo.model_prediction={args.model_prediction}")
    return overrides


def _compose_config(args):
    _register_resolvers()
    with hydra.initialize(version_base=None, config_path="configs"):
        return hydra.compose(
            config_name="config",
            overrides=_build_hydra_overrides(args),
        )


def _dump_json(path, payload):
    with path.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2)


def _mean(values):
    return float(sum(values) / max(1, len(values)))


def _resolve_diffusion_model(config):
    try:
        return DIFFUSION_MODEL_REGISTRY[str(config.algo.name)]
    except KeyError as exc:
        # This probe only works for models where we can convert denoiser logits
        # into leave-one-out probabilities.
        raise ValueError(f"Unsupported diffusion model: {config.algo.name}") from exc


def _resolve_sampling_step(config):
    ancestral_step = None
    ancestral_step_kwargs = {}

    if config.sampling.corrector is not None:
        ancestral_step = hydra.utils.get_method(
            str(config.sampling.corrector.ancestral_step)
        )
        ancestral_step_kwargs = {
            key: value
            for key, value in config.sampling.corrector.items()
            if key != "ancestral_step"
        }

    return ancestral_step, ancestral_step_kwargs


def _build_sampling_schedule(nfes, corrector_steps, device):
    if not corrector_steps:
        corrected_steps = 0
        ancestral_steps = nfes
    else:
        corrected_steps = (nfes - 1) // (1 + corrector_steps)
        ancestral_steps = nfes - corrected_steps * corrector_steps

    if ancestral_steps < 1:
        raise ValueError(
            f"nfes must induce at least one ancestral step, got nfes={nfes}."
        )

    timesteps = torch.linspace(1, 0, ancestral_steps + 1, device=device)[:-1]
    if timesteps.numel() == 1:
        dt = timesteps[0]
    else:
        dt = timesteps[0] - timesteps[1]

    return timesteps, dt, corrected_steps


def _get_probe_generator(model):
    generator_seed = model.config.sampling.get("generator_seed", None)
    if generator_seed is None:
        generator_seed = int(model.config.seed) + 1
    else:
        generator_seed = int(generator_seed) + 1

    generator_device = "cuda" if model.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(generator_seed)
    return generator


def _compute_step_probs(
    diffusion,
    xt,
    t,
    temperature,
    apply_temp_output,
    **sampling_kwargs,
):
    logits = diffusion.forward(
        xt=xt,
        t=t,
        temperature=temperature,
        apply_temp_output=apply_temp_output,
        **sampling_kwargs,
    ).to(torch.float64)
    return diffusion._truncate_prediction(logits.exp())


def _compute_loo_probe_probs(
    diffusion,
    xt,
    t,
    alpha_t,
    temperature,
    apply_temp_output,
):
    raw_logits = diffusion._get_logits(xt=xt, t=t).to(torch.float64)
    if apply_temp_output:
        raw_logits = raw_logits / temperature

    loo_logits = ddm.to_loo_denoiser_logits(
        logits=raw_logits,
        xt=xt,
        alpha_t=alpha_t,
        model_prediction_type=diffusion.model_prediction,
    )
    if not apply_temp_output:
        loo_logits = loo_logits / temperature

    loo_log_probs = loo_logits.log_softmax(dim=-1)
    return diffusion._truncate_prediction(loo_log_probs.exp())


@torch.no_grad()
def _self_position_distance(
    diffusion,
    xt,
    t,
    alpha_t,
    base_probs,
    temperature,
    apply_temp_output,
    positions_per_sample,
    perturbation_trials,
    generator,
):
    """Estimate self-position sensitivity via single-token perturbations."""
    batch_size, seq_len = xt.shape
    vocab_size = base_probs.shape[-1]
    total_distance = 0.0
    total_probes = 0
    first_positions = None

    for _ in range(perturbation_trials):
        trial_positions = []
        for _ in range(positions_per_sample):
            position = torch.randint(
                0,
                seq_len,
                (),
                device=xt.device,
                generator=generator,
            )
            positions = torch.full(
                (batch_size, 1),
                int(position.item()),
                device=xt.device,
                dtype=torch.long,
            )
            trial_positions.append(int(position.item()))

            gather_index = positions.unsqueeze(-1).expand(-1, -1, vocab_size)
            base_selected = torch.gather(base_probs, dim=1, index=gather_index)

            xt_perturbed = xt.clone()
            old_tokens = torch.gather(xt_perturbed, dim=1, index=positions)
            if vocab_size > 1:
                new_tokens = torch.randint(
                    0,
                    vocab_size - 1,
                    old_tokens.shape,
                    device=xt.device,
                    generator=generator,
                )
                new_tokens = new_tokens + (new_tokens >= old_tokens).long()
            else:
                new_tokens = old_tokens
            xt_perturbed.scatter_(dim=1, index=positions, src=new_tokens)

            pert_probs = _compute_loo_probe_probs(
                diffusion=diffusion,
                xt=xt_perturbed,
                t=t,
                alpha_t=alpha_t,
                temperature=temperature,
                apply_temp_output=apply_temp_output,
            )
            pert_selected = torch.gather(pert_probs, dim=1, index=gather_index)
            distance = 0.5 * torch.abs(pert_selected - base_selected).sum(dim=-1)
            total_distance += distance.mean().item()
            total_probes += 1

        if first_positions is None:
            first_positions = trial_positions

    mean_distance = total_distance / max(1, total_probes)
    return mean_distance, first_positions


@torch.no_grad()
def sample_with_probe(
    model,
    nfes,
    temperature,
    positions_per_sample,
    perturbation_trials,
    save_inputs,
):
    """Run ancestral sampling while recording leave-one-out sensitivity."""
    diffusion = model.diffusion
    device = next(diffusion.model.parameters()).device
    sampling_generator = model._get_sampling_generator()
    probe_generator = _get_probe_generator(model)
    ancestral_step, ancestral_step_kwargs = _resolve_sampling_step(model.config)
    corrector_steps = ancestral_step_kwargs.get("corrector_steps", 0)
    timesteps, dt, corrected_steps = _build_sampling_schedule(
        nfes=int(nfes),
        corrector_steps=corrector_steps,
        device=device,
    )

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    sampling_kwargs = dict(ancestral_step_kwargs)
    sampling_kwargs.setdefault("inference_steps", len(timesteps))
    apply_temp_output = bool(model.config.sampling.apply_temp_output)
    time_conditioning = getattr(diffusion.model, "time_conditioning", True)
    update_fn = ancestral_step or diffusion._ancestral_step

    xt = diffusion.sample_prior(
        n_samples=int(model.config.loader.eval_batch_size),
        generator=sampling_generator,
    )
    prev_xt = None

    per_step_distance = []
    trace = []

    for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
        s = t - dt
        alpha_t, _ = diffusion.schedule(t)
        alpha_s, _ = diffusion.schedule(s)
        alpha_t, alpha_s = torch.atleast_3d(alpha_t), torch.atleast_3d(alpha_s)

        should_refresh_probs = (
            prev_xt is None or time_conditioning or not torch.equal(xt, prev_xt)
        )
        if should_refresh_probs:
            sampling_probs = _compute_step_probs(
                diffusion=diffusion,
                xt=xt,
                t=t,
                temperature=temperature,
                apply_temp_output=apply_temp_output,
                **sampling_kwargs,
            )
            probe_probs = _compute_loo_probe_probs(
                diffusion=diffusion,
                xt=xt,
                t=t,
                alpha_t=alpha_t,
                temperature=temperature,
                apply_temp_output=apply_temp_output,
            )

        distance, sampled_positions = _self_position_distance(
            diffusion=diffusion,
            xt=xt,
            t=t,
            alpha_t=alpha_t,
            base_probs=probe_probs,
            temperature=temperature,
            apply_temp_output=apply_temp_output,
            positions_per_sample=positions_per_sample,
            perturbation_trials=perturbation_trials,
            generator=probe_generator,
        )
        per_step_distance.append(distance)

        step_record = {
            "step": step_idx,
            "t": float(t.item()),
            "loo_distance": float(distance),
            "sampled_positions_shared_batch": sampled_positions,
            "predicted_tokens": sampling_probs.argmax(dim=-1).detach().cpu().tolist(),
        }
        if save_inputs:
            step_record["xt"] = xt.detach().cpu().tolist()
        trace.append(step_record)

        prev_xt = xt
        step_kwargs = dict(sampling_kwargs)
        step_kwargs["corrector_steps"] = (
            corrector_steps if step_idx < corrected_steps else 0
        )
        xt = update_fn(
            xt=xt,
            x0=sampling_probs,
            t=t,
            s=s,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            model=diffusion,
            generator=sampling_generator,
            **step_kwargs,
        )

    return xt, per_step_distance, trace


def _plot_curve(curve, path, mode):
    plt.figure(figsize=(8, 4))
    plt.plot(list(range(len(curve))), curve)
    plt.xlabel("Sampling step")
    plt.ylabel("Self-position TV distance")
    plt.title(f"Leave-one-out sensitivity during sampling ({mode})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _resolve_output_dir(base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / timestamp


def run(config):
    L.seed_everything(int(config.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = dataloader.get_tokenizer(config)
    model_prediction = getattr(config.algo, "model_prediction", None)
    model = LitTextDDM(
        config=config,
        diffusion_model=_resolve_diffusion_model(config),
        tokenizer=tokenizer,
        model_prediction=model_prediction,
    )
    model.load_backbone_from_checkpoint(config.eval.checkpoint_path)
    model = model.to(device)
    model._eval_mode()

    try:
        final_xt, curve, trace = sample_with_probe(
            model=model,
            nfes=config.sampling.nfes,
            temperature=config.sampling.temperature,
            positions_per_sample=config.probe.positions_per_sample,
            perturbation_trials=config.probe.perturbation_trials,
            save_inputs=config.probe.save_inputs,
        )
    finally:
        # _eval_mode() may swap EMA weights into the live model; restore them.
        model._train_mode()

    output_dir = _resolve_output_dir(config.probe.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_png = output_dir / "loo_distance_curve.png"
    curve_json = output_dir / "loo_distance_curve.json"
    trace_json = output_dir / "sampling_trace.json"
    final_samples_json = output_dir / "final_samples.json"

    _plot_curve(
        curve=curve,
        path=curve_png,
        mode=config.algo.model_prediction,
    )
    _dump_json(
        curve_json,
        {
            "mean_distance": _mean(curve),
            "max_distance": float(max(curve) if curve else 0.0),
            "curve": [float(x) for x in curve],
        },
    )
    _dump_json(trace_json, trace)

    decoded = tokenizer.batch_decode(final_xt.detach().cpu())
    _dump_json(final_samples_json, {"samples": decoded})

    print(f"Saved curve plot: {curve_png}")
    print(f"Saved curve values: {curve_json}")
    print(f"Saved per-step trace: {trace_json}")
    print(f"Saved decoded final samples: {final_samples_json}")
    print(f"Mean self-position TV distance: {_mean(curve):.6f}")


def main(argv=None):
    args = parse_args(argv)
    config = _compose_config(args)
    run(config)


if __name__ == "__main__":
    main()
