#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

"""
Usage: 

python3 eval_scripts/temperature_sweep/temperature_sweep.py prepare --run-name temp_eval_foo
python3 eval_scripts/temperature_sweep/temperature_sweep.py launch --run-name temp_eval_foo
python3 eval_scripts/temperature_sweep/temperature_sweep.py finalize --run-name temp_eval_foo

"""
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/sweeps/temp_eval_manifest.yaml")
CHECKPOINT_OVERRIDE_PREFIXES = (
    "eval.checkpoint_path=",
    "+eval.checkpoint_path=",
    "++eval.checkpoint_path=",
)

LOSS_HINTS = [
    "max_coupling",
    "udlm",
    "duo_base",
    "duo",
    "mdlm",
    "ar",
    "d3pm",
    "sedd",
]


def infer_train_loss(checkpoint_path: str) -> str:
    lowered = normalize_checkpoint_path(checkpoint_path).lower()
    for hint in LOSS_HINTS:
        if hint in lowered:
            return hint
    return "unknown"


def temp_tag(temperature: float) -> str:
    return f"{temperature:.2f}".replace(".", "p")


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def entry_dir_name(manifest_id: str) -> str:
    identifier = slug(manifest_id)
    if not identifier:
        identifier = "unknown"
    return identifier if identifier.startswith("entry") else f"entry{identifier}"


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as file_obj:
        for line in file_obj:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")


def load_config(config_path: Path):
    return OmegaConf.load(config_path)


def strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_checkpoint_path(raw_path: str) -> str:
    path = str(raw_path).strip()
    for prefix in CHECKPOINT_OVERRIDE_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix) :].strip()
            break
    return strip_matching_quotes(path)


def hydra_quote(value) -> str:
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def checkpoint_path_override(raw_path: str) -> str:
    return f"eval.checkpoint_path={hydra_quote(normalize_checkpoint_path(raw_path))}"


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_run_dir(cfg, run_name: str) -> Path:
    return resolve_repo_path(str(cfg.run.output_root)) / run_name


def manifest_id_for_row(entry_idx: int, row, temperature: float) -> str:
    base_id = row.get("id")
    apply_temp_output = bool(row.get("apply_temp_output", True))
    temp_mode = "tempout" if apply_temp_output else "tempin"
    if base_id is None:
        sampler = str(row["sampler"])
        train_loss = str(row["train_loss"])
        return (
            f"entry{entry_idx:02d}_{sampler}_loss-{train_loss}_"
            f"temp{temp_tag(float(temperature))}_{temp_mode}"
        )
    return f"{base_id}_temp{temp_tag(float(temperature))}_{temp_mode}"


def build_manifest_rows(cfg):
    rows = []
    seeds = [int(seed) for seed in cfg.sweep.seeds]
    nfes = [int(step) for step in cfg.sweep.nfe_list]

    for entry_idx, entry in enumerate(cfg.entries):
        sampler = str(entry.sampler)
        checkpoint_path = normalize_checkpoint_path(str(entry.checkpoint_path))
        train_loss = str(entry.get("train_loss", "auto"))
        if train_loss == "auto":
            train_loss = infer_train_loss(checkpoint_path)

        base_row = {
            "sampler": sampler,
            "checkpoint_path": checkpoint_path,
            "train_loss": train_loss,
            "seeds": seeds,
            "nfes": nfes,
            "eval_batch_size": int(cfg.sweep.eval_batch_size),
            "num_sample_batches": int(cfg.sweep.num_sample_batches),
            "apply_temp_output": bool(cfg.sweep.apply_temp_output),
            "compute_mauve": bool(cfg.sweep.compute_mauve),
            "disable_wandb": bool(cfg.sweep.disable_wandb),
        }
        if entry.get("model_prediction") is not None:
            base_row["model_prediction"] = str(entry.model_prediction)
        if entry.get("apply_temp_output") is not None:
            base_row["apply_temp_output"] = bool(entry.apply_temp_output)
        if entry.get("extra_overrides") is not None:
            base_row["extra_overrides"] = [str(item) for item in entry.extra_overrides]

        for temperature in cfg.sweep.temperatures:
            row = dict(base_row)
            row["temperature"] = float(temperature)
            row["id"] = manifest_id_for_row(entry_idx, row, float(temperature))
            rows.append(row)

    return rows


def row_expected_count(row, cfg):
    seeds = [int(seed) for seed in row.get("seeds", cfg.sweep.seeds)]
    nfes = [int(step) for step in row.get("nfes", cfg.sweep.nfe_list)]
    return len(seeds) * len(nfes)


def entry_dir_for_row(run_dir: Path, row) -> Path:
    return run_dir / entry_dir_name(str(row.get("id")))


def is_row_complete(run_dir: Path, row, cfg, repair_ok: bool = False) -> bool:
    entry_dir = entry_dir_for_row(run_dir, row)
    ok_path = entry_dir / "OK"
    status_path = entry_dir / "status.json"

    if ok_path.exists():
        return True

    if status_path.exists():
        with status_path.open("r") as file_obj:
            payload = json.load(file_obj)
        complete = bool(payload.get("complete", False))
        if complete and repair_ok:
            ok_path.write_text("ok\n")
        return complete

    records_dir = entry_dir / "records"
    if not records_dir.exists():
        return False

    expected = row_expected_count(row, cfg)
    records = list(records_dir.glob("record_*.json"))
    if len(records) < expected:
        return False

    success_count = 0
    for record_path in records:
        with record_path.open("r") as file_obj:
            payload = json.load(file_obj)
        if payload.get("status") == "success":
            success_count += 1

    complete = success_count >= expected and len(records) >= expected
    if complete and repair_ok:
        ok_path.write_text("ok\n")
    return complete


def merge_run(run_dir: Path, output_dir=None):
    import pandas as pd

    record_files = sorted(run_dir.glob("entry*/records/record_*.json"))
    if not record_files:
        raise SystemExit(f"No record files found under {run_dir}")

    rows = []
    for path in record_files:
        with path.open("r") as file_obj:
            row = json.load(file_obj)
        row["source_record"] = str(path)
        rows.append(row)

    merged = pd.DataFrame(rows)
    preferred_dedup_keys = [
        "manifest_id",
        "sampler_algo",
        "model_prediction_algo",
        "train_loss_algo",
        "checkpoint_path",
        "temperature",
        "nfe",
        "seed",
    ]
    dedup_keys = [key for key in preferred_dedup_keys if key in merged.columns]
    if not dedup_keys:
        raise SystemExit("No valid dedup keys found in merged records")

    ranked = merged.copy()
    if "status" not in ranked.columns:
        if "return_code" in ranked.columns:
            ranked["status"] = ranked["return_code"].apply(
                lambda code: "success" if int(code) == 0 else "failed"
            )
        else:
            ranked["status"] = "unknown"

    ranked["_is_success"] = (ranked["status"] == "success").astype(int)
    sort_columns = dedup_keys + ["_is_success", "source_record"]
    ranked = ranked.sort_values(sort_columns)
    dedup = ranked.drop_duplicates(subset=dedup_keys, keep="last").drop(
        columns=["_is_success"]
    )
    success = dedup[dedup["status"] == "success"].copy()

    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "merged_all_rows.csv", index=False)
    dedup.to_csv(output_dir / "merged_dedup.csv", index=False)
    success.to_csv(output_dir / "merged_success.csv", index=False)

    return output_dir / "merged_success.csv"


def should_merge(run_dir: Path, merged_success: Path) -> bool:
    if not merged_success.exists():
        return True

    merged_all = run_dir / "merged_all_rows.csv"
    merged_dedup = run_dir / "merged_dedup.csv"
    if not merged_all.exists() or not merged_dedup.exists():
        return True

    record_files = sorted(run_dir.glob("entry*/records/record_*.json"))
    if not record_files:
        return False

    merged_mtime = min(
        merged_success.stat().st_mtime,
        merged_all.stat().st_mtime,
        merged_dedup.stat().st_mtime,
    )
    latest_record_mtime = max(path.stat().st_mtime for path in record_files)
    return latest_record_mtime > merged_mtime


def plot_frontier(
    summary_csv: Path,
    output_dir: Path,
    sampler_only: bool,
    seed,
    min_entropy,
    max_entropy,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(summary_csv)
    df = df[df["status"] == "success"].dropna(
        subset=["entropy", "temperature", "nfe", "seed"]
    )
    if seed is not None:
        df = df[df["seed"] == seed]
    if min_entropy is not None:
        df = df[df["entropy"] >= float(min_entropy)]
    if max_entropy is not None:
        df = df[df["entropy"] <= float(max_entropy)]
    if df.empty:
        raise SystemExit("No rows to plot after filtering")

    if sampler_only:
        df["curve"] = (
            df["sampler_algo"].astype(str)
            + " | pred="
            + df["model_prediction_algo"].fillna("none").astype(str)
        )
    else:
        df["curve"] = (
            df["sampler_algo"].astype(str)
            + " | loss="
            + df["train_loss_algo"].astype(str)
            + " | pred="
            + df["model_prediction_algo"].fillna("none").astype(str)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for seed_value, seed_df in df.groupby("seed"):
        seed_dir = output_dir / f"seed_{int(seed_value)}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for nfe, sub in seed_df.groupby("nfe"):
            ppl_sub = sub.dropna(subset=["generative_ppl"])
            if not ppl_sub.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                for curve_name, curve_df in ppl_sub.groupby("curve"):
                    curve_df = curve_df.sort_values("temperature")
                    line = ax.plot(
                        curve_df["entropy"],
                        curve_df["generative_ppl"],
                        marker="o",
                        markersize=2,
                        linewidth=1.0,
                        label=curve_name,
                    )[0]

                    temp_one = curve_df[
                        curve_df["temperature"].apply(
                            lambda value: abs(float(value) - 1.0) < 1e-9
                        )
                    ]
                    if not temp_one.empty:
                        ax.plot(
                            temp_one["entropy"],
                            temp_one["generative_ppl"],
                            linestyle="None",
                            marker="*",
                            markersize=10,
                            color=line.get_color(),
                            zorder=4,
                        )

                ax.set_xlabel("Entropy")
                ax.set_ylabel("Generative PPL")
                ax.set_yscale("log")
                ax.set_title(f"Frontier (seed={int(seed_value)}, NFE={int(nfe)})")
                ax.grid(True, alpha=0.3)
                ax.legend()

                out_path = seed_dir / f"frontier_nfe{int(nfe)}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            mauve_sub = sub.dropna(subset=["mauve"])
            if not mauve_sub.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                for curve_name, curve_df in mauve_sub.groupby("curve"):
                    curve_df = curve_df.sort_values("temperature")
                    line = ax.plot(
                        curve_df["entropy"],
                        curve_df["mauve"],
                        marker="o",
                        markersize=2,
                        linewidth=1.0,
                        label=curve_name,
                    )[0]

                    temp_one = curve_df[
                        curve_df["temperature"].apply(
                            lambda value: abs(float(value) - 1.0) < 1e-9
                        )
                    ]
                    if not temp_one.empty:
                        ax.plot(
                            temp_one["entropy"],
                            temp_one["mauve"],
                            linestyle="None",
                            marker="*",
                            markersize=10,
                            color=line.get_color(),
                            zorder=4,
                        )

                ax.set_xlabel("Entropy")
                ax.set_ylabel("MAUVE")
                ax.set_title(
                    f"MAUVE vs Entropy (seed={int(seed_value)}, NFE={int(nfe)})"
                )
                ax.grid(True, alpha=0.3)
                ax.legend()

                out_path = seed_dir / f"mauve_vs_entropy_nfe{int(nfe)}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            if not ppl_sub.empty:
                print_frontier_table(ppl_sub, seed_value=int(seed_value), nfe=int(nfe))


def format_frontier_cell(row) -> str:
    return f"({float(row['generative_ppl']):.4g}, {float(row['entropy']):.4g})"


def print_frontier_table(sub, seed_value: int, nfe: int):
    import pandas as pd

    temperatures = sorted(float(value) for value in sub["temperature"].unique())
    curve_names = sorted(str(value) for value in sub["curve"].unique())
    columns = [f"{temp:.2f}" for temp in temperatures]

    table = pd.DataFrame(index=curve_names, columns=columns, dtype=object)
    table.index.name = "configuration"

    for _, row in sub.iterrows():
        table.at[str(row["curve"]), f"{float(row['temperature']):.2f}"] = (
            format_frontier_cell(row)
        )

    table = table.fillna("")
    print()
    print(f"Frontier table (seed={seed_value}, NFE={nfe})")
    print(table.to_string())


def cmd_prepare(args):
    cfg = load_config(args.config.resolve())
    run_dir = resolve_run_dir(cfg, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (run_dir / "manifest.jsonl")
    pending_path = args.pending or (run_dir / "pending_manifest.jsonl")

    if args.refresh_manifest or not manifest_path.exists():
        manifest_rows = build_manifest_rows(cfg)
        write_jsonl(manifest_path, manifest_rows)
        manifest_mode = "refreshed" if args.refresh_manifest else "created"
    else:
        manifest_rows = read_jsonl(manifest_path)
        manifest_mode = "reused"

    pending_rows = []
    complete_rows = 0
    for row in manifest_rows:
        if is_row_complete(run_dir, row, cfg, repair_ok=args.repair_ok):
            complete_rows += 1
        else:
            pending_rows.append(row)

    write_jsonl(pending_path, pending_rows)

    print(f"Run dir: {run_dir}")
    print(f"Manifest ({manifest_mode}): {manifest_path}")
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Complete rows: {complete_rows}")
    print(f"Pending rows: {len(pending_rows)}")
    print(f"Pending manifest: {pending_path}")


def cmd_launch(args):
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, args.run_name)
    manifest_path = args.manifest or (run_dir / "pending_manifest.jsonl")
    manifest_path = manifest_path.resolve()

    rows = read_jsonl(manifest_path)
    if not rows:
        raise SystemExit(f"Manifest has no entries: {manifest_path}")

    array_range = f"0-{len(rows) - 1}"
    slurm_script = (
        Path(__file__).resolve().parent
        / "bash_scripts"
        / "run_temperature_manifest_slurm_array.sh"
    )

    cmd = [
        "sbatch",
        f"--array={array_range}",
        str(slurm_script),
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        f"manifest.path={manifest_path}",
        f"run.name={args.run_name}",
        *args.overrides,
    ]

    print(f"Submitting Slurm array {array_range} from manifest {manifest_path}")
    print(f"Using shared run.name={args.run_name}")
    subprocess.run(cmd, check=True)


def cmd_finalize(args):
    cfg = load_config(args.config.resolve())
    run_dir = resolve_run_dir(cfg, args.run_name)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    merged_success = run_dir / "merged_success.csv"
    if args.force_merge or should_merge(run_dir, merged_success):
        merged_success = merge_run(run_dir)
    else:
        print(f"Reusing merged file: {merged_success}")

    plots_dir = args.plots_dir or (run_dir / "frontier")
    plot_frontier(
        summary_csv=merged_success,
        output_dir=plots_dir,
        sampler_only=args.sampler_only,
        seed=args.seed,
        min_entropy=args.min_entropy,
        max_entropy=args.max_entropy,
    )

    print(f"Run dir: {run_dir}")
    print(f"Merged summary: {merged_success}")
    print(f"Plots dir: {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature sweep workflow.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser(
        "prepare", help="Create or reuse manifest.jsonl, then filter pending rows."
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--manifest", type=Path, default=None)
    prepare.add_argument("--pending", type=Path, default=None)
    prepare.add_argument("--refresh-manifest", action="store_true")
    prepare.add_argument("--repair-ok", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    launch = sub.add_parser(
        "launch", help="Submit the pending manifest as a Slurm array."
    )
    launch.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    launch.add_argument("--run-name", required=True)
    launch.add_argument("--manifest", type=Path, default=None)
    launch.add_argument("overrides", nargs=argparse.REMAINDER)
    launch.set_defaults(func=cmd_launch)

    finalize = sub.add_parser(
        "finalize", help="Merge run outputs if needed and plot frontiers."
    )
    finalize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize.add_argument("--run-name", required=True)
    finalize.add_argument("--force-merge", action="store_true")
    finalize.add_argument("--plots-dir", type=Path, default=None)
    finalize.add_argument("--sampler-only", action="store_true")
    finalize.add_argument("--seed", type=int, default=None)
    finalize.add_argument("--min-entropy", type=float, default=None)
    finalize.add_argument("--max-entropy", type=float, default=None)
    finalize.set_defaults(func=cmd_finalize)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
