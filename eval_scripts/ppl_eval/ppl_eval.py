#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

TEMPERATURE_SWEEP_DIR = Path(__file__).resolve().parents[1] / "temperature_sweep"
if str(TEMPERATURE_SWEEP_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPERATURE_SWEEP_DIR))

from temperature_sweep import (
    entry_dir_name,
    infer_train_loss,
    read_jsonl,
    write_jsonl,
)

"""
Usage:

python3 eval_scripts/ppl_eval/ppl_eval.py prepare --run-name ppl_eval_foo
python3 eval_scripts/ppl_eval/ppl_eval.py launch --run-name ppl_eval_foo
python3 eval_scripts/ppl_eval/ppl_eval.py finalize --run-name ppl_eval_foo

"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/sweeps/ppl_eval_manifest.yaml")


def latex_escape(value) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def format_metric(mean_value, std_value, count: int, precision: int = 4) -> str:
    if mean_value is None:
        return ""
    if count <= 1 or std_value is None:
        return f"{float(mean_value):.{precision}f}"
    return f"{float(mean_value):.{precision}f} $\\pm$ {float(std_value):.{precision}f}"


def build_latex_table(aggregated_df, run_name: str) -> str:
    row_columns = [
        ("sampler_algo", "Method"),
        ("train_loss_algo", "Train loss"),
        ("model_prediction_algo", "Model prediction"),
    ]

    if not aggregated_df.empty:
        work_df = aggregated_df.copy()
        work_df["model_prediction_algo"] = work_df["model_prediction_algo"].fillna("none")
        work_df = work_df.sort_values(
            [
                column
                for column in [
                    "sampler_algo",
                    "train_loss_algo",
                    "model_prediction_algo",
                    "data",
                    "val_ppl_mean",
                ]
                if column in work_df.columns
            ],
            na_position="last",
        )
        work_df = work_df.drop_duplicates(
            subset=[
                column
                for column in [
                    "sampler_algo",
                    "train_loss_algo",
                    "model_prediction_algo",
                    "data",
                ]
                if column in work_df.columns
            ],
            keep="first",
        )
        dataset_columns = [
            str(value) for value in sorted(work_df["data"].dropna().unique())
        ]
    else:
        work_df = aggregated_df.copy()
        dataset_columns = []

    header_columns = [header for _, header in row_columns] + dataset_columns
    alignment = "lll" + "r" * len(dataset_columns)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Validation PPL leaderboard for {latex_escape(run_name)}}}",
        rf"\label{{tab:ppl-eval-{latex_escape(run_name)}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in header_columns) + r" \\",
        r"\midrule",
    ]

    if work_df.empty:
        lines.append(
            rf"\multicolumn{{{max(len(header_columns), 1)}}}{{c}}{{No successful rows found}} \\")
    else:
        pivot_display = work_df.pivot(
            index=[key for key, _ in row_columns],
            columns="data",
            values="val_ppl_display",
        )
        pivot_display = pivot_display.reindex(columns=dataset_columns)

        pivot_values = work_df.pivot(
            index=[key for key, _ in row_columns],
            columns="data",
            values="val_ppl_mean",
        )
        pivot_values = pivot_values.reindex(columns=dataset_columns)

        dataset_ranks = {}
        for dataset_name in dataset_columns:
            valid_values = [
                float(value)
                for value in pivot_values[dataset_name].dropna().tolist()
            ]
            unique_values = sorted(set(valid_values))
            best_value = unique_values[0] if unique_values else None
            second_value = unique_values[1] if len(unique_values) > 1 else None
            dataset_ranks[dataset_name] = {
                "best": best_value,
                "second": second_value,
            }

        pivot_display = pivot_display.reset_index()
        pivot_values = pivot_values.reset_index()

        for row_idx, row in pivot_display.iterrows():
            row_values = [latex_escape(row.get(key, "")) for key, _ in row_columns]
            for dataset_name in dataset_columns:
                display_value = row.get(dataset_name, "") or ""
                numeric_value = pivot_values.iloc[row_idx].get(dataset_name)
                if display_value == "" or numeric_value is None:
                    row_values.append("")
                    continue

                rendered = str(display_value)
                best_value = dataset_ranks[dataset_name]["best"]
                second_value = dataset_ranks[dataset_name]["second"]
                numeric_value = float(numeric_value)

                if best_value is not None and numeric_value == best_value:
                    rendered = rf"\textbf{{{rendered}}}"
                elif second_value is not None and numeric_value == second_value:
                    rendered = rf"\underline{{{rendered}}}"
                row_values.append(rendered)
            lines.append(" & ".join(row_values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def load_config(config_path: Path):
    return OmegaConf.load(config_path)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_run_dir(cfg, run_name: str) -> Path:
    return resolve_repo_path(str(cfg.run.output_root)) / run_name


def manifest_id_for_row(entry_idx: int, row) -> str:
    base_id = row.get("id")
    if base_id is None:
        sampler = str(row["sampler"])
        train_loss = str(row["train_loss"])
        data_name = str(row.get("data", "dataset"))
        return f"entry{entry_idx:03d}_{sampler}_loss-{train_loss}_data-{data_name}"
    return str(base_id)


def build_entry_payload(entry_idx: int, entry, default_seeds, cfg):
    sampler = str(entry.sampler)
    checkpoint_path = str(entry.checkpoint_path)
    train_loss = str(entry.get("train_loss", "auto"))
    if train_loss == "auto":
        train_loss = infer_train_loss(checkpoint_path)

    payload = {
        "sampler": sampler,
        "checkpoint_path": checkpoint_path,
        "train_loss": train_loss,
        "seeds": default_seeds,
    }
    if entry.get("model_prediction") is not None:
        payload["model_prediction"] = str(entry.model_prediction)
    if entry.get("seeds") is not None:
        payload["seeds"] = [int(seed) for seed in entry.seeds]
    if entry.get("model_length") is not None:
        payload["model_length"] = int(entry.model_length)
    if entry.get("eval_batch_size") is not None:
        payload["eval_batch_size"] = int(entry.eval_batch_size)
    if entry.get("disable_ema") is not None:
        payload["disable_ema"] = bool(entry.disable_ema)
    if entry.get("disable_wandb") is not None:
        payload["disable_wandb"] = bool(entry.disable_wandb)
    if entry.get("extra_overrides") is not None:
        payload["extra_overrides"] = [str(item) for item in entry.extra_overrides]
    payload["entry_id"] = f"entry{entry_idx:02d}_{sampler}_loss-{train_loss}"
    return payload


def normalize_dataset_specs(cfg):
    dataset_specs = cfg.sweep.get("datasets", None)
    if dataset_specs is None:
        return [
            {
                "data": str(cfg.sweep.data),
                "model_length": int(cfg.sweep.model_length),
                "eval_batch_size": int(cfg.sweep.eval_batch_size),
                "disable_ema": bool(cfg.sweep.disable_ema),
                "disable_wandb": bool(cfg.sweep.disable_wandb),
            }
        ]

    normalized = []
    for spec in dataset_specs:
        if isinstance(spec, str):
            normalized.append(
                {
                    "data": spec,
                    "model_length": int(cfg.sweep.model_length),
                    "eval_batch_size": int(cfg.sweep.eval_batch_size),
                    "disable_ema": bool(cfg.sweep.disable_ema),
                    "disable_wandb": bool(cfg.sweep.disable_wandb),
                }
            )
            continue

        normalized.append(
            {
                "data": str(spec.data),
                "id": str(spec.get("id", spec.data)),
                "model_length": int(spec.get("model_length", cfg.sweep.model_length)),
                "eval_batch_size": int(
                    spec.get("eval_batch_size", cfg.sweep.eval_batch_size)
                ),
                "disable_ema": bool(spec.get("disable_ema", cfg.sweep.disable_ema)),
                "disable_wandb": bool(
                    spec.get("disable_wandb", cfg.sweep.disable_wandb)
                ),
            }
        )
    return normalized


def build_manifest_rows(cfg):
    rows = []
    default_seeds = [int(seed) for seed in cfg.sweep.seeds]

    entry_payloads = [
        build_entry_payload(entry_idx, entry, default_seeds, cfg)
        for entry_idx, entry in enumerate(cfg.entries)
    ]

    row_idx = 0
    for dataset_spec in normalize_dataset_specs(cfg):
        dataset_id = str(dataset_spec.get("id", dataset_spec["data"]))
        for entry_payload in entry_payloads:
            row = {
                "sampler": str(entry_payload["sampler"]),
                "checkpoint_path": str(entry_payload["checkpoint_path"]),
                "train_loss": str(entry_payload["train_loss"]),
                "entry_id": str(entry_payload["entry_id"]),
                "seeds": [int(seed) for seed in entry_payload.get("seeds", default_seeds)],
                "data": str(dataset_spec["data"]),
                "dataset_id": dataset_id,
                "model_length": int(
                    entry_payload.get("model_length", dataset_spec["model_length"])
                ),
                "eval_batch_size": int(
                    entry_payload.get(
                        "eval_batch_size", dataset_spec["eval_batch_size"]
                    )
                ),
                "disable_ema": bool(
                    entry_payload.get("disable_ema", dataset_spec["disable_ema"])
                ),
                "disable_wandb": bool(
                    entry_payload.get(
                        "disable_wandb", dataset_spec["disable_wandb"]
                    )
                ),
            }
            if entry_payload.get("model_prediction") is not None:
                row["model_prediction"] = str(entry_payload["model_prediction"])
            if entry_payload.get("extra_overrides") is not None:
                row["extra_overrides"] = [
                    str(item) for item in entry_payload["extra_overrides"]
                ]
            row["id"] = f"{entry_payload['entry_id']}_data-{dataset_id}"
            row["id"] = manifest_id_for_row(row_idx, row)
            rows.append(row)
            row_idx += 1

    return rows


def row_expected_count(row, cfg):
    seeds = [int(seed) for seed in row.get("seeds", cfg.sweep.seeds)]
    return len(seeds)


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


def merge_run(run_dir: Path, output_dir: Path | None = None):
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
        "entry_id",
        "sampler_algo",
        "model_prediction_algo",
        "train_loss_algo",
        "checkpoint_path",
        "data",
        "model_length",
        "disable_ema",
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
    merged_all_path = output_dir / "merged_all_rows.csv"
    merged_dedup_path = output_dir / "merged_dedup.csv"
    merged_success_path = output_dir / "merged_success.csv"
    leaderboard_path = output_dir / "leaderboard_val_ppl.csv"
    leaderboard_agg_path = output_dir / "leaderboard_val_ppl_agg.csv"
    leaderboard_tex_path = output_dir / "leaderboard_val_ppl.tex"

    merged.to_csv(merged_all_path, index=False)
    dedup.to_csv(merged_dedup_path, index=False)
    success.to_csv(merged_success_path, index=False)

    leaderboard = success.copy()
    leaderboard_sort = [
        column
        for column in [
            "data",
            "val_ppl",
            "val_nll",
            "sampler_algo",
            "model_prediction_algo",
            "manifest_id",
            "seed",
        ]
        if column in leaderboard.columns
    ]
    if leaderboard_sort:
        leaderboard = leaderboard.sort_values(leaderboard_sort, na_position="last")
    leaderboard.to_csv(leaderboard_path, index=False)

    group_columns = [
        column
        for column in [
            "data",
            "sampler_algo",
            "train_loss_algo",
            "model_prediction_algo",
            "model_length",
            "disable_ema",
        ]
        if column in success.columns
    ]

    if success.empty:
        aggregated = pd.DataFrame(
            columns=group_columns
            + [
                "num_seeds",
                "seed_list",
                "val_ppl_mean",
                "val_ppl_std",
                "val_nll_mean",
                "val_nll_std",
                "val_bpd_mean",
                "val_bpd_std",
                "val_ppl_display",
                "val_nll_display",
                "val_bpd_display",
            ]
        )
    else:
        aggregated = (
            success.groupby(group_columns, dropna=False)
            .agg(
                num_seeds=("seed", "nunique"),
                seed_list=(
                    "seed",
                    lambda values: ",".join(
                        str(int(seed)) for seed in sorted(set(values))
                    ),
                ),
                val_ppl_mean=("val_ppl", "mean"),
                val_ppl_std=("val_ppl", "std"),
                val_nll_mean=("val_nll", "mean"),
                val_nll_std=("val_nll", "std"),
                val_bpd_mean=("val_bpd", "mean"),
                val_bpd_std=("val_bpd", "std"),
            )
            .reset_index()
        )
        if "data" in aggregated.columns:
            aggregated["data"] = aggregated["data"].fillna("unknown")
        aggregated["model_prediction_algo"] = aggregated["model_prediction_algo"].fillna("none")
        aggregated["val_ppl_display"] = aggregated.apply(
            lambda row: format_metric(
                row["val_ppl_mean"],
                row["val_ppl_std"],
                int(row["num_seeds"]),
                precision=2,
            ),
            axis=1,
        )
        aggregated["val_nll_display"] = aggregated.apply(
            lambda row: format_metric(
                row["val_nll_mean"], row["val_nll_std"], int(row["num_seeds"])
            ),
            axis=1,
        )
        aggregated["val_bpd_display"] = aggregated.apply(
            lambda row: format_metric(
                row["val_bpd_mean"], row["val_bpd_std"], int(row["num_seeds"])
            ),
            axis=1,
        )
        aggregated = aggregated.sort_values(
            [
                column
                for column in [
                    "data",
                    "val_ppl_mean",
                    "sampler_algo",
                    "model_prediction_algo",
                ]
                if column in aggregated.columns
            ],
            na_position="last",
        )

    aggregated.to_csv(leaderboard_agg_path, index=False)
    latex_table = build_latex_table(aggregated, run_dir.name)
    leaderboard_tex_path.write_text(latex_table)

    return (
        merged_success_path,
        leaderboard_path,
        leaderboard_agg_path,
        leaderboard_tex_path,
    )


def should_merge(run_dir: Path, merged_success: Path, required_outputs=None) -> bool:
    if not merged_success.exists():
        return True

    if required_outputs is not None:
        for path in required_outputs:
            if not Path(path).exists():
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
        / "run_ppl_manifest_slurm_array.sh"
    )

    cmd = [
        "sbatch",
        f"--array={array_range}",
        f"--gres=gpu:{int(cfg.resources.gpus_per_node)}",
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

    merged_success, leaderboard, leaderboard_agg, leaderboard_tex = merge_run(run_dir)

    print(f"Run dir: {run_dir}")
    print(f"Merged summary: {merged_success}")
    print(f"Leaderboard: {leaderboard}")
    print(f"Aggregated leaderboard: {leaderboard_agg}")
    print(f"LaTeX table: {leaderboard_tex}")
    if leaderboard_tex.exists():
        print()
        print(leaderboard_tex.read_text())


def parse_args():
    parser = argparse.ArgumentParser(description="Validation PPL evaluation workflow.")
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

    finalize = sub.add_parser("finalize", help="Merge run outputs into CSV summaries.")
    finalize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize.add_argument("--run-name", required=True)
    finalize.add_argument("--force-merge", action="store_true")
    finalize.set_defaults(func=cmd_finalize)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
