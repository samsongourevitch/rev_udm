#!/usr/bin/env python3
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import hydra

TEMPERATURE_SWEEP_DIR = Path(__file__).resolve().parents[1] / "temperature_sweep"
if str(TEMPERATURE_SWEEP_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPERATURE_SWEEP_DIR))

from temperature_sweep import (
    checkpoint_path_override,
    entry_dir_name,
    hydra_quote,
    infer_train_loss,
    normalize_checkpoint_path,
    read_jsonl,
    slug,
)


CONFIG_DIR = str((Path(__file__).resolve().parents[2] / "configs" / "sweeps"))


def load_manifest(manifest_path: Path):
    rows = read_jsonl(manifest_path)
    for index, row in enumerate(rows):
        row["_manifest_idx"] = index
    return rows


def resolve_manifest_idx(cfg, total_rows: int) -> int:
    if cfg.launcher.manifest_idx >= 0:
        idx = int(cfg.launcher.manifest_idx)
    else:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        idx = int(task_id) if task_id is not None else 0
    if idx < 0 or idx >= total_rows:
        raise ValueError(f"Invalid manifest index {idx} for {total_rows} rows")
    return idx


def visible_gpu_count(cfg) -> int:
    for env_key in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = os.environ.get(env_key)
        if value:
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            if tokens:
                return len(tokens)
    return int(cfg.resources.gpus_per_node)


def run_single_eval(
    cfg,
    manifest_idx: int,
    manifest_id: str,
    entry_id: str,
    sampler: str,
    checkpoint_path: str,
    train_loss: str,
    model_prediction,
    seed: int,
    data_name: str,
    model_length: int,
    eval_batch_size: int,
    disable_ema: bool,
    disable_wandb: bool,
    extra_overrides,
    raw_dir: Path,
    records_dir: Path,
):
    metrics_filename = (
        f"metrics_dataset-{slug(manifest_id)}_entry-{slug(entry_id)}_seed{seed}.json"
    )
    metrics_path = raw_dir / metrics_filename
    num_devices = visible_gpu_count(cfg)

    cmd = [
        sys.executable,
        "trainer.py",
        f"algo={sampler}",
        "mode=ppl_eval",
        f"seed={seed}",
        f"data={data_name}",
        f"model.length={model_length}",
        checkpoint_path_override(checkpoint_path),
        f"eval.disable_ema={str(bool(disable_ema)).lower()}",
        "eval.generate_samples=false",
        "eval.compute_generative_perplexity=false",
        f"eval.metrics_output_path={hydra_quote(metrics_path)}",
        f"loader.eval_batch_size={eval_batch_size}",
        f"trainer.devices={num_devices}",
        "trainer.num_nodes=1",
        "hydra.run.dir=.",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
    ]

    if model_prediction is not None:
        cmd.append(f"++algo.model_prediction={model_prediction}")
    if disable_wandb:
        cmd.append("wandb=null")
    if extra_overrides:
        cmd.extend([str(item) for item in extra_overrides])

    start_time = time.perf_counter()
    completed = subprocess.run(cmd, env=os.environ.copy())
    runtime_sec = time.perf_counter() - start_time

    val_nll = None
    val_bpd = None
    val_ppl = None
    if completed.returncode == 0 and metrics_path.exists():
        with metrics_path.open("r") as file_obj:
            payload = json.load(file_obj)
        val_nll = payload.get("val/nll")
        val_bpd = payload.get("val/bpd")
        val_ppl = payload.get("val/ppl")

    record = {
        "manifest_idx": manifest_idx,
        "manifest_id": manifest_id,
        "entry_id": entry_id,
        "sampler_algo": sampler,
        "model_prediction_algo": model_prediction,
        "checkpoint_path": checkpoint_path,
        "train_loss_algo": train_loss,
        "data": data_name,
        "model_length": int(model_length),
        "eval_batch_size": int(eval_batch_size),
        "disable_ema": bool(disable_ema),
        "seed": int(seed),
        "runtime_sec": float(runtime_sec),
        "status": "success" if completed.returncode == 0 else "failed",
        "return_code": int(completed.returncode),
        "val_nll": val_nll,
        "val_bpd": val_bpd,
        "val_ppl": val_ppl,
        "metrics_file": str(metrics_path),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    record_filename = (
        f"record_dataset-{slug(manifest_id)}_entry-{slug(entry_id)}_seed{seed}.json"
    )
    with (records_dir / record_filename).open("w") as file_obj:
        json.dump(record, file_obj, indent=2)

    return record


def save_summary(summary_path: Path, rows):
    if not rows:
        return
    fieldnames = [
        "manifest_idx",
        "manifest_id",
        "entry_id",
        "sampler_algo",
        "model_prediction_algo",
        "checkpoint_path",
        "train_loss_algo",
        "data",
        "model_length",
        "eval_batch_size",
        "disable_ema",
        "seed",
        "runtime_sec",
        "status",
        "return_code",
        "val_nll",
        "val_bpd",
        "val_ppl",
        "metrics_file",
        "hostname",
        "slurm_job_id",
        "slurm_array_task_id",
    ]
    with summary_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item.get("data", ""),
                item.get("entry_id", ""),
                item["seed"],
            ),
        ):
            writer.writerow(row)


def normalize_entries(row, cfg):
    entries = row.get("entries")
    if entries is not None:
        return list(entries)

    train_loss = str(row.get("train_loss", "auto"))
    checkpoint_path = normalize_checkpoint_path(str(row["checkpoint_path"]))
    if train_loss == "auto":
        train_loss = infer_train_loss(checkpoint_path)

    single_entry = {
        "entry_id": str(
            row.get(
                "entry_id",
                f"entry00_{row['sampler']}_loss-{train_loss}",
            )
        ),
        "sampler": str(row["sampler"]),
        "checkpoint_path": normalize_checkpoint_path(checkpoint_path),
        "train_loss": train_loss,
        "seeds": [int(seed) for seed in row.get("seeds", cfg.sweep.seeds)],
    }
    if row.get("model_prediction") is not None:
        single_entry["model_prediction"] = row.get("model_prediction")
    if row.get("extra_overrides") is not None:
        single_entry["extra_overrides"] = row.get("extra_overrides")
    return [single_entry]


def write_status_files(entry_dir: Path, rows, expected_count: int):
    success_count = sum(1 for row in rows if row.get("status") == "success")
    failed_count = sum(1 for row in rows if row.get("status") != "success")
    complete = len(rows) == expected_count and success_count == expected_count

    status_payload = {
        "complete": complete,
        "expected": int(expected_count),
        "produced": int(len(rows)),
        "success": int(success_count),
        "failed": int(failed_count),
    }
    with (entry_dir / "status.json").open("w") as file_obj:
        json.dump(status_payload, file_obj, indent=2)

    ok_path = entry_dir / "OK"
    if complete:
        ok_path.write_text("ok\n")
    elif ok_path.exists():
        ok_path.unlink()

    return complete


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="ppl_eval_manifest")
def main(cfg):
    manifest_path = Path(str(cfg.manifest.path))
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    manifest_rows = load_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    manifest_idx = resolve_manifest_idx(cfg, len(manifest_rows))
    row = manifest_rows[manifest_idx]

    manifest_id = str(row.get("id", f"manifest_{manifest_idx:05d}"))
    data_name = str(row.get("data", cfg.sweep.data))
    model_length = int(row.get("model_length", cfg.sweep.model_length))
    eval_batch_size = int(row.get("eval_batch_size", cfg.sweep.eval_batch_size))
    disable_ema = bool(row.get("disable_ema", cfg.sweep.disable_ema))
    disable_wandb = bool(row.get("disable_wandb", cfg.sweep.disable_wandb))
    entries = normalize_entries(row, cfg)

    run_dir = Path(cfg.run.output_root) / str(cfg.run.name)
    entry_dir = run_dir / entry_dir_name(manifest_id)
    raw_dir = entry_dir / "raw"
    records_dir = entry_dir / "records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for entry in entries:
        sampler = str(entry["sampler"])
        checkpoint_path = str(entry["checkpoint_path"])
        train_loss = str(entry.get("train_loss", "auto"))
        if train_loss == "auto":
            train_loss = infer_train_loss(checkpoint_path)
        model_prediction = entry.get("model_prediction")
        extra_overrides = entry.get("extra_overrides", [])
        entry_id = str(
            entry.get(
                "entry_id",
                f"{sampler}_loss-{train_loss}",
            )
        )
        seeds = [int(seed) for seed in entry.get("seeds", cfg.sweep.seeds)]

        for seed in seeds:
            all_rows.append(
                run_single_eval(
                    cfg=cfg,
                    manifest_idx=manifest_idx,
                    manifest_id=manifest_id,
                    entry_id=entry_id,
                    sampler=sampler,
                    checkpoint_path=checkpoint_path,
                    train_loss=train_loss,
                    model_prediction=model_prediction,
                    seed=int(seed),
                    data_name=data_name,
                    model_length=model_length,
                    eval_batch_size=eval_batch_size,
                    disable_ema=disable_ema,
                    disable_wandb=disable_wandb,
                    extra_overrides=extra_overrides,
                    raw_dir=raw_dir,
                    records_dir=records_dir,
                )
            )

    save_summary(entry_dir / "summary.csv", all_rows)
    expected_count = sum(
        len([int(seed) for seed in entry.get("seeds", cfg.sweep.seeds)])
        for entry in entries
    )
    complete = write_status_files(entry_dir, all_rows, expected_count)
    print(f"Saved entry results to: {entry_dir}")
    if not complete:
        raise SystemExit(
            f"Incomplete run for {manifest_id}: expected {expected_count} results, got {len(all_rows)}"
        )


if __name__ == "__main__":
    main()
