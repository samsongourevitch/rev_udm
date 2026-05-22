#!/usr/bin/env python3
import csv
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra

TEMPERATURE_SWEEP_DIR = Path(__file__).resolve().parents[1] / "temperature_sweep"
if str(TEMPERATURE_SWEEP_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPERATURE_SWEEP_DIR))

from nucleus_sweep import nucleus_tag
from temperature_sweep import (
    checkpoint_path_override,
    entry_dir_name,
    hydra_quote,
    infer_train_loss,
    normalize_checkpoint_path,
    read_jsonl,
    slug,
    temp_tag,
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


def allocated_gpu_tokens(cfg):
    for env_key in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = os.environ.get(env_key)
        if value:
            tokens = [token.strip() for token in value.split(",") if token.strip()]
            if tokens:
                return tokens
    return [str(index) for index in range(int(cfg.resources.gpus_per_node))]


def visible_devices_string(primary_token: str, all_tokens):
    ordered = [str(primary_token)]
    ordered.extend(str(token) for token in all_tokens if str(token) != str(primary_token))
    return ",".join(ordered)


def run_single_eval(
    cfg,
    manifest_idx: int,
    manifest_id: str,
    sampler: str,
    checkpoint_path: str,
    train_loss: str,
    model_prediction,
    temperature: float,
    p_nucleus: float,
    apply_temp_output: bool,
    nfe: int,
    seed: int,
    gpu_token: str,
    all_gpu_tokens,
    eval_batch_size: int,
    num_sample_batches: int,
    compute_mauve: bool,
    disable_wandb: bool,
    extra_overrides,
    raw_dir: Path,
    records_dir: Path,
):
    sample_filename = (
        f"samples_id-{slug(manifest_id)}_{sampler}_loss-{train_loss}_"
        f"nfe{nfe}_pnucleus{nucleus_tag(p_nucleus)}_temp{temp_tag(temperature)}_"
        f"seed{seed}.json"
    )
    sample_path = raw_dir / sample_filename

    cmd = [
        sys.executable,
        "trainer.py",
        f"algo={sampler}",
        "mode=sample_eval",
        f"seed={seed}",
        "data=openwebtext-split",
        "model.length=1024",
        checkpoint_path_override(checkpoint_path),
        f"eval.generated_samples_path={hydra_quote(sample_path)}",
        f"loader.eval_batch_size={eval_batch_size}",
        f"sampling.num_sample_batches={num_sample_batches}",
        f"sampling.nfes={nfe}",
        f"sampling.temperature={temperature}",
        f"sampling.p_nucleus={p_nucleus}",
        f"sampling.apply_temp_output={str(bool(apply_temp_output)).lower()}",
        f"eval.compute_mauve={str(compute_mauve).lower()}",
        "trainer.devices=1",
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices_string(gpu_token, all_gpu_tokens)

    start_time = time.perf_counter()
    completed = subprocess.run(cmd, env=env)
    runtime_sec = time.perf_counter() - start_time

    generative_ppl = None
    entropy = None
    mauve = None
    num_generated = None
    if completed.returncode == 0 and sample_path.exists():
        with sample_path.open("r") as file_obj:
            payload = json.load(file_obj)
        generative_ppl = payload.get("generative_ppl")
        entropy = payload.get("entropy")
        mauve = payload.get("mauve")
        num_generated = len(payload.get("generated_seqs", []))

    record = {
        "manifest_idx": manifest_idx,
        "manifest_id": manifest_id,
        "sampler_algo": sampler,
        "model_prediction_algo": model_prediction,
        "checkpoint_path": checkpoint_path,
        "train_loss_algo": train_loss,
        "nfe": int(nfe),
        "temperature": float(temperature),
        "p_nucleus": float(p_nucleus),
        "apply_temp_output": bool(apply_temp_output),
        "seed": int(seed),
        "gpu_id": str(gpu_token),
        "runtime_sec": float(runtime_sec),
        "status": "success" if completed.returncode == 0 else "failed",
        "return_code": int(completed.returncode),
        "entropy": entropy,
        "generative_ppl": generative_ppl,
        "mauve": mauve,
        "num_generated": num_generated,
        "sample_file": str(sample_path),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    record_filename = (
        f"record_id-{slug(manifest_id)}_{sampler}_loss-{train_loss}_"
        f"nfe{nfe}_pnucleus{nucleus_tag(p_nucleus)}_temp{temp_tag(temperature)}_"
        f"seed{seed}.json"
    )
    with (records_dir / record_filename).open("w") as file_obj:
        json.dump(record, file_obj, indent=2)

    return record


def run_gpu_worker(
    cfg,
    manifest_idx: int,
    manifest_id: str,
    sampler: str,
    checkpoint_path: str,
    train_loss: str,
    model_prediction,
    temperature: float,
    p_nucleus: float,
    apply_temp_output: bool,
    seeds,
    nfe: int,
    gpu_token: str,
    all_gpu_tokens,
    eval_batch_size: int,
    num_sample_batches: int,
    compute_mauve: bool,
    disable_wandb: bool,
    extra_overrides,
    raw_dir: Path,
    records_dir: Path,
):
    rows = []
    for seed in seeds:
        rows.append(
            run_single_eval(
                cfg=cfg,
                manifest_idx=manifest_idx,
                manifest_id=manifest_id,
                sampler=sampler,
                checkpoint_path=checkpoint_path,
                train_loss=train_loss,
                model_prediction=model_prediction,
                temperature=temperature,
                p_nucleus=p_nucleus,
                apply_temp_output=apply_temp_output,
                nfe=nfe,
                seed=int(seed),
                gpu_token=gpu_token,
                all_gpu_tokens=all_gpu_tokens,
                eval_batch_size=eval_batch_size,
                num_sample_batches=num_sample_batches,
                compute_mauve=compute_mauve,
                disable_wandb=disable_wandb,
                extra_overrides=extra_overrides,
                raw_dir=raw_dir,
                records_dir=records_dir,
            )
        )
    return rows


def save_summary(summary_path: Path, rows):
    if not rows:
        return
    fieldnames = [
        "manifest_idx",
        "manifest_id",
        "sampler_algo",
        "model_prediction_algo",
        "checkpoint_path",
        "train_loss_algo",
        "nfe",
        "temperature",
        "p_nucleus",
        "apply_temp_output",
        "seed",
        "gpu_id",
        "runtime_sec",
        "status",
        "return_code",
        "entropy",
        "generative_ppl",
        "mauve",
        "num_generated",
        "sample_file",
        "hostname",
        "slurm_job_id",
        "slurm_array_task_id",
    ]
    with summary_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["nfe"], item["seed"])):
            writer.writerow(row)


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


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="nucleus_eval_manifest")
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
    sampler = str(row["sampler"])
    checkpoint_path = normalize_checkpoint_path(str(row["checkpoint_path"]))
    train_loss = str(row.get("train_loss", "auto"))
    if train_loss == "auto":
        train_loss = infer_train_loss(checkpoint_path)

    temperature = float(row.get("temperature", cfg.sweep.temperature))
    p_nucleus = float(row["p_nucleus"])
    seeds = [int(seed) for seed in row.get("seeds", cfg.sweep.seeds)]
    nfes = [int(step) for step in row.get("nfes", cfg.sweep.nfe_list)]
    model_prediction = row.get("model_prediction")
    apply_temp_output = bool(row.get("apply_temp_output", cfg.sweep.apply_temp_output))
    eval_batch_size = int(row.get("eval_batch_size", cfg.sweep.eval_batch_size))
    num_sample_batches = int(row.get("num_sample_batches", cfg.sweep.num_sample_batches))
    compute_mauve = bool(row.get("compute_mauve", cfg.sweep.compute_mauve))
    disable_wandb = bool(row.get("disable_wandb", cfg.sweep.disable_wandb))
    extra_overrides = row.get("extra_overrides", [])

    run_dir = Path(cfg.run.output_root) / str(cfg.run.name)
    entry_dir = run_dir / entry_dir_name(manifest_id)
    raw_dir = entry_dir / "raw"
    records_dir = entry_dir / "records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    gpu_tokens = allocated_gpu_tokens(cfg)
    if not gpu_tokens:
        raise ValueError("No GPUs available for nucleus sweep execution")

    all_rows = []
    for batch_start in range(0, len(nfes), len(gpu_tokens)):
        nfe_batch = nfes[batch_start : batch_start + len(gpu_tokens)]
        with ThreadPoolExecutor(max_workers=len(nfe_batch)) as executor:
            futures = []
            for gpu_idx, nfe in enumerate(nfe_batch):
                futures.append(
                    executor.submit(
                        run_gpu_worker,
                        cfg,
                        manifest_idx,
                        manifest_id,
                        sampler,
                        checkpoint_path,
                        train_loss,
                        model_prediction,
                        temperature,
                        p_nucleus,
                        apply_temp_output,
                        seeds,
                        int(nfe),
                        gpu_tokens[gpu_idx],
                        gpu_tokens,
                        eval_batch_size,
                        num_sample_batches,
                        compute_mauve,
                        disable_wandb,
                        extra_overrides,
                        raw_dir,
                        records_dir,
                    )
                )
            for future in as_completed(futures):
                all_rows.extend(future.result())

    save_summary(entry_dir / "summary.csv", all_rows)
    expected_count = len(seeds) * len(nfes)
    complete = write_status_files(entry_dir, all_rows, expected_count)
    print(f"Saved entry results to: {entry_dir}")
    if not complete:
        raise SystemExit(
            f"Incomplete run for {manifest_id}: expected {expected_count} results, got {len(all_rows)}"
        )


if __name__ == "__main__":
    main()
