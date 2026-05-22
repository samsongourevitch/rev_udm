# Revisited Uniform Diffusion Models 

This repository contains the code for the paper *Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation*.

The main point of the paper is that the usual plug-in parameterization of Uniform Diffusion Models (UDMs) does not learn the standard denoiser. It learns a leave-one-out (LOO) denoiser instead. We give exact formulas to convert between the standard denoiser, the LOO denoiser, and the score, so the two parameterizations can be used interchangeably at training time or at inference time. In practice, the LOO parameterization gives better results and also leads to useful inference tools such as predictor-corrector sampling.

All models were trained from scratch on LM1B and OpenWebText. Checkpoints are available [here](https://drive.google.com/drive/folders/12M4ADxzwLLZNy2qbflids6aLDAqnYtUl?usp=share_link).

## Structure

- `trainer.py`: main text training and evaluation entrypoint
- `trainer_sudoku.py`: separate entrypoint for Sudoku experiments
- `src/`: models, Lightning modules, dataloaders, metrics, and sampling code
- `configs/`: Hydra configs for data, models, algorithms, training, and sweep manifests
- `train_scripts/`: example Slurm launch scripts for LM1B and OpenWebText runs
- `eval_scripts/`: sweep helpers for sample eval and validation PPL
- `samples/`: default output folder for evaluation sweeps
- `outputs/`: default Hydra output folder for training runs

## Setup

The repo provides a pyproject.toml for installation. You can use `uv` for instance:

```bash
uv sync
uv pip install -e .
```

## Environment

`torch` and `flash_attn` are intentionally not pinned here because the correct install depends on your CUDA / ROCm / driver / GPU setup.

## Training

Text runs use `trainer.py` with Hydra. The default text config is `configs/config.yaml`; Sudoku has its own entrypoint and config in `trainer_sudoku.py` and `configs/sudoku_config.yaml`.

Main configs to change:

- `configs/config.yaml`: global defaults for text runs
- `configs/data/`: dataset choice and paths
- `configs/algo/`: training objective / sampler variant
- `configs/model/`: model size and architecture

Example:

```bash
python -m trainer \
  mode=train \
  data=openwebtext-split \
  algo=udlm \
  model=small \
  model.length=1024
```

For cluster runs, you can use `train_scripts/lm1b/` and `train_scripts/owt/`, for example:

```bash
sbatch train_scripts/lm1b/udlm.sh
```

By default, each text training run writes to `outputs/<data.train>/<date>/<time>/`, typically with:

```text
outputs/.../<run>/
  checkpoints/
    best.ckpt
    last.ckpt
  .hydra/
```

`last.ckpt` is the default resume target when `checkpointing.resume_from_ckpt=true`.

## Eval

Single-checkpoint eval still goes through `trainer.py`:

```bash
python -m trainer \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/last.ckpt \
  sampling.nfes=256 \
  sampling.temperature=1.0

python -m trainer \
  mode=ppl_eval \
  eval.checkpoint_path=/path/to/last.ckpt \
  data=openwebtext-split \
  eval.metrics_output_path=metrics.json
```

### LOO probe

As explain in Appendix H of the paper, `loo_sensitivity_probe.py` measures leave-one-out sensitivity along the ancestral sampling trajectory, which is a proxy of the quality of a checkpoint and saves the curve, step trace, and decoded samples under `--output-dir/<timestamp>/`.

```bash
python loo_sensitivity_probe.py --checkpoint-path /path/to/last.ckpt --algo udlm --model_prediction mean_loo
```

Sweep configs live in `configs/sweeps/`. In all sweep manifests:

- `entries:` is the list of checkpoints / methods to evaluate
- `sweep:` defines the grid to run
- `run.output_root:` controls where outputs are written

The sweep helpers all follow the same pattern: `prepare`, `launch`, `finalize`. `launch` submits a Slurm array; if needed, the bash helpers in `eval_scripts/bash_scripts/` show the non-Slurm/manual pattern.

After `prepare`, you can run the pending manifest locally without Slurm:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash eval_scripts/bash_scripts/run_sweep_wo_slurm.sh temperature my_run
CUDA_VISIBLE_DEVICES=0,1 bash eval_scripts/bash_scripts/run_sweep_wo_slurm.sh nucleus my_run
CUDA_VISIBLE_DEVICES=0,1 bash eval_scripts/bash_scripts/run_sweep_wo_slurm.sh ppl my_run
```

### Generative frontiers

Generative frontiers follow the evaluation from Pynadath, Shi, and Zhang, [Generative Frontiers: Why Evaluation Matters for Diffusion Language Models](https://arxiv.org/abs/2604.02718). The goal is to compare methods over a tradeoff curve rather than at a single sampling setting which can lead to misleading conclusions.

Each point on the curve evaluates generated samples at a fixed NFE and seed, then records:

- `entropy`: sample diversity
- `generative_ppl`: sample quality under an external language model
- `mauve`: optional distributional similarity metric

The frontier is obtained by varying either temperature or top-p (nucleus) sampling.

#### Temperature sweeps

Temperature sweeps vary `sampling.temperature`. Edit `configs/sweeps/temp_eval_manifest.yaml`, especially:

- `entries`
- `sweep.temperatures`
- `sweep.nfe_list`
- `sweep.seeds`
- `sweep.num_sample_batches`
- `sweep.compute_mauve`

```bash
python eval_scripts/temperature_sweep/temperature_sweep.py prepare --run-name my_run
python eval_scripts/temperature_sweep/temperature_sweep.py launch --run-name my_run
python eval_scripts/temperature_sweep/temperature_sweep.py finalize --run-name my_run
```

Outputs go to `samples/temp_sweep/<run-name>/`:

```text
samples/temp_sweep/<run-name>/
  manifest.jsonl
  pending_manifest.jsonl
  entry*/raw/
  entry*/records/
  entry*/status.json
  entry*/OK
  merged_success.csv
  frontier/
```

`prepare` creates `manifest.jsonl` and filters unfinished work into `pending_manifest.jsonl`. `launch` submits only pending rows. `finalize` merges successful records into `merged_success.csv` and writes frontier plots under `frontier/`.

To resume unfinished jobs, rerun:

```bash
python eval_scripts/temperature_sweep/temperature_sweep.py prepare --run-name my_run
python eval_scripts/temperature_sweep/temperature_sweep.py launch --run-name my_run
```

Useful flags:

- `--refresh-manifest`: rebuild `manifest.jsonl` from the YAML config
- `--repair-ok`: recreate missing `OK` files for entries already marked complete in `status.json`
- `--config`: use a different sweep config

#### Nucleus sweeps

Nucleus sweeps use the same workflow, but vary `sampling.p_nucleus` at fixed `sampling.temperature`. Edit `configs/sweeps/nucleus_eval_manifest.yaml`, especially:

- `entries`
- `sweep.p_nucleus_list`
- `sweep.nfe_list`
- `sweep.temperature`

Example:

```bash
python eval_scripts/nucleus_sweep/nucleus_sweep.py prepare --run-name my_nucleus_run
python eval_scripts/nucleus_sweep/nucleus_sweep.py launch --run-name my_nucleus_run
python eval_scripts/nucleus_sweep/nucleus_sweep.py finalize --run-name my_nucleus_run
```

### PPL evaluation

PPL evaluation benchmarks validation perplexity of trained checkpoints on one or more datasets. Unlike the frontier sweeps, it does not generate samples: it calls `trainer.py mode=ppl_eval`, runs the validation loader, and records `val/nll`, `val/bpd`, and `val/ppl`.

Edit `configs/sweeps/ppl_eval_manifest.yaml`, especially:

- `entries`: checkpoints / methods to evaluate
- `sweep.datasets`: datasets to benchmark on
- `sweep.model_length`
- `sweep.eval_batch_size`
- `sweep.seeds`

Example:

```bash
python eval_scripts/ppl_eval/ppl_eval.py prepare --run-name my_ppl_run
python eval_scripts/ppl_eval/ppl_eval.py launch --run-name my_ppl_run
python eval_scripts/ppl_eval/ppl_eval.py finalize --run-name my_ppl_run
```

## Citation

If you use this code, please cite:

```bibtex

```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Credits

- The code structure is inspired by [MDLM](https://github.com/kuleshov-group/mdlm).
