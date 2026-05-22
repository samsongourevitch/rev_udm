# script for debugging purposes

import json
import os
import subprocess
import sys
from pathlib import Path

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

models = {
    "udlm": [
        "algo=udlm",
        "eval.checkpoint_path='checkpoints/udlm_elbo_mean.ckpt'",
        "algo.model_prediction=mean",
    ],
    "mdlm": [
        "algo=mdlm",
        "eval.checkpoint_path='checkpoints/mdlm_elbo_mean.ckpt'",
    ],
    "audm": [
        "algo=audm",
        "++algo.conditioning_fusion=shared_mlp",
        "eval.checkpoint_path='checkpoints/audlm_elbo_mean.ckpt'",
        "algo.model_prediction=mean",
    ],
}


def run_sampler(model):
    cmd = [
        sys.executable,
        "trainer.py",
        *models[model],
        "eval.compute_generative_perplexity=True",
        "mode=sample_eval",
        # "sampling/corrector=pc_gibbs",
        # "sampling.corrector.selection=lowest_current_prob",
        "loader.eval_batch_size=2",
        "model=small",
        "data=lm1b",
        "model.length=128",
        "sampling.nfes=128",
        "sampling.temperature=1.0",
        "wandb=null",
        "hydra.run.dir=.",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "seed=1",
        "sampling.generator_seed=150",
        # "sampling.p_nucleus=0.98",
    ]

    subprocess.run(cmd, env=env)


run_sampler("udlm")
