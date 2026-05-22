#!/usr/bin/env bash
#SBATCH --job-name=mdlm_lm1b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --qos=ifm_others
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --exclude=auh7-4b-gpu-149,auh7-4b-gpu-148,auh7-3b-gpu-149,auh7-3b-gpu-148,auh7-3b-gpu-134,auh7-3b-gpu-133,auh7-3b-gpu-308,auh7-3b-gpu-309,auh7-4b-gpu-133,auh7-4b-gpu-134,auh7-3b-gpu-204,auh7-3b-gpu-205,auh7-3b-gpu-091,auh7-3b-gpu-092

NUM_NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}
GLOBAL_BATCH_SIZE=512

# Set proxy environment variables here if your cluster requires them.

source $HOME/venvs/revisited_udm/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

srun python -u -m trainer \
  trainer.num_nodes=${NUM_NODES} \
  trainer.devices=${GPUS_PER_NODE} \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  model=small \
  mode=train \
  data=lm1b-wrap \
  model.length=128 \
  wandb.name=mdlm-lm1b-$(date +%Y%m%d_%H%M%S) \
  algo=mdlm \
  ++algo.prediction=mean \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50000
