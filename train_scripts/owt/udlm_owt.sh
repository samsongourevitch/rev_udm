#!/usr/bin/env bash
#SBATCH --job-name=udlm_owt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --exclude=auh7-4b-gpu-149,auh7-4b-gpu-148,auh7-3b-gpu-149,auh7-3b-gpu-148,auh7-3b-gpu-134,auh7-3b-gpu-133,auh7-3b-gpu-308,auh7-3b-gpu-309,auh7-4b-gpu-133,auh7-4b-gpu-134,auh7-3b-gpu-204,auh7-3b-gpu-205,auh7-3b-gpu-091,auh7-3b-gpu-092

NUM_NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}
GLOBAL_BATCH_SIZE=512

# Set proxy environment variables here if your cluster requires them.

source $HOME/venvs/revisited_udm/bin/activate

RUN=/mnt/vast01/users/yazid.janati/projects/duo/outputs/openwebtext-train/2026.03.25/182558/
WANDB_RUN=udlm-owt-20260325_182550
WANDB_ID=udlm-owt-20260325_182550_1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

srun python -u -m trainer \
  trainer.num_nodes=${NUM_NODES} \
  trainer.devices=${GPUS_PER_NODE} \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
  mode=train \
  algo=udlm \
  model=small \
  model.length=1024 \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  wandb.name=udlm-owt-$(date +%Y%m%d_%H%M%S) \
  algo.model_prediction=mean_loo \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  hydra.run.dir=$RUN \
  checkpointing.save_dir=$RUN \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path=$RUN/checkpoints/last.ckpt \
  wandb.name=$WANDB_RUN \
  wandb.id=$WANDB_ID
