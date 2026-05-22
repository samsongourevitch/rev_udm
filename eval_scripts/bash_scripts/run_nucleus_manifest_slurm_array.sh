#!/usr/bin/env bash
#SBATCH --job-name=nucleus-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0
#SBATCH --output=outputs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm/slurm-%x-%A_%a.err

set -euo pipefail

# Set proxy environment variables here if your cluster requires them.

. $HOME/venvs/revisited_udm/bin/activate 
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

python3 "${REPO_ROOT}/eval_scripts/run_nucleus_manifest.py" "$@"
