#!/usr/bin/env bash
#SBATCH --job-name=ppl-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0
#SBATCH --output=outputs/slurm/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm/slurm-%x-%A_%a.err

set -euo pipefail

# Set proxy environment variables here if your cluster requires them.

. $HOME/venvs/revisited_udm/bin/activate
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

python3 "${REPO_ROOT}/eval_scripts/run_ppl_manifest.py" "$@"
