#!/bin/bash
#SBATCH --job-name=struct_metrics
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
CONFIG=${CONFIG:-${ROOT}/config.yaml}
PYTHON=${PYTHON:-python}
test -n "${SLURM_ARRAY_TASK_ID:-}" || { echo "Submit this script as a Slurm array" >&2; exit 1; }
cd "${ROOT}"
"${PYTHON}" -m src.compute_metrics --config "${CONFIG}" --chunk-id "${SLURM_ARRAY_TASK_ID}"

