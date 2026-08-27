#!/bin/bash
#SBATCH --job-name=struct_merge_plot
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${CONFIG:-${ROOT}/config.yaml}
PYTHON=${PYTHON:-python}
cd "${ROOT}"
"${PYTHON}" -m src.merge_results --config "${CONFIG}"
"${PYTHON}" -m src.plot_results --config "${CONFIG}"

