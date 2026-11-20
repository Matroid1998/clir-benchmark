#!/usr/bin/env bash
# Shared runtime for offline checks, submission and Slurm workers.
set -eo pipefail
base=${SCRATCH:?}/clir-benchmark
project=$(cd "$(dirname "$0")/.." && pwd)
source "$base/env.sh"
export PYTHONPATH="$project/src:$base/runs/narval-20261119-shared-indexes/python-deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
if [[ -n ${SLURM_TMPDIR:-} ]]; then export SQLITE_TMPDIR="$SLURM_TMPDIR"; fi
cd "$project"
exec "$base/venv/bin/python" "$project/scripts/submit_transformed_retrieval.py" "$@"
