#!/usr/bin/env bash
# Keep plotting dependencies separate from the pinned GPU inference environment.
set -eo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
base=${SCRATCH:?}/clir-benchmark
if [[ ${1:-} == figure ]]; then
    module load scipy-stack/2025a
    report_dir="$project/report/baseline_analysis/question_21_retrieval_tradeoffs"
    export MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/clir-rerank-mpl-${SLURM_JOB_ID:-$$}"
    exec python "$report_dir/reranker_table.py" "$project/.."
fi
source "$base/env.sh"
export PYTHONPATH="$project/src:$base/runs/narval-20261119-shared-indexes/python-deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$project"
exec "$base/venv/bin/python" "$project/scripts/submit_verified_rerank.py" "$@"
