#!/bin/bash
# sbatch --account=def-mahyarh_cpu --cpus-per-task=1 --mem=6G --time=00:20:00 narval.sh <sample-directory>
set -eo pipefail
run=${1:?Missing the directory containing the 30-document inputs}
source /scratch/matroid/clir-benchmark/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TORCH=0
export PYTHONPATH="$run/project/src:$run/python-deps"
cd "$run/project"
exec python report/baseline_analysis/question_15_shared_chunks/pilot.py "$run"
