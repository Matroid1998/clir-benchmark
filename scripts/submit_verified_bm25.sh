#!/bin/bash
# From the campaign checkout on Narval: bash scripts/submit_verified_bm25.sh
set -eo pipefail

benchmark_root="${SCRATCH:?Set SCRATCH to the shared Narval scratch directory}/clir-benchmark"
run="$benchmark_root/runs/narval-20261119-verified-bm25"
project="$run/project"
config="$project/configs/verified-20261118-bm25.yaml"
state="$run/job_status"
config_hash=$(sha256sum "$config" | cut -d' ' -f1)

if [[ ${1:-} == --worker ]]; then
    corpus=${2:?Missing corpus}
    case "$corpus" in chemistry|eurlex|un) ;; *) exit 2 ;; esac
    cd "$project"
    source "$benchmark_root/env.sh"
    export PYTHONPATH="$project/src"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export SQLITE_TMPDIR="${SLURM_TMPDIR:?Missing node-local job directory}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python_bin="$benchmark_root/venv/bin/python"
    "$python_bin" -m clir_benchmark --config "$config" prepare-corpus --corpus "$corpus" \
        --work-dir "$SLURM_TMPDIR"
    index_path=$("$python_bin" scripts/build_verified_bm25_index.py \
        --config "$config" --corpus "$corpus" \
        --build-temp "$SQLITE_TMPDIR" \
        --prior-artifacts "$benchmark_root/runs/fir-20261109-all-questions/artifacts")
    [[ -s "$index_path/manifest.json" && -s "$index_path/index.sqlite" ]]
    printf '%s\n' "$index_path" > "$state/$corpus-$config_hash.${SLURM_JOB_ID}.tmp"
    mv "$state/$corpus-$config_hash.${SLURM_JOB_ID}.tmp" "$state/$corpus-$config_hash.index"
    printf 'Finished %s: %s\n' "$corpus" "$index_path"
    exit 0
fi

if (( $# != 0 )); then
    echo "Usage: bash scripts/submit_verified_bm25.sh" >&2
    exit 2
fi
mkdir -p "$state" "$run/logs"
exec 9>"$state/submit.lock"
flock 9
for corpus in chemistry eurlex un; do
    receipt="$state/$corpus-$config_hash.index"
    if [[ -s "$receipt" ]]; then
        read -r index_path < "$receipt"
        if [[ -s "$index_path/manifest.json" && -s "$index_path/index.sqlite" ]]; then
            printf '%s already indexed: %s\n' "$corpus" "$index_path"
            continue
        fi
    fi
    job_name="clir-bm25-v20261118-$corpus"
    active=$(squeue --noheader --user="$USER" --name="$job_name" --format='%i')
    if [[ -n "$active" ]]; then
        printf '%s already queued: %s\n' "$corpus" "$active"
        continue
    fi
    job_id=$(sbatch --parsable --account=def-mahyarh_cpu --job-name="$job_name" \
        --cpus-per-task=1 --mem=16G --tmp=32G --time=24:00:00 \
        --chdir="$project" --output="$run/logs/$corpus-%j.out" --error="$run/logs/$corpus-%j.err" \
        "$project/scripts/submit_verified_bm25.sh" --worker "$corpus")
    printf '%s\t%s\t%s\n' "$corpus" "$job_id" "$config_hash" >> "$state/submissions.tsv"
    printf 'Submitted %s job %s\n' "$corpus" "$job_id"
done
