#!/usr/bin/env bash
# "prepare" builds judgments only. "retrieve" requires a validated preparation
# receipt for each requested corpus and submits original-question controls only.
set -euo pipefail
base=${SCRATCH:?}/clir-benchmark
run=${CLIR_RETRIEVAL_RUN:-$base/runs/narval-20261120-verified-retrieval}
project=$run/project
script=$project/scripts/submit_verified_retrieval.sh
config=$project/configs/verified-20261120-retrieval.yaml
state=$run/job_status
mkdir -p "$state" "$run/logs"
fingerprint=$(sha256sum "$config" | cut -c1-16)
worker() {
    local stage=$1 corpus=$2 model=$3 key
    key=$stage-$corpus
    if [[ -n $model ]]; then key=$key-$model; fi
    local receipt=$state/$key-$fingerprint.done
    exec 8>"$state/$key.lock"
    flock 8
    if [[ -s $receipt ]]; then return; fi
    set +u
    source "$base/env.sh"
    set -u
    export PYTHONPATH="$project/src:$base/runs/narval-20261119-shared-indexes/python-deps"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export SQLITE_TMPDIR="${SLURM_TMPDIR:?}" PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    cd "$project"
    local -a cli=("$base/venv/bin/python" -m clir_benchmark --config "$config") selector=(--corpus "$corpus")
    if [[ -n $model ]]; then selector+=(--model "$model"); fi
    case $stage in
        prepare)
            "${cli[@]}" prepare-relevance "${selector[@]}" --work-dir "$SLURM_TMPDIR"
            "$base/venv/bin/python" scripts/check_verified_retrieval_inputs.py --config "$config" --corpus "$corpus" > "$state/check-$corpus.tmp"
            mv "$state/check-$corpus.tmp" "$state/check-$corpus.json"
            ;;
        embed) "${cli[@]}" embed --kind query "${selector[@]}" ;;
        bm25|dense|hybrid)
            "${cli[@]}" retrieve --baseline "$stage" "${selector[@]}"
            "${cli[@]}" evaluate --baseline "$stage" "${selector[@]}"
            ;;
        *) exit 2 ;;
    esac
    printf '%s\n' "$SLURM_JOB_ID" > "$receipt.tmp"
    mv "$receipt.tmp" "$receipt"
}
if [[ ${1:-} == --worker ]]; then shift; worker "$@"; exit; fi
mode=${1:-}
case $mode in prepare|retrieve) ;; *) echo "Usage: $0 {prepare|retrieve} [chemistry|eurlex|un]..." >&2; exit 2;; esac
shift
corpora=("$@")
if (( ! ${#corpora[@]} )); then corpora=(chemistry eurlex un); fi
for c in "${corpora[@]}"; do
    case $c in chemistry|eurlex|un) ;; *) echo "Unknown corpus $c" >&2; exit 2;; esac
    if [[ $mode == retrieve && ! -s $state/prepare-$c-$fingerprint.done ]]; then
        echo "Missing successful prepare-relevance receipt for $c" >&2; exit 1
    fi
done
exec 9>"$state/submit.lock"
flock 9
models=(embeddinggemma bge-m3 qwen3-0.6B nomic-v2-moe granite-278m gte-multilingual-base arctic-embed-l-v2 e5-large-instruct)
submit() {
    local stage=$1 corpus=$2 model=$3 dep=$4 key name id status
    key=$stage-$corpus
    if [[ -n $model ]]; then key=$key-$model; fi
    local marker=$state/$key-$fingerprint
    JOB_ID=""
    if [[ -s $marker.done ]]; then return; fi
    if [[ -s $marker.job ]]; then
        read -r id < "$marker.job"
        status=$(squeue -h -j "$id" -o "%T" 2>/dev/null || true)
        if [[ -z $status ]]; then status=$(sacct -XnP -j "$id" -o JobIDRaw,State | awk -F'|' -v id="$id" '$1==id {print $2;exit}'); fi
        case $status in
            PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|SUSPENDED) JOB_ID=$id; return ;;
            "") echo "Scheduler state for previous job $id unavailable" >&2; exit 1 ;;
        esac
        if [[ -s $marker.done ]]; then return; fi
    fi
    name=clir-search-$key
    local -a spec=(--account=def-mahyarh_cpu)
    case $stage in
        prepare) spec+=(--cpus-per-task=2 --mem=24G --tmp=160G --time=1-00:00:00) ;;
        embed) spec=(--account=def-mahyarh_gpu --gpus=a100:1 --cpus-per-task=4 --mem=32G --time=03:00:00) ;;
        bm25) spec+=(--cpus-per-task=1 --mem=32G --tmp=160G --time=7-00:00:00) ;;
        dense) spec+=(--cpus-per-task=8 --mem=48G --time=3-00:00:00) ;;
        hybrid) spec+=(--cpus-per-task=2 --mem=20G --time=12:00:00) ;;
    esac
    if [[ -n $dep ]]; then spec+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes); fi
    id=$(sbatch --parsable --job-name="$name" "${spec[@]}" --chdir="$project" --output="$run/logs/$key-%j.out" --error="$run/logs/$key-%j.err" "$script" --worker "$stage" "$corpus" "$model")
    JOB_ID=${id%%;*}
    [[ $JOB_ID =~ ^[0-9]+$ ]]
    printf '%s\n' "$JOB_ID" > "$marker.job"
    printf '%s\t%s\t%s\t%s\n' "$key" "$JOB_ID" "$dep" "$fingerprint" | tee -a "$state/submissions.tsv"
}
for corpus in "${corpora[@]}"; do
    if [[ $mode == prepare ]]; then submit prepare "$corpus" "" ""; continue; fi
    submit bm25 "$corpus" "" ""; bm25=$JOB_ID
    for model in "${models[@]}"; do
        submit embed "$corpus" "$model" ""
        submit dense "$corpus" "$model" "$JOB_ID"
        dep=${bm25:+$bm25:}$JOB_ID
        submit hybrid "$corpus" "$model" "${dep%:}"
    done
done
