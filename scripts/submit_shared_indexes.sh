#!/bin/bash
# Run from the staged checkout on Narval. The default campaign fits the repo's
# project/, inputs/, artifacts/ layout. CLIR_SHARED_RUN can override its path.
# chunks | indexes | all | bm25 | dense [--corpus NAME]... [--model ALIAS]...
# indexes, bm25 and dense require a completed or queued chunk job per corpus.
# Use CLIR_INDEX_BUDGET_BYTES for the available campaign quota if below 2 TB.
set -eo pipefail

benchmark_root="${SCRATCH:?Set SCRATCH to the shared Narval scratch directory}/clir-benchmark"
run="${CLIR_SHARED_RUN:-$benchmark_root/runs/narval-20261119-shared-indexes}"
project="$run/project"
config="$project/configs/verified-20261118-shared-indexes.yaml"
state="$run/job_status"
script="$project/scripts/submit_shared_indexes.sh"
fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ $run == /* ]] || fail 'CLIR_SHARED_RUN must be an absolute path.'
source "$benchmark_root/env.sh"
set -u
export CLIR_SHARED_RUN="$run" PYTHONPATH="$project/src:$run/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin="$benchmark_root/venv/bin/python"
cli=("$python_bin" -m clir_benchmark --config "$config")
cd "$project"
# Include executable sources to keep job receipts and active job names isolated
# if a staged campaign changes. Complete immutable artifacts can still be reused.
identity=$(sha256sum "$config" "$script" scripts/build_verified_bm25_index.py src/clir_benchmark/{*.py,models.yaml} | sha256sum | cut -c1-16)
mkdir -p "$state" "$run/logs" "$run/artifacts"

receipt_for() { printf '%s/%s-%s.done\n' "$state" "$1" "$identity"; }
complete() {
    local key=$1 receipt dir spans
    receipt=$(receipt_for "$key")
    [[ -s $receipt ]] || return 1
    read -r dir < "$receipt"
    [[ -s $dir/manifest.json ]] || return 1
    case "$key" in
        chunks-*) spans=$(sed -n '2p' "$receipt"); [[ -s $dir/chunks.parquet && -s $dir/lookup.sqlite && -s $spans/manifest.json && -s $spans/span-map.jsonl && -s $spans/known-chunk-qrels.tsv ]] ;;
        bm25-*) [[ -s $dir/index.sqlite ]] ;;
        dense-*) [[ -s $dir/vectors.npy && -s $dir/rows.parquet ]] ;;
    esac
}

reserve_index_space() {
    # Serialize estimates for concurrent workers. Count allocated output bytes plus
    # outstanding reservations, conservatively including partial worker outputs.
    local stage=$1 corpus_dir=$2 model=$3 reservation=$4
    flock "$state/space.lock" "$python_bin" - "$stage" "$corpus_dir" "$model" "$reservation" "$config" "${CLIR_INDEX_BUDGET_BYTES:-2000000000000}" <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path
from clir_benchmark.config import load_config
from clir_benchmark.encoders import load_registry

stage, source, model, record, config_path, budget = sys.argv[1:]
source, record = Path(source), Path(record)
config = load_config(Path(config_path))
root = config.output_dir
count = json.loads((source / 'manifest.json').read_text())['chunk_count']
if stage == 'dense':
    dim = load_registry(config.model_registry)[model].dimension
    # Float32 vectors and a per-row allowance for compressed input/token audits.
    estimate = count * (4 * dim + 512) + 2**20
else:
    # SQLite postings size varies with language, vocabulary and document length.
    estimate = max(4096 * count, 2 * (source / 'lookup.sqlite').stat().st_size) + 2**26
used = int(subprocess.check_output(['du', '-s', '--block-size=1', str(root)]).split()[0])
reserved = sum(int(p.read_text()) for p in record.parent.glob('*.reserve') if p != record)
free = min(shutil.disk_usage(root).free, min(int(budget), 2_000_000_000_000) - used)
print(f'{stage} disk estimate {estimate:,} B; allocated {used:,} B; other reservations {reserved:,} B; available {free:,} B', flush=True)
if reserved + estimate > free:
    raise SystemExit('Index space estimate exceeds the available campaign budget')
temp = record.with_suffix('.reserve.tmp')
temp.write_text(str(estimate))
temp.replace(record)
PY
}

if [[ ${1:-} == --worker ]]; then
    stage=${2:?Missing stage}; corpus=${3:?Missing corpus}; model=${4:-}; expected=${5:?Missing campaign identity}
    [[ $identity == "$expected" ]] || fail 'Staged code or configuration changed after submission. Rerun the launcher.'
    case "$corpus" in chemistry|eurlex|un) ;; *) fail "Unknown corpus $corpus" ;; esac
    key="$stage-$corpus${model:+-$model}"
    exec 8>"$state/$key.lock"
    flock 8
    if complete "$key"; then printf 'Already validated %s\n' "$key"; exit 0; fi
    receipt=$(receipt_for "$key"); temp="$receipt.$SLURM_JOB_ID.tmp"
    export SQLITE_TMPDIR="${SLURM_TMPDIR:?Missing node-local scratch}"
    if [[ $stage == chunks ]]; then
        export USE_TORCH=0
        corpus_dir=$("${cli[@]}" prepare-corpus --corpus "$corpus" --workers "${SLURM_CPUS_PER_TASK:-1}" --work-dir "$SLURM_TMPDIR" | tee /dev/stderr | tail -n1)
        spans=$("${cli[@]}" map-verified-spans --corpus "$corpus" | tee /dev/stderr | tail -n1)
        "$python_bin" - "$corpus_dir" "$spans" <<'PY'
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
source, spans = map(Path, sys.argv[1:])
chunks = json.loads((source / 'manifest.json').read_text())
annotations = json.loads((spans / 'manifest.json').read_text())
if (not chunks['chunk_count'] or pq.read_metadata(source / 'chunks.parquet').num_rows != chunks['chunk_count']
        or not annotations['spans'] or annotations['corpus_fingerprint'] != chunks['corpus_fingerprint']):
    raise SystemExit('Prepared chunk or evidence manifest failed validation')
print(f"Validated {chunks['chunk_count']:,} chunks, span outcomes: {annotations['outcomes']}")
PY
        printf '%s\n%s\n' "$corpus_dir" "$spans" > "$temp"
    else
        complete "chunks-$corpus" || fail "Missing validated chunks for $corpus"
        read -r corpus_dir < "$(receipt_for "chunks-$corpus")"
        reservation="$state/$key.reserve"
        trap 'rm -f -- "$reservation"' EXIT
        reserve_index_space "$stage" "$corpus_dir" "$model" "$reservation"
        case "$stage" in
            bm25)
                artifact=$("$python_bin" scripts/build_verified_bm25_index.py --config "$config" --corpus "$corpus" --build-temp "$SLURM_TMPDIR" --prior-artifacts "$run/artifacts")
                ;;
            dense)
                output=$("${cli[@]}" embed --corpus "$corpus" --kind document --model "$model" | tee /dev/stderr | tail -n1)
                artifact=$("$python_bin" -c 'import json, sys; print(json.loads(sys.argv[1])["document"])' "$output")
                "$python_bin" - "$corpus_dir" "$artifact" "$model" <<'PY'
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
source, artifact = map(Path, sys.argv[1:3])
corpus = json.loads((source / 'manifest.json').read_text())
manifest = json.loads((artifact / 'manifest.json').read_text())
vectors = np.load(artifact / 'vectors.npy', mmap_mode='r', allow_pickle=False)
count = corpus['chunk_count']
if (manifest['kind'] != 'document' or manifest['input_fingerprint'] != corpus['corpus_fingerprint']
        or manifest['model']['alias'] != sys.argv[3] or manifest['rows'] != count
        or manifest['truncation']['all']['truncated'] != 0 or vectors.dtype != np.float32
        or vectors.shape != (count, manifest['model']['dimension'])):
    raise SystemExit('Document embedding manifest or vector shape failed validation')
def ids(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=['id']):
        yield from batch.column(0).to_pylist()
if any(left != right for left, right in itertools.zip_longest(ids(source / 'chunks.parquet'), ids(artifact / 'rows.parquet'))):
    raise SystemExit('Document embedding row IDs differ from shared chunk IDs')
print(f'Validated {count:,} ordered document vectors and zero truncated inputs')
PY
                ;;
            *) fail "Unknown index stage $stage" ;;
        esac
        printf '%s\n' "$artifact" > "$temp"
    fi
    mv -- "$temp" "$receipt"
    printf 'Validated %s in job %s\n' "$key" "$SLURM_JOB_ID"
    exit 0
fi

mode=${1:-}; shift || true
case "$mode" in chunks|indexes|bm25|dense|all) ;; *) fail "Usage: bash $script {chunks|indexes|bm25|dense|all} [--corpus NAME]... [--model ALIAS]..." ;; esac
corpora=(); models=()
while (( $# )); do
    case "$1" in
        --corpus) case "${2:-}" in chemistry|eurlex|un) corpora+=("$2") ;; *) fail 'Invalid corpus selection' ;; esac ;;
        --model) [[ $mode == all || $mode == dense || $mode == indexes ]] || fail '--model requires a dense index stage'; models+=("${2:?Missing model alias}") ;;
        *) fail "Unknown option $1" ;;
    esac
    shift 2
done
if (( ! ${#corpora[@]} )); then corpora=(chemistry eurlex un); fi
if [[ $mode == dense || $mode == indexes || $mode == all ]]; then
    roster=$("$python_bin" -c 'import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["models"]))' "$config")
    if (( ! ${#models[@]} )); then mapfile -t models <<< "$roster"; fi
    for model in "${models[@]}"; do grep -Fxq -- "$model" <<< "$roster" || fail "Unconfigured model $model"; done
fi

# Return through JOB_ID so the lock and job receipts stay in this shell.
queue_stage() {
    local stage=$1 corpus=$2 model=$3 dependency=$4 submit=$5
    local key="$stage-$corpus${model:+-$model}" name active receipt_id last_state
    name="clir-${identity:0:10}-$key"
    JOB_ID=''
    if complete "$key"; then printf 'Complete %s\n' "$key"; return; fi
    active=$(squeue --noheader --user="$USER" --name="$name" --format='%i')
    if [[ -n $active ]]; then
        [[ $active =~ ^[0-9]+$ ]] || fail "Multiple active jobs for $name: $active"
        JOB_ID=$active; printf 'Active %s %s\n' "$key" "$JOB_ID"; return
    fi
    if [[ -s $state/$key-$identity.job ]]; then
        read -r receipt_id < "$state/$key-$identity.job"
        last_state=$(sacct -X -n -P -j "$receipt_id" -o JobIDRaw,State | awk -F'|' -v id="$receipt_id" '$1 == id {print $2; exit}')
        case "$last_state" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED)
                JOB_ID=$receipt_id; printf 'Active %s %s\n' "$key" "$JOB_ID"; return ;;
            '') fail "Waiting for scheduler accounting for $key job $receipt_id. Retry this launcher shortly." ;;
        esac
        # The job may have published between our first receipt check and sacct.
        if complete "$key"; then printf 'Complete %s\n' "$key"; return; fi
    fi
    [[ $submit == yes ]] || fail "Submit the chunk job for $corpus first (use chunks or all)."
    local -a resources=()
    case "$stage" in
        chunks) resources=(--account="${CLIR_CPU_ACCOUNT:-def-mahyarh_cpu}" --cpus-per-task="${CLIR_CHUNK_CPUS:-8}" --mem="${CLIR_CHUNK_MEM:-40G}" --tmp="${CLIR_CHUNK_TMP:-160G}" --time="${CLIR_CHUNK_TIME:-7-00:00:00}") ;;
        bm25) resources=(--account="${CLIR_CPU_ACCOUNT:-def-mahyarh_cpu}" --cpus-per-task=1 --mem="${CLIR_BM25_MEM:-24G}" --tmp="${CLIR_BM25_TMP:-160G}" --time="${CLIR_BM25_TIME:-3-00:00:00}") ;;
        dense) resources=(--account="${CLIR_GPU_ACCOUNT:-def-mahyarh_gpu}" --gpus="${CLIR_GPU:-a100:1}" --cpus-per-task=4 --mem="${CLIR_DENSE_MEM:-32G}" --time="${CLIR_DENSE_TIME:-7-00:00:00}") ;;
    esac
    if [[ -n $dependency ]]; then resources+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes); fi
    JOB_ID=$(sbatch --parsable --job-name="$name" "${resources[@]}" --chdir="$project" --output="$run/logs/$key-%j.out" --error="$run/logs/$key-%j.err" "$script" --worker "$stage" "$corpus" "$model" "$identity")
    JOB_ID=${JOB_ID%%;*}
    [[ $JOB_ID =~ ^[0-9]+$ ]] || fail "Invalid Slurm job ID: $JOB_ID"
    printf '%s\n' "$JOB_ID" > "$state/$key-$identity.job"
    printf '%s\t%s\t%s\t%s\n' "$key" "$JOB_ID" "$dependency" "$identity" >> "$state/submissions.tsv"
    printf 'Submitted %s %s (after %s)\n' "$key" "$JOB_ID" "${dependency:-none}"
}
exec 9>"$state/submit.lock"
flock 9
declare -A parents
for corpus in "${corpora[@]}"; do
    submit=no; if [[ $mode == chunks || $mode == all ]]; then submit=yes; fi
    queue_stage chunks "$corpus" '' '' "$submit"
    parents[$corpus]=$JOB_ID
done
if [[ $mode != chunks ]]; then
    for corpus in "${corpora[@]}"; do
        if [[ $mode != dense ]]; then queue_stage bm25 "$corpus" '' "${parents[$corpus]}" yes; fi
        if [[ $mode != bm25 ]]; then
            for model in "${models[@]}"; do queue_stage dense "$corpus" "$model" "${parents[$corpus]}" yes; done
        fi
    done
fi
