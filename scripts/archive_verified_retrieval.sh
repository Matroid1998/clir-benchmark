#!/usr/bin/env bash
# Run base after indexing. Schedule final with afterany on every retrieval
# campaign job so logs and completed artifacts are retained even after a failure.
set -euo pipefail
umask 077
mode=${1:-}
case $mode in base|final) ;; *) echo "Usage: $0 base|final" >&2; exit 2 ;; esac
shared_name=narval-20261119-shared-indexes
current_name=narval-20261120-verified-retrieval
runs=${CLIR_ARCHIVE_RUNS:-${SCRATCH:?}/clir-benchmark/runs}
dest=${CLIR_ARCHIVE_DEST:-/project/def-mahyarh/matroid/clir-benchmark/$current_name}
runs=$(realpath -e "$runs")
shared=$runs/$shared_name
current=$runs/$current_name
provenance=$current/provenance
python_bin=${CLIR_ARCHIVE_PYTHON:-$runs/../venv/bin/python}
mkdir -p -- "$dest" "$provenance"
[[ -O $dest && ! -L $dest ]] || { echo "Archive destination must be owned by this user" >&2; exit 1; }
dest=$(realpath -e "$dest")
[[ $dest != "$runs" && $dest != "$runs/"* ]] || { echo "Archive destination overlaps source" >&2; exit 1; }
chmod 700 "$dest"
exec 8>"$provenance/.archive-$mode.lock"
flock 8

verify_published() {
    [[ -s $1/archive.tar && -s $1/archive.sha256 && -s $1/status.txt ]] &&
        (cd "$1" && sha256sum --check archive.sha256)
}
if [[ -e $dest/$mode ]]; then verify_published "$dest/$mode"; exit; fi

if [[ -f $runs/../env.sh ]]; then
    set +u
    source "$runs/../env.sh"
    set -u
fi
export PYTHONPATH="$current/project/src:$shared/python-deps${PYTHONPATH:+:$PYTHONPATH}"

# Never include dotfiles (including credentials and uncommitted worker output),
# environment files, key material, or intermediate chunker checkpoints.
exclusions=()
for pattern in '.*' '__pycache__' '*.pyc' '*.tmp' '*.lock' '*.reserve' '*.partial' \
    'chunk-checkpoints' 'checkpoints' '*.pem' '*.key' '*.p12' 'authorized_keys' \
    'id_rsa*' 'id_ed25519*' 'credentials' 'credentials.*' 'secrets' 'secrets.*' \
    'token' 'stored_tokens' 'venv' 'env.sh'; do
    exclusions+=("--exclude=$pattern")
done
stage=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/clir-archive.XXXXXXXX")
incoming=''
cleanup() {
    rm -rf -- "$stage"
    if [[ -n $incoming && -d $incoming ]]; then rm -rf -- "$incoming"; fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
"$python_bin" - "$stage/environment.json" <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import distributions

packages = sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])
with open(sys.argv[1], "w") as out:
    json.dump({
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "archive_node": platform.node(),
        "loaded_modules": os.getenv("LOADEDMODULES", "").split(":") if os.getenv("LOADEDMODULES") else [],
        "distributions": packages,
    }, out, indent=2)
    out.write("\n")
PY
mv "$stage/environment.json" "$provenance/archive-$mode-environment.json"
partition=$mode
status=complete
stable=(corpus embeddings/document lexical verified-spans)
source_paths=()
case $mode in
    base)
        for dir in "${stable[@]}"; do source_paths+=("$shared_name/artifacts/$dir"); done
        for dir in inputs project logs job_status; do source_paths+=("$shared_name/$dir"); done
        ;;
    final)
        # The successful base was checksummed on readback before atomic publication.
        # Keep its digest with these results without recalling its large tar a second time.
        base_bundle=$dest/base
        if [[ -s $base_bundle/archive.tar && -s $base_bundle/archive.sha256 && -s $base_bundle/status.txt ]] &&
            grep -Fxq 'status=complete' "$base_bundle/status.txt" &&
            grep -Fxq "bytes=$(stat -c %s "$base_bundle/archive.tar")" "$base_bundle/status.txt"; then
            cp "$base_bundle/archive.sha256" "$provenance/base-archive.sha256"
        else
            printf 'Base archive or its verified receipt is missing or inconsistent.\n' >&2
            status=incomplete
        fi
        # Keep accounting for all submission attempts and every Slurm job step.
        scheduler=$provenance/slurm-final.psv
        ledger=$current/job_status/submissions.tsv
        ids=$(awk -F '\t' '$2 ~ /^[0-9]+$/ {print $2}' "$ledger" | sort -u | paste -sd,)
        if [[ -n $ids ]] && sacct -P -j "$ids" \
            -o JobIDRaw,JobName%80,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,MaxRSS,TotalCPU \
            > "$stage/slurm.psv" 2>"$provenance/slurm-final-errors.log"; then
            mv "$stage/slurm.psv" "$scheduler"
            # Fail only the latest attempt for each stage. Retain older attempts
            # in the accounting even when a later retry succeeded.
            if ! "$python_bin" - "$ledger" "$scheduler" <<'PY'
import csv
import sys

with open(sys.argv[1]) as src:
    ledger = list(csv.reader(src, delimiter="\t"))
latest = {row[0]: row[1] for row in ledger}
with open(sys.argv[2]) as src:
    states = {row["JobIDRaw"]: row["State"] for row in csv.DictReader(src, delimiter="|")}
if (not latest or any(row[1] not in states for row in ledger)
        or any(states.get(job) != "COMPLETED" for job in latest.values())):
    sys.exit(1)
PY
            then status=incomplete; fi
        else
            mv "$stage/slurm.psv" "$scheduler" 2>/dev/null || touch "$scheduler"
            status=incomplete
        fi
        if ! "$python_bin" "$current/project/scripts/verify_retrieval_archive.py" \
            --campaign "$current" --artifacts "$shared/artifacts" \
            --output "$provenance/completion.json" >"$provenance/verify-final.log" 2>&1; then
            status=incomplete
        fi
        if [[ $status == incomplete ]]; then
            partition=partial-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
            if [[ -e $dest/$partition ]]; then verify_published "$dest/$partition"; exit 1; fi
        fi
        source_paths+=("$shared_name/artifacts" "$current_name/project" "$current_name/inputs")
        for dir in "${stable[@]}"; do exclusions+=("--exclude=$shared_name/artifacts/$dir"); done
        ;;
esac
source_paths+=("$current_name/qa.sha256")

# A live stdout file can grow while tar reads it. Copy control files to a stable
# local snapshot first; the final bundle supersedes the initial log snapshot.
controls=()
for dir in logs job_status provenance; do
    mkdir -p "$stage/$current_name/$dir"
    rsync -rt --safe-links "${exclusions[@]}" "$current/$dir/" "$stage/$current_name/$dir/"
    controls+=("$current_name/$dir")
done
incoming=$(mktemp -d "$dest/.writing-$partition.XXXXXXXX")
printf 'mode=%s\nstatus=%s\ncreated_utc=%s\nslurm_job_id=%s\n' \
    "$mode" "$status" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-manual}" > "$incoming/status.txt"
# Hash the exact stream on the producer side. Compare a second digest read
# from the destination filesystem before publishing the archive and its receipt.
tar --create --file=- --blocking-factor=2048 --verbose --index-file="$incoming/members.txt" \
    "${exclusions[@]}" --directory="$runs" "${source_paths[@]}" \
    --directory="$stage" "${controls[@]}" |
    tee "$incoming/archive.tar" | sha256sum | awk '{print $1 "  archive.tar"}' > "$incoming/archive.sha256"
sync -d "$incoming/archive.tar"
verify_published "$incoming"
printf 'bytes=%s\nreadback_verified_utc=%s\n' "$(stat -c %s "$incoming/archive.tar")" \
    "$(date -u +%FT%TZ)" >> "$incoming/status.txt"
mv -T -- "$incoming" "$dest/$partition"
incoming=''
printf 'Verified %s (%s)\n' "$dest/$partition/archive.tar" "$status"
[[ $status == complete ]]
