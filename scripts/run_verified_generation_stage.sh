#!/usr/bin/env bash
# Local supervisor for one November 18 verified generation stage.
set -euo pipefail
stage=${1:?Choose translate or generate-hyde}
case "$stage" in translate|generate-hyde) ;; *) exit 2 ;; esac
shift
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/.venv/bin/python"
local_run="$root/artifacts/verified-20261118-query-generation"
config=${GENERATION_CONFIG:-"$local_run/local.yaml"}
jobs=${GENERATION_STATE_DIR:-"$local_run/jobs"}
remote=/scratch/matroid/clir-benchmark/runs/narval-20261119-verified-query-generation
mode=--run
retry=()
record_refusals=0
for option in "$@"; do
    case "$option" in
        --transfer-only) mode=--assemble ;;
        --retry-failed|--retry-transient) retry+=("$option") ;;
        --record-refusals) retry+=("$option"); record_refusals=1 ;;
        *) echo "Unknown option $option" >&2; exit 2 ;;
    esac
done
mkdir -p "$jobs"
exec 9>"$jobs/$stage-supervisor.lock"
flock -n 9
"$python" scripts/run_generation_shards.py --config "$config" --state-dir "$jobs" \
    --stage "$stage" --shard-size 250 --workers "${GENERATION_WORKERS:-8}" "$mode" "${retry[@]}"

# Build an explicit file allowlist from the verified publication receipt.
inventory=$("$python" - "$config" "$jobs" "$stage" "$record_refusals" <<'PY'
import re
import sys
from pathlib import Path
from clir_benchmark.config import load_config
from clir_benchmark.generation import with_refusal_policy
from clir_benchmark.util import digest, file_digest, read_json
from scripts.run_generation_shards import campaign

config, jobs, stage, record_refusals = sys.argv[1:]
cfg = with_refusal_policy(load_config(Path(config)), record_refusals == "1")
base = cfg.output_dir.resolve()
directory = Path(jobs).resolve() / digest(campaign(cfg, [stage], 250))
verified = read_json(directory / "verified.json")
translate = stage == "translate"
prefix = "translation" if translate else "hyde"
namespace = "translations" if translate else "hyde"
table = "queries.parquet" if translate else "passages.parquet"
paths = set()
for row in verified:
    relative = Path(row["path"]).resolve().relative_to(base)
    if (row["stage"] != stage or len(relative.parts) != 4 or relative.parts[0] != namespace
            or not re.fullmatch(r"[0-9a-f]{64}", relative.name)):
        raise ValueError(f"Unexpected verified artifact {relative}")
    if file_digest(base / relative / "manifest.json") != row["manifest_sha256"]:
        raise ValueError(f"Verified manifest changed at {relative}")
    paths.update([relative / table, relative / "manifest.json"])
for kind, length in [("requests", 64), ("invocations", 32)]:
    parent = base / f"{prefix}_{kind}"
    paths.update(p.relative_to(base) for p in parent.glob("*.json")
                 if p.is_file() and not p.is_symlink() and re.fullmatch(rf"[0-9a-f]{{{length}}}", p.stem))
if not verified:
    raise ValueError("Empty generation receipt")
lines = []
for path in sorted(paths):
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)) or (base/path).resolve() != base/path:
        raise ValueError(f"Unexpected transfer path {path}")
    lines.append(f"{file_digest(base / path)}  {path}\n")
target = directory / "transfer"
target.mkdir(exist_ok=True)
(target / "files").write_text("\n".join(str(p) for p in sorted(paths))+"\n")
(target / "sha256").write_text("".join(lines))
(target / "source").write_text(str(base)+"\n")
print(target)
PY
)
read -r source < "$inventory/source"
key=$(sha256sum "$inventory/sha256" | cut -d' ' -f1)
incoming="$remote/incoming/$stage-$key"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=20 -o ServerAliveCountMax=3)
export RSYNC_RSH="ssh ${ssh_options[*]}"
transfer() {
    ssh "${ssh_options[@]}" narval mkdir -p "$incoming/data" || return
    rsync -rt --checksum --files-from="$inventory/files" "$source/" "narval:$incoming/data/" || return
    rsync -t --checksum "$inventory/files" "$inventory/sha256" "narval:$incoming/" || return
    ssh "${ssh_options[@]}" narval bash -s -- "$remote" "$stage" "$key" <<'REMOTE'
set -euo pipefail
run=$1 stage=$2 key=$3
incoming="$run/incoming/$stage-$key"
cd "$incoming"
[[ "$key" =~ ^[0-9a-f]{64}$ ]]
echo "$key  sha256" | sha256sum -c --status
(cd data && sha256sum -c --quiet ../sha256)
mkdir -p "$run/artifacts" "$run/transfer_receipts"
grep -v '/manifest.json$' files > dependencies
grep '/manifest.json$' files > manifests
grep -v '/manifest.json$' sha256 > dependency-sha256
rsync -rt --ignore-existing --files-from=dependencies data/ "$run/artifacts/"
(cd "$run/artifacts" && sha256sum -c --quiet "$incoming/dependency-sha256")
rsync -rt --ignore-existing --files-from=manifests data/ "$run/artifacts/"
(cd "$run/artifacts" && sha256sum -c --quiet "$incoming/sha256")
install -m 600 sha256 "$run/transfer_receipts/$stage-$key.sha256.tmp"
mv "$run/transfer_receipts/$stage-$key.sha256.tmp" "$run/transfer_receipts/$stage-$key.sha256"
cd "$run"
# Remove only this verified transfer's temporary staging tree.
rm -r -- "$incoming"
REMOTE
}
for attempt in 1 2 3; do
    if transfer; then
        {
            printf 'stage=%s\nsha256=%s\n' "$stage" "$key"
            printf 'remote=narval:%s/artifacts\n' "$remote"
            printf 'verified_utc=%s\n' "$(date -u +%FT%TZ)"
            printf 'files=%s\n' "$(wc -l < "$inventory/files")"
        } > "$inventory/verified.tmp"
        mv "$inventory/verified.tmp" "$inventory/verified"
        cat "$inventory/verified"
        exit 0
    fi
    if (( attempt < 3 )); then sleep $((attempt * 10)); fi
done
echo "Transfer verification failed. Resume with $0 $stage --transfer-only." >&2
exit 1
