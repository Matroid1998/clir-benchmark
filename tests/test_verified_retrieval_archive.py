import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_verified_retrieval.sh"
SHARED = "narval-20261119-shared-indexes"
CURRENT = "narval-20261120-verified-retrieval"


def test_base_and_terminal_archives(tmp_path):
    runs = tmp_path / "runs"
    dest = tmp_path / "nearline"
    old = runs / SHARED
    new = runs / CURRENT
    states = [
        (old, "inputs/old.parquet"),
        (old, "project/src/main.py"),
        (old, "logs/index.out"),
        (old, "job_status/index.done"),
        (old, "artifacts/corpus/chunks.parquet"),
        (old, "artifacts/lexical/index.sqlite"),
        (old, "artifacts/verified-spans/span-map.jsonl"),
        (old, "artifacts/embeddings/document/vectors.npy"),
        (old, "artifacts/embeddings/query/vectors.npy"),
        (old, "artifacts/reports/report.json"),
        (old, "artifacts/runs/candidates.parquet"),
        (old, "artifacts/chunk-checkpoints/work.txt"),
        (old, "artifacts/runs/.partial/scores"),
        (new, "inputs/new.parquet"),
        (new, "qa.sha256"),
        (new, "project/.env"),
        (new, "logs/start.out"),
    ]
    for root, rel in states:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel)
    ledger = new / "job_status/submissions.tsv"
    ledger.parent.mkdir()
    ledger.write_text("bm25-un\t12\t\tfingerprint\n")
    verifier = new / "project/scripts/verify_retrieval_archive.py"
    verifier.parent.mkdir()
    verifier.write_text(
        "from pathlib import Path\nimport os,sys\n"
        "Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{}')\n"
        "sys.exit(int(os.environ.get('FIXTURE_VERIFIER_EXIT', 0)))\n"
    )
    sacct = tmp_path / "sacct"
    sacct.write_text("#!/bin/sh\nprintf 'JobIDRaw|State\\n12|COMPLETED\\n12.batch|COMPLETED\\n'\n")
    sacct.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + ":" + os.environ["PATH"],
        "CLIR_ARCHIVE_RUNS": str(runs),
        "CLIR_ARCHIVE_DEST": str(dest),
        "CLIR_ARCHIVE_PYTHON": sys.executable,
        "SLURM_JOB_ID": "812",
    }

    def archive(mode, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT), mode],
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )

    done = archive("base")
    assert done.returncode == 0, done.stderr
    assert archive("base").returncode == 0
    with tarfile.open(dest / "base/archive.tar") as tar:
        assert f"{SHARED}/artifacts/embeddings/document/vectors.npy" in tar.getnames()
        assert f"{SHARED}/artifacts/runs/candidates.parquet" not in tar.getnames()
        assert f"{CURRENT}/logs/start.out" in tar.getnames()
    assert archive("final", FIXTURE_VERIFIER_EXIT="1").returncode != 0
    assert "status=incomplete" in (dest / "partial-812/status.txt").read_text()
    assert not (dest / "final").exists()
    done = archive("final")
    assert done.returncode == 0, done.stderr
    bundle = dest / "final/archive.tar"
    with tarfile.open(bundle) as tar:
        names = tar.getnames()
        assert f"{SHARED}/artifacts/runs/candidates.parquet" in names
        assert f"{SHARED}/artifacts/embeddings/query/vectors.npy" in names
        assert f"{CURRENT}/provenance/completion.json" in names
        assert f"{CURRENT}/provenance/slurm-final.psv" in names
        assert all(
            part not in name
            for name in names
            for part in ("chunk-checkpoints", ".partial", ".env", "embeddings/document")
        )
    checksum = (dest / "final/archive.sha256").read_text().split()[0]
    assert checksum == hashlib.sha256(bundle.read_bytes()).hexdigest()
    with bundle.open("ab") as stream:
        stream.write(b"corrupt")
    assert archive("final").returncode != 0
