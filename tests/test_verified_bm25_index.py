import runpy
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
from test_pipeline import fixture_config

from clir_benchmark.chunking import Chunker
from clir_benchmark.pipeline import prepare_corpus
from clir_benchmark.util import file_digest, read_json, write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_verified_bm25_index.py"


@pytest.fixture
def campaign(tmp_path, tokenizer, monkeypatch, capsys):
    config, path = fixture_config(tmp_path)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    previous, local = tmp_path / "prior", tmp_path / "node-local"
    local.mkdir()

    def run():
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--config",
                str(path),
                "--corpus",
                "eurlex",
                "--prior-artifacts",
                str(previous),
                "--build-temp",
                str(local),
            ],
        )
        runpy.run_path(str(SCRIPT), run_name="__main__")
        messages = capsys.readouterr()
        assert not list(local.iterdir())
        return Path(messages.out.strip()), messages.err

    return run, previous


def test_build_and_reuse_current_or_prior_index(campaign):
    run, previous = campaign
    index, status = run()
    assert "Built eurlex: 3 prepared and indexed chunks" in status
    sqlite_hash = file_digest(index / "index.sqlite")
    assert "Reused eurlex" in run()[1]
    old = previous / "lexical" / "eurlex" / index.name
    old.parent.mkdir(parents=True)
    index.rename(old)
    index, status = run()
    assert "Copied and verified identical prior index" in status
    assert file_digest(index / "index.sqlite") == sqlite_hash


@pytest.mark.parametrize("bad", ["fingerprint", "analyzer", "count", "unreadable"])
def test_invalid_prior_index_rebuilds_from_all_chunks(campaign, bad):
    run, previous = campaign
    index, _ = run()
    old = previous / "lexical" / "eurlex" / index.name
    old.parent.mkdir(parents=True)
    shutil.move(index, old)
    if bad in {"fingerprint", "analyzer"}:
        manifest = read_json(old / "manifest.json")
        manifest["corpus_fingerprint" if bad == "fingerprint" else "analyzer"] = "other"
        write_json(old / "manifest.json", manifest)
    elif bad == "count":
        with sqlite3.connect(old / "index.sqlite") as db:
            db.execute("DELETE FROM documents WHERE doc = 1")
    else:
        (old / "index.sqlite").write_text("incomplete sqlite file")
    index, status = run()
    assert "Discarding incompatible prior index" in status
    assert "Built eurlex: 3 prepared and indexed chunks" in status
    with sqlite3.connect(index / "index.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone() == (3,)
