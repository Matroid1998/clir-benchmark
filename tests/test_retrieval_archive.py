from dataclasses import replace
from pathlib import Path
from runpy import run_path

from conftest import FixtureEncoder
from test_lexical_pipeline import prepare_fixture

from clir_benchmark.artifacts import rows
from clir_benchmark.lexical import prepare_bm25
from clir_benchmark.pipeline import embed, evaluate, retrieve

verifier = run_path(str(Path(__file__).parents[1] / "scripts/verify_retrieval_archive.py"))


def test_verified_archive_checks_every_baseline_and_detects_equal_size_corruption(
    tmp_path, tokenizer
):
    config, _ = prepare_fixture(tmp_path, tokenizer)
    config = replace(config, cache_hybrid_candidates=True)
    prepare_bm25(config, "eurlex")
    embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    rel = config.relevance_dir("eurlex")
    eligible = {
        (r["setting_id"], r["query_id"])
        for r in rows(rel / "eligibility.parquet")
        if r["status"] == "eligible"
    }
    entries = {}
    for baseline in ("bm25", "dense", "hybrid"):
        alias = None if baseline == "bm25" else "bge-m3"
        result = retrieve(config, "eurlex", alias, baseline=baseline)
        evaluate(config, "eurlex", alias, baseline=baseline)
        out = {}
        verifier["check_run"](
            config,
            "eurlex",
            alias,
            baseline,
            out,
            entries,
            eligible,
            set(config.datasets["eurlex"].qa.identity()["files"].items()),
        )
        assert out["mixed_ranked_rows"] > 0
    assert verifier["check_file_inventory"](config.output_dir, entries, size_only=False) == {}
    path = result / "manifest.json"
    path.write_bytes(b" " * path.stat().st_size)
    assert verifier["check_file_inventory"](config.output_dir, entries, size_only=False) == {
        str(path.relative_to(config.output_dir)): "SHA-256 differs"
    }
