from dataclasses import replace

import pyarrow.parquet as pq
from test_rerank_pipeline import parent_fixture

from clir_benchmark.results import _ranking_details


def test_export_chunk_details_without_random_sqlite_reads(tmp_path, tokenizer, monkeypatch):
    config, _, _, _, run = parent_fixture(tmp_path, tokenizer)
    original = tmp_path / "from-sqlite.parquet"
    fast = tmp_path / "from-parquet.parquet"
    _ranking_details(config, "eurlex", run, original)

    def unexpected_sqlite(*args, **kwargs):
        raise AssertionError("Shared index ranking export must use the prepared chunk table")

    monkeypatch.setattr("clir_benchmark.results.CorpusStore", unexpected_sqlite)
    _ranking_details(replace(config, cache_hybrid_candidates=True), "eurlex", run, fast)
    assert pq.read_table(original).equals(pq.read_table(fast))
