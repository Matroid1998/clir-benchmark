import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clir_benchmark.adapters import DatasetReader, verify_chemistry_provenance
from clir_benchmark.artifacts import CorpusStore, rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.config import Config, DatasetSpec, SourceSpec
from clir_benchmark.errors import ValidationError
from clir_benchmark.pipeline import prepare_corpus
from clir_benchmark.util import read_json


def test_chemistry_retains_all_pair_endpoints_and_only_direct_accepted_links(tmp_path, tokenizer):
    upstream = tmp_path / "upstream"
    path = upstream / "data/train-00000.parquet"
    path.parent.mkdir(parents=True)
    pairs = [
        {
            "doc_id": "P",
            "source_language": "en",
            "target_language": "fr",
            "source_text": "alpha",
            "target_text": "preuve",
            "pair_ok": True,
            "judge_verdict": "exact",
            "example_id": "p1",
        },
        {
            "doc_id": "P",
            "source_language": "fr",
            "target_language": "zh",
            "source_text": "preuve",
            "target_text": "中文",
            "pair_ok": True,
            "judge_verdict": "high",
            "example_id": "p2",
        },
        {
            "doc_id": "R",
            "source_language": "ja",
            "target_language": "ru",
            "source_text": "日本語",
            "target_text": "русский",
            "pair_ok": False,
            "judge_verdict": "low",
            "example_id": "p3",
        },
        {
            "doc_id": "S",
            "source_language": "en",
            "target_language": "de",
            "source_text": "alpha",
            "target_text": "Beleg",
            "pair_ok": False,
            "judge_verdict": "medium",
            "example_id": "p4",
        },
    ]
    pq.write_table(pa.Table.from_pylist(pairs), path)
    source = SourceSpec("fixture/upstream", "a" * 40, str(upstream))
    config = Config(
        output_dir=tmp_path / "artifacts", datasets={"chemistry": DatasetSpec(source, source)}
    )
    destination = prepare_corpus(config, "chemistry", chunker=Chunker(tokenizer, config.chunking))
    manifest = read_json(destination / "manifest.json")
    assert manifest["source_count"] == 7
    assert manifest["accepted_direct_pairs"] == 2
    assert manifest["duplicate_source_occurrences"] == 1
    assert len(list(rows(destination / "upstream_pairs.parquet"))) == 4
    with CorpusStore(destination / "lookup.sqlite") as store:
        assert store.get_source("R_ja").text == "日本語"
        assert store.get_source("R_ru").text == "русский"
        assert store.get_source("S_en").text == store.get_source("P_en").text
        assert store.get_source("P_en").title == ""
        assert set(store.variants(store.get_source("P_en"))) == {"P_en", "P_fr"}
        assert store.variants(store.get_source("R_ja")) == ["R_ja"]

        qa_root = tmp_path / "qa"
        qa_path = qa_root / "data/corpus/train-00000.parquet"
        qa_path.parent.mkdir(parents=True)
        raw = {
            "_id": "P_en",
            "text": "alpha",
            "language": "en",
            "metadata_json": json.dumps(
                {"upstream_dataset": source.repo_id, "upstream_revision": source.revision}
            ),
        }
        pq.write_table(pa.Table.from_pylist([raw]), qa_path)
        reader = DatasetReader(SourceSpec("fixture/qa", "b" * 40, str(qa_root)))
        verify_chemistry_provenance(reader, store, source)
        pq.write_table(pa.Table.from_pylist([{**raw, "text": "normalized or changed"}]), qa_path)
        with pytest.raises(ValidationError, match="differs verbatim"):
            verify_chemistry_provenance(reader, store, source)
        pq.write_table(pa.Table.from_pylist([{**raw, "metadata_json": "{}"}]), qa_path)
        with pytest.raises(ValidationError, match="provenance"):
            verify_chemistry_provenance(reader, store, source)
