from dataclasses import replace

from conftest import FixtureEncoder

from clir_benchmark.config import RuntimeConfig
from clir_benchmark.embedding import (
    embedding_key,
    encode_records,
    load_embeddings,
    span_visible,
    support_exposure,
)
from clir_benchmark.encoders import audit_inputs, load_registry
from clir_benchmark.util import json_dumps


def test_all_models_are_pinned_and_prompts_are_explicit():
    registry = load_registry()
    assert len(registry) == 8
    assert all(len(spec.revision) == 40 for spec in registry.values())
    assert registry["SapBERT"].backend == "transformers_cls"
    assert registry["LaBSE"].max_length == 256
    assert registry["nomic-v2-moe"].document_prefix == "search_document: "
    assert registry["embeddinggemma"].document_prefix == "title: none | text: "
    assert registry["bge-m3"].query_prefix == ""


def test_audit_includes_prompt_and_special_tokens(tokenizer):
    audit = audit_inputs(tokenizer, ["abcdef"], "Q:", 7)[0]
    assert audit.input_tokens == 10
    assert audit.encoded_tokens == 7
    assert audit.truncated
    assert audit.visible_ranges == ((0, 3),)
    row = {"source_start": 10, "visible_ranges_json": json_dumps(audit.visible_ranges)}
    assert span_visible({"start": 10, "end": 13}, row)
    assert not span_visible({"start": 12, "end": 14}, row)


def test_empty_and_untruncated_audit(tokenizer):
    audits = audit_inputs(tokenizer, ["", " a "], "", 20)
    assert not any(audit.truncated for audit in audits)
    assert audits[1].visible_ranges == ((0, 3),)


def test_cache_reuse_and_model_setting_invalidation(tmp_path):
    runtime = RuntimeConfig(batch_size=1)
    spec = load_registry()["bge-m3"]
    encoder = FixtureEncoder(spec, runtime)
    data = [
        {"id": "a", "text": "alpha", "language": "en"},
        {"id": "b", "text": "beta", "language": "fr"},
    ]
    destination = tmp_path / "cache"
    before = FixtureEncoder.calls
    encode_records(destination, lambda: iter(data), 2, encoder, "document", runtime, "input-hash")
    assert FixtureEncoder.calls == before + 2
    encode_records(destination, lambda: iter(data), 2, encoder, "document", runtime, "input-hash")
    assert FixtureEncoder.calls == before + 2
    matrix, metadata, manifest = load_embeddings(destination)
    assert matrix.shape == (2, 1024)
    assert len(metadata) == 2 and manifest["rows"] == 2
    first = embedding_key("input-hash", "document", spec, runtime)
    assert first != embedding_key(
        "input-hash", "document", replace(spec, document_prefix="new: "), runtime
    )
    assert first != embedding_key("new-input", "document", spec, runtime)


def test_exposure_uses_any_surviving_qualifier_and_all_sources():
    rows = [
        {"id": "c1", "source_start": 0, "visible_ranges_json": "[[0,3]]"},
        {"id": "c2", "source_start": 0, "visible_ranges_json": "[[0,2]]"},
    ]
    matches = [
        {
            "setting_id": "s",
            "query_id": "q",
            "chunk_id": "c1",
            "required_source_id": "a",
            "start": 0,
            "end": 3,
        },
        {
            "setting_id": "s",
            "query_id": "q",
            "chunk_id": "c1",
            "required_source_id": "a",
            "start": 3,
            "end": 6,
        },
        {
            "setting_id": "s",
            "query_id": "q",
            "chunk_id": "c2",
            "required_source_id": "b",
            "start": 0,
            "end": 3,
        },
    ]
    result = support_exposure(matches, rows)["s"]
    assert result["positive_query_chunk_pairs"] == 2
    assert result["positive_query_chunk_pairs_without_visible_span"] == 1
    assert result["queries_with_all_sources_visible"] == 0


def test_audit_does_not_count_partially_encoded_unicode_character():
    class SharedOffsetsTokenizer:
        is_fast = True
        truncation_side = "right"

        def __call__(self, text, truncation=False, **kwargs):
            if truncation:
                return {
                    "input_ids": [1, 10, 11, 2],
                    "offset_mapping": [(0, 0), (0, 1), (1, 2), (0, 0)],
                }
            return {
                "input_ids": [1, 10, 11, 12, 2],
                "offset_mapping": [(0, 0), (0, 1), (1, 2), (1, 2), (0, 0)],
            }

    audit = audit_inputs(SharedOffsetsTokenizer(), ["a😀"], "", 4)[0]
    assert audit.truncated
    assert audit.visible_ranges == ((0, 1),)


def test_zero_query_embedding_artifact_is_valid(tmp_path):
    runtime = RuntimeConfig()
    encoder = FixtureEncoder(load_registry()["bge-m3"], runtime)
    path = tmp_path / "empty"
    encode_records(path, lambda: iter([]), 0, encoder, "query", runtime, "empty")
    matrix, metadata, manifest = load_embeddings(path)
    assert matrix.shape == (0, 1024)
    assert metadata == [] and manifest["rows"] == 0
