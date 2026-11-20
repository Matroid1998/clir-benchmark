import shutil
from dataclasses import replace

import pytest
from conftest import FixtureEncoder, source
from test_pipeline import fixture_config

from clir_benchmark import chunk_batches
from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.config import SharedSentenceConfig, load_config
from clir_benchmark.embedding import load_embeddings
from clir_benchmark.encoders import audit_inputs
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import BM25Index, prepare_bm25
from clir_benchmark.pipeline import embed, prepare_corpus
from clir_benchmark.shared_chunks import SharedSentenceChunker, TextLayout, shared_identity
from clir_benchmark.util import read_json
from clir_benchmark.verified_corpora import map_support


class RetryChunker(Chunker):
    def split(self, unit):
        if unit.id != "a:fr":
            raise AssertionError("A completed boundary batch was computed again")
        return super().split(unit)


def test_parallel_chunk_checkpoints_preserve_serial_results_on_partial_retry(
    tmp_path, monkeypatch, chunker
):
    config, _ = fixture_config(tmp_path)
    serial = prepare_corpus(config, "eurlex", chunker=chunker)
    config = replace(config, output_dir=tmp_path / "parallel")
    monkeypatch.setattr(chunk_batches, "_source_batches", lambda units: ([unit] for unit in units))
    full = prepare_corpus(config, "eurlex", chunker=chunker, workers=2, work_dir=tmp_path)
    cached = sorted((config.output_dir / "chunk-checkpoints").rglob("*.json.gz"))
    assert len(cached) == 3
    assert (
        read_json(full / "manifest.json")["corpus_fingerprint"]
        == read_json(serial / "manifest.json")["corpus_fingerprint"]
    )
    shutil.rmtree(full)
    cached[1].unlink()
    retry = RetryChunker(chunker.tokenizer, chunker.config)
    rebuilt = prepare_corpus(config, "eurlex", chunker=retry, workers=2, work_dir=tmp_path)
    for table in ("sources.parquet", "chunks.parquet", "upstream_pairs.parquet"):
        assert list(rows(rebuilt / table)) == list(rows(serial / table))
    assert (
        read_json(rebuilt / "manifest.json")["corpus_fingerprint"]
        == read_json(serial / "manifest.json")["corpus_fingerprint"]
    )


def test_native_budget_without_extra_common_token_cap(tokenizer):
    shared = make_shared(tokenizer, max_words=200, max_tokens=None, overlap_words=30)
    assert shared.limits == {key: b["native_limit"] for key, b in shared.budgets.items()}


def make_shared(tokenizer, **options):
    icu = pytest.importorskip("icu")
    config = SharedSentenceConfig(("bge-m3", "nomic-v2-moe"), icu.ICU_VERSION, **options)
    identity = shared_identity(config)
    identity["budgets"]["nomic-v2-moe"]["native_limit"] = 39
    return SharedSentenceChunker(config, identity, dict.fromkeys(config.models, tokenizer))


@pytest.mark.parametrize(
    "text,lang",
    [
        ("A test. A second. A third. A fourth. A fifth.", "en"),
        ("联合国。日本語を調べます。\n𠮷野さん。中文没有空格。", "ja"),
        ("🔥" + "a" * 120 + "  😃   ", "en"),
        ("   …" * 30, "fr"),
        ("", "ar"),
    ],
)
def test_shared_chunks_cover_source_and_fit_actual_encoder_format(tokenizer, text, lang):
    shared = make_shared(tokenizer, max_words=5, max_tokens=45, overlap_words=2)
    chunks = shared.split(source(text=text, language=lang))
    layout = TextLayout(text, lang)
    end = 0
    for c in chunks:
        assert c.start <= end and (end < c.end or not text)
        assert c.text == text[c.start : c.end]
        assert layout.words(c.start, c.end) <= 5
        for alias, budget in shared.budgets.items():
            audit = audit_inputs(
                tokenizer, [c.text], budget["document_prefix"], shared.limits[alias]
            )[0]
            assert not audit.truncated
            assert audit.input_tokens == shared.counts(c.text)[alias]
        end = c.end
    assert end == len(text)
    assert chunks == shared.split(source(text=text, language=lang))


def test_sentence_preservation_and_crossing_span_mapping(tokenizer):
    shared = make_shared(tokenizer, max_words=10, max_tokens=47, overlap_words=0)
    unit = source(text="Aaa. Bbb. Ccc. Ddd. Eee. Fff. Ggg.")
    chunks = shared.split(unit)
    layout = TextLayout(unit.text, "en")
    assert len(chunks) >= 2
    assert all(c.end in layout.sentences for c in chunks)
    cut = chunks[0].end
    a, b = cut - 3, cut + 3
    row = dict(
        start=a,
        end=b,
        query_id="q",
        span_index=0,
        match_status="exact",
        corpus_id=unit.source_id,
        language="en",
        source_unit_id=unit.id,
        text=unit.text[a:b],
    )
    m = map_support(row, unit, chunks)
    assert m["complete_chunk_ids"] == []
    assert [p["source_end"] - p["source_start"] for p in m["intersections"]] == [3, 3]
    with_overlap = make_shared(tokenizer, max_words=10, max_tokens=47, overlap_words=1)
    overlap_chunks = with_overlap.split(unit)
    assert map_support(row, unit, overlap_chunks)["complete_chunk_ids"]
    with pytest.raises(ValidationError, match="offsets"):
        map_support({**row, "text": "incorrect"}, unit, chunks)


def test_language_specific_200_word_windows_and_code_point_offsets():
    pytest.importorskip("icu")
    text = "𠮷野さん。今日は良い天気です。" * 140
    layout = TextLayout(text, "ja")
    assert len(text.split()) == 1
    windows = list(layout.word_probes(200))
    assert len(windows) > 1
    assert all(layout.words(start, end) == 200 for start, end in windows)
    assert layout.sentences[-1] == len(text)


def test_shared_corpus_identity_depends_on_budget_models_and_never_selected_retriever(tmp_path):
    config = load_config("report/baseline_analysis/question_15_shared_chunks/tokenizers.yaml")
    assert config.models == load_config("configs/benchmark.yaml").models
    assert len(config.chunking.models) == 8
    assert {"arctic-embed-l-v2", "gte-multilingual-base"} <= set(config.models)
    assert not {"LaBSE", "SapBERT"} & set(config.models)
    key = config.corpus_key("un")
    assert replace(config, models=("bge-m3",), baselines=("bm25",)).corpus_key("un") == key
    override = tmp_path / "models.yaml"
    override.write_text("bge-m3:\n  document_prefix: 'prefix: '\n")
    assert replace(config, model_registry=override).corpus_key("un") != key
    alternate = replace(config.chunking, overlap_words=0)
    assert replace(config, chunking=alternate).corpus_key("un") != key


def test_adding_verified_qa_files_keeps_shared_corpus_cache(tmp_path):
    config = load_config("report/baseline_analysis/question_15_shared_chunks/tokenizers.yaml")
    data = config.datasets["un"]
    src = replace(data.corpus, local_dir=str(tmp_path))
    config = replace(config, datasets={"un": replace(data, corpus=src, qa=src)})
    table = tmp_path / "data/corpus/a.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"original corpus shard")
    key = config.corpus_key("un")
    for name in ("evidence", "qas"):
        other = tmp_path / "data" / name / "a.parquet"
        other.parent.mkdir()
        other.write_bytes(b"new QA data in same release folder")
    assert config.corpus_key("un") == key
    table.write_bytes(b"changed corpus shard")
    assert config.corpus_key("un") != key


def test_prepared_chunks_feed_bm25_and_each_selected_encoder_once(tmp_path, tokenizer):
    icu = pytest.importorskip("icu")
    config, _ = fixture_config(tmp_path)
    models = ("arctic-embed-l-v2", "gte-multilingual-base")
    policy = SharedSentenceConfig(
        models, icu.ICU_VERSION, max_words=3, max_tokens=24, overlap_words=1
    )
    config = replace(config, chunking=policy, models=models)
    chunker = SharedSentenceChunker(
        policy, shared_identity(policy), dict.fromkeys(models, tokenizer)
    )
    corpus_dir = prepare_corpus(config, "eurlex", chunker=chunker)
    published_ids = [r["id"] for r in rows(corpus_dir / "chunks.parquet", ["id"])]
    lexical_only = replace(config, models=(models[0],), baselines=("bm25",))
    assert prepare_corpus(lexical_only, "eurlex") == corpus_dir
    with BM25Index(prepare_bm25(lexical_only, "eurlex")) as db:
        assert {r[0] for r in db.db.execute("SELECT id FROM documents")} == set(published_ids)
    for m in models:
        path = embed(config, "eurlex", m, kind="document", encoder_factory=FixtureEncoder)[
            "document"
        ]
        matrix, metadata, _ = load_embeddings(path)
        assert len(matrix) == len(published_ids)
        assert [r["id"] for r in metadata] == published_ids
