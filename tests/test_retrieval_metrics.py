import math

import numpy as np
import pytest

from clir_benchmark.encoders import normalize
from clir_benchmark.errors import ValidationError
from clir_benchmark.metrics import aggregate, score_query
from clir_benchmark.retrieval import DenseIndex, exact_topk


@pytest.mark.parametrize("block_size", [1, 3, 25])
def test_exact_search_against_brute_force(block_size):
    rng = np.random.default_rng(7)
    docs = normalize(rng.normal(size=(25, 8)))
    queries = normalize(rng.normal(size=(5, 8)))
    ids = [f"c{i:02d}" for i in range(len(docs))]
    languages = ["en" if i % 2 else "fr" for i in range(len(docs))]
    actual = list(
        exact_topk(
            queries,
            docs,
            ids,
            languages,
            ("fr",),
            k=5,
            query_batch_size=2,
            corpus_block_size=block_size,
        )
    )
    for query, ranking in zip(queries, actual, strict=True):
        reference = sorted(
            [(ids[i], float(query @ doc)) for i, doc in enumerate(docs) if languages[i] == "fr"],
            key=lambda item: (-item[1], item[0]),
        )[:5]
        assert [item[0] for item in ranking] == [item[0] for item in reference]
        assert [item[1] for item in ranking] == pytest.approx(
            [item[1] for item in reference], abs=1e-6
        )


def test_ties_at_pruning_boundaries_are_deterministic():
    ids = ["z", "c", "b", "a", "y"]
    documents = np.ones((5, 1), dtype=np.float32)
    for size in (1, 2, 5):
        result = list(
            exact_topk(
                documents[:1], documents, ids, ["en"] * 5, ("en",), k=2, corpus_block_size=size
            )
        )
        assert result == [[("a", 1.0), ("b", 1.0)]]


def test_empty_language_pool_and_short_corpus():
    data = np.ones((1, 1), dtype=np.float32)
    assert list(exact_topk(data, data, ["a"], ["en"], ("fr",))) == [[]]
    assert list(exact_topk(data, data, ["a"], ["en"], ("en",))) == [[("a", 1.0)]]


def test_chunk_and_source_metrics_differ():
    relevant = {"a1": {"a"}, "a2": {"a"}, "b1": {"b"}}
    scores = score_query(["a1", "irrelevant", "a2", "b1"], relevant, {"a", "b"})
    assert scores["Hit@1"] == 1
    assert scores["Recall@1"] == 1 / 3
    assert scores["SourceRecall@1"] == 0.5
    assert scores["AllSourceSuccess@1"] == 0
    assert scores["AllSourceSuccess@5"] == 1
    assert scores["Recall@5"] == 1
    expected = (1 + 1 / math.log2(4) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    assert scores["nDCG@10"] == pytest.approx(expected)
    assert scores["MRR@10"] == 1
    assert score_query([], relevant, {"a", "b"})["MRR@10"] == 0
    with pytest.raises(ValidationError, match="duplicate"):
        score_query(["a1", "a1"], relevant, {"a", "b"})


def test_aggregation_never_combines_regimes():
    records = [
        {
            "corpus": "un",
            "regime": regime,
            "setting_id": regime,
            "query_language": "en",
            "target_languages": "en" if regime == "monolingual" else "fr",
            "generator": "g",
            "persona": "p",
            "source_count": 1,
            "metrics": {"Hit@1": score},
        }
        for regime, score in [("monolingual", 1.0), ("cross_language", 0.0), ("mixed", 0.5)]
    ]
    result = aggregate(records)
    assert len(result["macro_by_regime"]) == 3
    assert {row["regime"]: row["metrics"]["Hit@1"] for row in result["macro_by_regime"]} == {
        "monolingual": 1,
        "cross_language": 0,
        "mixed": 0.5,
    }


@pytest.mark.parametrize("block_size", [1, 7, 83])
@pytest.mark.parametrize("targets", [("en",), ("en", "fr")])
def test_document_search_matches_all_chunk_maxima(block_size, targets):
    rng = np.random.default_rng(49)
    vectors = normalize(rng.normal(size=(83, 8)))
    queries = normalize(rng.normal(size=(5, 8)))
    chunk_ids = [f"c{i:02d}" for i in range(len(vectors))]
    documents = [f"d{i % 19:02d}" for i in range(len(vectors))]
    languages = ["en" if i % 19 % 2 else "fr" for i in range(len(vectors))]
    index = DenseIndex(vectors, chunk_ids, languages, document_ids=documents)
    actual = list(
        index.search_documents(
            queries, targets, k=8, query_batch_size=2, corpus_block_size=block_size
        )
    )
    for query, ranking in zip(queries, actual, strict=True):
        reference = {}
        for document, chunk, language, vector in zip(
            documents, chunk_ids, languages, vectors, strict=True
        ):
            if language not in targets:
                continue
            score = float(query @ vector)
            if document not in reference or (-score, chunk) < reference[document]:
                reference[document] = (-score, chunk)
        expected = sorted(
            [(document, chunk, -score) for document, (score, chunk) in reference.items()],
            key=lambda item: (-item[2], item[0]),
        )[:8]
        assert [(d, c) for d, c, _ in ranking] == [(d, c) for d, c, _ in expected]
        assert [s for _, _, s in ranking] == pytest.approx([s for _, _, s in expected], abs=1e-6)


@pytest.mark.parametrize("block_size", [1, 31, 201])
def test_many_high_scoring_chunks_do_not_crowd_out_documents(block_size):
    # A chunk-top-100 followed by deduplication would return only document a.
    vectors = np.array([[1.0]] * 101 + [[0.9]] * 100, dtype=np.float32)
    documents = ["a"] * 101 + [f"d{i:03d}" for i in range(100)]
    chunks = [f"c{i:03d}" for i in reversed(range(201))]
    index = DenseIndex(vectors, chunks, ["en"] * 201, document_ids=documents)
    ranking = next(
        index.search_documents(
            np.ones((1, 1), dtype=np.float32), ("en",), corpus_block_size=block_size
        )
    )
    assert len(ranking) == 100
    assert ranking[0] == ("a", "c100", 1.0)
    assert [document for document, _, _ in ranking[1:]] == [f"d{i:03d}" for i in range(99)]


@pytest.mark.parametrize("block_size", [1, 2, 6])
def test_document_and_winning_chunk_ties_are_deterministic(block_size):
    index = DenseIndex(
        np.ones((6, 1), dtype=np.float32),
        ["z", "a", "y", "b", "c", "d"],
        ["en"] * 6,
        document_ids=["doc-b", "doc-b", "doc-a", "doc-a", "doc-c", "doc-d"],
    )
    assert list(
        index.search_documents(
            np.ones((1, 1), dtype=np.float32), ("en",), k=2, corpus_block_size=block_size
        )
    ) == [[("doc-a", "b", 1.0), ("doc-b", "a", 1.0)]]


def test_document_versions_remain_separate_and_empty_pool_is_supported():
    index = DenseIndex(
        np.array([[0.9], [1.0], [0.8]], dtype=np.float32),
        ["en-1", "fr-1", "fr-2"],
        ["en", "fr", "fr"],
        document_ids=["a:en", "a:fr", "a:fr"],
    )
    query = np.ones((1, 1), dtype=np.float32)
    ranking = next(index.search_documents(query, ("en", "fr")))
    assert [(d, c) for d, c, _ in ranking] == [("a:fr", "fr-1"), ("a:en", "en-1")]
    assert list(index.search_documents(query, ("de",))) == [[]]
    scores = score_query([d for d, _, _ in ranking], {"a:en": {"a"}, "a:fr": {"a"}}, {"a"})
    assert scores["Recall@1"] == 0.5
    assert scores["SourceRecall@1"] == 1.0


def test_document_search_rejects_invalid_metadata_and_values():
    vectors = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValidationError, match="every chunk"):
        DenseIndex(vectors, ["a", "b"], ["en", "en"], document_ids=["x"])
    with pytest.raises(ValidationError, match="multiple language"):
        DenseIndex(vectors, ["a", "b"], ["en", "fr"], document_ids=["x", "x"])
    index = DenseIndex(vectors, ["a", "b"], ["en", "en"])
    with pytest.raises(ValidationError, match="source document ID"):
        list(index.search_documents(vectors, ("en",)))
    index = DenseIndex(vectors, ["a", "b"], ["en", "en"], document_ids=["x", "y"])
    with pytest.raises(ValidationError, match="positive"):
        list(index.search_documents(vectors, ("en",), k=0))
    with pytest.raises(ValidationError, match="matching dimensions"):
        list(index.search_documents(np.ones((1, 2)), ("en",)))
    with pytest.raises(ValidationError, match="nonfinite"):
        list(index.search_documents(np.array([[np.nan]], dtype=np.float32), ("en",)))
    vectors[0, 0] = np.inf
    with pytest.raises(ValidationError, match="nonfinite"):
        list(index.search_documents(np.ones((1, 1)), ("en",)))
