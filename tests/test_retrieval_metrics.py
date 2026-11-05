import math

import numpy as np
import pytest

from clir_benchmark.encoders import normalize
from clir_benchmark.errors import ValidationError
from clir_benchmark.metrics import aggregate, score_query
from clir_benchmark.retrieval import exact_topk


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
