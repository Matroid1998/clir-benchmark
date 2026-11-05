import math
from collections import Counter
from dataclasses import replace

import pytest

from clir_benchmark.config import BM25Config, HybridConfig
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import BM25Index, build_index, lexical_tokens
from clir_benchmark.retrieval import reciprocal_rank_fusion


def test_multilingual_analyzer_is_explicit_and_has_no_stoplist():
    text = "CAFÉ Cafe\u0301 Straße 中文化学 カタカナ русский العَرَبِيَّة H2O <p>The</p>"
    assert lexical_tokens(text) == [
        "café",
        "café",
        "strasse",
        "中",
        "文",
        "化",
        "学",
        "カ",
        "タ",
        "カ",
        "ナ",
        "русский",
        "العَرَبِيَّة",
        "h2o",
        "p",
        "the",
        "p",
    ]
    assert lexical_tokens(" ?!--__\n ") == []
    assert lexical_tokens("alpha-beta 12.5") == ["alpha", "beta", "12", "5"]


@pytest.mark.parametrize("languages", [("en",), ("fr",), ("en", "fr"), ("zh",)])
@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.0, 1.0), (2.0, 0.0)])
def test_bm25_matches_bruteforce_with_pool_statistics(tmp_path, languages, k1, b):
    # Independent reference uses the pre-tokenized fixture, not index internals.
    documents = [
        ("z", "en", ["alpha", "beta", "beta"]),
        ("b", "en", ["beta"]),
        ("a", "en", ["beta"]),
        ("empty", "en", []),
        ("d", "fr", ["alpha", "alpha", "beta", "gamma"]),
        ("e", "fr", ["gamma"]),
        ("f", "fr", ["alpha", "delta"]),
    ]
    records = [
        {"id": doc_id, "language": lang, "text": " ".join(terms)}
        for doc_id, lang, terms in documents
    ]
    config = BM25Config(k1=k1, b=b)
    path = build_index(tmp_path / "index", records, "fixture", config)
    pool = [(doc_id, terms) for doc_id, lang, terms in documents if lang in languages]
    with BM25Index(path, config) as index:
        queries = ["alpha beta beta", "beta", "delta", "absent", ""]
        results = list(index.search(queries, languages, k=2))
        for query, ranking in zip(queries, results, strict=True):
            expected = []
            for doc_id, terms in pool:
                score = 0.0
                tf = Counter(terms)
                for term in set(query.split()):
                    if tf[term]:
                        df = sum(term in dterms for _, dterms in pool)
                        idf = math.log(1 + (len(pool) - df + 0.5) / (df + 0.5))
                        avgdl = sum(len(dterms) for _, dterms in pool) / len(pool)
                        score += (
                            idf
                            * tf[term]
                            * (k1 + 1)
                            / (tf[term] + k1 * (1 - b + b * len(terms) / avgdl))
                        )
                if score > 0:
                    expected.append((doc_id, score))
            expected.sort(key=lambda item: (-item[1], item[0]))
            assert [d for d, _ in ranking] == [d for d, _ in expected[:2]]
            assert [s for _, s in ranking] == pytest.approx([s for _, s in expected[:2]], abs=1e-12)


def test_index_order_ties_empty_pool_and_query_repetition(tmp_path):
    records = [{"id": name, "language": "zh", "text": "化学"} for name in ["z", "b", "a"]]
    rankings = []
    for ordinal, inputs in enumerate((records, list(reversed(records)))):
        path = build_index(tmp_path / str(ordinal), inputs, "fixture", BM25Config())
        with BM25Index(path) as index:
            first, repeated = list(index.search(["化", "化化化"], ("zh", "zh"), k=2))
            assert first == repeated
            assert [doc for doc, _ in first] == ["a", "b"]
            assert list(index.search(["化"], ("en",))) == [[]]
            assert list(index.search(["unknown", ""], ("zh",))) == [[], []]
            rankings.append(first)
    assert rankings[0] == rankings[1]


@pytest.mark.parametrize("records", [[], [{"id": "empty", "language": "en", "text": ""}]])
def test_empty_index_and_tokenless_corpus(tmp_path, records):
    path = build_index(tmp_path / "index", records, "fixture", BM25Config())
    with BM25Index(path) as index:
        assert list(index.search(["test", ""], ("en",))) == [[], []]


def test_duplicate_ids_reject_build_without_publishing(tmp_path):
    path = tmp_path / "index"
    record = {"id": "same", "language": "en", "text": "test"}
    with pytest.raises(ValidationError, match="duplicate"):
        build_index(path, [record, record], "fixture", BM25Config())
    assert not path.exists()


def test_cached_index_does_not_read_input_or_depend_on_scoring_parameters(tmp_path):
    path = tmp_path / "index"
    build_index(path, [{"id": "a", "language": "en", "text": "alpha"}], "fixture", BM25Config())

    def forbidden():
        raise AssertionError("Cache hit must not consume corpus input")
        yield

    assert build_index(path, forbidden(), "fixture", BM25Config(k1=2)) == path
    with pytest.raises(ValidationError, match="identity"):
        build_index(path, forbidden(), "different-corpus", BM25Config())
    with BM25Index(path) as index:
        for k in (0, -1, True):
            with pytest.raises(ValidationError):
                list(index.search(["alpha"], ("en",), k=k))


def test_rrf_uses_branch_ranks_with_no_score_scaling():
    dense = [("d", -0.1), ("both", -0.2), ("a", -0.3)]
    lexical = [("b", 10000), ("both", 1), ("z", 0.1)]
    result = reciprocal_rank_fusion(dense, lexical)
    assert result == [("both", 2 / 62), ("b", 1 / 61), ("d", 1 / 61), ("a", 1 / 63), ("z", 1 / 63)]
    weighted = HybridConfig(dense_weight=2, bm25_weight=1, rank_constant=0)
    assert reciprocal_rank_fusion(dense, lexical, weighted, k=2) == [("d", 2.0), ("both", 1.5)]
    assert reciprocal_rank_fusion(dense, []) == [("d", 1 / 61), ("both", 1 / 62), ("a", 1 / 63)]
    assert reciprocal_rank_fusion([], []) == []
    with pytest.raises(ValidationError, match="Duplicate"):
        reciprocal_rank_fusion([("a", 1), ("a", 0)], [])


def test_fusion_candidate_depth_is_applied_before_final_topk():
    dense = [(f"d{index:03}", 1.0) for index in range(101)]
    lexical = [("d100", 1.0)]
    result = reciprocal_rank_fusion(dense, lexical, HybridConfig(candidate_k=100))
    scores = dict(result)
    assert scores["d100"] == 1 / 61  # Absent from truncated dense branch.
    assert len(result) == 100
    assert "d099" not in scores


@pytest.mark.parametrize(
    "changes",
    [
        {"k1": -1},
        {"b": 1.1},
        {"k1": math.inf},
        {"b": math.nan},
        {"k1": True},
        {"analyzer": "unknown"},
    ],
)
def test_invalid_bm25_parameters(changes):
    with pytest.raises(ValidationError):
        replace(BM25Config(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_k": 99},
        {"candidate_k": 100.1},
        {"rank_constant": -1},
        {"rank_constant": True},
        {"dense_weight": 0},
        {"bm25_weight": math.inf},
        {"dense_weight": "one"},
    ],
)
def test_invalid_hybrid_parameters(changes):
    with pytest.raises(ValidationError):
        replace(HybridConfig(), **changes)
