from clir_benchmark.rerank_cache import SQLitePairCache
from clir_benchmark.util import digest


def test_compact_cache_preserves_first_measurement_and_concurrent_attempts(tmp_path):
    identity = {
        "scoring": "fixture",
        "query": "q",
        "query_language": "en",
        "document": "d",
        "document_language": "fr",
    }
    key = digest(identity)
    first = {"identity": identity, "score": 0.7, "seconds_amortized": 2.0}
    second = {**first, "seconds_amortized": 3.0}
    cache, concurrent = SQLitePairCache(tmp_path), SQLitePairCache(tmp_path)
    try:
        assert cache.get(key, identity) is None
        assert concurrent.get(key, identity) is None
        assert cache.put_batch([(key, first)]) == {key: first}
        assert concurrent.put_batch([(key, second)]) == {key: first}
        assert concurrent.get(key, identity) == first
        assert (
            next(iter(cache.connections.values()))
            .execute("SELECT count(*) FROM concurrent_attempts")
            .fetchone()[0]
            == 1
        )
        assert len(list(tmp_path.glob("*.sqlite3"))) == 1
    finally:
        cache.close()
        concurrent.close()
    reopened = SQLitePairCache(tmp_path)
    try:
        assert reopened.get(key, identity) == first
        assert reopened.get(digest({**identity, "document": "other"}), identity) is None
    finally:
        reopened.close()
