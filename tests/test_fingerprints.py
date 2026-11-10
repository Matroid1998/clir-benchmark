import os

import pytest

from clir_benchmark import fingerprints
from clir_benchmark.errors import ValidationError
from clir_benchmark.util import file_digest


@pytest.fixture(autouse=True)
def empty_fingerprint_cache():
    fingerprints._cached_digest.cache_clear()
    yield
    fingerprints._cached_digest.cache_clear()


@pytest.fixture
def counted_hash(monkeypatch):
    calls = []

    def count(path):
        calls.append(path)
        return file_digest(path)

    monkeypatch.setattr(fingerprints, "file_digest", count)
    return calls


def test_unchanged_file_and_resolved_path_reuse_digest(tmp_path, counted_hash):
    path = tmp_path / "vectors.npy"
    path.write_bytes(b"immutable embedding contents")
    alias = tmp_path / "alias.npy"
    alias.symlink_to(path)

    expected = file_digest(path)
    assert fingerprints.cached_file_digest(path) == expected
    assert fingerprints.cached_file_digest(path) == expected
    assert fingerprints.cached_file_digest(alias) == expected
    assert counted_hash == [path.resolve()]


def test_same_size_rewrite_with_restored_mtime_invalidates_cache(tmp_path, counted_hash):
    path = tmp_path / "vectors.npy"
    path.write_bytes(b"first contents")
    initial = fingerprints.cached_file_digest(path)
    previous = path.stat()

    path.write_bytes(b"other contents")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    current = path.stat()
    assert current.st_size == previous.st_size
    assert current.st_mtime_ns == previous.st_mtime_ns
    assert current.st_ctime_ns != previous.st_ctime_ns
    assert fingerprints.cached_file_digest(path) == file_digest(path) != initial
    assert len(counted_hash) == 2


def test_atomic_replacement_invalidates_cache(tmp_path, counted_hash):
    path = tmp_path / "vectors.npy"
    path.write_bytes(b"first contents")
    initial = fingerprints.cached_file_digest(path)
    previous = path.stat()
    replacement = tmp_path / "replacement.npy"
    replacement.write_bytes(b"other contents")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(path)

    assert path.stat().st_ino != previous.st_ino
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert fingerprints.cached_file_digest(path) == file_digest(path) != initial
    assert len(counted_hash) == 2


def test_file_changed_while_hashing_is_rejected_and_not_cached(tmp_path, monkeypatch):
    path = tmp_path / "vectors.npy"
    path.write_bytes(b"first contents")

    def mutate_after_read(value):
        result = file_digest(value)
        value.write_bytes(b"other contents")
        return result

    monkeypatch.setattr(fingerprints, "file_digest", mutate_after_read)
    with pytest.raises(ValidationError, match="File changed while hashing"):
        fingerprints.cached_file_digest(path)
    assert fingerprints._cached_digest.cache_info().currsize == 0
    monkeypatch.setattr(fingerprints, "file_digest", file_digest)
    assert fingerprints.cached_file_digest(path) == file_digest(path)


def test_file_removed_while_hashing_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "vectors.npy"
    path.write_bytes(b"contents")

    def remove_after_read(value):
        result = file_digest(value)
        value.unlink()
        return result

    monkeypatch.setattr(fingerprints, "file_digest", remove_after_read)
    with pytest.raises(ValidationError, match="File changed while hashing"):
        fingerprints.cached_file_digest(path)
    assert fingerprints._cached_digest.cache_info().currsize == 0
