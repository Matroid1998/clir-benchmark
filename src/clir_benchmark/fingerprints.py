"""Process-local file fingerprints for repeated result exports.

The cache retains digests, never file contents, and invalidates on filesystem
identity or metadata changes. It does not replace the uncached hashing used by
other benchmark stages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .errors import ValidationError
from .util import file_digest

FileSignature = tuple[int, int, int, int, int]


def _signature(path: Path) -> FileSignature:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _require_unchanged(path: Path, expected: FileSignature) -> None:
    try:
        actual = _signature(path)
    except OSError as exc:
        raise ValidationError(f"File changed while hashing: {path}") from exc
    if actual != expected:
        raise ValidationError(f"File changed while hashing: {path}")


@lru_cache(maxsize=4096)
def _cached_digest(resolved_path: str, signature: FileSignature) -> str:
    path = Path(resolved_path)
    _require_unchanged(path, signature)
    result = file_digest(path)
    _require_unchanged(path, signature)
    return result


def cached_file_digest(path: Path) -> str:
    """Hash a stable file once per process and filesystem signature.

    Device/inode identity detects replacement; size, nanosecond modification
    time and change time detect edits, including same-size rewrites whose mtime
    was restored. Stat checks surround hashing and also verify cache hits so a
    concurrent change cannot silently publish a digest for a different version.
    """
    resolved = Path(path).resolve()
    signature = _signature(resolved)
    result = _cached_digest(str(resolved), signature)
    _require_unchanged(resolved, signature)
    return result
