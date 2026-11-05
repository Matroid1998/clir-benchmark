from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .errors import ValidationError


def json_dumps(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def object_field(value: Any, name: str) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{name}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{name}: expected an object")
    return value


def hf_token(env_file: Path | None = None) -> str | None:
    from dotenv import dotenv_values

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    values = dotenv_values(env_file or Path.cwd() / ".env")
    return values.get("HF_TOKEN") or None


def versions() -> dict[str, str]:
    result = {}
    for name in (
        "clir-benchmark",
        "numpy",
        "pyarrow",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "torch",
        "sentence-transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


@contextmanager
def new_directory(destination: Path) -> Iterator[Path]:
    """Publish a complete immutable artifact directory, never an incomplete stage."""
    if destination.exists():
        raise ValidationError(
            f"Artifact already exists: {destination}. Use a new output directory or configuration."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise ValidationError(f"Artifact was created concurrently: {destination}")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
