from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .errors import ValidationError
from .util import digest, file_digest

BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def pinned(revision: str, label: str) -> None:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValidationError(
            f"{label}: revision must be a full 40-character commit SHA, not a branch/tag"
        )


def language_tag(value: str, label: str = "language") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*", value):
        raise ValidationError(f"{label}: expected a language tag such as en, fr, or zh-Hans")
    return value


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    revision: str
    local_dir: str | None = None

    def __post_init__(self) -> None:
        pinned(self.revision, f"dataset {self.repo_id}")
        if not isinstance(self.repo_id, str) or not self.repo_id:
            raise ValidationError("Dataset repo_id must be a nonempty string")

    def identity(self) -> dict:
        value = asdict(self)
        if self.local_dir:
            root = Path(self.local_dir)
            if not root.is_dir():
                raise ValidationError(f"Local dataset directory does not exist: {root}")
            value["files"] = {
                str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.parquet"))
            }
        return value


@dataclass(frozen=True)
class DatasetSpec:
    qa: SourceSpec
    corpus: SourceSpec


@dataclass(frozen=True)
class ChunkingConfig:
    tokenizer_id: str = "BAAI/bge-m3"
    revision: str = BGE_REVISION
    max_tokens: int = 512
    overlap_tokens: int = 64
    boundary_policy: str = "source-token-windows-v1"
    special_tokens: str = "included"

    def __post_init__(self) -> None:
        pinned(self.revision, "chunking tokenizer")
        if self.tokenizer_id != "BAAI/bge-m3":
            raise ValidationError("Benchmark chunking requires BAAI/bge-m3")
        if not 2 < self.max_tokens <= 512 or not 0 <= self.overlap_tokens < self.max_tokens - 2:
            raise ValidationError(
                "Chunking requires 2 < max_tokens <= 512 and overlap < content budget"
            )
        if self.boundary_policy != "source-token-windows-v1" or self.special_tokens != "included":
            raise ValidationError("Unsupported boundary policy or special-token accounting")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 32
    query_batch_size: int = 16
    corpus_block_size: int = 8192

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValidationError("runtime.dtype must be float32, bfloat16, or float16")
        if min(self.batch_size, self.query_batch_size, self.corpus_block_size) < 1:
            raise ValidationError("Runtime batch/block sizes must be positive")


@dataclass(frozen=True)
class Config:
    output_dir: Path
    datasets: dict[str, DatasetSpec]
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    candidate_mode: str = "best"
    settings: tuple[str, ...] = ("pairwise", "mixed")
    query_languages: tuple[str, ...] = ()
    target_languages: tuple[str, ...] = ()
    models: tuple[str, ...] = ("bge-m3",)
    evidence_file: Path | None = None
    model_registry: Path | None = None
    env_file: Path | None = None

    def corpus_key(self, corpus: str) -> str:
        return digest(
            {
                "schema": 1,
                "corpus": corpus,
                "source": self.datasets[corpus].corpus.identity(),
                "chunking": asdict(self.chunking),
            }
        )

    def corpus_dir(self, corpus: str) -> Path:
        return self.output_dir / "corpus" / corpus / self.corpus_key(corpus)

    def relevance_key(self, corpus: str) -> str:
        return digest(
            {
                "schema": 1,
                "corpus_key": self.corpus_key(corpus),
                "qa": self.datasets[corpus].qa.identity(),
                "candidate_mode": self.candidate_mode,
                "settings": self.settings,
                "query_languages": self.query_languages,
                "target_languages": self.target_languages,
                "evidence": file_digest(self.evidence_file) if self.evidence_file else None,
            }
        )

    def relevance_dir(self, corpus: str) -> Path:
        return self.output_dir / "relevance" / corpus / self.relevance_key(corpus)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(raw, dict) or raw.pop("version", None) != 1:
        raise ValidationError("Configuration must be an object with version: 1")
    raw = dict(raw)

    def resolved(value: str | None) -> Path | None:
        return (path.parent / value).resolve() if value is not None else None

    datasets = {}
    raw_datasets = raw.pop("datasets", {})
    if not isinstance(raw_datasets, dict):
        raise ValidationError("datasets must be a mapping")
    for name, spec in raw_datasets.items():
        if name not in {"eurlex", "un", "chemistry"}:
            raise ValidationError(f"Unknown corpus adapter: {name}")
        sources = {}
        if not isinstance(spec, dict) or set(spec) != {"qa", "corpus"}:
            raise ValidationError(f"{name}: provide exactly qa and corpus source specifications")
        for kind in ("qa", "corpus"):
            if not isinstance(spec[kind], dict):
                raise ValidationError(f"{name}.{kind}: expected a source specification")
            source = dict(spec[kind])
            pinned(source.get("revision"), f"{name}.{kind}")
            if source.get("local_dir"):
                source["local_dir"] = str(resolved(source["local_dir"]))
            try:
                sources[kind] = SourceSpec(**source)
            except TypeError as exc:
                raise ValidationError(
                    f"{name}.{kind}: invalid source specification: {exc}"
                ) from exc
        datasets[name] = DatasetSpec(**sources)
    if not datasets:
        raise ValidationError("At least one dataset is required")
    for name in ("evidence_file", "model_registry", "env_file"):
        if name in raw:
            raw[name] = resolved(raw[name])
    raw["output_dir"] = resolved(raw.get("output_dir", "artifacts"))
    for name in ("settings", "query_languages", "target_languages", "models"):
        if name in raw:
            if not isinstance(raw[name], list):
                raise ValidationError(f"{name} must be a list")
            raw[name] = tuple(raw[name])
    try:
        config = Config(
            datasets=datasets,
            chunking=ChunkingConfig(**raw.pop("chunking", {})),
            runtime=RuntimeConfig(**raw.pop("runtime", {})),
            **raw,
        )
    except (TypeError, KeyError) as exc:
        raise ValidationError(f"Invalid configuration: {exc}") from exc
    if config.candidate_mode not in {"best", "all"}:
        raise ValidationError("candidate_mode must be best or all")
    if not config.settings or set(config.settings) - {"pairwise", "mixed"}:
        raise ValidationError("settings must contain pairwise and/or mixed")
    for value in (*config.query_languages, *config.target_languages):
        language_tag(value)
    if not config.models or any(not isinstance(name, str) or not name for name in config.models):
        raise ValidationError("models must contain at least one nonempty alias")
    return config
