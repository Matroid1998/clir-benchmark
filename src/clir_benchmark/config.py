from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import yaml

from .errors import ValidationError
from .util import digest, file_digest

BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BASELINES = (
    "dense",
    "bm25",
    "hybrid",
    "translate-dense",
    "translate-bm25",
    "translate-hybrid",
    "hyde-dense",
)


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
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    analyzer: str = "unicode-words-cjk-v1"

    def __post_init__(self) -> None:
        for name, value in (("k1", self.k1), ("b", self.b)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValidationError(f"bm25.{name} must be a finite number")
        if self.k1 < 0 or not 0 <= self.b <= 1:
            raise ValidationError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        if self.analyzer != "unicode-words-cjk-v1":
            raise ValidationError("Unsupported BM25 analyzer; use unicode-words-cjk-v1")


@dataclass(frozen=True)
class HybridConfig:
    candidate_k: int = 1000
    rank_constant: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("candidate_k", self.candidate_k, 100),
            ("rank_constant", self.rank_constant, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValidationError(f"hybrid.{name} must be an integer >= {minimum}")
        for name, value in (("dense_weight", self.dense_weight), ("bm25_weight", self.bm25_weight)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValidationError(f"hybrid.{name} must be a positive finite number")


@dataclass(frozen=True)
class GenerationConfig:
    section: ClassVar[str] = "generation"
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    response_format: str = "json_schema"
    request_options: dict = field(default_factory=dict)
    timeout_seconds: float = 120
    input_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None

    def __post_init__(self) -> None:
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValidationError(f"{self.section}.model must be a nonempty model ID or null")
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str):
                raise ValidationError(f"{self.section}.endpoint must be a URL or null")
            try:
                url = urlsplit(self.endpoint)
            except ValueError as exc:
                raise ValidationError(
                    f"{self.section}.endpoint must be a valid HTTP(S) URL"
                ) from exc
            if (
                url.scheme not in {"https", "http"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValidationError(
                    f"{self.section}.endpoint must be an HTTP(S) URL without credentials, query, or fragment"
                )
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)
        ):
            raise ValidationError(
                f"{self.section}.api_key_env must be an environment variable name or null"
            )
        if self.response_format not in {"json_schema", "json_object", "text"}:
            raise ValidationError(
                f"{self.section}.response_format must be json_schema, json_object, or text"
            )
        allowed = {
            "temperature",
            "top_p",
            "seed",
            "max_completion_tokens",
            "max_tokens",
            "reasoning_effort",
        }
        if not isinstance(self.request_options, dict) or self.request_options.keys() - allowed:
            raise ValidationError(f"{self.section}.request_options allows only {sorted(allowed)}")
        try:
            digest(self.request_options)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{self.section}.request_options must contain finite JSON values"
            ) from exc
        for name in (
            "timeout_seconds",
            "input_usd_per_million",
            "cached_input_usd_per_million",
            "output_usd_per_million",
        ):
            value = getattr(self, name)
            if value is None and name != "timeout_seconds":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "timeout_seconds" and value == 0)
            ):
                raise ValidationError(
                    f"{self.section}.{name} must be a finite nonnegative number (timeout > 0)"
                )


@dataclass(frozen=True)
class TranslationConfig(GenerationConfig):
    section: ClassVar[str] = "translation"
    api_key_env: str | None = "TRANSLATION_API_KEY"


@dataclass(frozen=True)
class HyDEConfig(GenerationConfig):
    section: ClassVar[str] = "hyde"
    api_key_env: str | None = "HYDE_API_KEY"
    output_languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.output_languages, (list, tuple)):
            raise ValidationError("hyde.output_languages must be a list of language tags")
        for language in self.output_languages:
            language_tag(language, "HyDE output language")
        if len(set(self.output_languages)) != len(self.output_languages):
            raise ValidationError("hyde.output_languages cannot contain duplicates")
        object.__setattr__(self, "output_languages", tuple(self.output_languages))


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
    baselines: tuple[str, ...] = ("dense",)
    bm25: BM25Config = field(default_factory=BM25Config)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)

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
                "schema": 2,
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
    for name in ("settings", "query_languages", "target_languages", "models", "baselines"):
        if name in raw:
            if not isinstance(raw[name], list):
                raise ValidationError(f"{name} must be a list")
            raw[name] = tuple(raw[name])
    try:
        config = Config(
            datasets=datasets,
            chunking=ChunkingConfig(**raw.pop("chunking", {})),
            runtime=RuntimeConfig(**raw.pop("runtime", {})),
            bm25=BM25Config(**raw.pop("bm25", {})),
            hybrid=HybridConfig(**raw.pop("hybrid", {})),
            translation=TranslationConfig(**raw.pop("translation", {})),
            hyde=HyDEConfig(**raw.pop("hyde", {})),
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
    if not config.baselines or any(value not in BASELINES for value in config.baselines):
        raise ValidationError(f"baselines must contain one or more of {BASELINES}")
    if len(set(config.baselines)) != len(config.baselines):
        raise ValidationError("baselines cannot contain duplicates")
    return config
