from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from .config import RuntimeConfig, pinned
from .errors import ValidationError


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    revision: str
    dimension: int
    max_length: int
    pooling: str
    backend: str = "sentence_transformers"
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        pinned(self.revision, f"model {self.alias}")
        if self.dimension < 1 or self.max_length < 2:
            raise ValidationError(f"{self.alias}: invalid dimension or maximum input length")
        if self.backend not in {"sentence_transformers", "transformers_cls"}:
            raise ValidationError(f"{self.alias}: unsupported encoder backend {self.backend}")


def load_registry(path: Path | None = None) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(
        files("clir_benchmark").joinpath("models.yaml").read_text(encoding="utf-8")
    )
    if path:
        custom = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(custom, dict):
            raise ValidationError(
                "Model registry must be a mapping of aliases to model specifications"
            )
        for name, overrides in custom.items():
            raw[name] = {**raw.get(name, {}), **overrides}
    return {name: ModelSpec(alias=name, **spec) for name, spec in raw.items()}


@dataclass(frozen=True)
class InputAudit:
    input_tokens: int
    encoded_tokens: int
    truncated: bool
    visible_ranges: tuple[tuple[int, int], ...]


@dataclass
class EncodedBatch:
    vectors: np.ndarray
    audits: list[InputAudit]


class Encoder(Protocol):
    spec: ModelSpec
    metadata: dict

    def encode(self, texts: list[str], kind: str) -> EncodedBatch: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValidationError("Encoder returned nonfinite or non-matrix embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValidationError("Encoder returned zero-length embeddings")
    return vectors / norms


def audit_inputs(
    tokenizer: Any, texts: list[str], prefix: str, max_length: int
) -> list[InputAudit]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValidationError("Truncation auditing requires a fast tokenizer with source offsets")
    audits = []
    for text in texts:
        formatted = prefix + text
        full = tokenizer(
            formatted, add_special_tokens=True, truncation=False, return_offsets_mapping=True
        )
        encoded = tokenizer(
            formatted,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ranges = []
        # Offset intervals retain gaps for normalized-away characters. Whitespace
        # between surviving tokens is still visible as part of the source slice;
        # only the retained content prefix/suffix defines truncation exposure.
        for start, end in encoded["offset_mapping"]:
            start, end = max(0, start - len(prefix)), min(len(text), end - len(prefix))
            if end > start:
                ranges.append((start, end))
        truncated = len(full["input_ids"]) > len(encoded["input_ids"])
        if not truncated:
            visible = ((0, len(text)),)
        elif ranges:
            first, last = min(start for start, _ in ranges), max(end for _, end in ranges)
            # Byte-fallback/Unicode tokens can share character offsets. If only
            # some tokens for a character survived, that character is not fully
            # exposed. Count occurrences instead of comparing offset sets.
            all_offsets = Counter(
                tuple(pair) for pair in full["offset_mapping"] if pair[1] > pair[0]
            )
            retained = Counter(
                tuple(pair) for pair in encoded["offset_mapping"] if pair[1] > pair[0]
            )
            omitted = list((all_offsets - retained).elements())
            if getattr(tokenizer, "truncation_side", "right") == "right":
                boundaries = [start - len(prefix) for start, end in omitted if end > len(prefix)]
                last = max(0, min([last, *boundaries]))
                visible = ((0, last),) if last > 0 else ()
            else:
                boundaries = [end - len(prefix) for start, end in omitted if end > len(prefix)]
                first = min(len(text), max([first, *boundaries]))
                visible = ((first, len(text)),) if first < len(text) else ()
        else:
            visible = ()
        audits.append(
            InputAudit(
                len(full["input_ids"]),
                len(encoded["input_ids"]),
                truncated,
                visible,
            )
        )
    return audits


class HuggingFaceEncoder:
    def __init__(self, spec: ModelSpec, runtime: RuntimeConfig, token: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ValidationError(
                "Embedding requires optional model dependencies: uv sync --extra models"
            ) from exc
        self.spec = spec
        self.runtime = runtime
        self.torch = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if runtime.device == "auto"
            else runtime.device
        )
        if spec.repo_id == "google/embeddinggemma-300m" and runtime.dtype == "float16":
            raise ValidationError(
                "EmbeddingGemma does not support float16 activations; use float32 or bfloat16"
            )
        dtype = getattr(torch, runtime.dtype)
        if spec.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ValidationError(
                    "Install optional dependencies: uv sync --extra models"
                ) from exc
            self.model = SentenceTransformer(
                spec.repo_id,
                revision=spec.revision,
                token=token,
                device=self.device,
                trust_remote_code=spec.trust_remote_code,
                model_kwargs={"torch_dtype": dtype},
                tokenizer_kwargs={"use_fast": True},
            )
            self.model.max_seq_length = spec.max_length
            self.model.default_prompt_name = None
            self.tokenizer = self.model.tokenizer
            # ST tokenization must not strip or lowercase text outside the model
            # tokenizer; forward the explicitly tokenized tensors below instead.
            if self.model.get_sentence_embedding_dimension() != spec.dimension:
                raise ValidationError(f"{spec.alias}: checkpoint dimension does not match registry")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.repo_id, revision=spec.revision, token=token, use_fast=True
            )
            self.model = AutoModel.from_pretrained(
                spec.repo_id, revision=spec.revision, token=token, torch_dtype=dtype
            ).to(self.device)
        if not self.tokenizer.is_fast:
            raise ValidationError(f"{spec.alias}: fast tokenizer required for truncation audit")
        self.tokenizer.truncation_side = "right"
        if spec.pooling == "last_token":
            self.tokenizer.padding_side = "left"
        self.model.eval()
        self.metadata = {
            "model": asdict(spec),
            "device": self.device,
            "dtype": runtime.dtype,
            "truncation_side": "right",
            "padding_side": self.tokenizer.padding_side,
            "normalization": "l2",
            "embedding_dtype": "float32",
            "title_policy": "no added source titles",
        }

    def encode(self, texts: list[str], kind: str) -> EncodedBatch:
        if kind not in {"query", "document"}:
            raise ValidationError(f"Unknown embedding input kind: {kind}")
        prefix = self.spec.query_prefix if kind == "query" else self.spec.document_prefix
        audits = audit_inputs(self.tokenizer, texts, prefix, self.spec.max_length)
        inputs = self.tokenizer(
            [prefix + text for text in texts],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            if self.spec.backend == "sentence_transformers":
                outputs = self.model(inputs)["sentence_embedding"]
            else:
                outputs = self.model(**inputs).last_hidden_state[:, 0, :]
        vectors = normalize(outputs.float().cpu().numpy())
        if vectors.shape != (len(texts), self.spec.dimension):
            raise ValidationError(
                f"{self.spec.alias}: encoder returned unexpected vector shape {vectors.shape}"
            )
        return EncodedBatch(vectors, audits)
