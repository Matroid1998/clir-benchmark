from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pytest

from clir_benchmark.chunking import Chunker
from clir_benchmark.config import ChunkingConfig
from clir_benchmark.encoders import EncodedBatch, InputAudit, normalize
from clir_benchmark.types import Query, SourceUnit


class CharacterTokenizer:
    """Offline fixture tokenizer; each Unicode code point is one content token."""

    is_fast = True
    truncation_side = "right"
    padding_side = "right"

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
        **kwargs,
    ):
        indices = list(range(len(text)))
        if truncation:
            capacity = max_length - (2 if add_special_tokens else 0)
            indices = indices[:capacity] if self.truncation_side == "right" else indices[-capacity:]
        ids = [ord(text[index]) + 1000 for index in indices]
        offsets = [(index, index + 1) for index in indices]
        if add_special_tokens:
            ids, offsets = [1, *ids, 2], [(0, 0), *offsets, (0, 0)]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FixtureEncoder:
    calls = 0

    def __init__(self, spec, runtime, token=None):
        self.spec = spec
        self.metadata = {
            "model": asdict(spec),
            "fixture": True,
            "device": "cpu",
            "dtype": "float32",
        }

    def encode(self, texts, kind):
        type(self).calls += 1
        vectors = np.zeros((len(texts), self.spec.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            raw = hashlib.sha256(text.encode()).digest()
            vectors[index, : min(len(raw), self.spec.dimension)] = list(raw)[: self.spec.dimension]
        return EncodedBatch(
            normalize(vectors),
            [InputAudit(len(text) + 2, len(text) + 2, False, ((0, len(text)),)) for text in texts],
        )


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def chunker(tokenizer):
    return Chunker(tokenizer, ChunkingConfig(max_tokens=12, overlap_tokens=3))


def source(unit_id="a:en", text="alpha beta", language="en", source_id="a", corpus="eurlex"):
    return SourceUnit(unit_id, corpus, source_id, "parent", language, text, "article")


def query(**changes):
    fields = dict(
        id="q1",
        corpus="eurlex",
        question="Find alpha",
        question_language="en",
        context_language="en",
        answer="alpha",
        evidence="",
        source_unit_id="a:en",
        involved_source_unit_ids=("a:en",),
        generator_model_id="generator-a",
        persona="lookup",
        task_id="task-a",
        is_best=None,
        candidate_rank=1,
        total_score=10.0,
    )
    fields.update(changes)
    return Query(**fields)
