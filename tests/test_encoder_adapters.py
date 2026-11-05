from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import CharacterTokenizer

from clir_benchmark.config import RuntimeConfig
from clir_benchmark.encoders import HuggingFaceEncoder, load_registry
from clir_benchmark.errors import ValidationError


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, device):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return Tensor(self.value[item])


class BatchTokenizer(CharacterTokenizer):
    last_batch = None

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return super().__call__(text, **kwargs)
        self.last_batch = text
        inputs = [
            super(BatchTokenizer, self).__call__(value, **kwargs)["input_ids"] for value in text
        ]
        width = max(map(len, inputs))
        return {
            "input_ids": Tensor([row + [0] * (width - len(row)) for row in inputs]),
            "attention_mask": Tensor([[1] * len(row) + [0] * (width - len(row)) for row in inputs]),
        }


@pytest.fixture
def model_stubs(monkeypatch):
    import sys

    tokenizer = BatchTokenizer()
    state = {}

    class SentenceModel:
        def __init__(self, repo_id, **kwargs):
            state["loaded"] = {"repo_id": repo_id, **kwargs}
            self.tokenizer = tokenizer

        def get_sentence_embedding_dimension(self):
            return 4

        def eval(self):
            return self

        def __call__(self, features):
            # This is the checkpoint module stack's result, not just a mean over
            # hidden states. The adapter must preserve its final projection.
            return {"sentence_embedding": Tensor([[3, 4, 0, 0]] * len(features["input_ids"].value))}

    class TransformerModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **features):
            batch = len(features["input_ids"].value)
            return SimpleNamespace(last_hidden_state=Tensor([[[3, 4, 0, 0], [0, 0, 8, 0]]] * batch))

    def load_transformer(repo_id, **kwargs):
        state["loaded"] = {"repo_id": repo_id, **kwargs}
        return TransformerModel()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=load_transformer),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceModel)
    )
    return tokenizer, state


def test_sentence_transformer_preserves_projection_and_applies_prompt_once(model_stubs):
    tokenizer, state = model_stubs
    spec = replace(
        load_registry()["e5-large-instruct"],
        dimension=4,
        max_length=12,
        query_prefix="Q:",
        document_prefix="D:",
    )
    encoder = HuggingFaceEncoder(spec, RuntimeConfig())
    result = encoder.encode(["  alpha "], "query")
    assert tokenizer.last_batch == ["Q:  alpha "]
    assert state["loaded"]["revision"] == spec.revision
    assert state["loaded"]["model_kwargs"]["torch_dtype"] == "float32"
    assert result.vectors[0] == pytest.approx([0.6, 0.8, 0, 0])
    assert result.audits[0].input_tokens == 12
    encoder.encode(["  alpha "], "document")
    assert tokenizer.last_batch == ["D:  alpha "]


def test_sapbert_uses_cls_not_mean(model_stubs):
    _, state = model_stubs
    spec = replace(load_registry()["SapBERT"], dimension=4)
    encoder = HuggingFaceEncoder(spec, RuntimeConfig())
    assert encoder.encode(["alpha"], "document").vectors[0] == pytest.approx([0.6, 0.8, 0, 0])
    assert state["loaded"]["revision"] == spec.revision


def test_embeddinggemma_rejects_float16(model_stubs):
    with pytest.raises(ValidationError, match="float16"):
        HuggingFaceEncoder(load_registry()["embeddinggemma"], RuntimeConfig(dtype="float16"))
