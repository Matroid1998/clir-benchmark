from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import CharacterTokenizer

from clir_benchmark.config import RerankerConfig
from clir_benchmark.errors import ValidationError
from clir_benchmark.reranking import (
    PREFIX,
    SUFFIX,
    Qwen3Reranker,
    format_pair,
    prepare_pair,
    scoring_identity,
)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, device):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()

    def __getitem__(self, item):
        return Tensor(self.value[item])


class RerankerTokenizer(CharacterTokenizer):
    pad_token_id = 0
    eos_token = "eos"

    def encode(self, text, add_special_tokens=False):
        if text == PREFIX:
            return [101]
        if text == SUFFIX:
            return [102, 103]
        if text in {"no", "yes"}:
            return [{"no": 3, "yes": 7}[text]]
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def convert_tokens_to_ids(self, text):
        return {"no": 3, "yes": 7}[text]

    def pad(self, values, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        assert self.padding_side == "left"
        rows = values["input_ids"]
        width = max(map(len, rows))
        self.last_input = rows
        return {
            "input_ids": Tensor([[0] * (width - len(row)) + row for row in rows]),
            "attention_mask": Tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }


@pytest.fixture
def causal_model(monkeypatch):
    import sys

    tokenizer, state = RerankerTokenizer(), {"sync": [], "inference_active": False}

    @contextmanager
    def inference_mode():
        state["inference_active"] = True
        try:
            yield
        finally:
            state["inference_active"] = False

    class Model:
        def to(self, device):
            state["device"] = device
            return self

        def eval(self):
            state["eval"] = True
            return self

        def __call__(self, **kwargs):
            assert state["inference_active"]
            state["forward"] = kwargs
            # Earlier positions and other vocabulary entries must not affect the
            # score. This would defeat full-vocabulary or first-position scoring.
            batch = len(kwargs["input_ids"].value)
            logits = np.zeros((batch, 2, 8), dtype=np.float64)
            logits[:, :, 0] = 100
            logits[:, 0, 7] = 30
            for row in range(batch):
                logits[row, -1, 3] = 0
                logits[row, -1, 7] = 2 if row % 2 == 0 else -2
            return SimpleNamespace(logits=Tensor(logits))

    def load_model(repo_id, **kwargs):
        state["model_load"] = {"repo_id": repo_id, **kwargs}
        return Model()

    def load_tokenizer(repo_id, **kwargs):
        state["tokenizer_load"] = {"repo_id": repo_id, **kwargs}
        return tokenizer

    def softmax(tensor, dim):
        assert tensor.value.dtype == np.float32
        values = tensor.value - tensor.value.max(axis=dim, keepdims=True)
        exponentials = np.exp(values)
        return Tensor(exponentials / exponentials.sum(axis=dim, keepdims=True))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            float16="float16",
            cuda=SimpleNamespace(
                is_available=lambda: False,
                synchronize=lambda device: state["sync"].append(device),
            ),
            mps=SimpleNamespace(synchronize=lambda: state["sync"].append("mps")),
            inference_mode=inference_mode,
            nn=SimpleNamespace(functional=SimpleNamespace(softmax=softmax)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    return tokenizer, state


def test_pinned_causal_model_scores_yes_no_at_final_position(causal_model):
    tokenizer, state = causal_model
    config = RerankerConfig()
    scorer = Qwen3Reranker(config, token="fixture-token")
    result = scorer.score([("Find alpha", "alpha"), ("Find beta", "longer irrelevant document")])
    assert result.scores == pytest.approx([0.880797, 0.119203])
    assert state["forward"]["logits_to_keep"] == 1
    assert state["forward"]["use_cache"] is False
    assert state["eval"] is True
    assert state["model_load"]["revision"] == config.revision
    assert state["tokenizer_load"]["revision"] == config.revision
    assert state["model_load"]["torch_dtype"] == "float32"
    assert state["tokenizer_load"]["trust_remote_code"] is False
    assert state["model_load"]["trust_remote_code"] is False
    assert state["model_load"]["token"] == "fixture-token"
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "right"
    assert state["forward"]["attention_mask"].value[0, 0] == 0
    assert np.all(state["forward"]["attention_mask"].value[:, -1] == 1)
    assert np.all(state["forward"]["input_ids"].value[:, -2:] == [102, 103])
    assert result.audits[0]["document_visible_ranges"] == [[0, len("alpha")]]
    assert result.audits[0]["query_visible_ranges"] == [[0, len("Find alpha")]]
    assert not result.audits[0]["truncated"]
    assert scorer.metadata["methodology"] == scoring_identity()


def test_prompt_wrapper_is_preserved_during_right_truncation(causal_model):
    tokenizer, _ = causal_model
    config = RerankerConfig(max_length=128, instruction="Find relevant passages")
    query, document = "Quelle règle?", "règle " * 100
    scorer = Qwen3Reranker(config)
    result = scorer.score([(query, document)])
    body, query_start, document_start = format_pair(config.instruction, query, document)
    assert tokenizer.last_input[0] == [101, *[ord(char) + 1000 for char in body[:125]], 102, 103]
    audit = result.audits[0]
    assert audit["input_tokens"] == len(body) + 3
    assert audit["encoded_tokens"] == 128
    assert audit["truncated"] is True
    assert audit["query_visible_ranges"] == [[0, len(query)]]
    assert audit["query_truncated"] is False
    assert audit["document_visible_ranges"] == [[0, 125 - document_start]]
    assert audit["document_truncated"] is True
    assert query_start < document_start


def test_partial_unicode_character_is_not_claimed_as_visible():
    class ByteTokenizer:
        def __call__(self, text, **kwargs):
            # Two byte-fallback tokens represent the single document character.
            offset = len(text) - 1
            return {
                "input_ids": [10, 11, 12],
                "offset_mapping": [(0, offset), (offset, offset + 1), (offset, offset + 1)],
            }

    config = SimpleNamespace(instruction="find", max_length=4)
    ids, audit = prepare_pair(ByteTokenizer(), config, "q", "é", [1], [2])
    assert ids == [1, 10, 11, 2]
    assert audit["document_visible_ranges"] == []
    assert audit["document_truncated"] is True


def test_long_query_can_truncate_document_but_audit_records_loss(causal_model):
    tokenizer, _ = causal_model
    scorer = Qwen3Reranker(RerankerConfig(max_length=128))
    result = scorer.score([("q" * 1000, "document")])
    assert result.audits[0]["query_truncated"]
    assert result.audits[0]["document_visible_ranges"] == []
    assert result.audits[0]["document_truncated"]
    assert tokenizer.last_input[0][-2:] == [102, 103]


@pytest.mark.parametrize("device", ["cuda:1", "mps"])
def test_device_synchronization_and_dtype_are_explicit(causal_model, device):
    _, state = causal_model
    scorer = Qwen3Reranker(RerankerConfig(device=device, dtype="bfloat16"))
    scorer.score([("q", "d")])
    assert state["device"] == device
    assert state["model_load"]["torch_dtype"] == "bfloat16"
    assert state["sync"] == [device, device]


def test_empty_batch_never_executes_model(causal_model):
    _, state = causal_model
    result = Qwen3Reranker(RerankerConfig()).score([])
    assert result.scores == [] and result.audits == []
    assert "forward" not in state


def test_nonfinite_scores_are_rejected(causal_model, monkeypatch):
    scorer = Qwen3Reranker(RerankerConfig())
    monkeypatch.setattr(
        scorer.torch.nn.functional,
        "softmax",
        lambda *args, **kwargs: Tensor([[0, np.nan]]),
    )
    with pytest.raises(ValidationError, match="invalid relevance scores"):
        scorer.score([("q", "d")])


def test_unknown_label_tokenization_is_rejected_before_weights_load(causal_model, monkeypatch):
    tokenizer, state = causal_model
    original = tokenizer.encode
    monkeypatch.setattr(
        tokenizer,
        "encode",
        lambda text, **kwargs: [7, 8] if text == "yes" else original(text, **kwargs),
    )
    with pytest.raises(ValidationError, match="single-token"):
        Qwen3Reranker(RerankerConfig())
    assert "model_load" not in state


def test_slow_tokenizer_fails_before_weights_load(causal_model):
    tokenizer, state = causal_model
    tokenizer.is_fast = False
    with pytest.raises(ValidationError, match="fast tokenizer"):
        Qwen3Reranker(RerankerConfig())
    assert "model_load" not in state


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"repo_id": "other/model"},
        {"candidate_k": 0},
        {"candidate_k": 101},
        {"candidate_k": True},
        {"max_length": 32769},
        {"max_length": 127},
        {"batch_size": 0},
        {"dtype": "int8"},
        {"device": " "},
        {"instruction": ""},
    ],
)
def test_reranker_config_rejects_invalid_or_unpinned_settings(changes):
    with pytest.raises(ValidationError):
        replace(RerankerConfig(), **changes)
