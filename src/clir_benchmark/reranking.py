"""Pinned Qwen3 causal-LM reranking with the model card's yes/no scoring contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .errors import ValidationError

if TYPE_CHECKING:
    from .config import RerankerConfig


PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def scoring_identity() -> dict:
    return {
        "schema": "qwen3-reranker-yes-no-v1",
        "prefix": PREFIX,
        "suffix": SUFFIX,
        "body_format": "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
        "score": "float32 softmax of final-position [no, yes] logits, probability of yes",
        "padding_side": "left",
        "truncation_policy": "right-truncate body tokens, reserving complete prefix and suffix",
        "source": "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
    }


@dataclass
class ScoredBatch:
    scores: list[float]
    audits: list[dict]


def format_pair(instruction: str, query: str, document: str) -> tuple[str, int, int]:
    query_prefix = f"<Instruct>: {instruction}\n<Query>: "
    document_prefix = f"{query_prefix}{query}\n<Document>: "
    return document_prefix + document, len(query_prefix), len(document_prefix)


def prepare_pair(tokenizer, config, query: str, document: str, prefix_ids, suffix_ids):
    """Tokenize once and audit retained source characters, including Unicode cut boundaries."""
    if not isinstance(query, str) or not isinstance(document, str):
        raise ValidationError("Reranking requires query/document text strings")
    body, query_start, document_start = format_pair(config.instruction, query, document)
    full = tokenizer(
        body,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    budget = config.max_length - len(prefix_ids) - len(suffix_ids)
    if budget < 1:
        raise ValidationError("reranker.max_length must leave room after the prompt wrapper")
    all_ids, offsets = full["input_ids"], full["offset_mapping"]
    if len(all_ids) != len(offsets):
        raise ValidationError("Reranker tokenizer returned inconsistent source offsets")
    body_ids = all_ids[:budget]
    truncated = len(body_ids) != len(all_ids)
    # If byte-fallback tokens share a character offset, the earliest omitted token
    # excludes that entire character. Only complete characters count as exposed.
    omitted_starts = [start for start, end in offsets[budget:] if end > start]
    if not truncated:
        cutoff = len(body)
    elif omitted_starts:
        cutoff = min(omitted_starts)
    else:
        cutoff = max((end for start, end in offsets[:budget] if end > start), default=0)

    def visible(text, start):
        end = max(0, min(len(text), cutoff - start))
        return [[0, end]] if end > 0 else []

    query_visible = visible(query, query_start)
    document_visible = visible(document, document_start)
    query_end = query_visible[-1][1] if query_visible else 0
    document_end = document_visible[-1][1] if document_visible else 0
    ids = [*prefix_ids, *body_ids, *suffix_ids]
    audit = {
        "input_tokens": len(prefix_ids) + len(all_ids) + len(suffix_ids),
        "encoded_tokens": len(ids),
        "truncated": truncated,
        "query_visible_ranges": query_visible,
        "document_visible_ranges": document_visible,
        "query_truncated": query_end < len(query),
        "document_truncated": document_end < len(document),
    }
    return ids, audit


class Qwen3Reranker:
    """Load only when explicitly instantiated; module import never imports model runtimes."""

    def __init__(self, config: RerankerConfig, token: str | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValidationError(
                "Reranking requires optional model dependencies: uv sync --extra models"
            ) from exc
        self.config, self.torch = config, torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.repo_id,
            revision=config.revision,
            token=token,
            use_fast=True,
            padding_side="left",
            trust_remote_code=False,
        )
        if not self.tokenizer.is_fast:
            raise ValidationError("Qwen3 reranking requires a fast tokenizer for truncation audits")
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValidationError("Qwen3 reranker tokenizer requires a padding or EOS token")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if config.max_length <= len(self.prefix_ids) + len(self.suffix_ids):
            raise ValidationError("reranker.max_length must leave room after the prompt wrapper")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if (
            self.no_id is None
            or self.yes_id is None
            or self.no_id == self.yes_id
            or self.tokenizer.encode("no", add_special_tokens=False) != [self.no_id]
            or self.tokenizer.encode("yes", add_special_tokens=False) != [self.yes_id]
        ):
            raise ValidationError("Qwen3 reranker requires distinct single-token yes/no labels")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.repo_id,
            revision=config.revision,
            token=token,
            torch_dtype=getattr(torch, config.dtype),
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()
        self.metadata = {
            "configuration": asdict(config),
            "device": self.device,
            "dtype": config.dtype,
            "methodology": scoring_identity(),
            "logits_to_keep": 1,
            "use_cache": False,
            "token_yes_id": self.yes_id,
            "token_no_id": self.no_id,
        }

    def synchronize(self) -> None:
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif str(self.device).startswith("mps"):
            self.torch.mps.synchronize()

    def score(self, pairs: list[tuple[str, str]]) -> ScoredBatch:
        if not pairs:
            return ScoredBatch([], [])
        token_rows, audits = [], []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValidationError("Reranking expects query/document pairs")
            ids, audit = prepare_pair(
                self.tokenizer, self.config, *pair, self.prefix_ids, self.suffix_ids
            )
            token_rows.append(ids)
            audits.append(audit)
        inputs = self.tokenizer.pad({"input_ids": token_rows}, padding=True, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        self.synchronize()
        with self.torch.inference_mode():
            # Qwen3ForCausalLM supports logits_to_keep in the pinned Transformers
            # range. Compute only the final position to avoid sequence-wide logits.
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            labels = logits[:, [self.no_id, self.yes_id]].float()
            scores = self.torch.nn.functional.softmax(labels, dim=1)[:, 1].cpu().tolist()
        self.synchronize()
        if len(scores) != len(pairs) or any(not math.isfinite(score) for score in scores):
            raise ValidationError("Qwen3 reranker returned invalid relevance scores")
        return ScoredBatch(scores, audits)
