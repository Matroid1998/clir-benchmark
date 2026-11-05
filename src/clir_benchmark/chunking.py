from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
from typing import Any

from .config import ChunkingConfig
from .errors import ValidationError
from .types import Chunk, SourceUnit
from .util import digest


def load_tokenizer(config: ChunkingConfig, token: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_id, revision=config.revision, token=token, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValidationError("Exact source offsets require a fast tokenizer")
    return tokenizer


class Chunker:
    """Windows over source token offsets, followed by validation of exact text slices."""

    def __init__(self, tokenizer: Any, config: ChunkingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.special_count = tokenizer.num_special_tokens_to_add(pair=False)
        self.capacity = config.max_tokens - self.special_count
        if self.capacity <= config.overlap_tokens:
            raise ValidationError("Overlap must be smaller than the actual content-token budget")
        self.manifest = {
            **asdict(config),
            "special_token_count": self.special_count,
            "content_token_budget": self.capacity,
            "offset_unit": "unicode_code_points",
            "end_exclusive": True,
            "overlap_measurement": "nominal source tokens; actual overlap re-tokenized without specials",
        }
        self.fingerprint = digest(self.manifest)

    def count(self, text: str, *, special: bool = True) -> int:
        return len(self.tokenizer(text, add_special_tokens=special, truncation=False)["input_ids"])

    def split(self, unit: SourceUnit) -> list[Chunk]:
        text = unit.text
        full_count = self.count(text)
        if full_count <= self.config.max_tokens:
            windows = [(0, len(text), full_count)]
        else:
            encoded = self.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
            )
            offsets = encoded["offset_mapping"]
            # Only cut between offset intervals. Several tokens can share one Unicode
            # character; splitting those intervals would lose or duplicate fragments.
            cuts = [(0, 0)]
            previous_end = 0
            for index, (start, end) in enumerate(offsets):
                if not 0 <= start <= end <= len(text):
                    raise ValidationError(f"Invalid tokenizer offsets for {unit.id}")
                if start > cuts[-1][0] and start >= previous_end:
                    cuts.append((start, index))
                previous_end = max(previous_end, end)
            if cuts[-1][0] != len(text):
                cuts.append((len(text), len(offsets)))
            token_indices = [item[1] for item in cuts]
            windows = []
            start_cut = 0
            while start_cut < len(cuts) - 1:
                start, token_start = cuts[start_cut]
                end_cut = bisect_right(token_indices, token_start + self.capacity) - 1
                end_cut = max(start_cut + 1, min(end_cut, len(cuts) - 1))
                while end_cut > start_cut:
                    end = cuts[end_cut][0]
                    count = self.count(text[start:end])
                    if count <= self.config.max_tokens:
                        break
                    end_cut -= 1
                if end_cut == start_cut:
                    raise ValidationError(
                        f"Cannot fit a Unicode-safe token window for {unit.id} at character {start}"
                    )
                windows.append((start, end, count))
                if end == len(text):
                    break
                wanted = cuts[end_cut][1] - self.config.overlap_tokens
                next_cut = bisect_right(token_indices, wanted) - 1
                # A heavily shortened window may be smaller than the overlap. Always
                # advance, while retaining complete coverage of the original string.
                start_cut = min(end_cut, max(start_cut + 1, next_cut))
        result = []
        previous_end = 0
        text_hash = digest(text)
        for index, (start, end, count) in enumerate(windows):
            overlap_end = max(start, previous_end)
            chunk_id = "c_" + digest(
                {
                    "corpus": unit.corpus,
                    "unit": unit.id,
                    "source_id": unit.source_id,
                    "parent_id": unit.parent_id,
                    "language": unit.language,
                    "text_hash": text_hash,
                    "start": start,
                    "end": end,
                    "chunking": self.fingerprint,
                }
            )
            result.append(
                Chunk(
                    id=chunk_id,
                    source_unit_id=unit.id,
                    source_id=unit.source_id,
                    parent_id=unit.parent_id,
                    corpus=unit.corpus,
                    language=unit.language,
                    text=text[start:end],
                    start=start,
                    end=end,
                    token_count=count,
                    index=index,
                    overlap_chars=overlap_end - start,
                    overlap_tokens=self.count(text[start:overlap_end], special=False),
                    chunking_hash=self.fingerprint,
                )
            )
            previous_end = end
        return result
