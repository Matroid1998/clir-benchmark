"""Sentence-first source slices constrained by every retriever's formatted input."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from .chunking import source_chunks
from .config import SharedSentenceConfig
from .encoders import load_registry
from .errors import ValidationError
from .types import Chunk, SourceUnit
from .util import digest


def shared_identity(config: SharedSentenceConfig, registry_path: Path | None = None) -> dict:
    registry = load_registry(registry_path)
    budgets = {}
    for name in config.models:
        if name not in registry:
            raise ValidationError(f"Unknown shared chunk tokenizer: {name}")
        spec = registry[name]
        budgets[name] = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "code_revision": spec.code_revision,
            "trust_remote_code": spec.trust_remote_code,
            "document_prefix": spec.document_prefix,
            "native_limit": spec.max_length,
        }
    return {
        **asdict(config),
        "budgets": budgets,
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "offset_unit": "unicode_code_points",
        "end_exclusive": True,
        "token_count": "largest complete prefixed input with specials",
        "overlap_tokens": "largest token count of the repeated source text without specials",
    }


class TextLayout:
    """ICU returns UTF-16 offsets. All benchmark artifacts use Python source offsets."""

    def __init__(self, text: str, language: str):
        import icu

        self.text = text
        # Record the interior UTF-16 offset of each surrogate pair. Each such
        # offset before a break accounts for one extra UTF-16 code unit.
        outside_bmp = []
        for i, char in enumerate(text):
            if ord(char) > 0xFFFF:
                outside_bmp.append(i + len(outside_bmp) + 1)
        locale = icu.Locale(language)
        words = icu.BreakIterator.createWordInstance(locale)
        words.setText(text)
        self.word_starts, self.word_ends = [], []
        self.breaks = [0]
        for pos in words:
            end = pos - bisect_left(outside_bmp, pos)
            # ICU statuses >= UBRK_WORD_NUMBER include numbers, letters, kana and ideographs.
            if words.getRuleStatus() >= 100:
                self.word_starts.append(self.breaks[-1])
                self.word_ends.append(end)
            self.breaks.append(end)
        sentences = icu.BreakIterator.createSentenceInstance(locale)
        sentences.setText(text)
        self.sentences = [pos - bisect_left(outside_bmp, pos) for pos in sentences]

    def words(self, start: int, end: int) -> int:
        if start == end:
            return 0
        return bisect_left(self.word_starts, end) - bisect_right(self.word_ends, start)

    def word_probes(self, size: int):
        """Disjoint full-size word windows per edition, including intervening punctuation."""
        for i in range(0, len(self.word_starts) - size + 1, size):
            end = self.word_starts[i + size] if i + size < len(self.word_starts) else len(self.text)
            yield self.word_starts[i] if i else 0, end


class SharedSentenceChunker:
    def __init__(self, config: SharedSentenceConfig, identity: dict, tokenizers: dict):
        try:
            import icu
        except ImportError as exc:
            raise ValidationError(
                "Shared sentence preparation requires the shared-chunks extra (PyICU)"
            ) from exc

        if icu.ICU_VERSION != config.icu_version:
            raise ValidationError(
                f"Chunking requires ICU {config.icu_version}, found {icu.ICU_VERSION}"
            )
        self.config = config
        self.manifest = identity
        self.fingerprint = digest(identity)
        self.tokenizers = tokenizers
        self.budgets = identity["budgets"]
        self.limits = {
            name: min(config.max_tokens, b["native_limit"])
            if config.max_tokens is not None
            else b["native_limit"]
            for name, b in self.budgets.items()
        }
        self.counts = lru_cache(maxsize=512)(self._counts)

    def __reduce__(self):
        # An injected chunker can be sent to a spawned process without serializing
        # the process-local cache of previously measured document slices.
        return type(self), (self.config, self.manifest, self.tokenizers)

    @classmethod
    def load(cls, config: SharedSentenceConfig, registry_path=None, token=None):
        from transformers import AutoTokenizer

        identity = shared_identity(config, registry_path)
        tokenizers = {}
        for name, spec in identity["budgets"].items():
            options = {"code_revision": spec["code_revision"]} if spec["code_revision"] else {}
            tokenizer = AutoTokenizer.from_pretrained(
                spec["repo_id"],
                revision=spec["revision"],
                use_fast=True,
                token=token,
                trust_remote_code=spec["trust_remote_code"],
                **options,
            )
            if not tokenizer.is_fast:
                raise ValidationError(f"{name} requires the same fast tokenizer as the encoder")
            tokenizers[name] = tokenizer
        return cls(config, identity, tokenizers)

    def _counts(self, text: str) -> dict[str, int]:
        return {
            name: len(
                tok(
                    self.budgets[name]["document_prefix"] + text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
            for name, tok in self.tokenizers.items()
        }

    def fits(self, layout: TextLayout, start: int, end: int) -> bool:
        if layout.words(start, end) > self.config.max_words:
            return False
        return all(
            n <= self.limits[name] for name, n in self.counts(layout.text[start:end]).items()
        )

    def _fragment_end(
        self, layout: TextLayout, start: int, previous_end: int, sentence_end: int
    ) -> int | None:
        """An overlong sentence needs a word break, or a code-point cut inside a single long word."""

        def last_fit(cuts, low, high):
            best = None
            while low <= high:
                mid = (low + high) // 2
                if self.fits(layout, start, cuts[mid]):
                    best = cuts[mid]
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        result = last_fit(
            layout.breaks,
            bisect_right(layout.breaks, previous_end),
            bisect_right(layout.breaks, sentence_end) - 1,
        )
        if result is not None:
            return result
        first_break = layout.breaks[bisect_right(layout.breaks, previous_end)]
        return last_fit(
            range(len(layout.text) + 1), previous_end + 1, min(first_break, sentence_end)
        )

    def _overlap_start(self, layout: TextLayout, start: int, end: int) -> int:
        budget = self.config.overlap_words
        if not budget:
            return end
        last = bisect_left(layout.word_starts, end)
        floor = max(start + 1, layout.word_starts[max(0, last - budget)] if last else end)
        first = bisect_left(layout.sentences, floor)
        if layout.sentences[first] < end:
            return layout.sentences[first]
        # When the last sentence is too long, repeat its suffix at a word boundary.
        word = bisect_left(layout.word_starts, floor)
        return min(end, layout.word_starts[word]) if word < last else end

    def split(self, unit: SourceUnit) -> list[Chunk]:
        text = unit.text
        layout = TextLayout(text, unit.language)
        start = previous_end = 0
        windows = []
        while previous_end < len(text):
            next_sentence = bisect_right(layout.sentences, previous_end)
            end = previous_end
            for index in range(next_sentence, len(layout.sentences)):
                candidate = layout.sentences[index]
                if not self.fits(layout, start, candidate):
                    break
                end = candidate
            if end == previous_end:
                candidate = layout.sentences[next_sentence]
                if start < previous_end and self.fits(layout, previous_end, candidate):
                    # Preserve a fresh whole sentence when its preceding overlap does not fit.
                    start = previous_end
                    continue
                end = self._fragment_end(layout, start, previous_end, candidate)
                if end is None:
                    if start < previous_end:
                        start = previous_end
                        continue
                    raise ValidationError(f"No valid shared chunk at {unit.id} offset {start}")
            windows.append((start, end, max(self.counts(text[start:end]).values())))
            if end == len(text):
                break
            start, previous_end = self._overlap_start(layout, start, end), end
        if not text:
            if not self.fits(layout, 0, 0):
                raise ValidationError("Model document prefix exceeds the shared chunk budget")
            windows = [(0, 0, max(self.counts("").values()))]

        def overlap_count(value):
            if not value:
                return 0
            return max(
                len(
                    tok(value, add_special_tokens=False, truncation=False, verbose=False)[
                        "input_ids"
                    ]
                )
                for tok in self.tokenizers.values()
            )

        return source_chunks(unit, windows, self.fingerprint, overlap_count)
