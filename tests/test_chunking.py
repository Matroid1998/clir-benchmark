from dataclasses import replace

import pytest
from conftest import source

from clir_benchmark.chunking import Chunker, load_tokenizer
from clir_benchmark.config import ChunkingConfig


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        " \n\t ",
        " leading and trailing  ",
        "中文化学法律文献" * 12,
        "العربية الكيمياء\n" * 10,
        "é e\u0301 😀⚗️\n" * 20,
        "<p>A &amp; B</p>" * 10,
    ],
)
def test_exact_slices_complete_coverage(chunker, text):
    unit = source(text=text)
    chunks = chunker.split(unit)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    rebuilt = ""
    previous_end = 0
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.token_count <= 12
        assert chunk.token_count == chunker.count(chunk.text)
        assert chunk.start <= previous_end
        assert chunk.overlap_chars == max(0, previous_end - chunk.start)
        rebuilt += text[previous_end : chunk.end]
        previous_end = chunk.end
    assert rebuilt == text


def test_default_budget_and_overlap(tokenizer):
    chunker = Chunker(tokenizer, ChunkingConfig())
    chunks = chunker.split(source(text="a" * 1200))
    assert chunker.special_count == 2
    assert [(c.start, c.end) for c in chunks] == [(0, 510), (446, 956), (892, 1200)]
    assert [c.overlap_tokens for c in chunks] == [0, 64, 64]
    assert [c.token_count for c in chunks] == [512, 512, 310]


def test_short_units_stay_separate_and_ids_are_stable(chunker):
    unit = source(text="short")
    chunks = chunker.split(unit)
    assert len(chunks) == 1
    assert chunks == chunker.split(unit)
    assert chunks[0].id != chunker.split(replace(unit, id="b:en"))[0].id
    assert chunks[0].id != chunker.split(replace(unit, id="a:fr", language="fr"))[0].id
    assert chunks == chunker.split(replace(unit, metadata={"irrelevant_qa_provenance": "changed"}))


class ExpandingSliceTokenizer:
    """A slice beginning at b tokenizes longer than the same text in context."""

    is_fast = True

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        ids = list(range(len(text)))
        if add_special_tokens and text.startswith("b"):
            ids += [99, 99]
        if add_special_tokens:
            ids += [1, 2]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_retokenizes_and_shrinks_exact_slice():
    chunker = Chunker(ExpandingSliceTokenizer(), ChunkingConfig(max_tokens=8, overlap_tokens=1))
    chunks = chunker.split(source(text="aaaaabbbbbbbbbbbb"))
    assert all(c.token_count <= 8 for c in chunks)
    assert any(len(c.text) < 6 for c in chunks[1:])
    assert chunks[-1].end == len("aaaaabbbbbbbbbbbb")


@pytest.mark.integration
def test_real_pinned_bge_tokenizer():
    config = ChunkingConfig()
    chunker = Chunker(load_tokenizer(config), config)
    text = (
        "  <p>Carbonyl C=O &amp; CO₂; e\u0301 and é.</p>\n中文化学文献。 العربية الروسية\n😀⚗️\t"
        * 120
        + "  \n"
    )
    chunks = chunker.split(source(text=text))
    assert len(chunks) > 1
    assert chunker.special_count == 2
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    previous_end = 0
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert len(chunker.tokenizer(chunk.text, add_special_tokens=True)["input_ids"]) <= 512
        assert chunk.start <= previous_end
        previous_end = chunk.end
    assert chunker.split(source(text=" \n\t "))[0].text == " \n\t "
