from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fir_gpu_probe.py"
module_spec = importlib.util.spec_from_file_location("fir_gpu_probe", SCRIPT)
probe = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(probe)


@pytest.mark.parametrize("side", ["left", "right"])
def test_mixed_padding_retains_full_rows_and_masks_short_rows(side):
    inputs = {"input_ids": np.full((4, 8), 7), "attention_mask": np.ones((4, 8), dtype=int)}
    probe.apply_mixed_padding(inputs, side, 99)
    assert inputs["attention_mask"].sum(axis=1).tolist() == [8, 4, 8, 4]
    assert inputs["input_ids"][0].tolist() == [7] * 8
    expected_ids = [99] * 4 + [7] * 4 if side == "left" else [7] * 4 + [99] * 4
    assert inputs["input_ids"][1].tolist() == expected_ids
    assert np.array_equal(inputs["input_ids"] == 99, inputs["attention_mask"] == 0)


def test_document_selection_is_bounded_stratified_and_order_independent():
    records = [
        {"id": f"{language}-{index}", "language": language, "text": "x" * index}
        for language in ("en", "zh")
        for index in range(200)
    ]
    forward, counts = probe.select_documents(iter(records))
    backward, _ = probe.select_documents(reversed(records))
    assert forward == backward
    assert counts == {"en": 200, "zh": 200}
    assert len(forward) <= 2 * (64 + 32)
    for language in counts:
        selected = {row["id"] for row in forward if row["language"] == language}
        assert {f"{language}-{index}" for index in range(136, 200)} <= selected


def test_stress_shapes_cover_caps_queries_and_qwen_observed_margin():
    qwen = SimpleNamespace(alias="qwen3-0.6B", max_length=32768)
    length, rationale = probe.stress_length(qwen, [1200], [80], [{"max_encoded_tokens": 1783}])
    assert length == 2048
    assert "NOT an exhaustive" in rationale
    assert probe.stress_length(qwen, [2700], [80], [])[0] == 3072
    assert probe.stress_length(qwen, [40000], [80], [])[0] == 32768
    assert (
        probe.stress_length(SimpleNamespace(alias="bge-m3", max_length=8192), [512], [1200], [])[0]
        == 1200
    )
    assert (
        probe.stress_length(
            SimpleNamespace(alias="embeddinggemma", max_length=2048), [1200], [80], []
        )[0]
        == 2048
    )


def test_prepare_cli_without_config_saves_queries_and_immutable_identity(tmp_path):
    corpus, relevance = tmp_path / "corpus", tmp_path / "relevance"
    corpus.mkdir()
    relevance.mkdir()
    (corpus / "manifest.json").write_text(json.dumps({"corpus_fingerprint": "corpus-1"}))
    (relevance / "manifest.json").write_text(json.dumps({"query_fingerprint": "queries-1"}))
    chunks, queries = corpus / "chunks.parquet", relevance / "queries.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": "d1", "language": "en", "text": "document", "token_count": 3}]
        ),
        chunks,
    )
    pq.write_table(
        pa.Table.from_pylist([{"id": "q1", "question_language": "zh", "question": "问题"}]), queries
    )
    output = tmp_path / "samples.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "prepare",
        "--corpus",
        "un",
        "--chunks",
        str(chunks),
        "--queries",
        str(queries),
        "--prior-qwen-max-tokens",
        "1783",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text())
    digest = payload.pop("fingerprint")
    assert digest == probe.fingerprint(payload)
    assert payload["queries"] == [{"id": "q1", "language": "zh", "text": "问题"}]
    assert payload["previous_qwen_maxima"][0]["max_encoded_tokens"] == 1783
    unchanged = output.read_bytes()
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert output.read_bytes() == unchanged
