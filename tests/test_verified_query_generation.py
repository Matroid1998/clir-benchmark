import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_pipeline import write_rows

from clir_benchmark.cli import main
from clir_benchmark.config import QUESTION_LANGUAGE, load_config
from clir_benchmark.generation import ChatCompletionsClient, CompletionError, with_refusal_policy
from clir_benchmark.hyde import load_hyde
from clir_benchmark.translation import load_translations
from clir_benchmark.util import read_json

MODELS = {
    "ar": "openai/gpt-5.6-luna",
    "fr": "anthropic/claude-haiku-5.5",
    "ru": "anthropic/claude-haiku-5.5",
    "zh": "qwen/qwen3.8-27b",
}


@pytest.fixture
def verified(tmp_path):
    recipe = Path(__file__).resolve().parents[1] / "configs/verified-20261118-query-generation.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["datasets"] = {
        "un": {
            **raw["datasets"]["un"],
            "qa": {**raw["datasets"]["un"]["qa"], "local_dir": "verified-release"},
            "corpus": {**raw["datasets"]["un"]["corpus"], "local_dir": "missing-corpus"},
        }
    }
    raw["env_file"] = "missing.env"
    raw["output_dir"] = "output"
    questions = {
        "ar": " ما الجواب؟ ",
        "fr": " Quelle est la règle ? ",
        "ru": "Какое правило?",
        "zh": "有什么规定？",
    }
    write_rows(
        tmp_path / "verified-release",
        "qas",
        [
            {
                "_id": language,
                "question_language": language,
                "question": text,
                "answer": "private fixture answer",
                "evidence": "private fixture evidence",
            }
            for language, text in questions.items()
        ],
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return ["--config", str(path)], load_config(path), questions


def test_offline_dry_run_counts_only_selected_requests(verified, monkeypatch, capsys):
    args, config, _ = verified

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run initialized an LLM client")

    monkeypatch.setattr(ChatCompletionsClient, "__init__", forbidden)
    assert main([*args, "translate", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)["jobs"]
    assert {row["output_language"] for row in result} == {"ar", "en", "es", "fr", "ru", "zh"}
    assert sum(row["variants"] for row in result) == 24
    assert sum(row["identity_variants"] for row in result) == 4
    assert sum(row["requests_before_cache_or_retries"] for row in result) == 20
    assert all(row["qa_release"]["revision"] == config.datasets["un"].qa.revision for row in result)
    assert main([*args, "generate-hyde", "--dry-run"]) == 0
    (native,) = json.loads(capsys.readouterr().out)["jobs"]
    assert {row["output_language"]: row["model"] for row in native["outputs"]} == MODELS
    assert all(row["variants"] == 1 for row in native["outputs"])
    assert not config.output_dir.exists()


@pytest.fixture
def mock_completions(monkeypatch):
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-fixture")

    def complete(self, payload):
        serialized = json.dumps(payload)
        assert "private fixture" not in serialized
        assert "unused-fixture" not in serialized
        schema = payload["response_format"]["json_schema"]["schema"]
        (field,) = schema["required"]
        calls.append(payload)
        return {
            "model": payload["model"] + "-snapshot",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({field: "fixed output"})},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cost": 0.01,
            },
        }

    monkeypatch.setattr(ChatCompletionsClient, "complete", complete)
    return calls


def test_translation_and_native_hyde_route_and_reuse_outputs(verified, mock_completions, capsys):
    args, config, questions = verified
    translate = [*args, "translate", "--translation-language", "fr"]
    assert main(translate) == 0
    _, rows, manifest = load_translations(config, "un", "fr")
    translations = {row["id"]: row for row in rows}
    assert translations["fr"]["text"] == questions["fr"]
    assert translations["fr"]["api_cost_usd"] == translations["fr"]["translation_seconds"] == 0
    assert translations["fr"]["requested_model"] is None
    assert {row["requested_model"] for row in rows if row["status"] == "translated"} == {
        MODELS["fr"]
    }
    assert (
        "Source language: ar\nTarget language: fr" in mock_completions[0]["messages"][1]["content"]
    )
    assert len(mock_completions) == 3
    assert not (config.output_dir / "hyde_requests").exists()
    assert manifest["by_language_and_model"][0]["api_cost_usd"] == 0.03

    hyde = [*args, "generate-hyde"]
    assert main(hyde) == 0
    _, rows, manifest = load_hyde(config, "un", QUESTION_LANGUAGE)
    assert {row["id"]: row["requested_model"] for row in rows} == MODELS
    assert all(row["language"] == row["source_language"] == row["id"] for row in rows)
    assert {row["id"]: row["original_text"] for row in rows} == questions
    assert all(row["returned_model"] == MODELS[row["id"]] + "-snapshot" for row in rows)
    assert sum(group["api_attempts"] for group in manifest["by_language_and_model"]) == 4
    chinese = next(row for row in rows if row["id"] == "zh")
    assert chinese["estimated_cost_usd"] is None and chinese["api_cost_usd"] == 0.01
    assert len(mock_completions) == 7

    for command in (translate, hyde):
        assert main(command) == 0
        capsys.readouterr()
        assert main([*command, "--verify"]) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "verify"
    assert len(mock_completions) == 7
    for stage in ("translation", "hyde"):
        audits = [
            read_json(path) for path in (config.output_dir / f"{stage}_invocations").glob("*.json")
        ]
        reused = next(audit for audit in audits if audit["artifact_reused"])
        assert not reused["new_request_keys"]
        assert reused["new_work_by_language_and_model"] == []


def test_failed_attempts_remain_in_cost_totals(verified, mock_completions, monkeypatch):
    args, config, _ = verified
    complete = ChatCompletionsClient.complete

    def fail_once(self, payload):
        response = complete(self, payload)
        if len(mock_completions) == 1:
            response["choices"][0]["finish_reason"] = "length"
        return response

    monkeypatch.setattr(ChatCompletionsClient, "complete", fail_once)
    command = [*args, "translate", "--translation-language", "fr"]
    assert main(command) == main(command) == 2
    assert len(mock_completions) == 1
    assert main([*command, "--retry-failed"]) == 0
    _, rows, _ = load_translations(config, "un", "fr")
    arabic = next(row for row in rows if row["id"] == "ar")
    assert arabic["attempts"] == 2 and arabic["api_cost_usd"] == 0.02
    assert arabic["translation_seconds"] > 0
    assert len(mock_completions) == 4
    request = read_json(
        config.output_dir / "translation_requests" / f"{arabic['request_key']}.json"
    )
    assert [attempt["status"] for attempt in request["attempts"]] == ["failed", "success"]


@pytest.mark.parametrize(("status", "attempts"), [(401, 1), (402, 1), (429, 3), (503, 3)])
def test_transient_retry_bound_survives_restart(
    verified, mock_completions, monkeypatch, status, attempts
):
    args, config, _ = verified
    failures = []

    def fail(self, payload):
        failures.append(payload["model"])
        raise CompletionError("sanitized provider failure", status=status)

    monkeypatch.setattr(ChatCompletionsClient, "complete", fail)
    monkeypatch.setattr("clir_benchmark.generation.time.sleep", lambda _: None)
    command = [*args, "translate", "--translation-language", "fr", "--retry-transient"]
    assert main(command) == main(command) == 2
    assert len(failures) == attempts
    history = read_json(next((config.output_dir / "translation_requests").glob("*.json")))
    assert all(a["http_status"] == status for a in history["attempts"])


@pytest.mark.parametrize("stage", ["translate", "generate-hyde"])
def test_malformed_completion_retried_and_accounted(verified, mock_completions, monkeypatch, stage):
    args, config, _ = verified
    complete = ChatCompletionsClient.complete

    def malformed_once(self, payload):
        response = complete(self, payload)
        if len(mock_completions) == 1:
            response["choices"][0]["message"]["content"] = "invalid json"
        return response

    monkeypatch.setattr(ChatCompletionsClient, "complete", malformed_once)
    monkeypatch.setattr("clir_benchmark.generation.time.sleep", lambda _: None)
    translate = stage == "translate"
    command = [
        *args,
        stage,
        "--retry-transient",
        *(["--translation-language", "fr"] if translate else []),
    ]
    assert main(command) == 0
    _, rows, manifest = (
        load_translations(config, "un", "fr")
        if translate
        else load_hyde(config, "un", QUESTION_LANGUAGE)
    )
    expected = 4 if translate else 5
    assert manifest["new_api_attempts"] == manifest["new_work"]["api_attempts"] == expected
    assert len(mock_completions) == expected
    assert sum(r["api_cost_usd"] for r in rows) == pytest.approx(0.01 * expected)


@pytest.mark.parametrize("stage", ["translate", "generate-hyde"])
def test_disjoint_generation_shards_assemble_offline(
    verified, mock_completions, monkeypatch, capsys, stage
):
    args, config, questions = verified
    translate = stage == "translate"
    language = "fr" if translate else QUESTION_LANGUAGE
    command = [*args, stage, "--translation-language" if translate else "--hyde-language", language]
    load = load_translations if translate else load_hyde
    # Group 1 has one query and group 0 has three. Finishing out of order must work.
    last = [*command, "--shard-size", "3", "--shard-index", "1"]
    first = [*command, "--shard-size", "3", "--shard-index", "0"]
    assert main(last) == 0
    assert main([*command, "--assemble-shards", "--shard-size", "3"]) == 2
    assert main([*command, "--verify"]) == 2
    assert main(last) == main([*last, "--verify"]) == 0
    assert len(mock_completions) == 1
    assert main(first) == 0
    calls = len(mock_completions)
    assert calls == (3 if translate else 4)
    _, saved, _ = load(config, "un", language, shard=(1, 3))
    assert [row["id"] for row in saved] == ["zh"]
    monkeypatch.delenv("OPENROUTER_API_KEY")

    def forbidden(*args, **kwargs):
        pytest.fail("Offline assembly initialized an LLM client")

    monkeypatch.setattr(ChatCompletionsClient, "__init__", forbidden)
    capsys.readouterr()
    for _ in range(2):
        assert main([*command, "--assemble-shards", "--shard-size", "3"]) == 0
    _, records, manifest = load(config, "un", language)
    assert [row["id"] for row in records] == sorted(questions)
    assert sum(row["api_cost_usd"] for row in records) == pytest.approx(0.01 * calls)
    assert manifest["summary"]["api_attempts"] == calls
    assert manifest["new_api_attempts"] == 0
    assert manifest["assembled_from_shards"]["size"] == 3
    assert "shard" not in manifest
    assert len(mock_completions) == calls


@pytest.mark.parametrize("stage", ["translate", "generate-hyde"])
def test_provider_refusal_is_terminal_and_cannot_be_published_as_text(
    verified, mock_completions, monkeypatch, stage
):
    args, config, questions = verified
    complete = ChatCompletionsClient.complete
    translate = stage == "translate"
    command = [*args, stage, *(["--translation-language", "zh"] if translate else [])]

    def provider(self, payload):
        response = complete(self, payload)
        user_text = payload["messages"][-1]["content"]
        choice = response["choices"][0]
        if questions["fr"] in user_text:
            choice["finish_reason"] = "content_filter"
            choice["message"]["content"] = "discard this partial text"
        if questions["ru"] in user_text:
            choice["message"]["refusal"] = "declined"
        return response

    monkeypatch.setattr(ChatCompletionsClient, "complete", provider)
    assert main(command) == 2
    assert len(mock_completions) == 2  # Arabic completed; French provider-declined.
    first = [*command, "--record-refusals", "--shard-index", "0", "--shard-size", "2"]
    assert main(first) == main(first) == 0
    assert len(mock_completions) == 2  # Both histories were reused from the interrupted strict run.
    assert (
        main(
            [
                *command,
                "--record-refusals",
                "--retry-transient",
                "--shard-index",
                "1",
                "--shard-size",
                "2",
            ]
        )
        == 0
    )
    paid = 3 if translate else 4
    assert len(mock_completions) == paid
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert main([*command, "--record-refusals", "--assemble-shards", "--shard-size", "2"]) == 0
    assert main([*command, "--record-refusals", "--verify"]) == 0
    assert main([*command, "--verify"]) == 2  # The strict artifact is still incomplete.
    cfg = with_refusal_policy(config, True)
    _, records, manifest = (
        load_translations(cfg, "un", "zh") if translate else load_hyde(cfg, "un", QUESTION_LANGUAGE)
    )
    declines = [row for row in records if row["status"] == "provider_refused"]
    assert len(records) == 4 and {row["id"] for row in declines} == {"fr", "ru"}
    assert manifest["summary"]["provider_refused_count"] == 2
    assert sum(g["provider_refused_count"] for g in manifest["by_language_and_model"]) == 2
    assert manifest["summary"]["api_attempts"] == paid
    for row in declines:
        assert row["text"] is None and row["original_text"] == questions[row["id"]]
        assert row["api_cost_usd"] == 0.01 and row["attempts"] == 1
        assert row["translation_seconds" if translate else "generation_seconds"] > 0
        parent = "translation_requests" if translate else "hyde_requests"
        ledger = read_json(cfg.output_dir / parent / (row["request_key"] + ".json"))
        assert ledger["status"] == "failed" and len(ledger["attempts"]) == 1
        assert ledger["attempts"][0]["response"]["choices"]
    assert len(mock_completions) == paid


def test_local_shard_launcher_stage_selection(verified, mock_completions, monkeypatch, tmp_path):
    args, config, _ = verified
    script = Path(__file__).resolve().parents[1] / "scripts/run_generation_shards.py"
    logs = tmp_path / "local-job-logs"
    command = [
        str(script),
        *args,
        "--state-dir",
        str(logs),
        "--workers",
        "1",
        "--shard-size",
        "3",
        "--stage",
        "generate-hyde",
    ]

    def fake_worker(command, *, stdout, stderr):
        # Execute the actual worker CLI in the fixture process with the mocked transport.
        assert command[1:3] == ["-m", "clir_benchmark"]
        return subprocess.CompletedProcess(command, main(command[3:]))

    monkeypatch.setattr(subprocess, "run", fake_worker)
    monkeypatch.setattr(sys, "argv", [*command, "--plan"])
    runpy.run_path(str(script), run_name="__main__")
    plan = read_json(next(logs.glob("*/plan.json")))
    assert (plan["outputs"], plan["shards"], plan["calls_before_cache_and_retries"]) == (4, 2, 4)
    assert not mock_completions
    monkeypatch.setattr(sys, "argv", [*command, "--run"])
    runpy.run_path(str(script), run_name="__main__")
    assert len(mock_completions) == 4
    receipt = read_json(next(logs.glob("*/verified.json")))
    assert [r["stage"] for r in receipt] == ["generate-hyde"]
    assert len(list(logs.glob("*/*.log"))) == 2
    assert not (config.output_dir / "translation_requests").exists()
    monkeypatch.setattr(sys, "argv", [*command, "--assemble"])
    runpy.run_path(str(script), run_name="__main__")
    assert len(mock_completions) == 4


@pytest.mark.skipif(not shutil.which("rsync"), reason="Local rsync needed for transfer fixture")
@pytest.mark.parametrize("policy_flags", [[], ["--record-refusals"]])
def test_stage_supervisor_publishes_only_verified_files(
    verified, mock_completions, tmp_path, policy_flags
):
    args, config, _ = verified
    assert (
        main([*args, "generate-hyde", "--shard-index", "0", "--shard-size", "250", *policy_flags])
        == 0
    )
    (config.output_dir / "local.yaml").write_text("private fixture configuration")
    remote = tmp_path / "scratch"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Replace the entire SSH transport with local execution against an isolated scratch tree.
    stub = """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
base = '/scratch/matroid/clir-benchmark/runs/narval-20261119-verified-query-generation'
if sys.argv[0].endswith('ssh'):
    args = args[args.index('narval') + 1:]
    binary = args.pop(0)
else:
    binary = os.environ['FIXTURE_RSYNC']
args = [a.removeprefix('narval:').replace(base, os.environ['FIXTURE_REMOTE']) for a in args]
os.execvp(binary, [binary, *args])
"""
    for binary in ["ssh", "rsync", "sleep"]:
        target = binaries / binary
        target.write_text(stub if binary != "sleep" else "#!/bin/sh\nexit 0\n")
        target.chmod(0o700)
    jobs = tmp_path / "jobs"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "FIXTURE_REMOTE": str(remote),
        "FIXTURE_RSYNC": shutil.which("rsync"),
        "GENERATION_CONFIG": args[1],
        "GENERATION_STATE_DIR": str(jobs),
    }
    script = Path(__file__).resolve().parents[1] / "scripts/run_verified_generation_stage.sh"
    command = ["bash", str(script), "generate-hyde", "--transfer-only", *policy_flags]
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    files = [
        path.relative_to(remote / "artifacts")
        for path in (remote / "artifacts").rglob("*")
        if path.is_file()
    ]
    assert len(files) == 7  # 4 requests, 1 invocation, canonical parquet and manifest.
    assert all(
        path.suffix in {".json", ".parquet"} and "shards" not in path.parts for path in files
    )
    assert len(list((remote / "transfer_receipts").glob("*.sha256"))) == 1
    assert len(list(jobs.glob("*/transfer/verified"))) == 1
    manifest = next((remote / "artifacts").rglob("manifest.json"))
    manifest.write_text("existing conflicting publication")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert manifest.read_text() == "existing conflicting publication"
