import runpy
from pathlib import Path
from types import SimpleNamespace


def test_retry_replaces_cancelled_arrays_and_preserves_the_ledger(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/submit_transformed_retrieval.py"
    submit = runpy.run_path(str(script))["submit"]
    ledger = tmp_path / "submissions.tsv"
    previous = [
        "translate-check\t10\t0\t",
        "translate-un-bm25\t11\t1\tafterok:10",
        "translate-un-embed\t12\t2\tafterok:10",
        "translate-un-dense\t13\t2\taftercorr:12",
        "translate-un-hybrid\t14\t2\taftercorr:13,afterok:11",
        "hyde-un-bm25\t15\t1\t",
    ]
    ledger.write_text("\n".join(previous) + "\n")
    for name, replacement in {
        "STATE": tmp_path,
        "LOGS": tmp_path,
        "verify_generation": lambda config, method: {"receipt": "verified", "lanes": {}},
        "languages": lambda config, method, corpus: ("zh",),
    }.items():
        monkeypatch.setitem(submit.__globals__, name, replacement)
    submissions = []

    def scheduler(command, *, text):
        if command[0] == "sacct":
            previous = int(command[3])
            assert previous != 10
            if previous >= 101:
                return "PENDING\n"
            assert 11 <= previous <= 14
            return "CANCELLED\n"
        assert command[0] == "sbatch"
        submissions.append(command)
        return f"{100 + len(submissions)}\n"

    monkeypatch.setattr("subprocess.check_output", scheduler)
    config = SimpleNamespace(datasets={"un": object()}, models=("a", "b"))
    submit(config, ("translate",), retry_cancelled=True)
    submit(config, ("translate",), retry_cancelled=True)
    assert len(submissions) == 4
    assert "--dependency=aftercorr:102" in submissions[2]
    assert "--dependency=aftercorr:103,afterok:101" in submissions[3]
    assert ledger.read_text().splitlines()[: len(previous)] == previous
    assert len(ledger.read_text().splitlines()) == len(previous) + 4


def test_stack_translation_covers_every_variant_with_47_tasks(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/submit_transformed_retrieval.py"
    submit = runpy.run_path(str(script))["submit"]
    languages = {"chemistry": tuple("abcdefg"), "eurlex": tuple("abcd"), "un": tuple("abcdef")}
    models = tuple(str(i) for i in range(8))
    previous = {
        f"translate-{corpus}-{stage}": str(10 * i + j + 1)
        for i, corpus in enumerate(languages)
        for j, stage in enumerate(("bm25", "embed", "dense", "hybrid"))
    }
    ledger = tmp_path / "submissions.tsv"
    ledger.write_text("".join(f"{k}\t{v}\t1\t\n" for k, v in previous.items()))
    retired = []
    for name, replacement in {
        "STATE": tmp_path,
        "LOGS": tmp_path,
        "verify_generation": lambda config, method: {"receipt": "verified", "lanes": {}},
        "languages": lambda config, method, corpus: languages[corpus],
        "retire_pending_arrays": retired.append,
    }.items():
        monkeypatch.setitem(submit.__globals__, name, replacement)
    submissions = []

    def scheduler(command, *, text):
        assert command[0] == "sbatch"
        submissions.append(command)
        return f"{100 + len(submissions)}\n"

    monkeypatch.setattr("subprocess.check_output", scheduler)
    config = SimpleNamespace(datasets=languages, models=models)
    submit(config, ("translate",), stack_translation=True)
    submit(config, ("translate",), stack_translation=True)
    assert retired[0] == {v for k, v in previous.items() if not k.endswith("-bm25")}
    assert len(submissions) == 9
    total = 17
    import json

    for corpus, lang_set in languages.items():
        for stage, groups in (("embed", 2), ("dense", 4), ("hybrid", 4)):
            slots = json.loads((tmp_path / f"translate-{corpus}-stacked-{stage}.json").read_text())
            assert len(slots) == groups
            assert sorted(pair for slot in slots for pair in slot) == sorted(
                [lang, model] for lang in lang_set for model in models
            )
            total += len(slots)
    assert total == 47
    assert "--dependency=afterany:102:1:4" in submissions[2]


def test_retire_pending_never_cancels_an_entire_array(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/submit_transformed_retrieval.py"
    retire = runpy.run_path(str(script))["retire_pending_arrays"]
    queue = iter(("123_3\n123_5\n", ""))
    cancellations = []

    def list_pending(command, *, text):
        assert "--states=PENDING" in command
        assert "--jobs=123" in command
        return next(queue)

    monkeypatch.setattr("subprocess.check_output", list_pending)
    monkeypatch.setattr("subprocess.run", lambda argv, **_: cancellations.append(argv))
    monkeypatch.setattr("time.sleep", lambda _: None)
    retire({"123"})
    assert cancellations == [["scancel", "--quiet", "--state=PENDING", "123_3", "123_5"]]
