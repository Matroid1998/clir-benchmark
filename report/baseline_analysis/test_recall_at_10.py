"""Offline checks for the analysis estimand, controls and baseline selection."""

import argparse
import json

import pandas as pd
import pytest
from recall_at_10 import COHORTS, add_controls, extract, summarize


def row(**changes):
    return {
        "cohort": COHORTS[0],
        "corpus": "chemistry",
        "model": "bge-m3",
        "baseline": "dense",
        "input_language": "original",
        "report_id": "dense",
        "regime": "mixed",
        "setting_id": "chemistry__en__mixed",
        "query_count": 1,
        "unique_queries_in_regime": 10,
        "relevance_hash": "qrels-a",
        "corpus_fingerprint": "corpus-a",
        "query_fingerprint": "queries-a",
        "recall_at_10": 0.25,
        **changes,
    }


def test_macro_weights_settings_equally_instead_of_repeated_query_counts():
    frame = pd.DataFrame(
        [
            row(recall_at_10=0.0),
            row(setting_id="chemistry__fr__mixed", query_count=9, recall_at_10=1.0),
        ]
    )
    result = summarize(frame).iloc[0]
    assert result.recall_at_10 == 0.5
    assert result.query_setting_count == 10
    assert result.unique_query_count == 10
    assert result.min_setting_queries == 1


def test_controls_match_retriever_language_cohort_and_qrels():
    frame = pd.DataFrame(
        [
            row(),
            row(
                baseline="translate-dense", input_language="de", report_id="t-de", recall_at_10=0.4
            ),
            row(
                baseline="translate-dense", input_language="en", report_id="t-en", recall_at_10=0.8
            ),
            row(baseline="hyde-dense", input_language="de", report_id="h-de", recall_at_10=0.5),
            row(baseline="hybrid", relevance_hash="qrels-b", report_id="other-qrels"),
            row(baseline="hybrid", cohort=COHORTS[1], report_id="other-cohort"),
            row(baseline="hybrid", model="LaBSE", report_id="other-model"),
        ]
    )
    result = add_controls(frame, settings=True).set_index("report_id")
    assert result.loc["h-de", "delta_vs_dense_pp"] == pytest.approx(25.0)
    assert result.loc["h-de", "delta_vs_same_language_translation_pp"] == pytest.approx(10.0)
    assert (
        result.loc[["other-qrels", "other-cohort", "other-model"], "delta_vs_dense_pp"].isna().all()
    )
    assert pd.isna(result.loc["t-en", "delta_vs_same_language_translation_pp"])


def test_missing_controls_stay_unknown_and_ambiguous_controls_fail():
    translated = row(baseline="translate-dense", input_language="fr")
    assert add_controls(pd.DataFrame([translated])).delta_vs_dense_pp.isna().all()
    with pytest.raises(ValueError, match="Ambiguous dense control"):
        add_controls(pd.DataFrame([row(), row(report_id="another-dense-run")]))


def fixture_report(directory, *, baseline="dense"):
    directory.mkdir(parents=True)
    setting = {
        "corpus": "chemistry",
        "regime": "mixed",
        "setting_id": "chemistry__en__mixed",
        "query_language": "en",
        "target_languages": "en-fr",
        "query_count": 2,
        "metrics": {"Recall@10": 0.75},
    }
    run = {
        "model": {"repo_id": "fixture/model", "revision": "pinned"},
        "relevance_hash": "qrels-a",
        "corpus_fingerprint": "corpus-a",
        "query_fingerprint": "queries-a",
        "settings": {setting["setting_id"]: "fixture"},
        "top_k": 100,
    }
    report = {
        "baseline": baseline,
        "model": "bge-m3",
        "corpus": "chemistry",
        "retrieval_unit": "document",
        "candidate_selection": {"candidate_mode": "best-per-document"},
        "run": run,
        "translation_language": "fr" if baseline == "translate-dense" else None,
        "metrics": {
            "setting": [setting],
            "macro_by_regime": [
                {"regime": "mixed", "setting_count": 1, "metrics": {"Recall@10": 0.75}},
            ],
        },
    }
    queries = [
        {
            **setting,
            "baseline": baseline,
            "query_id": str(i),
            "metrics": {"Recall@10": value},
        }
        for i, value in enumerate([0.5, 1.0])
    ]
    mteb = {
        "scores": {
            "test": [
                {"hf_subset": setting["setting_id"], "query_count": 2, "recall_at_10": 0.75},
            ]
        }
    }
    for name, value in (("report.json", report), ("per_query.json", queries), ("mteb.json", mteb)):
        (directory / name).write_text(json.dumps(value))


def test_selected_baseline_reads_only_enabled_query_files(tmp_path):
    dense = tmp_path / "reports/chemistry/dense"
    translated = tmp_path / "reports/chemistry/translated"
    fixture_report(dense)
    fixture_report(translated, baseline="translate-dense")
    (translated / "per_query.json").write_text("disabled-file-is-never-read")
    args = argparse.Namespace(
        artifacts=tmp_path,
        cohort=[COHORTS[0]],
        baseline=["dense"],
        corpus=None,
        model=None,
    )
    settings, summary, catalog, inputs, validation = extract(args, {dense, translated})
    assert settings.recall_at_10.tolist() == [0.75]
    assert summary.recall_at_10.tolist() == [0.75]
    assert len(catalog) == validation["report_count"] == 1
    assert str(translated / "per_query.json") not in inputs
    args.baseline = ["translate-dense"]
    with pytest.raises(json.JSONDecodeError):
        extract(args, {dense, translated})


def test_mteb_disagreement_fails_validation(tmp_path):
    dense = tmp_path / "reports/chemistry/dense"
    fixture_report(dense)
    mpath = dense / "mteb.json"
    mteb = json.loads(mpath.read_text())
    mteb["scores"]["test"][0]["recall_at_10"] = 0.8
    mpath.write_text(json.dumps(mteb))
    args = argparse.Namespace(
        artifacts=tmp_path,
        cohort=[COHORTS[0]],
        baseline=None,
        corpus=None,
        model=None,
    )
    with pytest.raises(ValueError, match="differs from saved reports or MTEB"):
        extract(args, {dense})
