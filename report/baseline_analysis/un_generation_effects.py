#!/usr/bin/env python3
"""Translation/HyDE effects on recall and hit rate for a selected corpus/retriever.

Defaults reproduce Question 6 (UN / EmbeddingGemma); --corpus also supports
chemistry and EUR-Lex for Question 7. Reads completed result files only.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANGUAGES = ["de", "en", "es", "fr", "zh"]
REGIMES = ["monolingual", "cross_language", "mixed"]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["chemistry", "eurlex", "un"], default="un")
    parser.add_argument("--model", default="embeddinggemma")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus_label = {"chemistry": "Chemistry", "eurlex": "EUR-Lex", "un": "UN"}[args.corpus]
    model_label = "EmbeddingGemma" if args.model == "embeddinggemma" else args.model
    with (root / "question_01_recall_at_10/tables/report_catalog.csv").open() as handle:
        entries = [
            r
            for r in csv.DictReader(handle)
            if r["cohort"] == "selected_documents"
            and r["corpus"] == args.corpus
            and r["model"] == args.model
            and r["baseline"] in ["dense", "translate-dense", "hyde-dense"]
        ]
    if len(entries) != 11:
        raise ValueError("Expected original queries and five languages for each generated method")
    rows, hashes, contracts = [], {}, []
    for entry in entries:
        path = Path(entry["report_path"])
        query_path = path.parent / "per_query.json"
        report = json.loads(path.read_text())
        queries = json.loads(query_path.read_text())
        for p in [path, query_path]:
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        run = report["run"]
        contracts.append(
            (
                run["relevance_hash"],
                run["corpus_fingerprint"],
                run["query_fingerprint"],
                json.dumps(run["model"], sort_keys=True),
                json.dumps(run["settings"], sort_keys=True),
                run["top_k"],
                frozenset((q["setting_id"], q["query_id"]) for q in queries),
            )
        )
        frame = pd.DataFrame(
            [
                {
                    "setting_id": q["setting_id"],
                    "regime": q["regime"],
                    "recall": q["metrics"]["Recall@10"],
                    "hit": q["metrics"]["Hit@10"],
                }
                for q in queries
            ]
        )
        if any(q["metrics"]["Hit@10"] != q["metrics"]["SourceRecall@10"] for q in queries):
            raise ValueError(
                "Hit@10 and source recall differ; source-success interpretation needs review"
            )
        calculated = (
            frame.groupby(["regime", "setting_id"])[["recall", "hit"]]
            .mean()
            .groupby("regime")
            .mean()
        )
        for saved in report["metrics"]["macro_by_regime"]:
            regime = saved["regime"]
            if not np.allclose(
                calculated.loc[regime],
                [saved["metrics"]["Recall@10"], saved["metrics"]["Hit@10"]],
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Reconstructed scores differ from saved report")
            rows.append(
                {
                    "model": args.model,
                    "corpus": args.corpus,
                    "baseline": entry["baseline"],
                    "input_language": entry["input_language"],
                    "regime": regime,
                    "recall_at_10_pct": 100 * calculated.loc[regime, "recall"],
                    "hit_at_10_pct": 100 * calculated.loc[regime, "hit"],
                    "report_path": str(path),
                }
            )
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("Query cohorts, model specifications or retrieval settings do not match")
    unique_questions = len({q[1] for q in contracts[0][-1]})
    result = pd.DataFrame(rows)
    control = result[result.baseline.eq("dense")].set_index("regime")
    translated = result[result.baseline.eq("translate-dense")].set_index(
        ["regime", "input_language"]
    )
    for metric in ["recall_at_10", "hit_at_10"]:
        result[f"{metric}_delta_vs_original_pp"] = result[f"{metric}_pct"] - result.regime.map(
            control[f"{metric}_pct"]
        )
        result[f"{metric}_delta_vs_same_language_translation_pp"] = [
            row[f"{metric}_pct"]
            - translated.loc[(row["regime"], row["input_language"]), f"{metric}_pct"]
            if row["baseline"] == "hyde-dense"
            else np.nan
            for row in rows
        ]
    suffix = "" if args.model == "embeddinggemma" else f"_{args.model}"
    output = args.output or root / f"question_06_{args.corpus}{suffix}_generation_effects"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "recall_and_hit_effects.csv", index=False)
    signed_rows = []
    for baseline, language in [("dense", "original")] + [
        (baseline, language)
        for baseline in ["translate-dense", "hyde-dense"]
        for language in LANGUAGES
    ]:
        subset = result[
            result.baseline.eq(baseline) & result.input_language.eq(language)
        ].set_index("regime")
        display = {"baseline": baseline, "input_language": language}
        for regime, metric in [(r, "recall_at_10") for r in REGIMES] + [("mixed", "hit_at_10")]:
            score = subset.loc[regime, f"{metric}_pct"]
            delta = subset.loc[regime, f"{metric}_delta_vs_original_pp"]
            display[f"{regime}_{metric}_score_and_delta_pp"] = f"{score:.2f} ({delta:+.2f})"
        signed_rows.append(display)
    signed = pd.DataFrame(signed_rows)
    signed.to_csv(output / "scores_with_signed_changes.csv", index=False)
    (output / "scores_with_signed_changes.txt").write_text(
        "Each cell: score (%) followed by signed change from embedding-only retrieval (percentage points).\n"
        + signed.to_string(index=False)
        + "\n"
    )
    panels = [
        ("monolingual", "recall_at_10", "Monolingual Recall@10"),
        ("cross_language", "recall_at_10", "Cross-language Recall@10"),
        ("mixed", "recall_at_10", "Mixed Recall@10 · version coverage"),
        ("mixed", "hit_at_10", "Mixed Hit@10 · finding any version"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, hspace=0.5, wspace=0.25)
    x = np.arange(5)
    for ax, (regime, metric, title) in zip(axes.flat, panels, strict=True):
        for baseline, label, offset, color in [
            ("translate-dense", "Translation", -0.19, "#327ca5"),
            ("hyde-dense", "HyDE", 0.19, "#d78e39"),
        ]:
            values = (
                result[result.baseline.eq(baseline) & result.regime.eq(regime)]
                .set_index("input_language")
                .loc[LANGUAGES, f"{metric}_pct"]
            )
            bars = ax.bar(x + offset, values, width=0.36, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        original = control.loc[regime, f"{metric}_pct"]
        ax.axhline(
            original,
            linestyle="--",
            color="#263442",
            linewidth=1.4,
            label=f"Original {original:.2f}%",
        )
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks(x, LANGUAGES)
        ax.set_xlabel("Generated input language")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        f"{corpus_label} · {model_label} · effects of translation and HyDE",
        x=0.04,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.04,
        0.02,
        f"Same {unique_questions} selected questions and retrieval settings; equal weight per nonempty setting.\nRegime names refer to ORIGINAL question language versus searched corpus language. Each generated language is a separate run.",
        fontsize=9,
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"generation_effects.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    pivot = result.pivot(
        index=["baseline", "input_language"], columns="regime", values="recall_at_10_pct"
    ).reindex(columns=REGIMES)
    mixed = result[result.regime.eq("mixed")].drop(
        columns=["model", "corpus", "regime", "report_path"]
    )
    text = (
        f"{corpus_label} / {model_label}: Recall@10 (%)\n"
        + pivot.to_string(float_format="%.2f")
        + "\n\nMIXED SEARCH EFFECTS\n"
        + mixed.to_string(index=False, float_format="%.2f")
        + "\n"
    )
    (output / "results.txt").write_text(text)
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "corpus": args.corpus,
                "model": args.model,
                "reports": len(entries),
                "unique_questions": unique_questions,
                "cohorts_and_settings_match": True,
                "per_query_aggregates_match_saved_reports": True,
                "methodology": "Equal setting means; deltas in percentage points; generated languages evaluated separately; no inference or paid calls.",
                "source_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
