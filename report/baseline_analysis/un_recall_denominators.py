#!/usr/bin/env python3
"""Question 5: explain UN Recall@10 versus any-version success from saved results."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    with (root / "question_01_recall_at_10/tables/report_catalog.csv").open() as handle:
        entry = next(
            r
            for r in csv.DictReader(handle)
            if r["corpus"] == "un"
            and r["model"] == "embeddinggemma"
            and r["baseline"] == "dense"
            and r["cohort"] == "selected_documents"
        )
    path = Path(entry["report_path"])
    report = json.loads(path.read_text())
    query_path = path.parent / "per_query.json"
    queries = json.loads(query_path.read_text())
    qrel_path = (
        repo
        / "artifacts/fir-20261109-all-questions/artifacts/relevance/un"
        / report["run"]["relevance_hash"]
        / "qrels.parquet"
    )
    qrels = pq.read_table(qrel_path, columns=["setting_id", "query_id", "document_id"]).to_pandas()
    counts = qrels.groupby(["setting_id", "query_id"]).document_id.nunique()
    rows = []
    for query in queries:
        count = int(counts.loc[(query["setting_id"], query["query_id"])])
        retrieved = query["metrics"]["Recall@10"] * count
        if not np.isclose(retrieved, round(retrieved)):
            raise ValueError("Recall cannot be reconstructed as a relevant-version count")
        if query["metrics"]["Hit@10"] != float(retrieved > 0):
            raise ValueError("Hit metric is inconsistent with retrieved-version count")
        rows.append(
            {
                "regime": query["regime"],
                "setting_id": query["setting_id"],
                "query_id": query["query_id"],
                "query_language": query["query_language"],
                "relevant_versions": count,
                "retrieved_versions_at_10": round(retrieved),
                "recall_at_10": query["metrics"]["Recall@10"],
                "hit_at_10": query["metrics"]["Hit@10"],
            }
        )
    frame = pd.DataFrame(rows)
    order = ["monolingual", "cross_language", "mixed"]
    summaries = []
    for regime in order:
        subset = frame[frame.regime.eq(regime)]
        denominators = subset.relevant_versions.unique().tolist()
        expected = 4 if regime == "mixed" else 1
        if denominators != [expected]:
            raise ValueError(f"Unexpected relevance denominator: {regime} {denominators}")
        means = (
            subset.groupby("setting_id")[["recall_at_10", "hit_at_10", "retrieved_versions_at_10"]]
            .mean()
            .mean()
        )
        saved = next(r for r in report["metrics"]["macro_by_regime"] if r["regime"] == regime)
        for metric, saved_name in [("recall_at_10", "Recall@10"), ("hit_at_10", "Hit@10")]:
            if not np.isclose(means[metric], saved["metrics"][saved_name], atol=1e-12, rtol=0):
                raise ValueError("Reconstructed summary differs from the saved report")
        summaries.append(
            {
                "regime": regime,
                "relevant_versions_per_query": expected,
                "unique_questions": subset.query_id.nunique(),
                "settings": subset.setting_id.nunique(),
                "recall_at_10_pct": 100 * means.recall_at_10,
                "any_version_hit_at_10_pct": 100 * means.hit_at_10,
                "mean_retrieved_versions_at_10": means.retrieved_versions_at_10,
            }
        )
    summary = pd.DataFrame(summaries)
    output = root / "question_05_un_recall_denominators"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "query_version_counts.csv", index=False)
    summary.to_csv(output / "regime_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(3)
    for offset, column, label, color in [
        (-0.19, "recall_at_10_pct", "Recall@10: fraction of relevant versions", "#367ba8"),
        (0.19, "any_version_hit_at_10_pct", "Hit@10: at least one relevant version", "#df9a39"),
    ]:
        bars = ax.bar(x + offset, summary[column], width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_xticks(
        x,
        [
            "Monolingual\n1 relevant version",
            "Cross-language\n1 relevant version",
            "Mixed\n4 relevant versions",
        ],
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("UN · embeddinggemma · original queries", loc="left", fontsize=16, pad=18)
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.05,
        0.02,
        "247 selected questions. Equal weight per nonempty setting.\nMixed Hit@10 measures finding the source in any of the four searched languages.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.23)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"recall_vs_hit.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    metadata = {
        "status": "passed",
        "model": "embeddinggemma",
        "baseline": "dense",
        "corpus": "un",
        "methodology": "Saved per-query recall multiplied by independently counted qrels; equal setting macros checked against report.json. No new retrieval or inference.",
        "mixed_retrieved_version_distribution": frame[frame.regime.eq("mixed")]
        .retrieved_versions_at_10.value_counts()
        .sort_index()
        .to_dict(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [path, query_path, qrel_path, Path(__file__)]
        },
    }
    (output / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(metadata["mixed_retrieved_version_distribution"]))


if __name__ == "__main__":
    main()
