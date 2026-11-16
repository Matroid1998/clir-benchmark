#!/usr/bin/env python3
"""Question 2: show only original-query dense retrieval from the validated tables."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from recall_at_10 import CORPORA, CORPUS_LABELS, MODELS, REGIMES, heatmap


def main():
    root = Path(__file__).resolve().parent
    source = root / "question_01_recall_at_10"
    output = root / "question_02_dense_only"
    output.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((source / "manifest.json").read_text())
    provenance = {}
    frames = {}
    for name in ("recall_at_10_by_regime", "recall_at_10_by_setting"):
        path = source / "tables" / f"{name}.csv"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != original_manifest["output_sha256"][f"tables/{name}.csv"]:
            raise ValueError(f"Validated source table has changed: {path}")
        provenance[str(path)] = checksum
        data = pd.read_csv(path)
        data = data[data.cohort.eq("selected_documents") & data.baseline.eq("dense")]
        if not data.input_language.eq("original").all():
            raise ValueError("Expected original queries only")
        data.to_csv(output / f"{name}.csv", index=False)
        frames[name] = data
    summary = frames["recall_at_10_by_regime"]
    if len(summary) != 72 or summary.report_id.nunique() != 24:
        raise ValueError("Expected eight original-query models across three corpora and regimes")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.23, wspace=0.85)
    text = [
        "ORIGINAL-QUERY DENSE RETRIEVAL ONLY",
        "",
        "Recall@10 (%); equal weight per nonempty original-query-language/target setting.",
        "638 selected questions: chemistry 197, EUR-Lex 194, UN 247.",
        "Original queries are embedded directly. No translation or HyDE.",
        "Mixed search counts relevant document-language versions; single-target search has one relevant version.",
        "Regime scores have different denominators and should be read separately.",
        "",
    ]
    for ax, regime in zip(axes, REGIMES, strict=True):
        rows = summary[summary.regime.eq(regime)]
        table = rows.pivot(index="model", columns="corpus", values="recall_at_10_pct").reindex(
            index=MODELS,
            columns=CORPORA,
        )
        table.to_csv(output / f"{regime}_percent.csv")
        text.extend([regime.upper(), table.to_string(float_format="%.2f"), ""])
        display = table.rename(columns=CORPUS_LABELS) / 100
        heatmap(ax, display, regime.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle(
        "Recall@10 (%) · embedding models with original queries",
        x=0.03,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.03,
        0.03,
        "638 selected questions; equal weight per nonempty language/target setting.\n"
        "Mixed and single-language retrieval have different relevant-document denominators.",
        fontsize=10,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"dense_only_recall_at_10.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    (output / "results.txt").write_text("\n".join(text))
    manifest = {
        "baseline": "dense",
        "input_language": "original",
        "cohort": "selected_documents",
        "report_count": 24,
        "regime_rows": len(summary),
        "source_table_sha256": provenance,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methodology": text[2:8],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
