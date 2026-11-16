#!/usr/bin/env python3
"""Render corpus-difficulty diagnostics and inspect answer-bearing unjudged chunks.

Run inspect_corpus_difficulty.py first. All computations use saved results/texts;
there is no new retrieval, model inference, or paid call.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from inspect_corpus_difficulty import CORPORA, normalized, tokens

LABELS = {"chemistry": "Chemistry", "eurlex": "EUR-Lex", "un": "UN"}
COLORS = {"chemistry": "#2978a4", "eurlex": "#c78435", "un": "#50855a"}


def savefig(fig, output, name):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    output = root / "question_08_chemistry_difficulty"
    features = pd.read_csv(output / "query_features.csv")
    factors = pd.read_csv(output / "corpus_factors.csv").set_index("corpus").loc[CORPORA]
    scores = pd.read_csv(output / "dense_quality_and_language_controls.csv")
    originals = json.loads((output / "selected_questions_and_native_sources.json").read_text())
    original_map = {(q["query"]["corpus"], q["query"]["id"]): q for q in originals}
    with (root / "question_01_recall_at_10/tables/report_catalog.csv").open() as handle:
        catalog = list(csv.DictReader(handle))
    ambiguity = []
    curves = []
    for corpus in CORPORA:
        entry = next(
            r
            for r in catalog
            if r["cohort"] == "selected_documents"
            and r["corpus"] == corpus
            and r["baseline"] == "dense"
            and r["model"] == "embeddinggemma"
        )
        report_path = Path(entry["report_path"])
        ranked = pq.read_table(
            report_path.parent / "analysis/rankings.parquet",
            columns=["query_id", "rank", "relevance", "document_id", "text"],
            filters=[("setting_id", "=", f"{corpus}__en__en")],
        ).to_pandas()
        for qid, group in ranked.groupby("query_id"):
            original = original_map[(corpus, qid)]
            answer = normalized(original["query"]["answer"])
            eligible = len(tokens(answer)) >= 6
            group = group.sort_values("rank")
            positives = group[group.relevance.gt(0)]
            first = int(positives["rank"].min()) if len(positives) else 101
            before = group[group["rank"].le(10) & group["rank"].lt(first) & group.relevance.eq(0)]
            matching = [
                r for r in before.to_dict("records") if eligible and answer in normalized(r["text"])
            ]
            ambiguity.append(
                {
                    "corpus": corpus,
                    "query_id": qid,
                    "answer_at_least_six_tokens": eligible,
                    "first_native_relevant_rank_saved100": first,
                    "unjudged_top10_before_positive_with_exact_answer": len(matching),
                    "answer_bearing_unjudged_ids_json": json.dumps(
                        [r["document_id"] for r in matching]
                    ),
                    "answer": original["query"]["answer"],
                }
            )
        report = json.loads(report_path.read_text())
        for macro in report["metrics"]["macro_by_regime"]:
            for cutoff in [1, 5, 10, 20, 100]:
                curves.append(
                    {
                        "corpus": corpus,
                        "regime": macro["regime"],
                        "k": cutoff,
                        "recall_pct": 100 * macro["metrics"][f"Recall@{cutoff}"],
                        "hit_pct": 100 * macro["metrics"][f"Hit@{cutoff}"],
                    }
                )
    ambiguity = pd.DataFrame(ambiguity)
    ambiguity.to_csv(output / "answer_bearing_unjudged_chunks.csv", index=False)
    ambiguity.groupby("corpus").agg(
        english_questions=("query_id", "size"),
        eligible_answer_lengths=("answer_at_least_six_tokens", "sum"),
        questions_with_answer_bearing_unjudged_above_positive=(
            "unjudged_top10_before_positive_with_exact_answer",
            lambda s: int(s.gt(0).sum()),
        ),
    ).to_csv(output / "answer_ambiguity_summary.csv")
    pd.DataFrame(curves).to_csv(output / "saved_score_curves.csv", index=False)
    # Language/settings controls use the original per-setting scores, retaining sample sizes.
    settings = pd.read_csv(root / "question_01_recall_at_10/tables/recall_at_10_by_setting.csv")
    setting = settings[
        settings.cohort.eq("selected_documents")
        & settings.model.eq("embeddinggemma")
        & settings.baseline.eq("dense")
    ].copy()
    setting.to_csv(output / "embeddinggemma_language_settings.csv", index=False)
    common_languages = ["en", "es", "fr"]
    common = setting[
        setting.query_language.isin(common_languages)
        & setting.target_language.isin(common_languages)
    ]
    controls = []
    for corpus in CORPORA:
        for name, subset in [
            ("original_settings", setting[setting.corpus.eq(corpus)]),
            ("common_en_es_fr_query_and_target_languages", common[common.corpus.eq(corpus)]),
        ]:
            for regime, group in subset.groupby("regime"):
                controls.append(
                    {
                        "corpus": corpus,
                        "selection": name,
                        "regime": regime,
                        "settings": len(group),
                        "equal_setting_recall_pct": 100 * group.recall_at_10.mean(),
                        "query_setting_weighted_recall_pct": 100
                        * np.average(group.recall_at_10, weights=group.query_count),
                        "smallest_setting_n": int(group.query_count.min()),
                    }
                )
    pd.DataFrame(controls).to_csv(output / "language_and_weighting_sensitivity.csv", index=False)
    # Figure 1: the different objectives, without pretending their gap is a causal decomposition.
    baseline = (
        scores[
            scores.scope.eq("all_questions")
            & scores.model.eq("embeddinggemma")
            & scores.regime.eq("mixed")
        ]
        .set_index("corpus")
        .loc[CORPORA]
    )
    baseline.to_csv(output / "mixed_version_and_source_scores.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(3)
    for offset, metric, label, color in [
        (-0.19, "Recall@10", "Recall@10: relevant-version coverage", "#327ca5"),
        (0.19, "Hit@10", "Hit@10: any relevant version", "#db9439"),
    ]:
        bars = ax.bar(x + offset, baseline[metric], width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_xticks(x, [LABELS[c] for c in CORPORA])
    ax.set_ylim(0, 114)
    ax.set_ylabel("Score (%)")
    ax.set_title("Mixed search · original queries · EmbeddingGemma", loc="left", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.08,
        0.03,
        "Chemistry: 2–3 relevant versions per question. EUR-Lex/UN: 4.\nEqual weight per nonempty original-query-language setting; all selected questions retained.",
        fontsize=9,
    )
    savefig(fig, output, "fig01_metric_definition")
    # Figure 2: an explicit language-controlled counterexample and lexical distinctiveness.
    en = scores[scores.scope.eq("english_original") & scores.regime.eq("monolingual")]
    model_order = [
        "embeddinggemma",
        "bge-m3",
        "qwen3-0.6B",
        "nomic-v2-moe",
        "granite-278m",
        "LaBSE",
        "SapBERT",
        "e5-large-instruct",
    ]
    english_scores = en.pivot(index="model", columns="corpus", values="Hit@10").reindex(model_order)
    english_scores.to_csv(output / "english_native_all_models.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    for i, corpus in enumerate(CORPORA):
        axes[0].bar(
            np.arange(8) + (i - 1) * 0.25,
            english_scores[corpus],
            width=0.24,
            color=COLORS[corpus],
            label=LABELS[corpus],
        )
    axes[0].set_xticks(range(8), model_order, rotation=40, ha="right")
    axes[0].set_ylabel("English → English Hit@10 / Recall@10 (%)")
    axes[0].set_ylim(0, 114)
    axes[0].legend(frameon=False, loc="upper center", ncol=3, fontsize=9)
    outranking = int(english_scores["eurlex"].gt(english_scores["chemistry"]).sum())
    axes[0].set_title(f"EUR-Lex exceeds chemistry for {outranking} of 8 encoders", loc="left")
    bars = axes[1].bar(
        [LABELS[c] for c in CORPORA],
        factors.english_questions_with_relative_rare_term_pct,
        color=[COLORS[c] for c in CORPORA],
    )
    axes[1].bar_label(bars, fmt="%.1f%%", padding=5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("English questions with a rare matching term (%)")
    axes[1].set_title("Rare matching vocabulary in English questions", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    fig.text(
        0.05,
        0.02,
        "English cohorts: chemistry n=46; EUR-Lex n=55; UN n=68. Descriptive comparisons on different questions.\nRare = non-numeric content token present in the relevant unit and in ≤0.1% of that corpus's English units.\nThe relative-frequency threshold controls the definition across pool sizes; it does not make the corpora or question cohorts equivalent.",
        fontsize=9,
    )
    savefig(fig, output, "fig02_english_controls")
    # Figure 3: actual text lengths and query-language composition.
    native = setting[setting.regime.eq("monolingual")]
    language_order = ["de", "en", "es", "fr", "zh"]
    matrix = native.pivot(
        index="query_language", columns="corpus", values="recall_at_10_pct"
    ).reindex(index=language_order, columns=CORPORA)
    counts = native.pivot(
        index="query_language", columns="corpus", values="query_count"
    ).reindex_like(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e9ebee")
    axes[0].imshow(
        np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=100, cmap=cmap, aspect="auto"
    )
    for i in range(5):
        for j in range(3):
            value = matrix.iloc[i, j]
            axes[0].text(
                j,
                i,
                "—" if pd.isna(value) else f"{value:.1f}%\n(n={int(counts.iloc[i, j])})",
                ha="center",
                va="center",
                color="white" if value >= 58 else "black",
                fontsize=10,
            )
    axes[0].set_xticks(range(3), [LABELS[c] for c in CORPORA])
    axes[0].set_yticks(range(5), language_order)
    axes[0].set_title("Native-language Recall@10 · EmbeddingGemma", loc="left", fontsize=12)
    axes[0].set_ylabel("Original question and target language")
    english_features = features[features.query_language.eq("en")]
    values = [english_features[english_features.corpus.eq(c)].native_source_words for c in CORPORA]
    boxes = axes[1].boxplot(
        values, tick_labels=[LABELS[c] for c in CORPORA], patch_artist=True, showfliers=False
    )
    for box, corpus in zip(boxes["boxes"], CORPORA, strict=True):
        box.set_facecolor(COLORS[corpus])
        box.set_alpha(0.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Word tokens in relevant English unit (log scale)")
    axes[1].set_title(
        "Chemistry abstracts are shorter than UN, longer than EUR-Lex", loc="left", fontsize=11
    )
    for i, corpus in enumerate(CORPORA, 1):
        median = factors.loc[corpus, "english_selected_source_median_words"]
        axes[1].annotate(
            f"median {median:g}",
            (i, median),
            xytext=(12, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.20, top=0.85, wspace=0.35)
    fig.text(
        0.05,
        0.02,
        "Left: same-language search, one relevant version per question; grey = unavailable. Different questions across languages.\nRight: selected English relevant units; boxes Q1–Q3, line median, whiskers 1.5 IQR, outliers hidden; lengths counted with fixed alphanumeric tokenization.",
        fontsize=9,
    )
    savefig(fig, output, "fig03_language_and_length")
    # Text samples are displayed with full relevant units, not just generated answers.
    case_files = sorted((output / "cases").glob("*/*.json"))
    case_index = []
    cards = []
    for path in case_files:
        case = json.loads(path.read_text())
        corpus = case["features"]["corpus"]
        query = case["query"]
        rank = case["features"]["first_native_relevant_rank_saved100"]
        rank_label = ">100 (censored)" if rank == 101 else str(rank)
        case_index.append(
            {
                "corpus": corpus,
                "query_id": query["id"],
                "first_relevant_rank": rank_label,
                "question": query["question"],
                "selection_rule": case["features"]["selection_rule"],
                "text_file": str(path.with_suffix(".txt").relative_to(output)),
            }
        )
        body = [
            f"<p><b>Question:</b> {html.escape(query['question'])}</p>",
            f"<p><b>Saved answer:</b> {html.escape(query['answer'])}</p>",
            f"<p>First relevant rank: {rank_label}. Quality: {case['features']['quality_status']}.</p>",
            f"<p><a href='{path.with_suffix('.txt').relative_to(output)}'>Complete inspection file</a></p>",
            f"<details><summary>Complete relevant source: {html.escape(case['relevant_source']['id'])}</summary><pre>{html.escape(case['relevant_source']['text'])}</pre></details>",
        ]
        for row in case["saved_top3_and_first_positive"]:
            body.append(
                f"<details><summary>Rank {row['rank']} · saved relevance {row['relevance']} · {html.escape(row['document_id'])}</summary><pre>{html.escape(row['text'])}</pre></details>"
            )
        cards.append(
            f"<details><summary>{LABELS[corpus]} · rank {rank_label} · {html.escape(query['question'][:100])}</summary>{''.join(body)}</details>"
        )
    pd.DataFrame(case_index).to_csv(output / "case_index.csv", index=False)
    gallery = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Why chemistry scores higher</title>
<style>body{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#192638}img{width:100%;height:auto}details{border:1px solid #d5dde8;padding:14px;margin:14px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;max-height:500px;overflow:auto;background:#f5f7fa;padding:14px}a{color:#145ca0}</style>
<h1>Corpus difficulty: evidence from questions, documents and saved rankings</h1>
<p>Original-query dense retrieval. Text examples use EmbeddingGemma / English → English to keep language, retriever and one-version relevance fixed. Corpus/question cohorts still differ.</p>
<p><a href="findings.txt">Findings and manually reviewed examples</a> · <a href="corpus_factors.csv">Corpus factors</a> · <a href="case_index.csv">Case index</a> · <a href="query_features.csv">All 638 query features</a></p>
<p>The 21 text cases are a seeded sample of up to four English-native hits and four misses per corpus; EUR-Lex has only one English miss. This is diagnostic sampling, not an estimate of success prevalence. All full relevant source texts and top-three winning chunks are retained. Relevance 0 means unjudged under known-source qrels.</p>"""
    for name in ["fig01_metric_definition", "fig02_english_controls", "fig03_language_and_length"]:
        gallery += f'<p><a href="{name}.png"><img src="{name}.png" alt="{name}"></a><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p>'
    (output / "index.html").write_text(
        gallery + "<h2>Actual text cases</h2>" + "\n".join(cards) + "</html>"
    )
    print("ANSWER AMBIGUITY\n", (output / "answer_ambiguity_summary.csv").read_text())
    print("LANGUAGE/WEIGHTING CONTROLS\n", pd.DataFrame(controls).round(3).to_string(index=False))
    print(f"Created three figures and a gallery of {len(case_files)} full-text cases")


if __name__ == "__main__":
    main()
