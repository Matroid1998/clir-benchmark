"""Render the results tables from the saved comparisons."""

import pandas as pd
from score import HERE

NAMES = {
    "chemistry": "Chemistry",
    "eurlex": "EUR-Lex",
    "un": "UN",
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
}
CORPORA = ["chemistry", "eurlex", "un"]
BACKENDS = ["bm25", "dense", "hybrid"]


def table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def metrics(d, regime, timed=False):
    rows = []
    for corpus in CORPORA:
        for backend in BACKENDS:
            s = d[(d.corpus == corpus) & (d.retriever == backend) & (d.regime == regime)].set_index(
                "route_summary"
            )
            a = s.loc["original"]
            t = s.loc["mean_independent_languages" if regime == "mixed" else "target_language"]
            h = s.loc["question_language"]
            if timed:
                values = [
                    f"{a.pipeline_s:.2f}",
                    f"{t.variant_paired_s:.2f} ({t.delta_pipeline_s:+.2f})",
                    f"{h.variant_paired_s:.2f} ({h.delta_pipeline_s:+.2f})",
                ]
            else:
                values = [
                    f"{100 * a.recall10:.1f} / {100 * a.recall50:.1f}",
                    f"{t.delta_recall10_pp:+.1f} / {t.delta_recall50_pp:+.1f}",
                    f"{h.delta_recall10_pp:+.1f} / {h.delta_recall50_pp:+.1f}",
                ]
            rows.append(
                [
                    NAMES[corpus],
                    NAMES[backend],
                    int(a.n_models) if backend != "bm25" else "—",
                    *values,
                ]
            )
    cols = ["Corpus", "Retriever", "Models"]
    if timed:
        cols += ["Original (s)", "Translation (s, Δ)", "HyDE (s, Δ)"]
    else:
        cols += ["Original @10 / @50 (%)", "Translation Δ @10 / @50 (pp)", "HyDE Δ @10 / @50 (pp)"]
    return table(cols, rows)


def main():
    d = pd.read_csv(HERE / "matched_by_regime.csv")
    gen = pd.read_csv(HERE / "generation_by_method.csv").set_index("method")
    ledger = pd.read_csv(HERE / "request_ledger.csv")
    costs = []
    for method in ("translate", "hyde"):
        x = gen.loc[method]
        costs.append(
            [
                "Translation" if method == "translate" else "HyDE",
                f"{int(x.query_variants):,}",
                f"{int(x.attempts):,}",
                f"{int(x.refusals):,}",
                f"${x.known_attempt_usd:.4f}",
                int(x.missing_attempt_costs),
                f"{x.mean_known_generation_s:.2f}",
                f"{x.p95_known_generation_s:.2f}",
            ]
        )
    content = [
        "# Translation and HyDE on the verified November 2026 cohort",
        "The evaluation covers 5,299 questions and 388 saved rankings across Chemistry patents, EUR-Lex and UN. Recall is the fraction of distinct relevant chunks found within the first 10 or 50 results. Percentage-point changes use the original query with the same retriever, embedding model, search pool, questions and judgments.",
        "## Cross-language search",
        "Each translated query uses the language of the collection being searched. HyDE generates a passage in the original question language. The tables average questions within each setting, then weight language settings and embedding models equally. BM25 has one lexical index per corpus.",
        metrics(d, "cross_language"),
        "## Monolingual search",
        "Translating to the collection language takes the identity route in these settings. It keeps the original text and incurs zero API charges.",
        metrics(d, "monolingual"),
        "## Mixed-language search",
        "The mixed pool uses the existing union of chunk judgments across supported document languages. The translation column averages separately evaluated output languages with equal weight. Each ranking is scored independently.",
        metrics(d, "mixed"),
        "![Recall at 10 effects](recall10_change.png)\n\n![Recall at 50 effects](recall50_change.png)",
        "## Latency",
        "The table gives mean seconds per question and search setting for cross-language retrieval. Pipeline time adds recorded generation, batch-amortized query encoding, and search. Hybrid includes its dense and BM25 searches and reciprocal rank fusion. Each variant and control latency difference uses the same queries with known timings. The original column covers the whole eligible cohort. Search timings were measured in separate offline jobs, so cache state, batching and node performance can also affect their differences. Online end-to-end and parallel service latency were not measured.",
        metrics(d, "cross_language", True),
        "![Latency by stage](latency_by_stage.png)",
        "## Generation cost",
        table(
            [
                "Method",
                "Retained variants",
                "API attempts",
                "Refusals",
                "Known USD",
                "Unpriced attempts",
                "Mean generation (s)",
                "P95 generation (s)",
            ],
            costs,
        ),
        f"The retained variants account for at least ${gen.known_attempt_usd.sum():.4f}. Across all recorded request keys, including requests outside the retained cohort, known charges total ${ledger.known_attempt_usd.sum():.4f}. Each physical attempt is counted once, including retries linked to an earlier request key. The 5,299 same-language translations required no call. Generation was cached before retrieval and reused across indexes and models. The analysis made zero API calls. The recorded provider fees exclude unpriced cluster compute.",
        "Mean generation time includes identity translations and recorded refusal attempts. The mean for a translation that required an API call is "
        + f"{gen.loc['translate', 'mean_known_paid_generation_s']:.2f} s. "
        + "Missing request durations and charges are excluded from observed averages. Provider-reported token charges and configured prices are recorded separately.",
        "![Generation cost and latency](generation_cost_latency.png)",
        "## Coverage and reproducibility",
        "The primary comparisons require a published control. Chemistry has seven comparable dense and seven hybrid models, EUR-Lex has eight of each, and UN has six dense and five hybrid models. The absolute-result tables retain all eight models for translation and HyDE. Four original jobs wrote rankings and metric tables before their final checks failed, and three original hybrid jobs stopped before producing rankings. The BM25 translation and HyDE numbers come from rescoring their saved component rankings.",
        "Recall@50 was calculated directly from the top-100 chunk IDs and binary chunk judgments. A chunk overlapping multiple evidence spans is counted once. All eligible questions remain in the denominator, including refused queries with empty rankings. Recomputed Recall@10 matches every available per-query report. Recall@50 lies between saved Recall@20 and Recall@100 in all 11,882 previously reported settings.",
        "The [per-setting table](by_setting.csv) includes corpus, model, query language, translation language, search pool, absolute scores, paired changes, counts and timing components. [Per-model comparisons](matched_by_model.csv), [independent translation languages](matched_by_translation_language.csv), [matched corpus and regime summaries](matched_by_regime.csv), and [absolute scores across all available models](all_available_by_model.csv) retain the alternative views. The [generation ledger](request_ledger.csv), [corpus-level charges](generation_by_corpus.csv), [route and model charges](generation_by_language_model.csv) and [configured prices](configured_pricing.csv) document cost and usage.",
        "Run the following from the repository root after downloading the results. The scripts read local artifacts.",
        "```bash\n.venv/bin/python report/baseline_analysis/question_21_retrieval_tradeoffs/cost.py\n.venv/bin/python report/baseline_analysis/question_21_retrieval_tradeoffs/score.py\n.venv/bin/python report/baseline_analysis/question_21_retrieval_tradeoffs/compare.py\n.venv/bin/python report/baseline_analysis/question_21_retrieval_tradeoffs/all_models.py\n.venv/bin/python report/baseline_analysis/question_21_retrieval_tradeoffs/report.py\n```",
    ]
    sensitivity = HERE / "all_models_recall.csv"
    if sensitivity.exists():
        full = pd.read_csv(sensitivity)
        rows = []
        for corpus in CORPORA:
            for backend in ("dense", "hybrid"):
                s = full[
                    (full.corpus == corpus)
                    & (full.retriever == backend)
                    & (full.regime == "cross_language")
                ].set_index("method")
                rows.append(
                    [
                        NAMES[corpus],
                        NAMES[backend],
                        *[
                            f"{s.loc[m, 'delta10_pp']:+.1f} / {s.loc[m, 'delta50_pp']:+.1f}"
                            for m in ("translate", "hyde")
                        ],
                    ]
                )
        content.extend(
            [
                "## All eight models",
                "This recall-only sensitivity check uses the saved rankings for four incomplete original controls and reconstructs three missing original hybrid controls from their saved BM25 and dense candidates. The reconstruction used the configured reciprocal rank fusion and reproduced a published hybrid control in both affected corpora. The table includes all eight models for each corpus and backend. Reconstructed controls carry no latency estimate. [All-model results for every regime](all_models_recall.csv) and [control provenance](all_model_controls.csv).",
                table(
                    [
                        "Corpus",
                        "Retriever",
                        "Translation Δ @10 / @50 (pp)",
                        "HyDE Δ @10 / @50 (pp)",
                    ],
                    rows,
                ),
            ]
        )
    (HERE / "README.md").write_text("\n\n".join(content) + "\n")


if __name__ == "__main__":
    main()
