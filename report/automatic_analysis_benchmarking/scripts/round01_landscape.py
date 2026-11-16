"""Round 1: inspect coverage and the complete mixed-retrieval landscape."""
from common import *


def main():
    out = ROOT / "round_01"
    d = pd.read_parquet(ROOT / "data/per_query.parquet")
    q = pd.read_parquet(ROOT / "data/selected_queries.parquet")
    mixed = d[d.regime.eq("mixed")].copy()
    assert mixed.groupby(["corpus", "model", "variant"]).size().nunique() == 3
    variants = ["original"] + [f"translate_{x}" for x in LANGUAGES] + [f"hyde_{x}" for x in LANGUAGES]
    # Extractor's original variant is intentionally explicit; don't guess its name.
    original_variants = mixed.loc[mixed.method.eq("original"), "variant"].unique()
    assert len(original_variants) == 1
    variants[0] = original_variants[0]
    summary = mixed.groupby(["corpus", "model", "method", "variant", "output_language"], observed=True).agg(
        n=("query_id", "nunique"), ndcg10=("ndcg10", "mean"), hit10=("hit10", "mean"),
        recall10=("recall10", "mean"), mrr10=("mrr10", "mean"), hit1=("hit1", "mean"),
    ).reset_index()
    table(summary, out, "table_01_complete_mixed_results")

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), layout="constrained")
    labels = ["Original"] + [f"T-{x}" for x in LANGUAGES] + [f"H-{x}" for x in LANGUAGES]
    for ax, corpus in zip(axes, CORPORA):
        a = summary[summary.corpus.eq(corpus)].pivot(index="model", columns="variant", values="ndcg10").reindex(index=MODELS, columns=variants)
        im = ax.imshow(a, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(8), [MODEL_LABEL[x] for x in MODELS])
        ax.set_xticks(range(11), labels)
        ax.set_title(f"{CORPUS_LABEL[corpus]}: mixed-language search, {q[q.corpus.eq(corpus)].shape[0]} source documents", loc="left")
        for i in range(8):
            for j in range(11):
                value = a.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value < .62 else "black")
        ax.axvline(.5, color="white", lw=1.5)
        ax.axvline(5.5, color="white", lw=1.5)
    fig.colorbar(im, ax=axes, label="Document-language version nDCG@10", shrink=.65, pad=.02)
    savefig(fig, out, "fig_01_mixed_landscape")

    original = d[d.method.eq("original") & d.model.eq(MODELS[0])]
    coverage = original.groupby(["corpus", "query_language", "target_language"], observed=True).query_id.nunique().rename("eligible_queries").reset_index()
    table(coverage, out, "table_02_language_coverage")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), layout="constrained")
    for ax, corpus in zip(axes, CORPORA):
        a = coverage[coverage.corpus.eq(corpus)].pivot(index="query_language", columns="target_language", values="eligible_queries").reindex(index=LANGUAGES, columns=LANGUAGES + ["mixed"]).fillna(0)
        im = ax.imshow(a, cmap="Blues", vmin=0, vmax=80, aspect="auto")
        ax.set_xticks(range(6), LANGUAGES + ["mix"])
        ax.set_yticks(range(5), LANGUAGES)
        ax.set_title(CORPUS_LABEL[corpus]); ax.set_xlabel("Searched language")
        for i in range(5):
            for j in range(6):
                value = int(a.iloc[i, j]); ax.text(j, i, str(value) if value else "—", ha="center", va="center", fontsize=8,
                                               color="white" if value > 50 else "black")
    axes[0].set_ylabel("Original query language")
    fig.colorbar(im, ax=axes, label="Eligible source questions", shrink=.8, pad=.02)
    savefig(fig, out, "fig_02_cohort_coverage")

    # Equal weighting of the five generation languages is a diagnostic average,
    # not a claim that a deployment should select a language uniformly at random.
    query_methods = mixed.groupby(["corpus", "model", "query_id", "method"], observed=True)[["ndcg10", "hit10", "recall10"]].mean().reset_index()
    deltas = []
    for (corpus, model), g in query_methods.groupby(["corpus", "model"], observed=True):
        for metric in ["ndcg10", "hit10", "recall10"]:
            a = g.pivot(index="query_id", columns="method", values=metric)
            assert not a.isna().any().any()
            for left, right in [("hyde", "translate"), ("hyde", "original"), ("translate", "original")]:
                value, lo, hi = paired_bootstrap(a[left] - a[right])
                deltas.append(dict(corpus=corpus, model=model, contrast=f"{left}-{right}", metric=metric, n=len(a),
                                   delta=value, ci_low=lo, ci_high=hi))
    delta = pd.DataFrame(deltas)
    table(delta, out, "table_03_paired_mixed_deltas")
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.5), layout="constrained")
    for i, metric in enumerate(["ndcg10", "hit10"]):
        for j, contrast in enumerate(["hyde-translate", "hyde-original"]):
            a = delta[delta.metric.eq(metric) & delta.contrast.eq(contrast)].pivot(index="model", columns="corpus", values="delta").reindex(index=MODELS, columns=CORPORA)
            im = centered_heatmap(axes[i, j], a * 100, [MODEL_LABEL[x] for x in MODELS], [CORPUS_LABEL[x] for x in CORPORA], limit=45, decimals=1)
            axes[i, j].set_title(f"{'HyDE − translation' if j == 0 else 'HyDE − original'}: {'nDCG@10' if metric == 'ndcg10' else 'SourceHit@10'}")
    fig.colorbar(im, ax=axes, label="Percentage-point difference (five-language mean)", shrink=.7, pad=.02)
    savefig(fig, out, "fig_03_control_choice_changes_conclusion")

    broad = query_methods.groupby(["corpus", "method"], observed=True)[["ndcg10", "hit10", "recall10"]].mean().reset_index()
    table(broad, out, "table_04_corpus_method_diagnostic_means")
    counts = q.groupby(["corpus", "query_language"], observed=True).size().rename("questions").reset_index()
    table(counts, out, "table_05_selected_query_languages")
    write_json(out / "round_manifest.json", {
        "round": 1, "question": "What coverage and broad patterns must later analyses explain?",
        "source": "data/per_query.parquet", "reports": 264, "queries": 638,
        "primary_estimand": "Within-corpus mixed retrieval; equal source-question weight; model-specific results; diagnostic means give each of five output languages equal weight.",
        "bootstrap": "5000 paired query/source resamples within each corpus and model; intervals are exploratory; no independent replication claimed across model/language rows.",
        "figures": ["fig_01_mixed_landscape", "fig_02_cohort_coverage", "fig_03_control_choice_changes_conclusion"],
        "tables": 5, "visual_review": "pending",
    })
    print(broad.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
