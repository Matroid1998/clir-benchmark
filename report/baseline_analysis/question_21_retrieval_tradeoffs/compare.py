"""Create matched control comparisons, preserving every translation language separately."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from score import ART, CACHE, HERE, RECALL, TIMES

CORPORA = ("chemistry", "eurlex", "un")
RETRIEVERS = ("bm25", "dense", "hybrid")
REGIMES = ("monolingual", "cross_language", "mixed")
GROUP = ["corpus", "retriever", "regime", "method", "route_summary"]
MATCH = ["corpus", "model", "retriever", "setting_id", "qrels_hash", "queries"]
VALUES = (
    RECALL
    + TIMES
    + [
        "mean_known_api_usd",
        "control_recall10",
        "control_recall50",
        "delta_recall10_pp",
        "delta_recall50_pp",
        "control_paired_s",
        "variant_paired_s",
        "delta_pipeline_s",
    ]
)


def summaries(frame, dims):
    # First summarize independent translation routes within a setting. Each model and
    # setting then has one vote irrespective of how many translation routes it has.
    grouped = frame.groupby(dims + ["model", "setting_id"], dropna=False, sort=False)
    settings = grouped[VALUES].mean().reset_index()
    means = settings.groupby(dims + ["model"], dropna=False)[VALUES].mean().reset_index()
    result = means.groupby(dims, dropna=False)[VALUES].mean()
    result["n_models"] = means.groupby(dims, dropna=False).model.nunique()
    counts = frame.groupby(dims, dropna=False)
    result["query_setting_route_observations"] = counts.queries.sum()
    result["timed_observations"] = counts.latency_observations.sum()
    result["paired_timed_observations"] = counts.paired_latency_observations.sum()
    result["run_count"] = counts.run_id.nunique()
    return result.reset_index()


def plots(matched, generation):
    colors = {"translate": "#286ca6", "hyde": "#c26324"}
    for cutoff in (10, 50):
        fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True, sharey=True)
        for i, regime in enumerate(REGIMES):
            for j, corp in enumerate(CORPORA):
                ax = axes[i, j]
                for shift, (method, color) in zip((-0.18, 0.18), colors.items(), strict=True):
                    rows = matched[
                        (matched.regime == regime)
                        & (matched.corpus == corp)
                        & (matched.method == method)
                    ]
                    rows = rows[
                        rows.route_summary.isin(
                            (
                                "target_language",
                                "mean_independent_languages"
                                if regime == "mixed"
                                else "target_language",
                                "question_language",
                            )
                        )
                    ]
                    vals = rows.set_index("retriever").reindex(RETRIEVERS)[
                        f"delta_recall{cutoff}_pp"
                    ]
                    ax.bar(
                        np.arange(3) + shift,
                        vals,
                        width=0.33,
                        color=color,
                        label="Translation" if method == "translate" else "HyDE",
                    )
                ax.axhline(0, lw=0.8, color="#333333")
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
                ax.set_xticks(range(3), ["BM25", "Dense", "Hybrid"])
                if i == 0:
                    ax.set_title({"chemistry": "Chemistry", "eurlex": "EUR-Lex", "un": "UN"}[corp])
                if j == 0:
                    ax.set_ylabel(
                        {
                            "cross_language": "Cross-language",
                            "mixed": "Mixed",
                            "monolingual": "Monolingual",
                        }[regime]
                        + "\nΔ recall (pp)"
                    )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False
        )
        fig.suptitle(f"Change in chunk Recall@{cutoff} versus the same original retriever", y=0.985)
        fig.text(
            0.5,
            0.015,
            "Translation uses the searched language in single-language pools. Mixed averages independently searched translations.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.89])
        fig.savefig(HERE / f"recall{cutoff}_change.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for shift, (method, color) in zip((-0.18, 0.18), colors.items(), strict=True):
        d = generation[generation.method == method].set_index("corpus").reindex(CORPORA)
        for ax, col in zip(axes, ["known_attempt_usd", "mean_known_generation_s"], strict=True):
            ax.bar(np.arange(3) + shift, d[col], 0.34, label=method, color=color)
            ax.set_xticks(range(3), ["Chemistry", "EUR-Lex", "UN"])
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    axes[0].set_ylabel("Known API charges (USD)")
    axes[1].set_ylabel("Mean generation time per variant (s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "generation_cost_latency.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    labels = {"original": "Original", "translate": "Translation", "hyde": "HyDE"}
    stages = [
        ("generation_s", "LLM generation", "#b85b29"),
        ("encoding_s", "Query encoding", "#8ab8cb"),
        ("retrieval_s", "Search and fusion", "#30597a"),
    ]
    for ax, corpus in zip(axes, CORPORA, strict=True):
        selected = matched[
            (matched.corpus == corpus)
            & (matched.regime == "cross_language")
            & matched.route_summary.isin(("original", "target_language", "question_language"))
        ]
        index = pd.MultiIndex.from_product([RETRIEVERS, labels], names=["retriever", "method"])
        rows = selected.set_index(["retriever", "method"]).reindex(index)
        left = np.zeros(len(index))
        for key, label, color in stages:
            ax.barh(range(len(rows)), rows[key], left=left, label=label, color=color)
            left += rows[key].to_numpy()
        ax.set_title({"chemistry": "Chemistry", "eurlex": "EUR-Lex", "un": "UN"}[corpus])
        ax.set_yticks(
            range(len(index)),
            [f"{r.upper() if r == 'bm25' else r.title()} · {labels[m]}" for r, m in index],
        )
        ax.set_xlabel("Seconds per query and search setting")
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False
    )
    fig.suptitle("Cross-language retrieval time by stage")
    fig.text(
        0.5,
        0.015,
        "Generation uses recorded request times. Encoding and search are batch-amortized. Hybrid includes both searches and fusion.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.86])
    fig.savefig(HERE / "latency_by_stage.png", dpi=170)
    plt.close(fig)


def main():
    frames = [pd.read_csv(f, keep_default_na=True) for f in sorted(CACHE.glob("*.csv"))]
    scores = pd.concat(frames, ignore_index=True)
    expected = {"original": 48, "translate": 289, "hyde": 51}
    assert scores.groupby("method").run_id.nunique().to_dict() == expected, "Incomplete rescoring"
    assert (
        "paired_latency_observations" in scores and scores.paired_latency_observations.notna().all()
    )
    saved = pd.read_csv(
        HERE.parent / "question_20_local_results" / "retrieval_settings.csv",
        usecols=["run_id", "setting_id", "Recall@10", "Recall@20", "Recall@100"],
    )
    checked = scores.merge(saved, on=["run_id", "setting_id"], validate="one_to_one")
    assert len(checked) == len(saved)
    np.testing.assert_allclose(checked.recall10, checked["Recall@10"], rtol=0, atol=1e-12)
    assert (checked.recall50 >= checked["Recall@20"] - 1e-12).all()
    assert (checked.recall50 <= checked["Recall@100"] + 1e-12).all()
    ctrl = scores[(scores.method == "original") & (scores.report_status == "published")][
        MATCH + RECALL + ["run_id"]
    ]
    d = scores.merge(
        ctrl.rename(columns={c: "control_" + c for c in RECALL + ["run_id"]}),
        on=MATCH,
        how="left",
        validate="many_to_one",
    )
    for k in (10, 50):
        d[f"delta_recall{k}_pp"] = 100 * (d[f"recall{k}"] - d[f"control_recall{k}"])
    d["delta_pipeline_s"] = d.variant_paired_s - d.control_paired_s
    stages = []
    for m in (
        d[["run_id", "method", "retriever", "corpus"]].drop_duplicates().itertuples(index=False)
    ):
        if m.retriever != "hybrid":
            continue
        folder = (
            "narval-20261119-shared-indexes"
            if m.method == "original"
            else "narval-20261120-transformed-retrieval"
        )
        t = pd.read_parquet(
            ART / folder / "artifacts" / "runs" / m.corpus / m.run_id / "timings.parquet"
        )
        cols = ["bm25_seconds_amortized", "dense_seconds_amortized", "fusion_seconds_amortized"]
        expected = d[d.run_id == m.run_id].set_index("setting_id").refused
        skipped = t.retrieval_seconds_amortized.isna()
        np.testing.assert_array_equal(
            skipped.groupby(t.setting_id).sum().reindex(expected.index), expected
        )
        assert not t.loc[~skipped, cols].isna().any().any()
        t.loc[skipped, cols] = 0.0
        t = t.groupby("setting_id")[cols].mean().reset_index()
        stages.append(t.assign(run_id=m.run_id))
    d = d.merge(pd.concat(stages), on=["run_id", "setting_id"], how="left", validate="one_to_one")
    d.sort_values(["corpus", "retriever", "model", "method", "language", "setting_id"]).to_csv(
        HERE / "by_setting.csv", index=False
    )
    clean = d[d.report_status != "incomplete"].copy()
    translated = clean[clean.method == "translate"]
    all_routes = translated.assign(route_summary="mean_independent_languages")
    target = translated[translated.language == translated.target_languages].assign(
        route_summary="target_language"
    )
    identity = translated[translated.language == translated.query_language].assign(
        route_summary="identity_language"
    )
    alternatives = translated[
        (translated.regime != "mixed") & (translated.language != translated.target_languages)
    ].assign(route_summary="other_than_target")
    original = clean[clean.method == "original"].assign(route_summary="original")
    hyde = clean[clean.method == "hyde"].assign(route_summary="question_language")
    selected = pd.concat(
        [original, hyde, target, all_routes, identity, alternatives], ignore_index=True
    )
    paired = selected[selected.control_run_id.notna()]
    absolute = summaries(selected, GROUP)
    matched = summaries(paired, GROUP)
    absolute.drop(
        columns=[
            c
            for c in absolute
            if c.startswith(("control_", "delta_", "variant_paired_", "paired_timed_"))
        ]
    ).to_csv(HERE / "all_available_by_regime.csv", index=False)
    matched.to_csv(HERE / "matched_by_regime.csv", index=False)
    selected.groupby(GROUP + ["model", "setting_id"])[RECALL + TIMES].mean().groupby(
        level=GROUP + ["model"]
    ).mean().reset_index().to_csv(HERE / "all_available_by_model.csv", index=False)
    paired.groupby(GROUP + ["model", "setting_id"])[VALUES].mean().groupby(
        level=GROUP + ["model"]
    ).mean().reset_index().to_csv(HERE / "matched_by_model.csv", index=False)
    individual = clean.assign(route_summary=clean.language)
    summaries(individual[individual.control_run_id.notna()], GROUP).to_csv(
        HERE / "matched_by_translation_language.csv", index=False
    )
    plots(matched, pd.read_csv(HERE / "generation_by_corpus.csv"))


if __name__ == "__main__":
    main()
