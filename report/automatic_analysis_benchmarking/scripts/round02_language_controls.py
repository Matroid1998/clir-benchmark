#!/usr/bin/env python3
"""Round 2: matched output languages and absent-corpus-language pivots.

Reads only the offline extracted data. Each confidence interval resamples one
paired value per original source question; target settings and fixed encoders
are averaged before resampling when a panel pools them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    CORPORA,
    CORPUS_LABEL,
    LANGUAGES,
    MODEL_LABEL,
    MODELS,
    ROOT,
    paired_bootstrap,
    savefig,
    table,
    write_json,
)

OUT = ROOT / "round_02"
OUT.mkdir(exist_ok=True)
METRICS = ["ndcg10", "hit10", "recall10"]
METRIC_LABEL = {
    "ndcg10": "nDCG@10",
    "hit10": "Hit@10: any source version",
    "recall10": "Recall@10: language versions",
}
ALL = "All fixed models"
DISPLAY_MODELS = [*MODELS, ALL]
DISPLAY_LABELS = [*(MODEL_LABEL[m] for m in MODELS), "Fixed-model mean"]
PIVOTS = {"un": "de", "eurlex": "zh"}
PIVOT_NAME = {"un": "German", "eurlex": "Chinese"}
INTERVAL_NOTE = (
    "95% paired source-question bootstrap intervals; exploratory, no multiplicity correction"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(values):
    point, low, high = paired_bootstrap(values)
    return {"estimate": point, "ci_low": low, "ci_high": high}


def add_fixed_model_mean(frame, keys, columns):
    """Append a fixed-encoder average, retaining one row per query cluster."""
    counts = frame.groupby(keys, observed=True)["model"].nunique()
    assert counts.eq(8).all()
    averaged = frame.groupby(keys, observed=True)[columns].mean().reset_index()
    averaged["model"] = ALL
    return pd.concat([frame, averaged], ignore_index=True)


def paired_wide(frame, keys, condition, expected):
    assert not frame.duplicated(keys + [condition]).any()
    result = frame.set_index(keys + [condition])[METRICS].unstack(condition)
    assert set(result.columns.get_level_values(1)) == set(expected)
    assert result.notna().all().all()
    result.columns = [f"{metric}__{name}" for metric, name in result.columns]
    return result.reset_index()


def target_matched(frame):
    matched = frame[
        frame["regime"].eq("cross_language")
        & (frame["method"].eq("original") | frame["output_language"].eq(frame["target_language"]))
    ].copy()
    keys = ["corpus", "model", "query_id", "query_language", "setting_id", "target_language"]
    wide = paired_wide(matched, keys, "method", ["original", "translate", "hyde"])
    score_columns = [c for c in wide.columns if "__" in c]
    pairs_by_corpus = (
        wide.groupby("corpus", observed=True)
        .apply(
            lambda g: len(g[["query_id", "target_language"]].drop_duplicates()),
            include_groups=False,
        )
        .to_dict()
    )
    per_query = (
        wide.groupby(["corpus", "model", "query_id", "query_language"], observed=True)[
            score_columns
        ]
        .mean()
        .reset_index()
    )
    per_query = add_fixed_model_mean(
        per_query, ["corpus", "query_id", "query_language"], score_columns
    )
    per_query.to_parquet(OUT / "target_matched_query_scores.parquet", index=False)
    levels, contrasts = [], []
    pairs = [("translate", "original"), ("hyde", "original"), ("hyde", "translate")]
    for (corpus, model), group in per_query.groupby(["corpus", "model"], observed=True):
        common = {
            "corpus": corpus,
            "model": model,
            "n_queries": len(group),
            "n_query_target_pairs": pairs_by_corpus[corpus],
            "fixed_models": 8 if model == ALL else 1,
        }
        for metric in METRICS:
            for method in ("original", "translate", "hyde"):
                levels.append(
                    {
                        **common,
                        "metric": metric,
                        "method": method,
                        **estimate(group[f"{metric}__{method}"]),
                    }
                )
            for method, reference in pairs:
                differences = group[f"{metric}__{method}"] - group[f"{metric}__{reference}"]
                contrasts.append(
                    {
                        **common,
                        "metric": metric,
                        "contrast": f"{method}-minus-{reference}",
                        "method_mean": group[f"{metric}__{method}"].mean(),
                        "reference_mean": group[f"{metric}__{reference}"].mean(),
                        **estimate(differences),
                    }
                )
    levels, contrasts = pd.DataFrame(levels), pd.DataFrame(contrasts)
    table(levels, OUT, "table01_target_matched_levels")
    table(contrasts, OUT, "table02_target_matched_contrasts")
    return levels, contrasts, per_query


def language_surfaces(frame):
    selected = frame[frame["method"].isin(["translate", "hyde"])]
    keys = [
        "corpus",
        "model",
        "query_id",
        "query_language",
        "setting_id",
        "target_language",
        "output_language",
    ]
    wide = paired_wide(selected, keys, "method", ["translate", "hyde"])
    for metric in METRICS:
        wide[metric] = wide[f"{metric}__hyde"] - wide[f"{metric}__translate"]
    per_query = (
        wide.groupby(["corpus", "query_id", "target_language", "output_language"], observed=True)[
            METRICS
        ]
        .mean()
        .reset_index()
    )
    fixed_counts = wide.groupby(
        ["corpus", "query_id", "target_language", "output_language"], observed=True
    )["model"].nunique()
    assert fixed_counts.eq(8).all()
    per_query.to_parquet(OUT / "surface_query_deltas.parquet", index=False)
    targets = [*LANGUAGES, "mixed"]
    pooled, specific = [], []
    for corpus in CORPORA:
        for language in LANGUAGES:
            for target in targets:
                cell = per_query[
                    (per_query["corpus"] == corpus)
                    & (per_query["output_language"] == language)
                    & (per_query["target_language"] == target)
                ]
                for metric in METRICS:
                    pooled.append(
                        {
                            "corpus": corpus,
                            "output_language": language,
                            "target_language": target,
                            "metric": metric,
                            "n_queries": len(cell),
                            "fixed_models": 8,
                            "available": bool(len(cell)),
                            **estimate(cell[metric]),
                        }
                    )
                for model in MODELS:
                    model_cell = wide[
                        (wide["corpus"] == corpus)
                        & (wide["output_language"] == language)
                        & (wide["target_language"] == target)
                        & (wide["model"] == model)
                    ]
                    for metric in METRICS:
                        specific.append(
                            {
                                "corpus": corpus,
                                "model": model,
                                "output_language": language,
                                "target_language": target,
                                "metric": metric,
                                "n_queries": len(model_cell),
                                "available": bool(len(model_cell)),
                                "estimate": model_cell[metric].mean(),
                            }
                        )
    pooled, specific = pd.DataFrame(pooled), pd.DataFrame(specific)
    table(pooled, OUT, "table03_same_language_surfaces")
    table(specific, OUT, "table04_same_language_surfaces_by_model")
    return pooled, specific


def absent_pivots(frame):
    pieces = []
    for corpus, pivot in PIVOTS.items():
        source = frame[(frame["corpus"] == corpus) & frame["regime"].eq("mixed")]
        assert pivot not in set(source["query_language"])
        assert pivot not in set(frame.loc[frame["corpus"] == corpus, "target_language"])
        chosen = source[
            source["method"].eq("original")
            | (source["method"].eq("translate") & source["output_language"].isin(["en", pivot]))
        ].copy()
        chosen["condition"] = np.where(
            chosen["method"].eq("original"),
            "original",
            np.where(
                chosen["output_language"].eq("en"), "english_translation", "absent_translation"
            ),
        )
        keys = ["corpus", "model", "query_id", "query_language"]
        pieces.append(
            paired_wide(
                chosen, keys, "condition", ["original", "english_translation", "absent_translation"]
            )
        )
    per_query = pd.concat(pieces, ignore_index=True)
    score_columns = [c for c in per_query.columns if "__" in c]
    per_query = add_fixed_model_mean(
        per_query, ["corpus", "query_id", "query_language"], score_columns
    )
    per_query.to_parquet(OUT / "pivot_query_scores.parquet", index=False)
    levels, contrasts, strata = [], [], []
    for (corpus, model), group in per_query.groupby(["corpus", "model"], observed=True):
        common = {
            "corpus": corpus,
            "model": model,
            "pivot_language": PIVOTS[corpus],
            "n_queries": len(group),
            "fixed_models": 8 if model == ALL else 1,
        }
        for metric in METRICS:
            for condition in ("original", "english_translation", "absent_translation"):
                levels.append(
                    {
                        **common,
                        "metric": metric,
                        "condition": condition,
                        **estimate(group[f"{metric}__{condition}"]),
                    }
                )
            for reference in ("original", "english_translation"):
                delta = group[f"{metric}__absent_translation"] - group[f"{metric}__{reference}"]
                contrasts.append(
                    {
                        **common,
                        "metric": metric,
                        "contrast": f"absent-minus-{reference}",
                        "pivot_mean": group[f"{metric}__absent_translation"].mean(),
                        "reference_mean": group[f"{metric}__{reference}"].mean(),
                        **estimate(delta),
                    }
                )
        for language, subset in group.groupby("query_language", observed=True):
            for metric in METRICS:
                for reference in ("original", "english_translation"):
                    delta = (
                        subset[f"{metric}__absent_translation"] - subset[f"{metric}__{reference}"]
                    )
                    strata.append(
                        {
                            **common,
                            "query_language": language,
                            "n_queries": len(subset),
                            "metric": metric,
                            "contrast": f"absent-minus-{reference}",
                            **estimate(delta),
                        }
                    )
    levels, contrasts, strata = pd.DataFrame(levels), pd.DataFrame(contrasts), pd.DataFrame(strata)
    table(levels, OUT, "table05_absent_pivot_levels")
    table(contrasts, OUT, "table06_absent_pivot_contrasts")
    table(strata, OUT, "table07_absent_pivot_language_strata")
    disagreement = contrasts.pivot(
        index=["corpus", "model", "pivot_language", "contrast", "n_queries"],
        columns="metric",
        values="estimate",
    ).reset_index()
    disagreement["ndcg_vs_hit_opposite_signs"] = disagreement["ndcg10"] * disagreement["hit10"] < 0
    disagreement["recall_vs_hit_opposite_signs"] = (
        disagreement["recall10"] * disagreement["hit10"] < 0
    )
    table(disagreement, OUT, "table08_absent_pivot_metric_disagreement")
    return levels, contrasts, strata, disagreement


def forest(ax, rows, *, groups, colors, labels, metric, xlim=None, show_y=True):
    y = np.arange(len(DISPLAY_MODELS))
    offsets = np.linspace(-0.16, 0.16, len(groups))
    for offset, group, color, label in zip(offsets, groups, colors, labels, strict=True):
        data = (
            rows[(rows["metric"] == metric) & (rows["contrast"] == group)]
            .set_index("model")
            .reindex(DISPLAY_MODELS)
        )
        x = data["estimate"].to_numpy() * 100
        low = data["ci_low"].to_numpy() * 100
        high = data["ci_high"].to_numpy() * 100
        ax.errorbar(
            x,
            y + offset,
            xerr=np.vstack([x - low, high - x]),
            fmt="o",
            markersize=3.5,
            color=color,
            capsize=2,
            elinewidth=0.8,
            label=label,
        )
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.axhline(len(MODELS) - 0.5, color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(DISPLAY_LABELS)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(y) - 0.5, -0.6)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)


def figures(a_contrasts, surfaces, pivot_contrasts, pivot_strata):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.6), sharey=True)
    subset = a_contrasts[
        (a_contrasts["metric"] == "ndcg10")
        & a_contrasts["contrast"].isin(["translate-minus-original", "hyde-minus-translate"])
    ]
    lo = np.floor(subset["ci_low"].min() * 100 / 5) * 5
    hi = np.ceil(subset["ci_high"].max() * 100 / 5) * 5
    for ax, corpus in zip(axes, CORPORA, strict=True):
        rows = a_contrasts[a_contrasts["corpus"] == corpus]
        forest(
            ax,
            rows,
            groups=["translate-minus-original", "hyde-minus-translate"],
            colors=["#0072B2", "#D55E00"],
            labels=["Target-language translation − original", "Target-language HyDE − translation"],
            metric="ndcg10",
            xlim=(lo - 1, hi + 1),
            show_y=ax is axes[0],
        )
        n = int(rows["n_queries"].iloc[0])
        ax.set_title(f"{CORPUS_LABEL[corpus]} · n = {n} questions")
        ax.set_xlabel("Paired change in nDCG@10 (points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.54, 0.02))
    fig.suptitle(
        "Target-language-matched controls in cross-language retrieval", fontsize=13, y=0.98
    )
    fig.text(
        0.54,
        0.085,
        "Cross-language single-target search; average available targets within each question before analysis.",
        ha="center",
        fontsize=8,
    )
    fig.text(0.54, -0.005, INTERVAL_NOTE, ha="center", fontsize=7)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.88, bottom=0.19, wspace=0.14)
    savefig(fig, OUT, "fig01_target_matched_cross_language")

    targets = [*LANGUAGES, "mixed"]
    ndcg = surfaces[surfaces["metric"].eq("ndcg10")]
    limit = max(5, np.ceil(np.abs(ndcg["estimate"]).max() * 100 / 5) * 5)
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad("#e7e7e7")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, corpus in zip(axes, CORPORA, strict=True):
        sub = ndcg[ndcg["corpus"] == corpus]
        matrix = (
            sub.pivot(index="output_language", columns="target_language", values="estimate")
            .reindex(index=LANGUAGES, columns=targets)
            .to_numpy()
            * 100
        )
        counts = (
            sub.pivot(index="output_language", columns="target_language", values="n_queries")
            .reindex(index=LANGUAGES, columns=targets)
            .to_numpy()
        )
        im = ax.imshow(
            np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto"
        )
        for i in range(len(LANGUAGES)):
            for j in range(len(targets)):
                value = matrix[i, j]
                label = f"{value:+.1f}\nn={int(counts[i, j])}" if np.isfinite(value) else "N/A"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if np.isfinite(value) and abs(value) > limit * 0.65 else "black",
                )
        ax.set_xticks(
            range(len(targets)), [x.upper() if x != "mixed" else "Mixed" for x in targets]
        )
        ax.set_yticks(range(len(LANGUAGES)), [x.upper() for x in LANGUAGES])
        ax.set_title(CORPUS_LABEL[corpus])
        ax.set_xlabel("Searched document language")
        ax.set_ylabel("Shared generation language" if ax is axes[0] else "")
    color_axis = fig.add_axes([0.925, 0.25, 0.012, 0.50])
    fig.colorbar(im, cax=color_axis, label="HyDE − translation: nDCG@10 points")
    fig.suptitle(
        "HyDE gains depend on the output language and searched collection", fontsize=13, y=0.98
    )
    fig.text(
        0.48,
        0.03,
        "Same output language for both methods; query-weighted mean over eight fixed encoders. N/A means no eligible cohort.\nN is distinct questions per cell; differing cohorts and mixed/single-language judgments limit cross-cell interpretation.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.065, right=0.89, top=0.86, bottom=0.20, wspace=0.22)
    savefig(fig, OUT, "fig02_same_language_hyde_translation_surfaces")

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 10.1))
    for i, corpus in enumerate(["un", "eurlex"]):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            rows = pivot_contrasts[pivot_contrasts["corpus"] == corpus]
            forest(
                ax,
                rows,
                groups=["absent-minus-english_translation", "absent-minus-original"],
                colors=["#0072B2", "#D55E00"],
                labels=[
                    "Absent-language pivot − English translation",
                    "Absent-language pivot − original",
                ],
                metric=metric,
                show_y=j == 0,
            )
            ax.set_title(
                f"{CORPUS_LABEL[corpus]} · {PIVOT_NAME[corpus]} pivot\n{METRIC_LABEL[metric]}"
            )
            ax.set_xlabel("Paired change (points)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.52, 0.025))
    fig.suptitle(
        "A pivot can improve version-based metrics while reducing source success",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.52,
        0.005,
        "Mixed search. The pivot language is absent from both the searched corpus and selected original queries. "
        + INTERVAL_NOTE
        + ".",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.135, right=0.99, top=0.92, bottom=0.10, hspace=0.28, wspace=0.23)
    savefig(fig, OUT, "fig03_absent_pivots_metric_tradeoffs")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.9))
    metric_colors = {"ndcg10": "#7B3294", "hit10": "#0072B2", "recall10": "#D55E00"}
    for i, corpus in enumerate(["un", "eurlex"]):
        for j, reference in enumerate(["english_translation", "original"]):
            ax = axes[i, j]
            sub = pivot_strata[
                (pivot_strata["corpus"] == corpus)
                & (pivot_strata["model"] == ALL)
                & (pivot_strata["contrast"] == f"absent-minus-{reference}")
            ]
            languages = [x for x in LANGUAGES if x in set(sub["query_language"])]
            for metric, offset in zip(METRICS, [-0.18, 0, 0.18], strict=True):
                cells = sub[sub["metric"] == metric].set_index("query_language").reindex(languages)
                means = cells["estimate"].to_numpy() * 100
                errors = np.vstack(
                    [
                        means - cells["ci_low"].to_numpy() * 100,
                        cells["ci_high"].to_numpy() * 100 - means,
                    ]
                )
                ax.errorbar(
                    np.arange(len(languages)) + offset,
                    means,
                    yerr=errors,
                    color=metric_colors[metric],
                    fmt="o",
                    markersize=4,
                    capsize=2,
                    elinewidth=0.8,
                    label=METRIC_LABEL[metric],
                )
            counts = sub[sub["metric"] == "ndcg10"].set_index("query_language")["n_queries"]
            ax.set_xticks(range(len(languages)), [f"{x.upper()}\nn={counts[x]}" for x in languages])
            ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_title(
                f"{CORPUS_LABEL[corpus]}: {PIVOT_NAME[corpus]} − {'English translation' if reference == 'english_translation' else 'original'}"
            )
            ax.set_ylabel("Paired change (points)")
            ax.set_xlabel("Original question language")
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.51, 0.025))
    fig.suptitle(
        "Absent-language pivot effects across original question languages", fontsize=13, y=0.99
    )
    fig.text(
        0.51,
        0.005,
        "Eight fixed encoders averaged within each question before resampling. "
        + INTERVAL_NOTE
        + ".",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.18, hspace=0.42, wspace=0.22)
    savefig(fig, OUT, "fig04_absent_pivots_original_language_strata")


def draft_review(a_levels, a_contrasts, surfaces, pivot_contrasts, pivot_strata, disagreement):
    lines = [
        "ROUND 2 — LANGUAGE-MATCHED CONTROLS AND ABSENT-LANGUAGE PIVOTS",
        "",
        "VISUAL REVIEW STATUS: Pending root inspection of the four rendered figures.",
        "This draft records numerical findings only and does not certify visual review.",
        "",
        "Question and estimands",
        "A. In cross-language single-target search, compare original queries, translations",
        "into the searched language, and HyDE passages in that same language. Average",
        "eligible target settings within a question first; then average questions.",
        "B. Compare HyDE with translation while holding output language fixed, for each",
        "searched target or mixed collection. Fixed encoders are averaged within each",
        "question. Cell cohorts differ; unavailable cells remain missing.",
        "C. In mixed search, compare UN German / EUR-Lex Chinese translation against",
        "English translation and original queries, per encoder and original language.",
        "All intervals resample source questions, not model-setting observations.",
        "",
        "Numerical findings for root visual review",
    ]
    for corpus in CORPORA:
        sub = a_levels[
            (a_levels["corpus"] == corpus)
            & (a_levels["model"] == ALL)
            & (a_levels["metric"] == "ndcg10")
        ].set_index("method")
        n = int(sub["n_queries"].iloc[0])
        lines.append(
            f"- {CORPUS_LABEL[corpus]}, target-matched nDCG (n={n}): original {sub.loc['original', 'estimate']:.4f}, translation {sub.loc['translate', 'estimate']:.4f}, HyDE {sub.loc['hyde', 'estimate']:.4f}."
        )
        c = a_contrasts[
            (a_contrasts["corpus"] == corpus)
            & (a_contrasts["model"] == ALL)
            & (a_contrasts["metric"] == "ndcg10")
        ]
        for row in c.itertuples():
            lines.append(
                f"  {row.contrast}: {100 * row.estimate:+.2f} points, paired 95% CI [{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}]."
            )
    lines += ["", "Absent-language pivots (point-estimate counts across the eight fixed models):"]
    for corpus in ["un", "eurlex"]:
        for reference in ["english_translation", "original"]:
            sub = disagreement[
                (disagreement["corpus"] == corpus)
                & (disagreement["model"] != ALL)
                & (disagreement["contrast"] == f"absent-minus-{reference}")
            ]
            lines.append(
                f"- {CORPUS_LABEL[corpus]} {PIVOT_NAME[corpus]} vs {reference}: nDCG positive for {int((sub.ndcg10 > 0).sum())}/8, Hit@10 negative for {int((sub.hit10 < 0).sum())}/8; opposite nDCG/Hit signs for {int(sub.ndcg_vs_hit_opposite_signs.sum())}/8."
            )
            pooled = pivot_contrasts[
                (pivot_contrasts["corpus"] == corpus)
                & (pivot_contrasts["model"] == ALL)
                & (pivot_contrasts["contrast"] == f"absent-minus-{reference}")
            ]
            for row in pooled.itertuples():
                lines.append(
                    f"  Fixed-model mean {row.metric}: {100 * row.estimate:+.2f} points [{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}]."
                )
    languages = pivot_strata[(pivot_strata["model"] == ALL) & (pivot_strata["metric"] == "hit10")]
    lines += ["", "Original-language stratification (fixed-model mean Hit@10 deltas):"]
    for row in languages.itertuples():
        lines.append(
            f"- {CORPUS_LABEL[row.corpus]} / {row.query_language} / {row.contrast}: {100 * row.estimate:+.2f} points [{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}], n={row.n_queries}."
        )
    lines += [
        "",
        "What these data support",
        "Target-matched translation is a materially different comparator from R1's",
        "uniform average over output languages. The latter is not a deployment policy.",
        "HyDE's incremental value must be assessed against translation in the same",
        "language and against the original query. Metric choices can reverse which",
        "absent-language pivot looks preferable. Point-estimate counts are not a claim",
        "of eight statistically independent model replications.",
        "",
        "Limitations and checks carried forward",
        "- All effects are exploratory and conditional on the selected synthetic questions.",
        "- Encoders and corpora are fixed; CIs describe question variation only.",
        "- Numerous contrasts are shown without familywise or FDR correction.",
        "- Settings and original languages have unequal cohort sizes. Single-target",
        "  and mixed nDCG also have different relevance-set sizes.",
        "- R1's mixed-search averages and R2's target-matched cross-language estimates",
        "  change collection regime, output policy, and aggregation together. Their",
        "  absolute levels cannot isolate a causal benefit from language matching.",
        "- SourceRecall@10 equals Hit@10 for these one-source questions; it is not an",
        "  independent confirmation. Recall@10 counts document-language versions.",
        "- The known 26 identity near-tie discrepancies remain uncorrected.",
        "- HyDE uses each model's document adapter whereas translation uses its query",
        "  adapter. A passage-generation mechanism is not isolated by this comparison.",
        "- No unseen evaluation data or new retrieval/generation runs were introduced.",
        "",
        "Questions for Round 3, after root inspects these figures",
        "1. For the UN German pivot, do higher version-based scores coincide with more",
        "   language versions retrieved per successful source, but more missed sources?",
        "2. Does the metric disagreement survive paired bootstrap at the individual",
        "   model level and original-language stratification? Which models drive it?",
        "3. Are mixed rankings crowded by multiple versions of a source, and does this",
        "   relate to the gap between Hit@10 and Recall/nDCG? Inspect ranking-level data.",
        "4. Why is the target-matched HyDE increment positive in EUR-Lex but negative",
        "   in chemistry/UN: query wins/losses, source length, or encoder adapters?",
        "   These require controlled descriptive checks, not causal attribution.",
        "5. What would a deployable language rule achieve without query-wise hindsight",
        "   or selecting the best policy on the same questions used to report it?",
    ]
    (OUT / "review_draft.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    source = ROOT / "data/per_query.parquet"
    columns = [
        "corpus",
        "model",
        "method",
        "output_language",
        "query_id",
        "query_language",
        "setting_id",
        "target_language",
        "regime",
        *METRICS,
    ]
    frame = pd.read_parquet(source, columns=columns)
    assert frame["query_id"].nunique() == 638 and len(frame) == 247192
    contract = {
        "round": 2,
        "depends_on_review": "round_01/review.txt",
        "questions": [
            "Target-matched cross-language controls",
            "Same-output-language HyDE minus translation",
            "Absent-corpus-language pivots and metric disagreement",
        ],
        "metrics": METRICS,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 42,
        "resampling_unit": "one source question; targets and fixed encoders averaged before resampling where pooled",
        "confidence_intervals": "paired percentile 95%; exploratory and not multiplicity adjusted",
        "target_availability": "never impute absent or empty target cohorts as zero performance",
        "source_data_sha256": sha(source),
        "prior_review_sha256": sha(ROOT / "round_01/review.txt"),
    }
    write_json(OUT / "analysis_contract.json", contract)
    a_levels, a_contrasts, _ = target_matched(frame)
    surfaces, _ = language_surfaces(frame)
    _, pivot_contrasts, pivot_strata, disagreement = absent_pivots(frame)
    figures(a_contrasts, surfaces, pivot_contrasts, pivot_strata)
    draft_review(a_levels, a_contrasts, surfaces, pivot_contrasts, pivot_strata, disagreement)
    outputs = {
        p.name: sha(p)
        for p in sorted(OUT.iterdir())
        if p.is_file() and p.name != "run_manifest.json"
    }
    write_json(
        OUT / "run_manifest.json",
        {
            "round": 2,
            "generated_at": datetime.now(UTC).isoformat(),
            "script_sha256": sha(__file__),
            "visual_review_status": "pending_root",
            "figures": 4,
            "figure_formats": ["png", "pdf", "svg"],
            "csv_latex_table_pairs": 8,
            "paired_target_matched_questions": {
                c: int(a_levels.loc[a_levels.corpus == c, "n_queries"].iloc[0]) for c in CORPORA
            },
            "cohort_matching": "strict paired key equality required before differences",
            "output_sha256": outputs,
        },
    )
    print(
        "Round 2 complete: four figure triplets, eight CSV/LaTeX table pairs, three query-level derivative tables; visual review pending root."
    )


if __name__ == "__main__":
    main()
