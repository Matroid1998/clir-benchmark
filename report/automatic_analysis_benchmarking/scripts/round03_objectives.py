#!/usr/bin/env python3
"""Offline Round 3: decompose version nDCG and examine source objectives.

Streams projected saved rankings, never reads a text column, and never executes
retrieval, embedding, or generation. All uncertainty resamples source questions;
the fixed-model mean averages the eight encoders before resampling questions.
"""

from __future__ import annotations

import hashlib
import json
import resource
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import (
    CORPORA,
    CORPUS_LABEL,
    LANGUAGES,
    MODEL_LABEL,
    MODELS,
    ROOT,
    savefig,
    table,
    write_json,
)
from matplotlib.patches import Patch, Rectangle

OUT = ROOT / "round_03"
OUT.mkdir(exist_ok=True)
ALL = "All fixed models"
DISPLAY_MODELS = [*MODELS, ALL]
DISPLAY_LABELS = [*(MODEL_LABEL[m] for m in MODELS), "Fixed-model mean"]
VARIANTS = ["original", *[f"translate_{v}" for v in LANGUAGES], *[f"hyde_{v}" for v in LANGUAGES]]
VARIANT_LABELS = ["Original", *[f"T-{v}" for v in LANGUAGES], *[f"H-{v}" for v in LANGUAGES]]
KEY = ["corpus", "model", "variant", "setting_id", "query_id"]
COMPONENTS = ["first_contribution", "additional_contribution", "ndcg10"]
DECOMP_METRICS = [*COMPONENTS, "hit10", "positive_versions10", "mrr10"]
COLLAPSED_METRICS = [
    "ndcg10",
    "hit10",
    "mrr10",
    "collapsed_hit10",
    "collapsed_mrr10",
    "collapsed_ndcg10",
]
CONTRASTS = [("translate_de", "translate_en"), ("translate_de", "original")]
FIRST_COLOR, EXTRA_COLOR = "#0072B2", "#E69F00"
NOTE = "95% paired source-question bootstrap; fixed eight encoders, exploratory intervals"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_matrix(frame, metrics, draws=5000):
    """Joint bootstrap: one matrix row per independent source/question."""
    values = frame[metrics].to_numpy(dtype=float)
    assert len(values) and np.isfinite(values).all()
    rng = np.random.default_rng(42)
    means = np.empty((draws, len(metrics)))
    for start in range(0, draws, 200):
        end = min(draws, start + 200)
        indices = rng.integers(len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], axis=0)
    return {
        metric: {
            "estimate": float(values[:, i].mean()),
            "ci_low": float(low[i]),
            "ci_high": float(high[i]),
        }
        for i, metric in enumerate(metrics)
    }


def add_fixed_mean(frame, keys, metrics):
    assert frame.groupby(keys, observed=True).model.nunique().eq(8).all()
    pooled = frame.groupby(keys, observed=True)[metrics].mean().reset_index()
    pooled["model"] = ALL
    return pd.concat([frame, pooled], ignore_index=True)


def extract_features():
    columns = [
        *KEY,
        "method",
        "output_language",
        "query_language",
        "canonical_source_id",
        "relevant_document_count",
        "required_source_count",
        "ndcg10",
        "mrr10",
        "hit10",
        "recall10",
        "source_recall10",
    ]
    recorded = pq.read_table(
        ROOT / "data/per_query.parquet",
        columns=columns,
        filters=[("target_language", "=", "mixed")],
    ).to_pandas()
    assert len(recorded) == 638 * 8 * 11
    assert not recorded.duplicated(KEY).any()
    assert recorded.required_source_count.eq(1).all()
    assert (recorded.hit10 == recorded.source_recall10).all()
    assert recorded.groupby(["corpus", "model", "variant"]).query_id.nunique().to_dict() == {
        (c, m, v): n
        for c, n in [("chemistry", 197), ("eurlex", 194), ("un", 247)]
        for m in MODELS
        for v in VARIANTS
    }
    assert (
        recorded[["corpus", "query_id", "canonical_source_id"]]
        .drop_duplicates()
        .groupby(["corpus", "canonical_source_id"])
        .size()
        .eq(1)
        .all()
    )
    lookup = recorded.set_index(KEY)
    ranking_path = ROOT / "data/mixed_rankings.parquet"
    ranking_file = pq.ParquetFile(ranking_path)
    projection = [*KEY, "rank", "relevance", "source_id", "document_id", "document_language"]
    records, input_rows, legal_rows_checked = [], 0, 0
    discount = 1 / np.log2(np.arange(1, 101) + 1)
    for row_group in range(ranking_file.num_row_groups):
        block = ranking_file.read_row_group(row_group, columns=projection).to_pandas()
        input_rows += len(block)
        for key, rows in block.groupby(KEY, sort=False, observed=True):
            rows = rows.sort_values("rank", kind="stable")
            assert len(rows) == 100 and np.array_equal(rows["rank"], np.arange(1, 101))
            assert rows.document_id.nunique() == 100
            reference = lookup.loc[key]
            relevant = rows.relevance.to_numpy(dtype=bool)
            positive_positions = np.flatnonzero(relevant)
            first_rank = int(positive_positions[0] + 1) if len(positive_positions) else 101
            n_positive = int(reference.relevant_document_count)
            idcg = float(discount[: min(10, n_positive)].sum())
            first = float(discount[first_rank - 1] / idcg) if first_rank <= 10 else 0.0
            ndcg = float(discount[:10][relevant[:10]].sum() / idcg)
            record = dict(zip(KEY, key, strict=True)) | {
                "first_positive_rank": first_rank,
                "first_positive_observed_in_top100": first_rank <= 100,
                "first_positive_language": str(rows.iloc[first_rank - 1].document_language)
                if first_rank <= 100
                else "",
                "positive_versions10": int(relevant[:10].sum()),
                "positive_versions100": int(relevant.sum()),
                "idcg10": idcg,
                "first_contribution": first,
                "additional_contribution": ndcg - first,
                "reconstructed_ndcg10": ndcg,
                "reconstructed_mrr10": 1.0 / first_rank if first_rank <= 10 else 0.0,
                "reconstructed_hit10": float(first_rank <= 10),
                "reconstructed_recall10": float(relevant[:10].sum() / n_positive),
                "collapsed_distinct_sources_available": None,
                "collapsed_top10_censored": None,
                "collapsed_first_positive_rank": None,
                "collapsed_hit10": None,
                "collapsed_mrr10": None,
                "collapsed_ndcg10": None,
            }
            if key[0] in ["eurlex", "un"]:
                # For legal corpora only, every saved version of the known source
                # must have its existing positive qrel; no relabeling is allowed.
                assert (
                    relevant == rows.source_id.eq(reference.canonical_source_id).to_numpy()
                ).all()
                assert rows.groupby("source_id").relevance.nunique().eq(1).all()
                legal_rows_checked += len(rows)
                collapsed = rows.drop_duplicates("source_id", keep="first")
                n_sources = len(collapsed)
                positives = np.flatnonzero(collapsed.relevance.to_numpy(dtype=bool))
                assert len(positives) <= 1
                first_source = int(positives[0] + 1) if len(positives) else 101
                record.update(
                    {
                        "collapsed_distinct_sources_available": n_sources,
                        "collapsed_top10_censored": n_sources < 10,
                        "collapsed_first_positive_rank": first_source,
                        "collapsed_hit10": float(first_source <= 10) if n_sources >= 10 else np.nan,
                        "collapsed_mrr10": (1.0 / first_source if first_source <= 10 else 0.0)
                        if n_sources >= 10
                        else np.nan,
                        "collapsed_ndcg10": float(discount[first_source - 1])
                        if first_source <= 10
                        else (0.0 if n_sources >= 10 else np.nan),
                    }
                )
            records.append(record)
        if (row_group + 1) % 66 == 0:
            print(
                f"Streamed {row_group + 1}/{ranking_file.num_row_groups} ranking row groups",
                flush=True,
            )
    features = pd.DataFrame(records)
    assert not features.duplicated(KEY).any()
    merged = recorded.merge(features, on=KEY, validate="one_to_one")
    assert len(merged) == len(recorded)
    errors = {}
    for metric in ["ndcg10", "mrr10", "hit10", "recall10"]:
        errors[metric] = float(np.abs(merged[metric] - merged[f"reconstructed_{metric}"]).max())
        assert errors[metric] < 1e-12
    assert np.allclose(
        merged.first_contribution + merged.additional_contribution,
        merged.ndcg10,
        atol=1e-12,
        rtol=0,
    )
    assert merged.additional_contribution.ge(-1e-15).all()
    legal = merged[merged.corpus.isin(["eurlex", "un"])]
    assert legal.collapsed_top10_censored.eq(False).all()
    assert (legal.collapsed_hit10 >= legal.hit10).all()
    assert (legal.collapsed_mrr10 >= legal.mrr10 - 1e-12).all()
    merged.to_parquet(OUT / "ranking_features.parquet", index=False, compression="zstd")
    validation = {
        "input_ranking_rows": input_rows,
        "input_row_groups": ranking_file.num_row_groups,
        "query_configurations": len(merged),
        "legal_ranking_rows_qrel_consistency_checked": legal_rows_checked,
        "reconstruction_max_absolute_error": errors,
        "legal_collapsed_min_distinct_sources": int(
            legal.collapsed_distinct_sources_available.min()
        ),
        "legal_collapsed_max_distinct_sources": int(
            legal.collapsed_distinct_sources_available.max()
        ),
        "legal_collapsed_top10_censored_configurations": int(legal.collapsed_top10_censored.sum()),
        "unique_sources_per_selected_query": 1,
        "text_columns_read": [],
        "ranking_projection": projection,
        "ranking_input_sha256": sha(ranking_path),
        "source_collapse_scope": "EUR-Lex articles/annexes and UN documents only; chemistry excluded",
    }
    return merged, validation


def decomposition_analysis(features):
    un = features[features.corpus.eq("un")].copy()
    levels = []
    three = un[un.variant.isin(["original", "translate_en", "translate_de"])]
    three = add_fixed_mean(three, ["corpus", "query_id", "variant"], DECOMP_METRICS)
    for (model, variant), group in three.groupby(["model", "variant"]):
        stats = bootstrap_matrix(group, DECOMP_METRICS)
        for metric in DECOMP_METRICS:
            levels.append(
                {
                    "corpus": "un",
                    "model": model,
                    "variant": variant,
                    "n_queries": len(group),
                    "metric": metric,
                    **stats[metric],
                }
            )
    contrasts, counts, both_hit = [], [], []
    paired_records = []
    for treatment, reference in CONTRASTS:
        left = un[un.variant.eq(treatment)].set_index(["model", "query_id"])
        right = un[un.variant.eq(reference)].set_index(["model", "query_id"])
        assert left.index.equals(right.index) or set(left.index) == set(right.index)
        right = right.reindex(left.index)
        differences = (left[DECOMP_METRICS] - right[DECOMP_METRICS]).reset_index()
        differences["contrast"] = f"{treatment}-minus-{reference}"
        differences["both_hit"] = (left.hit10.eq(1) & right.hit10.eq(1)).to_numpy()
        differences["source_gained"] = (
            (left.hit10.eq(1) & right.hit10.eq(0)).to_numpy().astype(float)
        )
        differences["source_lost"] = (left.hit10.eq(0) & right.hit10.eq(1)).to_numpy().astype(float)
        paired_records.append(differences)
        pooled = add_fixed_mean(
            differences, ["query_id", "contrast"], [*DECOMP_METRICS, "source_gained", "source_lost"]
        )
        for model, group in pooled.groupby("model", sort=False):
            stats = bootstrap_matrix(group, DECOMP_METRICS)
            assert (
                abs(
                    stats["ndcg10"]["estimate"]
                    - stats["first_contribution"]["estimate"]
                    - stats["additional_contribution"]["estimate"]
                )
                < 1e-12
            )
            base = {
                "corpus": "un",
                "model": model,
                "treatment": treatment,
                "reference": reference,
                "contrast": f"{treatment}-minus-{reference}",
                "n_queries": len(group),
                "fixed_models": 8 if model == ALL else 1,
            }
            for metric in DECOMP_METRICS:
                contrasts.append({**base, "metric": metric, **stats[metric]})
            counts.append(
                {
                    **base,
                    "source_hits_gained": group.source_gained.sum(),
                    "source_hits_lost": group.source_lost.sum(),
                    "net_source_hits": group.hit10.sum(),
                    "net_positive_version_hits": group.positive_versions10.sum(),
                    "count_units": "mean count per fixed model"
                    if model == ALL
                    else "questions or version hits",
                    "source_hit_delta": stats["hit10"]["estimate"],
                    "source_hit_ci_low": stats["hit10"]["ci_low"],
                    "source_hit_ci_high": stats["hit10"]["ci_high"],
                    "version_count_delta_per_question": stats["positive_versions10"]["estimate"],
                    "version_count_ci_low": stats["positive_versions10"]["ci_low"],
                    "version_count_ci_high": stats["positive_versions10"]["ci_high"],
                }
            )
            assert np.isclose(
                group.source_gained.sum() - group.source_lost.sum(), group.hit10.sum()
            )
        # The both-hit subset is separately defined for each encoder and pair.
        # There is no pooled estimate across these differently selected subsets.
        for model, group in differences[differences.both_hit].groupby("model", sort=False):
            stats = bootstrap_matrix(group, DECOMP_METRICS)
            assert group.hit10.eq(0).all()
            for metric in DECOMP_METRICS:
                both_hit.append(
                    {
                        "corpus": "un",
                        "model": model,
                        "contrast": f"{treatment}-minus-{reference}",
                        "treatment": treatment,
                        "reference": reference,
                        "n_both_hit": len(group),
                        "n_all_queries": 247,
                        "metric": metric,
                        **stats[metric],
                    }
                )
    levels, contrasts, counts, both_hit = map(pd.DataFrame, [levels, contrasts, counts, both_hit])
    table(levels, OUT, "table01_un_decomposition_levels")
    table(contrasts, OUT, "table02_un_decomposition_contrasts")
    table(counts, OUT, "table03_un_source_and_version_counts")
    table(both_hit, OUT, "table04_un_both_hit_contrasts")
    pd.concat(paired_records).to_parquet(OUT / "un_paired_differences.parquet", index=False)
    return levels, contrasts, counts, both_hit


def ranking_disagreement(features):
    means = (
        features.groupby(["corpus", "model", "variant"], observed=True)[
            ["ndcg10", "hit10", "mrr10"]
        ]
        .mean()
        .reset_index()
    )
    groups, rows = [], []
    for (corpus, model), group in means.groupby(["corpus", "model"], sort=False):
        group = group.set_index("variant").reindex(VARIANTS).reset_index()
        assert group.ndcg10.notna().all() and len(group) == 11
        group["ndcg_rank"] = group.ndcg10.rank(ascending=False, method="average")
        group["hit_rank"] = group.hit10.rank(ascending=False, method="average")
        group["hit_rank_minus_ndcg_rank"] = group.hit_rank - group.ndcg_rank
        best_ndcg = group.ndcg10.eq(group.ndcg10.max())
        best_hit = group.hit10.eq(group.hit10.max())
        group["best_ndcg"] = best_ndcg
        group["best_hit"] = best_hit
        groups.append(group)
        ndcg_winners = sorted(group.loc[best_ndcg, "variant"])
        hit_winners = sorted(group.loc[best_hit, "variant"])
        # With Hit ties, use the range of the tied best-Hit systems' nDCG.
        pairwise_discordant = pairwise_tied = pairwise_concordant = 0
        for i in range(11):
            for j in range(i + 1, 11):
                dn = group.ndcg10.iloc[i] - group.ndcg10.iloc[j]
                dh = group.hit10.iloc[i] - group.hit10.iloc[j]
                if abs(dn) < 1e-15 or abs(dh) < 1e-15:
                    pairwise_tied += 1
                elif dn * dh < 0:
                    pairwise_discordant += 1
                else:
                    pairwise_concordant += 1
        assert pairwise_discordant + pairwise_concordant + pairwise_tied == 55
        rows.append(
            {
                "corpus": corpus,
                "model": model,
                "best_ndcg_variants": "|".join(ndcg_winners),
                "best_hit_variants": "|".join(hit_winners),
                "disjoint_best_sets": not bool(set(ndcg_winners) & set(hit_winners)),
                "same_best_sets": set(ndcg_winners) == set(hit_winners),
                "spearman_ndcg_hit_ranks": group.ndcg_rank.corr(group.hit_rank),
                "best_hit": group.hit10.max(),
                "hit_of_best_ndcg_min": group.loc[best_ndcg, "hit10"].min(),
                "hit_of_best_ndcg_max": group.loc[best_ndcg, "hit10"].max(),
                "hit_loss_selecting_best_ndcg": group.hit10.max()
                - group.loc[best_ndcg, "hit10"].max(),
                "best_ndcg": group.ndcg10.max(),
                "ndcg_of_best_hit_min": group.loc[best_hit, "ndcg10"].min(),
                "ndcg_of_best_hit_max": group.loc[best_hit, "ndcg10"].max(),
                "pairwise_discordant": pairwise_discordant,
                "pairwise_concordant": pairwise_concordant,
                "pairwise_tied": pairwise_tied,
                "interpretation": "descriptive all-sample selection; not a deployable selection policy",
            }
        )
    ranked, summary = pd.concat(groups, ignore_index=True), pd.DataFrame(rows)
    table(ranked, OUT, "table05_variant_objective_ranks")
    table(summary, OUT, "table06_objective_selection_disagreement")
    return ranked, summary


def collapse_analysis(features):
    legal = features[
        features.corpus.isin(["eurlex", "un"])
        & features.variant.isin(["original", "translate_en", "translate_de"])
    ].copy()
    assert legal.collapsed_top10_censored.eq(False).all()
    legal = add_fixed_mean(legal, ["corpus", "query_id", "variant"], COLLAPSED_METRICS)
    levels, contrasts = [], []
    for (corpus, model, variant), group in legal.groupby(
        ["corpus", "model", "variant"], sort=False
    ):
        stats = bootstrap_matrix(group, COLLAPSED_METRICS)
        for metric in COLLAPSED_METRICS:
            levels.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "variant": variant,
                    "n_queries": len(group),
                    "metric": metric,
                    **stats[metric],
                }
            )
    for treatment, reference in CONTRASTS:
        left = legal[legal.variant.eq(treatment)].set_index(["corpus", "model", "query_id"])
        right = (
            legal[legal.variant.eq(reference)]
            .set_index(["corpus", "model", "query_id"])
            .reindex(left.index)
        )
        differences = (left[COLLAPSED_METRICS] - right[COLLAPSED_METRICS]).reset_index()
        for (corpus, model), group in differences.groupby(["corpus", "model"], sort=False):
            stats = bootstrap_matrix(group, COLLAPSED_METRICS)
            for metric in COLLAPSED_METRICS:
                contrasts.append(
                    {
                        "corpus": corpus,
                        "model": model,
                        "contrast": f"{treatment}-minus-{reference}",
                        "n_queries": len(group),
                        "metric": metric,
                        **stats[metric],
                    }
                )
    levels, contrasts = pd.DataFrame(levels), pd.DataFrame(contrasts)
    table(levels, OUT, "table07_source_collapsed_levels")
    table(contrasts, OUT, "table08_source_collapsed_contrasts")
    return levels, contrasts


def component_bars(ax, data, models, *, labels=True):
    y = np.arange(len(models))
    wide = data.pivot(index="model", columns="metric", values="estimate").reindex(models)
    first = wide.first_contribution.to_numpy() * 100
    additional = wide.additional_contribution.to_numpy() * 100
    ax.barh(y, first, color=FIRST_COLOR, height=0.55)
    base = np.where(np.sign(first) == np.sign(additional), first, 0)
    ax.barh(y, additional, left=base, color=EXTRA_COLOR, height=0.55)
    total = data[data.metric.eq("ndcg10")].set_index("model").reindex(models)
    ax.errorbar(
        total.estimate * 100,
        y,
        xerr=np.vstack(
            [(total.estimate - total.ci_low) * 100, (total.ci_high - total.estimate) * 100]
        ),
        fmt="D",
        color="black",
        markersize=4,
        capsize=2,
        lw=1,
    )
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_yticks(y, [MODEL_LABEL.get(m, "Fixed-model mean") for m in models] if labels else [])
    ax.invert_yaxis()
    ax.set_xlabel("Change in document-version nDCG@10 (points)")
    ax.grid(axis="x", alpha=0.15)


def figure_decomposition(contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (_, reference), title in zip(
        axes,
        CONTRASTS,
        ["German translation − English translation", "German translation − original query"],
        strict=True,
    ):
        component_bars(
            ax, contrasts[contrasts.reference.eq(reference)], DISPLAY_MODELS, labels=ax is axes[0]
        )
        ax.set_title(title)
    handles = [
        Patch(color=FIRST_COLOR, label="First relevant version / IDCG"),
        Patch(color=EXTRA_COLOR, label="Additional relevant versions / IDCG"),
        plt.Line2D([], [], color="black", marker="D", lw=1, label="Total nDCG change with 95% CI"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.94), ncol=3, frameon=False
    )
    fig.suptitle(
        "UN: exact nDCG decomposition separates first-version and additional-version credit", y=1.005
    )
    fig.text(
        0.5,
        -0.015,
        f"247 paired questions; four positive language versions per question. Components sum exactly to the total.\n{NOTE}.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.89])
    savefig(fig, OUT, "fig01_un_exact_ndcg_decomposition")


def figure_counts(counts, both_hit):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.3))
    contrast = "translate_de-minus-translate_en"
    ct = counts[counts.contrast.eq(contrast)].set_index("model").reindex(MODELS)
    bh = both_hit[both_hit.contrast.eq(contrast)]
    y = np.arange(8)
    ax = axes[0, 0]
    ax.barh(y, -ct.source_hits_lost, color="#D55E00", label="English-only source hit")
    ax.barh(y, ct.source_hits_gained, color="#009E73", label="German-only source hit")
    for i, row in enumerate(ct.itertuples()):
        ax.text(
            -row.source_hits_lost - 0.6,
            i,
            str(int(row.source_hits_lost)),
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.text(
            row.source_hits_gained + 0.6,
            i,
            str(int(row.source_hits_gained)),
            ha="left",
            va="center",
            fontsize=8,
        )
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_yticks(y, [MODEL_LABEL[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(-float(ct.source_hits_lost.max()) - 10, float(ct.source_hits_gained.max()) + 10)
    ax.set_title("A. Which known sources are found in the version top 10?")
    ax.set_xlabel("Source questions lost (left) or gained (right), out of 247")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.31), frameon=False, ncol=2)
    ax = axes[0, 1]
    ax.barh(
        y,
        ct.net_positive_version_hits,
        color=np.where(ct.net_positive_version_hits >= 0, "#009E73", "#D55E00"),
    )
    for i, value in enumerate(ct.net_positive_version_hits):
        ax.text(
            value + np.sign(value or 1) * 2,
            i,
            f"{value:+.0f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
        )
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.margins(x=0.15)
    ax.set_title("B. Relevant language-version hits, same 247 questions")
    ax.set_xlabel("German − English: total positive versions in top 10")
    ax = axes[1, 0]
    versions = bh[bh.metric.eq("positive_versions10")].set_index("model").reindex(MODELS)
    ax.errorbar(
        versions.estimate,
        y,
        xerr=np.vstack([versions.estimate - versions.ci_low, versions.ci_high - versions.estimate]),
        fmt="o",
        color="#009E73",
        markersize=4,
        capsize=2,
    )
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_yticks(y, [f"{MODEL_LABEL[m]} (n={int(versions.loc[m, 'n_both_hit'])})" for m in MODELS])
    ax.invert_yaxis()
    ax.set_title("C. BOTH-HIT: paired questions where both find the source")
    ax.set_xlabel("German − English: positive versions per paired question")
    component_bars(axes[1, 1], bh, MODELS, labels=False)
    axes[1, 1].set_title("D. BOTH-HIT: exact nDCG decomposition on that same subset")
    fig.suptitle(
        "UN German vs English translation: source losses and additional version hits", y=1.005
    )
    fig.text(
        0.5,
        -0.005,
        "A–B use all 247 questions; C–D compare methods on each encoder’s SAME both-hit subset, selected descriptively.\n"
        "Blue: first-version contribution; gold: additional-version contribution; diamond: total. "
        + NOTE
        + ".",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.98], h_pad=4)
    savefig(fig, OUT, "fig02_un_sources_versions_and_both_hit")


def figure_disagreement(ranked):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    limit = 10
    for ax, corpus in zip(axes, CORPORA, strict=True):
        group = ranked[ranked.corpus.eq(corpus)]
        values = (
            group.pivot(index="model", columns="variant", values="hit_rank_minus_ndcg_rank")
            .reindex(index=MODELS, columns=VARIANTS)
            .to_numpy()
        )
        im = ax.imshow(values, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")
        for i, model in enumerate(MODELS):
            for j, variant in enumerate(VARIANTS):
                cell = group[group.model.eq(model) & group.variant.eq(variant)].iloc[0]
                if cell.best_ndcg:
                    ax.add_patch(
                        Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec="black", lw=1.5)
                    )
                if cell.best_hit:
                    ax.plot(j, i - 0.24, marker="*", color="black", markersize=6)
                ax.text(
                    j,
                    i + 0.09,
                    f"{values[i, j]:+.1f}" if values[i, j] % 1 else f"{values[i, j]:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(values[i, j]) >= 5 else "black",
                )
        ax.set_xticks(np.arange(11), VARIANT_LABELS, rotation=60, ha="right")
        ax.set_yticks(
            np.arange(8), [MODEL_LABEL[m] for m in MODELS] if corpus == CORPORA[0] else []
        )
        ax.set_title(CORPUS_LABEL[corpus])
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.24, top=0.79, wspace=0.08)
    cax = fig.add_axes([0.955, 0.24, 0.012, 0.55])
    fig.colorbar(im, cax=cax, label="Hit rank − nDCG rank (positive: higher under nDCG)")
    fig.suptitle("The ordering of 11 variants depends on the retrieval objective", y=0.98)
    fig.text(
        0.54,
        0.875,
        "Box: best document-version nDCG; star: best source Hit@10 (all tied winners marked). Average ranks handle metric ties.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.54,
        0.06,
        "Each cell ranks variants within one corpus and encoder using all selected questions in the mixed pool.\n"
        "Descriptive all-sample system selection; this is not an out-of-sample or deployable selection policy. T: translation; H: HyDE.",
        ha="center",
        fontsize=8,
    )
    savefig(fig, OUT, "fig03_variant_rank_objective_disagreement")


def figure_collapse(contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = ["collapsed_hit10", "collapsed_mrr10", "collapsed_ndcg10"]
    names = ["Source Hit@10", "Source MRR@10", "Source nDCG@10 (one positive)"]
    colors = ["#009E73", "#CC79A7", "#0072B2"]
    for ax, (_, reference), title in zip(
        axes,
        CONTRASTS,
        ["German translation − English translation", "German translation − original query"],
        strict=True,
    ):
        subset = contrasts[
            contrasts.corpus.eq("un") & contrasts.contrast.eq(f"translate_de-minus-{reference}")
        ]
        for offset, metric, name, color in zip([-0.2, 0, 0.2], metrics, names, colors, strict=True):
            data = subset[subset.metric.eq(metric)].set_index("model").reindex(DISPLAY_MODELS)
            ax.errorbar(
                data.estimate * 100,
                np.arange(9) + offset,
                xerr=np.vstack(
                    [(data.estimate - data.ci_low) * 100, (data.ci_high - data.estimate) * 100]
                ),
                fmt="o",
                color=color,
                capsize=2,
                markersize=3,
                lw=1,
                label=name,
            )
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_yticks(np.arange(9), DISPLAY_LABELS if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel("Change after collapsing to unique sources (points)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.57, 0.95),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "UN: source objectives after collapsing repeated language versions", y=1.005
    )
    fig.text(
        0.5,
        -0.02,
        "Collapse preserves the first occurrence and existing qrels of each canonical source in the saved top 100 versions.\n"
        "All legal rankings contain at least 10 distinct sources; no top-10 censoring. This changes the retrieval unit and objective.\n"
        + NOTE
        + ".",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.89])
    savefig(fig, OUT, "fig04_un_source_collapsed_objectives")


def write_review(contrasts, counts, both_hit, disagreement, collapse_contrasts, validation):
    lines = [
        "ROUND 3 DRAFT — OBJECTIVE DECOMPOSITION AND SOURCE COLLAPSE",
        "",
        "Status: computed and validated; awaiting root visual inspection of all four figures.",
        "",
        "Questions answered",
        "1. nDCG@10 = first relevant version discount / per-query IDCG + all additional relevant version discounts / that IDCG.",
        "   The identity holds per query and after every mean contrast; this is a decomposition of the recorded metric, not a proposed new metric.",
        "2. The counts below compare the same 247 UN questions; BOTH-HIT results use the SAME paired subset per encoder, not separately selected successes.",
        "3. Variant-selection disagreement is descriptive on the entire analysis sample, not a deployable or held-out system-selection procedure.",
        "4. Legal source collapse preserves existing qrels and source-unit granularity. It only processes saved top-100 version rankings.",
        "",
        "UN paired decomposition (percentage points; 95% source-question bootstrap intervals)",
    ]
    for reference in ["translate_en", "original"]:
        for model in ["e5-large-instruct", "embeddinggemma", ALL]:
            sub = contrasts[
                contrasts.reference.eq(reference) & contrasts.model.eq(model)
            ].set_index("metric")
            row = sub.loc["ndcg10"]
            lines.append(
                f"  German - {reference}, {model}: first {100 * sub.loc['first_contribution', 'estimate']:+.3f}; additional {100 * sub.loc['additional_contribution', 'estimate']:+.3f}; total {100 * row.estimate:+.3f} [{100 * row.ci_low:+.3f}, {100 * row.ci_high:+.3f}]."
            )
            count = counts[counts.reference.eq(reference) & counts.model.eq(model)].iloc[0]
            lines.append(
                f"    Source hits gained {count.source_hits_gained:g}, lost {count.source_hits_lost:g}, net {count.net_source_hits:+g}; positive-version hits net {count.net_positive_version_hits:+g}."
                + (" Counts are averages per fixed encoder." if model == ALL else "")
            )
    lines += [
        "",
        "BOTH-HIT subset, German - English; per encoder, no pooled estimate across different subsets",
    ]
    for model in MODELS:
        sub = both_hit[both_hit.reference.eq("translate_en") & both_hit.model.eq(model)].set_index(
            "metric"
        )
        version, total = sub.loc["positive_versions10"], sub.loc["ndcg10"]
        lines.append(
            f"  {model}, n={int(version.n_both_hit)}: version hits/question {version.estimate:+.4f} [{version.ci_low:+.4f}, {version.ci_high:+.4f}]; first {100 * sub.loc['first_contribution', 'estimate']:+.3f}, additional {100 * sub.loc['additional_contribution', 'estimate']:+.3f}, nDCG {100 * total.estimate:+.3f} [{100 * total.ci_low:+.3f}, {100 * total.ci_high:+.3f}] points."
        )
    lines += ["", "Descriptive objective ordering"]
    for corpus, group in disagreement.groupby("corpus", sort=False):
        lines.append(
            f"  {corpus}: disjoint best-system sets in {int(group.disjoint_best_sets.sum())}/8 encoders; {int(group.pairwise_discordant.sum())} discordant variant pairs out of {8 * 55} total pairs ({int(group.pairwise_tied.sum())} tied on at least one metric)."
        )
        for row in group[group.disjoint_best_sets].itertuples():
            lines.append(
                f"    {row.model}: nDCG winner {row.best_ndcg_variants}; Hit winner(s) {row.best_hit_variants}; Hit loss at nDCG winner {100 * row.hit_loss_selecting_best_ndcg:.3f} points."
            )
    lines += ["", "Source collapse"]
    lines.append(
        f"  Checked {validation['legal_ranking_rows_qrel_consistency_checked']:,} legal ranking rows: relevance equals membership in the known canonical source, and all retrieved versions of each source agree on relevance."
    )
    lines.append(
        f"  Distinct sources available per legal top-100 ranking: {validation['legal_collapsed_min_distinct_sources']}–{validation['legal_collapsed_max_distinct_sources']}; censored top-10 configurations: {validation['legal_collapsed_top10_censored_configurations']}."
    )
    for reference in ["translate_en", "original"]:
        for model in ["e5-large-instruct", "embeddinggemma", ALL]:
            sub = collapse_contrasts[
                collapse_contrasts.corpus.eq("un")
                & collapse_contrasts.model.eq(model)
                & collapse_contrasts.contrast.eq(f"translate_de-minus-{reference}")
            ].set_index("metric")
            for metric in ["collapsed_hit10", "collapsed_mrr10", "collapsed_ndcg10"]:
                row = sub.loc[metric]
                lines.append(
                    f"  German - {reference}, {model}, {metric}: {100 * row.estimate:+.3f} [{100 * row.ci_low:+.3f}, {100 * row.ci_high:+.3f}] points."
                )
    lines += [
        "",
        "Interpretation limits and measurement contract",
        "- One known gold source per selected query; source-only qrels are not exhaustive human relevance judgments. Unjudged sources may contain useful information.",
        "- Version nDCG/Recall remain valid for a goal of recovering multiple language versions. Source Hit/collapsed ranks evaluate a different goal; neither is universally superior.",
        "- All UN queries have four gold versions, making IDCG constant within this comparison. Across corpora, the feature extraction retains each query's actual relevance denominator.",
        "- First-version contribution combines finding a source within 10 with the rank of its first version. BOTH-HIT isolates paired successful cases descriptively but does not identify a causal mechanism; conditioning can select easier queries.",
        "- The both-hit subset differs by encoder and pair. No fixed-model estimate pools those different subsets. All full-cohort fixed-model intervals first average eight encoder differences per question and then resample the 247 questions.",
        "- Legal collapse uses canonical articles/annexes for EUR-Lex and whole documents for UN. Chemistry is excluded because direct accepted translation pairs do not justify transitive source-ID relevance closure.",
        "- Top-100 versions cannot recover the entire source ranking. At least 10 distinct sources in every saved legal ranking makes the first 10 collapsed sources observable; no new retrieval or qrels were created.",
        "- Selection of a best variant uses these same questions: it is descriptive/oracle analysis, not a deployable selection rule. All bootstrap intervals are exploratory with no multiple-testing adjustment.",
        "- The fixed model set is not a random sample of encoders. Sampling uncertainty refers to selected source questions conditional on these encoders and candidate-selection procedure.",
        "- Generated HyDE text uses document adapters for four models; comparisons across methods do not identify content-only effects. German-vs-English translation uses the same query adapter.",
        "- No claim of a German semantic bridge, new retrieval metric, first metric reversal, or causal legal-language mechanism is supported by this analysis.",
        "",
        "Questions for the next reviewed round",
        "- Which source/question strata drive the first-versus-additional credit pattern, and does it persist under available quality and identity-control sensitivity exclusions?",
        "- Does source collapse mainly promote previously rank-11+ gold sources or change ordering among sources already found?",
        "- How do original question language, first relevant hit language, and version concentration align with the observed tradeoff, without treating correlation as a causal explanation?",
    ]
    (OUT / "review_draft.txt").write_text("\n".join(lines) + "\n")


def main():
    features, validation = extract_features()
    _, contrasts, counts, both_hit = decomposition_analysis(features)
    ranked, disagreement = ranking_disagreement(features)
    _, collapse_contrasts = collapse_analysis(features)
    figure_decomposition(contrasts)
    figure_counts(counts, both_hit)
    figure_disagreement(ranked)
    figure_collapse(collapse_contrasts)
    validation["all_contrast_decompositions_sum_exactly"] = True
    validation["fixed_model_bootstrap_unit"] = "one question after averaging eight encoders"
    validation["both_hit_bootstrap_unit"] = (
        "one question within an encoder-specific paired both-hit subset"
    )
    validation["peak_process_memory_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    assert validation["peak_process_memory_mib"] < 7000
    write_json(OUT / "validation.json", validation)
    write_review(contrasts, counts, both_hit, disagreement, collapse_contrasts, validation)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script": str(__file__),
        "script_sha256": sha(ROOT / "scripts/round03_objectives.py"),
        "inputs": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                ROOT / "data/mixed_rankings.parquet",
                ROOT / "data/per_query.parquet",
                ROOT / "round_02/review.txt",
            ]
        },
        "seed": 42,
        "bootstrap_draws": 5000,
        "benchmark_execution": False,
        "paid_calls": 0,
        "model_loads": 0,
        "ranking_read_method": "one Parquet row group at a time, projected non-text columns",
        "outputs": sorted(
            p.name for p in OUT.iterdir() if p.is_file() and p.name != "run_manifest.json"
        ),
    }
    write_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"validation": validation, "outputs": len(manifest["outputs"])}), flush=True)


if __name__ == "__main__":
    main()
