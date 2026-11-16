#!/usr/bin/env python3
"""Offline Round 4: query heterogeneity, encoder setups, and first-hit languages.

Uses frozen Round 2/3 outputs. No model, API, retrieval or server execution.
Outputs only round_04 root files; round_04/cases belongs to a separate audit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    CORPORA,
    CORPUS_LABEL,
    MODEL_LABEL,
    MODELS,
    ROOT,
    paired_bootstrap,
    savefig,
    table,
    write_json,
)
from matplotlib.patches import Patch

OUT = ROOT / "round_04"
OUT.mkdir(exist_ok=True)
EPSILON = 1e-10
SEVERE = 0.25
METRICS = ["ndcg10", "hit10", "recall10"]
COUNTS = {"chemistry": 197, "eurlex": 194, "un": 247}
GROUPS = ["all_eight", "same_prefix_four", "changed_prefix_four"]
GROUP_LABELS = ["All eight encoders", "Same query/document prefix (4)", "Changed prefix/instruction (4)"]
GROUP_COLORS = ["#444444", "#0072B2", "#D55E00"]
UN_VARIANTS = ["original", "translate_en", "translate_de"]
UN_VARIANT_LABEL = {"original": "Original", "translate_en": "English translation", "translate_de": "German translation"}
FIRST_LANGUAGES = ["en", "es", "fr", "zh", "no_hit"]
FIRST_LANGUAGE_LABEL = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese", "no_hit": "No relevant result in top 10"}
FIRST_COLORS = {"en": "#0072B2", "es": "#009E73", "fr": "#E69F00", "zh": "#CC79A7", "no_hit": "#D5D5D5"}
NOTE = "95% paired source-question bootstrap intervals; fixed encoders, exploratory, no multiplicity correction"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    point, low, high = paired_bootstrap(values)
    return {"estimate": point, "ci_low": low, "ci_high": high}


def load_target_matched():
    frame = pd.read_parquet(ROOT / "round_02/target_matched_query_scores.parquet")
    frame = frame[frame.model.isin(MODELS)].copy()
    assert not frame.duplicated(["corpus", "model", "query_id"]).any()
    assert frame.groupby(["corpus", "model"]).size().to_dict() == {
        (c, m): n for c, n in COUNTS.items() for m in MODELS
    }
    assert frame.groupby(["corpus", "query_id"]).model.nunique().eq(8).all()
    for metric in METRICS:
        assert frame[[f"{metric}__hyde", f"{metric}__translate"]].notna().all().all()
        frame[f"delta_{metric}"] = frame[f"{metric}__hyde"] - frame[f"{metric}__translate"]
    frame.to_parquet(OUT / "query_diagnostics.parquet", index=False)
    return frame


def distribution_tables(frame):
    records = []
    for (corpus, model), group in frame.groupby(["corpus", "model"], observed=True):
        for metric in METRICS:
            values = group[f"delta_{metric}"].to_numpy()
            wins, losses = values > EPSILON, values < -EPSILON
            ties = np.abs(values) <= EPSILON
            severe_wins, severe_losses = values >= SEVERE - EPSILON, values <= -SEVERE + EPSILON
            assert (wins.astype(int) + ties + losses == 1).all()
            for name, indicator in [
                ("win", wins), ("tie", ties), ("loss", losses),
                ("severe_gain", severe_wins), ("severe_loss", severe_losses),
            ]:
                records.append({
                    "corpus": corpus, "model": model, "metric": metric,
                    "statistic": name, "n_queries": len(values),
                    "count": int(indicator.sum()), **estimate(indicator),
                })
            records.append({
                "corpus": corpus, "model": model, "metric": metric,
                "statistic": "mean_delta", "n_queries": len(values), "count": None,
                **estimate(values),
            })
    summary = pd.DataFrame(records)
    table(summary, OUT, "table01_query_gain_loss_rates")
    quantiles = []
    for (corpus, model), group in frame.groupby(["corpus", "model"], observed=True):
        values = group.delta_ndcg10.to_numpy()
        quantiles.append({
            "corpus": corpus, "model": model, "n_queries": len(values),
            "min": float(values.min()), "p10": float(np.quantile(values, .10)),
            "p25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)),
            "max": float(values.max()), "mean": float(values.mean()),
        })
    table(pd.DataFrame(quantiles), OUT, "table02_query_delta_quantiles")
    return summary


def role_groups(frame):
    audit = json.loads((ROOT / "audit/encoder_role_audit.json").read_text())
    changed = audit["role_diff_models"]
    same = [m for m in MODELS if m not in changed]
    assert len(changed) == len(same) == 4
    assert all(audit["models"][m]["query_prefix"] == audit["models"][m]["document_prefix"] for m in same)
    assert all(audit["models"][m]["query_prefix"] != audit["models"][m]["document_prefix"] for m in changed)
    sets = {"all_eight": MODELS, "same_prefix_four": same, "changed_prefix_four": changed}
    pieces = []
    columns = [f"delta_{metric}" for metric in METRICS]
    for name, models in sets.items():
        cell = frame[frame.model.isin(models)]
        assert cell.groupby(["corpus", "query_id"]).model.nunique().eq(len(models)).all()
        q = cell.groupby(["corpus", "query_id", "query_language"], observed=True)[columns].mean().reset_index()
        q["encoder_group"] = name
        q["fixed_encoders"] = len(models)
        pieces.append(q)
    query = pd.concat(pieces, ignore_index=True)
    query.to_parquet(OUT / "encoder_group_query_deltas.parquet", index=False)
    rows = []
    for (corpus, name), group in query.groupby(["corpus", "encoder_group"], observed=True):
        for metric in METRICS:
            rows.append({
                "corpus": corpus, "encoder_group": name,
                "encoders": ";".join(sets[name]), "fixed_encoders": len(sets[name]),
                "metric": metric, "n_queries": len(group),
                **estimate(group[f"delta_{metric}"]),
            })
    summary = pd.DataFrame(rows)
    table(summary, OUT, "table03_encoder_group_contrasts")
    group_gaps = []
    for corpus in CORPORA:
        sub = query[query.corpus.eq(corpus)]
        for metric in METRICS:
            wide = sub.pivot(index="query_id", columns="encoder_group", values=f"delta_{metric}")
            assert wide.notna().all().all()
            gap = wide.same_prefix_four - wide.changed_prefix_four
            group_gaps.append({
                "corpus": corpus, "metric": metric, "n_queries": len(gap),
                "contrast": "same-prefix-fixed-four-minus-changed-prefix-fixed-four",
                "interpretation": "descriptive difference between fixed model sets; not causal prefix effect",
                **estimate(gap),
            })
    table(pd.DataFrame(group_gaps), OUT, "table04_descriptive_encoder_set_gaps")
    return summary, sets


def first_hit_languages():
    features = pd.read_parquet(ROOT / "round_03/ranking_features.parquet")
    rows = features[features.corpus.eq("un") & features.variant.isin(UN_VARIANTS)].copy()
    keys = ["model", "query_id"]
    assert not rows.duplicated(keys + ["variant"]).any()
    assert rows.groupby(["model", "variant"]).size().eq(247).all()
    assert rows.groupby(keys).variant.nunique().eq(3).all()
    assert rows.groupby("query_id").model.nunique().eq(8).all()
    assert np.array_equal(rows.first_positive_rank.le(10).to_numpy(), rows.hit10.eq(1).to_numpy())
    rows["first_top10_language"] = np.where(rows.first_positive_rank.le(10), rows.first_positive_language, "no_hit")
    assert set(rows.first_top10_language) <= set(FIRST_LANGUAGES)
    columns = ["model", "variant", "query_id", "query_language", "first_positive_rank", "first_top10_language", "hit10"]
    rows[columns].to_parquet(OUT / "un_first_hit_languages.parquet", index=False)
    records, contrasts, transitions = [], [], []
    for model in MODELS:
        cell = rows[rows.model.eq(model)]
        wide = cell.pivot(index="query_id", columns="variant", values="first_top10_language")
        assert wide.notna().all().all()
        for variant in UN_VARIANTS:
            for language in FIRST_LANGUAGES:
                indicator = wide[variant].eq(language)
                records.append({
                    "model": model, "variant": variant, "first_top10_language": language,
                    "n_queries": len(wide), "count": int(indicator.sum()), **estimate(indicator),
                })
        for reference in ["translate_en", "original"]:
            for language in FIRST_LANGUAGES:
                delta = wide.translate_de.eq(language).astype(float) - wide[reference].eq(language).astype(float)
                contrasts.append({
                    "model": model, "contrast": f"translate_de-minus-{reference}",
                    "first_top10_language": language, "n_queries": len(wide),
                    "count_change": int(delta.sum()), **estimate(delta),
                })
            transition = pd.crosstab(wide[reference], wide.translate_de).reindex(index=FIRST_LANGUAGES, columns=FIRST_LANGUAGES, fill_value=0)
            assert transition.to_numpy().sum() == 247
            for before in FIRST_LANGUAGES:
                for after in FIRST_LANGUAGES:
                    transitions.append({
                        "model": model, "reference": reference,
                        "reference_first_language": before, "german_first_language": after,
                        "n_queries": 247, "count": int(transition.loc[before, after]),
                    })
    summary = pd.DataFrame(records)
    assert summary.groupby(["model", "variant"])["count"].sum().eq(247).all()
    table(summary, OUT, "table05_un_first_relevant_language")
    table(pd.DataFrame(contrasts), OUT, "table06_un_first_language_paired_changes")
    table(pd.DataFrame(transitions), OUT, "table07_un_first_language_transitions")
    return summary


def query_truncation():
    audit = pd.read_parquet(ROOT / "data/truncation.parquet")
    query = audit[
        audit.encoding_scope.isin(["queries", "hypothetical_documents"])
        & audit.text_language.eq("all")
    ].copy()
    assert not query.duplicated(["corpus", "model", "variant"]).any()
    assert len(query) == 264
    assert query.loc[query.method.isin(["original", "translate"]), "truncated"].eq(0).all()
    nonzero = query[query.truncated.gt(0)]
    assert len(nonzero) == 1
    only = nonzero.iloc[0]
    assert (only.corpus, only.model, only.variant, int(only.truncated)) == ("eurlex", "LaBSE", "hyde_zh", 3)
    assert not pd.read_parquet(ROOT / "data/settings.parquet").query("corpus == 'eurlex' and target_language == 'zh' and eligible_queries > 0").shape[0]
    table(query, OUT, "table08_query_embedding_truncation")
    return {"original_truncated": 0, "translation_truncated": 0, "hyde_truncated": 3,
            "hyde_encoding_rows": int(query.loc[query.method.eq("hyde"), "inputs"].sum()),
            "nonzero_only": {"corpus": "eurlex", "model": "LaBSE", "output_language": "zh", "truncated": 3, "inputs": 194},
            "target_matched_cross_language_truncated": 0,
            "note": "The only three HyDE truncations are EUR-Lex Chinese outputs, absent from target-matched single-target comparisons because no searched Chinese target is available. Corpus-document truncation rows are excluded."}


def plot_gain_loss(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.1), sharey=True)
    states = ["severe_loss", "ordinary_loss", "tie", "ordinary_gain", "severe_gain"]
    colors = ["#B2182B", "#EF8A62", "#E7E7E7", "#67A9CF", "#2166AC"]
    for ax, corpus in zip(axes, CORPORA, strict=True):
        cell = summary[summary.corpus.eq(corpus) & summary.metric.eq("ndcg10")]
        wide = cell.pivot(index="model", columns="statistic", values="estimate").reindex(MODELS)
        wide["ordinary_loss"] = wide.loss - wide.severe_loss
        wide["ordinary_gain"] = wide.win - wide.severe_gain
        assert np.allclose(wide[states].sum(axis=1), 1)
        left = np.zeros(8)
        for name, color in zip(states, colors, strict=True):
            widths = 100 * wide[name].to_numpy()
            ax.barh(np.arange(8), widths, left=left, color=color, height=.7, edgecolor="white", linewidth=.35)
            for y, (start, width) in enumerate(zip(left, widths, strict=True)):
                if width >= 6:
                    ax.text(start + width/2, y, f"{width:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if name in ["severe_loss", "severe_gain"] else "black")
            left += widths
        ax.set_title(f"{CORPUS_LABEL[corpus]} · {COUNTS[corpus]} questions")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of source questions (%)")
        ax.set_yticks(np.arange(8))
        if ax is axes[0]:
            ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis() if ax is axes[0] else None
        ax.grid(axis="x", alpha=.15)
    fig.legend([Patch(facecolor=c) for c in colors],
               ["Loss ≥25 points", "Smaller loss", "Tie", "Smaller gain", "Gain ≥25 points"],
               ncol=5, loc="lower center", bbox_to_anchor=(.54, .06))
    fig.suptitle("Query-level gains and losses under target-matched HyDE − translation", fontsize=13, y=.98)
    fig.text(.54, .01, "ΔnDCG@10 is averaged across eligible cross-language targets within each question first. Ties: |Δ| ≤ 10⁻¹⁰.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.16, right=.99, top=.86, bottom=.22, wspace=.12)
    savefig(fig, OUT, "fig01_query_gain_loss_distribution")


def plot_ecdf(frame):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8), sharex=True, sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#332288", "#882255", "#555555"]
    handles = []
    for ax, corpus in zip(axes, CORPORA, strict=True):
        for i, model in enumerate(MODELS):
            values = np.sort(frame.loc[frame.corpus.eq(corpus) & frame.model.eq(model), "delta_ndcg10"].to_numpy())
            x = np.r_[-1, values, 1]
            y = np.r_[0, np.arange(1, len(values)+1)/len(values), 1]
            line, = ax.step(x*100, y*100, where="post", color=colors[i], lw=1.5, linestyle="--" if i >= 4 else "-")
            if ax is axes[0]:
                handles.append(line)
        ax.axvline(0, color="black", lw=.8)
        ax.axvline(-25, color="gray", lw=.7, ls=":")
        ax.axvline(25, color="gray", lw=.7, ls=":")
        ax.set_title(f"{CORPUS_LABEL[corpus]} · n={COUNTS[corpus]}")
        ax.set_xlabel("HyDE − translation nDCG@10 (points)")
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Cumulative share of source questions (%)")
    fig.legend(handles, [MODEL_LABEL[m] for m in MODELS], ncol=4, loc="lower center", bbox_to_anchor=(.51, .045))
    fig.suptitle("The full distribution behind each average language-controlled contrast", fontsize=13, y=.98)
    fig.text(.51, .005, "Each curve contains one target-averaged value per source question. Vertical dotted lines mark ±25 points; encoders are not independent samples.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.29, wspace=.13)
    savefig(fig, OUT, "fig02_query_delta_ecdfs")


def plot_role_groups(summary, sets):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
    for ax, corpus in zip(axes, CORPORA, strict=True):
        cell = summary[summary.corpus.eq(corpus) & summary.metric.eq("ndcg10")].set_index("encoder_group")
        for i, name in enumerate(GROUPS):
            row = cell.loc[name]
            ax.errorbar(row.estimate*100, i, xerr=[[100*(row.estimate-row.ci_low)], [100*(row.ci_high-row.estimate)]],
                        fmt="o", color=GROUP_COLORS[i], capsize=3, markersize=6)
            ax.annotate(f"{row.estimate*100:+.2f}", (row.estimate*100, i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(0, color="gray", lw=.9)
        ax.set_title(f"{CORPUS_LABEL[corpus]} · n={COUNTS[corpus]}")
        ax.set_yticks(range(3))
        if ax is axes[0]:
            ax.set_yticklabels(GROUP_LABELS)
            ax.set_ylim(2.5, -.55)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("HyDE − translation nDCG@10 (points)")
        ax.set_xlim(-9, 12)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Mean contrasts retain the corpus pattern in both fixed encoder subsets", fontsize=13, y=.98)
    fig.text(.58, .10, "Same-prefix set: " + ", ".join(MODEL_LABEL[m] for m in sets["same_prefix_four"]), ha="center", fontsize=8)
    fig.text(.58, .055, "Encoders averaged within each question before resampling. Differences between model sets are descriptive, not causal prefix effects.", ha="center", fontsize=8)
    fig.text(.58, .012, NOTE, ha="center", fontsize=8)
    fig.subplots_adjust(left=.25, right=.99, top=.79, bottom=.29, wspace=.15)
    savefig(fig, OUT, "fig03_encoder_subset_contrasts")


def plot_first_languages(summary):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True)
    for ax, model in zip(axes.flat, MODELS, strict=True):
        wide = summary[summary.model.eq(model)].pivot(index="variant", columns="first_top10_language", values="count").reindex(UN_VARIANTS)
        left = np.zeros(3)
        for language in FIRST_LANGUAGES:
            count = wide[language].to_numpy()
            width = count / 247 * 100
            ax.barh(range(3), width, left=left, color=FIRST_COLORS[language], height=.68,
                    edgecolor="white", linewidth=.4, hatch="///" if language == "no_hit" else None)
            for y, (start, w, n) in enumerate(zip(left, width, count, strict=True)):
                if w >= 9:
                    ax.text(start+w/2, y, str(n), ha="center", va="center", fontsize=7,
                            color="white" if language in ["en", "es"] else "black")
            left += width
        assert np.allclose(left, 100)
        ax.set_title(MODEL_LABEL[model])
        ax.set_yticks(range(3), ["Original", "T-English", "T-German"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        if ax in axes[1]:
            ax.set_xlabel("Share of all 247 UN questions (%)")
        ax.grid(axis="x", alpha=.1)
    fig.legend([Patch(facecolor=FIRST_COLORS[language], hatch="///" if language=="no_hit" else None) for language in FIRST_LANGUAGES],
               [FIRST_LANGUAGE_LABEL[language] for language in FIRST_LANGUAGES], ncol=5, loc="lower center", bbox_to_anchor=(.52, .055))
    fig.suptitle("UN: language of the first relevant top-10 result, including no-hit queries", fontsize=13, y=.98)
    fig.text(.52, .017, "Every bar includes the same 247 questions. Segment labels are counts. German is absent from the searched corpus; distributions do not establish language bias.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.99, top=.87, bottom=.21, wspace=.45, hspace=.45)
    savefig(fig, OUT, "fig04_un_first_relevant_languages")


def review_text(distribution, roles, first, truncation):
    lines = [
        "ROUND 4 — QUERY HETEROGENEITY, ENCODER SETS AND FIRST-HIT LANGUAGES",
        "", "Visual review: PENDING ROOT. This is a numerical review draft, not a completed visual-review round.",
        "", "Design and denominators",
        "HyDE and translation are paired in output==searched target language, using genuinely cross-language settings.",
        "Eligible target settings are averaged within each source question first. All model-specific cells contain",
        "197 chemistry, 194 EUR-Lex or 247 UN questions. No repeated encoder/query rows are bootstrapped as independent.",
        "The ±0.25 threshold is a descriptive magnitude threshold chosen for this diagnostic, not an outcome-based exclusion.",
        "Ties use |delta|<=1e-10. Intervals: 5,000 paired source-question bootstrap draws, seed42, percentile95%;",
        "exploratory, no multiplicity correction. All recorded near-tie exceptions remain included.",
        "", "1. Query-level gains and losses",
    ]
    for corpus in CORPORA:
        lines.append(CORPUS_LABEL[corpus])
        cell = distribution[distribution.corpus.eq(corpus) & distribution.metric.eq("ndcg10")]
        for model in MODELS:
            group = cell[cell.model.eq(model)].set_index("statistic")
            lines.append(
                f"  {MODEL_LABEL[model]}: mean {group.loc['mean_delta','estimate']*100:+.2f} points; "
                f"gain/tie/loss {int(group.loc['win','count'])}/{int(group.loc['tie','count'])}/{int(group.loc['loss','count'])}; "
                f"severe gain/loss {int(group.loc['severe_gain','count'])}/{int(group.loc['severe_loss','count'])}."
            )
    lines += [
        "EUR-Lex positive means coexist with losses for individual questions in every encoder.",
        "Chemistry has many ties; a mean near zero can still conceal a minority of substantial gains and losses.",
        "UN encoders disagree: LaBSE/SapBERT have positive mean contrasts while the other six have negative means.",
        "A query gain rate is distinct from the mean effect and from the probability of finding any relevant source.",
        "", "2. Fixed encoder subsets",
    ]
    for corpus in CORPORA:
        for name, label in zip(GROUPS, GROUP_LABELS, strict=True):
            row = roles[roles.corpus.eq(corpus) & roles.encoder_group.eq(name) & roles.metric.eq("ndcg10")].iloc[0]
            lines.append(f"  {CORPUS_LABEL[corpus]}, {label}: {row.estimate*100:+.2f} [{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}] nDCG points.")
    lines += [
        "Both fixed subsets retain the corpus sign pattern in their means: chemistry negative, EUR-Lex positive, UN negative.",
        "The same-prefix subset rules out a changed text prefix as a necessary condition for the observed EUR-Lex gain.",
        "It does not isolate a causal effect of expansion, prefix choice, encoder architecture, pretraining or embedding role.",
        "Four encoders per subset are fixed systems, not a random sample of encoder families; subgroup gaps remain descriptive.",
        "", "3. UN first relevant language, unconditional on retrieval success",
    ]
    for model in MODELS:
        cell = first[first.model.eq(model)].pivot(index="variant", columns="first_top10_language", values="count")
        for variant in UN_VARIANTS:
            counts = ", ".join(f"{language}={int(cell.loc[variant,language])}" for language in FIRST_LANGUAGES)
            lines.append(f"  {MODEL_LABEL[model]}, {UN_VARIANT_LABEL[variant]}: {counts}; denominator247.")
    lines += [
        "Every stacked bar keeps all 247 questions; no-hit includes first relevant ranks above 10 and absence from the saved top 100.",
        "Only the earliest relevant version is classified. The figure does not characterize all retrieved languages,",
        "language availability, query-language matching, intrinsic language bias, or a causal semantic bridge.",
        "Paired first-language category changes and full transitions are supplied; category percentages are compositional.",
        "", "4. Query-side truncation is not a boundary in the primary target-matched contrast",
        f"Saved query-side summaries contain {truncation['hyde_truncated']} truncated HyDE encodings among {truncation['hyde_encoding_rows']:,} query/model/output-language encodings.",
        "All three are LaBSE EUR-Lex Chinese outputs. EUR-Lex has no searched Chinese target, so they do not enter",
        "the target-matched cross-language comparison. Original and translated query encodings have zero truncations.",
        "Thus saved query truncation cannot account for the primary target-matched H−T pattern. This is not evidence",
        "that passage length is irrelevant, nor an ablation of document chunking or truncation elsewhere in the corpus.",
        "No query-level length effect was estimated from aggregate truncation summaries, and repeated document",
        "truncation summaries shared by variants are excluded from query accounting.",
        "", "Limitations and provenance",
        "Synthetic maximum-score-selected questions, variable generator candidate pools, source-associated qrels and fixed",
        "encoders limit external generalization. Candidate provenance is audited separately in audit/candidate_pool_provenance.txt.",
        "This round tests descriptive robustness within saved runs. It cannot attribute mechanisms or establish a deployable",
        "per-query policy. Text examples are separately owned under round_04/cases/ and are not selected by this script.",
        "", "Questions for Round5: held-out selection and costs",
        "1. On a fixed source-level held-out split, does a training-only choice of variant beat the original/English controls?",
        "   Keep language versions and all encoder rows from one source together; never choose using test outcomes.",
        "2. Does selecting for source Hit versus version nDCG produce different held-out gains and harm rates?",
        "3. What is the generation cost of a deployable one-language policy versus exploring all five languages?",
        "   Charge failures/retries, reuse generation across encoders, and retain unknown local compute costs.",
        "4. Can gains remain meaningful when reported with added API costs and offline component timings, without calling",
        "   amortized cached timings measured online latency? Avoid per-query oracle routing and in-sample best-of-five claims.",
        "5. Given source-question losses within favorable corpus means, which minimum control and harm report should accompany",
        "   any deployment recommendation? No new generation or policy execution is authorized by this diagnostic.",
    ]
    (OUT / "review_draft.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    sources = [
        ROOT / "round_03/review.txt", ROOT / "round_02/target_matched_query_scores.parquet",
        ROOT / "round_03/ranking_features.parquet", ROOT / "audit/encoder_role_audit.json",
        ROOT / "data/truncation.parquet", ROOT / "data/settings.parquet",
        ROOT / "scripts/common.py",
    ]
    frame = load_target_matched()
    distribution = distribution_tables(frame)
    roles, sets = role_groups(frame)
    first = first_hit_languages()
    truncation = query_truncation()
    plot_gain_loss(distribution)
    plot_ecdf(frame)
    plot_role_groups(roles, sets)
    plot_first_languages(first)
    review_text(distribution, roles, first, truncation)
    contract = {
        "round": 4, "root_previous_review": "round_03/review.txt",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "estimand": "HyDE-minus-translation, same output==target language, cross-language; average targets within question before summaries",
        "independent_unit": "selected canonical source question; fixed models averaged within question for encoder groups",
        "tie_tolerance": EPSILON, "severe_absolute_delta_threshold": SEVERE,
        "bootstrap": {"draws": 5000, "seed": 42, "interval": "percentile95", "multiplicity_adjustment": None},
        "encoder_groups": sets,
        "encoder_group_comparison": "descriptive fixed-model subset robustness, not causal role/prefix effect",
        "first_language_denominator": "same247 UN questions for each model and variant, including no-hit",
        "first_language_definition": "language at minimum relevant rank if rank<=10, else no_hit",
        "query_truncation": truncation,
        "case_directory": "round_04/cases is separate and not written or hashed by this script",
        "visual_review_status": "pending_root",
    }
    write_json(OUT / "analysis_contract.json", contract)
    write_json(OUT / "validation.json", {
        "status": "passed", "target_matched_query_model_rows": len(frame),
        "target_matched_counts_per_model": COUNTS,
        "all_eight_encoder_cohorts_equal": True,
        "un_first_language_rows_per_variant_model": 247,
        "un_categories_exhaustive_and_reconstruct_hit10": True,
        "query_truncation_excludes_corpus_document_repetitions": True,
        "tables": 8, "figures": 4,
    })
    owned = [p for p in OUT.iterdir() if p.is_file() and p.name != "run_manifest.json" and p.name != "review.txt"]
    write_json(OUT / "run_manifest.json", {
        "round": 4, "generated_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha(ROOT / "scripts/round04_diagnostics.py"),
        "visual_review_status": "pending_root", "figures": 4,
        "figure_formats": ["png", "pdf", "svg"], "csv_latex_table_pairs": 8,
        "output_sha256": {p.name: sha(p) for p in sorted(owned)},
    })
    print(json.dumps({"round": 4, "figures": 4, "tables": 8, "status": "numerical_validation_passed_visual_review_pending_root"}))


if __name__ == "__main__":
    main()
