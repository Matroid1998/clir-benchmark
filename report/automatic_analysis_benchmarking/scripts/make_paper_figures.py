#!/usr/bin/env python3
"""Compact publication figures from accepted Round 2/3 tables; no new analysis."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import CORPORA, CORPUS_LABEL, ROOT, savefig, write_json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)
ALL = "All fixed models"
SOURCES = [
    ROOT / "round_02/table02_target_matched_contrasts.csv",
    ROOT / "round_03/table02_un_decomposition_contrasts.csv",
    ROOT / "round_03/table03_un_source_and_version_counts.csv",
]
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.1, "savefig.dpi": 300,
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(ax, row, y, *, color, marker="o", zorder=5):
    ax.errorbar(
        100 * row.estimate, y,
        xerr=[[100 * (row.estimate - row.ci_low)], [100 * (row.ci_high - row.estimate)]],
        fmt=marker, color=color, markersize=4.5, capsize=2.5, lw=1.1,
        zorder=zorder,
    )


def figure1():
    source = pd.read_csv(SOURCES[0])
    contrasts = ["translate-minus-original", "hyde-minus-translate"]
    selected = source[
        source.model.eq(ALL) & source.metric.eq("ndcg10") & source.contrast.isin(contrasts)
    ].copy()
    assert len(selected) == 6
    assert selected.fixed_models.eq(8).all()
    assert selected.groupby("corpus").n_queries.first().to_dict() == {"chemistry": 197, "eurlex": 194, "un": 247}
    fig, ax = plt.subplots(figsize=(7, 2.55))
    colors = ["#0072B2", "#D55E00"]
    labels = ["Target-language translation − original", "Target-language HyDE − translation"]
    for y, corpus in enumerate(CORPORA):
        for offset, contrast, color in zip([-.13, .13], contrasts, colors, strict=True):
            row = selected[selected.corpus.eq(corpus) & selected.contrast.eq(contrast)].iloc[0]
            interval(ax, row, y + offset, color=color)
            ax.text(100 * row.ci_high + .30, y + offset, f"{row.estimate*100:+.2f}",
                    color=color, va="center", ha="left", fontsize=8)
    counts = {"chemistry": 197, "eurlex": 194, "un": 247}
    ax.set_yticks(range(3), [f"{CORPUS_LABEL[c]} (n={counts[c]})" for c in CORPORA])
    ax.set_ylim(2.45, -.45)
    ax.set_xlim(-7.2, 13.8)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_xlabel("Change in document-language nDCG@10 (percentage points)", labelpad=5)
    ax.axvline(0, color="#888888", lw=.8)
    ax.grid(axis="x", alpha=.18)
    ax.set_title("Cross-language, target-matched generation: mean across eight fixed encoders", pad=10)
    fig.legend(
        [Line2D([0], [0], color=color, marker="o", markersize=4.5) for color in colors],
        labels, ncol=2, loc="lower center", bbox_to_anchor=(.56, .005), frameon=False,
        handlelength=1.4, columnspacing=1.6,
    )
    fig.subplots_adjust(left=.22, right=.98, top=.80, bottom=.29)
    savefig(fig, OUT, "figure1_language_matched_controls")
    return selected.to_dict("records")


def figure2():
    source = pd.read_csv(SOURCES[1])
    counts = pd.read_csv(SOURCES[2])
    models = ["e5-large-instruct", "embeddinggemma", ALL]
    selected = source[source.model.isin(models) & source.reference.eq("translate_en")].copy()
    chosen_counts = counts[counts.model.isin(models) & counts.reference.eq("translate_en")].set_index("model")
    assert selected.n_queries.eq(247).all()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw={"width_ratios": [1.18, 1]}, sharey=True)
    first_color, extra_color = "#0072B2", "#E69F00"
    for y, model in enumerate(models):
        rows = selected[selected.model.eq(model)].set_index("metric")
        first = rows.loc["first_contribution"]
        extra = rows.loc["additional_contribution"]
        total = rows.loc["ndcg10"]
        assert np.isclose(first.estimate + extra.estimate, total.estimate, atol=1e-12)
        assert first.estimate < 0 < extra.estimate
        left.barh(y, 100 * first.estimate, height=.40, color=first_color, alpha=.85)
        left.barh(y, 100 * extra.estimate, height=.40, color=extra_color, alpha=.85)
        interval(left, total, y, color="#222222", marker="D")
        left.text(total.estimate*100, y-.29, f"{total.estimate*100:+.2f}",
                  ha="center", va="center", fontsize=8)
        hit = rows.loc["hit10"]
        interval(right, hit, y, color="#222222")
        count = chosen_counts.loc[model]
        assert np.isclose(hit.estimate, count.source_hit_delta)
        assert np.isclose(hit.ci_low, count.source_hit_ci_low)
        assert np.isclose(hit.ci_high, count.source_hit_ci_high)
        assert np.isclose(count.net_source_hits / 247, hit.estimate)
        label = f"{count.net_source_hits:.1f}*" if model == ALL else f"{count.net_source_hits:.0f}"
        right.text(5.0, y, label.replace("-", "−"), ha="right", va="center", fontsize=8)
    left.set_yticks(range(3), ["E5-large-instruct", "EmbeddingGemma", "Eight-encoder mean"])
    left.set_ylim(2.5, -.5)
    left.set_xlim(-4.3, 7.1)
    left.set_xticks([-4, -2, 0, 2, 4, 6])
    right.tick_params(axis="y", labelleft=False)
    right.set_xlim(-16.5, 5.5)
    right.set_xticks([-15, -10, -5, 0])
    for ax in [left, right]:
        ax.axvline(0, color="#888888", lw=.8)
        ax.grid(axis="x", alpha=.13)
        ax.set_xlabel("Change (percentage points)", labelpad=5)
    left.set_title("A. Exact version-nDCG@10 decomposition", pad=12)
    right.set_title("B. Source Hit@10", pad=12)
    right.text(3.2, -.43, "Net hits", ha="center", va="bottom", fontsize=8)
    fig.suptitle("UN mixed retrieval: German − English query translation (n=247)", fontsize=9.5, y=.99)
    fig.legend(
        [Patch(facecolor=first_color, alpha=.85), Patch(facecolor=extra_color, alpha=.85),
         Line2D([0], [0], color="#222222", marker="D", markersize=4.5)],
        ["First relevant version", "Additional relevant versions", "Total nDCG (95% CI)"],
        ncol=3, loc="lower center", bbox_to_anchor=(.57, .035), frameon=False,
        columnspacing=1.2, handlelength=1.4,
    )
    fig.text(.57, .008, "* Fractional count is the mean net source-hit change across eight fixed encoders.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.21, right=.98, top=.77, bottom=.29, wspace=.25)
    savefig(fig, OUT, "figure2_un_objective_decomposition")
    return {"contrasts": selected.to_dict("records"), "counts": chosen_counts.reset_index().to_dict("records")}


def captions():
    (OUT / "figure_captions.txt").write_text(
        "FIGURE 1. Language-matched controls in cross-language retrieval.\n"
        "Paired changes in document-language nDCG@10 for target-language translation versus the original query (blue), "
        "and target-language HyDE versus translation (orange). Only single-target settings whose searched language "
        "differs from the original question language are included; generated text is in that searched target language. "
        "For each source question, eligible target-language settings are averaged first, followed by the eight fixed "
        "encoders; source questions then receive equal weight within corpus (197 chemistry, 194 EUR-Lex, 247 UN). "
        "Each retrieved document is a document-language version, ranked by its maximum chunk similarity. Positive "
        "judgments are known source associations: canonical language variants for legal corpora and directly accepted "
        "translation pairs for chemistry. Bars are 95% percentile intervals from 5,000 paired source-question "
        "bootstrap draws (seed 42); encoders and repeated target settings are not independent resampling units. "
        "Labels report changes in percentage points. These are exploratory intervals without multiplicity adjustment. "
        "HyDE uses document encoding and translation uses query encoding; four encoders change prefixes/instructions, "
        "so the contrast does not isolate a causal effect of text expansion alone.\n\n"
        "FIGURE 2. More relevant language versions can coexist with fewer recovered sources.\n"
        "UN mixed-corpus retrieval with German rather than English query translation, evaluated on the same 247 "
        "source questions. Panel A decomposes the change in the recorded document-language nDCG@10 exactly: blue is "
        "the contribution of the first relevant language version and gold the contribution of additional relevant "
        "versions, each using the original per-question ideal-DCG denominator. Their sum is the total change (black "
        "diamond); its whisker is a paired 95% bootstrap interval. The decomposition is an accounting identity, "
        "not a new retrieval metric. Panel B reports source Hit@10, the fraction of questions retrieving any relevant "
        "version of the known source in the first ten document-language results, with paired 95% intervals. Net-hit "
        "labels are changes in recovered source-question counts: E5 loses 22 net hits (18 gained, 40 lost), "
        "EmbeddingGemma loses 20 (6 gained, 26 lost), and the eight-encoder mean loses 17.125 per encoder. "
        "E5 and EmbeddingGemma illustrate the observed metric disagreement; the fixed mean includes all eight "
        "encoders and has an uncertain total nDCG change (+0.22 points; 95% CI −0.71 to +1.14), alongside a lower "
        "source Hit rate (−6.93 points; −9.16 to −4.81). All questions, including no-hit cases, are retained. "
        "Encoders are averaged within each question before computing the fixed mean and resampling; model-specific "
        "rows use equal question weights. Intervals are 5,000 paired source-question percentile bootstrap draws "
        "(seed 42), exploratory and unadjusted for multiplicity. German is absent from the searched UN collection "
        "and from the original question languages; this descriptive result does not establish a semantic-bridge "
        "mechanism or intrinsic language bias. All values are copied from accepted Round 3 tables.\n",
        encoding="utf-8",
    )


def main():
    data = {"figure1": figure1(), "figure2": figure2()}
    captions()
    write_json(OUT / "figure_data.json", data)
    products = [p for p in OUT.glob("figure*") if p.is_file() and p.name != "figure_manifest.json"]
    write_json(OUT / "figure_manifest.json", {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Presentation only; all estimates, confidence intervals and counts copied from accepted Round 2/3 tables",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        "script_sha256": sha(ROOT / "scripts/make_paper_figures.py"),
        "nominal_figure_width_inches": 7,
        "font_sizes_points": {"body": 8.5, "ticks": 8, "legend": 8, "titles": 9},
        "visual_review_status": "pending_root",
        "output_sha256": {p.name: sha(p) for p in sorted(products)},
    })
    print("Saved two compact figure triplets, copied numerical data, captions and provenance manifest.")


if __name__ == "__main__":
    main()
