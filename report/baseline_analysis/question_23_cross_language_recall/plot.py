"""Plot Recall@10 or Recall@50 by language pair from saved November 20 evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = "narval-20261119-shared-indexes"
SHARED = ROOT / "artifacts" / CAMPAIGN / "artifacts"
CATALOG = HERE.parent / "question_20_local_results" / "retrieval_settings.csv"
RESCORED = HERE.parent / "question_21_retrieval_tradeoffs" / "by_setting.csv"
CORPORA = {"chemistry": "Chemistry patents", "eurlex": "EUR-Lex", "un": "UN"}
BASELINES = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
MODELS = ("bge-m3", "e5-large-instruct", "gte-multilingual-base", "qwen3-0.6B")
LANGUAGES = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "es": "Spanish",
}
RECALL_10 = "Recall@10"
PAIRWISE_REGIMES = ("monolingual", "cross_language")
PAIR_KEYS = ["question_language", "document_language"]
QUERY_KEYS = ["setting_id", "query_id"]
INK = "#172d3d"
MUTED = "#455c6b"
MISSING = "#e3e8ed"
COLORS = LinearSegmentedColormap.from_list(
    "recall", ["#f1fbfa", "#b9e2dc", "#56aea6", "#167670", "#083c42"]
).with_extremes(bad="white")


def with_recall50(catalog):
    """Match top-100 rescoring by run and setting, and cross-check published recall."""
    rescored = pd.read_csv(RESCORED, keep_default_na=False)
    rescored = rescored.loc[
        rescored.method.eq("original") & rescored.run_id.isin(catalog.run_id)
    ].rename(columns={"retriever": "baseline", "queries": "query_count"})
    rescored["model"] = rescored.model.replace("BM25", "")
    keys = [
        "run_id", "setting_id", "corpus", "baseline", "model", "report_status",
        "regime", "query_language", "target_languages", "query_count",
    ]
    catalog = catalog.merge(
        rescored[keys + ["qrels_hash", "recall10", "recall50"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    pd.testing.assert_series_equal(
        catalog.recall10, catalog[RECALL_10], check_names=False, atol=1e-12, rtol=0
    )
    recall50 = catalog.recall50
    if not (
        recall50.between(0, 1)
        & recall50.ge(pd.to_numeric(catalog["Recall@20"]) - 1e-12)
        & recall50.le(pd.to_numeric(catalog["Recall@100"]) + 1e-12)
    ).all():
        raise ValueError("Recall@50 is missing or outside the published recall bounds")
    return catalog.rename(columns={"recall50": "Recall@50"})


def corpus_scores(catalog, corpus, cutoff):
    """Check the published means and verify that the runs scored identical query IDs."""
    (qrels_path,) = (SHARED / "relevance" / corpus).glob("*/qrels.parquet")
    settings = json.loads(qrels_path.with_name("settings.json").read_text())
    paired = [s for s in settings.values() if s["kind"] in PAIRWISE_REGIMES]
    if any(s["status"] != "ready" or len(s["target_languages"]) != 1 for s in paired):
        raise ValueError(f"Unexpected pairwise settings for {corpus}")
    pairs = pd.DataFrame(
        {
            "setting_id": s["id"],
            "question_language": s["query_language"],
            "document_language": s["target_languages"][0],
            "regime": s["kind"],
            "eligible_questions": s["counts"].get("eligible", 0),
        }
        for s in paired
    ).set_index("setting_id")
    expected = pairs.loc[pairs.eligible_questions.gt(0), "eligible_questions"].sort_index()
    qrels = pd.read_parquet(qrels_path, columns=QUERY_KEYS + ["relevance"])
    judged = qrels.loc[
        qrels.setting_id.isin(pairs.index) & qrels.relevance.gt(0), QUERY_KEYS
    ].drop_duplicates()
    eligible = pd.MultiIndex.from_frame(judged).sort_values()
    pd.testing.assert_series_equal(judged.groupby("setting_id").size(), expected, check_names=False)

    runs = catalog[catalog.corpus.eq(corpus)].groupby(["baseline", "model"])
    expected_runs = {("bm25", "")}
    expected_runs.update((baseline, model) for baseline in ("dense", "hybrid") for model in MODELS)
    if set(runs.groups) != expected_runs:
        raise ValueError(f"Missing or unexpected runs for {corpus}")
    fingerprints = set()
    for (baseline, model), rows in runs:
        (relative_dir,) = rows.report_dir.unique()
        directory = ROOT / relative_dir
        run = json.loads((directory / "report.json").read_text())["run"]
        manifest = json.loads((directory / "analysis_manifest.json").read_text())
        alias = (run["model"] or {}).get("alias", "")
        if (run["corpus"], run["baseline"], alias) != (corpus, baseline, model):
            raise ValueError(f"Run identity mismatch in {directory}")
        if tuple(rows.run_id.unique()) != (manifest["run_id"],):
            raise ValueError(f"Published report mismatch in {directory}")
        if run["relevance_hash"] != qrels_path.parent.name:
            raise ValueError(f"Relevance artifact mismatch in {directory}")
        if cutoff == 50 and not rows.qrels_hash.eq(qrels_path.parent.name).all():
            raise ValueError(f"Recall@50 relevance artifact mismatch in {directory}")
        fingerprints.add(run["corpus_fingerprint"])

        columns = QUERY_KEYS + ["regime", RECALL_10]
        per_query = pd.read_csv(directory / "per_query.csv", usecols=columns)
        pairwise = per_query.loc[per_query.regime.isin(PAIRWISE_REGIMES)].set_index(QUERY_KEYS)
        if (
            not pairwise.index.sort_values().equals(eligible)
            or not pairwise[RECALL_10].between(0, 1).all()
        ):
            raise ValueError(f"Query cohort or Recall@10 mismatch in {directory}")
        per_setting = rows.set_index("setting_id").sort_index()
        pd.testing.assert_series_equal(per_setting.query_count, expected, check_names=False)
        pd.testing.assert_series_equal(
            pairwise.groupby("setting_id")[RECALL_10].mean(),
            per_setting[RECALL_10],
            atol=1e-12,
            rtol=0,
        )
    if len(fingerprints) != 1:
        raise ValueError(f"Runs used different chunk corpora for {corpus}")

    recall_column = f"recall_at_{cutoff}_percent"
    means = (
        catalog.loc[catalog.corpus.eq(corpus)]
        .groupby(["baseline", "setting_id"])[f"Recall@{cutoff}"]
        .mean()
        .mul(100)
        .rename(recall_column)
    )
    scores = pairs.reset_index().merge(pd.DataFrame({"baseline": list(BASELINES)}), how="cross")
    scores = scores.join(means, on=["baseline", "setting_id"], validate="one_to_one")
    if not scores[recall_column].notna().equals(scores.eligible_questions.gt(0)):
        raise ValueError(f"Missing or unexpected Recall@{cutoff} values for {corpus}")
    scores.insert(0, "corpus", corpus)
    scores["models"] = scores.baseline.map(
        {"bm25": "", "dense": " | ".join(MODELS), "hybrid": " | ".join(MODELS)}
    )
    return scores.sort_values(PAIR_KEYS + ["baseline"])


def render(corpus, scores, cutoff):
    languages = sorted(scores.question_language.unique(), key=list(LANGUAGES).index)
    labels = [LANGUAGES[code] for code in languages]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.65), facecolor="white")
    fig.subplots_adjust(left=0.096, right=0.976, bottom=0.26, top=0.795, wspace=0.13)
    fig.text(
        0.055,
        0.927,
        f"{CORPORA[corpus]}  ·  Recall@{cutoff} by language pair",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.876,
        "Original questions  ·  November 20 verified benchmark",
        fontsize=12,
        color=MUTED,
    )

    for ax, (baseline, label) in zip(axes, BASELINES.items(), strict=True):
        matrix = (
            scores.loc[scores.baseline.eq(baseline)]
            .pivot(index=PAIR_KEYS[0], columns=PAIR_KEYS[1], values=f"recall_at_{cutoff}_percent")
            .reindex(index=languages, columns=languages)
            .to_numpy()
        )
        image = ax.imshow(
            matrix, cmap=COLORS, vmin=0, vmax=100, interpolation="none", aspect="auto"
        )
        ax.set_title(label, fontsize=15, fontweight="bold", color=INK, pad=10)
        ax.set_xticks(positions, labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(positions, labels if ax is axes[0] else [""] * len(labels))
        ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors=INK)
        ax.set_xticks(positions - 0.5, minor=True)
        ax.set_yticks(positions - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for (row, col), value in np.ndenumerate(matrix):
            if np.isnan(value):
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=MISSING,
                        edgecolor="none",
                        zorder=1.5,
                    )
                )
                ax.text(col, row, "—", ha="center", va="center", color=MUTED, fontsize=10)
                continue

            text_color = "white" if value >= 70 else "black"
            if row == col:
                ax.add_patch(
                    Rectangle(
                        (col - 0.45, row - 0.45),
                        0.90,
                        0.90,
                        fill=False,
                        edgecolor=text_color,
                        linewidth=1.1,
                    )
                )
            ax.text(
                col,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if row == col else "normal",
            )

    axes[0].set_ylabel("Question language", labelpad=11, fontsize=11, color=INK)
    fig.text(0.536, 0.159, "Document language", ha="center", fontsize=11, color=INK)
    bar = fig.colorbar(
        image,
        cax=fig.add_axes([0.665, 0.154, 0.282, 0.016]),
        orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
        format=PercentFormatter(100, decimals=0),
    )
    bar.ax.tick_params(labelsize=9, length=0, colors=MUTED)
    bar.outline.set_visible(False)
    note = f"Outlined diagonal cells show same-language Recall@{cutoff}."
    if scores.eligible_questions.eq(0).any():
        note += " Gray cells with a dash had no eligible questions."
    fig.text(0.055, 0.087, note, fontsize=10, color=MUTED)
    fig.text(
        0.055,
        0.048,
        "Dense and hybrid average BGE-M3, E5 Large Instruct, GTE Multilingual Base, and Qwen3-0.6B.",
        fontsize=10,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff", type=int, choices=(10, 50), default=10,
        help="Rank cutoff. Use 50 to read the saved top-100 rescoring table (default: 10).",
    )
    cutoff = parser.parse_args().cutoff
    catalog = pd.read_csv(CATALOG, keep_default_na=False)
    catalog = catalog.loc[
        catalog.campaign.eq(CAMPAIGN)
        & catalog.report_status.eq("published")
        & catalog.regime.isin(PAIRWISE_REGIMES)
        & (catalog.baseline.eq("bm25") | catalog.model.isin(MODELS))
    ].copy()
    catalog[RECALL_10] = pd.to_numeric(catalog[RECALL_10])
    if cutoff == 50:
        catalog = with_recall50(catalog)
    scores = pd.concat(
        [corpus_scores(catalog, corpus, cutoff) for corpus in CORPORA], ignore_index=True
    )
    suffix = "" if cutoff == 10 else f"_recall_at_{cutoff}"
    table = "recall_by_language_pair.csv" if cutoff == 10 else f"recall_by_language_pair_at_{cutoff}.csv"
    scores.to_csv(HERE / table, index=False, float_format="%.8f")
    with PdfPages(HERE / f"cross_language_recall_at_{cutoff}.pdf") as pdf:
        for corpus in CORPORA:
            fig = render(corpus, scores.loc[scores.corpus.eq(corpus)], cutoff)
            fig.savefig(HERE / f"{corpus}{suffix}.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
