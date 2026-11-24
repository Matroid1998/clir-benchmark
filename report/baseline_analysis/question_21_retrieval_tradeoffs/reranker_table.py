"""Append matched reranking results to the saved cross-language comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
CORPORA = {"chemistry": "Google Patents Chemistry", "eurlex": "EUR-Lex", "un": "United Nations"}
MODELS = {
    "BM25": "BM25",
    "embeddinggemma": "EmbeddingGemma",
    "bge-m3": "BGE-M3",
    "qwen3-0.6B": "Qwen3 Embedding 0.6B",
    "nomic-v2-moe": "Nomic Embed v2 MoE",
    "granite-278m": "Granite 278M",
    "gte-multilingual-base": "GTE Multilingual Base",
    "arctic-embed-l-v2": "Arctic Embed L v2.0",
    "e5-large-instruct": "E5 Large Instruct",
}
DIM = ["corpus", "model", "backend"]
BLUE, GREEN, RED, GREY = "#254b61", "#006553", "#9b3929", "#526671"


def load(path):
    return json.loads(path.read_text())


def collect(campaign):
    state = campaign / "job_status"
    plan = load(state / "plan.json")
    records = []
    for corpus, spec in plan["corpora"].items():
        settings = load(Path(spec["relevance"]) / "settings.json")
        for group in spec["lanes"]:
            for alias, mode in group:
                receipt = load(state / f"report-{corpus}-{mode}-{alias or 'lexical'}.json")
                comparison = load(Path(receipt["comparison"]) / "report.json")
                if (comparison["baseline"], comparison["model"], comparison["corpus"]) != (
                    mode,
                    alias,
                    corpus,
                ):
                    raise ValueError(f"Wrong comparison for {receipt}")
                if comparison["reranker"] != plan["reranker"]:
                    raise ValueError("Mismatched reranker configuration")
                for row in comparison["settings"]:
                    records.append(
                        {
                            "corpus": corpus,
                            "model": alias or "BM25",
                            "backend": comparison["parent_baseline"],
                            "setting_id": row["setting_id"],
                            "regime": settings[row["setting_id"]]["kind"],
                            "query_language": row["query_language"],
                            "target_languages": row["target_languages"],
                            "query_count": row["query_count"],
                            **{
                                f"{kind}{k}_percent": 100 * row[f"{name}.Recall@{k}"]
                                for k in (10, 50)
                                for kind, name in (
                                    ("recall", "reranked_metrics"),
                                    ("control", "full_parent_metrics"),
                                )
                            },
                            "added_seconds": row["mean_added_component_seconds"],
                            "parent_seconds": row["mean_parent_component_seconds"],
                            "pipeline_component_seconds": row["mean_component_seconds"],
                            "added_api_usd": 0.0,
                            "gpu_cost_usd": row["mean_reranking_cost_usd"],
                            "comparison": receipt["comparison"],
                        }
                    )
    data = pd.DataFrame.from_records(records)
    if data.duplicated([*DIM, "setting_id"]).any():
        raise ValueError("Duplicate reranking comparison")
    return data, plan


def combine(settings, historical):
    columns = [
        "recall10_percent",
        "control10_percent",
        "recall50_percent",
        "control50_percent",
        "added_seconds",
        "parent_seconds",
        "pipeline_component_seconds",
    ]
    # Each eligible query has equal weight within its direction. Directions have equal weight.
    cross = settings.loc[settings.regime.eq("cross_language")].groupby(DIM)
    mean = cross[columns].agg(lambda x: x.mean(skipna=False))
    mean["directions"] = cross.size()
    mean["query_settings"] = cross.query_count.sum()
    identity = [*DIM, "method"]
    values = historical.set_index(identity, verify_integrity=True)[
        ["report_status", "recall10_percent"]
    ].to_dict("index")
    for key, row in mean.iterrows():
        control = (*key, "original")
        values[control] = {
            "previously_reported_percent": values.get(control, {}).get("recall10_percent"),
            "report_status": "verified",
            "recall10_percent": row.control10_percent,
        }
        values[(*key, "rerank")] = {
            "report_status": "verified",
            "recall10_percent": row.recall10_percent,
        }
    old = pd.DataFrame(
        [{**dict(zip(identity, key, strict=True)), **record} for key, record in values.items()]
    )
    controls = old.loc[old.method.eq("original"), [*DIM, "recall10_percent"]].rename(
        columns={"recall10_percent": "original_percent"}
    )
    combined = old.merge(controls, on=DIM, how="left", validate="many_to_one")
    combined["delta_pp"] = combined.recall10_percent - combined.original_percent
    return combined, mean


def value(text, delta):
    if pd.isna(text):
        return "—"
    if pd.isna(delta):
        return f"{text:.1f}"
    return f"{text:.1f} ({delta:+.1f})"


def render(directory, combined, means, costs, plan):
    indexed = combined.set_index([*DIM, "method"])
    methods = ["original", "translate", "hyde", "rerank"]
    questions = sum(spec["queries"] for spec in plan["corpora"].values())

    def page(title, subtitle):
        fig = plt.figure(figsize=(15, 12.7), facecolor="white")
        fig.text(0.04, 0.967, title, fontsize=22, weight="bold", color=BLUE)
        fig.text(0.04, 0.938, subtitle, fontsize=11, color=GREY)
        return fig

    def grid(fig, box, headers, records, widths, colors=None):
        ax = fig.add_axes(box)
        ax.axis("off")
        table = ax.table(
            cellText=records,
            colLabels=headers,
            colWidths=widths,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_facecolor("#e4edf1" if r == 0 else "#f2f5f6" if r % 2 == 0 else "white")
            if r == 0:
                cell.set_text_props(weight="bold", color=BLUE)
            elif colors and (r, c) in colors:
                cell.set_text_props(color=colors[r, c])
            if c == 0:
                cell.get_text().set_ha("left")
        return table

    with PdfPages(directory / "recall_cost_latency_table_with_reranker.pdf") as pdf:
        fig = page(
            "Cross-language chunk Recall@10",
            f"{questions:,} verified questions in the full cohort. Directions have equal weight. Parentheses show the change from the original retriever in percentage points.",
        )
        for idx, (corpus, title) in enumerate(CORPORA.items()):
            top = 0.901 - idx * 0.282
            fig.text(0.04, top, title, fontsize=14, weight="bold", color=BLUE)
            fig.text(0.36, top, "Single retriever", fontsize=11, weight="bold", color=BLUE)
            fig.text(
                0.77,
                top,
                "Hybrid  ·  BM25 + dense",
                fontsize=11,
                weight="bold",
                color=BLUE,
                ha="center",
            )
            cells, colors = [], {}
            for model, label in MODELS.items():
                display = [label]
                for backend in ("bm25" if model == "BM25" else "dense", "hybrid"):
                    for method in methods:
                        key = (corpus, model, backend, method)
                        text = "—"
                        if key in indexed.index:
                            row = indexed.loc[key]
                            delta = row.delta_pp if method != "original" else None
                            text = value(row.recall10_percent, delta)
                            if delta is not None and pd.notna(delta) and abs(delta) >= 0.05:
                                colors[len(cells) + 1, len(display)] = GREEN if delta > 0 else RED
                        display.append(text)
                cells.append(display)
            grid(
                fig,
                [0.035, top - 0.255, 0.93, 0.244],
                ["Retriever / model"] + ["Original", "Translate", "HyDE", "Rerank"] * 2,
                cells,
                [0.18] + [0.1025] * 8,
                colors,
            )
        fig.text(
            0.04,
            0.042,
            "Reranking scores up to 100 original-query candidates. Translation uses the searched language. HyDE uses the original question language.",
            fontsize=10,
            color=GREY,
        )
        pdf.savefig(fig)
        fig.savefig(directory / "recall10_with_reranker.png", dpi=160)
        plt.close(fig)

        fig = page(
            "Cost and latency added per query",
            "Reranking uses a local Narval A100 GPU. Its external API charge is $0. The GPU has no configured monetary rate.",
        )
        short_names = {**CORPORA, "chemistry": "Chemistry"}
        columns = [(c, b) for c in CORPORA for b in ("single", "hybrid")]
        latency = []
        for model, label in MODELS.items():
            row = [label]
            for corpus, kind in columns:
                key = (
                    corpus,
                    model,
                    "hybrid" if kind == "hybrid" else "bm25" if model == "BM25" else "dense",
                )
                seconds = means.loc[key, "added_seconds"] if key in means.index else None
                row.append("—" if pd.isna(seconds) else f"{seconds:.3f} s")
            latency.append(row)
        fig.text(
            0.04,
            0.889,
            "Reranker inference time per cross-language query",
            fontsize=15,
            color=BLUE,
            weight="bold",
        )
        grid(
            fig,
            [0.035, 0.51, 0.93, 0.355],
            ["First-stage retriever"]
            + [f"{short_names[c]}\n{kind.capitalize()}" for c, kind in columns],
            latency,
            [0.22] + [0.13] * 6,
        )
        fig.text(
            0.04,
            0.476,
            "Elapsed inference time includes tokenization and GPU synchronization. Each batch is divided across its query–chunk pairs.",
            fontsize=10,
            color=GREY,
        )
        fig.text(
            0.04,
            0.455,
            "Scores retain their original timings on reuse. Means include eligible queries with empty rankings. Model loading is recorded separately.",
            fontsize=10,
            color=GREY,
        )
        fig.text(0.04, 0.395, "Query-generation baselines", fontsize=15, color=BLUE, weight="bold")
        generation = []
        for corpus, label in CORPORA.items():
            display = [label]
            for method in ("translate", "hyde"):
                rec = costs.loc[corpus, method]
                attempts = rec.attempts - rec.missing_attempt_costs
                display.extend(
                    [
                        f"${rec.known_attempt_usd / attempts:.6f}" if attempts else "—",
                        f"{rec.mean_known_paid_generation_s:.2f} s",
                    ]
                )
            generation.append(display)
        grid(
            fig,
            [0.035, 0.209, 0.93, 0.158],
            [
                "Corpus",
                "Translate\nUSD / API attempt",
                "Translate\ns / paid query",
                "HyDE\nUSD / API attempt",
                "HyDE\ns / paid query",
            ],
            generation,
            [0.22, 0.195, 0.195, 0.195, 0.195],
        )
        fig.text(
            0.04,
            0.17,
            "Generation means use recorded prices and timings, including retries. Same-language translation passes through the original question.",
            fontsize=10,
            color=GREY,
        )
        fig.text(
            0.04,
            0.147,
            "The timing CSV also records parent retrieval and estimated component totals. End-to-end latency was not measured.",
            fontsize=10,
            color=GREY,
        )
        pdf.savefig(fig)
        fig.savefig(directory / "added_cost_latency_with_reranker.png", dpi=160)
        plt.close(fig)


def main(campaign):
    settings, plan = collect(campaign)
    combined, means = combine(settings, pd.read_csv(HERE / "table_values.csv"))
    costs = pd.read_csv(HERE / "generation_by_corpus.csv").set_index(["corpus", "method"])
    directory = campaign / "report"
    directory.mkdir(exist_ok=True)
    settings.to_csv(directory / "reranker_by_setting.csv", index=False)
    means.to_csv(directory / "reranker_cross_language.csv")
    combined.to_csv(
        directory / "table_values_with_reranker.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    render(directory, combined, means, costs, plan)
    print(directory / "recall_cost_latency_table_with_reranker.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    main(parser.parse_args().campaign)
