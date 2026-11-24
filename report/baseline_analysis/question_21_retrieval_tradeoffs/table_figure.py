"""Render one cross-language Recall@10 table and the cost of query generation."""

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from score import ART, CACHE, HERE, evaluate

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


def table_data():
    d = pd.read_csv(HERE.parent / "question_20_local_results/retrieval_settings.csv")
    d = d.loc[d.regime.eq("cross_language")].copy()
    d["method"] = d.baseline.str.extract(r"^(translate|hyde)-")[0].fillna("original")
    d["backend"] = d.baseline.str.split("-").str[-1]
    d["model"] = d.model.fillna("BM25")
    d = d.loc[d.method.ne("translate") | d.translation_language.eq(d.target_languages)]
    dims = ["corpus", "model", "backend", "method", "report_status"]
    main = d.groupby(dims)["Recall@10"].mean().mul(100).rename("recall10_percent").reset_index()
    parent = ART / "narval-20261120-transformed-retrieval/artifacts"
    component_paths = set()
    for p in (parent / "reports").glob("*/*/report.json"):
        m = json.loads(p.read_bytes())["run"]
        if m["baseline"].endswith("hybrid"):
            component_paths.add(
                parent / "runs" / m["corpus"] / Path(m["component_runs"]["bm25"]).name
            )
    assert len(component_paths) == 20
    lexical = []
    with tempfile.TemporaryDirectory() as tmp:
        for run in sorted(component_paths):
            f = CACHE / (run.name + ".csv")
            if not f.exists():
                f = Path(tmp) / (run.name + ".csv")
                evaluate(parent, run, f)
            lexical.append(pd.read_csv(f))
    lexical_scores = pd.concat(lexical).query("regime=='cross_language'")
    lexical_scores = lexical_scores[
        lexical_scores.method.ne("translate")
        | lexical_scores.language.eq(lexical_scores.target_languages)
    ].rename(columns={"retriever": "backend"})
    bm25 = (
        lexical_scores.groupby(dims)
        .recall10.mean()
        .mul(100)
        .rename("recall10_percent")
        .reset_index()
    )
    out = pd.concat([main, bm25], ignore_index=True)
    control = out[out.method == "original"][
        ["corpus", "model", "backend", "recall10_percent"]
    ].rename(columns={"recall10_percent": "original_percent"})
    out = out.merge(control, on=["corpus", "model", "backend"], how="left", validate="many_to_one")
    out["delta_pp"] = out.recall10_percent - out.original_percent
    assert len(out) == 150 and not out.duplicated(["corpus", "model", "backend", "method"]).any()
    return out


def main():
    scores = table_data()
    scores.to_csv(HERE / "table_values.csv", index=False)
    render(scores)


def render(scores, *, title="Cross-language", stem="recall_cost_latency_table", route_note=None, monolingual=False):
    data = scores.set_index(["corpus", "model", "backend", "method"])
    costs = pd.read_csv(HERE / "generation_by_corpus.csv").set_index(["corpus", "method"])
    fig, ax = plt.subplots(figsize=(12.7, 14.9))
    fig.subplots_adjust(left=0.025, right=0.978, bottom=0.018, top=0.992)
    ax.set(xlim=(0, 100), ylim=(111, 0))
    ax.axis("off")
    ink = "#172c39"
    muted = "#455d69"
    teal = "#006657"
    rust = "#9c392a"
    blue = "#214b66"

    def txt(x, y, s, sz=10, **kw):
        ax.text(x, y, s, fontsize=sz, va="center", color=kw.pop("color", ink), **kw)

    def rect(x, y, w, h, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=-1))

    def line(x1, x2, y, color="#c5cfd3", lw=0.65):
        ax.plot([x1, x2], [y, y], color=color, lw=lw)

    txt(0.8, 2, f"{title} chunk Recall@10", 23, weight="bold")
    txt(
        0.8,
        4.7,
        "Recall in %. Parentheses show the change from the original query in percentage points.",
        11,
        color=muted,
    )
    txt(
        0.8,
        6.7,
        route_note or "Translation uses the searched language. HyDE uses the original question language. Language directions have equal weight.",
        10.2,
        color=muted,
    )
    centers = [25.7, 38.6, 52.8, 65.7, 78.6, 92.8]
    rect(0, 8.4, 100, 5.65, "#e9eff2")
    txt(0.9, 12.6, "Retriever / model", 10.2, weight="bold")
    txt(39.2, 9.9, "Single retriever", 12, ha="center", weight="bold", color=blue)
    txt(79.2, 9.9, "Hybrid  ·  BM25 + dense", 12, ha="center", weight="bold", color=blue)
    for x, h in zip(centers, ["Original", "Translate", "HyDE"] * 2, strict=True):
        txt(x, 12.5, h, 10, ha="center", color=muted, weight="bold")
    line(21.2, 59, 11.1)
    line(60.7, 99.3, 11.1)
    y = 14.5
    for corpus, title in CORPORA.items():
        rect(0, y, 100, 2.25, "#dce8ed")
        txt(0.9, y + 1.12, title, 12, weight="bold", color=blue)
        y += 2.3
        for i, (model, label) in enumerate(MODELS.items()):
            if i % 2 == 1:
                rect(0, y, 100, 2.05, "#f5f7f7")
            cy = y + 1.03
            txt(0.9, cy, label, 10.1, weight="bold" if model == "BM25" else "normal")
            for j, x in enumerate(centers):
                method = ("original", "translate", "hyde")[j % 3]
                backend = ("bm25" if model == "BM25" else "dense") if j < 3 else "hybrid"
                key = (corpus, model, backend, method)
                if key not in data.index:
                    txt(x, cy, "—", 10, ha="center", color="#86959c")
                    continue
                row = data.loc[key]
                val = f"{row.recall10_percent:.1f}" + (
                    "†" if row.report_status == "incomplete" else ""
                )
                if method == "original" or pd.isna(row.delta_pp):
                    txt(x, cy, val, 10.7, ha="center")
                else:
                    txt(x - 0.35, cy, val, 10.7, ha="right")
                    delta = round(row.delta_pp, 1)
                    txt(
                        x + 0.12,
                        cy,
                        f"({delta:+.1f})" if delta else "(0.0)",
                        8.7,
                        color=teal if delta > 0 else rust if delta < 0 else muted,
                    )
            y += 2.05
        line(0, 100, y)
        y += 0.7
    ax.plot([60.1, 60.1], [8.4, y - 0.7], color="#b8c7ce", lw=0.8)
    y += 1.0
    txt(0.8, y, "Cost and latency added by query generation", 16, weight="bold")
    y += 2.7
    rect(0, y, 100, 4.5, "#e9eff2")
    txt(0.9, y + 3.0, "Corpus", 10.2, weight="bold")
    txt(39, y + 1.1, "Translate then retrieve", 11.4, ha="center", weight="bold", color=blue)
    txt(79, y + 1.1, "HyDE", 11.4, ha="center", weight="bold", color=blue)
    for x, label in [
        (29.5, "USD / API call"),
        (48.6, "Added s / paid query"),
        (69.5, "USD / API call"),
        (88.6, "Added s / paid query"),
    ]:
        txt(x, y + 3.1, label, 9.8, ha="center", color=muted)
    y += 4.6
    for i, (c, name) in enumerate(CORPORA.items()):
        if i % 2:
            rect(0, y, 100, 2.2, "#f5f7f7")
        txt(0.9, y + 1.1, name, 10.1)
        for method, x in [("translate", 29.5), ("hyde", 69.5)]:
            row = costs.loc[c, method]
            call = row.known_attempt_usd / (row.attempts - row.missing_attempt_costs)
            txt(x, y + 1.1, ("No call" if monolingual and method == "translate" else f"${call:.6f}"), 10.6, ha="center")
            txt(x + 19.1, y + 1.1, ("0 s / query" if monolingual and method == "translate" else f"{row.mean_known_paid_generation_s:.2f} s"), 10.6, ha="center")
        y += 2.2
    line(0, 100, y)
    for i, s in enumerate(
        [
            "† Original ranking available; its final report verification failed. — No corresponding run. Parentheses require an available original score.",
            "Cost is mean reported USD per API attempt. Added time is mean LLM time per query requiring a call, including recorded retries.",
            "Same-language translation costs $0 and adds 0 s. Means exclude unreported prices or durations (8 translation attempts, 6 HyDE attempts).",
            "Generation is shared across models. Retrieval, indexing and queue time are excluded from the added LLM latency.",
        ]
    ):
        txt(0.9, y + 1.5 + 1.55 * i, s, 8.9, color=muted)
    ax.set_ylim(y + 7.3, 0)
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / (stem + "." + suffix), dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
