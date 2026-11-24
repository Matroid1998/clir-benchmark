"""Report each model and baseline's weakest evaluated cross-language direction."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from table_figure import CORPORA, MODELS

HERE = Path(__file__).resolve().parent
STEM = "recall10_worst_cross_language"
KEY = ["corpus", "model", "retriever", "method"]
METHODS = ("original", "translate", "hyde")


def worst_rows():
    data = pd.read_csv(HERE / "by_setting.csv")
    pair = data.query("regime == 'cross_language' and (method != 'translate' or language == target_languages)").copy()
    assert pair.query_language.ne(pair.target_languages).all()
    assert not pair.target_languages.str.contains(",").any()
    assert not pair.duplicated(KEY + ["setting_id"]).any()
    assert pair.recall10.between(0, 1).all() and pair.queries.gt(0).all()
    expected = pd.read_csv(HERE / "table_values_cross_language.csv").rename(columns={"backend": "retriever"})
    means = pair.groupby(KEY).recall10.mean().mul(100).sort_index()
    previous = expected.set_index(KEY).recall10_percent.sort_index()
    assert means.index.equals(previous.index)
    np.testing.assert_allclose(means, previous, atol=1e-9)
    groups = pair.groupby(KEY)
    pair["evaluated_directions"] = groups.setting_id.transform("size")
    pair["recall10_percent"] = 100 * pair.recall10
    minima = groups.recall10.transform("min")
    picked = pair[np.isclose(pair.recall10, minima, rtol=0, atol=1e-12)].copy()
    picked["tied_directions"] = picked.groupby(KEY).setting_id.transform("size")
    return picked.sort_values(KEY + ["query_language", "target_languages"])


def render(rows):
    costs = pd.read_csv(HERE / "generation_by_corpus.csv").set_index(["corpus", "method"])
    with PdfPages(HERE / f"{STEM}.pdf") as pdf:
        for corpus, name in CORPORA.items():
            part = rows[rows.corpus.eq(corpus)]
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.subplots_adjust(left=.035, right=.98, top=.975, bottom=.025)
            ax.set(xlim=(0, 100), ylim=(100, 0))
            ax.axis("off")

            def label(x, y, s, size=10, **style):
                ax.text(x, y, s, fontsize=size, va="center", color=style.pop("color", "#193748"), **style)

            def band(y, h, color):
                ax.add_patch(plt.Rectangle((0, y), 100, h, facecolor=color, edgecolor="none"))

            label(.6, 3, f"{name}   |   Lowest cross-language Recall@10", 20, weight="bold")
            label(.6, 7.3, "Each cell gives the lowest mean chunk recall (%) across evaluated directions. Read arrows as original question → searched documents.", 11)
            label(.6, 10.3, "Signed changes are percentage points versus the original retriever on the same direction. Translation uses the document language.", 10)
            centers = [28, 41, 54, 67, 80, 93]
            band(13, 7, "#e3ecf0")
            label(1, 18, "Retriever / model", weight="bold")
            for x, title in [(41, "Single retriever"), (80, "Hybrid  ·  BM25 + dense")]:
                label(x, 15, title, 12, ha="center", weight="bold")
            for x, method in zip(centers, METHODS * 2, strict=True):
                label(x, 18.2, method.replace("hyde", "HyDE").replace("original","Original").replace("translate","Translate"), ha="center")
            ties = []
            for i, (model, model_name) in enumerate(MODELS.items()):
                y = 20.5 + 5.75 * i
                if i % 2:
                    band(y, 5.75, "#f1f5f6")
                label(1, y + 2.7, model_name, 10.5)
                for j, x in enumerate(centers):
                    backend = ("bm25" if model == "BM25" else "dense") if j < 3 else "hybrid"
                    method = METHODS[j % 3]
                    group = part[(part.model == model) & (part.retriever == backend) & (part.method == method)]
                    if group.empty:
                        label(x, y + 2.7, "—", ha="center")
                        continue
                    row = group.iloc[0]
                    value = f"{row.recall10_percent:.1f}%" + ("†" if row.report_status == "incomplete" else "")
                    if method != "original" and len(group) == 1 and pd.notna(row.delta_recall10_pp):
                        value += f" ({row.delta_recall10_pp:+.1f})"
                    routes = [f"{r.query_language}→{r.target_languages}" for r in group.itertuples()]
                    direction = routes[0]
                    if len(routes) > 1:
                        direction = f"{len(routes)} tied directions [{len(ties)+1}]"
                        ties.append(", ".join(routes))
                    label(x, y + 1.7, value, 10.8, ha="center", weight="bold")
                    label(x, y + 4.0, direction, 9.2, ha="center", color="#476373")
            for i, routes in enumerate(ties):
                label(1, 74 + 2.6*i, f"[{i+1}]  {routes}", 9.8)
            label(1, 81, "† Final report verification incomplete.   — Ranking unavailable or inapplicable.   The CSV gives eligible query counts and matched changes for every tie.", 9)
            band(85, 4.5, "#e3ecf0")
            label(1, 87.2, "Query generation cost and added latency across the corpus", 12, weight="bold")
            for i, method in enumerate(("translate","hyde")):
                r = costs.loc[corpus,method]
                price = r.known_attempt_usd / (r.attempts - r.missing_attempt_costs)
                label(1 + 48*i, 92, f"{'Translate' if i == 0 else 'HyDE'}   ${price:.6f} / API attempt     {r.mean_known_paid_generation_s:.2f} s / paid query", 10.5)
            label(1, 96, "Cost and latency use recorded generation attempts across languages. Only the recall columns select the worst direction.", 9.7)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    results = worst_rows()
    results[KEY + ["query_language","target_languages","queries","recall10_percent","delta_recall10_pp","evaluated_directions","tied_directions","run_id","report_status"]].to_csv(HERE / f"{STEM}.csv", index=False)
    render(results)
