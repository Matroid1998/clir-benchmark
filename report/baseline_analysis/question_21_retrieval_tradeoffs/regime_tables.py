"""Render the three retrieval regimes from the saved per-setting measurements."""
import numpy as np
import pandas as pd
from table_figure import HERE, render

TITLES = {"monolingual": "Monolingual", "cross_language": "Cross-language", "mixed": "Mixed-language pool"}
NOTES = {
    "monolingual": "Query and search language match. Translation reuses the original question. Each language has equal weight.",
    "mixed": "All corpus languages share one search pool. Translation averages separate runs per output language, then per query language.",
}
keys = ["corpus", "model", "backend", "method"]
data = pd.read_csv(HERE / "by_setting.csv").rename(columns={"retriever": "backend"})
for regime, title in TITLES.items():
    rows = data.loc[data.regime.eq(regime)].copy()
    if regime != "mixed":
        rows = rows.loc[rows.method.ne("translate") | rows.language.eq(rows.target_languages)]
    assert not rows.duplicated(keys + ["setting_id", "language"]).any()
    dims = keys + ["report_status"]
    scores = (rows.groupby(dims + ["setting_id"]).recall10.mean().groupby(level=dims).mean()
              .mul(100).rename("recall10_percent").reset_index())
    control = scores.loc[scores.method.eq("original"), keys[:-1] + ["recall10_percent"]].rename(
        columns={"recall10_percent": "original_percent"})
    scores = scores.merge(control, on=keys[:-1], how="left", validate="many_to_one")
    scores["delta_pp"] = scores.recall10_percent - scores.original_percent
    assert len(scores) == 150 and not scores.duplicated(keys).any()
    if regime == "cross_language":
        previous = pd.read_csv(HERE / "table_values.csv").set_index(keys).sort_index()
        current = scores.set_index(keys).sort_index()
        assert previous.index.equals(current.index)
        np.testing.assert_allclose(previous.recall10_percent, current.recall10_percent, atol=1e-9)
    scores.to_csv(HERE / f"table_values_{regime}.csv", index=False)
    render(scores, title=title, stem=f"recall_cost_latency_table_{regime}",
           route_note=NOTES.get(regime), monolingual=regime == "monolingual")
