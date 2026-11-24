"""Plot shared corpus sizes and check the cohort used in cross-language retrieval."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ART = HERE.parents[2] / "artifacts/narval-20261119-shared-indexes/artifacts"
NAMES = {"chemistry": "Google Patents Chemistry", "eurlex": "EUR-Lex", "un": "United Nations"}


def counts(corpus):
    runs = (
        json.loads(p.read_text())["run"] for p in (ART / "reports" / corpus).glob("*/report.json")
    )
    (run,) = (r for r in runs if r["baseline"] == "bm25")
    (chunks,) = (
        p.parent
        for p in (ART / "corpus" / corpus).glob("*/manifest.json")
        if json.loads(p.read_text())["corpus_fingerprint"] == run["corpus_fingerprint"]
    )
    n = pq.ParquetFile(chunks / "chunks.parquet").metadata.num_rows
    assert n == json.loads((chunks / "manifest.json").read_text())["chunk_count"]
    relevance = ART / "relevance" / corpus / run["relevance_hash"]
    query_ids = set(
        pq.read_table(relevance / "queries.parquet", columns=["id"]).column(0).to_pylist()
    )
    settings = json.loads((relevance / "settings.json").read_text())
    cross = {
        s for s, v in settings.items() if v["kind"] == "cross_language" and v["status"] == "ready"
    }
    eligible = pq.read_table(
        relevance / "eligibility.parquet", columns=["query_id", "setting_id", "status"]
    )
    observed = {
        r["query_id"]
        for r in eligible.to_pylist()
        if r["status"] == "eligible" and r["setting_id"] in cross
    }
    assert observed == query_ids
    return n, len(query_ids)


def main():
    values = [counts(c) for c in NAMES]
    chunks, questions = zip(*values, strict=True)
    print(dict(zip(NAMES, values, strict=True)), "totals", sum(chunks), sum(questions))
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    fig.subplots_adjust(left=0.255, right=0.95, top=0.77, bottom=0.18)
    fig.text(0.05, 0.9, "Indexed chunks by corpus", fontsize=19, weight="bold", color="#172c39")
    fig.text(
        0.05,
        0.81,
        f"{sum(chunks):,} shared chunks across all indexed languages",
        fontsize=11,
        color="#455d69",
    )
    bars = ax.barh(list(NAMES.values()), chunks, height=0.55, color="#326780")
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{n:,}" for n in chunks], padding=6, fontsize=11, weight="bold")
    ax.set_xlim(0, max(chunks) * 1.24)
    ax.set_xticks([0, 500000, 1000000, 1500000, 2000000], ["0", "0.5M", "1M", "1.5M", "2M"])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dce5e7")
    ax.tick_params(length=0, pad=6, colors="#354e5b")
    ax.spines[:].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"corpus_chunks.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
