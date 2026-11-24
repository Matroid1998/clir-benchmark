"""Rescore saved top-100 chunk rankings and audit their recorded latency. No models or APIs run."""

import argparse
import gzip
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts"
SHARED = ART / "narval-20261119-shared-indexes/artifacts"
GEN = ART / "verified-20261118-query-generation"
CACHE = ART / "analysis-20261124-recall-10-50"
KEY = ["setting_id", "query_id"]
RECALL = ["recall10", "recall50"]
TIMES = ["generation_s", "encoding_s", "retrieval_s", "pipeline_s"]


@cache
def judgments(corpus):
    (path,) = (SHARED / "relevance" / corpus).glob("*/qrels.parquet")
    q = pd.read_parquet(path, columns=KEY + ["chunk_id", "relevance"])
    q = q.loc[q.relevance > 0, KEY + ["chunk_id"]].drop_duplicates()
    denom = q.groupby(KEY).size().rename("relevant_chunks")
    settings = json.loads(path.with_name("settings.json").read_text())
    eligible = pd.read_parquet(path.with_name("eligibility.parquet"), columns=KEY + ["status"])
    eligible = eligible.loc[
        eligible.status.eq("eligible")
        & eligible.setting_id.isin(k for k, v in settings.items() if v["status"] == "ready"),
        KEY,
    ].set_index(KEY)
    assert eligible.index.is_unique and eligible.index.sort_values().equals(
        denom.index.sort_values()
    )
    return q, denom, pa.array(q.chunk_id.unique().tolist(), type=pa.string()), settings


@cache
def variants(method, corpus, lane):
    kind, name = (
        ("translations", "queries.parquet")
        if method == "translate"
        else ("hyde", "passages.parquet")
    )
    (path,) = (GEN / kind / corpus / lane).glob("*/" + name)
    d = pd.read_parquet(path)
    assert d.id.is_unique
    return d.set_index("id")


@cache
def controls():
    catalog = {}
    for m in SHARED.glob("reports/*/*/analysis_manifest.json"):
        run = json.loads(m.with_name("report.json").read_text())["run"]
        key = run["corpus"], run["baseline"], (run.get("model") or {}).get("alias", "BM25")
        catalog[key] = m.with_name("per_query.csv")
    return catalog


@cache
def control_times(path):
    fields = ["query_encoding_seconds_amortized", "retrieval_seconds_amortized"]
    return pd.read_csv(path, usecols=KEY + fields).set_index(KEY)[fields].sum(axis=1, skipna=False)


def evaluate(root, run, output):
    m = json.loads((run / "manifest.json").read_text())
    corpus, baseline = m["corpus"], m["baseline"]
    method, sep, backend = baseline.partition("-")
    if not sep:
        method, backend = "original", baseline
    lane = m.get("translation_language") or m.get("output_language") or "original"
    report = root / "reports" / corpus / run.name
    published = (report / "analysis_manifest.json").is_file()
    status = (
        "published"
        if published
        else "component"
        if method != "original" and backend == "bm25"
        else "incomplete"
    )
    qrels, denom, candidate_chunks, settings = judgments(corpus)
    hits = []
    for batch in pq.ParquetFile(run / "run.parquet").iter_batches(
        columns=KEY + ["rank", "chunk_id"], batch_size=131072
    ):
        ids = batch.column("chunk_id")
        if ids.type != candidate_chunks.type:
            ids = ids.cast(candidate_chunks.type)
        selected = pc.and_(
            pc.less_equal(batch.column("rank"), 50), pc.is_in(ids, value_set=candidate_chunks)
        )
        if pc.any(selected).as_py():
            hits.append(batch.filter(selected).to_pandas())
    data = denom.to_frame()
    if hits:
        true_hits = pd.concat(hits, ignore_index=True).merge(
            qrels, on=KEY + ["chunk_id"], validate="many_to_one"
        )
        assert not true_hits.duplicated(KEY + ["chunk_id"]).any()
        for k in (10, 50):
            n = true_hits[true_hits["rank"] <= k].groupby(KEY).size()
            data[f"recall{k}"] = n.reindex(data.index, fill_value=0) / denom
    else:
        data[RECALL] = 0.0
    assert (data.recall10 <= data.recall50).all() and data.recall50.between(0, 1).all()

    t = pd.read_parquet(run / "timings.parquet").set_index(KEY)
    assert t.index.is_unique and data.index.equals(t.sort_index().index)
    data["retrieval_s"] = t.retrieval_seconds_amortized
    data["encoding_s"] = 0.0
    if (report / "per_query.csv").is_file():
        legacy = (
            pd.read_csv(
                report / "per_query.csv",
                usecols=KEY + ["Recall@10", "query_encoding_seconds_amortized"],
            )
            .set_index(KEY)
            .sort_index()
        )
        assert data.index.equals(legacy.index)
        np.testing.assert_allclose(data.recall10, legacy["Recall@10"], rtol=0, atol=1e-12)
        data["encoding_s"] = legacy.query_encoding_seconds_amortized
    else:
        assert status == "component" and backend == "bm25"
    data["generation_s"] = 0.0
    data["api_usd"] = 0.0
    data["refused"] = False
    if method != "original":
        v = variants(method, corpus, lane).loc[data.index.get_level_values("query_id")]
        assert len(v) == len(data)
        data["refused"] = v.status.eq("provider_refused").to_numpy()
        data["generation_s"] = v[
            "translation_seconds" if method == "translate" else "generation_seconds"
        ].to_numpy()
        data["api_usd"] = v.api_cost_usd.to_numpy()
        assert data.loc[data.refused, RECALL].eq(0).all().all()
        data.loc[data.refused, ["encoding_s", "retrieval_s"]] = 0.0
    data["pipeline_s"] = data[TIMES[:3]].sum(axis=1, skipna=False)
    model = (m.get("model") or {}).get("alias", "BM25")
    control = controls().get((corpus, backend, model))
    data["control_paired_s"] = np.nan
    data["variant_paired_s"] = np.nan
    if control:
        control_s = control_times(control).reindex(data.index)
        paired = control_s.notna() & data.pipeline_s.notna()
        data["control_paired_s"] = control_s.where(paired)
        data["variant_paired_s"] = data.pipeline_s.where(paired)
    data["known_api_cost"] = data.api_usd.notna()
    by = data.groupby(level="setting_id")
    avg = (
        by[RECALL + TIMES + ["api_usd", "control_paired_s", "variant_paired_s"]]
        .mean()
        .rename(columns={"api_usd": "mean_known_api_usd"})
    )
    avg["latency_observations"] = by.pipeline_s.count()
    avg["paired_latency_observations"] = by.variant_paired_s.count()
    avg["queries"], avg["refused"], avg["known_api_cost_rows"] = (
        by.size(),
        by.refused.sum(),
        by.known_api_cost.sum(),
    )
    avg["regime"] = [settings[s]["kind"] for s in avg.index]
    avg["query_language"] = [settings[s]["query_language"] for s in avg.index]
    avg["target_languages"] = [",".join(settings[s]["target_languages"]) for s in avg.index]
    avg = avg.assign(
        corpus=corpus,
        model=model,
        method=method,
        retriever=backend,
        language=lane,
        report_status=status,
        run_id=run.name,
        qrels_hash=m["relevance_hash"],
        cohort_size=len(data),
    )
    temporary = output.with_suffix(".tmp")
    avg.to_csv(temporary)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--available",
        action="store_true",
        help="Process downloaded files and allow an incomplete inventory",
    )
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    inventory = HERE.parent / "question_20_local_results/narval_files.tsv.gz"
    with gzip.open(inventory, "rt") as stream:
        sizes = {s: int(n) for n, s in (line.rstrip("\n").split("\t", 1) for line in stream)}
    rankings = sorted(s for s in sizes if s.endswith("/run.parquet") and "/artifacts/runs/" in s)
    missing = []
    for name in rankings:
        path = ART / name
        run = path.parent
        target = CACHE / (run.name + ".csv")
        if target.exists():
            with target.open() as existing:
                if "paired_latency_observations" in existing.readline():
                    continue
        if any(
            not f.is_file() or f.stat().st_size != sizes[str(f.relative_to(ART))]
            for f in (path, run / "timings.parquet", run / "manifest.json")
        ):
            missing.append(name)
            continue
        evaluate(run.parents[2], run, target)
    print(f"{len(rankings) - len(missing)}/{len(rankings)} runs scored")
    if missing and not args.available:
        raise SystemExit("Some top-100 ranking files have not yet downloaded.")


if __name__ == "__main__":
    main()
