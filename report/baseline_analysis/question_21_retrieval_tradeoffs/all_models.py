"""Recall-only sensitivity analysis for all eight models, including missing original hybrid controls."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from score import HERE, KEY, SHARED, judgments

from clir_benchmark.config import HybridConfig
from clir_benchmark.retrieval import fuse_chunk_ids


def candidates(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=KEY + ["chunk_ids"], batch_size=16):
        yield from zip(*(col.to_pylist() for col in batch.columns), strict=True)


def fusion_summary(dense, lexical, config):
    corpus = dense.parent.name
    qrels, denom, _, _ = judgments(corpus)
    positives = qrels.groupby(KEY).chunk_id.agg(set).to_dict()
    scores = []
    for (sid, qid, ds), (bs, bq, ls) in zip(
        candidates(dense / "candidates.parquet"),
        candidates(lexical / "candidates.parquet"),
        strict=True,
    ):
        assert (sid, qid) == (bs, bq)
        ranking = [s for s, _ in fuse_chunk_ids(ds, ls, config, k=50)]
        relevant = positives[sid, qid]
        scores.append(
            (sid, qid, *(len(relevant.intersection(ranking[:k])) / len(relevant) for k in (10, 50)))
        )
    scores = (
        pd.DataFrame(scores, columns=KEY + ["recall10", "recall50"]).set_index(KEY).sort_index()
    )
    assert scores.index.equals(denom.index)
    return scores.groupby("setting_id").mean().reset_index()


def main():
    data = pd.read_csv(HERE / "by_setting.csv")
    originals = data[data.method == "original"]
    controls = originals[
        ["corpus", "retriever", "model", "setting_id", "recall10", "recall50", "report_status"]
    ].copy()
    controls["control_provenance"] = controls.pop("report_status").map(
        {"incomplete": "rescored_partial_ranking", "published": "published"}
    )
    components = {}
    for f in SHARED.glob("runs/*/*/manifest.json"):
        run = json.loads(f.read_text())
        components[run["corpus"], run["baseline"], (run["model"] or {}).get("alias", "BM25")] = (
            f.parent
        )
    recovered = []
    for corpus in ("chemistry", "un"):
        models = set(
            originals[(originals.corpus == corpus) & (originals.retriever == "dense")].model
        )
        hybrids = originals[(originals.corpus == corpus) & (originals.retriever == "hybrid")]
        check_model = hybrids.loc[hybrids.report_status == "published", "model"].iloc[0]
        observed = json.loads(
            (components[corpus, "hybrid", check_model] / "manifest.json").read_text()
        )["fusion"]
        assert observed["method"] == "rrf"
        conf = HybridConfig(**{k: v for k, v in observed.items() if k != "method"})
        for model in [check_model] + sorted(models - set(hybrids.model)):
            result = fusion_summary(
                components[corpus, "dense", model], components[corpus, "bm25", "BM25"], conf
            )
            if model == check_model:
                expected = hybrids[hybrids.model == model].set_index("setting_id")
                ordered = result.set_index("setting_id").loc[expected.index]
                np.testing.assert_allclose(
                    ordered[["recall10", "recall50"]],
                    expected[["recall10", "recall50"]],
                    atol=1e-12,
                    rtol=0,
                )
            else:
                recovered.append(
                    result.assign(
                        corpus=corpus,
                        retriever="hybrid",
                        model=model,
                        control_provenance="offline_fusion",
                    )
                )
    assert len(recovered) == 3
    controls = pd.concat([controls, *recovered], ignore_index=True).rename(
        columns={"recall10": "full_control_recall10", "recall50": "full_control_recall50"}
    )
    controls.to_csv(HERE / "all_model_controls.csv", index=False)
    d = data[data.method != "original"].merge(
        controls, on=["corpus", "retriever", "model", "setting_id"], validate="many_to_one"
    )
    assert d.run_id.nunique() == 340
    for k in (10, 50):
        d[f"delta{k}_pp"] = 100 * (d[f"recall{k}"] - d[f"full_control_recall{k}"])
    keep = (d.method == "hyde") | (d.language == d.target_languages) | (d.regime == "mixed")
    cols = [
        "recall10",
        "recall50",
        "full_control_recall10",
        "full_control_recall50",
        "delta10_pp",
        "delta50_pp",
    ]
    dims = ["corpus", "regime", "retriever", "method"]
    s = (
        d[keep]
        .groupby(dims + ["model", "setting_id"])[cols]
        .mean()
        .groupby(level=dims + ["model"])
        .mean()
    )
    out = s.groupby(level=dims).mean()
    out["models"] = s.groupby(level=dims).size()
    out.reset_index().to_csv(HERE / "all_models_recall.csv", index=False)


if __name__ == "__main__":
    main()
