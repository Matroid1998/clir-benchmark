#!/usr/bin/env python3
"""Expected English Hit@10 after uniform corpus thinning, from saved ranks only.

Keep every selected English question's relevant unit in one common pool, and
uniformly sample the other units down to chemistry's English pool size. Scores
and known-source qrels stay fixed. This is a counterfactual expectation, not a
newly measured benchmark run. Saved top100 censoring yields lower/upper bounds.
"""

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import hypergeom


def probability_hit(total, forced, retained, random_above, forced_above, k=10):
    if not (
        1 <= forced <= retained <= total
        and 0 <= forced_above < forced
        and 0 <= random_above <= total - forced
        and k >= 1
    ):
        raise ValueError("Invalid pool or rank counts")
    if retained == forced:
        return float(forced_above < k)
    return float(
        hypergeom.cdf(k - 1 - forced_above, total - forced, random_above, retained - forced)
    )


def main():
    root = Path(__file__).resolve().parent
    output = root / "question_08_chemistry_difficulty"
    feature_path = output / "query_features.csv"
    factors_path = output / "corpus_factors.csv"
    features = pd.read_csv(feature_path)
    factors = pd.read_csv(factors_path).set_index("corpus")
    target_size = int(factors.loc["chemistry", "english_corpus_versions"])
    with (root / "question_01_recall_at_10/tables/report_catalog.csv").open() as handle:
        catalog = list(csv.DictReader(handle))
    rows, summary, inputs = (
        [],
        [],
        {str(feature_path): hashlib.sha256(feature_path.read_bytes()).hexdigest()},
    )
    for corpus in ["chemistry", "eurlex", "un"]:
        questions = features[features.corpus.eq(corpus) & features.query_language.eq("en")]
        forced_ids = set(questions.source_unit_id)
        if len(forced_ids) != len(questions):
            raise ValueError("Expected distinct evaluation source units")
        total = int(factors.loc[corpus, "english_corpus_versions"])
        entry = next(
            r
            for r in catalog
            if r["cohort"] == "selected_documents"
            and r["baseline"] == "dense"
            and r["model"] == "embeddinggemma"
            and r["corpus"] == corpus
        )
        path = Path(entry["report_path"]).parent / "analysis/rankings.parquet"
        ranks = pq.read_table(
            path,
            columns=["query_id", "rank", "document_id"],
            filters=[("setting_id", "=", f"{corpus}__en__en")],
        ).to_pandas()
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        group_map = dict(tuple(ranks.groupby("query_id")))
        for row in questions.itertuples():
            # 101 stands for the least possible true rank when absent from saved100.
            rank = row.first_native_relevant_rank_saved100
            group = group_map[row.query_id]
            forced_above = int((group["rank"].lt(rank) & group.document_id.isin(forced_ids)).sum())
            upper = probability_hit(
                total, len(forced_ids), target_size, rank - 1 - forced_above, forced_above
            )
            lower = 0.0 if row.rank_censored_beyond100 else upper
            rows.append(
                {
                    "corpus": corpus,
                    "query_id": row.query_id,
                    "original_rank_or_censored101": rank,
                    "forced_units_above": forced_above,
                    "expected_hit_lower": lower,
                    "expected_hit_upper": upper,
                }
            )
        part = pd.DataFrame([r for r in rows if r["corpus"] == corpus])
        summary.append(
            {
                "corpus": corpus,
                "original_pool_versions": total,
                "retained_pool_versions": target_size,
                "forced_evaluation_source_units": len(forced_ids),
                "english_questions": len(questions),
                "measured_original_hit_pct": 100 * questions.native_hit_at_10.mean(),
                "counterfactual_expected_hit_lower_pct": 100 * part.expected_hit_lower.mean(),
                "counterfactual_expected_hit_upper_pct": 100 * part.expected_hit_upper.mean(),
                "questions_censored_beyond100": int(questions.rank_censored_beyond100.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(output / "pool_size_counterfactual_by_query.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "pool_size_counterfactual.csv", index=False)
    inputs[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "pool_size_counterfactual_method.json").write_text(
        json.dumps(
            {
                "method": __doc__,
                "source_sha256": inputs,
                "probability": "X ~ Hypergeometric(N - forced, random_higher_ranked, retained - forced); hit if forced_higher_ranked + X <= 9",
                "censoring": "If original rank >100, lower=0; upper uses rank101 and known forced units above. Additional worse-ranked candidates cannot increase hit probability.",
                "limitations": "Known-source relevance only; uniformly thinned shared pool conditional on all evaluation sources; fixed original scores. This is not a measured rerun, hard-negative-matched pool or a causal decomposition of corpus differences.",
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
