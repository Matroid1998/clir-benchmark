#!/usr/bin/env python3
"""Deterministically select six illustrative cases from saved results only.

Writes exclusively round_04/cases/. No generation, model loading, or retrieval.
Human/model analyst interpretation is kept separately in analyst_notes.txt and
is not overwritten by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUT = ROOT / "round_04" / "cases"
SEED = 42
TOLERANCE = 1e-12
METRICS = ["ndcg10", "hit10", "recall10", "mrr10", "relevant_document_count"]
RANK_COLUMNS = ["rank", "score", "relevance", "required_source_ids", "source_unit_id",
                "source_id", "parent_id", "document_language", "document_id",
                "winning_chunk_id", "text", "start", "end"]


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(x) for x in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def paired(frame, corpus, model, left, right):
    base = frame[frame.corpus.eq(corpus) & frame.model.eq(model) & frame.regime.eq("mixed")]
    a = base[base.variant.eq(left)].set_index("query_id")
    b = base[base.variant.eq(right)].set_index("query_id")
    assert a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index)
    b = b.reindex(a.index)
    assert (a.setting_id == b.setting_id).all()
    assert (a.relevant_document_count == b.relevant_document_count).all()
    result = pd.DataFrame(index=a.index)
    for field in METRICS:
        result[f"left_{field}"] = a[field]
        result[f"right_{field}"] = b[field]
    result["quality_audit_status"] = a.quality_audit_status
    assert (a.quality_audit_status == b.quality_audit_status).all()
    result["query_language"] = a.query_language
    result["ndcg_delta"] = a.ndcg10 - b.ndcg10
    result["left_versions10"] = np.rint(a.recall10 * a.relevant_document_count).astype(int)
    result["right_versions10"] = np.rint(b.recall10 * b.relevant_document_count).astype(int)
    result["version_delta"] = result.left_versions10 - result.right_versions10
    return result.reset_index()


def select(pool, case_id, rule, sort_columns=None, ascending=None, seeded=False):
    all_count = len(pool)
    clear = pool[pool.quality_audit_status.eq("clear")]
    restricted = len(clear) > 0
    eligible = (clear if restricted else pool).copy()
    assert len(eligible), case_id
    if seeded:
        eligible = eligible.sort_values("query_id", kind="stable")
        position = int(np.random.default_rng(SEED).integers(len(eligible)))
        chosen = eligible.iloc[position]
    else:
        eligible = eligible.sort_values([*(sort_columns or []), "query_id"],
                                        ascending=[*(ascending or []), True], kind="stable")
        position = 0
        chosen = eligible.iloc[0]
    audit = {"case_id": case_id, "selection_rule": rule,
             "original_language_restriction": None,
             "eligible_before_quality_restriction": all_count,
             "eligible_clear_quality": len(clear),
             "clear_quality_restriction_applied": restricted,
             "fallback_if_no_clear": "Use all quality statuses, retaining the same rule.",
             "seed": SEED if seeded else None, "selected_position": position,
             "candidate_order": eligible.query_id.tolist(),
             "selected_query_id": chosen.query_id,
             "selected_comparison": chosen.to_dict()}
    eligible.to_csv(OUT / f"{case_id}_candidate_pool.csv", index=False)
    return audit


def extract_case(audit, corpus, model, variants, left, right, per_query, queries, generation, qrels):
    query_id = audit["selected_query_id"]
    selected = queries[(queries.corpus == corpus) & (queries.query_id == query_id)]
    assert len(selected) == 1
    query = selected.iloc[0].to_dict()
    metadata = json.loads(query["metadata_json"])
    record = {"selection": audit, "corpus": corpus, "model": model,
              "comparison": {"left": left, "right": right, "setting": "mixed"},
              "selected_query": query, "parsed_query_metadata": metadata, "variants": {}}
    for variant in variants:
        subset = per_query[(per_query.corpus == corpus) & (per_query.model == model) &
                           (per_query.regime == "mixed") & (per_query.query_id == query_id) &
                           (per_query.variant == variant)]
        assert len(subset) == 1, (query_id, variant)
        row = subset.iloc[0]
        rank_path = PROJECT / row.report_path / "analysis" / "rankings.parquet"
        ranks = pd.read_parquet(rank_path, columns=RANK_COLUMNS,
                                filters=[("query_id", "==", query_id),
                                         ("setting_id", "==", row.setting_id)]).sort_values("rank")
        assert len(ranks) == 100 and ranks["rank"].tolist() == list(range(1, 101))
        relevant = ranks[ranks.relevance.gt(0)]
        top = ranks[ranks["rank"].le(10)]
        versions = int(top.relevance.gt(0).sum())
        assert versions == round(row.recall10 * row.relevant_document_count)
        assert bool(versions) == bool(row.hit10)
        ideal = sum(1 / np.log2(i + 1) for i in range(1, min(10, int(row.relevant_document_count)) + 1))
        recomputed = sum(1 / np.log2(r + 1) for r in top.loc[top.relevance.gt(0), "rank"]) / ideal
        assert abs(recomputed - row.ndcg10) < TOLERANCE
        if row.method == "original":
            generated = None
            input_text = query["question"]
        else:
            gen = generation[(generation.corpus == corpus) & (generation.query_id == query_id) &
                             (generation.method == row.method) &
                             (generation.output_language == row.output_language)]
            assert len(gen) == 1
            generated = gen.iloc[0].to_dict()
            input_text = generated["text"]
            assert generated["original_text"] == query["question"]
        stats = {"first_relevant_rank_within_saved100": int(relevant["rank"].iloc[0]) if len(relevant) else None,
                 "first_relevant_language_within_saved100": relevant.document_language.iloc[0] if len(relevant) else None,
                 "relevant_ranks_within10": top.loc[top.relevance.gt(0), "rank"].tolist(),
                 "relevant_languages_within10": top.loc[top.relevance.gt(0), "document_language"].tolist(),
                 "relevant_versions10": versions,
                 "relevant_ranks_within100": relevant["rank"].tolist(),
                 "top10_language_counts": top.document_language.value_counts().sort_index().to_dict()}
        # Retain all saved ranking identities/scores, but text only for top ten
        # and any accepted positive within 100; this bounds duplication.
        retained = ranks.copy()
        retained.loc[~(retained["rank"].le(10) | retained.relevance.gt(0)), "text"] = None
        record["variants"][variant] = {"input_text": input_text, "generation_record": generated,
                                        "per_query_record": row.to_dict(), "rank_statistics": stats,
                                        "rankings_source": str(rank_path.relative_to(PROJECT)),
                                        "saved_rankings100": retained.to_dict("records")}
    setting_id = subset.iloc[0].setting_id
    record["qrels"] = qrels[(qrels.corpus == corpus) & (qrels.query_id == query_id) &
                             (qrels.setting_id == setting_id)].to_dict("records")
    dump(OUT / f"{audit['case_id']}.json", record)
    lines = [audit["case_id"], "Rule: " + audit["selection_rule"],
             f"Corpus/model: {corpus}/{model}", f"Query ID: {query_id}",
             f"Original language: {query['query_language']}",
             f"Quality: {metadata.get('quality_audit_status')}",
             f"Canonical source: {query['canonical_source_id']}",
             f"Source unit: {query['source_unit_id']}",
             f"Query generator: {query['generator_model_id']}",
             "Provenance: " + json.dumps(metadata.get("provenance"), ensure_ascii=False),
             "", "ORIGINAL QUESTION", query["question"], "", "SAVED ANSWER (not provided to generation)",
             str(query["answer"]), "", "SAVED EVIDENCE (not provided to generation)", str(query["evidence"])]
    for variant, details in record["variants"].items():
        lines += ["", variant, details["input_text"],
                  "Metrics: " + json.dumps({k: plain(details["per_query_record"][k]) for k in METRICS}),
                  "Ranks: " + json.dumps(plain(details["rank_statistics"]), ensure_ascii=False)]
    (OUT / f"{audit['case_id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {name: ROOT / "data" / f"{name}.parquet" for name in
              ["per_query", "selected_queries", "generation", "qrels"]}
    per_query, queries, generation, qrels = [pd.read_parquet(inputs[name]) for name in inputs]
    un = paired(per_query, "un", "e5-large-instruct", "translate_de", "translate_en")
    both = un.left_hit10.eq(1) & un.right_hit10.eq(1)
    chosen = [
        select(un[un.left_hit10.eq(0) & un.right_hit10.eq(1)], "un_e5_source_loss",
               "Among Ten-hit/Tde-miss at 10, minimize Tde-minus-Ten nDCG@10; break ties by query ID.",
               ["ndcg_delta"], [True]),
        select(un[un.left_hit10.eq(1) & un.right_hit10.eq(0)], "un_e5_source_gain",
               "Among Tde-hit/Ten-miss at 10, maximize Tde-minus-Ten nDCG@10; break ties by query ID.",
               ["ndcg_delta"], [False]),
        select(un[both], "un_e5_version_gain_both_hit",
               "Among both-hit at 10, maximize Tde-minus-Ten relevant-version count, then nDCG@10; break ties by query ID.",
               ["version_delta", "ndcg_delta"], [False, False]),
        select(un[both & un.ndcg_delta.abs().le(TOLERANCE) & un.version_delta.eq(0)],
               "un_e5_stable_control",
               "Among both-hit at 10 with equal version count and absolute nDCG delta <=1e-12, sort query IDs and select using NumPy default_rng(42).integers(n).",
               seeded=True),
    ]
    records = [extract_case(a, "un", "e5-large-instruct", ["original", "translate_en", "translate_de"],
                            "translate_de", "translate_en", per_query, queries, generation, qrels) for a in chosen]
    for corpus, case_id, ascending in [("eurlex", "eurlex_bge_hyde_gain", False),
                                       ("chemistry", "chemistry_bge_hyde_loss", True)]:
        pool = paired(per_query, corpus, "bge-m3", "hyde_en", "translate_en")
        direction = "minimize" if ascending else "maximize"
        audit = select(pool, case_id, f"For BGE-M3 mixed search, {direction} English-HyDE minus English-translation nDCG@10; break ties by query ID.",
                       ["ndcg_delta"], [ascending])
        records.append(extract_case(audit, corpus, "bge-m3", ["original", "translate_en", "hyde_en"],
                                    "hyde_en", "translate_en", per_query, queries, generation, qrels))
    assert len({(r["corpus"], r["selection"]["selected_query_id"]) for r in records}) == 6
    manifest = {"scope": "Offline illustrative cases; not representative samples or causal tests.",
                "selection_precedes_text_inspection": True, "seed": SEED, "equality_tolerance": TOLERANCE,
                "quality_rule": "Restrict each rule's pool to clear-quality queries if nonempty; otherwise disclose fallback.",
                "original_language_restriction": None,
                "extreme_case_tie_rule": "Ascending full query ID after requested metric ordering.",
                "stable_control_rng": "numpy.random.default_rng(42).integers(len(sorted_pool))",
                "text_retention": "Complete query, generation, saved answer, evidence and provenance. Top100 rank identities/scores; winning-chunk text for top10 and accepted positives within100.",
                "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cases": [r["selection"] for r in records]}
    dump(OUT / "selection_manifest.json", manifest)
    index = []
    for record in records:
        a = record["selection"]
        index.append({"case_id": a["case_id"], "corpus": record["corpus"], "model": record["model"],
                      "query_id": a["selected_query_id"], "source_id": record["selected_query"]["canonical_source_id"],
                      "query_language": record["selected_query"]["query_language"],
                      "clear_only": a["clear_quality_restriction_applied"],
                      "candidate_count": len(a["candidate_order"]),
                      "ndcg_delta": a["selected_comparison"]["ndcg_delta"],
                      "version_delta": a["selected_comparison"]["version_delta"]})
    pd.DataFrame(index).to_csv(OUT / "case_index.csv", index=False)
    (OUT / "selection_rules.txt").write_text(
        "Case selection is deterministic and was specified before reading selected texts.\n"
        "All cases use mixed search with identical qrels and source questions within each pair.\n"
        "Ten/Tde denote English/German query translation, not the original query language.\n"
        "Clear-quality restriction is applied separately per rule when possible; no language filter.\n"
        "No case establishes a population frequency, causal mechanism, translation error, or hallucination.\n"
        "Saved answers/evidence are supplied here for analyst inspection only; generation did not receive them.\n"
        "BGE-M3 has identical empty query/document prefixes. HyDE still changes text type and length.\n"
        "The candidate-pool CSVs and selection_manifest.json retain all eligibility/order decisions.\n"
        "All .txt/.json case files are generated; analyst_notes.txt is separately authored after inspection.\n",
        encoding="utf-8")
    print(pd.DataFrame(index).to_string(index=False))


if __name__ == "__main__":
    main()
