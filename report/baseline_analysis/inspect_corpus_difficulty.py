#!/usr/bin/env python3
"""Inspect saved questions, relevant source texts and rankings without new retrieval.

English-only diagnostics hold original query/target language and relevance-count
definition fixed. Corpus contrasts remain observational, not causal estimates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORPORA = ["chemistry", "eurlex", "un"]
STOPWORDS = set(
    """a an and are as at be been being by can could did do does doing for from had has
have how i if in into is it its may must of on or shall should so than that the their them
then there these they this those through to under upon was were what when where which who
whom whose why will with would you your according about after all any both each following
given made make many most much only other over per such these under using used use within""".split()
)


def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def content_tokens(text):
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def normalized(text):
    return " ".join(tokens(text))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(value) for value in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(item):
            return None
        return item

    Path(path).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    output = root / "question_08_chemistry_difficulty"
    output.mkdir(parents=True, exist_ok=True)
    artifact_root = repo / "artifacts/fir-20261109-all-questions/artifacts"
    catalog_path = root / "question_01_recall_at_10/tables/report_catalog.csv"
    with catalog_path.open() as handle:
        catalog = [r for r in csv.DictReader(handle) if r["cohort"] == "selected_documents"]
    all_features, corpus_summaries, score_rows, query_texts, source_texts = [], [], [], {}, {}
    inputs = {str(catalog_path): sha(catalog_path)}
    case_rankings = {}
    for corpus in CORPORA:
        print(f"Reading {corpus} questions and saved rankings", flush=True)
        entry = next(
            r
            for r in catalog
            if r["corpus"] == corpus and r["model"] == "embeddinggemma" and r["baseline"] == "dense"
        )
        report_path = Path(entry["report_path"])
        report = json.loads(report_path.read_text())
        relevance = artifact_root / "relevance" / corpus / report["run"]["relevance_hash"]
        queries_path = relevance / "queries.parquet"
        qrels_path = relevance / "qrels.parquet"
        selection_path = relevance / "selection.json"
        queries = pq.read_table(queries_path).to_pylist()
        qrels = pq.read_table(qrels_path).to_pandas()
        selection = json.loads(selection_path.read_text())
        selection_by_query = {q: s for s in selection for q in s["selected_ids"]}
        query_map = {q["id"]: q for q in queries}
        query_texts.update({(corpus, q["id"]): q for q in queries})
        english_ids = {q["id"] for q in queries if q["question_language"] == "en"}
        native_settings = {
            q["id"]: f"{corpus}__{q['question_language']}__{q['question_language']}"
            for q in queries
        }
        qrel_sets = qrels.groupby(["setting_id", "query_id"]).document_id.agg(set).to_dict()
        native_ids = {q: qrel_sets[(setting, q)] for q, setting in native_settings.items()}
        if not all(len(v) == 1 for v in native_ids.values()):
            raise ValueError("Expected one relevant version in each native-language setting")
        positive_ids = {next(iter(v)) for v in native_ids.values()}
        ranking_path = report_path.parent / "analysis/rankings.parquet"
        rank_columns = [
            "setting_id",
            "query_id",
            "rank",
            "score",
            "relevance",
            "document_id",
            "source_id",
            "parent_id",
            "document_language",
        ]
        ranks = pq.read_table(ranking_path, columns=rank_columns).to_pandas()
        groups = {
            (sid, qid): g.sort_values("rank")
            for (sid, qid), g in ranks.groupby(["setting_id", "query_id"])
        }
        candidates = [
            p
            for p in (artifact_root / "corpus" / corpus).glob("*/manifest.json")
            if json.loads(p.read_text())["corpus_fingerprint"]
            == report["run"]["corpus_fingerprint"]
        ]
        if len(candidates) != 1:
            raise ValueError("Ambiguous source corpus")
        manifest_path = candidates[0]
        manifest = json.loads(manifest_path.read_text())
        source_path = manifest_path.parent / "sources.parquet"
        chunk_path = manifest_path.parent / "chunks.parquet"
        chunk_counts = (
            pq.read_table(chunk_path, columns=["source_unit_id"])
            .to_pandas()
            .source_unit_id.value_counts()
        )
        english_rank_rows = ranks[ranks.setting_id.eq(f"{corpus}__en__en") & ranks["rank"].le(10)]
        keep_ids = positive_ids | set(english_rank_rows.document_id)
        english_query_vocabulary = set().union(
            *(content_tokens(query_map[q]["question"]) for q in english_ids)
        )
        df = Counter()
        english_documents, corpus_doc_lengths, all_language_counts = {}, [], Counter()
        duplicate_counts = Counter()
        same_language_sizes = manifest["languages"]
        source_metadata = {}
        # Stream source texts, retaining only selected positives and English top-10 candidates.
        for batch in pq.ParquetFile(source_path).iter_batches(
            batch_size=256,
            columns=["id", "source_id", "parent_id", "language", "unit_type", "text"],
        ):
            for doc in batch.to_pylist():
                all_language_counts[doc["language"]] += 1
                if doc["id"] in keep_ids:
                    source_texts[(corpus, doc["id"])] = doc
                if doc["language"] == "en":
                    words = tokens(doc["text"])
                    lexical = set(words)
                    df.update(lexical & english_query_vocabulary)
                    norm = " ".join(words)
                    text_hash = hashlib.sha256(norm.encode()).hexdigest()
                    duplicate_counts[text_hash] += 1
                    english_documents[doc["id"]] = {
                        "words": len(words),
                        "characters": len(doc["text"]),
                        "chunks": int(chunk_counts.get(doc["id"], 0)),
                        "text_hash": text_hash,
                    }
                    corpus_doc_lengths.append(len(words))
                source_metadata[doc["id"]] = (doc["source_id"], doc["parent_id"], doc["unit_type"])
        if dict(all_language_counts) != same_language_sizes:
            raise ValueError("Source counts differ from corpus manifest")
        english_size = same_language_sizes["en"]
        per_query_path = report_path.parent / "per_query.json"
        scores = json.loads(per_query_path.read_text())
        score_lookup = {(q["setting_id"], q["query_id"]): q for q in scores}
        for q in queries:
            qid = q["id"]
            sid = native_settings[qid]
            positive_id = next(iter(native_ids[qid]))
            doc = source_texts[(corpus, positive_id)]
            ranked = groups[(sid, qid)]
            positives = ranked[ranked.relevance.gt(0)]
            first_rank = int(positives["rank"].min()) if len(positives) else 101
            mixed_sid = f"{corpus}__{q['question_language']}__mixed"
            mixed = score_lookup[(mixed_sid, qid)]
            native = score_lookup[(sid, qid)]
            if native["metrics"]["Hit@10"] != float(first_rank <= 10):
                raise ValueError("Saved ranking disagrees with saved per-query hit")
            metadata = json.loads(q["metadata_json"])
            record = {
                "corpus": corpus,
                "query_id": qid,
                "query_language": q["question_language"],
                "question": q["question"],
                "answer": q["answer"],
                "source_unit_id": positive_id,
                "generator": q["generator_model_id"],
                "persona": q["persona"],
                "quality_status": metadata.get("quality_audit_status", "unknown"),
                "alignment_status": q["evidence_alignment_eligibility"],
                "candidate_count": selection_by_query[qid]["candidate_count"],
                "total_score": q["total_score"],
                "question_characters": len(q["question"]),
                "native_source_characters": len(doc["text"]),
                "native_source_chunks": int(chunk_counts.get(positive_id, 0)),
                "native_hit_at_10": native["metrics"]["Hit@10"],
                "first_native_relevant_rank_saved100": first_rank,
                "rank_censored_beyond100": first_rank == 101,
                "mixed_recall_at_10": mixed["metrics"]["Recall@10"],
                "mixed_hit_at_10": mixed["metrics"]["Hit@10"],
                "mixed_relevant_versions": len(qrel_sets[(mixed_sid, qid)]),
            }
            if qid in english_ids:
                question_terms = content_tokens(q["question"])
                doc_terms = content_tokens(doc["text"])
                overlap = question_terms & doc_terms
                weights = {t: math.log((english_size + 1) / (df[t] + 1)) for t in question_terms}
                qwords = tokens(q["question"])
                trigrams = set(zip(qwords, qwords[1:], qwords[2:], strict=False))
                native_words = tokens(doc["text"])
                native_trigrams = set(
                    zip(native_words, native_words[1:], native_words[2:], strict=False)
                )
                rare = sorted(t for t in overlap if df[t] <= 5 and not t.isdigit())
                rare_relative = sorted(
                    t for t in overlap if df[t] / english_size <= 0.001 and not t.isdigit()
                )
                top = ranked.iloc[0]
                unjudged = ranked[ranked.relevance.eq(0)]
                top_unjudged = unjudged.iloc[0]
                record.update(
                    {
                        "question_words": len(qwords),
                        "native_source_words": len(native_words),
                        "content_word_coverage": len(overlap) / max(len(question_terms), 1),
                        "idf_weighted_coverage": sum(weights[t] for t in overlap)
                        / max(sum(weights.values()), 1e-12),
                        "question_trigram_coverage": len(trigrams & native_trigrams)
                        / max(len(trigrams), 1),
                        "rare_overlapping_terms_df_le5": len(rare),
                        "rare_terms": " | ".join(rare),
                        "rare_overlapping_terms_fraction_le001": len(rare_relative),
                        "rare_relative_terms": " | ".join(rare_relative),
                        "minimum_matching_term_document_fraction": min(
                            (df[t] / english_size for t in overlap if not t.isdigit()),
                            default=1.0,
                        ),
                        "answer_exact_normalized_substring": bool(normalized(q["answer"]))
                        and normalized(q["answer"]) in normalized(doc["text"]),
                        "native_source_duplicate_count": duplicate_counts[
                            english_documents[positive_id]["text_hash"]
                        ],
                        "top_document_id": top.document_id,
                        "top_unjudged_id": top_unjudged.document_id,
                        "top_unjudged_same_parent": top_unjudged.parent_id == doc["parent_id"],
                        "top_unjudged_source_words": english_documents[top_unjudged.document_id][
                            "words"
                        ]
                        if top_unjudged.document_id in english_documents
                        else None,
                        "top_unjudged_identical_full_text": english_documents[
                            top_unjudged.document_id
                        ]["text_hash"]
                        == english_documents[positive_id]["text_hash"],
                        "best_relevant_score_minus_best_unjudged": float(
                            positives.score.max() - top_unjudged.score
                        )
                        if len(positives)
                        else None,
                    }
                )
            all_features.append(record)
        corpus_frame = pd.DataFrame([f for f in all_features if f["corpus"] == corpus])
        english = corpus_frame[corpus_frame.query_language.eq("en")]
        corpus_summaries.append(
            {
                "corpus": corpus,
                "selected_questions": len(queries),
                "english_questions": len(english),
                "english_corpus_versions": english_size,
                "searched_corpus_versions": sum(
                    n
                    for lang, n in manifest["languages"].items()
                    if lang in {"de", "en", "es", "fr", "zh"}
                ),
                "english_corpus_median_words": float(np.median(corpus_doc_lengths)),
                "english_selected_source_median_words": float(english.native_source_words.median()),
                "english_selected_source_median_chunks": float(
                    english.native_source_chunks.median()
                ),
                "english_question_median_words": float(english.question_words.median()),
                "english_content_coverage_median": float(english.content_word_coverage.median()),
                "english_idf_coverage_median": float(english.idf_weighted_coverage.median()),
                "english_trigram_coverage_median": float(
                    english.question_trigram_coverage.median()
                ),
                "english_questions_with_rare_matching_term_pct": 100
                * float(english.rare_overlapping_terms_df_le5.gt(0).mean()),
                "english_questions_with_relative_rare_term_pct": 100
                * float(english.rare_overlapping_terms_fraction_le001.gt(0).mean()),
                "english_answer_substring_pct": 100
                * float(english.answer_exact_normalized_substring.mean()),
                "english_native_hit_pct": 100 * float(english.native_hit_at_10.mean()),
                "english_clear_native_hit_pct": 100
                * float(english[english.quality_status.eq("clear")].native_hit_at_10.mean()),
                "english_selected_duplicate_full_text_pct": 100
                * float(english.native_source_duplicate_count.gt(1).mean()),
                "english_top_unjudged_identical_full_text_pct": 100
                * float(english.top_unjudged_identical_full_text.mean()),
                "clear_question_pct": 100 * float(corpus_frame.quality_status.eq("clear").mean()),
                "alignment_eligible_pct": 100
                * float(corpus_frame.alignment_status.eq("eligible").mean()),
                "median_candidate_pool": float(corpus_frame.candidate_count.median()),
                "mixed_relevant_version_distribution": corpus_frame.mixed_relevant_versions.value_counts()
                .sort_index()
                .to_dict(),
            }
        )
        # Read every model's dense scores to distinguish corpus effects from one retriever.
        for catalog_entry in [
            r for r in catalog if r["corpus"] == corpus and r["baseline"] == "dense"
        ]:
            other_path = Path(catalog_entry["report_path"])
            other_report = json.loads(other_path.read_text())
            other_scores = json.loads((other_path.parent / "per_query.json").read_text())
            for scope, predicate in [
                ("all_questions", lambda q: True),
                ("clear_only", lambda q: q["quality_audit_status"] == "clear"),
                ("english_original", lambda q: q["query_language"] == "en"),
            ]:
                part = [q for q in other_scores if predicate(q)]
                for regime in ["mixed", "monolingual", "cross_language"]:
                    part_regime = [q for q in part if q["regime"] == regime]
                    setting_values = defaultdict(list)
                    for q in part_regime:
                        setting_values[q["setting_id"]].append(q["metrics"])
                    score_rows.append(
                        {
                            "corpus": corpus,
                            "model": catalog_entry["model"],
                            "scope": scope,
                            "regime": regime,
                            "unique_queries": len({q["query_id"] for q in part_regime}),
                            "nonempty_settings": len(setting_values),
                            **{
                                metric: 100
                                * float(
                                    np.mean(
                                        [
                                            np.mean([v[metric] for v in group])
                                            for group in setting_values.values()
                                        ]
                                    )
                                )
                                for metric in ["Recall@10", "Hit@10"]
                            },
                        }
                    )
            if other_report["run"]["relevance_hash"] != report["run"]["relevance_hash"]:
                raise ValueError("Dense models use different qrels")
        for p in [
            report_path,
            per_query_path,
            queries_path,
            qrels_path,
            selection_path,
            manifest_path,
        ]:
            inputs[str(p)] = sha(p)
        # Stratified seeded sample chosen before reading text: four English hits and four misses.
        chosen = []
        for hit in [True, False]:
            pool = english[english.native_hit_at_10.eq(float(hit))].sort_values("query_id")
            sampled = pool.sample(n=min(4, len(pool)), random_state=20261116)
            for row in sampled.to_dict("records"):
                row["selection_rule"] = (
                    f"Seed 20261116; sample up to four English-native {'hits' if hit else 'misses'} per corpus"
                )
                chosen.append(row)
        case_ids = [r["query_id"] for r in chosen]
        pd.DataFrame(
            [
                {
                    "term": term,
                    "english_document_frequency": count,
                    "english_document_fraction": count / english_size,
                }
                for term, count in sorted(df.items())
            ]
        ).to_csv(output / f"english_query_term_frequencies_{corpus}.csv", index=False)
        case_rankings[corpus] = pq.read_table(
            ranking_path,
            filters=[("setting_id", "=", f"{corpus}__en__en"), ("query_id", "in", case_ids)],
        ).to_pandas()
        case_dir = output / "cases" / corpus
        case_dir.mkdir(parents=True, exist_ok=True)
        for row in chosen:
            q = query_map[row["query_id"]]
            doc = source_texts[(corpus, row["source_unit_id"])]
            ranking = case_rankings[corpus]
            ranking = ranking[ranking.query_id.eq(row["query_id"])].sort_values("rank")
            retained = ranking[
                (ranking["rank"] <= 3)
                | (
                    (ranking.relevance > 0)
                    & (ranking["rank"] == row["first_native_relevant_rank_saved100"])
                )
            ]
            case = {
                "query": q,
                "features": row,
                "relevant_source": doc,
                "saved_top3_and_first_positive": retained.to_dict("records"),
                "first_relevant_rank_censoring": ">100 means absent from saved top100; exact rank unknown",
            }
            # Arrow lists may arrive as arrays in pandas cells.
            case = json.loads(
                json.dumps(case, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
            )
            save_json(case_dir / f"{q['id']}.json", case)
            text = [
                f"{corpus} / EmbeddingGemma / original English query / English target",
                row["selection_rule"],
                f"Query ID: {q['id']}",
                f"First relevant rank (saved top100): {row['first_native_relevant_rank_saved100']}",
                f"Quality: {row['quality_status']}; alignment: {row['alignment_status']}",
                "\nQUESTION\n" + q["question"],
                "\nSAVED ANSWER\n" + q["answer"],
                "\nCOMPLETE RELEVANT SOURCE " + doc["id"] + "\n" + doc["text"],
            ]
            for ranked in case["saved_top3_and_first_positive"]:
                text += [
                    f"\nRETRIEVED RANK {ranked['rank']} / relevance={ranked['relevance']} / {ranked['document_id']} / score={ranked['score']:.8f}",
                    ranked["text"],
                ]
            (case_dir / f"{q['id']}.txt").write_text("\n".join(text) + "\n")
        print(
            f"{corpus}: inspected features for {len(corpus_frame)} questions; exported {len(chosen)} text cases",
            flush=True,
        )
    features = pd.DataFrame(all_features)
    features.to_csv(output / "query_features.csv", index=False)
    pd.DataFrame(corpus_summaries).to_json(
        output / "corpus_factors.json", orient="records", indent=2
    )
    scalar = pd.DataFrame(corpus_summaries).drop(columns=["mixed_relevant_version_distribution"])
    scalar.to_csv(output / "corpus_factors.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "dense_quality_and_language_controls.csv", index=False)
    # Preserve all source questions/positive texts for follow-up inspection, not just selected examples.
    save_json(
        output / "selected_questions_and_native_sources.json",
        [
            {
                "query": q,
                "native_source": source_texts[
                    (
                        corpus,
                        next(
                            iter(
                                {
                                    f["source_unit_id"]
                                    for f in all_features
                                    if f["corpus"] == corpus and f["query_id"] == qid
                                }
                            )
                        ),
                    )
                ],
            }
            for (corpus, qid), q in query_texts.items()
        ],
    )
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    save_json(
        output / "extraction_validation.json",
        {
            "status": "passed",
            "query_count": len(features),
            "english_query_count": int(features.query_language.eq("en").sum()),
            "source_sha256": inputs,
            "methods": {
                "ranking": "Saved original-query EmbeddingGemma rankings; >100 is censored. No new retrieval.",
                "lexical": "English original questions and same-language relevant units only. NFKC lowercase alphanumeric tokens; fixed stopword list; corpus document-frequency computed on English units.",
                "rare_terms": "Non-numeric overlapping content tokens appearing in at most five English source units; descriptive diagnostic, not an independent retrieval score.",
                "sampling": "Seed 20261116; four English-native hits and four misses per corpus, sampled before reading texts. Not a prevalence-weighted sample.",
            },
        },
    )
    print(scalar.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
