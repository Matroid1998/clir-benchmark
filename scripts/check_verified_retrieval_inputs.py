"""Verify query identity, setting eligibility and binary chunk qrels before retrieval."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from clir_benchmark.adapters import DatasetReader
from clir_benchmark.artifacts import rows
from clir_benchmark.config import VERIFIED_OVERLAP_POLICY, load_config
from clir_benchmark.util import read_json


def check(config, corpus):
    source = config.datasets[corpus].qa
    directory = config.relevance_dir(corpus)
    incoming = list(
        DatasetReader(source, None).read(
            "qas", columns=["_id", "question", "question_language", "evidence_all_languages"]
        )
    )
    queries = {q["_id"]: q for q in incoming}
    prepared = {
        q["id"]: q
        for q in rows(directory / "queries.parquet", ["id", "question", "question_language"])
    }
    if prepared.keys() != queries.keys() or any(
        (q["question"], q["question_language"])
        != (prepared[k]["question"], prepared[k]["question_language"])
        for k, q in queries.items()
    ):
        raise ValueError(f"{corpus}: query cohort differs from release")
    settings = read_json(directory / "settings.json")
    evidence = {
        k: {lang for lang, spans in q["evidence_all_languages"].items() if spans}
        for k, q in queries.items()
    }
    expected = {
        (s, qid)
        for s, setting in settings.items()
        for qid, q in queries.items()
        if q["question_language"] == setting["query_language"]
        and evidence[qid].intersection(setting["target_languages"])
    }
    eligible = {
        (r["setting_id"], r["query_id"])
        for r in rows(directory / "eligibility.parquet")
        if r["status"] == "eligible"
    }
    relations = defaultdict(set)
    for row in rows(directory / "qrels.parquet"):
        if row["relevance"] != 1:
            raise ValueError(f"{corpus}: unexpected nonbinary qrel")
        relations[row["setting_id"], row["query_id"]].add(row["chunk_id"])
    if (
        not expected
        or expected != eligible
        or expected != set(relations)
        or any(s["status"] != "ready" for s in settings.values())
    ):
        raise ValueError(
            f"{corpus}: qrels or eligibility disagree with published evidence languages"
        )
    for qid, q in queries.items():
        base = f"{corpus}__{q['question_language']}__"
        pooled = set().union(*(relations[base + lang, qid] for lang in evidence[qid]))
        if relations[base + "mixed", qid] != pooled:
            raise ValueError(f"{qid}: mixed positives disagree with individual language judgments")
    manifest = read_json(directory / "manifest.json")
    if (
        manifest["relevance_basis"] != VERIFIED_OVERLAP_POLICY
        or manifest["qa_source"]["revision"] != source.revision
    ):
        raise ValueError(f"{corpus}: unexpected judgment policy or release")
    return {
        "corpus": corpus,
        "questions": len(queries),
        "eligible_query_settings": len(expected),
        "settings": len(settings),
        "distinct_query_setting_chunk_positives": sum(map(len, relations.values())),
        "relevance_directory": str(directory),
        "relevance_basis": manifest["relevance_basis"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    print(json.dumps(check(load_config(args.config), args.corpus)))
