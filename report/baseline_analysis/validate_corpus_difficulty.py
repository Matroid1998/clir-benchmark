#!/usr/bin/env python3
"""Validate and inventory the delivered corpus-difficulty analysis; no stage runs."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    output = root / "question_08_chemistry_difficulty"
    features = pd.read_csv(output / "query_features.csv")
    if features.groupby("corpus").size().to_dict() != {"chemistry": 197, "eurlex": 194, "un": 247}:
        raise ValueError("Unexpected question cohorts")
    if features.duplicated(["corpus", "query_id"]).any():
        raise ValueError("Duplicate feature records")
    if not features.native_hit_at_10.eq(
        features.first_native_relevant_rank_saved100.le(10).astype(float)
    ).all():
        raise ValueError("Native hit flags disagree with saved ranks")
    english = features[features.query_language.eq("en")]
    source_scores = pd.read_csv(output / "english_native_all_models.csv").set_index("model")
    for corpus, group in english.groupby("corpus"):
        if not np.isclose(
            100 * group.native_hit_at_10.mean(),
            source_scores.loc["embeddinggemma", corpus],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("English score reconstruction failed")
    expected_cases = set()
    for corpus in ["chemistry", "eurlex", "un"]:
        for hit in [0, 1]:
            pool = english[
                english.corpus.eq(corpus) & english.native_hit_at_10.eq(hit)
            ].sort_values("query_id")
            expected_cases.update(
                (corpus, q)
                for q in pool.sample(n=min(4, len(pool)), random_state=20261116).query_id
            )
    actual_cases = set()
    for path in (output / "cases").glob("*/*.json"):
        case = json.loads(path.read_text())
        actual_cases.add((case["query"]["corpus"], case["query"]["id"]))
        if not case["relevant_source"]["text"] or not path.with_suffix(".txt").exists():
            raise ValueError("Case lacks complete source text/inspection file")
    if actual_cases != expected_cases:
        raise ValueError("Case files differ from the declared seeded selection")
    pool = pd.read_csv(output / "pool_size_counterfactual.csv")
    if not pool.counterfactual_expected_hit_lower_pct.le(
        pool.counterfactual_expected_hit_upper_pct + 1e-12
    ).all():
        raise ValueError("Invalid censoring bounds")
    if not pool.counterfactual_expected_hit_lower_pct.ge(
        pool.measured_original_hit_pct - 1e-10
    ).all():
        raise ValueError("Removing competitors unexpectedly reduced expected hit rate")
    gallery = (output / "index.html").read_text()
    for link in re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", gallery):
        if not (output / link).exists():
            raise ValueError(f"Missing gallery artifact: {link}")
    extraction = json.loads((output / "extraction_validation.json").read_text())
    thinning = json.loads((output / "pool_size_counterfactual_method.json").read_text())
    for metadata in [extraction, thinning]:
        for path, expected in metadata["source_sha256"].items():
            if sha(path) != expected:
                raise ValueError(f"Source has changed since extraction: {path}")
    results = {
        "status": "passed",
        "feature_rows": len(features),
        "english_rows": len(english),
        "seeded_cases": len(actual_cases),
        "case_sampling_reconstructed": True,
        "native_english_scores_reconstructed": True,
        "gallery_links_resolve": True,
        "recorded_source_hashes_match": True,
        "counterfactual_bounds_and_monotonicity_valid": True,
        "manual_review": "findings.txt records inspected examples and separates mathematical effects, counterfactual estimates and observational explanations.",
    }
    (output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    scripts = [
        "inspect_corpus_difficulty.py",
        "explain_corpus_difficulty.py",
        "pool_size_counterfactual.py",
        "test_pool_size_counterfactual.py",
        "validate_corpus_difficulty.py",
    ]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scripts_sha256": {str(root / name): sha(root / name) for name in scripts},
        "output_sha256": {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
