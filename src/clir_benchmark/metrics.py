from __future__ import annotations

import math
from collections import defaultdict

from .errors import ValidationError

CUTOFFS = (1, 5, 10, 20, 100)


def score_query(
    ranking: list[str], relevant: dict[str, set[str]], required_sources: set[str]
) -> dict[str, float]:
    if not relevant or not required_sources:
        raise ValidationError(
            "Metrics require an eligible query with relevant chunks and required sources"
        )
    if len(ranking) != len(set(ranking)):
        raise ValidationError("Ranking contains duplicate chunk IDs")
    supported = set().union(*relevant.values())
    if supported != required_sources:
        raise ValidationError("Qrels do not cover exactly the query's indispensable sources")
    result = {}
    for k in CUTOFFS:
        positives = [item for item in ranking[:k] if item in relevant]
        sources = set().union(*(relevant[item] for item in positives)) if positives else set()
        result[f"Recall@{k}"] = len(positives) / len(relevant)
        result[f"Hit@{k}"] = float(bool(positives))
        result[f"SourceRecall@{k}"] = len(sources) / len(required_sources)
        result[f"AllSourceSuccess@{k}"] = float(sources == required_sources)
    hits = [index for index, chunk_id in enumerate(ranking[:10], 1) if chunk_id in relevant]
    result["MRR@10"] = 1.0 / hits[0] if hits else 0.0
    dcg = sum(1 / math.log2(index + 1) for index in hits)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(10, len(relevant)) + 1))
    result["nDCG@10"] = dcg / ideal
    return result


def aggregate(per_query: list[dict]) -> dict:
    dimensions = {
        "setting": ("corpus", "regime", "setting_id", "query_language", "target_languages"),
        "generator": ("corpus", "regime", "generator"),
        "persona": ("corpus", "regime", "persona"),
        "source_count": ("corpus", "regime", "source_count"),
    }
    result = {}
    for label, keys in dimensions.items():
        groups = defaultdict(list)
        for row in per_query:
            groups[tuple(row[key] for key in keys)].append(row["metrics"])
        result[label] = [
            {
                **dict(zip(keys, key, strict=True)),
                "query_count": len(values),
                "metrics": {
                    name: sum(value[name] for value in values) / len(values) for name in values[0]
                },
            }
            for key, values in sorted(groups.items())
        ]
    macros = defaultdict(list)
    for row in result["setting"]:
        macros[(row["corpus"], row["regime"])].append(row["metrics"])
    result["macro_by_regime"] = [
        {
            "corpus": corpus,
            "regime": regime,
            "setting_count": len(values),
            "metrics": {
                name: sum(value[name] for value in values) / len(values) for name in values[0]
            },
        }
        for (corpus, regime), values in sorted(macros.items())
    ]
    return result
