from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .config import Config
from .errors import ValidationError
from .pipeline import evaluate
from .translation import nullable_sum
from .util import digest, new_directory, read_json, write_json


def _label_document_comparison(config: Config, report: dict, query_rows: list[dict]) -> None:
    if config.retrieval_unit == "document":
        report.update(retrieval_unit="document", document_identity="source_unit_id")
        for row in [*report["settings"], *query_rows]:
            row["retrieval_unit"] = "document"


def compare_translations(
    config: Config,
    corpus: str,
    alias: str | None,
    *,
    baseline: str,
    languages: tuple[str, ...],
) -> Path:
    """Compare only identical query/setting cohorts; never execute retrieval or translation."""
    if baseline not in {"dense", "bm25", "hybrid"}:
        raise ValidationError(
            "compare-translations takes an untranslated baseline: dense, bm25, or hybrid"
        )
    original_dir = evaluate(config, corpus, alias, baseline=baseline)
    original_report = read_json(original_dir / "report.json")
    original = {
        (row["setting_id"], row["query_id"]): row
        for row in read_json(original_dir / "per_query.json")
    }
    summaries = []
    query_rows = []
    provenance = []
    for language in languages:
        translated_dir = evaluate(
            config, corpus, alias, baseline=f"translate-{baseline}", translation_language=language
        )
        translated = {
            (row["setting_id"], row["query_id"]): row
            for row in read_json(translated_dir / "per_query.json")
        }
        if original.keys() != translated.keys():
            raise ValidationError(
                "Original and translated evaluation cohorts differ; cannot compare"
            )
        report = read_json(translated_dir / "report.json")
        for name in ("relevance_hash", "corpus_fingerprint", "query_fingerprint", "model"):
            if original_report["run"].get(name) != report["run"].get(name):
                raise ValidationError(f"Original and translated {name} differ; cannot compare")
        if original_report["run"].get("retrieval_unit", "chunk") != report["run"].get(
            "retrieval_unit", "chunk"
        ):
            raise ValidationError("Original and translated retrieval units differ; cannot compare")
        provenance.append(
            {
                "translation_language": language,
                "report": str(translated_dir),
                "translation_summary": report["translation_summary"],
            }
        )
        grouped = defaultdict(list)
        for key, base in original.items():
            variant = translated[key]
            original_seconds = nullable_sum(
                base.get(name)
                for name in ("retrieval_seconds_amortized", "query_encoding_seconds_amortized")
            )
            translated_seconds = nullable_sum(
                variant.get(name)
                for name in (
                    "translation_seconds",
                    "retrieval_seconds_amortized",
                    "query_encoding_seconds_amortized",
                )
            )
            row = {
                "baseline": baseline,
                "model": alias,
                "translation_language": language,
                "setting_id": key[0],
                "query_id": key[1],
                "query_language": base["query_language"],
                "target_languages": base["target_languages"],
                "translation_status": variant["translation_status"],
                "translation_cost_usd": variant["translation_cost_usd"],
                "translation_seconds": variant["translation_seconds"],
                "original_component_seconds": original_seconds,
                "translated_component_seconds": translated_seconds,
                "added_component_seconds": translated_seconds - original_seconds
                if translated_seconds is not None and original_seconds is not None
                else None,
                "original_metrics": base["metrics"],
                "translated_metrics": variant["metrics"],
                "metric_deltas": {
                    name: variant["metrics"][name] - value
                    for name, value in base["metrics"].items()
                },
            }
            query_rows.append(row)
            grouped[key[0]].append(row)
        for setting_id, members in sorted(grouped.items()):
            summary = {
                "baseline": baseline,
                "model": alias,
                "translation_language": language,
                "setting_id": setting_id,
                "query_language": members[0]["query_language"],
                "target_languages": members[0]["target_languages"],
                "query_count": len(members),
            }
            for name in (
                "translation_cost_usd",
                "translation_seconds",
                "original_component_seconds",
                "translated_component_seconds",
                "added_component_seconds",
            ):
                total = nullable_sum(row[name] for row in members)
                summary[f"mean_{name}"] = total / len(members) if total is not None else None
            for prefix in ("original_metrics", "translated_metrics", "metric_deltas"):
                for name in members[0][prefix]:
                    summary[f"{prefix}.{name}"] = sum(row[prefix][name] for row in members) / len(
                        members
                    )
            summaries.append(summary)
    report = {
        "schema": 1,
        "corpus": corpus,
        "baseline": baseline,
        "model": alias,
        "original_report": str(original_dir),
        "translations": provenance,
        "settings": summaries,
        "notes": [
            "Metric deltas are translated minus original, in absolute metric units, on identical query/setting cohorts.",
            "Component seconds sum measured API attempts, amortized offline query embedding build time and amortized retrieval batch time; they are not online latency measurements.",
            "Translation costs are estimates using configured USD token rates; unknown usage/rates remain null. Local encoding/retrieval compute is not priced.",
            "Translation summaries count unique queries once per language. Do not sum costs across settings or retrievers that reuse the same requests.",
            "Existing historical runs without timing instrumentation yield null timing comparisons; use a fresh output directory to measure them.",
        ],
    }
    _label_document_comparison(config, report, query_rows)
    destination = (
        config.output_dir
        / "comparisons"
        / corpus
        / digest({"report": report, "queries": query_rows})
    )
    if destination.exists():
        return destination
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", report)
        write_json(temporary / "per_query.json", query_rows)
        with (temporary / "settings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(summaries[0]) if summaries else ["setting_id"]
            )
            writer.writeheader()
            writer.writerows(summaries)
    return destination


def _difference(value: float | None, control: float | None) -> float | None:
    return value - control if value is not None and control is not None else None


def compare_adore(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    baseline: str = "adore-bm25",
) -> Path:
    """Compare saved ADORE results to the same untranslated retriever, without live work."""
    if baseline not in {"adore-bm25", "adore-dense", "adore-hybrid"}:
        raise ValidationError("compare-adore requires an adore-* baseline")
    control = baseline.removeprefix("adore-")
    variant_dir = evaluate(config, corpus, alias, baseline=baseline)
    control_dir = evaluate(config, corpus, alias, baseline=control)
    variant_report = read_json(variant_dir / "report.json")
    control_report = read_json(control_dir / "report.json")

    def cohort(directory):
        records = read_json(directory / "per_query.json")
        indexed = {(row["setting_id"], row["query_id"]): row for row in records}
        if len(records) != len(indexed):
            raise ValidationError("Duplicate query/setting in ADORE comparison cohort")
        return indexed

    variants, originals = cohort(variant_dir), cohort(control_dir)
    if variants.keys() != originals.keys():
        raise ValidationError("ADORE and control evaluation cohorts differ; cannot compare")
    for name in (
        "relevance_hash",
        "corpus_fingerprint",
        "query_fingerprint",
        "model",
        "top_k",
        "bm25",
    ):
        if variant_report["run"][name] != control_report["run"][name]:
            raise ValidationError(f"ADORE and control {name} differ; cannot compare")
    if control == "hybrid":
        control_fusion = {
            key: value for key, value in control_report["run"]["fusion"].items() if key != "method"
        }
        if variant_report["run"]["fusion"] != control_fusion:
            raise ValidationError("ADORE and control fusion settings differ; cannot compare")
    grouped = defaultdict(list)
    query_rows = []
    for key, variant in variants.items():
        original = originals[key]
        for name in ("query_language", "target_languages", "source_count", "regime"):
            if variant[name] != original[name]:
                raise ValidationError(f"ADORE and control {name} differ for {key}")
        if variant["metrics"].keys() != original["metrics"].keys():
            raise ValidationError("ADORE and control metric definitions differ")
        original_total = nullable_sum(
            original.get(name)
            for name in ("query_encoding_seconds_amortized", "retrieval_seconds_amortized")
        )
        row = {
            "baseline": baseline,
            "control": control,
            "model": alias,
            "generation_model": variant_report["generation_model"],
            "setting_id": key[0],
            "query_id": key[1],
            "query_language": variant["query_language"],
            "target_languages": variant["target_languages"],
            "rounds": variant["rounds"],
            "stop_reason": variant["stop_reason"],
            "adore_generation_cost_usd": variant["generation_cost_usd"],
            "control_generation_cost_usd": 0.0,
            "added_generation_cost_usd": variant["generation_cost_usd"],
            "adore_generation_seconds": variant["generation_seconds"],
            "adore_query_encoding_seconds": variant["query_encoding_seconds"],
            "adore_retrieval_seconds": variant["retrieval_seconds"],
            "control_encoding_seconds_amortized": original.get("query_encoding_seconds_amortized"),
            "control_retrieval_seconds_amortized": original.get("retrieval_seconds_amortized"),
            "adore_component_seconds": variant["component_seconds"],
            "control_component_seconds": original_total,
            "added_component_seconds": _difference(variant["component_seconds"], original_total),
            "adore_metrics": variant["metrics"],
            "control_metrics": original["metrics"],
            "metric_deltas": {
                name: value - original["metrics"][name]
                for name, value in variant["metrics"].items()
            },
        }
        query_rows.append(row)
        grouped[key[0]].append(row)
    summaries = []
    for members in grouped.values():
        summary = {
            name: members[0][name]
            for name in (
                "baseline",
                "control",
                "model",
                "generation_model",
                "setting_id",
                "query_language",
                "target_languages",
            )
        }
        summary["query_count"] = len(members)
        summary["mean_rounds"] = sum(row["rounds"] for row in members) / len(members)
        for name in members[0]:
            if name.endswith(("cost_usd", "seconds", "seconds_amortized")):
                total = nullable_sum(row[name] for row in members)
                summary[f"mean_{name}"] = total / len(members) if total is not None else None
        for prefix in ("adore_metrics", "control_metrics", "metric_deltas"):
            for name in members[0][prefix]:
                summary[f"{prefix}.{name}"] = sum(row[prefix][name] for row in members) / len(
                    members
                )
        summaries.append(summary)
    report = {
        "schema": 1,
        "corpus": corpus,
        "baseline": baseline,
        "control": control,
        "model": alias,
        "generation_model": variant_report["generation_model"],
        "adore_report": str(variant_dir),
        "control_report": str(control_dir),
        "adore_summary": variant_report["adore_summary"],
        "settings": summaries,
        "notes": [
            "All metric deltas are ADORE minus the same untranslated retriever on identical query/setting cohorts, corpus, and relevance judgments.",
            "Costs cover expansion and relevance-assessment API attempts, including failures and retries. Local embedding and retrieval compute are unpriced. Missing usage or prices remain null.",
            "adore_summary counts each API request once across settings. Per-query costs describe standalone pipeline work and must not be summed to bill shared requests across settings, round caps, or retrievers.",
            "ADORE component seconds sum original API-attempt times, expanded-query embedding build times, and round retrieval times. Control encoding and retrieval times are batch-amortized. Their difference is a component estimate, not measured end-to-end latency.",
        ],
    }
    destination = (
        config.output_dir
        / "comparisons"
        / corpus
        / digest({"report": report, "queries": query_rows})
    )
    if destination.exists():
        return destination
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", report)
        write_json(temporary / "per_query.json", query_rows)
        for filename, records in (("settings.csv", summaries), ("per_query.csv", query_rows)):
            flattened = []
            for record in records:
                flat = {}
                for key, value in record.items():
                    if isinstance(value, dict):
                        flat.update({f"{key}.{name}": metric for name, metric in value.items()})
                    else:
                        flat[key] = value
                flattened.append(flat)
            with (temporary / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(flattened[0]) if flattened else ["setting_id"]
                )
                writer.writeheader()
                writer.writerows(flattened)
    return destination


def compare_hyde(
    config: Config,
    corpus: str,
    alias: str,
    *,
    languages: tuple[str, ...],
    controls: tuple[str, ...] = ("dense",),
) -> Path:
    """Score existing HyDE/control runs on identical cohorts without any generation."""
    if not controls or any(control not in {"dense", "translate-dense"} for control in controls):
        raise ValidationError("HyDE controls must be dense and/or translate-dense")
    if not languages:
        raise ValidationError("Choose at least one HyDE output language to compare")
    summaries, query_rows, provenance = [], [], []
    for language in dict.fromkeys(languages):
        variant_dir = evaluate(config, corpus, alias, baseline="hyde-dense", hyde_language=language)
        variant_report = read_json(variant_dir / "report.json")
        variants = {
            (row["setting_id"], row["query_id"]): row
            for row in read_json(variant_dir / "per_query.json")
        }
        language_info = {
            "output_language": language,
            "report": str(variant_dir),
            "generation_model": variant_report["generation_model"],
            "hyde_summary": variant_report["hyde_summary"],
            "controls": [],
        }
        for control in dict.fromkeys(controls):
            kwargs = {"translation_language": language} if control == "translate-dense" else {}
            control_dir = evaluate(config, corpus, alias, baseline=control, **kwargs)
            control_report = read_json(control_dir / "report.json")
            originals = {
                (row["setting_id"], row["query_id"]): row
                for row in read_json(control_dir / "per_query.json")
            }
            if originals.keys() != variants.keys():
                raise ValidationError("HyDE and control evaluation cohorts differ; cannot compare")
            for name in ("relevance_hash", "corpus_fingerprint", "query_fingerprint", "model"):
                if variant_report["run"][name] != control_report["run"][name]:
                    raise ValidationError(f"HyDE and control {name} differ; cannot compare")
            if variant_report["run"].get("retrieval_unit", "chunk") != control_report["run"].get(
                "retrieval_unit", "chunk"
            ):
                raise ValidationError("HyDE and control retrieval units differ; cannot compare")
            language_info["controls"].append(
                {
                    "baseline": control,
                    "report": str(control_dir),
                    "translation_summary": control_report.get("translation_summary"),
                }
            )
            grouped = defaultdict(list)
            for key, variant in variants.items():
                original = originals[key]
                control_cost = (
                    original.get("translation_cost_usd") if control == "translate-dense" else 0.0
                )
                control_seconds = (
                    original.get("translation_seconds") if control == "translate-dense" else 0.0
                )
                hyde_seconds = variant["generation_seconds"]
                hyde_cost = variant["generation_cost_usd"]
                original_total = nullable_sum(
                    [
                        control_seconds,
                        original.get("query_encoding_seconds_amortized"),
                        original.get("retrieval_seconds_amortized"),
                    ]
                )
                variant_total = nullable_sum(
                    [
                        hyde_seconds,
                        variant.get("query_encoding_seconds_amortized"),
                        variant.get("retrieval_seconds_amortized"),
                    ]
                )
                row = {
                    "baseline": "hyde-dense",
                    "control": control,
                    "model": alias,
                    "generation_model": variant_report["generation_model"],
                    "output_language": language,
                    "setting_id": key[0],
                    "query_id": key[1],
                    "query_language": variant["query_language"],
                    "target_languages": variant["target_languages"],
                    "hyde_generation_cost_usd": hyde_cost,
                    "control_generation_cost_usd": control_cost,
                    "added_generation_cost_usd": _difference(hyde_cost, control_cost),
                    "hyde_generation_seconds": hyde_seconds,
                    "control_generation_seconds": control_seconds,
                    "added_generation_seconds": _difference(hyde_seconds, control_seconds),
                    "hyde_encoding_seconds_amortized": variant.get(
                        "query_encoding_seconds_amortized"
                    ),
                    "control_encoding_seconds_amortized": original.get(
                        "query_encoding_seconds_amortized"
                    ),
                    "hyde_retrieval_seconds_amortized": variant.get("retrieval_seconds_amortized"),
                    "control_retrieval_seconds_amortized": original.get(
                        "retrieval_seconds_amortized"
                    ),
                    "hyde_component_seconds": variant_total,
                    "control_component_seconds": original_total,
                    "added_component_seconds": _difference(variant_total, original_total),
                    "hyde_metrics": variant["metrics"],
                    "control_metrics": original["metrics"],
                    "metric_deltas": {
                        name: value - original["metrics"][name]
                        for name, value in variant["metrics"].items()
                    },
                }
                query_rows.append(row)
                grouped[key[0]].append(row)
            for _setting, members in sorted(grouped.items()):
                summary = {
                    name: members[0][name]
                    for name in (
                        "baseline",
                        "control",
                        "model",
                        "generation_model",
                        "output_language",
                        "setting_id",
                        "query_language",
                        "target_languages",
                    )
                }
                summary["query_count"] = len(members)
                for name in members[0]:
                    if name.endswith(("cost_usd", "seconds", "seconds_amortized")):
                        total = nullable_sum(row[name] for row in members)
                        summary[f"mean_{name}"] = (
                            total / len(members) if total is not None else None
                        )
                for prefix in ("hyde_metrics", "control_metrics", "metric_deltas"):
                    for name in members[0][prefix]:
                        summary[f"{prefix}.{name}"] = sum(
                            row[prefix][name] for row in members
                        ) / len(members)
                summaries.append(summary)
        provenance.append(language_info)
    report = {
        "schema": 1,
        "corpus": corpus,
        "baseline": "hyde-dense",
        "model": alias,
        "controls": list(dict.fromkeys(controls)),
        "languages": provenance,
        "settings": summaries,
        "notes": [
            "All deltas are HyDE minus the selected control on identical query/setting cohorts. A translate-dense control uses the same output language and retrieval model.",
            "Costs are generation API estimates only; local embedding/retrieval compute is not priced. Missing prices or usage produce null costs and deltas.",
            "Each language's generation summary counts unique requests once. Do not add reused costs across controls, settings or retrieval models.",
            "Component seconds sum original recorded generation attempts, amortized offline encoding build time and amortized retrieval batch time. They are not measured online end-to-end latency; historical runs without timings yield null comparisons.",
        ],
    }
    _label_document_comparison(config, report, query_rows)
    destination = (
        config.output_dir
        / "comparisons"
        / corpus
        / digest({"report": report, "queries": query_rows})
    )
    if destination.exists():
        return destination
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", report)
        write_json(temporary / "per_query.json", query_rows)
        with (temporary / "settings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(summaries[0]) if summaries else ["setting_id"]
            )
            writer.writeheader()
            writer.writerows(summaries)
    return destination
