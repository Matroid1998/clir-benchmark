"""Measure a fixed, stratified subset of the production query-generation request ledgers."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from clir_benchmark import hyde, translation
from clir_benchmark.config import QUESTION_LANGUAGE, load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.generation import (
    generation_inputs,
    output_model,
    provider_cost_usd,
    provider_refused,
)
from clir_benchmark.util import digest, read_json, write_json

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[2] / "artifacts/verified-20261118-query-generation/local-record-refusals.yaml"
SEED = "verified-cost-sample-v1"


def sample(config):
    groups = []
    for corpus in config.datasets:
        questions = generation_inputs(config, corpus, action="sampling verified questions")
        for method in ("translate", "hyde"):
            for lang in config.query_generation.corpus_languages[corpus]:
                translate = method == "translate"
                eligible = [
                    (i, q)
                    for i, q in enumerate(questions)
                    if (q["question_language"] != lang) == translate
                ]

                rank_prefix = "\n".join([SEED, method, corpus, lang])

                def rank(item, prefix=rank_prefix):
                    return hashlib.sha256((prefix + "\n" + item[1]["id"]).encode()).digest()

                route = output_model(config, lang, config.translation if translate else config.hyde)
                payload = translation.request_payload if translate else hyde.request_payload
                selection = []
                for index, q in sorted(eligible, key=rank)[:6]:
                    identity = {
                        "schema": 1,
                        "query_id": q["id"],
                        "endpoint": route.endpoint,
                        "payload": payload(route, corpus, q, lang),
                    }
                    if not translate:
                        identity["corpus"] = corpus
                    key = digest(identity)
                    selection.append(
                        {
                            "method": method,
                            "corpus": corpus,
                            "language": lang,
                            "model": route.model,
                            "source_language": q["question_language"],
                            "query_id": q["id"],
                            "request_key": key,
                            "index": index,
                            "population": len(eligible),
                        }
                    )
                groups.append(selection)
    return [row for round_of_lanes in zip(*groups, strict=True) for row in round_of_lanes]


def read_state(config, row):
    parent = "translation_requests" if row["method"] == "translate" else "hyde_requests"
    path = config.output_dir / parent / (row["request_key"] + ".json")
    return read_json(path) if path.exists() else {}


def terminal(state):
    attempts = state.get("attempts", [])
    return state.get("status") == "success" or (
        state.get("status") == "failed"
        and attempts
        and provider_refused(attempts[-1].get("response"))
    )


def execute(config, row):
    if terminal(read_state(config, row)):
        return
    translate = row["method"] == "translate"
    action = translation.prepare_translations if translate else hyde.prepare_hyde
    language = row["language"] if translate else QUESTION_LANGUAGE
    for retry in range(12):
        try:
            action(config, row["corpus"], language, shard=(row["index"], 1), retry_transient=True)
            if not terminal(read_state(config, row)):
                raise RuntimeError(
                    "Production generation returned without the expected canonical request"
                )
            return
        except ValidationError as exc:
            if "running in another process" in str(exc) and retry < 11:
                time.sleep(5)
                if terminal(read_state(config, row)):
                    return
                continue
            # Keep this failed/pending attempt visible with unknown unreported cost.
            print(
                row["method"],
                row["corpus"],
                row["language"],
                row["request_key"],
                type(exc).__name__,
                flush=True,
            )
            return


def observed(config, row):
    state = read_state(config, row)
    attempts = state.get("attempts", [])
    costs = [provider_cost_usd(a.get("response")) for a in attempts]
    status = state.get("status", "missing")
    if status == "failed" and attempts and provider_refused(attempts[-1].get("response")):
        status = "provider_refused"
    usage = [((a.get("response") or {}).get("usage") or {}) for a in attempts]

    def tokens(k):
        values = [u.get(k) for u in usage]
        return sum(values) if values and all(isinstance(x, int) for x in values) else None

    return {
        **row,
        "status": status,
        "attempts": len(attempts),
        "unknown_cost_attempts": costs.count(None),
        "reported_usd": sum(c for c in costs if c is not None),
        "usd": sum(costs) if terminal(state) and costs and None not in costs else None,
        "initial_attempt_usd": costs[0] if costs else None,
        "providers": [(a.get("response") or {}).get("provider") for a in attempts],
        "prompt_tokens": tokens("prompt_tokens"),
        "completion_tokens": tokens("completion_tokens"),
    }


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["corpus"], row["language"]].append(row)
    lanes = []
    for (method, corpus, language), group in groups.items():
        costs = [r["usd"] for r in group if r["usd"] is not None]
        initial = [r["initial_attempt_usd"] for r in group if r["initial_attempt_usd"] is not None]
        mean = statistics.mean(costs) if costs else None
        population = group[0]["population"]
        usage = {
            k: [r[k] for r in group if r[k] is not None]
            for k in ("prompt_tokens", "completion_tokens")
        }
        lanes.append(
            {
                "method": method,
                "corpus": corpus,
                "language": language,
                "model": group[0]["model"],
                "population": population,
                "sample": len(group),
                "status_counts": dict(Counter(r["status"] for r in group)),
                "known_total_cost_records": len(costs),
                "unknown_cost_attempts": sum(r["unknown_cost_attempts"] for r in group),
                "provider_reported_usd": sum(r["reported_usd"] for r in group),
                "mean_usd_per_request_including_retries": mean,
                "mean_usd_first_attempt": statistics.mean(initial) if initial else None,
                **{f"mean_{k}": statistics.mean(v) if v else None for k, v in usage.items()},
                "projected_usd": population * mean if mean is not None else None,
                "variance_of_projected_usd": population**2
                * (1 - len(costs) / population)
                * statistics.variance(costs)
                / len(costs)
                if len(costs) > 1
                else None,
            }
        )

    def aggregate(groups):
        covered = [g for g in groups if g["projected_usd"] is not None]
        all_covered = len(groups) == len(covered)
        complete = all(g["known_total_cost_records"] == g["sample"] for g in groups)
        population = sum(g["population"] for g in groups)
        projection = sum(g["projected_usd"] for g in covered) if all_covered else None
        return {
            "population": population,
            "sample": sum(g["sample"] for g in groups),
            "known_cost_records": sum(g["known_total_cost_records"] for g in groups),
            "provider_reported_usd": sum(g["provider_reported_usd"] for g in groups),
            "unknown_cost_attempts": sum(g["unknown_cost_attempts"] for g in groups),
            "refusals": sum(g["status_counts"].get("provider_refused", 0) for g in groups),
            "covered_population": sum(g["population"] for g in covered),
            "projected_usd": projection,
            "weighted_mean_usd_per_initial_request": projection / population
            if projection is not None
            else None,
            "projection_sample_complete": complete,
            "projected_usd_sampling_se": sum(g["variance_of_projected_usd"] for g in groups) ** 0.5
            if complete and all(g["variance_of_projected_usd"] is not None for g in groups)
            else None,
        }

    old = read_json(HERE / "estimate.json")
    names = ["translate", "hyde"]
    return {
        "measured_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "methodology": "Six lowest SHA-256 ranks per stage/corpus/output language from the pinned QA cohort. Translation identities excluded. All attempts of a sampled initial request contribute provider-reported USD, including refusals. Missing prices remain unknown. Projection sums each lane's eligible population times its sample mean. Sampling SE uses stratified finite-population variance and excludes provider-price or traffic drift.",
        "historical_forecast_usd": {
            k: {n: old[k][n] for n in ("wafer_usd", "darkbloom_usd")} for k in names
        },
        "total": aggregate(lanes),
        "by_stage": {k: aggregate([g for g in lanes if g["method"] == k]) for k in names},
        "by_corpus": {
            k: aggregate([g for g in lanes if g["corpus"] == k])
            for k in {g["corpus"] for g in lanes}
        },
        "lanes": lanes,
    }


def ledger_snapshot(config):
    """Count each physical API attempt once while the main campaign continues."""
    result = {"scan_started_utc": datetime.now(UTC).isoformat()}
    for method, parent in [("translate", "translation_requests"), ("hyde", "hyde_requests")]:
        counts = Counter()
        usd = []
        for path in (config.output_dir / parent).glob("*.json"):
            state = read_json(path)
            counts["physical_request_keys"] += 1
            for attempt in state["attempts"]:
                counts["attempts"] += 1
                response = attempt.get("response")
                counts["provider_refusal_attempts"] += int(provider_refused(response))
                amount = provider_cost_usd(response)
                if amount is None:
                    counts["unpriced_attempts"] += 1
                else:
                    usd.append(amount)
        result[method] = {**counts, "priced_attempts": len(usd), "provider_reported_usd": sum(usd)}
    result["scan_finished_utc"] = datetime.now(UTC).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Perform uncached sampled production requests with up to six workers",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.query_generation.record_refusals:
        parser.error("Use the production record-refusals configuration")
    chosen = sample(cfg)
    if len(chosen) != 204 or len({r["request_key"] for r in chosen}) != len(chosen):
        raise ValueError("Expected 204 distinct production request keys")
    print(
        "Existing terminal requests:",
        sum(bool(terminal(read_state(cfg, r))) for r in chosen),
        "of",
        len(chosen),
        flush=True,
    )
    if args.run:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda r: execute(cfg, r), chosen))
    records = [observed(cfg, r) for r in chosen]
    result = summary(records)
    result["campaign_ledgers_including_sample"] = ledger_snapshot(cfg)
    write_json(HERE / "sample_records.json", records)
    write_json(HERE / "measured_cost.json", result)
    print(json.dumps(result["total"], indent=2))


if __name__ == "__main__":
    main()
