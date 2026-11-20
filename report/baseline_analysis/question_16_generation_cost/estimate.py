"""Forecast the verified-cohort generation bill from prior measured token usage."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE.parent / "question_13_query_generation" / "preflight.json"
HISTORY = HERE.parents[1] / "automatic_analysis_benchmarking" / "data" / "generation.parquet"
# Standard USD per million uncached input and billed output tokens, Nov 19, 2026.
RATES = {
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "qwen/qwen3.8-27b": (0.06, 2.20),  # Wafer
}


def main():
    history = pq.read_table(
        HISTORY,
        columns=[
            "method",
            "corpus",
            "output_language",
            "query_language",
            "status",
            "input_tokens",
            "output_tokens",
        ],
    ).to_pylist()
    samples = defaultdict(list)
    for row in history:
        if row["method"] == "translate" and row["status"] != "translated":
            continue
        if row["method"] == "hyde" and row["query_language"] != row["output_language"]:
            continue
        pair = (row["method"], row["corpus"])
        samples[pair + (row["output_language"],)].append(row)
        samples[pair].append(row)

    preflight = json.loads(PREFLIGHT.read_text())
    jobs = [("translate", job["corpus"], job) for job in preflight["translation"]["jobs"]]
    jobs.extend(
        ("hyde", job["corpus"], lane)
        for job in preflight["hyde"]["jobs"]
        for lane in job["outputs"]
    )
    result = defaultdict(lambda: defaultdict(float))
    for method, corpus, lane in jobs:
        group = samples.get((method, corpus, lane["output_language"])) or samples[method, corpus]
        requests = lane["requests_before_cache_or_retries"]
        src = requests * mean(row["input_tokens"] for row in group) / 1e6
        out = requests * mean(row["output_tokens"] for row in group) / 1e6
        model = lane["model"]
        prompt_rate, output_rate = RATES[model]
        for totals in (result[method], result[corpus]):
            totals["initial_calls"] += requests
            totals["wafer_usd"] += src * prompt_rate + out * output_rate
            if model.startswith("qwen/"):
                totals["darkbloom_usd"] += src * 0.035 + out * 1.50
            else:
                totals["darkbloom_usd"] += src * prompt_rate + out * output_rate
            totals["double_output_10pct_extra_wafer_usd"] += 1.10 * (
                src * prompt_rate + 2 * out * output_rate
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
