"""Account once for each saved generation request and report incomplete telemetry."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from score import GEN, HERE


def summarize(part):
    paid = part.loc[part.attempts > 0]
    return {
        "query_variants": len(part),
        "distinct_questions": part.id.nunique(),
        "identity": int(part.status.eq("identity").sum()),
        "refusals": int(part.status.eq("provider_refused").sum()),
        "attempts": int(part.attempts.sum()),
        "extra_attempts": int((paid.attempts - 1).sum()),
        "known_attempt_usd": part.known_attempt_usd.sum(),
        "missing_attempt_costs": int(part.missing_attempt_costs.sum()),
        "variants_missing_full_cost": int(part.api_cost_usd.isna().sum()),
        "mean_known_variant_usd": part.api_cost_usd.mean(),
        "mean_known_generation_s": part.seconds.mean(),
        "p50_known_generation_s": part.seconds.median(),
        "p95_known_generation_s": part.seconds.quantile(0.95),
        "mean_known_paid_generation_s": paid.seconds.mean(),
        "unknown_generation_durations": int(part.seconds.isna().sum()),
        **{
            "known_" + col: int(part[col].sum(min_count=1)) if part[col].notna().any() else None
            for col in ("input_tokens", "cached_input_tokens", "output_tokens")
        },
        "variants_missing_tokens": int(part.input_tokens.isna().sum()),
    }


def main():
    frames, ledger, prices = [], [], []
    for method, directory, requests, seconds in [
        ("translate", "translations", "translation_requests", "translation_seconds"),
        ("hyde", "hyde", "hyde_requests", "generation_seconds"),
    ]:
        variants = []
        for f in sorted((GEN / directory).glob("*/*/*/*.parquet")):
            d = pd.read_parquet(f)
            corpus, lane = f.parts[-4:-2]
            variants.append(
                d.rename(columns={seconds: "seconds"}).assign(
                    method=method, corpus=corpus, lane=lane
                )
            )
        v = pd.concat(variants, ignore_index=True)
        paid = v.loc[v.request_key.notna()].set_index("request_key")
        assert paid.index.is_unique
        raw, parents = {}, {}
        for f in sorted((GEN / requests).glob("*.json")):
            obj = json.loads(f.read_bytes())
            if not isinstance(obj, dict) or "attempts" not in obj:
                continue
            key = obj["request_key"]
            assert key == f.stem and key not in raw
            charges = [(a.get("response") or {}).get("usage") or {} for a in obj["attempts"]]
            raw[key] = dict(
                request_keys=1,
                attempts=len(charges),
                non_success_attempts=sum(a.get("status") != "success" for a in obj["attempts"]),
                known_attempt_usd=sum(u["cost"] for u in charges if u.get("cost") is not None),
                missing_attempt_costs=sum(u.get("cost") is None for u in charges),
            )
            parents[key] = obj.get("prior_request_keys", [])

        def ancestry(key, parents):
            keys, pending = set(), [key]
            while pending:
                k = pending.pop()
                if k not in keys:
                    keys.add(k)
                    pending.extend(parents[k])
            return keys

        retained = set()
        raw_cost = {}
        for key in paid.index:
            chain = ancestry(key, parents)
            assert not (chain & retained), "Overlapping retained generation chains"
            retained.update(chain)
            assert sum(raw[k]["attempts"] for k in chain) == paid.loc[key, "attempts"]
            raw_cost[key] = (
                sum(raw[k]["known_attempt_usd"] for k in chain),
                sum(raw[k]["missing_attempt_costs"] for k in chain),
            )
        counters = defaultdict(lambda: defaultdict(float))
        for k, record in raw.items():
            scope = "retained" if k in retained else "outside_retained_cohort"
            for field, value in record.items():
                counters[scope][field] += value
        v["known_attempt_usd"] = [raw_cost[k][0] if pd.notna(k) else 0.0 for k in v.request_key]
        v["missing_attempt_costs"] = [raw_cost[k][1] if pd.notna(k) else 0 for k in v.request_key]
        complete = v.missing_attempt_costs.eq(0)
        np.testing.assert_allclose(
            v.loc[complete, "known_attempt_usd"], v.loc[complete, "api_cost_usd"], atol=1e-10
        )
        assert v.loc[v.request_key.isna(), "status"].eq("identity").all()
        for scope, record in counters.items():
            ledger.append(dict(method=method, scope=scope, **record))
        frames.append(v)
    d = pd.concat(frames, ignore_index=True)
    for dims, name in [
        (["method"], "method"),
        (["method", "corpus"], "corpus"),
        (["method", "corpus", "lane"], "lane"),
        (["method", "corpus", "lane", "source_language", "requested_model"], "language_model"),
    ]:
        rows = [
            {
                **dict(zip(dims, keys if isinstance(keys, tuple) else (keys,), strict=True)),
                **summarize(g),
            }
            for keys, g in d.groupby(dims, dropna=False)
        ]
        pd.DataFrame(rows).to_csv(HERE / f"generation_by_{name}.csv", index=False)
    pd.DataFrame(ledger).to_csv(HERE / "request_ledger.csv", index=False)
    import yaml

    config = yaml.safe_load(
        (HERE.parents[2] / "configs/verified-20261120-transformed-retrieval.yaml").read_text()
    )
    for alias, model in config["query_generation"]["models"].items():
        prices.append(
            {
                "stage": "translation and HyDE",
                "alias": alias,
                "model": model["model"],
                **{
                    col: model.get(col)
                    for col in (
                        "input_usd_per_million",
                        "cached_input_usd_per_million",
                        "output_usd_per_million",
                    )
                },
            }
        )
    pd.DataFrame(prices).to_csv(HERE / "configured_pricing.csv", index=False)


if __name__ == "__main__":
    main()
