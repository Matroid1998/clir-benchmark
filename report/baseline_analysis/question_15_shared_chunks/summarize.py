"""Summarize completed Narval measurements. Reads saved data, without running models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clir_benchmark.util import read_json, write_json

NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "zh": "Chinese",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(root):
    policy = read_json(root / "tokenizer_policy.json")
    budgets = policy["budgets"]
    cap = policy["max_tokens"]
    probes = read_csv(root / "word_windows.csv")
    editions = read_csv(root / "editions.csv")
    aliases = sorted(budgets)

    def maximum(probe):
        return max(int(probe[m]) for m in aliases)

    def group(subset):
        peaks = [maximum(row) for row in subset]
        return {
            "windows": len(subset),
            "all_fit_512": sum(n <= cap for n in peaks),
            "all_fit_native": sum(
                all(int(r[m]) <= budgets[m]["native_limit"] for m in aliases) for r in subset
            ),
            "maximum_tokens": max(peaks, default=None),
        }

    languages = []
    for lang in sorted({e["language"] for e in editions}):
        subset = [r for r in probes if r["language"] == lang]
        languages.append(
            {
                "language": lang,
                "editions": sum(e["language"] == lang for e in editions),
                **group(subset),
            }
        )
    models = []
    for m in aliases:
        counts = [int(r[m]) for r in probes]
        models.append(
            {
                "model": m,
                "native_limit": budgets[m]["native_limit"],
                "windows": len(counts),
                "over_512": sum(n > cap for n in counts),
                "over_native": sum(n > budgets[m]["native_limit"] for n in counts),
                "max_200_word_tokens": max(counts),
            }
        )
    write_csv(root / "language_fit.csv", languages)
    write_csv(root / "model_fit.csv", models)
    totals = {
        **group(probes),
        "editions": len(editions),
        "source_records": len({(r["corpus"], r["source_id"]) for r in editions}),
        "word_target": policy["max_words"],
        "token_cap": cap,
        "languages": sorted({r["language"] for r in editions}),
    }
    write_json(root / "summary.json", totals)

    fig, ax = plt.subplots(figsize=(7.3, 4.5), layout="constrained")
    languages.sort(key=lambda r: NAMES[r["language"]])
    for pos, row in enumerate(languages):
        if row["windows"]:
            pct = 100 * row["all_fit_512"] / row["windows"]
            ax.barh(pos, pct, color="#25678b", height=0.65)
            ax.text(
                pct + 1.3,
                pos,
                f"{pct:.0f}%  ({row['all_fit_512']}/{row['windows']})",
                ha="left",
                va="center",
                fontsize=9,
            )
    ax.set(
        yticks=range(len(languages)),
        yticklabels=[NAMES[r["language"]] for r in languages],
        xlim=(0, 127),
        xticks=[0, 25, 50, 75, 100],
        xlabel="200-word windows fitting all eight tokenizers (%)",
        title="Which 200-word windows fit within 512 tokens?",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(root / "language_fit.png", dpi=180)
    plt.close(fig)
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2))
