"""Shared plotting and paired-analysis helpers; never executes benchmark stages."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["en", "de", "es", "fr", "zh"]
CORPORA = ["chemistry", "eurlex", "un"]
CORPUS_LABEL = {"chemistry": "Chemistry", "eurlex": "EUR-Lex", "un": "UN"}
MODELS = ["embeddinggemma", "bge-m3", "qwen3-0.6B", "nomic-v2-moe", "granite-278m", "LaBSE", "SapBERT", "e5-large-instruct"]
MODEL_LABEL = {"embeddinggemma": "EmbeddingGemma", "bge-m3": "BGE-M3", "qwen3-0.6B": "Qwen3-0.6B", "nomic-v2-moe": "Nomic v2 MoE", "granite-278m": "Granite-278M", "LaBSE": "LaBSE", "SapBERT": "SapBERT", "e5-large-instruct": "E5-large-instruct"}
COLORS = {"original": "#555555", "translate": "#0072B2", "hyde": "#D55E00"}
SEED = 42


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 200, "figure.dpi": 120,
    })


def savefig(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table(frame, directory, name, *, index=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=index)
    (directory / f"{name}.tex").write_text(
        frame.to_latex(index=index, float_format="%.4f", escape=True), encoding="utf-8"
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def paired_bootstrap(values, *, seed=SEED, draws=5000):
    """Mean and percentile CI for ONE independent value per source/query cluster."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        end = min(draws, start + 500)
        means[start:end] = values[rng.integers(len(values), size=(end-start, len(values)))].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(means, [0.025, 0.975]))


def centered_heatmap(ax, values, rows, columns, *, limit=None, decimals=2, cmap="RdBu_r"):
    values = np.asarray(values, dtype=float)
    limit = limit or max(0.01, float(np.nanmax(np.abs(values))))
    im = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(columns)):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.{decimals}f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > limit * .65 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
    return im


style()
