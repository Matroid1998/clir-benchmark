#!/usr/bin/env python3
"""Question 1: reproduce Recall@10 tables and figures from saved results only."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANGUAGES = ["de", "en", "es", "fr", "zh"]
CORPORA = ["chemistry", "eurlex", "un"]
CORPUS_LABELS = {"chemistry": "Chemistry patents", "eurlex": "EUR-Lex", "un": "UN"}
MODELS = [
    "embeddinggemma",
    "bge-m3",
    "qwen3-0.6B",
    "nomic-v2-moe",
    "granite-278m",
    "LaBSE",
    "SapBERT",
    "e5-large-instruct",
]
VARIANTS = ["Original"] + [f"Translate {x}" for x in LANGUAGES] + [f"HyDE {x}" for x in LANGUAGES]
COHORTS = ["selected_documents", "legacy_all_questions_chunks"]
REGIMES = ["mixed", "cross_language", "monolingual"]
IDENTITY = ["cohort", "corpus", "model", "baseline", "input_language", "report_id"]
SUMMARY_KEYS = IDENTITY + ["regime"]
DEFAULT_ARTIFACTS = "artifacts/fir-20261109-all-questions/artifacts"
DEFAULT_CAMPAIGN = "artifacts/narval-20261111-selected-document"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def variant(baseline, language):
    if baseline == "translate-dense":
        return f"Translate {language}"
    if baseline == "hyde-dense":
        return f"HyDE {language}"
    return {"dense": "Original", "bm25": "BM25", "hybrid": "Hybrid"}[baseline]


def latest_reports(campaign, artifacts):
    """Resolve successful attempts using journals, independent of older analyses."""
    status_path = campaign / ".local-monitor/latest-attempts-final-status-normalized-20261115.json"
    status = read_json(status_path)
    require(status["all_succeeded"], "Latest campaign status is not successful")
    sources = {str(status_path): digest(status_path)}

    def relocate(raw):
        p = Path(raw)
        for name, root in ((campaign.name, campaign), (artifacts.parent.name, artifacts.parent)):
            if name in p.parts:
                position = p.parts.index(name)
                return root.joinpath(*p.parts[position + 1 :])
        raise ValueError(f"Cannot relocate {raw}")

    selected = set()
    for job in status["jobs"]:
        if job["kind"] != "variant-score":
            continue
        require(job["state"] == "COMPLETED", f"Incomplete job {job['label']}")
        journals = [relocate(item["path"]) for item in job["stage_journals"]]
        path, journal = max(
            ((p, read_json(p)) for p in journals),
            key=lambda pair: (pair[1].get("started_at", ""), pair[0].name),
        )
        sources[str(path)] = digest(path)
        stages = [x for x in journal["stages"] if x["stage"].startswith("evaluate-")]
        require(journal["status"] == "success", f"Failed journal {path}")
        require(len(stages) == 11, f"Expected eleven evaluation stages in {path}")
        for stage in stages:
            require(stage["status"] == "success", f"Failed evaluation in {path}")
            selected.add(relocate(stage["result"]).resolve())
    require(len(selected) == 264, f"Expected 264 selected-document reports, found {len(selected)}")
    return selected, sources


def summarize(frame):
    """Equal setting weight matches report.metrics.macro_by_regime exactly."""
    return (
        frame.groupby(SUMMARY_KEYS, dropna=False, sort=False)
        .agg(
            recall_at_10=("recall_at_10", "mean"),
            setting_count=("setting_id", "size"),
            query_setting_count=("query_count", "sum"),
            min_setting_queries=("query_count", "min"),
            max_setting_queries=("query_count", "max"),
            unique_query_count=("unique_queries_in_regime", "first"),
        )
        .reset_index()
    )


def add_controls(frame, *, settings=False):
    """Use the same retriever, cohort, qrels, corpus and query cohort for controls."""
    keys = [
        "cohort",
        "corpus",
        "model",
        "relevance_hash",
        "corpus_fingerprint",
        "query_fingerprint",
        "regime",
    ]
    if settings:
        keys += ["setting_id", "query_count"]
    controls = frame[frame.baseline.eq("dense")]
    require(not controls.duplicated(keys).any(), "Ambiguous dense control")
    result = frame.merge(
        controls[keys + ["recall_at_10"]].rename(
            columns={"recall_at_10": "dense_control_recall_at_10"}
        ),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    result["delta_vs_dense_pp"] = 100 * (result.recall_at_10 - result.dense_control_recall_at_10)
    translated = frame[frame.baseline.eq("translate-dense")]
    language_keys = keys + ["input_language"]
    require(not translated.duplicated(language_keys).any(), "Ambiguous translation control")
    result = result.merge(
        translated[language_keys + ["recall_at_10"]].rename(
            columns={"recall_at_10": "translation_control_recall_at_10"}
        ),
        on=language_keys,
        how="left",
        validate="many_to_one",
    )
    result["delta_vs_same_language_translation_pp"] = (
        100 * (result.recall_at_10 - result.translation_control_recall_at_10)
    ).where(result.baseline.eq("hyde-dense"))
    result["recall_at_10_pct"] = 100 * result.recall_at_10
    return result


def extract(args, selected):
    settings, catalog, reported, inputs = [], [], [], {}
    cohorts, model_specs = {}, {}
    max_setting_error = max_macro_error = max_mteb_error = 0.0
    total_query_rows = 0
    for path in sorted((args.artifacts / "reports").glob("*/*/report.json")):
        report = read_json(path)
        directory = path.parent.resolve()
        selection = report["candidate_selection"]
        unit = report.get("retrieval_unit", "chunk")
        if directory in selected:
            require(
                unit == "document" and selection["candidate_mode"] == "best-per-document",
                f"Unexpected selected report {path}",
            )
            cohort = COHORTS[0]
        elif unit == "chunk" and selection["candidate_mode"] == "all":
            cohort = COHORTS[1]
        else:
            raise ValueError(f"Unclassified report; explicit cohort handling required: {path}")
        model = report["model"] or "none"
        if (
            cohort not in args.cohort
            or (args.baseline and report["baseline"] not in args.baseline)
            or (args.corpus and report["corpus"] not in args.corpus)
            or (args.model and model not in args.model)
        ):
            continue
        baseline = report["baseline"]
        language = report.get("output_language") or report.get("translation_language") or "original"
        run = report["run"]
        qpath = path.parent / "per_query.json"
        mpath = path.parent / "mteb.json"
        queries = read_json(qpath)
        mteb = {x["hf_subset"]: x for x in read_json(mpath)["scores"]["test"]}
        total_query_rows += len(queries)
        for source in (path, qpath, mpath):
            inputs[str(source.resolve())] = digest(source)
        keys = [(x["setting_id"], x["query_id"]) for x in queries]
        require(len(keys) == len(set(keys)), f"Duplicate per-query rows in {qpath}")
        cohort_key = (cohort, report["corpus"], run["relevance_hash"])
        contract = (
            run["corpus_fingerprint"],
            run["query_fingerprint"],
            frozenset(keys),
            json.dumps(run["settings"], sort_keys=True),
            run["top_k"],
        )
        if cohort_key in cohorts:
            require(cohorts[cohort_key] == contract, f"Mismatched cohort/settings in {path}")
        else:
            cohorts[cohort_key] = contract
        spec_key = (*cohort_key, model)
        specification = json.dumps(run["model"], sort_keys=True)
        require(
            model_specs.get(spec_key, specification) == specification,
            f"Mismatched model specification in {path}",
        )
        model_specs[spec_key] = specification
        generation = run.get("hyde") if baseline == "hyde-dense" else run.get("translation")
        model_spec = run["model"] or {}
        common = {
            "cohort": cohort,
            "corpus": report["corpus"],
            "model": model,
            "baseline": baseline,
            "input_language": language,
            "variant": variant(baseline, language),
            "retrieval_unit": unit,
            "candidate_mode": selection["candidate_mode"],
            "generation_model": (generation or {}).get("configuration", {}).get("model"),
            "embedding_model_id": model_spec.get("repo_id"),
            "embedding_model_revision": model_spec.get("revision"),
            "relevance_hash": run["relevance_hash"],
            "corpus_fingerprint": run["corpus_fingerprint"],
            "query_fingerprint": run["query_fingerprint"],
            "report_id": path.parent.name,
            "report_path": str(path.resolve()),
        }
        catalog.append(
            {
                **common,
                "unique_query_count": len({q["query_id"] for q in queries}),
                "query_setting_count": len(queries),
                "resolved_run_parameters_json": json.dumps(run, sort_keys=True),
            }
        )
        groups = defaultdict(list)
        regime_queries = defaultdict(set)
        for q in queries:
            value = q["metrics"]["Recall@10"]
            require(math.isfinite(value) and 0 <= value <= 1, f"Invalid Recall@10 in {qpath}")
            require(
                q["baseline"] == baseline and q["corpus"] == report["corpus"],
                f"Per-query identity mismatch in {qpath}",
            )
            groups[q["setting_id"]].append(q)
            regime_queries[q["regime"]].add(q["query_id"])
        seen_settings = set()
        for row in report["metrics"]["setting"]:
            setting_id = row["setting_id"]
            require(setting_id not in seen_settings, f"Duplicate setting in {path}")
            seen_settings.add(setting_id)
            group = groups[setting_id]
            require(len(group) == row["query_count"], f"Wrong query count in {path}")
            require(
                all(
                    q["query_language"] == row["query_language"]
                    and q["target_languages"] == row["target_languages"]
                    and q["regime"] == row["regime"]
                    for q in group
                ),
                f"Setting identity mismatch in {path}",
            )
            value = math.fsum(q["metrics"]["Recall@10"] for q in group) / len(group)
            error = abs(value - row["metrics"]["Recall@10"])
            max_setting_error = max(max_setting_error, error)
            max_mteb_error = max(max_mteb_error, abs(value - mteb[setting_id]["recall_at_10"]))
            require(mteb[setting_id]["query_count"] == len(group), f"MTEB count mismatch: {mpath}")
            targets = row["target_languages"]
            settings.append(
                {
                    **common,
                    "regime": row["regime"],
                    "setting_id": setting_id,
                    "query_language": row["query_language"],
                    "target_language": "mixed" if row["regime"] == "mixed" else targets,
                    "target_languages": targets,
                    "query_count": len(group),
                    "unique_queries_in_regime": len(regime_queries[row["regime"]]),
                    "recall_at_10": value,
                }
            )
        require(seen_settings == set(groups) == set(mteb), f"Setting coverage mismatch in {path}")
        for row in report["metrics"]["macro_by_regime"]:
            values = [
                x["metrics"]["Recall@10"]
                for x in report["metrics"]["setting"]
                if x["regime"] == row["regime"]
            ]
            require(len(values) == row["setting_count"], f"Macro setting count mismatch in {path}")
            max_macro_error = max(
                max_macro_error, abs(math.fsum(values) / len(values) - row["metrics"]["Recall@10"])
            )
            reported.append(
                {
                    **{k: common[k] for k in IDENTITY},
                    "regime": row["regime"],
                    "reported_recall": row["metrics"]["Recall@10"],
                }
            )
    require(bool(settings), "No reports match the requested selection")
    require(
        max(max_setting_error, max_macro_error, max_mteb_error) < 1e-12,
        "Recall@10 differs from saved reports or MTEB exports",
    )
    frame = pd.DataFrame(settings)
    summary = summarize(frame)
    merged = summary.merge(pd.DataFrame(reported), on=SUMMARY_KEYS, validate="one_to_one")
    reconstruction_error = float((merged.recall_at_10 - merged.reported_recall).abs().max())
    require(reconstruction_error < 1e-12, "Reconstructed macro differs from saved report")
    metadata = pd.DataFrame(catalog).drop(
        columns=[
            "unique_query_count",
            "query_setting_count",
            "resolved_run_parameters_json",
        ]
    )
    summary = summary.merge(metadata, on=IDENTITY, validate="many_to_one")
    validation = {
        "status": "passed",
        "report_count": len(catalog),
        "per_query_rows_checked": total_query_rows,
        "setting_rows": len(frame),
        "macro_rows": len(summary),
        "max_per_query_to_report_error": max_setting_error,
        "max_per_query_to_mteb_error": max_mteb_error,
        "max_saved_macro_error": max_macro_error,
        "max_reconstructed_macro_error": reconstruction_error,
        "cohort_and_setting_keys_match_within_corpus": True,
        "corpus_query_qrel_fingerprints_and_model_specs_match": True,
        "report_counts_by_cohort": pd.DataFrame(catalog).cohort.value_counts().to_dict(),
    }
    return add_controls(frame, settings=True), add_controls(summary), catalog, inputs, validation


def heatmap(ax, data, title, *, counts=None, small=False):
    values = data.to_numpy(dtype=float) * 100
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e9ebee")
    ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(
        range(len(data.columns)), data.columns, rotation=48, ha="right", fontsize=7 if small else 9
    )
    ax.set_yticks(range(len(data.index)), data.index, fontsize=8 if small else 10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            label = "—" if not np.isfinite(v) else f"{v:.1f}"
            if counts is not None and np.isfinite(v):
                n = counts.iloc[i, j]
                label += f"\n(n={int(n)})"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=6 if small else 8,
                color="white" if np.isfinite(v) and v >= 58 else "#18212b",
            )
    ax.set_title(title, fontsize=11 if small else 13, pad=12, loc="left", weight="bold")
    ax.set_xticks(np.arange(-0.5, len(data.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.index)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, output, name, title, caption, gallery):
    fig.suptitle(title, x=0.03, ha="left", fontsize=18, weight="bold")
    fig.text(0.03, 0.02, caption, fontsize=10, va="bottom", color="#374151")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / "figures" / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    gallery.append({"name": name, "title": title, "caption": caption})


def figures(settings, summary, output):
    gallery = []
    doc = summary[summary.cohort.eq(COHORTS[0])]
    for regime in REGIMES:
        subset = doc[doc.regime.eq(regime)]
        corpora = [c for c in CORPORA if c in subset.corpus.unique()]
        if not corpora:
            continue
        fig, axes = plt.subplots(1, len(corpora), figsize=(8 * len(corpora), 6.1), squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.28, wspace=0.35)
        for ax, corpus in zip(axes.flat, corpora, strict=True):
            f = subset[subset.corpus.eq(corpus)]
            data = f.pivot(index="model", columns="variant", values="recall_at_10").reindex(
                index=[m for m in MODELS if m in f.model.unique()],
                columns=[v for v in VARIANTS if v in f.variant.unique()],
            )
            n = f.unique_query_count.unique()
            require(len(n) == 1, "Inconsistent unique query denominators")
            heatmap(ax, data, f"{CORPUS_LABELS[corpus]} · {n[0]} unique questions")
        save_figure(
            fig,
            output,
            f"selected_documents_{regime}",
            f"Recall@10 (%) · {regime.replace('_', ' ')} search · selected questions",
            "Equal weight per nonempty original-query-language × target setting. Columns give generated input language.\n"
            "Original uses each question's original language. Translation and HyDE languages are separate runs; darker = higher recall.",
            gallery,
        )
    detail = settings[settings.cohort.eq(COHORTS[0])]
    for corpus in CORPORA:
        f = detail[detail.corpus.eq(corpus)]
        if f.empty:
            continue
        query_languages = [x for x in LANGUAGES if x in f.query_language.unique()]
        # Full original language × search-target language matrices, one figure per model.
        for model in MODELS:
            model_rows = f[f.model.eq(model)]
            available = [v for v in VARIANTS if v in model_rows.variant.unique()]
            if not available:
                continue
            fig, axes = plt.subplots(3, 4, figsize=(21, 13.5))
            fig.subplots_adjust(
                left=0.05, right=0.99, top=0.89, bottom=0.12, hspace=0.85, wspace=0.3
            )
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, v in zip(axes.flat, available, strict=False):
                ax.set_visible(True)
                rows = model_rows[model_rows.variant.eq(v)]
                matrix = rows.pivot(
                    index="query_language", columns="target_language", values="recall_at_10"
                )
                counts = rows.pivot(
                    index="query_language", columns="target_language", values="query_count"
                )
                matrix = matrix.reindex(index=query_languages, columns=LANGUAGES + ["mixed"])
                counts = counts.reindex_like(matrix)
                heatmap(ax, matrix, v, counts=counts, small=True)
                ax.set_ylabel("Original question language", fontsize=8)
            save_figure(
                fig,
                output,
                f"language_pairs_{corpus}_{model}",
                f"{CORPUS_LABELS[corpus]} · {model} · Recall@10 (%)",
                "Rows: original question language. Columns: searched corpus language; mixed searches all configured available languages.\n"
                "Each cell is a query mean, with its eligible question count. Grey/dash = no eligible questions. Sparse cells require care.\n"
                "Single-language and mixed search have different relevant-document denominators; do not pool their scores.",
                gallery,
            )
        # Mixed retrieval split by original query language, retaining all model/language variants.
        fig, axes = plt.subplots(
            len(query_languages), 1, figsize=(17, 4 * len(query_languages)), squeeze=False
        )
        fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.10, hspace=0.9)
        for ax, lang in zip(axes.flat, query_languages, strict=True):
            rows = f[f.query_language.eq(lang) & f.regime.eq("mixed")]
            matrix = rows.pivot(index="model", columns="variant", values="recall_at_10").reindex(
                index=[m for m in MODELS if m in rows.model.unique()],
                columns=[v for v in VARIANTS if v in rows.variant.unique()],
            )
            heatmap(
                ax, matrix, f"Original question language: {lang} · n={rows.query_count.iloc[0]}"
            )
        save_figure(
            fig,
            output,
            f"query_languages_{corpus}",
            f"{CORPUS_LABELS[corpus]} · mixed search · Recall@10 (%) by original question language",
            "Cell = mean over questions in the stated original language. Every variant uses the same questions and search pool.\n"
            "Columns identify the generated input language. Original preserves the question's language. Darker = higher recall.",
            gallery,
        )
    legacy = settings[settings.cohort.eq(COHORTS[1])].copy()
    legacy["system"] = legacy.apply(
        lambda x: "BM25 (no embedding)" if x.baseline == "bm25" else f"{x.model} · {x.variant}",
        axis=1,
    )
    for corpus in CORPORA:
        f = legacy[legacy.corpus.eq(corpus)]
        if f.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(23, 9.5))
        fig.subplots_adjust(left=0.16, right=0.99, top=0.85, bottom=0.20, wspace=0.8)
        systems = ["BM25 (no embedding)"] if "bm25" in f.baseline.unique() else []
        systems += [
            f"{m} · {v}"
            for m in MODELS
            for v in ["Original", "Hybrid"]
            if f"{m} · {v}" in f.system.unique()
        ]
        for ax, regime in zip(axes, REGIMES, strict=True):
            rows = f[f.regime.eq(regime)]
            values = rows.groupby(["system", "query_language"]).recall_at_10.mean().unstack()
            values = values.reindex(index=systems, columns=LANGUAGES)
            heatmap(ax, values, regime.replace("_", " "))
        save_figure(
            fig,
            output,
            f"legacy_chunks_{corpus}",
            f"{CORPUS_LABELS[corpus]} · earlier all-question/chunk experiment · Recall@10 (%)",
            "Separate experiment: all eligible questions, chunk relevance. Equal weight per nonempty target setting within original question language.\n"
            "Compare BM25, dense and hybrid within this cohort only. Grey/dash = unavailable. UN has no saved BM25/hybrid chunk reports.",
            gallery,
        )
    return gallery


def write_gallery(output, gallery, summary):
    items = []
    for item in gallery:
        name = html.escape(item["name"])
        items.append(
            f"<details open><summary>{html.escape(item['title'])}</summary>"
            f'<p>{html.escape(item["caption"])}</p><a href="figures/{name}.png">'
            f'<img loading="lazy" src="figures/{name}.png"></a>'
            f'<p><a href="figures/{name}.pdf">PDF</a> · <a href="figures/{name}.svg">SVG</a></p></details>'
        )
    display = summary[
        [
            "cohort",
            "corpus",
            "model",
            "baseline",
            "input_language",
            "regime",
            "recall_at_10_pct",
            "delta_vs_dense_pp",
            "unique_query_count",
            "setting_count",
        ]
    ]
    table = display.to_html(
        index=False, na_rep="—", float_format=lambda v: f"{v:.2f}", table_id="results"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Question 1 — Recall@10</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1600px;margin:36px auto;padding:0 24px;color:#172337;background:#f7f9fc}
h1{font-size:32px}a{color:#1559a5}details{background:white;border:1px solid #dce2eb;border-radius:8px;padding:18px;margin:24px 0}
summary{font-size:20px;font-weight:650;cursor:pointer}img{width:100%;height:auto}p{white-space:pre-line}
.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}
td,th{padding:7px;border:1px solid #dce2eb}th{position:sticky;top:0;background:#e8eef7}input{font:inherit;padding:8px;width:95%;margin:12px 0}
</style><h1>Recall@10 by baseline, embedding model, corpus and language</h1>
<p>Figures show percentages. Raw CSV recall is in [0,1]; deltas are percentage points.
The selected-document experiment has 638 questions: chemistry 197, EUR-Lex 194, UN 247.
Recall@10 counts known relevant document-language versions recovered in the top 10. It differs from source success in mixed search.
Regime names describe the ORIGINAL question language relative to the target, even after translation or HyDE.
Macro results give each nonempty language/target setting equal weight. Missing controls and unavailable settings stay unknown.</p>
<p><a href="tables/recall_at_10_by_setting.csv">All language pairs and controls (CSV)</a> ·
<a href="tables/recall_at_10_by_regime.csv">All regime summaries (CSV)</a> ·
<a href="tables/report_catalog.csv">Model identities and provenance (CSV)</a> ·
<a href="results.txt">Numerical reading guide</a> · <a href="validation.json">Validation</a></p>
<h2>All baseline summaries</h2><input id="filter" placeholder="Filter: e.g. qwen3, translate-dense, mixed, chemistry">
<div class="scroll">TABLE</div><h2>Figures</h2>FIGURES
<script>document.getElementById('filter').addEventListener('input',e=>{
const words=e.target.value.toLowerCase().split(/[ ,]+/).filter(Boolean);
document.querySelectorAll('#results tbody tr').forEach(r=>r.hidden=!words.every(w=>r.textContent.toLowerCase().includes(w)));
});</script></html>"""
    (output / "index.html").write_text(
        page.replace("TABLE", table).replace("FIGURES", "\n".join(items))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--artifacts", type=Path, default=repo / DEFAULT_ARTIFACTS)
    parser.add_argument("--campaign", type=Path, default=repo / DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "question_01_recall_at_10"
    )
    parser.add_argument("--cohort", action="append", choices=COHORTS)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=["dense", "translate-dense", "hyde-dense", "bm25", "hybrid"],
    )
    parser.add_argument("--corpus", action="append", choices=CORPORA)
    parser.add_argument("--model", action="append", choices=MODELS + ["none"])
    args = parser.parse_args()
    args.cohort = args.cohort or COHORTS
    args.artifacts, args.campaign, args.output = (
        args.artifacts.resolve(),
        args.campaign.resolve(),
        args.output.resolve(),
    )
    selected, sources = latest_reports(args.campaign, args.artifacts)
    settings, summary, catalog, inputs, validation = extract(args, selected)
    for subfolder in ("tables", "figures"):
        (args.output / subfolder).mkdir(parents=True, exist_ok=True)
    settings.to_csv(args.output / "tables/recall_at_10_by_setting.csv", index=False)
    summary.to_csv(args.output / "tables/recall_at_10_by_regime.csv", index=False)
    pd.DataFrame(catalog).to_csv(args.output / "tables/report_catalog.csv", index=False)
    for regime in REGIMES:
        rows = summary[summary.cohort.eq(COHORTS[0]) & summary.regime.eq(regime)]
        if not rows.empty:
            rows.pivot(
                index=["corpus", "model"], columns="variant", values="recall_at_10_pct"
            ).reindex(columns=[v for v in VARIANTS if v in rows.variant.unique()]).to_csv(
                args.output / f"tables/selected_documents_{regime}_percent.csv"
            )
    gallery = figures(settings, summary, args.output)
    write_gallery(args.output, gallery, summary)
    guide = [
        "QUESTION 1: RECALL@10",
        "",
        "All values below are percentages.",
        "Full breakdowns and matching-control deltas are in tables/; figure gallery: index.html.",
        "Selected-document and earlier all-question/chunk reports are separate cohorts.",
        "Mixed recall measures version coverage, not the probability of finding any source.",
        "All summaries use equal weights per nonempty setting, matching the saved reports.",
        "Generated de/en/es/fr/zh inputs remain five separate runs; no language fusion or best-language selection.",
        "Regime labels compare ORIGINAL query language to searched language, not generated language.",
        "Single-target language coverage: EUR-Lex has no zh targets; UN has no de targets.",
        "Grey cells have no eligible queries; zero is an observed retrieval score. Chemistry has cells with n=1.",
        "No corpus-level or across-model grand mean is used. Figures label their denominators.",
        "Changes are descriptive; no uncertainty intervals or significance claims are made.",
        "This analysis uses saved results only and incurs no inference or paid model calls.",
        "",
    ]
    guide += [
        "REPORT AND ELIGIBLE-QUERY COUNTS",
        pd.DataFrame(catalog)
        .groupby(["cohort", "corpus"])
        .agg(reports=("report_id", "size"), unique_questions=("unique_query_count", "first"))
        .to_string(),
        "",
    ]
    for regime in REGIMES:
        rows = summary[summary.cohort.eq(COHORTS[0]) & summary.regime.eq(regime)]
        if not rows.empty:
            guide += [
                f"SELECTED DOCUMENTS: {regime}",
                rows.pivot(index=["corpus", "model"], columns="variant", values="recall_at_10_pct")
                .reindex(columns=[v for v in VARIANTS if v in rows.variant.unique()])
                .to_string(float_format="%.2f"),
                "",
            ]
    (args.output / "results.txt").write_text("\n".join(guide) + "\n")
    write_json(args.output / "validation.json", validation)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "methodology": "Offline per-query Recall@10; equal setting means; validate against report and MTEB; separate cohorts; matched controls only.",
        "selection": {k: getattr(args, k) for k in ("cohort", "baseline", "corpus", "model")},
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_sha256": {**sources, **inputs, str(Path(__file__).resolve()): digest(__file__)},
        "figures": gallery,
        "output_sha256": {
            str(p.relative_to(args.output)): digest(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(validation, indent=2))
    print(f"Created {len(gallery)} figures in PNG/PDF/SVG: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
