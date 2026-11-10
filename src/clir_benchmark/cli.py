from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from . import __version__
from .adapters import (
    DatasetReader,
    apply_evidence_overlay,
    canonical_source_ids,
    query_from_row,
    select_candidates,
)
from .adore_pipeline import run_adore
from .artifacts import CorpusStore
from .comparison import compare_adore, compare_hyde, compare_translations
from .config import (
    BASELINES,
    CANDIDATE_MODES,
    RERANK_BASELINES,
    language_tag,
    load_config,
    validate_retrieval_baselines,
)
from .encoders import load_registry
from .errors import ValidationError
from .hyde import hyde_languages, prepare_hyde
from .lexical import prepare_bm25
from .pipeline import embed, evaluate, prepare_corpus, prepare_relevance, retrieve
from .rerank_pipeline import compare_rerank, rerank
from .translation import prepare_translations, translation_languages
from .util import hf_token, read_json


def retrieval_backend(baseline: str) -> str:
    return baseline.removeprefix("rerank-").removeprefix("translate-").removeprefix("adore-")


def _selection_arguments(command: argparse.ArgumentParser) -> None:
    # Suppressed defaults let these options work before or after the subcommand.
    command.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default=argparse.SUPPRESS)
    command.add_argument("--selection-seed", type=int, default=argparse.SUPPRESS)
    command.add_argument(
        "--retrieval-unit", choices=("chunk", "document"), default=argparse.SUPPRESS
    )


def _validation_source_ids(config, corpus, queries):
    required = {query.source_unit_id for query in queries}
    lookup = config.corpus_dir(corpus) / "lookup.sqlite"
    if lookup.is_file():
        with CorpusStore(lookup) as store:
            return {
                unit_id: unit.source_id
                for unit_id in sorted(required)
                if (unit := store.get_source(unit_id)) is not None
            }
    reader = DatasetReader(config.datasets[corpus].corpus, hf_token(config.env_file))
    return canonical_source_ids(reader, corpus, required)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Static CLIR benchmark. Preparation never runs retrieval or evaluation."
    )
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="YAML configuration; relative paths resolve beside this file",
    )
    _selection_arguments(result)
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate",
        help="Validate configuration; optionally inspect QA inputs without running models",
    )
    validate.add_argument(
        "--inputs",
        action="store_true",
        help="Read QA exports and validate candidate selection and overlay IDs",
    )
    sub.add_parser("list-models", help="Print the resolved model registry without loading models")
    subset = sub.add_parser(
        "prepare-subset",
        help="Prepare a deterministic small local dataset; no model inference or paid calls",
    )
    subset.add_argument("--corpus", required=True, choices=["eurlex", "un"])
    subset.add_argument("--max-questions", type=int, default=2)
    subset.add_argument("--max-sources", type=int, default=12)
    subset.add_argument("--seed", type=int, default=0)
    subset.add_argument("--destination", type=str)
    smoke = sub.add_parser(
        "smoke",
        help="Explicitly run selected baselines and their dependencies on a tiny prepared subset (may make paid calls)",
    )
    smoke.add_argument("--corpus", required=True, choices=["eurlex", "un", "chemistry"])
    smoke.add_argument(
        "--model", action="append", help="One embedding model alias; default: configured model"
    )
    smoke.add_argument("--baseline", action="append", choices=list(BASELINES))
    smoke.add_argument("--translation-language", action="append")
    smoke.add_argument("--hyde-language", action="append")
    smoke.add_argument("--retry-failed", action="store_true")
    smoke.add_argument("--max-queries", type=int, default=5)
    smoke.add_argument("--max-sources", type=int, default=40)
    smoke.add_argument("--max-chunks", type=int, default=200)
    smoke.add_argument("--adore-rounds", type=int, choices=[1, 2, 3])
    smoke.add_argument("--rerank-candidates", type=int)
    for name in (
        "prepare-corpus",
        "prepare-relevance",
        "index-bm25",
        "embed",
        "retrieve",
        "evaluate",
        "translate",
        "compare-translations",
        "generate-hyde",
        "compare-hyde",
        "run-adore",
        "compare-adore",
        "rerank",
        "compare-rerank",
    ):
        command = sub.add_parser(name)
        command.add_argument(
            "--corpus",
            action="append",
            choices=["eurlex", "un", "chemistry"],
            help="Repeat to select corpora; default: configured corpora",
        )
        if name in {
            "embed",
            "retrieve",
            "evaluate",
            "compare-translations",
            "compare-hyde",
            "run-adore",
            "compare-adore",
            "rerank",
            "compare-rerank",
        }:
            command.add_argument(
                "--model",
                action="append",
                help="Repeat to select model aliases; default: configured models",
            )
        if name == "embed":
            command.add_argument("--kind", choices=["document", "query", "both"], default="both")
        if name in {
            "translate",
            "embed",
            "retrieve",
            "evaluate",
            "compare-translations",
            "rerank",
            "compare-rerank",
        }:
            command.add_argument(
                "--translation-language",
                action="append",
                help="Repeat to select independent translated-query variants; translate and translate-* baselines default to all configured target languages",
            )
        if name in {
            "generate-hyde",
            "embed",
            "retrieve",
            "evaluate",
            "compare-hyde",
            "rerank",
            "compare-rerank",
        }:
            command.add_argument(
                "--hyde-language",
                action="append",
                help="Repeat to select independent HyDE output languages; defaults to hyde.output_languages or target languages for HyDE commands/baselines",
            )
        if name == "compare-hyde":
            command.add_argument(
                "--control",
                action="append",
                choices=["dense", "translate-dense"],
                help="Existing control run to compare against; repeat for both (default: dense)",
            )
        if name in {"translate", "generate-hyde", "run-adore"}:
            command.add_argument(
                "--retry-failed",
                action="store_true",
                help="Explicitly retry failed/interrupted requests; retains previous attempts for accounting",
            )
        if name in {
            "run-adore",
            "compare-adore",
            "retrieve",
            "evaluate",
            "rerank",
            "compare-rerank",
        }:
            command.add_argument(
                "--adore-rounds",
                type=int,
                choices=[1, 2, 3],
                help="Maximum expansion rounds for ADORE (default: adore.max_rounds, at most 3)",
            )
        if name in {"rerank", "compare-rerank", "retrieve", "evaluate"}:
            command.add_argument(
                "--rerank-candidates",
                type=int,
                help="Rerank the first N saved candidates (1-100; default: reranker.candidate_k)",
            )
        if name in {
            "retrieve",
            "evaluate",
            "compare-translations",
            "run-adore",
            "compare-adore",
            "rerank",
            "compare-rerank",
        }:
            command.add_argument(
                "--baseline",
                action="append",
                choices=list(RERANK_BASELINES)
                if name in {"rerank", "compare-rerank"}
                else ["dense", "bm25", "hybrid"]
                if name == "compare-translations"
                else [value for value in BASELINES if value.startswith("adore-")]
                if name in {"run-adore", "compare-adore"}
                else list(BASELINES),
                help="Repeat to select reranked baselines; defaults to configured rerank-* entries; at least one must be selected"
                if name in {"rerank", "compare-rerank"}
                else "Repeat to select ADORE baselines; defaults to configured adore-* entries; at least one must be selected"
                if name in {"run-adore", "compare-adore"}
                else "Repeat to select baselines; default: configured baselines (dense)",
            )
    for command in sub.choices.values():
        _selection_arguments(command)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if hasattr(args, "candidate_mode"):
            config = replace(config, candidate_mode=args.candidate_mode)
        if hasattr(args, "selection_seed"):
            config = replace(
                config,
                candidate_selection=replace(config.candidate_selection, seed=args.selection_seed),
            )
        if hasattr(args, "retrieval_unit"):
            config = replace(config, retrieval_unit=args.retrieval_unit)
        requested_models = getattr(args, "model", None) or config.models
        baselines = tuple(dict.fromkeys(getattr(args, "baseline", None) or config.baselines))
        if args.command == "prepare-subset":
            from pathlib import Path

            from .subset import prepare_subset

            print(
                prepare_subset(
                    config,
                    args.corpus,
                    max_questions=args.max_questions,
                    max_sources=args.max_sources,
                    seed=args.seed,
                    destination=Path(args.destination).resolve() if args.destination else None,
                )
                / "config.yaml"
            )
            return 0
        if args.command in {"rerank", "compare-rerank"}:
            baselines = tuple(value for value in baselines if value in RERANK_BASELINES)
            if not baselines:
                raise ValidationError(
                    "Select a reranked baseline with --baseline rerank-dense, rerank-bm25, etc., or enable it in baselines"
                )
        rerank_candidates = getattr(args, "rerank_candidates", None)
        if rerank_candidates is not None:
            if not any(value in RERANK_BASELINES for value in baselines):
                raise ValidationError("--rerank-candidates requires a rerank-* baseline")
            config = replace(
                config, reranker=replace(config.reranker, candidate_k=rerank_candidates)
            )
        if args.command in {"run-adore", "compare-adore"}:
            baselines = tuple(value for value in baselines if value.startswith("adore-"))
            if not baselines:
                raise ValidationError(
                    "Select an ADORE baseline with --baseline adore-bm25, adore-dense, or adore-hybrid, or enable it in baselines"
                )
        adore_rounds = getattr(args, "adore_rounds", None)
        if adore_rounds is not None:
            if not any(value.removeprefix("rerank-").startswith("adore-") for value in baselines):
                raise ValidationError("--adore-rounds requires an adore-* baseline")
            config = replace(config, adore=replace(config.adore, max_rounds=adore_rounds))
        if args.command == "compare-translations":
            baselines = tuple(
                dict.fromkeys(
                    value.removeprefix("translate-")
                    for value in baselines
                    if value != "hyde-dense" and not value.startswith(("adore-", "rerank-"))
                )
            ) or ("dense",)
        if args.command in {
            "validate",
            "smoke",
            "retrieve",
            "evaluate",
            "compare-translations",
            "compare-hyde",
            "run-adore",
            "compare-adore",
            "rerank",
            "compare-rerank",
        }:
            validate_retrieval_baselines(config, baselines)
        if args.command == "smoke":
            from .smoke import run_smoke

            config = replace(config, baselines=baselines)
            needs_encoder = any(retrieval_backend(value) != "bm25" for value in baselines)
            if needs_encoder and len(requested_models) != 1:
                raise ValidationError(
                    "smoke requires exactly one --model or one configured embedding model"
                )
            directory = run_smoke(
                config,
                args.corpus,
                requested_models[0] if needs_encoder else None,
                translation_languages=tuple(args.translation_language)
                if args.translation_language
                else None,
                hyde_languages=tuple(args.hyde_language) if args.hyde_language else None,
                retry_failed=args.retry_failed,
                max_queries=args.max_queries,
                max_sources=args.max_sources,
                max_chunks=args.max_chunks,
            )
            print(directory / "status.json")
            return 0 if read_json(directory / "status.json")["status"] == "success" else 2
        if args.command in {
            "run-adore",
            "compare-adore",
            "retrieve",
            "evaluate",
            "compare-translations",
            "rerank",
            "compare-rerank",
        }:
            config = replace(config, baselines=baselines)
        if (
            args.command
            in {
                "retrieve",
                "evaluate",
                "compare-translations",
                "run-adore",
                "compare-adore",
                "rerank",
                "compare-rerank",
            }
            and all(retrieval_backend(value) == "bm25" for value in baselines)
            and args.model
        ):
            raise ValidationError("--model applies to dense/hybrid retrieval; omit it for BM25")
        selected_languages = getattr(args, "translation_language", None)
        selected_hyde_languages = getattr(args, "hyde_language", None)
        for language in selected_hyde_languages or ():
            language_tag(language, "HyDE output language")
        if (
            args.command in {"retrieve", "evaluate", "rerank", "compare-rerank"}
            and selected_hyde_languages
            and not any(value.removeprefix("rerank-") == "hyde-dense" for value in baselines)
        ):
            raise ValidationError("--hyde-language requires the hyde-dense baseline")
        if args.command == "embed" and selected_hyde_languages:
            if selected_languages:
                raise ValidationError(
                    "Choose either --translation-language or --hyde-language for embed"
                )
            if args.kind == "document":
                raise ValidationError(
                    "--hyde-language applies to query-side passages; use --kind query or both"
                )
        if (
            args.command in {"retrieve", "evaluate", "rerank", "compare-rerank"}
            and selected_languages
            and not any(
                value.removeprefix("rerank-").startswith("translate-") for value in baselines
            )
        ):
            raise ValidationError("--translation-language requires a translate-* baseline")
        if args.command == "embed" and selected_languages and args.kind == "document":
            raise ValidationError(
                "--translation-language applies to query embeddings; use --kind query or both"
            )
        needs_model = args.command not in {
            "run-adore",
            "compare-adore",
            "retrieve",
            "evaluate",
            "compare-translations",
            "rerank",
            "compare-rerank",
        } or any(retrieval_backend(value) != "bm25" for value in baselines)
        registry = load_registry(config.model_registry) if needs_model else {}
        for alias in requested_models if needs_model else ():
            if alias not in registry:
                raise ValidationError(f"Unknown model alias: {alias}")
        if args.command == "list-models":
            from dataclasses import asdict

            print(json.dumps({name: asdict(spec) for name, spec in registry.items()}, indent=2))
            return 0
        if args.command == "validate":
            info = {
                "status": "valid",
                "corpora": list(config.datasets),
                "models": list(config.models),
                "candidate_mode": config.candidate_mode,
                "retrieval_unit": config.retrieval_unit,
                "baselines": list(config.baselines),
                "evaluation_executed": False,
            }
            if args.inputs:
                ids = set()
                counts = {}
                for corpus, dataset in config.datasets.items():
                    reader = DatasetReader(dataset.qa, hf_token(config.env_file))
                    queries = [query_from_row(row, corpus) for row in reader.read("qas")]
                    queries = apply_evidence_overlay(queries, config.evidence_file)
                    selected, selection_audit = select_candidates(
                        queries,
                        config.candidate_mode,
                        source_ids=_validation_source_ids(config, corpus, queries)
                        if config.candidate_mode == "best-per-document"
                        else None,
                        policy=config.candidate_selection,
                    )
                    ids.update(q.id for q in queries)
                    counts[corpus] = {
                        "input_queries": len(queries),
                        "selected_queries": len(selected),
                    }
                    if config.candidate_mode == "best-per-document":
                        counts[corpus].update(
                            canonical_documents=len(selection_audit),
                            excluded_unscored_documents=sum(
                                not row["selected_ids"] for row in selection_audit
                            ),
                            excluded_unscored_candidates=sum(
                                row["excluded_missing_score_count"] for row in selection_audit
                            ),
                        )
                if config.evidence_file:
                    seen = set()
                    with config.evidence_file.open(encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            query_id = record.get("query_id", record.get("_id"))
                            if query_id not in ids or query_id in seen:
                                raise ValidationError(
                                    f"Unknown or duplicate QA ID in evidence overlay: {query_id}"
                                )
                            seen.add(query_id)
                info["inputs"] = counts
            print(json.dumps(info, indent=2))
            return 0
        corpora = args.corpus or list(config.datasets)
        blocked = False
        for corpus in corpora:
            if corpus not in config.datasets:
                raise ValidationError(f"Corpus {corpus} is not configured")
            if args.command == "prepare-corpus":
                print(prepare_corpus(config, corpus))
            elif args.command == "prepare-relevance":
                path = prepare_relevance(config, corpus)
                print(path)
                missing = read_json(path / "manifest.json")["blocked_settings"]
                if missing:
                    blocked = True
                    print(
                        f"{corpus}: {len(missing)} settings blocked by missing multilingual annotations; see settings.json and eligibility.parquet",
                        file=sys.stderr,
                    )
            elif args.command == "index-bm25":
                print(prepare_bm25(config, corpus))
            elif args.command in {"run-adore", "compare-adore"}:
                for baseline in baselines:
                    for alias in (None,) if baseline == "adore-bm25" else requested_models:
                        if args.command == "run-adore":
                            print(
                                run_adore(
                                    config,
                                    corpus,
                                    alias,
                                    baseline=baseline,
                                    retry_failed=args.retry_failed,
                                )
                            )
                        else:
                            print(compare_adore(config, corpus, alias, baseline=baseline))
            elif args.command == "translate":
                for language in dict.fromkeys(
                    selected_languages or translation_languages(config, corpus)
                ):
                    print(
                        prepare_translations(
                            config, corpus, language, retry_failed=args.retry_failed
                        )
                    )
            elif args.command == "generate-hyde":
                for language in dict.fromkeys(
                    selected_hyde_languages or hyde_languages(config, corpus)
                ):
                    print(prepare_hyde(config, corpus, language, retry_failed=args.retry_failed))
            elif args.command == "compare-hyde":
                for alias in requested_models:
                    print(
                        compare_hyde(
                            config,
                            corpus,
                            alias,
                            languages=tuple(
                                dict.fromkeys(
                                    selected_hyde_languages or hyde_languages(config, corpus)
                                )
                            ),
                            controls=tuple(dict.fromkeys(args.control or ["dense"])),
                        )
                    )
            elif args.command == "compare-translations":
                for baseline in baselines:
                    for alias in (None,) if baseline == "bm25" else requested_models:
                        print(
                            compare_translations(
                                config,
                                corpus,
                                alias,
                                baseline=baseline,
                                languages=tuple(
                                    dict.fromkeys(
                                        selected_languages or translation_languages(config, corpus)
                                    )
                                ),
                            )
                        )
            elif args.command == "embed":
                for alias in requested_models:
                    for language in dict.fromkeys(
                        selected_hyde_languages or selected_languages or [None]
                    ):
                        print(
                            json.dumps(
                                {
                                    kind: str(path)
                                    for kind, path in embed(
                                        config,
                                        corpus,
                                        alias,
                                        kind=args.kind,
                                        **(
                                            {"hyde_language": language}
                                            if selected_hyde_languages
                                            else {"translation_language": language}
                                        ),
                                    ).items()
                                }
                            )
                        )
            else:
                for baseline in baselines:
                    for alias in (
                        (None,) if retrieval_backend(baseline) == "bm25" else requested_models
                    ):
                        operation = {
                            "retrieve": retrieve,
                            "evaluate": evaluate,
                            "rerank": rerank,
                            "compare-rerank": compare_rerank,
                        }[args.command]
                        parent_baseline = baseline.removeprefix("rerank-")
                        if parent_baseline == "hyde-dense":
                            for language in dict.fromkeys(
                                selected_hyde_languages or hyde_languages(config, corpus)
                            ):
                                print(
                                    operation(
                                        config,
                                        corpus,
                                        alias,
                                        baseline=baseline,
                                        hyde_language=language,
                                    )
                                )
                        elif parent_baseline.startswith("translate-"):
                            for language in dict.fromkeys(
                                selected_languages or translation_languages(config, corpus)
                            ):
                                print(
                                    operation(
                                        config,
                                        corpus,
                                        alias,
                                        baseline=baseline,
                                        translation_language=language,
                                    )
                                )
                        else:
                            print(operation(config, corpus, alias, baseline=baseline))
        return 2 if blocked else 0
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
