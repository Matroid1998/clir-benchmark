from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .adapters import DatasetReader, apply_evidence_overlay, query_from_row, select_candidates
from .comparison import compare_hyde, compare_translations
from .config import BASELINES, language_tag, load_config
from .encoders import load_registry
from .errors import ValidationError
from .hyde import hyde_languages, prepare_hyde
from .lexical import prepare_bm25
from .pipeline import embed, evaluate, prepare_corpus, prepare_relevance, retrieve
from .translation import prepare_translations, translation_languages
from .util import hf_token, read_json


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
    ):
        command = sub.add_parser(name)
        command.add_argument(
            "--corpus",
            action="append",
            choices=["eurlex", "un", "chemistry"],
            help="Repeat to select corpora; default: configured corpora",
        )
        if name in {"embed", "retrieve", "evaluate", "compare-translations", "compare-hyde"}:
            command.add_argument(
                "--model",
                action="append",
                help="Repeat to select model aliases; default: configured models",
            )
        if name == "embed":
            command.add_argument("--kind", choices=["document", "query", "both"], default="both")
        if name in {"translate", "embed", "retrieve", "evaluate", "compare-translations"}:
            command.add_argument(
                "--translation-language",
                action="append",
                help="Repeat to select independent translated-query variants; translate and translate-* baselines default to all configured target languages",
            )
        if name in {"generate-hyde", "embed", "retrieve", "evaluate", "compare-hyde"}:
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
        if name in {"translate", "generate-hyde"}:
            command.add_argument(
                "--retry-failed",
                action="store_true",
                help="Explicitly retry failed/interrupted requests; retains previous attempts for accounting",
            )
        if name in {"retrieve", "evaluate", "compare-translations"}:
            command.add_argument(
                "--baseline",
                action="append",
                choices=list(BASELINES)
                if name != "compare-translations"
                else ["dense", "bm25", "hybrid"],
                help="Repeat to select baselines; default: configured baselines (dense)",
            )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        registry = load_registry(config.model_registry)
        requested_models = getattr(args, "model", None) or config.models
        baselines = tuple(dict.fromkeys(getattr(args, "baseline", None) or config.baselines))
        if args.command == "compare-translations":
            baselines = tuple(
                dict.fromkeys(
                    value.removeprefix("translate-") for value in baselines if value != "hyde-dense"
                )
            ) or ("dense",)
        if (
            args.command in {"retrieve", "evaluate", "compare-translations"}
            and all(value.removeprefix("translate-") == "bm25" for value in baselines)
            and args.model
        ):
            raise ValidationError("--model applies to dense/hybrid retrieval; omit it for BM25")
        selected_languages = getattr(args, "translation_language", None)
        selected_hyde_languages = getattr(args, "hyde_language", None)
        for language in selected_hyde_languages or ():
            language_tag(language, "HyDE output language")
        if (
            args.command in {"retrieve", "evaluate"}
            and selected_hyde_languages
            and "hyde-dense" not in baselines
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
            args.command in {"retrieve", "evaluate"}
            and selected_languages
            and not any(value.startswith("translate-") for value in baselines)
        ):
            raise ValidationError("--translation-language requires a translate-* baseline")
        if args.command == "embed" and selected_languages and args.kind == "document":
            raise ValidationError(
                "--translation-language applies to query embeddings; use --kind query or both"
            )
        for alias in requested_models:
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
                    selected, _ = select_candidates(queries, config.candidate_mode)
                    ids.update(q.id for q in queries)
                    counts[corpus] = {
                        "input_queries": len(queries),
                        "selected_queries": len(selected),
                    }
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
                        (None,)
                        if baseline.removeprefix("translate-") == "bm25"
                        else requested_models
                    ):
                        operation = retrieve if args.command == "retrieve" else evaluate
                        if baseline == "hyde-dense":
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
                        elif baseline.startswith("translate-"):
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
