"""Extract 30 reproducibly selected source records and all their published editions."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from clir_benchmark.config import load_config
from clir_benchmark.util import file_digest, new_directory, read_json, write_json

SOURCES = {
    "chemistry": "patents-verified-20261118/release-with-abstracts",
    "eurlex": "verified-legal-20261118/eurlex",
    "un": "verified-legal-20261118/un",
}
STRATA = {
    "chemistry": "de en es fr ja ru zh ja ru zh",
    "eurlex": "de en es fr de en es fr de en",
    "un": "ar en es fr ru zh ar en ru zh",
}


def choose(corpus, evidence, seed):
    chosen = []
    for slot, language in enumerate(STRATA[corpus].split()):
        pool = [
            e
            for e in evidence
            if e["language"] == language
            and e["corpus_id"] not in chosen
            and e["match_status"] == "exact"
        ]
        # Include an annex and a long Japanese patent as explicit stress cases.
        if corpus == "eurlex" and slot == 0:
            pool = [e for e in pool if e["config"] == "annexes"]
        if corpus == "chemistry" and slot == 4:
            pool = [e for e in pool if len(e["text"]) >= 1000]
        ids = {e["corpus_id"] for e in pool}
        chosen.append(
            min(
                ids,
                key=lambda key: hashlib.sha256(
                    f"{seed}/{corpus}/{slot}/{key}".encode()
                ).hexdigest(),
            )
        )
    return chosen


def extract(root, out, config, seed):
    with new_directory(out) as temp:
        manifest = {"seed": seed, "documents_per_corpus": 10, "sources": {}}
        for corpus, folder in SOURCES.items():
            base = root / folder / "data"
            release_files = read_json(base.parent / "manifest.json")["files"]
            for relative, metadata in release_files.items():
                if (
                    relative.startswith(("data/corpus/", "data/annexes/", "data/evidence/"))
                    and file_digest(base.parent / relative) != metadata["sha256"]
                ):
                    raise ValueError(f"Release hash mismatch: {corpus}/{relative}")
            evidence = pq.read_table(base / "evidence")
            selection = choose(corpus, evidence.to_pylist(), seed)
            selected = ds.field("_id").isin(selection)
            counts, found = Counter(), set()
            for section in ["corpus", "annexes"] if corpus == "eurlex" else ["corpus"]:
                table = ds.dataset(base / section, format="parquet").to_table(filter=selected)
                records = table.to_pylist()
                found.update(row["_id"] for row in records)
                for row in records:
                    counts.update(row["available_languages"])
                if corpus == "chemistry":
                    # The pilot is the induced subcorpus; pair endpoints outside it
                    # cannot be looked up. Source IDs, languages and text stay exact.
                    edition_ids = {v for r in records for v in r["source_unit_ids"].values() if v}
                    for row in records:
                        for lang, partners in row["paired_document_ids"].items():
                            if partners is not None:
                                row["paired_document_ids"][lang] = [
                                    v for v in partners if v in edition_ids
                                ]
                    table = pa.Table.from_pylist(records, schema=table.schema)
                target = temp / corpus / "data" / section / "sample.parquet"
                target.parent.mkdir(parents=True)
                pq.write_table(table, target, compression="zstd")
            if found != set(selection):
                raise ValueError(f"Missing {corpus} records: {set(selection) - found}")
            spans = evidence.filter(ds.field("corpus_id").isin(selection))
            dest = temp / corpus / "data/evidence/sample.parquet"
            dest.parent.mkdir(parents=True)
            pq.write_table(spans, dest, compression="zstd")
            manifest["sources"][corpus] = {
                "source": config.datasets[corpus].corpus.repo_id,
                "revision": config.datasets[corpus].corpus.revision,
                "ids": selection,
                "edition_counts": dict(counts),
                "evidence_rows": len(spans),
            }
        manifest["sha256"] = {
            str(p.relative_to(temp)): file_digest(p) for p in sorted(temp.rglob("*.parquet"))
        }
        write_json(temp / "selection.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", default=Path(__file__).with_name("tokenizers.yaml"))
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    extract(args.hf_export, args.output, load_config(args.config), args.seed)
