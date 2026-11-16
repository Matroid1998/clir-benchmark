#!/usr/bin/env python3
"""Question 9: verify chunk-to-document aggregation and clarify length effects.

Reads saved contracts and an inspected case. Toy scores illustrate the maximum
operation only; they are not benchmark measurements or a new retrieval run.
"""

import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "question_09_chunking_and_length"
    output.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "question_01_recall_at_10/tables/report_catalog.csv"
    with catalog_path.open() as handle:
        entries = [
            r
            for r in csv.DictReader(handle)
            if r["cohort"] == "selected_documents"
            and r["baseline"] == "dense"
            and r["model"] == "embeddinggemma"
        ]
    contracts, sources = [], [catalog_path, Path(__file__)]
    for entry in entries:
        path = Path(entry["report_path"])
        report = json.loads(path.read_text())
        run = report["run"]
        if (
            run["retrieval_unit"] != "document"
            or run["document_scoring"] != "maximum_chunk_similarity_before_top_k"
        ):
            raise ValueError("Expected document ranking by maximum chunk similarity")
        contracts.append(
            {
                "corpus": entry["corpus"],
                "document_scoring": run["document_scoring"],
                "retrieval_unit": run["retrieval_unit"],
                "document_identity": run["document_identity"],
                "truncated_document_chunks": report["truncation"]["documents"]["all"]["truncated"],
                "report_path": str(path),
            }
        )
        sources.append(path)
    if len(contracts) != 3:
        raise ValueError("Expected the three corpus controls")
    case_path = (
        root
        / "question_08_chemistry_difficulty/cases/un/q_204bb5965a47b606176a76657963071a20a1bdbbc496e939a4f23ee2b337ff34.json"
    )
    case = json.loads(case_path.read_text())
    sources.append(case_path)
    positive = next(r for r in case["saved_top3_and_first_positive"] if r["relevance"] > 0)
    example = {
        "question": case["query"]["question"],
        "relevant_source_id": case["relevant_source"]["id"],
        "source_header": case["relevant_source"]["text"][:100],
        "document_rank": positive["rank"],
        "winning_chunk_start_character": positive["start"],
        "winning_chunk_contains_2011": "2011" in positive["text"],
        "winning_chunk_text": positive["text"],
        "interpretation": "The document header supplies the year, but this winning chunk does not. This illustrates context fragmentation, not proof that missing header context caused the ranking failure.",
    }
    toy = [
        {
            "scenario": "Initial chunks",
            "relevant_chunks": [0.80, 0.20],
            "distractor_chunks": [0.75],
        },
        {
            "scenario": "More low-scoring relevant chunks",
            "relevant_chunks": [0.80, 0.20, 0.01, 0.03],
            "distractor_chunks": [0.75],
        },
        {
            "scenario": "More chunks help the relevant document",
            "relevant_chunks": [0.80, 0.20, 0.90],
            "distractor_chunks": [0.75],
        },
        {
            "scenario": "A high-scoring distractor chunk changes the order",
            "relevant_chunks": [0.80, 0.20],
            "distractor_chunks": [0.75, 0.85],
        },
    ]
    for row in toy:
        row["relevant_document_score"] = max(row["relevant_chunks"])
        row["distractor_document_score"] = max(row["distractor_chunks"])
        row["relevant_ranks_first"] = (
            row["relevant_document_score"] > row["distractor_document_score"]
        )
    data = {
        "status": "passed",
        "saved_contracts": contracts,
        "inspected_example": example,
        "synthetic_illustrations_not_measurements": toy,
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    (output / "method_verification.json").write_text(json.dumps(data, indent=2) + "\n")
    explanation = """CHUNKING AND DOCUMENT LENGTH

The selected dense experiments rank each document-language version by the maximum
query similarity of ALL its chunks, then take the top unique versions. They do
not average chunks into a diluted whole-document score. One long version cannot
occupy multiple top-k positions. The saved contracts confirm this for all three
EmbeddingGemma corpora, with zero truncated document chunks recorded.

Length can matter indirectly:
- More chunks provide more chances for a high maximum, for relevant documents
  and distractors alike. Adding chunks while retaining all existing scores cannot
  lower a document's own maximum. Its rank can fall if competitors' maxima rise.
- A chunk can omit the title, year, scope or definitions needed to interpret a
  passage. Overlap helps nearby boundaries; it does not attach a distant heading
  to every chunk, and maximum scoring does not combine evidence across chunks.
- More chunks increase embedding/search work. This is an efficiency consideration,
  not evidence that longer units caused lower retrieval quality.

The saved Taliban example's relevant document ranks 11. Its header says 2011;
its winning chunk starts at character 21,725 and contains no 2011. This shows
fragmented distinguishing context, without establishing the cause of its rank.

The earlier document-length association is a hypothesis about chunk count and
context, not a demonstrated explanation for the chemistry score advantage.
Establishing its contribution needs an explicitly selected comparison with
different chunk context/aggregation while preserving queries, qrels and controls.
No retrieval settings or benchmark outputs have been changed here.
"""
    (output / "explanation.txt").write_text(explanation)
    print(explanation)


if __name__ == "__main__":
    main()
