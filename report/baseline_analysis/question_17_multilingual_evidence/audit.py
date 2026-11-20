"""Compare published QA revisions and validate the retained snippets against corpus text."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT.parent / "clir-dataset-creation/data/hf_export"
RELEASES = {
    "google-patents": (
        "9459adf9bfdc4f1f14ad6072de856b9e4a0d30ed",
        "2d6a4966a3373d48ef0a30fe9687ac5de9700e87",
        "patents-verified-20261118/release-with-abstracts",
    ),
    "eurlex": (
        "e1e4e3fd1a9f8c15adf1dbbb239ba4ddc0317ede",
        "4fe4f78a146692a98ed9e8c4dae7c7ed7629ddf0",
        "verified-legal-20261118/eurlex",
    ),
    "un": (
        "910ee8435bc4e5a41ccd5f913c30ebc1237eb130",
        "c3c9415b0636748bb007f8734ab07d27b48faa74",
        "verified-legal-20261118/un",
    ),
}


def audit(name, old, new, mirror):
    repo = f"MehdiAstaraki/{name}-generated-qa-verified-20261118"

    def fetch(rev, filename):
        return Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=rev,
                token=dotenv_values(ROOT / ".env").get("HF_TOKEN"),
            )
        )

    def corpus_files(rev):
        files = json.loads(fetch(rev, "manifest.json").read_text())["files"]
        return {
            k: v["sha256"]
            for k, v in files.items()
            if k.startswith(("data/corpus/", "data/annexes/"))
        }

    files = corpus_files(new)
    previous = corpus_files(old)
    report = json.loads(fetch(new, "report.json").read_text())
    qa_file = "data/qas/train-00000.parquet"
    fields = ["_id", "question", "question_language", "answer"]
    before = {r["_id"]: r for r in pq.read_table(fetch(old, qa_file), columns=fields).to_pylist()}
    after = pq.read_table(
        fetch(new, qa_file), columns=fields + ["evidence_all_languages"]
    ).to_pylist()
    ids = {q["_id"] for q in after}
    assert len(ids) == len(after)
    needed = defaultdict(list)
    supports = defaultdict(set)
    covered = defaultdict(set)
    for q in after:
        for lang, snippets in q["evidence_all_languages"].items():
            for snippet in snippets or []:
                source = (snippet["config"], snippet["corpus_id"])
                needed[source].append((lang, snippet))
                covered[q["_id"]].add(lang)
                if snippet["is_original"]:
                    supports[q["_id"]].add(source)

    availability = {}
    counts = Counter()
    for filename, expected in sorted(files.items()):
        local = BASE / mirror / filename
        if not local.exists():
            local = fetch(new, filename)
        with local.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected
        lang_list = "abstract_languages" if name == "google-patents" else "available_languages"
        for batch in pq.ParquetFile(local).iter_batches(
            batch_size=128, columns=["_id", lang_list, "text", "source_unit_ids"]
        ):
            for i, doc_id in enumerate(batch.column(0).to_pylist()):
                source = (filename.split("/")[1], doc_id)
                if source not in needed:
                    continue
                availability[source] = set(batch.column(1)[i].as_py())
                doc = batch.column(2)[i].as_py()
                editions = batch.column(3)[i].as_py()
                for lang, s in needed[source]:
                    assert s["language"] == lang and s["source_unit_id"] == editions[lang]
                    text = s["text"]
                    if s["start"] is None:
                        starts = s["occurrence_starts"]
                        assert starts and s["end"] is None
                        counts["ambiguous_offsets"] += 1
                    else:
                        starts = [s["start"]]
                        assert s["end"] == s["start"] + len(text)
                    assert text and all(
                        start >= 0 and doc[lang][start : start + len(text)] == text
                        for start in starts
                    )
                    counts["original" if s["is_original"] else "aligned"] += 1
    assert set(availability) == set(needed)
    missing_coverage = 0
    language_counts = Counter()
    for qid in ids:
        assert supports[qid]
        required = set.union(*(availability[doc] for doc in supports[qid]))
        missing_coverage += int(bool(required - covered[qid]))
        language_counts[len(covered[qid])] += 1
    return {
        "repo": repo,
        "baseline_revision": old,
        "updated_revision": new,
        "old_questions": len(before),
        "new_questions": len(ids),
        "removed_ids": len(before.keys() - ids),
        "added_ids": len(ids - before.keys()),
        "changed_retained_question_language_or_answer": sum(
            any(q[k] != before[q["_id"]][k] for k in fields) for q in after if q["_id"] in before
        ),
        "final_filter": report["question_filter"],
        "alignment": report["evidence_alignment"],
        "identical_corpus_parquet": previous == files,
        "corpus_files_compared": len(files),
        "validated_snippets": dict(counts),
        "questions_missing_required_languages": missing_coverage,
        "questions_by_number_of_evidence_languages": dict(sorted(language_counts.items())),
    }


if __name__ == "__main__":
    results = {key: audit(key, *args) for key, args in RELEASES.items()}
    (HERE / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for key, result in results.items():
        print(
            key,
            result["new_questions"],
            result["identical_corpus_parquet"],
            result["validated_snippets"],
            result["questions_missing_required_languages"],
            result["changed_retained_question_language_or_answer"],
        )
