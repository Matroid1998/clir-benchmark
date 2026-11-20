"""Check published snippet offsets against Narval's actual chunk boundaries, without retrieval."""

import json
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "/scratch/matroid/clir-benchmark/runs/narval-20261119-shared-indexes"


def read_boundaries(corpus, editions):
    # A single sequential scan of three numeric/ID columns avoids reading passage text.
    scan = """
import json, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
root, corpus = sys.argv[1:]
receipt, = (Path(root) / 'job_status').glob(f'chunks-{corpus}-*.done')
directory = Path(receipt.read_text().splitlines()[0])
wanted = pa.array(json.load(sys.stdin))
result = {}
for batch in pq.ParquetFile(directory / 'chunks.parquet').iter_batches(columns=['source_unit_id','start','end']):
    for row in batch.filter(pc.is_in(batch.column(0), value_set=wanted)).to_pylist():
        result.setdefault(row['source_unit_id'], []).append([row['start'],row['end']])
print(json.dumps(result))
"""
    cmd = "source /scratch/matroid/clir-benchmark/env.sh && " + shlex.join(
        ["python", "-c", scan, RUN, corpus]
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "narval", cmd],
        input=json.dumps(sorted(editions)),
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def inspect(name, release):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    qa = hf_hub_download(
        release["repo"],
        "data/qas/train-00000.parquet",
        repo_type="dataset",
        revision=release["updated_revision"],
        local_files_only=True,
    )
    records = pq.read_table(qa, columns=["_id", "evidence_all_languages"]).to_pylist()
    corpus = "chemistry" if name == "google-patents" else name
    items = [
        (q["_id"], lang, s)
        for q in records
        for lang, snippets in q["evidence_all_languages"].items()
        for s in snippets or []
    ]
    bounds = read_boundaries(corpus, {s["source_unit_id"] for _, _, s in items})
    groups = defaultdict(list)
    counts = Counter()
    for qid, lang, s in items:
        intervals = bounds[s["source_unit_id"]]
        starts = [s["start"]] if s["start"] is not None else s["occurrence_starts"]
        complete = False
        for start in starts:
            end = start + len(s["text"])
            covered = start
            for left, right in sorted(intervals):
                if left <= start and end <= right:
                    complete = True
                if left <= covered < right:
                    covered = right
            if covered < end:
                raise ValueError(f"Incomplete source coverage: {qid} {lang}")
        counts["contained" if complete else "split_only"] += 1
        groups[qid, lang].append(complete)
    return dict(
        questions=len(records),
        snippets=len(items),
        **counts,
        question_language_pairs=len(groups),
        pairs_without_a_containing_chunk=sum(not any(flags) for flags in groups.values()),
    )


if __name__ == "__main__":
    releases = json.loads(
        (HERE.parent / "question_17_multilingual_evidence/summary.json").read_text()
    )
    report = {name: inspect(name, release) for name, release in releases.items()}
    (HERE / "boundary_coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
