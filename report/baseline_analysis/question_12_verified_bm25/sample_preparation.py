"""Run on Narval with the verified BM25 campaign directory as the argument."""

import argparse
import json
from datetime import datetime
from pathlib import Path

EDITIONS = {"chemistry": 574_974, "eurlex": 395_525, "un": 88_134}


def sample(campaign, corpus, total, now):
    root = campaign / "artifacts" / "corpus" / corpus
    lookups = list(root.glob("*/lookup.sqlite"))
    if not lookups:
        return {"status": "no corpus database yet"}
    if len(lookups) != 1:
        raise ValueError(f"Found multiple preparations in {root}")
    db = lookups[0]
    manifest = db.with_name("manifest.json")
    if manifest.is_file():
        completed = json.loads(manifest.read_text())
        return {
            "status": "prepared",
            "editions": completed["source_count"],
            "chunks": completed["chunk_count"],
        }

    # CorpusStore creates five tables/indexes in separate transactions.
    # prepare_corpus then commits once per 1,000 editions. Read only the
    # rollback-journal database header so the poll never locks the writer.
    with db.open("rb") as stream:
        header = stream.read(100)
    if len(header) != 100 or header[:16] != b"SQLite format 3\0" or header[18:20] != b"\x01\x01":
        return {"status": "counter unavailable"}
    pair_sink = db.with_name("upstream_pairs.parquet")
    if not pair_sink.exists() or pair_sink.stat().st_size != 4:
        return {"status": "outside edition-writing stage"}
    editions = max(0, int.from_bytes(header[24:28], "big") - 5) * 1_000
    # The verified releases never write raw pairs. While preparing editions,
    # this four-byte Parquet header retains the stage's initial timestamp.
    elapsed = now.timestamp() - pair_sink.stat().st_mtime
    if not 0 < editions < total or elapsed <= 0:
        return {"status": "awaiting first batch or finalizing"}
    return {
        "status": "preparing chunks",
        "estimated_editions_written": editions,
        "total_editions": total,
        "percent_editions": round(100 * editions / total, 1),
        "minutes_writing_editions": round(elapsed / 60, 1),
        "estimated_remaining_hours_at_average_edition_rate": round(
            elapsed / 3600 * (total / editions - 1), 2
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    now = datetime.now().astimezone()
    print(
        json.dumps(
            {
                "sampled_at": now.isoformat(timespec="seconds"),
                "corpora": {c: sample(args.campaign, c, n, now) for c, n in EDITIONS.items()},
            },
            indent=2,
        )
    )
