"""Combine the saved November 20 evaluation tables without rerunning evaluation."""

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    folders = {
        "narval-20261119-shared-indexes": {"bm25": 3, "dense": 21, "hybrid": 20},
        "narval-20261120-transformed-retrieval": {
            "hyde-dense": 24,
            "hyde-hybrid": 24,
            "translate-dense": 136,
            "translate-hybrid": 136,
        },
    }
    rows = []
    fields = ["campaign", "run_id", "report_dir", "report_status", "translation_language"]
    for campaign, expected in folders.items():
        counts = Counter()
        reports = ROOT / "artifacts" / campaign / "artifacts" / "reports"
        for report in sorted(reports.glob("*/*/report.json")):
            directory = report.parent
            run = json.loads(report.read_text())["run"]
            manifest = directory / "analysis_manifest.json"
            status = "incomplete"
            if manifest.is_file():
                info = json.loads(manifest.read_text())
                if (info["run_id"], info["baseline"]) != (directory.name, run["baseline"]):
                    raise ValueError(f"Report identity mismatch in {directory}")
                counts[run["baseline"]] += 1
                status = "published"
            with (directory / "settings.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["baseline"] != run["baseline"] or row["corpus"] != run["corpus"]:
                        raise ValueError(f"Settings identity mismatch in {directory}")
                    row.update(
                        campaign=campaign,
                        run_id=directory.name,
                        report_status=status,
                        report_dir=directory.relative_to(ROOT).as_posix(),
                        translation_language=run.get("translation_language"),
                    )
                    rows.append(row)
                    fields.extend(key for key in row if key not in fields)
        if counts != expected:
            raise ValueError(f"Incomplete local report set in {campaign}: {counts}")
    target = HERE / "retrieval_settings.csv"
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    print(f"{len(rows):,} settings from {len({r['run_id'] for r in rows})} reports")


if __name__ == "__main__":
    main()
