"""Reproduce local analysis only; never invokes a benchmark, model or provider."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=range(1, 7), action="append",
                        dest="rounds", help="Repeat to select rounds; default is all six.")
    parser.add_argument("--rebuild-data", action="store_true",
                        help="Read the original local result artifacts again.")
    parser.add_argument("--list", action="store_true",
                        help="Print the execution order without running analyses.")
    args = parser.parse_args()
    selected = sorted(set(args.rounds or range(1, 7)))
    scripts = []
    if args.rebuild_data:
        scripts.extend(SCRIPTS / name for name in
                       ("build_dataset.py", "build_mixed_rankings.py"))
    for number in selected:
        matches = sorted(SCRIPTS.glob(f"round{number:02d}_*.py"))
        if len(matches) != 1:
            parser.error(f"Expected one Round {number} script, found {len(matches)}")
        scripts.extend(matches)
    for script in scripts:
        command = [sys.executable, str(script)]
        if script.name == "build_dataset.py":
            command.extend(["--output", str(ROOT / "data")])
        print(shlex.join(command), flush=True)
        if not args.list:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
