"""Fetch primary-source bibliographic metadata only; never run a benchmark."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


SOURCES = [
    ("hyde", "https://aclanthology.org/2023.acl-long.99.bib"),
    ("query2doc", "https://aclanthology.org/2023.emnlp-main.585.bib"),
    ("expansion_failures", "https://aclanthology.org/2024.findings-eacl.134.bib"),
    ("lareqa", "https://aclanthology.org/2020.emnlp-main.477.bib"),
    ("language_bias", "https://aclanthology.org/2024.mrl-1.23.bib"),
    ("delp_delta", "https://aclanthology.org/2026.findings-acl.1353.bib"),
    ("query_mixing", "https://aclanthology.org/2026.acl-long.1455.bib"),
    ("multilingual_expansion", "https://arxiv.org/bibtex/2511.19325"),
    ("clirudit", "https://aclanthology.org/2025.mrl-main.16.bib"),
    ("complementary_rewriting", "https://aclanthology.org/2026.emnlp-industry.143.bib"),
    ("mugi", "https://aclanthology.org/2024.findings-emnlp.103.bib"),
    ("mmteb", "https://arxiv.org/bibtex/2502.13595"),
]


def fetch(item):
    label, url = item
    with urlopen(url, timeout=40) as response:
        raw = response.read().decode("utf-8")
    # The Anthology places its full abstract last; keep bibliographic fields,
    # not a reproduction of the abstract in the local paper's bibliography.
    marker = re.search(r"\n\s*abstract\s*=", raw)
    if marker:
        raw = raw[:marker.start()].rstrip().rstrip(",") + "\n}\n"
    if not raw.lstrip().startswith("@") or not raw.rstrip().endswith("}"):
        raise ValueError(f"Unexpected bibliographic response: {url}")
    key = re.search(r"@\w+\s*\{\s*([^,]+),", raw).group(1)
    return raw.strip() + "\n", {
        "label": label,
        "citation_key": key,
        "metadata_url": url,
        "metadata_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def main():
    root = Path(__file__).resolve().parent
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, SOURCES))
    keys = [metadata["citation_key"] for _, metadata in results]
    if len(keys) != len(set(keys)) or len(keys) != 12:
        raise ValueError("Expected twelve unique primary-source references")
    (root / "references.bib").write_text(
        "% Primary-source citation metadata, fetched 2026-11-15.\n\n"
        + "\n".join(text for text, _ in results),
        encoding="utf-8",
    )
    (root / "retrieved_sources.json").write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Focused primary-source review; not an exhaustive novelty guarantee",
                "references": [metadata for _, metadata in results],
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(results)} primary-source bibliography entries")


if __name__ == "__main__":
    main()
