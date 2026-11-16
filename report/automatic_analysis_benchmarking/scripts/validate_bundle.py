"""Validate analysis deliverables, never recomputing retrieval or generation."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    rounds = json.loads((ROOT / 'rounds.json').read_text())['rounds']
    assert [row['round'] for row in rounds] == list(range(1, 7))
    figures = tables = hashes = 0
    for row in rounds:
        assert row['status'] == 'complete', row
        directory = ROOT / f"round_{row['round']:02d}"
        review = ROOT / row['review']
        assert review.stat().st_size > 300
        manifest = json.loads((ROOT / row['manifest']).read_text())
        assert 'complete' in str(manifest.get('visual_review_status',
                                            manifest.get('visual_review', '')))
        pngs = sorted(directory.glob('fig*.png'))
        csvs = sorted(directory.glob('table*.csv'))
        assert pngs and csvs
        for picture in pngs:
            with Image.open(picture) as loaded:
                assert loaded.width >= 600 and loaded.height >= 300
                loaded.verify()
            assert picture.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
            ET.parse(picture.with_suffix('.svg'))
            figures += 1
        for table in csvs:
            assert len(pd.read_csv(table)) > 0
            assert table.with_suffix('.tex').exists()
            tables += 1
        for name, expected in manifest.get('output_sha256', {}).items():
            assert sha256(directory / name) == expected, (directory.name, name)
            hashes += 1
        for name, expected in manifest.get('inputs', {}).items():
            if isinstance(expected, str) and len(expected) == 64:
                assert sha256(ROOT / name) == expected, ('input', directory.name, name)
                hashes += 1
        if 'script_sha256' in manifest:
            candidates = list((ROOT / 'scripts').glob(f"round{row['round']:02d}_*.py"))
            assert len(candidates) == 1
            assert sha256(candidates[0]) == manifest['script_sha256']
            hashes += 1
    validation = json.loads((ROOT / 'data/validation.json').read_text())
    assert validation['status'] == 'passed' and validation['unique_queries'] == 638
    assert len(pd.read_parquet(ROOT / 'data/report_catalog.parquet')) == 264
    assert len(pd.read_parquet(ROOT / 'data/generation.parquet')) == 6380
    assert len(pd.read_parquet(ROOT / 'data/per_query.parquet')) == 247192
    assert (ROOT / 'index.html').exists()
    paper = ROOT / 'paper'
    paper_manifest = json.loads((paper / 'figure_manifest.json').read_text())
    assert paper_manifest['visual_review_status'] == 'complete_root'
    for name, expected in paper_manifest['output_sha256'].items():
        assert sha256(paper / name) == expected, ('paper', name)
        hashes += 1
    for name, expected in paper_manifest['source_sha256'].items():
        assert sha256(ROOT / name) == expected, ('paper source', name)
        hashes += 1
    assert sha256(ROOT / 'scripts/make_paper_figures.py') == paper_manifest['script_sha256']
    paper_figures = len(list(paper.glob('figure*.png')))
    paper_tables = len(list(paper.glob('table*.csv')))
    assert paper_figures == 2 and paper_tables == 3
    for path in paper.glob('table*.csv'):
        assert len(pd.read_csv(path)) > 0 and path.with_suffix('.tex').exists()
    claims = pd.read_csv(paper / 'claim_evidence.csv')
    for sources in claims.evidence_files:
        assert all((ROOT / source).is_file() for source in sources.split(';'))

    class LinkCheck(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {'src', 'href'}:
                    target = urlparse(value)
                    if not target.scheme and target.path:
                        path = ROOT / unquote(target.path)
                        assert path.exists() or path == ROOT / 'audit/final_validation.json', value

    LinkCheck().feed((ROOT / 'index.html').read_text())
    inventory = {}
    excluded = {'audit/file_inventory.json', 'audit/final_validation.json'}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file() and relative not in excluded and '__pycache__' not in path.parts:
            inventory[relative] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    (ROOT / 'audit/file_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    receipt = {'status': 'passed', 'verified_at': datetime.now(UTC).isoformat(),
               'rounds_completed_and_visually_reviewed': 6,
               'round_figure_triplets_validated': figures, 'round_csv_latex_table_pairs': tables,
               'compact_paper_figure_triplets': paper_figures, 'paper_csv_latex_table_pairs': paper_tables,
               'total_figure_triplets': figures + paper_figures, 'total_csv_latex_table_pairs': tables + paper_tables,
               'manifest_hashes_verified': hashes, 'inventory_files': len(inventory),
               'matched_reports': 264, 'selected_questions': 638,
               'per_query_setting_variant_rows': 247192,
               'generation_records': 6380, 'benchmark_execution': False,
               'paid_api_calls_in_analysis': 0, 'raw_results_modified': False,
               'note': 'Figure visual review is the recorded analyst inspection; '
                       'this script checks completeness, file structure and provenance.'}
    (ROOT / 'audit/final_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
