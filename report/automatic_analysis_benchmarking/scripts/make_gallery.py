"""Build a portable static index of the six offline analysis rounds."""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def url(path):
    return quote(path.relative_to(ROOT).as_posix())


def link(path, label=None):
    return f'<a href="{url(path)}">{escape(label or path.name)}</a>'


def main():
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLIR: six rounds of analysis</title>
<style>
body{margin:0;background:#f5f7fa;color:#172b3a;font:17px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:32px 24px 64px}h1{line-height:1.2}
h2{margin-top:2.4em}a{color:#00669e}nav{display:flex;gap:20px;flex-wrap:wrap}
section,article{background:white;padding:20px 28px;margin:24px 0;border-radius:8px}
pre{white-space:pre-wrap;font:15px/1.65 system-ui,sans-serif;overflow-wrap:anywhere}
figure{margin:24px 0}img{display:block;width:100%;height:auto}figcaption{font-size:14px}
.notice{border-left:4px solid #0072b2;padding-left:18px}.tables{columns:2;column-gap:32px}
li{break-inside:avoid;margin:8px 0;font-size:14px}code{font-size:14px}
@media(max-width:720px){main{padding:16px}.tables{columns:1}section,article{padding:16px}}
</style><main><h1>CLIR: six rounds of analysis</h1>
<p class="notice">638 selected questions · 3 corpora · 8 fixed encoders ·
264 matched document-level reports. Offline exploratory analysis of completed
runs. Figures and tables do not execute models, APIs or server jobs.</p><nav>''']
    parts.extend(f'<a href="#round-{i}">Round {i}</a>' for i in range(1, 7))
    parts.append('</nav>')
    summary = ROOT / 'paper/overview.txt'
    if summary.exists():
        parts.append('<section><h2>Main findings</h2><pre>' + escape(summary.read_text()) + '</pre></section>')
    documents = [
        ('paper/findings.txt', 'Full findings and evidence summary'),
        ('protocol.txt', 'Analysis protocol'),
        ('paper/methods_and_limits.txt', 'Methods and limitations'),
        ('paper/claim_evidence.csv', 'Claim–evidence table'),
        ('paper/submission_priorities.txt', 'Before submission'),
        ('paper/paper_outline.txt', 'Suggested paper outline and abstract'),
        ('paper/prior_work_positioning.txt', 'Contribution and closest prior work'),
        ('REPRODUCE.txt', 'Reproduction instructions'),
        ('literature/review_notes.txt', 'Primary-source literature review'),
        ('literature/references.bib', 'Bibliography'),
        ('audit/READ_ME.txt', 'Data integrity audit'),
        ('audit/claim_risks.txt', 'Independent claim review'),
        ('audit/candidate_pool_provenance.txt', 'Candidate pool provenance'),
        ('audit/final_validation.json', 'Final validation'),
    ]
    parts.append('<section><h2>Study records</h2><ul>')
    for name, label in documents:
        path = ROOT / name
        if path.exists() or name == 'audit/final_validation.json':
            parts.append('<li>' + link(path, label) + '</li>')
    parts.append('</ul></section>')
    parts.append('<section><h2>Compact paper figures and tables</h2>')
    for picture in sorted((ROOT / 'paper').glob('figure*.png')):
        formats = ' · '.join(link(picture.with_suffix('.' + ext), ext.upper()) for ext in ('png', 'pdf', 'svg'))
        parts.append(f'<figure><img src="{url(picture)}" alt="{escape(picture.stem.replace("_", " "))}"><figcaption>{formats}</figcaption></figure>')
    captions = ROOT / 'paper/figure_captions.txt'
    if captions.exists():
        parts.append('<details><summary>Full figure captions</summary><pre>' + escape(captions.read_text()) + '</pre></details>')
    parts.append('<ul>')
    for table in sorted((ROOT / 'paper').glob('table*.csv')):
        parts.append('<li>' + link(table) + ' · ' + link(table.with_suffix('.tex'), 'LaTeX') + '</li>')
    parts.append('</ul></section>')
    for number in range(1, 7):
        directory = ROOT / f'round_{number:02d}'
        parts.append(f'<article id="round-{number}"><h2>Round {number}</h2>')
        review = directory / 'review.txt'
        if review.exists():
            parts.append('<pre>' + escape(review.read_text()) + '</pre>')
        for picture in sorted(directory.glob('fig*.png')):
            caption = ' · '.join(link(picture.with_suffix('.' + ext), ext.upper())
                                 for ext in ('png', 'pdf', 'svg')
                                 if picture.with_suffix('.' + ext).exists())
            parts.append(f'<figure><img loading="lazy" src="{url(picture)}" '
                         f'alt="{escape(picture.stem.replace("_", " "))}">'
                         f'<figcaption>{caption}</figcaption></figure>')
        parts.append('<h3>Numeric tables</h3><ul class="tables">')
        for path in sorted(directory.glob('table*.csv')):
            latex = path.with_suffix('.tex')
            parts.append('<li>' + link(path) + (' · ' + link(latex, 'LaTeX') if latex.exists() else '') + '</li>')
        parts.append('</ul></article>')
    parts.append('</main></html>')
    (ROOT / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(ROOT / 'index.html')


if __name__ == '__main__':
    main()
