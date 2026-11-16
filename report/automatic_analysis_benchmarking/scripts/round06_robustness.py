"""Final offline falsification checks; requires the accepted Round 5 review."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    CORPORA,
    CORPUS_LABEL,
    MODELS,
    ROOT,
    centered_heatmap,
    paired_bootstrap,
    savefig,
    table,
    write_json,
)
from round05_policy_costs import fit_and_score, make_folds

OUT = ROOT / 'round_06'
KEYS = ['corpus', 'model', 'query_id', 'query_language', 'setting_id']
LABELS = {
    'all': 'All selected questions', 'clear': 'Clear quality only',
    'no_identity': 'Exclude identity-affected sources',
    'language_macro': 'Equal original-language weight',
    'setting_macro': 'Equal setting weight',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def language_bootstrap(frame):
    groups = [part.delta.to_numpy() for _, part in frame.groupby('query_language')]
    assert groups and all(np.isfinite(values).all() for values in groups)
    rng = np.random.default_rng(42)
    means = np.zeros(5000)
    for values in groups:
        for start in range(0, 5000, 500):
            indices = rng.integers(len(values), size=(500, len(values)))
            means[start:start + 500] += values[indices].mean(axis=1) / len(groups)
    return float(np.mean([v.mean() for v in groups])), *np.quantile(means, [.025, .975])


def target_pairs(data):
    subset = data[(data.regime == 'cross_language') &
                  ((data.method == 'original') | (data.output_language == data.target_language))]
    assert not subset.duplicated(KEYS + ['method']).any()
    wide = subset.pivot(index=KEYS, columns='method', values='ndcg10')
    assert set(wide.columns) == {'original', 'translate', 'hyde'}
    assert wide.notna().all().all()
    pieces = []
    for label, first, second in [('T − original', 'translate', 'original'),
                                  ('H − T', 'hyde', 'translate')]:
        frame = (wide[first] - wide[second]).rename('delta').reset_index()
        frame['contrast'] = label
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def robustness_table(pairs, metadata, affected):
    rows = []
    meta = metadata.set_index(['corpus', 'query_id'])
    for (corpus, contrast), group in pairs.groupby(['corpus', 'contrast']):
        assert group.groupby(['query_id', 'setting_id']).model.nunique().eq(8).all()
        per_setting = group.groupby(['query_id', 'query_language', 'setting_id']).delta.mean().reset_index()
        per_query = per_setting.groupby(['query_id', 'query_language']).delta.mean().reset_index()
        quality = meta.loc[corpus].quality_audit_status
        for estimator in LABELS:
            values = per_query.copy()
            if estimator == 'clear':
                values = values[values.query_id.map(quality).eq('clear')]
            elif estimator == 'no_identity':
                values = values[~values.query_id.isin(affected.get(corpus, set()))]
            if estimator == 'language_macro':
                mean, lo, hi = language_bootstrap(values)
            elif estimator == 'setting_macro':
                mean = per_setting.groupby('setting_id').delta.mean().mean()
                lo = hi = np.nan  # tiny cells do not justify a source bootstrap precision claim
            else:
                mean, lo, hi = paired_bootstrap(values.delta)
            rows.append(dict(corpus=corpus, contrast=contrast, estimator=estimator,
                             n_questions=len(values), mean=mean, ci_low=lo, ci_high=hi,
                             ci_note='No interval: descriptive setting macro' if estimator == 'setting_macro'
                             else 'Paired source bootstrap; original-language stratified for language macro'))
    return pd.DataFrame(rows)


def un_pairs(data):
    subset = data[(data.corpus == 'un') & (data.regime == 'mixed') &
                  data.variant.isin(['original', 'translate_en', 'translate_de'])]
    ids = ['model', 'query_id', 'query_language']
    metrics = ['ndcg10'] + [f'{metric}{k}' for metric in ['hit', 'recall'] for k in [1, 5, 10, 20, 100]]
    assert not subset.duplicated(ids + ['variant']).any()
    wide = subset.pivot(index=ids, columns='variant', values=metrics)
    assert wide.notna().all().all()
    rows = []
    for control in ['translate_en', 'original']:
        for metric in metrics:
            part = (wide[(metric, 'translate_de')] - wide[(metric, control)]).rename('delta').reset_index()
            part['control'] = control
            part['metric'] = metric
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def un_summary(pairs, metadata):
    quality = metadata[metadata.corpus == 'un'].set_index('query_id').quality_audit_status
    rows = []
    for (control, metric), group in pairs.groupby(['control', 'metric']):
        assert group.groupby('query_id').model.nunique().eq(8).all()
        queries = group.groupby(['query_id', 'query_language']).delta.mean().reset_index()
        for estimator in ['all', 'clear', 'language_macro']:
            values = queries if estimator != 'clear' else queries[queries.query_id.map(quality).eq('clear')]
            stats = language_bootstrap(values) if estimator == 'language_macro' else paired_bootstrap(values.delta)
            rows.append(dict(control=control, metric=metric, estimator=estimator,
                             n_questions=len(values), mean=stats[0], ci_low=stats[1], ci_high=stats[2]))
    return pd.DataFrame(rows)


def influence_table(target, un):
    rows = []
    for (corpus, contrast), group in target.groupby(['corpus', 'contrast']):
        per_model = group.groupby(['model', 'query_id']).delta.mean().reset_index()
        for omit in ['none'] + MODELS:
            chosen = per_model if omit == 'none' else per_model[per_model.model != omit]
            rows.append(dict(family='target', corpus=corpus, contrast=contrast, metric='ndcg10',
                             omitted_model=omit, mean=chosen.groupby('query_id').delta.mean().mean()))
    for metric in ['ndcg10', 'hit10']:
        group = un[(un.control == 'translate_en') & (un.metric == metric)]
        for omit in ['none'] + MODELS:
            chosen = group if omit == 'none' else group[group.model != omit]
            rows.append(dict(family='UN pivot', corpus='un', contrast='T-de − T-en', metric=metric,
                             omitted_model=omit, mean=chosen.groupby('query_id').delta.mean().mean()))
    return pd.DataFrame(rows)


def policy_split_sensitivity(data, selected):
    generation = pd.read_parquet(ROOT / 'data/generation.parquet')
    generation['variant'] = generation.method + '_' + generation.output_language
    gen = generation[['corpus', 'query_id', 'variant', 'estimated_cost_usd']].rename(columns={'estimated_cost_usd': 'api_cost'})
    columns = ['corpus', 'model', 'query_id', 'query_language', 'canonical_source_id', 'variant', 'ndcg10', 'hit10']
    mixed = data.loc[data.regime == 'mixed', columns].merge(gen, on=['corpus', 'query_id', 'variant'], how='left', validate='many_to_one')
    mixed.loc[mixed.variant == 'original', 'api_cost'] = 0.0
    assert np.isfinite(mixed.api_cost).all()
    rows, fold_rows, choice_rows = [], [], []
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]:
        print(f'Refitting five source folds with seed {seed}', flush=True)
        folds = make_folds(selected, seed=seed)
        fold_rows.append(folds.assign(seed=seed))
        oof, choices = fit_and_score(mixed, folds)
        choice_rows.append(choices.assign(seed=seed))
        for corpus, group in oof.groupby('corpus'):
            for metric in ['ndcg10', 'hit10']:
                fixed = group.groupby(['query_id', 'policy'])[metric].mean().unstack('policy')
                for treatment, control in [('cv_ndcg', 'cv_hit'), ('cv_ndcg', 'original'), ('cv_hit', 'original')]:
                    rows.append(dict(seed=seed, corpus=corpus, metric=metric, policy=treatment, control=control,
                                     n_questions=len(fixed), mean=(fixed[treatment] - fixed[control]).mean()))
    summary = pd.DataFrame(rows)
    baseline = pd.read_csv(ROOT / 'round_05/table02_held_out_paired_contrasts.csv')
    baseline = baseline[baseline.model == 'fixed_mean']
    joined = summary[summary.seed == 42].merge(baseline, on=['corpus', 'metric', 'policy', 'control'], suffixes=('_new', '_r5'), validate='one_to_one')
    assert len(joined) == 18
    assert np.allclose(joined.mean_new, joined.mean_r5, rtol=0, atol=1e-12)
    pd.concat(fold_rows, ignore_index=True).to_parquet(OUT / 'sensitivity_source_folds.parquet', index=False)
    pd.concat(choice_rows, ignore_index=True).to_parquet(OUT / 'sensitivity_training_choices.parquet', index=False)
    return summary


def plot_policy_seeds(frame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    for ax, metric in zip(axes, ['ndcg10', 'hit10'], strict=True):
        for i, corpus in enumerate(CORPORA):
            group = frame[(frame.corpus == corpus) & (frame.metric == metric) &
                          (frame.policy == 'cv_ndcg') & (frame.control == 'cv_hit')].sort_values('seed')
            others = group[group.seed != 42]
            ax.scatter(others['mean'] * 100, i + np.linspace(-.12, .12, len(others)), color='#999999', s=20)
            ax.scatter(group.loc[group.seed == 42, 'mean'] * 100, [i], color='#0072B2', marker='D', s=35, zorder=3)
        ax.axvline(0, color='gray', lw=.8)
        ax.set_yticks(range(3), [CORPUS_LABEL[c] for c in CORPORA])
        ax.set_ylim(2.5, -.5)
        ax.set_title('Version nDCG@10' if metric == 'ndcg10' else 'Source Hit@10')
        ax.set_xlabel('nDCG-selected − Hit-selected policy (points)')
    fig.subplots_adjust(left=.12, right=.98, top=.78, bottom=.3, wspace=.25)
    fig.suptitle('Does the selection-objective trade-off depend on the source split?')
    fig.text(.5, .035, 'Blue diamond: original seed 42. Gray dots: seeds 0–8; each refits five source-grouped training folds.\n'
             'Eight fixed encoders averaged within each question. Dots are alternative splits of the same selected\n'
             'questions, not independent test sets or confidence intervals.', ha='center', fontsize=8)
    savefig(fig, OUT, 'fig04_policy_fold_assignment_sensitivity')


def plot_checks(robust, cutoff, influence):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    limit = float(np.ceil(abs(robust['mean']).max() * 100))
    for ax, contrast in zip(axes, ['T − original', 'H − T'], strict=True):
        frame = robust[robust.contrast == contrast].pivot(index='estimator', columns='corpus', values='mean')
        matrix = frame.reindex(index=list(LABELS), columns=CORPORA).to_numpy() * 100
        im = centered_heatmap(ax, matrix, list(LABELS.values()), [CORPUS_LABEL[c] for c in CORPORA],
                              limit=limit, decimals=2)
        ax.set_title('Translation − original' if contrast == 'T − original' else 'HyDE − translation')
    fig.subplots_adjust(left=.28, right=.9, top=.83, bottom=.22, wspace=.12)
    fig.colorbar(im, cax=fig.add_axes([.92, .24, .016, .56]), label='Paired change (points)')
    fig.suptitle('Target-matched cross-language nDCG@10: cohort and weighting checks', y=.98)
    fig.text(.5, .04, 'Eight fixed encoders averaged within each source. All and exclusions use equal question weight.\n'
             'Language/setting rows change the declared estimand; paired intervals and sample sizes are in Table 1.', ha='center', fontsize=8)
    savefig(fig, OUT, 'fig01_cohort_and_weighting_sensitivity')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, control in zip(axes, ['translate_en', 'original'], strict=True):
        for metric, color, label in [('hit', '#0072B2', 'Source Hit@k'), ('recall', '#D55E00', 'Version Recall@k')]:
            sub = cutoff[(cutoff.control == control) & (cutoff.estimator == 'all') & cutoff.metric.str.startswith(metric)].copy()
            sub['k'] = sub.metric.str.removeprefix(metric).astype(int)
            sub = sub.sort_values('k')
            ax.plot(sub.k, sub['mean'] * 100, marker='o', color=color, label=label)
            ax.fill_between(sub.k, sub.ci_low * 100, sub.ci_high * 100, color=color, alpha=.13)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 10, 20, 100], ['1', '5', '10', '20', '100'])
        ax.set_xlabel('Cutoff k in the document-version ranking')
        ax.set_title('German − ' + ('English translation' if control == 'translate_en' else 'original query'))
    axes[0].set_ylabel('Paired change (points)')
    axes[1].legend(loc='best')
    fig.suptitle('UN: source success and language-version coverage across cutoffs')
    fig.subplots_adjust(top=.79, bottom=.25, wspace=.2)
    fig.text(.5, .025, '247 paired source questions, fixed eight-encoder mean. Shading: 95% paired source bootstrap.\n'
             'Saved cutoffs are correlated sensitivity checks; nDCG is only evaluated at k=10 in these reports.', ha='center', fontsize=8)
    savefig(fig, OUT, 'fig02_un_cutoff_sensitivity')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups = list(influence.groupby(['family', 'corpus', 'contrast', 'metric'], sort=False))
    labels = []
    for i, ((_family, corpus, contrast, metric), frame) in enumerate(groups):
        full = frame.loc[frame.omitted_model == 'none', 'mean'].item() * 100
        omit = frame.loc[frame.omitted_model != 'none', 'mean'] * 100
        ax.plot([omit.min(), omit.max()], [i, i], color='#999999', linewidth=5, solid_capstyle='butt')
        ax.scatter([full], [i], color='#0072B2', s=24, zorder=3)
        metric_label = 'nDCG@10' if metric == 'ndcg10' else 'source Hit@10'
        labels.append(f'{CORPUS_LABEL[corpus]} · {contrast} · {metric_label}')
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=.8)
    ax.set_xlabel('Paired change (points)')
    ax.set_title('Influence of leaving out one fixed encoder')
    fig.subplots_adjust(left=.38, bottom=.24, top=.88)
    fig.text(.5, .035, 'Blue point: all eight encoders. Gray segment: range of eight leave-one-encoder-out means.\n'
             'These segments are sensitivity ranges, not confidence intervals or random-model uncertainty.', ha='center', fontsize=8)
    savefig(fig, OUT, 'fig03_fixed_encoder_influence')


def main():
    accepted_review = ROOT / 'round_05/review.txt'
    assert accepted_review.exists(), 'Finish and visually review Round 5 before running Round 6.'
    assert json.loads((ROOT / 'rounds.json').read_text())['rounds'][4]['status'] == 'complete'
    OUT.mkdir(exist_ok=True)
    data = pd.read_parquet(ROOT / 'data/per_query.parquet')
    assert np.isfinite(data[['ndcg10', 'hit10', 'recall10']]).all().all()
    metadata = data[['corpus', 'query_id', 'quality_audit_status']].drop_duplicates()
    assert len(metadata) == 638
    identity = json.loads((ROOT / 'audit/identity_control_diagnosis.json').read_text())
    affected = {corpus: {r['query_id'] for r in identity['cases'] if r['corpus'] == corpus} for corpus in CORPORA}
    assert sum(map(len, affected.values())) == 14
    target = target_pairs(data)
    robust = robustness_table(target, metadata, affected)
    un = un_pairs(data)
    cutoff = un_summary(un, metadata)
    influence = influence_table(target, un)
    table(robust, OUT, 'table01_target_contrast_sensitivity')
    table(cutoff, OUT, 'table02_un_pivot_cutoff_and_quality')
    table(influence, OUT, 'table03_fixed_encoder_influence')
    selected = pd.read_parquet(ROOT / 'data/selected_queries.parquet')
    strata = selected.merge(metadata, on=['corpus', 'query_id'], validate='one_to_one')
    count_table = strata.groupby(['corpus', 'quality_audit_status', 'evidence_alignment_eligibility']).size().rename('questions').reset_index()
    table(count_table, OUT, 'table04_quality_alignment_strata')
    policy_seeds = policy_split_sensitivity(data, selected)
    table(policy_seeds, OUT, 'table05_policy_split_sensitivity')
    target.to_parquet(OUT / 'target_paired_differences.parquet', index=False)
    un.to_parquet(OUT / 'un_paired_differences.parquet', index=False)
    plot_checks(robust, cutoff, influence)
    plot_policy_seeds(policy_seeds)
    lines = ['ROUND 6 NUMERICAL REVIEW — ROBUSTNESS AND FALSIFICATION', '',
             'Root visual review is pending. Full-cohort primary results are never replaced by filtered cohorts.', '',
             'Target-matched cross-language contrasts (points):']
    for row in robust.itertuples():
        interval = f'[{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]' if np.isfinite(row.ci_low) else '[descriptive; no CI]'
        lines.append(f'  {row.corpus}, {row.contrast}, {row.estimator}, n={row.n_questions}: {row.mean*100:+.2f} {interval}')
    lines.extend(['', 'UN T-de − T-en at saved cutoffs (all questions, fixed encoder mean):'])
    for row in cutoff[(cutoff.control == 'translate_en') & (cutoff.estimator == 'all')].itertuples():
        lines.append(f'  {row.metric}: {row.mean*100:+.2f} [{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]')
    lines.extend(['', 'Model influence ranges (not confidence intervals):'])
    for key, group in influence.groupby(['family', 'corpus', 'contrast', 'metric'], sort=False):
        values = group.loc[group.omitted_model != 'none', 'mean'] * 100
        lines.append(f'  {key}: {values.min():+.2f} to {values.max():+.2f}')
    lines.extend(['', 'Refitted internal selection across ten fold assignments (ranges, not confidence intervals):'])
    for key, group in policy_seeds.groupby(['corpus', 'metric', 'policy', 'control'], sort=False):
        values = group['mean'] * 100
        lines.append(f'  {key}: {values.min():+.2f} to {values.max():+.2f}; '
                     f'positive/negative/zero splits {(values > 1e-10).sum()}/{(values < -1e-10).sum()}/{(abs(values) <= 1e-10).sum()}.')
    lines.extend(['', 'Interpretation boundaries',
                  '- Clear-quality and identity exclusions are sensitivity subsets, not corrected primary results.',
                  '- Original-language balancing and setting macro change the estimand; small setting cells are descriptive.',
                  '- Question/source bootstrap conditions on fixed encoders, questions and saved generations.',
                  '- Cutoffs, subsets and model omissions are correlated; this adaptive analysis is not a multiplicity-adjusted confirmatory test.',
                  '- Source Hit and version Recall evaluate different objectives; source labels are not exhaustive relevance judgments.',
                  '- Round 5 internal CV is still exploratory and does not supply a new external test population.',
                  '', 'Final questions for synthesis',
                  'Which claims survive these checks, which claims are rejected, and which require new authorized experiments?',
                  'What must an ACL reader know about source-derived queries, candidate selection, adapter roles, and objective choice?',
                  'Do not claim universal HyDE utility, a new metric, a superior German pivot, or causally identified domain effects.'])
    (OUT / 'review_draft.txt').write_text('\n'.join(lines) + '\n')
    validation = {'status': 'passed', 'primary_questions': 638,
                  'clear_questions': int(metadata.quality_audit_status.eq('clear').sum()),
                  'identity_affected_questions': sum(map(len, affected.values())),
                  'target_pairing_complete': True, 'un_pairing_complete': True,
                  'models_fixed': 8, 'bootstrap_draws': 5000, 'seed': 42,
                  'all_raw_scores_preserved': True, 'benchmark_execution': False,
                  'refitted_fold_seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 42],
                  'seed42_reproduces_round5_exactly': True}
    write_json(OUT / 'validation.json', validation)
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in {'run_manifest.json', 'review.txt'}]
    write_json(OUT / 'run_manifest.json', {
        'round': 6, 'generated_at': datetime.now(UTC).isoformat(),
        'visual_review_status': 'pending_root', 'script_sha256': digest(__file_path()),
        'inputs': {str(p.relative_to(ROOT)): digest(p) for p in [accepted_review, ROOT / 'data/per_query.parquet',
                   ROOT / 'audit/identity_control_diagnosis.json', ROOT / 'data/generation.parquet',
                   ROOT / 'scripts/round05_policy_costs.py']},
        'output_sha256': {p.name: digest(p) for p in outputs},
        'paid_calls': 0, 'model_loads': 0})
    print('\n'.join(lines))


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__ == '__main__':
    main()
