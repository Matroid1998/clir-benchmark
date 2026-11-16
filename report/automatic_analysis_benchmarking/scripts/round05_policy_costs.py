"""Offline source-fold selection with separately attributed generation accounting."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    CORPORA,
    CORPUS_LABEL,
    LANGUAGES,
    MODEL_LABEL,
    MODELS,
    ROOT,
    paired_bootstrap,
    savefig,
    table,
    write_json,
)

OUT = ROOT / 'round_05'
VARIANTS = ['original'] + [f'{method}_{language}' for method in ['translate', 'hyde'] for language in LANGUAGES]
METRICS = ['ndcg10', 'hit10', 'recall10', 'mrr10']
POLICIES = ['original', 'english', 'cv_ndcg', 'cv_hit', 'oracle_ndcg', 'oracle_hit']
DEPLOYABLE = POLICIES[:4]
LABEL = {'original': 'Original', 'english': 'Fixed English translation',
         'cv_ndcg': 'CV: choose for nDCG', 'cv_hit': 'CV: choose for source Hit',
         'oracle_ndcg': 'Per-query nDCG oracle', 'oracle_hit': 'Per-query Hit oracle'}
COLOR = {'english': '#777777', 'cv_ndcg': '#0072B2', 'cv_hit': '#D55E00', 'original': '#222222'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_folds(queries, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    assert not queries.duplicated(['corpus', 'canonical_source_id']).any()
    for (corpus, language), group in queries.groupby(['corpus', 'query_language'], sort=True):
        ordered = group.sort_values('query_id').reset_index(drop=True)
        order = rng.permutation(len(ordered))
        for index, position in enumerate(order):
            row = ordered.iloc[position]
            rows.append(dict(corpus=corpus, query_language=language, query_id=row.query_id,
                             canonical_source_id=row.canonical_source_id, fold=index % 5))
    folds = pd.DataFrame(rows)
    assert len(folds) == 638 and not folds.duplicated(['corpus', 'query_id']).any()
    assert folds.groupby('corpus').fold.nunique().eq(5).all()
    return folds


def choose(training, metric):
    stats = training.groupby('variant').agg(score=(metric, 'mean'), cost=('api_cost', 'mean'))
    assert set(stats.index) == set(VARIANTS) and np.isfinite(stats).all().all()
    best = stats[np.isclose(stats.score, stats.score.max(), rtol=0, atol=1e-12)]
    best = best[np.isclose(best.cost, best.cost.min(), rtol=0, atol=1e-12)]
    return sorted(best.index, key=lambda value: (value != 'original', value))[0], stats


def fit_and_score(data, folds):
    data = data.merge(folds[['corpus', 'query_id', 'fold']], on=['corpus', 'query_id'], validate='many_to_one')
    choices, records = [], []
    for (corpus, model), frame in data.groupby(['corpus', 'model']):
        assert frame.groupby('query_id').variant.nunique().eq(11).all()
        assert not frame.duplicated(['query_id', 'variant']).any()
        for fold in range(5):
            training = frame[frame.fold != fold]
            testing = frame[frame.fold == fold]
            train_ids, test_ids = set(training.query_id), set(testing.query_id)
            assert train_ids.isdisjoint(test_ids)
            assert set(training.canonical_source_id).isdisjoint(set(testing.canonical_source_id))
            selections = {'original': 'original', 'english': 'translate_en'}
            for policy, metric in [('cv_ndcg', 'ndcg10'), ('cv_hit', 'hit10')]:
                selected, stats = choose(training, metric)
                selections[policy] = selected
                choices.append(dict(corpus=corpus, model=model, fold=fold, policy=policy,
                                    selected_variant=selected, training_questions=len(train_ids),
                                    held_out_questions=len(test_ids), training_objective=stats.loc[selected, 'score'],
                                    training_mean_api_cost=stats.loc[selected, 'cost']))
            for policy, selected in selections.items():
                chosen = testing[testing.variant == selected].copy()
                assert len(chosen) == len(test_ids)
                chosen['policy'] = policy
                chosen['oracle'] = False
                records.append(chosen)
            # These rows explicitly use held-out outcomes and are upper bounds only.
            for policy, metric in [('oracle_ndcg', 'ndcg10'), ('oracle_hit', 'hit10')]:
                ranked = testing.assign(original_priority=testing.variant.ne('original').astype(int))
                ranked = ranked.sort_values(['query_id', metric, 'api_cost', 'original_priority', 'variant'],
                                            ascending=[True, False, True, True, True])
                chosen = ranked.drop_duplicates('query_id').drop(columns='original_priority').copy()
                chosen['policy'] = policy
                chosen['oracle'] = True
                records.append(chosen)
    oof = pd.concat(records, ignore_index=True)
    assert len(oof) == 638 * 8 * len(POLICIES)
    assert not oof.duplicated(['corpus', 'model', 'query_id', 'policy']).any()
    assert oof.groupby(['corpus', 'query_id']).fold.nunique().eq(1).all()
    for metric, oracle in [('ndcg10', 'oracle_ndcg'), ('hit10', 'oracle_hit')]:
        wide = oof.pivot(index=['corpus', 'model', 'query_id'], columns='policy', values=metric)
        assert wide[DEPLOYABLE].le(wide[oracle] + 1e-12, axis=0).all().all()
    return oof, pd.DataFrame(choices)


def contrasts(oof):
    rows = []
    comparisons = [('english', 'original'), ('cv_ndcg', 'original'), ('cv_hit', 'original'),
                   ('cv_ndcg', 'english'), ('cv_hit', 'english'), ('cv_ndcg', 'cv_hit')]
    for corpus, group in oof.groupby('corpus'):
        for metric in ['ndcg10', 'hit10']:
            wide = group.pivot(index=['model', 'query_id'], columns='policy', values=metric)
            assert wide.notna().all().all()
            for treatment, control in comparisons:
                differences = (wide[treatment] - wide[control]).rename('delta').reset_index()
                for model in MODELS + ['fixed_mean']:
                    values = (differences.groupby('query_id').delta.mean() if model == 'fixed_mean'
                              else differences.loc[differences.model == model, 'delta'])
                    mean, low, high = paired_bootstrap(values)
                    rows.append(dict(corpus=corpus, model=model, metric=metric, policy=treatment,
                                     control=control, n_questions=len(values), mean=mean,
                                     ci_low=low, ci_high=high,
                                     gain_questions=int((values > 1e-10).sum()),
                                     loss_questions=int((values < -1e-10).sum()),
                                     tied_questions=int((abs(values) <= 1e-10).sum())))
    return pd.DataFrame(rows)


def accounting(oof, generation):
    rows = []
    for (corpus, model, policy), group in oof[~oof.oracle].groupby(['corpus', 'model', 'policy']):
        rows.append(dict(corpus=corpus, model=model, policy=policy,
                         n_questions=len(group), selected_historical_api_usd=group.api_cost.sum(),
                         api_usd_per_1000_queries=1000 * group.api_cost.mean(),
                         attempt_seconds_mean=group.api_seconds.mean(),
                         attempt_seconds_median=group.api_seconds.median(),
                         attempt_seconds_p95=group.api_seconds.quantile(.95),
                         encoding_seconds_amortized_mean=group.query_encoding_seconds_amortized.mean(),
                         retrieval_seconds_amortized_mean=group.retrieval_seconds_amortized.mean(),
                         compute_price_usd=np.nan, online_end_to_end_seconds=np.nan))
    attributed = pd.DataFrame(rows)
    unions = []
    for policy, group in oof[~oof.oracle].groupby('policy'):
        unique = group[group.variant != 'original'].drop_duplicates(['corpus', 'query_id', 'variant'])
        unions.append(dict(policy=policy, corpus_scope='all three corpora', encoders=8,
                           unique_selected_outputs=len(unique),
                           unique_selected_request_keys=unique.request_key.nunique(),
                           union_historical_api_usd=unique.api_cost.sum(),
                           interpretation='Attribution to selected outputs; training/exploration still requires the full pool'))
    genrows = []
    for (corpus, method), group in generation.groupby(['corpus', 'method']):
        genrows.append(dict(corpus=corpus, method=method, query_output_records=len(group),
                            identity_bypasses=int(group.status.eq('identity').sum()),
                            attempts=int(group.attempts.sum()), historical_api_usd=group.estimated_cost_usd.sum(),
                            input_tokens=int(group.input_tokens.sum()), cached_input_tokens=int(group.cached_input_tokens.sum()),
                            output_tokens=int(group.output_tokens.sum()),
                            api_usd_per_1000_query_outputs=1000 * group.estimated_cost_usd.mean(),
                            duration_sum_seconds=group.generation_seconds.sum(),
                            duration_mean_seconds=group.generation_seconds.mean(),
                            duration_median_seconds=group.generation_seconds.median(),
                            duration_p95_seconds=group.generation_seconds.quantile(.95)))
    return attributed, pd.DataFrame(unions), pd.DataFrame(genrows)


def plots(delta, choices, generation_totals, attributed):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex='row')
    for col, corpus in enumerate(CORPORA):
        for row, metric in enumerate(['ndcg10', 'hit10']):
            ax = axes[row, col]
            subset = delta[(delta.corpus == corpus) & (delta.model == 'fixed_mean') &
                           (delta.metric == metric) & (delta.control == 'original')].set_index('policy')
            for i, policy in enumerate(['english', 'cv_ndcg', 'cv_hit']):
                value = subset.loc[policy]
                ax.errorbar(value['mean'] * 100, i, xerr=[[100 * (value['mean'] - value.ci_low)],
                            [100 * (value.ci_high - value['mean'])]], fmt='o', color=COLOR[policy], capsize=3)
            ax.axvline(0, color='gray', lw=.8)
            ax.set_yticks(range(3), [LABEL[p] for p in ['english', 'cv_ndcg', 'cv_hit']] if col == 0 else [])
            ax.set_ylim(2.6, -.6)
            ax.set_title(CORPUS_LABEL[corpus] + (' · nDCG@10' if metric == 'ndcg10' else ' · source Hit@10'))
            ax.set_xlabel('Change from original query (points)')
    fig.suptitle('Five-fold internal evaluation: training objective changes held-out behavior')
    fig.subplots_adjust(left=.2, right=.98, bottom=.17, top=.88, hspace=.5, wspace=.2)
    fig.text(.5, .025, 'One training-selected variant per corpus, encoder and fold; all variants from a source share a fold.\n'
             'Eight fixed encoders averaged within each query. 95% paired intervals condition on fitted choices;\n'
             'the selector is not refitted inside the bootstrap. This adaptive internal analysis is not an external test.',
             ha='center', fontsize=8)
    savefig(fig, OUT, 'fig01_held_out_objective_tradeoffs')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharey=True)
    short = ['O'] + [v.replace('translate_', 'T-').replace('hyde_', 'H-') for v in VARIANTS[1:]]
    for row, policy in enumerate(['cv_ndcg', 'cv_hit']):
        for col, corpus in enumerate(CORPORA):
            ax = axes[row, col]
            sub = choices[(choices.corpus == corpus) & (choices.policy == policy)]
            counts = pd.crosstab(sub.model, sub.selected_variant).reindex(index=MODELS, columns=VARIANTS, fill_value=0)
            assert counts.sum(axis=1).eq(5).all()
            im = ax.imshow(counts, vmin=0, vmax=5, cmap='Blues', aspect='auto')
            for i in range(8):
                for j in range(11):
                    if counts.iloc[i, j]:
                        ax.text(j, i, str(counts.iloc[i, j]), ha='center', va='center', fontsize=7,
                                color='white' if counts.iloc[i, j] >= 3 else 'black')
            ax.set_xticks(range(11), short, rotation=55, ha='right')
            ax.set_yticks(range(8), [MODEL_LABEL[m] for m in MODELS])
            ax.set_title(CORPUS_LABEL[corpus] + (' · select by nDCG' if policy == 'cv_ndcg' else ' · select by Hit'))
    fig.subplots_adjust(left=.17, right=.94, top=.9, bottom=.16, wspace=.16, hspace=.45)
    fig.colorbar(im, cax=fig.add_axes([.96, .22, .012, .58]), label='Training folds choosing variant')
    fig.suptitle('A single fixed language/method is learned from each training fold')
    fig.text(.5, .025, 'Each row sums to five folds. O: original; T: translation; H: HyDE. Ties prefer lower training-cost means,\n'
             'then original and lexical order. No held-out outcomes enter the choice.', ha='center', fontsize=8)
    savefig(fig, OUT, 'fig02_training_fold_choices')

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    methods = ['translate', 'hyde']
    colors = ['#0072B2', '#D55E00']
    total = generation_totals.groupby('method').historical_api_usd.sum()
    bars = axes[0].bar(methods, total.reindex(methods), color=colors)
    axes[0].bar_label(bars, labels=[f'${total[m]:.4f}' for m in methods], padding=3)
    axes[0].set_ylim(0, .59)
    axes[0].set_title('Actual recorded generation pool')
    axes[0].set_ylabel('Configured-price USD, including retries')
    axes[0].set_xticks(range(2), ['Translation\n5 outputs/query', 'HyDE\n5 outputs/query'])
    for j, method in enumerate(methods):
        sub = generation_totals[generation_totals.method == method].set_index('corpus').reindex(CORPORA)
        x = np.arange(3) + (j - .5) * .25
        axes[1].scatter(x, sub.duration_median_seconds, color=colors[j], marker='o', label=method + ' median')
        axes[1].scatter(x, sub.duration_p95_seconds, color=colors[j], marker='^', alpha=.65, label=method + ' p95')
    axes[1].set_xticks(range(3), [CORPUS_LABEL[c] for c in CORPORA], rotation=20)
    axes[1].set_ylabel('Recorded HTTP attempt seconds/query-output')
    axes[1].set_title('Generation duration; identity T = 0')
    axes[1].legend(fontsize=6)
    cost = attributed.groupby(['corpus', 'policy']).api_usd_per_1000_queries.mean()
    for policy in DEPLOYABLE:
        values = [cost.loc[(corpus, policy)] for corpus in CORPORA]
        axes[2].plot(range(3), values, marker='o', color=COLOR[policy], label=LABEL[policy])
    axes[2].set_xticks(range(3), [CORPUS_LABEL[c] for c in CORPORA], rotation=20)
    axes[2].set_ylabel('Selected-output USD per 1,000 queries')
    axes[2].set_title('Attribution at observed prices/cache history')
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .145), ncol=4,
               fontsize=7, title='Right panel: selected policy', title_fontsize=7)
    fig.subplots_adjust(left=.07, right=.99, bottom=.35, top=.86, wspace=.5)
    fig.text(.5, .025, 'Left: unique outputs counted once across encoders/settings. Right: mean stand-alone attribution across eight encoders,\n'
             'excluding training exploration; these are not additional measured bills or fully cold forecasts.\n'
             'CPU/GPU prices and online end-to-end latency are unknown; offline component timings are provided separately in Table 4.',
             ha='center', fontsize=7)
    savefig(fig, OUT, 'fig03_generation_accounting_and_duration')


def main():
    assert json.loads((ROOT / 'rounds.json').read_text())['rounds'][3]['status'] == 'complete'
    OUT.mkdir(exist_ok=True)
    queries = pd.read_parquet(ROOT / 'data/selected_queries.parquet')
    generation = pd.read_parquet(ROOT / 'data/generation.parquet')
    assert not generation.duplicated(['corpus', 'method', 'output_language', 'query_id']).any()
    generation['variant'] = generation.method + '_' + generation.output_language
    gen = generation[['corpus', 'query_id', 'variant', 'request_key', 'estimated_cost_usd', 'generation_seconds']].rename(
        columns={'estimated_cost_usd': 'api_cost', 'generation_seconds': 'api_seconds'})
    data = pd.read_parquet(ROOT / 'data/per_query.parquet')
    data = data[data.regime == 'mixed'].copy()
    data = data.merge(gen, on=['corpus', 'query_id', 'variant'], how='left', validate='many_to_one')
    data.loc[data.variant == 'original', ['api_cost', 'api_seconds']] = 0.0
    assert np.isfinite(data[METRICS + ['api_cost', 'api_seconds']]).all().all()
    folds = make_folds(queries)
    oof, choices = fit_and_score(data, folds)
    delta = contrasts(oof)
    attributed, unions, generation_totals = accounting(oof, generation)
    levels = oof.groupby(['corpus', 'model', 'policy'])[METRICS].mean().reset_index()
    fixed = oof.groupby(['corpus', 'query_id', 'policy'])[METRICS].mean().groupby(['corpus', 'policy']).mean().reset_index()
    fixed['model'] = 'fixed_mean'
    levels = pd.concat([levels, fixed], ignore_index=True)
    table(levels, OUT, 'table01_held_out_levels_and_oracles')
    table(delta, OUT, 'table02_held_out_paired_contrasts')
    table(choices, OUT, 'table03_training_choices')
    table(attributed, OUT, 'table04_selected_output_costs_and_components')
    table(unions, OUT, 'table05_shared_output_union_cost')
    table(generation_totals, OUT, 'table06_unique_generation_pool')
    folds.to_csv(OUT / 'source_folds.csv', index=False)
    oof.to_parquet(OUT / 'out_of_fold_predictions.parquet', index=False)
    plots(delta, choices, generation_totals, attributed)
    lines = ['ROUND 5 NUMERICAL REVIEW — TRAINING-FOLD CHOICES AND HISTORICAL COST', '',
             'Visual review pending root. Five folds stratify original language within corpus; every source shares a fold across models.',
             'Choices use only training mean objective, then training mean cost, then original/lexical ordering.',
             'Each corpus/model/fold chooses one static variant. No per-query features or held-out outcomes drive fitted selection.', '',
             'Fixed-eight-encoder held-out contrasts (points; source bootstrap conditional on chosen policies):']
    for row in delta[(delta.model == 'fixed_mean') & delta.policy.isin(['cv_ndcg', 'cv_hit'])].itertuples():
        lines.append(f'  {row.corpus}, {row.metric}, {row.policy} − {row.control}: {row.mean*100:+.2f} '
                     f'[{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]; gains/losses/ties {row.gain_questions}/{row.loss_questions}/{row.tied_questions}.')
    lines.extend(['', 'Recorded generation accounting (unique outputs, not repeated model/settings records):'])
    for method, group in generation_totals.groupby('method'):
        lines.append(f'  {method}: ${group.historical_api_usd.sum():.8f}; {group.attempts.sum()} attempts; '
                     f'{group.query_output_records.sum()} outputs, including identity bypasses where applicable.')
    for row in unions.itertuples():
        lines.append(f'  {row.policy}: selected outputs used across all eight encoders have union historical attribution ${row.union_historical_api_usd:.8f}.')
    lines.extend(['', 'Interpretation boundaries',
                  '- Internal CV is exploratory: this design follows analysis of the same dataset, not an untouched external evaluation.',
                  '- Bootstrap intervals resample paired source-level scored outputs, averaging fixed encoders first; they do not refit selectors.',
                  '- Overlapping training folds and selection instability are not fully captured by those conditional intervals.',
                  '- Oracles use each held-out outcome and are upper bounds only. Their favorable scores are not deployable results.',
                  '- Every query appears in training in other folds; exploring/training these policies used the complete generation pool.',
                  '- Selected-output attribution excludes training exploration and is not an actual measured saving or a new API bill.',
                  '- Historical generation includes four failed attempts, retries and provider cache discounts; it is not fully cold pricing.',
                  '- Generation is reused across encoders. Table 5 deduplicates their selected-output unions.',
                  '- API attempt durations, offline amortized embedding and offline amortized retrieval are separate measurements.',
                  '- CPU/GPU monetary costs and measured online end-to-end latency remain unknown.', '',
                  'Questions driving Round 6',
                  'Do headline contrasts survive quality/identity exclusions, original-language and setting weighting, and omission of one encoder?',
                  'Does the UN source-success/version-coverage tradeoff persist across saved cutoffs, or is it a top-10-only operating point?',
                  'Which practical conclusions remain conditional on this one fold assignment and the selected synthetic-query population?',
                  'Which claims should be retained, narrowed or rejected in the final evidence table?'])
    (OUT / 'review_draft.txt').write_text('\n'.join(lines) + '\n')
    validation = {'status': 'passed', 'folds': 5, 'seed': 42, 'source_questions': 638,
                  'out_of_fold_rows': len(oof), 'training_choices': len(choices),
                  'train_test_source_overlap': 0, 'shared_source_folds_across_models': True,
                  'all_cohorts_identical': True, 'oracle_upper_bounds_verified': True,
                  'translation_historical_usd': float(generation.loc[generation.method == 'translate', 'estimated_cost_usd'].sum()),
                  'hyde_historical_usd': float(generation.loc[generation.method == 'hyde', 'estimated_cost_usd'].sum()),
                  'paid_api_calls_in_analysis': 0, 'benchmark_execution': False}
    assert np.isclose(validation['translation_historical_usd'], .230011, atol=1e-10, rtol=0)
    assert np.isclose(validation['hyde_historical_usd'], .49762452, atol=1e-10, rtol=0)
    write_json(OUT / 'validation.json', validation)
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in {'run_manifest.json', 'review.txt'}]
    inputs = [ROOT / 'round_04/review.txt', ROOT / 'data/per_query.parquet', ROOT / 'data/generation.parquet', ROOT / 'audit/generation_accounting.json']
    write_json(OUT / 'run_manifest.json', {'round': 5, 'generated_at': datetime.now(UTC).isoformat(),
        'visual_review_status': 'pending_root', 'script_sha256': digest(__file__),
        'inputs': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        'output_sha256': {p.name: digest(p) for p in outputs}, 'paid_calls': 0, 'model_loads': 0})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
