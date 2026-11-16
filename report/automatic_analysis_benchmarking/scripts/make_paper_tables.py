"""Presentation tables and claim links from the accepted six-round analyses."""
from __future__ import annotations

import pandas as pd
from common import CORPORA, CORPUS_LABEL, MODEL_LABEL, MODELS, ROOT, table


def interval(row):
    return f'{row.estimate * 100:+.2f} [{row.ci_low * 100:+.2f}, {row.ci_high * 100:+.2f}]'


def main():
    out = ROOT / 'paper'
    contrasts = pd.read_csv(ROOT / 'round_02/table02_target_matched_contrasts.csv')
    contrasts = contrasts[contrasts.metric == 'ndcg10']
    levels = pd.read_csv(ROOT / 'round_02/table01_target_matched_levels.csv')
    levels = levels[(levels.metric == 'ndcg10') & (levels.model == 'All fixed models')]
    rows = []
    for corpus in CORPORA:
        scores = levels[levels.corpus == corpus].set_index('method')
        delta = contrasts[(contrasts.corpus == corpus) & (contrasts.model == 'All fixed models')].set_index('contrast')
        rows.append({'Corpus': CORPUS_LABEL[corpus], 'Questions': int(scores.n_queries.iloc[0]),
                     'Original': scores.loc['original', 'estimate'],
                     'Translation': scores.loc['translate', 'estimate'], 'HyDE': scores.loc['hyde', 'estimate'],
                     'T - original (points, 95% CI)': interval(delta.loc['translate-minus-original']),
                     'H - T (points, 95% CI)': interval(delta.loc['hyde-minus-translate'])})
    table(pd.DataFrame(rows), out, 'table1_target_matched_summary')
    same_prefix = {'bge-m3', 'granite-278m', 'LaBSE', 'SapBERT'}
    rows = []
    for model in MODELS:
        values = contrasts[(contrasts.model == model) & (contrasts.contrast == 'hyde-minus-translate')].set_index('corpus')
        assert len(values) == 3
        rows.append({'Encoder': MODEL_LABEL[model], 'Query/document prefix': 'Same' if model in same_prefix else 'Different',
                     **{CORPUS_LABEL[corpus]: interval(values.loc[corpus]) for corpus in CORPORA}})
    table(pd.DataFrame(rows), out, 'table2_encoder_hyde_contrasts')
    choices = pd.read_csv(ROOT / 'round_05/table02_held_out_paired_contrasts.csv')
    choices = choices[(choices.model == 'fixed_mean') & (choices.control == 'original')]
    rows = []
    names = {'english': 'Fixed English translation', 'cv_ndcg': 'Select for nDCG', 'cv_hit': 'Select for source Hit'}
    for corpus in CORPORA:
        for policy, label in names.items():
            values = choices[(choices.corpus == corpus) & (choices.policy == policy)].set_index('metric').rename(columns={'mean': 'estimate'})
            rows.append({'Corpus': CORPUS_LABEL[corpus], 'Policy': label,
                         'nDCG change (points, 95% CI)': interval(values.loc['ndcg10']),
                         'Source Hit change (points, 95% CI)': interval(values.loc['hit10'])})
    table(pd.DataFrame(rows), out, 'table3_selection_objectives')
    claims = [
        ('C1', 'Target-matched translation improves the three fixed-encoder corpus means', 'Supported within study',
         '+8.29 chemistry; +9.84 EUR-Lex; +6.27 UN nDCG points',
         'round_02/table02_target_matched_contrasts.csv;round_06/table01_target_contrast_sensitivity.csv',
         'Original-query cross-language settings only; equal source weight; fixed encoders; synthetic selected queries'),
        ('C2', 'HyDE increment over same-output-language translation depends on corpus and encoder', 'Supported within study',
         '-3.21 chemistry; +4.26 EUR-Lex; -4.22 UN nDCG points in fixed means',
         'round_02/table02_target_matched_contrasts.csv;round_04/table03_encoder_group_contrasts.csv;round_06/table03_fixed_encoder_influence.csv',
         'Single-passage implementation; four encoder prefix changes; no causal domain interpretation'),
        ('C3', 'Version-level ranking gains can coexist with fewer recovered sources', 'Supported within study',
         'UN E5 Tde-Ten: +71 version hits, -22 source hits, +3.81 nDCG points',
         'round_03/table02_un_decomposition_contrasts.csv;round_03/table03_un_source_and_version_counts.csv;round_03/table04_un_both_hit_contrasts.csv',
         'Known-source qrels; nDCG decomposition is not a new metric; all-eight nDCG contrast uncertain'),
        ('C4', 'A general German-pivot advantage', 'Rejected',
         'Fixed-mean nDCG +0.22 [-0.71,1.14]; source Hit -6.93 [-9.16,-4.81] points versus English',
         'round_02/table06_absent_pivot_contrasts.csv;round_03/table08_source_collapsed_contrasts.csv',
         'UN has no German target; corpus/model/objective-specific behavior'),
        ('C5', 'Source/version trade-off extends beyond top 10', 'Supported within study',
         'UN Tde-Ten at100: source Hit -3.49; version Recall +2.15 points',
         'round_06/table02_un_pivot_cutoff_and_quality.csv',
         'Saved cutoffs are correlated; top-1 point effect has an interval including zero'),
        ('C6', 'Training objective changes internally held-out policy behavior', 'Supported internally; external validity untested',
         'UN CV nDCG-vs-Hit across10splits: nDCG +1.41 to+2.26; Hit -3.54 to-1.92 points',
         'round_05/table02_held_out_paired_contrasts.csv;round_06/table05_policy_split_sensitivity.csv',
         'Adaptive internal CV; overlapping folds; split ranges are not independent replications or confidence intervals'),
        ('C7', 'Favorable corpus means ensure every query benefits', 'Rejected',
         'EUR-Lex encoder-specific nDCG losses affect21-35% of questions',
         'round_04/table01_query_gain_loss_rates.csv;round_04/table02_query_delta_quantiles.csv',
         'Targets averaged within source; gain/loss frequency differs from mean and source Hit'),
        ('C8', 'Query truncation explains the primary HyDE corpus pattern', 'Rejected for primary comparison',
         'No primary target-matched query truncations;3 truncations outside that comparison',
         'round_04/table08_query_embedding_truncation.csv',
         'Does not establish length irrelevance or eliminate corpus-chunking differences'),
        ('C9', 'HyDE retrieval failures require a visibly incorrect answer', 'Unsupported as a universal explanation',
         'Chemistry case: plausible upper-transverse location, accepted-source first rank1 to19',
         'round_04/cases/chemistry_bge_hyde_loss.txt;round_04/cases/analyst_notes.txt',
         'Rule-selected extreme; not a factuality study or causal mechanism test'),
        ('C10', 'Recorded generation accounting reconciles including failures and reuse', 'Audited descriptive fact',
         'Translation$0.230011; HyDE$0.49762452; four failed calls included',
         'audit/generation_accounting.json;round_05/table06_unique_generation_pool.csv',
         'Configured prices and observed cache history; not invoices; CPU/GPU prices unknown'),
        ('C11', 'Oracle headroom is a deployable routing gain', 'Rejected',
         'Per-query oracle uses held-out labels; only training-fold choices are fitted',
         'round_05/table01_held_out_levels_and_oracles.csv;round_05/source_folds.csv',
         'No effective per-query router or external generalization established'),
        ('C12', 'Every document offered exactly30 candidates', 'Rejected by provenance audit',
         '10,691 total; chemistry1950/EUR4050/UN4691; pools vary; all candidates included',
         'audit/candidate_pool_provenance.txt;data/selection_audit.csv',
         'Ten query generators differ from eight retrieval encoders; highest provided score is not independent human quality'),
    ]
    pd.DataFrame(claims, columns=['claim_id', 'claim', 'status', 'quantitative_evidence', 'evidence_files', 'limits']).to_csv(out / 'claim_evidence.csv', index=False)
    (out / 'table_notes.txt').write_text(
        'Table 1: absolute nDCG is on [0,1]; paired differences are percentage points. '\
        'Targets and fixed encoders are averaged within questions before source averaging.\n'\
        'Table 2: all model-specific H-T differences are target-matched cross-language nDCG percentage points. '\
        'Intervals are exploratory paired source bootstrap, not multiplicity adjusted.\n'\
        'Table 3: internal five-fold policy changes relative to original mixed retrieval, seed42; '\
        'intervals condition on fitted choices. Round6 provides split sensitivity. '\
        'These are not an untouched external evaluation.\n')


if __name__ == '__main__':
    main()
