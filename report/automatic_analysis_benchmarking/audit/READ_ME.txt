CLIR RESULT AUDIT FOR PAPER ANALYSIS
==================================
Scope
-----
The primary comparable experiment is 264 document-unit reports: 8 retrievers x 3 corpora x (original dense + five translated dense + five HyDE dense). The separately completed EUR-Lex hybrid report uses a different, all-question chunk-level experiment and is NOT a directly comparable 265th arm. The report index here follows the latest successful Slurm attempt per job label, not an indiscriminate reports/ glob.

Selected report index: selected_report_index.json
Integrity checks: comparison_integrity_audit.json
Cohort/schema summary: cohort_schema_audit.json
Source granularity: source_granularity.json
Encoder roles: encoder_role_audit.json
Query quality flags: quality_audit_flags.json
Identity-control diagnosis: identity_control_diagnosis.json

Joining records
---------------
Per-query JSON is a list. Metrics are nested under metrics. A unique within-report row is (setting_id, query_id). Across experiments use (corpus, model, baseline, output_language, setting_id, query_id), where output_language is report.output_language for HyDE, report.translation_language for translation, and original for dense.

All 264 reports have identical eligible (setting_id, query_id) keys and matching corpus_fingerprint, query_fingerprint, relevance_hash and model specification within corpus/retriever. This was checked on all 247,192 query-setting rows. Chemistry has 604 rows and 25 nonempty settings per report; EUR-Lex 970/20; UN 1235/20.

Query metadata is in relevance/<corpus>/<relevance_hash>/queries.parquet. Important columns: id, question, question_language, source_unit_id, generator_model_id, persona, total_score, metadata_json, evidence_alignment_eligibility. metadata_json holds quality_audit_status and grader sub-scores. selection.json is a LIST of per-canonical-source audit dictionaries, not one dictionary. Each entry gives canonical_source_id and selected_ids; use it to map query_id to canonical source. Do not use query_id string parsing.

The query_language and regime fields always describe the ORIGINAL query and the target corpus, even after translation/HyDE. Therefore an arm called cross_language may encode a query in exactly the target language. Label original language, generated input language, and search-target language separately in plots.

Metric definitions
------------------
CUTOFFS: 1,5,10,20,100. Recall@k = number of known relevant document-language versions in top k / number of known relevant document-language versions. Hit@k = any such hit. SourceRecall@k counts required canonical source groups recovered / required groups. MRR@10 is reciprocal first relevant version rank within top 10; zero if absent. nDCG@10 uses binary gain and an ideal ranking of min(10, number of relevant versions).

ALL 638 selected questions have exactly one required canonical source. Accordingly Hit@k, SourceRecall@k and AllSourceSuccess@k are exactly equal throughout all 264 reports. They are not three independent signals. In pairwise search, each question has one relevant language version, so Recall@k is also equal to Hit@k. In mixed search, EUR-Lex and UN each have four relevant versions per question; chemistry has two for 184 questions and three for 13. Mixed Recall is therefore a language-version coverage metric and can disagree with source-finding success. Document nDCG also rewards recovering several versions of the same source.

report.metrics.macro_by_regime equally averages nonempty SETTING means. It does not average all questions equally. Mixed macros therefore equally weight original query languages. If a new analysis uses per-question means, label that estimand explicitly and report both weighting choices as sensitivity where consequential. Do not pool repeated model/language/target-setting observations as independent data. Bootstrap original query/canonical document clusters; stratify corpora and optionally original languages according to the declared estimand.

Corpus coverage and granularity
-------------------------------
Chemistry retrieval units are patent ABSTRACTS (20,283 source-language units over 9,432 canonical patent identifiers). EUR-Lex retrieval units are ARTICLES and ANNEXES (340,833 versions over 85,509 canonical units; selected questions all refer to articles). UN units are WHOLE DOCUMENTS (88,134 versions over 14,689 canonical documents).

Configured target languages are de/en/es/fr/zh, but not all are present: EUR-Lex has no Chinese documents; UN has no German documents. Those pairwise target settings are dropped for every query and must be marked unavailable, never scored as zero retrieval. Chinese EUR-Lex and German UN GENERATED INPUTS are still valid experiments against existing target languages. Chemistry has partial pair availability and one nonempty cross-language setting with only ONE question; show denominators and avoid strong inference from sparse cells.

The full input corpus also contains Japanese/Russian chemistry units and Arabic/Russian UN units, but these are not searched by the configured five-language target set. The mixed target label is the configured set, not evidence that every listed language is present.

Question selection and limitations
----------------------------------
638 winners: chemistry197, EUR-Lex194, UN247. The candidate data contains 10 generator IDs, with median available candidates per canonical source 10,22,20 respectively, not uniformly 30. Max-score ties occur for 175/197,171/194,219/247 sources. Ties favor three named generators equally, then use a deterministic per-source random choice with seed42. Thus generator representation and language representation are selection-biased; these are curated synthetic questions, not an unbiased human query sample. Do not interpret the winning-generator histogram as a controlled generator-quality evaluation.

Raw winning total_score ranges 14 to37; most are32. Do not assume32 is a universal ceiling or interpret score as a calibrated probability without auditing the original grading rubric. Selection ignores is_best and candidate_rank and uses the highest numeric total_score before the preferred tie rule.

Quality audit statuses:581 clear,42 repair,13 blocking,2 minor_issues. Alignment metadata includes55 no_exact_source_passage and1 quarantined_source_language. These were intentionally retained under source-association document relevance. Primary analysis should keep all638; a clear-only subset is a robustness analysis, not a replacement. Missing exact spans alone do not establish an invalid question.

Qrels are known-source associations, not exhaustive human judgments of every potentially relevant retrieval. Legal variants share canonical source_id; chemistry positives are limited to the original unit plus directly accepted translation pairs. No transitive chemistry pair closure is allowed. An unjudged document with identical or similar content is not automatically a false-positive, nor can it automatically be relabeled relevant without validating source-specific constraints.

Ranking analysis
----------------
analysis/rankings.parquet exposes setting_id, query_id, rank, score, relevance, required_source_ids, source_unit_id, source_id, parent_id, document_language, document_id, winning_chunk_id, text, start, end. Project only needed columns and filter mixed setting IDs; text is large and unnecessary for most metrics. First-relevant rank, first-hit language, and top-k language composition can be read directly. Existing relevance flags preserve the accepted qrels.

Canonical source collapse is well-defined by (corpus, source_id) for legal variants. For chemistry, preserve qrel membership/required_source_ids rather than promoting a same-source-id but unaccepted language variant to relevant. An exploratory deduplication of source identities must disclose this rule. Rankings retain only top100 document-language versions: collapsing them yields fewer than100 unique sources and cannot reconstruct an exact top100 source-level retrieval ranking beyond the saved boundary. A failure to find a source in the saved top100 is censored at that boundary, not proof of global absence.

Scores aggregate the maximum chunk cosine over ALL searched chunks before selecting100 unique document-language units; winning chunks are retained. Document length/chunk count can affect max-pooling opportunities and corpus granularity differs substantially. Such associations are observational, not causal effects of length.

Generation evidence and encoder-role confound
--------------------------------------------
Translation payload construction consumes question text, source_language, target_language and a fixed corpus-domain system prompt. HyDE consumes question text, source_language, output_language, collection/domain and fixed prompt. Neither supplies source text, answers, evidence spans, qrels, candidates or retrieved documents. This is query-only inference with domain/task instructions, not proof that model pretraining lacked the public source documents. Prompt instructions preserve terminology, citations and constraints; actual output faithfulness still needs analysis.

HyDE passages are encoded with the DOCUMENT adapter, whereas original/translated queries use the QUERY adapter. Gemma and Nomic switch prefixes; Qwen and E5 remove the query instruction for document encoding. BGE-M3, Granite, LaBSE and SapBERT have empty prefixes for both roles. Max token lengths are identical across roles within a model but differ across models (see encoder_role_audit.json). Therefore language-controlled HyDE minus translation bundles content expansion with adapter-role change for four models. Do not call it a pure expansion causal effect. Token truncation audits are available per embedding row and report; generated passages can truncate when short queries do not.

Identity-translation numerical diagnostic
----------------------------------------
Identity translation preserves the query text exactly and costs zero. All six retrievers other than Qwen/Nomic have identical identity metrics. For Qwen/Nomic,16 query-model cases covering14 distinct questions have26 changed query-setting metric rows. Text, token audits, model specification and runtime configuration are identical. Minimum double-precision cosine between stored vectors is0.9999999999962099, maximum coordinate difference3.5018e-7, and maximum matched-document score difference5.3644e-7.

In ALL26 changed rows, a relevant/unjudged pair with identical winning-chunk text changes relative order. Crossed-pair score gaps are at most1.1921e-7. This supports near-tie numerical sensitivity, not substantive translation gains. It does not identify a specific numerical backend cause without new controlled execution. Exact duplicate winning text also does not prove both whole sources are interchangeable. Keep original results, flag these14 queries, and report sensitivity excluding them or canonically treating identity inputs as the no-call control, with the intervention stated explicitly.

Cost and timing
---------------
Generation artifacts are shared across target settings and all eight retrievers. Deduplicate by corpus/method/output language/query or by request key, retaining failed/retried request ancestry. Do not sum per-query setting rows or report summaries over retrievers. The 30 unique corpus/method/output-language summaries agree across all eight retrievers.

Translation:3190 query-language records,638 identity records,2552 API attempts; estimated USD0.230011. HyDE:3190 generated records,3194 recorded attempts (including retries); estimated USD0.49762452. These are saved configured token-rate estimates, not verified provider invoices, and they exclude unpriced GPU/CPU/storage resources and dataset-generation costs. Costs reflect actual cached-input pricing; a fresh uncached deployment may cost differently. Policy costs should count only the generation chosen by that policy, with common work charged once; an oracle evaluating all languages may require all language generations.

Generation seconds sum recorded API attempt durations, including historical cached work. Retrieval seconds are measured batch search wall time divided by batch query count; encoding seconds are cached embedding build wall time divided by rows. Encoding measurement begins after model construction and includes offline token audit/output work. The sums are offline component-time estimates, not online end-to-end latency. Model loading, Slurm queues, serialization/export, some retry waiting, and hardware pricing are excluded. Different GPU partitions/slices and heterogeneous cluster execution make cross-model latency comparisons less controlled than paired retrieval-quality comparisons.

Inference policy
----------------
Six adaptive exploratory rounds are hypothesis generation. Avoid presenting post-hoc best-language/model choices on the evaluation queries as deployable improvements. Oracle values are upper bounds; learned routing needs train/test separation by canonical document (and domain shift sensitivity) or nested cross-validation. Prespecify any final confirmatory metric/contrast before held-out evaluation. Account for multiple comparisons in claims; provide paired uncertainty, effect size and query/corpus heterogeneity rather than p-values alone. Broad community claims need additional corpora, non-synthetic queries and external validation.
