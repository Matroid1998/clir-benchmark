Baseline results analysis
=========================

This folder contains code and outputs for the user's questions about the saved
experiments. Question 1 is Recall@10 by baseline, embedding model, corpus and
language. Later questions can have their own scripts/output folders here.

Question 8 investigates corpus difficulty using actual questions, relevant source
texts and saved rankings. Open question_08_chemistry_difficulty/index.html and
findings.txt. Its REPRODUCE.txt documents three offline scripts, the seeded text
sample, language/quality controls and a bounded corpus-size counterfactual.

Question 2 isolates original-query dense retrieval (no translation or HyDE).
Run .venv/bin/python report/baseline_analysis/dense_only.py to reproduce
question_02_dense_only/. Its figure shows all three search regimes; the output
also retains the original-language/target-language table. The script verifies
the Question 1 table hashes before extracting the 24 dense model/corpus reports.

Questions 6–7 compare translation and HyDE with embedding-only retrieval. The
shared driver un_generation_effects.py defaults to UN / EmbeddingGemma. Select
another corpus/model with --corpus chemistry|eurlex|un and --model MODEL_ALIAS;
--output selects a separate destination. Each output contains numeric scores,
signed percentage-point changes, same-language HyDE-versus-translation deltas,
and PNG/PDF/SVG figures. For the Question 7 comparison, reproduce with:

  .venv/bin/python report/baseline_analysis/un_generation_effects.py \
    --corpus chemistry \
    --output report/baseline_analysis/question_07_generation_effects/chemistry
  .venv/bin/python report/baseline_analysis/un_generation_effects.py \
    --corpus eurlex \
    --output report/baseline_analysis/question_07_generation_effects/eurlex

Each comparison checks identical query/setting keys, qrel/corpus/query identities,
model specifications and retrieval settings, and reconstructs saved Recall@10
and Hit@10 macros from per-query results. It makes no inference or API calls.

Open question_01_recall_at_10/index.html for a searchable results table and all
figures. Figures are exported as PNG, PDF and SVG. results.txt contains the
numerical overview. tables/recall_at_10_by_setting.csv contains every evaluated
original-query-language / searched-corpus-language pair, and separate generated
input languages. tables/recall_at_10_by_regime.csv matches saved regime macros.
The three selected_documents_*_percent.csv files are compact, human-readable
model-by-variant tables, with one row per corpus/model. Other CSV recall columns
are fractions in [0,1]; columns ending _pct and the figure labels are percentages.
Control differences ending _pp are percentage points. Missing controls stay blank.

Reproduce from the repository root:

  .venv/bin/python report/baseline_analysis/recall_at_10.py

Select a subset and write a separate output to preserve the complete report:

  .venv/bin/python report/baseline_analysis/recall_at_10.py \
    --cohort selected_documents --corpus chemistry --model bge-m3 \
    --baseline dense --baseline translate-dense \
    --output /tmp/clir-recall-chemistry-bge-m3

Repeated --baseline, --corpus, --model and --cohort select subsets; omitted filters
include all saved modes. A subset excluding dense has unknown dense-control
differences. BM25 has model alias 'none'. --artifacts and --campaign override the
input roots. Use a separate output directory for each selection to avoid leaving
old figures from a previous, wider selection in that directory.

Only existing JSON result files are read. No benchmark imports, retrieval/model
inference, remote jobs, credentials or API calls are needed. Inputs and outputs
are SHA-256 hashed in manifest.json; exact package versions are recorded there
and pinned in requirements.txt. The campaign's normalized final status and latest
successful stage journals select the 264 comparable document reports independently
of the earlier analysis folder. Another 34 earlier chunk reports (dense, BM25,
hybrid in chemistry/EUR-Lex) appear in separate tables/figures by cohort.

Measurement and comparisons:
- Recall@10 is the per-question fraction of known relevant units in the top 10.
  For document retrieval, language versions count as distinct relevant units.
  Mixed recall therefore measures language-version coverage, not source success.
- Macro averages give equal weight to each nonempty setting, exactly as the saved
  benchmark does. Mixed macros weight original query languages equally. They are
  not a pooled average over all questions. Each language-pair cell is a query mean.
- Regimes 'monolingual' and 'cross_language' refer to original query language
  versus target corpus language, even when generated inputs change language.
- Translation and HyDE each evaluate de/en/es/fr/zh separately. These are the
  generated INPUT languages; the language-pair plots separately show searched
  corpus languages. German generated inputs are valid on UN, and Chinese ones
  are valid on EUR-Lex, although those corpus languages are absent as targets.
- Grey/dash cells have no eligible questions, rather than zero recall. Chemistry
  eligibility differs by language pair; the plot counts include a cell with n=1.
- Dense controls match cohort, corpus, retriever, corpus/query/qrel fingerprints
  and regime/setting. HyDE also has a same-output-language translation control.
  BM25 has no embedding model; compare it within the earlier chunk cohort.
- The script checks matching query/setting keys and retrieval settings within
  each corpus/cohort, and pinned model specifications within each retriever.
- Saved per-query metrics are independently averaged and checked against both
  report.json and mteb.json with tolerance 1e-12. Rankings/relevance judgments
  are not re-scored. Values reflect known-source judgments, not exhaustive labels.
- This is descriptive reporting with no new significance tests or model/language
  selection claim. Corpus granularity differs (patent abstracts, legal articles
  and annexes, UN documents); cross-corpus scores are separate results.

Validation:

  .venv/bin/python -m pytest -q report/baseline_analysis/test_recall_at_10.py
  .venv/bin/ruff check report/baseline_analysis

Offline fixtures check aggregation weighting, matched controls, unknown/missing
comparisons, disabled-baseline file access and rejection of inconsistent MTEB
values. validation.json records checks of all delivered real-data aggregates.
