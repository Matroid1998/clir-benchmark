# Translation and HyDE retrieval on the November 20 cohort

The [Narval configuration](../../../configs/verified-20261120-transformed-retrieval.yaml) uses the original 5,299 questions, the existing shared chunk corpus and 27 indexes, and the original target-language eligibility and chunk judgments. The retrieval jobs read saved generation outputs. No translation or HyDE API calls run in this campaign.

| Method | Chemistry lanes | EUR-Lex lanes | UN lanes | Evaluated dense runs | Evaluated hybrid runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Translate then retrieve | 7 | 4 | 6 | 136 | 136 |
| HyDE | 1 | 1 | 1 | 24 | 24 |

Translation evaluates each translated language as a separate run. HyDE uses the original language of each question. Dense uses the query adapter for translated questions and the document adapter for HyDE passages. Hybrid searches the same translated text or passage with BM25 and dense, then uses equal-weight reciprocal rank fusion over up to 1,000 candidates from each branch, with rank constant 60. Each corpus and generation lane has one shared BM25 candidate task. Each model and lane has one query-embedding task reused by dense and hybrid. A refused variant keeps its original question and judgments, records an empty ranking, and contributes zero to Recall@10. Its generation attempts, statuses and costs remain available for analysis.

Use the original dense run for each dense variant and the original hybrid run for each hybrid variant, matched on corpus, model, setting and query ID. The same top 100 ranking depth, qrels and metrics allow later mixed-language rescoring. The per-query reports retain original generation time, attempts, token usage, and recorded API cost where available. Search times are batch-amortized. Hybrid timings include the measured branch and fusion times. API request histories are shared across the eight models and both retrieval approaches.

The campaign directory on Narval is `/scratch/matroid/clir-benchmark/runs/narval-20261120-transformed-retrieval/`. Task logs are in `logs/`. Artifacts are in `artifacts/`, with read-only links to the original index and judgment files and the verified generation files.

On November 22, the checked translation receipt was published under its final name. All 17 translation outputs passed Narval preflight against the receipt. The launcher resubmitted the 12 canceled translation arrays and recorded their new IDs alongside the original attempts. At 6:59 p.m. EST, 17 BM25 tasks and 14 query-embedding tasks were running. Dense tasks wait for their query embeddings. Hybrids wait for the matching dense candidates and their corpus's BM25 candidates. [Translation verification](verified-translate.json) · [Array IDs and dependencies](submissions.tsv)

## Grouped translation jobs

The grouped layout uses 47 Slurm tasks. It retains the 17 original BM25 tasks and adds 6 query-embedding tasks, 12 dense tasks, and 12 hybrid tasks. An embedding task covers four models and all translation languages for one corpus. A dense or hybrid task covers two models and all translation languages for one corpus. The 272 evaluated retrieval combinations each keep their own output identifiers and reports. Each grouped task records its individual language and model outcomes in `job_status/`.

The migration cancels only pending fine-grained embedding, dense, and hybrid tasks. Already-running tasks can finish and publish results. The grouped jobs wait for those tasks as needed and reuse completed artifacts. The original and grouped job IDs appear in the same ledger.

## November 20 search timings

The [per-job measurements](earlier_search_times.csv) cover the 27 original-question and 27 HyDE BM25 and dense search jobs. Each row includes the Slurm outcome, Toronto timestamps, elapsed time, CPU allocation and time, and a count of the searches executed. The [export script](earlier_search_times.py) sums each run’s batch-amortized search timings across query and language-setting pairs. For UN BM25 this accounts for 17 h 02 min of the 17 h 06 min original job and 41 h 05 min of the 41 h 06 min HyDE job, rounded to the nearest minute. The three failed original dense jobs also saved their rankings and timing files before their reporting checks failed.
