# Translation and HyDE retrieval on the November 20 cohort

The [Narval configuration](../../../configs/verified-20261120-transformed-retrieval.yaml) uses the original 5,299 questions, the existing shared chunk corpus and 27 indexes, and the original target-language eligibility and chunk judgments. The retrieval jobs read saved generation outputs. No translation or HyDE API calls run in this campaign.

| Method | Chemistry lanes | EUR-Lex lanes | UN lanes | Evaluated dense runs | Evaluated hybrid runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Translate then retrieve | 7 | 4 | 6 | 136 | 136 |
| HyDE | 1 | 1 | 1 | 24 | 24 |

Translation evaluates each translated language as a separate run. HyDE uses the original language of each question. Dense uses the query adapter for translated questions and the document adapter for HyDE passages. Hybrid searches the same translated text or passage with BM25 and dense, then uses equal-weight reciprocal rank fusion over up to 1,000 candidates from each branch, with rank constant 60. Each corpus and generation lane has one shared BM25 candidate task. Each model and lane has one query-embedding task reused by dense and hybrid. A refused variant keeps its original question and judgments, records an empty ranking, and contributes zero to Recall@10. Its generation attempts, statuses and costs remain available for analysis.

Use the original dense run for each dense variant and the original hybrid run for each hybrid variant, matched on corpus, model, setting and query ID. The same top 100 ranking depth, qrels and metrics allow later mixed-language rescoring. The per-query reports retain original generation time, attempts, token usage, and recorded API cost where available. Search times are batch-amortized. Hybrid timings include the measured branch and fusion times. API request histories are shared across the eight models and both retrieval approaches.

The campaign directory on Narval is `/scratch/matroid/clir-benchmark/runs/narval-20261120-transformed-retrieval/`. Task logs are in `logs/`. Artifacts are in `artifacts/`, with read-only links to the original index and judgment files and the verified generation files. Slurm waits for the translation checksum receipt before starting any translation retrieval. [Slurm arrays and dependencies](submissions.tsv)
