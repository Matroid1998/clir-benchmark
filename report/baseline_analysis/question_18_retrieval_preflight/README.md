# Retrieval on the November 20 verified questions

BM25, eight dense models and eight BM25–dense hybrids use the same shared chunks and evidence judgments within each corpus. The queries are the original published questions. Only a question's text goes into retrieval. Its answer and evidence are used to score the returned chunk IDs.

| Corpus | Questions | Indexed chunks | Verified snippets | Snippets crossing a boundary with no containing chunk | Eligible question–setting pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Google Patents Chemistry | 1,775 | 707,024 | 3,677 | 970 | 5,452 |
| EUR-Lex | 1,826 | 1,058,996 | 7,343 | 149 | 9,130 |
| UN | 1,698 | 2,025,156 | 10,188 | 245 | 11,886 |

## How to read Recall@10

For each eligible question and search setting, divide the number of distinct relevant chunks returned in the top 10 by the total number of distinct relevant chunks in that setting. A chunk is relevant when it shares at least one character with a verified snippet in the corresponding source and language edition. Repeated matches in a source count at each verified occurrence. A chunk matching more than one snippet still counts once.

Each single-language search uses that target language's judgments. The mixed search uses the union of the judgments across all supported editions and searches the whole multilingual corpus. A chemistry question is eligible for a single-language search only if its supporting patent has evidence in that language.

In UN mixed retrieval, 77 of 1,698 questions have more than ten relevant chunks. Even with perfect ranking, the highest possible mean Recall@10 is **98.59%**. The corresponding ceilings are **99.97%** for Chemistry and **99.97%** for EUR-Lex. These figures average each question's maximum possible recall, `min(10, relevant chunks) / relevant chunks`. [Calculation](recall_ceiling.py) and [counts by setting](recall_ceiling.json).

## Runs and verification

The [configuration](../../../configs/verified-20261120-retrieval.yaml) pins the QA and corpus revisions. Each hybrid uses reciprocal rank fusion with equal weights and rank constant 60, taking up to 1,000 candidates from BM25 and 1,000 from its matched dense model in the same language pool. Each run exports the top 100 and evaluates Recall@10 from that ranking.

The [boundary audit](preflight.py) checks the published offsets against the indexed chunk boundaries. Its [results](boundary_coverage.json) count each snippet once. The independent [cohort checker](../../../scripts/check_verified_retrieval_inputs.py) compares the prepared queries, eligible settings and mixed-language judgments with the pinned QA release. Its receipts are available for [Chemistry](check-chemistry.json), [EUR-Lex](check-eurlex.json) and [UN](check-un.json).

The Slurm [submission record](submissions.tsv) lists the stage, job ID, prerequisite job IDs and configuration fingerprint, in that order. A search job evaluates its own rankings. Narval keeps the job logs under `/scratch/matroid/clir-benchmark/runs/narval-20261120-verified-retrieval/logs` and the shared indexes, runs and reports under `/scratch/matroid/clir-benchmark/runs/narval-20261119-shared-indexes/artifacts`.

Slurm accepted the 75 retrieval and query-encoding jobs between 12:53:20 and 12:55:29 p.m. EST on November 20, 2026. In the [12:55 p.m. scheduler snapshot](slurm_at_submission.psv), the three preparation jobs had succeeded, all three BM25 and 24 query-encoding jobs were running, and 24 dense and 24 hybrid jobs were pending.

## Files for later analysis

| Analysis | Saved files |
| --- | --- |
| Rescore mixed Recall@10 or group hits by language or document | `runs/<corpus>/<run>/run.parquet` contains scored ranks through 100. `reports/<corpus>/<run>/analysis/rankings.parquet` adds query text and language, document language, source and chunk IDs, character offsets, chunk text, and original relevance labels. |
| Change the relevance rule | Per-corpus `relevance/` contains the queries, individual evidence matches, chunk judgments, eligibility, and language settings. The original QA Parquets retain published evidence in every supported edition. The shared chunk table supplies the original chunk boundaries and texts. |
| Recompute hybrid rank fusion | Each BM25 and dense component saves up to 1,000 ordered chunk IDs per eligible query and search pool in `candidates.parquet`. Raw scores are retained through rank 100. |
| Compare cost and latency | Per-query `timings.parquet` records batch-amortized search times. Hybrid records both component times and fusion time. Query embedding manifests record encoding duration. Slurm accounting preserves allocations, elapsed time, CPU time and available peak-memory records. These original-query baselines make no paid API calls. Cluster compute is unpriced. |
| Reproduce and audit | Pinned dataset and model revisions, configuration, executed source, software versions, all 24 document-vector sets, query vectors, BM25 indexes, per-job stdout/stderr, receipts, and scheduler accounting accompany the archive. Each completed result has an `analysis_manifest.json` with relative file paths, sizes and SHA-256 checksums. |

Local backups of all nine QA/evidence/source-judgment files and 185 prepared judgment, source-code and configuration files passed SHA-256 verification against Narval. [QA checksums](../../../artifacts/narval-20261120-verified-retrieval/qa.sha256) · [Code and judgment checksums](../../../artifacts/narval-20261120-verified-retrieval/provenance/submission-file-hashes.sha256)

The private persistent destination on Narval is `/project/def-mahyarh/matroid/clir-benchmark/narval-20261120-verified-retrieval/`. Job `9964409` writes the shared inputs and indexes to `base/archive.tar`. Job `9964111` waits for the campaign jobs and the base copy to terminate, then verifies 51 result manifests, 78 job receipts, and the files referenced by the manifests. A successful verification publishes `final/archive.tar`. If a job or required file is missing, the archive saves the available outputs under `partial-<jobid>/` with the diagnostic report. Each published bundle has a member list, a SHA-256 checksum and a readback receipt. Extract the base and final bundles into the same directory to restore the shared artifacts and result files in their original relative layout. [Archive job IDs and dependencies](../../../artifacts/narval-20261120-verified-retrieval/provenance/archive-jobs.tsv)
