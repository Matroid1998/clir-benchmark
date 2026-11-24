# Local retrieval results

[retrieval_settings.csv](retrieval_settings.csv) contains the recorded metrics for each run and search setting. Filter `report_status` to `published` for the 364 runs with published analysis manifests. The `report_dir` column gives the path from the repository root to that run’s report, per-query metrics and detailed rankings. [collect_settings.py](collect_settings.py) rebuilds the table from the local reports.

| Source | Local files |
| --- | --- |
| Translation and HyDE | [Reports](../../../artifacts/narval-20261120-transformed-retrieval/artifacts/reports), [raw rankings, candidate lists and timings](../../../artifacts/narval-20261120-transformed-retrieval/artifacts/runs), [logs](../../../artifacts/narval-20261120-transformed-retrieval/logs), [final Slurm accounting](../../../artifacts/narval-20261120-transformed-retrieval/job_status/slurm-final.psv) |
| Original queries | [Reports](../../../artifacts/narval-20261119-shared-indexes/artifacts/reports), [raw rankings, candidate lists and timings](../../../artifacts/narval-20261119-shared-indexes/artifacts/runs), [logs](../../../artifacts/narval-20261120-verified-retrieval/logs), [completion record](../../../artifacts/narval-20261120-verified-retrieval/provenance/completion.json) |
| Shared data | [Chunks and source documents](../../../artifacts/narval-20261119-shared-indexes/artifacts/corpus), [evidence and relevance judgments](../../../artifacts/narval-20261119-shared-indexes/artifacts/relevance), [generation outputs and API records](../../../artifacts/verified-20261118-query-generation) |

Four original runs produced metrics before final verification failed. Their rows carry `report_status=incomplete`. These are Chemistry Arctic dense, UN EmbeddingGemma dense, UN Granite dense and UN Nomic hybrid. The corresponding original hybrid jobs for Chemistry Arctic, UN EmbeddingGemma and UN Granite were canceled before producing results. The 272 translation and 48 HyDE reports all have published manifests.
