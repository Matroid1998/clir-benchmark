# November 18 release chunk indexes

Narval accepted the following jobs on November 19, 2026. Each job prepares the
published language editions, builds or reuses the chunk-level BM25 index, and
validates the SQLite file against the prepared chunk count.

| Release | Corpus rows | Language editions | Slurm job ID |
| --- | ---: | ---: | ---: |
| Google Patents chemistry | 332,578 publications | 574,974 | 9878730 |
| EUR-Lex | 79,193 articles + 20,000 annexes | 395,525 | 9878731 |
| UN | 14,689 documents | 88,134 | 9878733 |

The campaign uses `/scratch/matroid/clir-benchmark/runs/narval-20261119-verified-bm25`.
The `inputs/` directory contains the published corpus shards. The `artifacts/corpus/`
directory holds prepared chunks and `artifacts/lexical/<corpus>/<fingerprint>/index.sqlite`
holds each published index. A successful worker writes the final index path to
`job_status/<corpus>-<config-hash>.index`. Worker output is in `logs/`.

All available language editions participate. Each chunk has a maximum length of
512 BGE-M3 tokens including special tokens and a nominal overlap of 64 tokens.
The published corpus text includes titles where present for patents. EUR-Lex
articles and annexes are separate retrieval units. Each dataset's 2,000 QAs can
use its corresponding corpus index.

The [configuration](../../../configs/verified-20261118-bm25.yaml) pins the releases
and tokenizer. Narval's [input verification log](verified-inputs.log) records all
20 corpus and annex shards and three release manifests checked against the
[SHA-256 list](corpus_inputs.sha256). The [submission record](submissions.tsv)
maps the jobs to the configuration hash.
