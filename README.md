# CLIR benchmark

A Python package for reproducible, span-grounded dense retrieval over EUR-Lex,
UN documents, and chemistry patent abstracts. It prepares one fixed chunk corpus
and relevance judgments shared by every embedding baseline.

Preparation, embedding, retrieval, and scoring are separate CLI commands. Nothing
runs a benchmark automatically. Tests use synthetic corpora and lightweight
encoders; the tokenizer integration test downloads only BGE-M3 tokenizer files.

## Install and inspect

Requires Python 3.11+. Dependencies are pinned in `uv.lock`.

```bash
uv sync --extra test
uv run clir-benchmark --help
uv run clir-benchmark validate
uv run clir-benchmark list-models
```

The default configuration is [`configs/benchmark.yaml`](configs/benchmark.yaml).
Paths inside YAML resolve relative to the YAML file. The global `--config` flag
precedes the subcommand:

```bash
uv run clir-benchmark --config configs/benchmark.yaml validate --inputs
```

`validate` checks configuration/model definitions without downloading models.
`--inputs` additionally reads QA exports, validates multilingual annotation shapes
and overlays, and verifies candidate selection. Source associations and literal
span matching are checked during relevance preparation.

`HF_TOKEN` is read from the environment or configured `.env` file. Only that
variable is consumed; unrelated API keys are not loaded. `.env`, data, vectors,
and reports are ignored by Git. The token must have access to BASF and any gated
model being used.

## Prepare sources and chunks

```bash
uv run clir-benchmark prepare-corpus --corpus eurlex
uv run clir-benchmark prepare-relevance --corpus eurlex
```

Repeat `--corpus` to select multiple corpora, or omit it for all configured corpora.
Preparation never loads an embedding model or computes retrieval scores.

| Corpus | Retrieval input | Independent source unit |
| --- | --- | --- |
| EUR-Lex | `MehdiAstaraki/eurlex-generated-qa`, `corpus` | One article or annex in one language |
| UN | `MehdiAstaraki/un-generated-qa`, `corpus` | One complete document in one language |
| Chemistry | Both endpoints of `BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs` | One publication's stored abstract in one language |

Questions come from each `MehdiAstaraki/*-generated-qa` dataset's `qas`
configuration. Chemistry's small QA-associated `corpus` is used only to verify
source IDs, exact text, language, and upstream revision against the full corpus.
The specified BASF snapshot contains abstracts and no titles.

Source text is retained exactly as stored, including markup and upstream cleanup;
the benchmark performs no additional cleanup. Repeated chemistry
`(doc_id, language)` endpoints are deduplicated only if their text agrees;
conflicts are errors. Identical text in distinct units remains separate.
Rejected pair endpoints remain retrievable. Only `pair_ok=true` with verdict
`exact/high` creates a direct translation association. Neither transitive links
nor patent-family relationships create positives. Original pair rows are saved.

UN generation blocks remain provenance only. Every corpus unit is retained,
regardless of QA selection or eligibility. Empty sources remain single empty chunks.

All revisions must be full commit SHAs. The default QA pins include the published
`cross_language_evidence_terra_v1` annotations from
[EUR-Lex](https://huggingface.co/datasets/MehdiAstaraki/eurlex-generated-qa/tree/c924c03d0ac73ee45a3b6910d420fe523e433d2d),
[UN](https://huggingface.co/datasets/MehdiAstaraki/un-generated-qa/tree/677117c2e82908933821ba68dfd122d273c20be0), and
[chemistry](https://huggingface.co/datasets/MehdiAstaraki/chemistry-generated-qa/tree/bb2657593e877d75e747a4602468eb0add634895).
The corpus files are unchanged, so their pins and chunk caches are preserved.
Update `qa.revision` for later QA releases; retain `corpus.revision` if source
documents are unchanged. A source
can instead specify `local_dir`, with the same `data/<configuration>/*.parquet`
layout or `data/train-*.parquet` for BASF. Local file hashes are recorded.

### Chunking contract

- Default maximum: **512 BGE-M3 tokens including special tokens**. The pinned
  tokenizer adds two special tokens, leaving 510 nominal content tokens.
- Default overlap: **64 source content tokens**. Original token offsets determine
  windows. Cuts avoid splitting overlapping character-offset intervals. Actual
  overlap after adjustment is recorded in characters and re-tokenized content tokens.
- Each chunk is exactly `source.text[start:end]`: Unicode code-point offsets,
  start-inclusive/end-exclusive, with whitespace and markup intact. The union of
  chunks covers the complete source.
- Each slice is re-tokenized with special tokens and shortened deterministically
  until it fits. An unrepresentable safe boundary is an error. Token IDs are never
  decoded to reconstruct text.
- Short units remain single chunks. Different units/languages are never combined.

The supported boundary policy is `source-token-windows-v1`. Token limit and
overlap are configurable within the 512-token ceiling. The resolved policy is
recorded in the manifest and referenced by each chunk's `chunking_hash`.
Chunk IDs and the content-based corpus fingerprint do not depend on questions,
evidence, candidate selection, or source-row order.

## Candidates and multilingual evidence

`candidate_mode: best` selects one candidate per original generation task **and
generator model**. Legal task identity uses the generation file and provenance
task ID. Chemistry uses the generation file and `decider_routing_task`.

A unique exported `is_best=true` wins. Without one, select the highest grader
`total_score`, then ascending candidate rank, then QA ID; missing ranks sort
last. Missing task identity, multiple marked winners, or scores needed for
selection are errors. `candidate_mode: all` retains all candidates without
requiring scores. Selection precedes relevance filtering: a dropped winner is
never replaced. There are no additional quality thresholds.

At the included snapshot pins, read-only QA validation found **23 EUR-Lex task
groups and 22 UN task groups** without an explicit winner and with missing grader
scores. Chemistry candidate selection validates. Strict `best` mode stops on
these groups as specified; it never silently drops them or switches to `all`.
Correct the selection/grading metadata before preparing these best-mode cohorts,
or deliberately choose the supported all-candidates experiment. The local
`artifacts/input-validation-latest.json` report lists affected QA IDs and task
identities from the updated exports. All 10,691 exported QAs pass annotation-shape
validation; this does not assert that every QA has support in every setting.

The published QA field contains individual, source-specific spans:

```json
{
  "evidence_alignment_contract": "cross_language_evidence_terra_v1",
  "evidence_alignment_eligibility": "eligible",
  "evidence_all_languages": {
    "fr": [
      {
        "source_unit_id": "exact corpus unit ID",
        "language": null,
        "text": "A complete verbatim supporting span.",
        "origin": "cross_language_alignment",
        "anchor_id": "source-anchor",
        "alignment_id": "alignment-id",
        "validation_status": "pending_review",
        "match_status": "exact"
      }
    ]
  }
}
```

Each `text` is checked independently; several spans may refer to the same unit.
The language key is authoritative when a record's `language` is null, and must
agree with the corpus unit. Separate indispensable articles need source-specific
records. Alignment flags never establish relevance: spans marked `pending_review`
are accepted only when the literal substring and chunk-containment checks pass.
The original span record, including anchor IDs, origin, validation status, and
generation provenance, is preserved in each match's `annotation_json`.

`evidence_alignment_status` uses the same language-keyed lists with source IDs
and `status` values. `not_found`, `uncertain`, and `invalid` are supplied outcomes
with no qualifying span. Their reasons remain diagnostics, never searchable
evidence. `matched`/`source_exact` without a span, unknown statuses, and bare empty
language lists do not establish that annotation is complete. Missing annotations
block an affected setting; a source with no version in that language is separately
reported as unavailable.

Under this published contract, `no_exact_source_passage` and
`quarantined_source_language` are completed alignment exclusions. For language keys
present in both annotation mappings they supply no translated spans. They are
recorded as provenance and do **not** suppress independently qualifying raw
answers or evidence. Other eligibility values do not imply completion.

The release covers **en, de, es, fr, zh**, now the default target languages.
All available corpus languages are still prepared and cached. Configure additional
targets when their annotations are available, or use `target_languages: []` to
request every corpus language and report missing annotations. The convenience
fields `evidence_en`, `evidence_fr`, etc. are not used because they lack source IDs.

The original overlay format remains supported: one record per source-language
unit with `source_unit_id`, an `evidence` list, and an optional string `answer`.
An explicitly supplied `evidence: []` record can establish annotation completion
without a positive span. Duplicate aggregate records, language/ID mismatches,
and unassociated sources remain validation errors.

Alternatively, set `evidence_file: ../annotations/evidence.jsonl`. Each line has
`query_id` (or `_id`) and the complete `evidence_all_languages` mapping for that
QA, in either supported format. Native overlays may also provide
`evidence_alignment_status`, `evidence_alignment_contract`, and
`evidence_alignment_eligibility`. An overlay replaces all four annotation fields
together (omitted alignment fields are cleared) and preserves raw answer/evidence.
An overlay may span configured corpora; `validate --inputs` checks its QA IDs
and annotation shapes.

Original-context-language sources use raw answer/evidence without redundant
multilingual records. Other associated language versions require annotations.
An empty record is distinct from a missing record and does not suppress exact
matches from raw fields. This pipeline does not generate or translate annotations.

### Exact relevance

Both raw `answer` and `evidence` are checked independently, together with every
source-specific multilingual span. Any nonempty string qualifies if it occurs
contiguously in an associated source and fits completely inside a chunk. No
trimming, case folding, whitespace/Unicode normalization, markup removal,
translation, or fuzzy matching is applied. Nonempty whitespace-only strings are
also treated literally.

Every occurrence and containing chunk is labeled, including repeated occurrences
and overlapping chunks. Existing evidence offsets/statuses and source-level
`source_qrels` never directly become chunk positives.

Every explicitly involved source is indispensable; language versions of the
same source are alternatives for one requirement. Separate articles can supply
separate chunks. Unresolved involvement, unavailable language versions, absent
exact spans, and boundary-crossing spans are recorded in eligibility diagnostics.

Missing multilingual annotations **block the affected setting**; no partial
qrels are published for it. Other settings remain available. Preparation returns
exit code 2 if any setting is blocked. A supplied annotation without qualifying
support drops that QA from that setting, without changing the corpus.

## Run baselines later

These are explicit benchmark execution commands; preparation and tests do not
invoke them on real datasets:

```bash
uv sync --extra models --extra test
uv run clir-benchmark embed --corpus eurlex --model bge-m3
uv run clir-benchmark retrieve --corpus eurlex --model bge-m3
uv run clir-benchmark evaluate --corpus eurlex --model bge-m3
```

`embed --kind document` works independently of QA preparation; `--kind query`
prepares only query vectors. The default is `both`. Repeat `--model` or omit it
for the configured list. Only ready settings are retrieved/scored, with blocked
settings recorded in manifests.

[`models.yaml`](src/clir_benchmark/models.yaml) pins all eight checkpoints, full
dimensions, input limits, pooling descriptions, and exact prefixes. Published
SentenceTransformer module pipelines are retained, including LaBSE's dense layer
and EmbeddingGemma's projections. SapBERT uses an explicit CLS adapter. BGE-M3
uses dense embeddings only. Prefixes are applied once, with no added source
titles or QA evidence.

Inputs are right-truncated with each model's tokenizer, counting prompts and
special tokens. Benchmark chunks/qrels never change to accommodate an encoder.
Per-input audits record original/encoded token counts and retained character
ranges. Reports show truncation by language and whether qualifying spans and
complete source support survive. Partial Unicode token coverage is conservative.

Default inference uses float32, full dimensions, and L2 normalization. Configure
device and batch sizes in YAML. EmbeddingGemma rejects float16; use float32 or
bfloat16. Custom model code is explicitly enabled for the pinned Nomic entry.

Exact cosine retrieval uses float32 matrix multiplication over disk-backed
document blocks, searching the entire target-language corpus including
distractors. Up to 100 chunks are returned. Ties are resolved by ascending chunk
ID before pruning. Memory scales with scoring block sizes and corpus ID/language
metadata, not a full query-by-corpus score matrix. Manifests record hardware and
library versions; floating-point results can differ across execution environments.

Add a compatible model through a `model_registry` YAML file containing its alias,
`repo_id`, commit `revision`, `dimension`, `max_length`, `pooling`, and optional
prefix/backend fields. Entries merge with the packaged registry. Different
inference APIs can implement the `Encoder` protocol and supply a factory to `embed`.

## Settings, metrics, and artifacts

Corpora are evaluated separately. Empty language lists discover corpus coverage;
explicit lists select experiment languages without deleting stored documents.
Pairwise settings search one target language, including monolingual controls.
Mixed settings search all configured target languages; different indispensable
sources may be supported in different languages.

Report nDCG@10, MRR@10, and Recall/Hit/SourceRecall/AllSourceSuccess at
1, 5, 10, 20, and 100. SourceRecall measures the fraction of indispensable sources
covered by retrieved relevant chunks. AllSourceSuccess is 1 only when all are
covered. Overlapping positives count separately in chunk recall. Source coverage
prevents repeated hits from one article from counting as complete multi-article
success. Unassociated documents are not independently judged semantic negatives.

Reports contain per-query scores, per-setting eligibility/drop counts, generator,
persona, and source-count breakdowns. Macro scores average nonempty settings
equally within each corpus/regime. Monolingual, cross-language, and mixed scores
are never combined. Empty/blocked settings retain counts/status without scores.

| Directory | Contents |
| --- | --- |
| `corpus/<corpus>/<input hash>/` | Sources/chunks Parquet, indexed SQLite lookup, original chemistry pair rows, manifest |
| `relevance/<corpus>/<config hash>/` | Selected queries, selection audit, eligibility reasons, qrels/matches Parquet, standard qrels TSV, settings/manifest |
| `embeddings/<kind>/<cache key>/` | Float32 `vectors.npy`, ordered IDs/truncation audits Parquet, manifest |
| `runs/<corpus>/<run hash>/` | Ranked Parquet/TREC output and manifest |
| `reports/<corpus>/<run hash>/` | JSON report/per-query data, per-query/per-setting CSV |

Stages publish immutable directories only after completion. Compatible completed
stages are reused; changed settings produce new artifacts. Corrupt artifacts
must be explicitly removed before rebuilding. Document cache keys use the
content-based corpus fingerprint; query keys use selected IDs, languages, and
question text. Evidence and qrels are excluded. Encoder settings, model revisions,
prefixes, and dependency versions participate in cache identity.

Relevance schema version 2 preserves native alignment fields in `queries.parquet`
and span provenance in `matches.parquet`. Its cache key is separate from older
relevance artifacts; corpus and embedding cache identities remain unchanged.

Manifests record revision pins, configuration hashes, token accounting, execution
device/precision, versions, and timings. No train/dev/test split or fine-tuning is
introduced; upstream `train` is a storage convention. Answer generation,
reranking, hybrid retrieval, and additional QA quality filtering are excluded.

## Tests

```bash
uv run --extra test pytest -q
uv run --extra test ruff check src tests
uv run --extra test ruff format --check src tests
# Optional: downloads only the pinned public tokenizer, no model weights.
uv run --extra test pytest -q -m integration
```

Offline tests cover exact slicing, selection, literal matching, multilingual and
multi-source eligibility, chemistry provenance/pairs, full-corpus search,
deterministic ranking, metrics, caches, truncation, and the CLI. Real model quality
and performance require later explicit benchmark runs.
