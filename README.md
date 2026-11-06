# CLIR benchmark

A Python package for reproducible, span-grounded dense, BM25, hybrid, translate-then-retrieve, and HyDE baselines
over EUR-Lex, UN documents, and chemistry patent abstracts. It prepares one fixed
chunk corpus and relevance judgments shared by every baseline.

Preparation, generation, embedding, retrieval, and scoring are separate CLI commands. Nothing
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

`HF_TOKEN` is read from the environment or configured `.env` file. The explicit
`translate` command also reads the single variable named by
`translation.api_key_env` (default `TRANSLATION_API_KEY`); `generate-hyde` reads
`hyde.api_key_env` (default `HYDE_API_KEY`). Unrelated API keys are
not loaded into the process environment. `.env`, data, vectors,
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

### Translate then retrieve

The `translate-dense`, `translate-bm25`, and `translate-hybrid` baselines run each
translated query language **separately**, using the same query IDs, eligible
cohorts, full document pools, and qrels as the corresponding untranslated baseline.
For example, an English query translated into French is evaluated both against
each configured pairwise document pool and against the mixed pool. Translation
language and document-pool language are separate experiment dimensions. Rankings
from different translated languages are never fused. Hybrid still fuses its dense
and BM25 branches for one query translation.

Translation languages come from `target_languages` (currently en, de, es, fr, zh),
or all corpus languages when that list is empty. Only queries eligible in at least
one ready setting are translated. Queries already in the requested language are
copied exactly as an `identity` control with zero API calls, tokens, and cost.
Blocked settings remain blocked; translation does not create evidence or change
relevance judgments.

The supplied prompts are stored in
[`chemistry.txt`](src/clir_benchmark/prompts/chemistry.txt) and
[`law.txt`](src/clir_benchmark/prompts/law.txt), with the shared
[`query.txt`](src/clir_benchmark/prompts/query.txt) template. EUR-Lex and UN use the
law prompt. Only the original query and source/target language tags enter the
request: no answers, supporting passages, or document metadata. The prompts'
wording is preserved, with the pasted Markdown escapes removed.

The model and provider are deliberately unset. Before running live translation,
set `translation.model`, the full `translation.endpoint` URL, and your provider's
USD token prices in `configs/benchmark.yaml`. The adapter accepts the
[Chat Completions request and response format](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
For example, the OpenAI endpoint is `https://api.openai.com/v1/chat/completions`;
an alternative provider must implement that protocol. Credentials come from
`TRANSLATION_API_KEY`, or the environment variable you select with `api_key_env`.
Set `api_key_env: null` for an unauthenticated local endpoint. No new dependency
is needed; request-cache locking uses POSIX file locks.

`response_format: json_schema` requests exactly one `translation` string. Use
`json_object` or `text` if the selected endpoint does not support JSON schema;
all modes still enforce the same JSON contract locally. `request_options` allows
model-specific `temperature`, `top_p`, `seed`, `max_completion_tokens`, `max_tokens`,
and `reasoning_effort`. None are forced by default. Choose a stable model snapshot
when available and record suitable generation options for your experiment.

After corpus and relevance preparation, this example compares French translation
against untranslated BM25:

```bash
# Only this command makes translation API calls.
uv run clir-benchmark translate --corpus chemistry --translation-language fr
uv run clir-benchmark index-bm25 --corpus chemistry
uv run clir-benchmark retrieve --corpus chemistry --baseline bm25
uv run clir-benchmark retrieve --corpus chemistry --baseline translate-bm25 --translation-language fr
uv run clir-benchmark compare-translations --corpus chemistry --baseline bm25 --translation-language fr
```

`compare-translations` evaluates the existing runs and writes `report.json`,
`per_query.json`, and `settings.csv` under `artifacts/comparisons`. It reports
absolute metric deltas (translated minus original), mean translation cost, and
timing components on exactly matched query/setting cohorts. It never translates,
loads an encoder, or executes retrieval. To score a translated run alone, use
`evaluate --baseline translate-bm25 --translation-language fr`.

Dense and hybrid reuse existing document vectors and encode translations as
queries with the selected retrieval model's normal query adapter:

```bash
uv run clir-benchmark embed --corpus chemistry --model bge-m3
uv run clir-benchmark embed --corpus chemistry --model bge-m3 --kind query --translation-language fr
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline dense
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline translate-dense --translation-language fr
uv run clir-benchmark compare-translations --corpus chemistry --model bge-m3 --baseline dense --translation-language fr
```

For hybrid, prepare the BM25 index and replace `dense`/`translate-dense` with
`hybrid`/`translate-hybrid`. Repeat `--translation-language` to select multiple
independent variants. Omitting it from `translate`, `compare-translations`, or a
`translate-*` retrieval/evaluation runs all configured target languages. For
`embed`, omission continues to mean original queries; supply each desired
translation language explicitly. Ordinary commands and `baselines: [dense]`
retain their existing behavior. No preparation, embedding, retrieval, or scoring
command implicitly calls the translation API.

Each request is cached by query ID, query text, source/target languages, exact
prompts, endpoint, model, and request options. The cache is shared across retrieval
models and baselines. The request and raw response (including reported model and
usage), elapsed time, and each attempt's status are retained under
`artifacts/translation_requests`; credentials are excluded. Completed translation
sets publish immutable query Parquet files and manifests. Changing prices can
recompute cost estimates using cached responses without calling the API again.

Malformed JSON, extra or duplicate keys, blank translations, refusals, truncation,
and transport failures stop translation without substituting the original query.
Successful requests survive an interrupted batch. Failed or interrupted requests
require an explicit `translate --retry-failed`; retries preserve prior attempts
and include their known usage and time in accounting. Concurrent processes cannot
issue the same cached request simultaneously.

Cost uses reported input/output token counts and configured USD rates per million
tokens, including a cached-input rate when supplied. Without that rate, all input
tokens are charged at the regular configured input rate. Unknown usage or prices
produce `null`, not a free request. Summaries count each query once per translation
language; do not add costs across settings or retrievers that reuse those requests.
These are API cost estimates; local embedding and retrieval compute are not priced.

Latency reports retain the original API-call duration even when translations are
reused. Retrieval timing measures search and fusion, excluding index loading and
result writing, and divides each batch's time among its queries. Query encoding
time is the cached embedding build wall time divided by its input count. The
comparison adds these **offline component measurements**; it does not claim a
measured online end-to-end latency. Historical retrieval runs without timings
remain usable for quality comparison and show `null` latency; use a fresh output
directory to measure those runs. Keep runtime, device, and batch sizes consistent
when comparing timings.

### HyDE: hypothetical document embeddings

Enable **`hyde-dense`** with `--baseline hyde-dense` or include it in the YAML
`baselines` list. Omit it to disable it. The shipped default stays `[dense]`;
disabled HyDE requires no credentials, API calls, or model loading. This baseline
generates one hypothetical passage per eligible query and output language, encodes
the passage with the selected retrieval model's **document adapter**, and searches
the existing real-document vectors. Document prefixes, normalization, and input
limits are exactly those used for corpus documents. Passages remain whole inputs,
subject to the encoder's normal truncation, with their own truncation audit.
This follows the document-to-document retrieval approach in the
[HyDE paper](https://aclanthology.org/2023.acl-long.99.pdf), using one passage per
query/language in this experiment. It does not average the original query into
the passage vector, combine output languages, or run BM25/hybrid HyDE variants.

The supplied system prompts are packaged unchanged as
[`hyde_chemistry.txt`](src/clir_benchmark/prompts/hyde_chemistry.txt) and
[`hyde_law.txt`](src/clir_benchmark/prompts/hyde_law.txt), with the shared
[`hyde_query.txt`](src/clir_benchmark/prompts/hyde_query.txt) template. EUR-Lex and
UN use the law prompt with their respective `collection` value. The request
contains only the collection, original query, source language, and output language;
it never receives answers, supporting passages, retrieved documents, or tools.
Generation is direct in the requested language, including when source and output
languages match. There is no same-language identity shortcut or translation step.

Output languages are independent of document-pool languages. Resolve them from
repeated `--hyde-language` overrides, then `hyde.output_languages`, then
`target_languages`, then discovered corpus languages. Explicit output languages
can be any valid language tag, including a language absent from the corpus.
Each language produces a separate run on the original eligible query IDs and
ready settings, with the same document pools and qrels as the control.

Set the independent `hyde.endpoint`, `hyde.model`, and USD token prices in
`configs/benchmark.yaml` before live generation. Endpoint/model remain unset by
default; they do not inherit translation's choices. The endpoint uses the same
Chat Completions protocol and `response_format`/`request_options` controls as
translation. Credentials come from `HYDE_API_KEY` or the chosen `hyde.api_key_env`;
use `api_key_env: null` for an unauthenticated local endpoint.

After corpus/relevance preparation, generate and retrieve a French variant:

```bash
# Only generate-hyde makes HyDE generation API calls.
uv run clir-benchmark generate-hyde --corpus chemistry --hyde-language fr
uv run clir-benchmark embed --corpus chemistry --model bge-m3 --hyde-language fr
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline hyde-dense --hyde-language fr
uv run clir-benchmark evaluate --corpus chemistry --model bge-m3 --baseline hyde-dense --hyde-language fr
```

`embed --kind query --hyde-language fr` builds only hypothetical-passage vectors;
the default `--kind both` also builds or reuses corpus vectors. Omitting both
language selectors from `embed` continues to encode original queries, even if
HyDE baselines are configured. Repeat `--hyde-language` to embed multiple variants.
Translation and HyDE selectors cannot be combined within one embedding invocation.
For generation, retrieval, evaluation, and comparison, omitted HyDE language
flags use the configuration/default language resolution described above.

To compare with both controls, first prepare their runs explicitly:

```bash
uv run clir-benchmark embed --corpus chemistry --model bge-m3
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline dense
uv run clir-benchmark translate --corpus chemistry --translation-language fr
uv run clir-benchmark embed --corpus chemistry --model bge-m3 --kind query --translation-language fr
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline translate-dense --translation-language fr
uv run clir-benchmark compare-hyde --corpus chemistry --model bge-m3 --hyde-language fr --control dense --control translate-dense
```

`compare-hyde` defaults to the `dense` control. The optional `translate-dense`
control uses the same language and retrieval model as HyDE, and requires that
translation variant to be prepared within the existing translation language set.
Comparison only evaluates existing runs; it never generates, embeds, or retrieves.
Reports under `artifacts/comparisons` include per-query JSON and per-setting CSV
with absolute retrieval-metric deltas, generation-cost deltas, and timing-component
deltas. All differences are **HyDE minus the selected control**. For translation
controls, translation cost and time are subtracted, rather than treating all HyDE
generation overhead as additional. Missing usage, prices, or timings remain `null`.
Costs cover generation APIs only; local embedding and retrieval compute are unpriced.
Generation duration plus amortized passage/query encoding and retrieval timings
are offline component totals, not measured online end-to-end latency.

Hypothetical passages and their vectors have separate artifact namespaces from
translations, original queries, and corpus documents. The shared request runner
preserves translation's existing request keys and artifact identities. HyDE request
keys include collection, query identity/text, source/output languages, prompts,
endpoint, model, and generation options. Cached passages are reusable across
retrieval models; repricing reuses reported usage without generating again.

HyDE strictly requires one nonempty `hypothetical_document` JSON string. Duplicate
or extra keys, refusals, truncation, and invalid output stop the stage without
substituting the original query. Resume failed/interrupted requests explicitly
with `generate-hyde --retry-failed`; all prior attempts remain in accounting.
Sentence count and language quality follow the prompts, without heuristic rewriting.

Each invocation writes an audit under `artifacts/hyde_invocations`, including
zero-new-work cache hits and failed/interrupted attempts. Passage manifests retain
the original generation totals and the work performed when that artifact was built.
Summaries count a request once per output language: do not add reused costs across
retrieval models, settings, or comparison controls.

### BM25 and hybrid retrieval

After `prepare-corpus` and `prepare-relevance`, run BM25 without an embedding
model or model downloads:

```bash
uv run clir-benchmark index-bm25 --corpus eurlex
uv run clir-benchmark retrieve --corpus eurlex --baseline bm25
uv run clir-benchmark evaluate --corpus eurlex --baseline bm25
```

`index-bm25` builds a reusable SQLite inverted index over **every** prepared chunk,
including distractors and documents whose QAs were dropped. Indexing needs only
the corpus artifacts and can precede relevance preparation. Posting aggregation
and sorting can spill to disk; the implementation does not construct an in-memory
query-by-corpus score matrix. No new service or optional model dependency is needed.

BM25 defaults to `k1=1.2`, `b=0.75`, with positive IDF
`log(1 + (N - df + 0.5) / (df + 0.5))` and term contribution
`idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))`.
These parameter defaults and the IDF convention follow
[Lucene's BM25 documentation](https://lucene.apache.org/core/7_4_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).
This implementation uses exact token counts and float64 arithmetic rather than
Lucene's encoded length norms. Statistics are computed over **all chunks in the
setting's target-language pool**, including empty chunks. Mixed settings use
pooled statistics across their target languages. Repeated query terms count once.

The `unicode-words-cjk-v1` analyzer applies NFC, Unicode case folding, then NFC;
groups Unicode letters/numbers and attached combining marks into words; and
uses individual Han and Japanese kana letters as tokens. Other characters act
as separators. It has no stop list, stemmer, tag removal, or query translation.
The analyzer and Unicode database version are recorded. Lexical analysis affects
retrieval only: stored source text, chunk offsets, and exact relevance matching
remain unchanged. This is an untranslated lexical CLIR baseline. Only chunks
with lexical matches are returned, up to 100; zero-overlap queries have empty
rankings and remain in evaluation with zero retrieval metrics.

Hybrid retrieval combines BM25 with **any existing dense checkpoint**, reusing its
cached document/query embeddings:

```bash
# Skip embedding if the selected model's vectors are already cached.
uv run clir-benchmark embed --corpus eurlex --model bge-m3
uv run clir-benchmark retrieve --corpus eurlex --baseline hybrid --model bge-m3
uv run clir-benchmark evaluate --corpus eurlex --baseline hybrid --model bge-m3
```

Prepare the BM25 index first. Hybrid retrieval independently searches the same
language-filtered full corpus with both branches. It takes up to 1,000 candidates
from each, fuses their union with weighted reciprocal rank fusion (RRF), and
returns the top 100. Its score is the sum of
`branch_weight / (rank_constant + one_based_branch_rank)` over branches containing
the chunk; an absent chunk contributes zero. Defaults are equal weights and
`rank_constant=60`, following the
[original RRF formulation](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
Raw cosine and BM25 score scales do not enter fusion. Empty BM25 rankings leave
the dense ranking as the sole contributor. Chunk IDs break score ties in each
branch and the fused ranking. Fusion operates on the truncated candidate lists,
not hypothetical ranks below `candidate_k`.

The YAML configuration records these experiment choices:

```yaml
baselines: [dense, bm25, hybrid]
bm25:
  k1: 1.2
  b: 0.75
  analyzer: unicode-words-cjk-v1
hybrid:
  candidate_k: 1000  # Per branch, at least 100.
  rank_constant: 60
  dense_weight: 1.0
  bm25_weight: 1.0
```

The shipped default remains `baselines: [dense]`, preserving existing commands.
Repeated `--baseline` flags override that list. Dense and hybrid run once per
selected `--model`; BM25 runs once per corpus and takes no model argument when
selected alone. BM25 parameters and fusion weights/depth have separate run cache
identities. Changing them does not change chunks, QAs, qrels, or embedding caches;
changing BM25 scores also reuses the lexical index. Evidence and candidate
selection never enter its index key.

Each baseline produces the same Parquet/TREC runs and JSON/CSV metrics in separate
run/report directories, labeled with baseline and dense model (if applicable).
All eligibility and multi-source metrics apply unchanged. BM25 consumes full
query/chunk text and has no truncation audit; hybrid reports the existing
truncation/support-exposure audit explicitly for its **dense component**.

### Dense encoders

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
| `translation_requests/` | Resumable request/response and attempt audits, shared across retrieval models |
| `translations/<corpus>/<language>/<hash>/` | Translated query Parquet, exact prompts/configuration, usage/cost/latency manifest |
| `hyde_requests/` | Resumable HyDE request/response and attempt audits |
| `hyde_invocations/` | Actual new generation work and cache reuse for each invocation |
| `hyde/<corpus>/<language>/<hash>/` | Hypothetical passages Parquet, prompts/configuration, usage/cost/latency manifest |
| `embeddings/hyde/<cache key>/` | Hypothetical-passage vectors keyed by original query IDs, document-adapter audits |
| `embeddings/<kind>/<cache key>/` | Float32 `vectors.npy`, ordered IDs/truncation audits Parquet, manifest |
| `lexical/<corpus>/<index hash>/` | SQLite postings, exact lengths and language/term statistics, analyzer manifest |
| `runs/<corpus>/<run hash>/` | Ranked Parquet/TREC output, batch-amortized query timings, manifest |
| `reports/<corpus>/<run hash>/` | JSON report/per-query data, per-query/per-setting CSV |
| `comparisons/<corpus>/<hash>/` | Matched translation or HyDE/control metric deltas, costs and timing components in JSON/CSV |

Completed stages publish immutable directories only after completion; individual
translation request audits are updated as attempts finish. Compatible completed
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
reranking, model fine-tuning, and additional QA quality filtering are excluded.

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
BM25 tests compare against independent brute-force scores, including filtered and
mixed language pools; hybrid tests check RRF scores, candidate depth, empty
branches, cache independence, and end-to-end synthetic runs without model loads.
