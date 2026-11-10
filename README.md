# CLIR benchmark

A Python package for reproducible, span-grounded dense, BM25, hybrid, translate-then-retrieve, HyDE, ADORE, and Qwen3-reranked baselines
over EUR-Lex, UN documents, and chemistry patent abstracts. It prepares one fixed
chunk corpus and relevance judgments shared by every baseline.

Preparation, generation, embedding, retrieval, reranking, and scoring are separate CLI commands. Nothing
runs a benchmark automatically. ADORE's explicit `run-adore` command coordinates
generation, query embedding, and retrieval stages within its feedback loop.
Tests use synthetic corpora and lightweight
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
`hyde.api_key_env` (default `HYDE_API_KEY`); `run-adore` reads
`adore.api_key_env` (default `ADORE_API_KEY`). Unrelated API keys are
not loaded into the process environment. `.env`, data, vectors,
and reports are ignored by Git. The token must have access to BASF and any gated
model being used.

## Prepare sources and chunks

### Small laptop smoke test

[`configs/laptop-smoke.yaml`](configs/laptop-smoke.yaml) selects all 20 baselines
with the pinned `minilm-l6` encoder
([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
22,713,216 parameters, 384 dimensions). It uses CPU inference, small batches,
English/French retrieval settings, GPT-6 Luna for translation/HyDE/ADORE, and the
Qwen3 reranker with three candidates and a 512-token pair budget. ADORE retains
the original query language and allows at most three expansion rounds, with one
passage and two feedback documents per round. The ordinary benchmark defaults
remain unchanged. MiniLM is English-trained; this intentionally small experiment
checks execution and artifacts, not representative multilingual effectiveness.

On a Linux CPU laptop, install the model dependencies without CUDA wheels:

```bash
uv sync --extra test
uv pip install --python .venv/bin/python --torch-backend cpu -e '.[models,test]'

# No generation or embedding inference: select two eligible questions and
# at most 12 source units, preserving required documents and adding distractors.
.venv/bin/clir-benchmark --config configs/laptop-smoke.yaml prepare-subset \
  --corpus eurlex --max-questions 2 --max-sources 12 --seed 0 \
  --destination artifacts/laptop-smoke/subset

# Explicit live execution, including paid calls using OPENAI_API_KEY from .env.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/clir-benchmark \
  --config artifacts/laptop-smoke/subset/config.yaml smoke --corpus eurlex
```

Subset preparation supports EUR-Lex and UN. It freezes exact upstream QA/corpus
rows, complete candidate groups, source associations, selection/rejection reasons,
revision pins, and file hashes. It retains questions eligible in every selected
setting. Initial preparation may download the original corpus shards to find
those sources; only the bounded subset is embedded or searched. The smoke runner
requires local inputs and checks question/source/chunk limits before inference.
The generated configuration writes results under the subset's `artifacts/`.

All 20 baseline selections yield 28 runs because translation and HyDE are tested
separately in English and French. Use repeated `--baseline` flags to select fewer,
for example `smoke --corpus eurlex --baseline dense --baseline bm25`. Only selected
modes and required rerank parents execute. Shared generation, embeddings, and
indexes are reused; comparisons execute when their matching controls are present.
A failed stage blocks its dependents while independent controls continue.
The printed `smoke_invocations/<id>/status.json` records every outcome, dependency,
resolved configuration, current invocation timing, errors, and result locations.
The command exits nonzero unless all requested work succeeds. Rerun the same
command to reuse completed work; use `--retry-failed` explicitly to retry failed
generation requests while retaining their previous attempts and costs.

GPT-6 Luna's configured Standard rates are $0.10 input, $0.01 cached input, and
$0.50 output per million tokens, verified against the
[official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
Recorded provider responses retain the actual returned model ID and token usage.
These are API cost estimates; local compute prices remain unknown.

### Fir cluster runs

`clir_benchmark.cluster` plans and executes the original `dense`, `bm25`, and
`hybrid` baselines under Slurm. `write_plan()` accepts a resolved configuration,
remote source/configuration/environment paths, CPU/GPU accounts, and resource
requests. With eight encoders and three corpora, `split_embedding=True` creates
54 independent jobs: three shared preparation/index jobs, three BM25 jobs,
24 GPU embedding jobs, and 24 CPU dense/hybrid retrieval and evaluation jobs.
Exact dense search uses NumPy on CPU, so splitting releases GPUs before scoring.
CPU scoring uses the same encoder configuration as its embedding dependency,
preserving vector cache identity without loading an encoder.

Every job defaults to **one CPU**, including GPU embedding and dense scoring.
Preparation and BM25 use serial loops; embedding sends sequential batches to
the GPU. Dense scoring can use threaded matrix multiplication, but that alone
does not justify extra CPUs for its complete retrieval/evaluation job. Increase
a stage's request only when profiling shows a useful improvement across that
stage. `write_plan()` accepts independent `JobResources` through
`prepare_resources`, `bm25_resources`, `model_resources`, and `score_resources`;
for example, `score_resources=JobResources(cpus=2)` affects dense/hybrid scoring
without increasing BM25's allocation. These scheduler settings do not change
the selected modes, query cohort, model settings, or embedding cache identities.
The generated scripts set OpenMP, MKL, and OpenBLAS thread limits from the actual
`SLURM_CPUS_PER_TASK` allocation and disable tokenizer parallelism. Job journals
retain those thread settings together with Slurm allocation metadata. Existing
submitted plans remain the original request record; record any later scheduler
changes separately instead of regenerating the submitted plan.

GPU requests can vary by encoder: pass
`model_resources_by_alias={"LaBSE": JobResources(gpus="nvidia_h100_80gb_hbm3_1g.10gb:1")}`
to `write_plan()` to request that MIG profile for LaBSE jobs. Each mapping entry
replaces the complete `model_resources` request for that configured alias;
other aliases retain the shared request, which defaults to `h100:1`.
Unknown aliases are rejected, and resource overrides never enable disabled
baselines or add encoders. Choose a MIG profile after measuring model memory
and throughput with the intended batch size and inputs; changing the GPU
allocation does not change batch size, precision, or cache identities.

Use `candidate_mode: all` for every generated question candidate; all selected
questions are embedded. Retrieval metrics still require eligible query/setting
pairs. Keep `eligibility.parquet`, settings, and generator breakdowns to audit
questions without usable exact evidence. Preparation stops on blocked annotation
settings unless explicitly invoked with `--allow-blocked-settings`.

Stage pinned data, model weights, and Nomic's pinned external implementation in
the shared Hugging Face cache before execution. Preparation needs HF metadata
access; embedding and scoring jobs run with the staged cache offline. Prefer HF
source specifications over `local_dir` for full snapshots, avoiding repeated
large local-file hashing. The environment script should load the same Python,
Arrow, and CUDA modules in every job and set `SQLITE_TMPDIR=$SLURM_TMPDIR`.
Only the HF credential is needed; generation and reranking remain disabled.

After copying the plan, scripts, source tree, and configurations to Fir, submit
the whole dependency graph from Fir:

```bash
python -m clir_benchmark.cluster submit --plan /remote/run/plan.json --workers 16
python -m clir_benchmark.cluster status --submission /remote/run/submission.json
```

`--workers` controls concurrent submission calls, not running jobs. There is no
additional array/concurrency cap; Slurm controls execution according to available
resources and account policy. Submission records every job ID and dependency,
and rerunning a completed submission reuses those IDs. Incomplete submissions
require inspecting the scheduler before retrying, preventing duplicate jobs.
After fixing a failed job, explicitly select it and any cancelled dependents with
`python -m clir_benchmark.cluster retry --submission /remote/run/submission.json
--label <job-label> --label <dependent-label> --cpus 1 --reason "repair description"`.
Optional `--time 03:00:00` overrides the new jobs' walltime. `--cpus 1` explicitly
overrides CPUs per task, including when the original script requested more.
Optional `--gpus nvidia_h100_80gb_hbm3_1g.10gb:1` overrides the GPU request only
for selected jobs that originally requested GPUs; CPU dependents retain their
CPU-only allocation. Selecting only CPU jobs with `--gpus` is rejected.
`--memory 16G` overrides host RAM for all selected retries. For example, after
profiling confirms a model fits the chosen MIG profile and host memory:

```bash
python -m clir_benchmark.cluster retry --submission /remote/run/submission.json \
  --label clir-embed-un-LaBSE --cpus 1 --memory 16G \
  --gpus nvidia_h100_80gb_hbm3_1g.10gb:1 --reason "Resource profile verified"
```

Omitting these options retains the original script's requests. The overrides
are journaled separately without rewriting original scripts or allocations.
Retries verify that
old jobs are terminal, preserve their IDs and scheduler accounting, and connect
children to the new parent IDs. Already completed dependencies are verified
without referencing job IDs Slurm may have purged. Running or ambiguous jobs
are never resubmitted automatically.
Each job saves stage outcomes, errors, configuration, library versions, node and
allocation metadata, and measured wall time. Scheduler CPU/GPU allocation time
remains separate from model/retrieval timings and unpriced local compute.

Run the local collector in a persistent local process to copy the campaign back:

```bash
.venv/bin/python -m clir_benchmark.cluster_monitor \
  --host fir --remote-directory /remote/run \
  --local-directory artifacts/my-fir-run \
  --remote-environment /remote/env.sh --remote-python /remote/venv/bin/python \
  --interval 30
```

The collector polls status while one independent `rsync` transfers completed
artifacts, logs, source/configuration snapshots, rankings, MTEB metrics, and full
embedding vectors. It excludes unfinished temporary artifacts and credentials,
preserves existing local files, and retries transient connection failures. Keep
the local machine awake and connected. Its PID, heartbeat, transfer history,
and latest scheduler status are saved in `.local-monitor/`. When all jobs finish,
it performs a final sync and verifies report counts and all indexed checksums,
writing `verification.json`. Remote paths remain unchanged as provenance;
relative analysis inventories work in the local copy.

### Standard preparation commands

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

### ADORE: iterative retrieval with relevance feedback

Enable **`adore-bm25`**, **`adore-dense`**, or **`adore-hybrid`** through the
`baselines` configuration or repeated `--baseline` flags. Omit them to disable
ADORE; the shipped default remains `[dense]`. Disabled ADORE requires no endpoint,
credentials, generation calls, or additional model loads. Each mode uses the
same corpus, eligible queries, relevance judgments, and retrieval settings as
its corresponding untranslated control.

This is a CLIR adaptation of
[ADORE](https://arxiv.org/abs/2606.13905). The loop generates five pseudo-passages
by default, retrieves with the original query plus those passages, grades newly
seen top-10 chunks from 0 to 3 against the **original question**, and feeds the
accumulated graded chunks into the next expansion. Only actual retrieved text is
used as feedback; gold answers, evidence annotations, and relevance labels are
never sent to the LLM. The LLM's grades guide retrieval and do not replace qrels.
Expansion and assessment use the same explicitly configured model.

**Every expansion stays in the original `question_language`**, including when
the retrieved chunks are in other languages. Prompts require feedback vocabulary
to be expressed in the query language. Structured outputs must declare that exact
language tag; mismatches and invalid shapes fail the request. This validates the
declared language and prompt contract, not the natural language of every token:
there is no automatic language-identification model. Stored passages and prompts
remain available for auditing. ADORE has no output-language override or translation
stage, and never fuses query languages.

**At most three expansion rounds are allowed, counting the first expansion.**
`adore.max_rounds` defaults to `3` and accepts only `1`, `2`, or `3`.
`--adore-rounds` overrides it for `run-adore`, `retrieve`, `evaluate`, and
`compare-adore`; use the same value when accessing a saved variant. Smaller caps
provide one- and two-round comparisons. Early stopping can shorten the loop when
the full feedback window is graded maximally relevant or the top document IDs
are unchanged on two successive comparisons with the previous round. Each run
records the actual rounds and stopping reason.

The returned top 100 come from the **last expanded query**. Earlier rankings are
saved in traces but are not fused or reranked by the LLM. Every round replaces the
previous expansion while preserving the original query. Unlike the paper's
length-weighted query repetition, this adaptation includes the original query
once; the existing BM25 analyzer counts repeated query terms only once. Dense
and hybrid modes apply the retriever's normal **query adapter**, normalization,
and truncation to the expanded query. They reuse the control's document vectors.

Configure `adore.endpoint` (the complete Chat Completions URL), `adore.model`, and
USD token prices before live execution. They remain unset by default and are
independent of translation and HyDE. `adore.passages_per_round` and
`adore.feedback_k` configure expansion count and assessment depth; these choices
participate in artifact identity.

After corpus and relevance preparation, compare ADORE with BM25:

```bash
uv run clir-benchmark index-bm25 --corpus chemistry
uv run clir-benchmark retrieve --corpus chemistry --baseline bm25
# This explicit command makes expansion/assessment API calls and runs retrieval.
uv run clir-benchmark run-adore --corpus chemistry --baseline adore-bm25
uv run clir-benchmark evaluate --corpus chemistry --baseline adore-bm25
uv run clir-benchmark compare-adore --corpus chemistry --baseline adore-bm25
```

For dense retrieval, prepare document embeddings explicitly first. Original-query
embeddings are needed for the control; the ADORE command creates or reuses its own
expanded-query embeddings during the loop:

```bash
uv run clir-benchmark embed --corpus chemistry --model bge-m3
uv run clir-benchmark retrieve --corpus chemistry --baseline dense --model bge-m3
uv run clir-benchmark run-adore --corpus chemistry --baseline adore-dense --model bge-m3
uv run clir-benchmark compare-adore --corpus chemistry --baseline adore-dense --model bge-m3
```

For hybrid, prepare the BM25 index as well and select `hybrid` / `adore-hybrid`.
Repeated baseline/model options select independent experiments. `run-adore` only
executes selected ADORE modes; it never prepares missing corpus/index/document
artifacts. `retrieve --baseline adore-*` only checks and returns an existing run.
`evaluate` and `compare-adore` score saved results without API calls or model loads.
The comparison reports per-query and per-setting metrics and deltas against the
same untranslated retriever, rejecting mismatched cohorts or artifact identities.

Request caches reuse compatible expansions and judgments, including shared first
expansions across settings and retrievers. Later feedback depends on the selected
retriever and document pool. Run identities distinguish all parameters and model
identities. Completed runs preserve original costs/timings; invocation audits
record new work and cache reuse. Invalid, failed, and interrupted requests require
an explicit `run-adore --retry-failed`, retaining previous attempts in accounting.

Costs include expansion and assessment calls, token usage, and configured prices;
missing usage/prices remain `null`. Summaries deduplicate shared request keys;
per-setting costs must not be summed as a bill for reused work. Component timing
totals include all rounds and original cached work. Control timings are
batch-amortized, so component-time deltas are estimates, not measured end-to-end
latency. Local embedding/retrieval compute remains unpriced.

### Qwen3-Reranker-0.6B

Enable a reranked variant by prefixing any existing baseline with `rerank-` in
`baselines` or a repeated `--baseline` option: for example, `rerank-dense`,
`rerank-bm25`, `rerank-hybrid`, `rerank-translate-dense`, `rerank-hyde-dense`, or
`rerank-adore-dense`. All translated and ADORE backends are supported. Omission
disables reranking; the shipped default remains `[dense]`. Original runs remain
available as controls, and neither validation nor scoring loads the reranker.

The model is
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
pinned to revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
The adapter follows the model card's Transformers scoring recipe: format an
instruction/query/document input with its fixed chat prefix and assistant suffix,
then normalize the final-position **yes/no logits** to obtain a relevance score.
It performs no answer generation. The default instruction is the official English
retrieval instruction; the query and document retain their original text.

Reranking is a separate stage over **saved parent-run candidates**, not a new
full-corpus search. By default, it reranks the parent's first 100 chunks and returns
those candidates in descending reranker-score order, with deterministic tie
handling. `reranker.candidate_k` or `--rerank-candidates N` selects a smaller prefix
from 1 through 100. It never inserts unseen chunks, expands the candidate set, or
fuses languages or retrieval runs. An empty parent ranking stays empty.

All variants score each document against the **original question**, including
translated-query, HyDE, and ADORE parents. Translation language, HyDE language,
and ADORE round settings still identify the separate candidate-generation runs.
The reranker never receives gold answers, evidence spans, or relevance labels.
This lets us measure whether the same cross-language candidates are ordered better
for the original information need.

Install local model dependencies with `uv sync --extra models`. The `reranker`
configuration has independent `device`, `dtype`, `batch_size`, `max_length`, and
`instruction` settings; `--model` continues to select the **parent dense retriever**.
Defaults are device `auto`, float32, eight pairs per batch, and 8,192 total tokens
per pair. The token limit includes the fixed prefix/suffix. Body text is truncated
on the right while preserving the scoring suffix; padding is on the left.
Token counts, truncation, and retained query/document character ranges are audited.
The model's maximum supported limit is 32,768; corpus chunking and qrels do not
change when adjusting this limit.

After preparing the corpus/relevance and the original dense run:

```bash
uv run clir-benchmark embed --corpus chemistry --model bge-m3
uv run clir-benchmark retrieve --corpus chemistry --model bge-m3 --baseline dense
# Only rerank loads and executes Qwen3-Reranker-0.6B.
uv run clir-benchmark rerank --corpus chemistry --model bge-m3 --baseline rerank-dense
uv run clir-benchmark evaluate --corpus chemistry --model bge-m3 --baseline rerank-dense
uv run clir-benchmark compare-rerank --corpus chemistry --model bge-m3 --baseline rerank-dense
```

For BM25, prepare `index-bm25` and the `bm25` run, then select `rerank-bm25` with
no `--model` option. For hybrid, prepare its index, embeddings, and `hybrid` run
before selecting `rerank-hybrid`. For ADORE, run `run-adore` first, then select
the corresponding `rerank-adore-*` variant; use the same `--adore-rounds` value.
For translated and HyDE runs, repeat `--translation-language` or `--hyde-language`
to select independent parent variants. Reranking never runs these parent stages
or makes generation API calls implicitly.

Only selected `rerank-*` modes run under `rerank`. The `retrieve` command with a
reranked baseline only locates a completed artifact; it directs you to `rerank`
when one is missing. `evaluate` and `compare-rerank` read saved outputs. Use the
same candidate limit in all three commands when overriding the configuration.
For example, `--rerank-candidates 20` selects a distinct 20-candidate experiment.

`compare-rerank` reports deltas against the **same candidate prefix in the parent's
original order**. It also retains full-parent metrics so candidate truncation can
be distinguished from reranking. All comparisons use identical query/setting
cohorts and qrels, including queries with no retrieved candidates. The reranker
can improve ordering but cannot recover relevant documents missing from the
selected candidate set.

Run identities include the parent artifacts, candidate count, reranker checkpoint,
instruction, token limit, runtime, and scoring method. Compatible pair scores are
cached across settings and parents, while each result remains separate. Original
timings survive cache reuse and invocation audits distinguish newly performed work.
Reranking latency is measured and batch-amortized per pair; model initialization
is recorded separately. Pipeline component totals retain the parent's generation,
encoding, and retrieval overheads and add reranking time. These component sums
are not measured end-to-end latency. Local compute prices and total monetary
costs remain unknown (`null`); existing parent API-cost accounting is preserved.

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

[`models.yaml`](src/clir_benchmark/models.yaml) pins all nine checkpoints, full
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
| `adore_requests/` | Resumable expansion/assessment requests and all attempt audits |
| `adore_invocations/` | ADORE invocation work, cache reuse, and failure accounting |
| `adore_searches/<hash>/` | Cached rankings and original search timings per expansion and document pool |
| `embeddings/adore/<hash>/` | Expanded-query vectors and query-adapter truncation audits |
| `reranking_pairs/<hash>/` | Reusable Qwen3 pair scores, original batch timings, and token/character audits |
| `reranking_invocations/` | New reranking work, model initialization time, and score reuse per invocation |
| `embeddings/<kind>/<cache key>/` | Float32 `vectors.npy`, ordered IDs/truncation audits Parquet, manifest |
| `lexical/<corpus>/<index hash>/` | SQLite postings, exact lengths and language/term statistics, analyzer manifest |
| `runs/<corpus>/<run hash>/` | Ranked Parquet/TREC output, timings, manifest; reranked runs also preserve candidate ranks and pair audits |
| `reports/<corpus>/<run hash>/` | JSON report/per-query data, per-query/per-setting CSV, MTEB-style JSON and analysis inventory |
| `reports/<corpus>/<run hash>/analysis/` | Joined ranked documents with exact text, query, language, source offsets, rank, score, and relevance |
| `input_snapshots/` | Shared immutable copies of local raw inputs and subset provenance |
| `mteb/<model>/<revision>/experiments/<baseline and run hash>/` | MTEB-style custom-task metrics, isolated by all run parameters |
| `smoke_invocations/` | Execution journal with every selected baseline, stage outcome, cache reuse, runtime environment, and result paths |
| `comparisons/<corpus>/<hash>/` | Matched translation, HyDE, ADORE, or reranking/control metric deltas, costs and timing components in JSON/CSV |

Every `evaluate` call exports `mteb.json` beside its detailed report and into the
model/revision hierarchy above. This follows the
[MTEB TaskResult layout](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/results/task_result.py):
`task_name`, effective `dataset_revision`, and `scores.test` containing one record
per retrieval setting with `hf_subset`, `languages`, `main_score`, `ndcg_at_10`,
`mrr_at_10`, `recall_at_*`, and the benchmark's source-coverage metrics. Scores
remain on the 0–1 scale and `main_score` is nDCG@10. These are custom CLIR tasks;
MTEB itself is not run, so `mteb_version`, total evaluation time, and emissions
remain `null`. No incompatible scores or language variants overwrite each other.

**Retain the entire output directory for analysis.** Each report's
`analysis_manifest.json` inventories its dependencies with relative paths and
SHA-256 checksums: source/query/evidence snapshots, qrels and eligibility,
embeddings and truncation audits, ranked documents, exact generation prompts and
raw responses, all failed/retried attempts, reranker candidate scores, and original
cost/timing records. `evaluation_configs/` preserves resolved configurations for
each scoring/export invocation. Shared artifacts are stored once; references do
not imply additional cost. Cached reports receive missing analysis exports on an
offline `evaluate` call, without rerunning retrieval, inference, or generation.

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
introduced; upstream `train` is a storage convention. User-facing answer generation,
model fine-tuning, and additional QA quality filtering are excluded. Reranking is
available only through the explicitly selected Qwen3 stage described above.

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
ADORE tests cover same-language expansion contracts, the three-round ceiling,
adaptive stopping, final-round rankings, shared artifact reuse, matched control
comparisons, and accounting for cached, failed, and retried calls. They use fake
generation clients and encoders; they do not measure live retrieval quality.
Reranker tests use synthetic logits, tokenizers, and candidate runs to verify the
yes/no scoring contract, suffix-preserving truncation, candidate membership,
independent modes, cache reuse, timing accounting, and matched control metrics.
