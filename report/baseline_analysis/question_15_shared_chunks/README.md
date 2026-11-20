# Do 200 words fit the embedding models?

On Narval, **all 627 windows of exactly 200 words fit every model's native input limit**. Applying a common ceiling of 512 tokens to each of the eight models admitted **588 of 627 windows (93.8%)**. Qwen exceeded 512 on all 39 failures. EmbeddingGemma exceeded it on four of the same windows. The other six tokenizers stayed below 512 throughout the sample. [Per-model measurements](results/model_fit.csv)

| Model | Native input limit | Largest 200-word input | Windows above 512 |
|---|---:|---:|---:|
| Nomic Embed Text v2 MoE | 512 | 471 | 0 |
| Granite Embedding 278M Multilingual | 512 | 467 | 0 |
| Multilingual E5 Large Instruct | 512 | 467 | 0 |
| EmbeddingGemma 300M | 2,048 | 572 | 4 |
| BGE-M3 | 8,192 | 468 | 0 |
| GTE Multilingual Base | 8,192 | 467 | 0 |
| Snowflake Arctic Embed L v2.0 | 8,192 | 468 | 0 |
| Qwen3-Embedding-0.6B | 32,768 | 625 | 39 |

| Language | Windows of 200 words | Fit all eight at 512 | Highest token count |
|---|---:|---:|---:|
| Arabic | 73 | 70 | 565 |
| Chinese | 85 | 85 | 405 |
| English | 119 | 119 | 491 |
| French | 125 | 124 | 539 |
| German | 18 | 12 | 625 |
| Japanese | 4 | 4 | 368 |
| Russian | 77 | 49 | 567 |
| Spanish | 126 | 125 | 533 |

The sample contains ten records per verified November 18 release. Google Patents Chemistry contributes 21 language editions, EUR-Lex contributes 40, and UN contributes 60. Sampling was stratified by evidence language and included one EUR-Lex annex and one Japanese patent with a long supporting span. The fit rates describe these 30 records. [Selected documents and release hashes](results/selection.json) · [Figure](results/language_fit.png)

ICU 73.2 counted words using each edition's language, including dictionary-based segmentation for Chinese and Japanese. Each edition was divided into non-overlapping windows of exactly 200 words. Short final remainders were excluded from the 200-word test. We counted each complete document input with its pinned fast tokenizer, document prefix and special tokens, with truncation disabled. The figures come from Narval CPU job `9912721`. [All 627 windows](results/word_windows.csv) · [Tokenizer identities and formatting](results/tokenizer_policy.json) · [ICU boundary analysis](https://unicode-org.github.io/icu/userguide/boundaryanalysis/)

## 200 words with 50 words of overlap

Narval CPU job `9915888` checked the same 30 records with the overlap included in the 200-word limit. It counted the complete formatted inputs before applying any token-based shortening, including the final short chunk in each edition.

| Chunking method | Chunks tested | Fit all eight native limits | Fit a common 512-token ceiling |
|---|---:|---:|---:|
| Fixed windows, exactly 50 words repeated | 940 | 940 | 892 |
| Whole sentences where possible, up to 50 words repeated | 991 | 991 | 967 |

Among the three encoders with a native limit of 512, the largest sentence-based input was 468 tokens. [Measurements](results/word-fit-overlap-50.json) · [Token-counting script](overlap_fit.py)

## Reproduce the token counts

The sample extractor verifies the source corpus and evidence shard hashes. The CPU worker loads the eight tokenizers and counts words and tokens. The reporting command reads saved measurements.

```bash
q=report/baseline_analysis/question_15_shared_chunks
python "$q/sample.py" /path/to/hf_export /path/to/fresh-pilot/inputs
# On an allocated Narval CPU node, with the project, sample and PyICU staged.
bash "$q/narval.sh" /path/to/fresh-pilot
# Regenerate the figure and tables from the downloaded measurements.
python "$q/summarize.py" "$q/results"
```
