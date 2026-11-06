# Project coding instructions

These instructions apply throughout this repository.

## Core purpose

This project compares CLIR baselines by retrieval performance, monetary cost,
and latency. The user must be able to switch each baseline or experimental mode
on or off independently and measure what each mode adds relative to a control.
Treat this as a design requirement for every new feature and refactor.

## Independently selectable modes

- Expose mode selection through configuration and CLI options; switching modes
  must not require editing Python code. Support running one mode or a selected
  subset in the same experiment.
- Use the existing `baselines` configuration and repeated `--baseline` options
  for baseline selection. Inclusion enables a baseline; omission disables it.
  Make additional optional stages explicitly selectable as they are introduced.
- Keep existing baselines available as controls when adding a new method. New
  modes are opt-in unless the user explicitly requests a default change.
- A disabled mode must not execute, make API calls, load its models, or incur
  its exclusive preparation costs. Shared dependencies may run when an enabled
  mode requires them; make those dependencies explicit.
- Do not require credentials or model/provider choices for disabled modes.
  Validate required dependencies when the selected mode needs them.
- Keep results and artifact identities distinct for each mode and its parameters,
  including retrieval model, translation model, and translation language where
  applicable. Reuse compatible shared artifacts without mixing results.
- For translate-then-retrieve, evaluate each translation language separately.
  Any future fusion across query languages must be an explicitly selected mode.

## Comparable performance, cost, and latency

- Compare modes on the same query cohort, corpus/chunks, relevance judgments,
  and retrieval settings. Record intentional experimental differences explicitly.
- Report each mode's retrieval metrics and its change relative to the appropriate
  control, such as translation versus the same untranslated retriever.
- Record cost and latency for added stages as well as the comparable pipeline
  totals where available. Include token usage and configured pricing for paid
  model calls, and retain failed/retried attempts in accounting.
- Distinguish measured timings from estimates, amortized batch timings, and
  measured end-to-end latency. Missing measurements or prices must remain unknown,
  not silently become zero.
- Distinguish work performed in the current invocation from cached work. Retain
  original generation costs and timings for comparison, and avoid double-counting
  artifacts shared across settings, languages, or retrievers.
- Save the resolved mode selection, parameters, model identities, and measurement
  methodology with results so comparisons can be reproduced and audited.

## Implementation and validation

- Preserve separate preparation, translation, embedding, retrieval, and evaluation
  stages. Live benchmark execution and paid calls require explicit run commands;
  validation and scoring must not implicitly execute those stages.
- Document how to enable/disable each new mode and compare it with its control.
- For changes to mode selection or execution, use offline fixtures to verify that
  enabled modes run, disabled modes do not, shared caches remain reusable, and
  performance/cost/latency reports compare matching cohorts correctly.
