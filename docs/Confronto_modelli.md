# Controlled local model comparison

This development-only experiment compares Qwen3.8-27B UD-IQ4_XS, Ministral 3 14B Reasoning2512 Q4_K_M, and Qwen3.5-9B Q4_K_M. Public synthetic data only; product inference behavior remains unchanged. Windows packages remain the previously verified0.3.0-dev.4 build.

## Frozen task and data

The70 baseline cases are byte-for-byte equivalent to the evaluated Minerva reference corpus (`fcf7f964700e4cf1fcb274e7fc2dac5038ee723936afed8da4b25cd70b84dc6d`). Another14 cases add two per part; their file SHA256 is `f4208bea9ebc00d6b262bbf06c86fa4806732813c0f065435d8830511ab6a639`. All labels are assistant-authored engineering fixtures, not independent teacher validation. Expected labels and rationales never enter model requests. Every candidate receives the shared evidence task, with its pinned original template and official tokenizer oracle.

## Two separately reported profiles

| Setting | Original | budget128 |
|---|---:|---:|
| Temperature / seed | 0 / 0 | 0 / 0 |
| Context |4096|4096|
| Maximum generated tokens |1024|1024|
| Reasoning budget |512|128|
| HTTP request limit, seconds |180|300|
| Inclusive part limit, seconds |840|780|
| Per-part ordering |10baseline then2fresh|2fresh then10baseline|

The second profile was fixed after inspecting the first six completed Qwen3.5 baseline cases. No fresh-case model result had been inspected when the new settings were chosen; the concurrent original Ministral run later completed part1, including its two fresh cases. It is an explicit resource-constrained exploratory configuration, not an independent accuracy confirmation. Its metrics must never be pooled with the original profile. Sampling, task text, labels, model assets and official tokenizer gates are unchanged. Qwen thinking stays enabled, effort medium; native Ministral reasoning is observed rather than assumed.

The original run (`39560793744`, commit `8b1b1c00f9fa5c36e024dd6ca4c40428bbccc403`) exposed a harness defect: a socket timeout shortened by the part deadline was mislabeled as a fatal request timeout, skipping later parts. The correction preserves the distinction. A genuine full request timeout still stops further work. Original negative and partial evidence is retained.

## Resource budget and interpretation

Public standard Linux CPU runners only, one slot and at most4threads. The original Qwen attempts took approximately6and15minutes; the original Ministral run completed all84cases. The second Qwen jobs are bounded to105minutes each; Ministral receives only the14freshcases in a35minute job, avoiding another84case run. No paid runner or native Windows rebuild is part of this experiment. Qwen3.8 is evaluated with IQ4_XS because its Q4_K_M weights are impractical for16GB RAM; quantization differences are a comparison limitation.

Missing, invalid, truncated and technical outcomes stay explicit in all planned denominators. A response ending through length truncation is invalid even if its content can be decoded. Runtime token IDs must exactly match the official oracle before inference; native input counts are checked again afterward. Raw requests, responses, lifecycle, memory high-water mark, pinned assets and bounded sanitized logs are retained.

No result certifies participation, authorship, copying direction or cheating. No candidate is promoted to the application without further semantic and platform validation.

## Fresh-only Ministral continuation

The `heldout128` profile uses the same generation settings as `budget128` but selects only the14freshcases (two per part) and caps each part to180seconds. Its planned denominator is14, with zero baseline cases. This preserves a common14case comparison across all three candidates while allowing the original Ministral84case run to continue. Different time caps can cause unequal coverage, so comparisons require completed paired case IDs and must expose every missing response.


## Observed final outcomes (2026-10-21)

All three requested candidates were actually executed with the pinned native runtime and per-input official tokenizer parity checks. These are engineering-fixture label agreements, not independently validated classroom accuracy or cheating sensitivity.

| Model / profile | Original70: matching / valid | Fresh14: matching / valid | Total matching / planned | Recognized expected errors / planned errors | False alarms on valid correct cases |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B, budget128 |56/70|14/14|70/84|30/35|0/24|
| Ministral3 14B, original512 |46/70|13/14|59/84|33/35|5/24|
| Ministral3 14B, heldout128 retry |not planned|13/14|13/14|6/7|0/3|
| Qwen3.8-27B, budget128, partial |0/0|3/3|3/84|2/35|0/1|
| Minerva corrected reference |13/70|not planned|13/70|0/28|0/21|

Qwen3.5 had the highest overall agreement in these completed exploratory runs. Ministral recognized more expected errors in its full run, but produced five false alarms on24correct cases. Different512and128token reasoning profiles prevent treating these full runs as an identical-settings model ranking. On the14freshcases with matched128token generation settings, Qwen3.5 agreed on14/14and Ministral on13/14; their part time caps and case ordering still differed. Qwen3.8 completed only three fresh cases: all three agreed, but79cases were missing and two had infrastructure/deadline outcomes. It cannot be ranked from3/84coverage. Its genuine300second request timeout, memory demand and original negative attempt remain preserved.

The first Qwen3.5 attempt retained four matching labels in six valid responses, one deadline error, and77missing cases; the fixed budget128run subsequently completed all84without technical or format failures. Original Qwen3.8 had one timeout and no completed prediction. Ministral heldout preparation initially exceeded its seven-minute download step before inference; a dedicated ten-minute preparation retry completed all14cases. This failure is retained as preparation evidence and is not pooled with successful predictions.

Ministral's sole fresh error in both profiles was F1.01: it incorrectly accepted the claim that means of ordinal-category codes preserve group ordering under every strictly increasing recoding. No labels, questions or model weights were changed after inspecting this result. Native reasoning content was not exposed for Ministral in these runs; its final reasons were saved.

Actual workflow evidence:

- Original512run: https://github.com/valerioghezzi/UAllera/actions/runs/39560793744
- budget128and initial heldout setup: https://github.com/valerioghezzi/UAllera/actions/runs/39563761472
- Ministral heldout-only retry: https://github.com/valerioghezzi/UAllera/actions/runs/39565051093
- Corrected Minerva reference: https://github.com/valerioghezzi/UAllera/actions/runs/39490573473

`tools/render_model_comparison.py` recomputes metrics from preserved plans and case rows, checks frozen inputs, retains missing denominators, and renders an offline Italian HTML report and JSON sidecar. Paired fresh-case summaries expose completed intersections and unequal time caps. The final source suite passed395tests with2skipped. New candidate inference was tested only on Linux CPU; the previously verified Windows x64/ARM64 packages keep Minerva experimental. No candidate was automatically integrated into the application.
