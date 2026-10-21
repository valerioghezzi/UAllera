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

Public standard Linux CPU runners only, one slot and at most4threads. The original Qwen attempts took approximately6and15minutes; the original Ministral run continues. The second Qwen jobs are bounded to105minutes each; Ministral receives only the14freshcases in a35minute job, avoiding another84case run. No paid runner or native Windows rebuild is part of this experiment. Qwen3.8 is evaluated with IQ4_XS because its Q4_K_M weights are impractical for16GB RAM; quantization differences are a comparison limitation.

Missing, invalid, truncated and technical outcomes stay explicit in all planned denominators. A response ending through length truncation is invalid even if its content can be decoded. Runtime token IDs must exactly match the official oracle before inference; native input counts are checked again afterward. Raw requests, responses, lifecycle, memory high-water mark, pinned assets and bounded sanitized logs are retained.

No result certifies participation, authorship, copying direction or cheating. No candidate is promoted to the application without further semantic and platform validation.

## Fresh-only Ministral continuation

The `heldout128` profile uses the same generation settings as `budget128` but selects only the14freshcases (two per part) and caps each part to180seconds. Its planned denominator is14, with zero baseline cases. This preserves a common14case comparison across all three candidates while allowing the original Ministral84case run to continue. Different time caps can cause unequal coverage, so comparisons require completed paired case IDs and must expose every missing response.
