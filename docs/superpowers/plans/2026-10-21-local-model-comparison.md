# Local model comparison implementation plan

> **For agentic workers:** Execute the already authorized bounded experiment; production inference behavior stays unchanged.

**Goal:** Compare three local model candidates on frozen synthetic Italian statistical interpretation cases and deliver the existing complete jamovi file.

**Architecture:** A development-only runner shares the Minerva evidence task and strict response validator. Each candidate receives its own pinned official template and tokenizer oracle; GitHub public CPU runners execute seven observable parts with checkpoints.

**Tech stack:** Python 3.12, llama.cpp b11756, Transformers tokenizer, mistral-common.

**Spec:** User request: test Qwen3.8-27B, Ministral 3 14B Reasoning 2512 and Qwen3.5-9B; preserve local processing and include final OMV.

## Constraints

- Frozen baseline70 and fresh14; no expected labels or rationales in model inputs.
- CPU only, literal loopback, owned server, memory-only capability, no student files published.
- Temperature0, seed0, context4096, max output1024, reasoning budget512; record native differences.
- Each public runner job bounded to120minutes; no Windows rebuild or paid runner.
- Technical failures, missing cases and truncation stay visible in planned denominators.
- No inference result establishes cheating or authorship.

## Review focus

Tokenizer mismatch must stop evaluation; exact matching checks cover every submitted prompt. Truncation and malformed evidence must fail validity. Missing or interrupted cases must not improve agreement. Model and runtime hash failures must stop loading. Case labels must remain outside submitted payloads.

## Task1: Development runner

Files: tools/compare_local_models.py; tests/test_compare_local_models.py.

- [ ] Add and run failing tests for leakage, parity, response validity and planned denominators.
- [ ] Implement asset preparation, owned server, official oracle, chat inference and checkpoint aggregation.
- [ ] Run focused tests and inspect runner behavior before publication.

## Task2: Bounded workflow and frozen fixtures

Files: tests/fixtures/local_model_assets.json; tests/fixtures/local_model_fresh_cases.json; .github/workflows/confronto-modelli.yml; workflow branch exclusions.

- [ ] Pin exact assets and tokenizer dependencies.
- [ ] Exclude experimental branch from unrelated native builds.
- [ ] Publish explicit public source allowlist, run three candidates, retain raw evidence.
- [ ] Compare baseline and fresh results separately, including all incomplete cases.

## Task3: Deliverables

- [x] Recover Prova_Completa.omv without altering its bytes; inspect14analyses,26tables,24visible images.
- [ ] Save final OMV, comparison report and source export with concrete download links.
