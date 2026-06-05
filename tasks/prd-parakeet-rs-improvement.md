# PRD: parakeet-rs Crate-Wide Improvement Initiative

**Type:** Improvement roadmap (not a single feature). PLAN ONLY.
**Crate:** `parakeet-rs` v0.3.6 (`/Users/sujith/work/2026/exps/parakeet-rs`)
**Semver posture:** 0.x (pre-1.0). Breaking changes are permitted but MUST be deliberate and called out.

---

## Pre-drafting clarity pass (socratic + jobs-to-be-done)

**Primary job (one sentence):** A Rust application developer hires `parakeet-rs` to turn an audio stream (file or live mic) into accurate, correctly-timed text across many languages and model variants, without writing ONNX plumbing, and without surprises when they upgrade.

**Top 3 unstated assumptions surfaced by socratic questioning:**
1. "Improvement" is assumed to mean a *ranked backlog of one-PR tasks*, not one large rewrite. The four themes map to existing audit lanes (correctness, performance, API, models) plus code-quality and tests. (Confirmed by context pack and intent ledger.)
2. The multilingual language-lock is assumed to be the *trigger* finding, not the whole scope. Whether it is fixable in the library vs inherent to the ONNX graph is UNRESOLVED and is an explicit open question that gates the correctness/model themes.
3. "Success" for a published library is assumed to be measured by semver/back-compat discipline, streaming correctness, accuracy vs a reference, build-across-features, and test coverage; NOT by app-level KPIs.

**Assumptions that would materially change scope -> logged to Open Questions:** language-lock root cause (OQ-1), upstream NeMo language-conditioning behavior (OQ-2), whether a shared variant abstraction can be extracted without breaking the public API (OQ-5).

---

## 1. Introduction / Overview

`parakeet-rs` is a Rust ASR library wrapping NVIDIA Parakeet / Nemotron family ONNX models via `ort`. It ships ~7,766 lines across 24 `src/*.rs` modules and exposes 10 model variants (CTC, TDT, EOU, Nemotron, Unified, Multitalker, Cohere, Sortformer diarization, plus shared infra). It supports file and live-mic streaming, cache-aware chunked inference, word/segment timestamps, and 11 ONNX execution-provider feature flags.

A live multilingual test (Nemotron 3.5, `target_lang="auto"`) surfaced a correctness problem: the transcript locked onto one language and could not code-switch mid-stream. That single finding triggered a broader request: a **full, exhaustive, ranked improvement roadmap** across four dimensions. This PRD frames that initiative. It defines the goals, the four themes (treated as epics), library-appropriate success metrics, non-goals, and open questions. It does NOT design solutions or write code; per-theme audit lanes and the synthesis lane produce the concrete findings and the one-PR task backlog.

This PRD is the *what* and the *why*. The *how* lives in the audit lane files (`A1`..`A6`, `RU`, `02-architecture.md`) and the ranked backlog in `90-synthesis.md`.

---

## 2. Goals

- **G1 (Correctness & streaming):** Eliminate or correctly document the multilingual language-lock; guarantee streaming reset/flush/state-carry behavior is correct and tested across all stateful variants.
- **G2 (Performance & RTF):** Identify and reduce per-chunk hot-path costs (mel/realfft, ndarray copies, ort run, tokenizer) so real-time-factor (RTF) stays comfortably < 1.0 on CPU for streaming variants, and provide threading/async guidance.
- **G3 (API & ergonomics):** Make the public surface consistent across the 10 variants (construction, config, error type, timestamp options) with deliberate, documented semver discipline.
- **G4 (Model coverage & features):** Close feature gaps (timestamps, diarization, quantization, language re-detection) where they are library-fixable, and clearly document where a gap is model-inherent.
- **G5 (Code quality):** Reduce duplication across the 10 `model_X.rs` + wrapper pairs via shared abstractions that do NOT break the public API.
- **G6 (Tests & CI):** Establish regression guards for the correctness invariants that are currently unguarded (only 6 of 24 modules have inline tests; no `tests/` dir).
- **G7 (Roadmap output):** Produce a single ranked, deduplicated backlog of one-PR tasks with severity, evidence (`file:line`), and ordering.

---

## 3. User Stories (themes as epics)

Each "story" here is a theme-level epic. The concrete one-PR tasks are produced by the audit lanes and ranked in `90-synthesis.md`. These epics frame the jobs; they are not individually implementable in one session.

### EP-01: Correctness & streaming reliability (Theme 1)
**Progress:** To do

**Description:** As a developer streaming multilingual audio, I want the transcript to follow the spoken language (or at least to behave per a documented, predictable policy) so that a speaker who code-switches is not silently mistranscribed.

**Acceptance Criteria:**
- [ ] Root cause of the language-lock is resolved as model-inherent vs library-fixable, with evidence (`A1`, `A4`, `RU`).
- [ ] If library-fixable: a re-detect / re-prompt-at-boundary or change-language-mid-stream capability is specified (NOT implemented here).
- [ ] If model-inherent: the limitation is documented and `auto` semantics (detect-once vs continuous) are clarified.
- [ ] Streaming reset/flush and cross-chunk state-carry invariants are enumerated for every stateful variant (Nemotron, EOU, Unified).
- [ ] No regression to the English-only Nemotron path or other shipped variants.

### EP-02: Performance & RTF (Theme 2)
**Progress:** To do

**Description:** As a developer running streaming inference, I want per-chunk processing to be fast and predictable so that real-time transcription keeps up with input and I know how to thread it.

**Acceptance Criteria:**
- [ ] Hot paths quantified (mel/realfft, per-chunk ndarray allocations/copies, ort run, tokenizer decode) with evidence.
- [ ] Concrete, ranked reduction opportunities identified (e.g. buffer reuse, avoiding copies) without correctness regression.
- [ ] Threading/async guidance assessed: `transcribe_chunk` is `&mut self` and blocking; document the intended concurrency model.
- [ ] Execution-provider story assessed (default CPU; 11 EP flags) for build and runtime.

### EP-03: Public API consistency & ergonomics (Theme 3)
**Progress:** To do

**Description:** As a developer choosing among 10 variants, I want consistent construction, config, error handling, and timestamp options so that switching variants does not mean relearning a bespoke API each time.

**Acceptance Criteria:**
- [ ] Public surface (`lib.rs:76-99`) audited for consistency across variants (do they share a trait/shape or is each bespoke?).
- [ ] `from_pretrained` signatures, builder/config patterns, `ExecutionConfig`, and the `Error` enum (`error.rs`) audited for ergonomics.
- [ ] Each proposed API change tagged with its semver impact (breaking vs additive) given 0.x posture.
- [ ] Docs/examples coverage gaps identified across the 9 examples.

### EP-04: Model coverage & features (Theme 4)
**Progress:** To do

**Description:** As a developer needing timestamps, diarization, quantization, or language re-detection, I want each variant's feature matrix to be clear and the library-fixable gaps closed so that I can pick the right variant with confidence.

**Acceptance Criteria:**
- [ ] Per-variant feature matrix produced (timestamps, diarization, quantization, language handling).
- [ ] Each gap classified model-inherent vs library gap, with evidence.
- [ ] Variant-unification opportunities identified that do NOT break the public API.
- [ ] Language re-detection feasibility resolved against `RU` upstream research.

### EP-05: Code quality & structure (supporting theme)
**Progress:** To do

**Description:** As a maintainer, I want duplication across the 10 `model_X.rs` + wrapper pairs reduced so that fixes apply once, not ten times.

**Acceptance Criteria:**
- [ ] Duplication quantified across encoder/decoder run, mel, cache, EP setup.
- [ ] A shared abstraction proposed that preserves the public API (largest files: `sortformer.rs` 1254, `nemotron.rs` 786, `multitalker.rs` 771, `parakeet_unified.rs` 590).

### EP-06: Tests & CI (supporting theme)
**Progress:** To do

**Description:** As a maintainer, I want regression guards on the core correctness invariants so that future changes (esp. streaming/language fixes) cannot silently break shipped behavior.

**Acceptance Criteria:**
- [ ] Unguarded invariants mapped (streaming reset/flush, decoder greedy/blank handling, mel vs reference, timestamp alignment, language tag stripping).
- [ ] A test/fixture strategy proposed using locally available models (`./nemotron`, `./nemotron_multi`).

---

## 4. Functional Requirements

These are requirements on the *roadmap output and the planning run*, not on shipped code (PLAN ONLY).

- **FR-1:** The roadmap must cover all four user-prioritized themes plus the two supporting themes (code-quality, tests), each as a distinct audit lane.
- **FR-2:** Every audit finding must cite `file:line` or be explicitly labelled as an assumption or upstream source. No evidence -> labelled hypothesis.
- **FR-3:** Every finding must carry a severity (CRITICAL | HIGH | MEDIUM | LOW | NIT), root cause, impact, regression risk of the fix, recommendation, and reverification note.
- **FR-4:** The language-lock must be resolved as model-inherent vs library-fixable, against both the code and the upstream NVIDIA/NeMo reference (`RU`), before `A1`/`A4` conclude.
- **FR-5:** Every proposed API change must state its semver impact under the 0.x posture (additive vs breaking) so breaking changes are deliberate.
- **FR-6:** Success metrics must be library-appropriate (semver/back-compat, streaming correctness, accuracy vs reference, build-across-features, test coverage), NOT the Meetily 6-KPI set.
- **FR-7:** The final output must be a single ranked, deduplicated one-PR task backlog (`90-synthesis.md`), with findings refuted/confirmed by the adversarial verification lane (`85`).
- **FR-8:** The plan must preserve the crate's core value: multi-variant ONNX ASR. No proposal may remove a shipped variant or its public entry point without explicit justification and semver call-out.
- **FR-9:** Output must contain NO em dashes; no `.env`/`.pem`/`.p8`/`.key` files are read.

---

## 5. Non-Goals (Out of Scope)

Produced via inversion (enumerating what this initiative deliberately will NOT do):

- **NG-1:** No implementation. This run writes plans only; code is produced later by `orchestrator-implementor` / `/tdd-workflow`.
- **NG-2:** No rewrite from scratch. The crate's architecture (per-variant `model_X.rs` + wrapper) is the starting point, not a target for wholesale replacement.
- **NG-3:** No breaking of the crate's core value: multi-variant ONNX ASR stays. No variant or public entry point is removed casually.
- **NG-4:** No training, fine-tuning, or re-export of ONNX models. If the language-lock is model-inherent, the fix is documentation/API-policy, not model surgery.
- **NG-5:** No new model families beyond the 10 shipped variants in this initiative (coverage = closing gaps in existing variants, not adding new architectures).
- **NG-6:** No app-level concerns (Meetily KPIs, UI, recording lifecycle, Tauri IPC). This is a library, not an app integration.
- **NG-7:** No guarantee of 1.0 stabilization in this initiative. Semver posture stays 0.x; API changes are allowed but documented.
- **NG-8:** No commitment to a specific execution provider's runtime correctness (CUDA/CoreML/etc.); EP work is build/guidance assessment, not per-EP certification.
- **NG-9:** No multi-language *simultaneous* output guarantee unless the model supports it; the language-lock resolution may land on "documented limitation."

---

## 6. Design Considerations

Not applicable in the UI sense (this is a library). The relevant "design" surface is the **public API shape** and is owned by EP-03 / `A3-api.md`. Reuse existing shared infra (`execution.rs`, `timestamps.rs`, `decoder.rs`, `audio.rs`, `config.rs`, `error.rs`) rather than introducing parallel mechanisms.

---

## 7. Technical Considerations

- **Stack:** `ort 2.0.0-rc.12` (default-features=false: std, ndarray, api-24), `ndarray 0.17`, `tokenizers 0.23.1` (onig), `realfft 3`, `hound 3.5`, `eyre 0.6`, `serde`/`serde_json`; dev-dep `cpal 0.15` (mic example).
- **Streaming model state:** cache-aware encoder (`NemotronEncoderCache`), autoregressive decoder (`last_token` carried across chunks), per-variant chunk sizes (Nemotron 8960/560ms, EOU 2560/160ms). Any language/streaming fix must reason about carried state, not just the prompt index.
- **Language conditioning:** `auto` = prompt index 101 in `PROMPT_DICTIONARY` (`nemotron.rs:48`), set once via `set_target_lang` (`nemotron.rs:475-491`), fed every chunk (`nemotron.rs:591,691`), preserved across `reset()` (`nemotron.rs:495-509`).
- **Concurrency:** `transcribe_chunk` is `&mut self` and blocking (holds a model Mutex). Threading guidance is a deliverable, not assumed.
- **Test assets:** local models `./nemotron` (EN 2.3G), `./nemotron_multi` (multilingual 3.5 2.4G) available for integration fixtures.

---

## 8. Success Metrics (library-appropriate)

- **SM-1 (semver discipline):** 100% of proposed API changes carry an explicit additive-vs-breaking tag; zero undocumented breaking changes in the resulting backlog.
- **SM-2 (streaming correctness):** Every stateful variant has enumerated reset/flush/state-carry invariants, each with a proposed regression test.
- **SM-3 (accuracy vs reference):** Language-lock resolved with evidence; if fixable, a measurable acceptance (transcript follows language change within N chunks/seconds) is defined; if inherent, documented.
- **SM-4 (build across features):** The roadmap includes a task to verify the crate builds across all 11 EP feature flags and `multitalker`/`cohere`/`sortformer` combinations.
- **SM-5 (test coverage):** Coverage target raised from 6/24 modules toward guarding all named correctness invariants (exact target set by `A6`).
- **SM-6 (roadmap completeness):** All four primary themes + two supporting themes represented; every HIGH/CRITICAL finding survives adversarial verification (`85`).
- **SM-7 (no regression):** Roadmap explicitly protects the English-only Nemotron path and every shipped variant.

---

## 9. Open Questions

| # | Question | Owner lane | Why it matters |
|---|----------|-----------|----------------|
| OQ-1 | Is the multilingual language-lock inherent to the ONNX graph (fixed prompt_index + carried decoder state) or a fixable library choice (re-detect/re-prompt at boundaries, expose re-detect API)? | A1, A4 | Gates whether Theme 1/4 ships a fix or a documented limitation (SM-3). |
| OQ-2 | How does upstream NVIDIA/NeMo Nemotron 3.5 streaming multilingual handle language selection / auto-detection? Does it re-detect per utterance/buffer? | RU | Authoritative answer to OQ-1; must precede A1/A4 conclusions (FR-4). |
| OQ-3 | Where are the real performance hot paths, and is inference blocking the caller with no async/threading guidance? | A2 | Sizes Theme 2 tasks and the concurrency-guidance deliverable. |
| OQ-4 | How much code is duplicated across the 10 `model_X` + wrapper pairs? | A5, 02 | Determines feasibility of EP-05 shared abstraction. |
| OQ-5 | Can a shared variant abstraction be extracted WITHOUT breaking the public API (`lib.rs:76-99`)? | A5, 02, A3 | Constrains EP-05 against EP-03 semver discipline. |
| OQ-6 | Which variants lack word-level timestamps / diarization / quantization, and which gaps are model-inherent vs library? | A4 | Builds the per-variant feature matrix for Theme 4. |
| OQ-7 | Which correctness invariants are currently unguarded (streaming reset/flush, decoder, mel, timestamp alignment, language-tag stripping)? | A6 | Defines the test/fixture strategy and SM-5 target. |
| OQ-8 | What is the exact intended semver target (stay 0.x indefinitely vs path to 1.0)? | A3 | Affects how aggressively breaking changes are scheduled. Default assumption: stay 0.x, breaking allowed but deliberate. |
| OQ-9 | Should `auto` semantics be split into `detect-once` vs `detect-continuous` as distinct public modes? | A3, A4 | Depends on OQ-1; would be an additive API change if fixable. |

---

## Cross-references for other lanes

- **02-architecture:** This PRD treats per-variant `model_X.rs` + wrapper as the baseline (NG-2). Map shared infra vs duplication; EP-05/OQ-4/OQ-5 depend on your duplication quantification. Constrain any shared-abstraction proposal against the public API in `lib.rs:76-99`.
- **A1-correctness:** EP-01 owns the language-lock and streaming reset/flush/state-carry invariants. Resolve OQ-1 against code AND `RU` (FR-4) before concluding. Protect the English-only Nemotron path (SM-7).
- **A2-performance:** EP-02 / OQ-3. Quantify hot paths (mel/realfft, ndarray copies, ort run, tokenizer); deliver the concurrency model for the `&mut self` blocking `transcribe_chunk`.
- **A3-api:** EP-03. Every change carries a semver tag (FR-5, SM-1) under 0.x posture (OQ-8). Owns OQ-9 (`auto` mode split). Cross-check EP-05 abstraction against public API (OQ-5).
- **A4-models:** EP-04 / OQ-6. Build the per-variant feature matrix; classify each gap model-inherent vs library; co-own the language-lock resolution with A1 and OQ-9 with A3.
- **A5-quality:** EP-05 / OQ-4 / OQ-5. Quantify duplication across the 10 pairs; propose a public-API-preserving shared abstraction.
- **A6-tests:** EP-06 / OQ-7 / SM-5. Map unguarded invariants; propose fixtures using `./nemotron` and `./nemotron_multi`.
- **RU-upstream:** OQ-2. Authoritative answer that gates A1/A4 (FR-4). Report whether upstream re-detects language per utterance/buffer.
- **80-feasibility / 85-verify / 90-synthesis:** Enforce FR-2/FR-3/FR-7. Every HIGH/CRITICAL finding must survive adversarial refutation (85) before entering the ranked backlog (90). Success measured by SM-1..SM-7.

OUTPUT_PATH: /Users/sujith/work/2026/exps/parakeet-rs/tasks/prd-parakeet-rs-improvement.md
