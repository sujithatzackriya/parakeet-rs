# A6 — AUDIT: tests & CI

**/think frameworks:** (1) **Inversion** — "what change would silently ship a broken transcript today?" maps every unguarded path to a regression that current CI would not catch; (2) **Risk-surface / first-principles** — separate pure logic (testable with no ONNX) from model-bound logic (needs the 2.3-2.4 GB weights) to design a coverage strategy that is actually runnable in CI.

> PLAN ONLY. All findings cite `file:line`. Severity per the shared schema.

---

## 0. Ground truth (verified this lane)

| Fact | Evidence |
|------|----------|
| CI **does exist** (context pack line 57 "NO CI" is now stale/superseded) | `.github/workflows/rust.yml:1-44` |
| CI runs only `cargo build --verbose` + `cargo test --verbose`, default features, ubuntu-only | `.github/workflows/rust.yml:21-24` |
| No `fmt`, no `clippy`, no feature matrix, no MSRV check in CI | `.github/workflows/rust.yml` (absent) |
| No `tests/` dir, no `benches/` dir | filesystem |
| Inline `#[cfg(test)]` in **6 of 24** modules | `audio.rs:268`, `timestamps.rs:212`, `decoder_tdt.rs:87`, `cohere.rs:432`, `sortformer.rs:1202`, `parakeet_unified.rs:577` |
| Total ~20 `#[test]` fns: audio 2, timestamps 5, decoder_tdt 7, cohere 3, sortformer 2, parakeet_unified 1 | grep `#[test]` |
| `test_en.wav` (197 KB) exists locally but is **NOT git-tracked** | `git ls-files` returns nothing for wav; file present on disk |
| Gated features: `sortformer`, `multitalker` (implies sortformer), `cohere`, plus 9 EP flags | `Cargo.toml:58-74` |
| `cohere` example has `required-features = ["cohere"]`; default `cargo build` does NOT compile it | `Cargo.toml:40-43` |
| Gated modules are `#[cfg(feature=...)]` and absent from a default build | `lib.rs:62-99` |

---

## 1. COVERAGE MAP — what is tested vs the 24 modules

Legend: **REAL** = asserts a meaningful invariant; **THIN** = trivial/constant; **NONE** = no test.

| Module | Tests | Quality | What is actually asserted | Big unguarded surface |
|--------|-------|---------|---------------------------|------------------------|
| `audio.rs` | 2 (`:279`,`:318`) | REAL (partial) | STFT power concentrates at expected bin for a 1 kHz sine; STFT output shape | **Mel filterbank values, Slaney norm, log-guard, per-feature normalization (Bessel N-1), preemphasis, multi-channel downmix, sample-rate mismatch error** all untested — `extract_features_with_cache` (`:206-266`) has ZERO coverage despite being the front of every transcribe path |
| `timestamps.rs` | 5 (`:216-358`) | REAL | word grouping, hyphenated merge, sentence grouping, repetition preservation, space-token vs digit boundary | Sentence-split on `?`/`!`/abbreviations; empty-token input; `TimestampMode::Tokens` passthrough |
| `decoder_tdt.rs` | 7 (`:98-203`) | REAL | digit-spacing heuristic (word/article-a/uppercase/symbol/phrase), token spacing, full word+timestamp flow | Frame→time math (`:35-40`) uses hardcoded `encoder_stride=8`; the `start+0.01` last-token end (`:39`) is untested; special-token `<...>` skip (`:65-67`) untested for `<unk>` edge |
| `cohere.rs` | 3 (`:436-463`) | REAL | `argmax`, `SUPPORTED_LANGUAGES.len()==14`, n-gram repetition detection | Token-template assembly (`:41-60` constants), lang-token resolution (`:103`, `:194`), the whole transcribe path |
| `sortformer.rs` | 2 (`:1212`,`:1245`) | REAL but **DUPLICATE** | STFT bin + shape — same two assertions as `audio.rs`, re-implemented against `Sortformer::stft` | Clustering, speaker assignment, segment merging, the entire diarization logic (1254 L file) |
| `parakeet_unified.rs` | 1 (`:581`) | REAL | default streaming config frame arithmetic aligns to subsampling (left=560, chunk=56, rc=56, enc 70/7) | `validate()` rejection of bad configs; offline vs streaming path; chunk-boundary token stitching |
| `decoder.rs` (CTC) | **NONE** | — | — | **CTC collapse + blank/pad dedup (`:68-86`), frame-tracked collapse for timestamps (`:89-121`), progressive-decode word-boundary detection (`:160-191`)** — all unguarded pure logic; `decode_with_beam_search` (`:200`) is a silent stub that falls back to greedy |
| `nemotron.rs` | **NONE** | — | — | **`is_lang_tag` (`:78-93`), `lang_tag_ids` (`:256`), `set_target_lang`/prompt lookup (`:475-491`), `reset()` language preservation (`:495-509`), `get_transcript` lang-tag stripping (`:513-521`), SentencePiece protobuf parser (`:125-230`)** — the language-lock seed finding lives here, entirely untested |
| `model_tdt.rs` | **NONE** | — | — | **`greedy_decode` (`:176-290`): blank skip, duration-based frame advance, `max_tokens_per_step=10` stall guard, state carry on emit** — the RNN-T/TDT decode core |
| `vocab.rs` | **NONE** | — | — | `from_file` parser, blank-id detection, default-to-last fallback (`:46-49`) |
| `model.rs`, `model_eou.rs`, `model_unified.rs`, `model_nemotron.rs`, `model_multitalker.rs`, `model_cohere.rs` | **NONE** | — | — | ONNX session wiring, encoder/decoder runs, cache shapes |
| `parakeet.rs`, `parakeet_tdt.rs`, `parakeet_eou.rs`, `multitalker.rs` | **NONE** | — | — | wrapper transcribe paths, streaming reset/flush |
| `execution.rs` | **NONE** | — | — | EP selection logic, thread config |
| `config.rs`, `error.rs`, `transcriber.rs` | **NONE** | — | — | config JSON deserialize defaults |

**Headline gaps (highest-risk, all unguarded):**
1. **Streaming reset/flush** — `Nemotron::reset` (`nemotron.rs:495-509`) deliberately preserves `prompt_index` and clears decoder/encoder state; nothing pins this contract. A refactor could silently clear or preserve the wrong field. This is exactly the language-lock surface.
2. **TDT greedy/blank/duration** — `model_tdt.rs:176-290`. The duration-skip + blank-advance loop is the transcription core; no test pins "blank advances by 1", "duration>0 skips N frames", "max_tokens_per_step caps emission".
3. **CTC collapse** — `decoder.rs:68-121`. Dedup of repeated tokens and pad handling are pure and trivially testable, yet untested.
4. **Mel vs reference** — `audio.rs:206-266`. The comment (`:70-73`) admits wrong FFT once produced all-blank output; only a bin-location test guards it, not the mel-filterbank values or normalization that the model was trained against.
5. **Language-tag stripping** — `is_lang_tag` (`nemotron.rs:78-93`) is a self-contained byte matcher (`<en>`, `<en-US>`) begging for a table test; a regression here corrupts multilingual transcripts.
6. **Chunk-boundary correctness** — `parakeet_unified.rs` config math is tested but the actual token stitching across chunks for Nemotron/EOU/Unified is not.
7. **The 10 variants' transcribe paths** — zero end-to-end coverage; correctness is only verified by hand-running examples.

---

## 2. FINDINGS

### A6-F1 — CI never builds or tests the gated features (sortformer / multitalker / cohere)
- **SEVERITY: HIGH**
- **Root cause:** `rust.yml:21-24` runs `cargo build`/`cargo test` with default features only (`default = ["cpu","ort-defaults"]`, `Cargo.toml:59`). `sortformer`, `multitalker`, `cohere` are `#[cfg(feature=...)]`-gated (`lib.rs:62-99`) and never compiled in CI.
- **Evidence:** `.github/workflows/rust.yml:21-24`; `Cargo.toml:70-72`; `lib.rs:62-71,95-99`.
- **Impact:** A change that breaks `sortformer.rs` (1254 L), `multitalker.rs` (771 L), or `cohere.rs` (464 L) — including their inline tests — passes CI green and ships broken. The `cohere` example (`Cargo.toml:40-43`, `required-features=["cohere"]`) is likewise never compiled.
- **Regression risk of fix:** None. Adding feature-matrix jobs is additive.
- **Recommendation:** Add a feature-flag build/test matrix (see §3 CI).
- **Reverification:** `cargo build --features cohere` / `--features multitalker` / `--features sortformer` locally; confirm each currently compiles (likely yes, but unguarded going forward).

### A6-F2 — No regression guard for the multilingual language-lock (seed finding)
- **SEVERITY: HIGH**
- **Root cause:** The language conditioning surface — `set_target_lang` prompt lookup (`nemotron.rs:475-491`), `reset()` preserving `prompt_index` (`:495-509`), `is_lang_tag`/`lang_tag_ids` stripping (`:78-93`,`:256`), `get_transcript` filtering (`:513-521`) — has zero tests. Whatever the correctness/model lanes (A1/A4) decide (re-detect API, re-prompt at boundaries, reset semantics), there is no test that would have caught the lock or would catch a regression of the fix.
- **Evidence:** `nemotron.rs` has no `#[cfg(test)]`.
- **Impact:** The exact bug that triggered this whole run is invisible to the test suite. Any future change to reset/prompt semantics is unverified.
- **Regression risk of fix:** Tests for `is_lang_tag` and `reset`-field-contract are pure and zero-risk. A golden streaming transcript test (see A6-F6) needs the model and must be `#[ignore]`-gated.
- **Recommendation:** (a) pure unit tests for `is_lang_tag` (table of `<en>`,`<en-US>`,`<x>`,`<EN>`,`abc`,`<<>>`), `set_target_lang` (known lang → `Some(idx)`, unknown → `Err`, English variant → `Err`), and a `reset()` contract test asserting `prompt_index` is unchanged while `last_token`/`audio_buffer`/`accumulated_tokens` are cleared; (b) gated golden streaming test once A1 lands a re-detect API.
- **Reverification:** After A1/A4 design the fix, the golden test must encode the expected behavior (e.g. code-switch produces both languages, or documents that the lock is model-inherent).

### A6-F3 — Transcription core (TDT greedy, CTC collapse) is entirely unguarded
- **SEVERITY: HIGH**
- **Root cause:** `model_tdt.rs:176-290` (duration/blank greedy loop) and `decoder.rs:68-121` (CTC collapse + frame-tracked collapse) carry all transcription correctness yet have no tests. CTC collapse is pure and needs no model. TDT greedy is model-bound (calls `decoder_joint.run`) but the *frame-advance arithmetic* (`:279-287`) is pure and extractable.
- **Evidence:** neither module has `#[cfg(test)]`.
- **Impact:** A regression in blank handling or duration skip silently degrades every TDT/CTC transcript; only manual example runs would notice.
- **Regression risk of fix:** CTC-collapse tests zero-risk. Extracting the TDT frame-advance logic into a pure helper to test it is a small refactor (call out to A5/A3 lanes).
- **Recommendation:** Pure tests for `ctc_collapse` (repeated tokens dedup, pad/blank removal, empty input) and `ctc_collapse_with_frames` (start/end frame assignment). For TDT, either extract the advance rule into a pure fn and unit-test it, or cover via the gated golden test.
- **Reverification:** Build the tests RED against current code (they should pass once written if logic is correct), then they pin behavior.

### A6-F4 — Mel feature extraction untested against reference values
- **SEVERITY: MEDIUM**
- **Root cause:** `extract_features_with_cache` (`audio.rs:206-266`) — preemphasis, mel projection, Slaney norm, log-guard `2^-24`, per-feature normalization with Bessel `N-1` — is the input to every model and is only indirectly touched by the two STFT tests (`:279`,`:318`), which test the raw spectrogram, not the mel pipeline. The code comment (`:70-73`) notes a prior wrong-FFT bug that produced all-blank output.
- **Evidence:** `audio.rs:206-266` has no direct test; tests at `:279`,`:318` exercise `stft` only.
- **Impact:** A subtle change to normalization or the log-guard shifts features off the training distribution and degrades accuracy without any error — hard to debug.
- **Regression risk of fix:** None; reference vectors can be generated once from a committed short WAV (or from librosa offline) and pinned with a tolerance.
- **Recommendation:** Golden mel test — feed a deterministic signal (or the committed `test_en.wav`), assert mel shape and a few reference values within tolerance; assert sample-rate-mismatch returns `Error::Audio` (`:213-218`); assert multi-channel downmix averages (`:220-226`).
- **Reverification:** Generate references with `ndarray` debug-print on a known-good build, or cross-check against NeMo's preprocessor for the same input.

### A6-F5 — `test_en.wav` is not committed; no fixture exists for any integration test
- **SEVERITY: MEDIUM**
- **Root cause:** `test_en.wav` (197 KB) is present on disk but absent from `git ls-files`. There is no committed audio fixture, so no integration or golden test can run reproducibly in CI or on a fresh clone.
- **Evidence:** `git ls-files | grep wav` returns nothing; file exists at `./test_en.wav`.
- **Impact:** Any golden-transcript or mel-reference test is non-portable. Contributors cannot run the would-be integration suite.
- **Regression risk of fix:** A short (~2-3 s, 16 kHz mono) committed WAV is small and licensable; pick a clearly-licensed/synthetic clip to avoid IP issues.
- **Recommendation:** Commit a short, clearly-licensed mono 16 kHz WAV under `tests/fixtures/`. Use it for the no-model mel test (A6-F4) and as the audio input for the gated model integration tests (A6-F6). Do NOT commit the 2.3-2.4 GB ONNX models.
- **Reverification:** Confirm the chosen clip's license permits redistribution.

### A6-F6 — No integration / golden-transcript tests against the local models
- **SEVERITY: MEDIUM**
- **Root cause:** The only way the 10 variants' transcribe paths are validated today is hand-running the 9 examples. No `tests/` integration suite, no golden transcripts, no `#[ignore]`/feature-gated model tests.
- **Evidence:** no `tests/` dir; models present at `./nemotron`, `./nemotron_multi` (context pack lines 66) but not referenced by any test.
- **Impact:** Accuracy and streaming regressions across variants are caught only by manual testing. No automated proof that English-only Nemotron still works after a multilingual fix (a stated quality gate).
- **Regression risk of fix:** Gated tests must default-skip so CI without models stays green.
- **Recommendation:** Add `tests/integration_models.rs` with `#[ignore]` (or an `integration-models` feature / `PARAKEET_MODEL_DIR` env guard) tests that, when `./nemotron`/`./nemotron_multi` are present: (a) offline-transcribe the committed WAV and assert the transcript contains expected words (golden, fuzzy-match to tolerate minor decode drift); (b) feed the same audio in streaming chunks and assert the concatenated streaming transcript ~= the offline transcript (chunk-boundary correctness); (c) a multilingual code-switch golden that encodes whatever A1/A4 decide for the language-lock.
- **Reverification:** Run `cargo test -- --ignored` locally with models present; CI skips by default.

### A6-F7 — `sortformer.rs` STFT tests duplicate `audio.rs` instead of testing diarization
- **SEVERITY: LOW**
- **Root cause:** `sortformer.rs:1212-1253` re-implements the same "bin 32 dominates" + "shape" assertions already in `audio.rs:279-331`, against a private `Sortformer::stft`. The 1254-line diarization logic (clustering, speaker assignment) has no test.
- **Evidence:** `sortformer.rs:1212`,`audio.rs:279` — near-identical bodies.
- **Impact:** Test effort spent re-proving STFT; the actual diarization risk surface is unguarded. (Note: A5/A2 may flag the duplicated `stft` impl itself.)
- **Regression risk of fix:** None.
- **Recommendation:** Drop or thin the duplicate STFT tests; add tests for the diarization-specific pure logic (segment merge, speaker-label assignment) where extractable without the model.
- **Reverification:** Identify the pure helpers in `sortformer.rs` and confirm they can be unit-tested in isolation.

### A6-F8 — CI lacks fmt, clippy, and MSRV checks
- **SEVERITY: LOW**
- **Root cause:** `rust.yml` runs only build+test. No `cargo fmt --check`, no `cargo clippy -- -D warnings`, no `rust-version`/MSRV in `Cargo.toml` (`Cargo.toml:1-4` has none) nor an MSRV CI job.
- **Evidence:** `.github/workflows/rust.yml` (absent); `Cargo.toml` has no `rust-version`.
- **Impact:** Style/lint drift and accidental MSRV bumps land unnoticed; for a published crate this erodes contributor experience and downstream build stability.
- **Regression risk of fix:** `clippy -D warnings` may surface existing lints (coordinate with A5); start with `clippy` non-blocking or fix lints first.
- **Recommendation:** Add `fmt --check` and `clippy` jobs; set and pin an MSRV. (See §3.)
- **Reverification:** Run `cargo fmt --check` and `cargo clippy --all-features` locally to scope existing violations before making them blocking.

### A6-F9 — `decode_with_beam_search` is a silent stub
- **SEVERITY: NIT**
- **Root cause:** `decoder.rs:200-206` accepts `_beam_width` and silently falls back to greedy `decode`. No test, no deprecation, no doc warning surfaced to API users beyond the inline comment.
- **Evidence:** `decoder.rs:199-206`.
- **Impact:** Callers requesting beam search get greedy results with no signal. (API lane A3 should decide: remove, document, or implement.)
- **Regression risk of fix:** N/A for tests.
- **Recommendation:** Out of scope for tests beyond noting it; flag to A3. If kept, a test asserting it equals greedy at least pins current behavior.
- **Reverification:** none needed.

---

## 3. TEST & CI STRATEGY (concrete tasks)

### Strategy split: no-model vs model-bound
The hard constraint is that real model tests need 2.3-2.4 GB ONNX weights that cannot live in the repo. Split accordingly:

**Tier 1 — pure logic, NO model, runs in CI on every push (high ROI, do first):**
- `decoder.rs`: `ctc_collapse`, `ctc_collapse_with_frames` (dedup, pad/blank, frame assignment, empty input). [A6-F3]
- `model_tdt.rs`: extract the frame-advance rule (`:279-287`) into a pure fn, test blank-advance / duration-skip / max-token cap. [A6-F3]
- `nemotron.rs`: `is_lang_tag` table; `set_target_lang` lookup (known/unknown/English-variant-error); `reset()` field contract; `get_transcript` lang-tag filtering with a stub vocab. [A6-F2]
- `audio.rs`: mel reference values + Slaney norm + log-guard + Bessel normalization + sample-rate-mismatch error + multi-channel downmix, using a deterministic signal or committed WAV. [A6-F4]
- `vocab.rs`: `from_file` parsing, blank-id detection, default-to-last fallback.
- `cohere.rs`: extend existing tests to cover lang-token resolution and template assembly. [A6-F1 coverage]

**Tier 2 — model-bound, gated, runs only when weights present (do after fixtures):**
- `tests/integration_models.rs`, gated by `#[ignore]` + a `PARAKEET_MODEL_DIR` env check (or an `integration-models` feature). Golden offline transcript, streaming==offline equivalence, multilingual code-switch golden. [A6-F6]
- Requires the committed short WAV fixture. [A6-F5]

### CI matrix (replace the single default-feature job)
Propose `rust.yml` jobs (additive to the existing release job):
1. **lint** (ubuntu): `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings` (scope existing lints first with A5). [A6-F8]
2. **test-default** (ubuntu): `cargo test` (current behavior, keep).
3. **build-features** (ubuntu): build each gated feature so they never rot —
   `cargo build --features cohere`, `--features multitalker`, `--features sortformer`,
   plus `cargo test --features "cohere multitalker sortformer"` to run their inline tests. [A6-F1]
4. **build-eps** (matrix, build-only — no GPU runners): at minimum `cargo check --no-default-features --features "cpu ort-defaults"`; optionally `cargo check --features coreml` (macos runner) / `cuda` (build-only) to catch EP feature breakage. EP runtime tests are out of scope (no accelerators in CI).
5. **msrv** (optional): pin `rust-version` in `Cargo.toml` and add a job on that toolchain. [A6-F8]

Note: `--all-features` will fail to *link* if conflicting EP backends are enabled together; prefer per-feature `cargo check`/`build` over a single `--all-features` run. Verify which EP feature combos are mutually exclusive (assumption — confirm against `ort` docs / A2 lane).

### Regression guards mapped to other lanes' improvements
| Lane / improvement | Guard test (where) |
|---|---|
| A1 language-lock fix (re-detect / re-prompt / reset semantics) | `reset()` field-contract unit test + multilingual code-switch golden (Tier 2) [A6-F2] |
| A1/A2 streaming chunk-boundary changes | streaming==offline equivalence golden (Tier 2) [A6-F6] |
| A2 mel/realfft perf refactor | mel reference-value test (Tier 1) [A6-F4] |
| A3 API/error refactor, variant unification | per-variant offline golden + `set_target_lang` error contract |
| A4 timestamp/diarization features | extend `timestamps.rs` tests; add sortformer diarization tests [A6-F7] |
| A5 shared-abstraction extraction across model_X pairs | CTC/TDT pure-logic tests pin behavior before/after refactor [A6-F3] |

---

## Cross-references for other lanes

- **A1 (correctness/streaming):** Whatever you decide for the language-lock (re-detect API, re-prompt at silence, `reset()` semantics, or "model-inherent, document it"), it MUST come with a testable contract. A6 will need: the exact post-fix behavior for `reset()` (which fields cleared vs preserved, `nemotron.rs:495-509`) and the expected multilingual code-switch outcome, to encode as the golden test (A6-F2, A6-F6). Tell me if the fix is library-side (testable) or model-inherent (document-only).
- **A2 (performance):** If you extract/refactor the mel path (`audio.rs:206-266`) or realfft usage, the mel reference test (A6-F4) is your regression guard — coordinate on tolerances. Also confirm which `ort` EP feature flags are mutually exclusive so the CI matrix uses per-feature `cargo check` rather than `--all-features` (A6-F1, §3).
- **A3 (API/ergonomics):** `decode_with_beam_search` is a silent greedy stub (`decoder.rs:200-206`, A6-F9) — decide remove/document/implement. Also: if you unify the 10 variants behind a shared trait, that trait is the natural seam for per-variant golden integration tests (A6-F6).
- **A4 (models/features):** Diarization (`sortformer.rs`) and timestamps are the least-tested feature areas; the current sortformer tests only re-test STFT (A6-F7). Any new timestamp/diarization feature needs pure-logic tests where extractable.
- **A5 (code quality/structure):** Two STFT implementations exist (`audio.rs` and `Sortformer::stft`) with duplicated tests (A6-F7) — if you dedupe the impl, consolidate the tests too. If you extract shared encoder/decoder abstractions across the `model_X` pairs, do it AFTER the Tier-1 CTC/TDT pure-logic tests land so the refactor is guarded (A6-F3).
- **RU (upstream):** I need the upstream reference behavior for multilingual reset/re-detection to write a correct golden assertion (does NeMo re-detect per buffer?). The golden test should encode the *intended* behavior, not just the current locked behavior.
- **Context-pack correction for orchestrator:** Context pack line 57 says "NO CI" — this is **stale**. CI exists at `.github/workflows/rust.yml` but is minimal (build+test, default features only). Findings are framed as "improve the existing CI," not "add CI from scratch."
