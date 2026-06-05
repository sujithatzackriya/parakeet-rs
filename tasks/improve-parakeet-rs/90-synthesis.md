# 90 - Synthesis: reconciled roadmap + ordered one-PR task backlog (Wave S, lane 90)

**/think frameworks:** dialectic/synthesis (reconcile Wave-R/A theses against Wave-V antitheses into one ranked roadmap; Wave V is authoritative where it corrects) + theory-of-constraints (find the few load-bearing constraints - goldens, un-export, argmax-reconcile, additive-before-breaking - and sequence the whole backlog around them so no PR is built before its precondition exists). PLAN ONLY. No loop.

This lane consumes the merged 31-finding master list + dependency graph (`80-feasibility.md`), the five reconciled verdicts (`85-verify.md` + `85-verify-v1..v5`), and the harvested ledger. It produces: (1) a severity-ranked roadmap grouped by epic, each finding tagged with its Wave-V status; (2) the ordered task backlog (one worktree + one PR each); (3) the PR-wave sequencing with quick-wins vs deep flagged; (4) surviving risks + the open questions that need a human decision.

Wave V is AUTHORITATIVE where it corrects Wave R/A. The corrections folded in below: the mel-accuracy epic is DROPPED (V3); F1 is reclassified out of safe-perf (V2); consolidation is behavior-RECONCILING at ~350-550 lines split into 5 PRs (V4); the Mutex is necessary - concurrency is a docs deliverable (V5); the language epic is scoped to caller-driven boundaries (V1).

---

## Part A - Reconciled, severity-ranked roadmap (grouped by epic)

Each finding carries its merged id (M-/G-), reconciled severity, and **Wave-V status**: SURVIVED (verified, build as-stated), SCOPED (survives with a narrower boundary V drew), REFUTED (do not build / do not escalate), or N/A-V (not a load-bearing claim V examined; carried from 80 at audit severity). The four load-bearing ordering rules are stated once in Part B.

### EPIC L - Language / multilingual (the trigger). Owner correctness+model+API surface.

- **M1a detected_language() [CRITICAL epic, additive step] - V status: SURVIVED.** Surface the stripped in-band `<lang>` SentencePiece tags as a getter. Tags are real ids kept in `accumulated_tokens`, stripped only at render (`nemotron.rs:518,716`), so derivable with zero risk. Quick-win.
- **M1b reset_with_lang + per-chunk re-prompt [CRITICAL epic, breaking step] - V status: SCOPED.** Mechanism verified: prompt is a post-encoder `prompt_kernel` MLP head (`model_nemotron.rs:149-195`; export `:334-359`), encoder cache is produced BEFORE the prompt so it is language-agnostic and re-prompting next chunk does not corrupt it; `prompt_index` is a real per-call ONNX input (`nemotron.rs:489,591,691`). The SCOPE V drew: Nemotron emits NO VAD/LID boundary signal, so re-detection must be **caller-supplied boundary**, not autonomous. Atomically reset the carried decoder state (`last_token`/joint/LSTM) at the boundary while PRESERVING the encoder cache; fix the `set_target_lang`-needs-reset semantics. Deep.
- **M21/is_lang_tag hardening [LOW -> precondition for M1] - V status: SURVIVED.** `is_lang_tag` (`nemotron.rs:78-93`) is a string-shape heuristic that can over/under-strip; harden to **exact token-id membership FIRST** before any control-flow use (it gates both detected_language and re-prompt). Folds into M1a.
- **M1c auto-continuous re-detect [optional, LAST] - V status: SCOPED (deferred).** Requires an external boundary signal Nemotron lacks; document as **caller-VAD-driven**. Largest, optional, last.
- **Documented model-inherent limit:** sub-sentence / mid-word code-switch is **model-inherently unsupported** (NVIDIA upstream picks the dominant language per utterance). The roadmap must NOT promise it.

### EPIC C - Correctness & streaming. Owner A1.

- **M6 Nemotron flush() + offline/streaming length reconcile [HIGH] - V status: N/A-V (audit-solid).** Streaming drops the final partial chunk (`nemotron.rs:637-639`, ~560ms lost/utterance); offline uses true length (`:581`) vs streaming constant (`:689`) so offline cannot be the streaming oracle. Add additive `flush()` on the offline length convention. Pairs with the streaming==offline equivalence golden. Additive.
- **M7 EOU processed-sample cursor + chunk validation [HIGH] - V status: N/A-V.** No cursor, assumes exact 16-frame chunks, no validation (`parakeet_eou.rs:147-165,174`); token dup/loss on off-size chunks. Interim low-risk: enforce + reject the required chunk size; full fix adds a cursor.
- **M5 reconcile argmax to one first-wins+finite helper [HIGH] - V status: SURVIVED (and confirmed behavior-CHANGING).** Three divergent tie/NaN semantics: Nemotron first-wins/NaN->0 (`nemotron.rs:749-756`), Unified/CTC last-wins/NaN->Equal (`parakeet_unified.rs:477-482`, `decoder.rs:49-53,137-141`), EOU finite-guarded (`parakeet_eou.rs:208-215`), TDT last-wins. V4 confirms unifying CHANGES Nemotron/Unified/TDT output on ties/NaN - so M5 is an isolated behavior change with per-variant token regression tests FIRST, and it is the prerequisite that must land before M2's loop collapse.
- **M20 EOU soft-reset asymmetry + `<EOU>`-then-token loss [LOW] - N/A-V.** Specialized path; soft-reset intentional. `parakeet_eou.rs:217-255`. Document.
- **M22 Unified flush zero-pads right context [LOW] - N/A-V.** Standard streaming tradeoff; document.
- **M23 `max_symbols_per_step` caps differ 10/5/10, silent truncation [LOW] - N/A-V.** Unify + log on cap-hit.
- **M30 mel skips normalization for `num_frames <= 1` [LOW] - N/A-V.** Sub-30ms edge; document or guard.

### EPIC P - Performance & RTF. Owner A2.

- **F2 cached FFT plan for Nemotron/EOU/Multitalker [HIGH-leverage, low-risk] - V status: SURVIVED, bit-identical.** `stft` already delegates to `stft_with_plan` (`audio.rs:74-83`); Nemotron(781)/Multitalker(672)/EOU(259) bypass it and rebuild the planner per call. Routing them through the cached plan is byte-identical. Lands early/independent. Quick-win.
- **M4/F3 TensorRef zero-copy inputs/outputs [HIGH] - V status: SURVIVED, conditional.** `from_array` silently copies; `from_array_view` ERRORS on non-contiguous input. Equal only for C-contiguous input. Must add a **standard-layout assert + golden across all 4 `model_*.rs`**. ~7.7MB/chunk saved on encoder cache; LSTM-state clones up to ~560x/chunk removed. Best delivered inside the shared backend (M2) so it is one change not four. `model_cohere.rs:111,160-194` is the zero-copy template.
- **M3/F1 incremental mel / frame-accurate ring [HIGH perf, RECLASSIFIED] - V status: REFUTED as pure perf -> correctness-sensitive.** Current code recomputes mel over the whole growing buffer every chunk (`nemotron.rs:617,628,633-639`; `multitalker.rs:326,334`), O(n^2). V2: this is a STREAMING-CORRECTNESS change (preemphasis is a cross-sample filter `audio.rs:57-58`; `stft_with_plan` always prepends `n_fft/2=256` zeros `:94-97`; must match `PRE_ENCODE_CACHE=9` overlap; today's trim `nemotron.rs:705-712` is itself seam-bearing so an idealized continuous mel would DIVERGE). Requires an executed **byte-equal golden experiment**, sequence AFTER fixtures. **Defer if low ROI** - it only matters for long single streams. Embedded MEDIUM correctness sub-case: the `audio_processed` cursor can desync (`nemotron.rs:705-712`).
- **M18 concurrency model [MEDIUM] - V status: REFUTED (rewrite the recommendation).** ort `2.0.0-rc.12` `Session::run*` all take `&mut self` (`session/mod.rs:212,253,340,407`); the **Mutex is NECESSARY, not needless** - "drop the Mutex and call run(&self)" does not compile. Deliverable is **M18a DOCS**: `spawn_blocking` + bounded concurrency (tokio blocking pool up to 512 vs ~4 intra-op threads) + per-stream task pinning (chunk order = decode-state order). Record **one-Session-per-stream (N x 2.3GB)** as the real parallelism option (a different design, memory cost). M18b (`&self` runs) is re-opened ONLY if ort is bumped to expose `&self` run.
- **M19 EP/GPU discoverability [MEDIUM] - N/A-V.** Accelerators cfg-gated, never auto-discovered, CPU-only default (`execution.rs:16-36`). Document the EP matrix + CoreML-slower caveat; add `compiled_providers()` + opt-in `Auto`. Largest available speed lever but gated on hardware + opt-in.
- **M24 CTC timestamps re-decode every prefix, O(n^2) [LOW] - N/A-V.** Offline path only.

### EPIC A - API & ergonomics. Owner A3.

- **M8 un-export the model_X low-level types [HIGH - keystone] - V status: SURVIVED (with the SentencePieceVocab correction).** `lib.rs:86-91` publicly exports `ParakeetModel`, `ParakeetEOUModel`, `NemotronEncoderCache`/`NemotronModel`/`NemotronModelConfig`, `ParakeetUnifiedModel`/`UnifiedModelConfig`, `decoder::ParakeetDecoder`, and `SentencePieceVocab`. The model_X types have ZERO references in examples/README (V4) - removing them is the only public break and it unblocks `pub(crate)`. **CORRECTION (V4 + G-E):** `SentencePieceVocab` is used cross-module by `multitalker.rs` + `parakeet_unified.rs` - un-export it but **keep it `pub(crate)` and move to a `vocab/` home**; do NOT privatize to nemotron.rs or the build breaks. This is the one deliberate 0.x break that converts M2 from breaking to internal.
- **M10 StreamingTranscriber trait + uniform reset + EOU rename [HIGH] - N/A-V.** 5 streaming variants share NO trait, use 5 verbs (`transcribe`/`transcribe_chunk`/`feed`/`diarize_chunk`); reset spelled 4 ways; EOU has no public reset; Nemotron reset silently preserves language (= API half of M1). Trait is additive; the rename + reset-semantics change are breaking (deprecate-alias).
- **M13 thiserror + structured `#[non_exhaustive]` Error [MEDIUM] - N/A-V.** Keep the concrete enum (NOT eyre); `source()` returns None today, `.map_err(Error::Model(format!))` dozens of times. Breaking but cheap pre-1.0; do it before M2 deletes the boilerplate so M2 lands on the new type.
- **M14 constructor/config unification + triple-ModelConfig rename + WordTimestamp/TimedToken dedup [MEDIUM] - N/A-V.** Param drift (`config` vs `exec_config`); handle on 3 of 8; Sortformer exposes internal `ModelConfig`; three structs named `ModelConfig`; `WordTimestamp` cannot feed `process_timestamps`.
- **M15 Language newtype with Auto first-class [MEDIUM] - N/A-V.** Two variants disagree on codes; no `Auto` type. `Language` enum with `Other(&str)` escape hatch; `Auto` first-class (type-system half of M1). Additive if a `&str` convenience delegates.
- **M25 decode_with_beam_search is a public no-op [LOW] - N/A-V.** Ignores `beam_width` (`decoder.rs:199-206`). Remove / `#[deprecated]` / implement.
- **M26 `pub use transcriber::*` glob export [NIT] - N/A-V.** Semver hazard; make explicit.
- **A3-11 / M-doc crate-root rustdoc quick-start will NOT compile [NIT] - N/A-V.** Uses the OLD single-arg `from_pretrained`. Fix the doc example so `cargo test --doc` passes. Quick-win.
- **M29 `from_pretrained` silent auto-detect, no precision selector [NIT] - N/A-V.** Folds into M9.

### EPIC M - Model coverage & features. Owner A4.

- **M9 shared resolve_onnx_file + prefer:Quantization knob [HIGH] - N/A-V.** `find_encoder`/`find_decoder_joint` candidate lists diverge 3-4x; README advertises int8/int4 Nemotron files the loader cannot pick up (`README.md:155`, `model_nemotron.rs:72`); int8-first vs int8-last priority means two variants behave oppositely. One `resolve_onnx_file(dir, role, prefer)` + `prefer: Quantization`. Changing precedence can silently switch which file loads -> gate or document as a break.
- **M16 word timestamps on streaming Nemotron/EOU [MEDIUM] - N/A-V.** Graph supports it; `(token_id, frame)` accumulation exists in `parakeet_unified.rs:442-512` + `multitalker.rs:593-663`, missing in `nemotron.rs:723-770`. Reuse `timestamps::group_by_words`; document Words mode for multilingual (sentence split keys on Latin punctuation). Additive, low-risk.
- **M17 Cohere timestamps/diarize toggles [MEDIUM] - N/A-V.** Hardcodes `<|notimestamp|>`/`<|nodiarize|>` (`cohere.rs:308-309`) though the model supports both. Add a `CohereOptions` builder (avoids a 6-arg signature).
- **M31 Multitalker ASR-chunk vs Sortformer-stride mismatch [LOW] - N/A-V.** Nearest-neighbour speaker masks; documented known limitation.

### EPIC X - Code quality / consolidation. Owner A5 + 02.

- **M2 extract pub(crate) RnntBackend / src/onnx/ [CRITICAL structural - V status: REFUTED as stated, SPLIT].** V4: real behavior-safe net deletion is **~350-550 lines, NOT 700-900** (the figure counted gross duplicated text; after a ~60-line shared helper + thin call sites the four `run_decoder` bodies (~209L) collapse to ~150 net). The "4x greedy loop" collapses: only Nemotron + Unified share a body; EOU (`<EOU>`/soft-reset/overflow) and TDT (duration-skip) do NOT merge. It is **behavior-RECONCILING not preserving** (carries M5's argmax change). **Split into 5 PRs: M8 -> M5 -> M2a (session/find-encoder dedup) -> M2-decoder (parameterized helper, >=4 config knobs) -> M2b (mel front-end, numerics unchanged, golden-gated).** Each part is golden-gated.
- **M12 mel-accuracy epic [was MEDIUM, V status: REFUTED - DROP].** V3 is decisive and the **audit was BACKWARDS**: (a) A1-09 Hann is already SYMMETRIC and so is NeMo (`torch.hann_window(periodic=False)`) - implementing the proposed fix would INTRODUCE divergence, change NO code; (b) A1-10/Q2 `x.max(0.0)` log floor is a mathematical no-op (mel values are always >= 0); (c) Nemotron "no normalization" matches multilingual (`"normalize":"NA"`, model demonstrably coherent). All secondary params MATCH NeMo (n_fft=512, win=400, hop=160, n_mels=128, preemph=0.97, log guard 2^-24, Slaney mel [EOU intentionally HTK], dither omitted). **DROP M12 as an accuracy epic.** Keep ONLY an OPTIONAL EN normalize-dump verification fixture (EN export hardcodes `"per_feature"` in a dumped dict literal, ships no config.json - low-confidence, works in practice; the fixture is verification only, no code change unless it proves a mismatch). The mel-constant consolidation survives only inside M2b (numerics unchanged). **Implementor note: do NOT touch the Hann window, log floor, or normalization to "fix accuracy" - the only genuine multilingual accuracy problem is the language-lock (Epic L).**
- **M27 files over the 800-line ceiling [NIT-MEDIUM] - N/A-V.** `sortformer.rs` 1254, `nemotron.rs` 786, `multitalker.rs` 771; god-functions `model_cohere::run_decoder_step` 119L, `model_tdt::greedy_decode` 116L. Style; partially relieved by M2.
- **Positive baseline (do NOT "fix"):** all 8 lock sites use `.lock().map_err` (no poisoning panics); zero `panic!`/`todo!`/`unimplemented!`; no `#[allow(dead_code)]`.

### EPIC T - Tests & CI (the precondition). Owner A6.

- **M11a Tier-1 pure no-model tests + committed WAV fixture + feature-matrix CI [HIGH - Tier 0 precondition] - N/A-V (every lane's reverification depends on it).** `.github/workflows/rust.yml:21-24` builds default features ubuntu-only, so `sortformer`/`multitalker`/`cohere` + the cohere example **can ship broken green**. Tier-1 pure tests: CTC collapse, TDT frame-advance, `is_lang_tag` (exact-id), mel reference, `reset()` contract, argmax. The `reset()` contract + code-switch golden are the guards that would have caught the language-lock. `test_en.wav` exists locally but is NOT git-tracked -> commit a small WAV fixture (A6-F5).
- **M11b Tier-2 env-gated goldens [HIGH] - N/A-V.** `#[ignore]`/env-gated goldens vs `./nemotron`/`./nemotron_multi`: offline golden, streaming==offline equivalence, multilingual code-switch. These are the byte-equal guards M2/M3/M4 require.
- **M28 CI fmt/clippy/MSRV [LOW] - N/A-V.** Folds into the CI matrix PR.

### GAPS the whole audit missed (from 80 section 6).

- **G-A criterion benchmark harness [MEDIUM].** No `benches/`; every perf claim (M3/M4) and the RTF success metric (PRD G2 "RTF < 1.0 on CPU") is unverifiable without one. Add `benches/` criterion harness for `transcribe_chunk` latency vs buffer length + RTF per variant. **The missing acceptance instrument for Epic P.**
- **G-B migration guide / CHANGELOG [LOW-MEDIUM].** The roadmap spends several deliberate breaks (M8/M9/M10/M13/M15). Batch them into ONE version bump (a 0.4.0) with before/after snippets. Satisfies SM-1 (zero UNDOCUMENTED breaks).
- **G-C model-format/export compatibility contract [LOW].** Crate hardcodes ONNX I/O names + shapes; if NVIDIA re-exports, the loader breaks with a stringly `Error::Model`. Validate the encoder/decoder input-output name set at load, emit a structured "model format mismatch" error. Folds into M13.
- **G-D wasm/webgpu/nnapi `cargo check` in the matrix [LOW].** Flags exist but are unverified to even compile. Add `cargo check` for `webgpu`/`nnapi`/`wasm32` to the CI matrix (extends M11a's matrix) so the advertised surface does not rot.
- **G-E SentencePieceVocab export [LOW].** Folded into M8 (keep `pub(crate)`, move to `vocab/`).
- **G-F coverage measurement [LOW, optional].** `cargo-llvm-cov` CI step so SM-5 has an instrument.

---

## Part B - The four load-bearing ordering rules (non-negotiable)

1. **Goldens before refactors.** M11a (Tier-1) + M11b (Tier-2 goldens) before M2 / M3 / M4 / M2b. Every perf and consolidation step is correctness-preserving ONLY under a byte-equal transcript / mel guard. This is why Tests is Tier 0, not a follow-up.
2. **Un-export before consolidate.** M8 before M2. After M8 the model_X types are gone and M2 is internal; before M8, M2 is a break. M8 is the one deliberate break the roadmap spends. Do NOT bury M8 despite its small size.
3. **Additive language before breaking language.** M1a (detected_language) before M1b (reset_with_lang / re-prompt). You cannot auto-re-decide a language without first observing it; the reset-semantics change must come after the code-switch golden exists.
4. **Reconcile argmax before collapse loop.** M5 before the M2-decoder merge. The divergent argmax must be unified to ONE behavior (isolated, with per-variant token regression tests first) or the refactor silently picks one variant's tie-break for all.

---

## Part C - Ordered task backlog (one worktree + one PR each)

Each task: id, title, goal, files, invariants/success criteria, deps, size, known unknowns. Sizes: S (<~150 LOC + tests, < a day), M, L (multi-session). Library quality gates referenced: **semver** (additive vs breaking, tagged), **streaming-correctness** (reset/flush/state-carry), **accuracy** (byte-equal vs golden/reference), **build-matrix** (all EP flags + features), **golden** (Tier-2 fixture).

### WAVE 0 - Foundation (Tier 0). Nothing downstream is safe before these.

**T01 - CI feature-matrix + fmt/clippy + cargo check wasm/webgpu/nnapi** (M11a-CI, M28, G-D)
- Goal: every gated feature + EP flag compiles in CI; fmt + clippy gates; the advertised surface cannot rot.
- Files: `.github/workflows/rust.yml`, `Cargo.toml` (feature list).
- Invariants/success: build-matrix gate green for `sortformer`/`multitalker`/`cohere` + the cohere example + each EP flag; `cargo check --target wasm32-*` for webgpu/nnapi; `cargo fmt --check` + `cargo clippy -D warnings` pass.
- Deps: none. Size: S. Quick-win (zero-risk, additive).
- Known unknowns: some EP flags may not even `cargo check` on the CI host without the EP toolchain - may need per-flag `cargo check` only (no link).

**T02 - Tier-1 pure no-model unit tests + committed WAV fixture** (M11a-tests, A6-F5)
- Goal: pure-function invariants guarded in CI with no model download.
- Files: new `tests/` (CTC collapse from `decoder.rs`, TDT frame-advance from `model_tdt.rs`, `is_lang_tag` from `nemotron.rs:78-93`, mel reference vs `audio.rs`, `reset()` contract per variant, `argmax`); commit a small `tests/fixtures/test_en.wav`.
- Invariants/success: tests prove real invariants (CTC blank-collapse, reset clears the documented fields, argmax tie/finite behavior); fixture is git-tracked + small; runs in CI default features.
- Deps: T01 (matrix exists). Size: M. Quick-win (additive).
- Known unknowns: exact mel reference values - may need to dump from a trusted run as the golden seed.

**T03 - Tier-2 env-gated golden harness vs local models** (M11b)
- Goal: the byte-equal guards that M2/M3/M4 require, plus the code-switch + reset() guards that catch the language-lock.
- Files: new `tests/golden_*.rs` (`#[ignore]`/env-gated on `./nemotron`, `./nemotron_multi`): offline golden transcript, streaming==offline equivalence, multilingual code-switch, reset() contract.
- Invariants/success: golden (byte-equal transcript per variant), streaming-correctness (streaming==offline within tolerance), accuracy (code-switch test currently FAILS = reproduces the lock, becomes the M1b acceptance test).
- Deps: T02 (fixture). Size: M. Deep (needs local models, gated).
- Known unknowns: streaming==offline may NOT be byte-equal today (M6 length divergence) - the test may assert a tolerance, then tighten after M6.

### WAVE 1 - Keystone breaks + quick wins (Tier 0/1, parallelizable after Wave 0).

**T04 - Un-export model_X low-level types; relocate SentencePieceVocab to vocab/** (M8, G-E)
- Goal: remove the accidental public low-level surface; the one deliberate 0.x break that makes M2 internal.
- Files: `lib.rs:86-91` (remove the re-exports), new `src/vocab/` home for `SentencePieceVocab` (keep `pub(crate)`; used by `multitalker.rs`, `parakeet_unified.rs`).
- Invariants/success: semver (BREAKING, tagged; examples/README still compile since they reference none of the removed types - verify); build-matrix green; `SentencePieceVocab` stays `pub(crate)` not privatized.
- Deps: Wave 0. Size: S. Keystone (gates M2).
- Known unknowns: confirm no other crate-internal path relies on the public visibility.

**T05 - detected_language() + harden is_lang_tag to exact-id** (M1a, M21)
- Goal: expose the stripped `<lang>` tags as a getter; make tag detection exact-id so it is safe for control flow.
- Files: `nemotron.rs:78-93` (is_lang_tag -> exact-id membership), `nemotron.rs:511-521,716` (surface tags before stripping), public getter on `Nemotron`/`NemotronHandle`.
- Invariants/success: semver (ADDITIVE); detected_language returns the last committed `<lang>` under `auto`; exact-id strip has a Tier-1 unit test.
- Deps: Wave 0 (for the unit test). Size: S. Quick-win (additive, zero-risk).
- Known unknowns: behavior when no tag has been emitted yet (return None).

**T06 - Fix crate-root rustdoc quick-start example** (A3-11)
- Goal: the doc example compiles under `cargo test --doc`.
- Files: crate-root `//!` rustdoc in `lib.rs` (uses the OLD single-arg `from_pretrained`).
- Invariants/success: `cargo test --doc` green; example matches the current `from_pretrained` signature.
- Deps: none (can run anytime, but T01 makes it CI-enforced). Size: S. Quick-win.
- Known unknowns: none.

**T07 - F2: route Nemotron/EOU/Multitalker through the cached FFT plan** (F2)
- Goal: stop rebuilding the realfft planner per stft call; bit-identical.
- Files: `nemotron.rs:781`, `multitalker.rs:672`, `parakeet_eou.rs:259` (call `stft_with_plan` via `FeatureCache` instead of `stft`).
- Invariants/success: accuracy (byte-identical mel - V2 confirms; golden-mel from T02 guards it); build-matrix green.
- Deps: T02/T03 (golden). Size: S. Quick-win (V-confirmed bit-identical).
- Known unknowns: none (V2 settled it).

**T08 - thiserror + structured #[non_exhaustive] Error; model-format-mismatch variant** (M13, G-C)
- Goal: structured error enum with source chains; validate ONNX I/O name set at load.
- Files: `error.rs` (thiserror derive, `#[non_exhaustive]`, structured variants, preserve `source()`), `model_*.rs` load paths (replace `.map_err(Error::Model(format!))`; add a "model format mismatch" error on missing I/O names).
- Invariants/success: semver (BREAKING - enum shape; tagged + in migration guide); `Error::source()` chains; build-matrix green.
- Deps: Wave 0; land before M2 so the consolidation deletes boilerplate onto the new type. Size: M. Quick-win-ish (mechanical but breaking).
- Known unknowns: how many call sites; whether any external code matches on the enum (0.x, acceptable).

**T09 - Nemotron flush() + reconcile offline/streaming length** (M6)
- Goal: stop dropping the final partial chunk; make offline a valid streaming oracle.
- Files: `nemotron.rs:637-639` (flush path), `:581` vs `:689` (length convention).
- Invariants/success: streaming-correctness (no lost final ~560ms); after this, T03's streaming==offline equivalence tightens to byte-equal; semver (ADDITIVE - new flush()).
- Deps: T03 (equivalence golden to validate). Size: M. Additive.
- Known unknowns: whether matching offline length convention shifts any existing transcript (golden catches it).

### WAVE 2 - Consolidation chain (Tier 1, strictly ordered). Each golden-gated.

**T10 - M5: reconcile argmax to one shared first-wins + finite helper** (M5)
- Goal: ONE `argmax(&[f32]) -> usize` (first-wins + finite guard, matching EOU/NeMo); replace 4 divergent sites.
- Files: new shared helper; `nemotron.rs:749-756`, `parakeet_unified.rs:477-482`, `decoder.rs:49-53,137-141`, `parakeet_eou.rs:208-215`, TDT site.
- Invariants/success: this is the ONE deliberate behavior change (ties/NaN on Nemotron/Unified/TDT) - isolated for bisect, **per-variant token regression tests FIRST** (Tier-1), then the change; golden transcripts unchanged on real audio (ties/NaN are rare).
- Deps: T02/T03; MUST land before T12. Size: S-M. Deep (behavior-changing, isolated).
- Known unknowns: whether any real golden transcript shifts (expected: no, ties are pathological).

**T11 - M2a: extract session-build + find_encoder/resolve_onnx_file dedup into src/onnx/** (M2a, folds M9)
- Goal: the mechanical, numerics-free dedup; one `resolve_onnx_file(dir, role, prefer)` + `prefer: Quantization`.
- Files: new `src/onnx/`; the 6-8 verbatim session-build sites; `find_encoder`/`find_decoder_joint` across `model_*.rs`; `model_nemotron.rs:72` + `README.md:155` (int8/int4 pickup).
- Invariants/success: semver (internal after T04; the precedence change in resolve is a documented break - gate or tag); build-matrix green; goldens unchanged (no numerics).
- Deps: T04 (un-export), T08 (error shape), Wave 0 goldens. Size: M. Deep.
- Known unknowns: int8-first vs int8-last precedence - pick one, document the break (M9).

**T12 - M2-decoder: parameterized shared RNNT run_decoder helper (>=4 knobs); TensorRef zero-copy** (M2-decoder, M4/F3)
- Goal: collapse the Nemotron+Unified run_decoder bodies into one parameterized helper; migrate inputs to `TensorRef::from_array_view` with a standard-layout assert.
- Files: `model_nemotron.rs:232-287`, `model_unified.rs:134-186` (merge); inputs across the 4 `model_*.rs`; template is `model_cohere.rs:111,160-194`. EOU/TDT do NOT merge (V4) - leave them.
- Invariants/success: accuracy (byte-equal golden across all 4 model_*.rs - the F3 gate); standard-layout assert on every migrated input; M5's argmax already reconciled (dep); semver internal.
- Deps: T10 (M5), T11 (M2a), Wave 0 goldens. Size: L. Deep.
- Known unknowns: the >=4 config knobs (input names, axis order, dtype, cache layout) - V4 says parameterizable; confirm during build.

**T13 - M2b: shared mel front-end + single constants source (numerics UNCHANGED)** (M2b)
- Goal: one mel front-end, single constants source; byte-identical output. NOT an accuracy change (M12 dropped).
- Files: consolidate the 3-4 mel paths (`audio.rs`, `nemotron.rs:784`, `multitalker.rs:675`, `parakeet_eou.rs:261`) to one front-end; fold the `x.max(0.0)` cosmetic drift (no-op).
- Invariants/success: accuracy (golden-mel from T02 byte-equal); **do NOT change Hann/log-floor/normalization** (V3 - they already match NeMo); semver internal.
- Deps: T11/T12, Wave 0 golden-mel. Size: M. Deep (golden-gated).
- Known unknowns: EOU intentionally uses HTK mel (not Slaney) - keep that branch.

### WAVE 3 - Language breaking step + API surface (Tier 2).

**T14 - M1b: reset_with_lang + per-chunk re-prompt at caller-supplied boundary** (M1b)
- Goal: allow language change mid-stream at a caller-supplied boundary; atomically reset carried decoder state while preserving the (language-agnostic) encoder cache.
- Files: `nemotron.rs:475-491` (set_target_lang semantics), `:495-509` (reset preserves prompt today - add reset_with_lang), `:591,691` (per-chunk prompt_index), the carried `last_token`/joint/LSTM state.
- Invariants/success: accuracy (T03 code-switch golden goes from FAIL -> PASS = the SM-3 acceptance); streaming-correctness (encoder cache preserved, decoder state reset cleanly); semver (BREAKING reset semantics - deprecate-alias + migration guide); document mid-word code-switch as UNSUPPORTED and auto-continuous as caller-VAD-driven.
- Deps: T05 (must observe before re-deciding), T03 (code-switch golden). Size: L. Deep.
- Known unknowns: [V1 residual] whether resetting joint/LSTM while preserving the cache yields a clean continuation in PRACTICE (V verified the mechanism by code+infer, not a run) - the golden settles it. Cold-start (chunk-0 zero cache, `nemotron.rs:562-569`) may still mis-commit; fix where feasible (e.g. require/await a minimum context before first commit under auto).

**T15 - M10: StreamingTranscriber trait + uniform reset + EOU rename/public reset** (M10)
- Goal: one streaming trait across the 5 variants; uniform reset; EOU gets a public reset.
- Files: new trait; the 5 stream methods (`transcribe`/`transcribe_chunk`/`feed`/`diarize_chunk`); reset across variants; EOU public reset; include `flush()` (from T09) in the trait.
- Invariants/success: semver (trait ADDITIVE; rename + reset-semantics BREAKING with deprecate-alias); build-matrix green; trait object-safe or documented why not.
- Deps: T09 (flush in trait), T04/T11 (surface settled). Size: M. Deep.
- Known unknowns: whether all 5 fit one trait cleanly (diarize_chunk returns speakers, not text) - may need an associated Output type.

**T16 - M16: word timestamps on streaming Nemotron/EOU** (M16)
- Goal: emit word timestamps where the graph supports it; reuse existing accumulation.
- Files: `nemotron.rs:723-770` (add `(token_id, frame)` accumulation copied from `parakeet_unified.rs:442-512`); `timestamps::group_by_words`.
- Invariants/success: semver (ADDITIVE); document Words mode for multilingual (sentence split keys on Latin punctuation); golden timestamp alignment.
- Deps: T12 (shared timed-token accumulation lands in M2). Size: M. Additive.
- Known unknowns: multilingual word grouping accuracy on non-Latin scripts.

### WAVE 4 - Remaining API/feature polish (Tier 3) + perf instrument + docs.

**T17 - M15: Language newtype with Auto first-class** (M15)
- Goal: typed `Language` enum with `Auto` + `Other(&str)`; reconcile the two divergent code tables.
- Files: new `Language` type; `set_target_lang` + detected_language signatures; the two variants disagreeing on codes.
- Invariants/success: semver (ADDITIVE if a `&str` convenience delegates; the disagreement reconcile is BREAKING - tag); ties into T05/T14 language flow.
- Deps: T14, T15. Size: M.
- Known unknowns: the canonical code table (BCP-47 vs NeMo ids).

**T18 - M14: constructor/config unification + triple-ModelConfig rename + WordTimestamp/TimedToken dedup** (M14)
- Goal: uniform constructor params; rename the three `ModelConfig` collisions; let `WordTimestamp` feed `process_timestamps`.
- Files: constructors across variants; the three `ModelConfig` structs; `WordTimestamp`/`TimedToken`; Sortformer's exposed `ModelConfig`.
- Invariants/success: semver (BREAKING renames - tag + migration guide); build-matrix green.
- Deps: T15. Size: M.
- Known unknowns: blast radius of the renames across examples.

**T19 - M17: Cohere timestamps/diarize toggles via CohereOptions** (M17)
- Goal: expose the model's timestamp + diarize support behind a builder.
- Files: `cohere.rs:308-309` (hardcoded `<|notimestamp|>`/`<|nodiarize|>`); new `CohereOptions`.
- Invariants/success: semver (ADDITIVE builder, defaults preserve current behavior); cohere example + feature compile in matrix.
- Deps: T08/T18 (options pattern). Size: S.
- Known unknowns: whether both toggles work end-to-end (needs the cohere model).

**T20 - M19: EP Auto + compiled_providers() + EP-matrix rustdoc** (M19)
- Goal: discoverability of compiled accelerators; opt-in `Auto`; document the EP matrix + CoreML-slower caveat.
- Files: `execution.rs:16-36`; rustdoc.
- Invariants/success: semver (ADDITIVE); `compiled_providers()` reflects cfg-gated features; no change to CPU default.
- Deps: none (can float earlier). Size: S. Quick-win-ish.
- Known unknowns: per-EP runtime verification is a NON-GOAL (PRD) - this is discoverability only.

**T21 - M18a: concurrency DOCS (spawn_blocking + bounded concurrency + one-session-per-stream)** (M18a)
- Goal: document the real concurrency model; do NOT remove the Mutex.
- Files: rustdoc on the streaming types + a README/doc section.
- Invariants/success: documents `spawn_blocking` + bounded concurrency (blocking pool up to 512 vs ~4 intra-op threads) + per-stream task pinning (chunk order = decode-state order); records one-Session-per-stream (N x 2.3GB) as the real parallelism option. NO code change to the Mutex (V5: it is necessary).
- Deps: none. Size: S. Quick-win (docs only).
- Known unknowns: M18b (`&self` runs) re-opens only if ort is bumped to expose `&self` run - explicitly out of scope now.

**T22 - G-A: criterion benchmark harness** (G-A)
- Goal: the acceptance instrument for Epic P; measure `transcribe_chunk` latency vs buffer length + RTF per variant.
- Files: new `benches/` (criterion).
- Invariants/success: produces RTF per variant so PRD G2 ("RTF < 1.0 on CPU") is verifiable; baselines the F2/M3/M4 wins.
- Deps: none (but most useful before/after T07/T12/T13). Size: M.
- Known unknowns: needs local models for realistic RTF (gate like T03).

**T23 - M3/F1: incremental mel / frame-accurate ring (DEFERRED unless ROI proven)** (M3/F1)
- Goal: O(n) streaming mel instead of O(n^2) over the growing buffer; fix the `audio_processed` cursor desync.
- Files: `nemotron.rs:617,628,633-639,705-712`; `multitalker.rs:326,334`; `audio.rs:57-58,94-97`.
- Invariants/success: accuracy (executed byte-equal golden vs current seam-bearing output - V2: today's trim is itself seam-bearing so an idealized continuous mel would DIVERGE; the golden must match CURRENT behavior, not an idealized one); streaming-correctness (cursor cannot desync).
- Deps: T03 goldens + T22 (to prove ROI). Size: L. Deep, **defer if low ROI** (only long single streams benefit; V2 reclassified it out of safe-perf).
- Known unknowns: whether a frame-accurate ring can be made byte-equal to the current trim/re-pad at all, or whether it necessarily changes the seam (then it is a deliberate, golden-rebaselined change).

### WAVE 5 - Release hygiene + optional/last.

**T24 - G-B: migration guide / CHANGELOG; batch the 0.x breaks into 0.4.0** (G-B)
- Goal: one version bump documenting every deliberate break with before/after snippets.
- Files: `CHANGELOG.md`, `Cargo.toml` version bump.
- Invariants/success: SM-1 (zero UNDOCUMENTED breaks); covers M8, M9, M10, M13, M14, M15, M1b reset semantics.
- Deps: all breaking PRs (T04, T08, T09 length, T11 precedence, T14, T15, T17, T18). Size: S.
- Known unknowns: whether to ship one 0.4.0 or stagger (see open question Q1).

**T25 - M7: EOU processed-sample cursor + chunk-size validation** (M7)
- Goal: stop token dup/loss on off-size chunks; add a cursor.
- Files: `parakeet_eou.rs:147-165,174`.
- Invariants/success: streaming-correctness (cursor tracks processed samples; off-size chunks rejected or handled); golden EOU stream stable.
- Deps: T03. Size: M. (Interim low-risk reject-off-size can ship earlier as part of T15's EOU work.)
- Known unknowns: cold-start uses real audio as pre-encode context - preserve.

**T26 - Tail polish: M23 (cap unify+log), M25 (beam-search no-op), M26 (glob export), M20/M22/M30/M31/M24 docs** (LOW/NIT batch)
- Goal: close the LOW/NIT items in one or two small PRs.
- Files: `decoder.rs:199-206` (M25 - remove/`#[deprecated]`/implement), `transcriber` glob (M26), the cap sites (M23), doc the known limitations (M20/M22/M30/M31/M24).
- Invariants/success: semver (M25 removal/deprecate tagged; M26 explicit re-export); no behavior change beyond M23 cap logging.
- Deps: none. Size: S each. Quick-win.
- Known unknowns: whether to implement or delete beam search (likely delete/deprecate - it is a public no-op).

**T27 - M1c: auto-continuous re-detect (OPTIONAL, LAST)** (M1c)
- Goal: continuous re-detection driven by a caller VAD boundary.
- Files: builds on T14's reset_with_lang.
- Invariants/success: documented as caller-VAD-driven (Nemotron emits no boundary signal); additive API.
- Deps: T14. Size: L. Optional, last. **Needs human decision (Q3) on whether to build at all in v1.**

**T28 (optional) - G-F: coverage measurement (cargo-llvm-cov) + M28 MSRV** (G-F, M28)
- Goal: instrument SM-5; pin MSRV.
- Files: CI. Deps: T01. Size: S. Optional.

---

## Part D - PR-wave sequencing (recommended build order)

```
WAVE 0  (foundation - blocks everything)         T01 CI-matrix | T02 Tier-1+fixture | T03 Tier-2 goldens
WAVE 1  (keystone + quick-wins, parallel)         T04 un-export | T05 detected_language | T06 doc-fix
                                                  T07 F2 | T08 thiserror | T09 flush
WAVE 2  (consolidation chain, STRICT order)        T10 M5 -> T11 M2a -> T12 M2-decoder+F3 -> T13 M2b
WAVE 3  (language break + surface)                T14 reset_with_lang | T15 StreamingTranscriber | T16 word-ts
WAVE 4  (polish + perf instrument + docs)         T17 Language | T18 config | T19 Cohere | T20 EP
                                                  T21 concurrency-docs | T22 criterion | T23 incr-mel(defer)
WAVE 5  (release hygiene + optional/last)          T24 migration/0.4.0 | T25 EOU cursor | T26 NIT batch
                                                  T27 auto-continuous(optional) | T28 coverage(optional)
```

**Quick-wins (additive / zero-risk / V-confirmed - do first, ship value immediately):**
T05 detected_language, T07 F2 (bit-identical), T06 doc-example fix, T01 CI feature-matrix, T08 thiserror (mechanical), T20 EP discoverability, T21 concurrency docs, T26 NIT batch.

**Deep (multi-session, golden-gated, behavior-changing):**
T12 M2-decoder consolidation, T14 language re-prompt/reset, T23 incremental mel (and reclassified - defer unless ROI). T10 M5 and T13 M2b are golden-gated behavior/numerics work but smaller.

**Parallelism:** Within Wave 0, T01/T02 are independent (T03 needs T02's fixture). Within Wave 1 all six are independent of each other (after Wave 0). Wave 2 is the ONLY strictly-serial chain (T10 -> T11 -> T12 -> T13). Wave 3+ tasks are largely independent except the stated deps.

---

## Part E - Surviving risks + open questions needing a human decision

**Surviving risks (verified-but-not-eliminated):**
- **R1 [V1 residual] - the language re-prompt is mechanism-verified by code+inference, not a live run.** V1 traced the ONNX I/O and proved the cache is language-agnostic, but no one ran `./nemotron_multi` to confirm that resetting joint/LSTM while preserving the cache yields a clean continuation in practice. **Mitigation:** T03's code-switch golden is the live proof; T14 is gated on it going green. Cold-start mis-commit (chunk-0 zero cache) may persist even after re-prompt.
- **R2 - M5 changes real output on ties/NaN.** Verified behavior-changing. Low blast radius (ties are pathological), but it IS a transcript-affecting change folded under a "consolidation." **Mitigation:** isolated PR (T10), per-variant token tests first, golden on real audio.
- **R3 - M2-decoder parameterization (>=4 knobs) is asserted parameterizable, not built.** V4 says the four run_decoder bodies collapse to ~150 net after a ~60-line helper, but the knob count could grow and erode the ROI. **Mitigation:** if knobs exceed ~6 or the helper exceeds the bodies it replaces, STOP and keep them separate (the net deletion is only ~350-550 lines - it is not worth a leaky abstraction).
- **R4 - F3 TensorRef errors on non-contiguous input.** Survives only with a standard-layout assert; a future input that is non-contiguous would panic instead of silently copying. **Mitigation:** assert + golden across 4 files (T12).
- **R5 - streaming==offline may not be byte-equal until M6 lands.** T03 may start with a tolerance. **Mitigation:** tighten the golden after T09.

**Open questions needing a human decision:**
- **Q1 (semver) - batch all 0.x breaks into ONE 0.4.0, or stagger across several minor bumps?** The roadmap spends breaks in T04, T08, T09(length), T11(precedence), T14, T15, T17, T18. Recommendation: **batch into 0.4.0** with the T24 migration guide (satisfies SM-1, one disruption for users), but this forces the breaking PRs to merge to a release branch rather than ship continuously. Needs the maintainer's release cadence preference.
- **Q2 (scope/ROI) - how much consolidation is worth it given the corrected ~350-550 lines (not 700-900)?** M2a (T11, mechanical, safe) is clearly worth it. M2-decoder (T12, L, behavior-reconciling) is the marginal call: the deletion is modest and the abstraction carries M5's behavior change + the F3 assert. Recommendation: **do T11 + T13 always; treat T12 as conditional on the knob count staying low (R3).** Human decides the appetite.
- **Q3 (language v1 scope) - expose caller-boundary language re-detection (T14) in v1, or ship only detected_language() (T05) and document the lock?** T05 is zero-risk and resolves the user-facing "why is it locked" confusion immediately. T14 is the actual fix but is L-sized, breaking, and gated on a live golden (R1). Recommendation: **ship T05 in the next release regardless; gate T14 on R1's golden going green and a maintainer decision** - it is the headline fix but also the highest-risk change. Auto-continuous (T27) is explicitly optional/last.
- **Q4 (deferral) - build the incremental-mel refactor (T23) at all?** Reclassified out of safe-perf, L-sized, only benefits long single streams, and must be golden-rebaselined against today's seam-bearing output. Recommendation: **defer until T22's benchmark proves the O(n^2) is a real-world bottleneck for the target workloads.** Needs a workload/ROI judgment from the maintainer.

---

## Cross-references for other lanes

- **orchestrator (FINAL-plan):** the backlog is T01-T28 in the Part D wave order. The four non-negotiable ordering rules (Part B) are the build contract: goldens (Wave 0) before refactors; M8 un-export (T04) before the consolidation chain (Wave 2); detected_language (T05) before reset_with_lang (T14); M5 (T10) before the decoder merge (T12). The mel-accuracy epic is DROPPED (only T13 numerics-unchanged consolidation + an optional EN fixture survive) - the implementor must NOT touch Hann/log-floor/normalization. Concurrency is a DOCS task (T21), not a Mutex removal. Surviving risks R1-R5 and open questions Q1-Q4 (esp. Q3 - whether T14 ships in v1) need a human decision before the breaking work starts.
