# 80 - Coherence + Feasibility (Wave A)

**/think frameworks:** systems-thinking (treat the nine lanes as one system; find where findings couple, where ordering is load-bearing, and where one fix's output is another's precondition) + map-vs-territory (verify each cross-lane coherence claim against the actual lane text and, where load-bearing, against code, rather than trusting the harvest summary). Inversion used once as a cross-check ("what would have to be true for this roadmap to be incoherent / un-buildable?").

This lane does NOT re-run the audits. It de-duplicates across lanes, builds the fix-dependency graph, checks epic coverage, resolves contradictions, marks what only Wave V can settle, names the gaps the whole audit missed, and returns a GO/REVISE verdict.

Spot-checks performed this lane (to confirm coherence claims, not redo audits): `lib.rs:76-99` (the public re-export set), `nemotron.rs:495-521` (reset preserves prompt_index; get_transcript strips lang tags). Both confirm the lane claims. One addition surfaced: `SentencePieceVocab` is ALSO publicly exported (`lib.rs:91`), which 02 flagged as a layering smell but no lane logged as a public-export semver item.

---

## VERDICT: GO (with three pre-synthesis notes for lane 90, none blocking)

The nine lanes are coherent. They converge (not collide) on the same small set of root causes, the dependency ordering is clean and acyclic, every HIGH/CRITICAL finding has an epic home, and the one apparent contradiction (consolidation "non-breaking" vs A3's export-removal being a break) is a sequencing artifact that both lanes already named and that resolves cleanly. The roadmap can be ranked and built.

GO is conditional only on lane 90 honoring three things the lanes already agreed on but which must not be lost in synthesis (detailed in the dependency section): (1) golden tests gate the perf + consolidation PRs; (2) the model_X un-export must land before, or be bundled into, the consolidation, and counts as the one deliberate break; (3) the language epic ships additively first (expose detected_language) before any reset/re-prompt break. These are ordering constraints, not unresolved questions, so they do not warrant REVISE.

Five claims remain genuinely uncertain until Wave V (85) verifies them; they are marked [WAVE-V] below. None of them blocks ranking, because each has an additive, zero-risk first step that is safe regardless of how V resolves it.

---

## 1. Merged, severity-ranked master finding list

Findings that recurred across lanes are merged into ONE entry with all cross-lane evidence and a single reconciled severity. Per-lane IDs are preserved in brackets so 90 can trace provenance. Severity is the max across lanes unless a lane's higher rating was scoped to a sub-case (noted).

### CRITICAL

**M1 - Multilingual `auto` language-lock: cannot code-switch mid-stream; `reset()` preserves the prompt.**
- Lanes: A1-01 (CRITICAL), A4-F1 (HIGH), RU (verdict), A3-3 (HIGH, the API half), A6-F2 (HIGH, the test gap), 02 sec 4 (the architecture seam). The seed finding.
- Reconciled severity: **CRITICAL** (it is the trigger of the whole run and the only CRITICAL-rated correctness item; A4 rated its own half HIGH, but the merged user-facing defect is CRITICAL).
- Root cause (dual-sourced, decisive): `prompt_index` is set once (`nemotron.rs:475-491`), fed every chunk (`nemotron.rs:591,691`), preserved by `reset()` (verified `nemotron.rs:495-509`); compounded by carried `last_token` + encoder cache + cold-start instability (RU sec 4). The graph accepts a fresh `prompt_index` per chunk and the encoder cache is produced BEFORE the prompt MLP so it is language-agnostic (`export_nemotron_streaming_multilingual.py:334-359`). The only LID signal is the in-band `<lang>` tag the library strips (`nemotron.rs:511-521`, verified).
- Verdict: **fixable in-library, NOT model-inherent, NO re-export** for sentence/utterance-boundary re-detection. Sub-sentence / mid-word code-switch is **model-inherent unsupported** (matches NVIDIA upstream).
- Fix shape (additive -> breaking): (1) expose `detected_language()` by surfacing the stripped `<lang>` tags [additive, zero risk]; (2) re-prompt `prompt_index` mid-stream + reset carried decoder state at a boundary [breaking-ish, touches reset semantics]; (3) auto-continuous re-detect needs an external boundary signal Nemotron lacks [largest, optional].
- [WAVE-V] whether resetting joint/LSTM state while preserving the encoder cache yields valid continuation (A4 asked A1; A1 deferred to RU; RU says safe because cache is language-agnostic, but this is [CODE]+[INFER], not run-verified).

**M2 - Massive `model_X.rs` duplication; the consolidation is the biggest structural win and a dependency hub.**
- Lanes: A5-Q1 (CRITICAL), 02 sec 2 (the scorecard), A2-F3/F4/F5 (the per-step clone cost that the shared backend fixes once), A4-F2/F3 (timed-token accumulation + onnx-file resolver are copy-pasted), A6 (model_X files have ZERO tests).
- Reconciled severity: **CRITICAL** (A5's rating; it is the load-bearing structural item).
- Evidence: ~700-900 of 7766 lines are mechanical copies. Session-build idiom 6-8x; `run_decoder` step 5x (`model_nemotron.rs:232-287`, `model_unified.rs:134-186`, `model_multitalker.rs:197-249`, `model_eou.rs:149-195`, inlined `model_tdt.rs:214-271`); cache-rebuild verbatim 3x; greedy/argmax 4x WITH divergent tie-break/NaN (= M5); `find_encoder` candidate lists divergent 3-4x (= M9). `model_cohere.rs` already shows the right zero-copy `TensorRef` pattern.
- NON-BREAKING ONLY IF the publicly-exported low-level types are un-exported first (= M8). Safe ONLY with golden tests first (= M11).

### HIGH

**M3 - Streaming mel recomputed over the entire growing buffer every chunk (O(n^2)).**
- Lanes: A2-F1 (HIGH), A1-07 (MEDIUM, correctness-adjacent: cursor/buffer desync, not just perf), 02 sec 3a.
- Reconciled severity: **HIGH** (perf) with an embedded MEDIUM correctness sub-case (the `audio_processed` cursor can desync on large/irregular chunks, `nemotron.rs:705-712`). Keep both facets; the fix (frame-accurate incremental STFT / ring) addresses both at once.
- Evidence: `nemotron.rs:617,628,633-639`; `multitalker.rs:326,334`.
- [WAVE-V] is F1 purely perf or does it change numerics? (A2 flagged this for refutation; the desync sub-case proves it is NOT purely perf - frame alignment is load-bearing, so the fix MUST be golden-guarded.)

**M4 - Every `ort` tensor input is cloned (owning copy) instead of borrowed; outputs `to_vec()`'d.**
- Lanes: A2-F3 (HIGH inputs) + A2-F4 (MEDIUM outputs) + A2-F5 (MEDIUM frame scratch), A5-Q5 (per-step clone is a consequence of the duplicated `run_decoder`), 02 sec 2c-2e.
- Reconciled severity: **HIGH**. ~7.7 MB/chunk on encoder cache in + same on out; LSTM-state clones up to ~560x/chunk. `model_cohere.rs:111,160-194` is the zero-copy template.
- Coupling: this is fixed ONCE inside the shared RnntBackend (M2). Doing it per-variant is four copies of the same change.
- [WAVE-V] is the `TensorRef` migration numerics-preserving? (A2/A6 say yes if frame alignment + argmax are bit-identical; must be golden-guarded.)

**M5 - Divergent greedy argmax: non-deterministic tie-break + unguarded NaN across 4 decoders.**
- Lanes: A1-06 (MEDIUM), A5-Q5 (the 4-way reimplementation), 02 sec 2e, A6-F3 (untested).
- Reconciled severity: **HIGH** (raised from A1's MEDIUM: it is BOTH a latent correctness bug AND the prerequisite that must be reconciled before M2 can collapse the greedy loop - its blast radius through the consolidation makes it higher-leverage than a standalone MEDIUM).
- Evidence: Nemotron first-wins, no `is_finite` (`nemotron.rs:749-756`); Unified/CTC last-wins via `max_by(...unwrap_or(Equal))` (`parakeet_unified.rs:477-482`, `decoder.rs:49-53,137-141`); EOU has the `is_finite` filter (`parakeet_eou.rs:208-215`). A NaN logit in Nemotron silently decodes token 0.
- Fix is a shared `argmax(&[f32]) -> usize` (first-wins + finite guard, matching EOU and NeMo). This is a correctness prereq for M2's loop consolidation.

**M6 - Nemotron streaming drops the final partial chunk; offline vs streaming encode the same audio with different `length`.**
- Lanes: A1-02 (HIGH, no flush) + A1-03 (HIGH, constant vs true length), A3-4 (no flush is an API-consistency gap; Unified has one), A6-F6 (streaming==offline equivalence is the guard).
- Reconciled severity: **HIGH**. `nemotron.rs:637-639` (drop), `581` (offline true len) vs `689` (streaming constant len). Up to ~560 ms lost per utterance; the two public entry points diverge, so offline cannot be the streaming oracle (which also blocks a cheap test strategy).
- Fix: add `flush()` using the offline `length` convention (additive).

**M7 - EOU has no processed-sample cursor; assumes exact 16-frame chunks; no validation.**
- Lanes: A1-04 (HIGH), A3 (input-validation/ergonomics gap), 02 sec 3a.
- Reconciled severity: **HIGH**. `parakeet_eou.rs:147-165,174`. Token duplication/loss whenever chunk size is not exactly 16 new frames; cold-start uses real audio as pre-encode context. Low-risk interim: enforce + reject the required chunk size.

**M8 - The `model_X` low-level types are publicly exported; un-exporting them is itself a break AND the gate for M2.**
- Lanes: A3 (the decision is A3's), 02 sec 4b + "what would break" inventory, A5-Q1 (notes they are mostly internal-use-only).
- Reconciled severity: **HIGH** (it is the hinge that decides whether M2 is breaking or not).
- Verified (`lib.rs:86-91`): publicly exported low-level types are `ParakeetModel`, `ParakeetEOUModel`, `NemotronEncoderCache`/`NemotronModel`/`NemotronModelConfig`, `ParakeetUnifiedModel`/`UnifiedModelConfig`, plus `decoder::ParakeetDecoder`, and `SentencePieceVocab` (the last one no lane logged explicitly as a public-export item - flag for 90).
- These look like accidental exports (users need only the wrappers). Recommendation across lanes: stop exporting them, accept it as the one deliberate 0.x break, which makes M2 a non-breaking internal refactor thereafter.

**M9 - `find_encoder`/`find_decoder_joint` candidate lists diverge; no uniform quantization story; 6 bespoke file-pickers.**
- Lanes: A4-F3 (MEDIUM), A5-Q1 (the divergence is a correctness hazard), 02 sec 2b.
- Reconciled severity: **HIGH** (raised from A4's MEDIUM: the README advertises int8/int4 Nemotron files the loader literally cannot pick up (`README.md:155`, `model_nemotron.rs:72`), so it is a user-visible defect, not just a smell; and int8-first vs int8-last priority means two variants behave oppositely for the same on-disk layout).
- Fix: one `resolve_onnx_file(dir, role, prefer)` + a `prefer: Quantization` knob (coordinate A3 API shape). Changing precedence can silently switch which file loads -> gate or call out as a documented break.

**M10 - No `StreamingTranscriber` trait; streaming method names + reset are inconsistent across variants.**
- Lanes: A3-1/A3-2/A3-3 (all HIGH), 02 sec 4a, A4-F5 (the facade depends on this).
- Reconciled severity: **HIGH**. 5 streaming variants share NO trait and use 5 verbs (`transcribe`/`transcribe_chunk`/`feed`/`diarize_chunk`); reset is spelled 4 ways, EOU has no public reset, Nemotron's reset silently preserves language (= the API half of M1).
- Fix: additive `StreamingTranscriber` trait + one EOU rename + public `reset` on EOU. The trait is additive; the rename and reset-semantics change are the breaking parts (deprecate-alias).

**M11 - No regression guards for any of the above; CI never compiles the gated features.**
- Lanes: A6-F1 (HIGH, gated features never built in CI), A6-F2/F3 (HIGH, language-lock + transcription core untested), A6-F6 (golden/equivalence tests), plus EVERY other lane's reverification note depends on A6.
- Reconciled severity: **HIGH** and **the load-bearing precondition** (see dependency graph). `.github/workflows/rust.yml:21-24` builds default features ubuntu-only, so `sortformer`/`multitalker`/`cohere` + the cohere example can ship broken green.
- Fix: Tier-1 pure no-model tests (CTC collapse, TDT frame-advance, `is_lang_tag`, mel reference, `reset()` contract, argmax) in CI + a feature-matrix CI job; Tier-2 `#[ignore]`/env-gated goldens vs `./nemotron`/`./nemotron_multi`. Needs a committed WAV fixture (M-gap: `test_en.wav` is not git-tracked, A6-F5).

### MEDIUM

**M12 - Mel / Hann / normalization divergence across paths (accuracy vs reference).**
- Lanes: A1-09 (Hann symmetric vs periodic) + A1-10 (three log-mel treatments; Nemotron "no normalization" unverified), A5-Q2 (the `max(0.0)` divergence + 5 copies of constants), RU (confirmed the prompt path but mel/normalize params still need confirming).
- Reconciled severity: **MEDIUM** (could escalate to HIGH if the Nemotron normalize assumption is wrong - blast radius is every Nemotron transcript).
- Evidence: `audio.rs:64-68` (symmetric Hann); `nemotron.rs:784` vs `parakeet_eou.rs:261` vs `audio.rs:240-263` (three floors); `nemotron.rs:784` vs `multitalker.rs:675` (`max(0.0)` drift).
- [WAVE-V] is the mel/Hann/normalize divergence a real accuracy bug vs benign? RU could not finalize NeMo's `periodic`/`normalize` settings from the export script alone. **This is the single finding most dependent on Wave V** and the only one with a HIGH-escalation path. The safe first step regardless of V: consolidate the constants + a single mel front-end (M2/A5-Q2) WITHOUT changing numerics, guarded by a golden-mel fixture; defer any Hann/normalize value change until V confirms direction.

**M13 - `Error` is stringly-typed, not `#[non_exhaustive]`, loses source chains.**
- Lanes: A3-5 (MEDIUM), A5-Q4 (MEDIUM, `source()` returns None, `.map_err(Error::Model(format!))` dozens of times).
- Reconciled severity: **MEDIUM**. Both lanes agree: keep the concrete enum (NOT eyre), adopt `thiserror`, add structured `#[non_exhaustive]` variants. Breaking but cheap pre-1.0; do it once so later variant adds are not themselves breaks.

**M14 - Constructor / config inconsistency + triple `ModelConfig` name collision + `WordTimestamp`/`TimedToken` duplication.**
- Lanes: A3-7 + A3-8 (MEDIUM), A5 (the naming drift), A4 (Sortformer exposes internal `ModelConfig`).
- Reconciled severity: **MEDIUM**. Param name drift (`config` vs `exec_config`); handle pattern on only 3 of 8; Sortformer exposes `ModelConfig` not the `ExecutionConfig` alias; three structs named `ModelConfig`; `WordTimestamp` cannot feed `process_timestamps`.

**M15 - Stringly-typed language; two variants disagree on codes; no `Auto` type.**
- Lanes: A3-6 (MEDIUM), A4-F1 (the `Auto` distinction matters for M1), A4-F6 (TDT advertises auto-detect with no API).
- Reconciled severity: **MEDIUM**. `Language` newtype/enum with an `Other(&str)` escape hatch; make `Auto` a first-class state (the type-system half of M1). Additive if a `&str` convenience delegates.

**M16 - No word timestamps on streaming Nemotron/EOU though the graph supports it.**
- Lanes: A4-F2 (HIGH) + A3-4 (MEDIUM).
- Reconciled severity: **MEDIUM** (A4 rated HIGH; merged to MEDIUM because it is purely additive and the timed-token accumulation already exists in Unified/Multitalker to copy - low risk, clear win, but not a correctness bug). `(token_id, frame)` accumulation is in `parakeet_unified.rs:442-512` + `multitalker.rs:593-663`, missing in `nemotron.rs:723-770`. Reuse `timestamps::group_by_words`; document Words mode for multilingual (sentence split keys on Latin punctuation).

**M17 - Cohere hardcodes `<|notimestamp|>`/`<|nodiarize|>` though the model supports both.**
- Lanes: A4-F4 (MEDIUM). `cohere.rs:308-309`. Add `timestamps`/`diarize` toggles via a `CohereOptions` builder (avoids the 6-arg signature, coordinate A3-9).

**M18 - `transcribe_chunk` is blocking `&mut self`; `Arc<Mutex<Model>>` serializes concurrent streams; no async guidance.**
- Lanes: A2-F12 (MEDIUM), A3 (joint API+perf), 02 sec 3c.
- Reconciled severity: **MEDIUM**. ORT `Session::run(&self)` is itself thread-safe, so the Mutex is stricter than ORT requires; per-instance state is already separate (`nemotron.rs:312-324`). Document `spawn_blocking` (cheap, do now); investigate `&self` runs to parallelize shared-model streams (needs ORT `Send+Sync` confirmation).
- [WAVE-V] confirm `ort 2.0.0-rc.12` `Session::run` is `&self` + `Send+Sync` before promising the concurrency win.

**M19 - EP/GPU discoverability: accelerators are cfg-gated, never auto-discovered, CPU-only default; no `compiled_providers()`/`gpu()`/`Auto`.**
- Lanes: A2-F6 (HIGH impact, opt-in) + A3-10 (LOW). `execution.rs:16-36`.
- Reconciled severity: **MEDIUM** (A2 rated impact HIGH but it is gated on hardware + opt-in, and A3 rated the API friction LOW; merged to MEDIUM - the realized win needs the user to act, but the 5-10x is the largest available lever and the discoverability fix is cheap). Document the EP matrix + CoreML-slower caveat in rustdoc; add `compiled_providers()` + opt-in `Auto`.

### LOW / NIT (carried for completeness, one-line each)

- **M20** EOU soft-reset asymmetry + `<EOU>`-then-token loss (A1-05, MEDIUM->LOW: specialized path, soft-reset is intentional). `parakeet_eou.rs:217-255`.
- **M21** `is_lang_tag` strips by string shape, can over/under-strip (A1-08, MEDIUM->LOW: low frequency, silent). Strip by token-id membership. `nemotron.rs:78-93`.
- **M22** Unified flush zero-pads right context (A1-12, LOW) - standard streaming tradeoff, document.
- **M23** `max_symbols_per_step` magic caps differ 10/5/10, silent truncation (A1-13, LOW). Unify + log on cap-hit.
- **M24** CTC timestamps re-decode every prefix, O(n^2) (A2-F10, LOW), offline path only.
- **M25** `decode_with_beam_search` is a public no-op ignoring `beam_width` (A5-Q6, A6-F9, A3). Remove / `#[deprecated]` / implement. `decoder.rs:199-206`.
- **M26** `pub use transcriber::*` glob export (A3-12, NIT) - semver hazard, make explicit.
- **M27** Files over the 800-line ceiling: `sortformer.rs` 1254, `nemotron.rs` 786, `multitalker.rs` 771; god-functions `model_cohere.rs::run_decoder_step` 119L, `model_tdt.rs::greedy_decode` 116L (A5-Q3, NIT-MEDIUM, style).
- **M28** CI lacks fmt/clippy/MSRV (A6-F8, LOW).
- **M29** `from_pretrained` silent model-file auto-detect, no precision selector (A3-13, NIT) - folds into M9.
- **M30** `mel_spectrogram` skips normalization for `num_frames <= 1` (A1-11, LOW), sub-30ms edge.
- **M31** Multitalker ASR-chunk vs Sortformer-stride mismatch, nearest-neighbour speaker masks (A4-F7, LOW), documented known limitation.

Positive baseline (no finding, recorded so 90 does not "fix" it): all 8 lock sites use `.lock().map_err` (no poisoning panics); zero `panic!`/`todo!`/`unimplemented!`; no `#[allow(dead_code)]` (A5-Q6).

---

## 2. Fix-dependency graph (the load-bearing ordering)

Acyclic. Read top to bottom; an arrow means "must land before." Items on the same tier are independent and parallelizable.

```
TIER 0  (foundation - unblocks everything, no dependencies)
  M11a  Tier-1 pure no-model tests + committed WAV fixture + feature-matrix CI
        (CTC collapse, TDT frame-advance, is_lang_tag, mel reference, reset() contract, argmax)
  M5    Reconcile argmax into ONE shared first-wins+finite helper (correctness prereq for M2)
  M8    Un-export the model_X low-level types (the one deliberate 0.x break; gate for M2)
        |
        +--> requires M11a's reset()/argmax/collapse tests to exist as the guard

TIER 1  (depends only on Tier 0)
  M2    Extract pub(crate) RnntBackend / src/onnx/  [needs M5 reconciled, M8 un-exported, M11a goldens]
        |  this single refactor also delivers, once instead of N times:
        |   - M4   (TensorRef zero-copy inputs/outputs)
        |   - the per-step clone removal (A5-Q5)
  M3    Incremental STFT / frame-accurate ring  [needs M11a golden-transcript guard]
  M1a   Expose detected_language()  [ADDITIVE, zero risk - no dependency, can ship Tier 0 too]
  M6    Add Nemotron flush() using offline length convention  [additive; pairs with M11 equivalence test]
  M13   thiserror + structured #[non_exhaustive] Error  [breaking, cheap pre-1.0; do before M2 deletes the .map_err boilerplate so M2 lands on the new type]

TIER 2  (depends on Tier 1)
  M2b   Shared mel front-end + single constants source  [part of/after M2; needs M12 golden-mel guard; do NOT change numerics yet]
  M9    Shared resolve_onnx_file + prefer:Quantization knob  [after M13 for the error shape; after M2 for the shared home]
  M10   StreamingTranscriber trait + uniform reset + EOU rename  [trait additive; after M6 so flush() is in the trait; after M8/M2 so the surface is settled]
  M16   Word timestamps on Nemotron/EOU  [additive; reuses M2's shared timed-token accumulation]
  M1b   Re-prompt mid-stream + reset carried decoder state at boundary  [BREAKING-ish; AFTER M1a (must observe before re-deciding) and AFTER M11's code-switch golden]

TIER 3  (depends on Tier 2)
  M15   Language newtype with Auto first-class  [after M10's surface + M1a/M1b language flow]
  M14   Constructor/config unification + name-collision rename  [after M10]
  M17   Cohere timestamps/diarize toggles via CohereOptions  [after M13/M14 options pattern]
  M18b  &self runs for parallel streams  [after M2; needs Wave-V ORT Send+Sync confirmation]
  M19   EP Auto + compiled_providers()  [independent, can float earlier; placed late only by priority]
  M1c   Auto-continuous re-detect  [LAST; needs an external boundary signal Nemotron lacks; largest, optional]
```

The three load-bearing rules, stated plainly:
1. **M11a (goldens) before M2/M3/M4/M12.** Every perf and consolidation lane independently said its refactor is correctness-preserving ONLY under a byte-equal transcript / mel guard. Without the goldens, these PRs are unsafe. This is why M11 is HIGH and Tier 0.
2. **M8 (un-export) before M2 (consolidate).** The consolidation is "non-breaking" only after the `model_X` public types are removed. If M2 lands first, it is a break; if M8 lands first (or they bundle), M2 is internal. M8 is the one deliberate break the whole roadmap spends.
3. **M1a (additive detected_language) before M1b (breaking re-prompt/reset).** You cannot auto-re-decide a language without first observing it, and exposing the tag is zero-risk; the reset-semantics change must come after the code-switch golden exists.

M5 before M2 is the fourth, narrower rule: the divergent argmax must be reconciled to ONE behavior before the greedy loop is collapsed, or the refactor silently picks one variant's tie-break for all.

---

## 3. Coverage check (PRD's 6 epics vs every HIGH/CRITICAL finding)

| Epic | Findings homed | HIGH/CRITICAL covered? |
|------|----------------|------------------------|
| EP-01 Correctness & streaming | M1, M3(corr half), M5, M6, M7, M20, M21, M22, M23, M30 | M1(C), M3(H), M5(H), M6(H), M7(H) - all covered |
| EP-02 Performance & RTF | M3(perf), M4, M18, M19, M24 | M4(H) covered; M3 shared with EP-01; M19 MEDIUM |
| EP-03 API consistency | M8, M10, M13, M14, M15, M25, M26, M29 | M8(H), M10(H), M13 MED - covered |
| EP-04 Model coverage | M1(graph half), M9, M15, M16, M17, M31 | M9(H), M16 MED - covered |
| EP-05 Code quality | M2, M5(dup), M12(dup), M13(boilerplate), M27 | M2(C) - covered |
| EP-06 Tests & CI | M11, M28, + fixture gap | M11(H) - covered |

**Every HIGH/CRITICAL finding has an epic home. No orphan findings. No epic without findings.**

Two findings are legitimately co-owned (not orphaned): M1 spans EP-01 (streaming-state correctness) + EP-04 (the graph/feature capability) + EP-03 (the reset/Language API surface) - this is by design per the PRD's stated co-ownership (A1 owns correctness, A4 owns feature, A3 owns surface, RU is authoritative). M3 spans EP-01 (cursor desync) + EP-02 (O(n^2)). Synthesis should list each as ONE task with the merged scope, not duplicate it per epic.

One coverage nuance for 90: **M8 (un-export) is the keystone but is small and easy to drop**. It is not a user-facing feature, so it can be overlooked in ranking, yet M2 (the CRITICAL structural win) is blocked on it. 90 must rank M8 immediately ahead of M2, not bury it.

---

## 4. Contradictions between lanes - resolved or flagged

**C1 - "Is the language-lock model-inherent?" A1 hedged; RU+A4 say fixable. RESOLVED.**
A1 explicitly labelled the model-internal half as deferred-to-RU and did NOT claim inherence; it proved only the library half. RU (decisive, dual-sourced from the export script) and A4 (graph-capability) both conclude: fixable in-library for sentence/boundary granularity, model-inherent-unsupported for sub-sentence/mid-word. There is no actual contradiction - A1 deferred exactly the question RU answered, and all three converge. The nuance preserved: "re-detect every word" IS inherent-not-supported; "re-detect at boundaries" is fixable. 90 must carry both halves so the roadmap does not over-promise mid-word code-switch.

**C2 - "Consolidation is non-breaking" (A5/02) vs "un-exporting model_X is a break" (A3/02). RESOLVED - it is a sequencing artifact, not a contradiction.**
Both 02 (sec 4b) and A5 (Q1) already state the consolidation is non-breaking *conditional on* the un-export landing first. A3 owns the un-export decision and rates it a (cheap, deliberate) break. So the chain is: M8 is the break; after M8, M2 is non-breaking. "Consolidation non-breaking" and "export-removal is a break" are both true and refer to different PRs in sequence. Verified against `lib.rs:86-91` that the types are indeed public. No revision needed; 90 just sequences M8 -> M2 and counts M8 as the one spent break.

**C3 - Does the perf lane assume numerics change while correctness assumes they must not? RESOLVED - they agree, and A2 itself flagged it for V.**
A2 explicitly frames F1/F3/F4/F5 as "correctness-preserving refactors needing golden guards" and asked Wave V to refute "are F1/F3 purely perf or do they change numerics." A1's M3 (cursor desync) actually proves they are NOT purely cosmetic - frame alignment is load-bearing - which strengthens, not contradicts, A2's own caution. Both lanes land on the same rule: golden-guard them, change no numerics. Consistent.

**C4 - mel/Hann/normalize: A1 wants to change the window/normalization toward reference; A5 wants to consolidate to a single source; both could collide. FLAGGED, resolves by ordering.**
A5-Q2 (consolidate the 3-4 mel pipelines to one front-end) and A1-09/A1-10 (the symmetric-Hann and Nemotron-no-normalize MIGHT be wrong vs NeMo) are different operations on the same code. If consolidation changes numerics AND the accuracy fix changes numerics, doing them together makes it impossible to attribute a transcript change. Resolution (already implied by both lanes' "golden-guard" notes): consolidate first WITHOUT changing numerics (M2b, byte-identical, golden-mel guarded), THEN apply any Hann/normalize value change as a separate, V-confirmed PR (M12). Not a contradiction, but 90 must keep these as two ordered PRs, not one.

**C5 - A4 rated the language-lock and word-timestamps HIGH; A1 rated the lock CRITICAL and A3 rated the ts MEDIUM. Resolved by the merge** (M1 = CRITICAL, M16 = MEDIUM) using max-severity-by-user-impact. No conflict, just per-lane scoping; recorded for transparency.

No hard contradiction requires a lane to be sent back. All apparent conflicts are sequencing or scoping, already named by the lanes themselves.

---

## 5. Feasibility flags - what only Wave V (85) can settle

These are the findings whose fix mechanism is genuinely uncertain until verified. Each is marked with its safe additive first step so it does NOT block ranking.

| ID | Uncertain claim | Who must verify | Safe-regardless first step |
|----|-----------------|-----------------|----------------------------|
| [V1] M1b | Resetting joint/LSTM state while preserving the (language-agnostic) encoder cache yields valid continuation | 85, ideally with `./nemotron_multi` | Ship M1a (expose detected_language) - additive, zero risk |
| [V2] M3/M4 | F1 (incremental STFT) and F3/F4 (TensorRef) are numerics-preserving, not behavior-changing | 85 + golden-transcript test | Land M11a goldens first; refactor only behind them |
| [V3] M12 | The symmetric-Hann and Nemotron "no normalization" are real accuracy bugs vs the NeMo reference | 85 (RU could not finalize `periodic`/`normalize` from the export script) | Consolidate to one mel front-end WITHOUT changing numerics; defer value changes |
| [V4] M2 | ~700-900 lines are truly removable despite the input-name/axis-order/dtype drift; the abstraction is correctly parameterizable | 85 | Land M5/M9 reconciliation + goldens first; M2 is mechanical after |
| [V5] M18b | `ort 2.0.0-rc.12` `Session::run(&self)` is `Send+Sync` so the Mutex can be dropped for parallel streams | 85 against the ort version | Document `spawn_blocking` (M18a) - cheap, correct regardless |

The orchestrator harvest already named V1-V5 as the five claims for Wave V to refute. This lane confirms that set is complete and adds no new must-refute claims. Crucially: **every uncertain claim has an additive, zero-risk first step**, which is why the roadmap is GO and not REVISE - ranking can proceed; V only gates the second (breaking/numeric) step of each.

---

## 6. Gaps - what the whole audit missed

These are NOT in any lane and 90 should add them as roadmap items (most are LOW/MEDIUM but real):

- **G-A - No benchmark/criterion harness.** A2 quantifies costs by reading code and cites "needs a benchmark" repeatedly (F1 latency-vs-buffer-length, F12 speedup), and A6 notes "no `benches/` dir," but NO lane proposes a `criterion` harness as a roadmap task. Every perf claim (M3/M4) and the RTF goal (PRD G2/SM "RTF < 1.0 on CPU") is unverifiable without one. **Add: a `benches/` criterion harness for `transcribe_chunk` latency vs buffer length and RTF per variant. This is the missing acceptance instrument for EP-02.** Severity MEDIUM (the perf epic cannot prove its own success metric without it).

- **G-B - No migration guide for the 0.x breaks.** The roadmap spends several deliberate breaks (M8 un-export, M13 Error restructure, M10 renames, M9 precedence, M15 Language type). A3 lists per-change semver tags but no lane proposes a single CHANGELOG/migration-guide deliverable. **Add: a migration guide / CHANGELOG section that batches the breaking changes into one version bump with before/after snippets.** Severity LOW-MEDIUM; it is the difference between "deliberate breaks" and "documented deliberate breaks" (SM-1 says zero UNDOCUMENTED breaks - this is how that is satisfied).

- **G-C - No versioned model-format / export-script compatibility contract.** The crate hardcodes ONNX I/O names and tensor shapes per variant (the export scripts in `scripts/` produce them). If NVIDIA re-exports or the export scripts change names, the loader breaks silently with a stringly-typed `Error::Model`. No lane proposes pinning/validating the expected graph I/O signature at load. **Add (LOW): validate the encoder/decoder input-output name set at load and emit a structured "model format mismatch" error (folds into M13's structured errors).**

- **G-D - No wasm/webgpu target story.** `webgpu` and `nnapi` are listed as EP feature flags (context pack, `Cargo.toml`) but NO lane assesses whether the crate compiles to `wasm32` or whether webgpu actually works end to end - A2-F6 explicitly scopes EP runtime out ("no accelerators in CI"). The flags exist but are unverified. **Add (LOW, flag-only): a `cargo check` for the `webgpu`/`nnapi`/`wasm32` targets in the CI matrix so the advertised flags do not rot** (extends A6-F1's feature matrix). This is a "does the advertised surface even compile" gap, not a feature request.

- **G-E - The `SentencePieceVocab` public export.** Verified at `lib.rs:91` it is publicly exported from `nemotron.rs`. 02 flagged it as a layering smell and recommended moving it to a `vocab/` home, but no lane logged that **moving it is a public-API break** (it is exported). It belongs in the M8 un-export decision set. **Add to M8's scope: decide whether `SentencePieceVocab` stays exported or moves (a break) when it relocates out of `nemotron.rs`.** Severity LOW (folds into M8).

- **G-F (minor) - No coverage-measurement task.** A6 proposes raising coverage (SM-5) but proposes no `cargo-llvm-cov`/tarpaulin CI step to measure it, so SM-5's "toward guarding all named invariants" has no instrument. LOW; optional.

None of these gaps changes the verdict - they are additive roadmap items, mostly LOW, with G-A (benchmark harness) the only one rising to MEDIUM because the perf epic cannot self-verify without it.

---

## Cross-references for other lanes

- **85 (verify):** The complete must-refute set is V1-V5 in section 5 (= the orchestrator's stated five; confirmed complete, none added). Priority order for refutation: V3 (mel/normalize - the only finding with a HIGH-escalation path and the one RU could not close) first; then V1 (the language re-prompt mechanism, the key claim for the CRITICAL M1 epic); then V2/V4 (numerics-preserving + line-removal for the consolidation); then V5 (ORT Send+Sync). For each, refute the BREAKING/numeric step, not the additive first step - the additive steps (M1a, M2b-no-numerics, M18a) are safe regardless and should not be gated on V. Also re-confirm against code: (a) `lib.rs:86-91` public type set incl. `SentencePieceVocab` (this lane verified it; confirm nothing else leaks), (b) `nemotron.rs:495-509` reset field contract (verified preserves prompt_index, clears the rest).

- **90 (synthesis):** Rank from section 1's merged severities (M1, M2 CRITICAL; then the HIGH block M3-M11; then MEDIUM; then LOW/NIT). Honor section 2's tier ordering as the build order - the four load-bearing rules (goldens before refactors; un-export before consolidate; additive-language before breaking-language; reconcile-argmax before collapse-loop) are non-negotiable. Do NOT bury M8 (un-export) despite its small size - it gates the CRITICAL M2. List co-owned findings (M1, M3) as single tasks with merged scope, not per-epic duplicates. Keep M2b (mel consolidation, no numerics) and M12 (Hann/normalize value change, V-gated) as TWO ordered PRs (contradiction C4). Add the six gaps from section 6 as roadmap items - especially G-A (criterion harness), without which EP-02's RTF success metric is unprovable. Verdict is GO: the package is coherent and rankable.

- **A1 / A4 / RU (already harvested, for traceability):** C1 resolved - no inherence contradiction; the lock is fixable at boundary granularity, inherent-unsupported sub-sentence. Carry BOTH halves into M1 so the roadmap does not over-promise.

- **A2 / A5 / 02:** C3 resolved - perf and correctness agree the refactors must be golden-guarded and numerics-preserving; M3's cursor-desync sub-case proves frame alignment is load-bearing (not cosmetic), which raises M3 above a pure-perf item. M2/M4/M5 collapse into one shared-backend PR sequence; do M5 (argmax reconcile) before the loop collapse.

- **A3:** C2 resolved - M8 (un-export) is the single deliberate break that converts M2 from breaking to internal; sequence M8 -> M2. Add `SentencePieceVocab` (G-E) to the un-export decision. Batch all breaks (M8/M9/M10/M13/M15) into one version bump with the G-B migration guide so SM-1 (zero UNDOCUMENTED breaks) holds.

- **A6:** M11 is Tier 0 and the precondition for M2/M3/M4/M12 - it cannot be ranked as a "supporting" afterthought; the goldens + committed WAV fixture (A6-F5) must land first. Add G-A (criterion `benches/`), G-D (wasm/webgpu/nnapi `cargo check` in the matrix), and optionally G-F (coverage measurement) to the CI/test strategy.
