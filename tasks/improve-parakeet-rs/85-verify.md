# Wave V — Adversarial Verification (consolidated)

Five skeptics each tried to REFUTE one load-bearing claim against code + the export script + the ort
source + the NeMo reference. Per-verifier detail in `85-verify-v1..v5-*.md`. This is the reconciled
verdict synthesis consumes. Net: two claims survive, one survives scoped, one is partly refuted, one
is fully refuted — and one audit finding was BACKWARDS.

## V1 — language-lock fix is in-library, no re-export → **SURVIVES (scoped)**

Traced the ONNX graph I/O, not docstrings. The mechanism holds:
- Prompt is applied to encoder OUTPUT via `prompt_kernel` MLP AFTER `cache_aware_stream_step`
  (`export...multilingual.py:334-359`); Rust mirrors it (`model_nemotron.rs:149-195`). Cache is
  language-agnostic — the export's own verification feeds the same cache for every language and varies
  only `prompt_index` (`:556-586`). `prompt_index` is a per-call input with dynamic batch axis
  (`:409-413`); fixed only by library policy (`nemotron.rs:489,591,691`, `reset()` preserves it).
- `<lang>` tags are real SentencePiece ids kept in `accumulated_tokens`, stripped only at render
  (`nemotron.rs:518,716`), so `detected_language()` is derivable.
**SCOPING (the break found):** Nemotron emits NO boundary/VAD/LID signal, so "auto-continuous
re-detect at utterance boundaries" cannot be AUTONOMOUS in-library — the CALLER must supply the
boundary. True sub-sentence/mid-word code-switch is model-inherently unsupported (NVIDIA picks the
dominant language per utterance). `is_lang_tag` (`:78-93`) is a shape heuristic (A1-08) — harden to
exact-id membership before using it for control flow.
**Safe fix set (no re-export):** (1) `detected_language()` getter [additive, zero-risk]; (2) per-chunk
re-prompt + an atomic `reset_with_lang` helper + fix `set_target_lang`-needs-reset semantics [breaking];
caller-driven boundary re-detection. A frame-level LID posterior would need a re-export (out of scope).

## V2 — the three perf refactors are byte-equal → **umbrella REFUTED** (split verdict)

- **F2 (route Nemotron/EOU/Multitalker through cached FFT plan): SURVIVES, bit-identical.** `stft`
  already delegates to `stft_with_plan` (`audio.rs:74-83`); F2 only removes the per-call planner build.
  The 3 "mel flavors" live OUTSIDE stft, so F2 neither fixes nor worsens them. Can land early/independent.
- **F3 (TensorRef zero-copy): SURVIVES, conditional.** ort rc.12 `from_array` silently copies
  non-contiguous input; `from_array_view` ERRORS on it (`create.rs:452-477`). Equal only for
  C-contiguous input — verified all migrated Nemotron inputs are freshly-owned/contiguous. Must add a
  standard-layout assert + golden-gate across all 4 `model_*.rs`.
- **F1 (incremental mel over new frames only): REFUTED as stated.** This is a STREAMING-CORRECTNESS
  change, not pure perf. Three seam hazards: preemphasis is a cross-sample filter (`audio.rs:57-58`);
  `stft_with_plan` always prepends `n_fft/2=256` zeros (`:94-97`); frame alignment must match the
  `PRE_ENCODE_CACHE=9` overlap. Subtlety: the current trim (`nemotron.rs:705-712`) already re-pads and
  restarts preemphasis mid-stream, so today's output is itself seam-bearing and an idealized continuous
  mel would DIVERGE from it. RECLASSIFY F1 out of the safe-perf bucket; it needs an executed byte-equal
  golden experiment and must sequence after A6 fixtures.

## V3 — mel front-end divergence is a real accuracy bug → **REFUTED, do NOT escalate** (audit was wrong)

- **A1-09 (Hann window): REFUTED, BACKWARDS.** Rust `hann_window` (`audio.rs:64-68`) uses the `N-1`
  denominator = SYMMETRIC. NeMo `FilterbankFeatures` uses `torch.hann_window(..., periodic=False)` =
  SYMMETRIC. The code ALREADY matches; implementing the proposed "fix" would INTRODUCE divergence.
  Action: correct the audit, change no code.
- **A1-10 / Q2 (`x.max(0.0)` log floor): REFUTED, mathematical no-op.** Mel values are always >= 0, so
  the clamp changes nothing. Pure cosmetic drift; fold into Q2 consolidation, not an accuracy fix.
- **A1-10 (Nemotron "no normalization"): REFUTED for multilingual; LOW-confidence UNCERTAIN for EN.**
  Normalization lives outside the graph (both exports use `online_normalization=False`). Multilingual
  `config.json` says `"normalize":"NA"` = none in NeMo, Rust matches, and the model demonstrably
  produces coherent output. EN export hardcodes `"per_feature"` in a dumped dict literal (not read from
  cfg) and ships no config.json — low-confidence; EN works in practice. One OPTIONAL mel-dump fixture
  for EN; no code change unless it proves a mismatch.
- Secondary params all MATCH NeMo (n_fft=512, win=400, hop=160, n_mels=128, preemph=0.97, log guard
  2^-24, Slaney mel [EOU intentionally HTK], dither correctly omitted). **The only genuine multilingual
  accuracy problem is the language-lock, which is unrelated to the mel front-end.**

## V4 — consolidation removes ~700-900 lines AND preserves behavior → **REFUTED (as stated)**

- **Line count inflated.** Real behavior-safe net deletion is **~350-550 lines**, not 700-900 (the
  figure counts gross duplicated text; after a ~60-line shared helper + thin call sites the four
  `run_decoder` bodies (209L) collapse to ~150 net). The "4× greedy loop" claim collapses: only
  Nemotron + Unified share a body; EOU (`<EOU>`/soft-reset/overflow) and TDT (duration-skip) do not merge.
- **"Preserves behavior" is FALSE — it is behavior-RECONCILING.** The four argmaxes have three
  different tie/NaN semantics (`nemotron.rs:749-756` first-wins/NaN→0; `parakeet_unified.rs:477-482`
  last-wins/NaN→Equal; EOU finite-guarded; TDT last-wins). Unifying (A1-06/M5) CHANGES Nemotron/Unified/
  TDT output on ties/NaN. This is the M5-before-M2 dependency made concrete.
- **Claim (2) SURVIVES with a correction:** leaked `model_X` types have ZERO references in
  examples/README — removing `lib.rs:87-91` is the only public break and genuinely unblocks `pub(crate)`.
  BUT `SentencePieceVocab` (`lib.rs:91`) is used cross-module by `multitalker.rs` + `parakeet_unified.rs`
  — un-export it but keep it `pub(crate)` (move to a `vocab/` home); do NOT privatize to nemotron.rs or
  the build breaks. "Non-breaking" is true only in the API/semver sense; M2 still carries M5's output change.
- **M2 must SPLIT into ~5 PRs:** M8 (the one API break) → M5 (the one behavior change, isolated for
  bisect, per-variant token regression tests first) → M2a (session/find-encoder dedup) → M2-decoder
  (parameterized helper, ≥4 config knobs) → M2b (mel front-end, numerics unchanged, golden-mel gated).

## V5 — the shared-model Mutex needlessly serializes streams (run is &self) → **REFUTED**

The premise is factually false for `ort = 2.0.0-rc.12`: every public `Session::run*` takes **`&mut self`**
(`session/mod.rs:212,253,340,407`); only crate-private `run_inner` is `&self`. So "drop the Mutex and
call run(&self)" does not compile; the Mutex is the MINIMUM synchronization the signature forces (a
RwLock doesn't help — readers still need `&mut`). Per-stream state isolation IS correct (`NemotronModel`
holds only Sessions+config; all mutable state is on the per-stream `Nemotron`; `from_shared` clones only
Arcs) — so IF ort offered `&self` run, parallel streams would be safe (re-open trigger if ort is bumped).
**Corrections:** F12/A3-10 "drop the Mutex" recommendation must be rewritten. `spawn_blocking` guidance
survives only as a DOCS item and needs bounded concurrency (tokio blocking pool up to 512 vs 4 intra-op
threads) + per-stream task pinning (chunk order = decode-state order). True parallelism = one Session per
stream (N× the 2.3GB model), a different design, not a Mutex removal.

## Net effect on the roadmap (for synthesis)

1. **Language epic stands**, scoped: ship `detected_language()` (additive) + `reset_with_lang`/per-chunk
   re-prompt (breaking) + caller-supplied boundary; document that mid-word code-switch is unsupported and
   auto-continuous needs a caller VAD boundary. Harden `is_lang_tag` first.
2. **Drop / downgrade the mel "accuracy" work (M12):** Hann is already correct (fixing it = regression);
   log-floor is a no-op; normalization matches for multilingual. Keep only an OPTIONAL EN normalize
   fixture. This removes a whole epic-ish chunk of misdirected effort.
3. **Reclassify F1** from safe-perf to a correctness-sensitive change requiring a byte-equal golden;
   **F2 lands early**, **F3 needs a contiguity assert + golden across 4 files.**
4. **Consolidation (M2): ~350-550 lines, behavior-reconciling not preserving; split into ~5 PRs**
   (M8 → M5 → M2a → M2-decoder → M2b); keep `SentencePieceVocab` `pub(crate)`.
5. **Concurrency: rewrite the recommendation** — Mutex is necessary; offer spawn_blocking + bounded-
   concurrency docs now, one-Session-per-stream as the real (memory-costly) parallelism option.
6. Golden fixtures (A6) remain the precondition for M2/F1/F3.
