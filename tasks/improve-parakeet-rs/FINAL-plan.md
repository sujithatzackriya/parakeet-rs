# FINAL Plan — Crate-wide improvement of parakeet-rs

## TL;DR

A 28-task improvement roadmap for `parakeet-rs` across correctness/streaming, performance, API, and
model coverage, built around four non-negotiable ordering rules (golden tests first; un-export before
consolidate; observe-language before re-detect; reconcile-argmax before merging the decode loop). The
plan beat a naive "fix everything" pass because the adversarial wave (V) **corrected the audit**: the
mel-"accuracy" epic was dropped (the Hann window already matches NeMo — "fixing" it would *cause* a
regression), the language-lock was confirmed fixable in-library, the consolidation was right-sized
(~350–550 lines, behavior-*reconciling* not preserving) and split into 5 PRs, and the "remove the
Mutex" idea was killed (ort `run` takes `&mut self`).

## Artifacts (this run folder, `tasks/improve-parakeet-rs/`)

PRD `../prd-parakeet-rs-improvement.md` · architecture `02-architecture.md` · audits
`A1-correctness.md` `A2-performance.md` `A3-api.md` `A4-models.md` `A5-quality.md` `A6-tests.md` ·
upstream research `RU-upstream.md` · feasibility `80-feasibility.md` (merged 31-finding list) ·
verification `85-verify.md` (+ `85-verify-v1..v5`) · **synthesis `90-synthesis.md` (full roadmap +
all 28 task specs)** · thought ledger `collective_thoughts.txt`.

## The headline answer to the trigger question

**The multilingual "auto" language-lock is a fixable library choice, not a model limit** (RU + A4 +
V1, from the export script): the language prompt is applied to the encoder *output* via an MLP, so the
streaming cache is language-agnostic and `prompt_index` can change per chunk; the model even emits
`<lang>` tags the library currently discards. **Scope (V1):** Nemotron emits no VAD/boundary signal,
so re-detection must be **caller-driven at utterance boundaries**; true mid-word code-switching is
upstream-unsupported. v1 can ship `detected_language()` (additive, zero-risk) now; the breaking
`reset_with_lang` + per-chunk re-prompt is the real fix, gated on a live golden.

## Four load-bearing ordering rules (the build contract)

1. **Goldens before refactors** — T01–T03 before any consolidation/perf change (they're correctness-
   preserving only under a byte-equal guard).
2. **Un-export before consolidate** — T04 (remove leaked `model_X` exports; keep `SentencePieceVocab`
   `pub(crate)`) before Wave 2; it's the one deliberate break that makes consolidation internal.
3. **Observe before re-detect** — T05 `detected_language()` before T14 `reset_with_lang`.
4. **Reconcile argmax before collapsing the loop** — T10 (M5) before T12 (decoder merge).

## Ranked task backlog (one worktree + one PR each)

Full specs (files, invariants, known unknowns) are in `90-synthesis.md` Part C. Sizes: S/M/L.

| Wave | id | title | deps | size |
|---|---|---|---|---|
| 0 | T01 | CI feature-matrix + fmt/clippy + wasm/webgpu/nnapi `cargo check` | — | S |
| 0 | T02 | Tier-1 pure unit tests + committed WAV fixture | T01 | M |
| 0 | T03 | Tier-2 env-gated golden harness (offline/streaming/code-switch/reset) | T02 | M |
| 1 | T04 | Un-export `model_X` types; relocate `SentencePieceVocab` to `vocab/` (BREAK, keystone) | W0 | S |
| 1 | T05 | `detected_language()` + harden `is_lang_tag` to exact-id (additive) | W0 | S |
| 1 | T06 | Fix non-compiling crate-root rustdoc example | — | S |
| 1 | T07 | F2: route Nemotron/EOU/Multitalker through cached FFT plan (bit-identical) | T02/T03 | S |
| 1 | T08 | thiserror + `#[non_exhaustive]` Error + model-format-mismatch variant (BREAK) | W0 | M |
| 1 | T09 | Nemotron `flush()` + reconcile offline/streaming length | T03 | M |
| 2 | T10 | M5: one shared first-wins+finite argmax (isolated behavior change) | T02/T03 | S-M |
| 2 | T11 | M2a: extract `src/onnx/` session-build + `resolve_onnx_file` dedup (folds M9) | T04,T08 | M |
| 2 | T12 | M2-decoder: parameterized RNNT helper + TensorRef zero-copy (F3) | T10,T11 | L |
| 2 | T13 | M2b: shared mel front-end, **numerics unchanged** (golden-gated) | T11,T12 | M |
| 3 | T14 | M1b: `reset_with_lang` + per-chunk re-prompt at caller boundary (BREAK) | T05,T03 | L |
| 3 | T15 | M10: `StreamingTranscriber` trait + uniform reset + EOU public reset | T09,T04 | M |
| 3 | T16 | M16: word timestamps on streaming Nemotron/EOU (additive) | T12 | M |
| 4 | T17 | M15: `Language` newtype, `Auto` first-class | T14,T15 | M |
| 4 | T18 | M14: constructor/config unification + 3×`ModelConfig` rename + timestamp dedup | T15 | M |
| 4 | T19 | M17: Cohere timestamp/diarize toggles via `CohereOptions` | T08,T18 | S |
| 4 | T20 | M19: EP `Auto` + `compiled_providers()` + EP-matrix docs | — | S |
| 4 | T21 | M18a: concurrency DOCS (spawn_blocking, bounded concurrency, one-session-per-stream) | — | S |
| 4 | T22 | G-A: criterion benchmark harness (RTF per variant) | — | M |
| 4 | T23 | M3/F1: incremental mel (**defer unless T22 proves ROI**; correctness-sensitive) | T03,T22 | L |
| 5 | T24 | G-B: migration guide / CHANGELOG; batch breaks into 0.4.0 | all breaks | S |
| 5 | T25 | M7: EOU processed-sample cursor + chunk-size validation | T03 | M |
| 5 | T26 | NIT batch: beam-search no-op, glob export, cap unify+log, doc known limits | — | S |
| 5 | T27 | M1c: auto-continuous re-detect (OPTIONAL, caller-VAD-driven) | T14 | L |
| 5 | T28 | G-F: coverage (`cargo-llvm-cov`) + MSRV (optional) | T01 | S |

**Quick-wins (do first, additive/zero-risk):** T05, T07, T06, T01, T08, T20, T21, T26.
**Deep (golden-gated / behavior-changing):** T12, T14, T23 (T10, T13 smaller but gated).
**Only strictly-serial chain:** Wave 2 (T10→T11→T12→T13). Wave 1 tasks are mutually independent after Wave 0.

## Dropped / corrected by Wave V (do NOT build)

- **Mel-accuracy epic (old M12):** Hann window already matches NeMo (symmetric) — changing it would
  *introduce* a regression; the `x.max(0)` log floor is a math no-op; multilingual normalization matches.
  Keep only an *optional* EN normalize-dump verification fixture. **Implementor: do not touch
  Hann/log-floor/normalization to "fix accuracy."**
- **"Remove the Mutex" (old M18b):** ort rc.12 `Session::run*` take `&mut self` — the Mutex is necessary.
  Ship concurrency *docs* (T21) instead. Re-open only if ort is bumped to expose a `&self` run.

## Surviving risks

R1 the language re-prompt is mechanism-verified by code+inference, not a live run — **T03's code-switch
golden is the live proof, T14 gates on it** (cold-start mis-commit may persist). R2 M5 changes output on
ties/NaN (rare, isolated in T10). R3 M2-decoder knob count could erode ROI — stop and keep variants
separate if knobs exceed ~6. R4 F3 panics on non-contiguous input — needs a standard-layout assert. R5
streaming==offline not byte-equal until T09.

## Open questions needing a human decision (before the breaking work)

1. **Semver:** batch all 0.x breaks into one **0.4.0** (recommended) or stagger across minor bumps?
2. **Consolidation appetite:** always do T11 + T13; treat **T12 as conditional** on the knob count
   staying low (corrected savings are ~350–550 lines, not 700–900).
3. **Language v1 scope:** ship `detected_language()` (T05) now regardless; **gate `reset_with_lang`
   (T14)** on R1's golden going green and your sign-off — it's the headline fix and the highest-risk change.
4. **Incremental mel (T23):** defer until T22's benchmark proves the O(n²) actually bites your workloads.

## Hand-off

Next: `/orchestrator-implementor improve-parakeet-rs` (same run folder). Start at **Wave 0 (T01–T03)** —
the goldens are the precondition for everything downstream. Resolve open questions Q1/Q3 before the
breaking PRs (T04, T08, T14).
