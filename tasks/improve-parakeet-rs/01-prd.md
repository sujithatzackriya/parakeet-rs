# 01 - PRD lane: parakeet-rs crate-wide improvement initiative

**/think framework(s):** jobs-to-be-done (primary, to anchor every theme on a real developer job) + first-principles (to keep success metrics library-appropriate rather than borrowing app-level KPIs). Inversion was applied to enumerate Non-Goals; socratic self-questioning surfaced the open questions.

**PRD artifact:** `/Users/sujith/work/2026/exps/parakeet-rs/tasks/prd-parakeet-rs-improvement.md`

This lane frames the initiative as a ranked improvement ROADMAP across four user-prioritized themes plus two supporting themes, NOT a single feature. PLAN ONLY.

---

## Goals (summary)

- **G1 Correctness & streaming:** resolve/document the multilingual language-lock; guarantee reset/flush/state-carry correctness on every stateful variant.
- **G2 Performance & RTF:** quantify and reduce per-chunk hot-path cost; keep streaming RTF < 1.0 on CPU; deliver a concurrency model.
- **G3 API & ergonomics:** consistent surface across 10 variants with deliberate, tagged semver impact.
- **G4 Model coverage:** per-variant feature matrix (timestamps, diarization, quantization, language re-detect); close library-fixable gaps, document inherent ones.
- **G5 Code quality:** reduce `model_X` + wrapper duplication without breaking the public API.
- **G6 Tests & CI:** regression guards for currently-unguarded correctness invariants (only 6/24 modules tested today).
- **G7 Output:** one ranked, deduplicated one-PR task backlog.

## Themes as epics

EP-01 Correctness & streaming - EP-02 Performance & RTF - EP-03 API consistency - EP-04 Model coverage & features - EP-05 Code quality/structure (supporting) - EP-06 Tests & CI (supporting). Each maps 1:1 to an audit lane (A1, A2, A3, A4, A5, A6) with RU feeding A1/A4.

## Success metrics (LIBRARY-appropriate, not Meetily KPIs)

- SM-1 semver discipline: 100% of API changes tagged additive-vs-breaking; zero undocumented breaks.
- SM-2 streaming correctness: every stateful variant has enumerated reset/flush/state-carry invariants + proposed regression tests.
- SM-3 accuracy vs reference: language-lock resolved with evidence; measurable follow-the-language acceptance if fixable, documented if inherent.
- SM-4 build across features: task to verify build across all 11 EP flags + multitalker/cohere/sortformer.
- SM-5 test coverage: raise from 6/24 modules toward guarding all named invariants.
- SM-6 roadmap completeness: all 6 themes represented; every HIGH/CRITICAL finding survives adversarial verification (85).
- SM-7 no regression: protects English-only Nemotron path and every shipped variant.

## Non-goals (inversion)

No implementation (plan only); no rewrite from scratch; no removal of the multi-variant ONNX ASR core value or any shipped variant/entry point; no model training/fine-tuning/re-export; no new model families; no app-level concerns (Meetily KPIs/UI/Tauri); no forced 1.0 stabilization (stay 0.x, breaks allowed but deliberate); no per-EP runtime certification; no simultaneous multi-language output guarantee unless the model supports it.

## Open questions (gating)

- OQ-1 language-lock: model-inherent vs library-fixable? (A1, A4)
- OQ-2 upstream NeMo language conditioning / per-utterance re-detect? (RU; gates A1/A4 per FR-4)
- OQ-3 perf hot paths + missing async/threading guidance? (A2)
- OQ-4 duplication volume across the 10 pairs? (A5, 02)
- OQ-5 shared abstraction extractable without breaking public API? (A5, 02, A3)
- OQ-6 which variants lack timestamps/diarization/quantization; inherent vs library? (A4)
- OQ-7 which correctness invariants are unguarded? (A6)
- OQ-8 semver target: stay 0.x vs path to 1.0? Default: stay 0.x. (A3)
- OQ-9 split `auto` into detect-once vs detect-continuous public modes? Depends on OQ-1. (A3, A4)

---

## Cross-references for other lanes

- **02-architecture:** baseline is per-variant `model_X.rs` + wrapper (NG-2); quantify shared infra vs duplication; constrain abstractions against `lib.rs:76-99`. Feeds OQ-4/OQ-5.
- **A1-correctness:** owns language-lock + streaming reset/flush/state-carry; resolve OQ-1 against code AND RU before concluding (FR-4); protect EN Nemotron path (SM-7).
- **A2-performance:** OQ-3; quantify mel/realfft, ndarray copies, ort run, tokenizer; deliver concurrency model for `&mut self` blocking `transcribe_chunk`.
- **A3-api:** every change carries semver tag (FR-5, SM-1) under 0.x (OQ-8); owns OQ-9; cross-check EP-05 abstraction (OQ-5).
- **A4-models:** OQ-6 feature matrix; classify gaps inherent vs library; co-owns language-lock with A1 and OQ-9 with A3.
- **A5-quality:** OQ-4/OQ-5; quantify duplication; propose public-API-preserving shared abstraction.
- **A6-tests:** OQ-7/SM-5; map unguarded invariants; fixtures via `./nemotron`, `./nemotron_multi`.
- **RU-upstream:** OQ-2; authoritative answer gating A1/A4 (FR-4).
- **80/85/90:** enforce FR-2/FR-3/FR-7; HIGH/CRITICAL findings must survive refutation (85) before the ranked backlog (90); success = SM-1..SM-7.
