# Context Pack — Crate-wide improvement of parakeet-rs

> Every agent reads this FIRST. Shared ground truth. Do not re-derive the objective.

## Objective

Produce a **full, exhaustive, ranked improvement roadmap** for the `parakeet-rs` crate
(`/Users/sujith/work/2026/exps/parakeet-rs`, v0.3.6) across FOUR dimensions the user prioritized:
1. **Correctness & streaming** (incl. the observed multilingual language-lock)
2. **Performance** (inference/RTF, allocations, threading, GPU/execution providers)
3. **API & ergonomics** (public surface, variant consistency, back-compat, errors, docs/examples)
4. **Model coverage & features** (variant unification, timestamps, diarization, quantization, language re-detection)

Depth: **full deep audit, exhaustive.** This is a LIBRARY improvement initiative (a published Rust
crate), NOT an app integration. Output is a plan package + ranked one-PR task backlog; PLAN ONLY.

## Seed finding (the trigger, now one item among many)

Live mic test of the multilingual Nemotron 3.5 (`./nemotron_multi`, `target_lang="auto"`): transcript
began as a garbled EN/ZH/HI mix, then "locked" onto Hindi and stayed there even when the speaker
returned to English. PARTIALLY INVESTIGATED (orchestrator, before scope broadened):
- `"auto"` is **prompt index 101** in `PROMPT_DICTIONARY` (`nemotron.rs:48`) — a real model prompt
  slot, fed to `run_encoder(..., self.prompt_index)` on EVERY chunk (`nemotron.rs:591,691`).
- `prompt_index` is set once via `set_target_lang` (`nemotron.rs:475-491`) and is **never changed
  between chunks** by the library; `reset()` (`nemotron.rs:495-509`) explicitly PRESERVES the target
  language and only clears encoder/decoder/audio state.
- The decoder is autoregressive: `last_token` carries across chunks; encoder has a persistent
  `NemotronEncoderCache` + left context. So both the prompt conditioning AND the carried decoder
  state bias toward the first-committed language.
- OPEN: is the lock inherent to the ONNX graph (fixed prompt_index conditioning) or a fixable library
  choice (e.g. re-detect + re-prompt at silence boundaries, expose a re-detect API)? The
  correctness/model lanes must resolve this against code + the upstream reference.

## Architecture facts (verified — cite these; re-verify before relying)

Crate is ~7,766 lines across 24 `src/*.rs` modules. Pattern: most model families have a low-level
`model_X.rs` (ONNX session + encoder/decoder runs) + a high-level wrapper (`X.rs`).

- **Public API** (`lib.rs:76-99`): `Parakeet` (CTC), `ParakeetTDT`, `Nemotron`/`NemotronHandle`/
  `NemotronMode`, `ParakeetEOU`/`ParakeetEOUHandle`, `ParakeetUnified`/`...Handle`/`UnifiedStreamingConfig`,
  `MultitalkerASR` (feature `multitalker`), `CohereASR` (feature `cohere`), sortformer (feature
  `sortformer`); shared: `ExecutionProvider`/`ExecutionConfig` (`execution.rs`), `TimestampMode`,
  `FeatureCache`, decoder/model/config/vocab types.
- **Model variants** (10): CTC (`parakeet.rs`/`model.rs`), TDT (`parakeet_tdt.rs`/`model_tdt.rs`/
  `decoder_tdt.rs`), EOU (`parakeet_eou.rs`/`model_eou.rs`), Nemotron (`nemotron.rs` 786L /
  `model_nemotron.rs` 288L), Unified (`parakeet_unified.rs` 590L /`model_unified.rs`), Multitalker
  (`multitalker.rs` 771L /`model_multitalker.rs`), Cohere (`cohere.rs`/`model_cohere.rs`), Sortformer
  diarization (`sortformer.rs` 1254L). Shared: `audio.rs` (mel/feature, 332L), `timestamps.rs` (359L),
  `decoder.rs`, `transcriber.rs`, `execution.rs`, `vocab.rs`, `config.rs`, `error.rs`.
- **Streaming**: cache-aware (Nemotron `NemotronEncoderCache`, EOU, unified `UnifiedStreamingConfig`).
  Chunk sizes differ per model (Nemotron 8960/560ms, EOU 2560/160ms). Stateful `&mut self`
  `transcribe_chunk`.
- **Execution providers** (`Cargo.toml` features): cpu (default) + cuda, tensorrt, coreml, directml,
  migraphx, openvino, webgpu, nnapi; load-dynamic / preload-dylibs. ort `2.0.0-rc.12`,
  default-features=false (std, ndarray, api-24). Other deps: ndarray 0.17, tokenizers 0.23.1
  (onig), realfft 3, hound 3.5, eyre 0.6, serde/serde_json. dev-dep: cpal 0.15 (mic example).
- **Tests**: NO `tests/` directory. Inline `#[cfg(test)]` only in `timestamps.rs`, `cohere.rs`,
  `sortformer.rs`, `audio.rs`, `parakeet_unified.rs`, `decoder_tdt.rs` (6 of 24 modules). Sparse —
  a likely finding for the tests lane.
- **Examples** (9): `streaming.rs` (WAV), `streaming_mic.rs` (live mic, added this session),
  `streaming_diarization.rs`, `diarization.rs`, `multitalker.rs`, `unified.rs`, `cohere.rs`,
  `shared_model.rs`, `raw.rs`.
- **Export scripts** (`scripts/`): per-model ONNX export from NeMo (`export_nemotron_streaming_*.py`,
  `export_parakeet_unified.py`, `export_realtime_eou_120m.py`, `export_diar_sortformer.py`,
  `export_multitalker.py`). Relevant for understanding how prompt_index / graph inputs are conditioned.
- Models downloaded locally: `./nemotron` (EN, 2.3G), `./nemotron_multi` (multilingual 3.5, 2.4G).

## PLAN MAP (who writes what)

Wave R runs audit lanes + framing lanes in parallel; each writes its own file and ends with
`## Cross-references for other lanes`. Audit lanes emit concrete findings (severity + file:line +
recommendation). Framing lanes set goals/architecture.

| id | lane | file | wave |
|----|------|------|------|
| 01 | PRD framing (runs `/prd --sub-skill`) | `01-prd.md` | R |
| 02 | Architecture map (current crate, shared infra, duplication) | `02-architecture.md` | R |
| A1 | AUDIT: correctness & streaming (incl. language-lock) | `A1-correctness.md` | R |
| A2 | AUDIT: performance (hot paths, allocs, threading, EP/GPU) | `A2-performance.md` | R |
| A3 | AUDIT: API & ergonomics (surface, variant consistency, back-compat, docs) | `A3-api.md` | R |
| A4 | AUDIT: model coverage & features (unification, timestamps, diarization, lang re-detect) | `A4-models.md` | R |
| A5 | AUDIT: code quality & structure (duplication, idiom, error handling, module layout) | `A5-quality.md` | R |
| A6 | AUDIT: tests & CI (coverage gaps, fixtures, regression guards) | `A6-tests.md` | R |
| RU | RESEARCH: upstream NVIDIA/NeMo reference (language conditioning, cache-aware streaming) | `RU-upstream.md` | R |
| 80 | Coherence + feasibility (dedup findings, coverage) | `80-feasibility.md` | A |
| 85 | Adversarial verification (refute every HIGH/CRITICAL finding) | `85-verify.md` | V |
| 90 | Synthesis: ranked roadmap + ordered one-PR task backlog | `90-synthesis.md` | S |
| —  | Final plan package | `FINAL-plan.md` | orchestrator |

## Constraints

- **PLAN ONLY. Write NO code.** Implementation is `orchestrator-implementor` / `/tdd-workflow`.
- Quality gates are LIBRARY-appropriate (the 6 Meetily KPIs DO NOT apply here): public API
  stability / semver back-compat, streaming correctness, transcription accuracy vs reference, no
  regression to the English-only Nemotron path or other shipped variants, build across feature flags,
  test coverage.
- **No em dashes** anywhere. Never read `.env`/`.pem`/`.p8`/`.key`.
- Every finding cites `file:line` (or a clearly-labelled assumption / upstream source). No evidence →
  it is a hypothesis; label it.
- Each agent picks its OWN `/think` framework(s); state them in one line at the top.
- Audit findings use the finding schema (SEVERITY CRITICAL|HIGH|MEDIUM|LOW|NIT, root cause, evidence,
  impact, regression risk of the fix, recommendation, reverification).
- Be honest about model-inherent limits vs fixable library choices (esp. the language-lock).
