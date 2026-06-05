# Changelog

## 0.4.0

A correctness, API-consistency, and performance pass across the crate. All
breaking changes are 0.x and batched into this one release; see the migration
notes under each item.

### Breaking changes

- **Low-level model types are no longer public.** `ParakeetModel`,
  `ParakeetEOUModel`, `NemotronModel` / `NemotronEncoderCache` /
  `NemotronModelConfig`, `ParakeetUnifiedModel` / `UnifiedModelConfig`,
  `ParakeetDecoder`, and `SentencePieceVocab` were accidental exports and are now
  crate-internal.
  *Migration:* use the high-level wrappers (`Parakeet`, `ParakeetTDT`, `Nemotron`,
  `ParakeetEOU`, `ParakeetUnified`, `MultitalkerASR`, `CohereASR`).
- **`Error` is now `#[non_exhaustive]` and derives `thiserror::Error`.**
  `Error::source()` now returns the underlying cause for I/O and ONNX-Runtime
  errors. A new `Error::ModelFormat` variant is returned when a loaded ONNX graph
  is missing an expected input/output name.
  *Migration:* add a wildcard arm (`_ =>`) when exhaustively matching `Error`.
- **Decoder argmax is unified to one first-wins, finite-guarded policy.** Output
  can differ only on exact logit ties or non-finite logits (pathological; real
  transcripts are unchanged, verified by golden tests).
- **Language switching / reset semantics.** `set_target_lang` now re-prompts the
  model mid-stream, and a new `reset_with_lang` resets the decoder state at a
  language boundary while preserving the encoder cache. `reset()` still preserves
  the configured language.
  *Migration:* to switch language mid-stream, call `reset_with_lang(lang)` at an
  utterance boundary (see Known limitations).
- **`StreamingTranscriber` trait + stream-method aliases.** Streaming variants now
  share a `StreamingTranscriber` trait (associated `Output` type). `ParakeetEOU`
  gained a public `reset()` and a canonical `transcribe_chunk`; its `transcribe`
  (with the `reset_on_eou` flag) is retained.
  *Migration:* prefer `transcribe_chunk`; old inherent methods still work.
- **Typed `Language` with first-class `Auto`.** `set_target_lang` /
  `reset_with_lang` take `impl Into<Language>`; existing `&str` codes still work
  via `From<&str>` and resolve to the same prompt index. `detected_language_typed()`
  returns `Option<Language>` (`detected_language() -> Option<String>` is unchanged).
- **Config/type renames.** `execution::ModelConfig` -> `execution::ExecutionConfig`
  and `config::ModelConfig` -> `config::ModelConfigJson` (the public export names
  `ExecutionConfig` / `ModelConfigJson` are unchanged). `Sortformer`'s constructor
  now takes the shared `ExecutionConfig`. The execution-config constructor param is
  uniformly named `exec_config`. `multitalker::WordTimestamp` is removed:
  `SpeakerTranscript.words` is now `Vec<TimedToken>`.
  *Migration:* `WordTimestamp { word, start_secs, end_secs }` ->
  `TimedToken { text, start, end }`.

### Added

- **Nemotron word-level timestamps:** `get_timed_transcript(TimestampMode)` (streaming
  and offline), reusing the shared `group_by_words` machinery.
- **Nemotron language observability:** `detected_language()` / `detected_language_typed()`
  surface the model's in-band `<lang>` tag.
- **Mid-stream language control:** `reset_with_lang()` plus auto re-detection under
  `Language::Auto`.
- **`flush()` on Nemotron** to emit the final buffered chunk (no more dropped tail).
- **`StreamingTranscriber` trait** unifying the streaming variants.
- **Typed `Language`** with first-class `Auto` and an `Other` escape hatch.
- **`CohereOptions`** builder exposing Cohere's timestamp / diarize toggles.
- **Execution-provider discovery:** `ExecutionProvider::compiled()` / `auto()` and
  `ExecutionConfig::compiled_providers()` / `with_auto_provider()`, with an EP matrix doc.
- **Criterion benchmark harness** (`benches/transcribe.rs`) for latency / RTF.
- **EOU chunk-size validation** and a public `reset()`.
- **Live-microphone example** (`examples/streaming_mic.rs`) with live language-ID output.

### Fixed

- Nemotron no longer drops the final partial chunk; streaming output now byte-matches
  offline transcription.
- EOU silently duplicated/dropped tokens on off-size chunks; off-size chunks are now
  rejected with a structured error.
- The crate-root rustdoc quick-start example now compiles (`cargo test --doc`).
- Feature-gated modules (`sortformer`, `multitalker`, `cohere`) and the cohere example
  are now built in CI, so they cannot silently break.

### Internal / performance

- Shared ONNX session construction and `resolve_onnx_file` (with a `Quantization`
  preference knob); shared RNNT decoder step with zero-copy `TensorRef` inputs
  (removes ~7.7 MB/chunk of tensor clones); a single cached FFT plan and one
  parameterized mel front-end across the streaming variants. All verified
  numerics-identical against golden transcripts.
- Measured RTF ~2.7x faster than realtime offline (~221 ms per 560 ms chunk) on
  Apple-silicon-class CPU.

### Known limitations

- **Multilingual `auto` code-switch is bounded by the model.** Under `Language::Auto`
  the checkpoint commits to one language per utterance and does not flip its in-band
  language ID mid-stream on out-of-distribution audio; switching requires caller-driven
  boundaries via `reset_with_lang()`. True mid-word code-switching is not supported by
  the model architecture.
