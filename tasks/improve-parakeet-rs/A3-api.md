# A3 — AUDIT: API & ergonomics

**/think frameworks:** first-principles (what is the minimal consistent contract a user must learn?) +
opportunity-cost (this is 0.x; every breaking change is cheap now and expensive after 1.0, so rank by
"cost of fixing later"). Inversion as a cross-check (how would a user mis-use this API?).

**Scope:** public surface in `lib.rs:76-99` and each variant wrapper. PLAN ONLY, no code.

---

## Summary of the public contract (as shipped, v0.3.6)

| Variant | Constructor(s) | Streaming method | Offline method | Reset | Accumulated getter | Return type | Trait |
|---|---|---|---|---|---|---|---|
| `Parakeet` (CTC) | `from_pretrained(path, Option<ExecutionConfig>)` | — | `transcribe_samples/file/file_batch` | — | — | `TranscriptionResult` | `Transcriber` |
| `ParakeetTDT` | `from_pretrained(path, Option<ExecutionConfig>)` | — | `transcribe_samples/file/file_batch` | — | — | `TranscriptionResult` | `Transcriber` |
| `ParakeetEOU` | `from_pretrained` / `from_shared(&Handle)`; `Handle::load(path, Option<ExecutionConfig>)` | `transcribe(chunk, reset_on_eou: bool)` | — | private `reset_states` only | — | `String` | none |
| `Nemotron` | `from_pretrained` / `from_shared(&Handle)`; `Handle::load(path, Option<ExecutionConfig>)` | `transcribe_chunk(chunk)` | `transcribe_audio/file` | `reset()` | `get_transcript() -> String` | `String` | none |
| `ParakeetUnified` | `from_pretrained` / `from_pretrained_with_streaming_config` / `from_shared` / `from_shared_with_streaming_config`; `Handle::load` | `transcribe_chunk(chunk)` + `flush()` | `transcribe_audio/file/samples` | `reset()` | `get_transcript() -> String`, `get_timed_transcript(mode) -> TranscriptionResult` | `String` (stream) / `TranscriptionResult` | `Transcriber` |
| `MultitalkerASR` | `from_pretrained(asr_dir, sortformer_path, Option<ExecutionConfig>)` (TWO paths, no handle) | `transcribe_chunk(chunk)` | `transcribe_audio_multitalker/file_multitalker` | `reset()` | `get_transcripts() -> Vec<SpeakerTranscript>` | `Vec<SpeakerTranscript>` | none |
| `CohereASR` | `from_pretrained(path, Option<ExecutionConfig>)` | — | `transcribe_audio(audio, language, punctuation, itn)` | — | — | `String` | none |
| `Sortformer` (diarization) | `new(path)` / `with_config(path, Option<ModelConfig>, DiarizationConfig)` | `diarize_chunk` / `feed` + `flush` | `diarize(audio, sr, ch)` | `reset_state()` | — | `Vec<SpeakerSegment>` | none |

Evidence: `parakeet.rs:40-43,130-167`; `parakeet_tdt.rs:28-31,92-99`; `parakeet_eou.rs:59,93-98,105,137,247`;
`nemotron.rs:337-340,400-405,418,475,495,513,524,540,615`; `parakeet_unified.rs:148,197-228,261,272-310,565`;
`multitalker.rs:218-222,251,301,325,450,476`; `cohere.rs:169-172,253-259`; `sortformer.rs:212-217,282,294,356,428,477`;
`transcriber.rs:8-72`; `lib.rs:76-99`.

The shape of this table IS the central finding: **there is one shared trait (`Transcriber`) used by only 3 of 8
entry points, and the other contracts are bespoke per variant.** A user who learns one variant cannot transfer that
knowledge to the next.

---

## FINDINGS (ranked)

### A3-1 — No common abstraction across streaming variants; each is bespoke
**SEVERITY: HIGH**
- **Root cause:** `Transcriber` (`transcriber.rs:8`) only models *offline file/sample* transcription and is implemented
  by `Parakeet`, `ParakeetTDT`, `ParakeetUnified` (`parakeet.rs:130`, `parakeet_tdt.rs:92`, `parakeet_unified.rs:565`).
  The streaming variants (`Nemotron`, `ParakeetEOU`, `ParakeetUnified`, `MultitalkerASR`) each invent their own
  streaming method name, signature, reset, and accumulator. There is no `StreamingTranscriber` trait at all.
- **Evidence:** streaming entry points differ in *name and signature*:
  - `Nemotron::transcribe_chunk(&[f32]) -> Result<String>` (`nemotron.rs:615`)
  - `ParakeetUnified::transcribe_chunk(&[f32]) -> Result<String>` + separate `flush()` (`parakeet_unified.rs:303,308`)
  - `ParakeetEOU::transcribe(&[f32], reset_on_eou: bool) -> Result<String>` — different verb, extra bool, no flush
    (`parakeet_eou.rs:137`)
  - `MultitalkerASR::transcribe_chunk(&[f32]) -> Result<Vec<SpeakerTranscript>>` — same name, different return
    (`multitalker.rs:325`)
  - `Sortformer::diarize_chunk` / `feed` / `flush` — yet another vocabulary (`sortformer.rs:356,428,477`)
- **Impact:** the library cannot be consumed generically (no `Box<dyn StreamingTranscriber>`); downstream code that
  wants to swap models must rewrite the integration per variant. This is the single biggest ergonomics tax and the
  root of findings A3-2 through A3-6.
- **Regression risk of fix:** MEDIUM. Introducing a `StreamingTranscriber` trait is *additive* (existing inherent
  methods can stay), so it need not break callers; the risk is in *also* renaming inherent methods (see A3-2).
- **Recommendation:** define a `StreamingTranscriber` trait: `fn transcribe_chunk(&mut self, &[f32]) -> Result<T>`,
  `fn flush(&mut self) -> Result<T>`, `fn reset(&mut self)`, `type Output`. Implement for all streaming variants.
  `flush` can default to `Ok(Output::default())` for models that need no flush. Keep the existing offline `Transcriber`.
- **Reverification:** confirm `MultitalkerASR`/`Sortformer` outputs (`Vec<...>`) fit an associated `Output` type; confirm
  `ParakeetEOU`'s `reset_on_eou` semantics can move to a config field so the chunk signature is uniform.

---

### A3-2 — Streaming method names are inconsistent (`transcribe` vs `transcribe_chunk` vs `feed` vs `diarize_chunk`)
**SEVERITY: HIGH**
- **Root cause:** organic growth; each wrapper picked its own verb.
- **Evidence:** `ParakeetEOU::transcribe` (`parakeet_eou.rs:137`) vs `transcribe_chunk` everywhere else
  (`nemotron.rs:615`, `parakeet_unified.rs:303`, `multitalker.rs:325`); `Sortformer::feed` aliases `diarize_chunk`
  (`sortformer.rs:428` vs `356`).
- **Impact:** every variant requires re-reading docs; `feed` and `diarize_chunk` doing the same thing
  (`sortformer.rs:428-431` literally documents `feed` as an alias) is gratuitous surface.
- **Regression risk of fix:** HIGH if done by rename (breaks callers). LOW if done by adding the canonical name and
  `#[deprecated]`-aliasing the old.
- **Recommendation:** standardise on `transcribe_chunk`. For 0.x, rename now and document in CHANGELOG; or add the
  canonical name + `#[deprecated]` alias and drop the alias at 1.0. Remove the `feed` alias (`sortformer.rs:428`).
- **Reverification:** grep examples/ for each old name before renaming.

---

### A3-3 — `reset` is inconsistent: name, visibility, and semantics all differ
**SEVERITY: HIGH**
- **Root cause:** no shared streaming contract (A3-1), so each variant defined reset ad hoc.
- **Evidence:**
  - `Nemotron::reset()` is public and **preserves target language** (`nemotron.rs:495-509`).
  - `ParakeetUnified::reset()` is public, full reset (`parakeet_unified.rs:261`).
  - `MultitalkerASR::reset()` is public (`multitalker.rs:251`).
  - `Sortformer::reset_state()` — different name (`sortformer.rs:282`).
  - `ParakeetEOU` has **no public reset** at all; only private `reset_states()` triggered via the `reset_on_eou`
    bool (`parakeet_eou.rs:247`). A user cannot manually start a new utterance.
- **Impact:** "start a new utterance" is the most common streaming operation and it is spelled four different ways,
  one of them unreachable. The Nemotron "reset preserves language" surprise is also the documented seed of the
  language-lock bug (context pack lines 24-29) and is a correctness/API overlap — see cross-ref to A1.
- **Regression risk of fix:** LOW-MEDIUM. Adding a public `reset` to EOU is additive; renaming `reset_state` is
  breaking (deprecate-alias).
- **Recommendation:** uniform `fn reset(&mut self)` on every streaming variant (part of the A3-1 trait). Add public
  `reset` to `ParakeetEOU`. Decide and DOCUMENT one semantics: "reset clears all per-stream state". If language must
  persist across reset, expose that as a separate, explicit method (e.g. `reset_keep_language()`), not as a silent
  default (this is the API half of the language-lock issue).
- **Reverification:** confirm EOU's soft-reset (encoder cache + buffer preserved on EOU, `parakeet_eou.rs:248-254`) is
  a *deliberate* streaming behaviour distinct from a hard `reset`, and name them distinctly.

---

### A3-4 — Stream return types diverge: `String` vs `TranscriptionResult` vs `Vec<SpeakerTranscript>`; no timestamps in most streaming paths
**SEVERITY: MEDIUM**
- **Root cause:** wrappers return whatever was convenient.
- **Evidence:** streaming returns `String` (`nemotron.rs:615`, `parakeet_eou.rs:137`, `parakeet_unified.rs:303`) but
  `MultitalkerASR` returns `Vec<SpeakerTranscript>` (`multitalker.rs:325`). Only `ParakeetUnified` exposes a *timed*
  streaming accumulator (`get_timed_transcript`, `parakeet_unified.rs:272`); `Nemotron`/`EOU` give text-only
  `get_transcript` or nothing, even though TDT-style models can emit frame indices.
- **Impact:** caller cannot get word timestamps from streaming Nemotron/EOU without reimplementing alignment; a
  generic consumer cannot assume a return shape.
- **Regression risk of fix:** MEDIUM. Changing return types is breaking; adding timed getters is additive.
- **Recommendation:** at minimum add `get_timed_transcript(mode)` to `Nemotron` and `ParakeetEOU` (matching
  `parakeet_unified.rs:272`). Long term, make the A3-1 trait `Output` a structured type with text + optional timing.
- **Reverification:** confirm Nemotron's RNNT decode loop (`nemotron.rs:723-770`) discards frame indices that would be
  needed for timestamps; if so this is a feature gap, cross-ref A4.

---

### A3-5 — `Error` enum is stringly-typed and loses structured context
**SEVERITY: MEDIUM**
- **Root cause:** four of six variants carry only `String` (`error.rs:5-13`).
- **Evidence:** `Audio(String)`, `Model(String)`, `Tokenizer(String)`, `Config(String)` (`error.rs:9-12`). Distinct
  failure classes are flattened into `Config`: missing files (`parakeet.rs:60`, `cohere.rs:180`), unknown language
  (`nemotron.rs:485`), invalid streaming config (`parakeet_unified.rs:53-66`), and `serde_json` errors are *also*
  mapped to `Config` (`error.rs:45-48`). Mutex poisoning is stringified into `Model` on every chunk
  (`nemotron.rs:584`, `parakeet_unified.rs:344`, `parakeet_eou.rs:172`).
- **Impact:** callers cannot match on "file not found" vs "bad language code" vs "poisoned lock" — they must string-match,
  which is brittle. No `#[non_exhaustive]`, so adding a variant later is itself a breaking change.
- **Regression risk of fix:** MEDIUM. Restructuring `Error` is breaking but high-value at 0.x.
- **Recommendation:** keep `thiserror`-style `enum` (no need for `eyre` in a library — `eyre` is application-grade and
  the crate already correctly uses a concrete `Error`; do NOT adopt `eyre` for the public API). Add structured
  variants: `NotFound { path }`, `UnsupportedLanguage { lang, supported }`, `InvalidConfig { field, reason }`,
  `LockPoisoned`. Mark `#[non_exhaustive]`. Consider `#[from]` for `serde_json`/`hound` instead of stringifying.
  Adopting the `thiserror` derive macro would also remove the hand-written `Display`/`From` boilerplate
  (`error.rs:15-55`) — cross-ref A5.
- **Reverification:** confirm `ort::Error` is already structured and preserved (`error.rs:36-43`) — it is; only the
  hand-rolled string variants need work.

---

### A3-6 — Language selection is stringly-typed; `set_target_lang` is fallible and error-prone
**SEVERITY: MEDIUM**
- **Root cause:** language passed as `&str` and validated at runtime against a flat table.
- **Evidence:** `Nemotron::set_target_lang(&str) -> Result<()>` looks up `PROMPT_DICTIONARY` and errors on unknown
  codes (`nemotron.rs:475-491`); `CohereASR::transcribe_audio(.., language: &str, ..)` validates against
  `SUPPORTED_LANGUAGES` per call (`cohere.rs:253-270`). The accepted-key set differs between the two
  (Nemotron ~130 keys with aliases incl. `"auto"`, Cohere 14 ISO codes), and there is no shared `Language` type.
- **Impact:** typos are runtime errors, not compile errors; the two variants disagree on accepted spellings
  (`"en-US"` vs `"en"`); `available_languages()` (`nemotron.rs:386`) returns `Vec<&'static str>` while Cohere returns
  `Vec<String>` (`cohere.rs:388`) — even the discovery API is inconsistent.
- **Regression risk of fix:** MEDIUM. A `Language` enum is breaking but improves discoverability and IDE
  autocomplete. The 130-entry Nemotron table with regional aliases is the hard part — an enum may be too rigid for
  experimental locales; an alternative is a `Language` newtype with associated consts + `from_iso` fallible parse.
- **Recommendation:** introduce a `Language` newtype (or enum for the documented 40 locales + an `Other(&str)` escape
  hatch) shared by Nemotron and Cohere. Keep a `&str`-accepting convenience that delegates. Critically: distinguish
  `Language::Auto` (Nemotron prompt 101) from a fixed language *in the type system*, since "auto" is the seed bug.
  Cross-ref A1/A4: whether "auto" should re-detect mid-stream is a correctness question, but the *API* should at least
  make "auto" a first-class, visible state rather than a magic string.
- **Reverification:** confirm Cohere has no `Auto` (it requires explicit language, `cohere.rs:264`) — the shared type
  must allow per-variant capability differences.

---

### A3-7 — Constructor patterns are inconsistent (handle vs no-handle, param names, extra path args)
**SEVERITY: MEDIUM**
- **Root cause:** the shared-model/handle pattern was added to some variants and not others.
- **Evidence:**
  - Handle pattern present: `Nemotron`/`NemotronHandle`, `ParakeetEOU`/`...Handle`, `ParakeetUnified`/`...Handle`
    (`nemotron.rs:337/418`, `parakeet_eou.rs:59/105`, `parakeet_unified.rs:148/219`).
  - Handle pattern ABSENT: `Parakeet`, `ParakeetTDT`, `CohereASR`, `MultitalkerASR` — these own their model directly
    (`parakeet.rs:11`, `parakeet_tdt.rs:14`, `cohere.rs:141`, `multitalker.rs:199`). So concurrent shared-model use is
    impossible for those four.
  - Constructor param name drift: `Parakeet::from_pretrained(path, config)` (`parakeet.rs:41`) vs
    `Nemotron::from_pretrained(path, exec_config)` (`nemotron.rs:401`) — same type `Option<ExecutionConfig>`, two names.
  - `MultitalkerASR::from_pretrained(asr_dir, sortformer_path, exec_config)` takes TWO paths (`multitalker.rs:218`),
    breaking the otherwise-universal `(path, Option<ExecutionConfig>)` shape.
  - `Sortformer` uses `new(path)` + `with_config(path, Option<ModelConfig>, DiarizationConfig)` — a *third* constructor
    convention, and it exposes the type under its internal name `ModelConfig` rather than the public alias
    `ExecutionConfig` (`sortformer.rs:212-217`; alias defined in `lib.rs:77`).
- **Impact:** no learnable constructor rule; the `ExecutionConfig`-vs-`ModelConfig` exposure (A3-8) actively confuses.
- **Regression risk of fix:** MEDIUM (param rename is breaking; adding handles is additive).
- **Recommendation:** (1) standardise the param name to `config: Option<ExecutionConfig>` everywhere. (2) Decide
  whether the handle/shared-model pattern is universal; if it is the recommended concurrency story, add it to
  `Parakeet`/`ParakeetTDT`/`Cohere`/`Multitalker` (or document why streaming-only variants get it). (3) For
  `Sortformer`, re-export and use `ExecutionConfig`, not `ModelConfig`.
- **Reverification:** confirm `MultitalkerASR` genuinely needs two model dirs (ASR + Sortformer) — it does
  (`multitalker.rs:226-233`), so the two-path constructor is justified; only the param *name* and missing handle
  are findings.

---

### A3-8 — Type-name collisions across the public surface (`ModelConfig` x3, `from_pretrained` overloads, duplicate timestamp types)
**SEVERITY: MEDIUM**
- **Root cause:** three different structs are named `ModelConfig`, disambiguated only by re-export aliasing.
- **Evidence:** `lib.rs:77` re-exports `execution::ModelConfig as ExecutionConfig`; `lib.rs:84` re-exports
  `config::ModelConfig as ModelConfigJson`; `model_nemotron.rs` has `NemotronModelConfig`; `model_unified.rs` has
  `UnifiedModelConfig` (`lib.rs:89-90`). Internally every wrapper imports
  `use crate::execution::ModelConfig as ExecutionConfig` (`parakeet.rs:5`, `nemotron.rs:2`, etc.), but `sortformer.rs`
  exposes the un-aliased `ModelConfig` to users (A3-7). Timestamp types are also near-duplicates: `TimedToken`
  (`decoder.rs`, used everywhere) vs `WordTimestamp` (`multitalker.rs:58`, `{word, start_secs, end_secs}`) — two
  structs for "a word with start/end time".
- **Impact:** import confusion, docs.rs noise, and the `WordTimestamp`/`TimedToken` split means multitalker timestamps
  can't be fed to the shared `process_timestamps` helper (`timestamps.rs:43`).
- **Regression risk of fix:** LOW-MEDIUM. Renaming internal `ModelConfig` types is internal-only (not breaking).
  Merging `WordTimestamp` into `TimedToken` is breaking but small.
- **Recommendation:** rename `execution::ModelConfig` to `ExecutionConfig` *at the definition site* (drop the alias);
  rename `config::ModelConfig` to `ModelConfigJson` at its site. Replace `WordTimestamp` with the existing `TimedToken`
  (field rename `word->text`, `start_secs->start`); have `SpeakerTranscript.words: Vec<TimedToken>`.
- **Reverification:** confirm no external consumer relies on the `WordTimestamp` field names (only in examples/multitalker.rs).

---

### A3-9 — `CohereASR::transcribe_audio` uses positional `bool` flags instead of a config/options struct
**SEVERITY: LOW**
- **Root cause:** flags added inline as the model grew.
- **Evidence:** `transcribe_audio(&[f32], language: &str, punctuation: bool, itn: bool)` (`cohere.rs:253-259`) — two
  adjacent bools at the call site are easy to transpose (`transcribe_audio(a, "en", true, false)` reads ambiguously).
- **Impact:** call-site readability and accidental flag swaps.
- **Regression risk of fix:** LOW (breaking but tiny surface; Cohere is feature-gated).
- **Recommendation:** a small `CohereOptions { language, punctuation, itn }` with builder defaults, or at minimum a
  doc example. Cross-ref the broader "stringly-typed language" point (A3-6).
- **Reverification:** none needed.

---

### A3-10 — GPU opt-in friction: feature-gated EP enum variants + no discoverability of what's compiled in
**SEVERITY: LOW**
- **Root cause:** `ExecutionProvider` variants are `#[cfg(feature=...)]`-gated (`execution.rs:17-36`), so `Cuda` etc.
  simply do not exist unless the feature is enabled, and the default `ExecutionConfig` is always CPU
  (`execution.rs:69-78`).
- **Evidence:** to use GPU a user must (1) enable the right Cargo feature, (2) construct `ExecutionConfig::new()
  .with_execution_provider(ExecutionProvider::Cuda)`, (3) pass `Some(config)`. There is no runtime API to ask "is CUDA
  available?" or "what providers were compiled in?", and the gated variants make `match` over providers awkward
  downstream. The CoreML-slower-than-CPU and WebGPU-experimental caveats live only in a source comment
  (`execution.rs:7-15`), not in rustdoc.
- **Impact:** the most common performance question ("how do I turn on the GPU?") requires reading Cargo.toml + source.
- **Regression risk of fix:** LOW (additive helpers).
- **Recommendation:** add `ExecutionProvider::compiled_in() -> Vec<ExecutionProvider>` and a convenience
  `ExecutionConfig::gpu()` (picks the first compiled GPU EP) or `with_auto_gpu()`. Promote the EP caveats from the
  source comment (`execution.rs:7-15`) into `///` rustdoc on the variants so they appear on docs.rs. Document the
  feature-to-EP mapping in the README acceleration section.
- **Reverification:** confirm the gated variants compile-error cleanly when a user matches without the feature
  (they will; this is expected) — recommend a doc note.

---

### A3-11 — Rustdoc coverage is uneven and `missing_docs` is not enforced
**SEVERITY: LOW**
- **Root cause:** no `#![deny(missing_docs)]` / `#![warn(missing_docs)]` in `lib.rs` (grep confirms absent).
- **Evidence:** well-documented: `Nemotron`, `ParakeetEOU`, `CohereASR`, `MultitalkerASR`, `Transcriber`,
  `TimestampMode` (`nemotron.rs:290-300`, `transcriber.rs:1-72`, `cohere.rs:1-26`, `timestamps.rs:3-28`). Thin/absent:
  `Parakeet::model_dir/preprocessor_config` (`parakeet.rs:122-127`, no docs), `ParakeetUnified` public getters/`flush`
  (`parakeet_unified.rs:253-310`, mostly undocumented), `TranscriptionResult`/`TimedToken` fields
  (`decoder.rs:7-19`, comment-only not `///`), `UnifiedStreamingConfig` fields (`parakeet_unified.rs:25-30`).
  The crate-level docs (`lib.rs:18-27`) still show the *old single-arg* `from_pretrained(".")` signature, but the real
  one is `from_pretrained(".", None)` (`parakeet.rs:40`) — the headline example is wrong.
- **Impact:** docs.rs will show gaps; the crate-root quick-start example does not compile against the current API.
- **Regression risk of fix:** none (docs only).
- **Recommendation:** add `#![warn(missing_docs)]`, fix the crate-root example (`lib.rs:22-25`) to pass the `None`
  arg, and add `///` to public fields of `TimedToken`, `TranscriptionResult`, `UnifiedStreamingConfig`,
  `SpeakerSegment`, `SpeakerTranscript`.
- **Reverification:** `cargo doc` after enabling the lint to enumerate the exact gaps; build with `--all-features`.

---

### A3-12 — `pub use transcriber::*` glob export leaks surface and obscures the public API
**SEVERITY: NIT**
- **Root cause:** `lib.rs:81` re-exports the whole module with a glob.
- **Evidence:** `pub use transcriber::*;` (`lib.rs:81`) — currently only `Transcriber`, but a glob means any future
  `pub` item in `transcriber.rs` silently becomes public API (a semver hazard).
- **Impact:** accidental API surface growth; harder to audit what's public.
- **Recommendation:** replace with explicit `pub use transcriber::Transcriber;`.
- **Reverification:** none.

---

### A3-13 — `from_pretrained` model-file auto-detection is silent and variant-specific
**SEVERITY: NIT**
- **Root cause:** `Parakeet::find_model_file` picks the first of `model.onnx > model_fp16 > model_int8 > model_q4`,
  then any `*.onnx` (`parakeet.rs:90-119`), with no way to ask which it chose or to force one.
- **Evidence:** `parakeet.rs:90-119`; Cohere has its own different auto-detection of `encoder_model[_quantized|_fp16]`
  (`cohere.rs:160-164`).
- **Impact:** a user with both fp16 and q4 in a dir gets fp16 silently; no quantization selection knob (cross-ref A4
  quantization).
- **Recommendation:** add an optional explicit model-file/precision selector to `ExecutionConfig` or the constructor;
  log/expose the chosen file.
- **Reverification:** cross-ref A4 (quantization story).

---

## Sketch of a unified API shape (target for 1.0)

The goal: one mental model. Two traits, uniform constructors, structured errors, typed language.

```text
// Offline (already exists, keep)
trait Transcriber {
    fn transcribe_samples(&mut self, audio, sr, ch, mode: Option<TimestampMode>) -> Result<TranscriptionResult>;
    fn transcribe_file / transcribe_file_batch  // default impls
}

// Streaming (NEW — fixes A3-1..A3-4)
trait StreamingTranscriber {
    type Output;                                  // String | TranscriptionResult | Vec<SpeakerTranscript>
    fn transcribe_chunk(&mut self, &[f32]) -> Result<Self::Output>;
    fn flush(&mut self) -> Result<Self::Output> { Ok(Default::default()) }
    fn reset(&mut self);                          // ALWAYS full reset; documented
    fn timed_transcript(&self, mode: TimestampMode) -> TranscriptionResult;  // where supported
}

// Uniform construction (fixes A3-7)
//   Variant::from_pretrained(path, config: Option<ExecutionConfig>) -> Result<Self>
//   Variant::from_shared(&Handle) -> Self            // every streaming variant gets a Handle
//   Handle::load(path, config: Option<ExecutionConfig>) -> Result<Self>
// MultitalkerASR keeps its (asr_dir, sortformer_path, config) — documented exception.

// Typed config (fixes A3-6, A3-10)
enum Language { Auto, En, EnUs, Es, /* ...documented 40... */ Other(String) }
impl ExecutionConfig { fn gpu() -> Self; fn compiled_providers() -> Vec<ExecutionProvider>; }

// Structured, non-exhaustive errors (fixes A3-5) — thiserror, NOT eyre
#[non_exhaustive] enum Error {
    Io(#[from] io::Error), Ort(ort::Error), Tokenizer(String),
    NotFound { path: PathBuf },
    UnsupportedLanguage { lang: String, supported: Vec<&'static str> },
    InvalidConfig { field: &'static str, reason: String },
    LockPoisoned,
}
```

**eyre vs thiserror posture:** the crate is correct to expose a concrete `Error` enum rather than `eyre::Report`
(`error.rs`). `eyre` is for applications/binaries; a library should give callers a matchable type. Keep the enum;
just make it structured and adopt the `thiserror` derive to delete the hand-written `Display`/`From`
(`error.rs:15-55`). Do not introduce `eyre` into the public API.

---

## Semver guidance (0.x now vs 1.0)

**Do these breaking changes NOW (cheap pre-1.0, high value):**
- A3-5 restructure `Error` + `#[non_exhaustive]` (every later variant add is otherwise breaking — do it once).
- A3-2/A3-3 standardise `transcribe_chunk` + public uniform `reset` (rename via `#[deprecated]` alias this cycle,
  drop aliases at 1.0).
- A3-8 collapse `WordTimestamp` into `TimedToken`; rename internal `ModelConfig` types at definition site.
- A3-7 unify constructor param name to `config`; expose `ExecutionConfig` (not `ModelConfig`) on `Sortformer`.
- A3-12 replace the `transcriber::*` glob.

**Defer / do additively (no break needed):**
- A3-1 `StreamingTranscriber` trait — additive, land alongside the renames.
- A3-4 add `get_timed_transcript` to Nemotron/EOU — additive.
- A3-6 `Language` type — land as additive (`&str` convenience delegates) before committing the enum at 1.0.
- A3-10 `gpu()`/`compiled_providers()` helpers — additive.
- A3-11 docs + `#![warn(missing_docs)]` — non-breaking, do continuously.

**1.0 acceptance bar:** both traits stable; one constructor convention; structured `#[non_exhaustive]` errors; typed
`Language`; `missing_docs` clean on docs.rs; README quick-start compiles against the real signatures.

---

## Cross-references for other lanes

- **A1 (correctness/streaming):** A3-3 (Nemotron `reset()` silently preserves language, `nemotron.rs:495-509`) is the
  *API expression* of the language-lock seed finding. The API fix (make "language persists" explicit, not a silent
  reset default; make `Language::Auto` a first-class state) depends on A1/RU resolving whether re-detect mid-stream is
  feasible. Coordinate: A1 owns the correctness behaviour, A3 owns the surface that exposes it.
- **A2 (performance):** A3-10 (GPU opt-in friction, no `compiled_providers()`/`gpu()` helper) and the lack of any
  async/threading guidance on the `&mut self` blocking `transcribe_chunk` (holds the model `Mutex`,
  `nemotron.rs:584`) are joint API+perf items. The handle/`Arc<Mutex>` shared-model pattern (A3-7) is the crate's
  concurrency story — A2 should assess whether the Mutex granularity is the right perf tradeoff.
- **A4 (models/features):** A3-4 (no timestamps in streaming Nemotron/EOU), A3-6 (typed `Language` incl. `Auto`),
  A3-13 (no quantization/precision selector in `from_pretrained`) are feature gaps A4 owns; the API shape above is the
  surface they'd land on.
- **A5 (quality/structure):** A3-5 recommends adopting `thiserror` to delete hand-written `Display`/`From`
  (`error.rs:15-55`); A3-8 the triple `ModelConfig` naming and `WordTimestamp`/`TimedToken` duplication are also
  code-structure dedup items. The bespoke-per-variant wrappers (A3-1) overlap with A5's wrapper-duplication analysis —
  a shared `StreamingTranscriber` trait is both an API and a dedup win.
- **02 (architecture):** the unified-API sketch (two traits + uniform constructor + Handle on every streaming variant)
  is the target architecture the wrapper layer should converge to; feeds the architecture map.
- **A6 (tests):** any rename/trait extraction needs the missing streaming reset/flush tests as regression guards
  before the breaking changes land.
