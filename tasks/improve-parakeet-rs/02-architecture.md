# 02 - Architecture map (current state + target sketch)

**Lane:** Architecture (02), Wave R. PLAN ONLY, no code.
**/think frameworks:** systems-thinking (component boundaries, state flow, coupling) + map-territory (the
README/docstrings claim a "shared infra + per-variant" structure; I verify what is actually shared in
code vs copy-pasted, and label every claim with `file:line`).

All line citations verified against the working tree at the time of writing. Re-verify before relying.

---

## 1. Current-state layering

The crate has a **thin shared base** and a **wide, mostly-bespoke per-variant layer**. The base provides
audio loading, FFT/STFT, one mel filterbank flavor, the CTC decoder, vocab parsing, timestamp
post-processing, execution-provider config, and the error type. Everything model-specific (encoder/decoder
ONNX run, streaming cache, mel constants, per-variant mel filterbank, decode loop) is reimplemented per
variant.

### Shared infrastructure (genuinely reused across variants)

| Module | Lines | What it provides | Who consumes it |
|--------|-------|------------------|-----------------|
| `execution.rs` | 206 | `ExecutionProvider` enum + `ModelConfig` (`apply_to_session_builder`, EP wiring, threads, CoreML cache) | EVERY `model_X.rs` and `sortformer.rs` |
| `audio.rs` | 332 | `load_audio`, `apply_preemphasis`, `stft`/`stft_with_plan`, Slaney `create_mel_filterbank`, `extract_features_with_cache`, `FeatureCache` | CTC, TDT, Unified fully; Nemotron/Multitalker partially (filterbank + preemphasis + stft only) |
| `error.rs` | 55 | `Error` enum + `Result`, `From` impls | all |
| `config.rs` | 51 | `PreprocessorConfig`, `ModelConfig` (JSON) + defaults | CTC, TDT, Unified |
| `decoder.rs` | 211 | `ParakeetDecoder` (CTC collapse + timestamps), `TimedToken`, `TranscriptionResult` | CTC only; `TranscriptionResult`/`TimedToken` reused everywhere |
| `timestamps.rs` | 359 | `TimestampMode`, `process_timestamps` | CTC, TDT, Unified |
| `vocab.rs` | 66 | `Vocabulary` (vocab.txt parser) | TDT only |
| `transcriber.rs` | 72 | `Transcriber` trait (`transcribe_samples` + default `transcribe_file`/`_batch`) | CTC, TDT, Unified ONLY (3 of 10) |

### Per-variant pairs (the wide layer)

| Variant | wrapper | model_X | extra | shared base actually used |
|---------|---------|---------|-------|----------------------------|
| CTC | `parakeet.rs` 168 | `model.rs` 93 | `decoder.rs` | `audio` (full), `config`, `decoder`, `timestamps`, `transcriber` |
| TDT | `parakeet_tdt.rs` 154 | `model_tdt.rs` 292 | `decoder_tdt.rs` 204, `vocab.rs` | `audio` (full), `config`, `vocab`, `timestamps`, `transcriber` |
| EOU | `parakeet_eou.rs` 308 | `model_eou.rs` 196 | - | only `execution`, `error`; has its OWN mel filterbank (HTK) + preemphasis call |
| Nemotron | `nemotron.rs` 786 | `model_nemotron.rs` 288 | - | `execution`, `error`, `audio::{create_mel_filterbank, apply_preemphasis, stft}`; OWN tokenizer (SentencePiece) |
| Unified | `parakeet_unified.rs` 590 | `model_unified.rs` 187 | - | `audio` (full via FeatureCache), `timestamps`, `transcriber`, reuses Nemotron's `SentencePieceVocab` |
| Multitalker | `multitalker.rs` 771 | `model_multitalker.rs` | (feature) | `execution`, `error`, `audio::{create_mel_filterbank, apply_preemphasis}`; OWN mel constants |
| Cohere | `cohere.rs` | `model_cohere.rs` | (feature) | `execution`, `error` |
| Sortformer | `sortformer.rs` 1254 | (inline) | (feature) | `execution`, `error`, `audio::create_mel_filterbank`; OWN `apply_preemphasis` (`sortformer.rs:1113`), OWN mel constants |

### Dependency graph (ASCII)

```
                         lib.rs  (re-exports public API: lib.rs:76-99)
                            |
   +------------------------+-------------------------+--------------------+
   |              |             |          |           |          |        |
 Parakeet     ParakeetTDT   ParakeetEOU  Nemotron  Unified  Multitalker  Cohere   Sortformer
 (parakeet)   (parakeet_tdt)(parakeet_eou)(nemotron)(parakeet_unified) (feat)  (feat)   (feat, inline)
   |              |              |          |          |          |        |        |
 model.rs    model_tdt.rs   model_eou.rs model_     model_     model_   model_   (no model_X;
   |        + decoder_tdt   (own mel)    nemotron   unified   multitalker cohere   sessions inline)
   |        + vocab.rs        |            |          |          |        |        |
   +---- decoder.rs           |     SentencePieceVocab          |        |        |
   |     (CTC)                |     (defined in nemotron.rs,     |        |        |
   |                          |      reused by unified) <--------+        |        |
   |                          |                                           |        |
   v          v               v           v          v          v        v        v
 +--------------------------------------------------------------------------------------+
 | SHARED INFRA                                                                          |
 |  execution.rs (EP/session)  <-- used by ALL model layers                             |
 |  error.rs (Error/Result)    <-- used by ALL                                          |
 |  audio.rs (stft, preemphasis, Slaney filterbank, FeatureCache, extract_features)     |
 |     full use: CTC, TDT, Unified ; partial use: Nemotron, Multitalker, Sortformer     |
 |     NOT used: EOU (own HTK filterbank), and EOU/Sortformer reimplement preemphasis   |
 |  config.rs (PreprocessorConfig)  <-- CTC, TDT, Unified                               |
 |  timestamps.rs (process_timestamps, TimestampMode) <-- CTC, TDT, Unified             |
 |  decoder.rs (CTC), vocab.rs (TDT) <-- single-consumer each                           |
 |  transcriber.rs (Transcriber trait) <-- CTC, TDT, Unified ONLY (3 of 10)             |
 +--------------------------------------------------------------------------------------+
```

**Key structural observation:** the `Transcriber` trait (the only shared abstraction across variants)
covers exactly the 3 *offline, non-streaming* variants (CTC, TDT, Unified). The 5 streaming/diarization
variants (EOU, Nemotron, Multitalker, Cohere, Sortformer) share **no trait at all** and each invents its
own method names (`transcribe_chunk`, `transcribe`, `diarize_chunk`, `feed`, `transcribe_audio`).

---

## 2. The `model_X.rs` + wrapper PATTERN: shared vs duplicated

The pattern is real but enforced by **convention, not by code**. There is no trait `model_X` implements;
each is an independent struct with hand-rolled, near-identical methods. Below is what is genuinely shared
vs copy-pasted, quantified by citing the parallel functions.

### 2a. ONNX session setup — DUPLICATED 6x (only the EP-apply step is shared)

The `apply_to_session_builder` call is shared (`execution.rs:116`), but the surrounding
"build → apply → commit_from_file, twice for encoder+decoder" block is copy-pasted in every two-session
model:

- `model.rs:27-29` (CTC, single session)
- `model_tdt.rs:47-54`
- `model_eou.rs:53-60`
- `model_nemotron.rs:88-94`
- `model_unified.rs:44-50`
- `model_multitalker.rs:85-90`
- `model_cohere.rs:95-100`
- `sortformer.rs:225` (inline)

That is **6 verbatim copies** of the `let builder = Session::builder()?; let mut builder =
exec_config.apply_to_session_builder(builder)?; let X = builder.commit_from_file(&path)?;` idiom for the
encoder, repeated again for the decoder. A `load_session(path, &exec)` helper would collapse all of them.

### 2b. `find_encoder` / `find_decoder_joint` — DUPLICATED 3x, divergent

Candidate-filename probing is reimplemented with **different candidate lists** in:
- `model_tdt.rs:63-109` (`encoder-model.onnx`, `encoder.onnx`, `encoder-model.int8.onnx` + dir scan)
- `model_unified.rs:59-91` (`encoder.onnx`, `encoder.int8.onnx`, `encoder-model.onnx`)
- (Nemotron/EOU hardcode `encoder.onnx`/`decoder_joint.onnx` with an `exists()` check:
  `model_nemotron.rs:72-86`, `model_eou.rs:42-50`)

This is a correctness hazard: the candidate lists diverge, so the same quantized file naming convention is
accepted by one variant and rejected by another. A5/A3 should note the inconsistency.

### 2c. Encoder run — STRUCTURALLY IDENTICAL, 5x copies with input-name drift

`run_encoder` is the same shape every time: build input Value(s) → `session.run(ort::inputs!...)` →
`try_extract_tensor::<f32>()` on the output → `Array3::from_shape_vec((b,d,t), data.to_vec())` →
extract `encoded_len`. The only real differences are (a) ONNX input/output names and (b) whether a cache or
`prompt_index` is threaded:

| File:line | input names | cache? | output names |
|-----------|-------------|--------|--------------|
| `model.rs:49-52` | `input_features`, `attention_mask` | no | `logits` |
| `model_tdt.rs:142-145` | `audio_signal`, `length` | no | `outputs`, `encoded_lengths` |
| `model_unified.rs:105-108` | `audio_signal`, `length` | no | `outputs`, `encoded_lengths` |
| `model_eou.rs:79-85` | `audio_signal`, `length`, `cache_*` (3) | yes | `outputs`, `new_cache_*` (3) |
| `model_nemotron.rs:149-155` | `processed_signal`, `processed_signal_length`, `cache_*` (3), optional `prompt_index` | yes | `encoded`, `encoded_len`, `cache_*_next` (3) |

The cache-extraction-into-`Array4` block is byte-for-byte identical between `model_eou.rs:101-142` and
`model_nemotron.rs:185-225` (same three `try_extract_tensor` + `from_shape_vec` rebuilds, only the output
key strings differ: `new_cache_last_channel` vs `cache_last_channel_next`).

### 2d. Decoder/joint step — STRUCTURALLY IDENTICAL, 4x copies

`run_decoder` (RNN-T/TDT joint step: encoder_frame + targets + target_length + 2 LSTM states → logits + 2
new states) is the same in:
- `model_eou.rs:149-195`
- `model_nemotron.rs:232-287`
- `model_unified.rs:134-186`
- `model_tdt.rs:214-220` (inlined in `greedy_decode`, not a separate fn)

All four use the **same five ONNX input names** (`encoder_outputs`, `targets`, `target_length`,
`input_states_1`, `input_states_2`) and the **same two output state names** (`output_states_1`,
`output_states_2`). Differences are only in logits dtype/shape handling (`Array1` vs `Array3`) and the
clone-per-step cost. This is the single clearest extraction target.

### 2e. Greedy decode loop — DUPLICATED 4x

The `for frame in encoder_out { for _ in 0..MAX_SYMBOLS { run_decoder; argmax; if blank break; push token;
carry state+last_token } }` loop appears in:
- `nemotron.rs:723-770` (`decode_chunk`, `max_symbols_per_step = 10`)
- `parakeet_eou.rs:194-243` (`syms_added < 5`, plus EOU-token special case)
- `parakeet_unified.rs:442-496` (`decode_encoder_frames`, `MAX_SYMBOLS_PER_STEP = 10`)
- `model_tdt.rs:176-291` (`greedy_decode`, plus TDT duration skip)

The argmax-over-logits is hand-rolled **4 separate times** with subtly different tie-breaking and NaN
handling (`nemotron.rs:749-756` ignores NaN ordering via `>`; `parakeet_eou.rs:208-214` adds
`val.is_finite()`; `parakeet_unified.rs:477-482` uses `max_by`/`partial_cmp`). A5 should flag the
inconsistent argmax as a latent correctness divergence.

### 2f. Mel feature extraction — DUPLICATED with 3 incompatible flavors

This is the most fragmented area. There are **three different mel pipelines**:

1. **Slaney + per-feature normalization** — `audio.rs::extract_features_with_cache` (`audio.rs:206-266`).
   Used by CTC, TDT, Unified. Includes mean/std normalization with Bessel correction.
2. **Slaney filterbank, NO normalization, raw log-mel** — Nemotron `compute_mel_spectrogram`
   (`nemotron.rs:775-785`), Multitalker `compute_mel_spectrogram` (`multitalker.rs:666-676`), Sortformer
   `extract_mel_features` (`sortformer.rs:1179-1195`). All three call `audio::create_mel_filterbank` then
   `mel_basis.dot(spec)` then `(x + LOG_ZERO_GUARD).ln()`. Near-verbatim 3x.
3. **HTK filterbank** — EOU `create_mel_filterbank_htk` (`parakeet_eou.rs:268-308`), a *separate* mel-scale
   formula (`2595 * log10(1 + hz/700)`) from the Slaney one in `audio.rs:133-147`.

Plus `apply_preemphasis` is reimplemented inline in `sortformer.rs:1113-1118` despite
`audio::apply_preemphasis` existing (`audio.rs:49-62`). The mel constants
(`N_FFT=512, HOP=160, N_MELS=128, PREEMPH=0.97, LOG_ZERO_GUARD=5.96e-8`) are re-declared as private `const`
in `nemotron.rs:15-26`, `parakeet_eou.rs:9-17`, `multitalker.rs:27-32`, `sortformer.rs:31-36`. **Four
copies of the same constants.** (Note: Nemotron `LOG_ZERO_GUARD` at `nemotron.rs:21` = `5.96e-8` differs
from `audio.rs:240` which uses `2^-24` = `5.96e-8` — same value, expressed two ways; but the *normalization*
differs, which is model-correct, not a bug.)

### 2g. Streaming cache structs — DUPLICATED 2x (near-identical)

`EncoderCache` (`model_eou.rs:9-28`, hardcoded dims `17,1,70,512` / `17,1,512,8`) and
`NemotronEncoderCache` (`model_nemotron.rs:13-32`, dims supplied via `with_dims`) hold the **same three
fields** (`cache_last_channel: Array4`, `cache_last_time: Array4`, `cache_last_channel_len: Array1<i64>`).
The Nemotron one is the better design (dims read from the graph at `model_nemotron.rs:108-127`); EOU's is
hardcoded and brittle. Multitalker has a third copy (`MultitalkerEncoderCache`, `model_multitalker.rs:13`).

### 2h. Tokenizer/vocab — THREE incompatible loaders

- `tokenizers::Tokenizer` (HF JSON) — CTC (`decoder.rs:30`), EOU (`parakeet_eou.rs:62`).
- `Vocabulary` (vocab.txt) — TDT only (`vocab.rs`).
- `SentencePieceVocab` (hand-rolled protobuf parser, `nemotron.rs:109-263`) — Nemotron + Unified
  (`parakeet_unified.rs:7`).

The SentencePiece protobuf parser living inside `nemotron.rs` and being imported by `parakeet_unified.rs`
is a layering smell: a shared concern (vocab decoding) is owned by one variant module. A3/A5 note.

### Duplication scorecard (hand to A5)

| Concern | Copies | Lines/copy (approx) | Extractable? |
|---------|--------|---------------------|--------------|
| Session build+commit idiom | 6+ | 6 | trivially (a `load_session` fn) |
| `find_encoder`/`find_decoder_joint` | 3 (divergent) | 25-45 | yes, with a unified candidate list |
| `run_encoder` body | 5 | 40-90 | partially (names differ) |
| cache extraction → Array4 | 2 verbatim | 40 | yes |
| `run_decoder` joint step | 4 | 45 | yes (same I/O names) |
| greedy decode + argmax loop | 4 (divergent argmax) | 30-115 | yes, behavior must be reconciled first |
| raw log-mel (Slaney, no-norm) | 3 verbatim | 10 | yes |
| mel constants block | 4 | 8 | yes (shared `const`/config) |
| encoder cache struct | 3 | 20 | yes |
| preemphasis | 2 (1 inline dup) | 10 | already shared, just call it |

Conservative estimate: a shared streaming/RNN-T backend could remove **roughly 600-900 lines** of
duplicated logic across the 5 RNN-T-style variants (TDT, EOU, Nemotron, Unified, Multitalker) without
touching the public types. This is the single highest-leverage structural item.

---

## 3. The streaming abstraction (current)

There is **no streaming abstraction**. Each streaming variant hand-rolls its own stateful struct + chunk
geometry. Three distinct streaming strategies coexist:

### 3a. Strategy A: raw-audio rolling buffer + recompute full mel (Nemotron, EOU, Multitalker)

- **Nemotron** (`nemotron.rs:615-721`): `transcribe_chunk(&mut self, &[f32])` appends to `audio_buffer:
  Vec<f32>`, recomputes mel over the ENTIRE buffer every chunk (`nemotron.rs:628`), tracks `audio_processed`
  and `chunk_idx`, slices `PRE_ENCODE_CACHE(9) + CHUNK_SIZE(56)` mel frames, threads `NemotronEncoderCache`
  + `prompt_index`, then greedy-decodes carrying `last_token` + `state_1`/`state_2`. Trims buffer at
  `nemotron.rs:705-712`. Chunk geometry as `const` (`CHUNK_SIZE=56`, `PRE_ENCODE_CACHE=9`,
  `nemotron.rs:25-26`).
- **EOU** (`parakeet_eou.rs:137-245`): `transcribe(&mut self, chunk, reset_on_eou)`. 4-second `VecDeque`
  ring buffer (`parakeet_eou.rs:109`), recompute full mel, slice last `PRE_ENCODE_CACHE(9) +
  FRAMES_PER_CHUNK(16)` frames. EOU token triggers a *soft* reset (decoder state only, NOT encoder cache:
  `parakeet_eou.rs:247-255`). Geometry as inline `const` inside the method (`parakeet_eou.rs:159-161`).
- **Multitalker** (`multitalker.rs:325`): same family, per-speaker decode.

Cost note for A2: recomputing mel over the whole rolling buffer per chunk is O(buffer) every call, not
O(new-samples) — quadratic-ish in buffer length until the trim kicks in.

### 3b. Strategy B: explicit left/chunk/right window geometry (Unified)

- **Unified** (`parakeet_unified.rs:303-373`): `transcribe_chunk` + `flush`. Chunk geometry is a *typed
  config* `UnifiedStreamingConfig { left_context_secs, chunk_secs, right_context_secs }`
  (`parakeet_unified.rs:25-112`) with `validate()` enforcing subsampling alignment. `process_ready_chunks`
  drains the buffer in `chunk_samples` steps, `build_window_audio` assembles a left+main+right window with
  absolute-sample bookkeeping (`buffer_start_sample`, `next_chunk_start_sample`). This is the **most mature
  and most testable** streaming design in the crate (it even has the only streaming unit test,
  `parakeet_unified.rs:581-589`). It does NOT use an ONNX encoder cache; it recomputes context windows
  instead.

### 3c. Strategy C: stateful ONNX cache, no buffering exposed (Sortformer)

- **Sortformer** (`sortformer.rs:356-477`): `diarize_chunk` / `feed` / `flush`, ONNX-side streaming state
  via `reset_state` (`sortformer.rs:282`). Different domain (diarization, speaker segments) but same
  `&mut self` blocking-inference shape.

### Common shape of every streaming method

All streaming variants share: `&mut self`, blocking, lock the model `Arc<Mutex<Model>>` only during
inference (`nemotron.rs:583-593`, `parakeet_eou.rs:169-175`, `parakeet_unified.rs:343-348`), carry
`last_token` + 2 LSTM states across chunks, and return incremental text. The `Handle`/`from_shared` pattern
(load once, spawn N independent-state instances sharing one session) is **consistently good** and present in
Nemotron, EOU, Unified (`nemotron.rs:326-377/418-453`, `parakeet_eou.rs:53-123`,
`parakeet_unified.rs:145-251`). Multitalker/Cohere/Sortformer/CTC/TDT do NOT have it.

**Where chunk geometry lives:** scattered. Unified = typed validated config (good). Nemotron/EOU = private
`const` (opaque to callers). This inconsistency is itself an API finding for A3.

---

## 4. Where a shared trait COULD be extracted + blast radius

Two candidate abstractions, very different blast radius.

### 4a. `StreamingTranscriber` trait (LOW blast radius, additive) — RECOMMENDED

A trait capturing the common streaming shape:

```
trait StreamingTranscriber {
    fn transcribe_chunk(&mut self, audio: &[f32]) -> Result<String>;
    fn flush(&mut self) -> Result<String>;   // currently only Unified has flush
    fn reset(&mut self);
    fn get_transcript(&self) -> String;       // Nemotron + Unified have this
}
```

- **Implementable today by:** Nemotron, EOU (method is named `transcribe`, not `transcribe_chunk` — needs a
  rename or a wrapper), Unified.
- **Blast radius:** ADDITIVE. The structs and their inherent methods stay; the trait just gives a uniform
  interface. `lib.rs:76-99` re-exports gain one `pub use`. Nothing in the public API *breaks* if the inherent
  methods are kept. The one rename (EOU `transcribe` → `transcribe_chunk`) is a breaking change but trivial
  and pre-1.0-acceptable. This is the cheapest win and directly serves the "variant consistency" objective.
- **Risk:** `reset_on_eou`/EOU-token semantics and Unified's `flush` don't generalize cleanly; keep those as
  inherent methods, put only the common 4 in the trait.

### 4b. Internal `RnntBackend` (encoder/decoder run + greedy loop) (MEDIUM blast radius, INTERNAL) — RECOMMENDED

A *non-public* shared backend that owns the `run_encoder`/`run_decoder`/greedy-decode logic (sections
2c-2e), parameterized by ONNX input/output names and cache layout. This is where the ~600-900 line
reduction lives.

- **Blast radius:** ZERO public API change if kept `pub(crate)`. The wrappers keep their public surface; only
  their private internals delegate to the shared backend. The `model_X.rs` files shrink dramatically or
  merge. `lib.rs:87-90` currently `pub use`s `ParakeetModel`, `ParakeetEOUModel`, `NemotronModel`,
  `NemotronEncoderCache`, `ParakeetUnifiedModel`, `UnifiedModelConfig` — **these low-level types are public**,
  so collapsing them IS a breaking change. To stay non-breaking, either (i) keep the type names as thin
  re-exports, or (ii) accept the break under 0.x semver and document it. A3 owns this call.
- **Prereq:** the divergent argmax (2e) and divergent `find_encoder` candidate lists (2b) must be reconciled
  first, or the refactor will silently change behavior. Hand this ordering constraint to A5/A1.

### What would break — concrete inventory against `lib.rs:76-99`

Public low-level types currently exported (so any consolidation touches them):
- `model::ParakeetModel` (`lib.rs:87`)
- `model_eou::ParakeetEOUModel` (`lib.rs:88`)
- `model_nemotron::{NemotronEncoderCache, NemotronModel, NemotronModelConfig}` (`lib.rs:89`)
- `model_unified::{ParakeetUnifiedModel, UnifiedModelConfig}` (`lib.rs:90`)
- `decoder::{ParakeetDecoder, TimedToken, TranscriptionResult}` (`lib.rs:86`)

Wrappers exported (the intended public entry points, untouched by 4a/4b if inherent methods preserved):
`Parakeet`, `ParakeetTDT`, `Nemotron`/`NemotronHandle`/`NemotronMode`, `ParakeetEOU`/`Handle`,
`ParakeetUnified`/`Handle`/`UnifiedStreamingConfig`, `MultitalkerASR`, `CohereASR`, sortformer
(`lib.rs:78-99`).

**Conclusion:** 4a is safe and serves consistency. 4b is high-value but its blast radius depends entirely on
whether the `pub use` of `model_X` low-level types is considered API surface worth preserving. Recommend
A3 decide whether to **stop exporting the `model_X` types** (they look like accidental exports — most users
only need the wrappers) so 4b becomes internal and non-breaking.

---

## 5. Target-state sketch (one diagram)

```
                          PUBLIC API (wrappers only; model_X types NO LONGER exported)
   Parakeet  ParakeetTDT  ParakeetEOU  Nemotron  ParakeetUnified  Multitalker  Cohere  Sortformer
       \________\___________\___________|___________/____________/_______/________/
                                        |
                  +---------------------+---------------------+
                  | trait StreamingTranscriber (4a, public)   |  <- uniform streaming surface
                  | trait Transcriber (existing, offline)     |
                  +---------------------+---------------------+
                                        |
                  +---------------------+---------------------+
                  | pub(crate) RnntBackend (4b, internal)     |  <- one run_encoder/run_decoder/greedy
                  |   parameterized by io-names + cache spec  |     argmax reconciled once
                  +---------------------+---------------------+
                                        |
   +-------------------------------------------------------------------------------------+
   | SHARED INFRA (consolidated)                                                          |
   |  execution.rs (+ load_session helper)                                                |
   |  audio.rs  (Slaney filterbank, HTK filterbank moved here, raw-log-mel helper,        |
   |             ONE preemphasis, shared mel constants/config struct)                     |
   |  vocab/ (Vocabulary | SentencePieceVocab moved out of nemotron.rs | HF tokenizer)    |
   |  decoder.rs (CTC)  timestamps.rs  config.rs  error.rs                                |
   +-------------------------------------------------------------------------------------+
```

Target moves, in priority order (detailed task breakdown is for lane 90):
1. `load_session(path, &exec)` helper -> removes 2a duplication (zero API risk).
2. Move `SentencePieceVocab` + EOU's HTK filterbank into a `vocab`/`audio` home (fixes layering, low risk).
3. Shared raw-log-mel helper + single mel-constants source (fixes 2f/2g, low risk).
4. Reconcile the 4 argmax / 2 cache-extraction / 3 find_encoder copies (correctness prereq for 5).
5. `pub(crate) RnntBackend` + `StreamingTranscriber` trait (the big one; needs A3 decision on stopping
   `model_X` exports).

---

## Cross-references for other lanes

- **A5 (quality):** full duplication scorecard in section 2 (scorecard table at end of 2). Highest-leverage
  items: 6x session-build idiom (2a), 4x run_decoder (2d), 4x greedy/argmax loop with DIVERGENT argmax/NaN
  handling (2e) — flag the argmax divergence as a latent correctness bug, not just style. 3x divergent
  `find_encoder` candidate lists (2b) are a real inconsistency. 2 verbatim cache-extraction blocks (2c).
  `apply_preemphasis` reimplemented inline in `sortformer.rs:1113` despite `audio.rs:49` existing. Four
  copies of the mel-constants block. Estimated ~600-900 removable lines across the 5 RNN-T variants.
- **A3 (api):** the trait-shape question. (a) Only 3 of 10 variants implement any shared trait
  (`Transcriber`, offline only); 5 streaming variants share NO trait and use 5 different method names
  (`transcribe_chunk`/`transcribe`/`diarize_chunk`/`feed`/`transcribe_audio`). Propose `StreamingTranscriber`
  (4a, additive, one EOU rename). (b) DECISION NEEDED: are the `model_X` low-level types
  (`ParakeetModel`, `ParakeetEOUModel`, `NemotronModel`/`NemotronEncoderCache`/`NemotronModelConfig`,
  `ParakeetUnifiedModel`/`UnifiedModelConfig`, `lib.rs:87-90`) intentional public API or accidental exports?
  If accidental, stop exporting them so the 4b internal-backend refactor becomes non-breaking. (c) Chunk
  geometry inconsistency: Unified uses a typed validated `UnifiedStreamingConfig`; Nemotron/EOU bury it in
  private `const`. (d) `Handle`/`from_shared` exists only for Nemotron/EOU/Unified, not the other 5.
- **A1 (correctness) / A4 (models):** the language-lock seed lives in this architecture: `prompt_index` is a
  per-instance field set once (`nemotron.rs:447,489`), threaded into `run_encoder` every chunk
  (`nemotron.rs:591,691`) and PRESERVED by `reset()` (`nemotron.rs:495-509`). The autoregressive carry
  (`last_token` + `state_1`/`state_2` at `nemotron.rs:313-315`) compounds the lock. Architecturally, a
  re-detect-and-re-prompt hook would slot into `transcribe_chunk` at a silence boundary; whether that is
  *correct* vs model-inherent is A1/A4/RU's call. Also: EOU's soft-reset deliberately preserves encoder
  cache + buffer (`parakeet_eou.rs:247-255`) — A1 verify that is intended.
- **A2 (performance):** Strategy-A streaming (Nemotron/EOU/Multitalker) recomputes mel over the ENTIRE
  rolling buffer every chunk (`nemotron.rs:628`, `parakeet_eou.rs:153-154`) = O(buffer) per call, not
  O(new-samples). Every `run_encoder`/`run_decoder` does `.clone()` of inputs and `data.to_vec()` of outputs
  (e.g. `model_nemotron.rs:152-154,176`); per-decoder-step state clones in the greedy loop
  (`model_unified.rs:148-149`). One `Arc<Mutex<Model>>` serializes all concurrent streams through one
  session. Unified's window approach (Strategy B) avoids the full-buffer recompute and is the better
  performance baseline to converge on.
- **A6 (tests):** the only streaming-geometry test is `parakeet_unified.rs:581-589`. The shared backend (4b)
  is exactly what makes the argmax/greedy/cache logic testable in one place instead of 4. Reconciling the
  divergent argmax (2e) needs a regression test per variant FIRST. `audio.rs` STFT tests exist
  (`audio.rs:279-331`) but the three divergent mel pipelines (2f) and the two encoder-cache structs (2g) are
  unguarded.
- **90 (synthesis):** target-state move ordering is in section 5 (1->5). Items 1-3 are low-risk and can ship
  independently; item 4 is a correctness prerequisite for item 5; item 5 depends on the A3 export decision.
