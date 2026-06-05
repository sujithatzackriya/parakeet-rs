# A4 — Model Coverage & Features Audit

**/think frameworks:** jobs-to-be-done (what user-facing job does each variant promise vs deliver?) + gap-analysis (capability matrix → expected vs actual per cell) + first-principles for the language-lock (separate model-inherent constraint from library policy by reading the ONNX graph contract, not just the Rust).

Scope: feature coverage across the 10 shipped variants. PLAN ONLY. Every cell/finding cites `file:line` or a labelled assumption. Upstream "what does NeMo do" questions are deferred to lane RU and flagged inline.

---

## CAPABILITY MATRIX

Legend: ✅ supported · ⚠️ partial / awkward · ❌ absent · n/a not applicable to the architecture. Cite columns:
- **Word ts** = word-level timestamps surfaced on the public API.
- **Lang select** = caller can choose / model auto-detects a language.
- **Diar** = speaker attribution.
- **Stream** = stateful `transcribe_chunk`/`feed` incremental API.
- **Quant** = loader auto-selects int8/int4/fp16 file.
- **Punct/caps** = output carries punctuation and casing.

| Variant | Word ts | Lang select | Diar | Stream | Quant | Punct/caps | Evidence |
|---|---|---|---|---|---|---|---|
| **CTC** (`parakeet.rs`) | ✅ Words/Sentences via `TimestampMode` | ❌ EN-only | ❌ | ❌ offline only | ⚠️ `model_fp16/int8/q4` candidate list | ⚠️ CTC is lowercase, no punct (per `timestamps.rs:11-13`) | `parakeet.rs:91-96,136-155`; `timestamps.rs:11-13` |
| **TDT** (`parakeet_tdt.rs`) | ✅ Tokens/Words/Sentences | ⚠️ "multilingual / auto" but no `set_lang` API (`README.md:27`) | ❌ | ❌ offline only | ⚠️ `encoder-model.int8` candidate list | ✅ predicts punctuation (`timestamps.rs:14-15`) | `parakeet_tdt.rs:107-119`; `model_tdt.rs:64-99` |
| **EOU** (`parakeet_eou.rs`) | ❌ returns `String` only | ❌ EN-only (tokenizer) | ❌ | ✅ `transcribe(chunk, reset_on_eou)` + EOU detection | ❌ fixed `encoder.onnx`/`decoder_joint.onnx` | assumed yes (NeMo realtime model) | `parakeet_eou.rs:137`; `model_eou.rs:42-47` |
| **Nemotron EN** (`nemotron.rs`) | ❌ `transcribe_chunk` returns `String` | ❌ EN-only (`prompt_index=None`) | ❌ | ✅ cache-aware `transcribe_chunk` | ❌ fixed `encoder.onnx`; int8/int4 EXIST on HF but loader can't pick them | ✅ | `nemotron.rs:615,720`; `model_nemotron.rs:72-73`; `README.md:155` |
| **Nemotron multi 3.5** (`nemotron.rs`) | ❌ `String` only | ✅ `set_target_lang` + `"auto"` (idx 101) | ❌ | ✅ but **language-locks** (seed finding) | ❌ same fixed names | ✅ + emits/strips `<xx-XX>` tags | `nemotron.rs:475-491,615,720`; export script L380-394 |
| **Unified** (`parakeet_unified.rs`) | ✅ `get_timed_transcript(mode)` (Tokens/Words/Sentences) | ❌ EN-only | ❌ | ✅ buffered `transcribe_chunk` + `flush` | ⚠️ `encoder.int8` candidate list | ✅ | `parakeet_unified.rs:272-276,303-310`; `model_unified.rs:60-80` |
| **Multitalker** (`multitalker.rs`) | ✅ per-speaker `WordTimestamp` | ❌ EN vocab (`VOCAB_SIZE=1024`) | ✅ via Sortformer | ✅ `transcribe_chunk` + `LatencyMode` | ✅ int8-first (`model_multitalker.rs:54-79`) | assumed yes | `multitalker.rs:56-62,300-319,644-663` |
| **Cohere** (`cohere.rs`) | ❌ hardcodes `<|notimestamp|>` | ✅ explicit 14-lang select | ⚠️ hardcodes `<|nodiarize|>` (token EXISTS) | ❌ offline only | ✅ quantized-first incl. fp16 | ✅ `pnc`/`itn` toggles | `cohere.rs:253-259,299-310`; `model_cohere.rs:68-91` |
| **Sortformer** (`sortformer.rs`) | n/a (segment ts) | n/a | ✅ 4-spk + 3 presets | ✅ `diarize_chunk`/`feed`/`flush` | ❌ fixed path arg | n/a | `sortformer.rs:100-147,294-477` |

**Matrix takeaways (each a finding below):**
1. Word timestamps are absent on every streaming variant except Unified and Multitalker. The two newest, most-used streaming engines (Nemotron EN + multi) return bare `String`.
2. Language selection exists in three different shapes: `set_target_lang` (Nemotron), a `language` arg per call (Cohere), and "auto, no API" (TDT). No shared abstraction.
3. Quantized-file selection is reimplemented in 6 loaders with 4 different naming conventions and no fp16 support in half of them.
4. No single entry point picks a backend; the caller hard-codes a concrete type.

---

## RANKED FINDINGS

### F1 — Multilingual Nemotron language-lock: a FIXABLE library policy, not a graph limit `[SEVERITY: HIGH]`
Co-owned with A1 (correctness) and RU (upstream). This lane owns the **feature/graph-capability** half.

- **Root cause (library):** `prompt_index` is set once via `set_target_lang` (`nemotron.rs:475-491`), defaults to `Some(101)` = `"auto"` (`nemotron.rs:48,427`), is fed unchanged on EVERY chunk (`nemotron.rs:591,691`), and `reset()` deliberately preserves it (`nemotron.rs:495-509`). Combined with the autoregressive carried `last_token` (`nemotron.rs:763`) and persistent `NemotronEncoderCache` (`nemotron.rs:312,594,694`), both the prompt conditioning AND the decoder state bias toward the first-committed language and never re-open the decision.
- **Graph capability (decisive evidence):** `prompt_index` is a **real, dynamic ONNX input**, not a baked constant. The export wrapper inlines `_apply_prompt_to_encoded`, builds a per-frame one-hot from `prompt_index`, and concatenates it before `prompt_kernel` (`scripts/export_nemotron_streaming_multilingual.py:317-357,380-412`). The script's whole stated purpose is "one ONNX serves every language. The Rust side just looks up the index and feeds it" (L36-38). Therefore **the graph imposes no fixed-language constraint** — the lock is 100% a library choice to never vary `prompt_index` and to preserve it across `reset()`.
- **What the graph does NOT give us (constrains the fix):** the multilingual encoder's `output_names` are exactly `["encoded","encoded_len","cache_last_channel_next","cache_last_time_next","cache_last_channel_len_next"]` (export script L388-394). **There is NO language posterior / language-id logit output.** The only language signal the model emits is inline `<xx-XX>` SentencePiece tokens in the decoder text stream — which the library currently strips and discards (`nemotron.rs:78-93,256-262,511-521,716`). `lang_tag_ids` already exist and are computed (`nemotron.rs:256-262,442`), so the detected language IS observable, just thrown away.
- **Impact:** mid-stream code-switching (the live-mic symptom) is impossible with current API; once Hindi commits, it stays Hindi.
- **Feasible fixes, ranked by the graph capability:**
  1. **Expose detected language (cheap, no graph change):** stop discarding the `<xx-XX>` token — surface the last-seen lang tag via a `detected_language()` getter and/or include it in a richer return type. Pure library change, zero risk to EN path. This is the prerequisite for any auto-continuous mode (you can't re-decide without observing).
  2. **`set_target_lang` mid-stream + explicit re-prompt API (medium):** the setter already mutates `prompt_index` (`nemotron.rs:489`); the only thing stopping clean mid-stream switching is the carried decoder state + encoder cache. Add a `set_target_lang_and_reset_decoder()` that flips the index AND resets `last_token`/decoder LSTM state while OPTIONALLY preserving the encoder cache (needs A1 + RU to confirm whether resetting the joint state without the encoder cache produces valid continuation). Caller-driven; no auto-detect needed.
  3. **Auto-continuous re-detect (highest value, most risk):** at an utterance/silence boundary (no native silence signal in this graph — EOU has it, Nemotron does not), re-prompt with `"auto"` and let the model re-pick. Requires (a) a boundary signal the crate does not currently produce for Nemotron, and (b) clearing the autoregressive bias. **DEFER the "does upstream NeMo re-detect per buffer" question to RU** — if upstream re-runs auto each chunk and relies only on the encoder cache (not a sticky prompt), the lock is purely our `reset()`-preserves-language policy and fix #2 suffices.
- **Verdict on fixability:** **The lock is a fixable library choice, NOT a model-inherent limit.** The graph accepts a fresh `prompt_index` every chunk. The hard constraint is only that the model gives no language posterior, so any auto-continuous re-detect must infer the boundary externally and read the emitted `<xx-XX>` tag (option 1) rather than a clean logit. Recommend shipping option 1 first (unblocks everyone, zero risk), then option 2.
- **Regression risk of the fix:** option 1 is additive (new getter) — none to EN path. Options 2/3 touch `reset()` semantics and carried state — must not regress the English-only Nemotron or the single-language `set_target_lang` happy path; guard with the local `./nemotron` + `./nemotron_multi` models.
- **Reverification:** re-read `nemotron.rs:475-509,591,691` and export script `:317-412` after any change; assert EN-only path still passes `prompt_index=None`.

### F2 — Nemotron (EN + multi) has no word timestamps: LIBRARY gap, not a graph limit `[SEVERITY: HIGH]`
- **Root cause:** `transcribe_chunk` (`nemotron.rs:615,714-720`) and `transcribe_audio` (`nemotron.rs:540,603-608`) decode token ids and immediately `decode_single` to text, discarding the per-frame index. `decode_chunk` (`nemotron.rs:723-770`) iterates encoder frame `t` in the exact loop where Multitalker captures `absolute_frame` (`multitalker.rs:610,633`) and Unified captures `absolute_frame` (`parakeet_unified.rs:467,488`).
- **Evidence it's a library gap:** Multitalker and Unified share the identical RNNT encoder/decoder shape and DO produce `(token_id, frame)` pairs → `TimedToken` → `group_by_words` (`multitalker.rs:644-663`, `parakeet_unified.rs:502-512`, `timestamps.rs:52`). The encoder frame rate is known (Multitalker uses `SECONDS_PER_ENCODED_FRAME=0.08`, `multitalker.rs:50`; Unified derives it at `parakeet_unified.rs:498-500`). Nothing in `model_nemotron.rs` blocks this — the encoded length and frame loop are right there (`nemotron.rs:734`).
- **Impact:** the flagship streaming engines can't drive subtitle/karaoke/word-align use cases; callers must switch to Unified (EN-only) or lose timing.
- **Recommendation:** add `transcribe_chunk_timed`/`get_timed_transcript(mode)` to `Nemotron` mirroring Unified's `(id, absolute_frame)` accumulation + `tokens_to_timed`. Reuse `timestamps::group_by_words`. Multilingual caveat: `group_by_sentences` keys on Latin `.?!` (`timestamps.rs:153-154`) so Sentences mode is unreliable for ja/zh/etc — document Words mode for multilingual.
- **Regression risk:** additive method; existing `String`-returning API unchanged.
- **Reverification:** confirm `nemotron.rs:734` frame loop still maps 1:1 to encoded frames after adding offset bookkeeping.

### F3 — No uniform quantization story: 6 bespoke file-pickers, inconsistent naming, half lack fp16 `[SEVERITY: MEDIUM]`
- **Root cause:** every loader hand-rolls its own candidate list with different conventions:
  - CTC: `model.onnx > model_fp16 > model_int8 > model_q4` (`parakeet.rs:91-96`) — only variant naming int4 (`_q4`) AND fp16.
  - TDT: `encoder-model.onnx > encoder.onnx > encoder-model.int8.onnx` (`model_tdt.rs:64-67`) — int8 last, no fp16.
  - Unified: `encoder.onnx > encoder.int8.onnx > encoder-model.onnx` (`model_unified.rs:60`) — no fp16.
  - Multitalker: int8-FIRST (`model_multitalker.rs:54-59`) — opposite priority to TDT/Unified.
  - Cohere: `*_quantized > * > *_fp16`, flat or `onnx/`-nested (`model_cohere.rs:68-91`) — yet another name (`_quantized`) + fp16.
  - Nemotron & EOU: fixed `encoder.onnx` only (`model_nemotron.rs:72`, `model_eou.rs:42`) — **cannot load the int8/int4 Nemotron files the README itself links** (`README.md:155`).
- **Impact:** the README advertises int8/int4 Nemotron (`README.md:155`) but the loader silently can't use them. Naming is fp32 across `model_int8`, `encoder.int8`, `encoder-model.int8`, `*_quantized`, `*.int8`. Inconsistent priority (int8-first vs int8-last) means two variants behave oppositely for the same on-disk layout.
- **Recommendation:** extract one shared `resolve_onnx_file(dir, role, prefer_quant)` helper (role ∈ encoder/decoder/model) with a canonical precedence and a documented `prefer: {Auto, Fp32, Fp16, Int8, Int4}` knob on `ExecutionConfig` (coordinate with A3 for the API shape). Wire Nemotron + EOU through it so the linked quantized Nemotron files load.
- **Regression risk:** changing precedence can silently switch which file loads for existing users; gate behind explicit `prefer` default = current per-variant behavior, or call it a documented 0.x breaking change (semver pre-1.0 per ledger).
- **Reverification:** for each variant, list the HF repo's actual filenames (`README.md:149-165`) and confirm the resolver picks the intended one.

### F4 — Cohere hardcodes `<|notimestamp|>` and `<|nodiarize|>` though the model supports both `[SEVERITY: MEDIUM]`
- **Root cause:** the decoder prompt always appends `t.notimestamp` and `t.nodiarize` (`cohere.rs:308-309`); the `<|timestamp|>`/`<|diarize|>` counterparts are never wired. The tokens are resolved but only the negative forms exist as constants (`cohere.rs:57-58,112-113`).
- **Evidence it's a graph capability, not a limit:** the model card / processor prompt structure documents `<|notimestamp|>` and `<|nodiarize|>` as toggles in the same slot family as `pnc`/`itn`, which ARE exposed as bool args (`cohere.rs:253-259,297-298`). The HF export is the standard Optimum `cohere_asr` graph (`cohere.rs:14-22`) — timestamps/diarization are generation-time prompt choices, not separate graphs.
- **Impact:** Cohere users can't get timestamps or speaker labels even though the model produces them; this is the only multilingual offline engine, so the gap is felt.
- **Recommendation:** add `timestamps: bool` and `diarize: bool` to `transcribe_audio` (or a `CohereOptions` builder to avoid a 6-arg call — coordinate with A3) and select the positive tokens. Parsing the emitted timestamp/speaker tokens into structured output is a follow-up (LOW), but the prompt toggle is cheap.
- **Regression risk:** changing the `transcribe_audio` signature is breaking; do it as a builder/options struct or a new method to keep `(audio, lang, pnc, itn)` working.
- **Reverification:** confirm `<|timestamp|>`/`<|diarize|>` literals exist in the tokenizer before relying on them (mirror `require_token`, `cohere.rs:397-402`).

### F5 — No unified entry point / backend selector `[SEVERITY: MEDIUM]`
Coordinate with lane 02 (architecture) and A3 (API surface).
- **Root cause:** every variant is a bespoke concrete type with a different constructor and return shape (`String` vs `TranscriptionResult` vs `Vec<SpeakerTranscript>`); only CTC/TDT/Unified/Multitalker implement the `Transcriber` trait (`transcriber.rs:7`), and Nemotron/EOU/Cohere/Sortformer do not. `lib.rs:76-99` exports 10 entry points with no facade.
- **Impact:** a caller wanting "give me the best ASR for this dir" must know the variant, its file layout, its return type, and whether it streams. High onboarding cost; the "jobs-to-be-done" of "transcribe this audio" has 10 different answers.
- **Recommendation:** a single `from_pretrained`-style enum/facade (e.g. `AsrEngine::auto(dir)`) that sniffs files (CTC `model.onnx`, TDT `vocab.txt`, Nemotron `tokenizer.model`+`prompt_index`, Cohere `tokenizer.json`+`<|startoftranscript|>`, etc. — detection precedent already exists for Nemotron EN-vs-multi at `model_nemotron.rs:107-127`) and returns a common streaming + offline interface. Word timestamps (F2) and a shared lang-select trait (F1/F4) should land FIRST so the facade exposes a consistent surface; otherwise the facade just papers over the inconsistency.
- **Regression risk:** purely additive if built as a new type over existing ones; do not remove the concrete types (back-compat).
- **Reverification:** ensure feature-gated variants (multitalker/cohere/sortformer) compile in/out of the facade behind their cfg flags (`lib.rs:62-71`).

### F6 — TDT advertises "auto language detection" but exposes no language API or detected-language output `[SEVERITY: LOW]`
- **Root cause:** README claims "25 languages with auto-detection" (`README.md:27,190`) but `ParakeetTDT` has no `set_language`, no `detected_language`, and `transcribe_samples` returns text only (`parakeet_tdt.rs:92-153`). Detection is fully internal to the graph.
- **Impact:** users can't bias toward a known language (accuracy left on the table) nor read back what was detected. Minor vs F1 because TDT genuinely auto-detects per-decode and is offline (no lock).
- **Recommendation:** if the TDT graph/tokenizer carries lang tags (assumption — verify against `vocab.txt`; defer model-detail confirmation to RU), surface a `detected_language()` like F1 option 1. Otherwise document the limitation. Low priority.
- **Regression risk:** additive.
- **Reverification:** inspect `vocab.txt` for `<lang>`-style entries; confirm against `decoder_tdt.rs`.

### F7 — Multitalker / ASR chunk-rate vs Sortformer stride mismatch (feature correctness) `[SEVERITY: LOW]`
- **Root cause:** the ASR chunk (~1.12 s in Normal mode) is far smaller than Sortformer's internal stride (~10 s); the code itself flags this and pads (`multitalker.rs:344-349`). Speaker masks are nearest-neighbour interpolated from a coarse diarization grid (`multitalker.rs:521-527`).
- **Impact:** speaker attribution quality on short chunks is degraded by design; documented as a known future improvement (decouple the two chunk rates) in the source comment.
- **Recommendation:** track as a known limitation; real fix (buffer audio for Sortformer, run ASR sub-chunks against its predictions) is a larger change — hand to A1/02 for correctness sequencing. Listed here for matrix completeness.
- **Regression risk:** n/a (no change proposed in this lane).
- **Reverification:** `multitalker.rs:344-364`.

---

## Cross-references for other lanes

- **→ A1 (correctness) [F1]:** the language-lock is a fixable library policy — the ONNX graph takes a fresh `prompt_index` every chunk (export script L317-412) and `set_target_lang` already mutates it (`nemotron.rs:489`). The hard constraint: the encoder emits NO language posterior (export L388-394); the only language signal is the inline `<xx-XX>` token the library currently discards (`nemotron.rs:511-521,716`). Any auto-continuous re-detect needs an external boundary signal (Nemotron has none; EOU does) + clearing carried decoder/cache state. A1 owns the reset/flush/carried-state correctness; please confirm whether resetting the joint/LSTM state while preserving the encoder cache yields valid continuation.
- **→ RU (upstream) [F1, F6]:** DEFERRED questions — (a) does NeMo's reference multilingual streaming re-run `auto` detection per buffer or set the prompt once? If per-buffer, our `reset()`-preserves-language policy (`nemotron.rs:495-509`) is the entire bug. (b) Does NeMo expose a silence/utterance boundary for Nemotron streaming that we could re-prompt at? (c) Does TDT v3 carry a language tag we can surface (F6)?
- **→ A3 (API/ergonomics) [F1, F3, F4, F5]:** new surfaces proposed — `detected_language()` getter (F1), `prefer: Quantization` knob on `ExecutionConfig` + shared resolver (F3), `CohereOptions` builder to avoid signature breakage (F4), and the `Transcriber` trait gap (Nemotron/EOU/Cohere/Sortformer don't implement it — `transcriber.rs:7` vs `lib.rs`). The unified facade (F5) depends on A3 settling a common return type (`String` vs `TranscriptionResult` vs per-speaker vecs).
- **→ 02 (architecture) [F2, F3, F5]:** three shared abstractions worth extracting WITHOUT breaking the public API — (1) `(token_id, frame)` → `TimedToken` accumulation is copy-pasted in Unified (`parakeet_unified.rs:442-512`) and Multitalker (`multitalker.rs:593-663`) and MISSING in Nemotron (`nemotron.rs:723-770`); unify it (F2). (2) a single `resolve_onnx_file` (F3) replaces 6 bespoke pickers. (3) the variant facade (F5).
- **→ A5 (quality) [F3]:** the 6 divergent quantized-file pickers (`parakeet.rs:91-96`, `model_tdt.rs:64-99`, `model_unified.rs:60-80`, `model_multitalker.rs:54-79`, `model_cohere.rs:68-91`, and the two hardcoded ones) are a duplication + inconsistency smell beyond just the feature gap.
- **→ A6 (tests) [F1, F2]:** add regression fixtures using local `./nemotron` (EN) + `./nemotron_multi` — (a) assert EN-only path keeps `prompt_index=None` and never breaks after any lang-redetect work; (b) when word timestamps are added to Nemotron, assert frame→seconds alignment matches Unified/Multitalker conventions (`multitalker.rs:50`, `parakeet_unified.rs:498-500`).
