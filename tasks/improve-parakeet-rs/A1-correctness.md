# A1 — AUDIT: Correctness & Streaming

**/think frameworks:** inversion ("how would each streaming path silently produce wrong text?") + five-whys (trace each defect to its root, not its symptom) + scientific-method (every claim is a falsifiable statement tied to file:line; I label hypotheses I could not fully prove against a live model).

**Scope audited:** `src/nemotron.rs`, `src/model_nemotron.rs`, `src/parakeet_eou.rs`, `src/model_eou.rs`, `src/parakeet_unified.rs`, `src/model_unified.rs`, `src/decoder.rs`, `src/decoder_tdt.rs`, `src/audio.rs`, `src/timestamps.rs`, `examples/streaming_mic.rs`.

**Method note:** I could read all code but could not run the ONNX graphs (no inference in this lane). Findings tied to graph-internal behavior (the language-lock, exact frame conditioning) are labelled as such and deferred to RU-upstream / A4-models for the model-internal half; the *library-side* half is proven from code here.

---

## Ranked findings

### A1-01 — Multilingual `"auto"` cannot code-switch mid-stream; `reset()` does not re-enable detection — CRITICAL (the seed finding)

**Root cause (library half, proven):** `prompt_index` is set exactly once. `from_shared` defaults multilingual instances to `Some(101)` (`nemotron.rs:426-429`). It only changes through `set_target_lang` (`nemotron.rs:475-491`). Every chunk feeds that same fixed value to the encoder: `transcribe_chunk` passes `self.prompt_index` at `nemotron.rs:691` and `transcribe_audio` at `nemotron.rs:591`. `reset()` (`nemotron.rs:495-509`) clears `encoder_cache`, `state_1/2`, `last_token`, audio buffer and accumulated tokens but **deliberately preserves `prompt_index`** (no line touches it; doc comment at 493-494 confirms intent). So even an utterance-boundary `reset()` re-runs with the identical prompt slot.

**Why it locks (model half, two compounding mechanisms):**
1. **Fixed prompt conditioning.** `"auto"` (index 101) is a real prompt-embedding slot (`nemotron.rs:48`), not a per-chunk re-detection. The encoder is conditioned on whatever single slot it is handed. Whether index 101 internally re-detects per chunk is a property of the ONNX graph that this lane cannot observe. **Deferred to RU-upstream / A4-models.**
2. **Carried autoregressive state.** The decoder is autoregressive: `last_token` carries across chunks (`nemotron.rs:763`, used at `742`) and the encoder cache (`NemotronEncoderCache`) plus LSTM `state_1/2` carry across chunks (`694`, `764-765`). Once a language's tokens dominate, the carried state biases the next prediction toward the same language even if the encoder *would* re-detect. This half is library-controllable: a boundary `reset()` does clear `last_token` and states (`504`, `502-503`) — but `reset()` is never called automatically and is left entirely to the caller, and even when called it keeps the same prompt.

**Evidence:** `nemotron.rs:48, 426-429, 475-491, 495-509, 591, 691, 694, 742, 763-765`. Seed reproduction: live mic test with `streaming_mic auto` (example feeds `auto` and never resets per utterance — `streaming_mic.rs:66, 161-168`).

**Impact:** Multilingual `"auto"` produces a garbled mixed-language prefix while the model "settles," then locks to the first language that accumulates state and cannot return to a second language in the same stream. For any code-switching or multi-speaker-language session the transcript is wrong after the first language commits. This is the user-observed defect.

**Honest model-inherent vs library-fixable split:**
- *Library-fixable now:* (a) make `set_target_lang` mid-stream effective by documenting that it requires a `reset()` to flush carried decoder/encoder state (today `set_target_lang` only swaps the prompt; the carried state still biases — the doc at `473-474` already hints this but the API does not enforce it); (b) expose an explicit "re-detect at boundary" helper that pairs a prompt re-set with a state `reset()`; (c) offer a `reset_with_lang(lang)` convenience so callers can re-prompt + flush atomically; (d) distinguish "detect-once" from "detect-continuous" semantics in docs.
- *Possibly model-inherent:* whether index 101 re-detects per encode at all. If the graph conditions on a static prompt embedding with no internal per-chunk LID, then continuous code-switching under a single `"auto"` stream is **not achievable in the library** without re-prompting at boundaries (which needs an external VAD/boundary signal the crate does not currently produce). **RU-upstream must confirm against NeMo before A1/A4 state this as fixed-or-not.**

**Regression risk of fix:** LOW for additive APIs (`reset_with_lang`, docs). MEDIUM if `set_target_lang` is changed to auto-`reset()` (silently discards in-flight partial tokens — would surprise callers relying on current behavior; gate behind a new method instead). No risk to the English-only path (`prompt_index == None`, `nemotron.rs:427-428`).

**Reverification:** RU confirms whether NeMo's `"auto"` does per-buffer LID. Then: integration test on `./nemotron_multi` feeding an EN segment then a HI segment in one stream, asserting the EN tail is recognized as EN after a `reset_with_lang`/boundary reset; and asserting it is NOT recognized without the reset (documents the inherent limit). Coordinate with A4-models (do not duplicate the language-coverage / re-detect-API design — that is A4's; A1 owns only the streaming-state correctness claim).

---

### A1-02 — Nemotron streaming silently drops the final partial chunk (no flush API) — HIGH

**Root cause:** `transcribe_chunk` returns early when fewer than `CHUNK_SIZE` (56) new mel frames are available: `if available_new_frames < CHUNK_SIZE { return Ok(String::new()); }` (`nemotron.rs:637-639`). There is **no flush method** on `Nemotron`. Trailing audio shorter than one full chunk (~560 ms) is never encoded. The only way to flush is the example's manual trick of feeding zero-padded full-size chunks (`streaming_mic.rs:176-188`), which (a) is undocumented as the required pattern and (b) zero-pads *real trailing audio up to a full chunk* (`out16k.resize(CHUNK_SIZE, 0.0)`, `streaming_mic.rs:177`) — the padding silently corrupts the mel of the genuine tail frames near the boundary.

**Evidence:** `nemotron.rs:615-639`; absence of any `flush`/`finalize` method in `impl Nemotron` (contrast `ParakeetUnified::flush`, `parakeet_unified.rs:308-310`). Example workaround: `streaming_mic.rs:176-188`.

**Impact:** Last up-to-560 ms of every utterance is dropped unless the caller knows the zero-pad incantation; even then the boundary mel is distorted. End-of-sentence words are lost — directly degrades transcription accuracy at every stream end.

**Regression risk of fix:** LOW. Add a `flush(&mut self) -> Result<String>` that processes the remaining `< CHUNK_SIZE` frames using the *actual* frame count (mirroring `transcribe_audio`'s `chunk_length = PRE_ENCODE_CACHE + main_len`, `nemotron.rs:581`) rather than a forced full-size window. Purely additive.

**Reverification:** Test feeding N chunks of CHUNK_SIZE plus a 200 ms tail; assert `flush()` returns non-empty and that the tail word appears. Compare WER of streamed-with-flush vs `transcribe_audio` (offline) on the same WAV.

---

### A1-03 — Nemotron `transcribe_chunk` passes a constant `length` to the encoder; offline path passes the true length — HIGH

**Root cause:** Streaming `transcribe_chunk` always passes `expected_size as i64` (= `PRE_ENCODE_CACHE + CHUNK_SIZE` = 65) as the encoder `length` (`nemotron.rs:689`), regardless of how many real frames the window holds. The offline `transcribe_audio` correctly passes `chunk_length = PRE_ENCODE_CACHE + main_len` (`nemotron.rs:581, 589`). For interior chunks the streaming window is always full so the two agree; but the guard at `637-639` means the streaming path never *reaches* a short window — which is exactly why A1-02's dropped tail exists. The two code paths encode the same audio with **different `length` semantics**, so any future flush must use the offline convention, and the divergence today is a latent correctness inconsistency between the offline and streaming results.

**Evidence:** `nemotron.rs:581, 589` (offline true length) vs `nemotron.rs:689` (streaming constant length).

**Impact:** Offline `transcribe_audio` and streamed `transcribe_chunk` on identical audio can diverge at the final chunk (offline encodes the true short tail; streaming drops it). Two public entry points give different transcripts for the same input — a correctness/consistency bug and a testing hazard (you cannot use offline as the streaming oracle).

**Regression risk of fix:** LOW; aligning the streaming flush to the offline `length` convention is the natural fix and pairs with A1-02.

**Reverification:** Assert `transcribe_audio(wav)` == concatenation of `transcribe_chunk` calls + `flush()` over the same wav (token-level), within tolerance.

---

### A1-04 — EOU encoder length/`pre_encode_cache` are hardcoded and mismatch on the first chunks — HIGH

**Root cause:** EOU slices `SLICE_LEN = PRE_ENCODE_CACHE(9) + FRAMES_PER_CHUNK(16) = 25` frames from the tail of the full-buffer features (`parakeet_eou.rs:159-165`) and passes `time_steps` (the slice length) as the encoder `length` (`parakeet_eou.rs:174`). But:
1. `FRAMES_PER_CHUNK = 16` is a fixed assumption for a "160 ms chunk." If the caller passes a chunk of a different size (the API doc only *suggests* 2560 samples, `parakeet_eou.rs:126-127`, it is not enforced), the number of genuinely-new frames per call is not 16, so the slice either re-encodes already-seen frames (duplication) or skips new ones (loss). There is no `audio_processed`-style cursor as Nemotron has; EOU re-slices the tail of the rolling buffer every call and relies entirely on the chunk being exactly 16 frames of new audio.
2. Until the buffer reaches `MIN_BUFFER_SAMPLES` (1 s, `parakeet_eou.rs:147-150`), nothing is emitted, so the first ~1 s is buffered then encoded as one 25-frame slice — the pre-encode cache region for that first slice contains *real early audio*, not zero context, which differs from the cache-aware contract the encoder expects on a cold start (`EncoderCache::new()` zeros, `model_eou.rs:21-27`).

**Evidence:** `parakeet_eou.rs:147-165, 174`; no per-stream processed-sample cursor anywhere in `ParakeetEOU`.

**Impact:** Token duplication or loss at chunk boundaries whenever the input chunk size is not exactly 16 new frames; degraded recognition on the first second. Because the slice is recomputed from the rolling buffer each call rather than advanced by a cursor, the per-call "new frame" assumption is fragile and undocumented as a hard requirement.

**Regression risk of fix:** MEDIUM. Introducing a processed-sample cursor (as Nemotron has) changes the slicing math; must be validated against the EOU graph's expected frame stride. Lower-risk interim fix: document and enforce the exact required chunk size and reject others with a clear error (input validation at the boundary).

**Reverification:** Feed the same audio as (a) one chunk and (b) two half-chunks; assert identical token stream. Today this likely fails.

---

### A1-05 — EOU `reset_on_eou` resets decoder state but keeps the carried `last_token` flowing through one extra step; encoder cache intentionally retained — MEDIUM

**Root cause:** On EOU detection with `reset_on_eou`, the code drops the model lock, calls `reset_states()` (zeros `state_h/c`, sets `last_token = blank`), and returns `text_output + " [EOU]"` (`parakeet_eou.rs:221-226, 247-255`). `reset_states` deliberately keeps `encoder_cache` and `audio_buffer` (`parakeet_eou.rs:248-254`). That is a defensible "soft reset" for continuous context, but it means after an utterance boundary the encoder still carries acoustic context from the *previous* utterance across the EOU, while the decoder is hard-reset — an asymmetry that can produce a spurious leading token on the next utterance. Also, the EOU token itself is detected but the loop `break`s without consuming further symbols for that frame even when `!reset_on_eou` (`parakeet_eou.rs:221-228`), so a frame that emits `<EOU>` then a real token in `max_symbols` would lose the trailing real token.

**Evidence:** `parakeet_eou.rs:217-255`.

**Impact:** Possible spurious/dropped token at utterance boundaries; the decoder/encoder reset asymmetry is undocumented as a deliberate accuracy tradeoff. Lower severity because EOU is a specialized path and the soft-reset is intentional, but the `<EOU>`-then-token loss is a real edge case.

**Regression risk of fix:** MEDIUM (changing reset symmetry alters streaming behavior; needs A/B against reference). LOW for the loop fix (continue scanning symbols after a non-reset EOU rather than `break`).

**Reverification:** Construct audio with two short utterances separated by a pause; assert no spurious token at the second utterance start and no dropped final token of the first.

---

### A1-06 — Decoder greedy argmax does not break ties deterministically and ignores `NaN` in Nemotron/unified — MEDIUM

**Root cause:** Three decoders pick argmax differently:
- Nemotron `decode_chunk`: manual loop with `v > max_val`, seeded `max_val = NEG_INFINITY`, `max_idx = 0` (`nemotron.rs:749-756`). A leading `NaN` leaves `max_idx = 0` (it never compares true), so a NaN logit silently selects token 0. No `is_finite` filter (EOU *does* filter, `parakeet_eou.rs:212`).
- Unified `decode_encoder_frames`: `max_by(partial_cmp ... unwrap_or(Equal))` (`parakeet_unified.rs:477-482`). On `NaN`, `partial_cmp` returns `None` → treated as `Equal` → `max_by` keeps the *last* max-equal element; ties resolve to the highest index, opposite of Nemotron's first-wins.
- CTC `decode` / `decode_with_timestamps`: same `max_by ... unwrap_or(Equal)` (`decoder.rs:49-53, 137-141`).

**Evidence:** `nemotron.rs:749-756`, `parakeet_unified.rs:477-482`, `decoder.rs:49-53, 137-141`, `parakeet_eou.rs:208-215`.

**Impact:** (a) Inconsistent tie-breaking across variants is a latent reproducibility/divergence-from-reference issue (NeMo greedy uses first-wins argmax). (b) A `NaN` logit in Nemotron is silently decoded as token 0 (often a real token), corrupting output instead of being skipped; unified/CTC bury NaN as "Equal." None of these crash, so they are silent.

**Regression risk of fix:** LOW. Standardize on first-wins argmax with explicit `is_finite` guard across all four decoders (extract a shared helper). Matches EOU's existing finite check and NeMo semantics.

**Reverification:** Unit test the shared argmax helper: ties pick lowest index; a `NaN` is skipped; matches a known NeMo example's token sequence on a fixture.

---

### A1-07 — `transcribe_chunk` recomputes the full mel over the entire retained buffer every call — MEDIUM (correctness-adjacent; perf overlaps A2)

**Root cause:** Each `transcribe_chunk` call runs `compute_mel_spectrogram(&self.audio_buffer)` over the *whole* retained buffer (`nemotron.rs:628`), then derives the new-frame window from `audio_processed / HOP_LENGTH` (`nemotron.rs:633`). Because the buffer is trimmed only when it exceeds `keep_samples * 2` (`nemotron.rs:705-712`), the recomputed mel region shifts under the `processed_mel_frames` cursor after a trim. The trim adjusts `audio_processed` by `actual_remove` (`nemotron.rs:709-711`), but `actual_remove = remove.min(self.audio_processed)` can be *less than* `remove` when `audio_processed < remove`, so samples are drained from the front while the processed cursor under-decrements — **the cursor and buffer can desynchronize**, causing the next chunk's `main_start` (`nemotron.rs:647`) to point at the wrong mel frame (re-encoding or skipping frames).

**Evidence:** `nemotron.rs:628, 633, 700, 705-712`. The desync requires `audio_processed < remove`, which is reachable if a caller feeds one very large chunk (so `audio_buffer` >> `keep_samples*2` while `audio_processed` is still small after the first advance).

**Impact:** Frame drift on large/irregular chunk sizes → duplicated or skipped tokens. Reachable via the public API (chunk size is caller-controlled). Also the full-buffer mel recompute is O(buffer) per call (A2 perf concern), but the *correctness* risk is the cursor/buffer desync.

**Regression risk of fix:** MEDIUM. Replace the ad-hoc trim with a frame-accurate ring (drain in whole-`HOP_LENGTH` multiples and decrement `audio_processed` by exactly the drained sample count, asserting they stay aligned). Must re-verify interior-chunk equivalence.

**Reverification:** Property test: feed the same total audio split into many random chunk sizes vs one stream of uniform chunks; assert identical accumulated token stream. Add a debug assertion that `audio_processed % HOP_LENGTH == 0` and `audio_processed <= audio_buffer.len()*?` after every trim.

---

### A1-08 — `is_lang_tag` / `lang_tag_ids` stripping is heuristic and can over- or under-strip — MEDIUM

**Root cause:** Language tags are detected purely by shape: `<xx>` (two lowercase) or `<xx-XX>` (lower-lower-dash-upper-upper) (`nemotron.rs:78-93`). Real vocabulary pieces that happen to match this shape (e.g. a legitimate `<no>` or `<so>` content token, or any 2-lowercase-letter bracketed piece) would be silently stripped from the transcript (`get_transcript` filter, `nemotron.rs:516-519`; per-chunk filter `716`). Conversely, tags the model emits in other casings/locale forms not matching the 2/5-length pattern are *not* stripped and leak into output. The detection is decoupled from the actual prompt dictionary / the model's true special-token set.

**Evidence:** `nemotron.rs:78-93, 256-262, 516-519, 716`.

**Impact:** Silent text corruption (dropped or leaked tokens) for the multilingual variant, especially for the broad/adaptation-tier languages whose tag forms may differ. Low frequency but silent.

**Regression risk of fix:** LOW. Strip by exact token-id membership derived from the tokenizer's declared special tokens / the known prompt-dictionary locale set, not by string shape. Additive/internal.

**Reverification:** Assert that exactly the model's declared language/special tokens are stripped on a fixture, and that a content token of shape `<xx>` survives.

---

### A1-09 — `hann_window` uses `cos` symmetric (`win_length - 1`) denominator; NeMo/torch default is periodic — MEDIUM (accuracy vs reference)

**Root cause:** `hann_window` computes `0.5 - 0.5*cos(2πi/(N-1))` (`audio.rs:64-68`), i.e. a **symmetric** Hann window. `torch.stft` / `torchaudio` (what NeMo trains with) default to **periodic** windows: `0.5 - 0.5*cos(2πi/N)`. The off-by-one in the denominator slightly changes every window coefficient, which perturbs the spectrogram the model sees relative to training. This shared `stft` feeds Nemotron, EOU, unified, and CTC mel extraction.

**Evidence:** `audio.rs:64-68`; used by `stft_with_plan` (`audio.rs:99`) which every model path calls. Compare to the NeMo `audio_preprocessing.py` reference cited in `nemotron.rs:13-14`.

**Impact:** Systematic small mismatch from the training-time features across *all* variants → accuracy degradation that is hard to spot (everything "works" but WER is a few points worse than reference). This is the kind of numerical divergence the contract explicitly asks for.

**Regression risk of fix:** MEDIUM — changing the window changes every transcript slightly; must be validated as *closer* to reference, not just different. Confirm against the exact NeMo `STFT`/`FilterbankFeatures` window setting (RU/A4 can confirm whether NeMo uses `periodic=True`). Label: **hypothesis pending RU confirmation of NeMo's window periodicity**, but the symmetric-vs-periodic discrepancy with torch defaults is real.

**Reverification:** Dump mel for a fixed WAV from NeMo's preprocessor and from this `stft`; compare per-bin error with periodic vs symmetric window; pick the lower-error one.

---

### A1-10 — Inconsistent log-mel floor across paths: EOU clamps `x.max(0.0)`, Nemotron does not; CTC/unified normalize, Nemotron does not — MEDIUM (accuracy vs reference)

**Root cause:** Three different log-mel treatments for the same family of models:
- Nemotron: `(x + 5.96e-8).ln()`, no clamp, no normalization (`nemotron.rs:784`). Comment says NeMo feeds raw log-mel (`nemotron.rs:772-774`).
- EOU: `(x.max(0.0) + 5.96e-8).ln()` — extra `max(0.0)` clamp (`parakeet_eou.rs:261`).
- CTC/unified (`audio.rs`): `(x + 2^-24).ln()` then **per-feature mean/std normalization with Bessel correction** (`audio.rs:240-263`).

`5.96e-8` and `2^-24` are the same value (fine). The substantive divergences: (a) EOU's `max(0.0)` is a no-op for true power spectrograms (`norm_sqr` ≥ 0) *unless* the mel basis or numerical noise produces a tiny negative, so it is harmless but inconsistent; (b) the real risk is whether Nemotron *should* normalize. The doc comment claims NeMo does not normalize for this model (`nemotron.rs:291, 772-774`), which may be correct for Nemotron but is asserted, not verified.

**Evidence:** `nemotron.rs:772-784`, `parakeet_eou.rs:257-263`, `audio.rs:238-263`.

**Impact:** If the Nemotron "no normalization" assumption is wrong, *all* Nemotron output is systematically off from reference (the model would see un-normalized features it was trained to receive normalized). High blast radius if the assumption is incorrect; zero if correct. Cannot resolve in this lane.

**Regression risk of fix:** HIGH if normalization is added/removed incorrectly (changes every Nemotron transcript). Do nothing until RU confirms the NeMo `normalize` setting for `nemotron-3.5-asr-streaming-0.6b` and the English-only 0.6B.

**Reverification:** RU/A4 confirm NeMo preprocessor `normalize` field per model. Then mel-dump comparison against reference as in A1-09.

---

### A1-11 — `mel_spectrogram` returns early without normalization when `num_frames <= 1` — LOW

**Root cause:** In `extract_features_with_cache`, if `num_frames <= 1` the function returns the un-normalized mel (`audio.rs:249-251`) because variance is undefined for a single frame. Correct guard, but it means a 1-frame input gets a *different* feature scale than multi-frame input (un-normalized vs normalized), and the std uses Bessel `(N-1)` (`audio.rs:257`) which for `N=2` doubles the variance estimate vs population std. Edge case for ultra-short audio.

**Evidence:** `audio.rs:249-263`.

**Impact:** Tiny/degenerate inputs to CTC/unified produce mis-scaled features → garbage tokens, but only for sub-30 ms audio. Low reachability.

**Regression risk of fix:** LOW; clamp/return empty for `num_frames == 0` (already handled by `stft`) and document the `<=1` behavior.

**Reverification:** Test 1-frame and 2-frame inputs do not panic and produce empty/sane output.

---

### A1-12 — Unified `flush` right-context handling can emit a partial final chunk with truncated future context — LOW

**Root cause:** During `flush`, `available_right` is clamped to whatever audio remains (`parakeet_unified.rs:388-394`) and `build_window_audio` zero-pads the missing right context implicitly (the window vec is zero-initialized, `parakeet_unified.rs:403`, and only the available region is copied, `408-414`). The final chunk therefore sees zero-padded right context, which the encoder was not necessarily trained to expect at that position. This is the standard streaming flush tradeoff and is far better than Nemotron's silent drop (A1-02), but the final chunk's accuracy is reduced and this is undocumented.

**Evidence:** `parakeet_unified.rs:308-310, 367-369, 388-394, 403-414`.

**Impact:** Slightly degraded accuracy on the last `chunk_secs` of a unified stream. Acceptable and standard; flagged for completeness and docs.

**Regression risk of fix:** N/A (document; optionally hold back the last chunk until enough right context arrives, but that adds latency).

**Reverification:** Compare flushed last-chunk tokens vs offline tail tokens on a fixture.

---

### A1-13 — `decode_chunk` `max_symbols_per_step = 10` (Nemotron) / `5` (EOU) silently caps emissions, risking dropped tokens on dense frames — LOW

**Root cause:** The per-frame symbol cap differs: Nemotron 10 (`nemotron.rs:726`), EOU 5 (`parakeet_eou.rs:198`), unified 10 (`parakeet_unified.rs:23`). When a frame legitimately emits more non-blank symbols than the cap, the loop just stops without a blank — tokens are dropped with no signal. The cap exists to prevent infinite loops on a stuck `last_token`, but the value is a magic number and the silent truncation is not logged.

**Evidence:** `nemotron.rs:726, 741`, `parakeet_eou.rs:198`, `parakeet_unified.rs:23, 469`.

**Impact:** Rare token loss on dense audio (fast speech, dense languages). Low reachability with these caps but real, and inconsistent across variants.

**Regression risk of fix:** LOW; unify the constant and add a debug log/counter when the cap is hit so it is observable.

**Reverification:** Synthetic frame that wants >cap symbols; assert behavior and that the cap-hit is observable.

---

## Cross-references for other lanes

- **RU-upstream (BLOCKING for A1-01, A1-09, A1-10):** Resolve three model-internal questions against the NeMo reference: (1) does `nemotron-3.5` prompt index 101 (`"auto"`) perform per-buffer language ID, or does it condition on a static prompt embedding (determines whether continuous code-switch is achievable in-library at all)? (2) Does NeMo's `FilterbankFeatures`/STFT use a periodic Hann window (`periodic=True`)? (3) What is the NeMo preprocessor `normalize` setting for the Nemotron English-only 0.6B and multilingual 3.5 (per-feature normalize, or none as the code assumes)? A1 cannot finalize A1-09/A1-10 severity without (2)/(3).
- **A4-models (owns the re-detect API design; A1 owns streaming-state correctness):** A1-01's library-side remedies (`reset_with_lang`, boundary re-prompt+reset, detect-once vs detect-continuous semantics) are *correctness* hooks; the *feature/API surface* for language re-detection belongs to A4. Coordinate so the synthesis lists one task, not two. A1 also notes EOU/Nemotron lack timestamps that unified/CTC have — variant feature gap is A4's.
- **A2-performance:** A1-07 (full-buffer mel recompute every `transcribe_chunk`) is also a hot-path allocation/CPU concern — A2 should cost it. The per-chunk `.clone()` of every encoder cache tensor (`model_nemotron.rs:150-154`, `model_eou.rs:82-85`) and per-decoder-step state clones (`model_nemotron.rs:247-248`) are A2's allocation findings; A1 only flags them where they affect correctness (cursor desync A1-07).
- **A3-api:** A1-02 (missing `flush` on Nemotron/EOU while unified has one) is an API-consistency gap; A1-04 (EOU requires an exact undocumented chunk size with no validation) is an input-validation/ergonomics gap; A1-06/A1-13 (divergent argmax + magic caps across variants) argue for a shared decoder helper that A3's "common trait/shape" analysis should account for.
- **A5-quality:** A1-06, A1-10, A1-13 all stem from duplicated, slightly-divergent decode/mel/argmax logic across the four wrapper pairs — strong evidence for A5's shared-abstraction extraction (a single greedy-RNNT decode + single mel front-end). The three different log-mel floors (`5.96e-8` vs `2^-24` vs `+max(0.0)`) are duplication-induced drift.
- **A6-tests:** Every finding above lists a reverification test. Highest-value unguarded invariants for A6 to prioritize: (1) offline-vs-streamed token equivalence per model (catches A1-02/03/07), (2) chunk-size-invariance property test (catches A1-04/07), (3) argmax tie/NaN unit test (A1-06), (4) mel-vs-NeMo-reference dump (A1-09/10), (5) language-lock integration test on `./nemotron_multi` (A1-01). No `tests/` dir exists today, so these are all net-new.
