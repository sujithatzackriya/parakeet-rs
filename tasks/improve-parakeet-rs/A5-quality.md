# A5 — AUDIT: Code Quality & Structure

**/think frameworks:** (1) **DRY / single-source-of-truth analysis** (locate the canonical concept, then count its copies and measure drift between them) and (2) **map-vs-territory** (compare the user's own coding-style rules — many small files < 800 lines, functions < 50 lines, no mutation, comprehensive error handling — against the actual crate to surface the gap). Duplication is treated as the highest-value structural finding per the lane brief.

**Scope read:** all 6 `model_X.rs` low-level wrappers (`model_nemotron` 288, `model_tdt` 292, `model_eou` 196, `model_unified` 187, `model_multitalker` 250, `model_cohere` 312), `model.rs` (CTC), the largest high-level wrappers (`sortformer` 1254, `nemotron` 786, `multitalker` 771, `parakeet_unified` 590, `cohere` 464), plus `error.rs`, `execution.rs`, `audio.rs`, `decoder.rs`, `transcriber.rs`, `vocab.rs`. Idiom scan run crate-wide (grep counts cited).

**Verdict up front:** the crate is functionally coherent and the `model_X / X` split is a sound instinct, but the six `model_X` files are ~70-90% mechanical copies of each other, the audio front-end exists in 3-4 parallel implementations with duplicated constants, and the two largest files (`sortformer.rs` 1254, `nemotron.rs` 786) violate the user's own < 800-line / < 50-line-function rules. Error handling is structurally fine (a real `enum`, locks use `map_err` not `.unwrap()`) but is almost entirely stringly-typed at the call sites and loses source errors. Findings are ranked; the consolidation finding (Q1) is the load-bearing one and is handed to lanes 02/A3.

---

## Findings (ranked)

### Q1 — CRITICAL — Massive duplication across the six `model_X.rs` ONNX wrappers

**Root cause:** each model family got its own hand-written `model_X.rs` by copy-paste from the previous one, with no shared "RNNT ONNX session" abstraction. The result is six near-identical files implementing the same four concerns: (a) load encoder+decoder sessions, (b) locate the `.onnx` files, (c) run the encoder and rebuild an `ArrayN` from `(shape, data)`, (d) run the RNNT decoder step and rebuild two LSTM states.

**Evidence (parallel line ranges — these are near character-identical):**

- **EP / session setup is copied verbatim 6 times.** The exact 6-line block
  ```
  let builder = Session::builder()?;
  let mut builder = exec_config.apply_to_session_builder(builder)?;
  let encoder = builder.commit_from_file(&encoder_path)?;
  let builder = Session::builder()?;
  let mut builder = exec_config.apply_to_session_builder(builder)?;
  let decoder_joint = builder.commit_from_file(&decoder_path)?;
  ```
  appears at `model_nemotron.rs:88-94`, `model_tdt.rs:47-54`, `model_eou.rs:53-60`, `model_unified.rs:44-50`, `model_multitalker.rs:85-91`, `model_cohere.rs:95-101`, and once more in `model.rs:27-29` (single-session CTC) and `sortformer.rs:226`. Confirmed by `grep -c "Session::builder()"`: 2 occurrences in each of the 6 dual-session files.

- **The RNNT decoder step is the same function five times.** `run_decoder` returning `(Array1<f32>, Array3<f32>, Array3<f32>)` by extracting `outputs["outputs"]`, then rebuilding two states from `(h_shape,h_data)` / `(c_shape,c_data)`:
  - `model_nemotron.rs:232-287` (56 lines)
  - `model_unified.rs:134-186` (52 lines) — character-identical to Nemotron except the targets are built with `Array2::from_elem` vs `from_shape_vec`
  - `model_multitalker.rs:197-249` (52 lines) — identical except input names (`states_1` vs `output_states_1`) and no `target_length`
  - `model_eou.rs:149-195` (46 lines) — identical shape-rebuild, returns logits as `Array3` instead of `Array1`
  - `model_tdt.rs:214-271` — same step inlined inside `greedy_decode` instead of factored out
  The only real differences are: input-tensor names, `i32` vs `i64` targets, and whether `target_length` is passed. Everything else (clone tensors in, `try_extract_tensor`, `from_shape_vec` with `.map_err(|e| Error::Model(format!(...)))`, return tuple) is duplicated.

- **The encoder run + output-rebuild is duplicated.** The "extract `outputs[name]`, read `shape.as_ref()`, `Array3::from_shape_vec((b,d,t), data.to_vec())`" pattern appears in every `run_encoder`: `model_nemotron.rs:164-177`, `model_eou.rs:88-98`, `model_unified.rs:110-131`, `model_multitalker.rs:127-137`, `model_tdt.rs:150-170`, `model_cohere.rs:116-128`. The cache-rebuild block (3 tensors -> `Array4/Array4/Array1`) is **character-identical** between `model_nemotron.rs:197-225` and `model_eou.rs:114-142` and `model_multitalker.rs:156-184` — same 28-line body, three times.

- **The two streaming `EncoderCache` structs are the same struct twice.** `NemotronEncoderCache` (`model_nemotron.rs:13-32`) and `EncoderCache` (`model_eou.rs:10-28`) have identical fields (`cache_last_channel: Array4`, `cache_last_time: Array4`, `cache_last_channel_len: Array1<i64>`) and identical constructors differing only in hardcoded dims; `MultitalkerEncoderCache` (`model_multitalker.rs:13-37`) is the same struct with axis order `[1, n_layers, ...]` vs `[n_layers, 1, ...]`.

- **`find_encoder` / `find_decoder_joint` candidate-path search is duplicated** with drifting candidate lists: `model_tdt.rs:63-109`, `model_unified.rs:59-91`, `model_multitalker.rs:55-83` (inline), `model_cohere.rs:256-268` (`find_file`). Four independent implementations of "try these filenames in order, else error".

**Impact:** ~700-900 lines of the crate's ~7,766 are mechanical duplicates. Any bug fix to encoder output handling, cache rebuild, or EP setup must be applied in 6 places (and historically has not been — see drift below). New model families cost a full file of boilerplate. This is the single largest maintainability liability.

**Drift already present (proof the duplication is actively harmful):**
- `model.rs:33` (CTC) builds input as `(batch, time, features)` but `model_tdt.rs:131-135` and `model_unified.rs:97-101` build `(batch, features, time)` via `.t()` — same conceptual step, divergent code.
- mel log-guard differs silently: `nemotron.rs:784` uses `(x + guard).ln()` while `multitalker.rs:675` uses `(x.max(0.0) + guard).ln()` — same "compute_mel_spectrogram" name, different numerics (see Q2).

**Regression risk of the fix:** MEDIUM. Tensor I/O names and axis orders genuinely differ per export, so the abstraction must be parameterized, not collapsed. Mis-parameterizing (e.g. wrong axis order for multitalker's batch-first cache) would silently corrupt streaming. Must be done behind the existing public API (`model_X` are mostly `pub(crate)` already: multitalker/cohere are `pub(crate)`, nemotron/tdt/eou/unified are `pub` but only used internally) so it is a non-breaking internal refactor.

**Recommendation (hand to 02/A3):** extract a shared internal module, e.g. `src/onnx/`:
1. `fn run_dual_session(model_dir, exec_config) -> (Session, Session)` + a `find_first_existing(dir, &[&str])` helper — removes the EP-setup and file-discovery duplication outright (8 + 4 copies -> 1 each).
2. A `RnntDecoderStep` helper parameterized by `{input_names, target_dtype, pass_target_length}` that runs one decoder step and returns `(logits, state_1, state_2)` — collapses 5 copies.
3. A generic `extract_array3(outputs, name) -> Array3<f32>` / `extract_array4` / `rebuild_cache` helper (the `(shape, data) -> from_shape_vec` + `.map_err(Error::Model)` boilerplate) — removes the 28-line cache-rebuild triplicate and ~15 single-array rebuilds.
4. Unify the three streaming cache structs into one `StreamingCache { axis_order }` enum-tagged type.
This is a pure internal refactor, ideal as 2-3 sequenced one-PR tasks gated by golden-output tests (see A6).

**Reverification:** `grep -c "Session::builder()" src/model_*.rs` (expect drop from 12 to 0 in model_X after extraction); diff the five `run_decoder` bodies before/after; confirm `cargo build` across all EP feature flags and golden-transcript equality on `./nemotron` and `./nemotron_multi`.

---

### Q2 — HIGH — Audio front-end duplicated 3-4 times with divergent numerics and 5 copies of the same constants

**Root cause:** `audio.rs` provides a canonical `stft` / `create_mel_filterbank` / `extract_features_with_cache`, but the streaming wrappers each re-implement their own mel pipeline instead of reusing it, and a sixth STFT lives in `sortformer.rs`.

**Evidence:**
- `compute_mel_spectrogram` is hand-rolled in `nemotron.rs:775-785` and `multitalker.rs:666-676` with **different log-guard math**: `(x + LOG_ZERO_GUARD).ln()` (nemotron) vs `(x.max(0.0) + LOG_ZERO_GUARD).ln()` (multitalker). Same function name, same intent, silently different output — exactly the drift duplication causes.
- `parakeet_eou.rs:268` defines a *third* mel filterbank `create_mel_filterbank_htk()` (HTK scale) separate from `audio.rs::create_mel_filterbank` (Slaney). Legitimately different scale, but the surrounding STFT/preemphasis plumbing (`parakeet_eou.rs:259`) re-duplicates `audio.rs`.
- `sortformer.rs:1129-1177` contains a **fourth full STFT implementation** (49 lines) that is functionally the same as `audio.rs::stft_with_plan:87-125` but with centered-window padding inlined; it does not reuse `audio.rs` at all.
- The constant block `SAMPLE_RATE / N_FFT / WIN_LENGTH / HOP_LENGTH / N_MELS / PREEMPH / LOG_ZERO_GUARD` is copy-pasted **identically** into `nemotron.rs:15-21`, `multitalker.rs:26-32`, `sortformer.rs:31-37`, `parakeet_eou.rs:9-16`, and partially `parakeet_unified.rs:14-19` (note: `parakeet_unified.rs:19` spells it `PREEMPHASIS` while the others use `PREEMPH` — naming drift). `LOG_ZERO_GUARD = 5.960_464_5e-8` (= 2^-24) appears 4 times as a literal; `audio.rs:240` recomputes the same value as `2.0f32.powi(-24)` — a fifth, differently-spelled copy.

**Impact:** numeric correctness depends on which copy you read; the mel front-end (the most accuracy-sensitive code in an ASR crate) has no single source of truth. A1/A2 should treat the divergent log-guard (`max(0.0)` vs not) as a possible accuracy bug, not just a style nit.

**Regression risk of the fix:** MEDIUM — consolidating mel paths changes floating-point output unless done bit-exactly; gate with golden-mel fixtures (A6).

**Recommendation:** promote one `MelFrontend` (filterbank scale + preemphasis + log-guard as fields) into `audio.rs`; have all wrappers construct it from a shared const table. Resolve the `max(0.0)` divergence against the upstream NeMo preprocessor (cross-ref A1/RU). Move the 5 constant blocks to one `pub(crate) mod constants`.

**Reverification:** `grep -rn "const N_FFT" src/` should collapse to 1; assert mel output equality vs a recorded fixture before/after.

---

### Q3 — HIGH — `sortformer.rs` (1254) and `nemotron.rs` (786) exceed the user's own 800-line ceiling; god-functions exceed the 50-line rule

**Root cause:** each model wrapper bundles config, vocab, ONNX session, mel front-end, streaming state machine, decode loop, clustering, and tests in one file (mixed concerns).

**Evidence:**
- `sortformer.rs` is 1254 lines (`wc -l`) — 57% over the 800 ceiling stated in `~/.claude/rules/common/coding-style.md`. It contains its own STFT (Q2), mel extraction, the diarization session, clustering helpers (`sortformer.rs:865,899` sort-by-score), cache concat helpers (`sortformer.rs:986,997,1010`), and inline tests — at least 6 distinct concerns.
- `nemotron.rs` 786, `multitalker.rs` 771, `parakeet_unified.rs` 590, `cohere.rs` 464 are all single-file model+vocab+frontend+streaming bundles.
- Functions over 50 lines (sampled): `model_cohere.rs::run_decoder_step` is **119 lines** (`:136-254`) — 32 of them are the manually-unrolled `dk0..ev7` `TensorRef` declarations (`:166-197`) and a 32-input `ort::inputs!` literal; `model_tdt.rs::greedy_decode` is 116 lines (`:176-291`, mixes decode + state update + duration skip); `nemotron.rs` and `multitalker.rs` each have `transcribe_chunk` / `transcribe_samples` bodies well over 50 lines.

**Impact:** hard to review, hard to test in isolation, and directly contradicts the user's stated file-organization rule (cited in CLAUDE.md). Cohere's 8-layer KV cache hand-unroll (`model_cohere.rs:166-237`) is the clearest god-function: it cannot scale to a model with a different layer count without editing the literal.

**Regression risk of the fix:** LOW for file splitting (move-only), MEDIUM for de-unrolling Cohere's KV loop (must preserve exact input ordering).

**Recommendation:** split `sortformer.rs` into `sortformer/{frontend, cluster, session, mod}.rs`; split each large wrapper into `{config, vocab, streaming}` once Q1's shared module exists. Replace Cohere's `dk0..ev7` unroll with a loop over `NUM_DECODER_LAYERS` building a `Vec<(Cow<str>, SessionInputValue)>` (the names are already `format!("past_key_values.{i}...")`-shaped in `read_past_kv:294-305`, so the inverse loop is mechanical).

**Reverification:** `wc -l src/**.rs` all < 800; no public fn body > ~60 lines (clippy `too_many_lines` if enabled).

---

### Q4 — MEDIUM — Error handling is structurally sound but stringly-typed everywhere; source errors are flattened to `String`

**Root cause:** the `Error` enum (`error.rs:5-13`) has only `Io`, `Ort`, plus four `String` variants (`Audio/Model/Tokenizer/Config`). Nearly every fallible ONNX/ndarray call is mapped with `.map_err(|e| Error::Model(format!("...: {e}")))`, discarding the typed source.

**Evidence:**
- `.map_err(|e| Error::Model(format!(...)))` appears dozens of times — e.g. 12+ in `model_nemotron.rs` alone (`:169,177,182,187,...`), and the identical pattern in every `model_X.rs`. The original `ndarray::ShapeError` / extraction error type is lost; only its `Display` survives.
- `Error` does not implement `std::error::Error::source()` (it returns the default `None`), so downstream `eyre`/`anyhow` users get no error chain — `error.rs:28` is a bare `impl std::error::Error for Error {}`.
- `serde_json::Error` is folded into `Error::Config` via `.to_string()` (`error.rs:45-48`), again dropping the typed cause.
- Stringly-typed messages mean callers cannot match on failure modes (e.g. "missing file" vs "shape mismatch" are both `Error::Model`/`Error::Config(String)`).

**Impact:** medium. Nothing is swallowed (good — no silent `let _ =`), but the API is hard to program against and source chains are gone. For a published library this is an ergonomics/observability gap (cross-ref A3).

**Regression risk of the fix:** LOW-MEDIUM — adding `#[from]`/`source()` is additive; restructuring variants is a breaking change but acceptable pre-1.0 if called out.

**Recommendation:** adopt `thiserror` (or hand-write `source()`), add structured variants (`MissingFile { path }`, `ShapeMismatch { expected, got }`, `TensorExtract { name, source }`), and wrap rather than stringify. Combined with Q1, the repeated `.map_err(Error::Model)` boilerplate is deleted by the shared `extract_array*` helpers.

**Reverification:** `Error::source()` returns `Some` for wrapped ort/ndarray errors; `grep -c "Error::Model(format!" src/` drops sharply after Q1.

---

### Q5 — MEDIUM — Per-step allocation/`clone()` in the decode hot path; manual argmax reimplemented 4 ways

**Root cause:** the RNNT decode loop runs once per encoder frame x up to `max_symbols_per_step`, and each step clones every input tensor and rebuilds state arrays.

**Evidence:**
- `model_nemotron.rs:244-248` / `model_unified.rs:145-149` / `model_eou.rs:160-164`: each decoder step does `encoder_frame.clone()`, `state_1.clone()`, `state_2.clone()` into `Value::from_array`. `model_cohere.rs:160-164` already shows the better pattern (`TensorRef::from_array_view`) — the RNNT models do not use it, so they copy hidden states every step. This is also an A2 perf item; flagged here because it is a consequence of the duplicated `run_decoder` (Q1) — fixing it once in the shared helper fixes all five.
- Argmax over logits is reimplemented at least 4 ways: a hand `for` loop (`nemotron.rs:749-756`), `iter().enumerate().max_by(partial_cmp.unwrap_or(Equal))` (`parakeet_unified.rs:477-482`, `model_tdt.rs:231-236`, `decoder.rs:51`), and a standalone `argmax()` in cohere (`cohere.rs:427`). The `for`-loop version in nemotron is the clearest but is the odd one out; the others use the idiomatic iterator form. Consolidate into one `fn argmax(&[f32]) -> usize`.

**Impact:** medium perf (per-frame clones of `[1,1,640]` LSTM states) + minor maintainability (4 argmaxes to keep consistent on NaN handling — note `decoder.rs:51` etc. swallow NaN as `Equal`, which is acceptable but should be one decision in one place).

**Regression risk of the fix:** LOW.

**Recommendation:** in the Q1 shared decoder helper, pass states by `TensorRef` view (matching Cohere) and provide a single `argmax`. Hand the alloc detail to A2.

**Reverification:** A2 RTF benchmark before/after; `grep -c "max_by(|(_, a)" src/` collapses.

---

### Q6 — LOW — Dead/stub code, a misleading TODO, and inconsistent naming

**Evidence:**
- **Beam-search stub:** `decoder.rs:199-206` `decode_with_beam_search` ignores `_beam_width` and just calls `self.decode(logits)`; comment says "Full beam search ... is TODO." This is public-ish dead behavior — a caller passing `beam_width=10` silently gets greedy. Either implement, mark `#[deprecated]`, or remove (cross-ref A3 — it is on the public surface).
- **Commented-out "DON'T reset" code:** `parakeet_eou.rs:250,254` keep commented-out `self.encoder_cache = EncoderCache::new();` / `self.audio_buffer.clear();` with `// DON'T reset!!!`. The intent (streaming state must persist across `reset`) belongs in a doc comment, not commented code; mirrors the Nemotron `reset()`-preserves-state design (A1's language-lock context).
- **Naming drift:** `PREEMPH` (4 files) vs `PREEMPHASIS` (`parakeet_unified.rs:19`); cache structs `NemotronEncoderCache` vs `EncoderCache` vs `MultitalkerEncoderCache` for the same concept (Q1); `state_1/state_2` vs `state_h/state_c` vs `output_states_1/states_1` for the same LSTM states across files.
- **`expect` in a constructor:** `parakeet_unified.rs:222` `.expect("default UnifiedStreamingConfig is always valid")` — acceptable (invariant on a default), but it is the one `expect` in the crate and would be cleaner as a `debug_assert` or documented unwrap. The test-only `.unwrap()`s (`audio.rs:288,325`, `parakeet_unified.rs:583`, `sortformer.rs:1216,1248`, `decoder_tdt.rs:*`) are fine.
- **Good news to record:** locks correctly use `.lock().map_err(...)` (8 sites: `nemotron.rs:584,684,730`, `parakeet_eou.rs:172,191`, `parakeet_unified.rs:344,456,543`), **never** `.lock().unwrap()` — no poisoning-panic on the hot path. No `panic!`/`todo!`/`unimplemented!` anywhere (`grep` count 0). No `#[allow(dead_code)]`. This is a genuinely disciplined baseline; the issues above are about structure, not safety landmines.

**Impact / risk:** low. Mostly cleanup + one misleading public stub.

**Recommendation:** delete or `#[deprecated]` the beam-search stub; convert the EOU commented code to a doc comment; standardize `PREEMPH` and the cache/state names as part of Q1/Q2.

**Reverification:** `grep -rn "TODO\|FIXME"` -> 0 (currently 1); no commented-out statements remain.

---

## Cross-references for other lanes

- **-> 02 (architecture) and A3 (API):** Q1 is the load-bearing input. Propose a shared internal `src/onnx/` module (dual-session loader, `find_first_existing`, parameterized `RnntDecoderStep`, `extract_array{3,4}`/`rebuild_cache`, unified `StreamingCache`). All `model_X` are internal-only (multitalker/cohere `pub(crate)`; nemotron/tdt/eou/unified `pub` but unexported beyond the wrappers) so this is a **non-breaking** refactor. 02 should sequence it as 2-3 ordered one-PR tasks gated by golden tests.
- **-> A3 (API):** Q4 (stringly-typed `Error`, no `source()`) and Q6 (`decode_with_beam_search` is a public no-op honoring no `beam_width`) are public-surface ergonomics items. Pre-1.0, an `Error` restructure (thiserror + structured variants) is fair game but must be flagged as breaking.
- **-> A1 (correctness) and RU (upstream):** Q2 surfaces a real numeric divergence — `nemotron.rs:784` `(x+guard).ln()` vs `multitalker.rs:675` `(x.max(0.0)+guard).ln()` for the "same" mel log-guard. Resolve which matches the NeMo preprocessor; this is potentially an accuracy bug, not just style. Also note Q6: EOU's "DON'T reset" commented code and Nemotron's reset()-preserves-state are the same design choice that underlies the language-lock seed finding.
- **-> A2 (performance):** Q5 — per-step `clone()` of encoder frame + LSTM states in all five RNNT `run_decoder`s, vs Cohere's `TensorRef::from_array_view` (`model_cohere.rs:160-164`). Fixing once in Q1's shared decoder helper removes the per-frame copies for every variant. Also the 4 redundant argmax implementations.
- **-> A6 (tests):** every consolidation above (Q1, Q2, Q5) is unsafe without golden-output regression fixtures (mel features and full transcripts for `./nemotron`, `./nemotron_multi`, and one CTC/TDT model). A6 should make those fixtures the gate that lets the refactor PRs land. Note only `audio.rs`, `sortformer.rs`, `decoder_tdt.rs`, `parakeet_unified.rs`, `cohere.rs`, `timestamps.rs` have inline tests today — the six `model_X` files have **none**, which is precisely why their duplication has been allowed to drift undetected.
