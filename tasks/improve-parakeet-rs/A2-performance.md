# A2 — AUDIT: Performance (hot paths, allocations, threading, EP/GPU)

**/think frameworks:** theory-of-constraints (find the single binding constraint on RTF in the streaming
hot loop, then the next) + first-principles (count allocations/copies per chunk from the code, not from
intuition). PLAN ONLY — no code.

Scope audited: `src/audio.rs`, `src/nemotron.rs`, `src/model_nemotron.rs`, `src/decoder.rs`,
`src/decoder_tdt.rs`, `src/execution.rs`, `src/timestamps.rs`, with cross-checks into `src/model_tdt.rs`,
`src/model_eou.rs`, `src/model_cohere.rs`, `src/multitalker.rs`, `src/parakeet*.rs`.

---

## Constraint analysis (where the time actually goes)

Per the orchestrator note, `transcribe_chunk` is `&mut self`, blocking, and holds `Arc<Mutex<NemotronModel>>`.
Walking one Nemotron streaming chunk (560 ms audio, `CHUNK_SIZE=56` mel frames) from first principles, the
per-chunk cost ladder is:

1. **ONNX `encoder.run`** (one call) — the dominant compute, unavoidable, but its *inputs* are copied in.
2. **ONNX `decoder_joint.run`** — called in a greedy loop, up to `enc_frames * max_symbols_per_step`
   (`56 * 10 = 560` worst case) times per chunk (`nemotron.rs:734-766`). Each call copies 5 tensors in and 3 out.
3. **Mel recompute over the ENTIRE growing audio buffer every chunk** (`nemotron.rs:628`), with the FFT
   **planner rebuilt from scratch** inside `stft` (`audio.rs:80-82`).
4. Per-chunk `Vec`/`Array` allocations and `.clone()` of cache/state for the `ort` boundary.

The binding constraint for *throughput* is (1)+(2) ONNX compute, which is hard to cut without graph/EP
changes. But two library-controlled defects inflate wall-clock well beyond the model's floor: **F1**
(O(n^2) mel recompute) makes long streams degrade super-linearly, and **F2** (per-chunk planner rebuild)
adds fixed overhead every chunk. Those are the highest-leverage fixes because they are pure library
waste, not model cost. EP defaults (**F6**) are the largest *available* win (5-10x) but require user opt-in
and hardware.

---

## Findings (ranked by leverage = impact / effort, highest first)

### F1 — HIGH — Streaming mel recomputed over the entire growing buffer every chunk (O(n^2) work)
- **Root cause:** `Nemotron::transcribe_chunk` appends raw audio to `self.audio_buffer` then calls
  `self.compute_mel_spectrogram(&self.audio_buffer)` over the *whole* buffer on every chunk
  (`nemotron.rs:617-628`). Only `CHUNK_SIZE` *new* mel frames are consumed (`nemotron.rs:633-639`); the
  rest is thrown away and recomputed next call. The buffer is only trimmed once it exceeds
  `keep_samples * 2` (`nemotron.rs:705-712`), so within that window every STFT frame of the retained
  history is recomputed each chunk.
- **Evidence:** `nemotron.rs:617,628,633-639,705-712`. Same pattern in `multitalker.rs:326,334,439-440`.
- **Impact:** STFT + mel matmul cost per chunk grows with buffer length instead of staying constant.
  With the `keep_samples*2` cap the buffer holds ~`(9+56)*160+400 ≈ 10,800` samples ≈ 67 mel frames,
  so each chunk recomputes ~67 frames of STFT to use 56 — roughly **1.2x redundant STFT work steady-state**,
  and a full O(n^2) blowup for any caller who lets the buffer grow (e.g. larger trim threshold or a model
  with a longer left context). The mel matmul `mel_basis.dot(&spec)` (`nemotron.rs:782`) is
  `128 x 257 · 257 x frames`; recomputing it over the full window is pure waste.
- **Regression risk of fix:** MEDIUM. The buffer-and-recompute design exists to "avoid edge effects at
  chunk boundaries" (`nemotron.rs:613-614`). A switch to incremental STFT (only new hop frames + a small
  carried tail of `win_length - hop_length` samples) must reproduce the exact frame alignment, or it
  changes transcripts. Needs a golden-transcript regression test (coordinate with A6).
- **Recommendation:** Compute STFT only over the newly-arrived hop frames plus the minimal carry tail;
  keep a rolling mel ring buffer instead of recomputing. Alternatively, cap recompute to exactly the
  `PRE_ENCODE_CACHE + CHUNK_SIZE` window actually consumed. Estimate: removes the redundant STFT and the
  full-window mel matmul each chunk; steady-state mel cost drops to ~1 chunk's worth (the
  `keep_samples*2` trim already bounds it, so the realized win on the default path is modest — the larger
  win is eliminating the O(n^2) cliff for non-default usage).
- **Reverification:** Benchmark `transcribe_chunk` latency vs buffer length before/after; assert byte-equal
  transcript on a fixed WAV (`./nemotron`, `./nemotron_multi`).

### F2 — HIGH — FFT planner rebuilt on every `stft` call in Nemotron / EOU / Multitalker
- **Root cause:** `audio::stft` constructs a fresh `RealFftPlanner` and `plan_fft_forward(n_fft)` on every
  invocation (`audio.rs:80-82`). `FeatureCache` exists precisely to cache this plan (`audio.rs:10-28`,
  `extract_features_with_cache` at `audio.rs:230`), and CTC/TDT/Unified/Cohere use it — but Nemotron
  (`nemotron.rs:781`), Multitalker (`multitalker.rs:672`), and EOU (`parakeet_eou.rs:259`) bypass it and
  call the plan-rebuilding `stft`. EOU rebuilds the planner on *every* `transcribe_chunk`
  (`parakeet_eou.rs:139-153,259`).
- **Evidence:** `audio.rs:80-82` (rebuild); `nemotron.rs:781`, `multitalker.rs:672`, `parakeet_eou.rs:259`
  (callers that rebuild). Contrast `parakeet.rs:79,138`, `parakeet_tdt.rs:72,100` which build a
  `FeatureCache` once at load and reuse it.
- **Impact:** RustFFT plan construction is non-trivial (factorization + twiddle-factor precompute) and
  happens once per chunk in the three streaming-heaviest variants. The doc comment in
  `extract_features_with_cache` cites "~15-20 µs of arithmetic per request" for the *mel filterbank*
  alone (`audio.rs:193`); the FFT planner is additional and the `mel_basis` is *also* rebuilt at load in
  these three (each stores its own `mel_basis`, fine) but the *plan* is rebuilt per chunk. For a 560 ms
  chunk this is small in absolute terms but it is pure, repeated, avoidable allocation + setup every chunk
  on the real-time path.
- **Regression risk of fix:** LOW. The plan is deterministic from `n_fft`; `stft_with_plan` already exists
  and is numerically identical. Pass the cached `FeatureCache.fft_plan` (or store an `Arc<dyn RealToComplex>`
  on the struct) and call `stft_with_plan`.
- **Recommendation:** Give Nemotron/Multitalker/EOU a `FeatureCache` (or at least a cached `fft_plan`) at
  load time and route their mel path through `stft_with_plan`. Nemotron already stores `mel_basis:
  Arc<Array2<f32>>` (`nemotron.rs:276`) — add the plan beside it. Estimate: eliminates 1 planner
  construction per chunk on three variants.
- **Reverification:** Identical transcript on a fixed WAV; confirm planner constructor no longer appears in
  a per-chunk profile/flamegraph.

### F3 — HIGH — Every `ort` tensor input is `.clone()`d (owning copy) instead of borrowed; Cohere already does it right
- **Root cause:** `model_nemotron.rs`, `model_tdt.rs`, `model_eou.rs`, `model.rs` all build inputs with
  `ort::value::Value::from_array(x.clone())`, which copies the ndarray into an owned ORT tensor. The
  decoder loop is the worst offender: each `run_decoder` clones `state_1` and `state_2`
  (`model_nemotron.rs:247-248`) and the encoder frame, every step, up to ~560 times/chunk. By contrast
  `model_cohere.rs` uses `ort::value::TensorRef::from_array_view(x.view())` throughout
  (`model_cohere.rs:111,160-190+`), which borrows without copying.
- **Evidence:** clones — `model_nemotron.rs:150-154,244-248`; `model_tdt.rs:139-140,215-219`;
  `model_eou.rs:80-84,160-164`; `model.rs:46-47`. Zero-copy reference impl already present —
  `model_cohere.rs:111,160-194`.
- **Impact:** Per Nemotron chunk, the encoder run clones `processed_signal` (`1x128x65` ≈ 33 KB),
  `cache_last_channel` (`24x1x70x1024` ≈ 6.9 MB), `cache_last_time` (`24x1x1024x8` ≈ 0.8 MB) plus small
  arrays — ~7.7 MB copied **per chunk** just for the encoder cache, and the decoder loop clones the two
  LSTM states (`2x1x640` ≈ 5 KB each) up to 560 times = up to ~5.7 MB copied per chunk in the decode loop.
  These are memcpys on the real-time path that the Cohere pattern shows are avoidable.
- **Regression risk of fix:** MEDIUM. `TensorRef::from_array_view` borrows, so the backing ndarray must
  outlive the `run` call and not be mutated during it. In the decoder loop the state arrays are reassigned
  from outputs each step (`nemotron.rs:764-765`), so lifetimes need care (run, extract outputs, then swap).
  Mechanical but must preserve correctness; the Cohere code is the template.
- **Recommendation:** Migrate the four `model_*.rs` files to `TensorRef::from_array_view` for inputs that
  are not consumed by ORT, matching `model_cohere.rs`. Biggest single win: the encoder `cache_last_channel`
  clone (~6.9 MB/chunk) and the per-step LSTM-state clones in the decode loop.
- **Reverification:** Byte-equal transcript on fixed WAVs across all four variants; confirm reduced
  allocations via a chunk-level alloc counter or profiler.

### F4 — MEDIUM — Output tensors copied via `.to_vec()` + `from_shape_vec` on every ORT run
- **Root cause:** Every encoder/decoder output is materialized with `data.to_vec()` then
  `Array{1,3,4}::from_shape_vec(...)`, which allocates a fresh `Vec` and copies the extracted tensor data.
  In the decode loop `run_decoder` does this for `logits`, `output_states_1`, `output_states_2` every step
  (`model_nemotron.rs:256,266-284`). The encoder does it for `encoded` (~`1x1024x13` per output frame
  group) plus three cache tensors every chunk (`model_nemotron.rs:176,198-224`).
- **Evidence:** `model_nemotron.rs:176,205,217,222,256,272,283`; identical pattern `model_tdt.rs:169,258,268`,
  `model_eou.rs`.
- **Impact:** The cache outputs (`cache_last_channel_next` ~6.9 MB, `cache_last_time_next` ~0.8 MB) are
  `to_vec()`'d every chunk — another ~7.7 MB copied per chunk on output, mirroring F3's input copies. The
  decode-loop `logits.to_vec()` (vocab_size up to ~13k floats ≈ 52 KB) plus two state copies happen up to
  560x/chunk.
- **Regression risk of fix:** MEDIUM. `try_extract_tensor` returns a borrowed slice tied to the `outputs`
  value; to avoid the copy the consumer must use the data while `outputs` is alive, or accept an
  `ArrayView`/owned-from-raw without re-copy. Restructuring greedy argmax (F5) to read the borrowed logits
  slice directly removes the `logits` copy with no correctness change.
- **Recommendation:** (a) For logits in the decode loop: run argmax directly on the borrowed
  `l_data` slice instead of `Array1::from_vec(l_data.to_vec())` (`model_nemotron.rs:256` feeds
  `nemotron.rs:751-756`). (b) For the encoder cache outputs: reuse preallocated `Array4` buffers in
  `NemotronEncoderCache` (copy into existing storage rather than allocating a new `Vec` + array each chunk).
- **Reverification:** Transcript equality; alloc/flamegraph comparison.

### F5 — MEDIUM — Greedy decode loop allocates a fresh owned frame array per encoder frame
- **Root cause:** In `decode_chunk`, for each of `enc_frames` frames the code does
  `encoder_out.slice(...).to_owned()` then `.to_shape((1, hidden_dim, 1)).to_owned()` — two allocations per
  frame to reshape a contiguous column (`nemotron.rs:735-739`). Then `logits` is rebuilt as an owned
  `Array1` for the argmax (F4a). The argmax itself is a hand-rolled loop (fine), but it iterates the full
  vocab (~13k for multilingual) each of up to 560 calls.
- **Evidence:** `nemotron.rs:734-739,749-756`. TDT equivalent: `model_tdt.rs:210` builds `targets` and clones
  states per step in the `while t < time_steps` loop (`model_tdt.rs:201-268`).
- **Impact:** 2 array allocations per encoder frame (up to 56/chunk) plus the logits copy per decoder step
  (up to 560/chunk). Small individually, but on the per-chunk hot path and trivially reusable.
- **Regression risk of fix:** LOW. The frame reshape produces a `[1, hidden_dim, 1]` view of a contiguous
  column; a preallocated scratch `Array3` filled in place (or `TensorRef` over a reshaped view) is
  equivalent. Argmax on the borrowed slice is identical math.
- **Recommendation:** Preallocate one `[1, hidden_dim, 1]` scratch buffer per `Nemotron` instance, fill it
  per frame; argmax over the borrowed logits slice (combines with F3/F4a).
- **Reverification:** Transcript equality on fixed WAVs.

### F6 — HIGH (impact) / opt-in — GPU/CoreML execution providers are compile-time-gated and never auto-discovered; default build is CPU-only
- **Root cause:** `ExecutionProvider` defaults to `Cpu` (`execution.rs:16-18`) and every accelerator variant
  is behind a `#[cfg(feature = ...)]` (`execution.rs:20-36`). With no GPU feature enabled the enum only has
  `Cpu`, so a default-built binary *cannot* select GPU even at runtime. There is no auto-detect path that
  probes available EPs and picks the fastest; the user must both enable the cargo feature *and* set
  `ExecutionProvider::Cuda`/`CoreML`/etc. explicitly.
- **Evidence:** `execution.rs:16-36` (cfg-gated enum), `execution.rs:69-79` (default Cpu, intra=4 inter=1),
  `execution.rs:138-197` (match arms, all cfg-gated). Doc comment acknowledges GPU gives "5-10x speedup"
  (`execution.rs:8`).
- **Impact:** The single largest available perf lever (5-10x per the crate's own comment) is invisible to
  anyone who doesn't read the Cargo features. This is a discoverability/ergonomics-as-performance issue
  rather than a code defect.
- **Regression risk of fix:** LOW-MEDIUM. Auto-selection must respect the CoreML caveat already documented
  (`execution.rs:11-14`: CoreML runs *slower* than CPU for these dynamic-shape graphs), so "auto" cannot
  blindly prefer CoreML on macOS. Any auto-detect belongs behind explicit opt-in to avoid surprising a
  CPU-tuned deployment.
- **Recommendation:** (a) Document the EP/feature matrix prominently with the 5-10x note and the CoreML
  caveat (coordinate A3-api docs). (b) Optionally add an opt-in `ExecutionProvider::Auto` that probes and
  prefers CUDA/TensorRT/DirectML when their feature is compiled, but keeps CPU on macOS unless the graphs
  are made static-shape. Tie to F8 (cold-start) and the EP cross-ref.
- **Reverification:** N/A perf-correctness; benchmark each EP on `./nemotron` and record RTF in docs.

### F7 — MEDIUM — `ort` session options leave memory pattern / arena / parallel-execution at defaults; only opt level + thread counts are set
- **Root cause:** `apply_to_session_builder` sets `GraphOptimizationLevel::Level3`, `intra_threads`,
  `inter_threads` and nothing else (`execution.rs:133-136`). Memory pattern, memory arena, CPU-arena
  settings, and execution mode (sequential vs parallel) are left at ORT defaults. For a streaming model
  with *fixed* input shapes per chunk, enabling memory pattern can materially reduce per-run allocation;
  for the dynamic-shape graphs it may not help (and ORT disables it for dynamic shapes anyway). `inter_threads=1`
  with the default sequential execution mode means `inter_threads` is effectively inert.
- **Evidence:** `execution.rs:133-136`; default `intra=4, inter=1` (`execution.rs:72-78`); the
  `with_custom_configure` escape hatch exists (`execution.rs:101-107`) but is undocumented for perf tuning.
- **Impact:** Leaves a known ORT tuning lever (memory pattern for fixed-shape runs) on the table. Magnitude
  is graph-dependent and unmeasured here — label as a tuning opportunity, not a proven win. The bigger,
  certain point: `inter_threads=1` does nothing unless execution mode is parallel and the graph has parallel
  branches, so the knob may mislead users.
- **Regression risk of fix:** LOW for docs; MEDIUM if defaults change (thread counts interact with the host;
  4 intra-threads can oversubscribe on small cores or under multiple concurrent streams via `NemotronHandle`).
- **Recommendation:** (a) Document `with_custom_configure` as the perf-tuning entry point with examples
  (mem pattern, arena, exec mode). (b) Consider exposing `with_memory_pattern(bool)` and an exec-mode toggle.
  (c) Re-examine the `intra=4/inter=1` default and document that under N concurrent shared-model streams the
  effective thread demand is `N * intra`. Coordinate with A3-api.
- **Reverification:** Benchmark per-EP with/without memory pattern on fixed-shape chunks; record.

### F8 — MEDIUM — Cold-start: two full session loads per model, no graph caching except CoreML
- **Root cause:** `NemotronModel::from_pretrained` builds two separate `SessionBuilder`s and commits the
  encoder and decoder ONNX files (`model_nemotron.rs:88-94`) — each applies Level3 graph optimization at
  load. There is no on-disk optimized-graph cache (`SessionBuilder::with_optimized_model_path`) for the CPU
  path; only CoreML has a `coreml_cache_dir` to avoid ~5 s recompilation (`execution.rs:47,109-114,158-160`).
- **Evidence:** `model_nemotron.rs:88-94`; `execution.rs:47,158-160`. The 2.3-2.4 GB models
  (context-pack) make load non-trivial.
- **Impact:** Every process start re-runs Level3 optimization on large graphs. For a long-lived service
  this is amortized; for short-lived CLI invocations or frequent reloads it is repeated cost.
- **Regression risk of fix:** LOW. `with_optimized_model_path` is additive and optional.
- **Recommendation:** Expose an optional optimized-graph cache dir (analogous to `coreml_cache_dir`) wired
  to `SessionBuilder::with_optimized_model_path`, so repeat loads skip re-optimization. Document the
  `NemotronHandle::load` + `from_shared` path (`nemotron.rs:337,418`) as the way to amortize load across
  concurrent streams (it already shares the `Arc<Mutex<NemotronModel>>`).
- **Reverification:** Measure second-load time with vs without the cache dir.

### F9 — MEDIUM — `lang_tag_ids.contains(t)` is a linear scan inside per-token hot loops
- **Root cause:** Language-tag filtering calls `self.lang_tag_ids.contains(&t)` — a `Vec` linear scan — for
  every accumulated token in `get_transcript` (`nemotron.rs:518`), every token in `transcribe_audio`'s
  final filter (`nemotron.rs:605`), and every token emitted in the streaming `transcribe_chunk` loop
  (`nemotron.rs:716`). `lang_tag_ids` is a `Vec<usize>` (`nemotron.rs:287`) holding all `<xx-XX>` ids
  (dozens of entries for the multilingual vocab).
- **Evidence:** `nemotron.rs:287,518,605,716`; populated from `vocab.lang_tag_ids()` (`nemotron.rs:256-262`).
- **Impact:** O(tokens * lang_tags) per chunk on the multilingual path. Small (dozens x dozens) but trivially
  removable and on the per-emit path.
- **Regression risk of fix:** LOW. Swap `Vec` for a `HashSet<usize>` (or a `bool` lookup table sized to
  vocab) at load; membership semantics identical.
- **Recommendation:** Store `lang_tag_ids` as `Arc<HashSet<usize>>` or a `Vec<bool>` bitset. English-only
  path is unaffected (empty set).
- **Reverification:** Transcript equality on `./nemotron_multi`.

### F10 — LOW — Quadratic progressive re-decode for CTC timestamps
- **Root cause:** `decode_with_timestamps` re-decodes the entire token prefix `[0..=i]` through the
  tokenizer for *each* token to detect what text it added (`decoder.rs:163-191`). That is O(n^2) tokenizer
  decodes for n collapsed tokens.
- **Evidence:** `decoder.rs:163-170`.
- **Impact:** Only on the CTC timestamp path (offline, not the Nemotron streaming hot loop), so low
  priority, but genuinely quadratic for long utterances.
- **Regression risk of fix:** MEDIUM. The progressive-decode trick handles BPE spacing; a streaming/
  incremental tokenizer decode must reproduce the same word-boundary behavior. Needs the existing
  timestamp tests as guard (`timestamps.rs` tests, `decoder_tdt.rs` tests).
- **Recommendation:** Decode once and map token spans to character offsets, or use the tokenizer's offset
  mapping if available, instead of re-decoding every prefix. Defer behind the streaming fixes.
- **Reverification:** Existing timestamp tests pass; add a long-utterance timing benchmark.

### F11 — LOW — `apply_preemphasis` and `hann_window` allocate fresh Vecs every call
- **Root cause:** `apply_preemphasis` allocates a new `Vec` of the full audio length each call
  (`audio.rs:49-62`); `hann_window` rebuilds the window inside every `stft_with_plan` call
  (`audio.rs:64-68,99`). For streaming Nemotron these run once per chunk over the (currently full, see F1)
  buffer.
- **Evidence:** `audio.rs:49-62,99`; called from `nemotron.rs:780-781`.
- **Impact:** One audio-length `Vec` alloc + one `win_length`-sized window alloc per chunk. Minor; the
  window is constant and could be cached alongside the FFT plan (F2).
- **Regression risk of fix:** LOW. Window is deterministic from `win_length`; cache it. Preemphasis could
  write in place if the incremental-STFT refactor (F1) owns its buffer.
- **Recommendation:** Cache the Hann window in `FeatureCache` (next to `fft_plan`/`mel_basis`); fold the
  preemphasis allocation into the F1 incremental refactor.
- **Reverification:** Transcript equality.

### F12 — LOW (note) — No async / off-runtime guidance; `transcribe_chunk` is blocking `&mut self`
- **Root cause:** `transcribe_chunk` is synchronous and CPU-bound, and holds the model `Mutex`
  (`nemotron.rs:584-593,683-693,730-732`). There is no documented pattern for running it off an async
  runtime (e.g. `spawn_blocking`) or for driving multiple streams in parallel. The `NemotronHandle` +
  `from_shared` design (`nemotron.rs:326-453`) correctly enables N independent decoder states over one
  shared session, but all N serialize on the single `Arc<Mutex<NemotronModel>>` during inference.
- **Evidence:** No `tokio`/`spawn_blocking`/`rayon` anywhere in `src/` or `examples/` (grep, this session);
  `nemotron.rs:301,432` (`Arc<Mutex<NemotronModel>>` shared); `shared_model.rs` example exists per context
  pack.
- **Impact:** Two distinct issues: (a) callers on an async runtime will block the executor — needs docs.
  (b) The shared-model story serializes inference across streams: two concurrent streams cannot run the
  encoder in parallel even on a multi-core CPU because they contend on one `Mutex`. ORT `Session::run` is
  itself thread-safe (takes `&self`), so the `Mutex` is stricter than ORT requires. `run_encoder`/
  `run_decoder` take `&mut self` on `NemotronModel` only to call `self.encoder.run` — they don't mutate the
  session — so the `&mut`/`Mutex` is conservative.
- **Regression risk of fix:** MEDIUM. Confirm against the `ort` version in `Cargo.toml` (2.0.0-rc.12) that
  `Session::run(&self, ...)` is the signature and is `Send + Sync`; if so, the `Mutex` can become a shared
  `&Session` (or `RwLock`) allowing concurrent runs. The encoder/decoder cache and LSTM state are already
  per-`Nemotron`-instance (not in `NemotronModel`), so removing the lock does not introduce shared mutable
  state — but this must be verified, not assumed.
- **Recommendation:** (a) Document `spawn_blocking` for async callers and the per-chunk blocking cost
  (~20-50 ms cited at `nemotron.rs:411`). (b) Investigate replacing `Arc<Mutex<NemotronModel>>` with
  `Arc<NemotronModel>` + `&self` runs so concurrent streams parallelize (requires changing `run_encoder`/
  `run_decoder` to `&self` and confirming ORT `Send+Sync`). This is the real concurrency win for the
  shared-model use case. Coordinate with A1/A5.
- **Reverification:** Concurrency test: two `Nemotron` instances from one handle transcribing in parallel,
  assert independent correct transcripts and measure wall-clock speedup vs serialized.

---

## Summary table

| ID | Sev | Hot path | One-line fix | Est. win |
|----|-----|----------|--------------|----------|
| F1 | HIGH | nemotron/multitalker stream | incremental STFT, stop full-buffer recompute | removes O(n^2) cliff; ~1.2x steady STFT |
| F2 | HIGH | nemotron/eou/multitalker mel | route through cached `fft_plan` (`stft_with_plan`) | 1 planner build/chunk eliminated x3 variants |
| F3 | HIGH | all model_*.rs ORT inputs | `TensorRef::from_array_view` (copy Cohere) | ~7.7 MB/chunk + per-step state copies removed |
| F6 | HIGH* | EP selection | doc + opt-in `Auto` EP | 5-10x (hardware, opt-in) |
| F4 | MED | ORT outputs | reuse cache buffers; argmax on borrowed logits | ~7.7 MB/chunk output copy removed |
| F5 | MED | greedy decode loop | preallocate frame scratch | 2 allocs/frame removed |
| F7 | MED | session options | doc/expose mem-pattern, exec-mode | graph-dependent (unproven) |
| F8 | MED | cold start | optimized-graph cache dir | repeat-load opt skipped |
| F9 | MED | token filter loop | `HashSet`/bitset for lang_tag_ids | O(tok*tags) -> O(tok) |
| F12| MED | concurrency | drop Mutex -> `&self` runs; doc spawn_blocking | parallel streams (verify ORT Send+Sync) |
| F10| LOW | CTC timestamps | non-quadratic prefix decode | O(n^2)->O(n) offline path |
| F11| LOW | audio prep | cache Hann window; in-place preemphasis | minor per-chunk allocs |

\*F6 impact is large but gated on hardware + user opt-in.

---

## Cross-references for other lanes

- **A1-correctness:** F1 (incremental STFT) and F3/F4/F5 (tensor reuse) MUST be guarded by golden-transcript
  equality tests on `./nemotron` and `./nemotron_multi` — they are correctness-preserving refactors only if
  the frame alignment and argmax are bit-identical. F12 (dropping the `Mutex`) intersects the
  shared-model/streaming-state correctness story: confirm encoder cache + LSTM state are strictly
  per-instance (they are, `nemotron.rs:312-324`) so concurrent runs cannot corrupt each other.
- **A3-api:** F6 (EP discoverability), F7 (`with_custom_configure` as the documented perf-tuning hook,
  possible `with_memory_pattern`/exec-mode setters), F8 (`with_optimized_model_path` analogous to
  `with_coreml_cache_dir`), and F12 (documenting `spawn_blocking` + the blocking cost, and a possible
  `&self` run API) are all public-surface/ergonomics decisions. The `intra=4/inter=1` default
  (`execution.rs:72-78`) and its interaction with N concurrent shared-model streams needs an API/doc note.
- **A4-models:** F2/F1 touch the per-variant mel paths — Nemotron, Multitalker, EOU each hand-roll mel
  instead of using `FeatureCache`; unifying them (A5/02-architecture) would fix F2 across variants in one
  change. Quantization (model-coverage lane) is the other untapped perf lever not in scope here but
  complementary to F6.
- **A5-quality / 02-architecture:** F3/F4 are the same copy-everything anti-pattern duplicated across four
  `model_*.rs` files while `model_cohere.rs` already demonstrates the zero-copy `TensorRef` pattern — a
  shared encoder/decoder-run abstraction would fix F3/F4/F5 once instead of four times, and route all
  variants through one cached-mel path (F1/F2). This is the strongest argument for the shared-abstraction
  extraction the architecture lane is evaluating.
- **A6-tests:** Every HIGH/MED perf fix needs a regression guard: (a) byte-equal transcript on fixed WAVs
  per variant (guards F1-F5, F9), (b) a parallel-streams correctness + speedup test (guards F12), (c)
  optional criterion benchmarks for `transcribe_chunk` latency vs buffer length to prove F1 and catch
  regressions. Models are available locally for integration fixtures.
