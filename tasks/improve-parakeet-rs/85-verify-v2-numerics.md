# 85 — Adversarial Verification V2 (Wave V): HIGH performance refactors are correctness-preserving

> Verifier V2. Framework: **red-team** (assume the claim is false; hunt for the one input/seam where
> the "equivalent" refactor changes a byte of model output). Default REFUTED if the equivalence is not
> provable from the actual code. PLAN ONLY — no code written.

## Claim under test

> The HIGH performance refactors are correctness-preserving (byte-equal transcript):
> (F1) computing mel only over new frames instead of the whole growing buffer,
> (F2) routing Nemotron/EOU/Multitalker through the cached FFT plan (`stft_with_plan`),
> (F3) replacing owning `Value::from_array(x.clone())` with zero-copy `TensorRef::from_array_view`,
> all produce identical model output.

Verdict is rendered **per sub-claim**. The umbrella claim ("all produce identical output") is only as
strong as its weakest leg.

---

## F2 — route the three hand-rolled mel paths through `stft_with_plan` — **SURVIVES**

**What F2 actually changes.** Nemotron (`nemotron.rs:781`), Multitalker (`multitalker.rs:672`), and EOU
(`parakeet_eou.rs:259`) all call `crate::audio::stft(...)`. The body of `stft` (`audio.rs:74-83`) is:

```
let mut planner = realfft::RealFftPlanner::<f32>::new();
let plan = planner.plan_fft_forward(n_fft);
stft_with_plan(audio, &plan, n_fft, hop_length, win_length)
```

`stft` is a *thin wrapper that already delegates to `stft_with_plan`* after building a planner. F2 only
removes the per-call planner construction and passes a cached plan into the identical
`stft_with_plan` (`audio.rs:87-125`). The window (`hann_window(win_length)`, line 99), the `n_fft/2`
symmetric padding (lines 94-97), the framing (`frame_idx * hop_length`, line 109), and the
`output[k].norm_sqr()` power spectrum (line 120) are all inside `stft_with_plan` — the code path the
inline callers already execute today.

**Red-team attempt — "3 divergent mel flavors" (A5 Q2/Q3).** The prompt warns A1/A5 flagged divergent mel
math. I checked: the divergences are real but live **outside** `stft`/`stft_with_plan`:
- log-guard: `nemotron.rs:784` `(x + guard).ln()` vs `multitalker.rs:675` `(x.max(0.0) + guard).ln()`
  — applied AFTER `mel_basis.dot(spec)`, untouched by F2.
- filterbank: EOU uses `create_mel_filterbank_htk()` (`parakeet_eou.rs:268`), HTK scale, vs Slaney
  `create_mel_filterbank` — that is the `mel_basis`, computed once at load, untouched by F2.
- preemphasis (`apply_preemphasis`) — applied before `stft`, untouched by F2.

F2 swaps only the FFT-plan construction, which is **deterministic from `n_fft`** (realfft factorizes the
same `n_fft=512` to the same plan every time). The plan affects *which algorithm* computes the DFT, not
the mathematical result; `stft` vs `stft_with_plan` produce bit-identical spectrograms for the same
`n_fft`. The three flavors stay exactly as divergent as they are now — F2 neither fixes nor worsens them.

**Verdict: SURVIVES.** F2 is numerically identical by construction (`stft` already *is* `stft_with_plan` +
a thrown-away planner). This is the safest of the three; it does **not** require a golden test to be
*correct*, though one is cheap insurance. **Caveat:** F2 must not be conflated with "unify the mel
front-ends" (A5 Q2). Unifying the log-guard / filterbank WOULD change output and is a separate,
golden-gated change. F2 alone = plan caching only.

---

## F3 — `Value::from_array(x.clone())` -> `TensorRef::from_array_view(x.view())` — **SURVIVES (conditionally; must be gated)**

**Is the view truly equivalent to the owned clone?** I read the ort 2.0.0-rc.12 source
(`~/.cargo/.../ort-2.0.0-rc.12/src/value/impl_tensor/create.rs`). The two APIs are **not unconditionally
equivalent**; they differ exactly on non-contiguous input:

- `Tensor::from_array` (line 134-139, impl at 442-465): for a standard-layout array uses the data
  pointer as-is; **for a non-standard layout it silently copies** via `self.as_standard_layout().into_owned()`
  (line 452-463). It always succeeds.
- `TensorRef::from_array_view` (line 212-214, impl `ArrayView::ref_parts` at 472-479): calls
  `self.as_slice()` and **returns an `Error` if the layout is non-contiguous** ("Array has a non-contiguous
  layout and cannot be used to construct a Tensor", line 477). It cannot copy — it borrows.

**When both succeed they are byte-identical:** both read `shape` row-major (`shape().iter()...`) and the
same flat `&[T]` slice. So for any **C-contiguous** input the resulting tensor data is identical, and ORT
`Session::run` sees identical bytes -> identical output. The equivalence is therefore conditional on
contiguity, not universal.

**Red-team — are the Nemotron inputs contiguous?** Checked the actual call sites that would be migrated:
- `run_encoder` inputs (`model_nemotron.rs:150-154`): `features` is built by `Array3::from_shape_vec`
  (`nemotron.rs:680`, owned/contiguous); the cache arrays are built by `Array4/Array1::from_shape_vec`
  from `to_vec()` output (`model_nemotron.rs:198-224`, contiguous); `length_arr`/`prompt_arr` are
  `Array1::from_vec` (contiguous).
- `run_decoder` inputs (`model_nemotron.rs:244-248`): `encoder_frame` is `frame.to_shape((1,hidden_dim,1)).to_owned()`
  (`nemotron.rs:735-739`) — freshly owned, contiguous; `state_1`/`state_2` are reassigned each step from
  `from_shape_vec` outputs (`nemotron.rs:764-765`), contiguous.

All are standard-layout, so `from_array_view` will succeed and match the clone byte-for-byte. The Cohere
reference (`model_cohere.rs:111,160-197`) is the proven template and uses exactly this pattern over views
of owned `Array`s.

**Red-team — aliasing / mutated-during-run.** `TensorRef` borrows; the backing array must (a) outlive the
`run` call and (b) not be mutated during it. In `run_decoder` the input states are passed by `&`; the new
states are extracted from `outputs` *after* `decoder_joint.run()` returns, and the caller swaps
`self.state_1 = new_state_1` only after `run_decoder` returns (`nemotron.rs:764-765`). No input array is
mutated while borrowed by a live `TensorRef` -> no aliasing hazard, no "view sees mutated data" case.
This holds *if implemented in the run-then-extract-then-swap order the A2 finding already prescribes*; an
implementer who reorders (swaps before the view is dropped) could introduce a borrow-conflict, but that is
a compile error in Rust, not a silent numeric change.

**Verdict: SURVIVES — conditional.** Byte-identical whenever (i) every migrated input is C-contiguous
(verified true for Nemotron today) and (ii) the run-then-swap ordering is preserved. **Hidden scope the
A2 finding under-states:** the contiguity *error* path. If a future refactor ever feeds a sliced/transposed
(non-contiguous) view, `from_array` would silently succeed (copying) while `from_array_view` would *fail
the run*. That is a behavioral divergence (error, not wrong transcript), so F3 carries a latent
correctness/robustness scope beyond "just a copy elision." **MUST be gated by a golden-transcript test
across all four `model_*.rs` variants** (A6) and by a build/run smoke test that would surface any
non-contiguous input as an `Err` rather than a silent copy.

---

## F1 — compute mel only over new frames, not the whole growing buffer — **REFUTED as stated; UNCERTAIN under the right carry design (needs a byte-equal golden experiment)**

This is the leg that breaks the umbrella claim. F1 is **not** a pure perf change; the whole-buffer
recompute is partially load-bearing for seam correctness, and a naive "only new frames" implementation
changes output at three independent seams.

**Seam 1 — preemphasis is a cross-sample filter.** `apply_preemphasis` (`audio.rs:49-62`) emits
`out[0] = audio[0]` verbatim, then `out[i] = audio[i] - 0.97*audio[i-1]`. The whole-buffer path runs this
over the entire retained buffer, so every interior sample carries the `-0.97*prev` term. A literal
"compute mel only over new frames" that re-runs `apply_preemphasis` on the new slice would set
`out[0] = audio[k]` verbatim (no `-0.97*audio[k-1]`), changing every value in the first new frame.
Matching the current output requires carrying `audio[k-1]` and applying the filter *across* the seam.

**Seam 2 — `stft_with_plan` always prepends `n_fft/2 = 256` zeros (`audio.rs:94-97`).** In the
whole-buffer path this front-pad lands once at the current buffer start. Any incremental path that re-calls
`stft_with_plan` on a new segment injects 256 fresh zeros *mid-stream* at every seam -> totally wrong
framing. A correct incremental refactor must NOT re-pad internally and must reproduce the global frame
origin `frame_idx * hop_length` (`audio.rs:108-109`) measured from the current buffer's padded start.

**Seam 3 — frame alignment vs the PRE_ENCODE_CACHE overlap.** The chunk builder slices `full_mel` by
`processed_mel_frames = audio_processed / HOP_LENGTH` and reaches back `PRE_ENCODE_CACHE=9` frames for the
cache overlap (`nemotron.rs:633,647,657-677`). Incremental mel must yield frames at *exactly* the same
global hop indices or the 9-frame overlap and the `CHUNK_SIZE=56` main window shift, changing the encoder
input tensor and therefore the transcript.

**The decisive subtlety (cuts against an "idealized" F1).** The current code is **already not globally
continuous**: the buffer trim (`nemotron.rs:705-712`) drains the front once `len > keep_samples*2`,
decrements `audio_processed`, and the *next* `compute_mel_spectrogram` call re-pads 256 zeros before the
new `audio_buffer[0]` and **restarts preemphasis at that mid-stream sample**. So today's "golden" mel has a
re-pad + preemphasis-restart discontinuity at every trim boundary. Consequences:
- The F1 target is "byte-match the current *seam-bearing* output," not "match an ideal continuous mel." A
  "cleaner" incremental design (continuous preemphasis, single pad) would actually **DIVERGE** from current
  output and silently change transcripts.
- Within the default trim window the buffer holds only ~67 mel frames to consume 56 (A2's own estimate),
  so the realized perf win is modest (~1.2x STFT); the O(n^2) blowup only bites non-default callers. The
  cost/benefit of taking on this seam risk is therefore unfavorable unless paired with a strong golden gate.

**Why REFUTED-as-stated.** The claim says F1 "produces identical model output." Read literally
("compute mel only over new frames"), the preemphasis restart (Seam 1) and the re-pad (Seam 2) guarantee a
*different* mel at every chunk boundary -> different encoder input -> not byte-equal. The current
whole-buffer recompute is load-bearing for at least the preemphasis continuity and the single-pad
invariant within each trim window.

**Why UNCERTAIN (not flatly impossible).** A carefully built incremental refactor *can* in principle
reproduce the current output byte-for-byte if it: carries `win_length - hop_length = 240` samples of STFT
overlap **plus** 1 sample for preemphasis continuity, never re-pads mid-window, reproduces the global
hop-frame indexing including the post-trim re-pad/preemphasis-restart behavior exactly, and resets its
carry state at each trim the same way the current code does. Whether that reproduction is bit-exact under
f32 rounding cannot be proven from a static read — it needs an **executed byte-equal golden experiment**
(recorded mel + full transcript on `./nemotron` and `./nemotron_multi`, asserted equal before/after).

**Verdict: REFUTED as written; UNCERTAIN under a correct carry-and-realign design.** F1 carries **hidden
correctness scope** (it is a streaming-correctness change masquerading as a perf change) and **MUST** be
gated by a byte-equal golden-transcript + golden-mel regression test on both Nemotron variants and on
Multitalker (same pattern, `multitalker.rs:326,334,439-440`). Do not land F1 on perf grounds alone.

---

## Roll-up

| Sub-claim | Verdict | Why | Must be golden-gated? | Hidden scope |
|-----------|---------|-----|-----------------------|--------------|
| **F2** plan caching | **SURVIVES** | `stft` already delegates to `stft_with_plan`; plan is deterministic from `n_fft`; divergent mel flavors live outside `stft` | Optional (cheap insurance) | None, *if* kept distinct from mel-front-end unification (A5 Q2) |
| **F3** `TensorRef` view | **SURVIVES (conditional)** | Byte-identical for C-contiguous inputs (verified for Nemotron); ordering preserves no-aliasing | **Yes** (4 variants) | Contiguity-*error* path: non-contiguous input silently copies under `from_array` but **errors** under `from_array_view` |
| **F1** incremental mel | **REFUTED as stated / UNCERTAIN** | Preemphasis restart + `n_fft/2` re-pad + frame realignment change mel at seams; current whole-buffer recompute is load-bearing; trim already injects seams to match | **Yes — mandatory** | It is a streaming-correctness change, not pure perf; must reproduce current seam-bearing output incl. trim-time re-pad/preemph-restart |

**Umbrella claim ("all three produce identical model output"): REFUTED.** F2 holds and F3 holds
conditionally, but F1 as literally stated changes mel at every chunk boundary. The umbrella is only
salvageable if F1 is re-scoped from "compute mel only over new frames" to "incremental mel that reproduces
the current output byte-for-byte, gated by a golden test," at which point its equivalence is UNCERTAIN
pending an executed experiment rather than provable from code.

**Recommendations for the synthesis lane (90):**
1. Split F1 out of the "safe perf" bucket. Reclassify as a **streaming-correctness refactor** that depends
   on A6's golden-mel + golden-transcript fixtures landing first. Sequence: A6 fixtures -> F1.
2. F2 can land independently and early (lowest risk); keep it strictly separate from any mel-front-end
   unification (A5 Q2/Q3), which is a real numeric change.
3. F3 can land per-variant behind the same golden fixtures; add an explicit check/assert that migrated
   inputs are standard-layout so the `from_array` -> `from_array_view` switch never silently flips a
   would-be copy into a runtime `Err`.
