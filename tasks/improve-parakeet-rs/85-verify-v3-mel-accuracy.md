# 85-verify-v3 — Adversarial verification of the mel front-end accuracy worry

> Verifier V3 (Wave V). Scope: the single HIGH-escalation candidate bundling A1-09 (Hann window
> periodicity), A1-10 (log-mel floor / normalization divergence), and A5 Q2 (mel duplication with
> divergent numerics). Default = REFUTED unless proven.

**/think framework:** red-team + scientific-method. I steel-manned the worry (tried to prove the
divergences ARE real accuracy bugs), then sought the single observation that would falsify each
sub-claim against ACTUAL code + the NeMo reference.

---

## Claim under test (steel-manned toward "this IS a bug")

> "The mel front-end divergences (symmetric vs periodic Hann window; three different log-mel
> treatments; Nemotron assuming no feature normalization) are real deviations from the NeMo
> reference preprocessing that degrade transcription accuracy, especially for the multilingual model."

The claim is a **bundle of three sub-claims**. I verdict each separately, then the bundle.

---

## VERDICT SUMMARY

| Sub-claim | Verdict | Basis |
|---|---|---|
| A1-09 Hann window symmetric-vs-periodic | **REFUTED** | NeMo uses `periodic=False` (SYMMETRIC). The audit had it backwards; the Rust code already matches NeMo. |
| A1-10 / Q2 `x.max(0.0)` log-floor divergence | **REFUTED** | Mathematically a no-op (mel power is always >= 0). Cosmetic only, zero numerical effect. |
| A1-10 Nemotron "no normalization" assumption | **REFUTED for multilingual; UNCERTAIN-but-low for EN** | Multilingual config = `normalize:"NA"` (no-op in NeMo) and the model produces intelligible output, which is impossible if features were grossly mis-scaled. EN value is not web-verifiable, but the same empirical signal holds. Not a HIGH escalation. |
| **Bundle as a HIGH-escalation accuracy bug** | **REFUTED** | No sub-claim survives as a real accuracy regression. Do NOT escalate to HIGH/CRITICAL. |

**The flagged finding does NOT escalate. It is downgraded.** A1-09 should be CLOSED/CORRECTED
(the code is right, the audit text is wrong). A1-10 / Q2 collapse to a code-quality consolidation
item (cite Q2), not a correctness bug. The only residual is a LOW-confidence verification TODO for
the EN Nemotron `normalize` setting, satisfiable by one mel-dump fixture, NOT a code change.

---

## Sub-claim 1 — A1-09 Hann window periodicity — REFUTED (audit was backwards)

**Code (actual):** `src/audio.rs:64-68`

```
fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}
```

This is the **symmetric** Hann window (denominator `N-1`). A1-09 asserts NeMo trains with a
**periodic** window (denominator `N`) and that the Rust symmetric window is therefore a deviation.

**NeMo reference (verified, two independent sources):**
- NeMo `features.py` (`FilterbankFeatures`) builds the window with
  `window_tensor = window_fn(self.win_length, periodic=False)`, i.e. `torch.hann_window(win_length,
  periodic=False)` -> **SYMMETRIC**. [NeMo `nemo/collections/asr/parts/preprocessing/features.py`,
  fetched 2026-06]
- PyTorch identity confirmed by search: `hann_window(L, periodic=True) == hann_window(L+1,
  periodic=False)[:-1]`; `periodic=False` returns the symmetric window. The symmetric form for a
  length-`L` window is `0.5 - 0.5*cos(2*pi*n/(L-1))` — **exactly the Rust formula**.

**Conclusion:** The Rust window already matches the NeMo training-time window. A1-09 inverted the
NeMo default (NeMo does NOT use `periodic=True` for ASR; it explicitly passes `periodic=False`).
Implementing A1-09's "fix" (switch to periodic `N` denominator) would INTRODUCE the very divergence
the audit feared. **REFUTED. Correct the audit; do not change the code.**

Citations: `src/audio.rs:64-68`; NeMo `features.py` `window_fn(..., periodic=False)`; PyTorch
`torch.signal.windows.hann` / `torch.hann_window` periodic semantics.

---

## Sub-claim 2 — A1-10 / Q2 the `x.max(0.0)` log-floor divergence — REFUTED (no-op)

**Code (actual):**
- `src/nemotron.rs:784`  -> `(x + LOG_ZERO_GUARD).ln()`            (no clamp)
- `src/audio.rs:241`     -> `(x + log_zero_guard).ln()`            (no clamp)
- `src/multitalker.rs:675` -> `(x.max(0.0) + LOG_ZERO_GUARD).ln()` (clamp)
- `src/parakeet_eou.rs:261` -> `(x.max(0.0) + LOG_ZERO_GUARD).ln()` (clamp)

The guard constant is identical everywhere: `2^-24 == 5.9604645e-8` (audit already conceded this).
The only divergence is the `x.max(0.0)` clamp on two of four paths.

**Why it is a no-op:** `x` is `mel_basis.dot(spectrogram)`. `spectrogram[k] = output[k].norm_sqr()`
(`audio.rs:120`) which is `|complex|^2 >= 0` always. The mel filterbanks
(`create_mel_filterbank` Slaney, `audio.rs:174`, and `create_mel_filterbank_htk`,
`parakeet_eou.rs:268`) are built from `0.0.max(lower.min(upper))` style triangular weights, so
**all filterbank weights are >= 0**. A non-negative matrix times a non-negative vector is
non-negative. Therefore `x >= 0` always, and `x.max(0.0) == x` for every element. The clamp can
never alter a single output value.

**Conclusion:** The divergence is cosmetic. It changes no transcript on any input. A1-10's own text
already noted this is "harmless but inconsistent"; the steel-man does not survive. **REFUTED as an
accuracy bug.** It remains a legitimate but LOW/NIT code-quality item, correctly captured by A5 Q2
(consolidate into one `MelFrontend`). It is not a HIGH escalation.

Citations: `src/audio.rs:120,174`; `src/multitalker.rs:675`; `src/parakeet_eou.rs:261,268`;
`src/nemotron.rs:784`.

---

## Sub-claim 3 — Nemotron "no normalization" — REFUTED for multilingual; low-risk UNCERTAIN for EN

This is the sharpest sub-claim and the reason the bundle was flagged HIGH. I attacked it hardest.

**What the Rust code does:** ONE `compute_mel_spectrogram` (`src/nemotron.rs:775-785`) serves BOTH
`NemotronMode::EnglishOnly` and `NemotronMode::Multilingual` (single mel path, no mode branch — see
`nemotron.rs:300-366,775`). It returns raw log-mel and **never** applies per-feature normalization,
unlike the CTC/unified path (`audio.rs:245-263`). The crate does not read a sidecar `config.json`
for Nemotron (`nemotron.rs:30`), so this is a hardcoded design decision, not config-driven.

**What "should" happen, per NeMo reference:** normalization is governed by the preprocessor's
`normalize` field, applied during mel extraction. In NeMo `features.py`, `normalize="per_feature"`
performs per-channel mean/std; **any unrecognized value (including `"NA"`, `None`) returns the
features UNCHANGED** (no normalization). [NeMo `features.py`, fetched 2026-06]

The export path is decisive on WHERE normalization lives. Both export scripts build the streaming
buffer with `online_normalization=False`
(`export_nemotron_streaming.py:191`, `export_nemotron_streaming_multilingual.py:249`). In NeMo's
`CacheAwareStreamingAudioBuffer.extract_preprocessor`, `online_normalization=True` forces
`cfg.preprocessor.normalize = "None"`; with `online_normalization=False` the **preprocessor keeps
its own `normalize` setting** and normalizes (or not) once during mel extraction. [NeMo
`streaming_utils.py`, fetched 2026-06]. The ONNX graph is traced from `EncoderWithPromptWrapper`
whose INPUT is the already-extracted `processed_signal` mel
(`export_..._multilingual.py:262,318-341`). **Normalization is therefore OUTSIDE the ONNX graph** —
whatever the preprocessor applied must be replicated on the Rust side. So the question genuinely
matters: the Rust mel must match the preprocessor's `normalize` for each model.

**Multilingual — REFUTED:**
- Shipped `nemotron_multi/config.json` preprocessor block reads `"normalize": "NA"` (verified by
  reading the downloaded model config). `"NA"` is an unrecognized value in NeMo `features.py` ->
  **no normalization**. The Rust no-normalization path therefore MATCHES the multilingual model.
- Empirical falsification test: if the multilingual encoder expected per-feature-normalized inputs
  (typical mel log-energies sit around -5..+5 after `ln`, with large per-band offsets) and instead
  received raw un-normalized log-mel, the output would be near-random / collapse to blanks, not
  intelligible text. The seed finding (`00-context-pack.md:19-21`) reports the multilingual model
  produces a coherent (if language-locked) transcript. Coherent output is **incompatible** with a
  gross feature-scale mismatch. The language-lock is a prompt/decoder-state issue (A1-01), NOT a mel
  normalization issue. The claim that "especially the multilingual model" is degraded by missing
  normalization is **directly contradicted by observed behavior. REFUTED.**

**English-only — UNCERTAIN but NOT HIGH:**
- The EN export script dumps `"preprocessor": {... "normalize": "per_feature" ...}`
  (`export_nemotron_streaming.py:384`). Taken at face value this would mean the EN encoder expects
  per-feature-normalized mel, and the Rust EN path (which skips normalization) would be wrong.
- BUT this value is a **hardcoded string literal in the dumped dict**, not read from
  `model.cfg.preprocessor.normalize` (contrast `prompt_dict` at `..._multilingual.py:274-276`,
  which IS pulled from `model.cfg`). The multilingual script's `"NA"` is likewise hand-typed. So
  the dumped `"per_feature"` is the script author's annotation, not extracted ground truth, and is
  not reliable evidence of the real `.nemo` setting.
- The shipped `nemotron/` (EN) directory has NO `config.json` (verified: only `encoder.onnx`,
  `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model`), so there is no second source to
  cross-check on disk, and the model card README does not expose the preprocessor block.
- The same empirical argument as multilingual applies: the EN model is the primary shipped path and
  produces correct transcripts in normal use; gross un-normalized input would not.
- Residual genuine uncertainty: I cannot rule out, from web sources alone, that the EN `.nemo`
  truly uses `per_feature` AND the model is merely tolerant. This is the one thread that keeps this
  from a clean REFUTED. It is resolvable by a numeric fixture, NOT by code inspection.

**Net:** the "no normalization" worry is REFUTED for the multilingual model (config + behavior both
confirm) and is a LOW-confidence open question for EN that does NOT meet the bar for HIGH/CRITICAL
escalation. If EN ever proves to need `per_feature`, the fix is mode-conditional normalization in
`compute_mel_spectrogram`, scoped to EN only — but there is no positive evidence it is needed today.

Citations: `src/nemotron.rs:30,300-366,775-785`; `src/audio.rs:245-263`;
`export_nemotron_streaming.py:191,384`; `export_nemotron_streaming_multilingual.py:249,262,514`;
`nemotron_multi/config.json` (`"normalize":"NA"`); NeMo `features.py` normalize handling; NeMo
`streaming_utils.py` `online_normalization` / `extract_preprocessor`; seed `00-context-pack.md:19-21`.

---

## Cross-checked secondary mel parameters (no divergence found)

NeMo `features.py` defaults vs Rust constants (`nemotron.rs:15-21`, `audio.rs`):
- `n_fft=512`, `win_length=400` (25ms), `hop=160` (10ms), `n_mels=128` — MATCH.
- `preemph=0.97` — MATCH (`PREEMPH`).
- `log_zero_guard_value=2^-24`, `log_zero_guard_type="add"` — MATCH (`(x + 2^-24).ln()`).
- mel scale = Slaney (`mel_norm="slaney"`) — MATCH for `audio.rs`/Nemotron/multitalker; EOU uses HTK
  on purpose for a different model (`parakeet_eou.rs:268`), legitimately so.
- `dither=1e-5` but applied **only when `self.training`** -> NOT at inference. Rust correctly omits
  dither. MATCH (no action needed; A-lanes did not flag this, correctly).

None of these surface a real deviation.

---

## What the synthesis lane (90) should record

1. **A1-09: CLOSE / CORRECT.** Code already matches NeMo (`periodic=False`, symmetric). The audit's
   premise is inverted. Do not implement the proposed window change; it would cause the regression.
   If anything, add a one-line code comment + a golden-mel test to lock the symmetric window in.
2. **A1-10 `max(0.0)` + Q2: downgrade to code-quality (LOW/NIT).** Provably a no-op; fold into the
   A5 Q2 `MelFrontend` consolidation, gated by a golden-mel fixture. Not a correctness/accuracy item.
3. **A1-10 Nemotron normalization: REFUTED for multilingual; keep ONE LOW verification TODO for EN.**
   The TODO is a test fixture, not a code change: dump NeMo `processed_signal` for `./nemotron` via
   the streaming buffer (`online_normalization=False`) and compare per-bin against the Rust mel; if
   they match, close permanently. No code change unless the fixture proves a mismatch.
4. **Do NOT escalate this bundle to HIGH/CRITICAL.** The accuracy worry does not survive
   verification. The genuine multilingual accuracy problem is the language-lock (A1-01), which is
   unrelated to the mel front-end.

## Reverification (single fixture closes the only residual)

Export NeMo `processed_signal` (the buffer's mel) for a fixed WAV for BOTH `./nemotron` and
`./nemotron_multi` with `online_normalization=False`, and compare per-bin to the Rust
`compute_mel_spectrogram` output. Expectation: multilingual matches with no normalization; EN
matches with no normalization too (if it does, sub-claim 3 is fully REFUTED; if EN diverges,
re-open EN-only as MEDIUM with the exact per-bin error as evidence). This is the only outstanding
numeric check and it requires no source-code change to run.
