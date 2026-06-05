# RU — Upstream NVIDIA/NeMo reference (Nemotron 3.5 streaming multilingual)

> Lane: RESEARCH (upstream reference). Wave R. PLAN ONLY, no code.
> Purpose: establish what the ORIGINAL model/framework does so A1 (correctness) and A4 (models)
> can judge the multilingual language-lock as fixable-vs-inherent.

**`/think` frameworks used:** (1) **map-vs-territory** — separate what the ONNX *graph* actually
conditions on (the territory: the export script source) from how the library *talks about* it (the
map: docstrings, the `"auto"` index). (2) **five-whys** — trace the language-lock from symptom to the
exact conditioning mechanism. (3) **steel-manning** — argue the strongest case that the lock is
inherent before concluding it is partly fixable.

**Evidence tiers used below:**
- **[CODE]** = locally verified, file:line in this repo. Highest confidence.
- **[CARD]** = NVIDIA HuggingFace model card / NVIDIA blog. Authoritative for intended behavior.
- **[INFER]** = my inference combining CODE + CARD. Labelled; not independently verified.
- Web access was AVAILABLE. Nothing here is a blind fallback. One item (the exact LID granularity
  inside the `<lang>` tag emission) is **[CARD+INFER]** because NVIDIA documents the behavior but not
  the frame-level mechanism; I flag it.

---

## 1. The conditioning mechanism (decisive — from the export script)

The single most important fact, and it is **locally verified**, not web-sourced:

The language prompt is **NOT** an encoder *input*. It is a **one-hot language vector concatenated to
the encoder OUTPUT, then run through a small MLP head (`prompt_kernel`)**, applied **per chunk**.

From `scripts/export_nemotron_streaming_multilingual.py`:

- Lines 33-37 (comment): the prompt is "a tiny MLP (Linear 1152 -> ReLU -> Linear 1024) that NeMo
  applies AFTER the cache-aware step on the encoder output. The 1152 decomposes as 1024 (encoder
  hidden) + 128 (one-hot language vector)." [CODE]
- Lines 325-359 (`EncoderWithPromptWrapper.forward`): runs `encoder.cache_aware_stream_step(...)`
  first (the FastConformer encoder, which receives **no** language signal), transposes to `(B,T,D)`,
  builds a `(B,T,num_prompts)` one-hot at `prompt_index`, **concats on the feature axis**, runs
  `self.prompt_kernel(...)`, transposes back. [CODE]
- Lines 287-300: NeMo's own reference path does exactly this —
  `model.encoder.cache_aware_stream_step(...)` then `model._apply_prompt_to_encoded(enc_raw)`. The
  Rust wrapper "exactly mirror[s] `_apply_prompt_to_encoded`" (line 344 comment). [CODE]
- Lines 380-394: the ONNX graph inputs are
  `[processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
  cache_last_channel_len, prompt_index]` and outputs are
  `[encoded, encoded_len, cache_last_channel_next, cache_last_time_next,
  cache_last_channel_len_next]`. [CODE]

### Five consequences that decide the verdict

1. **The encoder cache is language-independent.** The three cache tensors
   (`cache_last_channel/time/len`) are produced by `cache_aware_stream_step` **before** the prompt is
   applied (`export...multilingual.py:334-342`). The prompt MLP touches only the post-encoder
   activations and emits no cache. Therefore **changing `prompt_index` on the next chunk does NOT
   corrupt or invalidate the encoder cache** — the cache never saw the language signal. [CODE][INFER]

2. **`prompt_index` is a real, free, per-call ONNX input.** It has `dynamic_axes {0: "batch"}`
   (`:413`) and is fed as a fresh `np.array([idx], dtype=np.int64)` on every encoder run
   (`:583`). The graph does **not** bake one language as a constant — that was the entire point of
   exposing it as an input (`:38-42`, `:170-177` raise if pointed at the English model). So the graph
   is *capable* of a different language every chunk. [CODE]

3. **`"auto"` (index 101) is a genuine trained model slot, not a parakeet-rs invention.** It is one
   row of the 128-wide one-hot prompt space, drawn from the `.nemo`'s `prompt_dictionary`
   (`export...multilingual.py:274-277, 509`), the same table parakeet-rs embeds
   (`src/nemotron.rs:47-73`, `("auto", 101)` at line 48). The model card confirms `target_lang=auto`
   is a first-class NVIDIA inference mode. [CODE][CARD] So index 101 picks a real prompt row the
   model was trained to interpret as "you decide the language and tell me via a `<lang>` tag."

4. **The decoder/joint is plain RNNT and identical to the English model** except for vocab width
   (`export...multilingual.py:452-456`, "nothing prompt-specific here ... just wider vocab ~13k vs
   1024"). The language signal reaches the decoder **only** through the prompt-conditioned encoder
   output and through the autoregressive token history. [CODE]

5. **There is NO separate language-identification output.** The graph emits 5 tensors (encoder output
   + 3 caches + length); none is a language posterior. [CODE] The model card's language detection is
   expressed **in-band**: under `target_lang=auto` the model emits a `<lang-code>` SentencePiece token
   into the text stream after terminal punctuation (e.g. `"Hello world. <en-US>"`). [CARD] parakeet-rs
   already recognizes and strips these tags (`src/nemotron.rs:75-93` `is_lang_tag`, `:511-521`
   `get_transcript`). [CODE]

---

## 2. How the ORIGINAL handles language switching (the central question)

**NeMo's reference granularity is the UTTERANCE, and the language signal is held FIXED for the whole
decode pass.** [CARD][INFER]

- The model card and the NVIDIA fine-tuning blog both describe two modes only:
  `target_lang=<lang>` (fixed, known language) and `target_lang=auto` (model detects, emits a
  `<lang>` tag per **completed sentence**). [CARD]
- In auto mode the documented unit of detection is the **completed sentence / utterance**: "predicts
  a language_tag at the **end of each completed sentence**", and "automatically label **each
  utterance** with its detected language." [CARD]
- For a single utterance containing multiple languages, NVIDIA states the auto mode "will identify
  the **dominant** language" — i.e. it commits to one. The card explicitly says the model **does not**
  support explicit code-switching configuration or mid-stream language switching within a single
  utterance. [CARD]

**Why this matters for the lock:** even NVIDIA's own reference inference
(`speech_to_text_cache_aware_streaming_infer.py`, invoked with one `target_lang=...`) sets the prompt
**once per run** and keeps it constant across all streaming chunks of that run. NeMo does **not**
re-prompt per chunk. So the per-chunk lock parakeet-rs shows mirrors NeMo's intended usage: one
language tag conditions the whole stream. [CARD][INFER]

**Is mid-stream code-switching supported by the architecture at all?** Two answers:
- **Single fixed-language mode:** No. The whole stream is conditioned on one prompt row; the decoder's
  autoregressive history reinforces it. This is by design and matches NeMo. [CARD]
- **`auto` (index 101) mode:** Partially, in principle. The prompt row says "infer the language,"
  so the model is *not* locked to a token by the prompt — it is free to emit any language's tokens
  and a `<lang>` tag per sentence. The lock parakeet-rs observed under `"auto"` is therefore **driven
  more by carried decoder/encoder state and cold-start instability than by the prompt itself.**
  [CODE][CARD][INFER] (See section 4.)

---

## 3. Is the lock inherent to the exported graph, or is there a NeMo mechanism parakeet-rs isn't using?

**Inherent-to-graph (the steel-man, then refuted):**
- One could argue: the encoder cache + autoregressive `last_token` carry the first language's bias, so
  the graph "wants" to stay locked. True in part — but the cache is language-agnostic (section 1.1)
  and the prompt input is per-call free (section 1.2). The graph imposes **no** structural barrier to
  switching `prompt_index` between chunks. So the lock is **not** baked into the ONNX graph. [CODE]

**Mechanisms NeMo offers that parakeet-rs is NOT using:**
1. **Per-call re-prompting.** `prompt_index` is a per-chunk input. parakeet-rs sets it once in
   `set_target_lang` (`src/nemotron.rs:475-491`), passes the same value to `run_encoder` on every
   chunk (`:587-589` in `transcribe_audio`, and the streaming path), and `reset()` deliberately
   **preserves** it (`:493-509`). Nothing in the graph forces this; it is a **library choice**. [CODE]
2. **The `<lang>` tag as a language posterior proxy.** Under `"auto"`, the model emits `<lang-code>`
   tags. parakeet-rs currently **strips and discards** them (`:511-521`). These tags are NVIDIA's
   only LID signal — they could instead be surfaced as a detected-language event and used to drive a
   detect-then-set-then-reset flow at sentence boundaries. **Not used today.** [CODE][CARD]
3. **Utterance/sentence boundary reset.** NeMo's reference operates per utterance and would naturally
   re-evaluate language at a new utterance because state resets between files. parakeet-rs streams
   continuously and resets only when the caller calls `reset()`; there is no automatic
   silence/sentence-boundary reset to clear the carried decoder bias. **Library choice.** [CODE][INFER]

**No re-export is required** to switch language per chunk: the `prompt_index` input already exists in
the shipped graph. A re-export would only be needed if one wanted a *true frame-level language
posterior tensor* as a separate output (the graph does not emit one). [CODE][INFER]

---

## 4. Cold-start / first-chunk instability in cache-aware streaming

The seed symptom ("began as a garbled EN/ZH/HI mix, then locked onto Hindi") has a well-documented
upstream cause beyond the prompt:

- **Cache-aware FastConformer is trained with limited left/right context; the first chunks have an
  empty/zero-filled cache and reduced left context, so early predictions are the least reliable.**
  NeMo discussions document large offline-vs-streaming WER gaps tied to context handling, and that
  larger chunk sizes (more right context) substantially improve accuracy. [CARD/Web]
- parakeet-rs zero-fills the initial cache (`reset()` builds a fresh `NemotronEncoderCache`,
  `src/nemotron.rs:495-501`) and zero-fills the pre-encode cache on chunk 0
  (`:562-569` gate `chunk_idx > 0`). So chunk 0 runs with no left acoustic context — expected
  cold-start fragility. [CODE]
- Under `"auto"`, an unreliable first chunk can emit a wrong-language `<lang>` tag and wrong-language
  tokens; the autoregressive decoder then carries that wrong history forward (`last_token` persists,
  `:504`), and because the library never re-detects or resets, the early mistake **self-reinforces
  into a lock**. This is the five-whys root cause: not the prompt graph, but **cold-start error +
  carried autoregressive state + no re-detection.** [CODE][CARD][INFER]
- Upstream mitigation guidance: use a larger right context (the model card supports
  {0,1,3,6,13} = 80/160/320/560/1120 ms; export defaults to 6/560 ms,
  `export...multilingual.py:101-104, 208-214`) and treat the first chunk(s) as warmup. [CARD][CODE]

---

## VERDICT (hand explicitly to A1 and A4)

The multilingual language-lock is **(a) achievable in-library — fixable without re-exporting the
graph** for the practical case, with one caveat about a true LID output.

Precise breakdown:

- **NOT model-inherent / NOT baked into the ONNX graph.** `prompt_index` is a live per-call int input
  with a batch dynamic axis (`export...multilingual.py:380-394, 413, 583`); the encoder cache is
  produced *before* prompt application and is language-agnostic (`:334-342`), so re-prompting between
  chunks does not corrupt the cache. The lock is a **library choice**: parakeet-rs sets the prompt
  once and `reset()` preserves it (`src/nemotron.rs:475-491, 493-509`). [CODE → verdict (a)]

- **Fixable in-library, three available levers, no re-export:**
  1. Allow `set_target_lang` to take effect mid-stream (it already mutates `prompt_index`; the gap is
     that callers are told to also `reset()`, and `reset()` keeps the carried decoder bias unless
     called). Expose a `detect-then-set` / `change-language` flow at sentence boundaries.
  2. **Surface the `<lang>` tags instead of silently stripping them** (`src/nemotron.rs:511-521`):
     they are NVIDIA's only LID signal and enable an auto-redetect loop.
  3. Add an optional **silence/sentence-boundary reset** so a wrong early commitment under `"auto"`
     does not self-reinforce (addresses the cold-start root cause in section 4).

- **Caveat — what is NOT fixable in-library:** the model has **no frame-level language posterior
  output**; LID is only the in-band `<lang>` tag emitted per completed sentence under `auto`
  (`:75-93`, [CARD]). Sub-sentence / true mid-word code-switching is **not supported by NVIDIA's
  reference either** (the card says it picks the dominant language per utterance and explicitly does
  not support mid-stream switching). So "re-detect language every word" is **model-inherent
  not-supported**; "re-detect at sentence/utterance boundaries and re-prompt" is **fixable in-library
  (a).** A graph re-export (verdict (b)) would only be justified if the team wanted a dedicated LID
  posterior output, which is a larger, optional effort, not required for boundary re-detection.

**One-line verdict:** Per-chunk language re-detection at *sub-sentence* granularity is model-inherent
and not supported (matches upstream). Language re-detection and re-prompting at *sentence/utterance
boundaries* is **fixable in the library today** by re-prompting `prompt_index`, surfacing the `<lang>`
tags, and resetting carried decoder state at boundaries — no graph re-export needed.

---

## Cross-references for other lanes

- **→ A1 (correctness & streaming):** The lock is a **library choice, severity = real bug not inherent
  limit**, root cause = three compounding issues: (1) `prompt_index` never re-prompted per chunk
  (`src/nemotron.rs:587-589` and streaming path), (2) `reset()` preserves both target lang AND is
  never auto-triggered, leaving carried `last_token` decoder bias (`:493-509`), (3) cold-start: chunk
  0 runs with zero left-context cache (`:562-569`) producing an unreliable first language commitment
  under `"auto"`. The encoder cache is language-agnostic (export script `:334-342`), so re-prompting
  mid-stream is **safe** — flag this when assessing regression risk of a fix. Recommend A1 file a
  finding: "auto mode self-reinforces an early wrong-language commitment; no boundary re-detection."

- **→ A4 (model coverage & features):** "Language re-detection" feature is **(a) in-library
  achievable, no re-export.** Concrete sub-tasks: (i) surface the currently-stripped `<lang>` tags
  (`:511-521`, `is_lang_tag` `:75-93`) as a detected-language API/event; (ii) a `reset()` variant /
  flag that also clears carried language bias for clean boundary switching; (iii) document that
  sub-sentence code-switching is model-inherently unsupported (matches NVIDIA). `"auto"` (idx 101) IS
  a genuine trained slot — keep it. Right-context is tunable {0,1,3,6,13}=80..1120 ms at export
  (`export...multilingual.py:101-104`); a larger-context export variant could reduce cold-start lock —
  optional A4 item.

- **→ A2 (performance):** `prompt_index` is a trivial scalar input; re-prompting per chunk adds
  ~zero cost (the prompt MLP runs every chunk regardless). No perf objection to a per-chunk
  re-prompt fix. Cold-start mitigation via larger right context trades latency for accuracy
  (560 ms default vs up to 1120 ms).

- **→ A3 (API & ergonomics):** `set_target_lang` docstring (`src/nemotron.rs:460-474`) tells callers
  to also `reset()` for clean mid-utterance switching — but `reset()` preserves the language and is
  manual. The API needs a clearer detect/switch/boundary-reset story. Consider exposing a
  detected-language accessor (from the `<lang>` tags) on `NemotronHandle`.

- **→ A6 (tests):** Add a regression fixture using `./nemotron_multi`: feed an EN-then-HI (or
  EN-then-ZH) clip under `"auto"` and assert the transcript is not monolingual-locked after a
  sentence boundary; and a unit test that `prompt_index` actually changes encoder output (the export
  script's own multi-language verification at `:556-586` — es-ES idx 2, ja-JP idx 10 — is the
  blueprint: a wrong "prompt accepted but ignored" path would pass a single-language test only).

- **→ 02 (architecture):** The prompt head is a post-encoder MLP outside the cache loop; this is the
  only structural difference between the English and multilingual Nemotron graphs (export script
  `:19-26, 452-456`). Any shared-abstraction extraction across model_X pairs can treat Nemotron
  multilingual as "English encoder + optional post-encoder prompt projection," not a distinct encoder.

### Key citations
- `scripts/export_nemotron_streaming_multilingual.py:33-42` (prompt = post-encoder MLP, 1024+128),
  `:170-177` (prompt is a real input, English model would bake it wrong), `:325-359`
  (`_apply_prompt_to_encoded` mirrored; one-hot concat per chunk), `:380-394` (graph I/O, no LID
  output), `:413, 583` (prompt_index is a per-call dynamic input), `:556-586` (multi-lang verification).
- `src/nemotron.rs:47-73` (PROMPT_DICTIONARY, `"auto"`=101), `:75-93` (`is_lang_tag`), `:460-491`
  (`set_target_lang`), `:493-521` (`reset` preserves lang; `get_transcript` strips lang tags),
  `:562-569` (chunk-0 zero pre-encode cache).
- NVIDIA model card — language conditioning, `target_lang=auto`, `<lang>` tag per sentence, no
  mid-stream code-switching, 40 locales / 3 tiers:
  https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- NVIDIA fine-tuning blog — two modes (known / auto), language_tag emitted at end of each completed
  sentence, every training clip carries a target_lang tag:
  https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
- NeMo cache-aware streaming (cold-start / context, offline-vs-streaming gap, right-context tuning):
  https://github.com/NVIDIA-NeMo/NeMo/discussions/7010 ,
  NeMo ASR models doc https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
