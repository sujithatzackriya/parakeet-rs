# 85 — Adversarial verification V1 (Wave V): the language epic load-bearing claim

> Verifier V1. PLAN ONLY. One claim, refuted hard against actual code + export script.
> Default posture: REFUTED unless the mechanism is positively traced end to end.

**`/think` framework (red-team):** I treat the claim as a hostile witness. I split it into its
independent conjuncts, then for each I look for the ONE graph input / carried tensor / missing signal
that breaks it. I privilege the *territory* (the ONNX I/O signature + the Rust calls that set it) over
the *map* (docstrings, the audit prose, the "auto" label). I only let a conjunct survive if I can
point at the exact `file:line` that makes it true, and I actively hunt for a hidden state path that
re-encodes the old language even after re-prompting.

---

## The claim, decomposed into 5 testable conjuncts

C1. The language prompt is concatenated to the encoder **OUTPUT** via the `prompt_kernel` MLP, **not**
    the encoder input.
C2. Because of C1, the streaming encoder **cache is language-agnostic** (re-prompting next chunk does
    not corrupt/invalidate the cache).
C3. The library can **change `prompt_index` per chunk** to re-prompt, with no model re-export.
C4. The library can **reset decoder state at utterance boundaries to re-detect** language.
C5. The model's emitted `<lang>` tags can be **surfaced as a detected-language signal**.

Overall claim = "fixable IN-LIBRARY, no re-export." I tested each conjunct against code.

---

## C1 — prompt applied to encoder OUTPUT via prompt_kernel MLP — SURVIVES

Traced in the export wrapper, not inferred from docstrings:

- `scripts/export_nemotron_streaming_multilingual.py:334-342` — `self.encoder.cache_aware_stream_step(...)`
  runs FIRST and returns `(encoded, enc_len, ch_n, tm_n, ln_n)`. The FastConformer step receives only
  acoustic inputs + caches; `prompt_index` is **not** passed to it.
- `:347-358` — only AFTER the encoder step: transpose to `(B,T,D)`, build a `(B,T,num_prompts)`
  one-hot at `prompt_index` (`:349-356`), `torch.cat([encoded, prompt], dim=-1)`, then
  `self.prompt_kernel(...)` (`:357`). So the prompt touches post-encoder activations only.
- `:380-394` — graph inputs `[processed_signal, processed_signal_length, cache_last_channel,
  cache_last_time, cache_last_channel_len, prompt_index]`; outputs `[encoded, encoded_len,
  cache_last_channel_next, cache_last_time_next, cache_last_channel_len_next]`. `prompt_index` is a
  top-level graph input, but the three `*_next` cache tensors are returned values `ch_n, tm_n, ln_n`
  from `cache_aware_stream_step` (`:359`), produced before the prompt path.

Rust side confirms the contract is wired the same way: `src/model_nemotron.rs:149-164` feeds
`prompt_index` as a *separate* ORT input alongside the caches, and `:185-195` reads
`cache_last_channel_next/cache_last_time_next/cache_last_channel_len_next` straight out as the new
cache. The Rust never folds the prompt into the cache it carries forward (`src/nemotron.rs:694`
`self.encoder_cache = new_cache`).

**Verdict C1: SURVIVES.** The prompt is genuinely a post-encoder MLP head. I could not find any path
where `prompt_index` feeds the encoder body or the emitted cache.

---

## C2 — encoder cache is language-agnostic — SURVIVES (with an independent proof the audits under-used)

The audits asserted this from the topology (prompt is post-cache). I found a STRONGER, executable
proof inside the export script that settles it without trusting topology:

- `:556-586` — the export's own verification loop feeds the **identical** `cache_last_channel /
  cache_last_time / cache_last_channel_len` tensors for every language (es-ES idx 2, ja-JP idx 10),
  varying ONLY `prompt_index` (`:583`), and gets **different** encoder outputs that each match NeMo's
  reference to `<1e-4` (`:585-586`). This proves two things at once: (a) `prompt_index` is honored
  (output changes by language), and (b) the cache INPUT is language-independent (same cache produces
  every language correctly). A "prompt accepted but ignored" bug or a "cache encodes language" bug
  would both fail this check; the comment at `:539-545` says exactly that was the design intent.
- The cache OUTPUT is also language-independent: `:359` returns `ch_n, tm_n, ln_n` directly from
  `cache_aware_stream_step`, which never saw `prompt_index`.

**Verdict C2: SURVIVES.** Re-prompting on chunk N+1 cannot corrupt the carried encoder cache, because
the cache is computed by a sub-graph the prompt never enters, and the export's multi-language parity
test demonstrates this empirically at export time.

---

## C3 — change `prompt_index` per chunk, no re-export — SURVIVES

- `src/model_nemotron.rs:140-162` — `run_encoder(..., prompt_index: Option<i64>)` builds a fresh
  `Array1::from_vec(vec![idx])` and pushes it as the `prompt_index` ORT input **on every call**. There
  is no caching, no freezing, no "set once" at the model layer.
- `scripts/export_nemotron_streaming_multilingual.py:409-413` — `prompt_index` has
  `dynamic_axes {0: "batch"}` and is exported as a live input, not a constant. `:170-177` (per RU) and
  the `:38-42` purpose comment confirm the graph was built specifically so one ONNX serves every
  language via this int input.
- The only reason it never changes today is library policy: `self.prompt_index` is set once in
  `set_target_lang` (`src/nemotron.rs:489`) and fed unchanged at `:591` and `:691`; `reset()`
  (`:495-509`) does not touch it. Nothing in the graph enforces this.

**Verdict C3: SURVIVES.** Per-chunk re-prompting needs zero re-export. It is a one-field library
change to pass a varying `prompt_index` to `run_encoder`.

---

## C4 — reset decoder state at utterance boundaries to re-detect — SURVIVES ONLY AS "CALLER-DRIVEN BOUNDARY", NOT AS AUTONOMOUS RE-DETECTION

This is where I attacked hardest, and the claim is **half true**. I split it:

C4a — *Is a decoder-state reset required, and is re-prompting alone insufficient?*
Yes, a reset is required, and the claim itself concedes this with "and/or reset decoder state."
The carried state that re-encodes the old language is real and is THREE tensors, all proven to persist
across chunks:
- `self.last_token` (autoregressive) — written at `src/nemotron.rs:763`, fed back at `:744`.
- LSTM `state_1/state_2` — carried at `:764-765`, fed at `:745-746`.
- `self.encoder_cache` — carried at `:694`.
`reset()` clears all three (`:496-504`). So pure re-prompting without a reset would leave the
autoregressive history biasing the next token toward the committed language (A1-01 root cause #2,
`A1-correctness.md:19`). The claim's "and/or reset decoder state" is therefore **necessary**, not
optional, for a real switch. The claim is internally honest here.

C4b — *Does resetting mid-utterance corrupt the transcript?*
Not the ALREADY-COMMITTED transcript (accumulated tokens are kept; `reset()` only clears
`accumulated_tokens` for a NEW utterance, `:508`). BUT a mid-stream `reset()` re-introduces COLD START:
it zero-fills a fresh `NemotronEncoderCache` (`:496-501`), sets `chunk_idx = 0` and
`audio_processed = 0` and **clears `audio_buffer`** (`:505-507`). The next `transcribe_chunk` then
hits the `is_first_chunk` branch (`:646-656`) which zero-pads the pre-encode cache region, i.e. runs
with no left acoustic context. That is precisely the cold-start fragility RU section 4 documents
(`RU-upstream.md:144-164`) and the original seed-symptom cause. So "reset at boundary to re-detect" is
mechanically safe for past text but **re-triggers the unreliable first chunk** for the new segment.
This is a real cost the claim glosses over, but it does not falsify the claim: it is the same
behavior NeMo's per-utterance reference has, and it does not corrupt prior output.

C4c — *Is "re-detect at boundary" achievable without an external boundary signal the model doesn't
provide?* **NO — and this is the genuine gap.** Nemotron's graph emits no VAD / EOU / silence signal
(graph outputs are encoder+caches+len only, `:388-394`; A4 `:51` confirms "no native silence signal
in this graph — EOU has it, Nemotron does not"). So the LIBRARY cannot autonomously locate an
utterance boundary; the *caller* must supply one (their own VAD, or the `<lang>`-tag-after-punctuation
heuristic from C5). The claim says "reset decoder state AT utterance boundaries" — the boundary itself
is not something the in-library Nemotron path produces. RU and A1/A4 all flag this
(`A1-correctness.md:27`, `A4-models.md:51`, `RU-upstream.md:194-197`).

**Verdict C4: SURVIVES, scoped.** The *mechanism* (re-prompt + reset to switch language) holds and is
in-library. The autonomous *trigger* ("at utterance boundaries") is NOT in-library for Nemotron: it
requires either a caller-supplied boundary or the `<lang>`-tag heuristic. Sub-sentence/mid-word
code-switching under one stream is model-inherently unsupported (NVIDIA picks dominant language per
utterance, `RU-upstream.md:96-97,194-197`). So the SAFE in-library fix steps are:
(1) make `set_target_lang` effective mid-stream (already mutates the index),
(2) add an atomic `reset_with_lang(lang)` / boundary-reset helper that re-prompts AND clears
   `last_token`+LSTM (encoder cache reset optional but triggers cold start),
(3) NOT an autonomous per-chunk re-detector — that needs an external boundary signal.

---

## C5 — surface `<lang>` tags as detected-language signal — SURVIVES, with a reliability caveat

Mechanically derivable, traced:
- `lang_tag_ids` are REAL token ids, not invented: `SentencePieceVocab::lang_tag_ids`
  (`src/nemotron.rs:256-262`) scans the actual vocab pieces and keeps every id whose piece passes
  `is_lang_tag`. They are the genuine `<xx>` / `<xx-XX>` SentencePiece tokens.
- They are KEPT in state: the decode loop pushes every token (including lang tags) into
  `accumulated_tokens` (`:697`, `decode_chunk:762`). Stripping happens only at string-render time
  (`get_transcript:518`, per-chunk `:716`). So the detected language is observable: scan
  `accumulated_tokens` for ids in `lang_tag_ids`, decode the most recent to its `<xx-XX>` string.
  A `detected_language()` getter is a pure-additive library change (A4 F1 option 1, `A4-models.md:49`).

Caveat that the claim should not over-sell: `is_lang_tag` (`:78-93`) is a **shape heuristic**
(`<` + 2 lowercase + `>`, or `<xx-XX>`), decoupled from the model's declared special-token set. A1-08
(`A1-correctness.md:124-134`) shows it can over-strip a legitimate content piece of shape `<no>`/`<so>`
or under-strip a tag in an unexpected casing. So the detected-language SIGNAL exists and is
surfaceable, but its precision depends on tightening detection to exact token-id membership rather
than the current string-shape match. That is a fix, not a refutation.

**Verdict C5: SURVIVES** (signal is real and surfaceable); reliability hardening (exact-id stripping)
is a coupled sub-task, not a blocker.

---

## OVERALL VERDICT: SURVIVES (scoped) — the claim is correct on the load-bearing mechanism; one
## sub-clause ("at utterance boundaries") is over-stated and must be re-scoped

The core, epic-load-bearing assertion — **"fixable in-library with no model re-export because the
prompt is a post-encoder MLP head, the cache is language-agnostic, so re-prompting per chunk +
resetting decoder state can switch language, and the `<lang>` tags are a usable detected-language
signal"** — holds against the actual graph I/O and the Rust calls. C1, C2, C3, C5 survive outright;
C2 has an independent executable proof (the export's multi-language parity check, `:556-586`) that the
audits under-cited.

The ONE thing I could break is the phrase **"re-detect at utterance boundaries"** read as *autonomous*
in-library behavior: Nemotron's graph emits **no boundary/VAD/LID signal**, so the library cannot by
itself know where an utterance ends. Boundary detection must come from the caller or be approximated
from the `<lang>`-tag-after-punctuation heuristic. And a mid-stream reset re-incurs cold-start
fragility (clears `audio_buffer` + zero-cache, `src/nemotron.rs:505-507,496-501`, hitting the
`is_first_chunk` cold path `:646-656`). Neither falsifies "in-library, no re-export"; both refine its
scope.

### What is SAFE to claim as in-library, no re-export (for the synthesis lane)
1. Per-chunk re-prompting via a varying `prompt_index` to `run_encoder` (C3) — safe, trivial,
   cache-proven-agnostic (C1/C2).
2. `reset_with_lang(lang)` / explicit boundary re-prompt+state-reset helper (C4 mechanism) — safe;
   note it re-incurs cold start and must NOT auto-`reset()` inside `set_target_lang` (would silently
   drop in-flight tokens; gate behind a new method, `A1-correctness.md:29`).
3. `detected_language()` getter from kept `<lang>` tag ids (C5) — pure additive.

### What is NOT in-library / NOT in this claim's safe scope (must be re-scoped in the roadmap)
- Autonomous "detect the boundary for me" — needs external VAD or the tag-after-punctuation heuristic;
  Nemotron emits no boundary signal (`A4-models.md:51`, `RU-upstream.md:194-197`).
- True sub-sentence / mid-word code-switching in a single stream — model-inherently unsupported
  (NVIDIA commits to the dominant language per utterance, `RU-upstream.md:96-97`).
- A frame-level language-posterior tensor — would require a graph re-export; the only LID is the
  in-band `<lang>` tag (`:388-394`).
- Reliable tag stripping — current `is_lang_tag` shape heuristic can over/under-strip (A1-08); harden
  to exact token-id set before relying on `detected_language()` for control flow.

### Reverification handed to A6 / implementer
- Unit test that `run_encoder` with two different `prompt_index` values on the SAME cache yields
  different encoder output (mirrors export `:556-586`; guards a regressed "prompt ignored" path).
- Integration test on `./nemotron_multi`: EN-then-HI in one stream under `"auto"`; assert HI tail is
  NOT recovered without a boundary reset (documents the inherent limit) and IS recovered after
  `reset_with_lang`/boundary reset (proves the in-library fix).
- Assert the English-only path keeps `prompt_index == None` (`src/nemotron.rs:427-428`) and is
  untouched by any re-prompt work.
