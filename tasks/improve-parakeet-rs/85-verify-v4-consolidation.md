# 85 - Adversarial Verification V4 (Wave V): Consolidation (M2) + Un-export (M8)

**/think framework:** red-team (assume each claim is false, hunt for the counter-evidence in actual
code), backed by direct file reads of all five RNNT decoder copies, the four greedy/argmax loops, and a
repo-wide grep for the leaked `model_X` types. Default verdict if unsure: REFUTED.

**Scope read (verified line ranges, working tree):** `model_nemotron.rs:232-287`, `model_unified.rs:134-186`,
`model_multitalker.rs:197-249`, `model_eou.rs:149-195`, `model_tdt.rs:176-291`; greedy loops
`nemotron.rs:723-770`, `parakeet_unified.rs:442-496`, `parakeet_eou.rs:194-245`; argmax sites; `lib.rs:76-99`;
cross-module imports of `model_X`/`SentencePieceVocab`; `examples/` + `README.md`.

---

## CLAIM (1): "~700-900 lines genuinely removable by a shared pub(crate) RnntBackend / src/onnx/, and the consolidation preserves behavior across all variants."

### Verdict: **UNCERTAIN, leaning REFUTED as stated.** Two sub-claims, different fates:
- "~700-900 removable" -> the **upper half (800-900) is REFUTED**; a realistic, behavior-safe figure is
  **~350-550 lines**. The 700-900 number in A5-Q1 / 02 sec 2 is a gross-duplication tally that double-counts
  lines the extraction cannot actually delete (call sites, per-variant glue, divergent bodies).
- "preserves behavior across all variants" -> **REFUTED.** The greedy-loop consolidation cannot be done
  without first picking one argmax tie/NaN policy (A1-06), and that pick **changes the output of at least
  Nemotron and Unified** on NaN/tie inputs. Consolidation is behavior-*reconciling*, not behavior-preserving.

### Evidence the ~700-900 figure is inflated

The five RNNT decoder-step copies are genuinely near-identical and ARE extractable:

| copy | span | lines | delta vs Nemotron |
|------|------|-------|-------------------|
| `model_nemotron.rs:232-287` | run_decoder | 56 | baseline (i32 targets via `from_shape_vec`, `target_length` passed) |
| `model_unified.rs:134-186` | run_decoder | 53 | targets via `Array2::from_elem`; identical otherwise |
| `model_multitalker.rs:197-249` | run_decoder | 53 | **i64 targets**; output names `states_1`/`states_2` not `output_states_*`; **NO `target_length` input** |
| `model_eou.rs:149-195` | run_decoder | 47 | targets passed in as `&Array2<i32>` (not built); returns logits as **`Array3` not `Array1`**; reshapes states to hardcoded `(1,1,640)` |
| `model_tdt.rs:176-291` | greedy_decode (step inlined) | 116 | step not factored out; **splits logits into vocab + 5 duration tokens** |

Raw duplicated decoder-step text = 56+53+53+47 = **209 lines** in the four factored copies, plus ~30 of the
TDT 116 are the same step. Total gross dup in the decoder step ~ **240 lines**. Replacing with one ~60-line
parameterized helper + 5 thin call sites (~5 lines each) = **net deletion ~150 lines**, not 240.

The differences are not cosmetic; the helper needs **at least four config knobs**:
1. target dtype `i32` vs `i64` (multitalker is i64) - `model_multitalker.rs:204`.
2. whether `target_length` is an input (multitalker omits it) - absent at `model_multitalker.rs:207-212`.
3. output state names `output_states_*` vs `states_*` - `model_multitalker.rs:220,224`.
4. logits return rank `Array1` vs `Array3` (EOU returns `[1,1,vocab]`) - `model_eou.rs:184`.

That is tolerable (a struct of names + a dtype enum), so the decoder step SURVIVES as extractable. But it is
~150 net lines, not ~250.

The greedy **loop** is where the 700-900 number breaks down. The four loops are NOT the same function:
- `nemotron.rs:734-766`: plain blank-break, `max_symbols_per_step = 10`, manual `>` argmax (no finite guard).
- `parakeet_unified.rs:459-492`: blank-break, `MAX_SYMBOLS_PER_STEP = 10`, iterator `max_by` argmax, returns
  `(token, absolute_frame)` tuples (timestamp bookkeeping the others lack).
- `parakeet_eou.rs:194-242`: `syms_added < 5`, **plus three special exits** - `max_idx == 0` break
  (`:217`), `<EOU>`-token soft-reset-and-return (`:221-228`), vocab-overflow break (`:230-232`), and inline
  tokenizer.decode per token (`:238`). This loop does NOT generalize.
- `model_tdt.rs:201-288`: TDT **duration-token frame skipping** (`t += duration_step`, `:281-283`) - a
  fundamentally different frame-advance rule, plus duration argmax.

So of the "4x greedy loop" duplication, only ~2 loops (Nemotron, Unified) share a body, and even those differ
in return type (plain `usize` vs `(usize,usize)` for timestamps). EOU and TDT each need their own loop. The
realistic loop-level saving is **~40-60 lines** (collapsing Nemotron+Unified, extracting a shared `argmax`),
not the ~120-200 implied by counting all four.

Adding the other genuinely-extractable pieces (all SURVIVE individually, citing 02 sec 2 / A5-Q1, re-verified):
- session build+commit idiom 6x (`load_session` helper): ~6 lines x 12 -> ~1; **net ~50-60 lines**.
- `find_encoder`/`find_decoder_joint` 3 divergent copies (`model_tdt.rs:63-109`, `model_unified.rs:59-91`,
  inline multitalker): **net ~40-60 lines** (and reconciling the divergent candidate lists is itself a small
  behavior change - see below).
- cache-extraction -> Array4 block, verbatim twice (`model_eou.rs:101-142` == `model_nemotron.rs:185-225`),
  plus a third in multitalker: **net ~50-60 lines**.
- mel raw-log helper 3x + 4 constant blocks (A5-Q2): **~30-50 lines**, but this is M2b, explicitly NOT to
  change numerics, and the `(x+guard)` vs `(x.max(0.0)+guard)` divergence (`nemotron.rs:784` vs
  `multitalker.rs:675`) means collapsing forces choosing one - another behavior change, deferred to Wave V.

**Realistic behavior-safe total: ~350-550 net lines** (decoder step 150 + session 55 + find 50 + cache 55 +
mel 40 + argmax/misc 50). The 700-900 figure is the gross duplicated-text count; it over-states deletions by
~40-50% because it counts call sites and per-variant glue that must remain, and counts EOU/TDT loop bodies
that do not actually merge.

### Evidence "preserves behavior" is FALSE

A1-06 is a hard prerequisite, not a footnote. The four argmaxes have **three observably different
semantics** (verified in code):
- Nemotron `nemotron.rs:749-756`: `v > max_val`, no finite guard -> a leading `NaN` leaves `max_idx = 0`
  (NaN silently decodes token 0); ties = **first wins**.
- Unified `parakeet_unified.rs:477-482`: `max_by(partial_cmp.unwrap_or(Equal))` -> NaN treated as Equal,
  `max_by` keeps the **last** max-equal element; ties = **last wins** (opposite of Nemotron).
- EOU `parakeet_eou.rs:210-214`: `val.is_finite() && val > max_val` -> NaN skipped; ties = first wins.
- TDT `model_tdt.rs:231-236`: `max_by(...unwrap_or(Equal))`, last-wins, plus a separate duration argmax.

To put one `argmax(&[f32]) -> usize` (the recommended first-wins + finite guard, matching EOU/NeMo) behind
the shared backend, you **must overwrite** Nemotron's NaN-as-token-0 behavior and Unified's/TDT's last-wins
tie-break. On any input with a tie or a NaN logit, the output token sequence for Nemotron, Unified, and TDT
**changes**. It is a strict improvement (matches NeMo), and ties on real logits are rare - but the claim's word
"preserves" is literally false. Honest framing: M5 (argmax reconcile) is a **deliberate, small behavior change
gated by a per-variant regression test**, and it is a prerequisite the consolidation cannot skip. 80-feasibility
already says this (M5 is "correctness prereq for M2"); the CLAIM as worded contradicts its own dependency.

Same pattern for the divergent `find_encoder` candidate lists (02 sec 2b): unifying them means some variant
now accepts a filename it previously rejected (or vice-versa) - a small but real behavior change at load time.

**Claim (1) verdict: REFUTED as stated.** Extractable yes; "~700-900" is ~350-550 realistic; "preserves
behavior across all variants" is false because unification forces the argmax (and mel-guard, and
find_encoder) reconciliation, each of which changes at least one variant's output.

---

## CLAIM (2): "Consolidation is NON-BREAKING provided the leaked public model_X exports (lib.rs:87-91) are removed first (M8)."

### Verdict: **SURVIVES, with one correction and one caveat.**

### Are the model_X types actually leaked / only-internal? YES.
- `examples/` references to any of `ParakeetModel`, `ParakeetEOUModel`, `NemotronModel`,
  `NemotronEncoderCache`, `NemotronModelConfig`, `ParakeetUnifiedModel`, `UnifiedModelConfig`,
  `SentencePieceVocab`: **zero** (grep clean).
- `README.md` references: **zero**.
- They are exported at `lib.rs:87-91` but consumed only by their sibling wrapper modules within the crate:
  `parakeet.rs:6` (`use crate::model::ParakeetModel`), `parakeet_eou.rs:3`
  (`use crate::model_eou::{EncoderCache, ParakeetEOUModel}`), `parakeet_unified.rs:6`
  (`use crate::model_unified::{ParakeetUnifiedModel, UnifiedModelConfig}`).

So they look like accidental exports; removing the `pub use` lines is the only public-surface break, and it
unblocks making the modules `pub(crate)`. M8 -> M2-non-breaking logic holds.

### CORRECTION to the claim's scope: `SentencePieceVocab` must NOT simply be un-exported and locked to one module.
It is used **cross-module** by two other wrappers:
- `multitalker.rs:18,202,226` (`use crate::nemotron::SentencePieceVocab`).
- `parakeet_unified.rs:7,122,130,153` (`Arc<SentencePieceVocab>`).

Un-exporting it from `lib.rs:91` is fine (no external user needs it; grep clean), but the type itself must
remain at least `pub(crate)` and reachable cross-module - it cannot become private to `nemotron.rs`. The
claim lists it among "remove first" exports, which is correct for the *public* surface, but an implementer
who reads "remove" as "make private to nemotron" breaks the build (`multitalker`, `unified`). This is the M3
"move SentencePieceVocab to a `vocab/` home" item; it is a layering move, not a deletion. Caveat, not a refutation.

### Is the un-export the ONLY break? Essentially yes for M8 itself, but M2 is NOT non-breaking on its own.
- Removing `lib.rs:87-90` (`ParakeetModel`, `ParakeetEOUModel`, `NemotronModel`/`NemotronEncoderCache`/
  `NemotronModelConfig`, `ParakeetUnifiedModel`/`UnifiedModelConfig`) and the `SentencePieceVocab` re-export
  at `:91` = the single deliberate 0.x break. After that, an internal RnntBackend refactor touches no public
  type. Confirmed: the wrappers' public methods (`Parakeet`, `Nemotron`, etc.) are unaffected.
- BUT M2 still carries the **behavior change from Claim (1)** (argmax/find/mel reconciliation). "Non-breaking"
  is true only in the **API/semver** sense, not the **observable-output** sense. M8 makes M2 *API*-non-breaking;
  it does nothing about the (small, intended) transcript-output change M5 introduces. The two senses must not
  be conflated - 80-feasibility correctly separates M8 (API) from M5 (behavior) + M11 (golden tests).

**Claim (2) verdict: SURVIVES.** M8 genuinely converts M2 from API-breaking to API-non-breaking; the leaked
types are unused externally; the only correction is that `SentencePieceVocab` must stay `pub(crate)`
cross-module (move it, don't privatize it), and "non-breaking" means semver-non-breaking, not
output-identical.

---

## Is M2 one PR or several? SPLIT IT. (one PR is the wrong shape.)

M2 as "extract pub(crate) RnntBackend / src/onnx/" bundles independent risk profiles and a hard ordering
chain. Recommended split (each independently testable, each its own PR, gated by M11a golden tests):

1. **M8** - delete the `lib.rs:87-91` exports (the one API break). Tiny, lands first, independent.
2. **M5** - reconcile argmax into one `argmax(&[f32])` first-wins+finite helper. The ONLY intended
   behavior-changing PR; needs a per-variant token-sequence regression test FIRST (it changes Nemotron/
   Unified/TDT on NaN/tie). Do not bundle with structural moves or the diff hides the behavior change.
3. **M2a** - `load_session` + `find_first_existing` (session-build + find_encoder dedup). Pure move except the
   find_encoder candidate-list reconciliation (small load-time behavior change; note it).
4. **M2-decoder** - the parameterized `RnntDecoderStep` helper (the 4-knob struct) replacing the five
   run_decoder copies + the `extract_array{3,4}`/`rebuild_cache` helpers. Pure internal once M5 and M2a land.
5. **M2b** - shared mel front-end + single constants source, numerics UNCHANGED, behind a golden-mel fixture;
   the `(x.max(0.0)+guard)` vs `(x+guard)` divergence resolution deferred to Wave V.

Reasons to split: (a) M5 is the only behavior change and must be isolated for review/bisect; (b) M8 is the
only API change; (c) M2-decoder is large and mechanical; (d) M2b touches the most accuracy-sensitive code and
needs its own fixture gate; (e) a single mega-PR cannot be golden-tested incrementally and a regression
becomes un-bisectable. This matches 80-feasibility's own ordering (M5 -> M8 -> M2 -> M2b) - so the synthesis
should NOT present M2 as one task.

---

## Summary table

| Claim | Verdict | Realistic number / correction |
|-------|---------|-------------------------------|
| (1) ~700-900 lines removable | **REFUTED (upper half)** | ~350-550 net behavior-safe lines; 700-900 is gross dup, over-counts ~40-50% |
| (1) preserves behavior all variants | **REFUTED** | argmax (A1-06), find_encoder lists, mel guard each FORCE a small output change in >=1 variant |
| (2) M8 makes M2 non-breaking | **SURVIVES** | true in API/semver sense only; model_X types unused externally (examples+README grep clean) |
| (2) just "remove" the leaked exports | **CORRECTION** | `SentencePieceVocab` must stay `pub(crate)` cross-module (multitalker, unified use it) - move, don't privatize |
| M2 = one PR? | **SPLIT** | 5 PRs: M8, M5(behavior), M2a(session/find), M2-decoder, M2b(mel) - isolate the one API break and the one behavior change |

**Bottom line:** the consolidation is real and worth doing, but the plan must (a) quote ~350-550 lines not
700-900, (b) state plainly that unification reconciles (changes) argmax/find/mel behavior in at least one
variant rather than preserving it, (c) keep `SentencePieceVocab` `pub(crate)` not private, and (d) ship M2 as
~5 sequenced PRs with golden-test gates, not one.
