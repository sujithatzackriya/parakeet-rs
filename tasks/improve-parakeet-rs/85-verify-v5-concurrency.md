# 85 — Verify V5 (Wave V): Concurrency / Mutex claim (F12 / A3-10)

**/think framework:** red-team (adversarial refutation). Default to REFUTED unless the code + the
pinned `ort` API positively prove the claim. Every assertion is cited to `file:line` in actual
source or to the vendored `ort 2.0.0-rc.12` crate in the local cargo registry.

---

## Claim under test

> "The `Arc<Mutex<NemotronModel>>` needlessly serializes concurrent shared-model streams because ort
> `Session::run` takes `&self`, so the Mutex could be removed/relaxed to allow true parallel inference
> across streams sharing one model, and the crate could offer spawn_blocking guidance for async
> runtimes."

This is the strong form of A2 finding **F12** and A3 finding **A3-10**'s perf cross-ref.

---

## VERDICT: **REFUTED** (as stated). The load-bearing premise is factually wrong for the pinned ort.

The claim rests on one factual premise: *"ort `Session::run` takes `&self`."* In the version this
crate actually depends on (`ort = 2.0.0-rc.12`, `Cargo.toml`), **that premise is false.** Every public
inference entry point on `ort::session::Session` takes `&mut self`:

- `pub fn run<...>(&'s mut self, ...)` — `ort-2.0.0-rc.12/src/session/mod.rs:212`
- `pub fn run_with_options<...>(&'s mut self, ...)` — `.../session/mod.rs:253-254`
- `pub fn run_async<...>(&'s mut self, ...)` — `.../session/mod.rs:407-408`
- `pub fn run_binding<...>(&'s mut self, ...)` — `.../session/mod.rs:340`

Only the **private** helpers `run_inner` (`.../session/mod.rs:272-273: &'s self`) and
`run_inner_async` (`.../session/mod.rs:426-427: &'s self`) take a shared `&self`. Those are
crate-private; a downstream consumer cannot call them. So at the public API surface a single `Session`
**cannot** be driven by two concurrent `&mut` borrows — the borrow checker forbids it. The claim's
mechanism ("just remove the Mutex and call `run(&self)` from N threads") does not compile against the
crate's actual dependency.

Because the central premise fails, the claim **as written is REFUTED.** The Mutex (or some equivalent
exclusion) is *required* to call `Session::run` at all from `&self`-shaped wrapper code, since `run`
demands `&mut Session`.

---

## What survives (the steel-manned, corrected version)

The *intent* behind the claim is partially sound, and the verifier must say so honestly. Three of the
four sub-conditions the claim needs are true; one is false and it is the decisive one.

### 1. Per-stream state is genuinely isolated — TRUE (claim's safety precondition holds)

The shared thing (`Arc<Mutex<NemotronModel>>`) wraps **only** the two ONNX `Session`s plus immutable
config:

- `NemotronModel { encoder: Session, decoder_joint: Session, config, has_prompt }`
  (`src/model_nemotron.rs:38-43`). No scratch buffers, no carried inference state.

All mutable per-chunk state lives on the per-stream `Nemotron` wrapper, not on the shared model:

- `encoder_cache: NemotronEncoderCache`, `state_1`, `state_2` (LSTM), `last_token`, `audio_buffer`,
  `audio_processed`, `chunk_idx`, `accumulated_tokens`, `prompt_index`
  (`src/nemotron.rs:312-323`).
- `from_shared` clones only the `Arc`s and gives each instance fresh per-stream state
  (`src/nemotron.rs:431-452`): `Arc::clone(&handle.model)` but brand-new `encoder_cache`, zeroed
  `state_1/2`, empty `audio_buffer`, etc.

So if ort *did* allow concurrent `&self` runs, two `Nemotron` instances over one handle would not
corrupt each other's decode state. The data-race safety argument is correct. This matches the F12
cross-ref note ("encoder cache + LSTM state are strictly per-instance"). **CONFIRMED.**

### 2. The wrapper's `&mut self` on `run_encoder`/`run_decoder` is not the obstacle — TRUE but moot

`NemotronModel::run_encoder` (`src/model_nemotron.rs:140-146`) and `run_decoder`
(`src/model_nemotron.rs:232-238`) take `&mut self`, but their bodies only call `self.encoder.run(...)`
(line 164) / `self.decoder_joint.run(...)` (line 243) and read `self.config`. They mutate no
session-level wrapper state. So the wrapper's `&mut self` is *conservative* — it could in principle be
`&self` **if** the underlying `Session::run` were `&self`. But it is not (see verdict), so relaxing the
wrapper to `&self` is impossible without changing how ort is called. The `&mut` propagates up *from
ort*, not down from a wrapper design choice.

### 3. `Session: Send + Sync` — TRUE

`unsafe impl Send for Session {}` / `unsafe impl Sync for Session {}`
(`ort-2.0.0-rc.12/src/session/mod.rs:675-676`); likewise `SharedSessionInner` (lines 84-85). So a
`Session` is `Sync` and can be *shared* across threads. **But `Sync` only grants `&Session` access
across threads, and `&Session` is useless for inference because `run` needs `&mut Session`.** `Sync`
without a `&self` run method does not enable parallel inference. This is the trap in the claim:
`Session: Sync` is true, "`run` takes `&self`" is false, and only the conjunction would let the Mutex
go away.

### 4. ORT's own threadsafety: even with `&self` it would not be free parallelism

Even setting Rust aside: the ONNX Runtime C API `OrtRun` is documented thread-safe on one
`OrtSession`, but concurrent calls share the session's intra-op thread pool. With the crate's default
`intra_threads = 4, inter_threads = 1` (`src/execution.rs:72-78`), two "parallel" streams would
oversubscribe cores and contend on the same pool, so wall-clock speedup is bounded and can regress
(A2 F7 already flags this). So the claim's "true parallel inference" overstates the achievable win
even in the hypothetical where the API allowed it.

---

## Why the Mutex is currently necessary (the refutation, concretely)

Given `run(&mut self)`:

- To call `run` from the `Nemotron` wrapper (which holds `Arc<...NemotronModel>`), the code needs
  `&mut NemotronModel` -> `&mut Session`. With `Arc` alone you cannot get `&mut` to shared data.
- `Mutex` (or `RwLock`, but a write lock is still exclusive) is the standard way to obtain that
  `&mut` safely. That is exactly what the code does:
  `self.model.lock()...; model.run_encoder(...)` (`src/nemotron.rs:584-592`, `683-693`) and the decode
  loop locks once and reuses it (`src/nemotron.rs:730-732, 742`).
- Therefore the Mutex is **not** "needlessly serializing" — it is the *minimum* synchronization that
  the `&mut self` signature forces. Removing it is not a relaxation; it is a type error.

A `RwLock` does not help: `run` needs `&mut`, so every reader would need a *write* lock anyway —
identical serialization, more overhead.

---

## The only paths to real parallel inference (all require more than "remove the Mutex")

The claim's "could be removed/relaxed" is achievable **only** by changing the sharing model, not by a
one-line lock removal:

1. **One `Session` per stream (no sharing of the session).** Give each `Nemotron` its own `Session`
   (drop the `Arc<Mutex<>>`, load or clone-commit per instance). Then each stream has its own
   `&mut Session` and runs truly in parallel. Cost: N x session memory and N x load time — defeats the
   `from_shared` design's whole purpose (amortizing the 2.3-2.4 GB model load, context-pack). This is
   the honest tradeoff, and it is a *different* design, not a Mutex removal.
2. **`run_async` + a shared executor.** `run_async` still takes `&mut self`
   (`.../session/mod.rs:407`), so it does **not** dodge the exclusivity at the Rust level; it only
   helps not block an async runtime thread while ORT computes. It is orthogonal to cross-stream
   parallelism.
3. **`unsafe` interior-mutability shim** to call the private-style `&self` FFI path: out of scope for
   a safe library and not exposed by ort. Reject.

So the correct roadmap item is **not** "remove the Mutex." It is: *document the serialization as
intended, and offer per-session instances as the opt-in parallelism path* (ties to A3-7's "is the
handle/shared pattern universal?" and A2 F7's thread-count interaction).

---

## On the `spawn_blocking` half of the claim — PARTIALLY SURVIVES, but insufficient alone

`transcribe_chunk` is synchronous and CPU-bound and holds the lock during inference
(`src/nemotron.rs:615, 683-693, 730-732`); there is no `tokio`/`spawn_blocking`/`rayon` anywhere in
`src/` or `examples/` (A2 F12 grep, re-confirmed by the context pack). So:

- **Guidance is warranted:** an async caller that calls `transcribe_chunk` directly on an executor
  thread will block it for the ~20-50 ms/chunk cited at `src/nemotron.rs:411-412`. Documenting
  `spawn_blocking` (tokio) / `block_in_place` / a dedicated thread is a real, additive doc improvement.
- **But `spawn_blocking` alone is NOT sufficient** and the claim under-specifies it:
  - Each `Nemotron` is `&mut self`-stateful, so a given stream must be pinned to one task; you cannot
    fan one stream's chunks across the blocking pool out of order (chunk order = decode-state order via
    `last_token`/`encoder_cache`). Needs a per-stream owned task or actor, not ad-hoc `spawn_blocking`.
  - tokio's default `spawn_blocking` pool is up to 512 threads — unbounded relative to cores. For a
    CPU-bound 4-intra-thread workload you need **bounded** concurrency (a semaphore or a sized pool)
    or you oversubscribe and regress (compounds A2 F7).
  - All those blocking tasks still serialize on the one `Mutex` during inference, so `spawn_blocking`
    keeps the executor responsive but does **not** deliver cross-stream parallelism. The claim conflates
    "don't block the async runtime" (true, fixable with docs) with "parallel inference" (false under
    this ort).

So: ship `spawn_blocking` *plus bounded-concurrency* guidance (SURVIVES as a doc/ergonomics item),
but do not present it as the concurrency-parallelism fix.

---

## Net assessment for the roadmap

- **REFUTE** the claim's headline mechanism: ort `Session::run` is `&mut self` in `2.0.0-rc.12`, so the
  Mutex cannot simply be removed/relaxed to allow concurrent `&self` runs. It is the minimum required
  synchronization, not a needless one. (`ort .../session/mod.rs:212`; `src/nemotron.rs:584-592`.)
- **KEEP** F12/A3-10 as a finding, but **rewrite its recommendation**:
  - Recommendation (a) spawn_blocking docs: **valid**, but extend to *bounded concurrency + per-stream
    task pinning*, and state it does not parallelize inference.
  - Recommendation (b) "drop `Arc<Mutex>` -> `&self` runs": **invalid as written** — replace with
    "offer an opt-in per-session instance path (one `Session` per stream) for callers who want true
    parallelism and can afford N x model memory; document that the shared handle intentionally
    serializes inference." Re-verify the memory/load tradeoff before recommending.
- **Severity:** keep at MED for the docs/ergonomics value; downgrade any claim of a "concurrency win
  from removing the Mutex" to **non-existent under this ort version**.

### Reverification hooks for downstream lanes
- Build check: attempt `model.encoder.run(...)` behind `&self` in a scratch branch — it will fail to
  compile (`&mut self` required), proving the Mutex is load-bearing. (Do NOT commit; this is a
  one-line type-check probe.)
- If ort is later bumped to a version that exposes a `&self` run, re-open this claim — the per-stream
  state isolation (point 1 above, `src/nemotron.rs:312-323`) already makes the safety side sound, so a
  future `&self` ort would flip this to SURVIVES.

---

## Citations

- `Cargo.toml` — `ort = 2.0.0-rc.12`, default-features=false.
- `ort-2.0.0-rc.12/src/session/mod.rs:212` — `pub fn run(&'s mut self, ...)`.
- `ort-2.0.0-rc.12/src/session/mod.rs:253-254` — `run_with_options(&'s mut self, ...)`.
- `ort-2.0.0-rc.12/src/session/mod.rs:340` — `run_binding(&'s mut self, ...)`.
- `ort-2.0.0-rc.12/src/session/mod.rs:407-408` — `run_async(&'s mut self, ...)`.
- `ort-2.0.0-rc.12/src/session/mod.rs:272-273, 426-427` — private `run_inner`/`run_inner_async(&'s self)`.
- `ort-2.0.0-rc.12/src/session/mod.rs:675-676` — `unsafe impl Send/Sync for Session`.
- `src/model_nemotron.rs:38-43` — `NemotronModel` holds only Sessions + config.
- `src/model_nemotron.rs:140-164, 232-249` — `run_encoder`/`run_decoder` take `&mut self`, only call
  `self.<session>.run(...)`, mutate no wrapper state.
- `src/nemotron.rs:312-323` — per-stream mutable state lives on `Nemotron`.
- `src/nemotron.rs:431-452` — `from_shared` clones `Arc`s, fresh per-stream state.
- `src/nemotron.rs:584-592, 683-693, 730-732, 742` — Mutex locked to obtain `&mut` for `run`.
- `src/nemotron.rs:411-412` — ~20-50 ms/chunk lock-hold note.
- `src/execution.rs:72-78` — default `intra_threads=4, inter_threads=1`.
