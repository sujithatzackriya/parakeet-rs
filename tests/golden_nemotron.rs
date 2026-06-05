//! Tier-2 golden harness: drives the REAL local Nemotron models.
//!
//! These tests need the 2.3-2.4 GB ONNX weights that do NOT live in the repo,
//! so they are gated TWO ways and stay out of normal CI:
//!   1. every test is `#[ignore]` (skipped by a plain `cargo test`), and
//!   2. each test returns early (with an eprintln) when its model dir is
//!      absent, so even `cargo test -- --ignored` is green on a machine with
//!      no models.
//!
//! Run locally with the models present (`./nemotron`, `./nemotron_multi`):
//!   cargo test --test golden_nemotron -- --ignored --nocapture
//!
//! Audio fixture: `tests/fixtures/test_en.wav` (6s, 16kHz mono English),
//! committed via the `!tests/fixtures/*.wav` .gitignore override.

use parakeet_rs::{Nemotron, NemotronMode, TimestampMode};
use std::path::Path;

const EN_MODEL_DIR: &str = "./nemotron";
const MULTI_MODEL_DIR: &str = "./nemotron_multi";
const EN_FIXTURE: &str = "tests/fixtures/test_en.wav";
const CODESWITCH_FIXTURE: &str = "tests/fixtures/test_codeswitch_en_es.wav";

/// Nemotron streaming chunk = 8960 samples (560ms @ 16kHz), per the
/// `examples/streaming.rs` / `examples/streaming_mic.rs` chunk loop.
const STREAM_CHUNK: usize = 8960;

// --- GOLDEN TRANSCRIPTS (captured from the real local models) -------------
// Byte-stable snapshots. These are the per-variant guards the later refactors
// (T07/T12/T13) lean on: any change that perturbs the transcript flips these.
// Re-capture with `--nocapture` (the tests print "ACTUAL: ...") only when a
// model/decode change is intentional.

/// `./nemotron` (English-only) offline transcript of `test_en.wav`.
const GOLDEN_EN_OFFLINE: &str =
    "The quick brown fox jumps over the lazy dog. Streaming speech recognition is working correctly.";

/// `./nemotron_multi` (multilingual 3.5) offline transcript of `test_en.wav`
/// with `set_target_lang("en-US")`. Note: differs from the English-only golden
/// in casing/punctuation (no comma, trailing space) — the multilingual vocab
/// renders this fixture slightly differently. Snapshot is exact, byte-for-byte.
const GOLDEN_MULTI_EN_OFFLINE: &str =
    "The quick brown fox jumps over the lazy dog streaming speech recognition is working correctly. ";

// --------------------------------------------------------------------------

fn model_present(dir: &str) -> bool {
    Path::new(dir).join("encoder.onnx").exists()
}

/// Load a WAV the same way the streaming examples do: hound, i16 -> f32/32768,
/// mono. NO peak normalization, so the offline and streaming paths in the
/// equivalence test see byte-identical samples (normalization would otherwise
/// be a confounder between the two paths).
fn load_wav_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open fixture wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "fixture must be 16kHz");
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    audio
}

/// Drive the streaming path: feed `audio` as 8960-sample chunks WITHOUT
/// zero-padding the final partial chunk (the genuine tail is left buffered),
/// then call `flush()` to drain it. This is the T09 contract — `flush()`
/// replaces the old "zero-pad the last chunk + 3x zero-chunk" incantation.
fn stream_transcript(model: &mut Nemotron, audio: &[f32]) -> String {
    for chunk in audio.chunks(STREAM_CHUNK) {
        model.transcribe_chunk(chunk).expect("transcribe_chunk");
    }
    model.flush().expect("flush");
    model.get_transcript()
}

/// Normalize a transcript for tolerance comparison: lowercase, strip
/// punctuation, collapse whitespace. Used where byte-equality is not yet
/// guaranteed (streaming flush / multilingual casing).
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ===========================================================================
// TEST 1 — OFFLINE GOLDEN (EN): per-variant byte-stable guard.
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron weights; run with --ignored"]
fn offline_golden_english() {
    if !model_present(EN_MODEL_DIR) {
        eprintln!("SKIP offline_golden_english: {EN_MODEL_DIR} not present");
        return;
    }
    let mut model = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load ./nemotron");
    assert_eq!(model.mode(), NemotronMode::EnglishOnly);

    let transcript = model.transcribe_file(EN_FIXTURE).expect("transcribe_file");
    eprintln!("ACTUAL en offline: {transcript:?}");

    assert_eq!(
        transcript, GOLDEN_EN_OFFLINE,
        "English offline transcript drifted from the golden snapshot"
    );
}

// ===========================================================================
// TEST 2 — STREAMING == OFFLINE EQUIVALENCE (EN), byte-equal.
//
// Tightened in T09. With flush() draining the final partial chunk and the
// streaming encoder length reconciled to the offline convention
// (PRE_ENCODE_CACHE + real frames), `transcribe_chunk* + flush` produces a
// BYTE-IDENTICAL transcript to `transcribe_audio` on the same samples. Offline
// is now a valid oracle for streaming. Verified on ./nemotron + test_en.wav:
// both render
//   "...lazy dog. Streaming speech recognition is working correctly."
// (the trailing '.' that the pre-T09 path dropped is now captured).
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron weights; run with --ignored"]
fn streaming_matches_offline_english() {
    if !model_present(EN_MODEL_DIR) {
        eprintln!("SKIP streaming_matches_offline_english: {EN_MODEL_DIR} not present");
        return;
    }
    // Same samples to BOTH paths (no normalization confounder).
    let audio = load_wav_mono(EN_FIXTURE);

    let mut offline_model = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load offline");
    let offline = offline_model.transcribe_audio(&audio).expect("offline");

    let mut stream_model = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load stream");
    let streaming = stream_transcript(&mut stream_model, &audio);

    eprintln!("OFFLINE  : {offline:?}");
    eprintln!("STREAMING: {streaming:?}");

    // Byte-equality: streaming (chunks + flush) MUST equal offline on identical
    // samples. Any drift (a dropped tail word/punctuation, a divergent length
    // convention, or a flush regression) flips this.
    assert_eq!(
        streaming, offline,
        "streaming (transcribe_chunk* + flush) diverged from offline transcribe_audio\n\
         offline={offline:?}\nstreaming={streaming:?}"
    );
}

// ===========================================================================
// TEST 2b — flush() IDEMPOTENCY + TAIL CAPTURE (T09).
//
// Feeds full chunks, leaving a genuine sub-chunk tail buffered, then asserts:
//   (1) flush() emits NON-EMPTY text (the dropped-tail bug, A1-02, would emit
//       "" and lose the final word/punctuation), and
//   (2) a SECOND flush() emits "" and leaves get_transcript() unchanged — no
//       double-emit, the processed cursor cannot desync.
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron weights; run with --ignored"]
fn flush_captures_tail_and_is_idempotent() {
    if !model_present(EN_MODEL_DIR) {
        eprintln!("SKIP flush_captures_tail_and_is_idempotent: {EN_MODEL_DIR} not present");
        return;
    }
    let audio = load_wav_mono(EN_FIXTURE);
    // The fixture length is not a multiple of STREAM_CHUNK, so the final
    // chunk is a partial tail that transcribe_chunk leaves buffered.
    assert_ne!(audio.len() % STREAM_CHUNK, 0, "fixture must have a partial tail");

    let mut model = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load ./nemotron");
    for chunk in audio.chunks(STREAM_CHUNK) {
        model.transcribe_chunk(chunk).expect("transcribe_chunk");
    }

    let first = model.flush().expect("first flush");
    let after_first = model.get_transcript();
    eprintln!("first flush emitted: {first:?}");

    let second = model.flush().expect("second flush");
    let after_second = model.get_transcript();
    eprintln!("second flush emitted: {second:?}");

    assert!(
        !first.trim().is_empty(),
        "flush() must capture the buffered tail (got empty) — dropped-final-chunk regression"
    );
    assert_eq!(
        second, "",
        "second flush() double-emitted {second:?} — cursor desync / not idempotent"
    );
    assert_eq!(
        after_first, after_second,
        "second flush() mutated the transcript: {after_first:?} -> {after_second:?}"
    );
}

// ===========================================================================
// TEST 3 — MULTILINGUAL OFFLINE GOLDEN: ./nemotron_multi + set_target_lang.
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron_multi weights; run with --ignored"]
fn offline_golden_multilingual_en() {
    if !model_present(MULTI_MODEL_DIR) {
        eprintln!("SKIP offline_golden_multilingual_en: {MULTI_MODEL_DIR} not present");
        return;
    }
    let mut model =
        Nemotron::from_pretrained(MULTI_MODEL_DIR, None).expect("load ./nemotron_multi");
    assert_eq!(model.mode(), NemotronMode::Multilingual);
    model.set_target_lang("en-US").expect("set en-US");

    let transcript = model.transcribe_file(EN_FIXTURE).expect("transcribe_file");
    eprintln!("ACTUAL multi en offline: {transcript:?}");

    assert_eq!(
        transcript, GOLDEN_MULTI_EN_OFFLINE,
        "Multilingual(en-US) offline transcript drifted from the golden snapshot"
    );
}

// ===========================================================================
// TEST 4 — RESET CONTRACT (behavioral, with model).
//
// Proves reset() actually clears streaming state at runtime: streaming the full
// fixture, then reset(), then streaming a short slice must produce the SAME
// output as a FRESH instance streaming that same slice. If carried encoder/
// decoder state leaked past reset(), the two would differ.
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron weights; run with --ignored"]
fn reset_clears_streaming_state() {
    if !model_present(EN_MODEL_DIR) {
        eprintln!("SKIP reset_clears_streaming_state: {EN_MODEL_DIR} not present");
        return;
    }
    let audio = load_wav_mono(EN_FIXTURE);
    // A short slice (~2 chunks worth) used for the post-reset comparison.
    let slice = &audio[..(STREAM_CHUNK * 2).min(audio.len())];

    // Instance A: stream the FULL fixture, then reset, then stream the slice.
    let mut used = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load A");
    let _ = stream_transcript(&mut used, &audio);
    used.reset();
    let after_reset = stream_transcript(&mut used, slice);

    // Instance B: a FRESH instance streaming the same slice.
    let mut fresh = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load B");
    let fresh_out = stream_transcript(&mut fresh, slice);

    eprintln!("after_reset: {after_reset:?}");
    eprintln!("fresh_out  : {fresh_out:?}");

    assert_eq!(
        after_reset, fresh_out,
        "reset() did not fully clear streaming state: post-reset output differs from a fresh instance"
    );
}

// ===========================================================================
// TEST 6 — detected_language() under `auto` (T05).
//
// Drives ./nemotron_multi on the English fixture with target_lang="auto" and
// observes detected_language(). Read-only: this only reads back the model's
// emitted <lang> tag, it does NOT switch language.
//
// NOTE: this is an OBSERVATION test, not a hard assertion. This checkpoint does
// not emit an inline <lang> tag for a short, clean, monolingual-English clip
// (verified: the 6s `test_en.wav` yields a correct transcript but no tag, so
// detected_language() is None here). Tag emission is sentence-boundary /
// code-switch driven and not guaranteed for every utterance, so asserting
// `Some(_)` on this fixture would be flaky. The pure id->code mapping that
// detected_language() is built on is fully covered model-free in
// `src/vocab.rs` (`language_from_tokens_*`). When a tag IS present we assert it
// is well-formed (an `xx` / `xx-XX` code); otherwise we only log.
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron_multi weights; run with --ignored"]
fn multilingual_auto_detected_language_observation() {
    if !model_present(MULTI_MODEL_DIR) {
        eprintln!(
            "SKIP multilingual_auto_detected_language_observation: {MULTI_MODEL_DIR} not present"
        );
        return;
    }
    let mut model =
        Nemotron::from_pretrained(MULTI_MODEL_DIR, None).expect("load ./nemotron_multi");
    model.set_target_lang("auto").expect("set auto");

    let audio = load_wav_mono(EN_FIXTURE);
    let _ = stream_transcript(&mut model, &audio);

    let detected = model.detected_language();
    eprintln!("detected_language (auto, en fixture): {detected:?}");

    // Read-only invariant: IF a code is surfaced it must be well-formed
    // (2-letter lang, optionally `-` + 2-letter region). No tag (None) is a
    // valid outcome for a short monolingual clip and is not a failure.
    if let Some(code) = detected {
        let ok = matches!(code.len(), 2 | 5)
            && code.chars().next().is_some_and(|c| c.is_ascii_lowercase());
        assert!(ok, "detected language code is malformed: {code:?}");
    }
}

// ===========================================================================
// TEST 5 — CODE-SWITCH ACCEPTANCE TEST (the T14 gate).
//
// ACCEPTANCE TEST for T14 (reset_with_lang / re-prompt). Currently FAILS =
// reproduces the language-lock. T14 makes it pass.
//
// Fixture: a `say` + `afconvert` generated WAV — an English sentence
// (dominant lead-in) followed by the Spanish sentence
//   "Hola, buenos días, muchas gracias, hace mucho calor hoy por la mañana."
//
// Under target_lang="auto", a correct code-switch would render that Spanish
// half in proper Spanish orthography. Today it does NOT: the model locks to the
// dominant (English) language and PHONETICALLY TRANSLITERATES the Spanish into
// garbled English-script — e.g. the observed output renders
//   hola -> "Oh last", días -> "Dius", hace -> "Ase", calor -> "Kalarhoi",
//   "por la" -> "Purla", mañana -> "Manyana".
// (A few words that happen to round-trip phonetically — "muchas", "gracias",
// "mucho" — DO survive; that is exactly why the gate below counts the
// accent/function words the lock corrupts, not just "any Spanish word".)
//
// This test encodes the DESIRED behavior and is expected to FAIL until T14.
// ===========================================================================
#[test]
#[ignore = "ACCEPTANCE for T14: currently FAILS (reproduces the language-lock). \
            T14 (reset_with_lang/re-prompt) makes it pass. Run with --ignored."]
fn multilingual_auto_code_switch_acceptance() {
    if !model_present(MULTI_MODEL_DIR) {
        eprintln!("SKIP multilingual_auto_code_switch_acceptance: {MULTI_MODEL_DIR} not present");
        return;
    }
    if !Path::new(CODESWITCH_FIXTURE).exists() {
        eprintln!("SKIP multilingual_auto_code_switch_acceptance: {CODESWITCH_FIXTURE} missing");
        return;
    }
    let audio = load_wav_mono(CODESWITCH_FIXTURE);

    let mut model = Nemotron::from_pretrained(MULTI_MODEL_DIR, None).expect("load multi");
    model.set_target_lang("auto").expect("set auto");
    let transcript = stream_transcript(&mut model, &audio);
    eprintln!("code-switch (auto) transcript: {transcript:?}");

    // The accent/function words the lock corrupts today (see comment above).
    // A correct Spanish render (post-T14) reproduces these exactly; today the
    // phonetic-transliteration lock mangles them, so few/none appear.
    let norm = normalize(&transcript);
    let words: Vec<&str> = norm.split_whitespace().collect();
    let spanish_markers = [
        "hola", "buenos", "días", "dias", "hace", "calor", "mañana", "manana",
    ];
    let found: Vec<&str> = spanish_markers
        .iter()
        .copied()
        .filter(|m| words.contains(m))
        .collect();
    eprintln!("spanish accent/function markers found: {found:?}");

    // GATE: require >= 3 of the corrupted-today markers. Today the lock yields
    // 0 (they are transliterated), so this FAILS = reproduces the lock. After
    // T14 re-prompts the Spanish half, the proper orthography returns and this
    // PASSES. Threshold 3 (not 1) so the few accidental phonetic round-trips
    // ("muchas"/"gracias"/"mucho", deliberately excluded above) can't mask a
    // still-locked model.
    assert!(
        found.len() >= 3,
        "DESIRED (T14): code-switch audio under auto must render the Spanish half \
         in proper Spanish, but only {}/{} accent/function markers survived ({:?}) \
         — this reproduces the language-lock (phonetic transliteration). \
         Transcript: {transcript:?}",
        found.len(),
        spanish_markers.len(),
        found,
    );
}

// ===========================================================================
// TEST 7 — WORD-LEVEL TIMESTAMPS (EN), the T16 gate.
//
// Streams ./nemotron over test_en.wav, then calls the additive
// get_timed_transcript(TimestampMode::Words) and asserts:
//   (1) words are non-empty,
//   (2) word start/end timestamps are monotonically non-decreasing and within
//       the audio duration (~6s, with a small encoder-frame slack), and
//   (3) the concatenated word text matches get_transcript() modulo spacing/
//       punctuation (the plain-text path is unchanged; the timed view is built
//       from the SAME accumulated tokens).
// ===========================================================================
#[test]
#[ignore = "needs ./nemotron weights; run with --ignored"]
fn word_timestamps_english() {
    if !model_present(EN_MODEL_DIR) {
        eprintln!("SKIP word_timestamps_english: {EN_MODEL_DIR} not present");
        return;
    }
    let audio = load_wav_mono(EN_FIXTURE);
    let audio_secs = audio.len() as f32 / 16000.0;

    let mut model = Nemotron::from_pretrained(EN_MODEL_DIR, None).expect("load ./nemotron");
    assert_eq!(model.mode(), NemotronMode::EnglishOnly);

    let _ = stream_transcript(&mut model, &audio);
    let result = model.get_timed_transcript(TimestampMode::Words);
    let plain = model.get_transcript();

    eprintln!("audio_secs={audio_secs:.2}");
    eprintln!("plain     : {plain:?}");
    for w in &result.tokens {
        eprintln!("  word {:?} [{:.2}, {:.2}]", w.text, w.start, w.end);
    }

    // (1) Non-empty words.
    assert!(!result.tokens.is_empty(), "expected non-empty word timestamps");

    // (2) Monotonic non-decreasing starts/ends, each within the audio duration.
    // Allow one encoder frame (80 ms) of slack on the upper bound: the final
    // token's `end` is its frame + 1.
    let slack = 0.1_f32;
    let mut prev_start = 0.0_f32;
    let mut prev_end = 0.0_f32;
    for w in &result.tokens {
        assert!(w.end >= w.start, "word end < start: {w:?}");
        assert!(
            w.start >= prev_start - 1e-4,
            "word starts not non-decreasing: {} < {}",
            w.start,
            prev_start
        );
        assert!(
            w.end >= prev_end - 1e-4,
            "word ends not non-decreasing: {} < {}",
            w.end,
            prev_end
        );
        assert!(
            w.start >= 0.0 && w.end <= audio_secs + slack,
            "word timestamp out of audio range [0, {:.2}]: {w:?}",
            audio_secs
        );
        prev_start = w.start;
        prev_end = w.end;
    }

    // (3) Concatenated word text == plain transcript, modulo spacing/punctuation.
    let joined = result
        .tokens
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(
        normalize(&joined),
        normalize(&plain),
        "timed word text diverged from plain transcript\njoined={joined:?}\nplain={plain:?}"
    );
}
