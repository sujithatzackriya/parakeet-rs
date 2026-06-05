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

use parakeet_rs::{Nemotron, NemotronMode};
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

/// Drive the streaming path exactly like `examples/streaming.rs`:
/// 8960-sample chunks (last one zero-padded) + 3x zero-chunk flush, then read
/// the accumulated transcript.
fn stream_transcript(model: &mut Nemotron, audio: &[f32]) -> String {
    for chunk in audio.chunks(STREAM_CHUNK) {
        let chunk_vec = if chunk.len() < STREAM_CHUNK {
            let mut p = chunk.to_vec();
            p.resize(STREAM_CHUNK, 0.0);
            p
        } else {
            chunk.to_vec()
        };
        model.transcribe_chunk(&chunk_vec).expect("transcribe_chunk");
    }
    // 3x zero-chunk flush (drains the decoder tail).
    for _ in 0..3 {
        model
            .transcribe_chunk(&vec![0.0; STREAM_CHUNK])
            .expect("flush chunk");
    }
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

/// Word-overlap ratio (Jaccard-ish, multiset intersection over the larger set)
/// of two normalized transcripts. 1.0 == same words.
fn word_overlap(a: &str, b: &str) -> f64 {
    let aw: Vec<&str> = a.split_whitespace().collect();
    let bw: Vec<&str> = b.split_whitespace().collect();
    if aw.is_empty() && bw.is_empty() {
        return 1.0;
    }
    let mut bw_pool = bw.clone();
    let mut hits = 0usize;
    for w in &aw {
        if let Some(pos) = bw_pool.iter().position(|x| x == w) {
            bw_pool.remove(pos);
            hits += 1;
        }
    }
    hits as f64 / aw.len().max(bw.len()) as f64
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
// TEST 2 — STREAMING == OFFLINE EQUIVALENCE (EN), with a documented tolerance.
//
// Streaming currently (Wave V / M6) drops the final partial chunk and uses a
// different length convention, so the streaming transcript is NOT byte-equal
// to the offline one yet. We assert a TOLERANCE (high word-overlap) instead of
// equality. This tightens to byte-equal after T09 (flush). DO NOT fix flush
// here.
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

    let overlap = word_overlap(&normalize(&offline), &normalize(&streaming));
    eprintln!("word_overlap = {overlap:.3}");

    // TOLERANCE: 0.85 word-overlap. Chosen because streaming may lose the final
    // ~560ms tail (a word or two) until T09 lands flush(); 0.85 catches a real
    // regression (garbled/empty streaming) while tolerating that known tail
    // loss. After T09 this assertion should be replaced by byte-equality.
    assert!(
        overlap >= 0.85,
        "streaming diverged from offline beyond tolerance (overlap {overlap:.3} < 0.85)\n\
         offline={offline:?}\nstreaming={streaming:?}"
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
