//! Criterion benchmark harness for Nemotron ASR latency / RTF.
//!
//! This is the acceptance instrument for Epic P (PRD G2: "RTF < 1.0 on CPU").
//! It measures two things against the local English Nemotron model:
//!
//!   1. `streaming_chunk` - per-560ms-chunk `transcribe_chunk` latency, with the
//!      decoder state reset before each sample so we capture steady-state
//!      single-chunk cost (not the F1 O(n^2) growing-buffer effect).
//!   2. `offline_rtf` - wall time of a full `transcribe_audio` over a fixture,
//!      with the RTF (audio_seconds / wall_seconds) printed for the run.
//!
//! Correctness is NOT the goal here - these benches only time the hot path.
//!
//! The harness SKIPS gracefully (prints a notice and registers no work) when the
//! model directory is absent, so it never breaks `cargo build` / CI on machines
//! without the ~2.4 GB model on disk.

use criterion::{criterion_group, criterion_main, Criterion};
use parakeet_rs::Nemotron;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

/// Local English Nemotron model dir (symlinked into the worktree).
const MODEL_DIR: &str = "./nemotron";
/// 560ms @ 16kHz - the native Nemotron streaming chunk size.
const CHUNK_SAMPLES: usize = 8960;
/// Sample rate the model expects.
const SAMPLE_RATE: f32 = 16000.0;
const FIXTURE: &str = "tests/fixtures/test_en.wav";

/// True only when every file `Nemotron::from_pretrained` needs is present.
fn model_available() -> bool {
    let dir = Path::new(MODEL_DIR);
    dir.is_dir()
        && ["encoder.onnx", "decoder_joint.onnx", "tokenizer.model"]
            .iter()
            .all(|f| dir.join(f).exists())
}

/// Load + normalize a mono 16kHz f32 fixture; returns `None` if the file is
/// missing so the bench can skip instead of panicking.
fn load_fixture(path: &str) -> Option<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return None;
    }
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>().ok()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()
            .ok()?,
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    Some(audio)
}

fn bench_streaming_chunk(c: &mut Criterion) {
    if !model_available() {
        eprintln!("[bench] SKIP streaming_chunk: model dir '{MODEL_DIR}' absent");
        return;
    }
    let mut model = match Nemotron::from_pretrained(MODEL_DIR, None) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[bench] SKIP streaming_chunk: load failed: {e}");
            return;
        }
    };
    // A single deterministic 560ms chunk. Reset the model before each iteration
    // so we time steady-state single-chunk cost, not a growing buffer.
    let chunk = vec![0.01f32; CHUNK_SAMPLES];

    let mut group = c.benchmark_group("streaming_chunk");
    group.bench_function("transcribe_chunk_560ms", |b| {
        // iter_custom lets us reset the decoder state OUTSIDE the timed region,
        // so each measured chunk is steady-state (constant buffer) work.
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                model.reset();
                let t = Instant::now();
                let _ = black_box(model.transcribe_chunk(black_box(&chunk)));
                total += t.elapsed();
            }
            total
        });
    });
    group.finish();
}

fn bench_offline_rtf(c: &mut Criterion) {
    if !model_available() {
        eprintln!("[bench] SKIP offline_rtf: model dir '{MODEL_DIR}' absent");
        return;
    }
    let audio = match load_fixture(FIXTURE) {
        Some(a) => a,
        None => {
            eprintln!("[bench] SKIP offline_rtf: fixture '{FIXTURE}' missing/unreadable");
            return;
        }
    };
    let mut model = match Nemotron::from_pretrained(MODEL_DIR, None) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[bench] SKIP offline_rtf: load failed: {e}");
            return;
        }
    };
    let audio_secs = audio.len() as f32 / SAMPLE_RATE;

    // Print the RTF once outside the timing loop (criterion times wall, not RTF).
    let t0 = Instant::now();
    model.reset();
    let _ = model.transcribe_audio(&audio);
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "[bench] offline_rtf: audio={audio_secs:.2}s wall={wall:.2}s RTF={:.3}x (<1.0 = realtime)",
        audio_secs / wall.max(1e-6)
    );

    let mut group = c.benchmark_group("offline_rtf");
    group.sample_size(10);
    group.bench_function("transcribe_audio_test_en", |b| {
        b.iter(|| {
            model.reset();
            let _ = black_box(model.transcribe_audio(black_box(&audio)));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_streaming_chunk, bench_offline_rtf);
criterion_main!(benches);
