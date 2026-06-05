/*
Live microphone streaming ASR (real-time, cache-aware stateful Nemotron).

Captures the default input device, downmixes to mono, resamples to 16kHz, and
feeds 560ms chunks to the Nemotron streaming model as you speak.

Usage (args in any order):
  cargo run --release --example streaming_mic                 # English-only, Enter to stop
  cargo run --release --example streaming_mic <lang>          # Multilingual, Enter to stop
  cargo run --release --example streaming_mic <secs>          # English-only, auto-stop after <secs>
  cargo run --release --example streaming_mic <lang> <secs>   # Multilingual, auto-stop after <secs>

Examples:
  cargo run --release --example streaming_mic auto 10         # multilingual, auto lang, 10s capture
  cargo run --release --example streaming_mic ja-JP           # multilingual, Japanese, Enter to stop
  cargo run --release --example streaming_mic 8               # English, 8s capture

A non-numeric arg selects the multilingual model in ./nemotron_multi and is the
target language code (`auto`, `en-US`, `ja-JP`, ... see src/nemotron.rs). A numeric
arg sets a capture duration in seconds (auto-stop; needed for non-interactive runs
where stdin has no Enter). With no language arg, ./nemotron (English-only) loads.

Stop: press Enter (interactive), or it auto-stops at the duration, or Ctrl+C.
The decoder is then flushed and the full transcript printed.

Note: this is a demo. It uses a simple linear resampler (the model is robust to
it). For production-grade capture/resampling see the host app's audio pipeline.
*/

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use parakeet_rs::{Nemotron, NemotronMode};
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

const TARGET_RATE: f32 = 16000.0;
const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Args (any order): a numeric arg = capture duration in seconds (auto-stop,
    // for non-interactive runs); any non-numeric arg = target language code
    // (selects the multilingual model). With no duration, stop by pressing Enter.
    let mut target_lang: Option<String> = None;
    let mut duration_secs: Option<u64> = None;
    for a in std::env::args().skip(1) {
        match a.parse::<u64>() {
            Ok(n) => duration_secs = Some(n),
            Err(_) => target_lang = Some(a),
        }
    }

    // Load model (multilingual if a language arg is given, else English-only).
    let model_dir = if target_lang.is_some() {
        "./nemotron_multi"
    } else {
        "./nemotron"
    };
    let mut model = Nemotron::from_pretrained(model_dir, None)?;
    match model.mode() {
        NemotronMode::Multilingual => {
            let lang = target_lang.as_deref().unwrap_or("auto");
            model.set_target_lang(lang)?;
            println!("[multilingual model, target_lang={lang}]");
        }
        NemotronMode::EnglishOnly => {
            if let Some(lang) = &target_lang {
                eprintln!("Warning: target_lang='{lang}' ignored, English-only model loaded");
            }
        }
    }

    // Open the default input device at its native config.
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no default input device available")?;
    let supported = device.default_input_config()?;
    let native_rate = supported.sample_rate().0 as f32;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.config();
    println!(
        "Input: {} ({} Hz, {} ch, {:?}) -> resampling to {} Hz mono",
        device.name().unwrap_or_else(|_| "unknown".into()),
        native_rate as u32,
        channels,
        sample_format,
        TARGET_RATE as u32
    );

    // Audio callback pushes mono frames (native rate) over a channel.
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(&device, &stream_config, channels, tx),
        cpal::SampleFormat::I16 => build_input_stream::<i16>(&device, &stream_config, channels, tx),
        cpal::SampleFormat::U16 => build_input_stream::<u16>(&device, &stream_config, channels, tx),
        cpal::SampleFormat::I32 => build_input_stream::<i32>(&device, &stream_config, channels, tx),
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    }?;
    stream.play()?;

    // Stop condition. Explicit duration wins. Otherwise, if stdin is NOT a
    // terminal (run via a non-interactive harness), default to a timed capture
    // so the program neither exits instantly on stdin EOF nor hangs forever.
    // Only a real interactive terminal uses Enter-to-stop.
    let interactive = std::io::stdin().is_terminal();
    let capture_secs: Option<u64> = duration_secs.or(if interactive { None } else { Some(30) });

    let stop = Arc::new(AtomicBool::new(false));
    let deadline = capture_secs.map(|n| Instant::now() + Duration::from_secs(n));
    if interactive && duration_secs.is_none() {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            stop.store(true, Ordering::SeqCst);
        });
    }

    match capture_secs {
        Some(n) => println!("\nListening... capturing for {n}s (speak now).\n"),
        None => println!("\nListening... speak now. Press Enter to stop.\n"),
    }
    print!("Streaming: ");
    std::io::stdout().flush()?;

    // Diagnostics: confirm the mic is actually delivering audio.
    let mut total_native: usize = 0;
    let mut peak: f32 = 0.0;

    // step = native input samples consumed per output (16kHz) sample.
    let step = native_rate / TARGET_RATE;
    let mut native_buf: Vec<f32> = Vec::new();
    let mut pos: f32 = 0.0;
    let mut out16k: Vec<f32> = Vec::new();

    while !stop.load(Ordering::SeqCst) {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        // Drain whatever the audio thread captured.
        let mut got = false;
        while let Ok(block) = rx.try_recv() {
            total_native += block.len();
            for &s in &block {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
            native_buf.extend(block);
            got = true;
        }
        if got {
            resample_into(&mut native_buf, &mut pos, step, &mut out16k);
            while out16k.len() >= CHUNK_SIZE {
                let chunk: Vec<f32> = out16k.drain(..CHUNK_SIZE).collect();
                let text = model.transcribe_chunk(&chunk)?;
                if !text.is_empty() {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    drop(stream); // stop capture

    // Feed any remaining partial chunk (zero-padded), then flush the decoder.
    if !out16k.is_empty() {
        out16k.resize(CHUNK_SIZE, 0.0);
        let text = model.transcribe_chunk(&out16k)?;
        if !text.is_empty() {
            print!("{text}");
        }
    }
    for _ in 0..3 {
        let text = model.transcribe_chunk(&vec![0.0; CHUNK_SIZE])?;
        if !text.is_empty() {
            print!("{text}");
        }
    }

    eprintln!(
        "\n[diagnostics] captured {} samples (~{:.1}s @ {} Hz), peak amplitude {:.4}{}",
        total_native,
        total_native as f32 / native_rate,
        native_rate as u32,
        peak,
        if peak < 1e-4 {
            "  <- near-silence: mic not delivering audio (permission? wrong device?)"
        } else {
            ""
        }
    );
    println!("\nFinal: {}", model.get_transcript());
    Ok(())
}

/// Build a cpal input stream for sample type `T`, downmixing to mono `f32` and
/// forwarding each callback's samples over `tx`.
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_fn = |e| eprintln!("input stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|frame| {
                    let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                    sum / channels as f32
                })
                .collect();
            let _ = tx.send(mono);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/// Linear-resample `native_buf` into `out` using a fractional read position.
/// Consumes the samples it used, keeping the tail needed for interpolation so
/// the resampler is stateful across callbacks.
fn resample_into(native_buf: &mut Vec<f32>, pos: &mut f32, step: f32, out: &mut Vec<f32>) {
    while (*pos as usize) + 1 < native_buf.len() {
        let i = *pos as usize;
        let frac = *pos - i as f32;
        out.push(native_buf[i] * (1.0 - frac) + native_buf[i + 1] * frac);
        *pos += step;
    }
    // `pos` may overshoot past the end of the buffer (it advances by `step`
    // beyond the last produced sample); clamp so we never drain more than we
    // have. The leftover fractional `pos` carries correctly into the next block.
    let consumed = (*pos as usize).min(native_buf.len());
    if consumed > 0 {
        native_buf.drain(..consumed);
        *pos -= consumed as f32;
    }
}
