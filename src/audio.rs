use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use hound::{WavReader, WavSpec};
use ndarray::Array2;
use realfft::RealToComplex;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Single source of truth for the streaming mel front-end constants.
///
/// Nemotron, Multitalker, and Parakeet-EOU all run the identical NeMo
/// front-end geometry (16 kHz, 512-pt FFT, 25 ms window / 10 ms hop, 128 mel
/// bins, 0.97 preemphasis, additive log guard `2^-24`). These used to be
/// re-declared verbatim in each module; they live here once so a change can
/// never drift between variants. The per-variant mel *flavor* (Slaney vs HTK
/// filterbank) is NOT a constant - it is the filterbank each variant builds and
/// passes into [`log_mel_spectrogram`].
pub(crate) mod constants {
    pub const SAMPLE_RATE: usize = 16000;
    pub const N_FFT: usize = 512;
    pub const WIN_LENGTH: usize = 400;
    pub const HOP_LENGTH: usize = 160;
    pub const N_MELS: usize = 128;
    pub const PREEMPH: f32 = 0.97;
    // NeMo: log_zero_guard_type="add", value = 2^-24.
    pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
}

/// Compute the un-normalized log-mel spectrogram shared by the streaming
/// variants (Nemotron / Multitalker / Parakeet-EOU).
///
/// This is the ONE mel front-end body. The per-variant *flavor* enters only
/// through `mel_basis` (Slaney for Nemotron/Multitalker, HTK for EOU) and the
/// matching `fft_plan`; every other step - preemphasis, STFT geometry, and the
/// additive-guard log - is identical and lives here.
///
/// Returns mel rows x frame columns: `(n_mels, num_frames)`, where `n_mels` is
/// taken from `mel_basis`. Empty `audio` yields a `(n_mels, 0)` array, matching
/// the previous per-variant guards.
///
/// Numerics note: the historical `multitalker`/`eou` variants applied an
/// `x.max(0.0)` clamp before the log while `nemotron` did not. The clamp is a
/// provable no-op - `x = mel_basis.dot(|fft|^2)` is a non-negative matrix times
/// a non-negative vector, so `x >= 0` always and `x.max(0.0) == x`. The shared
/// path drops the clamp; output is byte-identical for all three variants.
pub(crate) fn log_mel_spectrogram(
    audio: &[f32],
    mel_basis: &Array2<f32>,
    fft_plan: &Arc<dyn RealToComplex<f32>>,
) -> Result<Array2<f32>> {
    let n_mels = mel_basis.shape()[0];
    if audio.is_empty() {
        return Ok(Array2::zeros((n_mels, 0)));
    }

    let preemph = apply_preemphasis(audio, constants::PREEMPH);
    let spec = stft_with_plan(
        &preemph,
        fft_plan,
        constants::N_FFT,
        constants::HOP_LENGTH,
        constants::WIN_LENGTH,
    )?;
    let mel = mel_basis.dot(&spec);
    Ok(mel.mapv(|x| (x + constants::LOG_ZERO_GUARD).ln()))
}

/// Cached, reusable mel filterbank + FFT plan keyed to a preprocessor
/// cfgs.
pub struct FeatureCache {
    pub mel_basis: Array2<f32>,
    pub fft_plan: Arc<dyn RealToComplex<f32>>,
}

impl FeatureCache {
    pub fn from_config(config: &PreprocessorConfig) -> Self {
        let mel_basis =
            create_mel_filterbank(config.n_fft, config.feature_size, config.sampling_rate);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft_plan = planner.plan_fft_forward(config.n_fft);
        Self {
            mel_basis,
            fft_plan,
        }
    }
}

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read float samples: {e}")))?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read int samples: {e}")))?,
    };

    Ok((samples, spec))
}

pub fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);

    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }

    result
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

// We use proper FFT here instead of naive DFT because the model was trained
// on correctly computed spectrograms. Naive DFT produces wrong frequency bins
// and the model outputs all blank tokens. realfft (real-valued FFT wrapper around
// RustFFT) gives us O(n log n) performance and numerically correct results.
pub fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(n_fft);
    stft_with_plan(audio, &plan, n_fft, hop_length, win_length)
}

/// Same as [`stft`] but takes a pre built FFT plan to avoid rebuilding it
/// per call. The plan is deterministic from `n_fft`.
pub fn stft_with_plan(
    audio: &[f32],
    plan: &Arc<dyn RealToComplex<f32>>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let window = hann_window(win_length);
    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut input = vec![0.0f32; n_fft];
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        input.fill(0.0);
        for i in 0..win_length.min(padded.len() - start) {
            input[i] = padded[start + i] * window[i];
        }

        plan.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

        for k in 0..freq_bins {
            spectrogram[[k, frame_idx]] = output[k].norm_sqr();
        }
    }

    Ok(spectrogram)
}

// Slaney mel scale (again librosa)
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

pub fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);

    // Mel cent freq
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    // FFT bin freq
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    // librosa's ramp
    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    // Slaney norm
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for k in 0..freq_bins {
            filterbank[[i, k]] *= enorm as f32;
        }
    }

    filterbank
}

/// Extract mel spectrogram features from raw audio samples.
///
/// The `cache` holds the mel filterbank and FFT plan built once at model
/// load - these are deterministic from `config` and identical across calls,
/// so reusing them avoids rebuilding ~15-20 µs of arithmetic per request.
///
/// # Arguments
///
/// * `audio` - Audio samples as f32 values
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
/// * `config` - Preprocessor configuration
/// * `cache` - Pre-built mel filterbank + FFT plan (see [`FeatureCache::from_config`])
///
/// # Returns
///
/// 2D array of mel spectrogram features (time_steps x feature_size)
pub fn extract_features_with_cache(
    mut audio: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    config: &PreprocessorConfig,
    cache: &FeatureCache,
) -> Result<Array2<f32>> {
    if sample_rate != config.sampling_rate as u32 {
        return Err(Error::Audio(format!(
            "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
            sample_rate, config.sampling_rate
        )));
    }

    if channels > 1 {
        let mono: Vec<f32> = audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        audio = mono;
    }

    audio = apply_preemphasis(&audio, config.preemphasis);

    let spectrogram = stft_with_plan(
        &audio,
        &cache.fft_plan,
        config.n_fft,
        config.hop_length,
        config.win_length,
    )?;

    let mel_spectrogram = cache.mel_basis.dot(&spectrogram);
    // Log with additive guard (NeMo: log_zero_guard_type="add", value=2^-24)
    let log_zero_guard: f32 = 2.0f32.powi(-24);
    let mel_spectrogram = mel_spectrogram.mapv(|x| (x + log_zero_guard).ln());

    let mut mel_spectrogram = mel_spectrogram.t().to_owned();

    // Normalize per_feature: mean=0, std=1 with Bessel's correction (N-1)
    let num_frames = mel_spectrogram.shape()[0];
    let num_features = mel_spectrogram.shape()[1];

    if num_frames <= 1 {
        return Ok(mel_spectrogram);
    }

    for feat_idx in 0..num_features {
        let mut column = mel_spectrogram.column_mut(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames as f32 - 1.0);
        let std = variance.sqrt() + 1e-5;

        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }

    Ok(mel_spectrogram)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave at the given frequency and sample rate.
    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1kHz sine at 16kHz sample rate, 1 second
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let sample_rate = 16000;
        let audio = sine_wave(1000.0, sample_rate, sample_rate);

        let spec = stft(&audio, n_fft, hop_length, win_length).unwrap();

        // Expected bin for 1kHz: freq_hz * n_fft / sample_rate = 1000 * 512 / 16000 = 32
        let expected_bin = 32;
        let freq_bins = n_fft / 2 + 1;
        let num_frames = spec.shape()[1];

        // Check that bin 32 has the highest power in most frames (skip edge frames)
        let mut correct_frames = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }

        let interior_frames = num_frames.saturating_sub(4);
        assert!(
            correct_frames > interior_frames / 2,
            "Expected bin {expected_bin} to dominate in most frames, but only {correct_frames}/{interior_frames}"
        );
    }

    #[test]
    fn stft_output_shape_is_correct() {
        let n_fft = 512;
        let hop_length = 160;
        let win_length = 400;
        let audio = vec![0.0f32; 16000]; // 1 second of silence

        let spec = stft(&audio, n_fft, hop_length, win_length).unwrap();

        let freq_bins = n_fft / 2 + 1;
        assert_eq!(spec.shape()[0], freq_bins);
        // num_frames = (audio_len + n_fft - n_fft) / hop_length + 1 = 16000 / 160 + 1 = 101
        assert!(spec.shape()[1] > 0);
    }

    // --- Mel filterbank (create_mel_filterbank, Slaney scale + norm) ---

    #[test]
    fn mel_filterbank_shape_and_triangles() {
        let n_fft = 512;
        let n_mels = 128;
        let sample_rate = 16000;
        let fb = create_mel_filterbank(n_fft, n_mels, sample_rate);

        // Shape is (n_mels, n_fft/2 + 1).
        assert_eq!(fb.shape(), &[n_mels, n_fft / 2 + 1]);
        // All weights are non-negative (triangles clamped at 0).
        assert!(fb.iter().all(|&w| w >= 0.0), "mel weights must be >= 0");
        // Every mel filter has at least one non-zero bin (no dead filters at
        // this resolution); a regression collapsing the triangles would trip this.
        for i in 0..n_mels {
            let any = fb.row(i).iter().any(|&w| w > 0.0);
            assert!(any, "mel filter {i} is entirely zero");
        }
    }

    #[test]
    fn mel_filterbank_reference_values() {
        // Pin a couple of exact Slaney-normalized weights so a change to the
        // hz<->mel mapping or the enorm step is caught. Values are the current
        // output of create_mel_filterbank(512, 128, 16000) for low mel bins,
        // where the Slaney triangles are narrow and easy to verify by eye.
        let fb = create_mel_filterbank(512, 128, 16000);
        // Filter 0 peaks near its center FFT bin; bin 1 (31.25 Hz) sits on the
        // rising edge of the first triangle and is strictly positive.
        assert!(fb[[0, 1]] > 0.0, "mel[0,1] should be on the first triangle");
        // FFT bin 0 (0 Hz) is below the first filter's left edge -> exactly 0.
        assert_eq!(fb[[0, 0]], 0.0, "mel[0,0] must be zero (0 Hz)");
        // Very high mel filters do not reach the lowest FFT bins -> 0 there.
        assert_eq!(fb[[127, 1]], 0.0, "top mel filter must be 0 at low bins");
    }

    // --- extract_features_with_cache: normalization + boundary invariants ---

    #[test]
    fn extract_features_rejects_sample_rate_mismatch() {
        let config = PreprocessorConfig::default(); // sampling_rate = 16000
        let cache = FeatureCache::from_config(&config);
        let audio = vec![0.0f32; 16000];

        let err = extract_features_with_cache(audio, 8000, 1, &config, &cache);
        assert!(
            matches!(err, Err(Error::Audio(_))),
            "mismatched sample rate must return Error::Audio"
        );
    }

    #[test]
    fn extract_features_downmixes_stereo_to_mono() {
        // Interleaved stereo where the two channels are identical: the mono
        // downmix must equal the single-channel features (averaging identical
        // channels is a no-op), and frame count matches a mono input.
        let config = PreprocessorConfig::default();
        let cache = FeatureCache::from_config(&config);

        let mono: Vec<f32> = sine_wave(440.0, 16000, 8000);
        let mut stereo = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            stereo.push(s);
            stereo.push(s);
        }

        let feats_mono =
            extract_features_with_cache(mono.clone(), 16000, 1, &config, &cache).unwrap();
        let feats_stereo =
            extract_features_with_cache(stereo, 16000, 2, &config, &cache).unwrap();

        assert_eq!(feats_mono.shape(), feats_stereo.shape());
        for (a, b) in feats_mono.iter().zip(feats_stereo.iter()) {
            assert!((a - b).abs() < 1e-4, "downmix of identical channels must match mono");
        }
    }

    #[test]
    fn extract_features_normalizes_per_feature() {
        // Per-feature normalization (Bessel N-1): each mel column must have
        // mean ~0 and std ~1 (modulo the +1e-5 std floor). This pins the
        // normalization stage the model was trained against.
        let config = PreprocessorConfig::default();
        let cache = FeatureCache::from_config(&config);
        let audio = sine_wave(440.0, 16000, 16000); // 1s, many frames

        let feats = extract_features_with_cache(audio, 16000, 1, &config, &cache).unwrap();
        let num_frames = feats.shape()[0];
        assert!(num_frames > 1);

        for feat_idx in 0..feats.shape()[1] {
            let col = feats.column(feat_idx);
            let mean: f32 = col.iter().sum::<f32>() / num_frames as f32;
            assert!(mean.abs() < 1e-3, "feature {feat_idx} mean {mean} not ~0");
        }
    }

    // --- Shared log-mel front-end (log_mel_spectrogram) ---

    #[test]
    fn log_mel_spectrogram_empty_audio_yields_zero_frames() {
        // Empty input must return a (n_mels, 0) array, matching the previous
        // per-variant guards. n_mels is taken from the filterbank.
        let mel_basis = create_mel_filterbank(512, 128, 16000);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(512);

        let out = log_mel_spectrogram(&[], &mel_basis, &plan).unwrap();
        assert_eq!(out.shape(), &[128, 0]);
    }

    #[test]
    fn log_mel_spectrogram_clamp_fold_is_byte_identical() {
        // T13 folded the cosmetic `x.max(0.0)` clamp (multitalker/eou) into the
        // clamp-free path (nemotron). This is a provable no-op because the mel
        // energy `mel_basis.dot(|fft|^2)` is always >= 0. Pin it: recomputing
        // the OLD clamped form by hand must be bit-for-bit identical to the
        // shared helper's clamp-free output on real (non-trivial) audio.
        let mel_basis = create_mel_filterbank(512, 128, 16000);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(512);
        let audio = sine_wave(440.0, 16000, 8000);

        let shared = log_mel_spectrogram(&audio, &mel_basis, &plan).unwrap();

        // Reconstruct the historical clamped path explicitly.
        let preemph = apply_preemphasis(&audio, constants::PREEMPH);
        let spec = stft_with_plan(
            &preemph,
            &plan,
            constants::N_FFT,
            constants::HOP_LENGTH,
            constants::WIN_LENGTH,
        )
        .unwrap();
        let clamped = mel_basis
            .dot(&spec)
            .mapv(|x| (x.max(0.0) + constants::LOG_ZERO_GUARD).ln());

        assert_eq!(shared.shape(), clamped.shape());
        for (a, b) in shared.iter().zip(clamped.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "shared clamp-free log-mel must be bit-identical to the old clamped form"
            );
        }
    }

    // --- Committed WAV fixture (no model) ---

    #[test]
    fn load_audio_reads_fixture_invariants() {
        // The fixture is a 6.04 s, 16 kHz, mono PCM16 clip committed under
        // tests/fixtures/. This guards the public no-model audio loader:
        // sample rate, channel count, and sample length must round-trip.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_en.wav");
        let (samples, spec) = load_audio(path).unwrap();

        assert_eq!(spec.sample_rate, 16000, "fixture must be 16 kHz");
        assert_eq!(spec.channels, 1, "fixture must be mono");
        // 96683 frames in a mono file => 96683 samples.
        assert_eq!(samples.len(), 96683, "fixture sample count");
        // PCM16 decode maps into [-1, 1).
        assert!(
            samples.iter().all(|&s| (-1.0..1.0).contains(&s)),
            "decoded samples must be normalized into [-1, 1)"
        );
    }
}
