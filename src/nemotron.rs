use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_nemotron::{NemotronEncoderCache, NemotronModel};
use crate::vocab::SentencePieceVocab;
use ndarray::{s, Array2, Array3};
use std::path::Path;
use std::sync::{Arc, Mutex};

// Nemotron 0.6B model constants
// note that those numbers are coming from offical impl. and of course my onnx export decisions.
// Buffer logic and cache slicing strategy derived from:
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/utils/streaming_utils.py
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/audio_preprocessing.py
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

// Streaming chunk config (identical across English-only and multilingual variants:
// both use chunk_size_output=7 in NeMo's streaming_cfg which corresponds to 56 mel frames).
const CHUNK_SIZE: usize = 56;
const PRE_ENCODE_CACHE: usize = 9;

/// Language → prompt embedding index for the multilingual model. Mirrors
/// `cfg.model_defaults.prompt_dictionary` from the .nemo. Embedded here so
/// we don't require a sidecar `config.json` next to the ONNX files.
///
/// NVIDIA's model card documents 40 language-locales across 3 tiers:
///   - **Transcription-ready (19):** en, es, fr, it, pt, nl, de, tr, ru, ar,
///     hi, ja, ko, vi, uk (with locales).
///   - **Broad-coverage (13):** pl, sv, cs, nb, da, bg, fi, hr, sk, zh-CN,
///     hu, ro, et.
///   - **Adaptation-ready (8):** el, lt, lv, mt, sl, he, th, nn — recognized
///     by the tokenizer but need fine-tuning for production quality.
///
/// The full dictionary below contains additional entries because (a) several
/// codes alias the same prompt index (`en` == `en-US`, `hi` == `hi-IN`, ...)
/// and (b) some experimental languages (e.g. `qu-PE`, `mi-NZ`, `haw-US`)
/// have prompt slots but are not in the model card. Using those will work
/// but quality is not guaranteed.
///
/// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
const PROMPT_DICTIONARY: &[(&str, i64)] = &[
    ("af-ZA", 54), ("am-ET", 49), ("ar", 7), ("ar-AR", 7), ("auto", 101),
    ("ay-BO", 81), ("az-AZ", 66), ("bg", 30), ("bg-BG", 30), ("bn-IN", 36),
    ("cs", 22), ("cs-CZ", 22), ("da", 25), ("da-DK", 25), ("de", 9),
    ("de-DE", 9), ("el", 21), ("el-GR", 21), ("en", 0), ("en-GB", 1),
    ("en-US", 0), ("enGB", 1), ("es", 3), ("es-ES", 2), ("es-US", 3),
    ("esES", 2), ("et", 60), ("et-EE", 60), ("fa-IR", 38), ("fi", 26),
    ("fi-FI", 26), ("fr", 8), ("fr-CA", 100), ("fr-FR", 8), ("gn-PY", 82),
    ("gu-IN", 42), ("ha-NG", 50), ("haw-US", 97), ("he-IL", 64), ("hi", 6),
    ("hi-HI", 6), ("hi-IN", 6), ("hr", 29), ("hr-HR", 29), ("hu", 23),
    ("hu-HU", 23), ("hy-AM", 68), ("id-ID", 34), ("ig-NG", 53), ("it", 15),
    ("it-IT", 15), ("ja-JA", 10), ("ja-JP", 10), ("ka-GE", 67), ("km-KH", 47),
    ("kn-IN", 43), ("ko", 14), ("ko-KO", 14), ("ko-KR", 14), ("ku-TR", 65),
    ("ky-KG", 71), ("ln-CD", 58), ("lt", 31), ("lt-LT", 31), ("lv", 61),
    ("lv-LV", 61), ("mi-NZ", 96), ("ml-IN", 44), ("mr-IN", 41), ("ms-MY", 35),
    ("mt-MT", 102), ("nah-MX", 83), ("nb", 103), ("nb-NO", 103), ("ne-NP", 46),
    ("nl", 16), ("nl-NL", 16), ("nn", 104), ("nn-NO", 104), ("no", 27),
    ("no-NO", 27), ("ny-MW", 57), ("or-KE", 59), ("pl", 17), ("pl-PL", 17),
    ("pt", 13), ("pt-BR", 12), ("pt-PT", 13), ("qu-PE", 80), ("ro", 20),
    ("ro-RO", 20), ("ru", 11), ("ru-RU", 11), ("rw-RW", 55), ("si-LK", 45),
    ("sk", 28), ("sk-SK", 28), ("sl", 62), ("sl-SI", 62), ("sm-WS", 98),
    ("so-SO", 56), ("sv", 24), ("sv-SE", 24), ("sw-KE", 48), ("ta-IN", 39),
    ("te-IN", 40), ("tg-TJ", 70), ("th-TH", 32), ("to-TO", 99), ("tr", 18),
    ("tr-TR", 18), ("uk", 19), ("uk-UA", 19), ("ur-PK", 37), ("uz-UZ", 69),
    ("vi-VN", 33), ("yo-NG", 52), ("zh-CN", 4), ("zh-TW", 5), ("zh-ZH", 4),
    ("zu-ZA", 51),
];

/// Which Nemotron variant a handle wraps. Detected automatically from
/// the encoder ONNX graph (multilingual exposes a `prompt_index` input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronMode {
    /// English-only Nemotron 0.6B (vocab 1024, no language conditioning).
    EnglishOnly,
    /// Multilingual Nemotron 3.5 0.6B with `prompt_index` input,
    /// vocab ~13k, supports `target_lang` selection.
    Multilingual,
}

/// Shared handle to a loaded Nemotron model.
/// ONNX session is only loaded once and reference counted.
///
/// Use [`NemotronHandle::load`] to load from disk, then [`Nemotron::from_shared`]
/// to spawn each stream with its own independent decoder state.
/// Variant is auto-detected: both the en only 0.6B and the multi lang
/// 3.5 0.6B drop into the same type.
#[derive(Clone)]
pub struct NemotronHandle {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mel_basis: Arc<Array2<f32>>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    decoder_lstm_dim: usize,
    decoder_lstm_layers: usize,
    vocab_size: usize,
    blank_id: usize,
    /// Empty for en. populated for multilingual with all `<xx-XX>` token ids.
    lang_tag_ids: Arc<Vec<usize>>,
}

/// Nemotron streaming ASR model (0.6B parameters).
/// We dont apply mel normalization unlike others...
///
/// For a single stream, use [`Nemotron::from_pretrained`]. For multiple
/// concurrent streams (e.g. mic + system audio) sharing one loaded model,
/// use [`NemotronHandle::load`] followed by [`Nemotron::from_shared`].
///
/// For the multilingual variant call [`Nemotron::set_target_lang`] before
/// transcribing if you know the language; otherwise it defaults to `auto`
/// (prompt index 101) and lets the model pick.
pub struct Nemotron {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mel_basis: Arc<Array2<f32>>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    vocab_size: usize,
    blank_id: usize,
    lang_tag_ids: Arc<Vec<usize>>,
    encoder_cache: NemotronEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    /// `None` for English-only mode; `Some(idx)` for multilingual.
    prompt_index: Option<i64>,
    /// Raw audio sample buffer for proper mel computation
    audio_buffer: Vec<f32>,
    /// How many audio samples have been processed (converted to mel and sent to encoder)
    audio_processed: usize,
    chunk_idx: usize,
    accumulated_tokens: Vec<usize>,
}

impl NemotronHandle {
    /// Load the Nemotron model and vocabulary from a directory.
    ///
    /// Required files in `path`:
    /// - `encoder.onnx` + `encoder.onnx.data`
    /// - `decoder_joint.onnx`
    /// - `tokenizer.model`
    ///
    /// The returned handle is cheap to clone and can be used to spawn any
    /// number of [`Nemotron`] instances via [`Nemotron::from_shared`], each
    /// with its own independent decoder state.
    pub fn load<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let vocab_size = vocab.size();

        let exec = exec_config.unwrap_or_default();
        let model = NemotronModel::from_pretrained(path, exec, vocab_size)?;
        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        let mode = if model.has_prompt {
            NemotronMode::Multilingual
        } else {
            NemotronMode::EnglishOnly
        };
        let cfg = model.config.clone();
        let lang_tag_ids = if mode == NemotronMode::Multilingual {
            vocab.lang_tag_ids()
        } else {
            Vec::new()
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(vocab),
            mel_basis: Arc::new(mel_basis),
            mode,
            num_encoder_layers: cfg.num_encoder_layers,
            hidden_dim: cfg.hidden_dim,
            left_context: cfg.left_context,
            conv_context: cfg.conv_context,
            decoder_lstm_dim: cfg.decoder_lstm_dim,
            decoder_lstm_layers: cfg.decoder_lstm_layers,
            vocab_size: cfg.vocab_size,
            blank_id: cfg.blank_id,
            lang_tag_ids: Arc::new(lang_tag_ids),
        })
    }

    /// Which variant this handle wraps (auto detected at load time).
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Languages this model can transcribe, as accepted by
    /// [`Nemotron::set_target_lang`]. Empty for the English-only variant.
    pub fn available_languages(&self) -> Vec<&'static str> {
        match self.mode {
            NemotronMode::Multilingual => PROMPT_DICTIONARY.iter().map(|(k, _)| *k).collect(),
            NemotronMode::EnglishOnly => Vec::new(),
        }
    }
}

impl Nemotron {
    /// Load Nemotron from a directory and return a ready to use instance.
    /// Convenience wrapper for the single-stream case.
    ///
    /// For multiple concurrent streams sharing one loaded model, use
    /// [`NemotronHandle::load`] + [`Nemotron::from_shared`] instead.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Ok(Self::from_shared(&NemotronHandle::load(path, exec_config)?))
    }

    /// Spawn a new Nemotron instance bound to a shared model.
    ///
    /// Each instance owns independent decoder state (~7.5 MB) while the
    /// expensive ONNX session is shared through the handle.
    /// The model lock is held only during encoder/decoder inference
    /// (~20-50 ms per 560 ms audio chunk).
    ///
    /// For the multilingual variant the new instance defaults to `auto`
    /// (prompt index 101) — the model picks the language itself. Override
    /// via [`Self::set_target_lang`] when you know the language; that's
    /// strictly more accurate.
    pub fn from_shared(handle: &NemotronHandle) -> Self {
        let encoder_cache = NemotronEncoderCache::with_dims(
            handle.num_encoder_layers,
            handle.left_context,
            handle.hidden_dim,
            handle.conv_context,
        );

        let prompt_index = match handle.mode {
            NemotronMode::Multilingual => Some(101),
            NemotronMode::EnglishOnly => None,
        };

        Self {
            model: Arc::clone(&handle.model),
            vocab: Arc::clone(&handle.vocab),
            mel_basis: Arc::clone(&handle.mel_basis),
            mode: handle.mode,
            num_encoder_layers: handle.num_encoder_layers,
            hidden_dim: handle.hidden_dim,
            left_context: handle.left_context,
            conv_context: handle.conv_context,
            vocab_size: handle.vocab_size,
            blank_id: handle.blank_id,
            lang_tag_ids: Arc::clone(&handle.lang_tag_ids),
            encoder_cache,
            state_1: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            state_2: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            last_token: handle.blank_id as i32,
            prompt_index,
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            accumulated_tokens: Vec::new(),
        }
    }

    /// Which variant this instance wraps.
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Set the target language for the multilingual model. Accepts any key
    /// from [`NemotronHandle::available_languages`] (e.g. `"en-US"`, `"es-ES"`,
    /// `"ja-JP"`, `"auto"` for language-agnostic decoding).
    ///
    /// **Quality note:** NVIDIA's model card documents 40 language-locales
    /// across 3 tiers (transcription-ready, broad-coverage, adaptation-ready).
    /// Adaptation-ready locales need fine-tuning for production quality.
    /// The full prompt dictionary accepts additional codes (e.g. `qu-PE`,
    /// `mi-NZ`, `haw-US`) that the model has prompt slots for but are not
    /// in the model card — those will run, but accuracy is not guaranteed.
    /// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
    ///
    /// Returns an error on the English-only variant or for an unknown language.
    /// The new language takes effect on the next encoder call — for clean
    /// switching mid-utterance you usually also want [`Self::reset`].
    pub fn set_target_lang(&mut self, lang: &str) -> Result<()> {
        if self.mode != NemotronMode::Multilingual {
            return Err(Error::Config(
                "set_target_lang is only available on the multilingual variant".into(),
            ));
        }
        let idx = PROMPT_DICTIONARY
            .iter()
            .find_map(|(k, v)| (*k == lang).then_some(*v))
            .ok_or_else(|| {
                Error::Config(format!(
                    "Unknown target language '{lang}'. Try one of: en-US, es-ES, de-DE, fr-FR, ja-JP, zh-CN, auto, ..."
                ))
            })?;
        self.prompt_index = Some(idx);
        Ok(())
    }

    /// Reset all state for new utterance. Preserves the configured target
    /// language (call [`Self::set_target_lang`] again to change it).
    pub fn reset(&mut self) {
        self.encoder_cache = NemotronEncoderCache::with_dims(
            self.num_encoder_layers,
            self.left_context,
            self.hidden_dim,
            self.conv_context,
        );
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.accumulated_tokens.clear();
    }

    /// Get the full accumulated transcript. Language tag tokens (e.g. `<en-US>`)
    /// emitted by the multilingual model are stripped.
    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|t| *t < self.vocab_size && !self.lang_tag_ids.contains(t))
            .collect();
        self.vocab.decode(&valid)
    }

    /// note that, offline transcription for testing/debugging and for some curious ppl :-). with following function too (transcribe_audio)
    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;

        let audio = if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        };

        self.transcribe_audio(&audio)
    }

    /// Transcribe audio samples (non-streaming)
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        self.reset();

        let mel = self.compute_mel_spectrogram(audio)?;
        let total_frames = mel.shape()[1];

        if total_frames == 0 {
            return Ok(String::new());
        }

        let mut all_tokens: Vec<usize> = Vec::new();
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;

        while buffer_idx < total_frames {
            let chunk_end = (buffer_idx + CHUNK_SIZE).min(total_frames);
            let main_len = chunk_end - buffer_idx;

            let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
            let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

            // Fill pre-encode cache from previous frames
            if chunk_idx > 0 && buffer_idx >= PRE_ENCODE_CACHE {
                let cache_start = buffer_idx - PRE_ENCODE_CACHE;
                for f in 0..PRE_ENCODE_CACHE {
                    for m in 0..N_MELS {
                        chunk_data[m * expected_size + f] = mel[[m, cache_start + f]];
                    }
                }
            }

            // Fill main chunk
            for f in 0..main_len {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = mel[[m, buffer_idx + f]];
                }
            }

            let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
                .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

            let chunk_length = PRE_ENCODE_CACHE + main_len;

            let (encoded, enc_len, new_cache) = {
                let mut model = self.model.lock().map_err(|e| {
                    Error::Model(format!("Failed to acquire model lock: {e}"))
                })?;
                model.run_encoder(
                    &mel_chunk,
                    chunk_length as i64,
                    &self.encoder_cache,
                    self.prompt_index,
                )?
            };
            self.encoder_cache = new_cache;

            let new_tokens = self.decode_chunk(&encoded, enc_len as usize)?;
            all_tokens.extend(new_tokens);

            buffer_idx += CHUNK_SIZE;
            chunk_idx += 1;
        }

        let valid_tokens: Vec<usize> = all_tokens
            .into_iter()
            .filter(|t| *t < self.vocab_size && !self.lang_tag_ids.contains(t))
            .collect();

        Ok(self.vocab.decode(&valid_tokens))
    }

    /// Stream transcribe a chunk of audio (call repeatedly for real-time).
    ///
    /// This buffers raw audio and computes mel spectrograms over the full buffer
    /// to avoid edge effects at chunk boundaries.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        // Append raw audio to buffer
        self.audio_buffer.extend_from_slice(audio_chunk);

        // Calculate how many mel frames we can produce from buffered audio
        // mel_frames = 1 + (audio_len + 2*pad - win_length) / hop_length
        // For center=true padding, we need at least win_length samples to get 1 frame
        let total_audio = self.audio_buffer.len();
        if total_audio < WIN_LENGTH {
            return Ok(String::new());
        }

        // Compute mel spectrogram over the ENTIRE audio buffer
        let full_mel = self.compute_mel_spectrogram(&self.audio_buffer)?;
        let total_mel_frames = full_mel.shape()[1];

        // Calculate how many mel frames correspond to processed audio
        // Each CHUNK_SIZE mel frames = CHUNK_SIZE * HOP_LENGTH audio samples
        let processed_mel_frames = self.audio_processed / HOP_LENGTH;

        // Check if we have enough NEW frames to process a chunk
        let available_new_frames = total_mel_frames.saturating_sub(processed_mel_frames);
        if available_new_frames < CHUNK_SIZE {
            return Ok(String::new());
        }

        // Build encoder input chunk
        let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
        let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

        // Determine the mel frame range for this chunk
        let is_first_chunk = self.chunk_idx == 0;
        let main_start = processed_mel_frames;
        let _main_end = main_start + CHUNK_SIZE;

        if is_first_chunk {
            // First chunk: zero-pad for pre-encode cache
            for f in 0..CHUNK_SIZE.min(total_mel_frames) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            // Subsequent chunks: include pre-encode cache from previous frames
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;

            // Fill pre-encode cache
            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + cache_offset + f] =
                        full_mel[[m, cache_start + f]];
                }
            }

            // Fill main chunk
            for f in 0..CHUNK_SIZE.min(total_mel_frames - main_start) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] =
                        full_mel[[m, main_start + f]];
                }
            }
        }

        let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
            .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

        let (encoded, enc_len, new_cache) = {
            let mut model = self.model.lock().map_err(|e| {
                Error::Model(format!("Failed to acquire model lock: {e}"))
            })?;
            model.run_encoder(
                &mel_chunk,
                expected_size as i64,
                &self.encoder_cache,
                self.prompt_index,
            )?
        };
        self.encoder_cache = new_cache;

        let tokens = self.decode_chunk(&encoded, enc_len as usize)?;
        self.accumulated_tokens.extend(&tokens);

        // Advance processed position
        self.audio_processed += CHUNK_SIZE * HOP_LENGTH;
        self.chunk_idx += 1;

        // Trim audio buffer to keep memory bounded
        // Keep enough for pre-encode cache context
        let keep_samples = (PRE_ENCODE_CACHE + CHUNK_SIZE) * HOP_LENGTH + WIN_LENGTH;
        if self.audio_buffer.len() > keep_samples * 2 {
            let remove = self.audio_buffer.len() - keep_samples;
            // Adjust processed counter since we're removing from the start
            let actual_remove = remove.min(self.audio_processed);
            self.audio_buffer.drain(0..actual_remove);
            self.audio_processed -= actual_remove;
        }

        let mut result = String::new();
        for &t in &tokens {
            if t < self.vocab_size && !self.lang_tag_ids.contains(&t) {
                result.push_str(&self.vocab.decode_single(t));
            }
        }
        Ok(result)
    }

    fn decode_chunk(&mut self, encoder_out: &Array3<f32>, enc_frames: usize) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let max_symbols_per_step = 10;

        // Lock the model once for the entire decode loop to minimise
        // lock acquire/release overhead (many decoder steps per chunk).
        let mut model = self.model.lock().map_err(|e| {
            Error::Model(format!("Failed to acquire model lock: {e}"))
        })?;

        for t in 0..enc_frames {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame = frame
                .to_shape((1, hidden_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            for _ in 0..max_symbols_per_step {
                let (logits, new_state_1, new_state_2) = model.run_decoder(
                    &frame,
                    self.last_token,
                    &self.state_1,
                    &self.state_2,
                )?;

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }

                if max_idx == self.blank_id {
                    break;
                }

                tokens.push(max_idx);
                self.last_token = max_idx as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    /// Compute log mel spectrogram WITHOUT normalization.
    /// I use capitals because this gave me some trouble on the Python side :(). I realized they dont use it later.
    /// so offc nemo feeding raw log-mel spectrogram values (in decibels) directly to the encoder.
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        if audio.is_empty() {
            return Ok(Array2::zeros((N_MELS, 0)));
        }

        let preemph = crate::audio::apply_preemphasis(audio, PREEMPH);
        let spec = crate::audio::stft(&preemph, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel = self.mel_basis.dot(&spec);

        Ok(mel.mapv(|x| (x + LOG_ZERO_GUARD).ln()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- decode_chunk argmax (nemotron.rs:749-756): FIRST max wins on ties ---
    // Pins the CURRENT, deliberately-divergent argmax used in the Nemotron
    // decode loop (decoder.rs uses last-wins). Unifying the two is task T10;
    // this test only documents today's behavior so that change is a conscious
    // one. The argmax expression is replicated verbatim from decode_chunk
    // because the loop itself is wrapped around a model call and not callable
    // without weights.
    #[test]
    fn nemotron_argmax_is_first_wins_on_ties() {
        let logits = [0.1f32, 0.9, 0.2, 0.9]; // bins 1 and 3 tie at 0.9
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        assert_eq!(max_idx, 1, "nemotron loop must pick the FIRST tied index");
    }

    // --- Nemotron::reset() contract (nemotron.rs:495-509) ---
    // This is the language-lock surface. The CURRENT, documented contract is:
    // reset() clears decoder/encoder/audio state for a new utterance but
    // PRESERVES the configured target language (`prompt_index`). These tests
    // pin that contract as-is — they do NOT assert it is the desired behavior,
    // only that a refactor must not silently change which fields reset() touches.

    use crate::model_nemotron::{NemotronModel, NemotronModelConfig};

    /// Build a multilingual `Nemotron` backed by a tiny in-memory model so
    /// reset()'s pure state handling can be exercised with no model download.
    fn nemotron_for_reset_test() -> Nemotron {
        let cfg = NemotronModelConfig {
            num_encoder_layers: 2,
            hidden_dim: 4,
            left_context: 3,
            conv_context: 2,
            decoder_lstm_dim: 5,
            decoder_lstm_layers: 1,
            vocab_size: 16,
            blank_id: 15,
        };
        let model = NemotronModel::new_in_memory_for_test(cfg.clone(), true).unwrap();
        let encoder_cache = NemotronEncoderCache::with_dims(
            cfg.num_encoder_layers,
            cfg.left_context,
            cfg.hidden_dim,
            cfg.conv_context,
        );
        Nemotron {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(SentencePieceVocab { pieces: vec![] }),
            mel_basis: Arc::new(Array2::zeros((cfg.hidden_dim, 1))),
            mode: NemotronMode::Multilingual,
            num_encoder_layers: cfg.num_encoder_layers,
            hidden_dim: cfg.hidden_dim,
            left_context: cfg.left_context,
            conv_context: cfg.conv_context,
            vocab_size: cfg.vocab_size,
            blank_id: cfg.blank_id,
            lang_tag_ids: Arc::new(vec![]),
            encoder_cache,
            state_1: Array3::zeros((cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim)),
            state_2: Array3::zeros((cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim)),
            last_token: cfg.blank_id as i32,
            prompt_index: Some(101),
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            accumulated_tokens: Vec::new(),
        }
    }

    #[test]
    fn reset_clears_utterance_state() {
        let mut nem = nemotron_for_reset_test();

        // Dirty every field reset() is documented to clear.
        nem.state_1.fill(1.0);
        nem.state_2.fill(1.0);
        nem.last_token = 7;
        nem.audio_buffer = vec![0.5; 32];
        nem.audio_processed = 999;
        nem.chunk_idx = 4;
        nem.accumulated_tokens = vec![1, 2, 3];

        nem.reset();

        assert!(nem.state_1.iter().all(|&v| v == 0.0), "state_1 must be zeroed");
        assert!(nem.state_2.iter().all(|&v| v == 0.0), "state_2 must be zeroed");
        assert_eq!(nem.last_token, nem.blank_id as i32, "last_token -> blank");
        assert!(nem.audio_buffer.is_empty(), "audio_buffer must be cleared");
        assert_eq!(nem.audio_processed, 0, "audio_processed must reset");
        assert_eq!(nem.chunk_idx, 0, "chunk_idx must reset");
        assert!(nem.accumulated_tokens.is_empty(), "accumulated_tokens cleared");
    }

    #[test]
    fn reset_preserves_target_language() {
        // The language-lock contract: reset() must NOT clear prompt_index.
        let mut nem = nemotron_for_reset_test();
        nem.set_target_lang("es-ES").unwrap();
        let lang_before = nem.prompt_index;
        assert_eq!(lang_before, Some(2), "es-ES maps to prompt index 2");

        nem.reset();

        assert_eq!(
            nem.prompt_index, lang_before,
            "reset() must preserve the configured target language (current documented contract)"
        );
    }

    #[test]
    fn set_target_lang_rejects_unknown_and_english_only() {
        let mut multi = nemotron_for_reset_test();
        assert!(multi.set_target_lang("xx-ZZ").is_err(), "unknown lang -> Err");

        // English-only mode rejects set_target_lang outright.
        let mut eng = nemotron_for_reset_test();
        eng.mode = NemotronMode::EnglishOnly;
        assert!(
            eng.set_target_lang("en-US").is_err(),
            "set_target_lang must error on the English-only variant"
        );
    }
}
