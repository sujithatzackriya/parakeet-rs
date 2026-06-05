//! Cohere Transcribe ASR engine.
//!
//! 2B parameter Conformer encoder + lightweight Transformer decoder.
//! Takes raw 16 kHz mono f32 audio, returns transcribed text.
//! Supports 14 languages via explicit language selection.
//!
//! Consumes a HuggingFace-standard ONNX export: the encoder takes
//! pre-computed log-mel features and the decoder is a merged graph using
//! the standard `past_key_values.N.{decoder,encoder}.{key,value}` cache
//! convention.
//!
//! [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
//! is one such export (FP32, FP16, INT8, and INT4 variants available).
//! To produce your own from the upstream PyTorch checkpoint, install
//! [`optimum`](https://github.com/huggingface/optimum) and run:
//!
//! ```sh
//! optimum-cli export onnx \
//!     --model CohereLabs/cohere-transcribe-03-2026 \
//!     --task automatic-speech-recognition-with-past \
//!     ./cohere-onnx
//! ```
//!
//! No custom export script is needed - the `cohere_asr` model type is
//! supported by Optimum's standard exporter.

use crate::audio::{extract_features_with_cache, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_cohere::{CohereEncoderOutput, CohereModel, CoherePastKv, N_MELS};
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// Special token literals that drive the decoder prompt. The canonical
// prompt structure is produced by CohereAsrProcessor in transformers and
// matches the shape model.generate() expects:
//
//   [▁, <|startofcontext|>, <|startoftranscript|>, <|emo:undefined|>,
//    <|src_lang|>, <|tgt_lang|>, <|pnc|>/<|nopnc|>, <|itn|>/<|noitn|>,
//    <|notimestamp|>, <|nodiarize|>]
//
// `▁` (SentencePiece word boundary) is the model's
// `decoder_start_token_id`. Source and target languages are both the
// same ISO code for pure transcription. Emotion and diarisation are
// fixed to "undefined"/"no" since the model's ASR task does not use
// them.
const TOKEN_WORD_BOUNDARY: &str = "\u{2581}";
const TOKEN_STARTOFCONTEXT: &str = "<|startofcontext|>";
const TOKEN_STARTOFTRANSCRIPT: &str = "<|startoftranscript|>";
const TOKEN_EMO_UNDEFINED: &str = "<|emo:undefined|>";
const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";
const TOKEN_PNC: &str = "<|pnc|>";
const TOKEN_NOPNC: &str = "<|nopnc|>";
const TOKEN_TIMESTAMP: &str = "<|timestamp|>";
const TOKEN_NOTIMESTAMP: &str = "<|notimestamp|>";
const TOKEN_DIARIZE: &str = "<|diarize|>";
const TOKEN_NODIARIZE: &str = "<|nodiarize|>";
const TOKEN_ITN: &str = "<|itn|>";
const TOKEN_NOITN: &str = "<|noitn|>";

/// Hard upper bound on output tokens enforced by the model
/// (`max_position_embeddings = 1024`). The user-configurable
/// `max_decode_tokens` cannot exceed this.
const MAX_DECODE_TOKENS_LIMIT: usize = 1024;

/// Default maximum output tokens per transcription. 512 is enough for
/// ~40 seconds of typical speech at the model's tokenisation rate, which
/// covers the training range (`max_audio_clip_s = 35`).
const DEFAULT_MAX_DECODE_TOKENS: usize = 512;

/// Training chunk length recorded in `preprocessor_config.json`
/// (`max_audio_clip_s`). This is *not* a runtime limit — the official model
/// card lists long-form transcription as a supported feature and audio well
/// past this length transcribes fine. Exposed via
/// [`CohereASR::training_chunk_secs`] only as informational metadata :-)
const TRAINING_CHUNK_SECS: f32 = 35.0;

// The 14 languages officially supported by Cohere Transcribe
// (cohere-transcribe-03-2026). The tokenizer contains `<|xx|>` placeholders
// for ~180 ISO codes but only these have trained weights.
// See https://docs.cohere.com/docs/transcribe.
const SUPPORTED_LANGUAGES: &[&str] = &[
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
];


/// Optional generation toggles for the Cohere decoder prompt.
///
/// The model emits timestamp and speaker-diarization markers as prompt
/// choices in the same slot family as `pnc`/`itn`. Both default to `false`,
/// which selects the `<|notimestamp|>`/`<|nodiarize|>` tokens and reproduces
/// the engine's original behaviour byte-for-byte.
///
/// ```
/// # #[cfg(feature = "cohere")] {
/// use parakeet_rs::CohereOptions;
/// let opts = CohereOptions::default().with_timestamps(true);
/// assert!(opts.timestamps);
/// assert!(!opts.diarize);
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CohereOptions {
    /// Emit `<|timestamp|>` instead of `<|notimestamp|>` so the model
    /// produces timestamp markers in its output. Defaults to `false`.
    pub timestamps: bool,
    /// Emit `<|diarize|>` instead of `<|nodiarize|>` so the model produces
    /// speaker-diarization markers in its output. Defaults to `false`.
    pub diarize: bool,
}

impl CohereOptions {
    /// Set whether the model emits timestamp markers.
    pub fn with_timestamps(mut self, timestamps: bool) -> Self {
        self.timestamps = timestamps;
        self
    }

    /// Set whether the model emits speaker-diarization markers.
    pub fn with_diarize(mut self, diarize: bool) -> Self {
        self.diarize = diarize;
        self
    }
}

struct DecoderTokens {
    decoder_start: i64,
    startofcontext: i64,
    sot: i64,
    emo_undefined: i64,
    eos: i64,
    pnc: i64,
    nopnc: i64,
    timestamp: i64,
    notimestamp: i64,
    diarize: i64,
    nodiarize: i64,
    itn: i64,
    noitn: i64,
}

impl DecoderTokens {
    /// Build the canonical Cohere decoder prompt for one transcription.
    ///
    /// Mirrors what `CohereAsrProcessor` in transformers produces. The source
    /// and target language tokens are both `lang_token` since this is pure
    /// transcription (no translation). The `pnc`/`itn`/timestamp/diarize slots
    /// select their positive or negative variant from `punctuation`, `itn`,
    /// and `options`. With `CohereOptions::default()` this yields the original
    /// `<|notimestamp|><|nodiarize|>` sequence.
    fn build_prompt(
        &self,
        lang_token: i64,
        punctuation: bool,
        itn: bool,
        options: CohereOptions,
    ) -> Vec<i64> {
        let pnc_token = if punctuation { self.pnc } else { self.nopnc };
        let itn_token = if itn { self.itn } else { self.noitn };
        let timestamp_token = if options.timestamps {
            self.timestamp
        } else {
            self.notimestamp
        };
        let diarize_token = if options.diarize {
            self.diarize
        } else {
            self.nodiarize
        };
        vec![
            self.decoder_start,
            self.startofcontext,
            self.sot,
            self.emo_undefined,
            lang_token,
            lang_token,
            pnc_token,
            itn_token,
            timestamp_token,
            diarize_token,
        ]
    }

    fn resolve(tokenizer: &Tokenizer) -> Result<Self> {
        Ok(Self {
            decoder_start: require_token(tokenizer, TOKEN_WORD_BOUNDARY)?,
            startofcontext: require_token(tokenizer, TOKEN_STARTOFCONTEXT)?,
            sot: require_token(tokenizer, TOKEN_STARTOFTRANSCRIPT)?,
            emo_undefined: require_token(tokenizer, TOKEN_EMO_UNDEFINED)?,
            eos: require_token(tokenizer, TOKEN_ENDOFTEXT)?,
            pnc: require_token(tokenizer, TOKEN_PNC)?,
            nopnc: require_token(tokenizer, TOKEN_NOPNC)?,
            timestamp: require_token(tokenizer, TOKEN_TIMESTAMP)?,
            notimestamp: require_token(tokenizer, TOKEN_NOTIMESTAMP)?,
            diarize: require_token(tokenizer, TOKEN_DIARIZE)?,
            nodiarize: require_token(tokenizer, TOKEN_NODIARIZE)?,
            itn: require_token(tokenizer, TOKEN_ITN)?,
            noitn: require_token(tokenizer, TOKEN_NOITN)?,
        })
    }
}

/// Values on here are mirror from `preprocessor_config.json` in the upstream HF export. they are baked
/// into the encoder's ONNX graph (feature_size=128, hop=160, etc.) and
/// are not user tunable, so we hardcode them rather than requiring the
/// file on disk. If they share onnx script, we could consider something just like we did for the sorftformer.
fn cohere_preprocessor_config() -> PreprocessorConfig {
    PreprocessorConfig {
        feature_extractor_type: "CohereAsrFeatureExtractor".to_string(),
        feature_size: N_MELS,
        hop_length: 160,
        n_fft: 512,
        padding_side: "right".to_string(),
        padding_value: 0.0,
        preemphasis: 0.97,
        processor_class: "CohereAsrProcessor".to_string(),
        return_attention_mask: true,
        sampling_rate: 16000,
        win_length: 400,
    }
}

/// Cohere Transcribe ASR engine.
pub struct CohereASR {
    model: CohereModel,
    tokenizer: Tokenizer,
    /// Mel/STFT parameters (hardcoded — see [`cohere_preprocessor_config`]).
    preprocessor: PreprocessorConfig,
    /// Pre-built mel filterbank + FFT plan ->> reused across every transcribe call.
    feature_cache: FeatureCache,
    /// Map of supported ISO 639-1 language code -> language token id.
    lang_tokens: HashMap<String, i64>,
    tokens: DecoderTokens,
    /// Maximum number of tokens to generate per `transcribe_audio` call.
    /// Defaults to [`DEFAULT_MAX_DECODE_TOKENS`] (512). Capped at
    /// [`MAX_DECODE_TOKENS_LIMIT`] (1024).
    max_decode_tokens: usize,
}

impl CohereASR {
    /// Load the Cohere Transcribe model from a directory.
    ///
    /// The directory must contain (flat or under `onnx/`):
    /// - one of `encoder_model[_quantized|_fp16].onnx` (+ `.onnx_data` companions)
    /// - one of `decoder_model_merged[_quantized|_fp16].onnx` (+ `.onnx_data`)
    /// - `tokenizer.json`
    ///
    /// parameters are hardcoded since they are fixed by the encoder graph.
    ///
    /// This layout matches the [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
    /// HF repository.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let model = CohereModel::from_pretrained(model_dir, exec)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokenizer.json in {}",
                model_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let preprocessor = cohere_preprocessor_config();
        let feature_cache = FeatureCache::from_config(&preprocessor);
        let tokens = DecoderTokens::resolve(&tokenizer)?;

        let mut lang_tokens = HashMap::with_capacity(SUPPORTED_LANGUAGES.len());
        for code in SUPPORTED_LANGUAGES {
            let lit = format!("<|{code}|>");
            if let Some(id) = tokenizer.token_to_id(&lit) {
                lang_tokens.insert((*code).to_string(), id as i64);
            }
        }
        if lang_tokens.is_empty() {
            return Err(Error::Tokenizer(
                "No supported language tokens found in tokenizer.json".into(),
            ));
        }

        Ok(Self {
            model,
            tokenizer,
            preprocessor,
            feature_cache,
            lang_tokens,
            tokens,
            max_decode_tokens: DEFAULT_MAX_DECODE_TOKENS,
        })
    }

    /// Training chunk length (in seconds) recorded in the upstream
    /// `preprocessor_config.json`. Exposed as metadata only — the model
    /// card lists long-form transcription as supported and audio longer
    /// than this transcribes fine in practice.
    pub fn training_chunk_secs(&self) -> f32 {
        TRAINING_CHUNK_SECS
    }

    /// Current maximum number of tokens the decoder will emit per call.
    pub fn max_decode_tokens(&self) -> usize {
        self.max_decode_tokens
    }

    /// Set the maximum number of tokens the decoder will emit per call.
    /// Values above the model's hard limit (1024) are clamped.
    pub fn set_max_decode_tokens(&mut self, max: usize) {
        self.max_decode_tokens = max.clamp(1, MAX_DECODE_TOKENS_LIMIT);
    }

    /// Transcribe raw 16 kHz mono f32 audio samples.
    ///
    /// `language` accepts a [`Language`](crate::Language) or, via
    /// `impl Into<Language>`, a bare ISO 639-1 code string (e.g. `"en"`, `"fr"`,
    /// `"de"`, `"ja"`). Cohere uses bare ISO codes, not the locale form Nemotron
    /// uses (`"en-US"`); pass the ISO code (or [`Language::Other`](crate::Language::Other))
    /// and it resolves through Cohere's supported-language table unchanged.
    /// `punctuation` controls whether output includes punctuation and
    /// capitalisation. `itn` enables inverse text normalisation
    /// (e.g. "twenty three" -> "23").
    ///
    /// # Long-form audio
    ///
    /// The model was trained on clips up to 35 s ([`Self::training_chunk_secs`]).
    /// Longer audio still runs but quality drifts past that range. For
    /// long-form transcription split the waveform into <=35 s chunks
    /// yourself and call this method once per chunk. The reference
    /// implementation in [`nano-cohere-transcribe`](https://github.com/Deep-unlearning/nano-cohere-transcribe/blob/main/nano_cohere_transcribe/chunk.py)
    /// splits at the quietest point in the last 5 s of each 35 s window
    /// and joins per-chunk text with `""` for `ja`/`zh` and `" "` for
    /// every other language. parakeet-rs deliberately leaves the
    /// chunking policy to the caller.
    pub fn transcribe_audio(
        &mut self,
        audio: &[f32],
        language: impl Into<crate::Language>,
        punctuation: bool,
        itn: bool,
    ) -> Result<String> {
        self.transcribe_audio_with_options(
            audio,
            language,
            punctuation,
            itn,
            CohereOptions::default(),
        )
    }

    /// Transcribe with explicit [`CohereOptions`] controlling the model's
    /// timestamp/diarize prompt toggles.
    ///
    /// Behaves exactly like [`Self::transcribe_audio`] but additionally lets
    /// the caller request `<|timestamp|>` / `<|diarize|>` markers in the
    /// output. Passing [`CohereOptions::default()`] is byte-for-byte identical
    /// to [`Self::transcribe_audio`]. See that method for the `language` /
    /// `punctuation` / `itn` semantics and the long-form audio notes.
    pub fn transcribe_audio_with_options(
        &mut self,
        audio: &[f32],
        language: impl Into<crate::Language>,
        punctuation: bool,
        itn: bool,
        options: CohereOptions,
    ) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        let language = language.into();
        let code = language.as_str();
        let lang_token = self.lang_tokens.get(code).copied().ok_or_else(|| {
            Error::Config(format!(
                "Unsupported language '{}'. Supported: {:?}",
                code,
                self.supported_languages()
            ))
        })?;

        // 1. Mel features. extract_features_raw returns [T, N_MELS] after
        //    preemphasis + STFT + log-mel + per-feature normalisation, which
        //    matches the CohereAsrFeatureExtractor pipeline. We add a batch
        //    axis to get [1, T, N_MELS] for the encoder.
        //
        // `as_standard_layout().to_owned()` is required because `insert_axis`
        // on a view may produce non-standard strides, but ort::TensorRef
        // needs C-contiguous memory.
        let mel_2d = extract_features_with_cache(
            audio.to_vec(),
            self.preprocessor.sampling_rate as u32,
            1,
            &self.preprocessor,
            &self.feature_cache,
        )?;
        let mel_3d = mel_2d.insert_axis(Axis(0)).as_standard_layout().to_owned();

        // 2. Encoder
        let encoder_out = self.model.run_encoder(&mel_3d)?;

        // 3. Build the canonical Cohere decoder prompt matching what
        //    CohereAsrProcessor in transformers produces. The source and
        //    target language tokens are both the caller's `language` code
        //    since this is pure transcription (no translation).
        let prompt = self
            .tokens
            .build_prompt(lang_token, punctuation, itn, options);

        // 4. Greedy decode loop
        let token_ids = self.decode_greedy(&prompt, &encoder_out)?;

        // 5. Detokenise (skip special tokens)
        let text = self
            .tokenizer
            .decode(
                &token_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("Failed to decode tokens: {e}")))?;

        // Strip leading stray punctuation the decoder sometimes emits
        // before the first real token.
        let cleaned = text
            .trim()
            .trim_start_matches(['.', '?', '!', ','])
            .trim()
            .to_string();

        Ok(cleaned)
    }

    /// Greedy autoregressive decode using the merged decoder's growing
    /// `past_key_values` cache. The first call feeds the prompt and lets
    /// the model populate the cross-attention encoder cache; subsequent
    /// calls feed one token at a time.
    fn decode_greedy(
        &mut self,
        prompt: &[i64],
        encoder_out: &CohereEncoderOutput,
    ) -> Result<Vec<i64>> {
        let mut past_kv = CoherePastKv::empty();
        let mut output_tokens: Vec<i64> = Vec::new();

        // First step: feed entire prompt
        let prompt_tensor = Array2::from_shape_vec((1, prompt.len()), prompt.to_vec())
            .map_err(|e| Error::Model(format!("Prompt tensor shape error: {e}")))?;
        let (logits, new_past) =
            self.model
                .run_decoder_step(&prompt_tensor, &past_kv, encoder_out)?;
        past_kv = new_past;

        let mut next_token = argmax(logits.as_slice().unwrap());
        if next_token == self.tokens.eos {
            return Ok(output_tokens);
        }
        output_tokens.push(next_token);

        // Continue one token at a time up to the configured max.
        for _ in 1..self.max_decode_tokens {
            let token_tensor = Array2::from_shape_vec((1, 1), vec![next_token])
                .map_err(|e| Error::Model(format!("Token tensor shape error: {e}")))?;
            let (logits, new_past) =
                self.model
                    .run_decoder_step(&token_tensor, &past_kv, encoder_out)?;
            past_kv = new_past;

            next_token = argmax(logits.as_slice().unwrap());
            if next_token == self.tokens.eos {
                break;
            }
            output_tokens.push(next_token);

            // Detect n-gram repetition: if the last N tokens match a
            // previous sequence the model is stuck in a loop.
            if let Some(repeat_len) = find_ngram_repetition(&output_tokens, 8) {
                output_tokens.truncate(output_tokens.len() - repeat_len);
                break;
            }
        }

        Ok(output_tokens)
    }

    /// Sorted list of supported ISO 639-1 language codes.
    pub fn supported_languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self.lang_tokens.keys().cloned().collect();
        langs.sort();
        langs
    }
}

/// Look up a special token id by literal, returning a clear error if it's
/// not present in the tokenizer vocabulary.
fn require_token(tokenizer: &Tokenizer, literal: &str) -> Result<i64> {
    tokenizer
        .token_to_id(literal)
        .map(|id| id as i64)
        .ok_or_else(|| Error::Tokenizer(format!("Tokenizer is missing required token {literal}")))
}

/// Check if the token sequence ends with a repeated n-gram of length
/// `>= min_len`. Returns `Some(repeat_len)` if the last `repeat_len` tokens
/// are an exact copy of the preceding segment.
fn find_ngram_repetition(tokens: &[i64], min_len: usize) -> Option<usize> {
    let n = tokens.len();
    if n < min_len * 2 {
        return None;
    }
    for repeat_len in min_len..=(n / 2) {
        let tail = &tokens[n - repeat_len..];
        let prev = &tokens[n - 2 * repeat_len..n - repeat_len];
        if tail == prev {
            return Some(repeat_len);
        }
    }
    None
}

/// Greedy argmax over a slice of f32 logits, returning the token id as `i64`.
/// Delegates to the single shared decoder policy (first-wins + finite-guard;
/// unified by T10/M5).
fn argmax(logits: &[f32]) -> i64 {
    crate::vocab::argmax(logits) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_supported_languages_count() {
        // Cohere Transcribe officially ships trained weights for 14 languages
        assert_eq!(SUPPORTED_LANGUAGES.len(), 14);
    }

    // Distinct sentinel ids so a wrong slot is obvious in assertions. No
    // tokenizer/model needed - we exercise pure prompt construction.
    fn fake_tokens() -> DecoderTokens {
        DecoderTokens {
            decoder_start: 1,
            startofcontext: 2,
            sot: 3,
            emo_undefined: 4,
            eos: 5,
            pnc: 6,
            nopnc: 7,
            timestamp: 8,
            notimestamp: 9,
            diarize: 10,
            nodiarize: 11,
            itn: 12,
            noitn: 13,
        }
    }

    #[test]
    fn test_build_prompt_default_matches_legacy_behavior() {
        let t = fake_tokens();
        let lang = 99;
        // Default options (timestamps off, diarize off) must reproduce the
        // exact sequence the engine hardcoded before T19, ending in
        // notimestamp(9), nodiarize(11).
        let prompt = t.build_prompt(lang, /*punctuation*/ true, /*itn*/ false, CohereOptions::default());
        assert_eq!(prompt, vec![1, 2, 3, 4, 99, 99, /*pnc*/ 6, /*noitn*/ 13, /*notimestamp*/ 9, /*nodiarize*/ 11]);
    }

    #[test]
    fn test_build_prompt_toggles_swap_individual_tokens() {
        let t = fake_tokens();
        let lang = 99;

        // timestamps on -> timestamp(8) in slot 8, diarize still off -> nodiarize(11).
        let ts = t.build_prompt(lang, false, false, CohereOptions::default().with_timestamps(true));
        assert_eq!(ts[8], t.timestamp);
        assert_eq!(ts[9], t.nodiarize);

        // diarize on -> diarize(10) in slot 9, timestamp still off -> notimestamp(9).
        let dz = t.build_prompt(lang, false, false, CohereOptions::default().with_diarize(true));
        assert_eq!(dz[8], t.notimestamp);
        assert_eq!(dz[9], t.diarize);

        // both on.
        let both = t.build_prompt(lang, false, false, CohereOptions { timestamps: true, diarize: true });
        assert_eq!(both[8], t.timestamp);
        assert_eq!(both[9], t.diarize);
    }

    #[test]
    fn test_ngram_repetition_detection() {
        // No repetition
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 5, 6, 7, 8], 4), None);

        // Repeated 4-gram: [1,2,3,4] appears twice
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 1, 2, 3, 4], 4), Some(4));

        // Repeated 8-gram
        let mut tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];
        tokens.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(find_ngram_repetition(&tokens, 8), Some(8));

        // Too short to detect
        assert_eq!(find_ngram_repetition(&[1, 2, 1, 2], 4), None);
    }
}
