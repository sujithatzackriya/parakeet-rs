//! # parakeet-rs
//!
//! Rust bindings for NVIDIA's Parakeet speech recognition model using ONNX Runtime.
//!
//! Parakeet is a state-of-the-art automatic speech recognition (ASR) model developed by NVIDIA,
//! based on the FastConformer-TDT architecture with 600 million parameters.
//!
//! ## Features
//!
//! - Easy-to-use API for speech-to-text transcription
//! - Support for ONNX format models
//! - 16kHz mono audio input
//! - Punctuation and capitalization included in output
//! - Fast inference using ONNX Runtime
//!
//! ## Quick Start
//!
//! ```no_run
//! use parakeet_rs::{Parakeet, Transcriber, TimestampMode};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load the model (pass None for the default CPU execution provider)
//! let mut parakeet = Parakeet::from_pretrained(".", None)?;
//!
//! // Transcribe 16kHz mono audio samples (see examples/raw.rs for audio loading)
//! let audio: Vec<f32> = Vec::new();
//! let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Words))?;
//! println!("Transcription: {}", result.text);
//! # Ok(())
//! # }
//! ```
//!
//! ## Model Requirements
//!
//! Your model directory should contain:
//! - `model.onnx` - The ONNX model file
//! - `model.onnx_data` - External model weights
//! - `config.json` - Model configuration
//! - `preprocessor_config.json` - Audio preprocessing configuration
//! - `tokenizer.json` - Tokenizer vocabulary
//! - `tokenizer_config.json` - Tokenizer configuration
//!
//! ## Audio Requirements
//!
//! - Format: WAV
//! - Sample Rate: 16kHz
//! - Channels: Mono (stereo will be converted automatically)
//! - Bit Depth: 16-bit PCM or 32-bit float

mod audio;
mod config;
mod decoder;
mod decoder_tdt;
mod error;
mod execution;
mod model;
mod model_eou;
mod model_nemotron;
mod model_tdt;
mod model_unified;
mod nemotron;
mod onnx;
mod parakeet;
mod parakeet_eou;
mod parakeet_tdt;
mod parakeet_unified;
#[cfg(feature = "sortformer")]
pub mod sortformer;
#[cfg(feature = "multitalker")]
mod model_multitalker;
#[cfg(feature = "multitalker")]
pub mod multitalker;
#[cfg(feature = "cohere")]
mod model_cohere;
#[cfg(feature = "cohere")]
pub mod cohere;
mod timestamps;
mod transcriber;
mod vocab;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};
pub use parakeet::Parakeet;
pub use parakeet_tdt::ParakeetTDT;
pub use timestamps::TimestampMode;
pub use transcriber::Transcriber;

pub use audio::FeatureCache;
pub use config::{ModelConfig as ModelConfigJson, PreprocessorConfig};

pub use decoder::{TimedToken, TranscriptionResult};
pub use nemotron::{Nemotron, NemotronHandle, NemotronMode};
pub use parakeet_eou::{ParakeetEOU, ParakeetEOUHandle, EOU_CHUNK_SAMPLES};
pub use parakeet_unified::{ParakeetUnified, ParakeetUnifiedHandle, UnifiedStreamingConfig};

#[cfg(feature = "multitalker")]
pub use multitalker::{LatencyMode, MultitalkerASR, MultitalkerConfig, SpeakerTranscript, WordTimestamp};

#[cfg(feature = "cohere")]
pub use cohere::{CohereASR, CohereOptions};
