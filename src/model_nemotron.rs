use crate::error::{Error, Result};
use crate::execution::ExecutionConfig;
use ndarray::{Array1, Array3, Array4};
use ort::session::{Session, SessionInputValue};
use ort::value::ValueType;
use std::borrow::Cow;
use std::path::Path;

/// Encoder cache state for Nemotron streaming inference.
/// Shapes are model-dependent (English 0.6B uses left_context=70,
/// multilingual 3.5 uses left_context=56) so always construct via [`NemotronEncoderCache::with_dims`].
#[derive(Clone)]
pub struct NemotronEncoderCache {
    pub cache_last_channel: Array4<f32>,
    pub cache_last_time: Array4<f32>,
    pub cache_last_channel_len: Array1<i64>,
}

impl NemotronEncoderCache {
    pub fn with_dims(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            cache_last_channel: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            cache_last_time: Array4::zeros((num_layers, 1, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

/// Nemotron ONNX wrapper.
/// Encoder and decoder_joint sessions live side by side; [`Self::has_prompt`]
/// flips on automatically when the encoder graph exposes a `prompt_index` input
/// (the multilingual variant).
pub struct NemotronModel {
    encoder: Session,
    decoder_joint: Session,
    pub config: NemotronModelConfig,
    pub has_prompt: bool,
}

/// cfg for Nemotron model dims.
#[derive(Debug, Clone)]
pub struct NemotronModelConfig {
    pub num_encoder_layers: usize,
    pub hidden_dim: usize,
    pub left_context: usize,
    pub conv_context: usize,
    pub decoder_lstm_dim: usize,
    pub decoder_lstm_layers: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
}

impl NemotronModel {
    /// Load encoder + decoder/joint sessions and read all dimension info
    /// straight from the encoder graph. `vocab_size` is supplied by the
    /// caller (it comes from the tokenizer).
    ///
    /// Note that, multilang graph is identified by the presence of a
    /// `prompt_index` input that flips [`Self::has_prompt`] on.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_path = model_dir.join("decoder_joint.onnx");

        if !encoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing encoder.onnx in {}",
                model_dir.display()
            )));
        }
        if !decoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing decoder_joint.onnx in {}",
                model_dir.display()
            )));
        }

        let encoder = crate::onnx::build_session(&exec_config, &encoder_path)?;
        let decoder_joint = crate::onnx::build_session(&exec_config, &decoder_path)?;

        // Fail fast with a structured error if the export does not expose the
        // I/O names the encoder/decoder run paths rely on (G-C). `prompt_index`
        // is intentionally excluded: it is optional (English vs multilingual).
        crate::error::validate_input_names(
            &encoder,
            "nemotron encoder",
            &[
                "processed_signal",
                "processed_signal_length",
                "cache_last_channel",
                "cache_last_time",
                "cache_last_channel_len",
            ],
        )?;
        crate::error::validate_input_names(
            &decoder_joint,
            "nemotron decoder_joint",
            &[
                "encoder_outputs",
                "targets",
                "target_length",
                "input_states_1",
                "input_states_2",
            ],
        )?;

        let mut config = NemotronModelConfig {
            num_encoder_layers: 24,
            hidden_dim: 1024,
            left_context: 70,
            conv_context: 8,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            vocab_size,
            blank_id: vocab_size,
        };

        let mut has_prompt = false;
        for outlet in encoder.inputs() {
            let name = outlet.name();
            if name == "prompt_index" {
                has_prompt = true;
                continue;
            }
            let ValueType::Tensor { shape, .. } = outlet.dtype() else { continue };
            let dims: &[i64] = shape;
            match name {
                "cache_last_channel" if dims.len() == 4 => {
                    config.num_encoder_layers = dims[0] as usize;
                    config.left_context = dims[2] as usize;
                    config.hidden_dim = dims[3] as usize;
                }
                "cache_last_time" if dims.len() == 4 => {
                    config.conv_context = dims[3] as usize;
                }
                _ => {}
            }
        }

        Ok(Self {
            encoder,
            decoder_joint,
            config,
            has_prompt,
        })
    }

    /// Test-only constructor that builds both sessions from a tiny in-memory
    /// identity ONNX graph, so state-management tests (e.g. `Nemotron::reset`)
    /// can run with NO 2 GB model download. The sessions are never executed by
    /// those tests; only the surrounding Rust state is exercised.
    #[cfg(test)]
    pub(crate) fn new_in_memory_for_test(
        config: NemotronModelConfig,
        has_prompt: bool,
    ) -> Result<Self> {
        // Minimal `Identity` ONNX (ir_version 9, opset 13), emitted by the
        // onnx Python helper. Two independent sessions are built from it.
        const IDENTITY_ONNX: &[u8] = &[
            8, 9, 58, 55, 10, 16, 10, 1, 120, 18, 1, 121, 34, 8, 73, 100, 101, 110, 116, 105,
            116, 121, 18, 1, 103, 90, 15, 10, 1, 120, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1,
            98, 15, 10, 1, 121, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1, 66, 4, 10, 0, 16, 13,
        ];
        let encoder = Session::builder()?.commit_from_memory(IDENTITY_ONNX)?;
        let decoder_joint = Session::builder()?.commit_from_memory(IDENTITY_ONNX)?;
        Ok(Self {
            encoder,
            decoder_joint,
            config,
            has_prompt,
        })
    }

    /// Run encoder with cache-aware streaming.
    /// `prompt_index` must be `Some(_)` for multilingual models and `None`
    /// for eng only mistmaching will produce an ORT InvalidArgument err.
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &NemotronEncoderCache,
        prompt_index: Option<i64>,
    ) -> Result<(Array3<f32>, i64, NemotronEncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let mut inputs = ort::inputs![
            "processed_signal" => ort::value::Value::from_array(features.clone())?,
            "processed_signal_length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?
        ];
        if let Some(idx) = prompt_index {
            let prompt_arr = Array1::from_vec(vec![idx]);
            inputs.push((
                Cow::Borrowed("prompt_index"),
                SessionInputValue::from(ort::value::Value::from_array(prompt_arr)?),
            ));
        }

        let outputs = self.encoder.run(inputs)?;

        // [1, hidden_dim, time]
        let (shape, data) = outputs["encoded"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        let b = shape_dims[0] as usize;
        let d = shape_dims[1] as usize;
        let t = shape_dims[2] as usize;

        let encoder_out = Array3::from_shape_vec((b, d, t), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

        // on here we are extracting encoded length and new cache states.. and so on...
        let (_, enc_len_data) = outputs["encoded_len"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoded_len: {e}")))?;
        let encoded_len = enc_len_data[0];

        let (ch_shape, ch_data) = outputs["cache_last_channel_next"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_channel: {e}")))?;

        let (tm_shape, tm_data) = outputs["cache_last_time_next"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_time: {e}")))?;

        let (len_shape, len_data) = outputs["cache_last_channel_len_next"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_len: {e}")))?;

        let new_cache = NemotronEncoderCache {
            cache_last_channel: Array4::from_shape_vec(
                (
                    ch_shape[0] as usize,
                    ch_shape[1] as usize,
                    ch_shape[2] as usize,
                    ch_shape[3] as usize,
                ),
                ch_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_channel: {e}")))?,

            cache_last_time: Array4::from_shape_vec(
                (
                    tm_shape[0] as usize,
                    tm_shape[1] as usize,
                    tm_shape[2] as usize,
                    tm_shape[3] as usize,
                ),
                tm_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_time: {e}")))?,

            cache_last_channel_len: Array1::from_shape_vec(
                len_shape[0] as usize,
                len_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_len: {e}")))?,
        };

        Ok((encoder_out, encoded_len, new_cache))
    }

    /// Run decoder step.
    /// Returns: (logits [vocab_size], new_state_1, new_state_2)
    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, hidden_dim, 1]
        target_token: i32,
        state_1: &Array3<f32>, // [2, 1, 640]
        state_2: &Array3<f32>, // [2, 1, 640]
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        crate::onnx::run_rnnt_decoder_step(
            &mut self.decoder_joint,
            encoder_frame,
            target_token,
            state_1,
            state_2,
        )
    }
}
