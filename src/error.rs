pub type Result<T> = std::result::Result<T, Error>;

/// Library error type.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change.
/// Variants that wrap a typed cause (`Io`, `Ort`) expose it via
/// [`std::error::Error::source`]; the `String` variants are flattened messages
/// kept for backward compatibility with existing call sites.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[source] ort::Error),

    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Config error: {0}")]
    Config(String),

    /// The loaded ONNX graph is missing I/O names the loader requires,
    /// i.e. the model export does not match the expected format.
    #[error("Model format mismatch for {role}: missing expected I/O name(s) {missing:?}")]
    ModelFormat {
        role: &'static str,
        missing: Vec<String>,
    },
}

// `ort::Error<R>` is generic over a recover type; `Session::builder()` and the
// `commit_*`/`apply_*` calls can surface `Error<SessionBuilder>` as well as the
// default `Error<()>`. thiserror's `#[from]` would only cover `Error<()>`, so the
// generic conversion is kept hand-written and `Ort` is wired with `#[source]`
// (not `#[from]`) to avoid a conflicting impl.
impl<R> From<ort::Error<R>> for Error
where
    ort::Error<R>: Into<ort::Error<()>>,
{
    fn from(e: ort::Error<R>) -> Self {
        Error::Ort(e.into())
    }
}

/// Validate that an ONNX session exposes every input name the loader relies on.
///
/// Returns [`Error::ModelFormat`] listing the missing names if the export does
/// not match the expected format (folds the G-C model-format contract). Used at
/// model-load time so a re-exported graph fails fast with a structured error
/// instead of a stringly `Error::Model` deep in the inference loop.
pub(crate) fn validate_input_names(
    session: &ort::session::Session,
    role: &'static str,
    required: &[&str],
) -> Result<()> {
    let present: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    let missing: Vec<String> = required
        .iter()
        .filter(|name| !present.contains(*name))
        .map(|name| name.to_string())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::ModelFormat { role, missing })
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<hound::Error> for Error {
    fn from(e: hound::Error) -> Self {
        Error::Audio(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn io_variant_preserves_source_chain() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let err: Error = io.into();
        // The typed cause must be reachable via `source()` (was `None` before
        // the thiserror migration flattened it into the Display string).
        let src = err.source().expect("Io variant must expose its source");
        assert_eq!(src.to_string(), "missing file");
    }

    #[test]
    fn stringly_variant_has_no_source() {
        let err = Error::Model("flattened message".to_string());
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "Model error: flattened message");
    }

    #[test]
    fn model_format_variant_lists_missing_names() {
        let err = Error::ModelFormat {
            role: "nemotron encoder",
            missing: vec!["processed_signal".to_string(), "cache_last_time".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("nemotron encoder"));
        assert!(msg.contains("processed_signal"));
        assert!(msg.contains("cache_last_time"));
    }
}
