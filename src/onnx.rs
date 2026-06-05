//! Shared ONNX session-build and model-file-discovery helpers (M2a, M9).
//!
//! Before this module the same three-line `SessionBuilder` idiom and a set of
//! divergent `find_encoder` / `find_decoder_joint` candidate lists were
//! copy-pasted across every `model_*.rs` wrapper. They are consolidated here
//! WITHOUT changing behaviour:
//!
//! * [`build_session`] reproduces the exact `Session::builder()` +
//!   `ExecutionConfig::apply_to_session_builder` + `commit_from_file` sequence
//!   (same graph-opt level, thread counts, execution provider, and custom
//!   configure hook - all of that lives in `apply_to_session_builder`, which is
//!   unchanged).
//! * [`resolve_onnx_file`] replaces the per-family file pickers. Each family
//!   still searches for its own filename stems and directory layout (the
//!   `encoder` vs `encoder-model` vs `encoder_model` vs `model` stems are
//!   genuinely different on disk), but precision selection is now driven by one
//!   documented precedence via the [`Quantization`] knob.
//!
//! ## Precedence policy (M9)
//!
//! For [`Quantization::Auto`] the order is **highest precision first**:
//! `Fp32 -> Fp16 -> Int8 -> Int4`. This matches the historical CTC, TDT and
//! Unified pickers (which already listed fp32 first). The Multitalker and
//! Cohere pickers historically listed the quantized variant FIRST; to keep
//! their resolution byte-identical their callers pass `Quantization::Int8`
//! instead of `Auto`, which floats int8 (and the quantized/q4 family) to the
//! front. A directory that contains BOTH fp32 and int8 variants would resolve
//! differently under `Auto` than under the old int8-first pickers - that is the
//! documented M9 precedence break. The single-precision local test dirs
//! (`./nemotron`, `./nemotron_multi`: one fp32 `encoder.onnx` + one
//! `decoder_joint.onnx`) are unaffected: with only one variant present, every
//! `prefer` value resolves the same file.

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ort::session::Session;
use std::path::{Path, PathBuf};

/// Build an ONNX [`Session`] from a file, applying the execution config exactly
/// as the per-wrapper call sites used to do inline.
///
/// This is the single home for the verbatim
/// `Session::builder()? -> apply_to_session_builder -> commit_from_file`
/// idiom. Execution-provider selection, graph-optimisation level, thread
/// settings and the optional custom-configure hook are all owned by
/// [`ExecutionConfig::apply_to_session_builder`] and are unchanged.
pub(crate) fn build_session(exec_config: &ExecutionConfig, path: &Path) -> Result<Session> {
    let builder = Session::builder()?;
    let mut builder = exec_config.apply_to_session_builder(builder)?;
    Ok(builder.commit_from_file(path)?)
}

/// Precision variant preference for [`resolve_onnx_file`].
///
/// The full ladder is the documented `prefer` knob (M9); current callers only
/// pass `Auto` (fp32-first families) and `Int8` (multitalker/cohere), so the
/// other explicit variants are exercised by the unit tests rather than the
/// model loaders.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantization {
    /// Highest precision first: `Fp32 -> Fp16 -> Int8 -> Int4`.
    Auto,
    Fp32,
    Fp16,
    Int8,
    Int4,
}

/// Precision tag attached to each candidate filename so the resolver can order
/// candidates by [`Quantization`] preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Precision {
    Fp32,
    Fp16,
    Int8,
    Int4,
}

/// A single candidate file: its on-disk name (may include a `onnx/` subdir for
/// the Cohere nested layout) and the precision it represents.
pub(crate) type Candidate<'a> = (&'a str, Precision);

impl Quantization {
    /// Stable sort key: lower value = tried earlier. For `Auto` this yields
    /// `Fp32 < Fp16 < Int8 < Int4`; for an explicit preference the matching
    /// precision sorts to 0 (front) and the rest keep the `Auto` order.
    fn rank(self, p: Precision) -> u8 {
        let auto = match p {
            Precision::Fp32 => 0,
            Precision::Fp16 => 1,
            Precision::Int8 => 2,
            Precision::Int4 => 3,
        };
        let preferred = match self {
            Quantization::Auto => return auto,
            Quantization::Fp32 => Precision::Fp32,
            Quantization::Fp16 => Precision::Fp16,
            Quantization::Int8 => Precision::Int8,
            Quantization::Int4 => Precision::Int4,
        };
        if p == preferred {
            0
        } else {
            // keep relative Auto order behind the preferred variant
            auto + 1
        }
    }
}

/// Resolve a model file inside `dir` by trying the family's candidate filenames
/// ordered by the `prefer` precision policy, with the same first-match-wins
/// semantics the old per-family pickers used.
///
/// `role` is a human-readable label used only in the error message ("encoder",
/// "decoder_joint", "model", ...). `candidates` is the family's full filename
/// set, each tagged with its precision; the order WITHIN a precision class is
/// preserved (stable sort), so when `prefer` does not discriminate the original
/// listing order still decides.
pub(crate) fn resolve_onnx_file(
    dir: &Path,
    role: &str,
    prefer: Quantization,
    candidates: &[Candidate<'_>],
) -> Result<PathBuf> {
    let mut ordered: Vec<&Candidate<'_>> = candidates.iter().collect();
    ordered.sort_by_key(|(_, p)| prefer.rank(*p));

    for (name, _) in ordered {
        let path = dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(Error::Config(format!(
        "No {role} model found in {}",
        dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Build a unique temp dir containing the named (empty) files.
    fn dir_with(files: &[&str]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "parakeet-onnx-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        for f in files {
            if let Some(parent) = Path::new(f).parent() {
                fs::create_dir_all(base.join(parent)).unwrap();
            }
            fs::write(base.join(f), b"").unwrap();
        }
        base
    }

    const ENC: &[Candidate] = &[
        ("encoder.onnx", Precision::Fp32),
        ("encoder_fp16.onnx", Precision::Fp16),
        ("encoder.int8.onnx", Precision::Int8),
        ("encoder.q4.onnx", Precision::Int4),
    ];

    #[test]
    fn auto_prefers_highest_precision() {
        let dir = dir_with(&["encoder.onnx", "encoder.int8.onnx", "encoder.q4.onnx"]);
        let got = resolve_onnx_file(&dir, "encoder", Quantization::Auto, ENC).unwrap();
        assert_eq!(got, dir.join("encoder.onnx"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn int8_preference_floats_int8_to_front() {
        let dir = dir_with(&["encoder.onnx", "encoder.int8.onnx"]);
        let got = resolve_onnx_file(&dir, "encoder", Quantization::Int8, ENC).unwrap();
        assert_eq!(got, dir.join("encoder.int8.onnx"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn explicit_preference_falls_back_when_absent() {
        // Prefer int8 but only fp32 present -> fp32 still resolves.
        let dir = dir_with(&["encoder.onnx"]);
        let got = resolve_onnx_file(&dir, "encoder", Quantization::Int8, ENC).unwrap();
        assert_eq!(got, dir.join("encoder.onnx"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn single_precision_dir_is_prefer_invariant() {
        // The CRITICAL M9 guard: a dir with only fp32 resolves the SAME file
        // regardless of the prefer knob (mirrors ./nemotron, ./nemotron_multi).
        let dir = dir_with(&["encoder.onnx"]);
        for prefer in [
            Quantization::Auto,
            Quantization::Fp32,
            Quantization::Fp16,
            Quantization::Int8,
            Quantization::Int4,
        ] {
            let got = resolve_onnx_file(&dir, "encoder", prefer, ENC).unwrap();
            assert_eq!(got, dir.join("encoder.onnx"), "prefer={prefer:?}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_file_reports_role_and_dir() {
        let dir = dir_with(&[]);
        let err = resolve_onnx_file(&dir, "decoder_joint", Quantization::Auto, ENC).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("decoder_joint"), "got: {msg}");
        assert!(msg.contains(&dir.display().to_string()), "got: {msg}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn within_precision_order_is_stable() {
        // Two fp32 candidates: first listed wins (preserves old list order).
        const TWO_FP32: &[Candidate] = &[
            ("encoder-model.onnx", Precision::Fp32),
            ("encoder.onnx", Precision::Fp32),
        ];
        let dir = dir_with(&["encoder-model.onnx", "encoder.onnx"]);
        let got = resolve_onnx_file(&dir, "encoder", Quantization::Auto, TWO_FP32).unwrap();
        assert_eq!(got, dir.join("encoder-model.onnx"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
