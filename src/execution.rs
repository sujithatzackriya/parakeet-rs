use std::path::PathBuf;
use std::{fmt, rc::Rc};

use crate::error::Result;
use ort::session::builder::SessionBuilder;

/// Hardware acceleration backend used for ONNX inference.
///
/// `Cpu` is the default and the only variant present in a default build. Every
/// accelerator variant is gated behind a Cargo feature, so it only exists in the
/// enum when that feature is compiled in. To use a GPU/NPU backend you must both
/// enable the feature at build time *and* select the variant at runtime; the
/// default build is CPU-only and GPU must be opted into via features. All GPU
/// providers automatically fall back to CPU if they fail to initialise.
///
/// # Feature-to-provider matrix
///
/// | Cargo feature | Provider     | Notes                                    |
/// |---------------|--------------|------------------------------------------|
/// | (none)        | `Cpu`        | Always available; the default.           |
/// | `cuda`        | `Cuda`       | NVIDIA GPU. 5-10x speedup.                |
/// | `tensorrt`    | `TensorRT`   | NVIDIA, graph-optimised.                  |
/// | `coreml`      | `CoreML`     | Apple. See caveat below.                  |
/// | `directml`    | `DirectML`   | Windows GPU.                              |
/// | `migraphx`    | `MIGraphX`   | AMD GPU.                                  |
/// | `openvino`    | `OpenVINO`   | Intel CPU/GPU/NPU.                        |
/// | `webgpu`      | `WebGPU`     | Experimental; may produce wrong results. |
/// | `nnapi`       | `NNAPI`      | Android.                                  |
///
/// # CoreML caveat
///
/// CoreML currently runs *slower* than CPU for Sortformer/Parakeet models because
/// the ONNX graphs have dynamic input shapes, preventing CoreML from building
/// optimised execution plans for ANE/GPU. CoreML claims the nodes but runs them on
/// CPU with extra overhead. For this reason [`ExecutionProvider::auto`] never picks
/// CoreML automatically.
///
/// # WebGPU caveat
///
/// WebGPU is experimental and may produce incorrect results.
///
/// # Discoverability
///
/// Because the variants are feature-gated, downstream code cannot `match` on a
/// variant that was not compiled in. Use [`ExecutionProvider::compiled`] to learn
/// which providers the current build supports at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
}

impl ExecutionProvider {
    /// Returns every execution provider compiled into the current build.
    ///
    /// The result reflects the Cargo features enabled at build time: `Cpu` is
    /// always present, and each accelerator appears only when its feature is on
    /// (see the [feature-to-provider matrix](ExecutionProvider#feature-to-provider-matrix)).
    /// This is the runtime answer to "what providers can I select?" given the
    /// feature-gated variants.
    pub fn compiled() -> Vec<ExecutionProvider> {
        vec![
            ExecutionProvider::Cpu,
            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda,
            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT,
            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML,
            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML,
            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX,
            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO,
            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU,
            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI,
        ]
    }

    /// Picks the best *compiled* accelerator, preferring proven GPU backends and
    /// falling back to `Cpu`.
    ///
    /// This is opt-in only and never the default. Preference order is CUDA,
    /// TensorRT, DirectML, MIGraphX, OpenVINO, NNAPI, then `Cpu`. CoreML and the
    /// experimental WebGPU are deliberately excluded from auto-selection: CoreML
    /// runs slower than CPU for these dynamic-shape graphs (see the
    /// [CoreML caveat](ExecutionProvider#coreml-caveat)) and WebGPU may produce
    /// incorrect results. On a default (CPU-only) build this returns `Cpu`.
    pub fn auto() -> ExecutionProvider {
        #[cfg(feature = "cuda")]
        return ExecutionProvider::Cuda;
        #[cfg(all(not(feature = "cuda"), feature = "tensorrt"))]
        return ExecutionProvider::TensorRT;
        #[cfg(all(not(feature = "cuda"), not(feature = "tensorrt"), feature = "directml"))]
        return ExecutionProvider::DirectML;
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "tensorrt"),
            not(feature = "directml"),
            feature = "migraphx"
        ))]
        return ExecutionProvider::MIGraphX;
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "tensorrt"),
            not(feature = "directml"),
            not(feature = "migraphx"),
            feature = "openvino"
        ))]
        return ExecutionProvider::OpenVINO;
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "tensorrt"),
            not(feature = "directml"),
            not(feature = "migraphx"),
            not(feature = "openvino"),
            feature = "nnapi"
        ))]
        return ExecutionProvider::NNAPI;
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "tensorrt"),
            not(feature = "directml"),
            not(feature = "migraphx"),
            not(feature = "openvino"),
            not(feature = "nnapi")
        ))]
        ExecutionProvider::Cpu
    }
}

#[derive(Clone)]
pub struct ExecutionConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    pub coreml_cache_dir: Option<PathBuf>,
}

impl fmt::Debug for ExecutionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionConfig")
            .field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .finish()
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            configure: None,
            coreml_cache_dir: None,
        }
    }
}

impl ExecutionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    /// Every execution provider compiled into the current build.
    ///
    /// Delegates to [`ExecutionProvider::compiled`]; see that for the
    /// feature-to-provider matrix and discoverability notes.
    pub fn compiled_providers() -> Vec<ExecutionProvider> {
        ExecutionProvider::compiled()
    }

    /// Selects the best *compiled* accelerator (opt-in, never the default).
    ///
    /// Delegates to [`ExecutionProvider::auto`]; on a default CPU-only build this
    /// keeps `Cpu`. CoreML and WebGPU are excluded from auto-selection (see the
    /// [CoreML caveat](ExecutionProvider#coreml-caveat)).
    pub fn with_auto_provider(self) -> Self {
        self.with_execution_provider(ExecutionProvider::auto())
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + 'static,
    ) -> Self {
        self.configure = Some(Rc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML};
                let mut coreml = CoreML::default().with_compute_units(ComputeUnits::CPUAndGPU);

                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_always_includes_cpu_and_matches_features() {
        let providers = ExecutionProvider::compiled();
        assert!(
            providers.contains(&ExecutionProvider::Cpu),
            "Cpu must always be a compiled provider"
        );
        assert_eq!(
            providers[0],
            ExecutionProvider::Cpu,
            "Cpu must be first (the default)"
        );

        // Count is 1 (Cpu) plus one per enabled accelerator feature, so the
        // discovery list tracks the build configuration exactly.
        let expected = 1
            + cfg!(feature = "cuda") as usize
            + cfg!(feature = "tensorrt") as usize
            + cfg!(feature = "coreml") as usize
            + cfg!(feature = "directml") as usize
            + cfg!(feature = "migraphx") as usize
            + cfg!(feature = "openvino") as usize
            + cfg!(feature = "webgpu") as usize
            + cfg!(feature = "nnapi") as usize;
        assert_eq!(providers.len(), expected);

        // ExecutionConfig delegate returns the same set.
        assert_eq!(ExecutionConfig::compiled_providers(), providers);
    }

    #[test]
    fn auto_is_a_compiled_provider_and_never_coreml_or_webgpu() {
        let chosen = ExecutionProvider::auto();
        assert!(
            ExecutionProvider::compiled().contains(&chosen),
            "auto() must pick a provider that is compiled in"
        );
        #[cfg(feature = "coreml")]
        assert_ne!(
            chosen,
            ExecutionProvider::CoreML,
            "auto() must not pick CoreML (slower than CPU here)"
        );
        #[cfg(feature = "webgpu")]
        assert_ne!(
            chosen,
            ExecutionProvider::WebGPU,
            "auto() must not pick experimental WebGPU"
        );
    }

    #[test]
    fn auto_defaults_to_cpu_when_no_gpu_feature() {
        // Default build has no accelerator features, so auto() and the default
        // EP both resolve to Cpu - the default behaviour is unchanged.
        #[cfg(not(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "nnapi"
        )))]
        {
            assert_eq!(ExecutionProvider::auto(), ExecutionProvider::Cpu);
            assert_eq!(
                ExecutionConfig::new().with_auto_provider().execution_provider,
                ExecutionProvider::Cpu
            );
        }
    }
}
