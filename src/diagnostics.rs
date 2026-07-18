//! Runtime diagnostics for framework builds and CUDA execution.

use crate::config::{ComputeDevice, Config};
use crate::cuda::CudaRuntimeStats;
use serde::Serialize;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl DiagnosticStatus {
    pub fn is_success(self) -> bool {
        self != Self::Unhealthy
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

impl DiagnosticCheck {
    #[cfg(cuda)]
    fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail: detail.into(),
        }
    }

    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CudaDeviceReport {
    pub id: usize,
    pub name: String,
    pub compute_capability: String,
    pub free_memory_bytes: usize,
    pub total_memory_bytes: usize,
    pub compiled_code_compatible: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    pub status: DiagnosticStatus,
    pub package_version: &'static str,
    pub cuda_compiled: bool,
    pub python_compiled: bool,
    pub requested_device: &'static str,
    pub actual_device: &'static str,
    pub compiled_cuda_architectures: &'static str,
    pub nvcc_version: &'static str,
    pub fallback_reason: Option<String>,
    pub cuda_device: Option<CudaDeviceReport>,
    pub checks: Vec<DiagnosticCheck>,
    pub runtime_stats: Option<CudaRuntimeStats>,
}

impl DiagnosticReport {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "Talos-XII doctor: {}", self.status.as_str());
        let _ = writeln!(output, "  version: {}", self.package_version);
        let _ = writeln!(
            output,
            "  features: cuda={}, python={}",
            self.cuda_compiled, self.python_compiled
        );
        let _ = writeln!(
            output,
            "  device: requested={}, actual={}",
            self.requested_device, self.actual_device
        );
        let _ = writeln!(
            output,
            "  CUDA architectures: {}",
            self.compiled_cuda_architectures
        );
        let _ = writeln!(output, "  NVCC: {}", self.nvcc_version);
        if let Some(reason) = &self.fallback_reason {
            let _ = writeln!(output, "  fallback: {reason}");
        }
        if let Some(device) = &self.cuda_device {
            let _ = writeln!(
                output,
                "  GPU {}: {} (CC {}, memory {} free / {} total)",
                device.id,
                device.name,
                device.compute_capability,
                format_bytes(device.free_memory_bytes),
                format_bytes(device.total_memory_bytes)
            );
        }

        output.push_str("  checks:\n");
        for check in &self.checks {
            let state = if check.passed { "PASS" } else { "FAIL" };
            let _ = writeln!(output, "    [{state}] {}: {}", check.name, check.detail);
        }

        if let Some(stats) = self.runtime_stats {
            output.push_str("  CUDA runtime counters:\n");
            let _ = writeln!(
                output,
                "    matmul: attempts={}, successes={}, fallbacks={} (init={}, alloc={}, copy={}, gemm={})",
                stats.matmul_attempts,
                stats.matmul_successes,
                stats.matmul_fallbacks(),
                stats.matmul_fallback_init,
                stats.matmul_fallback_alloc,
                stats.matmul_fallback_copy,
                stats.matmul_fallback_gemm
            );
            let _ = writeln!(
                output,
                "    activation: attempts={}, successes={}, fallbacks={} (alloc={}, copy={}, kernel={})",
                stats.activation_attempts,
                stats.activation_successes,
                stats.activation_fallbacks(),
                stats.activation_fallback_alloc,
                stats.activation_fallback_copy,
                stats.activation_fallback_kernel
            );
            let _ = writeln!(
                output,
                "    log-softmax: attempts={}, successes={}, fallbacks={} (alloc={}, copy={}, kernel={})",
                stats.log_softmax_attempts,
                stats.log_softmax_successes,
                stats.log_softmax_fallbacks(),
                stats.log_softmax_fallback_alloc,
                stats.log_softmax_fallback_copy,
                stats.log_softmax_fallback_kernel
            );
            let _ = writeln!(
                output,
                "    backward: attempts={}, successes={}, fallbacks={}",
                stats.backward_attempts, stats.backward_successes, stats.backward_fallback_kernel
            );
            let _ = writeln!(
                output,
                "    optimizer: attempts={}, successes={}, failures={}",
                stats.optimizer_attempts, stats.optimizer_successes, stats.optimizer_fallback_param
            );
        }
        output
    }
}

pub fn run(config: &Config, _self_test: bool) -> DiagnosticReport {
    let requested_device = config.device.as_str();
    let mut report = DiagnosticReport {
        status: DiagnosticStatus::Healthy,
        package_version: env!("CARGO_PKG_VERSION"),
        cuda_compiled: cfg!(cuda),
        python_compiled: cfg!(feature = "python"),
        requested_device,
        actual_device: "cpu",
        compiled_cuda_architectures: env!("TALOS_CUDA_ARCH"),
        nvcc_version: env!("TALOS_CUDA_NVCC"),
        fallback_reason: None,
        cuda_device: None,
        checks: Vec::new(),
        runtime_stats: None,
    };

    #[cfg(cuda)]
    {
        run_cuda_diagnostics(config.device, _self_test, &mut report);
    }
    #[cfg(not(cuda))]
    {
        report.checks.push(DiagnosticCheck::fail(
            "CUDA build feature",
            "binary was built without --features cuda",
        ));
        report.fallback_reason = Some("CUDA support is not compiled into this binary".to_string());
        report.status = match config.device {
            ComputeDevice::Cpu => DiagnosticStatus::Healthy,
            ComputeDevice::Auto => DiagnosticStatus::Degraded,
            ComputeDevice::Cuda => DiagnosticStatus::Unhealthy,
        };
    }
    report
}

#[cfg(cuda)]
fn run_cuda_diagnostics(requested: ComputeDevice, self_test: bool, report: &mut DiagnosticReport) {
    report.checks.push(DiagnosticCheck::pass(
        "CUDA build feature",
        "CUDA support is compiled in",
    ));

    if let Err(error) = crate::cuda::init() {
        report.checks.push(DiagnosticCheck::fail(
            "CUDA runtime initialization",
            error.to_string(),
        ));
        report.fallback_reason = Some(error.to_string());
        report.status = match requested {
            ComputeDevice::Cpu => DiagnosticStatus::Healthy,
            ComputeDevice::Auto => DiagnosticStatus::Degraded,
            ComputeDevice::Cuda => DiagnosticStatus::Unhealthy,
        };
        report.runtime_stats = Some(crate::cuda::runtime_stats());
        return;
    }
    report.checks.push(DiagnosticCheck::pass(
        "CUDA runtime initialization",
        "driver, runtime, and cuBLAS initialized",
    ));

    let device = match crate::cuda::get_device_info(0) {
        Ok(device) => device,
        Err(error) => {
            report.checks.push(DiagnosticCheck::fail(
                "CUDA device query",
                error.to_string(),
            ));
            report.fallback_reason = Some(error.to_string());
            report.status = DiagnosticStatus::Unhealthy;
            report.runtime_stats = Some(crate::cuda::runtime_stats());
            return;
        }
    };
    report.actual_device = match requested {
        ComputeDevice::Cpu => "cpu",
        ComputeDevice::Cuda | ComputeDevice::Auto => "cuda",
    };

    let capability = device.compute_capability.0 * 10 + device.compute_capability.1;
    let capability_supported = capability >= 75;
    report.checks.push(if capability_supported {
        DiagnosticCheck::pass(
            "minimum compute capability",
            format!("sm_{capability} satisfies the CC 7.5 minimum"),
        )
    } else {
        DiagnosticCheck::fail(
            "minimum compute capability",
            format!("sm_{capability} is below the CC 7.5 minimum"),
        )
    });

    let code_compatible = compiled_code_supports(
        report.compiled_cuda_architectures,
        device.compute_capability,
    );
    report.checks.push(if code_compatible {
        DiagnosticCheck::pass(
            "compiled architecture coverage",
            format!(
                "{} covers sm_{capability}",
                report.compiled_cuda_architectures
            ),
        )
    } else {
        DiagnosticCheck::fail(
            "compiled architecture coverage",
            format!(
                "{} does not contain compatible SASS or PTX for sm_{capability}",
                report.compiled_cuda_architectures
            ),
        )
    });

    report.cuda_device = Some(CudaDeviceReport {
        id: device.id,
        name: device.name,
        compute_capability: format!(
            "{}.{}",
            device.compute_capability.0, device.compute_capability.1
        ),
        free_memory_bytes: device.free_memory,
        total_memory_bytes: device.total_memory,
        compiled_code_compatible: code_compatible,
    });

    if self_test && capability_supported && code_compatible {
        report.checks.extend(cuda_self_tests());
    }
    report.runtime_stats = Some(crate::cuda::runtime_stats());

    if report.checks.iter().any(|check| !check.passed) {
        report.status = DiagnosticStatus::Unhealthy;
    }
}

#[cfg(cuda)]
fn cuda_self_tests() -> Vec<DiagnosticCheck> {
    vec![
        catch_check("CUDA matmul", check_matmul),
        catch_check("CUDA activation", check_activation),
        catch_check("CUDA log-softmax", check_log_softmax),
        catch_check("CUDA backward", check_backward),
        catch_check("CUDA Adam optimizer", check_optimizer),
    ]
}

#[cfg(cuda)]
fn catch_check(
    name: &'static str,
    check: impl FnOnce() -> Result<String, String>,
) -> DiagnosticCheck {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)) {
        Ok(Ok(detail)) => DiagnosticCheck::pass(name, detail),
        Ok(Err(error)) => DiagnosticCheck::fail(name, error),
        Err(payload) => DiagnosticCheck::fail(name, panic_message(payload)),
    }
}

#[cfg(cuda)]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return format!("panicked: {message}");
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return format!("panicked: {message}");
    }
    "panicked with a non-string payload".to_string()
}

#[cfg(cuda)]
fn check_matmul() -> Result<String, String> {
    use crate::autograd::{Device, Tensor};

    let lhs = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let rhs = Tensor::new_f32(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let before = crate::cuda::runtime_stats();
    let output = lhs.matmul(&rhs);
    let values = output.data_to_f32_vec();
    let after = crate::cuda::runtime_stats();
    let expected = [4.0f32, 5.0, 10.0, 11.0];
    let numerically_correct = values
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-4);
    let used_cuda = output.device == Device::Cuda
        && after.matmul_successes == before.matmul_successes + 1
        && after.matmul_fallbacks() == before.matmul_fallbacks();
    if used_cuda && numerically_correct {
        Ok(format!("cuBLAS result verified: {values:?}"))
    } else {
        Err(format!(
            "expected CUDA success without fallback; device={:?}, values={values:?}, stats_before={before:?}, stats_after={after:?}",
            output.device
        ))
    }
}

#[cfg(cuda)]
fn check_activation() -> Result<String, String> {
    use crate::autograd::{Device, Tensor};

    let input = Tensor::new_f32(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let before = crate::cuda::runtime_stats();
    let output = input.gelu();
    let values = output.data_to_f32_vec();
    let after = crate::cuda::runtime_stats();
    let used_cuda = output.device == Device::Cuda
        && after.activation_successes == before.activation_successes + 1
        && after.activation_fallbacks() == before.activation_fallbacks();
    if used_cuda && values.iter().all(|value| value.is_finite()) {
        Ok(format!("GELU kernel result verified: {values:?}"))
    } else {
        Err(format!(
            "expected CUDA GELU without fallback; device={:?}, values={values:?}, stats_before={before:?}, stats_after={after:?}",
            output.device
        ))
    }
}

#[cfg(cuda)]
fn check_log_softmax() -> Result<String, String> {
    use crate::autograd::{Device, Tensor};

    let input = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let before = crate::cuda::runtime_stats();
    let output = input.log_softmax();
    let values = output.data_to_f32_vec();
    let after = crate::cuda::runtime_stats();
    let row_sums_are_one = values
        .chunks_exact(2)
        .all(|row| (row.iter().map(|value| value.exp()).sum::<f32>() - 1.0).abs() <= 1e-5);
    let used_cuda = output.device == Device::Cuda
        && after.log_softmax_successes == before.log_softmax_successes + 1
        && after.log_softmax_fallbacks() == before.log_softmax_fallbacks();
    if used_cuda && row_sums_are_one {
        Ok(format!("row normalization verified: {values:?}"))
    } else {
        Err(format!(
            "expected CUDA log-softmax without fallback; device={:?}, values={values:?}, stats_before={before:?}, stats_after={after:?}",
            output.device
        ))
    }
}

#[cfg(cuda)]
fn check_backward() -> Result<String, String> {
    use crate::autograd::Tensor;

    let lhs = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let rhs = Tensor::new_f32(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2])
        .to_cuda()
        .map_err(|error| error.to_string())?;
    let output = lhs.matmul(&rhs).sum();
    let before = crate::cuda::runtime_stats();
    output.backward();
    synchronize()?;
    let after = crate::cuda::runtime_stats();
    if after.backward_successes > before.backward_successes
        && after.backward_fallback_kernel == before.backward_fallback_kernel
    {
        Ok("matmul backward kernels completed without fallback".to_string())
    } else {
        Err(format!(
            "expected CUDA backward success without fallback; stats_before={before:?}, stats_after={after:?}"
        ))
    }
}

#[cfg(cuda)]
fn check_optimizer() -> Result<String, String> {
    use crate::cuda::memory::{alloc, copy_d2h, copy_h2d};

    let initial = [1.0f32, -2.0, 3.0, -4.0];
    let gradients = [0.1f32, -0.2, 0.3, -0.4];
    let params = alloc::<f32>(initial.len()).map_err(|error| error.to_string())?;
    let grads = alloc::<f32>(gradients.len()).map_err(|error| error.to_string())?;
    let first_moment = alloc::<f32>(initial.len()).map_err(|error| error.to_string())?;
    let second_moment = alloc::<f32>(initial.len()).map_err(|error| error.to_string())?;
    copy_h2d(&params, &initial).map_err(|error| error.to_string())?;
    copy_h2d(&grads, &gradients).map_err(|error| error.to_string())?;
    copy_h2d(&first_moment, &[0.0; 4]).map_err(|error| error.to_string())?;
    copy_h2d(&second_moment, &[0.0; 4]).map_err(|error| error.to_string())?;

    crate::cuda::record_optimizer_attempt();
    if let Err(error) = crate::cuda::kernels::adam_step_f32(
        &params,
        &grads,
        &first_moment,
        &second_moment,
        initial.len(),
        0.001,
        0.9,
        0.999,
        1e-8,
        0.0,
        0.1,
        0.001,
        1.0,
    ) {
        crate::cuda::record_optimizer_fallback();
        return Err(error.to_string());
    }
    synchronize()?;

    let mut updated = [0.0f32; 4];
    let mut moment = [0.0f32; 4];
    copy_d2h(&mut updated, &params).map_err(|error| error.to_string())?;
    copy_d2h(&mut moment, &first_moment).map_err(|error| error.to_string())?;
    let correct_direction =
        updated
            .iter()
            .zip(initial)
            .zip(gradients)
            .all(|((updated, initial), gradient)| {
                updated.is_finite()
                    && if gradient > 0.0 {
                        *updated < initial
                    } else {
                        *updated > initial
                    }
            });
    if !correct_direction
        || moment
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
    {
        crate::cuda::record_optimizer_fallback();
        return Err(format!(
            "Adam produced invalid state: params={updated:?}, first_moment={moment:?}"
        ));
    }
    crate::cuda::record_optimizer_success();
    Ok(format!(
        "parameter and moment updates verified: {updated:?}"
    ))
}

#[cfg(cuda)]
fn synchronize() -> Result<(), String> {
    let status = unsafe { crate::cuda::bindings::cudaDeviceSynchronize() };
    if status == 0 {
        Ok(())
    } else {
        Err(format!("cudaDeviceSynchronize failed with code {status}"))
    }
}

#[cfg(any(cuda, test))]
fn compiled_code_supports(specification: &str, device: (u32, u32)) -> bool {
    let device_capability = device.0 * 10 + device.1;
    specification.split(',').any(|token| {
        let token = token.trim();
        if let Some(value) = token.strip_prefix("sm_") {
            return value.parse::<u32>().ok() == Some(device_capability);
        }
        token
            .strip_prefix("compute_")
            .and_then(|value| value.split('(').next())
            .and_then(|value| value.parse::<u32>().ok())
            .is_some_and(|ptx_capability| ptx_capability <= device_capability)
    })
}

fn format_bytes(bytes: usize) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_coverage_accepts_exact_sass_and_compatible_ptx() {
        assert!(compiled_code_supports("sm_86,compute_89(PTX)", (8, 6)));
        assert!(compiled_code_supports("sm_86,compute_89(PTX)", (9, 0)));
        assert!(!compiled_code_supports("sm_86,compute_89(PTX)", (8, 0)));
    }

    #[test]
    fn cpu_only_status_respects_requested_device() {
        if cfg!(cuda) {
            return;
        }
        let cuda = Config {
            device: ComputeDevice::Cuda,
            ..Config::default()
        };
        let auto = Config {
            device: ComputeDevice::Auto,
            ..Config::default()
        };
        let cpu = Config {
            device: ComputeDevice::Cpu,
            ..Config::default()
        };
        assert_eq!(run(&cuda, false).status, DiagnosticStatus::Unhealthy);
        assert_eq!(run(&auto, false).status, DiagnosticStatus::Degraded);
        assert_eq!(run(&cpu, false).status, DiagnosticStatus::Healthy);
    }
}
