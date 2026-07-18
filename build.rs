use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const DEFAULT_CUDA_ARCH: &str = "sm_75,sm_80,sm_86,sm_89";

#[derive(Debug, Clone, Copy, Default)]
struct ArchCode {
    sass: bool,
    ptx: bool,
}

fn build_error(message: impl AsRef<str>) -> ! {
    panic!("CUDA build configuration error: {}", message.as_ref());
}

fn parse_compute_capability(token: &str) -> Result<u32, String> {
    let raw = token.trim().to_ascii_lowercase();
    let digits = raw
        .strip_prefix("sm_")
        .or_else(|| raw.strip_prefix("compute_"))
        .unwrap_or(&raw);

    if digits.len() < 2 || !digits.chars().all(|character| character.is_ascii_digit()) {
        return Err(format!(
            "invalid CUDA architecture '{token}'; expected values such as sm_86 or compute_89"
        ));
    }

    let capability = digits
        .parse::<u32>()
        .map_err(|_| format!("invalid CUDA architecture '{token}'"))?;
    if !(75..=999).contains(&capability) {
        return Err(format!(
            "CUDA architecture '{token}' is below the project minimum sm_75"
        ));
    }
    Ok(capability)
}

fn cuda_gencode_flags(specification: &str) -> Result<(Vec<String>, String), String> {
    let tokens = specification
        .split([',', ';', ' '])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err("CUDA_ARCH cannot be empty".to_string());
    }

    let explicitly_requests_ptx = tokens
        .iter()
        .any(|token| token.to_ascii_lowercase().starts_with("compute_"));
    let mut codes = BTreeMap::<u32, ArchCode>::new();
    for token in tokens {
        let normalized = token.to_ascii_lowercase();
        let capability = parse_compute_capability(&normalized)?;
        let code = codes.entry(capability).or_default();
        if normalized.starts_with("compute_") {
            code.ptx = true;
        } else {
            code.sass = true;
        }
    }

    if !explicitly_requests_ptx {
        let highest_sass = codes
            .iter()
            .rev()
            .find_map(|(capability, code)| code.sass.then_some(*capability))
            .ok_or_else(|| "CUDA_ARCH did not request any SASS or PTX code".to_string())?;
        codes.entry(highest_sass).or_default().ptx = true;
    }

    let mut flags = Vec::new();
    let mut display = Vec::new();
    for (capability, code) in codes {
        if code.sass {
            flags.push(format!(
                "-gencode=arch=compute_{capability},code=sm_{capability}"
            ));
            display.push(format!("sm_{capability}"));
        }
        if code.ptx {
            flags.push(format!(
                "-gencode=arch=compute_{capability},code=compute_{capability}"
            ));
            display.push(format!("compute_{capability}(PTX)"));
        }
    }
    Ok((flags, display.join(",")))
}

fn command_output(program: &Path, argument: &str) -> Result<Output, String> {
    Command::new(program)
        .arg(argument)
        .output()
        .map_err(|error| format!("failed to execute '{}': {error}", program.display()))
}

fn resolve_nvcc() -> PathBuf {
    if let Some(configured) = env::var_os("NVCC").filter(|value| !value.is_empty()) {
        return PathBuf::from(configured);
    }

    for variable in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = env::var_os(variable).filter(|value| !value.is_empty()) {
            let executable = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
            let candidate = PathBuf::from(root).join("bin").join(executable);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    PathBuf::from(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
}

fn validate_nvcc(nvcc: &Path) -> String {
    let output = command_output(nvcc, "--version").unwrap_or_else(|error| {
        build_error(format!(
            "{error}. Install CUDA Toolkit 12+, add nvcc to PATH, or set NVCC/CUDA_PATH"
        ))
    });
    if !output.status.success() {
        build_error(format!(
            "'{} --version' failed with status {}: {}",
            nvcc.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let release = stdout
        .lines()
        .find_map(|line| line.split_once("release ").map(|(_, value)| value))
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            build_error(format!(
                "could not determine CUDA Toolkit version from '{} --version'",
                nvcc.display()
            ))
        });
    let major = release
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or_else(|| build_error(format!("invalid NVCC release version '{release}'")));
    if major < 12 {
        build_error(format!(
            "CUDA Toolkit {release} is unsupported; install CUDA Toolkit 12.0 or newer"
        ));
    }

    stdout
        .lines()
        .last()
        .unwrap_or("unknown nvcc version")
        .trim()
        .to_string()
}

#[cfg(windows)]
fn latest_msvc_bin(installation_root: &Path) -> Option<PathBuf> {
    let toolchains = installation_root.join("VC").join("Tools").join("MSVC");
    let mut versions = std::fs::read_dir(toolchains)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    versions.sort_by_key(|entry| entry.file_name());
    versions
        .into_iter()
        .rev()
        .map(|entry| entry.path().join("bin").join("Hostx64").join("x64"))
        .find(|candidate| candidate.join("cl.exe").is_file())
}

#[cfg(windows)]
fn discover_msvc_bin() -> Option<PathBuf> {
    if let Some(configured) = env::var_os("MSVC_BIN_DIR").filter(|value| !value.is_empty()) {
        let configured = PathBuf::from(configured);
        if configured.join("cl.exe").is_file() {
            return Some(configured);
        }
        build_error(format!(
            "MSVC_BIN_DIR '{}' does not contain cl.exe",
            configured.display()
        ));
    }

    if let Some(tools_root) = env::var_os("VCToolsInstallDir").filter(|value| !value.is_empty()) {
        let candidate = PathBuf::from(tools_root)
            .join("bin")
            .join("Hostx64")
            .join("x64");
        if candidate.join("cl.exe").is_file() {
            return Some(candidate);
        }
    }

    let program_files_x86 = env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(program_files_x86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let installation = String::from_utf8_lossy(&output.stdout);
    latest_msvc_bin(Path::new(installation.trim()))
}

#[cfg(windows)]
fn prepend_msvc_to_path() {
    if Command::new("cl.exe").arg("/?").output().is_ok() {
        return;
    }

    if let Some(msvc_bin) = discover_msvc_bin() {
        let mut paths = vec![msvc_bin];
        paths.extend(env::split_paths(&env::var_os("PATH").unwrap_or_default()));
        let joined = env::join_paths(paths)
            .unwrap_or_else(|error| build_error(format!("invalid MSVC PATH: {error}")));
        env::set_var("PATH", joined);
    }
    if Command::new("cl.exe").arg("/?").output().is_err() {
        build_error("MSVC cl.exe was not found. Install the Visual Studio C++ build tools or set MSVC_BIN_DIR to VC/Tools/MSVC/<version>/bin/Hostx64/x64");
    }
}

#[cfg(not(windows))]
fn prepend_msvc_to_path() {}

fn nvcc_root(nvcc: &Path) -> Option<PathBuf> {
    let absolute = if nvcc.is_absolute() {
        nvcc.to_path_buf()
    } else {
        let path = env::var_os("PATH")?;
        env::split_paths(&path)
            .map(|directory| directory.join(nvcc))
            .find(|candidate| candidate.is_file())?
    };
    absolute.parent()?.parent().map(Path::to_path_buf)
}

fn cuda_library_candidates(nvcc: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for variable in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = env::var_os(variable).filter(|value| !value.is_empty()) {
            roots.push(PathBuf::from(root));
        }
    }
    if let Some(root) = nvcc_root(nvcc) {
        roots.push(root);
    }
    if cfg!(unix) {
        roots.push(PathBuf::from("/usr/local/cuda"));
    }

    let mut candidates = Vec::new();
    if let Some(explicit) = env::var_os("CUDA_LIB_DIR").filter(|value| !value.is_empty()) {
        candidates.push(PathBuf::from(explicit));
    }
    for root in roots {
        if cfg!(windows) {
            candidates.push(root.join("lib").join("x64"));
        } else {
            candidates.push(root.join("lib64"));
            candidates.push(root.join("targets").join("x86_64-linux").join("lib"));
            candidates.push(root.join("targets").join("aarch64-linux").join("lib"));
        }
    }

    let mut seen = BTreeSet::<OsString>::new();
    candidates
        .into_iter()
        .filter(|candidate| seen.insert(candidate.as_os_str().to_os_string()))
        .collect()
}

fn resolve_cuda_library_dir(nvcc: &Path) -> PathBuf {
    let candidates = cuda_library_candidates(nvcc);
    if let Some(directory) = candidates.iter().find(|candidate| candidate.is_dir()) {
        return directory.clone();
    }
    let searched = candidates
        .iter()
        .map(|path| format!("'{}'", path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    build_error(format!(
        "CUDA runtime library directory was not found (searched {searched}). Set CUDA_LIB_DIR"
    ));
}

fn run_checked(command: &mut Command, operation: &str) {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| build_error(format!("failed to {operation}: {error}")));
    if !output.status.success() {
        build_error(format!(
            "failed to {operation}\ncommand: {rendered}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda)");
    println!("cargo:rerun-if-changed=cuda");
    for variable in [
        "CUDA_ARCH",
        "CUDA_FAST_MATH",
        "CUDA_HOME",
        "CUDA_LIB_DIR",
        "CUDA_PATH",
        "MSVC_BIN_DIR",
        "NVCC",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        println!("cargo:rustc-env=TALOS_CUDA_ARCH=disabled");
        println!("cargo:rustc-env=TALOS_CUDA_NVCC=disabled");
        return;
    }
    if cfg!(target_os = "macos") {
        build_error("CUDA builds are unsupported on macOS");
    }
    println!("cargo:rustc-cfg=cuda");

    prepend_msvc_to_path();
    let nvcc = resolve_nvcc();
    let nvcc_version = validate_nvcc(&nvcc);
    let cuda_lib_dir = resolve_cuda_library_dir(&nvcc);
    let architecture_specification = env::var("CUDA_ARCH")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CUDA_ARCH.to_string());
    let (architecture_flags, architecture_display) =
        cuda_gencode_flags(&architecture_specification).unwrap_or_else(|error| build_error(error));
    let fast_math = matches!(
        env::var("CUDA_FAST_MATH").as_deref(),
        Ok("1" | "true" | "TRUE" | "on" | "ON")
    );

    println!("cargo:rustc-env=TALOS_CUDA_ARCH={architecture_display}");
    println!("cargo:rustc-env=TALOS_CUDA_NVCC={nvcc_version}");
    println!("cargo:warning=CUDA code generation: {architecture_display}");
    println!(
        "cargo:warning=CUDA fast math: {}",
        if fast_math { "enabled" } else { "disabled" }
    );

    let cuda_files = [
        "cuda/common.cu",
        "cuda/matmul.cu",
        "cuda/softmax.cu",
        "cuda/gelu.cu",
        "cuda/rope.cu",
        "cuda/attention_output.cu",
        "cuda/backward.cu",
        "cuda/rmsnorm.cu",
        "cuda/sparse.cu",
        "cuda/tensor_ops.cu",
    ];
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").unwrap_or_else(|| build_error("Cargo did not provide OUT_DIR")),
    );
    let object_extension = if cfg!(windows) { "obj" } else { "o" };
    let mut objects = Vec::with_capacity(cuda_files.len());

    for source in cuda_files {
        let stem = Path::new(source)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_else(|| build_error(format!("invalid CUDA source path '{source}'")));
        let object = out_dir.join(format!("{stem}.{object_extension}"));
        let mut command = Command::new(&nvcc);
        command
            .arg("--compile")
            .arg("-x")
            .arg("cu")
            .arg("-O3")
            .arg("-I")
            .arg("cuda")
            .arg("-o")
            .arg(&object)
            .arg(source);
        if fast_math {
            command.arg("--use_fast_math");
        }
        if cfg!(unix) {
            command.arg("-Xcompiler=-fPIC");
        }
        command.args(&architecture_flags);
        run_checked(&mut command, &format!("compile {source}"));
        println!("cargo:rerun-if-changed={source}");
        objects.push(object);
    }

    let library = if cfg!(windows) {
        out_dir.join("cuda_lib.lib")
    } else {
        out_dir.join("libcuda_lib.a")
    };
    let mut archive = Command::new(&nvcc);
    archive.arg("--lib").arg("-o").arg(&library).args(&objects);
    run_checked(&mut archive, "archive CUDA kernels");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    let stubs = cuda_lib_dir.join("stubs");
    if stubs.is_dir() {
        println!("cargo:rustc-link-search=native={}", stubs.display());
    }
    println!("cargo:rustc-link-lib=static=cuda_lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
}
