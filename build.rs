// build.rs - Compile CUDA .cu files with NVCC
// Usage: This file is automatically invoked by cargo before compiling the Rust code

use std::env;
use std::path::Path;

fn main() {
    // Declare expected cfgs unconditionally so rustc/rust-analyzer don't warn
    println!("cargo::rustc-check-cfg=cfg(cuda)");
    println!("cargo:rerun-if-changed=cuda/");

    // Only compile CUDA code when the "cuda" feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rustc-cfg=cuda");

    println!("Building with CUDA support...");

    // On Windows, NVCC requires MSVC's cl.exe in PATH.
    // Set the PATH to include MSVC bin directory before running NVCC.
    if cfg!(windows) {
        let msvc_bin = env::var("MSVC_BIN_DIR").unwrap_or_else(|_| {
            // Try to find MSVC automatically via VSINSTALLDIR
            let vs_install_dir = env::var("VSINSTALLDIR").ok();
            let toolset_version = env::var("VCToolsVersion").ok();

            if let (Some(vs_dir), Some(toolset)) = (&vs_install_dir, &toolset_version) {
                let path = format!(
                    "{}/VC/Tools/MSvc/{}/bin/Hostx64/x64",
                    vs_dir.trim_end_matches('\\'),
                    toolset
                );
                if Path::new(&path).exists() {
                    return path;
                }
            }

            // Fallback: search standard installation paths
            let possible_paths = [
                "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSvc",
                "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSvc",
                "C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSvc",
                "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSvc",
            ];

            for base in possible_paths {
                if let Ok(entries) = std::fs::read_dir(base) {
                    for entry in entries.flatten() {
                        if let Ok(name) = entry.file_name().into_string() {
                            let candidate = format!("{}/{}/bin/Hostx64/x64", base, name);
                            if Path::new(&candidate).exists() {
                                return candidate;
                            }
                        }
                    }
                }
            }

            eprintln!(
                "[build.rs] Warning: Could not find MSVC bin directory. Set MSVC_BIN_DIR environment variable."
            );
            String::new()
        });

        if !msvc_bin.is_empty() {
            let current_path = env::var("PATH").unwrap_or_default();
            let new_path = format!("{};{}", msvc_bin, current_path);
            env::set_var("PATH", new_path);
            println!("Updated PATH with MSVC: {}", msvc_bin);
        }
    }

    // Find NVCC - use NVCC env var or CUDA_PATH, or rely on PATH
    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        if cfg!(windows) {
            // Try CUDA_PATH environment variable first
            if let Ok(cuda_path) = env::var("CUDA_PATH") {
                format!("{}/bin/nvcc.exe", cuda_path)
            } else {
                // Fall back to PATH
                "nvcc".to_string()
            }
        } else {
            // Unix: rely on PATH
            "nvcc".to_string()
        }
    });

    println!("Using NVCC: {}", nvcc);

    let cuda_arch = env::var("CUDA_ARCH")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "sm_75".to_string());
    let cuda_arch_flag = format!("-arch={}", cuda_arch);
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:warning=Using CUDA architecture: {}", cuda_arch);

    // Parse sm_XX → major/minor for use as preprocessor constants in .cu files
    let (arch_major, arch_minor) = if let Some(stripped) = cuda_arch.strip_prefix("sm_") {
        let parts: Vec<&str> = stripped.split('_').collect();
        (
            parts.first().and_then(|s| s.parse().ok()).unwrap_or(75),
            parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
        )
    } else {
        (75, 0)
    };
    let arch_defines = format!(
        "-DCUDA_ARCH_MAJOR={} -DCUDA_ARCH_MINOR={}",
        arch_major, arch_minor
    );
    println!(
        "Defining CUDA arch constants: major={}, minor={}",
        arch_major, arch_minor
    );

    // CUDA source files
    let cuda_files = &[
        "cuda/common.cu",
        "cuda/matmul.cu",
        "cuda/softmax.cu",
        "cuda/gelu.cu",
        "cuda/rope.cu",
        "cuda/attention_output.cu",
        "cuda/backward.cu",
        "cuda/rmsnorm.cu",
    ];

    // Output directory for compiled objects
    let out_dir = env::var("OUT_DIR").unwrap();
    let obj_files: Vec<String> = cuda_files
        .iter()
        .map(|f| {
            let base = Path::file_stem(Path::new(f))
                .and_then(|s| s.to_str())
                .unwrap();
            let obj = format!("{}/{}.o", out_dir, base);
            let mut cmd = std::process::Command::new(&nvcc);
            cmd.arg("--compile")
                .arg("-x")
                .arg("cu")
                .arg("-O3")
                .arg("--use_fast_math")
                .arg("-I")
                .arg("cuda")
                .arg("-o")
                .arg(&obj)
                .arg(f)
                // Generate code for selected architecture
                .arg(&cuda_arch_flag);
            // Add defines
            cmd.arg("-D").arg("CUDA_VERSION=12000");
            // Pass CUDA arch as preprocessor constants (major/minor)
            for def in arch_defines.split_whitespace() {
                cmd.arg(def);
            }

            println!("Compiling {}...", f);
            let output = cmd.output().expect("Failed to execute nvcc");
            if !output.status.success() {
                eprintln!(
                    "NVCC Error compiling {}: {}",
                    f,
                    String::from_utf8_lossy(&output.stderr)
                );
                eprintln!("NVCC stdout: {}", String::from_utf8_lossy(&output.stdout));
                std::process::exit(1);
            }
            obj
        })
        .collect();

    // Build CUDA kernel library from object files
    let lib_path = if cfg!(windows) {
        format!("{}/cuda_lib.lib", out_dir)
    } else {
        format!("{}/libcuda_lib.so", out_dir)
    };

    // Find CUDA lib directory - use CUDA_LIB_DIR env var or CUDA_PATH.
    // Needed for runtime driver/cuBLAS linking from Rust target.
    let cuda_lib_dir = env::var("CUDA_LIB_DIR").unwrap_or_else(|_| {
        if cfg!(windows) {
            if let Ok(cuda_path) = env::var("CUDA_PATH") {
                format!("{}/lib/x64", cuda_path)
            } else {
                "D:/NvidiaDevTool/NVIDIA GPU Computing Toolkit/CUDA/lib/x64".to_string()
            }
        } else {
            "/usr/local/cuda/lib64".to_string()
        }
    });

    println!("Building CUDA kernel library {}...", lib_path);
    let mut link_cmd = std::process::Command::new(&nvcc);
    if cfg!(windows) {
        link_cmd
            .arg("-lib")
            .arg("-o")
            .arg(&lib_path)
            .args(&obj_files);
    } else {
        link_cmd
            .arg("-shared")
            .arg("-o")
            .arg(&lib_path)
            .args(&obj_files);
    }

    println!("Link command: {:?}", link_cmd);
    let output = link_cmd.output().expect("Failed to link CUDA library");
    if !output.status.success() {
        eprintln!(
            "NVCC Link Error: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!(
            "NVCC Link stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        std::process::exit(1);
    }

    // Tell cargo to link against the library
    println!("cargo:rustc-link-search=native={}", out_dir);
    // Add CUDA lib directory to search path
    if cfg!(windows) {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    }
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=static=cuda_lib");
    } else {
        println!("cargo:rustc-link-lib=dylib=cuda_lib");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Rebuild if CUDA files change
    for f in cuda_files {
        println!("cargo:rerun-if-changed={}", f);
    }
}
