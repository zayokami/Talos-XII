// build.rs - Compile CUDA .cu files with NVCC
// Usage: This file is automatically invoked by cargo before compiling the Rust code

use std::env;
use std::path::Path;

fn main() {
    // Only compile CUDA code when the "cuda" feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:rerun-if-changed=cuda/");
        return;
    }

    println!("Building with CUDA support...");

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

    // CUDA source files
    let cuda_files = &[
        "cuda/common.cu",
        "cuda/matmul.cu",
        "cuda/softmax.cu",
        "cuda/gelu.cu",
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
                // Generate position-independent code for linking
                .arg("--shared")
                // Generate code for current architecture if not specified
                .arg("-arch=sm_75");
            // Add defines
            cmd.arg("-D").arg("CUDA_VERSION=12000");

            println!("Compiling {}...", f);
            let output = cmd.output().expect("Failed to execute nvcc");
            if !output.status.success() {
                eprintln!("NVCC Error: {}", String::from_utf8_lossy(&output.stderr));
                std::process::exit(1);
            }
            obj
        })
        .collect();

    // Link object files into a shared library
    let lib_name = if cfg!(windows) {
        "cuda_lib.dll"
    } else {
        "libcuda_lib.so"
    };
    let lib_path = format!("{}/{}", out_dir, lib_name);

    println!("Linking to {}...", lib_path);
    let mut link_cmd = std::process::Command::new(&nvcc);
    link_cmd
        .arg("-shared")
        .arg("-o")
        .arg(&lib_path)
        .args(&obj_files)
        // Link CUDA runtime and cuBLAS
        .arg("-lcudart")
        .arg("-lcublas");

    let output = link_cmd.output().expect("Failed to link CUDA library");
    if !output.status.success() {
        eprintln!(
            "NVCC Link Error: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    // Tell cargo to link against the library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Rebuild if CUDA files change
    for f in cuda_files {
        println!("cargo:rerun-if-changed={}", f);
    }
}
