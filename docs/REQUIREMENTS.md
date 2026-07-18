# System Requirements / 系统要求

> Split out from the main [README](../README.md). Detailed hardware and platform requirements for building and running Talos-XII.

## Minimum & Recommended (CPU-only build)

| Item | Minimum | Recommended |
|---|---|---|
| CPU | Any x86_64 or ARM64 (including Apple Silicon) | x86_64 with AVX2+FMA |
| RAM | 1 GB free | 4 GB+ |
| Disk | 700 MB (exe + config) | 750 MB (with cache files) |
| OS | Windows 10+, macOS 11+, Linux kernel 4.0+ | — |

## GPU Build — Additional Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GPU with Compute Capability 7.5+ | RTX 20 series / GTX 16 series or newer |
| VRAM | 2 GB | 4 GB+ |
| CUDA Toolkit | 12.0+ | 12.0+ |
| NVIDIA Driver | 525+ | Latest stable |
| Host compiler | MSVC x64 build tools (Windows) or supported GCC/Clang (Linux) | Current vendor-supported toolchain |

GPU builds are **optional**. The CPU-only binary runs on all supported
platforms out of the box. The default CUDA fat binary contains `sm_75`,
`sm_80`, `sm_86`, and `sm_89` SASS plus `compute_89` PTX. CUDA Toolkit and
host compiler discovery are strict build-time checks; use `NVCC`,
`CUDA_PATH`/`CUDA_HOME`, `CUDA_LIB_DIR`, or `MSVC_BIN_DIR` only for explicit
overrides.

Initialization can fall back to CPU before training. Once GPU Adam starts a
step, an error is a hard training failure because earlier parameters may
already have changed. The optimizer is poisoned, training stops, and failed
state is not cached. Verify every deployed CUDA build with
`talos_xii doctor`; its operator self-tests treat CPU fallback as failure.
The official CUDA 12.8 archive requires a CUDA 12.8-compatible 570-series (or
newer) driver. A source build using CUDA 12.0 can use the 525-series minimum.

## Platform Details

**Windows (x86_64)**

| Item | Details |
|---|---|
| OS | Windows 10 1809+ / Windows 11 |
| Architecture | x86_64 (32-bit not supported) |
| Runtime deps (CPU) | None — statically linked, no MSVC runtime required |
| Runtime deps (GPU) | NVIDIA driver + CUDA Toolkit 12.0+ for compilation; `cudart.dll`/`cuda.dll`/`cublas.dll` required at runtime and are not bundled |
| Terminal | Windows Terminal / PowerShell / CMD — Windows Terminal recommended for full color output |
| SIMD | Auto-detected at runtime: AVX-512 → AVX2+FMA → Scalar |
| Thread affinity | Automatic core pinning via `SetThreadAffinityMask` |

**macOS (x86_64 / ARM64)**

| Item | Details |
|---|---|
| OS | macOS 11 Big Sur+ |
| Architecture | Apple Silicon (M1/M2/M3/M4) native ARM64, or Intel x86_64 |
| Runtime deps | None for CPU build |
| Terminal | Terminal.app / iTerm2 |
| SIMD | NEON on Apple Silicon; AVX2 on Intel Mac |
| Notes | Must compile from source (`cargo build --release`). GPU acceleration not available on macOS (no NVIDIA CUDA). |

**Linux (x86_64 / ARM64)**

| Item | Details |
|---|---|
| OS | Any mainstream distro (Ubuntu 18.04+, Debian 10+, CentOS 7+, Arch, etc.) |
| Architecture | x86_64 or ARM64 (Raspberry Pi 4/5 supported) |
| Runtime deps (CPU) | Source builds use the host glibc; official x86_64 archive requires glibc 2.35+, `libfontconfig.so.1`, and `libfreetype.so.6` (Ubuntu 22.04 level) |
| Runtime deps (GPU) | NVIDIA driver + CUDA Toolkit 12.0+ for compilation; `libcudart.so`/`libcuda.so`/`libcublas.so` required at runtime and are not bundled |
| Terminal | Any terminal with ANSI color support |
| SIMD | Same as Windows — runtime auto-detection |
| Notes | Must compile from source. GPU requires NVIDIA proprietary driver; Nouveau is not supported. |

---

## 最低 / 推荐配置（CPU 版）

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| CPU | 任意 x86_64 或 ARM64（含 Apple Silicon） | 支持 AVX2+FMA 的 x86_64 |
| 内存 | 1 GB 可用 | 4 GB + |
| 磁盘 | 700 MB（exe + config） | 750 MB（含缓存文件） |
| 操作系统 | Windows 10+、macOS 11+、Linux kernel 4.0+ | — |

## GPU 版 — 附加要求

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| GPU | NVIDIA GPU，计算能力 7.5+ | RTX 20 系列 / GTX 16 系列或更新 |
| 显存 | 2 GB | 4 GB+ |
| CUDA Toolkit | 12.0+ | 12.0+ |
| NVIDIA 驱动 | 525+ | 最新稳定版 |
| Host 编译器 | Windows 使用 MSVC x64 build tools，Linux 使用 CUDA 支持的 GCC/Clang | 当前厂商支持版本 |

GPU 版是**可选**的。默认 CUDA fat binary 包含 `sm_75`、`sm_80`、
`sm_86`、`sm_89` SASS 和 `compute_89` PTX。CUDA Toolkit 与 host compiler
在构建期严格检查；`NVCC`、`CUDA_PATH`/`CUDA_HOME`、`CUDA_LIB_DIR`、
`MSVC_BIN_DIR` 只用于显式覆盖。

训练开始前允许初始化回退 CPU。GPU Adam 步骤开始后，前面的参数可能已经更新，
因此任何错误都是训练硬失败：优化器会 poisoned、训练停止且失败状态不写缓存。
每个部署环境都应运行 `talos_xii doctor`；其算子自检出现 CPU fallback 会判失败。
官方 CUDA 12.8 包要求兼容 CUDA 12.8 的 570 系列或更新驱动；使用 CUDA 12.0
自行编译时可按 525 系列最低要求。

## 各平台详细说明

**Windows (x86_64)**

| 项目 | 说明 |
|---|---|
| 系统 | Windows 10 1809+ / Windows 11 |
| 架构 | x86_64（不支持 32 位） |
| 运行时依赖（CPU 版） | 无，静态链接，不需要 MSVC 运行库 |
| 运行时依赖（GPU 版） | NVIDIA 驱动 + CUDA Toolkit 12.0+（编译时）；运行时需 `cudart.dll`/`cuda.dll`/`cublas.dll`，发布包不捆绑这些库 |
| 终端 | Windows Terminal / PowerShell / CMD 均可，推荐 Windows Terminal（彩色输出更完整） |
| SIMD 加速 | 自动检测：有 AVX-512 走 AVX-512，有 AVX2+FMA 走 AVX2，都没有走标量 |
| 线程亲和性 | 通过 `SetThreadAffinityMask` 自动绑核 |

**macOS (x86_64 / ARM64)**

| 项目 | 说明 |
|---|---|
| 系统 | macOS 11 Big Sur+ |
| 架构 | Apple Silicon (M1/M2/M3/M4) 原生 ARM64，或 Intel Mac x86_64 |
| 运行时依赖 | 无（CPU 版） |
| 终端 | Terminal.app / iTerm2 |
| SIMD 加速 | Apple Silicon 走 NEON；Intel Mac 走 AVX2 |
| 注意事项 | 需自行编译（`cargo build --release`）。macOS 不支持 GPU 加速（无 NVIDIA CUDA）。 |

**Linux (x86_64 / ARM64)**

| 项目 | 说明 |
|---|---|
| 系统 | 任意主流发行版（Ubuntu 18.04+、Debian 10+、CentOS 7+、Arch 等） |
| 架构 | x86_64 或 ARM64（树莓派 4/5 也能跑） |
| 运行时依赖（CPU 版） | 源码构建使用 host glibc；官方 x86_64 包要求 glibc 2.35+、`libfontconfig.so.1`、`libfreetype.so.6`（Ubuntu 22.04 级别） |
| 运行时依赖（GPU 版） | NVIDIA 驱动 + CUDA Toolkit 12.0+（编译时）；运行时需 `libcudart.so`/`libcuda.so`/`libcublas.so`，发布包不捆绑这些库 |
| 终端 | 任意支持 ANSI 颜色的终端 |
| SIMD 加速 | 同 Windows，运行时自动检测 CPU 特征 |
| 注意事项 | 需自行编译。GPU 加速需要 NVIDIA 专有驱动；Nouveau 开源驱动暂不支持。 |
