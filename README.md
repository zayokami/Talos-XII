# Talos-XII

[![CI](https://github.com/zayokami/Talos-XII/actions/workflows/ci.yml/badge.svg)](https://github.com/zayokami/Talos-XII/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rustc-1.89.0+-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A neural network-driven gacha pull simulator for Arknights: Endfield.**

---

Most gacha simulators just roll dice according to a probability table. Talos-XII doesn't.

Before each simulation, it trains a set of neural networks to model the uncertainty of the gacha environment: a DBN (Deep Belief Network) fits the environment noise distribution, while DQN (Dueling Q-Network) and PPO (Proximal Policy Optimization) learn discrete decisions and continuous strategy respectively. Based on this, the simulator can answer questions like "what's my probability of getting the UP character as a F2P player?" or "given my current pity progress, should I keep pulling or wait for the next banner?"

Technically, Talos-XII is a single Rust binary. It accelerates matrix operations via SIMD (AVX2/AVX-512/NEON) and parallelizes simulation tasks using Rayon, achieving over 10,000 simulations per second on mainstream hardware while maintaining statistical reliability.

---

**面向《明日方舟：终末地》的神经网络驱动抽卡模拟器。** 大多数抽卡模拟器只是按概率表投骰子，Talos-XII 不是。它在模拟前先训练 DBN 建模环境噪声、DQN 和 PPO 学习决策策略，能回答"零氪靠免费资源抽到 UP 的概率"这类更有价值的问题。

---

## Quick Start

**Requirements:** Rust 1.89.0+, 16GB RAM recommended. CPUs with AVX2 yield best performance.

```bash
git clone https://github.com/zayokami/Talos-XII.git
cd Talos-XII
cargo build --release
./target/release/talos_xii
```

On first launch, the program trains DBN, DQN (50k steps), and PPO (200k steps), taking ~30–45 seconds. Models are cached to `neural.cache`, `dqn.cache.bin`, `ppo.cache.bin`; subsequent launches complete in under 1 second.

### CUDA Support (Optional)

```bash
cargo check --features cuda
cargo run --features cuda -- simulate -n 1000 -p 100
```

Set the NVCC target architecture via `CUDA_ARCH` (default `sm_75`):

```bash
CUDA_ARCH=sm_86 cargo check --features cuda
```

Control device selection in `data/config.json` via the `device` field:
- `cpu` — force CPU
- `cuda` — prefer CUDA, fall back to CPU if unavailable
- `auto` — auto-detect, prefer CUDA

When the binary is built without the `cuda` feature, `cuda`/`auto` fall back to CPU automatically. Device initialization and fallback reasons are logged at runtime for easy debugging. CUDA paths cover `matmul` (cuBLAS) and `relu`/`gelu`/`softmax` (CUDA kernels); errors are logged and fallback to CPU is automatic.

---

## Usage

All subcommands support `-c <path>` for config, `-s <seed>` for reproducible runs, and `-f` to force model retraining.

### Interactive Mode

```bash
cargo run --release
```

Enter numbers to pull. Commands:
- `p <n>` / `s <n>` — set default pulls / simulation count
- `w` — toggle welfare (free resources) mode
- `ppo` — toggle PPO strategy assist
- `pool list` / `pool <id>` / `pool all` — list/switch/simulate all pools
- `status` / `info` / `history` — view state, pool details, pull history
- `h` — help, `q` — quit

### Batch Simulation

```bash
cargo run --release -- simulate -n 1000 -p 100
```

Runs 1000 simulations of 100 pulls each, outputting average 6-star counts, UP rates, etc.

### F2P Analysis

The core feature: "Can I get the UP character with free resources as a F2P player?"

```bash
cargo run --release -- f2p
```

Simulates massive free-resource scenarios (1M in release mode), outputting:
- **F2P UP Probability** — percentage of simulations where UP was obtained without spending
- **Expected UP Count** — average number of UP characters obtained
- **Extra Jade Cost** — average additional paid currency needed if free resources weren't enough

If "Avg Extra Jade Cost" is N/A, all simulations cleared UP within the free pull budget.

### Data Collection & Model Calibration

```bash
cargo run -- collect add          # interactive single-record entry
cargo run -- collect import data.json  # bulk import from JSON
cargo run -- collect stats        # view collected data statistics
cargo run -- train                # calibrate models with collected data
```

### Benchmarks

```bash
cargo run --release -- benchmark
```

Runs built-in benchmarks: 10,000 fast simulations (200 pulls each) and 300 detailed simulations (120 pulls each).

### Paper-Grade ACHF Benchmark

Generates complete experimental data and charts (SVG/PNG) for the ACHF paper:

```bash
cargo run --release -- benchmark paper                      # all 7 experiments, 3 trials
cargo run --release -- benchmark paper --trials 5           # 5 independent trials w/ CI
cargo run --release -- benchmark paper --only ablation      # ablation study only
cargo run --release -- benchmark paper --format png         # PNG output
cargo run --release -- benchmark paper --output-dir results # custom output dir
```

Each experiment runs 3 trials by default (`--trials N` to adjust), outputting mean ± std and 95% CI. Output includes charts (SVG/PNG), `summary.json` (structured data for LaTeX/matplotlib), raw CSVs, and human-readable `summary.txt`.

7 experiments:

| Experiment | Description |
|---|---|
| `ablation` | ACHF on/off throughput + reward curves |
| `mode` | lite vs full mode comparison |
| `path` | Cached / LowRank / Dense inference path latency (boxplot) |
| `gate` | Training curves: gate, g_min, grad_ema, sparsity, adaptive_bias |
| `scale` | Throughput vs rank (with ACHF-off baseline) |
| `apply` | ACHF applied to different components (FFN/Attention/DQN) |
| `convergence` | Training loss + reward convergence with ACHF on/off |

Output goes to `bench_output/` (`--output-dir` to customize).

---

## Pools & Configuration

All configuration is centralized in [data/config.json](data/config.json). Field-level documentation is in the `_comment` entries.

### Pool System

Talos-XII supports four pool types: **character UP** (limited rate-up), **weapon UP** (weapon rate-up), **standard** (regular banner), and **beginner**. Each pool has independent ID, name, UP targets, and probability parameters. The `active_pool` field selects the current pool; archived pools (`is_archived: true`) can still be switched to manually for retrospective analysis.

### Probabilities & Pity

Character UP pool: 0.8% base 6-star rate, soft pity at 65 (rate gradually increases), hard pity at 80 (guaranteed 6-star), mega pity at 120 (guaranteed UP character). UP rate within 6-star is 75%.

Weapon UP pool: 4% base 6-star rate, hard pity at 40, mega pity at 180, UP weapon rate 50%.

---

## Architecture

### Simulation Engine (`src/sim.rs`)

Each simulation constructs an 8-dimensional feature vector (pity progress, env noise, consecutive non-UP count, etc.) and feeds it to the neural network for decision guidance. Three modes:
- **probability** — pure dice roll per config probability table
- **dqn** — Dueling Q-Network provides discrete action suggestions
- **ppo** — Actor-Critic network provides continuous pull strategy optimization

The engine also supports a fast inference path (`fast_inference`) that skips full Tensor construction during batch simulation, using precompiled prediction functions and KV caching to minimize overhead.

### Neural Networks

On first run, four components are trained sequentially:

1. **DBN** (8→16→8) — models gacha environment noise distribution; samples (env_noise, env_bias) per simulation as environment parameters
2. **NeuralLuckOptimizer** — linear regression + manifold RL on DBN-provided environment; learns 8-dim → "luck value" mapping
3. **DQN** (Dueling, 50k steps) — maps state to discrete action Q-values; decides "pull or wait"
4. **PPO** (Actor-Critic + MLA Transformer, 200k steps) — learns continuous pull strategy distribution; the heaviest and most expressive model

Cached models are portable — they learn pity/probability mechanisms, not character names — so no retraining needed after pool updates.

### ACHF (Adaptive Cache-aware Hyper-Connections)

A proprietary training and inference acceleration system. Core idea: low-rank projection reduces operator size, gating sparsity skips low-contribution channels, and runtime latency feedback dynamically adjusts gating/projection parameters, finding the balance between speed, stability, and accuracy.

**Problems solved:** CPU-bound neural network matrix multiplication; cache misses degrading throughput; varying optimal sparsity/projection frequency across hardware.

**Four mechanisms:**
- **Low-rank Projection** — row/column/dual projection (`proj_mode`), low-rank approximation replaces original weight, reduces compute and cache pressure (`proj_freq` controls frequency)
- **Gating Sparsity** — channels with gate values below threshold are skipped; `g_min` sets a floor to prevent output instability
- **Adaptive Control** — runtime latency sampling with EMA smoothing feeds back into gating and caching policy, avoiding performance jitter
- **Path-level Toggle** — independently enable/disable ACHF on Attention, FFN, and DQN paths to protect accuracy-sensitive pathways

Recommended starting point: `mode=lite` + `apply_ffn=true`. If oscillation or slow convergence occurs, raise `g_min` or lower `proj_freq`. For accuracy-sensitive scenarios, enable only the FFN path.

### SIMD Acceleration (`src/simd.rs`)

Runtime CPU capability detection with automatic dispatch: Scalar → AVX2 → AVX2+FMA → AVX-512F on x86_64; NEON on ARM. Operations: vector dot product, scaled row accumulation, FMA, ReLU, Softmax. Build-time `-C target-cpu=native` in `.cargo/config.toml` enables all instruction sets available on the current CPU.

### Tech Stack

- **Rust** — language and core framework
- **Rayon** — data parallelism
- **Portable SIMD** — AVX2 / AVX-512 / NEON hardware acceleration
- **Custom Neural Networks** — DBN, PPO (MLA Transformer), DQN (Dueling), NeuralLuckOptimizer
- **Custom Autograd** — automatic differentiation engine supporting matmul, conv2d, pool
- **Mmap Tensor I/O** — memory-mapped high-performance tensor I/O

---

## Testing (105 tests)

```bash
cargo test
```

Covers: pity logic, probability monotonicity, F2P verification, DQN/Q-value validation, PPO fast/slow alignment, Actor-Critic shapes, ACHF consistency, Transformer MLA/RoPE/RMSNorm, binary codec, config parsing, Beta calibration, and more.

CI runs on Ubuntu, Windows, and macOS with ARM64 cross-compilation validation.

---

## Contributing

Code follows [Conventional Commits](https://www.conventionalcommits.org/) format (`feat:`, `fix:`, `docs:`, etc.). Run `cargo fmt` and `cargo clippy -- -D warnings` before committing.

Workflow: Fork → create `feature/*` from `dev` → develop and pass `cargo test` → PR to `dev`. Code Review and CI required.

Branch strategy: `main` is production (merged from release/hotfix only), `dev` is development, `feature/*` is branched from dev, `release/v*` for releases, `hotfix/*` for urgent fixes.

---

## FAQ

**Why is first launch so slow?**
Training DBN, DQN, and PPO models takes ~30–45 seconds. Results are cached; subsequent launches take under 1 second. Enable `fast_init` in config to skip training during development.

**What does "Avg Extra Jade Cost: N/A" mean in F2P analysis?**
All simulations obtained UP within the free pull budget — no paid spending was required. This means free resources are sufficient under current pool conditions.

**Can I delete the cache files?**
Yes. Deleting `neural.cache`, `dqn.cache.bin`, `ppo.cache.bin` triggers retraining on next run. Usually only needed when pity/probability mechanics change.

---

## References

- *DeepSeek mHC: Manifold-Constrained Hyper-Connections* — theoretical foundation for the optimizer design
- *Proximal Policy Optimization Algorithms* (OpenAI) — PPO algorithm
- *Embarrassingly Simple Self-Distillation Improves Code Generation* — self-distillation for EMA teacher updates and Best-K Sampling

---

## Disclaimer

This project has no affiliation with Arknights: Endfield or Hypergryph Co., Ltd. This software is for simulation and educational purposes only. Simulation results are for reference and do not represent actual in-game probabilities. Do not use for gambling, superstition, or any illegal activities. Users bear their own risk.

---

**Copyright 2026 zayoka.** MIT licensed.

Contact: yuokaki1@163.com

---

## 快速开始

**环境要求：** Rust 1.89.0+，建议 16GB 内存，支持 AVX2 的 CPU 性能最佳。

```bash
git clone https://github.com/zayokami/Talos-XII.git
cd Talos-XII
cargo build --release
./target/release/talos_xii
```

首次启动训练 DBN、DQN（50k 步）和 PPO（200k 步），约 30～45 秒。模型缓存至 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin`，之后启动不到 1 秒。

### CUDA 支持（可选）

```bash
cargo check --features cuda
cargo run --features cuda -- simulate -n 1000 -p 100
```

通过 `CUDA_ARCH` 环境变量指定 NVCC 架构（默认 `sm_75`）：

```bash
CUDA_ARCH=sm_86 cargo check --features cuda
```

在 `data/config.json` 中通过 `device` 字段控制设备：
- `cpu` — 强制 CPU
- `cuda` — 优先 CUDA，不可用时自动回退 CPU
- `auto` — 自动探测，优先 CUDA

未启用 `cuda` feature 时，`cuda`/`auto` 自动回退 CPU。设备初始化和回退原因会在运行时明确记录。CUDA 路径覆盖 `matmul`（cuBLAS）和 `relu`/`gelu`/`softmax`（CUDA kernel）；遇到运行时错误会记录原因并自动回退到 CPU。

---

## 使用方法

所有子命令支持 `-c <path>` 指定配置、`-s <seed>` 固定随机种子、`-f` 强制重训模型。

### 交互模式

```bash
cargo run --release
```

输入数字为本次抽卡数量，支持指令：
- `p <n>` / `s <n>` — 设置默认抽数和模拟次数
- `w` — 切换福利（免费资源）模式
- `ppo` — 切换 PPO 策略辅助
- `pool list` / `pool <id>` / `pool all` — 列出/切换/模拟全部卡池
- `status` / `info` / `history` — 查看状态、卡池详情、历史记录
- `h` — 帮助，`q` — 退出

### 批量模拟

```bash
cargo run --release -- simulate -n 1000 -p 100
```

运行 1000 次模拟，每次 100 抽，输出平均 6 星数、UP 率等统计结果。

### F2P 分析

Talos-XII 的核心功能：回答"零氪/月卡玩家靠免费资源能不能拿到 UP"。

```bash
cargo run --release -- f2p
```

基于当前卡池配置模拟大量免费资源场景（release 模式默认百万次），输出：
- **F2P 获取 UP 概率** — 仅靠免费抽获得 UP 角色的百分比
- **期望 UP 数量** — 平均能获得多少个 UP
- **额外嵌晶玉成本** — 免费资源不够时，平均还需多少额外投入

"Avg Extra Jade Cost" 显示 N/A 表示所有模拟都在免费抽内出了 UP，无需额外付费。

### 数据采集与模型校准

```bash
cargo run -- collect add          # 交互式录入单条记录
cargo run -- collect import data.json  # 从 JSON 批量导入
cargo run -- collect stats        # 查看已采集数据统计
cargo run -- train                # 用采集数据校准模型
```

### 性能基准测试

```bash
cargo run --release -- benchmark
```

运行内置基准：10,000 次快速模拟（每次 200 抽）和 300 次详细模拟（每次 120 抽）。

### ACHF 论文级 Benchmark

为 ACHF 技术论文生成完整实验数据和图表（SVG/PNG）：

```bash
cargo run --release -- benchmark paper                      # 全部 7 项实验，默认 3 次试验
cargo run --release -- benchmark paper --trials 5           # 5 次独立试验，计算 mean/std/95%CI
cargo run --release -- benchmark paper --only ablation      # 仅消融实验
cargo run --release -- benchmark paper --format png         # PNG 格式输出
cargo run --release -- benchmark paper --output-dir results # 指定输出目录
```

每项实验默认 3 次独立试验（`--trials N` 调整），输出 mean ± std 及 95% 置信区间。结果包含带 error bars 的柱状图、训练曲线、箱线图（SVG/PNG）、`summary.json`（结构化数据，可导入 LaTeX/matplotlib）、原始 CSV 和人类可读的 `summary.txt`。

7 项实验：

| 实验 | 说明 |
|---|---|
| `ablation` | ACHF 开/关消融对比（吞吐量 + 奖励曲线） |
| `mode` | lite vs full 模式对比 |
| `path` | Cached / LowRank / Dense 推理路径延迟分布（箱线图） |
| `gate` | 训练过程中 gate、g_min、grad_ema、sparsity、adaptive_bias 曲线 |
| `scale` | 不同 rank 下的吞吐量（含 ACHF 关闭基线） |
| `apply` | ACHF 应用于不同组件（FFN/Attention/DQN）的组合效果 |
| `convergence` | ACHF 开/关状态下的训练 loss + reward 收敛曲线 |

图表输出到 `bench_output/`（`--output-dir` 自定义）。

---

## 卡池与配置

所有配置集中在 [data/config.json](data/config.json)，`_comment` 字段有逐项说明。

### 卡池系统

四种卡池类型：**角色 UP 池**、**武器 UP 池**、**常驻池**和**新手池**。每个池有独立 ID、名称、UP 对象和概率参数。`active_pool` 决定当前激活的池，`pools` 数组包含所有可切换的池定义。已归档的池（`is_archived: true`）仍可在交互模式中手动切换用于回溯分析。

### 概率与保底

角色 UP 池：基础 6 星概率 0.8%，65 抽起软保底（概率逐步提升），80 抽硬保底，120 抽大保底必出 UP 角色，UP 在 6 星中占 75%。

武器 UP 池：基础 6 星概率 4%，40 抽硬保底，180 抽大保底，UP 武器占 50%。

---

## 技术架构

### 模拟引擎（`src/sim.rs`）

每次模拟构建 8 维特征向量（保底进度、环境噪声、连续未出 UP 次数等），送入神经网络获取决策建议。三种模式：
- **probability** — 纯概率，按配置的概率表投骰
- **dqn** — Dueling Q-Network 提供离散动作建议
- **ppo** — Actor-Critic 网络提供连续抽卡策略优化

引擎还支持快速推理路径（`fast_inference`），批量模拟时跳过完整 Tensor 构建，使用预编译快速预测函数和 KV 缓存压缩推理开销。

### 神经网络

初始化时依次训练四个组件：

1. **DBN**（8→16→8）— 建模抽卡环境噪声分布，每次模拟采样一组 (env_noise, env_bias) 作为环境参数
2. **NeuralLuckOptimizer** — 在 DBN 提供的环境上做线性回归与流形 RL 优化，学习 8 维特征到"运气值"的映射
3. **DQN**（Dueling，50k 步）— 将状态映射为离散动作 Q 值，决定"抽还是不抽"
4. **PPO**（Actor-Critic + MLA Transformer，200k 步）— 学习连续抽卡策略分布，是最重也是最有表达力的模型

训练完成后缓存到磁盘。缓存是通用的——它们学习的是保底/概率机制，不依赖角色名，因此卡池更新后无需重训。

### ACHF（Adaptive Cache-aware Hyper-Connections）

自研训练与推理加速机制：通过低秩投影缩减算子规模，通过门控稀疏化跳过贡献小的通道，通过缓存与延迟统计动态调整参数，在速度、稳定性、精度间找到平衡。

**解决的问题：** CPU 上神经网络大矩阵乘法的瓶颈；缓存未命中拖慢吞吐量；不同机器最佳稀疏度和投影频率各异，固定超参难以兼顾。

**四个核心机制：**
- **低秩投影** — 对权重矩阵做行/列/行列联合投影（`proj_mode`），低秩近似替代原矩阵，减少计算和缓存压力（`proj_freq` 控制频率）
- **门控稀疏** — 通道门控值低于阈值的直接跳过计算，`g_min` 设定下限防止过度稀疏导致输出不稳定
- **自适应调参** — 运行时采样延迟统计，EMA 平滑后反馈到门控和缓存策略，避免性能抖动
- **路径级开关** — 可分别对 Attention、FFN、DQN 启用/关闭 ACHF，保护对精度敏感的链路

推荐起步：`mode=lite` + `apply_ffn=true`。出现震荡或收敛变慢时，提高 `g_min` 或降低 `proj_freq`。精度敏感场景可只开 FFN 路径。

### SIMD 加速（`src/simd.rs`）

运行时 CPU 能力检测，自动选择最优指令集：x86_64 上为 Scalar → AVX2 → AVX2+FMA → AVX-512F；ARM 上为 NEON。涵盖向量点积、缩放行累加、FMA、ReLU、Softmax 等常见运算。构建时 `.cargo/config.toml` 中 `-C target-cpu=native` 启用当前 CPU 所有可用指令集。

### 技术栈

- **Rust** — 语言与核心框架
- **Rayon** — 数据并行
- **Portable SIMD** — AVX2 / AVX-512 / NEON 硬件加速
- **自研神经网络** — DBN、PPO（MLA Transformer）、DQN（Dueling）、NeuralLuckOptimizer
- **自研 Autograd** — 支持 matmul、conv2d、pool 的自动微分引擎
- **Mmap Tensor I/O** — 内存映射的高性能张量读写

---

## 测试（105 个测试）

```bash
cargo test
```

覆盖：保底逻辑、概率单调性、F2P 验证、DQN Q 值有效性、PPO 快慢路径对齐、Actor-Critic 形状、ACHF 一致性、Transformer MLA/RoPE/RMSNorm、二进制编解码、配置解析、Beta 校准等。CI 在 Ubuntu、Windows、macOS 上执行，并验证 ARM64 交叉编译。

---

## 贡献

代码遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式。提交前请确保通过 `cargo fmt` 和 `cargo clippy -- -D warnings`。

开发流程：Fork → 从 `dev` 创建 `feature/*` 分支 → 开发并通过 `cargo test` → 向 `dev` 发起 PR，需要通过 Code Review 与 CI。

分支规范：`main` 为生产分支，`dev` 为开发分支，`feature/*` 从 dev 检出，`release/v*` 用于发布，`hotfix/*` 用于紧急修复。

---

## 常见问题

**首次启动为什么要等很久？**
需要训练 DBN、DQN、PPO 三个模型。完成后写入缓存，之后启动不到 1 秒。开发调试可开启配置中的 `fast_init` 跳过训练。

**F2P 分析的 "Avg Extra Jade Cost: N/A" 是什么意思？**
所有模拟都在免费抽数内出了 UP，没有产生需要额外付费的样本，说明当前卡池条件下免费资源够用。

**可以删除缓存文件吗？**
可以。删除 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin` 后下次运行会重新训练，一般只在保底/概率机制变化时才需要清缓存。

---

## 引用论文

- *DeepSeek mHC: Manifold-Constrained Hyper-Connections* — 流形约束超连接，本项目优化器设计的核心理论
- *Proximal Policy Optimization Algorithms* (OpenAI) — PPO 算法
- *Embarrassingly Simple Self-Distillation Improves Code Generation* — 自蒸馏技术，用于 EMA teacher 更新和 Best-K Sampling

---

## 免责声明

本项目与《明日方舟：终末地》官方及上海鹰角网络科技有限公司无任何关联。本软件仅用于模拟与学习交流，模拟结果仅供参考，不代表游戏内实际概率。严禁用于宗教迷信或任何违法用途，用户需自行承担风险。

---

**Copyright 2026 zayoka.** MIT 协议开源。

联系方式：yuokaki1@163.com
