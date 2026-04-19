# Talos-XII

[![CI](https://github.com/zayokami/Talos-XII/actions/workflows/ci.yml/badge.svg)](https://github.com/zayokami/Talos-XII/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rustc-1.89.0+-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**面向《明日方舟：终末地》的神经网络驱动抽卡模拟器**

---

## Talos-XII是什么？

大多数抽卡模拟器只是按概率表投骰子。Talos-XII 不是。

它在模拟开始前，会先训练一组神经网络来建模抽卡环境的不确定性：DBN（Deep Belief Network）拟合环境噪声分布，DQN（Dueling Q-Network）和 PPO（Proximal Policy Optimization）分别学习离散决策与连续策略。在这个基础上，模拟器不仅能"模拟抽卡结果"，还能回答更有价值的问题——比如"作为零氪玩家，我靠免费资源抽到 UP 角色的概率是多少"，或者"当前保底进度下继续抽还是等下期更划算"。

技术上，Talos-XII 是一个纯 Rust 单二进制程序。通过 SIMD（AVX2/AVX-512/NEON）指令集加速矩阵运算、Rayon 并行分发模拟任务，它可以在主流硬件上以每秒万次以上的速度完成模拟，同时保持结果的统计可靠性。

---

## 快速开始

环境要求：**Rust 1.89.0+**，建议 16GB 内存，支持 AVX2 的 CPU 可获得最佳性能。

```bash
git clone https://github.com/zayokami/Talos-XII.git
cd Talos-XII
cargo build --release
./target/release/talos_xii
```

首次启动时，程序会训练 DBN、DQN（50k 步）和 PPO（200k 步）模型，大约需要 30～45 秒。训练完成后模型会缓存到 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin`，之后启动通常不到 1 秒。

### CUDA 支持（可选）

项目支持可选 CUDA 编译：

```bash
cargo check --features cuda
cargo run --features cuda -- simulate -n 1000 -p 100
```

`data/config.json` 中可通过 `device` 控制设备选择：

- `cpu`：强制 CPU
- `cuda`：优先 CUDA，不可用时自动回退 CPU
- `auto`：自动探测，优先 CUDA

当二进制未启用 `cuda` feature 时，`cuda/auto` 会自动回退到 CPU。
运行时会输出明确的设备初始化/回退原因，便于定位 CUDA 环境与配置问题。
在启用 `cuda` feature 的构建中，`Tensor` 的 CUDA 路径已覆盖 `matmul`（cuBLAS）与 `relu/gelu/softmax`（CUDA kernel），遇到运行时错误会记录原因并回退到 CPU 路径。

启动成功会看到类似输出：

```
[INFO] [Neural Core] Cache detected. Cached weights loaded.
[INFO] [DQN] Cached model loaded.
[INFO] [PPO] Cached model loaded.
Neural Core: Online
```

---

## 使用方法

Talos-XII 提供多个子命令，覆盖从快速体验到深度分析的不同场景。所有子命令都支持 `-c <path>` 指定配置文件（默认 `data/config.json`）、`-s <seed>` 固定随机种子、`-f` 强制重训模型忽略缓存。

### 交互模式

不带参数直接运行即进入交互模式，这是最直观的体验方式：

```bash
cargo run --release
```

在交互模式中，输入数字即为本次抽卡数量。除此之外支持以下指令：

- `p <n>` / `s <n>` — 设置默认抽数和模拟次数
- `w` — 切换福利（免费资源）开关
- `ppo` — 切换 PPO 策略（开启后由 PPO 模型辅助决策）
- `pool list` — 列出所有可用卡池
- `pool <id>` — 切换到指定卡池
- `pool all` — 对全部卡池并行模拟
- `status` / `info` / `history` — 查看状态、卡池详情、历史记录
- `h` — 帮助，`q` — 退出

### 批量模拟

当你需要大量统计数据时，用 `simulate` 子命令：

```bash
cargo run --release -- simulate -n 1000 -p 100
```

这会运行 1000 次模拟，每次 100 抽，最终输出平均 6 星数、平均 UP 数等统计结果。`-n` 控制模拟次数，`-p` 控制每次抽数。

### F2P 分析

这是 Talos-XII 最核心的功能之一，回答"零氪/月卡玩家靠免费资源能不能拿到 UP"：

```bash
cargo run --release -- f2p
```

程序会基于当前卡池配置，模拟大量免费资源场景（release 模式下默认百万次），输出：

- **F2P 获取 UP 概率** — 仅靠免费抽获得 UP 角色的百分比
- **期望 UP 数量** — 平均能获得多少个 UP
- **额外嵌晶玉成本** — 如果免费资源不够，平均还需要多少额外投入

如果输出中 "Avg Extra Jade Cost" 显示为 N/A，说明所有模拟都在免费抽内出了 UP，无需额外投入。

### 数据采集与模型校准

Talos-XII 支持录入真实玩家的抽卡记录，并用这些数据来校准模型参数，让模拟更贴近实际体验：

```bash
cargo run -- collect add          # 交互式录入一次抽卡记录
cargo run -- collect import data.json  # 从 JSON 文件批量导入
cargo run -- collect stats        # 查看已采集数据的统计概览
cargo run -- train                # 用采集的数据校准模型
```

校准后的参数会自动保存，下次运行时加载。

### 性能基准测试

```bash
cargo run --release -- benchmark
```

运行内置基准测试，包括 10,000 次快速模拟（每次 200 抽）和 300 次详细模拟（每次 120 抽），输出类似：

```
[基准] simulate_fast: 10000 次模拟 200 抽，耗时 0.45s (22222 模拟/秒)
[基准] simulate_one: 300 次模拟 120 抽，耗时 0.12s (2500 模拟/秒)
```

### ACHF 论文级 Benchmark

用于为 ACHF 技术论文生成完整的实验数据和图表（SVG/PNG）：

```bash
cargo run --release -- benchmark paper                      # 运行全部 7 项实验（默认 3 次试验）
cargo run --release -- benchmark paper --trials 5           # 5 次独立试验，计算 mean/std/95%CI
cargo run --release -- benchmark paper --only ablation      # 仅运行消融实验
cargo run --release -- benchmark paper --format png         # 输出 PNG 格式
cargo run --release -- benchmark paper --output-dir results # 指定输出目录
```

每个实验默认运行 3 次独立试验（可通过 `--trials N` 调整），输出 mean +/- std 及 95% 置信区间。结果包含：

- **图表** — 带 error bars 的柱状图、训练曲线折线图、箱线图（SVG/PNG）
- **summary.json** — 结构化聚合数据（含 CI），可直接导入 LaTeX 或 matplotlib
- **{experiment}.csv** — 每次 trial 的原始数据
- **summary.txt** — 人类可读摘要

包含 7 项实验：

| 实验 | 说明 |
|---|---|
| ablation | ACHF 开/关消融对比（吞吐量 + 奖励曲线） |
| mode | lite vs full 模式对比 |
| path | Cached / LowRank / Dense 推理路径延迟分布（箱线图） |
| gate | 训练过程中 gate、g_min、grad_ema、sparsity、adaptive_bias 曲线 |
| scale | 不同 rank 下的吞吐量（含 ACHF 关闭基线） |
| apply | ACHF 应用于不同组件（FFN/Attention/DQN）的组合效果 |
| convergence | ACHF 开/关状态下的训练 loss + reward 收敛曲线 |

图表输出到 `bench_output/`（可通过 `--output-dir` 自定义）。

---

## 卡池与配置

所有配置集中在 [data/config.json](data/config.json) 中，文件内的 `_comment` 字段有逐项说明。

### 卡池系统

Talos-XII 支持四种卡池类型：**角色 UP 池**（限定角色概率提升）、**武器 UP 池**（限定武器概率提升）、**常驻池**（标准寻访）和**新手池**（启程寻访）。每个池有独立的 ID、名称、UP 对象和概率参数。

配置中的 `active_pool` 字段决定当前激活的卡池，`pools` 数组则包含所有可切换的池定义。已过期的池会标记 `is_archived: true`，但仍可在交互模式中手动切换用于回溯分析。

### 概率与保底

角色 UP 池的核心参数：基础 6 星概率 0.8%，65 抽起触发软保底（概率逐步提升），80 抽硬保底（必出 6 星），120 抽大保底（必出 UP 角色）。UP 角色在 6 星中的出率为 75%。

武器 UP 池的参数有所不同：基础 6 星概率 4%，40 抽硬保底，180 抽大保底，UP 武器出率 50%。

这些数值均可在配置中调整，方便适配游戏版本更新或测试不同假设。

---

## 技术架构

### 模拟引擎

模拟的核心逻辑在 `src/sim.rs` 中。每次模拟会构建一个 8 维特征向量（包含保底进度、环境噪声、连续未出 UP 次数等），将其送入神经网络获取决策建议。模拟引擎支持三种运行模式：

- **probability** — 纯概率模式，按配置的概率表直接投骰
- **dqn** — 由 Dueling Q-Network 提供离散动作建议
- **ppo** — 由 Actor-Critic 网络提供连续策略优化

引擎还支持快速推理路径（`fast_inference`），在批量模拟时跳过完整 Tensor 构建，使用预编译的快速预测函数和 KV 缓存来压缩推理开销。

### 神经网络

系统初始化时会依次训练四个组件：

1. **DBN**（8→16→8 架构）— 建模抽卡环境中的噪声分布，每次模拟开始前采样一组 (env_noise, env_bias) 作为环境参数
2. **NeuralLuckOptimizer** — 在 DBN 提供的环境上做线性回归拟合与流形 RL 优化，学习 8 维特征到"运气值"的映射
3. **DQN**（Dueling 架构，50k 训练步）— 将状态映射为离散动作的 Q 值，用于决定"抽还是不抽"
4. **PPO**（Actor-Critic + MLA Transformer，200k 训练步）— 学习连续的抽卡策略分布，是最重的也是最有表达力的模型

训练完成后模型缓存到磁盘。缓存是通用的——它们学习的是保底/概率机制下的决策策略，不依赖具体角色名，因此卡池更新后无需重训。

### ACHF（Adaptive Cache-aware Hyper-Connections）

ACHF 是本项目自研的训练与推理加速机制。它不是单一算法，而是一组可组合的策略，核心思路是：通过低秩投影缩减算子规模，通过门控稀疏化跳过贡献小的通道，再用缓存与延迟统计来动态调整参数，最终在速度、稳定性、精度之间找到平衡点。

**解决的问题：** 在 CPU 上跑神经网络，大矩阵乘法容易成为瓶颈；权重的缓存未命中会拖慢吞吐量；而且不同机器上的最佳稀疏度和投影频率各不相同，用固定超参很难兼顾。

**四个核心机制：**

- **低秩投影（Low-rank Projection）** — 对权重矩阵做行、列或行列联合投影（`proj_mode` 控制），用低秩近似替代原矩阵，减少计算量和缓存压力。投影频率由 `proj_freq` 控制。
- **门控稀疏（Gating Sparsity）** — 每个通道有一个门控值，低于阈值的通道直接跳过计算。`g_min` 设定门控下限，防止过度稀疏导致输出不稳定。
- **自适应调参（Adaptive Control）** — 运行时采样延迟统计，用 EMA 平滑后反馈到门控和缓存策略中，避免性能抖动。
- **路径级开关（Path-level Toggle）** — 可以分别对 Attention、FFN、DQN 三条路径启用或关闭 ACHF，保护对精度敏感的链路。

运行时流程：定期触发投影建立低秩近似 → 前向计算中门控过滤贡献小的通道 → 采样延迟更新 EMA 统计 → 将策略更新应用到后续计算。

配置位于 `data/config.json` 的 `achf` 字段。推荐起步方式：先用 `mode=lite` + `apply_ffn=true` 观察效果。如果出现震荡或收敛变慢，提高 `g_min` 或降低 `proj_freq`。精度敏感的场景下可只开 FFN 路径，保留 Attention 的完整计算。

### SIMD 加速

`src/simd.rs` 实现了多级 SIMD 分发：运行时检测 CPU 能力，自动选择最优指令集（x86_64 上是 Scalar → AVX2 → AVX2+FMA → AVX-512F；ARM 上是 NEON）。提供的操作涵盖向量点积、缩放行累加、FMA、ReLU、Softmax 等神经网络热路径中的常见运算。构建时 `.cargo/config.toml` 中的 `-C target-cpu=native` 会启用当前 CPU 的所有可用指令集。

### 技术栈

- **Rust** — 语言与核心框架
- **Rayon** — 数据并行
- **Portable SIMD** — AVX2 / AVX-512 / NEON 硬件加速
- **自研神经网络** — DBN、PPO（MLA Transformer）、DQN（Dueling）、NeuralLuckOptimizer
- **自研 Autograd** — 支持 matmul、conv2d、pool 的自动微分引擎
- **Mmap Tensor I/O** — 内存映射的高性能张量读写

---

## 测试

项目包含 68 个自动化测试，覆盖以下领域：

- **模拟逻辑** — 保底触发（大保底、边界值）、概率单调递增、UP 率为零时的行为、F2P 必出 UP 验证
- **神经网络** — DQN 训练产出有效 Q 值、PPO 快慢路径对齐、Actor-Critic 形状校验
- **ACHF** — 低秩缓存一致性、门控更新、稀疏跳过、自适应偏置追踪（共 9 个测试）
- **Transformer** — MLA 前向/反向、RoPE 反向、RMSNorm 反向、因果遮罩
- **编解码** — 二进制 roundtrip（基本类型、嵌套结构、大向量、流式 I/O、截断错误处理）
- **配置解析** — 自研 JSON 解析器的 Unicode 转义、科学计数法、嵌套结构等
- **校准** — Beta 后验、可信区间、正则化不完全 Beta 函数

运行全部测试：

```bash
cargo test
```

CI 在 Ubuntu、Windows、macOS 三个平台上执行测试和 Clippy 检查，并支持 ARM64（aarch64-linux）交叉编译验证。

---

## 贡献

欢迎参与开发。代码提交遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式（`feat:`、`fix:`、`docs:` 等），提交前请确保通过 `cargo fmt` 和 `cargo clippy -- -D warnings`。

开发流程：Fork → 从 `dev` 创建 `feature/*` 分支 → 开发并通过 `cargo test` → 向 `dev` 发起 PR。PR 需通过 Code Review 与 CI。

分支规范：`main` 为生产分支（仅从 release/hotfix 合并），`dev` 为开发分支，`feature/*` 从 dev 检出，`release/v*` 用于发布，`hotfix/*` 用于紧急修复。

---

## 常见问题

**首次启动为什么要等很久？**
因为需要训练 DBN、DQN、PPO 三个模型。完成后会写入缓存文件，之后启动不到 1 秒。如果想跳过等待用于开发调试，可以在配置中开启 `fast_init`。

**F2P 分析的 "Avg Extra Jade Cost: N/A" 是什么意思？**
所有模拟都在免费抽数内出了 UP，没有产生需要额外付费的样本，因此无法算平均值。这说明当前卡池条件下免费资源是够用的。

**缓存文件可以删吗？**
可以。删掉 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin` 后，下次运行会重新训练。一般只有在保底/概率机制发生变化时才需要清缓存。

---

## 引用论文

- *DeepSeek mHC: Manifold-Constrained Hyper-Connections* — 流形约束超连接，本项目优化器设计的理论基础
- *Proximal Policy Optimization Algorithms* (OpenAI) — PPO 算法
- *Embarrassingly Simple Self-Distillation Improves Code Generation* — 自蒸馏技术，用于 EMA teacher 更新与 Best-K Sampling

## 免责声明

本项目与《明日方舟：终末地》官方及上海鹰角网络科技有限公司无任何关联。本软件仅用于模拟与学习交流，模拟结果仅供参考，不代表游戏内实际概率。严禁用于宗教迷信或任何违法用途，用户需自行承担风险。

## 致谢

感谢上海鹰角网络科技有限公司带来的《明日方舟：终末地》，感谢开源社区提供的 Rust 生态支持，感谢杭州深度求索人工智能基础技术研究有限公司撰写的 *DeepSeek mHC* 论文。

---

**Copyright 2026 zayoka.** 本项目基于 MIT 协议开源。

如有问题请联系：yuokai1@163.com
