# Talos-XII（终末地抽卡模拟工具）

**项目开发者**：zayoka · **开源协议**：MIT license

Talos-XII 是《明日方舟：终末地》的抽卡模拟与 F2P 资源规划工具，支持批量模拟、概率分析、数据采集与模型校准，帮助玩家规划嵌晶玉与寻访凭证的使用。

---

## 快速开始

```bash
git clone https://github.com/zayokami/Talos-XII.git
cd Talos-XII
cargo build --release
./target/release/talos_xii
```

或直接运行 F2P 分析：`cargo run --release f2p`

> 首次启动需加载/训练神经网络，约 30～45 秒，后续从缓存加载通常 &lt; 1 秒。

---

## 功能特性

* **抽卡模拟**：交互式单抽、批量模拟，可切换卡池、福利开关与 PPO 策略。
* **F2P 福利分析**：针对零氪/月卡玩家，计算仅靠免费资源获取 UP 角色的概率与额外成本。
* **数据采集与校准**：支持录入玩家实战抽卡记录，基于真实数据校准模型参数。
* **策略优化**：内置 PPO、DQN、Neural Luck Optimizer，探索最优抽卡决策。
* **高性能计算**：Rust + SIMD（AVX2/NEON）+ 多线程，百万级模拟秒级完成。
* **高度可配置**：卡池、概率、保底、线程数等均可在 [data/config.json](data/config.json) 中调整。

---

## 使用方法

### 命令行速查

| 子命令 | 说明 | 示例 |
|--------|------|------|
| `interactive` | 交互式模拟器（默认） | `cargo run --` 或 `cargo run -- interactive` |
| `simulate` | 批量模拟 | `cargo run -- simulate -n 1000 -p 100` |
| `f2p` | F2P 福利分析 | `cargo run -- f2p` |
| `benchmark` | 性能基准测试 | `cargo run -- benchmark` |
| `collect add` | 交互式录入玩家抽卡数据 | `cargo run -- collect add` |
| `collect import <file>` | 从 JSON 导入玩家数据 | `cargo run -- collect import data.json` |
| `collect stats` | 查看已采集数据统计 | `cargo run -- collect stats` |
| `train` | 基于采集数据校准/训练模型 | `cargo run -- train` |

| 全局选项 | 说明 |
|----------|------|
| `-c, --config <path>` | 配置文件路径（默认 `data/config.json`） |
| `-s, --seed <seed>` | 随机种子（可选） |
| `-f, --force` | 强制重新训练模型（忽略缓存） |

### 运行流程

1. 确保 `data/config.json` 存在（程序会自动加载）。
2. 运行可执行文件，等待输出 "Neural Core: Online" 及初始化完成。
3. 按提示输入抽数，或使用 `f2p` 等子命令获取分析报告。

### 交互模式指令

| 指令 | 说明 |
|------|------|
| `<数字>` | 本次抽数 |
| `p <n>` | 设置默认抽数 |
| `s <n>` | 设置默认模拟次数 |
| `w` | 切换福利开关 |
| `ppo` | 切换 PPO 策略 |
| `pool list` | 查看卡池列表 |
| `pool <id>` | 切换卡池 |
| `pool all` | 全部卡池并行 |
| `status` | 查看当前状态 |
| `info` | 查看卡池详情 |
| `history` | 查看模拟历史 |
| `h` / `help` | 显示帮助 |
| `q` | 退出 |

### 配置要点

配置位于 [data/config.json](data/config.json)，`_comment` 字段中有详细说明。常用参数：

| 参数 | 说明 |
|------|------|
| `pool_name` | 当前卡池名称 |
| `up_six` | 当期 UP 六星列表 |
| `active_pool` | 当前激活的池 ID（对应 `pools` 中的 `id`） |
| `pools` | 卡池列表，含角色 UP、武器 UP、常驻池等 |
| `six_stars` / `five_stars` / `four_stars` | 各星级可出干员列表 |
| `prob_6_base` | 六星基础概率（默认 0.008） |
| `soft_pity_start` / `small_pity_guarantee` / `big_pity_cumulative` | 软/小/大保底抽数 |
| `fast_init` | 快速初始化（开发调试用） |
| `f2p_sim_count` | F2P 分析模拟次数（0 为自动） |
| `worker_max_threads` / `worker_reserve_cores` | 线程池与预留核心数 |

---

## 开发与部署

### 环境要求

* **操作系统**：Windows 10/11 (x86_64) 或 Linux (x86_64/aarch64)
* **编译器**：Rust 1.89.0 或更高版本
* **内存**：建议 16GB RAM 或以上（大规模模拟需较大内存）
* **处理器**：支持 AVX2 (Intel/AMD) 或 NEON (ARM) 的 CPU 推荐

### 构建与运行

```bash
cargo build --release
./target/release/talos_xii --help
```

开发模式使用 `cargo build`，会启用 opt-level 3 以保证神经网络性能。构建时会根据 CPU 架构启用 SIMD 优化。

---

## 技术栈

* **编程语言**：Rust
* **并行计算**：Rayon
* **硬件加速**：Portable SIMD (AVX2 / NEON)
* **神经网络**：DBN（环境噪声）、PPO（抽卡策略）、DQN、Transformer / Linear（特征与决策）

---

## 性能与 ACHF

本项目针对大规模模拟做了优化：Rayon 并行、零拷贝热路径、SIMD 矩阵运算，主流硬件可达百万级模拟/秒。

### ACHF（Adaptive Cache-aware Hyper-Connections）是什么？

ACHF 是本项目用于训练与推理加速的一套「自适应稀疏低秩连接」机制。它不是单一算法，而是一组可组合的策略：通过低秩投影减少算子规模，通过门控稀疏化减少无效通道，再结合缓存与延迟统计做动态调参，最终在速度、稳定性、精度之间取得平衡。

**它解决的问题**

1. **算子太重**：大型矩阵乘法在 CPU 上容易成为瓶颈。
2. **访存太慢**：权重访问的缓存未命中会拖慢整体吞吐。
3. **固定超参不鲁棒**：不同机器与负载下，最佳稀疏度/投影频率不同。

**核心机制**

1. **低秩投影（Low-rank Projection）**：在权重矩阵上做行/列或行列联合投影（`proj_mode`），用低秩结构近似原矩阵，减少计算量与缓存压力。
2. **门控稀疏（Gating Sparsity）**：通过门控值控制通道是否参与计算，`g_min` 作为下限，避免过度稀疏导致不稳定。
3. **自适应调参（Adaptive Control）**：使用延迟 EMA 与稀疏采样统计（`cache_latency_*` / `cache_*`）调整门控与缓存策略，避免性能抖动。
4. **路径级开关（Path-level Toggle）**：可分别对 Attention、FFN、DQN 路径启用，避免影响对精度敏感的链路。

**运行流程**：根据 `proj_freq` 定期触发投影建立低秩近似 → 在前向计算中应用门控过滤贡献小的通道 → 采样运行时延迟并更新 EMA 统计 → 将更新后的策略用于后续训练或推理。

**配置**：位于 [data/config.json](data/config.json) 的 `achf` 字段。常用参数：

| 参数 | 说明 |
|------|------|
| `enabled` | 是否启用 ACHF |
| `mode` | `lite`（保守、低开销）或 `full`（激进、潜在更高收益） |
| `proj_mode` | `rowcol` / `row` / `col`，投影维度 |
| `proj_freq` | 投影频率，数值越小投影越频繁 |
| `g_min` | 门控下限，过低可能引入不稳定 |
| `gate_mode` | 门控更新策略，如 `grad_ema` |
| `apply_attn` / `apply_ffn` / `apply_dqn` | 按路径启用 |
| `infer_gate` | 推理阶段门控策略 |

**启用建议**：训练阶段可先用 `mode=lite` 与 `apply_ffn=true` 观察吞吐提升与稳定性。若模型震荡或收敛变慢，提高 `g_min` 或降低 `proj_freq`。对精度敏感的场景可仅开启 `apply_ffn`，保留 Attention 的完整计算。

---

## 常见问题

**Q：首次启动为什么较慢？**  
A：需要训练 DBN、DQN（50k 步）、PPO（200k 步）等模型。完成后会写入 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin`，后续启动从缓存加载，通常 &lt; 1 秒。

**Q：F2P 分析中 "Avg Extra Jade Cost: N/A" 是什么意思？**  
A：表示所有模拟均在免费抽内出 UP，没有产生额外氪金样本，因此无法计算平均值。说明免费资源足够，无需额外投入。

**Q：缓存文件可以删除吗？**  
A：可以。删除 `neural.cache`、`dqn.cache.bin`、`ppo.cache.bin` 后，下次运行会重新训练。仅当保底/概率机制发生变动时建议清缓存重训。

---

## 贡献与分支管理

欢迎参与开发，请遵循以下规范。

**代码提交**：采用 [Conventional Commits](https://www.conventionalcommits.org/) 格式（`feat:`、`fix:`、`docs:` 等）。提交前运行 `cargo fmt` 与 `cargo clippy -- -D warnings`。

**开发流程**：Fork → 创建 `feature/*` 分支 → 开发与测试（`cargo test`）→ 向 `dev` 发起 PR，描述中关联 Issue。PR 需通过 Code Review 与 CI。

**分支规范**：
* `main`：生产分支，仅从 `release`/`hotfix` 合并。
* `dev`：开发分支，`feature/*` 合并目标。
* `feature/*`：功能分支，从 `dev` 检出。
* `release/v*`：发布分支，从 `dev` 检出，合并回 `main` 与 `dev`。
* `hotfix/*`：热修复，从 `main` 检出，合并回 `main` 与 `dev`。

---

## 引用论文

* *DeepSeek mHC: Manifold-Constrained Hyper-Connections*（流形约束超连接，用于优化器设计）
* *Proximal Policy Optimization Algorithms* (OpenAI)

---

## 免责声明

本项目与《明日方舟：终末地》官方及上海鹰角网络科技有限公司无任何关联。本软件仅用于模拟与学习交流，模拟结果仅供参考，不代表游戏内实际概率。严禁用于宗教迷信或任何违法用途，用户需自行承担风险。

---

## 致谢

感谢上海鹰角网络科技有限公司带来的《明日方舟：终末地》，感谢开源社区提供的 Rust 生态支持，感谢杭州深度求索人工智能基础技术研究有限公司撰写的 *DeepSeek mHC* 论文。

**Copyright 2026 zayoka. 本项目基于 MIT 协议开源。**

如有问题请联系：yuokai1@163.com
