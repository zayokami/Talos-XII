# Talos-XII（终末地抽卡模拟器）

**项目开发者**：zayoka  
**开源协议**：GPL-3.0 license



## 简介
Talos-XII是一款使用了深度学习技术的《明日方舟：终末地》的抽卡模拟工具。用户可自定义抽数和选择是否使用福利资源，模拟在不同策略下的资源消耗与获取情况。

本工具旨在通过深度学习算法（如 PPO 强化学习与 DBN 深度信念网络）寻找最优抽卡策略，帮助玩家规划“嵌晶玉”与“寻访凭证”的使用。

> **注意**：由于需要加载神经网络模型并进行环境初始化，软件启动的时候需要一点时间，请耐心等待。



## 功能特性
*   **深度学习模拟**：内置 Neural Luck Optimizer，基于历史数据预测并调整模拟概率。
*   **策略优化**：支持 PPO（近端策略优化）与 DQN（深度 Q 网络）算法，探索最优抽卡决策。
*   **高性能计算**：底层核心使用 Rust 编写，支持 SIMD（AVX2/NEON）指令集加速，多线程并行模拟百万级局数。
*   **F2P 友好分析**：专门针对“零氪/月卡”玩家的福利资源（如签到、活动赠送）进行模拟，计算仅靠免费资源获取 UP 角色的概率。
*   **高度可配置**：支持自定义卡池信息（UP 干员、概率）、福利策略、硬件加速参数等。



## 使用方法
1.  确保 `data` 目录下存在配置文件 `config.json`（程序会自动加载，若无则使用默认配置）。
2.  运行可执行文件 `talos_xii`（Windows 下为 `talos_xii.exe`）。
3.  等待控制台输出 "Neural Core: Online" 及系统初始化完成。
4.  根据提示输入想要模拟的抽卡次数，或查看自动生成的 F2P 概率分析报告。
5.  程序将输出模拟结果，包括六星获得数、UP 歪率、平均消耗资源等统计信息。



## 开发与部署

### 环境要求
*   **操作系统**：Windows 10/11 (x86_64) 或 Linux (x86_64/aarch64)
*   **编译器**：Rust 1.75.0 或更高版本 (支持 `portable_simd` 特性)
*   **内存**：建议 16GB RAM 或以上 (大规模模拟需要较大内存)
*   **处理器**：支持 AVX2 (Intel/AMD) 或 NEON (ARM) 指令集的 CPU 推荐

### 构建指南
1.  **克隆仓库**
    ```bash
    git clone https://github.com/zayokami/Talos-XII.git
    cd Talos-XII
    ```

2.  **编译项目**
    *   **开发模式** (调试构建，速度较慢，含调试符号):
        ```bash
        cargo build
        ```
    *   **生产模式** (高性能优化，推荐):
        ```bash
        cargo build --release
        ```
    *   *注：构建脚本会自动检测 CPU 架构并启用相应的 SIMD 优化 (AVX2/NEON)。*

3.  **运行程序**
    ```bash
    cargo run --release
    ```
    或直接运行生成的可执行文件：
    ```bash
    ./target/release/talos_xii
    ```

### 部署配置
项目包含一个默认配置文件 `data/config.json`，您可以根据需求修改以下参数：

*   `pool_name`: 卡池名称
*   `up_six`: 当期 UP 六星干员列表
*   `prob_6_base`: 六星基础概率 (默认 0.008)
*   `fast_init`: 是否启用快速初始化模式 (true/false) - 设为 `true` 可显著加快启动速度，适合开发调试
*   `ppo_mode`: PPO 训练模式 ("auto" / "fast" / "balanced")
*   `worker_max_threads`: 线程池最大线程数 (0 为自动)

## 技术栈
*   **编程语言**：Rust
*   **并行计算**：Rayon (线程池与任务调度)
*   **硬件加速**：Portable SIMD (AVX2 / NEON)
*   **神经网络**：
    *   Deep Belief Network (DBN) - 环境噪声模拟
    *   Proximal Policy Optimization (PPO) - 抽卡策略代理
    *   Transformer / Linear Layers - 特征提取与决策
    
    

## 贡献与分支管理指南

欢迎参与 Talos-XII 的开发！为确保项目质量与协作效率，请遵循以下规范。

### 🤝 贡献准则 (Contributing Guidelines)

#### 1. 代码提交规范
*   **Commit Message**: 采用 [Conventional Commits](https://www.conventionalcommits.org/) 格式。
    *   `feat: 新增功能描述` (新特性)
    *   `fix: 修复问题描述` (Bug 修复)
    *   `docs: 文档更新` (文档变更)
    *   `perf: 性能优化` (代码性能改进)
    *   `refactor: 代码重构` (不改变逻辑的重构)
    *   `test: 测试相关` (增加或修改测试)
    *   *示例*: `feat(ppo): implement history buffer ring queue`
*   **代码风格**: 
    *   Rust 代码必须通过 `cargo fmt` 格式化。
    *   提交前请运行 `cargo clippy` 确保无严重警告。

#### 2. 开发流程
1.  **Fork 仓库**并克隆到本地。
2.  **创建功能分支**: `git checkout -b feature/your-feature-name`。
3.  **开发与测试**:
    *   编写代码并添加对应的单元测试。
    *   确保 `cargo test` 全部通过。
    *   对于核心算法变更，需验证数值一致性。
4.  **提交代码**: 遵循上述 Commit 规范。
5.  **Pull Request (PR)**:
    *   将代码推送到你的 Fork 仓库。
    *   向本仓库的 `dev` 分支发起 PR。
    *   PR 描述中请关联 Issue (如 `Closes #123`)。

#### 3. 审查与合并
*   所有 PR 需经过至少一次 Code Review。
*   CI (GitHub Actions) 必须全部通过。
*   合并策略通常采用 `Squash and Merge` 以保持主分支历史整洁。

---

### 🌿 分支使用规范 (Branch Usage Guidelines)

#### 核心分支
*   **`main` (保护分支)**: 
    *   生产环境分支，时刻保持可发布状态。
    *   仅允许从 `release` 或 `hotfix` 分支合并，**禁止直接 Push**。
*   **`dev` (保护分支)**:
    *   主要开发分支，包含最新特性。
    *   所有 `feature` 分支开发完成后合并至此。

#### 辅助分支
*   **功能分支 (`feature/*`)**:
    *   命名规范: `feature/<issue-id>-<brief-desc>` (例: `feature/42-add-simd-support`)
    *   从 `dev` 检出，完成后合并回 `dev`。
    *   合并后删除。
*   **发布分支 (`release/*`)**:
    *   命名规范: `release/v<version>` (例: `release/v1.0.0`)
    *   从 `dev` 检出，进行发布前的测试与文档准备。
    *   完成后分别合并至 `main` (打 Tag) 和 `dev`。
*   **热修复分支 (`hotfix/*`)**:
    *   命名规范: `hotfix/<issue-id>-<brief-desc>` (例: `hotfix/55-fix-crash-on-startup`)
    *   从 `main` 检出，用于修复生产环境严重 Bug。
    *   完成后分别合并至 `main` 和 `dev`。

### 🛠️ 常用命令示例

```bash
# 开始新功能开发
git checkout dev
git pull origin dev
git checkout -b feature/new-algorithm

# 提交更改
git add .
git commit -m "feat: implement new algorithm"

# 同步上游变更 (Rebase)
git fetch origin
git rebase origin/dev

# 推送分支
git push -u origin feature/new-algorithm
```

## 引用论文
*   *DeepSeek mHC: Manifold-Constrained Hyper-Connections* (参考了其中的流形约束超连接思想用于优化器设计)
*   *Proximal Policy Optimization Algorithms* (OpenAI)



## 免责声明
本软件与《明日方舟：终末地》官方和上海鹰角网络科技有限公司无任何关联。本软件仅用于模拟与学习交流，模拟结果仅供参考，不代表游戏内实际概率。

本软件严禁用于宗教迷信或任何违反相关法律的行为。用户需自行承担风险。




## 致谢
感谢上海鹰角网络科技有限公司带来的游戏《明日方舟：终末地》。
感谢开源社区提供的 Rust 生态支持。
感谢杭州深度求索人工智能基础技术研究有限公司撰写的论文 *DeepSeek mHC: Manifold-Constrained Hyper-Connections*，为本项目的优化器设计提供了重要参考。




**Copyright 2025 zayoka. All rights reserved.**

**本项目基于 GPL-3.0 协议开源，您可以在遵守协议的前提下自由使用、修改和分发本项目的代码。**
