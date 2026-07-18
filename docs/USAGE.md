# Usage / 使用方法

> Split out from the main [README](../README.md). Full command reference for every Talos-XII subcommand.

All subcommands support `-c <path>` for config, `-s <seed>` for repeatable runs with a fixed config and worker topology, and `-f` to force model retraining.

## Configuration and Diagnostics

Validate configuration without initializing or training models:

```bash
cargo run -- config validate
```

The parser rejects unknown fields, wrong types, invalid enum strings, unsafe
dimensions/probabilities, and incompatible cross-field/ACHF combinations. Only
`_comment*` documentation fields are ignored. Relative paths are resolved from
the selected config file's directory.

Inspect the build and selected device:

```bash
cargo run -- doctor
cargo run --features cuda -- doctor --json
```

With CUDA enabled, the default doctor run executes matmul, GELU, log-softmax,
backward, and Adam on the GPU and fails if any operation falls back to CPU.
Use `--no-self-test` only when metadata-only inspection is intentional.

## Interactive Mode

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

## Batch Simulation

```bash
cargo run --release -- simulate -n 1000 -p 100
```

Runs 1000 simulations of 100 pulls each, outputting average 6-star counts, UP rates, etc.

## F2P Analysis

The core feature: "Can I get the UP character with free resources as a F2P player?"

```bash
cargo run --release -- f2p
```

Simulates massive free-resource scenarios (1M in release mode), outputting:
- **F2P UP Probability** — percentage of simulations where UP was obtained without spending
- **Expected UP Count** — average number of UP characters obtained
- **Extra Jade Cost** — average additional paid currency needed if free resources weren't enough

If "Avg Extra Jade Cost" is N/A, all simulations cleared UP within the free pull budget.

## Python Scripting (Optional)

The installable Python 3.9+ abi3 extension and the embedded interpreter expose
the same tensor/autograd API. Build a local extension with:

```bash
python -m pip install maturin
maturin develop --release
python -c "import talos_xii as tx; print(tx.__version__)"
```

The installable package also provides the PyTorch-style `tx.nn`, `tx.optim`,
and safe checkpoint APIs. It has no runtime dependency on PyTorch or NumPy:

```bash
python examples/python/nn_training.py cpu 120 model.txckpt
python examples/python/nn_training.py cuda 120 model-cuda.txckpt
```

The CUDA command requires a CUDA-enabled wheel and fails explicitly when the
runtime is unavailable. `Module.cuda()` and dtype conversion preserve Parameter
object identity, so existing optimizer references remain valid.

Enable the embedded PyO3 bridge to run a Python script inside the Talos-XII process:

```bash
cargo run --features python -- python <script.py> -- <args>
```

Example:

```bash
cargo run --features python -- python examples/python/autograd_minimal.py -- 1.0
```

The final `--` separates arguments passed to the script; inside Python they are available through `sys.argv[1:]`. The embedded interpreter exposes a built-in `talos_xii` module with `Tensor`, `tensor`, `full`, `zeros`, `ones`, `arange`, `eye`, `rand`, `randn`, dtype constants, scalar/Tensor arithmetic, and autograd operations such as `matmul`, `mse_loss`, `l2_loss`, `smooth_l1_loss`, cross-entropy losses, reductions, shape ops, normalization, pooling, convolution, `backward`, and `grad`.

To write your own script, create a normal `.py` file, import `talos_xii`, build tensors, compute a scalar loss, then call `backward()`:

```python
# scripts/my_train.py
import sys
import talos_xii as tx

target = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

x = tx.tensor([1.0, 2.0], [1, 2])
w = tx.tensor([0.25, -0.5], [2, 1], requires_grad=True)
y = x.matmul(w) + 0.1
loss = y.mse_loss(tx.tensor([target], [1, 1]))
loss.backward()

print("prediction:", y.item())
print("loss:", loss.item())
print("grad_w:", w.grad.tolist())
```

Run it from the project root:

```bash
cargo run --features python -- python scripts/my_train.py -- 1.0
```

The `talos_xii` package is the Python surface of the framework. Use `tx.tensor(...)` for your data, `tx.zeros/full/arange/eye/randn(...)` for common inputs, and normal Python operators like `x + 1.0`, `2.0 * x`, `x ** 2.0`, `x % 2.0`, and `abs(x)` for math. Tensor methods also cover common activation/special functions (`relu`, `gelu`, `relu6`, `elu`, `selu`, `softplus`, `softsign`, `sigmoid`, `tanh`, `sin`, `cos`, `acos`, `asin`, `atan`, `erf`, `erfc`, `sqrt`, `rsqrt`, `log1p`, `expm1`) plus `concat`, `split`, `strided_slice`, `l2_normalize`, `group_norm`, `instance_norm`, `batch_norm2d`, `avg_pool2d`, `max_pool2d`, `pooling`, `conv2d`, `conv2d_transpose`, `depthwise_conv2d`, `conv3d`, and `gemm`/`matmul`.

Operator names with explicit Grad or Backprop variants are handled through autograd: create trainable leaves with requires_grad=True, call loss.backward(), and then read the Tensor-valued tensor.grad property. SoftmaxV2 and LogSoftmaxV2 map to softmax(dim) and log_softmax(dim). Conv2DCompress is currently a compatibility alias for the standard conv2d math path; Talos-XII does not yet store a separate compressed convolution weight format.

No NumPy or PyTorch installation is required for Talos-XII tensor/autograd scripts. The optional bridge links against a local Python runtime supported by PyO3. Scripts are arbitrary Python code and are not sandboxed: they run with the same OS permissions as the Talos-XII process, can read/write files, read environment variables, import local modules, start child processes, perform network access, and terminate the process. Run only scripts you trust. The Python tensor constructors and shape-producing bridge APIs enforce per-tensor allocation guards, and exporting tensors to Python lists has a separate size guard.
Most tensor methods are also available as `tx.*` functional calls with the same names, so you can choose whichever style reads better in a script. Arithmetic is aligned too: `x.add(y)` / `tx.add(x, y)` match `x + y`, with the same pattern for `sub`, `mul`, `div`, and `neg`.

## Data Collection & Model Calibration

```bash
cargo run -- collect add          # interactive single-record entry
cargo run -- collect import data.json  # bulk import from JSON
cargo run -- collect stats        # view collected data statistics
cargo run -- train                # estimate and save calibrated pool parameters
```

## Benchmarks

```bash
cargo run --release -- benchmark
```

Runs the quick built-in benchmark: 500 fast simulations and 100 detailed simulations, each with 100 pulls.

## Paper-Grade ACHF Benchmark

Generates complete experimental data and charts (SVG/PNG) for the ACHF paper:

```bash
cargo run --release -- benchmark paper                      # all 9 experiments, 5 trials
cargo run --release -- benchmark paper --trials 5           # 5 independent trials w/ CI
cargo run --release -- benchmark paper --only ablation      # ablation study only
cargo run --release -- benchmark paper --format png         # PNG output
cargo run --release -- benchmark paper --output-dir results # custom output dir
```

Each experiment runs 5 trials by default (`--trials N` to adjust), outputting mean ± std and 95% CI. Output includes charts (SVG/PNG), `summary.json` (structured data for LaTeX/matplotlib), raw CSVs, and human-readable `summary.txt`.

Saved calibration overrides are validated and merged into the complete pool catalog before model cache selection, training, or inference on the next model-using startup.

9 experiments:

| Experiment | Description |
|---|---|
| `ablation` | ACHF on/off throughput + reward curves |
| `mode` | lite vs full mode comparison |
| `path` | Cached / Sparse / Dense inference path latency (boxplot) |
| `gate` | Reference-gate/floor, candidate admission/error, connection weight, and candidate-path curves |
| `scale` | Throughput vs rank (with ACHF-off baseline) |
| `apply` | ACHF applied to different components (FFN/Attention/DQN) |
| `convergence` | Training loss + reward convergence with ACHF on/off |
| `crossover` | Forced candidate-kernel crossover across dimensions and sparsities |
| `regime` | Guarded AMA vs plain EMA and fixed-path oracle across batch regimes |

Output goes to `bench_output/` (`--output-dir` to customize).

---

所有子命令支持 `-c <path>` 指定配置、`-s <seed>` 在配置和 worker 拓扑不变时复现实验、`-f` 强制重训模型。

## 配置与诊断

不初始化或训练模型，直接执行严格配置校验：

```bash
cargo run -- config validate
```

未知字段、错误类型、非法枚举、危险维度/概率以及不兼容的跨字段/ACHF 组合都会
报错；只有 `_comment*` 文档字段会忽略。相对路径以所选配置文件目录为基准。

检查构建特性和设备：

```bash
cargo run -- doctor
cargo run --features cuda -- doctor --json
```

CUDA 版默认会在 GPU 上真实执行 matmul、GELU、log-softmax、backward 和 Adam；
任何 CPU fallback 都会使自检失败。只有明确只看元数据时才使用
`--no-self-test`。

## 交互模式

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

## 批量模拟

```bash
cargo run --release -- simulate -n 1000 -p 100
```

运行 1000 次模拟，每次 100 抽，输出平均 6 星数、UP 率等统计结果。

## F2P 分析

Talos-XII 的核心功能：回答"零氪/月卡玩家靠免费资源能不能拿到 UP"。

```bash
cargo run --release -- f2p
```

基于当前卡池配置模拟大量免费资源场景（release 模式默认百万次），输出：
- **F2P 获取 UP 概率** — 仅靠免费抽获得 UP 角色的百分比
- **期望 UP 数量** — 平均能获得多少个 UP
- **额外嵌晶玉成本** — 免费资源不够时，平均还需多少额外投入

"Avg Extra Jade Cost" 显示 N/A 表示所有模拟都在免费抽内出了 UP，无需额外付费。

## Python 脚本支持（可选）

可安装的 Python 3.9+ abi3 extension 与嵌入式解释器暴露同一套 tensor/autograd
API。本地构建 extension：

```bash
python -m pip install maturin
maturin develop --release
python -c "import talos_xii as tx; print(tx.__version__)"
```

可安装包还提供 PyTorch 风格的 `tx.nn`、`tx.optim` 与安全 checkpoint API，
运行时不依赖 PyTorch 或 NumPy：

```bash
python examples/python/nn_training.py cpu 120 model.txckpt
python examples/python/nn_training.py cuda 120 model-cuda.txckpt
```

CUDA 命令要求安装 CUDA 版 wheel；运行时不可用会直接报错，不会静默回退。
`Module.cuda()` 与 dtype 转换保持 Parameter 对象身份，因此已有 optimizer
引用仍然有效。

启用可选 PyO3 桥接后，可在 Talos-XII 进程内执行 Python 脚本：

```bash
cargo run --features python -- python <script.py> -- <args>
```

示例：

```bash
cargo run --features python -- python examples/python/autograd_minimal.py -- 1.0
```

最后一个 `--` 用于分隔传给脚本的参数；在 Python 中可通过 `sys.argv[1:]` 读取。嵌入式解释器会提供内置 `talos_xii` 模块，包含 `Tensor`、`tensor`、`full`、`zeros`、`ones`、`arange`、`eye`、`rand`、`randn`、dtype 常量、标量/Tensor 混合运算，以及 `matmul`、`mse_loss`、`l2_loss`、`smooth_l1_loss`、交叉熵 loss、归约、shape 操作、归一化、池化、卷积、`backward`、`grad` 等 autograd 操作。

自定义脚本就是普通 `.py` 文件。先 `import talos_xii as tx`，再创建 Tensor，算出一个标量 loss，最后调用 `backward()`：

```python
# scripts/my_train.py
import sys
import talos_xii as tx

target = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

x = tx.tensor([1.0, 2.0], [1, 2])
w = tx.tensor([0.25, -0.5], [2, 1], requires_grad=True)
y = x.matmul(w) + 0.1
loss = y.mse_loss(tx.tensor([target], [1, 1]))
loss.backward()

print("prediction:", y.item())
print("loss:", loss.item())
print("grad_w:", w.grad.tolist())
```

在项目根目录运行：

```bash
cargo run --features python -- python scripts/my_train.py -- 1.0
```

`talos_xii` 是框架的 Python API。用 `tx.tensor(...)` 放入数据，用 `tx.zeros/full/arange/eye/randn(...)` 创建常见输入，也可以直接写 `x + 1.0`、`2.0 * x`、`x ** 2.0`、`x % 2.0`、`abs(x)` 这类普通 Python 数学表达式。Tensor 方法也覆盖常见激活/特殊函数（`relu`、`gelu`、`relu6`、`elu`、`selu`、`softplus`、`softsign`、`sigmoid`、`tanh`、`sin`、`cos`、`acos`、`asin`、`atan`、`erf`、`erfc`、`sqrt`、`rsqrt`、`log1p`、`expm1`），以及 `concat`、`split`、`strided_slice`、`l2_normalize`、`group_norm`、`instance_norm`、`batch_norm2d`、`avg_pool2d`、`max_pool2d`、`pooling`、`conv2d`、`conv2d_transpose`、`depthwise_conv2d`、`conv3d`、`gemm`/`matmul`。

带 Grad 或 Backprop 变体的算子通过 autograd 覆盖：用 requires_grad=True 创建可训练叶 Tensor，调用 loss.backward()，然后读取 Tensor 类型的 tensor.grad 属性。SoftmaxV2 和 LogSoftmaxV2 对应 softmax(dim) 与 log_softmax(dim)。Conv2DCompress 目前是标准 conv2d 数学路径的兼容别名；Talos-XII 还没有单独的压缩卷积权重存储格式。

编写 Talos-XII 张量/autograd 脚本不需要安装 NumPy 或 PyTorch。这个可选桥接会链接本机 PyO3 支持的 Python 运行时。脚本是任意 Python 代码，且不提供沙箱隔离：脚本会以 Talos-XII 进程相同的系统权限运行，可以读写文件、读取环境变量、导入本地模块、启动子进程、访问网络，也可以直接终止进程。请只运行可信脚本。Python Tensor 构造函数和会产生新 shape 的桥接 API 都加了单 Tensor 分配保护，导出到 Python list 也有单独的大小保护。
大多数 Tensor 方法也提供同名 `tx.*` functional 调用，例如 `x.relu()` 和 `tx.relu(x)` 都可以用，脚本里按可读性选择即可。算术 API 也保持一致：`x.add(y)` / `tx.add(x, y)` 等价于 `x + y`，`sub`、`mul`、`div`、`neg` 也是同一套规则。

## 数据采集与模型校准

```bash
cargo run -- collect add          # 交互式录入单条记录
cargo run -- collect import data.json  # 从 JSON 批量导入
cargo run -- collect stats        # 查看已采集数据统计
cargo run -- train                # 估计并保存校准后的卡池参数
```

## 性能基准测试

```bash
cargo run --release -- benchmark
```

运行快速内置基准：500 次快速模拟和 100 次详细模拟，每次 100 抽。

## ACHF 论文级 Benchmark

为 ACHF 技术论文生成完整实验数据和图表（SVG/PNG）：

```bash
cargo run --release -- benchmark paper                      # 全部 9 项实验，默认 5 次试验
cargo run --release -- benchmark paper --trials 5           # 5 次独立试验，计算 mean/std/95%CI
cargo run --release -- benchmark paper --only ablation      # 仅消融实验
cargo run --release -- benchmark paper --format png         # PNG 格式输出
cargo run --release -- benchmark paper --output-dir results # 指定输出目录
```

每项实验默认 5 次独立试验（`--trials N` 调整），输出 mean ± std 及 95% 置信区间。结果包含带 error bars 的柱状图、训练曲线、箱线图（SVG/PNG）、`summary.json`（结构化数据，可导入 LaTeX/matplotlib）、原始 CSV 和人类可读的 `summary.txt`。

保存的校准覆盖值会先经过合法性检查，并在下一次需要模型的启动中，于缓存选择、训练和推理之前合并到完整卡池目录。

9 项实验：

| 实验 | 说明 |
|---|---|
| `ablation` | ACHF 开/关消融对比（吞吐量 + 奖励曲线） |
| `mode` | lite vs full 模式对比 |
| `path` | Cached / Sparse / Dense 推理路径延迟分布（箱线图） |
| `gate` | reference gate/下限、候选准入/误差、连接权重与候选内部路径曲线 |
| `scale` | 不同 rank 下的吞吐量（含 ACHF 关闭基线） |
| `crossover` | 不同维度/稀疏率下强制候选执行核的交叉点 |
| `regime` | 不同 batch 区间下 guarded AMA、plain EMA 与固定路径 oracle 对比 |
| `apply` | ACHF 应用于不同组件（FFN/Attention/DQN）的组合效果 |
| `convergence` | ACHF 开/关状态下的训练 loss + reward 收敛曲线 |

图表输出到 `bench_output/`（`--output-dir` 自定义）。
