# ACHF: Adaptive Cache-aware Hyper-Connections - A Guarded Runtime Design and Single-CPU Case Study

Draft status: working manuscript with the final three-process CPU experiment integrated. The immutable public raw-artifact URL, cross-hardware replication, and a training run that activates the candidate gate remain outstanding.

Author: Stephen F. Zhuang

## Abstract

Neural inference cost depends on memory layout, sparse-dispatch overhead, reuse, and device behavior as well as arithmetic. Adaptive Cache-aware Hyper-Connections (ACHF) separates three decisions that are often conflated: whether a semantic candidate is accurate enough to replace a stable reference, which equivalent physical realization should execute an admitted candidate, and whether an exact output may be reused. During training, a constrained connection coefficient and a candidate-specific gate define a soft reference/candidate blend. After calibration, frozen inference uses a hard reference/candidate decision followed by Guarded Adaptive Manifold Adaptation (AMA), a selector over prepared-dense, sparse, and ordinary-dense realizations. Exact memoization is orthogonal to both decisions.

We evaluate a Rust CPU implementation in three independent operating-system processes with five paired trials per process and Student-t intervals over process means (N=3). On a same-run exploratory admission frontier, the 51.001% candidate was the only tested point admitted in all 15 trials; its output-relative error was 4.084% [3.926%, 4.242%]. The measured runtime instantiation kept its training gate inactive. Relative to the dense reference, it reduced end-to-end throughput by 16.62% [-18.05%, -15.19%], added 14.225 seconds of training, and produced unresolved reward and loss differences. Its materialized inference state was 4.235 times the reference-parameter bytes. At the admitted point, Prepared Candidate was faster than Candidate CSR. Candidate CSR had nominal paired 95% intervals below both alternatives only at 0.95 and 0.99 sparsity in the exploratory crossover grid. Guarded AMA had a lower mean oracle gap than Plain EMA at seven of eight independently warmed operating points; paired intervals were below zero at six and unresolved at two. These results support guarded admission and conditional selector performance on this steady-state grid, not universal quality, efficiency, active training-gate, memoization-speedup, or regime-transition claims.

## 1. Introduction

Residual connections are one of the most reliable architectural devices in modern neural networks because they preserve an identity path while allowing learnable transformations to refine the representation. Hyper-connection style architectures generalize this idea by widening or diversifying the connection topology. This can improve expressivity, but it also changes the stability and memory-access profile of the network. Recent manifold-constrained hyper-connection work argues that unconstrained connection matrices can break signal conservation, and that projecting connection maps onto stable manifolds such as the Birkhoff polytope can restore well-conditioned propagation.

The Talos-XII runtime exposes a related but more deployment-focused problem. The project uses compact learned models to guide stochastic decision processes: an environment network, a neural luck optimizer, a dueling DQN, and a PPO actor-critic backed by a transformer. These models run inside a Rust CLI without an external ML framework. In this setting, model quality and latency are both first-class constraints. A small architectural extension that improves reward but destroys inference locality may be unacceptable; a sparse path that is mathematically attractive but consistently misses cache may be slower than the dense path.

ACHF addresses this practical gap by distinguishing a semantic candidate from a physical realization of that candidate. The former may change the represented function and must pass a quality predicate; the latter is one of several numerically equivalent ways to execute an admitted candidate. Exact input/output memoization is a third, orthogonal layer. The design asks three questions:

1. Can a dedicated connection map constrain the reference/candidate training blend without projecting signed operator weights?
2. Can a frozen quality predicate admit a semantic candidate before the runtime chooses among its physical realizations?
3. Can measured runtime state support guarded realization selection while exact reuse remains independently validated?

The current implementation supports a compact subset of this design. It does not reproduce the full dynamic Hpre/Hpost/Hres formulation of manifold-constrained hyper-connections, the proposed output-discrepancy-aware training gate, or the complete AMA economic bypass. Instead, it adapts constrained and observable connection state to a single-binary policy runtime with explicit candidate admission and physical realizations.

## Contributions

The manuscript makes the following contributions:

1. A reference/candidate ACHF layer with candidate-specific quality evidence and an explicit admission boundary.
2. A dedicated connection-map constraint whose Sinkhorn scaling is not conflated with signed operator weights, pruning, or low-rank candidate construction.
3. A proposed AMA controller with validity, baseline-relative economics, stale-measurement, and switching controls. The evaluated case study instantiates only its guarded ranking and probing subset, after candidate admission, with exact memoization kept independent and disabled in the paper benchmark.
4. A ten-experiment, three-process CPU case study covering admission, endpoint invariants, fixed-path crossover, and independently warmed selector operating points.
5. An evidence-bounded negative result: the runtime instantiation is slower than dense, the training gate is inactive by construction, projection utility is not isolated, and neither task-quality nor convergence improvement is established.

## 2. Background

### 2.1 Residual and Hyper-Connection Stability

A standard residual block can be written as:

```text
x_{l+1} = x_l + F_l(x_l)
```

The direct identity term helps preserve signal and gradient flow across depth. Hyper-connection variants extend this by introducing learnable connection maps around the residual branch. In the broad form:

```text
x_{l+1} = H_res x_l + H_post^T F_l(H_pre x_l)
```

The additional maps increase topological flexibility, but unconstrained `H_res`, `H_pre`, and `H_post` can amplify or cancel signals. The mHC reference paper constrains the residual map with Sinkhorn-Knopp projection and nonnegative input/output maps. The key lesson for ACHF is not that every runtime needs the full mHC architecture, but that connection operators should be constrained and measured if they are allowed to modify the identity path.

### 2.2 Deployment Bottlenecks in Compact ML Runtimes

In a compact runtime, the cost model differs from a large framework deployment. Many calls are small, batch sizes may vary, and CPU cache behavior can dominate arithmetic throughput. A pruned matrix can be slower than a dense one if the sparse path has poor locality or extra dispatch overhead. Conversely, a prepared dense realization can outperform sparse traversal for repeated shapes. Therefore ACHF treats inference path selection as a runtime decision rather than a static compile-time assumption.

## 3. ACHF Method

### 3.1 Layer Formulation

An ACHF layer contains a stable reference operator `F`, one or more semantic candidates `C_k`, dedicated connection logits, candidate-specific quality gates, an execution selector, and runtime state. For one candidate, let `rho_{k,t}` be the candidate coordinate of the constrained connection map. The training-time candidate mass and quality-layer output are:

```text
a_{k,t} = (1 - g_{k,t}) rho_{k,t}
y = (1 - a_{k,t}) F(x) + a_{k,t} C_k(x)
```

Here `g_{k,t}` is the reference-path gate for candidate `k`; `g=1` gives the reference path, while candidate participation also depends on the dedicated connection coordinate. No physical realization index appears in this training blend. Frozen inference instead makes a hard reference/candidate decision. Only after a candidate passes calibration and admission does the runtime choose a Prepared Candidate, Candidate CSR, or Candidate Dense realization of that same candidate. Exact memoized reuse is a third, independent fast path. These roles are not encoded in one shared tensor or one overloaded gate.

The reference remains available when a candidate is absent, stale, malformed, or outside the configured error and storage boundary. In the current case study, the primary candidate is a calibrated magnitude-pruned operator; the formalism also permits low-rank or quantized semantic candidates, while device-specialized kernels remain physical realizations of a fixed candidate.

### 3.2 Candidate-quality-aware Gate

In the proposed gate, optimization stability is summarized by an EMA `q_t` of a gradient RMS or Fisher-style proxy. That statistic cannot establish candidate quality, so each candidate also has an output-discrepancy signal:

```text
d_{k,t} = ||F(x) - C_k(x)||_2 / (||F(x)||_2 + epsilon)
```

The reference coefficient is candidate-specific:

```text
g_{k,t} = clip(sigmoid(alpha_k + beta_q,k q_t + beta_d,k EMA(d_{k,t})),
               g_min,t, 1)
```

Warmup, transition windows, and a reference floor prevent abrupt removal of the stable path. Sparse, low-rank, and quantized semantic candidates need not share calibration coefficients because their discrepancy modes differ; equivalent physical layouts of one candidate do share quality evidence.

The evaluated implementation does not instantiate this output-discrepancy-aware gate. Its target uses gradient/Fisher state and a multiplier derived from candidate weight error. When post-training calibration is enabled, every optimizer update leaves the candidate structurally ineligible, so the target gate returns the pure-reference value. The final trace therefore stays at `g=1` and records no candidate output-discrepancy samples. The case study evaluates reference-anchored training followed by post-training construction, calibration, hard admission, and frozen execution selection; it does not validate adaptive gate movement or candidate-aware training.

### 3.3 Dedicated Connection-map Constraint

The dedicated connection matrix supports three projection modes:

```text
none
rowcol
sinkhorn
```

The projected object is connection-mixing state, not the signed reference or candidate operator weight. The `rowcol` mode performs one row/column matrix-scaling iteration. The `sinkhorn` mode repeats that scaling for the configured number of steps, eight in the case study.

Given raw logits `Z`, the case-study implementation constructs a positive matrix:

```text
K[i,j] = exp(8 * tanh(Z[i,j]))
```

Then it alternates row and column normalization:

```text
M <- row_normalize(M)
M <- col_normalize(M)
```

For square matrices, both row and column targets are 1, giving an approximate doubly stochastic matrix. For rectangular matrices, ACHF keeps row sums at 1 and sets the column target to `rows / cols`, preserving total mass. At convergence this is a KL/Bregman projection of the positive kernel, not generally the Frobenius-nearest Euclidean projection; the finite implementation is therefore described as matrix scaling. It exports `connection_normalization_iterations`, `connection_row_max_deviation`, `connection_col_max_deviation`, `connection_min_value`, and `connection_negative_ratio` so the endpoint constraint can be checked.

Low-rank candidate construction is independent of connection projection. When `0 < r < min(rows, cols)`, the candidate operator can use:

```text
W_r = Q (Q^T W),   Q = subspace_iteration(W, r)
```

where `Q` is an orthonormal basis of the dominant r-dimensional left-singular subspace, computed with a Gaussian sketch and power iterations. This operation does not reuse the connection matrix and is not stacked onto Sinkhorn constraints on the same tensor. The layer records the candidate's relative Frobenius error:

```text
low_rank_rel_err = ||W - W_r||_F / ||W||_F
```

together with the applied rank in the ACHF state snapshot. Setting `rank = 0` or `rank >= min(rows, cols)` disables the constraint entirely, which preserves backward compatibility with earlier configurations.

### 3.4 Pruning and Freezing

After training, ACHF creates `W_s` with magnitude pruning:

```text
W_s[i,j] = 0       if abs(W[i,j]) < tau
W_s[i,j] = W[i,j] otherwise
```

`candidate_target_sparsity` or a magnitude threshold controls the sparse mask. `achf.rank` controls low-rank candidate construction only; it is never reused as an NNZ budget. The candidate mask, candidate weight, masked gradients, and masked Adam moments are updated together so optimizer state cannot silently regrow removed entries. The trainable reference remains a separate object.

`freeze_for_inference` finalizes the dedicated connection scaling, derives and calibrates the candidate from the reference, checks per-layer output error and storage economics, prepares equivalent Prepared Candidate/Candidate CSR/Candidate Dense layouts, and disables further connection updates. A candidate that fails admission returns to the reference path.

### 3.5 Cache-aware Inference Path Selection

ACHF supports three inference paths:

```text
Prepared Candidate: use a prepared contiguous dense view of the admitted candidate
Candidate CSR: use a CSR or device-mask view of the admitted candidate
Candidate Dense: execute the admitted candidate through its ordinary dense operator
```

The quality layer first chooses Reference or Candidate. Rejected or malformed candidates never enter the execution selector. Only after admission does AMA choose among the three mathematically equivalent candidate layouts. It checks shape, row count, input nonzero ratio, stale measurements, warmup state, and switching guards before committing a path.

Measured latency is tracked with short and long exponential moving averages. The effective latency score uses a normalized blend plus a cold-start penalty:

```text
ema_path <- lambda * ema_path + (1 - lambda) * latency_path
effective_path <- (1 - gamma) * short_path + gamma * long_path + cold_penalty_path
```

When both prepared and CSR measurements exist, ACHF can prefer the empirically faster realization. If adaptive layout bias is enabled, the bias is adjusted from their latency ratio:

```text
bias <- clamp(bias * (1 + eta * ratio), bias_min, bias_max)
```

where `ratio` blends short- and long-window latency ratios. The proposed AMA cost also includes lookup, selection, dispatch, and switching cost and admits a non-baseline realization only when its predicted saving exceeds a margin. The evaluated selector measures its own overhead but does not instantiate this complete baseline-relative rejection and still executes selection. Its results therefore test guarded ranking and probing, not a no-selector bypass guarantee.

To keep its latency estimates fresh, the selector periodically re-probes non-winning paths, since a path's cost can change as the workload shifts. A fixed re-probe cadence, however, taxes every call: it forces the runtime onto a known-slower path on a fixed schedule regardless of how far behind that path is, capping the winning path's selection share and inflating latency (Section 6.6.3). ACHF instead uses per-path exponential backoff: a path that currently scores N times worse than the leader is re-probed once every `min(N, C) * base` calls, for a small cap `C`. The interval is a pure function of the live scores, so a path that becomes competitive again — for example after a batch-size or sparsity change — immediately collapses its interval back to `base` and is tracked tightly. Because latency measurements are also keyed by an input-batch bucket, the selector maintains separate estimates for decode-like and prefill-like regimes rather than blending them into one average.

ACHF also includes exact memoized reuse for repeated frozen inputs. It keys reuse by layer, frozen-state epoch, dtype, shape, length, and every IEEE-754 input bit pattern, then verifies the stored length and complete input vector before reusing the final output. The hash is only a lookup filter; a collision cannot pass the equality check. Memoization does not choose a candidate or an execution path. It is disabled in every paper benchmark, so this manuscript makes no memoization-speedup claim.

## 4. Implementation

ACHF is implemented in Talos-XII, a Rust runtime with:

- Custom autograd tensors and operators.
- Dueling DQN and PPO actor-critic policy models.
- A transformer backbone with MLA-style attention components.
- Optional CUDA acceleration.
- BF16 inference cache generation.
- Python scripting support through PyO3.

ACHF can be enabled independently for attention, FFN, and DQN paths:

```text
achf.apply_attn
achf.apply_ffn
achf.apply_dqn
```

`apply_ffn` replaces the second FFN projection of each transformer block, `apply_dqn` replaces a hidden layer of the dueling Q-network, and `apply_attn` replaces the MLA output projection `w_o`. The output projection is the most regular, highest-FLOP Linear in the attention path, and it sits after the KV cache, so swapping it does not change cache semantics on the incremental decoding path.

FFN-only placement is the only measured placement with a nonnegative throughput point estimate, but its process-level interval crosses zero and its training cost is much higher. It is therefore the least unfavorable follow-up target, not a demonstrated recommendation. Attention-only placement reduces both throughput and measured reward in this run.

### 4.1 Proposed Mechanism Versus Evaluated Implementation

The proposal/evidence boundary is explicit. Evidence from one row below is not used to claim that an unevaluated mechanism in another row worked.

| Mechanism | Proposed method | Evaluated implementation and evidence |
|---|---|---|
| Training quality control | Output-discrepancy-aware soft reference/candidate gate | Gradient/Fisher and candidate-weight-error target; post-training calibration leaves the candidate ineligible after each update, so the gate stays on the reference path. Active gating is not evaluated. |
| Connection map | Dedicated constrained map contributes through `a=(1-g)rho` | A 2x2 scaled map satisfies endpoint row/column invariants. With one candidate, `g` and `rho` are not separately identifiable from `a`; because `g=1` in training and frozen hard inference bypasses the blend, projection utility is not evaluated. |
| Deployment admission | Candidate-specific held-out quality predicate | Post-training output calibration and hard per-layer admission are evaluated under the configured predicate. |
| Execution control | Validity, complete baseline-relative economic bypass, probing, hysteresis, and dwell controls | Guarded ranking, probing, warmup, stale tracking, hysteresis, and dwell are implemented. Selector cost is measured, but the complete baseline-relative economic bypass is not instantiated. |
| Realization equivalence | Frozen-epoch or periodically refreshed numerical certificate plus runtime fingerprint, shape, dtype, and device checks | Equivalent layouts derive from one frozen candidate and are invalidated with candidate/layout state; no per-call canonical double execution is claimed. |
| Exact output reuse | Full-input key, frozen-state epoch, exact stored-input check, and invalidation | The single-entry implementation checks hash, length, and complete input and invalidates on state change. Paper benchmarks disable memoization, so no reuse speedup is evaluated. |

## 5. Experimental Design

The completed paper benchmark was launched with:

```powershell
cargo run --release -- benchmark paper --trials 5 --processes 3 `
  --format svg --output-dir target/release/bench_output_final
```

It spawned three independent operating-system processes with five paired, deterministically seeded trials per process and rotating interleaved condition order. The suite contains ten experiments:

| Experiment | Question | Primary metrics |
|---|---|---|
| `ablation` | What does the measured runtime instantiation change end to end? | reward, loss, throughput, training time |
| `mode` | How do fixed and adaptive execution compare? | throughput, route diagnostics |
| `path` | Which admitted candidate realization is fastest? | fixed-path latency and tails |
| `gate` | Which gate, connection, and mask invariants are exercised? | gate, discrepancy, projection, optimizer mask |
| `scale` | Does candidate rank recover the overhead? | paired quality and runtime deltas |
| `apply` | Where does ACHF hurt or help? | FFN, attention, FFN+attention, DQN |
| `convergence` | Does a longer budget change the outcome? | reward, loss, throughput, training time |
| `admission` | Which sparse candidates may enter? | sparsity, output error, eligible layers |
| `crossover` | Where does the fastest fixed path change? | latency by dimension and sparsity |
| `regime` | How do selectors compare at warmed operating points? | Guarded/Plain EMA oracle gap |

The benchmark uses runtime overrides rather than the larger defaults in `data/config.json`:

| Item | Paper-benchmark value |
|---|---|
| Decision task | 32-feature stochastic control state; five offsets `{0,+0.005,+0.015,-0.005,-0.015}`. The implementation terminates on the target event or after 300 decisions; the active cumulative guarantee makes 120 the effective maximum. Evaluation uses 128 seeded episodes with greedy argmax. |
| Event process | Base rarity probabilities 0.008/0.080/0.912; soft threshold 65 with slope 0.05; hard threshold 80; target-event conditional rate 0.5. |
| Reward | -0.1 per decision, +10 target event, +2 non-target top-rarity event, loss-streak penalty `-2 * streak` from streak two, +5 for a target before decision 80 and another +5 before decision 50. Only a positive applied offset `a` incurs `-8a`; zero and negative offsets have no reward penalty, and negative actions separately refund control budget. |
| Policy backbone | Input 32, hidden 64, two layers, four heads, KV low-rank dimension 16, RoPE dimension 4, context length 8. |
| PPO | One environment; 2000 steps for standard task runs and 4000 for convergence/gate traces; 256 steps per update; two epochs; batch 128. Top-k 3 is used only for stochastic training rollout; evaluation is greedy. Discount 0.99, GAE 0.95, clip 0.2, value coefficient 0.5, entropy coefficient 0.01. DQN placement uses 5000 DQN steps under fast-init mode. |
| Optimizer | Adam at `3e-4`, linearly decayed to a `3e-5` floor. |
| Candidate predicate | Requested sparsity 0.51, minimum sparsity 0.50, maximum output-relative error 0.05, plus per-layer storage-economics and shape checks. |
| Calibration | At most 256 Adam steps at `1e-3` and 256 retained rollout sequences. Indices divisible by four form 64 validation sequences and the other 192 form training sequences. With context padded to 8, artifacts report 512 token rows per layer and 2048 across four layers; these are not independent sequences. Validation occurs at steps 0/64/128/192/256 with early stopping; main trials stopped at 128 or 192. The same validation subset controls early stopping and final admission. |

Timing configurations are likewise fixed before aggregation:

| Measurement | Configuration |
|---|---|
| End-to-end throughput | 100 warmup simulations, then 1000 simulations with exactly 100 decisions each. |
| Admitted fixed path | Input dimension 128; 100 warmup rounds; 1000 timed samples; 64 calls per sample. |
| Crossover grid | Dimensions 256/1024/2048, batch 32, sparsities 0.50/0.80/0.90/0.95/0.99, 20 warmup rounds, 200 timed samples. |
| Steady-state selector grid | Dimension 1024, batches 1/128, sparsities 0.80/0.90/0.95/0.98; 300 adaptive warmup calls followed by 1200 measured calls per operating point, with condition order rotated. The same synthetic layer is reused within a trial. Fixed-path oracle measurements use 40 warmup rounds followed by 60 timed samples per path with 20 calls per sample. |
| Replication and reuse | Three sequential OS processes with five paired trials each. Exact-output memoization is disabled throughout; Prepared Candidate is a layout, not a memo hit. |

### 5.1 Metrics

Five trials inside one process are not treated as five independent system repetitions. Each condition first contributes one mean per process; all aggregate values and paired contrasts then use the three process means as the statistical unit (`N=3`). Confidence intervals are two-sided 95% Student-t intervals with two degrees of freedom. Microbenchmark inner-loop samples are never pooled as independent observations. An interval crossing zero is unresolved, not evidence of equivalence.

### 5.2 Environment and Integrity

- CPU: 13th Gen Intel Core i9-13900HX, 32 logical processors.
- Memory: 34,158,272,512 bytes (31.8 GiB).
- OS: Microsoft Windows display version 25H2, build 26200.8875.
- Toolchain: Rust/Cargo 1.93.1, `x86_64-pc-windows-msvc`, release profile, `-C target-cpu=native`.
- Features: CPU only; CUDA and embedded Python disabled.
- Source: `https://github.com/zayokami/Talos-XII`, commit `4ee46aecd31ae5dde4869dbe78e08640fe487e8e`.
- Runtime: 4167.34, 4188.69, and 4081.67 seconds, approximately 3 h 27 min total.
- Integrity: three complete child manifests, 30 artifacts each, all 90 hashes verified by the parent.
- Cache policy: existing model caches were recorded but ignored; benchmark models were rebuilt from domain-separated seeds.
- Artifact status: the immutable public raw-artifact URL is pending upload and must be inserted before submission. `target/release/bench_output_final` is a local verification directory, not a public artifact.

All results therefore describe one hardware/software environment. The benchmark's internal readiness report marks software diagnostics and independent-process repetition complete but keeps cross-environment readiness false.

## 6. Results

### 6.1 Candidate Admission and Main Ablation

Admission is layer-specific: every candidate layer must satisfy output-error, shape, nonzero-count, calibration-sample, and storage-economics guards. The global candidate error is shown only as context.

| Requested sparsity | Actual | Global output error [95% CI] | All layers admitted |
|---:|---:|---:|---:|
| 0.50 | 0.500000 | 0.0395 [0.0360, 0.0431] | 0/15 |
| **0.51** | **0.510010** | **0.0408 [0.0393, 0.0424]** | **15/15** |
| 0.55 | 0.550049 | 0.0423 [0.0387, 0.0459] | 14/15 |
| 0.60 | 0.600016 | 0.0525 [0.0469, 0.0581] | 3/15 |
| 0.70-0.98 | measured grid | 0.0948-0.7807 | 0/15 at every point |

The exact 0.50 point is not storage-economical once CSR metadata is counted. Requested 0.51 is the conservative intersection selected after inspecting this same exploratory frontier: it has full admission in every trial and is used for the admitted-path measurements below, but it is not an independently validated operating point. The 0.55 point is a boundary diagnostic because one trial rejects at least one layer. Rejected-point throughput is mostly reference fallback and is not evidence for the rejected candidate. This demonstrates the configured benchmark predicate, not production safety or a speedup.

Main end-to-end results, mean [95% CI] over process means:

| Condition | Throughput (sim/s) | Reward | Loss | Train time (s) |
|---|---:|---:|---:|---:|
| Dense reference | 926.63 [900.55, 952.70] | 5.240 [4.691, 5.789] | 1.166 [0.985, 1.346] | 5.76 [5.60, 5.92] |
| ACHF runtime instantiation (training gate inactive) | 772.52 [737.66, 807.39] | 5.255 [4.808, 5.702] | 1.209 [1.100, 1.317] | 19.98 [18.37, 21.59] |
| Fixed-mask sparse training | 828.61 [819.38, 837.83] | 5.162 [4.297, 6.027] | 1.185 [0.673, 1.697] | 5.59 [5.41, 5.78] |
| Static magnitude pruning | 510.30 [486.15, 534.45] | 5.357 [4.959, 5.755] | 1.166 [0.985, 1.346] | 5.76 [5.60, 5.92] |

The ACHF runtime instantiation changes paired throughput by **-16.62% [-18.05, -15.19]**, training time by **+14.225 s [12.463, 15.988]**, reward by +0.0145 [-0.8043, 0.8333], and loss by +0.0429 [-0.2424, 0.3282]. Fixed-mask sparse training is 10.56% slower than dense; static magnitude pruning is 44.93% slower. The current implementation has no demonstrated quality, convergence, or general end-to-end efficiency advantage. Raw artifacts retain the legacy label `Full ACHF (guarded AMA)`.

Aggregate figure: `docs/media/ablation_aggregate.pdf`.

### 6.2 Path Latency

At admitted sparsity 0.510010 and input dimension 128:

| Candidate realization | Mean latency (ns) [95% CI] |
|---|---:|
| Prepared Candidate | **456.9 [439.6, 474.3]** |
| Candidate Dense | 1022.0 [967.5, 1076.4] |
| Candidate CSR | 2784.4 [2714.8, 2854.0] |

Candidate CSR is about 6.1x slower than Prepared Candidate at the admitted point. Admission therefore cannot be reused as an execution decision.

The stationary mode experiment reuses one trained policy. Fixed Prepared Candidate reaches 899.51 [890.84, 908.17] sim/s; Fixed Candidate Dense 793.92 [777.84, 810.00]; Fixed Candidate CSR 497.04 [484.08, 510.00]; Guarded AMA 776.20 [767.32, 785.09]; and Plain EMA 742.04 [730.37, 753.71]. Guarding beats Plain EMA in mean throughput but remains 13.70% below the known Fixed Prepared Candidate winner. Online selection is not free. The raw benchmark labels `Cached`, `Dense`, and `Sparse` denote these three layouts, not memoization or different semantic candidates.

### 6.3 Gate Dynamics and Projection Quality

The gate trace does **not** show adaptive candidate training. At all 16 recorded points in every process, the reference gate is 1.0, gate velocity is zero, candidate eligibility is zero, and candidate output-sample count is zero. The floor reaches 0.199764 at step 4000, but the candidate never enters the blend. This validates reference anchoring only; it cannot support a learned gate-adaptation claim.

The invariants that are exercised remain consistent:

- Sinkhorn applies only to the dedicated nonnegative connection matrix.
- Final row and column maximum deviations are both `1.49e-8` in every process mean.
- Connection negative ratio is zero, minimum entry is 0.01948, and normalization takes eight iterations.
- Maximum absolute masked candidate weight, masked gradient, and masked Adam moment are all zero after sparse calibration.

After freeze and admission, inference selects the candidate and routes 93.2-94.3% of candidate calls through Prepared Candidate, 0.9-1.0% through Candidate CSR, and 4.7-5.9% through Candidate Dense. Mean decision time is 236.1-263.4 ns across processes. Training-gate evidence and frozen execution evidence are intentionally kept separate. These endpoint invariants do not isolate projection utility: in the one-candidate blend, `g` and `rho` enter only through `(1-g)rho`, the measured gate remains one, and hard inference bypasses the soft coefficient.

The four instrumented ACHF layers materialize the following inference state:

| Runtime state | Bytes |
|---|---:|
| Reference weights and bias | 98,816 |
| Connection logits | 64 |
| Candidate dense weights | 98,304 |
| Prepared Candidate weights and bias | 98,816 |
| CSR row pointers | 1,552 |
| CSR columns | 48,168 |
| CSR values | 48,168 |
| Sparse mask | 24,576 |
| Memo input/output | 0 |
| **Total** | **418,464** |

This is 4.2347797927 times the 98,816 reference-parameter bytes for those four layers. It is not whole-model memory, KV-cache memory, allocator overhead, process RSS, or a measurement of a lazy-layout/eviction policy.

### 6.4 Rank Scaling

Every tested rank is slower than no ACHF. Paired throughput deltas are -7.54% for rank 8, -6.79% for rank 16, -6.46% for rank 32, -7.15% for rank 48, and -6.23% for the rank-64 no-op control; all process-level 95% intervals remain below zero. No reward or loss change is resolved. Because the no-op control has zero approximation error but retains the slowdown, fixed bookkeeping and dispatch overhead are material on this compact model.

### 6.5 Component Placement

Paired contrasts against the matching no-ACHF policy:

| Placement | Throughput delta [95% CI] | Reward delta [95% CI] | Train-time delta |
|---|---:|---:|---:|
| PPO Attention | -6.48% [-8.08, -4.88] | **-0.1523 [-0.2641, -0.0405]** | +8.994 s [8.816, 9.171] |
| PPO FFN | +2.66% [-0.17, 5.48] | +0.0174 [-0.5787, 0.6135] | +11.108 s [7.279, 14.937] |
| PPO FFN+Attention | -3.71% [-5.79, -1.63] | +0.0415 [-0.5713, 0.6543] | +14.895 s [13.022, 16.767] |
| DQN hidden layer | -16.94% [-21.20, -12.69] | +0.0948 [-0.0973, 0.2870] | +0.208 s [0.176, 0.241] |

Attention-only placement has a resolved reward regression. FFN-only is the only nonnegative throughput point estimate, but its interval crosses zero and training cost is much higher. It is a follow-up target, not a validated recommendation.

The longer convergence run is also negative: enabled throughput changes by -3.76% [-5.96, -1.56], reward by -0.0629 [-0.6089, 0.4830], loss by +0.0419 [-0.3380, 0.4219], and training time by +15.084 s [13.425, 16.743]. It shows no faster convergence or final-quality gain.

### 6.6 Path Crossover and Steady-State Operating Points

Both experiments use the same three-process statistical hierarchy as the end-to-end results. Each microbenchmark contributes one warmed mean per trial and one mean per process; inner timing loops are not pooled. Aggregate figures are `docs/media/crossover_aggregate.pdf` and `docs/media/regime_aggregate.pdf`.

#### 6.6.1 The fastest fixed path depends on the operating point

An operating point is listed as supported only when its nominal paired 95% difference interval is below both alternatives. These exploratory intervals are not multiplicity-corrected:

| Dimension | Supported Prepared Candidate | Unresolved | Supported Candidate CSR |
|---:|---|---|---|
| 256 | none | 0.50, 0.80, 0.90 | 0.95, 0.99 |
| 1024 | 0.50, 0.80 | 0.90 | 0.95, 0.99 |
| 2048 | 0.50, 0.80 | 0.90 | 0.95, 0.99 |

At dimension 1024 and sparsity 0.50, mean latencies are 2.244 ms Prepared Candidate, 10.938 ms Candidate CSR, and 2.634 ms Candidate Dense. At sparsity 0.99 they are 1.931, 0.242, and 2.285 ms. The supported conclusion is a conditional crossover between roughly 0.90 and 0.95 on this CPU. Dimension-256 low-sparsity differences and the 0.90 transition are unresolved, so there is no universal exact crossover.

#### 6.6.2 Guarded versus Plain selection at steady-state points

Oracle gap is online-selector latency divided by the fastest fixed-path latency for that exact point. Each point is warmed independently before measurement; there is no within-run sparsity or batch transition. This is a steady-state operating-point comparison, not a regime-transition or nonstationary-adaptation experiment. The last column is the process-level paired Guarded-minus-Plain difference; intervals are nominal and not multiplicity-corrected:

| Sparsity | Batch | Guarded AMA [95% CI] | Plain EMA [95% CI] | Paired difference [95% CI] |
|---:|---:|---:|---:|---:|
| 0.80 | 1 | **1.085 [1.053, 1.118]** | 1.421 [1.377, 1.466] | **-0.3358 [-0.4129, -0.2587]** |
| 0.80 | 128 | **1.096 [1.067, 1.125]** | 1.154 [1.070, 1.237] | **-0.0575 [-0.1120, -0.0030]** |
| 0.90 | 1 | **1.198 [1.117, 1.279]** | 1.426 [1.329, 1.522] | **-0.2281 [-0.2481, -0.2081]** |
| 0.90 | 128 | 1.047 [1.015, 1.079] | **1.026 [1.022, 1.031]** | +0.0205 [-0.0136, +0.0547] |
| 0.95 | 1 | **1.209 [1.171, 1.247]** | 1.826 [1.751, 1.902] | **-0.6173 [-0.7195, -0.5151]** |
| 0.95 | 128 | **1.095 [1.075, 1.115]** | 1.152 [1.004, 1.300] | -0.0572 [-0.1858, +0.0713] |
| 0.98 | 1 | **1.309 [1.249, 1.369]** | 3.682 [3.432, 3.932] | **-2.3731 [-2.5715, -2.1747]** |
| 0.98 | 128 | **1.179 [1.164, 1.193]** | 1.859 [1.717, 2.000] | **-0.6799 [-0.8096, -0.5502]** |

Guarded AMA has the lower mean at seven of eight points. Six paired intervals lie below zero; the 0.90/batch-128 and 0.95/batch-128 contrasts are unresolved. At 0.90/batch 128, Plain EMA has the lower mean. Every guarded marginal interval remains above one, so overhead is measurable. This establishes a guarded-versus-plain difference on the independently warmed grid, not recovery, cumulative regret, or stability during an operating-point transition.

The old single-invocation draft claimed a batch-driven oracle flip at sparsity 0.90. The replicated data do **not** reproduce it: the process-level majority oracle is Prepared Candidate for both batch sizes in all three processes. Two batch-1 processes contain one Candidate CSR trial, but Prepared Candidate remains the majority. The old flip claim is removed rather than carried forward.

#### 6.6.3 Overhead is the premium paid for worst-case robustness

The oracle is a hindsight lower bound, but its gap is still real overhead. The guarded result is useful only in the specific sense that it improves over Plain EMA at most tested points. It does not justify paying online-selection cost in a stationary deployment where Fixed Prepared Candidate is already known to win. This distinction also prevents the high-sparsity microbenchmark from being used to claim end-to-end speed: candidates at those sparsities fail the current task-quality admission boundary.

Overall answers:

- Task quality: no general reward or loss improvement; Attention placement regresses reward.
- Runtime: end-to-end ACHF is slower; path benefits are conditional on sparsity and dimension.
- Stability: connection scaling and sparse optimizer masks satisfy measured endpoint invariants; adaptive gate movement and projection utility are untested.
- Component value: the configured admission predicate excludes candidates outside its boundary; guarded ranking has lower mean gap than Plain EMA at seven of eight warmed points; projection utility, active-gate value, complete economic bypass, and exact-reuse value remain unisolated.

## 7. Discussion

The data make the three-layer separation operational rather than cosmetic. At the only fully admitted tested frontier point, the sparse semantic candidate is valid under the configured predicate but Candidate CSR is its slowest realization; Prepared Candidate is best. At higher sparsity, Candidate CSR eventually wins the microbenchmark, but those candidates fail the current task-quality boundary. Candidate admission, candidate realization, and exact reuse therefore cannot share one gate or one overloaded state variable.

The results also expose costs that the formulation alone cannot remove. The ACHF runtime instantiation loses end-to-end throughput, the rank-64 no-op control remains slower, materialized ACHF-layer state is 4.235 times its reference-parameter bytes, and guarded selection retains a positive oracle gap. Guarding is useful here only as a measured improvement over a weaker online selector at most steady-state points, not as a replacement for a known fixed path or as evidence of transition recovery. The unchanged training gate means this case study does not test the most ambitious adaptive-gating part of the proposal. These are evidence limits, not numbers to explain away as automatic tradeoffs.

ACHF remains distinct from conventional pruning because the reference stays available and candidate quality is checked explicitly. It also remains distinct from full mHC: the case study does not implement dynamic `H_pre`, `H_post`, and `H_res` maps. Projection constrains only a dedicated connection map when that constraint has a clear interpretation. AMA is a separable execution controller, not the source of the semantic candidate. The evaluated implementation covers guarded ranking and probing but not the complete proposed economic bypass.

## 8. Threats to Validity

**Statistical validity.** The independent unit is a process mean and `N=3`, leaving two degrees of freedom. Diagnostic comparisons are exploratory and are not corrected for family-wise testing. An interval crossing zero is not interpreted as equivalence.

**Hardware and workload validity.** All processes ran sequentially on one CPU/software environment. The task uses compact PPO and DQN policies in a custom Rust runtime, while crossover and selector grids use controlled matrix microbenchmarks. The results do not establish behavior for another CPU, a GPU, large transformers, standard benchmark datasets, distributed systems, or mature external sparse kernels.

**Runtime-control validity.** The fixed-mode experiment shows that Fixed Prepared Candidate is faster than both online selectors. Every selector-grid point is independently warmed, so the experiment measures no transition recovery, cumulative regret, or nonstationary interference.

**Calibration validity.** Calibration uses an index-modulo-four 75/25 split of temporally adjacent rollout states. Context windows can overlap, padded token rows are not independent samples, and the same validation subset is used for early stopping and final admission. The 0.51 point was selected after inspecting the same frontier. A stronger evaluation requires episode- or seed-separated calibration, tuning, and final-admission data.

**Gate and projection validity.** The candidate gate does not move and output discrepancy is not sampled during training. Endpoint connection-scaling and sparse-mask invariants do not establish active-gate value, projection utility, or long-horizon optimization stability. In the one-candidate blend, `g` and `rho` appear only through `(1-g)rho`; hard inference bypasses that coefficient.

**Baseline validity.** The suite includes dense, static magnitude, fixed-mask sparse-training, fixed-path, Plain EMA, no-op rank, and placement controls. It does not include an optimized external sparse library, a non-ACHF AMA system, early exit, MoE routing, or a matched implementation in a standard framework.

## 9. Limitations

1. The ACHF runtime instantiation with inactive training gate is slower than dense in both main and convergence experiments, takes longer to train, and has no resolved reward or loss benefit.
2. Aggregate intervals contain only three independent process means, and all processes use one CPU/software environment.
3. The training trace remains on the reference gate; learned adaptive gating and candidate-aware optimizer dynamics are not empirically exercised.
4. The 0.51 point is a same-frontier exploratory selection, not an independently validated operating point; calibration uses overlapping temporal contexts and one validation subset for early stopping and final admission.
5. At the admitted point, Candidate CSR is the slowest realization. The points where it wins the microbenchmark are too inaccurate to pass the current task-level admission criteria.
6. Guarded AMA has lower mean oracle gap than Plain EMA at seven of eight independently warmed points, but two paired contrasts are unresolved, all guarded gaps remain above one, and Fixed Prepared Candidate wins the stationary mode experiment. No transition experiment was run.
7. Materialized runtime tensors for the four ACHF layers occupy 418,464 bytes versus 98,816 reference-parameter bytes (4.23478 times). This is not whole-model memory or process RSS, and no lazy-layout or eviction policy is evaluated.
8. The evaluated selector does not instantiate the proposed complete baseline-relative economic bypass; exact-output memoization is disabled and has no speedup evidence.
9. FFN-only placement has a positive throughput point estimate but an interval crossing zero and a large training-time cost; attention placement reduces measured reward.
10. CUDA, BF16, other CPUs, standard ML frameworks, larger architectures, and external sparse libraries are not evaluated.
11. A non-ACHF AMA system and matched early-exit or MoE routing baselines are absent.
12. Sinkhorn scaling invariants are checked only for dedicated nonnegative connection matrices, not signed operator weights; its component utility is not isolated.
13. The case study does not implement the full dynamic mHC formulation; ACHF is a related but distinct runtime design.

## 10. Conclusion

ACHF formulates runtime-adaptive neural connections as three explicit layers: reference/candidate quality selection, execution-path selection for an admitted candidate, and exact memoized reuse. A dedicated connection matrix may be scaled when its semantics justify the constraint, while operator weights and optimizer state follow their own sparse mask. AMA supplies guarded measurement, probing, and switching for the execution layer.

The completed Talos-XII case study supports only part of that proposal. The configured admission predicate has one fully admitted tested frontier point, connection-scaling and mask invariants hold, fixed-path winners change with sparsity, and Guarded AMA has lower mean oracle gap at seven of eight independently warmed grid points, with six nominal paired intervals below zero. At the same time, the runtime instantiation is 16.62% slower than dense, adds 14.225 seconds of training, materializes 4.23478 times the ACHF-layer reference-parameter bytes, does not improve reward or loss, and never activates the candidate gate during training. Prepared Candidate rather than Candidate CSR is fastest at the admitted point. These results make ACHF a conditional runtime design with an explicit negative case study, not a demonstration of universal neural quality or efficiency. Cross-hardware replication, separate final-admission data, a true transition test, and an experiment that genuinely exercises adaptive gating are the next empirical requirements.

## Reproducibility Checklist

- [x] Run three independent OS processes with five paired trials each.
- [x] Retain root/child manifests, JSON, CSV, and SVG artifacts in the local verification directory `target/release/bench_output_final`.
- [x] Verify all 90 child artifact hashes and source reproducibility from commit `4ee46aecd31ae5dde4869dbe78e08640fe487e8e`.
- [x] Record CPU, RAM, OS, Rust/Cargo, build profile, features, seeds, config hash, executable hash, and cache policy.
- [x] Record task, model, optimizer, calibration, warmup, sample-count, and call-count overrides used by the paper benchmark.
- [x] Aggregate confidence intervals over process means rather than pooled inner samples.
- [x] Regenerate aggregate PDF/SVG/PNG figures from `docs/plot_aggregate_results.py`.
- [x] Replace TODO and single-invocation result claims with final measurements.
- [x] Remove the non-reproduced batch-flip claim.
- [x] Record that exact-output memoization is disabled in every paper benchmark.
- [ ] Upload an immutable public archive of the raw artifacts and insert its real URL before submission; the local directory is not a public artifact.
- [ ] Repeat on a second hardware/software environment.
- [ ] Run a configuration in which candidate gating participates during training.
- [ ] Run a controlled CUDA study only if the paper later makes GPU claims.

## References

- He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. CVPR 2016.
- He, K., Zhang, X., Ren, S., and Sun, J. Identity Mappings in Deep Residual Networks. ECCV 2016.
- Sinkhorn, R. and Knopp, P. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 1967.
- Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.
- Zhu, D. et al. Hyper-Connections. arXiv 2024.
- Xie, Z. et al. mHC: Manifold-Constrained Hyper-Connections. arXiv 2026.
- Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv 2017.
- Chen, T. et al. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. OSDI 2018.
- Zheng, L. et al. Ansor: Generating High-Performance Tensor Programs for Deep Learning. OSDI 2020.
- Kjolstad, F. et al. The Tensor Algebra Compiler. OOPSLA 2017.
- Ye, Z. et al. SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning. ASPLOS 2023.
- Ning, L. and Shen, X. Deep Reuse: Streamline CNN Inference on the Fly via Coarse-Grained Computation Reuse. ICS 2019.

## Appendix A. Implementation Hooks

ACHF configuration fields used by the current implementation:

```text
enabled
mode
candidate_mode
candidate_refresh_freq
proj_mode
ortho_penalty_freq
proj_steps
lambda_ortho
gate_mode
gate_momentum
gate_beta
gate_alpha
g_min
gate_warmup_steps
gate_transition_steps
gate_k_clip
g_target_min
g_target_max
g_min_adapt_rate
g_min_momentum
cache_min_rows
cache_min_nonzero_ratio
cache_min_reuse
path_warmup_samples
path_min_dwell
cache_sparsity_sample_rows
cache_cost_bias
cache_adapt_rate
cache_bias_min
cache_bias_max
cache_latency_ema
cache_latency_long_ema
cache_adapt_blend
cache_latency_sample_every
rank
prune_threshold
candidate_target_sparsity
candidate_min_sparsity
candidate_max_relative_error
candidate_max_output_relative_error
candidate_min_calibration_samples
candidate_calibration_steps
candidate_calibration_lr
candidate_calibration_max_samples
candidate_train_from_scratch
candidate_weight_error_momentum
apply_attn
apply_ffn
apply_dqn
infer_gate
```

## Appendix B. Drafting Notes

`docs/ACHF-paper.tex` is the canonical submission source; this Markdown file is a synchronized editing draft. Final editorial work is limited to:

1. Preserve the current evidence boundary when shortening the manuscript.
2. Add cross-hardware or active-gate results only as separately identified experiments.
3. Choose an arXiv category and add any authorship or AI-assistance disclosure required by the submission policy.
