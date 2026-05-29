# AGENTS.md

<!-- context7 -->
Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer; training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use Context7 for refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

Context7 steps:

1. Start with `resolve-library-id` using the library name and the user's question, unless the user provides an exact library ID in `/org/project` format.
2. Pick the best match by exact name, description relevance, snippet count, source reputation, and benchmark score. Use version-specific IDs when the user mentions a version.
3. Call `query-docs` with the selected library ID and the user's full question.
4. Answer or implement using the fetched docs.
<!-- context7 -->

## Cursor Cloud specific instructions

### Project overview

Talos-XII is a single-binary Rust CLI application (gacha pull simulator for Arknights: Endfield). No external services, databases, or Docker required. Optional features enable CUDA acceleration and embedded Python scripting.

### Toolchain

- **MSRV**: `rust-version = "1.89.0"` in `Cargo.toml`. Ensure `rustup default stable` points to >= 1.89.0.
- Build flags in `.cargo/config.toml` set `-C target-cpu=native` for SIMD acceleration.

### Common commands

See `README.md` for full details. Quick reference:

| Task | Command |
|---|---|
| Build (dev) | `cargo build` |
| Test (default features, 169 tests) | `cargo test` |
| Test (all features, 190 tests) | `cargo test --all-features` |
| Lint (format) | `cargo fmt -- --check` |
| Lint (clippy) | `cargo clippy -- -D warnings` |
| Lint (all features) | `cargo clippy --all-features -- -D warnings` |
| Check all features | `cargo check --all-features` |
| Run (simulate) | `cargo run -- simulate -n 1000 -p 100` |
| Run (interactive) | `cargo run -- interactive` |
| Run (F2P analysis) | `cargo run -- f2p` |
| Run (benchmark) | `cargo run -- benchmark` |
| Run (ACHF paper bench) | `cargo run -- benchmark paper` |
| Run (ACHF paper, 5 trials) | `cargo run -- benchmark paper --trials 5` |
| Run (ACHF single exp) | `cargo run -- benchmark paper --only ablation` |
| Collect add (interactive) | `cargo run -- collect add` |
| Collect import | `cargo run -- collect import <file>` |
| Collect stats | `cargo run -- collect stats` |
| Train on collected data | `cargo run -- train` |
| Run Python script | `cargo run --features python -- python <script.py> -- <args>` |
| Python smoke example | `cargo run --features python -- python examples/python/autograd_minimal.py -- 1.0` |
| Force retrain (ignore cache) | `cargo run -- -f simulate -n 100` |

### Non-obvious caveats

- **First run trains neural models**: The first model-using `cargo run` invocation takes ~30-45s because it trains or loads EnvNet, NeuralLuckOptimizer, DQN (50k steps), and PPO (20k steps by default). Subsequent runs load cache files and start in <1s.
- **Cache files are generated in CWD**: Running model commands creates cache artifacts such as `env_net.cache`, `neural.cache`, `dqn.cache.bin`, `ppo.cache.bin`, `dqn.cache.bf16.bin`, and `ppo.cache.bf16.bin`. These are gitignored.
- **Config warnings are benign**: `[Config Warning] Unknown field: _comment*` messages at startup are expected — they come from documentation comment fields in `data/config.json`.
- **Dev profile uses opt-level 3**: The `[profile.dev]` in `Cargo.toml` enables full optimizations even in debug mode (needed for neural network perf). This means dev builds are slower to compile but run at near-release speed.
- **`interactive` subcommand reads stdin**: Avoid using `cargo run -- interactive` in non-TTY contexts as it blocks waiting for user input.
- **`benchmark paper` generates output files**: Charts (SVG/PNG) are written to `bench_output/` by default. This directory is gitignored.
- **Python scripting is optional**: Build with `--features python` to run embedded PyO3 scripts. The `python` subcommand exits before model initialization, exposes the `talos_xii` Python module, does not require NumPy or PyTorch, and is not sandboxed.
- **CUDA is optional but affects all-features checks**: `--features cuda` / `--all-features` requires a working CUDA toolchain (`nvcc`, CUDA runtime/libs, and MSVC `cl.exe` on Windows). Use default-feature checks when CUDA is unavailable.
