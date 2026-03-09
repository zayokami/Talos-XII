# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Talos-XII is a single-binary Rust CLI application (gacha pull simulator for Arknights: Endfield). No external services, databases, or Docker required.

### Toolchain

- **MSRV**: `rust-version = "1.90.0"` in `Cargo.toml`. Ensure `rustup default stable` points to >= 1.90.0.
- Build flags in `.cargo/config.toml` set `-C target-cpu=native` for SIMD acceleration.

### Common commands

See `README.md` for full details. Quick reference:

| Task | Command |
|---|---|
| Build (dev) | `cargo build` |
| Test (52 tests) | `cargo test` |
| Lint (format) | `cargo fmt -- --check` |
| Lint (clippy) | `cargo clippy -- -D warnings` |
| Run (simulate) | `cargo run -- simulate -n 1000 -p 100` |
| Run (interactive) | `cargo run -- interactive` |
| Run (F2P analysis) | `cargo run -- f2p` |
| Run (benchmark) | `cargo run -- benchmark` |

### Non-obvious caveats

- **First run trains neural models**: The first `cargo run` invocation takes ~45s because it trains DBN, DQN (50k steps), and PPO (200k steps) models from scratch. Subsequent runs load from cache files (`neural.cache`, `dqn.cache.bin`, `ppo.cache.bin`) and start in <1s.
- **Cache files are generated in CWD**: Running the binary creates `*.cache` and `*.cache.bin` files in the working directory. These are gitignored.
- **Config warnings are benign**: `[Config Warning] Unknown field: _comment*` messages at startup are expected — they come from documentation comment fields in `data/config.json`.
- **Dev profile uses opt-level 3**: The `[profile.dev]` in `Cargo.toml` enables full optimizations even in debug mode (needed for neural network perf). This means dev builds are slower to compile but run at near-release speed.
- **`interactive` subcommand reads stdin**: Avoid using `cargo run -- interactive` in non-TTY contexts as it blocks waiting for user input.
