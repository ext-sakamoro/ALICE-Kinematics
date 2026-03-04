# Contributing to ALICE-Kinematics

## Prerequisites

- Rust 1.70+ (stable)
- `clippy`, `rustfmt` コンポーネント (`rustup component add clippy rustfmt`)

## Code Style

- `cargo fmt` 準拠（CI で `--check` 実行）
- `cargo clippy --features encoder -- -W clippy::all -W clippy::pedantic` 警告ゼロ
- パブリック関数には `#[must_use]` を付与
- `no_std` 互換: `std` は使わない（`alloc` のみ）
- コード内コメント: 日本語

## Build

```bash
cargo build
cargo build --features encoder
```

## Test

```bash
cargo test
cargo test --features encoder
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **8-byte intent encoding**: 1000Hz raw coordinates → compact kinematic intent packets.
- **Quintic polynomial prediction**: minimum-jerk trajectory reconstruction from boundary conditions.
- **Dual-license**: MIT core (joint, intent, predictor); AGPL-3.0 encoder modules (feature-gated).
- **`no_std` + `alloc`**: runs on embedded and bare-metal targets without `std`.
- **Zero external dependencies**: all math and kinematics are self-contained.
