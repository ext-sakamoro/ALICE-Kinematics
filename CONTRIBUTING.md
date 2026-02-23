# Contributing to ALICE-Kinematics

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
