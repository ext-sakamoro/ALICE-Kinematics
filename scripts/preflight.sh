#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

need actionlint "brew install actionlint"

step "ci.yml / clippy: clippy (encoder)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo clippy --lib --tests --features encoder -- -W clippy::all -W clippy::pedantic )

step "ci.yml / clippy: Clippy canary data (generator tests)"
python3 scripts/test_gen_clippy_canary_calls.py

step "ci.yml / clippy: Clippy canary data (generated from clippy.toml, no drift)"
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: Clippy canary (tests)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: Clippy canary (every planted banned call is flagged)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; python3 scripts/clippy_canary.py )

step "ci.yml / float-environment: run_tests.sh and build configuration guard (tests, then check)"
python3 scripts/test_run_tests.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py

step "ci.yml / fmt: run"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo fmt -- --check )

step "ci.yml / doc: cargo doc"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" RUSTDOCFLAGS="-Dwarnings"; cargo doc --no-deps )

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / test: cargo test (default)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo test --lib --tests )

step "ci.yml / test: cargo test (encoder)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo test --lib --tests --features encoder )

step "ci.yml / test: cargo test (lol bridge)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo test --lib --tests --features lol )

step "ci.yml / float-environment: Float environment (debug and release)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment
scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment

step "ci.yml / float-environment: Effective rustc flags (default, encoder, all features)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --features encoder
python3 scripts/build_flags_check.py -- cargo check --release --lib --all-features

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

echo; echo "preflight OK"
