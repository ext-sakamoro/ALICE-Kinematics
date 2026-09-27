# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

### Changed
- `alice-lol` 要件を `0.3` → `0.4` に追随 (LOL 0.4.0 は `LawReport` に `unresolved` field を追加した breaking だが、本 crate が使うのは `intent` module (`HandSide` / `IntentNode` / `NodeId`) と `Vec3` だけなので code 変更なし) これで `ALICE-LOL` workspace 側の `cargo metadata` が `alice-lol-robot` → `alice-kinematics` の path dep を解決できるようになる

## [0.1.0] - 2026-02-23

### Added
- `joint` — `Vec3k`, `Joint`, `JointConstraint`, `ArmChain` kinematic primitives
- `intent` — `Intent`, `IntentType`, `IntentFlags` 8-byte motion intent encoding
- `predictor` — `Predictor`, `QuinticCoeffs` minimum-jerk trajectory reconstruction
- `jerk` — Jerk-minimization analysis (feature-gated: `encoder`, AGPL-3.0)
- `encoder` — Raw sensor → intent extraction (feature-gated: `encoder`, AGPL-3.0)
- Feature flags: `std`, `encoder`
- `no_std` + `alloc` support
- Zero external dependencies
- 59 unit tests (41 base + 18 encoder feature)
- CI/CD (GitHub Actions: test, clippy pedantic, fmt, doc)
- `#[must_use]` on all public query functions
- `Default` impl for `IntentEncoder`, `JerkFitter`
