# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-10-09

### Added
- `tests/analytic_oracle.rs` — 解析解 oracle 7 本 (Flash & Hogan minimum-jerk quintic / Fitts-Shannon / Rodrigues 回転と右腕 FK の独立 f64 chain 突合 / CCD IK の到達性と反復単調性 / predictor の dt 非依存 / skeleton 比例 scaling / hand grip 単調性) 期待値は閉形式解と本 file 内の f64 参照実装のみから作り、被検査関数を呼んで作らない

### Changed
- `tests/analytic_oracle.rs` の f32 → f64 昇格 57 箇所を `as` から `f64::from` に統一 残る意図的な narrowing 4 種 (`cast_possible_truncation` / `cast_precision_loss` / `float_cmp` / `many_single_char_names`) は file 冒頭に lint 単位の理由を付けて allow (`float_cmp` の比較対象は実装が literal で返す sentinel で、計算結果ではない)
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
