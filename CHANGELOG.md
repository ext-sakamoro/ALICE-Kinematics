# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

### Added
- `alice-law-conformance.toml` — ALICE-Law の 14 gate (G1-G14) について、この repo の tree を実測して記録した適合宣言 `pass` は 0 件 (committed な gate evidence が無いため)、`fail-known` 9 件 (位置付き)、`unverified` 5 件 (理由付き)

## [0.2.1] - 2026-10-13

### Changed
- `include` allowlist を Cargo.toml に追加し、配布 package を crate 本体に限定 (`.github/` / `docs/` / `scripts/` / `tests/` / `examples/` / `rust-toolchain.toml` / `CONTRIBUTING.md` を除外) code 変更なし

### Removed
- `rerun-viz` feature と optional な `rerun` 依存をライブラリから削除 dev 可視化は非公開の別 crate `examples/rerun-viz/` (独自 workspace、`publish = false`、`rust-toolchain.toml` 1.93.0 で build) に移動 ライブラリの実行時には不要で、全 feature を安定版 toolchain で build できるようになる

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
