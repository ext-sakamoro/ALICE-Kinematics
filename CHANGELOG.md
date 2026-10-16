# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

### Added
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査し、round-to-nearest-even と subnormal が flush されないことを実行時に確かめる (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` / `tests/float_environment.rs` と test、`scripts/run_tests.sh` / `scripts/test_run_tests.py`) 新しい float-environment job (ubuntu / macOS / Windows) で test を debug と release で `scripts/run_tests.sh float_environment=2` を通して実行し (2 本とも通らなければ失敗)、静的な検査を走らせ、ubuntu / macOS では新しい target dir で `cargo check -v` を走らせて rustc に実際に渡った flag を読む (default / encoder / all-features) 意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く `alice-law-conformance.toml` の G14 を fail-known から pass にした
- `alice-law-conformance.toml` — ALICE-Law の 14 gate (G1-G14) について、この repo の tree を実測して記録した適合状況 `pass` は 0 件 (committed な gate evidence が無いため)、`fail-known` 11 件 (位置付き)、`unverified` 3 件 (理由付き)

### Changed
- `rust-toolchain.toml` の pin を 1.92.0 から 1.98.1 に上げた 新しい clippy が指摘する手書きの midpoint `(a + b) * 0.5` は書き換えない (std の `f32::midpoint` は target ごとに実装が違い、結果が subnormal の範囲で bit が違う) crate の root に `clippy.toml` を置いて `f32::midpoint` を禁止し、4 箇所に理由付きで `manual_midpoint` を allow した `IntentEncoder::compression_estimate` の 1 秒あたりの motion 数は `checked_div` で求める (値は同じ) 出力の bit は変わらない
- CI: 公開 API の互換性を crates.io の最新の alice-kinematics と比べて失敗させる `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB ほかと同一) を追加した 既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせる (ALICE-Law release-train.toml は組を宣言しない) 版が公開済みの 0.2.1 と同じ間は次の最小の版 0.2.2 として検査する
- `lol` feature の `alice-lol` 要求を `0.4` から `>=0.4.1, <0.6` に広げ 公開済の 0.4.1 と 0.5 系の両方で解決できるようにした `IntentNode` への exhaustive match は 0.4.1 と 0.5 系の双方で build と test を確認済 code 変更なし
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加した 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し、CI が再生成して byte で比べる この repo は 1 件 (`disallowed-methods` の `f32::midpoint`、植える) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と同じ中身

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
