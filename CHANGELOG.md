# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

### Added
- `alice_kinematics::semantics` (ALICE-Law G6 / G8): 2 つの識別子 `KIN_SEMANTICS_ID` (この crate の算術、`tests/semantics_golden.rs` が bit を記録する 5 場面 (intent encode / extended intent encode / packet decode / quintic trajectory / arm inverse kinematics) と `alice_det_math::SEMANTICS_ID` を alice-det-math と同じ手順で畳んだ値、alice-det-math 0.5.1 で `440a6e99…d4df63`) と `KIN_WIRE_SEMANTICS_ID` (packet の形式、packet の encode / decode の 3 場面だけを畳んだ値 `d5ae6ab1…ef31ca`) 表 `KIN_SEMANTICS_BY_DET_MATH` に行の無い alice-det-math の世代では compile が止まる `KIN_SEMANTICS_CHECKED_AT` (crate の版と toolchain) は試験で照合する 試験 `tests/kin_semantics_id.rs` は各 entry を golden の試験の source から読んで比べる
- `alice_kinematics::stream` (ALICE-Law G7 / G8): stream と記録 file の先頭に置く 80 byte の hello frame (`AKINHELO`、形式 1、packet の種類、producer の `KIN_SEMANTICS_ID`、`KIN_WIRE_SEMANTICS_ID`、CRC-32) packet (8 / 16 byte) は変えていない `open_stream` は hello を読んで 2 つの識別子を確かめ、packet の byte 列を返す 形式の識別子が違えば `StreamError::WireMismatch` (byte の意味が違う)、算術の識別子が違えば `StreamError::SemanticsMismatch` (同じ形式だが別の算術で計算した intent) で断り、どちらも両方の識別子を表示する hello の無い stream (0.3.0 より前) は `StreamStatus::Unpinned`、`HelloPolicy::RequirePinned` では `StreamError::Unpinned` 未知の hello 形式は `StreamError::UnknownFormat` CRC-32 は偶発的な破損の検出だけで、別 build の stream を断るのは識別子の比較 **0.3.0 より前の reader は hello を検出できない** (`Intent::decode` はどの 8 byte も受け付け、flags の 256 値はすべて有効なので、hello は 10 個の intent として読まれる) hello で始まる stream は 0.3.0 以降の reader にだけ送る 試験 `tests/stream_hello.rs` (14 件)
- CI (ALICE-Law G2): f32 / f64 の `max` / `min` / `clamp` を `clippy.toml` で禁止し (test を含む全 target、例外なし)、library と test の呼び出しを `alice-det-math` 0.5.1 の Law helper (`max` / `min` / `metric::clamp`) に置き換えた std の `max` / `min` は同じ値の正負のゼロのどちらを返すかが未規定で target と最適化で変わり、`clamp` は NaN の payload をそのまま通し上下が逆だと panic する helper は正負のゼロの符号を定め (max は +0、min は -0)、NaN には正規の NaN を返す 各呼び出しの入力 (有限値、定数の範囲) では結果の bit は変わらない 依存に `alice-det-math` (Apache-2.0、`no_std`) を足した clippy の canary に 6 件を足した (59 件) `alice-law-conformance.toml` の G2 を pass にした
- CI (ALICE-Law G3): fma の結果全体を negate する形の検査を追加 (ALICE-LOL と同一の構文木の前段検査 `tools/no-outer-fneg-of-fma`、最適化後の LLVM IR の検査 `scripts/fneg_fma_ir_check.py` と試験 `scripts/test_fneg_fma_ir_check.py`、47 個の probe `tools/fneg-fma-ir-probes/`) この crate の source に fma の呼び出しは無いので、`scripts/no-outer-fneg-of-fma-allowlist.toml` に `expected-fma-calls = 0` / `expected-src-fma-calls = 0` と IR の下限 0 を宣言し、最初の呼び出しは件数で、negate する形は形で失敗する 構文木の検査は float-environment job (3 OS) と preflight、IR の検査は新しい `.github/workflows/g3-ir.yml` が x86_64 (ubuntu-latest) と aarch64 (ubuntu-24.04-arm) で source・manifest・toolchain・検査自身の変更時と毎週実行 (lol feature の alice-lol が依存する stacker → psm の build script が C を compile するので各 target は native の runner) `alice-law-conformance.toml` の G3 を fail-known から pass にした
- CI (ALICE-Law G1): `clippy.toml` に platform の libm に依存する f32 / f64 の超越関数と `powi` の禁止を足した (ALICE-DB と alice-zip の一覧のうち f32 / f64 の method の 52 件、libm に依存しないので libm の関数は含めない) 既定の build は `no_std` でこれらの method が無いので、clippy job に `std` と全 feature の build を足した (`RUSTFLAGS=-Dwarnings` なので lint は error になる) これまでの encoder の段だけでは std の code に置いた禁止の呼び出しが通っていた test の閉形式の参照値と test 入力は理由つきの scoped `#[expect]` にした canary の行を再生成した (53 行) `alice-law-conformance.toml` の G1 を fail-known から pass にした std と全 feature の clippy の段が std の build を実際に lint していることを `scripts/clippy_lint_probe.py` (試験 `scripts/test_clippy_lint_probe.py`) で確かめる: 同じ command を `--cfg alice_lint_probe` 付きで走らせ、`src/lib.rs` 末尾の probe の `f32::sin` 1 件を clippy が報告しなければ失敗する CI の workflow を `ci/**` の branch でも走らせ、同じ branch の古い run は止める
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査し、round-to-nearest-even と subnormal が flush されないことを実行時に確かめる (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` / `tests/float_environment.rs` と test、`scripts/run_tests.sh` / `scripts/test_run_tests.py`) 新しい float-environment job (ubuntu / macOS / Windows) で test を debug と release で `scripts/run_tests.sh float_environment=2` を通して実行し (2 本とも通らなければ失敗)、静的な検査を走らせ、ubuntu / macOS では新しい target dir で `cargo check -v` を走らせて rustc に実際に渡った flag を読む (default / encoder / all-features) 意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く `alice-law-conformance.toml` の G14 を fail-known から pass にした
- `alice-law-conformance.toml` — ALICE-Law の 14 gate (G1-G14) について、この repo の tree を実測して記録した適合状況 `pass` は 0 件 (committed な gate evidence が無いため)、`fail-known` 11 件 (位置付き)、`unverified` 3 件 (理由付き)
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision (779fa44) で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る

### Changed
- 版を 0.2.1 から 0.3.0 に (公開 API の追加と依存 `alice-det-math` の追加) `alice-law-conformance.toml` の G6 / G7 / G8 を pass にした dev-dependency に `sha2` を足した
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
