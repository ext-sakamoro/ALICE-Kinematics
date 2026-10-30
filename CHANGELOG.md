# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

## [Unreleased]

### Added
- `alice-lol-robot` crate (ALICE-LOL から移した、crate 名と API は同じ) を workspace の member にした 試験 (lib と `tests/analytic_robot_law.rs`) を CI の test と release の job で走らせる
- CI: `security-audit` workflow: cargo audit と cargo deny (`deny.toml` を追加) を、`.github/alice-siblings` に固定した実物の sibling (ALICE-LOL / ALICE-SDF / ALICE-Zip) と毎回新しい解決で走らせる (依存や設定の変更時、週次 水曜 04:23 UTC、手動) 解決と cargo deny の graph がこの crate と path の sibling を含むことを `scripts/sibling_coverage_guard.py` (試験付き) で確かめ、0 件の走査は失敗 bincode (unmaintained、alice-sdf 経由) は期限付きの保留 (`scripts/advisory_holds.toml`、`scripts/advisory_hold_guard.py` が版・依存元・期限を検査、2026-11-14 まで) で、それ以外の ignore は無い 保留は今日から 21 日以内に限る (guard が検査、延長は理由とともに日付を更新)
- `conformance/g11-own.toml`: ALICE-Law G11 の表のうち、この crate 固有の行 (G2 の max / min の tie、G4 / G5 の golden、G6 の識別子、G7 / G8 の hello frame の識別子の照合、比較 0 件は `scripts/run_tests.sh` の試験) 共有の検査器の行は ALICE-Law の `conformance/g11-shared-rows.toml` から生成する
- `tests/semantics_golden.rs`: `checker_rejects_a_wrong_digest` を追加 動いた bit の digest が記録と一致しないことを確かめる (比較を常に一致にする変異でこの試験だけが失敗する) CI と preflight の `scripts/run_tests.sh` の最小数を `semantics_golden=6` にした
- CI: text の file の行末の検査 `scripts/check_eol.py` (試験 `scripts/test_check_eol.py`) と `.github/workflows/eol.yml` を足した `git ls-files --eol` で text の file が index か working tree で CRLF / 混在なら失敗 (`-text` と binary は除く、比べた数を出し 0 件は失敗)、workflow は ubuntu / macos / windows で走り、preflight にも同じ検査を足した (ALICE の crate で同じ中身の file)
- CI (ALICE-Law G11): 他の ALICE crate と byte 一致で共有する file を ALICE-Law の表と照合する `scripts/shared_files.sh` (表と検査器は `.github/shared-files-pin` の版を SHA-256 付きで取得) と試験を float-environment job (ubuntu / macOS / Windows) と preflight に追加 違い・表に無い共有 path・比較 0 件は失敗する
- CI (ALICE-Law G9): `scripts/id_move_check.py` (ALICE の識別子を持つ crate で共有、ALICE-Zip と同じ file) を追加 最新の release tag の公開版と作業 tree で `KIN_SEMANTICS_ID` と `KIN_WIRE_SEMANTICS_ID` の値を読み、動いたのに破壊的な版になっていない / 公開版の値が以前の識別子の一覧に無い / CHANGELOG が両方の値を書いていない時に fail する (ci.yml の clippy job と preflight)
- `tests/float_edge_golden.rs` (ALICE-Law G4): 正負のゼロの座標が同じ packet に encode され +0 に decode されること、0 から 1 への静止→静止の 5 次軌道が T で位置 1.0・速度と加速度 +0 にちょうど打ち消し合うこと (c3 = 10、c4 = -15、c5 = 6)、変位 0 と所要時間 0 の分岐が値を保つことを固定の bit で比べる 3 件 debug と release、x86_64 (ubuntu) と aarch64 (macOS) で走る
- `tests/nan_boundaries.rs` と `scripts/nan_boundary_scan.py` (試験 `scripts/test_nan_boundary_scan.py`、ALICE-Law G2): f32 の値を返す公開関数ごとに、符号 bit と payload の立った NaN を入れて出てくる NaN が `0x7fc0_0000` であることを bit で比べる (既定 39 件、encoder 43 件) scan は公開関数を列挙し、試験も理由つきの除外も無いもの、試験の名前がどの関数も指さないもの、比較 0 件を失敗にする float の公開 field を持つ crate の型を返す 40 の関数はまだ正準化していないので `DEBT` として数え、増減を失敗にする
- CI (ALICE-Law G4 / G5): test job を ubuntu (x86_64) と macOS (aarch64) の matrix にし、`--release` で同じ試験を走らせる release job を足した Law の試験 (`semantics_golden` / `kin_semantics_id` / `float_edge_golden` / `law_max_min_ties` / `stream_hello` / `float_environment` / `nan_boundaries`) は `scripts/run_tests.sh` で件数を数える
- CI (ALICE-Law G10 / G13): `clippy.toml` で時計 (`Instant::now` / `::elapsed`、`SystemTime::now` / `::elapsed`)、`RandomState::new`、`HashMap` / `HashSet` を禁止した (canary 66 件) `alice-law-conformance.toml` の G5 / G10 / G13 を pass にした G4 は golden を足したが、struct を返す 40 の出力の NaN が未正準化なので fail-known のまま (DEBT として数える)
- `tests/law_max_min_ties.rs` (ALICE-Law G2): max / min / clamp の呼び出しを手書きの比較に置き換えると変わる結果を、公開 API で bit 比較する (`HandModel::grip` に -0 と payload 付き NaN、`ArmChain::inverse_kinematics` に NaN の目標、`JerkFitter::fit_trajectory` に NaN の時刻) clippy の禁止は手書きの比較を検出しないため 公開 API から tie の入力が届かない呼び出し (skinning の距離、jerk の定数との max、lol bridge の u8 変換) は理由を `alice-law-conformance.toml` に書いた
- `alice_kinematics::semantics` (ALICE-Law G6 / G8): 2 つの識別子 `KIN_SEMANTICS_ID` (この crate の算術、`tests/semantics_golden.rs` が bit を記録する 5 場面 (intent encode / extended intent encode / packet decode / quintic trajectory / arm inverse kinematics) と `alice_det_math::SEMANTICS_ID` を alice-det-math と同じ手順で畳んだ値、alice-det-math 0.5.1 で `440a6e99…d4df63`) と `KIN_WIRE_SEMANTICS_ID` (packet の形式、packet の encode / decode の 3 場面だけを畳んだ値 `d5ae6ab1…ef31ca`) 表 `KIN_SEMANTICS_BY_DET_MATH` に行の無い alice-det-math の世代では compile が止まる `KIN_SEMANTICS_CHECKED_AT` (crate の版と toolchain) は試験で照合する 試験 `tests/kin_semantics_id.rs` は各 entry を golden の試験の source から読んで比べる
- `alice_kinematics::stream` (ALICE-Law G7 / G8): stream と記録 file の先頭に置く 80 byte の hello frame (`AKINHELO`、形式 1、packet の種類、producer の `KIN_SEMANTICS_ID`、`KIN_WIRE_SEMANTICS_ID`、CRC-32) packet (8 / 16 byte) は変えていない `open_stream` は hello を読んで 2 つの識別子を確かめ、packet の byte 列を返す 形式の識別子が違えば `StreamError::WireMismatch` (byte の意味が違う)、算術の識別子が違えば `StreamError::SemanticsMismatch` (同じ形式だが別の算術で計算した intent) で断り、どちらも両方の識別子を表示する hello の無い stream (0.3.0 より前) は `StreamStatus::Unpinned`、`HelloPolicy::RequirePinned` では `StreamError::Unpinned` 未知の hello 形式は `StreamError::UnknownFormat` CRC-32 は偶発的な破損の検出だけで、別 build の stream を断るのは識別子の比較 **0.3.0 より前の reader は hello を検出できない** (`Intent::decode` はどの 8 byte も受け付け、flags の 256 値はすべて有効なので、hello は 10 個の intent として読まれる) hello で始まる stream は 0.3.0 以降の reader にだけ送る 試験 `tests/stream_hello.rs` (14 件)
- CI (ALICE-Law G2): f32 / f64 の `max` / `min` / `clamp` を `clippy.toml` で禁止し (test を含む全 target、例外なし)、library と test の呼び出しを `alice-det-math` 0.5.1 の Law helper (`max` / `min` / `metric::clamp`) に置き換えた std の `max` / `min` は同じ値の正負のゼロのどちらを返すかが未規定で target と最適化で変わり、`clamp` は NaN の payload をそのまま通し上下が逆だと panic する helper は正負のゼロの符号を定め (max は +0、min は -0)、NaN には正規の NaN を返す 各呼び出しの入力 (有限値、定数の範囲) では結果の bit は変わらない 依存に `alice-det-math` (Apache-2.0、`no_std`) を足した clippy の canary に 6 件を足した (59 件) `alice-law-conformance.toml` の G2 を pass にした
- CI (ALICE-Law G3): fma の結果全体を negate する形の検査を追加 (ALICE-LOL と同一の構文木の前段検査 `tools/no-outer-fneg-of-fma`、最適化後の LLVM IR の検査 `scripts/fneg_fma_ir_check.py` と試験 `scripts/test_fneg_fma_ir_check.py`、47 個の probe `tools/fneg-fma-ir-probes/`) この crate の source に fma の呼び出しは無いので、`scripts/no-outer-fneg-of-fma-allowlist.toml` に `expected-fma-calls = 0` / `expected-src-fma-calls = 0` と IR の下限 0 を宣言し、最初の呼び出しは件数で、negate する形は形で失敗する 構文木の検査は float-environment job (3 OS) と preflight、IR の検査は新しい `.github/workflows/g3-ir.yml` が x86_64 (ubuntu-latest) と aarch64 (ubuntu-24.04-arm) で source・manifest・toolchain・検査自身の変更時と毎週実行 (lol feature の alice-lol が依存する stacker → psm の build script が C を compile するので各 target は native の runner) `alice-law-conformance.toml` の G3 を fail-known から pass にした
- CI (ALICE-Law G1): `clippy.toml` に platform の libm に依存する f32 / f64 の超越関数と `powi` の禁止を足した (ALICE-DB と alice-zip の一覧のうち f32 / f64 の method の 52 件、libm に依存しないので libm の関数は含めない) 既定の build は `no_std` でこれらの method が無いので、clippy job に `std` と全 feature の build を足した (`RUSTFLAGS=-Dwarnings` なので lint は error になる) これまでの encoder の段だけでは std の code に置いた禁止の呼び出しが通っていた test の閉形式の参照値と test 入力は理由つきの scoped `#[expect]` にした canary の行を再生成した (53 行) `alice-law-conformance.toml` の G1 を fail-known から pass にした std と全 feature の clippy の段が std の build を実際に lint していることを `scripts/clippy_lint_probe.py` (試験 `scripts/test_clippy_lint_probe.py`) で確かめる: 同じ command を `--cfg alice_lint_probe` 付きで走らせ、`src/lib.rs` 末尾の probe の `f32::sin` 1 件を clippy が報告しなければ失敗する CI の workflow を `ci/**` の branch でも走らせ、同じ branch の古い run は止める
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査し、round-to-nearest-even と subnormal が flush されないことを実行時に確かめる (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` / `tests/float_environment.rs` と test、`scripts/run_tests.sh` / `scripts/test_run_tests.py`) 新しい float-environment job (ubuntu / macOS / Windows) で test を debug と release で `scripts/run_tests.sh float_environment=2` を通して実行し (2 本とも通らなければ失敗)、静的な検査を走らせ、ubuntu / macOS では新しい target dir で `cargo check -v` を走らせて rustc に実際に渡った flag を読む (default / encoder / all-features) 意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く `alice-law-conformance.toml` の G14 を fail-known から pass にした
- `alice-law-conformance.toml` — ALICE-Law の 14 gate (G1-G14) について、この repo の tree を実測して記録した適合状況 `pass` は 0 件 (committed な gate evidence が無いため)、`fail-known` 11 件 (位置付き)、`unverified` 3 件 (理由付き)
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision (779fa44) で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る

### Changed
- CI (ALICE-Law G9 / G11): 識別子の移動の検査器 (`scripts/id_move_check.py` と試験) を共有の新しい bytes にし、新しく足した associated const を受けるようにした 他の共有の file も ALICE-Law 40973a8a の表が計画した bytes にし、shared-files と G11 の表の pin を上げた
- CI (ALICE-Law G11): 共有の file を ALICE-Law 0a5de013 の表の目標の bytes にし、`.github/shared-files-pin` をその rev に移した `scripts/semver_gate.py` が使う `scripts/release_baseline.py` と試験を足し、semver-checks の workflow と preflight でその試験を走らせる `scripts/id_move_check.py` と試験は目標の bytes が決まり直すまで今の bytes のまま
- CI の doc job は workspace 全体 (`cargo doc --no-deps --workspace`、alice-lol-robot を含む) を `-Dwarnings` で build する alice-lol-robot の doc comment の単位 (`[rad]` / `[m]` / `[N]`) を link にしないように書き、非公開の関数への link を外した
- feature `lol` の alice-lol を path をやめて git の rev (ALICE-LOL 41268d9) で取る CI は ALICE-LOL / ALICE-Zip / ALICE-SDF を clone しない (security-audit の sibling の checkout と `.github/alice-siblings` も外した) `deny.toml` の `allow-git` に ALICE-LOL / ALICE-SDF / ALICE-Zip を足した
- CI: security-audit の変更検出は変更した file の一覧を pipe でなく here-string で `grep -q` に渡す (pipefail の下で grep が先に終わると printf が SIGPIPE で失敗し、一致があっても依存の検査を飛ばしていた)
- CI: `scripts/advisory_hold_guard.py` は deny.toml の `[advisories]` に `unused-ignored-advisory = "deny"` が無ければ失敗し、workflow と `scripts/preflight.sh` の `cargo audit` の command (`\` で続く行は 1 つの command、comment 行と他の tool の `--ignore` は読まない) の `--ignore <id>` がすべて保留 (`scripts/advisory_holds.toml`) にあり、その crate の版が Cargo.lock にあることを検査する id は引用符の中でも読み、id として読めない `--ignore` の引数 (変数、配列、綴りの誤り、引数なし) は失敗にする 保留があるのに ignore を 1 件も読めなければ失敗 deny.toml に `unused-ignored-advisory = "deny"` を足した
- Licence: `MIT OR Apache-2.0` (feature `encoder` の module は `AGPL-3.0`) → `LicenseRef-Proprietary`
- CI: `scripts/run_tests.sh` と試験 `scripts/test_run_tests.py` を ALICE-LLM と同じ bytes にし、`.github/shared-files-pin` を ALICE-Law の新しい版に移す 名前 `lib` を library の単体試験として数えるようになったが、この repository は `lib` を名指さないので各段の試験の名前と件数は変わらない 試験 `scripts/test_public_text.py` を ALICE-SDF と同じ版にした (公開する text の検査を pin の checker で走らせ、置いた note link を tree と commit message で検出すること、pin の 3 行目 `config-sha256` が設定を承認することを確かめる、network を使う) `scripts/public_text.sh` と public-text の pin は変えないので検査の結果は変わらない
- `alice-law-conformance.toml`: evidence / mutant / note の行番号参照 100 件を anchor (`path#token`、ALICE-Law の roster が ref の file で 1 か所に解決するか確かめる) にした 行番号がずれて別の行を指していた参照 (clippy.toml・ci.yml・scripts/preflight.sh の範囲と列挙、src/jerk.rs の clamp 等) は文章が名指す構文 (step 名、fn、表の entry) に直した 測定の記録に出る行番号は、その行を名指す文に置き換えた 各 gate の status は変わらない
- CI: `.github/workflows/eol.yml` に top-level の concurrency group を足し (同じ branch の古い run は止め、main の run は commit ごとに残す)、action を commit SHA で固定して checkout に `persist-credentials: false` を付け、`workflow_dispatch` で手動でも走らせられるようにした (ALICE の crate で同じ中身の file、workflow_concurrency.py と workflow_pin_check.py を持つ crate の検査に合わせた)
- CI: `.gitattributes` に `* text=auto eol=lf` を置き、text の file を保存も checkout も全 OS で LF にした (共有 file は ALICE-Law の表と byte で比べるので、Windows の CRLF の checkout ではずれになる) binary と既存の行はそのまま、`git add --renormalize` で commit 済みの中身は変わらない
- CI (ALICE-Law G3): `tools/no-outer-fneg-of-fma` と `scripts/fneg_fma_ir_check.py` (と試験) を ALICE-LOL と同じ新しい版にした `expected-src-fma-calls` は src/ の library code の fma 呼び出し数 (`#[test]` と test の時だけ成り立つ cfg の下の item と文、test-only の `mod y;` で読み込む file とその子は数えず、test-only の数を別に表示する) この repository の数は変わらない IR 検査は library の source に fma が無いのに IR に fma がある時を失敗にし、library の fma が 0 の時は共有の probe を target ごとに compile して判定を確かめる (0 件実行は失敗) tool は独自の workspace なので repository 全体の format 検査に入っていなかった CI と preflight に tool の format の検査を足した 共有 file の照合の pin (`.github/shared-files-pin`) を、配布中の行を計画された行として持つ ALICE-Law b9794216 に上げた (前の pin の表ではこの新しい中身がずれに見える)
- CI: clippy の canary の表の生成器 `scripts/gen_clippy_canary_calls.py` が generic 関数 (`std::thread::spawn` / `scope` / `Builder::spawn` / `spawn_scoped`) を turbofish 付きで植えるようにした (素の path では型が決まらず compile できないため) 他の行は byte 一致で、この repo の表は変わらない 共有 file の協調更新 (ALICE-Analytics などと同じ中身)
- 版を 0.2.1 から 0.3.0 に (公開 API の追加と依存 `alice-det-math` の追加) `alice-law-conformance.toml` の G6 / G7 / G8 を pass にした dev-dependency に `sha2` を足した
- `rust-toolchain.toml` の pin を 1.92.0 から 1.98.1 に上げた 新しい clippy が指摘する手書きの midpoint `(a + b) * 0.5` は書き換えない (std の `f32::midpoint` は target ごとに実装が違い、結果が subnormal の範囲で bit が違う) crate の root に `clippy.toml` を置いて `f32::midpoint` を禁止し、4 箇所に理由付きで `manual_midpoint` を allow した `IntentEncoder::compression_estimate` の 1 秒あたりの motion 数は `checked_div` で求める (値は同じ) 出力の bit は変わらない
- CI: 公開 API の互換性を crates.io の最新の alice-kinematics と比べて失敗させる `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB ほかと同一) を追加した 既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせる (ALICE-Law release-train.toml は組を宣言しない) 版が公開済みの 0.2.1 と同じ間は次の最小の版 0.2.2 として検査する
- `lol` feature の `alice-lol` 要求を `0.4` から `>=0.4.1, <0.6` に広げ 公開済の 0.4.1 と 0.5 系の両方で解決できるようにした `IntentNode` への exhaustive match は 0.4.1 と 0.5 系の双方で build と test を確認済 code 変更なし
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加した 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し、CI が再生成して byte で比べる この repo は 1 件 (`disallowed-methods` の `f32::midpoint`、植える) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と同じ中身

### Fixed
- alice-lol-robot: latent の速度 (`values[3]`) の clamp を `alice_det_math::metric::clamp` にした (ALICE-Law G2) NaN は payload と符号に依らず canonical NaN (0x7fc0_0000) になる (以前は `f32::clamp` で payload が残った) CI と preflight に alice-lol-robot の clippy (clippy.toml の禁止一覧) を足した
- `distance_based_skinning` が NaN の座標を持つ頂点で、距離 NaN を候補として残し `partial_cmp(..).unwrap_or(Equal)` (全順序でない) で並べていたため、残る 4 本の骨が sort の実装に依存していた NaN の距離は範囲外として除き (Hips 100% の孤立頂点の扱い)、並べ替えは `total_cmp` にした 試験 `a_nan_vertex_has_no_candidate_bone`
- `ArmChain::inverse_kinematics` に NaN の目標を与えた時の角度が release build では `0xffc0_0000` (debug は `0x7fc0_0000`) になっていた (`src/joint.rs` の `sign * angle` を最適化が select と符号反転に書き換え、NaN の符号 bit が立つ) NaN の符号と payload は未規定なので、crate から出る値と joint が保存する角度で NaN を正準の `0x7fc0_0000` に置き換える (`src/nan.rs`、`alice_det_math::metric::clamp` を無限の上下で使う: -0 と ±inf はそのまま) 新しい release job が debug と release の両方で固定する

- `.github/workflows/g3-ir.yml` の on.push から paths filter を外した 既存の SHA に作った branch では GitHub の filter が空の差分を見て run を skip していた 代わりに `scripts/changed_paths.py` (ALICE-DetMath と同一) と `.github/paths/g3-ir.txt` で merge-base(origin/main) から実行可否を決める changes job と、走るべき IR job が走らなかった時に red になる gate job を足した `scripts/test_changed_paths.py` を CI と preflight で実行する
- `alice-law-conformance.toml` の G3 の evidence を、検査器の 3 file (`tools/no-outer-fneg-of-fma/src/main.rs` / `Cargo.toml` / `Cargo.lock`) それぞれの path で始まる 3 項目に分けた ALICE-Law の roster 検査は evidence の先頭の語を main にある path として読むので、`{Cargo.toml,Cargo.lock,src/main.rs}` とまとめた形は存在しない path として拒否されていた

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
