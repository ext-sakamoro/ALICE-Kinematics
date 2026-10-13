# autorig rerun 可視化 example (dev 専用、非公開)

`alice-kinematics` の autorig 結果 (stick figure mesh / 22-joint skeleton / skinning weight) を
[rerun](https://github.com/rerun-io/rerun) viewer に log する dev 専用の example crate

公開ライブラリ `alice-kinematics` の feature には含めない rerun + native_viewer は重い依存で
MSRV も新しいため、本 crate を独自 workspace + `rust-toolchain.toml` で単独 build する

## 実行

```bash
cd examples/rerun-viz
cargo run --release
```

rerun viewer が spawn され、`mesh/points` / `skeleton/bones` / `skeleton/joints` / `stats/text` が log される
