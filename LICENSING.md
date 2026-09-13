# Licensing

`alice-kinematics` is dual-licensed under either:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

## Feature-gated AGPL modules

Enabling the **`encoder` feature** activates modules licensed under
[GNU Affero General Public License version 3](LICENSE-AGPL) instead of MIT / Apache-2.0.

Affected modules:
- `src/encoder.rs`
- `src/jerk.rs`

If you enable the `encoder` feature, your entire application becomes subject
to AGPL-3.0 network copyleft obligations (source-provision when running as a
network service).

Default features do NOT include `encoder`, so downstream users who omit the
feature can stay on MIT / Apache-2.0 terms.

## Commercial licensing

For commercial licensing to use the `encoder` modules without AGPL-3.0
obligations, contact: `sakamoro@alicelaw.net`
