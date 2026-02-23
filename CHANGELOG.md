# Changelog

All notable changes to ALICE-Kinematics will be documented in this file.

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
- 41 unit tests
