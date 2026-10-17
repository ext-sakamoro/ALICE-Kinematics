//! The NaN this crate returns (ALICE-Law G2): every `f32` that leaves the crate through a
//! public function, and every angle a joint stores, is either a number or the canonical
//! quiet NaN `0x7fc0_0000`
//!
//! The sign and payload of a NaN produced by arithmetic are not specified: the same source
//! gives different bits under different codegen. Measured in this crate: with a NaN target,
//! `ArmChain::inverse_kinematics` multiplied the canonical NaN angle by a ±1 sign
//! (`src/joint.rs`, `delta = sign * angle * limit_w`); the release build turns that
//! multiplication into a select of `angle` or `fneg angle`, and the `fneg` sets the NaN's
//! sign bit, so the stored angles were `0x7fc0_0000` in debug and `0xffc0_0000` in release.
//! A NaN is replaced here, at the outputs, rather than at each operation that may produce
//! one: whatever the codegen does inside, what comes out is the same bits.
//!
//! The replacement is `alice_det_math::metric::clamp` with infinite bounds, the same
//! canonicalisation the G2 helpers apply: a NaN becomes `f32::NAN`, every other value (a
//! signed zero and the infinities included) is returned unchanged.
//! `scripts/nan_boundary_scan.py` lists the public functions returning `f32` values and
//! fails when one has no NaN test in `tests/nan_boundaries.rs`.

use crate::joint::Vec3k;

/// `x`, or the canonical quiet NaN when `x` is a NaN
#[inline]
#[must_use]
pub(crate) fn canon(x: f32) -> f32 {
    alice_det_math::metric::clamp(x, f32::NEG_INFINITY, f32::INFINITY)
}

/// [`canon`] on each component
#[inline]
#[must_use]
pub(crate) fn canon3(v: Vec3k) -> Vec3k {
    Vec3k::new(canon(v.x), canon(v.y), canon(v.z))
}
