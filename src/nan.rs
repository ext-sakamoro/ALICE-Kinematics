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

/// [`canon`] for a `const fn`: the same bits (a NaN becomes `f32::NAN` = `0x7fc0_0000`, every
/// other value unchanged), written with `is_nan` because `alice_det_math::metric::clamp` is
/// not `const` (`tests::const_form_agrees` compares the two)
#[inline]
#[must_use]
pub(crate) const fn canon_const(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else {
        x
    }
}

/// [`canon`] on each component
#[inline]
#[must_use]
pub(crate) fn canon3(v: Vec3k) -> Vec3k {
    Vec3k::new(canon(v.x), canon(v.y), canon(v.z))
}

#[cfg(test)]
mod tests {
    use super::{canon, canon_const};

    #[test]
    fn const_form_agrees() {
        let inputs = [
            0.0_f32,
            -0.0,
            1.5,
            -1.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0xffc0_dead),
            f32::from_bits(0x7f80_0001),
        ];
        for x in inputs {
            assert_eq!(
                canon_const(x).to_bits(),
                canon(x).to_bits(),
                "input bits {:08x}",
                x.to_bits()
            );
        }
        assert_eq!(
            canon_const(f32::from_bits(0xffc0_dead)).to_bits(),
            0x7fc0_0000
        );
    }
}
