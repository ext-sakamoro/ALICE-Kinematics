//! ALICE-Law G2 tie tests: the max / min / clamp sites, through the public
//! API, at the inputs where `alice-det-math`'s helpers and other ways of
//! writing max / min disagree (an equal pair of zeros, a NaN operand)
//!
//! The clippy ban stops `f32::max` / `min` / `clamp` at a call site, but not
//! a comparison written out by hand (`if x < 0.0 { 0.0 } else { x }`), which
//! differs from the helper on -0 or on NaN. These tests compare bits at the
//! public API so that such a rewrite of a site changes a result here.
//!
//! Site classes (src/) with a test here:
//! - clamp of a caller's value into [0, 1]: `HandModel::grip` (hand.rs)
//! - clamp of a cosine into [-1, 1] before the arc cosine:
//!   `ArmChain::inverse_kinematics` (joint.rs)
//! - min of a duration ratio against 1: `JerkFitter::fit_trajectory`
//!   (jerk.rs, encoder feature)
//! - clamp of a projection parameter into [0, 1]:
//!   `point_to_segment_distance_sq` (autorig/skinning.rs): the test pins the
//!   public bits, but no rewrite of this site changes them (the squared
//!   distance drops the sign of zero and the point's own NaN reaches the
//!   result first)
//!
//! The remaining sites cannot see a tie input from the public API; the
//! reason for each is in alice-law-conformance.toml (G2).

use alice_kinematics::autorig::skinning::point_to_segment_distance_sq;
use alice_kinematics::hand::HandModel;
use alice_kinematics::Vec3k;

/// `f32::NAN` as `alice-det-math` returns it (its NaN policy)
const CANONICAL_NAN: u32 = 0x7fc0_0000;

/// A quiet NaN with a payload, which a hand-written comparison would carry
/// through unchanged
fn payload_nan() -> f32 {
    f32::from_bits(0x7fc0_dead)
}

fn grip_angles(amount: f32) -> Vec<u32> {
    let mut hand = HandModel::right_hand();
    hand.grip(amount);
    hand.fingers
        .iter()
        .flat_map(|f| f.joints[..3].iter().map(|j| j.angle.to_bits()))
        .collect()
}

/// -0 stays -0: the helper keeps the sign of an in-range zero, and every
/// joint range reached here includes 0 (the MCP lower limit is negative, the
/// PIP and DIP lower limit is 0, which a comparison keeps -0 at)
#[test]
fn grip_of_negative_zero_keeps_the_sign_of_zero() {
    let angles = grip_angles(-0.0);
    assert_eq!(angles.len(), 15);
    for (i, a) in angles.iter().enumerate() {
        assert_eq!(*a, (-0.0_f32).to_bits(), "joint {i}: {a:#010x}");
    }
    for (i, a) in grip_angles(0.0).iter().enumerate() {
        assert_eq!(*a, 0.0_f32.to_bits(), "joint {i}: {a:#010x}");
    }
}

/// A NaN amount gives the canonical NaN in every joint, whatever its
/// payload: the helper replaces it, a hand-written comparison passes it on
#[test]
fn grip_of_nan_gives_the_canonical_nan() {
    for (i, a) in grip_angles(payload_nan()).iter().enumerate() {
        assert_eq!(*a, CANONICAL_NAN, "joint {i}: {a:#010x}");
    }
}

/// A projection parameter of ±0 (the point is level with the segment start)
/// or NaN (a NaN point) through the distance
#[test]
fn segment_distance_at_zero_and_nan_projection() {
    let a = Vec3k::new(0.0, 0.0, 0.0);
    let b = Vec3k::new(0.0, 1.0, 0.0);
    // ab · ap = -0 and +0
    let neg = point_to_segment_distance_sq(Vec3k::new(1.0, -0.0, 0.0), a, b);
    let pos = point_to_segment_distance_sq(Vec3k::new(1.0, 0.0, 0.0), a, b);
    assert_eq!(neg.to_bits(), 1.0_f32.to_bits());
    assert_eq!(pos.to_bits(), 1.0_f32.to_bits());
    // the point's NaN reaches the result before the projection's does, so
    // its payload is what comes out
    let nan = point_to_segment_distance_sq(Vec3k::new(payload_nan(), 0.5, 0.0), a, b);
    assert!(nan.is_nan());
}

/// The cosine clamp of the arm inverse kinematics (`clamp(dot, -1, 1)`
/// before the arc cosine) with a NaN target
#[test]
fn arm_inverse_kinematics_with_a_nan_target() {
    let mut arm = alice_kinematics::ArmChain::right_arm();
    let (iterations, error) = arm.inverse_kinematics(Vec3k::new(payload_nan(), 0.2, 0.1), 8, 1e-4);
    let angles: Vec<u32> = arm.angles().iter().map(|a| a.to_bits()).collect();
    assert_eq!(iterations, 8, "error {error}");
    assert!(error.is_nan());
    // the clamp replaces the NaN cosine by the canonical NaN, which then
    // reaches every angle; a comparison carries the target's payload
    assert_eq!(
        angles,
        vec![CANONICAL_NAN; angles.len()],
        "angles {angles:#010x?}"
    );
}

/// The progress clamp of the jerk fit (`min(duration / total, 1)`, then
/// `max(.., 1e-6)`) with a NaN timestamp: the duration and the estimated
/// total are NaN, so `min` sees a NaN operand and the helper gives 1, which
/// keeps the predicted target finite. A comparison written as
/// `if a > 1.0 { 1.0 } else { a }` passes the NaN to the target.
#[cfg(feature = "encoder")]
#[test]
fn jerk_fit_progress_with_a_nan_timestamp() {
    use alice_kinematics::jerk::{JerkFitter, MotionSample};
    let mut fitter = JerkFitter::new();
    for i in 0..8_u8 {
        let t = f32::from(i) * 0.01;
        fitter.push(MotionSample {
            pos: Vec3k::new(f32::from(i) * 0.02, 0.0, 0.0),
            time: if i == 7 { payload_nan() } else { t },
        });
    }
    let fit = fitter.fit_trajectory().expect("a fit");
    // displacement 0.14 along x from the origin, progress 1: target = 0.14
    assert_eq!(
        fit.target.x.to_bits(),
        (0.14_f32).to_bits(),
        "{:?}",
        fit.target
    );
    assert_eq!(fit.target.y.to_bits(), 0.0_f32.to_bits());
}
