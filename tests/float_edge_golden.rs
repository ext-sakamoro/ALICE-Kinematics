//! Signed-zero and exact-cancellation cases pinned to fixed bit patterns
//! (ALICE-Law G4)
//!
//! Each case runs in CI under debug and release on `x86_64` (ubuntu) and `aarch64`
//! (macOS): `scripts/run_tests.sh float_edge_golden=4` fails a leg that ran
//! fewer of them. Besides the recorded bits, each case asserts what holds
//! independently of the recording: +0 and -0 encode to the same packet, and a
//! rest-to-rest quintic ends exactly at its target with exactly zero velocity
//! and acceleration.

use alice_kinematics::predictor::QuinticCoeffs;
use alice_kinematics::{Intent, IntentFlags, IntentType, Vec3k};

fn intent_at(x: f32, y: f32, z: f32) -> Intent {
    Intent {
        target: Vec3k::new(x, y, z),
        duration_ms: 0,
        flags: IntentFlags::new(IntentType::from_u8(0), false, false, 0),
    }
}

/// +0 and -0 coordinates quantise to the same Q8.8 packet, and the packet
/// decodes to +0 (the sign of a zero carries nothing through the wire)
#[test]
fn signed_zero_coordinates_encode_and_decode_to_the_recorded_bits() {
    let plus = intent_at(0.0, 0.0, 0.0).encode();
    let minus = intent_at(-0.0, -0.0, -0.0).encode();
    let mixed = intent_at(-0.0, 0.0, -0.0).encode();
    assert_eq!(plus, minus, "+0 and -0 encode differently");
    assert_eq!(plus, mixed, "a mix of signed zeros encodes differently");
    assert_eq!(plus, RECORDED_ZERO_PACKET, "the zero packet moved");
    let back = Intent::decode(&minus).target;
    let bits = [back.x.to_bits(), back.y.to_bits(), back.z.to_bits()];
    assert_eq!(
        bits, [0_u32; 3],
        "a zero packet decodes to {bits:08x?}, not +0"
    );
}

/// A rest-to-rest quintic from 0 to 1 over T = 1 has c3 = 10, c4 = -15,
/// c5 = 6: at T the position is 10 - 15 + 6 = 1 and the velocity and
/// acceleration are 30 - 60 + 30 and 60 - 180 + 120, every partial sum exact
#[test]
fn a_rest_to_rest_end_cancels_exactly() {
    let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
    let end = [q.position(1.0), q.velocity(1.0), q.acceleration(1.0)];
    assert_eq!(
        end.map(f32::to_bits),
        [1.0_f32, 0.0, 0.0].map(f32::to_bits),
        "the end of the move is not exact"
    );
    let bits = end.map(f32::to_bits);
    assert_eq!(
        bits, RECORDED_REST_TO_REST_END,
        "the end bits moved (a zero changed sign?)"
    );
}

/// No displacement: every higher coefficient is a cancellation to zero, so
/// the path stays exactly at the start; a duration below the threshold takes
/// the degenerate branch and holds the target
#[test]
fn zero_displacement_and_zero_duration_hold_exactly() {
    let still = QuinticCoeffs::from_boundary(1.5, 0.0, 0.0, 1.5, 0.5);
    let negative_zero_start = QuinticCoeffs::from_boundary(1.5, -0.0, -0.0, 1.5, 0.5);
    let instant = QuinticCoeffs::from_boundary(-2.25, 4.0, 1.0, 3.75, 0.0);
    let mut bits = Vec::new();
    for t in [0.0_f32, 0.125, 0.25, 0.5] {
        for q in [&still, &negative_zero_start] {
            assert_eq!(
                q.position(t).to_bits(),
                1.5_f32.to_bits(),
                "t = {t}: the path left the start"
            );
            bits.extend([q.position(t), q.velocity(t), q.acceleration(t)].map(f32::to_bits));
        }
        assert_eq!(
            instant.position(t).to_bits(),
            3.75_f32.to_bits(),
            "t = {t}: the degenerate path is not the target"
        );
        bits.extend(
            [
                instant.position(t),
                instant.velocity(t),
                instant.acceleration(t),
            ]
            .map(f32::to_bits),
        );
    }
    assert_eq!(bits, RECORDED_HOLD_BITS, "the held path's bits moved");
}

/// the zero packet: every coordinate quantises to 0, duration and flags 0
/// A NaN start (sign bit and payload set) gives the canonical quiet NaN at every
/// time, in position, velocity and acceleration: the NaN that leaves the crate is
/// `0x7fc0_0000` whatever its origin and whatever the codegen (ALICE-Law G4)
#[test]
fn a_nan_start_gives_the_canonical_nan() {
    let q = QuinticCoeffs::from_boundary(f32::from_bits(0xffc0_dead), 0.0, 0.0, 1.0, 1.0);
    for t in [0.0_f32, 0.5, 1.0] {
        let bits = [q.position(t), q.velocity(t), q.acceleration(t)].map(f32::to_bits);
        assert_eq!(bits, [0x7fc0_0000; 3], "t = {t}: {bits:08x?}");
    }
}

const RECORDED_ZERO_PACKET: [u8; 8] = [0; 8];
/// position 1.0, velocity +0, acceleration +0
const RECORDED_REST_TO_REST_END: [u32; 3] = [0x3f80_0000, 0, 0];
/// per t in (0, 0.125, 0.25, 0.5): the still path, the path started with
/// -0 velocity and acceleration (both 1.5 = `0x3fc0_0000` with +0 velocity and
/// acceleration: -0 + +0 is +0), then the degenerate path at 3.75 = `0x4070_0000`
#[rustfmt::skip]
const RECORDED_HOLD_BITS: [u32; 36] = [
    0x3fc0_0000, 0, 0, 0x3fc0_0000, 0, 0, 0x4070_0000, 0, 0,
    0x3fc0_0000, 0, 0, 0x3fc0_0000, 0, 0, 0x4070_0000, 0, 0,
    0x3fc0_0000, 0, 0, 0x3fc0_0000, 0, 0, 0x4070_0000, 0, 0,
    0x3fc0_0000, 0, 0, 0x3fc0_0000, 0, 0, 0x4070_0000, 0, 0,
];
