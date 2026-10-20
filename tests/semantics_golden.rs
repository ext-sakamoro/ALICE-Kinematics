//! Recorded bits of the decisions this crate computes itself
//!
//! Each scenario runs a fixed input set through a public entry point and
//! hashes the output bits with SHA-256. The digests are recorded here; a
//! change to the arithmetic of any scenario changes its digest. These
//! digests are the entries `KIN_SEMANTICS_PINS` (src/semantics.rs) folds, and
//! `tests/kin_semantics_id.rs` reads them from this file.
//!
//! These are change detectors, not correctness oracles: the analytic checks of
//! the same functions are in `tests/analytic_oracle.rs`.

use alice_kinematics::intent::ExtendedIntent;
use alice_kinematics::predictor::QuinticCoeffs;
use alice_kinematics::{ArmChain, Intent, IntentFlags, IntentType, Vec3k};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// The SHA-256 of `bytes` when it is not `expected`
fn golden_mismatch(bytes: &[u8], expected: &str) -> Option<String> {
    let got = hex(&Sha256::digest(bytes));
    (got != expected).then_some(got)
}

fn assert_golden(name: &str, bytes: &[u8], expected: &str) {
    if let Some(got) = golden_mismatch(bytes, expected) {
        panic!("scenario {name:?} moved to {got}: re-record only for an intended change");
    }
}

#[test]
fn checker_rejects_a_wrong_digest() {
    let recorded = hex(&Sha256::digest(b"recorded bits"));
    assert_eq!(golden_mismatch(b"recorded bits", &recorded), None);
    assert!(
        golden_mismatch(b"moved bits", &recorded).is_some(),
        "a scenario whose bits moved must not match its recorded digest"
    );
}

/// Positions from -130 to 130 m in steps that cross the Q8.8 saturation
/// bounds, plus signed zeros
fn coordinate_grid() -> Vec<f32> {
    let mut v: Vec<f32> = (-26_i16..=26)
        .map(|i| f32::from(i) * 5.0 + 0.123_456_79)
        .collect();
    v.extend_from_slice(&[0.0, -0.0, 127.996, -128.0, 1.0 / 256.0, -1.0 / 512.0]);
    v
}

#[test]
fn intent_encode_is_the_recorded_bits() {
    let grid = coordinate_grid();
    let mut out = Vec::new();
    for (i, &x) in grid.iter().enumerate() {
        let b = u8::try_from(i).expect("the grid has fewer than 256 points");
        let y = grid[(i * 7 + 3) % grid.len()];
        let z = grid[(i * 13 + 5) % grid.len()];
        let flags = IntentFlags::new(IntentType::from_u8(b), i % 2 == 0, i % 3 == 0, b);
        let intent = Intent {
            target: Vec3k::new(x, y, z),
            duration_ms: b.wrapping_mul(37),
            flags,
        };
        out.extend_from_slice(&intent.encode());
    }
    assert_golden(
        "intent encode",
        &out,
        "a4bdb7231e2252f6fda34ec652a2ca1aa2159f60c8c3a08a1ccb84cff751c5d4",
    );
}

#[test]
fn extended_intent_encode_is_the_recorded_bits() {
    let grid = coordinate_grid();
    let mut out = Vec::new();
    for (i, &x) in grid.iter().enumerate() {
        let base = Intent::reach(
            Vec3k::new(x, -x, x * 0.5),
            u8::try_from(i).expect("the grid has fewer than 256 points"),
        );
        let v = grid[(i * 11 + 1) % grid.len()];
        let ext = ExtendedIntent::new(base, Vec3k::new(v, v * -0.25, 0.75));
        out.extend_from_slice(&ext.encode());
    }
    assert_golden(
        "extended intent encode",
        &out,
        "421e2fe9bd513e92288fe41e7f87015d0c9cd68a025705cca88e3a9d7d19166f",
    );
}

#[test]
fn quintic_trajectory_is_the_recorded_bits() {
    let mut out = Vec::new();
    for (x0, v0, a0, xf, t) in [
        (0.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32),
        (0.5, 2.0, -3.0, -0.4, 0.25),
        (-1.0, -0.5, 4.0, 3.0, 0.15),
        (10.0, 0.0, 0.0, 10.0, 0.5),
        (0.0, 0.0, 0.0, 1.0, 1e-7),
    ] {
        let q = QuinticCoeffs::from_boundary(x0, v0, a0, xf, t);
        for c in q.c {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
        for k in 0..=16_u8 {
            let s = f32::from(k) / 16.0;
            for value in [q.position(s), q.velocity(s), q.acceleration(s)] {
                out.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    assert_golden(
        "quintic trajectory",
        &out,
        "70f8dfeaa9cfed81800d3fe470102b6df60988881a9bdc940ceefe4c3e82a0dd",
    );
}

#[test]
fn arm_inverse_kinematics_is_the_recorded_bits() {
    let mut out = Vec::new();
    for target in [
        Vec3k::new(0.3, -0.2, 0.1),
        Vec3k::new(0.0, -0.5, 0.0),
        Vec3k::new(-0.2, 0.1, 0.4),
        Vec3k::new(2.0, 2.0, 2.0),
        Vec3k::new(0.0, 0.0, 0.0),
    ] {
        let mut arm = ArmChain::right_arm();
        let (iterations, error) = arm.inverse_kinematics(target, 64, 1e-4);
        out.extend_from_slice(&iterations.to_le_bytes());
        out.extend_from_slice(&error.to_bits().to_le_bytes());
        for a in arm.angles() {
            out.extend_from_slice(&a.to_bits().to_le_bytes());
        }
        let p = arm.forward_kinematics();
        for c in [p.x, p.y, p.z] {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    assert_golden(
        "arm inverse kinematics",
        &out,
        "287771cc5d7cf2d0bdf00c95c8067f0b2154d3bc069a9ee8d8cca82343d44469",
    );
}

/// 4096 packets from a 64-bit linear congruential generator, decoded: the
/// bits a reader recovers from stored bytes
#[test]
fn packet_decode_is_the_recorded_bits() {
    let mut out = Vec::new();
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..4096 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let packet = state.to_be_bytes();
        let intent = Intent::decode(&packet);
        for c in [intent.target.x, intent.target.y, intent.target.z] {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&[intent.duration_ms, intent.flags.0]);
        let mut wide = [0u8; 16];
        wide[..8].copy_from_slice(&packet);
        wide[8..].copy_from_slice(&state.rotate_left(17).to_be_bytes());
        let ext = ExtendedIntent::decode(&wide);
        for c in [ext.velocity.x, ext.velocity.y, ext.velocity.z] {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    assert_golden(
        "packet decode",
        &out,
        "31ff5fcd3686b9d00bb0a9946ed9b899aa0ce4cc22ec556d332cec3030a2bc98",
    );
}
