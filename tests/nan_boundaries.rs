//! Every NaN that leaves the crate is the canonical quiet NaN `0x7fc0_0000` (ALICE-Law G2,
//! `src/nan.rs`)
//!
//! One test per public function returning an `f32` value (`scripts/nan_boundary_scan.py`
//! lists them and fails when one has no test here; its struct-returning outputs not yet
//! converted are its counted DEBT list). Each test feeds a NaN with the sign bit and a
//! payload set (`0xffc0_dead`) through the function's inputs or the public fields it reads,
//! asserts that a NaN does come out (so the test is not vacuous), and that every NaN that
//! comes out has the canonical bits. The release build once turned a NaN's sign bit on
//! inside the arm IK (`src/joint.rs`), so these run under debug and release on `x86_64` and
//! `aarch64` (`scripts/run_tests.sh nan_boundaries=49` in each leg; 58 with `encoder`, 50 with
//! `lol`).

use alice_kinematics::autorig::hypothesis::{
    build_hypothesis, hypothesis_to_skeleton, SkeletonHypothesis,
};
use alice_kinematics::autorig::skinning::point_to_segment_distance_sq;
use alice_kinematics::autorig::slice::{
    normalize_mesh, slice_normalized, NormalizedMesh, SliceStat,
};
use alice_kinematics::autorig::sphere::{auto_rig_on_sphere, SphereContext, TangentFrame};
use alice_kinematics::autorig::{auto_rig, AutoRigConfig, MeshView};
use alice_kinematics::hand::{Finger, FingerType, HandModel};
use alice_kinematics::joint::rotate_vec;
use alice_kinematics::predictor::QuinticCoeffs;
use alice_kinematics::skeleton::{BoneId, Skeleton};
use alice_kinematics::{
    ArmChain, Intent, IntentFlags, IntentType, JointConstraint, Predictor, Vec3k,
};

const CANONICAL: u32 = 0x7fc0_0000;

/// a NaN no arithmetic produces by default: sign bit and payload set
fn odd_nan() -> f32 {
    f32::from_bits(0xffc0_dead)
}

/// every value is a number or the canonical NaN; returns how many were NaN
fn check(what: &str, values: &[f32]) -> usize {
    let mut nans = 0;
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            nans += 1;
            assert_eq!(
                v.to_bits(),
                CANONICAL,
                "{what}[{i}]: a NaN with bits {:08x}",
                v.to_bits()
            );
        }
    }
    nans
}

/// as [`check`], and at least one NaN came out
fn assert_canonical(what: &str, values: &[f32]) {
    assert!(
        check(what, values) > 0,
        "{what}: no NaN came out, the test checks nothing"
    );
}

fn v3(v: Vec3k) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn nan3() -> Vec3k {
    Vec3k::new(odd_nan(), 1.0, odd_nan())
}

// ---- Vec3k -------------------------------------------------------------------------------

#[test]
fn boundary_vec3k_length_sq() {
    assert_canonical("length_sq", &[nan3().length_sq()]);
}

#[test]
fn boundary_vec3k_length() {
    assert_canonical("length", &[nan3().length()]);
}

#[test]
fn boundary_vec3k_normalize() {
    assert_canonical(
        "normalize",
        &v3(Vec3k::new(odd_nan(), 3.0, 4.0).normalize()),
    );
}

#[test]
fn boundary_vec3k_dot() {
    assert_canonical("dot", &[nan3().dot(Vec3k::new(1.0, 2.0, 3.0))]);
}

#[test]
fn boundary_vec3k_cross() {
    assert_canonical("cross", &v3(nan3().cross(Vec3k::new(1.0, -2.0, 3.0))));
}

#[test]
fn boundary_vec3k_distance() {
    assert_canonical("distance", &[nan3().distance(Vec3k::ZERO)]);
}

#[test]
fn boundary_vec3k_lerp() {
    assert_canonical(
        "lerp",
        &v3(Vec3k::new(1.0, 2.0, 3.0).lerp(Vec3k::ZERO, odd_nan())),
    );
}

#[test]
fn boundary_vec3k_scale() {
    assert_canonical("scale", &v3(Vec3k::new(1.0, -2.0, 3.0).scale(odd_nan())));
}

#[test]
fn boundary_vec3k_add() {
    assert_canonical("add", &v3(nan3() + Vec3k::new(1.0, 2.0, 3.0)));
}

#[test]
fn boundary_vec3k_sub() {
    assert_canonical("sub", &v3(Vec3k::new(1.0, 2.0, 3.0) - nan3()));
}

#[test]
fn boundary_vec3k_mul() {
    assert_canonical("mul", &v3(Vec3k::new(1.0, -2.0, 3.0) * odd_nan()));
}

/// negation is a bare sign flip: of a canonical NaN it would give `0xffc0_0000`
#[test]
fn boundary_vec3k_neg() {
    let canonical_in = Vec3k::new(f32::NAN, 1.0, f32::NAN);
    assert_canonical("neg of the canonical NaN", &v3(-canonical_in));
    assert_canonical("neg", &v3(-nan3()));
}

#[test]
fn boundary_rotate_vec() {
    assert_canonical(
        "rotate_vec",
        &v3(rotate_vec(
            Vec3k::new(1.0, 0.0, 0.0),
            Vec3k::new(0.0, 0.0, 1.0),
            odd_nan(),
        )),
    );
}

// ---- joints and the arm ------------------------------------------------------------------

#[test]
fn boundary_jointconstraint_clamp() {
    let c = JointConstraint {
        min_rad: -1.0,
        max_rad: 1.0,
    };
    assert_canonical("clamp", &[c.clamp(odd_nan())]);
}

#[test]
fn boundary_jointconstraint_range() {
    let c = JointConstraint {
        min_rad: odd_nan(),
        max_rad: 1.0,
    };
    assert_canonical("range", &[c.range()]);
}

/// the measured case: a NaN target, release built the stored angles as `0xffc0_0000`
#[test]
fn boundary_armchain_inverse_kinematics() {
    let mut arm = ArmChain::right_arm();
    let (_, error) = arm.inverse_kinematics(Vec3k::new(odd_nan(), 0.2, 0.1), 8, 1e-4);
    assert_canonical("inverse_kinematics error", &[error]);
    assert_canonical("angles after a NaN target", &arm.angles());
}

#[test]
fn boundary_armchain_angles() {
    let mut arm = ArmChain::right_arm();
    arm.joints[2].angle = odd_nan();
    assert_canonical("angles", &arm.angles());
}

#[test]
fn boundary_armchain_forward_kinematics() {
    let mut arm = ArmChain::right_arm();
    arm.joints[0].angle = odd_nan();
    assert_canonical("forward_kinematics", &v3(arm.forward_kinematics()));
}

#[test]
fn boundary_armchain_joint_position() {
    let mut arm = ArmChain::right_arm();
    arm.joints[0].link_length = odd_nan();
    assert_canonical("joint_position", &v3(arm.joint_position(2)));
}

#[test]
fn boundary_armchain_total_length() {
    let mut arm = ArmChain::right_arm();
    arm.joints[1].link_length = odd_nan();
    assert_canonical("total_length", &[arm.total_length()]);
}

// ---- hand ----------------------------------------------------------------------------------

#[test]
fn boundary_finger_angles() {
    let mut f = Finger::new(FingerType::Index);
    f.joints[1].angle = odd_nan();
    assert_canonical("angles", &f.angles());
}

#[test]
fn boundary_finger_tip_position() {
    let mut f = Finger::new(FingerType::Index);
    f.joints[0].angle = odd_nan();
    assert_canonical("tip_position", &v3(f.tip_position()));
}

#[test]
fn boundary_finger_total_length() {
    let mut f = Finger::new(FingerType::Index);
    f.joints[2].link_length = odd_nan();
    assert_canonical("total_length", &[f.total_length()]);
}

#[test]
fn boundary_handmodel_fingertip_positions() {
    let mut hand = HandModel::right_hand();
    hand.fingers[1].joints[0].angle = odd_nan();
    let tips: Vec<f32> = hand
        .fingertip_positions()
        .iter()
        .flat_map(|v| v3(*v))
        .collect();
    assert_canonical("fingertip_positions", &tips);
}

// ---- intent and predictor ----------------------------------------------------------------

/// computed from the u8 duration: no NaN can come out, so every value is checked to be a
/// number (the boundary stays listed so a change to the input type is seen here)
#[test]
fn boundary_intent_duration_secs() {
    for ms in 0..=u8::MAX {
        let intent = Intent {
            target: Vec3k::ZERO,
            duration_ms: ms,
            flags: IntentFlags::new(IntentType::from_u8(0), false, false, 0),
        };
        let s = intent.duration_secs();
        assert!(
            !s.is_nan() && check("duration_secs", &[s]) == 0,
            "{ms} ms gave {s}"
        );
    }
}

#[test]
fn boundary_quinticcoeffs_position() {
    let q = QuinticCoeffs::from_boundary(odd_nan(), 0.0, 0.0, 1.0, 0.5);
    assert_canonical("position", &[q.position(0.25)]);
}

#[test]
fn boundary_quinticcoeffs_velocity() {
    let q = QuinticCoeffs::from_boundary(0.0, odd_nan(), 0.0, 1.0, 0.5);
    assert_canonical("velocity", &[q.velocity(0.25)]);
}

#[test]
fn boundary_quinticcoeffs_acceleration() {
    let q = QuinticCoeffs::from_boundary(0.0, 0.0, odd_nan(), 1.0, 0.5);
    assert_canonical("acceleration", &[q.acceleration(0.25)]);
}

fn moving_predictor() -> Predictor {
    let mut p = Predictor::new();
    p.apply_intent(Intent {
        target: Vec3k::new(1.0, 2.0, -1.0),
        duration_ms: 200,
        flags: IntentFlags::new(IntentType::from_u8(0), false, false, 0),
    });
    p
}

#[test]
fn boundary_predictor_position_at() {
    let p = moving_predictor();
    assert_canonical("position_at", &v3(p.position_at(odd_nan())));
}

#[test]
fn boundary_predictor_velocity_at() {
    let p = moving_predictor();
    assert_canonical("velocity_at", &v3(p.velocity_at(odd_nan())));
}

#[test]
fn boundary_predictor_remaining() {
    let mut p = moving_predictor();
    p.update(odd_nan());
    assert_canonical("remaining", &[p.remaining()]);
}

#[test]
fn boundary_predictor_progress() {
    let mut p = moving_predictor();
    p.update(odd_nan());
    assert_canonical("progress", &[p.progress()]);
}

// ---- skeleton and auto-rig ---------------------------------------------------------------

#[test]
fn boundary_skeleton_joint_world_position() {
    let s = Skeleton::with_height(odd_nan());
    let head = s
        .joint_world_position(BoneId::Head)
        .expect("the humanoid has a head");
    assert_canonical("joint_world_position", &v3(head));
}

#[test]
fn boundary_skeleton_distance_between() {
    let s = Skeleton::with_height(odd_nan());
    let d = s
        .distance_between(BoneId::Head, BoneId::LeftHand)
        .expect("both bones exist");
    assert_canonical("distance_between", &[d]);
}

#[test]
fn boundary_point_to_segment_distance_sq() {
    let d = point_to_segment_distance_sq(nan3(), Vec3k::ZERO, Vec3k::new(1.0, 0.0, 0.0));
    assert_canonical("point_to_segment_distance_sq", &[d]);
}

fn nan_frame() -> TangentFrame {
    TangentFrame {
        right: Vec3k::new(1.0, 0.0, 0.0),
        up: Vec3k::new(0.0, 1.0, 0.0),
        forward: Vec3k::new(0.0, 0.0, 1.0),
        origin: nan3(),
    }
}

#[test]
fn boundary_tangentframe_world_to_local() {
    assert_canonical(
        "world_to_local",
        &v3(nan_frame().world_to_local(Vec3k::new(1.0, 2.0, 3.0))),
    );
}

#[test]
fn boundary_tangentframe_local_to_world() {
    assert_canonical(
        "local_to_world",
        &v3(nan_frame().local_to_world(Vec3k::new(1.0, 2.0, 3.0))),
    );
}

fn nan_mesh() -> NormalizedMesh {
    NormalizedMesh {
        vertices: Vec::new(),
        original_height: odd_nan(),
        original_min: Vec3k::ZERO,
        original_max: Vec3k::new(1.0, 2.0, 1.0),
    }
}

#[test]
fn boundary_normalizedmesh_denormalize() {
    assert_canonical(
        "denormalize",
        &v3(nan_mesh().denormalize(Vec3k::new(0.5, 0.5, 0.5))),
    );
}

#[test]
fn boundary_normalizedmesh_horizontal_extent() {
    assert_canonical("horizontal_extent", &[nan_mesh().horizontal_extent()]);
}

#[test]
fn boundary_quinticcoeffs_from_boundary() {
    // the start position and velocity are stored as c0 and c1; the end value alone for a
    // zero duration
    let moving = QuinticCoeffs::from_boundary(odd_nan(), odd_nan(), 0.0, 1.0, 0.5);
    assert_canonical("from_boundary", &moving.c);
    let instant = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, odd_nan(), 0.0);
    assert_canonical("from_boundary (zero duration)", &instant.c);
}

// ---- skeleton and auto-rig ------------------------------------------------------------------

/// a standing cylinder (radius 0.15, height 1.5, 30 rings of 12) that `auto_rig` rigs, one vertex
/// of its middle ring moved to x = NaN
fn cylinder_with_nan() -> Vec<Vec3k> {
    let mut v = Vec::new();
    for k in 0..30_u8 {
        let y = f32::from(k) / 30.0 * 1.5;
        for a in 0..12_u8 {
            // a regular 12-gon from its exact corner table (no trigonometry)
            const C: [(f32, f32); 12] = [
                (1.0, 0.0),
                (0.866_025_4, 0.5),
                (0.5, 0.866_025_4),
                (0.0, 1.0),
                (-0.5, 0.866_025_4),
                (-0.866_025_4, 0.5),
                (-1.0, 0.0),
                (-0.866_025_4, -0.5),
                (-0.5, -0.866_025_4),
                (0.0, -1.0),
                (0.5, -0.866_025_4),
                (0.866_025_4, -0.5),
            ];
            let (c, s) = C[usize::from(a)];
            v.push(Vec3k::new(c * 0.15, y, s * 0.15));
        }
    }
    v[15 * 12].x = odd_nan();
    v
}

fn skeleton_values(s: &Skeleton) -> Vec<f32> {
    let mut out = vec![s.height];
    for j in &s.joints {
        for p in [j.local_position, j.local_rotation, j.world_position] {
            out.extend(v3(p));
        }
        out.push(j.bone_length);
    }
    out
}

fn mesh_values(nm: &NormalizedMesh) -> Vec<f32> {
    let mut out = vec![nm.original_height];
    out.extend(v3(nm.original_min));
    out.extend(v3(nm.original_max));
    for &p in &nm.vertices {
        out.extend(v3(p));
    }
    out
}

fn slice_values(slices: &[SliceStat]) -> Vec<f32> {
    let mut out = Vec::new();
    for s in slices {
        out.extend([s.y, s.width]);
        out.extend(v3(s.centroid));
        for c in [s.left_centroid, s.right_centroid].into_iter().flatten() {
            out.extend(v3(c));
        }
    }
    out
}

fn hypothesis_values(h: &SkeletonHypothesis) -> Vec<f32> {
    let mut out = vec![h.confidence];
    for p in h.positions.iter().flatten() {
        out.extend(v3(*p));
    }
    out
}

/// the cylinder normalised, with the NaN vertex as given (the payload not yet canonical)
fn raw_normalized() -> NormalizedMesh {
    let v = cylinder_with_nan();
    let nm = normalize_mesh(&MeshView::from_vertices(&v)).expect("a cylinder has height");
    let mut vertices = nm.vertices;
    vertices[15 * 12].x = odd_nan();
    NormalizedMesh { vertices, ..nm }
}

#[test]
fn boundary_skeleton_with_height() {
    assert_canonical(
        "with_height",
        &skeleton_values(&Skeleton::with_height(odd_nan())),
    );
}

/// `rescale` stores what it computes; `find_joint` (exempt: a reference to the stored joint) then
/// reads it back canonical
#[test]
fn a_rescaled_skeleton_stores_the_canonical_nan() {
    let mut s = Skeleton::default_humanoid();
    s.rescale(odd_nan());
    let j = s.find_joint(BoneId::Spine).expect("the spine");
    let mut out = vec![s.height, j.bone_length];
    for p in [j.local_position, j.world_position] {
        out.extend(v3(p));
    }
    assert_canonical("rescale", &out);
}

#[test]
fn boundary_normalize_mesh() {
    let v = cylinder_with_nan();
    let nm = normalize_mesh(&MeshView::from_vertices(&v)).expect("a cylinder has height");
    assert_canonical("normalize_mesh", &mesh_values(&nm));
}

#[test]
fn boundary_slice_normalized() {
    assert_canonical(
        "slice_normalized",
        &slice_values(&slice_normalized(&raw_normalized(), 20)),
    );
}

#[test]
fn boundary_build_hypothesis() {
    let nm = raw_normalized();
    let mut slices = slice_normalized(&nm, 20);
    for s in &mut slices {
        // the centroids as a caller may hold them: the payload NaN, not the canonical one
        s.centroid.x = odd_nan();
    }
    let h = build_hypothesis(&nm, &slices).expect("the cylinder has a waist and a neck");
    assert_canonical("build_hypothesis", &hypothesis_values(&h));
}

#[test]
fn boundary_hypothesis_to_skeleton() {
    let nm = raw_normalized();
    let mut h = SkeletonHypothesis::default();
    h.positions[0] = Some(nan3());
    assert_canonical(
        "hypothesis_to_skeleton",
        &skeleton_values(&hypothesis_to_skeleton(&h, &nm)),
    );
}

#[test]
fn boundary_auto_rig() {
    let v = cylinder_with_nan();
    let r = auto_rig(&MeshView::from_vertices(&v), &AutoRigConfig::default()).expect("rigged");
    let mut out = vec![r.confidence];
    out.extend(skeleton_values(r.skeleton.as_ref().expect("a skeleton")));
    out.extend(
        r.skinning
            .iter()
            .flat_map(|s| s.weights.iter().flatten().copied()),
    );
    assert_canonical("auto_rig", &out);
}

#[test]
fn boundary_auto_rig_on_sphere() {
    let v = cylinder_with_nan();
    let ctx = SphereContext {
        center: Vec3k::new(0.0, -10.0, 0.0),
        character_pos: Vec3k::ZERO,
        radius: 10.0,
    };
    let r = auto_rig_on_sphere(
        &MeshView::from_vertices(&v),
        &ctx,
        &AutoRigConfig::default(),
    )
    .expect("rigged");
    let mut out = vec![r.confidence];
    out.extend(skeleton_values(r.skeleton.as_ref().expect("a skeleton")));
    assert_canonical("auto_rig_on_sphere", &out);
}

#[test]
fn boundary_tangentframe_from_context() {
    let ctx = SphereContext {
        center: Vec3k::new(0.0, -10.0, 0.0),
        character_pos: nan3(),
        radius: 10.0,
    };
    let f = TangentFrame::from_context(&ctx);
    let mut out = Vec::new();
    for p in [f.right, f.up, f.forward, f.origin] {
        out.extend(v3(p));
    }
    assert_canonical("from_context", &out);
}

// ---- encoder (AGPL feature) ----------------------------------------------------------------

#[cfg(feature = "encoder")]
mod encoder {
    use super::{assert_canonical, odd_nan, v3};
    use alice_kinematics::encoder::{IntentEncoder, SensorSample};
    use alice_kinematics::jerk::{
        fitts_law_duration, minimum_jerk_cost, FitResult, JerkFitter, MotionSample,
    };
    use alice_kinematics::Vec3k;

    fn fitter_with_nan() -> JerkFitter {
        let mut f = JerkFitter::new();
        for i in 0..3_u8 {
            let t = f32::from(i) * 0.01;
            let pos = if i == 2 {
                Vec3k::new(odd_nan(), 0.0, 0.0)
            } else {
                Vec3k::new(t, 0.0, 0.0)
            };
            f.push(MotionSample { pos, time: t });
        }
        f
    }

    #[test]
    fn boundary_jerkfitter_estimate_velocity() {
        assert_canonical(
            "estimate_velocity",
            &v3(fitter_with_nan().estimate_velocity()),
        );
    }

    #[test]
    fn boundary_jerkfitter_estimate_acceleration() {
        assert_canonical(
            "estimate_acceleration",
            &v3(fitter_with_nan().estimate_acceleration()),
        );
    }

    #[test]
    fn boundary_minimum_jerk_cost() {
        assert_canonical("minimum_jerk_cost", &[minimum_jerk_cost(odd_nan(), 1.0)]);
    }

    #[test]
    fn boundary_fitts_law_duration() {
        assert_canonical(
            "fitts_law_duration",
            &[fitts_law_duration(1.0, 0.1, odd_nan(), 0.3)],
        );
    }

    /// eight samples of a reach along x, the first and the last one's x a NaN: the fit's
    /// displacement and target are NaN, and its `start` is the first sample's position as stored
    /// (the payload NaN reaches the output unless the fit canonicalises it)
    fn reach_samples() -> Vec<MotionSample> {
        (0..8_u8)
            .map(|i| {
                let t = f32::from(i) * 0.02;
                let x = if i == 0 || i == 7 { odd_nan() } else { t * 0.5 };
                MotionSample {
                    pos: Vec3k::new(x, 0.01, 0.02),
                    time: t,
                }
            })
            .collect()
    }

    fn sensor(s: MotionSample) -> SensorSample {
        SensorSample {
            pos: s.pos,
            vel: Vec3k::ZERO,
            timestamp: s.time,
        }
    }

    fn fit_values(f: &FitResult) -> Vec<f32> {
        let mut v = Vec::new();
        for p in [f.target, f.start, f.start_velocity] {
            v.extend(v3(p));
        }
        v.extend([f.duration, f.fit_error, f.speed]);
        v
    }

    fn encoder_with_nan() -> IntentEncoder {
        let mut enc = IntentEncoder::new();
        for s in reach_samples() {
            let _ = enc.push_sample(sensor(s));
        }
        enc
    }

    #[test]
    fn boundary_jerkfitter_fit_trajectory() {
        let mut f = JerkFitter::new();
        for s in reach_samples() {
            f.push(s);
        }
        let fit = f.fit_trajectory().expect("eight samples over 0.14 s");
        assert_canonical("fit_trajectory", &fit_values(&fit));
    }

    #[test]
    fn boundary_intentencoder_push_sample() {
        // the eighth sample reaches the tracking minimum and extracts the intent
        let mut enc = IntentEncoder::new();
        let mut out = None;
        for s in reach_samples() {
            if let Some(intent) = enc.push_sample(sensor(s)) {
                out = Some(intent);
            }
        }
        let intent = out.expect("an intent is extracted");
        assert_canonical("push_sample", &v3(intent.target));
    }

    #[test]
    fn boundary_intentencoder_force_extract() {
        // a motion threshold no sample reaches: the encoder stays idle and keeps every
        // sample, and the intent is extracted on demand
        let mut enc = IntentEncoder::with_thresholds(1.0e9, 0.02, 0.0);
        for s in reach_samples() {
            assert!(enc.push_sample(sensor(s)).is_none(), "idle until forced");
        }
        let intent = enc.force_extract().expect("eight samples, the last a NaN");
        assert_canonical("force_extract", &v3(intent.target));
    }

    #[test]
    fn boundary_intentencoder_last_intent() {
        let intent = encoder_with_nan().last_intent().expect("extracted");
        assert_canonical("last_intent", &v3(intent.target));
    }

    #[test]
    fn boundary_intentencoder_last_fit() {
        let fit = encoder_with_nan().last_fit().expect("extracted");
        assert_canonical("last_fit", &fit_values(&fit));
    }
}

// ---- LOL bridge ----------------------------------------------------------------------------

#[cfg(feature = "lol")]
mod lol {
    use super::{assert_canonical, odd_nan, v3};
    use alice_kinematics::lol_bridge::intent_to_kinematics;
    use alice_lol::intent::IntentNode;
    use alice_lol::Vec3;

    #[test]
    fn boundary_intent_to_kinematics() {
        let walk = IntentNode::Walk {
            destination: Vec3::new(odd_nan(), 0.0, 1.0),
            speed: 1.0,
        };
        let intents = intent_to_kinematics(&walk, &[]).expect("walk maps to a reach");
        let values: Vec<f32> = intents.iter().flat_map(|i| v3(i.target)).collect();
        assert_canonical("intent_to_kinematics", &values);
    }
}
