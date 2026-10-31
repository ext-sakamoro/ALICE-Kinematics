//! Intent on the Kinematics skeleton: the 8-byte frames, the Walk law and the capsule chain.
//!
//! oracle: the frames are the bytes the examples' programs produced before the robot moved onto
//! the skeleton (the body only reaches a packet through the program's first SDF, whose centroid
//! is the origin either way); the Walk positions are summed here by hand from the bone lengths
//! (Spine 0.18, Chest 0.20, Neck 0.12, Head 0.10, Shoulder 0.15, UpperArm 0.28, LowerArm 0.25,
//! Hand 0.08, UpperLeg 0.42, LowerLeg 0.40, Foot 0.08, Toe 0.05 at h = 1.70) in the order the
//! hierarchy adds them, from the destination, not read back from the crate's tables.
#![allow(clippy::float_cmp)]

use std::sync::Arc;

use alice_kinematics::skeleton::{BoneId, Skeleton};
use alice_kinematics::Vec3k;
use alice_lol::intent::{grasp, parallel, release, rest, sequence, walk, HandSide, ProgramBuilder};
use alice_lol::SdfNode;
use alice_lol_robot::{skeleton_capsules, HumanoidIntent, IntentPacketStream};
use glam::Vec3;

fn body_sdf() -> SdfNode {
    skeleton_capsules(&Skeleton::default_humanoid(), 0.05, 0.05).expect("bones")
}

fn cup(offset: Vec3, radius: f32) -> SdfNode {
    SdfNode::Translate {
        child: Arc::new(SdfNode::Sphere { radius }),
        offset,
    }
}

/// (name, the frames now, the frames from before)
type Case = (&'static str, Vec<[u8; 8]>, Vec<[u8; 8]>);

fn frames(program: &alice_lol::intent::Program) -> Vec<[u8; 8]> {
    IntentPacketStream::from_program(program).expect("packets")
}

#[test]
fn the_frames_are_the_bytes_from_before() {
    // robot_walk
    let walk_p = ProgramBuilder::new()
        .with_sdf(body_sdf())
        .with_intent(walk(Vec3::new(2.5, 0.0, 1.0), 1.0))
        .build();
    // robot_grasp
    let mut b = ProgramBuilder::new().with_sdf(body_sdf());
    let c = b.register(cup(Vec3::new(0.5, 1.0, 0.3), 0.05));
    let grasp_p = b.with_intent(grasp(c, HandSide::Right, 3.0)).build();
    // robot_dance
    let mut b = ProgramBuilder::new().with_sdf(body_sdf());
    let c = b.register(cup(Vec3::new(0.6, 1.0, 0.2), 0.04));
    let dance = sequence(vec![
        sequence(vec![
            walk(Vec3::new(0.5, 0.0, 0.0), 1.0),
            grasp(c, HandSide::Right, 3.0),
            walk(Vec3::new(2.5, 0.0, 0.5), 1.2),
            release(c),
            rest(500),
        ]),
        parallel(vec![walk(Vec3::new(0.0, 0.0, 0.0), 0.8), rest(1000)]),
    ]);
    let dance_p = b.with_intent(dance).build();
    let cases: [Case; 3] = [
        (
            "walk",
            frames(&walk_p),
            vec![[0x02, 0x80, 0x00, 0x00, 0x01, 0x00, 0xff, 0x00]],
        ),
        (
            "grasp",
            frames(&grasp_p),
            vec![[0x00, 0x80, 0x01, 0x00, 0x00, 0x4c, 0x99, 0x06]],
        ),
        (
            "dance",
            frames(&dance_p),
            vec![
                [0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00],
                [0x00, 0x99, 0x01, 0x00, 0x00, 0x33, 0x99, 0x06],
                [0x02, 0x80, 0x00, 0x00, 0x00, 0x80, 0xff, 0x00],
                [0x00, 0x99, 0x01, 0x00, 0x00, 0x33, 0x32, 0x03],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x00],
            ],
        ),
    ];
    let mut compared = 0;
    for (name, got, want) in cases {
        assert_eq!(got, want, "{name}");
        compared += got.len();
    }
    assert_eq!(compared, 7);
}

#[test]
fn walk_moves_every_joint_by_the_root_to_the_destination() {
    let s = Skeleton::with_height(1.70);
    let d = Vec3::new(2.5, -0.25, 1.0);
    let posed = s.apply_intent(&walk(d, 1.0)).expect("walk");
    let at = |id| posed.joint_world_position(id).expect("joint");
    let chest_y = d.y + 0.18 + 0.20;
    let ankle_y = d.y + -0.42 + -0.40;
    let expect = [
        (BoneId::Hips, Vec3k::new(d.x, d.y, d.z)),
        (BoneId::Head, Vec3k::new(d.x, chest_y + 0.12 + 0.10, d.z)),
        (
            BoneId::LeftHand,
            Vec3k::new(d.x + 0.15 + 0.28 + 0.25 + 0.08, chest_y, d.z),
        ),
        (
            BoneId::RightHand,
            Vec3k::new(d.x + -0.15 + -0.28 + -0.25 + -0.08, chest_y, d.z),
        ),
        (BoneId::LeftLowerLeg, Vec3k::new(d.x, ankle_y, d.z)),
        (
            BoneId::RightToe,
            Vec3k::new(d.x, ankle_y, d.z + 0.08 + 0.05),
        ),
    ];
    for (id, want) in expect {
        assert_eq!(at(id), want, "{id:?}");
    }
    // a rigid translation: every joint keeps its offset from the root (to the rounding of one add)
    let mut compared = 0;
    for (a, b) in s.joints.iter().zip(&posed.joints) {
        let (ra, rb) = (
            a.world_position,
            b.world_position - Vec3k::new(d.x, d.y, d.z),
        );
        assert!((ra - rb).length() < 1e-6, "{:?}", a.id);
        compared += 1;
    }
    assert_eq!(compared, 22);
}

/// The capsules of a chain, left to right, as (from, to, radius)
fn capsules(n: &SdfNode, out: &mut Vec<(Vec3, Vec3, f32)>) {
    match n {
        SdfNode::SmoothUnion { a, b, .. } => {
            capsules(a, out);
            capsules(b, out);
        }
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => out.push((*point_a, *point_b, *radius)),
        other => panic!("not a capsule chain: {other:?}"),
    }
}

#[test]
fn the_capsules_are_the_bones_from_parent_to_child() {
    let s = Skeleton::default_humanoid();
    let mut got = Vec::new();
    capsules(&skeleton_capsules(&s, 0.05, 0.05).expect("bones"), &mut got);
    let v = |p: Vec3k| Vec3::new(p.x, p.y, p.z);
    let want: Vec<(Vec3, Vec3, f32)> = s
        .joints
        .iter()
        .filter_map(|j| {
            Some((
                v(s.joints[j.parent?].world_position),
                v(j.world_position),
                0.05,
            ))
        })
        .filter(|(a, b, _)| a != b)
        .collect();
    // 22 joints: the root has no bone, and the foot IK target sits on the hips (length 0)
    assert_eq!(want.len(), 20);
    assert_eq!(got, want);
}

#[test]
fn a_skeleton_without_bones_has_no_sdf() {
    let mut s = Skeleton::default_humanoid();
    s.joints.retain(|j| j.id == BoneId::Hips);
    assert!(skeleton_capsules(&s, 0.05, 0.05).is_none());
}
