//! The placeholder pose of `Skeleton::with_height`: a T-pose on the VRM 1.0 axes (+Y up, the
//! character faces +Z, the character's left is +X), each joint at its parent plus its bone's
//! direction times its bone's length.
//!
//! oracle: the expected positions are summed here by hand from the bone lengths (Spine 0.18,
//! Chest 0.20, Neck 0.12, Head 0.10, `UpperLeg` 0.42, `LowerLeg` 0.40, scaled by h / 1.70) in the
//! order the hierarchy adds them, not read back from the crate's tables.
#![allow(clippy::float_cmp)]

use alice_kinematics::skeleton::{BoneId, Skeleton};
use alice_kinematics::Vec3k;

fn at(s: &Skeleton, id: BoneId) -> Vec3k {
    s.joint_world_position(id).expect("joint present")
}

const LEGS: [(BoneId, BoneId); 4] = [
    (BoneId::LeftUpperLeg, BoneId::RightUpperLeg),
    (BoneId::LeftLowerLeg, BoneId::RightLowerLeg),
    (BoneId::LeftFoot, BoneId::RightFoot),
    (BoneId::LeftToe, BoneId::RightToe),
];
const ARMS: [(BoneId, BoneId); 4] = [
    (BoneId::LeftShoulder, BoneId::RightShoulder),
    (BoneId::LeftUpperArm, BoneId::RightUpperArm),
    (BoneId::LeftLowerArm, BoneId::RightLowerArm),
    (BoneId::LeftHand, BoneId::RightHand),
];

#[test]
fn the_legs_hang_below_the_hips_and_the_feet_point_forward() {
    for h in [1.70_f32, 1.87] {
        let s = Skeleton::with_height(h);
        let hips = at(&s, BoneId::Hips);
        let mut compared = 0;
        for (l, r) in LEGS {
            for id in [l, r] {
                assert!(at(&s, id).y < hips.y, "{id:?} not below the hips at h {h}");
                compared += 1;
            }
            // the placeholder has no hip width: left and right legs coincide at x = 0
            assert_eq!(at(&s, l), at(&s, r), "{l:?} / {r:?} at h {h}");
            assert_eq!(at(&s, l).x, 0.0);
        }
        // the foot and the toe go forward (+Z) from the ankle, at the ankle's height
        let ankle = at(&s, BoneId::LeftLowerLeg);
        for id in [BoneId::LeftFoot, BoneId::LeftToe] {
            assert_eq!(at(&s, id).y, ankle.y, "{id:?}");
            assert!(at(&s, id).z > ankle.z, "{id:?}");
        }
        assert_eq!(compared, 8);
    }
}

#[test]
fn the_shoulders_and_arms_mirror_across_the_midline() {
    let s = Skeleton::default_humanoid();
    let chest = at(&s, BoneId::Chest);
    for (l, r) in ARMS {
        let (a, b) = (at(&s, l), at(&s, r));
        assert_eq!((a.x, a.y, a.z), (-b.x, b.y, b.z), "{l:?} / {r:?}");
        assert!(a.x > 0.0, "{l:?} is on the character's left (+X)");
        assert_eq!(a.y, chest.y, "{l:?}: the arms are horizontal at the chest");
    }
}

#[test]
fn the_head_is_highest_and_the_ankles_lowest() {
    let s = Skeleton::default_humanoid();
    let head = at(&s, BoneId::Head);
    let ankle = at(&s, BoneId::LeftLowerLeg);
    let mut compared = 0;
    for j in &s.joints {
        if j.id != BoneId::Head {
            assert!(
                j.world_position.y < head.y,
                "{:?} at or above the head",
                j.id
            );
        }
        assert!(j.world_position.y >= ankle.y, "{:?} below the ankles", j.id);
        compared += 1;
    }
    assert_eq!(compared, 22);
}

/// The joint span (hips-relative joint ends, not stature: there is no skull-top or sole joint)
/// is (0.60 + 0.82) · h / 1.70, summed in the hierarchy's order
#[test]
fn the_joint_span_is_the_bone_lengths_up_and_down() {
    for h in [1.70_f32, 1.87, 0.5] {
        let s = Skeleton::with_height(h);
        let k = h / 1.70;
        let head_y = 0.0 + 0.18 * k + 0.20 * k + 0.12 * k + 0.10 * k;
        let ankle_y = 0.0 + -(0.42 * k) + -(0.40 * k);
        assert_eq!(at(&s, BoneId::Head).y, head_y, "head at h {h}");
        assert_eq!(at(&s, BoneId::LeftLowerLeg).y, ankle_y, "ankle at h {h}");
        assert_eq!(
            at(&s, BoneId::Head).y - at(&s, BoneId::LeftLowerLeg).y,
            head_y - ankle_y
        );
    }
}
