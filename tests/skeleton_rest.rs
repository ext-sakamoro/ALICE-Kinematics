//! A skeleton built from measured world positions, and its rest data (end points, bone frames).
//!
//! oracle: the expected parents are the humanoid hierarchy written out here by hand (not read
//! from the crate's tables); the expected frames are worked by hand from the rule in
//! `src/rest.rs` (depth reference front, or up for an axis within 45 degrees of the front; width
//! = depth x axis) on axes whose components are exact in f32 (0, +-1, 3-4-5 triangles).
#![allow(clippy::float_cmp, clippy::many_single_char_names)]

use alice_kinematics::rest::{BoneFrame, EndPoint, RestError, RestSkeleton};
use alice_kinematics::skeleton::{BoneId, Skeleton, SkeletonBuildError};
use alice_kinematics::Vec3k;

const fn v(x: f32, y: f32, z: f32) -> Vec3k {
    Vec3k::new(x, y, z)
}

/// A standing figure in metres on the VRM 1.0 axes (+Y up, facing +Z, left +X), arms out: every
/// left joint has a right twin at -x. Numbers chosen to be exact in f32.
fn figure() -> Vec<(BoneId, Vec3k)> {
    vec![
        (BoneId::Hips, v(0.0, 0.75, 0.0)),
        (BoneId::Spine, v(0.0, 0.875, 0.0)),
        (BoneId::Chest, v(0.0, 1.0, 0.0)),
        (BoneId::Neck, v(0.0, 1.125, 0.0)),
        (BoneId::Head, v(0.0, 1.25, 0.0)),
        (BoneId::LeftShoulder, v(0.0625, 1.0625, 0.0)),
        (BoneId::LeftUpperArm, v(0.125, 1.0625, 0.0)),
        (BoneId::LeftLowerArm, v(0.375, 1.0625, 0.0)),
        (BoneId::LeftHand, v(0.625, 1.0625, 0.0)),
        (BoneId::RightShoulder, v(-0.0625, 1.0625, 0.0)),
        (BoneId::RightUpperArm, v(-0.125, 1.0625, 0.0)),
        (BoneId::RightLowerArm, v(-0.375, 1.0625, 0.0)),
        (BoneId::RightHand, v(-0.625, 1.0625, 0.0)),
        (BoneId::LeftUpperLeg, v(0.0625, 0.6875, 0.0)),
        (BoneId::LeftLowerLeg, v(0.0625, 0.375, 0.0)),
        (BoneId::LeftFoot, v(0.0625, 0.125, 0.0)),
        // ankle to ball: (0, -0.06, 0.08), a 3-4-5 triangle scaled by 0.02
        (BoneId::LeftToe, v(0.0625, 0.065, 0.08)),
        (BoneId::RightUpperLeg, v(-0.0625, 0.6875, 0.0)),
        (BoneId::RightLowerLeg, v(-0.0625, 0.375, 0.0)),
        (BoneId::RightFoot, v(-0.0625, 0.125, 0.0)),
        (BoneId::RightToe, v(-0.0625, 0.065, 0.08)),
    ]
}

fn ends() -> Vec<(EndPoint, Vec3k)> {
    vec![
        (EndPoint::HeadTop, v(0.0, 1.5, 0.0)),
        (EndPoint::LeftFingertip, v(0.75, 1.0625, 0.0)),
        (EndPoint::RightFingertip, v(-0.75, 1.0625, 0.0)),
        (EndPoint::LeftToeTip, v(0.0625, 0.065, 0.125)),
        (EndPoint::RightToeTip, v(-0.0625, 0.065, 0.125)),
        (EndPoint::LeftHeel, v(0.0625, 0.125, -0.0625)),
        (EndPoint::RightHeel, v(-0.0625, 0.125, -0.0625)),
    ]
}

fn parent_of(s: &Skeleton, id: BoneId) -> Option<BoneId> {
    let j = s.find_joint(id).expect("joint present");
    j.parent.map(|p| s.joints[p].id)
}

fn close(a: Vec3k, b: Vec3k) {
    let d = (a - b).length();
    assert!(d <= 1e-6, "{a:?} vs {b:?} ({d})");
}

#[test]
fn world_positions_are_stored_exactly_with_the_hierarchy() {
    let pts = figure();
    let s = Skeleton::from_world_positions(&pts, 1.5).expect("valid");
    assert_eq!(s.joint_count(), 21);
    assert_eq!(s.height, 1.5);
    for &(id, p) in &pts {
        let j = s.find_joint(id).expect("present");
        assert_eq!(j.world_position, p, "{id:?}");
        assert_eq!(j.local_rotation, Vec3k::ZERO);
    }
    let want = [
        (BoneId::Hips, None),
        (BoneId::Spine, Some(BoneId::Hips)),
        (BoneId::Chest, Some(BoneId::Spine)),
        (BoneId::Neck, Some(BoneId::Chest)),
        (BoneId::Head, Some(BoneId::Neck)),
        (BoneId::LeftShoulder, Some(BoneId::Chest)),
        (BoneId::LeftUpperArm, Some(BoneId::LeftShoulder)),
        (BoneId::LeftLowerArm, Some(BoneId::LeftUpperArm)),
        (BoneId::LeftHand, Some(BoneId::LeftLowerArm)),
        (BoneId::RightShoulder, Some(BoneId::Chest)),
        (BoneId::RightHand, Some(BoneId::RightLowerArm)),
        (BoneId::LeftUpperLeg, Some(BoneId::Hips)),
        (BoneId::LeftFoot, Some(BoneId::LeftLowerLeg)),
        (BoneId::LeftToe, Some(BoneId::LeftFoot)),
        (BoneId::RightUpperLeg, Some(BoneId::Hips)),
    ];
    for (id, p) in want {
        assert_eq!(parent_of(&s, id), p, "{id:?}");
    }
    // local = world - parent world, bone length = its length (0.25 upper arm, 0.1 ankle to ball)
    let ua = s.find_joint(BoneId::LeftLowerArm).expect("elbow");
    assert_eq!(ua.local_position, v(0.25, 0.0, 0.0));
    assert_eq!(ua.bone_length, 0.25);
    let toe = s.find_joint(BoneId::LeftToe).expect("toe");
    assert!((toe.bone_length - 0.1).abs() <= 1e-7, "{}", toe.bone_length);
    assert_eq!(s.find_joint(BoneId::Hips).expect("root").bone_length, 0.0);
    // the crate's FK reproduces the given positions
    let mut f = s.clone();
    f.update_world_positions();
    for &(id, p) in &pts {
        close(f.joint_world_position(id).expect("present"), p);
    }
}

#[test]
fn input_order_does_not_change_the_result() {
    let mut pts = figure();
    let a = Skeleton::from_world_positions(&pts, 1.5).expect("valid");
    pts.reverse();
    let b = Skeleton::from_world_positions(&pts, 1.5).expect("valid");
    let ids = |s: &Skeleton| {
        s.joints
            .iter()
            .map(|j| (j.id, j.parent))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&a), ids(&b));
    let first: Vec<BoneId> = a.joints.iter().map(|j| j.id).collect();
    assert_eq!(first[0], BoneId::Hips);
    for (i, j) in a.joints.iter().enumerate() {
        if let Some(p) = j.parent {
            assert!(p < i, "parent of {:?} after it", j.id);
        }
    }
}

#[test]
fn invalid_input_is_refused() {
    let base = figure();
    let without = |id: BoneId| {
        base.iter()
            .copied()
            .filter(|&(b, _)| b != id)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        Skeleton::from_world_positions(&without(BoneId::Hips), 1.5).unwrap_err(),
        SkeletonBuildError::MissingRoot
    );
    assert_eq!(
        Skeleton::from_world_positions(&without(BoneId::LeftUpperArm), 1.5).unwrap_err(),
        SkeletonBuildError::MissingParent {
            bone: BoneId::LeftLowerArm,
            parent: BoneId::LeftUpperArm
        }
    );
    let mut dup = base.clone();
    dup.push((BoneId::Neck, v(0.0, 1.1, 0.0)));
    assert_eq!(
        Skeleton::from_world_positions(&dup, 1.5).unwrap_err(),
        SkeletonBuildError::Duplicate(BoneId::Neck)
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut p = base.clone();
        p[7].1 = v(0.375, bad, 0.0);
        assert_eq!(
            Skeleton::from_world_positions(&p, 1.5).unwrap_err(),
            SkeletonBuildError::NonFinite(BoneId::LeftLowerArm)
        );
    }
    for h in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Skeleton::from_world_positions(&base, h).unwrap_err(),
            SkeletonBuildError::InvalidHeight
        );
    }
    // a subset closed under the hierarchy is a skeleton
    let trunk = [
        (BoneId::Hips, v(0.0, 0.75, 0.0)),
        (BoneId::Spine, v(0.0, 0.875, 0.0)),
    ];
    assert_eq!(
        Skeleton::from_world_positions(&trunk, 1.5)
            .expect("subset")
            .joint_count(),
        2
    );
}

#[test]
fn upper_chest_takes_the_neck_and_shoulders() {
    let mut pts = figure();
    pts.push((BoneId::UpperChest, v(0.0, 1.0625, 0.0)));
    let s = Skeleton::from_world_positions(&pts, 1.5).expect("valid");
    assert_eq!(s.joint_count(), 22);
    assert_eq!(parent_of(&s, BoneId::UpperChest), Some(BoneId::Chest));
    for id in [BoneId::Neck, BoneId::LeftShoulder, BoneId::RightShoulder] {
        assert_eq!(parent_of(&s, id), Some(BoneId::UpperChest), "{id:?}");
    }
    let neck = s.find_joint(BoneId::Neck).expect("neck");
    assert_eq!(neck.local_position, v(0.0, 0.0625, 0.0));
    assert_eq!(BoneId::Neck.parent_in(true), Some(BoneId::UpperChest));
    assert_eq!(BoneId::Neck.parent_in(false), Some(BoneId::Chest));
    assert_eq!(BoneId::UpperChest.parent_in(false), Some(BoneId::Chest));
    // the placeholder is unchanged: no upper chest
    assert!(Skeleton::default_humanoid()
        .find_joint(BoneId::UpperChest)
        .is_none());
}

#[test]
fn both_ik_targets_have_a_slot() {
    let mut pts = figure();
    pts.push((BoneId::LeftFootIk, v(0.0625, 0.0, 0.0)));
    pts.push((BoneId::RightFootIk, v(-0.0625, 0.0, 0.0)));
    let s = Skeleton::from_world_positions(&pts, 1.5).expect("valid");
    assert_eq!(parent_of(&s, BoneId::LeftFootIk), Some(BoneId::Hips));
    assert_eq!(parent_of(&s, BoneId::RightFootIk), Some(BoneId::Hips));
    assert_eq!(BoneId::LeftFootIk.mirror(), BoneId::RightFootIk);
    assert_eq!(BoneId::RightFootIk.mirror(), BoneId::LeftFootIk);
    // every id: the mirror of the mirror is itself, left and right swap, midline stays, and the
    // parent of the mirror is the mirror of the parent
    for id in BoneId::ALL {
        let m = id.mirror();
        assert_eq!(m.mirror(), id);
        let name = format!("{id:?}");
        let mname = format!("{m:?}");
        if let Some(rest) = name.strip_prefix("Left") {
            assert_eq!(mname, format!("Right{rest}"));
        } else if let Some(rest) = name.strip_prefix("Right") {
            assert_eq!(mname, format!("Left{rest}"));
        } else {
            assert_eq!(m, id);
        }
        for uc in [false, true] {
            assert_eq!(
                id.parent_in(uc).map(BoneId::mirror),
                m.parent_in(uc),
                "{id:?}"
            );
        }
    }
    // ALL holds every id once, parents first
    for (i, id) in BoneId::ALL.iter().enumerate() {
        assert_eq!(BoneId::ALL.iter().filter(|b| *b == id).count(), 1);
        if let Some(p) = id.parent_in(true) {
            assert!(BoneId::ALL.iter().position(|b| *b == p).expect("listed") < i);
        }
    }
    assert_eq!(BoneId::ALL.len(), 24);
}

#[test]
fn end_points_are_carried_by_their_joints() {
    let s = Skeleton::from_world_positions(&figure(), 1.5).expect("valid");
    let want = [
        (EndPoint::HeadTop, BoneId::Head),
        (EndPoint::LeftFingertip, BoneId::LeftHand),
        (EndPoint::RightFingertip, BoneId::RightHand),
        (EndPoint::LeftToeTip, BoneId::LeftToe),
        (EndPoint::RightToeTip, BoneId::RightToe),
        (EndPoint::LeftHeel, BoneId::LeftFoot),
        (EndPoint::RightHeel, BoneId::RightFoot),
    ];
    for (e, b) in want {
        assert_eq!(e.bone(), b);
    }
    let mut r = RestSkeleton::new(s, &ends()).expect("valid");
    for (e, p) in ends() {
        assert_eq!(r.end_point(e), Some(p), "{e:?}");
    }
    // moving the hand (a pose by translation) carries the fingertip with it
    let k = r
        .skeleton()
        .joints
        .iter()
        .position(|j| j.id == BoneId::LeftHand)
        .expect("hand");
    r.skeleton_mut().joints[k].local_position = v(0.25, 0.125, 0.0);
    r.skeleton_mut().update_world_positions();
    close(
        r.end_point(EndPoint::LeftFingertip).expect("tip"),
        v(0.75, 1.1875, 0.0),
    );
    // refusals
    let s = Skeleton::from_world_positions(&figure(), 1.5).expect("valid");
    let mut dup = ends();
    dup.push((EndPoint::HeadTop, v(0.0, 1.4, 0.0)));
    assert_eq!(
        RestSkeleton::new(s.clone(), &dup).unwrap_err(),
        RestError::DuplicateEndPoint(EndPoint::HeadTop)
    );
    let nan = [(EndPoint::HeadTop, v(0.0, f32::NAN, 0.0))];
    assert_eq!(
        RestSkeleton::new(s, &nan).unwrap_err(),
        RestError::NonFinite
    );
    let trunk =
        Skeleton::from_world_positions(&[(BoneId::Hips, v(0.0, 0.75, 0.0))], 1.5).expect("root");
    assert_eq!(
        RestSkeleton::new(trunk, &ends()[..1]).unwrap_err(),
        RestError::MissingBone(BoneId::Head)
    );
}

#[test]
fn frames_follow_the_rule() {
    let r = RestSkeleton::new(
        Skeleton::from_world_positions(&figure(), 1.5).expect("valid"),
        &ends(),
    )
    .expect("valid");
    let f = |b| r.frame(b).expect("frame");
    // thigh, axis down: depth front, width = Z x (-Y) = +X
    let t = f(BoneId::LeftLowerLeg);
    assert_eq!(
        (t.axis(), t.width(), t.depth()),
        (v(0.0, -1.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0))
    );
    // spine, axis up: width = Z x Y = -X
    let s = f(BoneId::Chest);
    assert_eq!(
        (s.axis(), s.width(), s.depth()),
        (v(0.0, 1.0, 0.0), v(-1.0, 0.0, 0.0), v(0.0, 0.0, 1.0))
    );
    // arms: left axis +X, width = Z x X = +Y; right axis -X, width -Y; both depth front
    let l = f(BoneId::LeftLowerArm);
    assert_eq!(
        (l.axis(), l.width(), l.depth()),
        (v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0))
    );
    let rr = f(BoneId::RightLowerArm);
    assert_eq!(
        (rr.axis(), rr.width(), rr.depth()),
        (v(-1.0, 0.0, 0.0), v(0.0, -1.0, 0.0), v(0.0, 0.0, 1.0))
    );
    // foot: axis (0, -0.6, 0.8), |a.z| = 0.8 > 1/sqrt 2 so the reference is up:
    // depth = (0, 0.64, 0.48) / 0.8 = (0, 0.8, 0.6), width = depth x axis = (1, 0, 0)
    let ft = f(BoneId::LeftToe);
    close(ft.axis(), v(0.0, -0.6, 0.8));
    close(ft.depth(), v(0.0, 0.8, 0.6));
    close(ft.width(), v(1.0, 0.0, 0.0));
    // the root has no segment; end segments have frames: head top axis up, heel axis back
    assert!(r.frame(BoneId::Hips).is_none());
    let h = r.end_frame(EndPoint::HeadTop).expect("head");
    assert_eq!(h.axis(), v(0.0, 1.0, 0.0));
    let heel = r.end_frame(EndPoint::LeftHeel).expect("heel");
    assert_eq!(
        (heel.axis(), heel.depth()),
        (v(0.0, 0.0, -1.0), v(0.0, 1.0, 0.0))
    );
    // every frame is orthonormal and right-handed
    for id in BoneId::ALL {
        if let Some(fr) = r.frame(id) {
            for (a, b) in [
                (fr.axis(), fr.width()),
                (fr.axis(), fr.depth()),
                (fr.width(), fr.depth()),
            ] {
                assert!(a.dot(b).abs() <= 1e-6, "{id:?}");
            }
            for u in [fr.axis(), fr.width(), fr.depth()] {
                // length_sq: Vec3k::length is an approximate root
                assert!((u.length_sq() - 1.0).abs() <= 2e-6, "{id:?}");
            }
            close(fr.axis().cross(fr.width()), fr.depth());
        }
    }
}

#[test]
fn rest_frames_are_apart_from_the_pose() {
    let mut r = RestSkeleton::new(
        Skeleton::from_world_positions(&figure(), 1.5).expect("valid"),
        &ends(),
    )
    .expect("valid");
    let before: Vec<_> = BoneId::ALL.iter().map(|&b| r.frame(b)).collect();
    r.skeleton_mut()
        .set_rotation(BoneId::LeftLowerArm, v(0.5, 0.25, 0.125));
    r.skeleton_mut()
        .set_rotation(BoneId::Spine, v(-0.5, 0.0, 0.25));
    let after: Vec<_> = BoneId::ALL.iter().map(|&b| r.frame(b)).collect();
    assert_eq!(before, after);
    assert_eq!(
        r.skeleton()
            .find_joint(BoneId::LeftLowerArm)
            .expect("elbow")
            .local_rotation,
        v(0.5, 0.25, 0.125)
    );
}

#[test]
fn a_frame_can_be_rolled() {
    let mut r = RestSkeleton::new(
        Skeleton::from_world_positions(&figure(), 1.5).expect("valid"),
        &ends(),
    )
    .expect("valid");
    // roll the left forearm a quarter turn: depth up, width = Y x X = -Z
    r.set_frame(BoneId::LeftLowerArm, v(0.0, 1.0, 0.0))
        .expect("valid");
    let l = r.frame(BoneId::LeftLowerArm).expect("frame");
    assert_eq!(
        (l.axis(), l.width(), l.depth()),
        (v(1.0, 0.0, 0.0), v(0.0, 0.0, -1.0), v(0.0, 1.0, 0.0))
    );
    // the other frames keep theirs
    assert_eq!(
        r.frame(BoneId::RightLowerArm).expect("frame").depth(),
        v(0.0, 0.0, 1.0)
    );
    assert_eq!(
        r.set_frame(BoneId::LeftLowerArm, v(2.0, 0.0, 0.0))
            .unwrap_err(),
        RestError::Degenerate
    );
    assert_eq!(
        r.set_frame(BoneId::Hips, v(0.0, 0.0, 1.0)).unwrap_err(),
        RestError::NoSegment(BoneId::Hips)
    );
    assert_eq!(
        r.set_frame(BoneId::UpperChest, v(0.0, 0.0, 1.0))
            .unwrap_err(),
        RestError::NoSegment(BoneId::UpperChest)
    );
    assert_eq!(
        r.set_frame(BoneId::LeftLowerArm, v(0.0, f32::NAN, 0.0))
            .unwrap_err(),
        RestError::NonFinite
    );
    assert_eq!(
        BoneFrame::new(Vec3k::ZERO, v(0.0, 0.0, 1.0)).unwrap_err(),
        RestError::Degenerate
    );
    assert_eq!(
        BoneFrame::along(v(0.0, f32::INFINITY, 0.0)).unwrap_err(),
        RestError::NonFinite
    );
}

#[test]
fn building_twice_gives_the_same_bits() {
    let a = RestSkeleton::new(
        Skeleton::from_world_positions(&figure(), 1.5).expect("valid"),
        &ends(),
    )
    .expect("valid");
    let b = RestSkeleton::new(
        Skeleton::from_world_positions(&figure(), 1.5).expect("valid"),
        &ends(),
    )
    .expect("valid");
    let bits = |r: &RestSkeleton| {
        let mut out = Vec::new();
        for j in &r.skeleton().joints {
            for x in [j.world_position, j.local_position] {
                out.extend([x.x.to_bits(), x.y.to_bits(), x.z.to_bits()]);
            }
            out.push(j.bone_length.to_bits());
        }
        for id in BoneId::ALL {
            if let Some(f) = r.frame(id) {
                for x in [f.axis(), f.width(), f.depth()] {
                    out.extend([x.x.to_bits(), x.y.to_bits(), x.z.to_bits()]);
                }
            }
        }
        out
    };
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn a_non_finite_z_alone_is_refused() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut p = figure();
        p[7].1 = v(0.375, 1.0625, bad);
        assert_eq!(
            Skeleton::from_world_positions(&p, 1.5).unwrap_err(),
            SkeletonBuildError::NonFinite(BoneId::LeftLowerArm)
        );
        let e = [(EndPoint::HeadTop, v(0.0, 1.5, bad))];
        let s = Skeleton::from_world_positions(&figure(), 1.5).expect("valid");
        assert_eq!(RestSkeleton::new(s, &e).unwrap_err(), RestError::NonFinite);
        assert_eq!(
            BoneFrame::along(v(0.0, 1.0, bad)).unwrap_err(),
            RestError::NonFinite
        );
        assert_eq!(
            BoneFrame::new(v(1.0, 0.0, 0.0), v(0.0, 0.0, bad)).unwrap_err(),
            RestError::NonFinite
        );
    }
}

#[test]
fn a_nearly_parallel_depth_hint_is_refused() {
    // the hint's part across the axis is 1e-4 of its length, under the 1e-3 floor
    assert_eq!(
        BoneFrame::new(v(1.0, 0.0, 0.0), v(1.0, 1e-4, 0.0)).unwrap_err(),
        RestError::Degenerate
    );
    // 1e-2 across is accepted, and the depth is the across part, normalised
    let f = BoneFrame::new(v(1.0, 0.0, 0.0), v(1.0, 1e-2, 0.0)).expect("across enough");
    assert_eq!(f.depth(), v(0.0, 1.0, 0.0));
}

#[test]
fn huge_finite_input_gives_unit_frames_or_is_refused() {
    // |along|^2 overflows f32, |along| does not: the frame is the exact unit one
    let f = BoneFrame::along(v(1e20, 0.0, 0.0)).expect("finite length");
    assert_eq!(
        (f.axis(), f.width(), f.depth()),
        (v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0))
    );
    let g = BoneFrame::new(v(0.0, -3e38, 0.0), v(0.0, 0.0, 3e38)).expect("finite lengths");
    assert_eq!(
        (g.axis(), g.width(), g.depth()),
        (v(0.0, -1.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0))
    );
    // an axis of 1e20 toward the front takes the up reference (the rule read on the unit axis)
    let h = BoneFrame::along(v(0.0, 0.0, 1e20)).expect("finite length");
    assert_eq!(
        (h.axis(), h.width(), h.depth()),
        (v(0.0, 0.0, 1.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0))
    );
    // finite components whose length is not (3e38 sqrt 2 > f32::MAX): refused, not a zero axis
    assert_eq!(
        BoneFrame::along(v(3e38, 3e38, 0.0)).unwrap_err(),
        RestError::Degenerate
    );
    assert_eq!(
        BoneFrame::new(v(3e38, -3e38, 0.0), v(0.0, 0.0, 1.0)).unwrap_err(),
        RestError::Degenerate
    );
    // a bone 1e20 long: its length is 1e20, not infinity
    let s = Skeleton::from_world_positions(
        &[
            (BoneId::Hips, v(0.0, 0.0, 0.0)),
            (BoneId::Spine, v(0.0, 1e20, 0.0)),
        ],
        1.5,
    )
    .expect("finite");
    assert_eq!(
        s.find_joint(BoneId::Spine).expect("spine").bone_length,
        1e20
    );
    // finite ends whose difference overflows: refused, not an infinite local position
    assert_eq!(
        Skeleton::from_world_positions(
            &[
                (BoneId::Hips, v(0.0, -3e38, 0.0)),
                (BoneId::Spine, v(0.0, 3e38, 0.0))
            ],
            1.5,
        )
        .unwrap_err(),
        SkeletonBuildError::NonFinite(BoneId::Spine)
    );
    // a difference that is finite but whose length is not (each part 3e38): refused
    assert_eq!(
        Skeleton::from_world_positions(
            &[
                (BoneId::Hips, v(0.0, 0.0, 0.0)),
                (BoneId::Spine, v(3e38, 3e38, 0.0))
            ],
            1.5,
        )
        .unwrap_err(),
        SkeletonBuildError::NonFinite(BoneId::Spine)
    );
    // an end point whose offset overflows: refused
    let s = Skeleton::from_world_positions(
        &[
            (BoneId::Hips, v(0.0, 0.0, 0.0)),
            (BoneId::Spine, v(0.0, 1.0, 0.0)),
            (BoneId::Chest, v(0.0, 2.0, 0.0)),
            (BoneId::Neck, v(0.0, 3.0, 0.0)),
            (BoneId::Head, v(0.0, -3e38, 0.0)),
        ],
        1.5,
    )
    .expect("finite");
    assert_eq!(
        RestSkeleton::new(s, &[(EndPoint::HeadTop, v(0.0, 3e38, 0.0))]).unwrap_err(),
        RestError::NonFinite
    );
}
