//! Named-bone skeletons (any number of bones, parent indices, head / tail segments) and the
//! 22-joint humanoid as a preset of them.
//!
//! Oracles:
//! - the humanoid preset skins bit-equal to `distance_based_skinning` on `Skeleton` as it was
//!   before named bones existed (SHA-256 of the output bits, recorded on that build);
//! - forward kinematics of an arbitrary 60-bone chain equals its closed form;
//! - skinning on named bones weights each bone by its own head → tail segment, by the
//!   closed-form inverse-square rule.

use alice_kinematics::autorig::{distance_based_skinning, InfluenceRadii, MeshView, Skinning};
use alice_kinematics::skeleton::Skeleton;
use alice_kinematics::Vec3k;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// A lattice of points around a 1.7 m humanoid: x in [-0.6, 0.6], y in [-1.0, 0.8],
/// z in [-0.2, 0.2], step 0.05 (exact multiples of a power-of-two-free step, built by integer
/// counts so every build makes the same floats)
fn lattice() -> Vec<Vec3k> {
    let mut v = Vec::new();
    for ix in -12..=12_i32 {
        for iy in -20..=16_i32 {
            for iz in -4..=4_i32 {
                #[allow(clippy::cast_precision_loss)]
                v.push(Vec3k::new(
                    ix as f32 * 0.05,
                    iy as f32 * 0.05,
                    iz as f32 * 0.05,
                ));
            }
        }
    }
    v
}

fn digest(s: &Skinning) -> String {
    let mut h = Sha256::new();
    for (idx, w) in s.bone_indices.iter().zip(&s.weights) {
        for i in idx {
            h.update(i.to_le_bytes());
        }
        for x in w {
            h.update(x.to_bits().to_le_bytes());
        }
    }
    h.finalize().iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Recorded on main 3250992 (before named bones): `distance_based_skinning` of the lattice on
/// the 1.70 m and 1.85 m humanoids with the default radii.
const GOLDEN_170: &str = "f3942ca2dcd874c5d7dca86b7ad74ccf7c4fddc8164da9c50be55deebcf429cb";
const GOLDEN_185: &str = "8eaf760f8a9df2f59966b303baff8b2a1b0cb65fe58c3821744468b956ca3d0a";

#[test]
fn humanoid_skinning_is_bit_equal_to_the_recorded_build() {
    let verts = lattice();
    let mesh = MeshView::from_vertices(&verts);
    let r = InfluenceRadii::default();
    let a = distance_based_skinning(&mesh, &Skeleton::with_height(1.70), &r);
    let b = distance_based_skinning(&mesh, &Skeleton::with_height(1.85), &r);
    assert_eq!(a.weights.len(), verts.len());
    println!("GOLDEN_170 {}", digest(&a));
    println!("GOLDEN_185 {}", digest(&b));
    assert_eq!(digest(&a), GOLDEN_170);
    assert_eq!(digest(&b), GOLDEN_185);
}

use alice_kinematics::autorig::distance_based_skinning_bones;
use alice_kinematics::bones::{BoneClass, BoneOffsets, BoneSkeleton, BoneSkeletonError, WorldBone};
use alice_kinematics::skeleton::{BoneId, BONE_COUNT};

fn off(name: &str, parent: Option<usize>, head: [f32; 3], tail: [f32; 3]) -> BoneOffsets {
    BoneOffsets {
        name: name.to_string(),
        parent,
        head_offset: Vec3k::new(head[0], head[1], head[2]),
        tail_offset: Vec3k::new(tail[0], tail[1], tail[2]),
        class: BoneClass::Limb,
    }
}

fn world(
    name: &str,
    parent: Option<usize>,
    head: [f32; 3],
    tail: [f32; 3],
    class: BoneClass,
) -> WorldBone {
    WorldBone {
        name: name.to_string(),
        parent,
        head: Vec3k::new(head[0], head[1], head[2]),
        tail: Vec3k::new(tail[0], tail[1], tail[2]),
        class,
    }
}

fn bits(v: Vec3k) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

/// oracle: closed form of a chain. Bone k (k = 0..59) hangs from bone k - 1's tail with a head
/// offset of (0, 0, 0.5) and a tail offset of (0.25 (k + 1), 0, 0): its world head is
/// x = 0.25 k (k + 1) / 2, z = 0.5 (k + 1), and its tail adds 0.25 (k + 1) to x. Every value is a
/// multiple of 0.125 below 2^10, exact in f32, so the comparison is bit for bit.
#[test]
fn a_60_bone_chain_places_every_bone_at_its_closed_form() {
    let bones: Vec<_> = (0..60_u16)
        .map(|k| {
            let kf = f32::from(k);
            off(
                &format!("chain_{k:02}"),
                k.checked_sub(1).map(usize::from),
                [0.0, 0.0, 0.5],
                [0.25 * (kf + 1.0), 0.0, 0.0],
            )
        })
        .collect();
    let s = BoneSkeleton::from_offsets(bones, 1.0).unwrap();
    assert_eq!(s.len(), 60);
    for k in 0..60_u16 {
        let kf = f32::from(k);
        let b = s.bone(usize::from(k)).unwrap();
        // 0.25 · k (k + 1) / 2, written as 0.125 · k (k + 1) (the same exact value)
        let head_x = 0.125 * kf * (kf + 1.0);
        let head_z = 0.5 * kf + 0.5;
        assert_eq!(
            bits(b.head()),
            bits(Vec3k::new(head_x, 0.0, head_z)),
            "head of bone {k}"
        );
        assert_eq!(
            bits(b.tail()),
            bits(Vec3k::new(head_x + 0.25 * (kf + 1.0), 0.0, head_z)),
            "tail of bone {k}"
        );
        assert_eq!(b.name(), format!("chain_{k:02}"));
        assert_eq!(s.index_of(&format!("chain_{k:02}")), Some(usize::from(k)));
    }
    // the last tail: x = 0.25 · 60 · 61 / 2 = 457.5
    assert_eq!(s.bone(59).unwrap().tail().x.to_bits(), 457.5_f32.to_bits());
}

/// oracle: branches. Two chains from one root, each placed from its own parent's tail.
#[test]
fn branches_hang_from_their_own_parent() {
    let s = BoneSkeleton::from_offsets(
        vec![
            off("root", None, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]),
            off("left", Some(0), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            off("right", Some(0), [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            off("left_tip", Some(1), [0.5, 0.0, 0.0], [0.0, -2.0, 0.0]),
        ],
        1.0,
    )
    .unwrap();
    assert_eq!(
        bits(s.bone(0).unwrap().tail()),
        bits(Vec3k::new(1.0, 3.0, 3.0))
    );
    assert_eq!(
        bits(s.bone(1).unwrap().tail()),
        bits(Vec3k::new(2.0, 3.0, 3.0))
    );
    assert_eq!(
        bits(s.bone(2).unwrap().tail()),
        bits(Vec3k::new(0.0, 3.0, 3.0))
    );
    assert_eq!(
        bits(s.bone(3).unwrap().head()),
        bits(Vec3k::new(2.5, 3.0, 3.0))
    );
    assert_eq!(
        bits(s.bone(3).unwrap().tail()),
        bits(Vec3k::new(2.5, 1.0, 3.0))
    );
}

/// oracle: the inverse-square rule written out. Two bones along x, A (0,0,0)→(1,0,0) and
/// B (1,0,0)→(2,0,0); the vertex (0.5, 0.01, 0) is 0.01 from A (d² = 1e-4) and its nearest point
/// on B is (1,0,0) (d² = 0.25 + 1e-4). w = 1 / (d² + 1e-6), normalised. Height 20 makes the
/// limb radius 0.05 · 20 = 1, so both bones are candidates.
#[test]
fn named_bones_skin_by_the_inverse_square_rule() {
    let s = BoneSkeleton::from_world(
        vec![
            world("A", None, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], BoneClass::Limb),
            world(
                "B",
                Some(0),
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                BoneClass::Limb,
            ),
        ],
        20.0,
    )
    .unwrap();
    let v = [Vec3k::new(0.5, 0.01, 0.0)];
    let k =
        distance_based_skinning_bones(&MeshView::from_vertices(&v), &s, &InfluenceRadii::default());
    let da = 0.01_f32 * 0.01;
    let db = 0.5_f32 * 0.5 + 0.01 * 0.01;
    let (ra, rb) = (1.0 / (da + 1e-6), 1.0 / (db + 1e-6));
    let sum = ra + rb + 0.0 + 0.0;
    assert_eq!(k.bone_indices[0], [0, 1, 0, 0]);
    assert!(
        (k.weights[0][0] - ra / sum).abs() <= 1e-6,
        "{:?}",
        k.weights[0]
    );
    assert!(
        (k.weights[0][1] - rb / sum).abs() <= 1e-6,
        "{:?}",
        k.weights[0]
    );
    assert_eq!(k.weights[0][2].to_bits(), 0.0_f32.to_bits());
    assert_eq!(k.weights[0][3].to_bits(), 0.0_f32.to_bits());
}

/// A bone owns its own head → tail segment: a support bone that runs beside the upper arm takes
/// the weight of a vertex next to it, though its parent (the shoulder) ends elsewhere. With the
/// humanoid's parent → child rule the vertex would go to the child of the nearest joint.
#[test]
fn a_support_bone_owns_its_own_segment() {
    let s = BoneSkeleton::from_world(
        vec![
            world(
                "Shoulder",
                None,
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                BoneClass::Torso,
            ),
            world(
                "Upper_arm",
                Some(0),
                [0.2, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                BoneClass::Limb,
            ),
            world(
                "Upper_arm_support",
                Some(0),
                [0.2, 0.1, 0.0],
                [0.5, 0.1, 0.0],
                BoneClass::Limb,
            ),
        ],
        1.0,
    )
    .unwrap();
    // 0.005 above the support bone's middle, 0.095 from the upper arm, far from the shoulder
    let v = [Vec3k::new(0.35, 0.105, 0.0)];
    let k =
        distance_based_skinning_bones(&MeshView::from_vertices(&v), &s, &InfluenceRadii::default());
    assert_eq!(
        k.bone_indices[0][0],
        u16::try_from(s.index_of("Upper_arm_support").unwrap()).unwrap()
    );
    assert!(k.weights[0][0] > 0.99, "{:?}", k.weights[0]);
}

#[test]
fn the_humanoid_preset_names_its_bones_after_the_joints() {
    let skel = Skeleton::with_height(1.70);
    let s = BoneSkeleton::humanoid(1.70);
    assert_eq!(s.len(), BONE_COUNT);
    assert_eq!(s, BoneSkeleton::from_skeleton(&skel));
    for (i, j) in skel.joints.iter().enumerate() {
        let b = s.bone(i).unwrap();
        assert_eq!(b.name(), j.id.name());
        assert_eq!(b.parent(), j.parent);
        assert_eq!(bits(b.tail()), bits(j.world_position));
        let head = j
            .parent
            .map_or(j.world_position, |p| skel.joints[p].world_position);
        assert_eq!(bits(b.head()), bits(head));
        assert_eq!(b.class() == BoneClass::Limb, j.id.is_limb());
    }
    assert_eq!(
        s.index_of("LeftUpperArm"),
        Some(BoneId::LeftUpperArm as usize)
    );
    // the preset through the named-bone entry gives the same bits as the Skeleton entry
    let verts = lattice();
    let mesh = MeshView::from_vertices(&verts);
    let r = InfluenceRadii::default();
    assert_eq!(
        digest(&distance_based_skinning_bones(&mesh, &s, &r)),
        GOLDEN_170
    );
}

#[test]
fn a_list_that_is_not_a_skeleton_is_refused() {
    let ok = |n: &str, p| world(n, p, [0.0; 3], [1.0, 0.0, 0.0], BoneClass::Torso);
    assert_eq!(
        BoneSkeleton::from_world(vec![], 1.0),
        Err(BoneSkeletonError::Empty)
    );
    assert_eq!(
        BoneSkeleton::from_world(vec![ok("a", None), ok("", Some(0))], 1.0),
        Err(BoneSkeletonError::EmptyName(1))
    );
    assert_eq!(
        BoneSkeleton::from_world(vec![ok("a", None), ok("a", Some(0))], 1.0),
        Err(BoneSkeletonError::DuplicateName(1, "a".to_string()))
    );
    // a parent must come first: itself and a later bone are refused
    assert_eq!(
        BoneSkeleton::from_world(vec![ok("a", Some(0))], 1.0),
        Err(BoneSkeletonError::BadParent { bone: 0, parent: 0 })
    );
    assert_eq!(
        BoneSkeleton::from_world(vec![ok("a", None), ok("b", Some(2)), ok("c", Some(0))], 1.0),
        Err(BoneSkeletonError::BadParent { bone: 1, parent: 2 })
    );
    assert_eq!(
        BoneSkeleton::from_offsets(vec![off("a", Some(1), [0.0; 3], [0.0; 3])], 1.0),
        Err(BoneSkeletonError::BadParent { bone: 0, parent: 1 })
    );
    let many: Vec<_> = (0..65_537_u32)
        .map(|i| world(&format!("b{i}"), None, [0.0; 3], [0.0; 3], BoneClass::Torso))
        .collect();
    assert_eq!(
        BoneSkeleton::from_world(many, 1.0),
        Err(BoneSkeletonError::TooManyBones(65_537))
    );
    // 65,536 bones (the last index 65,535 fits a u16) is a skeleton
    let max: Vec<_> = (0..65_536_u32)
        .map(|i| world(&format!("b{i}"), None, [0.0; 3], [0.0; 3], BoneClass::Torso))
        .collect();
    assert_eq!(BoneSkeleton::from_world(max, 1.0).unwrap().len(), 65_536);
}

/// A NaN coordinate is stored as the canonical NaN (ALICE-Law G4) and its bone takes no weight.
#[test]
fn a_nan_bone_is_canonical_and_takes_no_weight() {
    let nan = f32::from_bits(0x7fc0_0001);
    let s = BoneSkeleton::from_world(
        vec![
            world("A", None, [0.0; 3], [1.0, 0.0, 0.0], BoneClass::Limb),
            world(
                "B",
                Some(0),
                [nan, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                BoneClass::Limb,
            ),
        ],
        10.0,
    )
    .unwrap();
    assert_eq!(s.bone(1).unwrap().head().x.to_bits(), f32::NAN.to_bits());
    let v = [Vec3k::new(1.5, 0.0, 0.0)];
    let k =
        distance_based_skinning_bones(&MeshView::from_vertices(&v), &s, &InfluenceRadii::default());
    assert_eq!(k.bone_indices[0][0], 0);
    assert_eq!(
        k.weights[0].map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 0.0].map(f32::to_bits)
    );
}
