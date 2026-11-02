//! Full-body skeleton — 22 joint hierarchical model
//!
//! 全身骨格の階層構造と FK 伝播。
//! VR/モーションキャプチャ向けの標準的な関節配置。
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use alloc::vec::Vec;

/// 骨格関節 ID。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BoneId {
    /// 腰 (ルート)。
    Hips = 0,
    /// 脊椎。
    Spine,
    /// 胸。
    Chest,
    /// 首。
    Neck,
    /// 頭。
    Head,
    /// 左肩。
    LeftShoulder,
    /// 左上腕。
    LeftUpperArm,
    /// 左前腕。
    LeftLowerArm,
    /// 左手首。
    LeftHand,
    /// 右肩。
    RightShoulder,
    /// 右上腕。
    RightUpperArm,
    /// 右前腕。
    RightLowerArm,
    /// 右手首。
    RightHand,
    /// 左大腿。
    LeftUpperLeg,
    /// 左下腿。
    LeftLowerLeg,
    /// 左足首。
    LeftFoot,
    /// 左つま先。
    LeftToe,
    /// 右大腿。
    RightUpperLeg,
    /// 右下腿。
    RightLowerLeg,
    /// 右足首。
    RightFoot,
    /// 右つま先。
    RightToe,
    /// 左足IKターゲット。
    LeftFootIk,
    /// 右足IKターゲット。
    RightFootIk,
    /// Upper chest, between `Chest` and the neck and shoulders. Not in the placeholder tables
    /// (`BONE_COUNT`, `with_height`): a skeleton has it when it is built with
    /// [`Skeleton::from_world_positions`].
    UpperChest,
}

/// 骨格関節の総数。
pub const BONE_COUNT: usize = 22;

/// 骨格関節データ。
#[derive(Debug, Clone)]
pub struct SkeletonJoint {
    /// 関節 ID。
    pub id: BoneId,
    /// 親関節インデックス (ルートは `None`)。
    pub parent: Option<usize>,
    /// ローカル位置 (親からのオフセット)。
    pub local_position: Vec3k,
    /// ローカル回転 (オイラー角、ラジアン)。
    pub local_rotation: Vec3k,
    /// ワールド位置 (FK計算後)。
    pub world_position: Vec3k,
    /// ボーン長 (メートル)。
    pub bone_length: f32,
}

impl SkeletonJoint {
    /// This joint with every NaN replaced by the canonical NaN (ALICE-Law G4, `src/nan.rs`)
    #[must_use]
    pub(crate) fn canonical(self) -> Self {
        use crate::nan::{canon, canon3};
        Self {
            local_position: canon3(self.local_position),
            local_rotation: canon3(self.local_rotation),
            world_position: canon3(self.world_position),
            bone_length: canon(self.bone_length),
            ..self
        }
    }
}

/// 全身骨格モデル。
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// 全関節。
    pub joints: Vec<SkeletonJoint>,
    /// 身長 (メートル)。
    pub height: f32,
}

impl Skeleton {
    /// This skeleton with every NaN replaced by the canonical NaN (ALICE-Law G4, `src/nan.rs`)
    #[must_use]
    pub(crate) fn canonical(self) -> Self {
        Self {
            joints: self
                .joints
                .into_iter()
                .map(SkeletonJoint::canonical)
                .collect(),
            height: crate::nan::canon(self.height),
        }
    }
}

/// デフォルトの親子関係テーブル。
const PARENT_TABLE: [Option<usize>; BONE_COUNT] = [
    None,     // Hips
    Some(0),  // Spine -> Hips
    Some(1),  // Chest -> Spine
    Some(2),  // Neck -> Chest
    Some(3),  // Head -> Neck
    Some(2),  // LeftShoulder -> Chest
    Some(5),  // LeftUpperArm -> LeftShoulder
    Some(6),  // LeftLowerArm -> LeftUpperArm
    Some(7),  // LeftHand -> LeftLowerArm
    Some(2),  // RightShoulder -> Chest
    Some(9),  // RightUpperArm -> RightShoulder
    Some(10), // RightLowerArm -> RightUpperArm
    Some(11), // RightHand -> RightLowerArm
    Some(0),  // LeftUpperLeg -> Hips
    Some(13), // LeftLowerLeg -> LeftUpperLeg
    Some(14), // LeftFoot -> LeftLowerLeg
    Some(15), // LeftToe -> LeftFoot
    Some(0),  // RightUpperLeg -> Hips
    Some(17), // RightLowerLeg -> RightUpperLeg
    Some(18), // RightFoot -> RightLowerLeg
    Some(19), // RightToe -> RightFoot
    Some(0),  // LeftFootIk -> Hips (virtual)
];

/// デフォルトのボーン長テーブル (身長1.7mの標準体型)。
const BONE_LENGTHS: [f32; BONE_COUNT] = [
    0.00, // Hips (root)
    0.18, // Spine
    0.20, // Chest
    0.12, // Neck
    0.10, // Head
    0.15, // LeftShoulder
    0.28, // LeftUpperArm
    0.25, // LeftLowerArm
    0.08, // LeftHand
    0.15, // RightShoulder
    0.28, // RightUpperArm
    0.25, // RightLowerArm
    0.08, // RightHand
    0.42, // LeftUpperLeg
    0.40, // LeftLowerLeg
    0.08, // LeftFoot
    0.05, // LeftToe
    0.42, // RightUpperLeg
    0.40, // RightLowerLeg
    0.08, // RightFoot
    0.05, // RightToe
    0.00, // LeftFootIk (virtual)
];

/// Each bone's direction in the placeholder T-pose, on the VRM 1.0 axes: +Y up, the character
/// faces +Z, the character's left is +X. The spine, neck and head go up, the shoulders and arms
/// go out horizontally (left +X, right -X), the legs go down and the feet and toes forward.
/// Directions only: the lengths are [`BONE_LENGTHS`]. The zero-length bones (the root and the
/// virtual IK target) take +Y and stay at their parent.
const BONE_DIRECTIONS: [[f32; 3]; BONE_COUNT] = [
    [0.0, 1.0, 0.0],  // Hips (root)
    [0.0, 1.0, 0.0],  // Spine
    [0.0, 1.0, 0.0],  // Chest
    [0.0, 1.0, 0.0],  // Neck
    [0.0, 1.0, 0.0],  // Head
    [1.0, 0.0, 0.0],  // LeftShoulder
    [1.0, 0.0, 0.0],  // LeftUpperArm
    [1.0, 0.0, 0.0],  // LeftLowerArm
    [1.0, 0.0, 0.0],  // LeftHand
    [-1.0, 0.0, 0.0], // RightShoulder
    [-1.0, 0.0, 0.0], // RightUpperArm
    [-1.0, 0.0, 0.0], // RightLowerArm
    [-1.0, 0.0, 0.0], // RightHand
    [0.0, -1.0, 0.0], // LeftUpperLeg
    [0.0, -1.0, 0.0], // LeftLowerLeg
    [0.0, 0.0, 1.0],  // LeftFoot
    [0.0, 0.0, 1.0],  // LeftToe
    [0.0, -1.0, 0.0], // RightUpperLeg
    [0.0, -1.0, 0.0], // RightLowerLeg
    [0.0, 0.0, 1.0],  // RightFoot
    [0.0, 0.0, 1.0],  // RightToe
    [0.0, 1.0, 0.0],  // LeftFootIk (virtual)
];

impl Skeleton {
    /// 標準体型の骨格を生成 (身長 1.7m)。
    #[must_use]
    pub fn default_humanoid() -> Self {
        Self::with_height(1.70)
    }

    /// 指定身長の骨格を生成。
    ///
    /// The placeholder pose is a T-pose on the VRM 1.0 axes (+Y up, facing +Z, the character's
    /// left is +X): each joint sits at its parent plus its bone's direction
    /// (`BONE_DIRECTIONS`) times its bone's length (`BONE_LENGTHS × height / 1.70`), with the
    /// hips at the origin. The joint span (hips-relative joint ends), not stature: there is no
    /// skull-top or sole joint, so the head joint is at (0.18 + 0.20 + 0.12 + 0.10) · h / 1.70
    /// above the hips and the ankles (0.42 + 0.40) · h / 1.70 below. The placeholder has no hip
    /// width: both legs hang from the hips at x = 0 and coincide.
    #[must_use]
    pub fn with_height(height: f32) -> Self {
        let scale = height / 1.70;

        let all_bones: [BoneId; BONE_COUNT] = [
            BoneId::Hips,
            BoneId::Spine,
            BoneId::Chest,
            BoneId::Neck,
            BoneId::Head,
            BoneId::LeftShoulder,
            BoneId::LeftUpperArm,
            BoneId::LeftLowerArm,
            BoneId::LeftHand,
            BoneId::RightShoulder,
            BoneId::RightUpperArm,
            BoneId::RightLowerArm,
            BoneId::RightHand,
            BoneId::LeftUpperLeg,
            BoneId::LeftLowerLeg,
            BoneId::LeftFoot,
            BoneId::LeftToe,
            BoneId::RightUpperLeg,
            BoneId::RightLowerLeg,
            BoneId::RightFoot,
            BoneId::RightToe,
            BoneId::LeftFootIk,
        ];

        let joints = all_bones
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let bone_length = BONE_LENGTHS[i] * scale;
                let [dx, dy, dz] = BONE_DIRECTIONS[i];
                SkeletonJoint {
                    id,
                    parent: PARENT_TABLE[i],
                    local_position: Vec3k::new(
                        dx * bone_length,
                        dy * bone_length,
                        dz * bone_length,
                    ),
                    local_rotation: Vec3k::ZERO,
                    world_position: Vec3k::ZERO,
                    bone_length,
                }
            })
            .collect();

        let mut skel = Self { joints, height };
        skel.update_world_positions();
        skel.canonical()
    }

    /// FK: ローカル位置からワールド位置を計算。
    pub fn update_world_positions(&mut self) {
        for i in 0..self.joints.len() {
            let world_pos = if let Some(parent_idx) = self.joints[i].parent {
                let parent_world = self.joints[parent_idx].world_position;
                parent_world + self.joints[i].local_position
            } else {
                self.joints[i].local_position
            };
            self.joints[i].world_position = world_pos;
        }
    }

    /// 関節数。
    #[must_use]
    pub const fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// 関節を ID で検索。
    #[must_use]
    pub fn find_joint(&self, id: BoneId) -> Option<&SkeletonJoint> {
        self.joints.iter().find(|j| j.id == id)
    }

    /// 関節のワールド位置を取得。
    #[must_use]
    pub fn joint_world_position(&self, id: BoneId) -> Option<Vec3k> {
        self.find_joint(id)
            .map(|j| crate::nan::canon3(j.world_position))
    }

    /// 関節のローカル回転を設定。
    pub fn set_rotation(&mut self, id: BoneId, rotation: Vec3k) {
        if let Some(j) = self.joints.iter_mut().find(|j| j.id == id) {
            j.local_rotation = rotation;
        }
    }

    /// 全関節のローカル回転をリセット。
    pub fn reset_pose(&mut self) {
        for j in &mut self.joints {
            j.local_rotation = Vec3k::ZERO;
        }
        self.update_world_positions();
    }

    /// 2関節間の距離。
    #[must_use]
    pub fn distance_between(&self, a: BoneId, b: BoneId) -> Option<f32> {
        let pa = self.joint_world_position(a)?;
        let pb = self.joint_world_position(b)?;
        Some(crate::nan::canon(pa.distance(pb)))
    }

    /// 子関節のリストを取得。
    #[must_use]
    pub fn children_of(&self, parent_idx: usize) -> Vec<usize> {
        self.joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.parent == Some(parent_idx))
            .map(|(i, _)| i)
            .collect()
    }

    /// 身長に応じたスケーリング。
    pub fn rescale(&mut self, new_height: f32) {
        let scale = new_height / self.height;
        for j in &mut self.joints {
            j.bone_length *= scale;
            j.local_position = j.local_position.scale(scale);
        }
        self.height = new_height;
        self.update_world_positions();
        // what it stores is read back through find_joint / the pub fields (ALICE-Law G4)
        let joints = core::mem::take(&mut self.joints);
        *self = Self {
            joints,
            height: self.height,
        }
        .canonical();
    }
}

impl BoneId {
    /// Every bone id, each parent before its children: the order [`Skeleton::from_world_positions`]
    /// lays the joints out in. `UpperChest` follows `Chest`; both IK targets come last.
    pub const ALL: [Self; 24] = [
        Self::Hips,
        Self::Spine,
        Self::Chest,
        Self::UpperChest,
        Self::Neck,
        Self::Head,
        Self::LeftShoulder,
        Self::LeftUpperArm,
        Self::LeftLowerArm,
        Self::LeftHand,
        Self::RightShoulder,
        Self::RightUpperArm,
        Self::RightLowerArm,
        Self::RightHand,
        Self::LeftUpperLeg,
        Self::LeftLowerLeg,
        Self::LeftFoot,
        Self::LeftToe,
        Self::RightUpperLeg,
        Self::RightLowerLeg,
        Self::RightFoot,
        Self::RightToe,
        Self::LeftFootIk,
        Self::RightFootIk,
    ];

    /// The parent in the humanoid hierarchy. With `upper_chest`, the neck and both shoulders hang
    /// from `UpperChest` (which hangs from `Chest`); without it they hang from `Chest`, as in the
    /// placeholder tables. Both IK targets hang from `Hips`. `Hips` is the root.
    #[must_use]
    pub const fn parent_in(self, upper_chest: bool) -> Option<Self> {
        let top = if upper_chest {
            Self::UpperChest
        } else {
            Self::Chest
        };
        match self {
            Self::Hips => None,
            Self::Spine
            | Self::LeftUpperLeg
            | Self::RightUpperLeg
            | Self::LeftFootIk
            | Self::RightFootIk => Some(Self::Hips),
            Self::Chest => Some(Self::Spine),
            Self::UpperChest => Some(Self::Chest),
            Self::Neck | Self::LeftShoulder | Self::RightShoulder => Some(top),
            Self::Head => Some(Self::Neck),
            Self::LeftUpperArm => Some(Self::LeftShoulder),
            Self::LeftLowerArm => Some(Self::LeftUpperArm),
            Self::LeftHand => Some(Self::LeftLowerArm),
            Self::RightUpperArm => Some(Self::RightShoulder),
            Self::RightLowerArm => Some(Self::RightUpperArm),
            Self::RightHand => Some(Self::RightLowerArm),
            Self::LeftLowerLeg => Some(Self::LeftUpperLeg),
            Self::LeftFoot => Some(Self::LeftLowerLeg),
            Self::LeftToe => Some(Self::LeftFoot),
            Self::RightLowerLeg => Some(Self::RightUpperLeg),
            Self::RightFoot => Some(Self::RightLowerLeg),
            Self::RightToe => Some(Self::RightFoot),
        }
    }

    /// The same bone on the other side (left and right swapped, the IK targets included); a
    /// bone on the midline is its own mirror.
    #[must_use]
    pub const fn mirror(self) -> Self {
        match self {
            Self::LeftShoulder => Self::RightShoulder,
            Self::LeftUpperArm => Self::RightUpperArm,
            Self::LeftLowerArm => Self::RightLowerArm,
            Self::LeftHand => Self::RightHand,
            Self::RightShoulder => Self::LeftShoulder,
            Self::RightUpperArm => Self::LeftUpperArm,
            Self::RightLowerArm => Self::LeftLowerArm,
            Self::RightHand => Self::LeftHand,
            Self::LeftUpperLeg => Self::RightUpperLeg,
            Self::LeftLowerLeg => Self::RightLowerLeg,
            Self::LeftFoot => Self::RightFoot,
            Self::LeftToe => Self::RightToe,
            Self::RightUpperLeg => Self::LeftUpperLeg,
            Self::RightLowerLeg => Self::LeftLowerLeg,
            Self::RightFoot => Self::LeftFoot,
            Self::RightToe => Self::LeftToe,
            Self::LeftFootIk => Self::RightFootIk,
            Self::RightFootIk => Self::LeftFootIk,
            Self::Hips | Self::Spine | Self::Chest | Self::UpperChest | Self::Neck | Self::Head => {
                self
            }
        }
    }
}

/// Why [`Skeleton::from_world_positions`] refused its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkeletonBuildError {
    /// There is no `Hips` joint (the root).
    MissingRoot,
    /// The same bone id is given more than once.
    Duplicate(BoneId),
    /// A coordinate of this joint is NaN or infinite.
    NonFinite(BoneId),
    /// The joint's parent in the hierarchy is not given, so the joint would hang outside the tree.
    MissingParent {
        /// The joint given.
        bone: BoneId,
        /// Its parent, not given.
        parent: BoneId,
    },
    /// The height is NaN, infinite, zero or negative.
    InvalidHeight,
}

impl core::fmt::Display for SkeletonBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRoot => write!(f, "no Hips joint (the root)"),
            Self::Duplicate(b) => write!(f, "bone {b:?} given more than once"),
            Self::NonFinite(b) => write!(f, "bone {b:?} has a NaN or infinite coordinate"),
            Self::MissingParent { bone, parent } => {
                write!(f, "bone {bone:?} given without its parent {parent:?}")
            }
            Self::InvalidHeight => write!(f, "height is not a finite positive number"),
        }
    }
}

impl Skeleton {
    /// A skeleton whose joints sit at the given world positions (a measured rest pose, for
    /// example), in the units of the input (metres by the crate's convention).
    ///
    /// Any subset of [`BoneId::ALL`] closed under [`BoneId::parent_in`] is accepted, rooted at
    /// `Hips`; giving `UpperChest` moves the neck and shoulders under it. The joints are laid out
    /// in the order of [`BoneId::ALL`], whatever the order of the input. Each joint stores the
    /// position given as its world position (exactly), the difference from its parent as its
    /// local position, the distance to its parent (correctly rounded) as its bone length (0 at the
    /// root) and a zero local rotation (the pose).
    ///
    /// # Errors
    /// [`SkeletonBuildError`]: no `Hips`, a repeated id, a NaN or infinite coordinate, a joint
    /// whose parent is not given, or a height that is not finite and positive.
    pub fn from_world_positions(
        positions: &[(BoneId, Vec3k)],
        height: f32,
    ) -> Result<Self, SkeletonBuildError> {
        if !(height.is_finite() && height > 0.0) {
            return Err(SkeletonBuildError::InvalidHeight);
        }
        let mut given: [Option<Vec3k>; 24] = [None; 24];
        let slot = |id: BoneId| BoneId::ALL.iter().position(|&b| b == id).unwrap_or(0);
        for &(id, p) in positions {
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return Err(SkeletonBuildError::NonFinite(id));
            }
            let k = slot(id);
            if given[k].is_some() {
                return Err(SkeletonBuildError::Duplicate(id));
            }
            given[k] = Some(p);
        }
        if given[slot(BoneId::Hips)].is_none() {
            return Err(SkeletonBuildError::MissingRoot);
        }
        let upper_chest = given[slot(BoneId::UpperChest)].is_some();
        // the joints in the order of ALL (parents first), each with its parent's index in `joints`
        let mut index: [Option<usize>; 24] = [None; 24];
        let mut joints = Vec::new();
        for (k, &id) in BoneId::ALL.iter().enumerate() {
            let Some(world) = given[k] else { continue };
            let (parent, parent_world) = match id.parent_in(upper_chest) {
                None => (None, Vec3k::ZERO),
                Some(p) => match (index[slot(p)], given[slot(p)]) {
                    (Some(i), Some(pw)) => (Some(i), pw),
                    _ => {
                        return Err(SkeletonBuildError::MissingParent {
                            bone: id,
                            parent: p,
                        })
                    }
                },
            };
            let local = world - parent_world;
            index[k] = Some(joints.len());
            joints.push(SkeletonJoint {
                id,
                parent,
                local_position: local,
                local_rotation: Vec3k::ZERO,
                world_position: world,
                // correctly rounded (Vec3k::length is an approximate root)
                bone_length: if parent.is_some() {
                    alice_det_math::sqrt(local.length_sq())
                } else {
                    0.0
                },
            });
        }
        Ok(Self { joints, height }.canonical())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_humanoid() {
        let s = Skeleton::default_humanoid();
        assert_eq!(s.joint_count(), BONE_COUNT);
        assert!((s.height - 1.70).abs() < 1e-6);
    }

    #[test]
    fn with_height() {
        let s = Skeleton::with_height(1.80);
        assert!((s.height - 1.80).abs() < 1e-6);
    }

    #[test]
    fn find_joint_hips() {
        let s = Skeleton::default_humanoid();
        let hips = s.find_joint(BoneId::Hips).unwrap();
        assert_eq!(hips.id, BoneId::Hips);
        assert!(hips.parent.is_none());
    }

    #[test]
    fn find_joint_head() {
        let s = Skeleton::default_humanoid();
        let head = s.find_joint(BoneId::Head).unwrap();
        assert_eq!(head.id, BoneId::Head);
        assert!(head.parent.is_some());
    }

    #[test]
    fn world_position_hips_at_origin() {
        let s = Skeleton::default_humanoid();
        let pos = s.joint_world_position(BoneId::Hips).unwrap();
        assert!((pos.x).abs() < 1e-6);
        assert!((pos.z).abs() < 1e-6);
    }

    #[test]
    fn head_above_hips() {
        let s = Skeleton::default_humanoid();
        let hips_y = s.joint_world_position(BoneId::Hips).unwrap().y;
        let head_y = s.joint_world_position(BoneId::Head).unwrap().y;
        assert!(head_y > hips_y);
    }

    #[test]
    fn spine_hierarchy() {
        let s = Skeleton::default_humanoid();
        let spine = s.find_joint(BoneId::Spine).unwrap();
        assert_eq!(spine.parent, Some(0)); // Hips
    }

    #[test]
    fn children_of_hips() {
        let s = Skeleton::default_humanoid();
        let children = s.children_of(0);
        assert!(children.len() >= 3); // Spine, LeftUpperLeg, RightUpperLeg, ...
    }

    #[test]
    fn children_of_leaf() {
        let s = Skeleton::default_humanoid();
        // Head has no children in this model
        let children = s.children_of(4); // Head
        assert!(children.is_empty());
    }

    #[test]
    fn distance_between_joints() {
        let s = Skeleton::default_humanoid();
        let dist = s.distance_between(BoneId::Hips, BoneId::Head).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn set_rotation() {
        let mut s = Skeleton::default_humanoid();
        s.set_rotation(BoneId::Neck, Vec3k::new(0.1, 0.0, 0.0));
        let neck = s.find_joint(BoneId::Neck).unwrap();
        assert!((neck.local_rotation.x - 0.1).abs() < 1e-6);
    }

    #[test]
    fn reset_pose() {
        let mut s = Skeleton::default_humanoid();
        s.set_rotation(BoneId::Neck, Vec3k::new(0.5, 0.3, 0.1));
        s.reset_pose();
        let neck = s.find_joint(BoneId::Neck).unwrap();
        assert!((neck.local_rotation.x).abs() < 1e-12);
    }

    #[test]
    fn rescale() {
        let mut s = Skeleton::default_humanoid();
        s.rescale(2.0);
        assert!((s.height - 2.0).abs() < 1e-6);
        // ボーン長もスケールされる
        let spine = s.find_joint(BoneId::Spine).unwrap();
        let expected = 0.18 * (2.0 / 1.70);
        assert!((spine.bone_length - expected).abs() < 0.01);
    }

    #[test]
    fn bone_id_eq() {
        assert_eq!(BoneId::Hips, BoneId::Hips);
        assert_ne!(BoneId::Hips, BoneId::Head);
    }

    #[test]
    fn symmetric_arms() {
        let s = Skeleton::default_humanoid();
        let l = s.find_joint(BoneId::LeftUpperArm).unwrap();
        let r = s.find_joint(BoneId::RightUpperArm).unwrap();
        assert!((l.bone_length - r.bone_length).abs() < 1e-6);
    }
}
