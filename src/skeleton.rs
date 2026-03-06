//! Full-body skeleton — 22 joint hierarchical model
//!
//! 全身骨格の階層構造と FK 伝播。
//! VR/モーションキャプチャ向けの標準的な関節配置。
//!
//! License: MIT
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

/// 全身骨格モデル。
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// 全関節。
    pub joints: Vec<SkeletonJoint>,
    /// 身長 (メートル)。
    pub height: f32,
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

impl Skeleton {
    /// 標準体型の骨格を生成 (身長 1.7m)。
    #[must_use]
    pub fn default_humanoid() -> Self {
        Self::with_height(1.70)
    }

    /// 指定身長の骨格を生成。
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
                let local_y = bone_length;
                SkeletonJoint {
                    id,
                    parent: PARENT_TABLE[i],
                    local_position: Vec3k::new(0.0, local_y, 0.0),
                    local_rotation: Vec3k::ZERO,
                    world_position: Vec3k::ZERO,
                    bone_length,
                }
            })
            .collect();

        let mut skel = Self { joints, height };
        skel.update_world_positions();
        skel
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
        self.find_joint(id).map(|j| j.world_position)
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
        Some(pa.distance(pb))
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
