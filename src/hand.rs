//! Hand model — 20+ `DoF` finger kinematics
//!
//! 5指 × 4関節 = 20 `DoF` のハンドモデル。
//! 各指に MCP/PIP/DIP/TIP の関節を持ち、
//! 解剖学的制約に基づく屈曲/伸展を実装。
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::joint::{Joint, JointConstraint, Vec3k};

/// 指の種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerType {
    /// 親指。
    Thumb,
    /// 人差し指。
    Index,
    /// 中指。
    Middle,
    /// 薬指。
    Ring,
    /// 小指。
    Pinky,
}

/// 指の関節インデックス。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerJoint {
    /// 中手指節関節 (根元)。
    Mcp,
    /// 近位指節間関節。
    Pip,
    /// 遠位指節間関節。
    Dip,
    /// 指先。
    Tip,
}

/// 指のリンク長 (メートル)。
const FINGER_LENGTHS: [[f32; 4]; 5] = [
    // Thumb: MCP, PIP(IP), DIP, TIP
    [0.040, 0.032, 0.024, 0.020],
    // Index
    [0.045, 0.025, 0.018, 0.015],
    // Middle
    [0.048, 0.028, 0.020, 0.015],
    // Ring
    [0.045, 0.025, 0.018, 0.015],
    // Pinky
    [0.038, 0.020, 0.015, 0.012],
];

/// 単指モデル (4関節)。
#[derive(Debug, Clone)]
pub struct Finger {
    /// 指の種別。
    pub finger_type: FingerType,
    /// 4関節 (MCP, PIP, DIP, TIP)。
    pub joints: [Joint; 4],
    /// 指の根元位置 (手の平座標系)。
    pub base_offset: Vec3k,
}

impl Finger {
    /// 解剖学的制約付きの指を生成。
    #[must_use]
    pub fn new(finger_type: FingerType) -> Self {
        let idx = finger_type as usize;
        let lengths = FINGER_LENGTHS[idx];

        let (constraints, base_offset) = match finger_type {
            FingerType::Thumb => (
                [
                    JointConstraint::new(-10.0, 80.0), // MCP
                    JointConstraint::new(-10.0, 90.0), // IP
                    JointConstraint::new(0.0, 70.0),   // DIP
                    JointConstraint::new(0.0, 0.0),    // TIP (fixed)
                ],
                Vec3k::new(-0.04, 0.0, -0.02),
            ),
            FingerType::Index => (
                [
                    JointConstraint::new(-20.0, 90.0),
                    JointConstraint::new(0.0, 110.0),
                    JointConstraint::new(0.0, 80.0),
                    JointConstraint::new(0.0, 0.0),
                ],
                Vec3k::new(-0.02, 0.0, 0.04),
            ),
            FingerType::Middle => (
                [
                    JointConstraint::new(-20.0, 90.0),
                    JointConstraint::new(0.0, 110.0),
                    JointConstraint::new(0.0, 80.0),
                    JointConstraint::new(0.0, 0.0),
                ],
                Vec3k::new(0.0, 0.0, 0.045),
            ),
            FingerType::Ring => (
                [
                    JointConstraint::new(-20.0, 90.0),
                    JointConstraint::new(0.0, 110.0),
                    JointConstraint::new(0.0, 80.0),
                    JointConstraint::new(0.0, 0.0),
                ],
                Vec3k::new(0.02, 0.0, 0.04),
            ),
            FingerType::Pinky => (
                [
                    JointConstraint::new(-20.0, 90.0),
                    JointConstraint::new(0.0, 100.0),
                    JointConstraint::new(0.0, 80.0),
                    JointConstraint::new(0.0, 0.0),
                ],
                Vec3k::new(0.04, 0.0, 0.035),
            ),
        };

        let axis = Vec3k::new(1.0, 0.0, 0.0);
        let names: [&[u8]; 4] = [
            b"mcp\0\0\0\0\0",
            b"pip\0\0\0\0\0",
            b"dip\0\0\0\0\0",
            b"tip\0\0\0\0\0",
        ];

        let joints =
            core::array::from_fn(|i| Joint::new(names[i], axis, lengths[i], constraints[i]));

        Self {
            finger_type,
            joints,
            base_offset,
        }
    }

    /// 指先の FK 位置 (指根元からの相対)。
    #[must_use]
    pub fn tip_position(&self) -> Vec3k {
        let mut pos = self.base_offset;
        let mut dir = Vec3k::new(0.0, 0.0, 1.0); // 指先方向

        for j in &self.joints {
            dir = crate::joint::rotate_vec(dir, j.axis, j.angle);
            pos = pos + dir.scale(j.link_length);
        }
        pos
    }

    /// 全関節角度を取得。
    #[must_use]
    pub fn angles(&self) -> [f32; 4] {
        core::array::from_fn(|i| self.joints[i].angle)
    }

    /// 全関節角度を設定。
    pub fn set_angles(&mut self, angles: &[f32; 4]) {
        for (j, &a) in self.joints.iter_mut().zip(angles.iter()) {
            j.set_angle(a);
        }
    }

    /// 指の総リンク長。
    #[must_use]
    pub fn total_length(&self) -> f32 {
        self.joints.iter().map(|j| j.link_length).sum()
    }

    /// PIP/DIP カップリング: DIP角度を PIP角度の 2/3 に連動。
    pub fn couple_pip_dip(&mut self) {
        let pip_angle = self.joints[1].angle;
        self.joints[2].set_angle(pip_angle * (2.0 / 3.0));
    }
}

/// ハンドモデル (5指 = 20 `DoF`)。
#[derive(Debug, Clone)]
pub struct HandModel {
    /// 5指。
    pub fingers: [Finger; 5],
    /// 手の原点位置。
    pub wrist_position: Vec3k,
    /// 左手/右手。
    pub is_right: bool,
}

impl HandModel {
    /// 右手モデルを生成。
    #[must_use]
    pub fn right_hand() -> Self {
        Self {
            fingers: [
                Finger::new(FingerType::Thumb),
                Finger::new(FingerType::Index),
                Finger::new(FingerType::Middle),
                Finger::new(FingerType::Ring),
                Finger::new(FingerType::Pinky),
            ],
            wrist_position: Vec3k::ZERO,
            is_right: true,
        }
    }

    /// 左手モデルを生成。
    #[must_use]
    pub fn left_hand() -> Self {
        Self {
            fingers: [
                Finger::new(FingerType::Thumb),
                Finger::new(FingerType::Index),
                Finger::new(FingerType::Middle),
                Finger::new(FingerType::Ring),
                Finger::new(FingerType::Pinky),
            ],
            wrist_position: Vec3k::ZERO,
            is_right: false,
        }
    }

    /// 全指先位置を取得。
    #[must_use]
    pub fn fingertip_positions(&self) -> [Vec3k; 5] {
        core::array::from_fn(|i| {
            let tip = self.fingers[i].tip_position();
            self.wrist_position + tip
        })
    }

    /// 総 `DoF` 数。
    #[must_use]
    pub const fn total_dof() -> usize {
        20
    }

    /// グリップ (全指の MCP + PIP を閉じる)。
    pub fn grip(&mut self, amount: f32) {
        let amount = amount.clamp(0.0, 1.0);
        for finger in &mut self.fingers {
            let max_mcp = finger.joints[0].constraint.max_rad;
            let max_pip = finger.joints[1].constraint.max_rad;
            finger.joints[0].set_angle(max_mcp * amount);
            finger.joints[1].set_angle(max_pip * amount);
            finger.couple_pip_dip();
        }
    }

    /// 全関節をリセット。
    pub fn reset(&mut self) {
        for finger in &mut self.fingers {
            for j in &mut finger.joints {
                j.angle = 0.0;
            }
        }
    }

    /// 指を名前で取得。
    #[must_use]
    pub const fn finger(&self, ft: FingerType) -> &Finger {
        &self.fingers[ft as usize]
    }

    /// 指を名前で取得 (可変)。
    pub const fn finger_mut(&mut self, ft: FingerType) -> &mut Finger {
        &mut self.fingers[ft as usize]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finger_creation() {
        let f = Finger::new(FingerType::Index);
        assert_eq!(f.finger_type, FingerType::Index);
        assert_eq!(f.joints.len(), 4);
    }

    #[test]
    fn finger_all_types() {
        let types = [
            FingerType::Thumb,
            FingerType::Index,
            FingerType::Middle,
            FingerType::Ring,
            FingerType::Pinky,
        ];
        for &ft in &types {
            let f = Finger::new(ft);
            assert_eq!(f.finger_type, ft);
            assert!(f.total_length() > 0.0);
        }
    }

    #[test]
    fn finger_tip_zero_angles() {
        let f = Finger::new(FingerType::Middle);
        let tip = f.tip_position();
        // 全角度0なら指先はz方向に伸びる
        assert!(tip.z > f.base_offset.z);
    }

    #[test]
    fn finger_angles_round_trip() {
        let mut f = Finger::new(FingerType::Index);
        let angles = [0.1, 0.2, 0.3, 0.0];
        f.set_angles(&angles);
        let got = f.angles();
        for i in 0..4 {
            assert!((got[i] - angles[i]).abs() < 0.01);
        }
    }

    #[test]
    fn pip_dip_coupling() {
        let mut f = Finger::new(FingerType::Middle);
        f.joints[1].set_angle(0.9);
        f.couple_pip_dip();
        let expected = 0.9 * (2.0 / 3.0);
        assert!((f.joints[2].angle - expected).abs() < 0.01);
    }

    #[test]
    fn hand_right() {
        let h = HandModel::right_hand();
        assert!(h.is_right);
        assert_eq!(h.fingers.len(), 5);
    }

    #[test]
    fn hand_left() {
        let h = HandModel::left_hand();
        assert!(!h.is_right);
    }

    #[test]
    fn hand_total_dof() {
        assert_eq!(HandModel::total_dof(), 20);
    }

    #[test]
    fn hand_fingertips() {
        let h = HandModel::right_hand();
        let tips = h.fingertip_positions();
        assert_eq!(tips.len(), 5);
    }

    #[test]
    fn hand_grip() {
        let mut h = HandModel::right_hand();
        h.grip(1.0);
        // 全指の MCP が0より大きい
        for f in &h.fingers {
            assert!(f.joints[0].angle > 0.0);
        }
    }

    #[test]
    fn hand_grip_zero() {
        let mut h = HandModel::right_hand();
        h.grip(0.0);
        for f in &h.fingers {
            assert!((f.joints[0].angle).abs() < 0.01);
        }
    }

    #[test]
    fn hand_reset() {
        let mut h = HandModel::right_hand();
        h.grip(0.8);
        h.reset();
        for f in &h.fingers {
            for j in &f.joints {
                assert!((j.angle).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hand_finger_by_type() {
        let h = HandModel::right_hand();
        let thumb = h.finger(FingerType::Thumb);
        assert_eq!(thumb.finger_type, FingerType::Thumb);
    }

    #[test]
    fn hand_finger_mut() {
        let mut h = HandModel::right_hand();
        let idx = h.finger_mut(FingerType::Index);
        idx.joints[0].set_angle(0.5);
        assert!((h.fingers[1].joints[0].angle - 0.5).abs() < 0.01);
    }
}
