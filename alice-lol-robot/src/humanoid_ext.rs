//! humanoid extension trait — LOL [`IntentNode`] を Kinematics の [`Skeleton`] に反映
//!
//! **Additive adapter pattern**: [`alice_kinematics::skeleton`] は触らず、
//! [`HumanoidIntent`] trait を本 crate 側で `impl` する
//!
//! # 単位と座標
//!
//! [`Skeleton`] の単位 (m、[`Skeleton::with_height`] の T-pose、`Hips` が根で原点)
//! 関節は自分の骨の端にある (例: `LeftUpperArm` の関節 = 肘)
//!
//! # MVP scope (Phase R.2)
//!
//! - [`IntentNode::Walk`] `{ destination, speed }` → 根 (`Hips`) の local を destination に置いて
//!   FK をやり直す (全関節が同じ量だけ平行移動する剛体移動)
//! - [`IntentNode::Rest`] `{ duration_ms }` → 変化なし (clone)
//! - [`IntentNode::Sequence`] → 順次 apply の fold
//! - [`IntentNode::Parallel`] → 先頭 verb のみ apply (concurrency は semantic loss)
//! - 他 verb (`Grasp` / `Point` / `Push` / `Pull` / `Follow` / `Avoid` 等) → 変化なし
//!
//! # Phase R.6+ 計画
//!
//! - `Grasp` / `Point` / `Throw`: `alice_kinematics::predictor` の jerk-min 軌道 solver を
//!   経由して腕の関節の回転を FK で更新
//! - `Follow` / `Avoid`: navigation controller (別 crate) と結線
//! - `Gaze` / `Rotate` / `Align` は Kinematics scope 外、Foundry / Social Intent 経路で表現

use std::sync::Arc;

use alice_kinematics::skeleton::{BoneId, Skeleton};
use alice_kinematics::Vec3k;
use alice_lol::intent::IntentNode;
use alice_lol::SdfNode;

use crate::error::RobotError;

/// 関節の名前 → [`Skeleton`] の関節 (名前で関節を指す所はこの表だけを使う)
///
/// 関節は自分の骨の端にあるので、肘は `LeftUpperArm`、手首は `LeftLowerArm`、膝は
/// `LeftUpperLeg`、足首は `LeftLowerLeg` 股関節は根 (`Hips`) にある (placeholder の
/// [`Skeleton`] は腰の幅を持たず、左右の脚は x = 0 で重なる)
pub const JOINTS: [(&str, BoneId); 16] = [
    ("head", BoneId::Head),
    ("neck", BoneId::Neck),
    ("chest", BoneId::Chest),
    ("waist", BoneId::Hips),
    ("left_shoulder", BoneId::LeftShoulder),
    ("right_shoulder", BoneId::RightShoulder),
    ("left_elbow", BoneId::LeftUpperArm),
    ("right_elbow", BoneId::RightUpperArm),
    ("left_wrist", BoneId::LeftLowerArm),
    ("right_wrist", BoneId::RightLowerArm),
    ("left_hip", BoneId::Hips),
    ("right_hip", BoneId::Hips),
    ("left_knee", BoneId::LeftUpperLeg),
    ("right_knee", BoneId::RightUpperLeg),
    ("left_ankle", BoneId::LeftLowerLeg),
    ("right_ankle", BoneId::RightLowerLeg),
];

/// [`JOINTS`] の名前の関節の world 位置 (表に無い名前・骨格に無い関節は `None`)
#[must_use]
pub fn joint_position(skeleton: &Skeleton, name: &str) -> Option<Vec3k> {
    let (_, id) = JOINTS.iter().find(|(n, _)| *n == name)?;
    skeleton.joint_world_position(*id)
}

/// [`Skeleton`] に LOL [`IntentNode`] を反映する trait
pub trait HumanoidIntent: Sized {
    /// Intent を骨格に反映し、新しい骨格を返す
    ///
    /// # Errors
    ///
    /// - [`RobotError::PoseFail`]: `Hips` 関節が骨格に無い
    fn apply_intent(&self, intent: &IntentNode) -> Result<Self, RobotError>;
}

impl HumanoidIntent for Skeleton {
    fn apply_intent(&self, intent: &IntentNode) -> Result<Self, RobotError> {
        apply_intent_impl(self, intent)
    }
}

/// Internal impl (recursive [`IntentNode::Sequence`] / [`IntentNode::Parallel`] handling
/// を分離)
fn apply_intent_impl(skeleton: &Skeleton, intent: &IntentNode) -> Result<Skeleton, RobotError> {
    match intent {
        IntentNode::Walk { destination, .. } => translate_to_destination(skeleton, *destination),

        // 変化なし verb 群 (MVP scope)
        IntentNode::Rest { .. }
        | IntentNode::Grasp { .. }
        | IntentNode::Release { .. }
        | IntentNode::Point { .. }
        | IntentNode::Throw { .. }
        | IntentNode::Catch { .. }
        | IntentNode::Push { .. }
        | IntentNode::Pull { .. }
        | IntentNode::Follow { .. }
        | IntentNode::Avoid { .. }
        | IntentNode::Gaze { .. }
        | IntentNode::Rotate { .. }
        | IntentNode::Align { .. }
        | IntentNode::LatentIntent { .. }
        | IntentNode::Music { .. } => Ok(skeleton.clone()),

        // 合成 verb
        IntentNode::Sequence(children) => {
            let mut acc = skeleton.clone();
            for child in children {
                acc = apply_intent_impl(&acc, child)?;
            }
            Ok(acc)
        }
        IntentNode::Parallel(children) => {
            // MVP: 先頭のみ apply (concurrency semantic loss、doc に明記済)
            children.first().map_or_else(
                || Ok(skeleton.clone()),
                |first| apply_intent_impl(skeleton, first),
            )
        }
    }
}

/// 根 (`Hips`) の local を destination に置いて FK をやり直す (剛体の平行移動)
///
/// # Errors
///
/// - [`RobotError::PoseFail`]: `Hips` が骨格に無い
fn translate_to_destination(
    skeleton: &Skeleton,
    destination: glam::Vec3,
) -> Result<Skeleton, RobotError> {
    let mut posed = skeleton.clone();
    let hips = posed
        .joints
        .iter_mut()
        .find(|j| j.id == BoneId::Hips)
        .ok_or(RobotError::PoseFail("Hips joint missing"))?;
    hips.local_position = Vec3k::new(destination.x, destination.y, destination.z);
    posed.update_world_positions();
    Ok(posed)
}

/// 骨格の骨を capsule にした鎖の SDF (親の関節から子の関節まで、半径 `radius`、`SmoothUnion` の
/// 滑らかさ `smoothness_k`、骨格の関節の順に左から畳む)
///
/// 長さ 0 の骨 (親と同じ位置の関節、例: 足の IK target) は入れない 骨が 1 本も無い骨格は
/// `None`
#[must_use]
pub fn skeleton_capsules(skeleton: &Skeleton, radius: f32, smoothness_k: f32) -> Option<SdfNode> {
    skeleton
        .joints
        .iter()
        .filter_map(|j| {
            let parent = skeleton.joints.get(j.parent?)?;
            let (a, b) = (parent.world_position, j.world_position);
            ((b - a).length_sq() > 0.0).then(|| SdfNode::Capsule {
                point_a: glam::Vec3::new(a.x, a.y, a.z),
                point_b: glam::Vec3::new(b.x, b.y, b.z),
                radius,
            })
        })
        .reduce(|acc, next| SdfNode::SmoothUnion {
            a: Arc::new(acc),
            b: Arc::new(next),
            k: smoothness_k,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alice_lol::intent::{grasp, parallel, point, rest, sequence, walk, HandSide};
    use glam::Vec3;

    fn at(s: &Skeleton, id: BoneId) -> Vec3k {
        s.joint_world_position(id).expect("joint present")
    }

    #[test]
    fn walk_puts_the_root_at_the_destination() {
        let s = Skeleton::default_humanoid();
        let dest = Vec3::new(2.0, 0.0, 1.0);
        let posed = s.apply_intent(&walk(dest, 1.0)).unwrap();
        assert_eq!(at(&posed, BoneId::Hips), Vec3k::new(2.0, 0.0, 1.0));
    }

    #[test]
    fn rest_grasp_and_point_leave_every_joint() {
        let s = Skeleton::default_humanoid();
        for intent in [
            rest(500),
            grasp(0, HandSide::Right, 3.0),
            point(Vec3::new(2.0, 1.0, 0.0), HandSide::Right),
        ] {
            let posed = s.apply_intent(&intent).unwrap();
            let mut compared = 0;
            for (a, b) in s.joints.iter().zip(&posed.joints) {
                assert_eq!(a.world_position, b.world_position, "{:?}", a.id);
                compared += 1;
            }
            assert_eq!(compared, 22);
        }
    }

    #[test]
    fn sequence_applies_verbs_in_order() {
        let s = Skeleton::default_humanoid();
        let seq = sequence(vec![
            walk(Vec3::new(1.0, 0.0, 0.0), 1.0),
            walk(Vec3::new(3.0, 0.0, 0.0), 1.0),
        ]);
        let posed = s.apply_intent(&seq).unwrap();
        // the last Walk wins
        assert_eq!(at(&posed, BoneId::Hips), Vec3k::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn parallel_applies_its_first_verb_only() {
        let s = Skeleton::default_humanoid();
        let par = parallel(vec![
            walk(Vec3::new(1.0, 0.0, 0.0), 1.0),
            walk(Vec3::new(5.0, 0.0, 0.0), 1.0),
        ]);
        let posed = s.apply_intent(&par).unwrap();
        assert_eq!(at(&posed, BoneId::Hips), Vec3k::new(1.0, 0.0, 0.0));
        let empty = s.apply_intent(&parallel(vec![])).unwrap();
        assert_eq!(at(&empty, BoneId::Hips), at(&s, BoneId::Hips));
    }

    #[test]
    fn walk_without_hips_is_a_pose_error() {
        let mut s = Skeleton::default_humanoid();
        s.joints.retain(|j| j.id != BoneId::Hips);
        let e = s.apply_intent(&walk(Vec3::ONE, 1.0)).unwrap_err();
        assert_eq!(e, RobotError::PoseFail("Hips joint missing"));
    }

    #[test]
    fn the_name_table_names_each_old_joint_once() {
        let mut names: Vec<&str> = JOINTS.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 16);
        let s = Skeleton::default_humanoid();
        assert_eq!(joint_position(&s, "head"), Some(at(&s, BoneId::Head)));
        assert_eq!(
            joint_position(&s, "left_elbow"),
            Some(at(&s, BoneId::LeftUpperArm))
        );
        assert_eq!(joint_position(&s, "tail"), None);
    }
}
