//! ALICE-LOL IntentNode → Kinematics Intent 翻訳器 (Milestone B.3、2026-08-06)
//!
//! LOL の `IntentNode` (16 variant、Phase 3 IR skeleton) を Kinematics の
//! `Intent` packet (8-byte、`IntentType` 4 variant) の Vec に変換する
//!
//! # Mapping 概要
//!
//! - **直接 1:1**: `Grasp` → Grasp、`Release` → Release、`Point` → Point、`Walk` → Reach
//! - **複合**: `Throw` → \[Reach, Release\]、`Catch` → \[Grasp\]、`Push`/`Pull` → \[Reach\] (方向合成)、`Follow`/`Avoid` → \[Reach\]
//! - **Unmappable**: `Gaze` / `Rotate` / `Align` (body kinematics scope 外 → Error)
//! - **Silent skip**: `Rest` → 空 Vec (motion なし)
//! - **合成**: `Sequence` / `Parallel` → 中身を flatten (`Parallel` は concurrency 喪失、doc に明記)
//!
//! # NodeId 解決
//!
//! `target_id: NodeId` を持つ verb (Grasp / Catch / Push / Pull / Follow / Avoid) は
//! `positions: &[glam::Vec3]` slice で解決する (NodeId は index) 疎結合設計で
//! LOL Program 型 import を回避
//!
//! # Usage
//!
//! ```ignore
//! use alice_kinematics::lol_bridge::intent_to_kinematics;
//! use alice_lol::intent::{grasp, HandSide};
//! use glam::Vec3;
//!
//! let positions = vec![Vec3::new(1.0, 0.0, 0.5)]; // NodeId 0 → world pos
//! let lol_intent = grasp(0, HandSide::Right, 3.0);
//! let intents = intent_to_kinematics(&lol_intent, &positions).unwrap();
//! assert_eq!(intents.len(), 1);
//! ```

use crate::intent::{Intent, IntentFlags, IntentType};
use crate::joint::Vec3k;
use alice_lol::intent::{HandSide, IntentNode, NodeId};
use alice_lol::Vec3; // glam::Vec3 re-export
use alloc::vec;
use alloc::vec::Vec;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// エラー型
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 翻訳エラー
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslationError {
    /// LOL verb が Kinematics scope 外で翻訳不可能
    /// (Gaze = 目線、Rotate/Align = 物体操作、body kinematics ではない)
    Unmappable(&'static str),
    /// `NodeId` が `positions` slice の range 外
    OutOfRange {
        /// 参照された NodeId
        id: NodeId,
        /// 実際の slice 長
        len: usize,
    },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// メイン翻訳関数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// LOL `IntentNode` を Kinematics `Intent` の Vec に翻訳
///
/// # Arguments
///
/// - `intent`: LOL IntentNode (16 variant のいずれか)
/// - `positions`: NodeId → world Vec3 の解決テーブル (index として利用)
///
/// # Returns
///
/// - Ok(Vec) — 翻訳成功 (Rest 等で 空 Vec もあり)
/// - Err(TranslationError) — Unmappable verb or NodeId out of range
///
/// # Semantics 補足
///
/// - `Parallel` は Vec に flatten されるが、Kinematics は single Intent stream しか
///   扱えないため concurrency 情報は失われる 受信側で「同時開始」semantics が
///   必要なら別途 metadata で伝達する必要
/// - `duration_ms` は heuristic (Walk/Follow: speed から推算、Throw: force から推算)
///   厳密な値が必要なら caller 側で post-process
pub fn intent_to_kinematics(
    intent: &IntentNode,
    positions: &[Vec3],
) -> Result<Vec<Intent>, TranslationError> {
    match intent {
        IntentNode::Grasp {
            target_id,
            hand,
            force,
        } => {
            let target = resolve_id(*target_id, positions)?;
            Ok(vec![make_intent(
                IntentType::Grasp,
                target,
                duration_from_force(*force),
                *hand,
                true, // grip closed
            )])
        }
        IntentNode::Release { target_id } => {
            let target = resolve_id(*target_id, positions)?;
            Ok(vec![make_intent(
                IntentType::Release,
                target,
                50, // default 50ms
                HandSide::Right,
                false,
            )])
        }
        IntentNode::Walk { destination, speed } => Ok(vec![make_intent(
            IntentType::Reach,
            *destination,
            duration_from_speed(destination.length(), *speed),
            HandSide::Right,
            false,
        )]),
        IntentNode::Gaze { .. } => Err(TranslationError::Unmappable(
            "Gaze: body kinematics scope 外 (目線制御)",
        )),
        IntentNode::Point { target, hand } => Ok(vec![make_intent(
            IntentType::Point,
            *target,
            200, // default 200ms
            *hand,
            false,
        )]),
        IntentNode::Throw {
            target,
            force,
            hand,
        } => Ok(vec![
            make_intent(
                IntentType::Reach,
                *target,
                duration_from_force(*force),
                *hand,
                true, // 投擲中は grip closed
            ),
            make_intent(
                IntentType::Release,
                *target,
                30, // 即 release
                *hand,
                false,
            ),
        ]),
        IntentNode::Catch { object_id } => {
            let target = resolve_id(*object_id, positions)?;
            Ok(vec![make_intent(
                IntentType::Grasp,
                target,
                150,
                HandSide::Right,
                true,
            )])
        }
        IntentNode::Push {
            target_id,
            direction,
            force,
        } => {
            let base = resolve_id(*target_id, positions)?;
            let dest = base + *direction * *force * 0.1; // heuristic: force × 0.1m
            Ok(vec![make_intent(
                IntentType::Reach,
                dest,
                duration_from_force(*force),
                HandSide::Right,
                false,
            )])
        }
        IntentNode::Pull {
            target_id,
            direction,
            force,
        } => {
            let base = resolve_id(*target_id, positions)?;
            let dest = base + *direction * *force * 0.1;
            Ok(vec![make_intent(
                IntentType::Reach,
                dest,
                duration_from_force(*force),
                HandSide::Right,
                true, // pull は grip closed
            )])
        }
        IntentNode::Rotate { .. } => Err(TranslationError::Unmappable(
            "Rotate: 物体回転は Kinematics scope 外",
        )),
        IntentNode::Align { .. } => Err(TranslationError::Unmappable(
            "Align: 物体整列は Kinematics scope 外",
        )),
        IntentNode::Follow {
            target_id,
            distance,
        } => {
            let target_pos = resolve_id(*target_id, positions)?;
            // heuristic: 距離 distance を保つ点 (target から自己方向に distance ずらす想定 で 単純化)
            let dest = target_pos - Vec3::new(*distance, 0.0, 0.0);
            Ok(vec![make_intent(
                IntentType::Reach,
                dest,
                200,
                HandSide::Right,
                false,
            )])
        }
        IntentNode::Avoid {
            target_id,
            min_distance,
        } => {
            let target_pos = resolve_id(*target_id, positions)?;
            // heuristic: target から min_distance 離れた場所へ Reach (逆方向単純化)
            let dest = target_pos - Vec3::new(*min_distance, 0.0, 0.0);
            Ok(vec![make_intent(
                IntentType::Reach,
                dest,
                200,
                HandSide::Right,
                false,
            )])
        }
        IntentNode::Rest { .. } => Ok(Vec::new()),
        IntentNode::Sequence(items) | IntentNode::Parallel(items) => {
            let mut acc = Vec::new();
            for item in items {
                acc.extend(intent_to_kinematics(item, positions)?);
            }
            Ok(acc)
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 内部 helper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn resolve_id(id: NodeId, positions: &[Vec3]) -> Result<Vec3, TranslationError> {
    positions
        .get(id as usize)
        .copied()
        .ok_or(TranslationError::OutOfRange {
            id,
            len: positions.len(),
        })
}

fn make_intent(
    ty: IntentType,
    target: Vec3,
    duration_ms: u8,
    hand: HandSide,
    grip_closed: bool,
) -> Intent {
    let is_left = matches!(hand, HandSide::Left);
    Intent {
        target: Vec3k::new(target.x, target.y, target.z),
        duration_ms,
        flags: IntentFlags::new(ty, grip_closed, is_left, 0),
    }
}

/// 移動距離と speed から duration_ms を推算 (heuristic、u8 range にクランプ)
fn duration_from_speed(distance: f32, speed: f32) -> u8 {
    if speed <= 0.0 {
        return 200;
    }
    let sec = distance / speed;
    let ms = (sec * 1000.0).clamp(20.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (ms as u8)
}

/// force から duration_ms を推算 (大きな力ほど短時間、heuristic)
fn duration_from_force(force: f32) -> u8 {
    if force <= 0.0 {
        return 200;
    }
    let ms = (200.0 / (1.0 + force * 0.1)).clamp(20.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (ms as u8)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use alice_lol::intent::{
        align, avoid, catch, follow, gaze, grasp, parallel, point, pull, push, release, rest,
        rotate, sequence, throw, walk, HandSide,
    };

    #[test]
    fn grasp_maps_to_grasp_intent() {
        let positions = vec![Vec3::new(1.0, 0.5, 0.0)];
        let result = intent_to_kinematics(&grasp(0, HandSide::Right, 3.0), &positions).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].flags.intent_type(), IntentType::Grasp);
        assert!(result[0].flags.grip_closed());
        assert!(!result[0].flags.is_left_hand());
    }

    #[test]
    fn release_maps_to_release_intent() {
        let positions = vec![Vec3::new(1.0, 0.5, 0.0)];
        let result = intent_to_kinematics(&release(0), &positions).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].flags.intent_type(), IntentType::Release);
    }

    #[test]
    fn walk_maps_to_reach() {
        let result = intent_to_kinematics(&walk(Vec3::new(3.0, 0.0, 0.0), 1.5), &[]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].flags.intent_type(), IntentType::Reach);
        assert!((result[0].target.x - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn point_maps_to_point() {
        let result =
            intent_to_kinematics(&point(Vec3::new(2.0, 1.0, 0.0), HandSide::Left), &[]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].flags.intent_type(), IntentType::Point);
        assert!(result[0].flags.is_left_hand());
    }

    #[test]
    fn throw_maps_to_reach_plus_release() {
        let result =
            intent_to_kinematics(&throw(Vec3::new(5.0, 3.0, 0.0), 10.0, HandSide::Right), &[])
                .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].flags.intent_type(), IntentType::Reach);
        assert!(result[0].flags.grip_closed());
        assert_eq!(result[1].flags.intent_type(), IntentType::Release);
    }

    #[test]
    fn catch_maps_to_grasp() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 0.5)];
        let result = intent_to_kinematics(&catch(1), &positions).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].flags.intent_type(), IntentType::Grasp);
    }

    #[test]
    fn push_pull_map_to_reach() {
        let positions = vec![Vec3::new(1.0, 0.0, 0.0)];
        let push_r =
            intent_to_kinematics(&push(0, Vec3::new(1.0, 0.0, 0.0), 5.0), &positions).unwrap();
        assert_eq!(push_r[0].flags.intent_type(), IntentType::Reach);
        assert!(!push_r[0].flags.grip_closed());

        let pull_r =
            intent_to_kinematics(&pull(0, Vec3::new(-1.0, 0.0, 0.0), 5.0), &positions).unwrap();
        assert_eq!(pull_r[0].flags.intent_type(), IntentType::Reach);
        assert!(pull_r[0].flags.grip_closed(), "pull は grip closed");
    }

    #[test]
    fn follow_avoid_map_to_reach() {
        let positions = vec![Vec3::new(3.0, 0.0, 0.0)];
        let f = intent_to_kinematics(&follow(0, 1.5), &positions).unwrap();
        assert_eq!(f[0].flags.intent_type(), IntentType::Reach);

        let a = intent_to_kinematics(&avoid(0, 2.0), &positions).unwrap();
        assert_eq!(a[0].flags.intent_type(), IntentType::Reach);
    }

    #[test]
    fn gaze_unmappable() {
        let err = intent_to_kinematics(&gaze(Vec3::ZERO, 500), &[]).unwrap_err();
        assert!(matches!(err, TranslationError::Unmappable(_)));
    }

    #[test]
    fn rotate_unmappable() {
        let err = intent_to_kinematics(&rotate(0, Vec3::Y, 1.0), &[]).unwrap_err();
        assert!(matches!(err, TranslationError::Unmappable(_)));
    }

    #[test]
    fn align_unmappable() {
        let err = intent_to_kinematics(&align(0, Vec3::Z), &[]).unwrap_err();
        assert!(matches!(err, TranslationError::Unmappable(_)));
    }

    #[test]
    fn rest_maps_to_empty_vec() {
        let result = intent_to_kinematics(&rest(1000), &[]).unwrap();
        assert!(result.is_empty(), "Rest は空 Vec (silent)");
    }

    #[test]
    fn sequence_flattens() {
        let positions = vec![Vec3::new(1.0, 0.0, 0.0)];
        let seq = sequence(vec![
            grasp(0, HandSide::Right, 3.0),
            walk(Vec3::new(2.0, 0.0, 0.0), 1.0),
            release(0),
        ]);
        let result = intent_to_kinematics(&seq, &positions).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].flags.intent_type(), IntentType::Grasp);
        assert_eq!(result[1].flags.intent_type(), IntentType::Reach);
        assert_eq!(result[2].flags.intent_type(), IntentType::Release);
    }

    #[test]
    fn parallel_flattens_losing_concurrency() {
        let par = parallel(vec![
            walk(Vec3::new(2.0, 0.0, 0.0), 1.0),
            point(Vec3::new(3.0, 1.0, 0.0), HandSide::Right),
        ]);
        let result = intent_to_kinematics(&par, &[]).unwrap();
        assert_eq!(result.len(), 2, "Parallel は flatten で 2 Intent");
    }

    #[test]
    fn sequence_with_rest_skips_empty() {
        let seq = sequence(vec![
            walk(Vec3::new(2.0, 0.0, 0.0), 1.0),
            rest(500),
            walk(Vec3::new(4.0, 0.0, 0.0), 1.0),
        ]);
        let result = intent_to_kinematics(&seq, &[]).unwrap();
        assert_eq!(result.len(), 2, "Rest は 空、他 2 verb 残る");
    }

    #[test]
    fn out_of_range_error() {
        let positions = vec![Vec3::ZERO]; // len = 1
        let err = intent_to_kinematics(&grasp(5, HandSide::Right, 3.0), &positions).unwrap_err();
        match err {
            TranslationError::OutOfRange { id, len } => {
                assert_eq!(id, 5);
                assert_eq!(len, 1);
            }
            _ => panic!("OutOfRange を期待"),
        }
    }

    #[test]
    fn empty_positions_makes_id_verbs_error() {
        let err = intent_to_kinematics(&catch(0), &[]).unwrap_err();
        assert!(matches!(err, TranslationError::OutOfRange { .. }));
    }
}
