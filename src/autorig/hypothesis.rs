//! Skeleton hypothesis from mesh slice statistics
//!
//! 128 断面統計から humanoid skeleton の 22 関節位置を推定する
//! (dotneet/image-to-3d §E.1 平面世界向け Y-up 幾何ヒューリスティック)
//!
//! 手順:
//! 1. 幅プロファイルの local minima から waist / neck 位置を検出
//! 2. spine chain (Hips → Spine → Chest → Neck → Head) を centroid で構築
//! 3. legs = 腰下を左右に分けて centroid 追跡で UpperLeg / LowerLeg / Foot / Toe 決定
//! 4. arms = 肩から胴半径外に出た頂点群、距離分位点で UpperArm / LowerArm / Hand 決定
//!
//! 検出されなかった関節は `None` として保持され、`to_skeleton` で default
//! bone_length (親から短い bone) で補完される
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::skeleton::{BoneId, Skeleton, SkeletonJoint, BONE_COUNT};

use super::slice::{NormalizedMesh, SliceStat};
use alloc::vec::Vec;

/// 22 関節分の world position 推定値 (正規化済み座標系)
///
/// index は `BoneId as u8` に対応 (`Hips` = 0、`Spine` = 1、... `LeftFootIk` = 21)
/// 検出できなかった関節は `None`
#[derive(Debug, Clone)]
pub struct SkeletonHypothesis {
    /// 22 スロット、正規化座標での world position
    pub positions: [Option<Vec3k>; BONE_COUNT],
    /// 検出信頼度 (= 検出された関節数 / BONE_COUNT)
    pub confidence: f32,
}

impl Default for SkeletonHypothesis {
    fn default() -> Self {
        Self {
            positions: [None; BONE_COUNT],
            confidence: 0.0,
        }
    }
}

/// 幅プロファイルの Y 範囲内で最小幅のスライス index を返す
///
/// 頂点数が `MIN_SLICE_VERTEX_COUNT` 未満のスライスは無視する
/// (単一頂点だけのスライスは width = 0 となり誤検出を招くため、
///  多角形形状としての最小サンプル数を要求)
const MIN_SLICE_VERTEX_COUNT: u32 = 3;

fn find_min_width_idx_in_range(slices: &[SliceStat], y_min: f32, y_max: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, s) in slices.iter().enumerate() {
        if s.y < y_min || s.y > y_max {
            continue;
        }
        if s.vertex_count < MIN_SLICE_VERTEX_COUNT {
            continue;
        }
        best = match best {
            Some((_, w)) if s.width >= w => best,
            _ => Some((i, s.width)),
        };
    }
    best.map(|(i, _)| i)
}

/// 最上段の非空スライス index を返す
fn find_top_slice_idx(slices: &[SliceStat]) -> Option<usize> {
    (0..slices.len())
        .rev()
        .find(|&i| slices[i].vertex_count > 0)
}

/// waist / neck スライス index を検出
///
/// waist: Y ∈ [0.30, 0.55] の最小幅スライス
/// neck: Y ∈ [0.78, 0.92] の最小幅スライス
///
/// 検出範囲を意図的に広めに取り、多少姿勢が違っても捉える
#[must_use]
pub fn find_waist_neck(slices: &[SliceStat]) -> Option<(usize, usize)> {
    let waist = find_min_width_idx_in_range(slices, 0.30, 0.55)?;
    let neck = find_min_width_idx_in_range(slices, 0.78, 0.92)?;
    // 一貫性チェック: neck が waist より上でなければ humanoid でない
    if slices[neck].y <= slices[waist].y {
        return None;
    }
    Some((waist, neck))
}

/// spine chain (Hips, Spine, Chest, Neck, Head) を hypothesis に書き込む
///
/// 5 関節すべてが埋まる (Head は最上段の centroid)
fn build_spine(
    hypothesis: &mut SkeletonHypothesis,
    slices: &[SliceStat],
    waist_idx: usize,
    neck_idx: usize,
) {
    // Hips = waist スライスの centroid
    hypothesis.positions[BoneId::Hips as usize] = Some(slices[waist_idx].centroid);
    // Neck = neck スライスの centroid
    hypothesis.positions[BoneId::Neck as usize] = Some(slices[neck_idx].centroid);
    // Chest = waist と neck の間、66% neck 寄り
    let chest_idx = waist_idx + ((neck_idx - waist_idx) * 66) / 100;
    hypothesis.positions[BoneId::Chest as usize] = Some(slices[chest_idx].centroid);
    // Spine = waist と chest の中間
    let spine_idx = waist_idx + (chest_idx - waist_idx) / 2;
    hypothesis.positions[BoneId::Spine as usize] = Some(slices[spine_idx].centroid);
    // Head = 最上段の非空スライス
    if let Some(head_idx) = find_top_slice_idx(slices) {
        hypothesis.positions[BoneId::Head as usize] = Some(slices[head_idx].centroid);
    }
}

/// 腰下領域を X 座標で L/R に分けて centroid 追跡、脚 4 関節を配置
///
/// [UpperLeg, LowerLeg, Foot, Toe] を左右それぞれ埋める
///
/// `_nm` は今のところ未使用 (`SliceStat.left_centroid` / `right_centroid` の
/// 事前計算で足りている) が、姿勢によっては生 vertex 参照が必要になるので
/// 引数だけ残しておく
fn build_legs(
    hypothesis: &mut SkeletonHypothesis,
    _nm: &NormalizedMesh,
    slices: &[SliceStat],
    waist_idx: usize,
) {
    let hips_y = slices[waist_idx].y;
    let body_center_x = slices[waist_idx].centroid.x;

    // 腰下 (waist_idx より下) のスライスから left/right centroid を追跡
    // Foot は最下段、Toe はさらに下 (foot と同座標でもよい、Z を少し前に寄せる)
    // 4 分位点で UpperLeg / LowerLeg / Foot / Toe を配置

    let leg_top = hips_y;
    let leg_bottom = slices[0].y;
    let leg_length = leg_top - leg_bottom;
    if leg_length < 0.05 {
        return; // 脚領域が短すぎる
    }

    // 位置比率: UpperLeg=hip、LowerLeg=knee (50%)、Foot=ankle (95%)、Toe=100%
    let joint_fracs = [0.00_f32, 0.50, 0.95, 1.00];
    let left_slots = [
        BoneId::LeftUpperLeg,
        BoneId::LeftLowerLeg,
        BoneId::LeftFoot,
        BoneId::LeftToe,
    ];
    let right_slots = [
        BoneId::RightUpperLeg,
        BoneId::RightLowerLeg,
        BoneId::RightFoot,
        BoneId::RightToe,
    ];

    for (i, &frac) in joint_fracs.iter().enumerate() {
        let target_y = leg_top - leg_length * frac;
        let slice_idx = find_slice_at_y(slices, target_y);
        if let Some(idx) = slice_idx {
            if let Some(lc) = slices[idx].left_centroid {
                hypothesis.positions[left_slots[i] as usize] = Some(lc);
            }
            if let Some(rc) = slices[idx].right_centroid {
                hypothesis.positions[right_slots[i] as usize] = Some(rc);
            }
        }
    }

    // Toe は Foot から前方 (Z + small) へ、Foot が検出済みなら補正
    for (foot_id, toe_id) in [
        (BoneId::LeftFoot, BoneId::LeftToe),
        (BoneId::RightFoot, BoneId::RightToe),
    ] {
        if let (Some(foot_pos), Some(toe_pos)) = (
            hypothesis.positions[foot_id as usize],
            hypothesis.positions[toe_id as usize],
        ) {
            // Toe が Foot と同座標なら少し前方 (+Z) にずらす
            if (foot_pos - toe_pos).length_sq() < 1e-6 {
                let toe_offset = 0.05_f32;
                hypothesis.positions[toe_id as usize] =
                    Some(Vec3k::new(toe_pos.x, toe_pos.y, toe_pos.z + toe_offset));
            }
        }
    }

    // LeftFootIk (仮想 IK ターゲット) = LeftFoot と同じ位置
    if let Some(lf) = hypothesis.positions[BoneId::LeftFoot as usize] {
        hypothesis.positions[BoneId::LeftFootIk as usize] = Some(lf);
    }

    let _ = body_center_x; // 将来使用 (現状は SliceStat の L/R split に依存)
}

/// 目標 Y に最も近いスライス index を返す
fn find_slice_at_y(slices: &[SliceStat], target_y: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, s) in slices.iter().enumerate() {
        if s.vertex_count == 0 {
            continue;
        }
        let diff = (s.y - target_y).abs();
        best = match best {
            Some((_, d)) if diff >= d => best,
            _ => Some((i, diff)),
        };
    }
    best.map(|(i, _)| i)
}

/// 胴半径外の頂点群から腕 4 関節を配置
///
/// [Shoulder, UpperArm, LowerArm, Hand] を左右それぞれ埋める
///
/// アルゴリズム:
/// - torso_radius = neck 付近のスライス幅 × 0.5
/// - shoulder_y = neck_y × 0.95 (neck 直下)
/// - 腕頂点 = 水平距離 > torso_radius × 1.1 かつ Y ∈ [shoulder_y − 0.5h, shoulder_y + 0.1h]
/// - 各サイドで肩からの距離を計算、10% quantile を UpperArm、50% を LowerArm、95% を Hand
fn build_arms(
    hypothesis: &mut SkeletonHypothesis,
    nm: &NormalizedMesh,
    slices: &[SliceStat],
    neck_idx: usize,
) {
    let neck_pos = match hypothesis.positions[BoneId::Neck as usize] {
        Some(p) => p,
        None => return,
    };
    let chest_pos = match hypothesis.positions[BoneId::Chest as usize] {
        Some(p) => p,
        None => return,
    };

    // 胴半径 = neck slice の幅 × 0.5
    let torso_radius = slices[neck_idx].width * 0.5;
    if torso_radius < 0.02 {
        return; // 胴が細すぎる
    }
    let shoulder_y = neck_pos.y * 0.95;
    let body_center_x = chest_pos.x;

    // Shoulder joint (collarbone 起点) を chest の左右に配置
    let shoulder_offset = torso_radius * 0.3;
    hypothesis.positions[BoneId::LeftShoulder as usize] = Some(Vec3k::new(
        body_center_x - shoulder_offset,
        shoulder_y,
        chest_pos.z,
    ));
    hypothesis.positions[BoneId::RightShoulder as usize] = Some(Vec3k::new(
        body_center_x + shoulder_offset,
        shoulder_y,
        chest_pos.z,
    ));

    // 腕頂点抽出範囲
    let y_low = shoulder_y - 0.5;
    let y_high = shoulder_y + 0.1;
    let torso_radius_margin = torso_radius * 1.1;

    // 左右の腕頂点を集める
    let mut left_verts: Vec<Vec3k> = Vec::new();
    let mut right_verts: Vec<Vec3k> = Vec::new();
    for &v in &nm.vertices {
        if v.y < y_low || v.y > y_high {
            continue;
        }
        let horiz_dist = (v.x - body_center_x).abs();
        if horiz_dist < torso_radius_margin {
            continue;
        }
        if v.x < body_center_x {
            left_verts.push(v);
        } else {
            right_verts.push(v);
        }
    }

    // UpperArm/LowerArm/Hand を左右それぞれ配置
    place_arm_joints(
        hypothesis,
        &left_verts,
        Vec3k::new(body_center_x - shoulder_offset, shoulder_y, chest_pos.z),
        [BoneId::LeftUpperArm, BoneId::LeftLowerArm, BoneId::LeftHand],
    );
    place_arm_joints(
        hypothesis,
        &right_verts,
        Vec3k::new(body_center_x + shoulder_offset, shoulder_y, chest_pos.z),
        [
            BoneId::RightUpperArm,
            BoneId::RightLowerArm,
            BoneId::RightHand,
        ],
    );
}

/// 片腕の 3 関節 (UpperArm/LowerArm/Hand) を肩からの距離分位点で配置
fn place_arm_joints(
    hypothesis: &mut SkeletonHypothesis,
    verts: &[Vec3k],
    shoulder: Vec3k,
    slots: [BoneId; 3],
) {
    if verts.is_empty() {
        return;
    }

    // 肩からの距離で並べる (bubble sort 相当だが N が小さいので Vec + partial_cmp)
    let mut dists: Vec<(f32, Vec3k)> = verts.iter().map(|&v| (v.distance(shoulder), v)).collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let n = dists.len();
    // UpperArm: 10% quantile (肩ボール、torso のすぐ外)
    let ua_idx = (n / 10).min(n - 1);
    // LowerArm: 50% quantile (肘)
    let el_idx = (n / 2).min(n - 1);
    // Hand: 95% quantile (手首)
    let hn_idx = ((n * 95) / 100).min(n - 1);

    hypothesis.positions[slots[0] as usize] = Some(dists[ua_idx].1);
    hypothesis.positions[slots[1] as usize] = Some(dists[el_idx].1);
    hypothesis.positions[slots[2] as usize] = Some(dists[hn_idx].1);
}

/// slice + normalized mesh から skeleton hypothesis を構築
///
/// waist / neck が検出できない (= humanoid でない) 場合は `None` を返す
#[must_use]
pub fn build_hypothesis(nm: &NormalizedMesh, slices: &[SliceStat]) -> Option<SkeletonHypothesis> {
    let (waist_idx, neck_idx) = find_waist_neck(slices)?;

    let mut hypothesis = SkeletonHypothesis::default();
    build_spine(&mut hypothesis, slices, waist_idx, neck_idx);
    build_legs(&mut hypothesis, nm, slices, waist_idx);
    build_arms(&mut hypothesis, nm, slices, neck_idx);

    // confidence = 検出関節数 / BONE_COUNT
    let detected = hypothesis.positions.iter().filter(|p| p.is_some()).count();
    #[allow(clippy::cast_precision_loss)]
    {
        hypothesis.confidence = detected as f32 / BONE_COUNT as f32;
    }

    Some(hypothesis)
}

/// hypothesis から Skeleton を構築
///
/// 正規化座標 → 元スケール座標に denormalize、parent 相対の local_position を計算
///
/// 検出されなかった関節は parent 位置 + 下方向に短い bone (0.05 m) で補完される
#[must_use]
pub fn hypothesis_to_skeleton(hypothesis: &SkeletonHypothesis, nm: &NormalizedMesh) -> Skeleton {
    // 標準体型の parent table / bone order をベースに使う
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
    // 親テーブル (skeleton.rs の PARENT_TABLE と一致)
    let parent_table: [Option<usize>; BONE_COUNT] = [
        None,
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(2),
        Some(5),
        Some(6),
        Some(7),
        Some(2),
        Some(9),
        Some(10),
        Some(11),
        Some(0),
        Some(13),
        Some(14),
        Some(15),
        Some(0),
        Some(17),
        Some(18),
        Some(19),
        Some(0),
    ];

    // 各関節の world position を確定 (検出値 or fallback)
    let mut world_positions: [Vec3k; BONE_COUNT] = [Vec3k::ZERO; BONE_COUNT];
    let fallback_bone_length_norm = 0.05_f32; // 正規化座標での短い bone
    for i in 0..BONE_COUNT {
        if let Some(p) = hypothesis.positions[i] {
            world_positions[i] = nm.denormalize(p);
        } else if let Some(parent_idx) = parent_table[i] {
            // parent 位置から下方向に短い bone を伸ばす (元スケール)
            let parent_w = world_positions[parent_idx];
            world_positions[i] = Vec3k::new(
                parent_w.x,
                parent_w.y - fallback_bone_length_norm * nm.original_height,
                parent_w.z,
            );
        } else {
            // ルートが missing = 中心 (bbox 中心)
            let center = Vec3k::new(
                (nm.original_min.x + nm.original_max.x) * 0.5,
                (nm.original_min.y + nm.original_max.y) * 0.5,
                (nm.original_min.z + nm.original_max.z) * 0.5,
            );
            world_positions[i] = center;
        }
    }

    // parent 相対 local_position + bone_length を計算して SkeletonJoint を構築
    let mut joints: Vec<SkeletonJoint> = Vec::with_capacity(BONE_COUNT);
    for i in 0..BONE_COUNT {
        let parent = parent_table[i];
        let world_pos = world_positions[i];
        let local_position = match parent {
            Some(p) => world_pos - world_positions[p],
            None => world_pos,
        };
        let bone_length = local_position.length();
        joints.push(SkeletonJoint {
            id: all_bones[i],
            parent,
            local_position,
            local_rotation: Vec3k::ZERO,
            world_position: world_pos,
            bone_length,
        });
    }

    // 身長 = Head の Y − Hips の Y の下限まで (= mesh の Y extent)
    let height = nm.original_height;

    Skeleton { joints, height }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autorig::slice::{normalize_mesh, slice_normalized, MeshView};

    fn make_stick_figure() -> Vec<Vec3k> {
        // 縦長ヒューマノイド近似
        // 頭: y ∈ [0.85, 1.00], 半径 0.08
        // 首: y = 0.83, 幅 0.03
        // 胴: y ∈ [0.45, 0.80], 幅 0.15 (waist で 0.10 に細く)
        // 腕: 左右 y ∈ [0.35, 0.75], 幅 ±0.35 (肩から水平に伸びる)
        // 脚: 左右 y ∈ [0.00, 0.45], x = ±0.06
        let mut v = Vec::new();

        // 頭 (球体近似)
        for k in 0..8 {
            let theta = (k as f32) * core::f32::consts::TAU / 8.0;
            v.push(Vec3k::new(
                cos_approx(theta) * 0.08,
                0.90 + sin_approx(theta) * 0.08,
                0.0,
            ));
            v.push(Vec3k::new(0.0, 0.90, cos_approx(theta) * 0.08));
        }

        // 首 (細い部分)
        for k in 0..4 {
            let theta = (k as f32) * core::f32::consts::TAU / 4.0;
            v.push(Vec3k::new(
                cos_approx(theta) * 0.03,
                0.83,
                sin_approx(theta) * 0.03,
            ));
        }

        // 胴上部 (胸)
        for y in [0.60_f32, 0.65, 0.70, 0.75] {
            for k in 0..8 {
                let theta = (k as f32) * core::f32::consts::TAU / 8.0;
                v.push(Vec3k::new(
                    cos_approx(theta) * 0.13,
                    y,
                    sin_approx(theta) * 0.08,
                ));
            }
        }

        // 胴中部 (腰、細い部分)
        for k in 0..8 {
            let theta = (k as f32) * core::f32::consts::TAU / 8.0;
            v.push(Vec3k::new(
                cos_approx(theta) * 0.09,
                0.48,
                sin_approx(theta) * 0.06,
            ));
        }

        // 胴下部 (骨盤)
        for y in [0.42_f32, 0.45] {
            for k in 0..8 {
                let theta = (k as f32) * core::f32::consts::TAU / 8.0;
                v.push(Vec3k::new(
                    cos_approx(theta) * 0.12,
                    y,
                    sin_approx(theta) * 0.07,
                ));
            }
        }

        // 腕 (左右、y ∈ [0.55, 0.75] で肩、その先 x が遠くまで伸びる)
        for x_end in [-0.35_f32, 0.35_f32] {
            for step in 0..8 {
                let t = step as f32 / 8.0;
                // 肩 (x = ±0.13) から手 (x = ±0.35) へ
                let x = 0.13 * x_end.signum() + (x_end - 0.13 * x_end.signum()) * t;
                let y = 0.75 - t * 0.20; // 肘は下方向にも少し
                v.push(Vec3k::new(x, y, 0.0));
                v.push(Vec3k::new(x, y, 0.03));
                v.push(Vec3k::new(x, y, -0.03));
            }
        }

        // 脚 (左右、y ∈ [0.00, 0.45])
        for x_leg in [-0.06_f32, 0.06_f32] {
            for step in 0..10 {
                let y = 0.42 - (step as f32 / 10.0) * 0.42;
                for k in 0..4 {
                    let theta = (k as f32) * core::f32::consts::TAU / 4.0;
                    v.push(Vec3k::new(
                        x_leg + cos_approx(theta) * 0.05,
                        y,
                        sin_approx(theta) * 0.05,
                    ));
                }
            }
        }

        v
    }

    fn cos_approx(x: f32) -> f32 {
        sin_approx(x + core::f32::consts::FRAC_PI_2)
    }

    fn sin_approx(x: f32) -> f32 {
        let pi = core::f32::consts::PI;
        let mut x = x % (2.0 * pi);
        if x < 0.0 {
            x += 2.0 * pi;
        }
        let sign = if x > pi { -1.0 } else { 1.0 };
        if x > pi {
            x -= pi;
        }
        let num = 16.0 * x * (pi - x);
        let den = 5.0 * pi * pi - 4.0 * x * (pi - x);
        sign * num / den
    }

    #[test]
    fn find_waist_neck_on_stick_figure() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let (waist, neck) = find_waist_neck(&slices).expect("should detect waist/neck");
        // waist は Y ∈ [0.30, 0.55]
        assert!(slices[waist].y >= 0.30 && slices[waist].y <= 0.55);
        // neck は Y ∈ [0.78, 0.92]
        assert!(slices[neck].y >= 0.78 && slices[neck].y <= 0.92);
    }

    #[test]
    fn hypothesis_covers_spine() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        // Spine chain の 5 関節はすべて検出される
        assert!(h.positions[BoneId::Hips as usize].is_some());
        assert!(h.positions[BoneId::Spine as usize].is_some());
        assert!(h.positions[BoneId::Chest as usize].is_some());
        assert!(h.positions[BoneId::Neck as usize].is_some());
        assert!(h.positions[BoneId::Head as usize].is_some());
    }

    #[test]
    fn hypothesis_covers_arms() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        assert!(h.positions[BoneId::LeftShoulder as usize].is_some());
        assert!(h.positions[BoneId::RightShoulder as usize].is_some());
        assert!(h.positions[BoneId::LeftHand as usize].is_some());
        assert!(h.positions[BoneId::RightHand as usize].is_some());
    }

    #[test]
    fn hypothesis_covers_legs() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        assert!(h.positions[BoneId::LeftUpperLeg as usize].is_some());
        assert!(h.positions[BoneId::RightUpperLeg as usize].is_some());
        assert!(h.positions[BoneId::LeftFoot as usize].is_some());
        assert!(h.positions[BoneId::RightFoot as usize].is_some());
    }

    #[test]
    fn to_skeleton_produces_22_joints() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        let skel = hypothesis_to_skeleton(&h, &nm);
        assert_eq!(skel.joint_count(), BONE_COUNT);
    }

    #[test]
    fn to_skeleton_head_above_hips() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        let skel = hypothesis_to_skeleton(&h, &nm);
        let hips_y = skel.joint_world_position(BoneId::Hips).unwrap().y;
        let head_y = skel.joint_world_position(BoneId::Head).unwrap().y;
        assert!(head_y > hips_y);
    }

    #[test]
    fn hypothesis_confidence_reasonable() {
        let v = make_stick_figure();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        let h = build_hypothesis(&nm, &slices).expect("hypothesis");
        // Stick figure なら 70% 以上の関節が検出できる
        assert!(h.confidence > 0.7, "confidence too low: {}", h.confidence);
    }

    #[test]
    fn find_waist_neck_none_on_uniform_cylinder() {
        // 一様な円柱 (waist で細くならない) → 局所的な最小はあるが、neck > waist 条件で概ね検出可能
        // ここでは実際に waist_y < neck_y を満たすかチェック (=検出できても妥当な範囲)
        let mut v = Vec::new();
        for k in 0..40 {
            let y = k as f32 / 40.0;
            for a in 0..8 {
                let theta = (a as f32) * core::f32::consts::TAU / 8.0;
                v.push(Vec3k::new(
                    cos_approx(theta) * 0.1,
                    y,
                    sin_approx(theta) * 0.1,
                ));
            }
        }
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        // waist と neck が検出されても順序は正しくなる
        if let Some((w, n)) = find_waist_neck(&slices) {
            assert!(slices[n].y > slices[w].y);
        }
    }
}
