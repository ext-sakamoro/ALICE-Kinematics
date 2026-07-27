//! Distance-based skinning (dotneet/image-to-3d §E.3 fallback c 距離ベース自前実装)
//!
//! 各頂点について全 bone segment との距離を計算、top-4 を採用して `1/(d² + ε)` で
//! 重みづけし正規化する 影響半径をボーン種別で分けることで、腕が近くの布 (スカート等)
//! を巻き込む問題を回避する (memory §E.4 準拠)
//!
//! 影響半径 (身長 1.0 に対する比率):
//! - 手足: 0.03 - 0.05 (デフォルト max = 0.05)
//! - 胴: 0.12 - 0.18 (デフォルト max = 0.18)
//!
//! Blender heat weighting (§E.3 fallback a) は水密メッシュ前提でモデル依存が強く、
//! Rust ネイティブ環境では実装保留、Blender を使わない前提ならこの距離ベースが
//! 実質デフォルト (memory §E.3 「実測でほぼ (c) 距離ベース」)
//!
//! License: MIT
//! Author: Moroya Sakamoto

use super::slice::MeshView;
use crate::joint::Vec3k;
use crate::skeleton::{BoneId, Skeleton};
use alloc::vec::Vec;

/// スキニング結果 (頂点数 × 4 の bone index + weight)
///
/// 重みは top-4 の合計が 1.0 (孤立頂点で全 bone が影響範囲外の場合は Hips に 100%)
#[derive(Debug, Clone)]
pub struct Skinning {
    /// 頂点 i の影響 bone インデックス (skeleton.joints への index) top-4
    pub bone_indices: Vec<[u16; 4]>,
    /// 頂点 i の bone i への重み (top-4、Σ = 1.0)
    pub weights: Vec<[f32; 4]>,
}

/// 影響半径 (身長 1.0 に対する比率)
///
/// 実際の距離判定では `radius × skeleton.height` を使う (real-scale)
#[derive(Debug, Clone, Copy)]
pub struct InfluenceRadii {
    /// 手足 (arm / leg) の最大影響半径 (height-relative)
    pub limb_max: f32,
    /// 胴 (torso / head / shoulder) の最大影響半径 (height-relative)
    pub torso_max: f32,
}

impl Default for InfluenceRadii {
    /// dotneet/image-to-3d §E.4 実測値 (手足 0.03-0.05 / 胴 0.12-0.18)
    /// 最大値を採用 (最小値だと届かないボーンが増える)
    fn default() -> Self {
        Self {
            limb_max: 0.05,
            torso_max: 0.18,
        }
    }
}

/// 手足ボーンか判定 (bone-specific radius 判定用)
#[must_use]
fn is_limb(bone_id: BoneId) -> bool {
    matches!(
        bone_id,
        BoneId::LeftUpperArm
            | BoneId::LeftLowerArm
            | BoneId::LeftHand
            | BoneId::RightUpperArm
            | BoneId::RightLowerArm
            | BoneId::RightHand
            | BoneId::LeftUpperLeg
            | BoneId::LeftLowerLeg
            | BoneId::LeftFoot
            | BoneId::LeftToe
            | BoneId::RightUpperLeg
            | BoneId::RightLowerLeg
            | BoneId::RightFoot
            | BoneId::RightToe
    )
}

/// 点 p から線分 (a, b) への距離の 2 乗
#[must_use]
pub fn point_to_segment_distance_sq(p: Vec3k, a: Vec3k, b: Vec3k) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let denom = ab.length_sq();
    if denom < 1e-10 {
        // 縮退線分 (a == b) は点との距離
        return ap.length_sq();
    }
    let t = (ab.dot(ap) / denom).clamp(0.0, 1.0);
    let closest = a + ab.scale(t);
    (p - closest).length_sq()
}

/// 距離ベーススキニングを計算
///
/// 各頂点について:
/// 1. 全 bone segment (parent.world → self.world) との距離を計算
/// 2. bone 種別ごとの影響半径 (limb / torso) 内の bone のみ候補に残す
/// 3. top-4 の bone を距離順に選択
/// 4. 重み = 1 / (d² + ε)、正規化して合計を 1.0 に
///
/// 全 bone が影響半径外の孤立頂点は Hips (index 0) に 100% 割り当てる
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn distance_based_skinning(
    mesh: &MeshView,
    skeleton: &Skeleton,
    radii: &InfluenceRadii,
) -> Skinning {
    let height = skeleton.height;
    // powi(2) は std 依存、no_std のため x * x に展開
    let limb_max = radii.limb_max * height;
    let torso_max = radii.torso_max * height;
    let limb_max_sq = limb_max * limb_max;
    let torso_max_sq = torso_max * torso_max;

    let n = mesh.vertices.len();
    let mut bone_indices: Vec<[u16; 4]> = Vec::with_capacity(n);
    let mut weights: Vec<[f32; 4]> = Vec::with_capacity(n);

    let eps = 1e-6_f32;

    for &v in mesh.vertices {
        // 各 bone との距離を計算し、影響半径内のものを候補に
        let mut dists: Vec<(f32, u16)> = Vec::with_capacity(skeleton.joints.len());
        for (i, joint) in skeleton.joints.iter().enumerate() {
            let dist_sq = if let Some(parent_idx) = joint.parent {
                let a = skeleton.joints[parent_idx].world_position;
                let b = joint.world_position;
                point_to_segment_distance_sq(v, a, b)
            } else {
                // ルート (Hips) は点距離
                (v - joint.world_position).length_sq()
            };
            let max_sq = if is_limb(joint.id) {
                limb_max_sq
            } else {
                torso_max_sq
            };
            if dist_sq > max_sq {
                continue;
            }
            dists.push((dist_sq, i as u16));
        }

        // 距離昇順ソート、top-4 を取る
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        dists.truncate(4);

        // 4 に満たない場合は sentinel で埋める (weight 0 で影響なし)
        while dists.len() < 4 {
            dists.push((f32::MAX, 0));
        }

        // 重み計算 (1 / (d² + ε))、sentinel は 0
        let raw: [f32; 4] = [
            if dists[0].0 < f32::MAX {
                1.0 / (dists[0].0 + eps)
            } else {
                0.0
            },
            if dists[1].0 < f32::MAX {
                1.0 / (dists[1].0 + eps)
            } else {
                0.0
            },
            if dists[2].0 < f32::MAX {
                1.0 / (dists[2].0 + eps)
            } else {
                0.0
            },
            if dists[3].0 < f32::MAX {
                1.0 / (dists[3].0 + eps)
            } else {
                0.0
            },
        ];
        let sum: f32 = raw.iter().sum();
        let normalized: [f32; 4] = if sum > 0.0 {
            [raw[0] / sum, raw[1] / sum, raw[2] / sum, raw[3] / sum]
        } else {
            // 影響半径外の孤立頂点 → Hips に 100%
            [1.0, 0.0, 0.0, 0.0]
        };

        let indices: [u16; 4] = [dists[0].1, dists[1].1, dists[2].1, dists[3].1];
        bone_indices.push(indices);
        weights.push(normalized);
    }

    Skinning {
        bone_indices,
        weights,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Skeleton;

    #[test]
    fn segment_distance_sq_endpoint() {
        let a = Vec3k::new(0.0, 0.0, 0.0);
        let b = Vec3k::new(1.0, 0.0, 0.0);
        // 線分の端点 a に一致する点
        let d = point_to_segment_distance_sq(a, a, b);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn segment_distance_sq_perpendicular() {
        let a = Vec3k::new(0.0, 0.0, 0.0);
        let b = Vec3k::new(1.0, 0.0, 0.0);
        // 線分に垂直方向、中央から 0.5 離れた点 → 距離² = 0.25
        let p = Vec3k::new(0.5, 0.5, 0.0);
        let d = point_to_segment_distance_sq(p, a, b);
        assert!((d - 0.25).abs() < 1e-4);
    }

    #[test]
    fn segment_distance_sq_beyond_end() {
        let a = Vec3k::new(0.0, 0.0, 0.0);
        let b = Vec3k::new(1.0, 0.0, 0.0);
        // 線分の外側 (b の先)、b から (1.0, 0.5, 0) → 距離² = 1.0² + 0.5² = 1.25
        let p = Vec3k::new(2.0, 0.5, 0.0);
        let d = point_to_segment_distance_sq(p, a, b);
        assert!((d - 1.25).abs() < 1e-4);
    }

    #[test]
    fn segment_distance_sq_degenerate_segment() {
        // a == b → 点距離
        let a = Vec3k::new(1.0, 1.0, 1.0);
        let b = a;
        let p = Vec3k::new(2.0, 1.0, 1.0);
        let d = point_to_segment_distance_sq(p, a, b);
        assert!((d - 1.0).abs() < 1e-4);
    }

    #[test]
    fn is_limb_hand() {
        assert!(is_limb(BoneId::LeftHand));
        assert!(is_limb(BoneId::RightHand));
    }

    #[test]
    fn is_limb_hips() {
        assert!(!is_limb(BoneId::Hips));
        assert!(!is_limb(BoneId::Chest));
        assert!(!is_limb(BoneId::Head));
    }

    #[test]
    fn skinning_weights_sum_to_one() {
        // 標準骨格の周辺に頂点を配置してスキニング
        let skel = Skeleton::default_humanoid();
        let verts = alloc::vec![
            Vec3k::new(0.0, 1.0, 0.0),   // 中央 (胸付近)
            Vec3k::new(0.2, 1.2, 0.0),   // 肩付近
            Vec3k::new(-0.15, 0.5, 0.0), // 腰左
            Vec3k::new(0.0, 0.1, 0.0),   // 足元中央
        ];
        let mesh = MeshView::from_vertices(&verts);
        let radii = InfluenceRadii::default();
        let sk = distance_based_skinning(&mesh, &skel, &radii);
        assert_eq!(sk.weights.len(), verts.len());
        for (i, w) in sk.weights.iter().enumerate() {
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "vertex {i} weight sum = {sum}, indices = {:?}",
                sk.bone_indices[i]
            );
        }
    }

    #[test]
    fn skinning_outputs_length_matches_vertices() {
        let skel = Skeleton::default_humanoid();
        let verts = alloc::vec![Vec3k::new(0.0, 1.0, 0.0), Vec3k::new(0.1, 0.5, 0.0)];
        let mesh = MeshView::from_vertices(&verts);
        let sk = distance_based_skinning(&mesh, &skel, &InfluenceRadii::default());
        assert_eq!(sk.bone_indices.len(), 2);
        assert_eq!(sk.weights.len(), 2);
    }

    #[test]
    fn skinning_empty_mesh() {
        let skel = Skeleton::default_humanoid();
        let verts: Vec<Vec3k> = Vec::new();
        let mesh = MeshView::from_vertices(&verts);
        let sk = distance_based_skinning(&mesh, &skel, &InfluenceRadii::default());
        assert!(sk.bone_indices.is_empty());
        assert!(sk.weights.is_empty());
    }

    #[test]
    fn skinning_isolated_vertex_falls_back_to_hips() {
        // 骨格から遠く離れた頂点 → 全 bone が影響半径外 → Hips (index 0) に 100%
        let skel = Skeleton::default_humanoid();
        let verts = alloc::vec![Vec3k::new(100.0, 100.0, 100.0)];
        let mesh = MeshView::from_vertices(&verts);
        let sk = distance_based_skinning(&mesh, &skel, &InfluenceRadii::default());
        assert_eq!(sk.bone_indices[0][0], 0); // Hips index
        assert!((sk.weights[0][0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn skinning_indices_are_valid() {
        let skel = Skeleton::default_humanoid();
        let verts = alloc::vec![
            Vec3k::new(0.0, 1.0, 0.0),
            Vec3k::new(0.1, 0.5, 0.0),
            Vec3k::new(0.2, 1.5, 0.0),
        ];
        let mesh = MeshView::from_vertices(&verts);
        let sk = distance_based_skinning(&mesh, &skel, &InfluenceRadii::default());
        let joint_count = skel.joints.len() as u16;
        for indices in &sk.bone_indices {
            for &i in indices {
                assert!(
                    i < joint_count,
                    "bone index {i} out of range (joint count = {joint_count})"
                );
            }
        }
    }
}
