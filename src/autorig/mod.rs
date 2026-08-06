//! Auto-rig: 幾何ヒューリスティックによる立位ヒューマノイド mesh → 22-joint skeleton 推定
//!
//! CUDA / 学習不要の Y-up 平面世界向け実装 dotneet/image-to-3d §E.1 準拠
//!
//! アルゴリズム:
//! 1. mesh を height = 1.0 に正規化
//! 2. aspect ratio (height / max horizontal) で humanoid 判定
//! 3. Y 軸方向 128 スライスで各断面 (幅 + 重心 + L/R 半分の重心) 計算
//! 4. 幅プロファイルの local minima から waist / neck 検出
//! 5. spine chain → legs → arms の順に joint 位置推定
//! 6. distance-based skinning (top-4, 影響半径可変) で weight 生成
//!
//! # 制約 (memory §E.0)
//!
//! - 立位ヒューマノイド専用 (四足動物 / 機械 / 極端デフォルメ / 腕胴密着ポーズは NG)
//! - 指のリグは作らない
//! - Y-up 座標系前提 (Z-up / 球面世界向けは別 module で対応予定)
//!
//! # Quick Start
//!
//! ```
//! use alice_kinematics::autorig::{auto_rig, AutoRigConfig, MeshView};
//! use alice_kinematics::Vec3k;
//!
//! // 立方体は not humanoid
//! let cube = [
//!     Vec3k::new(0.0, 0.0, 0.0), Vec3k::new(1.0, 0.0, 0.0),
//!     Vec3k::new(0.0, 1.0, 0.0), Vec3k::new(1.0, 1.0, 0.0),
//!     Vec3k::new(0.0, 0.0, 1.0), Vec3k::new(1.0, 0.0, 1.0),
//!     Vec3k::new(0.0, 1.0, 1.0), Vec3k::new(1.0, 1.0, 1.0),
//! ];
//! let mesh = MeshView::from_vertices(&cube);
//! let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
//! assert!(result.skeleton.is_none());  // aspect ratio 1.0 → not humanoid
//! ```
//!
//! License: MIT
//! Author: Moroya Sakamoto

// Pedantic warning 抑制 (2026-08-06 追加、autorig module 特有事情):
// - doc_markdown: 日本語 doc 内の英数字識別子 (config.is_humanoid_min_aspect 等) に
//   backtick を全部付けると可読性低下、config field ばかりのため一括 allow
// - cast_precision_loss / cast_possible_truncation: 幾何演算で
//   i32 (slice index) → f32 (座標) 変換が頻出 mesh vertex 数 usize → u16 も同様
//   全て意図的な cast、精度損失は許容範囲 (最大 mesh vertex 数 << u16::MAX 相当)
// - manual_let_else: `match Some/None → let-else` refactor は 好み、
//   本 module では既存の match style を維持
#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::manual_let_else
)]

pub mod hypothesis;
pub mod skinning;
pub mod slice;
pub mod sphere;

pub use hypothesis::{build_hypothesis, hypothesis_to_skeleton, SkeletonHypothesis};
pub use skinning::{
    distance_based_skinning, point_to_segment_distance_sq, InfluenceRadii, Skinning,
};
pub use slice::{normalize_mesh, slice_normalized, MeshView, NormalizedMesh, SliceStat};
pub use sphere::{auto_rig_on_sphere, SphereContext, TangentFrame};

use crate::skeleton::Skeleton;

/// Auto-rig の設定
#[derive(Debug, Clone, Copy)]
pub struct AutoRigConfig {
    /// Y 軸方向のスライス数 (dotneet/image-to-3d は 128 を採用)
    pub slice_count: usize,
    /// humanoid 判定の aspect ratio 下限 (height / max horizontal)
    ///
    /// 車のように aspect < 1.5 の形状は not humanoid と判定される
    pub is_humanoid_min_aspect: f32,
    /// スキニングの影響半径 (height-relative)
    pub influence_radii: InfluenceRadii,
}

impl Default for AutoRigConfig {
    fn default() -> Self {
        Self {
            slice_count: 128,
            // T-pose (腕を水平に伸ばした姿勢) だと aspect が 1.4-1.5 まで下がるので
            // 1.2 を採用 立方体 (1.0) や車 (< 1.0) は確実に reject される
            is_humanoid_min_aspect: 1.2,
            influence_radii: InfluenceRadii::default(),
        }
    }
}

/// Auto-rig の結果
///
/// - `skeleton`: 検出された骨格 (not humanoid の場合は `None`)
/// - `skinning`: 距離ベーススキニング (skeleton が `None` の場合は `None`)
/// - `confidence`: 検出信頼度 (0.0-1.0、検出された関節数 / BONE_COUNT)
#[derive(Debug, Clone)]
pub struct AutoRigResult {
    /// 推定された 22-joint skeleton
    pub skeleton: Option<Skeleton>,
    /// 頂点ごとの top-4 bone weight
    pub skinning: Option<Skinning>,
    /// 検出信頼度 (0.0-1.0)
    pub confidence: f32,
}

/// Auto-rig 実行時のエラー
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoRigError {
    /// 頂点が空
    EmptyMesh,
    /// slice_count が 0
    InvalidSliceCount,
    /// Y 方向の extent が 0 (平面メッシュ)
    NoVerticalExtent,
}

/// mesh から skeleton + skinning を推定
///
/// # Errors
///
/// - `EmptyMesh`: 頂点が空
/// - `InvalidSliceCount`: config の slice_count が 0
/// - `NoVerticalExtent`: メッシュに Y 方向の高さがない (平面)
///
/// # Not-humanoid
///
/// aspect ratio が config.is_humanoid_min_aspect 未満の場合、
/// `Ok(AutoRigResult { skeleton: None, skinning: None, confidence: 0.0 })` を返す
/// (エラーではなく「対応不可」として明示、memory §E.0 準拠)
pub fn auto_rig(mesh: &MeshView, config: &AutoRigConfig) -> Result<AutoRigResult, AutoRigError> {
    if mesh.vertices.is_empty() {
        return Err(AutoRigError::EmptyMesh);
    }
    if config.slice_count == 0 {
        return Err(AutoRigError::InvalidSliceCount);
    }

    // Step 1: normalize
    let nm = normalize_mesh(mesh).ok_or(AutoRigError::NoVerticalExtent)?;

    // Step 2: aspect ratio で humanoid 判定
    let horizontal = nm.horizontal_extent();
    let aspect = if horizontal > 1e-6 {
        1.0 / horizontal
    } else {
        f32::INFINITY
    };
    if aspect < config.is_humanoid_min_aspect {
        return Ok(AutoRigResult {
            skeleton: None,
            skinning: None,
            confidence: 0.0,
        });
    }

    // Step 3: slice
    let slices = slice_normalized(&nm, config.slice_count);

    // Step 4-7: hypothesis + spine + legs + arms
    let hypothesis = match build_hypothesis(&nm, &slices) {
        Some(h) => h,
        None => {
            return Ok(AutoRigResult {
                skeleton: None,
                skinning: None,
                confidence: 0.0,
            });
        }
    };
    let confidence = hypothesis.confidence;

    // Step 8: skeleton
    let skeleton = hypothesis_to_skeleton(&hypothesis, &nm);

    // Step 9: skinning
    let skinning = distance_based_skinning(mesh, &skeleton, &config.influence_radii);

    Ok(AutoRigResult {
        skeleton: Some(skeleton),
        skinning: Some(skinning),
        confidence,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::Vec3k;
    use alloc::vec::Vec;

    #[test]
    fn config_defaults() {
        let c = AutoRigConfig::default();
        assert_eq!(c.slice_count, 128);
        assert!((c.is_humanoid_min_aspect - 1.2).abs() < 1e-6);
    }

    #[test]
    fn auto_rig_empty_mesh_errors() {
        let v: Vec<Vec3k> = Vec::new();
        let mesh = MeshView::from_vertices(&v);
        let err = auto_rig(&mesh, &AutoRigConfig::default()).unwrap_err();
        assert_eq!(err, AutoRigError::EmptyMesh);
    }

    #[test]
    fn auto_rig_invalid_slice_count_errors() {
        let v = alloc::vec![Vec3k::new(0.0, 0.0, 0.0), Vec3k::new(0.0, 1.0, 0.0)];
        let mesh = MeshView::from_vertices(&v);
        let config = AutoRigConfig {
            slice_count: 0,
            ..AutoRigConfig::default()
        };
        let err = auto_rig(&mesh, &config).unwrap_err();
        assert_eq!(err, AutoRigError::InvalidSliceCount);
    }

    #[test]
    fn auto_rig_flat_mesh_errors() {
        // Y 方向 extent = 0
        let v = alloc::vec![
            Vec3k::new(0.0, 0.5, 0.0),
            Vec3k::new(1.0, 0.5, 0.0),
            Vec3k::new(0.5, 0.5, 1.0),
        ];
        let mesh = MeshView::from_vertices(&v);
        let err = auto_rig(&mesh, &AutoRigConfig::default()).unwrap_err();
        assert_eq!(err, AutoRigError::NoVerticalExtent);
    }

    #[test]
    fn auto_rig_cube_not_humanoid() {
        let v = alloc::vec![
            Vec3k::new(0.0, 0.0, 0.0),
            Vec3k::new(1.0, 0.0, 0.0),
            Vec3k::new(0.0, 1.0, 0.0),
            Vec3k::new(1.0, 1.0, 0.0),
            Vec3k::new(0.0, 0.0, 1.0),
            Vec3k::new(1.0, 0.0, 1.0),
            Vec3k::new(0.0, 1.0, 1.0),
            Vec3k::new(1.0, 1.0, 1.0),
        ];
        let mesh = MeshView::from_vertices(&v);
        let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
        assert!(result.skeleton.is_none());
        assert!(result.skinning.is_none());
        assert!(result.confidence.abs() < 1e-6);
    }
}
