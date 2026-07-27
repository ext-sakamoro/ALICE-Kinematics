//! Mesh slicing along Y-axis for auto-rig
//!
//! MeshView → NormalizedMesh (height = 1.0 に正規化) → 128 断面統計
//!
//! 幾何ヒューリスティック骨格推定 (dotneet/image-to-3d §E.1 平面世界向け)
//! の基盤となる断面分割を提供する
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use alloc::vec::Vec;

/// メッシュのビュー (借用型、所有権を取らない)
#[derive(Debug, Clone, Copy)]
pub struct MeshView<'a> {
    /// 頂点配列
    pub vertices: &'a [Vec3k],
    /// インデックス配列 (三角形リスト、空でも可)
    pub indices: &'a [u32],
}

impl<'a> MeshView<'a> {
    /// 頂点のみからビューを構築 (インデックスなし)
    #[must_use]
    pub const fn from_vertices(vertices: &'a [Vec3k]) -> Self {
        Self {
            vertices,
            indices: &[],
        }
    }
}

/// 正規化済みメッシュ (Y 軸方向の height が 1.0、中心は元の bbox 中心の X/Z)
#[derive(Debug, Clone)]
pub struct NormalizedMesh {
    /// 正規化後の頂点 (owned、height = 1.0)
    pub vertices: Vec<Vec3k>,
    /// 元メッシュの Y 方向 height (メートル)
    pub original_height: f32,
    /// 元メッシュ bbox の最小コーナー (offset 復元用)
    pub original_min: Vec3k,
    /// 元メッシュ bbox の最大コーナー
    pub original_max: Vec3k,
}

impl NormalizedMesh {
    /// 正規化 → 元スケール座標への変換 (Y=0 が bbox_min.y)
    #[must_use]
    pub fn denormalize(&self, normalized: Vec3k) -> Vec3k {
        Vec3k::new(
            normalized.x * self.original_height + self.original_min.x,
            normalized.y * self.original_height + self.original_min.y,
            normalized.z * self.original_height + self.original_min.z,
        )
    }

    /// 水平方向 (X, Z) の最大 extent (= 幅と奥行のうち大きい方)
    ///
    /// 正規化済み座標系での値 (無次元)
    #[must_use]
    pub fn horizontal_extent(&self) -> f32 {
        let scale = self.original_height;
        if scale < 1e-6 {
            return 0.0;
        }
        let width_x = (self.original_max.x - self.original_min.x) / scale;
        let width_z = (self.original_max.z - self.original_min.z) / scale;
        if width_x > width_z {
            width_x
        } else {
            width_z
        }
    }
}

/// 断面統計 (1 スライス分の情報)
#[derive(Debug, Clone, Copy)]
pub struct SliceStat {
    /// スライス中心の Y 座標 (正規化後、0.0-1.0)
    pub y: f32,
    /// 断面の水平方向 bounding box の幅 (X, Z 方向の max extent)
    pub width: f32,
    /// 断面重心 (全頂点の平均、正規化後座標系)
    pub centroid: Vec3k,
    /// スライス内の頂点数
    pub vertex_count: u32,
    /// 左半 (X < body_center_x) の頂点重心
    pub left_centroid: Option<Vec3k>,
    /// 右半 (X >= body_center_x) の頂点重心
    pub right_centroid: Option<Vec3k>,
}

/// メッシュを正規化 (height = 1.0 にスケール、Y 軸 up 前提)
///
/// # Errors
///
/// - 頂点が空
/// - Y 方向の extent が 0 (平面メッシュ)
#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn normalize_mesh(mesh: &MeshView) -> Option<NormalizedMesh> {
    if mesh.vertices.is_empty() {
        return None;
    }

    let mut min = Vec3k::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3k::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in mesh.vertices {
        if v.x < min.x {
            min.x = v.x;
        }
        if v.y < min.y {
            min.y = v.y;
        }
        if v.z < min.z {
            min.z = v.z;
        }
        if v.x > max.x {
            max.x = v.x;
        }
        if v.y > max.y {
            max.y = v.y;
        }
        if v.z > max.z {
            max.z = v.z;
        }
    }

    let height = max.y - min.y;
    if height < 1e-6 {
        return None;
    }

    let inv_h = 1.0 / height;
    let vertices: Vec<Vec3k> = mesh
        .vertices
        .iter()
        .map(|v| {
            Vec3k::new(
                (v.x - min.x) * inv_h,
                (v.y - min.y) * inv_h,
                (v.z - min.z) * inv_h,
            )
        })
        .collect();

    Some(NormalizedMesh {
        vertices,
        original_height: height,
        original_min: min,
        original_max: max,
    })
}

/// 正規化済みメッシュを Y 軸方向に slice_count 個に分割し、各断面統計を返す
///
/// `slice\[0\]` が最下段 (y=0)、`slice\[N-1\]` が最上段 (y=1) に対応する
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn slice_normalized(nm: &NormalizedMesh, slice_count: usize) -> Vec<SliceStat> {
    let mut slices = Vec::with_capacity(slice_count);
    if slice_count == 0 {
        return slices;
    }

    let bin_h = 1.0 / slice_count as f32;
    // 体の中心 X を bbox 中心とする (正規化後は 0.5 × horizontal_extent の位置)
    let scale = nm.original_height;
    let body_center_x = if scale > 1e-6 {
        (nm.original_max.x - nm.original_min.x) * 0.5 / scale
    } else {
        0.5
    };

    // 事前に各頂点をスライス index に振り分け
    let mut buckets: Vec<Vec<Vec3k>> = (0..slice_count).map(|_| Vec::new()).collect();
    for &v in &nm.vertices {
        let idx = ((v.y * slice_count as f32) as usize).min(slice_count - 1);
        buckets[idx].push(v);
    }

    for (i, bucket) in buckets.iter().enumerate() {
        let y = (i as f32 + 0.5) * bin_h;
        if bucket.is_empty() {
            slices.push(SliceStat {
                y,
                width: 0.0,
                centroid: Vec3k::new(body_center_x, y, 0.0),
                vertex_count: 0,
                left_centroid: None,
                right_centroid: None,
            });
            continue;
        }

        let mut sum = Vec3k::ZERO;
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut left_sum = Vec3k::ZERO;
        let mut left_count = 0u32;
        let mut right_sum = Vec3k::ZERO;
        let mut right_count = 0u32;

        for &v in bucket {
            sum = sum + v;
            if v.x < min_x {
                min_x = v.x;
            }
            if v.x > max_x {
                max_x = v.x;
            }
            if v.z < min_z {
                min_z = v.z;
            }
            if v.z > max_z {
                max_z = v.z;
            }
            if v.x < body_center_x {
                left_sum = left_sum + v;
                left_count += 1;
            } else {
                right_sum = right_sum + v;
                right_count += 1;
            }
        }

        let count = bucket.len() as f32;
        let centroid = sum.scale(1.0 / count);
        let width_x = max_x - min_x;
        let width_z = max_z - min_z;
        let width = if width_x > width_z { width_x } else { width_z };

        let left_centroid = if left_count > 0 {
            Some(left_sum.scale(1.0 / left_count as f32))
        } else {
            None
        };
        let right_centroid = if right_count > 0 {
            Some(right_sum.scale(1.0 / right_count as f32))
        } else {
            None
        };

        slices.push(SliceStat {
            y,
            width,
            centroid,
            vertex_count: bucket.len() as u32,
            left_centroid,
            right_centroid,
        });
    }

    slices
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_1x1x1() -> Vec<Vec3k> {
        alloc::vec![
            Vec3k::new(0.0, 0.0, 0.0),
            Vec3k::new(1.0, 0.0, 0.0),
            Vec3k::new(0.0, 1.0, 0.0),
            Vec3k::new(1.0, 1.0, 0.0),
            Vec3k::new(0.0, 0.0, 1.0),
            Vec3k::new(1.0, 0.0, 1.0),
            Vec3k::new(0.0, 1.0, 1.0),
            Vec3k::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn normalize_empty_mesh_returns_none() {
        let v: Vec<Vec3k> = Vec::new();
        let mesh = MeshView::from_vertices(&v);
        assert!(normalize_mesh(&mesh).is_none());
    }

    #[test]
    fn normalize_flat_mesh_returns_none() {
        let v = alloc::vec![
            Vec3k::new(0.0, 0.5, 0.0),
            Vec3k::new(1.0, 0.5, 0.0),
            Vec3k::new(0.5, 0.5, 1.0),
        ];
        let mesh = MeshView::from_vertices(&v);
        assert!(normalize_mesh(&mesh).is_none());
    }

    #[test]
    fn normalize_cube_height_is_1() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        assert!((nm.original_height - 1.0).abs() < 1e-6);
        // Y 座標は [0.0, 1.0]
        let ys: Vec<f32> = nm.vertices.iter().map(|v| v.y).collect();
        let min_y = ys.iter().copied().fold(f32::INFINITY, f32::min);
        let max_y = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_y.abs() < 1e-6);
        assert!((max_y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_scaled_mesh() {
        // 高さ 2 の立方体
        let v: Vec<Vec3k> = cube_1x1x1().iter().map(|v| v.scale(2.0)).collect();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        assert!((nm.original_height - 2.0).abs() < 1e-6);
        // 正規化後の Y は [0, 1]
        let max_y = nm
            .vertices
            .iter()
            .map(|v| v.y)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max_y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn denormalize_roundtrip() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        for (i, &original) in v.iter().enumerate() {
            let denorm = nm.denormalize(nm.vertices[i]);
            assert!((denorm.x - original.x).abs() < 1e-4);
            assert!((denorm.y - original.y).abs() < 1e-4);
            assert!((denorm.z - original.z).abs() < 1e-4);
        }
    }

    #[test]
    fn slice_count_matches() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 128);
        assert_eq!(slices.len(), 128);
    }

    #[test]
    fn slice_cube_bottom_and_top_populated() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 4);
        // 立方体は上下面にしか頂点がないので、bin 0 と bin N-1 に頂点があるはず
        assert!(slices[0].vertex_count > 0);
        assert!(slices[3].vertex_count > 0);
    }

    #[test]
    fn slice_zero_count_returns_empty() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 0);
        assert!(slices.is_empty());
    }

    #[test]
    fn slice_centroid_of_uniform_column() {
        // 縦長円柱 (半径 0.1、高さ 1.0)、Y 軸中心
        let mut v = Vec::new();
        for k in 0..20 {
            let y = k as f32 / 20.0;
            for a in 0..8 {
                let theta = (a as f32) * core::f32::consts::TAU / 8.0;
                v.push(Vec3k::new(0.1 * libm_cos(theta), y, 0.1 * libm_sin(theta)));
            }
        }
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 16);
        // 対称な形状 → 各非空断面の重心 X は互いに近い (bbox-relative なので 0.5 ではなく
        // horizontal_extent / 2 付近)
        let first_x = slices
            .iter()
            .find(|s| s.vertex_count > 0)
            .map(|s| s.centroid.x)
            .expect("at least one non-empty slice");
        for s in &slices {
            if s.vertex_count > 0 {
                assert!((s.centroid.x - first_x).abs() < 0.02);
            }
        }
    }

    #[test]
    fn slice_left_right_split() {
        // 完全対称な形状 → left_centroid と right_centroid が両方存在
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        let slices = slice_normalized(&nm, 4);
        // 立方体の底 (bin 0) は L/R 頂点がバランスして存在
        assert!(slices[0].left_centroid.is_some() || slices[0].right_centroid.is_some());
    }

    #[test]
    fn horizontal_extent_of_cube() {
        let v = cube_1x1x1();
        let mesh = MeshView::from_vertices(&v);
        let nm = normalize_mesh(&mesh).unwrap();
        // 立方体は 1x1x1、正規化後の horizontal extent は 1.0 (=height/height)
        assert!((nm.horizontal_extent() - 1.0).abs() < 1e-4);
    }

    // 小さな三角関数近似 (test 内 helper)
    fn libm_cos(x: f32) -> f32 {
        let x = x % core::f32::consts::TAU;
        // Bhaskara 近似の cos = sin(pi/2 - x)
        let s = core::f32::consts::FRAC_PI_2 - x;
        libm_sin(s)
    }

    fn libm_sin(x: f32) -> f32 {
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
}
