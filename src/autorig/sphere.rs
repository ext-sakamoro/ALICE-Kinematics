//! Sphere world auto-rig — 球面世界 (惑星上) 向け薄いラッパー
//!
//! §E.1 平面世界 auto-rig を、球面上のキャラクタに適用するための座標変換ラッパー
//!
//! # 手順
//!
//! 1. キャラクタ位置 `p_world` と球中心 `sphere_center` から
//!    **local tangent frame** (up, forward, right) を計算
//!    - up = normalize(p_world - sphere_center)
//!    - forward = arbitrary tangent (world Y に近い方向を採用、colinear なら world Z)
//!    - right = up × forward (再正規化)
//! 2. mesh 頂点を local frame に変換 (up が local Y = up 方向)
//! 3. 既存 §E.1 `auto_rig()` を local mesh に適用
//! 4. 生成された skeleton の world position / local position を、
//!    inverse tangent frame で world に戻す
//!
//! # 決定性 (Fix128 lockstep 用途) について
//!
//! 本モジュールは f32 (`Vec3k`) で実装される 決定性が必要な場合は
//! `tangent_frame()` を Fix128 上で計算 (= `alice-physics` 側の `Fix128` 版)、
//! transform 済 mesh を渡して `auto_rig()` を呼ぶ運用が必要
//! この場合の互換 wrapper は将来 v0.2.0+ で検討 (現状 v0.1.x は f32 のみ)
//!
//! # Quick Start
//!
//! ```
//! use alice_kinematics::autorig::sphere::{auto_rig_on_sphere, SphereContext};
//! use alice_kinematics::autorig::{AutoRigConfig, MeshView};
//! use alice_kinematics::Vec3k;
//!
//! // 半径 300 の球中心 (0,0,0)、キャラクタ位置は北極付近 (0, 300, 0) 相当
//! let ctx = SphereContext {
//!     center: Vec3k::ZERO,
//!     character_pos: Vec3k::new(0.0, 300.0, 0.0),
//!     radius: 300.0,
//! };
//! let vertices = [Vec3k::new(0.0, 300.0, 0.0), Vec3k::new(0.0, 301.7, 0.0)];
//! let mesh = MeshView::from_vertices(&vertices);
//! let _result = auto_rig_on_sphere(&mesh, &ctx, &AutoRigConfig::default());
//! ```
//!
//! License: MIT
//! Author: Moroya Sakamoto

use super::{auto_rig, AutoRigConfig, AutoRigError, AutoRigResult, MeshView};
use crate::joint::Vec3k;
use crate::skeleton::{Skeleton, SkeletonJoint};
use alloc::vec::Vec;

/// 球面世界における座標コンテキスト
#[derive(Debug, Clone, Copy)]
pub struct SphereContext {
    /// 球中心 (world 座標)
    pub center: Vec3k,
    /// キャラクタ位置 (world 座標、球面上 or 近傍)
    pub character_pos: Vec3k,
    /// 球半径 (m)
    pub radius: f32,
}

/// キャラクタ位置における tangent frame (right, up, forward)
///
/// - `up`: 球中心 → キャラクタ方向 (半径方向、単位ベクトル)
/// - `forward`: tangent 平面上の 1 方向 (world Y を投影して採用、colinear なら world Z 投影)
/// - `right`: `up × forward` (直交補完、単位ベクトル)
///
/// 3 ベクトルは正規直交基底を成す
#[derive(Debug, Clone, Copy)]
pub struct TangentFrame {
    /// tangent 平面の X 軸 (単位ベクトル、world 空間)
    pub right: Vec3k,
    /// 球面法線 = local Y 軸 (単位ベクトル、world 空間)
    pub up: Vec3k,
    /// tangent 平面の Z 軸 (単位ベクトル、world 空間)
    pub forward: Vec3k,
    /// キャラクタ位置 (world 空間、frame 原点)
    pub origin: Vec3k,
}

impl TangentFrame {
    /// SphereContext から tangent frame を計算
    ///
    /// `character_pos == center` の場合は panic せず、up = world Y の fallback を返す
    /// (球中心にキャラは実質ありえないが、退化ケースの安全性のため)
    #[must_use]
    pub fn from_context(ctx: &SphereContext) -> Self {
        let radial = ctx.character_pos - ctx.center;
        let radial_len_sq = radial.length_sq();
        let up = if radial_len_sq < 1e-10 {
            Vec3k::new(0.0, 1.0, 0.0)
        } else {
            radial.scale(1.0 / radial_len_sq.sqrt())
        };
        // forward: world Y を tangent 平面に投影 (up と colinear なら world Z を使う)
        let world_y = Vec3k::new(0.0, 1.0, 0.0);
        let dot = up.dot(world_y);
        let base = if dot.abs() > 0.999 {
            Vec3k::new(0.0, 0.0, 1.0)
        } else {
            world_y
        };
        // Gram-Schmidt: forward = normalize(base - (up . base) * up)
        let proj = up.scale(up.dot(base));
        let forward_raw = base - proj;
        let forward = forward_raw.normalize();
        // right = up × forward、右手系で整合
        let right = up.cross(forward).normalize();
        Self {
            right,
            up,
            forward,
            origin: ctx.character_pos,
        }
    }

    /// world 座標を tangent local (origin=0, Y=up) に変換
    #[must_use]
    pub fn world_to_local(&self, world: Vec3k) -> Vec3k {
        let rel = world - self.origin;
        Vec3k::new(rel.dot(self.right), rel.dot(self.up), rel.dot(self.forward))
    }

    /// tangent local 座標を world に逆変換
    #[must_use]
    pub fn local_to_world(&self, local: Vec3k) -> Vec3k {
        self.origin
            + self.right.scale(local.x)
            + self.up.scale(local.y)
            + self.forward.scale(local.z)
    }
}

/// 球面世界の mesh に対して auto-rig を実行
///
/// # 手順
///
/// 1. `TangentFrame::from_context(ctx)` で tangent frame 算出
/// 2. mesh 頂点を tangent local 座標に変換 (up が local Y に一致)
/// 3. §E.1 `auto_rig()` を local mesh で実行
/// 4. skeleton の world_position / local_position を world 座標に戻す
///
/// # Errors
///
/// §E.1 `auto_rig()` と同じエラー (`EmptyMesh` / `InvalidSliceCount` / `NoVerticalExtent`) を伝播
///
/// # Not-humanoid / partial skeleton
///
/// §E.1 と同じ挙動 (`skeleton: None` を返すか、部分骨格を返す)
///
/// # 決定性
///
/// 本関数は f32 (`Vec3k`) 演算のため、bit-exact な決定性 (lockstep / rollback) を
/// 必要とする場合は tangent frame 計算を Fix128 で行い、変換済み mesh を
/// §E.1 `auto_rig()` に直接渡す運用にする 詳細はモジュール top の docs 参照
pub fn auto_rig_on_sphere(
    mesh: &MeshView,
    ctx: &SphereContext,
    config: &AutoRigConfig,
) -> Result<AutoRigResult, AutoRigError> {
    let frame = TangentFrame::from_context(ctx);

    // Step 2: mesh を local に変換
    let local_vertices: Vec<Vec3k> = mesh
        .vertices
        .iter()
        .map(|&v| frame.world_to_local(v))
        .collect();
    let local_mesh = MeshView {
        vertices: &local_vertices,
        indices: mesh.indices,
    };

    // Step 3: 平面 auto_rig
    let mut result = auto_rig(&local_mesh, config)?;

    // Step 4: skeleton を world に戻す
    if let Some(ref mut skel) = result.skeleton {
        rebase_skeleton_to_world(skel, &frame);
    }

    Ok(result)
}

/// skeleton の world_position / local_position を tangent local → world に変換
///
/// - `world_position`: `local_to_world(local_world_position)` で world 座標に
/// - `local_position`: parent との world 差分から再計算 (親子関係は不変)
/// - `bone_length`: 変換後の parent 相対距離
fn rebase_skeleton_to_world(skel: &mut Skeleton, frame: &TangentFrame) {
    // まず全 joint の world_position を tangent local から world に変換
    let n = skel.joints.len();
    let mut new_world: Vec<Vec3k> = Vec::with_capacity(n);
    for joint in &skel.joints {
        new_world.push(frame.local_to_world(joint.world_position));
    }
    // world_position 更新
    for (i, joint) in skel.joints.iter_mut().enumerate() {
        joint.world_position = new_world[i];
    }
    // parent 相対 local_position + bone_length を再計算
    for i in 0..n {
        let parent_idx = skel.joints[i].parent;
        let local_position = match parent_idx {
            Some(p) => new_world[i] - new_world[p],
            None => new_world[i],
        };
        let bone_length = local_position.length();
        let joint: &mut SkeletonJoint = &mut skel.joints[i];
        joint.local_position = local_position;
        joint.bone_length = bone_length;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::BoneId;

    fn approx_sin(x: f32) -> f32 {
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

    fn approx_cos(x: f32) -> f32 {
        approx_sin(x + core::f32::consts::FRAC_PI_2)
    }

    /// Stick figure at local origin (Y-up), for later world transform
    fn stick_figure_local() -> Vec<Vec3k> {
        let mut v = Vec::new();
        // 頭
        for k in 0..12 {
            let theta = (k as f32) * core::f32::consts::TAU / 12.0;
            v.push(Vec3k::new(
                approx_cos(theta) * 0.08,
                0.90 + approx_sin(theta) * 0.08,
                0.0,
            ));
        }
        // 首
        for k in 0..6 {
            let theta = (k as f32) * core::f32::consts::TAU / 6.0;
            v.push(Vec3k::new(
                approx_cos(theta) * 0.03,
                0.83,
                approx_sin(theta) * 0.03,
            ));
        }
        // 胴
        for y_step in 0..6 {
            let y = 0.55 + (y_step as f32) * 0.05;
            for k in 0..12 {
                let theta = (k as f32) * core::f32::consts::TAU / 12.0;
                v.push(Vec3k::new(
                    approx_cos(theta) * 0.13,
                    y,
                    approx_sin(theta) * 0.08,
                ));
            }
        }
        // 腰
        for k in 0..12 {
            let theta = (k as f32) * core::f32::consts::TAU / 12.0;
            v.push(Vec3k::new(
                approx_cos(theta) * 0.09,
                0.48,
                approx_sin(theta) * 0.06,
            ));
        }
        // 骨盤
        for y in [0.42_f32, 0.45] {
            for k in 0..12 {
                let theta = (k as f32) * core::f32::consts::TAU / 12.0;
                v.push(Vec3k::new(
                    approx_cos(theta) * 0.12,
                    y,
                    approx_sin(theta) * 0.07,
                ));
            }
        }
        // 腕
        for x_end in [-0.35_f32, 0.35_f32] {
            for step in 0..12 {
                let t = (step as f32) / 12.0;
                let x = 0.13 * x_end.signum() + (x_end - 0.13 * x_end.signum()) * t;
                let y = 0.75 - t * 0.20;
                v.push(Vec3k::new(x, y, 0.0));
                v.push(Vec3k::new(x, y, 0.03));
                v.push(Vec3k::new(x, y, -0.03));
            }
        }
        // 脚
        for x_leg in [-0.06_f32, 0.06_f32] {
            for step in 0..12 {
                let y = 0.42 - (step as f32) / 12.0 * 0.42;
                for k in 0..6 {
                    let theta = (k as f32) * core::f32::consts::TAU / 6.0;
                    v.push(Vec3k::new(
                        x_leg + approx_cos(theta) * 0.05,
                        y,
                        approx_sin(theta) * 0.05,
                    ));
                }
            }
        }
        v
    }

    #[test]
    fn tangent_frame_north_pole() {
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(0.0, 300.0, 0.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        // up は +Y 方向
        assert!((frame.up.x).abs() < 1e-5);
        assert!((frame.up.y - 1.0).abs() < 1e-4);
        assert!((frame.up.z).abs() < 1e-5);
        // world Y と colinear なので forward は world Z 相当
        assert!((frame.forward.y).abs() < 0.02);
        // 直交条件
        assert!(frame.right.dot(frame.up).abs() < 1e-3);
        assert!(frame.up.dot(frame.forward).abs() < 1e-3);
        assert!(frame.right.dot(frame.forward).abs() < 1e-3);
    }

    #[test]
    fn tangent_frame_equator_x() {
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(300.0, 0.0, 0.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        // up は +X 方向
        assert!((frame.up.x - 1.0).abs() < 1e-4);
        assert!((frame.up.y).abs() < 1e-5);
        // 直交
        assert!(frame.right.dot(frame.up).abs() < 1e-3);
        assert!(frame.up.dot(frame.forward).abs() < 1e-3);
    }

    #[test]
    fn tangent_frame_degenerate_center() {
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::ZERO,
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        // 退化ケースでも panic せず有効な frame を返す
        assert!(frame.up.length().is_finite());
        assert!(frame.forward.length().is_finite());
        assert!(frame.right.length().is_finite());
    }

    #[test]
    fn world_to_local_roundtrip() {
        let ctx = SphereContext {
            center: Vec3k::new(1.0, 2.0, 3.0),
            character_pos: Vec3k::new(1.0, 302.0, 3.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        let world = Vec3k::new(2.0, 305.0, 4.0);
        let local = frame.world_to_local(world);
        let back = frame.local_to_world(local);
        assert!((back.x - world.x).abs() < 1e-4);
        assert!((back.y - world.y).abs() < 1e-4);
        assert!((back.z - world.z).abs() < 1e-4);
    }

    #[test]
    fn origin_at_character_pos() {
        // local (0,0,0) は world 側で character_pos に対応
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(10.0, 20.0, 30.0),
            radius: 50.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        let world = frame.local_to_world(Vec3k::ZERO);
        assert!((world.x - 10.0).abs() < 1e-5);
        assert!((world.y - 20.0).abs() < 1e-5);
        assert!((world.z - 30.0).abs() < 1e-5);
    }

    #[test]
    fn auto_rig_on_sphere_north_pole() {
        // stick figure を球北極に配置 (local origin → world (0, 300, 0))、up = +Y なので変換ほぼ identity
        let local_verts = stick_figure_local();
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(0.0, 300.0, 0.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        let world_verts: Vec<Vec3k> = local_verts
            .iter()
            .map(|&v| frame.local_to_world(v))
            .collect();
        let mesh = MeshView::from_vertices(&world_verts);
        let result = auto_rig_on_sphere(&mesh, &ctx, &AutoRigConfig::default()).unwrap();
        let skel = result
            .skeleton
            .expect("stick figure on sphere north pole should be detected as humanoid");
        // Hips は world 側で character_pos (北極) 付近
        let hips_world = skel.joint_world_position(BoneId::Hips).unwrap();
        assert!(hips_world.y > 250.0, "hips_world.y = {}", hips_world.y);
        // Head は Hips より up (world +Y) 方向 = 北極付近では world Y が大きい方
        let head_world = skel.joint_world_position(BoneId::Head).unwrap();
        assert!(
            head_world.y > hips_world.y,
            "head should be above hips: hips_y={}, head_y={}",
            hips_world.y,
            head_world.y
        );
    }

    #[test]
    fn auto_rig_on_sphere_equator() {
        // stick figure を赤道 (+X 方向) に配置、up = +X 方向
        let local_verts = stick_figure_local();
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(300.0, 0.0, 0.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        let world_verts: Vec<Vec3k> = local_verts
            .iter()
            .map(|&v| frame.local_to_world(v))
            .collect();
        let mesh = MeshView::from_vertices(&world_verts);
        let result = auto_rig_on_sphere(&mesh, &ctx, &AutoRigConfig::default()).unwrap();
        let skel = result
            .skeleton
            .expect("stick figure on sphere equator should be detected");
        // Hips は world 側で character_pos (300, 0, 0) 付近
        let hips_world = skel.joint_world_position(BoneId::Hips).unwrap();
        // 半径 300 の球面上、キャラは足元が球面 (X=300 付近)、頭は 300+1.7 相当
        assert!(hips_world.x > 250.0);
        // Head は Hips より up (world +X) 方向で外側
        let head_world = skel.joint_world_position(BoneId::Head).unwrap();
        // 頭は Hips より X が大きい (球面の外側方向)
        assert!(
            head_world.x > hips_world.x,
            "head should be radially outward: hips_x={}, head_x={}",
            hips_world.x,
            head_world.x
        );
    }

    #[test]
    fn empty_mesh_errors_on_sphere() {
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(0.0, 300.0, 0.0),
            radius: 300.0,
        };
        let empty: Vec<Vec3k> = Vec::new();
        let mesh = MeshView::from_vertices(&empty);
        let err = auto_rig_on_sphere(&mesh, &ctx, &AutoRigConfig::default()).unwrap_err();
        assert_eq!(err, AutoRigError::EmptyMesh);
    }

    #[test]
    fn cube_on_sphere_not_humanoid() {
        // 立方体は球面上でも not-humanoid のまま
        let local_cube = alloc::vec![
            Vec3k::new(0.0, 0.0, 0.0),
            Vec3k::new(1.0, 0.0, 0.0),
            Vec3k::new(0.0, 1.0, 0.0),
            Vec3k::new(1.0, 1.0, 0.0),
            Vec3k::new(0.0, 0.0, 1.0),
            Vec3k::new(1.0, 0.0, 1.0),
            Vec3k::new(0.0, 1.0, 1.0),
            Vec3k::new(1.0, 1.0, 1.0),
        ];
        let ctx = SphereContext {
            center: Vec3k::ZERO,
            character_pos: Vec3k::new(0.0, 300.0, 0.0),
            radius: 300.0,
        };
        let frame = TangentFrame::from_context(&ctx);
        let world_verts: Vec<Vec3k> = local_cube
            .iter()
            .map(|&v| frame.local_to_world(v))
            .collect();
        let mesh = MeshView::from_vertices(&world_verts);
        let result = auto_rig_on_sphere(&mesh, &ctx, &AutoRigConfig::default()).unwrap();
        assert!(result.skeleton.is_none());
    }
}
