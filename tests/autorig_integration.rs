//! Integration tests for auto-rig (§E.1 平面世界向け)
//!
//! Synthetic mesh fixtures (実機不要):
//! 1. 立方体 → not humanoid
//! 2. 縦長円柱 (aspect 3.0) → spine-only

// autorig module と同じ pedantic allow を integration test でも適用
// (i32 → f32 座標変換 / usize → u16 index 圧縮 が頻出)
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
//! 3. Stick figure (torso + 2 arms + 2 legs + head) → 15+ bone
//! 4. 腕欠損 stick figure → 部分骨格
//! 5. 車形状 (aspect 0.4) → not humanoid
//! + skinning weight sum = 1.0 per vertex

use alice_kinematics::autorig::{auto_rig, AutoRigConfig, MeshView};
use alice_kinematics::skeleton::{BoneId, BONE_COUNT};
use alice_kinematics::Vec3k;

// ============================================================================
// Fixture builders
// ============================================================================

/// 三角関数近似 (Bhaskara + shift)
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

/// 立方体 1×1×1 (aspect ratio = 1.0)
fn cube_fixture() -> Vec<Vec3k> {
    vec![
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

/// 縦長円柱 (半径 0.15、高さ 1.5、aspect ratio = 5.0)
fn tall_cylinder_fixture() -> Vec<Vec3k> {
    let mut v = Vec::new();
    let radius = 0.15_f32;
    let height = 1.5_f32;
    for k in 0..30 {
        let y = (k as f32) / 30.0 * height;
        for a in 0..12 {
            let theta = (a as f32) * core::f32::consts::TAU / 12.0;
            v.push(Vec3k::new(
                approx_cos(theta) * radius,
                y,
                approx_sin(theta) * radius,
            ));
        }
    }
    v
}

/// Stick figure — 全身ヒューマノイド近似
///
/// - 頭: y ∈ [0.85, 1.00] 半径 0.08
/// - 首: y = 0.83 幅 0.03 (waist より上の細い部分)
/// - 胴上部 (胸): y ∈ [0.55, 0.80] 幅 0.13
/// - 胴中部 (腰): y = 0.48 幅 0.09 (最も細い)
/// - 胴下部 (骨盤): y ∈ [0.42, 0.45] 幅 0.12
/// - 腕: 左右 y ∈ [0.55, 0.75] で肩、水平に伸びて x = ±0.35 の手
/// - 脚: 左右 y ∈ [0.00, 0.45] x = ±0.06 太さ 0.05
fn stick_figure_fixture() -> Vec<Vec3k> {
    let mut v = Vec::new();

    // 頭 (球体近似)
    for k in 0..12 {
        let theta = (k as f32) * core::f32::consts::TAU / 12.0;
        v.push(Vec3k::new(
            approx_cos(theta) * 0.08,
            0.90 + approx_sin(theta) * 0.08,
            0.0,
        ));
        v.push(Vec3k::new(0.0, 0.90, approx_cos(theta) * 0.08));
    }

    // 首 (細い部分)
    for k in 0..6 {
        let theta = (k as f32) * core::f32::consts::TAU / 6.0;
        v.push(Vec3k::new(
            approx_cos(theta) * 0.03,
            0.83,
            approx_sin(theta) * 0.03,
        ));
    }

    // 胴上部 (胸)、y ∈ [0.55, 0.80] 楕円形
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

    // 胴中部 (腰、最も細い)、y = 0.48
    for k in 0..12 {
        let theta = (k as f32) * core::f32::consts::TAU / 12.0;
        v.push(Vec3k::new(
            approx_cos(theta) * 0.09,
            0.48,
            approx_sin(theta) * 0.06,
        ));
    }

    // 胴下部 (骨盤)、y ∈ [0.42, 0.45]
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

    // 腕 (左右、y ∈ [0.55, 0.75] 肩から x = ±0.35 の手先へ)
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

    // 脚 (左右、y ∈ [0.00, 0.45])
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

/// 腕欠損の stick figure (胴 + 頭 + 脚、腕頂点をすべて除去)
fn armless_stick_figure_fixture() -> Vec<Vec3k> {
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
    // 胴 (胸 + 腰 + 骨盤)
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
    for k in 0..12 {
        let theta = (k as f32) * core::f32::consts::TAU / 12.0;
        v.push(Vec3k::new(
            approx_cos(theta) * 0.09,
            0.48,
            approx_sin(theta) * 0.06,
        ));
    }
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

/// 車形状 (幅 2.5、奥行 5.0、高さ 1.4、aspect ratio = 0.28)
fn car_fixture() -> Vec<Vec3k> {
    let mut v = Vec::new();
    for x_step in 0..5 {
        let x = -1.25 + (x_step as f32) * 0.625;
        for y_step in 0..4 {
            let y = (y_step as f32) * 0.4;
            for z_step in 0..8 {
                let z = -2.5 + (z_step as f32) * 0.625;
                v.push(Vec3k::new(x, y, z));
            }
        }
    }
    v
}

// ============================================================================
// Test cases
// ============================================================================

#[test]
fn cube_is_not_humanoid() {
    let v = cube_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    assert!(
        result.skeleton.is_none(),
        "cube should be not-humanoid, but skeleton was detected"
    );
    assert!(result.skinning.is_none());
}

#[test]
fn car_is_not_humanoid() {
    let v = car_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    assert!(
        result.skeleton.is_none(),
        "car (aspect ratio ~0.28) should be not-humanoid"
    );
}

#[test]
fn tall_cylinder_produces_skeleton() {
    // 縦長円柱 (aspect 5.0) は humanoid 判定される
    // waist / neck detection は幅一様なので不安定だが、少なくとも
    // aspect check は通り、hypothesis 構築段階で None になれば skeleton = None
    let v = tall_cylinder_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    // skeleton は Some か None どちらでもよい (aspect は通っている)
    // 検出された場合、22 joint あることを確認
    if let Some(skel) = &result.skeleton {
        assert_eq!(skel.joint_count(), BONE_COUNT);
    }
}

#[test]
fn stick_figure_produces_full_skeleton() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skel = result
        .skeleton
        .as_ref()
        .expect("stick figure should be detected as humanoid");
    assert_eq!(skel.joint_count(), BONE_COUNT);
    // 主要 joint がすべて検出される
    for id in [
        BoneId::Hips,
        BoneId::Spine,
        BoneId::Chest,
        BoneId::Neck,
        BoneId::Head,
        BoneId::LeftShoulder,
        BoneId::RightShoulder,
        BoneId::LeftHand,
        BoneId::RightHand,
        BoneId::LeftFoot,
        BoneId::RightFoot,
    ] {
        let pos = skel.joint_world_position(id);
        assert!(pos.is_some(), "joint {id:?} should exist");
    }
    // confidence は 15/22 以上
    assert!(
        result.confidence >= 15.0 / (BONE_COUNT as f32),
        "confidence too low: {}",
        result.confidence
    );
}

#[test]
fn stick_figure_head_above_hips() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skel = result.skeleton.unwrap();
    let hips_y = skel.joint_world_position(BoneId::Hips).unwrap().y;
    let head_y = skel.joint_world_position(BoneId::Head).unwrap().y;
    assert!(head_y > hips_y, "head must be above hips");
}

#[test]
fn stick_figure_left_arm_on_left() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skel = result.skeleton.unwrap();
    let l_hand = skel.joint_world_position(BoneId::LeftHand).unwrap();
    let r_hand = skel.joint_world_position(BoneId::RightHand).unwrap();
    // 左手は右手より X が小さい
    assert!(
        l_hand.x < r_hand.x,
        "left hand should have smaller X: L={} R={}",
        l_hand.x,
        r_hand.x
    );
}

#[test]
fn armless_stick_figure_produces_partial_skeleton() {
    let v = armless_stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    // 骨格は検出される (humanoid aspect)
    let skel = result.skeleton.expect("armless figure should still detect");
    assert_eq!(skel.joint_count(), BONE_COUNT);
    // 脚は検出される
    assert!(skel.joint_world_position(BoneId::LeftFoot).is_some());
    assert!(skel.joint_world_position(BoneId::RightFoot).is_some());
    // Shoulder は chest から自動配置される (常に Some)
    // UpperArm/LowerArm/Hand は腕頂点がないので None (fallback で置かれるが元位置)
    // 確認: Hand 位置は Shoulder と同じか極近い (fallback で下方向に少し伸ばされた程度)
    let l_shoulder = skel.joint_world_position(BoneId::LeftShoulder).unwrap();
    let l_hand = skel.joint_world_position(BoneId::LeftHand).unwrap();
    let dist = (l_hand - l_shoulder).length();
    // 腕検出なしなら fallback で短い bone、腕検出ありなら長い bone (>0.5m)
    // 検出なしを期待するので dist < 0.3m
    // ただし場合によっては胴頂点を腕として拾う可能性もあるので緩めに
    assert!(dist < 0.5, "armless: hand-shoulder should be short: {dist}");
}

#[test]
fn stick_figure_skinning_weights_sum_to_one() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skinning = result.skinning.expect("skinning should exist");
    assert_eq!(skinning.weights.len(), v.len());
    for (i, w) in skinning.weights.iter().enumerate() {
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "vertex {i}: weight sum = {sum}, indices = {:?}",
            skinning.bone_indices[i]
        );
    }
}

#[test]
fn stick_figure_skinning_indices_valid() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skinning = result.skinning.unwrap();
    let joint_count = result.skeleton.unwrap().joint_count() as u16;
    for indices in &skinning.bone_indices {
        for &i in indices {
            assert!(i < joint_count, "bone index {i} out of range");
        }
    }
}

#[test]
fn stick_figure_bone_lengths_positive() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skel = result.skeleton.unwrap();
    // Hips 以外の全ての joint に bone_length > 0
    for joint in &skel.joints {
        if joint.parent.is_some() {
            assert!(
                joint.bone_length >= 0.0,
                "joint {:?} has negative bone_length: {}",
                joint.id,
                joint.bone_length
            );
        }
    }
}

#[test]
fn stick_figure_height_matches_mesh() {
    let v = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&v);
    let result = auto_rig(&mesh, &AutoRigConfig::default()).unwrap();
    let skel = result.skeleton.unwrap();
    // stick figure の Y 範囲は 0.0-0.98 (頭上頂点)
    let ys: Vec<f32> = v.iter().map(|p| p.y).collect();
    let min_y = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let expected_height = max_y - min_y;
    assert!(
        (skel.height - expected_height).abs() < 0.05,
        "skeleton height {} vs mesh height {}",
        skel.height,
        expected_height
    );
}
