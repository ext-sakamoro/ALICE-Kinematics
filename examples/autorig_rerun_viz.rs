//! autorig 結果を rerun-io/rerun viewer で可視化する example
//!
//! 実行:
//!
//! ```bash
//! cargo run --example autorig_rerun_viz --features rerun-viz --release
//! ```
//!
//! rerun viewer が spawn され、以下が log される:
//! - `mesh/points`: stick figure mesh 頂点 (colored by skinning bone index)
//! - `skeleton/bones`: 22-joint skeleton (line segments、Hips 起点)
//! - `skeleton/joints`: 22 joint position (radius 0.02 sphere marker)
//! - `stats/text`: aspect ratio + confidence + joint count 等のメタ情報
//!
//! # 背景
//!
//! image-to-3d §E 系列の auto-rig を実装後、debug 目的で
//! rerun (Apache-2/MIT、11k★) を dev-dep として統合
//! release binary には影響なし (feature gate `rerun-viz`)
//!
//! License: MIT
//! Author: Moroya Sakamoto

use alice_kinematics::autorig::{auto_rig, AutoRigConfig, MeshView};
use alice_kinematics::skeleton::BoneId;
use alice_kinematics::Vec3k;

/// 三角関数近似 (Bhaskara + shift、std::f32::sin を避けて no_std 対応 code と揃える)
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

/// Stick figure fixture (integration test と同じ shape)
fn stick_figure_fixture() -> Vec<Vec3k> {
    let mut v = Vec::new();
    // 頭
    for k in 0..12 {
        let theta = (k as f32) * core::f32::consts::TAU / 12.0;
        v.push(Vec3k::new(
            approx_cos(theta) * 0.08,
            0.90 + approx_sin(theta) * 0.08,
            0.0,
        ));
        v.push(Vec3k::new(0.0, 0.90, approx_cos(theta) * 0.08));
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. mesh + auto_rig
    let vertices = stick_figure_fixture();
    let mesh = MeshView::from_vertices(&vertices);
    let config = AutoRigConfig::default();
    let result = auto_rig(&mesh, &config).map_err(|e| format!("auto_rig error: {e:?}"))?;

    let skel = match &result.skeleton {
        Some(s) => s,
        None => {
            eprintln!("Not humanoid (aspect ratio too low). Nothing to visualize.");
            return Ok(());
        }
    };
    let skinning = result.skinning.as_ref().ok_or("skinning missing")?;

    // 2. rerun viewer 起動
    let rec = rerun::RecordingStreamBuilder::new("alice_kinematics_autorig_viz").spawn()?;

    // 3. mesh 頂点を log (bone_indices の top-1 で色分け)
    // カラーパレット: 22 joint 分の distinguishable colors
    let palette = generate_palette(22);
    let vertex_positions: Vec<[f32; 3]> = vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let vertex_colors: Vec<[u8; 3]> = skinning
        .bone_indices
        .iter()
        .map(|indices| {
            let top_bone = indices[0] as usize;
            palette[top_bone % palette.len()]
        })
        .collect();

    rec.log(
        "mesh/points",
        &rerun::Points3D::new(vertex_positions)
            .with_colors(vertex_colors)
            .with_radii([0.006]),
    )?;

    // 4. skeleton bones (line segments、parent → child)
    let mut bone_lines: Vec<[[f32; 3]; 2]> = Vec::new();
    for joint in &skel.joints {
        if let Some(parent_idx) = joint.parent {
            let a = skel.joints[parent_idx].world_position;
            let b = joint.world_position;
            bone_lines.push([[a.x, a.y, a.z], [b.x, b.y, b.z]]);
        }
    }
    rec.log(
        "skeleton/bones",
        &rerun::LineStrips3D::new(bone_lines.iter().map(|seg| seg.to_vec()))
            .with_colors([[255, 200, 0]])
            .with_radii([0.005]),
    )?;

    // 5. joint markers (radius 0.02 spheres)
    let joint_positions: Vec<[f32; 3]> = skel
        .joints
        .iter()
        .map(|j| [j.world_position.x, j.world_position.y, j.world_position.z])
        .collect();
    let joint_labels: Vec<String> = skel.joints.iter().map(|j| format!("{:?}", j.id)).collect();
    rec.log(
        "skeleton/joints",
        &rerun::Points3D::new(joint_positions)
            .with_colors([[255, 100, 100]])
            .with_radii([0.015])
            .with_labels(joint_labels),
    )?;

    // 6. Stats text
    let stats = format!(
        "AutoRig Result\n\
         confidence: {:.2}\n\
         joint count: {}\n\
         mesh vertices: {}\n\
         hips y (world): {:.3}\n\
         head y (world): {:.3}",
        result.confidence,
        skel.joint_count(),
        vertices.len(),
        skel.joint_world_position(BoneId::Hips).map_or(0.0, |p| p.y),
        skel.joint_world_position(BoneId::Head).map_or(0.0, |p| p.y),
    );
    rec.log("stats/text", &rerun::TextDocument::new(stats))?;

    println!("Logged to rerun. Viewer window should be open.");
    println!("Press Ctrl+C to exit (rerun viewer stays open).");

    Ok(())
}

/// N 個の distinguishable RGB colors を HSV 空間で等間隔生成
fn generate_palette(n: usize) -> Vec<[u8; 3]> {
    (0..n)
        .map(|i| {
            let h = (i as f32) / (n as f32) * 360.0;
            hsv_to_rgb(h, 0.8, 0.9)
        })
        .collect()
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let hh = h / 60.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());
    let (r, g, b) = match hh as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}
