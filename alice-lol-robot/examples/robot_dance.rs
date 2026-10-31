//! Sequence / Parallel 合成 demo
//!
//! LOL Program の合成 verb (`Sequence` / `Parallel`) が Kinematics packet に flatten される
//! 挙動と骨格側の逐次適用を確認
//!
//! 実行:
//! ```bash
//! cargo run --example robot_dance -p alice-lol-robot
//! ```

use alice_kinematics::skeleton::Skeleton;
use alice_lol::intent::{grasp, parallel, release, rest, sequence, walk, HandSide, ProgramBuilder};
use alice_lol::SdfNode;
use alice_lol_robot::{
    joint_position, skeleton_capsules, HumanoidIntent, IntentPacketStream, SafetyLaw,
};
use glam::Vec3;
use std::sync::Arc;

/// capsule の半径と SmoothUnion の滑らかさ (m、骨格の単位)
const BONE_RADIUS: f32 = 0.05;
const BONE_SMOOTHNESS: f32 = 0.05;

fn main() {
    println!("=== alice-lol-robot: Sequence / Parallel composition demo ===\n");

    let body = Skeleton::default_humanoid();

    // 対象カップを (0.6, 1.0, 0.2) に置く
    let cup = SdfNode::Translate {
        child: Arc::new(SdfNode::Sphere { radius: 0.04 }),
        offset: Vec3::new(0.6, 1.0, 0.2),
    };
    let mut builder = ProgramBuilder::new().with_sdf(
        skeleton_capsules(&body, BONE_RADIUS, BONE_SMOOTHNESS).expect("the skeleton has bones"),
    );
    let cup_id = builder.register(cup);

    // Sequence: walk → grasp → walk → release → rest
    let dance = sequence(vec![
        walk(Vec3::new(0.5, 0.0, 0.0), 1.0),
        grasp(cup_id, HandSide::Right, 3.0),
        walk(Vec3::new(2.5, 0.0, 0.5), 1.2),
        release(cup_id),
        rest(500),
    ]);
    // Parallel 例: 歩きながら休む (concurrency semantic 情報は Kinematics flatten で失われる)
    let dance_with_par = sequence(vec![
        dance,
        parallel(vec![walk(Vec3::new(0.0, 0.0, 0.0), 0.8), rest(1000)]),
    ]);

    let program = builder.with_intent(dance_with_par).build();

    // safety check
    let law = SafetyLaw::default_collab();
    let violations = law.check_program(&program);
    println!("safety violations: {}", violations.len());
    for v in &violations {
        println!("  [{}] {}", v.rule, v.detail);
    }
    if !violations.is_empty() {
        return;
    }

    // 骨格に apply (Sequence = fold left、Parallel = 先頭のみ)
    let final_pose = program
        .intent
        .as_ref()
        .map_or_else(|| body.clone(), |i| body.apply_intent(i).unwrap());
    let waist = joint_position(&final_pose, "waist").expect("waist");
    println!(
        "\nfinal waist after dance: ({:.2}, {:.2}, {:.2})",
        waist.x, waist.y, waist.z
    );

    // packet stream
    let packets = IntentPacketStream::from_program(&program).unwrap();
    println!(
        "\npackets: {} × 8 byte = {} bytes",
        packets.len(),
        IntentPacketStream::total_bytes(&packets)
    );

    let decoded = IntentPacketStream::decode_all(&packets);
    for (i, intent) in decoded.iter().enumerate() {
        println!(
            "  [{i}] type={:?} target=({:.2}, {:.2}, {:.2}) duration={}ms",
            intent.flags.intent_type(),
            intent.target.x,
            intent.target.y,
            intent.target.z,
            intent.duration_ms
        );
    }

    println!("\n=== done ===");
}
