//! Walk verb → skeleton translation + 8-byte packet demo
//!
//! `Walk` verb は骨格の `apply_intent` で **実際に joint 位置が動く** 唯一の
//! MVP verb (他 verb は Phase R.6+ で IK 実装予定)
//!
//! 実行:
//! ```bash
//! cargo run --example robot_walk -p alice-lol-robot
//! ```

use alice_kinematics::skeleton::Skeleton;
use alice_lol::intent::{walk, ProgramBuilder};
use alice_lol_robot::{
    joint_position, skeleton_capsules, HumanoidIntent, IntentPacketStream, SafetyLaw,
};
use glam::Vec3;

/// capsule の半径と SmoothUnion の滑らかさ (m、骨格の単位)
const BONE_RADIUS: f32 = 0.05;
const BONE_SMOOTHNESS: f32 = 0.05;

fn main() {
    println!("=== alice-lol-robot: Walk verb demo ===\n");

    let body = Skeleton::default_humanoid();
    let initial_waist = joint_position(&body, "waist").expect("waist");
    let initial_head = joint_position(&body, "head").expect("head");
    println!(
        "initial waist: ({:.2}, {:.2}, {:.2})",
        initial_waist.x, initial_waist.y, initial_waist.z
    );
    println!(
        "initial head:  ({:.2}, {:.2}, {:.2})",
        initial_head.x, initial_head.y, initial_head.z
    );

    // Walk verb: (2.5, 0, 1.0) へ 1.0 m/s で移動
    let destination = Vec3::new(2.5, 0.0, 1.0);
    let intent = walk(destination, 1.0);

    // Safety check
    let law = SafetyLaw::default_collab();
    let violations = law.check_intent(&intent);
    if violations.is_empty() {
        println!("\nsafety: PASS (Walk to {destination:?} at 1.0 m/s)");
    } else {
        for v in &violations {
            println!("safety violation: [{}] {}", v.rule, v.detail);
        }
        return;
    }

    // 骨格に反映 (Walk = 剛体の平行移動)
    let posed = body.apply_intent(&intent).unwrap();
    let new_waist = joint_position(&posed, "waist").expect("waist");
    let new_head = joint_position(&posed, "head").expect("head");
    println!("\nafter walk:");
    println!(
        "  new waist: ({:.2}, {:.2}, {:.2})",
        new_waist.x, new_waist.y, new_waist.z
    );
    println!(
        "  new head:  ({:.2}, {:.2}, {:.2})",
        new_head.x, new_head.y, new_head.z
    );
    println!(
        "  delta:     ({:.2}, {:.2}, {:.2})",
        new_waist.x - initial_waist.x,
        new_waist.y - initial_waist.y,
        new_waist.z - initial_waist.z
    );

    // 8-byte packet 出力 (Walk → Kinematics IntentType::Reach)
    let program = ProgramBuilder::new()
        .with_sdf(
            skeleton_capsules(&body, BONE_RADIUS, BONE_SMOOTHNESS).expect("the skeleton has bones"),
        )
        .with_intent(intent)
        .build();
    let packets = IntentPacketStream::from_program(&program).unwrap();
    println!("\npackets: {} × 8 byte", packets.len());
    for (i, packet) in packets.iter().enumerate() {
        print!("  packet[{i}]: ");
        for byte in packet {
            print!("{byte:02x} ");
        }
        println!();
    }

    let decoded = IntentPacketStream::decode_all(&packets);
    for (i, intent) in decoded.iter().enumerate() {
        println!(
            "  decoded[{i}]: type={:?} target=({:.2}, {:.2}, {:.2})",
            intent.flags.intent_type(),
            intent.target.x,
            intent.target.y,
            intent.target.z
        );
    }

    println!("\n=== done ===");
}
