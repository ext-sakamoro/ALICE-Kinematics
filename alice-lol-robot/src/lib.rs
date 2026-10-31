//! # alice-lol-robot
//!
//! Robot template for `alice_lol` DSL — Phase 3 Intent verb を Kinematics の骨格と packet に結線
//!
//! # 三相原理での位置付け
//!
//! Phase 3 Intent (`IntentNode` 14 verb) → 骨格の FK + 8-byte Kinematics packet の
//! **結線層 (glue layer)** 既存資産:
//!
//! - `alice_lol::intent::IntentNode` (14 verb + Sequence/Parallel + Program)
//! - `alice_kinematics::skeleton::Skeleton` (22 関節の骨格と FK)
//! - `alice_kinematics::Intent` (8-byte packet、Reach/Point/Grasp/Release)
//! - `alice_kinematics::lol_bridge::intent_to_kinematics` (LOL → Kinematics 翻訳)
//!
//! 本 crate は上記 3 資産を **additive に glue** し、`IntentExecutor` として MVP 化
//! kinematics / alice-lol いずれの crate も破壊変更なし
//!
//! # Roadmap (Phase R.0-R.5)
//!
//! - R.0 Scaffolding + Cargo.toml + CI stub (本 commit)
//! - R.1 [`executor::IntentExecutor`] core (LOL → Kinematics 翻訳)
//! - R.2 `humanoid_ext::HumanoidIntent` adapter (骨格に Intent を反映)
//! - R.3 [`protocol::IntentPacketStream`] (8-byte packet Vec エクスポート)
//! - R.4 [`law::SafetyLaw`] (ISO 10218 参考 velocity/workspace/collision guard)
//! - R.5 examples + tests + polish
//!
//! # Quick start (Phase R.1+ で有効化予定)
//!
//! ```ignore
//! use alice_lol::intent::{grasp, HandSide, ProgramBuilder};
//! use alice_kinematics::skeleton::Skeleton;
//! use alice_lol_robot::{skeleton_capsules, IntentExecutor};
//!
//! let body = Skeleton::default_humanoid();
//! let intent = grasp(0, HandSide::Right, 3.0);
//! let program = ProgramBuilder::new()
//!     .with_sdf(skeleton_capsules(&body, 0.05, 0.03).unwrap())
//!     .with_intent(intent)
//!     .build();
//! let executor = IntentExecutor::from_program(&program);
//! let packets = executor.translate().unwrap();
//! // → Vec<alice_kinematics::Intent> (8-byte packet 群)
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
#[cfg(feature = "kinematics")]
pub mod humanoid_ext;
pub mod law;

#[cfg(feature = "kinematics")]
pub mod executor;

#[cfg(feature = "kinematics")]
pub mod protocol;

pub use error::RobotError;
#[cfg(feature = "kinematics")]
pub use humanoid_ext::{joint_position, skeleton_capsules, HumanoidIntent, JOINTS};
pub use law::{SafetyLaw, SafetyViolation};

#[cfg(feature = "kinematics")]
pub use executor::IntentExecutor;

#[cfg(feature = "kinematics")]
pub use protocol::IntentPacketStream;
