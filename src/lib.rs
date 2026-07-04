//! ALICE-Kinematics — Human motion intent compression
//!
//! Replaces 1000Hz raw coordinate streaming with 8-byte kinematic intents.
//! Decoder/predictor (MIT) reconstructs full trajectories from intents.
//! Encoder (AGPL-3.0, feature `encoder`) extracts intents from raw sensor data.
//!
//! # Quick Start
//!
//! ```
//! use alice_kinematics::{Intent, IntentFlags, IntentType, Predictor, Vec3k};
//!
//! // 送信側: 8バイトのintentパケットを生成
//! let flags = IntentFlags::new(IntentType::Reach, false, false, 0);
//! let intent = Intent {
//!     target: Vec3k::new(0.5, 0.3, 0.0),
//!     duration_ms: 200,
//!     flags,
//! };
//!
//! // 受信側: intentからトラジェクトリを復元
//! let mut predictor = Predictor::new();
//! predictor.apply_intent(intent);
//! let pos = predictor.position_at(0.1); // 100ms時点の位置
//! assert!(pos.x > 0.0);
//! ```
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always
)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod hand;
pub mod intent;
pub mod joint;
pub mod predictor;
pub mod prelude;
pub mod skeleton;

#[cfg(feature = "encoder")]
pub mod encoder;
#[cfg(feature = "encoder")]
pub mod jerk;

pub use intent::{Intent, IntentFlags, IntentType};
pub use joint::{ArmChain, Joint, JointConstraint, Vec3k};
pub use predictor::Predictor;
