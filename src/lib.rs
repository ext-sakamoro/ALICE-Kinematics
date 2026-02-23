//! ALICE-Kinematics — Human motion intent compression
//!
//! Replaces 1000Hz raw coordinate streaming with 8-byte kinematic intents.
//! Decoder/predictor (MIT) reconstructs full trajectories from intents.
//! Encoder (AGPL-3.0, feature `encoder`) extracts intents from raw sensor data.
//!
//! Author: Moroya Sakamoto

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod intent;
pub mod joint;
pub mod predictor;

#[cfg(feature = "encoder")]
pub mod encoder;
#[cfg(feature = "encoder")]
pub mod jerk;

pub use intent::{Intent, IntentFlags, IntentType};
pub use joint::{ArmChain, Joint, JointConstraint, Vec3k};
pub use predictor::Predictor;
