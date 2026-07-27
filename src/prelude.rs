//! Convenience re-export (= `use alice_kinematics::prelude::*;` で主要 API 一括取得)
//!
//! `intent` / `joint` / `predictor` / `autorig` facade の 4 系統から主要型 + 関数を
//! re-export する `hand` / `skeleton` は用途特化なので prelude 非対象、
//! `encoder` / `jerk` は feature gate 済み

pub use crate::autorig::{auto_rig, AutoRigConfig, AutoRigError, AutoRigResult, MeshView};
pub use crate::intent::{Intent, IntentFlags, IntentType};
pub use crate::joint::{ArmChain, Joint, JointConstraint, Vec3k};
pub use crate::predictor::Predictor;
