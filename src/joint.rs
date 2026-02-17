//! Joint model — 7-DoF kinematic chain for human upper limb
//!
//! Models shoulder (3-DoF), elbow (1-DoF), wrist (3-DoF) with
//! anatomical rotation constraints. Forward/Inverse kinematics.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use core::ops::{Add, Sub, Mul, Neg};

/// 3D vector for kinematics (12 bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3k {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3k {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(self) -> f32 {
        fast_sqrt(self.length_sq())
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        let inv = 1.0 / len;
        Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Add for Vec3k {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl Sub for Vec3k {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl Mul<f32> for Vec3k {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl Neg for Vec3k {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

/// Fast square root (Quake-style)
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5 - half * y * y);
    let y = y * (1.5 - half * y * y);
    x * y
}

/// Rotation constraint for a joint axis
#[derive(Debug, Clone, Copy)]
pub struct JointConstraint {
    /// Minimum angle in radians
    pub min_rad: f32,
    /// Maximum angle in radians
    pub max_rad: f32,
}

impl JointConstraint {
    pub const fn new(min_deg: f32, max_deg: f32) -> Self {
        Self {
            min_rad: min_deg * (core::f32::consts::PI / 180.0),
            max_rad: max_deg * (core::f32::consts::PI / 180.0),
        }
    }

    pub const fn free() -> Self {
        Self { min_rad: -core::f32::consts::PI, max_rad: core::f32::consts::PI }
    }

    pub fn clamp(&self, angle: f32) -> f32 {
        if angle < self.min_rad { self.min_rad }
        else if angle > self.max_rad { self.max_rad }
        else { angle }
    }

    pub fn range(&self) -> f32 {
        self.max_rad - self.min_rad
    }
}

/// Single joint with rotation angle and constraint
#[derive(Debug, Clone, Copy)]
pub struct Joint {
    /// Joint name (8 chars)
    pub name: [u8; 8],
    /// Current angle in radians
    pub angle: f32,
    /// Rotation axis (local frame)
    pub axis: Vec3k,
    /// Constraint
    pub constraint: JointConstraint,
    /// Link length to next joint (meters)
    pub link_length: f32,
}

impl Joint {
    pub fn new(name: &[u8], axis: Vec3k, link_length: f32, constraint: JointConstraint) -> Self {
        let mut n = [0u8; 8];
        let len = name.len().min(8);
        n[..len].copy_from_slice(&name[..len]);
        Self { name: n, angle: 0.0, axis, constraint, link_length }
    }

    pub fn set_angle(&mut self, angle: f32) {
        self.angle = self.constraint.clamp(angle);
    }
}

/// Maximum joints in a chain
pub const MAX_JOINTS: usize = 7;

/// 7-DoF kinematic chain (human arm)
///
/// Shoulder: flexion/extension, abduction/adduction, rotation (3-DoF)
/// Elbow: flexion/extension (1-DoF)
/// Wrist: flexion/extension, deviation, pronation/supination (3-DoF)
///
/// Size: ~224 bytes
pub struct ArmChain {
    pub joints: [Joint; MAX_JOINTS],
    /// Base position (shoulder origin)
    pub base: Vec3k,
}

impl ArmChain {
    /// Create a default right arm chain with anatomical constraints
    pub fn right_arm() -> Self {
        let joints = [
            // Shoulder flexion/extension
            Joint::new(b"sh_flex", Vec3k::new(1.0, 0.0, 0.0), 0.0,
                JointConstraint::new(-60.0, 180.0)),
            // Shoulder abduction/adduction
            Joint::new(b"sh_abd", Vec3k::new(0.0, 0.0, 1.0), 0.0,
                JointConstraint::new(-50.0, 180.0)),
            // Shoulder rotation
            Joint::new(b"sh_rot", Vec3k::new(0.0, -1.0, 0.0), 0.30,
                JointConstraint::new(-90.0, 90.0)),
            // Elbow flexion
            Joint::new(b"el_flex", Vec3k::new(1.0, 0.0, 0.0), 0.28,
                JointConstraint::new(0.0, 145.0)),
            // Wrist flexion/extension
            Joint::new(b"wr_flex", Vec3k::new(1.0, 0.0, 0.0), 0.0,
                JointConstraint::new(-80.0, 80.0)),
            // Wrist deviation
            Joint::new(b"wr_dev", Vec3k::new(0.0, 0.0, 1.0), 0.0,
                JointConstraint::new(-20.0, 30.0)),
            // Wrist pronation/supination
            Joint::new(b"wr_pro", Vec3k::new(0.0, -1.0, 0.0), 0.20,
                JointConstraint::new(-80.0, 80.0)),
        ];
        Self {
            joints,
            base: Vec3k::ZERO,
        }
    }

    /// Forward Kinematics — compute end-effector position from joint angles
    pub fn forward_kinematics(&self) -> Vec3k {
        let mut pos = self.base;
        let mut dir = Vec3k::new(0.0, -1.0, 0.0); // initial pointing down

        for j in &self.joints {
            // Rotate direction by joint angle around joint axis
            dir = rotate_vec(dir, j.axis, j.angle);
            // Advance position along link
            pos = pos + dir.scale(j.link_length);
        }
        pos
    }

    /// Simple CCD (Cyclic Coordinate Descent) Inverse Kinematics
    ///
    /// Returns number of iterations used, and final error distance.
    pub fn inverse_kinematics(&mut self, target: Vec3k, max_iter: u32, tolerance: f32) -> (u32, f32) {
        for iter in 0..max_iter {
            let end = self.forward_kinematics();
            let error = end.distance(target);
            if error < tolerance {
                return (iter, error);
            }

            // CCD: iterate joints from tip to base
            for i in (0..MAX_JOINTS).rev() {
                let joint_pos = self.joint_position(i);
                let end_pos = self.forward_kinematics();

                let to_end = (end_pos - joint_pos).normalize();
                let to_target = (target - joint_pos).normalize();

                // Angle between vectors
                let dot = to_end.dot(to_target);
                let dot = if dot > 1.0 { 1.0 } else if dot < -1.0 { -1.0 } else { dot };
                let angle = acos_approx(dot);

                // Determine rotation direction
                let cross = to_end.cross(to_target);
                let sign = if cross.dot(self.joints[i].axis) >= 0.0 { 1.0 } else { -1.0 };

                self.joints[i].set_angle(self.joints[i].angle + sign * angle * 0.5);
            }
        }

        let error = self.forward_kinematics().distance(target);
        (max_iter, error)
    }

    /// Get world-space position of joint i
    pub fn joint_position(&self, joint_idx: usize) -> Vec3k {
        let mut pos = self.base;
        let mut dir = Vec3k::new(0.0, -1.0, 0.0);

        for (i, j) in self.joints.iter().enumerate() {
            if i > joint_idx {
                break;
            }
            dir = rotate_vec(dir, j.axis, j.angle);
            pos = pos + dir.scale(j.link_length);
        }
        pos
    }

    /// Total arm length (sum of all link lengths)
    pub fn total_length(&self) -> f32 {
        let mut len = 0.0;
        for j in &self.joints {
            len += j.link_length;
        }
        len
    }

    /// Get all joint angles as array
    pub fn angles(&self) -> [f32; MAX_JOINTS] {
        let mut a = [0.0f32; MAX_JOINTS];
        for (i, j) in self.joints.iter().enumerate() {
            a[i] = j.angle;
        }
        a
    }

    /// Set all joint angles from array
    pub fn set_angles(&mut self, angles: &[f32; MAX_JOINTS]) {
        for (i, a) in angles.iter().enumerate() {
            self.joints[i].set_angle(*a);
        }
    }
}

/// Rotate vector `v` around axis `axis` by angle `theta` (Rodrigues' formula)
fn rotate_vec(v: Vec3k, axis: Vec3k, theta: f32) -> Vec3k {
    let (sin_t, cos_t) = sin_cos_approx(theta);
    let k = axis.normalize();
    let term1 = v.scale(cos_t);
    let term2 = k.cross(v).scale(sin_t);
    let term3 = k.scale(k.dot(v) * (1.0 - cos_t));
    term1 + term2 + term3
}

/// Approximate sin and cos (Bhaskara I + identity)
fn sin_cos_approx(theta: f32) -> (f32, f32) {
    (sin_approx(theta), sin_approx(theta + core::f32::consts::FRAC_PI_2))
}

/// Fast sine approximation (Bhaskara I, max error ~0.2%)
fn sin_approx(x: f32) -> f32 {
    let pi = core::f32::consts::PI;
    let mut x = x % (2.0 * pi);
    if x < 0.0 { x += 2.0 * pi; }

    let sign = if x > pi { -1.0 } else { 1.0 };
    if x > pi { x -= pi; }

    let num = 16.0 * x * (pi - x);
    let den = 5.0 * pi * pi - 4.0 * x * (pi - x);
    sign * num / den
}

/// Fast acos approximation (Abramowitz & Stegun)
fn acos_approx(x: f32) -> f32 {
    let abs_x = if x < 0.0 { -x } else { x };
    let result = -0.0187293 * abs_x;
    let result = (result + 0.0742610) * abs_x;
    let result = (result - 0.2121144) * abs_x;
    let result = result + core::f32::consts::FRAC_PI_2;
    let result = result * fast_sqrt(1.0 - abs_x);
    if x < 0.0 { core::f32::consts::PI - result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3k_basic() {
        let a = Vec3k::new(1.0, 2.0, 3.0);
        let b = Vec3k::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert!((c.x - 5.0).abs() < 0.001);
        assert!((c.y - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_vec3k_length() {
        let v = Vec3k::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_vec3k_normalize() {
        let v = Vec3k::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vec3k_cross() {
        let x = Vec3k::new(1.0, 0.0, 0.0);
        let y = Vec3k::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_joint_constraint() {
        let c = JointConstraint::new(0.0, 90.0);
        let clamped = c.clamp(2.0);
        assert!(clamped <= c.max_rad);
        assert!(clamped >= c.min_rad);
    }

    #[test]
    fn test_joint_constraint_clamp_below() {
        let c = JointConstraint::new(0.0, 90.0);
        let clamped = c.clamp(-1.0);
        assert!((clamped - c.min_rad).abs() < 0.001);
    }

    #[test]
    fn test_arm_chain_creation() {
        let arm = ArmChain::right_arm();
        assert_eq!(arm.joints.len(), 7);
        assert!(arm.total_length() > 0.0);
    }

    #[test]
    fn test_forward_kinematics_zero() {
        let arm = ArmChain::right_arm();
        let end = arm.forward_kinematics();
        // With all angles at 0, end effector should be offset from base
        assert!(end.distance(arm.base) > 0.0);
    }

    #[test]
    fn test_forward_kinematics_deterministic() {
        let arm = ArmChain::right_arm();
        let e1 = arm.forward_kinematics();
        let e2 = arm.forward_kinematics();
        assert!((e1.x - e2.x).abs() < 1e-6);
        assert!((e1.y - e2.y).abs() < 1e-6);
        assert!((e1.z - e2.z).abs() < 1e-6);
    }

    #[test]
    fn test_total_arm_length() {
        let arm = ArmChain::right_arm();
        // Upper arm ~0.30 + forearm ~0.28 + hand ~0.20 = ~0.78m
        let len = arm.total_length();
        assert!((len - 0.78).abs() < 0.01);
    }

    #[test]
    fn test_set_angles_within_constraints() {
        let mut arm = ArmChain::right_arm();
        let angles = [0.5, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0];
        arm.set_angles(&angles);
        // Elbow flexion constraint: 0..145 deg
        assert!((arm.joints[3].angle - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_inverse_kinematics_reachable() {
        let mut arm = ArmChain::right_arm();
        arm.base = Vec3k::new(0.0, 1.5, 0.0);
        // Target within reach
        let target = Vec3k::new(0.0, 1.0, 0.3);
        let (iters, error) = arm.inverse_kinematics(target, 50, 0.05);
        assert!(error < 0.15, "IK error too large: {error}");
        assert!(iters <= 50);
    }

    #[test]
    fn test_rotate_vec_identity() {
        let v = Vec3k::new(1.0, 0.0, 0.0);
        let r = rotate_vec(v, Vec3k::new(0.0, 1.0, 0.0), 0.0);
        assert!((r.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rotate_vec_90deg() {
        let v = Vec3k::new(1.0, 0.0, 0.0);
        let r = rotate_vec(v, Vec3k::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2);
        // Should rotate to approximately (0, 1, 0)
        assert!((r.x).abs() < 0.02);
        assert!((r.y - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_sin_approx() {
        let s = sin_approx(core::f32::consts::FRAC_PI_2);
        assert!((s - 1.0).abs() < 0.01);
        let s = sin_approx(0.0);
        assert!(s.abs() < 0.01);
    }

    #[test]
    fn test_acos_approx() {
        let a = acos_approx(0.0);
        assert!((a - core::f32::consts::FRAC_PI_2).abs() < 0.02);
        let a = acos_approx(1.0);
        assert!(a.abs() < 0.02);
    }
}
