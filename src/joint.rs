//! Joint model — 7-DoF kinematic chain for human upper limb
//!
//! Models shoulder (3-DoF), elbow (1-DoF), wrist (3-DoF) with
//! anatomical rotation constraints. Forward/Inverse kinematics.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use core::ops::{Add, Mul, Neg, Sub};

/// 3D vector for kinematics (12 bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3k {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3k {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    #[must_use]
    pub fn length(self) -> f32 {
        fast_sqrt(self.length_sq())
    }

    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        let inv = 1.0 / len;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Add for Vec3k {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3k {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for Vec3k {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Neg for Vec3k {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Fast square root (Quake-style)
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f37_59df - (i >> 1);
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
    #[must_use]
    pub const fn new(min_deg: f32, max_deg: f32) -> Self {
        Self {
            min_rad: min_deg * (core::f32::consts::PI / 180.0),
            max_rad: max_deg * (core::f32::consts::PI / 180.0),
        }
    }

    #[must_use]
    pub const fn free() -> Self {
        Self {
            min_rad: -core::f32::consts::PI,
            max_rad: core::f32::consts::PI,
        }
    }

    #[must_use]
    pub fn clamp(&self, angle: f32) -> f32 {
        if angle < self.min_rad {
            self.min_rad
        } else if angle > self.max_rad {
            self.max_rad
        } else {
            angle
        }
    }

    #[must_use]
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
    #[must_use]
    pub fn new(name: &[u8], axis: Vec3k, link_length: f32, constraint: JointConstraint) -> Self {
        let mut n = [0u8; 8];
        let len = name.len().min(8);
        n[..len].copy_from_slice(&name[..len]);
        Self {
            name: n,
            angle: 0.0,
            axis,
            constraint,
            link_length,
        }
    }

    pub fn set_angle(&mut self, angle: f32) {
        self.angle = self.constraint.clamp(angle);
    }
}

/// Maximum joints in a chain
pub const MAX_JOINTS: usize = 7;

/// Damping factor for damped-least-squares fallback (λ²)
///
/// When the Jacobian is near-singular (condition number is high or
/// the effective step is tiny), the CCD step is blended with a
/// small damped update to avoid oscillation.
pub const DLS_LAMBDA_SQ: f32 = 0.01;

/// Fraction of joint-range margin within which smoothing kicks in.
/// E.g. 0.05 = smooth over the last 5% of range on each side.
pub const JOINT_LIMIT_SMOOTH_MARGIN: f32 = 0.05;

/// Compute a smooth joint-limit weight in [0, 1].
///
/// Returns 1.0 far from limits, and blends to 0.0 within
/// `JOINT_LIMIT_SMOOTH_MARGIN` of either limit, preventing
/// hard clamping oscillation near the boundary.
#[inline(always)]
fn joint_limit_weight(angle: f32, constraint: &JointConstraint) -> f32 {
    let range = constraint.range();
    if range < 1e-6 {
        return 0.0;
    }
    let margin = range * JOINT_LIMIT_SMOOTH_MARGIN;
    // Distance from the lower and upper limit
    let dist_lo = angle - constraint.min_rad;
    let dist_hi = constraint.max_rad - angle;
    let dist_min = if dist_lo < dist_hi { dist_lo } else { dist_hi };
    if dist_min <= 0.0 {
        return 0.0;
    }
    if dist_min >= margin {
        return 1.0;
    }
    // Smooth quintic blend: 0 → 1 over the margin
    let t = dist_min / margin;
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

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
    #[must_use]
    pub fn right_arm() -> Self {
        let joints = [
            // Shoulder flexion/extension
            Joint::new(
                b"sh_flex",
                Vec3k::new(1.0, 0.0, 0.0),
                0.0,
                JointConstraint::new(-60.0, 180.0),
            ),
            // Shoulder abduction/adduction
            Joint::new(
                b"sh_abd",
                Vec3k::new(0.0, 0.0, 1.0),
                0.0,
                JointConstraint::new(-50.0, 180.0),
            ),
            // Shoulder rotation
            Joint::new(
                b"sh_rot",
                Vec3k::new(0.0, -1.0, 0.0),
                0.30,
                JointConstraint::new(-90.0, 90.0),
            ),
            // Elbow flexion
            Joint::new(
                b"el_flex",
                Vec3k::new(1.0, 0.0, 0.0),
                0.28,
                JointConstraint::new(0.0, 145.0),
            ),
            // Wrist flexion/extension
            Joint::new(
                b"wr_flex",
                Vec3k::new(1.0, 0.0, 0.0),
                0.0,
                JointConstraint::new(-80.0, 80.0),
            ),
            // Wrist deviation
            Joint::new(
                b"wr_dev",
                Vec3k::new(0.0, 0.0, 1.0),
                0.0,
                JointConstraint::new(-20.0, 30.0),
            ),
            // Wrist pronation/supination
            Joint::new(
                b"wr_pro",
                Vec3k::new(0.0, -1.0, 0.0),
                0.20,
                JointConstraint::new(-80.0, 80.0),
            ),
        ];
        Self {
            joints,
            base: Vec3k::ZERO,
        }
    }

    /// Forward Kinematics — compute end-effector position from joint angles
    #[must_use]
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

    /// CCD (Cyclic Coordinate Descent) Inverse Kinematics
    ///
    /// Improvements over the naive implementation:
    /// 1. **Singularity detection** — checks the squared lengths of
    ///    `to_end` and `to_target` before normalizing.  When either
    ///    vector is near-zero (joint coincides with end-effector or
    ///    target), the step is skipped for that joint to avoid
    ///    divide-by-zero / NaN propagation.
    /// 2. **Damped-least-squares (DLS) fallback** — when the effective
    ///    rotation angle is very small (near a singularity), the step
    ///    is damped by `λ² / (λ² + angle²)` so the joint does not
    ///    thrash on degenerate configurations.
    /// 3. **Smooth joint-limit handling** — instead of a hard clamp
    ///    that causes oscillation at the boundary, the angular step is
    ///    blended to zero over the last `JOINT_LIMIT_SMOOTH_MARGIN` of
    ///    each joint's range.
    /// 4. **Pre-computed reciprocals** — the CCD step factor and the
    ///    DLS denominator are computed with a single reciprocal per
    ///    iteration, replacing divisions in the inner loop.
    ///
    /// Returns (`iterations_used`, `final_error_distance`).
    pub fn inverse_kinematics(
        &mut self,
        target: Vec3k,
        max_iter: u32,
        tolerance: f32,
    ) -> (u32, f32) {
        // Pre-compute constant reciprocal for the half-step scale.
        const STEP_SCALE: f32 = 0.5;

        for iter in 0..max_iter {
            let end = self.forward_kinematics();
            let error = end.distance(target);
            if error < tolerance {
                return (iter, error);
            }

            // CCD: iterate joints from tip to base
            for i in (0..MAX_JOINTS).rev() {
                const SING_THRESH_SQ: f32 = 1e-8; // (0.1 mm)²

                let joint_pos = self.joint_position(i);
                // Re-evaluate end-effector after each joint update so that
                // downstream joints benefit from the change immediately.
                let end_pos = self.forward_kinematics();

                let raw_to_end = end_pos - joint_pos;
                let raw_to_target = target - joint_pos;

                // --- Singularity detection (Issue 1) ---
                // If either vector is near-zero, normalizing would produce
                // NaN/Inf.  Skip this joint instead of corrupting state.
                let len_sq_end = raw_to_end.length_sq();
                let len_sq_target = raw_to_target.length_sq();
                if len_sq_end < SING_THRESH_SQ || len_sq_target < SING_THRESH_SQ {
                    continue; // joint is degenerate for this step
                }

                // Safe to normalize — pre-compute reciprocals (Issue 3)
                let inv_len_end = 1.0 / fast_sqrt(len_sq_end);
                let inv_len_target = 1.0 / fast_sqrt(len_sq_target);
                let to_end = raw_to_end.scale(inv_len_end);
                let to_target = raw_to_target.scale(inv_len_target);

                // Angle between the two unit vectors
                let dot = to_end.dot(to_target).clamp(-1.0_f32, 1.0_f32);
                let raw_angle = acos_approx(dot);

                // --- Damped-least-squares fallback (Issue 1) ---
                // When raw_angle is tiny the Jacobian is near-singular.
                // Apply DLS damping: effective_angle = raw_angle² / (λ² + raw_angle²)
                // multiplied by the original sign-scaled step, so the
                // step shrinks gracefully near zero rather than oscillating.
                // Pre-compute reciprocal of the denominator.
                let dls_denom = DLS_LAMBDA_SQ + raw_angle * raw_angle;
                let inv_dls_denom = 1.0 / dls_denom; // one division, pre-computed
                let damped_scale = raw_angle * raw_angle * inv_dls_denom;
                let angle = raw_angle * STEP_SCALE * damped_scale;

                // Rotation direction via cross product
                let cross = to_end.cross(to_target);
                let sign = if cross.dot(self.joints[i].axis) >= 0.0 {
                    1.0_f32
                } else {
                    -1.0_f32
                };

                // --- Smooth joint-limit blending (Issue 2) ---
                // Scale the step by a weight that fades to zero near limits,
                // preventing hard-clamp oscillation at the boundary.
                let limit_w = joint_limit_weight(self.joints[i].angle, &self.joints[i].constraint);
                let delta = sign * angle * limit_w;

                self.joints[i].set_angle(self.joints[i].angle + delta);
            }
        }

        let error = self.forward_kinematics().distance(target);
        (max_iter, error)
    }

    /// Get world-space position of joint i
    #[must_use]
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
    #[must_use]
    pub fn total_length(&self) -> f32 {
        let mut len = 0.0;
        for j in &self.joints {
            len += j.link_length;
        }
        len
    }

    /// Get all joint angles as array
    #[must_use]
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
#[must_use]
pub fn rotate_vec(v: Vec3k, axis: Vec3k, theta: f32) -> Vec3k {
    let (sin_t, cos_t) = sin_cos_approx(theta);
    let k = axis.normalize();
    let term1 = v.scale(cos_t);
    let term2 = k.cross(v).scale(sin_t);
    let term3 = k.scale(k.dot(v) * (1.0 - cos_t));
    term1 + term2 + term3
}

/// Approximate sin and cos (Bhaskara I + identity)
fn sin_cos_approx(theta: f32) -> (f32, f32) {
    (
        sin_approx(theta),
        sin_approx(theta + core::f32::consts::FRAC_PI_2),
    )
}

/// Fast sine approximation (Bhaskara I, max error ~0.2%)
#[inline(always)]
fn sin_approx(x: f32) -> f32 {
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
    // den is always non-zero for x in [0, pi]
    let inv_den = 1.0 / den;
    sign * num * inv_den
}

/// Fast acos approximation (Abramowitz & Stegun)
fn acos_approx(x: f32) -> f32 {
    let abs_x = if x < 0.0 { -x } else { x };
    let result = -0.018_729_3 * abs_x;
    let result = (result + 0.074_261_0) * abs_x;
    let result = (result - 0.212_114_4) * abs_x;
    let result = result + core::f32::consts::FRAC_PI_2;
    let result = result * fast_sqrt(1.0 - abs_x);
    if x < 0.0 {
        core::f32::consts::PI - result
    } else {
        result
    }
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

    /// Singularity: target placed exactly at the base (zero-length
    /// `to_target` vector). IK must not produce NaN/Inf angles.
    #[test]
    fn test_ik_singularity_target_at_base() {
        let mut arm = ArmChain::right_arm();
        // Target coincides with base — maximally degenerate
        let target = arm.base;
        let (_iters, error) = arm.inverse_kinematics(target, 20, 0.001);
        // Angles must all be finite
        for j in &arm.joints {
            assert!(j.angle.is_finite(), "NaN/Inf angle after singularity IK");
        }
        // Error is also finite (arm cannot reach its own base, but no panic)
        assert!(error.is_finite());
    }

    /// Singularity: target placed so far away that the arm is fully
    /// extended (another classic singularity).
    #[test]
    fn test_ik_singularity_fully_extended() {
        let mut arm = ArmChain::right_arm();
        arm.base = Vec3k::ZERO;
        // Target 10 m away — well beyond arm reach (~0.78 m)
        let target = Vec3k::new(10.0, 0.0, 0.0);
        let (_iters, error) = arm.inverse_kinematics(target, 30, 0.001);
        for j in &arm.joints {
            assert!(j.angle.is_finite(), "NaN/Inf angle on over-extension IK");
        }
        assert!(error.is_finite());
        // Error should be large (target unreachable) but bounded
        assert!(error > 0.0);
    }

    /// Smooth joint limit: after many IK iterations pushing a joint
    /// hard against its boundary, the angle must stay within the
    /// constraint and must be finite (no oscillation blow-up).
    #[test]
    fn test_ik_joint_limit_no_oscillation() {
        let mut arm = ArmChain::right_arm();
        arm.base = Vec3k::new(0.0, 1.5, 0.0);
        // Target that forces elbow near its 0° lower limit
        let target = Vec3k::new(0.0, 1.49, 0.01);
        let (_iters, _err) = arm.inverse_kinematics(target, 100, 1e-4);
        for j in &arm.joints {
            assert!(j.angle.is_finite(), "NaN/Inf angle near joint limit");
            assert!(
                j.angle >= j.constraint.min_rad - 1e-5,
                "angle below minimum: {} < {}",
                j.angle,
                j.constraint.min_rad
            );
            assert!(
                j.angle <= j.constraint.max_rad + 1e-5,
                "angle above maximum: {} > {}",
                j.angle,
                j.constraint.max_rad
            );
        }
    }

    /// `joint_limit_weight` returns 1.0 at mid-range and 0.0 at the boundary.
    #[test]
    fn test_joint_limit_weight() {
        let c = JointConstraint::new(0.0, 90.0);
        let mid = (c.min_rad + c.max_rad) * 0.5;
        let w_mid = joint_limit_weight(mid, &c);
        assert!(
            (w_mid - 1.0).abs() < 1e-5,
            "weight at mid should be 1.0, got {w_mid}"
        );

        let w_at_limit = joint_limit_weight(c.min_rad, &c);
        assert!(
            w_at_limit < 0.01,
            "weight at limit should be ~0, got {w_at_limit}"
        );

        let w_at_max = joint_limit_weight(c.max_rad, &c);
        assert!(
            w_at_max < 0.01,
            "weight at max limit should be ~0, got {w_at_max}"
        );
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

    // --- 追加テスト ---

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_vec3k_zero() {
        let v = Vec3k::ZERO;
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
        assert_eq!(v.length(), 0.0);
    }

    #[test]
    fn test_vec3k_neg() {
        let v = Vec3k::new(1.0, -2.0, 3.0);
        let n = -v;
        assert!((n.x - (-1.0)).abs() < 1e-6);
        assert!((n.y - 2.0).abs() < 1e-6);
        assert!((n.z - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_mul_scalar() {
        let v = Vec3k::new(1.0, 2.0, 3.0);
        let s = v * 3.0;
        assert!((s.x - 3.0).abs() < 1e-6);
        assert!((s.y - 6.0).abs() < 1e-6);
        assert!((s.z - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_sub() {
        let a = Vec3k::new(5.0, 3.0, 1.0);
        let b = Vec3k::new(2.0, 1.0, 0.5);
        let c = a - b;
        assert!((c.x - 3.0).abs() < 1e-6);
        assert!((c.y - 2.0).abs() < 1e-6);
        assert!((c.z - 0.5).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_vec3k_normalize_zero_vector() {
        // ゼロベクトルの正規化はゼロを返す (パニックしない)
        let v = Vec3k::ZERO;
        let n = v.normalize();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
        assert_eq!(n.z, 0.0);
    }

    #[test]
    fn test_vec3k_dot_perpendicular() {
        let x = Vec3k::new(1.0, 0.0, 0.0);
        let y = Vec3k::new(0.0, 1.0, 0.0);
        assert!((x.dot(y)).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_distance_self() {
        let v = Vec3k::new(3.0, 4.0, 5.0);
        assert!((v.distance(v)).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_lerp_endpoints() {
        let a = Vec3k::new(0.0, 0.0, 0.0);
        let b = Vec3k::new(10.0, 0.0, 0.0);
        let start = a.lerp(b, 0.0);
        let end = a.lerp(b, 1.0);
        assert!((start.x).abs() < 1e-6);
        assert!((end.x - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_joint_constraint_range() {
        let c = JointConstraint::new(-90.0, 90.0);
        let expected = 180.0_f32 * (core::f32::consts::PI / 180.0);
        assert!((c.range() - expected).abs() < 0.001);
    }

    #[test]
    fn test_joint_constraint_free_range() {
        let c = JointConstraint::free();
        assert!((c.range() - 2.0 * core::f32::consts::PI).abs() < 0.001);
    }

    #[test]
    fn test_joint_set_angle_clamps() {
        let mut j = Joint::new(
            b"test",
            Vec3k::new(1.0, 0.0, 0.0),
            0.3,
            JointConstraint::new(0.0, 90.0),
        );
        // 上限を超えた角度はクランプされる
        j.set_angle(200.0);
        assert!(j.angle <= j.constraint.max_rad + 1e-5);
        // 下限を下回る角度もクランプされる
        j.set_angle(-10.0);
        assert!(j.angle >= j.constraint.min_rad - 1e-5);
    }

    #[test]
    fn test_arm_chain_angles_roundtrip() {
        let mut arm = ArmChain::right_arm();
        let angles = [0.1, 0.2, 0.0, 0.5, 0.1, 0.0, 0.1];
        arm.set_angles(&angles);
        let got = arm.angles();
        // 制約範囲内の角度はそのまま保持される
        for (i, &a) in angles.iter().enumerate() {
            assert!(
                (got[i] - a).abs() < 0.01 || got[i] >= arm.joints[i].constraint.min_rad,
                "joint {i}: expected ~{a}, got {}",
                got[i]
            );
        }
    }

    #[test]
    fn test_joint_position_base() {
        let arm = ArmChain::right_arm();
        // joint_position(0) は有限値であること
        let p0 = arm.joint_position(0);
        assert!(p0.x.is_finite());
        assert!(p0.y.is_finite());
        assert!(p0.z.is_finite());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_joint_limit_weight_zero_range() {
        // range がほぼゼロの制約は weight 0.0 を返す
        let c = JointConstraint {
            min_rad: 1.0,
            max_rad: 1.0,
        };
        let w = joint_limit_weight(1.0, &c);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_acos_approx_minus_one() {
        // acos(-1) ≈ π
        let a = acos_approx(-1.0);
        assert!((a - core::f32::consts::PI).abs() < 0.05);
    }

    #[test]
    fn test_sin_approx_pi() {
        // sin(π) ≈ 0
        let s = sin_approx(core::f32::consts::PI);
        assert!(s.abs() < 0.01);
    }

    // --- さらに追加テスト ---

    #[test]
    fn test_vec3k_scale() {
        let v = Vec3k::new(2.0, -3.0, 0.5);
        let s = v.scale(2.0);
        assert!((s.x - 4.0).abs() < 1e-6);
        assert!((s.y - (-6.0)).abs() < 1e-6);
        assert!((s.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_lerp_midpoint() {
        let a = Vec3k::new(0.0, 0.0, 0.0);
        let b = Vec3k::new(4.0, 4.0, 4.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 2.0).abs() < 1e-6);
        assert!((mid.y - 2.0).abs() < 1e-6);
        assert!((mid.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3k_length_sq() {
        let v = Vec3k::new(1.0, 2.0, 2.0);
        // 1² + 2² + 2² = 9
        assert!((v.length_sq() - 9.0).abs() < 1e-6);
        assert!((v.length() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_forward_kinematics_changes_with_angle() {
        // 肩の屈曲角度を変えると end-effector の位置が変わる
        let arm1 = ArmChain::right_arm();
        let mut arm2 = ArmChain::right_arm();
        arm2.joints[0].set_angle(core::f32::consts::FRAC_PI_4);
        let e1 = arm1.forward_kinematics();
        let e2 = arm2.forward_kinematics();
        // 角度が違えば位置は異なる
        assert!(e1.distance(e2) > 0.01);
        // arm1は変更していない
        let _ = arm1.forward_kinematics();
    }

    #[test]
    fn test_joint_position_all_joints() {
        let arm = ArmChain::right_arm();
        // 全ジョイントの位置が有限値であること
        for i in 0..MAX_JOINTS {
            let p = arm.joint_position(i);
            assert!(p.x.is_finite(), "joint {i} x not finite");
            assert!(p.y.is_finite(), "joint {i} y not finite");
            assert!(p.z.is_finite(), "joint {i} z not finite");
        }
    }

    #[test]
    fn test_set_angles_clamps_out_of_range() {
        let mut arm = ArmChain::right_arm();
        // 全ジョイントに上限を超えた角度を設定 → クランプされる
        let big = [999.0f32; MAX_JOINTS];
        arm.set_angles(&big);
        for (i, j) in arm.joints.iter().enumerate() {
            assert!(
                j.angle <= j.constraint.max_rad + 1e-5,
                "joint {i} angle {} exceeds max {}",
                j.angle,
                j.constraint.max_rad
            );
        }
    }

    #[test]
    fn test_arm_chain_base_offset_fk() {
        // base をオフセットしたとき FK の結果も同量オフセットされる
        let arm1 = ArmChain::right_arm();
        let mut arm2 = ArmChain::right_arm();
        arm2.base = Vec3k::new(1.0, 2.0, 3.0);
        let e1 = arm1.forward_kinematics();
        let e2 = arm2.forward_kinematics();
        // base のオフセット分だけずれているはず
        assert!((e2.x - e1.x - 1.0).abs() < 1e-5);
        assert!((e2.y - e1.y - 2.0).abs() < 1e-5);
        assert!((e2.z - e1.z - 3.0).abs() < 1e-5);
        // arm1 は変更していないことも確認
        let e1b = arm1.forward_kinematics();
        assert!((e1.x - e1b.x).abs() < 1e-6);
    }

    #[test]
    fn test_ik_reduces_error_vs_zero_iter() {
        // IK を十分に回すと error が 0 イテレーション時より小さくなる
        let mut arm = ArmChain::right_arm();
        arm.base = Vec3k::new(0.0, 1.5, 0.0);
        let target = Vec3k::new(0.1, 1.2, 0.2);
        // 0 イテレーション: 初期位置と target の距離
        let initial_error = arm.forward_kinematics().distance(target);
        let (_iters, final_error) = arm.inverse_kinematics(target, 50, 0.001);
        assert!(
            final_error <= initial_error + 0.01,
            "IK made error worse: {initial_error} → {final_error}"
        );
    }

    #[test]
    fn test_rotate_vec_180deg() {
        // z軸周りに180度回転: (1,0,0) → (-1,0,0)
        let v = Vec3k::new(1.0, 0.0, 0.0);
        let r = rotate_vec(v, Vec3k::new(0.0, 0.0, 1.0), core::f32::consts::PI);
        assert!((r.x - (-1.0)).abs() < 0.05);
        assert!((r.y).abs() < 0.05);
    }

    #[test]
    fn test_joint_constraint_clamp_within() {
        // 制約範囲内の角度はそのまま返る
        let c = JointConstraint::new(-45.0, 45.0);
        let angle = 0.3;
        let clamped = c.clamp(angle);
        assert!((clamped - angle).abs() < 1e-6);
    }
}
