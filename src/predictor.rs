//! Predictor — trajectory reconstruction from intent (decoder side)
//!
//! Receives an Intent packet and reconstructs a smooth, biomechanically
//! plausible trajectory using quintic polynomial interpolation.
//! This is the "free decoder" — MIT licensed.
//!
//! The quintic polynomial x(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
//! is the analytical solution to jerk minimization with boundary conditions:
//! - Start: position, velocity, acceleration (from previous state)
//! - End: target position, zero velocity, zero acceleration
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::intent::Intent;

/// Quintic polynomial coefficients for one axis
///
/// x(t) = c[0] + c[1]*t + c[2]*t² + c[3]*t³ + c[4]*t⁴ + c[5]*t⁵
/// where t is normalized to [0, 1]
#[derive(Debug, Clone, Copy)]
pub struct QuinticCoeffs {
    pub c: [f32; 6],
}

impl QuinticCoeffs {
    /// Compute quintic coefficients from boundary conditions
    ///
    /// Given: start position x0, start velocity v0, start acceleration a0
    ///        end position xf, end velocity vf=0, end acceleration af=0
    ///        movement duration T
    pub fn from_boundary(x0: f32, v0: f32, a0: f32, xf: f32, duration: f32) -> Self {
        let t = duration;
        if t < 1e-6 {
            return Self { c: [xf, 0.0, 0.0, 0.0, 0.0, 0.0] };
        }
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;

        // Boundary conditions: vf = 0, af = 0 (rest-to-rest or moving-to-rest)
        let c0 = x0;
        let c1 = v0;
        let c2 = a0 / 2.0;

        // Solve for c3, c4, c5 from endpoint conditions
        let dx = xf - x0 - v0 * t - (a0 / 2.0) * t2;
        let c3 = (10.0 * dx) / t3 - (4.0 * v0 + a0 * t) / t2 + a0 / (2.0 * t);
        let c4 = (-15.0 * dx) / t4 + (7.0 * v0 + 2.0 * a0 * t) / t3 - a0 / t2;
        let c5 = (6.0 * dx) / t5 - (3.0 * v0 + a0 * t) / t4 + a0 / (2.0 * t3);

        // Simplify: for rest-to-rest (v0=0, a0=0):
        // c3 = 10*dx/T³, c4 = -15*dx/T⁴, c5 = 6*dx/T⁵
        Self { c: [c0, c1, c2, c3, c4, c5] }
    }

    /// Evaluate position at time t
    pub fn position(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        self.c[0] + self.c[1] * t + self.c[2] * t2
            + self.c[3] * t3 + self.c[4] * t4 + self.c[5] * t5
    }

    /// Evaluate velocity at time t
    pub fn velocity(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        self.c[1] + 2.0 * self.c[2] * t + 3.0 * self.c[3] * t2
            + 4.0 * self.c[4] * t3 + 5.0 * self.c[5] * t4
    }

    /// Evaluate acceleration at time t
    pub fn acceleration(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        2.0 * self.c[2] + 6.0 * self.c[3] * t
            + 12.0 * self.c[4] * t2 + 20.0 * self.c[5] * t3
    }
}

/// Trajectory predictor — reconstructs full motion from intent
///
/// Maintains current state and generates smooth interpolation
/// between intents using quintic (minimum-jerk) polynomials.
///
/// Size: ~128 bytes
pub struct Predictor {
    /// Current position
    pub position: Vec3k,
    /// Current velocity
    pub velocity: Vec3k,
    /// Current acceleration
    pub acceleration: Vec3k,
    /// Active trajectory (per-axis quintic)
    traj_x: QuinticCoeffs,
    traj_y: QuinticCoeffs,
    traj_z: QuinticCoeffs,
    /// Time elapsed since last intent
    elapsed: f32,
    /// Duration of current trajectory
    duration: f32,
    /// Is trajectory active?
    active: bool,
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            position: Vec3k::ZERO,
            velocity: Vec3k::ZERO,
            acceleration: Vec3k::ZERO,
            traj_x: QuinticCoeffs { c: [0.0; 6] },
            traj_y: QuinticCoeffs { c: [0.0; 6] },
            traj_z: QuinticCoeffs { c: [0.0; 6] },
            elapsed: 0.0,
            duration: 0.0,
            active: false,
        }
    }

    /// Apply a new intent — compute quintic trajectory to target
    pub fn apply_intent(&mut self, intent: Intent) {
        let dur = intent.duration_secs();
        self.traj_x = QuinticCoeffs::from_boundary(
            self.position.x, self.velocity.x, self.acceleration.x,
            intent.target.x, dur,
        );
        self.traj_y = QuinticCoeffs::from_boundary(
            self.position.y, self.velocity.y, self.acceleration.y,
            intent.target.y, dur,
        );
        self.traj_z = QuinticCoeffs::from_boundary(
            self.position.z, self.velocity.z, self.acceleration.z,
            intent.target.z, dur,
        );
        self.duration = dur;
        self.elapsed = 0.0;
        self.active = true;
    }

    /// Advance time and update position/velocity/acceleration
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.elapsed += dt;
        let t = if self.elapsed > self.duration { self.duration } else { self.elapsed };

        self.position = Vec3k::new(
            self.traj_x.position(t),
            self.traj_y.position(t),
            self.traj_z.position(t),
        );
        self.velocity = Vec3k::new(
            self.traj_x.velocity(t),
            self.traj_y.velocity(t),
            self.traj_z.velocity(t),
        );
        self.acceleration = Vec3k::new(
            self.traj_x.acceleration(t),
            self.traj_y.acceleration(t),
            self.traj_z.acceleration(t),
        );

        if self.elapsed >= self.duration {
            self.active = false;
        }
    }

    /// Get predicted position at a specific time offset from intent start
    pub fn position_at(&self, t: f32) -> Vec3k {
        let t = if t > self.duration { self.duration } else if t < 0.0 { 0.0 } else { t };
        Vec3k::new(
            self.traj_x.position(t),
            self.traj_y.position(t),
            self.traj_z.position(t),
        )
    }

    /// Get predicted velocity at a specific time
    pub fn velocity_at(&self, t: f32) -> Vec3k {
        let t = if t > self.duration { self.duration } else if t < 0.0 { 0.0 } else { t };
        Vec3k::new(
            self.traj_x.velocity(t),
            self.traj_y.velocity(t),
            self.traj_z.velocity(t),
        )
    }

    /// Is the predictor actively interpolating?
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Remaining time in current trajectory
    pub fn remaining(&self) -> f32 {
        if self.active { self.duration - self.elapsed } else { 0.0 }
    }

    /// Fraction of trajectory completed [0, 1]
    pub fn progress(&self) -> f32 {
        if self.duration < 1e-6 { return 1.0; }
        let p = self.elapsed / self.duration;
        if p > 1.0 { 1.0 } else { p }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quintic_rest_to_rest() {
        // Simple rest-to-rest: 0 → 1.0 over 1 second
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        // At t=0: should be 0
        assert!((q.position(0.0)).abs() < 0.001);
        // At t=1: should be 1.0
        assert!((q.position(1.0) - 1.0).abs() < 0.001);
        // At t=0.5: should be ~0.5 (symmetric)
        assert!((q.position(0.5) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_quintic_velocity_at_endpoints() {
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        // Velocity at start = 0 (rest)
        assert!((q.velocity(0.0)).abs() < 0.001);
        // Velocity at end = 0 (rest)
        assert!((q.velocity(1.0)).abs() < 0.01);
    }

    #[test]
    fn test_quintic_bell_velocity() {
        // Velocity should be bell-shaped (peak at midpoint)
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        let v_mid = q.velocity(0.5);
        let v_quarter = q.velocity(0.25);
        assert!(v_mid > v_quarter);
    }

    #[test]
    fn test_predictor_reach_target() {
        let mut pred = Predictor::new();
        let target = Vec3k::new(1.0, 0.5, 0.0);
        let intent = Intent::reach(target, 200); // 200ms
        pred.apply_intent(intent);

        // Simulate at 1ms steps
        for _ in 0..200 {
            pred.update(0.001);
        }

        assert!((pred.position.x - 1.0).abs() < 0.02);
        assert!((pred.position.y - 0.5).abs() < 0.02);
        assert!(!pred.is_active());
    }

    #[test]
    fn test_predictor_velocity_zero_at_end() {
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 100);
        pred.apply_intent(intent);

        for _ in 0..100 {
            pred.update(0.001);
        }

        assert!((pred.velocity.x).abs() < 0.1);
    }

    #[test]
    fn test_predictor_progress() {
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 100);
        pred.apply_intent(intent);

        assert!((pred.progress() - 0.0).abs() < 0.01);
        for _ in 0..50 {
            pred.update(0.001);
        }
        assert!((pred.progress() - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_predictor_position_at() {
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(2.0, 0.0, 0.0), 200);
        pred.apply_intent(intent);

        let mid = pred.position_at(0.1); // halfway
        assert!(mid.x > 0.0 && mid.x < 2.0);
    }

    #[test]
    fn test_predictor_chained_intents() {
        let mut pred = Predictor::new();

        // First motion: 0 → 1
        let intent1 = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 100);
        pred.apply_intent(intent1);
        for _ in 0..100 { pred.update(0.001); }

        // Second motion: 1 → 2 (continues from current state)
        let intent2 = Intent::reach(Vec3k::new(2.0, 0.0, 0.0), 100);
        pred.apply_intent(intent2);
        for _ in 0..100 { pred.update(0.001); }

        assert!((pred.position.x - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_quintic_moving_start() {
        // Start with velocity: v0 = 1.0
        let q = QuinticCoeffs::from_boundary(0.0, 1.0, 0.0, 1.0, 1.0);
        assert!((q.position(0.0)).abs() < 0.001);
        assert!((q.position(1.0) - 1.0).abs() < 0.01);
        assert!((q.velocity(0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_predictor_inactive_no_update() {
        let mut pred = Predictor::new();
        pred.position = Vec3k::new(5.0, 0.0, 0.0);
        pred.update(0.1); // No active trajectory
        assert!((pred.position.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_predictor_remaining() {
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 200);
        pred.apply_intent(intent);
        assert!((pred.remaining() - 0.2).abs() < 0.001);
        for _ in 0..100 { pred.update(0.001); }
        assert!((pred.remaining() - 0.1).abs() < 0.01);
    }
}
