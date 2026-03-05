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

use crate::intent::Intent;
use crate::joint::Vec3k;

/// Quintic polynomial coefficients for one axis
///
/// `x(t) = c[0] + c[1]*t + c[2]*t² + c[3]*t³ + c[4]*t⁴ + c[5]*t⁵`
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
    #[must_use]
    pub fn from_boundary(x0: f32, v0: f32, a0: f32, xf: f32, duration: f32) -> Self {
        let t = duration;
        if t < 1e-6 {
            return Self {
                c: [xf, 0.0, 0.0, 0.0, 0.0, 0.0],
            };
        }
        let t2 = t * t;

        // Precompute reciprocals to replace repeated divisions
        let inv_t = 1.0 / t;
        let inv_t2 = inv_t * inv_t;
        let inv_t3 = inv_t2 * inv_t;
        let inv_t4 = inv_t3 * inv_t;
        let inv_t5 = inv_t4 * inv_t;

        // Boundary conditions: vf = 0, af = 0 (rest-to-rest or moving-to-rest)
        let c0 = x0;
        let c1 = v0;
        let c2 = a0 * 0.5;

        // Solve for c3, c4, c5 from endpoint conditions
        let dx = xf - x0 - v0 * t - c2 * t2;
        let c3 = 10.0 * dx * inv_t3 - (4.0 * v0 + a0 * t) * inv_t2 + a0 * 0.5 * inv_t;
        let c4 = -15.0 * dx * inv_t4 + (7.0 * v0 + 2.0 * a0 * t) * inv_t3 - a0 * inv_t2;
        let c5 = 6.0 * dx * inv_t5 - (3.0 * v0 + a0 * t) * inv_t4 + a0 * 0.5 * inv_t3;

        Self {
            c: [c0, c1, c2, c3, c4, c5],
        }
    }

    /// Evaluate position at time t
    #[inline(always)]
    #[must_use]
    pub fn position(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        self.c[0]
            + self.c[1] * t
            + self.c[2] * t2
            + self.c[3] * t3
            + self.c[4] * t4
            + self.c[5] * t5
    }

    /// Evaluate velocity at time t
    #[inline(always)]
    #[must_use]
    pub fn velocity(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        self.c[1]
            + 2.0 * self.c[2] * t
            + 3.0 * self.c[3] * t2
            + 4.0 * self.c[4] * t3
            + 5.0 * self.c[5] * t4
    }

    /// Evaluate acceleration at time t
    #[inline(always)]
    #[must_use]
    pub fn acceleration(&self, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        2.0 * self.c[2] + 6.0 * self.c[3] * t + 12.0 * self.c[4] * t2 + 20.0 * self.c[5] * t3
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

impl Default for Predictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Predictor {
    #[must_use]
    pub const fn new() -> Self {
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
            self.position.x,
            self.velocity.x,
            self.acceleration.x,
            intent.target.x,
            dur,
        );
        self.traj_y = QuinticCoeffs::from_boundary(
            self.position.y,
            self.velocity.y,
            self.acceleration.y,
            intent.target.y,
            dur,
        );
        self.traj_z = QuinticCoeffs::from_boundary(
            self.position.z,
            self.velocity.z,
            self.acceleration.z,
            intent.target.z,
            dur,
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
        let t = if self.elapsed > self.duration {
            self.duration
        } else {
            self.elapsed
        };

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
    #[must_use]
    pub fn position_at(&self, t: f32) -> Vec3k {
        let t = if t > self.duration {
            self.duration
        } else if t < 0.0 {
            0.0
        } else {
            t
        };
        Vec3k::new(
            self.traj_x.position(t),
            self.traj_y.position(t),
            self.traj_z.position(t),
        )
    }

    /// Get predicted velocity at a specific time
    #[must_use]
    pub fn velocity_at(&self, t: f32) -> Vec3k {
        let t = if t > self.duration {
            self.duration
        } else if t < 0.0 {
            0.0
        } else {
            t
        };
        Vec3k::new(
            self.traj_x.velocity(t),
            self.traj_y.velocity(t),
            self.traj_z.velocity(t),
        )
    }

    /// Is the predictor actively interpolating?
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Remaining time in current trajectory
    #[must_use]
    pub fn remaining(&self) -> f32 {
        if self.active {
            self.duration - self.elapsed
        } else {
            0.0
        }
    }

    /// Fraction of trajectory completed [0, 1]
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.duration < 1e-6 {
            return 1.0;
        }
        let inv_dur = 1.0 / self.duration;
        let p = self.elapsed * inv_dur;
        if p > 1.0 {
            1.0
        } else {
            p
        }
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
        for _ in 0..100 {
            pred.update(0.001);
        }

        // Second motion: 1 → 2 (continues from current state)
        let intent2 = Intent::reach(Vec3k::new(2.0, 0.0, 0.0), 100);
        pred.apply_intent(intent2);
        for _ in 0..100 {
            pred.update(0.001);
        }

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
        for _ in 0..100 {
            pred.update(0.001);
        }
        assert!((pred.remaining() - 0.1).abs() < 0.01);
    }

    // --- 追加テスト ---

    #[test]
    fn test_quintic_zero_duration() {
        // 時間ゼロのとき c[0] = xf を返す (パニックしない)
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 5.0, 0.0);
        assert!((q.position(0.0) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_quintic_acceleration_at_endpoints() {
        // rest-to-rest では端点の加速度はゼロ
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        assert!((q.acceleration(0.0)).abs() < 0.001);
        assert!((q.acceleration(1.0)).abs() < 0.05);
    }

    #[test]
    fn test_quintic_position_monotone() {
        // rest-to-rest (正の変位) では位置は単調増加
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        let mut prev = q.position(0.0);
        for i in 1..=10 {
            let t = i as f32 * 0.1;
            let cur = q.position(t);
            assert!(
                cur >= prev - 1e-4,
                "not monotone at t={t}: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_predictor_new_inactive() {
        // 初期状態では inactive でポジション ZERO
        let pred = Predictor::new();
        assert!(!pred.is_active());
        assert_eq!(pred.remaining(), 0.0);
    }

    #[test]
    fn test_predictor_progress_complete() {
        // 完了後は progress が 1.0
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 50);
        pred.apply_intent(intent);
        for _ in 0..60 {
            pred.update(0.001);
        }
        assert!((pred.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_predictor_remaining_zero_after_completion() {
        // 完了後は remaining が 0.0
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 50);
        pred.apply_intent(intent);
        for _ in 0..60 {
            pred.update(0.001);
        }
        assert!((pred.remaining()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_predictor_velocity_at_clamps_negative() {
        // 負の時間は 0.0 にクランプされる
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 200);
        pred.apply_intent(intent);
        let v_neg = pred.velocity_at(-1.0);
        let v_zero = pred.velocity_at(0.0);
        assert!((v_neg.x - v_zero.x).abs() < 1e-6);
    }

    #[test]
    fn test_predictor_position_at_clamps_over() {
        // 時間が duration を超えたら duration の位置に固定
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(2.0, 0.0, 0.0), 100);
        pred.apply_intent(intent);
        let p_end = pred.position_at(0.1);
        let p_over = pred.position_at(100.0);
        assert!((p_end.x - p_over.x).abs() < 1e-5);
    }

    #[test]
    fn test_quintic_nonzero_start_velocity() {
        // 初速あり: t=0 での velocity は v0 と一致
        let v0 = 2.0;
        let q = QuinticCoeffs::from_boundary(0.0, v0, 0.0, 3.0, 1.0);
        assert!((q.velocity(0.0) - v0).abs() < 0.001);
    }

    #[test]
    fn test_quintic_nonzero_start_acceleration() {
        // 初期加速度あり: t=0 での acceleration は a0 と一致
        let a0 = 4.0;
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, a0, 1.0, 1.0);
        assert!((q.acceleration(0.0) - a0).abs() < 0.001);
    }

    #[test]
    fn test_predictor_progress_zero_duration() {
        // duration = 0 のとき progress は 1.0 を返す
        let mut pred = Predictor::new();
        // duration_ms = 0 → duration_secs() = 0.0
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 0);
        pred.apply_intent(intent);
        assert!((pred.progress() - 1.0).abs() < 0.01);
    }

    // --- さらに追加テスト ---

    #[test]
    fn test_predictor_default_equals_new() {
        // Default トレイトは Predictor::new() と同じ状態を返す
        let p1 = Predictor::new();
        let p2 = Predictor::default();
        assert!(!p1.is_active());
        assert!(!p2.is_active());
        assert!((p1.remaining() - p2.remaining()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quintic_velocity_positive_at_midpoint() {
        // rest-to-rest (正の変位) では t=0.5 における速度が正
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        let v_mid = q.velocity(0.5);
        assert!(v_mid > 0.0, "midpoint velocity should be positive: {v_mid}");
    }

    #[test]
    fn test_quintic_acceleration_nonzero_mid() {
        // 加速フェーズ中は加速度が非ゼロ
        let q = QuinticCoeffs::from_boundary(0.0, 0.0, 0.0, 1.0, 1.0);
        let a_quarter = q.acceleration(0.25);
        // 加速フェーズなので正の値になる
        assert!(
            a_quarter > 0.0,
            "acceleration at t=0.25 should be positive: {a_quarter}"
        );
    }

    #[test]
    fn test_predictor_velocity_continuous_on_chain() {
        // チェーンされた intent 間で速度が急激に変化しない
        let mut pred = Predictor::new();

        // 1本目: 0 → (1,0,0), 100ms
        let intent1 = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 100);
        pred.apply_intent(intent1);
        for _ in 0..100 {
            pred.update(0.001);
        }
        let vel_before = pred.velocity.x;

        // 2本目: (1,0,0) → (2,0,0), 100ms
        let intent2 = Intent::reach(Vec3k::new(2.0, 0.0, 0.0), 100);
        pred.apply_intent(intent2);
        let vel_after = pred.velocity_at(0.0).x;

        // apply_intent 直後の速度は前の終端速度と連続（ゼロ付近）
        assert!(
            (vel_after - vel_before).abs() < 0.5,
            "velocity discontinuity: {vel_before} → {vel_after}"
        );
    }

    #[test]
    fn test_predictor_position_at_negative_clamped_to_start() {
        // 負の時間 → t=0 の位置が返る
        let mut pred = Predictor::new();
        let intent = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 200);
        pred.apply_intent(intent);
        let p_neg = pred.position_at(-5.0);
        let p_zero = pred.position_at(0.0);
        assert!((p_neg.x - p_zero.x).abs() < 1e-6);
        assert!((p_neg.y - p_zero.y).abs() < 1e-6);
    }

    #[test]
    fn test_predictor_update_reaches_target_exactly() {
        // 十分な時間 update() を呼ぶと位置が target に収束
        let mut pred = Predictor::new();
        let target = Vec3k::new(3.0, -1.0, 0.5);
        let intent = Intent::reach(target, 100);
        pred.apply_intent(intent);
        // 余分に 50 ステップ追加で確実に完了させる
        for _ in 0..150 {
            pred.update(0.001);
        }
        assert!(
            pred.position.distance(target) < 0.02,
            "distance from target: {}",
            pred.position.distance(target)
        );
    }
}
