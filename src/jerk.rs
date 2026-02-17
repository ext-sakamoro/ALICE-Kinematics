//! Jerk minimization — Flash-Hogan quintic trajectory fitting
//!
//! Given a sequence of observed positions (from IMU/camera),
//! fits a minimum-jerk trajectory and extracts the motion intent.
//!
//! Flash & Hogan (1985): "The Coordination of Arm Movements:
//! An Experimentally Confirmed Mathematical Model"
//!
//! License: AGPL-3.0 (encoder module)
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::predictor::QuinticCoeffs;

/// Maximum samples in the fitting buffer
pub const MAX_FIT_SAMPLES: usize = 64;

/// Observed sample from sensor
#[derive(Debug, Clone, Copy)]
pub struct MotionSample {
    /// Position (meters)
    pub pos: Vec3k,
    /// Timestamp (seconds from epoch or session start)
    pub time: f32,
}

/// Jerk minimization fitter
///
/// Accumulates sensor samples and fits a minimum-jerk trajectory.
/// When enough samples are collected and the motion is detected,
/// extracts intent parameters (target, duration).
pub struct JerkFitter {
    /// Sample ring buffer
    samples: [MotionSample; MAX_FIT_SAMPLES],
    /// Write index
    write_idx: usize,
    /// Number of valid samples
    count: usize,
    /// Minimum samples before fitting
    min_samples: usize,
}

impl JerkFitter {
    pub fn new() -> Self {
        Self {
            samples: [MotionSample { pos: Vec3k::ZERO, time: 0.0 }; MAX_FIT_SAMPLES],
            write_idx: 0,
            count: 0,
            min_samples: 8,
        }
    }

    /// Push a new sensor sample
    pub fn push(&mut self, sample: MotionSample) {
        self.samples[self.write_idx] = sample;
        self.write_idx = (self.write_idx + 1) % MAX_FIT_SAMPLES;
        if self.count < MAX_FIT_SAMPLES {
            self.count += 1;
        }
    }

    /// Number of samples in buffer
    pub fn sample_count(&self) -> usize {
        self.count
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.count = 0;
        self.write_idx = 0;
    }

    /// Get sample at index (0 = oldest)
    fn get_sample(&self, idx: usize) -> MotionSample {
        if self.count < MAX_FIT_SAMPLES {
            self.samples[idx]
        } else {
            self.samples[(self.write_idx + idx) % MAX_FIT_SAMPLES]
        }
    }

    /// Compute current velocity from recent samples (finite difference)
    pub fn estimate_velocity(&self) -> Vec3k {
        if self.count < 2 {
            return Vec3k::ZERO;
        }
        let s1 = self.get_sample(self.count - 2);
        let s2 = self.get_sample(self.count - 1);
        let dt = s2.time - s1.time;
        if dt < 1e-6 { return Vec3k::ZERO; }
        (s2.pos - s1.pos).scale(1.0 / dt)
    }

    /// Compute current acceleration from recent samples
    pub fn estimate_acceleration(&self) -> Vec3k {
        if self.count < 3 {
            return Vec3k::ZERO;
        }
        let s0 = self.get_sample(self.count - 3);
        let s1 = self.get_sample(self.count - 2);
        let s2 = self.get_sample(self.count - 1);

        let dt1 = s1.time - s0.time;
        let dt2 = s2.time - s1.time;
        if dt1 < 1e-6 || dt2 < 1e-6 { return Vec3k::ZERO; }

        let v1 = (s1.pos - s0.pos).scale(1.0 / dt1);
        let v2 = (s2.pos - s1.pos).scale(1.0 / dt2);
        let avg_dt = (dt1 + dt2) / 2.0;
        (v2 - v1).scale(1.0 / avg_dt)
    }

    /// Detect if motion has started (velocity exceeds threshold)
    pub fn motion_detected(&self, vel_threshold: f32) -> bool {
        self.estimate_velocity().length() > vel_threshold
    }

    /// Fit minimum-jerk trajectory to recent samples
    ///
    /// Returns (target_position, duration, fit_error)
    pub fn fit_trajectory(&self) -> Option<FitResult> {
        if self.count < self.min_samples {
            return None;
        }

        let first = self.get_sample(0);
        let last = self.get_sample(self.count - 1);
        let duration = last.time - first.time;
        if duration < 0.01 { return None; }

        // Estimate target using velocity extrapolation
        let vel = self.estimate_velocity();
        let speed = vel.length();

        // Predict remaining time using deceleration profile
        // For minimum-jerk, peak velocity is at t=0.5T, v_peak = 15*D/(8*T)
        // Current speed / peak speed gives rough progress estimate
        let displacement = last.pos - first.pos;
        let dist = displacement.length();
        if dist < 1e-4 { return None; }

        // Estimate total duration: D = v_peak * T * 8/15
        // At current progress, estimate remaining time
        let estimated_total_duration = duration * 2.0; // Simple heuristic
        let target = first.pos + displacement.normalize().scale(dist * 2.0);

        // Fit quintic per-axis and compute residual error
        let fit_x = QuinticCoeffs::from_boundary(
            first.pos.x, 0.0, 0.0, target.x, estimated_total_duration);
        let fit_y = QuinticCoeffs::from_boundary(
            first.pos.y, 0.0, 0.0, target.y, estimated_total_duration);
        let fit_z = QuinticCoeffs::from_boundary(
            first.pos.z, 0.0, 0.0, target.z, estimated_total_duration);

        // Compute fit error
        let mut total_error = 0.0f32;
        for i in 0..self.count {
            let s = self.get_sample(i);
            let t = s.time - first.time;
            let pred = Vec3k::new(fit_x.position(t), fit_y.position(t), fit_z.position(t));
            let err = pred.distance(s.pos);
            total_error += err * err;
        }
        let rmse = fast_sqrt_jerk(total_error / self.count as f32);

        Some(FitResult {
            target,
            start: first.pos,
            duration: estimated_total_duration,
            start_velocity: vel,
            fit_error: rmse,
            speed,
        })
    }
}

/// Result of trajectory fitting
#[derive(Debug, Clone, Copy)]
pub struct FitResult {
    /// Predicted target position
    pub target: Vec3k,
    /// Start position
    pub start: Vec3k,
    /// Estimated total movement duration (seconds)
    pub duration: f32,
    /// Velocity at fit time
    pub start_velocity: Vec3k,
    /// RMS fit error (meters)
    pub fit_error: f32,
    /// Current speed (m/s)
    pub speed: f32,
}

/// Fast sqrt for jerk module
fn fast_sqrt_jerk(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5 - half * y * y);
    let y = y * (1.5 - half * y * y);
    x * y
}

/// Compute the jerk integral for a quintic trajectory
///
/// J = ∫₀ᵀ |x'''(t)|² dt
///
/// For minimum-jerk trajectory: J = 720 * D² / T⁵
/// where D = displacement, T = duration
pub fn minimum_jerk_cost(displacement: f32, duration: f32) -> f32 {
    if duration < 1e-6 { return f32::MAX; }
    let t5 = duration * duration * duration * duration * duration;
    720.0 * displacement * displacement / t5
}

/// Compute the speed-accuracy tradeoff (Fitts' Law)
///
/// MT = a + b * log2(2D/W)
/// where MT = movement time, D = distance, W = target width
pub fn fitts_law_duration(distance: f32, target_width: f32, a: f32, b: f32) -> f32 {
    if target_width < 1e-6 { return a + b * 10.0; }
    let id = log2_approx(2.0 * distance / target_width);
    a + b * id
}

/// Fast log2 approximation
fn log2_approx(x: f32) -> f32 {
    if x <= 0.0 { return -10.0; }
    let bits = f32::to_bits(x);
    let exp = ((bits >> 23) & 0xFF) as f32 - 127.0;
    let frac = f32::from_bits((bits & 0x007FFFFF) | 0x3F800000) - 1.0;
    exp + frac * (1.0 - 0.3333 * frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jerk_fitter_push() {
        let mut fitter = JerkFitter::new();
        for i in 0..10 {
            fitter.push(MotionSample {
                pos: Vec3k::new(i as f32 * 0.01, 0.0, 0.0),
                time: i as f32 * 0.001,
            });
        }
        assert_eq!(fitter.sample_count(), 10);
    }

    #[test]
    fn test_estimate_velocity() {
        let mut fitter = JerkFitter::new();
        // Constant velocity: 1 m/s along x
        for i in 0..10 {
            fitter.push(MotionSample {
                pos: Vec3k::new(i as f32 * 0.01, 0.0, 0.0),
                time: i as f32 * 0.01,
            });
        }
        let vel = fitter.estimate_velocity();
        assert!((vel.x - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_motion_detected() {
        let mut fitter = JerkFitter::new();
        fitter.push(MotionSample { pos: Vec3k::ZERO, time: 0.0 });
        fitter.push(MotionSample { pos: Vec3k::new(0.1, 0.0, 0.0), time: 0.01 });
        assert!(fitter.motion_detected(0.5)); // 10 m/s > 0.5 threshold
    }

    #[test]
    fn test_fit_trajectory() {
        let mut fitter = JerkFitter::new();
        // Generate a straight-line reaching motion
        for i in 0..16 {
            let t = i as f32 * 0.01;
            fitter.push(MotionSample {
                pos: Vec3k::new(t * 2.0, 0.0, 0.0),
                time: t,
            });
        }
        let result = fitter.fit_trajectory();
        assert!(result.is_some());
        let fit = result.unwrap();
        assert!(fit.target.x > 0.0);
        assert!(fit.duration > 0.0);
    }

    #[test]
    fn test_minimum_jerk_cost() {
        let j1 = minimum_jerk_cost(1.0, 1.0);
        let j2 = minimum_jerk_cost(1.0, 0.5);
        // Faster motion → higher jerk cost
        assert!(j2 > j1);
    }

    #[test]
    fn test_fitts_law() {
        let mt1 = fitts_law_duration(0.1, 0.02, 0.0, 0.1);
        let mt2 = fitts_law_duration(0.5, 0.02, 0.0, 0.1);
        // Farther distance → longer duration
        assert!(mt2 > mt1);
    }

    #[test]
    fn test_log2_approx() {
        let l = log2_approx(8.0);
        assert!((l - 3.0).abs() < 0.1);
        let l = log2_approx(1.0);
        assert!(l.abs() < 0.1);
    }

    #[test]
    fn test_fitter_ring_buffer_wrap() {
        let mut fitter = JerkFitter::new();
        // Fill beyond capacity
        for i in 0..100 {
            fitter.push(MotionSample {
                pos: Vec3k::new(i as f32 * 0.001, 0.0, 0.0),
                time: i as f32 * 0.001,
            });
        }
        assert_eq!(fitter.sample_count(), MAX_FIT_SAMPLES);
    }

    #[test]
    fn test_fitter_clear() {
        let mut fitter = JerkFitter::new();
        fitter.push(MotionSample { pos: Vec3k::ZERO, time: 0.0 });
        fitter.clear();
        assert_eq!(fitter.sample_count(), 0);
    }

    #[test]
    fn test_estimate_acceleration() {
        let mut fitter = JerkFitter::new();
        // Constant acceleration: a = 2 m/s²
        for i in 0..10 {
            let t = i as f32 * 0.01;
            fitter.push(MotionSample {
                pos: Vec3k::new(t * t, 0.0, 0.0), // x = t²
                time: t,
            });
        }
        let acc = fitter.estimate_acceleration();
        // d²(t²)/dt² = 2
        assert!((acc.x - 2.0).abs() < 1.0);
    }
}
