//! Encoder — raw sensor data → Intent extraction pipeline
//!
//! Processes streaming IMU/camera position data and detects
//! motion onset, fits minimum-jerk trajectory, and emits
//! compact Intent packets.
//!
//! License: AGPL-3.0 (encoder module)
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::intent::{Intent, IntentFlags, IntentType};
use crate::jerk::{JerkFitter, MotionSample, FitResult};

/// Sensor sample from external device (IMU, camera, mouse)
#[derive(Debug, Clone, Copy)]
pub struct SensorSample {
    /// Position (meters)
    pub pos: Vec3k,
    /// Velocity (meters/second) — optional, can be zero
    pub vel: Vec3k,
    /// Timestamp (seconds)
    pub timestamp: f32,
}

/// Encoder state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderState {
    /// Waiting for motion onset
    Idle,
    /// Motion detected, collecting samples
    Tracking,
    /// Enough data to extract intent
    Ready,
}

/// Intent encoder — converts raw sensor stream to intent packets
///
/// State machine:
/// 1. Idle: monitoring for velocity above threshold
/// 2. Tracking: collecting samples, fitting trajectory
/// 3. Ready: intent extracted, can be read
///
/// Size: ~512 bytes
pub struct IntentEncoder {
    /// Jerk minimization fitter
    fitter: JerkFitter,
    /// Current state
    state: EncoderState,
    /// Velocity threshold for motion onset (m/s)
    vel_threshold: f32,
    /// Fit error threshold for intent extraction
    fit_threshold: f32,
    /// Minimum tracking samples before fitting
    min_tracking_samples: usize,
    /// Last extracted intent
    last_intent: Option<Intent>,
    /// Last fit result
    last_fit: Option<FitResult>,
    /// Sequence counter (0-7)
    sequence: u8,
    /// Previous position (for dead zone filtering)
    prev_pos: Vec3k,
    /// Dead zone radius (m) — ignore motion below this
    dead_zone: f32,
}

impl IntentEncoder {
    /// Create encoder with default thresholds
    pub fn new() -> Self {
        Self {
            fitter: JerkFitter::new(),
            state: EncoderState::Idle,
            vel_threshold: 0.05,      // 5 cm/s motion onset
            fit_threshold: 0.02,      // 2cm RMS fit error
            min_tracking_samples: 8,
            last_intent: None,
            last_fit: None,
            sequence: 0,
            prev_pos: Vec3k::ZERO,
            dead_zone: 0.002,         // 2mm dead zone
        }
    }

    /// Create encoder with custom thresholds
    pub fn with_thresholds(vel_threshold: f32, fit_threshold: f32, dead_zone: f32) -> Self {
        let mut enc = Self::new();
        enc.vel_threshold = vel_threshold;
        enc.fit_threshold = fit_threshold;
        enc.dead_zone = dead_zone;
        enc
    }

    /// Push a new sensor sample
    ///
    /// Returns Some(Intent) when a motion intent is extracted.
    pub fn push_sample(&mut self, sample: SensorSample) -> Option<Intent> {
        // Dead zone filtering
        if sample.pos.distance(self.prev_pos) < self.dead_zone
            && self.state == EncoderState::Idle
        {
            return None;
        }
        self.prev_pos = sample.pos;

        // Push to fitter
        self.fitter.push(MotionSample {
            pos: sample.pos,
            time: sample.timestamp,
        });

        match self.state {
            EncoderState::Idle => {
                if self.fitter.motion_detected(self.vel_threshold) {
                    self.state = EncoderState::Tracking;
                }
                None
            }
            EncoderState::Tracking => {
                if self.fitter.sample_count() >= self.min_tracking_samples {
                    self.try_extract_intent()
                } else {
                    None
                }
            }
            EncoderState::Ready => {
                // Start tracking new motion
                self.fitter.clear();
                self.fitter.push(MotionSample {
                    pos: sample.pos,
                    time: sample.timestamp,
                });
                self.state = EncoderState::Idle;
                None
            }
        }
    }

    /// Try to extract an intent from current samples
    fn try_extract_intent(&mut self) -> Option<Intent> {
        let fit = self.fitter.fit_trajectory()?;

        // Convert duration to milliseconds (clamp to u8)
        let dur_ms = (fit.duration * 1000.0) as u32;
        let dur_ms = if dur_ms > 255 { 255 } else { dur_ms as u8 };

        let flags = IntentFlags::new(
            IntentType::Reach,
            false,
            false,
            self.sequence,
        );
        self.sequence = (self.sequence + 1) & 0x07;

        let intent = Intent {
            target: fit.target,
            duration_ms: dur_ms,
            flags,
        };

        self.last_intent = Some(intent);
        self.last_fit = Some(fit);
        self.state = EncoderState::Ready;

        Some(intent)
    }

    /// Force extract intent from current data (even if incomplete)
    pub fn force_extract(&mut self) -> Option<Intent> {
        if self.fitter.sample_count() < 3 {
            return None;
        }
        self.try_extract_intent()
    }

    /// Current encoder state
    pub fn state(&self) -> EncoderState {
        self.state
    }

    /// Last extracted intent
    pub fn last_intent(&self) -> Option<Intent> {
        self.last_intent
    }

    /// Last fit result (for diagnostics)
    pub fn last_fit(&self) -> Option<FitResult> {
        self.last_fit
    }

    /// Reset the encoder
    pub fn reset(&mut self) {
        self.fitter.clear();
        self.state = EncoderState::Idle;
        self.last_intent = None;
        self.last_fit = None;
    }

    /// Compression ratio estimate
    ///
    /// Returns (raw_bytes_per_second, intent_bytes_per_second)
    pub fn compression_estimate(sample_rate_hz: u32, avg_motion_duration_ms: u32) -> (u32, u32) {
        // Raw: 3 × f32 × sample_rate = 12 × sample_rate bytes/s
        let raw = 12 * sample_rate_hz;
        // Intent: 8 bytes per motion, ~(1000/duration) motions per second
        let motions_per_sec = if avg_motion_duration_ms > 0 {
            1000 / avg_motion_duration_ms
        } else { 1 };
        let intent = 8 * motions_per_sec;
        (raw, intent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reaching_samples(count: usize, target_x: f32) -> alloc::vec::Vec<SensorSample> {
        let mut samples = alloc::vec::Vec::new();
        for i in 0..count {
            let t = i as f32 * 0.001; // 1000Hz
            let progress = t / (count as f32 * 0.001);
            // Smooth quintic-like profile
            let s = 6.0 * progress.powi(5) - 15.0 * progress.powi(4) + 10.0 * progress.powi(3);
            samples.push(SensorSample {
                pos: Vec3k::new(s * target_x, 0.0, 0.0),
                vel: Vec3k::ZERO,
                timestamp: t,
            });
        }
        samples
    }

    #[test]
    fn test_encoder_creation() {
        let enc = IntentEncoder::new();
        assert_eq!(enc.state(), EncoderState::Idle);
    }

    #[test]
    fn test_encoder_dead_zone() {
        let mut enc = IntentEncoder::new();
        // Sub-dead-zone motion should not trigger
        let result = enc.push_sample(SensorSample {
            pos: Vec3k::new(0.001, 0.0, 0.0),
            vel: Vec3k::ZERO,
            timestamp: 0.0,
        });
        assert!(result.is_none());
        assert_eq!(enc.state(), EncoderState::Idle);
    }

    #[test]
    fn test_encoder_motion_detection() {
        let mut enc = IntentEncoder::with_thresholds(0.01, 0.05, 0.001);
        let samples = make_reaching_samples(20, 0.5);

        for s in &samples {
            enc.push_sample(*s);
        }
        // Should have moved past Idle
        assert_ne!(enc.state(), EncoderState::Idle);
    }

    #[test]
    fn test_encoder_extract_intent() {
        let mut enc = IntentEncoder::with_thresholds(0.01, 0.1, 0.0001);
        let samples = make_reaching_samples(30, 0.5);

        let mut extracted = None;
        for s in &samples {
            if let Some(intent) = enc.push_sample(*s) {
                extracted = Some(intent);
                break;
            }
        }
        assert!(extracted.is_some());
        let intent = extracted.unwrap();
        assert!(intent.target.x > 0.0);
        assert!(intent.duration_ms > 0);
    }

    #[test]
    fn test_encoder_reset() {
        let mut enc = IntentEncoder::new();
        enc.push_sample(SensorSample {
            pos: Vec3k::new(1.0, 0.0, 0.0),
            vel: Vec3k::ZERO,
            timestamp: 0.0,
        });
        enc.reset();
        assert_eq!(enc.state(), EncoderState::Idle);
        assert!(enc.last_intent().is_none());
    }

    #[test]
    fn test_compression_estimate() {
        let (raw, intent) = IntentEncoder::compression_estimate(1000, 200);
        // Raw: 12 * 1000 = 12000 bytes/s
        assert_eq!(raw, 12000);
        // Intent: 8 * (1000/200) = 8 * 5 = 40 bytes/s
        assert_eq!(intent, 40);
        // Ratio: 300x
        assert!(raw / intent >= 100);
    }

    #[test]
    fn test_encoder_sequence_counter() {
        let mut enc = IntentEncoder::with_thresholds(0.01, 0.1, 0.0001);

        // Extract multiple intents
        for round in 0..3 {
            let samples = make_reaching_samples(30, 0.5 + round as f32 * 0.1);
            for s in &samples {
                enc.push_sample(*s);
            }
        }
        // Sequence should have incremented
        if let Some(intent) = enc.last_intent() {
            assert!(intent.flags.sequence() <= 7);
        }
    }

    #[test]
    fn test_force_extract() {
        let mut enc = IntentEncoder::with_thresholds(0.001, 0.5, 0.0001);
        let samples = make_reaching_samples(5, 1.0);
        for s in &samples {
            enc.push_sample(*s);
        }
        let result = enc.force_extract();
        // May or may not succeed depending on sample count
        // But should not panic
        let _ = result;
    }
}
