//! Intent — compact motion intent packet (8-16 bytes)
//!
//! Replaces 1000Hz coordinate streaming with a single intent:
//! "Move hand to (x,y,z) over `duration_ms` milliseconds"
//!
//! Packet format (8 bytes):
//! - `target_x`: i16 (Q8.8 fixed-point)
//! - `target_y`: i16 (Q8.8 fixed-point)
//! - `target_z`: i16 (Q8.8 fixed-point)
//! - duration: u8 (milliseconds, 0-255)
//! - flags: u8 (joint mask, grip state, intent type)
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;

/// Q8.8 fixed-point scale factor
const Q8_SCALE: f32 = 256.0;

/// Intent type — what kind of motion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntentType {
    /// Reaching motion (jerk-minimized trajectory)
    Reach = 0,
    /// Pointing / aiming
    Point = 1,
    /// Grasping motion
    Grasp = 2,
    /// Release / throw
    Release = 3,
}

impl IntentType {
    #[must_use]
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            1 => Self::Point,
            2 => Self::Grasp,
            3 => Self::Release,
            _ => Self::Reach,
        }
    }
}

/// Intent flags (packed in 1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntentFlags(pub u8);

impl IntentFlags {
    pub const EMPTY: Self = Self(0);

    /// Intent type (bits 0-1)
    #[must_use]
    pub const fn intent_type(self) -> IntentType {
        IntentType::from_u8(self.0 & 0x03)
    }

    /// Grip state: true = closed (bit 2)
    #[must_use]
    pub const fn grip_closed(self) -> bool {
        self.0 & 0x04 != 0
    }

    /// Left hand (bit 3): false = right, true = left
    #[must_use]
    pub const fn is_left_hand(self) -> bool {
        self.0 & 0x08 != 0
    }

    /// High precision mode (bit 4): use 16-byte extended packet
    #[must_use]
    pub const fn high_precision(self) -> bool {
        self.0 & 0x10 != 0
    }

    /// Sequence number (bits 5-7): 0-7 wrapping counter
    #[must_use]
    pub const fn sequence(self) -> u8 {
        (self.0 >> 5) & 0x07
    }

    #[must_use]
    pub const fn new(intent_type: IntentType, grip: bool, left: bool, seq: u8) -> Self {
        let mut f = intent_type as u8;
        if grip {
            f |= 0x04;
        }
        if left {
            f |= 0x08;
        }
        f |= (seq & 0x07) << 5;
        Self(f)
    }
}

/// Motion intent packet (8 bytes)
///
/// Encodes a human motion intention as:
/// - Target position (3 × Q8.8 = 6 bytes)
/// - Duration (1 byte, milliseconds)
/// - Flags (1 byte)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intent {
    /// Target position (world space, meters)
    pub target: Vec3k,
    /// Movement duration in milliseconds
    pub duration_ms: u8,
    /// Flags
    pub flags: IntentFlags,
}

impl Intent {
    /// Create a new reaching intent
    #[must_use]
    pub const fn reach(target: Vec3k, duration_ms: u8) -> Self {
        Self {
            target,
            duration_ms,
            flags: IntentFlags::new(IntentType::Reach, false, false, 0),
        }
    }

    /// Create a grasp intent
    #[must_use]
    pub const fn grasp(target: Vec3k, duration_ms: u8) -> Self {
        Self {
            target,
            duration_ms,
            flags: IntentFlags::new(IntentType::Grasp, true, false, 0),
        }
    }

    /// Encode intent to 8-byte packet
    #[must_use]
    pub fn encode(&self) -> [u8; 8] {
        let tx = f32_to_q8(self.target.x);
        let ty = f32_to_q8(self.target.y);
        let tz = f32_to_q8(self.target.z);
        [
            (tx >> 8) as u8,
            tx as u8,
            (ty >> 8) as u8,
            ty as u8,
            (tz >> 8) as u8,
            tz as u8,
            self.duration_ms,
            self.flags.0,
        ]
    }

    /// Decode intent from 8-byte packet
    #[must_use]
    pub fn decode(data: &[u8; 8]) -> Self {
        let tx = ((data[0] as i16) << 8) | data[1] as i16;
        let ty = ((data[2] as i16) << 8) | data[3] as i16;
        let tz = ((data[4] as i16) << 8) | data[5] as i16;
        Self {
            target: Vec3k::new(q8_to_f32(tx), q8_to_f32(ty), q8_to_f32(tz)),
            duration_ms: data[6],
            flags: IntentFlags(data[7]),
        }
    }

    /// Packet size in bytes
    #[must_use]
    pub const fn packet_size() -> usize {
        8
    }

    /// Duration in seconds
    #[inline(always)]
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        const INV_1000: f32 = 1.0 / 1000.0;
        self.duration_ms as f32 * INV_1000
    }
}

/// Convert f32 to Q8.8 fixed-point (i16)
fn f32_to_q8(v: f32) -> i16 {
    let scaled = v * Q8_SCALE;
    if scaled > 32767.0 {
        32767
    } else if scaled < -32768.0 {
        -32768
    } else {
        scaled as i16
    }
}

/// Convert Q8.8 fixed-point (i16) to f32
#[inline(always)]
fn q8_to_f32(v: i16) -> f32 {
    const INV_Q8_SCALE: f32 = 1.0 / 256.0;
    v as f32 * INV_Q8_SCALE
}

/// Extended intent with initial velocity (16 bytes)
///
/// For higher fidelity when motion is already in progress.
#[derive(Debug, Clone, Copy)]
pub struct ExtendedIntent {
    /// Base intent (8 bytes)
    pub base: Intent,
    /// Initial velocity at intent start (Q8.8)
    pub velocity: Vec3k,
    /// Reserved (2 bytes)
    pub reserved: u16,
}

impl ExtendedIntent {
    #[must_use]
    pub const fn new(base: Intent, velocity: Vec3k) -> Self {
        Self {
            base,
            velocity,
            reserved: 0,
        }
    }

    #[must_use]
    pub fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        let base_enc = self.base.encode();
        buf[0..8].copy_from_slice(&base_enc);

        let vx = f32_to_q8(self.velocity.x);
        let vy = f32_to_q8(self.velocity.y);
        let vz = f32_to_q8(self.velocity.z);
        buf[8] = (vx >> 8) as u8;
        buf[9] = vx as u8;
        buf[10] = (vy >> 8) as u8;
        buf[11] = vy as u8;
        buf[12] = (vz >> 8) as u8;
        buf[13] = vz as u8;
        buf[14] = 0;
        buf[15] = 0;
        buf
    }

    #[must_use]
    pub fn decode(data: &[u8; 16]) -> Self {
        let mut base_data = [0u8; 8];
        base_data.copy_from_slice(&data[0..8]);
        let base = Intent::decode(&base_data);

        let vx = ((data[8] as i16) << 8) | data[9] as i16;
        let vy = ((data[10] as i16) << 8) | data[11] as i16;
        let vz = ((data[12] as i16) << 8) | data[13] as i16;

        Self {
            base,
            velocity: Vec3k::new(q8_to_f32(vx), q8_to_f32(vy), q8_to_f32(vz)),
            reserved: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_encode_decode_roundtrip() {
        let intent = Intent::reach(Vec3k::new(0.5, 1.0, -0.3), 150);
        let encoded = intent.encode();
        assert_eq!(encoded.len(), 8);
        let decoded = Intent::decode(&encoded);
        assert!((decoded.target.x - 0.5).abs() < 0.01);
        assert!((decoded.target.y - 1.0).abs() < 0.01);
        assert!((decoded.target.z - (-0.3)).abs() < 0.01);
        assert_eq!(decoded.duration_ms, 150);
    }

    #[test]
    fn test_intent_flags() {
        let flags = IntentFlags::new(IntentType::Grasp, true, true, 5);
        assert_eq!(flags.intent_type(), IntentType::Grasp);
        assert!(flags.grip_closed());
        assert!(flags.is_left_hand());
        assert_eq!(flags.sequence(), 5);
    }

    #[test]
    fn test_intent_grasp() {
        let intent = Intent::grasp(Vec3k::new(0.3, 0.8, 0.1), 200);
        assert_eq!(intent.flags.intent_type(), IntentType::Grasp);
        assert!(intent.flags.grip_closed());
    }

    #[test]
    fn test_q8_range() {
        // Q8.8: range ±127.996, resolution 0.00390625
        let v = f32_to_q8(100.0);
        let back = q8_to_f32(v);
        assert!((back - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_q8_negative() {
        let v = f32_to_q8(-50.0);
        let back = q8_to_f32(v);
        assert!((back - (-50.0)).abs() < 0.01);
    }

    #[test]
    fn test_q8_saturation() {
        let v = f32_to_q8(200.0);
        assert_eq!(v, 32767);
    }

    #[test]
    fn test_packet_size() {
        assert_eq!(Intent::packet_size(), 8);
    }

    #[test]
    fn test_duration_secs() {
        let intent = Intent::reach(Vec3k::ZERO, 200);
        assert!((intent.duration_secs() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_extended_intent_roundtrip() {
        let base = Intent::reach(Vec3k::new(1.0, 0.5, -0.2), 100);
        let ext = ExtendedIntent::new(base, Vec3k::new(0.1, -0.2, 0.05));
        let encoded = ext.encode();
        assert_eq!(encoded.len(), 16);
        let decoded = ExtendedIntent::decode(&encoded);
        assert!((decoded.base.target.x - 1.0).abs() < 0.01);
        assert!((decoded.velocity.x - 0.1).abs() < 0.01);
        assert!((decoded.velocity.y - (-0.2)).abs() < 0.01);
    }

    #[test]
    fn test_intent_type_from_u8() {
        assert_eq!(IntentType::from_u8(0), IntentType::Reach);
        assert_eq!(IntentType::from_u8(1), IntentType::Point);
        assert_eq!(IntentType::from_u8(2), IntentType::Grasp);
        assert_eq!(IntentType::from_u8(3), IntentType::Release);
    }

    // --- 追加テスト ---

    #[test]
    fn test_intent_type_from_u8_upper_bits_ignored() {
        // 上位ビットは無視されて下位2ビットのみ使われる
        assert_eq!(IntentType::from_u8(0xFF), IntentType::Release); // 0xFF & 0x03 == 3
        assert_eq!(IntentType::from_u8(0xFC), IntentType::Reach); // 0xFC & 0x03 == 0
        assert_eq!(IntentType::from_u8(0xFD), IntentType::Point); // 0xFD & 0x03 == 1
    }

    #[test]
    fn test_intent_flags_empty() {
        let f = IntentFlags::EMPTY;
        assert_eq!(f.intent_type(), IntentType::Reach);
        assert!(!f.grip_closed());
        assert!(!f.is_left_hand());
        assert!(!f.high_precision());
        assert_eq!(f.sequence(), 0);
    }

    #[test]
    fn test_intent_flags_sequence_range() {
        // sequence は 3ビット (0-7)
        for seq in 0u8..8 {
            let f = IntentFlags::new(IntentType::Reach, false, false, seq);
            assert_eq!(f.sequence(), seq);
        }
        // 8以上は下位3ビットに切り捨て
        let f = IntentFlags::new(IntentType::Reach, false, false, 9);
        assert_eq!(f.sequence(), 1); // 9 & 0x07 == 1
    }

    #[test]
    fn test_intent_flags_all_bits() {
        let f = IntentFlags::new(IntentType::Release, true, true, 7);
        assert_eq!(f.intent_type(), IntentType::Release);
        assert!(f.grip_closed());
        assert!(f.is_left_hand());
        assert_eq!(f.sequence(), 7);
    }

    #[test]
    fn test_intent_flags_right_hand_no_grip() {
        let f = IntentFlags::new(IntentType::Point, false, false, 3);
        assert_eq!(f.intent_type(), IntentType::Point);
        assert!(!f.grip_closed());
        assert!(!f.is_left_hand());
        assert_eq!(f.sequence(), 3);
    }

    #[test]
    fn test_intent_zero_duration() {
        let intent = Intent::reach(Vec3k::ZERO, 0);
        assert_eq!(intent.duration_ms, 0);
        assert!((intent.duration_secs()).abs() < 1e-6);
    }

    #[test]
    fn test_intent_max_duration() {
        let intent = Intent::reach(Vec3k::ZERO, 255);
        assert_eq!(intent.duration_ms, 255);
        assert!((intent.duration_secs() - 0.255).abs() < 0.001);
    }

    #[test]
    fn test_intent_encode_decode_negative_coords() {
        let intent = Intent::reach(Vec3k::new(-1.0, -0.5, -0.25), 100);
        let encoded = intent.encode();
        let decoded = Intent::decode(&encoded);
        assert!((decoded.target.x - (-1.0)).abs() < 0.01);
        assert!((decoded.target.y - (-0.5)).abs() < 0.01);
        assert!((decoded.target.z - (-0.25)).abs() < 0.01);
    }

    #[test]
    fn test_intent_encode_decode_zero_target() {
        let intent = Intent::reach(Vec3k::ZERO, 50);
        let encoded = intent.encode();
        let decoded = Intent::decode(&encoded);
        assert!(decoded.target.x.abs() < 0.01);
        assert!(decoded.target.y.abs() < 0.01);
        assert!(decoded.target.z.abs() < 0.01);
        assert_eq!(decoded.duration_ms, 50);
    }

    #[test]
    fn test_q8_negative_saturation() {
        // -200.0 は Q8.8 の下限 -32768 に飽和する
        let v = f32_to_q8(-200.0);
        assert_eq!(v, -32768);
    }

    #[test]
    fn test_extended_intent_velocity_zero() {
        let base = Intent::reach(Vec3k::new(0.5, 0.0, 0.0), 100);
        let ext = ExtendedIntent::new(base, Vec3k::ZERO);
        let encoded = ext.encode();
        let decoded = ExtendedIntent::decode(&encoded);
        assert!(decoded.velocity.x.abs() < 0.01);
        assert!(decoded.velocity.y.abs() < 0.01);
        assert!(decoded.velocity.z.abs() < 0.01);
    }

    #[test]
    fn test_extended_intent_reserved_bytes_zero() {
        let base = Intent::reach(Vec3k::new(1.0, 0.0, 0.0), 100);
        let ext = ExtendedIntent::new(base, Vec3k::new(0.5, 0.0, 0.0));
        let encoded = ext.encode();
        assert_eq!(encoded.len(), 16);
        // 後半2バイトはreserved = 0
        assert_eq!(encoded[14], 0);
        assert_eq!(encoded[15], 0);
    }

    // --- さらに追加テスト ---

    #[test]
    fn test_intent_reach_type_is_reach() {
        // reach() ファクトリは IntentType::Reach を返す
        let intent = Intent::reach(Vec3k::new(0.1, 0.2, 0.3), 100);
        assert_eq!(intent.flags.intent_type(), IntentType::Reach);
        assert!(!intent.flags.grip_closed());
    }

    #[test]
    fn test_intent_high_precision_flag() {
        // bit4 が立っているとき high_precision は true
        let f = IntentFlags(0x10);
        assert!(f.high_precision());
        // 立っていなければ false
        let f2 = IntentFlags(0x00);
        assert!(!f2.high_precision());
    }

    #[test]
    fn test_intent_flags_raw_byte_roundtrip() {
        // IntentFlags の内部バイト値が encode/decode を経ても保たれる
        let flags = IntentFlags::new(IntentType::Grasp, true, false, 6);
        let raw = flags.0;
        let flags2 = IntentFlags(raw);
        assert_eq!(flags2.intent_type(), IntentType::Grasp);
        assert!(flags2.grip_closed());
        assert_eq!(flags2.sequence(), 6);
    }

    #[test]
    fn test_intent_encode_decode_preserves_flags() {
        // encode → decode でフラグが失われない
        let flags = IntentFlags::new(IntentType::Point, false, true, 3);
        let intent = Intent {
            target: Vec3k::new(0.5, 0.5, 0.0),
            duration_ms: 80,
            flags,
        };
        let encoded = intent.encode();
        let decoded = Intent::decode(&encoded);
        assert_eq!(decoded.flags.intent_type(), IntentType::Point);
        assert!(decoded.flags.is_left_hand());
        assert_eq!(decoded.flags.sequence(), 3);
        assert_eq!(decoded.duration_ms, 80);
    }

    #[test]
    fn test_q8_near_zero_precision() {
        // Q8.8 の分解能: 1/256 ≈ 0.0039
        let v = f32_to_q8(0.004);
        let back = q8_to_f32(v);
        // ±2 LSB = ±0.008 の精度で復元できる
        assert!((back - 0.004).abs() < 0.008);
    }

    #[test]
    fn test_intent_all_types_encode_decode() {
        // 4種類の IntentType がすべて encode/decode を経て保存される
        let types = [
            IntentType::Reach,
            IntentType::Point,
            IntentType::Grasp,
            IntentType::Release,
        ];
        for it in types {
            let flags = IntentFlags::new(it, false, false, 0);
            let intent = Intent {
                target: Vec3k::ZERO,
                duration_ms: 10,
                flags,
            };
            let encoded = intent.encode();
            let decoded = Intent::decode(&encoded);
            assert_eq!(decoded.flags.intent_type(), it);
        }
    }

    #[test]
    fn test_extended_intent_negative_velocity_roundtrip() {
        // 負の速度ベクトルが encode/decode を経て正しく復元される
        let base = Intent::reach(Vec3k::new(0.5, 0.0, 0.0), 100);
        let vel = Vec3k::new(-0.5, -1.0, 0.25);
        let ext = ExtendedIntent::new(base, vel);
        let encoded = ext.encode();
        let decoded = ExtendedIntent::decode(&encoded);
        assert!((decoded.velocity.x - (-0.5)).abs() < 0.01);
        assert!((decoded.velocity.y - (-1.0)).abs() < 0.01);
        assert!((decoded.velocity.z - 0.25).abs() < 0.01);
    }
}
