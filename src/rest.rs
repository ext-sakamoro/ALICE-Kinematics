//! Rest data of a skeleton, kept apart from its pose: the end points past the last joints
//! (head top, fingertips, toe tips, heels) and a frame per bone (its axis, and the width and
//! depth directions across it, which fix the bone's roll).
//!
//! A bone here is the crate's: the segment from a joint's parent to the joint, the segment whose
//! length is the joint's `bone_length`. An end point adds one more segment, from its joint to the
//! end point. The pose (`local_rotation`, `Skeleton::set_rotation`) does not change either.
//!
//! The frame of a segment with unit axis `a` takes a depth reference `r`: front (+Z) unless the
//! axis is closer to the front than 45 degrees (|a.z| > 1/sqrt 2, a foot), then up (+Y). Depth is
//! `r` with its part along `a` removed, normalised; width is `depth x a`, so `a x width = depth`
//! (right-handed). Left and right frames have the same depth and opposite-handed widths: a mirror
//! of a right-handed frame is left-handed.
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::nan::canon3;
use crate::skeleton::{BoneId, Skeleton};
use alloc::vec::Vec;

/// Shortest segment that has a direction (in the input's units).
const MIN_LEN: f32 = 1e-9;

/// Length, correctly rounded (`Vec3k::length` is an approximate root, about 1e-5 relative).
fn exact_len(v: Vec3k) -> f32 {
    alice_det_math::sqrt(v.length_sq())
}

fn finite(v: Vec3k) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// A point past the last joint of a chain, carried by that joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndPoint {
    /// Top of the head, carried by `Head`.
    HeadTop,
    /// Left fingertip, carried by `LeftHand`.
    LeftFingertip,
    /// Right fingertip, carried by `RightHand`.
    RightFingertip,
    /// Left toe tip, carried by `LeftToe`.
    LeftToeTip,
    /// Right toe tip, carried by `RightToe`.
    RightToeTip,
    /// Back of the left heel, carried by `LeftFoot`.
    LeftHeel,
    /// Back of the right heel, carried by `RightFoot`.
    RightHeel,
}

impl EndPoint {
    /// Every end point.
    pub const ALL: [Self; 7] = [
        Self::HeadTop,
        Self::LeftFingertip,
        Self::RightFingertip,
        Self::LeftToeTip,
        Self::RightToeTip,
        Self::LeftHeel,
        Self::RightHeel,
    ];

    /// The joint that carries this end point.
    #[must_use]
    pub const fn bone(self) -> BoneId {
        match self {
            Self::HeadTop => BoneId::Head,
            Self::LeftFingertip => BoneId::LeftHand,
            Self::RightFingertip => BoneId::RightHand,
            Self::LeftToeTip => BoneId::LeftToe,
            Self::RightToeTip => BoneId::RightToe,
            Self::LeftHeel => BoneId::LeftFoot,
            Self::RightHeel => BoneId::RightFoot,
        }
    }
}

/// A bone's rest frame: unit axis along the bone, and unit width and depth across it, mutually
/// orthogonal with `axis x width = depth`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneFrame {
    axis: Vec3k,
    width: Vec3k,
    depth: Vec3k,
}

impl BoneFrame {
    /// The frame along `along` (any non-zero length) whose depth is `depth_hint` with its part
    /// along the axis removed; width is `depth x axis`.
    ///
    /// # Errors
    /// [`RestError::NonFinite`] for a NaN or infinite input, [`RestError::Degenerate`] when
    /// `along` is (nearly) zero or `depth_hint` is (nearly) parallel to it.
    pub fn new(along: Vec3k, depth_hint: Vec3k) -> Result<Self, RestError> {
        if !(finite(along) && finite(depth_hint)) {
            return Err(RestError::NonFinite);
        }
        let len = exact_len(along);
        if len <= MIN_LEN {
            return Err(RestError::Degenerate);
        }
        let axis = along.scale(1.0 / len);
        let d = depth_hint - axis.scale(depth_hint.dot(axis));
        let dl = exact_len(d);
        // parallel within ~1e-3 rad of the hint's own length (all lengths finite: checked above)
        if dl <= 1e-3 * exact_len(depth_hint) {
            return Err(RestError::Degenerate);
        }
        let depth = d.scale(1.0 / dl);
        let width = depth.cross(axis);
        Ok(Self { axis, width, depth })
    }

    /// The frame along `along` with the default depth reference (front, or up for an axis
    /// within 45 degrees of the front).
    ///
    /// # Errors
    /// As [`BoneFrame::new`].
    pub fn along(along: Vec3k) -> Result<Self, RestError> {
        if !finite(along) {
            return Err(RestError::NonFinite);
        }
        let len = exact_len(along);
        if len <= MIN_LEN {
            return Err(RestError::Degenerate);
        }
        // |a.z| > 1/sqrt 2, compared squared so no root is taken: 2 z^2 > |along|^2
        let hint = if 2.0 * along.z * along.z > along.length_sq() {
            Vec3k::new(0.0, 1.0, 0.0)
        } else {
            Vec3k::new(0.0, 0.0, 1.0)
        };
        Self::new(along, hint)
    }

    /// Unit vector along the bone, from the parent joint toward the joint.
    #[must_use]
    pub fn axis(&self) -> Vec3k {
        canon3(self.axis)
    }

    /// Unit vector across the bone: `depth x axis`.
    #[must_use]
    pub fn width(&self) -> Vec3k {
        canon3(self.width)
    }

    /// Unit vector across the bone, toward the depth reference.
    #[must_use]
    pub fn depth(&self) -> Vec3k {
        canon3(self.depth)
    }
}

/// Why rest data was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestError {
    /// A NaN or infinite coordinate.
    NonFinite,
    /// The end point's joint is not in the skeleton.
    MissingBone(BoneId),
    /// The same end point is given more than once.
    DuplicateEndPoint(EndPoint),
    /// A zero-length segment, or a depth hint parallel to the axis.
    Degenerate,
    /// The bone has no parent (the root has no segment) or is not in the skeleton.
    NoSegment(BoneId),
}

impl core::fmt::Display for RestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "a NaN or infinite coordinate"),
            Self::MissingBone(b) => write!(f, "joint {b:?} is not in the skeleton"),
            Self::DuplicateEndPoint(e) => write!(f, "end point {e:?} given more than once"),
            Self::Degenerate => write!(f, "zero-length segment or depth hint parallel to the axis"),
            Self::NoSegment(b) => write!(f, "bone {b:?} has no segment (root or absent)"),
        }
    }
}

/// A skeleton with its rest data: end points (stored as offsets from their joints, so they move
/// with them) and one frame per segment, each defaulted from the rest positions and replaceable.
#[derive(Debug, Clone)]
pub struct RestSkeleton {
    skeleton: Skeleton,
    end_points: Vec<(EndPoint, Vec3k)>,
    frames: Vec<(BoneId, BoneFrame)>,
    end_frames: Vec<(EndPoint, BoneFrame)>,
}

impl RestSkeleton {
    /// Rest data for `skeleton` at its current world positions, with the end points given in
    /// world coordinates. Every segment (each non-root joint, each end point) gets its default
    /// frame ([`BoneFrame::along`]); a zero-length segment gets none.
    ///
    /// # Errors
    /// [`RestError::NonFinite`], [`RestError::MissingBone`] when an end point's joint is absent,
    /// [`RestError::DuplicateEndPoint`].
    pub fn new(skeleton: Skeleton, end_points: &[(EndPoint, Vec3k)]) -> Result<Self, RestError> {
        let mut offsets: Vec<(EndPoint, Vec3k)> = Vec::new();
        for &(end, p) in end_points {
            if !finite(p) {
                return Err(RestError::NonFinite);
            }
            if offsets.iter().any(|&(e, _)| e == end) {
                return Err(RestError::DuplicateEndPoint(end));
            }
            let at = skeleton
                .joint_world_position(end.bone())
                .ok_or(RestError::MissingBone(end.bone()))?;
            offsets.push((end, p - at));
        }
        let mut frames = Vec::new();
        for j in &skeleton.joints {
            if let Some(p) = j.parent {
                if let Ok(f) =
                    BoneFrame::along(j.world_position - skeleton.joints[p].world_position)
                {
                    frames.push((j.id, f));
                }
            }
        }
        let mut end_frames = Vec::new();
        for &(end, off) in &offsets {
            if let Ok(f) = BoneFrame::along(off) {
                end_frames.push((end, f));
            }
        }
        Ok(Self {
            skeleton,
            end_points: offsets,
            frames,
            end_frames,
        })
    }

    /// The skeleton (its pose included).
    #[must_use]
    pub const fn skeleton(&self) -> &Skeleton {
        &self.skeleton
    }

    /// The skeleton, to pose it: rotations and moved joints do not change the rest frames.
    pub fn skeleton_mut(&mut self) -> &mut Skeleton {
        &mut self.skeleton
    }

    /// The end point's world position: its joint's current world position plus the rest offset.
    #[must_use]
    pub fn end_point(&self, end: EndPoint) -> Option<Vec3k> {
        let off = self.end_points.iter().find(|&&(e, _)| e == end)?.1;
        // a skeleton holding NaN joints (with_height(NaN)) gives a NaN: the canonical one (G4)
        Some(canon3(
            self.skeleton.joint_world_position(end.bone())? + off,
        ))
    }

    /// The rest frame of the segment ending at `bone` (from its parent).
    #[must_use]
    pub fn frame(&self, bone: BoneId) -> Option<BoneFrame> {
        self.frames
            .iter()
            .find(|&&(b, _)| b == bone)
            .map(|&(_, f)| f)
    }

    /// The rest frame of the segment from the end point's joint to the end point.
    #[must_use]
    pub fn end_frame(&self, end: EndPoint) -> Option<BoneFrame> {
        self.end_frames
            .iter()
            .find(|&&(e, _)| e == end)
            .map(|&(_, f)| f)
    }

    /// Replace the rest frame of the segment ending at `bone` by the one along that segment with
    /// `depth_hint` (sets the bone's roll).
    ///
    /// # Errors
    /// [`RestError::NoSegment`] for the root or an absent bone, and as [`BoneFrame::new`].
    pub fn set_frame(&mut self, bone: BoneId, depth_hint: Vec3k) -> Result<(), RestError> {
        // the rest segment: the frame's own axis (the rest pose), not the posed joints
        let axis = self.frame(bone).ok_or(RestError::NoSegment(bone))?.axis();
        let f = BoneFrame::new(axis, depth_hint)?;
        for slot in &mut self.frames {
            if slot.0 == bone {
                slot.1 = f;
            }
        }
        Ok(())
    }
}
