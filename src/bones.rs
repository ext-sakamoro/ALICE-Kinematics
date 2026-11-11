//! Named-bone skeletons: any number of bones, each with a name, a parent index and a head → tail
//! segment
//!
//! [`crate::skeleton::Skeleton`] has a fixed set of 22 humanoid joints ([`BoneId`]), so it cannot
//! name a support bone, a finger or a costume bone. A [`BoneSkeleton`] holds any bones: each has a
//! unique name, a parent that comes before it (index order is a topological order), a world head
//! and a world tail, and an influence class for distance skinning. The 22-joint humanoid is a
//! preset of it ([`BoneSkeleton::humanoid`], [`BoneSkeleton::from_skeleton`]).
//!
//! # Segments
//!
//! A bone owns its own head → tail segment, as in a modelling tool's armature. The humanoid
//! preset maps joint `i` with parent `p` to a bone named after the joint whose head is joint
//! `p`'s world position and whose tail is joint `i`'s, so the segment and its owner are the ones
//! [`crate::autorig::distance_based_skinning`] has always used; the root joint is a bone whose
//! head and tail are both its position (a point).
//!
//! # Forward kinematics
//!
//! [`BoneSkeleton::from_offsets`] places bones by offsets: a bone's world head is its parent's
//! world tail plus its `head_offset` (a root's is its `head_offset`), and its world tail is its
//! world head plus its `tail_offset`, in index order. Translation only, like
//! [`Skeleton::update_world_positions`](crate::skeleton::Skeleton::update_world_positions).
//! [`BoneSkeleton::from_world`] takes world heads and tails as given (a rig read from a file).
//!
//! Stored coordinates have every NaN replaced by the canonical NaN (ALICE-Law G4, `src/nan.rs`).
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

use crate::joint::Vec3k;
use crate::nan::{canon, canon3};
use crate::skeleton::{BoneId, Skeleton};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Which influence radius distance skinning uses for a bone
/// ([`crate::autorig::InfluenceRadii`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoneClass {
    /// An arm, leg, hand, foot or toe bone: `InfluenceRadii::limb_max`.
    Limb,
    /// Any other bone (torso, head, shoulder, root, support): `InfluenceRadii::torso_max`.
    Torso,
}

/// A bone placed by offsets, the input of [`BoneSkeleton::from_offsets`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoneOffsets {
    /// The bone's name, unique in the skeleton and not empty.
    pub name: String,
    /// The index of the parent bone, smaller than this bone's index; `None` for a root.
    pub parent: Option<usize>,
    /// The world head minus the parent's world tail (a root: the world head).
    pub head_offset: Vec3k,
    /// The world tail minus the world head.
    pub tail_offset: Vec3k,
    /// The influence class for distance skinning.
    pub class: BoneClass,
}

/// A bone given in world space, the input of [`BoneSkeleton::from_world`].
#[derive(Debug, Clone, PartialEq)]
pub struct WorldBone {
    /// The bone's name, unique in the skeleton and not empty.
    pub name: String,
    /// The index of the parent bone, smaller than this bone's index; `None` for a root.
    pub parent: Option<usize>,
    /// The world position of the bone's head.
    pub head: Vec3k,
    /// The world position of the bone's tail.
    pub tail: Vec3k,
    /// The influence class for distance skinning.
    pub class: BoneClass,
}

/// Why a list of bones is not a skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoneSkeletonError {
    /// The list holds no bone.
    Empty,
    /// More bones than a skinning index (`u16`) can name.
    TooManyBones(usize),
    /// The bone at this index has an empty name.
    EmptyName(usize),
    /// Two bones share a name: the later index and the name.
    DuplicateName(usize, String),
    /// The bone at this index names a parent that is not an earlier bone.
    BadParent {
        /// The bone's index.
        bone: usize,
        /// The parent index it names.
        parent: usize,
    },
}

impl core::fmt::Display for BoneSkeletonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "a skeleton needs at least one bone"),
            Self::TooManyBones(n) => {
                write!(f, "{n} bones, more than a skinning index can name (65536)")
            }
            Self::EmptyName(i) => write!(f, "bone {i} has an empty name"),
            Self::DuplicateName(i, name) => write!(f, "bone {i} repeats the name {name:?}"),
            Self::BadParent { bone, parent } => {
                write!(
                    f,
                    "bone {bone} names parent {parent}, which is not an earlier bone"
                )
            }
        }
    }
}

/// One bone of a [`BoneSkeleton`]: its name, parent, world segment and influence class.
#[derive(Debug, Clone, PartialEq)]
pub struct Bone {
    name: String,
    parent: Option<usize>,
    head: Vec3k,
    tail: Vec3k,
    class: BoneClass,
}

impl Bone {
    /// The bone's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The index of the parent bone (`None` for a root).
    #[must_use]
    pub const fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// The world position of the bone's head.
    #[must_use]
    pub const fn head(&self) -> Vec3k {
        self.head
    }

    /// The world position of the bone's tail.
    #[must_use]
    pub const fn tail(&self) -> Vec3k {
        self.tail
    }

    /// The influence class for distance skinning.
    #[must_use]
    pub const fn class(&self) -> BoneClass {
        self.class
    }
}

/// A skeleton of named bones with world head → tail segments (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct BoneSkeleton {
    bones: Vec<Bone>,
    height: f32,
}

impl BoneSkeleton {
    /// A skeleton from bones given in world space.
    ///
    /// `height` scales the influence radii of distance skinning (`radius × height`).
    ///
    /// # Errors
    ///
    /// [`BoneSkeletonError`] when the list is empty or longer than 65,536 bones, a name is empty
    /// or repeated, or a parent is not an earlier bone.
    pub fn from_world(bones: Vec<WorldBone>, height: f32) -> Result<Self, BoneSkeletonError> {
        check(bones.iter().map(|b| (b.name.as_str(), b.parent)))?;
        Ok(Self {
            bones: bones
                .into_iter()
                .map(|b| Bone {
                    name: b.name,
                    parent: b.parent,
                    head: canon3(b.head),
                    tail: canon3(b.tail),
                    class: b.class,
                })
                .collect(),
            height: canon(height),
        })
    }

    /// A skeleton from bones placed by offsets, by forward kinematics in index order (see the
    /// module documentation).
    ///
    /// # Errors
    ///
    /// As [`BoneSkeleton::from_world`].
    pub fn from_offsets(bones: Vec<BoneOffsets>, height: f32) -> Result<Self, BoneSkeletonError> {
        check(bones.iter().map(|b| (b.name.as_str(), b.parent)))?;
        let mut placed: Vec<Bone> = Vec::with_capacity(bones.len());
        for b in bones {
            let head = match b.parent {
                Some(p) => placed[p].tail + b.head_offset,
                None => b.head_offset,
            };
            let tail = head + b.tail_offset;
            placed.push(Bone {
                name: b.name,
                parent: b.parent,
                head,
                tail,
                class: b.class,
            });
        }
        for b in &mut placed {
            b.head = canon3(b.head);
            b.tail = canon3(b.tail);
        }
        Ok(Self {
            bones: placed,
            height: canon(height),
        })
    }

    /// The humanoid preset: [`Skeleton::with_height`] as named bones (see
    /// [`BoneSkeleton::from_skeleton`]).
    #[must_use]
    pub fn humanoid(height: f32) -> Self {
        Self::from_skeleton(&Skeleton::with_height(height))
    }

    /// A [`Skeleton`] as named bones: joint `i` with parent `p` becomes the bone named
    /// [`BoneId::name`] with head at joint `p`'s world position and tail at joint `i`'s; a joint
    /// without a parent is a bone whose head and tail are both its world position. Limb joints
    /// are [`BoneClass::Limb`]. The bone indices are the joint indices.
    #[must_use]
    pub fn from_skeleton(skeleton: &Skeleton) -> Self {
        let bones = skeleton
            .joints
            .iter()
            .map(|j| {
                let tail = canon3(j.world_position);
                let head = j
                    .parent
                    .map_or(tail, |p| canon3(skeleton.joints[p].world_position));
                Bone {
                    name: j.id.name().to_string(),
                    parent: j.parent,
                    head,
                    tail,
                    class: if j.id.is_limb() {
                        BoneClass::Limb
                    } else {
                        BoneClass::Torso
                    },
                }
            })
            .collect();
        Self {
            bones,
            height: canon(skeleton.height),
        }
    }

    /// The bones, in index order (parents first).
    #[must_use]
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// The number of bones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    /// Always `false`: a skeleton has at least one bone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// The height that scales the influence radii.
    #[must_use]
    pub const fn height(&self) -> f32 {
        self.height
    }

    /// The bone at `index`.
    #[must_use]
    pub fn bone(&self, index: usize) -> Option<&Bone> {
        self.bones.get(index)
    }

    /// The index of the bone named `name`.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }
}

fn check<'a>(
    bones: impl ExactSizeIterator<Item = (&'a str, Option<usize>)>,
) -> Result<(), BoneSkeletonError> {
    let n = bones.len();
    if n == 0 {
        return Err(BoneSkeletonError::Empty);
    }
    if n > usize::from(u16::MAX) + 1 {
        return Err(BoneSkeletonError::TooManyBones(n));
    }
    let mut names: alloc::collections::BTreeSet<&str> = alloc::collections::BTreeSet::new();
    for (i, (name, parent)) in bones.enumerate() {
        if name.is_empty() {
            return Err(BoneSkeletonError::EmptyName(i));
        }
        if !names.insert(name) {
            return Err(BoneSkeletonError::DuplicateName(i, name.to_string()));
        }
        if let Some(p) = parent {
            if p >= i {
                return Err(BoneSkeletonError::BadParent { bone: i, parent: p });
            }
        }
    }
    Ok(())
}

impl BoneId {
    /// The joint's name as a bone name (`"Hips"`, `"LeftUpperArm"`, …), the variant's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hips => "Hips",
            Self::Spine => "Spine",
            Self::Chest => "Chest",
            Self::Neck => "Neck",
            Self::Head => "Head",
            Self::LeftShoulder => "LeftShoulder",
            Self::LeftUpperArm => "LeftUpperArm",
            Self::LeftLowerArm => "LeftLowerArm",
            Self::LeftHand => "LeftHand",
            Self::RightShoulder => "RightShoulder",
            Self::RightUpperArm => "RightUpperArm",
            Self::RightLowerArm => "RightLowerArm",
            Self::RightHand => "RightHand",
            Self::LeftUpperLeg => "LeftUpperLeg",
            Self::LeftLowerLeg => "LeftLowerLeg",
            Self::LeftFoot => "LeftFoot",
            Self::LeftToe => "LeftToe",
            Self::RightUpperLeg => "RightUpperLeg",
            Self::RightLowerLeg => "RightLowerLeg",
            Self::RightFoot => "RightFoot",
            Self::RightToe => "RightToe",
            Self::LeftFootIk => "LeftFootIk",
            Self::RightFootIk => "RightFootIk",
        }
    }

    /// Whether distance skinning uses the limb radius for this joint (arms, legs, hands, feet
    /// and toes).
    #[must_use]
    pub const fn is_limb(self) -> bool {
        matches!(
            self,
            Self::LeftUpperArm
                | Self::LeftLowerArm
                | Self::LeftHand
                | Self::RightUpperArm
                | Self::RightLowerArm
                | Self::RightHand
                | Self::LeftUpperLeg
                | Self::LeftLowerLeg
                | Self::LeftFoot
                | Self::LeftToe
                | Self::RightUpperLeg
                | Self::RightLowerLeg
                | Self::RightFoot
                | Self::RightToe
        )
    }
}
