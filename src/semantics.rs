//! Content hashes of this crate's arithmetic and of its packet format
//!
//! Two identifiers, each a SHA-256 fold of recorded-bit digests from
//! `tests/semantics_golden.rs`:
//!
//! - [`KIN_WIRE_SEMANTICS_ID`]: what stored packet bytes mean. It folds the
//!   scenarios that encode and decode the 8-byte [`Intent`](crate::Intent)
//!   and 16-byte [`ExtendedIntent`](crate::intent::ExtendedIntent) packets.
//!   Two builds with the same value read the same bytes as the same intents.
//! - [`KIN_SEMANTICS_ID`]: the arithmetic this build computes intents with.
//!   It folds every recorded scenario (the packet format, the quintic
//!   trajectory and the arm inverse kinematics) and the identifier of
//!   `alice-det-math`, whose helpers the crate calls. Two builds with the same
//!   value compute every pinned scenario to the same bits.
//!
//! The stream hello frame ([`crate::stream`]) carries both: a reader checks
//! the wire identifier before it decodes stored packets (the meaning of the
//! bytes) and the semantics identifier to know whether the producer computed
//! the intents with this build's arithmetic.
//!
//! # Fold
//!
//! The procedure of the `SEMANTICS_ID` fold of `alice-det-math`: sort the
//! entries by name (ascending byte order), then SHA-256 over, for each entry
//! in that order, the name length as 4 bytes big-endian (`u32`), the name
//! bytes and the 32 digest bytes.
//!
//! # Hex form (with `alice-det-math` 0.5.1)
//!
//! ```text
//! KIN_WIRE_SEMANTICS_ID d5ae6ab1fdf22312e9c0c6a99d1ea6ab26c218897d2ae045658bfe3decef31ca
//! KIN_SEMANTICS_ID      440a6e992e9d9697f9026154d68aaaa14e4094b317a66308f6bccdfe43d4df63
//! ```
//!
//! # Independent of cargo features
//!
//! The tables and identifiers are constants that no `cfg` gates, so a build
//! with any feature set reports the same identifiers.
//!
//! # Staleness
//!
//! [`KIN_SEMANTICS_CHECKED_AT`] names the crate version and toolchain the pins
//! were last checked on; a different version or toolchain fails
//! `tests/kin_semantics_id.rs` until they are checked again. An
//! `alice-det-math` generation with no row in [`KIN_SEMANTICS_BY_DET_MATH`]
//! does not compile.
//!
//! # Checked by
//!
//! `tests/kin_semantics_id.rs` reads every digest from the golden test that
//! records it (the source, not a copy), checks that the det-math entry equals
//! `alice_det_math::SEMANTICS_ID` and recomputes both folds. Re-recording
//! order after an intended change: update the golden first, then the entry
//! here, then the identifiers.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The entries [`KIN_WIRE_SEMANTICS_ID`] folds: the packet encode and decode
/// scenarios of `tests/semantics_golden.rs`, named as that file names them
pub const KIN_WIRE_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "intent encode",
        hex32("a4bdb7231e2252f6fda34ec652a2ca1aa2159f60c8c3a08a1ccb84cff751c5d4"),
    ),
    (
        "extended intent encode",
        hex32("421e2fe9bd513e92288fe41e7f87015d0c9cd68a025705cca88e3a9d7d19166f"),
    ),
    (
        "packet decode",
        hex32("31ff5fcd3686b9d00bb0a9946ed9b899aa0ce4cc22ec556d332cec3030a2bc98"),
    ),
];

/// Content hash of the packet format: the fold of
/// [`KIN_WIRE_SEMANTICS_PINS`] described in the [module documentation](self)
pub const KIN_WIRE_SEMANTICS_ID: [u8; 32] =
    hex32("d5ae6ab1fdf22312e9c0c6a99d1ea6ab26c218897d2ae045658bfe3decef31ca");

/// The entries [`KIN_SEMANTICS_ID`] folds: every recorded-bit scenario of
/// `tests/semantics_golden.rs` plus `"alice-det-math"` holding
/// `alice_det_math::SEMANTICS_ID`
pub const KIN_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "intent encode",
        hex32("a4bdb7231e2252f6fda34ec652a2ca1aa2159f60c8c3a08a1ccb84cff751c5d4"),
    ),
    (
        "extended intent encode",
        hex32("421e2fe9bd513e92288fe41e7f87015d0c9cd68a025705cca88e3a9d7d19166f"),
    ),
    (
        "packet decode",
        hex32("31ff5fcd3686b9d00bb0a9946ed9b899aa0ce4cc22ec556d332cec3030a2bc98"),
    ),
    (
        "quintic trajectory",
        hex32("70f8dfeaa9cfed81800d3fe470102b6df60988881a9bdc940ceefe4c3e82a0dd"),
    ),
    (
        "arm inverse kinematics",
        hex32("287771cc5d7cf2d0bdf00c95c8067f0b2154d3bc069a9ee8d8cca82343d44469"),
    ),
    // the resolved alice-det-math's own identifier, not a copy of it: the
    // identifier below is looked up by it
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// The crate version and toolchain the pins were last checked against
pub const KIN_SEMANTICS_CHECKED_AT: (&str, &str) = ("0.3.0", "1.98.1");

/// [`KIN_SEMANTICS_ID`] for each `alice-det-math` generation this crate
/// knows: `(alice_det_math::SEMANTICS_ID, fold of KIN_SEMANTICS_PINS with that
/// det-math entry)`. A build against a generation with no row does not
/// compile, rather than reporting an identifier of another generation.
pub const KIN_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.1
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("440a6e992e9d9697f9026154d68aaaa14e4094b317a66308f6bccdfe43d4df63"),
    ),
];

/// Content hash of this crate's arithmetic: the fold of
/// [`KIN_SEMANTICS_PINS`], taken from [`KIN_SEMANTICS_BY_DET_MATH`] by the
/// resolved `alice-det-math`
pub const KIN_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_det_math::SEMANTICS_ID);

/// Earlier values of [`KIN_SEMANTICS_ID`] whose builds compute every pinned
/// scenario to the same bits as this one (none yet)
pub const KIN_PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// Earlier values of [`KIN_WIRE_SEMANTICS_ID`] whose packets this build reads
/// as the same intents (none yet)
pub const KIN_PREVIOUS_WIRE_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// The [`KIN_SEMANTICS_ID`] of a build that resolved the `alice-det-math`
/// with identifier `det_math`.
///
/// # Panics
///
/// When [`KIN_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const
/// context this stops the compile).
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < KIN_SEMANTICS_BY_DET_MATH.len() {
        if same32(&KIN_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return KIN_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in KIN_SEMANTICS_BY_DET_MATH: add its fold");
}

pub(crate) const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
