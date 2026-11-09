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
//!   trajectory and the arm inverse kinematics) and the identifier of each
//!   `alice-det-math` function the crate calls ([`DET_MATH_PINS`]), not
//!   det-math's whole identifier, so a function it does not call moves
//!   nothing. Two builds with the same value compute every pinned scenario to
//!   the same bits.
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
//! bytes and the 32 digest bytes. A [`DET_MATH_PINS`] entry goes in under the
//! name `det-math <function>`.
//!
//! # Hex form
//!
//! ```text
//! KIN_WIRE_SEMANTICS_ID d5ae6ab1fdf22312e9c0c6a99d1ea6ab26c218897d2ae045658bfe3decef31ca
//! KIN_SEMANTICS_ID      edf156ec3f794d7a30dd996b7c4b7a484172650c4c02c93c899522863fbebc77
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
//! `tests/kin_semantics_id.rs` until they are checked again. Each
//! [`DET_MATH_PINS`] entry is compared at compile time with the resolved
//! `alice-det-math` (`assert_function_pins`): a moved function does not
//! compile.
//!
//! # Checked by
//!
//! `tests/kin_semantics_id.rs` reads every digest from the golden test that
//! records it (the source, not a copy), checks that [`DET_MATH_FUNCTIONS`] is
//! the set [`DET_MATH_PINS`] names and recomputes both folds. Re-recording
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

/// The entries [`KIN_SEMANTICS_ID`] folds besides [`DET_MATH_PINS`]: every
/// recorded-bit scenario of `tests/semantics_golden.rs`
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
        hex32("38be74d7e546d5e69a4e89236fe6a670804f5f6ead3cfeb62763a630746900c2"),
    ),
];

/// The functions of `alice-det-math` this crate calls (ALICE-Law's
/// `scripts/fold_set_check.py` compares this with the calls in the source)
pub const DET_MATH_FUNCTIONS: &[&str] = &["clamp", "max", "min"];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order, as
/// recorded when [`KIN_SEMANTICS_ID`] was computed. They are folded under the
/// name `det-math <function>`. A function whose identifier moves in the
/// resolved `alice-det-math` stops the compile, rather than reporting an
/// identifier computed from other bits; a function this crate does not call
/// moves nothing.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "clamp",
        hex32("8a35afd329a733bcd0c771a8921fbf21a82491954fc8293881220944dc820455"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// The crate version and toolchain the pins were last checked against
pub const KIN_SEMANTICS_CHECKED_AT: (&str, &str) = ("0.4.0", "1.98.1");

/// Content hash of this crate's arithmetic: the fold of
/// [`KIN_SEMANTICS_PINS`] and [`DET_MATH_PINS`] (each under `det-math
/// <function>`) described in the [module documentation](self)
pub const KIN_SEMANTICS_ID: [u8; 32] =
    hex32("edf156ec3f794d7a30dd996b7c4b7a484172650c4c02c93c899522863fbebc77");

/// Earlier values of [`KIN_SEMANTICS_ID`] whose builds compute every pinned
/// scenario to the same bits as this one (none yet)
pub const KIN_PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// Earlier values of [`KIN_WIRE_SEMANTICS_ID`] whose packets this build reads
/// as the same intents (none yet)
pub const KIN_PREVIOUS_WIRE_SEMANTICS_IDS: &[[u8; 32]] = &[];

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
