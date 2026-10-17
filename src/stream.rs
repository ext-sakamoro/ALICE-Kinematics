//! Stream hello frame: the identifiers a packet stream or a recorded file
//! carries
//!
//! Intent packets (8 bytes, [`Intent`]) and extended packets (16 bytes,
//! [`ExtendedIntent`]) use every bit for the motion itself, so they carry no
//! identifier. A stream (a live session or a recorded file) instead starts
//! with one 80-byte hello frame:
//!
//! ```text
//! offset  size  field
//!      0     8  magic "AKINHELO"
//!      8     2  format, u16 little-endian (HELLO_FORMAT = 1)
//!     10     1  packet kind: 8 (Intent) or 16 (ExtendedIntent)
//!     11     1  reserved, 0
//!     12    32  semantics identifier of the producer (KIN_SEMANTICS_ID)
//!     44    32  wire identifier of the packet format (KIN_WIRE_SEMANTICS_ID)
//!     76     4  CRC-32 (IEEE) of bytes 0..76, little-endian
//! ```
//!
//! followed by the packets of that kind. [`open_stream`] reads the frame,
//! checks both identifiers against this build's, and returns the packet bytes.
//!
//! # What each mismatch means to a consumer
//!
//! - [`StreamError::WireMismatch`]: the stored bytes have a different meaning
//!   (another packet format). Decoding them with this build would misread
//!   them, so the stream is refused.
//! - [`StreamError::SemanticsMismatch`]: the packet format is the same, but
//!   the producer computed the intents with other arithmetic, so this build
//!   would not reproduce the same bits from the same input. The packets
//!   would decode, but they are not this build's Law output, so the stream is
//!   refused.
//!
//! An identifier listed in `KIN_PREVIOUS_SEMANTICS_IDS` /
//! `KIN_PREVIOUS_WIRE_SEMANTICS_IDS` (builds verified to give the same bits)
//! is accepted and reported as [`IdState::VerifiedPrevious`].
//!
//! # Streams without a hello
//!
//! A stream from a writer before 0.3.0 has no frame. Under
//! [`HelloPolicy::AcceptUnpinned`] (the default) [`open_stream`] returns
//! [`StreamStatus::Unpinned`] and all bytes as packets; under
//! [`HelloPolicy::RequirePinned`] it refuses with [`StreamError::Unpinned`].
//!
//! A reader before 0.3.0 cannot detect a hello: [`Intent::decode`] and
//! [`ExtendedIntent::decode`] accept every byte string (every flags value is
//! valid), so the 80-byte frame reads as ten 8-byte intents (or five 16-byte
//! ones) with no error. A stream that starts with a hello must therefore go
//! only to readers of 0.3.0 or later; the transport has to agree on that.
//!
//! # What the CRC does and does not do
//!
//! The CRC-32 detects accidental corruption of the frame. It is not what
//! refuses another build's stream: that is the comparison of the identifiers.
//! A stream without a hello is mistaken for one only if its first 76 bytes
//! begin with the magic, format 1, a known kind and a zero reserved byte and
//! its next 4 bytes are their CRC-32.

use core::fmt;

use crate::intent::ExtendedIntent;
use crate::semantics::{
    same32, KIN_PREVIOUS_SEMANTICS_IDS, KIN_PREVIOUS_WIRE_SEMANTICS_IDS, KIN_SEMANTICS_ID,
    KIN_WIRE_SEMANTICS_ID,
};
use crate::Intent;

/// First 8 bytes of a hello frame
pub const HELLO_MAGIC: [u8; 8] = *b"AKINHELO";
/// The hello format this build writes and reads
pub const HELLO_FORMAT: u16 = 1;
/// Length of a format 1 hello frame in bytes
pub const HELLO_LEN: usize = 80;

/// Which packets follow the hello
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketKind {
    /// 8-byte [`Intent`] packets
    Intent,
    /// 16-byte [`ExtendedIntent`] packets
    Extended,
}

impl PacketKind {
    /// Length of one packet in bytes
    #[must_use]
    pub const fn packet_len(self) -> usize {
        match self {
            Self::Intent => 8,
            Self::Extended => 16,
        }
    }

    const fn from_byte(b: u8) -> Option<Self> {
        match b {
            8 => Some(Self::Intent),
            16 => Some(Self::Extended),
            _ => None,
        }
    }
}

/// Whether a stream without a hello is accepted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HelloPolicy {
    /// Read a stream without a hello as [`StreamStatus::Unpinned`]
    #[default]
    AcceptUnpinned,
    /// Refuse a stream without a hello ([`StreamError::Unpinned`])
    RequirePinned,
}

/// How a stored identifier relates to this build's
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdState {
    /// Equal to this build's identifier
    Current,
    /// An earlier identifier verified to give the same bits
    VerifiedPrevious([u8; 32]),
}

/// What [`open_stream`] found at the start of a stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    /// A hello whose identifiers this build accepts
    Pinned {
        /// Which packets follow
        kind: PacketKind,
        /// The producer's semantics identifier
        semantics: IdState,
        /// The packet format identifier
        wire: IdState,
    },
    /// No hello: a stream from a writer before 0.3.0
    Unpinned,
}

/// Why a stream was refused
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamError {
    /// Fewer bytes than the hello frame needs
    Truncated,
    /// A hello format this build does not know
    UnknownFormat(u16),
    /// A packet kind other than 8 or 16, or a non-zero reserved byte
    Malformed,
    /// The CRC-32 of the hello does not match (accidental corruption)
    Corrupt,
    /// The packet format identifier is not this build's
    WireMismatch {
        /// Identifier in the hello
        stored: [u8; 32],
        /// This build's identifier
        running: [u8; 32],
    },
    /// The producer's semantics identifier is not this build's
    SemanticsMismatch {
        /// Identifier in the hello
        stored: [u8; 32],
        /// This build's identifier
        running: [u8; 32],
    },
    /// No hello under [`HelloPolicy::RequirePinned`]
    Unpinned,
    /// The bytes after the hello are not a whole number of packets
    PartialPacket,
}

fn write_hex(f: &mut fmt::Formatter<'_>, id: &[u8; 32]) -> fmt::Result {
    for b in id {
        write!(f, "{b:02x}")?;
    }
    Ok(())
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("stream hello: truncated"),
            Self::UnknownFormat(v) => write!(f, "stream hello: unknown format {v}"),
            Self::Malformed => {
                f.write_str("stream hello: unknown packet kind or reserved byte set")
            }
            Self::Corrupt => f.write_str("stream hello: CRC-32 mismatch"),
            Self::WireMismatch { stored, running } => {
                f.write_str("stream packet format identifier ")?;
                write_hex(f, stored)?;
                f.write_str(" is not this build's ")?;
                write_hex(f, running)
            }
            Self::SemanticsMismatch { stored, running } => {
                f.write_str("stream producer semantics identifier ")?;
                write_hex(f, stored)?;
                f.write_str(" is not this build's ")?;
                write_hex(f, running)
            }
            Self::Unpinned => f.write_str("stream has no hello and a pin is required"),
            Self::PartialPacket => f.write_str("stream ends inside a packet"),
        }
    }
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320)
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A hello frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHello {
    /// Which packets follow
    pub kind: PacketKind,
    /// The producer's semantics identifier
    pub semantics_id: [u8; 32],
    /// The packet format identifier
    pub wire_id: [u8; 32],
}

impl StreamHello {
    /// The hello this build writes before packets of `kind`
    #[must_use]
    pub const fn current(kind: PacketKind) -> Self {
        Self {
            kind,
            semantics_id: KIN_SEMANTICS_ID,
            wire_id: KIN_WIRE_SEMANTICS_ID,
        }
    }

    /// The 80-byte frame
    #[must_use]
    pub fn encode(&self) -> [u8; HELLO_LEN] {
        let mut out = [0u8; HELLO_LEN];
        out[0..8].copy_from_slice(&HELLO_MAGIC);
        out[8..10].copy_from_slice(&HELLO_FORMAT.to_le_bytes());
        #[expect(clippy::cast_possible_truncation, reason = "packet_len is 8 or 16")]
        {
            out[10] = self.kind.packet_len() as u8;
        }
        out[12..44].copy_from_slice(&self.semantics_id);
        out[44..76].copy_from_slice(&self.wire_id);
        let crc = crc32(&out[..76]);
        out[76..80].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Reads a frame from the start of `bytes` (identifiers not checked)
    ///
    /// # Errors
    /// [`StreamError::Truncated`], [`StreamError::UnknownFormat`],
    /// [`StreamError::Corrupt`] and [`StreamError::Malformed`]. `bytes` that
    /// do not start with [`HELLO_MAGIC`] are not a hello; check with
    /// [`starts_with_hello`] first.
    pub fn decode(bytes: &[u8]) -> Result<Self, StreamError> {
        if bytes.len() < 10 {
            return Err(StreamError::Truncated);
        }
        let format = u16::from_le_bytes([bytes[8], bytes[9]]);
        if format != HELLO_FORMAT {
            return Err(StreamError::UnknownFormat(format));
        }
        if bytes.len() < HELLO_LEN {
            return Err(StreamError::Truncated);
        }
        let stored_crc = u32::from_le_bytes([bytes[76], bytes[77], bytes[78], bytes[79]]);
        if crc32(&bytes[..76]) != stored_crc {
            return Err(StreamError::Corrupt);
        }
        let kind = PacketKind::from_byte(bytes[10]).ok_or(StreamError::Malformed)?;
        if bytes[11] != 0 {
            return Err(StreamError::Malformed);
        }
        let mut semantics_id = [0u8; 32];
        semantics_id.copy_from_slice(&bytes[12..44]);
        let mut wire_id = [0u8; 32];
        wire_id.copy_from_slice(&bytes[44..76]);
        Ok(Self {
            kind,
            semantics_id,
            wire_id,
        })
    }

    /// Compares both identifiers with this build's
    ///
    /// # Errors
    /// [`StreamError::WireMismatch`] when the packet format identifier is not
    /// this build's or a verified earlier one, then
    /// [`StreamError::SemanticsMismatch`] for the semantics identifier.
    pub fn check(&self) -> Result<StreamStatus, StreamError> {
        let wire = id_state(
            &self.wire_id,
            &KIN_WIRE_SEMANTICS_ID,
            KIN_PREVIOUS_WIRE_SEMANTICS_IDS,
        )
        .ok_or(StreamError::WireMismatch {
            stored: self.wire_id,
            running: KIN_WIRE_SEMANTICS_ID,
        })?;
        let semantics = id_state(
            &self.semantics_id,
            &KIN_SEMANTICS_ID,
            KIN_PREVIOUS_SEMANTICS_IDS,
        )
        .ok_or(StreamError::SemanticsMismatch {
            stored: self.semantics_id,
            running: KIN_SEMANTICS_ID,
        })?;
        Ok(StreamStatus::Pinned {
            kind: self.kind,
            semantics,
            wire,
        })
    }
}

fn id_state(stored: &[u8; 32], running: &[u8; 32], previous: &[[u8; 32]]) -> Option<IdState> {
    if same32(stored, running) {
        return Some(IdState::Current);
    }
    previous
        .iter()
        .find(|p| same32(p, stored))
        .map(|p| IdState::VerifiedPrevious(*p))
}

/// Whether `bytes` start with [`HELLO_MAGIC`]
#[must_use]
pub fn starts_with_hello(bytes: &[u8]) -> bool {
    bytes.len() >= HELLO_MAGIC.len() && bytes[..HELLO_MAGIC.len()] == HELLO_MAGIC
}

/// Reads the start of a stream: the hello (checked) and the packet bytes
/// after it
///
/// A stream without a hello is [`StreamStatus::Unpinned`] under
/// [`HelloPolicy::AcceptUnpinned`], and all of `bytes` are packet bytes; its
/// packet kind is the caller's (pre-0.3.0 streams did not say).
///
/// # Errors
/// Every error of [`StreamHello::decode`] and [`StreamHello::check`],
/// [`StreamError::Unpinned`] under [`HelloPolicy::RequirePinned`], and
/// [`StreamError::PartialPacket`] when the bytes after a hello are not a
/// whole number of packets of its kind.
pub fn open_stream(
    bytes: &[u8],
    policy: HelloPolicy,
) -> Result<(StreamStatus, &[u8]), StreamError> {
    if !starts_with_hello(bytes) {
        return match policy {
            HelloPolicy::AcceptUnpinned => Ok((StreamStatus::Unpinned, bytes)),
            HelloPolicy::RequirePinned => Err(StreamError::Unpinned),
        };
    }
    let hello = StreamHello::decode(bytes)?;
    let status = hello.check()?;
    let rest = &bytes[HELLO_LEN..];
    if !rest.len().is_multiple_of(hello.kind.packet_len()) {
        return Err(StreamError::PartialPacket);
    }
    Ok((status, rest))
}

/// The 8-byte packets of a stream [`open_stream`] accepted with
/// [`PacketKind::Intent`]
///
/// # Errors
/// [`StreamError::PartialPacket`] when `packets` is not a whole number of
/// 8-byte packets.
pub fn intents(packets: &[u8]) -> Result<impl Iterator<Item = Intent> + '_, StreamError> {
    let (chunks, tail) = packets.as_chunks::<8>();
    if !tail.is_empty() {
        return Err(StreamError::PartialPacket);
    }
    Ok(chunks.iter().map(Intent::decode))
}

/// The 16-byte packets of a stream [`open_stream`] accepted with
/// [`PacketKind::Extended`]
///
/// # Errors
/// [`StreamError::PartialPacket`] when `packets` is not a whole number of
/// 16-byte packets.
pub fn extended_intents(
    packets: &[u8],
) -> Result<impl Iterator<Item = ExtendedIntent> + '_, StreamError> {
    let (chunks, tail) = packets.as_chunks::<16>();
    if !tail.is_empty() {
        return Err(StreamError::PartialPacket);
    }
    Ok(chunks.iter().map(ExtendedIntent::decode))
}
