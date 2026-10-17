//! Oracles for the stream hello frame (`alice_kinematics::stream`)
//!
//! Expected bytes and results are written here from the layout in the module
//! documentation, not read back from the encoder.

use alice_kinematics::intent::ExtendedIntent;
use alice_kinematics::stream::{
    extended_intents, intents, open_stream, starts_with_hello, HelloPolicy, IdState, PacketKind,
    StreamError, StreamHello, StreamStatus, HELLO_FORMAT, HELLO_LEN, HELLO_MAGIC,
};
use alice_kinematics::{Intent, Vec3k, KIN_SEMANTICS_ID, KIN_WIRE_SEMANTICS_ID};

/// An identifier no build reproduces
const FOREIGN: [u8; 32] = [0x22; 32];

/// CRC-32 (IEEE), written out here independently of the crate's
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
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

/// Rewrites the CRC of a hello after its first 76 bytes were changed
fn reseal(mut frame: [u8; HELLO_LEN]) -> [u8; HELLO_LEN] {
    let crc = crc32(&frame[..76]);
    frame[76..80].copy_from_slice(&crc.to_le_bytes());
    frame
}

fn sample_intents() -> Vec<Intent> {
    (0..5_u8)
        .map(|i| Intent::reach(Vec3k::new(f32::from(i) * 0.25, -0.5, 1.0), 40 + i))
        .collect()
}

fn stream_of(hello: &[u8], packets: &[Intent]) -> Vec<u8> {
    let mut out = hello.to_vec();
    for p in packets {
        out.extend_from_slice(&p.encode());
    }
    out
}

#[test]
fn the_crc_is_the_ieee_check_value() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
}

#[test]
fn the_frame_is_the_documented_layout() {
    let frame = StreamHello::current(PacketKind::Intent).encode();
    assert_eq!(frame.len(), 80);
    assert_eq!(&frame[0..8], b"AKINHELO");
    assert_eq!(&frame[8..10], &1u16.to_le_bytes());
    assert_eq!(frame[10], 8);
    assert_eq!(frame[11], 0);
    assert_eq!(&frame[12..44], &KIN_SEMANTICS_ID);
    assert_eq!(&frame[44..76], &KIN_WIRE_SEMANTICS_ID);
    assert_eq!(&frame[76..80], &crc32(&frame[..76]).to_le_bytes());
    assert_eq!(StreamHello::current(PacketKind::Extended).encode()[10], 16);
    assert_eq!(
        (HELLO_MAGIC, HELLO_FORMAT, HELLO_LEN),
        (*b"AKINHELO", 1, 80)
    );
}

#[test]
fn a_stream_this_build_writes_opens_as_current() {
    let packets = sample_intents();
    let bytes = stream_of(&StreamHello::current(PacketKind::Intent).encode(), &packets);
    for policy in [HelloPolicy::AcceptUnpinned, HelloPolicy::RequirePinned] {
        let (status, rest) = open_stream(&bytes, policy).expect("opened");
        assert_eq!(
            status,
            StreamStatus::Pinned {
                kind: PacketKind::Intent,
                semantics: IdState::Current,
                wire: IdState::Current,
            }
        );
        let read: Vec<Intent> = intents(rest).expect("whole packets").collect();
        assert_eq!(
            read,
            packets
                .iter()
                .map(|p| Intent::decode(&p.encode()))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn an_extended_stream_round_trips() {
    let ext = ExtendedIntent::new(sample_intents()[2], Vec3k::new(0.5, -0.25, 0.0));
    let mut bytes = StreamHello::current(PacketKind::Extended).encode().to_vec();
    bytes.extend_from_slice(&ext.encode());
    let (status, rest) = open_stream(&bytes, HelloPolicy::RequirePinned).expect("opened");
    assert!(matches!(
        status,
        StreamStatus::Pinned {
            kind: PacketKind::Extended,
            ..
        }
    ));
    let read: Vec<ExtendedIntent> = extended_intents(rest).expect("whole packets").collect();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].encode(), ext.encode());
}

// --- the two mismatches, each on its own ------------------------------------

#[test]
fn only_the_wire_identifier_changed_is_a_wire_mismatch() {
    let mut hello = StreamHello::current(PacketKind::Intent);
    hello.wire_id = FOREIGN;
    let bytes = stream_of(&hello.encode(), &sample_intents());
    for policy in [HelloPolicy::AcceptUnpinned, HelloPolicy::RequirePinned] {
        assert_eq!(
            open_stream(&bytes, policy).unwrap_err(),
            StreamError::WireMismatch {
                stored: FOREIGN,
                running: KIN_WIRE_SEMANTICS_ID
            }
        );
    }
}

#[test]
fn only_the_semantics_identifier_changed_is_a_semantics_mismatch() {
    let mut hello = StreamHello::current(PacketKind::Intent);
    hello.semantics_id = FOREIGN;
    let bytes = stream_of(&hello.encode(), &sample_intents());
    for policy in [HelloPolicy::AcceptUnpinned, HelloPolicy::RequirePinned] {
        assert_eq!(
            open_stream(&bytes, policy).unwrap_err(),
            StreamError::SemanticsMismatch {
                stored: FOREIGN,
                running: KIN_SEMANTICS_ID
            }
        );
    }
}

#[test]
fn a_mismatch_message_names_both_identifiers() {
    let hex = |id: &[u8; 32]| {
        use std::fmt::Write as _;
        id.iter().fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
    };
    let mut hello = StreamHello::current(PacketKind::Intent);
    hello.semantics_id = FOREIGN;
    let shown = hello.check().unwrap_err().to_string();
    assert!(
        shown.contains(&hex(&FOREIGN)) && shown.contains(&hex(&KIN_SEMANTICS_ID)),
        "{shown}"
    );
    let mut hello = StreamHello::current(PacketKind::Intent);
    hello.wire_id = FOREIGN;
    let shown = hello.check().unwrap_err().to_string();
    assert!(
        shown.contains(&hex(&FOREIGN)) && shown.contains(&hex(&KIN_WIRE_SEMANTICS_ID)),
        "{shown}"
    );
}

/// A mismatch is a refusal: a hello with a foreign identifier must not be
/// read as a stream without one (which the default policy accepts)
#[test]
fn a_foreign_identifier_is_never_read_as_unpinned() {
    let mut hello = StreamHello::current(PacketKind::Intent);
    hello.semantics_id = FOREIGN;
    hello.wire_id = FOREIGN;
    let bytes = stream_of(&hello.encode(), &sample_intents());
    assert!(open_stream(&bytes, HelloPolicy::AcceptUnpinned).is_err());
}

// --- streams without a hello ------------------------------------------------

#[test]
fn a_stream_without_a_hello_is_unpinned_or_refused() {
    let bytes = stream_of(&[], &sample_intents());
    assert!(!starts_with_hello(&bytes));
    let (status, rest) = open_stream(&bytes, HelloPolicy::AcceptUnpinned).expect("opened");
    assert_eq!(status, StreamStatus::Unpinned);
    assert_eq!(rest, &bytes[..]);
    assert_eq!(
        open_stream(&bytes, HelloPolicy::RequirePinned).unwrap_err(),
        StreamError::Unpinned
    );
    assert_eq!(HelloPolicy::default(), HelloPolicy::AcceptUnpinned);
}

/// A reader before 0.3.0 has only `Intent::decode`, which accepts every
/// 8-byte string: the hello reads as ten intents with no error. This pins
/// the fact the module documentation states (such a reader cannot detect a
/// hello), so a change that made it detectable would show here.
#[test]
fn a_reader_before_0_3_reads_the_hello_as_ten_intents_without_error() {
    let frame = StreamHello::current(PacketKind::Intent).encode();
    let as_intents: Vec<Intent> = frame
        .as_chunks::<8>()
        .0
        .iter()
        .map(Intent::decode)
        .collect();
    assert_eq!(as_intents.len(), 10);
    // "AKINHELO": target from "AK", "IN", "HE" as Q8.8, duration 'L', flags 'O'
    let first = as_intents[0];
    assert_eq!(first.duration_ms, b'L');
    assert_eq!(first.flags.0, b'O');
    assert_eq!(
        first.target.x.to_bits(),
        (f32::from(i16::from_be_bytes(*b"AK")) / 256.0).to_bits()
    );
}

// --- malformed frames -------------------------------------------------------

#[test]
fn an_unknown_hello_format_is_refused_by_number() {
    let mut frame = StreamHello::current(PacketKind::Intent).encode();
    frame[8..10].copy_from_slice(&2u16.to_le_bytes());
    let frame = reseal(frame);
    assert_eq!(
        open_stream(&frame, HelloPolicy::AcceptUnpinned).unwrap_err(),
        StreamError::UnknownFormat(2)
    );
}

#[test]
fn a_corrupted_frame_is_refused() {
    let good = StreamHello::current(PacketKind::Intent).encode();
    for at in [10, 11, 12, 43, 44, 75, 76, 79] {
        let mut frame = good;
        frame[at] ^= 0x01;
        assert_eq!(
            open_stream(&frame, HelloPolicy::AcceptUnpinned).unwrap_err(),
            StreamError::Corrupt,
            "byte {at}"
        );
    }
}

#[test]
fn an_unknown_kind_or_reserved_byte_is_refused() {
    for (at, value) in [(10, 12_u8), (10, 0), (11, 1)] {
        let mut frame = StreamHello::current(PacketKind::Intent).encode();
        frame[at] = value;
        assert_eq!(
            open_stream(&reseal(frame), HelloPolicy::AcceptUnpinned).unwrap_err(),
            StreamError::Malformed,
            "byte {at} = {value}"
        );
    }
}

#[test]
fn a_truncated_frame_or_partial_packet_is_refused() {
    let frame = StreamHello::current(PacketKind::Intent).encode();
    for len in [8, 9, 10, 79] {
        assert_eq!(
            open_stream(&frame[..len], HelloPolicy::AcceptUnpinned).unwrap_err(),
            StreamError::Truncated,
            "length {len}"
        );
    }
    let mut bytes = stream_of(&frame, &sample_intents());
    bytes.truncate(bytes.len() - 3);
    assert_eq!(
        open_stream(&bytes, HelloPolicy::AcceptUnpinned).unwrap_err(),
        StreamError::PartialPacket
    );
    assert!(intents(&[0u8; 9]).is_err());
    assert!(extended_intents(&[0u8; 17]).is_err());
}
