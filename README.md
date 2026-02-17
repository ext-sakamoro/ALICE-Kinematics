# ALICE-Kinematics

**Human Motion Intent Compression — 1000Hz raw coordinates → 8-byte kinematic intent**

> "Don't send coordinates. Send the intent."

## The Problem

Current VR/AR devices and gaming peripherals transmit raw X,Y,Z coordinates at 1000Hz — wasting bandwidth, battery, and CPU on data that follows predictable biomechanical laws.

| Device | Raw Data Rate | ALICE Intent Rate | Reduction |
|--------|--------------|-------------------|-----------|
| VR Hand Tracking (2×7-DoF) | 112 KB/s | **~12 bytes/intent** | **~10,000x** |
| Gaming Mouse (1000Hz) | 12 KB/s | **~8 bytes/intent** | **~1,500x** |
| Full Body MoCap (23 joints) | 276 KB/s | **~16 bytes/intent** | **~17,000x** |

## The Solution

Instead of streaming coordinates, ALICE-Kinematics:

1. **Models** human joints as kinematic chains (7-DoF arm, rotation constraints)
2. **Fits** the observed motion to a jerk-minimizing trajectory (Flash-Hogan quintic polynomial)
3. **Encodes** only the **intent** — target position + initial velocity — as an 8-16 byte packet
4. **Predicts** the full trajectory on the receiver side using the same biomechanical model

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  VR/Mouse   │────▶│   Encoder   │────▶│   Network   │────▶│  Predictor  │
│  1000Hz raw │     │ Jerk-min fit│     │  8-16 bytes │     │ Trajectory  │
│  (112 KB/s) │     │  (AGPL-3.0) │     │  per intent │     │ Reconstruct │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                    Intent extraction                        Full-frame
                    Biomechanical model                      interpolation (MIT)
```

## Architecture

### MIT Licensed (Decoder / Format)

- **`joint`** — 7-DoF joint chain, rotation constraints, FK/IK
- **`intent`** — Compact intent packet format (8-16 bytes)
- **`predictor`** — Trajectory reconstruction from intent (dead reckoning, interpolation)

### AGPL-3.0 Licensed (Encoder, feature `encoder`)

- **`jerk`** — Flash-Hogan jerk minimization, quintic polynomial fitting
- **`encoder`** — Raw sensor → Intent extraction pipeline

## Quick Start

```rust
use alice_kinematics::{Intent, ArmChain, Predictor};

// Receiver side (MIT) — reconstruct trajectory from intent
let intent = Intent::decode(&packet_bytes);
let mut predictor = Predictor::new();
predictor.apply_intent(intent);

// Get interpolated position at any time
let pos = predictor.position_at(0.016); // 16ms into the motion
```

```rust
// Encoder side (AGPL-3.0) — requires feature "encoder"
use alice_kinematics::encoder::{IntentEncoder, SensorSample};

let mut encoder = IntentEncoder::new();
encoder.push_sample(SensorSample { pos, vel, timestamp });

if let Some(intent) = encoder.extract_intent() {
    let packet = intent.encode(); // 8-16 bytes
    network.send(&packet);
}
```

## Mathematical Foundation

### Jerk Minimization (Flash-Hogan 1985)

Human reaching movements minimize the integral of squared jerk:

```
        T
J = ∫  |d³x/dt³|² dt  →  minimize
    0
```

The optimal solution is a **quintic (5th-degree) polynomial**:

```
x(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
```

Given boundary conditions (start position, velocity, acceleration; end position, velocity, acceleration), the 6 coefficients are uniquely determined. Only the **target position and movement duration** need to be transmitted.

### Intent Packet Format (8 bytes)

```
┌──────────────────────────────────────────────────────────────────┐
│  Byte 0-1  │  Byte 2-3  │  Byte 4-5  │  Byte 6   │  Byte 7   │
│  target_x  │  target_y  │  target_z  │  duration │  flags    │
│  (i16 q8)  │  (i16 q8)  │  (i16 q8)  │  (u8 ms)  │  (u8)     │
└──────────────────────────────────────────────────────────────────┘
```

- **target_xyz**: Fixed-point Q8.8 (±127.996 range, 0.004 resolution)
- **duration**: Movement duration in milliseconds (0-255ms)
- **flags**: Joint mask, grip state, intent type

## Memory Footprint

| Component | Size |
|-----------|------|
| ArmChain (7-DoF) | 224 bytes |
| Intent packet | 8-16 bytes |
| Predictor state | 128 bytes |
| IntentEncoder | 512 bytes |
| **Total (decoder)** | **< 400 bytes** |

## License

- **Core / Decoder**: MIT — joint model, intent format, predictor
- **Encoder** (feature `encoder`): AGPL-3.0 — jerk minimization, sensor → intent extraction

See [LICENSE](LICENSE) and [LICENSE-AGPL](LICENSE-AGPL).

## Author

Moroya Sakamoto
