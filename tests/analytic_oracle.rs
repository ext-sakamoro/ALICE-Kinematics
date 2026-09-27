//! Analytic oracles — closed-form checks for the kinematic laws in
//! ALICE-Kinematics (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or f64 references written in this
//! file, never from the crate function under test.  `Predictor::default()`
//! and `JerkFitter::default()` are the paths a consumer takes first.
//!
//! Oracle sources:
//! - Flash & Hogan (1985) minimum-jerk: x(τ) = x₀ + D(10τ³ − 15τ⁴ + 6τ⁵) for
//!   rest-to-rest; the six boundary conditions x/v/a at 0 and T for any start
//!   state; ∫ jerk² dt = 720 D²/T⁵; peak velocity 15D/(8T) at τ = ½
//! - Fitts (1954) / Shannon form: MT = a + b·log₂(2D/W)
//! - Rodrigues rotation; forward kinematics of the published right-arm chain
//!   as an independent f64 Rodrigues chain (Bhaskara sine bound 1.63e-3 per
//!   rotation ⇒ tolerance derived, not measured)
//! - CCD inverse kinematics: reachable target ⇒ tip error ≤ tolerance, and
//!   the error only decreases with more iterations
//! - Predictor integration is dt-independent (it evaluates the polynomial)
//! - skeleton scaling is proportional; hand grip is monotone
//!
//! Lint policy: the f32 → f64 widenings an oracle needs are written `f64::from`;
//! the four allows below are the remaining *intentional* narrowings and are
//! justified per lint — none of them is a blanket silencing.
#![allow(
    // f64 oracle values are compared against f32 tolerances (`1e-6 * scale as f32`)
    // and f64 references are narrowed back to f32 to feed the crate under test.
    clippy::cast_possible_truncation,
    // loop indices and sample counts turned into step sizes; every value here is
    // ≤ 100, far inside f32's exact integer range.
    clippy::cast_precision_loss,
    // these compare against sentinels the implementation returns as literals, not
    // against computed values: `remaining()` returns 0.0 when !active,
    // `progress()` clamps to 1.0, `from_boundary(T < 1e-6)` sets c = [xf, 0, …],
    // `minimum_jerk_cost(_, 0.0)` returns f32::MAX, and the `dot` / `lerp` results
    // (11.0, 1.0, −2.0) are exactly representable.
    clippy::float_cmp,
    // x0 / xf / v0 / a0 / t / d / k / s / c are the symbols of the Flash & Hogan
    // quintic and of Rodrigues' rotation formula.
    clippy::many_single_char_names
)]

use alice_kinematics::intent::Intent;
use alice_kinematics::joint::{rotate_vec, ArmChain, JointConstraint, Vec3k};
use alice_kinematics::predictor::{Predictor, QuinticCoeffs};
use alice_kinematics::skeleton::{BoneId, Skeleton};

fn s_of(tau: f64) -> f64 {
    10.0 * tau.powi(3) - 15.0 * tau.powi(4) + 6.0 * tau.powi(5)
}

// ───────────────────────── minimum-jerk quintic ───────────────────────────

#[test]
fn quintic_rest_to_rest_is_the_flash_hogan_polynomial() {
    let (x0, xf, t_total) = (0.2f32, 1.7f32, 0.8f32);
    let q = QuinticCoeffs::from_boundary(x0, 0.0, 0.0, xf, t_total);
    let d = f64::from(xf - x0);
    for i in 0..=100 {
        let tau = f64::from(i) / 100.0;
        let t = (tau * f64::from(t_total)) as f32;
        let expected = f64::from(x0) + d * s_of(tau);
        assert!(
            (f64::from(q.position(t)) - expected).abs() < 2e-6,
            "x(τ={tau}): {} vs {expected}",
            q.position(t)
        );
        // v = D/T·(30τ² − 60τ³ + 30τ⁴), a = D/T²·(60τ − 180τ² + 120τ³)
        let v =
            d / f64::from(t_total) * (30.0 * tau.powi(2) - 60.0 * tau.powi(3) + 30.0 * tau.powi(4));
        let a = d / (f64::from(t_total)).powi(2)
            * (60.0 * tau - 180.0 * tau.powi(2) + 120.0 * tau.powi(3));
        assert!(
            (f64::from(q.velocity(t)) - v).abs() < 2e-5,
            "v(τ={tau}): {} vs {v}",
            q.velocity(t)
        );
        assert!(
            (f64::from(q.acceleration(t)) - a).abs() < 2e-4,
            "a(τ={tau}): {} vs {a}",
            q.acceleration(t)
        );
    }
    // peak velocity 15D/(8T) at the midpoint, zero at both ends
    let v_peak = 15.0 * d / (8.0 * f64::from(t_total));
    assert!((f64::from(q.velocity(t_total / 2.0)) - v_peak).abs() < 1e-5);
    assert!(q.velocity(0.0).abs() < 1e-6 && q.velocity(t_total).abs() < 1e-5);
    assert!(q.acceleration(0.0).abs() < 1e-6 && q.acceleration(t_total).abs() < 1e-4);
}

#[test]
fn quintic_boundary_conditions_hold_for_any_start_state() {
    for (x0, v0, a0, xf, t) in [
        (0.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32),
        (0.5, 2.0, -3.0, -0.4, 0.25),
        (-1.0, -0.5, 4.0, 3.0, 0.15),
        (10.0, 0.0, 0.0, 10.0, 0.5), // no displacement
    ] {
        let q = QuinticCoeffs::from_boundary(x0, v0, a0, xf, t);
        let scale = f64::from(
            (xf - x0)
                .abs()
                .max(v0.abs() * t)
                .max(a0.abs() * t * t)
                .max(1.0),
        );
        // oracle: the six boundary conditions that define the quintic
        assert!((q.position(0.0) - x0).abs() < 1e-6 * scale as f32, "x(0)");
        assert!((q.velocity(0.0) - v0).abs() < 1e-6 * scale as f32, "v(0)");
        assert!(
            (q.acceleration(0.0) - a0).abs() < 1e-5 * scale as f32,
            "a(0)"
        );
        assert!(
            (f64::from(q.position(t)) - f64::from(xf)).abs() < 1e-5 * scale,
            "x(T) = {} vs {xf}",
            q.position(t)
        );
        assert!(
            (f64::from(q.velocity(t))).abs() < 1e-4 * scale / f64::from(t),
            "v(T) = {}",
            q.velocity(t)
        );
        assert!(
            (f64::from(q.acceleration(t))).abs() < 1e-3 * scale / f64::from(t * t),
            "a(T) = {}",
            q.acceleration(t)
        );
        // velocity / acceleration are the derivatives of position (f64 central difference)
        for k in 1..8 {
            let tt = t * k as f32 / 8.0;
            let h = 1e-3f32 * t;
            let dv = (f64::from(q.position(tt + h)) - f64::from(q.position(tt - h)))
                / (2.0 * f64::from(h));
            assert!(
                (f64::from(q.velocity(tt)) - dv).abs() < 1e-3 * scale / f64::from(t) + 1e-4,
                "dx/dt at {tt}"
            );
            let da = (f64::from(q.velocity(tt + h)) - f64::from(q.velocity(tt - h)))
                / (2.0 * f64::from(h));
            assert!(
                (f64::from(q.acceleration(tt)) - da).abs() < 1e-2 * scale / f64::from(t * t) + 1e-3,
                "dv/dt at {tt}"
            );
        }
    }
    // degenerate duration: parks at the target
    let q = QuinticCoeffs::from_boundary(1.0, 5.0, 5.0, 2.0, 0.0);
    assert_eq!(q.position(0.0), 2.0);
}

#[cfg(feature = "encoder")]
#[test]
fn jerk_cost_and_fitts_law_match_their_closed_forms() {
    use alice_kinematics::jerk::{fitts_law_duration, minimum_jerk_cost};
    // ∫₀ᵀ (d³x/dt³)² dt for the rest-to-rest quintic = 720 D²/T⁵
    for (d, t) in [(1.0f32, 1.0f32), (0.5, 0.25), (2.0, 3.0)] {
        let expected = 720.0 * (f64::from(d)).powi(2) / (f64::from(t)).powi(5);
        assert!(
            (f64::from(minimum_jerk_cost(d, t)) - expected).abs() < 1e-4 * expected,
            "cost({d},{t})"
        );
    }
    assert_eq!(minimum_jerk_cost(1.0, 0.0), f32::MAX);
    // Fitts: MT = a + b·log₂(2D/W)  (the crate's log2 is an approximation — bound 1 %)
    for (dist, w) in [(0.5f32, 0.05f32), (1.0, 0.1), (0.2, 0.2), (2.0, 0.01)] {
        let expected = 0.1 + 0.15 * (2.0 * f64::from(dist) / f64::from(w)).log2();
        let got = f64::from(fitts_law_duration(dist, w, 0.1, 0.15));
        assert!(
            (got - expected).abs() < 0.01 * expected.abs() + 1e-3,
            "fitts({dist},{w}): {got} vs {expected}"
        );
    }
}

// ───────────────────────── rotation / forward kinematics ──────────────────

fn rodrigues64(v: [f64; 3], k: [f64; 3], theta: f64) -> [f64; 3] {
    let n = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let k = [k[0] / n, k[1] / n, k[2] / n];
    let (s, c) = theta.sin_cos();
    let kv = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    let kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    [
        v[0] * c + kv[0] * s + k[0] * kd * (1.0 - c),
        v[1] * c + kv[1] * s + k[1] * kd * (1.0 - c),
        v[2] * c + kv[2] * s + k[2] * kd * (1.0 - c),
    ]
}

/// Bhaskara I sine: |Δsin| ≤ 1.63e-3, so one Rodrigues rotation of a unit
/// vector is off by at most ≈ 3·1.63e-3 (three trig-weighted terms)
const ROT_BOUND: f64 = 5e-3;

#[test]
fn rotate_vec_is_rodrigues_and_preserves_length() {
    let v = Vec3k::new(0.3, -0.7, 0.2);
    for k in [
        Vec3k::new(1.0, 0.0, 0.0),
        Vec3k::new(0.0, 1.0, 0.0),
        Vec3k::new(1.0, 1.0, 1.0),
        Vec3k::new(-0.2, 0.5, 0.9),
    ] {
        for i in -12..=12 {
            let theta = i as f32 * 0.25;
            let r = rotate_vec(v, k, theta);
            let e = rodrigues64(
                [f64::from(v.x), f64::from(v.y), f64::from(v.z)],
                [f64::from(k.x), f64::from(k.y), f64::from(k.z)],
                f64::from(theta),
            );
            let err = ((f64::from(r.x) - e[0]).powi(2)
                + (f64::from(r.y) - e[1]).powi(2)
                + (f64::from(r.z) - e[2]).powi(2))
            .sqrt();
            assert!(
                err < ROT_BOUND * f64::from(v.length()),
                "θ={theta} axis={k:?}: err {err}"
            );
            assert!(
                ((r.length() - v.length()) / v.length()).abs() < ROT_BOUND as f32,
                "length"
            );
        }
    }
    // exact special cases: θ = 0 identity, axis ∥ v invariant
    let r = rotate_vec(v, Vec3k::new(0.0, 0.0, 1.0), 0.0);
    assert!(r.distance(v) < 1e-6);
    let r = rotate_vec(v, v, 1.3);
    assert!(r.distance(v) < ROT_BOUND as f32 * v.length());
    // Vec3k algebra
    let (a, b) = (Vec3k::new(1.0, 2.0, 3.0), Vec3k::new(-2.0, 0.5, 4.0));
    assert_eq!(a.dot(b), 11.0);
    let c = a.cross(b);
    assert_eq!((c.x, c.y, c.z), (6.5, -10.0, 4.5));
    // fast inverse sqrt, 2 Newton steps (Lomont): ≤ 4.7e-6 relative
    assert!((a.normalize().length() - 1.0).abs() < 5e-6);
    assert_eq!(a.lerp(b, 0.0).x, 1.0);
    assert_eq!(a.lerp(b, 1.0).x, -2.0);
}

#[test]
fn right_arm_forward_kinematics_matches_an_independent_rodrigues_chain() {
    let mut arm = ArmChain::right_arm();
    let total: f32 = arm.total_length();
    // all joints at zero: the chain hangs straight down the −y axis
    arm.set_angles(&[0.0; 7]);
    let tip = arm.forward_kinematics();
    assert!((tip.x).abs() < 1e-6 && (tip.z).abs() < 1e-6, "{tip:?}");
    assert!((tip.y + total).abs() < 1e-6, "tip {} vs −{total}", tip.y);
    // arbitrary pose: independent f64 chain over the published axes / lengths
    let angles = [0.4f32, -0.3, 0.9, 1.2, -0.5, 0.7, 0.2];
    arm.set_angles(&angles);
    let tip = arm.forward_kinematics();
    let mut pos = [
        f64::from(arm.base.x),
        f64::from(arm.base.y),
        f64::from(arm.base.z),
    ];
    let mut dir = [0.0f64, -1.0, 0.0];
    for j in &arm.joints {
        dir = rodrigues64(
            dir,
            [
                f64::from(j.axis.x),
                f64::from(j.axis.y),
                f64::from(j.axis.z),
            ],
            f64::from(j.angle),
        );
        for d in 0..3 {
            pos[d] += dir[d] * f64::from(j.link_length);
        }
    }
    let err = ((f64::from(tip.x) - pos[0]).powi(2)
        + (f64::from(tip.y) - pos[1]).powi(2)
        + (f64::from(tip.z) - pos[2]).powi(2))
    .sqrt();
    // 7 approximate rotations accumulate at most 7·ROT_BOUND of the reach
    assert!(
        err < 7.0 * ROT_BOUND * f64::from(total),
        "FK error {err} m (reach {total})"
    );
    // set_angles clamps to the joint limits
    arm.set_angles(&[10.0; 7]);
    for j in &arm.joints {
        assert!(j.angle <= j.constraint.max_rad + 1e-6 && j.angle >= j.constraint.min_rad - 1e-6);
    }
    let c = JointConstraint::new(-30.0, 60.0);
    assert!((c.clamp(2.0) - 60f32.to_radians()).abs() < 1e-6);
    assert!((c.clamp(-2.0) + 30f32.to_radians()).abs() < 1e-6);
    assert!((c.range() - 90f32.to_radians()).abs() < 1e-6);
}

#[test]
fn ccd_inverse_kinematics_reaches_a_reachable_target_and_converges_with_iterations() {
    // a target that is reachable by construction: the tip of a pose inside
    // every joint limit
    let mut posed = ArmChain::right_arm();
    posed.set_angles(&[0.3, -0.2, 0.4, 0.9, -0.3, 0.2, 0.1]);
    let target = posed.forward_kinematics();
    let reach = posed.total_length();
    let mut prev = f32::MAX;
    for max_iter in [1u32, 4, 16, 64, 256] {
        let mut a = ArmChain::right_arm();
        let (_, err) = a.inverse_kinematics(target, max_iter, 1e-4);
        // more iterations never end further away (damped CCD is a descent
        // method; allow 1 mm of numerical slack)
        assert!(
            err <= prev + 1e-3,
            "iterations {max_iter}: error {err} > previous {prev}"
        );
        prev = err;
    }
    let mut arm = ArmChain::right_arm();
    let (_, err) = arm.inverse_kinematics(target, 256, 1e-4);
    assert!(
        err < 0.01,
        "CCD residual {err} m after 256 iterations (reach {reach})"
    );
    // the reported error is the true FK error
    assert!((arm.forward_kinematics().distance(target) - err).abs() < 1e-6);
    // unreachable target: cannot beat |target| − reach
    let far = Vec3k::new(0.0, -3.0, 0.0);
    let (_, err) = arm.inverse_kinematics(far, 256, 1e-4);
    assert!(err >= 3.0 - reach - 1e-3, "cannot beat |target| − reach");
    assert!(
        err < 3.0 - reach + 0.15,
        "should stretch towards an unreachable target ({err})"
    );
}

// ───────────────────────── predictor / fitter ─────────────────────────────

#[test]
fn predictor_reaches_the_intent_target_independently_of_dt() {
    let target = Vec3k::new(0.4, -0.2, 0.3);
    let intent = Intent::reach(target, 200);
    let t_total = intent.duration_secs();
    assert!((t_total - 0.2).abs() < 1e-6);
    for steps in [4u32, 16, 64, 512] {
        let mut p = Predictor::default();
        p.apply_intent(intent);
        assert!(p.is_active());
        let dt = t_total / steps as f32;
        // half way: exactly the midpoint of the quintic (s(½) = ½)
        for _ in 0..steps / 2 {
            p.update(dt);
        }
        let mid = target.scale(0.5);
        assert!(
            p.position.distance(mid) < 1e-4,
            "steps {steps}: midpoint {:?} vs {mid:?}",
            p.position
        );
        assert!((p.progress() - 0.5).abs() < 1e-4);
        for _ in 0..steps / 2 {
            p.update(dt);
        }
        // precision-parameter independence: the end state is the target, at rest
        // (Σ dt may fall an ulp short of T, so one more step closes the intent)
        assert!(
            p.position.distance(target) < 1e-4,
            "steps {steps}: {:?}",
            p.position
        );
        assert!(
            p.velocity.length() < 1e-3 && p.acceleration.length() < 2e-2,
            "at rest"
        );
        p.update(dt);
        assert!(p.position.distance(target) < 1e-5);
        assert!(!p.is_active() && p.remaining() == 0.0 && p.progress() == 1.0);
    }
    // position_at / velocity_at clamp to [0, T] and agree with the quintic
    let mut p = Predictor::default();
    p.apply_intent(intent);
    assert!(p.position_at(-1.0).length() < 1e-6);
    assert!(p.position_at(10.0).distance(target) < 1e-5);
    let v_peak = 15.0 * target.length() / (8.0 * t_total);
    assert!((p.velocity_at(t_total / 2.0).length() - v_peak).abs() < 1e-3 * v_peak);
}

#[cfg(feature = "encoder")]
fn samples_of(
    start: Vec3k,
    target: Vec3k,
    t_total: f32,
    fraction: f32,
    n: usize,
) -> Vec<alice_kinematics::jerk::MotionSample> {
    use alice_kinematics::jerk::MotionSample;
    (0..n)
        .map(|i| {
            let t = fraction * t_total * i as f32 / (n - 1) as f32;
            let s = s_of(f64::from(t / t_total)) as f32;
            MotionSample {
                pos: start.lerp(target, s),
                time: t,
            }
        })
        .collect()
}

#[cfg(feature = "encoder")]
#[test]
fn jerk_fitter_recovers_target_and_duration_from_the_first_half_of_a_reach() {
    use alice_kinematics::jerk::JerkFitter;
    // oracle: a minimum-jerk reach observed over the first half has covered
    // exactly D/2 (s(½) = ½) — the fitter must return the true target / T
    let (start, target, t_total) = (
        Vec3k::new(0.1, 0.2, 0.0),
        Vec3k::new(0.7, -0.4, 0.3),
        0.6f32,
    );
    for fraction in [0.5f32, 0.3, 0.7] {
        let mut fitter = JerkFitter::default();
        for s in samples_of(start, target, t_total, fraction, 24) {
            fitter.push(s);
        }
        let fit = fitter.fit_trajectory().expect("enough samples");
        let d = start.distance(target);
        assert!(
            fit.target.distance(target) < 0.02 * d,
            "fraction {fraction}: target {:?} vs {target:?}",
            fit.target
        );
        assert!(
            ((fit.duration - t_total) / t_total).abs() < 0.05,
            "fraction {fraction}: duration {} vs {t_total}",
            fit.duration
        );
        assert!(
            fit.fit_error < 0.01 * d,
            "fraction {fraction}: RMS fit error {}",
            fit.fit_error
        );
    }
    // velocity / acceleration estimates are the finite differences of the last samples
    let mut fitter = JerkFitter::default();
    let s = samples_of(start, target, t_total, 0.5, 16);
    for m in &s {
        fitter.push(*m);
    }
    let (a, b) = (s[14], s[15]);
    let v = (b.pos - a.pos).scale(1.0 / (b.time - a.time));
    assert!(fitter.estimate_velocity().distance(v) < 1e-5);
    assert!(fitter.motion_detected(0.01));
}

// ───────────────────────── skeleton / hand ────────────────────────────────

#[test]
fn skeleton_scaling_is_proportional_and_hand_grip_is_monotone() {
    let base = Skeleton::default_humanoid();
    let tall = Skeleton::with_height(1.87);
    let ratio = 1.87 / 1.70;
    for (a, b) in [
        (BoneId::Hips, BoneId::Head),
        (BoneId::LeftShoulder, BoneId::LeftHand),
        (BoneId::RightUpperLeg, BoneId::RightFoot),
    ] {
        let d0 = base.distance_between(a, b).expect("joint pair");
        let d1 = tall.distance_between(a, b).expect("joint pair");
        assert!((d1 / d0 - ratio).abs() < 1e-4, "{a:?}-{b:?}: {d1}/{d0}");
    }
    let mut rescaled = Skeleton::default_humanoid();
    rescaled.rescale(1.87);
    let d = rescaled
        .distance_between(BoneId::Hips, BoneId::Head)
        .unwrap();
    assert!((d - tall.distance_between(BoneId::Hips, BoneId::Head).unwrap()).abs() < 1e-5);

    // hand grip: every fingertip moves closer to its own knuckle (base_offset)
    // as the grip closes — the links curl the same way and the total bend at
    // grip 1.0 stays below a half turn on the base–tip chord
    let mut hand = alice_kinematics::hand::HandModel::right_hand();
    let chord = |h: &alice_kinematics::hand::HandModel| -> Vec<f32> {
        h.fingers
            .iter()
            .map(|f| f.tip_position().distance(f.base_offset))
            .collect()
    };
    let open = chord(&hand);
    for (i, f) in hand.fingers.iter().enumerate() {
        assert!(
            (open[i] - f.total_length()).abs() < 1e-5,
            "finger {i} open = straight"
        );
    }
    let mut prev = open;
    for k in 1..=4 {
        hand.grip(k as f32 * 0.25);
        let now = chord(&hand);
        for i in 0..5 {
            assert!(
                now[i] < prev[i] - 1e-6,
                "finger {i} at grip {}: chord {} ≥ {}",
                k as f32 * 0.25,
                now[i],
                prev[i]
            );
        }
        prev = now;
    }
}
