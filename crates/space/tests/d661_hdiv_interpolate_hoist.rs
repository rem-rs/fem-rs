//! D661 — bitwise pin + perf evidence for the hoisted `HDivSpace::interpolate_vector`
//! engine (rows / reference element / dual matrix hoisted out of the per-element
//! loop).
//!
//! The `*_bits` constants are the exact `f64::to_bits` hex of two mixing sums
//! over the dof vector, captured on the *pre-refactor* code (round 64) —
//! `assert_eq!` on the bit patterns is a bitwise identity check.

use fem_mesh::Mesh;
use fem_space::HDivSpace;

fn f2(x: &[f64]) -> Vec<f64> {
    vec![(3.0 * x[0]).sin() + 0.5 * x[1], x[1] - (2.0 * x[0]).cos()]
}

fn f3(x: &[f64]) -> Vec<f64> {
    vec![
        (3.0 * x[0]).sin() + 0.5 * x[1],
        x[1] * x[2] - (2.0 * x[0]).cos(),
        (x[0] + x[2]).sin(),
    ]
}

/// Two exact-bit mixing sums over the dof vector.
fn bits_of(v: &[f64]) -> (String, String) {
    let mut s = 0.0_f64;
    let mut m = 0.0_f64;
    for &x in v {
        s += x;
        m += x * 7.31 - (x * 1.7).cos();
    }
    (format!("{:016x}", s.to_bits()), format!("{:016x}", m.to_bits()))
}

#[test]
fn d661_tri_rt1_bitwise_pin() {
    let mesh = Mesh::<2>::unit_square_tri(6);
    let sp = HDivSpace::new(mesh, 1);
    let v = sp.interpolate_vector(&f2);
    let (s, m) = bits_of(v.as_slice());
    let (want_s, want_m) = (TRI_S, TRI_M);
    assert_eq!(s, want_s, "tri RT1 sum bits");
    assert_eq!(m, want_m, "tri RT1 mix bits");
}

#[test]
fn d661_tet_rt1_bitwise_pin() {
    let mesh = Mesh::<3>::unit_cube_tet(3);
    let sp = HDivSpace::new(mesh, 1);
    let v = sp.interpolate_vector(&f3);
    let (s, m) = bits_of(v.as_slice());
    let (want_s, want_m) = (TET_S, TET_M);
    assert_eq!(s, want_s, "tet RT1 sum bits");
    assert_eq!(m, want_m, "tet RT1 mix bits");
}

#[test]
fn d661_hex_rt0_bitwise_pin() {
    let mesh = Mesh::<3>::unit_cube_hex(3);
    let sp = HDivSpace::new(mesh, 0);
    let v = sp.interpolate_vector(&f3);
    let (s, m) = bits_of(v.as_slice());
    let (want_s, want_m) = (HEX_S, HEX_M);
    assert_eq!(s, want_s, "hex RT0 sum bits");
    assert_eq!(m, want_m, "hex RT0 mix bits");
}

/// Perf evidence (ignored by default): tet RT2 — the largest per-element cost
/// (moment-dual `TetRTk::new(2)` + 20x20 dual matrix + solve, all rebuilt per
/// element before the hoist).  Run with
/// `cargo test -p fem-space --release --test d661_hdiv_interpolate_hoist -- --ignored --nocapture`
#[test]
#[ignore]
fn d661_perf_tet_rt2() {
    let mesh = Mesh::<3>::unit_cube_tet(8);
    let sp = HDivSpace::new(mesh, 2);
    let t0 = std::time::Instant::now();
    let v = sp.interpolate_vector(&f3);
    let dt = t0.elapsed();
    let (s, _) = bits_of(v.as_slice());
    println!("d661 perf: interpolate_vector tet RT2 (3072 elems) took {dt:?} (sum bits {s})");
}

const TRI_S: &str = "4030602b50ed1ae9";
const TRI_M: &str = "c06f6c1ca87347e8";
const TET_S: &str = "4041c96c3353840b";
const TET_M: &str = "c094ea1b1c3f0663";
const HEX_S: &str = "401240d4b2b33247";
const HEX_M: &str = "c052663f86a36aa0";
