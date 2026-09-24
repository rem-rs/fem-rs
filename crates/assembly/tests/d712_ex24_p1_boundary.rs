//! D712/D714 boundary pins: why ex24 `-p 1` (beam-hex) still prints
//! `Iteration 1 (B r, r) = 1.47754e-22` / `ARF = 9.13443e-13` where MFEM 4.10
//! prints `1.47776e-22` / `9.13511e-13` — every other line of the run is
//! byte-identical (round 67).
//!
//! # The adjudication (round 68, evidence in `tmp/d712/`)
//!
//! A four-way splice matrix on WSL MFEM 4.10 (CG + DSmoother fed with the
//! triplets `M/C` and vector `v` of each side) proves the printed pair is
//! decided **entirely by M/C entry-level round-off**, not by v:
//!
//! | M, C | v   | iter 1        | ARF           |
//! |------|-----|---------------|---------------|
//! | rs   | rs  | 1.47754e-22   | 9.13444e-13   | (= Rust output)
//! | cpp  | rs  | 1.47776e-22   | 9.13511e-13   | (= MFEM output)
//! | rs   | cpp | 1.47754e-22   | 9.13443e-13   |
//! | cpp  | cpp | 1.47776e-22   | 9.13511e-13   |
//!
//! - v (H(curl) interpolant of `(sin πy, sin πz, sin πx)`) differs on
//!   11552/107168 dofs (1 ulp each): glibc vs UCRT `sin` at the edge-midpoint
//!   arguments (D714 platform fact — e.g. `sin(0.7853981633974483)` =
//!   `0.7071067811865475` on glibc, `0.7071067811865476` on UCRT).  The splice
//!   rows show this noise is **not load-bearing** for the printed lines.
//! - M/C differ bitwise on ~97% of entries (structural entries at the
//!   10–50 ulp level, `|Δ| ≤ 9e-14`) because the two codes integrate on
//!   **different reference conventions**: fem-rs' hex RT/ND elements live on
//!   `[-1,1]³` (quadrature points `±√0.6`, weights `(5/9)³…`, Jacobian
//!   `J = 0.5·I + junk`) while MFEM's live on `[0,1]³` (points
//!   `(1±√0.6)/2`, weights `(5/18)³…`, `J = I + 5.2e-18 junk`).  The
//!   mathematics agrees to 1e-11 everywhere (D680), but every intermediate
//!   double differs, and the `-p 1` first-iteration residual is pure
//!   cancellation debris 24 orders below the initial residual — it amplifies
//!   the entry-level noise into the printed 6th digit.
//!
//! # What closing the last 2 lines would take (registered as new debt)
//!
//! 1. `[0,1]³` reference-convention alignment for the hex RT/ND path:
//!    quadrature + basis evaluation (`fem-element`), hex geometry element
//!    (`fem-mesh`), dof/sign/prolongation semantics (`fem-space`) — the D680
//!    default-order table, `-p 0`/`-p 2` byte gauges and all d6xx pins then
//!    need a deliberate re-baseline.
//! 2. Solver-layer parity: MFEM's CG runs its SpMV over column-sorted CSR and
//!    folds the ARF from the error history in its own op order — even with
//!    bitwise M/C the splice shows a last-digit ARF difference
//!    (`9.13444e-13` via triplets vs Rust's `9.13443e-13`).
//!
//! Until then the two lines stay documented as a boundary, not forced.

use fem_element::{vec_ref_elem, VecFamily};
use fem_io::mfem::read_mfem_file;
use fem_space::{HCurlSpace, HDivSpace};

const UNIT_HEX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/d680_unit_hex_mfem.mesh");
const FIXTURES: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/d680_vector_mass_mfem.txt"));

/// **D721 FLIP TARGET — flipped.**  The ex24 `-p 1` RT/ND hex rule *was* the
/// 27-point Gauss–Legendre tensor on the fem-rs reference cube `[-1,1]³`
/// (`±√0.6`, weights `(5/9)³`); it is now MFEM's `[0,1]³` rule — points
/// `(1±√0.6)/2` / `0.5`, weights `(5/18)³` (with MFEM's 1-ulp face split
/// `…af8`/`…af9`) and x-fastest enumeration — so every intermediate double of
/// the M/C assembly is MFEM's.  This test is the deliberate fingerprint of
/// that flip (it asserted the `[-1,1]` values before D721).
#[test]
fn d712_hex_vector_quadrature_is_unit_cube_gauss() {
    let rt0 = vec_ref_elem(VecFamily::RaviartThomas, fem_mesh::ElementType::Hex8.to_elem_type(), 0);
    let quad = rt0.quadrature(4);
    assert_eq!(quad.points.len(), 27);

    let s = 0.6_f64.sqrt();
    let x0 = (1.0 - s) / 2.0; // outer [0,1] abscissa
    let w5 = 5.0_f64 / 18.0;
    let w8 = 8.0_f64 / 18.0;

    // Corner point and weight, bitwise (MFEM's IntRules CUBE order-4 dump).
    assert_eq!(quad.points[0][0], 0.11270166537925831174);
    assert_eq!(quad.points[0][1], 0.11270166537925831174);
    assert_eq!(quad.points[0][2], 0.11270166537925831174);
    assert_eq!(quad.weights[0].to_bits(), 0x3f95f2a7db7a8e94);
    // Face-centre point and weight.
    assert_eq!(quad.points[13][0], 0.5);
    assert_eq!(quad.points[13][1], 0.5);
    assert_eq!(quad.points[13][2], 0.5);
    assert_eq!(quad.weights[13].to_bits(), 0x3fb67980e0bf08c7);
    // The outer abscissa is the [0,1] table value, NOT `(1−√0.6)/2` by hand
    // (the two differ by 1 ulp — the D339 mapping fact).
    assert_ne!(quad.points[0][0], x0);
    assert_eq!(quad.points[0][0], 0.11270166537925831174);
    // The 1-ulp face-weight split MFEM's dump carries.
    assert_ne!(quad.weights[4].to_bits(), quad.weights[12].to_bits());

    // Total weight = reference volume 1 ([0,1]³ convention) — the physical
    // unit-hex volume with no affine-map factor.
    let total: f64 = quad.weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "total weight {total}");
}

/// Mathematical parity anchor (D680 fixtures, MFEM 4.10 truth): the unit-hex
/// RT0/ND1 mass agrees to 1e-11 — the divergence documented above is round-off
/// class, not formulation.
#[test]
fn d712_unit_hex_vector_mass_still_matches_mfem_truth() {
    let mfem_mesh = read_mfem_file(UNIT_HEX).expect("read unit hex");
    let mesh3 = mfem_mesh.mesh3d.expect("3d");

    // Parse the D680 fixture blocks (name rows cols + rows of values).
    let mut fixtures = std::collections::HashMap::new();
    let mut lines = FIXTURES.lines().peekable();
    while let Some(line) = lines.next() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() == 3 && tokens[1].parse::<usize>().is_ok() && !tokens[0].ends_with("RULE") {
            let rows: usize = tokens[1].parse().unwrap();
            let mut mat = Vec::with_capacity(rows);
            for _ in 0..rows {
                let vals: Vec<f64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|t| t.parse().unwrap())
                    .collect();
                mat.push(vals);
            }
            fixtures.insert(tokens[0].to_string(), mat);
        }
    }

    for (name, space, is_hdiv) in [
        ("RT0_HexAFF", 0u8, true),
        ("ND1_HexAFF", 1u8, false),
    ] {
        // Global = element on the single-hex mesh; unmangle interface signs.
        let csr;
        let signs: Vec<f64>;
        let dofs: Vec<usize>;
        let n: usize;
        if is_hdiv {
            let sp = HDivSpace::new(mesh3.clone(), space);
            csr = fem_assembly::VectorAssembler::assemble_bilinear(
                &sp,
                &[&fem_assembly::standard::VectorMassIntegrator { alpha: 1.0 }],
                1,
            );
            signs = sp.element_signs(0).to_vec();
            dofs = sp.element_dofs(0).iter().map(|&d| d as usize).collect();
            n = sp.n_dofs();
        } else {
            let sp = HCurlSpace::new(mesh3.clone(), space);
            csr = fem_assembly::VectorAssembler::assemble_bilinear(
                &sp,
                &[&fem_assembly::standard::VectorMassIntegrator { alpha: 1.0 }],
                1,
            );
            signs = sp.element_signs(0).to_vec();
            dofs = sp.element_dofs(0).iter().map(|&d| d as usize).collect();
            n = sp.n_dofs();
        }
        let _ = n;
        let want = &fixtures[name];
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let got = signs[i] * signs[j] * csr.get(gi, gj);
                assert!(
                    (got - want[i][j]).abs() <= 1e-11,
                    "{name} ({i},{j}): {got} vs MFEM {}",
                    want[i][j]
                );
            }
        }
    }
}

/// D714 platform pin (Windows/UCRT only): at the D714 argument the local libm
/// returns the value 1 ulp above glibc's.  Documents the cross-libm noise
/// source in the `-p 1` RHS; fails loudly if the toolchain libm changes.
#[cfg(all(target_os = "windows", target_env = "msvc"))]
#[test]
fn d714_windows_ucrt_sin_one_ulp_off_glibc_at_pi_over_4() {
    use std::f64::consts::PI;
    let got = (PI * 0.25_f64).sin();
    // UCRT (this platform):
    assert_eq!(got, 0.7071067811865476_f64);
    // glibc (the MFEM reference platform) returns 0.7071067811865475 —
    // exactly 1 ulp below:
    assert_eq!(f64::from_bits(got.to_bits() - 1), 0.7071067811865475_f64);
}
