//! d118 — coverage_matrix §4-3a: **ND triangle k ≥ 2 verification depth**.
//!
//! Round-102 proved `TriNDk` is a functional 1:1 port of MFEM
//! `ND_TriangleElement(p)` (documented construction, same functionals); the
//! matrix §1.2 ND-tri row "MACH（1 阶逐位）" only recorded that the *numerical*
//! cross-check had been run at order 1.  This pin closes that verification
//! gap at k = 2, 3 against the MFEM 4.10 oracle:
//!
//! Truth source (probed 2026-10-06, WSL `$HOME/mfem410_ser`,
//! `tmp/d118s43/probe_ndtri.cpp` → `d118_nd_tri_k23_ref.txt` next to this
//! file): `ND_TriangleElement::CalcVShape` / `CalcCurlShape` (fe_nd.cpp:1189 /
//! :1223) on a 15×15 half-lattice (x = i/14, y = j/14, x+y ≤ 1 — includes the
//! three vertices, the edge midpoints and the x+y = 1 edge) plus 12
//! deterministic interior pseudo-random points, all numbers %.17g, plus the
//! constructor DOF sites `FE::Nodes` (fe_nd.cpp:1122-1140).
//!
//! `ND_TriangleElement` has **no CalcDShape** — the scalar `CalcDShape` of
//! `VectorFiniteElement` is a `mfem_error` abort (fe_base.cpp:1035) — so the
//! derivative surface is `CalcCurlShape`, exactly what
//! [`VectorReferenceElement::eval_curl`] exposes for [`TriNDk`].
//!
//! # Expected deviation (inversion framework, not a basis defect)
//!
//! MFEM inverts the Vandermonde with `Ti.Factor(T)` (`DenseMatrixInverse` =
//! LUFactors, LU with partial pivoting); fem-rs `tri_ndk.rs::invert_dense` is
//! a Gauss-Jordan sweep.  The probe re-evaluates shape/curl through a verbatim
//! C++ port of the fem-rs Gauss-Jordan on the same Vandermonde and prints the
//! summary line `GJDIFF p=.. shape=.. curl=..` — the C++-internal
//! GJ-vs-LU deviation.  The pin asserts fem-rs' deviation from MFEM equals
//! that framework bound: the entire difference is the inversion algorithm,
//! none of it the Chebyshev u-basis, the Vandermonde assembly, the DOF
//! layout or the Ti application (all bit-tracked through the same
//! operation order in both implementations).

use fem_element::embedded::NdR2dTri;
use fem_element::{TriNDk, VectorReferenceElement};

const REF: &str = include_str!("d118_nd_tri_k23_ref.txt");

/// Monotone bit key for ulp distance (total order over f64).  Meaningful for
/// same-sign pairs only (used for the DOF sites); for near-zero shape/curl
/// entries the absolute and relative deviations below are the honest metric.
fn f64_key(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 { i64::MIN.wrapping_sub(b) } else { b }
}

fn ulp_dist(a: f64, b: f64) -> u64 {
    f64_key(a).wrapping_sub(f64_key(b)).unsigned_abs()
}

struct OrderDump {
    p: usize,
    n: usize,
    /// MFEM `FE::Nodes` (constructor DOF sites).
    dofs: Vec<(f64, f64)>,
    /// `(x, y, 2N VShape components, N CurlShape components)` per sample.
    pts: Vec<([f64; 2], Vec<f64>, Vec<f64>)>,
    /// C++-internal fem-rs-Gauss-Jordan-vs-MFEM-LU worst deviation.
    gj_shape: f64,
    gj_curl: f64,
}

fn parse_ref() -> Vec<OrderDump> {
    let mut out: Vec<OrderDump> = Vec::new();
    let mut cur: Option<OrderDump> = None;
    let mut pending_pt: Option<([f64; 2], Vec<f64>)> = None;
    for line in REF.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "NDTRI" => {
                if let Some(c) = cur.take() {
                    out.push(c);
                }
                cur = Some(OrderDump {
                    p: f[1][2..].parse().unwrap(),
                    n: f[2][2..].parse().unwrap(),
                    dofs: Vec::new(),
                    pts: Vec::new(),
                    gj_shape: f64::NAN,
                    gj_curl: f64::NAN,
                });
            }
            "DOF" => {
                cur.as_mut().unwrap()
                    .dofs
                    .push((f[2].parse().unwrap(), f[3].parse().unwrap()));
            }
            "PT" => {
                pending_pt = Some((
                    [f[1].parse().unwrap(), f[2].parse().unwrap()],
                    f[3..].iter().map(|v| v.parse().unwrap()).collect(),
                ));
            }
            "CR" => {
                let (xy, shape) = pending_pt.take().unwrap();
                let curl: Vec<f64> = f[3..].iter().map(|v| v.parse().unwrap()).collect();
                cur.as_mut().unwrap().pts.push((xy, shape, curl));
            }
            "GJDIFF" => {
                let mut c = cur.take().unwrap();
                c.gj_shape = f[2][6..].parse().unwrap();
                c.gj_curl = f[3][5..].parse().unwrap();
                out.push(c);
            }
            _ => {}
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    out
}

#[test]
fn d118_nd_tri_k23_matches_mfem_truth() {
    for d in parse_ref() {
        let elem = TriNDk::new(d.p);
        let n = elem.n_dofs();
        assert_eq!(d.p, elem.order() as usize, "order");
        assert_eq!(n, d.n, "p={}: dof count vs MFEM", d.p);

        // DOF sites: fem-rs gauss_legendre_01 constants vs MFEM Newton
        // Poly_1D points.  Recorded bit-level; must agree to a hair.
        let coords = elem.dof_coords();
        let mut dof_ulps = 0_u64;
        let mut dof_bit_mismatch = 0_usize;
        for (i, (rx, ry)) in d.dofs.iter().enumerate() {
            let (gx, gy) = (coords[i][0], coords[i][1]);
            dof_ulps = dof_ulps.max(ulp_dist(gx, *rx)).max(ulp_dist(gy, *ry));
            if gx.to_bits() != rx.to_bits() || gy.to_bits() != ry.to_bits() {
                dof_bit_mismatch += 1;
            }
        }

        // Shapes and curls on the dense lattice.
        let mut vals = vec![0.0_f64; n * 2];
        let mut curls = vec![0.0_f64; n];
        let (mut w_shape, mut w_curl) = (0.0_f64, 0.0_f64);
        let (mut m_shape, mut m_curl) = (0.0_f64, 0.0_f64);
        for (xi, shape, curl) in &d.pts {
            elem.eval_basis_vec(xi, &mut vals);
            elem.eval_curl(xi, &mut curls);
            for k in 0..n {
                for c in 0..2 {
                    let r = shape[k * 2 + c];
                    m_shape = m_shape.max(r.abs());
                    w_shape = w_shape.max((vals[k * 2 + c] - r).abs());
                }
                m_curl = m_curl.max(curl[k].abs());
                w_curl = w_curl.max((curls[k] - curl[k]).abs());
            }
        }
        let rel_shape = w_shape / m_shape;
        let rel_curl = w_curl / m_curl;
        println!(
            "p={}: {} points, shape worst |Δ| = {w_shape:.3e} (rel {rel_shape:.3e}), \
             curl worst |Δ| = {w_curl:.3e} (rel {rel_curl:.3e}); \
             GJDIFF(shape={:.3e}, curl={:.3e}); \
             dof sites worst {dof_ulps} ulp ({dof_bit_mismatch} bit-mismatched of {n})",
            d.p,
            d.pts.len(),
            d.gj_shape,
            d.gj_curl,
        );

        // The whole fem-rs-vs-MFEM deviation must be the inversion-framework
        // bound measured inside C++ (Gauss-Jordan vs LU on the same
        // Vandermonde): identical worst values, bit for bit.
        assert_eq!(w_shape, d.gj_shape, "p={}: shape deviation != GJ framework bound", d.p);
        assert_eq!(w_curl, d.gj_curl, "p={}: curl deviation != GJ framework bound", d.p);
        // Absolute and relative honesty guards (far above the framework bound).
        assert!(w_shape < 1e-13 && w_curl < 1e-13, "p={}", d.p);
        assert!(rel_shape < 1e-14 && rel_curl < 1e-14, "p={}", d.p);
        // DOF sites agree within a few ulp of the MFEM Newton points.
        assert!(dof_ulps <= 4, "p={}: dof sites drift {dof_ulps} ulp", d.p);

        // ── embedded arm delegation: NdR2dTri in-plane slots ────────────
        // Its in-plane engine *is* the TriNDk above (nd_r2d.rs
        // `NdTriEngine::Pk`); the embedded dof table is the ND dof sequence
        // (3p edge + interior x/y pairs) so the i-th in-plane slot maps to
        // ND dof i.  Pin that wiring against the same MFEM numbers.
        let nd = NdR2dTri::new(d.p);
        let tk = nd.dof2tk();
        let n_inplane = tk.iter().filter(|&&t| t != 4).count();
        assert_eq!(n_inplane, n, "p={}: embedded in-plane slot count", d.p);
        let mut vshape = vec![0.0_f64; nd.n_dofs() * 3];
        let mut e_worst = 0.0_f64;
        for (xi, shape, _curl) in &d.pts {
            nd.eval_vshape_ref(xi, &mut vshape);
            let mut i = 0_usize;
            for (k, &t) in tk.iter().enumerate() {
                if t == 4 {
                    continue;
                }
                e_worst = e_worst.max((vshape[k * 3] - shape[i * 2]).abs());
                e_worst = e_worst.max((vshape[k * 3 + 1] - shape[i * 2 + 1]).abs());
                i += 1;
            }
            assert_eq!(i, n);
        }
        assert_eq!(e_worst, w_shape, "p={}: embedded in-plane == TriNDk deviation", d.p);
    }
}
