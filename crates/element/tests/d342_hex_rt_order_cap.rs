//! D342: the hex RT element above the old `order <= 2` cap.
//!
//! `HDivSpace` used to reject hex RT orders ≥ 3 (`validate_order` asserted
//! "Hex RT supports orders 0, 1, and 2"), so `fem-rs` could not run MFEM ex24
//! `-o 4` (RT3) at all.  MFEM 4.10 has **no** order bound on the hex RT element
//! (`RT_FECollection`'s ctor only checks `p >= 0`, `fem/fe_coll.cpp:2531`, and
//! `RT_HexahedronElement` (`fem/fe/fe_rt.cpp:326`) — face frames, interior
//! enumeration and the `i <= p/2` orientation flips — is a plain `p`-loop), so
//! the cap was raised to the same house limit as the 2-D quad arm: `0..=6`.
//!
//! The capability this file pins is the property the hex `Project_RT`
//! interpolation solve in `HDivSpace::interpolate_vector` depends on: the
//! element's nodal GaussLegendre basis must be **exactly dual** to MFEM's
//! `Project_RT` sample functionals (`fem/fe/fe_rt.cpp:Project_RT` with
//! `dof2nk`), i.e.
//!
//! ```text
//!     W[i][j] = phi_j(x_i) · nk_i = (1/4)·delta_ij   for every order k
//! ```
//!
//! (the `1/4` is the `[-1,1]` pull-back of two halved open factors; the unit
//! cube's `cof(J) = (1/4)I` cancels it in the physical dof).  A *global* factor
//! error in the basis cannot hide here — the diagonal is checked against `1/4`
//! itself, not against the basis — and the slot ordering, the outward-normal
//! signs and the D225 interior flips are all exercised because any mismatch
//! shows up as an off-diagonal entry.
//!
//! The order-3 values are additionally pinned slot-for-slot against an MFEM
//! 4.10 dump in `fem-element`'s `hex_rtk::tests::rt_gl_matches_mfem_nodal_dump`.

#![allow(clippy::needless_range_loop)] // index arithmetic mirrors the C++ dump layout

use fem_element::quadrature::{gauss_lobatto_arbitrary, gauss_legendre_arbitrary};
use fem_element::raviart_thomas::{free_axes, HexRTk, HEX_RT_FACES};
use fem_element::reference::VectorReferenceElement;

/// MFEM's `RT_HexahedronElement` `Project_RT` sample rows: one pointwise flux
/// sample `phi(x_i)·nk_i` per element dof slot, in the element's own slot
/// order (`HEX_RT_FACES` face frames, then the x/y/z interior blocks with the
/// `i <= p/2` reference-orientation flips).  Mirrors `fem_space::hdiv`'s
/// `interp_rows` hex arm — deliberately re-derived here from the MFEM rule so
/// the element is checked against the specification, not against the space.
fn project_rt_rows(k: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let m = k + 1;
    let mut xs = Vec::new();
    let mut nks = Vec::new();
    let gl = gauss_legendre_arbitrary(m).0;
    for &(nc, at_max, _s, f1, f2) in &HEX_RT_FACES {
        let cnorm = if at_max { 1.0 } else { -1.0 };
        let (a1, a2) = free_axes(nc);
        let mut nk = [0.0_f64; 3];
        nk[nc] = cnorm;
        for j in 0..m {
            let q = if f2 { m - 1 - j } else { j };
            for i in 0..m {
                let p = if f1 { m - 1 - i } else { i };
                let mut xi = [0.0_f64; 3];
                xi[nc] = cnorm;
                xi[a1] = gl[p];
                xi[a2] = gl[q];
                xs.push(xi);
                nks.push(nk);
            }
        }
    }
    if k >= 1 {
        let cp = gauss_lobatto_arbitrary(k + 2).0;
        let push = |s: f64, axis: usize, xi: [f64; 3], xs: &mut Vec<[f64; 3]>, nks: &mut Vec<[f64; 3]>| {
            let mut nk = [0.0_f64; 3];
            nk[axis] = s;
            xs.push(xi);
            nks.push(nk);
        };
        for l in 0..m {
            for j in 0..m {
                for i in 1..=k {
                    let s = if i <= k / 2 { -1.0 } else { 1.0 };
                    push(s, 0, [cp[i], gl[j], gl[l]], &mut xs, &mut nks);
                }
            }
        }
        for l in 0..m {
            for j in 1..=k {
                let s = if j <= k / 2 { -1.0 } else { 1.0 };
                for i in 0..m {
                    push(s, 1, [gl[i], cp[j], gl[l]], &mut xs, &mut nks);
                }
            }
        }
        for l in 1..=k {
            let s = if l <= k / 2 { -1.0 } else { 1.0 };
            for j in 0..m {
                for i in 0..m {
                    push(s, 2, [gl[i], gl[j], cp[l]], &mut xs, &mut nks);
                }
            }
        }
    }
    (xs, nks)
}

/// D342: the element must own the exact dual of MFEM's `Project_RT` samples for
/// every order the raised cap allows (0..=6), i.e. `W = (1/4)·I` on the
/// reference hex.  This is what makes `HDivSpace::interpolate_vector`'s dense
/// solve return MFEM's dof values themselves.
#[test]
fn d342_hex_rt_project_rt_dual_matrix_is_quarter_identity() {
    for k in 0..=6usize {
        let e = HexRTk::new_gauss_legendre(k);
        let (xs, nks) = project_rt_rows(k);
        let n = e.n_dofs();
        assert_eq!(xs.len(), n, "k={k}: row count vs n_dofs");
        let mut phi = vec![0.0_f64; n * 3];
        let mut worst_diag = 0.0_f64;
        let mut worst_off = 0.0_f64;
        for i in 0..n {
            e.eval_basis_vec(&xs[i], &mut phi);
            for j in 0..n {
                let w = (0..3).map(|d| phi[j * 3 + d] * nks[i][d]).sum::<f64>();
                if i == j {
                    worst_diag = worst_diag.max((w - 0.25).abs());
                } else {
                    worst_off = worst_off.max(w.abs());
                }
            }
        }
        println!("  RT{k}: |W_ii - 1/4| <= {worst_diag:.3e}, |W_ij| <= {worst_off:.3e}");
        assert!(worst_diag < 1e-13, "k={k}: dual diagonal deviates from 1/4 by {worst_diag}");
        assert!(worst_off < 1e-12, "k={k}: dual matrix is not diagonal ({worst_off})");
    }
}

/// D342: the dof bookkeeping (`3(p+1)^2(p+2)` slots, MFEM's interior ordering)
/// and the dof coordinates must be right at the newly allowed orders — the
/// space's `interp_rows` samples the interior closed directions at the
/// **GaussLobatto** interior nodes and the open directions at the
/// **GaussLegendre** points, so a mixed-up node set shows up as a
/// non-diagonal dual matrix above.  Here the coordinates are pinned directly:
/// every interior closed coordinate must be a GLL node of the `k+2`-point
/// family and no node may sit at ±1 in an open direction.
#[test]
fn d342_hex_rt_dof_coords_and_counts_orders_3_to_6() {
    for k in 3..=6usize {
        let e = HexRTk::new_gauss_legendre(k);
        let m = k + 1;
        assert_eq!(e.order() as usize, k);
        assert_eq!(e.n_dofs(), 3 * m * m * (k + 2));
        let c = e.dof_coords();
        assert_eq!(c.len(), e.n_dofs());
        // Face slots: 6 faces x (k+1)^2, exactly one coordinate at +-1.
        for (i, x) in c.iter().enumerate().take(6 * m * m) {
            let at_face = x.iter().filter(|v| v.abs() == 1.0).count();
            assert_eq!(at_face, 1, "k={k} face slot {i}: {x:?}");
        }
        // Interior slots: no coordinate at +-1, and at least one coordinate an
        // interior closed (GLL) node of the k+2-point family (the closed-index
        // direction of the block).
        let glc = gauss_lobatto_arbitrary(k + 2).0;
        let interior = &c[6 * m * m..];
        assert_eq!(interior.len(), 3 * k * m * m);
        for (i, x) in interior.iter().enumerate() {
            assert!(x.iter().all(|v| v.abs() < 1.0), "k={k} interior slot {i}: {x:?}");
            let on_gll = x
                .iter()
                .filter(|v| glc.iter().any(|g| (g - **v).abs() < 1e-14))
                .count();
            assert_eq!(on_gll, 1, "k={k} interior slot {i}: {x:?}");
        }
    }
}
