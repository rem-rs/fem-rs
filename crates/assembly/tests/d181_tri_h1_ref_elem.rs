//! D181 — the tri arms of `mixed::ref_elem_vol` pair with the H¹ tri space.
//!
//! Round 36 closed D157 by moving the tet H¹ **field** reference element off
//! the equispaced `factory::TetPk` onto MFEM's `H1_TetrahedronElement`
//! ([`fem_element::lagrange::H1TetPk`]).  D181 is the same-family fix for
//! triangles: the tri H¹ space numbering is the GLL `H1TriPk` table
//! (`fem_space::DofManager::build_pk` + `assembler::ref_elem_vol_h1`), but
//! `mixed::ref_elem_vol`'s tri order-3 arm still built the equispaced
//! `factory::TriPk`, whose slots are *different functions* from p = 3 on.
//!
//! Facts pinned here (all measured in `tmp/d181_tri_evidence.md`):
//! - the two tri lattices coincide bit-for-bit at p = 1 and p = 2 (coords);
//!   at p = 3, 6/10 slots disagree (worst position delta 3.90e-1), at p = 4,
//!   9/15 (worst 5.77e-1);
//! - before the fix, an L2 mass projection through
//!   `MixedAssembler::assemble_bilinear` (whose col space goes through
//!   `mixed::ref_elem_vol`) produced garbage at p = 3 — errors
//!   9.85e-1 → 2.13e1 → 1.93e0 on n = 2, 4, 8, i.e. no convergence at all;
//! - after the fix the projection converges at O(h^{p+1}).

use std::f64::consts::PI;

use fem_assembly::{Assembler, MixedAssembler, mixed::{ref_elem_vol, ScalarMassIntegrator},
                   standard::DomainSourceIntegrator};
use fem_element::{ReferenceElement, lagrange::{factory::TriPk, H1TriPk}};
use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, element_type::ElementType, topology::MeshTopology};
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};
use nalgebra::{DMatrix, DVector};

/// Bit-for-bit coordinate equality of two elements' dof tables.
fn coords_bit_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter())
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| u.to_bits() == v.to_bits()))
}

/// D181 premise: at p ≤ 2 the equispaced and GLL tri lattices are the same
/// points (so the untouched `TriPk::new(2)` arms keep old results bitwise),
/// while from p = 3 on they are different lattices entirely.
#[test]
fn d181_tri_families_coincide_bitwise_at_p_le_2_only() {
    for p in [1usize, 2usize] {
        let equi = TriPk::new(p);
        let gll = H1TriPk::new(p);
        assert!(
            coords_bit_eq(&equi.dof_coords(), &gll.dof_coords()),
            "p={p}: equispaced TriPk and H1TriPk dof coords must be bit-identical"
        );
    }
    // p = 3: the lattices differ (worst slot delta 3.90e-1) — the arm must
    // therefore be off the equispaced element from here on.
    let equi3 = TriPk::new(3).dof_coords();
    let gll3 = H1TriPk::new(3).dof_coords();
    let n_mismatch = equi3.iter().zip(gll3.iter())
        .filter(|(a, b)| ((a[0] - b[0]).abs()) > 1e-12 || ((a[1] - b[1]).abs()) > 1e-12)
        .count();
    assert_eq!(n_mismatch, 6, "p=3: expected 6/10 slots to differ");
}

/// The p ≤ 2 arms are unchanged (bit-identical results for existing callers):
/// they still return the equispaced element's tables.
#[test]
fn d181_mixed_p_le_2_arms_unchanged() {
    for p in [1u8, 2u8] {
        let mixed_e = ref_elem_vol(ElementType::Tri3, p).unwrap();
        assert!(
            coords_bit_eq(&mixed_e.dof_coords(), &TriPk::new(p as usize).dof_coords()),
            "p={p}: mixed::ref_elem_vol tri arm must stay bit-identical to TriPk"
        );
        let mixed_e6 = ref_elem_vol(ElementType::Tri6, p).unwrap();
        assert!(coords_bit_eq(&mixed_e6.dof_coords(), &TriPk::new(p as usize).dof_coords()));
    }
}

/// D181 fix: every tri order of `mixed::ref_elem_vol` now returns exactly the
/// H¹ space's reference element (`H1TriPk`, MFEM `H1_TriangleElement`), and
/// the basis is nodal at those slots — i.e. slot k of the assembled matrix
/// row/col is the space's dof k.
#[test]
fn d181_mixed_ref_elem_vol_tri_is_h1tripk_at_every_order() {
    for p in 1..=10usize {
        for et in [ElementType::Tri3, ElementType::Tri6] {
            let e = ref_elem_vol(et, p as u8).unwrap();
            let g = H1TriPk::new(p);
            assert!(
                coords_bit_eq(&e.dof_coords(), &g.dof_coords()),
                "({et:?}, p={p}): mixed::ref_elem_vol tri dof coords must equal H1TriPk's"
            );
        }
        // Nodal property on the returned element (Tri3 arm): phi_k(x_k) = 1.
        // Tolerance: `H1TriPk` builds its basis from the monomial Vandermonde
        // inverse, whose conditioning degrades with order (measured residuals:
        // ~1.4e-10 at p = 8, ~1.7e-6 at p = 10) — a conditioning fact of the
        // element, not a slot mismatch (the coord equality above is
        // bit-for-bit at every order).
        let e = ref_elem_vol(ElementType::Tri3, p as u8).unwrap();
        let coords = e.dof_coords();
        let n = coords.len();
        let tol = if p <= 6 { 1e-10 } else if p <= 8 { 1e-8 } else { 1e-3 };
        let mut phi = vec![0.0f64; n];
        for (k, xk) in coords.iter().enumerate() {
            e.eval_basis(xk, &mut phi);
            for (j, v) in phi.iter().enumerate() {
                let target = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (v - target).abs() < tol,
                    "p={p}: nodal property failed at dof {k} slot {j}: {v}"
                );
            }
        }
    }
}

// ── MMS-style criterion: L2 mass projection through the mixed col space ────

fn dense_solve(mat: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = mat.nrows;
    let a = DMatrix::from_row_slice(n, n, &mat.to_dense());
    let b = DVector::from_column_slice(rhs);
    a.lu().solve(&b).unwrap().as_slice().to_vec()
}

fn f_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

/// L2 error of the discrete field with coefficients `c`, evaluated on the H¹
/// space's own GLL element (`H1TriPk` — the correct evaluator at every order).
fn l2_error(space: &H1Space<Mesh<2>>, c: &[f64]) -> f64 {
    let mesh = space.mesh();
    let order = space.order();
    let ref_elem: Box<dyn ReferenceElement> = Box::new(H1TriPk::new(order as usize));
    let quad = ref_elem.quadrature(2 * order + 2);
    let n_ld = ref_elem.n_dofs();
    let mut phi = vec![0.0; n_ld];
    let mut err_sq = 0.0;
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                   - (x2[0] - x0[0]) * (x1[1] - x0[1])).abs();
        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            let vh: f64 = dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| c[d as usize] * p).sum();
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];
            err_sq += w * (vh - f_exact(&xp)).powi(2);
        }
    }
    err_sq.sqrt()
}

/// Project f = sin(πx)·sin(πy) onto the H¹ space through a mass matrix
/// assembled by `MixedAssembler` — the col space of that call goes through
/// `mixed::ref_elem_vol`, so this exercises the D181 arm directly.
fn projection_error(n: usize, order: u8) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), order);
    let mut m: CsrMatrix<f64> =
        MixedAssembler::assemble_bilinear(&space, &space, &[&ScalarMassIntegrator], 2 * order + 1);
    let source = DomainSourceIntegrator::new(f_exact);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 2 * order + 1);
    let bdofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut m, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
    let c = dense_solve(&m, &rhs);
    l2_error(&space, &c)
}

/// D181 before: p = 3 produced garbage (errors 9.85e-1 → 2.13e1 → 1.93e0 on
/// n = 2, 4, 8 — no monotone convergence).  D181 after: O(h⁴).
#[test]
fn d181_mixed_mass_projection_tri_p3_converges_o_h4() {
    let ns = [2usize, 4, 8];
    let errs: Vec<f64> = ns.iter().map(|&n| projection_error(n, 3)).collect();
    let rates: Vec<f64> = (0..errs.len() - 1)
        .map(|i| (errs[i] / errs[i + 1]).ln() / (ns[i + 1] as f64 / ns[i] as f64).ln())
        .collect();
    eprintln!("D181 mixed projection p=3: errors={errs:?} rates={rates:?}");
    assert!(errs.windows(2).all(|w| w[0] > w[1]), "p=3 errors must decrease monotonically: {errs:?}");
    assert!(rates[0] > 3.5, "p=3 rate {:.2} < 3.5 (expected ~4)", rates[0]);
    assert!(rates[1] > 3.5, "p=3 rate {:.2} < 3.5 (expected ~4)", rates[1]);
}

/// p = 2 sanity: the projection was already correct (the p ≤ 2 arms are
/// untouched) and must stay at O(h³).
#[test]
fn d181_mixed_mass_projection_tri_p2_still_o_h3() {
    let ns = [2usize, 4, 8];
    let errs: Vec<f64> = ns.iter().map(|&n| projection_error(n, 2)).collect();
    let rates: Vec<f64> = (0..errs.len() - 1)
        .map(|i| (errs[i] / errs[i + 1]).ln() / (ns[i + 1] as f64 / ns[i] as f64).ln())
        .collect();
    eprintln!("D181 mixed projection p=2: errors={errs:?} rates={rates:?}");
    assert!(rates[0] > 2.5, "p=2 rate {:.2} < 2.5 (expected ~3)", rates[0]);
    assert!(rates[1] > 2.5, "p=2 rate {:.2} < 2.5 (expected ~3)", rates[1]);
}

/// p = 4: covered by the order-generic arm since the D46① extension, and it
/// must keep the O(h⁵) projection rate.
#[test]
fn d181_mixed_mass_projection_tri_p4_converges_o_h5() {
    let ns = [2usize, 4];
    let errs: Vec<f64> = ns.iter().map(|&n| projection_error(n, 4)).collect();
    let rate = (errs[0] / errs[1]).ln() / (ns[1] as f64 / ns[0] as f64).ln();
    eprintln!("D181 mixed projection p=4: errors={errs:?} rate={rate:?}");
    assert!(rate > 4.2, "p=4 rate {:.2} < 4.2 (expected ~5)", rate);
}
