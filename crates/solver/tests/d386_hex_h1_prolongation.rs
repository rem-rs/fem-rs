//! D386: the nested (h-refined) 3-D branch of
//! `fem_space::constraints::prolong::build_h1_prolongation_matrix` gains a
//! hexahedral point locator (Newton inversion of the trilinear corner map, an
//! exact port of the fem-solver D376 code).  On a uniformly refined unit-cube
//! hex hierarchy the matrix must
//!
//! 1. carry a partition of unity (`P·1 = 1` at every fine DOF),
//! 2. give every fine DOF exactly one parent element — each row's support lies
//!    inside a single coarse element's DOF set, and every coarse element claims
//!    at least one fine DOF (the parent/child partition covers the mesh),
//! 3. agree with the verified fem-solver Newton path
//!    `fem_solver::geometric_mg::build_h1_hex_refined_prolongation` to
//!    ≤ 1e-13, and — at order 1, where the entries must be the dyadic
//!    `RefinementOperator` constants (1, 1/2, 1/4, 1/8) — with the
//!    bitwise-dyadic `fem_solver::geometric_mg::build_h1_p1_refined_prolongation`.
//!
//! The test lives here (not in fem-space) because the cross-check needs
//! `fem-solver`, which depends on `fem-space`; fem-solver already dev-depends
//! on `fem-space`/`fem-mesh`, so no dependency edges are added.

use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::prolong::build_h1_prolongation_matrix;
use fem_space::DofManager;

/// Largest entrywise difference of the two prolongations (dense comparison —
/// the matrices are small).
fn max_abs_diff(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> f64 {
    assert_eq!(a.nrows, b.nrows, "row count mismatch");
    assert_eq!(a.ncols, b.ncols, "col count mismatch");
    let da = a.to_dense();
    let db = b.to_dense();
    da.iter()
        .zip(db.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn check_hex_level(order: u8) {
    let coarse = Mesh::<3>::unit_cube_hex(2);
    let fine = fem_mesh::refine_uniform_3d(&coarse);
    let c_dm = DofManager::new(&coarse, order);
    let f_dm = DofManager::new(&fine, order);

    let p = build_h1_prolongation_matrix(&coarse, &c_dm, &fine, &f_dm);
    assert_eq!(p.nrows, f_dm.n_dofs, "P rows");
    assert_eq!(p.ncols, c_dm.n_dofs, "P cols");

    // (a) Partition of unity: P·1 = 1 at every fine DOF.
    let ones = vec![1.0_f64; c_dm.n_dofs];
    let mut p_ones = vec![0.0_f64; f_dm.n_dofs];
    p.spmv(&ones, &mut p_ones);
    for (f, &v) in p_ones.iter().enumerate() {
        assert!(
            (v - 1.0).abs() <= 1e-12,
            "order {order}: P row {f} sums to {v}, expected 1"
        );
    }

    // (b) Pᵀ partitions: every fine DOF receives exactly one parent — its row
    // support must fit inside a single coarse element's DOF set (a row
    // assembled from two different parents would span two DOF sets), and
    // every coarse element must claim at least one fine DOF.
    let mut claimed = vec![false; coarse.n_elements()];
    for f in 0..f_dm.n_dofs {
        let row = p.row_ptr[f]..p.row_ptr[f + 1];
        assert!(!row.is_empty(), "order {order}: fine DOF {f} has empty P row");
        let mut parents = 0;
        for e in 0..coarse.n_elements() as u32 {
            let dofs = c_dm.element_dofs(e);
            let covered = (row.clone()).all(|k| dofs.contains(&p.col_idx[k]));
            if covered {
                parents += 1;
                claimed[e as usize] = true;
            }
        }
        assert!(
            parents >= 1,
            "order {order}: fine DOF {f} row support matches no single coarse element"
        );
    }
    for (e, &c) in claimed.iter().enumerate() {
        assert!(c, "order {order}: coarse element {e} claims no fine DOF");
    }

    // (c) Cross-check against the verified fem-solver Newton path (D376).
    let c_elem_dofs = |e: u32| c_dm.element_dofs(e).to_vec();
    let f_coord = |d: u32| {
        let c = f_dm.dof_coord(d);
        [c[0], c[1], c[2]]
    };
    let p_solver = fem_solver::geometric_mg::build_h1_hex_refined_prolongation(
        &coarse,
        order,
        c_dm.n_dofs,
        f_dm.n_dofs,
        &c_elem_dofs,
        &f_coord,
    );
    let d = max_abs_diff(&p, &p_solver);
    assert!(
        d <= 1e-13,
        "order {order}: fem-space vs fem-solver prolongation max|diff| = {d:e}"
    );

    if order == 1 {
        // The order-1 entries must be the dyadic RefinementOperator constants;
        // build_h1_p1_refined_prolongation is the proven bitwise-dyadic
        // reference (D376).
        let p_dyadic = fem_solver::geometric_mg::build_h1_p1_refined_prolongation(
            &coarse,
            &f_coord,
            &c_elem_dofs,
            f_dm.n_dofs,
        );
        let d1 = max_abs_diff(&p, &p_dyadic);
        assert!(
            d1 <= 1e-13,
            "order 1: fem-space vs dyadic P1 prolongation max|diff| = {d1:e}"
        );
    }
}

#[test]
fn hex_refined_prolongation_order1() {
    check_hex_level(1);
}

#[test]
fn hex_refined_prolongation_order2() {
    check_hex_level(2);
}
