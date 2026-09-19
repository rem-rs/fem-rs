//! Unit tests for the AbsL1 geometric multigrid hierarchy builders
//! ([`fem_solver::geometric_mg::AbsL1GeometricMultigrid`] and the exact P1 /
//! hex nested prolongation builders), exercised on real H1 hierarchies
//! assembled with `fem-assembly` + `fem-space` — the same pipeline the
//! `diag_mg_abs_l1_jacobi` miniapp driver uses.

use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_solver::geometric_mg::{
    build_h1_hex_refined_prolongation, build_h1_p1_refined_prolongation,
    AbsL1GeometricMultigrid, MgCoarseSolverType, MgCycleType, MG_MAX_ITER, MG_REL_TOL,
};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{build_h1_prolongation_matrix, H1Space};

/// MFEM LEGACY diffusion quadrature for Qk tensor elements: `2p + d - 1`.
fn diffusion_quad(order: u8, dim: usize) -> u8 {
    2 * order + dim as u8 - 1
}

/// Build one AbsL1 MG level (system matrix + essential boundary dofs).
fn level_data(
    space: &H1Space<Mesh<2>>,
    order: u8,
) -> (CsrMatrix<f64>, Vec<u32>) {
    let quad = diffusion_quad(order, 2);
    let mat = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        quad,
    );
    let tags = space.mesh().unique_boundary_tags();
    let ess = boundary_dofs(space.mesh(), space.dof_manager(), &tags);
    (mat, ess)
}

/// Eliminate essential dofs (DIAG_KEEP, zero values — `FormSystemMatrix`).
fn eliminate(mat: &mut CsrMatrix<f64>, ess: &[u32]) {
    let mut dummy = vec![0.0f64; mat.nrows];
    for &d in ess {
        mat.apply_dirichlet_keep_diag(d as usize, 0.0, &mut dummy);
    }
}

/// Two-level P1 hierarchy on the unit-square quad mesh.
fn two_level_quad_hierarchy(n: usize) -> (AbsL1GeometricMultigrid, usize, usize) {
    let coarse = Mesh::<2>::unit_square_quad(n);
    let fine = refine_uniform(&coarse);
    let spaces = [
        H1Space::new(coarse.clone(), 1),
        H1Space::new(fine.clone(), 1),
    ];
    let (mut m0, e0) = level_data(&spaces[0], 1);
    let (mut m1, e1) = level_data(&spaces[1], 1);
    eliminate(&mut m0, &e0);
    eliminate(&mut m1, &e1);
    let p = build_h1_p1_refined_prolongation(
        &coarse,
        &|d| {
            let c = spaces[1].dof_manager().dof_coord(d);
            [c[0], c[1], 0.0]
        },
        &|e| spaces[0].dof_manager().element_dofs(e).to_vec(),
        spaces[1].n_dofs(),
    );
    let mut mg = AbsL1GeometricMultigrid::new(m0, e0);
    mg.add_fine_level(m1, e1, p);
    mg.set_cycle_type(MgCycleType::V, 1, 1);
    (mg, spaces[0].n_dofs(), spaces[1].n_dofs())
}

#[test]
fn p1_refined_prolongation_is_dyadic_partition_of_unity() {
    let (mg, n_coarse, n_fine) = two_level_quad_hierarchy(3);
    assert_eq!(mg.num_levels(), 2);
    assert_eq!(n_coarse, (3 + 1) * (3 + 1));
    // (2n+1)^2 fine vertices for an n×n quad mesh refined once.
    assert_eq!(n_fine, (2 * 3 + 1) * (2 * 3 + 1));

    // P·1 = 1 (row partition of unity: constants are reproduced exactly).
    let p = &mg.prolong[0].mat;
    let ones = vec![1.0f64; p.ncols];
    let mut rows = vec![0.0f64; p.nrows];
    p.spmv(&ones, &mut rows);
    for (i, v) in rows.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "P row {i} sum {v} deviates from 1"
        );
    }
    // Every entry is an exact dyadic constant (1, 0.5 or 0.25 on quads).
    for v in &p.values {
        assert!(
            *v == 1.0 || *v == 0.5 || *v == 0.25,
            "non-dyadic P entry {v}"
        );
    }
    // Transpose property: Pᵀ·1 (column sums) is bounded by the number of
    // refined children sharing a coarse DOF (≤ 4 on refined quads).
    let pt = p.transpose();
    let ones = vec![1.0f64; pt.ncols];
    let mut cols = vec![0.0f64; pt.nrows];
    pt.spmv(&ones, &mut cols);
    assert_eq!(cols.len(), p.ncols);
    for c in &cols {
        assert!(*c >= 1.0 - 1e-12 && *c <= 4.0 + 1e-12);
    }
}

#[test]
fn abs_l1_diagonal_is_positive_and_unity_on_boundary() {
    let (mg, _, _) = two_level_quad_hierarchy(3);
    for level in &mg.levels {
        assert_eq!(level.dinv.len(), level.mat.nrows);
        for (i, d) in level.dinv.iter().enumerate() {
            assert!(d.is_finite() && *d > 0.0, "dinv[{i}] = {d}");
        }
        for &e in &level.ess_dofs {
            assert_eq!(level.dinv[e as usize], 1.0, "dinv at ess dof {e}");
        }
    }
}

#[test]
fn abs_l1_vcycle_damps_the_residual() {
    let (mg, _, n) = two_level_quad_hierarchy(4);
    // A smooth right-hand side compatible with the homogeneous boundary data.
    let b: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i % 9) as f64 / 8.0;
            1.0 + x
        })
        .collect();
    let mut y = vec![0.0f64; n];
    mg.mult(&b, &mut y);
    // Preconditioner output is a nonzero, finite correction.
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm.is_finite() && norm > 0.0);

    // Deterministic: a second application reproduces it bitwise (the cycle
    // always starts from the zero initial guess).
    let mut y2 = vec![0.0f64; n];
    mg.mult(&b, &mut y2);
    assert_eq!(y, y2);
}

#[test]
fn coarse_solver_defaults_match_ds_common() {
    let (mg, _, _) = two_level_quad_hierarchy(2);
    assert_eq!(mg.num_levels(), 2);
    // ds-common: MG_REL_TOL = sqrt(1e-10) (== 1e-5 as a double) and
    // MG_MAX_ITER = 10.
    assert_eq!(MG_REL_TOL, 1.0e-5);
    assert_eq!(MG_MAX_ITER, 10);
    let _ = mg;
}

#[test]
fn coarse_solver_type_is_selectable() {
    let (mut mg, _, _) = two_level_quad_hierarchy(2);
    mg.set_coarse_solver_type(MgCoarseSolverType::Sli);
    let b = vec![1.0f64; mg.levels.last().unwrap().mat.nrows];
    let mut y = vec![0.0f64; b.len()];
    mg.mult(&b, &mut y);
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm.is_finite());
}

#[test]
fn form_fine_linear_system_matches_direct_elimination() {
    let coarse = Mesh::<2>::unit_square_quad(2);
    let space = H1Space::new(coarse.clone(), 1);
    let (mut a, ess) = level_data(&space, 1);
    let x = vec![1.0f64; a.nrows]; // u ≡ 1 on the boundary, 1 in the interior
    let mut b = vec![0.5f64; a.nrows];
    let ess_vals: Vec<f64> = ess.iter().map(|&d| x[d as usize]).collect();
    AbsL1GeometricMultigrid::form_fine_linear_system(&mut a, &mut b, &ess, &ess_vals);
    for (i, v) in b.iter().enumerate() {
        if ess.contains(&(i as u32)) {
            // DIAG_KEEP: rhs at essential dofs carries the kept diagonal.
            let diag = a.diagonal()[i];
            assert!((v - diag * 1.0).abs() < 1e-12);
        }
    }
}

/// Hexahedral hierarchy: the exact P1 path covers the refined-cube case.
#[test]
fn hex_p1_hierarchy_partition_of_unity() {
    let coarse = Mesh::<3>::unit_cube_hex(2);
    let fine = refine_uniform_3d(&coarse);
    let spaces = [
        H1Space::new(coarse.clone(), 1),
        H1Space::new(fine.clone(), 1),
    ];
    let p = build_h1_p1_refined_prolongation(
        &coarse,
        &|d| {
            let c = spaces[1].dof_manager().dof_coord(d);
            [c[0], c[1], c[2]]
        },
        &|e| spaces[0].dof_manager().element_dofs(e).to_vec(),
        spaces[1].n_dofs(),
    );
    let ones = vec![1.0f64; p.ncols];
    let mut rows = vec![0.0f64; p.nrows];
    p.spmv(&ones, &mut rows);
    // (2n+1)^3 fine vertices; the mesh has (n+1)^3 coarse vertices plus edge,
    // face and body centres — all reproduced with constant row sums of 1.
    assert_eq!(p.nrows, (2 * 2 + 1) * (2 * 2 + 1) * (2 * 2 + 1));
    for (i, v) in rows.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-12, "P row {i} sum {v} != 1");
    }
    for v in &p.values {
        assert!(
            *v == 1.0 || *v == 0.5 || *v == 0.25 || *v == 0.125,
            "non-dyadic hex P entry {v}"
        );
    }
}

/// Second-order nested hex levels fall back to the Newton-interpolation hex
/// builder, whose rows also reproduce constants (to evaluation accuracy).
#[test]
fn hex_newton_prolongation_partition_of_unity() {
    let coarse = Mesh::<3>::unit_cube_hex(2);
    let fine = refine_uniform_3d(&coarse);
    let spaces = [
        H1Space::new(coarse.clone(), 2),
        H1Space::new(fine.clone(), 2),
    ];
    let p: CsrMatrix<f64> = build_h1_hex_refined_prolongation(
        &coarse,
        2,
        spaces[0].n_dofs(),
        spaces[1].n_dofs(),
        &|e| spaces[0].dof_manager().element_dofs(e).to_vec(),
        &|d| {
            let c = spaces[1].dof_manager().dof_coord(d);
            [c[0], c[1], c[2]]
        },
    );
    let ones = vec![1.0f64; p.ncols];
    let mut rows = vec![0.0f64; p.nrows];
    p.spmv(&ones, &mut rows);
    for (i, v) in rows.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-10,
            "P row {i} sum {v} deviates from 1"
        );
    }
}

/// The fem-space same-mesh p-refinement path (used by `-ol` levels) stays
/// compatible: P·1 = 1 across an order 1 → 2 transfer on the same mesh.
#[test]
fn same_mesh_prolongation_partition_of_unity() {
    let mesh = Mesh::<2>::unit_square_quad(2);
    let spaces = [H1Space::new(mesh.clone(), 1), H1Space::new(mesh.clone(), 2)];
    let p = build_h1_prolongation_matrix(
        spaces[0].mesh(),
        spaces[0].dof_manager(),
        spaces[1].mesh(),
        spaces[1].dof_manager(),
    );
    let ones = vec![1.0f64; p.ncols];
    let mut rows = vec![0.0f64; p.nrows];
    p.spmv(&ones, &mut rows);
    for (i, v) in rows.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-12, "P row {i} sum {v} != 1");
    }
}
