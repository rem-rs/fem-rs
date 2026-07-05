//! Cross-backend consistency tests for fem-rs.
//!
//! Verifies that the same mathematical problem produces the same numerical
//! result regardless of the execution backend:
//!
//! - Serial vs parallel assembly (identical matrix entries)
//! - Different solver paths (same solution from CG, PCG, GMRES)

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, solve_gmres, solve_pcg_jacobi, SolverConfig};
use fem_space::{
    fe_space::FESpace,
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
};

// ─── Helpers ────────────────────────────────────────────────────────────

/// Solve a Poisson problem with homogeneous Dirichlet BCs.
fn solve_poisson(n: usize, solver: &str) -> Vec<f64> {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let n_dof = space.n_dofs();

    let diff = DiffusionIntegrator { kappa: 1.0 };
    let src = DomainSourceIntegrator::new(|x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin());
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], 3);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n_dof];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let _ = match solver {
        "cg" => solve_cg(&mat, &rhs, &mut u, &cfg),
        "pcg" => solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg),
        "gmres" => solve_gmres(&mat, &rhs, &mut u, 30, &cfg),
        _ => panic!("unknown solver: {solver}"),
    };
    u
}

/// Maximum absolute difference between two vectors.
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Backend consistency — solver convergence
// ═══════════════════════════════════════════════════════════════════════

/// CG, PCG-Jacobi, and GMRES should all converge to the same solution
/// on a standard SPD Poisson problem (up to solver tolerance).
#[test]
fn cross_solver_consistency() {
    let u_cg = solve_poisson(8, "cg");
    let u_pcg = solve_poisson(8, "pcg");
    let u_gmres = solve_poisson(8, "gmres");

    let err_cg_pcg = max_diff(&u_cg, &u_pcg);
    let err_cg_gmres = max_diff(&u_cg, &u_gmres);

    // All solvers converge to the exact solution; differences should be
    // at the level of the solver tolerance (rtol = 1e-10)
    assert!(err_cg_pcg < 1e-8,
        "CG vs PCG differ by {:.3e}", err_cg_pcg);
    assert!(err_cg_gmres < 1e-8,
        "CG vs GMRES differ by {:.3e}", err_cg_gmres);
    eprintln!("  [cross] solver-consistency: CG vs PCG diff={:.3e}, CG vs GMRES diff={:.3e}",
        err_cg_pcg, err_cg_gmres);
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Serial vs parallel assembly (requires --features parallel)
// ═══════════════════════════════════════════════════════════════════════

/// Force serial assembly (high threshold) and parallel assembly (threshold=0)
/// and verify the resulting stiffness matrix is identical.
///
/// Run with:
/// ```bash
/// cargo test -p fem-examples --lib --features parallel -- cross_serial_parallel
/// ```
#[cfg_attr(not(feature = "parallel"), ignore)]
#[test]
fn cross_serial_parallel_assembly() {
    let mesh = SimplexMesh::<2>::unit_square_tri(32); // large enough to trigger parallel
    let space = H1Space::new(mesh, 1);
    let diff = DiffusionIntegrator { kappa: 1.0 };

    // Serial assembly: set threshold very high (parallel never kicks in)
    std::env::set_var("FEM_ASSEMBLY_PARALLEL_MIN_ELEMS", "9999999");
    let mat_serial = Assembler::assemble_bilinear(&space, &[&diff], 3);

    // Parallel assembly: threshold = 0 (always parallel)
    std::env::set_var("FEM_ASSEMBLY_PARALLEL_MIN_ELEMS", "0");
    let mat_parallel = Assembler::assemble_bilinear(&space, &[&diff], 3);

    // Compare matrices entry by entry
    assert_eq!(mat_serial.nrows, mat_parallel.nrows);
    assert_eq!(mat_serial.ncols, mat_parallel.ncols);
    assert_eq!(mat_serial.nnz(), mat_parallel.nnz());

    let mut max_entry_diff = 0.0_f64;
    for i in 0..mat_serial.nrows {
        let s_start = mat_serial.row_ptr[i];
        let s_end = mat_serial.row_ptr[i + 1];
        for pk in s_start..s_end {
            let col = mat_serial.col_idx[pk];
            let sv = mat_serial.values[pk];
            // Find corresponding entry in parallel matrix
            let p_start = mat_parallel.row_ptr[i];
            let p_end = mat_parallel.row_ptr[i + 1];
            for pl in p_start..p_end {
                if mat_parallel.col_idx[pl] == col {
                    let diff = (sv - mat_parallel.values[pl]).abs();
                    max_entry_diff = max_entry_diff.max(diff);
                    break;
                }
            }
        }
    }
    assert!(max_entry_diff < 1e-14,
        "serial vs parallel matrix differ by {:.3e}", max_entry_diff);
    eprintln!("  [cross] serial-vs-parallel: max entry diff = {:.3e}", max_entry_diff);
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Serial vs parallel SpMV (requires --features parallel)
// ═══════════════════════════════════════════════════════════════════════

/// SpMV with serial and parallel paths must produce identical results.
#[cfg_attr(not(feature = "parallel"), ignore)]
#[test]
fn cross_spmv_serial_parallel_consistent() {
    use fem_linalg::{CsrMatrix, FEM_LINALG_SPMV_PARALLEL_MIN_ROWS};

    // Build a large-enough CSR matrix
    let n = 500;
    let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0 { coo.add(i, i - 1, -1.0); }
        if i + 1 < n { coo.add(i, i + 1, -1.0); }
    }
    let a: CsrMatrix<f64> = coo.into_csr();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut y_serial = vec![0.0; n];
    let mut y_parallel = vec![0.0; n];

    // Serial SpMV: set parallel threshold very high
    std::env::set_var(FEM_LINALG_SPMV_PARALLEL_MIN_ROWS, "999999");
    a.spmv(&x, &mut y_serial);

    // Parallel SpMV: threshold = 0
    std::env::set_var(FEM_LINALG_SPMV_PARALLEL_MIN_ROWS, "0");
    a.spmv(&x, &mut y_parallel);

    let diff = max_diff(&y_serial, &y_parallel);
    assert!(diff < 1e-14,
        "serial vs parallel SpMV differ by {:.3e}", diff);
    eprintln!("  [cross] spmv-serial-vs-parallel: max diff = {:.3e}", diff);
}
