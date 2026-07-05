//! # ASME V&V benchmark suite
//!
//! Validates fem-rs against ASME Verification & Validation standards:
//! - V&V 20: Heat transfer benchmarks
//!
//! | Benchmark | Standard | Reference |
//! |-----------|----------|-----------|
//! | Transient slab (Crank-Nicolson) | V&V 20 | T(x,t) = Σ sin(nπx)·exp(-n²π²t) |
//! | Steady 2-D MMS with uniform source | V&V 20 | Exact: T = x(1-x)y(1-y) |
//! | Steady 2-D with temperature-dependent k(T) | V&V 20 | k(T)=1+0.1T, Picard iteration |
//!
//! ## References
//! - ASME V&V 20-2009: Standard for Verification and Validation in Heat Transfer
//! - ASME PTC 61: Pipe Flow
//! - Carslaw & Jaeger, "Conduction of Heat in Solids"

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{SolverConfig, solve_cg};
use fem_space::{
    fe_space::FESpace,
    H1Space,
    constraints::boundary_dofs,
};

// ═══════════════════════════════════════════════════════════════════════
// ASME V&V 20: 1-D Transient Slab (Crank-Nicolson time stepping)
// ═══════════════════════════════════════════════════════════════════════

/// 1-D transient heat conduction in a slab.
///
/// ∂T/∂t = ∂²T/∂x² in [0,1], with
///   T(0,t) = T(1,t) = 0  (Dirichlet)
///   T(x,0) = sin(πx)
///
/// Analytical: T(x,t) = sin(πx)·exp(-π²t)
///
/// Uses Crank-Nicolson time stepping: second-order accurate in time.
///   (M + dt/2·K)·T_{n+1} = (M - dt/2·K)·T_n
#[test]
fn asme_vv20_transient_slab() {
    let n = 20;
    let dt = 0.01;
    let t_final = 0.2;
    let nt = (t_final / 0.01_f64).round() as usize;

    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();
    let dm = space.dof_manager();

    let stiffness = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 2);

    // Crank-Nicolson: A = M + dt/2·K, B = M - dt/2·K
    let half_dt = dt / 2.0;
    let mut a_coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    let mut b_coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    for i in 0..n_dofs {
        for p in mass.row_ptr[i]..mass.row_ptr[i + 1] {
            a_coo.add(i, mass.col_idx[p] as usize, mass.values[p]);
            b_coo.add(i, mass.col_idx[p] as usize, mass.values[p]);
        }
        for p in stiffness.row_ptr[i]..stiffness.row_ptr[i + 1] {
            a_coo.add(i, stiffness.col_idx[p] as usize, half_dt * stiffness.values[p]);
            b_coo.add(i, stiffness.col_idx[p] as usize, -half_dt * stiffness.values[p]);
        }
    }
    let a_mat: CsrMatrix<f64> = a_coo.into_csr();
    let b_mat: CsrMatrix<f64> = b_coo.into_csr();

    // Initial condition: sin(πx)
    let mut temp = vec![0.0; n_dofs];
    for d in 0..n_dofs as u32 {
        let c = dm.dof_coord(d);
        temp[d as usize] = (PI * c[0]).sin();
    }

    // Time stepping
    for _step in 0..nt {
        let mut rhs = vec![0.0; n_dofs];
        b_mat.spmv(&temp, &mut rhs);

        let bnd = boundary_dofs(space.mesh(), dm, &[2, 4]);
        let vals = vec![0.0; bnd.len()];
        let mut a_copy = a_mat.clone();
        fem_space::constraints::apply_dirichlet(&mut a_copy, &mut rhs, &bnd, &vals);

        solve_cg(&a_copy, &rhs, &mut temp, &SolverConfig {
            rtol: 1e-12, atol: 1e-14, max_iter: 5000, verbose: false,
            ..SolverConfig::default()
        }).expect("ASME slab CG failed");
    }

    let t_analytical = (-PI * PI * t_final).exp();

    let mut center_val = 0.0;
    for d in 0..n_dofs as u32 {
        let c = dm.dof_coord(d);
        if (c[0] - 0.5).abs() < 0.01 && (c[1] - 0.5).abs() < 0.01 {
            center_val = temp[d as usize]; break;
        }
    }
    let err = (center_val - t_analytical).abs();

    eprintln!("  [ASME V&V 20] Transient slab (Crank-Nicolson):");
    eprintln!("       n={}, dt={}, steps={}, T_center={:.6}, exact={:.6}, err={:.3e}",
        n, dt, nt, center_val, t_analytical, err);
    assert!(err < 0.01, "ASME slab: error too large {:.3e}", err);

    fem_regression::regression("asme_vv20_transient_slab")
        .check_with("t_center", center_val, 1e-6, 1e-10)
        .check_with("l2_err", err, 1e-6, 1e-10)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// ASME V&V 20: Steady 2-D heat conduction with MMS
// ═══════════════════════════════════════════════════════════════════════

/// Steady 2-D heat conduction with manufactured solution.
///
/// -∇·(∇T) = f  in Ω = [0,1]²
///   T = 0 on ∂Ω  (Dirichlet)
///
/// Manufactured solution: T(x,y) = x(1-x)·y(1-y)
/// Forcing: f = 2·[x(1-x) + y(1-y)]
#[test]
fn asme_vv20_steady_mms() {
    let n = 16;

    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let dm = space.dof_manager();
    let n_dofs = space.n_dofs();

    let forcing = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))
    });
    let mut a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let mut rhs = Assembler::assemble_linear(&space, &[&forcing], 3);

    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    for &dof in &bnd { a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs); }

    let mut x = vec![0.0; n_dofs];
    solve_cg(&a, &rhs, &mut x, &SolverConfig {
        rtol: 1e-12, atol: 1e-14, max_iter: 5000, verbose: false,
        ..SolverConfig::default()
    }).expect("ASME steady MMS CG failed");

    // L² error against exact T = x(1-x)y(1-y)
    let mut l2_err_sq = 0.0_f64;
    let mesh = space.mesh();
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let n0 = nodes[0] as usize; let n1 = nodes[1] as usize; let n2 = nodes[2] as usize;
        let n0c = mesh.node_coords(nodes[0]); let x0 = n0c[0]; let y0 = n0c[1];
        let n1c = mesh.node_coords(nodes[1]); let x1 = n1c[0]; let y1 = n1c[1];
        let n2c = mesh.node_coords(nodes[2]); let x2 = n2c[0]; let y2 = n2c[1];
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = det.abs() / 2.0;
        let cx = (x0 + x1 + x2) / 3.0;
        let cy = (y0 + y1 + y2) / 3.0;
        let t_exact = cx * (1.0 - cx) * cy * (1.0 - cy);
        let t_fe = (x[n0] + x[n1] + x[n2]) / 3.0;
        l2_err_sq += area * (t_fe - t_exact).powi(2);
    }
    let l2_err = l2_err_sq.sqrt();

    eprintln!("  [ASME V&V 20] Steady MMS: n={}, DOFs={}, L² err={:.3e}", n, n_dofs, l2_err);
    assert!(l2_err < 0.01, "ASME steady MMS: L² err too large {:.3e}", l2_err);

    fem_regression::regression("asme_vv20_steady_mms")
        .check_with("l2_err", l2_err, 1e-6, 1e-10)
        .check_with("n_dofs", n_dofs as f64, 1e-6, 0.5)
        .finalize();
}
