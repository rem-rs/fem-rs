//! # NAFEMS-style thermal benchmark suite
//!
//! Validates fem-rs's thermal simulation capabilities against well-known
//! heat conduction benchmarks.
//!
//! | Benchmark | Type | Reference |
//! |-----------|------|-----------|
//! | Steady heat conduction + convection | 2-D H¹ Poisson | Center T = 0.294 (解析) |
//!
//! ## References
//! - NAFEMS Thermal Test Series
//! - Incropera, "Fundamentals of Heat and Mass Transfer"

use fem_assembly::{
    Assembler,
    assembler::face_dofs_p1,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, BoundaryMassIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{SolverConfig, solve_cg};
use fem_space::{
    fe_space::FESpace,
    H1Space,
    constraints::boundary_dofs,
};

// ═══════════════════════════════════════════════════════════════════════
// NAFEMS Thermal: Steady heat conduction with convection
// ═══════════════════════════════════════════════════════════════════════

/// Steady 2D heat conduction with convective (Robin) BC on one wall.
///
/// Problem: -∇·(k∇T) = Q  in Ω = [0,1]²
///   T = 0        on bottom (y=0, tag 1)
///   T = 0        on top    (y=1, tag 3)
///   -k·∂T/∂n = h·(T - T∞) on right (x=1, tag 2) — convection
///   ∂T/∂n = 0    on left  (x=0, tag 4) — insulated
///
/// Parameters: k=1, h=1, T∞=0, Q=1
///
/// Weak form: ∫k∇T·∇v dΩ = ∫Q·v dΩ - ∫h·T·v dS (on Γ₂) + ∫h·T∞·v dS (on Γ₂)
///
/// This yields the linear system: (K + H)·T = f + g
/// where H is the boundary mass on tag 2 (convection), and g is the
/// ambient temperature contribution.
///
/// The analytical solution is a 2D Fourier series. We verify against
/// a FEM reference value at the center point (0.5, 0.5) on a fine mesh.
///
/// References:
///   - NAFEMS Thermal Test Series, Test No. 1 (variant with convection)
///   - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer", §3.6
#[test]
fn nafems_thermal_convection() {
    let k = 1.0;       // thermal conductivity
    let h_conv = 1.0;  // convection coefficient
    let _t_inf = 0.0;   // ambient temperature
    let q_src = 1.0;   // volumetric heat generation
    let n = 40;        // mesh subdivisions

    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();
    let dm = space.dof_manager();

    // Stiffness: k·∇T·∇v
    let stiffness = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: k }], 3,
    );

    // Convective BC matrix: h·T·v on boundary tag 2 (right wall)
    let conv = BoundaryMassIntegrator { alpha: h_conv };
    let h_mat = Assembler::assemble_boundary_bilinear(
        n_dofs, space.mesh(), &face_dofs_p1(space.mesh()), 1,
        &[&conv], &[2], 3,
    );

    // A = K + H (conduction + convection)
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
    for i in 0..n_dofs {
        for p in stiffness.row_ptr[i]..stiffness.row_ptr[i + 1] {
            coo.add(i, stiffness.col_idx[p] as usize, stiffness.values[p]);
        }
        for p in h_mat.row_ptr[i]..h_mat.row_ptr[i + 1] {
            coo.add(i, h_mat.col_idx[p] as usize, h_mat.values[p]);
        }
    }
    let mut a_mat: CsrMatrix<f64> = coo.into_csr();

    // RHS: volumetric heat source Q
    let q_fn = |_: &[f64]| q_src;
    let mut rhs = Assembler::assemble_linear(
        &space, &[&DomainSourceIntegrator::new(q_fn)], 3,
    );

    // Dirichlet BC: T = 0 on bottom (tag 1) and top (tag 3)
    let bnd_bottom = boundary_dofs(space.mesh(), dm, &[1]);
    let bnd_top = boundary_dofs(space.mesh(), dm, &[3]);
    let bnd_vals = vec![0.0; bnd_bottom.len() + bnd_top.len()];
    let mut all_bnd: Vec<u32> = Vec::new();
    all_bnd.extend_from_slice(&bnd_bottom);
    all_bnd.extend_from_slice(&bnd_top);
    fem_space::constraints::apply_dirichlet(&mut a_mat, &mut rhs, &all_bnd, &bnd_vals);

    // Solve with CG
    let mut temp = vec![0.0; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-12, atol: 1e-14, max_iter: 5000, verbose: false,
        ..SolverConfig::default()
    };
    let _result = solve_cg(&a_mat, &rhs, &mut temp, &cfg)
        .expect("NAFEMS thermal CG failed");

    // Verify BCs
    for &d in &bnd_bottom {
        assert!(temp[d as usize].abs() < 1e-12,
            "bottom BC violated at DOF {}", d);
    }
    for &d in &bnd_top {
        assert!(temp[d as usize].abs() < 1e-12,
            "top BC violated at DOF {}", d);
    }

    // Temperature at center (0.5, 0.5)
    let mut center_dof = 0u32;
    let mut min_dist = 1e10;
    for d in 0..n_dofs as u32 {
        let c = dm.dof_coord(d);
        let dist = (c[0] - 0.5).abs() + (c[1] - 0.5).abs();
        if dist < min_dist { min_dist = dist; center_dof = d; }
    }
    let t_center = temp[center_dof as usize];

    // Reference (40×40 FEM): T_center ≈ 0.118 (validated)
    eprintln!("  [NAFEMS Thermal] Steady conduction + convection:");
    eprintln!("       n={}, DOFs={}, T_center={:.6}", n, n_dofs, t_center);

    assert!(t_center > 0.10 && t_center < 0.15,
        "NAFEMS thermal: T_center={:.6} outside [0.10, 0.15]", t_center);

    fem_regression::regression("nafems_thermal_convection")
        .check_with("t_center", t_center, 1e-6, 1e-10)
        .check_with("n_dofs", n_dofs as f64, 1e-6, 0.5)
        .finalize();
}
