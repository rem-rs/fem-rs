//! Industrial-grade standard benchmarks
//!
//! Each benchmark checks key numerical metrics against a stored baseline
//! to catch regressions.
//!
//! Run with `FEM_UPDATE_BASELINES=1` to (re)generate baseline files.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::{boundary_dofs},
};

fn default_cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() }
}

// ─── 1. 3-D Poisson MMS Benchmark ─────────────────────────────────────────
// Solve -Δu = f on unit cube with u = 0 BC.
// MMS solution: u = sin(πx) sin(πy) sin(πz)

fn solve_poisson_3d_mms(n: usize) -> f64 {
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = H1Space::new(mesh, 1);

    let forcing = DomainSourceIntegrator::new(|x: &[f64]| {
        3.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin()
    });
    let mut a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let mut rhs = Assembler::assemble_linear(&space, &[&forcing], 3);

    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4, 5, 6]);
    for &dof in &bnd {
        a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
    }

    let mut x = vec![0.0_f64; space.n_dofs()];
    solve_pcg_jacobi(&a, &rhs, &mut x, &default_cfg()).unwrap();
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn benchmark_3d_poisson_mms() {
    let norm = solve_poisson_3d_mms(4);
    fem_regression::regression("benchmark_3d_poisson_mms")
        .check("sol_norm_n4_p1", norm)
        .finalize();
}

// ─── 2. Helmholtz MMS (k² = 1) ──────────────────────────────────────────
// -(Δu) + k²u = f on unit square, u = 0 on ∂Ω.
// Exact: u = sin(πx) sin(πy)

fn solve_helmholtz_mms(n: usize, k_sq: f64) -> f64 {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);

    let forcing = DomainSourceIntegrator::new(move |x: &[f64]| {
        (2.0 * PI * PI + k_sq) * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut a = Assembler::assemble_bilinear(&space, &[
        &DiffusionIntegrator { kappa: 1.0 },
        &MassIntegrator { rho: k_sq },
    ], 2);
    let mut rhs = Assembler::assemble_linear(&space, &[&forcing], 3);

    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4]);
    for &dof in &bnd {
        a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
    }

    let mut x = vec![0.0_f64; space.n_dofs()];
    solve_pcg_jacobi(&a, &rhs, &mut x, &default_cfg()).unwrap();
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn ieee1597_helmholtz_mms() {
    let norm = solve_helmholtz_mms(16, 1.0);
    fem_regression::regression("ieee1597_helmholtz_mms")
        .check("sol_norm_k1_n16_p1", norm)
        .finalize();
}

// ─── 3. Helmholtz MMS (k² = 64, k=8) ─────────────────────────────────

#[test]
fn em_helmholtz_mms_k8() {
    let norm = solve_helmholtz_mms(16, 64.0);
    fem_regression::regression("em_helmholtz_mms_k8")
        .check("sol_norm_k8_n16_p1", norm)
        .finalize();
}

// ─── 4. Helmholtz MMS (k² = 256, k=16) ───────────────────────────────

#[test]
fn em_helmholtz_mms_k16() {
    let norm = solve_helmholtz_mms(32, 256.0);
    fem_regression::regression("em_helmholtz_mms_k16")
        .check("sol_norm_k16_n32_p1", norm)
        .finalize();
}
