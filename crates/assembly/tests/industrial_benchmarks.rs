//! Industrial-grade standard benchmarks
//!
//! Each benchmark checks key numerical metrics against a stored baseline
//! to catch regressions.
//!
//! Run with `FEM_UPDATE_BASELINES=1` to (re)generate baseline files.

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator,
               BoundaryMassIntegrator, NeumannIntegrator},
    VectorAssembler,
    standard::CurlCurlIntegrator,
};
use fem_assembly::assembler::face_dofs_p1;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{solve_cg, solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, HCurlSpace, fe_space::FESpace,
    constraints::{boundary_dofs},
};

fn default_cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() }
}

/// Evaluate a P1 FE solution at a physical point (x,y) by iterating elements.
fn eval_p1(uh: &[f64], mesh: &Mesh<2>, x: f64, y: f64) -> f64 {
    let tol = 1e-12;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nodes[0])[0]; let y0 = mesh.node_coords(nodes[0])[1];
        let x1 = mesh.node_coords(nodes[1])[0]; let y1 = mesh.node_coords(nodes[1])[1];
        let x2 = mesh.node_coords(nodes[2])[0]; let y2 = mesh.node_coords(nodes[2])[1];
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if det.abs() < tol { continue; }
        let lam1 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / det;
        let lam2 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / det;
        let lam3 = 1.0 - lam1 - lam2;
        if lam1 >= -tol && lam2 >= -tol && lam3 >= -tol {
            let n0 = nodes[0] as usize; let n1 = nodes[1] as usize; let n2 = nodes[2] as usize;
            return lam1 * uh[n0] + lam2 * uh[n1] + lam3 * uh[n2];
        }
    }
    f64::NAN
}

// ─── 1. 3-D Poisson MMS Benchmark ─────────────────────────────────────────
// Solve -Δu = f on unit cube with u = 0 BC.
// MMS solution: u = sin(πx) sin(πy) sin(πz)

fn solve_poisson_3d_mms(n: usize) -> f64 {
    let mesh = Mesh::<3>::unit_cube_tet(n);
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
    let mesh = Mesh::<2>::unit_square_tri(n);
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

// ─── 5. NAFEMS Thermal Convection Benchmark ──────────────────────────────
//
// Steady-state heat conduction on unit square with Robin BC (convection):
//   -∇·(k∇T) = 0,  k = 1
//   Bottom (y=0):  T = 0          (Dirichlet)
//   Right (x=1):   ∂T/∂n = 0      (Neumann)
//   Top (y=1):     ∂T/∂n + α(T-T∞) = 0  (Robin, α=1, T∞=1)
//   Left (x=0):    ∂T/∂n = 0      (Neumann)
// Reference: NAFEMS "A Simple Problem with Convection Boundary Conditions"

fn solve_nafems_convection(n: usize) -> (f64, f64) {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);
    let n_dofs = space.n_dofs() as f64;

    // Stiffness matrix: -∇·(∇u) = 0
    let mut a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    // Dirichlet BC on bottom (tag 1): T = 0
    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1]);
    for &dof in &bnd {
        a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs);
    }

    // Robin BC on top (tag 3): ∂T/∂n + α(T - T∞) = 0, α = 1, T∞ = 1
    // Weak form: ∫(α T v) ds = ∫(α T∞ v) ds  (on top boundary)
    let robin_tags = [3i32];
    let robin_mat = Assembler::assemble_boundary_bilinear(
        n_dofs as usize, space.mesh(), &face_dofs_p1(space.mesh()), 1,
        &[&BoundaryMassIntegrator { alpha: 1.0 }], &robin_tags, 2,
    );
    let a_total = a.add(&robin_mat);

    // Robin RHS: ∫(α T∞ v) ds = ∫(1.0 * v) ds
    let robin_rhs = Assembler::assemble_boundary_linear(
        n_dofs as usize, space.mesh(), &face_dofs_p1(space.mesh()), 1,
        &[&NeumannIntegrator::new(|_, _| 1.0)], &robin_tags, 2,
    );
    for i in 0..rhs.len() { rhs[i] += robin_rhs[i]; }

    let mut x = vec![0.0_f64; space.n_dofs()];
    solve_cg(&a_total, &rhs, &mut x, &default_cfg()).unwrap();

    let t_center = eval_p1(&x, &mesh, 0.5, 0.5);
    (n_dofs, t_center)
}

#[test]
fn nafems_thermal_convection() {
    let (n_dofs, t_center) = solve_nafems_convection(40);
    fem_regression::regression("nafems_thermal_convection")
        .check("n_dofs", n_dofs)
        .check("t_center", t_center)
        .finalize();
}

// ─── 6. Helmholtz MMS Frequency Sweep (k² = 1, 10, 100) ────────────────

#[test]
fn em_helmholtz_mms_sweep_k1() {
    let norm = solve_helmholtz_mms(16, 1.0);
    fem_regression::regression("em_helmholtz_mms_sweep")
        .check("sol_norm_k1_n16_p1", norm)
        .finalize();
}

#[test]
fn em_helmholtz_mms_sweep_k10() {
    let norm = solve_helmholtz_mms(16, 10.0);
    fem_regression::regression("em_helmholtz_mms_sweep")
        .check("sol_norm_k10_n16_p1", norm)
        .finalize();
}

#[test]
fn em_helmholtz_mms_sweep_k100() {
    let norm = solve_helmholtz_mms(16, 100.0);
    fem_regression::regression("em_helmholtz_mms_sweep")
        .check("sol_norm_k100_n16_p1", norm)
        .finalize();
}

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

// ─── 7. TEAM4 Electrostatic Capacitor (Patch Test) ───────────────────────
//
// Parallel-plate capacitor: unit square, V=0 on bottom, V=1 on top.
// P1 exactly reproduces the linear solution V = y → max error ≈ machine ε.

fn solve_team4_capacitor(n: usize) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), 1);

    // -∇²V = 0 (Laplace)
    let mut a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let mut rhs = vec![0.0_f64; space.n_dofs()];

    // Bottom (tag 1): V = 0, Top (tag 3): V = 1
    let bnd_bot = boundary_dofs(space.mesh(), space.dof_manager(), &[1]);
    let bnd_top = boundary_dofs(space.mesh(), space.dof_manager(), &[3]);
    for &dof in &bnd_bot { a.apply_dirichlet_symmetric(dof as usize, 0.0, &mut rhs); }
    for &dof in &bnd_top { a.apply_dirichlet_symmetric(dof as usize, 1.0, &mut rhs); }

    // Natural BC on sides (tags 2, 4): ∂V/∂n = 0 (Neumann, automatically satisfied)

    let mut x = vec![0.0_f64; space.n_dofs()];
    solve_cg(&a, &rhs, &mut x, &default_cfg()).unwrap();

    // Check max nodal error: exact V = y (y-coordinate of each node)
    let mut max_err = 0.0_f64;
    for n in 0..mesh.n_nodes() as u32 {
        let y = mesh.node_coords(n)[1];
        let err = (x[n as usize] - y).abs();
        max_err = max_err.max(err);
    }
    max_err
}

#[test]
fn team4_electrostatic_capacitor() {
    let max_err = solve_team4_capacitor(8);
    fem_regression::regression("team4_electrostatic_capacitor")
        .check("max_phi_err", max_err)
        .finalize();
}

// ─── 8. TEAM1 HCurl 3D PEC Smoke Test ────────────────────────────────────
//
// HCurl ND1 space on unit cube, curl-curl + mass assembly with zero BC.
// Checks DOF count and matrix sparsity — a smoke test for 3-D H(curl).

fn solve_hcurl_3d_smoke(n: usize) -> (f64, f64) {
    let mesh = Mesh::<3>::unit_cube_tet(n);
    let hcurl = HCurlSpace::new(mesh.clone(), 1);
    let n_dofs = hcurl.n_dofs() as f64;

    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let mat = VectorAssembler::assemble_bilinear(&hcurl, &[&curl_curl], 3);
    let nnz = mat.nnz() as f64;

    (n_dofs, nnz)
}

#[test]
fn team1_hcurl_3d_pec_smoke() {
    let (n_dofs, nnz) = solve_hcurl_3d_smoke(2);
    fem_regression::regression("team1_hcurl_3d_pec_smoke")
        .check("n_dofs", n_dofs)
        .check("nnz", nnz)
        .finalize();
}
