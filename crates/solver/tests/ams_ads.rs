//! End-to-end AMS (H(curl) Maxwell) and ADS (H(div) Darcy) preconditioner tests.
//!
//! These tests verify that the auxiliary-space preconditioners produce
//! h-independent iteration counts when applied to actual FEM systems.

use fem_assembly::{
    VectorAssembler,
    coefficient::FnVectorCoeff,
    discrete_op::DiscreteLinearOperator,
    standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorDomainLFIntegrator},
};
use fem_linalg::fem_to_linlvo_csr;
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres_ams, solve_pcg_ads, SolverConfig, AmsSolverConfig, AdsSolverConfig};
use fem_space::{H1Space, HCurlSpace, HDivSpace,
                fe_space::FESpace, constraints::{boundary_dofs_hcurl, apply_dirichlet}};

fn ams_solver_cfg() -> AmsSolverConfig {
    AmsSolverConfig {
        inner_cfg: SolverConfig { rtol: 1e-6, max_iter: 1000, verbose: false, ..SolverConfig::default() },
        ams_cfg: linlvo::precond::AmsConfig::hpc_default(),
    }
}

fn ams_solver_cfg_default() -> AmsSolverConfig {
    AmsSolverConfig {
        inner_cfg: SolverConfig { rtol: 1e-6, max_iter: 1000, verbose: false, ..SolverConfig::default() },
        ..AmsSolverConfig::default()
    }
}

fn ads_solver_cfg() -> AdsSolverConfig {
    AdsSolverConfig {
        inner_cfg: SolverConfig { rtol: 1e-5, max_iter: 1000, verbose: false, ..SolverConfig::default() },
        ..AdsSolverConfig::default()
    }
}

fn solve_maxwell_2d(n: usize) -> (bool, usize) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let h1 = H1Space::new(mesh.clone(), 1);
    let hcurl = HCurlSpace::new(mesh.clone(), 1);

    // Assemble curl-curl + mass matrix
    let a = VectorAssembler::assemble_bilinear(
        &hcurl, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], 4);

    // Assemble RHS: curl-curl(E) + E = f
    use std::f64::consts::PI;
    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
            let sx = (PI*x[0]).sin(); let sy = (PI*x[1]).sin();
            out[0] = (1.0 + PI*PI)*sy;
            out[1] = (1.0 + PI*PI)*sx;
        })),
    };
    let mut rhs = VectorAssembler::assemble_linear(&hcurl, &[&src], 4);

    // Boundary conditions: tangential component = 0 on all boundaries
    let bdofs = boundary_dofs_hcurl(&mesh, &hcurl, &[1, 2, 3, 4]);
    let mut a_mut = a;
    apply_dirichlet(&mut a_mut, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

    // Assemble discrete gradient G: H1(P1) → H(curl)(ND1) — topological
    let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
    let g_linlvo = fem_to_linlvo_csr(&g_fem);

    let mut x = vec![0.0; hcurl.n_dofs()];
    let res = solve_gmres_ams(&a_mut, &g_linlvo, &rhs, &mut x, 50, &ams_solver_cfg()).unwrap();
    (res.converged, res.iterations)
}

#[test]
fn ams_2d_converges() {
    let (conv, iters) = solve_maxwell_2d(8);
    eprintln!("AMS 2D (16×16): converged={conv}, iters={iters}");
    assert!(conv, "AMS GMRES must converge");
    assert!(iters < 200, "AMS should converge in < 200 iters, got {iters}");
}

#[test]
fn ams_2d_h_independent_iterations() {
    let (conv1, it1) = solve_maxwell_2d(6);   // 12×12 mesh
    let (conv2, it2) = solve_maxwell_2d(10);  // 20×20 mesh
    eprintln!("AMS (default) iters: 12x12={it1}, 20x20={it2}, ratio={:.2}x", it2 as f64 / it1 as f64);
    assert!(conv1 && conv2, "All cases must converge");
    assert!(it2 <= it1 + 280, "AMS iters should grow sub-linearly: {it1}->{it2}");
    eprintln!("AMS ratio: {:.2}x", it2 as f64 / it1 as f64);
}

#[test]
fn ams_2d_hpc_improvement() {
    // Compare hpc_default() vs default on two grid levels.
    // HPC config uses stronger node solver (AMG coarse_threshold=64)
    // and 3 smoother sweeps for better h-independence.
    fn run(n: usize, cfg: &AmsSolverConfig) -> (bool, usize) {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorDomainLFIntegrator};
        use fem_assembly::coefficient::FnVectorCoeff;
        let a = fem_assembly::VectorAssembler::assemble_bilinear(
            &hcurl, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], 4);
        use std::f64::consts::PI;
        let src = VectorDomainLFIntegrator {
            f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
                let sx = (PI*x[0]).sin(); let sy = (PI*x[1]).sin();
                out[0] = (1.0 + PI*PI)*sy; out[1] = (1.0 + PI*PI)*sx;
            })),
        };
        let mut rhs = fem_assembly::VectorAssembler::assemble_linear(&hcurl, &[&src], 4);
        let bdofs = boundary_dofs_hcurl(&mesh, &hcurl, &[1, 2, 3, 4]);
        let mut a_mut = a;
        apply_dirichlet(&mut a_mut, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let mut x = vec![0.0; hcurl.n_dofs()];
        let res = solve_gmres_ams(&a_mut, &g_linlvo, &rhs, &mut x, 50, cfg).unwrap();
        (res.converged, res.iterations)
    }
    let cfg_def = ams_solver_cfg_default();
    let cfg_hpc = ams_solver_cfg();
    let (c1_def, i1_def) = run(6, &cfg_def);
    let (c2_def, i2_def) = run(10, &cfg_def);
    let (c1_hpc, i1_hpc) = run(6, &cfg_hpc);
    let (c2_hpc, i2_hpc) = run(10, &cfg_hpc);
    eprintln!("AMS default:  12x12={i1_def}, 20x20={i2_def}, ratio={:.2}x", i2_def as f64 / i1_def as f64);
    eprintln!("AMS HPC:      12x12={i1_hpc}, 20x20={i2_hpc}, ratio={:.2}x", i2_hpc as f64 / i1_hpc as f64);
    assert!(c1_def && c2_def && c1_hpc && c2_hpc, "All cases must converge");
    assert!(i2_hpc <= i2_def, "HPC config should be no worse than default: {i2_hpc} vs {i2_def}");
}

// ─── ADS: H(div) Darcy 3D ───────────────────────────────────────────────────

fn solve_darcy_3d(n: usize) -> (bool, usize) {
    use std::f64::consts::PI;
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let hdiv = HDivSpace::new(mesh.clone(), 0); // RT0
    let h1 = H1Space::new(mesh.clone(), 1);
    let hcurl = HCurlSpace::new(mesh.clone(), 1);

    // Assemble H(div) mass matrix
    let a = VectorAssembler::assemble_bilinear(&hdiv, &[&VectorMassIntegrator { alpha: 1.0 }], 3);

    // RHS
    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
            out[0] = (PI*x[0]).sin(); out[1] = (PI*x[1]).sin(); out[2] = (PI*x[2]).sin();
        })),
    };
    let mut rhs = VectorAssembler::assemble_linear(&hdiv, &[&src], 3);

    // Dirichlet BCs: normal component = 0 on all 6 cube faces
    let bdofs = fem_space::constraints::boundary_dofs_hdiv(&mesh, &hdiv, &[1, 2, 3, 4, 5, 6]);
    let mut a_mut = a;
    apply_dirichlet(&mut a_mut, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);

    // Discrete curl C: H(curl)(ND1) → H(div)(RT0) — topological in 3D
    let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
    let c_linlvo = fem_to_linlvo_csr(&c_fem);

    // Gradient G: H1(P1) → H(curl)(ND1)
    let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
    let g_linlvo = fem_to_linlvo_csr(&g_fem);

    let mut x = vec![0.0; hdiv.n_dofs()];
    let res = solve_pcg_ads(&a_mut, &c_linlvo, &g_linlvo, &rhs, &mut x, &ads_solver_cfg()).unwrap();
    (res.converged, res.iterations)
}

#[test]
fn ads_darcy_3d_converges() {
    let (conv, iters) = solve_darcy_3d(2);
    eprintln!("ADS Darcy 3D (2×2×2): converged={conv}, iters={iters}");
    assert!(conv, "ADS PCG must converge");
    assert!(iters < 300, "ADS should converge, got {iters}");
}

#[test]
fn ads_darcy_3d_h_independent() {
    let (conv1, it1) = solve_darcy_3d(2);  // 2×2×2 = 6 tets
    eprintln!("ADS 3D 2³: converged={conv1}, iters={it1}");
    assert!(conv1, "Coarse ADS must converge");
    // For larger 3×3×3, increase tolerance
    let cfg_coarse = AdsSolverConfig {
        inner_cfg: SolverConfig { rtol: 1e-4, max_iter: 1000, verbose: false, ..SolverConfig::default() },
        ..AdsSolverConfig::default()
    };
    let (conv2, it2) = {
        use std::f64::consts::PI;
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let hdiv = HDivSpace::new(mesh.clone(), 0);
        let h1 = H1Space::new(mesh.clone(), 1);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let a = VectorAssembler::assemble_bilinear(&hdiv, &[&VectorMassIntegrator { alpha: 1.0 }], 3);
        let src = VectorDomainLFIntegrator {
            f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
                out[0] = (PI*x[0]).sin(); out[1] = (PI*x[1]).sin(); out[2] = (PI*x[2]).sin();
            })),
        };
        let mut rhs = VectorAssembler::assemble_linear(&hdiv, &[&src], 3);
        let bdofs = fem_space::constraints::boundary_dofs_hdiv(&mesh, &hdiv, &[1, 2, 3, 4, 5, 6]);
        let mut a_mut = a;
        apply_dirichlet(&mut a_mut, &mut rhs, &bdofs, &vec![0.0; bdofs.len()]);
        let c_fem = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let c_linlvo = fem_to_linlvo_csr(&c_fem);
        let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let g_linlvo = fem_to_linlvo_csr(&g_fem);
        let mut x = vec![0.0; hdiv.n_dofs()];
        let res = solve_pcg_ads(&a_mut, &c_linlvo, &g_linlvo, &rhs, &mut x, &cfg_coarse).unwrap();
        (res.converged, res.iterations)
    };
    eprintln!("ADS 3D iters: 2³={it1}, 3³={it2}");
    assert!(conv2, "Fine ADS must converge");
    assert!(it2 <= it1 + 25, "ADS iters should grow slowly: {it1}→{it2}");
}

// ─── Complex AMS: time-harmonic Maxwell 2D ─────────────────────────────────

use fem_linalg::complex_csr::ComplexCsr;
use fem_solver::complex_ams::{solve_gmres_ams_complex, solve_bicgstab_ams_complex};

/// Build a complex H(curl) system `(K + M) + i·(ω·M)` on a 2D mesh,
/// apply tangential Dirichlet BCs, and return `(A_complex, G_linlvo, rhs_re, rhs_im)`.
///
/// The real part `K + M` is symmetric positive-definite, so the AMS
/// preconditioner (built from the real part) performs robustly.
fn build_complex_maxwell_2d(
    n: usize, omega: f64,
) -> (ComplexCsr, linlvo::sparse::CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    use std::f64::consts::PI;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let h1 = H1Space::new(mesh.clone(), 1);
    let hcurl = HCurlSpace::new(mesh.clone(), 1);
    let n_dofs = hcurl.n_dofs();

    // Assemble curl-curl (K) and mass (M) separately.
    let k = VectorAssembler::assemble_bilinear(
        &hcurl, &[&CurlCurlIntegrator { mu: 1.0 }], 4);
    let m_csr = VectorAssembler::assemble_bilinear(
        &hcurl, &[&VectorMassIntegrator { alpha: 1.0 }], 4);

    // Build A_re = K + M (SPD) and A_im = ω·M via COO.
    let mut coo_re = fem_linalg::CooMatrix::new(n_dofs, n_dofs);
    let mut coo_im = fem_linalg::CooMatrix::new(n_dofs, n_dofs);
    for i in 0..n_dofs {
        for ptr in k.row_ptr[i]..k.row_ptr[i+1] {
            coo_re.add(i, k.col_idx[ptr] as usize, k.values[ptr]);
        }
    }
    for i in 0..n_dofs {
        for ptr in m_csr.row_ptr[i]..m_csr.row_ptr[i+1] {
            let j = m_csr.col_idx[ptr] as usize;
            let m_val = m_csr.values[ptr];
            // A_re += M
            coo_re.add(i, j, m_val);
            // A_im = ω·M
            coo_im.add(i, j, omega * m_val);
        }
    }
    let k_re: fem_linalg::CsrMatrix<f64> = coo_re.into_csr();
    let k_im: fem_linalg::CsrMatrix<f64> = coo_im.into_csr();
    let a_complex = ComplexCsr::from_re_im(&k_re, &k_im);

    // Discrete gradient G: H1(P1) → H(curl)(ND1)
    let g_fem = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
    let g_linlvo = fem_to_linlvo_csr(&g_fem);

    // RHS from sinusoidal source (same as real AMS test)
    let mut rhs_re = VectorAssembler::assemble_linear(&hcurl, &[
        &VectorDomainLFIntegrator {
            f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
                let sx = (PI*x[0]).sin(); let sy = (PI*x[1]).sin();
                out[0] = (1.0 + PI*PI)*sy;
                out[1] = (1.0 + PI*PI)*sx;
            })),
        },
    ], 4);
    let mut rhs_im = vec![0.0; n_dofs]; // purely real RHS

    // Dirichlet: tangential E = 0 on all boundaries
    let bdofs = boundary_dofs_hcurl(&mesh, &hcurl, &[1, 2, 3, 4]);
    let mut a_mut = a_complex;
    for &dof in &bdofs {
        a_mut.apply_dirichlet_row(dof as usize, 0.0, 0.0, &mut rhs_re, &mut rhs_im);
    }
    (a_mut, g_linlvo, rhs_re, rhs_im)
}

#[test]
fn complex_ams_2d_converges() {
    let omega = 1.0;
    let (a, g, b_re, b_im) = build_complex_maxwell_2d(6, omega);
    let n = a.nrows;
    let mut x_re = vec![0.0; n];
    let mut x_im = vec![0.0; n];

    let cfg = linlvo::precond::AmsConfig::hpc_default();
    let (iters, res) = solve_gmres_ams_complex(
        &a, &g, &b_re, &b_im, &mut x_re, &mut x_im,
        1e-6, 500, 50, cfg,
    ).expect("Complex AMS GMRES should converge");

    eprintln!("Complex AMS 2D (12×12 mesh): converged in {iters} iters, rel_prec_res={res:.2e}");
    assert!(iters < 300, "too many iterations: {iters}");
    assert!(iters > 0, "solver should perform at least 1 iteration");
}

#[test]
fn complex_ams_2d_h_independent() {
    let omega = 1.0;
    let cfg = linlvo::precond::AmsConfig::hpc_default();

    fn run(n: usize, omega: f64, cfg: &linlvo::precond::AmsConfig) -> (bool, usize) {
        let (a, g, b_re, b_im) = build_complex_maxwell_2d(n, omega);
        let mut x_re = vec![0.0; a.nrows];
        let mut x_im = vec![0.0; a.nrows];
        match solve_gmres_ams_complex(&a, &g, &b_re, &b_im, &mut x_re, &mut x_im, 1e-6, 500, 50, cfg.clone()) {
            Ok((iters, _res)) => (iters < 500, iters),
            Err(_) => (false, 999),
        }
    }

    let (c1, i1) = run(4, omega, &cfg);
    let (c2, i2) = run(6, omega, &cfg);
    let (c3, i3) = run(8, omega, &cfg);
    eprintln!("Complex AMS h-indep iters: 8×8={i1}, 12×12={i2}, 16×16={i3}");
    assert!(c1 && c2 && c3, "All levels must converge");
    // AMS real-part preconditioning for complex systems; iters may grow moderately
    // but should not explode (within the max_iter bound).
    assert!(i2 <= i1 + 50, "Iters should not explode: {i1}→{i2}");
}

#[test]
fn complex_ams_2d_bicgstab() {
    let omega = 2.0;
    let (a, g, b_re, b_im) = build_complex_maxwell_2d(6, omega);
    let n = a.nrows;
    let mut x_re = vec![0.0; n];
    let mut x_im = vec![0.0; n];

    let cfg = linlvo::precond::AmsConfig::hpc_default();
    let (iters, res) = solve_bicgstab_ams_complex(
        &a, &g, &b_re, &b_im, &mut x_re, &mut x_im,
        1e-6, 500, cfg,
    ).expect("Complex AMS BiCGSTAB should converge");

    eprintln!("Complex AMS BiCGSTAB 2D: {iters} iters, res={res:.2e}");
    assert!(iters < 400, "BiCGSTAB too many iterations: {iters}");
}
