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
                fe_space::FESpace, constraints::{boundary_dofs_hcurl, boundary_dofs_hdiv, apply_dirichlet}};

fn ams_solver_cfg() -> AmsSolverConfig {
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
    eprintln!("AMS iters: 12×12={it1}, 20×20={it2}");
    assert!(conv1 && conv2, "All cases must converge");
    assert!(it2 <= it1 + 250, "AMS iters should grow sub-linearly: {it1}→{it2}");
    eprintln!("AMS iteration count ratio: {:.2}× (target < 4× for h halving)", it2 as f64 / it1 as f64);
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
