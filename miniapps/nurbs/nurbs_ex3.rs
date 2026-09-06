//! Miniapp: NURBS Example 3 — Electromagnetic Diffusion (H(curl)).
//! 1:1 port of MFEM nurbs_ex3.cpp. curl curl E + E = f.

use std::f64::consts::PI;
use fem_assembly::{
    Assembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorFEDomainLFIntegrator},
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_space::{HCurlSpace, fe_space::FESpace};
use fem_solver::GSSmoother;
use fem_linalg::fem_to_linlvo_csr;

fn exact_e(x: &[f64], kappa: f64, dim: usize) -> Vec<f64> {
    if dim == 3 { vec![(kappa * x[1]).sin(), (kappa * x[2]).sin(), (kappa * x[0]).sin()] }
    else { let mut e = vec![(kappa * x[1]).sin(), (kappa * x[0]).sin()]; if x.len() == 3 { e.push(0.0); } e }
}

fn exact_f(x: &[f64], kappa: f64, dim: usize) -> Vec<f64> {
    let k2 = 1.0 + kappa * kappa;
    if dim == 3 { vec![k2 * (kappa * x[1]).sin(), k2 * (kappa * x[2]).sin(), k2 * (kappa * x[0]).sin()] }
    else { let mut f = vec![k2 * (kappa * x[1]).sin(), k2 * (kappa * x[0]).sin()]; if x.len() == 3 { f.push(0.0); } f }
}

struct Args { mesh: String, order: i32, ref_levels: i32 }

fn parse_args() -> Args {
    let mut a = Args { mesh: "data/square-nurbs.mesh".to_string(), order: 1, ref_levels: -1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next().unwrap_or(a.mesh); }
            "-o" | "--order" => { a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1); }
            "-r" | "--refine" => { a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1); }
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let kappa = PI;
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let dim = mfem.dim as usize;
    let mesh = if dim == 2 { mfem.mesh2d.expect("2D mesh expected") } else { mfem.mesh3d.expect("3D mesh expected") };

    let ne = mesh.n_elems() as f64;
    let ref_levels = if args.ref_levels < 0 { ((50000.0_f64 / ne).ln() / 2.0_f64.ln() / dim as f64).floor() as i32 } else { args.ref_levels };
    let mesh = if ref_levels > 0 { let mut m = mesh; for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); } m } else { mesh };

    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    println!("Number of finite element unknowns: {}", space.n_dofs());

    let qo = (args.order as u8) * 2 + 1;
    let f_coeff = |x: &[f64]| exact_f(x, kappa, dim);

    let mut b = fem_assembly::LinearForm::new(&space);
    b.add_domain_integrator(VectorFEDomainLFIntegrator::new(f_coeff));
    b.assemble();
    let b_vec = b.to_vec();

    let a_mat = Assembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], qo);

    let (a_mod, b_mod, dof_map) = a_mat.apply_dirichlet_bc(&[], &b_vec);
    let a_linlvo = fem_to_linlvo_csr(&a_mod);
    let gs = GSSmoother::from_csr(&a_linlvo).expect("GS failed");
    let mut x = vec![0.0_f64; b_mod.len()];
    let params = linlvo::SolverParams { rtol: 1e-12, atol: 1e-12, max_iter: 500, verbose: linlvo::VerboseLevel::Iterations, check_interval: 1 };
    let mut solver = linlvo::CgSolver::new(&a_linlvo, &params);
    solver.solve(&mut x, &b_mod).expect("CG failed");

    let x_full = fem_linalg::recover_dirichlet_solution(&x, &dof_map, space.n_dofs());
    let gf = GridFunction::new(&space, x_full.clone());
    let e_coeff = |x: &[f64]| exact_e(x, kappa, dim);
    let err = gf.compute_l2_error(&e_coeff, (2 * args.order as u8 + 2).max(3));
    println!("\n|| E_h - E ||_{{L^2}} = {}", err);

    write_mfem_file("refined.mesh", space.mesh()).ok();
    write_mfem_gf_file("sol.gf", dim, &x_full, "ND", args.order as u8, 1, 8).ok();
}
