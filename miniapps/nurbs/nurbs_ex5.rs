//! Miniapp: NURBS Example 5 — Navier-Stokes with NURBS.
//! 1:1 port of MFEM nurbs_ex5.cpp.

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, ConvectionIntegrator},
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_space::{H1Space, fe_space::FESpace};
use fem_solver::GSSmoother;
use fem_linalg::fem_to_linlvo_csr;

struct Args { mesh: String, order: i32, ref_levels: i32 }

fn parse_args() -> Args {
    let mut a = Args { mesh: "data/square-nurbs.mesh".to_string(), order: 2, ref_levels: -1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next().unwrap_or(a.mesh); }
            "-o" | "--order" => { a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2); }
            "-r" | "--refine" => { a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1); }
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let dim = mfem.dim as usize;
    let mesh = if dim == 2 { mfem.mesh2d.expect("2D mesh expected") } else { mfem.mesh3d.expect("3D mesh expected") };

    let ne = mesh.n_elems() as f64;
    let ref_levels = if args.ref_levels < 0 { ((50000.0_f64 / ne).ln() / 2.0_f64.ln() / dim as f64).floor() as i32 } else { args.ref_levels };
    let mesh = if ref_levels > 0 { let mut m = mesh; for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); } m } else { mesh };

    let space = H1Space::new(mesh.clone(), args.order as u8);
    println!("Number of finite element unknowns: {}", space.n_dofs());

    let qo = (args.order as u8) * 2 + 1;
    let nu = 0.01;

    let mut u = vec![0.0_f64; space.n_dofs()];
    for iter in 0..10 {
        let a_mat = Assembler::assemble_bilinear(&space, &[
            &DiffusionIntegrator { kappa: nu },
            &ConvectionIntegrator::new(&u, dim),
        ], qo);

        let mut b = fem_assembly::LinearForm::new(&space);
        b.add_domain_integrator(fem_assembly::standard::DomainSourceIntegrator::new(|_x| 0.0));
        b.assemble();
        let b_vec = b.to_vec();

        let (a_mod, b_mod, dof_map) = a_mat.apply_dirichlet_bc(&[], &b_vec);
        let a_linlvo = fem_to_linlvo_csr(&a_mod);
        let gs = GSSmoother::from_csr(&a_linlvo).expect("GS failed");
        let mut x = vec![0.0_f64; b_mod.len()];
        let params = linlvo::SolverParams { rtol: 1e-10, atol: 1e-10, max_iter: 1000, verbose: linlvo::VerboseLevel::Iterations, check_interval: 1 };
        let mut solver = linlvo::CgSolver::new(&a_linlvo, &params);
        solver.solve(&mut x, &b_mod).expect("CG failed");

        u = fem_linalg::recover_dirichlet_solution(&x, &dof_map, space.n_dofs());
        println!("Picard iteration {} done", iter);
    }

    let gf = GridFunction::new(&space, u.clone());
    println!("Solution computed");

    write_mfem_file("refined.mesh", space.mesh()).ok();
    write_mfem_gf_file("sol.gf", dim, &u, "H1", args.order as u8, 1, 8).ok();
}
