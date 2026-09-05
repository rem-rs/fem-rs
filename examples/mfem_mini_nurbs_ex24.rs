//! Miniapp: NURBS Example 24 — Mixed Formulation with NURBS.
//! 1:1 port of MFEM nurbs_ex24.cpp.

use fem_assembly::{
    mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator},
    standard::VectorMassIntegrator,
    vector_assembler::VectorAssembler,
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::CsrMatrix;
use fem_solver::block::BlockSystem;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_solver::GSSmoother;
use fem_linalg::fem_to_linlvo_csr;

struct Args { mesh: String, order: i32, ref_levels: i32 }

fn parse_args() -> Args {
    let mut a = Args { mesh: "data/pipe-nurbs-2d.mesh".to_string(), order: 2, ref_levels: -1 };
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
    let mesh = mfem.mesh2d.expect("2D mesh expected");
    let dim = 2usize;

    let ne = mesh.n_elems() as f64;
    let ref_levels = if args.ref_levels < 0 { ((5000.0_f64 / ne).ln() / 2.0_f64.ln() / dim as f64).floor() as i32 } else { args.ref_levels };
    let mesh = if ref_levels > 0 { let mut m = mesh; for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); } m } else { mesh };

    let u_sp = HDivSpace::new(mesh.clone(), args.order as u8);
    let p_sp = L2Space::new(mesh.clone(), (args.order - 1).max(0) as u8);
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    println!("dim(R) = {}\ndim(W) = {}\ndim(R+W) = {}", n_u, n_p, n_u + n_p);

    let qo = (args.order as u8) * 2 + 1;
    let m_mat = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator { alpha: 1.0 }], qo);

    let b_mat = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);

    let rhs = vec![0.0_f64; n_u];

    let bt = b_mat.transpose();
    let zero_l2 = CsrMatrix::<f64>::new_empty(n_p, n_p);
    let flat = BlockSystem { a: m_mat.clone(), bt: bt.clone(), b: b_mat.clone(), c: Some(zero_l2) }.to_flat_csr();

    let n = n_u + n_p;
    let mut x = vec![0.0_f64; n];
    let mut rhs_full = vec![0.0_f64; n];
    rhs_full[..n_u].copy_from_slice(&rhs);

    let flat_linlvo = fem_to_linlvo_csr(&flat);
    let params = linlvo::SolverParams { rtol: 1e-10, atol: 1e-10, max_iter: 10000, verbose: linlvo::VerboseLevel::Iterations, check_interval: 1 };
    let mut solver = linlvo::MinresSolver::new(&flat_linlvo, &params);
    solver.solve(&mut x, &rhs_full).expect("MINRES failed");

    let u_sol = x[..n_u].to_vec();
    let p_sol = x[n_u..].to_vec();

    write_mfem_file("exsol.mesh", u_sp.mesh()).ok();
    write_mfem_gf_file("sol_u.gf", dim, &u_sol, "RT", args.order as u8, 1, 8).ok();
    write_mfem_gf_file("sol_p.gf", 1, &p_sol, "L2", (args.order - 1).max(0) as u8, 1, 8).ok();

    let u_gf = GridFunction::new(&u_sp, u_sol);
    let p_gf = GridFunction::new(&p_sp, p_sol);
    println!("|| u_h - u_ex ||  = {}", u_gf.compute_l2_error(&|_| 0.0, (2 * args.order as u8 + 2).max(3)));
    println!("|| p_h - p_ex ||  = {}", p_gf.compute_l2_error(&|_| 0.0, (2 * args.order as u8 + 2).max(3)));
}
