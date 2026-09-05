//! Miniapp: NURBS Solenoidal — Project solenoidal velocity field.
//! 1:1 port of MFEM nurbs_solenoidal.cpp.

use fem_assembly::{
    mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator},
    standard::VectorMassIntegrator,
    vector_assembler::VectorAssembler,
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::block::BlockSystem;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_solver::GSSmoother;
use fem_linalg::fem_to_linlvo_csr;

fn exact_velocity_2d(x: &[f64]) -> [f64; 2] {
    let p = 4.0_f64;
    [x[0].powf(p + 1.0) * x[1].powf(p), -x[0].powf(p) * x[1].powf(p + 1.0)]
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
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh = mfem.mesh2d.expect("2D mesh expected");
    let dim = 2usize;

    let ne = mesh.n_elems() as f64;
    let ref_levels = if args.ref_levels < 0 { ((5000.0_f64 / ne).ln() / 2.0_f64.ln() / dim as f64).floor() as i32 } else { args.ref_levels };
    let mesh = if ref_levels > 0 { let mut m = mesh; for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); } m } else { mesh };

    let u_sp = HDivSpace::new(mesh.clone(), args.order as u8);
    let p_sp = L2Space::new(mesh.clone(), args.order as u8);
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    println!("dim(R) = {}\ndim(W) = {}\ndim(R+W) = {}", n_u, n_p, n_u + n_p);

    let qo = (args.order as u8) * 2 + 1;
    let m_mat = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator { alpha: 1.0 }], qo);

    let mut b_mat = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut b_mat.values { *v *= -1.0; }

    let rhs = { let mut r = vec![0.0_f64; n_u]; let u_ex_proj = u_sp.interpolate_vector(&|x| exact_velocity_2d(x).to_vec()); m_mat.spmv(&u_ex_proj, &mut r); r };

    let bt = b_mat.transpose();
    let zero_l2 = CsrMatrix::<f64>::new_empty(n_p, n_p);
    let flat = BlockSystem { a: m_mat.clone(), bt: bt.clone(), b: b_mat.clone(), c: Some(zero_l2) }.to_flat_csr();

    let n = n_u + n_p;
    let mut x = vec![0.0_f64; n];
    let mut rhs_full = vec![0.0_f64; n];
    rhs_full[..n_u].copy_from_slice(&rhs);

    let diag_m: Vec<f64> = (0..n_u).map(|i| m_mat.get(i, i).max(1e-30)).collect();
    let bt_t = b_mat.transpose();
    let mut minvbt = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / diag_m[i];
        for ptr in bt_t.row_ptr[i]..bt_t.row_ptr[i+1] {
            let j = bt_t.col_idx[ptr] as usize;
            minvbt.add(i, j, bt_t.values[ptr] * inv_d);
        }
    }
    let s_mat = b_mat.multiply(&minvbt.into_csr());

    let m_linlvo = fem_to_linlvo_csr(&m_mat);
    let s_linlvo = fem_to_linlvo_csr(&s_mat);
    let _m_gs = GSSmoother::from_csr(&m_linlvo).expect("GS(M) failed");
    let _s_gs = GSSmoother::from_csr(&s_linlvo).expect("GS(S) failed");
    let flat_linlvo = fem_to_linlvo_csr(&flat);

    let params = linlvo::SolverParams { rtol: 1e-10, atol: 1e-10, max_iter: 10000, verbose: linlvo::VerboseLevel::Iterations, check_interval: 1 };
    let mut solver = linlvo::MinresSolver::new(&flat_linlvo, &params);
    solver.solve(&mut x, &rhs_full).expect("MINRES failed");

    let u_sol = x[..n_u].to_vec();
    let p_sol = x[n_u..].to_vec();

    write_mfem_file("exsol.mesh", u_sp.mesh()).ok();
    write_mfem_gf_file("sol_u.gf", dim, &u_sol, "RT", args.order as u8, 1, 8).ok();
    write_mfem_gf_file("sol_p.gf", 1, &p_sol, "L2", args.order as u8, 1, 8).ok();

    let u_gf = GridFunction::new(&u_sp, u_sol);
    let p_gf = GridFunction::new(&p_sp, p_sol);
    println!("|| u_h - u_ex ||  = {}", u_gf.compute_l2_error(&|_| 0.0, (2 * args.order as u8 + 2).max(3)));
    println!("|| p_h - p_ex ||  = {}", p_gf.compute_l2_error(&|_| 0.0, (2 * args.order as u8 + 2).max(3)));
}
