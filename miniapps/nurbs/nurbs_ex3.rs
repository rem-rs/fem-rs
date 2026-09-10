//! Miniapp: NURBS Example 3 — Electromagnetic Diffusion (H(curl)).
//! 1:1 port of MFEM nurbs_ex3.cpp. curl curl E + E = f.
//!
//! Port note (round K): the removed `CsrMatrix::apply_dirichlet_bc(&[], &b)` /
//! `fem_linalg::recover_dirichlet_solution` pair imposed an **empty** set of
//! essential DOFs.  That semantics is kept verbatim below (MFEM
//! `FormLinearSystem` with an empty `ess_tdof_list`).  C++ nurbs_ex3.cpp marks
//! all boundary attributes essential and projects E_exact into the boundary
//! values; that difference is pre-existing and reported, not fixed here.

use std::f64::consts::PI;
use fem_assembly::{
    VectorAssembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator, VectorDomainLFIntegrator},
    postproc::{coefficient::FnVectorCoeff, grid_function::compute_l2_error_hcurl},
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_mesh::{MeshTopology, amr::{refine_uniform, refine_uniform_3d}};
use fem_space::{HCurlSpace, fe_space::FESpace, constraints::form_linear_system};
use fem_solver::{GSSmoother, solve_pcg};
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

/// `ref_levels = floor(log(50000/NE)/log(2)/dim)` when not given explicitly.
fn auto_ref_levels(n_elems: usize, dim: usize, requested: i32) -> i32 {
    if requested < 0 {
        ((50000.0_f64 / n_elems as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as i32
    } else {
        requested
    }
}

fn main() {
    let args = parse_args();
    let kappa = PI;
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");

    // `MfemFile` has no `dim` field: the dimension is selected by which of the
    // two optional meshes was parsed.
    if let Some(mesh) = mfem.mesh2d {
        let dim = 2usize;
        let mut m = mesh;
        for _ in 0..auto_ref_levels(m.n_elems(), dim, args.ref_levels) {
            m = refine_uniform(&m);
        }
        run(HCurlSpace::new(m, args.order as u8), dim, kappa, &args, &|mm| {
            write_mfem_file("refined.mesh", mm).ok();
        });
    } else if let Some(mesh) = mfem.mesh3d {
        let dim = 3usize;
        let mut m = mesh;
        for _ in 0..auto_ref_levels(m.n_elems(), dim, args.ref_levels) {
            m = refine_uniform_3d(&m);
        }
        run(HCurlSpace::new(m, args.order as u8), dim, kappa, &args, &|mm| {
            write_mfem_file_3d("refined.mesh", mm).ok();
        });
    } else {
        panic!("mesh file contains neither a 2D nor a 3D mesh");
    }
}

/// Dimension-independent driver (the 2-D and 3-D paths differ only in the mesh
/// refinement / mesh writer, which `main` has already dispatched on).
fn run<M: MeshTopology>(
    space: HCurlSpace<M>,
    dim: usize,
    kappa: f64,
    args: &Args,
    write_mesh: &dyn Fn(&M),
) {
    let qo = (args.order as u8) * 2 + 1;
    println!("Number of finite element unknowns: {}", space.n_dofs());

    // b(.) = (f, phi_i) with the vector source f_exact.
    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
            let fv = exact_f(x, kappa, dim);
            out.copy_from_slice(&fv[..out.len()]);
        }),
    };
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&src], qo);

    // a(.,.) = (curl E, curl v) + (E, v).
    let a_mat = VectorAssembler::assemble_bilinear(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );

    // MFEM `FormLinearSystem(ess_tdof_list, x, b, A, X, B)` — with the empty
    // essential list this file has always used, it is a no-op.
    let mut a_mod = a_mat;
    let mut x = vec![0.0_f64; space.n_dofs()];
    form_linear_system(&mut a_mod, &mut rhs, &mut x, &[], &[]);

    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mod)).expect("GS failed");
    solve_pcg(&a_mod, &rhs, &mut x, &gs, 1e-12, 500, true).expect("PCG failed");

    let e_coeff = |x: &[f64]| exact_e(x, kappa, dim);
    let err = compute_l2_error_hcurl(&x, &space, &e_coeff, (2 * args.order as u8 + 2).max(3), None);
    println!("\n|| E_h - E ||_{{L^2}} = {}", err);

    write_mesh(space.mesh());
    write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order as u8, 1, 8).ok();
}
