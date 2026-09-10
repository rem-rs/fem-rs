//! Miniapp: NURBS Example 5 — Navier-Stokes with NURBS.
//! 1:1 port of MFEM nurbs_ex5.cpp.
//!
//! Port note (round K): this file used the removed
//! `CsrMatrix::apply_dirichlet_bc(&[], &b)` / `fem_linalg::recover_dirichlet_solution`
//! pair (an **empty** essential-DOF set) and `ConvectionIntegrator::new(&u, dim)`,
//! neither of which exists in the current API.  The Picard loop, the source
//! term (zero), the viscosity `nu = 0.01`, the solver tolerances and the
//! printed lines are unchanged.  The convective velocity is supplied as a
//! `ConstantVectorCoeff` because fem-rs has no `VectorCoeff` backed by a DOF
//! vector (see the round report); the frozen velocity it represents is exactly
//! the (all-zero) iterate this loop starts from and never leaves, since the
//! source is zero and no essential DOFs are imposed.

use fem_assembly::{
    Assembler,
    standard::{ConvectionIntegrator, DiffusionIntegrator, DomainSourceIntegrator},
    postproc::coefficient::ConstantVectorCoeff,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_mesh::{MeshTopology, amr::{refine_uniform, refine_uniform_3d}};
use fem_space::{H1Space, fe_space::FESpace, constraints::form_linear_system};
use fem_solver::{GSSmoother, solve_pcg};
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
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");

    // `MfemFile` has no `dim` field: the dimension is selected by which of the
    // two optional meshes was parsed.
    if let Some(mesh) = mfem.mesh2d {
        let dim = 2usize;
        let mut m = mesh;
        for _ in 0..auto_ref_levels(m.n_elems(), dim, args.ref_levels) {
            m = refine_uniform(&m);
        }
        run(H1Space::new(m, args.order as u8), dim, &args, &|mm| {
            write_mfem_file("refined.mesh", mm).ok();
        });
    } else if let Some(mesh) = mfem.mesh3d {
        let dim = 3usize;
        let mut m = mesh;
        for _ in 0..auto_ref_levels(m.n_elems(), dim, args.ref_levels) {
            m = refine_uniform_3d(&m);
        }
        run(H1Space::new(m, args.order as u8), dim, &args, &|mm| {
            write_mfem_file_3d("refined.mesh", mm).ok();
        });
    } else {
        panic!("mesh file contains neither a 2D nor a 3D mesh");
    }
}

/// Dimension-independent driver (the 2-D and 3-D paths differ only in the mesh
/// refinement / mesh writer, which `main` has already dispatched on).
fn run<M: MeshTopology>(space: H1Space<M>, dim: usize, args: &Args, write_mesh: &dyn Fn(&M)) {
    let qo = (args.order as u8) * 2 + 1;
    let nu = 0.01;
    println!("Number of finite element unknowns: {}", space.n_dofs());

    let mut u = vec![0.0_f64; space.n_dofs()];
    for iter in 0..10 {
        // Picard linearisation: the convective velocity is frozen at the
        // previous iterate (zero here — see the module-level port note).
        let a_mat = Assembler::assemble_bilinear(&space, &[
            &DiffusionIntegrator { kappa: nu },
            &ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![0.0; dim]) },
        ], qo);

        let source = DomainSourceIntegrator::new(|_: &[f64]| 0.0);
        let mut rhs = Assembler::assemble_linear(&space, &[&source], qo);

        // MFEM `FormLinearSystem(ess_tdof_list, x, b, A, X, B)` — with the
        // empty essential list this file has always used, it is a no-op.
        let mut a_mod = a_mat;
        let mut x = vec![0.0_f64; u.len()];
        form_linear_system(&mut a_mod, &mut rhs, &mut x, &[], &[]);

        let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mod)).expect("GS failed");
        solve_pcg(&a_mod, &rhs, &mut x, &gs, 1e-10, 1000, true).expect("PCG failed");

        u = x;
        println!("Picard iteration {} done", iter);
    }

    println!("Solution computed");

    write_mesh(space.mesh());
    write_mfem_gf_file("sol.gf", dim, &u, "H1", args.order as u8, 1, 8).ok();
}
