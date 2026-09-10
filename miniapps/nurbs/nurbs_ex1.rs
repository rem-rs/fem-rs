//! Miniapp: NURBS Example 1 — Poisson with NURBS.
//! 1:1 port of MFEM nurbs_ex1.cpp. -Delta u = 1, Dirichlet BC.
//!
//! Port note (round O): the essential boundary condition is now the one of
//! the C++ miniapp — *every* boundary attribute is marked essential
//! (`ess_bdr = 1; GetEssentialTrueDofs(...)`) and the solution is initialized
//! to zero (`homogenousBC = true`), so `FormLinearSystem` imposes homogeneous
//! Dirichlet data on the whole boundary.  The earlier port passed an **empty**
//! essential list, which left the system singular with an incompatible
//! right-hand side (PCG diverged).
//!
//! Known remaining difference (D23): fem-rs discretises this NURBS mesh with
//! the H1 Lagrange space on the refined surface mesh, not with a NURBS
//! `NURBSFECollection` + `NURBSExtension`, so the number of unknowns (and
//! hence the iteration count) cannot match the C++ binary exactly.

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_mesh::{MeshTopology, amr::{refine_uniform, refine_uniform_3d}};
use fem_space::{H1Space, fe_space::FESpace, constraints::{boundary_dofs, form_linear_system}};
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

/// `ref_levels = floor(log(5000/NE)/log(2)/dim)` when not given explicitly
/// (C++ nurbs_ex1.cpp uses 5000, not 50000).
fn auto_ref_levels(n_elems: usize, dim: usize, requested: i32) -> i32 {
    if requested < 0 {
        ((5000.0_f64 / n_elems as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as i32
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
        // C++ marks every boundary attribute as essential (`ess_bdr = 1`).
        let tags = m.unique_boundary_tags();
        run(H1Space::new(m, args.order as u8), dim, &args, &tags, &|mm| {
            write_mfem_file("refined.mesh", mm).ok();
        });
    } else if let Some(mesh) = mfem.mesh3d {
        let dim = 3usize;
        let mut m = mesh;
        for _ in 0..auto_ref_levels(m.n_elems(), dim, args.ref_levels) {
            m = refine_uniform_3d(&m);
        }
        let tags = m.unique_boundary_tags();
        run(H1Space::new(m, args.order as u8), dim, &args, &tags, &|mm| {
            write_mfem_file_3d("refined.mesh", mm).ok();
        });
    } else {
        panic!("mesh file contains neither a 2D nor a 3D mesh");
    }
}

/// Dimension-independent driver (the 2-D and 3-D paths differ only in the mesh
/// refinement / mesh writer, which `main` has already dispatched on).
fn run<M: MeshTopology>(
    space: H1Space<M>,
    dim: usize,
    args: &Args,
    bdr_tags: &[i32],
    write_mesh: &dyn Fn(&M),
) {
    let qo = (args.order as u8) * 2 + 1;
    println!("Number of finite element unknowns: {}", space.n_dofs());

    let a_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);

    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], qo);

    // C++: ess_bdr = 1 (all boundary attributes) → GetEssentialTrueDofs.
    let ess_dofs = if bdr_tags.is_empty() {
        Vec::new()
    } else {
        boundary_dofs(space.mesh(), space.dof_manager(), bdr_tags)
    };
    println!("Boundary conditions:");
    println!(" - Essential : {}", ess_dofs.len());

    // C++: `GridFunction x(fespace); x = 0.0;` (homogenousBC = true), then
    // `a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B)` — homogeneous
    // Dirichlet data on the full boundary.
    let mut a_mod = a_mat;
    let mut x = vec![0.0_f64; space.n_dofs()];
    let ess_vals = vec![0.0_f64; ess_dofs.len()];
    form_linear_system(&mut a_mod, &mut rhs, &mut x, &ess_dofs, &ess_vals);

    // C++ prints `A.Height()`; with MFEM's eliminated (but not reduced) matrix
    // this is the full number of unknowns.
    println!("Size of linear system: {}", space.n_dofs());

    // C++: `GSSmoother M(A); PCG(A, M, B, X, 1, 200, 1e-12, 0.0);`
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mod)).expect("GS failed");
    solve_pcg(&a_mod, &rhs, &mut x, &gs, 1e-12, 200, true).expect("PCG failed");

    // C++ nurbs_ex1.cpp defines no exact solution and prints no error.
    // Since the essential data is u = 0, this norm is ||u_h||_{L^2} (not an
    // error); it is reported as a sanity check that the solve is finite.
    let gf = GridFunction::new(&space, x.clone());
    let l2_norm = gf.compute_l2_error(&|_: &[f64]| 0.0, (2 * args.order as u8 + 2).max(3));
    println!("\n|| u_h ||_{{L^2}} = {}", l2_norm);

    write_mesh(space.mesh());
    write_mfem_gf_file("sol.gf", dim, &x, "H1", args.order as u8, 1, 8).ok();
}
