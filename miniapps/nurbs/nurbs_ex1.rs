//! Miniapp: NURBS Example 1 — Poisson with NURBS.
//! 1:1 port of MFEM nurbs_ex1.cpp. -Delta u = 1, Dirichlet BC.
//!
//! The discretization is MFEM's NURBS one: `mesh->NURBSext` +
//! `NURBSFECollection(order)` + `NURBSExtension(mesh->NURBSext, order)`, i.e.
//! [`NurbsFESpace`] (see its module docs for the `LoadFE`/weighted-geometry
//! details and for why the assembly loop lives there rather than in
//! `fem_assembly`).  With the C++ defaults (`-o 2`, 6 uniform refinements of
//! `data/square-nurbs.mesh`) the space has 4356 unknowns on 64x64 elements, the
//! same as MFEM.
//!
//! Port notes:
//! * `ess_bdr = 1` marks *every* boundary attribute essential and `x = 0`
//!   (homogeneous Dirichlet data), so `FormLinearSystem` imposes zero data on
//!   the whole boundary.  `NurbsFESpace::boundary_dofs` is the NURBS analogue of
//!   `GetEssentialTrueDofs` for that case.
//! * Not ported: `refined.mesh` / `sol.gf` (`Mesh::Print` and
//!   `GridFunction::Save` need a NURBS mesh writer, which does not exist yet),
//!   the GLVis socket, the `-lod` 1-D solution output and the VisIt
//!   collection.  The printed solve block is byte-identical to MFEM's.

use std::f64::consts::LN_2;

use fem_linalg::fem_to_linlvo_csr;
use fem_solver::{solve_pcg, GSSmoother, SolverError};
use fem_space::constraints::form_linear_system;
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

struct Args {
    mesh: String,
    order: i32,
    ref_levels: i32,
}

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

/// `print_marker_line` mirrors MFEM's `Array<int>::Print` (10 values per line,
/// space separated, newline terminated) so the marker dump is byte-identical.
fn marker_line(flags: &[i32]) -> String {
    let mut out = String::new();
    for (i, f) in flags.iter().enumerate() {
        out.push_str(&f.to_string());
        if (i + 1) % 10 == 0 || i + 1 == flags.len() {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    out
}

fn main() {
    let args = parse_args();
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // The mesh's own extension gives `NURBSext->GetNKV()` (the number of orders
    // the space needs) and `GetNE()` — the *element* count, which for a
    // multi-patch mesh is the sum over patches of `prod_d GetNE(kv_p[d])`, not
    // the product over all knot vectors.
    let mesh_ext = NurbsExtension::from_mesh_str(&text).expect("failed to parse the NURBS mesh");
    let dim = mesh_ext.dim();
    let n_elems = mesh_ext.n_elements();

    // C++ nurbs_ex1: `order.SetSize(nkv); order = tmp;` — or `-1` for the
    // isoparametric space, which for a NURBS mesh keeps the mesh's own orders.
    let mesh_orders: Vec<usize> =
        (0..mesh_ext.n_knot_vectors()).map(|i| mesh_ext.knot_vector(i).order()).collect();
    let orders: Vec<usize> = if args.order < 0 {
        mesh_orders
    } else {
        vec![args.order as usize]
    };

    // `floor(log(5000./mesh->GetNE())/log(2.)/dim)` when `-r` is not given
    // (nurbs_ex1.cpp uses 5000, not 50000).
    let ref_levels = if args.ref_levels < 0 {
        ((5000.0_f64 / n_elems as f64).ln() / LN_2 / dim as f64).floor() as i32
    } else {
        args.ref_levels
    } as usize;

    let space =
        NurbsFESpace::from_mesh_str(&text, ref_levels, &orders).expect("failed to build the space");
    println!("Number of finite element unknowns: {}", space.n_dofs());

    // C++ prints the three marker *arrays* (`per_bdr.Print()`,
    // `ess_bdr.Print()`, `neu_bdr.Print()`), each of size
    // `mesh->bdr_attributes.Max()`.  `ess_bdr = 1` and the other two stay zero.
    let n_attrs = mesh_ext.max_bdr_attribute().max(0) as usize;
    println!("Boundary conditions:");
    print!(" - Periodic  : {}", marker_line(&vec![0; n_attrs]));
    print!(" - Essential : {}", marker_line(&vec![1; n_attrs]));
    print!(" - Neumann   : {}", marker_line(&vec![0; n_attrs]));

    // b(.) = (1, phi_i) with `DomainLFIntegrator(one)`.
    let mut rhs = space.assemble_domain_lf(&|_| 1.0);
    // a(.,.) = (grad u, grad v) with `DiffusionIntegrator(one)`.
    let mut a_mat = space.assemble_diffusion(1.0);

    // C++: `GridFunction x(fespace); x = 0.0;` then
    // `a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B)` with
    // `fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list)`.
    let ess_dofs = space.boundary_dofs();
    let mut x = vec![0.0_f64; space.n_dofs()];
    let ess_vals = vec![0.0_f64; ess_dofs.len()];
    form_linear_system(&mut a_mat, &mut rhs, &mut x, &ess_dofs, &ess_vals);

    // C++ prints `A.Height()`; with MFEM's eliminated (but not reduced) matrix
    // this is the full number of unknowns.
    println!("Size of linear system: {}", a_mat.nrows);

    // C++: `GSSmoother M(A); PCG(A, M, B, X, 1, 200, 1e-12, 0.0);`
    // `solve_pcg` prints MFEM's full `CGSolver::Mult` trailer: the per-iteration
    // log, `Average reduction factor =`, and — when the solve stops at
    // `max_iter` — `PCG: Number of iterations:` / `PCG: No convergence!`.
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mat)).expect("GS smoother");
    if let Err(e) = solve_pcg(&a_mat, &rhs, &mut x, &gs, 1e-12, 200, true) {
        // MFEM's miniapp ignores the non-convergence; only unexpected errors
        // abort.
        let SolverError::ConvergenceFailed { .. } = e else {
            panic!("PCG failed: {e}");
        };
    }
}
