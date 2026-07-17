//! # Example 26 — Geometric Multigrid for Poisson  [1:1 translation of MFEM ex26]
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BCs using
//! a geometric multigrid preconditioner.
//!
//! Demonstrates a hierarchy of H¹ discretisation spaces: P1 on the (auto-refined)
//! coarse mesh, `gr` uniform geometric refinement levels, then `or` order
//! refinements (orders 2, 4, …, 2^or) on the finest mesh. All levels use
//! Chebyshev(2) smoothing with a CG solver on the coarsest level, and the
//! multigrid V(1,1)-cycle preconditioners an outer PCG — exactly as MFEM's
//! `DiffusionMultigrid` in `examples/ex26.cpp`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m data/fichera.mesh  # (2D meshes only)
//! ```
//!
//! ## Output
//! Prints DOF count, linear system size, PCG iteration history and average
//! reduction factor (same format as MFEM). Writes `refined.mesh` and `sol.gf`.

use std::fs::File;
use std::io::Write;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    GeometricMgLevel, GeometricMgHierarchy, GeometricMgConfig, GeometricMgPrecond,
    GeometricMgAsPrecond, MgCycleType, MgSmootherType, solve_pcg,
};
use fem_space::{
    H1Space, fe_space::FESpace, constraints::boundary_dofs,
    build_h1_prolongation_matrix,
};

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class).

    // 3. Read the mesh from the given mesh file.
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let dim = 2;

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("built-in"));
    println!("   --geometric-refinements {}", args.geometric_refs);
    println!("   --order-refinements {}", args.order_refs);
    println!("   --device cpu");
    println!("   --no-visualization");
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // 4. Uniform refinement: largest level count giving ≤ 5000 elements
    //    (matching the C++ code — the comment in ex26.cpp says 50,000, but
    //    the formula uses 5000).
    let coarse_mesh = {
        let ne = mesh.n_elements();
        let ref_levels = if ne > 0 {
            ((5000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        } else { 0 };
        let mut m = mesh;
        for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); }
        m
    };

    // 5. Finite element space hierarchy: start with P1 on the coarse mesh,
    //    add `gr` geometrically refined P1 levels, then `or` order-refined
    //    levels (order 2^k) on the finest mesh — as in ex26.cpp step 5.
    let mut meshes = vec![coarse_mesh];
    for _ in 0..args.geometric_refs {
        let fine = fem_mesh::refine_uniform(meshes.last().unwrap());
        meshes.push(fine);
    }

    let mut spaces: Vec<H1Space<Mesh<2>>> = Vec::new();
    for m in &meshes {
        spaces.push(H1Space::new(m.clone(), 1));
    }
    let finest_mesh = meshes.last().unwrap().clone();
    for k in 1..=args.order_refs {
        spaces.push(H1Space::new(finest_mesh.clone(), 1u8 << k));
    }
    let n_spaces = spaces.len();

    println!("Number of finite element unknowns: {}", spaces.last().unwrap().n_dofs());

    // 6. RHS linear form (1, φ_i) on the finest space.
    let fine_space = spaces.last().unwrap();
    let n_dofs = fine_space.n_dofs();
    let mut rhs = Assembler::assemble_linear(fine_space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);

    // 7. Solution vector, initialised to zero (satisfies the homogeneous BCs).
    let mut x = vec![0.0; n_dofs];

    // 8. Multigrid operator: per-level stiffness matrices with symmetric
    //    essential-BC elimination (ess_bdr = all boundary attributes), plus
    //    nodal prolongation operators between consecutive levels.
    let boundary_tags: Vec<i32> = fine_space.mesh().unique_boundary_tags();
    {
        // Zero the RHS at essential DOFs (homogeneous Dirichlet, cf.
        // MFEM Multigrid::FormFineLinearSystem).
        let bc_fine = boundary_dofs(fine_space.mesh(), fine_space.dof_manager(), &boundary_tags);
        for &d in &bc_fine { rhs[d as usize] = 0.0; }
    }

    let mut levels: Vec<GeometricMgLevel> = Vec::new();
    let mut prolong: Vec<fem_linalg::CsrMatrix<f64>> = Vec::new();
    for i in 0..n_spaces {
        let space = &spaces[i];
        let qo = (2 * space.order() + 1).max(3) as u8;
        let mut mat = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let bc = boundary_dofs(space.mesh(), space.dof_manager(), &boundary_tags);
        let mut dummy = vec![0.0; mat.nrows];
        for &d in &bc { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy); }
        levels.push(GeometricMgLevel { mat, bc_dofs: bc });
    }
    for i in 0..n_spaces - 1 {
        // spaces[i] is coarser, spaces[i+1] is finer.
        prolong.push(build_h1_prolongation_matrix(
            spaces[i].mesh(), spaces[i].dof_manager(),
            spaces[i + 1].mesh(), spaces[i + 1].dof_manager(),
        ));
    }

    // GeometricMgHierarchy expects levels[0] = finest, prolong[l]: level l+1 → l.
    levels.reverse();
    prolong.reverse();
    let hierarchy = GeometricMgHierarchy::new(levels, prolong);
    println!("Size of linear system: {}", hierarchy.finest_matrix().nrows);

    // 9. Solve A X = B with PCG preconditioned by one V(1,1)-cycle
    //    (Chebyshev(2) smoothing, CG on the coarsest level with rtol 1e-2,
    //    max 200 iterations — matching DiffusionMultigrid in ex26.cpp).
    let mg_config = GeometricMgConfig {
        pre_sweeps: 1, post_sweeps: 1,
        smoother: MgSmootherType::Chebyshev(2),
        max_eig_override: None,
        jacobi_omega: 0.8,
        coarse_max_iter: 200, coarse_rtol: 1e-2,
        cycle_type: MgCycleType::V,
    };
    let mg = GeometricMgPrecond::new(mg_config, &hierarchy);
    let precond = GeometricMgAsPrecond { mg: &mg, hierarchy: &hierarchy };
    // MFEM: PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0) — stopping when
    // (B r, r) ≤ 1e-12 · (B r₀, r₀).
    if let Err(e) = solve_pcg(hierarchy.finest_matrix(), &rhs, &mut x, &precond, 1e-12, 2000, true) {
        eprintln!("PCG: No convergence! ({e})");
    }

    // 10. x already holds the finest-level grid function.

    // 11. Save the refined mesh and the solution.
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, fine_space.mesh(), None).expect("mesh write failed");
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &x {
            writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
        }
    }

    // 12. GLVis visualisation — not available in this port (-no-vis).
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    geometric_refs: usize,
    order_refs: usize,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 10, geometric_refs: 0, order_refs: 2 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|s| s.parse().ok()).unwrap_or(10),
            "-gr" | "--geometric-refinements" => {
                a.geometric_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(0)
            }
            "-or" | "--order-refinements" => {
                a.order_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(2)
            }
            _ => {}
        }
    }
    a
}
