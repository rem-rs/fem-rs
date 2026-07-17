//! # Example 26 — Geometric Multigrid for Poisson  [1:1 translation of MFEM ex26]
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BCs using
//! a geometric multigrid preconditioner.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m data/fichera.mesh
//! ```

use std::fs::File;
use std::io::Write;

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    GeometricMgLevel, GeometricMgHierarchy, GeometricMgConfig, GeometricMgPrecond,
    GeometricMgAsPrecond, MgCycleType, MgSmootherType, solve_pcg,
    StoredElementOperator, PADiffusionOp, SumFactDiffusionOp,
};
use fem_space::{
    H1Space, fe_space::FESpace, constraints::boundary_dofs,
    build_h1_prolongation_matrix,
};
use fem_mesh::ElementType;

fn main() {
    let args = parse_args();
    let dim = 2;

    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("built-in"));
    println!("   --geometric-refinements {}", args.geometric_refs);
    println!("   --order-refinements {}", args.order_refs);
    println!("   --device cpu");
    println!("   --no-visualization");
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // 4. Uniform refinement onto coarse mesh.
    let coarse_mesh = {
        let ne = mesh.n_elements();
        let ref_levels = if ne > 0 {
            ((5000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        } else { 0 };
        let mut m = mesh;
        for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); }
        m
    };

    // 5. FE space hierarchy.
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

    // 6. RHS.
    let fine_space = spaces.last().unwrap();
    let n_dofs = fine_space.n_dofs();
    let mut rhs = Assembler::assemble_linear(fine_space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);

    // 7. Solution vector.
    let mut x = vec![0.0; n_dofs];

    // 8. Build MG hierarchy: per-level matrices with symmetric BC elimination.
    let boundary_tags: Vec<i32> = fine_space.mesh().unique_boundary_tags();
    // Zero RHS at BC DOFs (matching MFEM Multigrid::FormFineLinearSystem).
    let bc_fine = boundary_dofs(fine_space.mesh(), fine_space.dof_manager(), &boundary_tags);
    for &d in &bc_fine { rhs[d as usize] = 0.0; }

    let mut levels: Vec<GeometricMgLevel> = Vec::new();
    let mut prolong: Vec<fem_linalg::CsrMatrix<f64>> = Vec::new();
    for i in 0..n_spaces {
        let space = &spaces[i];
        let qo = (2 * space.order() + 1).max(3) as u8;
        // Assemble CSR + element matrices in one pass (same integration)
        let (mut mat, elem_dofs, elem_mats, ldofs, n_elems) =
            Assembler::assemble_bilinear_with_elements(
                space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        // Save raw diagonal before BC modification
        let raw_diag = mat.diagonal();
        let raw_dinv: Vec<f64> = raw_diag.iter()
            .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }).collect();
        // Apply symmetric BC elimination for the CSR matrix (PCG outer / coarse CG)
        let bc = boundary_dofs(space.mesh(), space.dof_manager(), &boundary_tags);
        let mut dummy = vec![0.0; mat.nrows];
        for &d in &bc { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy); }
        // Build element-by-element operator (raw matrices, no BC mods)
        let elem_op = StoredElementOperator {
            elem_dofs: elem_dofs.clone(), elem_mats: elem_mats.clone(),
            ldofs, n_elems, n_dofs: mat.nrows,
        };
        // Build on-the-fly PA operator (matches MFEM AddMultPA)
        let elem_dofs_clone = elem_dofs.clone();
        let pa_op = PADiffusionOp::build(
            space.mesh(), mat.nrows, space.order(), qo, 1.0,
            |e| {
                let e32 = e;
                let start = e32 as usize * ldofs;
                elem_dofs_clone[start..start + ldofs].to_vec()
            },
        );
        // Build sum-factorization PA operator (bitwise match to MFEM)
        let sf_op = if space.mesh().element_type(0) == ElementType::Quad4 {
            let e_dofs = elem_dofs.clone();
            Some(SumFactDiffusionOp::build(
                space.mesh(), mat.nrows, space.order(), qo, 1.0,
                |e| {
                    let start = e as usize * ldofs;
                    e_dofs[start..start + ldofs].to_vec()
                },
            ))
        } else {
            None
        };
        levels.push(GeometricMgLevel {
            mat, bc_dofs: bc,
            elem_op: Some(elem_op), raw_diag, raw_dinv,
            pa_op: Some(pa_op), sf_op,
        });
    }
    for i in 0..n_spaces - 1 {
        prolong.push(build_h1_prolongation_matrix(
            spaces[i].mesh(), spaces[i].dof_manager(),
            spaces[i + 1].mesh(), spaces[i + 1].dof_manager(),
        ));
    }

    levels.reverse();
    prolong.reverse();
    let hierarchy = GeometricMgHierarchy::new(levels, prolong);
    println!("Size of linear system: {}", hierarchy.finest_matrix().nrows);

    // 9. Solve with PCG + MG V(1,1)-cycle.
    let mg_config = GeometricMgConfig {
        pre_sweeps: 1, post_sweeps: 1,
        smoother: MgSmootherType::Chebyshev(2),
        max_eig_override: None,
        jacobi_omega: 0.8,
        coarse_max_iter: 200, coarse_rtol: 1e-8,
        cycle_type: MgCycleType::V,
    };
    let mg = GeometricMgPrecond::new(mg_config, &hierarchy);
    let precond = GeometricMgAsPrecond { mg: &mg, hierarchy: &hierarchy };

    if let Err(e) = solve_pcg(hierarchy.finest_matrix(), &rhs, &mut x, &precond, 1e-12, 2000, true) {
        eprintln!("PCG: No convergence! ({e})");
    }

    // 10. Save.
    {
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, fine_space.mesh(), None).expect("mesh write failed");
        let mut sol_f = File::create("sol.gf").expect("cannot create sol.gf");
        for &v in &x {
            writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
        }
    }
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
