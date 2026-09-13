//! # LOR Solvers Miniapp — **declared stub** (round 31, D140)
//!
//! ⚠ This file is **not** a port of MFEM `miniapps/solvers/lor_solvers.cpp`.
//! It never builds a low-order-refined (LOR) space and never uses an LOR/AMG
//! preconditioner: it assembles the H¹ mass and diffusion matrices, **throws
//! the mass matrix away** (the `_mass` binding below), solves that
//! diffusion-only system with plain PCG + Jacobi and prints
//! `LOR solvers complete` with **exit code 0** — i.e. it claims to be the LOR
//! miniapp while computing the wrong operator.  It imported no LOR/AMG symbol
//! at all, `-fe` other than `h` exited with 1, and there was no gap list.
//!
//! `main` therefore prints the gap list and exits with status **3** for every
//! input.  The 1:1 miniapp of the same MFEM source is
//! `miniapps/solvers/plor_solvers.rs` (registered as `miniapp_plor_solvers`),
//! which has the LOR space, the LOR preconditioner and the AMG comparison —
//! **use that one**; this file is kept only as scaffolding and deliberately
//! does not solve anything.
//!
//! Nothing here is worth wiring up as-is: the missing pieces for a real port
//! are the LOR space (`LORFECollection` / `LORSolver` in C++), the
//! `-fe {h,n,r}` coarse-space selection, the LOR space's own sparse
//! representation, and the `M + K` operator combination (fem-rs has LOR
//! machinery — see `crates/assembly/src/lor_factory.rs` and
//! `miniapps/solvers/plor_solvers.rs` — so the port belongs there, not here).

use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_mesh::Mesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::{apply_dirichlet, boundary_dofs};
use fem_solver::{solve_pcg_jacobi, SolverConfig};

/// D140: prints the gap list and terminates with the project's "honest partial
/// delivery" status.  Deliberately not `-> !` so the scaffolding below stays
/// type-checked while this guard is the only path that executes.
fn not_ported() {
    eprintln!(
        "miniapp_lor_solvers: NOT PORTED — declared stub (D140); this file is not a port of \
         lor_solvers.cpp.\n\
         \x20 It assembles M and K, discards M (the `_mass` binding), solves K with plain PCG+Jacobi \
         and used to print\n\
         \x20 \"LOR solvers complete\" with exit code 0: no LOR space, no LOR preconditioner, no \
         AMG comparison, no -fe handling.\n\
         \x20 Use miniapps/solvers/plor_solvers.rs (example `miniapp_plor_solvers`), which is the \
         1:1 miniapp of the same source.\n\
         \x20 A real port needs the LOR space/`LORSolver`, the `-fe {{h,n,r}}` coarse-space \
         selection and the M + K combination.\n\
         Exiting with status 3."
    );
    std::process::exit(3);
}

fn main() {
    // D140: refuse before parsing, so `-m/-r/-o/-fe` are never accepted and
    // then silently ignored.
    not_ported();
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/star.mesh".to_string();
    let mut ref_levels = 1usize;
    let mut order = 3u8;
    let mut fe = "h".to_string();

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { if let Some(v) = it.next() { mesh_file = v.clone(); } }
            "-r" | "--refine" => { if let Some(v) = it.next() { ref_levels = v.parse().unwrap_or(1); } }
            "-o" | "--order" => { if let Some(v) = it.next() { order = v.parse().unwrap_or(3); } }
            "-fe" | "--fe-type" => { if let Some(v) = it.next() { fe = v.clone(); } }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    if fe != "h" {
        eprintln!(
            "miniapp_lor_solvers: -fe {fe} (non-H1 coarse space) is not ported — declared gap, \
             exiting with status 3"
        );
        std::process::exit(3);
    }

    // Read mesh (C++: Mesh(mesh_file, 1, 1))
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| {
            eprintln!("miniapp_lor_solvers: failed to read mesh {mesh_file}: {e} — exiting with status 3");
            std::process::exit(3)
        });

    // Use 2D mesh (simplified)
    let mut mesh: Mesh<2> = if let Some(m2) = mfem.mesh2d { m2 }
    else {
        eprintln!("miniapp_lor_solvers: expected a 2-D mesh in {mesh_file} — exiting with status 3");
        std::process::exit(3)
    };

    for _ in 0..ref_levels {
        mesh = match mesh.elem_type {
            fem_mesh::element_type::ElementType::Tri3 => fem_mesh::amr::refine_uniform(&mesh),
            fem_mesh::element_type::ElementType::Quad4 => fem_mesh::amr::refine_uniform(&mesh),
            _ => { eprintln!("Uniform refinement not supported for {:?}; stopping", mesh.elem_type); break; }
        };
    }

    // Create H1 FE space
    let space = H1Space::new(mesh.clone(), order);
    println!("Number of DOFs: {}", space.n_dofs());

    // Essential BCs
    let mut ess_dofs = Vec::new();
    let mut bnd_vals = Vec::new();
    let bdr_tags = mesh.unique_boundary_tags();
    if !bdr_tags.is_empty() {
        for tag in &bdr_tags {
            let tag_dofs = boundary_dofs(&mesh, space.dof_manager(), &[*tag]);
            ess_dofs.extend(tag_dofs.clone());
            bnd_vals.extend(std::iter::repeat(0.0).take(tag_dofs.len()));
        }
    }

    let qo = (2 * order + 1) as u8;

    // Assemble A = M + K (mass + diffusion) for definite Helmholtz u - Delta u = f.
    // ⚠ The mass matrix is **not** combined — see the header gap list (D140); the
    // operator actually solved below is K alone.  Unreachable behind
    // `not_ported()`, kept only as scaffolding.
    let _mass = fem_assembly::Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: 1.0 }], qo);
    let mut a_ho = fem_assembly::Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);

    // RHS: b = ∫ 1 * v dx
    let src = |_x: &[f64]| 1.0f64;
    let mut b_vec = fem_assembly::Assembler::assemble_linear(
        &space, &[&fem_assembly::standard::DomainSourceIntegrator::new(src)], qo);

    // Apply Dirichlet BCs
    apply_dirichlet(&mut a_ho, &mut b_vec, &ess_dofs, &bnd_vals);

    let mut x = vec![0.0f64; space.n_dofs()];
    let b_slice = b_vec.as_slice().to_vec();

    let cfg = SolverConfig {
        rtol: 1e-12,
        max_iter: 500,
        verbose: true,
        ..SolverConfig::default()
    };

    match solve_pcg_jacobi(&a_ho, &b_slice, &mut x, &cfg) {
        Ok(res) => {
            println!("PCG converged in {} iterations", res.iterations);
            let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("|x| = {x_norm:.6e}");
        }
        Err(e) => eprintln!("Solver failed: {e:?}"),
    }

    println!("LOR solvers complete (simplified H1 2D version).");
}
