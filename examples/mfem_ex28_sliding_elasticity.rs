//! # Example 28 — Elastic contact with Coulomb friction (2D Signorini)
//!
//! A deformable block pressed against a rigid obstacle, demonstrating:
//! - Normal contact with penalty / Augmented Lagrangian regularisation
//! - Coulomb friction (stick–slip) on the contact interface
//! - Nonlinear Newton solver for the coupled system
//!
//! This example aligns with MFEM's contact miniapp and ex36 (obstacle problem).
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex28_sliding_elasticity --release
//! cargo run --example mfem_ex28_sliding_elasticity -- --mu 0.3 --penalty 1e6
//! ```

use fem_assembly::contact::*;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_space::fe_space::FESpace;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 28: Elastic contact with Coulomb friction ===");

    // 1. Build mesh: a block [0,1]×[0,1] that will be pressed onto an obstacle
    let n_refine = args.n_refine;
    let n_el = 2usize.pow(n_refine as u32);
    let mesh = SimplexMesh::<2>::unit_square_tri(n_el);
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // 2. Vector H¹ space (2 DOFs per node)
    use fem_space::VectorH1Space;
    let space = VectorH1Space::new(mesh, 1, 2);
    let mesh_ref = space.mesh();
    let dim = 2usize;
    let n_dofs = space.n_dofs();
    println!("  DOFs: {}", n_dofs);

    // 3. Assemble stiffness matrix (linear elasticity via vector Laplacian)
    use fem_assembly::Assembler;
    use fem_assembly::standard::DiffusionIntegrator;
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    // 4. Body force (downward) and BCs
    let mut rhs = vec![0.0_f64; n_dofs];
    for i in 0..mesh_ref.n_nodes() {
        rhs[i * dim + 1] = -args.body_force / mesh_ref.n_nodes() as f64;
    }

    // Fix top boundary (tag=3 on unit square): set ux=uy=0
    let mut is_fixed = vec![false; n_dofs];
    for f in 0..mesh_ref.n_boundary_faces() as u32 {
        if mesh_ref.face_tag(f) == 3 {
            for &n in mesh_ref.face_nodes(f) {
                for c in 0..dim { is_fixed[n as usize * dim + c] = true; }
            }
        }
    }
    let mut stiff_fixed = stiff;
    for dof in 0..n_dofs {
        if is_fixed[dof] {
            let row_s = stiff_fixed.row_ptr[dof] as usize;
            let row_e = stiff_fixed.row_ptr[dof + 1] as usize;
            // Find and zero the row, keep track of diagonal
            let mut diag_idx = None;
            for nz in row_s..row_e {
                if stiff_fixed.col_idx[nz] as usize == dof {
                    diag_idx = Some(nz);
                }
                stiff_fixed.values[nz] = 0.0;
            }
            if let Some(di) = diag_idx {
                stiff_fixed.values[di] = 1.0;
            }
            rhs[dof] = 0.0;
        }
    }

    // 5. Configure contact on the bottom boundary (attribute 1)
    //    Obstacle is a flat surface at y = -0.02 (slight interference)
    let gap_fn: fn(&[f64]) -> f64 = |x: &[f64]| -0.02 - x[1];
    let contact_cfg = ContactConfig {
        penalty_normal: args.penalty,
        contact_type: if args.al_iters > 1 {
            ContactType::AugmentedLagrangian { max_al_iter: args.al_iters, al_tol: 1e-6 }
        } else {
            ContactType::Penalty
        },
        friction: if args.mu > 0.0 {
            FrictionModel::Coulomb { mu: args.mu, penalty_tangential: args.penalty * 0.1 }
        } else {
            FrictionModel::Frictionless
        },
        gap_function: gap_fn,
        contact_tags: vec![1],
    };

    println!("  Contact: penalty={:.1e}, mu={}, AL-iter={}",
             args.penalty, args.mu, args.al_iters);

    // 6. Solve the contact problem with Newton
    let u = solve_contact_newton(
        &stiff_fixed, &rhs, mesh_ref, &contact_cfg, 50, 1e-8,
    );

    // 7. Post-process
    let max_uy = u.iter().skip(1).step_by(2).cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_uy = u.iter().skip(1).step_by(2).cloned().fold(f64::INFINITY, f64::min);
    println!("  u_y range: [{:.6e}, {:.6e}]", min_uy, max_uy);

    // 8. Verify no penetration on the contact boundary
    let mut max_penetration = 0.0_f64;
    for f in 0..mesh_ref.n_boundary_faces() as u32 {
        if mesh_ref.face_tag(f) == 1 {
            for &n in mesh_ref.face_nodes(f) {
                let ni = n as usize;
                let uy = u[ni * 2 + 1];
                let x = mesh_ref.node_coords(n);
                let gap = (gap_fn)(&x);
                let penetration = (uy - gap).max(0.0);
                if penetration > max_penetration {
                    max_penetration = penetration;
                }
            }
        }
    }
    println!("  Max penetration: {:.6e} (target < 1e-4)", max_penetration);

    // Check if friction is active — look at horizontal displacement of bottom nodes
    if args.mu > 0.0 {
        let mut max_ux_bottom = 0.0_f64;
        for f in 0..mesh_ref.n_boundary_faces() as u32 {
            if mesh_ref.face_tag(f) == 1 {
                for &n in mesh_ref.face_nodes(f) {
                    let ni = n as usize;
                    let ux = u[ni * 2].abs();
                    if ux > max_ux_bottom { max_ux_bottom = ux; }
                }
            }
        }
        println!("  Max horizontal slip (bottom): {:.6e}", max_ux_bottom);
    }

    println!("Done.");
}

struct Args { penalty: f64, mu: f64, al_iters: usize, body_force: f64, n_refine: u32 }

fn parse_args() -> Args {
    let mut a = Args { penalty: 1e6, mu: 0.3, al_iters: 0, body_force: -1.0, n_refine: 3 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--penalty" => { a.penalty = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e6); }
            "--mu" => { a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3); }
            "--al-iters" => { a.al_iters = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
            "--body-force" => { a.body_force = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0); }
            "--n-refine" => { a.n_refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(3); }
            _ => {}
        }
    }
    a
}
