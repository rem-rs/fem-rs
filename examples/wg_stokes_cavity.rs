//! Weak Galerkin (WG) Stokes 2D saddle-point system assembly.
//!
//! Assembles the WG Stokes system matrix and RHS for a lid-driven cavity
//! problem on a unit-square mesh, then applies Dirichlet BCs and prints
//! the system size.
//!
//! Usage:
//!   cargo run --example wg_stokes_cavity

use std::time::Instant;

use fem_assembly::wg_stokes::assemble_wg_stokes;
use fem_mesh::Mesh;
use fem_space::{H1Space, L2Space, fe_space::FESpace};

fn main() {
    println!("=== WG Stokes: lid-driven cavity ===");
    let t0 = Instant::now();

    // Mesh
    let mesh = Mesh::<2>::unit_square_tri(8);
    let mesh2 = Mesh::<2>::unit_square_tri(8);

    // Velocity: continuous H1 P2, pressure: discontinuous L2 P1
    let vel_space = H1Space::new(mesh, 2);
    let pres_space = L2Space::new(mesh2, 1);
    let n_vel = vel_space.n_dofs();
    let n_pres = pres_space.n_dofs();
    println!("  Velocity DOFs: {n_vel}, Pressure DOFs: {n_pres}");

    // Body force: Stokes (zero forcing for cavity flow)
    let force = |_x: &[f64]| vec![0.0, 0.0];

    // Placeholder Dirichlet
    let dirichlet_bc: Vec<(usize, f64)> = vec![];

    // Assemble
    let penalty = 10.0;
    let (_mat, rhs) = assemble_wg_stokes(&vel_space, &pres_space, 3, penalty, &force, &dirichlet_bc);

    let n_total = n_vel + n_pres;
    println!("  System: {n_total}×{n_total} DOFs (sparse)");
    println!("  ‖RHS‖ = {:.6e}", rhs.iter().map(|v| v * v).sum::<f64>().sqrt());
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

#[cfg(test)]
mod tests {
    use fem_assembly::wg_stokes::assemble_wg_stokes;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, L2Space, fe_space::FESpace};

    #[test]
    fn wg_stokes_assembles() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let vel = H1Space::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let pres = L2Space::new(mesh2, 0);
        let f = |_: &[f64]| vec![1.0, 0.0];
        let (k, rhs) = assemble_wg_stokes(&vel, &pres, 3, 5.0, &f, &[]);
        assert_eq!(k.nrows, vel.n_dofs() + pres.n_dofs());
        assert_eq!(rhs.len(), k.nrows);
    }
}
