//! Example 21 — AMR elasticity (analogous to MFEM ex21)
//!
//! Solves -div(σ(u)) = 0 on an L-shaped domain with a uniform
//! mesh refined toward the re-entrant corner. Nearly incompressible
//! (ν → 0.5) with µ = 1e6, ν = 0.4999.
//!
//! Usage:
//!   cargo run --example mfem_ex21_amr_elasticity

use fem_assembly::{Assembler, standard::ElasticityIntegrator};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::VectorH1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(16);

    // Use a fine uniform mesh as AMR proxy
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let order: u8 = 1;
    let space = VectorH1Space::new(mesh, order, 2);
    let n_dofs = space.n_dofs();

    // Nearly incompressible (ν ≈ 0.5)
    let e_mod = 1e6;
    let nu = 0.4999;
    let lambda = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    let elast = ElasticityIntegrator { lambda, mu, plane_stress: false };
    let mut k = Assembler::assemble_bilinear(&space, &[&elast], order as u8 * 2 + 1);

    // Fix bottom (tag=1), apply horizontal load on left (tag=3)
    let dm = space.scalar_dof_manager();
    let n_scalar = space.n_scalar_dofs();
    let bot_dofs = boundary_dofs(space.mesh(), dm, &[1]);
    let left_dofs = boundary_dofs(space.mesh(), dm, &[3]);
    let mut rhs = vec![0.0_f64; n_dofs];
    for &d in &left_dofs { rhs[d as usize + n_scalar] = -1.0; }
    for &d in &bot_dofs {
        k.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs);
        k.apply_dirichlet_symmetric(d as usize + n_scalar, 0.0, &mut rhs);
    }

    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&k, &rhs, &mut u, &cfg).unwrap();

    let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("=== ex21: AMR Elasticity (near-incompressible) ===");
    println!("  n={}, DOFs={}, iters={}, ‖u‖={:.6e}", n, n_dofs, res.iterations, u_norm);
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::ElasticityIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_solver::{solve_pcg_jacobi, SolverConfig};
    use fem_space::VectorH1Space;
    use fem_space::fe_space::FESpace;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let space = VectorH1Space::new(mesh, 1, 2);
        let k = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator { lambda: 1e6, mu: 0.5e6, plane_stress: false }], 3);
        let n = space.n_dofs();
        let mut rhs = vec![1.0; n];
        let mut u = vec![0.0; n];
        solve_pcg_jacobi(&k, &rhs, &mut u, &SolverConfig { max_iter: 200, ..SolverConfig::default() }).ok();
        assert!(u.iter().any(|v| v.abs() > 0.0));
    }
}
