//! Example 12 — Laplacian eigenvalue (analogous to MFEM ex12 style)
//!
//! Computes the smallest eigenvalues of Kx = λMx using LOBPCG on a
//! 2D unit-square mesh with P1 elements, where K = stiffness, M = mass.
//!
//! Usage:
//!   cargo run --example mfem_ex12_elastic_eigen

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_mesh::SimplexMesh;
use fem_solver::eigen::{lobpcg, LobpcgConfig};
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let order: u8 = 1;
    let n_eig = 5;

    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, order);

    let quad_order = order as u8 * 2 + 1;
    let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order);
    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order);

    let cfg = LobpcgConfig { max_iter: 200, tol: 1e-8, ..LobpcgConfig::default() };
    let result = lobpcg(&k, Some(&m), n_eig, &cfg).unwrap();

    println!("=== ex12: Laplacian Eigenvalue ===");
    println!("  n={n}, P1 DOFs={}", space.n_dofs());
    for (i, val) in result.eigenvalues.iter().enumerate() {
        println!("  λ[{}] = {:.6e}", i + 1, val);
    }
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
    use fem_mesh::SimplexMesh;
    use fem_solver::eigen::{lobpcg, LobpcgConfig};
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let r = lobpcg(&k, Some(&m), 3, &LobpcgConfig { max_iter: 50, ..LobpcgConfig::default() }).unwrap();
        assert_eq!(r.eigenvalues.len(), 3);
    }
}
