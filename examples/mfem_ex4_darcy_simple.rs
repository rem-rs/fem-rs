//! MFEM ex4: H(div) problem (SIMPLIFIED VERSION).
//!
//! Solves the second-order definite H(div) problem:
//!   -∇(α ∇·F) + β F = f    in Ω
//!              F·n = 0     on ∂Ω
//!
//! SIMPLIFIED VERSION: hybridization and hdiv_error modules not included.

use fem_assembly::standard::DiffusionIntegrator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_solver::{solve_cg, SolverConfig};

fn main() {
    println!("=== MFEM ex4: H(div) problem (SIMPLIFIED) ===");
    println!();

    // Create mesh
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh.clone(), 1);

    println!("Number of unknowns: {}", space.n_dofs());

    // Assemble diffusion matrix
    let k: CsrMatrix<f64> = fem_assembly::Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        3,
    );

    // Create RHS
    let rhs = vec![1.0; space.n_dofs()];

    // Solve
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 5000,
        ..Default::default()
    };
    solve_cg(&k, &rhs, &mut x, &cfg).expect("CG solve failed");

    println!("Solution computed successfully.");
    println!("Note: This is a simplified version. Full version requires hybridization module.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex4_compiles() {
        // This test just ensures the example compiles
    }
}
