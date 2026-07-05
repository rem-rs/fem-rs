//! Example 14 — DG Poisson (analogous to MFEM ex14)
//!
//! Solves -Δu = f using SIP-DG on a triangular mesh.
//!
//! Usage:
//!   cargo run --example mfem_ex14_dg_poisson

use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(8);
    let order: u8 = args.iter().position(|a| a == "--order").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let sigma: f64 = match order { 1 => 4.0, 2 => 10.0, _ => 24.0 };

    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = L2Space::new(mesh, order);
    let ifl = InteriorFaceList::build(space.mesh());

    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        let p = std::f64::consts::PI;
        2.0 * p * p * (p * x[0]).sin() * (p * x[1]).sin()
    });
    let rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);
    let a_mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, sigma, order * 2 + 1);

    let mut x = vec![0.0_f64; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let result = solve_gmres(&a_mat, &rhs, &mut x, 30, &cfg).unwrap();

    let sol_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("=== ex14: DG Poisson ===");
    println!("  n={}, P{} DOFs={}, iters={}, ‖u‖={:.6e}",
             n, order, space.n_dofs(), result.iterations, sol_norm);
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_solver::solve_gmres;
    use fem_solver::SolverConfig;
    use fem_space::L2Space;
    use fem_space::fe_space::FESpace;

    fn solve_dg_poisson(n: usize, order: u8) -> f64 {
        let sigma: f64 = match order { 1 => 4.0, 2 => 10.0, _ => 24.0 };
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = L2Space::new(mesh, order);
        let ifl = InteriorFaceList::build(space.mesh());

        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            let p = std::f64::consts::PI;
            2.0 * p * p * (p * x[0]).sin() * (p * x[1]).sin()
        });
        let rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);
        let a_mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, sigma, order * 2 + 1);

        let mut x = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        solve_gmres(&a_mat, &rhs, &mut x, 30, &cfg).unwrap();
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);
        let a = DgAssembler::assemble_sip(&space, &ifl, 1.0, 4.0, 3);
        let mut x = vec![0.0; space.n_dofs()];
        let r = solve_gmres(&a, &rhs, &mut x, 30, &SolverConfig { max_iter: 500, ..SolverConfig::default() }).unwrap();
        assert!(r.iterations > 0);
    }

    #[test]
    fn regression_solution_norm() {
        let norm = solve_dg_poisson(8, 1);
        fem_regression::regression("mfem_ex14_dg_poisson")
            .check("sol_norm_n8_p1", norm)
            .finalize();
    }
}
