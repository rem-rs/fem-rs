//! Monolithic Newton for a 2-field coupled linear system.
//!
//! Solves:
//!   F₀(u, v) = 2·u + 1·v − rhs₀ = 0    (field u)
//!   F₁(u, v) = 1·u + 3·v − rhs₁ = 0    (field v)
//!
//! The `CoupledProblem` trait defines block sizes, residual, and Jacobian.
//! `CoupledNewtonSolver` drives monolithic Newton with line-search.
//!
//! Usage:
//!   cargo run --example multiphysics_coupled_newton

use fem_linalg::{BlockMatrix, BlockVector, CooMatrix};
use fem_solver::{
    CoupledLinearStrategy, CoupledNewtonConfig,
    CoupledNewtonSolver, CoupledProblem, SolverConfig,
};

// A constant-coefficient 2×2 coupled linear system.
struct Linear2x2;

impl CoupledProblem for Linear2x2 {
    fn block_sizes(&self) -> &[usize] {
        &[1, 1] // one DOF per field
    }

    fn residual(
        &self,
        _t: f64,
        state: &BlockVector,
        rhs: &BlockVector,
        out: &mut BlockVector,
    ) {
        let u = state.block(0)[0];
        let v = state.block(1)[0];
        let b0 = rhs.block(0)[0];
        let b1 = rhs.block(1)[0];
        out.block_mut(0)[0] = 2.0 * u + 1.0 * v - b0;
        out.block_mut(1)[0] = 1.0 * u + 3.0 * v - b1;
    }

    fn jacobian(&self, _t: f64, _state: &BlockVector) -> BlockMatrix {
        let mut j = BlockMatrix::new_square(vec![1, 1]);
        let csr = |v: f64| {
            let mut coo = CooMatrix::new(1, 1);
            coo.add(0, 0, v);
            coo.into_csr()
        };
        j.set(0, 0, csr(2.0));
        j.set(0, 1, csr(1.0));
        j.set(1, 0, csr(1.0));
        j.set(1, 1, csr(3.0));
        j
    }
}

fn main() {
    println!("=== Monolithic coupled Newton (2-field linear system) ===");

    let problem = Linear2x2;
    let solver = CoupledNewtonSolver::new(CoupledNewtonConfig {
        atol: 1e-12,
        rtol: 1e-12,
        max_iter: 10,
        gmres_restart: 8,
        line_search: true,
        ..CoupledNewtonConfig::default()
    });

    // rhs = [1, 2]ᵀ
    let mut rhs = BlockVector::new(vec![1, 1]);
    rhs.block_mut(0)[0] = 1.0;
    rhs.block_mut(1)[0] = 2.0;

    let mut state = BlockVector::new(vec![1, 1]);
    let result = solver
        .solve(&problem, 0.0, &rhs, &mut state)
        .expect("CoupledNewton solve failed");

    println!("  Converged:  {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Residual:   {:.3e}", result.final_residual);
    println!("  Solution:");
    println!("    u = {:.12e}", state.block(0)[0]);
    println!("    v = {:.12e}", state.block(1)[0]);

    // Analytical: [2 1; 1 3]⁻¹·[1, 2] = [0.2, 0.6]ᵀ
    let u_exact = 0.2;
    let v_exact = 0.6;
    let err_u = (state.block(0)[0] - u_exact).abs();
    let err_v = (state.block(1)[0] - v_exact).abs();
    println!("  Error u:    {:.3e}", err_u);
    println!("  Error v:    {:.3e}", err_v);

    assert!(result.converged, "solver must converge");
    assert!(err_u < 1e-10, "u error too large: {err_u}");
    assert!(err_v < 1e-10, "v error too large: {err_v}");
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test() {
        let problem = Linear2x2;
        let solver = CoupledNewtonSolver::new(CoupledNewtonConfig {
            atol: 1e-12,
            rtol: 1e-12,
            max_iter: 10,
            gmres_restart: 8,
            line_search: false,
            linear_strategy: CoupledLinearStrategy::Gmres,
            ..CoupledNewtonConfig::default()
        });
        let mut rhs = BlockVector::new(vec![1, 1]);
        rhs.block_mut(0)[0] = 1.0;
        rhs.block_mut(1)[0] = 2.0;
        let mut state = BlockVector::new(vec![1, 1]);
        let r = solver.solve(&problem, 0.0, &rhs, &mut state).unwrap();
        assert!(r.converged);
    }

    #[test]
    fn schur_strategy() {
        let problem = Linear2x2;
        let solver = CoupledNewtonSolver::new(CoupledNewtonConfig {
            linear_strategy: CoupledLinearStrategy::BlockSchur2x2,
            ..CoupledNewtonConfig::default()
        });
        let mut rhs = BlockVector::new(vec![1, 1]);
        rhs.block_mut(0)[0] = 1.0;
        rhs.block_mut(1)[0] = 2.0;
        let mut state = BlockVector::new(vec![1, 1]);
        let r = solver.solve(&problem, 0.0, &rhs, &mut state).unwrap();
        assert!(r.converged);
    }
}
