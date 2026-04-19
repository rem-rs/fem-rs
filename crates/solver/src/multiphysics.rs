//! Multiphysics coupling abstractions and monolithic solve utilities.
//!
//! This module provides a minimal, reusable interface for coupled systems with
//! multiple fields and a monolithic Newton solver that reuses the existing
//! linear solvers in `fem-solver`.

use fem_linalg::{BlockMatrix, BlockVector, CooMatrix, CsrMatrix};
use thiserror::Error;

use crate::{
    solve_gmres,
    BlockSystem,
    SchurComplementSolver,
    SolveResult,
    SolverConfig,
    SolverError,
};

/// A generic coupled multiphysics problem in block form.
///
/// The state is partitioned into blocks (fields), e.g. `[u, p, T]`.
/// Implementors provide residual/Jacobian assembly in the same block layout.
pub trait CoupledProblem: Send + Sync {
    /// Field block sizes in the global state vector.
    fn block_sizes(&self) -> &[usize];

    /// Assemble residual `F(x, t) - rhs` into `out`.
    fn residual(&self, t: f64, state: &BlockVector, rhs: &BlockVector, out: &mut BlockVector);

    /// Assemble Jacobian `J(x, t) = dF/dx` in block form.
    fn jacobian(&self, t: f64, state: &BlockVector) -> BlockMatrix;

    /// Optional hook to enforce algebraic constraints after each Newton update.
    fn apply_constraints(&self, _t: f64, _state: &mut BlockVector) {}
}

/// Newton convergence and linearization parameters for monolithic coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoupledLinearStrategy {
    /// Flatten block Jacobian and solve with GMRES.
    #[default]
    Gmres,
    /// For 2x2 systems, solve using `BlockSystem + SchurComplementSolver`.
    BlockSchur2x2,
}

/// Newton convergence and linearization parameters for monolithic coupling.
#[derive(Debug, Clone)]
pub struct CoupledNewtonConfig {
    /// Absolute tolerance on `||R||_2`.
    pub atol: f64,
    /// Relative tolerance on `||R_k||_2 / ||R_0||_2`.
    pub rtol: f64,
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// Restart for inner GMRES solve.
    pub gmres_restart: usize,
    /// Enable backtracking line-search on Newton updates.
    pub line_search: bool,
    /// Minimum step size in line-search.
    pub line_search_min_alpha: f64,
    /// Multiplicative shrink factor used during backtracking (0, 1).
    pub line_search_shrink: f64,
    /// Maximum number of backtracking reductions per Newton iteration.
    pub line_search_max_backtracks: usize,
    /// Sufficient residual decrease factor for Armijo-like acceptance.
    pub line_search_sufficient_decrease: f64,
    /// Inner linear solve configuration.
    pub linear: SolverConfig,
    /// Linear strategy for each Newton correction.
    pub linear_strategy: CoupledLinearStrategy,
}

impl Default for CoupledNewtonConfig {
    fn default() -> Self {
        Self {
            atol: 1e-10,
            rtol: 1e-8,
            max_iter: 50,
            gmres_restart: 40,
            line_search: true,
            line_search_min_alpha: 1e-6,
            line_search_shrink: 0.5,
            line_search_max_backtracks: 20,
            line_search_sufficient_decrease: 1e-4,
            linear: SolverConfig {
                rtol: 1e-10,
                atol: 0.0,
                max_iter: 2000,
                verbose: false,
                print_level: crate::PrintLevel::Silent,
            },
            linear_strategy: CoupledLinearStrategy::Gmres,
        }
    }
}

/// Outcome of monolithic coupled Newton solve.
#[derive(Debug, Clone)]
pub struct CoupledNewtonResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub last_linear_result: Option<SolveResult>,
}

/// Errors for coupled Newton solves.
#[derive(Debug, Error)]
pub enum CoupledSolveError {
    #[error("invalid block layout: {0}")]
    InvalidLayout(String),
    #[error("linear solve failed: {0}")]
    Linear(#[from] SolverError),
    #[error("coupled Newton did not converge in {max_iter} iterations (residual = {residual:.3e})")]
    NonConverged { max_iter: usize, residual: f64 },
}

/// Monolithic Newton solver for coupled multiphysics systems.
pub struct CoupledNewtonSolver {
    cfg: CoupledNewtonConfig,
}

impl CoupledNewtonSolver {
    pub fn new(cfg: CoupledNewtonConfig) -> Self { Self { cfg } }

    /// Solve `F(x,t) = rhs` using monolithic Newton iterations.
    ///
    /// On success, `state` is updated in place to the converged coupled state.
    pub fn solve<P: CoupledProblem>(
        &self,
        problem: &P,
        t: f64,
        rhs: &BlockVector,
        state: &mut BlockVector,
    ) -> Result<CoupledNewtonResult, CoupledSolveError> {
        validate_layout(problem.block_sizes(), rhs, "rhs")?;
        validate_layout(problem.block_sizes(), state, "state")?;

        let sizes = problem.block_sizes().to_vec();
        let mut residual = BlockVector::new(sizes.clone());
        let mut trial_state = BlockVector::new(sizes.clone());
        let mut trial_residual = BlockVector::new(sizes.clone());

        problem.residual(t, state, rhs, &mut residual);
        let r0 = norm2(residual.as_slice());
        if r0 <= self.cfg.atol {
            return Ok(CoupledNewtonResult {
                converged: true,
                iterations: 0,
                final_residual: r0,
                last_linear_result: None,
            });
        }

        let mut rnorm = r0;

        for iter in 0..self.cfg.max_iter {
            let jac_block = problem.jacobian(t, state);
            validate_jacobian_layout(problem.block_sizes(), &jac_block)?;
            let neg_r: Vec<f64> = residual.as_slice().iter().map(|&v| -v).collect();
            let (dx, lin) = self.solve_linearized(problem.block_sizes(), &jac_block, &neg_r)?;

            if self.cfg.line_search {
                let mut alpha = 1.0_f64;
                let mut accepted = false;
                let mut best_norm = f64::INFINITY;
                let mut best_alpha = 1.0_f64;

                for _ in 0..=self.cfg.line_search_max_backtracks {
                    for (ti, (si, dxi)) in trial_state
                        .as_slice_mut()
                        .iter_mut()
                        .zip(state.as_slice().iter().zip(dx.iter()))
                    {
                        *ti = *si + alpha * *dxi;
                    }
                    problem.apply_constraints(t, &mut trial_state);
                    problem.residual(t, &trial_state, rhs, &mut trial_residual);
                    let trial_norm = norm2(trial_residual.as_slice());
                    if trial_norm < best_norm {
                        best_norm = trial_norm;
                        best_alpha = alpha;
                    }

                    let target = ((1.0 - self.cfg.line_search_sufficient_decrease * alpha).max(0.0)) * rnorm;
                    if trial_norm <= target || trial_norm < rnorm {
                        accepted = true;
                        break;
                    }

                    if alpha <= self.cfg.line_search_min_alpha {
                        break;
                    }
                    alpha *= self.cfg.line_search_shrink;
                }

                let use_alpha = if accepted { alpha } else { best_alpha };
                for (si, dxi) in state.as_slice_mut().iter_mut().zip(dx.iter()) {
                    *si += use_alpha * *dxi;
                }
                problem.apply_constraints(t, state);
            } else {
                for (si, dxi) in state.as_slice_mut().iter_mut().zip(dx.iter()) {
                    *si += *dxi;
                }
                problem.apply_constraints(t, state);
            }

            problem.residual(t, state, rhs, &mut residual);
            rnorm = norm2(residual.as_slice());
            if rnorm <= self.cfg.atol || rnorm <= self.cfg.rtol * r0 {
                return Ok(CoupledNewtonResult {
                    converged: true,
                    iterations: iter + 1,
                    final_residual: rnorm,
                    last_linear_result: Some(lin),
                });
            }
        }

        Err(CoupledSolveError::NonConverged {
            max_iter: self.cfg.max_iter,
            residual: rnorm,
        })
    }

    fn solve_linearized(
        &self,
        sizes: &[usize],
        jac_block: &BlockMatrix,
        rhs: &[f64],
    ) -> Result<(Vec<f64>, SolveResult), CoupledSolveError> {
        match self.cfg.linear_strategy {
            CoupledLinearStrategy::Gmres => {
                let jac = block_matrix_to_csr(jac_block);
                let mut dx = vec![0.0_f64; rhs.len()];
                let lin = solve_gmres(
                    &jac,
                    rhs,
                    &mut dx,
                    self.cfg.gmres_restart,
                    &self.cfg.linear,
                )?;
                Ok((dx, lin))
            }
            CoupledLinearStrategy::BlockSchur2x2 => {
                if sizes.len() != 2 {
                    return Err(CoupledSolveError::InvalidLayout(
                        "BlockSchur2x2 requires exactly 2 blocks".to_string(),
                    ));
                }

                let a = jac_block.get(0, 0).cloned().ok_or_else(|| {
                    CoupledSolveError::InvalidLayout("missing Jacobian block (0,0)".to_string())
                })?;
                let bt = jac_block.get(0, 1).cloned().ok_or_else(|| {
                    CoupledSolveError::InvalidLayout("missing Jacobian block (0,1)".to_string())
                })?;
                let b = jac_block.get(1, 0).cloned().ok_or_else(|| {
                    CoupledSolveError::InvalidLayout("missing Jacobian block (1,0)".to_string())
                })?;
                let c = jac_block.get(1, 1).cloned();

                let sys = BlockSystem { a, bt, b, c };
                let n0 = sizes[0];
                let n1 = sizes[1];
                let f = &rhs[..n0];
                let g = &rhs[n0..n0 + n1];
                let mut u = vec![0.0_f64; n0];
                let mut p = vec![0.0_f64; n1];
                let lin = SchurComplementSolver::solve(&sys, f, g, &mut u, &mut p, &self.cfg.linear)?;

                let mut dx = vec![0.0_f64; rhs.len()];
                dx[..n0].copy_from_slice(&u);
                dx[n0..n0 + n1].copy_from_slice(&p);
                Ok((dx, lin))
            }
        }
    }
}

fn validate_layout(
    sizes: &[usize],
    v: &BlockVector,
    name: &str,
) -> Result<(), CoupledSolveError> {
    if sizes.len() != v.n_blocks() {
        return Err(CoupledSolveError::InvalidLayout(format!(
            "{name} has {} blocks, but problem declares {}",
            v.n_blocks(),
            sizes.len()
        )));
    }
    for (i, &s) in sizes.iter().enumerate() {
        let got = v.block_size(i);
        if s != got {
            return Err(CoupledSolveError::InvalidLayout(format!(
                "{name} block {i} size mismatch: expected {s}, got {got}"
            )));
        }
    }
    Ok(())
}

fn validate_jacobian_layout(
    sizes: &[usize],
    j: &BlockMatrix,
) -> Result<(), CoupledSolveError> {
    if j.n_row_blocks() != sizes.len() || j.n_col_blocks() != sizes.len() {
        return Err(CoupledSolveError::InvalidLayout(format!(
            "jacobian block grid mismatch: expected {}x{}, got {}x{}",
            sizes.len(),
            sizes.len(),
            j.n_row_blocks(),
            j.n_col_blocks()
        )));
    }

    for (i, &s) in sizes.iter().enumerate() {
        if j.row_sizes[i] != s {
            return Err(CoupledSolveError::InvalidLayout(format!(
                "jacobian row block {i} size mismatch: expected {s}, got {}",
                j.row_sizes[i]
            )));
        }
        if j.col_sizes[i] != s {
            return Err(CoupledSolveError::InvalidLayout(format!(
                "jacobian col block {i} size mismatch: expected {s}, got {}",
                j.col_sizes[i]
            )));
        }
    }
    Ok(())
}

fn block_matrix_to_csr(a: &BlockMatrix) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(a.total_rows(), a.total_cols());

    let mut row_offset = 0usize;
    for bi in 0..a.n_row_blocks() {
        let mut col_offset = 0usize;
        for bj in 0..a.n_col_blocks() {
            if let Some(b) = a.get(bi, bj) {
                for r in 0..b.nrows {
                    let start = b.row_ptr[r];
                    let end = b.row_ptr[r + 1];
                    for k in start..end {
                        let c = b.col_idx[k] as usize;
                        coo.add(row_offset + r, col_offset + c, b.values[k]);
                    }
                }
            }
            col_offset += a.col_sizes[bj];
        }
        row_offset += a.row_sizes[bi];
    }

    coo.into_csr()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_csr(v: f64) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(1, 1);
        coo.add(0, 0, v);
        coo.into_csr()
    }

    struct Linear2x2Problem {
        sizes: Vec<usize>,
    }

    impl Linear2x2Problem {
        fn new() -> Self { Self { sizes: vec![1, 1] } }
    }

    impl CoupledProblem for Linear2x2Problem {
        fn block_sizes(&self) -> &[usize] { &self.sizes }

        fn residual(&self, _t: f64, state: &BlockVector, rhs: &BlockVector, out: &mut BlockVector) {
            let x0 = state.block(0)[0];
            let x1 = state.block(1)[0];
            let b0 = rhs.block(0)[0];
            let b1 = rhs.block(1)[0];
            out.block_mut(0)[0] = 2.0 * x0 + 1.0 * x1 - b0;
            out.block_mut(1)[0] = 1.0 * x0 + 3.0 * x1 - b1;
        }

        fn jacobian(&self, _t: f64, _state: &BlockVector) -> BlockMatrix {
            let mut j = BlockMatrix::new_square(self.sizes.clone());
            j.set(0, 0, scalar_csr(2.0));
            j.set(0, 1, scalar_csr(1.0));
            j.set(1, 0, scalar_csr(1.0));
            j.set(1, 1, scalar_csr(3.0));
            j
        }
    }

    #[test]
    fn monolithic_newton_solves_linear_coupled_system() {
        let problem = Linear2x2Problem::new();
        let solver = CoupledNewtonSolver::new(CoupledNewtonConfig {
            atol: 1e-12,
            rtol: 1e-12,
            max_iter: 8,
            gmres_restart: 8,
            line_search: true,
            line_search_min_alpha: 1e-6,
            line_search_shrink: 0.5,
            line_search_max_backtracks: 20,
            line_search_sufficient_decrease: 1e-4,
            linear: SolverConfig { rtol: 1e-14, atol: 0.0, max_iter: 50, verbose: false, print_level: crate::PrintLevel::Silent },
            linear_strategy: CoupledLinearStrategy::Gmres,
        });

        let mut rhs = BlockVector::new(vec![1, 1]);
        rhs.block_mut(0)[0] = 1.0;
        rhs.block_mut(1)[0] = 2.0;

        let mut state = BlockVector::new(vec![1, 1]);
        let res = solver.solve(&problem, 0.0, &rhs, &mut state).unwrap();

        assert!(res.converged);
        assert!(res.final_residual < 1e-12);
        assert!((state.block(0)[0] - 0.2).abs() < 1e-12);
        assert!((state.block(1)[0] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn monolithic_newton_schur_strategy_solves_linear_coupled_system() {
        let problem = Linear2x2Problem::new();
        let solver = CoupledNewtonSolver::new(CoupledNewtonConfig {
            atol: 1e-12,
            rtol: 1e-12,
            max_iter: 8,
            gmres_restart: 8,
            line_search: true,
            line_search_min_alpha: 1e-6,
            line_search_shrink: 0.5,
            line_search_max_backtracks: 20,
            line_search_sufficient_decrease: 1e-4,
            linear: SolverConfig { rtol: 1e-14, atol: 0.0, max_iter: 50, verbose: false, print_level: crate::PrintLevel::Silent },
            linear_strategy: CoupledLinearStrategy::BlockSchur2x2,
        });

        let mut rhs = BlockVector::new(vec![1, 1]);
        rhs.block_mut(0)[0] = 1.0;
        rhs.block_mut(1)[0] = 2.0;

        let mut state = BlockVector::new(vec![1, 1]);
        let res = solver.solve(&problem, 0.0, &rhs, &mut state).unwrap();

        assert!(res.converged);
        assert!(res.final_residual < 1e-12);
        assert!((state.block(0)[0] - 0.2).abs() < 1e-12);
        assert!((state.block(1)[0] - 0.6).abs() < 1e-12);
    }
}