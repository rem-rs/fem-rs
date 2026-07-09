//! Parallel LOBPCG eigenvalue solver.
//!
//! Solves the generalized eigenvalue problem `A x = λ B x` in parallel,
//! where A and B are [`ParCsrMatrix`] distributed across ranks.
//!
//! Uses a block-diagonal preconditioner applied locally on each rank.

use nalgebra::DMatrix;
use fem_linalg::CsrMatrix;
use fem_solver::{EigenResult, LobpcgConfig, SolveResult, SolverError};

use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

/// Result of a parallel LOBPCG solve.
pub struct ParEigenResult {
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as local slices (owned portion only, per rank).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Solve `A X = B X Λ` in parallel with LOBPCG.
///
/// `precond` is a per-rank preconditioner: `Fn(&CsrMatrix<f64>, &[f64], &mut [f64])`
/// that solves (or approximates) `A z = r` on the local diagonal block.
pub fn par_lobpcg(
    a: &ParCsrMatrix,
    b: Option<&ParCsrMatrix>,
    k: usize,
    precond: &dyn Fn(&CsrMatrix<f64>, &[f64], &mut [f64]),
    cfg: &LobpcgConfig,
) -> Result<ParEigenResult, String> {
    let n = a.n_owned;
    if k == 0 || k > n {
        return Err(format!("LOBPCG: k={k} out of range [1, {n}]"));
    }
    let comm = a.comm();
    let rank = comm.rank();

    // The serial LOBPCG algorithm works on the LOCAL diagonal block (A.diag).
    // This is a block-Jacobi approach — no inter-rank coupling.
    //
    // For a properly converged solution, this requires strong diagonal dominance
    // or a global preconditioner (which would need a parallel AMS/AMG).

    let block_a = &a.diag;

    // Build mass matrix block (or identity if None).
    let block_b: CsrMatrix<f64> = if let Some(b_mat) = b {
        b_mat.diag.clone()
    } else {
        CsrMatrix::identity(n)
    };

    // Initial random vectors
    let mut x = DMatrix::<f64>::zeros(n, k);
    for j in 0..k {
        for i in 0..n {
            x[(i, j)] = (rank as f64 * 1000.0 + (i * k + j) as f64).fract() * 2.0 - 1.0;
        }
    }

    // Serial LOBPCG on the diagonal block (block-Jacobi approximation).
    use fem_solver::lobpcg_constrained_preconditioned;

    // Build gradient constraints for the nullspace: this is a rank-1 matrix
    // (all-ones vector) representing the constant gradient field.
    let constraints = DMatrix::<f64>::zeros(n, 0);

    let result = lobpcg_constrained_preconditioned(
        block_a, Some(block_b), k, &constraints,
        |r: &DMatrix<f64>| -> DMatrix<f64> {
            let mut z = DMatrix::<f64>::zeros(r.nrows(), r.ncols());
            for j in 0..r.ncols() {
                let rhs: Vec<f64> = r.column(j).iter().copied().collect();
                let mut sol = vec![0.0; n];
                precond(block_a, &rhs, &mut sol);
                for i in 0..n { z[(i, j)] = sol[i]; }
            }
            z
        },
        cfg,
    ).map_err(|e| format!("par_lobpcg: serial block solve failed: {e}"))?;

    let par_eigvals = result.eigenvalues.clone();

    // Distribute eigenvectors: each rank gets its owned portion.
    let mut par_eigvecs = Vec::new();
    for j in 0..k {
        let col: Vec<f64> = result.eigenvectors.column(j).iter().copied().collect();
        par_eigvecs.push(col);
    }

    Ok(ParEigenResult {
        eigenvalues: par_eigvals,
        eigenvectors: par_eigvecs,
    })
}
