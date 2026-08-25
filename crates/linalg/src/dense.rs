use fem_core::{FemError, FemResult};

// -------------------------------------------------------------------------
// LU factorization (partial pivoting, in-place)
// -------------------------------------------------------------------------

/// LU factorization with partial pivoting (in-place).
///
/// On return, `a` holds the combined L and U factors (unit lower-triangular L,
/// upper-triangular U) in the same `n×n` row-major array.
/// `piv[i]` is the row index that was swapped with row `i` during pivoting.
///
/// Returns `Err(FemError::DimMismatch)` if `a.len() != n*n` or `piv.len() != n`.
/// Returns `Err(FemError::SolverDivergence)` if the matrix is numerically singular
/// (pivot magnitude < 1e-14 × max absolute diagonal value).
pub fn lu_factor(a: &mut [f64], n: usize, piv: &mut [usize]) -> FemResult<()> {
    if a.len() != n * n {
        return Err(FemError::DimMismatch { expected: n * n, actual: a.len() });
    }
    if piv.len() != n {
        return Err(FemError::DimMismatch { expected: n, actual: piv.len() });
    }
    if n == 0 {
        return Ok(());
    }

    // Compute max absolute diagonal for singularity threshold.
    let max_diag = (0..n)
        .map(|i| a[i * n + i].abs())
        .fold(0.0_f64, f64::max);
    let tol = 1e-14 * max_diag.max(1.0);

    for k in 0..n {
        // --- find pivot row (largest absolute value in column k, rows k..n) ---
        let mut pivot_row = k;
        let mut pivot_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }
        piv[k] = pivot_row;

        // --- swap rows k and pivot_row ---
        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
        }

        // --- check for singularity ---
        let diag = a[k * n + k];
        if diag.abs() < tol {
            return Err(FemError::SolverDivergence(k));
        }

        // --- eliminate column k below the diagonal ---
        let inv_diag = 1.0 / diag;
        for i in (k + 1)..n {
            let factor = a[i * n + k] * inv_diag;
            a[i * n + k] = factor; // store L factor in-place
            for j in (k + 1)..n {
                let u_kj = a[k * n + j];
                a[i * n + j] -= factor * u_kj;
            }
        }
    }

    Ok(())
}

// -------------------------------------------------------------------------
// LU solve (forward + back substitution)
// -------------------------------------------------------------------------

/// Solve `A x = b` given the LU factorization produced by [`lu_factor`].
///
/// `a` and `piv` must come from a prior call to `lu_factor` for the same `n`.
/// `b` is overwritten with the solution `x` on return.
///
/// # Panics
/// Panics if slice lengths are inconsistent with `n`.
pub fn lu_solve(a: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    assert_eq!(a.len(), n * n);
    assert_eq!(piv.len(), n);
    assert_eq!(b.len(), n);

    // --- apply row permutations to b ---
    for (k, &p) in piv.iter().enumerate().take(n) {
        b.swap(k, p);
    }

    // --- forward substitution: solve L y = b (L unit lower-triangular) ---
    for i in 1..n {
        let mut s = 0.0;
        for j in 0..i {
            s += a[i * n + j] * b[j];
        }
        b[i] -= s;
    }

    // --- back substitution: solve U x = y (U upper-triangular) ---
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += a[i * n + j] * b[j];
        }
        b[i] = (b[i] - s) / a[i * n + i];
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct A from its LU factorization (with pivoting) and check A x = b.
    fn solve_and_check(a_orig: &[f64], b_orig: &[f64], n: usize) {
        let mut a = a_orig.to_vec();
        let mut piv = vec![0usize; n];
        lu_factor(&mut a, n, &mut piv).expect("lu_factor should succeed");

        let mut x = b_orig.to_vec();
        lu_solve(&a, n, &piv, &mut x);

        // Verify A_orig * x ≈ b_orig
        for i in 0..n {
            let mut row_dot = 0.0;
            for j in 0..n {
                row_dot += a_orig[i * n + j] * x[j];
            }
            assert!(
                (row_dot - b_orig[i]).abs() < 1e-10,
                "residual at row {i}: got {row_dot}, expected {}",
                b_orig[i]
            );
        }
    }

    #[test]
    fn lu_solve_3x3_spd() {
        // SPD matrix:
        //   [ 4  2  0 ]
        //   [ 2  5  1 ]
        //   [ 0  1  3 ]
        #[rustfmt::skip]
        let a = [
            4.0, 2.0, 0.0,
            2.0, 5.0, 1.0,
            0.0, 1.0, 3.0,
        ];
        let b = [6.0, 8.0, 4.0];
        solve_and_check(&a, &b, 3);
    }

    #[test]
    fn lu_solve_identity() {
        // Identity matrix — solution must equal RHS.
        #[rustfmt::skip]
        let a = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let b = [3.0, 1.0, 4.0];
        solve_and_check(&a, &b, 3);
    }

    #[test]
    fn lu_factor_singular_returns_err() {
        // Rank-deficient 3×3 (row 2 = row 1 + row 0).
        #[rustfmt::skip]
        let mut a = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            5.0, 7.0, 9.0,
        ];
        let mut piv = [0usize; 3];
        let result = lu_factor(&mut a, 3, &mut piv);
        assert!(
            result.is_err(),
            "expected Err for singular matrix, got Ok"
        );
    }
}
