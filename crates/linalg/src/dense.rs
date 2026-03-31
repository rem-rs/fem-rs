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
    for k in 0..n {
        b.swap(k, piv[k]);
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
// Dense matrix-matrix product
// -------------------------------------------------------------------------

/// Compute `C = A * B` for dense matrices.
///
/// - `A` is `m × k` (row-major, stride `k`)
/// - `B` is `k × n` (row-major, stride `n`)
/// - `C` is `m × n` (row-major, stride `n`) — **overwritten**, not accumulated
///
/// # Panics
/// Panics if slice lengths are inconsistent with the given dimensions.
pub fn dense_matmat(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
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

    #[test]
    fn dense_matmat_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
        //   = [[19,22],[43,50]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];
        dense_matmat(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-14, "C[0,0] = {}", c[0]);
        assert!((c[1] - 22.0).abs() < 1e-14, "C[0,1] = {}", c[1]);
        assert!((c[2] - 43.0).abs() < 1e-14, "C[1,0] = {}", c[2]);
        assert!((c[3] - 50.0).abs() < 1e-14, "C[1,1] = {}", c[3]);
    }

    #[test]
    fn dense_matmat_non_square() {
        // A (2×3) × B (3×2) = C (2×2)
        // A = [[1,0,2],[0,3,1]]
        // B = [[1,2],[0,1],[4,0]]
        // C[0,0]=1+0+8=9  C[0,1]=2+0+0=2
        // C[1,0]=0+0+4=4  C[1,1]=0+3+0=3
        let a = [1.0, 0.0, 2.0, 0.0, 3.0, 1.0];
        let b = [1.0, 2.0, 0.0, 1.0, 4.0, 0.0];
        let mut c = [0.0f64; 4];
        dense_matmat(&a, &b, &mut c, 2, 3, 2);
        assert!((c[0] - 9.0).abs() < 1e-14);
        assert!((c[1] - 2.0).abs() < 1e-14);
        assert!((c[2] - 4.0).abs() < 1e-14);
        assert!((c[3] - 3.0).abs() < 1e-14);
    }
}
