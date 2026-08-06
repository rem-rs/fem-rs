//! Chebyshev polynomial smoother (matching MFEM `OperatorChebyshevSmoother`).
//!
//! Applies a Chebyshev polynomial of the form
//! ```text
//!   y = p(D⁻¹A) D⁻¹ x
//! ```
//! where `p(t)` is the degree-`(order-1)` Chebyshev polynomial mapped to
//! `[λ_min, λ_max]`, the estimated spectral interval of `D⁻¹A`.
//!
//! Used as a smoother in geometric multigrid.

use fem_linalg::CsrMatrix;

/// Chebyshev polynomial smoother.
///
/// `Mult(rhs, sol)` computes `sol = p(D⁻¹A) D⁻¹ rhs` where `p` is a
/// Chebyshev polynomial of degree `order-1` on `[λ_min, λ_max]`.
pub struct ChebyshevSmoother {
    n: usize,
    /// Inverse diagonal (1.0 for BC DOFs).
    dinv: Vec<f64>,
    /// Chebyshev coefficients (length = order).
    coeffs: Vec<f64>,
    /// BC DOF list (zeroed after smoothing).
    bc_dofs: Vec<u32>,
    /// Scratch vector.
    tmp: Vec<f64>,
}

impl ChebyshevSmoother {
    /// Build a Chebyshev smoother for the given matrix.
    ///
    /// - `a` — system matrix (used for eigenvalue estimation).
    /// - `diag` — diagonal of the operator (for Jacobi preconditioning).
    /// - `bc_dofs` — essential BC DOFs (diag entry = 1 for these).
    /// - `order` — number of Chebyshev terms (2 = quadratic, matching C++ ex26).
    /// - `max_eig_estimate` — estimate of λ_max(D⁻¹A).  Use `None` for auto-estimate.
    pub fn new(
        a: &CsrMatrix<f64>,
        diag: &[f64],
        bc_dofs: &[u32],
        order: usize,
        max_eig_estimate: Option<f64>,
    ) -> Self {
        let n = diag.len();

        // Inverse diagonal (1 for BC DOFs, matching C++ OperatorChebyshevSmoother)
        let mut dinv = vec![0.0; n];
        for i in 0..n {
            dinv[i] = if diag[i].abs() > 1e-30 {
                1.0 / diag[i]
            } else {
                1.0
            };
        }
        for &d in bc_dofs {
            if (d as usize) < n {
                dinv[d as usize] = 1.0;
            }
        }

        // Estimate λ_max via power iteration on D⁻¹A
        let max_eig =
            max_eig_estimate.unwrap_or_else(|| estimate_max_eigenvalue(a, &dinv, bc_dofs));

        // Chebyshev parameters (matching MFEM OperatorChebyshevSmoother::Setup)
        let upper = 1.2 * max_eig; // over-estimate upper bound
        let lower = 0.3 * max_eig; // under-estimate lower bound
        let theta = 0.5 * (upper + lower);
        let delta = 0.5 * (upper - lower);
        let th2 = theta * theta;
        let d2 = delta * delta;

        // Precompute coefficients for given order (matching MFEM's analytic formulas)
        let coeffs = match order - 1 {
            0 => vec![1.0 / theta],
            1 => {
                let tmp = 1.0 / (d2 - 2.0 * th2);
                vec![-4.0 * theta * tmp, 2.0 * tmp]
            }
            2 => {
                let tmp0 = 3.0 * d2;
                let tmp1 = th2;
                let tmp2 = 1.0 / (-4.0 * theta * th2 + theta * tmp0);
                vec![
                    tmp2 * (tmp0 - 12.0 * tmp1),
                    12.0 / (tmp0 - 4.0 * tmp1),
                    -4.0 * tmp2,
                ]
            }
            3 => {
                let tmp0 = 8.0 * d2;
                let tmp1 = th2;
                let tmp3 = 1.0 / (d2 * d2 + 8.0 * th2 * th2 - tmp1 * tmp0);
                vec![
                    tmp3 * (32.0 * theta * th2 - 16.0 * theta * d2),
                    tmp3 * (-48.0 * tmp1 + tmp0),
                    32.0 * theta * tmp3,
                    -8.0 * tmp3,
                ]
            }
            _ => panic!("ChebyshevSmoother: order {order} not supported (use 1-4)"),
        };

        ChebyshevSmoother {
            n,
            dinv,
            coeffs,
            bc_dofs: bc_dofs.to_vec(),
            tmp: vec![0.0; n],
        }
    }

    /// Apply the Chebyshev smoother: `sol = p(D⁻¹A) D⁻¹ * rhs`.
    /// This matches MFEM `OperatorChebyshevSmoother::Mult`.
    pub fn mult(&mut self, a: &CsrMatrix<f64>, rhs: &[f64], sol: &mut [f64]) {
        let n = self.n;
        // y = 0
        for i in 0..n {
            sol[i] = 0.0;
        }

        // residual starts as a copy of rhs
        let mut r = rhs.to_vec();

        for k in 0..self.coeffs.len() {
            // Apply operator (r = A * r) for k > 0
            if k > 0 {
                a.spmv(&self.tmp, &mut r);
            }

            // r = D⁻¹ * r
            for i in 0..n {
                r[i] *= self.dinv[i];
            }

            // sol += coeffs[k] * r
            let c = self.coeffs[k];
            for i in 0..n {
                sol[i] += c * r[i];
            }

            // Save r as tmp for next iteration (need to multiply by A)
            // Actually we need A * r for the next iteration
            // Save r in tmp for next oper->Mult
            self.tmp.copy_from_slice(&r);
        }

        // Zero BC DOFs
        for &d in &self.bc_dofs {
            if (d as usize) < n {
                sol[d as usize] = 0.0;
            }
        }
    }
}

/// Estimate the largest eigenvalue of `D⁻¹ A` using power iteration.
fn estimate_max_eigenvalue(a: &CsrMatrix<f64>, dinv: &[f64], bc_dofs: &[u32]) -> f64 {
    let n = a.nrows;
    let mut v = vec![1.0; n];
    let mut w = vec![0.0; n];

    // Seed with non-uniform values
    for i in 0..n {
        v[i] = 1.0 + (i as f64) / (n as f64);
    }
    for &d in bc_dofs {
        if (d as usize) < n {
            v[d as usize] = 0.0;
        }
    }

    let mut lambda = 0.0;
    for _iter in 0..50 {
        // w = D⁻¹ * A * v
        a.spmv(&v, &mut w);
        for i in 0..n {
            w[i] *= dinv[i];
        }
        for &d in bc_dofs {
            if (d as usize) < n {
                w[d as usize] = 0.0;
            }
        }

        // Rayleigh quotient
        let vw: f64 = (0..n).map(|i| v[i] * w[i]).sum();
        let vv: f64 = (0..n).map(|i| v[i] * v[i]).sum();
        if vv < 1e-30 {
            break;
        }
        let new_lambda = vw / vv;

        if (new_lambda - lambda).abs() < 1e-6 * new_lambda.abs() {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;

        // Normalize
        let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        for i in 0..n {
            v[i] = w[i] / nrm;
        }
        for &d in bc_dofs {
            if (d as usize) < n {
                v[d as usize] = 0.0;
            }
        }
    }
    lambda.abs().max(0.1) // ensure minimum estimate
}
