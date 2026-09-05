//! Newton iteration for inverse isoparametric mapping.
//!
//! Solves x(ξ) = p for reference coordinates ξ, given a physical point p
//! and an element transformation.  Used by FindPoints to map physical
//! points back to reference coordinates within candidate elements.

use nalgebra::DMatrix;

/// Status of a Newton solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonStatus {
    Converged,
    NotConverged,
    Diverged,
    InvalidJacobian,
}

/// Result of a Newton iteration.
#[derive(Debug, Clone)]
pub struct NewtonResult {
    pub status: NewtonStatus,
    pub residual: f64,
    pub iterations: usize,
}

impl NewtonResult {
    pub fn is_converged(&self) -> bool {
        self.status == NewtonStatus::Converged
    }
}

/// Compute the initial reference-coordinate guess for a point inside a simplex.
pub fn initial_guess_simplex<const D: usize>(
    x0: &[f64],
    jacobian: &DMatrix<f64>,
    p: &[f64],
) -> Vec<f64> {
    let rhs: Vec<f64> = (0..D).map(|i| p[i] - x0[i]).collect();
    if let Some(inv) = jacobian.clone().try_inverse() {
        let xi = inv * nalgebra::DVector::from_vec(rhs);
        xi.iter().copied().collect()
    } else {
        vec![1.0 / (D as f64 + 1.0); D]
    }
}

/// Newton iteration for inverse mapping on an affine simplex.
pub fn newton_inverse_simplex<const D: usize>(
    x0: &[f64],
    jacobian: &DMatrix<f64>,
    p: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, NewtonResult) {
    let dim = D;
    let mut xi = initial_guess_simplex::<D>(x0, jacobian, p);

    let inv_j = match jacobian.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            return (
                xi,
                NewtonResult {
                    status: NewtonStatus::InvalidJacobian,
                    residual: f64::INFINITY,
                    iterations: 0,
                },
            );
        }
    };

    for it in 0..max_iter {
        let mut x_map = vec![0.0; dim];
        for i in 0..dim {
            x_map[i] = x0[i];
            for j in 0..dim {
                x_map[i] += jacobian[(i, j)] * xi[j];
            }
        }
        let mut r = vec![0.0; dim];
        let mut rnorm = 0.0;
        for i in 0..dim {
            r[i] = x_map[i] - p[i];
            rnorm += r[i] * r[i];
        }
        rnorm = rnorm.sqrt();

        if rnorm < tol {
            return (
                xi,
                NewtonResult {
                    status: NewtonStatus::Converged,
                    residual: rnorm,
                    iterations: it,
                },
            );
        }

        let delta = inv_j.clone() * nalgebra::DVector::from_vec(r.clone());
        for i in 0..dim {
            xi[i] -= delta[i];
        }

        if xi.iter().any(|&v| !v.is_finite()) {
            return (
                xi,
                NewtonResult {
                    status: NewtonStatus::Diverged,
                    residual: rnorm,
                    iterations: it + 1,
                },
            );
        }
    }

    (
        xi,
        NewtonResult {
            status: NewtonStatus::NotConverged,
            residual: f64::NAN,
            iterations: max_iter,
        },
    )
}

/// Check if reference coordinates are inside a simplex (with tolerance).
pub fn is_inside_simplex(xi: &[f64], tol: f64) -> bool {
    let mut sum = 0.0;
    for &v in xi {
        if v < -tol {
            return false;
        }
        sum += v;
    }
    sum <= 1.0 + tol
}

/// Barycentric coordinates from reference coordinates of a simplex.
pub fn barycentric_from_ref(xi: &[f64]) -> Vec<f64> {
    let dim = xi.len();
    let mut lam = vec![0.0; dim + 1];
    let mut sum = 0.0;
    for i in 0..dim {
        lam[i] = xi[i];
        sum += xi[i];
    }
    lam[dim] = 1.0 - sum;
    lam
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_converges_for_affine_tri() {
        let x0 = [0.0, 0.0];
        let jac = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let p = [0.25, 0.25];
        let (xi, res) = newton_inverse_simplex::<2>(&x0, &jac, &p, 1e-12, 10);
        assert!(res.is_converged(), "should converge for affine tri");
        assert!((xi[0] - 0.25).abs() < 1e-10);
        assert!((xi[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn newton_converges_for_affine_tet() {
        let x0 = [0.0, 0.0, 0.0];
        let jac = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let p = [0.1, 0.2, 0.3];
        let (xi, res) = newton_inverse_simplex::<3>(&x0, &jac, &p, 1e-12, 10);
        assert!(res.is_converged());
        assert!((xi[0] - 0.1).abs() < 1e-10);
        assert!((xi[1] - 0.2).abs() < 1e-10);
        assert!((xi[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn is_inside_accepts_valid() {
        assert!(is_inside_simplex(&[0.2, 0.3], 1e-9));
        assert!(is_inside_simplex(&[0.0, 0.0], 1e-9));
    }

    #[test]
    fn is_inside_rejects_outside() {
        assert!(!is_inside_simplex(&[0.6, 0.6], 1e-9));
        assert!(!is_inside_simplex(&[-0.1, 0.5], 1e-9));
    }

    #[test]
    fn barycentric_sums_to_one() {
        let lam = barycentric_from_ref(&[0.1, 0.2, 0.3]);
        let sum: f64 = lam.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(lam.len(), 4);
        assert!((lam[3] - 0.4).abs() < 1e-12);
    }
}
