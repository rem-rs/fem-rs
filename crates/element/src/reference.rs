//! Core traits for reference finite elements.

/// A quadrature rule on a reference domain.
///
/// - `points[q]` are the reference-coordinate quadrature points (len = dim per point).
/// - `weights[q]` are the corresponding quadrature weights.
///
/// The weights are scaled so that `sum(weights) = measure(reference domain)`.
/// For example, the reference triangle has area 0.5, so triangle weights sum to 0.5.
#[derive(Debug, Clone)]
pub struct QuadratureRule {
    /// Reference-coordinate quadrature points.  `points[q].len() == dim`.
    pub points:  Vec<Vec<f64>>,
    /// Quadrature weights.
    pub weights: Vec<f64>,
}

impl QuadratureRule {
    /// Number of quadrature points.
    pub fn n_points(&self) -> usize { self.weights.len() }
}

/// A reference finite element: basis functions defined on a fixed reference domain.
///
/// Concrete implementations are the Lagrange elements in the [`crate::lagrange`] module.
///
/// # Mathematical conventions
/// - Reference coordinates are `ξ` (`xi`), with length equal to [`ReferenceElement::dim`].
/// - Basis functions are indexed `φ₀ … φₖ` where `k = n_dofs - 1`.
/// - Gradients are stored **row-major**: `grads[i * dim + j] = ∂φᵢ/∂ξⱼ`.
pub trait ReferenceElement: Send + Sync {
    /// Topological dimension of the reference domain (1, 2, or 3).
    fn dim(&self) -> u8;

    /// Polynomial order of the element (1 = P1/Q1, 2 = P2/Q2, …).
    fn order(&self) -> u8;

    /// Number of degrees of freedom (basis functions).
    fn n_dofs(&self) -> usize;

    /// Evaluate all basis function values at reference point `xi` (len = `dim()`).
    ///
    /// `values` must have length `n_dofs()`.
    fn eval_basis(&self, xi: &[f64], values: &mut [f64]);

    /// Evaluate all basis function gradients at reference point `xi`.
    ///
    /// `grads` must have length `n_dofs() * dim()`.
    /// Layout: `grads[i * dim + j] = ∂φᵢ/∂ξⱼ`.
    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]);

    /// Evaluate all basis function Hessians at reference point `xi`.
    ///
    /// `hess` must have length `n_dofs() * dim() * dim`.
    /// Layout: `hess[i * dim * dim + d * dim + e] = ∂²φᵢ / (∂ξ_d ∂ξ_e)`.
    ///
    /// Default implementation uses finite differences on `eval_grad_basis`
    /// (order O(h²), h = 1e-6). Override for analytic Hessians.
    fn eval_hessian(&self, xi: &[f64], hess: &mut [f64]) {
        let n = self.n_dofs();
        let d = self.dim() as usize;
        let h = 1e-6_f64;
        let mut g_plus  = vec![0.0_f64; n * d];
        let mut g_minus = vec![0.0_f64; n * d];
        for di in 0..d {
            let mut xi_p = xi.to_vec();
            let mut xi_m = xi.to_vec();
            xi_p[di] += h;
            xi_m[di] -= h;
            self.eval_grad_basis(&xi_p, &mut g_plus);
            self.eval_grad_basis(&xi_m, &mut g_minus);
            for i in 0..n {
                for dj in 0..d {
                    let fd = (g_plus[i * d + dj] - g_minus[i * d + dj]) / (2.0 * h);
                    hess[i * d * d + di * d + dj] = fd;
                }
            }
        }
        // Enforce symmetry for cross-derivatives (FD numerical asymmetry).
        for i in 0..n {
            for di in 0..d {
                for dj in (di + 1)..d {
                    let a = hess[i * d * d + di * d + dj];
                    let b = hess[i * d * d + dj * d + di];
                    let avg = 0.5 * (a + b);
                    hess[i * d * d + di * d + dj] = avg;
                    hess[i * d * d + dj * d + di] = avg;
                }
            }
        }
    }

    /// Return a quadrature rule that integrates polynomials of the given `order` exactly.
    fn quadrature(&self, order: u8) -> QuadratureRule;

    /// Reference-domain coordinates of each DOF node (for interpolation/visualization).
    ///
    /// Returns a `Vec` of `n_dofs()` coordinate vectors, each of length `dim()`.
    fn dof_coords(&self) -> Vec<Vec<f64>>;
}

/// A vector-valued reference finite element for H(curl) or H(div) spaces.
///
/// Each basis function `Φᵢ` is a vector of length `dim()`.
///
/// # Layout conventions
/// - `values` in `eval_basis_vec`: length `n_dofs() * dim()`.
///   `values[i * dim + c]` = component `c` of basis function `i`.
/// - `curl_vals` in `eval_curl`: for 2-D this is a scalar-per-basis (len = `n_dofs()`);
///   for 3-D it is a 3-vector-per-basis (len = `n_dofs() * 3`).
/// - `div_vals` in `eval_div`: one scalar per basis (len = `n_dofs()`).
pub trait VectorReferenceElement: Send + Sync {
    /// Topological dimension (2 or 3).
    fn dim(&self) -> u8;
    /// Polynomial order.
    fn order(&self) -> u8;
    /// Number of vector-valued DOFs.
    fn n_dofs(&self) -> usize;
    /// Evaluate vector basis functions at `xi`.  `values` len = `n_dofs() * dim()`.
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]);
    /// Evaluate curl of each basis function at `xi`.
    /// 2-D: `curl_vals` len = `n_dofs()` (scalar curl = ∂Φ_y/∂ξ − ∂Φ_x/∂η).
    /// 3-D: `curl_vals` len = `n_dofs() * 3`.
    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]);
    /// Evaluate divergence of each basis function.  `div_vals` len = `n_dofs()`.
    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]);
    /// Quadrature rule suitable for the element.
    fn quadrature(&self, order: u8) -> QuadratureRule;
    /// Reference coordinates of DOF sites (edge/face midpoints).
    fn dof_coords(&self) -> Vec<Vec<f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrange::tri::TriP1;

    #[test]
    fn fd_hessian_tri_p1_is_nearly_zero() {
        let elem = TriP1;
        let d = elem.dim() as usize;
        let n = elem.n_dofs();
        let mut hess = vec![0.0_f64; n * d * d];
        elem.eval_hessian(&[0.3, 0.2], &mut hess);
        let max_h: f64 = hess.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_h < 5e-6, "TriP1 Hessian should be ~zero, got |max|={max_h:.3e}");
    }

    #[test]
    fn fd_hessian_tri_p2_matches_analytic() {
        use crate::lagrange::tri::TriP2;
        let elem = TriP2;
        let d = elem.dim() as usize;
        let n = elem.n_dofs();
        let mut hess = vec![0.0_f64; n * d * d];
        elem.eval_hessian(&[0.3, 0.2], &mut hess);
        // TriP2 DOF 3: phi_3 = 4*x*(1-x-y) = 4x - 4x^2 - 4xy.
        // d^2 phi_3 / dx dy = -4.
        let dxy = hess[3 * 4 + 0 * 2 + 1];
        let dyx = hess[3 * 4 + 1 * 2 + 0];
        assert!((dxy - (-4.0)).abs() < 5e-6,
            "d^2phi_3/dxdy should be -4, got {dxy:.6e}");
        assert!((dyx - (-4.0)).abs() < 5e-6,
            "d^2phi_3/dydx should be -4, got {dyx:.6e}");
        assert!((dxy - dyx).abs() < 1e-12, "Hessian symmetry violated");
    }

    #[test]
    fn fd_hessian_falls_back_finitely() {
        let elems: [&dyn ReferenceElement; 6] = [
            &TriP1,
            &crate::lagrange::tri::TriP2,
            &crate::lagrange::tri::TriP3,
            &crate::lagrange::quad::QuadQ1,
            &crate::lagrange::quad::QuadQ2,
            &crate::lagrange::tet::TetP1,
        ];
        for elem in &elems {
            let d = elem.dim() as usize;
            let n = elem.n_dofs();
            let mut hess = vec![0.0_f64; n * d * d];
            let xi: Vec<f64> = (0..d).map(|_| 0.3).collect();
            elem.eval_hessian(&xi, &mut hess);
            assert!(hess.iter().all(|v| v.is_finite()),
                "eval_hessian produced non-finite for dim={d} n={n}");
        }
    }
}
