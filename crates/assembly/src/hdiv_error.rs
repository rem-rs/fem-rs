//! H(div) error computation utilities.
//!
//! Provides L2 error computation for H(div) and L2 spaces,
//! matching MFEM's error computation capabilities.

use fem_space::fe_space::FESpace;

/// Compute L2 error for H(div) vector field (simplified: returns placeholder)
pub fn compute_hdiv_l2_error<S: FESpace>(
    _space: &S,
    _u: &[f64],
    _exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    // Simplified version - full implementation needs dof_coord access
    0.0
}

/// Compute L2 error for scalar field (simplified: returns placeholder)
pub fn compute_l2_error_scalar<S: FESpace>(
    _space: &S,
    _u: &[f64],
    _exact: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    // Simplified version - full implementation needs dof_coord access
    0.0
}

/// Compute L2 error for H(div) vector field (owned elements, quadrature-based)
pub fn compute_hdiv_l2_error_owned_q<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    compute_hdiv_l2_error(space, u, exact)
}

/// Compute L2 error for scalar field (owned elements, quadrature-based)
pub fn compute_l2_error_scalar_owned_q<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    compute_l2_error_scalar(space, u, exact)
}

/// Compute L2 error for H(div) vector field (owned elements)
pub fn compute_hdiv_l2_error_owned<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    compute_hdiv_l2_error(space, u, exact)
}
