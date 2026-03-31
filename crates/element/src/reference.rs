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

    /// Return a quadrature rule that integrates polynomials of the given `order` exactly.
    fn quadrature(&self, order: u8) -> QuadratureRule;

    /// Reference-domain coordinates of each DOF node (for interpolation/visualization).
    ///
    /// Returns a `Vec` of `n_dofs()` coordinate vectors, each of length `dim()`.
    fn dof_coords(&self) -> Vec<Vec<f64>>;
}
