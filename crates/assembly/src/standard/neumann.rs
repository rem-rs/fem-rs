//! Neumann boundary linear form integrator.
//!
//! Computes the boundary contribution to the load vector:
//!
//! ```text
//! F_Γ(v) = ∫_Γ g(x) v ds
//! ```
//!
//! where `g` is the prescribed outward normal flux (or any boundary data).

use crate::integrator::{BdQpData, BoundaryLinearIntegrator};

/// Linear integrator for a Neumann (natural) boundary condition `∫_Γ g(x) v ds`.
///
/// `g` may depend on the physical position `x` and optionally the outward unit normal.
///
/// # Example
/// ```
/// # use fem_assembly::standard::NeumannIntegrator;
/// // Constant flux g = 1.0 on the boundary.
/// let integ = NeumannIntegrator::new(|_x, _n| 1.0);
/// ```
pub struct NeumannIntegrator<F>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    g: F,
}

impl<F> NeumannIntegrator<F>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    /// Create a new Neumann integrator.
    ///
    /// `g(x, n)` receives the physical coordinates `x` and the outward unit normal `n`.
    pub fn new(g: F) -> Self { NeumannIntegrator { g } }
}

impl<F> BoundaryLinearIntegrator for NeumannIntegrator<F>
where
    F: Fn(&[f64], &[f64]) -> f64 + Send + Sync,
{
    /// `f_face[i] += w · g(x, n) · φᵢ`
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let gval = (self.g)(qp.x_phys, qp.normal);
        let w_g  = qp.weight * gval;
        for i in 0..qp.n_dofs {
            f_face[i] += w_g * qp.phi[i];
        }
    }
}
