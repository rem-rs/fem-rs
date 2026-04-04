//! Integrator traits and quadrature-point data for vector finite elements
//! (H(curl) Nédélec, H(div) Raviart-Thomas).
//!
//! These are the vector analogues of [`QpData`](crate::integrator::QpData),
//! [`BilinearIntegrator`](crate::integrator::BilinearIntegrator), and
//! [`LinearIntegrator`](crate::integrator::LinearIntegrator).

// ─── Quadrature-point data ──────────────────────────────────────────────────

/// Data available to vector integrators at each volume quadrature point.
///
/// All basis-function data is **already Piola-transformed and sign-corrected**
/// by the [`VectorAssembler`](crate::vector_assembler::VectorAssembler).
#[derive(Debug)]
pub struct VectorQpData<'a> {
    /// Number of local DOFs on this element.
    pub n_dofs: usize,
    /// Spatial dimension (2 or 3).
    pub dim: usize,
    /// Effective integration weight: quadrature weight × |det J|.
    pub weight: f64,
    /// Vector basis function values at this quadrature point.
    ///
    /// Layout: `phi_vec[i * dim + c]` = component `c` of basis function `i`.
    /// Length: `n_dofs × dim`.
    pub phi_vec: &'a [f64],
    /// Curl of each basis function (Piola-transformed).
    ///
    /// - **2-D**: scalar curl, length `n_dofs`.
    ///   `curl[i]` = scalar 2-D curl of basis function `i`.
    /// - **3-D**: vector curl, length `n_dofs × 3`.
    ///   `curl[i * 3 + c]` = component `c` of curl of basis function `i`.
    pub curl: &'a [f64],
    /// Divergence of each basis function (Piola-transformed).
    ///
    /// Length: `n_dofs`.  `div[i]` = divergence of basis function `i`.
    pub div: &'a [f64],
    /// Physical coordinates of this quadrature point; length `dim`.
    pub x_phys: &'a [f64],
}

// ─── Integrator traits ──────────────────────────────────────────────────────

/// Accumulate a bilinear-form contribution for vector FE into the element
/// stiffness matrix.
///
/// `k_elem` is row-major with shape `[n_dofs × n_dofs]`.
///
/// Implementors must **add** their contribution (not overwrite), as multiple
/// integrators may share the same element matrix.
pub trait VectorBilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]);
}

/// Accumulate a linear-form contribution for vector FE into the element
/// load vector.
///
/// `f_elem` has length `n_dofs`.
pub trait VectorLinearIntegrator: Send + Sync {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]);
}
