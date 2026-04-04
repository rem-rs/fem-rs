//! Integrator traits and quadrature-point data.
//!
//! Integrators receive per-quadrature-point data and accumulate contributions
//! into an element matrix (bilinear) or element vector (linear).

use fem_core::types::ElemId;

// ─── Volume integrals ─────────────────────────────────────────────────────────

/// Data available to integrators at each volume quadrature point.
#[derive(Debug)]
pub struct QpData<'a> {
    /// Number of local DOFs on this element.
    pub n_dofs:    usize,
    /// Spatial dimension.
    pub dim:       usize,
    /// Effective integration weight: quadrature weight × |det J|.
    pub weight:    f64,
    /// Basis function values at this quadrature point; length `n_dofs`.
    pub phi:       &'a [f64],
    /// Physical-space gradients, row-major `[n_dofs × dim]`:
    /// `grad_phys[i * dim + j] = ∂φᵢ/∂xⱼ`.
    pub grad_phys: &'a [f64],
    /// Physical coordinates of this quadrature point; length `dim`.
    pub x_phys:    &'a [f64],
    /// Element index (for piecewise coefficients).
    pub elem_id:   ElemId,
    /// Element material / region tag (from mesh physical groups).
    pub elem_tag:  i32,
    /// Global DOF indices for this element (for [`GridFunctionCoeff`]).
    pub elem_dofs: Option<&'a [u32]>,
}

/// Accumulate a bilinear-form contribution into the element stiffness matrix.
///
/// `k_elem` is row-major with shape `[n_dofs × n_dofs]`.
///
/// Implementors must **add** their contribution (not overwrite), as multiple
/// integrators may share the same element matrix.
pub trait BilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]);
}

/// Accumulate a linear-form contribution into the element load vector.
///
/// `f_elem` has length `n_dofs`.
///
/// Implementors must **add** their contribution.
pub trait LinearIntegrator: Send + Sync {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]);
}

// ─── Boundary (face) integrals ────────────────────────────────────────────────

/// Data available to boundary integrators at each face quadrature point.
#[derive(Debug)]
pub struct BdQpData<'a> {
    /// Number of local DOFs on this face.
    pub n_dofs:  usize,
    /// Spatial dimension of the embedding space.
    pub dim:     usize,
    /// Effective integration weight: quadrature weight × face Jacobian (length in 2-D, area in 3-D).
    pub weight:  f64,
    /// Basis function values at this quadrature point; length `n_dofs`.
    pub phi:     &'a [f64],
    /// Physical coordinates of this quadrature point; length `dim`.
    pub x_phys:  &'a [f64],
    /// Outward unit normal to the face; length `dim`.
    pub normal:  &'a [f64],
    /// Element index that owns this boundary face.
    pub elem_id: ElemId,
    /// Element material / region tag.
    pub elem_tag: i32,
}

/// Accumulate a boundary linear-form contribution into a face load vector.
///
/// `f_face` has length `n_dofs` (number of DOFs on the face).
pub trait BoundaryLinearIntegrator: Send + Sync {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]);
}

/// Accumulate a boundary bilinear-form contribution into a face stiffness matrix.
///
/// `k_face` is row-major with shape `[n_dofs × n_dofs]`.
///
/// Implementors must **add** their contribution (not overwrite).
pub trait BoundaryBilinearIntegrator: Send + Sync {
    fn add_to_face_matrix(&self, qp: &BdQpData<'_>, k_face: &mut [f64]);
}
