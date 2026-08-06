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
    /// Physical integration weight: quadrature weight × |det J|.
    ///
    /// Unlike [`weight`](Self::weight) (which follows the DiffusionIntegrator
    /// MFEM convention `ip.weight / |det J|` on the non-affine path), this is
    /// always the physical measure: `quadrature weight × |det J|`.  Use this
    /// for integrators whose integrand is a physical volume form (e.g.
    /// [`MassIntegrator`](crate::standard::MassIntegrator), `∫ ρ u v dΩ`).
    pub phys_weight: f64,
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

    /// Optional per-integrator quadrature order (MFEM semantics: each
    /// integrator selects its own `IntRules` order via `GetIntegrationOrder`).
    ///
    /// Returns `None` to use the assembler's global `quad_order`.  When any
    /// integrator in a form returns `Some(order)`, the assembler evaluates
    /// that integrator on its own quadrature rule of the requested order
    /// (exact polynomials of degree ≤ `order`), independent of the others.
    fn integration_order(&self, _space_order: u8) -> Option<u8> { None }
}

/// Accumulate a linear-form contribution into the element load vector.
///
/// `f_elem` has length `n_dofs`.
///
/// Implementors must **add** their contribution.
pub trait LinearIntegrator: Send + Sync {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]);

    /// Optional per-integrator quadrature order (MFEM semantics: each
    /// integrator selects its own `IntRules` order via `GetIntegrationOrder`).
    ///
    /// Returns `None` to use the assembler's global `quad_order`.
    fn integration_order(&self, _space_order: u8) -> Option<u8> { None }
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

/// Boundary mass bilinear integrator: `∫ κ · φᵢ · φⱼ ds`.
///
/// Adds the Robin / mass contribution on boundary faces.
/// Combine with a [`RobinLFIntegrator`] for the full Robin BC.
pub struct BoundaryMassIntegrator {
    pub kappa: f64,
    pub bdr_tags: Vec<i32>,
}

impl BoundaryBilinearIntegrator for BoundaryMassIntegrator {
    fn add_to_face_matrix(&self, qp: &BdQpData<'_>, k_face: &mut [f64]) {
        let n = qp.n_dofs;
        let w = qp.weight;
        for i in 0..n {
            let phi_i = qp.phi[i];
            for j in 0..n {
                k_face[i * n + j] += w * self.kappa * phi_i * qp.phi[j];
            }
        }
    }
}

/// Robin boundary linear integrator: `∫ (g − κ · u_D) · φᵢ ds`.
///
/// The RHS contribution of a Robin BC: `∂u/∂n + κ·u = g` on the boundary.
/// The bilinear part `∫ κ·u·v ds` should be added via a
/// [`BoundaryMassIntegrator`] on the same boundary tags.
pub struct RobinLFIntegrator {
    pub kappa: f64,
    #[allow(clippy::type_complexity)]
    pub u_bdr: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    #[allow(clippy::type_complexity)]
    pub g: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    pub bdr_tags: Vec<i32>,
}

impl BoundaryLinearIntegrator for RobinLFIntegrator {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let n = qp.n_dofs;
        let w = qp.weight;
        let x = qp.x_phys;
        let ud = (self.u_bdr)(x);
        let gv = (self.g)(x);
        let source = gv - self.kappa * ud;
        for i in 0..n {
            f_face[i] += w * source * qp.phi[i];
        }
    }
}

// ─── Per-integrator quadrature order wrapper ─────────────────────────────────

/// Wrap a bilinear integrator and force its quadrature order (MFEM semantics:
/// e.g. `CurlCurlIntegrator` on Pk spaces uses order `2p − 2`).
///
/// When an integrator wrapped with this is passed to
/// [`Assembler::assemble_bilinear`] / [`VectorAssembler::assemble_bilinear`],
/// it is evaluated on a quadrature rule of exactly `order` (exact for
/// polynomials of degree ≤ `order`), independent of the form's global
/// `quad_order`.
///
/// ```rust,ignore
/// // Force the curl-curl (z-block) diffusion contribution to MFEM's
/// // `IntRules.Get(SQUARE, 2p-2)` = order-0 single-point rule:
/// let integ = FixedOrder::new(DiffusionIntegrator { kappa: 1.0 }, 0);
/// ```
pub struct FixedOrder<I> {
    inner: I,
    order: u8,
}

impl<I> FixedOrder<I> {
    pub fn new(inner: I, order: u8) -> Self {
        Self { inner, order }
    }

    /// Unwrap to the inner integrator.
    pub fn into_inner(self) -> I { self.inner }
}

impl<I: BilinearIntegrator> BilinearIntegrator for FixedOrder<I> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        self.inner.add_to_element_matrix(qp, k_elem);
    }

    fn integration_order(&self, _space_order: u8) -> Option<u8> { Some(self.order) }
}

impl<I: LinearIntegrator> LinearIntegrator for FixedOrder<I> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        self.inner.add_to_element_vector(qp, f_elem);
    }

    fn integration_order(&self, _space_order: u8) -> Option<u8> { Some(self.order) }
}

impl<I: crate::vector_integrator::VectorBilinearIntegrator>
    crate::vector_integrator::VectorBilinearIntegrator for FixedOrder<I>
{
    fn add_to_element_matrix(&self, qp: &crate::vector_integrator::VectorQpData<'_>, k_elem: &mut [f64]) {
        self.inner.add_to_element_matrix(qp, k_elem);
    }

    fn integration_order(&self, _space_order: u8) -> Option<u8> { Some(self.order) }
}

impl<I: crate::vector_integrator::VectorLinearIntegrator>
    crate::vector_integrator::VectorLinearIntegrator for FixedOrder<I>
{
    fn add_to_element_vector(&self, qp: &crate::vector_integrator::VectorQpData<'_>, f_elem: &mut [f64]) {
        self.inner.add_to_element_vector(qp, f_elem);
    }

    fn integration_order(&self, _space_order: u8) -> Option<u8> { Some(self.order) }
}
