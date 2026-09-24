//! Vector mass bilinear form integrators for H(curl) / H(div) spaces.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω α u · v dx          (isotropic, α scalar)
//! a(u, v) = ∫_Ω (A u) · v dx        (anisotropic, A tensor dim×dim)
//! ```
//!
//! where `u` and `v` are vector-valued basis functions.

use crate::postproc::coefficient::{CoeffCtx, MatrixCoeff, ScalarCoeff, ScalarMatrixCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

// ─── Isotropic ───────────────────────────────────────────────────────────────

/// Bilinear integrator for the isotropic vector mass operator `α u·v`.
///
/// `α` is a **scalar** coefficient.  Use [`VectorMassTensorIntegrator`] for
/// anisotropic (tensor-valued) permittivity or other material tensors.
///
/// Used in Maxwell cavity eigenvalue problems (`∇×∇×E + E = f`)
/// and as the B-matrix in H(div) mixed formulations.
pub struct VectorMassIntegrator<C: ScalarCoeff = f64> {
    /// Scalar mass coefficient (α).
    pub alpha: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for VectorMassIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w_a = qp.weight * self.alpha.eval(&ctx);

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..d {
                    dot += qp.phi_vec[i * d + c] * qp.phi_vec[j * d + c];
                }
                k_elem[i * n + j] += w_a * dot;
            }
        }
    }

    /// MFEM `VectorFEMassIntegrator` (bilininteg.cpp): `order =
    /// Trans.OrderW() + 2*el.GetOrder()`, where the RT/ND element order is
    /// `GetOrder() = k + 1` (not the collection order `k`) and `OrderW()` is
    /// geometry-dependent: 0 (affine simplex), 1 (affine quad), 2 (affine
    /// hex).  A single geometry-free value must cover all reference
    /// geometries, so return the affine-quad value `2k + 3`: it is bit-matching
    /// on quads, and on simplices/hexes it over-integrates by one order — the
    /// integral itself is exact, so only summation round-off differs.
    /// Under-integrating (the previous `1 + 2k`) made the local RT mass blocks
    /// rank-deficient (quad RT0 with a 1×1 rule: diag = pair coupling = 1/4),
    /// i.e. a globally singular mass matrix.
    fn integration_order(&self, space_order: u8) -> Option<u8> {
        Some(2 * space_order + 3)
    }

    /// D680: family-aware MFEM default.  The blind `2k + 3` above cannot
    /// reproduce MFEM on hexes for both families at once — `GetOrder()` is
    /// `k + 1` for RT but `k` for ND, and affine-hex `OrderW() = 2`, so MFEM's
    /// RT0 hex default is order 4 (a 27-point rule) where `2k + 3` picks the
    /// 8-point rule.  Both rules integrate the RT0 mass polynomial exactly,
    /// but the summation round-off differs and leaks into noise-level solve
    /// outputs (ex24 `-p 1` iteration residuals).  Quad RT (`1 + 2·(k+1)` =
    /// `2k + 3`) and every ND geometry (same 1D point count as `2k + 3`) keep
    /// their previous rules bit-for-bit.
    fn integration_order_for_space(
        &self,
        space_type: fem_space::fe_space::SpaceType,
        space_order: u8,
        elem_type: fem_mesh::element_type::ElementType,
    ) -> Option<u8> {
        use fem_space::fe_space::SpaceType;
        let get_order = match space_type {
            // fe_rt.cpp: every RT_*Element(p) ctor passes `p + 1` as the FE
            // order; fe_nd.cpp: every ND_*Element(p) passes `p`.
            SpaceType::HDiv => space_order + 1,
            SpaceType::HCurl => space_order,
            _ => return self.integration_order(space_order),
        };
        Some(mfem_vector_mass_order_w(elem_type) + 2 * get_order)
    }

    /// D690: the order hook must read `OrderW()` off the **element
    /// transformation**, not the connectivity.  A hex refined from a curved
    /// parent carries a quadratic map (`Trans.OrderW() = 3g − 1 = 5` at g = 2)
    /// while its connectivity stays `Hex8`, so MFEM's
    /// `VectorFEMassIntegrator` assembles it at order `5 + 2·GetOrder`
    /// (multidomain_rt cylinder: 9) where the connectivity-only hook above
    /// picked the affine value 2 (order 6).  Curved counterparts reuse the
    /// same table (Hex27→5, Quad9→3, Tri6→2, Tet10→3, Prism18→5,
    /// Pyramid13→5); affine meshes (geom_order ≤ 1) behave exactly as
    /// before, bit-for-bit.
    fn integration_order_for_space_geom(
        &self,
        space_type: fem_space::fe_space::SpaceType,
        space_order: u8,
        elem_type: fem_mesh::element_type::ElementType,
        geom_order: u8,
    ) -> Option<u8> {
        use fem_mesh::element_type::ElementType;
        use fem_space::fe_space::SpaceType;
        let curved = |et: ElementType| match (et, geom_order) {
            (ElementType::Hex8, g) if g > 1 => ElementType::Hex27,
            (ElementType::Quad4, g) if g > 1 => ElementType::Quad9,
            (ElementType::Tri3, g) if g > 1 => ElementType::Tri6,
            (ElementType::Tet4, g) if g > 1 => ElementType::Tet10,
            (ElementType::Prism6, g) if g > 1 => ElementType::Prism18,
            (ElementType::Pyramid5, g) if g > 1 => ElementType::Pyramid13,
            (et, _) => et,
        };
        let get_order = match space_type {
            SpaceType::HDiv => space_order + 1,
            SpaceType::HCurl => space_order,
            _ => return self.integration_order(space_order),
        };
        Some(mfem_vector_mass_order_w(curved(elem_type)) + 2 * get_order)
    }
}

/// MFEM `IsoparametricTransformation::OrderW()` (eltrans.cpp:493) for the
/// geometry carried by `elem_type`: `(g−1)·dim` for the simplex Pk geometry
/// maps, `g·dim − 1` for the quad/hex/prism Qk maps and the pyramid Uk map,
/// with the geometry order `g` implied by the connectivity type (straight
/// elements `g = 1`, their quadratic counterparts `g = 2`).
fn mfem_vector_mass_order_w(elem_type: fem_mesh::element_type::ElementType) -> u8 {
    use fem_mesh::element_type::ElementType;
    match elem_type {
        ElementType::Point1 => 0,
        ElementType::Line2 | ElementType::Line3 => 0,
        ElementType::Tri3 => 0,
        ElementType::Tri6 => 2,
        ElementType::Quad4 => 1,
        ElementType::Quad8 | ElementType::Quad9 => 3,
        ElementType::Tet4 => 0,
        ElementType::Tet10 => 3,
        ElementType::Hex8 => 2,
        ElementType::Hex20 | ElementType::Hex27 => 5,
        ElementType::Prism6 => 2,
        ElementType::Prism15 | ElementType::Prism18 => 5,
        ElementType::Pyramid5 => 2,
        ElementType::Pyramid13 => 5,
        // VEM polygonal cells have no MFEM isoparametric counterpart; the
        // affine-frame value 1 keeps the arm total without inventing MFEM
        // semantics (no RT/ND space exists on `Polygon` today).
        ElementType::Polygon => 1,
    }
}

// ─── Anisotropic (tensor) ────────────────────────────────────────────────────/// Bilinear integrator for the anisotropic vector mass operator `(A u)·v`.
///
/// `A` is a **matrix** coefficient (dim×dim, row-major).  This handles:
/// - Anisotropic electric permittivity tensor ε (electromagnetic problems)
/// - Anisotropic mass density tensors (structural dynamics)
/// - General symmetric positive-definite tensor weights
///
/// # Formula
///
/// ```text
/// a(u, v) = ∫_Ω (A(x) u) · v dx   where A is dim×dim at each point
/// ```
///
/// # Example (anisotropic permittivity)
/// ```rust,ignore
/// use fem_assembly::coefficient::ConstantMatrixCoeff;
/// use fem_assembly::standard::VectorMassTensorIntegrator;
/// // ε_r = diag(2, 1) — uniaxial medium
/// let integ = VectorMassTensorIntegrator {
///     alpha: ConstantMatrixCoeff(vec![2.0, 0.0, 0.0, 1.0]),
/// };
/// ```
pub struct VectorMassTensorIntegrator<C: MatrixCoeff> {
    /// Matrix mass coefficient (dim×dim, row-major).
    pub alpha: C,
}

impl<C: MatrixCoeff> VectorMassTensorIntegrator<C> {
    /// Construct from a matrix coefficient.
    ///
    /// MFEM: `new VectorFEMassIntegrator(MatrixCoefficient &mq)`.
    pub fn new(alpha: C) -> Self {
        VectorMassTensorIntegrator { alpha }
    }
}

impl VectorMassTensorIntegrator<ScalarMatrixCoeff<f64>> {
    /// The unweighted (`σ = I`) vector mass operator.
    ///
    /// MFEM: `new VectorFEMassIntegrator()` (null coefficient → 1.0).  Also
    /// reachable as `VectorFEMassIntegrator::identity()`.
    pub fn identity() -> Self {
        VectorMassTensorIntegrator { alpha: ScalarMatrixCoeff(1.0) }
    }
}

/// **MFEM-compatible name**: `VectorFEMassIntegrator` = the general
/// `(σ u, v)` vector mass integrator for H(curl) / H(div) spaces, where `σ` is
/// a matrix (tensor) coefficient or — through [`ScalarMatrixCoeff`] — a scalar
/// one.
///
/// Alias of [`VectorMassTensorIntegrator`] (the math lives there):
///
/// ```rust,ignore
/// use fem_assembly::coefficient::{ConstantMatrixCoefficient, ScalarMatrixCoeff};
/// use fem_assembly::standard::VectorFEMassIntegrator;
///
/// // MFEM: MatrixConstantCoefficient sigma(sigmaMat);
/// //       a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
/// let sigma = ConstantMatrixCoefficient::diag(&[2.0, 1.0]);
/// let integ = VectorFEMassIntegrator::new(sigma);
///
/// // Scalar coefficient (MFEM VectorFEMassIntegrator(Coefficient&)) → α·I:
/// let integ = VectorFEMassIntegrator { alpha: ScalarMatrixCoeff(2.0) };
///
/// // No coefficient (σ = I):
/// let integ = VectorFEMassIntegrator::identity();
/// ```
pub type VectorFEMassIntegrator<C = ScalarMatrixCoeff<f64>> =
    VectorMassTensorIntegrator<C>;

impl<C: MatrixCoeff> VectorBilinearIntegrator for VectorMassTensorIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n   = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, dim, qp.elem_id, qp.elem_tag,
            None, None,
        );
        let w = qp.weight;

        // Evaluate tensor A into a local stack buffer (max 3×3 = 9).
        let mut a_buf = [0.0_f64; 9];
        self.alpha.eval(&ctx, &mut a_buf[..dim * dim]);

        for i in 0..n {
            // A φᵢ: matrix-vector product, result in `au` (length dim).
            let mut au = [0.0_f64; 3];
            for r in 0..dim {
                for c in 0..dim {
                    au[r] += a_buf[r * dim + c] * qp.phi_vec[i * dim + c];
                }
            }
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += au[c] * qp.phi_vec[j * dim + c];
                }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}

/// Boundary mass integrator for H(curl) / H(div) spaces.
/// Computes `∫_Γ α u · v dS` on boundary faces.
pub struct VectorBoundaryMassIntegrator<C: ScalarCoeff = f64> {
    pub alpha: C,
}

use crate::boundary::vector_boundary::{VectorBdQpData, VectorBoundaryBilinearIntegrator};

impl<C: ScalarCoeff> VectorBoundaryBilinearIntegrator for VectorBoundaryMassIntegrator<C> {
    fn add_to_face_matrix(&self, qp: &VectorBdQpData<'_>, k_face: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.alpha.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += qp.phi_vec[i * dim + c] * qp.phi_vec[j * dim + c];
                }
                k_face[i * n + j] += w * dot;
            }
        }
    }
}
