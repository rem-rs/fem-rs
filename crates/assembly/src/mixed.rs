//! Mixed bilinear form assembler and coupling integrators.
//!
//! A **mixed bilinear form** acts on two potentially different spaces:
//! `b(u, v)` where `u ∈ U` and `v ∈ V`.  The assembled matrix is rectangular
//! with `n_V` rows and `n_U` columns.
//!
//! # Typical usage (Stokes pressure-velocity coupling)
//! ```rust,ignore
//! // B = ∫ div(u) p dx   →   B is n_p × n_u
//! let b = MixedAssembler::assemble_bilinear(
//!     &pressure_space,   // trial space V (row space)
//!     &velocity_space,   // test space  U (col space)
//!     &[&PressureDivIntegrator],
//!     3,
//! );
//! ```

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, VectorReferenceElement, lagrange::{TetP1, TetP2, TetP3, TriP1, TriP3, QuadQ1, QuadQ2, QuadQk, HexQ1, HexQ2, HexQ3}, lagrange::factory::TriPk, serendipity::{QuadSerendipityPk, HexSerendipityPk}};
use fem_element::raviart_thomas::{QuadRT0, QuadRT1, TriRT0, TriRT1, TetRT0, TetRT1, HexRT0, HexRT1, PrismRTk};
use fem_element::nedelec::{TriND1, TetND1, QuadND1, QuadNDk, HexND1, HexNDk, PrismND1, PrismNDk};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use crate::vector_assembler::{isoparametric_jacobian, geo_ref_elem_from_mesh};
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{HCurlSpace, H1Space, HDivSpace, L2Space};

use crate::integrator::QpData;
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

use crate::assembler::ref_elem_vol_for_space;
#[cfg(feature = "parallel")]
use crate::assembler::assembly_parallel_min_elems;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ─── MixedBilinearIntegrator ──────────────────────────────────────────────────

/// An integrator for a mixed bilinear form `b(u, v)`.
///
/// At each quadrature point, `add_to_element_matrix` accumulates into a
/// rectangular element matrix of shape `n_row_dofs × n_col_dofs`.
///
/// - Row index corresponds to the **row space** (first/left space, e.g. pressure).
/// - Column index corresponds to the **column space** (second/right space, e.g. velocity).
pub trait MixedBilinearIntegrator: Send + Sync {
    /// Accumulate into `m_elem` (row-major, `n_row_dofs × n_col_dofs`).
    ///
    /// `qp_row` and `qp_col` carry basis/gradient data for the row and column
    /// spaces respectively.  Only `phi`, `grad_phys`, `weight`, `x_phys`, `dim`
    /// from `qp_col` are used; `n_dofs` from each reflects the respective space.
    fn add_to_element_matrix(
        &self,
        qp_row: &QpData<'_>,
        qp_col: &QpData<'_>,
        m_elem: &mut [f64],
    );
}

// ─── Built-in mixed integrators ───────────────────────────────────────────────

/// `b(u, p) = -∫ p (∇·u) dx` — velocity-pressure coupling for incompressible flows.
///
/// Row space = pressure (L²/H¹), column space = velocity ([H¹]^d).
///
/// The assembled matrix `B` satisfies `B[j, i] += w * p_j * (∇·u_i)`.
/// Since the velocity DOFs are interleaved by component, the divergence is
/// computed from the velocity basis gradients.
pub struct PressureDivIntegrator;

impl MixedBilinearIntegrator for PressureDivIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_row: &QpData<'_>,  // pressure (scalar)
        qp_col: &QpData<'_>,  // velocity (vector, interleaved)
        m_elem: &mut [f64],
    ) {
        let n_p   = qp_row.n_dofs;
        let n_u   = qp_col.n_dofs;
        let dim   = qp_col.dim;
        let w     = qp_col.weight;
        let n_nodes_u = n_u / dim;

        for j in 0..n_p {
            let pj = qp_row.phi[j];
            for k in 0..n_nodes_u {
                for c in 0..dim {
                    let col = k * dim + c; // interleaved velocity DOF (k, c)
                    // Only component c contributes to ∂u^{k,c}/∂x_c.
                    let div_ukc = qp_col.grad_phys[k * dim + c];
                    m_elem[j * n_u + col] += -w * pj * div_ukc;
                }
            }
        }
    }
}

/// `b(v, u) = ∫ v · u dx` — scalar mass coupling (MFEM `MixedScalarMassIntegrator`).
///
/// Row space = test space (e.g. L²), column space = trial space (e.g. H¹).
/// The assembled matrix satisfies `B[j, i] += w * φ_row_j * φ_col_i`.
pub struct ScalarMassIntegrator;

impl MixedBilinearIntegrator for ScalarMassIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_row: &QpData<'_>,
        qp_col: &QpData<'_>,
        m_elem: &mut [f64],
    ) {
        let n_r = qp_row.n_dofs;
        let n_c = qp_col.n_dofs;
        let w = qp_row.weight;
        for j in 0..n_r {
            let pj = qp_row.phi[j];
            for i in 0..n_c {
                m_elem[j * n_c + i] += w * pj * qp_col.phi[i];
            }
        }
    }
}

/// `b(u, p) = ∫ (∇·u) p dx` — positive sign variant, also useful for Darcy.
pub struct DivIntegrator;

impl MixedBilinearIntegrator for DivIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_row: &QpData<'_>,
        qp_col: &QpData<'_>,
        m_elem: &mut [f64],
    ) {
        let n_p  = qp_row.n_dofs;
        let n_u  = qp_col.n_dofs;
        let dim  = qp_col.dim;
        let w    = qp_col.weight;
        let n_nodes_u = n_u / dim;

        for j in 0..n_p {
            let pj = qp_row.phi[j];
            for k in 0..n_nodes_u {
                for c in 0..dim {
                    let col = k * dim + c;
                    let div_ukc = qp_col.grad_phys[k * dim + c];
                    m_elem[j * n_u + col] += w * pj * div_ukc;
                }
            }
        }
    }
}

// ─── MixedAssembler ──────────────────────────────────────────────────────────

/// Stateless driver for mixed bilinear form assembly.
///
/// With the **`parallel`** feature, volume assembly uses Rayon when
/// `row_space.mesh().n_elements() >= assembly_parallel_min_elems()` (same
/// threshold and env `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` as [`crate::Assembler`]).
pub struct MixedAssembler;

impl MixedAssembler {
    /// Assemble a mixed bilinear form `b(u, v)` into a rectangular `CsrMatrix`.
    ///
    /// - `row_space` — the "row" / "test" space (V); determines number of rows.
    /// - `col_space` — the "col" / "trial" space (U); determines number of columns.
    /// - Both spaces must be defined on the same mesh.
    ///
    /// # Returns
    /// A `CsrMatrix` with `row_space.n_dofs()` rows and `col_space.n_dofs()` cols.
    pub fn assemble_bilinear<SR, SC>(
        row_space:   &SR,
        col_space:   &SC,
        integrators: &[&dyn MixedBilinearIntegrator],
        quad_order:  u8,
    ) -> CsrMatrix<f64>
    where
        SR: FESpace,
        SC: FESpace,
    {
        let mesh = row_space.mesh();
        let n_rows = row_space.n_dofs();
        let n_cols = col_space.n_dofs();

        #[cfg(feature = "parallel")]
        {
            if mesh.n_elements() >= assembly_parallel_min_elems() {
                return assemble_mixed_bilinear_volume_parallel(
                    row_space,
                    col_space,
                    integrators,
                    quad_order,
                );
            }
        }

        let mut coo = CooMatrix::<f64>::new(n_rows, n_cols);
        for e in mesh.elem_iter() {
            accumulate_mixed_volume_element(
                row_space,
                col_space,
                e,
                integrators,
                quad_order,
                &mut coo,
            );
        }
        coo.into_csr()
    }
}

// ─── HDiv × L² mixed assembly (Darcy coupling) ──────────────────────────

/// Integrator for HDiv×L² mixed bilinear form `b(v, p) = ∫ p · α·div(v) dx`.
pub trait HDivL2Integrator: Send + Sync {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        div_col: &[f64],
        dim: usize,
        m_elem: &mut [f64],
    );
}

/// ∫ p · div(v) dx — divergence coupling for Darcy/Stokes.
pub struct HDivL2DivIntegrator;

impl HDivL2Integrator for HDivL2DivIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        div_col: &[f64],
        _dim: usize,
        m_elem: &mut [f64],
    ) {
        let n_r = qp_scalar.n_dofs;
        let n_c = div_col.len();
        let w = qp_scalar.weight;
        for j in 0..n_r {
            let pj = qp_scalar.phi[j]; // pressure basis
            for i in 0..n_c {
                m_elem[j * n_c + i] += w * pj * div_col[i];
            }
        }
    }
}

/// ∫ α(elem) · p · div(v) dx — scaled divergence coupling.
///
/// The coefficient `alpha` is evaluated at each quadrature point (element
/// tag-aware via [`CoeffCtx`]), enabling material-dependent scaling such
/// as `1/κ` (inverse thermal conductivity) or `1/c` (inverse heat capacity).
///
/// MFEM equivalent: `VectorFEDivergenceIntegrator(coeff)` on
/// `MixedBilinearForm(HDiv, L2)`.
pub struct HDivL2ScaledDiv<C: ScalarCoeff = f64> {
    pub alpha: C,
}

impl<C: ScalarCoeff> HDivL2Integrator for HDivL2ScaledDiv<C> {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        div_col: &[f64],
        _dim: usize,
        m_elem: &mut [f64],
    ) {
        let ctx = CoeffCtx::from_qp(
            qp_scalar.x_phys, qp_scalar.dim, qp_scalar.elem_id,
            qp_scalar.elem_tag, None, None,
        );
        let coeff = self.alpha.eval(&ctx);
        let n_r = qp_scalar.n_dofs;
        let n_c = div_col.len();
        let w = qp_scalar.weight * coeff;
        for j in 0..n_r {
            let pj = qp_scalar.phi[j];
            for i in 0..n_c {
                m_elem[j * n_c + i] += w * pj * div_col[i];
            }
        }
    }
}

/// Assemble HDiv × L² mixed bilinear form.
pub fn assemble_hdiv_l2_mixed<SR, SC>(
    row_space: &SR,   // L² (pressure)
    col_space: &SC,   // HDiv (velocity)
    integrators: &[&dyn HDivL2Integrator],
    quad_order: u8,
) -> CsrMatrix<f64>
where
    SR: FESpace,
    SC: FESpace,
{
    let mesh = row_space.mesh();
    let dim = mesh.dim() as usize;
    let n_rows = row_space.n_dofs();
    let n_cols = col_space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_rows, n_cols);

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let order_c = col_space.order();
        let ref_c = ref_elem_vec(elem_type, order_c, SpaceType::HDiv).unwrap();
        let n_c = ref_c.n_dofs();

        let ref_r = ref_elem_vol_for_space(row_space, elem_type, row_space.order());
        let n_r = ref_r.n_dofs();

        // Use the vector element's quadrature (P0 has no quadrature).
        let quad = ref_c.quadrature(quad_order);

        let global_rows: Vec<usize> = row_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_cols: Vec<usize> = col_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let elem_tag = mesh.element_tag(e);

        // Isoparametric geometry (e.g. Tri6 from SetCurvature(2)): the
        // Jacobian must be built from the mesh's geometry nodes and the
        // matching reference element; otherwise use the affine P1 simplex
        // transformation.  Non-simplex elements (Quad4, Hex8, ...) always use
        // the isoparametric/affine-mapped path — `from_simplex_nodes` is only
        // valid for simplices.
        let use_iso = mesh.geom_order() > 1
            || !matches!(
                elem_type,
                ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2
            );
        let geo_elem = geo_ref_elem_from_mesh(mesh, e);

        // Apply H(div) orientation signs (contravariant Piola sign) from
        // the velocity space — VectorAssembler does this for the mass matrix,
        // so the mixed divergence matrix must match.
        let signs_opt = col_space.element_signs(e);

        let n_elem_r = global_rows.len();
        let n_elem_c = global_cols.len();
        let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];

        let mut phi_r = vec![0.0; n_r];
        let mut div_c_vec = vec![0.0; n_c];
        let mut div_c_signed = vec![0.0; n_c];

        for (q, xi) in quad.points.iter().enumerate() {
            let (jac_q, det_j, xp) = if use_iso {
                let ge = geo_elem
                    .as_ref()
                    .expect("missing geometry reference element for isoparametric mixed assembly");
                let geo_nds = mesh.geometry_nodes(e);
                isoparametric_jacobian(mesh, geo_nds, ge.as_ref(), xi, dim)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(xi))
            };
            let w = quad.weights[q] * det_j.abs();

            ref_r.eval_basis(xi, &mut phi_r);
            ref_c.eval_div(xi, &mut div_c_vec);

            // Apply orientation signs and the contravariant Piola transform:
            // phys_div = ref_div / detJ  (since the Piola transform for H(div)
            // has div_phys = div_ref / detJ).  The quadrature weight already
            // includes |detJ|, so the product w * div_phys = q_weight * div_ref.
            if let Some(signs) = signs_opt {
                for i in 0..n_c.min(signs.len()) {
                    div_c_signed[i] = signs[i] * div_c_vec[i] / det_j;
                }
                for i in n_c.min(signs.len())..n_c {
                    div_c_signed[i] = div_c_vec[i] / det_j;
                }
            } else {
                for i in 0..n_c {
                    div_c_signed[i] = div_c_vec[i] / det_j;
                }
            }

            let qp_r = QpData {
                n_dofs: n_elem_r,
                dim,

                weight: w,
                phys_weight: w,
                ref_weight: quad.weights[q],
                phi: &phi_r,
                grad_phys: &[], // not needed
                x_phys: &xp,
                elem_id: e,
                elem_tag,
                elem_dofs: None,
            };

            for integ in integrators {
                integ.add_to_element_matrix(&qp_r, &div_c_signed, dim, &mut m_elem);
            }
        }

        for (ir, &gr) in global_rows.iter().enumerate() {
            for (ic, &gc) in global_cols.iter().enumerate() {
                coo.add(gr, gc, m_elem[ir * n_elem_c + ic]);
            }
        }
    }

    coo.into_csr()
}

// ─── HCurl × H¹ mixed assembly (gauge fixing / potential coupling) ─────

/// Integrator for HCurl×H¹ mixed bilinear form `b(E, φ) = ∫ φ · α·curl(E) dx`.
pub trait HCurlH1Integrator: Send + Sync {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        curl_col: &[f64],
        dim: usize,
        m_elem: &mut [f64],
    );
}

/// ∫ φ · curl(E) dx — curl coupling for Maxwell gauge fixing.
pub struct HCurlH1CurlIntegrator;

impl HCurlH1Integrator for HCurlH1CurlIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        curl_col: &[f64],
        dim: usize,
        m_elem: &mut [f64],
    ) {
        let n_r = qp_scalar.n_dofs;
        let n_c = curl_col.len() / dim; // per-DOF curl length
        let w = qp_scalar.weight;
        for j in 0..n_r {
            let pj = qp_scalar.phi[j];
            for i in 0..n_c {
                let curl_z = curl_col[i * dim + dim - 1];
                m_elem[j * n_c + i] += w * pj * curl_z;
            }
        }
    }
}

/// Assemble HCurl × H¹ mixed bilinear form.
pub fn assemble_hcurl_h1_mixed<SR, SC>(
    row_space: &SR,   // H¹ (scalar potential)
    col_space: &SC,   // HCurl (vector field)
    integrators: &[&dyn HCurlH1Integrator],
    quad_order: u8,
) -> CsrMatrix<f64>
where
    SR: FESpace,
    SC: FESpace,
{
    let mesh = row_space.mesh();
    let dim = mesh.dim() as usize;
    let n_rows = row_space.n_dofs();
    let n_cols = col_space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_rows, n_cols);

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let order_c = col_space.order();
        let ref_c = ref_elem_vec(elem_type, order_c, SpaceType::HCurl).unwrap();
        let n_c = ref_c.n_dofs();
        let ref_r = ref_elem_vol(elem_type, row_space.order()).unwrap();
        let n_r = ref_r.n_dofs();
        let quad = ref_r.quadrature(quad_order);

        let global_rows: Vec<usize> = row_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_cols: Vec<usize> = col_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let elem_tag = mesh.element_tag(e);
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);

        let n_elem_r = global_rows.len();
        let n_elem_c = global_cols.len();
        let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];
        let mut phi_r = vec![0.0; n_r];
        let mut curl_c_vec = vec![0.0; n_c * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * tr.det_j().abs();
            ref_r.eval_basis(xi, &mut phi_r);
            ref_c.eval_curl(xi, &mut curl_c_vec);
            let xp = tr.map_to_physical(xi);
            let qp_r = QpData {
                n_dofs: n_elem_r, dim, weight: w, phys_weight: w,
                ref_weight: quad.weights[q], phi: &phi_r, grad_phys: &[],
                x_phys: &xp, elem_id: e, elem_tag, elem_dofs: None,
            };
            for integ in integrators {
                integ.add_to_element_matrix(&qp_r, &curl_c_vec, dim, &mut m_elem);
            }
        }
        for (ir, &gr) in global_rows.iter().enumerate() {
            for (ic, &gc) in global_cols.iter().enumerate() {
                coo.add(gr, gc, m_elem[ir * n_elem_c + ic]);
            }
        }
    }
    coo.into_csr()
}

/// Assemble HCurl × H¹ mixed GRADIENT bilinear form (transpose of the existing curl pairing).
///
/// Computes `B[i,j] = ∫ ∇φ_j · ψ_i dx` where:
/// - Row space (test):  H(curl) — Nédélec basis `ψ_i`
/// - Column space (trial): H¹ — scalar Lagrange basis `φ_j`
///
/// This is the transpose of `assemble_hcurl_h1_mixed`: here HCurl is the ROW,
/// not the column.  Uses `eval_basis_vec` (not `eval_curl`) to evaluate the
/// HCurl basis vectors at each quadrature point.
pub fn assemble_hcurl_h1_gradient<M: fem_mesh::topology::MeshTopology + Clone + 'static>(
    nd_space: &HCurlSpace<M>,
    h1_space: &H1Space<M>,
    quad_order: u8,
) -> CsrMatrix<f64>
where
    M: fem_mesh::topology::MeshTopology,
{
    use fem_element::ReferenceElement;
    use fem_linalg::CooMatrix;

    let mesh = h1_space.mesh();
    let edim = mesh.dim() as usize;
    let tdim = mesh.topological_dim() as usize;
    let is_surface = edim != tdim;
    let dim = edim;
    let n_rows = nd_space.n_dofs();
    let n_cols = h1_space.n_dofs();
    let mut coo = CooMatrix::new(n_rows, n_cols);

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let h1_ref = crate::assembler::ref_elem_vol(elem_type, h1_space.order());
        let n_h1 = h1_ref.n_dofs();

        let nd_ref = ref_elem_vec(elem_type, nd_space.order(), SpaceType::HCurl)
            .expect("assemble_hcurl_h1_gradient: HCurl ref elem");
        let n_nd = nd_ref.n_dofs();
        let ref_dim = nd_ref.dim() as usize; // 2 on surfaces
        let signs = nd_space.element_signs(e);

        let global_h1: Vec<usize> = h1_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_nd: Vec<usize> = nd_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_g_h1 = global_h1.len();
        let n_g_nd = global_nd.len();
        let quad = h1_ref.quadrature(quad_order);

        let mut me = vec![0.0; n_g_nd * n_g_h1];
        let mut phi = vec![0.0; n_h1];
        let mut gr = vec![0.0; n_h1 * ref_dim];
        let mut gp = vec![0.0; n_h1 * dim];
        let mut nd_basis = vec![0.0; n_nd * ref_dim];

        let use_iso = !matches!(elem_type, fem_mesh::element_type::ElementType::Tri3 | fem_mesh::element_type::ElementType::Tet4 | fem_mesh::element_type::ElementType::Line2);
        let geo_elem = if use_iso || is_surface { crate::geo_ref_elem_from_mesh(mesh, e) } else { None };
        let nodes = mesh.element_nodes(e);

        for (qi, xi) in quad.points.iter().enumerate() {
            if is_surface {
                // ── Surface path (2-D elements embedded in 3-D) ─────────────
                // Geometry: P1 simplex (Tri3) or the mesh's isoparametric
                // geometry (Quad4), on the same reference domain as the bases.
                let geo_p1 = crate::assembler::ref_elem_vol(elem_type, 1);
                let (geo, geo_nds): (&dyn ReferenceElement, &[u32]) =
                    if let Some(ref ge) = geo_elem {
                        (ge.as_ref(), mesh.geometry_nodes(e))
                    } else {
                        (geo_p1.as_ref(), nodes)
                    };
                let (measure, j, ginv, _xp) =
                    crate::assembler::surface_jacobian(mesh, geo_nds, geo, xi, edim, tdim);
                let w = quad.weights[qi] * measure;

                h1_ref.eval_basis(xi, &mut phi);
                h1_ref.eval_grad_basis(xi, &mut gr);
                // Tangential gradient: ∇_surf φ = J·G⁻¹·∇_ref φ (3 comps)
                for k in 0..n_h1 {
                    let (g0, g1) = (gr[k * 2], gr[k * 2 + 1]);
                    let t0 = ginv[0] * g0 + ginv[1] * g1;
                    let t1 = ginv[1] * g0 + ginv[2] * g1;
                    gp[k * 3] = j[0] * t0 + j[3] * t1;
                    gp[k * 3 + 1] = j[1] * t0 + j[4] * t1;
                    gp[k * 3 + 2] = j[2] * t0 + j[5] * t1;
                }
                nd_ref.eval_basis_vec(xi, &mut nd_basis);
                // Physical ND basis: ψ = J·G⁻¹·ψ_ref (3 comps)
                for i in 0..n_g_nd {
                    let s = signs[i];
                    let (p0, p1) = (nd_basis[i * 2], nd_basis[i * 2 + 1]);
                    let t0 = ginv[0] * p0 + ginv[1] * p1;
                    let t1 = ginv[1] * p0 + ginv[2] * p1;
                    let psi0 = s * (j[0] * t0 + j[3] * t1);
                    let psi1 = s * (j[1] * t0 + j[4] * t1);
                    let psi2 = s * (j[2] * t0 + j[5] * t1);
                    for jj in 0..n_g_h1 {
                        let g_j = &gp[jj * 3..][..3];
                        me[i * n_g_h1 + jj] += w * (g_j[0] * psi0 + g_j[1] * psi1 + g_j[2] * psi2);
                    }
                }
                continue;
            }

            let (w, jit): (f64, nalgebra::DMatrix<f64>) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, _xp) = crate::isoparametric_jacobian(mesh, &nodes, ge.as_ref(), xi, dim);
                (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose())
            } else {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone())
            };
            h1_ref.eval_basis(xi, &mut phi);
            h1_ref.eval_grad_basis(xi, &mut gr);
            transform_grads(&jit, &gr, &mut gp, n_h1, dim);
            nd_ref.eval_basis_vec(xi, &mut nd_basis);

            // Physical ND basis (covariant Piola): ψ_phys = J^{-T} ψ_ref,
            // i.e. `psi[c] = Σ_k jit[(c,k)]·nd_basis[i·dim+k]`.  Works for
            // both 2-D and 3-D (dim components).
            for j in 0..n_g_h1 {
                let g_j = &gp[j * dim..][..dim];
                for i in 0..n_g_nd {
                    let s = signs[i];
                    let mut dot = 0.0;
                    for c in 0..dim {
                        let mut psi_c = 0.0;
                        for k in 0..dim {
                            psi_c += jit[(c, k)] * nd_basis[i * dim + k];
                        }
                        dot += g_j[c] * (s * psi_c);
                    }
                    me[i * n_g_h1 + j] += w * dot;
                }
            }
        }
        for (ir, &r) in global_nd.iter().enumerate() {
            for (ic, &c) in global_h1.iter().enumerate() {
                let v = me[ir * n_g_h1 + ic];
                if v != 0.0 { coo.add(r, c, v); }
            }
        }
    }
    coo.into_csr()
}

// ─── H¹ × HDiv mixed assembly (gradient-vector coupling) ───────────────────

/// Integrator for H¹×HDiv mixed bilinear form `b(φ, w) = ∫ σ ∇φ · w dx`.
///
/// Row space: H¹ (scalar), column space: HDiv (vector).
pub trait H1HDivIntegrator: Send + Sync {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,  // H¹: phi + grad_phys
        vec_col: &[f64],          // HDiv basis vector values (flat: dim × n_dofs)
        dim: usize,
        m_elem: &mut [f64],
    );
}

/// ∫ σ ∇φ · w dx — gradient-vector coupling for J = -σ∇φ projection.
pub struct MixedVectorGradientIntegrator {
    pub sigma: f64,
}

impl H1HDivIntegrator for MixedVectorGradientIntegrator {
    fn add_to_element_matrix(
        &self,
        qp_scalar: &QpData<'_>,
        vec_col: &[f64],
        dim: usize,
        m_elem: &mut [f64],
    ) {
        let n_r = qp_scalar.n_dofs;
        let n_c = vec_col.len() / dim;
        let w = qp_scalar.weight * self.sigma;
        for j in 0..n_r {
            let gj = &qp_scalar.grad_phys[j * dim..][..dim];
            for i in 0..n_c {
                let vi = &vec_col[i * dim..][..dim];
                let dot = gj[0] * vi[0] + gj[1] * vi[1] + if dim > 2 { gj[2] * vi[2] } else { 0.0 };
                m_elem[j * n_c + i] += w * dot;
            }
        }
    }
}

/// Assemble H¹ × HDiv mixed bilinear form.
///
/// Row space = H¹ (scalar), column space = HDiv (vector-valued).
pub fn assemble_h1_hdiv_mixed<SR, SC>(
    row_space: &SR,   // H¹
    col_space: &SC,   // HDiv
    integrators: &[&dyn H1HDivIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64>
where
    SR: FESpace,
    SC: FESpace,
{
    let mesh = row_space.mesh();
    let dim = mesh.dim() as usize;
    let n_rows = row_space.n_dofs();
    let n_cols = col_space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_rows, n_cols);

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let order_r = row_space.order();
        let order_c = col_space.order();
        let ref_r = ref_elem_vol(elem_type, order_r).unwrap();
        let n_r = ref_r.n_dofs();
        let ref_c = ref_elem_vec(elem_type, order_c, SpaceType::HDiv).unwrap();
        let n_c = ref_c.n_dofs();
        let quad = ref_r.quadrature(quad_order);

        let global_rows: Vec<usize> = row_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_cols: Vec<usize> = col_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let elem_tag = mesh.element_tag(e);
        let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
        let geo_elem = if use_iso { crate::geo_ref_elem_from_mesh(mesh, e) } else { None };

        let n_elem_r = global_rows.len();
        let n_elem_c = global_cols.len();
        let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];
        let mut phi_r = vec![0.0; n_r];
        let mut grad_r = vec![0.0; n_r * dim];
        let mut grad_phys = vec![0.0; n_r * dim];
        let mut vec_col = vec![0.0; n_c * dim];
        let mut j_inv_t = nalgebra::DMatrix::<f64>::identity(dim, dim);

        for (q, xi) in quad.points.iter().enumerate() {
            let (w, det_j, xp) = if use_iso {
                let ge = geo_elem.as_ref().expect("geo_ref_elem");
                let (jac, det_j, x) = isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim);
                j_inv_t = jac.clone().try_inverse().expect("invertible Jacobian").transpose();
                (quad.weights[q] * det_j.abs(), det_j, x)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                j_inv_t = tr.jacobian_inv_t().clone();
                (quad.weights[q] * tr.det_j().abs(), tr.det_j(), tr.map_to_physical(xi))
            };
            ref_r.eval_basis(xi, &mut phi_r);
            ref_r.eval_grad_basis(xi, &mut grad_r);
            transform_grads(&j_inv_t, &grad_r, &mut grad_phys, n_r, dim);
            ref_c.eval_basis_vec(xi, &mut vec_col);
            // Piola-transform the HDiv trial shapes (MFEM CalcTestShape uses
            // the Piola map for RT elements):  w = (1/det J) J ŵ.
            // Together with the |det J| quadrature weight this yields
            // ∫ ∇φ·w = Σ w_q · sign(det J) · ∇φ̂·ŵ, i.e. the reference-domain
            // integral — the geometric factors cancel, matching MFEM.
            let jac = j_inv_t.clone().try_inverse().map(|m| m.transpose())
                .unwrap_or_else(|| nalgebra::DMatrix::identity(dim, dim));
            let mut vec_col_piola = vec![0.0; n_c * dim];
            for i in 0..n_c {
                for r in 0..dim {
                    let mut acc = 0.0;
                    for c in 0..dim {
                        acc += jac[(r, c)] * vec_col[i * dim + c];
                    }
                    vec_col_piola[i * dim + r] = acc / det_j;
                }
            }
            let qp_r = QpData {
                n_dofs: n_elem_r, dim, weight: w, phys_weight: w,
                ref_weight: quad.weights[q], phi: &phi_r,
                grad_phys: &grad_phys,
                x_phys: &xp, elem_id: e, elem_tag, elem_dofs: None,
            };
            for integ in integrators {
                integ.add_to_element_matrix(&qp_r, &vec_col_piola, dim, &mut m_elem);
            }
        }
        // HDiv column signs: MFEM encodes the RT face-orientation sign into
        // the (possibly negative) column vdofs, so the assembled entries must
        // be scaled by the column sign (row = H¹ is unsigned).
        let col_signs = col_space.element_signs(e);
        for (ir, &gr) in global_rows.iter().enumerate() {
            for (ic, &gc) in global_cols.iter().enumerate() {
                let sc = col_signs.and_then(|s| s.get(ic)).copied().unwrap_or(1.0);
                coo.add(gr, gc, sc * m_elem[ir * n_elem_c + ic]);
            }
        }
    }
    coo.into_csr()
}

/// Assemble the mixed HCurl × HDiv mass matrix (weak curl coupling).
///
/// Computes `M[i,j] = ∫ ε · ψ_j · w_i dx`
/// where `ψ_j ∈ H(curl)` (columns) and `w_i ∈ H(div)` (rows).
///
/// This is the matrix form of MFEM's `VectorFEMassIntegrator` on
/// `ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_)`.  Used by
/// Tesla (magnetostatics) and Volta (electrostatics) mini apps for
/// coupling the vector potential A ∈ H(curl) to the flux B ∈ H(div).
pub fn assemble_hcurl_hdiv_mixed<M: fem_mesh::topology::MeshTopology + Clone + 'static>(
    nd_space: &HCurlSpace<M>,
    rt_space: &HDivSpace<M>,
    quad_order: u8,
    eps: f64,
) -> CsrMatrix<f64>
where
    M: fem_mesh::topology::MeshTopology,
{
    use crate::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    use fem_element::ReferenceElement;

    let mesh = nd_space.mesh();
    let dim = mesh.dim() as usize;
    let n_rows = rt_space.n_dofs();
    let n_cols = nd_space.n_dofs();
    let mut coo = CooMatrix::new(n_rows, n_cols);

    let nd_ord = nd_space.order();
    let rt_ord = rt_space.order();

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let nd_ref = ref_elem_vec(et, nd_ord, SpaceType::HCurl)
            .expect("HCurl ref elem");
        let rt_ref = ref_elem_vec(et, rt_ord, SpaceType::HDiv)
            .expect("HDiv ref elem");
        let n_nd = nd_ref.n_dofs();
        let n_rt = rt_ref.n_dofs();

        let nd_s = nd_space.element_signs(e);
        let rt_s = rt_space.element_signs(e);

        let global_nd: Vec<usize> = nd_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let global_rt: Vec<usize> = rt_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let ng_nd = global_nd.len();
        let ng_rt = global_rt.len();

        let quad = nd_ref.quadrature(quad_order);
        let mut me = vec![0.0; ng_rt * ng_nd];
        let mut nd_basis = vec![0.0; n_nd * dim];
        let mut rt_basis = vec![0.0; n_rt * dim];

        let use_iso = !matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
        let ge = if use_iso { geo_ref_elem_from_mesh(mesh, e) } else { None };
        let nodes = mesh.element_nodes(e);

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jit, det_j): (f64, nalgebra::DMatrix<f64>, f64) = if use_iso {
                let g: &dyn ReferenceElement = ge.as_deref().unwrap();
                let (jac, det, _) = isoparametric_jacobian(mesh, &nodes, g, xi, dim);
                (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose(), det)
            } else {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone(), tr.det_j())
            };

            nd_ref.eval_basis_vec(xi, &mut nd_basis);
            rt_ref.eval_basis_vec(xi, &mut rt_basis);

            // Original Jacobian J (from J^{-T})
            let jac = jit.clone().try_inverse().map(|m| m.transpose())
                .unwrap_or_else(|| nalgebra::DMatrix::identity(dim, dim));

            for j in 0..ng_nd {
                let sj_nd = nd_s.get(j).copied().unwrap_or(1.0);
                // HCurl Piola: ψ = J^{-T} · ψ_ref
                let nd_x = sj_nd * (0..dim).map(|d| jit[(0,d)] * nd_basis[j*dim + d]).sum::<f64>();
                let nd_y = if dim > 1 { sj_nd * (0..dim).map(|d| jit[(1,d)] * nd_basis[j*dim + d]).sum::<f64>() } else { 0.0 };
                let nd_z = if dim > 2 { sj_nd * (0..dim).map(|d| jit[(2,d)] * nd_basis[j*dim + d]).sum::<f64>() } else { 0.0 };

                for i in 0..ng_rt {
                    let sj_rt = rt_s.get(i).copied().unwrap_or(1.0);
                    let inv_det = 1.0 / det_j;
                    // HDiv Piola: w = (1/det_J) * J · w_ref
                    let rt_x = sj_rt * inv_det * (0..dim).map(|d| jac[(0,d)] * rt_basis[i*dim + d]).sum::<f64>();
                    let rt_y = if dim > 1 { sj_rt * inv_det * (0..dim).map(|d| jac[(1,d)] * rt_basis[i*dim + d]).sum::<f64>() } else { 0.0 };
                    let rt_z = if dim > 2 { sj_rt * inv_det * (0..dim).map(|d| jac[(2,d)] * rt_basis[i*dim + d]).sum::<f64>() } else { 0.0 };

                    let dot = nd_x * rt_x + nd_y * rt_y + nd_z * rt_z;
                    me[i * ng_nd + j] += w * eps * dot;
                }
            }
        }
        for (ir, &r) in global_rt.iter().enumerate() {
            for (ic, &c) in global_nd.iter().enumerate() {
                let v = me[ir * ng_nd + ic];
                if v != 0.0 { coo.add(r, c, v); }
            }
        }
    }
    coo.into_csr()
}

/// Assemble the mixed HCurl × HDiv weak curl matrix (magnetization coupling).
///
/// Computes `M[i,j] = ∫ ν · curl(ψ_j) · w_i dx`
/// where `ψ_j ∈ H(curl)` (columns) and `w_i ∈ H(div)` (rows).
///
/// This is the matrix form of MFEM's `VectorFECurlIntegrator` on
/// `ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_)`.  Used by Tesla
/// (magnetostatics) for magnetization source coupling `weakCurlMuInv_`.
pub fn assemble_hcurl_hdiv_weak_curl<M: fem_mesh::topology::MeshTopology + Clone + 'static>(
    nd_space: &HCurlSpace<M>,
    rt_space: &HDivSpace<M>,
    quad_order: u8,
    nu: f64,
) -> CsrMatrix<f64>
where
    M: fem_mesh::topology::MeshTopology,
{
    use crate::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    use fem_element::ReferenceElement;

    let mesh = nd_space.mesh();
    let dim = mesh.dim() as usize;
    let n_rows = rt_space.n_dofs();
    let n_cols = nd_space.n_dofs();
    let mut coo = CooMatrix::new(n_rows, n_cols);

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let nd_ref = ref_elem_vec(et, nd_space.order(), SpaceType::HCurl)
            .expect("HCurl ref elem");
        let rt_ref = ref_elem_vec(et, rt_space.order(), SpaceType::HDiv)
            .expect("HDiv ref elem");
        let n_nd = nd_ref.n_dofs();
        let n_rt = rt_ref.n_dofs();

        let global_nd: Vec<usize> = nd_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let global_rt: Vec<usize> = rt_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let ng_nd = global_nd.len();
        let ng_rt = global_rt.len();

        let quad = nd_ref.quadrature(quad_order);
        let mut me = vec![0.0; ng_rt * ng_nd];
        let mut nd_curl = vec![0.0; n_nd * dim]; // curl of HCurl basis
        let mut rt_basis = vec![0.0; n_rt * dim];

        let use_iso = !matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
        let ge = if use_iso { geo_ref_elem_from_mesh(mesh, e) } else { None };
        let nodes = mesh.element_nodes(e);

        // Orientation signs for BOTH spaces — the H(curl) sign (edge direction
        // vs global min→max) and the H(div) sign (face orientation vs the
        // canonical face).  The missing RT sign flipped every odd-orientation
        // face row (serial ex24 prob-1 weak-curl form error).
        let nd_signs = nd_space.element_signs(e);
        let rt_signs = rt_space.element_signs(e);

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jit, det_j): (f64, nalgebra::DMatrix<f64>, f64) = if use_iso {
                let g: &dyn ReferenceElement = ge.as_deref().unwrap();
                let (jac, det, _) = isoparametric_jacobian(mesh, &nodes, g, xi, dim);
                (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose(), det)
            } else {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone(), tr.det_j())
            };

            nd_ref.eval_curl(xi, &mut nd_curl);
            rt_ref.eval_basis_vec(xi, &mut rt_basis);

            let jac = jit.clone().try_inverse().map(|m| m.transpose())
                .unwrap_or_else(|| nalgebra::DMatrix::identity(dim, dim));

            for j in 0..ng_nd {
                // Physical curl of HCurl basis: curl(ψ)_phys = (1/det_J) * J · curl(ψ)_ref (3D)
                // or curl_2d(ψ)_phys = (1/det_J) * curl(ψ)_ref (2D scalar)
                let (cx, cy, cz) = if dim == 2 {
                    // 2D: curl gives scalar (z-component)
                    let curl_ref = nd_curl[j];
                    let curl_phys = curl_ref / det_j;
                    (0.0, 0.0, curl_phys)
                } else {
                    // 3D: curl gives 3-vector
                    // curl(v_phys) = (1/det_J) · J · curl_ref(v_ref) — the
                    // H(div) Piola carries its own 1/det_J, so the curl needs
                    // its own 1/det_J factor for the volume integral to come
                    // out right (pex24 prob-1 error: RT mass-solve off).
                    let crx = nd_curl[j*dim];
                    let cry = nd_curl[j*dim + 1];
                    let crz = nd_curl[j*dim + 2];
                    let id = 1.0 / det_j;
                    (id * (jac[(0,0)]*crx + jac[(0,1)]*cry + jac[(0,2)]*crz),
                     id * (jac[(1,0)]*crx + jac[(1,1)]*cry + jac[(1,2)]*crz),
                     id * (jac[(2,0)]*crx + jac[(2,1)]*cry + jac[(2,2)]*crz))
                };

                for i in 0..ng_rt {
                    // HDiv Piola: w = (1/det_J) * J · w_ref
                    let id = 1.0 / det_j;
                    let wx = id * (jac[(0,0)]*rt_basis[i*dim] + if dim>1 {jac[(0,1)]*rt_basis[i*dim+1]} else {0.0} + if dim>2 {jac[(0,2)]*rt_basis[i*dim+2]} else {0.0});
                    let wy = if dim>1 { id * (jac[(1,0)]*rt_basis[i*dim] + jac[(1,1)]*rt_basis[i*dim+1] + if dim>2 {jac[(1,2)]*rt_basis[i*dim+2]} else {0.0}) } else { 0.0 };
                    let wz = if dim>2 { id * (jac[(2,0)]*rt_basis[i*dim] + jac[(2,1)]*rt_basis[i*dim+1] + jac[(2,2)]*rt_basis[i*dim+2]) } else { 0.0 };

                    let dot = cx*wx + cy*wy + cz*wz;
                    let s_nd = if j < nd_signs.len() { nd_signs[j] } else { 1.0 };
                    let s_rt = if i < rt_signs.len() { rt_signs[i] } else { 1.0 };
                    me[i * ng_nd + j] += w * nu * s_nd * s_rt * dot;
                }
            }
        }
        for (ir, &r) in global_rt.iter().enumerate() {
            for (ic, &c) in global_nd.iter().enumerate() {
                let v = me[ir * ng_nd + ic];
                if v != 0.0 { coo.add(r, c, v); }
            }
        }
    }
    coo.into_csr()
}

/// Constant P0 reference element: 1 DOF, constant basis = 1.0, zero gradient.
struct P0;

impl ReferenceElement for P0 {
    fn dim(&self) -> u8 { unreachable!("P0 dim depends on context") }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], values: &mut [f64]) { values[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], grads: &mut [f64]) { for g in grads { *g = 0.0; } }
    fn quadrature(&self, _order: u8) -> fem_element::QuadratureRule {
        unreachable!("P0 quadrature: use the paired element")
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0; 3]] }
}

pub fn ref_elem_vol(elem_type: ElementType, order: u8) -> Result<Box<dyn ReferenceElement>, String> {
    Ok(match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) |
        (ElementType::Tet4 | ElementType::Tet10, 0) |
        (ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9, 0) |
        (ElementType::Hex8 | ElementType::Hex20, 0) => Box::new(P0),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4 | ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4 | ElementType::Tet10, 3) => Box::new(TetP3),
        (ElementType::Quad4, 1) => Box::new(QuadQk::new(1)),
        (ElementType::Quad4, 2) => Box::new(QuadQk::new(2)),
        (ElementType::Quad4, 3) => Box::new(QuadQk::new(3)),
        (ElementType::Quad8 | ElementType::Quad9, 1) => Box::new(QuadSerendipityPk::new(1)),
        (ElementType::Quad8 | ElementType::Quad9, 2) => Box::new(QuadSerendipityPk::new(2)),
        (ElementType::Quad8 | ElementType::Quad9, 3) => Box::new(QuadSerendipityPk::new(3)),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) => Box::new(HexQ2),
        (ElementType::Hex8, 3) => Box::new(HexQ3),
        (ElementType::Hex20, 1) => Box::new(HexSerendipityPk::new(1)),
        (ElementType::Hex20, 2) => Box::new(HexSerendipityPk::new(2)),
        (ElementType::Hex20, 3) => Box::new(HexSerendipityPk::new(3)),
        (ElementType::Prism6 | ElementType::Prism15, 1) => Box::new(fem_element::lagrange::PrismPk::new(1)),
        (ElementType::Prism6 | ElementType::Prism15, 2) => Box::new(fem_element::lagrange::PrismPk::new(2)),
        (ElementType::Prism6 | ElementType::Prism15, 3) => Box::new(fem_element::lagrange::PrismPk::new(3)),
        _ => return Err(format!("ref_elem_vol: unsupported ({elem_type:?}, order={order})")),
    })
}

pub fn ref_elem_vec(elem_type: ElementType, order: u8, space: SpaceType) -> Result<Box<dyn VectorReferenceElement>, String> {
    Ok(match (space, elem_type, order) {
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(TriRT0),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, ElementType::Quad4, 0) => Box::new(QuadRT0),
        (SpaceType::HDiv, ElementType::Quad4, 1) => Box::new(QuadRT1),
        (SpaceType::HDiv, ElementType::Quad4, o) if o >= 2 => {
            Box::new(fem_element::raviart_thomas::QuadRTk::new(o as usize))
        }
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 0) => Box::new(TetRT0),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetRT1),
        (SpaceType::HDiv, ElementType::Hex8, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Hex8, 1) => Box::new(HexRT1),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriND1),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetND1),
        (SpaceType::HCurl, ElementType::Quad4, 1) => Box::new(QuadND1),
        (SpaceType::HCurl, ElementType::Quad4, 2) => Box::new(QuadNDk::new(2)),
        (SpaceType::HCurl, ElementType::Quad8 | ElementType::Quad9, 1) => Box::new(QuadND1),
        (SpaceType::HCurl, ElementType::Quad8 | ElementType::Quad9, 2) => Box::new(QuadNDk::new(2)),
        (SpaceType::HCurl, ElementType::Hex8, 1) => Box::new(HexND1),
        (SpaceType::HCurl, ElementType::Hex8, 2) => Box::new(HexNDk::new(2)),
        (SpaceType::HCurl, ElementType::Hex20, 1) => Box::new(HexND1),
        (SpaceType::HCurl, ElementType::Hex20, 2) => Box::new(HexNDk::new(2)),
        (SpaceType::HDiv, ElementType::Hex8, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Hex8, 1) => Box::new(HexRT1),
        (SpaceType::HDiv, ElementType::Hex20, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Hex20, 1) => Box::new(HexRT1),
        (SpaceType::HDiv, ElementType::Prism6, 0) => Box::new(PrismRTk::new(0)),
        (SpaceType::HDiv, ElementType::Prism6, 1) => Box::new(PrismRTk::new(1)),
        (SpaceType::HCurl, ElementType::Prism6, 1) => Box::new(PrismND1),
        (SpaceType::HCurl, ElementType::Prism6, o) if o >= 2 => Box::new(PrismNDk::new(o as usize)),
        _ => return Err(format!("ref_elem_vec: unsupported (space={space:?}, {elem_type:?}, order={order})")),
    })
}

fn transform_grads(j_inv_t: &DMatrix<f64>, grad_ref: &[f64], grad_phys: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += j_inv_t[(j, k)] * grad_ref[i * dim + k]; }
            grad_phys[i * dim + j] = s;
        }
    }
}

fn accumulate_mixed_volume_element<SR, SC>(
    row_space: &SR,
    col_space: &SC,
    e: u32,
    integrators: &[&dyn MixedBilinearIntegrator],
    quad_order: u8,
    coo: &mut CooMatrix<f64>,
) where
    SR: FESpace,
    SC: FESpace,
{
    let mesh = row_space.mesh();
    let dim = mesh.dim() as usize;
    let order_r = row_space.order();
    let order_c = col_space.order();
    let space_r = row_space.space_type();
    let space_c = col_space.space_type();

    // If either space is HDiv or HCurl, dispatch to vector path.
    if matches!(space_r, SpaceType::HDiv | SpaceType::HCurl) ||
       matches!(space_c, SpaceType::HDiv | SpaceType::HCurl) {
        if cfg!(debug_assertions) {
            eprintln!("WARN: MixedAssembler fallback for vector-valued spaces; consider assemble_hdiv_l2_mixed");
        }
        return; // silently skip — caller should use dedicated HDiv×L² path
    }

    let mut phi_r = Vec::<f64>::new();
    let mut phi_c = Vec::<f64>::new();
    let mut grad_ref_r = Vec::<f64>::new();
    let mut grad_ref_c = Vec::<f64>::new();
    let mut grad_phys_r = Vec::<f64>::new();
    let mut grad_phys_c = Vec::<f64>::new();

    let elem_type = mesh.element_type(e);
    // L2/DG row spaces use the Gauss-Legendre nodal basis on [0,1]²
    // (ref_elem_vol_l2 → QuadL2GL); H1 keeps the topological QuadQk.
    let ref_r = ref_elem_vol_for_space(row_space, elem_type, order_r);
    let ref_c = ref_elem_vol(elem_type, order_c).unwrap();
    let n_r = ref_r.n_dofs();
    let n_c = ref_c.n_dofs();

    // Use the higher-order element for quadrature (P0 has no meaningful quadrature).
    let quad = if order_r >= order_c { ref_r.quadrature(quad_order) } else { ref_c.quadrature(quad_order) };

    let global_rows: Vec<usize> = row_space.element_dofs(e).iter().map(|&d| d as usize).collect();
    let global_cols: Vec<usize> = col_space.element_dofs(e).iter().map(|&d| d as usize).collect();
    let n_elem_r = global_rows.len();
    let n_elem_c = global_cols.len();
    let nodes = mesh.element_nodes(e);
    let elem_tag = mesh.element_tag(e);

    let use_iso = !matches!(elem_type, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
    let geo_elem = if use_iso { crate::vector_assembler::geo_ref_elem_from_mesh(mesh, e) } else { None };
    // High-order geometry: use the geometry nodes (P2 etc.) not just the
    // linear element vertices, so isoparametric_jacobian can index them.
    let geom_nodes = if mesh.geom_order() > 1 { mesh.geometry_nodes(e) } else { nodes };

    let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];
    phi_r.resize(n_r, 0.0);
    phi_c.resize(n_c, 0.0);
    grad_ref_r.resize(n_r * dim, 0.0);
    grad_ref_c.resize(n_c * dim, 0.0);
    grad_phys_r.resize(n_r * dim, 0.0);
    grad_phys_c.resize(n_c * dim, 0.0);

    for (q, xi) in quad.points.iter().enumerate() {
        let (w, j_inv_t, xp) = if use_iso {
            let ge = geo_elem.as_ref().unwrap();
            let (jac, det_j, xp) = isoparametric_jacobian(mesh, geom_nodes, ge.as_ref(), xi, dim);
            let w_q = quad.weights[q] * det_j.abs();
            let jit = jac.try_inverse().expect("invertible isoparametric Jacobian").transpose();
            (w_q, jit, xp)
        } else {
            let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
            let w_q = quad.weights[q] * tr.det_j().abs();
            let jit = tr.jacobian_inv_t().clone();
            let xp_q = tr.map_to_physical(xi);
            (w_q, jit, xp_q)
        };

        ref_r.eval_basis(xi, &mut phi_r);
        ref_c.eval_basis(xi, &mut phi_c);
        ref_r.eval_grad_basis(xi, &mut grad_ref_r);
        ref_c.eval_grad_basis(xi, &mut grad_ref_c);
        transform_grads(&j_inv_t, &grad_ref_r, &mut grad_phys_r, n_r, dim);
        transform_grads(&j_inv_t, &grad_ref_c, &mut grad_phys_c, n_c, dim);

        let qp_r = QpData {
            n_dofs: n_elem_r,
            dim,

            weight: w,
                phys_weight: w,
                ref_weight: quad.weights[q],
            phi: &phi_r,
            grad_phys: &grad_phys_r,
            x_phys: &xp,
            elem_id: e,
            elem_tag,
            elem_dofs: None,
        };
        let qp_c = QpData {
            n_dofs: n_elem_c,
            dim,

            weight: w,
                phys_weight: w,
                ref_weight: quad.weights[q],
            phi: &phi_c,
            grad_phys: &grad_phys_c,
            x_phys: &xp,
            elem_id: e,
            elem_tag,
            elem_dofs: None,
        };

        for integ in integrators {
            integ.add_to_element_matrix(&qp_r, &qp_c, &mut m_elem);
        }
    }

    for (ir, &gr) in global_rows.iter().enumerate() {
        for (ic, &gc) in global_cols.iter().enumerate() {
            coo.add(gr, gc, m_elem[ir * n_elem_c + ic]);
        }
    }
}

#[cfg(feature = "parallel")]
fn assemble_mixed_bilinear_volume_parallel<SR, SC>(
    row_space: &SR,
    col_space: &SC,
    integrators: &[&dyn MixedBilinearIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64>
where
    SR: FESpace,
    SC: FESpace,
{
    let mesh = row_space.mesh();
    let n_rows = row_space.n_dofs();
    let n_cols = col_space.n_dofs();
    let merged = mesh
        .elem_iter()
        .into_par_iter()
        .map(|e| {
            let mut local = CooMatrix::<f64>::new(n_rows, n_cols);
            accumulate_mixed_volume_element(
                row_space,
                col_space,
                e,
                integrators,
                quad_order,
                &mut local,
            );
            local
        })
        .reduce(
            || CooMatrix::<f64>::new(n_rows, n_cols),
            |mut a, b| {
                a.append(b);
                a
            },
        );
    merged.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    /// B = ∫ p (∇·u) dx should have the right shape.
    #[test]
    fn mixed_assembler_shape() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        // Create separate owned meshes for each space.
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let vel_space = fem_space::VectorH1Space::new(mesh, 1, 2);
        let pre_space = H1Space::new(mesh2, 1);
        let b = MixedAssembler::assemble_bilinear(
            &pre_space,
            &vel_space,
            &[&DivIntegrator],
            3,
        );
        assert_eq!(b.nrows, pre_space.n_dofs());
        assert_eq!(b.ncols, vel_space.n_dofs());
    }

    /// `unit_square_tri(6)` has 72 triangles (≥ default parallel threshold 64).
    #[cfg(feature = "parallel")]
    #[test]
    fn mixed_volume_parallel_matches_elementwise_reference() {
        use fem_linalg::CooMatrix;

        let mesh = Mesh::<2>::unit_square_tri(6);
        let mesh2 = Mesh::<2>::unit_square_tri(6);
        let vel_space = fem_space::VectorH1Space::new(mesh, 1, 2);
        let pre_space = H1Space::new(mesh2, 1);
        let n_rows = pre_space.n_dofs();
        let n_cols = vel_space.n_dofs();

        let mut coo_ref = CooMatrix::<f64>::new(n_rows, n_cols);
        for e in pre_space.mesh().elem_iter() {
            accumulate_mixed_volume_element(
                &pre_space,
                &vel_space,
                e,
                &[&DivIntegrator],
                3,
                &mut coo_ref,
            );
        }
        let b_ref = coo_ref.into_csr();

        let b_par = MixedAssembler::assemble_bilinear(
            &pre_space,
            &vel_space,
            &[&DivIntegrator],
            3,
        );

        assert_eq!(b_par.nrows, b_ref.nrows);
        assert_eq!(b_par.ncols, b_ref.ncols);
        for i in 0..b_par.nrows {
            for j in 0..b_par.ncols {
                let a = b_par.get(i, j);
                let b = b_ref.get(i, j);
                assert!((a - b).abs() < 1e-12, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn hdiv_l2_darcy_shape() {
        use fem_space::{HDivSpace, L2Space};
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let vel = HDivSpace::new(mesh, 0);
        let pre = L2Space::new(mesh2, 0);
        let b = assemble_hdiv_l2_mixed(&pre, &vel, &[&HDivL2DivIntegrator], 3);
        assert_eq!(b.nrows, pre.n_dofs());
        assert_eq!(b.ncols, vel.n_dofs());
        assert!(b.nrows > 0);
        assert!(b.ncols > 0);
    }

    #[test]
    fn hcurl_h1_curl_shape() {
        use fem_space::{HCurlSpace, H1Space};
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let curl_space = HCurlSpace::new(mesh, 1);
        let h1_space = H1Space::new(mesh2, 1);
        let b = assemble_hcurl_h1_mixed(&h1_space, &curl_space, &[&HCurlH1CurlIntegrator], 3);
        assert_eq!(b.nrows, h1_space.n_dofs());
        assert_eq!(b.ncols, curl_space.n_dofs());
        assert!(b.nrows > 0);
        assert!(b.ncols > 0);
    }

    #[test]
    fn ref_elem_unsupported_returns_err() {
        assert!(ref_elem_vol(ElementType::Tri3, 99).is_err());
        assert!(ref_elem_vec(ElementType::Tri3, 99, SpaceType::HDiv).is_err());
    }
}
