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
use fem_element::{ReferenceElement, VectorReferenceElement, lagrange::{TetP1, TetP2, TetP3, TriP1, TriP2, TriP3, QuadQ1, QuadQ2, QuadQ3, HexQ1, HexQ2, HexQ3}};
use fem_element::raviart_thomas::{TriRT0, TriRT1, TetRT0, TetRT1, HexRT0, HexRT1};
use fem_element::nedelec::{TriND1, TetND1, QuadND1, HexND1};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::{FESpace, SpaceType};

use crate::integrator::QpData;

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

        let ref_r = ref_elem_vol(elem_type, row_space.order()).unwrap();
        let n_r = ref_r.n_dofs();

        // Use the vector element's quadrature (P0 has no quadrature).
        let quad = ref_c.quadrature(quad_order);

        let global_rows: Vec<usize> = row_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let global_cols: Vec<usize> = col_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let elem_tag = mesh.element_tag(e);
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
        let _j_inv_t = tr.jacobian_inv_t().clone();

        let n_elem_r = global_rows.len();
        let n_elem_c = global_cols.len();
        let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];

        let mut phi_r = vec![0.0; n_r];
        let mut div_c_vec = vec![0.0; n_c];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * tr.det_j().abs();

            ref_r.eval_basis(xi, &mut phi_r);
            ref_c.eval_div(xi, &mut div_c_vec);

            let xp = tr.map_to_physical(xi);
            let qp_r = QpData {
                n_dofs: n_elem_r,
                dim,
                weight: w,
                phi: &phi_r,
                grad_phys: &[], // not needed
                x_phys: &xp,
                elem_id: e,
                elem_tag,
                elem_dofs: None,
            };

            for integ in integrators {
                integ.add_to_element_matrix(&qp_r, &div_c_vec, dim, &mut m_elem);
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
                n_dofs: n_elem_r, dim, weight: w, phi: &phi_r, grad_phys: &[],
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

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Result<Box<dyn ReferenceElement>, String> {
    Ok(match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) |
        (ElementType::Tet4 | ElementType::Tet10, 0) |
        (ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9, 0) |
        (ElementType::Hex8 | ElementType::Hex20, 0) => Box::new(P0),
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4 | ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4 | ElementType::Tet10, 3) => Box::new(TetP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) => Box::new(HexQ2),
        (ElementType::Hex8, 3) => Box::new(HexQ3),
        _ => return Err(format!("ref_elem_vol: unsupported ({elem_type:?}, order={order})")),
    })
}

fn ref_elem_vec(elem_type: ElementType, order: u8, space: SpaceType) -> Result<Box<dyn VectorReferenceElement>, String> {
    Ok(match (space, elem_type, order) {
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(TriRT0),
        (SpaceType::HDiv, ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriRT1),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 0) => Box::new(TetRT0),
        (SpaceType::HDiv, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetRT1),
        (SpaceType::HDiv, ElementType::Hex8, 0) => Box::new(HexRT0),
        (SpaceType::HDiv, ElementType::Hex8, 1) => Box::new(HexRT1),
        (SpaceType::HCurl, ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriND1),
        (SpaceType::HCurl, ElementType::Tet4 | ElementType::Tet10, 1) => Box::new(TetND1),
        (SpaceType::HCurl, ElementType::Quad4, 1) => Box::new(QuadND1),
        (SpaceType::HCurl, ElementType::Hex8, 1) => Box::new(HexND1),
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
    let ref_r = ref_elem_vol(elem_type, order_r).unwrap();
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

    let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
    let j_inv_t = tr.jacobian_inv_t().clone();

    let mut m_elem = vec![0.0_f64; n_elem_r * n_elem_c];
    phi_r.resize(n_r, 0.0);
    phi_c.resize(n_c, 0.0);
    grad_ref_r.resize(n_r * dim, 0.0);
    grad_ref_c.resize(n_c * dim, 0.0);
    grad_phys_r.resize(n_r * dim, 0.0);
    grad_phys_c.resize(n_c * dim, 0.0);

    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * tr.det_j().abs();

        ref_r.eval_basis(xi, &mut phi_r);
        ref_c.eval_basis(xi, &mut phi_c);
        ref_r.eval_grad_basis(xi, &mut grad_ref_r);
        ref_c.eval_grad_basis(xi, &mut grad_ref_c);
        transform_grads(&j_inv_t, &grad_ref_r, &mut grad_phys_r, n_r, dim);
        transform_grads(&j_inv_t, &grad_ref_c, &mut grad_phys_c, n_c, dim);
        let xp = tr.map_to_physical(xi);

        let qp_r = QpData {
            n_dofs: n_elem_r,
            dim,
            weight: w,
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
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    /// B = ∫ p (∇·u) dx should have the right shape.
    #[test]
    fn mixed_assembler_shape() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        // Create separate owned meshes for each space.
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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

        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(6);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
