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
use fem_element::{ReferenceElement, lagrange::{TetP1, TriP1, TriP2, QuadQ1, QuadQ2, HexQ1}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

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

// ─── Jacobian / transform helpers (duplicated from assembler.rs) ──────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tet4, 1)                           => Box::new(TetP1),
        (ElementType::Quad4, 1)                          => Box::new(QuadQ1),
        (ElementType::Quad4, 2)                          => Box::new(QuadQ2),
        (ElementType::Hex8, 1)                           => Box::new(HexQ1),
        _ => panic!("mixed_assembler ref_elem_vol: unsupported ({elem_type:?}, order={order})"),
    }
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

    let mut phi_r = Vec::<f64>::new();
    let mut phi_c = Vec::<f64>::new();
    let mut grad_ref_r = Vec::<f64>::new();
    let mut grad_ref_c = Vec::<f64>::new();
    let mut grad_phys_r = Vec::<f64>::new();
    let mut grad_phys_c = Vec::<f64>::new();

    let elem_type = mesh.element_type(e);
    let ref_r = ref_elem_vol(elem_type, order_r);
    let ref_c = ref_elem_vol(elem_type, order_c);
    let n_r = ref_r.n_dofs();
    let n_c = ref_c.n_dofs();

    let quad = ref_r.quadrature(quad_order);

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
}
