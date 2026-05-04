//! Reed-backed matrix-free H(curl) operator: `μ⁻¹ K_curl + α M_e`.
//!
//! [`HcurlReedOperator`] implements the libCEED `Eᵀ Bᵀ D B E` pattern for
//! H(curl) / Nédélec elements:
//!
//! * **E**: DOF gather / scatter via precomputed `elem_dofs`.
//! * **B**: Covariant Piola-transformed basis functions and their curls,
//!   computed on-the-fly per element during `apply`.
//! * **D**: Per-quadrature-point geometric factors (Jacobian data), stored once.
//!
//! ## Memory comparison with [`crate::partial::HcurlMatrixFreeOperator`]
//!
//! | Operator         | Storage per element       | Notes |
//! |------------------|---------------------------|-------|
//! | Matrix-free      | `n_ldofs²`                | grows quadratically |
//! | **Reed (this)**  | `dim² + 1` (affine simplex)| Jacobian + det, constant |
//!
//! For HexND2: 2916 vs 10 floats/element (excluding shared reference basis).
//! The shared reference basis (`n_qp × n_ldofs × dim`) is computed once for
//! all elements of the same type.
//!
//! ## Accuracy
//!
//! Mathematically identical to [`crate::partial::HcurlMatrixFreeOperator`];
//! unit tests verify relative error < 1 × 10⁻¹⁰ vs the assembled sparse matrix.

use nalgebra::DMatrix;

use fem_mesh::element_type::ElementType;
use fem_mesh::{ElementTransformation, topology::MeshTopology};
use fem_space::fe_space::{FESpace, SpaceType};

use crate::vector_assembler::{
    apply_signs, geo_ref_elem, isoparametric_jacobian,
    piola_hcurl_basis, piola_hcurl_curl, vec_ref_elem,
};

// ── HcurlReedOperator ─────────────────────────────────────────────────────────

/// Matrix-free H(curl) operator backed by precomputed per-qpt geometric data.
///
/// # Construction
///
/// Call [`HcurlReedOperator::new`] once to build the operator.  It stores:
///
/// * **Shared** reference basis values and curls at all quadrature points
///   (`ref_phi_data`, `ref_curl_data`) — same for every element.
/// * **Per-element** Jacobian matrix and its determinant (`elem_jac`,
///   `elem_det`) — constant within affine (simplex) elements; for
///   isoparametric Hex/Quad each quadrature point gets its own Jacobian.
/// * **Per-element** global DOF indices and orientation signs.
///
/// # Usage
///
/// Implements the [`crate::partial::MatFreeOperator`] trait:
///
/// ```text
/// y = 0;
/// op.apply(&x, &mut y);   // y += (μ⁻¹ K_curl + α M) x
/// ```
pub struct HcurlReedOperator {
    n_dofs:     usize,
    n_ldofs:    usize,   // DOFs per element
    n_elem:     usize,
    n_qp:       usize,   // quadrature points per element
    dim:        usize,
    curl_dim:   usize,   // 1 for 2D, 3 for 3D
    is_affine:  bool,    // true = simplex (constant J per elem), false = iso (J per qpt)

    /// Reference basis values at each quadrature point:
    /// `ref_phi_data[q * n_ldofs * dim + i * dim + c]`
    ref_phi_data:  Vec<f64>,
    /// Reference curl at each quadrature point:
    /// `ref_curl_data[q * n_ldofs * curl_dim + i * curl_dim + k]`
    ref_curl_data: Vec<f64>,
    /// Quadrature weights `w_q`.
    q_weights:     Vec<f64>,

    /// Per-element Jacobian entries (row-major, dim×dim).
    /// Affine: `elem_jac[e * dim² .. (e+1)*dim²]`  — one J per element.
    /// Iso: `elem_jac[e * n_qp * dim² .. ]`         — one J per qpt per element.
    elem_jac:   Vec<f64>,
    /// Per-element `|det J|` values.
    /// Affine: `elem_det[e]`  — one per element.
    /// Iso: `elem_det[e * n_qp + q]` — one per qpt per element.
    elem_det:   Vec<f64>,

    /// Global DOF indices: `elem_dofs[e * n_ldofs + i]`.
    elem_dofs:  Vec<u32>,
    /// Orientation signs: `elem_signs[e * n_ldofs + i]`.
    elem_signs: Vec<f64>,

    mu_inv: f64,
    alpha:  f64,
}

impl HcurlReedOperator {
    /// Build the operator for space `S` with material parameters `mu_inv` and `alpha`.
    ///
    /// - `mu_inv`     — scalar inverse permeability 1/μ (curl-curl coefficient).
    /// - `alpha`      — scalar mass coefficient (e.g. ω² ε or 1 for pure mass).
    /// - `quad_order` — quadrature order; `order + 2` is recommended.
    pub fn new<S: FESpace>(space: &S, mu_inv: f64, alpha: f64, quad_order: u8) -> Self {
        let mesh     = space.mesh();
        let dim      = mesh.dim() as usize;
        let n_dofs_g = space.n_dofs();
        let n_elem   = mesh.n_elements();

        assert_eq!(space.space_type(), SpaceType::HCurl,
            "HcurlReedOperator requires an H(curl) space");

        let elem_type0 = mesh.element_type(0);
        let ref_elem   = vec_ref_elem(SpaceType::HCurl, elem_type0, dim, space.order());
        let n_ldofs    = ref_elem.n_dofs();
        let curl_dim   = if dim == 2 { 1 } else { 3 };
        let quad       = ref_elem.quadrature(quad_order);
        let n_qp       = quad.points.len();

        let is_affine = !matches!(elem_type0, ElementType::Quad4 | ElementType::Hex8);

        // ── Precompute shared reference basis at all qpts ─────────────────
        let mut ref_phi_data  = vec![0.0_f64; n_qp * n_ldofs * dim];
        let mut ref_curl_data = vec![0.0_f64; n_qp * n_ldofs * curl_dim];
        let mut tmp_phi  = vec![0.0_f64; n_ldofs * dim];
        let mut tmp_curl = vec![0.0_f64; n_ldofs * curl_dim];
        let mut tmp_div  = vec![0.0_f64; n_ldofs];

        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis_vec(xi, &mut tmp_phi);
            ref_elem.eval_curl(xi, &mut tmp_curl);
            ref_elem.eval_div(xi, &mut tmp_div);  // not used but required by trait
            ref_phi_data [q * n_ldofs * dim     .. (q + 1) * n_ldofs * dim    ]
                .copy_from_slice(&tmp_phi);
            ref_curl_data[q * n_ldofs * curl_dim .. (q + 1) * n_ldofs * curl_dim]
                .copy_from_slice(&tmp_curl);
        }

        // ── Precompute per-element Jacobian data ──────────────────────────
        let jac_per_elem = if is_affine { 1 } else { n_qp };
        let mut elem_jac  = Vec::with_capacity(n_elem * jac_per_elem * dim * dim);
        let mut elem_det  = Vec::with_capacity(n_elem * jac_per_elem);
        let mut elem_dofs  = Vec::with_capacity(n_elem * n_ldofs);
        let mut elem_signs = Vec::with_capacity(n_elem * n_ldofs);

        let geo_elem = geo_ref_elem(elem_type0);

        for e in mesh.elem_iter() {
            let nodes     = mesh.element_nodes(e);
            let dofs      = space.element_dofs(e);
            let signs_opt = space.element_signs(e);

            elem_dofs.extend_from_slice(dofs);
            if let Some(s) = signs_opt {
                elem_signs.extend_from_slice(s);
            } else {
                elem_signs.extend(std::iter::repeat(1.0_f64).take(n_ldofs));
            }

            if is_affine {
                let tr  = ElementTransformation::from_simplex_nodes(mesh, nodes);
                let jac = tr.jacobian();
                let det = tr.det_j();
                for r in 0..dim {
                    for c in 0..dim {
                        elem_jac.push(jac[(r, c)]);
                    }
                }
                // Store signed det_j — piola_hcurl_curl needs the sign for 2D.
                elem_det.push(det);
            } else {
                // Isoparametric: store one Jacobian per quadrature point.
                let ge = geo_elem.as_ref()
                    .expect("geo_ref_elem must exist for isoparametric elements");
                for xi in &quad.points {
                    let (jac, det, _) = isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, dim);
                    for r in 0..dim {
                        for c in 0..dim {
                            elem_jac.push(jac[(r, c)]);
                        }
                    }
                    elem_det.push(det);
                }
            }
        }

        HcurlReedOperator {
            n_dofs: n_dofs_g, n_ldofs, n_elem, n_qp, dim, curl_dim, is_affine,
            ref_phi_data, ref_curl_data,
            q_weights: quad.weights.clone(),
            elem_jac, elem_det,
            elem_dofs, elem_signs,
            mu_inv, alpha,
        }
    }
}

impl crate::partial::MatFreeOperator for HcurlReedOperator {
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let Self {
            n_dofs: _, n_ldofs, n_elem, n_qp, dim, curl_dim, is_affine,
            ref_phi_data, ref_curl_data, q_weights,
            elem_jac, elem_det,
            elem_dofs, elem_signs,
            mu_inv, alpha,
        } = self;

        let dim2 = dim * dim;
        let jac_per_elem = if *is_affine { 1 } else { *n_qp };

        // Reusable work buffers (allocated per call to avoid heap fragmentation
        // inside the element loop — stack allocation avoided for large n_ldofs).
        let mut phi      = vec![0.0_f64; n_ldofs * dim];
        let mut curl_buf = vec![0.0_f64; n_ldofs * curl_dim];
        let mut div_buf  = vec![0.0_f64; *n_ldofs];
        let mut v_e      = vec![0.0_f64; *n_ldofs];

        for e in 0..*n_elem {
            let dofs  = &elem_dofs [e * n_ldofs .. (e + 1) * n_ldofs];
            let signs = &elem_signs[e * n_ldofs .. (e + 1) * n_ldofs];

            // Gather: x_e[i] = x[dofs[i]]
            let x_e: Vec<f64> = (0..*n_ldofs).map(|i| x[dofs[i] as usize]).collect();
            v_e.fill(0.0);

            for q in 0..*n_qp {
                let jac_idx = e * jac_per_elem + if *is_affine { 0 } else { q };
                let jac_flat = &elem_jac[jac_idx * dim2 .. (jac_idx + 1) * dim2];
                let _abs_det  = elem_det[jac_idx];

                // Rebuild nalgebra DMatrix for the Jacobian (dim × dim, row-major).
                let jac = DMatrix::from_row_slice(*dim, *dim, jac_flat);
                let j_inv_t = jac.clone()
                    .try_inverse()
                    .expect("degenerate element in HcurlReedOperator::apply")
                    .transpose();

                // det_j is signed (needed by piola_hcurl_curl for sign of 2D curl).
                let det_j   = elem_det[jac_idx];
                let abs_det = det_j.abs();
                // Physical weight = reference weight × |det J|  (same as partial.rs)
                let w = q_weights[q] * abs_det;

                // Load reference basis and curl for this qpt.
                let ref_phi_q  = &ref_phi_data [q * n_ldofs * dim     .. (q + 1) * n_ldofs * dim];
                let ref_curl_q = &ref_curl_data[q * n_ldofs * curl_dim .. (q + 1) * n_ldofs * curl_dim];

                // Apply Piola transforms.
                piola_hcurl_basis(&j_inv_t, ref_phi_q,  &mut phi,      *n_ldofs, *dim);
                piola_hcurl_curl (&jac, det_j, ref_curl_q, &mut curl_buf, *n_ldofs, *dim);

                // Apply DOF signs (signs baked into contrib — same as partial.rs).
                div_buf.fill(0.0);
                apply_signs(signs, &mut phi, &mut curl_buf, &mut div_buf, *n_ldofs, *dim, *curl_dim);

                // Accumulate contributions:
                //   curl-curl: mu_inv · w · Σ_i Σ_j curl_i · curl_j · x_j
                //   mass:      alpha  · w · Σ_i Σ_j phi_i  · phi_j  · x_j
                //
                // Rather than building the element matrix, stream over qpts:
                //   u_curl = Σ_j curl_j · x_e[j]        (scalar in 2D, 3-vec in 3D)
                //   u_phi  = Σ_j phi_j  · x_e[j]        (dim-vector)
                //   v_e[i] += mu_inv*w * curl_i · u_curl  +  alpha*w * phi_i · u_phi

                if *curl_dim == 1 {
                    // 2D
                    let u_curl: f64 = (0..*n_ldofs)
                        .map(|j| curl_buf[j] * x_e[j])
                        .sum();
                    let mut u_phi = [0.0_f64; 2];
                    for j in 0..*n_ldofs {
                        u_phi[0] += phi[j * 2]     * x_e[j];
                        u_phi[1] += phi[j * 2 + 1] * x_e[j];
                    }
                    let cc_scale = mu_inv * w;
                    let mm_scale = alpha  * w;
                    for i in 0..*n_ldofs {
                        v_e[i] += cc_scale * curl_buf[i] * u_curl
                            + mm_scale * (phi[i * 2] * u_phi[0] + phi[i * 2 + 1] * u_phi[1]);
                    }
                } else {
                    // 3D
                    let mut u_curl = [0.0_f64; 3];
                    let mut u_phi  = [0.0_f64; 3];
                    for j in 0..*n_ldofs {
                        for k in 0..3 {
                            u_curl[k] += curl_buf[j * 3 + k] * x_e[j];
                            u_phi [k] += phi     [j * 3 + k] * x_e[j];
                        }
                    }
                    let cc_scale = mu_inv * w;
                    let mm_scale = alpha  * w;
                    for i in 0..*n_ldofs {
                        let cc: f64 = (0..3).map(|k| curl_buf[i * 3 + k] * u_curl[k]).sum();
                        let mm: f64 = (0..3).map(|k| phi    [i * 3 + k] * u_phi [k]).sum();
                        v_e[i] += cc_scale * cc + mm_scale * mm;
                    }
                }
            }

            // Scatter: y[dofs[i]] += v_e[i]
            for i in 0..*n_ldofs {
                y[dofs[i] as usize] += v_e[i];
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partial::{HcurlMatrixFreeOperator, MatFreeOperator};

    /// Check that HcurlReedOperator produces the same result as HcurlMatrixFreeOperator
    /// for a 2D TriND1 mesh (relative error < 1e-10).
    #[test]
    fn hcurl_reed_matches_matfree_tri_nd1() {
        use fem_mesh::SimplexMesh;
        use fem_space::{HCurlSpace, fe_space::FESpace};

        // Unit square: 2 × 2 structured triangulation (8 triangles)
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();

        let mu_inv = 1.5_f64;
        let alpha  = 0.7_f64;
        let quad   = 3_u8;

        let op_mf   = HcurlMatrixFreeOperator::new(&space, mu_inv, alpha, quad);
        let op_reed = HcurlReedOperator::new(&space, mu_inv, alpha, quad);

        // Test vector: x[i] = sin(i+1)
        let x: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let mut y_mf   = vec![0.0_f64; n];
        let mut y_reed = vec![0.0_f64; n];
        op_mf.apply(&x, &mut y_mf);
        op_reed.apply(&x, &mut y_reed);

        let norm_ref: f64 = y_mf.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff: f64 = y_mf.iter().zip(y_reed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(norm_ref > 1e-12, "reference norm too small");
        let rel_err = diff / norm_ref;
        assert!(rel_err < 1e-10,
            "HcurlReedOperator 2D rel_err = {rel_err:.3e} (expected < 1e-10)");
    }

    /// Check HcurlReedOperator for a 3D TetND1 mesh.
    #[test]
    fn hcurl_reed_matches_matfree_tet_nd1() {
        use fem_mesh::SimplexMesh;
        use fem_space::{HCurlSpace, fe_space::FESpace};

        let mesh  = SimplexMesh::<3>::unit_cube_tet(1);
        let space = HCurlSpace::new(mesh, 1);
        let n     = space.n_dofs();

        let mu_inv = 1.0_f64;
        let alpha  = 1.0_f64;
        let quad   = 3_u8;

        let op_mf   = HcurlMatrixFreeOperator::new(&space, mu_inv, alpha, quad);
        let op_reed = HcurlReedOperator::new(&space, mu_inv, alpha, quad);

        let x: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).cos()).collect();

        let mut y_mf   = vec![0.0_f64; n];
        let mut y_reed = vec![0.0_f64; n];
        op_mf.apply(&x, &mut y_mf);
        op_reed.apply(&x, &mut y_reed);

        let norm_ref: f64 = y_mf.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diff: f64 = y_mf.iter().zip(y_reed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(norm_ref > 1e-12, "reference norm too small");
        let rel_err = diff / norm_ref;
        assert!(rel_err < 1e-10,
            "HcurlReedOperator 3D rel_err = {rel_err:.3e} (expected < 1e-10)");
    }
}
