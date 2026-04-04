//! Assembly loop for vector finite element spaces (H(curl), H(div)).
//!
//! [`VectorAssembler`] mirrors [`Assembler`](crate::assembler::Assembler) but
//! works with [`VectorReferenceElement`] instead of [`ReferenceElement`].
//! It applies Piola transforms and DOF orientation signs automatically.

use nalgebra::DMatrix;

use fem_element::reference::VectorReferenceElement;
use fem_element::nedelec::{TriND1, TetND1};
use fem_element::raviart_thomas::{TriRT0, TetRT0};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

use crate::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};

// ─── Reference element factory ──────────────────────────────────────────────

fn vec_ref_elem(space_type: SpaceType, dim: usize) -> Box<dyn VectorReferenceElement> {
    match (space_type, dim) {
        (SpaceType::HCurl, 2) => Box::new(TriND1),
        (SpaceType::HCurl, 3) => Box::new(TetND1),
        (SpaceType::HDiv,  2) => Box::new(TriRT0),
        (SpaceType::HDiv,  3) => Box::new(TetRT0),
        _ => panic!("vec_ref_elem: unsupported (space_type={space_type:?}, dim={dim})"),
    }
}

// ─── Jacobian helpers (same as assembler.rs) ────────────────────────────────

fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],
    dim: usize,
) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(geo_nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(geo_nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64> {
    let mut xp = x0.to_vec();
    for i in 0..dim {
        for k in 0..dim {
            xp[i] += j[(i, k)] * xi[k];
        }
    }
    xp
}

// ─── Piola transforms ───────────────────────────────────────────────────────

/// Covariant Piola transform for H(curl): φ_phys = J^{-T} φ_ref
///
/// Transforms `n_dofs` vector basis functions from reference to physical space.
fn piola_hcurl_basis(
    j_inv_t: &DMatrix<f64>,
    ref_vals: &[f64],    // [n_dofs × dim]
    phys_vals: &mut [f64], // [n_dofs × dim]
    n_dofs: usize,
    dim: usize,
) {
    for i in 0..n_dofs {
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim {
                s += j_inv_t[(r, c)] * ref_vals[i * dim + c];
            }
            phys_vals[i * dim + r] = s;
        }
    }
}

/// H(curl) curl transform.
///
/// - 2-D: `curl_phys[i] = curl_ref[i] / det_j` (scalar)
/// - 3-D: `curl_phys[i] = J · curl_ref[i] / det_j` (vector)
fn piola_hcurl_curl(
    jac: &DMatrix<f64>,
    det_j: f64,
    ref_curl: &[f64],
    phys_curl: &mut [f64],
    n_dofs: usize,
    dim: usize,
) {
    let inv_det = 1.0 / det_j;
    if dim == 2 {
        // Scalar curl
        for i in 0..n_dofs {
            phys_curl[i] = ref_curl[i] * inv_det;
        }
    } else {
        // 3-D vector curl: J · curl_ref / det_j
        for i in 0..n_dofs {
            for r in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += jac[(r, c)] * ref_curl[i * 3 + c];
                }
                phys_curl[i * 3 + r] = s * inv_det;
            }
        }
    }
}

/// Contravariant Piola transform for H(div): φ_phys = J φ_ref / det_j
fn piola_hdiv_basis(
    jac: &DMatrix<f64>,
    det_j: f64,
    ref_vals: &[f64],
    phys_vals: &mut [f64],
    n_dofs: usize,
    dim: usize,
) {
    let inv_det = 1.0 / det_j;
    for i in 0..n_dofs {
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim {
                s += jac[(r, c)] * ref_vals[i * dim + c];
            }
            phys_vals[i * dim + r] = s * inv_det;
        }
    }
}

/// H(div) divergence transform: div_phys = div_ref / det_j
fn piola_hdiv_div(
    det_j: f64,
    ref_div: &[f64],
    phys_div: &mut [f64],
    n_dofs: usize,
) {
    let inv_det = 1.0 / det_j;
    for i in 0..n_dofs {
        phys_div[i] = ref_div[i] * inv_det;
    }
}

/// Apply DOF orientation signs to all per-DOF arrays.
fn apply_signs(
    signs: &[f64],
    phi_vec: &mut [f64],   // [n_dofs × dim]
    curl: &mut [f64],
    div: &mut [f64],
    n_dofs: usize,
    dim: usize,
    curl_dim: usize,       // 1 for 2-D scalar curl, 3 for 3-D vector curl
) {
    for i in 0..n_dofs {
        let s = signs[i];
        for c in 0..dim {
            phi_vec[i * dim + c] *= s;
        }
        for c in 0..curl_dim {
            curl[i * curl_dim + c] *= s;
        }
        div[i] *= s;
    }
}

// ─── VectorAssembler ────────────────────────────────────────────────────────

/// Assembly driver for vector finite element spaces (H(curl), H(div)).
///
/// Applies Piola transforms and DOF orientation signs automatically.
pub struct VectorAssembler;

impl VectorAssembler {
    /// Assemble the global stiffness matrix for a vector bilinear form.
    pub fn assemble_bilinear<S: FESpace>(
        space: &S,
        integrators: &[&dyn VectorBilinearIntegrator],
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        let dim = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let stype = space.space_type();

        let ref_elem = vec_ref_elem(stype, dim);
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(quad_order);

        let curl_dim = if dim == 2 { 1 } else { 3 }; // scalar curl in 2D, vector in 3D

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

        // Pre-allocate work buffers.
        let mut ref_phi = vec![0.0; n_ldofs * dim];
        let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
        let mut ref_div = vec![0.0; n_ldofs];
        let mut phys_phi = vec![0.0; n_ldofs * dim];
        let mut phys_curl = vec![0.0; n_ldofs * curl_dim];
        let mut phys_div = vec![0.0; n_ldofs];

        for e in mesh.elem_iter() {
            let global_dofs: Vec<usize> =
                space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = space.element_signs(e);
            let nodes = mesh.element_nodes(e);

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse()
                .expect("degenerate element — zero-area/volume")
                .transpose();

            let mut k_elem = vec![0.0_f64; n_ldofs * n_ldofs];
            let x0 = mesh.node_coords(nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                // Evaluate reference basis, curl, div.
                ref_elem.eval_basis_vec(xi, &mut ref_phi);
                ref_elem.eval_curl(xi, &mut ref_curl);
                ref_elem.eval_div(xi, &mut ref_div);

                // Apply Piola transforms.
                match stype {
                    SpaceType::HCurl => {
                        piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_ldofs, dim);
                        piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_curl, n_ldofs, dim);
                        // H(curl) div is zero; copy as-is.
                        phys_div.copy_from_slice(&ref_div[..n_ldofs]);
                    }
                    SpaceType::HDiv => {
                        piola_hdiv_basis(&jac, det_j, &ref_phi, &mut phys_phi, n_ldofs, dim);
                        // H(div) curl is zero; copy as-is.
                        phys_curl[..ref_curl.len()].copy_from_slice(&ref_curl);
                        piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_ldofs);
                    }
                    _ => panic!("VectorAssembler: unsupported space type {stype:?}"),
                }

                // Apply DOF orientation signs.
                if let Some(s) = signs {
                    apply_signs(
                        s, &mut phys_phi, &mut phys_curl, &mut phys_div,
                        n_ldofs, dim, curl_dim,
                    );
                }

                let xp = phys_coords(x0, &jac, xi, dim);

                let qp = VectorQpData {
                    n_dofs: n_ldofs,
                    dim,
                    weight: w,
                    phi_vec: &phys_phi,
                    curl: &phys_curl,
                    div: &phys_div,
                    x_phys: &xp,
                };

                for integ in integrators {
                    integ.add_to_element_matrix(&qp, &mut k_elem);
                }
            }

            coo.add_element_matrix(&global_dofs, &k_elem);
        }

        coo.into_csr()
    }

    /// Assemble the global load vector for a vector linear form.
    pub fn assemble_linear<S: FESpace>(
        space: &S,
        integrators: &[&dyn VectorLinearIntegrator],
        quad_order: u8,
    ) -> Vec<f64> {
        let mesh = space.mesh();
        let dim = mesh.dim() as usize;
        let n_dofs = space.n_dofs();
        let stype = space.space_type();

        let ref_elem = vec_ref_elem(stype, dim);
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(quad_order);

        let curl_dim = if dim == 2 { 1 } else { 3 };

        let mut rhs = vec![0.0_f64; n_dofs];

        let mut ref_phi = vec![0.0; n_ldofs * dim];
        let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
        let mut ref_div = vec![0.0; n_ldofs];
        let mut phys_phi = vec![0.0; n_ldofs * dim];
        let mut phys_curl = vec![0.0; n_ldofs * curl_dim];
        let mut phys_div = vec![0.0; n_ldofs];

        for e in mesh.elem_iter() {
            let global_dofs: Vec<usize> =
                space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = space.element_signs(e);
            let nodes = mesh.element_nodes(e);

            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.clone().try_inverse().unwrap().transpose();

            let mut f_elem = vec![0.0_f64; n_ldofs];
            let x0 = mesh.node_coords(nodes[0]);

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();

                ref_elem.eval_basis_vec(xi, &mut ref_phi);
                ref_elem.eval_curl(xi, &mut ref_curl);
                ref_elem.eval_div(xi, &mut ref_div);

                match stype {
                    SpaceType::HCurl => {
                        piola_hcurl_basis(&j_inv_t, &ref_phi, &mut phys_phi, n_ldofs, dim);
                        piola_hcurl_curl(&jac, det_j, &ref_curl, &mut phys_curl, n_ldofs, dim);
                        phys_div.copy_from_slice(&ref_div[..n_ldofs]);
                    }
                    SpaceType::HDiv => {
                        piola_hdiv_basis(&jac, det_j, &ref_phi, &mut phys_phi, n_ldofs, dim);
                        phys_curl[..ref_curl.len()].copy_from_slice(&ref_curl);
                        piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_ldofs);
                    }
                    _ => panic!("VectorAssembler: unsupported space type {stype:?}"),
                }

                if let Some(s) = signs {
                    apply_signs(
                        s, &mut phys_phi, &mut phys_curl, &mut phys_div,
                        n_ldofs, dim, curl_dim,
                    );
                }

                let xp = phys_coords(x0, &jac, xi, dim);

                let qp = VectorQpData {
                    n_dofs: n_ldofs,
                    dim,
                    weight: w,
                    phi_vec: &phys_phi,
                    curl: &phys_curl,
                    div: &phys_div,
                    x_phys: &xp,
                };

                for integ in integrators {
                    integ.add_to_element_vector(&qp, &mut f_elem);
                }
            }

            for (&d, &v) in global_dofs.iter().zip(f_elem.iter()) {
                rhs[d] += v;
            }
        }

        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::HCurlSpace;

    #[test]
    fn vector_assembler_hcurl_matrix_size() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();

        struct Zero;
        impl VectorBilinearIntegrator for Zero {
            fn add_to_element_matrix(&self, _: &VectorQpData<'_>, _: &mut [f64]) {}
        }

        let mat = VectorAssembler::assemble_bilinear(&space, &[&Zero], 2);
        assert_eq!(mat.nrows, n);
        assert_eq!(mat.ncols, n);
    }

    #[test]
    fn vector_assembler_hcurl_linear_size() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();

        struct Zero;
        impl VectorLinearIntegrator for Zero {
            fn add_to_element_vector(&self, _: &VectorQpData<'_>, _: &mut [f64]) {}
        }

        let rhs = VectorAssembler::assemble_linear(&space, &[&Zero], 2);
        assert_eq!(rhs.len(), n);
    }
}
