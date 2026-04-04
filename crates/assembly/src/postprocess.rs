//! Post-processing utilities: element-wise gradient, curl, divergence, and
//! nodal gradient recovery (Zienkiewicz-Zhu style).

use nalgebra::DMatrix;

use fem_element::lagrange::{TetP1, TriP1, TriP2};
use fem_element::nedelec::{TetND1, TriND1};
use fem_element::raviart_thomas::{TetRT0, TriRT0};
use fem_element::reference::VectorReferenceElement;
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

// ─── Reference element factories ───────────────────────────────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

fn vec_ref_elem(space_type: SpaceType, dim: usize) -> Box<dyn VectorReferenceElement> {
    match (space_type, dim) {
        (SpaceType::HCurl, 2) => Box::new(TriND1),
        (SpaceType::HCurl, 3) => Box::new(TetND1),
        (SpaceType::HDiv, 2) => Box::new(TriRT0),
        (SpaceType::HDiv, 3) => Box::new(TetRT0),
        _ => panic!("vec_ref_elem: unsupported (space_type={space_type:?}, dim={dim})"),
    }
}

// ─── Jacobian helpers ──────────────────────────────────────────────────────

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

fn transform_grads(
    j_inv_t: &DMatrix<f64>,
    grad_ref: &[f64],
    grad_phys: &mut [f64],
    n_ldofs: usize,
    dim: usize,
) {
    for i in 0..n_ldofs {
        for d in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(d, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + d] = s;
        }
    }
}

// ─── Element-wise gradient (scalar H1/L2 spaces) ──────────────────────────

/// Compute element-wise gradient of a scalar FE solution.
///
/// Evaluates the gradient at the element centroid.
/// Returns a `Vec` of length `n_elements`, each entry is `[du/dx, du/dy, ...]`.
pub fn compute_element_gradients<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let mut result = Vec::with_capacity(mesh.n_elements());

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, _det_j) = simplex_jacobian(mesh, nodes, dim);
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Evaluate at centroid: (1/3, 1/3) for tri, (1/4, 1/4, 1/4) for tet.
        let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }
        result.push(grad);
    }

    result
}

// ─── Element-wise curl (H(curl) spaces) ────────────────────────────────────

/// Compute element-wise curl of an H(curl) FE solution.
///
/// Evaluates at the element centroid.
/// For 2D: each entry is a `Vec<f64>` of length 1 (scalar curl).
/// For 3D: each entry is a `Vec<f64>` of length 3 (vector curl).
pub fn compute_element_curl<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let ref_elem = vec_ref_elem(stype, dim);
    let n_ldofs = ref_elem.n_dofs();
    let curl_dim = if dim == 2 { 1 } else { 3 };

    let mut result = Vec::with_capacity(mesh.n_elements());

    // Centroid in reference coordinates.
    let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

    let mut ref_curl = vec![0.0; n_ldofs * curl_dim];
    let mut phys_curl = vec![0.0; n_ldofs * curl_dim];

    for e in mesh.elem_iter() {
        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);

        ref_elem.eval_curl(&xi, &mut ref_curl);

        // H(curl) curl transform: curl_phys = J * curl_ref / det_j (3D)
        //                          curl_phys = curl_ref / det_j (2D scalar)
        let inv_det = 1.0 / det_j;
        if dim == 2 {
            for i in 0..n_ldofs {
                phys_curl[i] = ref_curl[i] * inv_det;
            }
        } else {
            for i in 0..n_ldofs {
                for r in 0..3 {
                    let mut s = 0.0;
                    for c in 0..3 {
                        s += jac[(r, c)] * ref_curl[i * 3 + c];
                    }
                    phys_curl[i * 3 + r] = s * inv_det;
                }
            }
        }

        // Apply orientation signs.
        if let Some(s) = signs {
            for i in 0..n_ldofs {
                for c in 0..curl_dim {
                    phys_curl[i * curl_dim + c] *= s[i];
                }
            }
        }

        // Sum contributions: curl(u_h) = Σ_i c_i curl(φ_i)
        let mut curl_val = vec![0.0; curl_dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..curl_dim {
                curl_val[d] += c * phys_curl[i * curl_dim + d];
            }
        }
        result.push(curl_val);
    }

    result
}

// ─── Element-wise divergence (H(div) spaces) ──────────────────────────────

/// Compute element-wise divergence of an H(div) FE solution.
///
/// Evaluates at the element centroid.
/// Returns a `Vec<f64>` of length `n_elements`.
pub fn compute_element_divergence<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<f64> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let stype = space.space_type();

    let ref_elem = vec_ref_elem(stype, dim);
    let n_ldofs = ref_elem.n_dofs();

    let mut result = Vec::with_capacity(mesh.n_elements());

    let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];

    let mut ref_div = vec![0.0; n_ldofs];
    let mut phys_div = vec![0.0; n_ldofs];

    for e in mesh.elem_iter() {
        let elem_dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        let (_jac, det_j) = simplex_jacobian(mesh, nodes, dim);

        ref_elem.eval_div(&xi, &mut ref_div);

        // H(div) divergence transform: div_phys = div_ref / det_j
        let inv_det = 1.0 / det_j;
        for i in 0..n_ldofs {
            phys_div[i] = ref_div[i] * inv_det;
        }

        // Apply orientation signs.
        if let Some(s) = signs {
            for i in 0..n_ldofs {
                phys_div[i] *= s[i];
            }
        }

        // Sum: div(u_h) = Σ_i c_i div(φ_i)
        let mut div_val = 0.0;
        for i in 0..n_ldofs {
            div_val += dofs[elem_dofs[i] as usize] * phys_div[i];
        }
        result.push(div_val);
    }

    result
}

// ─── Nodal gradient recovery (Zienkiewicz-Zhu) ────────────────────────────

/// Nodal gradient recovery via area-weighted averaging (Zienkiewicz-Zhu style).
///
/// For a scalar H1 solution, computes the smoothed gradient at each mesh node
/// by averaging the element-wise constant gradients weighted by element area.
///
/// Returns a `Vec` of `dim` vectors, each of length `n_nodes`:
/// `result[d][node]` = d-th component of the recovered gradient at `node`.
pub fn recover_gradient_nodal<S: FESpace>(space: &S, dofs: &[f64]) -> Vec<Vec<f64>> {
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    let order = space.order();

    // Accumulate area-weighted gradient at each node.
    let mut grad_accum = vec![vec![0.0; n_nodes]; dim];
    let mut area_accum = vec![0.0; n_nodes];

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();

        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
        let j_inv_t = jac.try_inverse().expect("degenerate element").transpose();

        // Element area/volume (for a simplex: |det_j| / d!)
        let elem_area = det_j.abs() / match dim {
            2 => 2.0,
            3 => 6.0,
            _ => 1.0,
        };

        // Gradient at centroid.
        let xi: Vec<f64> = vec![1.0 / (dim as f64 + 1.0); dim];
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut grad_phys = vec![0.0; n_ldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

        let mut grad = vec![0.0; dim];
        for i in 0..n_ldofs {
            let c = dofs[elem_dofs[i] as usize];
            for d in 0..dim {
                grad[d] += c * grad_phys[i * dim + d];
            }
        }

        // Distribute to element vertices (first dim+1 nodes are vertices).
        let n_verts = dim + 1;
        for v in 0..n_verts {
            let node = nodes[v] as usize;
            area_accum[node] += elem_area;
            for d in 0..dim {
                grad_accum[d][node] += elem_area * grad[d];
            }
        }
    }

    // Divide by total accumulated weight.
    for node in 0..n_nodes {
        if area_accum[node] > 0.0 {
            for d in 0..dim {
                grad_accum[d][node] /= area_accum[node];
            }
        }
    }

    grad_accum
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, HCurlSpace, HDivSpace, fe_space::FESpace};

    #[test]
    fn element_gradients_linear_function() {
        // u(x,y) = 3x - 2y → ∇u = [3, -2] everywhere (constant per element).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grads = compute_element_gradients(&space, dofs);
        assert_eq!(grads.len(), space.mesh().n_elements());

        for (e, g) in grads.iter().enumerate() {
            assert!(
                (g[0] - 3.0).abs() < 1e-10,
                "elem {e}: ∂u/∂x = {}, expected 3.0",
                g[0]
            );
            assert!(
                (g[1] + 2.0).abs() < 1e-10,
                "elem {e}: ∂u/∂y = {}, expected -2.0",
                g[1]
            );
        }
    }

    #[test]
    fn recover_gradient_nodal_linear() {
        // For u(x,y) = 3x - 2y, the recovered gradient at every node should
        // be [3, -2] exactly (since all contributing elements have the same gradient).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| 3.0 * x[0] - 2.0 * x[1]);
        let dofs = v.as_slice();

        let grad = recover_gradient_nodal(&space, dofs);
        assert_eq!(grad.len(), 2); // dim = 2
        assert_eq!(grad[0].len(), space.mesh().n_nodes());

        for node in 0..space.mesh().n_nodes() {
            assert!(
                (grad[0][node] - 3.0).abs() < 1e-10,
                "node {node}: ∂u/∂x = {}, expected 3.0",
                grad[0][node]
            );
            assert!(
                (grad[1][node] + 2.0).abs() < 1e-10,
                "node {node}: ∂u/∂y = {}, expected -2.0",
                grad[1][node]
            );
        }
    }

    #[test]
    fn element_curl_hcurl_basic() {
        // Sanity check: compute element curls on a simple mesh, verify the
        // output has the right length.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let n = space.n_dofs();
        let dofs = vec![0.0; n]; // zero field → curl should be zero

        let curls = compute_element_curl(&space, &dofs);
        assert_eq!(curls.len(), space.mesh().n_elements());

        for (e, c) in curls.iter().enumerate() {
            assert_eq!(c.len(), 1, "2D curl should be scalar");
            assert!(
                c[0].abs() < 1e-12,
                "elem {e}: curl of zero field should be 0, got {}",
                c[0]
            );
        }
    }

    #[test]
    fn element_divergence_hdiv_basic() {
        // Sanity check: compute element divergences for zero field.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let n = space.n_dofs();
        let dofs = vec![0.0; n];

        let divs = compute_element_divergence(&space, &dofs);
        assert_eq!(divs.len(), space.mesh().n_elements());

        for (e, &d) in divs.iter().enumerate() {
            assert!(
                d.abs() < 1e-12,
                "elem {e}: div of zero field should be 0, got {d}"
            );
        }
    }
}
