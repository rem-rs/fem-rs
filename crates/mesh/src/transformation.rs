//! Element geometry transformation utilities.
//!
//! Provides:
//! - [`ElementTransformation`] — affine simplex transformation (MFEM `ElementTransformation`)
//! - [`geometry_jacobian`] — compute Jacobian at a reference point for any element type
//! - [`xform_grads`] — transform reference gradients to physical space

use fem_core::ElemId;
use nalgebra::DMatrix;

use crate::topology::MeshTopology;

/// Affine element transformation for simplex geometries.
///
/// For a simplex with vertex coordinates `x0, x1, ..., x_dim`,
/// `J[:,k] = x_{k+1} - x_0` and `x(ξ) = x0 + J ξ`.
#[derive(Debug, Clone)]
pub struct ElementTransformation {
    dim: usize,
    x0: Vec<f64>,
    jacobian: DMatrix<f64>,
    det_j: f64,
    jacobian_inv_t: DMatrix<f64>,
}

impl ElementTransformation {
    /// Build a simplex transformation from mesh element id.
    pub fn from_simplex<M: MeshTopology>(mesh: &M, elem: ElemId) -> Self {
        let nodes = mesh.element_nodes(elem);
        Self::from_simplex_nodes(mesh, nodes)
    }

    /// Build a simplex transformation from a node slice.
    ///
    /// Uses the first `dim + 1` nodes as simplex vertices.
    pub fn from_simplex_nodes<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> Self {
        let dim = mesh.dim() as usize;
        assert!(
            geo_nodes.len() > dim,
            "ElementTransformation::from_simplex_nodes: need at least dim+1 nodes"
        );

        let x0 = mesh.node_coords(geo_nodes[0]).to_vec();
        let mut jac = DMatrix::<f64>::zeros(dim, dim);
        // Column order must match the reference-element axes of the SOLUTION
        // basis.  For Tet4/Hex8 the node order is the axis order, but for
        // Prism6 the PrismPk reference is (ξ0 = layer xi, ξ1 = tri eta,
        // ξ2 = tri zeta) with vertices [0,1,2,3,4,5] =
        // (0,0,0),(0,1,0),(0,0,1),(1,0,0),(1,1,0),(1,0,1) — so ∂x/∂ξ0 comes
        // from vertex 3, ∂x/∂ξ1 from vertex 1, ∂x/∂ξ2 from vertex 2.
        let col_of: Vec<usize> = match geo_nodes.len() {
            6 => vec![3, 1, 2], // Prism6: (ξ0, ξ1, ξ2) = (layer, tri-eta, tri-zeta)
            _ => (0..dim).map(|i| i + 1).collect(),
        };
        for col in 0..dim {
            let xc = mesh.node_coords(geo_nodes[col_of[col]]);
            for row in 0..dim {
                jac[(row, col)] = xc[row] - x0[row];
            }
        }

        let det_j = jac.determinant();
        let jacobian_inv_t = jac
            .clone()
            .try_inverse()
            .expect("ElementTransformation: degenerate simplex element")
            .transpose();

        Self {
            dim,
            x0,
            jacobian: jac,
            det_j,
            jacobian_inv_t,
        }
    }

    /// Spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Jacobian matrix `J`.
    pub fn jacobian(&self) -> &DMatrix<f64> {
        &self.jacobian
    }

    /// Jacobian determinant `det(J)`.
    pub fn det_j(&self) -> f64 {
        self.det_j
    }

    /// Inverse-transpose Jacobian `J^{-T}`.
    pub fn jacobian_inv_t(&self) -> &DMatrix<f64> {
        &self.jacobian_inv_t
    }

    /// Reference-to-physical map for affine simplex elements.
    pub fn map_to_physical(&self, xi: &[f64]) -> Vec<f64> {
        assert_eq!(
            xi.len(),
            self.dim,
            "ElementTransformation::map_to_physical: xi dimension mismatch"
        );
        let mut xp = self.x0.clone();
        for i in 0..self.dim {
            for k in 0..self.dim {
                xp[i] += self.jacobian[(i, k)] * xi[k];
            }
        }
        xp
    }
}

/// Compute the geometry Jacobian determinant and inverse-transpose at a
/// reference point for a mesh element of any type (MFEM: `ElementTransformation`).
///
/// Returns `(detJ, J^{-T})` where `J_{ij} = ∂x_i/∂ξ_j` is the Jacobian of
/// the reference-to-physical mapping, computed from the element's nodal
/// coordinates and the linear (P1) reference-element gradient basis.
///
/// Supports all element types: Tri3, Quad4, Tet4, Hex8, Prism6, etc.
///
/// # Panics
/// Panics if the element's geometry Jacobian is singular.
pub fn geometry_jacobian(
    mesh: &dyn MeshTopology,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (f64, DMatrix<f64>) {
    let et = mesh.element_type(elem);
    let nd = mesh.element_nodes(elem);
    let n_ldofs = nd.len();
    let re_geom = et.ref_elem(1);
    let mut grad = vec![0.0_f64; n_ldofs * dim];
    re_geom.eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..n_ldofs {
        let x = mesh.node_coords(nd[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += x[i] * grad[k * dim + j];
            }
        }
    }
    let det = jac.determinant();
    let inv = jac.try_inverse().expect("singular Jacobian in geometry_jacobian");
    (det, inv.transpose())
}

/// Transform reference-element gradients to physical space:
/// `∇_phys = J^{-T} ∇_ref` (MFEM: `ElementTransformation` gradient transform).
pub fn xform_grads(ji: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for a in 0..n {
        for j in 0..dim {
            gp[a * dim + j] = (0..dim).map(|k| ji[(j, k)] * gr[a * dim + k]).sum();
        }
    }
}

/// Compute the full Jacobian matrix `J` and the physical point `x_phys`
/// at a reference point for a given element.
///
/// Returns `(J, x_phys)` where `J_{ij} = ∂x_i/∂ξ_j` and `x_phys = x(ξ)`.
/// Uses the linear (P1) reference element for the geometry mapping.
///
/// Supported element types: Tri3, Quad4, Tet4, Hex8, Prism6.
///
/// MFEM: `ElementTransformation::Jacobian()` + `ElementTransformation::Transform()`
pub fn element_jacobian_at<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, Vec<f64>) {
    let et = mesh.element_type(elem);
    let re = et.ref_elem(1);
    let npe = re.n_dofs();
    let mut grad = vec![0.0_f64; npe * dim];
    let mut phi = vec![0.0_f64; npe];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    let nodes = mesh.element_nodes(elem);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];
    for k in 0..npe {
        let c = mesh.node_coords(nodes[k]);
        for i in 0..dim {
            xp[i] += c[i] * phi[k];
            for j in 0..dim {
                jac[(i, j)] += c[i] * grad[k * dim + j];
            }
        }
    }
    (jac, xp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    #[test]
    fn tri2d_det_and_map() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let tr = ElementTransformation::from_simplex(&mesh, 0);
        assert_eq!(tr.dim(), 2);
        assert!(tr.det_j().abs() > 1e-14);

        // Reference centroid for triangle.
        let x = tr.map_to_physical(&[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(x.len(), 2);
    }
}
