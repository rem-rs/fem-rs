//! Element geometry transformation utilities.
//!
//! This module provides a lightweight wrapper equivalent to MFEM's
//! `ElementTransformation` for affine simplex elements.

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
        for col in 0..dim {
            let xc = mesh.node_coords(geo_nodes[col + 1]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimplexMesh;

    #[test]
    fn tri2d_det_and_map() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let tr = ElementTransformation::from_simplex(&mesh, 0);
        assert_eq!(tr.dim(), 2);
        assert!(tr.det_j().abs() > 1e-14);

        // Reference centroid for triangle.
        let x = tr.map_to_physical(&[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(x.len(), 2);
    }
}
