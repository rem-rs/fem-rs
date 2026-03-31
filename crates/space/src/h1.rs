//! Continuous Lagrange (H¹) finite element space.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::DofManager;
use crate::fe_space::{FESpace, SpaceType};

/// Scalar H¹ finite element space using continuous Lagrange basis functions.
///
/// Constructed from any [`MeshTopology`] plus a polynomial order (1 or 2).
///
/// - **P1**: one DOF per mesh node; globally C⁰ piecewise-linear.
/// - **P2**: one DOF per node plus one per mesh edge; globally C⁰ piecewise-quadratic.
pub struct H1Space<M: MeshTopology> {
    mesh:   M,
    dm:     DofManager,
    order:  u8,
}

impl<M: MeshTopology> H1Space<M> {
    /// Construct a new H¹ space of the given polynomial order on `mesh`.
    pub fn new(mesh: M, order: u8) -> Self {
        let dm = DofManager::new(&mesh, order);
        H1Space { mesh, dm, order }
    }

    /// Reference to the DOF manager.
    pub fn dof_manager(&self) -> &DofManager { &self.dm }
}

impl<M: MeshTopology> FESpace for H1Space<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.dm.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        self.dm.element_dofs(elem)
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.dm.n_dofs;
        let dim = self.dm.dim;
        let mut v = Vector::zeros(n);
        for dof in 0..n as u32 {
            let coords = self.dm.dof_coord(dof);
            v.as_slice_mut()[dof as usize] = f(&coords[..dim]);
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 { self.order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn h1_p1_n_dofs_equals_n_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        assert_eq!(space.n_dofs(), space.mesh().n_nodes());
    }

    #[test]
    fn h1_p2_n_dofs_greater_than_n_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_nodes = mesh.n_nodes();
        let space = H1Space::new(mesh, 2);
        assert!(space.n_dofs() > n_nodes);
    }

    #[test]
    fn h1_interpolate_constant() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|_x| 3.14);
        for &c in v.as_slice() {
            assert!((c - 3.14).abs() < 1e-14);
        }
    }

    #[test]
    fn h1_interpolate_linear_x() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| x[0]);
        // All DOF values should be in [0,1].
        for &c in v.as_slice() {
            assert!(c >= -1e-14 && c <= 1.0 + 1e-14);
        }
    }

    #[test]
    fn h1_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = H1Space::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::H1);
    }
}
