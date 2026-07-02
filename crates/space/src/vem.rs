//! VEM (Virtual Element Method) finite element space for polygonal meshes.
//!
//! P1 VEM: one degree of freedom per polygon vertex.
//! `element_dofs(elem)` returns the same global indices as `element_nodes(elem)`.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

/// VEM finite element space on a polygonal mesh.
///
/// For P1 VEM, DOFs are at polygon vertices. Higher-order VEM (P2+, adding
/// edge and interior DOFs) is reserved for future work.
#[derive(Debug, Clone)]
pub struct VEMSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    /// Flat DOF array: element e has DOFs at `dofs_flat[e * dofs_per_elem ..]`.
    /// For P1 VEM this matches the mesh connectivity.
    dofs_flat: Vec<DofId>,
    dofs_per_elem: Vec<usize>, // per-element DOF count
}

impl<M: MeshTopology> VEMSpace<M> {
    /// Build a P1 VEM space on mesh `mesh`.
    ///
    /// Each polygon vertex is one DOF. The DOF indices match the mesh node indices.
    pub fn new(mesh: M) -> Self {
        let n_elems = mesh.n_elements();
        let mut dofs_flat = Vec::new();
        let mut dofs_per_elem = Vec::with_capacity(n_elems);
        for e in 0..n_elems as u32 {
            let nodes = mesh.element_nodes(e);
            for &n in nodes { dofs_flat.push(n); }
            dofs_per_elem.push(nodes.len());
        }
        Self { mesh, order: 1, dofs_flat, dofs_per_elem }
    }
}

impl<M: MeshTopology> VEMSpace<M> {
    /// Return the number of polygon vertices for element `elem`.
    pub fn element_n_vertices(&self, elem: u32) -> usize {
        self.mesh.element_nodes(elem).len()
    }
}

impl<M: MeshTopology + 'static> FESpace for VEMSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }
    fn n_dofs(&self) -> usize { self.mesh.n_nodes() }
    fn order(&self) -> u8 { self.order }
    fn space_type(&self) -> SpaceType { SpaceType::VEM }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let n = self.dofs_per_elem[elem as usize];
        let start: usize = self.dofs_per_elem[..elem as usize].iter().sum();
        &self.dofs_flat[start..start + n]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let mut result = Vector::zeros(n);
        for node in 0..n as u32 {
            let coord = self.mesh.node_coords(node);
            result.as_slice_mut()[node as usize] = f(coord);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::poly_mesh::PolyMesh;

    #[test]
    fn vem_space_quad_4x4() {
        let mesh = PolyMesh::unit_square_quad(3, 3);
        let space = VEMSpace::new(mesh);
        assert_eq!(space.n_dofs(), 16);
        assert_eq!(space.order(), 1);
        assert_eq!(space.space_type(), SpaceType::VEM);
        // Element 0 has 4 vertex DOFs = [0, 1, 4+1=5, 4] for a 4x4 quad mesh
        let dofs0 = space.element_dofs(0);
        assert_eq!(dofs0.len(), 4);
    }

    #[test]
    fn vem_space_interpolate_constant() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh);
        let v = space.interpolate(&|_| 3.0);
        for i in 0..space.n_dofs() {
            assert!((v.as_slice()[i] - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn vem_space_interpolate_linear() {
        let mesh = PolyMesh::unit_square_quad(2, 2);
        let space = VEMSpace::new(mesh);
        let v = space.interpolate(&|x| x[0] + 2.0 * x[1]);
        for i in 0..space.n_dofs() {
            let c = mesh.node_coords(i as u32);
            let expected = c[0] + 2.0 * c[1];
            assert!((v.as_slice()[i] - expected).abs() < 1e-14);
        }
    }
}
