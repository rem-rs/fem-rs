//! Discontinuous Lagrange (L²) finite element space.
//!
//! Each element has independent DOFs — no continuity across element boundaries.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

/// Scalar L² (discontinuous) finite element space.
///
/// - **P0** (`order = 0`): one DOF per element (piecewise constant).
/// - **P1** (`order = 1`): one DOF per element node, no inter-element sharing.
///   DOFs are numbered element-by-element.
pub struct L2Space<M: MeshTopology> {
    mesh:          M,
    order:         u8,
    /// `elem_dofs[e * dofs_per_elem .. (e+1) * dofs_per_elem]` = global DOF indices.
    elem_dofs:     Vec<DofId>,
    dofs_per_elem: usize,
    n_dofs:        usize,
    /// DOF node coordinates (flat, `n_dofs * dim`).
    dof_coords:    Vec<f64>,
}

impl<M: MeshTopology> L2Space<M> {
    /// Build the L² space of given order over `mesh`.
    ///
    /// Orders supported: 0 (P0 — element constant) and 1 (P1 — discontinuous).
    ///
    /// # Panics
    /// Panics if `order > 1`.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order <= 1, "L2Space: order {order} not supported (max 1)");
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();

        match order {
            0 => {
                // P0: 1 DOF per element, located at element centroid.
                let n_dofs = n_elems;
                let elem_dofs: Vec<DofId> = (0..n_elems as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];
                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    let base  = e as usize * dim;
                    for &n in nodes {
                        let c = mesh.node_coords(n);
                        for d in 0..dim { dof_coords[base + d] += c[d]; }
                    }
                    let npe = nodes.len() as f64;
                    for d in 0..dim { dof_coords[base + d] /= npe; }
                }
                L2Space { mesh, order, elem_dofs, dofs_per_elem: 1, n_dofs, dof_coords }
            }
            1 => {
                // P1 discontinuous: one DOF per node per element (no sharing).
                let npe   = mesh.element_nodes(0).len();
                let n_dofs = n_elems * npe;
                let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];
                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    for (k, &n) in nodes.iter().enumerate() {
                        let c    = mesh.node_coords(n);
                        let base = (e as usize * npe + k) * dim;
                        dof_coords[base .. base + dim].copy_from_slice(c);
                    }
                }
                L2Space { mesh, order, elem_dofs, dofs_per_elem: npe, n_dofs, dof_coords }
            }
            _ => unreachable!(),
        }
    }
}

impl<M: MeshTopology> FESpace for L2Space<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.elem_dofs[start .. start + self.dofs_per_elem]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let dim = self.mesh.dim() as usize;
        let n = self.n_dofs;
        let mut v = Vector::zeros(n);
        for dof in 0..n {
            let base   = dof * dim;
            let coords = &self.dof_coords[base .. base + dim];
            v.as_slice_mut()[dof] = f(coords);
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::L2 }

    fn order(&self) -> u8 { self.order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn l2_p0_n_dofs_equals_n_elems() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 0);
        assert_eq!(space.n_dofs(), n_elems);
    }

    #[test]
    fn l2_p0_element_dofs_are_sequential() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 0);
        for e in 0..space.mesh().n_elements() as u32 {
            let dofs = space.element_dofs(e);
            assert_eq!(dofs.len(), 1);
            assert_eq!(dofs[0], e);
        }
    }

    #[test]
    fn l2_p1_n_dofs_equals_n_elems_times_npe() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let npe = mesh.element_nodes(0).len();
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 1);
        assert_eq!(space.n_dofs(), n_elems * npe);
    }

    #[test]
    fn l2_p0_interpolate_constant() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 0);
        let v = space.interpolate(&|_x| 2.0);
        for &c in v.as_slice() {
            assert!((c - 2.0).abs() < 1e-14);
        }
    }
}
