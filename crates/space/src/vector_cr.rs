//! Vector-valued Crouzeix–Raviart space (`[CR]^d`).
//!
//! Global DOFs are block-ordered (all x-DOFs, then y-DOFs, …).
//! Element DOFs are interleaved: `(x₀,y₀, x₁,y₁, …)`.
//!
//! This is the natural velocity space for Stokes with CR/P1 elements.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};
use crate::cr_space::CRSpace;

/// Vector-valued CR space: `dim` copies of [`CRSpace`].
///
/// For Stokes in 2D, `dim = 2` gives 2× n_edges velocity DOFs.
pub struct VectorCRSpace<M: MeshTopology> {
    mesh:   M,
    cr:     CRSpace<M>,
    order:  u8,
    dim:    u8,
    elem_dofs: Vec<DofId>,
    dofs_per_elem: usize,
}

impl<M: MeshTopology + Clone> VectorCRSpace<M> {
    pub fn new(mesh: M, order: u8, dim: u8) -> Self {
        let cr = CRSpace::new(mesh.clone(), order);
        let n_scalar = cr.n_dofs();
        let n_elems  = mesh.n_elements();
        let d = dim as usize;
        let n_ldofs = if n_elems > 0 { cr.element_dofs(0).len() } else { 0 };
        let dofs_per_elem = n_ldofs * d;
        let mut elem_dofs = Vec::with_capacity(n_elems * dofs_per_elem);
        for e in 0..n_elems as u32 {
            let s = cr.element_dofs(e);
            for k in 0..n_ldofs {
                for c in 0..d {
                    elem_dofs.push(c as DofId * n_scalar as DofId + s[k]);
                }
            }
        }
        VectorCRSpace { mesh, cr, order, dim, elem_dofs, dofs_per_elem }
    }
}

impl<M: MeshTopology> FESpace for VectorCRSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }
    fn n_dofs(&self) -> usize { self.cr.n_dofs() * self.dim as usize }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.elem_dofs[start..start + self.dofs_per_elem]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n_scalar = self.cr.n_dofs();
        let mut v = Vector::zeros(self.n_dofs());
        for c in 0..self.dim as usize {
            let off = c * n_scalar;
            let cv = self.cr.interpolate(f);
            for i in 0..n_scalar { v[off + i] = cv[i]; }
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::VectorH1(self.dim) }
    fn order(&self) -> u8 { self.order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn vector_cr_tri1_dof_count() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let vs = VectorCRSpace::new(m, 1, 2);
        assert!(vs.n_dofs() > 0);
        // n_vel = 2 * n_edges ≥ 16 for a 2×2 tri mesh
        assert!(vs.n_dofs() >= 16, "n_dofs={}", vs.n_dofs());
        // Per element: 3 edges × 2 components = 6
        assert_eq!(vs.element_dofs(0).len(), 6);
    }
}
