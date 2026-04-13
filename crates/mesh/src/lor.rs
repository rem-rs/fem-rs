//! LOR (Low-Order Refined) mesh utilities.
//!
//! This module provides a lightweight LOR mesh constructor for 2-D Tri3 meshes.
//! For a high-order discretization order `p`, we build a low-order surrogate by
//! applying `p-1` uniform red-refinement levels.

use crate::{SimplexMesh, refine_uniform};

/// Low-order refined mesh descriptor.
#[derive(Debug, Clone)]
pub struct LorMesh {
    /// Refined low-order mesh.
    pub mesh: SimplexMesh<2>,
    /// Original polynomial order.
    pub order: u8,
    /// Number of uniform refinement levels applied.
    pub levels: u8,
}

impl LorMesh {
    /// Build a 2-D LOR mesh from a high-order parent mesh.
    ///
    /// Current policy: apply `levels = max(p-1, 0)` uniform refinement passes.
    pub fn from_high_order(mesh: &SimplexMesh<2>, p: u8) -> Self {
        let levels = p.saturating_sub(1);
        let mut out = mesh.clone();
        for _ in 0..levels {
            out = refine_uniform(&out);
        }
        LorMesh { mesh: out, order: p, levels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lor_mesh_p1_is_identity() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let lor = LorMesh::from_high_order(&m, 1);
        assert_eq!(lor.levels, 0);
        assert_eq!(lor.mesh.n_elems(), m.n_elems());
        assert_eq!(lor.mesh.n_nodes(), m.n_nodes());
    }

    #[test]
    fn lor_mesh_refines_for_higher_order() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let lor = LorMesh::from_high_order(&m, 3);
        assert_eq!(lor.levels, 2);
        assert!(lor.mesh.n_elems() > m.n_elems());
    }
}
