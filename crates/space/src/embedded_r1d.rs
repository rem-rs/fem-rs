//! Embedded (restricted) H(curl) space on a **1-D mesh** — the space-layer
//! counterpart of MFEM's `ND_R1D_FECollection` (`fem/fe_coll.cpp:3093`,
//! MFEM 4.10): a segment chain carrying a 3-component vector field, as used
//! by the `dim == 1` branch of `examples/ex31.cpp`
//! (`FiniteElementSpace(mesh, ND_R1D_FECollection(order, 1))`).
//!
//! # Global dof layout (1:1 with MFEM `FiniteElementSpace` numbering)
//!
//! * **vertex dofs** — `ND_dof[POINT] = 2` (`ND_R1D_PointElement`: the
//!   y-directed and z-directed H¹ dofs; the tangential x component has *no*
//!   vertex dof): global ids `2v` (y) and `2v + 1` (z) per mesh vertex `v`
//!   in vertex order;
//! * **element-interior dofs** — `ND_dof[SEGMENT] = 3p − 2` per segment
//!   (the `ND_R1D_SegmentElement`'s `3p + 2` dofs minus its four vertex
//!   slots), assigned per segment in element order;
//! * `DofOrderForOrientation(SEGMENT, Or) = NULL` — segments never share
//!   segment-dofs in 1-D, so there is **no orientation bookkeeping** and no
//!   signed dofs.
//!
//! ## Deliberately *not* a `FESpace` impl
//!
//! Same rationale as [`crate::embedded_r2d`]: the generic vector assembler
//! would pair the regular ND segment element (different dof count/layout)
//! with this space and silently misassemble.  Consumers assemble through
//! `fem_element::embedded::NdR1dSegment` directly (see the ported `ex31`).

use fem_core::types::DofId;
use fem_element::embedded::NdR1dSegment;
use fem_mesh::topology::MeshTopology;

/// H(curl) space of MFEM's `ND_R1D_FECollection(order, 1)` on a segment mesh.
pub struct HCurlR1dSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_vertex_dofs: usize,
    /// Per-element interior-dof base (element order); block size `3p − 2`.
    interior_base: Vec<usize>,
    n_dofs: usize,
    seg: NdR1dSegment,
}

impl<M: MeshTopology> HCurlR1dSpace<M> {
    /// `FiniteElementSpace(mesh, ND_R1D_FECollection(order, 1))`.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order >= 1, "ND_R1D_FECollection requires order >= 1");
        let p = order as usize;
        let n_vertex_dofs = 2 * mesh.n_nodes();
        let block = 3 * p - 2;
        let n_elems = mesh.n_elements() as usize;
        let mut interior_base = Vec::with_capacity(n_elems);
        let mut base = n_vertex_dofs;
        for _ in 0..n_elems {
            interior_base.push(base);
            base += block;
        }
        HCurlR1dSpace {
            mesh,
            order,
            n_vertex_dofs,
            interior_base,
            n_dofs: base,
            seg: NdR1dSegment::new(p),
        }
    }

    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    pub fn order(&self) -> u8 {
        self.order
    }

    /// The shared `ND_R1D_SegmentElement` (reference data: `Nodes`, `dof2tk`).
    pub fn segment_element(&self) -> &NdR1dSegment {
        &self.seg
    }

    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Element dofs: the two vertices' `(y, z)` pairs in element vertex
    /// order, then the segment's `3p − 2` interior dofs.  All signs `+1`
    /// (no orientation table — MFEM returns `NULL` for `SEGMENT`).
    pub fn element_dofs(&self, e: u32) -> Vec<DofId> {
        let verts = self.mesh.element_nodes(e);
        let mut out = Vec::with_capacity(4 + self.interior_block());
        out.push(2 * verts[0] as DofId);
        out.push(2 * verts[0] as DofId + 1);
        out.push(2 * verts[1] as DofId);
        out.push(2 * verts[1] as DofId + 1);
        let base = self.interior_base[e as usize];
        for k in 0..self.interior_block() {
            out.push((base + k) as DofId);
        }
        out
    }

    /// `GetEssentialTrueDofs` over the given boundary tags: every boundary
    /// **POINT** whose tag is marked contributes both vertex dofs `(y, z)`,
    /// ascending (MFEM's essential list is ascending).
    pub fn boundary_dofs(&self, tags: &[i32]) -> Vec<DofId> {
        let mut set = std::collections::BTreeSet::new();
        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if tags.contains(&self.mesh.face_tag(f)) {
                for &n in self.mesh.face_nodes(f) {
                    set.insert(2 * n as DofId);
                    set.insert(2 * n as DofId + 1);
                }
            }
        }
        set.into_iter().collect()
    }

    /// MFEM `GridFunction::ProjectCoefficient` →
    /// `ND_R1D_SegmentElement::Project(VectorCoefficient&)` (`fe_nd.cpp:2766`):
    /// for dof slot `k` with tangent row `tk[dof2tk[k]]` (identity rows),
    /// `dofs(k) = J00·tk[0]·E_x(x_k) + tk[1]·E_y(x_k) + tk[2]·E_z(x_k)` where
    /// `x_k` is the physical node position and `J00 = dx/dξ`.  Shared vertex
    /// dofs are overwritten per element (MFEM `SetSubVector` last-write;
    /// the values coincide across neighbours — plain point evaluations).
    ///
    /// `elem_geo` returns `(J00, x_phys)` for a reference coordinate `ξ`.
    pub fn project(
        &self,
        field: &dyn Fn([f64; 1]) -> [f64; 3],
        elem_geo: &dyn Fn(u32, f64) -> (f64, [f64; 1]),
    ) -> Vec<f64> {
        let nd = self.seg.n_dofs();
        let mut x = vec![0.0_f64; self.n_dofs];
        let nodes = self.seg.nodes().to_vec();
        let dof2tk = self.seg.dof2tk().to_vec();
        for e in 0..self.mesh.n_elements() as u32 {
            let el_dofs = self.element_dofs(e);
            for k in 0..nd {
                let xi = nodes[k];
                let (j00, xp) = elem_geo(e, xi);
                let ev = field(xp);
                let v = match dof2tk[k] {
                    0 => j00 * ev[0],
                    1 => ev[1],
                    _ => ev[2],
                };
                x[el_dofs[k] as usize] = v;
            }
        }
        x
    }

    fn interior_block(&self) -> usize {
        3 * self.order as usize - 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout smoke test on a 2-segment chain built directly: 3 vertices →
    /// 6 vertex dofs, 2 × (3p−2) interior dofs (p = 1: 1 each → 8 total).
    #[test]
    fn d104_hcurl_r1d_layout_matches_mfem_dof_table() {
        // MFEM ND_R1D_FECollection(1, 1): ND_dof[POINT] = 2,
        // ND_dof[SEGMENT] = 3·1 − 2 = 1.
        let coords = vec![0.0_f64, 0.5, 1.0];
        let conn = vec![0u32, 1u32, 1u32, 2u32];
        let face_conn = vec![0u32, 2u32]; // boundary POINTs at both ends
        let mesh = fem_mesh::Mesh::<1>::uniform(
            coords,
            conn,
            vec![1, 1],
            fem_mesh::ElementType::Line2,
            face_conn,
            vec![1, 2],
            fem_mesh::ElementType::Point1,
        );
        let space = HCurlR1dSpace::new(mesh, 1);
        assert_eq!(space.n_dofs(), 2 * 3 + 2 * (3 - 2));

        // Element 0 dofs: vertices (y,z) of v0, v1, then 1 interior dof.
        assert_eq!(
            space.element_dofs(0),
            vec![0, 1, 2, 3, 6]
        );
        assert_eq!(space.element_dofs(1), vec![2, 3, 4, 5, 7]);

        // Essential dofs for both boundary POINTs (tags 1 and 2).
        assert_eq!(space.boundary_dofs(&[1, 2]), vec![0, 1, 4, 5]);
    }
}
