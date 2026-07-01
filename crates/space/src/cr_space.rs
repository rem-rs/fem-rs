//! Crouzeix–Raviart (CR) finite element space.
//!
//! CR DOFs reside at **edge midpoints** (TriCR1, TriCR2) or **face centroids**
//! (TetCR1, TetCR2), making this a **non‑conforming** H¹ space suitable for
//! Stokes velocity discretisation.
//!
//! # DOF numbering
//!
//! | element | order | DOFs per element | entity |
//! |---------|-------|-----------------|--------|
//! | TriCR1  | 1     | 3               | 1 per edge (average) |
//! | TriCR2  | 2     | 6               | 2 per edge (avg + moment) |
//! | TetCR1  | 1     | 4               | 1 per face (average) |
//! | TetCR2  | 2     | 10              | 4 face‑avg + 6 edge‑moment |
//!
//! The `order` field selects CR1 (`order = 1`) or CR2 (`order = 2`).

use std::collections::HashMap;
use fem_core::types::{DofId, NodeId};
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;
use fem_mesh::element_type::ElementType;

use crate::dof_manager::EdgeKey;
use crate::fe_space::{FESpace, SpaceType};

/// Face key (sorted 3‑tuple) → first DofId for 3‑D CR spaces.
type FaceKey3 = (NodeId, NodeId, NodeId);
fn fk3(a: NodeId, b: NodeId, c: NodeId) -> FaceKey3 {
    let mut v = [a, b, c]; v.sort(); (v[0], v[1], v[2])
}

/// Crouzeix‑Raviart finite element space.
///
/// # Type parameter
/// `D` — spatial dimension (2 for triangles, 3 for tetrahedra).
#[derive(Clone)]
pub struct CRSpace<M: MeshTopology> {
    mesh:      M,
    order:     u8,
    n_dofs:    usize,
    elem_dofs: Vec<Vec<DofId>>,
}

impl<M: MeshTopology> CRSpace<M> {
    /// Build a CR space of the given order on `mesh`.
    ///
    /// # Panics
    /// Panics if the mesh dimension and element type are incompatible with CR.
    pub fn new(mesh: M, order: u8) -> Self {
        let dim = mesh.dim();
        let n_elems = mesh.n_elements();
        let mut elem_dofs = Vec::with_capacity(n_elems);
        let mut next = 0u32;

        match dim {
            2 => {
                // Tri3 edges: 1 DOF per edge (CR1) or 2 DOFs per edge (CR2)
                let dofs_per_edge = order as usize; // 1 for CR1, 2 for CR2
                let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();

                for e in mesh.elem_iter() {
                    let ns = mesh.element_nodes(e);
                    assert_eq!(mesh.element_type(e), ElementType::Tri3,
                        "CRSpace<2>: Tri3 required");
                    let pairs = [(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[0])];
                    let mut edofs = Vec::with_capacity(3 * dofs_per_edge);
                    for &(a, b) in &pairs {
                        let key = EdgeKey::new(a, b);
                        let first = *edge_map.entry(key).or_insert_with(|| {
                            let id = next; next += dofs_per_edge as DofId; id
                        });
                        for k in 0..dofs_per_edge {
                            edofs.push(first + k as DofId);
                        }
                    }
                    elem_dofs.push(edofs);
                }

                CRSpace { mesh, order, n_dofs: next as usize, elem_dofs }
            }
            3 => {
                // Tet4: faces for CR1, faces+edges for CR2
                assert!(order == 1 || order == 2,
                    "CRSpace<3>: only order 1 or 2 supported");
                let mut face_map: HashMap<FaceKey3, DofId> = HashMap::new();
                let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();

                for e in mesh.elem_iter() {
                    assert_eq!(mesh.element_type(e), ElementType::Tet4,
                        "CRSpace<3>: Tet4 required");
                    let ns = mesh.element_nodes(e);
                    // 4 faces (one opposite each vertex)
                    let faces = [
                        (ns[1], ns[2], ns[3]),
                        (ns[0], ns[2], ns[3]),
                        (ns[0], ns[1], ns[3]),
                        (ns[0], ns[1], ns[2]),
                    ];
                    let mut edofs = Vec::new();
                    for &(a, b, c) in &faces {
                        let key = fk3(a, b, c);
                        let fid = *face_map.entry(key).or_insert_with(|| {
                            let id = next; next += 1; id
                        });
                        edofs.push(fid);
                    }
                    if order == 2 {
                        // 6 edge-moment DOFs
                        let edges = [
                            (ns[0], ns[1]), (ns[0], ns[2]), (ns[0], ns[3]),
                            (ns[1], ns[2]), (ns[1], ns[3]), (ns[2], ns[3]),
                        ];
                        for &(a, b) in &edges {
                            let key = EdgeKey::new(a, b);
                            let eid = *edge_map.entry(key).or_insert_with(|| {
                                let id = next; next += 1; id
                            });
                            edofs.push(eid);
                        }
                    }
                    elem_dofs.push(edofs);
                }

                CRSpace { mesh, order, n_dofs: next as usize, elem_dofs }
            }
            _ => panic!("CRSpace: unsupported dimension {dim}"),
        }
    }
}

impl<M: MeshTopology> FESpace for CRSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }
    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        &self.elem_dofs[elem as usize]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let mut v = Vector::zeros(self.n_dofs);
        let dim = self.mesh.dim() as usize;

        for e in self.mesh.elem_iter() {
            let ns = self.mesh.element_nodes(e);
            let dofs = &self.elem_dofs[e as usize];
            let npe = dofs.len();

            if dim == 2 {
                // CR1/CR2 on triangles: evaluate at edge midpoints / edge-moment pts
                let pairs = [(ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[0])];
                let mut di = 0;
                for &(a, b) in &pairs {
                    let mid = [
                        (self.mesh.node_coords(a)[0] + self.mesh.node_coords(b)[0]) * 0.5,
                        (self.mesh.node_coords(a)[1] + self.mesh.node_coords(b)[1]) * 0.5,
                    ];
                    v[dofs[di] as usize] = f(&mid);
                    di += 1;
                    if self.order == 2 {
                        // Second edge-moment DOF: use same midpoint (basis is moment, not nodal)
                        // For simplicity, evaluate the function again; the actual moment
                        // value is better computed via quadrature.
                        v[dofs[di] as usize] = f(&mid);
                        di += 1;
                    }
                }
            } else {
                // TetCR1/CR2: evaluate at face centroids
                let faces = [
                    (ns[1], ns[2], ns[3]),
                    (ns[0], ns[2], ns[3]),
                    (ns[0], ns[1], ns[3]),
                    (ns[0], ns[1], ns[2]),
                ];
                for &(a, b, c) in &faces[..4.min(npe)] {
                    let mut ctr = [0.0_f64; 3];
                    for n in &[a, b, c] {
                        let x = self.mesh.node_coords(*n);
                        for d in 0..3 { ctr[d] += x[d]; }
                    }
                    for d in 0..3 { ctr[d] /= 3.0; }
                    let idx = dofs[0];
                    v[idx as usize] = f(&ctr);
                }
                if self.order == 2 && npe > 4 {
                    // Edge-moment DOFs: evaluate at edge midpoints (approximate)
                    let edges = [
                        (ns[0], ns[1]), (ns[0], ns[2]), (ns[0], ns[3]),
                        (ns[1], ns[2]), (ns[1], ns[3]), (ns[2], ns[3]),
                    ];
                    for (k, &(a, b)) in edges.iter().enumerate() {
                        if 4 + k < npe {
                            let mut mid = [0.0_f64; 3];
                            for d in 0..3 {
                                mid[d] = (self.mesh.node_coords(a)[d]
                                        + self.mesh.node_coords(b)[d]) * 0.5;
                            }
                            v[dofs[4 + k] as usize] = f(&mid);
                        }
                    }
                }
            }
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }  // non-conforming H¹
    fn order(&self) -> u8 { self.order }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn crspace_tri1_dof_count() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let s = CRSpace::new(m, 1);
        assert!(s.n_dofs() > 0);
        assert_eq!(s.order(), 1);
    }

    #[test]
    fn crspace_tri2_dof_count() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let s = CRSpace::new(m, 2);
        assert_eq!(s.order(), 2);
        assert!(s.n_dofs() > 0);
        // CR2: 2 DOFs per edge
        let n_edges_est = s.n_dofs() / 2;
        assert!(n_edges_est >= 8);
    }

    #[test]
    fn crspace_per_element_dofs_tri() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let s = CRSpace::new(m, 1);
        assert_eq!(s.element_dofs(0).len(), 3, "TriCR1: 3 DOFs per element");
    }

    #[test]
    fn crspace_tet1_dof_count() {
        let m = SimplexMesh::<3>::unit_cube_tet(1);
        let s = CRSpace::new(m.clone(), 1);
        assert!(s.n_dofs() > 0);
        // A unit cube with 6 tets has 4 interior faces + boundary faces
        // Total unique faces > 4
        assert!(s.n_dofs() > 4);
        assert_eq!(s.element_dofs(0).len(), 4, "TetCR1: 4 DOFs per element");
    }
}
