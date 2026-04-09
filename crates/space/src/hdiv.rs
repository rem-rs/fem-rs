//! H(div) finite element space for Raviart-Thomas face elements.
//!
//! ## DOF association
//!
//! Each DOF corresponds to a unique mesh face (edge in 2-D, triangular face
//! in 3-D).  The DOF functional is the normal flux integral:
//! `DOF_f(u) = ∫_f u · n̂ ds`.
//!
//! For lowest-order Raviart-Thomas (RT0):
//! - **2-D triangles**: 3 face (= edge) DOFs per element, `n_dofs = n_unique_edges`
//! - **3-D tetrahedra**: 4 face DOFs per element, `n_dofs = n_unique_faces`
//!
//! ## Sign convention
//!
//! Each face is given a *global* orientation.  In 2-D this is the canonical
//! edge direction (from smaller to larger vertex index).  In 3-D it is defined
//! by the sorted vertex triple.  The sign on an element is +1 when the local
//! outward normal agrees with the global normal, and −1 otherwise.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::{EdgeKey, FaceKey};
use crate::fe_space::{FESpace, SpaceType};

// ─── Local face tables ──────────────────────────────────────────────────────

/// Local face definitions for 2-D triangles (TriRT0 ordering).
/// Face `i` is the edge opposite vertex `i`.
const TRI_FACES: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

/// Local face definitions for 3-D tetrahedra (TetRT0 ordering).
/// Face `i` is the triangle opposite vertex `i`.
const TET_FACES: [(usize, usize, usize); 4] = [
    (1, 2, 3), // opposite v₀
    (0, 2, 3), // opposite v₁
    (0, 1, 3), // opposite v₂
    (0, 1, 2), // opposite v₃
];

// ─── Face DOF map ───────────────────────────────────────────────────────────

/// Unified face-to-DOF lookup: edges in 2-D, triangular faces in 3-D.
enum FaceDofMap {
    Edges(HashMap<EdgeKey, DofId>),
    Faces(HashMap<FaceKey, DofId>),
}

// ─── HDivSpace ──────────────────────────────────────────────────────────────

/// H(div) finite element space using Raviart-Thomas face elements.
///
/// Constructed from a [`MeshTopology`] with triangular or tetrahedral elements.
/// Currently supports order 0 (RT0).
pub struct HDivSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    dofs_flat: Vec<DofId>,
    signs_flat: Vec<f64>,
    dofs_per_elem: usize,
    face_map: FaceDofMap,
}

impl<M: MeshTopology> HDivSpace<M> {
    /// Construct an H(div) space of the given order on `mesh`.
    ///
    /// # Panics
    /// - If `order > 1` (only RT0 and RT1 are currently supported).
    /// - If the mesh is neither 2-D triangles nor 3-D tetrahedra.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order <= 1, "HDivSpace: only orders 0 (RT0) and 1 (RT1) are supported");
        let dim = mesh.dim() as usize;
        match dim {
            2 => Self::build_2d(mesh, order),
            3 => Self::build_3d(mesh, order),
            _ => panic!("HDivSpace: unsupported dimension {dim}"),
        }
    }

    // ─── 2-D construction ───────────────────────────────────────────────────

    fn build_2d(mesh: M, order: u8) -> Self {
        // RT0: 1 DOF per edge; RT1: 2 DOFs per edge + 2 interior bubble DOFs
        let dofs_per_face = (order as usize) + 1;
        let interior_dofs = if order == 0 { 0 } else { 2 };
        let dofs_per_elem = TRI_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut edge_map: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for (face_idx, &(li, lj)) in TRI_FACES.iter().enumerate() {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = Self::compute_sign_2d(&mesh, verts, face_idx, gi, gj);

                if dofs_per_face == 1 {
                    let dof = *edge_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    // RT1: 2 DOFs per edge (first and second normal moments)
                    let first = *edge_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=2; d });
                    dofs_flat.push(first);
                    dofs_flat.push(first + 1);
                    signs_flat.push(sign);
                    signs_flat.push(sign);
                }
            }
            // Interior bubble DOFs
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::Edges(edge_map),
        }
    }

    /// Compute the orientation sign for a 2-D face (edge).
    ///
    /// Global edge normal is the 90° CCW rotation of (p_max − p_min).
    /// Local outward normal points away from the opposite vertex.
    /// Sign = +1 if they agree, −1 otherwise.
    fn compute_sign_2d(mesh: &M, verts: &[u32], face_idx: usize, gi: u32, gj: u32) -> f64 {
        let pa = mesh.node_coords(gi);
        let pb = mesh.node_coords(gj);
        // Edge tangent gi→gj
        let tx = pb[0] - pa[0];
        let ty = pb[1] - pa[1];
        // Normal of edge gi→gj (90° CCW rotation): (−ty, tx)
        let nx = -ty;
        let ny = tx;

        // Opposite vertex
        let opp_local = face_idx; // face i is opposite vertex i
        let opp_global = verts[opp_local];
        let po = mesh.node_coords(opp_global);

        // The outward normal should point AWAY from the opposite vertex.
        // Test: (midpoint_of_edge → opposite_vertex) · normal < 0 means
        // the normal already points away from the opposite vertex.
        let mx = 0.5 * (pa[0] + pb[0]);
        let my = 0.5 * (pa[1] + pb[1]);
        let to_opp_x = po[0] - mx;
        let to_opp_y = po[1] - my;
        let dot = nx * to_opp_x + ny * to_opp_y;

        // Global orientation: the canonical edge goes min→max.
        // If gi < gj, the edge tangent is in global direction, and the normal
        // (nx, ny) is the global normal.  If gi > gj, we need to flip.
        let global_flip = if gi < gj { 1.0 } else { -1.0 };

        // dot < 0 → normal already points away from opp → outward direction agrees
        // with the tangent-based normal direction.
        let outward_flip = if dot < 0.0 { 1.0 } else { -1.0 };

        global_flip * outward_flip
    }

    // ─── 3-D construction ───────────────────────────────────────────────────

    fn build_3d(mesh: M, order: u8) -> Self {
        // RT0: 1 DOF per face; RT1: 3 DOFs per face + 3 interior bubble DOFs
        let dofs_per_face = if order == 0 { 1 } else { 3 };
        let interior_dofs = if order == 0 { 0 } else { 3 };
        let dofs_per_elem = TET_FACES.len() * dofs_per_face + interior_dofs;
        let n_elem = mesh.n_elements();

        let mut face_map: HashMap<FaceKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for (face_idx, &(la, lb, lc)) in TET_FACES.iter().enumerate() {
                let (ga, gb, gc) = (verts[la], verts[lb], verts[lc]);
                let key = FaceKey::new(ga, gb, gc);
                let sign = Self::compute_sign_3d(&mesh, verts, face_idx, &key);

                if dofs_per_face == 1 {
                    let dof = *face_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    // RT1: 3 DOFs per face
                    let first = *face_map.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=3; d });
                    dofs_flat.push(first); dofs_flat.push(first+1); dofs_flat.push(first+2);
                    signs_flat.push(sign); signs_flat.push(sign); signs_flat.push(sign);
                }
            }
            for _ in 0..interior_dofs {
                dofs_flat.push(next_dof); next_dof+=1; signs_flat.push(1.0);
            }
        }

        HDivSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            face_map: FaceDofMap::Faces(face_map),
        }
    }

    /// Compute the orientation sign for a 3-D face (triangle).
    ///
    /// The global face normal is defined by the cross product of edges
    /// of the sorted vertex triple.  The local outward normal points
    /// away from the opposite vertex.  Sign = +1 if they agree.
    fn compute_sign_3d(mesh: &M, verts: &[u32], face_idx: usize, key: &FaceKey) -> f64 {
        let p0 = mesh.node_coords(key.0);
        let p1 = mesh.node_coords(key.1);
        let p2 = mesh.node_coords(key.2);

        // Global face normal: (p1−p0) × (p2−p0)
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n_global = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // The outward direction is away from the opposite vertex.
        let opp_local = face_idx;
        let opp_global = verts[opp_local];
        let po = mesh.node_coords(opp_global);

        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        // outward = centroid − opposite_vertex
        let outward = [
            centroid[0] - po[0],
            centroid[1] - po[1],
            centroid[2] - po[2],
        ];

        let dot = n_global[0] * outward[0]
            + n_global[1] * outward[1]
            + n_global[2] * outward[2];

        if dot > 0.0 { 1.0 } else { -1.0 }
    }

    // ─── Public API ─────────────────────────────────────────────────────────

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = elem as usize * self.dofs_per_elem;
        &self.signs_flat[start..start + self.dofs_per_elem]
    }

    /// Look up the global DOF for a 2-D face (edge).
    pub fn edge_face_dof(&self, edge: EdgeKey) -> Option<DofId> {
        match &self.face_map {
            FaceDofMap::Edges(map) => map.get(&edge).copied(),
            FaceDofMap::Faces(_) => None,
        }
    }

    /// Look up the global DOF for a 3-D face (triangle).
    pub fn tri_face_dof(&self, face: FaceKey) -> Option<DofId> {
        match &self.face_map {
            FaceDofMap::Faces(map) => map.get(&face).copied(),
            FaceDofMap::Edges(_) => None,
        }
    }

    /// Vector-valued interpolation via the RT DOF functional.
    ///
    /// For RT0, `DOF_f(F) = F(centroid_f) · n_f · |f|` where `n_f` is the
    /// outward unit normal and `|f|` is the face measure.  For the global DOF
    /// value we use the global orientation.
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);
        match &self.face_map {
            FaceDofMap::Edges(map) => {
                // 2-D: faces are edges
                for (&EdgeKey(a, b), &dof) in map {
                    let pa = self.mesh.node_coords(a);
                    let pb = self.mesh.node_coords(b);
                    let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                    // Global edge tangent a→b (a < b), normal = 90° CCW rotation
                    let tx = pb[0] - pa[0];
                    let ty = pb[1] - pa[1];
                    let normal = [-ty, tx]; // length = edge length
                    let fval = f(&mid);
                    let dot = fval[0] * normal[0] + fval[1] * normal[1];
                    result.as_slice_mut()[dof as usize] = dot;
                }
            }
            FaceDofMap::Faces(map) => {
                // 3-D: faces are triangles
                for (&FaceKey(a, b, c), &dof) in map {
                    let pa = self.mesh.node_coords(a);
                    let pb = self.mesh.node_coords(b);
                    let pc = self.mesh.node_coords(c);
                    let centroid = [
                        (pa[0] + pb[0] + pc[0]) / 3.0,
                        (pa[1] + pb[1] + pc[1]) / 3.0,
                        (pa[2] + pb[2] + pc[2]) / 3.0,
                    ];
                    // Global face normal = (pb−pa) × (pc−pa)  (length = 2 × area)
                    let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let normal = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let fval = f(&centroid);
                    let dot = fval[0] * normal[0] + fval[1] * normal[1] + fval[2] * normal[2];
                    result.as_slice_mut()[dof as usize] = dot;
                }
            }
        }
        result
    }
}

impl<M: MeshTopology> FESpace for HDivSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.dofs_flat[start..start + self.dofs_per_elem]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // Scalar interpolation is meaningless for H(div).
        // Use `interpolate_vector` instead.
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HDiv }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hdiv_dof_count_tri_2d() {
        // 4×4 unit-square mesh: 32 triangles, 56 unique edges.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.dofs_per_elem, 3);
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges in 2-D");
    }

    #[test]
    fn hdiv_shared_face_dof_2d() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one face DOF");
    }

    #[test]
    fn hdiv_signs_opposite_on_shared_face_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh, 0);

        let dofs0 = space.element_dofs(0);
        let signs0 = space.element_signs(0);
        let dofs1 = space.element_dofs(1);
        let signs1 = space.element_signs(1);

        for (i, &d0) in dofs0.iter().enumerate() {
            for (j, &d1) in dofs1.iter().enumerate() {
                if d0 == d1 {
                    assert!(
                        (signs0[i] + signs1[j]).abs() < 1e-14,
                        "shared face DOF {d0}: signs {}, {} should be opposite",
                        signs0[i], signs1[j]
                    );
                }
            }
        }
    }

    #[test]
    fn hdiv_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.space_type(), SpaceType::HDiv);
    }

    #[test]
    fn hdiv_dof_count_tet_3d() {
        // Unit-cube tet mesh.
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let space = HDivSpace::new(mesh, 0);
        assert_eq!(space.dofs_per_elem, 4);
        // Each tet has 4 faces; total unique faces > n_elements (interior faces shared).
        assert!(space.n_dofs() > 0);
        // For a 2×2×2 cube mesh: 48 tets, each with 4 faces, many shared.
        // The exact count depends on the mesh generator, but verify consistency:
        // total face references = n_elem × 4, all dof indices valid.
        for e in 0..space.mesh().n_elements() as u32 {
            for &d in space.element_dofs(e) {
                assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
            }
        }
    }

    #[test]
    fn hdiv_interpolate_vector_constant_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 0);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }
}
