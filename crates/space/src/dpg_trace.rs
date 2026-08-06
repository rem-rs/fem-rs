//! DPG trace (skeleton) finite element space on all mesh faces.
//!
//! `DpgTraceSpace` assigns DOFs to every mesh face (interior + boundary).
//! In 2D, each edge gets `order + 1` DOFs, matching the `RT_Trace_FECollection`
//! of MFEM ex8.  Provides face-level DOF access and element-face adjacency for
//! DPG mixed assembly (TraceJumpIntegrator).
//!
//! # MFEM equivalence
//! This type corresponds to MFEM's `RT_Trace_FECollection` used in ex8.cpp
//! for the interfacial (trace) unknown `xhat`.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

/// Information about a single skeleton face.
#[derive(Debug, Clone)]
pub enum FaceInfo {
    /// Boundary face: adjacent to exactly one element.
    Boundary {
        /// The adjacent element.
        elem: u32,
        /// Local face index within that element (0, 1, …).
        local_face: usize,
        /// Face node indices in original (non-sorted) order.
        nodes: Vec<u32>,
    },
    /// Interior face: shared by two elements.
    Interior {
        /// Left element (picked arbitrarily for sign convention).
        elem_l: u32,
        /// Right element.
        elem_r: u32,
        /// Local face index within the left element.
        local_l: usize,
        /// Local face index within the right element.
        local_r: usize,
        /// Face node indices in original (non-sorted) order.
        nodes: Vec<u32>,
    },
}

/// DPG trace space on the full mesh skeleton (interior + boundary faces).
///
/// DOFs per face = `order + 1` in 2D (one DOF per Lagrange node on the edge),
/// matching the H(div) trace space `RT_Trace_FECollection` in MFEM.
///
/// # Layout
/// - Faces are ordered: **boundary faces first**, then interior faces.
/// - Each face carries `dofs_per_face` consecutive DOF indices.
/// - `element_dofs(elem)` concatenates the face DOFs for all faces of that
///   element, in the local face order of the element.
pub struct DpgTraceSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    /// DOFs per face: `order + 1`.
    dofs_per_face: usize,
    /// Total number of trace DOFs.
    n_dofs: usize,
    /// Flat: `face_dofs[face * dpf .. (face + 1) * dpf]`.
    /// Boundary faces first (0..n_boundary), then interior faces.
    face_dofs: Vec<DofId>,
    /// Number of boundary faces.
    n_boundary_faces: usize,
    /// Total number of faces (boundary + interior).
    n_faces: usize,
    /// Per-face element adjacency and geometry.
    face_info: Vec<FaceInfo>,
    /// Per-element concatenated face DOFs: `elem_face_dofs[elem as usize]`.
    elem_face_dofs: Vec<Vec<DofId>>,
}

impl<M: MeshTopology> DpgTraceSpace<M> {
    /// Build a DPG trace space of given `order` over `mesh`.
    ///
    /// `order` is the polynomial order on each face (trace unknown).
    /// For 2D, each edge gets `order + 1` DOFs.  `order` must fit in `u8`.
    ///
    /// # Panics
    /// Panics if the mesh has mixed element types with incompatible face
    /// topology.
    pub fn new(mesh: M, order: u8) -> Self {
        let dpf = (order as usize + 1).max(1);
        let dim = mesh.dim() as usize;

        // ── Step 1: enumerate ALL faces in element-traversal order ─────────
        // MFEM's RT_Trace_FECollection trace DOFs follow the mesh *edge*
        // table numbering (GetVertexToVertexTable: first-encounter order
        // while walking elements 0..NE), NOT "boundary faces first".
        // The old boundary-first ordering misnumbered the trace DOFs — the
        // 20 boundary edges took DOFs 0..19 so every interior edge shifted,
        // scrambling Bhat columns versus MFEM (ex8 comparison).
        let mut face_map: HashMap<Vec<u32>, usize> = HashMap::new(); // sorted key → face id
        let mut face_info: Vec<FaceInfo> = Vec::new();
        let mut first_seen: Vec<(u32, usize, Vec<u32>)> = Vec::new(); // (elem, local_face, nodes)
        let mut n_boundary = 0usize;

        for e in mesh.elem_iter() {
            let en = mesh.element_nodes(e);
            let local_faces = local_faces_of_element(en, dim);
            for (li, lf_nodes) in local_faces.iter().enumerate() {
                let unsorted: Vec<u32> = lf_nodes.iter().map(|&k| en[k]).collect();
                let mut key: Vec<u32> = unsorted.clone();
                key.sort_unstable();
                match face_map.get(&key) {
                    Some(&fid) => {
                        // Second encounter → this is an interior face.  It was
                        // tentatively recorded as boundary on first sight.
                        if let FaceInfo::Boundary { .. } = face_info[fid] {
                            let (e1, l1, ref n1) = first_seen[fid];
                            // Keep the FIRST-seen node order: MFEM's face
                            // direction is the first-encounter edge direction
                            // (GetVertexToVertexTable walk), and the RT0
                            // trace DOF sign is derived from that direction.
                            face_info[fid] = FaceInfo::Interior {
                                elem_l: e1, elem_r: e,
                                local_l: l1, local_r: li,
                                nodes: n1.clone(),
                            };
                            n_boundary -= 1;
                        }
                        // Third+ encounter (non-manifold): keep first Interior.
                    }
                    None => {
                        let fid = face_info.len();
                        face_map.insert(key, fid);
                        first_seen.push((e, li, unsorted.clone()));
                        face_info.push(FaceInfo::Boundary {
                            elem: e, local_face: li, nodes: unsorted,
                        });
                        n_boundary += 1;
                    }
                }
            }
        }

        let n_faces = face_info.len();

        // ── Step 2: assign DOFs (consecutive per face, traversal order) ────
        let mut face_dofs = vec![u32::MAX; n_faces * dpf];
        let mut next_dof = 0u32;
        for f in 0..n_faces {
            let base = f * dpf;
            for k in 0..dpf {
                face_dofs[base + k] = next_dof;
                next_dof += 1;
            }
        }

        // ── Step 3: build per-element face-DOF concatenation ─────────────────
        let n_elems = mesh.n_elements() as usize;
        let mut elem_face_dofs = vec![Vec::new(); n_elems];
        for (fi, info) in face_info.iter().enumerate() {
            let base = fi * dpf;
            let dofs = &face_dofs[base..base + dpf];
            match info {
                FaceInfo::Boundary { elem, .. } => {
                    elem_face_dofs[*elem as usize].extend_from_slice(dofs);
                }
                FaceInfo::Interior { elem_l, elem_r, .. } => {
                    elem_face_dofs[*elem_l as usize].extend_from_slice(dofs);
                    elem_face_dofs[*elem_r as usize].extend_from_slice(dofs);
                }
            }
        }

        DpgTraceSpace {
            mesh,
            order,
            dofs_per_face: dpf,
            n_dofs: next_dof as usize,
            face_dofs,
            n_boundary_faces: n_boundary,
            n_faces,
            face_info,
            elem_face_dofs,
        }
    }

    // ─── Public accessors ───────────────────────────────────────────────────────

    /// DOF indices for skeleton face `face_idx`.
    ///
    /// Boundary faces have indices `0..n_boundary_faces`,
    /// interior faces have indices `n_boundary_faces..n_faces`.
    pub fn face_dofs(&self, face_idx: usize) -> &[DofId] {
        let base = face_idx * self.dofs_per_face;
        &self.face_dofs[base..base + self.dofs_per_face]
    }

    /// Element adjacency and geometry for face `face_idx`.
    pub fn face_info(&self, face_idx: usize) -> &FaceInfo {
        &self.face_info[face_idx]
    }

    /// Number of boundary faces.
    pub fn n_boundary_faces(&self) -> usize {
        self.n_boundary_faces
    }

    /// Number of interior faces.
    pub fn n_interior_faces(&self) -> usize {
        self.n_faces - self.n_boundary_faces
    }

    /// Total number of faces (boundary + interior).
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// Number of DOFs per face.
    pub fn dofs_per_face(&self) -> usize {
        self.dofs_per_face
    }

    /// Compute the outward normal and face Jacobian (length/area) for face `face_idx`.
    ///
    /// For boundary faces the normal points outward.
    /// For interior faces the normal points from left element to right element.
    pub fn face_normal(&self, face_idx: usize) -> (Vec<f64>, f64) {
        let info = &self.face_info[face_idx];
        let nodes = match info {
            FaceInfo::Boundary { nodes, .. } | FaceInfo::Interior { nodes, .. } => nodes,
        };
        let dim = self.mesh.dim() as usize;

        if dim == 2 {
            let pa = self.mesh.node_coords(nodes[0]);
            let pb = self.mesh.node_coords(nodes[1]);
            let dx = pb[0] - pa[0];
            let dy = pb[1] - pa[1];
            let len = (dx * dx + dy * dy).sqrt();
            // For a boundary face, we need to determine whether (-dy, dx) or (dy, -dx)
            // is the outward normal.  For the left element (the one that contains this
            // face), the CCW normal is (-dy, dx).
            // For interior faces, the convention is left → right.
            (vec![-dy / len, dx / len], len)
        } else {
            let pa = self.mesh.node_coords(nodes[0]);
            let pb = self.mesh.node_coords(nodes[1]);
            let pc = self.mesh.node_coords(nodes[2]);
            let d1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let d2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let nx = d1[1] * d2[2] - d1[2] * d2[1];
            let ny = d1[2] * d2[0] - d1[0] * d2[2];
            let nz = d1[0] * d2[1] - d1[1] * d2[0];
            let area = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();
            let norm = 1.0 / (2.0 * area);
            (vec![nx * norm, ny * norm, nz * norm], area)
        }
    }
}

// ─── FESpace trait implementation ─────────────────────────────────────────────

impl<M: MeshTopology + Clone> FESpace for DpgTraceSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &Self::Mesh {
        &self.mesh
    }

    fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        &self.elem_face_dofs[elem as usize]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // Interpolation on trace spaces is not meaningful in the standard sense.
        Vector::from_vec(vec![0.0; self.n_dofs])
    }

    fn space_type(&self) -> SpaceType {
        SpaceType::HDiv // closest analogue: H(div) trace
    }

    fn order(&self) -> u8 {
        self.order
    }

    fn element_signs(&self, _elem: u32) -> Option<&[f64]> {
        None
    }
}

// ─── Helper: local face extraction ───────────────────────────────────────────

/// Extract the local faces of an element given its node list and spatial dim.
///
/// - 2D (triangle/quad): faces = edges (pairs of consecutive nodes, wrapping)
/// - 3D (tet/hex): faces = triples/quadruples
fn local_faces_of_element(elem_nodes: &[u32], dim: usize) -> Vec<Vec<usize>> {
    match (elem_nodes.len(), dim) {
        (3, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
        (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
        (4, 3) => vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ],
        (8, 3) => vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
        _ => panic!(
            "local_faces_of_element: unsupported (npe={}, dim={})",
            elem_nodes.len(),
            dim
        ),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn unit_square_tri_face_count() {
        // 2×2 tri mesh: 8 triangles → boundary edges = perimeter of 2×2 square = 8
        // interior edges = (3n² - 2n) for n=2 = 12 - 4 = 8
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 1);
        assert_eq!(trace.n_boundary_faces(), 8, "boundary faces");
        assert_eq!(trace.n_interior_faces(), 8, "interior faces");
        assert_eq!(trace.n_faces(), 16);
    }

    #[test]
    fn dof_count_matches_face_count() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 1);
        // order=1 → dpf=2
        assert_eq!(trace.dofs_per_face(), 2);
        assert_eq!(trace.n_dofs(), trace.n_faces() * 2);
    }

    #[test]
    fn dof_indices_are_unique() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let trace = DpgTraceSpace::new(mesh, 1);
        let mut seen = std::collections::HashSet::new();
        for f in 0..trace.n_faces() {
            for &d in trace.face_dofs(f) {
                assert!(seen.insert(d), "duplicate DOF {d} at face {f}");
            }
        }
        assert_eq!(seen.len(), trace.n_dofs());
    }

    #[test]
    fn element_dofs_length_is_consistent() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 1);
        // Tri3: 3 edges per element, 2 DOFs per edge = 6 DOFs per element
        for e in 0..trace.mesh.n_elements() as u32 {
            let dofs = trace.element_dofs(e);
            assert_eq!(dofs.len(), 3 * 2, "elem {e} should have 6 trace DOFs");
        }
    }

    #[test]
    fn trace_order_0_has_one_dof_per_face() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 0);
        assert_eq!(trace.dofs_per_face(), 1);
        assert_eq!(trace.n_dofs(), trace.n_faces());
    }

    #[test]
    fn trace_order_2_has_three_dofs_per_face() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 2);
        assert_eq!(trace.dofs_per_face(), 3);
        assert_eq!(trace.n_dofs(), trace.n_faces() * 3);
    }

    #[test]
    fn face_normal_2d_has_positive_length() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh, 1);
        for f in 0..trace.n_boundary_faces() {
            let (n, len) = trace.face_normal(f);
            assert!(len > 0.0, "boundary face {f} length should be positive, got {len}");
            let n_norm = (n[0] * n[0] + n[1] * n[1]).sqrt();
            assert!((n_norm - 1.0).abs() < 1e-12, "boundary face {f} normal not unit: {n_norm}");
        }
    }

    #[test]
    fn single_square_face_count() {
        // 1×1 square divided into 2 triangles:
        // boundary edges = 4 (the perimeter), interior edges = 1 (the diagonal)
        let mesh = Mesh::<2>::unit_square_tri(1);
        let trace = DpgTraceSpace::new(mesh, 1);
        assert_eq!(trace.n_boundary_faces(), 4);
        assert_eq!(trace.n_interior_faces(), 1);
        assert_eq!(trace.n_faces(), 5);
    }

    #[test]
    fn quad_mesh_face_count() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let trace = DpgTraceSpace::new(mesh, 1);
        // 2×2 quads = 4 elements, each has 4 edges
        // interior edges = shared edges between quads
        assert!(trace.n_interior_faces() > 0);
        assert_eq!(
            trace.n_dofs(),
            trace.n_faces() * trace.dofs_per_face()
        );
    }

    #[test]
    fn elem_face_dofs_include_all_faces() {
        // Each element's concatenated face DOFs should cover all faces of that element.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let trace = DpgTraceSpace::new(mesh.clone(), 1);
        for e in 0..mesh.n_elements() as u32 {
            let en = mesh.element_nodes(e);
            let n_faces_e = en.len(); // 3 for tri, 4 for quad
            let dofs = trace.element_dofs(e);
            assert_eq!(
                dofs.len(),
                n_faces_e * trace.dofs_per_face(),
                "elem {e}: expected {} trace DOFs, got {}",
                n_faces_e * trace.dofs_per_face(),
                dofs.len()
            );
        }
    }
}
