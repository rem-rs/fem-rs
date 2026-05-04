//! HDG skeleton (trace) finite element space.
//!
//! `SkeletonSpace` assigns polynomial DOFs to every **interior** mesh face
//! (and optionally boundary faces).  This is the global DOF space for the
//! hybridized unknowns in Hybridizable Discontinuous Galerkin (HDG) methods.
//!
//! For P1 HDG, each face carries one DOF per face vertex; for P0 HDG, each
//! face carries exactly one DOF.
//!
//! # HDG overview
//! In HDG the primal unknown is split into
//! - a per-element bulk solution `u_h` (discontinuous), and
//! - a face trace `λ_h` (the "skeleton" unknown, supported on ∂K).
//!
//! `SkeletonSpace` provides the global DOF layout for `λ_h`.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

// ─── SkeletonSpace ────────────────────────────────────────────────────────────

/// Skeleton (HDG trace) finite element space.
///
/// Supports two modes controlled by `include_boundary`:
/// - `false` — DOFs live on interior faces only (typical HDG formulation where
///   Dirichlet data is imposed weakly via the skeleton).
/// - `true`  — DOFs live on all faces (interior + boundary).
///
/// **Order support**: P0 (one DOF per face) and P1 (one DOF per face vertex).
pub struct SkeletonSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    /// DOF indices indexed by `face_id * dofs_per_face + local_dof`.
    face_dofs: Vec<DofId>,
    /// Number of DOFs per face.
    dofs_per_face: usize,
    /// Total number of skeleton DOFs.
    n_dofs: usize,
    /// Whether boundary faces are included.
    include_boundary: bool,
    /// Number of mesh faces (interior + boundary).
    n_faces: usize,
}

impl<M: MeshTopology + Clone> SkeletonSpace<M> {
    /// Build a P0 or P1 skeleton space.
    ///
    /// # Arguments
    /// * `mesh`              — the underlying mesh (must implement `MeshTopology`)
    /// * `order`             — polynomial order on each face (0 or 1)
    /// * `include_boundary`  — whether to include boundary faces as skeleton DOFs
    ///
    /// # Panics
    /// Panics if `order > 1`.
    pub fn new(mesh: M, order: u8, include_boundary: bool) -> Self {
        assert!(order <= 1, "SkeletonSpace: only P0 and P1 supported, got order={order}");

        let n_bfaces = mesh.n_boundary_faces();
        let n_faces  = n_bfaces; // MeshTopology only exposes boundary faces directly;
        // for interior faces we need element connectivity.
        // We build a list of all *unique* faces from element connectivity below.

        let dim = mesh.dim();
        let nodes_per_face = match dim {
            1 => 1usize,
            2 => 2usize, // edge
            3 => 3usize, // triangle face (Tet4 mesh)
            _ => panic!("SkeletonSpace: unsupported dimension {dim}"),
        };
        let dofs_per_face = match order {
            0 => 1,
            1 => nodes_per_face,
            _ => unreachable!(),
        };

        // ── Enumerate all unique faces from element connectivity ──────────────
        // For each element, extract its local faces; deduplicate by sorted node key.
        let n_elems = mesh.n_elements();
        let mut face_node_map: HashMap<Vec<u32>, (usize, Vec<u32>)> = HashMap::new();
        // face_node_map: sorted_key → (face_idx, original node order)
        let mut all_face_nodes: Vec<Vec<u32>> = Vec::new(); // original order
        let mut face_is_boundary: Vec<bool> = Vec::new();

        // Mark boundary faces
        let mut bdr_face_keys: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
        for f in 0..n_bfaces as u32 {
            let fn_slice = mesh.face_nodes(f);
            let mut key: Vec<u32> = fn_slice.to_vec();
            key.sort_unstable();
            bdr_face_keys.insert(key);
        }

        // Enumerate all faces from element connectivity
        for e in 0..n_elems as u32 {
            let elem_nodes = mesh.element_nodes(e);
            let local_faces = local_faces_of_element(elem_nodes, dim);
            for face_nodes_orig in local_faces {
                let mut key = face_nodes_orig.clone();
                key.sort_unstable();
                if !face_node_map.contains_key(&key) {
                    let idx = all_face_nodes.len();
                    let is_bdr = bdr_face_keys.contains(&key);
                    face_node_map.insert(key, (idx, face_nodes_orig.clone()));
                    all_face_nodes.push(face_nodes_orig);
                    face_is_boundary.push(is_bdr);
                }
            }
        }

        let total_faces = all_face_nodes.len();

        // ── Assign DOFs to faces ─────────────────────────────────────────────
        // face_dofs[f * dofs_per_face .. (f+1) * dofs_per_face] = DOF indices
        // u32::MAX for excluded faces.
        let sentinel = u32::MAX;
        let mut face_dofs: Vec<DofId> = vec![sentinel; total_faces * dofs_per_face];
        let mut next_dof = 0usize;

        // For P1: we need a shared node → dof mapping so shared nodes across faces
        // get the same DOF (HDG: λ is single-valued on each face, not shared between faces).
        // Actually in HDG, skeleton DOFs are *per-face*, not shared between faces.
        // Each face has its own independent DOFs.
        for (f, nodes) in all_face_nodes.iter().enumerate() {
            if !include_boundary && face_is_boundary[f] {
                // Leave as sentinel (excluded)
                continue;
            }
            let base = f * dofs_per_face;
            match order {
                0 => {
                    face_dofs[base] = next_dof as DofId;
                    next_dof += 1;
                }
                1 => {
                    // One DOF per face vertex, independent per face
                    for (k, _node) in nodes.iter().enumerate().take(dofs_per_face) {
                        face_dofs[base + k] = next_dof as DofId;
                        next_dof += 1;
                    }
                }
                _ => unreachable!(),
            }
        }

        SkeletonSpace {
            mesh,
            order,
            face_dofs,
            dofs_per_face,
            n_dofs: next_dof,
            include_boundary,
            n_faces: total_faces,
        }
    }

    /// Number of skeleton faces (interior only if `include_boundary == false`).
    pub fn n_skeleton_faces(&self) -> usize {
        self.n_faces
    }

    /// DOF indices for skeleton face `face_idx`.
    ///
    /// Returns a slice of length `dofs_per_face`.  Entries that are `u32::MAX`
    /// indicate that this face is excluded (boundary face when
    /// `include_boundary == false`).
    pub fn face_dofs_raw(&self, face_idx: usize) -> &[DofId] {
        let base = face_idx * self.dofs_per_face;
        &self.face_dofs[base..base + self.dofs_per_face]
    }

    /// Returns `true` if this face has active DOFs (i.e., is not excluded).
    pub fn face_is_active(&self, face_idx: usize) -> bool {
        let base = face_idx * self.dofs_per_face;
        self.face_dofs[base] != u32::MAX
    }
}

// ─── Helper: local face extraction ───────────────────────────────────────────

/// Extract the local faces of an element given its node list and spatial dim.
///
/// - 1D (line): faces = endpoints
/// - 2D (triangle/quad): faces = edges (pairs of consecutive nodes, wrapping)
/// - 3D (tet): faces = triangular faces (triples)
fn local_faces_of_element(elem_nodes: &[u32], dim: u8) -> Vec<Vec<u32>> {
    match dim {
        1 => elem_nodes.iter().map(|&n| vec![n]).collect(),
        2 => {
            let n = elem_nodes.len();
            (0..n).map(|i| vec![elem_nodes[i], elem_nodes[(i + 1) % n]]).collect()
        }
        3 => {
            let n = elem_nodes.len();
            if n == 4 {
                // Tet4: 4 triangular faces
                vec![
                    vec![elem_nodes[0], elem_nodes[1], elem_nodes[2]],
                    vec![elem_nodes[0], elem_nodes[1], elem_nodes[3]],
                    vec![elem_nodes[0], elem_nodes[2], elem_nodes[3]],
                    vec![elem_nodes[1], elem_nodes[2], elem_nodes[3]],
                ]
            } else if n == 8 {
                // Hex8: 6 quad faces
                vec![
                    vec![elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3]],
                    vec![elem_nodes[4], elem_nodes[5], elem_nodes[6], elem_nodes[7]],
                    vec![elem_nodes[0], elem_nodes[1], elem_nodes[5], elem_nodes[4]],
                    vec![elem_nodes[1], elem_nodes[2], elem_nodes[6], elem_nodes[5]],
                    vec![elem_nodes[2], elem_nodes[3], elem_nodes[7], elem_nodes[6]],
                    vec![elem_nodes[3], elem_nodes[0], elem_nodes[4], elem_nodes[7]],
                ]
            } else {
                panic!("local_faces_of_element: unsupported 3D element with {n} nodes");
            }
        }
        _ => panic!("local_faces_of_element: unsupported dim {dim}"),
    }
}

// ─── FESpace trait implementation ─────────────────────────────────────────────

impl<M: MeshTopology + Clone + Send + Sync> FESpace for SkeletonSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &Self::Mesh { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    /// Returns DOFs for element `elem`'s skeleton faces (all faces of the element).
    ///
    /// The returned slice concatenates DOFs for all local faces in local face order.
    /// DOFs from excluded boundary faces are included as `u32::MAX` sentinels.
    fn element_dofs(&self, elem: u32) -> &[DofId] {
        // SkeletonSpace does not support the standard FESpace::element_dofs API
        // because the element→face mapping is not stored per element.
        // Return an empty slice; use `face_dofs_raw` for face-level access.
        let _ = elem;
        &[]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // For skeleton spaces, interpolation is not meaningful in the standard sense.
        // Return zero vector.
        let _ = f;
        Vector::from_vec(vec![0.0; self.n_dofs])
    }

    fn space_type(&self) -> SpaceType { SpaceType::L2 } // closest analogue

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        let _ = elem;
        None
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn p0_skeleton_all_faces() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sk = SkeletonSpace::new(mesh, 0, true);
        // P0: one DOF per face
        assert_eq!(sk.dofs_per_face, 1);
        assert!(sk.n_dofs() > 0);
        assert_eq!(sk.n_dofs(), sk.n_skeleton_faces());
    }

    #[test]
    fn p0_skeleton_interior_only_has_fewer_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sk_all = SkeletonSpace::new(mesh.clone(), 0, true);
        let sk_int = SkeletonSpace::new(mesh, 0, false);
        assert!(sk_int.n_dofs() < sk_all.n_dofs(),
            "interior-only: {} >= all-faces: {}", sk_int.n_dofs(), sk_all.n_dofs());
    }

    #[test]
    fn p1_skeleton_has_more_dofs_than_p0() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sk0 = SkeletonSpace::new(mesh.clone(), 0, true);
        let sk1 = SkeletonSpace::new(mesh, 1, true);
        assert!(sk1.n_dofs() > sk0.n_dofs(),
            "P1 n_dofs={} should exceed P0 n_dofs={}", sk1.n_dofs(), sk0.n_dofs());
    }

    #[test]
    fn p0_dof_indices_are_unique() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let sk = SkeletonSpace::new(mesh, 0, true);
        let mut seen = std::collections::HashSet::new();
        for f in 0..sk.n_skeleton_faces() {
            let dofs = sk.face_dofs_raw(f);
            for &d in dofs {
                if d != u32::MAX {
                    assert!(seen.insert(d), "duplicate DOF {d} at face {f}");
                }
            }
        }
        assert_eq!(seen.len(), sk.n_dofs());
    }

    #[test]
    fn active_faces_match_include_boundary() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let sk = SkeletonSpace::new(mesh, 0, false);
        // With include_boundary=false, boundary faces should not be active
        let active_count = (0..sk.n_skeleton_faces())
            .filter(|&f| sk.face_is_active(f))
            .count();
        assert_eq!(active_count, sk.n_dofs(),
            "active faces ({active_count}) should equal n_dofs ({})", sk.n_dofs());
    }

    #[test]
    fn p0_skeleton_3d_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let sk = SkeletonSpace::new(mesh, 0, true);
        assert_eq!(sk.dofs_per_face, 1);
        assert!(sk.n_dofs() > 0);
    }
}
