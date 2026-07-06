use std::collections::HashSet;

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;

use crate::dof_manager::{DofManager, EdgeKey, FaceKey, QuadFaceKey};
use crate::hcurl::HCurlSpace;
use crate::hdiv::HDivSpace;

/// Apply Dirichlet boundary conditions to the assembled system `(K, f)`.
///
/// For each DOF in `constrained_dofs`:
/// 1. Zero the row.
/// 2. Set the diagonal to 1.
/// 3. Set `rhs[dof] = value[i]`.
///
/// This is the **non-symmetric** row-zeroing approach — fast and sufficient
/// for most FEM solves.
///
/// # Panics
/// Panics if `constrained_dofs.len() != values.len()`.
pub fn apply_dirichlet(
    mat:              &mut CsrMatrix<f64>,
    rhs:              &mut [f64],
    constrained_dofs: &[DofId],
    values:           &[f64],
) {
    assert_eq!(constrained_dofs.len(), values.len(),
        "constrained_dofs and values must have the same length");
    for (&dof, &val) in constrained_dofs.iter().zip(values.iter()) {
        mat.apply_dirichlet_row_zeroing(dof as usize, val, rhs);
    }
}

/// Identify which DOFs lie on boundary faces with the given tag(s).
///
/// Returns sorted global DOF indices for all boundary nodes (and, for any
/// order, edge and face DOFs) that lie on boundary faces whose tag is in `tags`.
///
/// Uses `edge_pk_map` and `face_pk_map` from DofManager, which support
/// arbitrary polynomial orders (no per-order hardcoded branches).
///
/// # Arguments
/// * `mesh`  — mesh providing boundary face data
/// * `dm`    — DOF manager for the space
/// * `tags`  — boundary tags to select (e.g. `&[1, 2, 3, 4]` for all sides)
pub fn boundary_dofs(
    mesh: &dyn fem_mesh::topology::MeshTopology,
    dm:   &DofManager,
    tags: &[i32],
) -> Vec<DofId> {
    let mut dof_set: HashSet<DofId> = HashSet::new();

    // Collect boundary edge keys from boundary faces.
    // In 2D: boundary face = edge (2 nodes) → 1 edge.
    // In 3D: boundary face = triangle (3+ nodes) → edges + potentially face-interior.
    let mut boundary_edges: HashSet<EdgeKey> = HashSet::new();
    let mut boundary_faces_3d: HashSet<FaceKey> = HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            // Vertex DOFs: all boundary face nodes
            for &node in nodes {
                dof_set.insert(node as DofId);
            }
            // Edge keys from face boundary
            for i in 0..nodes.len() {
                let a = nodes[i];
                let b = nodes[(i + 1) % nodes.len()];
                boundary_edges.insert(EdgeKey::new(a, b));
            }
            // 3D face key
            if nodes.len() >= 3 {
                boundary_faces_3d.insert(FaceKey::new(nodes[0], nodes[1], nodes[2]));
            }
        }
    }

    // Edge DOFs: look up each boundary edge in edge_pk_map (arbitrary order).
    for ek in &boundary_edges {
        if let Some(edge_dofs) = dm.edge_pk_map.get(ek) {
            for &dof in edge_dofs {
                dof_set.insert(dof);
            }
        }
        // Also check legacy edge_dof_map (P2) and edge_dof2_map (P3) for backward compat
        if let Some(&dof) = dm.edge_dof_map.get(ek) {
            dof_set.insert(dof);
        }
        if let Some(&[d0, d1]) = dm.edge_dof2_map.get(ek) {
            dof_set.insert(d0);
            dof_set.insert(d1);
        }
    }

    // Face-interior DOFs on boundary faces (3D, arbitrary order).
    if mesh.dim() == 3 {
        for fk in &boundary_faces_3d {
            if let Some(face_dofs) = dm.face_pk_map.get(fk) {
                for &dof in face_dofs {
                    dof_set.insert(dof);
                }
            }
        }
    }

    let mut out: Vec<DofId> = dof_set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Identify H(curl) DOFs on boundary faces with the given tag(s).
///
/// Collects all edges that lie on tagged boundary faces, then looks up
/// the corresponding global DOF in the space.
pub fn boundary_dofs_hcurl<M: fem_mesh::topology::MeshTopology>(
    mesh: &M,
    space: &HCurlSpace<M>,
    tags: &[i32],
) -> Vec<DofId> {
    // Collect boundary edges from tagged boundary faces.
    let mut boundary_edges: HashSet<EdgeKey> = HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            // Treat face nodes as a polygon ring and collect consecutive edges.
            // Works for 2D edge-faces (2 nodes), 3D triangles (3 nodes), and
            // 3D quadrilateral faces (4 nodes).
            if nodes.len() >= 2 {
                for i in 0..nodes.len() {
                    let a = nodes[i];
                    let b = nodes[(i + 1) % nodes.len()];
                    boundary_edges.insert(EdgeKey::new(a, b));
                }
            }
        }
    }

    let mut out: Vec<DofId> = Vec::new();
    for ek in &boundary_edges {
        if let Some(mut edofs) = space.edge_dofs(*ek) {
            out.append(&mut edofs);
        }
    }

    // Collect hex face DOFs on tagged boundary faces (3D quad faces).
    if mesh.dim() == 3 && space.order() >= 2 {
        for f in 0..mesh.n_boundary_faces() as u32 {
            if tags.contains(&mesh.face_tag(f)) {
                let nodes = mesh.face_nodes(f);
                if nodes.len() == 4 {
                    let key = QuadFaceKey::new(nodes[0], nodes[1], nodes[2], nodes[3]);
                    if let Some(mut fdofs) = space.quad_face_dofs(key) {
                        out.append(&mut fdofs);
                    }
                }
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}

/// Identify H(div) DOFs on boundary faces with the given tag(s).
///
/// In 2-D, boundary faces are edges; in 3-D, they are triangular faces.
pub fn boundary_dofs_hdiv<M: fem_mesh::topology::MeshTopology>(
    mesh: &M,
    space: &HDivSpace<M>,
    tags: &[i32],
) -> Vec<DofId> {
    let dim = mesh.dim() as usize;
    let mut out: Vec<DofId> = Vec::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            let dof = if dim == 2 {
                if nodes.len() >= 2 {
                    space.edge_face_dof(EdgeKey::new(nodes[0], nodes[1]))
                } else {
                    None
                }
            } else {
                if nodes.len() >= 3 {
                    space.tri_face_dof(FaceKey::new(nodes[0], nodes[1], nodes[2]))
                } else {
                    None
                }
            };
            if let Some(d) = dof {
                out.push(d);
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}
