//! Essential (Dirichlet) boundary condition enforcement and hanging-node
//! constraint application.
//!
//! After assembly, call [`apply_dirichlet`] to modify the stiffness matrix and
//! right-hand side so that constrained DOFs are set to their prescribed values.
//!
//! For non-conforming meshes, call [`apply_hanging_constraints`] to enforce
//! `u_hang = 0.5*(u_a + u_b)` and then [`recover_hanging_values`] after solving.

use fem_core::types::DofId;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::amr::{HangingNodeConstraint, HangingFaceConstraint};
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::{DofManager, EdgeKey, FaceKey};
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
/// Returns sorted global DOF indices for all boundary nodes (and, for P2,
/// edge-midpoint DOFs) that lie on boundary faces whose tag is in `tags`.
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
    use std::collections::HashSet;
    let mut node_set: HashSet<DofId> = HashSet::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            for &node in mesh.face_nodes(f) {
                node_set.insert(node as DofId);
            }
        }
    }

    let mut dof_set: HashSet<DofId> = node_set.clone();

    // Build a set of actual boundary edges from boundary face connectivity.
    // This is correct for all orders — using vertex heuristics is wrong because
    // an interior edge can have both endpoints on the boundary (e.g., the short
    // diagonal of a corner triangle element in a structured mesh).
    // In 2D: boundary face = edge (2 nodes) → 1 edge.
    // In 3D: boundary face = triangle (3 nodes) → 3 edges.
    let mut boundary_edges: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            if nodes.len() == 2 {
                // 2D: face is a line segment
                boundary_edges.insert(EdgeKey::new(nodes[0], nodes[1]));
            } else if nodes.len() >= 3 {
                // 3D: face is a triangle — add all 3 edges
                boundary_edges.insert(EdgeKey::new(nodes[0], nodes[1]));
                boundary_edges.insert(EdgeKey::new(nodes[1], nodes[2]));
                boundary_edges.insert(EdgeKey::new(nodes[0], nodes[2]));
            }
        }
    }

    // For P2 in 2D: include edge-midpoint DOFs on actual boundary edges.
    // Triangle P2 (6 DOFs): edge(v0,v1)→dofs[3], edge(v1,v2)→dofs[4], edge(v0,v2)→dofs[5]
    // Quad Q2 (9 DOFs): edge(v0,v1)→dofs[4], edge(v1,v2)→dofs[5], edge(v2,v3)→dofs[6], edge(v3,v0)→dofs[7]
    if dm.order == 2 && mesh.dim() == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        if dm.dofs_per_elem == 6 {
            // Triangle P2
            for e in 0..n_elems as u32 {
                let dofs  = dm.element_dofs(e);
                let nodes = mesh.element_nodes(e);
                let edge_pairs = [
                    (nodes[0], nodes[1], dofs[3]),
                    (nodes[1], nodes[2], dofs[4]),
                    (nodes[0], nodes[2], dofs[5]),
                ];
                for (a, b, edge_dof) in edge_pairs {
                    if boundary_edges.contains(&EdgeKey::new(a, b)) {
                        dof_set.insert(edge_dof);
                    }
                }
            }
        } else if dm.dofs_per_elem == 9 {
            // Quad Q2: 4 corners + 4 edge midpoints + 1 interior
            for e in 0..n_elems as u32 {
                let dofs  = dm.element_dofs(e);
                let nodes = mesh.element_nodes(e);
                // Edge midpoints at positions 4-7; interior DOF at 8 is never on boundary.
                let edge_pairs = [
                    (nodes[0], nodes[1], dofs[4]),  // bottom
                    (nodes[1], nodes[2], dofs[5]),  // right
                    (nodes[2], nodes[3], dofs[6]),  // top
                    (nodes[3], nodes[0], dofs[7]),  // left
                ];
                for (a, b, edge_dof) in edge_pairs {
                    if boundary_edges.contains(&EdgeKey::new(a, b)) {
                        dof_set.insert(edge_dof);
                    }
                }
                // Interior DOF (dofs[8]) is always inside the element, never on boundary.
            }
        }
    }

    // For P2 in 3D (TetP2): use edge_dof_map to find edge midpoint DOFs.
    // Edge DOF positions in the element: (v0,v1)→4, (v0,v2)→5, (v0,v3)→6,
    //                                    (v1,v2)→7, (v1,v3)→8, (v2,v3)→9
    if dm.order == 2 && mesh.dim() == 3 {
        for (&edge_key, &dof_id) in &dm.edge_dof_map {
            if boundary_edges.contains(&edge_key) {
                dof_set.insert(dof_id);
            }
        }
    }

    // For P3, include the two edge interior DOFs (at 1/3 and 2/3) on actual boundary edges.
    // DOF layout per element: verts 0,1,2; edge(v0→v1) at 3,4; edge(v1→v2) at 5,6; edge(v0→v2) at 7,8.
    if dm.order == 3 && mesh.dim() == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let edge_pairs = [
                (nodes[0], nodes[1], dofs[3], dofs[4]),
                (nodes[1], nodes[2], dofs[5], dofs[6]),
                (nodes[0], nodes[2], dofs[7], dofs[8]),
            ];
            for (a, b, edge_dof0, edge_dof1) in edge_pairs {
                if boundary_edges.contains(&EdgeKey::new(a, b)) {
                    dof_set.insert(edge_dof0);
                    dof_set.insert(edge_dof1);
                }
            }
        }
        // Note: bubble DOFs (position 9) are always interior — never boundary.
    }

    // For TetP3 (3D P3): use edge_dof2_map for edge interior DOFs.
    // Face DOFs (positions 16-19) are always interior — no face DOFs are on the boundary.
    if dm.order == 3 && mesh.dim() == 3 {
        for (&edge_key, &[d0, d1]) in &dm.edge_dof2_map {
            if boundary_edges.contains(&edge_key) {
                dof_set.insert(d0);
                dof_set.insert(d1);
            }
        }
        // Note: TetP3 face DOFs (16-19) are interior to faces, but they ARE on the boundary surface.
        // Unlike 2D bubble DOFs (interior to elements), 3D face DOFs lie on the boundary faces.
        // Include face DOFs that belong to boundary faces.
        let mut boundary_faces: std::collections::HashSet<FaceKey> = std::collections::HashSet::new();
        for f in 0..mesh.n_boundary_faces() as u32 {
            if tags.contains(&mesh.face_tag(f)) {
                let ns = mesh.face_nodes(f);
                if ns.len() >= 3 {
                    boundary_faces.insert(FaceKey::new(ns[0], ns[1], ns[2]));
                }
            }
        }
        // Iterate elements to find face DOFs on boundary faces.
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            if nodes.len() < 4 { continue; }
            let (n0, n1, n2, n3) = (nodes[0], nodes[1], nodes[2], nodes[3]);
            let faces = [
                (FaceKey::new(n0,n1,n2), dofs[16]),
                (FaceKey::new(n0,n1,n3), dofs[17]),
                (FaceKey::new(n0,n2,n3), dofs[18]),
                (FaceKey::new(n1,n2,n3), dofs[19]),
            ];
            for (fkey, face_dof) in faces {
                if boundary_faces.contains(&fkey) {
                    dof_set.insert(face_dof);
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
    use std::collections::HashSet;

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

// ─── Hanging-node constraints ───────────────────────────────────────────────

/// Apply hanging-node constraints to the assembled system `(K, f)`.
///
/// For each constraint `u_c = 0.5*(u_a + u_b)`, the constrained DOF is
/// eliminated by substituting the interpolation into the variational form.
///
/// The implementation rebuilds the matrix via COO format to handle new
/// sparsity entries that arise from the distribution step.
///
/// After solving, call [`recover_hanging_values`] to fill in constrained DOFs.
pub fn apply_hanging_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    // Build interpolation matrix P conceptually:
    //   For free DOF i:  u_i = x_i  (identity)
    //   For constrained c: u_c = 0.5*x_a + 0.5*x_b
    //
    // The constrained system is: P^T K P x = P^T f
    // where x has the constrained DOFs set to 0 (they'll be recovered later).
    //
    // In practice, we compute K' = P^T K P and f' = P^T f directly.

    let mut constraint_map = std::collections::HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, (c.parent_a, c.parent_b));
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // Handles chains: if DOF is constrained to parents that are also constrained,
    // the expansion follows through until only free DOFs remain.
    fn expand_dof(
        dof: usize,
        weight: f64,
        constraint_map: &std::collections::HashMap<usize, (usize, usize)>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; } // safety guard against cycles
        if let Some(&(a, b)) = constraint_map.get(&dof) {
            expand_dof(a, weight * 0.5, constraint_map, out, depth + 1);
            expand_dof(b, weight * 0.5, constraint_map, out, depth + 1);
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' in COO format.
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        // Effective row indices: recursively expand if constrained.
        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            // Effective column indices: recursively expand if constrained.
            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof(j, 1.0, &constraint_map, &mut j_targets, 0);

            // Add v * alpha_i * alpha_j to K'[ii, jj] for all target pairs.
            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs.
    for c in constraints {
        coo.add(c.constrained, c.constrained, 1.0);
    }

    // Build f' = P^T f — also with recursive expansion.
    // Process in reverse topological order (constrained DOFs that depend on
    // other constrained DOFs need those resolved first).
    // Simpler approach: expand each constrained DOF recursively.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    // Zero out constrained DOF RHS.
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Recover hanging-node DOF values after solving.
///
/// Sets `x[c] = 0.5*(x[a] + x[b])` for each hanging-node constraint.
/// Handles chained constraints by processing in topological order:
/// constraints whose parents are free are resolved first, then constraints
/// whose parents are now resolved, etc.
///
/// Call this after the linear solve and before post-processing.
pub fn recover_hanging_values(
    x: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    // Topological sort: process constraints whose parents are NOT constrained first.
    let mut remaining: Vec<&HangingNodeConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    // Iterate until all resolved (bounded by constraint count).
    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let a_free = !constrained_set.contains(&c.parent_a) || resolved.contains(&c.parent_a);
            let b_free = !constrained_set.contains(&c.parent_b) || resolved.contains(&c.parent_b);
            if a_free && b_free {
                x[c.constrained] = 0.5 * (x[c.parent_a] + x[c.parent_b]);
                resolved.insert(c.constrained);
                progress = true;
                false // remove from remaining
            } else {
                true // keep
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Handle any remaining (shouldn't happen with valid constraints, but just in case).
    for c in remaining {
        x[c.constrained] = 0.5 * (x[c.parent_a] + x[c.parent_b]);
    }
}

/// Apply hanging face constraints (3-D) to the assembled system `(K, f)`.
///
/// For each 3-D face constraint: `u_hang = (1/3)*(u_a + u_b + u_c)`.
/// Implements static condensation via P^T K P and P^T f, similar to edges.
pub fn apply_hanging_face_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[HangingFaceConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    let mut constraint_map = std::collections::HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, (c.parent_a, c.parent_b, c.parent_c));
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // For face constraints, each constrained DOF is a weighted sum of 3 parents.
    fn expand_dof_faces(
        dof: usize,
        weight: f64,
        constraint_map: &std::collections::HashMap<usize, (usize, usize, usize)>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; } // safety guard
        if let Some(&(a, b, c)) = constraint_map.get(&dof) {
            let w = weight / 3.0;
            expand_dof_faces(a, w, constraint_map, out, depth + 1);
            expand_dof_faces(b, w, constraint_map, out, depth + 1);
            expand_dof_faces(c, w, constraint_map, out, depth + 1);
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' in COO format.
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand_dof_faces(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof_faces(j, 1.0, &constraint_map, &mut j_targets, 0);

            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs.
    for c in constraints {
        coo.add(c.constrained, c.constrained, 1.0);
    }

    // Build f' = P^T f with recursive expansion.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets = Vec::new();
        expand_dof_faces(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    // Zero out constrained DOF RHS.
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Recover hanging face DOF values after solving.
///
/// Sets `x[c] = (1/3)*(x[a] + x[b] + x[c])` for each hanging-face constraint.
/// Handles chained constraints by processing in topological order.
pub fn recover_hanging_face_values(
    x: &mut [f64],
    constraints: &[HangingFaceConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    // Topological sort
    let mut remaining: Vec<&HangingFaceConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let a_free = !constrained_set.contains(&c.parent_a) || resolved.contains(&c.parent_a);
            let b_free = !constrained_set.contains(&c.parent_b) || resolved.contains(&c.parent_b);
            let c_free = !constrained_set.contains(&c.parent_c) || resolved.contains(&c.parent_c);
            if a_free && b_free && c_free {
                x[c.constrained] = (x[c.parent_a] + x[c.parent_b] + x[c.parent_c]) / 3.0;
                resolved.insert(c.constrained);
                progress = true;
                false
            } else {
                true
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Handle remaining
    for c in remaining {
        x[c.constrained] = (x[c.parent_a] + x[c.parent_b] + x[c.parent_c]) / 3.0;
    }
}

/// Prolongate an H1-P2 solution from a coarse Tri3 mesh to a refined Tri3 mesh.
///
/// The coarse P2 field is evaluated at every fine-space DOF coordinate using
/// the coarse element P2 basis, which works for hanging-node refinement and
/// multi-level NC refinement chains.
pub fn prolongate_p2_hanging<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    u_coarse: &[f64],
) -> Vec<f64> {
    assert_eq!(coarse_dm.order, 2, "prolongate_p2_hanging: coarse_dm must be P2");
    assert_eq!(fine_dm.order, 2, "prolongate_p2_hanging: fine_dm must be P2");
    assert_eq!(coarse_mesh.dim(), 2, "prolongate_p2_hanging: only 2-D supported");
    assert_eq!(u_coarse.len(), coarse_dm.n_dofs, "u_coarse length mismatch");

    let mut u_fine = vec![0.0_f64; fine_dm.n_dofs];
    let n_coarse_elems = coarse_mesh.n_elements() as u32;

    for dof in 0..fine_dm.n_dofs as u32 {
        let c = fine_dm.dof_coord(dof);
        let px = c[0];
        let py = c[1];

        let mut val = None;
        for e in 0..n_coarse_elems {
            let ns = coarse_mesh.element_nodes(e);
            if ns.len() < 3 {
                continue;
            }

            let c0 = coarse_mesh.node_coords(ns[0]);
            let c1 = coarse_mesh.node_coords(ns[1]);
            let c2 = coarse_mesh.node_coords(ns[2]);

            let x0 = c0[0]; let y0 = c0[1];
            let x1 = c1[0]; let y1 = c1[1];
            let x2 = c2[0]; let y2 = c2[1];

            let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if det.abs() < 1e-14 {
                continue;
            }

            let l1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / det;
            let l2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / det;
            let l0 = 1.0 - l1 - l2;

            let eps = 1e-10;
            if l0 < -eps || l1 < -eps || l2 < -eps {
                continue;
            }

            let edofs = coarse_dm.element_dofs(e);
            if edofs.len() < 6 {
                continue;
            }

            // P2 basis on triangle in barycentric coordinates.
            let n0 = l0 * (2.0 * l0 - 1.0);
            let n1 = l1 * (2.0 * l1 - 1.0);
            let n2 = l2 * (2.0 * l2 - 1.0);
            let n3 = 4.0 * l0 * l1;
            let n4 = 4.0 * l1 * l2;
            let n5 = 4.0 * l0 * l2;

            val = Some(
                n0 * u_coarse[edofs[0] as usize]
                    + n1 * u_coarse[edofs[1] as usize]
                    + n2 * u_coarse[edofs[2] as usize]
                    + n3 * u_coarse[edofs[3] as usize]
                    + n4 * u_coarse[edofs[4] as usize]
                    + n5 * u_coarse[edofs[5] as usize]
            );
            break;
        }

        u_fine[dof as usize] = val.unwrap_or_else(|| {
            panic!("prolongate_p2_hanging: fine DOF {dof} lies outside coarse mesh")
        });
    }

    u_fine
}

/// Identify pairs of DOFs that should be identified via periodic boundary conditions.
///
/// Given a `master_tag` boundary and a `slave_tag` boundary, finds pairs
/// `(slave_dof, master_dof)` such that `slave_coord + offset ≈ master_coord`.
///
/// Works for P1, P2, and P3 spaces:
/// - P1: vertex DOFs only
/// - P2: vertex DOFs + edge-midpoint DOFs (matched by pair of vertex pairs)
/// - P3: vertex DOFs + 2 edge DOFs per boundary edge + bubble DOFs are interior (skipped)
///
/// # Arguments
/// * `mesh`       — provides boundary face/node data
/// * `dm`         — DOF manager for the space
/// * `master_tag` — boundary tag of the "master" side
/// * `slave_tag`  — boundary tag of the "slave" side
/// * `offset`     — vector such that `x_slave + offset ≈ x_master`
/// * `tol`        — coordinate matching tolerance
///
/// # Returns
/// Sorted list of `(slave_dof, master_dof)` pairs.
pub fn identify_periodic_dof_pairs(
    mesh:       &dyn fem_mesh::topology::MeshTopology,
    dm:         &DofManager,
    master_tag: i32,
    slave_tag:  i32,
    offset:     &[f64],
    tol:        f64,
) -> Vec<(DofId, DofId)> {
    use std::collections::HashMap;

    let dim = dm.dof_coord(0).len();

    // Collect master boundary nodes with their coordinates.
    let mut master_nodes: HashMap<u32, Vec<f64>> = HashMap::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == master_tag {
            for &node in mesh.face_nodes(f) {
                let coords = mesh.node_coords(node).to_vec();
                master_nodes.insert(node, coords);
            }
        }
    }

    // For P1, pairs are just vertex node DOFs (node index == DOF index for P1).
    // Collect slave nodes and match to master by x_slave + offset ≈ x_master.
    let mut pairs: Vec<(DofId, DofId)> = Vec::new();

    // Map: master_node -> master_dof (for P1, node == dof).
    // For higher orders, we look up via dm.
    let find_master_dof = |master_node: u32| -> DofId { master_node as DofId };

    // Match slave vertex nodes to master vertex nodes.
    let mut slave_node_to_master_node: HashMap<u32, u32> = HashMap::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == slave_tag {
            for &slave_node in mesh.face_nodes(f) {
                let sc = mesh.node_coords(slave_node);
                // shifted coordinates
                let shifted: Vec<f64> = (0..dim).map(|i| sc[i] + offset[i]).collect();

                // Find matching master node
                let mut best: Option<(u32, f64)> = None;
                for (&mn, mc) in &master_nodes {
                    let dist: f64 = (0..dim).map(|i| (shifted[i] - mc[i]).powi(2)).sum::<f64>().sqrt();
                    if dist < tol
                        && best.is_none_or(|(_, d)| dist < d) {
                            best = Some((mn, dist));
                        }
                }

                if let Some((master_node, _)) = best {
                    slave_node_to_master_node.insert(slave_node, master_node);
                    let slave_dof = slave_node as DofId;
                    let master_dof = find_master_dof(master_node);
                    if slave_dof != master_dof {
                        pairs.push((slave_dof, master_dof));
                    }
                }
            }
        }
    }

    // For P2: also match edge-midpoint DOFs.
    // An edge midpoint DOF on the slave side is matched to the edge midpoint DOF
    // on the master side where both endpoints of the slave edge are matched.
    if dm.order == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        // Build set of slave boundary edges and master boundary edges.
        let mut slave_edges: HashMap<(u32, u32), DofId> = HashMap::new();
        let mut master_edges: HashMap<(u32, u32), DofId> = HashMap::new();

        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let edge_list = [
                (nodes[0], nodes[1], dofs[3]),
                (nodes[1], nodes[2], dofs[4]),
                (nodes[0], nodes[2], dofs[5]),
            ];
            for (a, b, edge_dof) in edge_list {
                let key = if a < b { (a, b) } else { (b, a) };
                // Check if both nodes are on slave boundary
                let a_slave = slave_node_to_master_node.contains_key(&a);
                let b_slave = slave_node_to_master_node.contains_key(&b);
                if a_slave && b_slave {
                    slave_edges.insert(key, edge_dof);
                }
                // Check if both nodes are on master boundary
                let a_master = master_nodes.contains_key(&a);
                let b_master = master_nodes.contains_key(&b);
                if a_master && b_master {
                    master_edges.insert(key, edge_dof);
                }
            }
        }

        // Match slave edge to master edge: the master edge has endpoints
        // that correspond to the master nodes matched to the slave edge's endpoints.
        for ((sa, sb), slave_dof) in &slave_edges {
            let ma = slave_node_to_master_node.get(sa);
            let mb = slave_node_to_master_node.get(sb);
            if let (Some(&ma), Some(&mb)) = (ma, mb) {
                let master_key = if ma < mb { (ma, mb) } else { (mb, ma) };
                if let Some(&master_dof) = master_edges.get(&master_key) {
                    if *slave_dof != master_dof {
                        pairs.push((*slave_dof, master_dof));
                    }
                }
            }
        }
    }

    // For P3: match the 2 edge interior DOFs per boundary edge.
    if dm.order == 3 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        let mut slave_edges: HashMap<(u32, u32), [DofId; 2]> = HashMap::new();
        let mut master_edges: HashMap<(u32, u32), [DofId; 2]> = HashMap::new();

        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            // [near_a, near_b] for edge a→b in element order
            let edge_list: [(u32, u32, [DofId; 2]); 3] = [
                (nodes[0], nodes[1], [dofs[3], dofs[4]]),
                (nodes[1], nodes[2], [dofs[5], dofs[6]]),
                (nodes[0], nodes[2], [dofs[7], dofs[8]]),
            ];
            for (a, b, edge_dofs) in edge_list {
                let (key, canonical_dofs) = if a < b {
                    ((a, b), edge_dofs)   // [near_a, near_b]
                } else {
                    ((b, a), [edge_dofs[1], edge_dofs[0]])  // flip to canonical order [near_min, near_max]
                };
                let a_slave = slave_node_to_master_node.contains_key(&a);
                let b_slave = slave_node_to_master_node.contains_key(&b);
                if a_slave && b_slave {
                    slave_edges.insert(key, canonical_dofs);
                }
                let a_master = master_nodes.contains_key(&a);
                let b_master = master_nodes.contains_key(&b);
                if a_master && b_master {
                    master_edges.insert(key, canonical_dofs);
                }
            }
        }

        // Match slave P3 edge DOFs to master P3 edge DOFs.
        // The 1/3 point near slave_min matches the 1/3 point near master_min.
        for ((sa, sb), slave_dofs) in &slave_edges {
            let ma = slave_node_to_master_node.get(sa);
            let mb = slave_node_to_master_node.get(sb);
            if let (Some(&ma), Some(&mb)) = (ma, mb) {
                let master_key = if ma < mb { (ma, mb) } else { (mb, ma) };
                if let Some(&master_dofs) = master_edges.get(&master_key) {
                    // slave canonical [near_sa, near_sb] matches master canonical [near_ma, near_mb]
                    // We need to check if the mapping preserves orientation:
                    // If sa→ma and sb→mb in the same "increasing" direction, dofs match directly.
                    // If sa→mb and sb→ma (orientation flip), dofs are swapped.
                    let master_near_sa = if ma < mb { master_dofs[0] } else { master_dofs[1] };
                    let master_near_sb = if ma < mb { master_dofs[1] } else { master_dofs[0] };
                    if slave_dofs[0] != master_near_sa {
                        pairs.push((slave_dofs[0], master_near_sa));
                    }
                    if slave_dofs[1] != master_near_sb {
                        pairs.push((slave_dofs[1], master_near_sb));
                    }
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Apply periodic boundary conditions to the assembled system `(K, f)`.
///
/// Converts each `(slave_dof, master_dof)` pair into a
/// `HangingNodeConstraint { constrained: slave, parent_a: master, parent_b: master }`
/// (degenerate: both parents are the same, giving `u_slave = master`).
///
/// Delegates to [`apply_hanging_constraints`] for the actual constraint application.
pub fn apply_periodic(
    mat:   &mut CsrMatrix<f64>,
    rhs:   &mut [f64],
    pairs: &[(DofId, DofId)],
) {
    let constraints: Vec<HangingNodeConstraint> = pairs.iter()
        .map(|&(slave, master)| HangingNodeConstraint {
            constrained: slave as usize,
            parent_a:    master as usize,
            parent_b:    master as usize,
        })
        .collect();
    apply_hanging_constraints(mat, rhs, &constraints);
}


#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{SimplexMesh, NCState};
    use fem_linalg::CooMatrix;
    

    fn simple_system() -> (CsrMatrix<f64>, Vec<f64>) {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0,  2.0); coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0); coo.add(1, 1,  2.0); coo.add(1, 2, -1.0);
        coo.add(2, 1, -1.0); coo.add(2, 2,  2.0);
        (coo.into_csr(), vec![1.0_f64; 3])
    }

    #[test]
    fn apply_dirichlet_zero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[0], &[0.0]);
        assert!((mat.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((mat.get(0, 1)).abs() < 1e-14);
        assert!((rhs[0]).abs() < 1e-14);
    }

    #[test]
    fn apply_dirichlet_nonzero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[2], &[5.0]);
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14);
        assert!((rhs[2] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn boundary_dofs_returns_sorted_valid_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 1);
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty());
        for &d in &dofs {
            assert!((d as usize) < dm.n_dofs, "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i-1]);
        }
    }

    #[test]
    fn boundary_dofs_p2_includes_edge_midpoints() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 2);
        let n_nodes = mesh.n_nodes();
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        // At least some DOFs should be edge-midpoint DOFs (index >= n_nodes)
        let edge_dofs: Vec<_> = dofs.iter().filter(|&&d| d as usize >= n_nodes).collect();
        assert!(!edge_dofs.is_empty(), "no edge-midpoint boundary DOFs found for P2");
    }

    #[test]
    fn boundary_dofs_hcurl_unit_square() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let dofs = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary edge DOFs");
        // 4×4 grid boundary has 4×4 = 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i - 1]);
        }
    }

    #[test]
    fn boundary_dofs_hdiv_unit_square() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let dofs = boundary_dofs_hdiv(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary face DOFs");
        // Same count as HCurl in 2-D: 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
    }

    // ── Hanging-node constraint tests ────────────────────────────────────────

    #[test]
    fn recover_hanging_values_simple() {
        let mut x = vec![2.0, 6.0, 0.0]; // DOF 2 is hanging between 0 and 1
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1,
        }];
        recover_hanging_values(&mut x, &constraints);
        assert!((x[2] - 4.0).abs() < 1e-14, "expected 0.5*(2+6)=4, got {}", x[2]);
    }

    #[test]
    fn recover_hanging_values_chained() {
        // DOF 2 = mid(0, 1), DOF 3 = mid(1, 2)
        // DOF 2 should be recovered first since its parents are free,
        // then DOF 3 uses the recovered DOF 2.
        let mut x = vec![0.0, 4.0, 0.0, 0.0];
        let constraints = vec![
            HangingNodeConstraint { constrained: 2, parent_a: 0, parent_b: 1 },
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2 },
        ];
        recover_hanging_values(&mut x, &constraints);
        // DOF 2 = 0.5*(0 + 4) = 2
        assert!((x[2] - 2.0).abs() < 1e-14, "expected x[2]=2, got {}", x[2]);
        // DOF 3 = 0.5*(4 + 2) = 3
        assert!((x[3] - 3.0).abs() < 1e-14, "expected x[3]=3, got {}", x[3]);
    }

    #[test]
    fn apply_hanging_constraints_chained() {
        // 6-DOF system: DOF 3 = mid(1, 2), DOF 4 = mid(2, 3).
        // DOF 4 depends on DOF 3 which is also constrained.
        // After expansion: DOF 4 = 0.5*(u2 + 0.5*(u1 + u2)) = 0.25*u1 + 0.75*u2.
        let n = 6;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2 },
            HangingNodeConstraint { constrained: 4, parent_a: 2, parent_b: 3 },
        ];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Constrained rows should be identity.
        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);
    }

    #[test]
    fn apply_hanging_constraints_identity_row() {
        // 4-DOF system: DOF 2 is constrained to 0.5*(DOF 0 + DOF 1).
        // After apply_hanging_constraints, row 2 should be identity.
        let mut coo = CooMatrix::<f64>::new(4, 4);
        for i in 0..4 {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < 3     { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; 4];
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1,
        }];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Row 2 should be: K[2,2] = 1, all others 0.
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14, "K[2,2] should be 1");
        assert!((mat.get(2, 0)).abs() < 1e-14, "K[2,0] should be 0");
        assert!((mat.get(2, 1)).abs() < 1e-14, "K[2,1] should be 0");
        assert!((mat.get(2, 3)).abs() < 1e-14, "K[2,3] should be 0");
        assert!((rhs[2]).abs() < 1e-14, "rhs[2] should be 0");

        // Column 2 should be zero in all other rows.
        assert!((mat.get(0, 2)).abs() < 1e-14, "K[0,2] should be 0");
        assert!((mat.get(1, 2)).abs() < 1e-14, "K[1,2] should be 0");
        assert!((mat.get(3, 2)).abs() < 1e-14, "K[3,2] should be 0");
    }

    #[test]
    fn hanging_constraint_preserves_solvability() {
        // Build a small system, apply constraint, solve, recover.
        // 5-DOF 1-D Laplacian: DOF 2 is hanging between 1 and 3.
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        // Dirichlet: DOF 0 = 0, DOF 4 = 0.
        apply_dirichlet(&mut mat, &mut rhs, &[0, 4], &[0.0, 0.0]);

        // Hanging constraint: DOF 2 = 0.5*(DOF 1 + DOF 3).
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 1, parent_b: 3,
        }];
        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Solve with simple direct solver (small enough).
        let mut x = vec![0.0; n];
        // Simple Gauss-Seidel iteration for this small system.
        for _ in 0..1000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                x[i] = s / diag;
            }
        }

        // Recover hanging DOF.
        recover_hanging_values(&mut x, &constraints);

        // x[2] should be average of x[1] and x[3].
        assert!(
            (x[2] - 0.5 * (x[1] + x[3])).abs() < 1e-8,
            "hanging DOF: x[2]={}, 0.5*(x[1]+x[3])={}",
            x[2], 0.5 * (x[1] + x[3])
        );

        // Boundary conditions should hold.
        assert!(x[0].abs() < 1e-10, "x[0] = {}, expected 0", x[0]);
        assert!(x[4].abs() < 1e-10, "x[4] = {}, expected 0", x[4]);
    }

    #[test]
    fn prolongate_p2_hanging_is_exact_for_quadratic() {
        let coarse = SimplexMesh::<2>::unit_square_tri(2);
        let coarse_dm = DofManager::new(&coarse, 2);

        let f = |x: f64, y: f64| -> f64 { x * x + x * y + y * y + 2.0 * x - y + 1.0 };
        let mut u_coarse = vec![0.0_f64; coarse_dm.n_dofs];
        for d in 0..coarse_dm.n_dofs as u32 {
            let c = coarse_dm.dof_coord(d);
            u_coarse[d as usize] = f(c[0], c[1]);
        }

        let mut nc = NCState::new();
        let (fine, _, _) = nc.refine(&coarse, &[0, 1, 2]);
        let fine_dm = DofManager::new(&fine, 2);

        let u_fine = prolongate_p2_hanging(&coarse, &coarse_dm, &fine_dm, &u_coarse);

        for d in 0..fine_dm.n_dofs as u32 {
            let c = fine_dm.dof_coord(d);
            let expected = f(c[0], c[1]);
            assert!(
                (u_fine[d as usize] - expected).abs() < 1e-10,
                "dof {d}: got {}, expected {}",
                u_fine[d as usize],
                expected
            );
        }
    }

    #[test]
    fn recover_hanging_face_values_simple() {
        // Test face constraint recovery: u[c] = (1/3)*(u[a] + u[b] + u[c])
        let mut x = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        let constraints = vec![
            HangingFaceConstraint {
                constrained: 2,
                parent_a: 0,
                parent_b: 1,
                parent_c: 3,
            },
        ];

        recover_hanging_face_values(&mut x, &constraints);

        // x[2] should be 1/3 * (1 + 2 + 4) = 7/3 ≈ 2.333...
        let expected = (1.0 + 2.0 + 4.0) / 3.0;
        assert!(
            (x[2] - expected).abs() < 1e-10,
            "hanging face DOF: x[2]={}, expected {}", x[2], expected
        );
    }

    #[test]
    fn recover_hanging_face_values_chained() {
        // Test chained face constraints
        let mut x = vec![1.0, 2.0, 0.0, 3.0, 0.0];
        let constraints = vec![
            // x[2] = (1/3)*(x[0] + x[1] + x[3])
            HangingFaceConstraint {
                constrained: 2,
                parent_a: 0,
                parent_b: 1,
                parent_c: 3,
            },
            // x[4] = (1/3)*(x[0] + x[2] + x[3]) — depends on x[2]
            HangingFaceConstraint {
                constrained: 4,
                parent_a: 0,
                parent_b: 2,
                parent_c: 3,
            },
        ];

        recover_hanging_face_values(&mut x, &constraints);

        // x[2] = (1/3)*(1 + 2 + 3) = 2
        assert!(
            (x[2] - 2.0).abs() < 1e-10,
            "first constraint: x[2]={}, expected 2", x[2]
        );

        // x[4] = (1/3)*(1 + 2 + 3) = 2
        assert!(
            (x[4] - 2.0).abs() < 1e-10,
            "second constraint: x[4]={}, expected 2", x[4]
        );
    }
}
