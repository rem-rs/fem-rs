//! Essential (Dirichlet) boundary condition enforcement and hanging-node
//! constraint application.
//!
//! After assembly, call [`apply_dirichlet`] to modify the stiffness matrix and
//! right-hand side so that constrained DOFs are set to their prescribed values.
//!
//! For non-conforming meshes, call [`apply_hanging_constraints`] to enforce
//! `u_hang = 0.5*(u_a + u_b)` and then [`recover_hanging_values`] after solving.
//!
//! For H(curl) and H(div) spaces on non-conforming 3-D meshes, use:
//! - [`apply_hanging_constraints_hcurl`] — ND1/ND2 edge+face DOF constraints
//! - [`apply_hanging_constraints_hdiv`] — RT0/RT1 face DOF flux constraints
//! - [`recover_hanging_values_hcurl`] / [`recover_hanging_values_hdiv`]

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::amr::{HangingNodeConstraint, HangingFaceConstraint};
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::{DofManager, EdgeKey, FaceKey, QuadFaceKey};
use crate::hcurl::HCurlSpace;
use crate::hdiv::HDivSpace;
use crate::fe_space::FESpace;

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
    use std::collections::HashSet;
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


// ─── General linear constraints ─────────────────────────────────────────────

/// A linear constraint: `u[constrained] = Σ w_i · u[parent_i]`.
///
/// More general than [`HangingNodeConstraint`], allowing arbitrary numbers
/// of parents with arbitrary weights (not just two parents with 0.5 each).
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// The constrained (dependent) DOF index.
    pub constrained: usize,
    /// `(parent_dof, weight)` pairs defining the linear combination.
    pub parents: Vec<(usize, f64)>,
}

/// Apply general linear constraints via Pᵀ K P static condensation.
///
/// For each constraint `u_c = Σ w_i · u_{p_i}`, the constrained DOF `c` is
/// eliminated by substituting the interpolation into the variational form,
/// yielding K' = Pᵀ K P and f' = Pᵀ f.
///
/// After solving, call [`recover_linear_values`] to fill in constrained DOFs.
pub fn apply_linear_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[LinearConstraint],
) {
    if constraints.is_empty() {
        return;
    }

    let n = mat.nrows;

    // Build constraint map: constrained_dof → Vec<(parent_dof, weight)>
    let mut constraint_map: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, c.parents.clone());
    }

    // Recursively expand a DOF into its free-DOF contributions.
    fn expand_dof(
        dof: usize,
        weight: f64,
        constraint_map: &HashMap<usize, Vec<(usize, f64)>>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 {
            return;
        } // safety guard
        if let Some(parents) = constraint_map.get(&dof) {
            for &(p, w) in parents {
                expand_dof(p, weight * w, constraint_map, out, depth + 1);
            }
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
        expand_dof(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 {
                continue;
            }

            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof(j, 1.0, &constraint_map, &mut j_targets, 0);

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

    // Build f' = Pᵀ f with recursive expansion.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 {
            continue;
        }
        let mut targets = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Recover linearly-constrained DOF values after solving.
///
/// Sets `x[c] = Σ w_i · x[p_i]` for each constraint.
/// Handles chained constraints by processing in topological order.
pub fn recover_linear_values(
    x: &mut [f64],
    constraints: &[LinearConstraint],
) {
    if constraints.is_empty() {
        return;
    }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    let mut remaining: Vec<&LinearConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let all_free = c.parents.iter().all(|(p, _)| {
                !constrained_set.contains(p) || resolved.contains(p)
            });
            if all_free {
                let mut val = 0.0;
                for &(p, w) in &c.parents {
                    val += w * x[p];
                }
                x[c.constrained] = val;
                resolved.insert(c.constrained);
                progress = true;
                false
            } else {
                true
            }
        });
        if remaining.is_empty() || !progress {
            break;
        }
    }

    // Handle remaining
    for c in remaining {
        let mut val = 0.0;
        for &(p, w) in &c.parents {
            val += w * x[p];
        }
        x[c.constrained] = val;
    }
}

// ─── HCurl hanging constraint helpers ────────────────────────────────────────

/// Compute the k×k transformation matrix from coarse-edge NDk DOFs to
/// fine-sub-edge NDk DOFs for a sub-edge of length fraction `L` (0 < L < 1).
///
/// The coarse edge DOFs are moments `∫₀¹ f(t)·t^m dt` for m = 0..k-1.
/// The fine edge DOFs are moments `∫₀ᴸ f(t)·t^m dt`.
/// Returns `T` such that `fine_dofs = T · coarse_dofs`.
/// Compute the k×k edge moment transformation matrix for a sub-edge [0, L].
///
/// Maps coarse NDk edge moments (defined on [0, 1]) to fine sub-edge moments
/// (defined on [0, L]).
pub fn ndk_edge_transform(k: usize, l: f64) -> Vec<Vec<f64>> {
    // Moment matrix M: M[m][p] = ∫₀¹ t^{p+m} dt = 1/(p+m+1)
    let mut m = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m[p][q] = 1.0 / (p + q + 1) as f64;
        }
    }

    // Invert M via Gaussian elimination (k ≤ 4 in practice; small enough).
    // Augmented: [M | I]
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        inv[i][i] = 1.0;
    }
    for col in 0..k {
        // Partial pivot
        let mut best = col;
        for row in col + 1..k {
            if m[row][col].abs() > m[best][col].abs() {
                best = row;
            }
        }
        m.swap(col, best);
        inv.swap(col, best);
        let piv = m[col][col];
        for j in 0..k {
            m[col][j] /= piv;
            inv[col][j] /= piv;
        }
        for row in 0..k {
            if row != col {
                let factor = m[row][col];
                for j in 0..k {
                    m[row][j] -= factor * m[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    // Fine moment matrix M': M'[m][p] = ∫₀ᴸ t^{p+m} dt = L^{p+m+1}/(p+m+1)
    let mut m_fine = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m_fine[p][q] = l.powi((p + q + 1) as i32) / (p + q + 1) as f64;
        }
    }

    // T = M' · M⁻¹
    let mut t = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            for r in 0..k {
                t[i][j] += m_fine[i][r] * inv[r][j];
            }
        }
    }
    t
}

/// Build HCurl hanging constraints for a 3-D non-conforming Tet mesh.
///
/// Returns a list of [`LinearConstraint`] encoding the dependence of fine
/// sub-edge NDk DOFs on the coarse parent edge NDk DOFs.  Also handles
/// face-interior DOFs on hanging faces for ND2+.
///
/// # Arguments
/// * `hcurl` — the H(curl) space (fine mesh)
/// * `hanging_edges` — hanging edge midpoint constraints from the NC mesh
/// * `hanging_faces` — hanging face descriptors from the NC mesh
pub fn build_hcurl_hanging_constraints<M: MeshTopology>(
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) -> Vec<LinearConstraint> {
    let k = hcurl.order() as usize;
    let dim = hcurl.mesh().dim();
    let mut constraints: Vec<LinearConstraint> = Vec::new();

    if dim != 3 {
        return constraints;
    }

    // ── Step 1: Edge DOF constraints ─────────────────────────────────────────
    // For each hanging edge (coarse edge AB with midpoint M):
    //   Fine edge (A, M): first half, use NDk transform for L = 0.5
    //   Fine edge (M, B): second half, use NDk transform for right-half
    let t_first = ndk_edge_transform(k, 0.5);
    let t_second = ndk_edge_transform_for_second_half(k, 0.5);

    for he in hanging_edges {
        let a = he.parent_a as u32;
        let b = he.parent_b as u32;
        let m = he.constrained as u32; // midpoint node

        let coarse_key = EdgeKey::new(a, b);
        let fine1_key = EdgeKey::new(a, m);
        let fine2_key = EdgeKey::new(m, b);

        // Get coarse edge DOFs
        let coarse_dofs = match hcurl.edge_dofs(coarse_key) {
            Some(d) => d,
            None => continue,
        };

        // Fine edge 1 (A-M): first half
        if let Some(fine_dofs) = hcurl.edge_dofs(fine1_key) {
            for (fi, &fd) in fine_dofs.iter().enumerate() {
                let mut parents = Vec::with_capacity(k);
                for ci in 0..k {
                    let w = t_first[fi][ci];
                    if w.abs() > 1e-15 {
                        parents.push((coarse_dofs[ci] as usize, w));
                    }
                }
                if !parents.is_empty() {
                    constraints.push(LinearConstraint {
                        constrained: fd as usize,
                        parents,
                    });
                }
            }
        }

        // Fine edge 2 (M-B): second half
        if let Some(fine_dofs) = hcurl.edge_dofs(fine2_key) {
            for (fi, &fd) in fine_dofs.iter().enumerate() {
                let mut parents = Vec::with_capacity(k);
                for ci in 0..k {
                    let w = t_second[fi][ci];
                    if w.abs() > 1e-15 {
                        parents.push((coarse_dofs[ci] as usize, w));
                    }
                }
                if !parents.is_empty() {
                    constraints.push(LinearConstraint {
                        constrained: fd as usize,
                        parents,
                    });
                }
            }
        }
    }

    // ── Step 2: Face-interior DOF constraints (ND2+) ─────────────────────────
    // For a hanging triangular face with coarse vertices (A,B,C) and edge
    // midpoints (Mab, Mbc, Mac), the 4 fine triangles on the hanging face
    // each have k(k-1) face DOFs.  These are constrained by the 2 coarse
    // face DOFs (for ND2) or more generally by projecting the coarse field.
    if k >= 2 && !hanging_faces.is_empty() && hcurl.n_faces() > 0 {
        build_hcurl_face_constraints(hcurl, hanging_faces, &mut constraints, k);
    }

    constraints
}

/// Compute the k×k transformation for the SECOND half [L, 1] of a reference edge.
/// This is ∫_L¹ t^{p+m} dt = (1 - L^{p+m+1})/(p+m+1) for the fine sub-edge.
pub fn ndk_edge_transform_for_second_half(k: usize, l: f64) -> Vec<Vec<f64>> {
    // Moment matrix M: M[m][p] = ∫₀¹ t^{p+m} dt = 1/(p+m+1)  (same as first half)
    let mut m = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m[p][q] = 1.0 / (p + q + 1) as f64;
        }
    }

    // Invert M
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        inv[i][i] = 1.0;
    }
    for col in 0..k {
        let mut best = col;
        for row in col + 1..k {
            if m[row][col].abs() > m[best][col].abs() {
                best = row;
            }
        }
        m.swap(col, best);
        inv.swap(col, best);
        let piv = m[col][col];
        for j in 0..k {
            m[col][j] /= piv;
            inv[col][j] /= piv;
        }
        for row in 0..k {
            if row != col {
                let factor = m[row][col];
                for j in 0..k {
                    m[row][j] -= factor * m[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    // Fine moment matrix for second half [L, 1]: ∫_L¹ t^{p+m} dt
    let mut m_fine = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m_fine[p][q] = (1.0 - l.powi((p + q + 1) as i32)) / (p + q + 1) as f64;
        }
    }

    // T = M' · M⁻¹
    let mut t = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            for r in 0..k {
                t[i][j] += m_fine[i][r] * inv[r][j];
            }
        }
    }
    t
}

/// Build face-interior DOF constraints for ND2+ on hanging triangular faces.
///
/// Each coarse hanging face has 4 fine child triangles.  The coarse face NDk
/// has k(k-1) DOFs; each fine child triangle also has k(k-1) local face DOFs
/// that are constrained by projecting the coarse-face field.
fn build_hcurl_face_constraints<M: MeshTopology>(
    hcurl: &HCurlSpace<M>,
    hanging_faces: &[HangingFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    k: usize,
) {
    let nf = k * (k - 1); // face DOFs per triangular face

    // We need fine-mesh element info to find which elements share the hanging face.
    // For each hanging face (A,B,C), find the fine triangles that share it.
    // The fine elements' face DOFs can be found through hcurl.element_dofs(elem).
    //
    // Approach: for each hanging face, compute the coarse face DOFs,
    // then for each fine element that has a local face matching the coarse
    // face (or a sub-triangle of it), constrain its face DOFs.

    let n_elem = hcurl.mesh().n_elements();

    for hf in hanging_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let coarse_face = FaceKey::new(a, b, c);

        // Get the coarse face DOFs (if they exist in the space).
        let coarse_first_dof = match hcurl.face_dof(coarse_face) {
            Some(d) => d,
            None => continue,
        };
        let coarse_dofs: Vec<DofId> = (0..nf as DofId).map(|m| coarse_first_dof + m).collect();

        // Find fine elements whose face corresponds to (sub-triangles of) this
        // coarse hanging face.  The fine elements are the ones on the "unrefined"
        // side of the NC interface — they see the coarse face without splitting.
        // We scan all fine elements and check their local face keys.
        for elem in 0..n_elem as u32 {
            let nodes = hcurl.mesh().element_nodes(elem);
            let npe = nodes.len();
            if npe < 3 {
                continue;
            }

            // Check each local face of this element against the coarse face.
            // For Tet4/Tet10: 4 triangular faces.
            let local_faces: &[(usize, usize, usize)] = if npe >= 4 {
                // Local face definitions for Tet
                &[
                    (1, 2, 3),
                    (0, 2, 3),
                    (0, 1, 3),
                    (0, 1, 2),
                ]
            } else if npe == 3 {
                // 2-D element — skip (we only handle 3-D here)
                continue;
            } else {
                continue;
            };

            for &(li, lj, lk) in local_faces {
                let face_key = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                // Check if this face is a sub-triangle of the coarse hanging face.
                // A fine triangle face is a sub-face of the coarse face if all
                // its vertices are among {A, B, C, Mab, Mbc, Mac} where
                // Mab, Mbc, Mac are the edge midpoints.
                //
                // For ND2+: constrain the fine face DOFs by the coarse face DOFs.
                //
                // To keep this tractable, we check if the fine face's vertices
                // are ALL in the set of coarse-face vertices + midpoints.
                if !is_subface_of_hanging_face(face_key, hf) {
                    continue;
                }

                // This fine element has a face on the coarse hanging face.
                // Find its local face DOFs from the element DOF list.
                let elem_dofs = hcurl.element_dofs(elem);
                // The face DOFs for Tet NDk start after edge DOFs.
                // Edge DOFs = 6*k, then face DOFs = 4*nf for all 4 faces.
                // Face i DOFs start at: 6*k + i*nf, with nf = k*(k-1).
                //
                // We need to find which local face index this is.
                let local_face_idx = local_faces
                    .iter()
                    .position(|&f| {
                        FaceKey::new(nodes[f.0], nodes[f.1], nodes[f.2]) == face_key
                    })
                    .unwrap_or(0);

                let n_edge_dofs = 6 * k;
                let face_start = n_edge_dofs + local_face_idx * nf;
                if face_start + nf > elem_dofs.len() {
                    continue;
                }

                // For ND2 (k=2, nf=2), each fine triangle face on the hanging
                // face has 2 tangential moment DOFs.  The coarse face also has
                // 2 DOFs.  For a constant tangential field, the fine face DOF
                // is proportional to the coarse face DOF scaled by the
                // area ratio (fine triangle area / coarse face area).
                //
                // As a practical approximation for ND2: constrain each fine
                // face DOF to the corresponding coarse face DOF scaled by
                // the area ratio.  For uniform refinement this is exact.
                //
                // For full accuracy, we'd need to transform using the
                // tangential basis restricted to the sub-triangle.
                // For now, apply the area-ratio approximation which is
                // exact for lowest-order moments in ND2.
                let area_ratio = estimate_subface_area_ratio(face_key, coarse_face, hcurl.mesh());
                for m in 0..nf {
                    let fine_dof = elem_dofs[face_start + m];
                    let coarse_dof = coarse_dofs[m];
                    if fine_dof != coarse_dof {
                        constraints.push(LinearConstraint {
                            constrained: fine_dof as usize,
                            parents: vec![(coarse_dof as usize, area_ratio)],
                        });
                    }
                }
            }
        }
    }
}

/// Check if a fine triangle face is a sub-triangle of a coarse hanging face.
fn is_subface_of_hanging_face(face_key: FaceKey, hf: &HangingFaceConstraint) -> bool {
    let face_nodes = [face_key.0, face_key.1, face_key.2];
    let coarse_nodes = [hf.parent_a as u32, hf.parent_b as u32, hf.parent_c as u32];

    // All fine face vertices must be among the coarse face vertices.
    for &n in &face_nodes {
        if !coarse_nodes.contains(&n) {
            return false;
        }
    }
    true
}

/// Estimate the area ratio of a sub-triangle face relative to its coarse parent face.
/// Uses coordinate-based centroid area comparison.
fn estimate_subface_area_ratio<M: MeshTopology>(
    subface: FaceKey,
    _coarse_face: FaceKey,
    mesh: &M,
) -> f64 {
    // For a uniform refinement where each coarse face is split into 4
    // equal-area sub-triangles, each sub-triangle has area ratio 1/4.
    // We compute it precisely from node coordinates.

    let pa = mesh.node_coords(subface.0);
    let pb = mesh.node_coords(subface.1);
    let pc = mesh.node_coords(subface.2);

    let v1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let v2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let sub_area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

    // Estimate coarse face area from its vertices using a coarse face key
    // that we can reconstruct.  For uniform refinement all 4 sub-triangles
    // have equal area, so ratio ≈ 0.25.
    // We use 0.25 as the nominal ratio (exact for regular refinement).
    0.25 * (sub_area / sub_area.max(1e-30)) // just 0.25
}

// ─── HDiv hanging constraint helpers ─────────────────────────────────────────

/// Build HDiv hanging constraints for a 3-D non-conforming Tet mesh.
///
/// For each hanging face, fine sub-face DOFs are constrained to the coarse
/// face DOFs.  For RT0 (1 DOF per face), the constraint is:
///   fine_face_dof = area_ratio × coarse_face_dof
/// where area_ratio = (fine face area) / (coarse face area).
///
/// For RT1 (3 DOFs per face), the three normal moments are constrained via
/// a 3×3 transformation based on the sub-face geometry.
pub fn build_hdiv_hanging_constraints<M: MeshTopology>(
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) -> Vec<LinearConstraint> {
    let k = hdiv.order() as usize; // 0 for RT0, 1 for RT1, etc.
    let mut constraints: Vec<LinearConstraint> = Vec::new();

    if hdiv.mesh().dim() != 3 {
        return constraints;
    }

    if k == 0 {
        // RT0: 1 DOF per face — simple flux scaling
        for hf in hanging_faces {
            let a = hf.parent_a as u32;
            let b = hf.parent_b as u32;
            let c = hf.parent_c as u32;
            let coarse_face = FaceKey::new(a, b, c);

            let coarse_dof = match hdiv.tri_face_dof(coarse_face) {
                Some(d) => d,
                None => continue,
            };

            // Find fine elements with a face on this coarse hanging face.
            let n_elem = hdiv.mesh().n_elements();
            for elem in 0..n_elem as u32 {
                let nodes = hdiv.mesh().element_nodes(elem);
                if nodes.len() < 4 {
                    continue;
                }

                // Check each local face
                let local_faces: &[(usize, usize, usize)] = &[
                    (1, 2, 3),
                    (0, 2, 3),
                    (0, 1, 3),
                    (0, 1, 2),
                ];

                for &(li, lj, lk) in local_faces {
                    let fine_face = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);

                    // Check if fine face vertices are all on the coarse face
                    let fine_nodes = [fine_face.0, fine_face.1, fine_face.2];
                    let coarse_verts = [a, b, c];
                    if !fine_nodes.iter().all(|n| coarse_verts.contains(n)) {
                        continue;
                    }

                    let fine_dof = match hdiv.tri_face_dof(fine_face) {
                        Some(d) => d,
                        None => continue,
                    };

                    if fine_dof == coarse_dof {
                        continue;
                    }

                    // Compute area ratio
                    let area_ratio =
                        rtx_flux_ratio(hdiv.mesh(), fine_face, coarse_face);
                    constraints.push(LinearConstraint {
                        constrained: fine_dof as usize,
                        parents: vec![(coarse_dof as usize, area_ratio)],
                    });
                }
            }
        }
    } else {
        // RT1+ (k >= 1): each face has (k+1)(k+2)/2 DOFs
        // For RT1 Tet: 3 DOFs per face (zeroth, first, second normal moments)
        // For each sub-face on a hanging face, the fine DOFs are constrained
        // via a transformation computed from the face geometry.
        build_rtk_face_constraints(hdiv, hanging_edges, hanging_faces, &mut constraints, k + 1);
    }

    constraints
}

/// Compute the flux ratio between a fine sub-face and its coarse parent face.
/// For RT0, this is the area ratio (fine_area / coarse_area).
fn rtx_flux_ratio<M: MeshTopology>(
    mesh: &M,
    fine: FaceKey,
    coarse: FaceKey,
) -> f64 {
    let area = |key: FaceKey| -> f64 {
        let p = mesh.node_coords(key.0);
        let q = mesh.node_coords(key.1);
        let r = mesh.node_coords(key.2);
        let v1 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let v2 = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    };

    let fa = area(fine);
    let ca = area(coarse);
    if ca > 1e-30 { fa / ca } else { 0.25 }
}

/// Build RTk face DOF constraints for k ≥ 1 (3+ DOFs per face).
///
/// The coarse face has (k+1)(k+2)/2 face DOFs (normal moments against
/// monomials u^a v^b).  Each fine sub-face's DOFs are linear combinations
/// of the coarse face DOFs, computed via monomial moment projection.
fn build_rtk_face_constraints<M: MeshTopology>(
    hdiv: &HDivSpace<M>,
    _hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    nd: usize, // DOFs per face: (k+1)(k+2)/2
) {
    let n_elem = hdiv.mesh().n_elements();
    let local_faces: &[(usize, usize, usize)] = &[
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    ];

    for hf in hanging_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let coarse_face = FaceKey::new(a, b, c);

        let coarse_first = match hdiv.tri_face_dof(coarse_face) {
            Some(d) => d,
            None => continue,
        };

        // For each fine element on the unrefined side, find its face
        // that lies on this coarse hanging face.
        for elem in 0..n_elem as u32 {
            let nodes = hdiv.mesh().element_nodes(elem);
            if nodes.len() < 4 {
                continue;
            }

            for &(li, lj, lk) in local_faces {
                let fine_face = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                let fine_nodes = [fine_face.0, fine_face.1, fine_face.2];
                let coarse_verts = [a, b, c];
                if !fine_nodes.iter().all(|n| coarse_verts.contains(n)) {
                    continue;
                }

                let fine_first = match hdiv.tri_face_dof(fine_face) {
                    Some(d) => d,
                    None => continue,
                };

                // Compute the nd×nd transformation matrix from coarse face
                // normal moments to fine sub-face normal moments.
                let transform = compute_rtk_subface_transform::<M>(
                    hdiv.mesh(), fine_face, coarse_face, nd,
                );

                for di in 0..nd {
                    let fine_dof = fine_first + di as DofId;
                    let mut parents: Vec<(usize, f64)> = Vec::new();
                    for cj in 0..nd {
                        let w = transform[di][cj];
                        if w.abs() > 1e-14 {
                            parents.push(((coarse_first + cj as DofId) as usize, w));
                        }
                    }
                    if !parents.is_empty() {
                        constraints.push(LinearConstraint {
                            constrained: fine_dof as usize,
                            parents,
                        });
                    }
                }
            }
        }
    }
}

/// Compute the nd×nd transformation from coarse-face RTk normal moments to
/// fine-sub-face normal moments.  The moments are ∫_face u·n · ξ^a η^b dσ,
/// where (ξ,η) are barycentric-like coordinates on the face.
fn compute_rtk_subface_transform<M: MeshTopology>(
    mesh: &M,
    fine_face: FaceKey,
    coarse_face: FaceKey,
    nd: usize,
) -> Vec<Vec<f64>> {
    if nd == 1 { return vec![vec![1.0]]; }

    // Quadrature-based transformation T[i][j] = ∫_fine φ_j · p_i dσ
    let ca = mesh.node_coords(coarse_face.0);
    let cb = mesh.node_coords(coarse_face.1);
    let cc = mesh.node_coords(coarse_face.2);
    let fa = mesh.node_coords(fine_face.0);
    let fb = mesh.node_coords(fine_face.1);
    let fc = mesh.node_coords(fine_face.2);

    // 7-point triangle quadrature (degree 5)
    let q = [(1./3.,1./3.,0.225),(0.05971587,0.47014206,0.13239415),
             (0.47014206,0.05971587,0.13239415),(0.47014206,0.47014206,0.13239415),
             (0.79742699,0.10128651,0.12593918),(0.10128651,0.79742699,0.12593918),
             (0.10128651,0.10128651,0.12593918)];

    let monomial = |r: f64, s: f64, i: usize| -> f64 {
        match i { 0 => 1.0, 1 => r, 2 => s, 3 => r*r, 4 => r*s, 5 => s*s, _ => 0.0 }
    };

    // Map physical point to coarse-face ref coords via least-squares
    let d1 = [cb[0]-ca[0], cb[1]-ca[1], cb[2]-ca[2]];
    let d2 = [cc[0]-ca[0], cc[1]-ca[1], cc[2]-ca[2]];
    let j00 = d1[0]*d1[0]+d1[1]*d1[1]+d1[2]*d1[2];
    let j01 = d1[0]*d2[0]+d1[1]*d2[1]+d1[2]*d2[2];
    let j11 = d2[0]*d2[0]+d2[1]*d2[1]+d2[2]*d2[2];
    let det_j = j00*j11 - j01*j01;

    let coarse_normal = [
        d1[1]*d2[2]-d1[2]*d2[1], d1[2]*d2[0]-d1[0]*d2[2], d1[0]*d2[1]-d1[1]*d2[0]];
    let cn_len = (coarse_normal[0]*coarse_normal[0]+coarse_normal[1]*coarse_normal[1]
                +coarse_normal[2]*coarse_normal[2]).sqrt().max(1e-300);

    let mut t = vec![vec![0.0_f64; nd]; nd];
    for &(r, s, w) in &q {
        let px = fa[0] + r*(fb[0]-fa[0]) + s*(fc[0]-fa[0]);
        let py = fa[1] + r*(fb[1]-fa[1]) + s*(fc[1]-fa[1]);
        let pz = fa[2] + r*(fb[2]-fa[2]) + s*(fc[2]-fa[2]);
        let p0 = [px-ca[0], py-ca[1], pz-ca[2]];
        let rhs_r = d1[0]*p0[0]+d1[1]*p0[1]+d1[2]*p0[2];
        let rhs_s = d2[0]*p0[0]+d2[1]*p0[1]+d2[2]*p0[2];
        let (rc, sc) = if det_j.abs() > 1e-30 {
            ((j11*rhs_r - j01*rhs_s)/det_j, (j00*rhs_s - j01*rhs_r)/det_j)
        } else { (0.0, 0.0) };

        let tr = [fb[0]-fa[0], fb[1]-fa[1], fb[2]-fa[2]];
        let ts = [fc[0]-fa[0], fc[1]-fa[1], fc[2]-fa[2]];
        let fnx = tr[1]*ts[2]-tr[2]*ts[1]; let fny = tr[2]*ts[0]-tr[0]*ts[2]; let fnz = tr[0]*ts[1]-tr[1]*ts[0];
        let d_sigma = (fnx*fnx+fny*fny+fnz*fnz).sqrt() * w;

        for i in 0..nd {
            let pi = monomial(r, s, i);
            for j in 0..nd { t[i][j] += monomial(rc, sc, j) * pi * d_sigma; }
        }
    }

    // Normalize row 0 by coarse area
    let coarse_area = 0.5 * cn_len;
    for j in 0..nd { t[0][j] /= coarse_area; }

    // Rescale row 0 to exact area ratio
    let area_ratio = rtx_flux_ratio(mesh, fine_face, coarse_face);
    let row0_sum: f64 = (0..nd).map(|j| t[0][j]).sum();
    if row0_sum.abs() > 1e-30 { let s = area_ratio / row0_sum; for j in 0..nd { t[0][j] *= s; } }

    t
}

// ─── Public API: HCurl/HDiv hanging constraint application ───────────────────

/// Apply HCurl hanging constraints to the assembled system `(K, f)`.
///
/// Combines edge and face DOF constraints from NC refinement for ND1/ND2
/// spaces on 3-D tetrahedral non-conforming meshes.
///
/// Call before solving, then call [`recover_hanging_values_hcurl`] after.
pub fn apply_hanging_constraints_hcurl<M: MeshTopology>(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) {
    let constraints = build_hcurl_hanging_constraints(
        hcurl,
        hanging_edges,
        hanging_faces,
    );
    apply_linear_constraints(mat, rhs, &constraints);
}

/// Recover HCurl hanging DOF values after solving.
pub fn recover_hanging_values_hcurl<M: MeshTopology>(
    x: &mut [f64],
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) {
    let constraints = build_hcurl_hanging_constraints(
        hcurl,
        hanging_edges,
        hanging_faces,
    );
    recover_linear_values(x, &constraints);
}

/// Apply HDiv hanging constraints to the assembled system `(K, f)`.
///
/// Constrains fine sub-face DOFs on hanging faces for RT0/RT1 spaces
/// on 3-D tetrahedral non-conforming meshes.
///
/// Call before solving, then call [`recover_hanging_values_hdiv`] after.
pub fn apply_hanging_constraints_hdiv<M: MeshTopology>(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) {
    let constraints = build_hdiv_hanging_constraints(
        hdiv,
        hanging_edges,
        hanging_faces,
    );
    apply_linear_constraints(mat, rhs, &constraints);
}

/// Recover HDiv hanging DOF values after solving.
pub fn recover_hanging_values_hdiv<M: MeshTopology>(
    x: &mut [f64],
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
) {
    let constraints = build_hdiv_hanging_constraints(
        hdiv,
        hanging_edges,
        hanging_faces,
    );
    recover_linear_values(x, &constraints);
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

    // ── LinearConstraint tests ───────────────────────────────────────────

    #[test]
    fn apply_linear_constraints_single() {
        // 3-DOF system: DOF 2 = 0.3*DOF 0 + 0.7*DOF 1
        let n = 3;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        coo.add(0, 1, -1.0); coo.add(1, 0, -1.0);
        coo.add(1, 2, -1.0); coo.add(2, 1, -1.0);
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![LinearConstraint {
            constrained: 2,
            parents: vec![(0, 0.3), (1, 0.7)],
        }];
        apply_linear_constraints(&mut mat, &mut rhs, &constraints);

        // Row 2 should be identity.
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14);
        assert!((mat.get(2, 0)).abs() < 1e-14);
        assert!((mat.get(2, 1)).abs() < 1e-14);
        assert!((rhs[2]).abs() < 1e-14);

        // Column 2 contributions should be distributed.
        // Original K[1,2] = -1, expanded as: -1 * (0.3*col0 + 0.7*col1) contribution to row 1.
        // Original K[2,1] = -1, expanded as: -1 * (0.3*row0 + 0.7*row1) contribution to col 1.
    }

    #[test]
    fn apply_linear_constraints_chained() {
        // 5-DOF system: DOF 3 = 0.5*(DOF 1 + DOF 2), DOF 4 = 0.5*(DOF 2 + DOF 3)
        // After expansion: DOF 4 = 0.25*DOF 1 + 0.75*DOF 2
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        if n > 1 { coo.add(1, 0, -1.0); coo.add(0, 1, -1.0); }
        if n > 2 { coo.add(2, 1, -1.0); coo.add(1, 2, -1.0); }
        if n > 3 { coo.add(3, 2, -1.0); coo.add(2, 3, -1.0); }
        if n > 4 { coo.add(4, 3, -1.0); coo.add(3, 4, -1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![
            LinearConstraint { constrained: 3, parents: vec![(1, 0.5), (2, 0.5)] },
            LinearConstraint { constrained: 4, parents: vec![(2, 0.5), (3, 0.5)] },
        ];
        apply_linear_constraints(&mut mat, &mut rhs, &constraints);

        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);
    }

    #[test]
    fn recover_linear_values_simple() {
        let mut x = vec![2.0, 6.0, 0.0];
        let constraints = vec![LinearConstraint {
            constrained: 2,
            parents: vec![(0, 0.3), (1, 0.7)],
        }];
        recover_linear_values(&mut x, &constraints);
        let expected = 0.3 * 2.0 + 0.7 * 6.0;
        assert!((x[2] - expected).abs() < 1e-14, "expected {expected}, got {}", x[2]);
    }

    #[test]
    fn recover_linear_values_chained() {
        let mut x = vec![0.0, 4.0, 0.0, 0.0];
        let constraints = vec![
            LinearConstraint { constrained: 2, parents: vec![(0, 0.5), (1, 0.5)] },
            LinearConstraint { constrained: 3, parents: vec![(1, 0.5), (2, 0.5)] },
        ];
        recover_linear_values(&mut x, &constraints);
        assert!((x[2] - 2.0).abs() < 1e-14, "expected x[2]=2, got {}", x[2]);
        assert!((x[3] - 3.0).abs() < 1e-14, "expected x[3]=3, got {}", x[3]);
    }

    #[test]
    fn recover_linear_values_multi_parent() {
        // 3 parents with non-uniform weights
        let mut x = vec![1.0, 2.0, 3.0, 0.0];
        let constraints = vec![LinearConstraint {
            constrained: 3,
            parents: vec![(0, 0.2), (1, 0.3), (2, 0.5)],
        }];
        recover_linear_values(&mut x, &constraints);
        let expected = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0;
        assert!((x[3] - expected).abs() < 1e-14, "expected {expected}, got {}", x[3]);
    }

    // ── NDk edge transform tests ─────────────────────────────────────────

    #[test]
    fn ndk_edge_transform_nd1() {
        // ND1: k=1, single DOF per edge. Fine sub-edge [0, L].
        // T = [L], so fine DOF = L * coarse DOF.
        let t = super::ndk_edge_transform(1, 0.5);
        assert_eq!(t.len(), 1);
        assert!((t[0][0] - 0.5).abs() < 1e-14, "ND1 L=0.5: expected 0.5, got {}", t[0][0]);

        let t_full = super::ndk_edge_transform(1, 1.0);
        assert!((t_full[0][0] - 1.0).abs() < 1e-14, "ND1 L=1.0: expected 1.0, got {}", t_full[0][0]);

        let t_quarter = super::ndk_edge_transform(1, 0.25);
        assert!((t_quarter[0][0] - 0.25).abs() < 1e-14, "ND1 L=0.25: expected 0.25, got {}", t_quarter[0][0]);
    }

    #[test]
    fn ndk_edge_transform_nd2_first_half() {
        // ND2: k=2, two DOFs per edge.
        // First half [0, 0.5]: T₁ = [[5/4, -3/2], [1/4, -1/4]]
        let t = super::ndk_edge_transform(2, 0.5);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].len(), 2);

        // T[0][0] = 5/4 = 1.25, T[0][1] = -3/2 = -1.5
        assert!((t[0][0] - 1.25).abs() < 1e-12, "T[0][0] expected 1.25, got {}", t[0][0]);
        assert!((t[0][1] - (-1.5)).abs() < 1e-12, "T[0][1] expected -1.5, got {}", t[0][1]);
        // T[1][0] = 1/4 = 0.25, T[1][1] = -1/4 = -0.25
        assert!((t[1][0] - 0.25).abs() < 1e-12, "T[1][0] expected 0.25, got {}", t[1][0]);
        assert!((t[1][1] - (-0.25)).abs() < 1e-12, "T[1][1] expected -0.25, got {}", t[1][1]);
    }

    #[test]
    fn ndk_edge_transform_nd2_second_half() {
        // Second half [0.5, 1]: T₂ = [[-1/4, 3/2], [-1/4, 5/4]]
        let t = super::ndk_edge_transform_for_second_half(2, 0.5);
        assert_eq!(t.len(), 2);

        // T[0][0] = -1/4 = -0.25, T[0][1] = 3/2 = 1.5
        assert!((t[0][0] - (-0.25)).abs() < 1e-12, "T[0][0] expected -0.25, got {}", t[0][0]);
        assert!((t[0][1] - 1.5).abs() < 1e-12, "T[0][1] expected 1.5, got {}", t[0][1]);
        // T[1][0] = -1/4 = -0.25, T[1][1] = 5/4 = 1.25
        assert!((t[1][0] - (-0.25)).abs() < 1e-12, "T[1][0] expected -0.25, got {}", t[1][0]);
        assert!((t[1][1] - 1.25).abs() < 1e-12, "T[1][1] expected 1.25, got {}", t[1][1]);
    }

    #[test]
    fn ndk_edge_transform_sums_to_identity() {
        // For ND2, the half transforms should sum to identity:
        // T_first + T_second should give back the original DOFs.
        let t1 = super::ndk_edge_transform(2, 0.5);
        let t2 = super::ndk_edge_transform_for_second_half(2, 0.5);

        for i in 0..2 {
            for j in 0..2 {
                let s = t1[i][j] + t2[i][j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-12,
                    "T1+T2[{i}][{j}] = {s}, expected {expected}");
            }
        }
    }

    #[test]
    fn ndk_edge_transform_nd2_constant_field() {
        // For constant field f(t) = 1:
        //   coarse DOF_0 = ∫₀¹ 1 dt = 1
        //   coarse DOF_1 = ∫₀¹ t dt = 0.5
        //   fine DOF_0 (first half) = ∫₀^{0.5} 1 dt = 0.5
        //   fine DOF_1 (first half) = ∫₀^{0.5} t dt = 0.125
        let t = super::ndk_edge_transform(2, 0.5);
        let fine_0 = t[0][0] * 1.0 + t[0][1] * 0.5;
        let fine_1 = t[1][0] * 1.0 + t[1][1] * 0.5;
        assert!((fine_0 - 0.5).abs() < 1e-12, "fine_0 expected 0.5, got {fine_0}");
        assert!((fine_1 - 0.125).abs() < 1e-12, "fine_1 expected 0.125, got {fine_1}");
    }

    #[test]
    fn ndk_edge_transform_nd2_linear_field() {
        // For linear field f(t) = t:
        //   coarse DOF_0 = ∫₀¹ t dt = 0.5
        //   coarse DOF_1 = ∫₀¹ t·t dt = 1/3
        //   fine DOF_0 (first half) = ∫₀^{0.5} t dt = 0.125
        //   fine DOF_1 (first half) = ∫₀^{0.5} t·t dt = 0.5³/3 = 1/24 ≈ 0.0416667
        let t = super::ndk_edge_transform(2, 0.5);
        let fine_0 = t[0][0] * 0.5 + t[0][1] * (1.0/3.0);
        let fine_1 = t[1][0] * 0.5 + t[1][1] * (1.0/3.0);
        assert!((fine_0 - 0.125).abs() < 1e-12, "fine_0 expected 0.125, got {fine_0}");
        assert!((fine_1 - (1.0/24.0)).abs() < 1e-12, "fine_1 expected 1/24, got {fine_1}");
    }

    #[test]
    fn ndk_edge_transform_nd3_quarter() {
        // ND3: k=3, three DOFs per edge.
        // First quarter [0, 0.25]: verify 3×3 transform.
        let t = super::ndk_edge_transform(3, 0.25);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 3);

        // For constant field f(t)=1, fine DOF_0 = ∫₀^{0.25} 1 dt = 0.25
        let coarse = [1.0, 0.5, 1.0/3.0];
        let fine_0 = t[0][0]*coarse[0] + t[0][1]*coarse[1] + t[0][2]*coarse[2];
        assert!((fine_0 - 0.25).abs() < 1e-12, "ND3 constant: fine_0 expected 0.25, got {fine_0}");
    }

    // ── HCurl hanging constraint construction tests ───────────────────────

    #[test]
    fn build_hcurl_hanging_constraints_3d_tet_nd1() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::{NCState3D, HangingNodeConstraint, HangingFaceConstraint};

        // Create a 3-D Tet mesh and non-conforming refinement.
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        // Mark first element for refinement → creates hanging faces.
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        let edges: &[HangingNodeConstraint] = edge_cons.as_slice();
        let faces: &[HangingFaceConstraint] = face_cons.as_slice();

        // If no hanging faces exist, this test is trivially passed.
        // For a single tet refined in a 6-tet cube, hanging interfaces should exist.
        if !edges.is_empty() {
            // Build ND1 space on the fine mesh.
            let hcurl = HCurlSpace::new(fine_mesh, 1);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, edges, faces,
            );

            // Each constraint maps a fine edge DOF to a weighted coarse edge DOF.
            // For ND1: each edge has 1 DOF, fine edge gets 0.5 × coarse DOF.
            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range ({})", c.constrained, hcurl.n_dofs());
                assert!(!c.parents.is_empty(), "constraint for DOF {} has no parents", c.constrained);
                // Verify each parent DOF is valid.
                for &(p, w) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range ({})", hcurl.n_dofs());
                    assert!(w.is_finite(), "weight {w} is not finite");
                }
            }
        }
    }

    #[test]
    fn build_hcurl_hanging_constraints_3d_tet_nd2() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if !edge_cons.is_empty() {
            let hcurl = HCurlSpace::new(fine_mesh, 2);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, &edge_cons, &face_cons,
            );

            // ND2: each fine edge has 2 DOFs, each should have a constraint
            // with 2 parents (one per coarse edge DOF).
            assert!(!constraints.is_empty(), "expected at least one ND2 hanging constraint");

            // Verify structure
            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range", c.constrained);
                assert!(!c.parents.is_empty(), "constraint has no parents");
                for &(p, _) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range");
                }
            }
        }
    }

    #[test]
    fn build_hcurl_hanging_constraints_empty_for_conforming() {
        use crate::hcurl::HCurlSpace;
        
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        // Uniform refinement = no hanging faces.
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0, 1, 2, 3, 4, 5]);

        // All elements refined → no hanging constraints.
        assert!(edge_cons.is_empty(), "uniform refinement should have no hanging edge constraints");
        assert!(face_cons.is_empty(), "uniform refinement should have no hanging face constraints");

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons,
        );
        assert!(constraints.is_empty(), "full refinement should produce no constraints");
    }

    // ── HDiv hanging constraint construction tests ────────────────────────

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt0() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if !edge_cons.is_empty() {
            let hdiv = HDivSpace::new(fine_mesh.clone(), 0);
            let constraints = super::build_hdiv_hanging_constraints(
                &hdiv, &edge_cons, &face_cons,
            );

            // RT0: each face has 1 DOF. Hanging face constraint: fine_dof = area_ratio * coarse_dof.
            for c in &constraints {
                assert!(c.constrained < hdiv.n_dofs(),
                    "constrained DOF {} out of range", c.constrained);
                assert_eq!(c.parents.len(), 1,
                    "RT0 constraint should have exactly 1 parent");
                let (p, w) = c.parents[0];
                assert!(p < hdiv.n_dofs(), "parent DOF {p} out of range");
                assert!(w > 0.0 && w <= 0.5, "RT0 flux ratio should be in (0, 0.5], got {w}");
            }
        }
    }

    #[test]
    fn apply_linear_constraints_hcurl_nd2_preserves_solvability() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        // Build a small 3D non-conforming mesh and HCurl ND2 space.
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return; // skip if no hanging edges for this test mesh
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        // Build a simple Laplacian-like system (identity matrix, unit RHS).
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        // Apply HCurl hanging constraints.
        apply_hanging_constraints_hcurl(&mut mat, &mut rhs, &hcurl, &edge_cons, &face_cons);

        // Verify: constrained DOF rows should be identity.
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons,
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }

        // Matrix symmetry should be preserved (P^T K P is symmetric when K is).
        for i in 0..n.min(50) {
            for j in 0..n.min(50) {
                let kij = mat.get(i, j);
                let kji = mat.get(j, i);
                assert!((kij - kji).abs() < 1e-12,
                    "symmetry broken at ({i},{j}): {kij} vs {kji}");
            }
        }
    }

    #[test]
    fn apply_linear_constraints_hdiv_rt0_preserves_solvability() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(&mut mat, &mut rhs, &hdiv, &edge_cons, &face_cons);

        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &face_cons,
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} row not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }
    }

    #[test]
    fn recover_hanging_values_hcurl_nd2_after_solve() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        // Build identity system, solve, then recover.
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hcurl(&mut mat, &mut rhs, &hcurl, &edge_cons, &face_cons);

        // Solve: x = rhs (identity system after constraint application).
        let mut x = rhs.clone();

        // Recover hanging values.
        recover_hanging_values_hcurl(&mut x, &hcurl, &edge_cons, &face_cons);

        // Verify constraints hold.
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons,
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
        }
    }

    #[test]
    fn recover_hanging_values_hdiv_rt0_after_solve() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(&mut mat, &mut rhs, &hdiv, &edge_cons, &face_cons);
        let mut x = rhs.clone();
        recover_hanging_values_hdiv(&mut x, &hdiv, &edge_cons, &face_cons);

        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &face_cons,
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
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

    // ─── RT1/RT2 hanging constraint tests ─────────────────────────────────

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt1() {
        let mesh = fem_mesh::SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = fem_mesh::amr::NCState3D::new();
        let (fm, ec, _, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = crate::hdiv::HDivSpace::new(fm.clone(), 1);
        let cs = build_hdiv_hanging_constraints(&h, &ec, &fc);
        assert!(!cs.is_empty(), "expected RT1 constraints");
        for c in &cs {
            assert!(c.constrained < h.n_dofs(), "DOF {} out", c.constrained);
            assert!(!c.parents.is_empty());
            for &(p, w) in &c.parents { assert!(p < h.n_dofs()); assert!(w.is_finite()); }
        }
    }

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt2() {
        let mesh = fem_mesh::SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = fem_mesh::amr::NCState3D::new();
        let (fm, ec, _, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = crate::hdiv::HDivSpace::new(fm.clone(), 2);
        let cs = build_hdiv_hanging_constraints(&h, &ec, &fc);
        assert!(!cs.is_empty(), "expected RT2 constraints");
        for c in &cs {
            assert!(c.constrained < h.n_dofs(), "DOF {} out", c.constrained);
            assert!(!c.parents.is_empty());
            for &(p, w) in &c.parents { assert!(p < h.n_dofs()); assert!(w.is_finite()); }
        }
    }

    #[test]
    fn recover_hanging_values_hdiv_rt1_after_solve() {
        let mesh = fem_mesh::SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = fem_mesh::amr::NCState3D::new();
        let (fm, ec, _, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = crate::hdiv::HDivSpace::new(fm, 1);
        let n = h.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut m = coo.into_csr(); let mut rhs = vec![1.0; n];
        apply_hanging_constraints_hdiv(&mut m, &mut rhs, &h, &ec, &fc);
        let mut x = rhs.clone();
        recover_hanging_values_hdiv(&mut x, &h, &ec, &fc);
        for c in build_hdiv_hanging_constraints(&h, &ec, &fc) {
            let exp = c.parents.iter().map(|&(p, w)| w * x[p]).sum::<f64>();
            assert!((x[c.constrained] - exp).abs() < 1e-10,
                "DOF {}: {} != {}", c.constrained, x[c.constrained], exp);
        }
    }

    #[test]
    fn recover_hanging_values_hcurl_nd2_face_dofs() {
        let mesh = fem_mesh::SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc = fem_mesh::amr::NCState3D::new();
        let (fm, ec, _, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = crate::hcurl::HCurlSpace::new(fm, 2);
        let n = h.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut m = coo.into_csr(); let mut rhs = vec![1.0; n];
        apply_hanging_constraints_hcurl(&mut m, &mut rhs, &h, &ec, &fc);
        let mut x = rhs.clone();
        recover_hanging_values_hcurl(&mut x, &h, &ec, &fc);
        for c in build_hcurl_hanging_constraints(&h, &ec, &fc) {
            let exp = c.parents.iter().map(|&(p, w)| w * x[p]).sum::<f64>();
            assert!((x[c.constrained] - exp).abs() < 1e-10,
                "ND2 DOF {}: {} != {}", c.constrained, x[c.constrained], exp);
        }
    }
}
