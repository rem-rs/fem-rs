use std::collections::HashSet;

use fem_core::types::DofId;
use fem_linalg::{CooMatrix, CsrMatrix};

use crate::dof_manager::{DofManager, EdgeKey, FaceKey, QuadFaceKey};
use crate::hcurl::HCurlSpace;
use crate::hdiv::HDivSpace;

/// Apply Dirichlet boundary conditions to the assembled system `(K, f)`.
///
/// For each DOF in `constrained_dofs`:
/// 1. Eliminate both row and column (symmetric elimination).
/// 2. Set the diagonal to 1.
/// 3. Set `rhs[dof] = value[i]`.
///
/// For zero-valued BCs (the common case) the RHS is unchanged; for non-zero
/// BCs the RHS is adjusted to account for eliminated column contributions.
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
        // MFEM 4.9 BilinearForm::FormLinearSystem defaults to diag_policy =
        // DIAG_KEEP (bilinearform.hpp: diag_policy = DIAG_KEEP): the diagonal
        // A[i,i] is KEPT and rhs[i] = A[i,i]·val.  DIAG_ONE (diagonal = 1)
        // gives the same solution in exact arithmetic but a different matrix,
        // hence a different PCG/GS history (ex33: 2× iteration count).
        mat.apply_dirichlet_keep_diag(dof as usize, val, rhs);
    }
}

/// Apply Dirichlet BCs with MFEM's `DIAG_ONE` policy: row/column elimination
/// with the diagonal set to 1 and `rhs[i] = val[i]` (instead of keeping the
/// diagonal like [`apply_dirichlet`]).
///
/// MFEM's plain `BilinearForm::EliminateEssentialBC(bdr)` (without a
/// `diag_policy` argument) uses the form's `diag_policy` member, which for
/// `BilinearForm` defaults to `DIAG_ONE` (ex8's S0 — ex8.cpp calls
/// `S0->EliminateEssentialBC(ess_bdr)` and the resulting diagonal entries
/// are 1).
pub fn apply_dirichlet_diag_one(
    mat:              &mut CsrMatrix<f64>,
    rhs:              &mut [f64],
    constrained_dofs: &[DofId],
    values:           &[f64],
) {
    assert_eq!(constrained_dofs.len(), values.len(),
        "constrained_dofs and values must have the same length");
    for (&dof, &val) in constrained_dofs.iter().zip(values.iter()) {
        // Row zeroing + column elimination + diagonal = 1 (DIAG_ONE).
        let row = dof as usize;
        let start = mat.row_ptr[row];
        let end = mat.row_ptr[row + 1];
        for k in start..end {
            let other = mat.col_idx[k] as usize;
            if other != row {
                // rhs[other] -= A[other,row]·val (symmetric: A[row,other]).
                let a_ij = mat.values[k];
                if a_ij != 0.0 {
                    rhs[other] -= a_ij * val;
                    if let Some(pos) = mat.find_entry(other, row) {
                        mat.values[pos] = 0.0;
                    }
                }
            }
        }
        for k in start..end {
            if mat.col_idx[k] as usize == row {
                mat.values[k] = 1.0;
            } else {
                mat.values[k] = 0.0;
            }
        }
        rhs[row] = val;
    }
}

/// Apply Dirichlet BCs following MFEM's `FormLinearSystem` convention.
///
/// Modifies the matrix and RHS in-place so that constrained DOFs are set
/// to their prescribed values, keeping the full N×N system (unlike
/// [`eliminate_dirichlet`] which produces a reduced system).
///
/// After this call, for each constrained DOF `i`:
/// - `mat[i,i] = 1`, `mat[i,j] = 0` for all `j ≠ i`
/// - `rhs[i] = val[i]`
/// - `rhs[other] -= mat[other, i] * val[i]` (column elimination, matching
///   MFEM's symmetric elimination pass)
///
/// The caller should then solve the full N×N system (PCG / AMS / …) and
/// read `x` directly — no `expand_from_reduced` needed.
///
/// # Panics
/// Panics if `constrained_dofs.len() != values.len()` or `x.len() < n_full`.
pub fn form_linear_system(
    mat:              &mut CsrMatrix<f64>,
    rhs:              &mut [f64],
    x:                &mut [f64],
    constrained_dofs: &[DofId],
    values:           &[f64],
) {
    assert_eq!(constrained_dofs.len(), values.len(),
        "constrained_dofs and values must have the same length");
    // 1. Initialise solution vector with BC values.
    for (&dof, &val) in constrained_dofs.iter().zip(values.iter()) {
        x[dof as usize] = val;
    }
    // 2. Column elimination + row elimination (MFEM FormLinearSystem step).
    apply_dirichlet(mat, rhs, constrained_dofs, values);
}

/// Build a reduced system by eliminating Dirichlet DOFs from the matrix.
///
/// Returns `(reduced_mat, reduced_rhs, free_map, constrained_map)` where:
/// - `reduced_mat` is `m×m` (m = n − |constrained|)
/// - `reduced_rhs` has length `m`
/// - `free_map[i]` = original DOF index for reduced DOF `i`
/// - `constrained_map[j]` = original DOF index for constrained DOF `j`
///
/// The caller can solve `reduced_mat * x_red = reduced_rhs`, then expand
/// with [`expand_from_reduced`].
///
/// This produces **exactly the same linear system** that MFEM's
/// `eliminate_bc` creates, ensuring matching DOF counts and identical
/// numerical solutions at unconstrained nodes.
pub fn eliminate_dirichlet(
    mat:              &CsrMatrix<f64>,
    rhs:              &[f64],
    constrained_dofs: &[DofId],
    values:           &[f64],
) -> (CsrMatrix<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    assert_eq!(constrained_dofs.len(), values.len(),
        "constrained_dofs and values must have the same length");

    let n = mat.nrows;
    let constrained_set: HashSet<usize> = constrained_dofs.iter().map(|&d| d as usize).collect();

    // Build free → original mapping and constrained → original mapping
    let mut free_map: Vec<usize> = Vec::with_capacity(n - constrained_set.len());
    let mut constrained_map: Vec<usize> = constrained_dofs.iter().map(|&d| d as usize).collect();
    constrained_map.sort_unstable();

    // Map original → reduced index (-1 for constrained)
    let mut orig_to_red: Vec<isize> = vec![-1; n];
    for (red_idx, &orig_idx) in constrained_map.iter().enumerate() {
        orig_to_red[orig_idx] = -(red_idx as isize + 1); // negative = constrained
    }
    for (red_idx, orig_idx) in (0..n).filter(|i| !constrained_set.contains(i)).enumerate() {
        orig_to_red[orig_idx] = red_idx as isize;
        free_map.push(orig_idx);
    }

    let m = free_map.len();
    let mut coo = CooMatrix::<f64>::new(m, m);
    let mut reduced_rhs = vec![0.0_f64; m];

    // Build reduced matrix and RHS
    for &ri in &free_map {
        let r_red = orig_to_red[ri] as usize;
        let mut rhs_val = rhs[ri];

        for k in mat.row_ptr[ri]..mat.row_ptr[ri + 1] {
            let cj = mat.col_idx[k] as usize;
            let v = mat.values[k];

            if let Some(c_red) = orig_to_red.get(cj) {
                if *c_red >= 0 {
                    // Free column → add to reduced matrix
                    coo.add(r_red, *c_red as usize, v);
                } else {
                    // Constrained column → move to RHS: subtract K[i,c] * u[c]
                    let constrained_idx = (-*c_red - 1) as usize;
                    rhs_val -= v * values[constrained_idx];
                }
            }
        }
        reduced_rhs[r_red] = rhs_val;
    }

    let reduced_mat = coo.into_csr();
    (reduced_mat, reduced_rhs, free_map, constrained_map)
}

/// Expand a reduced solution back to the full vector.
///
/// `x_full[i] = x_red[free_idx(i)]` for free DOFs,
/// `x_full[c[j]] = values[j]` for constrained DOFs.
pub fn expand_from_reduced(
    x_red: &[f64],
    free_map: &[usize],
    constrained_map: &[usize],
    values: &[f64],
    n_full: usize,
) -> Vec<f64> {
    let mut x = vec![0.0_f64; n_full];
    for (&orig, &val) in constrained_map.iter().zip(values.iter()) {
        x[orig] = val;
    }
    for (&orig, &val) in free_map.iter().zip(x_red.iter()) {
        x[orig] = val;
    }
    x
}

/// Identify which DOFs lie on boundary faces with the given tag(s).
///
/// Return sorted global DOF indices for all boundary nodes (and, for any
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
            // Vertex DOFs: all boundary face nodes.  On NC meshes the global
            // vertex DOF ids follow MFEM's vertex-view order (phys_to_vertex_dof),
            // so a physical node id must NOT be used directly as the DOF id.
            for &node in nodes {
                let d = dm
                    .phys_to_vertex_dof
                    .get(&node)
                    .copied()
                    .unwrap_or(node as DofId);
                dof_set.insert(d);
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

/// Extract the edge DOFs on the perimeter loop of a set of boundary elements.
///
/// Given a set of boundary tags, this function finds the edges that lie on
/// the **boundary of the boundary region** — i.e., edges that belong to exactly
/// one boundary face in the selected set.  These are the "perimeter" edges
/// of the region.
///
/// This is useful for imposing boundary conditions on boundary edge DOFs
/// (e.g., for H(curl) problems where tangential components on the boundary
/// of a surface must be constrained).
///
/// Matches MFEM 4.10 `FiniteElementSpace::GetBoundaryLoopEdgeDofs`.
///
/// # Arguments
/// * `mesh`  — mesh providing boundary face data
/// * `dm`    — DOF manager for the space (provides edge-to-DOF mapping)
/// * `tags`  — boundary tags to select
///
/// # Returns
/// Sorted vector of edge DOF IDs on the perimeter loop.
///
/// # Example
/// ```rust,ignore
/// use fem_space::constraints::boundary_loop_edge_dofs;
///
/// // Get edge DOFs on the perimeter of boundary tags 1, 2
/// let edge_dofs = boundary_loop_edge_dofs(&mesh, &dm, &[1, 2]);
/// ```
pub fn boundary_loop_edge_dofs(
    mesh: &dyn fem_mesh::topology::MeshTopology,
    dm:   &DofManager,
    tags: &[i32],
) -> Vec<DofId> {
    use std::collections::HashMap;

    // Count how many times each edge appears in the selected boundary faces.
    // Edges that appear exactly once are on the perimeter.
    let mut edge_count: HashMap<EdgeKey, u32> = HashMap::new();

    for f in 0..mesh.n_boundary_faces() as u32 {
        if tags.contains(&mesh.face_tag(f)) {
            let nodes = mesh.face_nodes(f);
            if nodes.len() <= 2 {
                // 2D: boundary face is an edge (2 nodes) → 1 edge
                if nodes.len() == 2 {
                    let ek = EdgeKey::new(nodes[0], nodes[1]);
                    *edge_count.entry(ek).or_insert(0) += 1;
                }
            } else {
                // 3D: boundary face is a polygon (3+ nodes) → multiple edges
                for i in 0..nodes.len() {
                    let a = nodes[i];
                    let b = nodes[(i + 1) % nodes.len()];
                    let ek = EdgeKey::new(a, b);
                    *edge_count.entry(ek).or_insert(0) += 1;
                }
            }
        }
    }

    // Collect DOFs from edges that appear exactly once (perimeter edges)
    let mut dof_set: std::collections::HashSet<DofId> = std::collections::HashSet::new();

    for (ek, count) in &edge_count {
        if *count == 1 {
            // This edge is on the perimeter
            if let Some(edge_dofs) = dm.edge_pk_map.get(ek) {
                for &dof in edge_dofs {
                    dof_set.insert(dof);
                }
            }
            // Also check legacy edge_dof_map (P2) and edge_dof2_map (P3)
            if let Some(&dof) = dm.edge_dof_map.get(ek) {
                dof_set.insert(dof);
            }
            if let Some(&[d0, d1]) = dm.edge_dof2_map.get(ek) {
                dof_set.insert(d0);
                dof_set.insert(d1);
            }
        }
    }

    let mut out: Vec<DofId> = dof_set.into_iter().collect();
    out.sort_unstable();
    out
}

/// Convenience wrapper around [`boundary_dofs`] that returns `Vec<usize>` instead of `Vec<DofId>`.
///
/// This is the most common type needed for constraint matrix construction,
/// Dirichlet elimination, and LOBPCG essential BC handling.
///
/// # Arguments
/// * `mesh`  — mesh providing boundary face data
/// * `dm`    — DOF manager for the space
/// * `tags`  — boundary tags to select (e.g. `&[1, 2, 3, 4]` for all sides)
pub fn collect_essential_dofs(
    mesh: &dyn fem_mesh::topology::MeshTopology,
    dm:   &DofManager,
    tags: &[i32],
) -> Vec<usize> {
    boundary_dofs(mesh, dm, tags)
        .into_iter()
        .map(|d| d as usize)
        .collect()
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
                    if nodes.len() == 3 {
                        space.tri_face_dof(FaceKey::new(nodes[0], nodes[1], nodes[2]))
                    } else {
                        // Quad face: the HDivSpace quad DOF key uses the
                        // first 3 vertices of the element-face ring, but the
                        // boundary ring may start at a different vertex —
                        // try all 4 triplets of the quad.
                        let mut found = None;
                        for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
                            if let Some(d) =
                                space.tri_face_dof(FaceKey::new(nodes[i], nodes[j], nodes[k]))
                            {
                                found = Some(d);
                                break;
                            }
                        }
                        found
                    }
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
