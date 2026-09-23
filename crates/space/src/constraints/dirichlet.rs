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
/// BCs the RHS is adjusted by the **true column entries**: `rhs[j] -=
/// A[j,dof]·val` (MFEM `EliminateRowCol`, sparsemat.cpp:1959).  This is valid
/// for numerically nonsymmetric systems too (e.g. saddle-point
/// `[A −Bᵀ; B 0]`, whose coupling blocks are antisymmetric — D409), requiring
/// only structural symmetry of the sparsity pattern.
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
        // Delegates to the library entry — the former hand-rolled copy made
        // the same "row value as column reaction" mistake as D409: reactions
        // must be read from the true column entries A[j,dof].
        mat.apply_dirichlet_symmetric(dof as usize, val, rhs);
    }
}

/// MFEM `BilinearForm::FormLinearSystem` — full-contract variant of
/// [`form_linear_system`] (D641).
///
/// Takes the **full projected solution** `x` (a value at *every* dof, e.g. an
/// interpolation of non-homogeneous boundary data — MFEM's
/// `x.ProjectCoefficient(E)`), then performs the conforming, non-hybridized
/// `FormLinearSystem` sequence exactly:
///
/// 1. `FormSystemMatrix`: eliminate the rows and columns of every constrained
///    dof with `DIAG_KEEP` (diagonal kept — `BilinearForm::diag_policy`);
/// 2. `EliminateVDofsInRHS` (`bilinearform.cpp:1239`):
///    `b -= mat_e · x` (the eliminated off-diagonal row/column entries times
///    the full `x`), followed by `PartMult` which **assigns**
///    `b[r] = (mat · x)[r] = A_rr · x[r]` on the constrained rows — so the
///    interior values of `x` never reach the free rows of the RHS and are
///    overwritten on the constrained rows;
/// 3. `copy_interior = false` (MFEM's default) zeroes `x` at the free dofs,
///    keeping the prescribed values on the constrained dofs — the initial
///    guess `X` handed to the solver; `copy_interior = true` keeps the full
///    projection as the initial guess.
///
/// The net RHS equals the one produced by [`apply_dirichlet`] with
/// `values = x[r]` at the constrained dofs (the `mat_e · x` row terms are
/// overwritten by the `PartMult` assignment on those very rows), which the
/// caller can rely on; the entry point exists to make the MFEM contract —
/// *pass the full projection, get back the solver-ready `X`* — expressible
/// without hand-extracting boundary values.
///
/// # Panics
/// Panics if `constrained_dofs` contains an out-of-range dof.
pub fn form_linear_system_vdofs(
    mat:              &mut CsrMatrix<f64>,
    rhs:              &mut [f64],
    x:                &mut [f64],
    constrained_dofs: &[DofId],
    copy_interior:    bool,
) {
    // 1+2. DIAG_KEEP elimination with the essential values of the full x
    // (column reactions `b[j] -= A[j,r]·x[r]`, `b[r] = A_rr·x[r]`).
    for &dof in constrained_dofs {
        let r = dof as usize;
        let value = x[r];
        mat.apply_dirichlet_keep_diag(r, value, rhs);
    }
    // 3. `PartMult` assignment on the constrained rows: after the DIAG_KEEP
    // elimination each such row holds only its diagonal, so `(mat·x)[r] =
    // A_rr·x[r]` — recompute it from the (kept) diagonal for exactness.
    for &dof in constrained_dofs {
        let r = dof as usize;
        let pos = mat.find_entry(r, r);
        if let Some(k) = pos {
            rhs[r] = mat.values[k] * x[r];
        }
    }
    // 4. `X.SetSubVectorComplement(ess_tdof_list, 0.0)`.
    if !copy_interior {
        let mut ess = vec![false; x.len()];
        for &dof in constrained_dofs {
            ess[dof as usize] = true;
        }
        for (i, is_ess) in ess.iter().enumerate() {
            if !is_ess {
                x[i] = 0.0;
            }
        }
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
            if mesh.dim() == 3 {
                if nodes.len() == 3 {
                    boundary_faces_3d.insert(FaceKey::new(nodes[0], nodes[1], nodes[2]));
                } else if nodes.len() == 4 {
                    // Quad boundary face: look up its interior DOFs through the
                    // quad face table (hex mesh boundary faces).
                    if let Some(face_dofs) = dm.quad_face_pk_map.get(&QuadFaceKey::new(
                        nodes[0], nodes[1], nodes[2], nodes[3],
                    )) {
                        for &dof in face_dofs {
                            dof_set.insert(dof);
                        }
                    }
                }
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

    // Collect face-interior DOFs on tagged boundary faces (3D).
    if mesh.dim() == 3 && space.order() >= 2 {
        for f in 0..mesh.n_boundary_faces() as u32 {
            if !tags.contains(&mesh.face_tag(f)) {
                continue;
            }
            let nodes = mesh.face_nodes(f);
            match nodes.len() {
                // Hex quad face (hex NDk, k >= 2).
                4 => {
                    let key = QuadFaceKey::new(nodes[0], nodes[1], nodes[2], nodes[3]);
                    if let Some(mut fdofs) = space.quad_face_dofs(key) {
                        out.append(&mut fdofs);
                    }
                }
                // Triangular face interior DOFs (NDk, k >= 2): every NDk
                // triangular face block carries `k(k-1)` dofs — `k(k-1)/2`
                // point sites x 2 tangent slots — for tets (the
                // `face_anchor`), prisms (MFEM `ND_WedgeElement`) and
                // pyramids (the MFEM Fuentes pyramid) alike, so the count
                // comes from the order, not from the tet-only anchor (D525).
                3 => {
                    let key = FaceKey::new(nodes[0], nodes[1], nodes[2]);
                    if let Some(first) = space.face_dof(key) {
                        let k = space.order() as DofId;
                        out.extend(first..first + k * (k - 1));
                    }
                }
                _ => {}
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
            let dofs = if dim == 2 {
                if nodes.len() >= 2 {
                    // D368: an RT edge carries `order + 1` dofs (MFEM's
                    // `GetBoundaryTrueDofs` essential-constrains the whole
                    // block); `edge_face_dof` alone exposed only the first.
                    space.edge_face_dofs(EdgeKey::new(nodes[0], nodes[1]))
                } else {
                    None
                }
            } else {
                if nodes.len() >= 3 {
                    if nodes.len() == 3 {
                        // D377: an order-k RT triangular face carries
                        // `(k+1)(k+2)/2` dofs (MFEM's `GetBoundaryTrueDofs`
                        // essential-constrains the whole block);
                        // `tri_face_dof` alone exposed only the first.
                        space.face_dofs(FaceKey::new(nodes[0], nodes[1], nodes[2]))
                    } else {
                        // Quad face: the HDivSpace quad DOF key uses the
                        // first 3 vertices of the element-face ring, but the
                        // boundary ring may start at a different vertex —
                        // try all 4 triplets of the quad.  D377: the hit
                        // returns the face's whole `(k+1)^2`-sized block.
                        let mut found = None;
                        for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
                            if let Some(block) =
                                space.face_dofs(FaceKey::new(nodes[i], nodes[j], nodes[k]))
                            {
                                found = Some(block);
                                break;
                            }
                        }
                        found
                    }
                } else {
                    None
                }
            };
            if let Some(mut block) = dofs {
                out.append(&mut block);
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod vdofs_tests {
    use super::*;

    /// `[[ 4., -1.,  0.], [-1.,  3., -1.], [ 0., -1.,  2.]]` (SPD).
    fn mat3() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        for (i, j, v) in [
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 3.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ] {
            coo.add(i, j, v);
        }
        coo.into_csr()
    }

    /// D641: the full-contract entry reproduces MFEM's `FormLinearSystem`
    /// exactly — RHS reactions use only the constrained values of the full
    /// projection (`mat_e·x` row terms are overwritten by the `PartMult`
    /// assignment), `b[r] = A_rr·x[r]` on the constrained rows, and
    /// `copy_interior = false` zeroes the interior of `x`.
    #[test]
    fn form_linear_system_vdofs_matches_mfem_semantics() {
        let b = [1.0, 2.0, 3.0];
        // Full projection with NONZERO interior: if the interior ever reached
        // the free RHS rows, row 1 would pick up a spurious `-1.0·x[2]` term.
        let mut x = vec![7.0, -4.0, 5.0];
        let mut mat = mat3();
        let mut rhs = b.to_vec();
        form_linear_system_vdofs(&mut mat, &mut rhs, &mut x, &[0, 2], false);

        // Constrained rows: assignment semantics from the kept diagonal.
        assert_eq!(rhs[0], 4.0 * 7.0);
        assert_eq!(rhs[2], 2.0 * 5.0);
        // Free row: b - A[1,0]·x[0] - A[1,2]·x[2] — no interior-free coupling.
        assert_eq!(rhs[1], 2.0 - (-1.0) * 7.0 - (-1.0) * 5.0);
        // Matrix: DIAG_KEEP on rows/cols 0 and 2 (structural zeros retained).
        let mut kept: Vec<(usize, f64)> = Vec::new();
        for i in 0..3usize {
            for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                if mat.values[k] != 0.0 {
                    kept.push((mat.col_idx[k] as usize, mat.values[k]));
                }
            }
        }
        assert_eq!(kept, vec![(0, 4.0), (1, 3.0), (2, 2.0)]);
        // copy_interior = false: interior zeroed, prescribed values kept.
        assert_eq!(x, vec![7.0, 0.0, 5.0]);
    }

    /// `copy_interior = true` keeps the full projection as the initial guess.
    #[test]
    fn form_linear_system_vdofs_copy_interior_keeps_projection() {
        let mut x = vec![7.0, -4.0, 5.0];
        let mut mat = mat3();
        let mut rhs = vec![1.0, 2.0, 3.0];
        form_linear_system_vdofs(&mut mat, &mut rhs, &mut x, &[0, 2], true);
        assert_eq!(x, vec![7.0, -4.0, 5.0]);
        // Same RHS as the copy_interior = false run.
        assert_eq!(rhs[1], 2.0 - (-1.0) * 7.0 - (-1.0) * 5.0);
    }

    /// For conforming systems the new entry is numerically equivalent to the
    /// historical [`form_linear_system`] with `values = x[ess]` — the net
    /// `EliminateVDofsInRHS` effect is the sequential `DIAG_KEEP` elimination.
    #[test]
    fn form_linear_system_vdofs_equivalent_to_form_linear_system() {
        let b = [1.0, -2.0, 3.0, 0.5];
        let mk = || {
            let mut coo = CooMatrix::<f64>::new(4, 4);
            let v: [(usize, usize); 10] = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
            ];
            for (k, &(i, j)) in v.iter().enumerate() {
                coo.add(i as usize, j as usize, 4.0 - k as f64 * 0.25);
            }
            coo.into_csr()
        };
        let ess = [0u32, 3];
        let x_full = vec![1.5, 9.0, -7.0, -2.5];

        let mut mat_a = mk();
        let mut rhs_a = b.to_vec();
        let mut x_a = vec![0.0; 4];
        let bc: Vec<f64> = ess.iter().map(|&d| x_full[d as usize]).collect();
        form_linear_system(&mut mat_a, &mut rhs_a, &mut x_a, &ess, &bc);

        let mut mat_b = mk();
        let mut rhs_b = b.to_vec();
        let mut x_b = x_full.clone();
        form_linear_system_vdofs(&mut mat_b, &mut rhs_b, &mut x_b, &ess, false);

        assert_eq!(rhs_a, rhs_b);
        assert_eq!(mat_a.values.to_vec(), mat_b.values.to_vec());
        assert_eq!(x_a, x_b);
    }
}
