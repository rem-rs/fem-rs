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
        mat.apply_dirichlet_symmetric(dof as usize, val, rhs);
    }
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
