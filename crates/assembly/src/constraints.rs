//! Normal (sliding) boundary constraints — 1:1 port of MFEM
//! `linalg/constraints.cpp::BuildNormalConstraints`.
//!
//! Builds the sparse constraint matrix `C` for a vector FE space such that
//! `C u = 0` restricts the *normal* displacement on the given boundary
//! attributes while leaving tangential motion free (sliding contact).
//!
//! # Semantics (matching MFEM)
//! - **One constraint row per (scalar DOF, attribute):** a DOF touched by
//!   faces of several constrained attributes gets one row per attribute (a
//!   *block constraint*).  E.g. a corner vertex on two constrained edges gets
//!   two linearly independent rows and is therefore fully pinned.
//! - **DOFs on a boundary face** are the face's vertex DOFs plus the
//!   mid-edge DOFs for order ≥ 2 (see [`boundary_face_dofs`]).
//! - **Row ordering:** scalar DOFs in ascending order; within a DOF, rows are
//!   sorted by ascending attribute (as in MFEM, where each block's rows are
//!   held in a `std::map` keyed by attribute).  The second return value is
//!   `lagrange_rowstarts` — the start index of each block constraint.
//! - **Normals** are the unscaled boundary-segment normals `(dy, −dx)`
//!   (MFEM `CalcOrtho`), averaged over the faces of the *same attribute*
//!   touching a DOF.  Row scaling only rescales the Lagrange multiplier and
//!   does not change the displacement solution.
//!
//! The DOF layout is [`fem_space::Ordering::ByNodes`] (= MFEM `byNODES`):
//! component `c` of scalar DOF `s` has global index `c * n_scalar_dofs + s`
//! — `Ordering::ByNodes.map(n_scalar_dofs, 2, s, c)` (block layout).

use std::collections::BTreeMap;

use fem_core::types::DofId;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::{DofManager, EdgeKey};

/// Scalar-space DOFs on a boundary face: the face's vertex DOFs plus the
/// mid-edge DOFs (order ≥ 2).
///
/// Mirrors MFEM `FiniteElementSpace::GetBdrElementDofs` for H1 spaces on a
/// linear-geometry mesh (vertex DOFs first, then the canonical edge DOFs).
pub fn boundary_face_dofs<M: MeshTopology>(
    mesh: &M,
    scalar_dofs: &DofManager,
    face: u32,
) -> Vec<DofId> {
    let nodes = mesh.face_nodes(face);
    let mut dofs: Vec<DofId> = nodes.iter().map(|&n| n as DofId).collect();
    if scalar_dofs.order >= 2 && nodes.len() >= 2 {
        let key = EdgeKey::new(nodes[0], nodes[1]);
        // Order-2 spaces store one mid-edge DOF in `edge_dof_map`; the general
        // Pk (order ≥ 3) path stores p−1 edge DOFs in `edge_pk_map`.
        if scalar_dofs.order == 2 {
            if let Some(&d) = scalar_dofs.edge_dof_map.get(&key) {
                dofs.push(d);
            }
        } else if let Some(edge_dofs) = scalar_dofs.edge_pk_map.get(&key) {
            dofs.extend_from_slice(edge_dofs);
        }
    }
    dofs
}

/// Build the normal-constraint matrix for the given boundary attributes.
///
/// - `mesh` — the (refined) mesh.
/// - `scalar_dofs` — the scalar-space DOF manager of one component of the
///   vector space (e.g. `VectorH1Space::scalar_dof_manager()`).
/// - `constrained_att` — boundary attributes on which the normal displacement
///   is constrained.
///
/// Returns `(C, lagrange_rowstarts)` where `C` is `n_rows × (vdim·n_scalar)`
/// and `lagrange_rowstarts[b]` is the first row of block-constraint `b`.
pub fn build_normal_constraints<M: MeshTopology>(
    mesh: &M,
    scalar_dofs: &DofManager,
    constrained_att: &[i32],
) -> (CsrMatrix<f64>, Vec<usize>) {
    // Accumulate the summed normal + face count per (scalar DOF, attribute).
    // BTreeMap iterates (dof, attr) in ascending order — the same row
    // ordering MFEM produces (ascending tdof, ascending attribute).
    let mut dof_normals: BTreeMap<(DofId, i32), (f64, f64, usize)> = BTreeMap::new();

    for &att in constrained_att {
        for f in 0..mesh.n_boundary_faces() as u32 {
            if mesh.face_tag(f) != att {
                continue;
            }
            let nodes = mesh.face_nodes(f);
            if nodes.len() < 2 {
                continue;
            }
            let p0 = mesh.node_coords(nodes[0]);
            let p1 = mesh.node_coords(nodes[1]);
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            if dx * dx + dy * dy < 1e-28 {
                continue;
            }
            // Unscaled outward normal (CCW perpendicular): (dy, -dx).
            // For straight segments this is the same at every DOF of the face
            // (vertex and mid-edge), matching MFEM's CalcOrtho evaluated at
            // each FE node.
            let (nx, ny) = (dy, -dx);
            for dof in boundary_face_dofs(mesh, scalar_dofs, f) {
                let entry = dof_normals.entry((dof, att)).or_insert((0.0, 0.0, 0));
                entry.0 += nx;
                entry.1 += ny;
                entry.2 += 1;
            }
        }
    }

    let n_rows = dof_normals.len();
    let n_scalar = scalar_dofs.n_dofs;
    let n_total = 2 * n_scalar; // vdim = 2
    let mut coo = CooMatrix::new(n_rows, n_total);
    let mut rowstarts = vec![0usize];
    let mut prev_dof: Option<DofId> = None;
    for (row, ((dof, _att), (nx_sum, ny_sum, count))) in dof_normals.iter().enumerate() {
        if prev_dof != Some(*dof) {
            if row > 0 {
                rowstarts.push(row);
            }
            prev_dof = Some(*dof);
        }
        let inv = 1.0 / (*count as f64);
        let nx = nx_sum * inv;
        let ny = ny_sum * inv;
        let dof_x = *dof as usize;
        let dof_y = dof_x + n_scalar;
        coo.add(row, dof_x, nx);
        coo.add(row, dof_y, ny);
    }
    rowstarts.push(n_rows);

    (coo.into_csr(), rowstarts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{Mesh, element_type::ElementType, topology::MeshTopology};
    use fem_space::DofManager;

    /// The ex28 trapezoid mesh, uniformly refined 4 times (17×17 nodes).
    fn trapezoid_mesh() -> Mesh<2> {
        let coords = vec![0.0, 0.0, 1.0, 0.0, 0.3, 1.0, 1.0, 1.0];
        let conn = vec![0u32, 1, 3, 2];
        let face_conn = vec![0u32, 1, 1, 3, 2, 3, 0, 2];
        let face_tags = vec![1, 2, 3, 4];
        let mut mesh = Mesh::uniform(
            coords, conn, vec![1], ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        );
        for _ in 0..4 { mesh = fem_mesh::refine_uniform(&mesh); }
        mesh
    }

    #[test]
    fn normal_constraints_corner_is_pinned() {
        // Order 1: 33 constrained nodes; the corner (0,0) is on attrs 1 and 4
        // → one block constraint with 2 rows → 34 rows total (matching MFEM).
        let mesh = trapezoid_mesh();
        let dm = DofManager::new(&mesh, 1);
        let (c, rowstarts) = build_normal_constraints(&mesh, &dm, &[1, 4]);
        assert_eq!(c.nrows, 34);
        assert_eq!(rowstarts, vec![0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                   27, 28, 29, 30, 31, 32, 33, 34]);
        // Rows 0 and 1 are the corner's two independent constraints (block 0):
        // row 0 has only a y-entry (bottom face normal (0,-1)),
        // row 1 has x and y entries (left face normal ∝ (1,-0.3)).
        // Together they fully pin the corner: n0x and n0y are both non-zero
        // across the two rows, i.e. the 2×2 sub-block is non-singular.
        let n = dm.n_dofs;
        let m = c.to_dense();
        let sub = [[m[0 * n + 0], m[0 * n + n]], [m[1 * n + 0], m[1 * n + n]]];
        let det = sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0];
        assert!(det.abs() > 1e-6, "corner constraint block must be non-singular");
        // row 0 = bottom face: only the y component is non-zero.
        assert_eq!(m[0 * n + 0], 0.0);
        assert!(m[0 * n + n].abs() > 0.0);
    }

    #[test]
    fn normal_constraints_order2_includes_edge_dofs() {
        // Order 2: boundary vertex dofs (33) + boundary mid-edge dofs (32)
        // → 65 nodes/dofs; the corner block has 2 rows → 66 rows (matching MFEM).
        let mesh = trapezoid_mesh();
        let dm = DofManager::new(&mesh, 2);
        let (c, _rowstarts) = build_normal_constraints(&mesh, &dm, &[1, 4]);
        assert_eq!(c.nrows, 66);
    }
}
