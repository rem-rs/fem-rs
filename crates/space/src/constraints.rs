//! Essential (Dirichlet) boundary condition enforcement.
//!
//! After assembly, call [`apply_dirichlet`] to modify the stiffness matrix and
//! right-hand side so that constrained DOFs are set to their prescribed values.

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;

use crate::dof_manager::DofManager;

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

    // For P2, also include edge-midpoint DOFs whose both vertex endpoints are boundary nodes.
    if dm.order == 2 {
        let n_elems = dm.dofs_flat.len() / dm.dofs_per_elem;
        for e in 0..n_elems as u32 {
            let dofs  = dm.element_dofs(e);
            // Vertex DOFs are at positions 0,1,2; edge DOFs at 3,4,5.
            // Edge mapping: edge(v0→v1)=dofs[3], edge(v1→v2)=dofs[4], edge(v0→v2)=dofs[5]
            let edge_pairs = [
                (dofs[0], dofs[1], dofs[3]),
                (dofs[1], dofs[2], dofs[4]),
                (dofs[0], dofs[2], dofs[5]),
            ];
            for (a, b, edge_dof) in edge_pairs {
                if node_set.contains(&a) && node_set.contains(&b) {
                    dof_set.insert(edge_dof);
                }
            }
        }
    }

    let mut out: Vec<DofId> = dof_set.into_iter().collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use fem_mesh::SimplexMesh;

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
}
