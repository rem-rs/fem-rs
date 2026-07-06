use std::collections::HashMap;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::amr::{HangingFaceConstraint, HangingNodeConstraint};

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

    let mut constraint_map = HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, (c.parent_a, c.parent_b, c.parent_c));
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // For face constraints, each constrained DOF is a weighted sum of 3 parents.
    fn expand_dof_faces(
        dof: usize,
        weight: f64,
        constraint_map: &HashMap<usize, (usize, usize, usize)>,
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
