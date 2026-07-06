use std::collections::HashMap;

use fem_linalg::{CooMatrix, CsrMatrix};

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
