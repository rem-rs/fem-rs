//! hp‑constraint builder for mixed‑order hanging‑node interfaces.
//!
//! When adjacent elements have different polynomial orders (*p*) and share
//! a non‑conforming (hanging) interface, the higher‑order side has edge‑
//! interior DOFs that the lower‑order side does not.  These DOFs must be
//! constrained so that the trace matches the lower‑order side *in a
//! least‑squares / interpolation sense*.
//!
//! This module builds [`LinearConstraint`] objects by evaluating the
//! Lagrange polynomial basis of the **coarse** (low‑order) edge at the
//! positions of the **fine** (high‑order) edge‑interior DOFs.
//!
//! The resulting constraints are applied via
//! [`apply_linear_constraints`](super::apply_linear_constraints) and
//! recovered via [`recover_linear_values`](super::recover_linear_values) —
//! the same machinery that handles general linear constraints.
//!
//! ## Multi‑level inheritance (Task 2.3 Step 2)
//!
//! When constraints form a chain (e.g. DOF A depends on DOF B, which itself
//! depends on DOFs C + D), [`resolve_constraint_chain`] flattens the chain
//! so that A is expressed directly in terms of free (unconstrained) DOFs
//! only.  The underlying linear constraint application
//! [`apply_linear_constraints`](super::apply_linear_constraints) already
//! performs this expansion implicitly via its recursive `expand_dof` helper.

use std::collections::HashMap;
use crate::constraints::{LinearConstraint, apply_linear_constraints, recover_linear_values};
use fem_mesh::HangingNodeConstraint;

// ============================================================================
// Lagrange interpolation helpers
// ============================================================================

/// Node positions on a reference edge *[0, 1]* for polynomial order `p`.
///
/// Returns `p + 1` positions: equispaced including endpoints.  P1 → `[0, 1]`,
/// P2 → `[0, 0.5, 1]`, P3 → `[0, 1/3, 2/3, 1]`, etc.
pub fn edge_node_positions(p: usize) -> Vec<f64> {
    if p <= 1 {
        return vec![0.0, 1.0];
    }
    let n = p; // p+1 nodes = p intervals
    (0..=p).map(|i| i as f64 / n as f64).collect()
}

/// Compute the Lagrange interpolation weight for basis function *i* at
/// position *x*, given all node positions `nodes`.
///
/// ℓᵢ(x) = Πⱼ₌₀,ⱼ≠ᵢⁿ (x − xⱼ) / (xᵢ − xⱼ)
fn lagrange_basis(i: usize, x: f64, nodes: &[f64]) -> f64 {
    let xi = nodes[i];
    let mut li = 1.0;
    for (j, &xj) in nodes.iter().enumerate() {
        if j == i {
            continue;
        }
        li *= (x - xj) / (xi - xj);
    }
    li
}

/// Lagrange interpolation weights for a fine‑edge DOF constrained to
/// a coarse edge of order `p_coarse`.
///
/// The fine DOF is at reference position `x_target` ∈ *[0, 1]* on the
/// **coarse** edge (i.e. the position mapped to the coarse element's
/// coordinate system).  Returns `(local_index, weight)` pairs for each
/// coarse‑edge DOF.
///
/// # Panics
/// Panics if `p_coarse < 1` or `x_target` is outside *[0, 1]*.
pub fn hp_edge_interpolation_weights(
    p_coarse: usize,
    x_target: f64,
) -> Vec<(usize, f64)> {
    assert!(p_coarse >= 1, "p_coarse must be at least 1");
    assert!(
        x_target >= 0.0 && x_target <= 1.0,
        "x_target must be in [0, 1]"
    );

    let nodes = edge_node_positions(p_coarse); // p+1 nodes
    (0..nodes.len())
        .map(|i| (i, lagrange_basis(i, x_target, &nodes)))
        .filter(|(_, w)| w.abs() > 1e-30)
        .collect()
}

// ============================================================================
// hp constraint builder
// ============================================================================

/// Options for building hp hanging constraints.
#[derive(Debug, Clone)]
pub struct HpConstraintConfig {
    /// Polynomial order of the coarse (lower‑order) element.
    pub p_coarse: usize,
    /// Polynomial order of the fine (higher‑order) element.
    pub p_fine: usize,
    /// Index of the constrained (fine) edge among the fine element's edges.
    /// The fine edge runs from `t=0` to `t=1` in the fine element's local
    /// coordinate system.
    pub fine_edge_idx: usize,
    /// Index of the coarse edge among the coarse element's edges.
    pub coarse_edge_idx: usize,
}

/// Build hp‑constraints for a single edge interface.
///
/// When `p_coarse == p_fine` the result is empty — standard hanging node
/// constraints handle that case.  When `p_fine > p_coarse`, the extra
/// fine‑side edge‑interior DOFs are constrained via Lagrange interpolation
/// from the coarse‑side edge DOFs.
///
/// For each extra fine DOF at position `x` on the **fine** edge, the
/// constraint maps the position to the coarse edge's coordinate system
/// (via a linear mapping of the child sub‑interval) and computes the
/// Lagrange weights.
///
/// # Arguments
/// * `fine_dofs` — global DOF indices of the fine element's edge, in order
///   from the first vertex to the second.  Length = `p_fine + 1`.
/// * `coarse_dofs` — global DOF indices of the coarse element's edge.
///   Length = `p_coarse + 1`.
/// * `child_start` — relative start of the fine child's edge along the
///   coarse parent edge (0.0 = at vertex A, 0.5 = midpoint, etc.).
/// * `child_end` — relative end along the coarse parent edge.
///
/// # Returns
/// A list of [`LinearConstraint`]s, one per constrained fine DOF.
pub fn build_edge_hp_constraints(
    fine_dofs: &[u32],
    coarse_dofs: &[u32],
    child_start: f64,
    child_end: f64,
) -> Vec<LinearConstraint> {
    let p_fine = fine_dofs.len().saturating_sub(1);
    let p_coarse = coarse_dofs.len().saturating_sub(1);

    if p_fine <= p_coarse {
        return Vec::new(); // no extra DOFs to constrain
    }

    // Build the fine-edge node positions in the fine coordinate system [0,1]
    let fine_nodes = edge_node_positions(p_fine);
    // Coarse-edge node positions in the coarse coordinate system [0,1]
    let coarse_nodes = edge_node_positions(p_coarse);

    let mut constraints = Vec::new();

    for fi in 0..fine_dofs.len() {
        let fine_dof = fine_dofs[fi];
        let x_fine = fine_nodes[fi]; // position on fine edge

        // Skip vertex DOFs (shared between coarse and fine)
        if x_fine.abs() < 1e-14 || (x_fine - 1.0).abs() < 1e-14 {
            continue;
        }

        // Map fine position to coarse coordinate system:
        // child maps [0,1] in fine coords to [child_start, child_end] in coarse coords
        let x_coarse = child_start + x_fine * (child_end - child_start);

        // Compute interpolation weights from coarse edge DOFs
        let mut parents = Vec::new();
        for ci in 0..coarse_dofs.len() {
            let w = lagrange_basis(ci, x_coarse, &coarse_nodes);
            if w.abs() > 1e-30 {
                parents.push((coarse_dofs[ci] as usize, w));
            }
        }

        if !parents.is_empty() {
            constraints.push(LinearConstraint {
                constrained: fine_dof as usize,
                parents,
            });
        }
    }

    constraints
}

/// Build hp‑constraints for all edges of a non‑conforming mesh with
/// variable p‑orders.
///
/// This is a convenience wrapper for the case where the caller has:
/// - A list of hanging edges with their child/coarse topology
/// - Element DOF arrays for the fine and coarse sides
///
/// Each entry in `hanging_edges` describes one non‑conforming edge:
/// ```text
/// (fine_dofs, coarse_dofs, child_start, child_end)
/// ```
///
/// Returns a single flat vector of `LinearConstraint`s.
pub fn build_all_hp_constraints(
    edges: &[(&[u32], &[u32], f64, f64)],
) -> Vec<LinearConstraint> {
    let mut all = Vec::new();
    for &(fine_dofs, coarse_dofs, start, end) in edges {
        all.extend(build_edge_hp_constraints(fine_dofs, coarse_dofs, start, end));
    }
    all
}

/// Unified application of hp‑aware hanging constraints.
///
/// First builds the hp constraints using `build_edge_hp_constraints` for
/// each hanging edge, then applies ALL constraints (both standard hanging
/// and hp) via
/// [`apply_linear_constraints`](super::apply_linear_constraints).
///
/// This is the primary entry point for hp‑AMR constraint enforcement.
pub fn apply_hp_hanging_constraints(
    mat: &mut fem_linalg::CsrMatrix<f64>,
    rhs: &mut [f64],
    standard_constraints: &[HangingNodeConstraint],
    hp_constraints: &[LinearConstraint],
) {
    // Combine all constraints into one list
    let mut all: Vec<LinearConstraint> = hp_constraints.to_vec();

    // Convert standard HangingNodeConstraint to LinearConstraint
    for sc in standard_constraints {
        all.push(LinearConstraint {
            constrained: sc.constrained,
            parents: vec![(sc.parent_a, 0.5), (sc.parent_b, 0.5)],
        });
    }

    apply_linear_constraints(mat, rhs, &all);
}

/// Recover constrained DOF values after solving an hp‑constrained system.
///
/// Handles both standard hanging constraints and hp constraints in one call.
pub fn recover_hp_values(
    x: &mut [f64],
    standard_constraints: &[HangingNodeConstraint],
    hp_constraints: &[LinearConstraint],
) {
    // Combine all constraints
    let mut all: Vec<LinearConstraint> = hp_constraints.to_vec();
    for sc in standard_constraints {
        all.push(LinearConstraint {
            constrained: sc.constrained,
            parents: vec![(sc.parent_a, 0.5), (sc.parent_b, 0.5)],
        });
    }
    recover_linear_values(x, &all);
}

// ============================================================================
// Multi‑level constraint chain resolution (Task 2.3 Step 2)
// ============================================================================

/// Flatten a chain of constraints to express `target_dof` in terms of free
/// (unconstrained) DOFs only.
///
/// For example, if the constraints are:
/// - `DOF 1 = 0.5·DOF 2 + 0.5·DOF 3`
/// - `DOF 2 = 0.3·DOF 4 + 0.7·DOF 5`
///
/// Then `resolve_constraint_chain(1, &constraints)` returns:
/// `[(4, 0.15), (5, 0.35), (3, 0.5)]`
///
/// # Arguments
/// * `target_dof` — the DOF to resolve (must appear as `constrained` in one
///   of the constraints, or it is returned as `[(target_dof, 1.0)]`).
/// * `constraints` — the full constraint list (used to build a lookup map).
///
/// # Returns
/// `(free_dof, weight)` pairs.  The weights sum to 1.0 if the constraint
/// graph is closed under the target DOF.
pub fn resolve_constraint_chain(
    target_dof: usize,
    constraints: &[LinearConstraint],
) -> Vec<(usize, f64)> {
    let map: HashMap<usize, Vec<(usize, f64)>> = constraints
        .iter()
        .map(|c| (c.constrained, c.parents.clone()))
        .collect();

    fn expand(
        dof: usize,
        weight: f64,
        map: &HashMap<usize, Vec<(usize, f64)>>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 {
            return;
        }
        if let Some(parents) = map.get(&dof) {
            for &(p, w) in parents {
                expand(p, weight * w, map, out, depth + 1);
            }
        } else {
            // Accumulate; merge weights for the same free DOF
            if let Some(pos) = out.iter().position(|(d, _)| *d == dof) {
                out[pos].1 += weight;
            } else {
                out.push((dof, weight));
            }
        }
    }

    let mut result = Vec::new();
    expand(target_dof, 1.0, &map, &mut result, 0);
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    // ── Lagrange basis tests ───────────────────────────────────────────────

    #[test]
    fn edge_node_positions_p1() {
        let nodes = edge_node_positions(1);
        assert_eq!(nodes, vec![0.0, 1.0]);
    }

    #[test]
    fn edge_node_positions_p2() {
        let nodes = edge_node_positions(2);
        assert_eq!(nodes.len(), 3);
        assert!((nodes[0] - 0.0).abs() < 1e-15);
        assert!((nodes[1] - 0.5).abs() < 1e-15);
        assert!((nodes[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn edge_node_positions_p3() {
        let nodes = edge_node_positions(3);
        assert_eq!(nodes.len(), 4);
        assert!((nodes[0] - 0.0).abs() < 1e-15);
        assert!((nodes[1] - 1.0 / 3.0).abs() < 1e-15);
        assert!((nodes[2] - 2.0 / 3.0).abs() < 1e-15);
        assert!((nodes[3] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn lagrange_basis_p1_interpolation() {
        // For P1: nodes at 0, 1. ℓ₀(x) = 1-x, ℓ₁(x) = x
        let nodes = edge_node_positions(1);
        // At x = 0.3
        let l0 = lagrange_basis(0, 0.3, &nodes);
        let l1 = lagrange_basis(1, 0.3, &nodes);
        assert!((l0 - 0.7).abs() < 1e-15, "ℓ₀(0.3) = {}, expected 0.7", l0);
        assert!((l1 - 0.3).abs() < 1e-15, "ℓ₁(0.3) = {}, expected 0.3", l1);
        assert!((l0 + l1 - 1.0).abs() < 1e-15, "partition of unity");
    }

    #[test]
    fn lagrange_basis_p2_interpolation() {
        // P2 nodes at 0, 0.5, 1. ℓ₀(x) = 2(x-0.5)(x-1)
        let nodes = edge_node_positions(2);
        let l0 = lagrange_basis(0, 0.25, &nodes);
        // ℓ₀(0.25) = 2(0.25-0.5)(0.25-1) = 2(-0.25)(-0.75) = 0.375
        assert!((l0 - 0.375).abs() < 1e-15, "ℓ₀(0.25) = {}, expected 0.375", l0);
        // Partition of unity
        let sum: f64 = (0..3).map(|i| lagrange_basis(i, 0.25, &nodes)).sum();
        assert!((sum - 1.0).abs() < 1e-15, "partition of unity: {}", sum);
    }

    // ── hp_edge_interpolation_weights tests ────────────────────────────────

    #[test]
    fn hp_weights_p2_midpoint_to_p1() {
        // P2 midpoint (x=0.5 in fine) constrained to P1 coarse edge
        // child occupies full edge: child_start=0, child_end=1
        // x_target = 0.5 in coarse coords
        let weights = hp_edge_interpolation_weights(1, 0.5);
        // P1 coarse: nodes at 0, 1 → weights: 0.5, 0.5
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < 1e-15, "w₀ = {}", weights[0].1);
        assert!((weights[1].1 - 0.5).abs() < 1e-15, "w₁ = {}", weights[1].1);
    }

    #[test]
    fn hp_weights_p3_interior_to_p1() {
        // P3 interior DOF at x=1/3 constrained to P1 coarse edge
        let w13 = hp_edge_interpolation_weights(1, 1.0 / 3.0);
        assert_eq!(w13.len(), 2);
        // ℓ₀(1/3) = 1-1/3 = 2/3, ℓ₁(1/3) = 1/3
        assert!((w13[0].1 - 2.0 / 3.0).abs() < 1e-15, "w₀ = {}", w13[0].1);
        assert!((w13[1].1 - 1.0 / 3.0).abs() < 1e-15, "w₁ = {}", w13[1].1);

        // P3 interior DOF at x=2/3 constrained to P1 coarse edge
        let w23 = hp_edge_interpolation_weights(1, 2.0 / 3.0);
        assert_eq!(w23.len(), 2);
        assert!((w23[0].1 - 1.0 / 3.0).abs() < 1e-15, "w₀ = {}", w23[0].1);
        assert!((w23[1].1 - 2.0 / 3.0).abs() < 1e-15, "w₁ = {}", w23[1].1);
    }

    #[test]
    fn hp_weights_p3_interior_to_p2() {
        // P3 interior DOF at x=1/3 constrained to P2 coarse edge
        // P2 coarse nodes at 0, 0.5, 1
        // ℓ₀(1/3) = (1/3-0.5)(1/3-1)/((0-0.5)(0-1)) = (-1/6)(-2/3)/((-0.5)(-1)) = (1/9)/(0.5) = 2/9
        // ℓ₁(1/3) = (1/3-0)(1/3-1)/((0.5-0)(0.5-1)) = (1/3)(-2/3)/((0.5)(-0.5)) = (-2/9)/(-0.25) = 8/9
        // ℓ₂(1/3) = (1/3-0)(1/3-0.5)/((1-0)(1-0.5)) = (1/3)(-1/6)/((1)(0.5)) = (-1/18)/0.5 = -1/9
        let w = hp_edge_interpolation_weights(2, 1.0 / 3.0);
        assert_eq!(w.len(), 3);
        assert!((w[0].1 - 2.0 / 9.0).abs() < 1e-15, "w₀ = {} (expected 2/9)", w[0].1);
        assert!((w[1].1 - 8.0 / 9.0).abs() < 1e-15, "w₁ = {} (expected 8/9)", w[1].1);
        assert!((w[2].1 + 1.0 / 9.0).abs() < 1e-15, "w₂ = {} (expected -1/9)", w[2].1);
        // Sum should be 1 (partition of unity)
        let sum: f64 = w.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-15, "partition of unity: {}", sum);
    }

    // ── build_edge_hp_constraints tests ────────────────────────────────────

    #[test]
    fn edge_constraint_p2_full_edge_to_p1() {
        // Fine edge: P2 (3 DOFs: vertices 0,1 + midpoint interior DOF)
        // Coarse edge: P1 (2 DOFs: vertices only)
        // Child occupies full coarse edge: start=0, end=1
        // fine[0]=10 (vertex, skipped), fine[1]=11 (midpoint, x=0.5, constrained),
        // fine[2]=12 (vertex, skipped)
        let fine = vec![10u32, 11, 12];
        let coarse = vec![10u32, 12];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        assert_eq!(cs.len(), 1, "P2→P1: 1 constraint for midpoint");
        // Constrained DOF is the interior (non-vertex) DOF
        assert_eq!(cs[0].constrained, 11usize);
        assert_eq!(cs[0].parents.len(), 2);
        // Weights: P1 linear interpolation at x=0.5 → w₀=0.5, w₁=0.5
        let w0 = cs[0].parents.iter().find(|(d, _)| *d == 10).map(|(_, w)| *w);
        let w1 = cs[0].parents.iter().find(|(d, _)| *d == 12).map(|(_, w)| *w);
        assert!((w0.unwrap_or(-1.0) - 0.5).abs() < 1e-15);
        assert!((w1.unwrap_or(-1.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn edge_constraint_p3_half_edge_to_p1() {
        // Fine edge: P3 (4 DOFs: vertices + 2 interior)
        // Coarse edge: P1 (2 DOFs: vertices)
        // Child occupies FIRST HALF of coarse edge: start=0, end=0.5
        let fine = vec![20u32, 21, 22, 23];
        let coarse = vec![20u32, 24]; // vertex 20 shared, 24 = other end
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 0.5);
        // 2 interior P3 DOFs → 2 constraints
        assert_eq!(cs.len(), 2, "P3 half-edge → 2 constraints");
        for c in &cs {
            assert!(c.constrained == 21 || c.constrained == 22);
            assert_eq!(c.parents.len(), 2);
            // Sum of weights should be 1
            let sum: f64 = c.parents.iter().map(|(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn edge_constraint_equal_order_no_constraints() {
        // Both P2 → no extra constraints
        let fine = vec![0u32, 1, 2];
        let coarse = vec![0u32, 1, 3];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        assert!(cs.is_empty(), "equal order → no hp constraints");
    }

    #[test]
    fn edge_constraint_fine_lower_order_no_constraints() {
        // Fine P1, coarse P2 → no extra fine DOFs to constrain
        let fine = vec![0u32, 1];
        let coarse = vec![0u32, 1, 2];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        assert!(cs.is_empty(), "fine lower order → no constraints");
    }

    // ── build_all_hp_constraints tests ─────────────────────────────────────

    #[test]
    fn all_hp_constraints_multiple_edges() {
        let f0 = vec![0u32, 1, 2];
        let c0 = vec![0u32, 1];
        let f1 = vec![3u32, 4, 5];
        let c1 = vec![3u32, 6];
        let edges = vec![
            (f0.as_slice(), c0.as_slice(), 0.0, 0.5),
            (f1.as_slice(), c1.as_slice(), 0.5, 1.0),
        ];
        let cs = build_all_hp_constraints(&edges);
        assert_eq!(cs.len(), 2); // one midpoint per child edge
    }

    // ── apply/recover integration tests ────────────────────────────────────

    #[test]
    fn apply_hp_hanging_constraints_system() {
        use fem_linalg::CooMatrix;
        // 3-DOF system: DOF 1 is a standard hanging node (midpoint of 0, 2)
        // DOF 0, 1, 2, with DOF 1 hanging between 0 and 2
        // Additionally, DOF 3 is an HP-constrained DOF (P3→P1 at 1/3)
        let n = 4;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let standard = vec![HangingNodeConstraint {
            constrained: 1, parent_a: 0, parent_b: 2,
        }];
        let hp = vec![LinearConstraint {
            constrained: 3,
            parents: vec![(0, 2.0 / 3.0), (2, 1.0 / 3.0)], // P3→P1 at x=1/3
        }];

        apply_hp_hanging_constraints(&mut mat, &mut rhs, &standard, &hp);

        // Standard and HP constrained rows should be identity
        assert!((mat.get(1, 1) - 1.0).abs() < 1e-14);
        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((rhs[1]).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
    }

    #[test]
    fn recover_hp_values_check() {
        let mut x = vec![3.0, 0.0, 9.0, 0.0];
        let standard = vec![HangingNodeConstraint {
            constrained: 1, parent_a: 0, parent_b: 2,
        }];
        let hp = vec![LinearConstraint {
            constrained: 3,
            parents: vec![(0, 2.0 / 3.0), (2, 1.0 / 3.0)],
        }];

        recover_hp_values(&mut x, &standard, &hp);

        // Standard: x[1] = 0.5*(3+9) = 6
        assert!((x[1] - 6.0).abs() < 1e-14, "x[1] = {}, expected 6", x[1]);
        // HP: x[3] = (2/3)*3 + (1/3)*9 = 2+3 = 5
        assert!((x[3] - 5.0).abs() < 1e-14, "x[3] = {}, expected 5", x[3]);
    }

    // ── hp constraint solve + recover round-trip ───────────────────────────

    #[test]
    fn hp_constraint_solve_and_recover() {
        // 6‑DOF system: DOF 4 hanging (standard) between 1 and 3,
        // DOF 5 HP‑constrained (P3→P1 at x=1/3) between 0 and 2.
        // Free DOFs: 0, 1, 2, 3 → solve 4×4, then recover DOFs 4 and 5.
        let n = 6;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        for i in 0..n {
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let standard = vec![HangingNodeConstraint {
            constrained: 4, parent_a: 1, parent_b: 3,
        }];
        let hp = vec![LinearConstraint {
            constrained: 5,
            parents: vec![(0, 2.0 / 3.0), (2, 1.0 / 3.0)],
        }];

        apply_hp_hanging_constraints(&mut mat, &mut rhs, &standard, &hp);

        // Gauss-Seidel on the 6×6 system
        let mut x = vec![0.0; n];
        for _ in 0..4000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag: f64 = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                if diag.abs() > 1e-30 {
                    x[i] = s / diag;
                }
            }
        }

        // Verify no NaN
        for (i, &v) in x.iter().enumerate() {
            assert!(v.is_finite(), "x[{}] = NaN/Inf", i);
        }

        // Recover
        recover_hp_values(&mut x, &standard, &hp);

        // Standard hanging constraint should hold
        assert!((x[4] - 0.5 * (x[1] + x[3])).abs() < 1e-8,
            "hanging: x[4]={}, 0.5*(x[1]+x[3])={}", x[4], 0.5*(x[1]+x[3]));
        // HP constraint should hold
        let expected = (2.0 / 3.0) * x[0] + (1.0 / 3.0) * x[2];
        assert!((x[5] - expected).abs() < 1e-8,
            "hp: x[5]={}, expected={}", x[5], expected);
    }

    #[test]
    fn hp_constraint_matrix_symmetry_preserved() {
        // With constraints only (no Dirichlet), PᵀKP should be symmetric
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        for i in 0..n {
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let hp = vec![LinearConstraint {
            constrained: 4,
            parents: vec![(0, 0.5), (2, 0.5)],
        }];
        apply_linear_constraints(&mut mat, &mut rhs, &hp);

        // Check symmetry within tolerance
        for i in 0..n {
            let start = mat.row_ptr[i];
            let end = mat.row_ptr[i + 1];
            for p in start..end {
                let j = mat.col_idx[p] as usize;
                let kij = mat.values[p];
                let kji = mat.get(j, i);
                assert!((kij - kji).abs() < 1e-12,
                    "symmetry broken at ({i},{j}): {kij} vs {kji}");
            }
        }
    }

    // ── Child edge sub-interval constraint tests ───────────────────────────

    #[test]
    fn hp_constraint_first_half_of_coarse_edge() {
        // Fine P2 edge occupies first half of coarse P1 edge
        // Fine DOFs: [0, mid_first_half, 1] (but 1 = coarse midpoint in global numbering)
        // Coarse P1 DOFs: [0, 4]
        // Fine vertex 0 = coarse vertex 0, fine vertex 1 = coarse midpoint position
        // x_target in coarse = 0 + 0.5*0.5 = 0.25 (midpoint of first half)
        let fine = vec![0u32, 5, 1]; // 0=shared, 1=midpoint, 5=new midpoint of child
        let coarse = vec![0u32, 4];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 0.5);
        assert!(!cs.is_empty());
        // The interior DOF (index 1 in fine, DOF=5) should be constrained
        let c = cs.iter().find(|c| c.constrained == 5);
        assert!(c.is_some(), "DOF 5 should be constrained");
        let c = c.unwrap();
        // x_target in coarse = 0 + 0.25*(0.5-0) = 0.25
        // P1 interpolation at x=0.25: w₀=0.75, w₁=0.25
        let w0 = c.parents.iter().find(|(d, _)| *d == 0).map(|(_, w)| *w);
        let w4 = c.parents.iter().find(|(d, _)| *d == 4).map(|(_, w)| *w);
        assert!((w0.unwrap_or(-1.0) - 0.75).abs() < 1e-15,
            "w₀ = {} (expected 0.75)", w0.unwrap_or(-1.0));
        assert!((w4.unwrap_or(-1.0) - 0.25).abs() < 1e-15,
            "w₁ = {} (expected 0.25)", w4.unwrap_or(-1.0));
    }

    #[test]
    fn hp_constraint_second_half_of_coarse_edge() {
        // Fine P2 edge occupies second half of coarse P1 edge
        // Coarse P1 DOFs: [0, 4] (only vertices)
        // The fine P2 mid‑point (x_fine=0.5) maps to:
        //   x_coarse = 0.5 + 0.5·(1.0 − 0.5) = 0.75
        // P1 Lagrange at x=0.75: ℓ₀=0.25, ℓ₁=0.75
        let fine = vec![1u32, 6, 2]; // 1=shared mid, 2=shared end, 6=new mid
        let coarse = vec![0u32, 4];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.5, 1.0);
        // Interior DOF 6 should be constrained
        let c = cs.iter().find(|c| c.constrained == 6);
        assert!(c.is_some(), "DOF 6 should be constrained");
        let c = c.unwrap();
        let w0 = c.parents.iter().find(|(d, _)| *d == 0).map(|(_, w)| *w);
        let w4 = c.parents.iter().find(|(d, _)| *d == 4).map(|(_, w)| *w);
        // P1 interpolation at x=0.75: w₀=0.25, w₁=0.75
        assert!((w0.unwrap_or(-1.0) - 0.25).abs() < 1e-15);
        assert!((w4.unwrap_or(-1.0) - 0.75).abs() < 1e-15);
    }
    // ── Multi‑level constraint chain resolution tests (Task 2.3 Step 2) ─────

    fn make_chain_constraints() -> Vec<LinearConstraint> {
        vec![
            LinearConstraint { constrained: 2, parents: vec![(0, 0.5), (1, 0.5)] },
            LinearConstraint { constrained: 3, parents: vec![(2, 0.4), (1, 0.6)] },
            LinearConstraint { constrained: 4, parents: vec![(3, 0.7), (0, 0.3)] },
        ]
    }

    #[test]
    fn resolve_chain_single_level() {
        let cs = vec![LinearConstraint {
            constrained: 2,
            parents: vec![(0, 0.5), (1, 0.5)],
        }];
        let resolved = resolve_constraint_chain(2, &cs);
        assert_eq!(resolved.len(), 2);
        let w0 = resolved.iter().find(|(d, _)| *d == 0).map(|(_, w)| *w).unwrap();
        let w1 = resolved.iter().find(|(d, _)| *d == 1).map(|(_, w)| *w).unwrap();
        assert!((w0 - 0.5).abs() < 1e-15);
        assert!((w1 - 0.5).abs() < 1e-15);
    }

    #[test]
    fn resolve_chain_two_levels() {
        // DOF 3 = 0.5*DOF 1 + 0.5*DOF 2
        // DOF 2 = 0.3*DOF 0 + 0.7*DOF 1
        // Resolved: DOF 3 = 0.15*DOF 0 + 0.85*DOF 1
        let cs = vec![
            LinearConstraint { constrained: 3, parents: vec![(1, 0.5), (2, 0.5)] },
            LinearConstraint { constrained: 2, parents: vec![(0, 0.3), (1, 0.7)] },
        ];
        let resolved = resolve_constraint_chain(3, &cs);
        assert_eq!(resolved.len(), 2, "got {:?}", resolved);
        let w0 = resolved.iter().find(|(d, _)| *d == 0).map(|(_, w)| *w).unwrap();
        let w1 = resolved.iter().find(|(d, _)| *d == 1).map(|(_, w)| *w).unwrap();
        assert!((w0 - 0.15).abs() < 1e-15, "expected 0.15, got {}", w0);
        assert!((w1 - 0.85).abs() < 1e-15, "expected 0.85, got {}", w1);
        let sum: f64 = resolved.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-15);
    }

    #[test]
    fn resolve_chain_three_levels() {
        // Level 0 (free): DOF 0, DOF 1
        // Level 1: DOF 2 = 0.5*0 + 0.5*1
        // Level 2: DOF 3 = 0.4*2 + 0.6*1
        // Level 3: DOF 4 = 0.7*3 + 0.3*0
        // Resolved: DOF 4 = 0.44*0 + 0.56*1
        let cs = make_chain_constraints();
        let resolved = resolve_constraint_chain(4, &cs);
        assert!(!resolved.is_empty());
        let w0 = resolved.iter().find(|(d, _)| *d == 0).map(|(_, w)| *w).unwrap_or(0.0);
        let w1 = resolved.iter().find(|(d, _)| *d == 1).map(|(_, w)| *w).unwrap_or(0.0);
        assert!((w0 - 0.44).abs() < 1e-14);
        assert!((w1 - 0.56).abs() < 1e-14);
    }

    #[test]
    fn resolve_chain_free_dof_returns_self() {
        let cs = make_chain_constraints();
        let resolved = resolve_constraint_chain(0, &cs);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].0, 0);
        assert!((resolved[0].1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn resolve_chain_empty_constraints() {
        let resolved = resolve_constraint_chain(0, &[]);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].0, 0);
    }

    #[test]
    fn multi_level_chain_applied_to_system() {
        let n = 5;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let cs = vec![
            LinearConstraint { constrained: 4, parents: vec![(3, 0.5), (2, 0.5)] },
            LinearConstraint { constrained: 3, parents: vec![(0, 0.5), (1, 0.5)] },
        ];
        apply_linear_constraints(&mut mat, &mut rhs, &cs);

        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);

        let mut x = rhs.clone();
        for _ in 0..2000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag: f64 = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                if diag.abs() > 1e-30 { x[i] = s / diag; }
            }
        }
        recover_linear_values(&mut x, &cs);
        assert!((x[3] - 0.5 * (x[0] + x[1])).abs() < 1e-8);
        assert!((x[4] - 0.5 * (x[2] + x[3])).abs() < 1e-8);
    }

    // ── Mixed‑p hp constraint solve tests (Task 2.3 Step 4) ─────────────────

    #[test]
    fn hp_constraint_mixed_p_solve_recover() {
        // P2->P1 hp interface with Dirichlet BCs.  DOF 2 (P2 interior)
        // must satisfy: x[2] = 0.5*(x[3] + x[1]).
        let n = 4;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        for i in 1..n { coo.add(i, i - 1, -1.0); }
        for i in 0..n - 1 { coo.add(i, i + 1, -1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![0.0; n];

        let fine = vec![0u32, 2, 3];
        let coarse = vec![3u32, 1];
        let hp_cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        assert_eq!(hp_cs.len(), 1);

        // Constraint application before Dirichlet (hp constraints may
        // reference Dirichlet DOFs).
        apply_linear_constraints(&mut mat, &mut rhs, &hp_cs);
        crate::apply_dirichlet(&mut mat, &mut rhs, &[0, 1], &[1.0, 0.0]);
        let mut x = rhs.clone();
        for _ in 0..4000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag: f64 = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                if diag.abs() > 1e-30 { x[i] = s / diag; }
            }
        }
        recover_linear_values(&mut x, &hp_cs);

        let expected = 0.5 * (x[3] + x[1]);
        assert!((x[2] - expected).abs() < 1e-8,
            "P2-P1 hp: x[2]={}, expected={}", x[2], expected);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1]).abs() < 1e-10);
    }

    #[test]
    fn hp_constraint_p3_to_p2_edge_solve() {
        // P3 fine -> P2 coarse with Dirichlet BCs.
        let fine = vec![0u32, 2, 3, 4];
        let coarse = vec![0u32, 5, 4];
        let cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        assert_eq!(cs.len(), 2);

        // Each constraint should satisfy partition of unity
        for c in &cs {
            let sum: f64 = c.parents.iter().map(|(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-14);
        }

        let n = 6;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        for i in 1..n { coo.add(i, i - 1, -1.0); }
        for i in 0..n - 1 { coo.add(i, i + 1, -1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![0.0; n];
        // Constraint application before Dirichlet: hp constraints may
        // reference Dirichlet DOFs (e.g. DOF 0), and the expansion writes
        // to those rows.  Applying constraints first lets the expansion
        // distribute, then Dirichlet BCs overwrite the boundary rows.
        apply_linear_constraints(&mut mat, &mut rhs, &cs);
        crate::apply_dirichlet(&mut mat, &mut rhs, &[0, 4], &[1.0, 0.0]);

        let mut x = rhs.clone();
        for _ in 0..4000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag: f64 = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                if diag.abs() > 1e-30 { x[i] = s / diag; }
            }
        }
        recover_linear_values(&mut x, &cs);

        for c in &cs {
            let expected: f64 = c.parents.iter().map(|&(p, w)| w * x[p]).sum();
            assert!((x[c.constrained] - expected).abs() < 1e-8,
                "P3 DOF {}: {} != {}", c.constrained, x[c.constrained], expected);
        }
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[4]).abs() < 1e-10);
    }

    #[test]
    fn hp_constraint_mixed_p_symmetry_preserved() {
        // P2->P1 hp constraints must preserve PTKP symmetry.
        let n = 4;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        for i in 1..n { coo.add(i, i - 1, -1.0); }
        for i in 0..n - 1 { coo.add(i, i + 1, -1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let fine = vec![0u32, 2, 3];
        let coarse = vec![3u32, 1];
        let hp_cs = build_edge_hp_constraints(&fine, &coarse, 0.0, 1.0);
        apply_linear_constraints(&mut mat, &mut rhs, &hp_cs);

        for i in 0..n {
            for j in 0..n {
                let kij = mat.get(i, j);
                let kji = mat.get(j, i);
                assert!((kij - kji).abs() < 1e-12,
                    "symmetry broken at ({i},{j}): {kij} != {kji}");
            }
        }
    }
}
