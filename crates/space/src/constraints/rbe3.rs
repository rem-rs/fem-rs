//! RBE3 distributing coupling constraint.
//!
//! RBE3 defines a reference (master) node whose motion is the **weighted
//! average** of a set of slave nodes:
//!
//! ```text
//! u_master = Σ (w_i · u_i) / Σ w_i
//! ```
//!
//! Unlike RBE2 (rigid connection), RBE3 distributes loads according to the
//! weighting factors — the master node "floats" with the weighted average
//! of the slave nodes.
//!
//! ## Constraint equation (per DOF component)
//!
//! ```text
//! Σ_i w_i · u_i^slave - (Σ_i w_i) · u_master = 0
//! ```
//!
//! This is implemented as an [`MpcEquation`] (general linear constraint) and
//! applied via penalty method through the existing MPC infrastructure.

use crate::constraints::mpc::{MpcEquation, apply_mpc_penalty_coo};
use fem_linalg::CsrMatrix;

/// RBE3 distributing coupling constraint.
///
/// The master node's displacement (or each component) equals the weighted
/// average of the slave node displacements.
#[derive(Debug, Clone)]
pub struct Rbe3Constraint {
    /// Master node index.
    pub master_node: usize,
    /// Slave node indices with their weights.
    pub slaves: Vec<(usize, f64)>,
    /// Number of DOF components (2 for 2D, 3 for 3D).
    pub n_comp: usize,
}

/// Apply RBE3 constraints via penalty method.
///
/// Converts each RBE3 into `n_comp` MPC equations (one per DOF component)
/// and delegates to [`apply_mpc_penalty_coo`].
pub fn apply_rbe3_penalty_coo(
    mat: CsrMatrix<f64>,
    constraints: &[Rbe3Constraint],
) -> CsrMatrix<f64> {
    let mpc_equations = rbe3_to_mpc(constraints);
    apply_mpc_penalty_coo(mat, &mpc_equations)
}

/// Convert RBE3 constraints to MPC equations.
fn rbe3_to_mpc(constraints: &[Rbe3Constraint]) -> Vec<MpcEquation> {
    let mut equations = Vec::new();

    for rbe3 in constraints {
        let w_sum: f64 = rbe3.slaves.iter().map(|&(_, w)| w).sum();
        if w_sum.abs() < 1e-30 {
            continue;
        }

        for comp in 0..rbe3.n_comp {
            let master_dof = rbe3.master_node * rbe3.n_comp + comp;
            let mut dofs = vec![master_dof];
            let mut weights = vec![-w_sum]; // -Σw · u_master

            for &(slave_node, w) in &rbe3.slaves {
                let slave_dof = slave_node * rbe3.n_comp + comp;
                dofs.push(slave_dof);
                weights.push(w); // +w · u_slave
            }

            equations.push(MpcEquation { dofs, weights });
        }
    }

    equations
}

/// Create RBE3 constraints from master/slave node lists with uniform weights.
///
/// Each slave gets weight = 1.0 / n_slaves (so u_master = average of slaves).
pub fn make_rbe3_uniform(
    master_node: usize,
    slave_nodes: &[usize],
    n_comp: usize,
) -> Vec<Rbe3Constraint> {
    let n = slave_nodes.len();
    if n == 0 {
        return Vec::new();
    }
    let w = 1.0 / n as f64;
    let slaves: Vec<(usize, f64)> = slave_nodes.iter().map(|&s| (s, w)).collect();
    vec![Rbe3Constraint { master_node, slaves, n_comp }]
}

/// Create RBE3 constraints from master/slave node lists with explicit weights.
pub fn make_rbe3_weighted(
    master_node: usize,
    slave_nodes: &[usize],
    weights: &[f64],
    n_comp: usize,
) -> Vec<Rbe3Constraint> {
    assert_eq!(slave_nodes.len(), weights.len(),
        "RBE3: slave_nodes and weights must have equal length");
    let slaves: Vec<(usize, f64)> = slave_nodes.iter()
        .zip(weights.iter())
        .map(|(&n, &w)| (n, w))
        .collect();
    vec![Rbe3Constraint { master_node, slaves, n_comp }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    #[test]
    fn rbe3_to_mpc_produces_correct_equation() {
        // RBE3: master=0, slaves=[(1, 0.5), (2, 0.5)], n_comp=2
        let rbe3 = Rbe3Constraint {
            master_node: 0,
            slaves: vec![(1, 0.5), (2, 0.5)],
            n_comp: 2,
        };
        let eqs = rbe3_to_mpc(&[rbe3]);
        assert_eq!(eqs.len(), 2);

        // Component 0: -1.0·u_0 + 0.5·u_2 + 0.5·u_4 = 0
        assert_eq!(eqs[0].dofs, vec![0, 2, 4]);
        assert!((eqs[0].weights[0] - (-1.0)).abs() < 1e-15);
        assert!((eqs[0].weights[1] - 0.5).abs() < 1e-15);
        assert!((eqs[0].weights[2] - 0.5).abs() < 1e-15);

        // Component 1: -1.0·u_1 + 0.5·u_3 + 0.5·u_5 = 0
        assert_eq!(eqs[1].dofs, vec![1, 3, 5]);
    }

    #[test]
    fn make_rbe3_uniform_creates_equal_weights() {
        let constraints = make_rbe3_uniform(0, &[1, 2, 3], 2);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].slaves.len(), 3);
        for &(_, w) in &constraints[0].slaves {
            assert!((w - 1.0/3.0).abs() < 1e-15);
        }
    }

    #[test]
    fn make_rbe3_weighted_preserves_weights() {
        let constraints = make_rbe3_weighted(
            0, &[1, 2], &[2.0, 1.0], 2);
        assert_eq!(constraints[0].slaves[0].1, 2.0);
        assert_eq!(constraints[0].slaves[1].1, 1.0);
    }

    #[test]
    fn apply_rbe3_adds_penalty_terms() {
        let mut coo = CooMatrix::<f64>::new(6, 6);
        for i in 0..6 { coo.add(i, i, 1.0); }
        let mat = coo.into_csr();

        let rbe3 = Rbe3Constraint {
            master_node: 0,
            slaves: vec![(1, 0.5), (2, 0.5)],
            n_comp: 2,
        };
        let result = apply_rbe3_penalty_coo(mat, &[rbe3]);
        let dense = result.to_dense();
        let n = result.nrows;

        // Penalty added to master diagonal (positive)
        assert!(dense[0 * n + 0] > 1e6, "K[0,0] = {:.3e}", dense[0 * n + 0]);
        // Penalty added to slave diagonal (positive)
        assert!(dense[2 * n + 2] > 1e6, "K[2,2] = {:.3e}", dense[2 * n + 2]);
        // Off-diagonal penalty exists (magnitude > 1e6)
        assert!(dense[0 * n + 2].abs() > 1e6, "K[0,2] = {:.3e}", dense[0 * n + 2]);
    }

    #[test]
    fn rbe3_empty_slaves_does_not_panic() {
        let constraints = make_rbe3_uniform(0, &[], 2);
        assert!(constraints.is_empty());
    }
}
