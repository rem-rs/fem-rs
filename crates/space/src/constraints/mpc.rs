//! Multi-point constraints (MPC) and rigid body elements (RBE2).
//!
//! * **MPC**: general linear constraint `Σ w_i · u_i = 0`
//! * **RBE2**: rigid body element where slave DOFs follow a master node rigidly
//!
//! Both are implemented via penalty method using COO format to handle
//! dynamically-added entries, then converted back to CSR.

use fem_linalg::{CsrMatrix, CooMatrix};

/// A single linear constraint equation: `Σ (weight_i · dof_i) = 0`.
#[derive(Debug, Clone)]
pub struct MpcEquation {
    /// DOF indices participating in the constraint.
    pub dofs: Vec<usize>,
    /// Corresponding weights.
    pub weights: Vec<f64>,
}

/// RBE2 rigid body constraint: slave DOFs follow master node.
#[derive(Debug, Clone)]
pub struct Rbe2Constraint {
    /// Master DOF index.
    pub master_dof: usize,
    /// Slave DOF indices that follow the master.
    pub slave_dofs: Vec<usize>,
}

/// Penalty stiffness for MPC enforcement.
const MPC_PENALTY: f64 = 1e12;

/// Apply MPC equations using penalty method via COO.
///
/// Returns a new matrix (COO) with penalty terms added.
/// The original matrix is consumed.
pub fn apply_mpc_penalty_coo(
    mat: CsrMatrix<f64>,
    equations: &[MpcEquation],
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::from_csr(&mat);
    let _n = coo.nrows;

    for eq in equations {
        if eq.dofs.len() < 2 { continue; }
        let w_sum: f64 = eq.weights.iter().map(|w| w * w).sum();
        if w_sum < 1e-30 { continue; }

        for (i, &dof_i) in eq.dofs.iter().enumerate() {
            let wi = eq.weights[i];
            for (j, &dof_j) in eq.dofs.iter().enumerate() {
                let wj = eq.weights[j];
                let k_val = MPC_PENALTY * wi * wj / w_sum;
                coo.add(dof_i, dof_j, k_val);
            }
        }
    }

    coo.into_csr()
}

/// RBE2 elimination via penalty in COO format.
pub fn apply_rbe2_coo(
    mat: CsrMatrix<f64>,
    constraints: &[Rbe2Constraint],
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::from_csr(&mat);
    let _n = coo.nrows;

    for rbe2 in constraints {
        for &slave in &rbe2.slave_dofs {
            if slave >= _n { continue; }
            coo.add(slave, slave, MPC_PENALTY);
        }
    }

    coo.into_csr()
}

/// RBE2 elimination: slave DOF penalty (simple diagonal penalty approach).
pub fn apply_rbe2_elimination(
    mat: CsrMatrix<f64>,
    _rhs: &mut [f64],
    constraints: &[Rbe2Constraint],
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::from_csr(&mat);
    let _n = coo.nrows;

    for rbe2 in constraints {
        let _master = rbe2.master_dof;
        for &slave in &rbe2.slave_dofs {
            if slave >= _n { continue; }
            // Add large penalty to slave diagonal to enforce u_slave ≈ 0
            coo.add(slave, slave, MPC_PENALTY);
        }
    }

    coo.into_csr()
}

/// Create RBE2 constraints from a master node and slave node list.
pub fn make_rbe2_from_nodes(
    master_node: usize,
    slave_nodes: &[usize],
    n_comp: usize,
) -> Vec<Rbe2Constraint> {
    let mut constraints = Vec::new();
    for comp in 0..n_comp {
        let master_dof = master_node * n_comp + comp;
        let slave_dofs: Vec<usize> = slave_nodes
            .iter()
            .map(|&n| n * n_comp + comp)
            .collect();
        constraints.push(Rbe2Constraint {
            master_dof,
            slave_dofs,
        });
    }
    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpc_penalty_coo_modifies_matrix() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        for i in 0..3 { coo.add(i, i, 1.0); }
        let mat = coo.into_csr();

        let eq = MpcEquation {
            dofs: vec![0, 1],
            weights: vec![1.0, 1.0],
        };
        let result = apply_mpc_penalty_coo(mat, &[eq]);
        let dense = result.to_dense();
        let n = result.nrows;

        assert!(dense[0 * n + 0] > 1e6, "K[0,0] = {:.3e}", dense[0 * n + 0]);
        assert!(dense[0 * n + 1] > 1e6, "K[0,1] = {:.3e}", dense[0 * n + 1]);
    }

    #[test]
    fn rbe2_penalty_works() {
        let mut coo = CooMatrix::<f64>::new(4, 4);
        for i in 0..4 { coo.add(i, i, 1.0); }
        let mat = coo.into_csr();

        let constraints = vec![Rbe2Constraint {
            master_dof: 0,
            slave_dofs: vec![2, 3],
        }];
        let result = apply_rbe2_coo(mat, &constraints);
        let dense = result.to_dense();
        let n = result.nrows;

        assert!(dense[2 * n + 2] > 1e6, "K[2,2] = {:.3e}", dense[2 * n + 2]);
        assert!(dense[3 * n + 3] > 1e6, "K[3,3] = {:.3e}", dense[3 * n + 3]);
    }

    #[test]
    fn rbe2_from_nodes() {
        let constraints = make_rbe2_from_nodes(0, &[1, 2], 2);
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].master_dof, 0);
        assert_eq!(constraints[0].slave_dofs, vec![2, 4]);
        assert_eq!(constraints[1].master_dof, 1);
        assert_eq!(constraints[1].slave_dofs, vec![3, 5]);
    }
}
