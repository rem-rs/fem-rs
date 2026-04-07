//! Parallel distributed CSR matrix.
//!
//! [`ParCsrMatrix`] stores a distributed sparse matrix split into diagonal
//! (owned x owned) and off-diagonal (owned x ghost) blocks.  The parallel
//! SpMV uses ghost exchange to fetch remote vector entries before the local
//! matrix-vector products.

use std::sync::Arc;

use fem_linalg::{CooMatrix, CsrMatrix};

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_vector::ParVector;

/// A distributed CSR matrix: `diag` block (owned columns) + `offd` block
/// (ghost columns).
///
/// Only rows for owned DOFs are stored; ghost rows are discarded during
/// construction (they are handled by the owning rank).
pub struct ParCsrMatrix {
    /// Diagonal block: `n_owned x n_owned`.
    pub(crate) diag: CsrMatrix<f64>,
    /// Off-diagonal block: `n_owned x n_ghost`.
    pub(crate) offd: CsrMatrix<f64>,
    /// Number of owned DOFs (= number of rows).
    pub(crate) n_owned: usize,
    /// Number of ghost DOFs.
    pub(crate) n_ghost: usize,
    /// Ghost exchange pattern for vector data.
    #[allow(dead_code)]
    dof_ghost_exchange: Arc<GhostExchange>,
    /// MPI communicator.
    #[allow(dead_code)]
    comm: Comm,
}

impl ParCsrMatrix {
    /// Build from pre-split diagonal and off-diagonal blocks.
    pub fn from_blocks(
        diag: CsrMatrix<f64>,
        offd: CsrMatrix<f64>,
        n_owned: usize,
        n_ghost: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        ParCsrMatrix { diag, offd, n_owned, n_ghost, dof_ghost_exchange, comm }
    }

    /// Build from a local matrix (n_local x n_local where n_local = n_owned + n_ghost).
    ///
    /// Discards ghost rows (they are handled by the owning rank).  Splits
    /// columns into `diag` (col < n_owned) and `offd` (col >= n_owned,
    /// remapped to 0-based ghost index).
    pub fn from_local_matrix(
        local: &CsrMatrix<f64>,
        n_owned: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        let n_local = local.nrows;
        let n_ghost = if n_local > n_owned { n_local - n_owned } else { 0 };

        let mut diag_coo = CooMatrix::<f64>::new(n_owned, n_owned);
        let offd_cols = if n_ghost > 0 { n_ghost } else { 0 };
        let mut offd_coo = CooMatrix::<f64>::new(n_owned, offd_cols.max(1));

        // Only process owned rows.
        for row in 0..n_owned {
            for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                let col = local.col_idx[k] as usize;
                let val = local.values[k];
                if val == 0.0 { continue; }
                if col < n_owned {
                    diag_coo.add(row, col, val);
                } else if n_ghost > 0 {
                    offd_coo.add(row, col - n_owned, val);
                }
            }
        }

        let diag = diag_coo.into_csr();
        let offd = if n_ghost > 0 {
            // Rebuild with exact column count to get correct ncols.
            let mut c = CooMatrix::<f64>::new(n_owned, n_ghost);
            for row in 0..n_owned {
                for k in local.row_ptr[row]..local.row_ptr[row + 1] {
                    let col = local.col_idx[k] as usize;
                    let val = local.values[k];
                    if val == 0.0 { continue; }
                    if col >= n_owned {
                        c.add(row, col - n_owned, val);
                    }
                }
            }
            c.into_csr()
        } else {
            CsrMatrix::new_empty(n_owned, 0)
        };

        ParCsrMatrix { diag, offd, n_owned, n_ghost, dof_ghost_exchange, comm }
    }

    /// Parallel SpMV: `y = A * x`.
    ///
    /// 1. Update ghost values in `x`.
    /// 2. `y[owned] = diag * x[owned]`.
    /// 3. `y[owned] += offd * x[ghost]`.
    pub fn spmv(&self, x: &mut ParVector, y: &mut ParVector) {
        // Update ghost values in x before reading off-diagonal entries.
        x.update_ghosts();

        // y[0..n_owned] = diag * x[0..n_owned]
        self.diag.spmv(
            &x.data[..self.n_owned],
            &mut y.data[..self.n_owned],
        );

        // y[0..n_owned] += offd * x[n_owned..]
        if self.n_ghost > 0 {
            self.offd.spmv_add(
                1.0,
                &x.data[self.n_owned..self.n_owned + self.n_ghost],
                1.0,
                &mut y.data[..self.n_owned],
            );
        }
    }

    /// Extract the diagonal of the owned block.
    pub fn diagonal(&self) -> Vec<f64> {
        self.diag.diagonal()
    }

    /// Arc-wrapped ghost exchange (for sharing with other structures).
    pub fn ghost_exchange_arc(&self) -> Arc<GhostExchange> {
        Arc::clone(&self.dof_ghost_exchange)
    }

    /// The MPI communicator.
    pub fn comm(&self) -> &Comm { &self.comm }

    /// Apply Dirichlet BC at a local owned DOF: zero the row, set diagonal
    /// to 1, set `rhs[dof] = value`.
    pub fn apply_dirichlet_row(&mut self, local_dof: usize, value: f64, rhs: &mut [f64]) {
        assert!(local_dof < self.n_owned, "can only apply Dirichlet to owned DOFs");

        // Zero diag row and set diagonal to 1.
        self.diag.apply_dirichlet_row_zeroing(local_dof, value, rhs);

        // Zero offd row.
        if self.n_ghost > 0 {
            let start = self.offd.row_ptr[local_dof];
            let end = self.offd.row_ptr[local_dof + 1];
            for k in start..end {
                self.offd.values[k] = 0.0;
            }
        }
    }

    /// Apply Dirichlet BC at a local owned DOF using a `ParVector` as the RHS.
    pub fn apply_dirichlet_par(&mut self, local_dof: usize, value: f64, rhs: &mut ParVector) {
        self.apply_dirichlet_row(local_dof, value, rhs.as_slice_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use crate::ghost::GhostExchange;
    use crate::par_vector::ParVector;
    use fem_linalg::CooMatrix;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test]
    fn par_csr_from_local_splits_correctly() {
        // 4x4 matrix with n_owned=2, n_ghost=2.
        // Row 0: (0,0)=1, (0,1)=2, (0,2)=3, (0,3)=4
        // Row 1: (1,0)=5, (1,1)=6, (1,2)=7, (1,3)=8
        // Row 2: (2,0)=9, ... (ghost row, should be discarded)
        // Row 3: (3,3)=10  (ghost row, should be discarded)
        let mut coo = CooMatrix::<f64>::new(4, 4);
        coo.add(0, 0, 1.0); coo.add(0, 1, 2.0); coo.add(0, 2, 3.0); coo.add(0, 3, 4.0);
        coo.add(1, 0, 5.0); coo.add(1, 1, 6.0); coo.add(1, 2, 7.0); coo.add(1, 3, 8.0);
        coo.add(2, 0, 9.0); coo.add(2, 1, 10.0); coo.add(2, 2, 11.0); coo.add(2, 3, 12.0);
        coo.add(3, 3, 10.0);
        let csr = coo.into_csr();

        // Use a trivial (serial) ghost exchange — no actual communication needed.
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let ghost_ex = Arc::new(GhostExchange::from_trivial());
            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, 2, ghost_ex, comm.clone(),
            );

            // diag should be 2x2: [(0,0)=1, (0,1)=2; (1,0)=5, (1,1)=6]
            assert_eq!(par_mat.diag.nrows, 2);
            assert_eq!(par_mat.diag.ncols, 2);
            assert!((par_mat.diag.get(0, 0) - 1.0).abs() < 1e-14);
            assert!((par_mat.diag.get(0, 1) - 2.0).abs() < 1e-14);
            assert!((par_mat.diag.get(1, 0) - 5.0).abs() < 1e-14);
            assert!((par_mat.diag.get(1, 1) - 6.0).abs() < 1e-14);

            // offd should be 2x2: [(0,0)=3, (0,1)=4; (1,0)=7, (1,1)=8]
            assert_eq!(par_mat.offd.nrows, 2);
            assert_eq!(par_mat.offd.ncols, 2);
            assert!((par_mat.offd.get(0, 0) - 3.0).abs() < 1e-14);
            assert!((par_mat.offd.get(0, 1) - 4.0).abs() < 1e-14);
            assert!((par_mat.offd.get(1, 0) - 7.0).abs() < 1e-14);
            assert!((par_mat.offd.get(1, 1) - 8.0).abs() < 1e-14);

            assert_eq!(par_mat.n_owned, 2);
            assert_eq!(par_mat.n_ghost, 2);
        });
    }

    #[test]
    fn par_csr_spmv_identity() {
        // Parallel SpMV with identity matrix on 2 ranks.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n_local = par_space.n_local_dofs();
            let n_owned = par_space.dof_partition().n_owned_dofs;

            // Build identity matrix over all local DOFs.
            let mut coo = CooMatrix::<f64>::new(n_local, n_local);
            for i in 0..n_local { coo.add(i, i, 1.0); }
            let csr = coo.into_csr();

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, n_owned,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            // x = [1, 2, 3, ...] for owned, ghosts will be filled by exchange.
            let mut x = ParVector::zeros(&par_space);
            for (i, v) in x.owned_slice_mut().iter_mut().enumerate() {
                *v = (i + 1) as f64;
            }
            // Set ghost values by exchange.
            x.update_ghosts();

            let mut y = ParVector::zeros(&par_space);
            par_mat.spmv(&mut x, &mut y);

            // y[owned] should equal x[owned] (identity).
            for i in 0..n_owned {
                assert!(
                    (y.as_slice()[i] - x.as_slice()[i]).abs() < 1e-14,
                    "rank {}: spmv identity mismatch at owned DOF {i}: y={}, x={}",
                    comm.rank(), y.as_slice()[i], x.as_slice()[i]
                );
            }
        });
    }

    #[test]
    fn par_csr_from_local_serial() {
        // 3x3 tridiag on 1 rank: all columns are "owned".
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0, 2.0); coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0); coo.add(1, 1, 2.0); coo.add(1, 2, -1.0);
        coo.add(2, 1, -1.0); coo.add(2, 2, 2.0);
        let csr = coo.into_csr();

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let mesh = SimplexMesh::<2>::unit_square_tri(2);
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, 3,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            assert_eq!(par_mat.diag.nrows, 3);
            assert_eq!(par_mat.n_ghost, 0);
            // Diagonal values.
            let diag = par_mat.diagonal();
            assert!((diag[0] - 2.0).abs() < 1e-14);
            assert!((diag[1] - 2.0).abs() < 1e-14);
            assert!((diag[2] - 2.0).abs() < 1e-14);
        });
    }

    #[test]
    fn par_csr_spmv_serial() {
        // Verify serial SpMV gives correct result.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let n = par_space.n_local_dofs();
            // Build identity matrix.
            let mut coo = CooMatrix::<f64>::new(n, n);
            for i in 0..n { coo.add(i, i, 1.0); }
            let csr = coo.into_csr();

            let par_mat = ParCsrMatrix::from_local_matrix(
                &csr, n,
                par_space.dof_ghost_exchange_arc(),
                comm.clone(),
            );

            // x = [1, 2, 3, ...]
            let mut x = ParVector::zeros(&par_space);
            for (i, v) in x.as_slice_mut().iter_mut().enumerate() {
                *v = (i + 1) as f64;
            }
            let mut y = ParVector::zeros(&par_space);
            par_mat.spmv(&mut x, &mut y);

            // y should equal x (identity).
            for i in 0..n {
                assert!(
                    (y.as_slice()[i] - x.as_slice()[i]).abs() < 1e-14,
                    "spmv mismatch at {i}"
                );
            }
        });
    }
}
