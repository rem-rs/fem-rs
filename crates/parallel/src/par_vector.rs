//! Parallel distributed vector.
//!
//! [`ParVector`] stores a local vector partitioned into owned and ghost DOFs,
//! with communication primitives for ghost exchange and global reductions.

use std::sync::Arc;

use fem_space::fe_space::FESpace;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_space::ParallelFESpace;

/// A distributed vector partitioned across MPI ranks.
///
/// Local layout: `[owned DOFs 0..n_owned) [ghost DOFs n_owned..n_owned+n_ghost)`.
///
/// Ghost DOFs are read-only mirrors of values owned by other ranks.  Call
/// [`update_ghosts`](ParVector::update_ghosts) before any operation that reads
/// ghost values (e.g. SpMV with off-diagonal entries).
pub struct ParVector {
    /// Local data (owned + ghost).
    pub(crate) data: Vec<f64>,
    /// Number of owned DOFs.
    pub(crate) n_owned: usize,
    /// Ghost exchange pattern (shared with ParCsrMatrix).
    dof_ghost_exchange: Arc<GhostExchange>,
    /// MPI communicator.
    comm: Comm,
}

impl ParVector {
    /// Create a zero vector with explicit owned/ghost counts and exchange pattern.
    pub fn zeros_raw(
        n_owned: usize,
        n_ghost: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        ParVector {
            data: vec![0.0; n_owned + n_ghost],
            n_owned,
            dof_ghost_exchange,
            comm,
        }
    }

    /// Create a vector from raw local data with explicit layout.
    pub fn from_local_raw(
        data: Vec<f64>,
        n_owned: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        debug_assert!(data.len() >= n_owned);
        ParVector {
            data,
            n_owned,
            dof_ghost_exchange,
            comm,
        }
    }

    /// Create a zero vector matching the DOF layout of `par_space`.
    pub fn zeros<S: FESpace>(par_space: &ParallelFESpace<S>) -> Self {
        Self::zeros_raw(
            par_space.dof_partition().n_owned_dofs,
            par_space.dof_partition().n_ghost_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        )
    }

    /// Alias for `zeros` — create from a parallel FE space.
    pub fn zeros_from_space<S: FESpace>(par_space: &ParallelFESpace<S>) -> Self {
        Self::zeros(par_space)
    }

    /// Create a vector from existing local data.
    pub fn from_local<S: FESpace>(data: Vec<f64>, par_space: &ParallelFESpace<S>) -> Self {
        debug_assert_eq!(data.len(), par_space.n_local_dofs());
        Self::from_local_raw(
            data,
            par_space.dof_partition().n_owned_dofs,
            par_space.dof_ghost_exchange_arc(),
            par_space.comm().clone(),
        )
    }

    /// Create a zero vector with the same layout as `other`.
    pub fn zeros_like(other: &ParVector) -> Self {
        ParVector {
            data: vec![0.0; other.data.len()],
            n_owned: other.n_owned,
            dof_ghost_exchange: Arc::clone(&other.dof_ghost_exchange),
            comm: other.comm.clone(),
        }
    }

    /// Clone this vector (data + metadata).
    pub fn clone_vec(&self) -> Self {
        ParVector {
            data: self.data.clone(),
            n_owned: self.n_owned,
            dof_ghost_exchange: Arc::clone(&self.dof_ghost_exchange),
            comm: self.comm.clone(),
        }
    }

    /// Clone data into a new ParVector with the same exchange/comm.
    pub fn clone_data(&self) -> Self {
        self.clone_vec()
    }

    // -- slices ---------------------------------------------------------------

    /// View of owned DOFs only.
    #[inline]
    pub fn owned_slice(&self) -> &[f64] { &self.data[..self.n_owned] }

    /// Mutable view of owned DOFs only.
    #[inline]
    pub fn owned_slice_mut(&mut self) -> &mut [f64] { &mut self.data[..self.n_owned] }

    /// Full local data (owned + ghost).
    #[inline]
    pub fn as_slice(&self) -> &[f64] { &self.data }

    /// Mutable full local data.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f64] { &mut self.data }

    /// Total local length (owned + ghost).
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    /// Number of owned DOFs.
    #[inline]
    pub fn n_owned(&self) -> usize { self.n_owned }

    /// Access the communicator.
    #[inline]
    pub fn comm(&self) -> &Comm { &self.comm }

    // -- communication --------------------------------------------------------

    /// Forward exchange: push owned values into ghost slots on neighbour ranks.
    pub fn update_ghosts(&mut self) {
        self.dof_ghost_exchange.forward(&self.comm, &mut self.data);
    }

    /// Reverse exchange: accumulate ghost contributions back to owned slots.
    pub fn accumulate_ghosts(&mut self) {
        self.dof_ghost_exchange.reverse(&self.comm, &mut self.data);
    }

    // -- linear algebra (global reductions) -----------------------------------

    /// Global dot product: `sum_owned(self[i] * other[i])` over all ranks.
    pub fn global_dot(&self, other: &ParVector) -> f64 {
        let local: f64 = self.data[..self.n_owned]
            .iter()
            .zip(&other.data[..self.n_owned])
            .map(|(a, b)| a * b)
            .sum();
        self.comm.allreduce_sum_f64(local)
    }

    /// Global L2 norm: `sqrt(global_dot(self, self))`.
    pub fn global_norm(&self) -> f64 {
        self.global_dot(self).sqrt()
    }

    // -- pointwise operations -------------------------------------------------

    /// `self += alpha * x` (over full local data including ghosts).
    pub fn axpy(&mut self, alpha: f64, x: &ParVector) {
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            *si += alpha * xi;
        }
    }

    /// `self *= alpha`.
    pub fn scale(&mut self, alpha: f64) {
        for v in &mut self.data {
            *v *= alpha;
        }
    }

    /// Copy data from `other` into `self`.
    pub fn copy_from(&mut self, other: &ParVector) {
        self.data.copy_from_slice(&other.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test]
    fn par_vector_global_dot() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            // Set all owned DOFs to 1.0.
            let mut v = ParVector::zeros(&par_space);
            for x in v.owned_slice_mut().iter_mut() {
                *x = 1.0;
            }

            // global_dot(ones, ones) = total number of global DOFs.
            let dot = v.global_dot(&v);
            let expected = par_space.n_global_dofs() as f64;
            assert!(
                (dot - expected).abs() < 1e-10,
                "global_dot = {dot}, expected {expected}"
            );
        });
    }

    #[test]
    fn par_vector_axpy() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let mut v = ParVector::zeros(&par_space);
            let mut w = ParVector::zeros(&par_space);

            // v = 1.0 everywhere, w = 2.0 everywhere (owned + ghost).
            for x in v.as_slice_mut().iter_mut() { *x = 1.0; }
            for x in w.as_slice_mut().iter_mut() { *x = 2.0; }

            // v += 3.0 * w => v should be 7.0 everywhere
            v.axpy(3.0, &w);

            for (i, &val) in v.as_slice().iter().enumerate() {
                assert!(
                    (val - 7.0).abs() < 1e-14,
                    "rank {}: axpy mismatch at index {i}: got {val}, expected 7.0",
                    comm.rank()
                );
            }
        });
    }

    #[test]
    fn par_vector_global_norm() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let mut v = ParVector::zeros(&par_space);
            for x in v.owned_slice_mut().iter_mut() {
                *x = 1.0;
            }

            let norm = v.global_norm();
            let expected = (par_space.n_global_dofs() as f64).sqrt();
            assert!(
                (norm - expected).abs() < 1e-10,
                "global_norm = {norm}, expected {expected}"
            );
        });
    }
}
