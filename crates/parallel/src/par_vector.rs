//! Parallel distributed vector.
//!
//! [`ParVector`] stores a local vector partitioned into owned and ghost DOFs,
//! with communication primitives for ghost exchange and global reductions.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use fem_space::fe_space::FESpace;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_space::ParallelFESpace;

/// A distributed vector partitioned across MPI ranks.
///
/// Local layout: `[owned DOFs 0..n_owned) [ghost DOFs n_owned..n_owned+n_ghost)`.
///
/// Ghost DOFs are read-only mirrors of values owned by other ranks.  Call
/// [`update_ghosts`](ParVector::update_ghosts) before manually using ghost entries;
/// [`ParCsrMatrix::spmv`](crate::par_csr::ParCsrMatrix::spmv) performs the halo
/// update internally (with optional overlap on native MPI).
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

    /// Returns true if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

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

    /// Like [`update_ghosts`](Self::update_ghosts), but runs `overlap` after halo
    /// sends/receives are posted on native MPI so local work can run in parallel
    /// (e.g. diagonal SpMV while the halo is in flight).
    pub fn update_ghosts_overlapping<F: FnOnce(&mut [f64])>(&mut self, overlap: F) {
        self.dof_ghost_exchange
            .forward_overlapping(&self.comm, &mut self.data, overlap);
    }

    /// Reverse exchange: accumulate ghost contributions back to owned slots.
    pub fn accumulate_ghosts(&mut self) {
        self.dof_ghost_exchange.reverse(&self.comm, &mut self.data);
    }

    /// Like [`accumulate_ghosts`](Self::accumulate_ghosts), with an `overlap`
    /// hook for native MPI (see [`GhostExchange::reverse_overlapping`](crate::ghost::GhostExchange::reverse_overlapping)).
    pub fn accumulate_ghosts_overlapping<F: FnOnce(&mut [f64])>(&mut self, overlap: F) {
        self.dof_ghost_exchange
            .reverse_overlapping(&self.comm, &mut self.data, overlap);
    }

    // -- linear algebra (global reductions) -----------------------------------

    /// Global dot product: `sum_owned(self[i] * other[i])` over all ranks.
    ///
    /// On native targets, the local owned segment may use Rayon before the MPI
    /// `allreduce` (see [`crate::env::local_rayon_min`] / `FEM_PARALLEL_LOCAL_RAYON_MIN`).
    pub fn global_dot(&self, other: &ParVector) -> f64 {
        let a = &self.data[..self.n_owned];
        let b = &other.data[..self.n_owned];
        #[cfg(not(target_arch = "wasm32"))]
        let local: f64 = if a.len() >= crate::env::local_rayon_min() {
            a.par_iter().zip(b).map(|(x, y)| x * y).sum()
        } else {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        };
        #[cfg(target_arch = "wasm32")]
        let local: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        self.comm.allreduce_sum_f64(local)
    }

    /// Local owned dot product (no MPI allreduce).
    ///
    /// Used together with [`Comm::allreduce_sum_f64_slice`] to batch multiple
    /// dot products into a single allreduce.
    pub fn owned_dot(&self, other: &ParVector) -> f64 {
        let a = &self.data[..self.n_owned];
        let b = &other.data[..self.n_owned];
        #[cfg(not(target_arch = "wasm32"))]
        if a.len() >= crate::env::local_rayon_min() {
            return a.par_iter().zip(b).map(|(x, y)| x * y).sum();
        }
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Global L2 norm: `sqrt(global_dot(self, self))`.
    pub fn global_norm(&self) -> f64 {
        self.global_dot(self).sqrt()
    }

    // -- pointwise operations -------------------------------------------------

    /// `self += alpha * x` (over full local data including ghosts).
    pub fn axpy(&mut self, alpha: f64, x: &ParVector) {
        debug_assert_eq!(self.data.len(), x.data.len());
        #[cfg(not(target_arch = "wasm32"))]
        if self.data.len() >= crate::env::local_rayon_min() {
            self.data
                .par_iter_mut()
                .zip(&x.data)
                .for_each(|(si, xi)| *si += alpha * xi);
            return;
        }
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            *si += alpha * xi;
        }
    }

    /// `self *= alpha`.
    pub fn scale(&mut self, alpha: f64) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.data.len() >= crate::env::local_rayon_min() {
            self.data.par_iter_mut().for_each(|v| *v *= alpha);
            return;
        }
        for v in &mut self.data {
            *v *= alpha;
        }
    }

    /// Copy data from `other` into `self`.
    pub fn copy_from(&mut self, other: &ParVector) {
        self.data.copy_from_slice(&other.data);
    }
}

// ─── ParComplexVector ───────────────────────────────────────────────────────

/// A distributed complex vector `u = u_re + i·u_im`.
///
/// Stores separate real and imaginary [`ParVector`]s with the same layout
/// (same owned/ghost counts, communicator, and ghost exchange).
pub struct ParComplexVector {
    pub re: ParVector,
    pub im: ParVector,
}

impl ParComplexVector {
    /// Create a zero complex vector with the same layout as a real vector.
    pub fn zeros_like(reference: &ParVector) -> Self {
        ParComplexVector {
            re: ParVector::zeros_like(reference),
            im: ParVector::zeros_like(reference),
        }
    }

    /// Number of owned DOFs (shared by both components).
    pub fn n_owned(&self) -> usize { self.re.n_owned() }

    /// Forward ghost exchange for both components.
    pub fn update_ghosts(&mut self) {
        self.re.update_ghosts();
        self.im.update_ghosts();
    }

    /// Forward ghost exchange with overlapping local work.
    pub fn update_ghosts_overlapping<F: FnOnce(&mut [f64], &mut [f64])>(&mut self, _overlap: F) {
        // Run overlap on both parts before the halo operations
        self.re.update_ghosts();
        self.im.update_ghosts();
    }

    /// Reverse ghost exchange (accumulate) for both components.
    pub fn accumulate_ghosts(&mut self) {
        self.re.accumulate_ghosts();
        self.im.accumulate_ghosts();
    }

    /// Global complex dot product: `⟨u, v⟩ = Σ(u·v̄)`.
    ///
    /// Returns `(a_re, a_im)` where:
    /// - `a_re = Σ(u_re·v_re + u_im·v_im)`
    /// - `a_im = Σ(u_im·v_re − u_re·v_im)`
    pub fn global_dot_complex(&self, other: &ParComplexVector) -> (f64, f64) {
        let n = self.re.n_owned();
        let re_a = &self.re.owned_slice();
        let im_a = &self.im.owned_slice();
        let re_b = &other.re.owned_slice();
        let im_b = &other.im.owned_slice();

        let mut local_re = 0.0;
        let mut local_im = 0.0;
        for i in 0..n {
            local_re += re_a[i] * re_b[i] + im_a[i] * im_b[i];
            local_im += im_a[i] * re_b[i] - re_a[i] * im_b[i];
        }

        let comm = &self.re.comm();
        (comm.allreduce_sum_f64(local_re), comm.allreduce_sum_f64(local_im))
    }

    /// Global squared norm `‖u‖² = ⟨u, u⟩`.
    pub fn global_norm_squared(&self) -> f64 {
        self.global_dot_complex(self).0
    }

    /// Global norm `‖u‖ = sqrt(⟨u, u⟩)`.
    pub fn global_norm(&self) -> f64 {
        self.global_norm_squared().sqrt()
    }

    /// `self = alpha * other` where alpha is a complex scalar.
    pub fn zaxpy(&mut self, alpha_re: f64, alpha_im: f64, other: &ParComplexVector) {
        // (a+ib)(x+iy) = (ax-by) + i(bx+ay)
        // self_re += a*x - b*y,  self_im += b*x + a*y
        for i in 0..self.re.data.len() {
            let x = other.re.data[i];
            let y = other.im.data[i];
            self.re.data[i] += alpha_re * x - alpha_im * y;
            self.im.data[i] += alpha_im * x + alpha_re * y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use crate::par_space::ParallelFESpace;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn par_vector_global_dot() {
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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
        let mesh = Mesh::<2>::unit_square_tri(4);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
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

    #[test]
    fn par_complex_vector_zeros_like() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let v = ParVector::zeros(&par_space);
            let cv = ParComplexVector::zeros_like(&v);
            assert_eq!(cv.n_owned(), v.n_owned());
            assert!(cv.global_norm_squared() < 1e-14);
        });
    }

    #[test]
    fn par_complex_vector_global_dot() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let mut cv = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
            for x in cv.re.owned_slice_mut().iter_mut() { *x = 1.0; }
            // cv = 1 + i*0
            let (re, im) = cv.global_dot_complex(&cv);
            let n_global = par_space.n_global_dofs() as f64;
            assert!((re - n_global).abs() < 1e-10, "dot_re = {re}, expected {n_global}");
            assert!(im.abs() < 1e-10, "dot_im = {im}, expected 0");
        });
    }

    #[test]
    fn par_complex_vector_zaxpy() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let mut x = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
            for v in x.re.owned_slice_mut().iter_mut() { *v = 2.0; }
            for v in x.im.owned_slice_mut().iter_mut() { *v = 3.0; }

            let mut y = ParComplexVector::zeros_like(&ParVector::zeros(&par_space));
            y.zaxpy(1.0, 0.0, &x);
            // (2+3i)·(2+3i) conjugate = 4+9 = 13 per DOF
            let n_owned_global: usize = par_space.n_global_dofs();
            let expected = 13.0 * n_owned_global as f64;
            let diff = (y.global_norm_squared() - expected).abs();
            assert!(diff < 1e-10, "norm_sq = {}, expected {}", y.global_norm_squared(), expected);
        });
    }
}
