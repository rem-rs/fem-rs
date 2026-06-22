//! Reduced-Order Models via Proper Orthogonal Decomposition (POD).
//!
//! # Overview
//!
//! 1. **Snapshot collection** — solve the full-order model at N parameter
//!    values / time steps; store each solution as a column of `Snapshots`.
//! 2. **POD basis** — compute the SVD of the snapshot matrix and retain
//!    the `r` dominant left singular vectors.
//! 3. **Galerkin projection** — project the full-order operator `A` and
//!    RHS `b` onto the POD basis: `A_r = V^T A V`, `b_r = V^T b`.
//! 4. **Online solve** — solve the r×r reduced system.
//!
//! # Usage
//! ```rust,ignore
//! use fem_solver::rom::{Snapshots, PodBasis, project_system};
//!
//! // Collect N solution vectors (each of length n):
//! let mut snaps = Snapshots::new(n);
//! for param in params {
//!     let u = solve_fom(param);
//!     snaps.add_snapshot(&u);
//! }
//!
//! // Extract r dominant POD modes:
//! let pod = PodBasis::compute(&snaps, r)?;
//!
//! // Project the system:
//! let (a_r, b_r) = project_system(&a, &b, &pod);
//!
//! // Solve the reduced system:
//! let x_r = a_r.solve(&b_r);
//! ```

use nalgebra::{DMatrix, DVector, SVD};
use fem_linalg::CsrMatrix;

/// Snapshot matrix: columns are solution vectors at different parameters / times.
///
/// Shape: `(n_dofs, n_snapshots)`.
#[derive(Debug, Clone)]
pub struct Snapshots {
    n_dofs: usize,
    snapshots: Vec<Vec<f64>>,
}

impl Snapshots {
    /// Create an empty snapshot collector for an `n`-DOF system.
    pub fn new(n_dofs: usize) -> Self {
        Snapshots { n_dofs, snapshots: Vec::new() }
    }

    /// Add a solution vector as a new snapshot column.
    pub fn add_snapshot(&mut self, u: &[f64]) {
        assert_eq!(u.len(), self.n_dofs, "snapshot length {} != n_dofs {}", u.len(), self.n_dofs);
        self.snapshots.push(u.to_vec());
    }

    /// Number of DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Number of snapshots collected.
    pub fn n_snapshots(&self) -> usize { self.snapshots.len() }

    /// Access the i-th snapshot.
    pub fn snapshot(&self, i: usize) -> &[f64] { &self.snapshots[i] }

    /// Convert to a dense matrix (n_dofs × n_snapshots).
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.n_dofs, self.snapshots.len());
        for (j, snap) in self.snapshots.iter().enumerate() {
            for (i, &val) in snap.iter().enumerate() {
                m[(i, j)] = val;
            }
        }
        m
    }
}

/// POD basis: `r` dominant modes (columns) = left singular vectors of the
/// snapshot matrix.
#[derive(Debug, Clone)]
pub struct PodBasis {
    /// POD modes (columns). Shape: `(n_dofs, r)`.
    pub modes: DMatrix<f64>,
    /// Singular values (sorted descending).
    pub singular_values: Vec<f64>,
    /// Number of retained modes.
    pub r: usize,
}

impl PodBasis {
    /// Compute the POD basis from snapshots, retaining the top `r` modes.
    ///
    /// Internally computes the thin SVD of the snapshot matrix.
    /// Returns an error if `r` exceeds the snapshot count or DOF count.
    pub fn compute(snaps: &Snapshots, r: usize) -> Result<Self, String> {
        let n_snap = snaps.n_snapshots();
        if n_snap == 0 {
            return Err("POD: no snapshots".to_string());
        }
        let r_eff = r.min(n_snap).min(snaps.n_dofs());
        if r_eff == 0 {
            return Err("POD: effective rank is 0".to_string());
        }

        let s = snaps.to_matrix();
        let svd = SVD::new(s, true, false);
        let u = svd.u.ok_or("POD: SVD did not return U")?;

        let n_modes = r_eff.min(u.ncols());
        let modes = u.columns(0, n_modes).into_owned();

        // Build a small intermediate S matrix to extract singular values
        // Start from a minimal size to get the first n_modes singular values
        // Note: in nalgebra, svd.singular_values is a Vector of min(n_dofs, n_snap) values
        let sv = svd.singular_values;
        let sv_total = sv.len();
        let n_sv = n_modes.min(sv_total);
        let singular_values: Vec<f64> = sv.iter().take(n_sv).copied().collect();

        Ok(PodBasis { modes, singular_values, r: n_modes })
    }

    /// Number of basis vectors (reduced dimension).
    pub fn n_modes(&self) -> usize { self.r }

    /// The i-th POD mode (column vector, length n_dofs).
    pub fn mode(&self, i: usize) -> Vec<f64> {
        (0..self.modes.nrows()).map(|row| self.modes[(row, i)]).collect()
    }

    /// Energy fraction captured by the first `k` modes: sum(σᵢ²) / sum(all σᵢ²).
    pub fn energy_fraction(&self, k: usize) -> f64 {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total == 0.0 { return 1.0; }
        let partial: f64 = self.singular_values.iter().take(k).map(|s| s * s).sum();
        partial / total
    }

    /// Cumulative energy fraction across all computed modes.
    pub fn cumulative_energy(&self) -> Vec<f64> {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        if total == 0.0 { return vec![1.0; self.singular_values.len()]; }
        let mut cum = Vec::with_capacity(self.singular_values.len());
        let mut running = 0.0;
        for s in &self.singular_values {
            running += s * s;
            cum.push(running / total);
        }
        cum
    }
}

/// Project a full-order system `(A, b)` onto the POD basis.
///
/// Returns `(A_r, b_r)` where:
/// - `A_r = V^T A V`  (r × r reduced operator)
/// - `b_r = V^T b`    (r × 1 reduced RHS)
pub fn project_system(
    a: &CsrMatrix<f64>,
    b: &[f64],
    pod: &PodBasis,
) -> (DMatrix<f64>, DVector<f64>) {
    let n = a.nrows;
    let r = pod.r;
    assert_eq!(n, b.len());

    // Compute AV: each column = A * v_i
    let mut av = DMatrix::zeros(n, r);
    for j in 0..r {
        let mode_j = pod.mode(j);
        let mut col = vec![0.0; n];
        a.spmv(&mode_j, &mut col);
        for i in 0..n {
            av[(i, j)] = col[i];
        }
    }

    // A_r = V^T * (A * V) = V^T * AV
    let a_r = pod.modes.transpose() * &av;

    // b_r = V^T * b
    let b_vec = DVector::from_vec(b.to_vec());
    let b_r = pod.modes.transpose() * b_vec;

    (a_r, b_r)
}

/// Reconstruct the full-order solution from reduced coefficients.
///
/// `u_full = V * u_r`
pub fn reconstruct(pod: &PodBasis, u_r: &[f64]) -> Vec<f64> {
    let u_vec = DVector::from_vec(u_r.to_vec());
    let full = &pod.modes * u_vec;
    full.data.as_vec().clone()
}

/// Compute the relative L² error between reduced and full-order solutions.
pub fn relative_error(u_full: &[f64], u_reduced: &[f64]) -> f64 {
    let diff: f64 = u_full.iter().zip(u_reduced.iter())
        .map(|(a, b)| (a - b).powi(2)).sum();
    let norm: f64 = u_full.iter().map(|a| a.powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SolverConfig, solve_cg};
    use fem_linalg::CsrMatrix;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0 { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        coo.into_csr()
    }

    fn solve_laplacian_1d(n: usize, rhs_scale: f64) -> Vec<f64> {
        let a = laplacian_1d(n);
        let mut x = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mid = n / 2;
        b[mid - 1] = rhs_scale;
        b[mid] = 2.0 * rhs_scale;
        b[mid + 1] = rhs_scale;
        let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
        solve_cg(&a, &b, &mut x, &cfg).expect("CG");
        x
    }

    #[test]
    fn pod_snapshot_creation() {
        let mut snaps = Snapshots::new(10);
        assert_eq!(snaps.n_dofs(), 10);
        assert_eq!(snaps.n_snapshots(), 0);

        snaps.add_snapshot(&vec![1.0; 10]);
        assert_eq!(snaps.n_snapshots(), 1);
    }

    #[test]
    fn pod_basis_from_snapshots() {
        let mut snaps = Snapshots::new(20);
        for scale in 0..5 {
            let u = solve_laplacian_1d(20, scale as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 3).expect("POD basis");
        assert_eq!(pod.n_modes(), 3);
        assert_eq!(pod.modes.nrows(), 20);
        assert_eq!(pod.modes.ncols(), 3);
        assert!(pod.singular_values[0] >= pod.singular_values[1]);
    }

    #[test]
    fn pod_energy_fraction() {
        let mut snaps = Snapshots::new(20);
        for scale in 0..5 {
            let u = solve_laplacian_1d(20, scale as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 5).expect("POD basis");
        let e = pod.energy_fraction(5);
        assert!((e - 1.0).abs() < 1e-10, "all modes should capture all energy: {e:.10}");
    }

    #[test]
    fn pod_projection_reduces_dimension() {
        let n = 32;
        let mut snaps = Snapshots::new(n);
        for scale in 0..6 {
            let u = solve_laplacian_1d(n, scale as f64);
            snaps.add_snapshot(&u);
        }

        let r = 4;
        let pod = PodBasis::compute(&snaps, r).expect("POD basis");

        let a = laplacian_1d(n);
        let mut b = vec![0.0; n];
        b[n / 2] = 1.0;

        let (a_r, b_r) = project_system(&a, &b, &pod);
        assert_eq!(a_r.nrows(), r);
        assert_eq!(a_r.ncols(), r);
        assert_eq!(b_r.len(), r);
    }

    #[test]
    fn pod_reconstruction_small_error() {
        let n = 32;
        let mut snaps = Snapshots::new(n);
        // Snapshots from many RHS configurations
        for k in 0..8 {
            let mut b = vec![0.0; n];
            b[k * 4] = 1.0;
            let a = laplacian_1d(n);
            let mut x = vec![0.0; n];
            let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
            solve_cg(&a, &b, &mut x, &cfg).expect("CG");
            snaps.add_snapshot(&x);
        }

        let r = 5;
        let pod = PodBasis::compute(&snaps, r).expect("POD basis");

        // Project and solve for a RHS that is a linear combination of snapshots
        let a = laplacian_1d(n);
        let mut b = vec![0.0; n];
        // Use the SAME delta RHS as one of the snapshots (k=1 → at index 4)
        b[4] = 1.0;
        let (a_r, b_r) = project_system(&a, &b, &pod);

        // Solve reduced system (dense solve via nalgebra)
        let a_r_dense = a_r;
        let u_r = a_r_dense.lu().solve(&b_r).expect("reduced solve");
        let u_full = reconstruct(&pod, u_r.as_slice());

        // Compare with full solve
        let mut x_full = vec![0.0; n];
        let cfg = SolverConfig { rtol: 1e-10, ..Default::default() };
        solve_cg(&a, &b, &mut x_full, &cfg).expect("CG");

        let err = relative_error(&x_full, &u_full);
        assert!(err < 0.1, "ROM reconstruction error should be reasonable, got {err:.3e}");
    }

    #[test]
    fn pod_energy_monotonic() {
        let mut snaps = Snapshots::new(16);
        for k in 1..=6 {
            let u = solve_laplacian_1d(16, k as f64);
            snaps.add_snapshot(&u);
        }
        let pod = PodBasis::compute(&snaps, 6).expect("POD basis");
        let cum = pod.cumulative_energy();
        for i in 1..cum.len() {
            assert!(cum[i] >= cum[i - 1], "energy should be monotonic: {cum:?}");
        }
    }
}
