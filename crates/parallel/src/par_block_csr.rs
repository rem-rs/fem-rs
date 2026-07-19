//! Parallel 2×2 block matrix SpMV (saddle-point / multiphysics systems).
//!
//! Layout:
//! ```text
//! [A00  A01]     A00, A11: ParCsrMatrix (square, owned+ghost)
//! [A10  A11]     A01, A10: CsrMatrix<f64> (rectangular, serial mixed)
//! ```

use fem_linalg::CsrMatrix;

use crate::ghost::GhostExchange;
use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

/// Parallel 2×2 block vector.
pub struct ParBlockVector2 {
    pub v0: ParVector,
    pub v1: ParVector,
}

impl ParBlockVector2 {
    pub fn new(v0: ParVector, v1: ParVector) -> Self { ParBlockVector2 { v0, v1 } }
    pub fn n_owned_0(&self) -> usize { self.v0.n_owned() }
    pub fn n_owned_1(&self) -> usize { self.v1.n_owned() }
}

/// Parallel 2×2 block matrix with mixed rectangular sub-blocks.
#[allow(dead_code)]
pub struct ParBlockCsrMatrix2 {
    pub a00: ParCsrMatrix,
    pub a01: CsrMatrix<f64>,
    pub a10: CsrMatrix<f64>,
    pub a11: ParCsrMatrix,
    ghost0: std::sync::Arc<GhostExchange>,
    ghost1: std::sync::Arc<GhostExchange>,
    n0: usize,
    n1: usize,
}

impl ParBlockCsrMatrix2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a00: ParCsrMatrix, a01: CsrMatrix<f64>,
        a10: CsrMatrix<f64>, a11: ParCsrMatrix,
        _ghost0: std::sync::Arc<GhostExchange>,
        _ghost1: std::sync::Arc<GhostExchange>,
        n0: usize, n1: usize,
    ) -> Self {
        ParBlockCsrMatrix2 { a00, a01, a10, a11, ghost0: _ghost0, ghost1: _ghost1, n0, n1 }
    }

    pub fn n0(&self) -> usize { self.n0 }
    pub fn n1(&self) -> usize { self.n1 }

    /// Parallel block SpMV: `y = A·x`.
    ///
    /// Ghost entries of `x.v0` and `x.v1` are exchanged before off-diagonal
    /// block multiplies.  Mixed sub-blocks `A01` / `A10` use only the owned
    /// (local) portion of the opposite-space vector (no ghost exchange needed
    /// for those — they come from `ParMixedAssembler` with owned columns).
    pub fn spmv(&self, x: &mut ParBlockVector2, y: &mut ParBlockVector2) {
        let n0 = self.n0;
        let n1 = self.n1;

        // ── y0 = A00·x0 + A01·x1 ──
        let mut a01_x1 = vec![0.0_f64; n0];
        if self.a01.nrows > 0 {
            let x1_owned = &x.v1.as_slice()[..x.v1.n_owned()];
            self.a01.spmv(x1_owned, &mut a01_x1);
        }
        // A00: overlap ghost exchange with diag multiply
        x.v0.update_ghosts_overlapping(|data| {
            self.a00.diag_block().spmv(&data[..n0], &mut y.v0.as_slice_mut()[..n0]);
        });
        let ng0 = self.a00.n_ghost();
        if ng0 > 0 {
            self.a00.offd_block().spmv_add(1.0, &x.v0.as_slice()[n0..n0 + ng0], 1.0, &mut y.v0.as_slice_mut()[..n0]);
        }
        for i in 0..n0 { y.v0.as_slice_mut()[i] += a01_x1[i]; }

        // ── y1 = A10·x0 + A11·x1 ──
        let mut a10_x0 = vec![0.0_f64; n1];
        if self.a10.nrows > 0 {
            let x0_owned = &x.v0.as_slice()[..x.v0.n_owned()];
            self.a10.spmv(x0_owned, &mut a10_x0);
        }
        x.v1.update_ghosts_overlapping(|data| {
            self.a11.diag_block().spmv(&data[..n1], &mut y.v1.as_slice_mut()[..n1]);
        });
        let ng1 = self.a11.n_ghost();
        if ng1 > 0 {
            self.a11.offd_block().spmv_add(1.0, &x.v1.as_slice()[n1..n1 + ng1], 1.0, &mut y.v1.as_slice_mut()[..n1]);
        }
        for i in 0..n1 { y.v1.as_slice_mut()[i] += a10_x0[i]; }
    }

    /// Global dot product of two block vectors.
    pub fn global_dot(&self, a: &ParBlockVector2, b: &ParBlockVector2) -> f64 {
        let mut dot = 0.0_f64;
        for i in 0..self.n0 { dot += a.v0.as_slice()[i] * b.v0.as_slice()[i]; }
        for i in 0..self.n1 { dot += a.v1.as_slice()[i] * b.v1.as_slice()[i]; }
        a.v0.comm().allreduce_sum_f64(dot)
    }

    /// Global norm of a block vector.
    pub fn global_norm(&self, x: &ParBlockVector2) -> f64 {
        let mut sum = 0.0_f64;
        for i in 0..self.n0 { let v = x.v0.as_slice()[i]; sum += v * v; }
        for i in 0..self.n1 { let v = x.v1.as_slice()[i]; sum += v * v; }
        x.v0.comm().allreduce_sum_f64(sum).sqrt()
    }
}
