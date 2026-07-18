//! Parallel complex-valued sparse matrix (CSR format).
//!
//! `ParComplexCsrMatrix` stores `A = A_re + i·A_im` split across MPI ranks.
//! Layout mirrors [`ParCsrMatrix`]: `diag` (owned×owned) + `offd` (owned×ghost).

use std::sync::Arc;

use fem_linalg::complex_csr::ComplexCsr;

use crate::comm::Comm;
use crate::ghost::GhostExchange;
use crate::par_vector::ParComplexVector;

/// Parallel complex sparse matrix in split re/im CSR format.
#[derive(Clone)]
pub struct ParComplexCsrMatrix {
    pub(crate) diag: ComplexCsr,
    pub(crate) offd: ComplexCsr,
    pub(crate) n_owned: usize,
    pub(crate) n_ghost: usize,
    pub(crate) dof_ghost_exchange: Arc<GhostExchange>,
    pub(crate) comm: Comm,
}

impl ParComplexCsrMatrix {
    /// Build from pre-constructed diag/offd complex blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        diag: ComplexCsr,
        offd: ComplexCsr,
        n_owned: usize,
        n_ghost: usize,
        dof_ghost_exchange: Arc<GhostExchange>,
        comm: Comm,
    ) -> Self {
        ParComplexCsrMatrix { diag, offd, n_owned, n_ghost, dof_ghost_exchange, comm }
    }

    /// Number of locally owned rows.
    pub fn n_owned(&self) -> usize { self.n_owned }

    /// Number of ghost columns.
    pub fn n_ghost(&self) -> usize { self.n_ghost }

    /// Ghost exchange handle.
    pub fn ghost_exchange_handle(&self) -> Arc<GhostExchange> { self.dof_ghost_exchange.clone() }

    /// Diagonal block reference.
    pub fn diag_block(&self) -> &ComplexCsr { &self.diag }

    /// Off-diagonal block reference.
    pub fn offd_block(&self) -> &ComplexCsr { &self.offd }

    /// Diagonal block (mutable).
    pub fn diag_block_mut(&mut self) -> &mut ComplexCsr { &mut self.diag }

    /// MPI communicator.
    pub fn comm(&self) -> &Comm { &self.comm }

    /// Parallel complex SpMV: `y = A·x`.
    ///
    /// Ghost entries of `x` are fetched before the offd multiply.
    pub fn spmv(&self, x: &mut ParComplexVector, y: &mut ParComplexVector) {
        let n = self.n_owned;
        let ng = self.n_ghost;

        // Overlap ghost exchange with diagonal SpMV.
        x.update_ghosts_overlapping(|re_data, im_data| {
            self.diag.spmv_into(
                &re_data[..n],
                &im_data[..n],
                &mut y.re.as_slice_mut()[..n],
                &mut y.im.as_slice_mut()[..n],
            );
        });

        // Off-diagonal contribution (from ghost columns).
        if ng > 0 {
            let x_re = &x.re.as_slice()[n..n + ng];
            let x_im = &x.im.as_slice()[n..n + ng];
            let mut tmp_re = vec![0.0_f64; n];
            let mut tmp_im = vec![0.0_f64; n];
            self.offd.spmv_into(x_re, x_im, &mut tmp_re, &mut tmp_im);
            for i in 0..n {
                y.re.as_slice_mut()[i] += tmp_re[i];
                y.im.as_slice_mut()[i] += tmp_im[i];
            }
        }
    }

    /// Global Euclidean norm of a complex vector.
    pub fn global_norm(&self, x: &ParComplexVector) -> f64 {
        let n = self.n_owned;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let re = x.re.as_slice()[i];
            let im = x.im.as_slice()[i];
            sum += re * re + im * im;
        }
        sum = self.comm.allreduce_sum_f64(sum);
        sum.sqrt()
    }

    /// Global complex dot: `⟨x, y⟩ = Σ x_i · y_i`.
    pub fn global_dot(&self, x: &ParComplexVector, y: &ParComplexVector) -> (f64, f64) {
        let n = self.n_owned;
        let mut dot_re = 0.0_f64;
        let mut dot_im = 0.0_f64;
        for i in 0..n {
            let xr = x.re.as_slice()[i];
            let xi = x.im.as_slice()[i];
            let yr = y.re.as_slice()[i];
            let yi = y.im.as_slice()[i];
            dot_re += xr * yr + xi * yi;
            dot_im += xr * yi - xi * yr;
        }
        dot_re = self.comm.allreduce_sum_f64(dot_re);
        dot_im = self.comm.allreduce_sum_f64(dot_im);
        (dot_re, dot_im)
    }
}
