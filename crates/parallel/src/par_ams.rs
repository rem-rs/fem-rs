//! Parallel AMS preconditioner (block-diagonal).
//!
//! Each rank builds a serial `AmsPrecond` on its diagonal block of the
//! parallel H(Curl) matrix, using the local portion of the discrete
//! gradient `G: H¹ → H(Curl)`.
//!
//! This gives a block-Jacobi preconditioner suitable for use with
//! parallel CG or LOBPCG on well-conditioned subdomains.

use fem_linalg::{CsrMatrix, fem_to_linlvo_csr};
use linlvo::precond::{AmsConfig, AmsPrecond};
use linlvo::DenseVec;
use linlvo::Preconditioner;

use crate::par_csr::ParCsrMatrix;

/// Parallel AMS preconditioner (block-Jacobi).
///
/// Each rank independently constructs [`AmsPrecond`] on the diagonal block
/// of the parallel matrix, preconditioned by the local portion of G.
pub struct ParAmsPrecond {
    /// Local diagonal-block AMS on this rank.
    local_ams: AmsPrecond<f64>,
    n_owned: usize,
}

impl ParAmsPrecond {
    /// Build the parallel AMS preconditioner from the parallel matrix and
    /// the local portion of the discrete gradient.
    ///
    /// * `a` — parallel H(Curl) matrix (used for its diagonal block).
    /// * `g_local` — owned rows of the gradient matrix `G: H¹ → H(Curl)`.
    ///   Shape: `n_owned_edges × n_local_nodes`.
    /// * `config` — AMS configuration (smoother weight, coarse solver, …).
    pub fn new(
        a: &ParCsrMatrix,
        g_local: &CsrMatrix<f64>,
        config: AmsConfig,
    ) -> Self {
        let la = fem_to_linlvo_csr(&a.diag);
        let lg = fem_to_linlvo_csr(g_local);
        let local_ams = AmsPrecond::<f64>::new(&la, &lg, config)
            .expect("ParAmsPrecond: local AmsPrecond setup failed");
        ParAmsPrecond {
            local_ams,
            n_owned: a.n_owned,
        }
    }

    /// Apply the preconditioner: `z = M⁻¹ r`.
    ///
    /// Each rank applies its local `AmsPrecond` to its owned portion of `r`,
    /// writing to `z`. No ghost exchange is performed — this is a
    /// block-diagonal (additive Schwarz) preconditioner.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        let lx = DenseVec::from_vec(r.to_vec());
        let mut lz = DenseVec::zeros(self.n_owned);
        self.local_ams.apply_precond(&lx, &mut lz);
        z.copy_from_slice(lz.as_slice());
    }
}
