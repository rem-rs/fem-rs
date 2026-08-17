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

        // Fetch ghost values, then run the diagonal SpMV.
        // NOTE: `ParComplexVector::update_ghosts_overlapping` does NOT invoke
        // its overlap callback (complex halo has no overlap hook), so the
        // diagonal multiply must be done explicitly here.
        x.update_ghosts();
        self.diag.spmv_into(
            &x.re.as_slice()[..n],
            &x.im.as_slice()[..n],
            &mut y.re.as_slice_mut()[..n],
            &mut y.im.as_slice_mut()[..n],
        );

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

    /// Apply a complex Dirichlet BC on one **owned** DOF, MFEM `DIAG_ONE`
    /// style (the same row/column elimination the serial
    /// [`fem_assembly::ComplexSystem::apply_dirichlet`] performs):
    ///
    /// * for every other owned row `j`: `rhs[j] -= A_re[j,i]·v_re − A_im[j,i]·v_im`
    ///   and `rhs_im[j] -= A_im[j,i]·v_re + A_re[j,i]·v_im`, then zero the
    ///   column entries (symmetric elimination);
    /// * zero the entire row `i` and set `A[i,i] = 1 + 0i`;
    /// * `rhs[i] = v_re`, `rhs_im[i] = v_im`.
    ///
    /// The `offd` (ghost-column) part of row `i` is zeroed as well; the
    /// mirror-image ghost-column elimination on the *other* ranks is handled
    /// by [`Self::apply_ghost_ess_columns`].
    pub fn apply_dirichlet_par(
        &mut self,
        local_dof: usize,
        val_re: f64,
        val_im: f64,
        rhs: &mut ParComplexVector,
    ) {
        assert!(local_dof < self.n_owned, "can only apply Dirichlet to owned DOFs");
        let n = self.n_owned;
        let rhs_re = rhs.re.as_slice_mut();
        let rhs_im = rhs.im.as_slice_mut();

        // ── Column elimination on the diag block (rows j != i) ──────────────
        let diag = &mut self.diag;
        for j in 0..n {
            if j == local_dof {
                continue;
            }
            let mut a_re = 0.0_f64;
            let mut a_im = 0.0_f64;
            for p in diag.row_ptr[j]..diag.row_ptr[j + 1] {
                if diag.col_idx[p] as usize == local_dof {
                    a_re = diag.re_vals[p];
                    a_im = diag.im_vals[p];
                    diag.re_vals[p] = 0.0;
                    diag.im_vals[p] = 0.0;
                    break;
                }
            }
            if a_re != 0.0 || a_im != 0.0 {
                rhs_re[j] -= a_re * val_re - a_im * val_im;
                rhs_im[j] -= a_im * val_re + a_re * val_im;
            }
        }

        // ── Row elimination: zero the whole row, set diagonal = 1 + 0i ──────
        for p in diag.row_ptr[local_dof]..diag.row_ptr[local_dof + 1] {
            if diag.col_idx[p] as usize == local_dof {
                diag.re_vals[p] = 1.0;
                diag.im_vals[p] = 0.0;
            } else {
                diag.re_vals[p] = 0.0;
                diag.im_vals[p] = 0.0;
            }
        }
        // Zero the ghost-column part of this row.
        if self.n_ghost > 0 {
            for p in self.offd.row_ptr[local_dof]..self.offd.row_ptr[local_dof + 1] {
                self.offd.re_vals[p] = 0.0;
                self.offd.im_vals[p] = 0.0;
            }
        }

        rhs_re[local_dof] = val_re;
        rhs_im[local_dof] = val_im;
    }

    /// Complete the complex Dirichlet elimination for essential DOFs whose
    /// **ghost columns** cross ranks (mirror image of
    /// [`Self::apply_dirichlet_par`] on the other ranks):
    ///
    /// * `ghost_ess` lists local ghost slots `0..n_ghost` that are essential
    ///   with their complex Dirichlet values;
    /// * for each owned row `j`, the RHS receives
    ///   `rhs[j] -= A_re[j,g]·v_re − A_im[j,g]·v_im`,
    ///   `rhs_im[j] -= A_im[j,g]·v_re + A_re[j,g]·v_im`, then the column
    ///   entries are zeroed (symmetry).
    pub fn apply_ghost_ess_columns(
        &mut self,
        ghost_ess: &[(usize, f64, f64)],
        rhs: &mut ParComplexVector,
    ) {
        if self.n_ghost == 0 || ghost_ess.is_empty() {
            return;
        }
        let offd = &mut self.offd;
        let rhs_re = rhs.re.as_slice_mut();
        let rhs_im = rhs.im.as_slice_mut();
        for &(g, vr, vi) in ghost_ess {
            if vr != 0.0 || vi != 0.0 {
                for row in 0..self.n_owned {
                    for k in offd.row_ptr[row]..offd.row_ptr[row + 1] {
                        if offd.col_idx[k] as usize == g && (offd.re_vals[k] != 0.0 || offd.im_vals[k] != 0.0) {
                            let ar = offd.re_vals[k];
                            let ai = offd.im_vals[k];
                            rhs_re[row] -= ar * vr - ai * vi;
                            rhs_im[row] -= ai * vr + ar * vi;
                        }
                    }
                }
            }
        }
        // Zero the essential ghost columns (symmetry).
        for &(g, _, _) in ghost_ess {
            for row in 0..self.n_owned {
                for k in offd.row_ptr[row]..offd.row_ptr[row + 1] {
                    if offd.col_idx[k] as usize == g {
                        offd.re_vals[k] = 0.0;
                        offd.im_vals[k] = 0.0;
                    }
                }
            }
        }
    }
}
