//! Port of MFEM's `NNLSSolver` (`linalg/solvers.cpp`, MFEM 4.10) — the
//! interval-constrained sparse least-squares solver used by the NURBS
//! reduced integration rules (`DiffusionIntegrator` + `GetReducedRule`,
//! D533(b)).
//!
//! 1:1 translation of the LAPACK-gated host branch of `NNLSSolver::Solve`:
//! the active-set loop (add the column with the largest Lagrange multiplier,
//! solve the least-squares subproblem through an incrementally updated
//! Householder QR, prune non-positive weights) with every tolerance and
//! scaling step kept exactly as in MFEM so the resulting sparse rules match
//! the LAPACK reference build.  The LAPACK calls are served by
//! [`crate::qr`] (`dgeqr2`/`dorm2r` mirrors) and `dgemv`-order products.
//!
//! Deviations from MFEM (documented gap, D561):
//! the `verbosity_` trace prints are not ported — MFEM's only consumer
//! (`GetReducedRule`) runs at the default `verbosity = 0`, and carrying
//! unreachable print code would violate the dead-code rules.  All kernels
//! (QR through [`crate::qr`], the `dgemv`-order products, and the triangular
//! solve) replicate the reference (netlib LAPACK/BLAS) operation order bit
//! for bit, verified against a reference-LAPACK probe.

// The index-based loops are deliberate: every loop mirrors a named
// LAPACK/MFEM loop whose rounding order must not be re-associated (the
// golden tests assert bit equality with the reference).
#![allow(clippy::needless_range_loop)]

use crate::qr::{apply_q, apply_q_transpose, qr_factor, solve_upper_triangular};

/// `NNLSSolver::QRresidualMode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QrResidualMode {
    /// Never compute the residual through the QR factorization.
    Off,
    /// Always compute the residual through the QR factorization.
    On,
    /// Direct residual, switching to the QR residual when the
    /// add/remove-same-column stall is detected (MFEM default).
    #[default]
    Hybrid,
}

/// `Vector::Max()` (non-negative sizes only in the ported paths).
fn vmax(x: &[f64]) -> f64 {
    let mut it = x.iter();
    let mut m = *it.next().unwrap_or(&0.0);
    for &v in it {
        if v > m {
            m = v;
        }
    }
    m
}

/// `Vector::Norml2()` (MFEM `linalg/vector.cpp:968`): the scaled
/// `(scale, sumsq)` norm with MFEM's exact update order
/// (`first = first·arg² + 1`, `scale = n` on rescale), sequential over the
/// entries.
fn vnorm2(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut first = 0.0_f64;
    let mut scale = 0.0_f64;
    for &v in x {
        let n = v.abs();
        if n > 0.0 {
            if scale <= n {
                let arg = scale / n;
                first = first * (arg * arg) + 1.0;
                scale = n;
            } else {
                let arg = n / scale;
                first += arg * arg;
            }
        }
    }
    scale * first.sqrt()
}

/// MFEM `NNLSSolver`: solves for a sparse `s ≥ 0` with `G·s` inside the
/// interval `[rhs_lb, rhs_ub]`.  `mat` is the **column-major** `m × n`
/// matrix `G` with `n ≥ m` (the solver works on the transpose system, like
/// MFEM where the operator's height is the matrix width).
pub struct NnlsSolver {
    /// Column-major `m × n` matrix (`DenseMatrix::Data` layout).
    mat: Vec<f64>,
    m: usize,
    n: usize,
    /// Row scaling of the normalized constraints (`row_scaling_`).
    row_scaling: Vec<f64>,
    const_tol: f64,
    min_nnz: i32,
    max_nnz: i32,
    res_change_termination_tol: f64,
    zero_tol: f64,
    rhs_delta: f64,
    n_outer: u32,
    n_inner: u32,
    n_stall_check: u32,
    normalize: bool,
    qr_residual_mode: QrResidualMode,
    /// `NNLS_qrres_on_`: sticky flag flipped by the hybrid stall detection.
    nnls_qrres_on: bool,
}

impl NnlsSolver {
    /// `NNLSSolver` defaults + `SetOperator`: take the `m × n` column-major
    /// matrix and reset the row scaling to one.
    pub fn new(mat: Vec<f64>, m: usize, n: usize) -> Self {
        assert_eq!(mat.len(), m * n, "NNLSSolver: matrix size mismatch");
        Self {
            mat,
            m,
            n,
            row_scaling: vec![1.0; m],
            const_tol: 1.0e-14,
            min_nnz: 0,
            max_nnz: 0,
            res_change_termination_tol: 1.0e-4,
            zero_tol: 1.0e-14,
            rhs_delta: 1.0e-11,
            n_outer: 100_000,
            n_inner: 100_000,
            n_stall_check: 100,
            normalize: true,
            qr_residual_mode: QrResidualMode::Hybrid,
            nnls_qrres_on: false,
        }
    }

    /// `SetQRResidualMode`.
    pub fn set_qr_residual_mode(&mut self, mode: QrResidualMode) {
        self.qr_residual_mode = mode;
        if mode == QrResidualMode::On {
            self.nnls_qrres_on = true;
        }
    }

    /// `DenseMatrix::Mult` / `dgemv('N')` order: per column, ascending rows,
    /// `y_i += x_j·A(i,j)`.
    fn mult_mat(&self, x: &[f64], y: &mut [f64]) {
        for j in 0..self.n {
            let t = x[j];
            for i in 0..self.m {
                y[i] += t * self.mat[i + j * self.m];
            }
        }
    }

    /// `DenseMatrix::MultTranspose`: `y_j = Σ_i A(i,j)·x_i` (ascending rows).
    fn mult_transpose(&self, x: &[f64], y: &mut [f64]) {
        for j in 0..self.n {
            let mut t = 0.0_f64;
            for i in 0..self.m {
                t += self.mat[i + j * self.m] * x[i];
            }
            y[j] = t;
        }
    }

    /// `NNLSSolver::NormalizeConstraints`: scale each row so the rescaled
    /// half gap is `1e3·const_tol` for all constraints.
    fn normalize_constraints(&mut self, rhs_lb: &mut [f64], rhs_ub: &mut [f64]) {
        let m = self.m;
        let rhs_avg: Vec<f64> = (0..m).map(|i| (rhs_ub[i] + rhs_lb[i]) * 0.5).collect();
        let rhs_halfgap: Vec<f64> = (0..m).map(|i| (rhs_ub[i] - rhs_lb[i]) * 0.5).collect();
        // halfgap_target = 1.0e3 * const_tol_ (scalar broadcast).
        let halfgap_target = 1.0e3 * self.const_tol;
        for i in 0..m {
            let s = halfgap_target / rhs_halfgap[i];
            self.row_scaling[i] = s;
            rhs_lb[i] = (rhs_avg[i] * s) - halfgap_target;
            rhs_ub[i] = (rhs_avg[i] * s) + halfgap_target;
        }
    }

    /// `NNLSSolver::Mult`: bound `G·w` within `± rhs_delta`, normalize the
    /// constraints, solve.
    pub fn mult(&mut self, w: &[f64], sol: &mut [f64]) {
        let mut rhs_ub = vec![0.0_f64; self.m];
        self.mult_mat(w, &mut rhs_ub);
        for (u, s) in rhs_ub.iter_mut().zip(&self.row_scaling) {
            *u *= *s;
        }
        let mut rhs_lb = rhs_ub.clone();
        let rhs_gw = rhs_ub.clone();
        for i in 0..self.m {
            rhs_lb[i] -= self.rhs_delta;
            rhs_ub[i] += self.rhs_delta;
        }
        if self.normalize {
            self.normalize_constraints(&mut rhs_lb, &mut rhs_ub);
        }
        self.solve(&rhs_lb, &rhs_ub, sol);
        let _ = rhs_gw; // only referenced by the unported verbosity trace
    }

    /// `NNLSSolver::Solve` — the active-set loop (MFEM `solvers.cpp:3999+`).
    fn solve(&mut self, rhs_lb: &[f64], rhs_ub: &[f64], soln: &mut [f64]) {
        let m = self.m;
        let n = self.n;
        assert!(rhs_lb.len() == m && rhs_ub.len() == m && soln.len() == n);
        assert!(n >= m, "NNLSSolver system cannot be over-determined.");

        if self.max_nnz == 0 {
            self.max_nnz = self.n as i32;
        }
        let max_nnz = self.max_nnz as usize;

        let rhs_avg_glob: Vec<f64> = (0..m).map(|i| (rhs_ub[i] + rhs_lb[i]) * 0.5).collect();
        let rhs_halfgap_glob: Vec<f64> =
            (0..m).map(|i| (rhs_ub[i] - rhs_lb[i]) * 0.5).collect();

        let min_nnz_cap = self.min_nnz.min(m.min(n) as i32).max(0) as usize;

        let mut nz_ind = vec![0_usize; m];
        let mut res_glob = rhs_avg_glob.clone();
        let mut qt_rhs_glob = rhs_avg_glob.clone();
        let mut qqt_rhs_glob = qt_rhs_glob.clone();
        let mut sub_qt = vec![0.0_f64; m];
        let mut mu = vec![0.0_f64; n];
        let mut n_nz_ind = 0_usize;
        let mut n_glob = 0_usize;
        let mut l2_res_hist: Vec<f64> = Vec::new();
        let mut stalled_indices: Vec<usize> = Vec::new();
        let mut soln_nz_glob = vec![0.0_f64; m];
        let mut soln_nz_glob_up = vec![0.0_f64; m];

        // Column-major work matrices (MFEM keeps these in plain Vectors).
        let mut mat_0_data = vec![0.0_f64; m * n];
        let mut mat_qr_data = vec![0.0_f64; m * n];
        let mut submat_data = vec![0.0_f64; m * n];
        let mut tau = vec![0.0_f64; n];
        let mut sub_tau = vec![0.0_f64; n];
        let mut vec1 = vec![0.0_f64; m];
        // Staging copy of the reflector columns for the in-place ormqr.
        let mut refl_buf = vec![0.0_f64; m * n];

        let mut i_qr_start;

        // Threshold tolerance for the Lagrange multiplier mu.
        let mu_tol = {
            let mut rhs_scaled = rhs_halfgap_glob.clone();
            for i in 0..m {
                rhs_scaled[i] *= self.row_scaling[i];
            }
            let mut tmp = vec![0.0_f64; n];
            self.mult_transpose(&rhs_scaled, &mut tmp);
            1.0e-15 * vmax(&tmp)
        };

        for oiter in 0..self.n_outer {
            // rmax = max_i(|res_i| - halfgap_i), unscaled residual.
            let mut rmax = res_glob[0].abs() - rhs_halfgap_glob[0];
            for i in 1..m {
                rmax = rmax.max(res_glob[i].abs() - rhs_halfgap_glob[i]);
            }
            l2_res_hist.push(vnorm2(&res_glob));

            if rmax <= self.const_tol && n_glob >= min_nnz_cap {
                break; // NNLS target tolerance met (exit_flag 0)
            }
            if n_glob >= max_nnz {
                break; // NNLS target nnz met (exit_flag 0)
            }
            if n_glob >= m {
                break; // NNLS system is square (exit_flag 3)
            }

            // Stall detection after nStallCheck outer iterations.
            if oiter > self.n_stall_check {
                let half = (self.n_stall_check / 2) as usize;
                let o = oiter as usize;
                let back = self.n_stall_check as usize;
                let mut mean0 = 0.0_f64;
                let mut mean1 = 0.0_f64;
                for i in 0..half {
                    mean0 += l2_res_hist[o - i];
                    mean1 += l2_res_hist[o - back - i];
                }
                let mean_res_change = (mean1 / mean0) - 1.0;
                if mean_res_change.abs() < self.res_change_termination_tol {
                    break; // NNLSSolver stall detected (exit_flag 2)
                }
            }

            // Find the next index: largest entry of mu = Gᵀ·(scaled residual).
            for i in 0..m {
                res_glob[i] *= self.row_scaling[i];
            }
            self.mult_transpose(&res_glob, &mut mu);
            for i in 0..n_nz_ind {
                mu[nz_ind[i]] = 0.0;
            }
            for &si in &stalled_indices {
                mu[si] = 0.0;
            }

            let mut mumax = vmax(&mu);
            if mumax < mu_tol {
                let num_stalled = stalled_indices.len();
                if num_stalled > 0 {
                    // Reset the stalled indices and recompute mu.
                    stalled_indices.clear();
                    self.mult_transpose(&res_glob, &mut mu);
                    for i in 0..n_nz_ind {
                        mu[nz_ind[i]] = 0.0;
                    }
                    mumax = vmax(&mu);
                }
            }

            // argmax mu, first index attaining the maximum.
            let mut imax = 0_usize;
            let mut tmax = mu[0];
            for (i, &mui) in mu.iter().enumerate().skip(1) {
                if mui > tmax {
                    tmax = mui;
                    imax = i;
                }
            }

            // Record the new index and append the scaled column.
            nz_ind[n_nz_ind] = imax;
            n_nz_ind += 1;
            for i in 0..m {
                let v = self.mat[i + imax * m] * self.row_scaling[i];
                mat_0_data[i + n_glob * m] = v;
                mat_qr_data[i + n_glob * m] = v;
            }
            i_qr_start = n_glob;
            n_glob += 1;

            // Inner loop: solve the least-squares subproblem, prune.
            let mut stalled_flag = false;
            for _iiter in 0..self.n_inner {
                // (incremental_update = true — the only MFEM code path)
                let n_update = n_glob - i_qr_start;
                let m_update = m - i_qr_start;

                // Apply the leading i_qr_start reflectors: Qᵀ·(new columns),
                // in place on columns i_qr_start..n_glob.  The reflector
                // columns are staged in `refl_buf` (LAPACK ormqr reads the
                // factorization and writes C through disjoint arguments).
                if i_qr_start > 0 {
                    refl_buf[..i_qr_start * m].copy_from_slice(&mat_qr_data[..i_qr_start * m]);
                    apply_q_transpose(
                        i_qr_start,
                        &refl_buf,
                        m,
                        &tau,
                        &mut mat_qr_data[i_qr_start * m..],
                        m,
                        n_update,
                        m,
                    );
                }

                // QR of the trailing (m_update × n_update) sub-matrix.
                for j in 0..n_update {
                    for i in 0..m_update {
                        submat_data[i + j * m_update] =
                            mat_qr_data[i + i_qr_start + (j + i_qr_start) * m];
                    }
                    sub_tau[j] = tau[i_qr_start + j];
                }
                qr_factor(
                    &mut submat_data[..m_update * n_update],
                    m_update,
                    n_update,
                    &mut sub_tau[..n_update],
                )
                .expect("NNLS: QR update factorization");
                for j in 0..n_update {
                    for i in 0..m_update {
                        mat_qr_data[i + i_qr_start + (j + i_qr_start) * m] =
                            submat_data[i + j * m_update];
                    }
                    tau[i_qr_start + j] = sub_tau[j];
                }

                // Apply the Householder reflectors to compute Qᵀ·b.
                if _iiter == 0 {
                    // Apply only the last reflector H(i_qr_start) to
                    // qt_rhs_glob[i_qr_start..m] (sub_qt/sub_tau staging).
                    for i in 0..m_update {
                        submat_data[i] =
                            mat_qr_data[i + i_qr_start + i_qr_start * m];
                        sub_qt[i] = qt_rhs_glob[i + i_qr_start];
                    }
                    sub_tau[0] = tau[i_qr_start];
                    apply_q_transpose(
                        1,
                        &submat_data,
                        m_update,
                        &sub_tau,
                        &mut sub_qt,
                        m_update,
                        1,
                        m_update,
                    );
                    qt_rhs_glob[i_qr_start..i_qr_start + m_update]
                        .copy_from_slice(&sub_qt[..m_update]);
                } else {
                    // Recompute Qᵀ·b from scratch after pruning.
                    qt_rhs_glob.copy_from_slice(&rhs_avg_glob);
                    apply_q_transpose(
                        n_glob,
                        &mat_qr_data,
                        m,
                        &tau,
                        &mut qt_rhs_glob,
                        m,
                        1,
                        m,
                    );
                }

                // R⁻¹·(Qᵀb) for the first n_glob entries.
                vec1.copy_from_slice(&qt_rhs_glob);
                solve_upper_triangular(&mat_qr_data, m, &mut vec1, n_glob);

                // Check if all entries are positive.
                let mut pos_ibool = false;
                let mut smin = if n_glob > 0 { vec1[0] } else { 0.0 };
                for i in 0..n_glob {
                    soln_nz_glob_up[i] = vec1[i];
                    smin = smin.min(soln_nz_glob_up[i]);
                }
                if smin > self.zero_tol {
                    pos_ibool = true;
                    soln_nz_glob[..n_glob].copy_from_slice(&soln_nz_glob_up[..n_glob]);
                }
                if pos_ibool {
                    break;
                }

                if soln_nz_glob_up[n_glob - 1] <= self.zero_tol {
                    // Adding and removing the same column.
                    stalled_flag = true;
                    if self.qr_residual_mode == QrResidualMode::Hybrid {
                        self.nnls_qrres_on = true;
                        break;
                    }
                }

                // Find the maximum permissible step and update.
                let mut alpha = f64::MAX;
                for i in 0..n_glob {
                    if soln_nz_glob_up[i] <= self.zero_tol {
                        alpha = alpha
                            .min(soln_nz_glob[i] / (soln_nz_glob[i] - soln_nz_glob_up[i]));
                    }
                }
                smin = 0.0;
                for i in 0..n_glob {
                    soln_nz_glob[i] += alpha * (soln_nz_glob_up[i] - soln_nz_glob[i]);
                    if i == 0 || soln_nz_glob[i] < smin {
                        smin = soln_nz_glob[i];
                    }
                }
                while smin > self.zero_tol {
                    // Rounding error: an element that should be zero is not.
                    // Recompute alpha at that index and re-update.
                    let mut index_min = 0_usize;
                    smin = soln_nz_glob[0];
                    for i in 1..n_glob {
                        if soln_nz_glob[i] < smin {
                            smin = soln_nz_glob[i];
                            index_min = i;
                        }
                    }
                    alpha = soln_nz_glob[index_min]
                        / (soln_nz_glob[index_min] - soln_nz_glob_up[index_min]);
                    for i in 0..n_glob {
                        soln_nz_glob[i] +=
                            alpha * (soln_nz_glob_up[i] - soln_nz_glob[i]);
                    }
                }

                // Prune zeroed entries (shift columns left, drop the index).
                i_qr_start = n_glob + 1;
                loop {
                    let mut smin = if n_glob > 0 { soln_nz_glob[0] } else { 0.0 };
                    for i in 1..n_glob {
                        smin = smin.min(soln_nz_glob[i]);
                    }
                    if smin >= self.zero_tol {
                        break; // no more zero entries
                    }
                    let ind_zero = soln_nz_glob[..n_glob]
                        .iter()
                        .position(|&v| v < self.zero_tol)
                        .expect("NNLS: zero entry not found");
                    let nz_ind_zero = ind_zero;

                    // Shift mat_0/mat_qr columns [ind_zero+1..n_glob) left.
                    for i in 0..m {
                        for j in ind_zero..n_glob - 1 {
                            mat_qr_data[i + j * m] = mat_0_data[i + (j + 1) * m];
                        }
                        for j in ind_zero..n_glob - 1 {
                            mat_0_data[i + j * m] = mat_qr_data[i + j * m];
                        }
                    }

                    // Remove the zeroed entry from the local index list.
                    for i in nz_ind_zero..n_nz_ind - 1 {
                        nz_ind[i] = nz_ind[i + 1];
                    }
                    n_nz_ind -= 1;

                    // Shift soln_nz_glob.
                    for i in ind_zero..n_glob - 1 {
                        soln_nz_glob[i] = soln_nz_glob[i + 1];
                    }
                    i_qr_start = i_qr_start.min(ind_zero);
                    n_glob -= 1;
                }
            } // inner loop

            if stalled_flag {
                n_glob -= 1;
                n_nz_ind -= 1;
                stalled_indices.push(imax);
            }

            // Compute the residual.
            if !self.nnls_qrres_on {
                res_glob.copy_from_slice(&rhs_avg_glob);
                for j in 0..n_glob {
                    let t = soln_nz_glob[j];
                    for i in 0..m {
                        res_glob[i] -= t * mat_0_data[i + j * m];
                    }
                }
            } else {
                // res = b − Q·Qᵀ·b through the economical QR.
                qqt_rhs_glob.fill(0.0);
                qqt_rhs_glob[..n_glob].copy_from_slice(&qt_rhs_glob[..n_glob]);
                apply_q(n_glob, &mat_qr_data, m, &tau, &mut qqt_rhs_glob, m, 1, m);
                res_glob.copy_from_slice(&rhs_avg_glob);
                for i in 0..m {
                    res_glob[i] -= qqt_rhs_glob[i];
                }
            }
        } // outer loop

        // Insert the solutions (MFEM: `soln = 0.0`, then scatter the active
        // entries through `nz_ind`).
        soln.fill(0.0);
        for i in 0..n_glob {
            soln[nz_ind[i]] = soln_nz_glob[i];
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG matching the C++ probe `tmp/d533/d533_nnls_probe.cpp`
    /// (MFEM 4.10 LAPACK build `NNLSSolver`), so both sides generate the same
    /// seeded matrices.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Golden run of MFEM's `NNLSSolver` through the LAPACK reference build
    /// with the **reference netlib BLAS** (probe `tmp/d533/d533_nnls_probe.cpp`,
    /// LAPACK 3.12.0 + netlib libblas 3.12.0, 2026-09-21): identical seeds must
    /// reproduce the identical active set **and** identical weights bit for
    /// bit — the port replicates the reference operation order exactly.
    #[test]
    fn nnls_matches_mfem_probe_golden() {
        const GOLDEN: [&[(usize, f64)]; 2] = [
            &[
                (0, 0.56959129978601275),
                (1, 0.91484706145449934),
                (2, 1.1667568056150293),
                (5, 0.41566295908917572),
                (6, 0.68542402754121501),
            ],
            &[
                (0, 0.84375385994732754),
                (2, 0.13238925569675589),
                (3, 0.058294505699475814),
                (4, 0.25439263810181401),
                (8, 2.3205591031477759),
                (14, 0.29347761592198668),
                (16, 0.56414123552146245),
                (18, 2.3017493924118613),
                (22, 2.0235887166223137),
                (24, 0.27381484084372637),
                (27, 0.25414323810381467),
                (29, 2.0355414539042385),
            ],
        ];
        for (tcase, (m, n)) in [(0usize, (5usize, 8usize)), (1, (12, 30))] {
            let mut rng = Lcg(100 + tcase as u64);
            let mut mat = vec![0.0_f64; m * n];
            for j in 0..n {
                for i in 0..m {
                    mat[i + j * m] = rng.next_f64();
                }
                if tcase == 1 && j % 3 == 0 {
                    mat[j * m] = 0.0; // some structure
                }
            }
            let mut rng2 = Lcg(7 + tcase as u64);
            let w: Vec<f64> = (0..n).map(|_| (rng2.next_f64() + 1.0) * 0.5).collect();

            let mut nnls = NnlsSolver::new(mat.clone(), m, n);
            let mut sol = vec![0.0_f64; n];
            nnls.mult(&w, &mut sol);

            // Active set identical; every weight bit-identical.
            let expected_nnz = if tcase == 0 { 5 } else { 12 };
            let nnz = sol.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(nnz, expected_nnz, "case {tcase}: active set size");
            for &(i, v) in GOLDEN[tcase] {
                assert_eq!(sol[i], v, "case {tcase} sol[{i}] exact");
            }

            // Interval constraints must hold: G·s within [Gw - δ, Gw + δ]
            // after normalization (loose absolute check on the unnormalized
            // residual).
            let mut gs = vec![0.0_f64; m];
            for j in 0..n {
                let sj = sol[j];
                if sj != 0.0 {
                    for i in 0..m {
                        gs[i] += sj * mat[i + j * m];
                    }
                }
            }
            let mut gw = vec![0.0_f64; m];
            for j in 0..n {
                for i in 0..m {
                    gw[i] += w[j] * mat[i + j * m];
                }
            }
            for i in 0..m {
                let band = 1.0e-9 + 1e-6 * gw[i].abs();
                assert!(
                    (gs[i] - gw[i]).abs() <= band,
                    "case {tcase} row {i}: G·s = {} vs G·w = {}",
                    gs[i],
                    gw[i]
                );
            }
        }
    }

    /// Trivial sanity: an identity operator with non-negative weights
    /// reproduces the weights exactly (the NNLS feasible set is `s ≥ 0`).
    #[test]
    fn nnls_identity_operator() {
        let n = 4;
        let mut mat = vec![0.0_f64; n * n];
        for i in 0..n {
            mat[i + i * n] = 1.0;
        }
        let w = [0.5_f64, 1.5, 0.25, 2.0];
        let mut nnls = NnlsSolver::new(mat, n, n);
        let mut sol = vec![0.0_f64; n];
        nnls.mult(&w, &mut sol);
        for i in 0..n {
            assert!((sol[i] - w[i]).abs() <= 1e-14, "sol[{i}] = {}", sol[i]);
        }
    }
}
