//! Parallel iterative solvers.
//!
//! Provides parallel Conjugate Gradient (CG), Jacobi-preconditioned CG (PCG),
//! Jacobi-preconditioned restarted GMRES, and MINRES on [`ParCsrMatrix`] /
//! [`ParVector`].

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use fem_solver::{SolveResult, SolverConfig, SolverError};

use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

#[cfg(not(target_arch = "wasm32"))]
fn sub_assign_owned(r: &mut [f64], b: &[f64], ax: &[f64], n: usize) {
    if n >= crate::env::local_rayon_min() {
        r[..n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, ri)| *ri = b[i] - ax[i]);
    } else {
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn sub_assign_owned(r: &mut [f64], b: &[f64], ax: &[f64], n: usize) {
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
}

/// `dst[i] = src[i] + beta * dst[i]` for `i < len` (Krylov search direction).
#[cfg(not(target_arch = "wasm32"))]
fn add_scaled_inplace(dst: &mut [f64], src: &[f64], beta: f64, len: usize) {
    if len >= crate::env::local_rayon_min() {
        dst[..len]
            .par_iter_mut()
            .zip(&src[..len])
            .for_each(|(d, s)| *d = *s + beta * *d);
    } else {
        for i in 0..len {
            dst[i] = src[i] + beta * dst[i];
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn add_scaled_inplace(dst: &mut [f64], src: &[f64], beta: f64, len: usize) {
    for i in 0..len {
        dst[i] = src[i] + beta * dst[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mul_assign_diag(z: &mut [f64], inv_diag: &[f64], r: &[f64], n: usize) {
    if n >= crate::env::local_rayon_min() {
        z[..n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, zi)| *zi = inv_diag[i] * r[i]);
    } else {
        for i in 0..n {
            z[i] = inv_diag[i] * r[i];
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn mul_assign_diag(z: &mut [f64], inv_diag: &[f64], r: &[f64], n: usize) {
    for i in 0..n {
        z[i] = inv_diag[i] * r[i];
    }
}

/// `dst[i] -= ca * a[i] + cb * b[i]` for `i < n`.
#[cfg(not(target_arch = "wasm32"))]
fn sub_lincomb2(dst: &mut [f64], a: &[f64], ca: f64, b: &[f64], cb: f64, n: usize) {
    if n >= crate::env::local_rayon_min() {
        dst[..n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, di)| *di -= ca * a[i] + cb * b[i]);
    } else {
        for i in 0..n {
            dst[i] -= ca * a[i] + cb * b[i];
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn sub_lincomb2(dst: &mut [f64], a: &[f64], ca: f64, b: &[f64], cb: f64, n: usize) {
    for i in 0..n {
        dst[i] -= ca * a[i] + cb * b[i];
    }
}

/// `dst[i] = (v[i] - r1 * p[i] - r2 * c[i]) / gamma` for `i < n`.
#[cfg(not(target_arch = "wasm32"))]
fn lincomb3_div(
    dst: &mut [f64],
    v: &[f64],
    p: &[f64],
    c: &[f64],
    r1: f64,
    r2: f64,
    gamma: f64,
    n: usize,
) {
    if n >= crate::env::local_rayon_min() {
        dst[..n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, di)| *di = (v[i] - r1 * p[i] - r2 * c[i]) / gamma);
    } else {
        for i in 0..n {
            dst[i] = (v[i] - r1 * p[i] - r2 * c[i]) / gamma;
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn lincomb3_div(
    dst: &mut [f64],
    v: &[f64],
    p: &[f64],
    c: &[f64],
    r1: f64,
    r2: f64,
    gamma: f64,
    n: usize,
) {
    for i in 0..n {
        dst[i] = (v[i] - r1 * p[i] - r2 * c[i]) / gamma;
    }
}

/// `dst[i] += scale * src[i]` for `i < n`.
#[cfg(not(target_arch = "wasm32"))]
fn add_scaled_slice(dst: &mut [f64], src: &[f64], scale: f64, n: usize) {
    if n >= crate::env::local_rayon_min() {
        dst[..n]
            .par_iter_mut()
            .zip(&src[..n])
            .for_each(|(di, si)| *di += scale * si);
    } else {
        for i in 0..n {
            dst[i] += scale * src[i];
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn add_scaled_slice(dst: &mut [f64], src: &[f64], scale: f64, n: usize) {
    for i in 0..n {
        dst[i] += scale * src[i];
    }
}

/// `v[i] /= divisor` for `i < len` when `divisor` is finite and non-zero.
#[cfg(not(target_arch = "wasm32"))]
fn div_assign_slice(v: &mut [f64], divisor: f64, len: usize) {
    if divisor.abs() <= 1e-30 {
        return;
    }
    if len >= crate::env::local_rayon_min() {
        v[..len].par_iter_mut().for_each(|x| *x /= divisor);
    } else {
        for i in 0..len {
            v[i] /= divisor;
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn div_assign_slice(v: &mut [f64], divisor: f64, len: usize) {
    if divisor.abs() <= 1e-30 {
        return;
    }
    for i in 0..len {
        v[i] /= divisor;
    }
}

/// Parallel Conjugate Gradient solver for SPD systems.
///
/// Solves `A x = b` where `A` is a distributed SPD matrix.  All inner
/// products are global reductions over owned DOFs.
pub fn par_solve_cg(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);

    let mut p = r.clone_vec();
    let mut rr = r.global_dot(&r);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        a.spmv(&mut p, &mut ap);

        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rr / pap;

        // x += alpha * p
        x.axpy(alpha, &p);
        // r -= alpha * ap
        r.axpy(-alpha, &ap);

        let rr_new = r.global_dot(&r);
        let res_norm = rr_new.sqrt() / b_norm;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_cg iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr_new.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        let beta = rr_new / rr;
        // p = r + beta * p
        let plen = p.data.len();
        add_scaled_inplace(&mut p.data, &r.data, beta, plen);
        rr = rr_new;
    }

    let final_res = rr.sqrt() / b_norm;
    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}

/// Parallel Jacobi-preconditioned Conjugate Gradient.
///
/// Uses diagonal scaling `M = diag(A)` as the preconditioner.
pub fn par_solve_pcg_jacobi(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;

    // Build inverse diagonal preconditioner.
    let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag.iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);

    // z = M^{-1} r
    let mut z = ParVector::zeros_like(b);
    mul_assign_diag(&mut z.data, &inv_diag, &r.data, n);

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();

    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    let mut ap = ParVector::zeros_like(b);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        a.spmv(&mut p, &mut ap);

        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap;

        // x += alpha * p
        x.axpy(alpha, &p);
        // r -= alpha * ap
        r.axpy(-alpha, &ap);

        let rr = r.global_dot(&r);
        let res_norm = rr.sqrt() / b_norm;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_pcg_jacobi iter {}: residual = {:.3e}", iter + 1, res_norm);
        }

        if res_norm < cfg.rtol || rr.sqrt() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm,
            });
        }

        // z = M^{-1} r
        mul_assign_diag(&mut z.data, &inv_diag, &r.data, n);

        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;

        // p = z + beta * p
        let plen = p.data.len();
        add_scaled_inplace(&mut p.data, &z.data, beta, plen);
        rz = rz_new;
    }

    let final_res = r.global_dot(&r).sqrt() / b_norm;
    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}

/// Parallel restarted GMRES with Jacobi (`M = diag(A)`) right preconditioning.
///
/// Targets general (possibly nonsymmetric) distributed systems. For SPD
/// problems, [`par_solve_pcg_jacobi`] is usually more efficient.
///
/// `restart` is the Krylov subspace dimension before restart (must be `> 0`).
pub fn par_solve_gmres_jacobi(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    if restart == 0 {
        return Err(SolverError::Linlvo("GMRES restart must be > 0".to_string()));
    }

    let n = a.n_owned;
    let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
        .collect();

    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    // r = b - A*x
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    let mut r = b.clone_vec();
    r.axpy(-1.0, &ax);

    let mut iter_total = 0usize;
    let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: rel_res,
        });
    }

    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: 0.0,
            });
        }

        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r);
        v[0].scale(1.0 / beta);

        let mut z_basis: Vec<ParVector> = (0..restart).map(|_| ParVector::zeros_like(b)).collect();
        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart];
        let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1];
        g[0] = beta;

        let mut inner_done = 0usize;
        let mut converged = false;

        for j in 0..restart {
            if iter_total >= cfg.max_iter {
                break;
            }

            // Right preconditioning: z_j = M^{-1} v_j (owned DOFs only; halo refresh in spmv).
            mul_assign_diag(&mut z_basis[j].data, &inv_diag, &v[j].data, n);

            // w = A z_j
            let mut w = ParVector::zeros_like(b);
            a.spmv(&mut z_basis[j], &mut w);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[i][j] = v[i].global_dot(&w);
                w.axpy(-h[i][j], &v[i]);
            }

            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 {
                v[j + 1].copy_from(&w);
                v[j + 1].scale(1.0 / h[j + 1][j]);
            }

            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 {
                cs[j] = h[j][j] / denom;
                sn[j] = h[j + 1][j] / denom;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            h[j + 1][j] = 0.0;

            let g_next = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            g[j + 1] = g_next;

            iter_total += 1;
            inner_done = j + 1;
            rel_res = g[j + 1].abs() / b_norm;

            if cfg.verbose && x.comm().is_root() {
                log::info!("par_gmres_jacobi iter {}: residual = {:.3e}", iter_total, rel_res);
            }

            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol {
                converged = true;
                break;
            }
        }

        if inner_done == 0 {
            break;
        }

        let m = inner_done;
        let mut y = vec![0.0_f64; m];
        for i in (0..m).rev() {
            let mut s = g[i];
            for k in (i + 1)..m {
                s -= h[i][k] * y[k];
            }
            let diag_h = h[i][i];
            if diag_h.abs() < 1e-30 {
                return Err(SolverError::Linlvo(
                    "par_gmres_jacobi breakdown: near-singular Hessenberg diagonal".to_string(),
                ));
            }
            y[i] = s / diag_h;
        }

        for i in 0..m {
            x.axpy(y[i], &z_basis[i]);
        }

        if converged {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: rel_res,
            });
        }

        a.spmv(x, &mut ax);
        r.copy_from(b);
        r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: rel_res,
            });
        }
    }

    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: rel_res,
    })
}

/// Parallel MINRES solver for symmetric (possibly indefinite) systems.
///
/// Solves `A x = b` where `A` is a distributed symmetric matrix.
/// Uses the Lanczos-based MINRES algorithm (Choi-Paige-Saunders).
pub fn par_solve_minres(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);

    let mut beta1 = r.global_norm();
    if beta1 / b_norm < cfg.rtol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: beta1 / b_norm });
    }

    // Lanczos vectors
    let mut v_old = ParVector::zeros_like(b);
    let mut v_cur = r.clone_vec();
    let v_cur_len = v_cur.len();
    div_assign_slice(&mut v_cur.data, beta1, v_cur_len);
    let mut v_new = ParVector::zeros_like(b);

    // MINRES recurrence scalars
    let mut _beta_prev = 0.0_f64;
    let mut beta_cur = beta1;
    let mut c_old = 1.0_f64;
    let mut c_cur = 1.0_f64;
    let mut s_old = 0.0_f64;
    let mut s_cur = 0.0_f64;

    // Direction vectors
    let mut w_prev = ParVector::zeros_like(b);
    let mut w_cur = ParVector::zeros_like(b);

    let mut res_norm = beta1 / b_norm;

    for iter in 0..cfg.max_iter {
        // Lanczos step: v_new = A*v_cur - beta_cur * v_old
        a.spmv(&mut v_cur, &mut v_new);
        let alpha = v_cur.global_dot(&v_new);
        sub_lincomb2(
            &mut v_new.data,
            &v_cur.data,
            alpha,
            &v_old.data,
            beta_cur,
            n,
        );
        let beta_next = v_new.global_norm();

        // Apply previous Givens rotations
        let r1 = s_old * beta_cur;
        let r2 = c_old * c_cur * beta_cur + s_cur * alpha;
        let r3 = -s_old * s_cur * beta_cur + c_cur * alpha;

        // this step's value before new rotation
        let r3_hat = r3;
        let r4 = beta_next;

        // New Givens rotation to zero out beta_next
        let gamma = (r3_hat * r3_hat + r4 * r4).sqrt();
        let c_new = if gamma > 1e-30 { r3_hat / gamma } else { 1.0 };
        let s_new = if gamma > 1e-30 { r4 / gamma } else { 0.0 };

        // Update direction vectors
        let mut w_new = ParVector::zeros_like(b);
        if gamma.abs() > 1e-30 {
            lincomb3_div(
                &mut w_new.data,
                &v_cur.data,
                &w_prev.data,
                &w_cur.data,
                r1,
                r2,
                gamma,
                n,
            );
        }

        // Update solution: x += c_new * beta1 * ... * w_new
        // In MINRES, the update is: x += (c_new * phi) * w_new
        // where phi tracks the residual components
        let _phi = c_new * res_norm * b_norm;
        // Actually, simplified MINRES update:
        let tau = c_new * beta1;
        add_scaled_slice(&mut x.data, &w_new.data, tau, n);

        // Update residual norm
        res_norm = s_new.abs() * res_norm;
        beta1 = s_new * beta1;

        if cfg.verbose && x.comm().is_root() {
            log::info!("par_minres iter {}: residual = {:.3e}", iter + 1, res_norm / b_norm);
        }

        if res_norm / b_norm < cfg.rtol || res_norm < cfg.atol {
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: res_norm / b_norm,
            });
        }

        // Prepare for next iteration
        let vnl = v_new.len();
        div_assign_slice(&mut v_new.data, beta_next, vnl);

        // Shift vectors
        std::mem::swap(&mut v_old, &mut v_cur);
        std::mem::swap(&mut v_cur, &mut v_new);
        v_new = ParVector::zeros_like(b);

        std::mem::swap(&mut w_prev, &mut w_cur);
        w_cur = w_new;

        _beta_prev = beta_cur;
        beta_cur = beta_next;
        c_old = c_cur;
        c_cur = c_new;
        s_old = s_cur;
        s_cur = s_new;
    }

    // Compute true residual
    let mut true_r = b.clone_vec();
    let mut true_ax = ParVector::zeros_like(b);
    a.spmv(x, &mut true_ax);
    sub_assign_owned(&mut true_r.data, &b.data, &true_ax.data, n);
    let final_res = true_r.global_norm() / b_norm;

    Ok(SolveResult {
        converged: false,
        iterations: cfg.max_iter,
        final_residual: final_res,
    })
}


pub fn par_solve_bicgstab(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    let r_hat = r.clone_vec();
    let mut rho = 1.0; let mut alpha = 1.0; let mut omega = 1.0;
    let mut p = ParVector::zeros_like(b);
    let mut ap = ParVector::zeros_like(b);
    for iter in 0..cfg.max_iter {
        let rho_new = r_hat.global_dot(&r);
        if rho_new.abs() < 1e-30 { break; }
        let beta = (rho_new / rho) * (alpha / omega);
        p.axpy(-omega, &ap); p.scale(beta); p.axpy(1.0, &r);
        a.spmv(&mut p, &mut ap);
        let pap = r_hat.global_dot(&ap);
        if pap.abs() < 1e-30 { break; }
        alpha = rho_new / pap;
        let mut s = r.clone_vec(); s.axpy(-alpha, &ap);
        let mut t = ParVector::zeros_like(b); a.spmv(&mut s, &mut t);
        let tau = t.global_dot(&s); let t2 = t.global_norm();
        if t2 < 1e-30 { break; }
        omega = tau / (t2 * t2);
        x.axpy(alpha, &p); x.axpy(omega, &s);
        r.copy_from(&s); r.axpy(-omega, &t);
        let res_norm = r.global_norm() / b_norm;
        if res_norm < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter + 1, final_residual: res_norm });
        }
        rho = rho_new; if omega.abs() < 1e-30 { break; }
    }
    let mut tr = b.clone_vec(); let mut tax = ParVector::zeros_like(b);
    a.spmv(x, &mut tax); sub_assign_owned(&mut tr.data, &b.data, &tax.data, n);
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: tr.global_norm() / b_norm })
}

// ── 7. TFQMR (Transpose-Free QMR) ──────────────────────────────────────────────

/// Parallel TFQMR (Transpose-Free Quasi-Minimal Residual) solver.
///
/// A short-recurrence Krylov method for non-symmetric systems (Freund 1993).
/// Requires 2 matvecs per outer step.  No preconditioner in this entry point.
pub fn par_solve_tfqmr(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let atol = cfg.atol.max(1e-30);
    let rtol = cfg.rtol.max(1e-30);

    // Initial residual r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    let norm_r0 = r.global_norm();
    if norm_r0 <= atol || norm_r0 <= rtol * b_norm {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: norm_r0 / b_norm });
    }

    let r_shadow = r.clone_vec();
    let mut y = r.clone_vec();
    let mut u = ParVector::zeros_like(b);
    a.spmv(&mut y, &mut u);
    let mut v = u.clone_vec();
    let mut w = r.clone_vec();
    let mut d = ParVector::zeros_like(b);

    let mut tau = norm_r0;
    let mut theta = 0.0;
    let mut eta = 0.0;
    let mut rho = r_shadow.global_dot(&r);

    let mut iters = 0;
    for k in 0..cfg.max_iter {
        let sigma = r_shadow.global_dot(&v);
        if sigma.abs() < 1e-30 { break; }
        let alpha = rho / sigma;

        // y_half = y - alpha * v
        let mut y_half = y.clone_vec();
        y_half.axpy(-alpha, &v);

        // ay_half = A * y_half  (matvec 1)
        let mut ay_half = ParVector::zeros_like(b);
        a.spmv(&mut y_half, &mut ay_half);

        // ── Half-step a (odd m = 2k+1) ──
        w.axpy(-alpha, &u);
        let coeff_a = if alpha.abs() < 1e-30 { 0.0 } else { theta * theta * eta / alpha };
        // d = y_half + coeff_a * d
        d.scale(coeff_a); d.axpy(1.0, &y_half);
        let w_norm = w.global_norm();
        theta = w_norm / tau;
        let ca = 1.0 / (1.0 + theta * theta).sqrt();
        tau *= theta * ca;
        eta = ca * ca * alpha;
        x.axpy(eta, &d);

        iters += 1;
        let est_a = tau * ((2 * k + 1) as f64).sqrt();
        let rel_a = est_a / b_norm;
        if rel_a <= rtol || est_a <= atol {
            return Ok(SolveResult { converged: true, iterations: iters, final_residual: rel_a });
        }
        if iters >= cfg.max_iter { break; }

        // ── Half-step b (even m = 2k+2) ──
        w.axpy(-alpha, &ay_half);
        let coeff_b = if alpha.abs() < 1e-30 { 0.0 } else { theta * theta * eta / alpha };
        // d = ay_half + coeff_b * d
        d.scale(coeff_b); d.axpy(1.0, &ay_half);
        let w_norm = w.global_norm();
        theta = w_norm / tau;
        let cb = 1.0 / (1.0 + theta * theta).sqrt();
        tau *= theta * cb;
        eta = cb * cb * alpha;
        x.axpy(eta, &d);

        iters += 1;
        let est_b = tau * ((2 * k + 2) as f64).sqrt();
        let rel_b = est_b / b_norm;
        if rel_b <= rtol || est_b <= atol {
            return Ok(SolveResult { converged: true, iterations: iters, final_residual: rel_b });
        }
        if iters >= cfg.max_iter { break; }

        // ── Update for next outer step ──
        let rho_new = r_shadow.global_dot(&w);
        if rho.abs() < 1e-30 { break; }
        let beta = rho_new / rho;
        rho = rho_new;

        // y = w + beta * y_half
        y.copy_from(&w);
        y.axpy(beta, &y_half);
        // u = A * y  (matvec 2)
        a.spmv(&mut y, &mut u);
        // v = u + beta * ay_half + beta^2 * v  (using temp for v_old)
        let mut v_new = u.clone_vec();
        v_new.axpy(beta, &ay_half);
        v_new.axpy(beta * beta, &v);
        v.copy_from(&v_new);
    }

    // Exact final residual
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    let final_res = r.global_norm() / b_norm;
    Ok(SolveResult { converged: false, iterations: iters, final_residual: final_res })
}

pub fn par_solve_fgmres_jacobi(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector, restart: usize, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    if restart == 0 { return Err(SolverError::Linlvo("FGMRES restart must be > 0".to_string())); }
    let n = a.n_owned; let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag.iter().map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }).collect();
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut ax = ParVector::zeros_like(b); a.spmv(x, &mut ax);
    let mut r = b.clone_vec(); r.axpy(-1.0, &ax);
    let mut iter_total = 0usize; let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: rel_res });
    }
    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 { return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: 0.0 }); }
        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        let mut z: Vec<ParVector> = (0..restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r); v[0].scale(1.0 / beta);
        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart]; let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1]; g[0] = beta;
        for j in 0..restart {
            if iter_total >= cfg.max_iter { break; } iter_total += 1;
            mul_assign_diag(&mut z[j].data, &inv_diag, &v[j].data, n);
            z[j].update_ghosts();
            let mut w = ParVector::zeros_like(b); a.spmv(&mut z[j], &mut w);
            for i in 0..=j { h[i][j] = v[i].global_dot(&w); w.axpy(-h[i][j], &v[i]); }
            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 { v[j + 1].copy_from(&w); v[j + 1].scale(1.0 / h[j + 1][j]); }
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j]; h[i][j] = tmp;
            }
            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 { cs[j] = h[j][j] / denom; sn[j] = h[j + 1][j] / denom; }
            else { cs[j] = 1.0; sn[j] = 0.0; }
            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j];
            rel_res = g[j + 1].abs() / b_norm;
            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol {
                let mut y = vec![0.0_f64; j + 1];
                for k in (0..=j).rev() { y[k] = g[k]; for kk in k + 1..=j { y[k] -= h[k][kk] * y[kk]; }
                    if h[k][k].abs() > 1e-30 { y[k] /= h[k][k]; } }
                for k in 0..=j { if y[k].abs() > 1e-30 { for i in 0..n { x.data[i] += y[k] * z[k].data[i]; } } }
                return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
            }
        }
        let mut y = vec![0.0_f64; restart];
        for k in (0..restart).rev() { y[k] = g[k]; for kk in k + 1..restart { y[k] -= h[k][kk] * y[kk]; }
            if h[k][k].abs() > 1e-30 { y[k] /= h[k][k]; } }
        for k in 0..restart { if y[k].abs() > 1e-30 { for i in 0..n { x.data[i] += y[k] * z[k].data[i]; } } }
        a.spmv(x, &mut ax); r.copy_from(b); r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
        }
    }
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: rel_res })
}

pub struct ParIlu0Precond { ilu: linlvo::precond::Ilu0Precond<f64>, n_owned: usize }
impl ParIlu0Precond {
    pub fn new(a: &ParCsrMatrix) -> Self {
        let n_owned = a.n_owned; let local = &a.diag;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned, n_owned);
        for i in 0..n_owned.min(local.nrows) {
            let s = local.row_ptr[i]; let e = local.row_ptr[i + 1];
            for k in s..e { let j = local.col_idx[k] as usize; if j < n_owned { coo.add(i, j, local.values[k]); } }
        }
        let diag = coo.into_csr(); let la = fem_solver::fem_to_linlvo_csr(&diag);
        let ilu = linlvo::precond::Ilu0Precond::from_csr(&la).expect("ILU(0) for local diag block");
        ParIlu0Precond { ilu, n_owned }
    }
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        use linlvo::Preconditioner;
        let xv = linlvo::DenseVec::from_vec(r[..self.n_owned].to_vec());
        let mut yv = linlvo::DenseVec::zeros(self.n_owned);
        self.ilu.apply_precond(&xv, &mut yv);
        z[..self.n_owned].copy_from_slice(yv.as_slice());
    }
}

/// Distributed ILU(k) preconditioner built from the local diagonal block.
///
/// `fill_level = 0` reproduces ILU(0); higher values allow more fill.
pub struct ParIlukPrecond {
    ilu: linlvo::precond::IlukPrecond<f64>,
    n_owned: usize,
}
impl ParIlukPrecond {
    pub fn new(a: &ParCsrMatrix, fill_level: usize) -> Self {
        let n_owned = a.n_owned;
        let local = &a.diag;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned, n_owned);
        for i in 0..n_owned.min(local.nrows) {
            let s = local.row_ptr[i];
            let e = local.row_ptr[i + 1];
            for k in s..e {
                let j = local.col_idx[k] as usize;
                if j < n_owned {
                    coo.add(i, j, local.values[k]);
                }
            }
        }
        let diag = coo.into_csr();
        let la = fem_solver::fem_to_linlvo_csr(&diag);
        let ilu =
            linlvo::precond::IlukPrecond::from_csr(&la, fill_level).expect("ILU(k) for local diag block");
        ParIlukPrecond { ilu, n_owned }
    }
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        use linlvo::Preconditioner;
        let xv = linlvo::DenseVec::from_vec(r[..self.n_owned].to_vec());
        let mut yv = linlvo::DenseVec::zeros(self.n_owned);
        self.ilu.apply_precond(&xv, &mut yv);
        z[..self.n_owned].copy_from_slice(yv.as_slice());
    }
}

/// Distributed ILUT(τ, p) preconditioner built from the local diagonal block.
///
/// * `tau` — relative drop tolerance (e.g. `0.01`)
/// * `p_fill` — max off-diagonal entries per row in L and U
pub struct ParIlutPrecond {
    ilu: linlvo::precond::IlutPrecond<f64>,
    n_owned: usize,
}
impl ParIlutPrecond {
    pub fn new(a: &ParCsrMatrix, tau: f64, p_fill: usize) -> Self {
        let n_owned = a.n_owned;
        let local = &a.diag;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned, n_owned);
        for i in 0..n_owned.min(local.nrows) {
            let s = local.row_ptr[i];
            let e = local.row_ptr[i + 1];
            for k in s..e {
                let j = local.col_idx[k] as usize;
                if j < n_owned {
                    coo.add(i, j, local.values[k]);
                }
            }
        }
        let diag = coo.into_csr();
        let la = fem_solver::fem_to_linlvo_csr(&diag);
        let ilu =
            linlvo::precond::IlutPrecond::from_csr(&la, tau, p_fill).expect("ILUT for local diag block");
        ParIlutPrecond { ilu, n_owned }
    }
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        use linlvo::Preconditioner;
        let xv = linlvo::DenseVec::from_vec(r[..self.n_owned].to_vec());
        let mut yv = linlvo::DenseVec::zeros(self.n_owned);
        self.ilu.apply_precond(&xv, &mut yv);
        z[..self.n_owned].copy_from_slice(yv.as_slice());
    }
}

pub fn par_solve_pcg_ilu0(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector, ilu: &ParIlu0Precond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned; let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut r = b.clone_vec(); let mut ax = ParVector::zeros_like(b); a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    let mut z = ParVector::zeros_like(b); ilu.apply(&r.data, &mut z.data); z.update_ghosts();
    let mut p = z.clone_vec(); let mut rz = r.global_dot(&z); let mut ap = ParVector::zeros_like(b);
    for iter in 0..cfg.max_iter {
        a.spmv(&mut p, &mut ap); let pap = p.global_dot(&ap); if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap; x.axpy(alpha, &p); r.axpy(-alpha, &ap);
        let res_norm = r.global_norm() / b_norm;
        if res_norm < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter + 1, final_residual: res_norm });
        }
        ilu.apply(&r.data, &mut z.data); z.update_ghosts();
        let rz_new = r.global_dot(&z); let beta = rz_new / rz;
        p.scale(beta); p.axpy(1.0, &z); rz = rz_new;
    }
    let mut tr = b.clone_vec(); let mut tax = ParVector::zeros_like(b);
    a.spmv(x, &mut tax); sub_assign_owned(&mut tr.data, &b.data, &tax.data, n);
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: tr.global_norm() / b_norm })
}

/// PCG with ILU(k) preconditioner (symmetric positive definite).
///
/// `fill_level = 0` reproduces `par_solve_pcg_ilu0`.
pub fn par_solve_pcg_iluk(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    ilu: &ParIlukPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned; let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut r = b.clone_vec(); let mut ax = ParVector::zeros_like(b); a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    let mut z = ParVector::zeros_like(b); ilu.apply(&r.data, &mut z.data); z.update_ghosts();
    let mut p = z.clone_vec(); let mut rz = r.global_dot(&z); let mut ap = ParVector::zeros_like(b);
    for iter in 0..cfg.max_iter {
        a.spmv(&mut p, &mut ap); let pap = p.global_dot(&ap); if pap.abs() < 1e-30 { break; }
        let alpha = rz / pap; x.axpy(alpha, &p); r.axpy(-alpha, &ap);
        let res_norm = r.global_norm() / b_norm;
        if res_norm < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter + 1, final_residual: res_norm });
        }
        ilu.apply(&r.data, &mut z.data); z.update_ghosts();
        let rz_new = r.global_dot(&z); let beta = rz_new / rz;
        p.scale(beta); p.axpy(1.0, &z); rz = rz_new;
    }
    let mut tr = b.clone_vec(); let mut tax = ParVector::zeros_like(b);
    a.spmv(x, &mut tax); sub_assign_owned(&mut tr.data, &b.data, &tax.data, n);
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: tr.global_norm() / b_norm })
}

/// GMRES with ILU(0) right preconditioner.
pub fn par_solve_gmres_ilu0(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlu0Precond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_gmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// GMRES with ILU(k) right preconditioner.
pub fn par_solve_gmres_iluk(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlukPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_gmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// GMRES with ILUT right preconditioner.
pub fn par_solve_gmres_ilut(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlutPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_gmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// Core GMRES + ILU right-preconditioned implementation (shared by ILU0/ILUK/ILUT).
///
/// `apply_precond(r, z)` solves `M z_j = v_j` (owned DOFs).  The caller must
/// ensure ghosts are updated after each application — this function calls
/// `update_ghosts` on the preconditioned basis vectors automatically.
fn par_solve_gmres_ilu_precond<F>(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, cfg: &SolverConfig,
    apply_precond: F,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    if restart == 0 {
        return Err(SolverError::Linlvo("GMRES restart must be > 0".to_string()));
    }
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut ax = ParVector::zeros_like(b); a.spmv(x, &mut ax);
    let mut r = b.clone_vec(); r.axpy(-1.0, &ax);
    let mut iter_total = 0usize;
    let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: rel_res });
    }
    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 { return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: 0.0 }); }
        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r); v[0].scale(1.0 / beta);
        let mut z_basis: Vec<ParVector> = (0..restart).map(|_| ParVector::zeros_like(b)).collect();
        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart];
        let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1]; g[0] = beta;
        let mut inner_done = 0usize;
        let mut converged = false;
        for j in 0..restart {
            if iter_total >= cfg.max_iter { break; }
            // Right preconditioning: z_j = M^{-1} v_j
            apply_precond(&v[j].data, &mut z_basis[j].data);
            z_basis[j].update_ghosts();
            let mut w = ParVector::zeros_like(b); a.spmv(&mut z_basis[j], &mut w);
            for i in 0..=j { h[i][j] = v[i].global_dot(&w); w.axpy(-h[i][j], &v[i]); }
            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 { v[j + 1].copy_from(&w); v[j + 1].scale(1.0 / h[j + 1][j]); }
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j]; h[i][j] = tmp;
            }
            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 { cs[j] = h[j][j] / denom; sn[j] = h[j + 1][j] / denom; }
            else { cs[j] = 1.0; sn[j] = 0.0; }
            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j]; h[j + 1][j] = 0.0;
            let g_next = -sn[j] * g[j]; g[j] = cs[j] * g[j]; g[j + 1] = g_next;
            iter_total += 1; inner_done = j + 1;
            rel_res = g[j + 1].abs() / b_norm;
            if cfg.verbose && x.comm().is_root() {
                log::info!("par_gmres_ilu iter {}: residual = {:.3e}", iter_total, rel_res);
            }
            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol { converged = true; break; }
        }
        if inner_done == 0 { break; }
        let m = inner_done;
        let mut y = vec![0.0_f64; m];
        for i in (0..m).rev() {
            let mut s = g[i]; for k in (i + 1)..m { s -= h[i][k] * y[k]; }
            if h[i][i].abs() < 1e-30 {
                return Err(SolverError::Linlvo("GMRES+ILU breakdown: near-singular Hessenberg diagonal".to_string()));
            }
            y[i] = s / h[i][i];
        }
        for i in 0..m { x.axpy(y[i], &z_basis[i]); }
        if converged { return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res }); }
        a.spmv(x, &mut ax); r.copy_from(b); r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
        }
    }
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: rel_res })
}

// ── FGMRES + ILU(0/k/UT) ──────────────────────────────────────────────────

/// FGMRES with ILU(0) right preconditioner.
pub fn par_solve_fgmres_ilu0(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlu0Precond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_fgmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// FGMRES with ILU(k) right preconditioner.
pub fn par_solve_fgmres_iluk(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlukPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_fgmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// FGMRES with ILUT right preconditioner.
pub fn par_solve_fgmres_ilut(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, ilu: &ParIlutPrecond, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    par_solve_fgmres_ilu_precond(a, b, x, restart, cfg, |r, z| ilu.apply(r, z))
}

/// Core FGMRES + ILU right-preconditioned implementation (shared by ILU0/ILUK/ILUT).
fn par_solve_fgmres_ilu_precond<F>(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector,
    restart: usize, cfg: &SolverConfig,
    apply_precond: F,
) -> Result<SolveResult, SolverError>
where
    F: Fn(&[f64], &mut [f64]),
{
    if restart == 0 { return Err(SolverError::Linlvo("FGMRES restart must be > 0".to_string())); }
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let mut ax = ParVector::zeros_like(b); a.spmv(x, &mut ax);
    let mut r = b.clone_vec(); r.axpy(-1.0, &ax);
    let mut iter_total = 0usize; let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: rel_res });
    }
    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 { return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: 0.0 }); }
        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        let mut z: Vec<ParVector> = (0..restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r); v[0].scale(1.0 / beta);
        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart]; let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1]; g[0] = beta;
        for j in 0..restart {
            if iter_total >= cfg.max_iter { break; } iter_total += 1;
            apply_precond(&v[j].data, &mut z[j].data);
            z[j].update_ghosts();
            let mut w = ParVector::zeros_like(b); a.spmv(&mut z[j], &mut w);
            for i in 0..=j { h[i][j] = v[i].global_dot(&w); w.axpy(-h[i][j], &v[i]); }
            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 { v[j + 1].copy_from(&w); v[j + 1].scale(1.0 / h[j + 1][j]); }
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j]; h[i][j] = tmp;
            }
            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 { cs[j] = h[j][j] / denom; sn[j] = h[j + 1][j] / denom; }
            else { cs[j] = 1.0; sn[j] = 0.0; }
            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j];
            rel_res = g[j + 1].abs() / b_norm;
            if cfg.verbose && x.comm().is_root() {
                log::info!("par_fgmres_ilu iter {}: residual = {:.3e}", iter_total, rel_res);
            }
            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol {
                let mut y = vec![0.0_f64; j + 1];
                for k in (0..=j).rev() {
                    y[k] = g[k];
                    for kk in k + 1..=j { y[k] -= h[k][kk] * y[kk]; }
                    if h[k][k].abs() > 1e-30 { y[k] /= h[k][k]; }
                }
                for k in 0..=j { if y[k].abs() > 1e-30 { for i in 0..a.n_owned { x.data[i] += y[k] * z[k].data[i]; } } }
                return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
            }
        }
        let mut y = vec![0.0_f64; restart];
        for k in (0..restart).rev() {
            y[k] = g[k];
            for kk in k + 1..restart { y[k] -= h[k][kk] * y[kk]; }
            if h[k][k].abs() > 1e-30 { y[k] /= h[k][k]; }
        }
        for k in 0..restart { if y[k].abs() > 1e-30 { for i in 0..a.n_owned { x.data[i] += y[k] * z[k].data[i]; } } }
        a.spmv(x, &mut ax); r.copy_from(b); r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
        }
    }
    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: rel_res })
}

// ── 10. IDR(s) — Induced Dimension Reduction ────────────────────────────────

/// Parallel IDR(s) for non-symmetric systems.
///
/// Uses `s` sequential projection steps per outer iteration.  Each step
/// projects the residual along `r` against `A*r`, reducing components
/// in an induced-dimension fashion.
///
/// `s = 4` is a good default.  For SPD systems [`par_solve_cg`] or
/// [`par_solve_pcg_jacobi`] is preferred.
pub fn par_solve_idrs(
    a: &ParCsrMatrix, b: &ParVector, x: &mut ParVector, s: usize, cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.n_owned;
    let b_norm = b.global_norm();
    if b_norm < 1e-30 { return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 }); }
    let atol = cfg.atol.max(1e-30);
    let rtol = cfg.rtol.max(1e-30);

    // r = b - A*x
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);

    let mut r_norm = r.global_norm();
    if r_norm <= atol || r_norm <= rtol * b_norm {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: r_norm / b_norm });
    }

    for iter in 0..cfg.max_iter {
        // s inner subspace projection steps
        for col in 0..s {
            let mut f = r.clone_vec();
            let mut af = ParVector::zeros_like(b);
            a.spmv(&mut f, &mut af);

            // Step length: minimise ||r - lambda*A*f||
            // First step uses r·r; subsequent steps use r·f (different subspaces).
            let fdot = if col == 0 {
                r.global_dot(&r)
            } else {
                r.global_dot(&f)
            };
            let af_dot = f.global_dot(&af);
            if af_dot.abs() < 1e-30 { break; }
            let lambda = fdot / af_dot;

            x.axpy(lambda, &f);
            r.axpy(-lambda, &af);

            r_norm = r.global_norm();
            if r_norm <= atol || r_norm <= rtol * b_norm {
                return Ok(SolveResult { converged: true, iterations: iter + 1, final_residual: r_norm / b_norm });
            }
        }
    }

    // Exact final residual
    a.spmv(x, &mut ax);
    sub_assign_owned(&mut r.data, &b.data, &ax.data, n);
    Ok(SolveResult {
        converged: false, iterations: cfg.max_iter,
        final_residual: r.global_norm() / b_norm,
    })
}

// ── 9. Distributed direct solve ─────────────────────────────────────────────

/// Solve Ax = b via distributed direct LU with full gather to rank 0.
///
/// Each rank sends its local matrix rows (diag + off-diagonal blocks) and RHS
/// to rank 0, which assembles the global system, performs sparse LU
/// factorization, and broadcasts the solution back to all ranks.
///
/// # Arguments
/// * `a`               — parallel CSR matrix
/// * `b`               — parallel RHS vector
/// * `x`               — output solution vector (written on all ranks)
/// * `global_dof_ids`  — mapping `local_dof_id -> global_dof_id` from
///                        `DofPartition::global_dof_ids`; length `n_owned + n_ghost`
///
/// This is suitable for small-to-moderate problems (up to ~1e5 DOFs) where
/// iterative solver convergence is unreliable.
///
/// # Panics
/// Panics if the serial LU solve fails.
pub fn par_direct_solve(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    global_dof_ids: &[u32],
) -> Result<SolveResult, SolverError> {
    let comm = a.comm();
    let n_ranks = comm.size();
    let rank = comm.rank();
    let n_owned = a.n_owned;
    let n_ghost = a.n_ghost;

    // ── Step 1: Each rank serialises its local matrix as COO triplets ──────
    // with global (row, col) indices, plus its owned RHS segment.
    //
    // Triplet format on wire: flat [global_row, global_col, value] interleaved.
    // We build triplets from diag + offd blocks with global indexing.

    let offset = if global_dof_ids.is_empty() {
        0
    } else {
        // The offset is the first owned DOF's global ID.
        global_dof_ids[0]
    };

    // Owned RHS segment
    let local_rhs: Vec<f64> = b.owned_slice().to_vec();

    // Build COO triplets from diag block: (offset+i, offset+j, val)
    let mut triplets: Vec<(u32, u32, f64)> = Vec::new();
    for i in 0..n_owned {
        let gr = offset + i as u32;
        for k in a.diag.row_ptr[i]..a.diag.row_ptr[i + 1] {
            let lc = a.diag.col_idx[k] as usize;
            let val = a.diag.values[k];
            if val.abs() >= 1e-30 {
                triplets.push((gr, offset + lc as u32, val));
            }
        }
    }

    // Offdiagonal block: column j maps to local ghost n_owned + j
    if n_ghost > 0 {
        for i in 0..n_owned {
            let gr = offset + i as u32;
            for k in a.offd.row_ptr[i]..a.offd.row_ptr[i + 1] {
                let gj = a.offd.col_idx[k] as usize; // ghost local index
                let val = a.offd.values[k];
                if val.abs() >= 1e-30 {
                    // Map ghost index to global DOF via global_dof_ids
                    let gc = global_dof_ids[n_owned + gj];
                    triplets.push((gr, gc, val));
                }
            }
        }
    }

    // ── Step 2: Gather all data to rank 0 ──────────────────────────────────
    // First, allgather each rank's triplet count and RHS length via send to rank 0.
    let n_triplets = triplets.len();
    let n_rhs = local_rhs.len();

    let mut all_n_triplets = vec![0usize; n_ranks];
    let mut all_n_rhs = vec![0usize; n_ranks];

    if rank == 0 {
        all_n_triplets[0] = n_triplets;
        all_n_rhs[0] = n_rhs;
        for src in 1..n_ranks {
            let buf = comm.recv_bytes(src as i32, 0x7000);
            let nt = usize::from_le_bytes(buf[..8].try_into().unwrap());
            let nr = usize::from_le_bytes(buf[8..16].try_into().unwrap());
            all_n_triplets[src] = nt;
            all_n_rhs[src] = nr;
        }
    } else {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&n_triplets.to_le_bytes());
        buf.extend_from_slice(&n_rhs.to_le_bytes());
        comm.send_bytes(0, 0x7000, &buf);
    }

    // Barrier to ensure all size info is received before the payload.
    comm.barrier();

    // Compute total sizes
    let total_triplets: usize = all_n_triplets.iter().sum();
    let total_rhs: usize = all_n_rhs.iter().sum();

    // ── Step 3: Send/recv actual data ──────────────────────────────────────
    // Serialise triplets: flat [row_u32, col_u32, val_f64] × n_triplets.
    let triplet_bytes: Vec<u8> = triplets
        .iter()
        .flat_map(|&(r, c, v)| {
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&r.to_le_bytes());
            b.extend_from_slice(&c.to_le_bytes());
            b.extend_from_slice(&v.to_le_bytes());
            b
        })
        .collect();

    let rhs_bytes: Vec<u8> = local_rhs
        .iter()
        .flat_map(|&v| v.to_le_bytes().to_vec())
        .collect();

    // Gather triplets to rank 0
    let all_triplets: Vec<u8>;
    let all_rhs_bytes: Vec<u8>;

    if rank == 0 {
        let mut combined_t = Vec::with_capacity(total_triplets * 16);
        combined_t.extend_from_slice(&triplet_bytes);
        for src in 1..n_ranks {
            let chunk = comm.recv_bytes(src as i32, 0x7001);
            combined_t.extend_from_slice(&chunk);
        }
        all_triplets = combined_t;

        let mut combined_r = Vec::with_capacity(total_rhs * 8);
        combined_r.extend_from_slice(&rhs_bytes);
        for src in 1..n_ranks {
            let chunk = comm.recv_bytes(src as i32, 0x7002);
            combined_r.extend_from_slice(&chunk);
        }
        all_rhs_bytes = combined_r;
    } else {
        comm.send_bytes(0, 0x7001, &triplet_bytes);
        comm.send_bytes(0, 0x7002, &rhs_bytes);
        all_triplets = Vec::new();
        all_rhs_bytes = Vec::new();
    }

    // ── Step 4: Rank 0 builds global matrix and solves ─────────────────────
        let global_n = total_rhs; // total owned DOFs across all ranks

    let global_x: Vec<f64> = if rank == 0 {
        // Deserialise triplets into a COO matrix
        let _n_triplet_values = all_triplets.len() / 16;
        let mut coo = fem_linalg::CooMatrix::<f64>::new(global_n, global_n);
        for chunk in all_triplets.chunks_exact(16) {
            let r = u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize;
            let c = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as usize;
            let v = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            if v.abs() >= 1e-30 {
                coo.add(r, c, v);
            }
        }

        // Build global RHS
        let mut global_rhs = vec![0.0_f64; global_n];
        let mut rhs_offset = 0usize;
        for src in 0..n_ranks {
            let n_src = all_n_rhs[src];
            for j in 0..n_src {
                let start = (rhs_offset + j) * 8;
                let val = f64::from_le_bytes(
                    all_rhs_bytes[start..start + 8].try_into().unwrap()
                );
                // Assign to the correct global position: the offset of rank src
                let dof_pos = if src == 0 { 0 } else {
                    all_n_rhs[..src].iter().sum()
                };
                global_rhs[dof_pos + j] = val;
            }
            rhs_offset += n_src;
        }

        // Convert to CSR and solve
        let global_mat: fem_linalg::CsrMatrix<f64> = coo.into_csr();

        let sol = fem_solver::solve_sparse_lu(&global_mat, &global_rhs)
            .map_err(|e| SolverError::Linlvo(format!("distributed direct solve: {e}")))?;
        sol
    } else {
        Vec::new()
    };

    // ── Step 5: Broadcast solution from rank 0 to all ranks ────────────────
    comm.barrier();
    let mut sol_bytes = if rank == 0 {
        let mut bytes = Vec::with_capacity(global_n * 8);
        for &v in &global_x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    } else {
        vec![0u8; global_n * 8]
    };
    comm.broadcast_bytes(0, &mut sol_bytes);

    // Deserialise solution on all ranks
    let global_solution: Vec<f64> = sol_bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Copy owned portion into output vector
    for i in 0..n_owned {
        x.data[i] = global_solution[offset as usize + i];
    }

    Ok(SolveResult {
        converged: true,
        iterations: 1,
        final_residual: 0.0,
    })
}

/// Parallel restarted GMRES (unpreconditioned) for general non-symmetric systems.
///
/// No preconditioner is applied. For most problems preconditioned variants
/// ([`par_solve_gmres_jacobi`], [`par_solve_gmres_ilu0`], etc.) converge faster.
///
/// `restart` is the Krylov subspace dimension before restart (must be `> 0`).
pub fn par_solve_gmres(
    a: &ParCsrMatrix,
    b: &ParVector,
    x: &mut ParVector,
    restart: usize,
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    if restart == 0 {
        return Err(SolverError::Linlvo("GMRES restart must be > 0".to_string()));
    }

    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return Ok(SolveResult { converged: true, iterations: 0, final_residual: 0.0 });
    }

    // r = b - A*x
    let mut ax = ParVector::zeros_like(b);
    a.spmv(x, &mut ax);
    let mut r = b.clone_vec();
    r.axpy(-1.0, &ax);

    let mut iter_total = 0usize;
    let mut rel_res = r.global_norm() / b_norm;
    if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
        return Ok(SolveResult {
            converged: true,
            iterations: 0,
            final_residual: rel_res,
        });
    }

    while iter_total < cfg.max_iter {
        let beta = r.global_norm();
        if beta < 1e-30 {
            return Ok(SolveResult {
                converged: true,
                iterations: iter_total,
                final_residual: 0.0,
            });
        }

        let mut v: Vec<ParVector> = (0..=restart).map(|_| ParVector::zeros_like(b)).collect();
        v[0].copy_from(&r);
        v[0].scale(1.0 / beta);

        let mut h = vec![vec![0.0_f64; restart]; restart + 1];
        let mut cs = vec![0.0_f64; restart];
        let mut sn = vec![0.0_f64; restart];
        let mut g = vec![0.0_f64; restart + 1];
        g[0] = beta;

        let mut inner_done = 0usize;
        let mut converged = false;

        for j in 0..restart {
            if iter_total >= cfg.max_iter { break; }

            // w = A v_j (no preconditioner)
            let mut w = ParVector::zeros_like(b);
            a.spmv(&mut v[j], &mut w);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[i][j] = v[i].global_dot(&w);
                w.axpy(-h[i][j], &v[i]);
            }

            h[j + 1][j] = w.global_norm();
            if h[j + 1][j] > 1e-30 {
                v[j + 1].copy_from(&w);
                v[j + 1].scale(1.0 / h[j + 1][j]);
            }

            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            let denom = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
            if denom > 1e-30 {
                cs[j] = h[j][j] / denom;
                sn[j] = h[j + 1][j] / denom;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
            h[j + 1][j] = 0.0;

            let g_next = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            g[j + 1] = g_next;

            iter_total += 1;
            inner_done = j + 1;
            rel_res = g[j + 1].abs() / b_norm;

            if cfg.verbose && x.comm().is_root() {
                log::info!("par_gmres iter {}: residual = {:.3e}", iter_total, rel_res);
            }

            if rel_res < cfg.rtol || g[j + 1].abs() < cfg.atol {
                converged = true;
                break;
            }
        }

        if inner_done == 0 { break; }

        let m = inner_done;
        let mut y = vec![0.0_f64; m];
        for i in (0..m).rev() {
            let mut s = g[i];
            for k in (i + 1)..m { s -= h[i][k] * y[k]; }
            let diag_h = h[i][i];
            if diag_h.abs() < 1e-30 {
                return Err(SolverError::Linlvo(
                    "par_gmres breakdown: near-singular Hessenberg diagonal".to_string(),
                ));
            }
            y[i] = s / diag_h;
        }

        // Unpreconditioned GMRES: update from V basis
        for i in 0..m { x.axpy(y[i], &v[i]); }

        if converged {
            return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
        }

        a.spmv(x, &mut ax);
        r.copy_from(b);
        r.axpy(-1.0, &ax);
        rel_res = r.global_norm() / b_norm;
        if rel_res < cfg.rtol || r.global_norm() < cfg.atol {
            return Ok(SolveResult { converged: true, iterations: iter_total, final_residual: rel_res });
        }
    }

    Ok(SolveResult { converged: false, iterations: cfg.max_iter, final_residual: rel_res })
}

// ── ParIlu0Precond, ParIlukPrecond, ParIlutPrecond ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::coefficient::ConstantVectorCoeff;
    use fem_assembly::standard::{ConvectionIntegrator, DiffusionIntegrator, DomainSourceIntegrator};
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn par_cg_laplacian_serial() {
        // Single-rank parallel CG on a simple Poisson problem.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            // Apply Dirichlet BCs: u=0 on boundary.
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_cg(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged, "CG did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_jacobi_two_ranks() {
        // Two-rank parallel PCG on Poisson.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            // Apply Dirichlet BCs: u=0 on boundary.
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_gmres_jacobi_two_ranks() {
        // Two-rank parallel GMRES+Jacobi on Poisson (SPD; exercises GMRES path).
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 500, ..SolverConfig::default() };
            let res = par_solve_gmres_jacobi(&a_mat, &rhs, &mut u, 30, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: GMRES+Jacobi did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_gmres_jacobi_conv_diff_two_ranks() {
        // Convection–diffusion (nonsymmetric): diffusion dominates for a stable solve.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let conv = ConvectionIntegrator {
                velocity: ConstantVectorCoeff(vec![0.2, 0.0]),
            };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff, &conv], 3);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);

            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs {
                let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs {
                    a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-7, max_iter: 800, ..SolverConfig::default() };
            let res = par_solve_gmres_jacobi(&a_mat, &rhs, &mut u, 40, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: GMRES+Jacobi (conv–diff) did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    #[test]
    fn par_pcg_jacobi_p2_two_ranks() {
        // Two-rank parallel PCG on Poisson with P2 elements.
        let mesh = SimplexMesh::<2>::unit_square_tri(8);

        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = fem_space::dof_manager::DofManager::new(&local_mesh, 2);
            let local_space = H1Space::new(local_mesh, 2);
            let par_space = ParallelFESpace::new_with_dof_manager(
                local_space, &pmesh, &dm, comm.clone(),
            );

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 4);

            let source = fem_assembly::standard::DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 5);

            // Apply Dirichlet BCs: u=0 on boundary.
            // boundary_dofs returns DOF IDs in DofManager numbering.
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(),
                par_space.local_space().dof_manager(), &[1, 2, 3, 4]);
            let dof_part = par_space.dof_partition();
            for &d in &bc_dofs {
                let pid = dof_part.permute_dof(d) as usize;
                if pid < dof_part.n_owned_dofs {
                    a_mat.apply_dirichlet_row(pid, 0.0, &mut rhs.data);
                }
            }

            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, ..SolverConfig::default() };
            let res = par_solve_pcg_jacobi(&a_mat, &rhs, &mut u, &cfg).unwrap();

            assert!(res.converged,
                "rank {}: P2 PCG did not converge: {} iters, res={:.3e}",
                comm.rank(), res.iterations, res.final_residual);
        });
    }

    // ── BiCGStab (two ranks) ───────────────────────────────────────────────

    #[test]
    fn par_bicgstab_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 500, ..SolverConfig::default() };
            let res = par_solve_bicgstab(&a_mat, &rhs, &mut u, &cfg).unwrap();
            assert!(res.converged, "BiCGStab did not converge");
        });
    }

    // ── TFQMR (two ranks) ──────────────────────────────────────────────────

    #[test]
    fn par_tfqmr_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 1000, ..SolverConfig::default() };
            let res = par_solve_tfqmr(&a_mat, &rhs, &mut u, &cfg).unwrap();
            assert!(res.converged, "TFQMR did not converge: iters={}, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── IDR(4) (two ranks) ─────────────────────────────────────────────────

    #[test]
    fn par_idrs_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-7, max_iter: 500, ..SolverConfig::default() };
            let res = par_solve_idrs(&a_mat, &rhs, &mut u, 4, &cfg).unwrap();
            assert!(res.converged, "IDR(4) on Poisson did not converge: iters={}, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── FGMRES+Jacobi (two ranks) ──────────────────────────────────────────

    #[test]
    fn par_fgmres_jacobi_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 500, ..SolverConfig::default() };
            let res = par_solve_fgmres_jacobi(&a_mat, &rhs, &mut u, 30, &cfg).unwrap();
            assert!(res.converged, "FGMRES+Jacobi did not converge");
        });
    }

    // ── PCG+ILU0 (two ranks) ───────────────────────────────────────────────

    #[test]
    fn par_pcg_ilu0_poisson_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlu0Precond::new(&a_mat);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_pcg_ilu0(&a_mat, &rhs, &mut u, &ilu, &cfg).unwrap();
            assert!(res.converged, "PCG+ILU0 did not converge");
        });
    }

    // ── GMRES (unpreconditioned, single rank) ───────────────────────────────

    #[test]
    fn par_gmres_laplacian_serial() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 2);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-7, max_iter: 1000, ..SolverConfig::default() };
            let res = par_solve_gmres(&a_mat, &rhs, &mut u, 30, &cfg).unwrap();
            assert!(res.converged, "GMRES did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── GMRES+ILU0 (two ranks) ──────────────────────────────────────────────

    #[test]
    fn par_gmres_ilu0_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlu0Precond::new(&a_mat);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_gmres_ilu0(&a_mat, &rhs, &mut u, 30, &ilu, &cfg).unwrap();
            assert!(res.converged, "GMRES+ILU0 did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── GMRES+ILU(1) (two ranks) ─────────────────────────────────────────────

    #[test]
    fn par_gmres_iluk1_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlukPrecond::new(&a_mat, 1);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_gmres_iluk(&a_mat, &rhs, &mut u, 30, &ilu, &cfg).unwrap();
            assert!(res.converged, "GMRES+ILU(1) did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── GMRES+ILUT (two ranks) ──────────────────────────────────────────────

    #[test]
    fn par_gmres_ilut_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlutPrecond::new(&a_mat, 0.01, 10);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_gmres_ilut(&a_mat, &rhs, &mut u, 30, &ilu, &cfg).unwrap();
            assert!(res.converged, "GMRES+ILUT did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── FGMRES+ILU0 (two ranks) ─────────────────────────────────────────────

    #[test]
    fn par_fgmres_ilu0_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlu0Precond::new(&a_mat);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_fgmres_ilu0(&a_mat, &rhs, &mut u, 30, &ilu, &cfg).unwrap();
            assert!(res.converged, "FGMRES+ILU0 did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }

    // ── PCG+ILU(2) (two ranks) ──────────────────────────────────────────────

    #[test]
    fn par_pcg_iluk2_two_ranks() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_simplex(&mesh, &comm);
            let local_space = H1Space::new(pmesh.local_mesh().clone(), 1);
            let par_space = ParallelFESpace::new(local_space, &pmesh, comm.clone());
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&diff], 3);
            let source = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
            let mut rhs = ParAssembler::assemble_linear(&par_space, &[&source], 3);
            let dm = par_space.local_space().dof_manager();
            let bc_dofs = boundary_dofs(par_space.local_space().mesh(), dm, &[1, 2, 3, 4]);
            for &d in &bc_dofs { let lid = d as usize;
                if lid < par_space.dof_partition().n_owned_dofs { a_mat.apply_dirichlet_row(lid, 0.0, &mut rhs.data); } }
            let ilu = ParIlukPrecond::new(&a_mat, 2);
            let mut u = ParVector::zeros(&par_space);
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 200, ..SolverConfig::default() };
            let res = par_solve_pcg_iluk(&a_mat, &rhs, &mut u, &ilu, &cfg).unwrap();
            assert!(res.converged, "PCG+ILU(2) did not converge: {} iters, res={:.3e}",
                res.iterations, res.final_residual);
        });
    }
}
