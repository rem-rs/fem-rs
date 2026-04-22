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
        return Err(SolverError::Linger("GMRES restart must be > 0".to_string()));
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
                return Err(SolverError::Linger(
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_assembler::ParAssembler;
    use crate::par_simplex::partition_simplex;
    use crate::par_space::ParallelFESpace;
    use fem_assembly::coefficient::ConstantVectorCoeff;
    use fem_assembly::standard::{ConvectionIntegrator, DiffusionIntegrator};
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
}
