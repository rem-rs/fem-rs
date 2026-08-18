//! Parallel LOBPCG eigenvalue solver.
//!
//! Solves `A X = B X Λ` for the smallest eigenvalues.  This is a 1:1 port of
//! the serial `fem_solver::eigen::lobpcg_projected` iteration (the structure
//! that converges on the Maxwell curl-curl pencils of ex13p/ex32p):
//!
//! 1. Rayleigh–Ritz on `X` (k×k dense GEVP);
//! 2. residuals `R = AX − BX·Λ`, projected into the nullspace-free subspace;
//! 3. preconditioned residuals `Z = T(R)` (also projected);
//! 4. combined basis `W = [X | Z | P]`, B-orthonormalised (implicit QR);
//! 5. Rayleigh–Ritz on `W`; new `X = W·C[:, ..k]`, new `P = W·C[:, k..]`;
//! 6. `X` re-B-orthonormalised each iteration.
//!
//! `projector` (optional) is applied to the residual / trial blocks — for
//! singular curl-curl pencils pass a gradient-nullspace projector such as
//! [`crate::par_projection::ParGradientProjector`] composed with essential-
//! (BC-) dof zeroing, the parallel analog of HYPRE AME's
//! `hypre_AMEDiscrDivFreeComponent`.

use nalgebra as na;
use na::DMatrix;
use fem_solver::solve_dense_generalized_eig;
use crate::par_csr::ParCsrMatrix;
use crate::par_vector::ParVector;

/// Parallel LOBPCG result.
pub struct ParLobpcgResult {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
    pub iterations: usize,
    pub converged: bool,
    pub final_residual: f64,
}

/// Solve `A X = B X Λ` with parallel LOBPCG.
///
/// `nullspace_skip` (default 0.0): when > 0, Ritz values with |λ| below this
/// threshold (e.g. the gradient nullspace of a curl-curl pencil) are excluded
/// from the Ritz selection — the next `k` values are taken instead.  This is
/// the serial `lobpcg_projected` mechanism (`LobpcgConfig::nullspace_skip`)
/// and an alternative to a nullspace projector for singular pencils.
pub fn par_lobpcg(
    a: &ParCsrMatrix,
    b: Option<&ParCsrMatrix>,
    k: usize,
    precond: &dyn Fn(&[f64], &mut [f64]),
    projector: Option<&dyn Fn(&mut [ParVector])>,
    nullspace_skip: f64,
    max_iter: usize,
    tol: f64,
) -> ParLobpcgResult {
    let n_owned = a.n_owned();
    let n_ghost = a.n_ghost();
    let comm = a.comm();
    let bm = b.unwrap_or(a);
    let eps = f64::EPSILON;

    let alloc = || -> Vec<ParVector> {
        (0..k)
            .map(|_| {
                ParVector::from_local_raw(
                    vec![0.0; n_owned + n_ghost],
                    n_owned,
                    a.ghost_exchange_arc(),
                    comm.clone(),
                )
            })
            .collect()
    };

    // Apply the projector (if any) to a block of vectors and refresh ghosts.
    let apply_projector = |block: &mut [ParVector]| {
        if let Some(proj) = projector {
            proj(block);
        }
        for v in block.iter_mut() {
            v.update_ghosts();
        }
    };

    // B-orthonormalise a block via the eigendecomposition of the Gram
    // (robust to rank deficiency: columns whose Gram eigenvalue is below the
    // cutoff are dropped).  Returns the number of kept columns.
    let b_orthonormalize = |block: &mut Vec<ParVector>, n: usize| -> usize {
        let mut bvec: Vec<ParVector> = Vec::with_capacity(n);
        for _ in 0..n {
            bvec.push(ParVector::zeros_like(&block[0]));
        }
        for j in 0..n {
            bm.spmv(&mut block[j], &mut bvec[j]);
        }
        let mut gram = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                gram[(i, j)] = block[i].global_dot(&bvec[j]);
            }
        }
        let se = na::SymmetricEigen::new((&gram + gram.transpose()) * 0.5);
        let max_eig = se.eigenvalues.iter().cloned().fold(f64::NAN, f64::max).max(1e-30);
        let mut kept: Vec<usize> = Vec::new();
        for i in 0..n {
            if se.eigenvalues[i] > 1e-12 * max_eig {
                kept.push(i);
            }
        }
        if kept.is_empty() {
            return 0;
        }
        let mut out: Vec<ParVector> = Vec::with_capacity(kept.len());
        for _ in &kept {
            out.push(ParVector::zeros_like(&block[0]));
        }
        for (rj, &ej) in kept.iter().enumerate() {
            let scale = 1.0 / se.eigenvalues[ej].sqrt();
            for i in 0..n {
                out[rj].axpy(scale * se.eigenvectors[(i, ej)], &block[i]);
            }
        }
        let n_kept = out.len();
        *block = out;
        n_kept
    };

    // ── 1. Initial X: random → projector (div-free + BC zeroing) ─────────
    let mut x: Vec<ParVector> = (0..k)
        .map(|j| {
            let mut state: u64 = 0x9E3779B97F4A7C15u64
                ^ (comm.rank() as u64).wrapping_mul(0x100000001b3)
                ^ (j as u64).wrapping_mul(0x9e3779b97f4a7c15);
            let mut data = vec![0.0; n_owned + n_ghost];
            for i in 0..n_owned {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 33) as u32 as f64 / (u32::MAX as f64);
                data[i] = u * 2.0 - 1.0;
            }
            let mut v =
                ParVector::from_local_raw(data, n_owned, a.ghost_exchange_arc(), comm.clone());
            v.update_ghosts();
            v
        })
        .collect();
    apply_projector(&mut x);
    if b_orthonormalize(&mut x, k) < k {
        return ParLobpcgResult {
            eigenvalues: vec![0.0; k],
            eigenvectors: vec![vec![0.0; n_owned]; k],
            iterations: 0,
            converged: false,
            final_residual: f64::INFINITY,
        };
    }

    let mut p: Vec<ParVector> = alloc();
    let mut use_p = false;
    let mut lambdas = vec![0.0_f64; k];
    let mut converged = false;
    let mut iterations = 0usize;
    // Soft-locking mask (BLOPEX `lobpcg_checkResiduals`): modes whose absolute
    // residual satisfies ||AX − λBX|| ≤ λ·tol + tol + eps are locked and drop
    // out of the active search — the HYPRE AME mechanism visible in ex13p's
    // shrinking block size (bsize 5 → 1).
    let mut active = vec![true; k];
    let mut res_norms_abs = vec![0.0_f64; k];

    for iter in 0..max_iter {
        iterations = iter + 1;

        // ── 1. Soft-lock converged modes (skip on the first iteration) ─────
        if iter > 0 {
            for j in 0..k {
                if active[j] {
                    let lam = lambdas[j].abs();
                    if res_norms_abs[j] <= lam * tol + tol + eps {
                        active[j] = false;
                    }
                }
            }
        }
        let act: Vec<usize> = (0..k).filter(|&j| active[j]).collect();
        let na = act.len();
        if na == 0 {
            converged = true;
            break;
        }

        // ── 2. AX, BX (all columns; locked ones are untouched) ─────────────
        let mut ax = alloc();
        let mut bx = alloc();
        for j in 0..k {
            a.spmv(&mut x[j], &mut ax[j]);
            bm.spmv(&mut x[j], &mut bx[j]);
        }

        // ── 3. Rayleigh–Ritz on the active X block; rotate the active
        //        columns to the Ritz basis so x and λ stay aligned ─────────
        let mut xtax = DMatrix::<f64>::zeros(na, na);
        let mut xtbx = DMatrix::<f64>::zeros(na, na);
        for (i, &ri) in act.iter().enumerate() {
            for (j, &cj) in act.iter().enumerate() {
                xtax[(i, j)] = x[ri].global_dot(&ax[cj]);
                xtbx[(i, j)] = x[ri].global_dot(&bx[cj]);
            }
        }
        let ritz = solve_dense_generalized_eig(&xtax, &xtbx);
        for (t, &j) in act.iter().enumerate() {
            lambdas[j] = ritz.0[t];
        }
        // Rotate x, ax, bx of the active columns by the Ritz coefficients
        // (keeps the residual / soft-lock / next-W consistent; without it the
        // sorted λ values no longer correspond to the stored columns once a
        // mode is soft-locked and `act` is no longer a contiguous prefix).
        {
            let mut xr: Vec<ParVector> = Vec::with_capacity(k);
            let mut ar: Vec<ParVector> = Vec::with_capacity(k);
            let mut br: Vec<ParVector> = Vec::with_capacity(k);
            for j in 0..k {
                xr.push(if active[j] {
                    ParVector::zeros_like(&x[0])
                } else {
                    x[j].clone_vec()
                });
                ar.push(if active[j] {
                    ParVector::zeros_like(&x[0])
                } else {
                    ax[j].clone_vec()
                });
                br.push(if active[j] {
                    ParVector::zeros_like(&x[0])
                } else {
                    bx[j].clone_vec()
                });
            }
            for (t, &j) in act.iter().enumerate() {
                for (i, &ri) in act.iter().enumerate() {
                    xr[j].axpy(ritz.1[(i, t)], &x[ri]);
                    ar[j].axpy(ritz.1[(i, t)], &ax[ri]);
                    br[j].axpy(ritz.1[(i, t)], &bx[ri]);
                }
            }
            x = xr;
            ax = ar;
            bx = br;
        }

        // ── 4. Residuals R = AX − BX·Λ (active); project ───────────────────
        let mut r = alloc();
        for j in 0..k {
            for i in 0..n_owned {
                r[j].owned_slice_mut()[i] = ax[j].owned_slice()[i] - lambdas[j] * bx[j].owned_slice()[i];
            }
        }
        apply_projector(&mut r);

        // ── 5. Residual norms (absolute for the soft-lock check) ───────────
        let mut max_res = 0.0f64;
        for &j in &act {
            res_norms_abs[j] = r[j].global_norm();
            max_res = max_res.max(res_norms_abs[j] / lambdas[j].abs().max(1e-14));
        }
        if comm.rank() == 0 && (iter == 0 || (iter + 1) % 10 == 0 || max_res < tol) {
            let tstr: Vec<String> = lambdas.iter().map(|t| format!("{t:.4e}")).collect();
            eprintln!(
                "  [ParLOBPCG] iter={}: bsize={na} lambda=[{}] max_rel_res={max_res:.3e}",
                iter + 1,
                tstr.join(" ")
            );
        }
        if max_res < tol {
            converged = true;
            break;
        }

        // ── 6. Z = T(R) = precond(R), projected (active) ───────────────────
        let mut z = alloc();
        for &j in &act {
            let mut zd = vec![0.0; n_owned];
            precond(r[j].owned_slice(), &mut zd);
            for i in 0..n_owned {
                z[j].owned_slice_mut()[i] = zd[i];
            }
        }
        apply_projector(&mut z);

        // ── 7. W = [X_a | Z_a | P_a]; project + B-orthonormalise ───────────
        let w_ncols = if use_p { 3 * na } else { 2 * na };
        let mut w: Vec<ParVector> = Vec::with_capacity(w_ncols);
        for &j in &act {
            w.push(x[j].clone_vec());
        }
        for &j in &act {
            w.push(z[j].clone_vec());
        }
        if use_p {
            for &j in &act {
                w.push(p[j].clone_vec());
            }
        }
        apply_projector(&mut w);
        let mut w_ncols = b_orthonormalize(&mut w, w_ncols);
        if w_ncols < na {
            if use_p {
                // Rank loss with P — fall back to [X_a | Z_a].
                let mut w2: Vec<ParVector> = Vec::with_capacity(2 * na);
                for &j in &act {
                    w2.push(x[j].clone_vec());
                }
                for &j in &act {
                    w2.push(z[j].clone_vec());
                }
                apply_projector(&mut w2);
                w_ncols = b_orthonormalize(&mut w2, 2 * na);
                if w_ncols < na {
                    break;
                }
                w = w2;
            } else {
                break;
            }
        }

        // ── 8. Rayleigh–Ritz on W ──────────────────────────────────────────
        let mut aw = Vec::with_capacity(w_ncols);
        let mut bw = Vec::with_capacity(w_ncols);
        for _ in 0..w_ncols {
            aw.push(ParVector::zeros_like(&x[0]));
            bw.push(ParVector::zeros_like(&x[0]));
        }
        for j in 0..w_ncols {
            a.spmv(&mut w[j], &mut aw[j]);
            bm.spmv(&mut w[j], &mut bw[j]);
        }
        let mut wtaw = DMatrix::<f64>::zeros(w_ncols, w_ncols);
        let mut wtbw = DMatrix::<f64>::zeros(w_ncols, w_ncols);
        for i in 0..w_ncols {
            for j in 0..w_ncols {
                wtaw[(i, j)] = w[i].global_dot(&aw[j]);
                wtbw[(i, j)] = w[i].global_dot(&bw[j]);
            }
        }
        let (ritz_vals, ritz_vecs) = solve_dense_generalized_eig(&wtaw, &wtbw);

        // ── 9. X_a = W·C[:, skip..skip+na], P_a = W·C[:, skip+na..] ───────
        // Skip nullspace modes (|λ| < nullspace_skip).
        let skip = if nullspace_skip > 0.0 {
            ritz_vals
                .iter()
                .take_while(|&&v| v.abs() < nullspace_skip)
                .count()
                .min(w_ncols.saturating_sub(na))
        } else {
            0
        };
        let mut x_new: Vec<ParVector> = Vec::with_capacity(k);
        for _ in 0..k {
            x_new.push(ParVector::zeros_like(&x[0]));
        }
        // Locked columns are preserved; active columns get the new Ritz vectors.
        for j in 0..k {
            if !active[j] {
                x_new[j] = x[j].clone_vec();
            }
        }
        for (t, &j) in act.iter().enumerate() {
            for i in 0..w_ncols {
                x_new[j].axpy(ritz_vecs[(i, skip + t)], &w[i]);
            }
            lambdas[j] = ritz_vals[skip + t];
        }
        let p_cols = (w_ncols - skip - na).min(na);
        let mut p_new: Vec<ParVector> = Vec::with_capacity(k);
        for _ in 0..k {
            p_new.push(ParVector::zeros_like(&x[0]));
        }
        for (t, &j) in act.iter().enumerate() {
            if t < p_cols {
                for i in 0..w_ncols {
                    p_new[j].axpy(ritz_vecs[(i, skip + na + t)], &w[i]);
                }
            }
        }
        x = x_new;
        p = p_new;
        use_p = true;

        apply_projector(&mut x);
        apply_projector(&mut p);
        if b_orthonormalize(&mut x, k) < k {
            break;
        }
    }

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a2, &b2| lambdas[a2].partial_cmp(&lambdas[b2]).unwrap());
    let eigenvalues: Vec<f64> = order.iter().map(|&j| lambdas[j]).collect();
    let eigenvectors: Vec<Vec<f64>> = order.iter().map(|&j| x[j].owned_slice().to_vec()).collect();

    // Recompute final residuals on the sorted vectors.
    let mut final_res = 0.0f64;
    for &j in &order {
        let mut axj = ParVector::zeros_like(&x[0]);
        let mut bxj = ParVector::zeros_like(&x[0]);
        let mut xj = x[j].clone_vec();
        a.spmv(&mut xj, &mut axj);
        bm.spmv(&mut xj, &mut bxj);
        let lam = lambdas[j];
        let mut rj = ParVector::zeros_like(&x[0]);
        for i in 0..n_owned {
            rj.owned_slice_mut()[i] = axj.owned_slice()[i] - lam * bxj.owned_slice()[i];
        }
        final_res = final_res.max(rj.global_norm() / lam.abs().max(1e-14));
    }

    ParLobpcgResult {
        eigenvalues,
        eigenvectors,
        iterations,
        converged,
        final_residual: final_res,
    }
}
