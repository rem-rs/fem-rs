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

    for iter in 0..max_iter {
        iterations = iter + 1;

        // ── 2. AX, BX ──────────────────────────────────────────────────────
        let mut ax = alloc();
        let mut bx = alloc();
        for j in 0..k {
            a.spmv(&mut x[j], &mut ax[j]);
            bm.spmv(&mut x[j], &mut bx[j]);
        }

        // ── 3. Rayleigh–Ritz on X ──────────────────────────────────────────
        let mut xtax = DMatrix::<f64>::zeros(k, k);
        let mut xtbx = DMatrix::<f64>::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                xtax[(i, j)] = x[i].global_dot(&ax[j]);
                xtbx[(i, j)] = x[i].global_dot(&bx[j]);
            }
        }
        let ritz = solve_dense_generalized_eig(&xtax, &xtbx);
        for j in 0..k {
            lambdas[j] = ritz.0[j];
        }

        // ── 4. Residuals R = AX − BX·Λ; project ───────────────────────────
        let mut r = alloc();
        for j in 0..k {
            for i in 0..n_owned {
                r[j].owned_slice_mut()[i] = ax[j].owned_slice()[i] - lambdas[j] * bx[j].owned_slice()[i];
            }
        }
        apply_projector(&mut r);

        // ── 5. Convergence check (relative residual) ───────────────────────
        let res_norms: Vec<f64> = (0..k)
            .map(|j| {
                r[j].global_norm() / lambdas[j].abs().max(1e-14)
            })
            .collect();
        let max_res = res_norms.iter().cloned().fold(0.0f64, f64::max);
        if comm.rank() == 0 && (iter == 0 || (iter + 1) % 10 == 0 || max_res < tol) {
            let tstr: Vec<String> = lambdas.iter().map(|t| format!("{t:.4e}")).collect();
            eprintln!(
                "  [ParLOBPCG] iter={}: lambda=[{}] max_res={max_res:.3e}",
                iter + 1,
                tstr.join(" ")
            );
        }
        if max_res < tol {
            converged = true;
            break;
        }

        // ── 6. Z = T(R) = precond(R), projected ────────────────────────────
        let mut z = alloc();
        for j in 0..k {
            let mut zd = vec![0.0; n_owned];
            precond(r[j].owned_slice(), &mut zd);
            for i in 0..n_owned {
                z[j].owned_slice_mut()[i] = zd[i];
            }
        }
        apply_projector(&mut z);

        // ── 7. W = [X | Z | P]; project + B-orthonormalise ────────────────
        let w_ncols = if use_p { 3 * k } else { 2 * k };
        let mut w: Vec<ParVector> = Vec::with_capacity(w_ncols);
        for j in 0..k {
            w.push(x[j].clone_vec());
        }
        for j in 0..k {
            w.push(z[j].clone_vec());
        }
        if use_p {
            for j in 0..k {
                w.push(p[j].clone_vec());
            }
        }
        apply_projector(&mut w);
        let mut w_ncols = b_orthonormalize(&mut w, w_ncols);
        if w_ncols < k {
            if use_p {
                // Rank loss with P — fall back to [X | Z].
                let mut w2: Vec<ParVector> = Vec::with_capacity(2 * k);
                for j in 0..k {
                    w2.push(x[j].clone_vec());
                }
                for j in 0..k {
                    w2.push(z[j].clone_vec());
                }
                apply_projector(&mut w2);
                w_ncols = b_orthonormalize(&mut w2, 2 * k);
                if w_ncols < k {
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

        // ── 9. X = W·C[:, skip..skip+k], P = W·C[:, skip+k..] ─────────────
        // Skip nullspace modes (|λ| < nullspace_skip) — the serial
        // `lobpcg_projected` mechanism for singular pencils.
        let skip = if nullspace_skip > 0.0 {
            ritz_vals
                .iter()
                .take_while(|&&v| v.abs() < nullspace_skip)
                .count()
                .min(w_ncols.saturating_sub(k))
        } else {
            0
        };
        let n_avail = w_ncols;
        let mut x_new: Vec<ParVector> = Vec::with_capacity(k);
        for _ in 0..k {
            x_new.push(ParVector::zeros_like(&x[0]));
        }
        for i in 0..w_ncols {
            for j in 0..k {
                x_new[j].axpy(ritz_vecs[(i, skip + j)], &w[i]);
            }
        }
        let p_cols = (n_avail - skip - k).min(k);
        let mut p_new: Vec<ParVector> = Vec::with_capacity(k);
        for _ in 0..k {
            p_new.push(ParVector::zeros_like(&x[0]));
        }
        for i in 0..w_ncols {
            for j in 0..p_cols {
                p_new[j].axpy(ritz_vecs[(i, skip + k + j)], &w[i]);
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
