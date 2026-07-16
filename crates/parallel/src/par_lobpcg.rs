//! Parallel LOBPCG eigenvalue solver (simplified Rayleigh–Ritz).
//!
//! Uses [`ParCsrMatrix`] global SpMV and [`ParVector`] dot products.
//! Preconditioner applied locally (block-Jacobi).

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
pub fn par_lobpcg(
    a: &ParCsrMatrix,
    b: Option<&ParCsrMatrix>,
    k: usize,
    precond: &dyn Fn(&[f64], &mut [f64]),
    max_iter: usize,
    tol: f64,
) -> ParLobpcgResult {
    let n_owned = a.n_owned();
    let n_ghost = a.n_ghost();
    let comm = a.comm();

    // Helper: allocate block of k parallel vectors (with ghost space).
    let alloc_vec = || -> Vec<ParVector> {
        (0..k).map(|_|
            ParVector::from_local_raw(
                vec![0.0; n_owned + n_ghost], n_owned, a.ghost_exchange_arc(), comm.clone(),
            )
        ).collect()
    };

    let mut x: Vec<ParVector> = (0..k).map(|j| {
        let mut data = vec![0.0; n_owned + n_ghost];
        for i in 0..n_owned {
            data[i] = ((comm.rank() as usize * 1000 + i * k + j) as f64).fract() * 2.0 - 1.0;
        }
        let mut v = ParVector::from_local_raw(data, n_owned, a.ghost_exchange_arc(), comm.clone());
        v.update_ghosts();
        v
    }).collect();

    let mut ax = alloc_vec();
    let mut bx = alloc_vec();
    let mut p = alloc_vec();
    let mut ap = alloc_vec();
    let mut bp = alloc_vec();
    let mut r = alloc_vec();
    let mut z = alloc_vec();

    // Initial A*X, B*X.
    for j in 0..k {
        a.spmv(&mut x[j], &mut ax[j]);
        let bm = b.unwrap_or(a);
        bm.spmv(&mut x[j], &mut bx[j]);
    }

    // Rayleigh–Ritz on initial subspace.
    let (theta, coeffs) = rayleigh_ritz(k, &x, &ax, &bx, &p, &ap, &bp, false);
    // Rotate X, AX, BX.
    let x_new: Vec<ParVector> = (0..k).map(|j| linear_comb(&x, &coeffs, j, k)).collect();
    let ax_new: Vec<ParVector> = (0..k).map(|j| linear_comb(&ax, &coeffs, j, k)).collect();
    let bx_new: Vec<ParVector> = (0..k).map(|j| linear_comb(&bx, &coeffs, j, k)).collect();
    for j in 0..k { x[j] = x_new[j].clone_vec(); }
    for j in 0..k { ax[j] = ax_new[j].clone_vec(); }
    for j in 0..k { bx[j] = bx_new[j].clone_vec(); }

    // Initial residuals + precondition.
    let mut residuals = vec![0.0_f64; k];
    for j in 0..k {
        let mut rj = ParVector::zeros_like(&x[0]);
        rj.axpy(1.0, &ax[j]);
        rj.axpy(-theta[j], &bx[j]);

        let mut zd = vec![0.0; n_owned];
        precond(rj.owned_slice(), &mut zd);
        let mut zv = ParVector::from_local_raw(
            { let mut d = vec![0.0; n_owned + n_ghost]; d[..n_owned].copy_from_slice(&zd); d },
            n_owned, a.ghost_exchange_arc(), comm.clone());
        zv.update_ghosts();
        z[j] = zv;
        r[j] = rj;
    }

    let mut iter = 0;
    let mut converged = false;
    while iter < max_iter && !converged {
        iter += 1;

        // Update P.
        if iter == 1 {
            for j in 0..k { p[j].copy_from(&z[j]); }
        } else {
            for j in 0..k {
                let betta = -z[j].global_dot(&ap[j]) / ap[j].global_dot(&ap[j]).max(1e-30);
                let mut new_p = ParVector::zeros_like(&x[0]);
                new_p.axpy(1.0, &z[j]);
                new_p.axpy(betta, &p[j]);
                p[j] = new_p;
            }
        }
        for pj in &mut p { pj.update_ghosts(); }

        // A*P, B*P.
        for j in 0..k { a.spmv(&mut p[j], &mut ap[j]); }
        if let Some(bm) = b {
            for j in 0..k { bm.spmv(&mut p[j], &mut bp[j]); }
        } else {
            for j in 0..k { bp[j].copy_from(&p[j]); }
        }

        // Rayleigh–Ritz on [X, P].
        let (theta2, coeffs2) = rayleigh_ritz(k, &x, &ax, &bx, &p, &ap, &bp, true);

        let x_new2: Vec<ParVector> = (0..k).map(|j| linear_comb(&x, &coeffs2, j, 2 * k)).collect();
        let ax_new2: Vec<ParVector> = (0..k).map(|j| linear_comb(&ax, &coeffs2, j, 2 * k)).collect();
        let bx_new2: Vec<ParVector> = (0..k).map(|j| linear_comb(&bx, &coeffs2, j, 2 * k)).collect();
        for j in 0..k { x[j] = x_new2[j].clone_vec(); }
        for j in 0..k { ax[j] = ax_new2[j].clone_vec(); }
        for j in 0..k { bx[j] = bx_new2[j].clone_vec(); }

        let mut max_res = 0.0f64;
        for j in 0..k {
            let mut rj = ParVector::zeros_like(&x[0]);
            rj.axpy(1.0, &ax[j]);
            rj.axpy(-theta2[j], &bx[j]);
            residuals[j] = rj.global_norm();
            max_res = max_res.max(residuals[j]);

            let mut zd = vec![0.0; n_owned];
            precond(rj.owned_slice(), &mut zd);
            let mut zv = ParVector::from_local_raw(
                { let mut d = vec![0.0; n_owned + n_ghost]; d[..n_owned].copy_from_slice(&zd); d },
                n_owned, a.ghost_exchange_arc(), comm.clone());
            zv.update_ghosts();
            z[j] = zv;
            r[j] = rj;
        }

        if comm.rank() == 0 && (iter % 10 == 0 || max_res <= tol) {
            eprintln!("  [ParLOBPCG] iter={iter}: max_res={max_res:.3e}");
        }
        if max_res <= tol { converged = true; }
    }

    let eigenvalues: Vec<f64> = (0..k)
        .map(|j| ax[j].global_dot(&x[j]) / x[j].global_dot(&x[j]).max(1e-30))
        .collect();
    let eigenvectors: Vec<Vec<f64>> = x.iter().map(|xj| xj.owned_slice().to_vec()).collect();

    ParLobpcgResult {
        eigenvalues,
        eigenvectors,
        iterations: iter,
        converged,
        final_residual: residuals.iter().copied().fold(0.0f64, f64::max),
    }
}

/// Rayleigh–Ritz on a subspace span{X} or span{[X,P]}.
fn rayleigh_ritz(
    k: usize,
    x: &[ParVector], ax: &[ParVector], bx: &[ParVector],
    p: &[ParVector], ap: &[ParVector], bp: &[ParVector],
    use_p: bool,
) -> (Vec<f64>, DMatrix<f64>) {
    let m = if use_p { 2 * k } else { k };
    let mut a_proj = DMatrix::<f64>::zeros(m, m);
    let mut b_proj = DMatrix::<f64>::zeros(m, m);

    for i in 0..k {
        for j in 0..k {
            a_proj[(i, j)] = x[i].global_dot(&ax[j]);
            b_proj[(i, j)] = x[i].global_dot(&bx[j]);
            if use_p {
                a_proj[(i, j + k)] = x[i].global_dot(&ap[j]);
                a_proj[(i + k, j)] = p[i].global_dot(&ax[j]);
                a_proj[(i + k, j + k)] = p[i].global_dot(&ap[j]);
                b_proj[(i, j + k)] = x[i].global_dot(&bp[j]);
                b_proj[(i + k, j)] = p[i].global_dot(&bx[j]);
                b_proj[(i + k, j + k)] = p[i].global_dot(&bp[j]);
            }
        }
    }

    // Generalized eigendecomposition: A_proj s = λ B_proj s.
    // Use B^{-1/2} transform via eigendecomposition of B (numerically robust
    // even for near-singular B, matching the serial LOBPCG approach).
    let (eigenvalues, eigenvectors) = solve_dense_generalized_eig(&a_proj, &b_proj);

    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

    let mut sorted_theta = Vec::with_capacity(k);
    let mut sorted_coeffs = DMatrix::<f64>::zeros(m, k);
    for (idx, &src) in indices.iter().enumerate().take(k) {
        sorted_theta.push(eigenvalues[src]);
        for i in 0..m {
            sorted_coeffs[(i, idx)] = eigenvectors[(i, src)];
        }
    }
    (sorted_theta, sorted_coeffs)
}

fn linear_comb(basis: &[ParVector], coeffs: &DMatrix<f64>, j: usize, m: usize) -> ParVector {
    let mut v = ParVector::zeros_like(&basis[0]);
    for i in 0..m.min(basis.len()) {
        v.axpy(coeffs[(i, j)], &basis[i]);
    }
    v
}
