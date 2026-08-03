//! `SchurConstrainedSolver` — 1:1 port of MFEM `linalg/constraints.cpp`
//! (`SchurConstrainedSolver`).
//!
//! Solves the constrained saddle-point system
//! `[A Bᵀ; B 0] [x; λ] = [f; g]` with **GMRES(m = 50)** on the full flat
//! system, left-preconditioned by the block-diagonal operator
//! `M = diag(GS(A), I)` where `GS(A)` is the MFEM `GSSmoother` (symmetric
//! Gauss-Seidel: one forward + one backward sweep over *all* off-diagonal
//! entries, starting from a zeroed output vector — MFEM's `iterative_mode`
//! defaults to `false`).
//!
//! The implementation mirrors MFEM `GMRESSolver::Mult` step by step:
//! - preconditioned residual `r = M (b − A x)`, initial norm `β₀ = ‖r‖`;
//! - convergence test `resid = |s(i+1)| ≤ max(rel_tol·β₀, abs_tol)` after
//!   Givens rotations (MFEM's `GeneratePlaneRotation`/`ApplyPlaneRotation`);
//! - restarts every `m = 50` iterations with `Restarting...` printed, and
//!   the MFEM `Pass / Iteration  ||B r||` log format at `print_level = 1`.
//!
//! With bit-identical `A`, `B` and `C` (see `fem_assembly::constraints` for
//! the incremental-average normal constraints), the iteration trajectory
//! matches MFEM to ~1 ulp and the solution to machine precision (ex28:
//! sol.gf max|diff| = 1e-15).

use fem_linalg::{CsrMatrix, SolverConfig, SolveResult};

use crate::block::BlockSystem;

/// Solve a constrained saddle-point system the way MFEM's
/// `SchurConstrainedSolver(A, C, GSSmoother(A))` does (ex28).
pub struct SchurConstrainedSolver;

impl SchurConstrainedSolver {
    /// Solve `[A Bᵀ; B 0] [u; λ] = [f; g]`.
    ///
    /// `sys` is the block system; `f` the displacement RHS, `g` the
    /// constraint RHS (zero for normal constraints).  On return `u`/`p`
    /// hold the solution.  `cfg` matches MFEM's solver options
    /// (`rtol` ↔ `SetRelTol`, `max_iter` ↔ `SetMaxIter`, `verbose` ↔
    /// `SetPrintLevel(1)`).
    pub fn solve(
        sys: &BlockSystem,
        f: &[f64],
        g: &[f64],
        u: &mut [f64],
        p: &mut [f64],
        cfg: &SolverConfig,
    ) -> Result<SolveResult, fem_linalg::SolverError> {
        let n_u = sys.n_u();
        let n_p = sys.n_p();
        assert_eq!(u.len(), n_u);
        assert_eq!(p.len(), n_p);
        assert_eq!(f.len(), n_u);
        assert_eq!(g.len(), n_p);

        let n = n_u + n_p;
        let mut rhs = vec![0.0_f64; n];
        rhs[..n_u].copy_from_slice(f);
        rhs[n_u..].copy_from_slice(g);

        let mut x = vec![0.0_f64; n];
        let (converged, iterations, final_residual) =
            gmres_schur(sys, &rhs, &mut x, 50, cfg);

        u.copy_from_slice(&x[..n_u]);
        p.copy_from_slice(&x[n_u..]);

        Ok(SolveResult { converged, iterations, final_residual })
    }
}

// ─── MFEM GMRESSolver::Mult (left-preconditioned, restart m) ────────────────

/// Generic MFEM `GMRESSolver::Mult` — the 1:1 port used by
/// [`SchurConstrainedSolver`] (via [`gmres_schur`]) and by the 2×2 block
/// solver [`solve_gmres_block_diag_gs`].
///
/// `apply` computes `y = A x`; `apply_pc` computes `y = M x` (the
/// preconditioner, e.g. a block-diagonal GS smoother).  `x` is the in/out
/// initial guess; like MFEM's `GMRESSolver` (`iterative_mode = false` by
/// default) the iteration always starts from the zero vector and `x`'s input
/// value only matters through the caller's RHS/BC handling.
#[allow(clippy::too_many_arguments)]
fn gmres_core(
    n: usize,
    apply: &dyn Fn(&[f64], &mut [f64]),
    apply_pc: &dyn Fn(&[f64], &mut [f64]),
    b: &[f64],
    x: &mut [f64],
    m: usize,               // restart dimension (MFEM default 50)
    cfg: &SolverConfig,
) -> (bool, usize, f64) {
    let mut r = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];

    // MFEM iterative_mode = false: x starts at 0.
    x.fill(0.0);

    // r = A x (= 0);  w = b − A x = b;  r = M w  (r is zero here, so the GS
    // initial guess is 0 — same as MFEM).
    apply(x, &mut r);
    for i in 0..n {
        w[i] = b[i] - r[i];
    }
    apply_pc(&w, &mut r);

    let mut beta = norm2(&r);
    let final_norm = (cfg.rtol * beta).max(cfg.atol);

    let mut j = 1usize; // global iteration counter (1-based, as in MFEM)
    if cfg.verbose {
        println!("   Pass : {:2}   Iteration : {:3}  ||B r|| = {:.8}", 1, 0, beta);
    }
    if beta <= final_norm {
        return (true, 0, beta);
    }

    let mut v: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; m + 1];
    let mut h = vec![vec![0.0_f64; m]; m + 1];
    let mut s = vec![0.0_f64; m + 1];
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];
    let mut pass = 1usize;

    while j <= cfg.max_iter {
        // v[0] = r / β
        v[0].copy_from_slice(&r);
        scale_inplace(1.0 / beta, &mut v[0]);
        s.fill(0.0);
        s[0] = beta;

        let mut i = 0usize;
        while i < m && j <= cfg.max_iter {
            // r = A v[i];  w = M r  (GS zeroes its output first, so the
            // previous contents of w do not matter — MFEM iterative_mode=false).
            apply(&v[i], &mut r);
            apply_pc(&r, &mut w);

            // Arnoldi: H(k,i) = w·v[k],  w −= H(k,i) v[k]
            for k in 0..=i {
                h[k][i] = dot(&w, &v[k]);
                axpy_inplace(-h[k][i], &v[k], &mut w);
            }
            h[i + 1][i] = norm2(&w);
            v[i + 1].copy_from_slice(&w);
            scale_inplace(1.0 / h[i + 1][i], &mut v[i + 1]);

            // Apply previous Givens rotations, then generate + apply a new one.
            for k in 0..i {
                let (mut a_ki, mut a_k1i) = (h[k][i], h[k + 1][i]);
                apply_plane_rotation(&mut a_ki, &mut a_k1i, cs[k], sn[k]);
                h[k][i] = a_ki;
                h[k + 1][i] = a_k1i;
            }
            let (mut hii, mut hii1) = (h[i][i], h[i + 1][i]);
            generate_plane_rotation(&mut hii, &mut hii1, &mut cs[i], &mut sn[i]);
            apply_plane_rotation(&mut hii, &mut hii1, cs[i], sn[i]);
            h[i][i] = hii;
            h[i + 1][i] = hii1;
            let (mut si, mut si1) = (s[i], s[i + 1]);
            apply_plane_rotation(&mut si, &mut si1, cs[i], sn[i]);
            s[i] = si;
            s[i + 1] = si1;

            let resid = s[i + 1].abs();
            if resid <= final_norm {
                update_x(x, i, &h, &s, &v);
                return (true, j, resid);
            }
            if cfg.verbose {
                println!("   Pass : {:2}   Iteration : {:3}  ||B r|| = {:.8}", pass, j, resid);
            }
            i += 1;
            j += 1;
        }

        if cfg.verbose && j <= cfg.max_iter {
            println!("Restarting...");
        }
        update_x(x, i.saturating_sub(1), &h, &s, &v);

        // r = M (b − A x) (GS output is zeroed first, as in MFEM).
        apply(x, &mut r);
        for t in 0..n {
            w[t] = b[t] - r[t];
        }
        apply_pc(&w, &mut r);
        beta = norm2(&r);
        if beta <= final_norm {
            return (true, j, beta);
        }
        pass += 1;
    }

    (false, cfg.max_iter, beta)
}

#[allow(clippy::too_many_arguments)]
fn gmres_schur(
    sys: &BlockSystem,      // the saddle-point system [A Bᵀ; B 0]
    b: &[f64],
    x: &mut [f64],
    m: usize,               // restart dimension (MFEM default 50)
    cfg: &SolverConfig,
) -> (bool, usize, f64) {
    let n = b.len();
    let n_u = sys.n_u();
    gmres_core(
        n,
        &|xv, y| apply_system(sys, xv, y),
        &|xv, y| apply_block_pc(&sys.a, n_u, xv, y),
        b, x, m, cfg,
    )
}

// ─── Block-diagonal preconditioner diag(GS(A), I) ───────────────────────────

/// Apply `M = diag(GS(A), I)`: symmetric GS on the A block, identity on the
/// Lagrange-multiplier block.
///
/// MFEM `GSSmoother`/`IdentitySolver` have `iterative_mode = false` (the
/// `Solver` default), so `GSSmoother::Mult` zeroes `y` before the sweeps and
/// the output does **not** depend on its previous contents.
fn apply_block_pc(gs_a: &CsrMatrix<f64>, n_u: usize, x: &[f64], y: &mut [f64]) {
    y[..n_u].fill(0.0);
    gs_symmetric(gs_a, x, &mut y[..n_u]);
    for i in n_u..x.len() {
        y[i] = x[i];
    }
}

/// MFEM `GSSmoother(type = 0, iterations = 1)`: forward then backward sweep.
fn gs_symmetric(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    gs_forward(a, x, y);
    gs_backward(a, x, y);
}

/// MFEM `SparseMatrix::Gauss_Seidel_forw`: yᵢ = (xᵢ − Σ_{c≠i} Aᵢc y_c)/Aᵢᵢ,
/// i ascending; **all** off-diagonal entries of the row are summed using the
/// current y values (forward GS, `y` doubles as initial guess).
fn gs_forward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    let n = a.nrows;
    for i in 0..n {
        let mut sum = 0.0;
        let mut diag = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[p] as usize;
            if c == i {
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        y[i] = (x[i] - sum) / diag;
    }
}

/// MFEM `SparseMatrix::Gauss_Seidel_back`: same as forward but i descending.
fn gs_backward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    let n = a.nrows;
    for i in (0..n).rev() {
        let mut sum = 0.0;
        let mut diag = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[p] as usize;
            if c == i {
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        y[i] = (x[i] - sum) / diag;
    }
}

// ─── MFEM GMRES helpers ──────────────────────────────────────────────────────

/// MFEM `Update`: column-wise backsolve of the rotated Hessenberg system and
/// `x += Σ y(j) v[j]`.
fn update_x(x: &mut [f64], k: usize, h: &[Vec<f64>], s: &[f64], v: &[Vec<f64>]) {
    let mut y = s[..=k].to_vec();
    for i in (0..=k).rev() {
        y[i] /= h[i][i];
        for j in (0..i).rev() {
            y[j] -= h[j][i] * y[i];
        }
    }
    for j in 0..=k {
        axpy_inplace(y[j], &v[j], x);
    }
}

/// MFEM `GeneratePlaneRotation` (exact same numerics).
fn generate_plane_rotation(dx: &mut f64, dy: &mut f64, cs: &mut f64, sn: &mut f64) {
    if *dy == 0.0 {
        *cs = 1.0;
        *sn = 0.0;
    } else if dy.abs() > dx.abs() {
        let temp = *dx / *dy;
        *sn = 1.0 / (1.0 + temp * temp).sqrt();
        *cs = temp * *sn;
    } else {
        let temp = *dy / *dx;
        *cs = 1.0 / (1.0 + temp * temp).sqrt();
        *sn = temp * *cs;
    }
}

/// MFEM `ApplyPlaneRotation` (exact same numerics).
fn apply_plane_rotation(dx: &mut f64, dy: &mut f64, cs: f64, sn: f64) {
    let temp = cs * *dx + sn * *dy;
    *dy = -sn * *dx + cs * *dy;
    *dx = temp;
}

// ─── Small vector helpers (kept local to this module) ───────────────────────

fn spmv_into(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        let mut acc = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            acc += a.values[p] * x[a.col_idx[p] as usize];
        }
        y[i] = acc;
    }
}

/// Apply the full saddle-point operator the way MFEM `BlockOperator::Mult`
/// does: per block, `tmp = op·x_block` (overwrite) then `y_block += tmp`,
/// iterating column blocks in registration order — so `y_u = A·x_u` first,
/// then `+= Bᵀ·x_p`; `y_p = B·x_u`, then `+= C·x_p`.  This preserves MFEM's
/// floating-point summation order (a merged flat CSR would not).
fn apply_system(sys: &BlockSystem, x: &[f64], y: &mut [f64]) {
    let n_u = sys.n_u();
    // Row block 0 (displacements): tmp = A x_u; y_u += tmp;  tmp = Bᵀ x_p; y_u += tmp.
    spmv_into(&sys.a, &x[..n_u], &mut y[..n_u]);
    spmv_add(&sys.bt, &x[n_u..], &mut y[..n_u]);
    // Row block 1 (multipliers): tmp = B x_u; y_p += tmp;  (+ C x_p if present).
    spmv_into(&sys.b, &x[..n_u], &mut y[n_u..]);
    if let Some(c) = &sys.c {
        spmv_add(c, &x[n_u..], &mut y[n_u..]);
    }
}

/// `y += A x` (as `Vector::Add` after a block SpMV in MFEM).
fn spmv_add(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for i in 0..a.nrows {
        let mut acc = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            acc += a.values[p] * x[a.col_idx[p] as usize];
        }
        y[i] += acc;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn scale_inplace(alpha: f64, v: &mut [f64]) {
    for x in v.iter_mut() {
        *x *= alpha;
    }
}

fn axpy_inplace(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

// ─── 2×2 block GMRES with block-diagonal GS preconditioner (ex36) ───────────

/// Solve the 2×2 block system
/// `[A00 A01; A10 A11] [x0; x1] = [b0; b1]` with GMRES preconditioned by
/// `M = diag(GS(A00), GS(A11))` — a 1:1 port of MFEM's `GMRES` +
/// `BlockDiagonalPreconditioner(GSSmoother(A00), GSSmoother(A11))` (ex36
/// obstacle problem).
///
/// `A01` must be the transpose of `A10` (as in MFEM, where `A01 =
/// Transpose(A10)`).  `x0`/`x1` are in/out initial guesses; the GMRES
/// iteration itself starts from zero (MFEM `iterative_mode = false`), so the
/// input values only enter through the caller's essential-BC RHS handling.
/// Returns `(converged, iterations, final_preconditioned_residual)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_gmres_block_diag_gs(
    a00: &CsrMatrix<f64>,
    a01: &CsrMatrix<f64>,
    a10: &CsrMatrix<f64>,
    a11: &CsrMatrix<f64>,
    b0: &[f64],
    b1: &[f64],
    x0: &mut [f64],
    x1: &mut [f64],
    restart: usize,      // MFEM GMRES restart / MR dimension (ex36: 500)
    cfg: &SolverConfig,
) -> (bool, usize, f64) {
    let n0 = x0.len();
    let n1 = x1.len();
    assert_eq!(a00.nrows, n0, "A00 rows must match x0");
    assert_eq!(a10.nrows, n1, "A10 rows must match x1");
    assert_eq!(a11.nrows, n1, "A11 rows must match x1");
    assert_eq!(b0.len(), n0);
    assert_eq!(b1.len(), n1);

    let n = n0 + n1;
    let mut b = vec![0.0_f64; n];
    b[..n0].copy_from_slice(b0);
    b[n0..].copy_from_slice(b1);

    let mut x = vec![0.0_f64; n];

    let res = gmres_core(
        n,
        &|xv, y| {
            // Row block 0: y0 = A00 x0, then += A01 x1  (MFEM BlockOperator
            // column-iteration order preserves the floating-point sum).
            spmv_into(a00, &xv[..n0], &mut y[..n0]);
            spmv_add(a01, &xv[n0..], &mut y[..n0]);
            // Row block 1: y1 = A10 x0, then += A11 x1.
            spmv_into(a10, &xv[..n0], &mut y[n0..]);
            spmv_add(a11, &xv[n0..], &mut y[n0..]);
        },
        &|xv, y| {
            // BlockDiagonalPreconditioner::Mult: each GSSmoother zeroes its
            // output first (iterative_mode = false), then sweeps fwd+back.
            y[..n0].fill(0.0);
            gs_symmetric(a00, &xv[..n0], &mut y[..n0]);
            y[n0..].fill(0.0);
            gs_symmetric(a11, &xv[n0..], &mut y[n0..]);
        },
        &b, &mut x, restart, cfg,
    );

    x0.copy_from_slice(&x[..n0]);
    x1.copy_from_slice(&x[n0..]);
    res
}
