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

use fem_linalg::{CsrMatrix, SolveResult, SolverConfig, SolverError};

use crate::block::BlockSystem;
use crate::smoother::{gauss_seidel_back, gauss_seidel_forw};

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
        let (converged, iterations, final_residual) = gmres_schur(sys, &rhs, &mut x, 50, cfg);

        u.copy_from_slice(&x[..n_u]);
        p.copy_from_slice(&x[n_u..]);

        Ok(SolveResult {
            converged,
            iterations,
            final_residual,
        })
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
    m: usize, // restart dimension (MFEM default 50)
    iterative_mode: bool,
    cfg: &SolverConfig,
) -> (bool, usize, f64) {
    let mut r = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];

    // MFEM GMRESSolver default iterative_mode = true (Solver(0, true) in
    // IterativeSolver()); when false, x starts at 0.
    if !iterative_mode {
        x.fill(0.0);
    }

    // r = A x;  w = b − A x;  r = M w  (for iterative_mode=false, x = 0 so
    // r = M·b and the GS initial guess is 0 — same as MFEM).
    apply(x, &mut r);
    for i in 0..n {
        w[i] = b[i] - r[i];
    }
    apply_pc(&w, &mut r);

    let mut beta = norm2(&r);
    let final_norm = (cfg.rtol * beta).max(cfg.atol);

    let mut j = 1usize; // global iteration counter (1-based, as in MFEM)
    if cfg.verbose {
        println!(
            "   Pass : {:2}   Iteration : {:3}  ||B r|| = {:.17}",
            1, 0, beta
        );
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
            if h[i + 1][i].abs() < 1e-300 {
                // Arnoldi breakdown: the Krylov subspace has closed, so the
                // residual-minimising solution is exact — MFEM's GMRES does
                // the same (happy break).  Dividing by ~0 would NaN the
                // basis and return a wrong solution.
                update_x(x, i, &h, &s, &v);
                return (true, j, s[i].abs());
            }
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
                println!(
                    "   Pass : {:2}   Iteration : {:3}  ||B r|| = {:.17}",
                    pass, j, resid
                );
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
    sys: &BlockSystem, // the saddle-point system [A Bᵀ; B 0]
    b: &[f64],
    x: &mut [f64],
    m: usize, // restart dimension (MFEM default 50)
    cfg: &SolverConfig,
) -> (bool, usize, f64) {
    let n = b.len();
    let n_u = sys.n_u();
    gmres_core(
        n,
        &|xv, y| apply_system(sys, xv, y),
        &|xv, y| apply_block_pc(&sys.a, n_u, xv, y),
        b,
        x,
        m,
        false,
        cfg,
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
    gauss_seidel_forw(a, x, y);
    gauss_seidel_back(a, x, y);
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
    restart: usize, // MFEM GMRES restart / MR dimension (ex36: 500)
    iterative_mode: bool, // MFEM GMRESSolver default true
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
    // With iterative_mode (MFEM default true) GMRES iterates from the current
    // x0/x1 (previous Newton solution); gmres_core zeroes it when false.
    x[..n0].copy_from_slice(x0);
    x[n0..].copy_from_slice(x1);

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
        &b,
        &mut x,
        restart,
        iterative_mode,
        cfg,
    );

    x0.copy_from_slice(&x[..n0]);
    x1.copy_from_slice(&x[n0..]);
    res
}

// ═════════════════════════════════════════════════════════════════════════════
// MFEM `linalg/constraints.{hpp,cpp}` — the full `ConstrainedSolver` family.
//
// Serial port: `HypreParMatrix` → [`CsrMatrix`], `HypreBoomerAMG` → a
// pluggable preconditioner closure (default: Jacobi / diagonal inverse —
// recorded as a port deviation; AMG can be plugged via
// [`EliminationSolver::set_preconditioner`]).
// `SchurConstrainedHypreSolver` is PAR-only (hypre `ParMult`/`RAP`) and is
// not ported — see the module-level TODO below.
//
// TODO(parallel): port `SchurConstrainedHypreSolver` once the fem-rs
// parallel layer exposes `HypreParMatrix`-equivalents (`ParMult`, `RAP`,
// `InvScaleRows`); it derives from `SchurConstrainedSolver` with an AMG
// preconditioner on both blocks and a `B diag(A)⁻¹ Bᵀ` Schur approximation.
// ═════════════════════════════════════════════════════════════════════════════

/// MFEM `Eliminator` — perform elimination of a single (block of)
/// constraint(s).
///
/// Keeps the primary/secondary dof split of one constraint block and does
/// small dense block solves:
/// * [`Eliminator::eliminate`] applies `−B_s⁻¹ B_p` (primary → secondary),
/// * [`Eliminator::lagrange_secondary`] applies `B_s⁻¹` (Lagrange →
///   secondary).
pub struct Eliminator {
    lagrange_tdofs: Vec<usize>,
    primary_tdofs: Vec<usize>,
    secondary_tdofs: Vec<usize>,
    /// `B_p` (`m × p`, row-major dense).
    bp: Vec<f64>,
    /// `B_s` LU-factored in place (`m × m`).
    bs_lu: Vec<f64>,
    bs_ipiv: Vec<i32>,
    /// `B_sᵀ` LU-factored in place.
    bst_lu: Vec<f64>,
    bst_ipiv: Vec<i32>,
}

impl Eliminator {
    /// MFEM `Eliminator::Eliminator(B, lagrange_tdofs, primary_tdofs,
    /// secondary_tdofs)`.  `lagrange_tdofs.len()` must equal
    /// `secondary_tdofs.len()`.
    pub fn new(
        b: &CsrMatrix<f64>,
        lagrange_tdofs: &[usize],
        primary_tdofs: &[usize],
        secondary_tdofs: &[usize],
    ) -> Self {
        assert_eq!(
            lagrange_tdofs.len(),
            secondary_tdofs.len(),
            "Eliminator: dof sizes don't match!"
        );
        let m = lagrange_tdofs.len();

        let bp = dense_submatrix(b, lagrange_tdofs, primary_tdofs);
        let bs = dense_submatrix(b, lagrange_tdofs, secondary_tdofs);
        let mut bs_lu = bs.clone();
        let mut bs_ipiv = vec![0_i32; m];
        dense_lu_factor(&mut bs_lu, m, &mut bs_ipiv)
            .unwrap_or_else(|| panic!("Eliminator: singular B_s block"));
        let mut bst_lu = transpose_dense(&bs, m, m);
        let mut bst_ipiv = vec![0_i32; m];
        dense_lu_factor(&mut bst_lu, m, &mut bst_ipiv)
            .unwrap_or_else(|| panic!("Eliminator: singular B_sᵀ block"));

        Eliminator {
            lagrange_tdofs: lagrange_tdofs.to_vec(),
            primary_tdofs: primary_tdofs.to_vec(),
            secondary_tdofs: secondary_tdofs.to_vec(),
            bp,
            bs_lu,
            bs_ipiv,
            bst_lu,
            bst_ipiv,
        }
    }

    /// Lagrange dofs of this constraint block.
    pub fn lagrange_dofs(&self) -> &[usize] {
        &self.lagrange_tdofs
    }

    /// Primary dofs of this constraint block.
    pub fn primary_dofs(&self) -> &[usize] {
        &self.primary_tdofs
    }

    /// Secondary dofs of this constraint block.
    pub fn secondary_dofs(&self) -> &[usize] {
        &self.secondary_tdofs
    }

    /// Given primary dofs in `vin`, return secondary dofs in `vout`.
    /// Applies `−B_s⁻¹ B_p` (MFEM `Eliminate`).
    pub fn eliminate(&self, vin: &[f64], vout: &mut [f64]) {
        let m = self.secondary_tdofs.len();
        for (r, out) in vout.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (c, &v) in vin.iter().enumerate() {
                acc += self.bp[r * self.primary_tdofs.len() + c] * v;
            }
            *out = acc;
        }
        dense_lu_solve(&self.bs_lu, &self.bs_ipiv, m, vout, 1);
        for v in vout.iter_mut() {
            *v = -*v;
        }
    }

    /// Transpose of [`Eliminator::eliminate`]: applies `−B_pᵀ B_s⁻ᵀ`
    /// (MFEM `EliminateTranspose`).
    pub fn eliminate_transpose(&self, vin: &[f64], vout: &mut [f64]) {
        let m = self.secondary_tdofs.len();
        let mut work = vin.to_vec();
        dense_lu_solve(&self.bst_lu, &self.bst_ipiv, m, &mut work, 1);
        let p = self.primary_tdofs.len();
        for (r, out) in vout.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (c, &w) in work.iter().enumerate() {
                acc += self.bp[c * p + r] * w;
            }
            *out = -acc;
        }
    }

    /// Maps Lagrange multipliers to secondary dofs: applies `B_s⁻¹`
    /// (MFEM `LagrangeSecondary`).
    pub fn lagrange_secondary(&self, vin: &[f64], vout: &mut [f64]) {
        vout.copy_from_slice(vin);
        let m = self.secondary_tdofs.len();
        dense_lu_solve(&self.bs_lu, &self.bs_ipiv, m, vout, 1);
    }

    /// Transpose of [`Eliminator::lagrange_secondary`]: applies `B_s⁻ᵀ`
    /// (MFEM `LagrangeSecondaryTranspose`).
    pub fn lagrange_secondary_transpose(&self, vin: &[f64], vout: &mut [f64]) {
        vout.copy_from_slice(vin);
        let m = self.secondary_tdofs.len();
        dense_lu_solve(&self.bst_lu, &self.bst_ipiv, m, vout, 1);
    }

    /// Return `−B_s⁻¹ B_p` explicitly as a dense `m × p` row-major matrix
    /// (MFEM `ExplicitAssembly`).
    pub fn explicit_assembly(&self) -> Vec<f64> {
        let p = self.primary_tdofs.len();
        let mut mat = self.bp.clone();
        let m = self.secondary_tdofs.len();
        dense_lu_solve(&self.bs_lu, &self.bs_ipiv, m, &mut mat, p);
        for v in mat.iter_mut() {
            *v = -*v;
        }
        mat
    }
}

/// MFEM `EliminationProjection` — the constraint-elimination projector
/// `P` assembled from a set of [`Eliminator`]s.
///
/// `Mult`    maps a full vector onto the constrained subspace (secondary
///           dofs are written in terms of the primary ones);
/// `MultTranspose` is its (weighted) transpose, used to reduce RHSs.
pub struct EliminationProjection {
    n: usize,
    eliminators: Vec<Eliminator>,
}

impl EliminationProjection {
    /// MFEM constructor: takes ownership of the eliminators; `n` is
    /// `A.Height()`.
    pub fn new(n: usize, eliminators: Vec<Eliminator>) -> Self {
        EliminationProjection { n, eliminators }
    }

    /// MFEM `Mult`: `y = x`, then for each eliminator set
    /// `y[secondary] = Eliminate(x[primary])`.
    pub fn mult(&self, vin: &[f64], vout: &mut [f64]) {
        assert_eq!(vin.len(), self.n, "EliminationProjection::mult: wrong size");
        vout.copy_from_slice(vin);
        for elim in &self.eliminators {
            let sub_in: Vec<f64> = elim.primary_dofs().iter().map(|&i| vin[i]).collect();
            let mut sub_out = vec![0.0_f64; elim.secondary_dofs().len()];
            elim.eliminate(&sub_in, &mut sub_out);
            for (&d, &v) in elim.secondary_dofs().iter().zip(sub_out.iter()) {
                vout[d] = v;
            }
        }
    }

    /// MFEM `MultTranspose`: `y = x`; `y[primary] += EliminateTranspose(
    /// x[secondary])`; then `y[secondary] = 0`.
    pub fn mult_transpose(&self, vin: &[f64], vout: &mut [f64]) {
        assert_eq!(
            vin.len(),
            self.n,
            "EliminationProjection::mult_transpose: wrong size"
        );
        vout.copy_from_slice(vin);
        for elim in &self.eliminators {
            let sub_in: Vec<f64> = elim.secondary_dofs().iter().map(|&i| vin[i]).collect();
            let mut sub_out = vec![0.0_f64; elim.primary_dofs().len()];
            elim.eliminate_transpose(&sub_in, &mut sub_out);
            for (&d, &v) in elim.primary_dofs().iter().zip(sub_out.iter()) {
                vout[d] += v;
            }
            for &d in elim.secondary_dofs() {
                vout[d] = 0.0;
            }
        }
    }

    /// Assemble the projector as a processor-local sparse matrix
    /// (MFEM `AssembleExact`): identity plus `−B_s⁻¹ B_p` on the
    /// secondary rows.
    pub fn assemble_exact(&self) -> CsrMatrix<f64> {
        let mut coo = fem_linalg::CooMatrix::<f64>::new(self.n, self.n);
        for i in 0..self.n {
            coo.add(i, i, 1.0);
        }
        for elim in &self.eliminators {
            let mat_k = elim.explicit_assembly();
            let p = elim.primary_dofs().len();
            for (iz, &i) in elim.secondary_dofs().iter().enumerate() {
                for (jz, &j) in elim.primary_dofs().iter().enumerate() {
                    coo.add(i, j, mat_k[iz * p + jz]);
                }
                // Set(i, i, 0.0): MFEM zeroes the diagonal entry after the
                // block insert; with additive COO semantics we record the
                // offset −1 to cancel the identity added above.
                coo.add(i, i, -1.0);
            }
        }
        coo.into_csr()
    }

    /// Given Lagrange multiplier RHS `g`, return `g̃` (MFEM
    /// `BuildGTilde`): `g̃[secondary] += B_s⁻¹ g[lagrange]`.
    pub fn build_g_tilde(&self, g: &[f64], gtilde: &mut [f64]) {
        gtilde.fill(0.0);
        for elim in &self.eliminators {
            let subr: Vec<f64> = elim.lagrange_dofs().iter().map(|&i| g[i]).collect();
            let mut bsinvr = vec![0.0_f64; subr.len()];
            elim.lagrange_secondary(&subr, &mut bsinvr);
            for (&d, &v) in elim.secondary_dofs().iter().zip(bsinvr.iter()) {
                gtilde[d] += v;
            }
        }
    }

    /// Recover the Lagrange multiplier after a solve (MFEM
    /// `RecoverMultiplier`): `λ[lagrange] = −B_s⁻ᵀ ((A·disp − disprhs)
    /// )[secondary]`.
    pub fn recover_multiplier(
        &self,
        a: &CsrMatrix<f64>,
        disprhs: &[f64],
        disp: &[f64],
        lagrangem: &mut [f64],
    ) {
        lagrangem.fill(0.0);
        let mut fullrhs = vec![0.0_f64; a.nrows];
        spmv_into(a, disp, &mut fullrhs);
        for i in 0..fullrhs.len() {
            fullrhs[i] -= disprhs[i];
            fullrhs[i] = -fullrhs[i];
        }
        for elim in &self.eliminators {
            let localsec: Vec<f64> =
                elim.secondary_dofs().iter().map(|&i| fullrhs[i]).collect();
            let mut locallag = vec![0.0_f64; localsec.len()];
            elim.lagrange_secondary_transpose(&localsec, &mut locallag);
            for (&d, &v) in elim.lagrange_dofs().iter().zip(locallag.iter()) {
                lagrangem[d] += v;
            }
        }
    }

    /// Number of eliminators (constraint blocks).
    pub fn n_elimimators(&self) -> usize {
        self.eliminators.len()
    }
}

/// Krylov solver selection for the eliminated/penalized system
/// (MFEM `BuildKrylov`: `CGSolver` or `GMRESSolver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrainedKrylov {
    /// MFEM `CGSolver` — for SPD eliminated/penalized operators.
    CG,
    /// MFEM `GMRESSolver` — for indefinite operators.
    GMRES,
}

/// MFEM `EliminationSolver` — solve the constrained system by eliminating
/// the constraint.
///
/// Solves with the operator `Pᵀ A P + Z_P`, where `P` is the
/// [`EliminationProjection`] and `Z_P` is the identity on the eliminated
/// (secondary) dofs.  The two MFEM constructors are provided:
/// [`EliminationSolver::from_primary_secondary`] (single elimination
/// block) and [`EliminationSolver::from_constraint_rowstarts`]
/// (block-wise, from `BuildNormalConstraints`-style row starts).
///
/// Serial-port deviation: MFEM uses `HypreBoomerAMG` as the preconditioner
/// (`EliminationCGSolver` / `EliminationGMRESSolver`); here the default is
/// a Jacobi (diagonal) preconditioner and an arbitrary preconditioner
/// closure can be installed with [`EliminationSolver::set_preconditioner`].
pub struct EliminationSolver {
    /// The (local) original operator `A` (MFEM `hA`).
    a: CsrMatrix<f64>,
    /// The projector `P` (owns the eliminators, MFEM `projector`).
    projector: EliminationProjection,
    /// `Pᵀ A P + Z_P` assembled explicitly (MFEM `h_explicit_operator`).
    explicit_operator: CsrMatrix<f64>,
    /// Diagonal of the explicit operator (Jacobi default preconditioner).
    diag: Vec<f64>,
    /// User-supplied preconditioner override (`y = M⁻¹ r`).
    precond: Option<std::sync::Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
    /// Krylov method (CG ↔ `EliminationCGSolver`, GMRES ↔
    /// `EliminationGMRESSolver`).
    krylov: ConstrainedKrylov,
    /// Lagrange-multiplier solution (MFEM `multiplier_sol`).
    multiplier_sol: Vec<f64>,
    /// Constraint right-hand side (MFEM `constraint_rhs`).
    constraint_rhs: Vec<f64>,
    /// GMRES restart dimension (MFEM default 50).
    restart: usize,
    /// Solver parameters (MFEM `SetRelTol`/`SetAbsTol`/`SetMaxIter`).
    pub cfg: SolverConfig,
}

impl EliminationSolver {
    /// MFEM constructor with an explicit primary/secondary split (a single
    /// elimination block).  `secondary_dofs.len()` must equal `B.Height()`.
    pub fn from_primary_secondary(
        a: CsrMatrix<f64>,
        b: &CsrMatrix<f64>,
        primary_dofs: &[usize],
        secondary_dofs: &[usize],
        krylov: ConstrainedKrylov,
    ) -> Self {
        assert_eq!(
            secondary_dofs.len(),
            b.nrows,
            "EliminationSolver: wrong number of dofs for elimination!"
        );
        let lagrange_dofs: Vec<usize> = (0..secondary_dofs.len()).collect();
        let elim = Eliminator::new(b, &lagrange_dofs, primary_dofs, secondary_dofs);
        Self::build(a, vec![elim], krylov)
    }

    /// MFEM constructor by blocks: the nonzeros of `B` are assumed to be in
    /// disjoint rows/columns; the rows of constraint block `k` are
    /// `constraint_rowstarts[k] .. constraint_rowstarts[k+1]−1` and the
    /// secondary dofs are the first nonzeros in those rows.
    pub fn from_constraint_rowstarts(
        a: CsrMatrix<f64>,
        b: &CsrMatrix<f64>,
        constraint_rowstarts: &[usize],
        krylov: ConstrainedKrylov,
    ) -> Self {
        let mut eliminators: Vec<Eliminator> = Vec::new();
        if b.nrows > 0 && b.ncols > 0 {
            for k in 0..constraint_rowstarts.len().saturating_sub(1) {
                let r0 = constraint_rowstarts[k];
                let r1 = constraint_rowstarts[k + 1];
                let csize = r1 - r0;
                let lagrange_dofs: Vec<usize> = (r0..r1).collect();
                let mut secondary_dofs = vec![usize::MAX; csize];
                // Identify one secondary dof for each row (first sufficiently
                // nonzero column entry not already chosen).
                for (local, &i) in lagrange_dofs.iter().enumerate() {
                    for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
                        let j = b.col_idx[ptr] as usize;
                        let val = b.values[ptr];
                        if val.abs() > 1e-12 && !secondary_dofs.contains(&j) {
                            secondary_dofs[local] = j;
                            break;
                        }
                    }
                }
                // Assign the remaining (non-secondary) dofs as primary.
                let mut primary_dofs: Vec<usize> = Vec::new();
                for &i in lagrange_dofs.iter() {
                    for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
                        let j = b.col_idx[ptr] as usize;
                        if !secondary_dofs.contains(&j) {
                            primary_dofs.push(j);
                        }
                    }
                }
                primary_dofs.sort_unstable();
                primary_dofs.dedup();
                assert!(
                    secondary_dofs.iter().all(|&s| s != usize::MAX),
                    "EliminationSolver: secondary dofs don't match rows!"
                );
                eliminators.push(Eliminator::new(
                    b,
                    &lagrange_dofs,
                    &primary_dofs,
                    &secondary_dofs,
                ));
            }
        }
        Self::build(a, eliminators, krylov)
    }

    fn build(a: CsrMatrix<f64>, eliminators: Vec<Eliminator>, krylov: ConstrainedKrylov) -> Self {
        let n = a.nrows;
        let projector = EliminationProjection::new(n, eliminators);
        // BuildExplicitOperator: Pᵀ A P + (identity on eliminated dofs).
        let p = projector.assemble_exact();
        let ap = a.multiply(&p);
        let pap = p.transpose().multiply(&ap);
        // Add Z_P: 1 on the secondary dofs (where P lost its diagonal).
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            for ptr in pap.row_ptr[i]..pap.row_ptr[i + 1] {
                coo.add(i, pap.col_idx[ptr] as usize, pap.values[ptr]);
            }
        }
        // Z_P: 1 on rows that PᵀAP leaves zero (fully eliminated dofs) —
        // MFEM forms RAP then calls EliminateZeroRows on the result.
        for i in 0..n {
            let nnz = pap.row_ptr[i + 1] - pap.row_ptr[i];
            if nnz == 0 {
                coo.add(i, i, 1.0);
            }
        }
        let explicit_operator = coo.into_csr();
        let diag = explicit_operator.diagonal();
        EliminationSolver {
            a,
            projector,
            explicit_operator,
            diag,
            precond: None,
            krylov,
            multiplier_sol: Vec::new(),
            constraint_rhs: Vec::new(),
            restart: 50,
            cfg: SolverConfig::default(),
        }
    }

    /// Install a custom preconditioner `y = M⁻¹ r` (serial stand-in for
    /// MFEM's `SetPreconditioner`, default Jacobi).
    pub fn set_preconditioner(
        &mut self,
        precond: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) {
        self.precond = Some(std::sync::Arc::new(precond));
    }

    /// Set the constraint RHS `r` for `B x = r` (MFEM `SetConstraintRHS`).
    pub fn set_constraint_rhs(&mut self, r: &[f64]) {
        self.constraint_rhs = r.to_vec();
    }

    /// Lagrange-multiplier solution of the last `mult` (MFEM
    /// `GetMultiplierSolution`).
    pub fn get_multiplier_solution(&self) -> &[f64] {
        &self.multiplier_sol
    }

    /// The eliminated operator `Pᵀ A P + Z_P` (exposed for preconditioner
    /// setup and testing).
    pub fn explicit_operator(&self) -> &CsrMatrix<f64> {
        &self.explicit_operator
    }

    /// MFEM `EliminationSolver::Mult` (takes `&mut self` for the
    /// multiplier-solution output state that C++ marks `mutable`).
    pub fn mult(&mut self, rhs: &[f64], sol: &mut [f64]) -> Result<SolveResult, SolverError> {
        let n = self.a.nrows;
        // r̃ = BuildGTilde(constraint_rhs) (or 0).
        let mut rtilde = vec![0.0_f64; n];
        if !self.constraint_rhs.is_empty() {
            self.projector.build_g_tilde(&self.constraint_rhs, &mut rtilde);
        }
        // temprhs = rhs − A r̃.
        let mut temprhs = rhs.to_vec();
        let mut ar = vec![0.0_f64; n];
        spmv_into(&self.a, &rtilde, &mut ar);
        for i in 0..n {
            temprhs[i] -= ar[i];
        }
        // reducedrhs = Pᵀ temprhs.
        let mut reducedrhs = vec![0.0_f64; n];
        self.projector.mult_transpose(&temprhs, &mut reducedrhs);
        // Solve (Pᵀ A P + Z_P) x = reducedrhs.
        let mut reducedsol = vec![0.0_f64; n];
        let res = self.krylov_solve(&reducedrhs, &mut reducedsol)?;
        // sol = P x;  recover multiplier;  sol += r̃.
        self.projector.mult(&reducedsol, sol);
        self.multiplier_sol = vec![0.0_f64; n];
        self.projector
            .recover_multiplier(&self.a, &temprhs, sol, &mut self.multiplier_sol);
        for i in 0..n {
            sol[i] += rtilde[i];
        }
        Ok(res)
    }

    fn krylov_solve(
        &self,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveResult, SolverError> {
        let n = b.len();
        let op = &self.explicit_operator;
        let apply = |xv: &[f64], y: &mut [f64]| spmv_into(op, xv, y);
        match self.krylov {
            ConstrainedKrylov::CG => {
                let diag = self.diag.clone();
                let precond = |r: &[f64], z: &mut [f64]| jacobi_apply(&diag, r, z);
                crate::iterative::solve_pcg_operator_precond(n, apply, b, x, precond, &self.cfg)
            }
            ConstrainedKrylov::GMRES => {
                let jac = self.diag.clone();
                let (converged, iterations, final_residual) = gmres_core(
                    n,
                    &apply,
                    &|r: &[f64], z: &mut [f64]| jacobi_apply(&jac, r, z),
                    b,
                    x,
                    self.restart,
                    false,
                    &self.cfg,
                );
                Ok(SolveResult {
                    converged,
                    iterations,
                    final_residual,
                })
            }
        }
    }
}

/// MFEM `PenaltyConstrainedSolver` — solve the constrained system with the
/// penalty method `A + Bᵀ D B`, `D = diag(penalty)`.
///
/// Only approximates the solution; better approximation with higher
/// penalty at reduced preconditioner effectiveness (MFEM doc).
pub struct PenaltyConstrainedSolver {
    /// The constraint matrix `B` (`M × N`).
    b: CsrMatrix<f64>,
    /// Penalty values per constraint (diagonal of `D`).
    penalty: Vec<f64>,
    /// `A + Bᵀ D B` (MFEM `penalized_mat`).
    penalized_mat: CsrMatrix<f64>,
    /// Diagonal of the penalized operator (Jacobi default preconditioner).
    diag: Vec<f64>,
    /// User-supplied preconditioner override (`y = M⁻¹ r`).
    precond: Option<std::sync::Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
    /// Krylov method (CG ↔ `PenaltyPCGSolver`, GMRES ↔
    /// `PenaltyGMRESSolver`).
    krylov: ConstrainedKrylov,
    /// Lagrange-multiplier estimate `penalty·(B x − r)` (MFEM
    /// `multiplier_sol`).
    multiplier_sol: Vec<f64>,
    /// Constraint right-hand side (MFEM `constraint_rhs`).
    constraint_rhs: Vec<f64>,
    /// GMRES restart dimension (MFEM default 50).
    restart: usize,
    /// Solver parameters (MFEM `SetRelTol`/`SetAbsTol`/`SetMaxIter`).
    pub cfg: SolverConfig,
}

impl PenaltyConstrainedSolver {
    /// MFEM `PenaltyConstrainedSolver(A, B, penalty)` with a scalar
    /// penalty (serial port; the `HypreParMatrix`/`Vector` penalty
    /// variants collapse to the same `D = diag(penalty)` initialization).
    pub fn new(
        a: CsrMatrix<f64>,
        b: CsrMatrix<f64>,
        penalty: f64,
        krylov: ConstrainedKrylov,
    ) -> Self {
        let penalty_vec = vec![penalty; b.nrows];
        Self::from_penalty_vector(a, b, penalty_vec, krylov)
    }

    /// MFEM variant with a per-constraint penalty vector.
    pub fn from_penalty_vector(
        a: CsrMatrix<f64>,
        b: CsrMatrix<f64>,
        penalty: Vec<f64>,
        krylov: ConstrainedKrylov,
    ) -> Self {
        assert_eq!(penalty.len(), b.nrows, "penalty size must match B.Height()");
        // Initialize: penalized = A + Bᵀ D B  (D diagonal).
        let mut db = fem_linalg::CooMatrix::<f64>::new(b.nrows, b.ncols);
        for i in 0..b.nrows {
            for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
                db.add(i, b.col_idx[ptr] as usize, b.values[ptr] * penalty[i]);
            }
        }
        let db = db.into_csr();
        let bt = b.transpose();
        let btdb = bt.multiply(&db);
        let penalized_mat = a.add(&btdb);
        let diag = penalized_mat.diagonal();
        PenaltyConstrainedSolver {
            b,
            penalty,
            penalized_mat,
            diag,
            precond: None,
            krylov,
            multiplier_sol: Vec::new(),
            constraint_rhs: Vec::new(),
            restart: 50,
            cfg: SolverConfig::default(),
        }
    }

    /// Install a custom preconditioner `y = M⁻¹ r` (serial stand-in for
    /// MFEM's `SetPreconditioner`, default Jacobi).
    pub fn set_preconditioner(
        &mut self,
        precond: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) {
        self.precond = Some(std::sync::Arc::new(precond));
    }

    /// Set the constraint RHS `r` for `B x = r` (MFEM `SetConstraintRHS`).
    pub fn set_constraint_rhs(&mut self, r: &[f64]) {
        self.constraint_rhs = r.to_vec();
    }

    /// Lagrange-multiplier estimate of the last `mult` (MFEM
    /// `GetMultiplierSolution`).
    pub fn get_multiplier_solution(&self) -> &[f64] {
        &self.multiplier_sol
    }

    /// The penalized operator `A + Bᵀ D B`.
    pub fn penalized_matrix(&self) -> &CsrMatrix<f64> {
        &self.penalized_mat
    }

    /// MFEM `PenaltyConstrainedSolver::Mult` (takes `&mut self` for the
    /// multiplier-solution output state that C++ marks `mutable`).
    pub fn mult(&mut self, b: &[f64], x: &mut [f64]) -> Result<SolveResult, SolverError> {
        let n = self.penalized_mat.nrows;
        // Form the penalized right-hand side: b + Bᵀ D r.
        let mut penalized_rhs = b.to_vec();
        if !self.constraint_rhs.is_empty() {
            let m = self.constraint_rhs.len();
            let temp_rhs: Vec<f64> = (0..m)
                .map(|i| self.penalty[i] * self.constraint_rhs[i])
                .collect();
            let mut temp = vec![0.0_f64; n];
            spmv_into(&self.b.transpose(), &temp_rhs, &mut temp);
            for i in 0..n {
                penalized_rhs[i] += temp[i];
            }
        }
        // Solve (A + Bᵀ D B) x = penalized_rhs.
        let op = &self.penalized_mat;
        let apply = |xv: &[f64], y: &mut [f64]| spmv_into(op, xv, y);
        let res = match self.krylov {
            ConstrainedKrylov::CG => {
                if let Some(pc) = &self.precond {
                    let pc = pc.clone();
                    crate::iterative::solve_pcg_operator_precond(
                        n, apply, &penalized_rhs, x, |r, z| pc(r, z), &self.cfg,
                    )?
                } else {
                    let diag = self.diag.clone();
                    crate::iterative::solve_pcg_operator_precond(
                        n,
                        apply,
                        &penalized_rhs,
                        x,
                        |r, z| jacobi_apply(&diag, r, z),
                        &self.cfg,
                    )?
                }
            }
            ConstrainedKrylov::GMRES => {
                let pc = self.precond.clone();
                let (converged, iterations, final_residual) = gmres_core(
                    n,
                    &apply,
                    &|r: &[f64], z: &mut [f64]| match &pc {
                        Some(p) => p(r, z),
                        None => jacobi_apply(&self.diag, r, z),
                    },
                    &penalized_rhs,
                    x,
                    self.restart,
                    false,
                    &self.cfg,
                );
                SolveResult {
                    converged,
                    iterations,
                    final_residual,
                }
            }
        };
        // multiplier_sol = penalty·(B x − r).
        let mut bx = vec![0.0_f64; self.b.nrows];
        spmv_into(&self.b, x, &mut bx);
        for i in 0..self.b.nrows {
            bx[i] -= self.constraint_rhs.get(i).copied().unwrap_or(0.0);
            bx[i] *= self.penalty[i];
        }
        self.multiplier_sol = bx;
        Ok(res)
    }
}

// ─── Small dense/sparse helpers for the constraints family ────────────────

/// Jacobi (diagonal) preconditioner application `z = D⁻¹ r`.
fn jacobi_apply(diag: &[f64], r: &[f64], z: &mut [f64]) {
    for ((zi, &ri), &di) in z.iter_mut().zip(r.iter()).zip(diag.iter()) {
        *zi = if di.abs() > 1e-300 { ri / di } else { ri };
    }
}

/// Dense `B[rows, cols]` gather (row-major, `rows.len() × cols.len()`).
fn dense_submatrix(b: &CsrMatrix<f64>, rows: &[usize], cols: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows.len() * cols.len()];
    for (r, &gr) in rows.iter().enumerate() {
        for ptr in b.row_ptr[gr]..b.row_ptr[gr + 1] {
            let gc = b.col_idx[ptr] as usize;
            if let Some(c) = cols.iter().position(|&x| x == gc) {
                out[r * cols.len() + c] = b.values[ptr];
            }
        }
    }
    out
}

/// Transpose of a dense row-major matrix.
fn transpose_dense(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// In-place dense LU factorization with partial pivoting (`None` if
/// singular).
pub(crate) fn dense_lu_factor(a: &mut [f64], n: usize, ipiv: &mut [i32]) -> Option<()> {
    for k in 0..n {
        let mut p = k;
        let mut maxv = a[k * n + k].abs();
        for i in k + 1..n {
            let v = a[i * n + k].abs();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if maxv < 1e-300 {
            return None;
        }
        ipiv[k] = p as i32;
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
        }
        let piv = a[k * n + k];
        for i in k + 1..n {
            let m = a[i * n + k] / piv;
            a[i * n + k] = m;
            for c in k + 1..n {
                a[i * n + c] -= m * a[k * n + c];
            }
        }
    }
    Some(())
}

/// Solve `A X = B` in place (`B` is `n × nrhs`, row-major) given LU
/// factors of `A`.
pub(crate) fn dense_lu_solve(lu: &[f64], ipiv: &[i32], n: usize, b: &mut [f64], nrhs: usize) {
    for k in 0..n {
        let p = ipiv[k] as usize;
        if p != k {
            for j in 0..nrhs {
                b.swap(k * nrhs + j, p * nrhs + j);
            }
        }
    }
    for col in 0..nrhs {
        for i in 1..n {
            let mut acc = b[i * nrhs + col];
            for k in 0..i {
                acc -= lu[i * n + k] * b[k * nrhs + col];
            }
            b[i * nrhs + col] = acc;
        }
    }
    for col in 0..nrhs {
        for i in (0..n).rev() {
            let mut acc = b[i * nrhs + col];
            for k in i + 1..n {
                acc -= lu[i * n + k] * b[k * nrhs + col];
            }
            b[i * nrhs + col] = acc / lu[i * n + i];
        }
    }
}

#[cfg(test)]
mod tests;
