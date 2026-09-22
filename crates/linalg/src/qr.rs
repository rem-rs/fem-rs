//! Dense Householder QR — the minimal LAPACK face of MFEM's `NNLSSolver`
//! (D533(b)).
//!
//! MFEM's `NNLSSolver` (`linalg/solvers.cpp`, `#ifdef MFEM_USE_LAPACK`) calls
//! exactly four LAPACK/BLAS kernels:
//!
//! * `dgeqrf` — QR factorization.  The NNLS matrices are small
//!   (`min(m,n) ≤ ILAENV block size 32`), so LAPACK dispatches to the
//!   **unblocked** `dgeqr2`; that is what is ported here as
//!   [`qr_factor`].
//! * `dormqr` — apply the leading `k` Householder reflectors
//!   (`Qᵀ·C` / `Q·C`).  Again small sizes run the unblocked `dorm2r`;
//!   ported as [`apply_q_transpose`] / [`apply_q`].
//! * `dtrsm`  — upper-triangular solve `R·x = b` for a single right-hand
//!   side ([`solve_upper_triangular`], netlib `dtrsm` operation order).
//! * `dgemv`  — plain matrix-vector products, kept inline in the NNLS
//!   module (`nnls.rs`).
//!
//! Storage is **column-major** throughout (LAPACK convention, MFEM
//! `DenseMatrix::Data` layout, same choice as [`crate::dense::CholeskyFactors`]).
//! The operations replicate the netlib reference algorithms operation by
//! operation (scaled 2-norm, `dlapy2`-style hypotenuse, reflector
//! generation and application), so results agree with a netlib-LAPACK +
//! reference-BLAS build to the last ulp for well-scaled inputs.  Against
//! optimized BLAS (e.g. OpenBLAS, which the MFEM LAPACK reference build
//! links) agreement is at the rounding-noise level rather than bit-exact;
//! the NNLS discrete path is insensitive to that in practice.
//!
//! The *blocked* drivers (D561) are ported as well: [`qr_factor_blocked`]
//! (LAPACK `dgeqrf` — `dgeqr2` panels of width `NB = 32` with the
//! [`ILAENV(1,'DGEQRF')`] crossover `NX = 128`, `dlarft` triangular factors
//! and `dlarfb` blocked trailing updates) and [`apply_q_transpose_blocked`] /
//! [`apply_q_blocked`] (LAPACK `dormqr`, which blocks for `k > 32` and has no
//! crossover).  All of them replicate the netlib reference-BLAS operation
//! order (`dgemv`/`dtrmv`/`dgemm`/`dtrmm` loops), so with a netlib-LAPACK +
//! reference-BLAS build they agree to the last bit (see the D561 golden
//! tests).  Against optimized BLAS (OpenBLAS) the results differ at the
//! rounding-noise level, exactly like the unblocked path.

use fem_core::{FemError, FemResult};

// -------------------------------------------------------------------------
// BLAS/LAPACK auxiliary kernels (netlib reference semantics)
// -------------------------------------------------------------------------

/// `dnrm2` (reference BLAS, LAPACK 3.12 `dnrm2.f90`): the 3-accumulator
/// (abig/asml/amed) sum of squares.  For mid-range inputs this is the plain
/// ascending `sqrt(Σ xᵢ²)`.
fn dnrm2(x: &[f64]) -> f64 {
    // Blue's scaling constants (f64): tbig = 2^486, tsml = 2^-511,
    // ssml = 2^537, sbig = 2^-538 (exact powers of two).
    let (tbig, tsml, ssml, sbig) = (
        2f64.powi(486),
        2f64.powi(-511),
        2f64.powi(537),
        2f64.powi(-538),
    );
    const MAXN: f64 = f64::MAX;

    if x.is_empty() {
        return 0.0;
    }
    let mut abig = 0.0_f64;
    let mut asml = 0.0_f64;
    let mut amed = 0.0_f64;
    let mut notbig = true;
    for &xi in x {
        let ax = xi.abs();
        if ax > tbig {
            abig += (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if ax < tsml {
            if notbig {
                asml += (ax * ssml) * (ax * ssml);
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            amed = amed.sqrt();
            asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            scl = 1.0;
            sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        // All values mid-range.
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// `dlapy2`: `sqrt(x² + y²)` in the overflow-safe LAPACK form.
fn dlapy2(x: f64, y: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let w = xabs.max(yabs);
    let z = xabs.min(yabs);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// Fortran `SIGN(z, a)`: `|z|` with the sign of `a` (`a >= 0` → `+|z|`).
fn fortran_sign(z: f64, a: f64) -> f64 {
    if a < 0.0 {
        -z.abs()
    } else {
        z.abs()
    }
}

/// `dlarfg` (LAPACK 3.12): generate an elementary reflector
/// `H = I - tau·v·vᵀ` such that `H·(alpha, x) = (beta, 0, …, 0)`.
///
/// On entry `x` holds the `n-1` subdiagonal entries (`n = x.len() + 1`); on
/// exit `x` holds the reflector vector `v[1..n]` scaled by
/// `1/(alpha - beta)` (the implicit `v[0] == 1` is not stored) and `alpha`
/// is overwritten with `beta`.  `tau` receives the scalar factor.  An empty
/// `x` yields `H = I` (`tau = 0`, everything untouched), as does `n <= 1`
/// in LAPACK.
fn larfg(alpha: &mut f64, x: &mut [f64], tau: &mut f64) {
    let safmin = f64::MIN_POSITIVE / f64::EPSILON; // DLAMCH('S')/DLAMCH('E')
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        // H = I.
        *tau = 0.0;
        return;
    }
    let mut alpha_s = *alpha;
    let mut beta = -fortran_sign(dlapy2(alpha_s, xnorm), alpha_s);
    let mut knt = 0_u32;
    if beta.abs() < safmin {
        // XNORM, BETA may be inaccurate; scale X and recompute them.
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi *= rsafmn;
            }
            beta *= rsafmn;
            alpha_s *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        let xnorm = dnrm2(x);
        beta = -fortran_sign(dlapy2(alpha_s, xnorm), alpha_s);
    }
    *tau = (beta - alpha_s) / beta;
    let scal = 1.0 / (alpha_s - beta);
    for xi in x.iter_mut() {
        *xi *= scal;
    }
    // If ALPHA was subnormal it may have lost relative accuracy.
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
}

/// `dlarf` with `side = 'L'` (LAPACK 3.12): apply `H = I - tau·v·vᵀ` from
/// the left to the `m × nc` column-major matrix `c` (leading dimension
/// `ldc`), `v` the reflector with implicit leading 1.  Mirrors the
/// `dgemv('T') + dger` pair exactly, including the `dger` product grouping
/// `A(i,j) += V(i)·((-tau)·WORK(j))` and its zero-column skip.
fn larf_left(m: usize, nc: usize, v: &[f64], tau: f64, c: &mut [f64], ldc: usize) {
    if tau == 0.0 || nc == 0 {
        return;
    }
    // WORK = Cᵀ·V (dgemv 'T' with alpha = 1, beta = 0: plain ascending dots).
    let mut work = vec![0.0_f64; nc];
    for j in 0..nc {
        let mut t = 0.0_f64;
        for i in 0..m {
            t += c[i + j * ldc] * v[i];
        }
        work[j] = t;
    }
    // C := C - tau·V·WORKᵀ (dger with alpha = -tau).
    for j in 0..nc {
        if work[j] != 0.0 {
            let temp = -tau * work[j];
            for i in 0..m {
                c[i + j * ldc] += v[i] * temp;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Unblocked Householder QR of a column-major `m × n` matrix, in place
/// (LAPACK `dgeqr2`, i.e. what `dgeqrf` runs for `min(m,n)` up to its
/// block-size threshold).
///
/// On exit the strict upper triangle holds `R` (including the diagonal),
/// the strict lower triangle holds the reflector vectors `v[1..]`, and
/// `tau[k]` (length ≥ `min(m,n)`) the reflector scalars.  `Q` is the product
/// of the `min(m,n)` elementary reflectors
/// `Hₖ = I - tau[k]·vₖ·vₖᵀ` with `vₖ` stored in column `k`.
pub fn qr_factor(a: &mut [f64], m: usize, n: usize, tau: &mut [f64]) -> FemResult<()> {
    if tau.len() < m.min(n) {
        return Err(FemError::DimMismatch {
            expected: m.min(n),
            actual: tau.len(),
        });
    }
    dgeqr2_block(a, m, tau, 0, 0, m, n);
    Ok(())
}

/// Apply the leading `k` Householder reflectors produced by [`qr_factor`]
/// from the left to the `m × nc` column-major matrix `c`
/// (`dorm2r` with `side = 'L', trans = 'T'`): `C ← Qₖᵀ·C`.  Since
/// `Q = H₀·H₁···Hₖ₋₁`, the transpose product applies the reflectors in
/// generation order (ascending `i`), each on rows `i..m`.
#[allow(clippy::too_many_arguments)] // argument list mirrors LAPACK dorm2r
pub fn apply_q_transpose(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    for i in 0..k {
        // v = [1, a(i+1..m, i)] with the implicit leading one.
        let mut v = vec![0.0_f64; m - i];
        v[0] = 1.0;
        v[1..].copy_from_slice(&a[i + 1 + i * lda..i * lda + m]);
        larf_left(m - i, nc, &v, tau[i], &mut c[i..], ldc);
    }
}

/// Apply the leading `k` Householder reflectors in reverse generation order
/// (`dorm2r` with `side = 'L', trans = 'N'`): `C ← Qₖ·C`, i.e. the
/// reflectors run from the last one down to the first.
// The argument list mirrors LAPACK's dorm2r parameter order (k, A, lda,
// tau, C, m, nc, ldc).
#[allow(clippy::too_many_arguments)]
pub fn apply_q(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    for i in (0..k).rev() {
        let mut v = vec![0.0_f64; m - i];
        v[0] = 1.0;
        v[1..].copy_from_slice(&a[i + 1 + i * lda..i * lda + m]);
        larf_left(m - i, nc, &v, tau[i], &mut c[i..], ldc);
    }
}

/// Solve `R·x = b` in place for the `n × n` leading upper triangle of the
/// column-major matrix `r` (leading dimension `ldr`), single right-hand
/// side.  Netlib `dtrsm` operation order for `side = 'L', uplo = 'U',
/// transa = 'N', diag = 'N'`: sweep `k` bottom-up, divide by the diagonal,
/// then rank-1-update all preceding entries.
pub fn solve_upper_triangular(r: &[f64], ldr: usize, b: &mut [f64], n: usize) {
    for k in (0..n).rev() {
        b[k] /= r[k + k * ldr];
        let bk = b[k];
        if bk != 0.0 {
            for i in 0..k {
                b[i] -= bk * r[i + k * ldr];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Blocked path (D561): LAPACK dgeqrf / dormqr panel machinery
// -------------------------------------------------------------------------
//
// Everything below replicates the LAPACK 3.12 blocked drivers operation by
// operation, including the netlib reference-BLAS loop orders of the `dgemv`,
// `dtrmv`, `dgemm` and `dtrmm` calls the drivers make.  The blocked and
// unblocked paths round differently (a block update `C -= V·Tᵀ·Vᵀ·C` is not
// the same floating-point sequence as the individual `dlarf`s), so the
// blocked results are pinned against a netlib-LAPACK + reference-BLAS
// oracle, not against [`qr_factor`].

/// `ILAENV(1, 'DGEQRF', ...)` (LAPACK 3.12 `ilaenv.f`, `C2='GE', C3='QRF'`):
/// the panel width of the blocked code.
const DGEQRF_NB: usize = 32;
/// `ILAENV(3, 'DGEQRF', ...)`: crossover point — once `min(m,n)` is within
/// `NX` columns of the tail, the remaining block is factored by plain
/// `dgeqr2`.
const DGEQRF_NX: usize = 128;
/// `ILAENV(1, 'DORMQR', ...)` capped by 3.12's `NBMAX = 64`: `min(64, 32)`.
/// `dormqr` consults no crossover (`ISPEC = 3`); it blocks whenever
/// `NB < k`.
const DORMQR_NB: usize = 32;

/// LAPACK `dgeqr2` on the `n_rows × n_cols` block of the column-major
/// matrix `a` (leading dimension `lda`) whose top-left corner is
/// `(row0, col0)`: the panel factorization used by both the unblocked and
/// the blocked `dgeqrf`.  `tau[col0..]` receives the reflector scalars.
fn dgeqr2_block(
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
    row0: usize,
    col0: usize,
    n_rows: usize,
    n_cols: usize,
) {
    // dgeqr2 factors K = MIN(M, N) columns, but each Hₖ is applied to all
    // remaining N-K columns of the invocation (`DLARF('Left', M-K+1, N-K,
    // ...)` uses the uncapped N).
    let kmax = n_cols.min(n_rows);
    for kk in 0..kmax {
        let col = col0 + kk;
        let head = row0 + kk + col * lda;
        let n_elems = n_rows - kk;
        {
            let (head_v, tail) = a[head..].split_at_mut(1);
            larfg(&mut head_v[0], &mut tail[..n_elems - 1], &mut tau[col]);
        }
        if kk + 1 < n_cols {
            let aii = a[head];
            a[head] = 1.0;
            let v: Vec<f64> = a[head..head + n_elems].to_vec();
            larf_left(
                n_elems,
                n_cols - kk - 1,
                &v,
                tau[col],
                &mut a[head + lda..],
                lda,
            );
            a[head] = aii;
        }
    }
}

/// `dgeqrf` with the blocked-code workspace regime (the workspace-query
/// allocation `lwork = n·nb`, i.e. no `nb` reduction): factor `dgeqr2`
/// panels of width `nb = 32` while `min(m,n)` exceeds the `nx = 128`
/// crossover, form each panel's triangular factor with `dlarft` and apply it
/// to the trailing columns with `dlarfb`, then finish the tail with the
/// unblocked code.  Same factorization contract as [`qr_factor`], but for
/// `min(m,n) > 128` the floating-point results differ from it (block updates
/// round differently) and match LAPACK's blocked `dgeqrf` instead.
pub fn qr_factor_blocked(a: &mut [f64], m: usize, n: usize, tau: &mut [f64]) -> FemResult<()> {
    if tau.len() < m.min(n) {
        return Err(FemError::DimMismatch {
            expected: m.min(n),
            actual: tau.len(),
        });
    }
    let k = m.min(n);
    let nb = DGEQRF_NB;
    let mut i = 0;
    if nb > 1 && nb < k && DGEQRF_NX < k {
        let mut t = vec![0.0_f64; nb * nb];
        let mut w = vec![0.0_f64; nb * n];
        while i < k - DGEQRF_NX {
            let ib = (k - i).min(nb);
            dgeqr2_block(a, m, tau, i, i, m - i, ib);
            if i + ib < n {
                let n_cols = n - i - ib;
                larft_forward_columnwise(a, m, tau, &mut t, nb, i, i, m - i, ib);
                dlarfb_left_f_c_inplace(
                    a,
                    m,
                    i,
                    i,
                    m - i,
                    ib,
                    n_cols,
                    true,
                    &t,
                    nb,
                    &mut w,
                    n_cols,
                );
            }
            i += nb;
        }
    }
    if i < k {
        dgeqr2_block(a, m, tau, i, i, m - i, n - i);
    }
    Ok(())
}

/// `dlarft('Forward', 'Columnwise')`: the upper-triangular factor `T` (`k ×
/// k`, leading dimension `ldt`) of the block reflector built from the `k`
/// Householder vectors stored below the diagonal of the `n_rows × k` block
/// of `a` at `(row0, col0)`.  LAPACK 3.12 `dlarft.f`, including the trailing
/// zero skip (`lastv`) and the `prevlastv` cutoff of the `dgemv`.
#[allow(clippy::too_many_arguments)] // argument list mirrors LAPACK dlarft
fn larft_forward_columnwise(
    a: &[f64],
    lda: usize,
    tau: &[f64],
    t: &mut [f64],
    ldt: usize,
    row0: usize,
    col0: usize,
    n_rows: usize,
    k: usize,
) {
    let mut prevlastv = n_rows;
    for i in 0..k {
        prevlastv = prevlastv.max(i + 1);
        if tau[col0 + i] == 0.0 {
            // H(i) = I.
            for j in 0..=i {
                t[j + i * ldt] = 0.0;
            }
        } else {
            // Skip any trailing zeros of V(i+1.., i).
            let mut lastv = n_rows;
            while lastv > i + 1 && a[row0 + lastv - 1 + (col0 + i) * lda] == 0.0 {
                lastv -= 1;
            }
            // T(1:i-1, i) = -tau(i) * V(i, 1:i-1).
            for j in 0..i {
                t[j + i * ldt] = -tau[col0 + i] * a[row0 + i + (col0 + j) * lda];
            }
            let j2 = lastv.min(prevlastv);
            // T(1:i-1, i) += -tau(i) * V(i+1:j2, 1:i-1)**T * V(i+1:j2, i)
            // (`dgemv('T')` with alpha = -tau, beta = one; the reference
            // dgemv accumulates the ascending dot then adds alpha·temp).
            let xbase = row0 + i + 1 + (col0 + i) * lda;
            for j in 0..i {
                let mut temp = 0.0;
                for l in 0..(j2 - i - 1) {
                    temp += a[row0 + i + 1 + l + (col0 + j) * lda] * a[xbase + l];
                }
                t[j + i * ldt] += -tau[col0 + i] * temp;
            }
            // T(1:i-1, i) := T(1:i-1, 1:i-1) * T(1:i-1, i)
            // (`dtrmv('U', 'N', 'N')`, reference saxpy form with the zero
            // skip — in place on T's own column).
            for j in 0..i {
                let xj = i * ldt + j;
                if t[xj] != 0.0 {
                    let temp = t[xj];
                    for l in 0..j {
                        t[i * ldt + l] += temp * t[l + j * ldt];
                    }
                    t[xj] *= t[j + j * ldt];
                }
            }
            t[i + i * ldt] = tau[col0 + i];
            if i > 0 {
                prevlastv = prevlastv.max(lastv);
            } else {
                prevlastv = lastv;
            }
        }
    }
}

/// `dlarfb('Left', TRANS, 'Forward', 'Columnwise')` with columnwise-stored
/// reflectors, in-place flavor (LAPACK `dgeqrf`): `V` is the `n_rows × k`
/// block of `a` at `(row0, col0)` and the target `C` the `n_rows × n_cols`
/// block at `(row0, col0 + k)` of the same matrix — the two never overlap.
/// See [`dlarfb_left_f_c_kernel`].
#[allow(clippy::too_many_arguments)]
fn dlarfb_left_f_c_inplace(
    a: &mut [f64],
    lda: usize,
    row0: usize,
    col0: usize,
    n_rows: usize,
    k: usize,
    n_cols: usize,
    trans_is_t: bool,
    t: &[f64],
    ldt: usize,
    w: &mut [f64],
    ldw: usize,
) {
    let v = copy_v(a, lda, row0, col0, n_rows, k);
    dlarfb_left_f_c_kernel(
        a,
        lda,
        row0,
        col0 + k,
        &v,
        n_rows,
        n_rows,
        k,
        n_cols,
        trans_is_t,
        t,
        ldt,
        w,
        ldw,
    );
}

/// Out-of-place flavor (LAPACK `dormqr`): `V` lives in `a` at
/// `(v_row0, v_col0)`, the target `C` is the separate matrix `c`.
/// See [`dlarfb_left_f_c_kernel`].
#[allow(clippy::too_many_arguments)]
fn dlarfb_left_f_c_split(
    a: &[f64],
    lda: usize,
    v_row0: usize,
    v_col0: usize,
    c: &mut [f64],
    ldc: usize,
    c_row0: usize,
    c_col0: usize,
    n_rows: usize,
    k: usize,
    n_cols: usize,
    trans_is_t: bool,
    t: &[f64],
    ldt: usize,
    w: &mut [f64],
    ldw: usize,
) {
    let v = copy_v(a, lda, v_row0, v_col0, n_rows, k);
    dlarfb_left_f_c_kernel(
        c, ldc, c_row0, c_col0, &v, n_rows, n_rows, k, n_cols, trans_is_t, t, ldt, w, ldw,
    );
}

/// Copy the strictly-lower part of the `n_rows × k` reflector block at
/// `(row0, col0)` into a compact `ldv = n_rows` working copy (the algorithm
/// only reads V, so the copy is value-neutral).
fn copy_v(a: &[f64], lda: usize, row0: usize, col0: usize, n_rows: usize, k: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; n_rows * k];
    for j in 0..k {
        for l in 0..n_rows {
            v[l + j * n_rows] = a[row0 + l + (col0 + j) * lda];
        }
    }
    v
}

/// The `dlarfb` side-L columnwise computation on the compact reflector copy
/// `v` (`ldv` leading dimension): apply the block reflector `H = H(0)···H(k-1)`
/// (`TRANS = 'N'`, `trans_is_t == false`) or `Hᵀ` (`TRANS = 'T'`,
/// `trans_is_t == true`) to the `n_rows × n_cols` block of `c` at
/// `(c_row0, c_col0)`.  `t` (`k × k`, leading dimension `ldt`) comes from
/// [`larft_forward_columnwise`]; `w` (`n_cols × k`, leading dimension `ldw`)
/// is workspace.  LAPACK 3.12 `dlarfb.f` side-L columnwise path with the
/// netlib `dtrmm`/`dgemm` loop orders.
#[allow(clippy::too_many_arguments)]
fn dlarfb_left_f_c_kernel(
    c: &mut [f64],
    ldc: usize,
    c_row0: usize,
    c_col0: usize,
    v: &[f64],
    ldv: usize,
    n_rows: usize,
    k: usize,
    n_cols: usize,
    trans_is_t: bool,
    t: &[f64],
    ldt: usize,
    w: &mut [f64],
    ldw: usize,
) {
    // W := C1ᵀ (one `dcopy` per column of W / row of C1).
    for j in 0..k {
        for i in 0..n_cols {
            w[i + j * ldw] = c[c_row0 + j + (c_col0 + i) * ldc];
        }
    }
    // W := W * V1 (`dtrmm('R','L','N','U')`, alpha = one: the multiply by
    // TEMP = one is an exact no-op and is skipped).
    for j in 0..k {
        for k2 in (j + 1)..k {
            let vk = v[k2 + j * ldv];
            if vk != 0.0 {
                for i in 0..n_cols {
                    w[i + j * ldw] += vk * w[i + k2 * ldw];
                }
            }
        }
    }
    // W += C2ᵀ * V2 (`dgemm('T','N')`, alpha = beta = one: reference dot
    // accumulation, then `C = ALPHA*TEMP + BETA*C`).
    for j in 0..k {
        for i in 0..n_cols {
            let mut temp = 0.0;
            for l in 0..(n_rows - k) {
                temp += c[c_row0 + k + l + (c_col0 + i) * ldc] * v[(k + l) + j * ldv];
            }
            w[i + j * ldw] += temp;
        }
    }
    // W := W * T**T (TRANS = 'T', i.e. Hᵀ applied — TRANST = 'N') or
    // W * T (TRANS = 'N' — TRANST = 'T')
    // (`dtrmm('R','U',TRANST,'N')`, alpha = one).
    if trans_is_t {
        for j in (0..k).rev() {
            let temp0 = t[j + j * ldt];
            for i in 0..n_cols {
                w[i + j * ldw] *= temp0;
            }
            for k2 in 0..j {
                let tk = t[k2 + j * ldt];
                if tk != 0.0 {
                    for i in 0..n_cols {
                        w[i + j * ldw] += tk * w[i + k2 * ldw];
                    }
                }
            }
        }
    } else {
        for k2 in 0..k {
            for j in 0..k2 {
                let tk = t[j + k2 * ldt];
                if tk != 0.0 {
                    for i in 0..n_cols {
                        w[i + j * ldw] += tk * w[i + k2 * ldw];
                    }
                }
            }
            let temp0 = t[k2 + k2 * ldt];
            if temp0 != 1.0 {
                for i in 0..n_cols {
                    w[i + k2 * ldw] *= temp0;
                }
            }
        }
    }
    // C2 -= V2 * W**T (`dgemm('N','T')`, alpha = -one, beta = one).
    for j in 0..n_cols {
        for l in 0..k {
            let temp = -w[j + l * ldw];
            for i in 0..(n_rows - k) {
                c[c_row0 + k + i + (c_col0 + j) * ldc] += temp * v[(k + i) + l * ldv];
            }
        }
    }
    // W := W * V1**T (`dtrmm('R','L','T','U')`, alpha = one; the unit
    // diagonal keeps TEMP = one so the final rescale is skipped).
    for k2 in (0..k).rev() {
        for j in (k2 + 1)..k {
            let vk = v[j + k2 * ldv];
            if vk != 0.0 {
                for i in 0..n_cols {
                    w[i + j * ldw] += vk * w[i + k2 * ldw];
                }
            }
        }
    }
    // C1 -= W**T.
    for j in 0..k {
        for i in 0..n_cols {
            let idx = c_row0 + j + (c_col0 + i) * ldc;
            c[idx] -= w[i + j * ldw];
        }
    }
}

/// Blocked counterpart of [`apply_q_transpose`]: LAPACK `dormqr` with
/// `side = 'L', trans = 'T'`.  `dormqr` has no crossover — it partitions the
/// `k` reflectors into blocks of `nb = 32` (`nb < k` required) and applies
/// each with `dlarft` + `dlarfb`, first to last; `k ≤ nb` falls back to the
/// unblocked [`apply_q_transpose`].
#[allow(clippy::too_many_arguments)] // argument list mirrors LAPACK dormqr
pub fn apply_q_transpose_blocked(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    apply_q_blocked_impl(k, a, lda, tau, c, m, nc, ldc, true);
}

/// Blocked counterpart of [`apply_q`]: LAPACK `dormqr` with
/// `side = 'L', trans = 'N'` — the blocks run from the last one backwards.
#[allow(clippy::too_many_arguments)] // argument list mirrors LAPACK dormqr
pub fn apply_q_blocked(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
) {
    apply_q_blocked_impl(k, a, lda, tau, c, m, nc, ldc, false);
}

/// Shared body of [`apply_q_transpose_blocked`] / [`apply_q_blocked`].
#[allow(clippy::too_many_arguments)]
fn apply_q_blocked_impl(
    k: usize,
    a: &[f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    m: usize,
    nc: usize,
    ldc: usize,
    trans_is_t: bool,
) {
    let nb = DORMQR_NB;
    if nb >= k {
        // Unblocked code (dorm2r).
        if trans_is_t {
            apply_q_transpose(k, a, lda, tau, c, m, nc, ldc);
        } else {
            apply_q(k, a, lda, tau, c, m, nc, ldc);
        }
        return;
    }
    let mut t = vec![0.0_f64; nb * nb];
    let mut w = vec![0.0_f64; nb * nc];
    if trans_is_t {
        // I1 = 1, I2 = K, I3 = NB (forward).
        let mut i = 0;
        while i < k {
            let ib = nb.min(k - i);
            larft_forward_columnwise(a, lda, tau, &mut t, nb, i, i, m - i, ib);
            dlarfb_left_f_c_split(
                a,
                lda,
                i,
                i,
                c,
                ldc,
                i,
                0,
                m - i,
                ib,
                nc,
                true,
                &t,
                nb,
                &mut w,
                nc,
            );
            i += nb;
        }
    } else {
        // I1 = ((K-1)/NB)*NB + 1, I2 = 1, I3 = -NB (backward).
        let mut i = ((k - 1) / nb) * nb;
        loop {
            let ib = nb.min(k - i);
            larft_forward_columnwise(a, lda, tau, &mut t, nb, i, i, m - i, ib);
            dlarfb_left_f_c_split(
                a,
                lda,
                i,
                i,
                c,
                ldc,
                i,
                0,
                m - i,
                ib,
                nc,
                false,
                &t,
                nb,
                &mut w,
                nc,
            );
            if i == 0 {
                break;
            }
            i -= nb;
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (same as the C++ LAPACK probe in tmp/d533) so the
    /// test matrices match the probe-generated ones bit for bit.
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

    fn random_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut g = Lcg(seed);
        (0..m * n).map(|_| g.next_f64()).collect()
    }

    #[test]
    fn qr_rectangular_reconstructs_a() {
        // A = Q·R must reproduce A; Q from applying the reflectors to the
        // identity columns.
        for (m, n) in [(6usize, 4usize), (5, 5), (4, 6), (1, 3), (7, 1)] {
            let a = random_matrix(m, n, 1000 + (m * 31 + n) as u64);
            let mut fac = a.clone();
            let mut tau = vec![0.0_f64; m.min(n)];
            qr_factor(&mut fac, m, n, &mut tau).expect("qr_factor");

            // Build Q explicitly: apply the reflectors (transposed order for
            // Q itself) to each identity column.
            let mut q = vec![0.0_f64; m * m];
            for j in 0..m {
                q[j + j * m] = 1.0;
            }
            for j in 0..m {
                let mut col = vec![0.0_f64; m];
                col.copy_from_slice(&q[j * m..j * m + m]);
                apply_q(m.min(n), &fac, m, &tau, &mut col, m, 1, m);
                q[j * m..j * m + m].copy_from_slice(&col);
            }
            // A - Q·R
            let mut err = 0.0_f64;
            let mut nrm = 0.0_f64;
            for j in 0..n {
                for i in 0..m {
                    let mut qr_ij = 0.0_f64;
                    for k in 0..m.min(n) {
                        qr_ij += q[i + k * m] * if k <= j { fac[k + j * m] } else { 0.0 };
                    }
                    err = err.max((qr_ij - a[i + j * m]).abs());
                    nrm = nrm.max(a[i + j * m].abs());
                }
            }
            assert!(
                err <= 1e-13 * nrm.max(1.0),
                "({m},{n}) reconstruction error {err:.3e}"
            );

            // Qᵀ·Q = I
            let mut orth = 0.0_f64;
            for j in 0..m {
                for i in 0..m {
                    let mut d = 0.0_f64;
                    for k in 0..m {
                        d += q[k + i * m] * q[k + j * m];
                    }
                    orth = orth.max((d - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(orth <= 1e-13, "({m},{n}) orthogonality error {orth:.3e}");
        }
    }

    #[test]
    fn qr_apply_q_transpose_is_inverse_of_apply_q() {
        let m = 6;
        let n = 4;
        let a = random_matrix(m, n, 42);
        let mut fac = a.clone();
        let mut tau = vec![0.0_f64; m.min(n)];
        qr_factor(&mut fac, m, n, &mut tau).expect("qr_factor");

        // C ← Qᵀ C then C ← Q C must return the original C.
        let mut c = random_matrix(m, 3, 7);
        let c0 = c.clone();
        apply_q_transpose(m.min(n), &fac, m, &tau, &mut c, m, 3, m);
        apply_q(m.min(n), &fac, m, &tau, &mut c, m, 3, m);
        for (x, y) in c.iter().zip(&c0) {
            assert!((x - y).abs() <= 1e-14, "Q Qᵀ round-trip {x} vs {y}");
        }
    }

    #[test]
    fn solve_upper_triangular_matches_back_substitution() {
        // R upper triangular (column-major), known solution x.
        #[rustfmt::skip]
        let r = vec![
            2.0, 0.5, 0.25, // col 0
            0.0, -1.5, 0.5, // col 1
            0.0, 0.0,  3.0, // col 2
        ];
        let x = [1.0, -2.0, 0.5];
        let mut b = [0.0_f64; 3];
        for j in 0..3 {
            for i in 0..=j {
                b[i] += r[i + j * 3] * x[j];
            }
        }
        solve_upper_triangular(&r, 3, &mut b, 3);
        for (got, want) in b.iter().zip(&x) {
            assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
        }
    }

    /// Golden values dumped by the LAPACK probe `tmp/d533/d533_qr_probe.cpp`
    /// (MFEM 4.10 LAPACK build: netlib dgeqrf + OpenBLAS kernels) for the
    /// probe's seeded matrices: R diagonal and tau vectors must agree to
    /// rounding noise.
    #[test]
    fn qr_matches_lapack_probe_golden() {
        // Probe case (m=6, n=4, seed 12345): tau values and R diagonal from
        // `dgeqrf` (probe output, LAPACK 3.12.0 + OpenBLAS, 2026-09-21).
        let tau_golden: [f64; 4] = [
            1.5505122211865232,
            1.2448066214047795,
            1.1873890551513562,
            1.971849343662508,
        ];
        let rdiag_golden: [f64; 4] = [
            1.4183931945162893,
            1.7149207833232352,
            -1.053766558962163,
            1.2216953772482007,
        ];
        let a = random_matrix(6, 4, 12345);
        let mut fac = a.clone();
        let mut tau = vec![0.0_f64; 4];
        qr_factor(&mut fac, 6, 4, &mut tau).expect("qr_factor");
        for k in 0..4 {
            assert!(
                (tau[k] - tau_golden[k]).abs() <= 1e-12 * tau_golden[k].abs().max(1.0),
                "tau[{k}] {} vs golden {}",
                tau[k],
                tau_golden[k]
            );
            let rd = fac[k + k * 6];
            assert!(
                (rd - rdiag_golden[k]).abs() <= 1e-12 * rdiag_golden[k].abs(),
                "R[{k},{k}] {rd} vs golden {}",
                rdiag_golden[k]
            );
        }

        // dormqr('L','T') / dormqr('L','N') of a seeded 6x2 matrix C
        // (probe output, LAPACK 3.12.0 + reference netlib BLAS 3.12.0 via
        // `LD_LIBRARY_PATH`, 2026-09-21).  The port is compared **exactly**:
        // with the reference BLAS the whole pipeline is deterministic, and
        // the port replicates its operation order bit for bit.
        let qt_golden: [f64; 12] = [
            -0.60857618693025428,
            -0.50363620712832113,
            -0.88223798848879487,
            0.86963877978091797,
            -0.047430846374018162,
            0.69981940104482954,
            1.2584458193488615,
            -0.06495726628684928,
            -0.26961992229261111,
            0.49446031325655948,
            -0.6091217302640336,
            0.85386074106241772,
        ];
        let q_golden: [f64; 12] = [
            0.28309652046929079,
            1.1433012274646133,
            0.3942152696725994,
            -0.39609998117693285,
            0.93366845832804302,
            0.28161633243703432,
            1.0864166206454875,
            0.55565175470787331,
            -0.27478846272045959,
            -0.5920523285427759,
            -0.75990314149086291,
            0.71600369302180833,
        ];
        let mut g = Lcg(777);
        let mut c = vec![0.0_f64; 12];
        for v in c.iter_mut() {
            *v = g.next_f64();
        }
        let mut ct = c.clone();
        apply_q_transpose(4, &fac, 6, &tau, &mut ct, 6, 2, 6);
        let mut cn = c;
        apply_q(4, &fac, 6, &tau, &mut cn, 6, 2, 6);
        for i in 0..12 {
            assert_eq!(ct[i], qt_golden[i], "QᵀC[{i}] exact");
            assert_eq!(cn[i], q_golden[i], "QC[{i}] exact");
        }
    }

    #[test]
    fn larfg_handles_trivial_columns() {
        // n <= 1 and zero subdiagonal -> H = I (tau = 0, alpha untouched).
        let mut a = 0.75_f64;
        let mut x = [0.0_f64; 3];
        let mut tau = 9.0_f64;
        larfg(&mut a, &mut x, &mut tau);
        assert_eq!(tau, 0.0);
        assert_eq!(a, 0.75);
        let mut tau = 9.0_f64;
        larfg(&mut a, &mut [], &mut tau);
        assert_eq!(tau, 0.0);
        assert_eq!(a, 0.75);
    }

    // -----------------------------------------------------------------
    // D561: blocked path (dgeqrf / dormqr) — bit-level golden tests.
    //
    // Golden oracle: LAPACK 3.12.0 + netlib reference BLAS 3.12.0 (the
    // `LD_LIBRARY_PATH` libblas3 override of tmp/d533; the same run under
    // OpenBLAS hashes differently — the D563 backend split extends to the
    // blocked path).  Probe: `tmp/d562/d561_qr_blocked_probe.cpp`, output
    // `tmp/d562/d561_qr_blocked_netlib.log` (2026-09-22).  All sizes take
    // the blocked code path (min(m,n) > NX = 128 for dgeqrf; k > 32 for
    // dormqr); equality is asserted on the f64 bit patterns.
    // -----------------------------------------------------------------

    /// FNV-1a 64 over the raw little-endian f64 bits (the probe's
    /// fingerprint of the full output array).
    fn fnv1a_bits(v: &[f64]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &val in v {
            for b in val.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    include!("qr_blocked_golden.rs");

    #[test]
    fn qr_blocked_matches_lapack_netlib_golden() {
        let cases: [(usize, usize, u64, u64, u64, &[u64], &[(usize, u64)]); 3] = [
            (
                210,
                200,
                12345,
                GE210X200_TAU_HASH,
                GE210X200_A_HASH,
                &GE210X200_TAU_BITS,
                &GE210X200_SAMPLE_BITS,
            ),
            (
                200,
                210,
                999,
                GE200X210_TAU_HASH,
                GE200X210_A_HASH,
                &GE200X210_TAU_BITS,
                &GE200X210_SAMPLE_BITS,
            ),
            (
                160,
                160,
                31337,
                GE160X160_TAU_HASH,
                GE160X160_A_HASH,
                &GE160X160_TAU_BITS,
                &GE160X160_SAMPLE_BITS,
            ),
        ];
        for &(m, n, seed, tau_hash, a_hash, tau_bits, samples) in &cases {
            let a0 = random_matrix(m, n, seed);
            let mut fac = a0.clone();
            let mut tau = vec![0.0_f64; m.min(n)];
            qr_factor_blocked(&mut fac, m, n, &mut tau).expect("qr_factor_blocked");

            assert_eq!(fnv1a_bits(&tau), tau_hash, "({m},{n}) tau hash");
            for (k, bits) in tau_bits.iter().enumerate() {
                assert_eq!(tau[k].to_bits(), *bits, "({m},{n}) tau[{k}] exact");
            }
            assert_eq!(fnv1a_bits(&fac), a_hash, "({m},{n}) factored A hash");
            for &(idx, bits) in samples {
                assert_eq!(fac[idx].to_bits(), bits, "({m},{n}) sample a[{idx}] exact");
            }

            // The factorization is still a valid QR (orthonality + A = Q·R).
            let mut q = vec![0.0_f64; m * m];
            for j in 0..m {
                q[j + j * m] = 1.0;
            }
            for j in 0..m {
                let mut col = vec![0.0_f64; m];
                col.copy_from_slice(&q[j * m..j * m + m]);
                apply_q_blocked(m.min(n), &fac, m, &tau, &mut col, m, 1, m);
                q[j * m..j * m + m].copy_from_slice(&col);
            }
            let mut err = 0.0_f64;
            let mut nrm = 0.0_f64;
            for j in 0..n {
                for i in 0..m {
                    let mut qr_ij = 0.0_f64;
                    for kk in 0..m.min(n) {
                        qr_ij += q[i + kk * m] * if kk <= j { fac[kk + j * m] } else { 0.0 };
                    }
                    err = err.max((qr_ij - a0[i + j * m]).abs());
                    nrm = nrm.max(a0[i + j * m].abs());
                }
            }
            assert!(
                err <= 1e-12 * nrm.max(1.0),
                "({m},{n}) reconstruction {err:.3e}"
            );
        }
    }

    #[test]
    fn qr_blocked_dispatches_to_unblocked_below_crossover() {
        // min(m,n) <= NX = 128 runs plain dgeqr2 (also when m is large):
        // the blocked driver must be bit-identical to qr_factor there.
        for (m, n, seed) in [
            (6usize, 4usize, 7001u64),
            (40, 30, 7002),
            (150, 100, 7003),
            (129, 90, 7004),
        ] {
            let a = random_matrix(m, n, seed);
            let mut f1 = a.clone();
            let mut f2 = a.clone();
            let mut t1 = vec![0.0_f64; m.min(n)];
            let mut t2 = vec![0.0_f64; m.min(n)];
            qr_factor(&mut f1, m, n, &mut t1).expect("qr_factor");
            qr_factor_blocked(&mut f2, m, n, &mut t2).expect("qr_factor_blocked");
            assert_eq!(f1, f2, "({m},{n}) dispatch");
            assert_eq!(t1, t2, "({m},{n}) dispatch tau");
        }
    }

    #[test]
    fn dormqr_blocked_matches_lapack_netlib_golden() {
        // k = 150 > NB = 32 blocks; both dormqr('L','T') and dormqr('L','N').
        let m = 160;
        let n = 3;
        let k = 150;
        let a = random_matrix(m, k, 777);
        let mut fac = a.clone();
        let mut tau = vec![0.0_f64; k];
        qr_factor_blocked(&mut fac, m, k, &mut tau).expect("factor");

        let mut g = Lcg(2024);
        let c0: Vec<f64> = (0..m * n).map(|_| g.next_f64()).collect();

        let mut ct = c0.clone();
        apply_q_transpose_blocked(k, &fac, m, &tau, &mut ct, m, n, m);
        assert_eq!(fnv1a_bits(&ct), DORMQR_CT_HASH, "QᵀC hash");
        for (idx, bits) in DORMQR_CT_SAMPLE_BITS {
            assert_eq!(ct[idx].to_bits(), bits, "QᵀC[{idx}] exact");
        }

        let mut cn = c0.clone();
        apply_q_blocked(k, &fac, m, &tau, &mut cn, m, n, m);
        assert_eq!(fnv1a_bits(&cn), DORMQR_CN_HASH, "QC hash");
        for (idx, bits) in DORMQR_CN_SAMPLE_BITS {
            assert_eq!(cn[idx].to_bits(), bits, "QC[{idx}] exact");
        }

        // Qᵀ then Q round-trips C (orthonality of the blocked application).
        let mut c = c0.clone();
        apply_q_transpose_blocked(k, &fac, m, &tau, &mut c, m, n, m);
        apply_q_blocked(k, &fac, m, &tau, &mut c, m, n, m);
        for (x, y) in c.iter().zip(&c0) {
            assert!((x - y).abs() <= 1e-13, "blocked Q Qᵀ round-trip {x} vs {y}");
        }
    }
}
