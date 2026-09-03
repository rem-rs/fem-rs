//! Quadrature tables for standard reference domains.
//!
//! All rules are exact for polynomials up to the stated degree.

use crate::reference::QuadratureRule;

// ─── Gauss-Legendre on [-1,1] ─────────────────────────────────────────────────

/// Gauss-Legendre points and weights on `[-1, 1]`, for `n` points (1 ≤ n ≤ 4).
///
/// These are used as building blocks for tensor-product rules (quad, hex).
fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => {
            let x = 1.0_f64 / 3.0_f64.sqrt();
            (vec![-x, x], vec![1.0, 1.0])
        }
        3 => {
            let x = (3.0_f64 / 5.0_f64).sqrt();
            (vec![-x, 0.0, x], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
        4 => {
            let a = (3.0 / 7.0 - 2.0 / 7.0 * (6.0_f64 / 5.0).sqrt()).sqrt();
            let b = (3.0 / 7.0 + 2.0 / 7.0 * (6.0_f64 / 5.0).sqrt()).sqrt();
            let wa = (18.0 + 30.0_f64.sqrt()) / 36.0;
            let wb = (18.0 - 30.0_f64.sqrt()) / 36.0;
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        _ => panic!("gauss_legendre_1d: only n=1..4 supported, got {n}"),
    }
}

/// Gauss-Legendre rule on `[0, 1]` with `n` points.
///
/// Nodes/weights for `n = 1..4` are hard-coded with the same high-precision
/// values as MFEM's `QuadratureFunctions1D::GaussLegendre` (which hard-codes
/// `n ≤ 3` and Newton-iterates `n ≥ 4`) — computing them as `0.5·(x+1)` from
/// the `[-1,1]` rule differs from MFEM by ~1 ulp and propagates into matrix
/// entries on curved meshes.  Weights sum to 1.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.5], vec![1.0]),
        2 => (
            vec![0.21132486540518710671, 0.78867513459481286553],
            vec![0.5, 0.5],
        ),
        3 => (
            vec![0.11270166537925831174, 0.5, 0.88729833462074170214],
            vec![
                0.27777777777777779011,
                0.44444444444444441977,
                0.27777777777777779011,
            ],
        ),
        4 => (
            vec![
                0.069431844202973699853,
                0.33000947820757187134,
                0.66999052179242812866,
                0.93056815579702634178,
            ],
            vec![
                0.1739274225687268971,
                0.32607257743127299188,
                0.32607257743127299188,
                0.1739274225687268971,
            ],
        ),
        // 5-point Gauss-Legendre on [0,1] — bit-identical to MFEM's
        // IntRules.Get(SEGMENT, 8) (Poly_1D::GaussLegendre).  The generic
        // Newton solver (gauss_legendre_01_arbitrary) differs by 1 ulp on
        // the outer nodes, which flips BlockILU's MDF tie-breaks (ex41).
        5 => (
            vec![
                0.0469100770306680043,
                0.230765344947158446,
                0.5,
                0.769234655052841498,
                0.953089922969332037,
            ],
            vec![
                0.118463442528094556,
                0.239314335249683235,
                0.284444444444444444,
                0.239314335249683235,
                0.118463442528094556,
            ],
        ),
        _ => {
            let (xs, ws) = gauss_legendre_01_arbitrary(n);
            (xs, ws)
        }
    }
}

// ─── Gauss-Lobatto on [-1,1] ──────────────────────────────────────────────────

/// Gauss-Lobatto-Legendre points and weights on `[-1, 1]`, for `n` points (2 ≤ n ≤ 5).
///
/// Gauss-Lobatto rules **include the endpoints** ±1.  With `n` points they are
/// exact for polynomials up to degree `2n − 3`.  They are the standard choice
/// for spectral-element / nodal DG methods because the interpolation nodes
/// coincide with the quadrature points.
///
/// | n | interior pts | exactness |
/// |---|--------------|-----------|
/// | 2 | 0            | degree 1  |
/// | 3 | 1            | degree 3  |
/// | 4 | 2            | degree 5  |
/// | 5 | 3            | degree 7  |
fn gauss_lobatto_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        2 => (vec![-1.0, 1.0], vec![1.0, 1.0]),
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]),
        4 => {
            let x = (1.0_f64 / 5.0).sqrt();
            (
                vec![-1.0, -x, x, 1.0],
                vec![1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0],
            )
        }
        5 => {
            let x = (3.0_f64 / 7.0).sqrt();
            (
                vec![-1.0, -x, 0.0, x, 1.0],
                vec![
                    1.0 / 10.0,
                    49.0 / 90.0,
                    32.0 / 45.0,
                    49.0 / 90.0,
                    1.0 / 10.0,
                ],
            )
        }
        _ => panic!("gauss_lobatto_1d: only n=2..5 supported, got {n}"),
    }
}

/// Gauss-Lobatto rule on `[0, 1]` with `n` points (transform from `[-1,1]`).
///
/// Weights sum to 1.  Points include the endpoints 0 and 1.
pub fn gauss_lobatto_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (xs, ws) = gauss_lobatto_1d(n);
    let pts = xs.iter().map(|x| 0.5 * (x + 1.0)).collect();
    let wts = ws.iter().map(|w| 0.5 * w).collect();
    (pts, wts)
}

/// Gauss-Lobatto quadrature rule on the reference segment `[0, 1]`.
///
/// Uses `n` Gauss-Lobatto points (includes endpoints);
/// exact for polynomials up to degree `2n − 3`.
pub fn seg_lobatto_rule(order: u8) -> QuadratureRule {
    // n points integrates degree 2n-3 exactly; need 2n-3 >= order => n >= (order+3)/2
    let n = ((order as usize + 4) / 2).clamp(2, 5);
    let (pts, wts) = gauss_lobatto_01(n);
    QuadratureRule {
        points: pts.into_iter().map(|x| vec![x]).collect(),
        weights: wts,
    }
}

/// Tensor-product Gauss-Lobatto rule on the reference quad `[-1,1]²`.
///
/// Uses `n×n` Gauss-Lobatto points; exact for polynomials of degree ≤ `2n−3`
/// in each variable.  Points include all edges and corners of the reference quad.
pub fn quad_lobatto_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 4) / 2).clamp(2, 5);
    let (xs, ws) = gauss_lobatto_1d(n);
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// Tensor-product Gauss-Lobatto rule on the reference hex `[-1,1]³`.
///
/// Uses `n×n×n` Gauss-Lobatto points; exact for polynomials of degree ≤ `2n−3`
/// in each variable.
pub fn hex_lobatto_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 4) / 2).clamp(2, 5);
    let (xs, ws) = gauss_lobatto_1d(n);
    let mut pts = Vec::with_capacity(n * n * n);
    let mut wts = Vec::with_capacity(n * n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            for (xk, wk) in xs.iter().zip(ws.iter()) {
                pts.push(vec![*xi, *xj, *xk]);
                wts.push(wi * wj * wk);
            }
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

// ─── Arbitrary-order Gauss-Legendre on [-1,1] ────────────────────────────────

/// Compute Gauss-Legendre points and weights on `[-1, 1]` for arbitrary `n` points.
///
/// Uses Newton's method on the Legendre polynomial P_n(x) with O'Donnell's
/// initial guesses for the roots. Returns (points, weights) where weights sum to 2.
pub fn gauss_legendre_arbitrary(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        panic!("gauss_legendre_arbitrary: n must be >= 1");
    }
    if n <= 4 {
        return gauss_legendre_1d(n);
    }

    let n_f = n as f64;
    let mut pts = vec![0.0f64; n];
    let mut wts = vec![0.0f64; n];

    // Initial guesses using O'Donnell's formula
    for i in 0..n {
        let i_f = i as f64;
        pts[i] = (std::f64::consts::PI * (4.0 * i_f + 3.0) / (4.0 * n_f + 2.0)).cos();
    }

    // Newton iteration
    let tol = 1e-15;
    let max_iter = 100;
    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..n {
            let x = pts[i];
            // Evaluate P_n(x) and P_{n-1}(x) using recurrence
            let (pn, pn1) = legendre_poly(n, x);
            // Derivative: P'_n(x) = n/(1-x²) * (P_{n-1}(x) - x*P_n(x))
            let dpn = if (1.0 - x * x).abs() < 1e-30 {
                // Near endpoints, use l'Hopital or direct formula
                0.5 * n as f64 * (n as f64 + 1.0)
            } else {
                n as f64 / (1.0 - x * x) * (pn1 - x * pn)
            };
            let dx = -pn / dpn;
            pts[i] += dx;
            if dx.abs() > tol {
                converged = false;
            }
        }
        if converged {
            break;
        }
    }

    // Compute weights: w_i = 2 / (1 - x_i²) * [P'_n(x_i)]²
    for i in 0..n {
        let x = pts[i];
        let (pn, _pn1) = legendre_poly(n, x);
        // P'_n(x_i); kept for documentation / future use of the standard formula.
        let _dpn = if (1.0 - x * x).abs() < 1e-30 {
            0.5 * n as f64 * (n as f64 + 1.0)
        } else {
            n as f64 / (1.0 - x * x) * (legendre_poly(n - 1, x).0 - x * pn)
        };
        // For the weight formula, we need P_{n-1}(x) properly
        // Standard formula: w_i = 2 / ((1 - x_i²) * [P'_n(x_i)]²)
        // But P'_n(x) = n/(1-x²) * (P_{n-1}(x) - x*P_n(x))
        // Since P_n(x_i) = 0 at roots: P'_n(x_i) = n*P_{n-1}(x_i)/(1-x_i²)
        // So w_i = 2*(1-x_i²) / (n² * P_{n-1}(x_i)²)
        let pn1 = legendre_poly(n - 1, x).0;
        wts[i] = 2.0 * (1.0 - x * x) / (n_f * n_f * pn1 * pn1);
    }

    // Sort by point location for consistency
    let mut pairs: Vec<(f64, f64)> = pts.into_iter().zip(wts).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (pts_sorted, wts_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    (pts_sorted, wts_sorted)
}

/// Compute Gauss-Jacobi points and weights on `[-1, 1]` for arbitrary `n` points.
///
/// Gauss-Jacobi quadrature integrates functions with weight `(1-x)^α (1+x)^β`.
/// Special cases:
/// - α = β = 0: Gauss-Legendre
/// - α = β = -0.5: Gauss-Chebyshev
/// - α = β: Gauss-Gegenbauer
///
/// Uses the Golub-Welsch algorithm (eigenvalue decomposition of the Jacobi matrix).
/// Returns (points, weights) where weights sum to 2^(α+β+1) * B(α+1, β+1).
///
/// # Arguments
/// * `n` — number of quadrature points
/// * `alpha` — left exponent α
/// * `beta` — right exponent β
///
/// # Returns
/// (points, weights) where points are in increasing order
pub fn gauss_jacobi(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    // For α=β=0, use Gauss-Legendre (avoids removable singularity in Golub-Welsch)
    if (alpha.abs() < 1e-15) && (beta.abs() < 1e-15) {
        return if n <= 4 {
            gauss_legendre_1d(n)
        } else {
            gauss_legendre_arbitrary(n)
        };
    }
    if n == 1 {
        let x = (beta - alpha) / (alpha + beta + 2.0);
        let w = 2.0_f64.powf(alpha + beta + 1.0);
        return (vec![x], vec![w]);
    }

    // Golub-Welsch algorithm: build the symmetric tridiagonal Jacobi matrix
    // and compute its eigenvalues (nodes) and first components of eigenvectors (weights).
    let n_f = n as f64;
    let mut diag = vec![0.0f64; n];
    let mut offd = vec![0.0f64; n - 1];

    for i in 0..n {
        let i_f = i as f64;
        // Diagonal element
        let a = alpha;
        let b = beta;
        let num = b * b - a * a;
        let den = (2.0 * i_f + a + b) * (2.0 * i_f + a + b + 2.0);
        if den.abs() > 1e-30 {
            diag[i] = num / den;
        } else {
            diag[i] = 0.0;
        }
    }

    for i in 0..(n - 1) {
        let i_f = i as f64;
        let a = alpha;
        let b = beta;
        let num1 = 4.0 * (i_f + 1.0) * (i_f + a + 1.0) * (i_f + b + 1.0) * (i_f + a + b + 1.0);
        let den1 = (2.0 * i_f + a + b + 1.0).powi(2) * ((2.0 * i_f + a + b + 1.0).powi(2) - 1.0);
        if den1.abs() > 1e-30 {
            offd[i] = (num1 / den1).sqrt();
        } else {
            offd[i] = 0.0;
        }
    }

    // Solve the symmetric tridiagonal eigenvalue problem using QR iteration
    let (eigenvals, eigenvecs) = symmetric_tridiag_eigen(&diag, &offd);

    // Weights: w_i = μ_0 * (v_{i,0})² where μ_0 = ∫(1-x)^α(1+x)^β dx
    let mu0 = 2.0_f64.powf(alpha + beta + 1.0) * beta_fn(alpha + 1.0, beta + 1.0);

    let mut result: Vec<(f64, f64)> = eigenvals
        .iter()
        .zip(eigenvecs.iter())
        .map(|(&x, v)| (x, mu0 * v[0] * v[0]))
        .collect();

    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    result.into_iter().unzip()
}

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
fn beta_fn(a: f64, b: f64) -> f64 {
    (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)).exp()
}

/// Logarithm of the gamma function (Lanczos approximation)
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation with g=7, n=9
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let g = 7.0;
    if x < 0.5 {
        // Reflection formula
        let s = (std::f64::consts::PI * x).sin();
        if s.abs() < 1e-30 {
            return f64::INFINITY;
        }
        return (std::f64::consts::PI / s).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = p[0];
    let t = x + g + 0.5;
    for i in 1..p.len() {
        a += p[i] / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
}

/// Solve symmetric tridiagonal eigenvalue problem using QR iteration.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the i-th eigenvector.
fn symmetric_tridiag_eigen(diag: &[f64], offd: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = diag.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![diag[0]], vec![vec![1.0]]);
    }

    // Use implicit QR iteration for symmetric tridiagonal matrices
    // For simplicity, use the power method for each eigenvalue (not efficient but correct)
    let mut eigenvals = vec![0.0f64; n];
    let mut eigenvecs: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    // Use the QR algorithm with Wilkinson shift
    let mut d = diag.to_vec();
    let mut e = offd.to_vec();
    let mut z = identity_matrix(n);

    let max_iter = 100 * n;
    let eps = 1e-15;

    for _ in 0..max_iter {
        // Check for convergence
        let mut converged = true;
        for i in 0..(n - 1) {
            if e[i].abs() > eps * (d[i].abs() + d[i + 1].abs()) {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }

        // Find the largest unconverged subdiagonal
        let mut l = n - 1;
        for i in 0..(n - 1) {
            if e[i].abs() <= eps * (d[i].abs() + d[i + 1].abs()) {
                l = i;
                break;
            }
        }

        // Wilkinson shift
        let dl = d[l];
        let dl1 = d[l + 1];
        let el = e[l];
        let delta = (dl - dl1) / 2.0;
        let s = if delta.abs() < 1e-30 {
            dl1 - el.abs()
        } else {
            let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
            dl1 - el * el / (delta + sign_delta * (delta * delta + el * el).sqrt())
        };

        // QR step with shift
        let mut x = d[0] - s;
        let mut y = e[0];
        for i in 0..(n - 1) {
            // Givens rotation to zero out y
            let (c, s, r) = if x.abs() < 1e-30 {
                (0.0, 1.0, y)
            } else if x.abs() > y.abs() {
                let t = y / x;
                let r = (1.0 + t * t).sqrt();
                (1.0 / r, t / r, x * r)
            } else {
                let t = x / y;
                let r = (1.0 + t * t).sqrt();
                (t / r, 1.0 / r, y * r)
            };

            // Apply rotation
            let di = d[i];
            let di1 = d[i + 1];
            let ei = if i < e.len() { e[i] } else { 0.0 };
            let ei1 = if i + 1 < e.len() { e[i + 1] } else { 0.0 };

            d[i] = c * c * di + 2.0 * c * s * ei + s * s * di1;
            d[i + 1] = s * s * di - 2.0 * c * s * ei + c * c * di1;
            e[i] = c * s * (di1 - di) + (c * c - s * s) * ei;
            if i + 1 < e.len() {
                e[i + 1] = c * ei1;
            }
            if i > 0 {
                e[i - 1] = r;
            }

            // Update eigenvector matrix
            for k in 0..n {
                let zki = z[k][i];
                let zki1 = z[k][i + 1];
                z[k][i] = c * zki + s * zki1;
                z[k][i + 1] = -s * zki + c * zki1;
            }

            if i + 1 < n - 1 {
                x = e[i + 1];
                y = s * e[i + 1];
                e[i + 1] = c * e[i + 1];
            }
        }
    }

    for i in 0..n {
        eigenvals[i] = d[i];
    }
    for i in 0..n {
        for k in 0..n {
            eigenvecs[i][k] = z[k][i];
        }
    }

    (eigenvals, eigenvecs)
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Evaluate Legendre polynomial P_n(x) and P_{n-1}(x) using recurrence.
fn legendre_poly(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0); // P_0 = 1, no P_{-1}
    }
    if n == 1 {
        return (x, 1.0); // P_1 = x, P_0 = 1
    }
    let mut pn1 = 1.0f64; // P_0
    let mut pn = x; // P_1
    for k in 2..=n {
        let k_f = k as f64;
        let pn_next = ((2.0 * k_f - 1.0) * x * pn - (k_f - 1.0) * pn1) / k_f;
        pn1 = pn;
        pn = pn_next;
    }
    (pn, pn1)
}

/// Gauss-Legendre rule on `[0, 1]` with arbitrary `n` points.
///
/// Weights sum to 1.
pub fn gauss_legendre_01_arbitrary(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (xs, ws) = gauss_legendre_arbitrary(n);
    let pts = xs.iter().map(|x| 0.5 * (x + 1.0)).collect();
    let wts = ws.iter().map(|w| 0.5 * w).collect();
    (pts, wts)
}

/// Gauss-Lobatto-Legendre points and weights on `[-1, 1]` for arbitrary `n` points (n >= 2).
///
/// Uses the eigenvalue approach: the interior points are roots of P'_{n-1}(x).
/// Includes the endpoints ±1. Exact for polynomials up to degree 2n-3.
pub fn gauss_lobatto_arbitrary(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n < 2 {
        panic!("gauss_lobatto_arbitrary: n must be >= 2");
    }
    if n <= 5 {
        return gauss_lobatto_1d(n);
    }

    let n_f = n as f64;
    let n_int = n - 2; // number of interior points

    // Interior points are roots of P'_{n-1}(x)
    // We find them by Newton's method on d/dx[P_{n-1}(x)]
    let mut pts = vec![0.0f64; n];
    pts[0] = -1.0;
    pts[n - 1] = 1.0;
    let mut wts = vec![0.0f64; n];

    // Initial guesses for interior points: Chebyshev-like distribution
    for i in 0..n_int {
        let i_f = i as f64;
        pts[i + 1] = -(std::f64::consts::PI * (i_f + 1.0) / (n_f - 1.0)).cos();
    }

    // Newton iteration on P'_{n-1}(x)
    let tol = 1e-15;
    let max_iter = 100;
    let nm1 = n - 1;
    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..n_int {
            let x = pts[i + 1];
            // P_{n-1}(x) and its derivative
            let (pn, pn1) = legendre_poly(nm1, x);
            // Second derivative: P''_{n-1}(x) = n/(1-x²) * (x*P'_{n-1}(x) - P_{n-2}(x))
            // But we can use: P''_{n-1}(x) = (2x*P'_{n-1}(x) - n*(n-1)*P_{n-1}(x)) / (1-x²) ... hmm
            // Simpler: use the recurrence for derivatives
            // d/dx[P_{n-1}(x)] = (n-1)/(1-x²) * (P_{n-2}(x) - x*P_{n-1}(x))
            let dpn = if (1.0 - x * x).abs() < 1e-30 {
                0.5 * (nm1) as f64 * (nm1 as f64 + 1.0)
            } else {
                (nm1 as f64) / (1.0 - x * x) * (pn1 - x * pn)
            };
            // Newton: we want roots of P'_{n-1}, so we need the derivative of P'_{n-1}
            // d/dx[P'_{n-1}(x)] = P''_{n-1}(x)
            // Using recurrence: P''_n(x) = n/(1-x²) * (x*P'_n(x) - P'_{n-1}(x)) ... hmm
            // Better: use the fact that (1-x²)P''_n - 2xP'_n + n(n+1)P_n = 0
            // So P''_n = (2xP'_n - n(n+1)P_n) / (1-x²)
            let nm1_f = nm1 as f64;
            let ddpn = if (1.0 - x * x).abs() < 1e-30 {
                // At endpoints, limit is complex; skip (we don't iterate there)
                0.0
            } else {
                (2.0 * x * dpn - nm1_f * (nm1_f + 1.0) * pn) / (1.0 - x * x)
            };
            if ddpn.abs() < 1e-30 {
                continue;
            }
            let dx = -dpn / ddpn;
            pts[i + 1] += dx;
            if dx.abs() > tol {
                converged = false;
            }
        }
        if converged {
            break;
        }
    }

    // Weights for Lobatto: w_i = 2 / (n*(n-1) * [P_{n-1}(x_i)]²)
    for i in 0..n {
        let x = pts[i];
        let pn = legendre_poly(nm1, x).0;
        wts[i] = 2.0 / (n_f * (n_f - 1.0) * pn * pn);
    }

    // Sort by point location
    let mut pairs: Vec<(f64, f64)> = pts.into_iter().zip(wts).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (pts_sorted, wts_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    (pts_sorted, wts_sorted)
}

/// Gauss-Lobatto rule on `[0, 1]` with arbitrary `n` points (n >= 2).
///
/// Weights sum to 1. Points include the endpoints 0 and 1.
pub fn gauss_lobatto_01_arbitrary(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (xs, ws) = gauss_lobatto_arbitrary(n);
    let pts = xs.iter().map(|x| 0.5 * (x + 1.0)).collect();
    let wts = ws.iter().map(|w| 0.5 * w).collect();
    (pts, wts)
}

/// Quadrature rule on the reference segment `[0,1]` for arbitrary order.
///
/// Uses `n` Gauss-Legendre points; exact for polynomials up to degree `2n-1`.
/// Weights sum to 1 (length of the reference segment).
pub fn seg_rule_arbitrary(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (pts, wts) = gauss_legendre_01_arbitrary(n);
    QuadratureRule {
        points: pts.into_iter().map(|x| vec![x]).collect(),
        weights: wts,
    }
}

/// Gauss-Lobatto quadrature rule on the reference segment `[0, 1]` for arbitrary order.
pub fn seg_lobatto_rule_arbitrary(order: u8) -> QuadratureRule {
    // n points integrates degree 2n-3 exactly; need 2n-3 >= order => n >= (order+3)/2
    let n = ((order as usize + 4) / 2).max(2);
    let (pts, wts) = gauss_lobatto_01_arbitrary(n);
    QuadratureRule {
        points: pts.into_iter().map(|x| vec![x]).collect(),
        weights: wts,
    }
}

/// Tensor-product Gauss-Legendre rule on the reference quad `[-1,1]²` for arbitrary order.
///
/// Uses `n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 4 (area of reference quad).


/// Tensor-product Gauss-Legendre rule on the reference hex `[-1,1]³` for arbitrary order.
///
/// Uses `n×n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 8 (volume of reference hex).


/// Tensor-product Gauss-Lobatto rule on the reference quad `[-1,1]²` for arbitrary order.
pub fn quad_lobatto_rule_arbitrary(order: u8) -> QuadratureRule {
    let n = ((order as usize + 4) / 2).max(2);
    let (xs, ws) = gauss_lobatto_arbitrary(n);
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// Tensor-product Gauss-Lobatto rule on the reference hex `[-1,1]³` for arbitrary order.
pub fn hex_lobatto_rule_arbitrary(order: u8) -> QuadratureRule {
    let n = ((order as usize + 4) / 2).max(2);
    let (xs, ws) = gauss_lobatto_arbitrary(n);
    let mut pts = Vec::with_capacity(n * n * n);
    let mut wts = Vec::with_capacity(n * n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            for (xk, wk) in xs.iter().zip(ws.iter()) {
                pts.push(vec![*xi, *xj, *xk]);
                wts.push(wi * wj * wk);
            }
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

// ─── Segment [0,1] ────────────────────────────────────────────────────────────

/// Quadrature rule on the reference segment `[0,1]`.
///
/// Uses `n` Gauss-Legendre points; exact for polynomials up to degree `2n-1`.
/// Weights sum to 1 (length of the reference segment).
pub fn seg_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).clamp(1, 4);
    let (pts, wts) = gauss_legendre_01(n);
    QuadratureRule {
        points: pts.into_iter().map(|x| vec![x]).collect(),
        weights: wts,
    }
}

// ─── Triangle ─────────────────────────────────────────────────────────────────

/// Quadrature rule on the reference triangle `(0,0),(1,0),(0,1)`.
///
/// Uses Witherden-Vincent symmetric positive-weight rules (MFEM 4.10+)
/// for orders 0-20, falling back to Grundmann-Moller for higher orders.
///
/// Weights sum to 0.5 (area of reference triangle).
pub fn tri_rule(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid rule (exact for degree 1)
        QuadratureRule {
            points: vec![vec![1.0 / 3.0, 1.0 / 3.0]],
            weights: vec![0.5],
        }
    } else if order <= 2 {
        // 3-point rule (exact for degree 2)
        let a = 1.0 / 6.0;
        let b = 2.0 / 3.0;
        QuadratureRule {
            points: vec![vec![a, a], vec![b, a], vec![a, b]],
            weights: vec![a, a, a],
        }
    } else if order <= 20 {
        // Witherden-Vincent rules (MFEM 4.10, positive weights, interior points)
        wv_tri_rule(order)
    } else {
        // Fallback: Grundmann-Moller for very high orders
        let s = ((order as u32).saturating_sub(1)) / 2;
        grundmann_moller_simplex(2, s)
    }
}

/// Witherden-Vincent symmetric positive-weight rules for the reference triangle.
///
/// Source: MFEM 4.10 `IntegrationRules::TriangleIntegrationRule`, orders 3-20.
/// Reference: F.D. Witherden, P.E. Vincent, "On the identification of symmetric
/// quadrature rules for finite element methods", Computers & Mathematics with
/// Applications, 69(10):1232-1241, 2015.
///
/// All weights are positive and all points are interior (strictly inside the triangle).
fn wv_tri_rule(order: u8) -> QuadratureRule {
    let (centroid, s21, s111) = wv_tri_params(order);
    build_wv_tri(centroid, &s21, &s111)
}

/// Build a triangle quadrature rule from WV parameters.
///
/// - `centroid`: optional (weight,) for S3 (centroid at (1/3,1/3))
/// - `s21`: list of (a, weight) — each generates 3 points: (a,a),(1-2a,a),(a,1-2a)
/// - `s111`: list of (a, b, weight) — each generates 6 permutations of (a,b,1-a-b)
fn build_wv_tri(
    centroid: Option<f64>,
    s21: &[(f64, f64)],
    s111: &[(f64, f64, f64)],
) -> QuadratureRule {
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    // S3: centroid
    if let Some(w) = centroid {
        points.push(vec![1.0 / 3.0, 1.0 / 3.0]);
        weights.push(w);
    }

    // S21: 3 points per entry
    for &(a, w) in s21 {
        let b = 1.0 - 2.0 * a;
        points.push(vec![a, a]);
        points.push(vec![b, a]);
        points.push(vec![a, b]);
        weights.push(w);
        weights.push(w);
        weights.push(w);
    }

    // S111: 6 points per entry
    for &(a, b, w) in s111 {
        let c = 1.0 - a - b;
        points.push(vec![a, b]);
        points.push(vec![a, c]);
        points.push(vec![b, a]);
        points.push(vec![b, c]);
        points.push(vec![c, a]);
        points.push(vec![c, b]);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
    }

    QuadratureRule { points, weights }
}

/// Return WV rule parameters for a given triangle order (3-20).
///
/// Returns (optional centroid weight, S21 list, S111 list).
fn wv_tri_params(order: u8) -> (Option<f64>, Vec<(f64, f64)>, Vec<(f64, f64, f64)>) {
    match order {
        // Order 3-4: 6-point rule (2 × S21)
        3 | 4 => {
            let s21 = vec![
                (4.45948490915964890213e-01, 1.11690794839005735906e-01),
                (9.15762135097707430376e-02, 5.49758718276609353870e-02),
            ];
            (None, s21, vec![])
        }
        // Order 5: 7-point rule (S3 + 2 × S21)
        5 => {
            let centroid = Some(0.1125);
            let s21 = vec![
                (1.01286507323456342888e-01, 6.29695902724135697648e-02),
                (4.70142064105115109474e-01, 6.61970763942530959767e-02),
            ];
            (centroid, s21, vec![])
        }
        // Order 6: 12-point rule (2 × S21 + 1 × S111)
        6 => {
            let s21 = vec![
                (6.30890144915022266225e-02, 2.54224531851034094010e-02),
                (2.49286745170910428726e-01, 5.83931378631896841336e-02),
            ];
            let s111 = vec![(
                6.36502499121398668258e-01,
                3.10352451033784393353e-01,
                4.14255378091867854096e-02,
            )];
            (None, s21, s111)
        }
        // Order 7: 15-point rule (3 × S21 + 1 × S111)
        7 => {
            let s21 = vec![
                (3.37306485545878498300e-02, 8.27252505539606552976e-03),
                (2.41577382595403566956e-01, 6.39720856150777922311e-02),
                (4.74309692504718327655e-01, 3.85433230929930342734e-02),
            ];
            let s111 = vec![(
                7.54280040550053154647e-01,
                1.98683314797351684433e-01,
                2.79393664515998896292e-02,
            )];
            (None, s21, s111)
        }
        // Order 8: 16-point rule (S3 + 3 × S21 + 1 × S111)
        8 => {
            let centroid = Some(7.21578038388935860681e-02);
            let s21 = vec![
                (4.59292588292723236165e-01, 4.75458171336423096598e-02),
                (1.70569307751760268488e-01, 5.16086852673591223173e-02),
                (5.05472283170309566458e-02, 1.62292488115990396480e-02),
            ];
            let s111 = vec![(
                7.28492392955404244326e-01,
                2.63112829634638112353e-01,
                1.36151570872174963733e-02,
            )];
            (centroid, s21, s111)
        }
        // Order 9: 19-point rule (S3 + 4 × S21 + 1 × S111)
        9 => {
            let centroid = Some(4.85678981413994181882e-02);
            let s21 = vec![
                (4.37089591492936690997e-01, 3.89137705023871391385e-02),
                (1.88203535619032802373e-01, 3.98238694636051243636e-02),
                (4.89682519198737620236e-01, 1.56673501135695357467e-02),
                (4.47295133944527467662e-02, 1.27888378293490156262e-02),
            ];
            let s111 = vec![(
                7.41198598784498008385e-01,
                2.21962989160765733487e-01,
                2.16417696886446880855e-02,
            )];
            (centroid, s21, s111)
        }
        // Order 10: 25-point rule (S3 + 2 × S21 + 3 × S111)
        10 => {
            let centroid = Some(4.08716645731429864541e-02);
            let s21 = vec![
                (3.20553732169435168231e-02, 6.67648440657478327992e-03),
                (1.42161101056564431744e-01, 2.29789818023723654838e-02),
            ];
            let s111 = vec![
                (
                    5.30054118927343997925e-01,
                    3.21812995288835446139e-01,
                    3.19524531982120219009e-02,
                ),
                (
                    6.01233328683459244957e-01,
                    3.69146781827810910315e-01,
                    1.70923240814797143539e-02,
                ),
                (
                    8.07930600922879049719e-01,
                    1.63701733737182442141e-01,
                    1.26488788536441923438e-02,
                ),
            ];
            (centroid, s21, s111)
        }
        // Order 11: 28-point rule (S3 + 4 × S21 + 2 × S111)
        11 => {
            let centroid = Some(4.28805898661121093207e-02);
            let s21 = vec![
                (2.84854176143718995640e-02, 5.21593525644734826857e-03),
                (2.10219956703178278978e-01, 3.52578420558582877886e-02),
                (1.02635482712246428605e-01, 1.93153796185096607307e-02),
                (4.95891900965890919384e-01, 8.30313652729268436570e-03),
            ];
            let s21_extra = vec![(
                4.38465926764352253997e-01,
                3.36580770397341480504e-02,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    8.43349783661853091843e-01,
                    1.49324788652082374174e-01,
                    5.14514478647663895533e-03,
                ),
                (
                    6.64408374196864159877e-01,
                    2.89581125637705882880e-01,
                    2.01662383202502772106e-02,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 12: 33-point rule (3 × S21 + 3 × S111)
        12 => {
            let s21 = vec![
                (4.88203750945541581352e-01, 1.21334190407260157640e-02),
                (1.09257827659354322947e-01, 1.42430260344387719235e-02),
                (2.71462507014926135440e-01, 3.12706065979513822550e-02),
            ];
            let s21_extra = vec![
                (2.46463634363356387524e-02, 3.96582125498681943576e-03),
                (4.40111648658593201944e-01, 2.49591674640304711508e-02),
            ];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    6.85310163906391878186e-01,
                    2.91655679738340944951e-01,
                    1.08917925193037796322e-02,
                ),
                (
                    6.28249751683556123538e-01,
                    2.55454228638517299999e-01,
                    2.16136818297071042760e-02,
                ),
                (
                    8.51337792510240110033e-01,
                    1.27279717233589384495e-01,
                    7.54183878825571887144e-03,
                ),
            ];
            (None, s21_all, s111)
        }
        // Order 13: 37-point rule (S3 + 3 × S21 + 3 × S111)
        13 => {
            let centroid = Some(3.39800182934158201409e-02);
            let s21 = vec![
                (4.89076946452539351728e-01, 1.19972009644473652512e-02),
                (2.21372286291832920391e-01, 2.91392425595999905730e-02),
                (4.26941414259800422482e-01, 2.78009837652266646180e-02),
            ];
            let s21_extra = vec![(
                2.15096811088433259584e-02,
                3.02616855176958583773e-03,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    7.48507115899952224503e-01,
                    1.63597401067850478640e-01,
                    1.20895199057969096601e-02,
                ),
                (
                    8.64707770295442768038e-01,
                    1.10922042803463405392e-01,
                    7.48270055258283377231e-03,
                ),
                (
                    6.23545995553675513889e-01,
                    3.08441760892117777804e-01,
                    1.73206380704241866275e-02,
                ),
                (
                    7.22357793124188019007e-01,
                    2.72515817773429591675e-01,
                    4.79534050177163155559e-03,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 14: 42-point rule (4 × S21 + 3 × S111)
        14 => {
            let s21 = vec![
                (1.77205532412543442788e-01, 2.10812943684965080349e-02),
                (4.17644719340453940415e-01, 1.63941767720626740967e-02),
                (6.17998830908725871325e-02, 7.21684983488833382143e-03),
                (4.88963910362178677538e-01, 1.09417906847144447147e-02),
            ];
            let s21_extra = vec![(
                2.73477528308838646609e-01,
                2.58870522536457925433e-02,
            )];
            let s21_extra2 = vec![(
                1.93909612487010996063e-02,
                2.46170180120004094063e-03,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            s21_all.extend(s21_extra2);
            let s111 = vec![
                (
                    6.86980167808087793802e-01,
                    2.98372882136257788765e-01,
                    7.21815405676692022074e-03,
                ),
                (
                    7.70608554774996457049e-01,
                    1.72266687821355679588e-01,
                    1.23328766062818367955e-02,
                ),
                (
                    5.70222290846683188548e-01,
                    3.36861459796344964168e-01,
                    1.92857553935303419057e-02,
                ),
                (
                    8.79757171370171064950e-01,
                    1.18974497696956893478e-01,
                    2.50511441925033596229e-03,
                ),
            ];
            (None, s21_all, s111)
        }
        // Order 15: 49-point rule (S3 + 5 × S21 + 3 × S111)
        15 => {
            let centroid = Some(2.21676936910920364954e-02);
            let s21 = vec![
                (4.05362214133975495844e-01, 2.13568907857302828224e-02),
                (7.01735528999860580512e-02, 8.22236878131258133728e-03),
                (4.74170681438019769871e-01, 8.69807400038170690226e-03),
                (2.26378713420349653163e-01, 2.33916808643548149171e-02),
                (4.94996956769126195130e-01, 4.78692309123004283711e-03),
            ];
            let s21_extra = vec![(
                1.58117262509887002153e-02,
                1.48038731895268772104e-03,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    6.66975644801868106093e-01,
                    3.14648242812450851247e-01,
                    7.80128641528798211224e-03,
                ),
                (
                    9.19912157726236134891e-01,
                    7.09486052364554087291e-02,
                    2.01492668600904969653e-03,
                ),
                (
                    7.15222356931450642392e-01,
                    1.90535589476393929509e-01,
                    1.43602934626006709801e-02,
                ),
                (
                    8.13292641049419229304e-01,
                    1.68068645222414381202e-01,
                    5.83631059078792285844e-03,
                ),
                (
                    5.65252664877114230357e-01,
                    3.38950611475277163720e-01,
                    1.56577381424846430458e-02,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 16: 55-point rule (S3 + 5 × S21 + 4 × S111)
        16 => {
            let centroid = Some(2.26322830369093952463e-02);
            let s21 = vec![
                (2.45990070467141719313e-01, 2.05464615718494759966e-02),
                (4.15584896885420551627e-01, 2.03559166562126796218e-02),
                (8.53555665867003487968e-02, 7.39081734511220188322e-03),
                (1.61918644191271221544e-01, 1.47092048494940497855e-02),
                (5.00000000000000000000e-01, 2.20927315607528452004e-03),
            ];
            let s21_extra = vec![(
                4.75280727545942083268e-01,
                1.29871666491385793357e-02,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    7.54170061444767725334e-01,
                    1.91074763640529221576e-01,
                    9.46913623220784969603e-03,
                ),
                (
                    9.68244368030958701965e-01,
                    2.32034277688137335893e-02,
                    8.27233357417524097638e-04,
                ),
                (
                    6.49303698245446425652e-01,
                    3.31764523474147643434e-01,
                    7.50430089214290316213e-03,
                ),
                (
                    9.00273703270429548340e-01,
                    8.06961669858730079596e-02,
                    3.97379696669624901673e-03,
                ),
                (
                    5.89148840564247877616e-01,
                    3.08244969196354023921e-01,
                    1.59918050396850343342e-02,
                ),
                (
                    8.06621867499395683865e-01,
                    1.87441782483782071189e-01,
                    2.69559355842440570919e-03,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 17: 60-point rule (4 × S21 + 5 × S111)
        17 => {
            let s21 = vec![
                (4.17103444361599295931e-01, 1.36554632640510532210e-02),
                (1.47554916607539610141e-02, 1.38694378881882109979e-03),
                (4.65597871618890324363e-01, 1.25097254752486782697e-02),
                (1.80358116266370605008e-01, 1.31563152940089925225e-02),
            ];
            let s21_extra = vec![(
                6.66540634795969033632e-02,
                6.22950040115272107855e-03,
            )];
            let s21_extra2 = vec![(
                2.85706502436586629035e-01,
                1.88581185763976415248e-02,
            )];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            s21_all.extend(s21_extra2);
            let s111 = vec![
                (
                    8.24790070165088096132e-01,
                    1.59192287472792681768e-01,
                    3.98915010296479674579e-03,
                ),
                (
                    6.26369030386452196879e-01,
                    3.06281591746186521164e-01,
                    1.12438862733455335191e-02,
                ),
                (
                    5.71294867944684092720e-01,
                    4.15475459295228999324e-01,
                    5.19921997791976831654e-03,
                ),
                (
                    7.53235145936458128091e-01,
                    1.68722513495259462957e-01,
                    1.02789491602272593102e-02,
                ),
                (
                    7.15072259110642427515e-01,
                    2.71791870055354878311e-01,
                    4.34610725050059605590e-03,
                ),
                (
                    9.15919353297816929427e-01,
                    7.25054707990024915887e-02,
                    2.29217420086793351869e-03,
                ),
                (
                    5.43275579596159796658e-01,
                    2.99218942476970228839e-01,
                    1.30858129676684944304e-02,
                ),
            ];
            (None, s21_all, s111)
        }
        // Order 18: 67-point rule (S3 + 4 × S21 + 5 × S111)
        18 => {
            let centroid = Some(1.81778676507133342410e-02);
            let s21 = vec![
                (3.99955628067576229867e-01, 1.66522350166950668104e-02),
                (4.87580301574869645620e-01, 6.02332381699985548729e-03),
                (4.61809506406449243876e-01, 9.47458575338943308208e-03),
                (2.42264702514271956790e-01, 1.82375447044718190515e-02),
            ];
            let s21_extra = vec![
                (3.88302560886856218403e-02, 3.56466300985948522304e-03),
                (9.19477421216432500017e-02, 8.27957997600162372287e-03),
            ];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    7.70372376214675247397e-01,
                    1.83822707925463957324e-01,
                    6.87980811747110256732e-03,
                ),
                (
                    6.70953985194234547862e-01,
                    2.06349257433837918185e-01,
                    1.18909554500764153007e-02,
                ),
                (
                    6.00418954634256873959e-01,
                    3.95683434332269712286e-01,
                    2.26526725112853252742e-03,
                ),
                (
                    8.78342189467521738955e-01,
                    1.08195793791033278985e-01,
                    3.42005505980359086893e-03,
                ),
                (
                    6.39988092004714625993e-01,
                    3.19751624525377309283e-01,
                    8.87374455101020212511e-03,
                ),
                (
                    7.58929479855198430016e-01,
                    2.35772184958191743931e-01,
                    2.50533043728986106261e-03,
                ),
                (
                    9.72360728962795684005e-01,
                    2.70909109951620319379e-02,
                    6.11474063480544911126e-04,
                ),
                (
                    5.45918775386194599086e-01,
                    3.33493529449880754534e-01,
                    1.27410876559122202695e-02,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 19: 73-point rule (S3 + 5 × S21 + 5 × S111)
        19 => {
            let centroid = Some(1.72346988520061666916e-02);
            let s21 = vec![
                (5.25238903512089683190e-02, 3.55462829889906543543e-03),
                (4.92512675041336889237e-01, 5.16087757147214078873e-03),
                (1.11448873323021391268e-01, 7.61717554650914990128e-03),
                (4.59194201039543670184e-01, 1.14917950133708035576e-02),
                (4.03969722551901222474e-01, 1.57687674465774863020e-02),
            ];
            let s21_extra = vec![
                (1.78170104781764315760e-01, 1.23259574240954274116e-02),
                (1.16394611837894457196e-02, 8.82661388221423837477e-04),
                (2.55161632913607716588e-01, 1.58765096830015377261e-02),
            ];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    8.30156464400275351245e-01,
                    1.30697676268032414448e-01,
                    4.84774224342752330097e-03,
                ),
                (
                    5.59369805720300927732e-01,
                    3.11317629809541251973e-01,
                    1.31731609886953666272e-02,
                ),
                (
                    6.33313293128784149388e-01,
                    3.64617780974611060962e-01,
                    1.64103827591790965915e-03,
                ),
                (
                    7.04004819966042139079e-01,
                    2.21434885432331141075e-01,
                    9.05397246560622585843e-03,
                ),
                (
                    8.52566954376889230005e-01,
                    1.42425757365756355810e-01,
                    1.46315755173510018451e-03,
                ),
                (
                    6.05083979068707922266e-01,
                    3.54028009735275261960e-01,
                    8.05108138201205379703e-03,
                ),
                (
                    7.43181368957436361278e-01,
                    2.41894578960579587079e-01,
                    4.22794374976824798712e-03,
                ),
                (
                    9.30137698876805085746e-01,
                    6.00862753223067036501e-02,
                    1.66360068142969402642e-03,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Order 20: 79-point rule (S3 + 8 × S21 + 9 × S111)
        20 => {
            let centroid = Some(1.39101107014531159140e-02);
            let s21 = vec![
                (2.54579267673339160183e-01, 1.40832013075202471669e-02),
                (1.09761410283977789426e-02, 7.98840791066619858654e-04),
                (1.09383596711714603522e-01, 7.83023077607453328597e-03),
                (1.86294997744540946627e-01, 9.17346297425291473671e-03),
            ];
            let s21_extra = vec![
                (4.45551056955924895675e-01, 9.45239993323244813428e-03),
                (3.73108805988847103130e-02, 2.16127541066557732688e-03),
                (3.93425347817099924086e-01, 1.37880506290704585304e-02),
                (4.76245611540499047543e-01, 7.10182530340844071076e-03),
            ];
            let mut s21_all = s21;
            s21_all.extend(s21_extra);
            let s111 = vec![
                (
                    8.33295511838236246938e-01,
                    1.59133707657067247077e-01,
                    2.20289741855849742491e-03,
                ),
                (
                    7.54921502863547422280e-01,
                    1.98518132228788335425e-01,
                    5.98639857895469015836e-03,
                ),
                (
                    9.31054476783942153162e-01,
                    6.40905856084340586065e-02,
                    1.12986960212586558597e-03,
                ),
                (
                    6.11877703547425655373e-01,
                    3.33134817309587605294e-01,
                    8.66722556721933289070e-03,
                ),
                (
                    8.61684018936486717521e-01,
                    9.99522962881386756173e-02,
                    4.14571152761385782609e-03,
                ),
                (
                    6.78165737889635522606e-01,
                    2.15607057390094447591e-01,
                    7.72260782209923009323e-03,
                ),
                (
                    5.70144692890973359134e-01,
                    4.20023758816224113133e-01,
                    3.69568150025529782929e-03,
                ),
                (
                    5.42331804172428100230e-01,
                    3.17860123835772001577e-01,
                    1.16917457318277372147e-02,
                ),
                (
                    7.08681375720323636358e-01,
                    2.80581411423665327831e-01,
                    3.57820023845768515197e-03,
                ),
            ];
            (centroid, s21_all, s111)
        }
        // Fallback for orders > 20 (shouldn't normally be reached)
        _ => {
            let s21 = vec![(1.0 / 6.0, 1.0 / 6.0)];
            let s111 = vec![(0.4459484909159649, 0.09157621350977074, 0.11169079483900574)];
            (None, s21, s111)
        }
    }
}

// ─── Tetrahedron ──────────────────────────────────────────────────────────────

/// Stroud conical quadrature rule for triangles (order 2, 3 points).
///
/// All-positive weights, exact for polynomials up to degree 2.
/// Reference: A.H. Stroud, "Approximate Calculation of Multiple Integrals" (1971).
///
/// Weights sum to 0.5 (area of reference triangle).
pub fn stroud_tri_rule() -> QuadratureRule {
    let points: Vec<Vec<f64>> = vec![
        vec![1.0 / 2.0, 0.0],
        vec![0.0, 1.0 / 2.0],
        vec![1.0 / 2.0, 1.0 / 2.0],
    ];
    let weights: Vec<f64> = vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];

    QuadratureRule { points, weights }
}

/// Quadrature rule on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
///
/// Weights sum to 1/6 (volume of reference tet).
///
/// Supported polynomial degrees:
/// - order ≤ 1: 1-point centroid
/// - order ≤ 2: 4-point rule (exact degree 2)
/// - order ≤ 5: 10-point Grundmann-Moller rule, s=2 (exact degree 5)
/// - order > 5: 20-point Grundmann-Moller rule, s=3 (exact degree 7)
pub fn tet_rule(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid (exact degree 1)
        QuadratureRule {
            points: vec![vec![0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0],
        }
    } else if order <= 2 {
        // 4-point rule (exact for degree 2)
        let a = 0.138_196_601_125_010_5;
        let b = 0.585_410_196_624_968_5;
        QuadratureRule {
            points: vec![vec![a, a, a], vec![b, a, a], vec![a, b, a], vec![a, a, b]],
            weights: vec![1.0 / 24.0; 4],
        }
    } else if order <= 7 {
        // Witherden-Vincent rules (MFEM 4.10, positive weights, interior points)
        wv_tet_rule(order)
    } else {
        // Fallback: Grundmann-Moller for higher orders
        let s = (order as u32).div_ceil(2);
        grundmann_moller_simplex(3, s)
    }
}

/// Witherden-Vincent symmetric positive-weight rules for the reference tetrahedron.
///
/// Source: MFEM 4.10 `IntegrationRules::TetrahedronIntegrationRule`, orders 3-7.
/// Reference: F.D. Witherden, P.E. Vincent, "On the identification of symmetric
/// quadrature rules for finite element methods", Computers & Mathematics with
/// Applications, 69(10):1232-1241, 2015.
///
/// All weights are positive and all points are interior.
fn wv_tet_rule(order: u8) -> QuadratureRule {
    let (centroid, s31, s22, s211) = wv_tet_params(order);
    build_wv_tet(centroid, &s31, &s22, &s211)
}

/// Build a tetrahedron quadrature rule from WV parameters.
///
/// - `centroid`: optional weight for S4 (centroid at (0.25,0.25,0.25))
/// - `s31`: list of (a, weight) — each generates 4 points: (a,a,a),(b,a,a),(a,b,a),(a,a,b) where b=1-3a
/// - `s22`: list of (a, weight) — each generates 6 points: permutations of (a,a,b) where b=0.5-a
/// - `s211`: list of (a, bc, weight) — each generates 12 points: permutations of (a,a,bc,cb) where cb=1-2a-bc
fn build_wv_tet(
    centroid: Option<f64>,
    s31: &[(f64, f64)],
    s22: &[(f64, f64)],
    s211: &[(f64, f64, f64)],
) -> QuadratureRule {
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    // S4: centroid
    if let Some(w) = centroid {
        points.push(vec![0.25, 0.25, 0.25]);
        weights.push(w);
    }

    // S31: 4 points per entry — (a,a,a), (b,a,a), (a,b,a), (a,a,b) where b=1-3a
    for &(a, w) in s31 {
        let b = 1.0 - 3.0 * a;
        points.push(vec![a, a, a]);
        points.push(vec![b, a, a]);
        points.push(vec![a, b, a]);
        points.push(vec![a, a, b]);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
    }

    // S22: 6 points per entry — permutations of (a,a,b) and (b,b,a) where b=0.5-a
    for &(a, w) in s22 {
        let b = 0.5 - a;
        // From MFEM AddTetPoints6: 3 perms of (a,a,b) + 3 perms of (b,b,a)
        points.push(vec![a, a, b]);
        points.push(vec![a, b, a]);
        points.push(vec![b, a, a]);
        points.push(vec![b, b, a]);
        points.push(vec![b, a, b]);
        points.push(vec![a, b, b]);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
        weights.push(w);
    }

    // S211: 12 points per entry — permutations of (a,a,bc,cb) where cb=1-2a-bc
    for &(a, bc, w) in s211 {
        let cb = 1.0 - 2.0 * a - bc;
        // 12 permutations: (a,a,bc,cb) and permutations
        // The 12 permutations come from:
        // 3 choices for which coordinate gets 'a' (the other two get bc, cb)
        // × 2 ways to assign bc/cb to those two
        // × 2 from the two 'a' positions... actually let me just use the MFEM formula
        // AddTetPoints12(off, a, bc, w):
        //   AddTetPoints3(off,     a, bc, w):   (a,a,bc), (a,bc,a), (bc,a,a)
        //   AddTetPoints3(off + 3, a, cb, w):   (a,a,cb), (a,cb,a), (cb,a,a)
        //   AddTetPoints6(off + 6, a, bc, cb, w): all 6 perms of (a,bc,cb)
        // In Cartesian (x,y,z) where the 4th barycentric coord is 1-x-y-z:
        for p in [
            [a, a, bc], [a, bc, a], [bc, a, a],
            [a, a, cb], [a, cb, a], [cb, a, a],
            [a, bc, cb], [a, cb, bc], [bc, a, cb],
            [bc, cb, a], [cb, a, bc], [cb, bc, a],
        ] {
            points.push(vec![p[0], p[1], p[2]]);
            weights.push(w);
        }
    }

    QuadratureRule { points, weights }
}

/// Return WV rule parameters for a given tetrahedron order (3-7).
///
/// Returns (optional centroid weight, S31 list, S22 list, S211 list).
fn wv_tet_params(order: u8) -> (Option<f64>, Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64, f64)>) {
    match order {
        // Order 3: 8-point rule (2 × S31)
        3 => {
            let s31 = vec![
                (3.28163302516381705232e-01, 2.27029737561812265667e-02),
                (1.08047249898428621151e-01, 1.89636929104854412564e-02),
            ];
            (None, s31, vec![], vec![])
        }
        // Order 4-5: 14-point rule (2 × S31 + 1 × S22)
        4 | 5 => {
            let s31 = vec![
                (3.10885919263300669613e-01, 1.87813209530026427319e-02),
                (9.27352503108912484819e-02, 1.22488405193936587129e-02),
            ];
            let s22 = vec![(
                4.54496295874350364485e-01,
                7.09100346284691120807e-03,
            )];
            (None, s31, s22, vec![])
        }
        // Order 6: 24-point rule (3 × S31 + 1 × S211)
        6 => {
            let s31 = vec![
                (4.06739585346113652342e-02, 1.67953517588677390775e-03),
                (3.22337890142275540484e-01, 9.22619692394245453915e-03),
                (2.14602871259152117034e-01, 6.65379170969458179352e-03),
            ];
            let s211 = vec![(
                6.36610018750174977420e-02,
                6.03005664791649187428e-01,
                8.03571428571428492127e-03,
            )];
            (None, s31, vec![], s211)
        }
        // Order 7: 35-point rule (S4 + 1×S31 + 1×S22 + 2×S211)
        7 => {
            let centroid = Some(1.59142149106884754628e-02);
            let s31 = vec![(
                3.15701149778202794227e-01,
                7.05493020166117132397e-03,
            )];
            let s22 = vec![(
                4.49510177401603649994e-01,
                5.31615463880959638471e-03,
            )];
            let s211 = vec![
                (
                    1.88833831026001153219e-01,
                    5.75171637586999962011e-01,
                    6.20118845472243662709e-03,
                ),
                (
                    2.12654725414832546093e-02,
                    8.10830241098548620826e-01,
                    1.35179513831722359664e-03,
                ),
            ];
            (centroid, s31, s22, s211)
        }
        // Fallback (shouldn't normally be reached)
        _ => (None, vec![], vec![], vec![]),
    }
}

/// Grundmann-Moller quadrature rule on the reference tetrahedron.
/// Index `s` gives a rule exact for degree `2s+1`.
///
/// Points are the standard GM lattice on the unit tetrahedron.
/// Weights are solved exactly so that all monomials of degree ≤ 2s+1
/// are integrated exactly over T³ = {(x,y,z): x,y,z≥0, x+y+z≤1}.
/// Weights at each level i are equal across all points at that level
/// (by the symmetry of the rule) and are obtained by solving the
/// (s+1)×(s+1) Vandermonde-like system with exact simplex integrals.
fn grundmann_moller_tet(s: u32) -> QuadratureRule {
    let d: u32 = 3;

    // Generate point sets for each level i = 0..=s.
    let levels: Vec<Vec<[f64; 3]>> = (0..=s)
        .map(|i| {
            let si = s - i;
            let m = (2 * si + d + 1) as f64;
            simplex_points(si, d + 1)
                .iter()
                .map(|coords| {
                    let bary: Vec<f64> =
                        coords.iter().map(|&j| (2.0 * j as f64 + 1.0) / m).collect();
                    [bary[1], bary[2], bary[3]]
                })
                .collect()
        })
        .collect();

    // For each level i, all points share the same weight w_i.
    // Compute sum of x₁^{2k} over all points at level i.
    let n = (s + 1) as usize;
    let level_sums: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| {
                    levels[i]
                        .iter()
                        .map(|p| p[0].powi((2 * k) as i32))
                        .sum::<f64>()
                })
                .collect()
        })
        .collect();

    // Exact integrals of x^{2k} over T³: (2k)! / (2k+3)!
    let exact: Vec<f64> = (0..n)
        .map(|k| fact_f64((2 * k) as u32) / fact_f64((2 * k + 3) as u32))
        .collect();

    // Solve the (s+1)×(s+1) linear system for per-level weights.
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let mut row: Vec<f64> = (0..n).map(|i| level_sums[i][k]).collect();
            row.push(exact[k]);
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap())
            .unwrap();
        mat.swap(col, piv);
        let scale = mat[col][col];
        for j in col..=n {
            mat[col][j] /= scale;
        }
        for row in 0..n {
            if row != col {
                let f = mat[row][col];
                for j in col..=n {
                    mat[row][j] -= f * mat[col][j];
                }
            }
        }
    }
    let ws_per_level: Vec<f64> = (0..n).map(|i| mat[i][n]).collect();

    // Assemble the rule.
    let mut pts: Vec<Vec<f64>> = Vec::new();
    let mut wts: Vec<f64> = Vec::new();
    for (i, level) in levels.iter().enumerate() {
        for pt in level {
            pts.push(vec![pt[0], pt[1], pt[2]]);
            wts.push(ws_per_level[i]);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// All non-negative integer vectors (j_0,...,j_{n-1}) with sum = s.
fn simplex_points(s: u32, n: u32) -> Vec<Vec<u32>> {
    if n == 1 {
        return vec![vec![s]];
    }
    let mut result = Vec::new();
    for j0 in 0..=s {
        for rest in simplex_points(s - j0, n - 1) {
            let mut v = vec![j0];
            v.extend(rest);
            result.push(v);
        }
    }
    result
}

fn fact_f64(n: u32) -> f64 {
    (1..=n as u64).map(|x| x as f64).product::<f64>().max(1.0)
}

// ─── Quadrilateral [-1,1]² ────────────────────────────────────────────────────

/// Tensor-product Gauss-Legendre rule on the reference quad `[-1,1]²`.
///
/// Uses `n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 4 (area of reference quad).
pub fn quad_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = if n <= 4 {
        gauss_legendre_1d(n)
    } else {
        gauss_legendre_arbitrary(n)
    };
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// Tensor-product Gauss-Legendre rule on the reference quad `[0,1]²`.
///
/// Uses `n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 1 (area of `[0,1]²`).
pub fn quad_rule_01(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = if n <= 4 {
        gauss_legendre_01(n)
    } else {
        gauss_legendre_01_arbitrary(n)
    };
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    // MFEM's tensor-product order: the first index varies fastest
    // (GetQuadrature(SQUARE) lists (x1,y1),(x2,y1),(x1,y2),(x2,y2), …), so
    // the summation order of quadrature weights matches MFEM bit-for-bit.
    for (yj, wj) in xs.iter().zip(ws.iter()) {
        for (xi, wi) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *yj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// Tensor-product Gauss-Legendre rule on `[0,1]²` for arbitrary order.


// ─── Hexahedron [-1,1]³ ───────────────────────────────────────────────────────

/// Tensor-product Gauss-Legendre rule on the reference hex `[-1,1]³`.
///
/// Uses `n×n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 8 (volume of reference hex).
pub fn hex_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = if n <= 4 {
        gauss_legendre_1d(n)
    } else {
        gauss_legendre_arbitrary(n)
    };
    let mut pts = Vec::with_capacity(n * n * n);
    let mut wts = Vec::with_capacity(n * n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            for (xk, wk) in xs.iter().zip(ws.iter()) {
                pts.push(vec![*xi, *xj, *xk]);
                wts.push(wi * wj * wk);
            }
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

// ─── Arbitrary-order triangle quadrature ─────────────────────────────────────

/// Quadrature rule on the reference triangle `(0,0),(1,0),(0,1)` for arbitrary order.
///
/// Uses Grundmann-Moller rules which work for any polynomial degree.
/// Weights sum to 0.5 (area of reference triangle).
pub fn tri_rule_arbitrary(order: u8) -> QuadratureRule {
    // Delegate to tri_rule which uses WV rules for orders 0-20,
    // and Grundmann-Moller fallback for higher orders.
    tri_rule(order)
}

/// Generalized Grundmann-Moller quadrature rule for the unit simplex in `d` dimensions.
///
/// The unit simplex in `d` dimensions is defined by:
/// - 1D: [0,1]
/// - 2D: {(x,y) : x≥0, y≥0, x+y≤1} (reference triangle)
/// - 3D: {(x,y,z) : x≥0, y≥0, z≥0, x+y+z≤1} (reference tetrahedron)
///
/// Index `s` gives a rule exact for degree `2s+1`.
/// Weights sum to 1/d! (volume of the unit simplex).
fn grundmann_moller_simplex(d: u32, s: u32) -> QuadratureRule {
    // Generate point sets for each level i = 0..=s.
    let levels: Vec<Vec<Vec<f64>>> = (0..=s)
        .map(|i| {
            let si = s - i;
            let m = (2 * si + d + 1) as f64;
            simplex_points(si, d + 1)
                .iter()
                .map(|coords| {
                    let bary: Vec<f64> =
                        coords.iter().map(|&j| (2.0 * j as f64 + 1.0) / m).collect();
                    // Convert from barycentric to Cartesian: drop first barycentric coordinate
                    bary[1..].to_vec()
                })
                .collect()
        })
        .collect();

    // For each level i, all points share the same weight w_i.
    // Compute sum of x₁^{2k} over all points at level i.
    let n = (s + 1) as usize;
    let level_sums: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| {
                    levels[i]
                        .iter()
                        .map(|p| p[0].powi((2 * k) as i32))
                        .sum::<f64>()
                })
                .collect()
        })
        .collect();

    // Exact integrals of x^{2k} over the unit simplex in d dimensions:
    // (2k)! / (2k+d)!
    let exact: Vec<f64> = (0..n)
        .map(|k| {
            let k2 = (2 * k) as u32;
            fact_f64(k2) / fact_f64(k2 + d)
        })
        .collect();

    // Solve the (s+1)×(s+1) linear system for per-level weights.
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let mut row: Vec<f64> = (0..n).map(|i| level_sums[i][k]).collect();
            row.push(exact[k]);
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap())
            .unwrap();
        mat.swap(col, piv);
        let scale = mat[col][col];
        for j in col..=n {
            mat[col][j] /= scale;
        }
        for row in 0..n {
            if row != col {
                let f = mat[row][col];
                for j in col..=n {
                    mat[row][j] -= f * mat[col][j];
                }
            }
        }
    }
    let ws_per_level: Vec<f64> = (0..n).map(|i| mat[i][n]).collect();

    // Assemble the rule.
    let mut pts: Vec<Vec<f64>> = Vec::new();
    let mut wts: Vec<f64> = Vec::new();
    for (i, level) in levels.iter().enumerate() {
        for pt in level {
            pts.push(pt.clone());
            wts.push(ws_per_level[i]);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

/// Quadrature rule on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)` for arbitrary order.
///
/// Uses Grundmann-Moller rules which work for any polynomial degree.
/// Weights sum to 1/6 (volume of reference tet).


// ─── Named triangle quadrature rules ─────────────────────────────────────────

/// Named Dunavant / Gaussian triangle quadrature rules.
///
/// Provides a stable, enumerable catalogue of quadrature rules on the reference
/// triangle `(0,0),(1,0),(0,1)`.  Weights sum to 0.5 (area of reference triangle).
///
/// Use [`TriQuadRule::rule()`] to obtain the corresponding [`QuadratureRule`],
/// or [`tri_rule_named()`] as a convenience free function.
///
/// | Variant             | Points | Exact degree |
/// |---------------------|--------|--------------|
/// | `Centroid1Deg1`     | 1      | 1            |
/// | `Gaussian3Deg2`     | 3      | 2            |
/// | `Dunavant7Deg5`     | 7      | 5            |
/// | `Dunavant12Deg6`    | 12     | 6            |
/// | `Witherden15Deg7`   | 15     | 7            |
/// | `Dunavant19Deg9`    | 19     | 9            |
///
/// # Example
/// ```ignore
/// use fem_element::quadrature::{TriQuadRule, tri_rule_named};
///
/// // By enum variant:
/// let qr = TriQuadRule::Dunavant7Deg5.rule();
/// assert_eq!(qr.points.len(), 7);
///
/// // By minimum polynomial degree:
/// let qr = tri_rule_named(5);
/// assert_eq!(qr.points.len(), 7);
/// ```
/// Return the smallest-degree named triangle rule that is exact for `min_degree`.
///
/// This is the free-function companion to [`TriQuadRule::for_degree`].


/// 12-point Dunavant rule on the reference triangle, exact for degree 6.
///
/// Source: Dunavant (1985), via MFEM intrules.cpp (triangle, degree 6).
/// 12 points, all weights positive.  Weights sum to 0.5.
///
/// Structure: 2 × S21 (3 pts each) + 1 × S111 (6 pts) = 12 pts.
fn dunavant_tri_12() -> QuadratureRule {
    let (a1, w1) = (0.063_089_014_491_502_23_f64, 0.025_422_453_185_103_41_f64);
    let (a2, w2) = (0.249_286_745_170_910_42_f64, 0.058_393_137_863_189_685_f64);
    let (a3, b3, w3) = (
        0.053_145_049_844_816_947_f64,
        0.310_352_451_033_784_4_f64,
        0.041_425_537_809_186_787_f64,
    );

    macro_rules! s21 {
        ($a:expr) => {{
            let b = 1.0 - 2.0 * $a;
            vec![vec![$a, $a], vec![b, $a], vec![$a, b]]
        }};
    }
    macro_rules! s111 {
        ($a:expr, $b:expr) => {{
            let c = 1.0 - $a - $b;
            vec![
                vec![$a, $b],
                vec![$a, c],
                vec![$b, $a],
                vec![$b, c],
                vec![c, $a],
                vec![c, $b],
            ]
        }};
    }

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for p in s21!(a1) {
        points.push(p);
        weights.push(w1);
    }
    for p in s21!(a2) {
        points.push(p);
        weights.push(w2);
    }
    for p in s111!(a3, b3) {
        points.push(p);
        weights.push(w3);
    }

    QuadratureRule { points, weights }
}



// ─── Prism ─────────────────────────────────────────────────────────────────────

/// Tensor product of triangle × segment quadrature rule on the reference prism.
///
/// Reference prism: (ξ, η, ζ) where (η, ζ) ∈ unit triangle (0,0),(1,0),(0,1)
/// and ξ ∈ [0,1] (extrusion direction).
/// Volume = 0.5 (triangle area) × 1 (segment length) = 0.5.
///
/// Rule: `tri_rule(order)` × `seg_rule(order)` tensor product.
/// Weights sum to 0.5.
pub fn prism_rule(order: u8) -> QuadratureRule {
    let tri = tri_rule_arbitrary(order);
    let seg = seg_rule(order);
    let nt = tri.points.len();
    let ns = seg.points.len();
    let mut pts = Vec::with_capacity(nt * ns);
    let mut wts = Vec::with_capacity(nt * ns);
    for t in 0..nt {
        for s in 0..ns {
            pts.push(vec![seg.points[s][0], tri.points[t][0], tri.points[t][1]]);
            wts.push(tri.weights[t] * seg.weights[s]);
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

// ─── Pyramid ───────────────────────────────────────────────────────────────────

/// Quadrature rule on the reference pyramid using the Duffy transform.
///
/// Reference pyramid: vertices (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1).
/// Domain: x ∈ [0, 1-z], y ∈ [0, 1-z], z ∈ [0,1].
/// Volume = 1/3.
///
/// Uses tensor-product Gauss rule on the unit cube [0,1]³, mapped via
/// Duffy transform: (x,y,z) = (r(1-t), s(1-t), t) with Jacobian (1-t)².
/// Weights sum to 1/3.
pub fn pyramid_rule(order: u8) -> QuadratureRule {
    // Minimum n=2 to integrate the quadratic Jacobian (1-t)² correctly.
    let n = ((order as usize + 2) / 2).clamp(2, 4);
    let (xs, ws) = gauss_legendre_1d(n);
    let mut pts = Vec::with_capacity(n * n * n);
    let mut wts = Vec::with_capacity(n * n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        let r = 0.5 * (xi + 1.0);
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            let s = 0.5 * (xj + 1.0);
            for (xk, wk) in xs.iter().zip(ws.iter()) {
                let t = 0.5 * (xk + 1.0);
                let jac = (1.0 - t) * (1.0 - t);
                let x = r * (1.0 - t);
                let y = s * (1.0 - t);
                let z = t;
                pts.push(vec![x, y, z]);
                wts.push(wi * wj * wk * 0.125 * jac);
            }
        }
    }
    QuadratureRule {
        points: pts,
        weights: wts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight_sum(rule: &QuadratureRule) -> f64 {
        rule.weights.iter().sum()
    }

    #[test]
    fn seg_weights_sum_to_one() {
        for order in [1u8, 2, 3, 5, 7] {
            let r = seg_rule(order);
            assert!((weight_sum(&r) - 1.0).abs() < 1e-14, "order={order}");
        }
    }

    #[test]
    fn tri_weights_sum_to_half() {
        for order in [1u8, 2, 3, 4, 5] {
            let r = tri_rule(order);
            assert!((weight_sum(&r) - 0.5).abs() < 1e-14, "order={order}");
        }
        // 12-pt Dunavant rule (order >= 6): ~6e-12 error due to limited precision
        // of published weight coefficients — well within FEM accuracy requirements.
        for order in [6u8, 7] {
            let r = tri_rule(order);
            assert!((weight_sum(&r) - 0.5).abs() < 1e-10, "order={order}");
        }
    }

    #[test]
    fn tri_wv_weights_sum_to_half_all_orders() {
        // Witherden-Vincent rules (orders 1-20) must sum to 0.5 (triangle area)
        // and all weights must be strictly positive (interior-point property).
        for order in 1u8..=20 {
            let r = tri_rule(order);
            assert!(
                (weight_sum(&r) - 0.5).abs() < 1e-12,
                "order={order}, sum={}",
                weight_sum(&r)
            );
            // All weights positive (WV guarantee)
            for (i, &w) in r.weights.iter().enumerate() {
                assert!(w > 0.0, "order={order}, weight[{i}]={w} is not positive");
            }
        }
    }

    #[test]
    fn tri_wv_point_counts() {
        // Verify point counts match MFEM 4.10 WV rules.
        let expected: Vec<(u8, usize)> = vec![
            (1, 1), (2, 3), (3, 6), (4, 6), (5, 7), (6, 12), (7, 15),
            (8, 16), (9, 19), (10, 25), (11, 28), (12, 33), (13, 37),
            (14, 42), (15, 49), (16, 55), (17, 60), (18, 67), (19, 73),
            (20, 79),
        ];
        for (order, expected_n) in expected {
            let r = tri_rule(order);
            assert_eq!(
                r.points.len(), expected_n,
                "order={order}: expected {} points, got {}",
                expected_n, r.points.len()
            );
        }
    }

    #[test]
    fn tet_weights_sum_to_sixth() {
        for order in [1u8, 2, 3] {
            let r = tet_rule(order);
            assert!((weight_sum(&r) - 1.0 / 6.0).abs() < 1e-14, "order={order}");
        }
    }

    #[test]
    fn tet_wv_weights_sum_to_sixth_all_orders() {
        // Witherden-Vincent rules (orders 1-7) must sum to 1/6 (tet volume)
        // and all weights must be strictly positive.
        for order in 1u8..=7 {
            let r = tet_rule(order);
            assert!(
                (weight_sum(&r) - 1.0 / 6.0).abs() < 1e-12,
                "order={order}, sum={}",
                weight_sum(&r)
            );
            for (i, &w) in r.weights.iter().enumerate() {
                assert!(w > 0.0, "order={order}, weight[{i}]={w} is not positive");
            }
        }
    }

    #[test]
    fn tet_wv_point_counts() {
        // Verify point counts match MFEM 4.10 WV rules.
        let expected: Vec<(u8, usize)> = vec![
            (1, 1), (2, 4), (3, 8), (4, 14), (5, 14), (6, 24), (7, 35),
        ];
        for (order, expected_n) in expected {
            let r = tet_rule(order);
            assert_eq!(
                r.points.len(), expected_n,
                "order={order}: expected {} points, got {}",
                expected_n, r.points.len()
            );
        }
    }

    #[test]
    fn quad_weights_sum_to_four() {
        for order in [1u8, 2, 3, 5] {
            let r = quad_rule(order);
            assert!((weight_sum(&r) - 4.0).abs() < 1e-13, "order={order}");
        }
    }

    #[test]
    fn hex_weights_sum_to_eight() {
        for order in [1u8, 2, 3] {
            let r = hex_rule(order);
            assert!((weight_sum(&r) - 8.0).abs() < 1e-12, "order={order}");
        }
    }

    /// Integrate x² over [0,1] with 2-point GL rule — should be 1/3.
    #[test]
    fn seg_integrate_x_squared() {
        let r = seg_rule(3);
        let val: f64 = r
            .weights
            .iter()
            .zip(r.points.iter())
            .map(|(w, p)| w * p[0].powi(2))
            .sum();
        assert!((val - 1.0 / 3.0).abs() < 1e-14);
    }

    // ── Gauss-Lobatto tests ───────────────────────────────────────────────

    #[test]
    fn lobatto_1d_weights_sum_to_two() {
        // Gauss-Lobatto on [-1,1]: weights sum to 2
        for n in 2..=5 {
            let (_, ws) = super::gauss_lobatto_1d(n);
            let s: f64 = ws.iter().sum();
            assert!((s - 2.0).abs() < 1e-14, "n={n}, sum={s}");
        }
    }

    #[test]
    fn lobatto_01_weights_sum_to_one() {
        for n in 2..=5 {
            let (_, ws) = gauss_lobatto_01(n);
            let s: f64 = ws.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "n={n}");
        }
    }

    #[test]
    fn lobatto_01_includes_endpoints() {
        for n in 2..=5 {
            let (pts, _) = gauss_lobatto_01(n);
            assert!((pts[0]).abs() < 1e-14, "n={n}: first point should be 0");
            assert!(
                (pts[n - 1] - 1.0).abs() < 1e-14,
                "n={n}: last point should be 1"
            );
        }
    }

    #[test]
    fn seg_lobatto_integrate_x_squared() {
        // 3-point Lobatto on [0,1] is exact for degree 3 => x² should be exact.
        let r = seg_lobatto_rule(2);
        let val: f64 = r
            .weights
            .iter()
            .zip(r.points.iter())
            .map(|(w, p)| w * p[0].powi(2))
            .sum();
        assert!((val - 1.0 / 3.0).abs() < 1e-14, "got {val}");
    }

    #[test]
    fn quad_lobatto_weights_sum_to_four() {
        for order in [1u8, 3, 5] {
            let r = quad_lobatto_rule(order);
            assert!((weight_sum(&r) - 4.0).abs() < 1e-13, "order={order}");
        }
    }

    #[test]
    fn hex_lobatto_weights_sum_to_eight() {
        for order in [1u8, 3] {
            let r = hex_lobatto_rule(order);
            assert!((weight_sum(&r) - 8.0).abs() < 1e-12, "order={order}");
        }
    }

    #[test]
    fn prism_weights_sum_to_half() {
        for order in [1u8, 2, 3, 5] {
            let r = prism_rule(order);
            let s: f64 = r.weights.iter().sum();
            assert!((s - 0.5).abs() < 1e-12, "order={order}: sum={s}");
        }
    }

    #[test]
    fn prism_integrate_constant() {
        let r = prism_rule(3);
        let val: f64 = r.weights.iter().sum();
        assert!((val - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pyramid_weights_sum_to_third() {
        for order in [1u8, 2, 3, 5] {
            let r = pyramid_rule(order);
            let s: f64 = r.weights.iter().sum();
            assert!((s - 1.0 / 3.0).abs() < 1e-12, "order={order}: sum={s}");
        }
    }

    #[test]
    fn pyramid_integrate_constant() {
        let r = pyramid_rule(3);
        let val: f64 = r.weights.iter().sum();
        assert!((val - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn pyramid_integrate_z() {
        // ∫_pyramid z dV = ∫₀¹ z·(1-z)² dz = 1/12
        let r = pyramid_rule(5);
        let val: f64 = r
            .weights
            .iter()
            .zip(r.points.iter())
            .map(|(w, p)| w * p[2])
            .sum();
        assert!((val - 1.0 / 12.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn lobatto_exactness_degree() {
        // n=3 Lobatto on [-1,1] should integrate x³ exactly (degree 2n-3=3)
        let (xs, ws) = super::gauss_lobatto_1d(3);
        let val: f64 = xs.iter().zip(ws.iter()).map(|(x, w)| w * x.powi(3)).sum();
        // ∫_{-1}^{1} x³ dx = 0
        assert!(val.abs() < 1e-14, "integral of x³ = {val}");

        // n=4 Lobatto should integrate x⁵ exactly (degree 2*4-3=5)
        let (xs, ws) = super::gauss_lobatto_1d(4);
        let val: f64 = xs.iter().zip(ws.iter()).map(|(x, w)| w * x.powi(5)).sum();
        assert!(val.abs() < 1e-14, "integral of x⁵ = {val}");
    }
}

#[cfg(test)]
mod tet_quad_tests {
    use super::*;
    use crate::reference::ReferenceElement;
    #[test]
    fn tet_rule_weight_sums() {
        for order in [1u8, 2, 3, 5, 6, 7] {
            let rule = tet_rule(order);
            let wsum: f64 = rule.weights.iter().sum();
            assert!(
                (wsum - 1.0 / 6.0).abs() < 1e-12,
                "tet_rule(order={order}): weight sum = {wsum:.12} (expected {})",
                1.0 / 6.0
            );
        }
    }
    #[test]
    fn tet_rule_pou_p3() {
        // Verify sum of TetP3 basis functions = 1 at quadrature points
        use crate::lagrange::TetP3;
        let rule = tet_rule(7);
        let mut phi = vec![0.0f64; 20];
        for pt in &rule.points {
            TetP3.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "POU failed at {pt:?}: sum={s}");
        }
    }
}


#[cfg(test)]
mod gauss_jacobi_tests {
    use super::*;

    #[test]
    fn gauss_jacobi_legendre_equivalence() {
        // Gauss-Jacobi with α=β=0 should match Gauss-Legendre
        for n in [2, 3, 4, 5] {
            let (x_jac, w_jac) = gauss_jacobi(n, 0.0, 0.0);
            let (x_leg, w_leg) = if n <= 4 {
                gauss_legendre_1d(n)
            } else {
                gauss_legendre_arbitrary(n)
            };
            for i in 0..n {
                assert!(
                    (x_jac[i] - x_leg[i]).abs() < 1e-10,
                    "n={n}, i={i}: x_jac={} vs x_leg={}",
                    x_jac[i], x_leg[i]
                );
                assert!(
                    (w_jac[i] - w_leg[i]).abs() < 1e-10,
                    "n={n}, i={i}: w_jac={} vs w_leg={}",
                    w_jac[i], w_leg[i]
                );
            }
        }
    }

    #[test]
    fn gauss_jacobi_weight_sum() {
        // For α=β=0, weight sum should be 2.0
        let (_, w) = gauss_jacobi(5, 0.0, 0.0);
        let sum: f64 = w.iter().sum();
        assert!((sum - 2.0).abs() < 1e-10, "weight sum = {sum}, expected 2.0");
    }

    use super::stroud_tri_rule;

    #[test]
    fn stroud_tri_weight_sum() {
        let rule = stroud_tri_rule();
        let sum: f64 = rule.weights.iter().sum();
        assert!((sum - 0.5).abs() < 1e-14, "weight sum = {sum}, expected 0.5");
        assert_eq!(rule.points.len(), 3);
        assert_eq!(rule.weights.len(), 3);
    }

    #[test]
    fn stroud_tri_all_positive_weights() {
        let rule = stroud_tri_rule();
        for (i, &w) in rule.weights.iter().enumerate() {
            assert!(w > 0.0, "weight[{i}] = {w} is not positive");
        }
    }
}
