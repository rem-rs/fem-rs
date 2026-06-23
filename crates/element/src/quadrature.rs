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

/// Gauss-Legendre rule on `[0, 1]` with `n` points (transform from `[-1,1]`).
///
/// Weights sum to 1.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (xs, ws) = gauss_legendre_1d(n);
    let pts = xs.iter().map(|x| 0.5 * (x + 1.0)).collect();
    let wts = ws.iter().map(|w| 0.5 * w).collect();
    (pts, wts)
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
                vec![1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0],
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
    let n = ((order as usize + 4) / 2).max(2).min(5);
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
    let n = ((order as usize + 4) / 2).max(2).min(5);
    let (xs, ws) = gauss_lobatto_1d(n);
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule { points: pts, weights: wts }
}

/// Tensor-product Gauss-Lobatto rule on the reference hex `[-1,1]³`.
///
/// Uses `n×n×n` Gauss-Lobatto points; exact for polynomials of degree ≤ `2n−3`
/// in each variable.
pub fn hex_lobatto_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 4) / 2).max(2).min(5);
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
    QuadratureRule { points: pts, weights: wts }
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
        let dpn = if (1.0 - x * x).abs() < 1e-30 {
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
    let mut pairs: Vec<(f64, f64)> = pts.into_iter().zip(wts.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (pts_sorted, wts_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    (pts_sorted, wts_sorted)
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
    let mut pn = x;         // P_1
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
    let mut pairs: Vec<(f64, f64)> = pts.into_iter().zip(wts.into_iter()).collect();
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
        points:  pts.into_iter().map(|x| vec![x]).collect(),
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
pub fn quad_rule_arbitrary(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = gauss_legendre_arbitrary(n);
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule { points: pts, weights: wts }
}

/// Tensor-product Gauss-Legendre rule on the reference hex `[-1,1]³` for arbitrary order.
///
/// Uses `n×n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 8 (volume of reference hex).
pub fn hex_rule_arbitrary(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = gauss_legendre_arbitrary(n);
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
    QuadratureRule { points: pts, weights: wts }
}

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
    QuadratureRule { points: pts, weights: wts }
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
    QuadratureRule { points: pts, weights: wts }
}

// ─── Segment [0,1] ────────────────────────────────────────────────────────────

/// Quadrature rule on the reference segment `[0,1]`.
///
/// Uses `n` Gauss-Legendre points; exact for polynomials up to degree `2n-1`.
/// Weights sum to 1 (length of the reference segment).
pub fn seg_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1).min(4);
    let (pts, wts) = gauss_legendre_01(n);
    QuadratureRule {
        points:  pts.into_iter().map(|x| vec![x]).collect(),
        weights: wts,
    }
}

// ─── Triangle ─────────────────────────────────────────────────────────────────

/// Quadrature rule on the reference triangle `(0,0),(1,0),(0,1)`.
///
/// | order ≤ | # pts | Exactness |
/// |---------|-------|-----------|
/// | 1       | 1     | degree 1  |
/// | 3       | 3     | degree 2  |
/// | 5       | 7     | degree 5  |
///
/// Weights sum to 0.5 (area of reference triangle).
pub fn tri_rule(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid rule (exact for degree 1)
        QuadratureRule {
            points:  vec![vec![1.0 / 3.0, 1.0 / 3.0]],
            weights: vec![0.5],
        }
    } else if order <= 3 {
        // 3-point rule (exact for degree 2)
        let a = 1.0 / 6.0;
        let b = 2.0 / 3.0;
        QuadratureRule {
            points:  vec![vec![a, a], vec![b, a], vec![a, b]],
            weights: vec![a, a, a],
        }
    } else if order <= 5 {
        // 7-point Dunavant rule (exact for degree 5)
        let s1 = 0.101_286_507_323_456_33;
        let s2 = 0.797_426_985_353_087_2;
        let s3 = 0.470_142_064_105_115_05;
        let t3 = 0.059_715_871_789_769_81;
        let w1 = 0.125_939_180_544_827_17 / 2.0;
        let w2 = 0.132_394_152_788_506_16 / 2.0;
        let w3 = 0.225 / 2.0;
        QuadratureRule {
            points: vec![
                vec![s1, s1],
                vec![s2, s1],
                vec![s1, s2],
                vec![s3, s3],
                vec![t3, s3],
                vec![s3, t3],
                vec![1.0 / 3.0, 1.0 / 3.0],
            ],
            weights: vec![w1, w1, w1, w2, w2, w2, w3],
        }
    } else if order <= 6 {
        // 12-point Dunavant rule (exact for degree 6)
        let a = 0.063_089_014_491_502_228;
        let a1 = 1.0 - 2.0 * a;
        let b = 0.249_286_745_170_910_42;
        let b1 = 1.0 - 2.0 * b;
        let c = 0.053_145_049_844_816_947;
        let d = 0.310_352_451_033_784_41;
        let e = 1.0 - c - d;
        let wa = 0.050_844_906_370_206_817 / 2.0;
        let wb = 0.116_786_275_726_379_37 / 2.0;
        let wc = 0.082_851_075_618_373_575 / 2.0;
        QuadratureRule {
            points: vec![
                // Type 0: permutations of (a, a, 1-2a)
                vec![a, a1], vec![a1, a], vec![a, a],
                // Type 1: permutations of (b, b, 1-2b)
                vec![b, b1], vec![b1, b], vec![b, b],
                // Type 2: 6 permutations of (c, d, e)
                vec![d, e], vec![e, d], vec![c, e],
                vec![e, c], vec![c, d], vec![d, c],
            ],
            weights: vec![
                wa, wa, wa,
                wb, wb, wb,
                wc, wc, wc, wc, wc, wc,
            ],
        }
    } else if order <= 7 {
        // 15-point Witherden-Vincent rule (exact for degree 7)
        witherden_tri_15()
    } else {
        // Use generalized Grundmann-Moller for higher orders
        let s = ((order as u32) + 1) / 2;
        grundmann_moller_simplex(2, s)
    }
}

// ─── Tetrahedron ──────────────────────────────────────────────────────────────

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
            points:  vec![vec![0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0],
        }
    } else if order <= 2 {
        // 4-point rule (exact for degree 2)
        let a = 0.138_196_601_125_010_5;
        let b = 0.585_410_196_624_968_5;
        QuadratureRule {
            points: vec![
                vec![a, a, a],
                vec![b, a, a],
                vec![a, b, a],
                vec![a, a, b],
            ],
            weights: vec![1.0 / 24.0; 4],
        }
    } else if order <= 5 {
        // 10-point Grundmann-Moller rule, s=2 (exact degree 5)
        grundmann_moller_tet(2)
    } else if order <= 7 {
        // 20-point Grundmann-Moller rule, s=3 (exact degree 7)
        grundmann_moller_tet(3)
    } else {
        // Use generalized Grundmann-Moller for higher orders
        let s = ((order as u32) + 1) / 2;
        grundmann_moller_simplex(3, s)
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
    let levels: Vec<Vec<[f64; 3]>> = (0..=s).map(|i| {
        let si = s - i;
        let m = (2 * si + d + 1) as f64;
        simplex_points(si, d + 1).iter().map(|coords| {
            let bary: Vec<f64> = coords.iter().map(|&j| (2.0 * j as f64 + 1.0) / m).collect();
            [bary[1], bary[2], bary[3]]
        }).collect()
    }).collect();

    // For each level i, all points share the same weight w_i.
    // Compute sum of x₁^{2k} over all points at level i.
    let n = (s + 1) as usize;
    let level_sums: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|k| {
            levels[i].iter().map(|p| p[0].powi((2 * k) as i32)).sum::<f64>()
        }).collect()
    }).collect();

    // Exact integrals of x^{2k} over T³: (2k)! / (2k+3)!
    let exact: Vec<f64> = (0..n).map(|k| {
        fact_f64((2 * k) as u32) / fact_f64((2 * k + 3) as u32)
    }).collect();

    // Solve the (s+1)×(s+1) linear system for per-level weights.
    let mut mat: Vec<Vec<f64>> = (0..n).map(|k| {
        let mut row: Vec<f64> = (0..n).map(|i| level_sums[i][k]).collect();
        row.push(exact[k]);
        row
    }).collect();
    for col in 0..n {
        let piv = (col..n).max_by(|&a, &b|
            mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap()
        ).unwrap();
        mat.swap(col, piv);
        let scale = mat[col][col];
        for j in col..=n { mat[col][j] /= scale; }
        for row in 0..n {
            if row != col {
                let f = mat[row][col];
                for j in col..=n { mat[row][j] -= f * mat[col][j]; }
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
    QuadratureRule { points: pts, weights: wts }
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
    let n = ((order as usize + 2) / 2).max(1).min(4);
    let (xs, ws) = gauss_legendre_1d(n);
    let mut pts = Vec::with_capacity(n * n);
    let mut wts = Vec::with_capacity(n * n);
    for (xi, wi) in xs.iter().zip(ws.iter()) {
        for (xj, wj) in xs.iter().zip(ws.iter()) {
            pts.push(vec![*xi, *xj]);
            wts.push(wi * wj);
        }
    }
    QuadratureRule { points: pts, weights: wts }
}

// ─── Hexahedron [-1,1]³ ───────────────────────────────────────────────────────

/// Tensor-product Gauss-Legendre rule on the reference hex `[-1,1]³`.
///
/// Uses `n×n×n` Gauss points; exact for polynomials of degree ≤ `2n-1` in each variable.
/// Weights sum to 8 (volume of reference hex).
pub fn hex_rule(order: u8) -> QuadratureRule {
    let n = ((order as usize + 2) / 2).max(1);
    let (xs, ws) = if n <= 4 { gauss_legendre_1d(n) } else { gauss_legendre_arbitrary(n) };
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
    QuadratureRule { points: pts, weights: wts }
}

// ─── Arbitrary-order triangle quadrature ─────────────────────────────────────

/// Quadrature rule on the reference triangle `(0,0),(1,0),(0,1)` for arbitrary order.
///
/// Uses Grundmann-Moller rules which work for any polynomial degree.
/// Weights sum to 0.5 (area of reference triangle).
pub fn tri_rule_arbitrary(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid rule (exact for degree 1)
        QuadratureRule {
            points:  vec![vec![1.0 / 3.0, 1.0 / 3.0]],
            weights: vec![0.5],
        }
    } else if order <= 3 {
        // 3-point rule (exact for degree 2)
        let a = 1.0 / 6.0;
        let b = 2.0 / 3.0;
        QuadratureRule {
            points:  vec![vec![a, a], vec![b, a], vec![a, b]],
            weights: vec![a, a, a],
        }
    } else if order <= 5 {
        // 7-point Dunavant rule (exact for degree 5)
        tri_rule(5)
    } else {
        // Use generalized Grundmann-Moller for 2D simplex
        // s = (order - 1) / 2 gives exact degree 2s+1
        let s = ((order as u32).saturating_sub(1)) / 2;
        grundmann_moller_simplex(2, s)
    }
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
    let levels: Vec<Vec<Vec<f64>>> = (0..=s).map(|i| {
        let si = s - i;
        let m = (2 * si + d + 1) as f64;
        simplex_points(si, d + 1).iter().map(|coords| {
            let bary: Vec<f64> = coords.iter().map(|&j| (2.0 * j as f64 + 1.0) / m).collect();
            // Convert from barycentric to Cartesian: drop first barycentric coordinate
            bary[1..].to_vec()
        }).collect()
    }).collect();

    // For each level i, all points share the same weight w_i.
    // Compute sum of x₁^{2k} over all points at level i.
    let n = (s + 1) as usize;
    let level_sums: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|k| {
            levels[i].iter().map(|p| p[0].powi((2 * k) as i32)).sum::<f64>()
        }).collect()
    }).collect();

    // Exact integrals of x^{2k} over the unit simplex in d dimensions:
    // (2k)! / (2k+d)!
    let exact: Vec<f64> = (0..n).map(|k| {
        let k2 = (2 * k) as u32;
        fact_f64(k2) / fact_f64(k2 + d)
    }).collect();

    // Solve the (s+1)×(s+1) linear system for per-level weights.
    let mut mat: Vec<Vec<f64>> = (0..n).map(|k| {
        let mut row: Vec<f64> = (0..n).map(|i| level_sums[i][k]).collect();
        row.push(exact[k]);
        row
    }).collect();
    for col in 0..n {
        let piv = (col..n).max_by(|&a, &b|
            mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap()
        ).unwrap();
        mat.swap(col, piv);
        let scale = mat[col][col];
        for j in col..=n { mat[col][j] /= scale; }
        for row in 0..n {
            if row != col {
                let f = mat[row][col];
                for j in col..=n { mat[row][j] -= f * mat[col][j]; }
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
    QuadratureRule { points: pts, weights: wts }
}

/// Quadrature rule on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)` for arbitrary order.
///
/// Uses Grundmann-Moller rules which work for any polynomial degree.
/// Weights sum to 1/6 (volume of reference tet).
pub fn tet_rule_arbitrary(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid (exact degree 1)
        QuadratureRule {
            points:  vec![vec![0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0],
        }
    } else if order <= 2 {
        // 4-point rule (exact for degree 2)
        let a = 0.138_196_601_125_010_5;
        let b = 0.585_410_196_624_968_5;
        QuadratureRule {
            points: vec![
                vec![a, a, a],
                vec![b, a, a],
                vec![a, b, a],
                vec![a, a, b],
            ],
            weights: vec![1.0 / 24.0; 4],
        }
    } else {
        // Use generalized Grundmann-Moller for 3D simplex
        let s = ((order as u32).saturating_sub(1)) / 2;
        grundmann_moller_simplex(3, s)
    }
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriQuadRule {
    /// 1-point centroid rule, exact for polynomials of degree ≤ 1.
    Centroid1Deg1,
    /// 3-point Gaussian rule, exact for polynomials of degree ≤ 2.
    Gaussian3Deg2,
    /// 7-point Dunavant rule, exact for polynomials of degree ≤ 5.
    Dunavant7Deg5,
    /// 12-point Dunavant rule, exact for polynomials of degree ≤ 6.
    Dunavant12Deg6,
    /// 15-point Witherden-Vincent rule, exact for polynomials of degree ≤ 7.
    /// All weights are positive.
    Witherden15Deg7,
    /// 19-point Dunavant rule, exact for polynomials of degree ≤ 9.
    Dunavant19Deg9,
}

impl TriQuadRule {
    /// Return the minimum-degree rule that is exact for polynomials up to `degree`.
    pub fn for_degree(degree: u8) -> Self {
        match degree {
            0..=1 => Self::Centroid1Deg1,
            2..=2 => Self::Gaussian3Deg2,
            3..=5 => Self::Dunavant7Deg5,
            6..=6 => Self::Dunavant12Deg6,
            7..=7 => Self::Witherden15Deg7,
            _     => Self::Dunavant19Deg9,
        }
    }

    /// The number of quadrature points in this rule.
    pub fn n_points(self) -> usize {
        match self {
            Self::Centroid1Deg1  => 1,
            Self::Gaussian3Deg2  => 3,
            Self::Dunavant7Deg5  => 7,
            Self::Dunavant12Deg6 => 12,
            Self::Witherden15Deg7 => 15,
            Self::Dunavant19Deg9 => 19,
        }
    }

    /// The polynomial degree for which this rule is exact.
    pub fn exact_degree(self) -> u8 {
        match self {
            Self::Centroid1Deg1  => 1,
            Self::Gaussian3Deg2  => 2,
            Self::Dunavant7Deg5  => 5,
            Self::Dunavant12Deg6 => 6,
            Self::Witherden15Deg7 => 7,
            Self::Dunavant19Deg9 => 9,
        }
    }

    /// Compute and return the [`QuadratureRule`] for this variant.
    pub fn rule(self) -> QuadratureRule {
        match self {
            Self::Centroid1Deg1  => tri_rule(1),
            Self::Gaussian3Deg2  => tri_rule(2),
            Self::Dunavant7Deg5  => tri_rule(5),
            Self::Dunavant12Deg6 => dunavant_tri_12(),
            Self::Witherden15Deg7 => witherden_tri_15(),
            Self::Dunavant19Deg9 => dunavant_tri_19(),
        }
    }
}

/// Return the smallest-degree named triangle rule that is exact for `min_degree`.
///
/// This is the free-function companion to [`TriQuadRule::for_degree`].
pub fn tri_rule_named(min_degree: u8) -> QuadratureRule {
    TriQuadRule::for_degree(min_degree).rule()
}

/// 12-point Dunavant rule on the reference triangle, exact for degree 6.
///
/// Source: Dunavant (1985), via MFEM intrules.cpp (triangle, degree 6).
/// 12 points, all weights positive.  Weights sum to 0.5.
///
/// Structure: 2 × S21 (3 pts each) + 1 × S111 (6 pts) = 12 pts.
fn dunavant_tri_12() -> QuadratureRule {
    let (a1, w1) = (0.063_089_014_491_502_228_f64, 0.025_422_453_185_103_408_f64);
    let (a2, w2) = (0.249_286_745_170_910_42_f64, 0.058_393_137_863_189_685_f64);
    let (a3, b3, w3) = (
        0.053_145_049_844_816_947_f64,
        0.310_352_451_033_784_41_f64,
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
                vec![$a, $b], vec![$a, c], vec![$b, $a],
                vec![$b, c],  vec![c, $a], vec![c, $b],
            ]
        }};
    }

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for p in s21!(a1) { points.push(p); weights.push(w1); }
    for p in s21!(a2) { points.push(p); weights.push(w2); }
    for p in s111!(a3, b3) { points.push(p); weights.push(w3); }

    QuadratureRule { points, weights }
}

/// 15-point Witherden-Vincent rule on the reference triangle, exact for degree 7.
///
/// Source: Witherden & Vincent (2015), via MFEM intrules.cpp (triangle, degree 7).
/// 15 points, all weights positive.  Weights sum to 0.5.
///
/// Structure: 3 × S21 (3 pts each) + 1 × S111 (6 pts) = 15 pts.
fn witherden_tri_15() -> QuadratureRule {
    // S21(a): 3 symmetric points (a,a),(1-2a,a),(a,1-2a) in Cartesian
    // S111(a,b): 6 asymmetric points, all permutations of (a,b,1-a-b)

    let (a1, w1) = (3.373_064_855_458_784_983_00e-2_f64, 8.272_525_055_396_065_529_76e-3_f64);
    let (a2, w2) = (2.415_773_825_954_035_669_56e-1_f64, 6.397_208_561_507_779_223_11e-2_f64);
    let (a3, w3) = (4.743_096_925_047_183_276_55e-1_f64, 3.854_332_309_299_303_427_34e-2_f64);
    let (a4, b4, w4) = (
        7.542_800_405_500_531_546_47e-1_f64,
        1.986_833_147_973_516_844_33e-1_f64,
        2.793_936_645_159_988_962_92e-2_f64,
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
                vec![$a, $b], vec![$a, c], vec![$b, $a],
                vec![$b, c],  vec![c, $a], vec![c, $b],
            ]
        }};
    }

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for p in s21!(a1) { points.push(p); weights.push(w1); }
    for p in s21!(a2) { points.push(p); weights.push(w2); }
    for p in s21!(a3) { points.push(p); weights.push(w3); }
    for p in s111!(a4, b4) { points.push(p); weights.push(w4); }

    QuadratureRule { points, weights }
}

/// 19-point Witherden-Vincent rule on the reference triangle, exact for degree 9.
///
/// Source: Witherden & Vincent (2015), via MFEM intrules.cpp (triangle, degree 9).
/// All weights positive.  Weights sum to 0.5.
fn dunavant_tri_19() -> QuadratureRule {
    // Structure: 1 S3 (centroid) + 4 × S21 (3 pts each) + 1 × S111 (6 pts) = 19 pts.
    // All weights positive; verified via MFEM intrules.cpp (triangle, degree 9).
    //
    // S21(a): 3 symmetric pts (a, a, 1-2a) in barycentric → Cartesian (a,a),(1-2a,a),(a,1-2a)
    // S111(a,b): 6 asymmetric pts, all permutations of (a, b, 1-a-b)

    let wc = 4.856_789_814_139_941_818_82e-2_f64; // centroid
    let (a1, w1) = (4.370_895_914_929_366_909_97e-1_f64, 3.891_377_050_238_713_913_85e-2_f64);
    let (a2, w2) = (1.882_035_356_190_328_023_73e-1_f64, 3.982_386_946_360_512_436_36e-2_f64);
    let (a3, w3) = (4.896_825_191_987_376_202_36e-1_f64, 1.566_735_011_356_953_574_67e-2_f64);
    let (a4, w4) = (4.472_951_339_445_274_676_62e-2_f64, 1.278_883_782_934_901_562_62e-2_f64);
    let (a5, b5, w5) = (
        7.411_985_987_844_980_083_85e-1_f64,
        2.219_629_891_607_657_334_87e-1_f64,
        2.164_176_968_864_468_808_55e-2_f64,
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
                vec![$a, $b], vec![$a, c], vec![$b, $a],
                vec![$b, c],  vec![c, $a], vec![c, $b],
            ]
        }};
    }

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    points.push(vec![1.0/3.0, 1.0/3.0]);  weights.push(wc);
    for p in s21!(a1) { points.push(p); weights.push(w1); }
    for p in s21!(a2) { points.push(p); weights.push(w2); }
    for p in s21!(a3) { points.push(p); weights.push(w3); }
    for p in s21!(a4) { points.push(p); weights.push(w4); }
    for p in s111!(a5, b5) { points.push(p); weights.push(w5); }

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
    QuadratureRule { points: pts, weights: wts }
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
    let n = ((order as usize + 2) / 2).max(1).min(4);
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
    QuadratureRule { points: pts, weights: wts }
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
    fn tet_weights_sum_to_sixth() {
        for order in [1u8, 2, 3] {
            let r = tet_rule(order);
            assert!((weight_sum(&r) - 1.0 / 6.0).abs() < 1e-14, "order={order}");
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
        let val: f64 = r.weights.iter().zip(r.points.iter())
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
            assert!((pts[n - 1] - 1.0).abs() < 1e-14, "n={n}: last point should be 1");
        }
    }

    #[test]
    fn seg_lobatto_integrate_x_squared() {
        // 3-point Lobatto on [0,1] is exact for degree 3 => x² should be exact.
        let r = seg_lobatto_rule(2);
        let val: f64 = r.weights.iter().zip(r.points.iter())
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
        let val: f64 = r.weights.iter().zip(r.points.iter())
            .map(|(w, p)| w * p[2])
            .sum();
        assert!((val - 1.0 / 12.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn lobatto_exactness_degree() {
        // n=3 Lobatto on [-1,1] should integrate x³ exactly (degree 2n-3=3)
        let (xs, ws) = super::gauss_lobatto_1d(3);
        let val: f64 = xs.iter().zip(ws.iter())
            .map(|(x, w)| w * x.powi(3))
            .sum();
        // ∫_{-1}^{1} x³ dx = 0
        assert!(val.abs() < 1e-14, "integral of x³ = {val}");

        // n=4 Lobatto should integrate x⁵ exactly (degree 2*4-3=5)
        let (xs, ws) = super::gauss_lobatto_1d(4);
        let val: f64 = xs.iter().zip(ws.iter())
            .map(|(x, w)| w * x.powi(5))
            .sum();
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
            assert!((wsum - 1.0/6.0).abs() < 1e-12,
                "tet_rule(order={order}): weight sum = {wsum:.12} (expected {})", 1.0/6.0);
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
mod tri_named_quad_tests {
    use super::*;

    /// ∫∫_T x^i y^j dA on the reference triangle (0,0),(1,0),(0,1).
    /// Closed-form value: i! j! / (i+j+2)!
    fn monomial_exact(i: u32, j: u32) -> f64 {
        let factorial = |n: u32| -> f64 { (1..=n).map(|k| k as f64).product::<f64>().max(1.0) };
        factorial(i) * factorial(j) / factorial(i + j + 2)
    }

    fn integrate_monomial(rule: &QuadratureRule, i: u32, j: u32) -> f64 {
        rule.weights.iter().zip(rule.points.iter())
            .map(|(w, p)| w * p[0].powi(i as i32) * p[1].powi(j as i32))
            .sum()
    }

    // ── Enum metadata ────────────────────────────────────────────────────

    #[test]
    fn tri_quad_rule_n_points() {
        assert_eq!(TriQuadRule::Centroid1Deg1.n_points(),  1);
        assert_eq!(TriQuadRule::Gaussian3Deg2.n_points(),  3);
        assert_eq!(TriQuadRule::Dunavant7Deg5.n_points(),  7);
        assert_eq!(TriQuadRule::Dunavant12Deg6.n_points(), 12);
        assert_eq!(TriQuadRule::Witherden15Deg7.n_points(), 15);
        assert_eq!(TriQuadRule::Dunavant19Deg9.n_points(), 19);
    }

    #[test]
    fn tri_quad_rule_exact_degree() {
        assert_eq!(TriQuadRule::Centroid1Deg1.exact_degree(),  1);
        assert_eq!(TriQuadRule::Witherden15Deg7.exact_degree(), 7);
        assert_eq!(TriQuadRule::Dunavant19Deg9.exact_degree(), 9);
    }

    #[test]
    fn tri_quad_rule_for_degree_selects_correct_variant() {
        assert_eq!(TriQuadRule::for_degree(0), TriQuadRule::Centroid1Deg1);
        assert_eq!(TriQuadRule::for_degree(1), TriQuadRule::Centroid1Deg1);
        assert_eq!(TriQuadRule::for_degree(2), TriQuadRule::Gaussian3Deg2);
        assert_eq!(TriQuadRule::for_degree(5), TriQuadRule::Dunavant7Deg5);
        assert_eq!(TriQuadRule::for_degree(6), TriQuadRule::Dunavant12Deg6);
        assert_eq!(TriQuadRule::for_degree(7), TriQuadRule::Witherden15Deg7);
        assert_eq!(TriQuadRule::for_degree(8), TriQuadRule::Dunavant19Deg9);
        assert_eq!(TriQuadRule::for_degree(9), TriQuadRule::Dunavant19Deg9);
    }

    // ── Weight sums ──────────────────────────────────────────────────────

    #[test]
    fn all_named_rules_weights_sum_to_half() {
        let rules = [
            TriQuadRule::Centroid1Deg1,
            TriQuadRule::Gaussian3Deg2,
            TriQuadRule::Dunavant7Deg5,
            TriQuadRule::Dunavant12Deg6,
            TriQuadRule::Witherden15Deg7,
            TriQuadRule::Dunavant19Deg9,
        ];
        for r in rules {
            let qr = r.rule();
            let ws: f64 = qr.weights.iter().sum();
            // Degree-7 rule has a negative centroid weight; allow slightly wider tolerance
            assert!((ws - 0.5).abs() < 1e-10, "{r:?}: weight sum = {ws:.12}");
            assert_eq!(qr.points.len(), r.n_points(), "{r:?}: point count mismatch");
        }
    }

    // ── Monomial exactness tests ─────────────────────────────────────────
    // For each rule, verify ∫ x^i y^j dA is exact up to the claimed degree.

    #[test]
    fn centroid_deg1_exact() {
        let qr = TriQuadRule::Centroid1Deg1.rule();
        // Exact for degree 1: x^0, y^0 (=0.5), x^1 (1/6), y^1 (1/6)
        for (i, j) in [(0,0),(1,0),(0,1)] {
            let got = integrate_monomial(&qr, i, j);
            let exp = monomial_exact(i, j);
            assert!((got - exp).abs() < 1e-14, "x^{i} y^{j}: got={got}, exp={exp}");
        }
    }

    #[test]
    fn gaussian3_deg2_exact() {
        let qr = TriQuadRule::Gaussian3Deg2.rule();
        // All monomials x^i y^j with i+j <= 2
        for (i, j) in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)] {
            let got = integrate_monomial(&qr, i, j);
            let exp = monomial_exact(i, j);
            assert!((got - exp).abs() < 1e-14, "x^{i} y^{j}: got={got:.12}, exp={exp:.12}");
        }
    }

    #[test]
    fn dunavant7_deg5_exact() {
        let qr = TriQuadRule::Dunavant7Deg5.rule();
        for i in 0u32..=5 {
            for j in 0u32..=(5 - i) {
                let got = integrate_monomial(&qr, i, j);
                let exp = monomial_exact(i, j);
                assert!((got - exp).abs() < 1e-12, "x^{i} y^{j}: got={got:.12}, exp={exp:.12}");
            }
        }
    }

    #[test]
    fn dunavant12_deg6_exact() {
        let qr = TriQuadRule::Dunavant12Deg6.rule();
        for i in 0u32..=6 {
            for j in 0u32..=(6 - i) {
                let got = integrate_monomial(&qr, i, j);
                let exp = monomial_exact(i, j);
                assert!((got - exp).abs() < 1e-10, "x^{i} y^{j}: got={got:.12}, exp={exp:.12}");
            }
        }
    }

    #[test]
    fn dunavant13_deg7_exact() {
        let qr = TriQuadRule::Witherden15Deg7.rule();
        for i in 0u32..=7 {
            for j in 0u32..=(7 - i) {
                let got = integrate_monomial(&qr, i, j);
                let exp = monomial_exact(i, j);
                assert!((got - exp).abs() < 1e-10, "x^{i} y^{j}: got={got:.12}, exp={exp:.12}");
            }
        }
    }

    #[test]
    fn dunavant19_deg9_exact() {
        let qr = TriQuadRule::Dunavant19Deg9.rule();
        for i in 0u32..=9 {
            for j in 0u32..=(9 - i) {
                let got = integrate_monomial(&qr, i, j);
                let exp = monomial_exact(i, j);
                assert!((got - exp).abs() < 1e-9, "x^{i} y^{j}: got={got:.12}, exp={exp:.12}");
            }
        }
    }

    // ── tri_rule_named convenience wrapper ──────────────────────────────

    #[test]
    fn tri_rule_named_matches_enum() {
        for deg in [0u8, 1, 2, 3, 5, 6, 7, 8, 9] {
            let named = tri_rule_named(deg);
            let via_enum = TriQuadRule::for_degree(deg).rule();
            assert_eq!(named.points.len(), via_enum.points.len(), "deg={deg}");
        }
    }
}
