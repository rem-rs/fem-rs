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
    } else {
        // 7-point Dunavant rule (exact for degree 5)
        let s1 = 0.101_286_507_323_456_33;
        let s2 = 0.797_426_985_353_087_2;
        let s3 = 0.470_142_064_105_115_05;
        let t3 = 0.059_715_871_789_769_81;
        let w1 = 0.125_939_180_544_827_17 / 2.0;
        let w2 = 0.132_394_152_788_506_16 / 2.0;
        let w3 = 0.225_000_000_000_000_00 / 2.0;
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
    }
}

// ─── Tetrahedron ──────────────────────────────────────────────────────────────

/// Quadrature rule on the reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`.
///
/// Weights sum to 1/6 (volume of reference tet).
pub fn tet_rule(order: u8) -> QuadratureRule {
    if order <= 1 {
        // 1-point centroid (exact degree 1)
        QuadratureRule {
            points:  vec![vec![0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0],
        }
    } else {
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
    }
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
    let n = ((order as usize + 2) / 2).max(1).min(4);
    let (xs, ws) = gauss_legendre_1d(n);
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
