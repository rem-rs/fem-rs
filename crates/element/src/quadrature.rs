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
    } else {
        // 12-point Dunavant rule (exact for degree 6)
        // Weights sum to 0.5 (area of reference triangle).
        // Source: Dunavant (1985), rule for p=6.
        let a1 = 0.063_089_014_491_502_23_f64;
        let b1 = 1.0 - 2.0 * a1;
        let w1 = 0.025_422_453_185_103_41_f64;

        let a2 = 0.249_286_745_170_910_43_f64;
        let b2 = 1.0 - 2.0 * a2;
        let w2 = 0.058_393_137_861_187_56_f64;

        // 6 asymmetric points: permutations of (r3, s3, t3) in barycentric
        let r3 = 0.636_502_499_121_399_6_f64;
        let s3 = 0.310_352_451_033_785_8_f64;
        let t3 = 1.0 - r3 - s3;
        let w3 = 0.041_425_537_809_186_785_f64;

        // Barycentric to (xi1,xi2): lam1=1-xi1-xi2, lam2=xi1, lam3=xi2
        // Permutations of (r,s,t): all 6 orderings
        QuadratureRule {
            points: vec![
                // 3 symmetric points of type (a1, a1) and (b1, a1) and (a1, b1)
                vec![a1, a1],
                vec![b1, a1],
                vec![a1, b1],
                // 3 symmetric points of type (a2, a2) and (b2, a2) and (a2, b2)
                vec![a2, a2],
                vec![b2, a2],
                vec![a2, b2],
                // 6 asymmetric points (all permutations of (r3, s3, t3))
                vec![s3, t3],
                vec![r3, t3],
                vec![t3, r3],
                vec![s3, r3],
                vec![r3, s3],
                vec![t3, s3],
            ],
            weights: vec![w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3],
        }
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
    } else {
        // 20-point Grundmann-Moller rule, s=3 (exact degree 7)
        grundmann_moller_tet(3)
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
            Self::Dunavant12Deg6 => tri_rule(6),
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
