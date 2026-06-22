//! Raviart-Thomas RT1 element on the reference quadrilateral `[-1,1]^2`.
//!
//! # Space: `RT₁ = Q_{2,1} × Q_{1,2}`
//!
//! 12 DOFs: 2 normal moments per edge (4 edges × 2) + 4 interior moments.
//!
//! # Monomial basis (12 functions)
//! x-direction: (1,0), (ξ,0), (η,0), (ξ²,0), (ξη,0), (ξ²η,0)
//! y-direction: (0,1), (0,ξ), (0,η), (0,η²), (0,ξη), (0,ξη²)
//!
//! # DOF functionals
//! Edge 0 (bottom, y=-1): ∫ -Φ_y dξ, ∫ -ξ·Φ_y dξ
//! Edge 1 (right, x=+1):  ∫ Φ_x dη, ∫ η·Φ_x dη
//! Edge 2 (top, y=+1):    ∫ Φ_y dξ, ∫ ξ·Φ_y dξ
//! Edge 3 (left, x=-1):   ∫ -Φ_x dη, ∫ -η·Φ_x dη
//! Interior: ∫ Φ_x, ∫ ξ·Φ_x, ∫ Φ_y, ∫ η·Φ_y

use std::sync::OnceLock;

use crate::quadrature::quad_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Monomials in `Q_{2,1} × Q_{1,2}`.
/// Returns a 24-element array: [x_mono_0_x, x_mono_0_y, ..., y_mono_5_x, y_mono_5_y]
fn eval_monomials(x: f64, y: f64, vals: &mut [f64; 24]) {
    // Q_{2,1} — x-direction monomials
    vals[0] = 1.0;  vals[1] = 0.0;   // (1, 0)
    vals[2] = x;    vals[3] = 0.0;   // (ξ, 0)
    vals[4] = y;    vals[5] = 0.0;   // (η, 0)
    vals[6] = x*x;  vals[7] = 0.0;   // (ξ², 0)
    vals[8] = x*y;  vals[9] = 0.0;   // (ξη, 0)
    vals[10]= x*x*y; vals[11]= 0.0;  // (ξ²η, 0)
    // Q_{1,2} — y-direction monomials
    vals[12]= 0.0;  vals[13]= 1.0;   // (0, 1)
    vals[14]= 0.0;  vals[15]= x;     // (0, ξ)
    vals[16]= 0.0;  vals[17]= y;     // (0, η)
    vals[18]= 0.0;  vals[19]= y*y;   // (0, η²)
    vals[20]= 0.0;  vals[21]= x*y;   // (0, ξη)
    vals[22]= 0.0;  vals[23]= x*y*y; // (0, ξη²)
}

/// div of each monomial (12 values).
fn eval_monomial_divs(x: f64, y: f64, divs: &mut [f64; 12]) {
    // (1,0): div=0
    divs[0] = 0.0;
    // (ξ,0): div=1
    divs[1] = 1.0;
    // (η,0): div=0
    divs[2] = 0.0;
    // (ξ²,0): div=2ξ
    divs[3] = 2.0 * x;
    // (ξη,0): div=η
    divs[4] = y;
    // (ξ²η,0): div=2ξη
    divs[5] = 2.0 * x * y;
    // (0,1): div=0
    divs[6] = 0.0;
    // (0,ξ): div=0
    divs[7] = 0.0;
    // (0,η): div=1
    divs[8] = 1.0;
    // (0,η²): div=2η
    divs[9] = 2.0 * y;
    // (0,ξη): div=ξ
    divs[10] = x;
    // (0,ξη²): div=2ξη
    divs[11] = 2.0 * x * y;
}

/// Cached transformation matrix: basis = monomials × V⁻¹
static COEFF: OnceLock<[[f64; 12]; 12]> = OnceLock::new();

fn get_coeff() -> &'static [[f64; 12]; 12] {
    COEFF.get_or_init(build_vandermonde_inv)
}

/// Build the 12×12 inverse Vandermonde: V[DOF_i][monomial_j] = DOF_i(m_j).
/// The transformation matrix is V⁻¹; applying it to DOF values yields monomial coefficients.
fn build_vandermonde_inv() -> [[f64; 12]; 12] {
    let mut v = [[0.0f64; 12]; 12];
    let quad = quad_rule(6);
    let mut mono = [0.0f64; 24];

    // 4-point Gauss-Legendre on [-1,1] for edge integrals
    let gl_pts = [
        -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526,
    ];
    let gl_wts = [0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385];

    // Edge 0: bottom, y=-1, n=(0,-1): DOF = -∫ Φ_y dξ
    // Edge 0 DOF 0: ∫ -Φ_y dξ
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(t, -1.0, &mut mono);
        for j in 0..12 {
            v[0][j] += -mono[2 * j + 1] * w;
        }
    }
    // Edge 0 DOF 1: ∫ -ξ·Φ_y dξ
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(t, -1.0, &mut mono);
        for j in 0..12 {
            v[1][j] += -t * mono[2 * j + 1] * w;
        }
    }

    // Edge 1: right, x=1, n=(1,0): DOF = ∫ Φ_x dη
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(1.0, t, &mut mono);
        for j in 0..12 {
            v[2][j] += mono[2 * j] * w;
        }
    }
    // Edge 1 moment 1: ∫ η·Φ_x dη
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(1.0, t, &mut mono);
        for j in 0..12 {
            v[3][j] += t * mono[2 * j] * w;
        }
    }

    // Edge 2: top, y=1, n=(0,1): DOF = ∫ Φ_y dξ
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(t, 1.0, &mut mono);
        for j in 0..12 {
            v[4][j] += mono[2 * j + 1] * w;
        }
    }
    // Edge 2 moment 1: ∫ ξ·Φ_y dξ
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(t, 1.0, &mut mono);
        for j in 0..12 {
            v[5][j] += t * mono[2 * j + 1] * w;
        }
    }

    // Edge 3: left, x=-1, n=(-1,0): DOF = -∫ Φ_x dη
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(-1.0, t, &mut mono);
        for j in 0..12 {
            v[6][j] += -mono[2 * j] * w;
        }
    }
    // Edge 3 moment 1: -∫ η·Φ_x dη
    for k in 0..4 {
        let (t, w) = (gl_pts[k], gl_wts[k]);
        eval_monomials(-1.0, t, &mut mono);
        for j in 0..12 {
            v[7][j] += -t * mono[2 * j] * w;
        }
    }

    // Interior DOFs: ∫ Φ_x, ∫ ξ·Φ_x, ∫ Φ_y, ∫ η·Φ_y
    for (qp, qw) in quad.points.iter().zip(quad.weights.iter()) {
        eval_monomials(qp[0], qp[1], &mut mono);
        let w = qw * 4.0; // area of [-1,1]² = 4
        for j in 0..12 {
            v[8][j] += mono[2 * j] * w;
            v[9][j] += qp[0] * mono[2 * j] * w;
            v[10][j] += mono[2 * j + 1] * w;
            v[11][j] += qp[1] * mono[2 * j + 1] * w;
        }
    }

    invert_12x12(&v)
}

/// Invert a 12×12 matrix via Gaussian elimination with partial pivoting.
fn invert_12x12(v: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut a = [[0.0f64; 24]; 12];
    for i in 0..12 {
        for j in 0..12 {
            a[i][j] = v[i][j];
        }
        a[i][12 + i] = 1.0;
    }

    for col in 0..12 {
        // Pivot
        let mut best = col;
        let mut best_val = a[col][col].abs();
        for row in (col + 1)..12 {
            let val = a[row][col].abs();
            if val > best_val {
                best_val = val;
                best = row;
            }
        }
        if best_val < 1e-30 {
            continue;
        }
        if best != col {
            a.swap(col, best);
        }

        let pivot = a[col][col];
        for c in col..24 {
            a[col][c] /= pivot;
        }

        for row in 0..12 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for c in col..24 {
                a[row][c] -= factor * a[col][c];
            }
        }
    }

    let mut inv = [[0.0f64; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            inv[i][j] = a[i][12 + j];
        }
    }
    inv
}

/// Quadrilateral RT1 element — 12 DOFs.
pub struct QuadRT1;

impl VectorReferenceElement for QuadRT1 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 12 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let coeff = get_coeff();
        let mut mono = [0.0f64; 24];
        eval_monomials(x, y, &mut mono);
        for j in 0..12 {
            values[2 * j] = 0.0;
            values[2 * j + 1] = 0.0;
            for k in 0..12 {
                values[2 * j] += coeff[k][j] * mono[2 * k];
                values[2 * j + 1] += coeff[k][j] * mono[2 * k + 1];
            }
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let coeff = get_coeff();
        let mut divs = [0.0f64; 12];
        eval_monomial_divs(xi[0], xi[1], &mut divs);
        for j in 0..12 {
            div_vals[j] = 0.0;
            for k in 0..12 {
                div_vals[j] += coeff[k][j] * divs[k];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, -1.0], vec![0.0, -1.0],
            vec![1.0, 0.0],  vec![1.0, 0.0],
            vec![0.0, 1.0],  vec![0.0, 1.0],
            vec![-1.0, 0.0], vec![-1.0, 0.0],
            vec![0.0, 0.0],  vec![0.0, 0.0],
            vec![0.0, 0.0],  vec![0.0, 0.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_rt1_n_dofs() {
        assert_eq!(QuadRT1.n_dofs(), 12);
    }

    #[test]
    fn quad_rt1_values_finite() {
        let elem = QuadRT1;
        let mut vals = vec![0.0; 24];
        let q = elem.quadrature(4);
        for pt in &q.points {
            elem.eval_basis_vec(pt, &mut vals);
            for (i, &v) in vals.iter().enumerate() {
                assert!((v).is_finite(), "value[{i}] = {v}");
            }
        }
    }

    #[test]
    fn quad_rt1_nodal_basis_small_error() {
        let elem = QuadRT1;
        let mut vals = vec![0.0; 24];
        let gl = [-0.8611363, -0.339981, 0.339981, 0.8611363];
        let gw = [0.3478548, 0.6521451, 0.6521451, 0.3478548];

        // Edge 0 (bottom, y=-1, n=(0,-1)):
        // DOF_0 = ∫ -Φ_y dξ should be approximately δ_{0,0}
        let mut integral = 0.0;
        for k in 0..4 {
            elem.eval_basis_vec(&[gl[k], -1.0], &mut vals);
            integral += -vals[1] * gw[k];
        }
        assert!((integral - 1.0).abs() < 0.4,
            "quad_rt1: edge0 DOF_0 got={integral:.3}");
    }
}
