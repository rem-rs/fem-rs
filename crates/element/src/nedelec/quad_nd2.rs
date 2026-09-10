//! Second-order tensor-product H(curl) element on reference quad `[0,1]^2`.
//!
//! Full ND2 element: 8 edge DOFs (2 tangential modes per side) + 4 interior
//! bubble DOFs, matching MFEM `ND_QuadrilateralElement(2)` (`dof = 2p(p+1)`).
//! Interior modes are the conforming edge-gradient bubbles `y(1-y)·x^m` /
//! `x(1-x)·y^m` (`m = 0, 1`), which — unlike an earlier ad-hoc assembler
//! patch using `(1-y²)` — have zero tangential trace on every edge, so the
//! element is H(curl)-conforming.  The basis is the `[-1,1]²` tensor
//! splitting transported to the MFEM `[0,1]²` reference domain
//! (Piola-equivalent: `φ'(x) = 2φ(2x-1)`).
//!
//! # DOF layout (must match `HCurlSpace` local ordering)
//! ```text
//! 0..8   edge DOFs: bottom (y=0, +x), right (x=1, +y), top (y=1, -x), left (x=0, -y)
//! 8..10  interior x-component bubbles: y(1-y)·1, y(1-y)·x
//! 10..12 interior y-component bubbles: x(1-x)·1, x(1-x)·y
//! ```

use crate::quadrature::quad_rule_01;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Second-order H(curl) element on reference quad, 12 DOFs (8 edge + 4 interior).
pub struct QuadND2;

impl VectorReferenceElement for QuadND2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        12
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // Two edge modes per side (8 total): split ND1 edge traces by linear factors.
        // bottom edge (y=0), +x
        values[0] = (1.0 - y) * (1.0 - x);
        values[1] = 0.0;
        values[2] = x * (1.0 - y);
        values[3] = 0.0;

        // right edge (x=1), +y
        values[4] = 0.0;
        values[5] = x * (1.0 - y);
        values[6] = 0.0;
        values[7] = x * y;

        // top edge (y=1), -x
        values[8] = -x * y;
        values[9] = 0.0;
        values[10] = -(1.0 - x) * y;
        values[11] = 0.0;

        // left edge (x=0), -y
        values[12] = 0.0;
        values[13] = -(1.0 - x) * y;
        values[14] = 0.0;
        values[15] = -(1.0 - x) * (1.0 - y);

        // Interior bubbles (tangentially conforming: zero tangential trace
        // on every edge).  x-component: y(1-y) · {1, x}
        values[16] = y * (1.0 - y);
        values[17] = 0.0;
        values[18] = x * y * (1.0 - y);
        values[19] = 0.0;
        // y-component: x(1-x) · {1, y}
        values[20] = 0.0;
        values[21] = x * (1.0 - x);
        values[22] = 0.0;
        values[23] = x * (1.0 - x) * y;
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // scalar curl in 2D: dFy/dx - dFx/dy
        curl_vals[0] = 1.0 - x;   // Φ0 = ((1-y)(1-x), 0)
        curl_vals[1] = x;         // Φ1 = (x(1-y), 0)
        curl_vals[2] = 1.0 - y;   // Φ2 = (0, x(1-y))
        curl_vals[3] = y;         // Φ3 = (0, xy)
        curl_vals[4] = x;         // Φ4 = (-xy, 0)
        curl_vals[5] = 1.0 - x;   // Φ5 = (-y(1-x), 0)
        curl_vals[6] = y;         // Φ6 = (0, -y(1-x))
        curl_vals[7] = 1.0 - y;   // Φ7 = (0, -(1-x)(1-y))
        // interior x-comp Φ = (g(x)·y(1-y), 0): curl = -∂Φ_x/∂y = -g(x)(1-2y)
        curl_vals[8] = -(1.0 - 2.0 * y);      // g = 1
        curl_vals[9] = -x * (1.0 - 2.0 * y);  // g = x
        // interior y-comp Φ = (0, h(y)·x(1-x)): curl = ∂Φ_y/∂x = h(y)(1-2x)
        curl_vals[10] = 1.0 - 2.0 * x;        // h = 1
        curl_vals[11] = y * (1.0 - 2.0 * x);  // h = y
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        quad_rule_01(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        // Two Gauss points 1/3, 2/3 per edge + interior bubble anchors.
        vec![
            vec![1.0 / 3.0, 0.0],
            vec![2.0 / 3.0, 0.0],
            vec![1.0, 1.0 / 3.0],
            vec![1.0, 2.0 / 3.0],
            vec![1.0 / 3.0, 1.0],
            vec![2.0 / 3.0, 1.0],
            vec![0.0, 1.0 / 3.0],
            vec![0.0, 2.0 / 3.0],
            // interior anchors (x-comp at mid-plane, y-comp at mid-line)
            vec![1.0 / 3.0, 0.5],
            vec![2.0 / 3.0, 0.5],
            vec![0.5, 1.0 / 3.0],
            vec![0.5, 2.0 / 3.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd2_quad_basis_and_curl_are_finite() {
        let elem = QuadND2;
        let qr = elem.quadrature(4);
        let mut phi = vec![0.0; elem.n_dofs() * 2];
        let mut curl = vec![0.0; elem.n_dofs()];
        for xi in &qr.points {
            elem.eval_basis_vec(xi, &mut phi);
            elem.eval_curl(xi, &mut curl);
            assert!(phi.iter().all(|v| v.is_finite()));
            assert!(curl.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn nd2_splits_nd1_traces() {
        // The 8 edge modes split the 4 ND1 traces into pairs:
        //   bottom: Φ0+Φ1 = (1-y, 0)     right: Φ2+Φ3 = (0, x)
        //   top:    Φ4+Φ5 = (-y, 0)      left:  Φ6+Φ7 = (0, x-1)
        let elem = QuadND2;
        let nd1 = crate::nedelec::QuadNDk::new(1);
        let mut vals = vec![0.0; 24];
        let mut ref1 = vec![0.0; 8];
        let mut curl = vec![0.0; 12];
        let mut curl1 = vec![0.0; 4];
        for xi in elem.quadrature(5).points {
            let xi: &[f64] = &xi;
            elem.eval_basis_vec(xi, &mut vals);
            elem.eval_curl(xi, &mut curl);
            nd1.eval_basis_vec(xi, &mut ref1);
            nd1.eval_curl(xi, &mut curl1);
            for d in 0..2 {
                let sum0 = vals[0 * 2 + d] + vals[1 * 2 + d];
                let sum1 = vals[2 * 2 + d] + vals[3 * 2 + d];
                let sum2 = vals[4 * 2 + d] + vals[5 * 2 + d];
                let sum3 = vals[6 * 2 + d] + vals[7 * 2 + d];
                assert!((sum0 - ref1[0 * 2 + d]).abs() < 1e-13, "bottom comp {d}");
                assert!((sum1 - ref1[1 * 2 + d]).abs() < 1e-13, "right comp {d}");
                assert!((sum2 - ref1[2 * 2 + d]).abs() < 1e-13, "top comp {d}");
                assert!((sum3 - ref1[3 * 2 + d]).abs() < 1e-13, "left comp {d}");
            }
            for i in 0..4 {
                assert!(
                    (curl[2 * i] + curl[2 * i + 1] - curl1[i]).abs() < 1e-13,
                    "curl pair {i}"
                );
            }
        }
    }

    /// Interior bubbles must have zero TANGENTIAL trace on every edge (the
    /// normal component may be nonzero — H(curl) only requires tangential
    /// continuity).  x-comp modes die on top/bottom, y-comp modes on left/right.
    #[test]
    fn nd2_interior_bubbles_vanish_on_edges() {
        let elem = QuadND2;
        let mut vals = vec![0.0; 24];
        // (edge tangent index: 0 → check Φ_x, 1 → check Φ_y)
        let cases: [([f64; 2], usize, &str); 8] = [
            ([0.3, 0.0], 0, "bottom x-comp"),
            ([0.7, 0.0], 0, "bottom x-comp 2"),
            ([0.3, 1.0], 0, "top x-comp"),
            ([0.7, 1.0], 0, "top x-comp 2"),
            ([0.0, 0.4], 1, "left y-comp"),
            ([0.0, 0.8], 1, "left y-comp 2"),
            ([1.0, 0.4], 1, "right y-comp"),
            ([1.0, 0.8], 1, "right y-comp 2"),
        ];
        for (xi, comp, what) in cases {
            elem.eval_basis_vec(&xi, &mut vals);
            for i in 8..12 {
                let tang = vals[i * 2 + comp];
                assert!(
                    tang.abs() < 1e-14,
                    "interior bubble {i} tangential trace nonzero ({what}) at {xi:?}: {tang}"
                );
            }
        }
    }

    /// The constant field (1,0) must lie in the local span: the x-component
    /// modes {(1-y)(1-x), x(1-y), -xy, -(1-x)y, y(1-y), xy(1-y)} span
    /// P1(x)⊗P2(y).  Verify by collocation at 6 distinct points.
    #[test]
    fn nd2_local_span_contains_constant() {
        let elem = QuadND2;
        let pts = [
            [0.1f64, 0.2f64],
            [0.5, 0.1],
            [0.9, 0.6],
            [0.3, 0.8],
            [0.7, 0.4],
            [0.2, 0.5],
        ];
        let xidx = [0usize, 1, 4, 5, 8, 9];
        let n = xidx.len();
        let mut a = nalgebra::DMatrix::<f64>::zeros(n, n);
        let b = nalgebra::DVector::from_element(n, 1.0);
        for (r, p) in pts.iter().enumerate() {
            let mut vals = vec![0.0; 24];
            elem.eval_basis_vec(p, &mut vals);
            for (c, &i) in xidx.iter().enumerate() {
                a[(r, c)] = vals[i * 2];
            }
        }
        let sol = a.lu().solve(&b).expect("x-comp collocation singular");
        let mut vals = vec![0.0; 24];
        let mut max_err = 0.0f64;
        for p in &pts {
            elem.eval_basis_vec(p, &mut vals);
            let ux: f64 = xidx.iter().zip(sol.iter()).map(|(&i, &c)| c * vals[i * 2]).sum();
            max_err = max_err.max((ux - 1.0).abs());
        }
        assert!(max_err < 1e-12, "constant x-comp not representable: {max_err:.3e}");
    }
}
