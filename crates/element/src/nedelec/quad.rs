//! Nedelec-I order-1 element on the reference quadrilateral `[0,1]^2`.

use crate::quadrature::seg_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Lowest-order H(curl) Nedelec element on `[0,1]^2`.
///
/// Local edge ordering:
/// 0. `(0,0) -> (1,0)`
/// 1. `(1,0) -> (1,1)`
/// 2. `(0,1) -> (1,1)`
/// 3. `(0,0) -> (0,1)`
pub struct QuadND1;

impl VectorReferenceElement for QuadND1 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 4 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        // Edge 0 (bottom): tangential x-component on y=0.
        values[0] = 1.0 - y;
        values[1] = 0.0;

        // Edge 1 (right): tangential y-component on x=1.
        values[2] = 0.0;
        values[3] = x;

        // Edge 2 (top): tangential x-component on y=1.
        values[4] = y;
        values[5] = 0.0;

        // Edge 3 (left): tangential y-component on x=0.
        values[6] = 0.0;
        values[7] = 1.0 - x;
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let y = xi[1];
        let x = xi[0];

        // curl(u) = d(u_y)/dx - d(u_x)/dy
        curl_vals[0] = 1.0;
        curl_vals[1] = 1.0;
        curl_vals[2] = -1.0;
        curl_vals[3] = -1.0;

        let _ = (x, y);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let q1 = seg_rule(order);
        let mut points = Vec::with_capacity(q1.n_points() * q1.n_points());
        let mut weights = Vec::with_capacity(q1.n_points() * q1.n_points());
        for (i, xi) in q1.points.iter().enumerate() {
            for (j, eta) in q1.points.iter().enumerate() {
                points.push(vec![xi[0], eta[0]]);
                weights.push(q1.weights[i] * q1.weights[j]);
            }
        }
        QuadratureRule { points, weights }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0],
            vec![1.0, 0.5],
            vec![0.5, 1.0],
            vec![0.0, 0.5],
        ]
    }
}

/// Second-order H(curl) Nedelec element on `[0,1]^2`.
///
/// Tensor-product family (MFEM-style count): `2 * p * (p + 1)` with `p=2`,
/// so `n_dofs = 12`.
pub struct QuadND2;

impl QuadND2 {
    #[inline]
    fn c(i: usize, t: f64) -> f64 {
        match i {
            0 => (1.0 - t) * (1.0 - 2.0 * t),
            1 => 4.0 * t * (1.0 - t),
            2 => t * (2.0 * t - 1.0),
            _ => panic!("QuadND2::c: index out of range"),
        }
    }

    #[inline]
    fn dc(i: usize, t: f64) -> f64 {
        match i {
            0 => -3.0 + 4.0 * t,
            1 => 4.0 - 8.0 * t,
            2 => -1.0 + 4.0 * t,
            _ => panic!("QuadND2::dc: index out of range"),
        }
    }

    #[inline]
    fn o(i: usize, t: f64) -> f64 {
        match i {
            0 => 1.0 - t,
            1 => t,
            _ => panic!("QuadND2::o: index out of range"),
        }
    }

    #[inline]
    fn do_(i: usize) -> f64 {
        match i {
            0 => -1.0,
            1 => 1.0,
            _ => panic!("QuadND2::do_: index out of range"),
        }
    }
}

impl VectorReferenceElement for QuadND2 {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 12 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        let mut n = 0;
        // x-directed family: o_i(x) * c_j(y), i=0..1, j=0..2
        for j in 0..3 {
            for i in 0..2 {
                values[2 * n] = Self::o(i, x) * Self::c(j, y);
                values[2 * n + 1] = 0.0;
                n += 1;
            }
        }
        // y-directed family: c_i(x) * o_j(y), i=0..2, j=0..1
        for j in 0..2 {
            for i in 0..3 {
                values[2 * n] = 0.0;
                values[2 * n + 1] = Self::c(i, x) * Self::o(j, y);
                n += 1;
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];

        let mut n = 0;
        // curl_z = -d/dy (u_x)
        for j in 0..3 {
            for i in 0..2 {
                curl_vals[n] = -Self::o(i, x) * Self::dc(j, y);
                n += 1;
            }
        }
        // curl_z = d/dx (u_y)
        for j in 0..2 {
            for i in 0..3 {
                curl_vals[n] = Self::dc(i, x) * Self::o(j, y);
                n += 1;
            }
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let q1 = seg_rule(order);
        let mut points = Vec::with_capacity(q1.n_points() * q1.n_points());
        let mut weights = Vec::with_capacity(q1.n_points() * q1.n_points());
        for (i, xi) in q1.points.iter().enumerate() {
            for (j, eta) in q1.points.iter().enumerate() {
                points.push(vec![xi[0], eta[0]]);
                weights.push(q1.weights[i] * q1.weights[j]);
            }
        }
        QuadratureRule { points, weights }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.25, 0.0], vec![0.75, 0.0],
            vec![0.25, 0.5], vec![0.75, 0.5],
            vec![0.25, 1.0], vec![0.75, 1.0],
            vec![0.0, 0.25], vec![0.5, 0.25], vec![1.0, 0.25],
            vec![0.0, 0.75], vec![0.5, 0.75], vec![1.0, 0.75],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_nd1_nodal_moments() {
        let elem = QuadND1;
        let mut vals = vec![0.0; elem.n_dofs() * 2];

        // Midpoint tangential moments on unit-length edges.
        let tangents = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        for (j, xi) in elem.dof_coords().iter().enumerate() {
            elem.eval_basis_vec(xi, &mut vals);
            for i in 0..elem.n_dofs() {
                let dot = vals[i * 2] * tangents[j][0] + vals[i * 2 + 1] * tangents[j][1];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "DOF_{j}(Phi_{i}) = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn quad_nd2_counts_and_finite_values() {
        let elem = QuadND2;
        assert_eq!(elem.n_dofs(), 12);

        let xi = [0.37, 0.61];
        let mut v = vec![0.0; elem.n_dofs() * 2];
        let mut c = vec![0.0; elem.n_dofs()];
        elem.eval_basis_vec(&xi, &mut v);
        elem.eval_curl(&xi, &mut c);

        assert!(v.iter().all(|x| x.is_finite()));
        assert!(c.iter().all(|x| x.is_finite()));
    }
}
