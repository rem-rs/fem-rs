//! Nedelec-I order-1 element on the reference hexahedron `[0,1]^3`.

use crate::quadrature::seg_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// Lowest-order H(curl) Nedelec element on `[0,1]^3`.
///
/// Local edge ordering (matches `HCurlSpace` Hex8 table):
/// 0:(0,1) 1:(3,2) 2:(4,5) 3:(7,6)
/// 4:(0,3) 5:(1,2) 6:(4,7) 7:(5,6)
/// 8:(0,4) 9:(1,5) 10:(2,6) 11:(3,7)
pub struct HexND1;

impl VectorReferenceElement for HexND1 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 1 }
    fn n_dofs(&self) -> usize { 12 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        // x-directed edge basis functions
        set_vec(values, 0, (1.0 - y) * (1.0 - z), 0.0, 0.0);
        set_vec(values, 1, y * (1.0 - z), 0.0, 0.0);
        set_vec(values, 2, (1.0 - y) * z, 0.0, 0.0);
        set_vec(values, 3, y * z, 0.0, 0.0);

        // y-directed edge basis functions
        set_vec(values, 4, 0.0, (1.0 - x) * (1.0 - z), 0.0);
        set_vec(values, 5, 0.0, x * (1.0 - z), 0.0);
        set_vec(values, 6, 0.0, (1.0 - x) * z, 0.0);
        set_vec(values, 7, 0.0, x * z, 0.0);

        // z-directed edge basis functions
        set_vec(values, 8, 0.0, 0.0, (1.0 - x) * (1.0 - y));
        set_vec(values, 9, 0.0, 0.0, x * (1.0 - y));
        set_vec(values, 10, 0.0, 0.0, x * y);
        set_vec(values, 11, 0.0, 0.0, (1.0 - x) * y);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        // Edge 0..3: u=(f,0,0) => curl=(0, df/dz, -df/dy)
        set_vec(curl_vals, 0, 0.0, -(1.0 - y), 1.0 - z);
        set_vec(curl_vals, 1, 0.0, -y, -(1.0 - z));
        set_vec(curl_vals, 2, 0.0, 1.0 - y, z);
        set_vec(curl_vals, 3, 0.0, y, -z);

        // Edge 4..7: u=(0,g,0) => curl=(-dg/dz,0,dg/dx)
        set_vec(curl_vals, 4, 1.0 - x, 0.0, -(1.0 - z));
        set_vec(curl_vals, 5, x, 0.0, 1.0 - z);
        set_vec(curl_vals, 6, -(1.0 - x), 0.0, -z);
        set_vec(curl_vals, 7, -x, 0.0, z);

        // Edge 8..11: u=(0,0,h) => curl=(dh/dy,-dh/dx,0)
        set_vec(curl_vals, 8, -(1.0 - x), 1.0 - y, 0.0);
        set_vec(curl_vals, 9, -x, -(1.0 - y), 0.0);
        set_vec(curl_vals, 10, x, -y, 0.0);
        set_vec(curl_vals, 11, 1.0 - x, y, 0.0);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        let q1 = seg_rule(order);
        let nq = q1.n_points();
        let mut points = Vec::with_capacity(nq * nq * nq);
        let mut weights = Vec::with_capacity(nq * nq * nq);
        for (i, xi) in q1.points.iter().enumerate() {
            for (j, eta) in q1.points.iter().enumerate() {
                for (k, zeta) in q1.points.iter().enumerate() {
                    points.push(vec![xi[0], eta[0], zeta[0]]);
                    weights.push(q1.weights[i] * q1.weights[j] * q1.weights[k]);
                }
            }
        }
        QuadratureRule { points, weights }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.5, 0.0, 0.0],
            vec![0.5, 1.0, 0.0],
            vec![0.5, 0.0, 1.0],
            vec![0.5, 1.0, 1.0],
            vec![0.0, 0.5, 0.0],
            vec![1.0, 0.5, 0.0],
            vec![0.0, 0.5, 1.0],
            vec![1.0, 0.5, 1.0],
            vec![0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.5],
            vec![1.0, 1.0, 0.5],
            vec![0.0, 1.0, 0.5],
        ]
    }
}

/// Second-order H(curl) Nedelec element on `[0,1]^3`.
///
/// Tensor-product family (MFEM-style count): `3 * p * (p + 1)^2` with `p=2`,
/// so `n_dofs = 54`.
pub struct HexND2;

impl HexND2 {
    #[inline]
    fn c(i: usize, t: f64) -> f64 {
        match i {
            0 => (1.0 - t) * (1.0 - 2.0 * t),
            1 => 4.0 * t * (1.0 - t),
            2 => t * (2.0 * t - 1.0),
            _ => panic!("HexND2::c: index out of range"),
        }
    }

    #[inline]
    fn dc(i: usize, t: f64) -> f64 {
        match i {
            0 => -3.0 + 4.0 * t,
            1 => 4.0 - 8.0 * t,
            2 => -1.0 + 4.0 * t,
            _ => panic!("HexND2::dc: index out of range"),
        }
    }

    #[inline]
    fn o(i: usize, t: f64) -> f64 {
        match i {
            0 => 1.0 - t,
            1 => t,
            _ => panic!("HexND2::o: index out of range"),
        }
    }

    #[inline]
    fn do_(i: usize) -> f64 {
        match i {
            0 => -1.0,
            1 => 1.0,
            _ => panic!("HexND2::do_: index out of range"),
        }
    }
}

impl VectorReferenceElement for HexND2 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 2 }
    fn n_dofs(&self) -> usize { 54 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        let mut n = 0;
        // x-directed: o_i(x) * c_j(y) * c_k(z), i=0..1, j=0..2, k=0..2
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..2 {
                    set_vec(values, n, Self::o(i, x) * Self::c(j, y) * Self::c(k, z), 0.0, 0.0);
                    n += 1;
                }
            }
        }

        // y-directed: c_i(x) * o_j(y) * c_k(z), i=0..2, j=0..1, k=0..2
        for k in 0..3 {
            for j in 0..2 {
                for i in 0..3 {
                    set_vec(values, n, 0.0, Self::c(i, x) * Self::o(j, y) * Self::c(k, z), 0.0);
                    n += 1;
                }
            }
        }

        // z-directed: c_i(x) * c_j(y) * o_k(z), i=0..2, j=0..2, k=0..1
        for k in 0..2 {
            for j in 0..3 {
                for i in 0..3 {
                    set_vec(values, n, 0.0, 0.0, Self::c(i, x) * Self::c(j, y) * Self::o(k, z));
                    n += 1;
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];

        let mut n = 0;
        // x-directed u=(f,0,0): curl=(0, df/dz, -df/dy)
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..2 {
                    let ox = Self::o(i, x);
                    let cy = Self::c(j, y);
                    let cz = Self::c(k, z);
                    let dcy = Self::dc(j, y);
                    let dcz = Self::dc(k, z);
                    set_vec(curl_vals, n, 0.0, ox * cy * dcz, -ox * dcy * cz);
                    n += 1;
                }
            }
        }

        // y-directed u=(0,g,0): curl=(-dg/dz, 0, dg/dx)
        for k in 0..3 {
            for j in 0..2 {
                for i in 0..3 {
                    let cx = Self::c(i, x);
                    let oy = Self::o(j, y);
                    let cz = Self::c(k, z);
                    let dcx = Self::dc(i, x);
                    let dcz = Self::dc(k, z);
                    set_vec(curl_vals, n, -cx * oy * dcz, 0.0, dcx * oy * cz);
                    n += 1;
                }
            }
        }

        // z-directed u=(0,0,h): curl=(dh/dy, -dh/dx, 0)
        for k in 0..2 {
            for j in 0..3 {
                for i in 0..3 {
                    let cx = Self::c(i, x);
                    let cy = Self::c(j, y);
                    let oz = Self::o(k, z);
                    let dcx = Self::dc(i, x);
                    let dcy = Self::dc(j, y);
                    set_vec(curl_vals, n, cx * dcy * oz, -dcx * cy * oz, 0.0);
                    n += 1;
                }
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
        let nq = q1.n_points();
        let mut points = Vec::with_capacity(nq * nq * nq);
        let mut weights = Vec::with_capacity(nq * nq * nq);
        for (i, xi) in q1.points.iter().enumerate() {
            for (j, eta) in q1.points.iter().enumerate() {
                for (k, zeta) in q1.points.iter().enumerate() {
                    points.push(vec![xi[0], eta[0], zeta[0]]);
                    weights.push(q1.weights[i] * q1.weights[j] * q1.weights[k]);
                }
            }
        }
        QuadratureRule { points, weights }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let cp = [0.0, 0.5, 1.0];
        let op = [0.25, 0.75];
        let mut pts = Vec::with_capacity(54);

        for &z in &cp {
            for &y in &cp {
                for &x in &op {
                    pts.push(vec![x, y, z]);
                }
            }
        }
        for &z in &cp {
            for &y in &op {
                for &x in &cp {
                    pts.push(vec![x, y, z]);
                }
            }
        }
        for &z in &op {
            for &y in &cp {
                for &x in &cp {
                    pts.push(vec![x, y, z]);
                }
            }
        }
        pts
    }
}

#[inline]
fn set_vec(buf: &mut [f64], i: usize, x: f64, y: f64, z: f64) {
    buf[i * 3] = x;
    buf[i * 3 + 1] = y;
    buf[i * 3 + 2] = z;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_nd1_nodal_moments() {
        let elem = HexND1;
        let mut vals = vec![0.0; elem.n_dofs() * 3];

        let tangents: [[f64; 3]; 12] = [
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        ];

        for (j, xi) in elem.dof_coords().iter().enumerate() {
            elem.eval_basis_vec(xi, &mut vals);
            for i in 0..elem.n_dofs() {
                let dot = vals[i * 3] * tangents[j][0]
                    + vals[i * 3 + 1] * tangents[j][1]
                    + vals[i * 3 + 2] * tangents[j][2];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "DOF_{j}(Phi_{i}) = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn hex_nd2_counts_and_finite_values() {
        let elem = HexND2;
        assert_eq!(elem.n_dofs(), 54);

        let xi = [0.21, 0.47, 0.69];
        let mut v = vec![0.0; elem.n_dofs() * 3];
        let mut c = vec![0.0; elem.n_dofs() * 3];
        elem.eval_basis_vec(&xi, &mut v);
        elem.eval_curl(&xi, &mut c);

        assert!(v.iter().all(|x| x.is_finite()));
        assert!(c.iter().all(|x| x.is_finite()));
    }
}
