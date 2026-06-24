//! PrismRT0 built via Vandermonde: 5 monomials × 5 face DOF functionals.

use std::sync::OnceLock;
use crate::quadrature::{tri_rule, seg_rule, prism_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// 5 monomials spanning RT₀ on the prism:
// m₀=(1,0,0), m₁=(0,1,0), m₂=(0,0,1), m₃=(ξ,0,0), m₄=(0,η,0)
static COEFF: OnceLock<[[f64; 5]; 5]> = OnceLock::new();

fn eval_monos(x: f64, y: f64, _z: f64, out: &mut [f64; 15]) {
    out[0] = 1.0;  out[1] = 0.0;  out[2] = 0.0;
    out[3] = 0.0;  out[4] = 1.0;  out[5] = 0.0;
    out[6] = 0.0;  out[7] = 0.0;  out[8] = 1.0;
    out[9] = x;    out[10] = 0.0; out[11] = 0.0;
    out[12] = 0.0; out[13] = y;   out[14] = 0.0;
}

fn build_vandermonde() -> [[f64; 5]; 5] {
    let mut v = [[0.0; 5]; 5];

    // Face quadrature: tri_rule for triangular faces, seg_rule×seg_rule for quads
    let tri_qr = tri_rule(4);
    let seg_qr = seg_rule(4);

    // f₀ (ξ=0, triangle, n̂=(-1,0,0), area element = dη·dζ)
    for q in 0..tri_qr.n_points() {
        let (e, z) = (tri_qr.points[q][0], tri_qr.points[q][1]);
        let w = tri_qr.weights[q];
        let mut m = [0.0; 15];
        eval_monos(0.0, e, z, &mut m);
        for j in 0..5 {
            v[0][j] += w * (-m[j*3]); // ·(-1,0,0)
        }
    }

    // f₁ (ξ=1, triangle, n̂=(1,0,0), area element = dη·dζ)
    for q in 0..tri_qr.n_points() {
        let (e, z) = (tri_qr.points[q][0], tri_qr.points[q][1]);
        let w = tri_qr.weights[q];
        let mut m = [0.0; 15];
        eval_monos(1.0, e, z, &mut m);
        for j in 0..5 {
            v[1][j] += w * m[j*3]; // ·(1,0,0)
        }
    }

    // f₂ (η=0, quad ξ∈[0,1]×ζ∈[0,1], n̂=(0,-1,0))
    for s in 0..seg_qr.n_points() {
        let xi = seg_qr.points[s][0];
        let ws = seg_qr.weights[s];
        for t in 0..seg_qr.n_points() {
            let zeta = seg_qr.points[t][0];
            let wt = seg_qr.weights[t];
            let w = ws * wt;
            let mut m = [0.0; 15];
            eval_monos(xi, 0.0, zeta, &mut m);
            for j in 0..5 {
                v[2][j] += w * (-m[j*3+1]); // ·(0,-1,0)
            }
        }
    }

    // f₃ (ζ=0, quad ξ∈[0,1]×η∈[0,1], n̂=(0,0,-1))
    for s in 0..seg_qr.n_points() {
        let xi = seg_qr.points[s][0];
        let ws = seg_qr.weights[s];
        for t in 0..seg_qr.n_points() {
            let eta = seg_qr.points[t][0];
            let wt = seg_qr.weights[t];
            let w = ws * wt;
            let mut m = [0.0; 15];
            eval_monos(xi, eta, 0.0, &mut m);
            for j in 0..5 {
                v[3][j] += w * (-m[j*3+2]); // ·(0,0,-1)
            }
        }
    }

    // f₄ (η+ζ=1, ξ∈[0,1], t=η∈[0,1]→ζ=1-t, n̂=(0,1,1)/√2, ds=√2·dξ·dt)
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for s_i in 0..seg_qr.n_points() {
        let xi = seg_qr.points[s_i][0];
        let ws = seg_qr.weights[s_i];
        for t_i in 0..seg_qr.n_points() {
            let eta = seg_qr.points[t_i][0];
            let zeta = 1.0 - eta;
            let wt = seg_qr.weights[t_i];
            let w = ws * wt * 2.0f64.sqrt(); // ds = √2 dξ dt
            let mut m = [0.0; 15];
            eval_monos(xi, eta, zeta, &mut m);
            for j in 0..5 {
                let ndot = s * (m[j*3+1] + m[j*3+2]); // ·(0,1,1)/√2
                v[4][j] += w * ndot;
            }
        }
    }

    v
}

fn invert_5x5(mut a: [[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let n = 5;
    let mut inv = [[0.0f64; 5]; 5];
    for i in 0..n { inv[i][i] = 1.0; }
    for col in 0..n {
        let mut mr = col;
        let mut mv = a[col][col].abs();
        for r in (col+1)..n { let v = a[r][col].abs(); if v > mv { mv = v; mr = r; } }
        if mv < 1e-15 { panic!("PrismRT0 Vandermonde singular at col {col}"); }
        a.swap(col, mr); inv.swap(col, mr);
        let pv = a[col][col]; let ipv = 1.0 / pv;
        for j in 0..n { a[col][j] *= ipv; inv[col][j] *= ipv; }
        for r in 0..n {
            if r == col { continue; }
            let f = a[r][col];
            for j in 0..n { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
        }
    }
    inv
}

fn coeff() -> &'static [[f64; 5]; 5] {
    COEFF.get_or_init(|| {
        let v = build_vandermonde();
        let v_inv = invert_5x5(v);
        // Transpose: C[i][j] = V_inv[j][i]
        let mut c = [[0.0; 5]; 5];
        for i in 0..5 { for j in 0..5 { c[i][j] = v_inv[j][i]; } }
        c
    })
}

pub struct PrismRT0;

impl VectorReferenceElement for PrismRT0 {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 5 }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let c = coeff();
        let mut m = [0.0; 15];
        eval_monos(x, y, z, &mut m);
        for i in 0..5 {
            let mut vx = 0.0; let mut vy = 0.0; let mut vz = 0.0;
            for j in 0..5 {
                vx += c[i][j] * m[j*3];
                vy += c[i][j] * m[j*3+1];
                vz += c[i][j] * m[j*3+2];
            }
            values[i*3] = vx; values[i*3+1] = vy; values[i*3+2] = vz;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() { *v = 0.0; }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        let c = coeff();
        // div(m₃)=1, div(m₄)=1, others 0
        for i in 0..5 {
            div_vals[i] = c[i][3] + c[i][4];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule { prism_rule(order) }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0/3.0, 1.0/3.0],
            vec![1.0, 1.0/3.0, 1.0/3.0],
            vec![0.5, 0.0,     0.5     ],
            vec![0.5, 0.5,     0.0     ],
            vec![0.5, 1.0/3.0, 1.0/3.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadrature::{tri_rule, seg_rule};

    #[test]
    fn prism_rt0_basis_finite() {
        let elem = PrismRT0;
        let mut vals = vec![0.0; 15];
        for pt in &elem.quadrature(3).points {
            elem.eval_basis_vec(pt, &mut vals);
            for v in &vals { assert!(v.is_finite()); }
        }
    }

    #[test]
    fn prism_rt0_vandermonde_invertible() {
        let c = coeff();
        let diag_sum: f64 = (0..5).map(|i| c[i][i].abs()).sum();
        assert!(diag_sum > 1.0, "coeff diagonal too small: {diag_sum}");
    }

    #[test]
    fn prism_rt0_nodal_basis() {
        let elem = PrismRT0;
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        // (outward normal, area) for reference
        let faces: [([f64; 3], f64); 5] = [
            ([-1.0, 0.0, 0.0], 0.5),
            ([1.0,  0.0, 0.0], 0.5),
            ([0.0, -1.0, 0.0], 1.0),
            ([0.0,  0.0,-1.0], 1.0),
            ([0.0, s2, s2], 2.0f64.sqrt()),
        ];

        // Use face quadrature to compute DOF_j(Φ_i) = ∫_{face_j} Φ_i·n̂_j ds.
        // Triangular faces f₀, f₁: tri_rule on (η,ζ)
        // Quad faces f₂, f₃: seg_rule × seg_rule on (ξ,ζ) or (ξ,η)
        // Diagonal face f₄: seg_rule × seg_rule on ξ, η (ζ = 1-η), ds = √2 dξdη
        let tri_qr = tri_rule(4);
        let seg_qr = seg_rule(4);
        let mut vals = vec![0.0; 15];

        for (j, &(n, _area)) in faces.iter().enumerate() {
            let mut dof_i = [0.0f64; 5];

            match j {
                0 => {
                    for q in 0..tri_qr.n_points() {
                        let (eta, zeta) = (tri_qr.points[q][0], tri_qr.points[q][1]);
                        let w = tri_qr.weights[q];
                        elem.eval_basis_vec(&[0.0, eta, zeta], &mut vals);
                        for i in 0..5 {
                            dof_i[i] += w * (vals[i*3]*n[0] + vals[i*3+1]*n[1] + vals[i*3+2]*n[2]);
                        }
                    }
                }
                1 => {
                    for q in 0..tri_qr.n_points() {
                        let (eta, zeta) = (tri_qr.points[q][0], tri_qr.points[q][1]);
                        let w = tri_qr.weights[q];
                        elem.eval_basis_vec(&[1.0, eta, zeta], &mut vals);
                        for i in 0..5 {
                            dof_i[i] += w * (vals[i*3]*n[0] + vals[i*3+1]*n[1] + vals[i*3+2]*n[2]);
                        }
                    }
                }
                2 => {
                    for s in 0..seg_qr.n_points() {
                        let xi = seg_qr.points[s][0];
                        for t in 0..seg_qr.n_points() {
                            let zeta = seg_qr.points[t][0];
                            let w = seg_qr.weights[s] * seg_qr.weights[t];
                            elem.eval_basis_vec(&[xi, 0.0, zeta], &mut vals);
                            for i in 0..5 {
                                dof_i[i] += w * (vals[i*3]*n[0] + vals[i*3+1]*n[1] + vals[i*3+2]*n[2]);
                            }
                        }
                    }
                }
                3 => {
                    for s in 0..seg_qr.n_points() {
                        let xi = seg_qr.points[s][0];
                        for t in 0..seg_qr.n_points() {
                            let eta = seg_qr.points[t][0];
                            let w = seg_qr.weights[s] * seg_qr.weights[t];
                            elem.eval_basis_vec(&[xi, eta, 0.0], &mut vals);
                            for i in 0..5 {
                                dof_i[i] += w * (vals[i*3]*n[0] + vals[i*3+1]*n[1] + vals[i*3+2]*n[2]);
                            }
                        }
                    }
                }
                4 => {
                    for s in 0..seg_qr.n_points() {
                        let xi = seg_qr.points[s][0];
                        for t in 0..seg_qr.n_points() {
                            let eta = seg_qr.points[t][0];
                            let zeta = 1.0 - eta;
                            let w = seg_qr.weights[s] * seg_qr.weights[t] * 2.0f64.sqrt();
                            elem.eval_basis_vec(&[xi, eta, zeta], &mut vals);
                            for i in 0..5 {
                                dof_i[i] += w * (vals[i*3]*n[0] + vals[i*3+1]*n[1] + vals[i*3+2]*n[2]);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }

            for i in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof_i[i] - expected).abs() < 1e-12,
                    "DOF_{j}(Φ_{i}) = {}, expected {expected}", dof_i[i]
                );
            }
        }
    }

    #[test]
    fn prism_rt0_divergence_theorem() {
        let elem = PrismRT0;
        let qr = elem.quadrature(3);
        let mut div = vec![0.0; 5];
        for i in 0..5 {
            let mut integral = 0.0;
            for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
                elem.eval_div(pt, &mut div);
                integral += div[i] * w;
            }
            assert!((integral - 1.0).abs() < 1e-12, "∫div Φ_{i} = {integral}, expected 1");
        }
    }
}
