//! Arbitrary-order Nedelec-I element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! Uses the same construction as MFEM `ND_TriangleElement`:
//! - Edge DOFs at Gauss-Legendre open points
//! - Whitney 1-forms for lowest order, polynomial expansion for higher order
//! - Interior DOFs for k >= 2

use crate::quadrature::tri_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};
use std::sync::OnceLock;

struct TriNDkData {
    coeff: Vec<f64>, // [n×n] row-major, C[i][j] for Φ_i = Σ_j C[i][j]·m_j
    n: usize,        // dimension = k(k+2)
    order: usize,
    monomap: Vec<usize>, // selected monomial indices (length n)
}

fn edge_integral_x(p: usize, xi: f64, eta: f64) -> f64 {
    if eta == 0.0 {
        1.0 / (xi + p as f64 + 1.0)
    } else {
        0.0
    }
}

fn edge_integral_y(p: usize, xi: f64, eta: f64) -> f64 {
    if eta == 0.0 {
        1.0 / (xi + p as f64 + 1.0)
    } else {
        0.0
    }
}

fn area_integral(ix: usize, iy: usize) -> f64 {
    let mut num = 1.0_f64;
    for i in 1..=ix {
        num *= i as f64;
    }
    for i in 1..=iy {
        num *= i as f64;
    }
    let mut den = 1.0_f64;
    for i in 1..=(ix + iy + 2) {
        den *= i as f64;
    }
    num / den
}

fn beta_int(a: usize, bp: usize) -> f64 {
    let mut num = 1.0_f64;
    for i in 1..=a {
        num *= i as f64;
    }
    for i in 1..=bp {
        num *= i as f64;
    }
    let mut den = 1.0_f64;
    for i in 1..=(a + bp + 1) {
        den *= i as f64;
    }
    num / den
}

fn tri_data(k: usize) -> &'static TriNDkData {
    static CACHE: [OnceLock<TriNDkData>; 9] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k - 1].get_or_init(|| build_tri_data(k))
}

fn build_tri_data(k: usize) -> TriNDkData {
    let n = k * (k + 2); // dimension

    // Generate ALL monomials (comp, a, b) with degree a+b ≤ k
    struct Mono {
        comp: u8,
        a: usize,
        b: usize,
    }
    let mut monos: Vec<Mono> = Vec::new();
    for deg in 0..=k {
        for a in 0..=deg {
            let b = deg - a;
            monos.push(Mono { comp: 0, a, b });
            monos.push(Mono { comp: 1, a, b });
        }
    }
    let m_total = monos.len(); // (k+1)(k+2)

    // Build Vandermonde V[i][j] = DOF_i(m_j) for i=0..n-1, j=0..m_total-1
    let mut v = vec![vec![0.0_f64; m_total]; n];

    // DOFs 0..k-1: edge e0 (η=0), x-component tangential, moments ξ^p
    for p in 0..k {
        for (j, m) in monos.iter().enumerate() {
            let val = if m.comp == 0 {
                edge_integral_x(p, m.a as f64, m.b as f64)
            } else {
                edge_integral_y(p, m.a as f64, m.b as f64)
            };
            v[p][j] = val;
        }
    }

    // DOFs k..2k-1: edge e1 (1-t,t), tangential = -Φ_x+Φ_y, moments t^p
    for p in 0..k {
        for (j, m) in monos.iter().enumerate() {
            if m.comp == 0 {
                let val = -beta_int(m.a, m.b + p);
                v[k + p][j] = val;
            } else {
                let val = beta_int(m.a, m.b + p);
                v[k + p][j] = val;
            }
        }
    }

    // DOFs 2k..3k-1: edge e2 (ξ=0, η from 0 to 1), tangential = Φ_y, moments η^p
    for p in 0..k {
        for (j, m) in monos.iter().enumerate() {
            let val = if m.comp == 0 {
                0.0
            } else {
                if m.a == 0 {
                    1.0 / (m.b as f64 + p as f64 + 1.0)
                } else {
                    0.0
                }
            };
            v[2 * k + p][j] = val;
        }
    }

    // Interior DOFs 3k..n-1: ∫ Φ_x·x^ix y^iy dA and ∫ Φ_y·x^ix y^iy dA
    let mut dof_idx = 3 * k;
    if k >= 2 {
        for deg in 0..=(k - 2) {
            for ix in 0..=deg {
                let iy = deg - ix;
                for (j, m) in monos.iter().enumerate() {
                    if m.comp == 0 {
                        v[dof_idx][j] = area_integral(m.a + ix, m.b + iy);
                    }
                }
                dof_idx += 1;
                for (j, m) in monos.iter().enumerate() {
                    if m.comp == 1 {
                        v[dof_idx][j] = area_integral(m.a + ix, m.b + iy);
                    }
                }
                dof_idx += 1;
            }
        }
    }
    assert_eq!(dof_idx, n);

    // Gauss-Jordan with column pivoting to select n linearly independent monomials
    let mut col_perm: Vec<usize> = (0..m_total).collect();
    let mut row = vec![vec![0.0_f64; n + m_total]; n];
    for i in 0..n {
        for j in 0..m_total {
            row[i][j] = v[i][j];
        }
        row[i][m_total + i] = 1.0;
    }

    let mut selected = Vec::new();
    for col in 0..n {
        let mut best_col = col;
        let mut best_val = 0.0_f64;
        for c in col..m_total {
            let mut sum = 0.0_f64;
            for r in 0..n {
                sum += row[r][c].abs();
            }
            if sum > best_val {
                best_val = sum;
                best_col = c;
            }
        }
        if best_col != col {
            for r in 0..n {
                row[r].swap(col, best_col);
            }
            col_perm.swap(col, best_col);
        }
        selected.push(col_perm[col]);

        let pivot = row[col][col];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for j in 0..(n + m_total) {
            row[col][j] /= pivot;
        }
        for r in 0..n {
            if r != col {
                let factor = row[r][col];
                for j in 0..(n + m_total) {
                    row[r][j] -= factor * row[col][j];
                }
            }
        }
    }

    let mut coeff = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            coeff[i * n + j] = row[i][m_total + j];
        }
    }

    TriNDkData {
        coeff,
        n,
        order: k,
        monomap: selected,
    }
}

/// Arbitrary-order Nedelec-I element on the reference triangle.
pub struct TriNDk {
    order: usize,
}

impl TriNDk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "TriNDk requires order ≥ 1");
        TriNDk { order: p }
    }
}

impl VectorReferenceElement for TriNDk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.order * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;

        // Special case k=1: Whitney 1-forms (matches TriND1)
        if k == 1 {
            let x = xi[0];
            let y = xi[1];
            // Φ₀ = w_{01} = (1−η, ξ)
            values[0] = 1.0 - y;
            values[1] = x;
            // Φ₁ = w_{12} = (−η, ξ)
            values[2] = -y;
            values[3] = x;
            // Φ₂ = w_{02} = (η, 1−ξ)
            values[4] = y;
            values[5] = 1.0 - x;
            return;
        }

        let d = tri_data(k);
        let x = xi[0];
        let y = xi[1];
        let n = d.n;
        let m_total = (k + 1) * (k + 2);

        let mut mono_vals = vec![0.0_f64; m_total * 2];
        let mut idx = 0usize;
        for deg in 0..=k {
            for a in 0..=deg {
                let b = deg - a;
                let v = x.powi(a as i32) * y.powi(b as i32);
                mono_vals[idx * 2] = v;
                mono_vals[idx * 2 + 1] = 0.0;
                idx += 1;
                mono_vals[idx * 2] = 0.0;
                mono_vals[idx * 2 + 1] = v;
                idx += 1;
            }
        }

        for i in 0..n {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for (ji, &sel) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                vx += c * mono_vals[sel * 2];
                vy += c * mono_vals[sel * 2 + 1];
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;

        // Special case k=1: constant curl [2, 2, -2]
        if k == 1 {
            curl_vals[0] = 2.0;
            curl_vals[1] = 2.0;
            curl_vals[2] = -2.0;
            return;
        }

        let d = tri_data(k);
        let x = xi[0];
        let y = xi[1];
        let n = d.n;

        let mut curl_mono = vec![0.0_f64; d.monomap.len()];
        for (ji, &sel) in d.monomap.iter().enumerate() {
            let mut rem = sel;
            let mut deg = 0usize;
            loop {
                let n_at_deg = 2 * (deg + 1);
                if rem < n_at_deg {
                    break;
                }
                rem -= n_at_deg;
                deg += 1;
            }
            let comp = rem % 2;
            let inner = rem / 2;
            let a = inner;
            let b = deg - inner;

            if comp == 0 && b > 0 {
                curl_mono[ji] = -(b as f64) * x.powi(a as i32) * y.powi((b - 1) as i32);
            } else if comp == 1 && a > 0 {
                curl_mono[ji] = (a as f64) * x.powi((a - 1) as i32) * y.powi(b as i32);
            }
        }

        for i in 0..n {
            let mut s = 0.0;
            for ji in 0..d.monomap.len() {
                s += d.coeff[i * n + ji] * curl_mono[ji];
            }
            curl_vals[i] = s;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;
        let n = k * (k + 2);
        let mut coords = Vec::with_capacity(n);

        // Special case k=1: edge midpoints (matches TriND1)
        if k == 1 {
            coords.push(vec![0.5, 0.0]);
            coords.push(vec![0.5, 0.5]);
            coords.push(vec![0.0, 0.5]);
            return coords;
        }

        // Edge e0 (η=0): k points
        for p in 0..k {
            let t = (p + 1) as f64 / (k + 1) as f64;
            coords.push(vec![t, 0.0]);
        }
        // Edge e1 ((1-t,t)): k points
        for p in 0..k {
            let t = (p + 1) as f64 / (k + 1) as f64;
            coords.push(vec![1.0 - t, t]);
        }
        // Edge e2 (ξ=0): k points
        for p in 0..k {
            let t = (p + 1) as f64 / (k + 1) as f64;
            coords.push(vec![0.0, t]);
        }
        // Interior: k(k-1) DOFs at barycentric coords
        let remaining = n - coords.len();
        for _ in 0..remaining {
            coords.push(vec![1.0 / 3.0, 1.0 / 3.0]);
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd1_curl_constant() {
        let elem = TriNDk::new(1);
        let mut curl = vec![0.0; 3];
        let expected = [2.0, 2.0, -2.0];
        let qr = elem.quadrature(3);
        for pt in &qr.points {
            elem.eval_curl(pt, &mut curl);
            for (i, &c) in curl.iter().enumerate() {
                assert!(
                    (c - expected[i]).abs() < 1e-13,
                    "curl[{i}] = {c}, expected {}",
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn nd1_nodal_basis() {
        let elem = TriNDk::new(1);
        let tangents: [[f64; 2]; 3] = [
            [1.0, 0.0],
            [-1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()],
            [0.0, 1.0],
        ];
        let edge_len = [1.0_f64, 2f64.sqrt(), 1.0_f64];

        let mut vals = vec![0.0; 6];
        for (j, (mid, (t, l))) in elem
            .dof_coords()
            .iter()
            .zip(tangents.iter().zip(edge_len.iter()))
            .enumerate()
        {
            elem.eval_basis_vec(mid, &mut vals);
            for i in 0..3 {
                let dof = (vals[i * 2] * t[0] + vals[i * 2 + 1] * t[1]) * l;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }
}
