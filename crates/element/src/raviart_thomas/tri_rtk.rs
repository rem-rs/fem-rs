//! Arbitrary-order RT_k on reference triangle.
//!
//! Uses the same construction as MFEM `RT_TriangleElement`:
//! - Edge DOFs at Gauss-Legendre open points
//! - Piola forms for lowest order, polynomial expansion for higher order
//! - Interior DOFs for k >= 1

use crate::reference::VectorReferenceElement;
use std::sync::OnceLock;

struct TriRTkData {
    coeff: Vec<f64>,
    n: usize,
    /// Total number of monomials spanning the trial space: the `[P_k]²` block
    /// (two per `(a, b)`) plus the `x^a y^b·(x, y)` bubble block.  Exceeds
    /// `n` — the Gauss-Jordan pivot selection drops the redundant columns.
    mt: usize,
    /// For each basis function, the monomial it was pivoted on.
    monomap: Vec<usize>,
}

fn tri_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 2).fold(1.0, |p, i| p * i as f64)
}

fn beta_int(a: usize, b: usize) -> f64 {
    (1..=a).fold(1.0, |p, i| p * i as f64) * (1..=b).fold(1.0, |p, i| p * i as f64)
        / (1..=a + b + 1).fold(1.0, |p, i| p * i as f64)
}

fn tri_data(k: usize) -> &'static TriRTkData {
    static CACHE: [OnceLock<TriRTkData>; 9] = [
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
    CACHE[k].get_or_init(|| {
        let n = (k + 1) * (k + 3);
        let mut mc = Vec::new();
        let mut mb = Vec::new();
        for deg in 0..=k {
            for a in 0..=deg {
                let b = deg - a;
                mc.push((a, b, 0));
                mc.push((a, b, 1));
                mb.push((a, b));
            }
        }
        let mt = mc.len() + mb.len();
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0usize;

        // Edge 0: hypotenuse (1-t,t), normal (1,1)/√2, length √2 → ∫(Φ_x+Φ_y)dt
        for p in 0..=k {
            for (j, &(a, b, _comp)) in mc.iter().enumerate() {
                v[di][j] = beta_int(a, b + p);
            }
            for (j, &(a, b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                v[di][jj] = beta_int(a + 1, b + p) + beta_int(a, b + p + 1);
            }
            di += 1;
        }

        // Edge 1: left edge (0,t), normal (-1,0)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                let val = if comp == 0 && a == 0 {
                    -1.0 / (b + p + 1) as f64
                } else {
                    0.0
                };
                v[di][j] = val;
            }
            for (j, &(_a, _b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                v[di][jj] = 0.0;
            }
            di += 1;
        }

        // Edge 2: bottom edge (t,0), normal (0,-1)
        for p in 0..=k {
            for (j, &(a, b, comp)) in mc.iter().enumerate() {
                let val = if comp == 1 && b == 0 {
                    -1.0 / (a + p + 1) as f64
                } else {
                    0.0
                };
                v[di][j] = val;
            }
            for (j, &(_a, _b)) in mb.iter().enumerate() {
                let jj = mc.len() + j;
                v[di][jj] = 0.0;
            }
            di += 1;
        }

        // Interior DOFs: ∫ Φ_x·x^ix y^iy dA and ∫ Φ_y·x^ix y^iy dA for ix+iy ≤ k-1
        if k >= 1 {
            for deg in 0..=(k - 1) {
                for ix in 0..=deg {
                    let iy = deg - ix;
                    for comp in 0..2 {
                        for (j, &(a, b, mc_comp)) in mc.iter().enumerate() {
                            let val = if mc_comp == comp {
                                tri_int(a + ix, b + iy)
                            } else {
                                0.0
                            };
                            v[di][j] = val;
                        }
                        for (j, &(a, b)) in mb.iter().enumerate() {
                            let jj = mc.len() + j;
                            let val = if comp == 0 {
                                tri_int(a + 1 + ix, b + iy)
                            } else {
                                tri_int(a + ix, b + 1 + iy)
                            };
                            v[di][jj] = val;
                        }
                        di += 1;
                    }
                }
            }
        }

        assert_eq!(di, n, "TriRTk({k}): DOF count {di} vs {n}");

        // Gauss-Jordan with column + row pivoting.  Mirrors `tet_rtk.rs`
        // (D33/D44): see the D33 note there for why both the column
        // permutation and the transposed coefficient extraction are needed.
        let mut cp: Vec<usize> = (0..mt).collect();
        let mut row = vec![vec![0.0; n + mt]; n];
        for i in 0..n {
            for j in 0..mt {
                row[i][j] = v[i][j];
            }
            row[i][mt + i] = 1.0;
        }
        let mut sel = Vec::new();
        for c in 0..n {
            let mut bc = c;
            let mut bv = 0.0_f64;
            for cc in c..mt {
                let mut mr = 0.0_f64;
                for rr in c..n {
                    mr = mr.max(row[rr][cc].abs());
                }
                if mr > bv {
                    bv = mr;
                    bc = cc;
                }
            }
            cp.swap(c, bc);
            for rr in 0..n {
                row[rr].swap(c, bc);
            }
            // Row pivoting: the column with the largest residual entry need
            // not have a usable entry in row `c` (tri_rtk previously silently
            // `continue`d, leaving the row unnormalised — a corrupted basis
            // instead of a loud failure).
            let mut pr = c;
            let mut pv = row[c][c].abs();
            for rr in c + 1..n {
                if row[rr][c].abs() > pv {
                    pv = row[rr][c].abs();
                    pr = rr;
                }
            }
            row.swap(c, pr);
            let pivot = row[c][c];
            assert!(pivot.abs() > 1e-14, "TriRTk({k}): singular at col {c}");
            let ip = 1.0 / pivot;
            for j in c..(n + mt) {
                row[c][j] *= ip;
            }
            for r in 0..n {
                if r != c {
                    let factor = row[r][c];
                    for j in c..(n + mt) {
                        row[r][j] -= factor * row[c][j];
                    }
                }
            }
            // `sel` records cp[c]: the monomial pivot column c was drawn from.
            sel.push(cp[c]);
        }

        // D44/D33: R · V · P = [I | X], so the dual basis is
        // Φ_j = Σ_a R[a][j]·mono_{cp[a]} — the coefficient matrix is the
        // TRANSPOSE of the reduced right block, paired with the permuted
        // monomials `cp`.  (The previous `row[i][mt + j]` dropped the column
        // permutation and transposition, so the basis was not dual to the
        // face-flux functionals.)
        let mut coeff = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                coeff[i * n + j] = row[j][mt + i];
            }
        }

        TriRTkData { coeff, n, mt, monomap: sel }
    })
}

/// Arbitrary-order Raviart-Thomas RT_k H(div) element on the reference triangle.
pub struct TriRTk {
    order: usize,
}

impl TriRTk {
    pub fn new(p: usize) -> Self {
        TriRTk { order: p }
    }
}

impl VectorReferenceElement for TriRTk {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * (self.order + 3)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let x = xi[0];
        let y = xi[1];

        // Special case k=0: classical Piola form (matches TriRT0)
        if k == 0 {
            // Φ₀ = (ξ, η)
            values[0] = x;
            values[1] = y;
            // Φ₁ = (ξ−1, η)
            values[2] = x - 1.0;
            values[3] = y;
            // Φ₂ = (ξ, η−1)
            values[4] = x;
            values[5] = y - 1.0;
            return;
        }

        let d = tri_data(k);
        let n = d.n;
        // Monomial value table, in the exact enumeration order of
        // `tri_data`: the `[P_k]²` monomials `x^a y^b e_c` (two per `(a, b)`,
        // component 0 then 1) followed by the "bubble" monomials
        // `x^a y^b·(x, y)` (one per `(a, b)`).  Omitting the bubble block
        // (as the pre-D44 code did, sizing the table as `(k+1)(k+2)` entries)
        // made `mono_vals[monomap[·] * 2]` read out of bounds whenever the
        // pivot selection picked a bubble column — a panic for every `k ≥ 1`.
        let mt = d.mt;
        let mut mono_vals = vec![0.0_f64; mt * 2];
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
        for deg in 0..=k {
            for a in 0..=deg {
                let b = deg - a;
                let v = x.powi(a as i32) * y.powi(b as i32);
                mono_vals[idx * 2] = x * v;
                mono_vals[idx * 2 + 1] = y * v;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, mt, "TriRTk({k}): monomial table mismatch");

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

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;

        // Special case k=0: div = 2 for all basis functions
        if k == 0 {
            for v in div_vals.iter_mut() {
                *v = 2.0;
            }
            return;
        }

        let d = tri_data(k);
        let x = xi[0];
        let y = xi[1];
        let n = d.n;

        // The divergence table follows the same monomial enumeration as
        // `eval_basis_vec`: ∂_x/∂_y of each `[P_k]²` monomial, then
        // ∇·(x^a y^b·(x, y)) = (a+b+2)·x^a y^b for the bubble block.
        let mut dm = vec![0.0_f64; d.mt];
        let mut idx = 0usize;
        for deg in 0..=k {
            for a in 0..=deg {
                let b = deg - a;
                let xp = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
                let yp = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
                dm[idx] = (a as f64) * xp * y.powi(b as i32);
                idx += 1;
                dm[idx] = (b as f64) * x.powi(a as i32) * yp;
                idx += 1;
            }
        }
        for deg in 0..=k {
            for a in 0..=deg {
                let b = deg - a;
                let v = x.powi(a as i32) * y.powi(b as i32);
                dm[idx] = (a + b + 2) as f64 * v;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, d.mt, "TriRTk({k}): divergence table mismatch");

        for i in 0..n {
            let mut s = 0.0;
            for (ji, &sel) in d.monomap.iter().enumerate() {
                s += d.coeff[i * n + ji] * dm[sel];
            }
            div_vals[i] = s;
        }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;

        // Special case k=0: edge midpoints (matches TriRT0)
        if k == 0 {
            return vec![
                vec![0.5, 0.5],
                vec![0.0, 0.5],
                vec![0.5, 0.0],
            ];
        }

        let n = (k + 1) * (k + 3);
        let mut c = Vec::with_capacity(n);
        for p in 0..=k {
            let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![1.0 - t, t]);
        }
        for p in 0..=k {
            let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![0.0, t]);
        }
        for p in 0..=k {
            let t = (p + 1) as f64 / (k + 2) as f64;
            c.push(vec![t, 0.0]);
        }
        let remaining = n - c.len();
        for _ in 0..remaining {
            c.push(vec![1.0 / 3.0, 1.0 / 3.0]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt0_div_constant() {
        let elem = TriRTk::new(0);
        let mut div = vec![0.0; 3];
        for pt in &elem.quadrature(4).points {
            elem.eval_div(pt, &mut div);
            for (i, &d) in div.iter().enumerate() {
                assert!((d - 2.0).abs() < 1e-13, "div[{i}] = {d}");
            }
        }
    }

    #[test]
    fn rt0_nodal_basis() {
        let elem = TriRTk::new(0);
        let faces: [([f64; 2], f64); 3] = [
            ([1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt()], 2f64.sqrt()),
            ([-1.0, 0.0], 1.0),
            ([0.0, -1.0], 1.0),
        ];

        let mids = elem.dof_coords();
        let mut vals = vec![0.0; 6];
        for (j, (normal, len)) in faces.iter().enumerate() {
            elem.eval_basis_vec(&mids[j], &mut vals);
            for i in 0..3 {
                let dof = (vals[i * 2] * normal[0] + vals[i * 2 + 1] * normal[1]) * len;
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dof - expected).abs() < 1e-12,
                    "DOF_{j}(Phi_{i}) = {dof}, expected {expected}"
                );
            }
        }
    }

    /// D44 regression: `eval_basis_vec` must evaluate **all** `k ≥ 1` orders
    /// without panicking.  The monomial table used to omit the `x^a y^b·(x,y)`
    /// bubble block, so `mono_vals[monomap[·]*2]` indexed out of bounds as
    /// soon as the pivot selection picked a bubble column (dpg_poisson_2d
    /// `-tri -o 1` panicked at tri_rtk.rs:257).
    #[test]
    fn eval_all_orders_is_in_bounds() {
        for k in 0..=4usize {
            let e = TriRTk::new(k);
            let n = e.n_dofs();
            let mut vals = vec![0.0f64; n * 2];
            let mut div = vec![0.0f64; n];
            for pt in &e.quadrature(4).points {
                e.eval_basis_vec(pt, &mut vals);
                e.eval_div(pt, &mut div);
                for &v in vals.iter().chain(div.iter()) {
                    assert!(v.is_finite(), "k={k}: non-finite basis value at {pt:?}");
                }
            }
        }
    }

    /// D44 regression (same invariant as `tet_rtk`'s D33 test): the
    /// constructed basis must be *dual* to the moment functionals used to
    /// build it, `D_i(Φ_j) = δ_ij`.  Before the fix the transposed
    /// coefficient extraction paired the basis functions with the wrong
    /// monomials whenever a column pivot kicked in (`k ≥ 1`), and `k ≥ 1`
    /// additionally panicked.  The functionals below are the analytic moments
    /// of `tri_data` evaluated with independent quadrature:
    ///   edges  `∫ (Φ·n_f) t^p ds` with `n_0 = (1,1)/√2·(√2 ds)`, and the
    ///   interior `∫_T Φ_c x^a y^b dA`, `a+b ≤ k-1`.
    #[test]
    fn basis_dual_to_moment_functionals() {
        for k in 0..=3usize {
            let e = TriRTk::new(k);
            let n = e.n_dofs();
            let qs = crate::quadrature::seg_rule_arbitrary((2 * k + 1) as u8);
            let qt = crate::quadrature::tri_rule_arbitrary(((3 * k) as u8).max(2));

            let mut dual = vec![0.0f64; n * n];
            let mut phi = vec![0.0f64; n * 2];
            let mut row = 0usize;

            // Edge 0: hypotenuse x+y=1, outward normal (1,1)/√2, ds = √2 dt →
            // ∫₀¹ (Φ_x+Φ_y)(1−t, t) t^p dt.
            for p in 0..=k {
                for (t, w) in qs.points.iter().zip(qs.weights.iter()) {
                    let (t, w) = (t[0], *w);
                    e.eval_basis_vec(&[1.0 - t, t], &mut phi);
                    for j in 0..n {
                        dual[row * n + j] +=
                            w * (phi[j * 2] + phi[j * 2 + 1]) * t.powi(p as i32);
                    }
                }
                row += 1;
            }
            // Edge 1: x=0, outward normal (−1,0), length 1 → ∫₀¹ (−Φ_x)(0,t) t^p dt.
            for p in 0..=k {
                for (t, w) in qs.points.iter().zip(qs.weights.iter()) {
                    let (t, w) = (t[0], *w);
                    e.eval_basis_vec(&[0.0, t], &mut phi);
                    for j in 0..n {
                        dual[row * n + j] -= w * phi[j * 2] * t.powi(p as i32);
                    }
                }
                row += 1;
            }
            // Edge 2: y=0, outward normal (0,−1), length 1 → ∫₀¹ (−Φ_y)(t,0) t^p dt.
            for p in 0..=k {
                for (t, w) in qs.points.iter().zip(qs.weights.iter()) {
                    let (t, w) = (t[0], *w);
                    e.eval_basis_vec(&[t, 0.0], &mut phi);
                    for j in 0..n {
                        dual[row * n + j] -= w * phi[j * 2 + 1] * t.powi(p as i32);
                    }
                }
                row += 1;
            }
            // Interior monomial moments ∫_T Φ_c x^a y^b dA, deg ≤ k−1.
            if k >= 1 {
                for dg in 0..=(k - 1) {
                    for a in 0..=dg {
                        let b = dg - a;
                        for comp in 0..2 {
                            for (qp, w) in qt.points.iter().zip(qt.weights.iter()) {
                                e.eval_basis_vec(qp, &mut phi);
                                let mono = qp[0].powi(a as i32) * qp[1].powi(b as i32);
                                for j in 0..n {
                                    dual[row * n + j] += w * phi[j * 2 + comp] * mono;
                                }
                            }
                            row += 1;
                        }
                    }
                }
            }
            assert_eq!(row, n, "k={k}");

            for i in 0..n {
                for j in 0..n {
                    let exp = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dual[i * n + j] - exp).abs() < 1e-9,
                        "k={k}: D_{i}(phi_{j}) = {}, expected {exp}",
                        dual[i * n + j]
                    );
                }
            }
        }
    }
}
