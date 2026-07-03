//! BDM_k reference element on the reference hexahedron [0,1]³.
//!
//! Brezzi-Douglas-Durán-Fortin (BDF) construction (Brezzi et al. 1987 §III):
//! DOFs = 6 × (k+1)(k+2)/2 face normal moments + k(k-1)(k+1)/2 interior L² moments.
//!
//! Total dim = 3(k+1)(k+2) + k(k-1)(k+1)/2.

use crate::reference::VectorReferenceElement;

// ─── Monomial helpers ────────────────────────────────────────────────────────

#[derive(Clone)]
struct Mono { comp: u8, a: usize, b: usize, c: usize }

fn hex_monos(max_deg: usize) -> Vec<Mono> {
    let mut m = Vec::new();
    for dg in 0..=max_deg {
        for a in 0..=dg { for b in 0..=(dg - a) { let c = dg - a - b;
            for comp in 0..3u8 { m.push(Mono { comp, a, b, c }); }
        }}
    }
    m
}

fn eval_mono(m: &Mono, x: f64, y: f64, z: f64) -> f64 {
    x.powi(m.a as i32) * y.powi(m.b as i32) * z.powi(m.c as i32)
}

// ─── Quadrature ──────────────────────────────────────────────────────────────

fn gauss_2d(k: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let n = (k + 2).max(2);
    let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(n);
    let mut pts = Vec::new(); let mut wts = Vec::new();
    for i in 0..n { for j in 0..n { pts.push([x1d[i], x1d[j]]); wts.push(w1d[i] * w1d[j]); }}
    (pts, wts)
}

/// 3D Gauss-Legendre tensor product on the reference cube [0,1]³.
fn gauss_3d(k: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    let n = (k + 2).max(2);
    let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(n);
    let mut pts = Vec::new(); let mut wts = Vec::new();
    for i in 0..n { for j in 0..n { for k2 in 0..n {
        pts.push([x1d[i], x1d[j], x1d[k2]]);
        wts.push(w1d[i] * w1d[j] * w1d[k2]);
    }}}
    (pts, wts)
}

// ─── HexBDMk dimension ──────────────────────────────────────────────────────

fn hex_bdmk_dim(k: usize) -> usize {
    let face = 6 * (k + 1) * (k + 2) / 2; // 6 faces × (k+1)(k+2)/2 moments
    let interior = k * k.saturating_sub(1) * (k + 1) / 2;
    face + interior
}

// ─── Vandermonde construction ───────────────────────────────────────────────

fn solve_normal_eq(v: &[Vec<f64>], n: usize, m: usize) -> Vec<f64> {
    let mut vvt = vec![vec![0.0_f64; n]; n];
    for i in 0..n { for j in 0..n {
        let mut s = 0.0;
        for col in 0..m { s += v[i][col] * v[j][col]; }
        vvt[i][j] = s;
    }}
    let mut a = vvt.clone();
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n { inv[i][i] = 1.0; }
    for c in 0..n {
        let mut best = c; let mut bv = a[c][c].abs();
        for r in (c+1)..n { if a[r][c].abs() > bv { bv = a[r][c].abs(); best = r; } }
        if bv < 1e-30 { continue; }
        a.swap(c, best); inv.swap(c, best);
        let ip = 1.0 / a[c][c];
        for j in 0..n { a[c][j] *= ip; inv[c][j] *= ip; }
        for r in 0..n { if r == c { continue; } let f = a[r][c];
            for j in 0..n { a[r][j] -= f * a[c][j]; inv[r][j] -= f * inv[c][j]; }
        }
    }
    let mut coeff = vec![0.0_f64; n * m];
    for i in 0..n { for j in 0..m {
        let mut s = 0.0;
        for k in 0..n { s += v[k][j] * inv[k][i]; }
        coeff[i * m + j] = s;
    }}
    coeff
}

fn build_hex_bdmk(k: usize) -> (Vec<f64>, usize) {
    let n = hex_bdmk_dim(k);
    // Use monomials up to degree k+1 to span [P_{k+1}]³ plus bubble space
    let monos = hex_monos(k + 1);
    let m = monos.len();

    // Also generate bubble functions: (1-x²)·P_{k-1}, (1-y²)·P_{k-1}, (1-z²)·P_{k-1}
    // by adding a few extra columns for each component/bubble type.
    // For simplicity, rely on k+1 degree monomials to span the space.

    let mut vand = vec![vec![0.0_f64; m]; n];
    let mut row = 0;

    // Face quadrature data
    let (g2_pts, g2_wts) = gauss_2d(k);

    // Face definitions for hex [0,1]³
    // Each face is a unit square [0,1]² with outward normal.
    let faces: [(usize, usize, usize, f64, f64, [f64; 3]); 6] = [
        (0, 1, 2, 0.0, 0.0, [-1.0, 0.0, 0.0]),  // x=0
        (0, 1, 2, 1.0, 0.0, [1.0, 0.0, 0.0]),   // x=1
        (1, 0, 2, 0.0, 0.0, [0.0, -1.0, 0.0]),  // y=0
        (1, 0, 2, 1.0, 0.0, [0.0, 1.0, 0.0]),   // y=1
        (2, 0, 1, 0.0, 0.0, [0.0, 0.0, -1.0]),  // z=0
        (2, 0, 1, 1.0, 0.0, [0.0, 0.0, 1.0]),   // z=1
    ];

    let mut mom_pairs = Vec::new();
    for p in 0..=k { for q in 0..=k { if p + q <= k { mom_pairs.push((p, q)); } }}

    for &(n_axis, u_axis, v_axis, u_val, _v_val, norm) in &faces {
        let nn = [norm[0], norm[1], norm[2]];
        for &(p, q) in &mom_pairs {
            for j in 0..m {
                let mut sum = 0.0;
                for (pt, &w) in g2_pts.iter().zip(g2_wts.iter()) {
                    let u = pt[0]; let v = pt[1];
                    let mut xyz = [0.0_f64; 3];
                    xyz[n_axis] = u_val;
                    xyz[u_axis] = u;
                    xyz[v_axis] = v;
                    let mv = eval_mono(&monos[j], xyz[0], xyz[1], xyz[2]);
                    let dot = match monos[j].comp {
                        0 => nn[0], 1 => nn[1], 2 => nn[2], _ => 0.0,
                    };
                    let poly = u.powi(p as i32) * v.powi(q as i32);
                    sum += w * dot * mv * poly;
                }
                vand[row][j] = sum;
            }
            row += 1;
        }
    }

    // Interior DOFs: L² moments of div(v) weighted by monomials.
    // The interior space for BDMk has dim = k(k-1)(k+1)/2.
    // We integrate div(M) · φ over [0,1]³ where φ ranges over (some of)
    // the P_{k-1} monomials.  For k=2 this gives 3 interior DOFs (using
    // weight functions {1, x, y}); for k=3, 12 DOFs; etc.
    if k >= 2 {
        let n_int = k * (k - 1) * (k + 1) / 2;
        let (g3_pts, g3_wts) = gauss_3d(k);
        // Build weight-function exponent list from first n_int P_{k-1} monomials
        let mut exponents: Vec<(usize, usize, usize)> = Vec::new();
        for deg in 0..=k {
            for i in 0..=deg { for j in 0..=(deg - i) {
                let ij = deg - i - j;
                exponents.push((i, j, ij));
                if exponents.len() >= n_int { break; }
            } if exponents.len() >= n_int { break; } }
            if exponents.len() >= n_int { break; }
        }
        for &(ei, ej, ek) in &exponents {
            for midx in 0..m {
                let mut sum = 0.0;
                for (pt, &w) in g3_pts.iter().zip(g3_wts.iter()) {
                    let div_mv = match monos[midx].comp {
                        0 => { let a = monos[midx].a as f64;
                               if a > 0.0 { a * pt[0].powi(monos[midx].a as i32 - 1)
                                                  * pt[1].powi(monos[midx].b as i32)
                                                  * pt[2].powi(monos[midx].c as i32) } else { 0.0 } }
                        1 => { let b = monos[midx].b as f64;
                               if b > 0.0 { b * pt[0].powi(monos[midx].a as i32)
                                                  * pt[1].powi(monos[midx].b as i32 - 1)
                                                  * pt[2].powi(monos[midx].c as i32) } else { 0.0 } }
                        2 => { let c = monos[midx].c as f64;
                               if c > 0.0 { c * pt[0].powi(monos[midx].a as i32)
                                                  * pt[1].powi(monos[midx].b as i32)
                                                  * pt[2].powi(monos[midx].c as i32 - 1) } else { 0.0 } }
                        _ => 0.0,
                    };
                    let phi = pt[0].powi(ei as i32) * pt[1].powi(ej as i32) * pt[2].powi(ek as i32);
                    sum += w * div_mv * phi;
                }
                vand[row][midx] = sum;
            }
            row += 1;
        }
    }

    assert_eq!(row, n, "row {row} != dim {n} for k={k}");

    let coeff = solve_normal_eq(&vand, n, m);
    (coeff, m)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Brezzi-Douglas-Marini H(div) element on the reference hex — arbitrary order k.
pub struct HexBDMk { k: usize, coeff: Vec<f64>, n: usize, m: usize, monos: Vec<Mono> }

impl HexBDMk {
    pub fn new(order: usize) -> Self {
        assert!(order >= 1, "HexBDMk: order >= 1");
        let (coeff, m) = build_hex_bdmk(order);
        let n = hex_bdmk_dim(order);
        HexBDMk { k: order, coeff, n, m, monos: hex_monos(order + 1) }
    }
}

impl VectorReferenceElement for HexBDMk {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { self.k as u8 }
    fn n_dofs(&self) -> usize { self.n }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let mut mv = vec![0.0_f64; self.monos.len()];
        for (j, m) in self.monos.iter().enumerate() { mv[j] = eval_mono(m, xi[0], xi[1], xi[2]); }
        values.fill(0.0);
        for i in 0..self.n {
            for j in 0..self.m {
                if i * self.m + j < self.coeff.len() {
                    let c = self.coeff[i * self.m + j];
                    if c != 0.0 { values[i * 3 + self.monos[j].comp as usize] += c * mv[j]; }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], cv: &mut [f64]) {
        let h = 1e-6; let n3 = self.n * 3;
        let mut vp = vec![0.0; n3]; let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0]+h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0]-h, xi[1], xi[2]], &mut vm);
            let dfy_dx = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            let dfz_dx = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1]+h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1]-h, xi[2]], &mut vm);
            let dfx_dy = (vp[i*3]-vm[i*3])/(2.0*h);
            let dfz_dy = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]+h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]-h], &mut vm);
            let dfx_dz = (vp[i*3]-vm[i*3])/(2.0*h);
            let dfy_dz = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            cv[i*3] = dfz_dy - dfy_dz; cv[i*3+1] = dfx_dz - dfz_dx; cv[i*3+2] = dfy_dx - dfx_dy;
        }
    }

    fn eval_div(&self, xi: &[f64], dv: &mut [f64]) {
        let h = 1e-6; let n3 = self.n * 3;
        let mut vp = vec![0.0; n3]; let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0]+h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0]-h, xi[1], xi[2]], &mut vm);
            let dfx = (vp[i*3]-vm[i*3])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1]+h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1]-h, xi[2]], &mut vm);
            let dfy = (vp[i*3+1]-vm[i*3+1])/(2.0*h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]+h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2]-h], &mut vm);
            let dfz = (vp[i*3+2]-vm[i*3+2])/(2.0*h);
            dv[i] = dfx + dfy + dfz;
        }
    }

    fn quadrature(&self, order: u8) -> crate::QuadratureRule {
        let o = order.max(self.k as u8 + 1) as usize;
        let (x1d, w1d) = crate::quadrature::gauss_legendre_arbitrary(o);
        let nq = x1d.len();
        let mut pts = Vec::with_capacity(nq * nq * nq);
        let mut wts = Vec::with_capacity(nq * nq * nq);
        for i in 0..nq { for j in 0..nq { for k2 in 0..nq {
            pts.push(vec![x1d[i], x1d[j], x1d[k2]]);
            wts.push(w1d[i] * w1d[j] * w1d[k2]);
        }}}
        crate::QuadratureRule { points: pts, weights: wts }
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.k;
        let n_mom = (k + 1) * (k + 2) / 2;
        let mut c = Vec::new();
        for _ in 0..n_mom { c.push(vec![0.0, 0.3, 0.3]); }
        for _ in 0..n_mom { c.push(vec![1.0, 0.3, 0.3]); }
        for _ in 0..n_mom { c.push(vec![0.3, 0.0, 0.3]); }
        for _ in 0..n_mom { c.push(vec![0.3, 1.0, 0.3]); }
        for _ in 0..n_mom { c.push(vec![0.3, 0.3, 0.0]); }
        for _ in 0..n_mom { c.push(vec![0.3, 0.3, 1.0]); }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn hex_bdmk_k1_dim() { assert_eq!(hex_bdmk_dim(1), 18); }
    #[test] fn hex_bdmk_k2_dim() { assert_eq!(hex_bdmk_dim(2), 39); }
    #[test] fn hex_bdmk_k3_dim() { assert_eq!(hex_bdmk_dim(3), 72); }

    #[test] fn hex_bdmk_k1_basis_finite() {
        let e = HexBDMk::new(1); let mut v = vec![0.0; 54];
        for p in &e.quadrature(2).points { e.eval_basis_vec(p, &mut v); for x in &v { assert!(x.is_finite()); } }
    }

    #[test] fn hex_bdmk_k2_basis_finite() {
        let e = HexBDMk::new(2); let mut v = vec![0.0; 117];
        for p in &e.quadrature(2).points { e.eval_basis_vec(p, &mut v); for x in &v { assert!(x.is_finite()); } }
    }
}
