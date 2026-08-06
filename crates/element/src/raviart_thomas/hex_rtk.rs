//! Arbitrary-order RT_k on reference hexahedron [-1,1]³ via Vandermonde.

use crate::reference::VectorReferenceElement;
use std::sync::OnceLock;

struct HexRTkData {
    coeff: Vec<f64>,
    monomap: Vec<usize>,
    n: usize,
}

fn i1(p: usize) -> f64 {
    if p % 2 == 1 {
        0.0
    } else {
        2.0 / (p + 1) as f64
    }
}
fn i2(a: usize, b: usize) -> f64 {
    i1(a) * i1(b)
}
fn i3(a: usize, b: usize, c: usize) -> f64 {
    i1(a) * i1(b) * i1(c)
}

fn hex_data(k: usize) -> &'static HexRTkData {
    static CACHE: [OnceLock<HexRTkData>; 9] = [
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
        let n = 3 * (k + 1) * (k + 1) * (k + 2);
        let nx = (k + 2) * (k + 1) * (k + 1);
        let ny = (k + 1) * (k + 2) * (k + 1);
        let nz = (k + 1) * (k + 1) * (k + 2);
        let mt = nx + ny + nz;
        let mut v = vec![vec![0.0; mt]; n];
        let mut di = 0;

        // Face -x: ∫ -Φ_x(-1,y,z)·y^a z^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    let am = j / ((k + 1) * (k + 1));
                    let r = j % ((k + 1) * (k + 1));
                    v[di][j] = -(-1.0_f64).powi(am as i32) * i2(r / (k + 1) + a, r % (k + 1) + b);
                }
                for j in nx..mt {
                    v[di][j] = 0.0;
                }
                di += 1;
            }
        }
        // Face +x: ∫ Φ_x(1,y,z)·y^a z^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    let r = j % ((k + 1) * (k + 1));
                    v[di][j] = 1.0 * i2(r / (k + 1) + a, r % (k + 1) + b);
                }
                for j in nx..mt {
                    v[di][j] = 0.0;
                }
                di += 1;
            }
        }
        // Face -y: ∫ -Φ_y(x,-1,z)·x^a z^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    v[di][j] = 0.0;
                }
                for j in 0..ny {
                    let am = j / ((k + 2) * (k + 1));
                    let r = j % ((k + 2) * (k + 1));
                    v[di][nx + j] =
                        -(-1.0_f64).powi((r / (k + 1)) as i32) * i2(am + a, r % (k + 1) + b);
                }
                for j in nx + ny..mt {
                    v[di][j] = 0.0;
                }
                di += 1;
            }
        }
        // Face +y: ∫ Φ_y(x,1,z)·x^a z^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    v[di][j] = 0.0;
                }
                for j in 0..ny {
                    let am = j / ((k + 2) * (k + 1));
                    let r = j % ((k + 2) * (k + 1));
                    v[di][nx + j] = 1.0 * i2(am + a, r % (k + 1) + b);
                }
                for j in nx + ny..mt {
                    v[di][j] = 0.0;
                }
                di += 1;
            }
        }
        // Face -z: ∫ -Φ_z(x,y,-1)·x^a y^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    v[di][j] = 0.0;
                }
                for j in nx..nx + ny {
                    v[di][j] = 0.0;
                }
                for j in 0..nz {
                    let am = j / ((k + 1) * (k + 2));
                    let r = j % ((k + 1) * (k + 2));
                    v[di][nx + ny + j] =
                        -(-1.0_f64).powi((r % (k + 2)) as i32) * i2(am + a, r / (k + 2) + b);
                }
                di += 1;
            }
        }
        // Face +z: ∫ Φ_z(x,y,1)·x^a y^b dA
        for a in 0..=k {
            for b in 0..=k {
                for j in 0..nx {
                    v[di][j] = 0.0;
                }
                for j in nx..nx + ny {
                    v[di][j] = 0.0;
                }
                for j in 0..nz {
                    let am = j / ((k + 1) * (k + 2));
                    let r = j % ((k + 1) * (k + 2));
                    v[di][nx + ny + j] = 1.0 * i2(am + a, r / (k + 2) + b);
                }
                di += 1;
            }
        }
        // Interior
        if k >= 1 {
            for a in 0..k {
                for b in 0..=k {
                    for c in 0..=k {
                        for j in 0..nx {
                            let r = j % ((k + 1) * (k + 1));
                            v[di][j] = i3(
                                j / ((k + 1) * (k + 1)) + a,
                                r / (k + 1) + b,
                                r % (k + 1) + c,
                            );
                        }
                        for j in nx..mt {
                            v[di][j] = 0.0;
                        }
                        di += 1;
                    }
                }
            }
            for a in 0..=k {
                for b in 0..k {
                    for c in 0..=k {
                        for j in 0..nx {
                            v[di][j] = 0.0;
                        }
                        for j in 0..ny {
                            let am = j / ((k + 2) * (k + 1));
                            let r = j % ((k + 2) * (k + 1));
                            v[di][nx + j] = i3(am + a, r / (k + 1) + b, r % (k + 1) + c);
                        }
                        for j in nx + ny..mt {
                            v[di][j] = 0.0;
                        }
                        di += 1;
                    }
                }
            }
            for a in 0..=k {
                for b in 0..=k {
                    for c in 0..k {
                        for j in 0..nx {
                            v[di][j] = 0.0;
                        }
                        for j in nx..nx + ny {
                            v[di][j] = 0.0;
                        }
                        for j in 0..nz {
                            let am = j / ((k + 1) * (k + 2));
                            let r = j % ((k + 1) * (k + 2));
                            v[di][nx + ny + j] = i3(am + a, r / (k + 2) + b, r % (k + 2) + c);
                        }
                        di += 1;
                    }
                }
            }
        }
        assert_eq!(di, n);

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
            let mut pr = c;
            let mut pv = row[c][c].abs();
            for rr in c + 1..n {
                if row[rr][c].abs() > pv {
                    pv = row[rr][c].abs();
                    pr = rr;
                }
            }
            row.swap(c, pr);
            let piv = row[c][c];
            assert!(piv.abs() > 1e-12, "HexRTk({k}) singular at col {c}");
            let ip = 1.0 / piv;
            for j in c..n + mt {
                row[c][j] *= ip;
            }
            for rr in 0..n {
                if rr != c {
                    let f = row[rr][c];
                    for j in c..n + mt {
                        row[rr][j] -= f * row[c][j];
                    }
                }
            }
            sel.push(c);
        }
        let mut coeff = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                coeff[i * n + j] = row[i][mt + sel[j]];
            }
        }
        HexRTkData {
            coeff,
            monomap: sel,
            n,
        }
    })
}

pub struct HexRTk {
    order: usize,
}
impl HexRTk {
    pub fn new(p: usize) -> Self {
        HexRTk { order: p }
    }
}

impl VectorReferenceElement for HexRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * (self.order + 1) * (self.order + 1) * (self.order + 2)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let k = self.order;
        let d = hex_data(k);
        let n = d.n;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let nx = (k + 2) * (k + 1) * (k + 1);
        let ny = (k + 1) * (k + 2) * (k + 1);
        let nz = (k + 1) * (k + 1) * (k + 2);
        let mt = nx + ny + nz;
        let mut mv = vec![0.0; mt * 3];
        let mut idx = 0;
        for a in 0..=k + 1 {
            for b in 0..=k {
                for c in 0..=k {
                    let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
                    mv[idx * 3] = v;
                    mv[idx * 3 + 1] = 0.0;
                    mv[idx * 3 + 2] = 0.0;
                    idx += 1;
                }
            }
        }
        for a in 0..=k {
            for b in 0..=k + 1 {
                for c in 0..=k {
                    let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
                    mv[idx * 3] = 0.0;
                    mv[idx * 3 + 1] = v;
                    mv[idx * 3 + 2] = 0.0;
                    idx += 1;
                }
            }
        }
        for a in 0..=k {
            for b in 0..=k {
                for c in 0..=k + 1 {
                    let v = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32);
                    mv[idx * 3] = 0.0;
                    mv[idx * 3 + 1] = 0.0;
                    mv[idx * 3 + 2] = v;
                    idx += 1;
                }
            }
        }
        for i in 0..n {
            let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
            for (ji, &s) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                vx += c * mv[s * 3];
                vy += c * mv[s * 3 + 1];
                vz += c * mv[s * 3 + 2];
            }
            values[i * 3] = vx;
            values[i * 3 + 1] = vy;
            values[i * 3 + 2] = vz;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let k = self.order;
        let d = hex_data(k);
        let n = d.n;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let nx = (k + 2) * (k + 1) * (k + 1);
        let ny = (k + 1) * (k + 2) * (k + 1);
        let nz = (k + 1) * (k + 1) * (k + 2);
        let mt = nx + ny + nz;
        let mut dm = vec![0.0; mt];
        let mut idx = 0;
        for a in 0..=k + 1 {
            for b in 0..=k {
                for c in 0..=k {
                    let xm1 = if a > 0 { x.powi((a - 1) as i32) } else { 0.0 };
                    dm[idx] = (a as f64) * xm1 * y.powi(b as i32) * z.powi(c as i32);
                    idx += 1;
                }
            }
        }
        for a in 0..=k {
            for b in 0..=k + 1 {
                for c in 0..=k {
                    let ym1 = if b > 0 { y.powi((b - 1) as i32) } else { 0.0 };
                    dm[idx] = (b as f64) * x.powi(a as i32) * ym1 * z.powi(c as i32);
                    idx += 1;
                }
            }
        }
        for a in 0..=k {
            for b in 0..=k {
                for c in 0..=k + 1 {
                    let zm1 = if c > 0 { z.powi((c - 1) as i32) } else { 0.0 };
                    dm[idx] = (c as f64) * x.powi(a as i32) * y.powi(b as i32) * zm1;
                    idx += 1;
                }
            }
        }
        for i in 0..n {
            let mut s = 0.0;
            for (ji, &sel) in d.monomap.iter().enumerate() {
                s += d.coeff[i * n + ji] * dm[sel];
            }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let k = self.order;
        let d = hex_data(k);
        let n = d.n;
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let nx = (k + 2) * (k + 1) * (k + 1);
        let ny = (k + 1) * (k + 2) * (k + 1);
        let nz = (k + 1) * (k + 1) * (k + 2);
        let mt = nx + ny + nz;
        let mut cm = vec![0.0; mt * 3];
        let mut idx = 0;
        // X-directed: m = (x^a y^b z^c, 0, 0), curl = (0, -c x^a y^b z^(c-1), b x^a y^(b-1) z^c)
        for a in 0..=k + 1 {
            for b in 0..=k {
                for c in 0..=k {
                    cm[idx * 3] = 0.0;
                    cm[idx * 3 + 1] = if c > 0 {
                        -(c as f64) * x.powi(a as i32) * y.powi(b as i32) * z.powi((c - 1) as i32)
                    } else {
                        0.0
                    };
                    cm[idx * 3 + 2] = if b > 0 {
                        (b as f64) * x.powi(a as i32) * y.powi((b - 1) as i32) * z.powi(c as i32)
                    } else {
                        0.0
                    };
                    idx += 1;
                }
            }
        }
        // Y-directed: m = (0, x^a y^b z^c, 0), curl = (c x^a y^b z^(c-1), 0, -a x^(a-1) y^b z^c)
        for a in 0..=k {
            for b in 0..=k + 1 {
                for c in 0..=k {
                    cm[idx * 3] = if c > 0 {
                        (c as f64) * x.powi(a as i32) * y.powi(b as i32) * z.powi((c - 1) as i32)
                    } else {
                        0.0
                    };
                    cm[idx * 3 + 1] = 0.0;
                    cm[idx * 3 + 2] = if a > 0 {
                        -(a as f64) * x.powi((a - 1) as i32) * y.powi(b as i32) * z.powi(c as i32)
                    } else {
                        0.0
                    };
                    idx += 1;
                }
            }
        }
        // Z-directed: m = (0, 0, x^a y^b z^c), curl = (-b x^a y^(b-1) z^c, a x^(a-1) y^b z^c, 0)
        for a in 0..=k {
            for b in 0..=k {
                for c in 0..=k + 1 {
                    cm[idx * 3] = if b > 0 {
                        -(b as f64) * x.powi(a as i32) * y.powi((b - 1) as i32) * z.powi(c as i32)
                    } else {
                        0.0
                    };
                    cm[idx * 3 + 1] = if a > 0 {
                        (a as f64) * x.powi((a - 1) as i32) * y.powi(b as i32) * z.powi(c as i32)
                    } else {
                        0.0
                    };
                    cm[idx * 3 + 2] = 0.0;
                    idx += 1;
                }
            }
        }
        for i in 0..n {
            let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
            for (ji, &s) in d.monomap.iter().enumerate() {
                let c = d.coeff[i * n + ji];
                cx += c * cm[s * 3];
                cy += c * cm[s * 3 + 1];
                cz += c * cm[s * 3 + 2];
            }
            curl_vals[i * 3] = cx;
            curl_vals[i * 3 + 1] = cy;
            curl_vals[i * 3 + 2] = cz;
        }
    }

    fn quadrature(&self, order: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let k = self.order;
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        // Face DOFs: (k+1) per face × 6 faces = 6(k+1)
        let _gl = |i: usize| -> f64 { -1.0 + 2.0 * i as f64 / k as f64 };
        for &(sx, sy, sz, _ax, _ay, _az) in &[
            (-1.0, 0.0, 0.0, 1, 2, 3), // -x
            (1.0, 0.0, 0.0, 1, 2, 3),  // +x
            (0.0, -1.0, 0.0, 0, 1, 2), // -y
            (0.0, 1.0, 0.0, 0, 1, 2),  // +y
            (0.0, 0.0, -1.0, 0, 1, 2), // -z
            (0.0, 0.0, 1.0, 0, 1, 2),  // +z
        ] {
            for _ in 0..=k {
                c.push(vec![sx, sy, sz]);
            }
        }
        // Interior DOFs (k >= 1): 3k(k+1)
        let _step = 1.0 / (k + 2) as f64;
        for _ in 0..(n - 6 * (k + 1)) {
            c.push(vec![0.0, 0.0, 0.0]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn n_dofs() {
        assert_eq!(HexRTk::new(0).n_dofs(), 6);
        assert_eq!(HexRTk::new(1).n_dofs(), 36);
        assert_eq!(HexRTk::new(2).n_dofs(), 108);
    }
    #[test]
    fn coeff_non_singular() {
        for k in 1..=2 {
            let d = hex_data(k);
            let mut ok = true;
            for i in 0..d.n {
                let mut s = 0.0;
                for (ji, _) in d.monomap.iter().enumerate() {
                    s += d.coeff[i * d.n + ji].abs();
                }
                if s < 1e-10 {
                    ok = false;
                }
            }
            assert!(ok, "k={k}");
        }
    }
    #[test]
    fn finite() {
        for k in 0..=2 {
            let e = HexRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            let mut d = vec![0.0; n];
            for p in &[(0.0, 0.0, 0.0), (0.3, -0.5, 0.7), (-0.2, 0.4, -0.6)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                e.eval_div(&[p.0, p.1, p.2], &mut d);
                for &val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite());
                }
            }
        }
    }
}
