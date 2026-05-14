//! Raviart–Thomas RT2 element on the reference triangle `(0,0),(1,0),(0,1)`.
//!
//! Uses the same **15** primal vector fields as MFEM `RT_TriangleElement` for `p = 2`
//! (tensor product of quadratic 1-D **Chebyshev** bases via `z = 2t − 1` on each of
//! `(x, y, 1−x−y)` plus the standard RT bubble rows), then inverts the Vandermonde
//! matrix **`T`** from
//! MFEM’s construction: each DOF is **`Φ · n̂`** evaluated at an **open** Gauss–Legendre
//! node on an edge or at an interior MFEM node, with `n̂` drawn from the fixed
//! `RT_TriangleElement::nk` table (see MFEM `fem/fe/fe_rt.cpp`).
//!
//! # DOFs (15)
//! - Edges (hypotenuse, left, bottom as in [`super::tri_rt1::TriRT1`]): three point
//!   values of `Φ·n̂` at the degree-2 GL open nodes on each edge (`n̂` unnormalized,
//!   same convention as RT1).
//! - Interior: six point values following MFEM’s interior loop (three reference points,
//!   each with two normals `(0,−1)` and `(−1,0)`).

use std::sync::OnceLock;

use nalgebra::DMatrix;

use crate::quadrature::{gauss_legendre_01, tri_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

const N: usize = 15;
const C: f64 = 1.0 / 3.0;

static COEFF: OnceLock<[[f64; N]; N]> = OnceLock::new();

/// MFEM `Poly_1D::CalcChebyshev` on `[0,1]` (`z = 2x − 1`) through degree `p = 2`.
#[inline]
fn cheb_t_u_d(x: f64) -> ([f64; 3], [f64; 3]) {
    let z = 2.0 * x - 1.0;
    let u0 = 1.0;
    let u1 = z;
    let u2 = 2.0 * z * u1 - u0;
    let d0 = 0.0;
    let d1 = 2.0;
    let d2 = 2.0 * (z * d1 + 2.0 * u1);
    ([u0, u1, u2], [d0, d1, d2])
}

/// The 15 MFEM-style primal fields `u_j` at `(x,y)`.
fn eval_primitives(x: f64, y: f64, u: &mut [[f64; 2]; N]) {
    let l = 1.0 - x - y;
    let (sx, _dsx) = cheb_t_u_d(x);
    let (sy, _dsy) = cheb_t_u_d(y);
    let (sl, _dsl) = cheb_t_u_d(l);

    let mut o = 0usize;
    for j in 0..=2 {
        for i in 0..=(2 - j) {
            let k = 2 - i - j;
            let s = sx[i] * sy[j] * sl[k];
            u[o] = [s, 0.0];
            o += 1;
            u[o] = [0.0, s];
            o += 1;
        }
    }
    for i in 0..=2 {
        let j = 2 - i;
        let s = sx[i] * sy[j];
        u[o] = [(x - C) * s, (y - C) * s];
        o += 1;
    }
    debug_assert_eq!(o, N);
}

/// Divergence of each primal field.
fn eval_primitive_divs(x: f64, y: f64, div: &mut [f64; N]) {
    let l = 1.0 - x - y;
    let (sx, dsx) = cheb_t_u_d(x);
    let (sy, dsy) = cheb_t_u_d(y);
    let (sl, dsl) = cheb_t_u_d(l);

    let mut o = 0usize;
    for j in 0..=2 {
        for i in 0..=(2 - j) {
            let k = 2 - i - j;
            let _s = sx[i] * sy[j] * sl[k];
            let ds_dx = dsx[i] * sy[j] * sl[k] - sx[i] * sy[j] * dsl[k];
            let ds_dy = sx[i] * dsy[j] * sl[k] - sx[i] * sy[j] * dsl[k];
            div[o] = ds_dx;
            o += 1;
            div[o] = ds_dy;
            o += 1;
        }
    }
    for i in 0..=2 {
        let j = 2 - i;
        let s = sx[i] * sy[j];
        let ds_dx = dsx[i] * sy[j];
        let ds_dy = sx[i] * dsy[j];
        div[o] = 2.0 * s + (x - C) * ds_dx + (y - C) * ds_dy;
        o += 1;
    }
    debug_assert_eq!(o, N);
}

/// MFEM `RT_TriangleElement::nk`: pairs `(n_x, n_y)` for `dof2nk = 0,1,2`.
const NK: [[f64; 2]; 3] = [[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]];

/// `V[row][col] = DOF_row(primitive_col)` — MFEM-style nodal `Φ·n̂` at `OpenPoints`.
fn build_vandermonde() -> [[f64; N]; N] {
    let (bop, _) = gauss_legendre_01(3);
    assert_eq!(bop.len(), 3, "RT2 expects three open GL nodes per edge");
    let (iop, _) = gauss_legendre_01(2);
    assert_eq!(iop.len(), 2, "RT2 interior uses OpenPoints(p−1), p=2");

    let mut v = [[0.0f64; N]; N];
    let mut prim = [[0.0f64; 2]; N];
    let mut row = 0usize;

    // Face 0 — hypotenuse `(1−t, t)`, `dof2nk = 1` → `n = (1,1)`.
    for t in &bop {
        let x = 1.0 - t;
        let y = *t;
        let n = NK[1];
        eval_primitives(x, y, &mut prim);
        for col in 0..N {
            v[row][col] = prim[col][0] * n[0] + prim[col][1] * n[1];
        }
        row += 1;
    }
    // Face 1 — left `(0, t)`, `dof2nk = 2` → `n = (−1, 0)`.
    for t in &bop {
        let n = NK[2];
        eval_primitives(0.0, *t, &mut prim);
        for col in 0..N {
            v[row][col] = prim[col][0] * n[0] + prim[col][1] * n[1];
        }
        row += 1;
    }
    // Face 2 — bottom `(t, 0)`, `dof2nk = 0` → `n = (0, −1)`.
    for t in &bop {
        let n = NK[0];
        eval_primitives(*t, 0.0, &mut prim);
        for col in 0..N {
            v[row][col] = prim[col][0] * n[0] + prim[col][1] * n[1];
        }
        row += 1;
    }

    // Interior — same `(x, y)` loop as MFEM; two normals per point: `nk[0]` then `nk[2]`.
    let p = 2usize;
    for j in 0..p {
        for i in 0..(p - j) {
            let wsum = iop[i] + iop[j] + iop[p - 1 - i - j];
            let x = iop[i] / wsum;
            let y = iop[j] / wsum;
            eval_primitives(x, y, &mut prim);
            for n in [NK[0], NK[2]] {
                for col in 0..N {
                    v[row][col] = prim[col][0] * n[0] + prim[col][1] * n[1];
                }
                row += 1;
            }
        }
    }
    debug_assert_eq!(row, N);
    v
}

/// Returns `V^{-1}` for square `V` (rows = DOF index).
fn invert_v(v: [[f64; N]; N]) -> [[f64; N]; N] {
    let mut data = [0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            data[i * N + j] = v[i][j];
        }
    }
    let dm = DMatrix::from_row_slice(N, N, &data);
    let inv = dm
        .try_inverse()
        .expect("TriRT2 Vandermonde matrix is singular or ill-conditioned");
    let mut r = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            r[i][j] = inv[(i, j)];
        }
    }
    r
}

fn transpose(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut t = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn coeff() -> &'static [[f64; N]; N] {
    COEFF.get_or_init(|| {
        let v = build_vandermonde();
        transpose(&invert_v(v))
    })
}

/// Raviart–Thomas RT2 H(div) element on the reference triangle — 15 DOFs, order 2.
pub struct TriRT2;

impl VectorReferenceElement for TriRT2 {
    fn dim(&self) -> u8 {
        2
    }
    fn order(&self) -> u8 {
        2
    }
    fn n_dofs(&self) -> usize {
        N
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut prim = [[0.0f64; 2]; N];
        eval_primitives(x, y, &mut prim);
        for i in 0..N {
            let mut vx = 0.0;
            let mut vy = 0.0;
            for j in 0..N {
                vx += c[i][j] * prim[j][0];
                vy += c[i][j] * prim[j][1];
            }
            values[i * 2] = vx;
            values[i * 2 + 1] = vy;
        }
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        let (x, y) = (xi[0], xi[1]);
        let c = coeff();
        let mut dprim = [0.0f64; N];
        eval_primitive_divs(x, y, &mut dprim);
        for i in 0..N {
            let mut s = 0.0;
            for j in 0..N {
                s += c[i][j] * dprim[j];
            }
            div_vals[i] = s;
        }
    }

    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tri_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let (bop, _) = gauss_legendre_01(3);
        let (iop, _) = gauss_legendre_01(2);
        let mut v = Vec::with_capacity(N);
        for t in &bop {
            v.push(vec![1.0 - t, *t]);
        }
        for t in &bop {
            v.push(vec![0.0, *t]);
        }
        for t in &bop {
            v.push(vec![*t, 0.0]);
        }
        let p = 2usize;
        for j in 0..p {
            for i in 0..(p - j) {
                let wsum = iop[i] + iop[j] + iop[p - 1 - i - j];
                let x = iop[i] / wsum;
                let y = iop[j] / wsum;
                v.push(vec![x, y]);
                v.push(vec![x, y]);
            }
        }
        debug_assert_eq!(v.len(), N);
        v
    }
}

impl TriRT2 {
    /// Fifteen MFEM-style reference RT2 primitives and their reference divergences
    /// at `ξ = (xi0, xi1)` on the reference triangle.
    #[inline]
    pub fn mfem_primitives_and_divs(
        xi0: f64,
        xi1: f64,
        prim: &mut [[f64; 2]; N],
        div_prim: &mut [f64; N],
    ) {
        eval_primitives(xi0, xi1, prim);
        eval_primitive_divs(xi0, xi1, div_prim);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::linalg::SVD;

    use super::*;

    fn mat_mul(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
        let mut p = [[0.0f64; N]; N];
        for i in 0..N {
            for k in 0..N {
                let mut s = 0.0f64;
                for j in 0..N {
                    s += a[i][j] * b[j][k];
                }
                p[i][k] = s;
            }
        }
        p
    }

    #[test]
    fn rt2_primitive_gram_rank() {
        let qr = tri_rule(6);
        let mut prim = [[0.0f64; 2]; N];
        let mut g = [[0.0f64; N]; N];
        for (xi, w) in qr.points.iter().zip(qr.weights.iter()) {
            let (x, y) = (xi[0], xi[1]);
            eval_primitives(x, y, &mut prim);
            for i in 0..N {
                for j in 0..N {
                    g[i][j] += w * (prim[i][0] * prim[j][0] + prim[i][1] * prim[j][1]);
                }
            }
        }
        let mut data = [0.0f64; N * N];
        for i in 0..N {
            for j in 0..N {
                data[i * N + j] = g[i][j];
            }
        }
        let gm = DMatrix::from_row_slice(N, N, &data);
        let svd = SVD::new(gm, false, false);
        let sv = svd.singular_values;
        let npos = sv.iter().filter(|s| **s > 1e-10).count();
        assert_eq!(
            npos, N,
            "primitive L² Gram rank {npos} (expected {N}); svs={sv:?}"
        );
    }

    #[test]
    fn rt2_vandermonde_inverse() {
        let v = build_vandermonde();
        let mut data = [0.0f64; N * N];
        for i in 0..N {
            for j in 0..N {
                data[i * N + j] = v[i][j];
            }
        }
        let dm = DMatrix::from_row_slice(N, N, &data);
        let svd = SVD::new(dm.clone(), true, true);
        let sv = svd.singular_values;
        let npos = sv.iter().filter(|s| **s > 1e-10).count();
        assert_eq!(
            npos, N,
            "V has rank {npos} (expected {N}); singular values={sv:?}"
        );
        let inv_dm = dm.clone().try_inverse().expect("invert");
        let err = (dm * &inv_dm - DMatrix::identity(N, N)).norm();
        assert!(err < 1e-9, "V*V^-1 err norm = {err}");

        let inv = invert_v(v);
        let prod = mat_mul(&build_vandermonde(), &inv);
        for i in 0..N {
            for k in 0..N {
                let exp = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][k] - exp).abs() < 1e-9,
                    "manual V*V^-1 at ({i},{k}) = {}",
                    prod[i][k]
                );
            }
        }
    }

    #[test]
    fn rt2_coeff_finite() {
        let c = coeff();
        let s: f64 = (0..N).map(|i| c[i][i].abs()).sum();
        assert!(s > 0.1 && s.is_finite());
    }

    #[test]
    fn rt2_basis_finite() {
        let elem = TriRT2;
        let mut buf = vec![0.0f64; 2 * N];
        for xi in &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0]] {
            elem.eval_basis_vec(xi, &mut buf);
            for &z in &buf {
                assert!(z.is_finite());
            }
        }
    }

    #[test]
    fn rt2_nodal_basis() {
        let elem = TriRT2;
        let (bop, _) = gauss_legendre_01(3);
        let (iop, _) = gauss_legendre_01(2);
        let mut vals = vec![0.0f64; 2 * N];
        let mut dof_mat = [[0.0f64; N]; N];
        let mut row = 0usize;

        for t in &bop {
            elem.eval_basis_vec(&[1.0 - t, *t], &mut vals);
            let n = NK[1];
            for i in 0..N {
                dof_mat[row][i] = vals[i * 2] * n[0] + vals[i * 2 + 1] * n[1];
            }
            row += 1;
        }
        for t in &bop {
            elem.eval_basis_vec(&[0.0, *t], &mut vals);
            let n = NK[2];
            for i in 0..N {
                dof_mat[row][i] = vals[i * 2] * n[0] + vals[i * 2 + 1] * n[1];
            }
            row += 1;
        }
        for t in &bop {
            elem.eval_basis_vec(&[*t, 0.0], &mut vals);
            let n = NK[0];
            for i in 0..N {
                dof_mat[row][i] = vals[i * 2] * n[0] + vals[i * 2 + 1] * n[1];
            }
            row += 1;
        }

        let p = 2usize;
        for j in 0..p {
            for i in 0..(p - j) {
                let wsum = iop[i] + iop[j] + iop[p - 1 - i - j];
                let x = iop[i] / wsum;
                let y = iop[j] / wsum;
                elem.eval_basis_vec(&[x, y], &mut vals);
                for n in [NK[0], NK[2]] {
                    for k in 0..N {
                        dof_mat[row][k] = vals[k * 2] * n[0] + vals[k * 2 + 1] * n[1];
                    }
                    row += 1;
                }
            }
        }
        assert_eq!(row, N);

        for k in 0..N {
            for i in 0..N {
                let exp = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (dof_mat[k][i] - exp).abs() < 1e-8,
                    "DOF_{k}(Phi_{i}) = {}, expected {exp}",
                    dof_mat[k][i]
                );
            }
        }
    }
}
