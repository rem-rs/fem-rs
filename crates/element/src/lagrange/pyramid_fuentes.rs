//! MFEM 4.10's **default** pyramid family — `H1_FuentesPyramidElement`
//! (`ScalarPyramid::DefaultType = 1`, `fem/fe/fe_pyramid.hpp:23`).
//!
//! Reference pyramid `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)`; DOF count
//! `p(p²+3)+1` (5 / 15 / 37 / 77 / 141 for `p = 1..5`, `fe_h1.cpp:1045`).
//!
//! This is a 1:1 port of MFEM's Fuentes–Keith–Demkowicz family:
//!
//! * the node table of `H1_FuentesPyramidElement::H1_FuentesPyramidElement`
//!   (`fe_h1.cpp:1043-1168`) via [`h1_fuentes_pyramid_nodes`],
//! * the raw expansion `calcBasis` (`fe_h1.cpp:1231-1408`) via
//!   [`fuentes_raw_basis`],
//! * its analytic gradients `calcGradBasis` (`fe_h1.cpp:1410-1626`) via
//!   [`fuentes_raw_grad_basis`],
//! * the pyramid "affine" coordinates `lam1..lam5`, `mu`, `nu` and the
//!   polynomial building blocks `phi_E`/`phi_Q`/`phi_T` of
//!   `fem/fe/fe_pyramid.cpp`.
//!
//! The nodal basis is `φ_m = Σ_o T⁻¹(m, o) u_o` where `T(o, m) = u_o(node_m)`
//! is the Vandermonde matrix of the raw expansion sampled at the element's own
//! nodes (MFEM's `Ti.Factor(T)`, `Ti.Mult(u, shape)`).
//!
//! Differences from [`super::pyramid::H1PyramidPk`] (MFEM's Bergot pyramid,
//! `pyr_type = 0`, `(p+1)(p+2)(2p+3)/6` DOFs), all measured against MFEM 4.10
//! (probe `tmp/d324/fuentes_probe.cpp`, table in `tmp/d324/EVIDENCE.md`):
//!
//! | | Fuentes | Bergot |
//! |---|---|---|
//! | DOFs, `p = 2/3/4/5` | 15 / 37 / 77 / 141 | 14 / 30 / 55 / 91 |
//! | base-quad node order | `(cp[i], cp[p−j], 0)` (`j` reversed) | `(cp[i], cp[j], 0)` |
//! | tri-face node formulas | 4 distinct barycentric combos | `w`-normalised |
//! | interior block | `(p−1)³` tensor grid `(cp[i](1−cp[k]), cp[j](1−cp[k]), cp[k])` | `(p−1−k)²` stump grid |
//!
//! The slot *blocks* (5 vertices, 8 edge blocks, base quad, 4 tri faces,
//! interior) are the same as Bergot's; only the intra-block node placement and
//! the DOF counts differ.  `PyramidFECollection`/`L2_FECollection` pyramid
//! elements default to this family too (`fe_coll.cpp:1955`, `fe_l2.cpp:927`).

use crate::quadrature::pyramid_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

use super::pyramid::{calc_legendre_d, calc_scaled_jacobi};

/// MFEM `FuentesPyramid::apex_tol` (`fe_pyramid.hpp:88`): the `|z − 1|`
/// threshold below which the singular `lam*`/`mu`/`nu` expressions switch to
/// their apex-limit values.
const APEX_TOL: f64 = 1e-8;

/// MFEM `FuentesPyramid::CheckZ` (`fe_pyramid.hpp:93`).
fn check_z(z: f64) -> bool {
    (z - 1.0).abs() > APEX_TOL
}

// ─── Pyramid "affine" coordinates (fe_pyramid.hpp:96-122) ───────────────────

fn lam1(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) {
        (1.0 - x - z) * (1.0 - y - z) / (1.0 - z)
    } else {
        0.0
    }
}
fn lam2(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) {
        x * (1.0 - y - z) / (1.0 - z)
    } else {
        0.0
    }
}
fn lam3(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) {
        x * y / (1.0 - z)
    } else {
        0.0
    }
}
fn lam4(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) {
        (1.0 - x - z) * y / (1.0 - z)
    } else {
        0.0
    }
}
fn lam5(z: f64) -> f64 {
    if check_z(z) {
        z
    } else {
        1.0
    }
}

// ─── Gradients of the "affine" coordinates (fe_pyramid.cpp:21-52) ───────────

fn grad_lam1(x: f64, y: f64, z: f64) -> [f64; 3] {
    if check_z(z) {
        [
            -(1.0 - y - z) / (1.0 - z),
            -(1.0 - x - z) / (1.0 - z),
            x * y / ((1.0 - z) * (1.0 - z)) - 1.0,
        ]
    } else {
        [-0.5, -0.5, -0.75]
    }
}
fn grad_lam2(x: f64, y: f64, z: f64) -> [f64; 3] {
    if check_z(z) {
        [
            (1.0 - y - z) / (1.0 - z),
            -x / (1.0 - z),
            -x * y / ((1.0 - z) * (1.0 - z)),
        ]
    } else {
        [0.5, -0.5, -0.25]
    }
}
fn grad_lam3(x: f64, y: f64, z: f64) -> [f64; 3] {
    if check_z(z) {
        [
            y / (1.0 - z),
            x / (1.0 - z),
            x * y / ((1.0 - z) * (1.0 - z)),
        ]
    } else {
        [0.5, 0.5, 0.25]
    }
}
fn grad_lam4(x: f64, y: f64, z: f64) -> [f64; 3] {
    if check_z(z) {
        [
            -y / (1.0 - z),
            (1.0 - x - z) / (1.0 - z),
            -x * y / ((1.0 - z) * (1.0 - z)),
        ]
    } else {
        [-0.5, 0.5, -0.25]
    }
}
fn grad_lam5() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

// ─── mu / nu (fe_pyramid.hpp:173-226, fe_pyramid.cpp:186-231) ───────────────

fn mu0_xy(z: f64, xy: [f64; 2], ab: usize) -> f64 {
    1.0 - xy[ab - 1] / (1.0 - z)
}
fn mu1_xy(z: f64, xy: [f64; 2], ab: usize) -> f64 {
    xy[ab - 1] / (1.0 - z)
}
fn grad_mu0_xy(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, -xy[ab - 1] / ((1.0 - z) * (1.0 - z))];
    d[ab - 1] = -1.0 / (1.0 - z);
    d
}
fn grad_mu1_xy(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, xy[ab - 1] / ((1.0 - z) * (1.0 - z))];
    d[ab - 1] = 1.0 / (1.0 - z);
    d
}
fn mu0_z(z: f64) -> f64 {
    1.0 - z
}
fn mu1_z(z: f64) -> f64 {
    z
}
const GRAD_MU0_Z: [f64; 3] = [0.0, 0.0, -1.0];
const GRAD_MU1_Z: [f64; 3] = [0.0, 0.0, 1.0];

fn nu0(z: f64, xy: [f64; 2], ab: usize) -> f64 {
    1.0 - xy[ab - 1] - z
}
fn nu1(xy: [f64; 2], ab: usize) -> f64 {
    xy[ab - 1]
}
fn nu2(z: f64) -> f64 {
    z
}
/// MFEM's `grad_nu0(z, xy, ab)` (`fe_pyramid.cpp:216`) does not use `z`/`xy`.
fn grad_nu0(ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, -1.0];
    d[ab - 1] = -1.0;
    d
}
fn grad_nu1(ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, 0.0];
    d[ab - 1] = 1.0;
    d
}
const GRAD_NU2: [f64; 3] = [0.0, 0.0, 1.0];

// ─── Polynomial building blocks (fe_pyramid.cpp:293-958) ────────────────────

/// MFEM `FuentesPyramid::CalcScaledLegendre(p, x, t, u, dudx, dudt)`
/// (`fe_pyramid.cpp:313`): `u[i] = P̃_i(x/t)·t^i` with `P̃_i` the shifted
/// Legendre polynomial on `[0,1]`, plus `∂/∂x` and `∂/∂t`.
fn calc_scaled_legendre(p: usize, x: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if t > 0.0 {
        let (mut u, mut dudx) = calc_legendre_d(p, x / t);
        dudx[0] = 0.0;
        let mut dudt = vec![0.0; p + 1];
        // MFEM: `dudt[0] = - dudx[0] * x / t;` after `dudx[0] = 0.0`, i.e. 0.
        for i in 1..=p {
            // MFEM uses `pow(t, i)`; `powf` reproduces libm's (correctly
            // rounded) result, `powi` would drift by up to a few ulps.
            let ti = t.powf(i as f64);
            u[i] *= ti;
            dudx[i] *= t.powf(i as f64 - 1.0);
            dudt[i] = (u[i] * i as f64 - dudx[i] * x) / t;
        }
        (u, dudx, dudt)
    } else {
        // MFEM's `t == 0` branch (`fe_pyramid.cpp:329-348`).  Unreachable from
        // this element (every `phi_E` call passes `t = s0 + s1 > 0`), kept 1:1.
        let mut u = vec![0.0; p + 1];
        let mut dudx = vec![0.0; p + 1];
        let mut dudt = vec![0.0; p + 1];
        u[0] = 1.0;
        if p >= 1 {
            dudx[1] = 2.0;
            dudt[1] = -1.0;
        }
        (u, dudx, dudt)
    }
}

/// MFEM `FuentesPyramid::CalcIntegratedLegendre(p, x, t, u, dudx, dudt)`
/// (`fe_pyramid.cpp:395`): `L_i = ∫₀ˣ P_{i−1}(y; t) dy`, `L_0 = 0`, `L_1 = x`.
fn calc_integrated_legendre(p: usize, x: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if t > 0.0 {
        let (mut u, mut dudx, mut dudt) = calc_scaled_legendre(p, x, t);
        for i in (2..=p).rev() {
            let d = 4.0 * i as f64 - 2.0;
            u[i] = (u[i] - t * t * u[i - 2]) / d;
            dudx[i] = (dudx[i] - t * t * dudx[i - 2]) / d;
            dudt[i] = (dudt[i] - t * t * dudt[i - 2] - 2.0 * t * u[i - 2]) / d;
        }
        if p >= 1 {
            u[1] = x;
            dudx[1] = 1.0;
            dudt[1] = 0.0;
        }
        u[0] = 0.0;
        dudx[0] = 0.0;
        dudt[0] = 0.0;
        (u, dudx, dudt)
    } else {
        (vec![0.0; p + 1], vec![0.0; p + 1], vec![0.0; p + 1])
    }
}

/// MFEM `FuentesPyramid::CalcIntegratedJacobi(p, α, x, t, u, dudx, dudt)`
/// (`fe_pyramid.cpp:559`).
///
/// MFEM's value-only overload (`fe_pyramid.cpp:525`) has an extra
/// `t == 0` early-out returning `u₀ = 1`; every call from this element passes
/// `t = t₀ + t₁ = 1` (the `nu`/`mu` coordinates of a pyramid edge sum to one),
/// so the two overloads agree and only the derivative one is ported.
fn calc_integrated_jacobi(
    p: usize,
    alpha: f64,
    x: f64,
    t: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut u, mut dudx, mut dudt) = calc_scaled_jacobi(p, alpha, x, t);
    for i in (2..=p).rev() {
        let d0 = 2.0 * i as f64 + alpha;
        let d1 = d0 - 1.0;
        let d2 = d0 - 2.0;
        let a = (alpha + i as f64) / (d0 * d1);
        let b = alpha / (d0 * d2);
        let c = (i - 1) as f64 / (d1 * d2);
        u[i] = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
        dudx[i] = a * dudx[i] + b * t * dudx[i - 1] - c * t * t * dudx[i - 2];
        dudt[i] =
            a * dudt[i] + b * t * dudt[i - 1] + b * u[i - 1] - c * t * t * dudt[i - 2]
                - 2.0 * c * t * u[i - 2];
    }
    if p >= 1 {
        u[1] = x;
        dudx[1] = 1.0;
        dudt[1] = 0.0;
    }
    u[0] = 0.0;
    dudx[0] = 0.0;
    dudt[0] = 0.0;
    (u, dudx, dudt)
}

/// MFEM `CalcHomogenizedIntLegendre(p, t₀, t₁, u, ∂/∂t₀, ∂/∂t₁)`
/// (`fe_pyramid.cpp:493`): `φ(t₀,t₁) = L_p(x = t₁; t = t₀+t₁)`,
/// `∂φ/∂t₀ = ∂L/∂t`, `∂φ/∂t₁ = ∂L/∂x + ∂L/∂t`.
fn calc_homogenized_int_legendre(
    p: usize,
    t0: f64,
    t1: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (u, dudx, dudt) = calc_integrated_legendre(p, t1, t0 + t1);
    let mut d1 = dudx;
    for i in 0..=p {
        d1[i] += dudt[i];
    }
    (u, dudt, d1)
}

/// `phi_E(p, s)` (`fe_pyramid.cpp:730-749`) — the (homogenised) edge function
/// of an edge whose coordinates are `s = (s₀, s₁)`.
fn phi_e(p: usize, s: [f64; 2]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    calc_homogenized_int_legendre(p, s[0], s[1])
}

/// `phi_E(p, s, ∇s, u, ∇u)` (`fe_pyramid.cpp:762`) — value and reference
/// gradient, `∇u = (∂u/∂s₀)∇s₀ + (∂u/∂s₁)∇s₁`.
fn phi_e_grad(p: usize, s: [f64; 2], grad_s: [[f64; 3]; 2]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (u, d0, d1) = calc_homogenized_int_legendre(p, s[0], s[1]);
    let mut gu = vec![[0.0; 3]; p + 1];
    for i in 0..=p {
        for d in 0..3 {
            gu[i][d] = d0[i] * grad_s[0][d] + d1[i] * grad_s[1][d];
        }
    }
    (u, gu)
}

/// `phi_Q(p, s, t)` (`fe_pyramid.cpp:786`) — the quadrilateral-face function,
/// returned as the `(p+1)×(p+1)` row-major matrix `u[i*(p+1) + j]`.
fn phi_q(p: usize, s: [f64; 2], t: [f64; 2]) -> Vec<f64> {
    let ei = phi_e(p, s).0;
    let ej = phi_e(p, t).0;
    let mut u = vec![0.0; (p + 1) * (p + 1)];
    for j in 0..=p {
        for i in 0..=p {
            u[i * (p + 1) + j] = ei[i] * ej[j];
        }
    }
    u
}

/// `phi_Q(p, s, ∇s, t, ∇t, u, ∇u)` (`fe_pyramid.cpp:814`) — value matrix plus
/// its reference gradients (`(p+1)×(p+1)` each).
fn phi_q_grad(
    p: usize,
    s: [f64; 2],
    grad_s: [[f64; 3]; 2],
    t: [f64; 2],
    grad_t: [[f64; 3]; 2],
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (ei, gei) = phi_e_grad(p, s, grad_s);
    let (ej, gej) = phi_e_grad(p, t, grad_t);
    let mut u = vec![0.0; (p + 1) * (p + 1)];
    let mut gu = vec![[0.0; 3]; (p + 1) * (p + 1)];
    for j in 0..=p {
        for i in 0..=p {
            let m = i * (p + 1) + j;
            u[m] = ei[i] * ej[j];
            for d in 0..3 {
                gu[m][d] = ei[i] * gej[j][d] + gei[i][d] * ej[j];
            }
        }
    }
    (u, gu)
}

/// `phi_T(p, ν)` (`fe_pyramid.cpp:868`) — the triangular-face function,
/// returned as the `p×(p−1)` row-major matrix `u[i*(p-1) + j]` (`= 0` outside
/// `2 ≤ i < p`, `1 ≤ j ≤ p − i`).
fn phi_t(p: usize, nu: [f64; 3]) -> Vec<f64> {
    let ei = phi_e(p - 1, [nu[0], nu[1]]).0;
    let mut u = vec![0.0; p * (p - 1)];
    for i in 2..p {
        let alpha = 2.0 * i as f64;
        let lj = calc_integrated_jacobi(p - 2, alpha, nu[2], (nu[0] + nu[1]) + nu[2]).0;
        for j in 1..=(p - i) {
            u[i * (p - 1) + j] = ei[i] * lj[j];
        }
    }
    u
}

/// `phi_T(p, ν, ∇ν, u, ∇u)` (`fe_pyramid.cpp:900`).
fn phi_t_grad(p: usize, nu: [f64; 3], grad_nu: [[f64; 3]; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (ei, gei) = phi_e_grad(p - 1, [nu[0], nu[1]], [grad_nu[0], grad_nu[1]]);
    let mut u = vec![0.0; p * (p - 1)];
    let mut gu = vec![[0.0; 3]; p * (p - 1)];
    for i in 2..p {
        let alpha = 2.0 * i as f64;
        let (lj, d01, d2) = calc_homogenized_int_jacobi(p - 2, alpha, nu[0] + nu[1], nu[2]);
        for j in 1..=(p - i) {
            let m = i * (p - 1) + j;
            u[m] = ei[i] * lj[j];
            for d in 0..3 {
                gu[m][d] = gei[i][d] * lj[j]
                    + ei[i] * (d01[j] * (grad_nu[0][d] + grad_nu[1][d]) + d2[j] * grad_nu[2][d]);
            }
        }
    }
    (u, gu)
}

/// MFEM `CalcHomogenizedIntJacobi(p, α, t₀, t₁, u, ∂/∂t₀, ∂/∂t₁)`
/// (`fe_pyramid.cpp:698`).
fn calc_homogenized_int_jacobi(
    p: usize,
    alpha: f64,
    t0: f64,
    t1: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (u, dudx, dudt) = calc_integrated_jacobi(p, alpha, t1, t0 + t1);
    let mut d1 = dudx;
    for i in 0..=p {
        d1[i] += dudt[i];
    }
    (u, dudt, d1)
}

// ─── The element ────────────────────────────────────────────────────────────

/// Number of DOFs of MFEM's Fuentes pyramid of order `p`: `p(p²+3)+1`
/// (`fe_h1.cpp:1045`).
pub const fn fuentes_pyramid_n_dofs(p: usize) -> usize {
    p * (p * p + 3) + 1
}

/// Node table of MFEM `H1_FuentesPyramidElement(p)` in the element's own DOF
/// order (`fe_h1.cpp:1064-1153`), with `cp` the `p+1` closed Gauss–Lobatto
/// points on `[0,1]`.
///
/// The block structure is the same as [`super::pyramid::h1_pyramid_slot_labels`]
/// (5 vertices, 8 edge blocks of `p−1`, the base quad face, 4 triangular faces,
/// the interior), but the intra-block placements differ:
///
/// * base quad face: `(cp[i], cp[p−j], 0)` — `j` runs **reversed** w.r.t.
///   Bergot's `(cp[i], cp[j], 0)`;
/// * triangular faces: four distinct `w`-normalised barycentric combinations
///   (`w = cp[i] + cp[j] + cp[p−i−j]`), e.g. `(1,2,4)` is
///   `((cp[i]+cp[p−i−j])/w, cp[i]/w, cp[j]/w)`;
/// * interior: the `(p−1)³` tensor grid `(cp[i](1−cp[k]), cp[j](1−cp[k]), cp[k])`
///   (Bergot instead uses the `(p−1−k)²` "stump" grid inside its own
///   collapsed-lattice labels).
pub fn h1_fuentes_pyramid_nodes(p: usize) -> Vec<[f64; 3]> {
    assert!(p >= 1, "order must be >= 1");
    let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    let mut n: Vec<[f64; 3]> = Vec::with_capacity(fuentes_pyramid_n_dofs(p));

    // vertices
    n.push([cp[0], cp[0], cp[0]]);
    n.push([cp[p], cp[0], cp[0]]);
    n.push([cp[p], cp[p], cp[0]]);
    n.push([cp[0], cp[p], cp[0]]);
    n.push([cp[0], cp[0], cp[p]]);

    // edges
    if p >= 2 {
        for i in 1..p {
            n.push([cp[i], cp[0], cp[0]]); // (0,1)
        }
        for i in 1..p {
            n.push([cp[p], cp[i], cp[0]]); // (1,2)
        }
        for i in 1..p {
            n.push([cp[i], cp[p], cp[0]]); // (3,2)
        }
        for i in 1..p {
            n.push([cp[0], cp[i], cp[0]]); // (0,3)
        }
        for i in 1..p {
            n.push([cp[0], cp[0], cp[i]]); // (0,4)
        }
        for i in 1..p {
            n.push([cp[p - i], cp[0], cp[i]]); // (1,4)
        }
        for i in 1..p {
            n.push([cp[p - i], cp[p - i], cp[i]]); // (2,4)
        }
        for i in 1..p {
            n.push([cp[0], cp[p - i], cp[i]]); // (3,4)
        }
    }

    // quadrilateral base face
    for j in 1..p {
        for i in 1..p {
            n.push([cp[i], cp[p - j], cp[0]]);
        }
    }

    // triangular faces
    for j in 1..p {
        for i in 1..p - j {
            let w = cp[i] + cp[j] + cp[p - i - j];
            n.push([cp[i] / w, cp[0], cp[j] / w]); // (0,1,4)
        }
    }
    for j in 1..p {
        for i in 1..p - j {
            let w = cp[i] + cp[j] + cp[p - i - j];
            n.push([(cp[i] + cp[p - i - j]) / w, cp[i] / w, cp[j] / w]); // (1,2,4)
        }
    }
    for j in 1..p {
        for i in 1..p - j {
            let w = cp[i] + cp[j] + cp[p - i - j];
            n.push([cp[p - i - j] / w, (cp[i] + cp[p - i - j]) / w, cp[j] / w]); // (2,3,4)
        }
    }
    for j in 1..p {
        for i in 1..p - j {
            let w = cp[i] + cp[j] + cp[p - i - j];
            n.push([cp[0], cp[p - i - j] / w, cp[j] / w]); // (3,0,4)
        }
    }

    // interior, on Fuentes' bubbles
    for k in 1..p {
        for j in 1..p {
            for i in 1..p {
                n.push([cp[i] * (1.0 - cp[k]), cp[j] * (1.0 - cp[k]), cp[k]]);
            }
        }
    }
    n
}

/// Raw `calcBasis` expansion of MFEM `H1_FuentesPyramidElement`
/// (`fe_h1.cpp:1231`), in the element's DOF order.
///
/// `u` is pre-zeroed by the caller; MFEM's explicit zero-fills of the
/// `CheckZ`-guarded blocks only advance the cursor `o`, so this port advances
/// it by the block size.
pub fn fuentes_raw_basis(p: usize, x: f64, y: f64, z: f64, u: &mut [f64]) {
    let xy = [x, y];
    let mut o = 0usize;

    u[0] = lam1(x, y, z);
    u[1] = lam2(x, y, z);
    u[2] = lam3(x, y, z);
    u[3] = lam4(x, y, z);
    u[4] = lam5(z);
    o += 5;

    // base edges — the x-oriented pair (a,b) = (1,2), then the y-oriented (2,1)
    if check_z(z) && p >= 2 {
        let pe = phi_e(p, [nu0(z, xy, 1), nu1(xy, 1)]).0;
        let mu = mu0_xy(z, xy, 2);
        for i in 2..=p {
            u[o] = mu * pe[i];
            o += 1;
        }
        let mu = mu1_xy(z, xy, 2);
        for i in 2..=p {
            u[o] = mu * pe[i];
            o += 1;
        }
        let pe = phi_e(p, [nu0(z, xy, 2), nu1(xy, 2)]).0;
        let mu = mu0_xy(z, xy, 1);
        for i in 2..=p {
            u[o] = mu * pe[i];
            o += 1;
        }
        let mu = mu1_xy(z, xy, 1);
        for i in 2..=p {
            u[o] = mu * pe[i];
            o += 1;
        }
    } else {
        o += 4 * (p - 1);
    }

    // apex (upright) edges — never `CheckZ`-guarded
    if p >= 2 {
        let l = [
            (lam1(x, y, z), lam5(z)),
            (lam2(x, y, z), lam5(z)),
            (lam3(x, y, z), lam5(z)),
            (lam4(x, y, z), lam5(z)),
        ];
        for &(s0, s1) in &l {
            let pe = phi_e(p, [s0, s1]).0;
            for i in 2..=p {
                u[o] = pe[i];
                o += 1;
            }
        }
    }

    // quadrilateral base face
    if check_z(z) && p >= 2 {
        let ph = phi_q(p, [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)], [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)]);
        let mu = mu0_z(z);
        for j in 2..=p {
            for i in 2..=p {
                u[o] = mu * ph[i * (p + 1) + j];
                o += 1;
            }
        }
    } else {
        o += (p - 1) * (p - 1);
    }

    // triangular faces — two `phi_T` evaluations, each shared by the c = 0/1
    // homogenising factors of one coordinate pair (as in MFEM)
    if check_z(z) && p >= 3 {
        let ph = phi_t(p, [nu0(z, xy, 1), nu1(xy, 1), nu2(z)]);
        for i in 2..p {
            for j in 1..=(p - i) {
                u[o] = mu0_xy(z, xy, 2) * ph[i * (p - 1) + j];
                o += 1;
            }
        }
        for i in 2..p {
            for j in 1..=(p - i) {
                u[o] = mu1_xy(z, xy, 2) * ph[i * (p - 1) + j];
                o += 1;
            }
        }
        let ph = phi_t(p, [nu0(z, xy, 2), nu1(xy, 2), nu2(z)]);
        for i in 2..p {
            for j in 1..=(p - i) {
                u[o] = mu0_xy(z, xy, 1) * ph[i * (p - 1) + j];
                o += 1;
            }
        }
        for i in 2..p {
            for j in 1..=(p - i) {
                u[o] = mu1_xy(z, xy, 1) * ph[i * (p - 1) + j];
                o += 1;
            }
        }
    } else if p >= 3 {
        o += 2 * (p - 1) * (p - 2);
    }

    // interior
    if check_z(z) && p >= 2 {
        let ph = phi_q(p, [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)], [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)]);
        let pe = phi_e(p, [mu0_z(z), mu1_z(z)]).0;
        for k in 2..=p {
            for j in 2..=p {
                for i in 2..=p {
                    u[o] = ph[i * (p + 1) + j] * pe[k];
                    o += 1;
                }
            }
        }
    } else {
        o += (p - 1) * (p - 1) * (p - 1);
    }
    debug_assert_eq!(o, fuentes_pyramid_n_dofs(p));
}

/// Raw `calcGradBasis` expansion of MFEM `H1_FuentesPyramidElement`
/// (`fe_h1.cpp:1410`): the reference gradients of [`fuentes_raw_basis`] in the
/// element's DOF order.
pub fn fuentes_raw_grad_basis(p: usize, x: f64, y: f64, z: f64, du: &mut [[f64; 3]]) {
    let xy = [x, y];
    let mut o = 0usize;

    du[0] = grad_lam1(x, y, z);
    du[1] = grad_lam2(x, y, z);
    du[2] = grad_lam3(x, y, z);
    du[3] = grad_lam4(x, y, z);
    du[4] = grad_lam5();
    o += 5;

    if check_z(z) && p >= 2 {
        let s = [nu0(z, xy, 1), nu1(xy, 1)];
        let gs = [grad_nu0(1), grad_nu1(1)];
        let (pe, dpe) = phi_e_grad(p, s, gs);
        for (mu, dmu) in [
            (mu0_xy(z, xy, 2), grad_mu0_xy(z, xy, 2)),
            (mu1_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)),
        ] {
            for i in 2..=p {
                for d in 0..3 {
                    du[o][d] = dmu[d] * pe[i] + mu * dpe[i][d];
                }
                o += 1;
            }
        }
        let s = [nu0(z, xy, 2), nu1(xy, 2)];
        let gs = [grad_nu0(2), grad_nu1(2)];
        let (pe, dpe) = phi_e_grad(p, s, gs);
        for (mu, dmu) in [
            (mu0_xy(z, xy, 1), grad_mu0_xy(z, xy, 1)),
            (mu1_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)),
        ] {
            for i in 2..=p {
                for d in 0..3 {
                    du[o][d] = dmu[d] * pe[i] + mu * dpe[i][d];
                }
                o += 1;
            }
        }
    } else {
        o += 4 * (p - 1);
    }

    if p >= 2 {
        let l: [[f64; 2]; 4] = [
            [lam1(x, y, z), lam5(z)],
            [lam2(x, y, z), lam5(z)],
            [lam3(x, y, z), lam5(z)],
            [lam4(x, y, z), lam5(z)],
        ];
        let gl: [[[f64; 3]; 2]; 4] = [
            [grad_lam1(x, y, z), grad_lam5()],
            [grad_lam2(x, y, z), grad_lam5()],
            [grad_lam3(x, y, z), grad_lam5()],
            [grad_lam4(x, y, z), grad_lam5()],
        ];
        for b in 0..4 {
            let (_, dpe) = phi_e_grad(p, l[b], gl[b]);
            for i in 2..=p {
                du[o] = dpe[i];
                o += 1;
            }
        }
    }

    if check_z(z) && p >= 2 {
        let (ph, dph) = phi_q_grad(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
        );
        let mu = mu0_z(z);
        for j in 2..=p {
            for i in 2..=p {
                let m = i * (p + 1) + j;
                for d in 0..3 {
                    du[o][d] = GRAD_MU0_Z[d] * ph[m] + mu * dph[m][d];
                }
                o += 1;
            }
        }
    } else {
        o += (p - 1) * (p - 1);
    }

    if check_z(z) && p >= 3 {
        // (a,b) = (1,2): c = 0 then c = 1, on one `phi_T` evaluation
        let nu_a = [nu0(z, xy, 1), nu1(xy, 1), nu2(z)];
        let gnu_a = [grad_nu0(1), grad_nu1(1), GRAD_NU2];
        let (ph, dph) = phi_t_grad(p, nu_a, gnu_a);
        for (mu, dmu) in [
            (mu0_xy(z, xy, 2), grad_mu0_xy(z, xy, 2)),
            (mu1_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)),
        ] {
            for i in 2..p {
                for j in 1..=(p - i) {
                    let m = i * (p - 1) + j;
                    for d in 0..3 {
                        du[o][d] = dmu[d] * ph[m] + mu * dph[m][d];
                    }
                    o += 1;
                }
            }
        }
        // (a,b) = (2,1)
        let nu_b = [nu0(z, xy, 2), nu1(xy, 2), nu2(z)];
        let gnu_b = [grad_nu0(2), grad_nu1(2), GRAD_NU2];
        let (ph, dph) = phi_t_grad(p, nu_b, gnu_b);
        for (mu, dmu) in [
            (mu0_xy(z, xy, 1), grad_mu0_xy(z, xy, 1)),
            (mu1_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)),
        ] {
            for i in 2..p {
                for j in 1..=(p - i) {
                    let m = i * (p - 1) + j;
                    for d in 0..3 {
                        du[o][d] = dmu[d] * ph[m] + mu * dph[m][d];
                    }
                    o += 1;
                }
            }
        }
    } else if p >= 3 {
        o += 2 * (p - 1) * (p - 2);
    }

    if check_z(z) && p >= 2 {
        let (ph, dph) = phi_q_grad(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
        );
        let (pe, dpe) = phi_e_grad(p, [mu0_z(z), mu1_z(z)], [GRAD_MU0_Z, GRAD_MU1_Z]);
        for k in 2..=p {
            for j in 2..=p {
                for i in 2..=p {
                    let m = i * (p + 1) + j;
                    for d in 0..3 {
                        du[o][d] = dph[m][d] * pe[k] + ph[m] * dpe[k][d];
                    }
                    o += 1;
                }
            }
        }
    } else {
        o += (p - 1) * (p - 1) * (p - 1);
    }
    debug_assert_eq!(o, fuentes_pyramid_n_dofs(p));
}

/// MFEM `H1_FuentesPyramidElement(p)` — MFEM's **default** pyramid H¹ element
/// (`ScalarPyramid::DefaultType = 1`), `p(p²+3)+1` DOFs on the GLL-barycentric
/// Fuentes nodes, basis `φ = T⁻¹ u` of the raw expansion
/// ([`fuentes_raw_basis`]).
///
/// The DOF order is the element's own node order
/// ([`h1_fuentes_pyramid_nodes`]) — MFEM's entity-slot order for pyramids
/// (vertices, 8 edge blocks, base quad face, 4 triangular faces, interior),
/// which is also the order in which MFEM's `H1_FECollection`
/// `FiniteElementSpace` numbers the element's DOFs (the pyramid FE carries no
/// `DofMap`).
///
/// This element is **not wired into** the space/assembly layers yet: fem-rs
/// numbers pyramid H¹ spaces with the Bergot family (D191/D299), and switching
/// the default requires a `pyr_type` knob threaded through
/// `DofManager::build_pyramid_pk`, `ref_elem_vol_h1`, `SetCurvature`'s pyramid
/// geometry and the L2 pyramid element (D324).
pub struct H1FuentesPyramidPk {
    inner: std::sync::Arc<H1FuentesPyramidPkInner>,
}

struct H1FuentesPyramidPkInner {
    order: usize,
    nodes: Vec<[f64; 3]>,
    /// `φ_m = Σ_o ti[m·n + o] · u_o(x)` — row-major `T⁻¹`.
    ti: Vec<f64>,
}

impl H1FuentesPyramidPk {
    /// Build (or fetch from the per-order cache) the element.
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<H1FuentesPyramidPkInner>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let inner = {
            let mut m = cache.lock().expect("H1FuentesPyramidPk cache poisoned");
            m.entry(p)
                .or_insert_with(|| Arc::new(h1_fuentes_pyramid_pk_build(p)))
                .clone()
        };
        Self { inner }
    }

    /// MFEM's `H1_FuentesPyramidElement(p)` node table, in DOF order.
    pub fn nodes(p: usize) -> Vec<[f64; 3]> {
        h1_fuentes_pyramid_nodes(p)
    }
}

fn h1_fuentes_pyramid_pk_build(p: usize) -> H1FuentesPyramidPkInner {
    let nodes = h1_fuentes_pyramid_nodes(p);
    let n = nodes.len();
    let mut u = vec![0.0; n];
    let mut t = nalgebra::DMatrix::<f64>::zeros(n, n);
    for (m, node) in nodes.iter().enumerate() {
        fuentes_raw_basis(p, node[0], node[1], node[2], &mut u);
        for (o, &v) in u.iter().enumerate() {
            t[(o, m)] = v;
        }
    }
    let ti_m = t
        .try_inverse()
        .expect("H1FuentesPyramidPk: singular Vandermonde matrix");
    let mut ti = vec![0.0; n * n];
    for m in 0..n {
        for o in 0..n {
            ti[m * n + o] = ti_m[(m, o)];
        }
    }
    H1FuentesPyramidPkInner { order: p, nodes, ti }
}

impl ReferenceElement for H1FuentesPyramidPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.inner.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.inner.nodes.len()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let mut u = vec![0.0; n];
        fuentes_raw_basis(p, xi[0], xi[1], xi[2], &mut u);
        for m in 0..n {
            let mut acc = 0.0;
            for o in 0..n {
                acc += self.inner.ti[m * n + o] * u[o];
            }
            values[m] = acc;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let p = self.inner.order;
        let n = self.inner.nodes.len();
        let mut du = vec![[0.0; 3]; n];
        fuentes_raw_grad_basis(p, xi[0], xi[1], xi[2], &mut du);
        for m in 0..n {
            let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
            for o in 0..n {
                let c = self.inner.ti[m * n + o];
                gx += c * du[o][0];
                gy += c * du[o][1];
                gz += c * du[o][2];
            }
            grads[m * 3] = gx;
            grads[m * 3 + 1] = gy;
            grads[m * 3 + 2] = gz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}
