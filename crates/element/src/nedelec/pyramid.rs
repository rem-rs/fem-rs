//! Nédélec-I H(curl) element on the reference pyramid (arbitrary order).
//!
//! Reference pyramid: base quad `(0,0),(1,0),(1,1),(0,1)` at z = 0, apex
//! `(0,0,1)` — the same frame MFEM (and the fem-rs assembly geometry) uses.
//!
//! `PyraNDk` is a 1:1 port of MFEM `ND_FuentesPyramidElement(p)`
//! (`fe_nd.cpp` + the `FuentesPyramid` helpers in `fe_pyramid.{hpp,cpp}`,
//! MFEM 4.10): the orientation-embedded Fuentes/Keith/Demkowicz basis of the
//! paper "Orientation embedded high order shape functions for the exact
//! sequence elements of all shapes" (§9.2).  The raw basis `u` (mixed edge
//! `E_E`, quadrilateral-face `E_Q`, triangular-face `E_T` families and the
//! interior families built on the integrated-Legendre/Jacobi generators) is
//! combined through the inverse interpolation matrix `Ti = T⁻¹` with
//! `T(b, m) = u_b(x_m)·tk_{dof2tk[m]}` — the DOF functionals are MFEM's nodal
//! point values `σ_m(Φ) = Φ(x_m)·(J tk_m)` (`fe_base.cpp::Project_ND`).

use crate::gll_basis::gll_nodes;
use crate::nedelec::tri_ndk::invert_dense;
use crate::quadrature::{gauss_legendre_01, pyramid_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

const APEX_TOL: f64 = 1e-8;
const SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SQRT2: f64 = std::f64::consts::SQRT_2;

/// `FuentesPyramid::CheckZ`.
#[inline]
fn check_z(z: f64) -> bool {
    (z - 1.0).abs() > APEX_TOL
}

// ─── Pyramid "affine" coordinates λ1..λ5 and their gradients ────────────────

#[inline]
fn lam1(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) { (1.0 - x - z) * (1.0 - y - z) / (1.0 - z) } else { 0.0 }
}
#[inline]
fn lam2(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) { x * (1.0 - y - z) / (1.0 - z) } else { 0.0 }
}
#[inline]
fn lam3(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) { x * y / (1.0 - z) } else { 0.0 }
}
#[inline]
fn lam4(x: f64, y: f64, z: f64) -> f64 {
    if check_z(z) { (1.0 - x - z) * y / (1.0 - z) } else { 0.0 }
}
#[inline]
fn lam5(_x: f64, _y: f64, z: f64) -> f64 {
    if check_z(z) { z } else { 1.0 }
}

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
fn grad_lam5(_x: f64, _y: f64, _z: f64) -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

/// `λ_i ∇λ_5 − λ_5 ∇λ_i` for the apex-touching edges (`FuentesPyramid::
/// lam15_grad_lam15` etc.).
fn lam15_grad_lam15(x: f64, y: f64, z: f64) -> [f64; 3] {
    let (l1, l5) = (lam1(x, y, z), lam5(x, y, z));
    let g1 = grad_lam1(x, y, z);
    let g5 = grad_lam5(x, y, z);
    std::array::from_fn(|d| l1 * g5[d] - l5 * g1[d])
}
fn lam25_grad_lam25(x: f64, y: f64, z: f64) -> [f64; 3] {
    let (l2, l5) = (lam2(x, y, z), lam5(x, y, z));
    let g2 = grad_lam2(x, y, z);
    let g5 = grad_lam5(x, y, z);
    std::array::from_fn(|d| l2 * g5[d] - l5 * g2[d])
}
fn lam35_grad_lam35(x: f64, y: f64, z: f64) -> [f64; 3] {
    let (l3, l5) = (lam3(x, y, z), lam5(x, y, z));
    let g3 = grad_lam3(x, y, z);
    let g5 = grad_lam5(x, y, z);
    std::array::from_fn(|d| l3 * g5[d] - l5 * g3[d])
}
fn lam45_grad_lam45(x: f64, y: f64, z: f64) -> [f64; 3] {
    let (l4, l5) = (lam4(x, y, z), lam5(x, y, z));
    let g4 = grad_lam4(x, y, z);
    let g5 = grad_lam5(x, y, z);
    std::array::from_fn(|d| l4 * g5[d] - l5 * g4[d])
}

// ─── μ/ν coordinates (`ab` ∈ {1, 2} indexes the xy component) ───────────────

#[inline]
fn mu0(z: f64) -> f64 {
    1.0 - z
}
#[inline]
fn mu1(z: f64) -> f64 {
    z
}
#[inline]
fn grad_mu0_1(_z: f64) -> [f64; 3] {
    [0.0, 0.0, -1.0]
}
#[inline]
fn grad_mu1_1(_z: f64) -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

#[inline]
fn mu0_3(z: f64, xy: [f64; 2], ab: usize) -> f64 {
    1.0 - xy[ab - 1] / (1.0 - z)
}
#[inline]
fn mu1_3(z: f64, xy: [f64; 2], ab: usize) -> f64 {
    xy[ab - 1] / (1.0 - z)
}

fn grad_mu0_3(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, -xy[ab - 1] / ((1.0 - z) * (1.0 - z))];
    d[ab - 1] = -1.0 / (1.0 - z);
    d
}
fn grad_mu1_3(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, xy[ab - 1] / ((1.0 - z) * (1.0 - z))];
    d[ab - 1] = 1.0 / (1.0 - z);
    d
}

/// `mu0 ∇mu1 − mu1 ∇mu0` (MFEM `mu01_grad_mu01`).
fn mu01_grad_mu01(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let (m0, m1) = (mu0_3(z, xy, ab), mu1_3(z, xy, ab));
    let g0 = grad_mu0_3(z, xy, ab);
    let g1 = grad_mu1_3(z, xy, ab);
    std::array::from_fn(|d| m0 * g1[d] - m1 * g0[d])
}

#[inline]
fn nu0(_z: f64, xy: [f64; 2], ab: usize) -> f64 {
    1.0 - xy[ab - 1] - _z
}
#[inline]
fn nu1(_z: f64, xy: [f64; 2], ab: usize) -> f64 {
    xy[ab - 1]
}
#[inline]
fn nu2(z: f64, _xy: [f64; 2], _ab: usize) -> f64 {
    z
}

fn grad_nu0(_z: f64, _xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, -1.0];
    d[ab - 1] = -1.0;
    d
}
fn grad_nu1(_z: f64, _xy: [f64; 2], ab: usize) -> [f64; 3] {
    let mut d = [0.0, 0.0, 0.0];
    d[ab - 1] = 1.0;
    d
}
fn grad_nu2(_z: f64, _xy: [f64; 2], _ab: usize) -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

fn nu01(z: f64, xy: [f64; 2], ab: usize) -> [f64; 2] {
    [nu0(z, xy, ab), nu1(z, xy, ab)]
}
fn nu12(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    [nu1(z, xy, ab), nu2(z, xy, ab), nu0(z, xy, ab)]
}
fn nu012(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    [nu0(z, xy, ab), nu1(z, xy, ab), nu2(z, xy, ab)]
}
fn nu120(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    [nu1(z, xy, ab), nu2(z, xy, ab), nu0(z, xy, ab)]
}

fn grad_nu01(z: f64, xy: [f64; 2], ab: usize) -> [[f64; 3]; 2] {
    [grad_nu0(z, xy, ab), grad_nu1(z, xy, ab)]
}
fn grad_nu012(z: f64, xy: [f64; 2], ab: usize) -> [[f64; 3]; 3] {
    [grad_nu0(z, xy, ab), grad_nu1(z, xy, ab), grad_nu2(z, xy, ab)]
}
fn grad_nu120(z: f64, xy: [f64; 2], ab: usize) -> [[f64; 3]; 3] {
    [grad_nu1(z, xy, ab), grad_nu2(z, xy, ab), grad_nu0(z, xy, ab)]
}

/// `nu0 ∇nu1 − nu1 ∇nu0` (MFEM `nu01_grad_nu01`).
fn nu01_grad_nu01(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let (n0, n1) = (nu0(z, xy, ab), nu1(z, xy, ab));
    let g0 = grad_nu0(z, xy, ab);
    let g1 = grad_nu1(z, xy, ab);
    std::array::from_fn(|d| n0 * g1[d] - n1 * g0[d])
}
/// `nu1 ∇nu2 − nu2 ∇nu1` (MFEM `nu12_grad_nu12`).
fn nu12_grad_nu12(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let (n1, n2) = (nu1(z, xy, ab), nu2(z, xy, ab));
    let g1 = grad_nu1(z, xy, ab);
    let g2 = grad_nu2(z, xy, ab);
    std::array::from_fn(|d| n1 * g2[d] - n2 * g1[d])
}

// ─── Scaled/integrated Legendre & Jacobi generators ─────────────────────────

/// `Poly_1D::CalcLegendre` (values + d/dz derivatives on the [-1,1] image).
fn calc_legendre(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0_f64; p + 1];
    let mut d = vec![0.0_f64; p + 1];
    u[0] = 1.0;
    if p == 0 {
        return (u, d);
    }
    let z = 2.0 * x - 1.0;
    u[1] = z;
    d[1] = 2.0;
    for n in 1..p {
        let nf = n as f64;
        u[n + 1] = ((2.0 * nf + 1.0) * z * u[n] - nf * u[n - 1]) / (nf + 1.0);
        d[n + 1] = (4.0 * nf + 2.0) * u[n] + d[n - 1];
    }
    (u, d)
}

/// `FuentesPyramid::CalcScaledLegendre(p, x, t, u, dudx, dudt)`.
fn calc_scaled_legendre_d(p: usize, x: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0_f64; p + 1];
    let mut dudx = vec![0.0_f64; p + 1];
    let mut dudt = vec![0.0_f64; p + 1];
    if t > 0.0 {
        let (lu, ld) = calc_legendre(p, x / t);
        u.copy_from_slice(&lu);
        dudx.copy_from_slice(&ld);
        dudx[0] = 0.0;
        dudt[0] = -dudx[0] * x / t;
        for i in 1..=p {
            u[i] *= t.powi(i as i32);
            dudx[i] *= t.powi(i as i32 - 1);
            dudt[i] = (u[i] * i as f64 - dudx[i] * x) / t;
        }
    } else {
        u[0] = 1.0;
        if p >= 1 {
            u[1] = 0.0;
            dudx[1] = 2.0;
            dudt[1] = -1.0;
        }
    }
    (u, dudx, dudt)
}

/// `FuentesPyramid::CalcIntegratedLegendre(p, x, t, u, dudx, dudt)`.
fn calc_integrated_legendre_d(p: usize, x: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut u, mut dudx, mut dudt) = calc_scaled_legendre_d(p, x, t);
    if t > 0.0 {
        for i in (2..=p).rev() {
            let c = 4.0 * i as f64 - 2.0;
            u[i] = (u[i] - t * t * u[i - 2]) / c;
            dudx[i] = (dudx[i] - t * t * dudx[i - 2]) / c;
            dudt[i] = (dudt[i] - t * t * dudt[i - 2] - 2.0 * t * u[i - 2]) / c;
        }
        if p >= 1 {
            u[1] = x;
            dudx[1] = 1.0;
            dudt[1] = 0.0;
        }
        u[0] = 0.0;
        dudx[0] = 0.0;
        dudt[0] = 0.0;
    } else {
        for v in u.iter_mut() {
            *v = 0.0;
        }
        for v in dudx.iter_mut() {
            *v = 0.0;
        }
        for v in dudt.iter_mut() {
            *v = 0.0;
        }
    }
    (u, dudx, dudt)
}

/// `CalcHomogenizedScaLegendre(p, s0, s1, u, duds0, duds1)`.
fn calc_homog_sca_legendre_d(p: usize, s0: f64, s1: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (u, duds1, duds0) = calc_scaled_legendre_d(p, s1, s0 + s1);
    let mut duds1 = duds1;
    for i in 0..=p {
        duds1[i] += duds0[i];
    }
    (u, duds0, duds1)
}

/// `CalcHomogenizedIntLegendre(p, t0, t1, u, dudt0, dudt1)`.
fn calc_homog_int_legendre_d(p: usize, t0: f64, t1: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (u, dudt1, dudt0) = calc_integrated_legendre_d(p, t1, t0 + t1);
    let mut dudt1 = dudt1;
    for i in 0..=p {
        dudt1[i] += dudt0[i];
    }
    (u, dudt0, dudt1)
}

/// `FuentesPyramid::CalcScaledJacobi(p, alpha, x, t, u, dudx, dudt)`.
fn calc_scaled_jacobi_d(p: usize, alpha: f64, x: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0_f64; p + 1];
    let mut dudx = vec![0.0_f64; p + 1];
    let mut dudt = vec![0.0_f64; p + 1];
    u[0] = 1.0;
    if p >= 1 {
        u[1] = (2.0 + alpha) * x - t;
        dudx[1] = 2.0 + alpha;
        dudt[1] = -1.0;
    }
    for i in 2..=p {
        let fi = i as f64;
        let a = 2.0 * fi * (alpha + fi) * (2.0 * fi + alpha - 2.0);
        let b = 2.0 * fi + alpha - 1.0;
        let c = (2.0 * fi + alpha) * (2.0 * fi + alpha - 2.0);
        let d = 2.0 * (alpha + fi - 1.0) * (fi - 1.0) * (2.0 * fi + alpha);
        u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1]
            - d * t * t * u[i - 2])
            / a;
        dudx[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudx[i - 1]
            + 2.0 * c * u[i - 1])
            - d * t * t * dudx[i - 2])
            / a;
        dudt[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudt[i - 1]
            + (alpha * alpha - c) * u[i - 1])
            - d * t * t * dudt[i - 2]
            - 2.0 * d * t * u[i - 2])
            / a;
    }
    (u, dudx, dudt)
}

/// `FuentesPyramid::CalcIntegratedJacobi(p, alpha, x, t, u, dudx, dudt)`.
fn calc_integrated_jacobi_d(
    p: usize,
    alpha: f64,
    x: f64,
    t: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut u, mut dudx, mut dudt) = calc_scaled_jacobi_d(p, alpha, x, t);
    for i in (2..=p).rev() {
        let fi = i as f64;
        let d0 = 2.0 * fi + alpha;
        let d1 = d0 - 1.0;
        let d2 = d0 - 2.0;
        let a = (alpha + fi) / (d0 * d1);
        let b = alpha / (d0 * d2);
        let c = (fi - 1.0) / (d1 * d2);
        u[i] = a * u[i] + b * t * u[i - 1] - c * t * t * u[i - 2];
        dudx[i] = a * dudx[i] + b * t * dudx[i - 1] - c * t * t * dudx[i - 2];
        dudt[i] = a * dudt[i]
            + b * t * dudt[i - 1]
            + b * u[i - 1]
            - c * t * t * dudt[i - 2]
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

/// `CalcHomogenizedIntJacobi(p, alpha, t0, t1, u, dudt0, dudt1)`.
fn calc_homog_int_jacobi_d(
    p: usize,
    alpha: f64,
    t0: f64,
    t1: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (u, dudt1, dudt0) = calc_integrated_jacobi_d(p, alpha, t1, t0 + t1);
    let mut dudt1 = dudt1;
    for i in 0..=p {
        dudt1[i] += dudt0[i];
    }
    (u, dudt0, dudt1)
}

/// Value-only `CalcHomogenizedScaLegendre(p, s0, s1, u)`.
fn calc_homog_sca_legendre(p: usize, s0: f64, s1: f64) -> Vec<f64> {
    let (u, _, _) = calc_homog_sca_legendre_d(p, s0, s1);
    u
}
/// Value-only `CalcHomogenizedIntLegendre(p, t0, t1, u)`.
fn calc_homog_int_legendre(p: usize, t0: f64, t1: f64) -> Vec<f64> {
    let (u, _, _) = calc_homog_int_legendre_d(p, t0, t1);
    u
}
/// Value-only `CalcHomogenizedIntJacobi(p, alpha, t0, t1, u)`.
fn calc_homog_int_jacobi(p: usize, alpha: f64, t0: f64, t1: f64) -> Vec<f64> {
    let (u, _, _) = calc_homog_int_jacobi_d(p, alpha, t0, t1);
    u
}

// ─── Vector generators: E_E / E_Q / E_T (values + curls) ────────────────────

/// `FuentesPyramid::E_E(p, s, sds, u)`: `u(i,:) = P_i(s)·sds` with
/// `P = CalcHomogenizedScaLegendre(p−1, s0, s1)`.
fn e_e(p: usize, s: [f64; 2], sds: [f64; 3]) -> Vec<[f64; 3]> {
    let p_i = calc_homog_sca_legendre(p - 1, s[0], s[1]);
    (0..p)
        .map(|i| [p_i[i] * sds[0], p_i[i] * sds[1], p_i[i] * sds[2]])
        .collect()
}

/// `E_E(p, s, grad_s, u, curl_u)`.
fn e_e_curl(p: usize, s: [f64; 2], grad_s: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let p_i = calc_homog_sca_legendre(p - 1, s[0], s[1]);
    let sds: [f64; 3] = std::array::from_fn(|d| s[0] * grad_s[1][d] - s[1] * grad_s[0][d]);
    let dsxds: [f64; 3] = std::array::from_fn(|d| {
        grad_s[0][(d + 1) % 3] * grad_s[1][(d + 2) % 3]
            - grad_s[0][(d + 2) % 3] * grad_s[1][(d + 1) % 3]
    });
    let u = (0..p)
        .map(|i| [p_i[i] * sds[0], p_i[i] * sds[1], p_i[i] * sds[2]])
        .collect();
    let curl_u = (0..p)
        .map(|i| {
            let c = (i as f64 + 2.0) * p_i[i];
            [c * dsxds[0], c * dsxds[1], c * dsxds[2]]
        })
        .collect();
    (u, curl_u)
}

/// `phi_E(p, t)` — homogenized integrated Legendre (values).
fn phi_e(p: usize, t: [f64; 2]) -> Vec<f64> {
    calc_homog_int_legendre(p, t[0], t[1])
}

/// `phi_E(p, t, grad_t, u, grad_u)` — values and gradients
/// (`grad_u(i,:) = Σ_r duds(i,r)·grad_t(r,:)`, `duds = [d/dt0, d/dt1]`).
fn phi_e_grad(p: usize, t: [f64; 2], grad_t: &[[f64; 3]]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (u, dudt0, dudt1) = calc_homog_int_legendre_d(p, t[0], t[1]);
    let grad_u = (0..=p)
        .map(|i| {
            std::array::from_fn(|d| dudt0[i] * grad_t[0][d] + dudt1[i] * grad_t[1][d])
        })
        .collect();
    (u, grad_u)
}

/// `FuentesPyramid::E_Q(p, s, sds, t, u)`: `u(i,j,:) = phi_E_j(j)·E_E_i(i,:)`
/// for `j ≥ 2` (the `j < 2` columns are zero).
fn e_q(p: usize, s: [f64; 2], sds: [f64; 3], t: [f64; 2]) -> Vec<Vec<[f64; 3]>> {
    // indexed [i][j]
    let e_e_i = e_e(p, s, sds);
    let phi_j = phi_e(p, t);
    let mut u = vec![vec![[0.0_f64; 3]; p + 1]; p];
    for i in 0..p {
        for j in 2..=p {
            for k in 0..3 {
                u[i][j][k] = phi_j[j] * e_e_i[i][k];
            }
        }
    }
    u
}

/// `E_Q(p, s, grad_s, t, grad_t, u, curl_u)`.
#[allow(clippy::type_complexity)]
fn e_q_curl(
    p: usize,
    s: [f64; 2],
    grad_s: &[[f64; 3]],
    t: [f64; 2],
    grad_t: &[[f64; 3]],
) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>) {
    let (e_e_i, d_e_e_i) = e_e_curl(p, s, grad_s);
    let (phi_j, dphi_j) = phi_e_grad(p, t, grad_t);
    let mut u = vec![vec![[0.0_f64; 3]; p + 1]; p];
    let mut curl_u = u.clone();
    for i in 0..p {
        for j in 2..=p {
            for k in 0..3 {
                u[i][j][k] = phi_j[j] * e_e_i[i][k];
            }
            curl_u[i][j] = std::array::from_fn(|k| {
                phi_j[j] * d_e_e_i[i][k]
                    + dphi_j[j][(k + 1) % 3] * e_e_i[i][(k + 2) % 3]
                    - dphi_j[j][(k + 2) % 3] * e_e_i[i][(k + 1) % 3]
            });
        }
    }
    (u, curl_u)
}

/// `FuentesPyramid::E_T(p, s, sds, u)` with `p−1` rows and `j = 1..i+j<p`.
fn e_t(p: usize, s: [f64; 3], sds: [f64; 3]) -> Vec<Vec<[f64; 3]>> {
    let e_e_i = e_e(p - 1, [s[0], s[1]], sds);
    let mut u = vec![vec![[0.0_f64; 3]; p]; p - 1];
    for i in 0..p - 1 {
        let alpha = 2.0 * i as f64 + 1.0;
        let l = calc_homog_int_jacobi(p - 1, alpha, s[0] + s[1], s[2]);
        for j in 1..(p - i) {
            for k in 0..3 {
                u[i][j][k] = l[j] * e_e_i[i][k];
            }
        }
    }
    u
}

/// `E_T(p, s, grad_s, u, curl_u)`.
#[allow(clippy::type_complexity)]
fn e_t_curl(
    p: usize,
    s: [f64; 3],
    grad_s: &[[f64; 3]],
) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>) {
    let (e_e_i, d_e_e_i) = e_e_curl(p - 1, [s[0], s[1]], grad_s);
    let mut u = vec![vec![[0.0_f64; 3]; p]; p - 1];
    let mut curl_u = u.clone();
    for i in 0..p - 1 {
        let alpha = 2.0 * i as f64 + 1.0;
        let (l, dl_dx, dl_dt) = calc_homog_int_jacobi_d(p - 1, alpha, s[0] + s[1], s[2]);
        for j in 1..(p - i) {
            // grad_L = dL_dx·(grad_s0 + grad_s1) + dL_dt·grad_s2
            let grad_l: [f64; 3] = std::array::from_fn(|d| {
                dl_dx[j] * (grad_s[0][d] + grad_s[1][d]) + dl_dt[j] * grad_s[2][d]
            });
            for k in 0..3 {
                u[i][j][k] = l[j] * e_e_i[i][k];
            }
            curl_u[i][j] = std::array::from_fn(|k| {
                l[j] * d_e_e_i[i][k]
                    + grad_l[(k + 1) % 3] * e_e_i[i][(k + 2) % 3]
                    - grad_l[(k + 2) % 3] * e_e_i[i][(k + 1) % 3]
            });
        }
    }
    (u, curl_u)
}

/// `phi_Q(p, s, t)` — the product of two `phi_E` factors.
fn phi_q(p: usize, s: [f64; 2], t: [f64; 2]) -> Vec<Vec<f64>> {
    let phi_s = phi_e(p, s);
    let phi_t = phi_e(p, t);
    (0..=p)
        .map(|i| (0..=p).map(|j| phi_s[i] * phi_t[j]).collect())
        .collect()
}

/// `phi_Q(p, s, grad_s, t, grad_t, u, grad_u)`.
#[allow(clippy::type_complexity)]
fn phi_q_grad(
    p: usize,
    s: [f64; 2],
    grad_s: &[[f64; 3]],
    t: [f64; 2],
    grad_t: &[[f64; 3]],
) -> (Vec<Vec<f64>>, Vec<Vec<[f64; 3]>>) {
    let (phi_s, dphi_s) = phi_e_grad(p, s, grad_s);
    let (phi_t, dphi_t) = phi_e_grad(p, t, grad_t);
    let u = (0..=p)
        .map(|i| (0..=p).map(|j| phi_s[i] * phi_t[j]).collect())
        .collect();
    let grad_u = (0..=p)
        .map(|i| {
            (0..=p)
                .map(|j| {
                    std::array::from_fn(|d| {
                        phi_s[i] * dphi_t[j][d] + dphi_s[i][d] * phi_t[j]
                    })
                })
                .collect()
        })
        .collect();
    (u, grad_u)
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─── The Fuentes raw basis (`calcBasis` / `calcCurlBasis`) ──────────────────

/// MFEM `ND_FuentesPyramidElement::calcBasis`: the `dof` raw vector functions
/// in slot order.
fn calc_basis(p: usize, x_in: f64, y_in: f64, z_in: f64) -> Vec<[f64; 3]> {
    let dof = p * (3 * p * p + 5);
    let mut w = vec![[0.0_f64; 3]; dof];
    let (mut x, mut y, mut z) = (x_in, y_in, z_in);
    if (1.0 - z).abs() < APEX_TOL {
        z = 1.0 - APEX_TOL;
        y = 0.5 * (1.0 - z);
        x = 0.5 * (1.0 - z);
    }
    let xy = [x, y];
    let mut o = 0;

    if z < 1.0 {
        // Mixed edges: (a,b) = (1,2) then (2,1), each with c = 0, 1.
        let ee12 = e_e(p, nu01(z, xy, 1), nu01_grad_nu01(z, xy, 1));
        let mu = mu0_3(z, xy, 2);
        for i in 0..p {
            w[o] = std::array::from_fn(|k: usize| mu * ee12[i][k]);
            o += 1;
        }
        let mu = mu1_3(z, xy, 2);
        for i in 0..p {
            w[o] = std::array::from_fn(|k: usize| mu * ee12[i][k]);
            o += 1;
        }
        let ee21 = e_e(p, nu01(z, xy, 2), nu01_grad_nu01(z, xy, 2));
        let mu = mu0_3(z, xy, 1);
        for i in 0..p {
            w[o] = std::array::from_fn(|k| mu * ee21[i][k]);
            o += 1;
        }
        let mu = mu1_3(z, xy, 1);
        for i in 0..p {
            w[o] = std::array::from_fn(|k| mu * ee21[i][k]);
            o += 1;
        }

        // Triangle edges.
        let lams = [
            lam15_grad_lam15(x, y, z),
            lam25_grad_lam25(x, y, z),
            lam35_grad_lam35(x, y, z),
            lam45_grad_lam45(x, y, z),
        ];
        let lam_vals = [
            [lam1(x, y, z), lam5(x, y, z)],
            [lam2(x, y, z), lam5(x, y, z)],
            [lam3(x, y, z), lam5(x, y, z)],
            [lam4(x, y, z), lam5(x, y, z)],
        ];
        for f in 0..4 {
            let u = e_e(p, lam_vals[f], lams[f]);
            for i in 0..p {
                w[o] = u[i];
                o += 1;
            }
        }
    }

    if z < 1.0 && p >= 2 {
        // Quadrilateral face: Family I then Family II.
        let mu = mu0(z);
        let mu2 = mu * mu;
        let eq1 = e_q(
            p,
            mu01_2(z, xy, 1),
            mu01_grad_mu01(z, xy, 1),
            mu01_2(z, xy, 2),
        );
        for j in 2..=p {
            for i in 0..p {
                w[o] = std::array::from_fn(|k| mu2 * eq1[i][j][k]);
                o += 1;
            }
        }
        let eq2 = e_q(
            p,
            mu01_2(z, xy, 2),
            mu01_grad_mu01(z, xy, 2),
            mu01_2(z, xy, 1),
        );
        for j in 2..=p {
            for i in 0..p {
                w[o] = std::array::from_fn(|k| mu2 * eq2[i][j][k]);
                o += 1;
            }
        }

        // Triangular faces: Family I ((a,b) = (1,2), (2,1) × c = 0, 1 with
        // nu012) then Family II (nu120).  The μ factor carries the *other*
        // ab than the E_T family (MFEM: mu0(z,xy,2) with nu012(z,xy,1)).
        for (ab, fam) in [(1usize, 0usize), (2, 0), (1, 1), (2, 1)] {
            let mab = 3 - ab;
            let mu = mu0_3(z, xy, mab);
            let et = if fam == 0 {
                e_t(p, nu012(z, xy, ab), nu01_grad_nu01(z, xy, ab))
            } else {
                e_t(p, nu120(z, xy, ab), nu12_grad_nu12(z, xy, ab))
            };
            for j in 1..p {
                for i in 0..(p - j) {
                    w[o] = std::array::from_fn(|k| mu * et[i][j][k]);
                    o += 1;
                }
            }
            let mu = mu1_3(z, xy, mab);
            for j in 1..p {
                for i in 0..(p - j) {
                    w[o] = std::array::from_fn(|k| mu * et[i][j][k]);
                    o += 1;
                }
            }
        }
    }

    if z < 1.0 && p >= 2 {
        // Interior.
        let (phi_q1, dphi_q1) = phi_q_grad(
            p,
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
        );
        let (phi_e_k, dphi_e_k) = phi_e_grad(p, mu01_1(z), &grad_mu01_1());
        // Family I
        for k in 2..=p {
            for j in 2..=p {
                for i in 2..=p {
                    w[o] = std::array::from_fn(|l| {
                        dphi_q1[i][j][l] * phi_e_k[k] + phi_q1[i][j] * dphi_e_k[k][l]
                    });
                    o += 1;
                }
            }
        }
        // Family II (reuses eq1) and III (eq2) need the E_Q blocks again.
        let mu = mu0(z);
        let eq1 = e_q(
            p,
            mu01_2(z, xy, 1),
            mu01_grad_mu01(z, xy, 1),
            mu01_2(z, xy, 2),
        );
        for k in 2..=p {
            for j in 2..=p {
                for i in 0..p {
                    w[o] = std::array::from_fn(|l| mu * eq1[i][j][l] * phi_e_k[k]);
                    o += 1;
                }
            }
        }
        let eq2 = e_q(
            p,
            mu01_2(z, xy, 2),
            mu01_grad_mu01(z, xy, 2),
            mu01_2(z, xy, 1),
        );
        for k in 2..=p {
            for j in 2..=p {
                for i in 0..p {
                    w[o] = std::array::from_fn(|l| mu * eq2[i][j][l] * phi_e_k[k]);
                    o += 1;
                }
            }
        }
        // Family IV
        let dmu = grad_mu0_1(z);
        let phi_q2 = phi_q(p, mu01_2(z, xy, 2), mu01_2(z, xy, 1));
        for j in 2..=p {
            for i in 2..=p {
                let n = i.max(j) as f64;
                let nmu = n * mu.powf(n - 1.0);
                w[o] = std::array::from_fn(|l| nmu * phi_q2[i][j] * dmu[l]);
                o += 1;
            }
        }
    }
    debug_assert_eq!(o, dof);
    w
}

/// MFEM `ND_FuentesPyramidElement::calcCurlBasis`.
fn calc_curl_basis(p: usize, x_in: f64, y_in: f64, z_in: f64) -> Vec<[f64; 3]> {
    let dof = p * (3 * p * p + 5);
    let mut dw = vec![[0.0_f64; 3]; dof];
    let (mut x, mut y, mut z) = (x_in, y_in, z_in);
    if (1.0 - z).abs() < APEX_TOL {
        z = 1.0 - APEX_TOL;
        y = 0.5 * (1.0 - z);
        x = 0.5 * (1.0 - z);
    }
    let xy = [x, y];
    let mut o = 0;

    if z < 1.0 {
        // Mixed edges.
        let (ee12, dee12) = e_e_curl(p, nu01(z, xy, 1), &grad_nu01(z, xy, 1));
        for (mu, gmu) in [
            (mu0_3(z, xy, 2), grad_mu0_3(z, xy, 2)),
            (mu1_3(z, xy, 2), grad_mu1_3(z, xy, 2)),
        ] {
            for i in 0..p {
                let mx = cross3(&gmu, &ee12[i]);
                dw[o] = std::array::from_fn(|k| mu * dee12[i][k] + mx[k]);
                o += 1;
            }
        }
        let (ee21, dee21) = e_e_curl(p, nu01(z, xy, 2), &grad_nu01(z, xy, 2));
        for (mu, gmu) in [
            (mu0_3(z, xy, 1), grad_mu0_3(z, xy, 1)),
            (mu1_3(z, xy, 1), grad_mu1_3(z, xy, 1)),
        ] {
            for i in 0..p {
                let mx = cross3(&gmu, &ee21[i]);
                dw[o] = std::array::from_fn(|k| mu * dee21[i][k] + mx[k]);
                o += 1;
            }
        }

        // Triangle edges.
        let lam_grads: [[[f64; 3]; 2]; 4] = [
            [grad_lam1(x, y, z), grad_lam5(x, y, z)],
            [grad_lam2(x, y, z), grad_lam5(x, y, z)],
            [grad_lam3(x, y, z), grad_lam5(x, y, z)],
            [grad_lam4(x, y, z), grad_lam5(x, y, z)],
        ];
        let lam_vals = [
            [lam1(x, y, z), lam5(x, y, z)],
            [lam2(x, y, z), lam5(x, y, z)],
            [lam3(x, y, z), lam5(x, y, z)],
            [lam4(x, y, z), lam5(x, y, z)],
        ];
        for f in 0..4 {
            let (_, cu) = e_e_curl(p, lam_vals[f], &lam_grads[f]);
            for i in 0..p {
                dw[o] = cu[i];
                o += 1;
            }
        }
    }

    if z < 1.0 && p >= 2 {
        // Quadrilateral face.
        let mu = mu0(z);
        let mu2 = mu * mu;
        let dmu = grad_mu0_1(z);
        let (eq1, deq1) = e_q_curl(
            p,
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
        );
        for j in 2..=p {
            for i in 0..p {
                let mx = cross3(&dmu, &eq1[i][j]);
                dw[o] = std::array::from_fn(|k| mu2 * deq1[i][j][k] + 2.0 * mu * mx[k]);
                o += 1;
            }
        }
        let (eq2, deq2) = e_q_curl(
            p,
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
        );
        for j in 2..=p {
            for i in 0..p {
                let mx = cross3(&dmu, &eq2[i][j]);
                dw[o] = std::array::from_fn(|k| mu2 * deq2[i][j][k] + 2.0 * mu * mx[k]);
                o += 1;
            }
        }

        // Triangular faces (μ carries the other ab than the E_T family).
        for (ab, fam) in [(1usize, 0usize), (2, 0), (1, 1), (2, 1)] {
            let mab = 3 - ab;
            let (mu0v, gmu0) = (mu0_3(z, xy, mab), grad_mu0_3(z, xy, mab));
            let (mu1v, gmu1) = (mu1_3(z, xy, mab), grad_mu1_3(z, xy, mab));
            let (et, det) = if fam == 0 {
                e_t_curl(p, nu012(z, xy, ab), &grad_nu012(z, xy, ab))
            } else {
                e_t_curl(p, nu120(z, xy, ab), &grad_nu120(z, xy, ab))
            };
            for (mu, gmu) in [(mu0v, gmu0), (mu1v, gmu1)] {
                for j in 1..p {
                    for i in 0..(p - j) {
                        let mx = cross3(&gmu, &et[i][j]);
                        dw[o] = std::array::from_fn(|k| mu * det[i][j][k] + mx[k]);
                        o += 1;
                    }
                }
            }
        }
    }

    if z < 1.0 && p >= 2 {
        // Interior: Family I has zero curl (skip).
        o += (p - 1) * (p - 1) * (p - 1);
        let mu = mu0(z);
        let dmu = grad_mu0_1(z);
        let (phi_e_k, dphi_e_k) = phi_e_grad(p, mu01_1(z), &grad_mu01_1());
        // Family II
        let (eq1, deq1) = e_q_curl(
            p,
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
        );
        for k in 2..=p {
            let muphi = std::array::from_fn(|l| {
                mu * dphi_e_k[k][l] + phi_e_k[k] * dmu[l]
            });
            for j in 2..=p {
                for i in 0..p {
                    let mx = cross3(&muphi, &eq1[i][j]);
                    dw[o] = std::array::from_fn(|l| {
                        mu * deq1[i][j][l] * phi_e_k[k] + mx[l]
                    });
                    o += 1;
                }
            }
        }
        // Family III
        let (eq2, deq2) = e_q_curl(
            p,
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
        );
        for k in 2..=p {
            let muphi = std::array::from_fn(|l| {
                mu * dphi_e_k[k][l] + phi_e_k[k] * dmu[l]
            });
            for j in 2..=p {
                for i in 0..p {
                    let mx = cross3(&muphi, &eq2[i][j]);
                    dw[o] = std::array::from_fn(|l| {
                        mu * deq2[i][j][l] * phi_e_k[k] + mx[l]
                    });
                    o += 1;
                }
            }
        }
        // Family IV
        let (_, dphi_q2) = phi_q_grad(
            p,
            mu01_2(z, xy, 2),
            &grad_mu01_2(z, xy, 2),
            mu01_2(z, xy, 1),
            &grad_mu01_2(z, xy, 1),
        );
        for j in 2..=p {
            for i in 2..=p {
                let n = i.max(j) as f64;
                let nmu = n * mu.powf(n - 1.0);
                let muphi = cross3(&dphi_q2[i][j], &dmu);
                dw[o] = std::array::from_fn(|l| nmu * muphi[l]);
                o += 1;
            }
        }
    }
    debug_assert_eq!(o, dof);
    dw
}

// frame helpers ---------------------------------------------------------------

#[inline]
fn mu01_1(z: f64) -> [f64; 2] {
    [mu0(z), mu1(z)]
}
#[inline]
fn grad_mu01_1() -> [[f64; 3]; 2] {
    [grad_mu0_1(0.0), grad_mu1_1(0.0)]
}
#[inline]
fn mu01_2(z: f64, xy: [f64; 2], ab: usize) -> [f64; 2] {
    [mu0_3(z, xy, ab), mu1_3(z, xy, ab)]
}
#[inline]
fn grad_mu01_2(z: f64, xy: [f64; 2], ab: usize) -> [[f64; 3]; 2] {
    [grad_mu0_3(z, xy, ab), grad_mu1_3(z, xy, ab)]
}

// ─── Element ─────────────────────────────────────────────────────────────────

/// MFEM `tk` table (27 entries = 9 tangent directions).
const PYRA_TK: [[f64; 3]; 9] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, -1.0, 1.0],
    [0.0, -1.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [-SQRT1_2, -SQRT1_2, SQRT2],
];

/// The `dof2tk` array of MFEM `ND_FuentesPyramidElement`: tangent index per
/// slot, in the constructor's enumeration (edges, base quad face, four tri
/// faces, interior).
fn pyra_dof2tk(p: usize) -> Vec<u8> {
    let pm2 = p.saturating_sub(2);
    let mut t = Vec::with_capacity(p * (3 * p * p + 5));
    // edges (0,1) (1,2) (3,2) (0,3) (0,4) (1,4) (2,4) (3,4)
    for _ in 0..p {
        t.push(0);
    }
    for _ in 0..p {
        t.push(1);
    }
    for _ in 0..p {
        t.push(0);
    }
    for _ in 0..p {
        t.push(1);
    }
    for _ in 0..p {
        t.push(2);
    }
    for _ in 0..p {
        t.push(3);
    }
    for _ in 0..p {
        t.push(4);
    }
    for _ in 0..p {
        t.push(5);
    }
    if p >= 2 {
        // base quad face (3,2,1,0): x-tangent block then y-tangent block
        for _ in 0..(p - 1) * p {
            t.push(0);
        }
        for _ in 0..p * (p - 1) {
            t.push(7);
        }
        // tri faces (0,1,4) (1,2,4) (2,3,4) (3,0,4): (pm2+1)(pm2+2)/2
        // barycentric GL points, two tangent slots per point.
        let n_pt = (pm2 + 1) * (pm2 + 2) / 2;
        for &(a, b) in [(0u8, 2u8), (1, 3), (6, 4), (7, 5)].iter() {
            for _ in 0..n_pt {
                t.push(a);
                t.push(b);
            }
        }
        // interior: x block, y block, z block (each (p−1)²·p slots)
        for _ in 0..(p - 1) * (p - 1) * p {
            t.push(0);
        }
        for _ in 0..(p - 1) * p * (p - 1) {
            t.push(1);
        }
        for _ in 0..p * (p - 1) * (p - 1) {
            t.push(8);
        }
    }
    debug_assert_eq!(t.len(), p * (3 * p * p + 5));
    t
}

/// The `FE::Nodes` points of MFEM `ND_FuentesPyramidElement(p)` in the
/// constructor's enumeration (single source of truth for the slot table).
fn pyra_nodes(p: usize) -> Vec<[f64; 3]> {
    let pm2 = p.saturating_sub(2);
    let (qop, _) = gauss_legendre_01(p);
    let qcp: Vec<f64> = gll_nodes(p).iter().map(|&x| 0.5 * (x + 1.0)).collect();
    let (top, _) = gauss_legendre_01((p - 1).max(1));

    let mut pts = Vec::with_capacity(p * (3 * p * p + 5));
    // edges: (0,1) (1,2) (3,2) (0,3) (0,4) (1,4) (2,4) (3,4)
    for i in 0..p {
        pts.push([qop[i], 0.0, 0.0]);
    }
    for i in 0..p {
        pts.push([1.0, qop[i], 0.0]);
    }
    for i in 0..p {
        pts.push([qop[i], 1.0, 0.0]);
    }
    for i in 0..p {
        pts.push([0.0, qop[i], 0.0]);
    }
    for i in 0..p {
        pts.push([0.0, 0.0, qop[i]]);
    }
    for i in 0..p {
        pts.push([1.0 - qop[i], 0.0, qop[i]]);
    }
    for i in 0..p {
        pts.push([1.0 - qop[i], 1.0 - qop[i], qop[i]]);
    }
    for i in 0..p {
        pts.push([0.0, 1.0 - qop[i], qop[i]]);
    }
    if p >= 2 {
        // base quad face (3,2,1,0): x-tangent block then y-tangent block.
        for j in 1..p {
            for i in 0..p {
                pts.push([qop[i], qcp[p - j], 0.0]);
            }
        }
        for j in 0..p {
            for i in 1..p {
                pts.push([qcp[i], qop[p - 1 - j], 0.0]);
            }
        }
        // tri faces: two tangent slots per barycentric GL point.
        for f in 0..4 {
            for j in 0..=pm2 {
                for i in 0..=(pm2 - j) {
                    let w = top[i] + top[j] + top[pm2 - i - j];
                    let pt = match f {
                        0 => [top[i] / w, 0.0, top[j] / w],
                        1 => [(top[i] + top[pm2 - i - j]) / w, top[i] / w, top[j] / w],
                        2 => {
                            [top[pm2 - i - j] / w, (top[i] + top[pm2 - i - j]) / w, top[j] / w]
                        }
                        _ => [0.0, top[pm2 - i - j] / w, top[j] / w],
                    };
                    pts.push(pt);
                    pts.push(pt);
                }
            }
        }
    }
    // interior: x block, then y block, then z block (collapsed scaling about
    // the apex axis).
    for k in 1..p {
        let w = 1.0 - qcp[k];
        for j in 1..p {
            for i in 0..p {
                pts.push([qop[i] * w, qcp[j] * w, qcp[k]]);
            }
        }
    }
    for k in 1..p {
        let w = 1.0 - qcp[k];
        for j in 0..p {
            for i in 1..p {
                pts.push([qcp[i] * w, qop[j] * w, qcp[k]]);
            }
        }
    }
    for k in 0..p {
        let w = 1.0 - qop[k];
        for j in 1..p {
            for i in 1..p {
                pts.push([qcp[i] * w, qcp[j] * w, qop[k]]);
            }
        }
    }
    debug_assert_eq!(pts.len(), p * (3 * p * p + 5));
    pts
}

/// MFEM-faithful arbitrary-order Nédélec-I element on the reference pyramid
/// (1:1 port of MFEM `ND_FuentesPyramidElement(p)`).
///
/// The dof functionals are the nodal point values `σ_m(Φ) = Φ(x_m)·(J tk_m)`;
/// the fem-rs reference frame coincides with MFEM's, so the reference basis
/// *is* MFEM's.
pub struct PyraNDk {
    order: usize,
    nodes: Vec<[f64; 3]>,
    dof2tk: Vec<u8>,
    /// Row-major `Ti = T⁻¹` with `T(b, m) = u_b(x_m)·tk_{dof2tk[m]}`.
    ti: Vec<f64>,
}

pub type PyraND1 = PyraNDk;

impl PyraNDk {
    pub fn new(order: usize) -> Self {
        assert!(order >= 1, "PyraNDk: order >= 1");
        let nodes = pyra_nodes(order);
        let dof2tk = pyra_dof2tk(order);
        let dof = nodes.len();
        // Interpolation matrix: column m = raw basis at node m dotted with
        // the slot tangent (MFEM `T.GetColumn(m)`).
        let mut t = vec![0.0_f64; dof * dof];
        for (m, nd) in nodes.iter().enumerate() {
            let raw = calc_basis(order, nd[0], nd[1], nd[2]);
            let tm = PYRA_TK[dof2tk[m] as usize];
            for (b, ub) in raw.iter().enumerate() {
                t[b * dof + m] = ub[0] * tm[0] + ub[1] * tm[1] + ub[2] * tm[2];
            }
        }
        let ti = invert_dense(dof, &t, "PyraNDk");
        PyraNDk {
            order,
            nodes,
            dof2tk,
            ti,
        }
    }

    /// Reference tangents `tk[dof2tk[m]]` of every local DOF (MFEM's own
    /// frame = the fem-rs frame); the physical dual tangent is `J·tk`.
    pub fn dof_tangents(&self) -> Vec<[f64; 3]> {
        self.dof2tk
            .iter()
            .map(|&tk| PYRA_TK[tk as usize])
            .collect()
    }

    /// Reference points of the MFEM `ND_FuentesPyramidElement(p)` nodal
    /// layout (`FE::Nodes`) — the layout the HCurlSpace pyramid slot tables
    /// mirror (D525); derived from the same slot table as the basis, so the
    /// element cannot drift from its own DOF layout.
    pub fn mfem_layout_points(&self) -> Vec<[f64; 3]> {
        self.nodes.clone()
    }

    fn apply_ti(&self, u: &[[f64; 3]], out: &mut [f64]) {
        let dof = self.nodes.len();
        for j in 0..dof {
            let mut acc = [0.0_f64; 3];
            for (b, ub) in u.iter().enumerate() {
                let c = self.ti[j * dof + b];
                if c != 0.0 {
                    acc[0] += c * ub[0];
                    acc[1] += c * ub[1];
                    acc[2] += c * ub[2];
                }
            }
            out[j * 3] = acc[0];
            out[j * 3 + 1] = acc[1];
            out[j * 3 + 2] = acc[2];
        }
    }
}

impl VectorReferenceElement for PyraNDk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.order * (3 * self.order * self.order + 5)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let u = calc_basis(self.order, xi[0], xi[1], xi[2]);
        self.apply_ti(&u, values);
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let cu = calc_curl_basis(self.order, xi[0], xi[1], xi[2]);
        self.apply_ti(&cu, curl_vals);
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|p| p.to_vec()).collect()
    }
}

// MFEM 4.10 truth for the ND pyramid elements (golden dump generated by
// `tmp/d546/gen_dump.py` from `tmp/d546/d548_nd_prism_pyra_probe.cpp`).
#[cfg(test)]
mod mfem_dump {
    include!("pyramid/pyramid_mfem_dump.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MFEM `ND_FuentesPyramidElement` layout table: `p(3p²+5)` slots for
    /// every order (34 at p = 2), now the element's own dimension.
    #[test]
    fn mfem_layout_point_counts() {
        let want = |p: usize| p * (3 * p * p + 5);
        for p in 1..=4usize {
            assert_eq!(PyraNDk::new(p).n_dofs(), want(p));
            assert_eq!(PyraNDk::new(p).mfem_layout_points().len(), want(p));
        }
    }

    #[test]
    fn pyra_ndk_k1_n_dofs() {
        assert_eq!(PyraNDk::new(1).n_dofs(), 8);
    }
    #[test]
    fn pyra_ndk_k2_dim() {
        assert_eq!(PyraNDk::new(2).n_dofs(), 34);
    }
    #[test]
    fn pyra_ndk_k3_dim() {
        assert_eq!(PyraNDk::new(3).n_dofs(), 96);
    }

    #[test]
    fn pyra_ndk_basis_finite() {
        for k in [1usize, 2] {
            let ndk = PyraNDk::new(k);
            let mut v = vec![0.0; ndk.n_dofs() * 3];
            let qr = ndk.quadrature(3);
            for p in &qr.points {
                ndk.eval_basis_vec(p, &mut v);
                for x in &v {
                    assert!(x.is_finite(), "non-finite at {p:?}");
                }
            }
        }
    }

    #[test]
    fn pyra_ndk_curl_finite() {
        for k in [1usize, 2] {
            let ndk = PyraNDk::new(k);
            let mut c = vec![0.0; ndk.n_dofs() * 3];
            let qr = ndk.quadrature(3);
            for p in &qr.points {
                ndk.eval_curl(p, &mut c);
                for x in &c {
                    assert!(x.is_finite(), "non-finite curl at {p:?}");
                }
            }
        }
    }

    /// Per-slot MFEM 4.10 parity (`d548_out.txt` truth): for p = 1..3 the
    /// fem-rs basis and curl at the two probe points equal MFEM's
    /// `CalcVShape`/`CalcCurlShape` rows slot by slot (the frames coincide).
    #[test]
    fn pyra_ndk_matches_mfem_410_probe() {
        let pts = [[0.137, 0.413, 0.621], [0.71, 0.22, 0.53]];
        for p in 1..=3usize {
            let (ndofs, vc): (usize, [&[[f64; 6]]; 2]) = match p {
                1 => (
                    mfem_dump::NDOFS_1,
                    [&mfem_dump::VC_1_0, &mfem_dump::VC_1_1],
                ),
                2 => (
                    mfem_dump::NDOFS_2,
                    [&mfem_dump::VC_2_0, &mfem_dump::VC_2_1],
                ),
                _ => (
                    mfem_dump::NDOFS_3,
                    [&mfem_dump::VC_3_0, &mfem_dump::VC_3_1],
                ),
            };
            let e = PyraNDk::new(p);
            assert_eq!(e.n_dofs(), ndofs);
            let n = e.n_dofs();
            let mut v = vec![0.0_f64; n * 3];
            let mut c = vec![0.0_f64; n * 3];
            for (q, xi) in pts.iter().enumerate() {
                e.eval_basis_vec(xi, &mut v);
                e.eval_curl(xi, &mut c);
                let golden = vc[q];
                for i in 0..n {
                    for d in 0..3 {
                        assert!(
                            (v[i * 3 + d] - golden[i][d]).abs() < 5e-11,
                            "p={p} q={q} V[{i}][{d}]: {} vs {}",
                            v[i * 3 + d],
                            golden[i][d]
                        );
                        assert!(
                            (c[i * 3 + d] - golden[i][3 + d]).abs() < 5e-11,
                            "p={p} q={q} C[{i}][{d}]: {} vs {}",
                            c[i * 3 + d],
                            golden[i][3 + d]
                        );
                    }
                }
            }
        }
    }

    /// `FE::Nodes` and `dof2tk` parity: the layout equals the probe's node
    /// table slot by slot, and the reference tangents equal the probe's
    /// unit-vector `Project` tangents.
    #[test]
    fn pyra_ndk_nodes_and_tangents_match_mfem_410_probe() {
        for p in 1..=3usize {
            let (nodes_mfem, tk_mfem): (&[[f64; 3]], &[[f64; 3]]) = match p {
                1 => (&mfem_dump::NODES_1, &mfem_dump::TK_1),
                2 => (&mfem_dump::NODES_2, &mfem_dump::TK_2),
                _ => (&mfem_dump::NODES_3, &mfem_dump::TK_3),
            };
            let e = PyraNDk::new(p);
            let n = e.n_dofs();
            let nodes = e.mfem_layout_points();
            let tks = e.dof_tangents();
            for i in 0..n {
                for d in 0..3 {
                    assert!(
                        (nodes[i][d] - nodes_mfem[i][d]).abs() < 5e-14,
                        "p={p} NODES[{i}][{d}]"
                    );
                    assert!(
                        (tks[i][d] - tk_mfem[i][d]).abs() < 5e-13,
                        "p={p} TK[{i}][{d}]: {} vs {}",
                        tks[i][d],
                        tk_mfem[i][d]
                    );
                }
            }
        }
    }

    /// Nodal property: `σ_j(Φ_i) = Φ(x_j)·t̂_j = δ_ij` with the element's own
    /// `(dof_coords, dof_tangents)` — MFEM's `Project_ND` duals.
    #[test]
    fn pyra_ndk_dof_functionals_are_point_values() {
        for p in 1..=3usize {
            let e = PyraNDk::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tks = e.dof_tangents();
            let mut v = vec![0.0_f64; n * 3];
            for j in 0..n {
                e.eval_basis_vec(&coords[j], &mut v);
                for i in 0..n {
                    let s = v[i * 3] * tks[j][0]
                        + v[i * 3 + 1] * tks[j][1]
                        + v[i * 3 + 2] * tks[j][2];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-10,
                        "p={p}: σ_{j}(Φ_{i}) = {s} (want {want})"
                    );
                }
            }
        }
    }
}

