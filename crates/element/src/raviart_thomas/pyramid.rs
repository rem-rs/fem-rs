//! Raviart-Thomas H(div) element on the reference pyramid — a 1:1 port of
//! MFEM `RT_FuentesPyramidElement` (D445/D534).
//!
//! Reference pyramid (MFEM `Geometry::PYRAMID`, identical in fem-rs axes
//! `(xi,eta,zeta) = (x,y,z)`): base quad {0,1,2,3} on z=0 ([0,1]², CCW),
//! apex {4} = (0,0,1) (`fem/geom.cpp` `GeomVert[7]`).
//!
//! # Slot layout (D445)
//!
//! Slot order = MFEM `RT_FuentesPyramidElement::RT_FuentesPyramidElement`
//! construction order (`fem/fe/fe_rt.cpp:1273-1373`):
//!
//! ```text
//!   [ base quad ((p+1)^2, FaceVert (3,2,1,0)),
//!     tri (0,1,4) (eta=0), tri (1,2,4) (xi+zeta=1),
//!     tri (2,3,4) (eta+zeta=1), tri (3,0,4) (xi=0),
//!     each (p+1)(p+2)/2,
//!     interior 3p(p+1)^2  (x-, y-, z-component blocks) ]
//! ```
//!
//! Total `(p+1)(3p(p+2)+5)` (`fe_rt.cpp:1273-1275`): 5, 28, 87, 164 at
//! p = 0..3.  The base quad uses the MFEM canonical frame (`u` = c0→c1 =
//! vert 3→2 = +xi, `v` = c0→c3 = vert 3→0 = −eta, slot `n = v*(p+1)+u`);
//! the four triangular faces carry MFEM's **own** (heterogeneous) internal
//! enumerations verbatim — transposed lattice points on (0,1,4)/(1,2,4),
//! `i`-descending on (2,3,4)/(3,0,4), `fe_rt.cpp:1304-1330` — so the slot
//! order is now MFEM's *exactly* (D534; the previous port normalised the tri
//! faces to the standard barycentric grid).
//!
//! # Basis (D534)
//!
//! The basis functions are MFEM's Fuentes–Keith–Demkowicz expansion
//! (`fem/fe/fe_rt.cpp:1503-1760` `calcBasis`, building blocks
//! `fem/fe/fe_pyramid.cpp`): the raw vector fields `u_o` are made nodal
//! through the Vandermonde `T(m, o) = u_o(node_m) · nk_{dof2nk[m]}` and its
//! inverse (`fe_rt.cpp:1374-1396`, `Ti.Factor(T)` + `Ti.Mult(u, shape)`), so
//! the element's DOFs are MFEM's nodal flux samples
//! `dof_m = φ·(cof J n̂) at node_m` and `φ_m(node_l)·nk_l = δ_{ml}`.  The
//! raw helpers `E_E/V_Q/V_T/VT_T/E_Q/V_L/V_R` live here (RT-specific); the
//! scalar `mu`/`nu`/`phi_E`/`phi_Q` blocks are shared with the H¹ Fuentes
//! pyramid ([`crate::lagrange::pyramid_fuentes`]).
//!
//! # DOF functionals
//!
//! Nodal point samples (MFEM `Project_RT`), not moments — this is the
//! convention the space layer's `interpolate_vector` engine and the
//! prolongation builder (`HdivRt0Family::Pyramid`) pair against (D535).

use crate::lagrange::pyramid::calc_scaled_jacobi;
use crate::lagrange::pyramid_fuentes::{
    calc_scaled_legendre, grad_mu0_xy, grad_mu1_xy, grad_nu0, grad_nu1, mu0_xy, mu0_z, mu1_xy,
    mu1_z, nu0, nu1, nu2, phi_e_grad, phi_q_grad, GRAD_MU0_Z, GRAD_MU1_Z, GRAD_NU2,
};
use crate::quadrature::{gauss_lobatto_01, gauss_legendre_01, pyramid_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// MFEM `FuentesPyramid::apex_tol` (`fe_pyramid.hpp:88`): the `|z − 1|`
/// threshold below which `calcBasis` switches to its apex-limit values
/// `(x, y, z) = ((1−z)/2, (1−z)/2, z)`.
const APEX_TOL: f64 = 1e-8;

// ─── Small vector helpers (MFEM Vector/DenseMatrix idioms) ──────────────────

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// MFEM `add(a, A, b, B, out)`: `out = a·A + b·B`.
fn add_scaled(a: f64, va: [f64; 3], b: f64, vb: [f64; 3]) -> [f64; 3] {
    [
        a * va[0] + b * vb[0],
        a * va[1] + b * vb[1],
        a * va[2] + b * vb[2],
    ]
}

/// MFEM `mu01_grad_mu01(z, xy, ab)` (`fe_pyramid.cpp:208`):
/// `mu0·∇mu1 − mu1·∇mu0` — the face-lattice `s ∧ ∇s` vector.
fn mu01_grad_mu01(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    add_scaled(
        mu0_xy(z, xy, ab),
        grad_mu1_xy(z, xy, ab),
        -mu1_xy(z, xy, ab),
        grad_mu0_xy(z, xy, ab),
    )
}

/// MFEM `nu01_grad_nu01(z, xy, ab)` (`fe_pyramid.cpp:259`):
/// `nu0·∇nu1 − nu1·∇nu0`.
fn nu01_grad_nu01(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    add_scaled(nu0(z, xy, ab), grad_nu1(ab), -nu1(xy, ab), grad_nu0(ab))
}

/// MFEM `nu012_grad_nu012(z, xy, ab)` (`fe_pyramid.cpp:275`):
/// `nu0(∇nu1×∇nu2) + nu1(∇nu2×∇nu0) + nu2(∇nu0×∇nu1)`.
fn nu012_grad_nu012(z: f64, xy: [f64; 2], ab: usize) -> [f64; 3] {
    let g0 = grad_nu0(ab);
    let g1 = grad_nu1(ab);
    let v01 = cross3(g0, g1);
    let v12 = cross3(g1, GRAD_NU2);
    let v20 = cross3(GRAD_NU2, g0);
    add_scaled(
        1.0,
        add_scaled(nu0(z, xy, ab), v12, nu1(xy, ab), v20),
        nu2(z),
        v01,
    )
}

// ─── Raw building blocks (fe_pyramid.cpp) ───────────────────────────────────

/// MFEM `CalcHomogenizedScaLegendre(p, s0, s1, u)` values
/// (`fe_pyramid.cpp:447` = `CalcScaledLegendre(p, s1, s0+s1, u)`).
fn hom_sca_legendre(p: usize, s0: f64, s1: f64) -> Vec<f64> {
    calc_scaled_legendre(p, s1, s0 + s1).0
}

/// MFEM `CalcHomogenizedScaJacobi(p, α, t0, t1, u)` values
/// (`fe_pyramid.cpp:666` = `CalcScaledJacobi(p, α, t1, t0+t1, u)`).
fn hom_sca_jacobi(p: usize, alpha: f64, t0: f64, t1: f64) -> Vec<f64> {
    calc_scaled_jacobi(p, alpha, t1, t0 + t1).0
}

/// MFEM `E_E(p, s, sds, u)` (`fe_pyramid.cpp:960`): the `p×3` edge block
/// `u(i,:) = P_i(s)·sds`, `P` the homogenised scaled Legendre of `s`.
fn e_e(p: usize, s: [f64; 2], sds: [f64; 3]) -> Vec<[f64; 3]> {
    let pi = hom_sca_legendre(p - 1, s[0], s[1]);
    (0..p).map(|i| [pi[i] * sds[0], pi[i] * sds[1], pi[i] * sds[2]]).collect()
}

/// MFEM `E_E(p, s, grad_s, u, curl_u)` (`fe_pyramid.cpp:983`): values plus
/// curls `curl_u(i,:) = (i+2)·P_i(s)·(∇s0×∇s1)`.
fn e_e_curl(p: usize, s: [f64; 2], grad_s: [[f64; 3]; 2]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let pi = hom_sca_legendre(p - 1, s[0], s[1]);
    let sds = add_scaled(s[0], grad_s[1], -s[1], grad_s[0]);
    let dsxds = cross3(grad_s[0], grad_s[1]);
    let u: Vec<[f64; 3]> =
        (0..p).map(|i| [pi[i] * sds[0], pi[i] * sds[1], pi[i] * sds[2]]).collect();
    let cu: Vec<[f64; 3]> = (0..p)
        .map(|i| {
            let c = (i as f64 + 2.0) * pi[i];
            [c * dsxds[0], c * dsxds[1], c * dsxds[2]]
        })
        .collect();
    (u, cu)
}

/// MFEM `V_Q(p, s, sds, t, tdt, u)` (`fe_pyramid.cpp:1238`): the quadrilateral
/// block `u(i,j,:) = E_i(i) × E_j(j)`, flat `p×p×3`.
fn v_q(p: usize, s: [f64; 2], sds: [f64; 3], t: [f64; 2], tdt: [f64; 3]) -> Vec<f64> {
    let ei = e_e(p, s, sds);
    let ej = e_e(p, t, tdt);
    let mut u = vec![0.0; p * p * 3];
    for j in 0..p {
        for i in 0..p {
            let c = cross3(ei[i], ej[j]);
            let m = (i * p + j) * 3;
            u[m..m + 3].copy_from_slice(&c);
        }
    }
    u
}

/// MFEM `V_T(p, s, sdsxds, u)` (`fe_pyramid.cpp:1272`): the triangular-face
/// block `u(i,j,:) = P_i(s)·J_j(s)·sdsxds` for `i+j < p`, flat `p×p×3`.
fn v_t(p: usize, s: [f64; 3], sdsxds: [f64; 3]) -> Vec<f64> {
    let pi = hom_sca_legendre(p - 1, s[0], s[1]);
    let mut u = vec![0.0; p * p * 3];
    for i in 0..p {
        let alpha = 2.0 * i as f64 + 1.0;
        // HomScaJacobi(p-1, α, t0 = s0+s1, t1 = s2)
        let pj = hom_sca_jacobi(p - 1, alpha, s[0] + s[1], s[2]);
        for j in 0..(p - i) {
            let vij = pi[i] * pj[j];
            let m = (i * p + j) * 3;
            for k in 0..3 {
                u[m + k] = vij * sdsxds[k];
            }
        }
    }
    u
}

/// MFEM `VT_T(p, s, sds, sdsxds, mu, grad_mu, u)` (`fe_pyramid.cpp:1345`):
/// the homogenised triangular-face block, flat `p×p×3` (`i+j < p`).
fn vt_t(
    p: usize,
    s: [f64; 3],
    sds: [f64; 3],
    sdsxds: [f64; 3],
    mu: f64,
    grad_mu: [f64; 3],
) -> Vec<f64> {
    let ms = [mu * s[0], mu * s[1], s[2]];
    let pi = hom_sca_legendre(p - 1, ms[0], ms[1]);
    // E_E(1, s2, sds) with s2 = (s0, s1): P_0 = 1, so EE0[0] = sds (MFEM
    // passes the 1×3 matrix through `E_E` only to reuse the helper).
    let ee0 = e_e(1, [s[0], s[1]], sds);
    let dmuxee = cross3(grad_mu, ee0[0]);
    // V_T(1, s, sdsxds)(0,0,:): both P_0 and J_0 are 1, so VT00 = sdsxds.
    let vt00 = v_t(1, s, sdsxds);
    let mut u = vec![0.0; p * p * 3];
    for i in 0..p {
        let jj = hom_sca_jacobi(p - i - 1, 2.0 * i as f64 + 1.0, ms[0] + ms[1], ms[2]);
        for j in 0..(p - i) {
            for k in 0..3 {
                u[(i * p + j) * 3 + k] =
                    pi[i] * jj[j] * (mu * vt00[k] + s[2] * dmuxee[k]);
            }
        }
    }
    u
}

/// MFEM `E_Q(p, s, grad_s, t, grad_t, u, curl_u)` (`fe_pyramid.cpp:1066`):
/// the interior edge-face block `u(i,j,:) = phi_E_j(j)·E_E_i(i,:)` with its
/// curl, for `j ≥ 2, i < p`, flat `p×(p+1)×3` (values first, then curls).
fn e_q_curl(
    p: usize,
    s: [f64; 2],
    grad_s: [[f64; 3]; 2],
    t: [f64; 2],
    grad_t: [[f64; 3]; 2],
) -> (Vec<f64>, Vec<f64>) {
    let (pe, dpe) = phi_e_grad(p, t, grad_t);
    let (ei, dei) = e_e_curl(p, s, grad_s);
    let mut u = vec![0.0; p * (p + 1) * 3];
    let mut cu = vec![0.0; p * (p + 1) * 3];
    for j in 2..=p {
        for i in 0..p {
            let m = (i * (p + 1) + j) * 3;
            for k in 0..3 {
                u[m + k] = pe[j] * ei[i][k];
            }
            cu[m] = pe[j] * dei[i][0] + dpe[j][1] * ei[i][2] - dpe[j][2] * ei[i][1];
            cu[m + 1] = pe[j] * dei[i][1] + dpe[j][2] * ei[i][0] - dpe[j][0] * ei[i][2];
            cu[m + 2] = pe[j] * dei[i][2] + dpe[j][0] * ei[i][1] - dpe[j][1] * ei[i][0];
        }
    }
    (u, cu)
}

/// MFEM `V_L(p, sx, grad_sx, sy, grad_sy, t, grad_t, u)` (`fe_pyramid.cpp:1469`):
/// Fuentes' `V^unlhd_{ij}`, flat `(p+1)×(p+1)×3` (filled for `i, j ≥ 2`).
#[allow(clippy::too_many_arguments)]
fn v_l(
    p: usize,
    sx: [f64; 2],
    grad_sx: [[f64; 3]; 2],
    sy: [f64; 2],
    grad_sy: [[f64; 3]; 2],
    t: f64,
    grad_t: [f64; 3],
) -> Vec<f64> {
    let (ei, dei) = phi_e_grad(p, sx, grad_sx);
    let (ej, dej) = phi_e_grad(p, sy, grad_sy);
    let mut u = vec![0.0; (p + 1) * (p + 1) * 3];
    for j in 2..=p {
        for i in 2..=p {
            let dphidphi = cross3(dei[i], dej[j]);
            let phidphi = add_scaled(ei[i], dej[j], -ej[j], dei[i]);
            let dtphidphi = cross3(grad_t, phidphi);
            let m = (i * (p + 1) + j) * 3;
            for l in 0..3 {
                u[m + l] = t * (t * dphidphi[l] + dtphidphi[l]);
            }
        }
    }
    u
}

/// MFEM `V_R(p, s, grad_s, mu, dmu, t, dt, u)` (`fe_pyramid.cpp:1538`):
/// Fuentes' `V^unrhd_i`, `(p+1)×3` (rows 0, 1 zero; filled for `i ≥ 2`).
fn v_r(
    p: usize,
    s: [f64; 2],
    grad_s: [[f64; 3]; 2],
    mu: f64,
    dmu: [f64; 3],
    t: f64,
    dt: [f64; 3],
) -> Vec<[f64; 3]> {
    let _ = mu; // MFEM's V_R signature carries `mu` but its body evaluates
                // `phi_E(p, s, grad_s)` on the raw `s` only — kept 1:1.
    let (ei, dei) = phi_e_grad(p, s, grad_s);
    let mut u = vec![[0.0; 3]; p + 1];
    for i in 2..=p {
        let dphit2 = add_scaled(t * t, dei[i], 2.0 * t * ei[i], dt);
        u[i] = cross3(dphit2, dmu);
    }
    u
}

// ─── calcBasis (fe_rt.cpp:1503-1760) ────────────────────────────────────────

/// Fuentes interior dof count: `3p(p+1)²` (`fe_rt.cpp:1343-1373` loops;
/// `RT_dof[PYRAMID] = 3p(p+1)²`, `fe_coll.cpp:2584-2586`).
fn fuentes_interior_dofs(p: usize) -> usize {
    3 * p * (p + 1) * (p + 1)
}

/// Total dof count of MFEM `RT_FuentesPyramidElement(p)`:
/// `(p+1)(3p(p+2)+5)` (probe: 28 @ p=1, 87 @ p=2).
fn pyramid_rtk_dim(p: usize) -> usize {
    let tri = (p + 1) * (p + 2) / 2;
    (p + 1) * (p + 1) + 4 * tri + fuentes_interior_dofs(p)
}

/// MFEM `RT_FuentesPyramidElement::calcBasis` (`fe_rt.cpp:1503-1760`) — the
/// raw Fuentes expansion at `(x, y, z)` in the element's RAW slot order
/// (quad face, 4 tri blocks, interior families I–VII).  `p` is the value
/// MFEM passes in (`order` = the constructor's `p + 1`); `u` has exactly
/// [`pyramid_rtk_dim`]`(p - 1)` entries.
fn fuentes_rt_raw_basis(p: usize, x_in: f64, y_in: f64, z_in: f64, f: &mut [[f64; 3]]) {
    let mut x = x_in;
    let mut y = y_in;
    let mut z = z_in;
    let mut xy = [x, y];
    if (1.0 - z).abs() < APEX_TOL {
        z = 1.0 - APEX_TOL;
        y = 0.5 * (1.0 - z);
        x = 0.5 * (1.0 - z);
        xy = [x, y];
    }

    f.fill([0.0; 3]);
    let mut o = 0usize;

    // Quadrilateral face
    if z < 1.0 {
        let vq = v_q(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            mu01_grad_mu01(z, xy, 1),
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            mu01_grad_mu01(z, xy, 2),
        );
        let muz3 = mu0_z(z).powi(3);
        for j in 0..p {
            for i in 0..p {
                let m = (i * p + j) * 3;
                f[o] = [muz3 * vq[m], muz3 * vq[m + 1], muz3 * vq[m + 2]];
                o += 1;
            }
        }
    }

    // Triangular faces
    if z < 1.0 {
        // (a,b) = (1,2), c = 0 then c = 1 (V_T shared between the two)
        let s_nu = [nu0(z, xy, 1), nu1(xy, 1), nu2(z)];
        let s_gnu = nu012_grad_nu012(z, xy, 1);
        let vt = v_t(p, s_nu, s_gnu);
        let vtt_sd = nu01_grad_nu01(z, xy, 1);
        for (mu, dmuz) in [
            (mu0_xy(z, xy, 2), grad_mu0_xy(z, xy, 2)),
            (mu1_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)),
        ] {
            let vtt = vt_t(p, s_nu, vtt_sd, s_gnu, mu, dmuz);
            for j in 0..p {
                for i in 0..(p - j) {
                    let m = (i * p + j) * 3;
                    for k in 0..3 {
                        f[o][k] = 0.5 * (mu * vt[m + k] + vtt[m + k]);
                    }
                    o += 1;
                }
            }
        }

        // (a,b) = (2,1), c = 0 then c = 1
        let s_nu = [nu0(z, xy, 2), nu1(xy, 2), nu2(z)];
        let s_gnu = nu012_grad_nu012(z, xy, 2);
        let vt = v_t(p, s_nu, s_gnu);
        let vtt_sd = nu01_grad_nu01(z, xy, 2);
        for (mu, dmuz) in [
            (mu0_xy(z, xy, 1), grad_mu0_xy(z, xy, 1)),
            (mu1_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)),
        ] {
            let vtt = vt_t(p, s_nu, vtt_sd, s_gnu, mu, dmuz);
            for j in 0..p {
                for i in 0..(p - j) {
                    let m = (i * p + j) * 3;
                    for k in 0..3 {
                        f[o][k] = 0.5 * (mu * vt[m + k] + vtt[m + k]);
                    }
                    o += 1;
                }
            }
        }
    }

    // Interior families I–VII (all p >= 2)
    if z < 1.0 && p >= 2 {
        // Family I (+ the shared phi_E evaluation reused by II and IV)
        let (e1, de1) = e_q_curl(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
        );
        let (phi_k, dphi_k) = phi_e_grad(p, [mu0_z(z), mu1_z(z)], [GRAD_MU0_Z, GRAD_MU1_Z]);
        let muz = mu0_z(z);
        // dmuphi = muz·∇phi_k(k) + (∇mu0(z))·phi_k(k), the MFEM line
        //   dmuphi(i) = muz*dphi_k(k,i) + dmuz(i)*phi_k(k)
        // with dmuz = grad_mu0(z) = (0, 0, −1).
        let dmuphi_f1 = |k: usize| {
            [
                muz * dphi_k[k][0] + GRAD_MU0_Z[0] * phi_k[k],
                muz * dphi_k[k][1] + GRAD_MU0_Z[1] * phi_k[k],
                muz * dphi_k[k][2] + GRAD_MU0_Z[2] * phi_k[k],
            ]
        };
        for k in 2..=p {
            let dmuphi = dmuphi_f1(k);
            for j in 2..=p {
                for i in 0..p {
                    let m = (i * (p + 1) + j) * 3;
                    let e_ij = [e1[m], e1[m + 1], e1[m + 2]];
                    let v = cross3(dmuphi, e_ij);
                    for l in 0..3 {
                        f[o][l] = muz * phi_k[k] * de1[m + l] + v[l];
                    }
                    o += 1;
                }
            }
        }

        // Family II (s/t swapped; phi_E reused from Family I)
        let (e2, de2) = e_q_curl(
            p,
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
        );
        let muz = mu0_z(z);
        for k in 2..=p {
            let dmuphi = dmuphi_f1(k);
            for j in 2..=p {
                for i in 0..p {
                    let m = (i * (p + 1) + j) * 3;
                    let e_ij = [e2[m], e2[m + 1], e2[m + 2]];
                    let v = cross3(dmuphi, e_ij);
                    for l in 0..3 {
                        f[o][l] = muz * phi_k[k] * de2[m + l] + v[l];
                    }
                    o += 1;
                }
            }
        }

        // Family III
        let (_phi_ij, dphi_ijk) = phi_q_grad(
            p,
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
        );
        let muz = mu0_z(z);
        for j in 2..=p {
            for i in 2..=p {
                let n = i.max(j);
                let nmu = n as f64 * muz.powf((n - 1) as f64);
                let d = dphi_ijk[i * (p + 1) + j];
                f[o] = [
                    nmu * (d[1] * GRAD_MU0_Z[2] - d[2] * GRAD_MU0_Z[1]),
                    nmu * (d[2] * GRAD_MU0_Z[0] - d[0] * GRAD_MU0_Z[2]),
                    nmu * (d[0] * GRAD_MU0_Z[1] - d[1] * GRAD_MU0_Z[0]),
                ];
                o += 1;
            }
        }

        // Family IV (reuses V_Q from the quadrilateral face)
        let vq = v_q(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            mu01_grad_mu01(z, xy, 1),
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            mu01_grad_mu01(z, xy, 2),
        );
        let muz2 = mu0_z(z).powi(2);
        for k in 2..=p {
            for j in 0..p {
                for i in 0..p {
                    let m = (i * p + j) * 3;
                    for l in 0..3 {
                        f[o][l] = muz2 * vq[m + l] * phi_k[k];
                    }
                    o += 1;
                }
            }
        }

        // Family V
        let vl = v_l(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
            mu0_z(z),
            GRAD_MU0_Z,
        );
        let muz = mu1_z(z);
        for j in 2..=p {
            for i in 2..=p {
                let n = i.max(j);
                let muzi = muz.powf((n - 1) as f64);
                let m = (i * (p + 1) + j) * 3;
                f[o] = [muzi * vl[m], muzi * vl[m + 1], muzi * vl[m + 2]];
                o += 1;
            }
        }

        // Family VI
        let vr = v_r(
            p,
            [mu0_xy(z, xy, 1), mu1_xy(z, xy, 1)],
            [grad_mu0_xy(z, xy, 1), grad_mu1_xy(z, xy, 1)],
            mu1_xy(z, xy, 2),
            grad_mu1_xy(z, xy, 2),
            mu0_z(z),
            GRAD_MU0_Z,
        );
        let muz = mu1_z(z);
        for i in 2..=p {
            let muzi = muz.powf((i - 1) as f64);
            f[o] = [muzi * vr[i][0], muzi * vr[i][1], muzi * vr[i][2]];
            o += 1;
        }

        // Family VII
        let vr = v_r(
            p,
            [mu0_xy(z, xy, 2), mu1_xy(z, xy, 2)],
            [grad_mu0_xy(z, xy, 2), grad_mu1_xy(z, xy, 2)],
            mu1_xy(z, xy, 1),
            grad_mu1_xy(z, xy, 1),
            mu0_z(z),
            GRAD_MU0_Z,
        );
        let muz = mu1_z(z);
        for i in 2..=p {
            let muzi = muz.powf((i - 1) as f64);
            f[o] = [muzi * vr[i][0], muzi * vr[i][1], muzi * vr[i][2]];
            o += 1;
        }
    }

    debug_assert_eq!(o, pyramid_rtk_dim(p - 1), "raw basis slot count for p={p}");
}

// ─── Nodes, dof→nk map, nk table (fe_rt.cpp:1276-1373) ──────────────────────

/// MFEM `RT_FuentesPyramidElement::nk[24]` (`fe_rt.cpp:1270-1272`) — the
/// reference normal of each dof class (0..4 = the five faces, 5..7 = the
/// interior x/y/z component samples; the triangular-face normals are
/// unnormalised).
fn nk_of(class: u8) -> [f64; 3] {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    const SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    match class {
        0 => [0.0, 0.0, -1.0],
        1 => [0.0, -1.0, 0.0],
        2 => [1.0, 0.0, 1.0],
        3 => [0.0, 1.0, 1.0],
        4 => [-1.0, 0.0, 0.0],
        5 => [SQRT2, 0.0, SQRT1_2],
        6 => [0.0, SQRT2, SQRT1_2],
        _ => [0.0, 0.0, 1.0],
    }
}

/// The `dof2nk` class of element slot `m` at constructor order `p`
/// (`fe_rt.cpp:1276-1373`): base quad → 0, tri faces → 1..4, interior
/// x/y/z blocks → 5..7.
fn dof2nk_slot(m: usize, p: usize) -> u8 {
    let quad = (p + 1) * (p + 1);
    let tri = (p + 1) * (p + 2) / 2;
    if m < quad {
        return 0;
    }
    let m = m - quad;
    if m < 4 * tri {
        return 1 + (m / tri) as u8;
    }
    let m = m - 4 * tri;
    let x_block = p * (p + 1) * (p + 1);
    if m < x_block {
        return 5;
    }
    if m < 2 * x_block {
        return 6;
    }
    7
}

/// Node table of MFEM `RT_FuentesPyramidElement(p)` in the element's slot
/// order (`fe_rt.cpp:1276-1373`), with `bop`/`iop` the `p+1` open
/// Gauss–Legendre points and `icp` the `p+2` closed Gauss–Lobatto points on
/// `[0,1]` (`Poly_1D::OpenPoints` / `ClosedPoints`).
fn fuentes_rt_nodes(p: usize) -> Vec<[f64; 3]> {
    let bop = gauss_legendre_01(p + 1).0;
    let icp = gauss_lobatto_01(p + 2).0;
    let mut n = Vec::with_capacity(pyramid_rtk_dim(p));

    // quadrilateral face (3,2,1,0): node (bop[i], bop[p-j], 0)
    for j in 0..=p {
        for i in 0..=p {
            n.push([bop[i], bop[p - j], 0.0]);
        }
    }
    // triangular faces — MFEM's own (heterogeneous) enumerations
    for j in 0..=p {
        for i in 0..=(p - j) {
            let w = bop[i] + bop[j] + bop[p - i - j];
            n.push([bop[i] / w, 0.0, bop[j] / w]); // (0,1,4)
        }
    }
    for j in 0..=p {
        for i in 0..=(p - j) {
            let w = bop[i] + bop[j] + bop[p - i - j];
            n.push([1.0 - bop[j] / w, bop[i] / w, bop[j] / w]); // (1,2,4)
        }
    }
    for j in 0..=p {
        for i in (0..=(p - j)).rev() {
            let w = bop[i] + bop[j] + bop[p - i - j];
            n.push([bop[i] / w, 1.0 - bop[j] / w, bop[j] / w]); // (2,3,4)
        }
    }
    for j in 0..=p {
        for i in (0..=(p - j)).rev() {
            let w = bop[i] + bop[j] + bop[p - i - j];
            n.push([0.0, bop[i] / w, bop[j] / w]); // (3,0,4)
        }
    }
    // interior x / y / z component samples
    for k in 0..=p {
        for j in 0..=p {
            for i in 1..=p {
                let w = 1.0 - bop[k];
                n.push([icp[i] * w, bop[j] * w, bop[k]]);
            }
        }
    }
    for k in 0..=p {
        for j in 1..=p {
            for i in 0..=p {
                let w = 1.0 - bop[k];
                n.push([bop[i] * w, icp[j] * w, bop[k]]);
            }
        }
    }
    for k in 1..=p {
        for j in 0..=p {
            for i in 0..=p {
                let w = 1.0 - icp[k];
                n.push([bop[i] * w, bop[j] * w, icp[k]]);
            }
        }
    }
    n
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Raviart-Thomas H(div) element on the reference pyramid — a 1:1 port of
/// MFEM `RT_FuentesPyramidElement(p)` (D445 slot layout, D534 Fuentes basis):
/// `(p+1)(3p(p+2)+5)` DOFs, nodal flux-sample functionals
/// `dof_m = φ(node_m)·(cof J n̂_m)`, `φ_m(node_l)·nk_l = δ_{ml}`.
pub struct PyraRTk {
    p: usize,
    n: usize,
    nodes: Vec<[f64; 3]>,
    /// `φ_m = Σ_o ti[m·n + o] · u_o(x)` — row-major `T⁻¹` of the Fuentes
    /// Vandermonde `T(m, o) = u_o(node_m)·nk_{dof2nk[m]}` (MFEM `Ti`).
    ti: Vec<f64>,
}

/// Order-0 element (alias for `PyraRTk::new(0)`, kept for backward compat).
pub type PyraRT0 = PyraRTk;

impl PyraRTk {
    pub fn new(order: usize) -> Self {
        let n = pyramid_rtk_dim(order);
        let nodes = fuentes_rt_nodes(order);
        let mut u = vec![[0.0; 3]; n];
        let mut t = nalgebra::DMatrix::<f64>::zeros(n, n);
        for (m, node) in nodes.iter().enumerate() {
            fuentes_rt_raw_basis(order + 1, node[0], node[1], node[2], &mut u);
            let nm = nk_of(dof2nk_slot(m, order));
            for (o, uv) in u.iter().enumerate() {
                // MFEM `u.Mult(nm, T.GetColumn(m))`: T's COLUMN m is the raw
                // expansion sampled at node m against the dof's normal —
                // rows = raw functions, columns = nodes.  With Ti = T⁻¹ the
                // nodal basis phi_m = sum_o Ti(m,o) u_o satisfies
                // phi_m(node_l)·nk_l = (Ti·T)(m,l) = delta_ml.
                t[(o, m)] = uv[0] * nm[0] + uv[1] * nm[1] + uv[2] * nm[2];
            }
        }
        let ti_m = t
            .try_inverse()
            .expect("PyraRTk: singular Fuentes Vandermonde matrix");
        let mut ti = vec![0.0; n * n];
        for m in 0..n {
            for o in 0..n {
                ti[m * n + o] = ti_m[(m, o)];
            }
        }
        PyraRTk { p: order, n, nodes, ti }
    }

    /// Half-open slot range of face `face` (0..5; 5 = interior) in the
    /// element's slot layout — `[base, tri 0,1,4 / 1,2,4 / 2,3,4 / 3,0,4,
    /// interior]`.  Public for MFEM-layout parity tests.
    pub fn slot_range(&self, face: usize) -> std::ops::Range<usize> {
        let tri = (self.p + 1) * (self.p + 2) / 2;
        let quad = (self.p + 1) * (self.p + 1);
        let start = match face {
            0 => 0,
            1 => quad,
            2 => quad + tri,
            3 => quad + 2 * tri,
            4 => quad + 3 * tri,
            _ => quad + 4 * tri,
        };
        let len = match face {
            0 => quad,
            1 | 2 | 3 | 4 => tri,
            _ => fuentes_interior_dofs(self.p),
        };
        start..start + len
    }
}

impl VectorReferenceElement for PyraRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.p as u8
    }
    fn n_dofs(&self) -> usize {
        self.n
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let mut u = vec![[0.0; 3]; self.n];
        fuentes_rt_raw_basis(self.p + 1, xi[0], xi[1], xi[2], &mut u);
        for m in 0..self.n {
            let mut acc = [0.0; 3];
            for o in 0..self.n {
                let c = self.ti[m * self.n + o];
                if c != 0.0 {
                    acc[0] += c * u[o][0];
                    acc[1] += c * u[o][1];
                    acc[2] += c * u[o][2];
                }
            }
            values[m * 3..m * 3 + 3].copy_from_slice(&acc);
        }
    }

    fn eval_curl(&self, xi: &[f64], cv: &mut [f64]) {
        let h = 1e-6;
        let n3 = self.n * 3;
        let mut vp = vec![0.0; n3];
        let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0] + h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0] - h, xi[1], xi[2]], &mut vm);
            let dfy_dx = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            let dfz_dx = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1] + h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1] - h, xi[2]], &mut vm);
            let dfx_dy = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            let dfz_dy = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] + h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] - h], &mut vm);
            let dfx_dz = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            let dfy_dz = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            cv[i * 3] = dfz_dy - dfy_dz;
            cv[i * 3 + 1] = dfx_dz - dfz_dx;
            cv[i * 3 + 2] = dfy_dx - dfx_dy;
        }
    }

    fn eval_div(&self, xi: &[f64], dv: &mut [f64]) {
        let h = 1e-6;
        let n3 = self.n * 3;
        let mut vp = vec![0.0; n3];
        let mut vm = vec![0.0; n3];
        for i in 0..self.n {
            self.eval_basis_vec(&[xi[0] + h, xi[1], xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0] - h, xi[1], xi[2]], &mut vm);
            let dfx = (vp[i * 3] - vm[i * 3]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1] + h, xi[2]], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1] - h, xi[2]], &mut vm);
            let dfy = (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] + h], &mut vp);
            self.eval_basis_vec(&[xi[0], xi[1], xi[2] - h], &mut vm);
            let dfz = (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
            dv[i] = dfx + dfy + dfz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    /// MFEM `RT_FuentesPyramidElement` node positions in slot order
    /// (`fem/fe/fe_rt.cpp:1276-1373`; probe `tmp/d534/d534_fuentes_basis.txt`
    /// NODE lines) — fem-rs axes equal MFEM's `(x,y,z)`.
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// MFEM `RT_FuentesPyramidElement(p)` dof counts — probe
    /// `tmp/d444/probe_d444.out` (`dof=28` @ p=1, `dof=87` @ p=2) and the
    /// closed formula `(p+1)(3p(p+2)+5)` (`fe_rt.cpp:1273-1275`).
    #[test]
    fn pyra_rtk_dim_matches_mfem_fuentes() {
        assert_eq!(pyramid_rtk_dim(0), 5);
        assert_eq!(pyramid_rtk_dim(1), 28);
        assert_eq!(pyramid_rtk_dim(2), 87);
        assert_eq!(pyramid_rtk_dim(3), 200);
        assert_eq!(PyraRTk::new(1).n_dofs(), 28);
        assert_eq!(PyraRTk::new(2).n_dofs(), 87);
    }

    /// Fuentes interior count `3p(p+1)²` (`RT_dof[PYRAMID]`,
    /// `fe_coll.cpp:2584-2586`): 0, 12, 54, 108 at p = 0..3.
    #[test]
    fn fuentes_interior_counts_match_mfem() {
        assert_eq!(fuentes_interior_dofs(0), 0);
        assert_eq!(fuentes_interior_dofs(1), 12);
        assert_eq!(fuentes_interior_dofs(2), 54);
        assert_eq!(fuentes_interior_dofs(3), 144);
    }

    /// Point on reference face `h` with frame coords (u, v) and the outward
    /// normal / surface element.
    fn face_point(h: usize, u: f64, v: f64) -> ([f64; 3], [f64; 3], f64) {
        match h {
            0 => ([u, 1.0 - v, 0.0], [0.0, 0.0, -1.0], 1.0),
            1 => ([u, 0.0, v], [0.0, -1.0, 0.0], 1.0),
            2 => ([1.0 - v, u, v], [1.0, 0.0, 1.0], std::f64::consts::SQRT_2),
            3 => (
                [1.0 - u - v, 1.0 - v, v],
                [0.0, 1.0, 1.0],
                std::f64::consts::SQRT_2,
            ),
            _ => ([0.0, 1.0 - u - v, v], [-1.0, 0.0, 0.0], 1.0),
        }
    }

    /// Exact normal-flux moments `∫_h (φ_slot·n̂_h)·q dA`; `q` runs over
    /// {1, u, v} (the RT1 dof functional set — the foreign-face vanishing
    /// set) plus {u², uv, v²} for the own-face nonzero check.
    fn face_flux_moments(e: &PyraRTk, slot: usize, h: usize) -> Vec<f64> {
        let mut phi = vec![0.0_f64; e.n_dofs() * 3];
        let mut mom = vec![0.0_f64; 6];
        let tests: [fn(f64, f64) -> f64; 6] = [
            |_, _| 1.0,
            |u, _| u,
            |_, v| v,
            |u, _| u * u,
            |u, v| u * v,
            |_, v| v * v,
        ];
        let rule = if h == 0 {
            crate::quadrature::quad_rule_01(12)
        } else {
            crate::quadrature::tri_rule(12)
        };
        for (pt, &w) in rule.points.iter().zip(rule.weights.iter()) {
            let (x, n, ds) = face_point(h, pt[0], pt[1]);
            e.eval_basis_vec(&x, &mut phi);
            for (t, tc) in tests.iter().enumerate() {
                let q = tc(pt[0], pt[1]);
                for c in 0..3 {
                    mom[t] += w * ds * q * phi[slot * 3 + c] * n[c];
                }
            }
        }
        mom
    }

    /// D445 core property: every face group's basis functions carry
    /// vanishing normal-flux moments (RT1 functional set) on all *other*
    /// faces and a nonzero own-face moment; interior basis functions vanish
    /// on all five faces.  This is the slot ↔ face agreement the (future)
    /// pyramid RT1 assembly pairing needs — the original D445 lesion had
    /// element slots [base, x=0, y=0, x+z, y+z] against the space's
    /// [4 tris, base] (and now Fuentes [base, 4 tris]).
    #[test]
    fn pyra_rtk1_slot_groups_are_face_conforming() {
        let e = PyraRTk::new(1);
        assert_eq!(e.n_dofs(), 28);
        for g in 0..5 {
            let range = e.slot_range(g);
            assert_eq!(range.len(), if g == 0 { 4 } else { 3 });
            for slot in range.clone() {
                for h in 0..5 {
                    let mom = face_flux_moments(&e, slot, h);
                    if h == g {
                        let max = mom.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                        assert!(
                            max > 1e-8,
                            "slot {slot} (face {g}): own-face moments vanish: {mom:?}"
                        );
                    } else {
                        for (t, &mv) in mom.iter().take(3).enumerate() {
                            assert!(
                                mv.abs() < 1e-7,
                                "slot {slot} (face {g}): moment {t} on foreign face {h} = {mv}"
                            );
                        }
                    }
                }
            }
        }
        for slot in e.slot_range(5) {
            for h in 0..5 {
                let mom = face_flux_moments(&e, slot, h);
                for (t, &mv) in mom.iter().take(3).enumerate() {
                    assert!(
                        mv.abs() < 1e-7,
                        "interior slot {slot}: moment {t} on face {h} = {mv}"
                    );
                }
            }
        }
    }

    /// `dof_coords` reproduce the MFEM `RT_FuentesPyramidElement(1)` Nodes
    /// table (probe `tmp/d444/probe_d444.out`) as a sorted multiset — with
    /// the D534 port the enumeration itself is MFEM's (probe
    /// `tmp/d534/d534_fuentes_basis.txt` NODE lines pin it slot-by-slot).
    #[test]
    fn pyra_rtk1_dof_coords_match_mfem_probe() {
        // MFEM Fuentes(1) node table, fem-rs axes == MFEM (x,y,z)
        // (values %.17e from tmp/d444/probe_d444.out lines 28 dof block).
        const A: f64 = 0.211324865405187107;
        const B: f64 = 0.788675134594812866;
        const C: f64 = 0.174457630187009438; // bop/w lattice low
        const D: f64 = 0.651084739625981124; // bop/w lattice high
        const E: f64 = 1.0 - D; // 1 - lattice high (probe 3.48915260374018876e-01)
        const F: f64 = 1.0 - C; // 1 - lattice low (probe 8.25542369812990562e-01)
        let e = PyraRTk::new(1);
        let coords = e.dof_coords();
        assert_eq!(coords.len(), 28);
        let mut sorted: Vec<[f64; 3]> = coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<[f64; 3]> = vec![
            // base quad (zeta = 0)
            [A, B, 0.0],
            [B, B, 0.0],
            [A, A, 0.0],
            [B, A, 0.0],
            // tri (0,1,4): eta = 0
            [C, 0.0, C],
            [D, 0.0, C],
            [C, 0.0, D],
            // tri (1,2,4): xi + zeta = 1
            [F, C, C],
            [F, D, C],
            [E, C, D],
            // tri (2,3,4): eta + zeta = 1
            [D, F, C],
            [C, F, C],
            [C, E, D],
            // tri (3,0,4): xi = 0
            [0.0, D, C],
            [0.0, C, C],
            [0.0, C, D],
            // interior x (icp[1]=0.5 * (1-iop[k]), iop[j]*(1-iop[k]), iop[k])
            [0.5 * (1.0 - A), A * (1.0 - A), A],
            [0.5 * (1.0 - A), B * (1.0 - A), A],
            [0.5 * (1.0 - B), A * (1.0 - B), B],
            [0.5 * (1.0 - B), B * (1.0 - B), B],
            // interior y
            [A * (1.0 - A), 0.5 * (1.0 - A), A],
            [B * (1.0 - A), 0.5 * (1.0 - A), A],
            [A * (1.0 - B), 0.5 * (1.0 - B), B],
            [B * (1.0 - B), 0.5 * (1.0 - B), B],
            // interior z ((1-icp[1]) scaling = 0.5)
            [A * 0.5, A * 0.5, 0.5],
            [B * 0.5, A * 0.5, 0.5],
            [A * 0.5, B * 0.5, 0.5],
            [B * 0.5, B * 0.5, 0.5],
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let _ = (E, F);
        for (i, (g, w)) in sorted.iter().zip(expected.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (g[c] - w[c]).abs() < 1e-14,
                    "coord {i} comp {c}: got {} want {} (got {g:?} want {w:?})",
                    g[c],
                    w[c]
                );
            }
        }
    }

    /// D534 bitwise pin against the MFEM 4.10 probe
    /// (`tmp/d534/probe_d534.cpp` → `tmp/d534/d534_fuentes_basis.txt`):
    /// RAW lines are `CalcRawVShape` (the Fuentes raw expansion), NOD lines
    /// `CalcVShape` (`Ti·u`, the nodal basis) and NODE lines the element's
    /// own node table — sampled at 8 interior reference points for orders
    /// 0..2.  The port must reproduce every entry at MFEM's print precision.
    #[test]
    fn d534_fuentes_rt_basis_matches_mfem_410_probe() {
        let path = format!(
            "{}/../../tmp/d534/d534_fuentes_basis.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        let mut cached: Vec<Option<PyraRTk>> = (0..3).map(|_| None).collect();
        let mut n_raw = 0usize;
        let mut n_nod = 0usize;
        let mut n_node = 0usize;
        let mut max_raw = 0.0_f64;
        let mut max_nod = 0.0_f64;
        let mut u = vec![[0.0; 3]; 200];
        let mut phi = vec![0.0_f64; 200 * 3];
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            match f[0] {
                "RAW" | "NOD" => {
                    // `RAW p pt slot v0 v1 v2` (three integers, three doubles)
                    let p: usize = f[1].parse().unwrap();
                    let t: usize = f[2].parse().unwrap();
                    let slot: usize = f[3].parse().unwrap();
                    let want: [f64; 3] = [
                        f[4].parse().unwrap(),
                        f[5].parse().unwrap(),
                        f[6].parse().unwrap(),
                    ];
                    let n = pyramid_rtk_dim(p);
                    let pt = nod_point(t);
                    if f[0] == "RAW" {
                        for v in u.iter_mut().take(n) {
                            *v = [0.0; 3];
                        }
                        fuentes_rt_raw_basis(p + 1, pt[0], pt[1], pt[2], &mut u[..n]);
                        for c in 0..3 {
                            let d = (u[slot][c] - want[c]).abs();
                            max_raw = max_raw.max(d);
                            let tol = 5e-12 * (1.0 + want[c].abs());
                            let wc = want[c];
                            assert!(
                                d <= tol,
                                "RAW p={p} pt={t} dof={slot} comp={c}: {} vs {wc}",
                                u[slot][c]
                            );
                        }
                        n_raw += 1;
                    } else {
                        if cached[p].is_none() {
                            cached[p] = Some(PyraRTk::new(p));
                        }
                        let e = cached[p].as_ref().unwrap();
                        assert_eq!(e.n_dofs(), n);
                        e.eval_basis_vec(&pt, &mut phi[..n * 3]);
                        for c in 0..3 {
                            let d = (phi[slot * 3 + c] - want[c]).abs();
                            max_nod = max_nod.max(d);
                            // the nodal basis values grow like cond(T) — the
                            // comparison must scale with the entry magnitude
                            let tol = 5e-12 * (1.0 + want[c].abs());
                            let got = phi[slot * 3 + c];
                            let wc = want[c];
                            assert!(
                                d <= tol,
                                "NOD p={p} pt={t} dof={slot} comp={c}: {got} vs {wc}"
                            );
                        }
                        n_nod += 1;
                    }
                }
                "NODE" => {
                    let p: usize = f[1].parse().unwrap();
                    let m: usize = f[2].parse().unwrap();
                    if cached[p].is_none() {
                        cached[p] = Some(PyraRTk::new(p));
                    }
                    let c = &cached[p].as_ref().unwrap().dof_coords()[m];
                    for d in 0..3 {
                        let want: f64 = f[3 + d].parse().unwrap();
                        assert!(
                            (c[d] - want).abs() <= 1e-15,
                            "NODE p={p} dof={m} comp={d}: {} vs {want}",
                            c[d]
                        );
                    }
                    n_node += 1;
                }
                "COUNT" => {}
                other => panic!("unexpected probe line {other:?}"),
            }
        }
        assert_eq!(n_node, 5 + 28 + 87, "node rows checked");
        assert!(n_raw >= (5 + 28 + 87) * 8, "no RAW parity run: {n_raw}");
        assert!(n_nod >= (5 + 28 + 87) * 8, "no NOD parity run: {n_nod}");
        eprintln!(
            "d534 Fuentes RT pyramid vs MFEM 4.10: {n_raw} raw + {n_nod} nodal dofs matched, \
             max|delta| raw {max_raw:.3e} / nodal {max_nod:.3e}"
        );
    }

    /// The probe's sample point `t` (same table as probe_d534.cpp).
    fn nod_point(t: usize) -> [f64; 3] {
        const PTS: [[f64; 3]; 8] = [
            [0.25, 0.25, 0.125],
            [0.75, 0.25, 0.125],
            [0.25, 0.75, 0.125],
            [0.75, 0.75, 0.125],
            [0.5, 0.5, 0.375],
            [0.5, 0.5, 0.625],
            [0.125, 0.375, 0.75],
            [0.625, 0.875, 0.9],
        ];
        PTS[t]
    }

    /// Nodal duality on the element's own nodes: `phi_m(node_l)·nk_l = δ_ml`
    /// (MFEM `Ti.Factor(T)`'s defining property) — the dof value convention
    /// the space layer's interpolation engine relies on (D535).
    #[test]
    fn pyra_rtk_is_point_dual_to_its_nodes() {
        for p in 0..=2usize {
            let e = PyraRTk::new(p);
            let n = e.n_dofs();
            let mut phi = vec![0.0; n * 3];
            for l in 0..n {
                e.eval_basis_vec(&e.dof_coords()[l], &mut phi);
                for m in 0..n {
                    // the functional belongs to the evaluation node l
                    let nk = nk_of(dof2nk_slot(l, p));
                    let s: f64 = (0..3).map(|c| phi[m * 3 + c] * nk[c]).sum();
                    let want = if m == l { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-10,
                        "p={p}: phi_{m}(node_{l})·nk = {s}, want {want}"
                    );
                }
            }
        }
    }

    /// Higher-order constructions stay finite (MFEM probe counts 87/164
    /// pinned in [`pyra_rtk_dim_matches_mfem_fuentes`]).
    #[test]
    fn pyra_rtk_higher_orders_finite() {
        for p in 2..=3 {
            let e = PyraRTk::new(p);
            let mut v = vec![0.0; e.n_dofs() * 3];
            for pt in &e.quadrature(6).points {
                e.eval_basis_vec(pt, &mut v);
                assert!(v.iter().all(|x| x.is_finite()), "p={p}: non-finite basis");
            }
        }
    }

    /// k=0 behaviour: dim 5 (face blocks only, no interior).
    #[test]
    fn pyra_rt0_dim() {
        assert_eq!(PyraRTk::new(0).n_dofs(), 5);
        let e = PyraRTk::new(0);
        let mut v = vec![0.0; 15];
        for pt in &e.quadrature(4).points {
            e.eval_basis_vec(pt, &mut v);
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }
}
