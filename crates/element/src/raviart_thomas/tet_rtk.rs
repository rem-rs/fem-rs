//! Arbitrary-order Raviart-Thomas on the reference tetrahedron — a 1:1 port
//! of MFEM 4.10 `RT_TetrahedronElement` (D540).
//!
//! Reference tetrahedron `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`,
//! `RT_k = [P_k]³ + (x−c, y−c, z−c)·P̃_k` with `c = 1/4`
//! (`fem/fe/fe_rt.cpp:897`), `dim = (k+1)(k+2)(k+4)/2`.
//!
//! # Basis (D540)
//!
//! MFEM constructs the element nodally (`fe_rt.cpp:899-999`): the raw
//! expansion `u_o` is the product of the hierarchical 1-D bases
//! `poly1d.CalcBasis` (which **is** `CalcChebyshev`, `fe_base.hpp:1220` —
//! `T_i(2x−1)` with the `+2u` derivative recurrence) on `(x, y, z, 1−x−y−z)`
//! plus the `(x−c, y−c, z−c)` bubbles; the Vandermonde
//! `T(o, m) = u_o(node_m) · nk_{dof2nk[m]}` and its inverse (`Ti.Factor(T)`)
//! make every DOF a **point value** `dof_m = φ(node_m)·(cof J n̂_m)` with
//! `φ_m(node_l)·nk_l = δ_{ml}` — exactly the pyramid-side construction of
//! round 56 (D534).  The historical fem-rs construction in this file was a
//! moment-dual basis (`∫_F φ·n̂ q dA` functionals, Vandermonde `W = 2I` at
//! `k = 0`), which made `HDivSpace::interpolate_vector` store half the MFEM
//! flux sample on every tet dof (the D543/D540 lesion) — mixed pyr↔tet
//! meshes wrote inconsistent shared-face values.  This port removes the
//! convention split: every tet RT order is now MFEM's nodal element, so
//! `TetRTk`, [`TetRT1`](super::tet_rt1::TetRT1),
//! [`TetRT2`](super::tet_rt2::TetRT2) and
//! [`TetRTNodal`](super::tet_rt1::TetRTNodal) are all the same point-dual
//! family (`W = I`).
//!
//! # Slot order
//!
//! MFEM's own: faces `(1,2,3), (0,3,2), (0,1,3), (0,2,1)` with
//! `(k+1)(k+2)/2` Gauss-Legendre lattice nodes each (`bop = OpenPoints(k)`),
//! then `k(k+1)(k+2)/2` interior component samples (`iop = OpenPoints(k−1)`,
//! `dof2nk` 1, 2, 3 → `nk ∈ {−x, −y, −z}`).  The node table is the shared
//! [`crate::raviart_thomas::tet_rt1::mfem_nodal_dofs`] (single source for the
//! space engine's dual rows, the assembler and this basis).

use std::sync::OnceLock;

use nalgebra::DMatrix;

use super::tet_rt1::{mfem_nodal_dofs, nodal_tet_dof_coords, nodal_tet_n_dofs};
use crate::quadrature::tet_rule;
use crate::reference::{QuadratureRule, VectorReferenceElement};

/// MFEM `RT_TetrahedronElement::c` (`fe_rt.cpp:897`): the bubble centre.
const C: f64 = 0.25;

/// MFEM `Poly_1D::CalcChebyshev` (`fe_base.cpp:2376`): the hierarchical 1-D
/// basis `u_i = T_i(2x−1)` with derivatives w.r.t. `x`
/// (`d[i] = 2·T'_i(2x−1)`, recurrence `d[n+1] = (n+1)(z·d[n]/n + 2·u[n])`).
fn calc_chebyshev(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0; p + 1];
    let mut d = vec![0.0; p + 1];
    u[0] = 1.0;
    if p == 0 {
        return (u, d);
    }
    let z = 2.0 * x - 1.0;
    u[1] = z;
    d[1] = 2.0;
    for n in 1..p {
        u[n + 1] = 2.0 * z * u[n] - u[n - 1];
        d[n + 1] = (n + 1) as f64 * (z * d[n] / n as f64 + 2.0 * u[n]);
    }
    (u, d)
}

struct TetRTkData {
    /// `φ_m = Σ_o ti[m·n + o] · u_o(x)` — row-major `T⁻¹` of the nodal
    /// Vandermonde `T(o, m) = u_o(node_m)·nk_{dof2nk[m]}` (MFEM `Ti`).
    ti: Vec<f64>,
}

fn tet_data(k: usize) -> &'static TetRTkData {
    static CACHE: [OnceLock<TetRTkData>; 5] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[k].get_or_init(|| {
        let n = nodal_tet_n_dofs(k);
        let p = k;
        let (dof_pts, dof_nks) = mfem_nodal_dofs(k);
        debug_assert_eq!(dof_pts.len(), n);
        // T(o, m) row-major: rows = raw functions, columns = dofs/nodes.
        let mut t = DMatrix::<f64>::zeros(n, n);
        for m in 0..n {
            let (pt, nk) = (&dof_pts[m], &dof_nks[m]);
            let (sx, _) = calc_chebyshev(p, pt[0]);
            let (sy, _) = calc_chebyshev(p, pt[1]);
            let (sz, _) = calc_chebyshev(p, pt[2]);
            let (sl, _) = calc_chebyshev(p, 1.0 - pt[0] - pt[1] - pt[2]);
            let mut o = 0usize;
            for kk in 0..=p {
                for j in 0..=(p - kk) {
                    for i in 0..=(p - kk - j) {
                        let s = sx[i] * sy[j] * sz[kk] * sl[p - i - j - kk];
                        t[(o, m)] = s * nk[0];
                        o += 1;
                        t[(o, m)] = s * nk[1];
                        o += 1;
                        t[(o, m)] = s * nk[2];
                        o += 1;
                    }
                }
            }
            for j in 0..=p {
                for i in 0..=(p - j) {
                    let s = sx[i] * sy[j] * sz[p - i - j];
                    t[(o, m)] = s * ((pt[0] - C) * nk[0] + (pt[1] - C) * nk[1] + (pt[2] - C) * nk[2]);
                    o += 1;
                }
            }
            debug_assert_eq!(o, n);
        }
        let ti_m = t
            .try_inverse()
            .expect("TetRTk: singular MFEM nodal Vandermonde matrix");
        let mut ti = vec![0.0; n * n];
        for m in 0..n {
            for o in 0..n {
                ti[m * n + o] = ti_m[(m, o)];
            }
        }
        TetRTkData { ti }
    })
}

/// Raw expansion `u_o` at `(x, y, z)` in MFEM's row order (`CalcVShape`
/// body, `fe_rt.cpp:1001-1035`): the `[P_p]³` Chebyshev-product block, then
/// the `(x−c, y−c, z−c)` bubbles.
fn fill_raw_row(k: usize, x: f64, y: f64, z: f64, u: &mut [f64]) {
    let p = k;
    let (sx, _) = calc_chebyshev(p, x);
    let (sy, _) = calc_chebyshev(p, y);
    let (sz, _) = calc_chebyshev(p, z);
    let (sl, _) = calc_chebyshev(p, 1.0 - x - y - z);
    let mut o = 0usize;
    for kk in 0..=p {
        for j in 0..=(p - kk) {
            for i in 0..=(p - kk - j) {
                let s = sx[i] * sy[j] * sz[kk] * sl[p - i - j - kk];
                u[o * 3] = s;
                u[o * 3 + 1] = 0.0;
                u[o * 3 + 2] = 0.0;
                o += 1;
                u[o * 3] = 0.0;
                u[o * 3 + 1] = s;
                u[o * 3 + 2] = 0.0;
                o += 1;
                u[o * 3] = 0.0;
                u[o * 3 + 1] = 0.0;
                u[o * 3 + 2] = s;
                o += 1;
            }
        }
    }
    for j in 0..=p {
        for i in 0..=(p - j) {
            let s = sx[i] * sy[j] * sz[p - i - j];
            u[o * 3] = (x - C) * s;
            u[o * 3 + 1] = (y - C) * s;
            u[o * 3 + 2] = (z - C) * s;
            o += 1;
        }
    }
    debug_assert_eq!(o, nodal_tet_n_dofs(k));
}

/// Divergence of each raw generator (same layout as [`fill_raw_row`]) —
/// MFEM `RT_TetrahedronElement::CalcDivShape` (`fe_rt.cpp:1037-1080`).
fn fill_raw_div_row(k: usize, x: f64, y: f64, z: f64, divs: &mut [f64]) {
    let p = k;
    let (sx, dsx) = calc_chebyshev(p, x);
    let (sy, dsy) = calc_chebyshev(p, y);
    let (sz, dsz) = calc_chebyshev(p, z);
    // dsl is the derivative w.r.t. the ARGUMENT s = 1−x−y−z; the ∂s/∂x = −1
    // chain factor is the explicit minus sign in the MFEM lines below.
    let (sl, dsl) = calc_chebyshev(p, 1.0 - x - y - z);
    let mut o = 0usize;
    for kk in 0..=p {
        for j in 0..=(p - kk) {
            for i in 0..=(p - kk - j) {
                let l = p - i - j - kk;
                divs[o] = (dsx[i] * sl[l] - sx[i] * dsl[l]) * sy[j] * sz[kk];
                o += 1;
                divs[o] = (dsy[j] * sl[l] - sy[j] * dsl[l]) * sx[i] * sz[kk];
                o += 1;
                divs[o] = (dsz[kk] * sl[l] - sz[kk] * dsl[l]) * sx[i] * sy[j];
                o += 1;
            }
        }
    }
    for j in 0..=p {
        for i in 0..=(p - j) {
            let kk = p - i - j;
            divs[o] = (sx[i] + (x - C) * dsx[i]) * sy[j] * sz[kk]
                + (sy[j] + (y - C) * dsy[j]) * sx[i] * sz[kk]
                + (sz[kk] + (z - C) * dsz[kk]) * sx[i] * sy[j];
            o += 1;
        }
    }
    debug_assert_eq!(o, nodal_tet_n_dofs(k));
}

/// Shared MFEM `RT_TetrahedronElement` evaluation used by [`TetRTk`],
/// [`super::tet_rt1::TetRT1`] and [`super::tet_rt2::TetRT2`].
pub(super) fn eval_mfem_rt_tet_basis(k: usize, xi: &[f64], values: &mut [f64]) {
    let d = tet_data(k);
    let n = nodal_tet_n_dofs(k);
    let mut u = vec![0.0; n * 3];
    fill_raw_row(k, xi[0], xi[1], xi[2], &mut u);
    for m in 0..n {
        let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
        for (o, c) in d.ti[m * n..m * n + n].iter().enumerate() {
            vx += c * u[o * 3];
            vy += c * u[o * 3 + 1];
            vz += c * u[o * 3 + 2];
        }
        values[m * 3] = vx;
        values[m * 3 + 1] = vy;
        values[m * 3 + 2] = vz;
    }
}

/// Shared analytic divergence (MFEM `CalcDivShape` port) for the tet RT
/// family — [`TetRTk`], [`super::tet_rt1::TetRT1`], [`super::tet_rt2::TetRT2`].
pub(super) fn eval_mfem_rt_tet_div(k: usize, xi: &[f64], div_vals: &mut [f64]) {
    let d = tet_data(k);
    let n = nodal_tet_n_dofs(k);
    let mut divs = vec![0.0; n];
    fill_raw_div_row(k, xi[0], xi[1], xi[2], &mut divs);
    for m in 0..n {
        let mut s = 0.0;
        for (o, c) in d.ti[m * n..m * n + n].iter().enumerate() {
            s += c * divs[o];
        }
        div_vals[m] = s;
    }
}

/// Raviart-Thomas H(div) element on the reference tetrahedron — a 1:1 port
/// of MFEM `RT_TetrahedronElement(k)` (D540): `(k+1)(k+2)(k+4)/2` DOFs,
/// nodal flux-sample functionals `dof_m = φ(node_m)·(cof J n̂_m)`,
/// `φ_m(node_l)·nk_l = δ_{ml}` (`W = I`).
pub struct TetRTk {
    order: usize,
}

impl TetRTk {
    pub fn new(p: usize) -> Self {
        assert!(
            p <= 4,
            "TetRTk: the shared MFEM nodal table cache covers k = 0..=4, got {p}"
        );
        TetRTk { order: p }
    }
}

impl VectorReferenceElement for TetRTk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        nodal_tet_n_dofs(self.order)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        eval_mfem_rt_tet_basis(self.order, xi, values);
    }

    fn eval_div(&self, xi: &[f64], div_vals: &mut [f64]) {
        eval_mfem_rt_tet_div(self.order, xi, div_vals);
    }

    /// MFEM exposes no curl for H(div) elements (`RT_TetrahedronElement`
    /// carries no `CalcCurlShape`); consumers of this trait's curl on RT
    /// bases are the identity-zero paths.
    fn eval_curl(&self, _xi: &[f64], curl_vals: &mut [f64]) {
        for v in curl_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        tet_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        nodal_tet_dof_coords(self.order)
    }
}

/// The order-generic alias under which the space/assembler layers dispatch
/// tet RT ≥ 3 (D392); since D540 this is literally the same MFEM-nodal
/// element as [`TetRTk`], [`super::tet_rt1::TetRT1`] and
/// [`super::tet_rt2::TetRT2`].
pub type TetRTNodal = TetRTk;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raviart_thomas::tet_rt1::TET_NK;

    /// D540 bitwise pin against the MFEM 4.10 probe
    /// (`tmp/d540/probe_d540.cpp` → `tmp/d540/d540_tet_nodal_basis.txt`):
    /// NODE lines are the element's own node table, NOD lines `CalcVShape`
    /// (`Ti·u`, the nodal basis) and DIV lines `CalcDivShape` (the analytic
    /// divergence) — sampled at 8 interior reference points for orders
    /// 0..=4.  The port must reproduce every entry at the probe's print
    /// precision.
    #[test]
    fn d540_tet_rt_basis_matches_mfem_410_probe() {
        let path = format!(
            "{}/../../tmp/d540/d540_tet_nodal_basis.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        let mut n_node = 0usize;
        let mut n_nod = 0usize;
        let mut n_div = 0usize;
        let mut max_nod = 0.0_f64;
        let mut max_div = 0.0_f64;
        let mut phi = vec![0.0_f64; 120 * 3];
        let mut div = vec![0.0_f64; 120];
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            match f[0] {
                "COUNT" => {
                    let p: usize = f[2].parse().unwrap();
                    let dof: usize = f[4].parse().unwrap();
                    assert_eq!(TetRTk::new(p).n_dofs(), dof, "probe COUNT p={p}");
                }
                "NODE" => {
                    let p: usize = f[1].parse().unwrap();
                    let m: usize = f[2].parse().unwrap();
                    let c = &TetRTk::new(p).dof_coords()[m];
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
                "NOD" => {
                    let p: usize = f[1].parse().unwrap();
                    let t: usize = f[2].parse().unwrap();
                    let slot: usize = f[3].parse().unwrap();
                    let want: [f64; 3] = [
                        f[4].parse().unwrap(),
                        f[5].parse().unwrap(),
                        f[6].parse().unwrap(),
                    ];
                    TetRTk::new(p).eval_basis_vec(&nod_point(t), &mut phi);
                    for c in 0..3 {
                        let d = (phi[slot * 3 + c] - want[c]).abs();
                        max_nod = max_nod.max(d);
                        // the nodal values grow like cond(T); scale the
                        // comparison with the entry magnitude
                        let tol = 5e-11 * (1.0 + want[c].abs());
                        let got = phi[slot * 3 + c];
                        let wc = want[c];
                        assert!(
                            d <= tol,
                            "NOD p={p} pt={t} dof={slot} comp={c}: {got} vs {wc}"
                        );
                    }
                    n_nod += 1;
                }
                "DIV" => {
                    let p: usize = f[1].parse().unwrap();
                    let t: usize = f[2].parse().unwrap();
                    let slot: usize = f[3].parse().unwrap();
                    let want: f64 = f[4].parse().unwrap();
                    TetRTk::new(p).eval_div(&nod_point(t), &mut div);
                    let d = (div[slot] - want).abs();
                    max_div = max_div.max(d);
                    let tol = 5e-11 * (1.0 + want.abs());
                    let got = div[slot];
                    assert!(d <= tol, "DIV p={p} pt={t} dof={slot}: {got} vs {want}");
                    n_div += 1;
                }
                other => panic!("unexpected probe line {other:?}"),
            }
        }
        assert_eq!(n_node, 4 + 15 + 36 + 70 + 120, "node rows checked");
        assert!(n_nod >= (4 + 15 + 36 + 70 + 120) * 8, "no NOD run: {n_nod}");
        assert!(n_div >= (4 + 15 + 36 + 70 + 120) * 8, "no DIV run: {n_div}");
        eprintln!(
            "d540 tet RT vs MFEM 4.10: {n_node} nodes + {n_nod} nodal + {n_div} div entries \
             matched, max|delta| nodal {max_nod:.3e} / div {max_div:.3e}"
        );
    }

    /// The probe's sample point `t` (same table as probe_d540.cpp).
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

    /// MFEM `Ti.Factor(T)`'s defining property on the element's own nodes:
    /// `phi_m(node_l)·nk_l = δ_ml` — the point duality D540 restores for the
    /// whole tet RT family (`W = I`), k = 0..=4.
    #[test]
    fn tet_rtk_is_point_dual_to_its_nodes() {
        for k in 0..=4usize {
            let e = TetRTk::new(k);
            let n = e.n_dofs();
            let (pts, nks) = mfem_nodal_dofs(k);
            let mut phi = vec![0.0; n * 3];
            for (l, (pt, nk)) in pts.iter().zip(nks.iter()).enumerate() {
                e.eval_basis_vec(pt, &mut phi);
                for m in 0..n {
                    let s = phi[m * 3] * nk[0] + phi[m * 3 + 1] * nk[1] + phi[m * 3 + 2] * nk[2];
                    let want = if m == l { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-10,
                        "k={k}: phi_{m}(node_{l})·nk = {s}, want {want}"
                    );
                }
            }
        }
    }

    /// D33 regression, nodal edition: every face block's basis functions
    /// carry a vanishing normal trace on all faces outside their own block,
    /// interior-supported functions on no face at all (the slot ↔ face
    /// agreement `HDivSpace::build_3d_tet`'s pairing relies on).
    #[test]
    fn per_face_normal_support() {
        let pts: [[f64; 3]; 4] = [
            [0.42, 0.27, 0.31],
            [0.0, 0.3, 0.4],
            [0.3, 0.0, 0.4],
            [0.3, 0.4, 0.0],
        ];
        for k in 0..=2usize {
            let e = TetRTk::new(k);
            let n = e.n_dofs();
            let face_dofs = (k + 1) * (k + 2) / 2;
            let mut phi = vec![0.0f64; n * 3];
            for (f, pt) in pts.iter().enumerate() {
                e.eval_basis_vec(pt, &mut phi);
                for j in 0..n {
                    let flux = phi[j * 3] * TET_NK[f][0]
                        + phi[j * 3 + 1] * TET_NK[f][1]
                        + phi[j * 3 + 2] * TET_NK[f][2];
                    let in_block = j < 4 * face_dofs && j / face_dofs == f;
                    if !in_block {
                        assert!(
                            flux.abs() < 1e-9,
                            "k={k}: face {f}: basis {j} normal trace = {flux}"
                        );
                    }
                }
            }
        }
    }

    /// The analytic divergence (`CalcDivShape` port) against a central
    /// finite difference of the basis — guards the derivative recurrence.
    #[test]
    fn analytic_div_matches_finite_difference() {
        for k in 0..=3usize {
            let e = TetRTk::new(k);
            let n = e.n_dofs();
            let h = 1e-6;
            let mut vp = vec![0.0; n * 3];
            let mut vm = vec![0.0; n * 3];
            let mut dv = vec![0.0; n];
            for pt in [[0.25, 0.25, 0.25], [0.1, 0.2, 0.15], [0.5, 0.1, 0.1]] {
                e.eval_basis_vec(&[pt[0] + h, pt[1], pt[2]], &mut vp);
                e.eval_basis_vec(&[pt[0] - h, pt[1], pt[2]], &mut vm);
                let mut fdx: Vec<f64> = (0..n).map(|i| (vp[i * 3] - vm[i * 3]) / (2.0 * h)).collect();
                e.eval_basis_vec(&[pt[0], pt[1] + h, pt[2]], &mut vp);
                e.eval_basis_vec(&[pt[0], pt[1] - h, pt[2]], &mut vm);
                for i in 0..n {
                    fdx[i] += (vp[i * 3 + 1] - vm[i * 3 + 1]) / (2.0 * h);
                }
                e.eval_basis_vec(&[pt[0], pt[1], pt[2] + h], &mut vp);
                e.eval_basis_vec(&[pt[0], pt[1], pt[2] - h], &mut vm);
                for i in 0..n {
                    fdx[i] += (vp[i * 3 + 2] - vm[i * 3 + 2]) / (2.0 * h);
                }
                e.eval_div(&pt, &mut dv);
                for i in 0..n {
                    assert!(
                        (dv[i] - fdx[i]).abs() < 1e-5,
                        "k={k} at {pt:?}: div[{i}] = {} vs fd {}",
                        dv[i],
                        fdx[i]
                    );
                }
            }
        }
    }

    /// Basis and divergence stay finite across the reference tetrahedron.
    #[test]
    fn basis_and_div_finite() {
        for k in 1..=3usize {
            let e = TetRTk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            let mut d = vec![0.0; n];
            for pt in &[[0.25, 0.25, 0.25], [0.1, 0.2, 0.15], [0.5, 0.1, 0.1]] {
                e.eval_basis_vec(pt, &mut v);
                e.eval_div(pt, &mut d);
                for val in v.iter().chain(d.iter()) {
                    assert!(val.is_finite(), "k={k} at {pt:?}");
                }
            }
        }
    }
}
