//! D814-3 — the **wg family's volume path** (`wg_poisson`, `wg_stokes`,
//! `wg_maxwell`) on curved simplex meshes: the "build the oracle first" round.
//!
//! # The debt
//!
//! Round 79 registered that the family's **face** stabilizer walked a chord
//! route (fixed in D810-1, `d810_wg_face_geometry.rs`) while the **volume**
//! path kept its own vertex geometry: `local_jac` built the element Jacobian
//! from the element's corner differences (constant over the element) and the
//! Stokes body force interpolated the physical QP affinely from the corners.
//! Exact only on a straight mesh; on a mesh carrying an order-`g` geometry
//! table the body term silently uses a *different element* than the (already
//! isoparametric) face term of the very same assembly — the D808-4/D783 class.
//!
//! # The MFEM counterpart survey (why the oracle is geometric)
//!
//! MFEM 4.10 has **no weak-Galerkin integrator**: `fem/bilininteg.hpp`'s
//! `Mixed*Weak*Integrator` classes (`MixedScalarWeakGradientIntegrator`
//! `:951`, `MixedScalarWeakCurlIntegrator` `:1059`,
//! `MixedVectorWeakDivergenceIntegrator` `:2142`) are integration-by-parts
//! *mixed* integrators — a different discretization from the Wang–Ye weak
//! gradient/curl the three modules implement (element-local `M_Σ⁻¹G`
//! systems + face stabilizers), and `miniapps/` has no WG port either.  So
//! there is no *method-level* MFEM truth to port — the registered处方:
//! build the oracle at the level of the arithmetic the body path actually
//! consumes, which MFEM *does* own: `ElementTransformation::Jacobian()`
//! (what `local_jac` approximates), `Weight()` (what the QP weights
//! multiply), `Transform()` (the point the Stokes force evaluates `f` at).
//! `tmp/d87b/d814_probe.cpp` dumps all three, per element-QP, at MFEM's own
//! `IntRules.Get(TRIANGLE/TETRAHEDRON, 3)` points (the rule fem-rs'
//! `tri_rule(3)`/`tet_rule(3)` are pinned to), on the same straight and
//! curved fixtures D810-1 used; the gold files are the probe's stdout
//! verbatim.
//!
//! # What the tests prove
//!
//! * the fixtures are the probe's meshes (vertices, connectivity, measures),
//! * `element_jacobian_at` — the isoparametric single source the body path is
//!   switched to — reproduces MFEM's `J`/`W`/`x` entry by entry on both
//!   (D787's result, re-bound here to the wg fixtures),
//! * the **pre-fix vertex route** reproduced verbatim (`Route::Vertex`)
//!   matches MFEM to round-off on the straight meshes (the negative pin: the
//!   switch is straight-neutral) and is far from it on the curved ones (the
//!   teeth),
//! * and — the binding — each module's assembled **volume block**
//!   (`assemble_wg_*(…, penalty = 0, …)`) equals an independent re-assembly
//!   through `element_jacobian_at` and is far from the vertex-route one on the
//!   curved meshes, while on the straight meshes the two routes coincide.

use fem_assembly::{assemble_wg_maxwell, assemble_wg_poisson, assemble_wg_stokes};
use fem_element::lagrange::{TetPk, TriPk};
use fem_element::nedelec::{TetNDk, TriNDk};
use fem_element::quadrature::{tet_rule, tri_rule};
use fem_element::{ReferenceElement, VectorReferenceElement};
use fem_linalg::CsrMatrix;
use fem_mesh::element_type::ElementType;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, L2Space};
use nalgebra::DMatrix;

const GOLD_TRI_STRAIGHT: &str = include_str!("data/d831_mfem_volume_tri_n1_straight.txt");
const GOLD_TRI_CURVED: &str = include_str!("data/d831_mfem_volume_tri_n1_curved.txt");
const GOLD_TET_STRAIGHT: &str = include_str!("data/d831_mfem_volume_tet_n1_straight.txt");
const GOLD_TET_CURVED: &str = include_str!("data/d831_mfem_volume_tet_n1_curved.txt");

// ─── Gold parsing (the probe's stdout verbatim) ─────────────────────────────

struct Eqp {
    e: usize,
    r: usize,
    j: Vec<f64>,
    w: f64,
    x: Vec<f64>,
}

struct Gold {
    dim: usize,
    verts: Vec<[f64; 3]>,
    elems: Vec<Vec<u32>>,
    evols: Vec<f64>,
    rqps: Vec<(Vec<f64>, f64)>,
    eqps: Vec<Eqp>,
}

fn parse_gold(s: &str) -> Gold {
    let mut g =
        Gold { dim: 0, verts: Vec::new(), elems: Vec::new(), evols: Vec::new(), rqps: Vec::new(), eqps: Vec::new() };
    for line in s.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "[MESH]" => {
                for tok in &f {
                    if let Some(v) = tok.strip_prefix("dim=") {
                        g.dim = v.parse().unwrap();
                    }
                }
            }
            // "[V] <index> x y z" — the vertex id comes first.
            "[V]" => g.verts.push([
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            ]),
            "[ELEM]" => {
                g.elems.push(f[3..].iter().map(|t| t.parse().unwrap()).collect());
            }
            "[EVOL]" => g.evols.push(f[2].parse().unwrap()),
            "[NQP]" => {}
            "[RQP]" => g.rqps.push((
                f[2..5].iter().map(|t| t.parse().unwrap()).collect(),
                f[5].parse().unwrap(),
            )),
            "[EQP]" => {
                let e: usize = f[1].trim_start_matches("e=").parse().unwrap();
                let r: usize = f[2].trim_start_matches("r=").parse().unwrap();
                let mut j = Vec::new();
                let mut k = 3;
                loop {
                    let mut tok = f[k].trim_start_matches("J=(");
                    if tok.ends_with(')') {
                        tok = &tok[..tok.len() - 1];
                        j.push(tok.parse().unwrap());
                        break;
                    }
                    j.push(tok.parse().unwrap());
                    k += 1;
                }
                k += 1; // the W= token
                let w: f64 = f[k].trim_start_matches("W=").parse().unwrap();
                let mut x = Vec::new();
                k += 1;
                loop {
                    let mut tok = f[k].trim_start_matches("x=(");
                    if tok.ends_with(')') {
                        tok = &tok[..tok.len() - 1];
                        x.push(tok.parse().unwrap());
                        break;
                    }
                    x.push(tok.parse().unwrap());
                    k += 1;
                }
                g.eqps.push(Eqp { e, r, j, w, x });
            }
            _ => panic!("unexpected gold line: {line}"),
        }
    }
    g
}

// ─── Fixtures (the probe's meshes, cell for cell — d810's recipe) ───────────

fn warp3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn warp2(x: [f64; 2]) -> [f64; 2] {
    [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
}

fn warp_all<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
    let n_geom = mesh.geometry.as_ref().expect("curved table").coords.len() / D;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; D];
            x.copy_from_slice(&geo.coords[k * D..(k + 1) * D]);
            let y = f(x);
            geo.coords[k * D..(k + 1) * D].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; D];
        x.copy_from_slice(&mesh.coords[k * D..(k + 1) * D]);
        let y = f(x);
        mesh.coords[k * D..(k + 1) * D].copy_from_slice(&y);
    }
}

fn straight_tri_mesh() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 0, 2, 3],
        vec![1, 1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1, 1, 1, 1],
        ElementType::Line2,
    )
}

fn curved_tri_mesh() -> Mesh<2> {
    let mut m = straight_tri_mesh();
    m.set_curvature(2);
    warp_all(&mut m, warp2);
    m
}

fn straight_tet_mesh() -> Mesh<3> {
    Mesh::<3>::unit_cube_tet(1)
}

fn curved_tet_mesh() -> Mesh<3> {
    let mut m = Mesh::<3>::unit_cube_tet(1);
    m.set_curvature(2);
    warp_all(&mut m, warp3);
    m
}

// ─── Mesh identity helpers ──────────────────────────────────────────────────

fn match_elem<const D: usize>(mesh: &Mesh<D>, tuple: &[u32]) -> u32 {
    for e in 0..mesh.n_elems() as u32 {
        if mesh.element_nodes(e) == tuple {
            return e;
        }
    }
    panic!("no element with the ordered vertex tuple {tuple:?} — the fixtures diverged");
}

/// `∫_T |det J|` over the mesh's own geometry map (the gold's `[EVOL]`).
fn element_measure<const D: usize>(mesh: &Mesh<D>, e: u32) -> f64 {
    let dim = D;
    let qf = if dim == 2 { tri_rule(10) } else { tet_rule(10) };
    let mut v = 0.0;
    for (qi, xi) in qf.points.iter().enumerate() {
        let (j, _x) = element_jacobian_at(mesh, e, xi, dim);
        v += qf.weights[qi] * j.determinant().abs();
    }
    v
}

fn check_mesh_identity<const D: usize>(mesh: &Mesh<D>, gold: &Gold, name: &str) {
    assert_eq!(D, gold.dim, "{name}: dimension");
    assert_eq!(mesh.n_nodes(), gold.verts.len(), "{name}: vertex count");
    let mut worst_v = 0.0_f64;
    for (i, gv) in gold.verts.iter().enumerate() {
        for d in 0..gold.dim {
            worst_v = worst_v.max((mesh.coords[i * gold.dim + d] - gv[d]).abs());
        }
    }
    assert!(worst_v < 1e-15, "{name}: vertices differ by {worst_v:.3e}");
    let mut worst_vol = 0.0_f64;
    for (i, ge) in gold.elems.iter().enumerate() {
        let e = match_elem::<D>(mesh, ge);
        let want = gold.evols[i];
        worst_vol = worst_vol.max((element_measure(mesh, e) - want).abs() / want.abs());
    }
    println!("{name}: mesh identity max|Δvertex|={worst_v:.3e} max rel|Δvol|={worst_vol:.3e}");
    assert!(worst_vol < 1e-13, "{name}: element measures differ by {worst_vol:.3e}");
}

/// The wg body path's quadrature at `quad_order = 3` must be **the same rule**
/// the gold sampled (`IntRules.Get(TRIANGLE/TETRAHEDRON, 3)`) — as a *set*:
/// fem-rs' tabulated WV rules list the orbits in their own order, so only the
/// point/weight multiset is pinned here (the geometry comparison below
/// evaluates at the gold's coordinates regardless of order).
fn check_rule_parity(gold: &Gold, dim: usize) {
    let mut rule = if dim == 2 { tri_rule(3) } else { tet_rule(3) };
    let pts: Vec<(Vec<f64>, f64)> = rule.points.drain(..).zip(rule.weights.drain(..)).collect();
    assert_eq!(pts.len(), gold.rqps.len(), "rule point count");
    // fem-rs stores the reference dimension's coordinates; the gold pads the
    // third with zeros — compare the first `dim` components.
    #[allow(clippy::redundant_closure_for_method_calls)]
    let norm = |p: &[f64]| {
        p[..dim].iter().map(|v| format!("{v:.12}")).collect::<Vec<_>>().join(",")
    };
    let mut a: Vec<String> = pts.iter().map(|(p, w)| format!("{}|{w:.12}", norm(p))).collect();
    let mut b: Vec<String> =
        gold.rqps.iter().map(|(p, w)| format!("{}|{w:.12}", norm(p))).collect();
    a.sort();
    b.sort();
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x, y, "rule point/weight multiset differs: {x} vs {y}");
    }
}

// ─── The two geometry routes ────────────────────────────────────────────────

/// The wg volume path's geometry, per route.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    /// The **pre-fix** route (`local_jac`): corner-difference Jacobian
    /// (constant over the element) and affine corner interpolation of the
    /// physical point — verbatim the code D814-3 removed.
    Vertex,
    /// The isoparametric single source (`element_jacobian_at`, MFEM
    /// `ElementTransformation::Jacobian()/Transform()`).
    Iso,
}

fn route_geom<M: MeshTopology + ?Sized>(
    mesh: &M, e: u32, xi: &[f64], dim: usize, route: Route,
) -> (DMatrix<f64>, Vec<f64>) {
    match route {
        Route::Iso => element_jacobian_at(mesh, e, xi, dim),
        Route::Vertex => {
            let nodes = mesh.element_nodes(e);
            let x0 = mesh.node_coords(nodes[0]);
            let mut jac = DMatrix::zeros(dim, dim);
            for i in 0..dim {
                let xip = mesh.node_coords(nodes[1 + i]);
                for d in 0..dim {
                    jac[(d, i)] = xip[d] - x0[d];
                }
            }
            let xp: Vec<f64> = (0..dim)
                .map(|d| x0[d] + (0..dim).map(|k| jac[(d, k)] * xi[k]).sum::<f64>())
                .collect();
            (jac, xp)
        }
    }
}

/// Worst entry-wise deviation of the route's geometry from the gold's
/// `[EQP]` records: `(rel J, rel W, max |Δx|)`.  Evaluated at the **gold's**
/// rule-point coordinates (the two sides' rules are the same multiset —
/// [`check_rule_parity`] — but list the orbits in their own order).
fn route_vs_gold<const D: usize>(mesh: &Mesh<D>, gold: &Gold, route: Route) -> (f64, f64, f64) {
    check_rule_parity(gold, D);
    let (mut wj, mut ww, mut wx) = (0.0_f64, 0.0_f64, 0.0_f64);
    for qp in &gold.eqps {
        let e = match_elem::<D>(mesh, &gold.elems[qp.e]);
        let (jac, xp) = route_geom(mesh, e, &gold.rqps[qp.r].0, D, route);
        for (k, gj) in qp.j.iter().enumerate() {
            wj = wj.max((jac[(k / D, k % D)] - gj).abs() / gj.abs().max(1e-1));
        }
        ww = ww.max((jac.determinant().abs() - qp.w).abs() / qp.w.max(1e-1));
        for d in 0..D {
            wx = wx.max((xp[d] - qp.x[d]).abs());
        }
    }
    (wj, ww, wx)
}

// ─── Independent volume-block re-assemblies (the binding) ───────────────────

fn dense_of(k: &CsrMatrix<f64>, n: usize) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for p in k.row_ptr[i]..k.row_ptr[i + 1] {
            d[i][k.col_idx[p] as usize] += k.values[p];
        }
    }
    d
}

/// Norm-relative difference `max|a−b| / max|a,b|`: the wg stiffness blocks
/// span magnitudes from O(1) down to the round-off floor (near-null gauge
/// directions), where an *entrywise* relative metric explodes on last-bit
/// arithmetic noise between compilation units.
fn max_rel_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            num = num.max((va - vb).abs());
            den = den.max(va.abs()).max(vb.abs());
        }
    }
    num / den.max(1e-30)
}

fn max_rel_vec(a: &[f64], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (va, vb) in a.iter().zip(b.iter()) {
        num = num.max((va - vb).abs());
        den = den.max(va.abs()).max(vb.abs());
    }
    num / den.max(1e-30)
}

fn scatter(dst: &mut [Vec<f64>], dofs: &[usize], block: &DMatrix<f64>) {
    for (i, &di) in dofs.iter().enumerate() {
        for (j, &dj) in dofs.iter().enumerate() {
            dst[di][dj] += block[(i, j)];
        }
    }
}

/// `(G, M_Σ)` of the wg Poisson/Stokes weak gradient — the module's
/// `weak_gradient_matrix` with the geometry source abstracted into `route`.
fn grad_blocks<const D: usize>(
    mesh: &Mesh<D>, e: u32, order: usize, qo: u8, route: Route,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = D;
    let ref_v: Box<dyn ReferenceElement> =
        if dim == 2 { Box::new(TriPk::new(order)) } else { Box::new(TetPk::new(order)) };
    let n_v = ref_v.n_dofs();
    let os = if order > 0 { order - 1 } else { 0 };
    let (n_ss, use_const) = if os == 0 {
        (1_usize, true)
    } else {
        let rs: Box<dyn ReferenceElement> =
            if dim == 2 { Box::new(TriPk::new(os)) } else { Box::new(TetPk::new(os)) };
        (rs.n_dofs(), false)
    };
    let n_s = dim * n_ss;
    let qr = if dim == 2 { tri_rule(qo) } else { tet_rule(qo) };
    let mut g = DMatrix::zeros(n_v, n_s);
    let mut ms = DMatrix::zeros(n_s, n_s);
    let mut pv = vec![0.0; n_v];
    let mut ps = vec![0.0; n_ss];
    let mut gsp = vec![0.0; n_ss * dim];
    for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
        let (jac, _xp) = route_geom(mesh, e, pt, dim, route);
        let det_j = jac.determinant();
        if det_j.abs() < 1e-30 {
            continue;
        }
        let wq = w * det_j.abs();
        let jit = jac.try_inverse().unwrap().transpose();
        ref_v.eval_basis(pt, &mut pv);
        if use_const {
            ps[0] = 1.0;
            for d in 0..dim {
                gsp[d] = 0.0;
            }
        } else {
            let rs: Box<dyn ReferenceElement> =
                if dim == 2 { Box::new(TriPk::new(os)) } else { Box::new(TetPk::new(os)) };
            let mut gs = vec![0.0; n_ss * dim];
            rs.eval_basis(pt, &mut ps);
            rs.eval_grad_basis(pt, &mut gs);
            for i in 0..n_ss {
                for d in 0..dim {
                    gsp[i * dim + d] = (0..dim).map(|k| jit[(d, k)] * gs[i * dim + k]).sum();
                }
            }
        }
        for i in 0..n_v {
            for j in 0..n_s {
                let sc = j / n_ss;
                let sd = j % n_ss;
                g[(i, j)] -= wq * pv[i] * gsp[sd * dim + sc];
            }
        }
        for p in 0..n_s {
            let pc = p / n_ss;
            let pd = p % n_ss;
            for q in 0..n_s {
                let qc = q / n_ss;
                let qd = q % n_ss;
                if pc == qc {
                    ms[(p, q)] += wq * ps[pd] * ps[qd];
                }
            }
        }
    }
    (g, ms)
}

/// `(C_w, M_Σ)` of the wg Maxwell weak curl — the module's
/// `weak_curl_matrix` with the geometry source abstracted into `route`.
/// D1080 arithmetic: the pairing uses the **physical** curl of the mapped
/// Nédélec basis (`2-D curl̂/detJ`, `3-D (J·curl̂)/detJ`) with the D696
/// **signed** weight (`ip.weight·Trans.Weight()` — det cancels on affine
/// elements); the flux mass matrix keeps the |detJ| measure.
#[allow(clippy::needless_range_loop)]
fn curl_blocks<const D: usize>(
    mesh: &Mesh<D>, e: u32, order: usize, qo: u8, route: Route,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let dim = D;
    let nd_elem: Box<dyn VectorReferenceElement> =
        if dim == 2 { Box::new(TriNDk::new(order)) } else { Box::new(TetNDk::new(order)) };
    let n_v = nd_elem.n_dofs();
    let os = if order > 0 { order - 1 } else { 0 };
    let n_ss: usize;
    let mut ref_s: Option<Box<dyn ReferenceElement>> = None;
    if os == 0 {
        n_ss = 1;
    } else {
        let rs: Box<dyn ReferenceElement> =
            if dim == 2 { Box::new(TriPk::new(os)) } else { Box::new(TetPk::new(os)) };
        n_ss = rs.n_dofs();
        ref_s = Some(rs);
    }
    let n_s = if dim == 2 { n_ss } else { dim * n_ss };
    let qr = if dim == 2 { tri_rule(qo) } else { tet_rule(qo) };
    let mut cw = DMatrix::zeros(n_v, n_s);
    let mut ms = DMatrix::zeros(n_s, n_s);
    let mut nv_curl = vec![0.0_f64; n_v * dim];
    let mut ps = vec![0.0_f64; n_ss];
    for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
        let (jac, _xp) = route_geom(mesh, e, pt, dim, route);
        let det_j = jac.determinant();
        if det_j.abs() < 1e-30 {
            continue;
        }
        let wq_signed = w * det_j;
        let wq_abs = w * det_j.abs();
        nd_elem.eval_curl(pt, &mut nv_curl);
        if os == 0 {
            ps[0] = 1.0;
        } else {
            let rs = ref_s.as_ref().unwrap();
            rs.eval_basis(pt, &mut ps);
        }
        if dim == 2 {
            for i in 0..n_v {
                for j in 0..n_s {
                    cw[(i, j)] -= wq_signed * (nv_curl[i] / det_j) * ps[j];
                }
            }
            for p in 0..n_s {
                for q in 0..n_s {
                    ms[(p, q)] += wq_abs * ps[p] * ps[q];
                }
            }
        } else {
            for i in 0..n_v {
                for j in 0..n_s {
                    let sc = j / n_ss;
                    let sd = j % n_ss;
                    let curl_phys_sc =
                        (0..3).map(|k| jac[(sc, k)] * nv_curl[i * 3 + k]).sum::<f64>() / det_j;
                    cw[(i, j)] -= wq_signed * curl_phys_sc * ps[sd];
                }
            }
            for p in 0..n_s {
                let pc = p / n_ss;
                let pd = p % n_ss;
                for q in 0..n_s {
                    let qc = q / n_ss;
                    let qd = q % n_ss;
                    if pc == qc {
                        ms[(p, q)] += wq_abs * ps[pd] * ps[qd];
                    }
                }
            }
        }
    }
    (cw, ms)
}

/// The wg Poisson volume block (`G M_Σ⁻¹ Gᵀ`, scattered), through `route`.
fn poisson_volume_dense<const D: usize>(
    mesh: &Mesh<D>, order: usize, qo: u8, route: Route,
) -> Vec<Vec<f64>> {
    let sp = L2Space::new(mesh.clone(), order as u8);
    let n = sp.n_dofs();
    let mut d = vec![vec![0.0_f64; n]; n];
    for e in 0..mesh.n_elems() as u32 {
        let (g, ms) = grad_blocks::<D>(mesh, e, order, qo, route);
        let x = ms.clone().lu().solve(&g.transpose()).unwrap_or(DMatrix::zeros(g.ncols(), g.nrows()));
        let a = &g * x;
        let dofs: Vec<usize> = sp.element_dofs(e).iter().map(|&d| d as usize).collect();
        scatter(&mut d, &dofs, &a);
    }
    d
}

/// The wg Maxwell volume block (`C_w M_Σ⁻¹ C_wᵀ`, scattered), through `route`.
/// D1080: the scatter carries the element's H(curl) orientation signs —
/// `A ← S·A·S` per element (MFEM's signed `GetElementDofs` scatter,
/// `sparsemat.cpp:2795-2811`).
fn maxwell_volume_dense<const D: usize>(
    mesh: &Mesh<D>, order: usize, qo: u8, route: Route,
) -> Vec<Vec<f64>> {
    let sp = HCurlSpace::new(mesh.clone(), order as u8);
    let n = sp.n_dofs();
    let mut d = vec![vec![0.0_f64; n]; n];
    for e in 0..mesh.n_elems() as u32 {
        let (cw, ms) = curl_blocks::<D>(mesh, e, order, qo, route);
        let x =
            ms.clone().lu().solve(&cw.transpose()).unwrap_or(DMatrix::zeros(cw.ncols(), cw.nrows()));
        let a = &cw * x;
        let dofs: Vec<usize> = sp.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs: &[f64] = sp.element_signs(e);
        for (i, &di) in dofs.iter().enumerate() {
            let si = signs.get(i).copied().unwrap_or(1.0);
            for (j, &dj) in dofs.iter().enumerate() {
                let sj = signs.get(j).copied().unwrap_or(1.0);
                d[di][dj] += si * sj * a[(i, j)];
            }
        }
    }
    d
}

/// The wg Stokes velocity volume block + body-force rhs, through `route`
/// (the module's loops with the geometry source abstracted).  Takes the same
/// force the module under test is run with.
fn stokes_volume_dense<const D: usize>(
    mesh: &Mesh<D>, order: usize, qo: u8, route: Route, force: &dyn Fn(&[f64]) -> Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let vel = H1Space::new(mesh.clone(), order as u8);
    let n_vel = vel.n_dofs();
    let dim = D;
    let mut a = vec![vec![0.0_f64; n_vel]; n_vel];
    let mut rhs = vec![0.0_f64; n_vel];
    for e in 0..mesh.n_elems() as u32 {
        let (g, ms) = grad_blocks::<D>(mesh, e, order, qo, route);
        let x = ms.clone().lu().solve(&g.transpose()).unwrap_or(DMatrix::zeros(g.ncols(), g.nrows()));
        let a_e = &g * x;
        let v_dofs: Vec<usize> = vel.element_dofs(e).iter().map(|&d| d as usize).collect();
        scatter(&mut a, &v_dofs, &a_e);

        let ref_v: Box<dyn ReferenceElement> =
            if dim == 2 { Box::new(TriPk::new(order)) } else { Box::new(TetPk::new(order)) };
        let qr = if dim == 2 { tri_rule(qo) } else { tet_rule(qo) };
        let n_vl = v_dofs.len();
        let mut pv = vec![0.0; n_vl];
        for (pt, &w) in qr.points.iter().zip(qr.weights.iter()) {
            let (jac, xp) = route_geom(mesh, e, pt, dim, route);
            let wq = w * jac.determinant().abs();
            let f = force(&xp);
            ref_v.eval_basis(pt, &mut pv);
            for comp in 0..dim {
                for i in 0..n_vl / dim {
                    let row = v_dofs[i + comp * (n_vl / dim)];
                    rhs[row] += wq * pv[i] * f[comp];
                }
            }
        }
    }
    (a, rhs)
}

// ─── Tests: the geometry oracle ─────────────────────────────────────────────

#[test]
fn d831_fixtures_are_the_mfem_meshes() {
    check_mesh_identity(&straight_tri_mesh(), &parse_gold(GOLD_TRI_STRAIGHT), "tri straight");
    check_mesh_identity(&curved_tri_mesh(), &parse_gold(GOLD_TRI_CURVED), "tri curved");
    check_mesh_identity(&straight_tet_mesh(), &parse_gold(GOLD_TET_STRAIGHT), "tet straight");
    check_mesh_identity(&curved_tet_mesh(), &parse_gold(GOLD_TET_CURVED), "tet curved");
}

/// The primary oracle: `element_jacobian_at` — the single source the wg body
/// path is switched to — reproduces MFEM's `Jacobian()`/`Weight()`/
/// `Transform()` entry by entry on the curved fixtures.
#[test]
fn d831_element_jacobian_at_matches_mfem_on_curved_cells() {
    let (tj, tw, tx) = route_vs_gold::<3>(&curved_tet_mesh(), &parse_gold(GOLD_TET_CURVED), Route::Iso);
    let (rj, rw, rx) = route_vs_gold::<2>(&curved_tri_mesh(), &parse_gold(GOLD_TRI_CURVED), Route::Iso);
    println!(
        "curved iso route vs MFEM: tet (J {tj:.3e}, W {tw:.3e}, x {tx:.3e}), \
         tri (J {rj:.3e}, W {rw:.3e}, x {rx:.3e})"
    );
    assert!(tj < 1e-12 && tw < 1e-12 && tx < 1e-12, "tet: {tj:.3e} {tw:.3e} {tx:.3e}");
    assert!(rj < 1e-12 && rw < 1e-12 && rx < 1e-12, "tri: {rj:.3e} {rw:.3e} {rx:.3e}");
}

/// …and equally on the straight ones (there the map is affine, so both routes
/// are the same map — this is the round-off floor of the comparison).
#[test]
fn d831_element_jacobian_at_matches_mfem_on_straight_cells() {
    let (tj, tw, tx) =
        route_vs_gold::<3>(&straight_tet_mesh(), &parse_gold(GOLD_TET_STRAIGHT), Route::Iso);
    let (rj, rw, rx) =
        route_vs_gold::<2>(&straight_tri_mesh(), &parse_gold(GOLD_TRI_STRAIGHT), Route::Iso);
    println!(
        "straight iso route vs MFEM: tet (J {tj:.3e}, W {tw:.3e}, x {tx:.3e}), \
         tri (J {rj:.3e}, W {rw:.3e}, x {rx:.3e})"
    );
    assert!(tj < 1e-13 && tw < 1e-13 && tx < 1e-13, "tet: {tj:.3e} {tw:.3e} {tx:.3e}");
    assert!(rj < 1e-13 && rw < 1e-13 && rx < 1e-13, "tri: {rj:.3e} {rw:.3e} {rx:.3e}");
}

/// The negative pin: on **straight** meshes the pre-fix vertex route is the
/// affine map itself, so it matches MFEM to round-off — the switch to the
/// isoparametric source cannot move straight-mesh numbers.
#[test]
fn d831_vertex_route_matches_mfem_on_straight_cells() {
    let (tj, tw, tx) =
        route_vs_gold::<3>(&straight_tet_mesh(), &parse_gold(GOLD_TET_STRAIGHT), Route::Vertex);
    let (rj, rw, rx) =
        route_vs_gold::<2>(&straight_tri_mesh(), &parse_gold(GOLD_TRI_STRAIGHT), Route::Vertex);
    println!(
        "straight vertex route vs MFEM: tet (J {tj:.3e}, W {tw:.3e}, x {tx:.3e}), \
         tri (J {rj:.3e}, W {rw:.3e}, x {rx:.3e})"
    );
    assert!(tj < 1e-13 && tw < 1e-13 && tx < 1e-13, "tet: {tj:.3e} {tw:.3e} {tx:.3e}");
    assert!(rj < 1e-13 && rw < 1e-13 && rx < 1e-13, "tri: {rj:.3e} {rw:.3e} {rx:.3e}");
}

/// The teeth: on **curved** meshes the pre-fix vertex route is far from MFEM's
/// order-`g` map — in the Jacobian, the weight and the physical point alike.
#[test]
fn d831_vertex_route_fails_mfem_on_curved_cells() {
    let (tj, tw, tx) = route_vs_gold::<3>(&curved_tet_mesh(), &parse_gold(GOLD_TET_CURVED), Route::Vertex);
    let (rj, rw, rx) = route_vs_gold::<2>(&curved_tri_mesh(), &parse_gold(GOLD_TRI_CURVED), Route::Vertex);
    println!(
        "curved vertex route vs MFEM: tet (J {tj:.3e}, W {tw:.3e}, x {tx:.3e}), \
         tri (J {rj:.3e}, W {rw:.3e}, x {rx:.3e})"
    );
    assert!(tj > 1e-3 && tw > 1e-3 && tx > 1e-3, "tet witness has no teeth: {tj:.3e} {tw:.3e} {tx:.3e}");
    assert!(rj > 1e-3 && rw > 1e-3 && rx > 1e-3, "tri witness has no teeth: {rj:.3e} {rw:.3e} {rx:.3e}");
}

// ─── Tests: the module bindings ─────────────────────────────────────────────

/// Shared body of the binding check: on the curved mesh the module's volume
/// block IS the isoparametric re-assembly and is far from the vertex-route
/// one; on the straight mesh the two routes (hence the module, either way)
/// coincide to round-off.
fn volume_binding(
    name: &str,
    curved: (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>),
    straight: (Vec<Vec<f64>>, Vec<Vec<f64>>),
) {
    let (wg_c, iso_c, vtx_c) = curved;
    let rel_iso = max_rel_diff(&wg_c, &iso_c);
    let rel_vtx = max_rel_diff(&wg_c, &vtx_c);
    println!("{name} curved: vs iso={rel_iso:.3e}, vs pre-fix vertex route={rel_vtx:.3e}");
    assert!(rel_iso < 1e-12, "{name}: volume block != isoparametric ({rel_iso:.3e})");
    assert!(rel_vtx > 1e-2, "{name}: the vertex witness has no teeth ({rel_vtx:.3e})");

    let (wg_s, vtx_s) = straight;
    let rel_s = max_rel_diff(&wg_s, &vtx_s);
    println!("{name} straight: vs pre-fix vertex route={rel_s:.3e}");
    assert!(rel_s < 1e-12, "{name} straight: the switch moved numbers ({rel_s:.3e})");
}

#[test]
fn d831_wg_poisson_volume_block_is_the_isoparametric_one_2d() {
    const D: usize = 2;
    // Order 2: at order 1 the sigma space is P₀ (`os = 0`, `gsp ≡ 0`) and the
    // volume block never touches the Jacobian at all — there would be nothing
    // to bind and no teeth.
    let order: usize = 2;
    let (curved, straight) = (curved_tri_mesh(), straight_tri_mesh());
    let sp_c = L2Space::new(curved.clone(), order as u8);
    let sp_s = L2Space::new(straight.clone(), order as u8);
    let wg_c = dense_of(&assemble_wg_poisson(&sp_c, 3, 0.0), sp_c.n_dofs());
    let wg_s = dense_of(&assemble_wg_poisson(&sp_s, 3, 0.0), sp_s.n_dofs());
    volume_binding(
        "poisson tri",
        (
            wg_c,
            poisson_volume_dense::<D>(&curved, order, 3, Route::Iso),
            poisson_volume_dense::<D>(&curved, order, 3, Route::Vertex),
        ),
        (wg_s, poisson_volume_dense::<D>(&straight, order, 3, Route::Vertex)),
    );
}

#[test]
fn d831_wg_poisson_volume_block_is_the_isoparametric_one_3d() {
    const D: usize = 3;
    let order: usize = 2;
    let (curved, straight) = (curved_tet_mesh(), straight_tet_mesh());
    let sp_c = L2Space::new(curved.clone(), order as u8);
    let sp_s = L2Space::new(straight.clone(), order as u8);
    let wg_c = dense_of(&assemble_wg_poisson(&sp_c, 3, 0.0), sp_c.n_dofs());
    let wg_s = dense_of(&assemble_wg_poisson(&sp_s, 3, 0.0), sp_s.n_dofs());
    volume_binding(
        "poisson tet",
        (
            wg_c,
            poisson_volume_dense::<D>(&curved, order, 3, Route::Iso),
            poisson_volume_dense::<D>(&curved, order, 3, Route::Vertex),
        ),
        (wg_s, poisson_volume_dense::<D>(&straight, order, 3, Route::Vertex)),
    );
}

#[test]
fn d831_wg_maxwell_volume_block_is_the_isoparametric_one_2d() {
    const D: usize = 2;
    let (curved, straight) = (curved_tri_mesh(), straight_tri_mesh());
    let sp_c = HCurlSpace::new(curved.clone(), 1);
    let sp_s = HCurlSpace::new(straight.clone(), 1);
    let (wg_c, _) = assemble_wg_maxwell(&sp_c, 3, 0.0, &[]);
    let (wg_s, _) = assemble_wg_maxwell(&sp_s, 3, 0.0, &[]);
    volume_binding(
        "maxwell tri",
        (
            dense_of(&wg_c, sp_c.n_dofs()),
            maxwell_volume_dense::<D>(&curved, 1, 3, Route::Iso),
            maxwell_volume_dense::<D>(&curved, 1, 3, Route::Vertex),
        ),
        (dense_of(&wg_s, sp_s.n_dofs()), maxwell_volume_dense::<D>(&straight, 1, 3, Route::Vertex)),
    );
}

#[test]
fn d831_wg_maxwell_volume_block_is_the_isoparametric_one_3d() {
    const D: usize = 3;
    let (curved, straight) = (curved_tet_mesh(), straight_tet_mesh());
    let sp_c = HCurlSpace::new(curved.clone(), 1);
    let sp_s = HCurlSpace::new(straight.clone(), 1);
    let (wg_c, _) = assemble_wg_maxwell(&sp_c, 3, 0.0, &[]);
    let (wg_s, _) = assemble_wg_maxwell(&sp_s, 3, 0.0, &[]);
    volume_binding(
        "maxwell tet",
        (
            dense_of(&wg_c, sp_c.n_dofs()),
            maxwell_volume_dense::<D>(&curved, 1, 3, Route::Iso),
            maxwell_volume_dense::<D>(&curved, 1, 3, Route::Vertex),
        ),
        (dense_of(&wg_s, sp_s.n_dofs()), maxwell_volume_dense::<D>(&straight, 1, 3, Route::Vertex)),
    );
}

/// The Stokes binding covers both body consumers: the `A` block (Jacobian
/// route) and the body-force rhs (physical-point route).
#[test]
fn d831_wg_stokes_volume_block_is_the_isoparametric_one() {
    const D: usize = 2;
    let (curved, straight) = (curved_tri_mesh(), straight_tri_mesh());
    let vel_c = H1Space::new(curved.clone(), 2);
    let pres_c = L2Space::new(curved.clone(), 1);
    let vel_s = H1Space::new(straight.clone(), 2);
    let pres_s = L2Space::new(straight.clone(), 1);
    let f = |x: &[f64]| vec![x[0] + 1.0, x[1]];
    let (k_c, rhs_c) = assemble_wg_stokes(&vel_c, &pres_c, 3, 0.0, &f, &[]);
    let (k_s, rhs_s) = assemble_wg_stokes(&vel_s, &pres_s, 3, 0.0, &f, &[]);
    let n_c = vel_c.n_dofs();
    let n_s = vel_s.n_dofs();

    let (a_iso, rhs_iso) = stokes_volume_dense::<D>(&curved, 2, 3, Route::Iso, &f);
    let (a_vtx, rhs_vtx) = stokes_volume_dense::<D>(&curved, 2, 3, Route::Vertex, &f);
    let a_wg_c = dense_of(&k_c, n_c)[..n_c].to_vec();
    let a_iso_c = a_iso[..n_c].to_vec();
    let a_vtx_c = a_vtx[..n_c].to_vec();

    volume_binding("stokes tri", (a_wg_c, a_iso_c, a_vtx_c), (dense_of(&k_s, n_s), {
        let (a_v, _) = stokes_volume_dense::<D>(&straight, 2, 3, Route::Vertex, &f);
        a_v[..n_s].to_vec()
    }));

    let r_iso = max_rel_vec(&rhs_c, &rhs_iso);
    let r_vtx = max_rel_vec(&rhs_c, &rhs_vtx);
    let (_, rhs_v_s) = stokes_volume_dense::<D>(&straight, 2, 3, Route::Vertex, &f);
    let r_s = max_rel_vec(&rhs_s, &rhs_v_s);
    println!("stokes rhs curved: vs iso={r_iso:.3e}, vs pre-fix vertex route={r_vtx:.3e}; straight: {r_s:.3e}");
    assert!(r_iso < 1e-12, "stokes rhs != isoparametric ({r_iso:.3e})");
    assert!(r_vtx > 1e-2, "stokes rhs vertex witness has no teeth ({r_vtx:.3e})");
    assert!(r_s < 1e-12, "stokes straight rhs moved ({r_s:.3e})");
}
