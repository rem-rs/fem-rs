//! D799-3 — DG face geometry of `dg_advection` / `dg_elasticity` /
//! `dg_hyperbolic` on the **isoparametric** route (the D795-1 mechanism).
//!
//! # The debt
//!
//! `dg.rs` was moved to `face_point_geom` in D795-1, but three modules kept the
//! pre-D795 route: a normal built from the face's **corner nodes**
//! (`(dy/h, -dx/h)` in 2-D, three face vertices in 3-D), a QP weight scaled by
//! that chord, and the element reference point recovered by *inverting the
//! physical map* (`phys_to_ref` over an affine or centroid Jacobian).  MFEM
//! instead composes the face transformation through `Elem1`'s isoparametric
//! map: `Loc1`, `nor = CalcOrtho(Trans.Jacobian())` (magnitude = face measure),
//! `Trans.Elem1->Weight()` and `Elem1->Transform(eip)`.
//!
//! # Mesh
//!
//! The d795 fixture: 2×2 quadrilateral elements on `[0,1]²`,
//! `SetCurvature(3, true)` (order-3 discontinuous geometry), then the smooth
//! bend `(X,Y) ↦ (X(1+0.15Y(1−Y)), Y+0.15XY(1−Y))`.
//!
//! # Gold
//!
//! `tmp/d805/d805_probe.cpp` dumps MFEM 4.10's own numbers over the same mesh:
//! `[FQP]` (per-QP `nor`, `eip1`, `Elem1->Transform(eip1)`, `Weight()`),
//! `[MATDIF]` and `[MATELA]`/`[MATELAV]` (`DGElasticityIntegrator` with and
//! without the interior faces), compared by `tmp/d805/cmp.py`.

use fem_assembly::dg::dg_advection::{DGAdvectionIntegrator, assemble_dg_interior_faces};
use fem_assembly::dg::dg_base::{face_point_geom, ref_elem_face};
use fem_assembly::dg::dg_elasticity::DgElasticityAssembler;
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_assembly::{Assembler, DgAssembler, InteriorFaceList};
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::L2Space;

/// Tolerance for every gold comparison (17-digit dumps; the two codes sum the
/// same products in a different order, so ~1e-13 is the recoverable limit).
const RTOL: f64 = 1e-12;

fn rel(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    let s = a.abs().max(b.abs());
    if s < 1e-12 { d } else { d / s }
}

/// The d795 fixture's 2×2 quad mesh: flat vertex/connectivity tables exactly as
/// MFEM's `MakeCartesian2D(2, 2, QUADRILATERAL, 1, 1, 1)` numbers them.
fn build_mesh(curved: bool) -> Mesh<2> {
    let mut c = Vec::new();
    for j in 0..3 {
        for i in 0..3 {
            c.push(i as f64 * 0.5);
            c.push(j as f64 * 0.5);
        }
    }
    let e: Vec<u32> = vec![0, 1, 4, 3, 3, 4, 7, 6, 4, 5, 8, 7, 1, 2, 5, 4];
    let bn: Vec<u32> = vec![0, 1, 1, 2, 7, 6, 8, 7, 3, 0, 6, 3, 2, 5, 5, 8];
    let bt: Vec<i32> = vec![1, 1, 3, 3, 4, 4, 2, 2];
    let mut m = Mesh::<2>::uniform(
        c,
        e,
        vec![1; 4],
        ElementType::Quad4,
        bn,
        bt,
        ElementType::Line2,
    );
    if curved {
        m.set_curvature(3);
        m.transform(|[x, y]| {
            [x * (1.0 + 0.15 * y * (1.0 - y)), y + 0.15 * x * y * (1.0 - y)]
        });
    }
    m
}

/// The owner element of boundary face `f` (the mesh stores only boundary faces,
/// so scan the elements for one that owns the pair).
fn owner_of(m: &Mesh<2>, f: u32) -> u32 {
    let fnodes = m.face_nodes(f);
    for e in m.elem_iter() {
        let en = m.element_nodes(e);
        let nv = en.len();
        for le in 0..nv {
            let (p, q) = (en[le], en[(le + 1) % nv]);
            if (p == fnodes[0] && q == fnodes[1]) || (p == fnodes[1] && q == fnodes[0]) {
                return e;
            }
        }
    }
    panic!("no owner for boundary face {f}");
}

/// The pre-D795 (legacy) face route: chord from the element's **flat corner**
/// node coordinates and an outward unit normal chosen by the element centroid —
/// a literal transcription of the code D799-3 replaced, used as the "teeth"
/// reference (the gold must fail on it).
fn legacy_chord_nor(m: &Mesh<2>, elem: u32, a: u32, b: u32) -> [f64; 2] {
    let p0 = m.node_coords(a);
    let p1 = m.node_coords(b);
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let len = (dx * dx + dy * dy).sqrt();
    let mut n = [dy / len, -dx / len];
    let en = m.element_nodes(elem);
    let (mut cx, mut cy) = (0.0, 0.0);
    for &nid in en {
        let c = m.node_coords(nid);
        cx += c[0];
        cy += c[1];
    }
    let k = en.len() as f64;
    let (mx, my) = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5);
    if n[0] * (mx - cx / k) + n[1] * (my - cy / k) < 0.0 {
        n = [-n[0], -n[1]];
    }
    [len * n[0], len * n[1]]
}

fn dense(mat: &fem_linalg::CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = mat.nrows;
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            out[i][mat.col_idx[k] as usize] = mat.values[k];
        }
    }
    out
}

fn print_mat(tag: &str, m: &[Vec<f64>]) {
    println!("[{tag}] n={}", m.len());
    for row in m {
        print!("   ");
        for v in row {
            print!(" {v:.17e}");
        }
        println!();
    }
}

/// Dump every quantity `tmp/d805/cmp.py` compares, in its `[RS_*]` format.
/// Run with `cargo test --release -p fem-assembly --test d805_d799_3_dg_face_geometry
/// dg_face_geometry_dump -- --nocapture`.
#[test]
fn dg_face_geometry_dump() {
    if std::env::var("D805_DUMP").is_err() {
        return;
    }
    let m = build_mesh(true);
    let space = L2Space::new(m.clone(), 1);
    let ifl = InteriorFaceList::build(&m);
    let q_face = ref_elem_face(ElementType::Line2, 1).quadrature(2);

    // ── [RS_FQP] / [RS_LEGACY] boundary-face geometry ──────────────────────
    for f in 0..m.n_boundary_faces() as u32 {
        let fnodes = m.face_nodes(f);
        let (a, b) = (fnodes[0], fnodes[1]);
        let elem = owner_of(&m, f);
        for (q, xi) in q_face.points.iter().enumerate() {
            let g = face_point_geom(&m, elem, a, b, xi[0]);
            println!(
                "[RS_FQP] {elem} {a} {b} {q} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                g.nor[0], g.nor[1], g.eip[0], g.eip[1], g.xp[0], g.xp[1], g.det_j
            );
            let ln = legacy_chord_nor(&m, elem, a, b);
            println!(
                "[RS_LEGACY] {elem} {a} {b} {q} {:.17e} {:.17e}",
                ln[0], ln[1]
            );
        }
    }

    // ── [RS_MATDIF]: volume Diffusion + interior DGDiffusion (dg.rs control)
    let dif = DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, 2, Some(&[]));
    print_mat("RS_MATDIF", &dense(&dif));

    // ── [RS_MATDIFIF]: interior faces only (= MATDIF − the volume term)
    let vol = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let vd = dense(&vol);
    let dd = dense(&dif);
    let n = dd.len();
    let difif: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| dd[i][j] - vd[i][j]).collect())
        .collect();
    print_mat("RS_MATDIFIF", &difif);

    // ── [RS_MATADV]: `assemble_dg_interior_faces` + NonconservativeDGTrace ──
    // D805-2/round 76: the rule order is MFEM's `DGTraceIntegrator` default
    // `min(OrderW1,OrderW2) + 2*max(o1,o2)` (+1 for Pk) = 5 + 2*1 = 7 on this
    // curved fixture (4 points), NOT the old `quad_order = 2` (2 points); the
    // gold `tmp/d805/cpp_truth.txt` was produced with MFEM's default.
    let dg_adv = DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]), alpha: -1.0 };
    let mut coo = fem_linalg::CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
    assemble_dg_interior_faces(&mut coo, &m, &space, &ifl, 1, 7, &dg_adv);
    print_mat("RS_MATADV", &dense(&coo.into_csr()));

    // ── [RS_MATELA] / [RS_MATELAV]: DG elasticity with/without interior faces
    let n_elem = m.n_elements();
    let lam = vec![1.0; n_elem];
    let mu = vec![1.0; n_elem];
    let empty = InteriorFaceList { faces: Vec::new() };
    let ela = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 4.0, -1.0, 2, 2, &[],
    );
    let elav = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &empty, &lam, &mu, 4.0, -1.0, 2, 2, &[],
    );
    print_mat("RS_MATELA", &dense(&ela));
    print_mat("RS_MATELAV", &dense(&elav));

    // ── [RS_MATELZ] / [RS_MATELZV]: kappa = 0 — MFEM's consistency + symmetry
    // blocks only, the part that is pure face geometry.
    let elz = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 0.0, -1.0, 2, 2, &[],
    );
    let elzv = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &empty, &lam, &mu, 0.0, -1.0, 2, 2, &[],
    );
    print_mat("RS_MATELZ", &dense(&elz));
    print_mat("RS_MATELZV", &dense(&elzv));
}

// ─── MFEM 4.10 gold: per-QP face geometry on the curved fixture ─────────────
//
// `tmp/d805/d805_probe.cpp` `[FQP]` rows: for every boundary face of the curved
// 2×2 mesh and every `IntRules.Get(GEOMETRY::SEGMENT, 2)` point, MFEM's
// `Tr.SetAllIntPoints(&ip)`; `nor = CalcOrtho(Tr.Jacobian())`;
// `eip1 = Tr.GetElement1IntPoint()`; `x1 = Tr.Elem1->Transform(eip1)`;
// `W1 = Tr.Elem1->Weight()`.
//
// Key = `(owner element, face node a, face node b, q)`; the node pair is
// fem-rs' `mesh.face_nodes(f)`, which reproduces MFEM's boundary-element vertex
// order on this mesh.  Columns: `nor0, nor1, eip0, eip1, xp0, xp1, det_j`.
const MFEM_FQP: [(u32, u32, u32, usize, f64, f64, f64, f64, f64, f64, f64); 16] = [
    (0, 0, 1, 0, 0.00000000000000000e+00, -4.99999999999999889e-01, 2.11324865405187107e-01, 0.00000000000000000e+00, 1.05662432702593539e-01, 0.00000000000000000e+00, 2.53962341226347144e-01),
    (0, 0, 1, 1, 0.00000000000000000e+00, -5.00000000000000111e-01, 7.88675134594812866e-01, 0.00000000000000000e+00, 3.94337567297406377e-01, 0.00000000000000000e+00, 2.64787658773653012e-01),
    (3, 1, 2, 0, 0.00000000000000000e+00, -4.99999999999999389e-01, 2.11324865405187107e-01, 0.00000000000000000e+00, 6.05662432702593456e-01, 0.00000000000000000e+00, 2.72712341226346910e-01),
    (3, 1, 2, 1, 0.00000000000000000e+00, -5.00000000000000222e-01, 7.88675134594812866e-01, 0.00000000000000000e+00, 8.94337567297406433e-01, 0.00000000000000000e+00, 2.83537658773653001e-01),
    (1, 7, 6, 0, -8.32667268468867405e-16, 5.00000000000000333e-01, 7.88675134594812866e-01, 1.00000000000000000e+00, 3.94337567297406377e-01, 9.99999999999999889e-01, 2.35212341226347876e-01),
    (1, 7, 6, 1, 4.44089209850062616e-16, 4.99999999999999889e-01, 2.11324865405187134e-01, 1.00000000000000000e+00, 1.05662432702593567e-01, 1.00000000000000000e+00, 2.46037658773653273e-01),
    (2, 8, 7, 0, -8.32667268468867405e-16, 5.00000000000000666e-01, 7.88675134594812866e-01, 1.00000000000000000e+00, 8.94337567297406433e-01, 9.99999999999999889e-01, 2.16462341226347665e-01),
    (2, 8, 7, 1, 4.44089209850062616e-16, 4.99999999999999778e-01, 2.11324865405187134e-01, 1.00000000000000000e+00, 6.05662432702593456e-01, 1.00000000000000000e+00, 2.27287658773652368e-01),
    (0, 3, 0, 0, -5.00000000000000333e-01, -0.00000000000000000e+00, 0.00000000000000000e+00, 7.88675134594812866e-01, 0.00000000000000000e+00, 3.94337567297406377e-01, 2.58956329386826500e-01),
    (0, 3, 0, 1, -4.99999999999999889e-01, -0.00000000000000000e+00, 0.00000000000000000e+00, 2.11324865405187134e-01, 0.00000000000000000e+00, 1.05662432702593567e-01, 2.53543670613173566e-01),
    (1, 6, 3, 0, -5.00000000000000666e-01, -0.00000000000000000e+00, 0.00000000000000000e+00, 7.88675134594812866e-01, 0.00000000000000000e+00, 8.94337567297406433e-01, 2.53543670613173899e-01),
    (1, 6, 3, 1, -4.99999999999999778e-01, -0.00000000000000000e+00, 0.00000000000000000e+00, 2.11324865405187134e-01, 0.00000000000000000e+00, 6.05662432702593456e-01, 2.58956329386826223e-01),
    (3, 2, 5, 0, 5.59150635094610826e-01, -5.91506350946101045e-02, 1.00000000000000000e+00, 2.11324865405187107e-01, 1.01417468245269449e+00, 1.19837115155288054e-01, 2.83118988160479479e-01),
    (3, 2, 5, 1, 5.15849364905389129e-01, -1.58493649053892405e-02, 1.00000000000000000e+00, 7.88675134594812866e-01, 1.03582531754730534e+00, 4.30162884844711879e-01, 2.66881011839521398e-01),
    (2, 5, 8, 0, 4.84150635094610260e-01, 1.58493649053900731e-02, 1.00000000000000000e+00, 2.11324865405187107e-01, 1.03582531754730534e+00, 6.41487750249898903e-01, 2.51031646934132324e-01),
    (2, 5, 8, 1, 4.40849364905389063e-01, 5.91506350946109372e-02, 1.00000000000000000e+00, 7.88675134594812866e-01, 1.01417468245269449e+00, 9.08512249750100920e-01, 2.23968353065868014e-01),
];

/// The shared isoparametric face route: `face_point_geom` must reproduce MFEM's
/// `nor`, `eip1`, `Elem1->Transform(eip1)` and `Weight()` on every boundary face
/// of the curved fixture — and the pre-fix chord route must be far off on the
/// **curved** edges (faces 6/7, nodes `(2,5)`/`(5,8)`), which is what makes this
/// pin bite.
#[test]
fn dg_face_geometry_matches_mfem_on_curved_faces() {
    let m = build_mesh(true);
    let q_face = ref_elem_face(ElementType::Line2, 1).quadrature(2);
    assert_eq!(q_face.points.len(), 2, "MFEM's SEGMENT order-2 rule has 2 points");
    let mut worst = 0.0_f64;
    let mut worst_key = (0u32, 0u32, 0u32, 0usize);
    let mut n_curved_teeth = 0;
    for &(e1, a, b, q, nor0, nor1, eip0, eip1, xp0, xp1, det) in MFEM_FQP.iter() {
        let g = face_point_geom(&m, e1, a, b, q_face.points[q][0]);
        let r = [
            rel(g.nor[0], nor0),
            rel(g.nor[1], nor1),
            rel(g.eip[0], eip0),
            rel(g.eip[1], eip1),
            rel(g.xp[0], xp0),
            rel(g.xp[1], xp1),
            rel(g.det_j, det),
        ]
        .into_iter()
        .fold(0.0_f64, f64::max);
        if r > worst {
            worst = r;
            worst_key = (e1, a, b, q);
        }
        // teeth: on the two bent edges the pre-fix chord normal is off by >30%;
        // on the straight edges it is *identical* (that is the no-move property).
        let ln = legacy_chord_nor(&m, e1, a, b);
        let rl = rel(ln[0], nor0).max(rel(ln[1], nor1));
        let curved = (a == 2 && b == 5) || (a == 5 && b == 8);
        if curved {
            assert!(
                rl > 1e-1,
                "e1={e1} v=({a},{b}) q={q}: the pre-fix chord route differs from MFEM by only \
                 {rl:.3e} — this pin would not bite on the curved edge"
            );
            n_curved_teeth += 1;
        } else {
            assert!(
                rl < 1e-12,
                "e1={e1} v=({a},{b}) q={q}: on a straight edge the two routes must agree, \
                 but they differ by {rl:.3e}"
            );
        }
    }
    assert_eq!(n_curved_teeth, 4, "four (face, qp) rows sit on the bent edges");
    assert!(
        worst < RTOL,
        "curved face geometry: max relative deviation {worst:.3e} at {worst_key:?} vs MFEM"
    );
}

/// `|nor|`/unit-`nor` on a **straight** mesh must equal the pre-fix chord
/// construction exactly (algebraic identity), for boundary *and* interior faces
/// — the property that keeps every straight-mesh example on its old numbers.
#[test]
fn dg_face_geometry_is_algebraically_identical_on_straight_meshes() {
    let m = build_mesh(false);
    assert_eq!(m.geom_order(), 1, "the fixture must be straight here");
    let q_face = ref_elem_face(ElementType::Line2, 1).quadrature(2);
    let ifl = InteriorFaceList::build(&m);
    assert_eq!(ifl.faces.len(), 4);

    let mut checked = 0;
    for f in 0..m.n_boundary_faces() as u32 {
        let fnodes = m.face_nodes(f);
        let elem = owner_of(&m, f);
        for xi in q_face.points.iter() {
            let g = face_point_geom(&m, elem, fnodes[0], fnodes[1], xi[0]);
            let ln = legacy_chord_nor(&m, elem, fnodes[0], fnodes[1]);
            let h = (ln[0] * ln[0] + ln[1] * ln[1]).sqrt();
            let hn = (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1]).sqrt();
            assert!(
                (hn - h).abs() <= 1e-15 * h.abs().max(1.0),
                "|nor| {hn:.17e} vs chord {h:.17e} on boundary face {f}"
            );
            // unit normals agree component-wise
            assert!(
                rel(g.nor[0] / hn, ln[0] / h) < 1e-14 && rel(g.nor[1] / hn, ln[1] / h) < 1e-14,
                "unit nor ({:.17e},{:.17e}) vs chord ({:.17e},{:.17e}) on boundary face {f}",
                g.nor[0] / hn, g.nor[1] / hn, ln[0] / h, ln[1] / h
            );
            checked += 1;
        }
    }
    for f in &ifl.faces {
        let (a, b) = (f.face_nodes[0], f.face_nodes[1]);
        for xi in q_face.points.iter() {
            for &(elem, rev) in &[(f.elem_left, false), (f.elem_right, true)] {
                let t = if rev { 1.0 - xi[0] } else { xi[0] };
                let g = face_point_geom(&m, elem, a, b, t);
                let ln = legacy_chord_nor(&m, elem, a, b);
                let h = (ln[0] * ln[0] + ln[1] * ln[1]).sqrt();
                let hn = (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1]).sqrt();
                assert!(
                    (hn - h).abs() <= 1e-15 * h.abs().max(1.0),
                    "interior face {a}-{b} elem {elem}: |nor| {hn:.17e} vs chord {h:.17e}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 24, "expected at least 24 geometry rows, checked {checked}");
}
