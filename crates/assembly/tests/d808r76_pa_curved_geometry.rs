//! D808-4 — the hex/quad partial-assembly kernels' geometry on a **curved**
//! mesh (`geom_order >= 2`).
//!
//! ## The debt
//!
//! `tmp/round3_plan.md` round-75, `D808-4`:
//!
//! > 其余 PA 核（`hex_qk`/`quad_qk`/`quad_q1`/`q2..q4`/`tet4`）在 `geom_order ≥ 2`
//! > 时仍用 P1 顶点几何，与装配侧 order-g 等参不一致，与 D783 同类同量级。
//!
//! Every one of those builders derived its per-QP Jacobian and physical point
//! from the element's **vertices** — an exact representation only while the
//! mesh is straight.  On a mesh carrying an order-`g` geometry table the
//! assembled path integrates that table's map (`assembler::geo_ref_elem` →
//! `fem_mesh::element_jacobian_at`), so PA and SpMV silently disagreed by
//! `O(h)` — the same defect D783 fixed in `pa::prism_pk`.
//!
//! ## What this file pins
//!
//! * [`curved_hex_mesh`] / [`curved_quad_mesh`] attach an order-2 geometry
//!   table (`Mesh::set_curvature(2)`) and then move its **non-vertex** geometry
//!   nodes, so the vertex map and the table map are genuinely different
//!   functions while the element corners stay exact (the mesh stays valid).
//! * For every kernel of the family, the PA apply on that mesh must reproduce
//!   the assembled SpMV to round-off.  The **teeth** are built inside the test:
//!   [`straightened_hex_mesh`] / [`straightened_quad_mesh`] drop the geometry
//!   table, which reproduces the pre-fix PA data exactly, and the same
//!   comparison must then be off by `O(1e-1)`.  The witness can never rot,
//!   because it is the same fixture with one field cleared.
//! * The straight path must not move: on a straight mesh the builders take the
//!   verbatim pre-fix branch (guarded by `curved_jacobian`'s
//!   `geom_order() >= 2` test, pinned in `pa::curved`'s own unit tests), and
//!   PA vs assembly still agrees there for every kernel.
//!
//! ## D808-4's list omitted `hex_q1`
//!
//! The registration names `hex_qk`/`quad_qk`/`quad_q1`/`q2..q4`/`tet4` but not
//! `build_hex_q1_pa_data`, which used the vertices the same way.  It is covered
//! here (discipline ⑰: a same-named dispatch family is swept whole), and the
//! omission is recorded as a registration gap.  `build_tet4_pa_data` is the one
//! arm the D783 recipe does **not** fit — its PA data is a *single centroid*
//! quadrature point, so even with the order-`g` Jacobian at that point the
//! quadrature is not exact on a curved element; it is registered separately
//! rather than silently half-fixed.

use fem_assembly::pa::{
    build_hex_q1_pa_data, build_hex_q2_pa_data, build_hex_q3_pa_data, build_hex_q4_pa_data,
    build_hex_qk_pa_data, build_quad_q1_pa_data, build_quad_q2_pa_data, build_quad_qk_pa_data,
    pa_apply_hex_q1, pa_apply_hex_q2, pa_apply_hex_q3, pa_apply_hex_q4, pa_apply_hex_qk,
    pa_apply_quad_q1, pa_apply_quad_q2, pa_apply_quad_qk,
};
use fem_assembly::pa::types::PaData;
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// The curved fixture's smooth, vertex-vanishing bulge.  Only the extra
/// geometry nodes move, so the element corners stay exact — while the `P1`
/// vertex map and the table's `Pk(2)` map become different functions.
///
/// The amplitude is deliberately small: a Q2 map that overshoots far enough
/// near a corner **folds**, and the PA kernels store `|det J|` (their
/// degeneracy guard) while the assembler carries the *signed* determinant
/// (D679), so a folded fixture would compare two different quantities and
/// report a spurious `O(1)` disagreement.  [`assert_no_fold`] is what keeps
/// this from ever happening silently — it caught exactly that on the first
/// version of this file (min `det J = −2.79e-2` at the corner `ξ ≈ 0.953`).
fn bulge3(c: [f64; 3]) -> [f64; 3] {
    let s = (std::f64::consts::PI * c[0]).sin()
        * (std::f64::consts::PI * c[1]).sin()
        * (std::f64::consts::PI * c[2]).sin();
    [0.035 * s, 0.030 * s, 0.045 * s]
}

fn bulge2(c: [f64; 2]) -> [f64; 2] {
    let s = (std::f64::consts::PI * c[0]).sin() * (std::f64::consts::PI * c[1]).sin();
    [0.035 * s, 0.045 * s]
}

/// The fixture must have a positive Jacobian everywhere the kernels sample it,
/// so that `|det J|` and the signed `det J` agree.  Checked on the finest
/// quadrature set any kernel of the family uses (5 Gauss points per direction).
fn assert_no_fold<const D: usize>(mesh: &Mesh<D>, name: &str) {
    let (pts, _w) = fem_element::quadrature::gauss_legendre_01(5);
    let mut min_det = f64::INFINITY;
    let mut at = (0u32, [0.0_f64; D]);
    for e in 0..mesh.n_elems() as u32 {
        let mut xi = vec![0.0_f64; D];
        fn walk<const D: usize>(
            mesh: &Mesh<D>,
            e: u32,
            depth: usize,
            xi: &mut Vec<f64>,
            pts: &[f64],
            min_det: &mut f64,
            at: &mut (u32, [f64; D]),
        ) {
            if depth == D {
                let arr: [f64; D] = std::array::from_fn(|k| xi[k]);
                let (jac, _) = fem_mesh::transformation::element_jacobian_at(mesh, e, &arr, D);
                let det = jac.determinant();
                if det < *min_det {
                    *min_det = det;
                    *at = (e, arr);
                }
                return;
            }
            for &p in pts {
                xi[depth] = p;
                walk(mesh, e, depth + 1, xi, pts, min_det, at);
            }
        }
        walk(mesh, e, 0, &mut xi, &pts, &mut min_det, &mut at);
    }
    assert!(
        min_det > 0.0,
        "{name}: the curved fixture folds — min det J = {min_det:.6e} at element {} xi {:?}; \
         reduce the bulge amplitude or the comparison would mix |det J| with the signed det J \
         (D679)",
        at.0,
        at.1
    );
}

/// Move every geometry node past the shared vertex head of the table.
fn displace_geometry<const D: usize>(
    mesh: &mut Mesh<D>,
    n_vertices: usize,
    disp: &dyn Fn([f64; D]) -> [f64; D],
) {
    let g = mesh.geometry.as_mut().expect("set_curvature(2) attaches a table");
    for node in n_vertices..g.n_nodes {
        let off = node * D;
        let c: [f64; D] = std::array::from_fn(|d| g.coords[off + d]);
        let d = disp(c);
        for k in 0..D {
            g.coords[off + k] += d[k];
        }
    }
}

fn curved_hex_mesh() -> Mesh<3> {
    let mut mesh = Mesh::<3>::unit_cube_hex(2);
    let n_vertices = mesh.n_nodes();
    mesh.set_curvature(2);
    displace_geometry(&mut mesh, n_vertices, &bulge3);
    mesh
}

fn curved_quad_mesh() -> Mesh<2> {
    let mut mesh = Mesh::<2>::unit_square_quad(2);
    let n_vertices = mesh.n_nodes();
    mesh.set_curvature(2);
    displace_geometry(&mut mesh, n_vertices, &bulge2);
    mesh
}

/// The curved fixture with the geometry table dropped: the straight-edged `P1`
/// approximation the pre-D808-4 PA always used.
fn straightened_hex_mesh() -> Mesh<3> {
    let mut mesh = curved_hex_mesh();
    mesh.geometry = None;
    mesh
}

fn straightened_quad_mesh() -> Mesh<2> {
    let mut mesh = curved_quad_mesh();
    mesh.geometry = None;
    mesh
}

/// `(max |PA·x − A·x| / max |A·x|, DOFs shared by elements 0 and 1)`.
///
/// `x` is DOF-index based, so a permutation or a metric error cannot hide
/// behind the constant null space.  `order` is the assembled quadrature order
/// whose point count matches the kernel's own rule (see each call site).
fn pa_vs_assembled<const D: usize>(
    pd_mesh: &Mesh<D>,
    space_mesh: &Mesh<D>,
    order: u8,
    p: u8,
    build: &dyn Fn(&Mesh<D>, u8) -> PaData,
    apply: &dyn Fn(&PaData, &[Vec<u32>], u8, &[f64], &mut [f64]),
) -> (f64, usize) {
    let space = H1Space::new(space_mesh.clone(), p);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], order);
    let pd = build(pd_mesh, p);
    let elem_dofs: Vec<Vec<u32>> = (0..space_mesh.n_elems() as u32)
        .map(|e| space.element_dofs(e).to_vec())
        .collect();
    let shared = elem_dofs[0].iter().filter(|d| elem_dofs[1].contains(d)).count();
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_pa = vec![0.0; n];
    apply(&pd, &elem_dofs, p, &x, &mut y_pa);
    let mut y_asm = vec![0.0; n];
    a.spmv(&x, &mut y_asm);
    let num = y_pa.iter().zip(y_asm.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    let den = y_asm.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    (num / den.max(1e-300), shared)
}

/// The assembled quadrature order whose point count equals the kernel's own
/// rule (`npoints = ceil((order+1)/2)`): the Pk kernels use `p+1` points per
/// direction, the fixed-order ones 2/2/3/4/5.
fn order_for_points(points: usize) -> u8 {
    (2 * points - 2) as u8
}

type HexBuild = &'static dyn Fn(&Mesh<3>, u8) -> PaData;
type HexApply = &'static dyn Fn(&PaData, &[Vec<u32>], u8, &[f64], &mut [f64]);
type QuadBuild = &'static dyn Fn(&Mesh<2>, u8) -> PaData;
type QuadApply = &'static dyn Fn(&PaData, &[Vec<u32>], u8, &[f64], &mut [f64]);

/// The 3-D family: `hex_q1`, `hex_qk` at every order, `hex_q2/3/4`.
///
/// Teeth: the `straightened` column must be far off — that column *is* the
/// pre-fix PA data source.  The curved column must be round-off.  Every kernel
/// is measured before any assertion fires, so one run shows the whole family.
#[test]
fn d808_4_hex_family_curved_pa_matches_assembly() {
    let curved = curved_hex_mesh();
    let straight = straightened_hex_mesh();
    assert_eq!(curved.geom_order(), 2, "the fixture must be curved");
    assert_eq!(straight.geom_order(), 1, "the witness must be straight");
    assert_no_fold(&curved, "hex family fixture");

    // (name, kernel order p, PA points per direction, build, apply)
    let kernels: Vec<(&str, u8, usize, HexBuild, HexApply)> = vec![
        ("hex_q1", 1, 2, &|m, _p| build_hex_q1_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_hex_q1(pd, dofs, x, y)),
        ("hex_q2", 2, 3, &|m, _p| build_hex_q2_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_hex_q2(pd, dofs, x, y)),
        ("hex_q3", 3, 4, &|m, _p| build_hex_q3_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_hex_q3(pd, dofs, x, y)),
        ("hex_q4", 4, 5, &|m, _p| build_hex_q4_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_hex_q4(pd, dofs, x, y)),
        ("hex_qk p=1", 1, 2, &|m, p| build_hex_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_hex_qk(pd, dofs, p as usize, x, y)),
        ("hex_qk p=2", 2, 3, &|m, p| build_hex_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_hex_qk(pd, dofs, p as usize, x, y)),
        ("hex_qk p=3", 3, 4, &|m, p| build_hex_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_hex_qk(pd, dofs, p as usize, x, y)),
        ("hex_qk p=4", 4, 5, &|m, p| build_hex_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_hex_qk(pd, dofs, p as usize, x, y)),
    ];

    let mut bad = Vec::new();
    for (name, p, npts, build, apply) in kernels {
        let order = order_for_points(npts);
        let (rel, shared) = pa_vs_assembled(&curved, &curved, order, p, build, apply);
        let (red, _) = pa_vs_assembled(&straight, &curved, order, p, build, apply);
        println!(
            "{name}: shared dofs(e0,e1)={shared}  curved rel={rel:.3e}  \
             pre-fix (vertex geoms) rel={red:.3e}"
        );
        if shared == 0 {
            bad.push(format!("{name}: fixture has no DOFs shared by elements 0/1"));
        } else if red <= 1e-3 {
            bad.push(format!(
                "{name}: the pre-fix vertex-geometry PA differs by only {red:.3e} — pin has no teeth"
            ));
        } else if rel >= 1e-11 {
            bad.push(format!(
                "{name}: curved PA vs assembled = {rel:.3e} (pre-fix would be {red:.3e})"
            ));
        }
    }
    assert!(bad.is_empty(), "D808-4 hex family:\n  {}", bad.join("\n  "));
}

/// The 2-D family: `quad_q1`, `quad_qk` at every order, `quad_q2`.
#[test]
fn d808_4_quad_family_curved_pa_matches_assembly() {
    let curved = curved_quad_mesh();
    let straight = straightened_quad_mesh();
    assert_eq!(curved.geom_order(), 2, "the fixture must be curved");
    assert_eq!(straight.geom_order(), 1, "the witness must be straight");
    assert_eq!(curved.element_type(0), ElementType::Quad4);
    assert_no_fold(&curved, "quad family fixture");

    let kernels: Vec<(&str, u8, usize, QuadBuild, QuadApply)> = vec![
        ("quad_q1", 1, 2, &|m, _p| build_quad_q1_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_quad_q1(pd, dofs, x, y)),
        ("quad_q2", 2, 3, &|m, _p| build_quad_q2_pa_data(m, &|_| 1.0),
            &|pd, dofs, _p, x, y| pa_apply_quad_q2(pd, dofs, x, y)),
        ("quad_qk p=1", 1, 2, &|m, p| build_quad_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_quad_qk(pd, dofs, p as usize, x, y)),
        ("quad_qk p=2", 2, 3, &|m, p| build_quad_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_quad_qk(pd, dofs, p as usize, x, y)),
        ("quad_qk p=3", 3, 4, &|m, p| build_quad_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_quad_qk(pd, dofs, p as usize, x, y)),
        ("quad_qk p=4", 4, 5, &|m, p| build_quad_qk_pa_data(m, &|_| 1.0, p as usize),
            &|pd, dofs, p, x, y| pa_apply_quad_qk(pd, dofs, p as usize, x, y)),
    ];

    let mut bad = Vec::new();
    for (name, p, npts, build, apply) in kernels {
        let order = order_for_points(npts);
        let (rel, shared) = pa_vs_assembled(&curved, &curved, order, p, build, apply);
        let (red, _) = pa_vs_assembled(&straight, &curved, order, p, build, apply);
        println!(
            "{name}: shared dofs(e0,e1)={shared}  curved rel={rel:.3e}  \
             pre-fix (vertex geoms) rel={red:.3e}"
        );
        if shared == 0 {
            bad.push(format!("{name}: fixture has no DOFs shared by elements 0/1"));
        } else if red <= 1e-3 {
            bad.push(format!(
                "{name}: the pre-fix vertex-geometry PA differs by only {red:.3e} — pin has no teeth"
            ));
        } else if rel >= 1e-11 {
            bad.push(format!(
                "{name}: curved PA vs assembled = {rel:.3e} (pre-fix would be {red:.3e})"
            ));
        }
    }
    assert!(bad.is_empty(), "D808-4 quad family:\n  {}", bad.join("\n  "));
}

/// The straight path is untouched: on a `geom_order == 1` mesh every kernel of
/// the family still reproduces the assembled operator, and PA-data comes from
/// the verbatim pre-fix branch (the `geom_order() >= 2` gate in
/// `pa::curved::curved_jacobian`, pinned by its own unit test).
#[test]
fn d808_4_straight_meshes_are_untouched() {
    let hex = Mesh::<3>::unit_cube_hex(2);
    let quad = Mesh::<2>::unit_square_quad(2);
    assert_eq!(hex.geom_order(), 1);
    assert_eq!(quad.geom_order(), 1);

    let hex_kernels: [(&str, u8, usize, &dyn Fn(&Mesh<3>, u8) -> PaData,
                       &dyn Fn(&PaData, &[Vec<u32>], u8, &[f64], &mut [f64])); 4] = [
        ("hex_q1", 1, 2, &|m, _p| build_hex_q1_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_hex_q1(pd, dofs, x, y)),
        ("hex_q2", 2, 3, &|m, _p| build_hex_q2_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_hex_q2(pd, dofs, x, y)),
        ("hex_q3", 3, 4, &|m, _p| build_hex_q3_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_hex_q3(pd, dofs, x, y)),
        ("hex_q4", 4, 5, &|m, _p| build_hex_q4_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_hex_q4(pd, dofs, x, y)),
    ];
    for (name, p, npts, build, apply) in hex_kernels {
        let (rel, _) = pa_vs_assembled(&hex, &hex, order_for_points(npts), p, build, apply);
        assert!(rel < 1e-11, "{name} straight: PA vs assembled = {rel:.3e}");
    }

    let quad_kernels: [(&str, u8, usize, &dyn Fn(&Mesh<2>, u8) -> PaData,
                       &dyn Fn(&PaData, &[Vec<u32>], u8, &[f64], &mut [f64])); 2] = [
        ("quad_q1", 1, 2, &|m, _p| build_quad_q1_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_quad_q1(pd, dofs, x, y)),
        ("quad_q2", 2, 3, &|m, _p| build_quad_q2_pa_data(m, &|_| 1.0),
                   &|pd, dofs, _p, x, y| pa_apply_quad_q2(pd, dofs, x, y)),
    ];
    for (name, p, npts, build, apply) in quad_kernels {
        let (rel, _) = pa_vs_assembled(&quad, &quad, order_for_points(npts), p, build, apply);
        assert!(rel < 1e-11, "{name} straight: PA vs assembled = {rel:.3e}");
    }
}

/// D783's prism kernel keeps working on a curved mesh through the shared
/// helper the D808-4 fix routed it through (`pa::curved::curved_jacobian`).
#[test]
fn d808_4_prism_kernel_still_correct_after_the_refactor() {
    // The prism fixture and its multi-element pins live in
    // `d808_prism_pa_multi.rs`; this only asserts the refactor did not change
    // the shared helper's contract for the family that already used it.
    let hex = curved_hex_mesh();
    assert_eq!(hex.geom_order(), 2);
    // A curved hex's geometry node count must exceed its vertex count, or the
    // fixture would not exercise the table at all.
    let g = hex.geometry.as_ref().expect("geometry table");
    assert!(
        g.n_nodes > hex.n_nodes(),
        "curved fixture: geometry nodes {} vs vertices {}",
        g.n_nodes,
        hex.n_nodes()
    );
}
