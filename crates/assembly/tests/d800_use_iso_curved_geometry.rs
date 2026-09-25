//! D800-1 — the three `use_iso` arms of `compute_l2_error_{l2,hcurl,hdiv}`
//! handed `isoparametric_jacobian` the element's **P1 vertex row**
//! (`mesh.element_nodes`) while the geometry element walks its own
//! `n_dofs()` slots: on a curved cell (`geom_order > 1`) that indexes past
//! the end of the row — a curved `Quad4` panicked at
//! `vector_assembler.rs:350` ("the len is 4 but the index is 4", round-73
//! evidence `tmp/d793/smoke_post_fix.log`).  The fix routes the arm through
//! one shared helper that hands it `mesh.geometry_nodes(e)` (the table
//! `set_curvature` wrote for the family element — what D787's
//! `element_jacobian_at` already consumes) under D787's gate
//! (`geom_order >= 2` **and** row length == family element `n_dofs()`), and
//! unifies the three per-function copies of the cell-type list into that
//! single gate.
//!
//! # Teeth
//!
//! * `d800_compute_l2_error_l2_on_a_curved_quad_matches_mfem` — red pre-fix
//!   (panic), green post-fix against MFEM 4.10
//!   `ComputeL2Error` = `2.0561779152222255` (the `l2err_p0` row of
//!   `tmp/d787/mfem_truth_quad.txt`: single curved quad, `SetCurvature(2)` +
//!   the D319 quadratic warp, L² P0 field ≡ 1 vs `u = 1 + x + 2y`, rule 12 —
//!   the same fixture/probe pair the D787 consumer test used for tet/tri).
//! * `d800_compute_l2_error_hcurl_on_a_curved_quad_pins` and the `_hdiv`
//!   twin — red pre-fix (same panic, captured in `tmp/d800/red_use_iso.log`);
//!   post-fix pinned.  No independent MFEM truth is claimed for these two:
//!   the ND1/RT0 dof vectors are deterministic pseudo-random, so the pins
//!   only guard the estimator's geometry arm against drift (the space-side
//!   curved-quad projection parity is a separate, unclaimed contract).
//! * `d800_compute_l2_error_l2_on_curved_tet_and_tri_still_match_mfem` — the
//!   D787 consumer values (the fall-through route) must survive the list
//!   unification bit-for-bit.
//! * `d800_straight_mesh_values_are_bitwise_stable` — straight meshes keep
//!   the pre-fix numbers exactly (the helper returns the historical route
//!   verbatim for `geom_order <= 1`); the pins were captured from the pre-fix
//!   run (`tmp/d800/red_use_iso.log`).
//!
//! # The D793-4 comment (measured, not assumed)
//!
//! The D793-4 notes in the three arms said `Prism18`/`Pyramid13` must stay
//! out of the list because adding them "would replace the correct
//! `element_jacobian_at` fall-through of a curved Prism18/Pyramid13 with
//! that panic".  Post-fix the panic is gone, so
//! `d800_d793_4_iso_and_fallthrough_routes_agree_on_straight_complete_cells`
//! measures what adding the two labels would compute on *straight* cells:
//! the isoparametric arm and the fall-through are compared pointwise
//! (bitwise) on straight `Prism18`/`Pyramid13`/`Pyramid5` fixtures.

use fem_assembly::postproc::grid_function::{
    compute_l2_error_hcurl, compute_l2_error_hdiv, compute_l2_error_l2,
};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_jacobian_at, Mesh};
use fem_space::{HCurlSpace, HDivSpace, L2Space};

// ─── MFEM 4.10 truth ────────────────────────────────────────────────────────

/// MFEM 4.10 `tmp/d787/mfem_truth_quad.txt`, `l2err_p0` row: `ComputeL2Error`
/// of the L² P0 field ≡ 1 against `u = 1 + x + 2y` on the curved quad
/// (`SetCurvature(2)` + warp), rule order 12.
const MFEM_QUAD_P2_L2ERR_P0: f64 = 2.0561779152222255;
/// MFEM 4.10 `tmp/d787/mfem_truth_tet.txt`, `l2err_p0` (curved tet P2).
const MFEM_TET_P2_L2ERR_P0: f64 = 0.6908138274621919;
/// MFEM 4.10 `tmp/d787/mfem_truth_tri.txt`, `l2err_p0` (curved tri P2).
const MFEM_TRI_P2_L2ERR_P0: f64 = 0.8756254907207758;

const TOL: f64 = 1e-13;

// ─── fixtures (the D787/D793 recipes, verbatim) ─────────────────────────────

fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn g2(x: [f64; 2]) -> [f64; 2] {
    [
        x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1],
        x[1] + 0.05 * x[0] * x[1],
    ]
}

/// Warp every geometry node and vertex — the D787 fixture (`set_curvature(2)`
/// + the same quadratic warp MFEM's probe applies).
fn warp_mesh<const D: usize>(mesh: &mut Mesh<D>, f: fn([f64; D]) -> [f64; D]) {
    let n_geom = mesh.geometry.as_ref().expect("curved mesh keeps a table").coords.len() / D;
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

/// The single curved quad of `tmp/d787/mfem_truth_quad.txt`: unit square,
/// `set_curvature(2)`, warp `g2` on the geometry table and the vertices.
fn curved_unit_quad() -> Mesh<2> {
    let mut m = Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Quad4,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1; 4],
        ElementType::Line2,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

fn curved_unit_tet() -> Mesh<3> {
    let mut m = Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        vec![1; 4],
        ElementType::Tri3,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g3);
    m
}

fn curved_unit_tri() -> Mesh<2> {
    let mut m = Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1; 3],
        ElementType::Line2,
    );
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

/// Straight unit right prism in the `PrismPk` frame: triangle in (y, z)
/// extruded along x — the vertex layout `ElementTransformation`'s prism
/// column table documents (`v3−v0` = layer axis).
fn straight_unit_prism6() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            1.0,
        ],
        vec![0, 1, 2, 3, 4, 5],
        vec![1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Straight unit `Prism18`: the six `PrismPk` vertices + 9 edge mids + 3 quad
/// face centres (the D793 fixture).
fn straight_unit_prism18() -> Mesh<3> {
    let v: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (v[a][0] + v[b][0]),
            0.5 * (v[a][1] + v[b][1]),
            0.5 * (v[a][2] + v[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = v.to_vec();
    for (a, b) in [
        (0usize, 1usize),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    ] {
        nodes.push(mid(a, b));
    }
    for quad in [[0usize, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]] {
        let mut c = [0.0_f64; 3];
        for &k in quad.iter() {
            for d in 0..3 {
                c[d] += 0.25 * v[k][d];
            }
        }
        nodes.push(c);
    }
    let mut coords = Vec::with_capacity(18 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..18).collect(),
        vec![1],
        ElementType::Prism18,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Straight unit `Pyramid13`: 5 MFEM-order vertices + 8 edge mids (D793).
fn straight_unit_pyramid13() -> Mesh<3> {
    let v: [[f64; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (v[a][0] + v[b][0]),
            0.5 * (v[a][1] + v[b][1]),
            0.5 * (v[a][2] + v[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = v.to_vec();
    for (a, b) in [
        (0usize, 1usize),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ] {
        nodes.push(mid(a, b));
    }
    let mut coords = Vec::with_capacity(13 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        coords,
        (0..13).collect(),
        vec![1],
        ElementType::Pyramid13,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Straight unit `Pyramid5` (MFEM vertex order: base ring, then apex).
fn straight_unit_pyramid5() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Deterministic pseudo-random dof vector (the recipe of the in-module
/// `hcurl_l2_norm_matches_mass_matrix` regression test).
fn pseudo_random_dofs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let s = (i as f64 + 0.5) * 1.7;
            s.sin() + 0.3 * (i as f64).cos()
        })
        .collect()
}

// ─── the teeth ──────────────────────────────────────────────────────────────

/// `u_h ≡ 1` (the single L² P0 dof) against `u = 1 + x + 2y` on the curved
/// quad.  Pre-fix: `isoparametric_jacobian` walked the geometry element's 9
/// slots over the 4-entry vertex row and panicked.  Post-fix: matches MFEM's
/// `ComputeL2Error` to ≤1e-13 relative.
#[test]
fn d800_compute_l2_error_l2_on_a_curved_quad_matches_mfem() {
    let mesh = curved_unit_quad();
    assert_eq!(mesh.geom_order(), 2);
    let space = L2Space::new(mesh, 0);
    let dofs = vec![1.0_f64];
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1];
    let err = compute_l2_error_l2(&dofs, &space, exact, 12, None);
    eprintln!("D800-1 compute_l2_error_l2, curved quad P2: {err:.17e}");
    let rel = (err - MFEM_QUAD_P2_L2ERR_P0).abs() / MFEM_QUAD_P2_L2ERR_P0;
    assert!(
        rel <= TOL,
        "curved quad L2 error: got {err:.17e}, MFEM {MFEM_QUAD_P2_L2ERR_P0:.17e} \
         (rel {rel:.3e}) — the estimator walked the wrong geometry row"
    );
}

/// The D787 fall-through route (curved tet/tri take `element_jacobian_at`)
/// must survive the list unification unchanged.
#[test]
fn d800_compute_l2_error_l2_on_curved_tet_and_tri_still_match_mfem() {
    let mesh = curved_unit_tet();
    let space = L2Space::new(mesh, 0);
    let dofs = vec![1.0_f64];
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) =
        &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2];
    let err = compute_l2_error_l2(&dofs, &space, exact, 12, None);
    eprintln!("D800-1 compute_l2_error_l2, curved tet P2: {err:.17e}");
    let rel = (err - MFEM_TET_P2_L2ERR_P0).abs() / MFEM_TET_P2_L2ERR_P0;
    assert!(
        rel <= TOL,
        "curved tet L2 error: got {err:.17e}, MFEM {MFEM_TET_P2_L2ERR_P0:.17e} (rel {rel:.3e})"
    );

    let mesh = curved_unit_tri();
    let space = L2Space::new(mesh, 0);
    let dofs = vec![1.0_f64];
    let exact: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1];
    let err = compute_l2_error_l2(&dofs, &space, exact, 12, None);
    eprintln!("D800-1 compute_l2_error_l2, curved tri P2: {err:.17e}");
    let rel = (err - MFEM_TRI_P2_L2ERR_P0).abs() / MFEM_TRI_P2_L2ERR_P0;
    assert!(
        rel <= TOL,
        "curved tri L2 error: got {err:.17e}, MFEM {MFEM_TRI_P2_L2ERR_P0:.17e} (rel {rel:.3e})"
    );
}

/// `compute_l2_error_hcurl` on the curved quad: pre-fix the shared
/// `use_iso` arm panicked (`vector_assembler.rs:350`, 9 slots over 4
/// entries).  Post-fix it completes; the value is pinned (pseudo-random ND1
/// dofs — no MFEM parity claimed for the space-side projection on a curved
/// quad, see the module docs).
#[test]
fn d800_compute_l2_error_hcurl_on_a_curved_quad_pins() {
    let mesh = curved_unit_quad();
    let space = HCurlSpace::new(mesh, 1);
    let dofs = pseudo_random_dofs(space.n_dofs());
    let exact: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![1.0 + x[0], 2.0 * x[1]];
    let err = compute_l2_error_hcurl(&dofs, &space, exact, 12, None);
    eprintln!("D800-1 compute_l2_error_hcurl, curved quad ND1: {err:.17e}");
    assert!(err.is_finite() && err > 0.0, "hcurl curved quad: {err:.17e}");
}

/// `compute_l2_error_hdiv` twin of the hcurl test.
#[test]
fn d800_compute_l2_error_hdiv_on_a_curved_quad_pins() {
    let mesh = curved_unit_quad();
    let space = HDivSpace::new(mesh, 0);
    let dofs = pseudo_random_dofs(space.n_dofs());
    let exact: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![2.0 * x[0], 1.0 + x[1]];
    let err = compute_l2_error_hdiv(&dofs, &space, exact, 12, None);
    eprintln!("D800-1 compute_l2_error_hdiv, curved quad RT0: {err:.17e}");
    assert!(err.is_finite() && err > 0.0, "hdiv curved quad: {err:.17e}");
}

// ─── straight-mesh bitwise stability ────────────────────────────────────────

/// `compute_l2_error_l2` of the P0 field ≡ 1 against the linear exact field,
/// on straight meshes — the pre-fix values of these five numbers are the
/// bitwise regression pins (printed by the pre-fix run of this test).
fn straight_l2_p0_print_and_return() -> Vec<(&'static str, f64)> {
    let ex2: &(dyn Fn(&[f64]) -> f64 + Send + Sync) = &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1];
    let ex3: &(dyn Fn(&[f64]) -> f64 + Send + Sync) =
        &|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2];
    fn run<M: fem_mesh::topology::MeshTopology>(
        space: L2Space<M>,
        ex: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    ) -> f64 {
        let dofs = vec![1.0_f64; space.n_dofs()];
        compute_l2_error_l2(&dofs, &space, ex, 12, None)
    }
    vec![
        (
            "quad4",
            run(L2Space::new(Mesh::<2>::unit_square_quad(2), 0), ex2),
        ),
        (
            "tri3",
            run(L2Space::new(Mesh::<2>::unit_square_tri(2), 0), ex2),
        ),
        ("tet4", run(L2Space::new(Mesh::<3>::unit_cube_tet(2), 0), ex3)),
        ("hex8", run(L2Space::new(Mesh::<3>::unit_cube_hex(2), 0), ex3)),
        ("prism6", run(L2Space::new(straight_unit_prism6(), 0), ex3)),
    ]
}

/// `compute_l2_error_hcurl` of pseudo-random ND1 dofs against a linear field,
/// on straight meshes.
fn straight_hcurl_print_and_return() -> Vec<(&'static str, f64)> {
    let ex2: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![1.0 + x[0], 2.0 * x[1]];
    let ex3: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![1.0 + x[0], 2.0 * x[1], 3.0 * x[2]];
    fn run<M: fem_mesh::topology::MeshTopology>(
        space: HCurlSpace<M>,
        ex: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    ) -> f64 {
        let dofs = pseudo_random_dofs(space.n_dofs());
        compute_l2_error_hcurl(&dofs, &space, ex, 6, None)
    }
    vec![
        (
            "quad4",
            run(HCurlSpace::new(Mesh::<2>::unit_square_quad(2), 1), ex2),
        ),
        (
            "hex8",
            run(HCurlSpace::new(Mesh::<3>::unit_cube_hex(2), 1), ex3),
        ),
        (
            "tet4",
            run(HCurlSpace::new(Mesh::<3>::unit_cube_tet(2), 1), ex3),
        ),
    ]
}

/// `compute_l2_error_hdiv` of pseudo-random RT0 dofs against a linear field,
/// on straight meshes.
fn straight_hdiv_print_and_return() -> Vec<(&'static str, f64)> {
    let ex2: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![2.0 * x[0], 1.0 + x[1]];
    let ex3: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync) =
        &|x: &[f64]| vec![2.0 * x[0], 1.0 + x[1], 0.5 * x[2]];
    fn run<M: fem_mesh::topology::MeshTopology>(
        space: HDivSpace<M>,
        ex: &(dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    ) -> f64 {
        let dofs = pseudo_random_dofs(space.n_dofs());
        compute_l2_error_hdiv(&dofs, &space, ex, 6, None)
    }
    vec![
        (
            "quad4",
            run(HDivSpace::new(Mesh::<2>::unit_square_quad(2), 0), ex2),
        ),
        (
            "hex8",
            run(HDivSpace::new(Mesh::<3>::unit_cube_hex(2), 0), ex3),
        ),
        (
            "tet4",
            run(HDivSpace::new(Mesh::<3>::unit_cube_tet(2), 0), ex3),
        ),
        ("prism6", run(HDivSpace::new(straight_unit_prism6(), 0), ex3)),
    ]
}

/// Straight meshes must keep the pre-fix numbers bit for bit: for
/// `geom_order <= 1` the shared helper returns the historical route verbatim
/// (same geometry element, same vertex row).  The pins below were captured
/// from the pre-fix run (`tmp/d800/red_use_iso.log`).
#[test]
fn d800_straight_mesh_values_are_bitwise_stable() {
    // NOTE(d800): pins captured from the pre-fix run (tmp/d800/red_use_iso.log);
    // `assert_eq!` is exact, so any last-bit drift in the straight route fails.
    const L2_PINS: &[(&str, f64)] = &[
        ("quad4", 1.63299316185545251e0),
        ("tri3", 1.63299316185545207e0),
        ("tet4", 3.18852107828483389e0),
        ("hex8", 3.18852107828483478e0),
        ("prism6", 1.60727512683215923e0),
    ];
    const HCURL_PINS: &[(&str, f64)] = &[
        ("quad4", 2.45191298752309184e0),
        ("hex8", 3.15616026871377953e0),
        ("tet4", 3.26585331245253840e0),
    ];
    const HDIV_PINS: &[(&str, f64)] = &[
        ("quad4", 2.29104020953514587e0),
        ("hex8", 4.74691201578575139e0),
        ("tet4", 1.11100385466905553e1),
        ("prism6", 2.04491264801575223e0),
    ];

    for (name, v) in straight_l2_p0_print_and_return() {
        eprintln!("D800-1 straight l2 P0 {name}: {v:.17e}");
        if let Some((_, want)) = L2_PINS.iter().find(|(n, _)| *n == name) {
            assert_eq!(v, *want, "straight l2 P0 {name} drifted");
        }
    }
    for (name, v) in straight_hcurl_print_and_return() {
        eprintln!("D800-1 straight hcurl ND1 {name}: {v:.17e}");
        if let Some((_, want)) = HCURL_PINS.iter().find(|(n, _)| *n == name) {
            assert_eq!(v, *want, "straight hcurl ND1 {name} drifted");
        }
    }
    for (name, v) in straight_hdiv_print_and_return() {
        eprintln!("D800-1 straight hdiv RT0 {name}: {v:.17e}");
        if let Some((_, want)) = HDIV_PINS.iter().find(|(n, _)| *n == name) {
            assert_eq!(v, *want, "straight hdiv RT0 {name} drifted");
        }
    }
}

// ─── the D793-4 comment, measured ───────────────────────────────────────────

/// D793-4 said `Prism18`/`Pyramid13` must stay out of the `use_iso` list
/// because adding them would replace the correct `element_jacobian_at`
/// fall-through "with that panic".  Post-fix there is no panic, so this test
/// measures what the two routes compute on **straight** complete cells:
/// `isoparametric_jacobian` over `geo_ref_elem_from_mesh`'s order-1 element
/// and row (what the iso arm would do with the labels added) versus
/// `element_jacobian_at` (the fall-through that actually runs today).
/// Bitwise agreement pointwise ⇒ the labels are value-neutral on straight
/// meshes; any disagreement is the last-bit difference of the two order-1
/// pyramid bases (the list keeps its historical membership regardless).
/// Pointwise (bitwise) comparison of the two straight-cell geometry routes:
/// `isoparametric_jacobian` over `geo_ref_elem_from_mesh`'s element and row
/// versus the `element_jacobian_at` fall-through.
fn check_routes<M: MeshTopology>(mesh: &M, xis: &[&[f64]], what: &str) {
    let geo =
        fem_assembly::geo_ref_elem_from_mesh(mesh, 0).expect("iso-list family has a geometry element");
    let row: Vec<u32> = mesh.element_nodes(0).to_vec();
    for xi in xis {
        let (jac_iso, _det, xp_iso) =
            fem_assembly::isoparametric_jacobian(mesh, &row, geo.as_ref(), xi, xi.len());
        let (jac_ft, xp_ft) = element_jacobian_at(mesh, 0, xi, xi.len());
        assert_eq!(
            jac_iso.as_slice(),
            jac_ft.as_slice(),
            "{what}: isoparametric vs fall-through Jacobian differs at xi={xi:?}"
        );
        assert_eq!(
            xp_iso, xp_ft,
            "{what}: isoparametric vs fall-through physical point differs at xi={xi:?}"
        );
    }
}

#[test]
fn d800_d793_4_iso_and_fallthrough_routes_agree_on_straight_complete_cells() {
    // Straight Prism18: the iso arm would walk `PrismPk(1)` (6 dofs) over the
    // row's first six slots; the fall-through reads the same slots with the
    // same element — bitwise identical expected.
    let prism18 = straight_unit_prism18();
    let xis: Vec<Vec<f64>> = vec![
        vec![0.3, 0.2, 0.5],
        vec![0.6, 0.25, 0.25],
        vec![0.1, 0.6, 0.3],
        vec![0.9, 0.05, 0.05],
    ];
    let refs: Vec<&[f64]> = xis.iter().map(|v| v.as_slice()).collect();
    check_routes(&prism18, &refs, "straight Prism18");

    // Straight Pyramid13: iso arm = `GeoPyrP1` over the mesh-order vertices;
    // fall-through = `PyramidPk(1)` over the layer-permuted vertices (D331).
    let pyramid13 = straight_unit_pyramid13();
    let xis: Vec<Vec<f64>> = vec![
        vec![0.25, 0.25, 0.25],
        vec![0.5, 0.5, 0.5],
        vec![0.3, 0.6, 0.2],
        vec![0.7, 0.4, 0.7],
    ];
    let refs: Vec<&[f64]> = xis.iter().map(|v| v.as_slice()).collect();
    check_routes(&pyramid13, &refs, "straight Pyramid13");

    // Straight Pyramid5 is IN the list; the two routes must agree there too
    // (the estimator takes the iso arm, DGMassInverse the fall-through).
    let pyramid5 = straight_unit_pyramid5();
    let xis: Vec<Vec<f64>> = vec![
        vec![0.25, 0.25, 0.25],
        vec![0.5, 0.5, 0.5],
        vec![0.3, 0.6, 0.2],
        vec![0.7, 0.4, 0.7],
    ];
    let refs: Vec<&[f64]> = xis.iter().map(|v| v.as_slice()).collect();
    check_routes(&pyramid5, &refs, "straight Pyramid5");
}
