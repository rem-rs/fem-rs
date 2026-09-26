//! D813-3 — `Mesh::get_bounding_box(ref)`: MFEM's `Mesh::GetBoundingBox(min,
//! max, ref)` for a mesh that carries a geometry table (`Nodes != NULL`).
//!
//! MFEM's call has two arms (`mesh/mesh.cpp:142`):
//!
//! ```text
//! if (Nodes == NULL)        // vertex extrema  == Mesh::bounding_box()
//!    for i in 0..NV { minmax(GetVertex(i)) }
//! else                      // geometry extrema
//!    for i in 0..NE {
//!       RefG = GlobGeometryRefiner.Refine(GetElementBaseGeometry(i), ref);
//!       T = GetElementTransformation(i);
//!       T->Transform(RefG->RefPts, pointmat);
//!       minmax(pointmat) }
//! ```
//!
//! fem-rs had only the first arm, so every consumer of the second one silently
//! got a *different box*.  `examples/mfem_ex9_dg_advection.rs` is the loud case:
//! `ex9.cpp:241` calls `mesh.GetBoundingBox(bb_min, bb_max, max(order, 1))` on
//! the **refined** default mesh `data/periodic-hexagon.mesh`, whose geometry
//! table is a folded `L2_T1_2D_P1` field.  The geometry reaches
//! `[-1, 1] x [-sqrt(3)/2, sqrt(3)/2]` while the mesh-level vertex table (the
//! *means* of the folded copies, D813-2) spans only
//! `[-0.5, 0.75] x [-0.866, 0.433]`, and ex9 normalises both its initial
//! condition and its velocity field by that box — so the whole solution moved.
//!
//! Three oracles, all MFEM 4.10 (`tmp/d78b/probe/*.cpp`):
//!
//! 1. `d813_mfem_geometry_refiner.txt` — `GlobGeometryRefiner.Refine`'s `RefPts`
//!    for SEGMENT/TRIANGLE/SQUARE/TETRAHEDRON/CUBE/PRISM/PYRAMID at
//!    `times = 1..6`, element-by-element in order (so the lattice's *order* is
//!    pinned too, not only its point set);
//! 2. `d813_mfem_bbox_curved.txt` + `d813_curved_*.mesh.txt` — a curved mesh per
//!    family (`SetCurvature(p)` + a corner-preserving `sin(pi x)` bump, saved by
//!    MFEM at precision 16) with MFEM's own box for `ref = 1..5`; the box is
//!    ref-dependent there, so the sampling *level* is exercised, and every
//!    family's isoparametric map is exercised through `element_jacobian_at`;
//! 3. `d813_mfem_bbox.txt` — the folded periodic meshes (`data/periodic-*`),
//!    the ex9 case, `refine in {0, 2}` x `ref = 1..5`.

use fem_io::mfem::read_mfem_file;
use fem_mesh::{
    mfem_geometry_refiner_points, refine_uniform, refine_uniform_3d, ElementType, Mesh,
};
use std::path::{Path, PathBuf};

/// Relative-to-1 tolerance.  The boxes below agree to ~1e-16; the sample
/// *points* can differ by an ulp (MFEM's `cp[i]/w` normalisation for simplicial
/// families), so the comparison is not bit-exact by construction.
const TOL: f64 = 1e-14;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn test_data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

fn read2(path: &Path) -> Mesh<2> {
    read_mfem_file(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .mesh2d
        .expect("2-D mesh")
}

fn read3(path: &Path) -> Mesh<3> {
    read_mfem_file(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .mesh3d
        .expect("3-D mesh")
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

/// `(min…, max…)` of a 2-D mesh's curved box, flattened for the fixture rows.
fn box2(m: &Mesh<2>, ref_level: i32) -> Vec<f64> {
    let (lo, hi) = m.get_bounding_box(ref_level);
    lo.iter().chain(hi.iter()).copied().collect()
}

fn box3(m: &Mesh<3>, ref_level: i32) -> Vec<f64> {
    let (lo, hi) = m.get_bounding_box(ref_level);
    lo.iter().chain(hi.iter()).copied().collect()
}

// ─── 1. the sampler itself, against MFEM's RefPts ───────────────────────────

/// `GlobGeometryRefiner.Refine(geom, times)`'s reference points, verbatim.
///
/// MFEM's global refiner is built with the default `Quadrature1D::ClosedUniform`
/// type (`fem/geom.hpp:369`), so the lattice is the closed-uniform tensor /
/// barycentric lattice of `times + 1` points (`cp[i] = i/times`), and `times` is
/// clamped to `>= 1`.  The fixture is element-by-element, in MFEM's own
/// enumeration order.
///
/// **One axis permutation**, named and local to the comparison: MFEM's PRISM
/// reference domain is `(triangle_x, triangle_y, layer)`, while fem-rs's
/// `PrismPk` puts the extrusion first — `(layer, triangle_eta, triangle_zeta)`
/// (D152/D164).  `prism_axis_swap` below applies that fixed permutation to the
/// *fixture* row so the two coordinate triples can be compared component-wise;
/// `mfem_geometry_refiner_points` returns points in the fem-rs element's own
/// domain, which is what `element_jacobian_at` consumes.
#[test]
fn d813_geometry_refiner_points_match_mfem() {
    let src = text(&test_data("d813_mfem_geometry_refiner.txt"));
    let fam = |name: &str| -> ElementType {
        match name {
            "SEGMENT" => ElementType::Line2,
            "TRIANGLE" => ElementType::Tri3,
            "SQUARE" => ElementType::Quad4,
            "TETRAHEDRON" => ElementType::Tet4,
            "CUBE" => ElementType::Hex8,
            "PRISM" => ElementType::Prism6,
            "PYRAMID" => ElementType::Pyramid5,
            other => panic!("unmapped oracle family {other}"),
        }
    };
    // MFEM `(x, y, z)` → fem-rs `(ξ0, ξ1, ξ2)`: identity except for the prism.
    let prism_axis_swap = |name: &str, v: &[f64]| -> [f64; 3] {
        if name == "PRISM" {
            [v[2], v[0], v[1]]
        } else {
            [v[0], v[1], v[2]]
        }
    };
    let mut lines = src.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let mut tables = 0;
    while let Some(header) = lines.next() {
        let h: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(h.len(), 3, "bad header {header:?}");
        let et = fam(h[0]);
        let times: usize = h[1].parse().expect("times");
        let npts: usize = h[2].parse().expect("npts");
        let want: Vec<Vec<f64>> = (0..npts)
            .map(|_| {
                let row: Vec<f64> = lines
                    .next()
                    .expect("point row")
                    .split_whitespace()
                    .map(|s| s.parse::<f64>().expect("coord"))
                    .collect();
                prism_axis_swap(h[0], &row).to_vec()
            })
            .collect();
        let got = mfem_geometry_refiner_points(et, times)
            .unwrap_or_else(|| panic!("no sampler for {et:?}"));
        assert_eq!(got.len(), npts, "{h:?}: point count");
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            for d in 0..3 {
                assert!(
                    (g[d] - w[d]).abs() <= 1e-15,
                    "{h:?}: point {k} component {d}: {g:?} vs MFEM {w:?}"
                );
            }
        }
        tables += 1;
    }
    assert_eq!(tables, 42, "7 families x times 1..6");
}

// ─── 2. the box over curved geometry, all families ─────────────────────────

#[test]
fn d813_bounding_box_matches_mfem_over_curved_geometry() {
    let src = text(&test_data("d813_mfem_bbox_curved.txt"));
    let mut rows = 0;
    for line in src.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let family = t[0];
        let ref_level: i32 = t[1].parse().expect("ref");
        let sdim: usize = t[2].parse().expect("sdim");
        let want: Vec<f64> = t[3..].iter().map(|s| s.parse::<f64>().expect("v")).collect();
        assert_eq!(want.len(), 2 * sdim);
        let mesh_path = test_data(&format!("d813_curved_{family}.mesh.txt"));
        let got = if sdim == 2 {
            box2(&read2(&mesh_path), ref_level)
        } else {
            box3(&read3(&mesh_path), ref_level)
        };
        let d = max_abs_diff(&got, &want);
        assert!(
            d <= TOL,
            "{family} ref={ref_level}: |box - MFEM| = {d:e}\n  fem-rs {got:?}\n  MFEM   {want:?}"
        );
        rows += 1;
    }
    assert_eq!(rows, 35, "7 curved meshes x ref 1..5");
}

/// The fixture's `ref = 1` row is the *element-corner* box: with the bump
/// `sin(pi x)` (zero on the domain boundary but **not** at an interior element's
/// corners) the element corners moved, so ref 1 is neither the node box nor the
/// ref-3 box — and this pins that the ref level is actually used.
#[test]
fn d813_refinement_level_changes_the_box() {
    let m = read2(&test_data("d813_curved_quad-p2.mesh.txt"));
    let (_, r1) = m.get_bounding_box(1);
    let (_, r3) = m.get_bounding_box(3);
    assert!((r1[0] - 1.1742640687119286).abs() < TOL, "ref 1 max x = {r1:?}");
    assert!((r3[0] - 1.1870166548308927).abs() < TOL, "ref 3 max x = {r3:?}");
    assert!(
        r3[0] > r1[0] + 1e-6,
        "ref 3 must find a deeper interior extreme than ref 1"
    );
    // …and the vertex table (`bounding_box`, which for a curved mesh is the
    // high-order *node* box) is not the answer either: MFEM's `Nodes != NULL`
    // arm never looks at it.
    let (vlo, vhi) = m.bounding_box();
    assert!(vlo[0] == 0.0 && vhi[0] > 1.17, "node box {vlo:?} {vhi:?}");
    assert!(
        (vhi[0] - r3[0]).abs() > 1e-9 || (vhi[0] - r1[0]).abs() > 1e-9,
        "the node box coincides with the geometry box here; use another fixture"
    );
}

// ─── 3. the folded periodic meshes (the ex9 driver) ─────────────────────────

/// `Mesh::bounding_box()` still returns the vertex box (its documented `f64`
/// contract, used by other consumers); `get_bounding_box(ref)` returns MFEM's.
///
/// The 3-D `refine = 2` rows are excluded here and pinned separately by
/// `d813_cube_refined_folded_geometry_is_a_registered_residual` — they diverge
/// in `refine_uniform_3d`'s folded-geometry transport, not in the box.
#[test]
fn d813_bounding_box_matches_mfem_on_folded_periodic_meshes() {
    let src = text(&test_data("d813_mfem_bbox.txt"));
    let mut rows = 0;
    for line in src.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let file = t[0];
        let refine: usize = t[1].parse().expect("refine");
        let sdim: usize = t[2].parse().expect("sdim");
        let ref_level: i32 = t[3].parse().expect("ref");
        let want: Vec<f64> = t[4..].iter().map(|s| s.parse::<f64>().expect("v")).collect();
        if sdim == 3 && refine > 0 {
            continue; // registered residual, see below
        }
        let path = data_dir().join(file);
        let got = if sdim == 2 {
            let mut m = read2(&path);
            for _ in 0..refine {
                m = refine_uniform(&m);
            }
            box2(&m, ref_level)
        } else {
            let mut m = read3(&path);
            for _ in 0..refine {
                m = refine_uniform_3d(&m);
            }
            box3(&m, ref_level)
        };
        let d = max_abs_diff(&got, &want);
        assert!(
            d <= TOL,
            "{file} refine={refine} ref={ref_level}: |box - MFEM| = {d:e}\n  \
             fem-rs {got:?}\n  MFEM   {want:?}"
        );
        rows += 1;
    }
    assert_eq!(rows, 25, "3 meshes x ref 1..5 minus the 5 3-D refined rows");
}

/// The ex9 numbers, named: `data/periodic-hexagon.mesh` at the example's
/// `ref = max(order, 1) = 3` is `[-1, -0.866…] x [1, 0.866…]`, while the
/// vertex box is `[-0.5, -0.433…] x [0.5, 0.433…]` — a different box, hence a
/// different initial condition and a different velocity field.
///
/// (Round 77 measured the vertex box as `[-0.5, -0.866] x [0.75, 0.433]`; that
/// was the *first-wins* table of D813-2, whose minimum x came from one folded
/// copy.  With the mean rule the vertex box is symmetric and smaller still.)
#[test]
fn d813_periodic_hexagon_box_is_mfem_s_not_the_vertex_box() {
    let m = read2(&data_dir().join("periodic-hexagon.mesh"));
    let (lo, hi) = m.get_bounding_box(3);
    assert!((lo[0] + 1.0).abs() < 1e-14, "{lo:?}");
    assert!((lo[1] + 0.86602540378443871).abs() < 1e-14, "{lo:?}");
    assert!((hi[0] - 1.0).abs() < 1e-14, "{hi:?}");
    assert!((hi[1] - 0.86602540378443871).abs() < 1e-14, "{hi:?}");

    let (vlo, vhi) = m.bounding_box();
    assert!((vlo[0] + 0.5).abs() < 1e-15 && (vhi[0] - 0.5).abs() < 1e-15, "{vlo:?} {vhi:?}");
    assert!((vlo[1] + 0.4330127018922193).abs() < 1e-15, "{vlo:?}");
    assert!(
        (vhi[0] - hi[0]).abs() > 1e-3,
        "the vertex box must not be mistaken for MFEM's box"
    );

    // After the example's two uniform refinements the box is unchanged (the
    // folded geometry still reaches the same extremes) — the call sits *after*
    // the refinement loop in `ex9.cpp:232-241`.
    let mut r = m;
    for _ in 0..2 {
        r = refine_uniform(&r);
    }
    let (rlo, rhi) = r.get_bounding_box(3);
    assert!((rlo[0] + 1.0).abs() < 1e-14 && (rhi[1] - 0.86602540378443871).abs() < 1e-14);
}

/// **Registered residual (D813-3 sweep).**  The oracle's 3-D folded rows with
/// `refine = 2` do *not* reproduce: `periodic-cube.mesh` refined twice gives
/// `+/-1/3` here against MFEM's `+/-1`.  `+/-1/3` is the **unrefined vertex
/// box** of that mesh, so the divergence is in `refine_uniform_3d`'s handling
/// of the folded per-element (`L2_T1_3D_P1`) table — the *unrefined* 3-D rows
/// (5/5) and every 2-D row, refined or not (20/20), match MFEM exactly.
///
/// This test pins the measured gap so the debt cannot silently change shape;
/// it is **not** an endorsement of the value.  The refined 3-D folded geometry
/// is out of this lane's scope (`bbox` is correct: it reports what the mesh it
/// is handed actually contains).
#[test]
fn d813_cube_refined_folded_geometry_is_a_registered_residual() {
    let mut m = read3(&data_dir().join("periodic-cube.mesh"));
    assert!(m.geometry.is_some(), "the unrefined cube carries the folded table");
    let (lo, hi) = m.get_bounding_box(1);
    assert!((lo[0] + 1.0).abs() < 1e-14 && (hi[0] - 1.0).abs() < 1e-14, "{lo:?} {hi:?}");

    for _ in 0..2 {
        m = refine_uniform_3d(&m);
    }
    let (lo, hi) = m.get_bounding_box(1);
    let one_third = 1.0 / 3.0;
    for d in 0..3 {
        // `refine_uniform_3d` keeps *a* table (or none) but loses the folding:
        // the extremes collapse to the unrefined vertex box.  The mesh file's
        // own coordinates are the rounded `0.333333`, hence the loose bound.
        assert!(
            (lo[d] + one_third).abs() < 1e-5 && (hi[d] - one_third).abs() < 1e-5,
            "registered residual changed — re-measure: lo={lo:?} hi={hi:?}"
        );
    }
    eprintln!(
        "registered residual: periodic-cube refine=2 box = (+/-1/3) vs MFEM (+/-1); \
         geometry present = {}",
        m.geometry.is_some()
    );
}

// ─── 4. the `Nodes == NULL` arm is untouched ────────────────────────────────

/// A straight mesh (no geometry table ⇒ `Nodes == NULL`) falls back to the
/// vertex box for every ref level, and that is `bounding_box()` bit for bit.
#[test]
fn d813_straight_mesh_box_is_the_vertex_box() {
    for name in ["star.mesh", "inline-quad.mesh"] {
        let m = read2(&data_dir().join(name));
        assert!(m.geometry.is_none(), "{name} must be straight");
        let (vlo, vhi) = m.bounding_box();
        for ref_level in 1..=4 {
            let (lo, hi) = m.get_bounding_box(ref_level);
            assert_eq!(lo, vlo, "{name} ref={ref_level}");
            assert_eq!(hi, vhi, "{name} ref={ref_level}");
        }
    }
}

/// MFEM clamps `Times = max(ref, 1)` (`fem/geom.cpp:1139`), so ref 0 and ref 1
/// must agree and a negative ref must not panic.
#[test]
fn d813_ref_level_clamp_matches_mfem() {
    let m = read2(&data_dir().join("periodic-hexagon.mesh"));
    assert_eq!(m.get_bounding_box(0), m.get_bounding_box(1));
    assert_eq!(m.get_bounding_box(-3), m.get_bounding_box(1));
}

