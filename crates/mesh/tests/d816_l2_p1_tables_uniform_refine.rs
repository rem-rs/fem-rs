//! D816 — **order-1 discontinuous (`L2_T1_*_P1`) geometry tables** must
//! survive uniform refinement on every family that can carry them, exactly
//! like MFEM's `Mesh::UniformRefinement` on a `nodes`-carrying mesh.
//!
//! The 3-D Hex8 case is D814-2 (`d814_l2_p1_hex_uniform_refine.rs`).  This
//! file pins the three remaining families (D816-1: 2-D quads; D816-2: tets
//! and wedges):
//!
//! * **2-D quads** — `data/periodic-hexagon.mesh` (12 quads) and
//!   `data/periodic-square.mesh` (9 quads): the folded `L2_T1_2D_P1` table.
//!   The *table values* were already transported element-major by
//!   `refine_uniform_quad4` (the registered "table dropped" premise was
//!   refuted by measurement: r0/r1/r2 geom max |diff| = 0.0 against the
//!   oracle before this round's fix).  What *was* broken is the fine
//!   **vertex table**: MFEM's `UniformRefinement2D_base(update_nodes)` ends
//!   with `UpdateNodes` → `SetVerticesFromNodes` — the fine vertices are the
//!   mean over the element references of the refined folded values — while
//!   the kernels kept straight averages of the coarse *compatibility*
//!   vertices.  On a seam those differ (measured before the fix: r1 max
//!   |diff| = 0.5 on 25/48 hexagon vertices, r2 = 0.75 on 129/192; square:
//!   0.5 on 12/36 and 0.75 on 72/144).
//! * **Tet4 / Prism6** — `Mesh::SetCurvature(1, true)` fixtures
//!   (`d816_beam_tet_l2p1.mesh.txt`, `d816_beam_wedge_l2p1.mesh.txt`): the
//!   table was dropped entirely (`geometry = None`, straight-sided children)
//!   and is now transported — every fine element owns fresh corner dofs
//!   holding the parent's own P1 shape functions evaluated at the child
//!   embedding's reference points (`tet_children` / `pri_children` point
//!   matrices) — followed by the same `SetVerticesFromNodes` mean.
//!
//! **Oracle**: MFEM 4.10 serial (`$HOME/mfem410_ser`), probe
//! `tmp/d81a/probe.cpp` — read the mesh (optionally `SetCurvature(1, true)`),
//! `UniformRefinement()` twice, and after each step dump the per-element
//! `nodes` values (the FE's own dof order = the file's `L2` order) and the
//! vertex table at precision 17.  Fixtures (`crates/mesh/tests/data/`, `.txt`
//! because `*.mesh` is git-ignored):
//!
//! * `d816_periodic_{hexagon,square}.mesh.txt` — byte copies of the coarse
//!   inputs; `d816_beam_{tet,wedge}_l2p1.mesh.txt` — MFEM's own `Mesh::Print`
//!   of the coarse mesh *after* `SetCurvature(1, true)` (precision 17), so
//!   the coarse read-back is bit-comparable with the oracle r0 rows;
//! * `d816_mfem_{hexagon,square,beam_tet,beam_wedge}_l2_r{0,1,2}.txt` — per
//!   element: id, then the geometry dof values in MFEM's file dof order;
//! * `d816_mfem_{...}_vertices_r{1,2}.txt` — the refined vertex tables.
//!
//! File dof order vs the mesh's own slot order: the 2-D `L2_T1_2D_P1` lattice
//! is lexicographic (x fastest) — `[0, 1, 3, 2]` over the H1 vertex slots;
//! the order-1 tet and wedge enumerations are the vertex order (identity).

use fem_core::NodeId;
use fem_io::mfem::read_mfem;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use std::io::Cursor;

/// Relative tolerance of the oracle comparisons (acceptance: ≤1e-13; measured
/// agreements are listed per test).
const TOL: f64 = 1e-13;

/// Per-element oracle rows: `(id, dofs)`, each dof's components in MFEM's
/// file dof order.
fn oracle_elems(text: &'static str, npe: usize, dim: usize) -> Vec<(u32, Vec<Vec<f64>>)> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<f64> = l.split_whitespace().map(|v| v.parse().expect("value")).collect();
            assert_eq!(t.len(), 1 + npe * dim, "id + {npe} dof rows, got {l:?}");
            let dofs = (0..npe)
                .map(|k| t[1 + k * dim..1 + (k + 1) * dim].to_vec())
                .collect();
            (t[0] as u32, dofs)
        })
        .collect()
}

fn oracle_vertices(text: &'static str, dim: usize) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<f64> = l.split_whitespace().map(|v| v.parse().expect("value")).collect();
            assert_eq!(t.len(), 3, "one xyz row per line, got {l:?}");
            t[..dim].to_vec()
        })
        .collect()
}

/// The geometry table is the *discontinuous* layout: element-major rows of
/// fresh, unshared dof ids (the folding cannot survive an averaged, shared
/// table).
fn assert_discontinuous_layout<const D: usize>(mesh: &Mesh<D>, npe: usize, what: &str) {
    let g = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(g.order, 1, "{what}: refined geometry stays order 1 (L2_T1_*_P1)");
    assert_eq!(g.nodes_per_elem, npe, "{what}");
    assert_eq!(g.n_nodes, npe * mesh.n_elems(), "{what}: no dof sharing across elements");
    assert_eq!(g.conn.len(), npe * mesh.n_elems(), "{what}");
    for (e, row) in g.conn.chunks_exact(npe).enumerate() {
        for (i, &d) in row.iter().enumerate() {
            assert_eq!(d as usize, e * npe + i, "{what}: element {e} must own fresh dof ids");
        }
    }
}

fn compare_geometry<const D: usize>(
    mesh: &Mesh<D>,
    oracle: &[(u32, Vec<Vec<f64>>)],
    slot_to_file: &[usize],
    what: &str,
) {
    let g = mesh.geometry.as_ref().expect(what);
    let npe = slot_to_file.len();
    assert_eq!(mesh.n_elems(), oracle.len(), "{what}: element count");
    let mut max_diff = 0.0_f64;
    let mut worst = (0_usize, 0_usize);
    for (e, (id, file_order)) in oracle.iter().enumerate() {
        assert_eq!(*id as usize, e, "{what}: oracle rows are element-major");
        for (i, &file) in slot_to_file.iter().enumerate() {
            let dof = g.conn[e * npe + i] as usize;
            for c in 0..2 {
                let got = g.coords[dof * D + c];
                let want = file_order[file][c];
                let d = (got - want).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (e, i);
                }
                assert!(
                    d <= TOL,
                    "{what}: element {e} slot {i} component {c}: {got} vs MFEM {want}"
                );
            }
        }
    }
    eprintln!("{what}: max |diff| = {max_diff:.3e} (worst element/slot {worst:?})");
}

fn compare_vertices<const D: usize>(mesh: &Mesh<D>, oracle: &[Vec<f64>], what: &str) {
    assert_eq!(mesh.n_nodes(), oracle.len(), "{what}: vertex count");
    let mut max_diff = 0.0_f64;
    for (v, want) in oracle.iter().enumerate() {
        let got = mesh.coords_of(v as NodeId);
        for c in 0..2 {
            let d = (got[c] - want[c]).abs();
            max_diff = max_diff.max(d);
            assert!(
                d <= TOL,
                "{what}: vertex {v} component {c}: {} vs MFEM {}",
                got[c],
                want[c]
            );
        }
    }
    eprintln!("{what}: max |diff| = {max_diff:.3e}");
}

/// The 2-D folded `L2_T1_2D_P1` tables: values bit-exact (already so before
/// this round — the premise "table dropped" was refuted), vertices now the
/// `SetVerticesFromNodes` means, two refinements in a row.
fn check_quad(
    mesh_txt: &'static str,
    l2: [&'static str; 3],
    verts: [&'static str; 2],
    r1_elems: usize,
    r2_elems: usize,
) {
    const SLOT_TO_FILE: [usize; 4] = [0, 1, 3, 2];
    let mesh = read_mfem(Cursor::new(mesh_txt.as_bytes()))
        .expect("read the coarse periodic mesh")
        .mesh2d
        .expect("2-D mesh");
    compare_geometry(&mesh, &oracle_elems(l2[0], 4, 2), &SLOT_TO_FILE, "coarse (read-back)");

    let r1 = refine_uniform(&mesh);
    assert_eq!(r1.n_elems(), r1_elems, "one refinement quarters every quad");
    assert_discontinuous_layout(&r1, 4, "r1");
    compare_geometry(&r1, &oracle_elems(l2[1], 4, 2), &SLOT_TO_FILE, "geometry r1");
    compare_vertices(&r1, &oracle_vertices(verts[0], 2), "vertices r1");

    let r2 = refine_uniform(&r1);
    assert_eq!(r2.n_elems(), r2_elems);
    assert_discontinuous_layout(&r2, 4, "r2");
    compare_geometry(&r2, &oracle_elems(l2[2], 4, 2), &SLOT_TO_FILE, "geometry r2");
    compare_vertices(&r2, &oracle_vertices(verts[1], 2), "vertices r2");

    // Folding semantics: the periodic seam keeps its full-range coordinates on
    // the fine elements (an averaged representation would shrink them).
    let g = r2.geometry.as_ref().unwrap();
    let max_abs_x = (0..g.n_nodes).map(|d| g.coords[d * 2].abs()).fold(0.0_f64, f64::max);
    assert_eq!(max_abs_x, 1.0, "the folded seam coordinates must survive, not average");
}

/// `data/periodic-hexagon.mesh`: folded 12-quad ring.
#[test]
fn d816_folded_l2_p1_quad_hexagon_survives_two_uniform_refinements() {
    check_quad(
        include_str!("data/d816_periodic_hexagon.mesh.txt"),
        [
            include_str!("data/d816_mfem_hexagon_l2_r0.txt"),
            include_str!("data/d816_mfem_hexagon_l2_r1.txt"),
            include_str!("data/d816_mfem_hexagon_l2_r2.txt"),
        ],
        [
            include_str!("data/d816_mfem_hexagon_vertices_r1.txt"),
            include_str!("data/d816_mfem_hexagon_vertices_r2.txt"),
        ],
        48,
        192,
    );
}

/// `data/periodic-square.mesh`: folded 9-quad torus.
#[test]
fn d816_folded_l2_p1_quad_square_survives_two_uniform_refinements() {
    check_quad(
        include_str!("data/d816_periodic_square.mesh.txt"),
        [
            include_str!("data/d816_mfem_square_l2_r0.txt"),
            include_str!("data/d816_mfem_square_l2_r1.txt"),
            include_str!("data/d816_mfem_square_l2_r2.txt"),
        ],
        [
            include_str!("data/d816_mfem_square_vertices_r1.txt"),
            include_str!("data/d816_mfem_square_vertices_r2.txt"),
        ],
        36,
        144,
    );
}

/// The 3-D order-1 tables (tet / wedge): the table was dropped entirely
/// before this round; it is now transported and the vertices are the
/// `SetVerticesFromNodes` means.
fn check_3d(
    mesh_txt: &'static str,
    l2: [&'static str; 3],
    verts: [&'static str; 2],
    npe: usize,
    slot_to_file: &[usize],
    r1_elems: usize,
    r2_elems: usize,
) {
    let mesh = read_mfem(Cursor::new(mesh_txt.as_bytes()))
        .expect("read the coarse L2 mesh")
        .mesh3d
        .expect("3-D mesh");
    compare_geometry(&mesh, &oracle_elems(l2[0], npe, 3), slot_to_file, "coarse (read-back)");

    let r1 = refine_uniform_3d(&mesh);
    assert_eq!(r1.n_elems(), r1_elems);
    assert_discontinuous_layout(&r1, npe, "r1");
    compare_geometry(&r1, &oracle_elems(l2[1], npe, 3), slot_to_file, "geometry r1");
    compare_vertices(&r1, &oracle_vertices(verts[0], 3), "vertices r1");

    let r2 = refine_uniform_3d(&r1);
    assert_eq!(r2.n_elems(), r2_elems);
    assert_discontinuous_layout(&r2, npe, "r2");
    compare_geometry(&r2, &oracle_elems(l2[2], npe, 3), slot_to_file, "geometry r2");
    compare_vertices(&r2, &oracle_vertices(verts[1], 3), "vertices r2");
}

/// `beam-tet.mesh` + `Mesh::SetCurvature(1, true)`: 48 tets with a
/// discontinuous order-1 table.
#[test]
fn d816_l2_p1_tet_table_survives_two_uniform_refinements() {
    check_3d(
        include_str!("data/d816_beam_tet_l2p1.mesh.txt"),
        [
            include_str!("data/d816_mfem_beam_tet_l2_r0.txt"),
            include_str!("data/d816_mfem_beam_tet_l2_r1.txt"),
            include_str!("data/d816_mfem_beam_tet_l2_r2.txt"),
        ],
        [
            include_str!("data/d816_mfem_beam_tet_vertices_r1.txt"),
            include_str!("data/d816_mfem_beam_tet_vertices_r2.txt"),
        ],
        4,
        &[0, 1, 2, 3],
        384,
        3072,
    );
}

/// `beam-wedge.mesh` + `Mesh::SetCurvature(1, true)`: 8 prisms with a
/// discontinuous order-1 table.
#[test]
fn d816_l2_p1_prism_table_survives_two_uniform_refinements() {
    check_3d(
        include_str!("data/d816_beam_wedge_l2p1.mesh.txt"),
        [
            include_str!("data/d816_mfem_beam_wedge_l2_r0.txt"),
            include_str!("data/d816_mfem_beam_wedge_l2_r1.txt"),
            include_str!("data/d816_mfem_beam_wedge_l2_r2.txt"),
        ],
        [
            include_str!("data/d816_mfem_beam_wedge_vertices_r1.txt"),
            include_str!("data/d816_mfem_beam_wedge_vertices_r2.txt"),
        ],
        6,
        &[0, 1, 2, 3, 4, 5],
        64,
        512,
    );
}
