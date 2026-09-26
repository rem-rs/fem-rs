//! D814-2 — `refine_uniform_3d` must transport a **folded `L2_T1_3D_P1`
//! geometry table** (`data/periodic-cube.mesh`, the per-element `nodes`
//! encoding of a geometrically periodic cube) through uniform refinement,
//! exactly like MFEM.
//!
//! MFEM semantics (`Mesh::UniformRefinement` with `Nodes != NULL`): the fine
//! `nodes` grid function is the refinement operator applied to the coarse
//! one — for a trilinear discontinuous parent, every fine element evaluates
//! *its parent's own* 8 corner dofs at the child's corner reference points
//! mapped into the parent frame (`parent_ref = origin + 0.5·child_ref`) — and
//! the fine element's 8 geometry dofs stay *its own* (element-major fresh
//! ids, no sharing): a face on the periodic seam keeps a different coordinate
//! on each side, which is the whole point of the folded representation.
//! `Mesh::UpdateNodes` then ends with `SetVerticesFromNodes`
//! (`mesh/mesh.cpp:7246` → `GridFunction::GetNodalValues`,
//! `fem/gridfunc.cpp:1889`): the fine `vertices` are the arithmetic mean over
//! every element reference of those folded values.
//!
//! **The registered defect (round 78)**: fem-rs dropped the table entirely —
//! `HexQkGeometry::new` refuses order 1, so the Hex8 refinement fell back to
//! straight vertex averaging and the refined mesh was *straight-sided*.
//! Measured against the MFEM 4.10 oracle below, before the fix: after one
//! refinement 3024/5184 geometry entries differ (>1e-9, max |diff| = 1.0 —
//! MFEM keeps the folded ±1 seam coordinates where the averaged vertices
//! land at ±2/3 or 0), after two refinements 27072/41472 entries (max 1.0);
//! the vertex table diverges the same way (max 0.99999950 / 1.16666625).
//!
//! **Oracle**: MFEM 4.10 serial (`$HOME/mfem410_ser`), probe
//! `tmp/d80a/probe.cpp` — read `data/periodic-cube.mesh`, `UniformRefinement()`
//! twice, and after each step dump the per-element `nodes` values (the FE's
//! own dof order, which is the file's `L2` order: the lexicographic tensor
//! order) and the vertex table at precision 17.  Fixtures (`crates/mesh/tests/
//! data/`, `.txt` because `*.mesh` is git-ignored):
//!
//! * `d814_periodic_cube.mesh.txt` — the coarse input (byte copy of
//!   `data/periodic-cube.mesh`);
//! * `d814_mfem_periodic_cube_l2_r{0,1,2}.txt` — per element: id, then the 8
//!   geometry dof xyz triples in MFEM's `L2_T1_3D_P1` file dof order;
//! * `d814_mfem_periodic_cube_vertices_r{1,2}.txt` — the vertex tables.
//!
//! The file dof order is the *lexicographic* lattice (x fastest), a
//! permutation of the mesh's own slot order (MFEM vertex order): file index
//! of slot `i` is `x + 2y + 4z` over the slot's reference corner, i.e.
//! `[0, 1, 3, 2, 4, 5, 7, 6]` (the same pairing `fem-io`'s reader/writer use,
//! pinned against MFEM's own `L2_T1_3D_P3` output there).

use fem_core::NodeId;
use fem_io::mfem::read_mfem;
use fem_mesh::{Mesh, refine_uniform_3d};
use std::io::Cursor;

const COARSE_MESH: &str = include_str!("data/d814_periodic_cube.mesh.txt");
const ORACLE_L2_R0: &str = include_str!("data/d814_mfem_periodic_cube_l2_r0.txt");
const ORACLE_L2_R1: &str = include_str!("data/d814_mfem_periodic_cube_l2_r1.txt");
const ORACLE_L2_R2: &str = include_str!("data/d814_mfem_periodic_cube_l2_r2.txt");
const ORACLE_VERTS_R1: &str = include_str!("data/d814_mfem_periodic_cube_vertices_r1.txt");
const ORACLE_VERTS_R2: &str = include_str!("data/d814_mfem_periodic_cube_vertices_r2.txt");

/// Relative tolerance of the oracle comparison (acceptance: ≤1e-13; the
/// measured agreement is 0.0 after one refinement and 1 ulp (1.1e-16) after
/// two, where MFEM's interpolation-matrix summation order and the direct
/// trilinear accumulation differ in the last bit).
const TOL: f64 = 1e-13;

/// File (`L2` lexicographic) dof index of the mesh's own geometry slot `i`.
const SLOT_TO_FILE: [usize; 8] = [0, 1, 3, 2, 4, 5, 7, 6];

/// Per-element oracle rows: `(id, [[x, y, z]; 8])`, values in MFEM's file dof
/// order.
fn oracle_elems(text: &'static str) -> Vec<(u32, [[f64; 3]; 8])> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<f64> = l.split_whitespace().map(|v| v.parse().expect("value")).collect();
            assert_eq!(t.len(), 25, "id + 8 xyz triples per line, got {l:?}");
            let mut dofs = [[0.0_f64; 3]; 8];
            for (k, d) in dofs.iter_mut().enumerate() {
                *d = [t[1 + 3 * k], t[2 + 3 * k], t[3 + 3 * k]];
            }
            (t[0] as u32, dofs)
        })
        .collect()
}

fn oracle_vertices(text: &'static str) -> Vec<[f64; 3]> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<f64> = l.split_whitespace().map(|v| v.parse().expect("value")).collect();
            assert_eq!(t.len(), 3, "one xyz row per line, got {l:?}");
            [t[0], t[1], t[2]]
        })
        .collect()
}

fn coarse_mesh() -> Mesh<3> {
    read_mfem(Cursor::new(COARSE_MESH.as_bytes()))
        .expect("read the coarse periodic cube")
        .mesh3d
        .expect("3-D mesh")
}

/// The geometry table is the *discontinuous* layout: element-major rows of
/// fresh, unshared dof ids (the folding cannot survive an averaged, shared
/// table).
fn assert_discontinuous_layout(mesh: &Mesh<3>, what: &str) {
    let g = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(g.order, 1, "{what}: refined geometry stays order 1 (L2_T1_3D_P1)");
    assert_eq!(g.nodes_per_elem, 8, "{what}");
    assert_eq!(g.n_nodes, 8 * mesh.n_elems(), "{what}: no dof sharing across elements");
    assert_eq!(g.conn.len(), 8 * mesh.n_elems(), "{what}");
    for (e, row) in g.conn.chunks_exact(8).enumerate() {
        for (i, &d) in row.iter().enumerate() {
            assert_eq!(d as usize, e * 8 + i, "{what}: element {e} must own fresh dof ids");
        }
    }
}

fn compare_geometry(mesh: &Mesh<3>, oracle: &[(u32, [[f64; 3]; 8])], what: &str) {
    let g = mesh.geometry.as_ref().expect(what);
    assert_eq!(mesh.n_elems(), oracle.len(), "{what}: element count");
    let mut max_diff = 0.0_f64;
    let mut worst = (0_usize, 0_usize);
    for (e, &(id, file_order)) in oracle.iter().enumerate() {
        assert_eq!(id as usize, e, "{what}: oracle rows are element-major");
        for (i, &file) in SLOT_TO_FILE.iter().enumerate() {
            let dof = g.conn[e * 8 + i] as usize;
            for c in 0..3 {
                let got = g.coords[dof * 3 + c];
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

fn compare_vertices(mesh: &Mesh<3>, oracle: &[[f64; 3]], what: &str) {
    assert_eq!(mesh.n_nodes(), oracle.len(), "{what}: vertex count");
    let mut max_diff = 0.0_f64;
    for (v, want) in oracle.iter().enumerate() {
        let got = mesh.coords_of(v as NodeId);
        for c in 0..3 {
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

/// The folded table survives two uniform refinements, every dof value and the
/// vertex table matching MFEM's own refinement of the same mesh.
#[test]
fn d814_folded_l2_p1_geometry_survives_two_uniform_refinements() {
    let mesh = coarse_mesh();

    // Coarse cross-check of the slot-order pairing: the read-back table (mesh
    // slot order) must equal the oracle file rows through `SLOT_TO_FILE`.
    compare_geometry(&mesh, &oracle_elems(ORACLE_L2_R0), "coarse (read-back)");
    assert_eq!(mesh.n_elems(), 27);

    let r1 = refine_uniform_3d(&mesh);
    assert_eq!(r1.n_elems(), 216, "one refinement octs every hex");
    assert_discontinuous_layout(&r1, "r1");
    compare_geometry(&r1, &oracle_elems(ORACLE_L2_R1), "geometry r1");
    compare_vertices(&r1, &oracle_vertices(ORACLE_VERTS_R1), "vertices r1");

    let r2 = refine_uniform_3d(&r1);
    assert_eq!(r2.n_elems(), 1728);
    assert_discontinuous_layout(&r2, "r2");
    compare_geometry(&r2, &oracle_elems(ORACLE_L2_R2), "geometry r2");
    compare_vertices(&r2, &oracle_vertices(ORACLE_VERTS_R2), "vertices r2");

    // Folding semantics: the periodic seam keeps its ±1 coordinates on the
    // fine elements (the registered pre-fix failure averaged them toward 0 —
    // the straight-sided children never exceeded 2/3).
    let g = r2.geometry.as_ref().unwrap();
    let max_abs_x = (0..g.n_nodes).map(|d| g.coords[d * 3].abs()).fold(0.0_f64, f64::max);
    assert_eq!(max_abs_x, 1.0, "the folded seam coordinates must survive, not average");
}
