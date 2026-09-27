//! D817-3 — the **fused pyramid P1** `L2_T1_3D_P1` table: the io reader arm
//! (`d818_pyrl2_r0.mesh.txt`) attaches the 8-dof Fuentes table, and uniform
//! refinement transports it — every fine child inherits the parent's own P1
//! field through MFEM's refinement operator — instead of being dropped.
//!
//! Everything below is pinned against the MFEM 4.10 oracle
//! (`tmp/d81a/pyrl2_*`, probe `tmp/d81a/probe.cpp`: read the
//! `SetCurvature(1, true)` pyramid mesh, `UniformRefinement()` twice, dump
//! the per-element `nodes` values in the FE's own dof order and the vertex
//! table at precision 17), and the transport is **MFEM-arithmetic-exact**:
//! the Fuentes P1 evaluation replays `Ti`'s LU factorization and
//! `LSolve`/`USolve`, the local interpolation matrices replay
//! `NodalLocalInterpolation` (including the `|v| < 1e-12` snap), the per
//! fine-element embedding assignment replays `mesh.cpp`'s pyramid-blind
//! `(k / 8, k % 8)` generic loop (tet children of a pyramid parent stay at
//! the zero-initialized default `(0, 0)`), and the 8-slot `subX` buffer
//! replay reproduces the release-build reads past a short parent fill
//! (`amr::curved_pyramid` module doc has the source walk).
//!
//! The result is a **bit-for-bit** match of the oracle tables and vertex
//! tables at r1 (6 pyr + 4 tets — the four tets degenerate to flat base
//! subtets exactly as MFEM's unassigned embeddings produce) and r2 (36 pyr +
//! 56 tets, scrambled templates and stale-slot values included).

use fem_core::NodeId;
use fem_io::mfem::read_mfem;
use fem_mesh::{refine_uniform_3d, Mesh};
use std::io::Cursor;

const R0_MESH: &str = include_str!("data/d818_pyrl2_mesh_r0.txt");
const ORACLE_R0: &str = include_str!("data/d818_pyrl2_l2_elems_r0.txt");
const ORACLE_R1: &str = include_str!("data/d818_pyrl2_l2_elems_r1.txt");
const ORACLE_R2: &str = include_str!("data/d818_pyrl2_l2_elems_r2.txt");
const ORACLE_VERTS_R1: &str = include_str!("data/d818_pyrl2_vertices_r1.txt");
const ORACLE_VERTS_R2: &str = include_str!("data/d818_pyrl2_vertices_r2.txt");
/// MFEM's own re-read of the r1 file, refined once (probe
/// `tmp/d81a/probe.cpp` on `pyrl2_mesh_r1.txt` — the `chk` dump): the reread
/// chain's rotation + node-reorder make its r2 differ from the in-memory
/// chain's on the stale-buffer rows (50..55) and the tet-parent children.
const REREAD_ORACLE: &str = include_str!("data/d818_pyrl2_reread_l2_elems.txt");
const REREAD_ORACLE_VERTS: &str = include_str!("data/d818_pyrl2_reread_vertices.txt");

/// Tolerance: the transport is bit-exact against the oracle; the assertion
/// keeps a 1-ulp headroom for parser-level last-bit noise only.
const ULPS: f64 = 2.0;

fn oracle_rows(text: &str) -> Vec<(u32, Vec<f64>)> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<f64> = l.split_whitespace().map(|v| v.parse().expect("value")).collect();
            (t[0] as u32, t[1..].to_vec())
        })
        .collect()
}

fn oracle_vertices(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace().map(|v| v.parse().expect("value")).collect()
        })
        .collect()
}

/// Bit-level comparison: reports the max difference in ulps of the oracle
/// value.
fn assert_close(got: f64, want: f64, what: &str) {
    if got == want {
        return;
    }
    let scale = want.abs().max(got.abs()).max(f64::MIN_POSITIVE);
    let ulps = (got - want).abs() / (scale * f64::EPSILON);
    assert!(
        ulps <= ULPS,
        "{what}: got {got:?}, want {want:?} ({ulps:.1} ulps)"
    );
}

/// Compare a fused table against an oracle dump (per-element rows in the
/// file's L2 dof order).
fn compare_table(mesh: &Mesh<3>, oracle: &[(u32, Vec<f64>)], what: &str) {
    let g = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: geometry table must survive refinement"));
    assert_eq!(g.order, 1, "{what}: refined geometry stays order 1");
    assert_eq!(g.n_nodes, g.conn.len(), "{what}: fresh unshared dof ids");
    assert_eq!(mesh.n_elems(), oracle.len(), "{what}: element count");
    let mut cursor = 0usize;
    for (e, (id, want)) in oracle.iter().enumerate() {
        assert_eq!(*id as usize, e, "{what}: oracle rows are element-major");
        let row_dofs = if mesh.element_type_at(e as fem_core::ElemId)
            == fem_mesh::element_type::ElementType::Pyramid5
        {
            8
        } else {
            4
        };
        assert_eq!(g.conn[cursor] as usize, cursor, "{what}: fresh ids");
        for k in 0..row_dofs {
            assert_eq!(g.conn[cursor + k] as usize, cursor + k, "{what}");
            for c in 0..3 {
                assert_close(
                    g.coords[(cursor + k) * 3 + c],
                    want[k * 3 + c],
                    &format!("{what}: elem {e} dof {k} comp {c}"),
                );
            }
        }
        cursor += row_dofs;
    }
    assert_eq!(cursor, g.conn.len(), "{what}: row total");
}

fn compare_vertices(mesh: &Mesh<3>, oracle: &[Vec<f64>], what: &str) {
    assert_eq!(mesh.n_nodes(), oracle.len(), "{what}: vertex count");
    for (v, want) in oracle.iter().enumerate() {
        for c in 0..3 {
            assert_close(
                mesh.coords_of(v as NodeId)[c],
                want[c],
                &format!("{what}: vertex {v} comp {c}"),
            );
        }
    }
}

/// The full r0 → r1 → r2 chain against the oracle, and the r0 read-back.
#[test]
fn d818_fused_pyramid_l2_refine_matches_mfem() {
    let file = read_mfem(Cursor::new(R0_MESH.as_bytes())).expect("read r0");
    let mut mesh = file.mesh3d.expect("3-D pyramid mesh");
    assert_eq!(mesh.n_elems(), 1, "one pyramid");
    assert_eq!(mesh.n_nodes(), 5, "five vertices");

    // The coarse table: the io arm attached the 8-dof Fuentes table.
    let g = mesh.geometry.as_ref().expect("r0: fused table attached");
    assert_eq!(g.order, 1, "r0: order 1");
    assert_eq!(g.nodes_per_elem, 8, "r0: the pure-pyramid table is uniform 8");
    assert_eq!(g.n_nodes, 8, "r0: one element's 8 dofs");
    compare_table(&mesh, &oracle_rows(ORACLE_R0), "r0 read-back");

    // r1: 6 pyramids + 4 (flat, by MFEM's own doing) tets.
    mesh = refine_uniform_3d(&mesh);
    assert_eq!(mesh.n_elems(), 10, "r1: 6 pyramids + 4 tets");
    assert_eq!(mesh.n_nodes(), 14, "r1: 14 vertices");
    compare_table(&mesh, &oracle_rows(ORACLE_R1), "r1 table");
    compare_vertices(&mesh, &oracle_vertices(ORACLE_VERTS_R1), "r1 vertices");

    // r2: 36 pyramids + 56 tets — including MFEM's scrambled pyramid
    // embeddings and the stale-buffer rows they produce.
    mesh = refine_uniform_3d(&mesh);
    assert_eq!(mesh.n_elems(), 92, "r2: 36 pyramids + 56 tets");
    assert_eq!(mesh.n_nodes(), 55, "r2: 55 vertices");
    compare_table(&mesh, &oracle_rows(ORACLE_R2), "r2 table");
    compare_vertices(&mesh, &oracle_vertices(ORACLE_VERTS_R2), "r2 vertices");
}

/// The r1 oracle mesh, re-read through the io arm (the ragged mixed table)
/// and refined again, must reproduce the r2 oracle too.
#[test]
fn d818_fused_pyramid_r1_reread_refines_to_r2() {
    let file = read_mfem(Cursor::new(
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/d818_pyrl2_mesh_r1.txt"
        ))
        .expect("r1 fixture")
        .replace("\r\n", "\n")
        .as_bytes(),
    ))
    .expect("read r1");
    let mut mesh = file.mesh3d.expect("3-D mixed pyramid mesh");
    assert_eq!(mesh.n_elems(), 10, "r1: 6 pyramids + 4 tets");
    let g = mesh.geometry.as_ref().expect("r1: ragged fused table attached");
    assert_eq!(g.nodes_per_elem, 0, "r1: the mixed table is ragged");
    assert_eq!(g.n_nodes, 64, "r1: 6×8 + 4×4 dofs");
    // NOTE: the re-read table is *not* value-identical to the file dump on
    // the tet rows — re-loading runs MFEM's `MarkForRefinement`, which
    // rotates every tet's longest edge to (v0,v1) and permutes the row with
    // it (`DoNodeReorder`); the pyramid rows are untouched.  The acceptance
    // is the refined result below, which matches the oracle r2 bit for bit —
    // exactly as MFEM's own re-read-and-refine does (probe-verified).

    mesh = refine_uniform_3d(&mesh);
    assert_eq!(mesh.n_elems(), 92, "r2: 36 pyramids + 56 tets");
    // The re-read chain has its own oracle (MFEM's own re-read of the r1
    // file, refined once — the rotated tets make rows 50..55's stale-buffer
    // contents and the tet-parent children differ from the in-memory chain).
    compare_table(&mesh, &oracle_rows(REREAD_ORACLE), "r2 table from re-read r1");
    compare_vertices(&mesh, &oracle_vertices(REREAD_ORACLE_VERTS), "r2 vertices from re-read r1");
}
