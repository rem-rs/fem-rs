//! D817-3 — the **fused pyramid P1** arm of the `L2_T1_3D_P1` `nodes`
//! reader.
//!
//! Before this round a pyramid mesh's `L2_T1_3D_P1` section produced an
//! all-zero *vertex* table (the folded-vertex arm's `n_local_verts` came from
//! `l2_geometry_slots(Pyramid5, 1) = None` → 0 references), and a *mixed*
//! pyramid+tet mesh (what a pyramid uniform refinement produces) dropped the
//! table entirely (the uniform-stride row length `raw.len()/(NE·3)` is not an
//! integer for ragged 8/4 rows).
//!
//! The new arm attaches the table and reconstructs the vertex table the way
//! MFEM's loader does (`SetVerticesFromNodes` → `GridFunction::GetNodalValues`):
//! vertex value = (element shape functions at the reference vertex) · (element
//! dof values), averaged over element references — with the Fuentes P1's
//! base-vertex unit rows at dofs (0, 1, 3, 2) and the apex's fixed
//! combination `CalcShape(0, 0, 1)`.
//!
//! **Oracle** (MFEM 4.10, `$HOME/mfem410_ser`): `tmp/d83a/probe6.cpp` prints
//! the loaded mesh's vertex table at precision 17; the table values are the
//! file's own (precision 17) `nodes` dump.

use std::io::Cursor;
use std::path::Path;

fn fixture(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {name}: {e}"))
        .replace("\r\n", "\n")
}

/// The nodes-section values of an MFEM mesh fixture (Ordering: 1, byVDIM).
fn nodes_values(text: &str) -> Vec<f64> {
    let idx = text.find("Ordering: 1").expect("Ordering: 1");
    text[idx..].lines().skip(1).filter(|l| !l.trim().is_empty())
        .flat_map(|l| l.split_whitespace().map(|v| v.parse::<f64>().expect("value")))
        .collect()
}

/// MFEM's vertex table after loading (probe6 output, transcribed).
const R0_VERTICES: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [-5.5511151231257827e-17, -5.5511151231257827e-17, 1.0],
];

#[test]
fn d818_pyramid_l2_table_and_vertices_are_attached() {
    let text = fixture("d818_pyrl2_r0.mesh.txt");
    let file = fem_io::mfem::read_mfem(Cursor::new(text.as_bytes())).expect("read r0");
    let mesh = file.mesh3d.expect("3-D pyramid mesh");
    assert_eq!(mesh.n_elems(), 1);
    assert_eq!(mesh.n_nodes(), 5);

    // The geometry table: 8-dof Fuentes rows, fresh ids, the file's values.
    let g = mesh.geometry.as_ref().expect("fused pyramid table attached");
    assert_eq!(g.order, 1);
    assert_eq!(g.nodes_per_elem, 8, "pure-pyramid table: uniform 8-dof rows");
    assert_eq!(g.n_nodes, 8);
    for (i, &d) in g.conn.iter().enumerate() {
        assert_eq!(d as usize, i, "fresh unshared dof ids");
    }
    let want = nodes_values(&text);
    assert_eq!(want.len(), 24, "one element's 8 xyz dofs");
    assert_eq!(g.coords, want, "the table carries the file values verbatim");

    // The vertex table: MFEM's SetVerticesFromNodes recovery, apex included.
    for (v, want) in R0_VERTICES.iter().enumerate() {
        for c in 0..3 {
            let got = mesh.coords_of(v as u32)[c];
            assert_eq!(got, want[c], "vertex {v} comp {c}");
        }
    }
}

/// MFEM's vertex table after loading the refined (mixed pyr+tet) r1 mesh —
/// probe6 on `pyrl2_mesh_r1.txt`, precision 17.
const R1_VERTICES: [[f64; 3]; 14] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [-3.4694469519536142e-17, -3.4694469519536142e-17, 0.99999999999999989],
    [0.33333333333333331, 0.0, 0.0],
    [0.66666666666666663, 0.33333333333333331, 0.0],
    [0.33333333333333331, 0.66666666666666663, 0.0],
    [0.0, 0.33333333333333331, 0.0],
    [0.10000000000000001, 0.10000000000000001, 0.29999999999999999],
    [0.40000000000000002, 0.10000000000000001, 0.29999999999999999],
    [0.40000000000000002, 0.40000000000000002, 0.29999999999999999],
    [0.10000000000000001, 0.40000000000000002, 0.29999999999999999],
    [0.5, 0.5, -1.5419764230904951e-18],
];

#[test]
fn d818_mixed_pyramid_l2_table_is_attached() {
    let text = fixture("d818_pyrl2_r1.mesh.txt");
    let file = fem_io::mfem::read_mfem(Cursor::new(text.as_bytes())).expect("read r1");
    let mesh = file.mesh3d.expect("3-D mixed pyramid mesh");
    assert_eq!(mesh.n_elems(), 10, "6 pyramids + 4 tets");
    assert_eq!(mesh.n_nodes(), 14);

    let g = mesh.geometry.as_ref().expect("ragged fused table attached");
    assert_eq!(g.order, 1);
    assert_eq!(g.nodes_per_elem, 0, "the mixed pyr+tet table is ragged");
    assert_eq!(g.n_nodes, 64, "6×8 + 4×4 dofs");
    for (i, &d) in g.conn.iter().enumerate() {
        assert_eq!(d as usize, i, "fresh unshared dof ids");
    }
    let want = nodes_values(&text);
    assert_eq!(want.len(), 64 * 3);
    // The re-load runs `MarkForRefinement` (the mesh carries tets), whose
    // longest-edge rotation permutes the four tet rows with it — MFEM keeps
    // the pair consistent through `DoNodeReorder`.  The six pyramid rows
    // (elements 0..5, 8 dofs each) are untouched and carry the file's values
    // verbatim; the permuted tet rows are exercised end-to-end by the mesh
    // test's re-read chain, which reproduces MFEM's own re-read refinement
    // bit for bit.
    assert_eq!(g.coords[..48 * 3], want[..48 * 3], "pyramid rows verbatim");

    for (v, want) in R1_VERTICES.iter().enumerate() {
        for c in 0..3 {
            let got = mesh.coords_of(v as u32)[c];
            assert_eq!(got, want[c], "vertex {v} comp {c}");
        }
    }
}
