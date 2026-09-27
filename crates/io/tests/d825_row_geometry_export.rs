//! D821-1 / D821-2 — the round-84 Quad9 export mode ("base code + nodes
//! payload") generalised to the remaining row-geometry cells, pinned per
//! family.
//!
//! A row-geometry cell (Line3/Tri6/Quad9/Tet10/Hex27/Prism18 — the Gmsh
//! second-order imports) has no MFEM element code: its `.mesh` encoding is a
//! base row with the corner prefix plus an order-2 `nodes` section.  Round 84
//! derived that for the Quad9 cell; this round derives it for the other five
//! (discontinuous `L2_T1_*_P2`) and for the Quad9 cell's *continuous*
//! `H1_2D_P2` numbering (D821-2), and aligns the emitted vertex table with
//! MFEM's own post-compaction state (`RemoveUnusedVertices` drops every
//! non-corner node and renumbers, so MFEM's `Mesh::Printer` writes
//! `vertices / <corner count>` and compacted row ids — measured on MFEM 4.10's
//! re-save of the round-84 Quad9 output: `vertices\n4` over a 9-node table).
//!
//! Every family here was verified end-to-end against real MFEM 4.10
//! (`tmp/d84fixB/probe_high_order.cpp`, compiled against `$HOME/mfem410_ser`):
//! the read-back mesh is topology-identical, its `nodes` field equals the
//! original node coordinates point for point, and MFEM's own Save/Load
//! preserves it.  Re-verify after any change:
//!
//! ```text
//! D825_DUMP_DIR=tmp/d84fixB cargo test -p fem-io --test d825_row_geometry_export
//! wsl -e bash -lc 'cd /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/d84fixB && \
//!   for f in line3 tri6 tet10 hex27 prism18 quad9h1b; do \
//!     ./probe_high_order $f rs_$f.mesh; done'
//! ```

use fem_io::mfem::{read_mfem, write_mfem_nodes, write_mfem_nodes_1d, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::{GeometryData, Mesh};

// ─── helpers ────────────────────────────────────────────────────────────────

/// Dump a written file for the WSL probe when `D825_DUMP_DIR` is set.  A
/// `want` payload (the probe's expected dof table) rides along as
/// `rs_<name>.want`.
fn dump(name: &str, bytes: &[u8], want: &[Vec<f64>]) {
    if let Ok(dir) = std::env::var("D825_DUMP_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::write(dir.join(format!("rs_{name}.mesh")), bytes).expect("dump written file");
        let want_text: String = want
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join(format!("rs_{name}.want")), want_text)
            .expect("dump want file");
    }
}

/// The `x y (z)` lines of the `nodes` payload, everything after `Ordering: 1`.
fn dof_lines(text: &str) -> Vec<Vec<f64>> {
    let start = text.find("Ordering: 1\n").expect("nodes header") + "Ordering: 1\n".len();
    text[start..]
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse::<f64>().unwrap()).collect())
        .collect()
}

fn assert_lines(got: &[Vec<f64>], want: &[(f64, f64)], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: dof count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.len() == 2 && (g[0] - w.0).abs() < 1e-13 && (g[1] - w.1).abs() < 1e-13,
            "{what}: dof {i} = {g:?}, want {w:?}"
        );
    }
}

// ─── Line3 (1-D): SEGMENT row + L2_T1_1D_P2 ─────────────────────────────────

/// `L2_T1_1D_P2` enumerates its 3 dofs on the ascending closed lattice
/// [v0 mid v1], while the fem-rs Line3 row is the Gmsh type-8 order
/// [v0 v1 mid]: the file order is `slots [0, 2, 1]`.  The vertices count is
/// the compacted corner set {0, 1} (the midside is a `nodes` dof), and the
/// second boundary POINT row carries the compacted id 1.
#[test]
fn d825_line3_exports_segment_l2() {
    // fem-rs Line3 row order = the Gmsh type-8 order [v0, v1, mid].
    let mesh = Mesh::<1>::uniform(
        vec![0.0, 2.0, 0.75],
        vec![0, 1, 2],
        vec![1],
        ElementType::Line3,
        vec![0, 1],
        vec![1, 2],
        ElementType::Point1,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes_1d(&mut bytes, &mesh, NodesSpace::Discontinuous).expect("export");
    dump("line3", &bytes, &[]);
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(text.contains("dimension\n1\n"), "1-D header:\n{text}");
    assert!(text.contains("elements\n1\n1 1 0 1\n"), "SEGMENT corner row:\n{text}");
    assert!(
        text.contains("boundary\n2\n1 0 0\n2 0 1\n"),
        "boundary POINTs in the compacted numbering:\n{text}"
    );
    assert!(text.contains("vertices\n2\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: L2_T1_1D_P2"),
        "collection:\n{text}"
    );
    // A 1-D payload carries one component per dof line.
    let got = dof_lines(&text);
    assert_eq!(
        got,
        vec![vec![0.0], vec![0.75], vec![2.0]],
        "line3 L2 dofs in the ascending file order"
    );

    // Read back: base SEGMENT topology plus the order-1-stride table whose
    // per-element coordinates are the original rows bit for bit.
    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m1 = back.mesh1d.expect("1-D container");
    assert_eq!(m1.elem_type, ElementType::Line2, "base cell type");
    let geo = m1.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 3);
    // The read-back table is slotted on the ascending lattice [v0 mid v1].
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![0.0f64.to_bits(), 0.75f64.to_bits(), 2.0f64.to_bits()],
        "geometry coordinates round-trip bit for bit"
    );
    assert_eq!(m1.coords, vec![0.0, 2.0], "vertices recovered to the corners");
}

// ─── Tri6 (2-D): TRIANGLE row + L2_T1_2D_P2 ─────────────────────────────────

/// The fem-rs Tri6 slot order [v0 v1 v2 e01 e12 e20] (the H1 triangle order)
/// meets MFEM's L2 triangle order [v0 e01 v1 e20 e12 v2]: the file order is
/// `slots [0, 3, 1, 5, 4, 2]`.
#[test]
fn d825_tri6_exports_triangle_l2() {
    let mesh = Mesh::<2>::uniform(
        // v0 v1 v2 e01 e12 e20
        vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, -0.1, 0.8, 0.7, -0.1, 1.0],
        (0..6).collect(),
        vec![1],
        ElementType::Tri6,
        vec![0, 1, 3],
        vec![1],
        ElementType::Line3,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &mesh, None, NodesSpace::Discontinuous).expect("export");
    dump("tri6", &bytes, &[]);
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(text.contains("elements\n1\n1 2 0 1 2\n"), "TRIANGLE corner row:\n{text}");
    assert!(text.contains("boundary\n1\n1 1 0 1\n"), "SEGMENT corner prefix:\n{text}");
    assert!(text.contains("vertices\n3\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: L2_T1_2D_P2"),
        "collection:\n{text}"
    );
    // File order [v0, e01, v1, e20, e12, v2].
    assert_lines(
        &dof_lines(&text),
        &[
            (0.0, 0.0),    // v0
            (1.0, -0.1),   // e01
            (2.0, 0.0),    // v1
            (-0.1, 1.0),   // e20
            (0.8, 0.7),    // e12
            (0.0, 2.0),    // v2
        ],
        "tri6 L2 dofs",
    );

    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m2 = back.mesh2d.expect("2-D container");
    assert_eq!(m2.elem_type, ElementType::Tri3, "base cell type");
    let geo = m2.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 6);
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        mesh.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates round-trip bit for bit"
    );
    assert_eq!(
        &m2.coords[..6],
        &[0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        "vertices recovered to the corners"
    );
}

// ─── Tet10 (3-D): TETRAHEDRON row + L2_T1_3D_P2 ─────────────────────────────

/// The fem-rs Tet10 slot order [v0 v1 v2 v3 e01 e02 e03 e12 e13 e23] (the H1
/// tetrahedron order) meets MFEM's L2 order — the (k, j, i) barycentric loop
/// [v0 e01 v1 e02 e12 v2 e03 e13 e23 v3]: the file order is
/// `slots [0, 4, 1, 5, 7, 2, 6, 8, 9, 3]`.
#[test]
fn d825_tet10_exports_tetrahedron_l2() {
    let mesh = Mesh::<3>::uniform(
        // v0 v1 v2 v3 e01 e02 e03 e12 e13 e23
        vec![
            0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 2.0, //
            1.0, -0.1, 0.0, //
            0.1, 1.0, 0.0, //
            0.1, 0.0, 1.0, //
            1.0, 1.0, -0.05, //
            1.0, 0.05, 1.0, //
            0.05, 1.0, 1.0, //
        ],
        (0..10).collect(),
        vec![1],
        ElementType::Tet10,
        // the z=0 face as a Tri6 row: corners v0 v1 v2, midsides e01 e12 e20
        vec![0, 1, 2, 4, 7, 5],
        vec![1],
        ElementType::Tri6,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Discontinuous).expect("export");
    dump("tet10", &bytes, &[]);
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(
        text.contains("elements\n1\n1 4 0 1 2 3\n"),
        "TETRAHEDRON corner row:\n{text}"
    );
    assert!(
        text.contains("boundary\n1\n1 2 0 1 2\n"),
        "TRIANGLE corner prefix:\n{text}"
    );
    assert!(text.contains("vertices\n4\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: L2_T1_3D_P2"),
        "collection:\n{text}"
    );
    let got = dof_lines(&text);
    let want: Vec<Vec<f64>> = [
        [0.0, 0.0, 0.0],
        [1.0, -0.1, 0.0],
        [2.0, 0.0, 0.0],
        [0.1, 1.0, 0.0],
        [1.0, 1.0, -0.05],
        [0.0, 2.0, 0.0],
        [0.1, 0.0, 1.0],
        [1.0, 0.05, 1.0],
        [0.05, 1.0, 1.0],
        [0.0, 0.0, 2.0],
    ]
    .iter()
    .map(|c| c.to_vec())
    .collect();
    assert_eq!(got, want, "tet10 L2 dofs in MFEM's file order");

    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m3 = back.mesh3d.expect("3-D container");
    assert_eq!(m3.elem_type, ElementType::Tet4, "base cell type");
    let geo = m3.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 10);
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        mesh.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates round-trip bit for bit"
    );
}

// ─── Hex27 (3-D): CUBE row + L2_T1_3D_P2 ────────────────────────────────────

/// The fem-rs Hex27 slot order is MFEM's H1 entity order (8 vertices, 12
/// `CUBE::Edges` midsides, 6 `FaceVert` face centres, the cell centre); MFEM's
/// `L2_T1_3D_P2` file order is the tensor-lex one (ix fastest).  Three dofs
/// are perturbed off the 0/1/2 lattice so any ordering slip shows: lex 4
/// (the z=0 face centre), lex 13 (cell centre), lex 26 (corner v6).
#[test]
fn d825_hex27_exports_cube_l2() {
    // Build the node table from the H1 slot layout: slots 0-7 vertices,
    // 8-19 edges in CUBE::Edges order, 20-25 faces in FaceVert order, 26
    // centre — with the same three perturbations the probe expects.
    let mut c: Vec<Vec<f64>> = Vec::new();
    let corners: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.3], // v6 lifted (lex 26)
        [0.0, 2.0, 2.0],
    ];
    c.extend(corners.iter().map(|v| v.to_vec()));
    const EDGES: [[usize; 2]; 12] = [
        [0, 1], [1, 2], [3, 2], [0, 3], [4, 5], [5, 6], [7, 6], [4, 7], [0, 4], [1, 5], [2, 6],
        [3, 7],
    ];
    for &[a, b] in EDGES.iter() {
        let m: Vec<f64> = (0..3).map(|d| (corners[a][d] + corners[b][d]) / 2.0).collect();
        c.push(m);
    }
    c.push(vec![1.0, 1.0, -0.2]); // f0 [3,2,1,0] z=0 face centre (lex 4)
    c.push(vec![1.0, 0.0, 1.0]); // f1 [0,1,5,4] y=0 face centre (lex 10)
    c.push(vec![2.0, 1.0, 1.0]); // f2 [1,2,6,5] x=2 face centre (lex 14)
    c.push(vec![1.0, 2.0, 1.0]); // f3 [2,3,7,6] y=2 face centre (lex 16)
    c.push(vec![0.0, 1.0, 1.0]); // f4 [3,0,4,7] x=0 face centre (lex 12)
    c.push(vec![1.0, 1.0, 2.0]); // f5 [4,5,6,7] z=2 face centre (lex 22)
    c.push(vec![1.1, 0.9, 1.05]); // cell centre (lex 13)
    let coords: Vec<f64> = c.iter().flat_map(|v| v.iter().copied()).collect();

    let mesh = Mesh::<3>::uniform(
        coords,
        (0..27).collect(),
        vec![1],
        ElementType::Hex27,
        // the y=0 face as a Quad9 row: corners v0 v1 v5 v4, the edge
        // midsides e(0,1) e(1,5) e(5,4) e(4,0), the face centre
        vec![0, 1, 5, 4, 8, 17, 12, 16, 21],
        vec![1],
        ElementType::Quad9,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Discontinuous).expect("export");
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(
        text.contains("elements\n1\n1 5 0 1 2 3 4 5 6 7\n"),
        "CUBE corner row:\n{text}"
    );
    assert!(
        text.contains("boundary\n1\n1 3 0 1 5 4\n"),
        "SQUARE corner prefix:\n{text}"
    );
    assert!(text.contains("vertices\n8\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: L2_T1_3D_P2"),
        "collection:\n{text}"
    );
    // The expected file table, built by arranging the mesh's own node table
    // into MFEM's lex order through the hand-derived slot→lex map for the H1
    // hex layout: vertices {0→0, 1→2, 2→8, 3→6, 4→18, 5→20, 6→26, 7→24},
    // the CUBE::Edges midsides {1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15},
    // the FaceVert face centres {4, 10, 14, 16, 12, 22}, the centre 13.
    // (Probing the lex target of every slot independently pins the whole
    // permutation — the probe's lattice table does the same job end-to-end.)
    const SLOT_TO_LEX: [usize; 27] = [
        0, 2, 8, 6, 18, 20, 26, 24, // vertices v0..v7
        1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, // CUBE::Edges midsides
        4, 10, 14, 16, 12, 22, // FaceVert face centres
        13, // cell centre
    ];
    let mut want: Vec<Vec<f64>> = vec![vec![0.0; 3]; 27];
    for (slot, lex) in SLOT_TO_LEX.iter().enumerate() {
        want[*lex] = c[slot].clone();
    }
    assert_eq!(dof_lines(&text), want, "hex27 L2 dofs in tensor-lex file order");
    dump("hex27", &bytes, &want);

    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m3 = back.mesh3d.expect("3-D container");
    assert_eq!(m3.elem_type, ElementType::Hex8, "base cell type");
    let geo = m3.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 27);
    let flat: Vec<f64> = c.iter().flat_map(|v| v.iter().copied()).collect();
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        flat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates round-trip bit for bit"
    );
}

// ─── Prism18 (3-D): PRISM row + L2_T1_3D_P2 ─────────────────────────────────

/// Both sides are layer-major over the ascending segment GLL points; they
/// differ in the *inner* triangle order — the mesh rows carry the H1 triangle
/// order [v0 v1 v2 e01 e12 e20] per layer, the file the L2 triangle order
/// [v0 e01 v1 e20 e12 v2]: the file order is `k*6 + [0, 3, 1, 5, 4, 2][h]`.
#[test]
fn d825_prism18_exports_prism_l2() {
    // Cross-section: v0(0,0) v1(2,0) v2(2,2), midsides e01(1,0) e12(2,1)
    // e20(1,1).  Layers z = 0, 0.6 (lifted), 2.
    let sect: [(f64, f64); 6] = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 0.0), (2.0, 1.0), (1.0, 1.0)];
    let z = [0.0f64, 0.6, 2.0];
    let mut coords: Vec<f64> = Vec::with_capacity(18 * 3);
    for &zk in z.iter() {
        for &(x, y) in sect.iter() {
            coords.extend_from_slice(&[x, y, zk]);
        }
    }
    let mesh = Mesh::<3>::uniform(
        coords,
        (0..18).collect(),
        vec![1],
        ElementType::Prism18,
        // the bottom triangle face (v0 v1 v2 e01 e12 e20), corner prefix only
        vec![0, 1, 2, 3, 4, 5],
        vec![1],
        ElementType::Tri6,
    );
    let mut bytes = Vec::new();
    let scratch = Mesh::<2>::unit_square_tri(1);
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Discontinuous).expect("export");
    dump("prism18", &bytes, &[]);
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(
        text.contains("elements\n1\n1 6 0 1 2 3 4 5\n"),
        "PRISM corner row:\n{text}"
    );
    assert!(
        text.contains("boundary\n1\n1 2 0 1 2\n"),
        "TRIANGLE corner prefix:\n{text}"
    );
    assert!(text.contains("vertices\n6\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: L2_T1_3D_P2"),
        "collection:\n{text}"
    );
    // File order: layer k, L2 triangle node l — the probe's table.
    let file_sect: [(f64, f64); 6] =
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)];
    let got = dof_lines(&text);
    let mut want: Vec<Vec<f64>> = Vec::with_capacity(18);
    for &zk in z.iter() {
        for &(x, y) in file_sect.iter() {
            want.push(vec![x, y, zk]);
        }
    }
    assert_eq!(got, want, "prism18 L2 dofs in the wedge file order");

    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m3 = back.mesh3d.expect("3-D container");
    assert_eq!(m3.elem_type, ElementType::Prism6, "base cell type");
    let geo = m3.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 18);
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        mesh.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates round-trip bit for bit"
    );
}

// ─── D821-2: the Quad9 cell's continuous H1_2D_P2 numbering ─────────────────

/// Two Quad9 cells sharing the edge (1,2): the H1 space numbers
/// `[vertices | edges | cell interiors]` = 6 + 7 + 2 = 15 dofs, the shared
/// edge carrying ONE dof.  This pins the dof map the probe verified with
/// `GetElementDofs` (elem0 `[0 1 2 3 | 6 7 8 9 | 13]`, elem1
/// `[1 4 5 2 | 10 11 12 7 | 14]`): corners in ascending id order, edges in
/// element-wise `SQUARE::Edges` first-encounter order, interiors private.
#[test]
fn d825_quad9_h1_shared_edge_exports() {
    // Node table (Quad9 slot order [v0 v1 v2 v3 e01 e12 e23 e30 centre]):
    //   e0: 0 1 2 3 | 6 7 8 9 | 10     e1: 1 4 5 2 | 11 12 13 7 | 14
    let coords: Vec<f64> = [
        (0.0, 0.0),   // 0  v0
        (2.0, 0.0),   // 1  v1
        (2.0, 2.0),   // 2  v2
        (0.0, 2.0),   // 3  v3
        (4.0, 0.0),   // 4  v4
        (4.0, 2.0),   // 5  v5
        (1.0, -0.1),  // 6  e(0,1)
        (2.1, 1.0),   // 7  e(1,2) — the shared midside
        (1.0, 2.1),   // 8  e(2,3)
        (-0.1, 1.0),  // 9  e(3,0)
        (1.0, 1.0),   // 10 centre e0
        (3.0, 0.1),   // 11 e(1,4)
        (4.1, 1.0),   // 12 e(4,5)
        (3.0, 2.1),   // 13 e(5,2)
        (3.0, 1.0),   // 14 centre e1
    ]
    .iter()
    .flat_map(|(x, y)| [*x, *y])
    .collect();
    let mesh = Mesh::<2>::uniform(
        coords,
        vec![0, 1, 2, 3, 6, 7, 8, 9, 10, 1, 4, 5, 2, 11, 12, 13, 7, 14],
        vec![1, 1],
        ElementType::Quad9,
        vec![0, 1, 6, 1, 4, 11],
        vec![1, 2],
        ElementType::Line3,
    );
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &mesh, None, NodesSpace::Continuous).expect("export");
    dump("quad9h1b", &bytes, &[]);
    let text = String::from_utf8(bytes.clone()).unwrap();

    assert!(
        text.contains("elements\n2\n1 3 0 1 2 3\n1 3 1 4 5 2\n"),
        "two SQUARE corner rows:\n{text}"
    );
    assert!(text.contains("vertices\n6\n"), "compacted corner count:\n{text}");
    assert!(
        text.contains("FiniteElementCollection: H1_2D_P2"),
        "collection:\n{text}"
    );
    let got = dof_lines(&text);
    let wx = [0.0, 2.0, 2.0, 0.0, 4.0, 4.0, 1.0, 2.1, 1.0, -0.1, 3.0, 4.1, 3.0, 1.0, 3.0];
    let wy = [0.0, 0.0, 2.0, 2.0, 0.0, 2.0, -0.1, 1.0, 2.1, 1.0, 0.1, 1.0, 2.1, 1.0, 1.0];
    assert_eq!(got.len(), 15, "15 H1 dofs, got:\n{text}");
    for (d, g) in got.iter().enumerate() {
        assert!(
            g.len() == 2 && (g[0] - wx[d]).abs() < 1e-13 && (g[1] - wy[d]).abs() < 1e-13,
            "dof {d} = {g:?}, want ({}, {})",
            wx[d],
            wy[d]
        );
    }

    // Read back: base Quad4 topology plus the order-2 H1 geometry table, with
    // the shared edge dof stored once.
    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m2 = back.mesh2d.expect("2-D container");
    assert_eq!(m2.elem_type, ElementType::Quad4, "base cell type");
    assert_eq!(m2.n_elems(), 2, "two cells");
    let geo = m2.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2);
    assert_eq!(geo.nodes_per_elem, 9);
    assert_eq!(geo.n_nodes, 15, "the shared edge dof is one dof");
    assert_eq!(m2.n_nodes(), 6, "vertices recovered to the corners");
}

// ─── the table path (按表): an attached geometry table wins ─────────────────

/// A Quad9 mesh carrying an attached geometry table (what the Gmsh reader
/// attaches as a clone of the rows) exports through the *table* — same values,
/// same bytes as the row-synthesised export when the table clones the rows.
#[test]
fn d825_row_geometry_table_path_matches_rows() {
    let mesh = Mesh::<2>::uniform(
        vec![0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.25, 1.0, 1.25, 0.0, 0.5, 1.0, 0.625],
        (0..9).collect(),
        vec![1],
        ElementType::Quad9,
        vec![0, 1, 4],
        vec![7],
        ElementType::Line3,
    );
    let mut bytes_rows = Vec::new();
    write_mfem_nodes(&mut bytes_rows, &mesh, None, NodesSpace::Discontinuous).expect("row export");

    // The reader's clone-of-the-rows table: identity conn over the same coords.
    let mut with_table = mesh.clone();
    let n = mesh.coords.len() / 2;
    with_table.geometry = Some(GeometryData {
        order: 2,
        conn: (0..n as u32).collect(),
        nodes_per_elem: 9,
        coords: mesh.coords.clone(),
        n_nodes: n,
    });
    let mut bytes_table = Vec::new();
    write_mfem_nodes(&mut bytes_table, &with_table, None, NodesSpace::Discontinuous)
        .expect("table export");
    assert_eq!(
        bytes_rows, bytes_table,
        "the attached clone table and the element rows must produce the same file"
    );

    // And a table that DISAGREES with the rows is not silently ignored: a
    // wrong stride is refused loudly.
    let mut bad = mesh.clone();
    bad.geometry = Some(GeometryData {
        order: 1,
        conn: (0..n as u32).collect(),
        nodes_per_elem: 9,
        coords: mesh.coords.clone(),
        n_nodes: n,
    });
    let err = write_mfem_nodes(&mut Vec::new(), &bad, None, NodesSpace::Discontinuous)
        .expect_err("an inconsistent table must be refused")
        .to_string();
    assert!(
        err.contains("does not match") && err.contains("Quad9"),
        "refusal must name the mismatch, got: {err}"
    );
}

// ─── Pyramid13: the round-85 split (D825-2) ──────────────────────────────────

/// Round 84 refused the 13-node pyramid for both spaces: "a 13-node row
/// cannot fill the 27-dof Fuentes lattice".  **D825-2 superseded the
/// discontinuous arm**: the Fuentes payload is not a permutation of the row
/// but a *synthesis from the five corners* — MFEM's own
/// `SetCurvature(2, true, 3, byVDIM)` on a straight pyramid fills the 27
/// dofs with the straight P1 map at the Fuentes nodal points (probe
/// `tmp/d85b/probe_pyr_gen.txt`, max deviation 5.6e-17), so a straight row
/// now exports (pinned end-to-end in `d825_row_geometry_h1`).  This test
/// keeps the two halves of that split pinned: the discontinuous export goes
/// through, and the continuous space stays refused (MFEM's continuous
/// pyramid container is the 15-dof H1 Fuentes element, probe
/// `tmp/d85b/probe_pyr_gen_h1.txt`; D827-3).
#[test]
fn d825_pyramid13_l2_exported_h1_refused() {
    // All-zero coordinates: every midsides sits at its exact midpoint, i.e.
    // the row is straight and the corner-only synthesis is exact.
    let mesh = Mesh::<3>::uniform(
        vec![0.0; 13 * 3],
        (0..13 as u32).collect(),
        vec![1],
        ElementType::Pyramid13,
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
    );
    let scratch = Mesh::<2>::unit_square_tri(1);
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &scratch, Some(&mesh), NodesSpace::Discontinuous)
        .expect("a straight pyramid exports its 27-dof Fuentes container");
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("elements\n1\n1 7 0 1 2 3 4\n"),
        "PYRAMID corner row:\n{text}"
    );
    assert!(
        text.contains("FiniteElementCollection: L2_T1_3D_P2"),
        "collection:\n{text}"
    );

    let err = write_mfem_nodes(&mut Vec::new(), &scratch, Some(&mesh), NodesSpace::Continuous)
        .expect_err("the continuous pyramid space stays refused")
        .to_string();
    assert!(
        err.contains("Pyramid13") && err.contains("H1_3D_P2") && err.contains("D827-3"),
        "the continuous refusal must name the H1 Fuentes container, got: {err}"
    );
}
