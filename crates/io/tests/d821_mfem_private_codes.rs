//! D819-C — the private MFEM `.mesh` element codes are gone from both sides
//! of the io reader/writer.
//!
//! Round-83 Lane B registered the defect: `mfem_elem_type` (read) and the
//! `elem_type_to_mfem_code` reverse table (write) carried the private codes
//! 8-14 (Line3=8, Tri6=9, Quad8=10, Tet10=11, Hex20=12, Prism15=13,
//! Pyramid13=14).  The MFEM `.mesh` format defines geometry codes **0-7**
//! only (`Geometry::Type`, geom.hpp:39-44); MFEM 4.10 aborts on every one of
//! 8..=14 with "invalid Geometry::Type, geom = N" (`Mesh::
//! ReadElementWithoutAttr`, mesh.cpp:5002-5010) — probe
//! `tmp/d84b/probe_reject_results.txt`, both the `elements` and the
//! `boundary` section, with 1..=7/0..=1 accepted by the same harness.  So the
//! read side accepted files real MFEM refuses, and the write side emitted
//! them: **a fem-rs `.mesh` with a high-order code was not portable.**
//!
//! The fix follows MFEM's own semantics:
//!
//! * read: codes outside 0-7 are refused loudly, naming the code and the
//!   reason (this file, `d821_read_refuses_private_codes`);
//! * write: the strict table drives the `elements`/`boundary` sections; a
//!   Quad9 *cell* (Gmsh type-10 import, the tensor-Q2 rows) is written as
//!   MFEM's own curved-mesh encoding — SQUARE rows + a synthesised 9-dof
//!   `L2_T1_2D_P2` `nodes` section — probe-verified TOPOLOGY-IDENTICAL
//!   (`tmp/d84b/probe_quad9.cpp`, `d821_quad9_exports_topology_identical`);
//! * every other high-order cell type refuses loudly instead of emitting a
//!   private code (`d821_write_refuses_undereivable_high_order_cells`).

use fem_io::mfem::{read_mfem, write_mfem_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

// ─── fixtures ───────────────────────────────────────────────────────────────

/// A minimal straight 2-D mesh text whose single element row carries `code`.
fn mesh_with_elem_code(code: u32) -> String {
    format!(
        "MFEM mesh v1.0\n\ndimension\n2\n\nelements\n1\n1 {code} 0 1 2 3\n\
         \nboundary\n0\n\nvertices\n4\n2\n0 0\n1 0\n1 1\n0 0\n"
    )
}

/// A minimal straight 2-D mesh text whose single boundary row carries `code`.
fn mesh_with_bdr_code(code: u32) -> String {
    format!(
        "MFEM mesh v1.0\n\ndimension\n2\n\nelements\n1\n1 3 0 1 2 3\n\
         \nboundary\n1\n1 {code} 0 1 2 3\n\nvertices\n4\n2\n0 0\n1 0\n1 1\n0 0\n"
    )
}

/// The curved Quad9 of `tmp/d84b/quad9_l2.mesh`: one unit-square cell scaled
/// 2x with a lifted top edge, in the fem-rs Quad9 slot order (corners CCW,
/// then the bottom/right/top/left edge midsides, then the centre).
fn quad9_mesh() -> Mesh<2> {
    let coords: Vec<f64> = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.25),
        (1.0, 1.25),
        (0.0, 0.5),
        (1.0, 0.625),
    ]
    .iter()
    .flat_map(|(x, y)| [*x, *y])
    .collect();
    Mesh::uniform(
        coords,
        (0..9).collect(),
        vec![1],
        ElementType::Quad9,
        vec![0, 1, 4],
        vec![7],
        ElementType::Line3,
    )
}

// ─── read side ──────────────────────────────────────────────────────────────

/// Every private code is refused on read, from both sections, with a message
/// that names the code and the 0-7 rule.  (Pre-D819-C all of these parsed
/// into `Line3`/`Tri6`/`Quad8`/`Tet10`/`Hex20`/`Prism15`/`Pyramid13`
/// elements.)
#[test]
fn d821_read_refuses_private_codes() {
    for code in [8u32, 9, 10, 11, 12, 13, 14, 15, 200] {
        let err = match read_mfem(std::io::Cursor::new(mesh_with_elem_code(code))) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("elements-section private code {code} must be refused"),
        };
        assert!(
            err.contains(&format!("type {code}")) && err.contains("0-7"),
            "elements code {code}: message must name the code and the 0-7 rule, got: {err}"
        );
        // The boundary section goes through the same table; probe one private
        // code plus a legal control (a SEGMENT boundary row is valid in 2-D).
        if matches!(code, 10 | 14) {
            let err = match read_mfem(std::io::Cursor::new(mesh_with_bdr_code(code))) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("boundary-section private code {code} must be refused"),
            };
            assert!(
                err.contains(&format!("type {code}")),
                "boundary code {code}: {err}"
            );
        }
    }
    // Legal controls: the same harness accepts codes 2 and 3 (each row
    // carrying its own vertex count).
    read_mfem(std::io::Cursor::new(
        "MFEM mesh v1.0\n\ndimension\n2\n\nelements\n1\n1 2 0 1 2\n\
         \nboundary\n0\n\nvertices\n4\n2\n0 0\n1 0\n1 1\n0 0\n",
    ))
    .unwrap_or_else(|e| panic!("control code 2 must read: {e}"));
    read_mfem(std::io::Cursor::new(mesh_with_elem_code(3)))
        .unwrap_or_else(|e| panic!("control code 3 must read: {e}"));
}

// ─── write side: refusals ───────────────────────────────────────────────────

/// A mesh whose cells have no derived MFEM encoding is refused loudly — no
/// private code file may be emitted (pre-D819-C, a Quad8 cell wrote
/// `1 10 <8 ids>` and a Hex20 cell `1 12 <20 ids>`; MFEM aborts on both).
///
/// (Round 84's Tri6 refusal case moved to the green side with D821-1: the
/// row-geometry cells Line3/Tri6/Tet10/Hex27/Prism18 now export as their base
/// code plus the synthesised `L2_T1` `nodes` section — see the d825 tests.
/// Round 85 moved Pyramid13 to the green side the same way with D825-2: the
/// PYRAMID base row plus the 27-dof `L2_T1_3D_P2` Fuentes container
/// (`d825_row_geometry_h1::d825_pyramid13_exports_fuentes_l2`, pinned against
/// MFEM's own `SetCurvature` oracle), so the refusal here shrinks to the
/// dof-count-gap families Quad8/Hex20/Prism15.)
#[test]
fn d821_write_refuses_underivable_high_order_cells() {
    // Quad8 is a 2-D cell; Hex20/Prism15 are 3-D cells.  (Pyramid13 was
    // refused here until D825-2 derived its Fuentes export.)
    for (name, et, npe, dim3) in [
        ("Quad8", ElementType::Quad8, 8usize, false),
        ("Hex20", ElementType::Hex20, 20, true),
        ("Prism15", ElementType::Prism15, 15, true),
    ] {
        let err = if dim3 {
            let mesh3 = Mesh::<3>::uniform(
                vec![0.0; npe * 3],
                (0..npe as u32).collect(),
                vec![1],
                et,
                vec![],
                vec![],
                ElementType::Quad4,
            );
            let scratch = Mesh::<2>::unit_square_tri(1);
            write_mfem_nodes(&mut Vec::new(), &scratch, Some(&mesh3), NodesSpace::Discontinuous)
                .expect_err("a private code must not be emitted")
                .to_string()
        } else {
            let mesh = Mesh::<2>::uniform(
                vec![0.0; npe * 2],
                (0..npe as u32).collect(),
                vec![1],
                et,
                vec![],
                vec![],
                ElementType::Line2,
            );
            write_mfem_nodes(&mut Vec::new(), &mesh, None, NodesSpace::Discontinuous)
                .expect_err("a private code must not be emitted")
                .to_string()
        };
        assert!(
            err.contains(name) && err.contains("0-7"),
            "{name}: refusal must name the type and the format rule, got: {err}"
        );
    }
}

// ─── write side: the derived Quad9 encoding ─────────────────────────────────

/// The Quad9 cell exports as MFEM's own curved-mesh encoding: SQUARE element
/// rows (corner prefix) plus the synthesised 9-dof `L2_T1_2D_P2` section —
/// byte-for-byte the layout probe-verified against MFEM 4.10
/// (`tmp/d84b/probe_quad9.cpp` on `tmp/d84b/quad9_l2.mesh`).
///
/// The written bytes can be re-verified end-to-end against the MFEM probe:
/// `D821_DUMP=<path> cargo test -p fem-io --test d821_mfem_private_codes`
/// copies the file there (the probe lives in `tmp/d84b/probe_quad9.cpp`).
#[test]
fn d821_quad9_exports_topology_identical() {
    let mesh = quad9_mesh();
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &mesh, None, NodesSpace::Discontinuous)
        .expect("the Quad9 mesh must export");

    if let Ok(path) = std::env::var("D821_DUMP") {
        std::fs::write(&path, &bytes).expect("dump written file");
    }

    let text = String::from_utf8(bytes.clone()).unwrap();
    // Element rows: the base SQUARE code and the corner prefix only.
    assert!(
        text.contains("elements\n1\n1 3 0 1 2 3\n"),
        "element row must be `1 3 0 1 2 3`, got:\n{text}"
    );
    // The nodes section is the exact per-element Q2 table.
    assert!(
        text.contains("FiniteElementCollection: L2_T1_2D_P2"),
        "nodes collection must be L2_T1_2D_P2, got:\n{text}"
    );
    // The permuted dof values (lex order), one dof per line, `x y`:
    // fem-rs slot f sits at lex dof [0, 2, 8, 6, 1, 5, 7, 3, 4][f].
    let want_lines = [
        "0 0",     // lex 0 = v0
        "1 0",     // lex 1 = e01
        "2 0",     // lex 2 = v1
        "0 0.5",   // lex 3 = e30
        "1 0.625", // lex 4 = centre
        "2 1.25",  // lex 5 = e12
        "0 1",     // lex 6 = v3
        "1 1.25",  // lex 7 = e23
        "2 1",     // lex 8 = v2
    ];
    let values_start = text
        .find("Ordering: 1\n")
        .expect("nodes header") + "Ordering: 1\n".len();
    let got_lines: Vec<&str> = text[values_start..]
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(got_lines.len(), 9, "nine dof lines, got:\n{text}");
    for (i, (got, want)) in got_lines.iter().zip(want_lines).enumerate() {
        let g: Vec<f64> = got
            .split_whitespace()
            .map(|v| v.parse::<f64>().unwrap())
            .collect();
        let w: Vec<f64> = want
            .split_whitespace()
            .map(|v| v.parse::<f64>().unwrap())
            .collect();
        assert!(
            g.len() == 2 && (g[0] - w[0]).abs() < 1e-13 && (g[1] - w[1]).abs() < 1e-13,
            "dof {i}: got {got:?}, want {want:?}"
        );
    }
    // The boundary edge is a SEGMENT row with the endpoint prefix (the Line3
    // midside lives in the volume's nodes section).
    assert!(
        text.contains("boundary\n1\n7 1 0 1\n"),
        "boundary row must be `7 1 0 1`, got:\n{text}"
    );

    // Read back: base Quad4 topology plus the order-2 L2 geometry table whose
    // per-element coordinates are the mesh slot-order originals bit for bit
    // (the reader's D153 perm is the inverse of the writer's).
    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m2 = back.mesh2d.expect("2-D container");
    assert_eq!(m2.elem_type, ElementType::Quad4, "base cell type");
    let geo = m2.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2, "geometry order");
    assert_eq!(geo.nodes_per_elem, 9, "nodes per element");
    let original: Vec<f64> = quad9_mesh().coords.clone();
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates must round-trip bit for bit"
    );
    // The folded vertex table is MFEM's own recovery: the four corner means.
    assert_eq!(
        &m2.coords[..8],
        &[0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0],
        "vertices recovered to the corners"
    );
}

/// A Quad9 mesh under the *continuous* space request exports the derived
/// `H1_2D_P2` numbering (D821-2 — this case refused before the derivation;
/// the refusal premise is retired by it).  For the single-cell fixture with
/// corner ids 0..4 the H1 dof order `[vertices | edges | interior]` equals the
/// fem-rs Quad9 slot order, so the nine dof rows are the node coordinates in
/// slot order; MFEM probe `tmp/d84fixB/probe_high_order.cpp` (families
/// `quad9h1`/`quad9h1b`) verifies the read-back numbering, the shared-edge dof
/// of two adjacent cells and the Save/Load round-trip.
#[test]
fn d821_quad9_continuous_space_exports() {
    let mesh = quad9_mesh();
    let mut bytes = Vec::new();
    write_mfem_nodes(&mut bytes, &mesh, None, NodesSpace::Continuous)
        .expect("the derived continuous Quad9 numbering exports");

    if let Ok(path) = std::env::var("D821_DUMP") {
        std::fs::write(&path, &bytes).expect("dump written file");
    }

    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(
        text.contains("elements\n1\n1 3 0 1 2 3\n"),
        "element row must be the SQUARE corner prefix, got:\n{text}"
    );
    assert!(
        text.contains("FiniteElementCollection: H1_2D_P2"),
        "nodes collection must be H1_2D_P2, got:\n{text}"
    );
    // The nine dof lines in H1 order = slot order for this fixture.
    let want_lines = [
        "0 0",     // dof 0 = vertex 0
        "2 0",     // dof 1 = vertex 1
        "2 1",     // dof 2 = vertex 2
        "0 1",     // dof 3 = vertex 3
        "1 0",     // dof 4 = edge (0,1)
        "2 1.25",  // dof 5 = edge (1,2)
        "1 1.25",  // dof 6 = edge (2,3)
        "0 0.5",   // dof 7 = edge (3,0)
        "1 0.625", // dof 8 = cell interior
    ];
    let values_start = text
        .find("Ordering: 1\n")
        .expect("nodes header") + "Ordering: 1\n".len();
    let got_lines: Vec<&str> = text[values_start..]
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(got_lines.len(), 9, "nine dof lines, got:\n{text}");
    for (i, (got, want)) in got_lines.iter().zip(want_lines).enumerate() {
        let g: Vec<f64> = got
            .split_whitespace()
            .map(|v| v.parse::<f64>().unwrap())
            .collect();
        let w: Vec<f64> = want
            .split_whitespace()
            .map(|v| v.parse::<f64>().unwrap())
            .collect();
        assert!(
            g.len() == 2 && (g[0] - w[0]).abs() < 1e-13 && (g[1] - w[1]).abs() < 1e-13,
            "dof {i}: got {got:?}, want {want:?}"
        );
    }

    // Read back: the same base topology plus an order-2 H1 geometry table
    // whose dof coordinates are the original node coordinates bit for bit.
    let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
    let m2 = back.mesh2d.expect("2-D container");
    assert_eq!(m2.elem_type, ElementType::Quad4, "base cell type");
    let geo = m2.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.order, 2, "geometry order");
    assert_eq!(geo.nodes_per_elem, 9, "nodes per element");
    let original: Vec<f64> = quad9_mesh().coords.clone();
    assert_eq!(
        geo.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "geometry coordinates must round-trip bit for bit"
    );
}
