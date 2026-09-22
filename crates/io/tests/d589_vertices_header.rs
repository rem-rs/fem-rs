//! D589 — the MFEM `vertices` header must parse with MFEM 4.10's token
//! semantics, and `data/d394_prism_stack.mesh` must read to the mesh MFEM
//! itself reads.
//!
//! MFEM truth (serial 4.10 tree, `mesh/mesh_readers.cpp:100-118`,
//! `Mesh::ReadMFEMMesh`):
//!
//! ```text
//! input >> ident;             // 'vertices'
//! input >> NumOfVertices;
//! input >> ws >> ident;       // the NEXT *token* (not line!) decides:
//! ident != "nodes" → spaceDim = atoi(ident); read NV*spaceDim coord tokens
//! ident == "nodes" → curved = 1; a nodes GridFunction section follows
//! ```
//!
//! So the "vertices 9 3"-style header is the canonical STRAIGHT mesh form —
//! MFEM's own writer emits `<NV>` and `<spaceDim>` (on separate lines,
//! `Mesh::Printer`, `mesh/mesh.cpp:12560-12572`, cf. `data/star.mesh`:
//! `vertices / 31 / 2`), and the curved form is `<NV>` followed by a `nodes`
//! token (mesh.cpp:12573-12578).  Nothing about "9 3" ever selects the curved
//! reading; that round-58 hypothesis was disproven by probe
//! (`tmp/d589/probe1.cpp`, compiled against `~/mfem410_ser`):
//!
//! * the original `d394_prism_stack.mesh` (1-based element indices!) loaded as
//!   a straight mesh with `Nodes == NULL`, but `RemoveUnusedVertices`
//!   (mesh_readers.cpp:38/130, default on) dropped the now-unreferenced
//!   vertex 0 → `NV=9→8`, the table shifted, and element 2's out-of-range
//!   index `9` wrapped onto 0 → "Elements with wrong orientation: 1 / 2" and
//!   FE nodes mapped outside the element (z = 1.65 > 1);
//! * the 0-based re-indexed twin loads perfectly: `NV=9`, both prisms intact,
//!   no orientation complaints, point matrices exactly the intended table.
//!
//! That is why the fixture was normalized to 0-based indices (fem-rs reads it
//! bit-identically to before: the old reader's 1-based repair heuristic
//! subtracted 1, the new verbatim 0-based indices need no repair) — C++ and
//! Rust probes can now exchange this file directly.
//!
//! Deliberate residual divergences (documented, not fixed — D600/D601):
//! * fem-rs keeps slack (unreferenced) vertices where MFEM drops + renumbers;
//! * fem-rs tolerates 1-based index files that MFEM reads verbatim (garbling).

use std::io::Cursor;

use fem_io::mfem::read_mfem;

const ONE_TRI_0BASED: &str = "\
MFEM mesh v1.0

dimension
2

elements
1
1 2 0 1 2

boundary
1
1 1 0 1

vertices
3 2
0.0 0.0
1.0 0.0
0.0 1.0
";

fn tri_mesh_with_header(vertices_header: &str) -> String {
    // Everything up to (excluding) the `vertices` keyword, then the header
    // under test.
    let head = ONE_TRI_0BASED.split("vertices").next().unwrap();
    format!("{head}vertices\n{vertices_header}")
}

#[test]
fn canonical_two_line_nv_sdim_header_parses_straight_like_mfem()
{
    // MFEM `Mesh::Printer` layout (mesh.cpp:12560-12572): NV and spaceDim on
    // separate lines — also the `data/d394_prism_stack.mesh` layout.
    let f = read_mfem(Cursor::new(
        tri_mesh_with_header("3\n2\n0.0 0.0\n1.0 0.0\n0.0 1.0\n").into_bytes(),
    ))
    .expect("canonical two-line header must parse");
    let m = f.mesh2d.as_ref().unwrap();
    assert_eq!(m.coords, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(m.conn, vec![0, 1, 2]);
    assert_eq!(m.face_conn, vec![0, 1]);
}

#[test]
fn one_line_nv_sdim_header_parses_straight_like_mfem()
{
    // Token rule (mesh_readers.cpp:104-108): "3 2" on one line is the same
    // token stream as "3\n2".
    let f = read_mfem(Cursor::new(
        tri_mesh_with_header("3 2\n0.0 0.0\n1.0 0.0\n0.0 1.0\n").into_bytes(),
    ))
    .expect("single-line 'NV SDIM' header must parse (MFEM token rule)");
    let m = f.mesh2d.as_ref().unwrap();
    assert_eq!(m.coords, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn keyword_line_with_trailing_tokens_parses_straight_like_mfem()
{
    // "vertices 3 2" on one line — still the same token stream to MFEM.
    let text = tri_mesh_with_header("3 2\n0.0 0.0\n1.0 0.0\n0.0 1.0\n")
        .replace("vertices\n3 2\n", "vertices 3 2\n");
    let f = read_mfem(Cursor::new(text.into_bytes()))
        .expect("'vertices NV SDIM' on one line must parse (MFEM token rule)");
    let m = f.mesh2d.as_ref().unwrap();
    assert_eq!(m.coords, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn vertices_then_nodes_token_selects_curved_gridfunction_section()
{
    // mesh_readers.cpp:108-124: ident == "nodes" → curved = 1; the section is
    // the nodes GridFunction and (mesh.cpp:5291-5300) the vertex coordinates
    // come from its first NV dofs (`SetVerticesFromNodes`).  "3 nodes" on the
    // NV line exercises the carried-token path.
    let text = "\
MFEM mesh v1.0

dimension
2

elements
1
1 2 0 1 2

boundary
1
1 1 0 1

vertices
3 nodes
FiniteElementSpace
FiniteElementCollection: H1_2D_P1
VDim: 2
Ordering: 1
0.25 0.25
0.75 0.125
0.5 0.75
";
    let f = read_mfem(Cursor::new(text.as_bytes().to_vec()))
        .expect("'NV nodes' header must route into the nodes (curved) section");
    let m = f.mesh2d.as_ref().unwrap();
    assert_eq!(m.coords, vec![0.25, 0.25, 0.75, 0.125, 0.5, 0.75]);
}

#[test]
fn legacy_header_without_sdim_token_is_rejected()
{
    // Quirk pin: a pre-`spaceDim` file ("vertices" then NV then coordinates)
    // makes MFEM 4.10 treat the FIRST COORDINATE as spaceDim (atoi("0") == 0
    // here → zero coordinates consumed, uninitialized vertex table, silent
    // garbage).  fem_io refuses loudly instead — the formats disagree on this
    // input by design, and the MFEM behavior is not worth reproducing.
    let f = read_mfem(Cursor::new(
        tri_mesh_with_header("3\n0.0 0.0\n1.0 0.0\n0.0 1.0\n").into_bytes(),
    ));
    assert!(f.is_err(), "no-SDIM header must not silently parse");
}

#[test]
fn d394_fixture_reads_as_the_mfem_410_probe_golden()
{
    // Golden from `tmp/d589/probe1.cpp` (WSL, `g++ -std=c++17 -O2 -I
    // ~/mfem410_ser probe1.cpp ~/mfem410_ser/libmfem.a`) run on the 0-based
    // `d394_zb.mesh` (byte-identical to the normalized
    // `data/d394_prism_stack.mesh`): NV=9, NE=2, NBE=8, spaceDim=3,
    // Nodes == NULL (straight), no wrong-orientation complaints, element
    // point matrices equal to the table below.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/d394_prism_stack.mesh");
    let f = fem_io::mfem::read_mfem_file(path).expect("normalized d394 fixture must parse");
    let m = f.mesh3d.as_ref().expect("d394 fixture is a 3-D mesh");

    assert_eq!(m.coords.len(), 27, "NV=9 x spaceDim=3 (MFEM probe: NV=9)");
    assert_eq!(&m.coords[..3], &[0.0, 0.0, 0.0], "vertex 0 at the origin");
    assert_eq!(&m.coords[24..], &[0.0, 1.0, 2.0], "vertex 8 = (0,1,2)");
    assert_eq!(
        m.conn,
        vec![0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8],
        "two stacked prisms sharing tri face {{3,4,5}} (probe ELEMENTS block)"
    );
    assert_eq!(m.face_tags.len(), 8, "probe NBE=8");
    assert_eq!(&m.face_conn[..3], &[0, 2, 1], "first boundary tri (probe)");
}

#[test]
fn one_based_index_files_are_a_documented_femrs_extension()
{
    // The PRE-normalization d394 content (1-based element indices).  MFEM
    // 4.10 reads indices verbatim: vertex 0 goes unreferenced,
    // `RemoveUnusedVertices` drops it (NV 9→8, table shifts) and element 2's
    // index 9 is out of range (probe: "Elements with wrong orientation:
    // 1 / 2", NV=8, e1 = "3 4 5 6 7 0").  fem-rs instead repairs 1-based
    // files (mfem.rs "Detect 0-based vs 1-based" heuristic — D601 divergence)
    // and recovers the intended mesh.  This test pins the repair so it cannot
    // change silently; normalize fixtures to 0-based instead of relying on it.
    let text = "\
MFEM mesh v1.0

dimension
3

elements
2
1 6 1 2 3 4 5 6
1 6 4 5 6 7 8 9

boundary
8
1 2 1 3 2
1 3 1 2 5 4
1 3 2 3 6 5
1 3 3 1 4 6
1 2 7 8 9
1 3 4 5 8 7
1 3 5 6 9 8
1 3 6 4 7 9

vertices
9
3
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
0.0 0.0 2.0
1.0 0.0 2.0
0.0 1.0 2.0
";
    let f = read_mfem(Cursor::new(text.as_bytes().to_vec()))
        .expect("1-based file is repaired by the documented heuristic");
    let m = f.mesh3d.as_ref().unwrap();
    assert_eq!(m.coords.len(), 27, "fem-rs keeps all 9 declared vertices (MFEM: 8 — D600)");
    assert_eq!(
        m.conn,
        vec![0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8],
        "repaired to the intended stacked prisms"
    );
}
