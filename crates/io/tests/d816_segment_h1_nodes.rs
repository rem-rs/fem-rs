//! D816-3 — 1-D *continuous* (`H1_1D_P*`) `nodes`: reading and writing.
//!
//! Round 80 (D813-4) landed the 1-D reader/writer but left the continuous
//! side of the fence open: an `H1_1D_P*` section was read with a warning
//! (vertex coordinates recovered, curved geometry dropped) and the writer
//! refused `Line2` outright.  Both halves are closed now:
//!
//! * the reader attaches the shared-dof geometry table for `H1_1D_P2` —
//!   MFEM's numbering is vertices first (`dof v = v`) plus (p-1)
//!   element-private interior dofs per element, because a 1-D space has no
//!   edge dofs (`fem/fespace.cpp:3458`, probe
//!   `$HOME/work/d81b/h1_1d_probe.cpp`: `GetElementDofs(e)` = `[e, e+1, NV+e]`);
//! * the writer emits the same numbering through `line1d_slot_map`
//!   (`crates/io/src/mfem.rs`), so a curved `H1_1D_P2` 1-D mesh round-trips
//!   **byte for byte** against MFEM 4.10's own
//!   `Mesh::Save(out, 16)` re-save — the round-77/80 oracle protocol;
//! * `H1_1D_P3+` stays refused on both sides with the reason on record: the
//!   collection interpolates on the closed Gauss-Lobatto points, which from
//!   order 3 on are a different point set than the equispaced `SegPk` lattice
//!   the mesh's 1-D geometry is evaluated with (the `L2_T1_1D_P3+`
//!   limitation, D153).  The reader keeps the vertex table (always correct —
//!   `SetVerticesFromNodes` semantics) and reads the mesh straight-sided.
//!
//! Fixtures (all MFEM 4.10 output, `$HOME/work/d81b/h1_1d_probe.cpp`):
//!
//! * `tests/data/d816_mfem_h1_1d_p2.mesh.txt` — `MakeCartesian1D(4)` +
//!   `SetCurvature(2, false, 1, byVDIM)` + a nonlinear projection
//!   `f(x) = x + 0.05 x²` (every dof value distinct, so the file pins the
//!   dof *positions* too), plus two boundary `POINT`s; and its re-load →
//!   `Save(out, 16)` twin `d816_mfem_resave_h1_1d_p2.mesh.txt` (byte
//!   identical — the loader path is a fixed point);
//! * `tests/data/d816_mfem_resave_h1_1d_p2_straight.mesh.txt` — the straight
//!   `MakeCartesian1D(3)` + `SetCurvature(2, false)` twin, the writer-only
//!   oracle (the table is built by hand in the test, proving the writer pulls
//!   values through the table's own dof ids instead of assuming the file's);
//! * `tests/data/d816_mfem_h1_1d_p3.mesh.txt` — the p = 3 refusal fixture;
//! * `tests/fixtures/d816_mfem_h1seg_getelementdofs_p1to3.txt` — the
//!   `GetElementDofs` layout + `H1_SegmentElement` node positions, pinned in
//!   a unit test inside `crates/io/src/mfem.rs`.
//!
//! Set `D80B_DUMP_DIR=<dir>` to drop every written file on disk for an
//! external diff (that is how the byte-level results below were captured).

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes_1d, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::{GeometryData, Mesh};
use fem_mesh::MeshTopology;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// MFEM 4.10's curved `H1_1D_P2` file (`MakeCartesian1D(4)` + the nonlinear
/// projection), included verbatim.
const MFEM_H1_P2: &str = include_str!("data/d816_mfem_h1_1d_p2.mesh.txt");

/// MFEM 4.10's `Save(out, 16)` re-save of the file above (probe reload step).
const MFEM_RESAVE_H1_P2: &str = include_str!("data/d816_mfem_resave_h1_1d_p2.mesh.txt");

/// MFEM 4.10's `Save(out, 16)` of the straight `MakeCartesian1D(3)` +
/// `SetCurvature(2, false)` mesh — the writer-only oracle.
const MFEM_RESAVE_H1_P2_STRAIGHT: &str =
    include_str!("data/d816_mfem_resave_h1_1d_p2_straight.mesh.txt");

/// MFEM 4.10's curved `H1_1D_P3` file — the reader-refusal fixture.
const MFEM_H1_P3: &str = include_str!("data/d816_mfem_h1_1d_p3.mesh.txt");

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// The MFEM reference files are checked out through `core.autocrlf=true`, so
/// comparisons normalise line endings; nothing else is normalised.
fn normalized(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn dump(name: &str, bytes: &[u8]) {
    if let Some(dir) = std::env::var_os("D80B_DUMP_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create D80B_DUMP_DIR");
        std::fs::write(dir.join(name), bytes).expect("dump");
    }
}

/// The `nodes` section's dof values of an MFEM `.mesh` text (VDim: 1,
/// Ordering: 1 — one value per line after the header), for float-exact
/// assertions against the file's own numbers.
fn dof_values(text: &str) -> Vec<f64> {
    let section = text
        .split("Ordering: 1")
        .nth(1)
        .expect("the nodes section header");
    section
        .split('\n')
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect()
}

// ─── the reader side ─────────────────────────────────────────────────────────

/// The read recovers MFEM's own in-memory state: the element table verbatim,
/// the vertex table = the vertex dofs (the `SetVerticesFromNodes` mean of a
/// continuous space is the shared dof value bitwise — the probe's reloaded
/// `GetVertex(v)` equals dof v to the last digit), and the `H1_1D_P2` table
/// as 2 shared vertex dofs + 1 element-private interior dof per element.
#[test]
fn d816_segment_h1_p2_read_recovers_mfems_state() {
    let file = read_mfem(Cursor::new(normalized(MFEM_H1_P2).as_bytes())).expect("read");
    let mesh = file.mesh1d.expect("the 1-D container must be filled");
    assert_eq!(mesh.n_elems(), 4);
    assert_eq!(mesh.n_nodes(), 5);
    assert_eq!(mesh.topological_dim(), 1);
    assert_eq!(mesh.elem_type, ElementType::Line2);

    // Elements verbatim (no 1-D orientation pass on MFEM's loader).
    assert_eq!(mesh.elem_nodes(0), &[0, 1]);
    assert_eq!(mesh.elem_nodes(1), &[1, 2]);
    assert_eq!(mesh.elem_nodes(2), &[2, 3]);
    assert_eq!(mesh.elem_nodes(3), &[3, 4]);

    // Boundary POINTs, attributes 1 and 2 (what MakeCartesian1D writes).
    assert_eq!(mesh.face_type, ElementType::Point1);
    assert_eq!(mesh.face_conn, vec![0, 4]);
    assert_eq!(mesh.face_tags, vec![1, 2]);

    // Vertex table = the vertex dofs of the `nodes` section, bit for bit
    // (f(x) = x + 0.05 x² at x = 0, 1/4, 1/2, 3/4, 1).
    let dof = dof_values(&normalized(MFEM_H1_P2));
    assert_eq!(dof.len(), 9, "5 vertex dofs + 4 element interiors");
    assert_eq!(
        mesh.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dof[..5].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // The continuous geometry table: order 2, 3 dofs per element, 9 in total.
    let g = mesh.geometry.as_ref().expect("the H1 geometry table");
    assert_eq!(g.order, 2);
    assert_eq!(g.nodes_per_elem, 3);
    assert_eq!(g.n_nodes, 9);
    for e in 0..4u32 {
        // [v0, NV + e*(p-1) + j, v1] in the mesh's ascending slot order.
        assert_eq!(
            &g.conn[(e * 3) as usize..(e * 3 + 3) as usize],
            &[e, 5 + e, e + 1],
            "elem {e}: shared endpoints, private interior"
        );
    }
    // …and the table's values are the file's dofs at exactly those ids.
    assert_eq!(
        g.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dof.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

// ─── the byte oracle: read → write == MFEM's own re-save ─────────────────────

#[test]
fn d816_segment_h1_p2_round_trip_matches_mfem_resave() {
    let file = read_mfem(Cursor::new(normalized(MFEM_H1_P2).as_bytes())).expect("read");
    let mesh = file.mesh1d.expect("1-D container");
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous).expect("write_mfem_nodes_1d");
    dump("d816_rs_h1_p2.mesh", &buf);
    assert_eq!(
        normalized(&String::from_utf8(buf).expect("utf-8")),
        normalized(MFEM_RESAVE_H1_P2),
        "the written file differs from MFEM's own re-save \
         (`Mesh::Save(out, 16)` of the same input)"
    );

    // The re-save is a fixed point of the reader, too (MFEM's loader is the
    // one that produced it).
    let back = read_mfem(Cursor::new(normalized(MFEM_RESAVE_H1_P2).as_bytes())).expect("read");
    let mesh2 = back.mesh1d.expect("1-D container again");
    let mut buf2: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf2, &mesh2, NodesSpace::Continuous).expect("write");
    assert_eq!(
        normalized(&String::from_utf8(buf2).expect("utf-8")),
        normalized(MFEM_RESAVE_H1_P2),
        "read → write must be a fixed point of MFEM's own re-save"
    );
}

/// The writer pin that does not go through the reader: the straight
/// `MakeCartesian1D(3)` + `SetCurvature(2, false)` mesh is rebuilt by hand
/// (coords + a `GeometryData` whose *table dof ids are deliberately not the
/// file's* — 20/21/22 for the interior dofs), and the write must still
/// produce MFEM's file byte for byte.  This proves the writer pulls each
/// slot's value through the table's own connectivity and places it at the
/// numbering `line1d_slot_map` derives from the mesh topology.
#[test]
fn d816_segment_h1_p2_straight_writer_matches_mfem_resave() {
    let text = normalized(MFEM_RESAVE_H1_P2_STRAIGHT);
    let dof = dof_values(&text);
    assert_eq!(dof.len(), 7, "4 vertex dofs + 3 element interiors");

    let mesh = Mesh::<1>::uniform(
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        vec![0, 1, 1, 2, 2, 3],
        vec![1; 3],
        ElementType::Line2,
        vec![0, 3],
        vec![1, 2],
        ElementType::Point1,
    );
    // The table MFEM's SetCurvature builds: shared vertex dofs 0..3, private
    // interior dofs (numbered 20..22 here — deliberately *not* the file's dof
    // ids, so the test proves the writer pulls each slot's value through the
    // table's own connectivity instead of assuming the file numbering; the
    // zeros between the blocks would surface immediately if it did).
    //
    // The interior values are the element midpoints computed from the vertex
    // dofs — the same arithmetic MFEM's `SetCurvature` runs, landing on the
    // same bits (their 16-digit renderings are the fixture's lines; note the
    // parsed 16-digit *text* of 1/6 does not round-trip to those bits, so no
    // bit assertion against `dof[4..]` is possible here — the byte comparison
    // below is the check).
    let mut coords = vec![0.0f64; 23];
    coords[0] = 0.0;
    coords[1] = dof[1];
    coords[2] = dof[2];
    coords[3] = 1.0;
    coords[20] = (0.0f64 + dof[1]) / 2.0;
    coords[21] = (dof[1] + dof[2]) / 2.0;
    coords[22] = (dof[2] + 1.0) / 2.0;
    let table = GeometryData {
        order: 2,
        conn: vec![0, 20, 1, 1, 21, 2, 2, 22, 3],
        nodes_per_elem: 3,
        coords,
        n_nodes: 23,
    };

    let mut mesh = mesh;
    mesh.geometry = Some(table);
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous).expect("write_mfem_nodes_1d");
    dump("d816_rs_h1_p2_straight.mesh", &buf);
    assert_eq!(
        normalized(&String::from_utf8(buf).expect("utf-8")),
        text,
        "the written file differs from MFEM's `Save(out, 16)` of the same mesh"
    );
}

// ─── teeth ───────────────────────────────────────────────────────────────────

/// `H1_1D_P3+` stays refused at *read* time: the collection's node lattice is
/// the closed Gauss-Lobatto set, which from order 3 on differs from the
/// equispaced `SegPk` lattice the geometry is evaluated with.  The mesh
/// degrades to straight-sided with the vertex coordinates recovered — which
/// are always correct — and the element/boundary tables are untouched.
#[test]
fn d816_segment_h1_p3_read_refuses_the_table_but_keeps_the_vertices() {
    let file = read_mfem(Cursor::new(normalized(MFEM_H1_P3).as_bytes())).expect("read");
    let mesh = file.mesh1d.expect("1-D container");
    assert!(mesh.geometry.is_none(), "the order-3 table must be refused");
    assert_eq!(mesh.n_elems(), 4);
    assert_eq!(mesh.elem_nodes(3), &[3, 4]);

    // The vertex dofs (first NV values of the section) are recovered exactly.
    let dof = dof_values(&normalized(MFEM_H1_P3));
    assert_eq!(dof.len(), 13, "5 vertex dofs + 2 interiors per element");
    assert_eq!(
        mesh.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dof[..5].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    // The subsequent write is a well-formed *straight* file (the curvature
    // loss was already named loudly at read time).
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous).expect("straight write");
    let text = normalized(&String::from_utf8(buf).expect("utf-8"));
    assert!(
        !text.contains("nodes\n"),
        "a refused table must not leave a nodes section behind"
    );
    assert!(text.contains("vertices\n5\n1\n"), "straight coordinates block");
}

/// The writer must not accept an order-3+ 1-D table either (defence in
/// depth against a future table source): the refusal names the lattice
/// split instead of re-labelling values onto wrong positions.
#[test]
fn d816_segment_h1_p3_writer_refuses() {
    let dof: Vec<f64> = (0..13).map(|i| i as f64).collect();
    let mesh = Mesh::<1>::uniform(
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0, 1, 1, 2, 2, 3, 3, 4],
        vec![1; 4],
        ElementType::Line2,
        vec![],
        vec![],
        ElementType::Point1,
    );
    let mut mesh = mesh;
    mesh.geometry = Some(GeometryData {
        order: 3,
        conn: vec![
            0, 20, 21, 1, 1, 22, 23, 2, 2, 24, 25, 3, 3, 26, 27, 4,
        ],
        nodes_per_elem: 4,
        coords: dof,
        n_nodes: 28,
    });
    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous)
        .expect_err("an order-3 1-D table must not be writable");
    let msg = err.to_string();
    assert!(
        msg.contains("closed Gauss-Lobatto") && msg.contains("equispaced"),
        "unexpected order-3 refusal diagnostic: {msg}"
    );
    assert!(buf.is_empty(), "a refused write must emit nothing");
}

/// The round-80 folded `L2_T1_1D_P1` table is still not writable as a
/// continuous space — but for the *faithful* reason now: the folded
/// per-element geometry genuinely contradicts a shared-dof table at the
/// periodic seam, and the writer's continuity check refuses it (previously
/// the refusal was "no Line2 numbering implemented", which D816-3 closed).
#[test]
fn d816_segment_folded_l2_still_refuses_continuous_write() {
    let file = read_mfem_file(data_dir().join("periodic-segment.mesh")).expect("read");
    let mesh = file.mesh1d.expect("1-D container");
    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous)
        .expect_err("the folded table must not be foldable into a continuous space");
    let msg = err.to_string();
    assert!(
        msg.contains("is not continuous"),
        "unexpected continuous-writer diagnostic: {msg}"
    );
    assert!(buf.is_empty(), "a refused write must emit nothing");
}
