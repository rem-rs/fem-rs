//! D813-4 — the `dimension 1` branch of the `.mesh` reader, end to end.
//!
//! `read_mfem` refused `dimension 1` files outright ("dim=1 unsupported"), so
//! `data/periodic-segment.mesh` — a 1-D periodic ring whose folded geometry is
//! a `L2_T1_1D_P1` `nodes` section — had no reader at all (its `L2_T1_1D_P1`
//! table was the one D812-1 found that could "never reach the writer").  The
//! reader now fills the same `Mesh<1>` container the INLINE `type = segment`
//! arm uses (D724), and the writer got a matching 1-D entry point
//! ([`fem_io::mfem::write_mfem_nodes_1d`]) over the dimension-generic engine.
//!
//! The oracles are MFEM 4.10's own re-saves (`Mesh::Save(out, 16)`, probes
//! `$HOME/work/d80b/{probe,mk1d_probe}.cpp`):
//!
//! * `tests/data/d813_mfem_resave_periodic-segment.mesh.txt` — the folded
//!   `L2_T1_1D_P1` mesh re-saved: element table verbatim (MFEM's loader
//!   finalizes 1-D meshes with `FinalizeTopology` only — no orientation pass,
//!   no refinement marking), vertex table = the **mean** over references
//!   (D813-2's rule, dim 1), `nodes` values one per line (`VDim: 1`).  The
//!   read → write result must equal it **byte for byte** (include_str!'d);
//! * `tests/data/d814_mfem_resave_inline-segment.txt` — the straight arm:
//!   `Make1D(4, 1)` (via `data/inline-segment.mesh`) carries two boundary
//!   `POINT` records (MFEM geometry code 0), which the writer emits and the
//!   reader parses back.
//!
//! Set `D80B_DUMP_DIR=<dir>` to have every written file dropped on disk for an
//! external diff (that is how the byte-level results below were captured).

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes_1d, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;
use fem_mesh::MeshTopology;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// MFEM 4.10's `Mesh::Save(out, 16)` re-save of `data/periodic-segment.mesh`
/// (`$HOME/work/d80b/probe`), included verbatim.
const MFEM_RESAVE_SEGMENT: &str =
    include_str!("data/d813_mfem_resave_periodic-segment.mesh.txt");

/// MFEM 4.10's `Mesh::Save(out, 16)` re-save of `Mesh::Make1D(4, 1.0)`
/// (`$HOME/work/d80b/mk1d_probe` on `data/inline-segment.mesh`), included
/// verbatim.
const MFEM_RESAVE_MAKE1D: &str = include_str!("data/d814_mfem_resave_inline-segment.txt");

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

fn read_segment() -> Mesh<1> {
    let file = read_mfem_file(data_dir().join("periodic-segment.mesh")).expect("read");
    file.mesh1d.expect("the 1-D container must be filled")
}

// ─── the reader side ─────────────────────────────────────────────────────────

/// The read recovers MFEM's own in-memory state: the element table verbatim
/// (no 1-D orientation pass on MFEM's loader, `Mesh::FinalizeTopology` only),
/// the folded vertex table as the arithmetic mean over every element
/// reference, and the `L2_T1_1D_P1` geometry table as 2 private dofs per
/// element.
#[test]
fn d813_segment_read_recovers_mfems_folded_state() {
    let mesh = read_segment();
    assert_eq!(mesh.n_elems(), 4);
    assert_eq!(mesh.n_nodes(), 4);
    assert_eq!(mesh.topological_dim(), 1);
    assert_eq!(mesh.elem_type, ElementType::Line2);

    // Element 3 wraps around the periodic identification: `1 1 3 0`, verbatim.
    assert_eq!(mesh.elem_nodes(0), &[0, 1]);
    assert_eq!(mesh.elem_nodes(1), &[1, 2]);
    assert_eq!(mesh.elem_nodes(2), &[2, 3]);
    assert_eq!(mesh.elem_nodes(3), &[3, 0]);

    // Vertex table = mean over references: vertex 0 is claimed by element 0
    // (x = 0) and element 3 (x = 1) — the wrapped corner averages to 0.5.
    let verts: Vec<f64> = (0..4).map(|v| mesh.coords[v]).collect();
    assert_eq!(verts, [0.5, 0.25, 0.5, 0.75]);

    // The folded geometry table: order 1, two private dofs per element, the
    // file's values in MFEM's `L2_SegmentElement` order (ascending, = the
    // mesh's own slot order at p = 1).
    let g = mesh.geometry.as_ref().expect("the L2 geometry table");
    assert_eq!(g.order, 1);
    assert_eq!(g.nodes_per_elem, 2);
    assert_eq!(g.n_nodes, 8);
    assert_eq!(g.conn, (0..8).collect::<Vec<u32>>());
    assert_eq!(
        g.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0.0f64, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

// ─── the byte oracle: read → write == MFEM's own re-save ─────────────────────

#[test]
fn d813_segment_l2_round_trip_matches_mfem_resave() {
    let mesh = read_segment();
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Discontinuous).expect("write_mfem_nodes_1d");
    dump("d813_rs_periodic-segment.mesh", &buf);
    assert_eq!(
        normalized(&String::from_utf8(buf).expect("utf-8")),
        normalized(MFEM_RESAVE_SEGMENT),
        "the written file differs from MFEM's own re-save \
         (`Mesh::Save(out, 16)` of the same input)"
    );
}

/// The round trip is bit-exact through fem-rs's own reader, too: read → write
/// → read must return the same geometry table and the same vertex means.
#[test]
fn d813_segment_geometry_survives_the_round_trip_bitexact() {
    let mesh = read_segment();
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Discontinuous).expect("write");
    let back = read_mfem(Cursor::new(&buf)).expect("read back");
    let mesh2 = back.mesh1d.expect("1-D container again");

    assert_eq!(mesh2.coords, mesh.coords, "vertex means");
    let (g, g2) = (mesh.geometry.as_ref().unwrap(), mesh2.geometry.as_ref().unwrap());
    assert_eq!(g2.order, g.order);
    assert_eq!(g2.nodes_per_elem, g.nodes_per_elem);
    assert_eq!(g2.conn, g.conn);
    assert_eq!(
        g2.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        g.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // …and the recovered table is still a discontinuous one.
    assert_eq!(g.order, 1);
}

// ─── teeth ───────────────────────────────────────────────────────────────────

/// The *continuous* writer must not silently fold the periodic table into a
/// shared vertex space: `Line2` has no continuous (`H1_1D_P*`) numbering yet,
/// so the request is refused loudly and nothing is emitted.
#[test]
fn d813_segment_continuous_writer_refuses() {
    let mesh = read_segment();
    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous)
        .expect_err("the folded table must not be writable as a continuous space");
    let msg = err.to_string();
    assert!(
        msg.contains("no MFEM-faithful continuous `nodes` numbering for Line2"),
        "unexpected continuous-writer diagnostic: {msg}"
    );
    assert!(buf.is_empty(), "a refused write must emit nothing");
}

// ─── the straight arm: Make1D's boundary POINTs ──────────────────────────────

/// A straight 1-D mesh with two boundary `POINT` records — exactly what MFEM's
/// `Make1D(4, 1.0)` produces — must be written byte for byte like MFEM's own
/// re-save, and read back with both tables intact (geometry code 0 on the
/// boundary, one vertex per record).
#[test]
fn d814_straight_segment_with_boundary_points_matches_make1d_resave() {
    let mesh = Mesh::<1>::uniform(
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0, 1, 1, 2, 2, 3, 3, 4],
        vec![1; 4],
        ElementType::Line2,
        vec![0, 4],
        vec![1, 2],
        ElementType::Point1,
    );
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous).expect("write");
    dump("d814_rs_inline-segment.mesh", &buf);
    assert_eq!(
        normalized(&String::from_utf8(buf.clone()).expect("utf-8")),
        normalized(MFEM_RESAVE_MAKE1D),
        "the written file differs from MFEM's `Make1D(4, 1)` re-save"
    );

    // Read back: 5 vertices, 4 segments, and the two POINT boundary records
    // with their attributes.
    let back = read_mfem(Cursor::new(&buf)).expect("read back");
    let mesh2 = back.mesh1d.expect("1-D container");
    assert_eq!(mesh2.n_nodes(), 5);
    assert_eq!(mesh2.n_elems(), 4);
    assert_eq!(mesh2.coords, mesh.coords);
    assert_eq!(mesh2.face_type, ElementType::Point1);
    assert_eq!(mesh2.face_conn, vec![0, 4]);
    assert_eq!(mesh2.face_tags, vec![1, 2]);
    assert!(mesh2.geometry.is_none(), "a straight mesh stays straight");
}
