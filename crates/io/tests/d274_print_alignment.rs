//! D274: `write_mfem`'s text alignment with MFEM 4.10's `Mesh::Print`.
//!
//! Three C++ behaviours, each pinned byte-for-byte against the round-31
//! probe `tmp/round31_spot_save.cpp` (`Mesh mesh(in); mesh.Save(out, 16)`,
//! WSL `$HOME/work/r31_save`), whose products live in
//! `tests/data/d314_cpp_save/`:
//!
//! 1. straight-sided vertex rows start at the first coordinate
//!    (`Mesh::Printer`, `mesh/mesh.cpp:12544` — `os << vertices[i](0)` first),
//!    no leading blank;
//! 2. every printed value uses the stream precision, whose file-writer
//!    default is `Mesh::Save`'s `precision = 16` (`mesh/mesh.hpp:2616`) —
//!    not Rust's shortest-roundtrip `Display` and not `mfem::out`'s 6-digit
//!    console precision;
//! 3. the curved `nodes` section carries `Ordering: 1` (`byVDIM`): MFEM 4.10
//!    `Mesh::SetCurvature` defaults to `ordering = 1` (`mesh/mesh.hpp:2439`)
//!    and `Mesh::Printer` round-trips whatever the read file declared — the
//!    C++ re-save of the `Ordering: 1` toroid fixture still says
//!    `Ordering: 1`.
//!
//! D294 closed the last gap: `write_mfem` now emits `Mesh::Printer`'s
//! geometry-type comment block itself (`mesh/mesh.cpp:12521-12531`), so the
//! comparison against C++ `Save` is byte-for-byte over the whole file —
//! block included (the round-43 pins had to strip the block from the
//! references; the old stripped copies were removed with this change).
//!
//! The formatting fixed point is pinned too: write → read → write is
//! byte-stable.  That is the same fixed point C++'s own 16-digit save
//! satisfies (a 16-digit decimal parses to the f64 whose %.16g rendering is
//! the same 16 digits); the previous `Display`-based writer was exact across
//! the cycle, but only because Rust's shortest-roundtrip formatting is a
//! superset of anything MFEM prints.

use std::path::PathBuf;

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::simplex::Mesh;

/// (input fixture, C++ `Mesh::Save(out, 16)` re-save, dimension).  The first
/// two entries' fixtures live in the `fem-mesh` test data, the rest in this
/// crate's own `tests/data`.
const CASES: &[(&str, &str, u8, bool)] = &[
    // Straight-sided wedge torus: exercises the vertex-row format (leading
    // space removal + %.16g).
    ("toroid_wedge_o1_r1.mesh", "cpp_toroid_wedge_o1_r1.mesh", 3, true),
    // Curved hex cube: hex `nodes` numbering at %.16g.
    ("curved_hex_p3.mesh", "cpp_curved_hex_p3.mesh", 3, false),
    // 2-D triangles, H1_2D_P2/P3/P4 (the fixtures whose bit-exact round trip
    // the old Display writer guaranteed and %.16g cannot — C++ truncates
    // `0.16666666666666666` to `0.1666666666666667` exactly the same way).
    ("tri2d_0_p2_g32.mesh", "cpp_tri2d_0_p2_g32.mesh", 2, false),
    ("tri2d_0_p3_g32.mesh", "cpp_tri2d_0_p3_g32.mesh", 2, false),
    ("tri2d_0_p4_g32.mesh", "cpp_tri2d_0_p4_g32.mesh", 2, false),
];

fn fixture_path(name: &str, mesh_data: bool) -> PathBuf {
    if mesh_data {
        [env!("CARGO_MANIFEST_DIR"), "..", "mesh", "tests", "data"]
            .iter()
            .collect::<PathBuf>()
            .join(name)
    } else {
        [env!("CARGO_MANIFEST_DIR"), "tests", "data"]
            .iter()
            .collect::<PathBuf>()
            .join(name)
    }
}

fn reference_path(name: &str) -> PathBuf {
    [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "data",
        "d314_cpp_save",
    ]
    .iter()
    .collect::<PathBuf>()
    .join(name)
}

/// The fixture read into the dimension its case declares.
fn read_case(fixture: &str, dim: u8) -> (Mesh<2>, Option<Mesh<3>>) {
    let file = read_mfem_file(fixture).unwrap_or_else(|e| panic!("read {fixture}: {e}"));
    match dim {
        2 => (file.mesh2d.expect("2-D mesh"), None),
        _ => (
            Mesh::<2>::unit_square_tri(1), // ignored by the 3-D write path
            Some(file.mesh3d.expect("3-D mesh")),
        ),
    }
}

fn write_case(mesh2d: &Mesh<2>, mesh3d: Option<&Mesh<3>>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, mesh2d, mesh3d, NodesSpace::Continuous)
        .expect("write_mfem_nodes must succeed");
    String::from_utf8(buf).expect("written mesh is ASCII")
}

/// The written text must equal MFEM's own re-save of the same fixture, byte
/// for byte, geometry comment block included (D294).
#[test]
fn write_matches_cpp_save_byte_for_byte() {
    for (fixture, reference, dim, mesh_data) in CASES {
        let fixture = fixture_path(fixture, *mesh_data);
        let fixture_name = fixture.display().to_string();
        let mesh2d = read_case(&fixture_name, *dim);
        let got = write_case(&mesh2d.0, mesh2d.1.as_ref());
        let want = std::fs::read_to_string(reference_path(reference))
            .unwrap_or_else(|e| panic!("{reference}: {e}"));
        assert!(
            got == want,
            "{fixture_name} vs C++ re-save {reference}: {} byte mismatch\nfirst difference: {:?}",
            if got.len() != want.len() {
                format!("{} vs {} ", got.len(), want.len())
            } else {
                String::new()
            },
            got.bytes()
                .zip(want.bytes())
                .position(|(a, b)| a != b)
                .map(|i| &got[i.saturating_sub(40)..(i + 40).min(got.len())]),
        );
    }
}

/// The written text is a formatting fixed point: read(write(read(f))) writes
/// back the same bytes — the C++-parity notion of round-trip exactness at a
/// finite stream precision.
#[test]
fn write_read_write_is_byte_stable() {
    for (fixture, _, dim, mesh_data) in CASES {
        let fixture = fixture_path(fixture, *mesh_data);
        let fixture_name = fixture.display().to_string();
        let first = read_case(&fixture_name, *dim);
        let text1 = write_case(&first.0, first.1.as_ref());
        let again = read_case_text(&text1, *dim);
        let text2 = write_case(&again.0, again.1.as_ref());
        assert_eq!(
            text1, text2,
            "{fixture_name}: the second write must reproduce the first byte for byte"
        );
    }
}

fn read_case_text(text: &str, dim: u8) -> (Mesh<2>, Option<Mesh<3>>) {
    let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec()))
        .expect("read_mfem must accept the written text");
    match dim {
        2 => (file.mesh2d.expect("2-D mesh"), None),
        _ => (Mesh::<2>::unit_square_tri(1), Some(file.mesh3d.expect("3-D mesh"))),
    }
}
