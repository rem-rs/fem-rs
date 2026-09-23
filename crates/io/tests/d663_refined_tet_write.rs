//! D663/D676: `write_mfem` must emit tetrahedral meshes in **storage order**,
//! exactly like MFEM 4.10's `Mesh::Printer`.
//!
//! Adjudication (MFEM 4.10 probes, `$HOME/work/d663/probe*.cpp` against serial
//! MFEM 4.10, beam-tet.mesh; verdicts in `tmp/d663/`):
//!
//! * `Mesh::Printer` writes the `elements` / `boundary` tables verbatim from
//!   storage (`mesh/mesh.cpp:12528-12545`); `MarkTetMeshForRefinement` runs
//!   only inside `Mesh::Finalize(refine = true)` — reached through
//!   `Mesh::Load`'s `refine = 1` default (`mesh.hpp:824`) — never at write
//!   time.  The write path therefore mirrors **storage**, and the read-side
//!   mark (`read_mfem`, mirroring `Load`) is the only normalization.
//! * `MarkTetMeshForRefinement` is idempotent on marked storage (probe `(b1)`:
//!   re-loading a marked Print changes 0 lines), but a uniformly *refined* tet
//!   mesh is NOT in marked orientation (probe `(b2)`: re-loading the refined
//!   Print with `refine = 1` rotates 16128 element lines and 0 boundary
//!   lines).  The old write-path clone+mark rotated exactly those 16128 lines
//!   relative to C++ — that was the entire D651 file-level divergence (the
//!   `tmp/d651` diff is 16128 differing element lines, 0 boundary/vertex
//!   lines), so the refined mesh must be written unmarked.
//! * Boundary-triangle `MarkEdge` rotations are cyclic (same orientation);
//!   `Mesh::Finalize()` defaults to `refine = false` (`mesh.hpp:1169`), which
//!   is why MFEM's trimmer prints its cut faces in first-encounter
//!   orientation.
//!
//! The fixtures are MFEM 4.10 ground truth (`Load(beam-tet.mesh,
//! refine=1)` + `UniformRefinement` × levels + `Print`): 1 refinement level
//! (`d663_refined_tet_l1_cpp.mesh`, NE=384) and the ex3-level 3 refinements
//! (`d663_refined_tet_l3_cpp.mesh`, NE=24576 — the `tmp/d651` acceptance
//! artifact).  Byte equality with these files is the D663 acceptance
//! (`refined_tet_rs.mesh` vs `refined_tet_cpp.mesh` diff = 0).

use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::Mesh;

/// `beam-tet.mesh` from the repo's MFEM data directory (48 tets, MFEM stores it
/// already in marked orientation, so `read_mfem`'s Load-parity mark is a no-op
/// here — exactly like C++ `Mesh::Load`).
fn beam_tet() -> Mesh<3> {
    let path = format!("{}/../../data/beam-tet.mesh", env!("CARGO_MANIFEST_DIR"));
    read_mfem_file(&path)
        .expect("read data/beam-tet.mesh")
        .mesh3d
        .expect("beam-tet.mesh holds a 3-D mesh")
}

/// Render the mesh through the serial writer (the `Mesh<2>` argument is the
/// writer signature's unused 2-D slot).
fn render(mesh: &Mesh<3>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem(&mut buf, &Mesh::<2>::unit_square_tri(1), Some(mesh))
        .expect("write_mfem must accept the refined tet mesh");
    let text = String::from_utf8(buf).expect("written mesh is ASCII");
    text.replace("\r\n", "\n")
}

fn assert_bytes_equal_c_cpp(got: &str, fixture: &str, label: &str) {
    if got != fixture {
        let a = got.lines().collect::<Vec<_>>();
        let b = fixture.lines().collect::<Vec<_>>();
        let shown = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .take(3)
            .collect::<Vec<_>>();
        panic!(
            "{label}: written mesh diverges from the MFEM 4.10 Print (got {} lines, want {} \
             lines; differing lines = {})\n{}",
            a.len(),
            b.len(),
            a.iter().zip(b.iter()).filter(|(x, y)| x != y).count(),
            shown
                .iter()
                .map(|(i, (x, y))| format!("  L{}: got {x:?}\n      want {y:?}", i + 1))
                .collect::<String>()
        );
    }
}

/// One uniform refinement: the written file must be byte-identical to MFEM's
/// `Load + UniformRefinement + Print` (storage order, unmarked children).
#[test]
fn refined_tet_l1_is_written_byte_identical_to_mfem_print() {
    let refined = fem_mesh::refine_uniform_3d(&beam_tet());
    assert_eq!(refined.n_elems(), 384, "one uniform refinement of 48 tets");
    assert_eq!(refined.n_faces(), 272);
    let got = render(&refined);
    let cpp = include_str!("data/d663_refined_tet_l1_cpp.mesh").replace("\r\n", "\n");
    assert_bytes_equal_c_cpp(&got, &cpp, "refined_tet_l1");
}

/// Three uniform refinements — the ex3 refined size and the D663 acceptance
/// artifact (`tmp/d651/refined_tet_cpp.mesh` vs the fem-rs `refined.mesh`).
#[test]
fn refined_tet_l3_is_written_byte_identical_to_mfem_print() {
    let mut mesh = beam_tet();
    for _ in 0..3 {
        mesh = fem_mesh::refine_uniform_3d(&mesh);
    }
    assert_eq!(mesh.n_elems(), 24576, "three uniform refinements of 48 tets");
    let got = render(&mesh);
    let cpp = include_str!("data/d663_refined_tet_l3_cpp.mesh").replace("\r\n", "\n");
    assert_bytes_equal_c_cpp(&got, &cpp, "refined_tet_l3");
}
