//! D843-2: NC refinement boundary-segment ORDER matches MFEM's
//! `Mesh::boundary` array order (ascending NCMesh face id).
//!
//! MFEM fills `Mesh::boundary` in `GetMeshComponents` by collecting each
//! leaf element's boundary faces into a `std::map<face id, verts>` and
//! appending in map order (ncmesh.cpp:2790/2866) — i.e. **ascending NCMesh
//! face id**.  Face ids evolve through refinement as the `faces` hash
//! allocates (children-first `faces.Get`, freelist LIFO reuse,
//! hash.hpp:627) and frees (`DeleteUnusedFaces`) per refined element;
//! `Mesh::vertex_parents`-style renumbering remaps the keys each batch.
//! `refine_nonconforming_quad_aniso` simulates exactly this lifecycle
//! (`advance_nc_face_ids_quad`) and emits `face_conn` in MFEM's order.
//!
//! Truth: MFEM 4.10 serial, ex6 verbatim AMR loop on star.mesh (the
//! round-95 C++ probe): the fixture holds C++'s boundary segment order for
//! rounds 0..=12 (bit-verified scope) and the segment COUNTS for all 20
//! rounds.  The mark batches are the ex6 driver's real per-round marks
//! (dumped from the bit-identical Rust driver), so the replay needs no
//! solver.
//!
//! Tooth: on the pre-D843-2 code the boundary segments keep old-face order
//! with in-place splits — the first re-split round (R1) already differs
//! from MFEM's face-id order, so every ordered assertion below goes red.

use fem_mesh::amr::{refine_nonconforming_quad_aniso, QuadRefineDir};
use fem_mesh::Mesh;

const MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/star.mesh");
const MARKS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d846_marks.txt");
const CPP_ORDER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d846_cpp_boundary_all.txt");
const CPP_NBE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d846_nbe_all.txt");

fn star_mesh() -> Mesh<2> {
    fem_io::mfem::read_mfem_file(MESH)
        .expect("read star.mesh")
        .mesh2d
        .expect("star.mesh must be 2-D")
}

/// (round, element, dir) mark batches, grouped by round, array order kept.
fn marks_by_round() -> Vec<Vec<(u32, QuadRefineDir)>> {
    let text = std::fs::read_to_string(MARKS).expect("read marks");
    let mut rounds: Vec<Vec<(u32, QuadRefineDir)>> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 3, "mark line: {line}");
        let r: usize = f[0][1..].parse().expect("round");
        let e: u32 = f[1].parse().expect("elem");
        let dir = match f[2] {
            "X" => QuadRefineDir::X,
            "Y" => QuadRefineDir::Y,
            _ => QuadRefineDir::Both,
        };
        if rounds.len() <= r {
            rounds.resize(r + 1, Vec::new());
        }
        rounds[r].push((e, dir));
    }
    rounds
}

/// C++ boundary order for rounds 0..=12: `round -> [(v0, v1), …]`.
fn cpp_order() -> Vec<Vec<(u32, u32)>> {
    let text = std::fs::read_to_string(CPP_ORDER).expect("read cpp order fixture");
    let mut rounds: Vec<Vec<(u32, u32)>> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 7, "fixture line: {line}");
        assert_eq!(f[1], "BE");
        let r: usize = f[0][1..].parse().expect("round");
        let seg = (f[5].parse().expect("v0"), f[6].parse().expect("v1"));
        if rounds.len() <= r {
            rounds.resize(r + 1, Vec::new());
        }
        rounds[r].push(seg);
    }
    rounds
}

/// C++ boundary segment counts for all 20 rounds.
fn cpp_nbe() -> Vec<(usize, usize)> {
    let text = std::fs::read_to_string(CPP_NBE).expect("read cpp nbe fixture");
    text.lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f[0][1..].parse().expect("round"), f[1].parse().expect("nbe"))
        })
        .collect()
}

#[test]
fn d8432_boundary_order_matches_mfem() {
    let cpp_order = cpp_order();
    let cpp_nbe = cpp_nbe();
    let marks = marks_by_round();
    assert_eq!(cpp_nbe.len(), 20, "20 ex6 AMR rounds expected");

    let mut mesh = star_mesh();
    for (r, batch) in marks.iter().enumerate() {
        let (fine, _constraints) = refine_nonconforming_quad_aniso(&mesh, batch, None);
        mesh = fine;

        // Segment count vs MFEM, all rounds.
        assert_eq!(
            mesh.n_faces(),
            cpp_nbe[r + 1].1,
            "R{}: NBE {} vs C++ {}",
            r + 1,
            mesh.n_faces(),
            cpp_nbe[r + 1].1
        );

        // Full boundary order vs MFEM, all 20 rounds (the face-id lifecycle
        // is event-identical to the instrumented MFEM run — round 96).
        let got: Vec<(u32, u32)> = (0..mesh.n_faces())
            .map(|f| (mesh.face_conn[2 * f], mesh.face_conn[2 * f + 1]))
            .collect();
        assert_eq!(
            got,
            cpp_order[r + 1],
            "R{}: boundary order diverged from MFEM face-id order",
            r + 1
        );
    }
}

#[test]
fn d8432_round0_boundary_is_file_order() {
    // R0: before the first NC batch the boundary array is the file order —
    // the face-id machinery must not touch it.
    let mesh = star_mesh();
    let cpp = cpp_order();
    let got: Vec<(u32, u32)> = (0..mesh.n_faces())
        .map(|f| (mesh.face_conn[2 * f], mesh.face_conn[2 * f + 1]))
        .collect();
    assert_eq!(got, cpp[0], "R0 must be the file order (no NC refinement yet)");
}
