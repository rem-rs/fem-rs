//! D837-1 — NC quad refinement of `data/star.mesh` (the ex6 default mesh)
//! must reproduce serial MFEM 4.10's fine mesh element order, vertex
//! numbering and counts bit-for-bit (mesh level).
//!
//! Background (round-91 Lane A, evidence `tmp/d91a/`): round-90 registered
//! the ex6 iter1 unknowns divergence (C++ 76 vs Rust 86) as an AMR tri
//! closure bug.  That hypothesis is refuted twice over: `data/star.mesh` is a
//! **quad** mesh (20 straight quads, 31 vertices), so ex6 runs the
//! nonconforming aniso path, and the MFEM probe shows the refined mesh has
//! **86** vertices on both sides — MFEM merely *prints* `GetTrueVSize()`
//! (86 − 10 hanging-node dofs = 76) where the Rust example printed
//! `space.n_dofs()`.  What *did* diverge at the mesh level was the fine-mesh
//! ORDERING: MFEM emits leaf elements along the Hilbert SFC
//! (`NCMesh::CollectLeafElements`, `quad_hilbert_child_order`, driven by
//! `InitRootState`) and numbers vertices top-level-first-then-leaf-order
//! (`NCMesh::UpdateVertices`), while the flat refinement kept creation order.
//!
//! C++ truth (`tmp/d91a/probe2_d91a.cpp`, serial MFEM 4.10, WSL
//! `$HOME/mfem410_ser`): `Mesh::GeneralRefinement(refs, -1, 0)` with the 15
//! elements ex6 marks at AMR iter0 (the ZZ `aniso_flags` are all 3 = iso),
//! then a second geometric batch (mark iff `|centroid| < 0.45`, all iso).
//! Fixtures hold the probe dumps verbatim (`EL`/`BE`/`V` lines).

use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::{refine_nonconforming_quad, refine_nonconforming_quad_aniso, QuadRefineDir};

const MESH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/star.mesh");
const R1_CPP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d837_star_r1_cpp.txt");
const R2_CPP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/d837_star_r2_cpp.txt");

/// The 15 elements MFEM ex6 marks at AMR iteration 0 (ZZ eta > 0.7·max); the
/// C++ probe showed all their aniso flags are 3 (iso).
const EX6_MARKED: [u32; 15] = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19];

/// One parsed fixture: `EL i:` connectivity rows and `V i:` coordinate rows.
struct CppDump {
    elems: Vec<[u32; 4]>,
    verts: Vec<[f64; 2]>,
}

fn parse_dump(path: &str) -> CppDump {
    let text = std::fs::read_to_string(path).expect("read fixture");
    let mut elems = Vec::new();
    let mut verts = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("EL ") {
            let ids: Vec<u32> = rest[rest.find(':').unwrap() + 1..]
                .split_whitespace()
                .map(|t| t.parse().expect("EL id"))
                .collect();
            assert_eq!(ids.len(), 4, "fixture EL row: {line}");
            elems.push([ids[0], ids[1], ids[2], ids[3]]);
        } else if let Some(rest) = line.strip_prefix("V ") {
            let mut it = rest.split_whitespace();
            let _id: u32 = it.next().expect("V id").parse().expect("V id");
            let x: f64 = it.next().expect("V x").parse().expect("V x");
            let y: f64 = it.next().expect("V y").parse().expect("V y");
            verts.push([x, y]);
        }
    }
    CppDump { elems, verts }
}

fn star_mesh() -> fem_mesh::Mesh<2> {
    read_mfem_file(MESH)
        .expect("read star.mesh")
        .mesh2d
        .expect("star.mesh is 2-D")
}

fn ex6_marks() -> Vec<(u32, QuadRefineDir)> {
    EX6_MARKED.iter().map(|&e| (e, QuadRefineDir::Both)).collect()
}

/// Batch 1 must equal MFEM bit-for-bit: element rows (connectivity *and*
/// order), vertex coordinates, the hanging-node set (10 constraints, i.e.
/// MFEM's printed 76 = 86 − 10 true dofs) and the split boundary set.
#[test]
fn d837_batch1_matches_mfem_bitwise() {
    let mesh = star_mesh();
    let (fine, constraints) = refine_nonconforming_quad_aniso(&mesh, &ex6_marks(), None);

    let cpp = parse_dump(R1_CPP);
    assert_eq!(fine.n_nodes(), 86, "NV");
    assert_eq!(fine.n_elems(), 65, "NE");
    assert_eq!(cpp.elems.len(), 65, "fixture sanity");
    assert_eq!(cpp.verts.len(), 86, "fixture sanity");

    // Element connectivity, row by row, in MFEM's order.
    for e in 0..65u32 {
        let ns = fine.elem_nodes(e);
        for k in 0..4 {
            assert_eq!(ns[k], cpp.elems[e as usize][k], "EL {e} node {k}");
        }
    }

    // Vertex coordinates: midpoints use the same 0.5·(a+b) arithmetic on both
    // sides (bit-identical); element centers average the two diagonal
    // midpoints in MFEM (GetId(mid01, mid23)) vs the corner average here, so
    // a 1e-12 absolute guard covers those instead of demanding bit equality.
    for (v, cp) in fine.coords.chunks_exact(2).zip(cpp.verts.iter()) {
        assert!(
            (v[0] - cp[0]).abs() <= 1e-12 && (v[1] - cp[1]).abs() <= 1e-12,
            "V: rust {:?} vs cpp {:?}",
            v,
            cp
        );
    }

    // 10 hanging nodes (two per spike-tip quad on the inner pentagon edges)
    // => MFEM ex6 iter1 prints GetTrueVSize = 86 - 10 = 76.
    assert_eq!(constraints.len(), 10, "hanging-node constraints");
    assert_eq!(fine.n_nodes() - constraints.len(), 76, "MFEM true dofs");

    // Boundary: the same 30 directed segments as MFEM.  (The SEGMENT ORDER
    // differs — MFEM orders by NCMesh face id, see the D843-2 debt.)
    let mut rust_bdr: Vec<(u32, u32)> = (0..fine.n_faces() as u32)
        .map(|f| (fine.face_conn[2 * f as usize], fine.face_conn[2 * f as usize + 1]))
        .collect();
    rust_bdr.sort();
    let mut cpp_bdr: Vec<(u32, u32)> = Vec::new();
    let text = std::fs::read_to_string(R1_CPP).unwrap();
    for line in text.lines().filter(|l| l.starts_with("BE ")) {
        let mut it = line[line.find(':').unwrap() + 1..].split_whitespace();
        let a: u32 = it.next().unwrap().parse().unwrap();
        let b: u32 = it.next().unwrap().parse().unwrap();
        cpp_bdr.push((a, b));
    }
    cpp_bdr.sort();
    assert_eq!(rust_bdr.len(), 30);
    assert_eq!(rust_bdr, cpp_bdr, "boundary segment set (as directed pairs)");
}

/// The iso entry point (`refine_nonconforming_quad`) must produce the same
/// batch-1 mesh as the aniso entry with all-`Both` marks (MFEM: both are the
/// iso `RefineElement` branch).
#[test]
fn d837_batch1_iso_entry_matches_aniso() {
    let mesh = star_mesh();
    let marked: Vec<u32> = EX6_MARKED.to_vec();
    let (iso_mesh, iso_constraints) = refine_nonconforming_quad(&mesh, &marked, None);
    let (aniso_mesh, aniso_constraints) =
        refine_nonconforming_quad_aniso(&mesh, &ex6_marks(), None);

    assert_eq!(iso_mesh.n_nodes(), aniso_mesh.n_nodes());
    assert_eq!(iso_mesh.n_elems(), aniso_mesh.n_elems());
    for e in 0..iso_mesh.n_elems() as u32 {
        assert_eq!(
            iso_mesh.elem_nodes(e),
            aniso_mesh.elem_nodes(e),
            "EL {e} differs between iso and aniso-Both entries"
        );
    }
    assert_eq!(iso_constraints.len(), aniso_constraints.len());
}

/// Batch 2 (geometric mark rule `|centroid| < 0.45`, all iso): MFEM counts
/// 141 vertices / 110 elements, and the element SEQUENCE must still follow
/// the Hilbert leaf expansion (per-leaf states recovered by the
/// `InitRootState` rule).  Rows are compared through their vertex
/// coordinates so the pin is insensitive to the D843-1 node-id permutation
/// of the first batch's midpoints.
#[test]
fn d837_batch2_leaf_order_and_counts_match_mfem() {
    let mesh = star_mesh();
    let (fine1, _) = refine_nonconforming_quad_aniso(&mesh, &ex6_marks(), None);

    let marked2: Vec<(u32, QuadRefineDir)> = {
        let mut out = Vec::new();
        for e in 0..fine1.n_elems() as u32 {
            let (mut cx, mut cy) = (0.0f64, 0.0f64);
            for &n in fine1.elem_nodes(e) {
                let c = fine1.coords_of(n);
                cx += c[0];
                cy += c[1];
            }
            cx /= 4.0;
            cy /= 4.0;
            if cx * cx + cy * cy < 0.45 * 0.45 {
                out.push((e, QuadRefineDir::Both));
            }
        }
        out
    };
    assert_eq!(marked2.len(), 15, "the probe marks 15 leaves in round 2");
    let (fine2, _) = refine_nonconforming_quad_aniso(&fine1, &marked2, None);

    let cpp = parse_dump(R2_CPP);
    assert_eq!(fine2.n_nodes(), 141, "NV after batch 2");
    assert_eq!(fine2.n_elems(), 110, "NE after batch 2");
    assert_eq!(cpp.elems.len(), 110, "fixture sanity");
    assert_eq!(cpp.verts.len(), 141, "fixture sanity");

    // Row-by-row geometry: each C++ row's four vertex coordinates (through
    // the C++ V table) must equal this row's four vertex coordinates, in
    // order — that pins the leaf order without pinning the D843-1 ids.
    for (e, ce) in cpp.elems.iter().enumerate() {
        for (k, &id) in ce.iter().enumerate() {
            let cc = cpp.verts[id as usize];
            let n = fine2.elem_nodes(e as u32)[k];
            let rc = fine2.coords_of(n);
            assert!(
                (rc[0] - cc[0]).abs() <= 1e-12 && (rc[1] - cc[1]).abs() <= 1e-12,
                "batch-2 EL {e} node {k}: rust {rc:?} vs cpp {cc:?}"
            );
        }
    }
}
