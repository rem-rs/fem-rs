//! Round 31 (D126) acceptance tests for `write_mfem`'s boundary section.
//!
//! The writer used to derive every boundary record from a hard-coded
//! "3 nodes ⇒ TRIANGLE, geom code 2" assumption whenever `Mesh::face_offsets`
//! was `None`, ignoring the mesh's own `face_type` / `face_types`.  A
//! quadrilateral boundary (e.g. every extruded hex mesh) was therefore written
//! as a truncated triangle, and a wedge mesh whose `elem_type` no longer
//! matched its connectivity was written as a *different* element list read from
//! the same buffer — producing `.mesh` files that neither MFEM 4.10 nor
//! `read_mfem` could read back (MFEM aborts with `Invalid mesh topology`).
//!
//! These tests pin the two halves of the fix:
//!
//! 1. `write_mfem` must use `Mesh::face_type_at` for both the node count and
//!    the MFEM geometry code, so a written mesh reads back identically
//!    (`n_elems`, `n_faces`, per-face node counts, face connectivity and node
//!    coordinates).
//! 2. `write_mfem` must *refuse* an internally inconsistent mesh (a
//!    `FemError`, not a silently corrupt file) — the length of `face_conn` /
//!    `conn` must agree with the declared per-entity types.
//!
//! Ground truth for "MFEM can read it" is the external probe
//! `tmp/round31_meshread.cpp` (built against serial MFEM 4.10); see the round
//! 31 report for its per-file output.

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem, write_mfem_file_3d};
use fem_mesh::{element_type::ElementType, simplex::Mesh};

// ─── helpers ────────────────────────────────────────────────────────────────

/// A 2-D mesh is required by the `write_mfem` signature even when a 3-D mesh
/// is written; any valid tiny 2-D mesh works (it is ignored).
fn scratch_2d() -> Mesh<2> {
    Mesh::<2>::unit_square_tri(1)
}

fn write_to_string(mesh: &Mesh<3>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem(&mut buf, &scratch_2d(), Some(mesh)).expect("write_mfem must succeed");
    String::from_utf8(buf).expect("written mesh is ASCII")
}

fn read_back(text: &str) -> Mesh<3> {
    let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec()))
        .expect("read_mfem must accept the file write_mfem produced");
    file.mesh3d.expect("3-D mesh")
}

/// Every boundary face as a sorted node list, so orientation changes (which the
/// tet canonicalization is allowed to apply) do not mask a connectivity bug.
fn faces_as_sets(mesh: &Mesh<3>) -> Vec<Vec<u32>> {
    (0..mesh.n_faces() as u32)
        .map(|f| {
            let mut v: Vec<u32> = mesh.bface_nodes(f).to_vec();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Full round-trip equality check: element/face counts, each face's node count,
/// the face connectivity (as sets) and every node coordinate bit-for-bit.
fn assert_round_trip(mesh: &Mesh<3>, label: &str) {
    let text = write_to_string(mesh);
    let back = read_back(&text);

    assert_eq!(back.n_elems(), mesh.n_elems(), "{label}: element count");
    assert_eq!(back.n_faces(), mesh.n_faces(), "{label}: boundary face count");
    assert_eq!(
        back.elem_type, mesh.elem_type,
        "{label}: element type must survive the round trip"
    );

    for f in 0..mesh.n_faces() as u32 {
        assert_eq!(
            back.face_type_at(f).nodes_per_element(),
            mesh.face_type_at(f).nodes_per_element(),
            "{label}: node count of boundary face {f}"
        );
    }
    assert_eq!(
        faces_as_sets(&back),
        faces_as_sets(mesh),
        "{label}: boundary face connectivity"
    );

    assert_eq!(back.n_nodes(), mesh.n_nodes(), "{label}: node count");
    assert_eq!(
        back.coords.len(),
        mesh.coords.len(),
        "{label}: coordinate array length"
    );
    for (i, (a, b)) in back.coords.iter().zip(mesh.coords.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{label}: coordinate {i} differs: wrote {b}, read {a}"
        );
    }
}

// ─── 1. round trips ─────────────────────────────────────────────────────────

/// Uniform Hex8: 24 Quad4 boundary faces, `face_offsets == None`.
#[test]
fn round_trip_uniform_hex8() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    assert_eq!(mesh.face_type, ElementType::Quad4);
    assert!(mesh.face_offsets.is_none());
    assert_eq!(mesh.face_conn.len(), mesh.n_faces() * 4);
    assert_round_trip(&mesh, "unit_cube_hex(2)");

    // The failure this test exists for: before D126 every record was written as
    // `TRIANGLE` (code 2) with 3 of the 4 quad nodes.
    let text = write_to_string(&mesh);
    let boundary = text.split("\nboundary\n").nth(1).expect("boundary section");
    let mut lines = boundary.lines();
    assert_eq!(lines.next().unwrap().trim(), "24", "24 boundary faces");
    for line in lines.take(24) {
        let t: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(t[1], "3", "quad face must be written with geom code 3");
        assert_eq!(t.len(), 2 + 4, "quad face must carry 4 nodes");
    }
}

/// Uniform Tet4: triangular boundary faces (reader canonicalizes tet vertex
/// order on both sides — `mark_tet_mesh_for_refinement`).
#[test]
fn round_trip_uniform_tet4() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    assert_eq!(mesh.face_type, ElementType::Tri3);
    assert_round_trip(&mesh, "unit_cube_tet(2)");
}

/// A prism mesh exercises the *mixed boundary* path: the wedge's two end faces
/// are `Tri3` and its three side faces are `Quad4`, so the writer must emit
/// `face_types` + `face_offsets` and use each face's own type.
#[test]
fn round_trip_prism_mixed_boundary() {
    // Unit wedge: triangle (0,0),(1,0),(0,1) extruded from z=0 to z=1.
    let coords = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 2
        0.0, 0.0, 1.0, // 3
        1.0, 0.0, 1.0, // 4
        0.0, 1.0, 1.0, // 5
    ];
    let conn = vec![0, 1, 2, 3, 4, 5];
    let face_conn = vec![
        // bottom triangle (z=0), top triangle (z=1)
        0, 1, 2, //
        3, 4, 5, //
        // three quads
        0, 1, 4, 3, //
        1, 2, 5, 4, //
        2, 0, 3, 5,
    ];
    let mesh = Mesh::<3> {
        coords,
        conn,
        elem_tags: vec![1],
        elem_type: ElementType::Prism6,
        face_conn,
        face_tags: vec![1, 2, 3, 3, 3],
        face_type: ElementType::Tri3,
        elem_types: None,
        elem_offsets: None,
        face_types: Some(vec![
            ElementType::Tri3,
            ElementType::Tri3,
            ElementType::Quad4,
            ElementType::Quad4,
            ElementType::Quad4,
        ]),
        face_offsets: Some(vec![0, 3, 6, 10, 14, 18]),
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    };

    assert_round_trip(&mesh, "unit wedge (mixed boundary)");

    // The written codes must follow each face's own type.
    let text = write_to_string(&mesh);
    let boundary = text.split("\nboundary\n").nth(1).expect("boundary section");
    let codes: Vec<String> = boundary
        .lines()
        .skip(1)
        .take(5)
        .map(|l| l.split_whitespace().nth(1).unwrap().to_string())
        .collect();
    assert_eq!(codes, vec!["2", "2", "3", "3", "3"], "Tri3=2, Quad4=3");
}

/// A hand-built two-hex brick with `face_offsets == None` and the six boundary
/// quads deliberately listed in a non-canonical order: the writer must follow
/// `face_conn` in order and still use the mesh's `face_type` (Quad4).
#[test]
fn round_trip_manual_hex_face_offsets_none() {
    let mut coords = Vec::new();
    for x in 0..=2 {
        for y in 0..=1 {
            for z in 0..=1 {
                coords.push(x as f64);
                coords.push(y as f64);
                coords.push(z as f64);
            }
        }
    }
    // node id = x*4 + y*2 + z
    let n = |x: u32, y: u32, z: u32| x * 4 + y * 2 + z;
    let mut conn = Vec::new();
    for x in 0..2 {
        conn.extend_from_slice(&[
            n(x, 0, 0),
            n(x + 1, 0, 0),
            n(x + 1, 1, 0),
            n(x, 1, 0),
            n(x, 0, 1),
            n(x + 1, 0, 1),
            n(x + 1, 1, 1),
            n(x, 1, 1),
        ]);
    }
    // Six outer faces, listed x=1 first, then the ends, then the four sides.
    let face_conn = vec![
        n(2, 0, 0), n(2, 0, 1), n(2, 1, 1), n(2, 1, 0), // x = 2
        n(0, 0, 0), n(0, 1, 0), n(0, 1, 1), n(0, 0, 1), // z = 1
        n(0, 0, 0), n(0, 1, 0), n(1, 1, 0), n(1, 0, 0), // z = 0
        n(0, 0, 0), n(1, 0, 0), n(1, 0, 1), n(0, 0, 1), // y = 0
        n(0, 1, 0), n(1, 1, 0), n(1, 1, 1), n(0, 1, 1), // y = 1
        n(0, 0, 1), n(0, 0, 0), n(0, 1, 0), n(0, 1, 1), // x = 0
    ];
    let mesh = Mesh::<3> {
        coords,
        conn,
        elem_tags: vec![1, 1],
        elem_type: ElementType::Hex8,
        face_conn,
        face_tags: vec![1, 2, 3, 4, 5, 6],
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    };
    assert_round_trip(&mesh, "hand-built 2-hex brick");

    // Tags must be carried through unchanged, in the order they were listed.
    let back = read_back(&write_to_string(&mesh));
    assert_eq!(back.face_tags, vec![1, 2, 3, 4, 5, 6]);
}

/// 2-D path: the boundary faces are `Line2` edges; the writer used to hard-code
/// the 2-D stride as 2 nodes / code 1 without consulting `face_type`.
#[test]
fn round_trip_2d_tri_and_quad() {
    for (label, mesh) in [
        ("unit_square_tri(2)", Mesh::<2>::unit_square_tri(2)),
        ("unit_square_quad(2)", Mesh::<2>::unit_square_quad(2)),
    ] {
        let mut buf: Vec<u8> = Vec::new();
        write_mfem(&mut buf, &mesh, None).expect("write_mfem must succeed");
        let text = String::from_utf8(buf).unwrap();
        let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec())).unwrap();
        let back = file.mesh2d.expect("2-D mesh");
        assert_eq!(back.n_elems(), mesh.n_elems(), "{label}: element count");
        assert_eq!(back.n_faces(), mesh.n_faces(), "{label}: face count");
        assert_eq!(back.face_type, ElementType::Line2, "{label}: face type");
        assert_eq!(back.face_conn, mesh.face_conn, "{label}: face connectivity");
        assert_eq!(back.coords, mesh.coords, "{label}: coordinates");
    }
}

// ─── 2. inconsistent meshes must be refused ─────────────────────────────────

/// `face_conn` truncated by one entry: the declared types cannot describe it.
/// The old writer silently consumed 3 quads' worth of nodes as 4 triangles.
#[test]
fn rejects_truncated_face_conn() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    // Drop one node from the last face and one tag, so the number of faces and
    // the tag table stay consistent and only the connectivity length is wrong.
    mesh.face_conn.pop();
    mesh.face_tags.pop();

    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem(&mut buf, &scratch_2d(), Some(&mesh))
        .expect_err("truncated face_conn must be refused");
    let msg = err.to_string();
    println!("[truncated face_conn] {msg}");
    assert!(msg.contains("face_conn"), "message should name face_conn: {msg}");
    assert!(
        buf.is_empty(),
        "nothing may be written when validation fails (got {} bytes)",
        buf.len()
    );

    // And no file at all must appear on disk.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.mesh");
    let err2 = write_mfem_file_3d(&path, &mesh).expect_err("must refuse");
    assert!(err2.to_string().contains("face_conn"), "{err2}");
    assert!(
        !path.exists(),
        "write_mfem_file_3d must validate before creating the file"
    );
}

/// `face_offsets` says a face has 3 nodes while its `face_types` entry says
/// `Quad4` — the per-face stride is unusable.
#[test]
fn rejects_face_offsets_type_mismatch() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    let n_faces = mesh.n_faces();
    mesh.face_types = Some(vec![ElementType::Quad4; n_faces]);
    let mut offsets: Vec<usize> = (0..=n_faces).map(|f| f * 4).collect();
    offsets[1] = 3; // face 0 now claims 3 nodes
    mesh.face_offsets = Some(offsets);

    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem(&mut buf, &scratch_2d(), Some(&mesh))
        .expect_err("offset/type mismatch must be refused");
    let msg = err.to_string();
    println!("[face-type vs face_offsets mismatch] {msg}");
    assert!(msg.contains("boundary face 0"), "message should name the face: {msg}");
    assert!(msg.contains("face_offsets"), "{msg}");
    assert!(buf.is_empty());
}

/// The `toroid` miniapp bug in isolation: `elem_type = Hex8` while `conn`
/// holds 6-node wedges.  The old writer read this as `conn.len() / 8` hexes and
/// emitted a mesh with 6 CUBE elements and an empty boundary section.
#[test]
fn rejects_element_type_connectivity_mismatch() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    assert_eq!(mesh.elem_type, ElementType::Hex8);
    // Replace the 8-node hex connectivity with a 6-node wedge.
    mesh.conn = vec![0, 1, 2, 4, 5, 6];
    mesh.elem_tags = vec![1];
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.face_type = ElementType::Quad4;

    let mut buf: Vec<u8> = Vec::new();
    let err = write_mfem(&mut buf, &scratch_2d(), Some(&mesh))
        .expect_err("elem_type/conn mismatch must be refused");
    let msg = err.to_string();
    println!("[elem_type vs conn mismatch] {msg}");
    assert!(msg.contains("needs 8 nodes per element"), "{msg}");
    assert!(
        buf.is_empty(),
        "a rejected mesh must not emit a single byte (got {} bytes)",
        buf.len()
    );
}

/// `elem_types` without `elem_offsets` cannot be located at all.
#[test]
fn rejects_mixed_elements_without_offsets() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    mesh.elem_types = Some(vec![ElementType::Hex8; mesh.n_elems()]);

    let err = write_mfem(&mut Vec::<u8>::new(), &scratch_2d(), Some(&mesh))
        .expect_err("elem_types without elem_offsets must be refused");
    assert!(err.to_string().contains("elem_offsets"), "{err}");
}

/// The written file must use **0-based** vertex indices, like MFEM's own
/// `Mesh::Print` (equivalently: the ids must be direct indices into the
/// `vertices` array).  MFEM's `Mesh::ReadElementWithoutAttr` reads them
/// verbatim, so 1-based ids shift the whole connectivity and MFEM aborts with
/// "Invalid mesh topology" (or overruns its vertex array).
#[test]
fn writes_zero_based_indices() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let text = write_to_string(&mesh);

    // `#` comment lines are optional in the format; skip them.
    let body: Vec<&str> = text.lines().filter(|l| !l.trim_start().starts_with('#')).collect();
    let n_vert: usize = {
        let i = body.iter().position(|l| l.trim() == "vertices").unwrap();
        body[i + 1].trim().parse().unwrap()
    };

    let mut min_idx = usize::MAX;
    let mut max_idx = 0usize;
    let mut in_elem_or_bnd = false;
    let mut skip = 0usize;
    for line in &body {
        let t = line.trim();
        if t == "elements" || t == "boundary" {
            in_elem_or_bnd = true;
            skip = 1; // the count line follows
            continue;
        }
        if t == "vertices" || t == "dimension" {
            in_elem_or_bnd = false;
            continue;
        }
        if !in_elem_or_bnd {
            continue;
        }
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let nums: Vec<usize> = t.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        // `<attr> <geom> <v0> ...`: the vertices are the tail.  Blank lines and
        // the `dim` value line have no vertex ids.
        if nums.len() < 3 {
            continue;
        }
        for v in &nums[2..] {
            min_idx = min_idx.min(*v);
            max_idx = max_idx.max(*v);
        }
    }

    assert_eq!(min_idx, 0, "the file must reference vertex 0 (0-based indices)");
    assert_eq!(max_idx, n_vert - 1, "the highest id must be n_vert - 1");
}

/// A well-formed mesh must still be written, and re-reading it must not depend
/// on the file name: sanity that the new validation is not a blanket refusal.
#[test]
fn valid_mesh_writes_a_file() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cube.mesh");
    write_mfem_file_3d(&path, &mesh).expect("valid mesh must be written");
    assert!(path.exists());
    let back = read_mfem_file(&path).unwrap().mesh3d.unwrap();
    assert_eq!(back.n_elems(), mesh.n_elems());
    assert_eq!(back.n_faces(), mesh.n_faces());
}
