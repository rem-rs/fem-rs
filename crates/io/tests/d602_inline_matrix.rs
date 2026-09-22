//! D582 (round 60): the `MFEM INLINE mesh v1.0` reader must be 1:1 with
//! MFEM 4.10's `Mesh::ReadInlineMesh` (`mesh/mesh_readers.cpp:1355`) for
//! **every** element type, including the `pyramid` arm added here and the
//! `tet` / `hex` / `wedge` boundary tables corrected here.
//!
//! Ground truth: `fixtures/d602_inline_mfem.txt`, dumped by
//! `tmp/d602/d602_inline_probe.cpp` against serial MFEM 4.10
//! (`$HOME/mfem410_ser`) via `Mesh::Load` on the exact same INLINE texts.
//! `Mesh::Load` = `Loader` + `Finalize(refine=1)` (mesh.hpp:823), so tri/tet
//! tables carry the `MarkTri/TetMeshForRefinement` rotations and the tables
//! are otherwise the raw `Make2D`/`Make3D` output:
//!
//! * `tet`: row-major cells, `AddHexAsTets` split (mesh.cpp:2243); the old
//!   arm delegated to `Mesh::unit_cube_tet`, whose Freudenthal split numbers
//!   the six tets of each cell in a different order than MFEM.
//! * `hex` / `wedge` / `tet` / `pyramid` boundaries: Make3D's six loops in
//!   Make3D order (bottom 1, top 6, left 5, right 3, front 2, back 4) with
//!   Make3D's loop nesting (`front`/`back` are `for (x) for (z)`); quads are
//!   split into {(0,1,2),(0,2,3)} for the tet bottom/top/sides and the
//!   wedge bottom/top, and stay quads for hex/pyramid and the wedge sides.
//!   The old tet arm used `unit_cube_tet`'s boundary (different attributes
//!   and attribute order) and the old wedge arm interleaved left/right and
//!   front/back with shifted attributes (2↔3, 4↔2).
//! * `pyramid` (new arm): `AddHexAsPyramids` (mesh.cpp:2280) — 6 pyramids
//!   per cell around a NEW apex vertex at the cell centre (`VTXP`,
//!   mesh.cpp:3882), boundary stays quadrilateral.

use fem_io::mfem::read_mfem;
use fem_mesh::Mesh;

/// Parsed `fixtures/d602_inline_mfem.txt` block.
struct Ref {
    name: String,
    elems: Vec<(i32, Vec<u32>)>,
    bdr: Vec<(i32, Vec<u32>)>,
}

fn parse_ref() -> Vec<Ref> {
    let mut out = Vec::new();
    let mut cur: Option<Ref> = None;
    for line in include_str!("fixtures/d602_inline_mfem.txt").lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("TYPE") => {
                if let Some(r) = cur.take() {
                    out.push(r);
                }
                cur = Some(Ref { name: t[1].to_string(), elems: Vec::new(), bdr: Vec::new() });
            }
            Some("E") => {
                let attr: i32 = t[1].parse().unwrap();
                let verts: Vec<u32> = t[3..].iter().map(|s| s.parse().unwrap()).collect();
                cur.as_mut().unwrap().elems.push((attr, verts));
            }
            Some("B") => {
                let attr: i32 = t[1].parse().unwrap();
                let verts: Vec<u32> = t[3..].iter().map(|s| s.parse().unwrap()).collect();
                cur.as_mut().unwrap().bdr.push((attr, verts));
            }
            _ => {}
        }
    }
    if let Some(r) = cur.take() {
        out.push(r);
    }
    out
}

/// The INLINE texts — must match `d602_inline_probe.cpp` byte for byte.
fn inline_text(name: &str) -> String {
    match name {
        "tri_2x2" => "MFEM INLINE mesh v1.0\n\ntype = tri\nnx = 2\nny = 2\nsx = 2.0\nsy = 1.0\n".into(),
        "quad_2x2" => "MFEM INLINE mesh v1.0\n\ntype = quad\nnx = 2\nny = 2\nsx = 2.0\nsy = 1.0\n".into(),
        "tet_2x1x2" => {
            "MFEM INLINE mesh v1.0\n\ntype = tet\nnx = 2\nny = 1\nnz = 2\nsx = 2.0\nsy = 1.0\nsz = 0.5\n".into()
        }
        "hex_2x1x2" => {
            "MFEM INLINE mesh v1.0\n\ntype = hex\nnx = 2\nny = 1\nnz = 2\nsx = 2.0\nsy = 1.0\nsz = 0.5\n".into()
        }
        "wedge_2x1x2" => {
            "MFEM INLINE mesh v1.0\n\ntype = wedge\nnx = 2\nny = 1\nnz = 2\nsx = 2.0\nsy = 1.0\nsz = 0.5\n".into()
        }
        "pyramid_2x1x2" => {
            "MFEM INLINE mesh v1.0\n\ntype = pyramid\nnx = 2\nny = 1\nnz = 2\nsx = 2.0\nsy = 1.0\nsz = 0.5\n".into()
        }
        "pyramid_1x1x1" => {
            "MFEM INLINE mesh v1.0\n\ntype = pyramid\nnx = 1\nny = 1\nnz = 1\nsx = 1.0\nsy = 1.0\nsz = 1.0\n".into()
        }
        other => panic!("no inline text for {other}"),
    }
}

fn nv_of(elems: &[(i32, Vec<u32>)]) -> usize {
    elems.iter().flat_map(|(_, v)| v.iter()).copied().max().unwrap() as usize + 1
}

/// Element/boundary table comparison shared by the 2-D and 3-D containers.
fn check_tables<const D: usize>(
    name: &str,
    mesh: &Mesh<D>,
    elems: &[(i32, Vec<u32>)],
    bdr: &[(i32, Vec<u32>)],
    tags: &[i32],
) {
    use fem_mesh::MeshTopology as _;
    assert_eq!(mesh.n_elems(), elems.len(), "{name}: NE");
    assert_eq!(mesh.n_nodes(), nv_of(elems), "{name}: NV");
    assert_eq!(mesh.n_faces(), bdr.len(), "{name}: NBE");
    for (e, (attr, verts)) in elems.iter().enumerate() {
        assert_eq!(mesh.element_nodes(e as u32), verts.as_slice(), "{name}: elem {e}");
        assert_eq!(mesh.element_tag(e as u32), *attr, "{name}: elem {e} attribute");
    }
    for (f, (attr, verts)) in bdr.iter().enumerate() {
        assert_eq!(mesh.bface_nodes(f as u32), verts.as_slice(), "{name}: bdr {f}");
        assert_eq!(tags[f], *attr, "{name}: bdr {f} attribute");
    }
}

#[test]
fn inline_all_types_match_mfem() {
    let refs = parse_ref();
    assert_eq!(refs.len(), 7, "seven ground-truth blocks");
    for r in &refs {
        let text = inline_text(&r.name);
        let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec()))
            .unwrap_or_else(|e| panic!("{}: {e}", r.name));
        if let Some(mesh) = file.mesh2d.as_ref() {
            check_tables::<2>(&r.name, mesh, &r.elems, &r.bdr, &mesh.face_tags);
        } else if let Some(mesh) = file.mesh3d.as_ref() {
            check_tables::<3>(&r.name, mesh, &r.elems, &r.bdr, &mesh.face_tags);
        } else {
            panic!("{}: no mesh container", r.name);
        }
    }
}

#[test]
fn inline_segment_is_rejected_clearly() {
    let text = "MFEM INLINE mesh v1.0\n\ntype = segment\nnx = 4\nsx = 1.0\n";
    match read_mfem(std::io::Cursor::new(text.as_bytes().to_vec())) {
        Err(err) => {
            let msg = format!("{err}");
            assert!(msg.contains("segment") || msg.contains("ny"), "mentions the gap: {msg}");
        }
        Ok(_) => panic!("segment inline mesh must be rejected (no 1-D MfemFile container)"),
    }
}

#[test]
fn inline_tet_unit_cell_matches_d559_mesh() {
    use fem_mesh::MeshTopology as _;
    // The D559 probe mesh (MFEM MakeCartesian3D(1,1,1,TETRAHEDRON), saved)
    // uses the same AddHexAsTets split; after MFEM's Finalize marking its
    // element SETS must equal the raw inline split for one cell.
    let text = "MFEM INLINE mesh v1.0\n\ntype = tet\nnx = 1\nny = 1\nnz = 1\nsx = 1.0\nsy = 1.0\nsz = 1.0\n";
    let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec())).expect("inline tet");
    let mesh = file.mesh3d.expect("3-D");
    let mut got: Vec<Vec<u32>> = (0..mesh.n_elems() as u32)
        .map(|e| {
            let mut v = mesh.element_nodes(e).to_vec();
            v.sort_unstable();
            v
        })
        .collect();
    got.sort();
    let want_sets: [[u32; 4]; 6] = [
        [7, 0, 3, 1],
        [7, 0, 1, 5],
        [7, 0, 5, 4],
        [7, 0, 2, 3],
        [7, 0, 6, 2],
        [7, 0, 4, 6],
    ];
    let mut want: Vec<Vec<u32>> = want_sets
        .iter()
        .map(|v| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        })
        .collect();
    want.sort();
    assert_eq!(got, want, "unit-cell tet split == d559_tet6.mesh element sets");
}
