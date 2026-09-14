//! D154: the `MFEM INLINE mesh` reader must be 1:1 with MFEM's `Make2D` /
//! `Make3D` element numbering.
//!
//! `ReadInlineMesh` (`mesh/mesh_readers.cpp:1355`) is a thin front end: it
//! parses the keywords and then calls `Mesh::Make2D(nx, ny, type, sx, sy,
//! generate_edges, true)` for `type = tri|quad` or `Make3D(..., true)` for the
//! 3-D types (`mesh/mesh.cpp:4388` / `:3841`).
//!
//! The `tri` branch was the one that was **not** isomorphic: it was generated
//! with `Mesh::unit_square_tri(max(nx, ny))`, whose cell split is the
//! *anti*-diagonal `(n0,n1,n3)+(n1,n2,n3)`, whereas `Make2D`'s triangle branch
//! numbers elements row-major and splits every cell along the main diagonal:
//!
//! ```text
//!   elem[2k]   = { i + j*m,  i+1 + (j+1)*m,  i + (j+1)*m   }   // (v0,v2,v3)
//!   elem[2k+1] = { i + j*m,  i+1 + j*m,      i+1 + (j+1)*m }   // (v0,v1,v2)
//! ```
//!
//! (with `m = nx+1`).  The quad and hex branches already follow MFEM's
//! space-filling-curve element order; only the triangle branch ignores
//! `sfc_ordering`, because `Make2D` implements the SFC order for quadrilateral
//! elements only — the equivalent of the quad branch's `hilbert_sfc_2d` call
//! must therefore *not* be applied to triangles.
//!
//! The ground truth below is the output of a C++ probe built against serial
//! MFEM 4.10:
//!
//! ```text
//!   Mesh m("../../data/inline-tri.mesh", 1, 1, false);   // the miniapp's call
//!   for (i) print m.GetElement(i)->GetVertices()
//! ```
//!
//! `Mesh(..., refine = 1, ...)` also runs `FinalizeTriMesh` →
//! `MarkTriMeshForRefinement` (`mesh/mesh.cpp:2588`), which rotates every
//! triangle so its longest edge is `(v0,v1)` — visible in the C++ reference as
//! `elem 1 = {6,0,1}` (the unrotated second triangle of cell 0 is `{0,1,6}`).

use fem_io::mfem::read_mfem;
use fem_mesh::{Mesh, MeshTopology};

/// `data/inline-tri.mesh`: `type = tri`, `nx = ny = 4`, `sx = sy = 1`.
fn inline_tri_text() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/inline-tri.mesh"
    ))
    .expect("data/inline-tri.mesh")
}

fn read_inline(text: &str) -> Mesh<2> {
    let file = read_mfem(std::io::Cursor::new(text.as_bytes().to_vec())).expect("read_mfem");
    file.mesh2d.expect("inline-tri.mesh is 2-D")
}

fn elements(mesh: &Mesh<2>) -> Vec<Vec<u32>> {
    (0..mesh.n_elems() as u32)
        .map(|e| mesh.element_nodes(e).to_vec())
        .collect()
}

fn boundary(mesh: &Mesh<2>) -> Vec<Vec<u32>> {
    (0..mesh.n_faces() as u32)
        .map(|f| mesh.bface_nodes(f).to_vec())
        .collect()
}

/// C++ (MFEM 4.10) element table of `data/inline-tri.mesh`, from the probe
/// dump — includes the `MarkTriMeshForRefinement` rotations.
const CPP_ELEMS: [[u32; 3]; 32] = [
    [0, 6, 5], [6, 0, 1], [1, 7, 6], [7, 1, 2],
    [2, 8, 7], [8, 2, 3], [3, 9, 8], [9, 3, 4],
    [5, 11, 10], [11, 5, 6], [6, 12, 11], [12, 6, 7],
    [7, 13, 12], [13, 7, 8], [8, 14, 13], [14, 8, 9],
    [10, 16, 15], [16, 10, 11], [11, 17, 16], [17, 11, 12],
    [12, 18, 17], [18, 12, 13], [13, 19, 18], [19, 13, 14],
    [15, 21, 20], [21, 15, 16], [16, 22, 21], [22, 16, 17],
    [17, 23, 22], [23, 17, 18], [18, 24, 23], [24, 18, 19],
];

/// C++ (MFEM 4.10) boundary table: bottom attr 1, right attr 2, top attr 3,
/// left attr 4, each in `Make2D`'s index order.
const CPP_BDR: [(i32, [u32; 2]); 16] = [
    (1, [0, 1]), (1, [1, 2]), (1, [2, 3]), (1, [3, 4]),
    (3, [21, 20]), (3, [22, 21]), (3, [23, 22]), (3, [24, 23]),
    (4, [5, 0]), (4, [10, 5]), (4, [15, 10]), (4, [20, 15]),
    (2, [4, 9]), (2, [9, 14]), (2, [14, 19]), (2, [19, 24]),
];

#[test]
fn inline_tri_matches_mfem_make2d() {
    let mesh = read_inline(&inline_tri_text());

    assert_eq!(mesh.n_elems(), 32, "NE");
    assert_eq!(mesh.n_nodes(), 25, "NV");
    assert_eq!(mesh.n_faces(), 16, "NBE");

    let elems = elements(&mesh);
    println!("C++ elem 0 = {:?}", CPP_ELEMS[0]);
    println!("Rust elem 0 = {:?}", elems[0]);
    assert_eq!(elems[0], CPP_ELEMS[0].to_vec(), "elem 0");
    for (e, (got, want)) in elems.iter().zip(CPP_ELEMS.iter()).enumerate() {
        assert_eq!(got, &want.to_vec(), "elem {e}");
    }

    // Vertex coordinates: row-major grid on [0, sx] × [0, sy].
    for j in 0..5usize {
        for i in 0..5usize {
            let n = j * 5 + i;
            assert_eq!(mesh.coords[2 * n], i as f64 / 4.0, "vertex {n} x");
            assert_eq!(mesh.coords[2 * n + 1], j as f64 / 4.0, "vertex {n} y");
        }
    }

    let bfaces = boundary(&mesh);
    let tags: Vec<i32> = (0..mesh.n_faces() as u32)
        .map(|f| mesh.face_tags[f as usize] as i32)
        .collect();
    for (f, (tag, verts)) in CPP_BDR.iter().enumerate() {
        assert_eq!(tags[f], *tag, "bdr {f} attribute");
        assert_eq!(&bfaces[f], &verts.to_vec(), "bdr {f} vertices");
    }
}

/// The same `Make2D` generator with `nx != ny`: the old
/// `unit_square_tri(max(nx, ny))` path silently built a `max(nx,ny)²` square,
/// so both the element count and the domain were wrong.
#[test]
fn inline_tri_honours_nx_ny() {
    let text = "MFEM INLINE mesh v1.0\n\ntype = tri\nnx = 3\nny = 2\nsx = 2.0\nsy = 1.0\n";
    let mesh = read_inline(text);

    assert_eq!(mesh.n_elems(), 2 * 3 * 2, "NE = 2*nx*ny");
    assert_eq!(mesh.n_nodes(), 4 * 3, "NV = (nx+1)*(ny+1)");
    assert_eq!(mesh.n_faces(), 2 * 3 + 2 * 2, "NBE = 2*nx + 2*ny");

    // Row-major vertices on [0, 2] × [0, 1].
    for j in 0..3usize {
        for i in 0..4usize {
            let n = j * 4 + i;
            assert_eq!(mesh.coords[2 * n], i as f64 / 3.0 * 2.0, "vertex {n} x");
            assert_eq!(mesh.coords[2 * n + 1], j as f64 / 2.0, "vertex {n} y");
        }
    }
    // First cell, before the MarkTriMeshForRefinement rotation: both triangles
    // share the main diagonal (v0 = 0, v2 = 1+4 = 5) of the quad {0,1,5,4}.
    // The rotation only permutes an element's local order, so compare as a set.
    let elems = elements(&mesh);
    assert_eq!(sorted(elems[0].clone()), sorted(vec![0, 5, 4]), "elem 0 set");
    assert_eq!(sorted(elems[1].clone()), sorted(vec![0, 1, 5]), "elem 1 set");

    // Every boundary attribute appears; the bottom row is attr 1 and its first
    // segment is (0,1).
    assert_eq!(boundary(&mesh)[0], vec![0, 1]);
}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}
