//! D166 acceptance test: uniform refinement of a **curved** hexahedral mesh
//! must keep the high-order geometry — topology *and* MFEM's vertex numbering.
//!
//! Reference: serial MFEM 4.10 `miniapps/meshing/toroid.cpp -e 1 -o 3`
//! (generator defaults `-nphi 8`), refined once with `-rs 1`.  The parent
//! fixture `toroid_hex_o3.mesh` is the C++ miniapp's own `-rs 0` output and
//! `toroid_hex_o3_r1.mesh` its `-rs 1` output, both copied verbatim from the
//! C++ run (8-digit print precision):
//!
//! ```text
//! wsl cd $HOME/mfem410_ser/miniapps/meshing && g++ -std=c++17 -O2 \
//!     -I$HOME/mfem410_ser toroid.cpp -o $HOME/work/d34/toroid_cpp \
//!     $HOME/mfem410_ser/miniapps/common/libmfem-common.a $HOME/mfem410_ser/libmfem.a
//! wsl cd $HOME/work/d34 && ./toroid_cpp -e 1 -o 3 -no-vis        # parent
//! wsl cd $HOME/work/d34 && ./toroid_cpp -e 1 -rs 1 -o 3 -no-vis  # refined
//! ```
//!
//! What this pins (the two halves of MFEM's `Mesh::UniformRefinement` on a
//! curved mesh):
//!
//! 1. **Geometry values** (`amr::curved_hex`): every child's `nodes` are the
//!    parent's order-`p` field evaluated on the child's reference domain, and
//!    the fine vertices sit on the parent geometry (`UpdateNodes` →
//!    `SetVerticesFromNodes`).  Without it the refined mesh silently comes out
//!    straight-sided.
//! 2. **Vertex numbering** (`MfemHexRefineIds`): MFEM lays the fine vertices
//!    out as `[coarse vertices | edge midpoints | face centers | body centers]`
//!    with first-touch global edge/face ids, so the refined file's
//!    `vertices`/`elements`/`boundary` sections match MFEM's section by
//!    section, not just up to a node relabeling.
//!
//! The test refines the C++ parent with `refine_uniform_3d`, writes it back
//! out through `fem_io`'s MFEM writer (to a scratch file, the same path the
//! `mesh_toroid` miniapp takes) and compares every section against the C++
//! refined file.  Tolerance 1e-7: both sides print 8 significant digits, so
//! the ~1e-16 arithmetic noise of MFEM's refinement operator is invisible
//! except through re-rounding.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::{refine_uniform_3d, Mesh};

const PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_hex_o3.mesh");
const CPP_REFINED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_hex_o3_r1.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

/// One parsed `.mesh` file: everything the comparison looks at.
struct Parsed {
    /// `(geom type, connectivity)` per element, in file order.
    elems: Vec<(u32, Vec<u32>)>,
    /// `(geom type, connectivity)` per boundary element, in file order.
    bdr: Vec<(u32, Vec<u32>)>,
    /// Vertex coordinates, 3 floats per id.
    verts: Vec<f64>,
    /// `nodes` section values (Ordering: 1 = byVDIM), empty when absent.
    nodes: Vec<f64>,
    /// The `nodes` section's FE collection name (`H1_3D_P3`), empty when absent.
    fec: String,
}

/// Minimal MFEM v1.0 mesh reader for the fixtures (hex meshes with an
/// `H1_3D_P3` `nodes` section written by MFEM itself).
fn parse_mesh(path: &str) -> Parsed {
    let text = std::fs::read_to_string(path).expect("read mesh file");
    let mut it = text.lines().peekable();
    let skip_to = |it: &mut std::iter::Peekable<std::str::Lines>, key: &str| loop {
        let l = it.next().expect("section").trim().to_string();
        if l == key {
            break;
        }
    };

    skip_to(&mut it, "elements");
    let n: usize = it.next().expect("element count").trim().parse().expect("int");
    let mut elems = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<u32> = it
            .next()
            .expect("element row")
            .split_whitespace()
            .map(|t| t.parse().expect("int"))
            .collect();
        elems.push((v[1], v[2..].to_vec())); // (geom type, connectivity)
    }

    skip_to(&mut it, "boundary");
    let n: usize = it.next().expect("boundary count").trim().parse().expect("int");
    let mut bdr = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<u32> = it
            .next()
            .expect("boundary row")
            .split_whitespace()
            .map(|t| t.parse().expect("int"))
            .collect();
        bdr.push((v[1], v[2..].to_vec()));
    }

    skip_to(&mut it, "vertices");
    let n: usize = it.next().expect("vertex count").trim().parse().expect("int");
    let mut verts = Vec::with_capacity(3 * n);
    // MFEM writes the vertex *rows* only for straight-sided meshes; a curved
    // mesh stores its geometry in `nodes` and the section is just the count
    // followed by a blank line (mesh/mesh.cpp `Mesh::Print`).
    loop {
        let l = it.peek().map(|l| l.trim().to_string()).unwrap_or_default();
        if l.is_empty() || l == "nodes" {
            break;
        }
        let v: Vec<f64> = it
            .next()
            .expect("vertex row")
            .split_whitespace()
            .map(|t| t.parse().expect("float"))
            .collect();
        assert_eq!(v.len(), 3, "one vertex per row");
        verts.extend_from_slice(&v);
    }

    // Optional `nodes` section: `nodes` / FiniteElementSpace / … / Ordering: k
    // / blank / values (the file ends after them).
    let mut nodes = Vec::new();
    let mut fec = String::new();
    for l in it.by_ref() {
        if l.trim() == "nodes" {
            loop {
                let h = it.next().expect("FE space header").trim().to_string();
                if let Some(name) = h.strip_prefix("FiniteElementCollection:") {
                    fec = name.trim().to_string();
                }
                if h.starts_with("Ordering:") {
                    break;
                }
            }
            nodes = it
                .flat_map(|l| {
                    l.split_whitespace()
                        .map(|t| t.parse::<f64>().expect("node value"))
                        .collect::<Vec<f64>>()
                })
                .collect();
            break;
        }
    }
    Parsed { elems, bdr, verts, nodes, fec }
}

#[test]
fn d166_refined_curved_toroid_matches_mfem_file() {
    let f = read_mfem_file(PARENT).expect("read parent fixture");
    let parent: Mesh<3> = f.mesh3d.expect("3D parent");
    assert_eq!(parent.n_elems(), 8);
    assert!(parent.geometry.is_some(), "parent must carry curved geometry");
    assert_eq!(parent.geom_order(), 3);

    // Mesh-level invariants of the refinement itself.
    let fine = refine_uniform_3d(&parent);
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 3, "geom_order must survive uniform refinement");
    assert_eq!(g.nodes_per_elem, 64);
    assert_eq!(g.conn.len(), fine.n_elems() as usize * 64);
    assert_eq!(fine.n_elems(), 64);
    assert_eq!(fine.n_nodes(), 144);
    assert_eq!(fine.n_faces(), 128);

    // Whole-file comparison: write the refined mesh exactly like the miniapp
    // does and diff every section against MFEM's own output.
    let want = parse_mesh(CPP_REFINED);
    let out_path = std::env::temp_dir().join("d166_toroid_refined.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &fine, NodesSpace::Continuous)
        .expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&out_path);

    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
    assert_eq!(got.verts.len(), want.verts.len(), "vertex count");
    assert_eq!(got.fec, want.fec, "nodes FE collection");
    assert_eq!(got.nodes.len(), want.nodes.len(), "nodes payload length");

    for (e, (gtype, conn)) in want.elems.iter().enumerate() {
        assert_eq!(*gtype, got.elems[e].0, "element {e}: geometry type");
        assert_eq!(
            got.elems[e].1, *conn,
            "element {e}: connectivity disagrees (vertex numbering or child order)"
        );
    }
    for (f, (gtype, conn)) in want.bdr.iter().enumerate() {
        assert_eq!(*gtype, got.bdr[f].0, "boundary {f}: geometry type");
        assert_eq!(got.bdr[f].1, *conn, "boundary {f}: connectivity disagrees");
    }

    let mut max_dv = 0.0_f64;
    for (v, &w) in got.verts.iter().zip(want.verts.iter()) {
        max_dv = max_dv.max((v - w).abs());
    }
    assert!(max_dv <= TOL, "vertex coordinates deviate by {max_dv:e} (tol {TOL:e})");

    let mut max_dn = 0.0_f64;
    for (v, &w) in got.nodes.iter().zip(want.nodes.iter()) {
        max_dn = max_dn.max((v - w).abs());
    }
    assert!(max_dn <= TOL, "nodes values deviate by {max_dn:e} (tol {TOL:e})");

    eprintln!(
        "toroid hex o3 rs1: 64/64 elements, 128/128 boundary faces byte-matched; \
         max |Δ| vertices {max_dv:e}, nodes {max_dn:e}"
    );
}
