//! D173 acceptance tests: uniform refinement of a **curved** tetrahedral mesh
//! must keep the high-order geometry — topology, MFEM's vertex numbering and
//! the geometry field itself.
//!
//! Reference: serial MFEM 4.10 `Mesh::UniformRefinement()` on `data/
//! escher-p2.mesh` and `data/escher-p3.mesh` (42 curved tets each).  The
//! fixtures `curved_tet_escher_p2_rs1.mesh` / `curved_tet_escher_p3_rs1.mesh`
//! are the C++ runs' own `-rs 1` outputs, copied verbatim (8-digit print):
//!
//! ```text
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem410_ser tmp/round36_tet/probe.cpp \
//!     -o $HOME/work/r36c/probe -L$HOME/mfem410_ser -lmfem -lm
//! wsl $HOME/work/r36c/probe data/escher-p2.mesh 1 parent.mesh child.mesh
//! ```
//! (the probe loads with the constructor defaults `generate_edges = 1,
//! refine = 1`, i.e. the canonical `MarkTetMeshForRefinement`-at-load pipeline;
//! fem-rs's reader canonicalizes the same way on read).
//!
//! What this pins (the halves of MFEM's `Mesh::UniformRefinement` on a curved
//! tet mesh):
//!
//! 1. **Geometry values** (`amr::curved_tet`): every child's `nodes` are the
//!    parent's order-`p` field evaluated on the child's reference domain
//!    (MFEM's 16 `tet_children` point matrices — corner children 0..4,
//!    interior children `4·(rt+1)+k`), and the fine vertices sit on the parent
//!    geometry (`UpdateNodes` → `SetVerticesFromNodes`).  Without it the
//!    refined mesh silently comes out straight-sided.
//! 2. **Vertex numbering** (`MfemTetRefineIds`): MFEM's tet split creates only
//!    edge-midpoint vertices, laid out as `[coarse | oedge + e2v[E]]` with the
//!    `GetVertexToVertexTable` edges re-sorted per vertex row (not the
//!    first-touch order of the hex/wedge splits), so the refined file's
//!    `vertices`/`elements`/`boundary` sections match MFEM's section by
//!    section.
//! 3. **Child order**: MFEM emits the corner children with the coarse vertex
//!    at the child's *own* reference vertex (the historical fem-rs order is a
//!    mirrored copy of the same children), and the boundary triangles as
//!    corner 0, center, corner 1, corner 2.
//! 4. **rt selection**: the octahedron-diagonal refinement type is evaluated
//!    on the *isoparametric* Jacobian at the reference-tet center for curved
//!    meshes (this exposed the `mfem_kernels::eigenvalues2s` `copysign` swap,
//!    fixed here).
//!
//! Family note (p3): `escher-p3.mesh` declares the legacy `Cubic` collection
//! (closed-uniform node lattice).  MFEM refines it as a `Cubic` space, so its
//! fine dofs sample each child patch at closed-uniform parameters, while
//! fem-rs re-expresses the geometry in the Gauss-Lobatto family
//! (`H1_3D_P3` — the same piecewise-polynomial map, sampled elsewhere).  The
//! p3 dof *values* therefore differ between the files by construction; the
//! test compares the geometry **as a map** instead (isoparametric positions at
//! common reference points) and pins topology/counts exactly.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::{refine_uniform_3d, Mesh};

const PARENT_P2: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/escher-p2.mesh");
const PARENT_P3: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/escher-p3.mesh");
const CPP_P2_RS1: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_escher_p2_rs1.mesh");
const CPP_P3_RS1: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curved_tet_escher_p3_rs1.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

/// One parsed `.mesh` file: everything the comparison looks at.
struct Parsed {
    /// `(geom type, connectivity)` per element, in file order.
    elems: Vec<(u32, Vec<u32>)>,
    /// `(geom type, connectivity)` per boundary element, in file order.
    bdr: Vec<(u32, Vec<u32>)>,
    /// `nodes` section dofs as `(x, y, z)`, in FES dof order (the file's
    /// `Ordering` is normalized away).
    nodes: Vec<[f64; 3]>,
    /// The `nodes` section's FE collection name.
    fec: String,
}

/// Minimal MFEM v1.0 mesh reader for the fixtures (tet meshes with an H1
/// `nodes` section written by MFEM itself, either `Ordering`).
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
    let _n: usize = it.next().expect("vertex count").trim().parse().expect("int");
    // A curved mesh stores its geometry in `nodes`; the `vertices` section is
    // just the count followed by a blank line (`Mesh::Print`).

    loop {
        let l = it.next().expect("nodes section").trim().to_string();
        if l == "nodes" {
            break;
        }
    }
    let mut header = Vec::new();
    for h in it.by_ref() {
        header.push(h.trim().to_string());
        if h.trim().starts_with("Ordering:") {
            break;
        }
    }
    let field = |prefix: &str| -> String {
        header
            .iter()
            .find_map(|h| h.strip_prefix(prefix).map(|v| v.trim().to_string()))
            .unwrap_or_default()
    };
    let fec = field("FiniteElementCollection:");
    let vdim: usize = field("VDim:").parse().expect("vdim");
    let ordering: usize = field("Ordering:").parse().expect("ordering");
    let scalars: Vec<f64> = it
        .flat_map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<f64>().expect("node value"))
                .collect::<Vec<f64>>()
        })
        .collect();
    let ndofs = scalars.len() / vdim;
    let mut nodes = Vec::with_capacity(ndofs);
    if ordering == 0 {
        // byNODES: component-major ([x0..xn, y0..yn, z0..zn]).
        for d in 0..ndofs {
            nodes.push([
                scalars[d],
                scalars[ndofs + d],
                scalars[2 * ndofs + d],
            ]);
        }
    } else {
        // byVDIM: dof-major ((x, y, z) per dof).
        for c in scalars.chunks_exact(3) {
            nodes.push([c[0], c[1], c[2]]);
        }
    }
    Parsed { elems, bdr, nodes, fec }
}

#[test]
fn d173_refined_curved_tet_p2_matches_mfem_file() {
    let file = read_mfem_file(PARENT_P2).expect("read parent");
    let parent: Mesh<3> = file.mesh3d.expect("3D mesh");
    assert_eq!(parent.n_elems(), 42);
    assert!(parent.geometry.is_some(), "parent must carry curved geometry");
    assert_eq!(parent.geom_order(), 2);

    let fine = refine_uniform_3d(&parent);
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 2, "geom_order must survive uniform refinement");
    assert_eq!(g.nodes_per_elem, 10, "(p+1)(p+2)(p+3)/6 dofs per order-2 tet");
    assert_eq!(g.conn.len(), fine.n_elems() as usize * 10);
    // MFEM's split creates only edge-midpoint vertices: NV + NE = 26 + 91.
    assert_eq!(fine.n_elems(), 336);
    assert_eq!(fine.n_nodes(), 117, "edge-midpoint vertices only");
    assert_eq!(fine.n_faces(), 192, "every coarse boundary face splits into 4");

    // Whole-file comparison against MFEM's own output.
    let want = parse_mesh(CPP_P2_RS1);
    let out_path = std::env::temp_dir().join("d173_tet_p2_refined.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &fine, NodesSpace::Continuous)
        .expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));

    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
    assert_eq!(got.fec, want.fec, "nodes FE collection");
    assert_eq!(got.nodes.len(), want.nodes.len(), "nodes dof count");
    for (e, (gtype, conn)) in want.elems.iter().enumerate() {
        assert_eq!(*gtype, got.elems[e].0, "element {e}: geometry type");
        assert_eq!(
            got.elems[e].1, *conn,
            "element {e}: connectivity disagrees (vertex numbering or child order)"
        );
    }
    for (f, (gtype, conn)) in want.bdr.iter().enumerate() {
        assert_eq!(got.bdr[f].0, *gtype, "boundary {f}: geometry type");
        assert_eq!(got.bdr[f].1, *conn, "boundary {f}: connectivity disagrees");
    }
    let mut max_dn = 0.0_f64;
    for (v, &w) in got.nodes.iter().zip(want.nodes.iter()) {
        for k in 0..3 {
            max_dn = max_dn.max((v[k] - w[k]).abs());
        }
    }
    assert!(max_dn <= TOL, "nodes values deviate by {max_dn:e} (tol {TOL:e})");
    eprintln!(
        "escher-p2 rs1: 336/336 elements, 192/192 boundary faces byte-matched; \
         nodes max |Δ| {max_dn:e}"
    );
}

#[test]
fn d173_refined_curved_tet_p3_matches_mfem_topology_and_map() {
    let file = read_mfem_file(PARENT_P3).expect("read parent");
    let parent: Mesh<3> = file.mesh3d.expect("3D mesh");
    assert_eq!(parent.n_elems(), 42);
    assert_eq!(parent.geom_order(), 3);

    let fine = refine_uniform_3d(&parent);
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 3);
    assert_eq!(g.nodes_per_elem, 20, "(p+1)(p+2)(p+3)/6 dofs per order-3 tet");
    assert_eq!(fine.n_elems(), 336);
    assert_eq!(fine.n_nodes(), 117);

    // Topology against MFEM's own output (the rt selection must agree on
    // every parent — it drives the interior child connectivity).
    let want = parse_mesh(CPP_P3_RS1);
    let out_path = std::env::temp_dir().join("d173_tet_p3_refined.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &fine, NodesSpace::Continuous)
        .expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
    assert_eq!(got.nodes.len(), want.nodes.len(), "nodes dof count");
    for (e, (gtype, conn)) in want.elems.iter().enumerate() {
        assert_eq!(*gtype, got.elems[e].0, "element {e}: geometry type");
        assert_eq!(got.elems[e].1, *conn, "element {e}: connectivity disagrees");
    }
    for (f, (gtype, conn)) in want.bdr.iter().enumerate() {
        assert_eq!(got.bdr[f].0, *gtype, "boundary {f}: geometry type");
        assert_eq!(got.bdr[f].1, *conn, "boundary {f}: connectivity disagrees");
    }

    // The dof *values* legitimately differ: MFEM keeps the legacy `Cubic`
    // (closed-uniform) family — its fine dofs sample each child patch at
    // closed-uniform parameters — while fem-rs re-expresses the geometry in
    // the Gauss-Lobatto family.  Compare the geometry as a map instead: both
    // meshes' isoparametric maps must agree at common reference points.
    assert_eq!(want.fec, "Cubic", "MFEM refines the legacy family in place");
    assert_eq!(got.fec, "H1_3D_P3", "fem-rs writes the GLL family");

    let cpp_fine_file = read_mfem_file(CPP_P3_RS1).expect("read C++ fixture");
    let cpp_fine: Mesh<3> = cpp_fine_file.mesh3d.expect("3D mesh");
    // Read our own written file back through the same reader: both sides then
    // carry the reader's canonical tet orientation and a geometry table built
    // for it, so the children line up element by element.
    let ours_file = read_mfem_file(out_path.to_str().expect("temp path")).expect("read ours");
    let ours: Mesh<3> = ours_file.mesh3d.expect("3D mesh");
    assert_eq!(ours.n_elems(), cpp_fine.n_elems());
    let samples: [[f64; 3]; 8] = [
        [0.25, 0.25, 0.25],
        [0.2, 0.3, 0.1],
        [0.4, 0.3, 0.2],
        [0.1, 0.1, 0.1],
        [0.5, 0.2, 0.2],
        [0.2, 0.5, 0.2],
        [0.2, 0.2, 0.5],
        [0.1, 0.4, 0.3],
    ];
    let mut max_d = 0.0_f64;
    for e in 0..ours.n_elems() {
        for xi in samples {
            let (_, _, xp) = ours.element_jacobian(e as u32, &xi);
            let (_, _, want_xp) = cpp_fine.element_jacobian(e as u32, &xi);
            for k in 0..3 {
                max_d = max_d.max((xp[k] - want_xp[k]).abs());
            }
        }
    }
    assert!(
        max_d <= 1e-11,
        "refined geometry maps deviate by {max_d:e} (tol 1e-11)"
    );
    eprintln!(
        "escher-p3 rs1: 336/336 elements byte-matched; geometry map max |Δ| {max_d:e}"
    );
}

/// The p2 refinement's vertex numbering is MFEM's `[coarse | oedge + e2v[E]]`
/// layout: the child connectivity references exactly the vertex ids MFEM's
/// `UniformRefinement3D_base` computes (checked element-by-element in the
/// file comparison above); this test pins the count-level invariants of the
/// numbering gate itself.
#[test]
fn d173_canonical_ids_are_dense_and_geometry_sharing_holds() {
    let file = read_mfem_file(PARENT_P2).expect("read parent");
    let parent: Mesh<3> = file.mesh3d.expect("3D mesh");
    let fine = refine_uniform_3d(&parent);
    let g = fine.geometry.as_ref().expect("geometry kept");

    // Dense id space: NV + NE vertices, every geometry dof id < that plus the
    // created edge/face/interior geometry dofs.
    assert_eq!(fine.n_nodes(), 117);
    assert!(g.n_nodes > fine.n_nodes(), "geometry dofs beyond the vertices");

    // Every geometry dof id is referenced and within range.
    let mut used = vec![false; g.n_nodes];
    for &n in g.conn.iter() {
        used[n as usize] = true;
    }
    assert!(used.iter().all(|&u| u), "every geometry node must be referenced");
    assert_eq!(g.conn.len(), fine.n_elems() as usize * g.nodes_per_elem);
}
