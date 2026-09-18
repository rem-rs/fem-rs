//! D173 acceptance test: uniform refinement of a **curved** wedge (Prism6)
//! mesh must keep the high-order geometry — topology *and* MFEM's vertex
//! numbering *and* MFEM's child order.
//!
//! Reference: serial MFEM 4.10 `miniapps/meshing/toroid.cpp -e 0 -o 3`
//! (generator defaults `-nphi 8`), refined once with `-rs 1`.  The fixture
//! `toroid_wedge_o3_r1.mesh` is the C++ miniapp's own `-rs 1` output, copied
//! verbatim from the C++ run (8-digit print precision):
//!
//! ```text
//! wsl cd $HOME/mfem410_ser/miniapps/meshing && g++ -std=c++17 -O2 \
//!     -I$HOME/mfem410_ser toroid.cpp -o $HOME/work/r35b/toroid_cpp \
//!     $HOME/mfem410_ser/miniapps/common/libmfem-common.a $HOME/mfem410_ser/libmfem.a
//! wsl cd $HOME/work/r35b && ./toroid_cpp -e 0 -rs 1 -o 3 -no-vis  # refined
//! ```
//!
//! The curved **parent** is built in-process exactly like the `mesh_toroid`
//! miniapp builds it (`SetCurvature` → `Transform` → stitch) rather than read
//! back through `fem_io`: the D41 wedge reader numbers its dofs per element
//! first-touch, which (unlike MFEM's entity-wise numbering) diverges from the
//! file's numbering on multi-element wedge meshes, so a read-back curved wedge
//! parent used to carry scrambled non-vertex geometry dofs.
//!
//! **That gap is closed** (round 44 ③/④, D295–D314): the reader now bridges the
//! file's dof rows into `PrismPk`'s layer-major geometry-table order — the
//! layout every in-memory consumer evaluates (`geo_ref_elem`,
//! `element_jacobian`, `curved_prism`, `prism_nodes_dof_values`) — and the
//! writer re-emits MFEM's own bytes
//! (`crates/io/tests/d190_wedge_curved_nodes_roundtrip.rs` pins
//! `curved_prism_write_matches_cpp_save_byte_for_byte`, including on the
//! `toroid_wedge_o3` fixtures used here).  The construction below stays
//! in-process for the miniapp-level comparison, and
//! `read_back_parent_matches_in_process_parent` (bottom of this file) asserts
//! the read-back parent is equivalent to it — so the in-process path is a
//! convenience, not a workaround.
//!
//! What this pins (the two halves of MFEM's `Mesh::UniformRefinement` on a
//! curved wedge mesh):
//!
//! 1. **Geometry values** (`amr::curved_prism`): every child's `nodes` are the
//!    parent's order-`p` field evaluated on the child's reference domain
//!    (MFEM's `pri_children` point matrices, including the rotated center
//!    children), and the fine vertices sit on the parent geometry
//!    (`UpdateNodes` → `SetVerticesFromNodes`).  Without it the refined mesh
//!    silently comes out straight-sided.
//! 2. **Vertex numbering** (`MfemPrismRefineIds`): MFEM's wedge split creates
//!    *no* triangular face centers and *no* body centers — only 9 edge
//!    midpoints and 3 quad-face centers per prism — laid out as
//!    `[coarse | oedge + E | oface + f2qf(F)]` with first-touch global edge
//!    ids and quad-face ranks, so the refined file's
//!    `vertices`/`elements`/`boundary` sections match MFEM's section by
//!    section, not just up to a node relabeling.
//! 3. **Child order**: MFEM emits the 8 children as corner 0, center,
//!    corner 1, corner 2 per layer (the center children with cyclically
//!    rotated node order) — the historical fem-rs straight-side order was
//!    removed by D180 (round 42); the straight path now emits this order too
//!    (`toroid_wedge_straight_refine.rs` pins it on `-o 1`).
//!
//! The test refines the parent with `refine_uniform_3d`, writes it back out
//! through `fem_io`'s MFEM writer (to a scratch file, the same path the
//! `mesh_toroid` miniapp takes) and compares every section against the C++
//! refined file.  Tolerance 1e-7: both sides print 8 significant digits, so
//! the ~1e-15 arithmetic noise of MFEM's refinement operator is invisible
//! except through re-rounding.

use std::collections::HashMap;
use std::f64::consts::PI;

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::{refine_uniform_3d, Mesh, MeshTopology};

const CPP_REFINED: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_wedge_o3_r1.mesh");
const CPP_PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_wedge_o3.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

// ─── The `mesh_toroid` miniapp's construction (toroid.rs, `-e 0 -o 3`) ──────

/// MFEM `Geometry::Constants<WEDGE>::FaceVert` — the local face vertex lists in
/// MFEM's local-face order (`{0,2,1}, {3,4,5}` are the triangles, then the
/// three quads).
const WEDGE_FACES: [[usize; 4]; 5] = [
    [0, 2, 1, usize::MAX],
    [3, 4, 5, usize::MAX],
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [2, 0, 3, 5],
];

/// MFEM `Mesh::FinalizeTopology` for the wedge stack: `GenerateFaces`
/// (first encounter wins) followed by `GenerateBoundaryElements` (faces owned
/// by exactly one element, attribute 1).
fn generate_boundary(mesh: &Mesh<3>) -> (Vec<u32>, Vec<i32>, Vec<ElementType>) {
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut all: Vec<(Vec<u32>, ElementType, u32)> = Vec::new();
    for e in 0..mesh.n_elems() {
        let ns = mesh.elem_nodes(e as u32).to_vec();
        for fv in &WEDGE_FACES {
            let n = if fv[3] == usize::MAX { 3 } else { 4 };
            let key: Vec<u32> = fv[..n].iter().map(|&i| ns[i]).collect();
            let ft = if n == 3 { ElementType::Tri3 } else { ElementType::Quad4 };
            let mut sorted = key.clone();
            sorted.sort_unstable();
            match seen.get(&sorted) {
                Some(&fi) => all[fi].2 += 1,
                None => {
                    seen.insert(sorted, all.len());
                    all.push((key, ft, 1));
                }
            }
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut face_types = Vec::new();
    for (key, ft, count) in all {
        if count == 1 {
            face_conn.extend_from_slice(&key);
            face_tags.push(1);
            face_types.push(ft);
        }
    }
    (face_conn, face_tags, face_types)
}

/// Face table rebuild after [`generate_boundary`] (the miniapp's
/// `set_face_tables`: uniform types collapse to `face_type`, mixed types get
/// offsets).
fn set_face_tables(mesh: &mut Mesh<3>, face_types: Vec<ElementType>) {
    let uniform = face_types.windows(2).all(|w| w[0] == w[1]);
    if uniform {
        mesh.face_type = face_types[0];
        mesh.face_types = None;
        mesh.face_offsets = None;
    } else {
        let mut offsets = Vec::with_capacity(face_types.len() + 1);
        offsets.push(0);
        for t in &face_types {
            offsets.push(offsets.last().unwrap() + t.nodes_per_element());
        }
        mesh.face_type = face_types[0];
        mesh.face_types = Some(face_types);
        mesh.face_offsets = Some(offsets);
    }
}

/// Torus transformation for the wedge cross-section (MFEM `trans`).
fn trans_wedge(x: &[f64], nphi: usize, ns: i32, r: f64, r_maj: f64, nnode: i32) -> [f64; 3] {
    let phi = 2.0 * PI * x[2] / nphi as f64;
    let theta = phi * ns as f64 / nnode as f64;
    let u = (1.5 * (x[0] + x[1]) - 1.0) * r;
    let v = (0.75f64).sqrt() * (x[0] - x[1]) * r;
    [
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.cos(),
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.sin(),
        v * theta.cos() - u * theta.sin(),
    ]
}

/// `Mesh::Transform(trans)` on the curved mesh: every `nodes` dof (and every
/// vertex, keeping the mesh self-consistent) moves through the torus map.
fn apply_transform(mesh: &mut Mesh<3>, f: &impl Fn(&[f64]) -> [f64; 3]) {
    if let Some(g) = mesh.geometry.as_mut() {
        for i in 0..g.n_nodes {
            let x = [g.coords[3 * i], g.coords[3 * i + 1], g.coords[3 * i + 2]];
            let p = f(&x);
            g.coords[3 * i..3 * i + 3].copy_from_slice(&p);
        }
    }
    for i in 0..mesh.n_nodes() {
        let x = [mesh.coords[3 * i], mesh.coords[3 * i + 1], mesh.coords[3 * i + 2]];
        let p = f(&x);
        mesh.coords[3 * i..3 * i + 3].copy_from_slice(&p);
    }
}

/// `Mesh::RemoveInternalBoundaries` with the prism face table: drop the
/// boundary faces shared by two elements (the two stitched end triangles).
fn remove_internal_boundaries(mesh: &mut Mesh<3>) {
    let mut count: HashMap<Vec<u32>, u32> = HashMap::new();
    for e in 0..mesh.n_elems() {
        let ns = mesh.elem_nodes(e as u32).to_vec();
        for fv in &WEDGE_FACES {
            let n = if fv[3] == usize::MAX { 3 } else { 4 };
            let mut k: Vec<u32> = fv[..n].iter().map(|&i| ns[i]).collect();
            k.sort_unstable();
            *count.entry(k).or_insert(0) += 1;
        }
    }
    let off = mesh.face_offsets.clone().unwrap_or_default();
    let mut new_conn = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();
    let mut new_types = Vec::new();
    for f in 0..mesh.n_faces() {
        let o = off[f];
        let nv = mesh.face_type_at(f as u32).nodes_per_element();
        let mut k: Vec<u32> = mesh.face_conn[o..o + nv].to_vec();
        k.sort_unstable();
        if count.get(&k).copied().unwrap_or(0) <= 1 {
            new_conn.extend_from_slice(&mesh.face_conn[o..o + nv]);
            new_tags.push(mesh.face_tags[f]);
            new_types.push(mesh.face_type_at(f as u32));
        }
    }
    mesh.face_conn = new_conn;
    mesh.face_tags = new_tags;
    set_face_tables(mesh, new_types);
}

/// The `mesh_toroid -e 0 -o 3` parent: the torus of 8 wedges, curvature first,
/// transform, then the end stitch (see toroid.rs for the ordering argument).
fn build_parent() -> Mesh<3> {
    let nphi = 8usize;
    let order = 3u8;
    let (r_maj, r_min) = (1.0_f64, 0.2_f64);
    let nnode = 3_i32;

    let mut mesh: Mesh<3> =
        Mesh::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 0.0, 0.0, 0.0, false);
    mesh.conn.clear();
    mesh.elem_tags.clear();
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.coords.clear();
    mesh.elem_type = ElementType::Prism6;

    for i in 0..=nphi {
        let z = i as f64;
        mesh.add_vertex_3d(0.0, 0.0, z);
        mesh.add_vertex_3d(1.0, 0.0, z);
        mesh.add_vertex_3d(0.0, 1.0, z);
    }
    for i in 0..nphi {
        let v: [u32; 6] = [
            3 * i as u32,
            3 * i as u32 + 1,
            3 * i as u32 + 2,
            3 * (i + 1) as u32,
            3 * (i + 1) as u32 + 1,
            3 * (i + 1) as u32 + 2,
        ];
        mesh.add_wedge(&v, 1);
    }

    let (face_conn, face_tags, face_types) = generate_boundary(&mesh);
    mesh.face_conn = face_conn;
    mesh.face_tags = face_tags;
    set_face_tables(&mut mesh, face_types);

    if order > 1 {
        mesh.set_curvature(order);
    }
    apply_transform(&mut mesh, &|x| trans_wedge(x, nphi, 0, r_min, r_maj, nnode));

    // Stitch: the stack's last layer collapses onto the first.
    let nv = mesh.n_nodes();
    let mut v2v = vec![0_i32; nv];
    for i in 0..nv - nnode as usize {
        v2v[i] = i as i32;
    }
    for i in 0..nnode {
        v2v[nv - nnode as usize + i as usize] = i;
    }
    mesh.renumber_vertices(&v2v);
    mesh.remove_unused_vertices();
    remove_internal_boundaries(&mut mesh);
    mesh
}

/// One parsed `.mesh` file: everything the comparison looks at.
struct Parsed {
    /// `(geom type, connectivity)` per element, in file order.
    elems: Vec<(u32, Vec<u32>)>,
    /// `(geom type, connectivity)` per boundary element, in file order.
    bdr: Vec<(u32, Vec<u32>)>,
    /// `nodes` section values (Ordering: 1 = byVDIM).
    nodes: Vec<f64>,
    /// The `nodes` section's FE collection name (`H1_3D_P3`).
    fec: String,
}

/// Minimal MFEM v1.0 mesh reader for the fixtures (pure-wedge meshes with an
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
    let _n: usize = it.next().expect("vertex count").trim().parse().expect("int");
    // A curved mesh stores its geometry in `nodes`; the `vertices` section is
    // just the count followed by a blank line (`Mesh::Print`).

    loop {
        let l = it.next().expect("nodes section").trim().to_string();
        if l == "nodes" {
            break;
        }
    }
    let mut fec = String::new();
    loop {
        let h = it.next().expect("FE space header").trim().to_string();
        if let Some(name) = h.strip_prefix("FiniteElementCollection:") {
            fec = name.trim().to_string();
        }
        if h.starts_with("Ordering:") {
            break;
        }
    }
    let nodes = it
        .flat_map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<f64>().expect("node value"))
                .collect::<Vec<f64>>()
        })
        .collect();
    Parsed { elems, bdr, nodes, fec }
}

#[test]
fn d173_refined_curved_toroid_wedge_matches_mfem_file() {
    let parent = build_parent();
    assert_eq!(parent.n_elems(), 8);
    assert!(parent.geometry.is_some(), "parent must carry curved geometry");
    assert_eq!(parent.geom_order(), 3);

    // The in-process construction must reproduce the C++ `-rs 0` file (the
    // same bytes the `mesh_toroid` miniapp writes) — this is what makes the
    // refined comparison below a *miniapp*-level check.
    let want_parent = parse_mesh(CPP_PARENT);
    let parent_path = std::env::temp_dir().join("d173_toroid_wedge_parent.mesh");
    write_mfem_file_3d_nodes(
        parent_path.to_str().expect("temp path"),
        &parent,
        NodesSpace::Continuous,
    )
    .expect("write parent");
    let got_parent = parse_mesh(parent_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&parent_path);
    assert_eq!(got_parent.elems, want_parent.elems, "parent elements");
    assert_eq!(got_parent.bdr, want_parent.bdr, "parent boundary");
    assert_eq!(got_parent.fec, want_parent.fec, "parent nodes FE collection");
    let mut max_dp = 0.0_f64;
    for (v, &w) in got_parent.nodes.iter().zip(want_parent.nodes.iter()) {
        max_dp = max_dp.max((v - w).abs());
    }
    assert!(max_dp <= TOL, "parent nodes deviate by {max_dp:e} (tol {TOL:e})");

    // Mesh-level invariants of the refinement itself.  MFEM's wedge split
    // gains 9 edge midpoints + 3 quad face centers per prism (no triangular
    // face centers, no body centers); the toroid ring shares edges and quad
    // faces between neighbouring prisms: 24 + 48 + 24 = 96 fine vertices.
    let fine = refine_uniform_3d(&parent);
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 3, "geom_order must survive uniform refinement");
    assert_eq!(g.nodes_per_elem, 40, "(p+1)²(p+2)/2 dofs per order-3 wedge");
    assert_eq!(g.conn.len(), fine.n_elems() as usize * 40);
    assert_eq!(fine.n_elems(), 64);
    assert_eq!(fine.n_nodes(), 96, "no tri-face centers, no body centers");
    assert_eq!(fine.n_faces(), 96, "every coarse quad face splits into 4");

    // Whole-file comparison: write the refined mesh exactly like the miniapp
    // does and diff every section against MFEM's own output.
    let want = parse_mesh(CPP_REFINED);
    let out_path = std::env::temp_dir().join("d173_toroid_wedge_refined.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &fine, NodesSpace::Continuous)
        .expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&out_path);

    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
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
        assert_eq!(got.bdr[f].0, *gtype, "boundary {f}: geometry type");
        assert_eq!(got.bdr[f].1, *conn, "boundary {f}: connectivity disagrees");
    }

    let mut max_dn = 0.0_f64;
    for (v, &w) in got.nodes.iter().zip(want.nodes.iter()) {
        max_dn = max_dn.max((v - w).abs());
    }
    assert!(max_dn <= TOL, "nodes values deviate by {max_dn:e} (tol {TOL:e})");

    eprintln!(
        "toroid wedge o3 rs1: 64/64 elements, 96/96 boundary faces byte-matched; \
         parent nodes max |Δ| {max_dp:e}, refined nodes max |Δ| {max_dn:e}"
    );
}

/// Sanity: the committed parent fixture parses and carries MFEM's FEC name.
#[test]
fn d173_parent_fixture_is_the_cpp_rs0_file() {
    let want = parse_mesh(CPP_PARENT);
    assert_eq!(want.fec, "H1_3D_P3");
    assert_eq!(want.elems.len(), 8);
}

/// D314: the read-back curved wedge parent must carry the **same geometry** as
/// the in-process `mesh_toroid` construction.
///
/// The module doc above used to record the opposite ("read-back curved wedge
/// parent carries scrambled non-vertex geometry dofs"); D295 bridged the
/// reader's rows into `PrismPk`'s layer-major order.  This asserts it on the
/// fixture: per element and slot, the geometry node coordinates of
/// `read_mfem_file(toroid_wedge_o3.mesh)` and of `build_parent()` agree to the
/// fixture's 8-digit print precision, and re-writing the read-back mesh
/// reproduces the same `nodes`/`elements`/`boundary` sections.
#[test]
fn d314_read_back_parent_matches_in_process_parent() {
    let built = build_parent();
    let readback = fem_io::mfem::read_mfem_file(CPP_PARENT)
        .expect("read toroid_wedge_o3.mesh")
        .mesh3d
        .expect("3-D mesh");

    assert_eq!(readback.n_elems(), built.n_elems(), "element count");
    assert_eq!(readback.geom_order(), built.geom_order(), "geometry order");
    let (gb, gr) = (
        built.geometry.as_ref().expect("in-process parent carries geometry"),
        readback.geometry.as_ref().expect("read-back parent carries geometry"),
    );
    assert_eq!(gr.order, gb.order);
    assert_eq!(gr.nodes_per_elem, gb.nodes_per_elem, "nodes per element");
    assert_eq!(gr.conn.len(), gb.conn.len(), "geometry table length");

    // Per element and per slot (the layer-major table order every consumer
    // evaluates): the two parents must describe the same geometry.  A
    // scrambled read-back table (the pre-D295 defect) fails here even though
    // both meshes hold the same set of node values.
    let npe = gb.nodes_per_elem;
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize, 0usize);
    for e in 0..built.n_elems() as usize {
        for k in 0..npe {
            let xb = built.geom_coords_of(gb.conn[e * npe + k]);
            let xr = readback.geom_coords_of(gr.conn[e * npe + k]);
            for d in 0..3 {
                let diff = (xb[d] - xr[d]).abs();
                if diff > worst {
                    worst = diff;
                    worst_at = (e, k, d);
                }
            }
        }
    }
    eprintln!(
        "D314 read-back vs in-process parent geometry: max |Δ| = {worst:.3e} \
         (element {}, slot {}, component {})",
        worst_at.0, worst_at.1, worst_at.2
    );
    assert!(worst <= TOL, "geometry node mismatch {worst:e} > {TOL:e}");

    // Round-trip the read-back mesh and compare with the C++ `-rs 0` file: the
    // read-back parent must be as faithful as the in-process one.
    let want = parse_mesh(CPP_PARENT);
    let out_path = std::env::temp_dir().join("d314_toroid_wedge_parent_readback.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &readback, NodesSpace::Continuous)
        .expect("write read-back parent");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&out_path);
    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
    assert_eq!(got.fec, want.fec, "nodes FE collection");
    assert_eq!(got.nodes.len(), want.nodes.len(), "nodes payload length");
    for (e, (gtype, conn)) in want.elems.iter().enumerate() {
        assert_eq!(*gtype, got.elems[e].0, "element {e}: geometry type");
        assert_eq!(got.elems[e].1, *conn, "element {e}: connectivity");
    }
    for (f, (gtype, conn)) in want.bdr.iter().enumerate() {
        assert_eq!(*gtype, got.bdr[f].0, "boundary {f}: geometry type");
        assert_eq!(got.bdr[f].1, *conn, "boundary {f}: connectivity");
    }
    let mut max_dn = 0.0_f64;
    for (v, &w) in got.nodes.iter().zip(want.nodes.iter()) {
        max_dn = max_dn.max((v - w).abs());
    }
    assert!(max_dn <= TOL, "read-back parent nodes deviate by {max_dn:e} (tol {TOL:e})");
}
