//! D180 acceptance test: uniform refinement of a **straight-sided** wedge
//! (Prism6) mesh must be MFEM's topology, vertex numbering and child order —
//! the historical fem-rs straight path (tri-face centers + body centers,
//! first-touch vertex ids, corner-corner-corner-center child order) is gone.
//!
//! Reference: serial MFEM 4.10 `miniapps/meshing/toroid.cpp -e 0 -o 1`
//! (generator defaults `-nphi 8`), refined once with `-rs 1`.  The fixtures
//! are the C++ miniapp's own outputs, copied verbatim (8-digit print
//! precision):
//!
//! ```text
//! wsl cd $HOME/work/d279 && g++ -std=c++17 -O2 -I$HOME/mfem410_ser toroid.cpp \
//!     -o toroid_cpp $HOME/mfem410_ser/miniapps/common/libmfem-common.a \
//!     $HOME/mfem410_ser/libmfem.a
//! wsl cd $HOME/work/d279 && ./toroid_cpp -e 0 -rs 0 -o 1 -no-vis  # parent
//! wsl cd $HOME/work/d279 && ./toroid_cpp -e 0 -rs 1 -o 1 -no-vis  # refined
//! ```
//!
//! The parent is built in-process exactly like the `mesh_toroid` miniapp
//! builds it (`Transform` → stitch, no `SetCurvature` at `-o 1`).
//!
//! What this pins (MFEM `UniformRefinement3D_base`, `case Element::WEDGE`, on
//! a mesh without a `nodes` table):
//!
//! 1. **Vertex set** (`MfemPrismRefineIds`): the wedge split creates *no*
//!    triangular face centers and *no* body centers — the pre-D180 straight
//!    path allocated 16 unused vertices (112 written where MFEM writes 96).
//! 2. **Vertex numbering**: `[coarse | oedge + E | oface + f2qf(F)]` — the
//!    historical first-touch straight numbering put quad-face centers and
//!    midpoints in different slots, so even the shared `elements` ids
//!    disagreed (element 0 slot 4: fem-rs 35 vs MFEM 72).
//! 3. **Child order**: MFEM emits the 8 children as corner 0, center,
//!    corner 1, corner 2 per layer (center children cyclically rotated) —
//!    not the historical corner 0, corner 1, corner 2, center order.

use std::collections::HashMap;
use std::f64::consts::PI;

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::{refine_uniform_3d, Mesh};

const CPP_REFINED: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_wedge_o1_r1.mesh");
const CPP_PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/toroid_wedge_o1.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

// ─── The `mesh_toroid` miniapp's construction (toroid.rs, `-e 0 -o 1`) ──────

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
/// `set_face_tables`).
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

/// `Mesh::Transform(trans)` on the straight mesh: every vertex moves through
/// the torus map.
fn apply_transform(mesh: &mut Mesh<3>, f: &impl Fn(&[f64]) -> [f64; 3]) {
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

/// The `mesh_toroid -e 0 -o 1` parent: the straight-sided torus of 8 wedges —
/// transform, then the end stitch; no `SetCurvature` at order 1.
fn build_straight_parent() -> Mesh<3> {
    let nphi = 8usize;
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
    /// Vertex coordinates from the `vertices` section (a straight mesh's only
    /// geometry payload).
    verts: Vec<[f64; 3]>,
}

/// Minimal MFEM v1.0 mesh reader for the fixtures (pure-wedge straight meshes
/// without a `nodes` section).
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
    // The line after the count is the space dimension (`3`), then n rows of
    // `x y z`.
    let _space_dim: usize = it.next().expect("space dim").trim().parse().expect("int");
    let mut verts = Vec::with_capacity(n);
    while verts.len() < n {
        let l = it.next().expect("vertex row");
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        let c: Vec<f64> = t.split_whitespace().map(|v| v.parse().expect("f64")).collect();
        assert_eq!(c.len(), 3, "vertex row {t:?}");
        verts.push([c[0], c[1], c[2]]);
    }
    Parsed { elems, bdr, verts }
}

#[test]
fn d180_refined_straight_toroid_wedge_matches_mfem_file() {
    let parent = build_straight_parent();
    assert_eq!(parent.n_elems(), 8);
    assert!(parent.geometry.is_none(), "order-1 parent has no curved geometry");
    assert_eq!(parent.n_nodes(), 24);

    // The in-process construction must reproduce the C++ `-rs 0` file (the
    // same bytes the `mesh_toroid` miniapp writes) — this is what makes the
    // refined comparison below a *miniapp*-level check.
    let want_parent = parse_mesh(CPP_PARENT);
    assert_eq!(want_parent.elems.len(), 8, "fixture: 8 elements");
    let parent_path = std::env::temp_dir().join("d180_toroid_wedge_straight_parent.mesh");
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
    let mut max_dp = 0.0_f64;
    for (g, w) in got_parent.verts.iter().zip(want_parent.verts.iter()) {
        for d in 0..3 {
            max_dp = max_dp.max((g[d] - w[d]).abs());
        }
    }
    assert!(max_dp <= TOL, "parent vertices deviate by {max_dp:e} (tol {TOL:e})");

    // Mesh-level invariants of the refinement itself.  MFEM's wedge split
    // gains 9 edge midpoints + 3 quad face centers per prism (no triangular
    // face centers, no body centers); the toroid ring shares edges and quad
    // faces between neighbouring prisms: 24 + 48 + 24 = 96 fine vertices.
    // The pre-D180 straight path wrote 112 (16 unused tri-face/body centers).
    let fine = refine_uniform_3d(&parent);
    assert_eq!(fine.n_elems(), 64);
    assert_eq!(fine.n_nodes(), 96, "no tri-face centers, no body centers");
    assert_eq!(fine.n_faces(), 96, "every coarse boundary face splits into 4");
    fine.check().expect("refined mesh must stay valid");

    // Whole-file comparison: write the refined mesh exactly like the miniapp
    // does and diff every section against MFEM's own output.  The `elements`
    // and `boundary` sections must match BYTE FOR BYTE (same ids, same child
    // order); only the `vertices` section carries print-precision noise.
    let want = parse_mesh(CPP_REFINED);
    let out_path = std::env::temp_dir().join("d180_toroid_wedge_straight_refined.mesh");
    write_mfem_file_3d_nodes(out_path.to_str().expect("temp path"), &fine, NodesSpace::Continuous)
        .expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&out_path);

    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary element count");
    assert_eq!(got.verts.len(), want.verts.len(), "vertex count (96 vs 112 pre-D180)");

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

    let mut max_dv = 0.0_f64;
    for (g, w) in got.verts.iter().zip(want.verts.iter()) {
        for d in 0..3 {
            max_dv = max_dv.max((g[d] - w[d]).abs());
        }
    }
    assert!(max_dv <= TOL, "vertices deviate by {max_dv:e} (tol {TOL:e})");

    eprintln!(
        "toroid wedge o1 rs1: 64/64 elements, 96/96 boundary faces, 96/96 vertices; \
         parent max |Δ| {max_dp:e}, refined max |Δ| {max_dv:e}"
    );
}
