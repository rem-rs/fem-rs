//! D173 stage-2 acceptance test: uniform refinement of a **curved** 2-D
//! triangle mesh must keep the high-order geometry — topology *and* MFEM's
//! dof numbering.
//!
//! Reference: serial MFEM 4.10 refining its own `data/square-disc-p2.mesh`
//! (154 curved `H1_2D_P2` triangles) once with `Mesh::UniformRefinement` and
//! saving the result, loaded **without** the constructor's `refine` flag:
//!
//! ```text
//! # tri_ref0.cpp: mfem::Mesh mesh(in, 0, 0); mesh.UniformRefinement(); mesh.Save(out);
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem410_ser tri_ref0.cpp -o tri_ref0 $HOME/mfem410_ser/libmfem.a
//! wsl ./tri_ref0 square-disc-p2.mesh square-disc-p2-r1.mesh
//! ```
//!
//! The C++ loader's `refine=1` flag marks (rotates) each triangle so its
//! longest edge is local edge 0 — fem-rs's 1:1 port is
//! `mark_tri_mesh_for_refinement`, but (new finding) that rotation currently
//! permutes only the connectivity, not a curved mesh's geometry-table slots,
//! so this test compares the unmarked path, which is what the refinement
//! itself must get right.
//!
//! What this pins (`amr::curved_tri`):
//!
//! 1. **Geometry values**: every child's `nodes` are the parent's order-`p`
//!    field evaluated on the child's reference triangle (MFEM's `tri_children`
//!    point matrices, including the rotated center child), and the fine
//!    vertices sit on the parent geometry (`UpdateNodes` →
//!    `SetVerticesFromNodes`).  The new-vertex *ids* already match MFEM's
//!    first-touch edge numbering, so (unlike the wedge path) no separate id
//!    gate is needed.
//! 2. **MFEM's `nodes` numbering**: `fem_io` has no 2-D triangle `nodes`
//!    writer yet ("only Hex8, Tet4, Prism6 and Quad4 are implemented" — a
//!    pre-existing gap, reported), so the test walks MFEM's H1_2D dof
//!    numbering itself (vertices | edge dofs per el_to_edge edge, oriented by
//!    ascending vertex id | per-element interiors) and compares every dof
//!    value against MFEM's refined file.

use std::collections::HashMap;

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};

const PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/square_disc_p2.mesh");
const CPP_REFINED: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/square_disc_p2_r1.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

/// One parsed `.mesh` file: everything the comparison looks at.
struct Parsed {
    /// `(geom type, connectivity)` per element, in file order.
    elems: Vec<(u32, Vec<u32>)>,
    /// `(geom type, connectivity)` per boundary element, in file order.
    bdr: Vec<(u32, Vec<u32>)>,
    /// Vertex rows (a curved mesh stores none).
    verts: Vec<f64>,
    /// `nodes` section values normalized to per-dof interleaving
    /// (`[x y]` per dof, whatever the file's Ordering was).
    nodes: Vec<f64>,
    /// The `nodes` section's FE collection name (`H1_2D_P2`), empty when absent.
    fec: String,
}

/// Minimal MFEM v1.0 mesh reader for the fixtures (2-D triangle meshes).
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
        elems.push((v[1], v[2..].to_vec()));
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
    let mut verts = Vec::with_capacity(2 * n);
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
        assert_eq!(v.len(), 2, "one 2-D vertex per row");
        verts.extend_from_slice(&v);
    }

    let mut nodes = Vec::new();
    let mut fec = String::new();
    let mut ordering = 1usize;
    for l in it.by_ref() {
        if l.trim() == "nodes" {
            loop {
                let h = it.next().expect("FE space header").trim().to_string();
                if let Some(name) = h.strip_prefix("FiniteElementCollection:") {
                    fec = name.trim().to_string();
                }
                if let Some(o) = h.strip_prefix("Ordering:") {
                    ordering = o.trim().parse().expect("ordering int");
                }
                if h.starts_with("Ordering:") {
                    break;
                }
            }
            let raw: Vec<f64> = it
                .flat_map(|l| {
                    l.split_whitespace()
                        .map(|t| t.parse::<f64>().expect("node value"))
                        .collect::<Vec<f64>>()
                })
                .collect();
            // Normalize Ordering: 0 (byNODES: all x, then all y) into
            // per-dof interleaving.
            if ordering == 0 {
                let ndof = raw.len() / 2;
                for d in 0..ndof {
                    nodes.push(raw[d]);
                    nodes.push(raw[ndof + d]);
                }
            } else {
                nodes = raw;
            }
            break;
        }
    }
    Parsed { elems, bdr, verts, nodes, fec }
}

/// MFEM's H1_2D global dof numbering for a Tri3 mesh: `n_verts` vertices,
/// then `(p-1)` dofs per edge (edges first-touch in element × local-edge
/// order, each edge oriented from its lower-id vertex), then
/// `(p+1)(p+2)/2 - 3 - 3(p-1)` interior dofs per element in element order.
///
/// Returns `geo dof id -> MFEM global dof id` (the geometry table's own ids
/// beyond the vertices are private to this mapping).
fn mfem_global_map(mesh: &Mesh<2>, p: usize, geo_conn: &[u32], n_geo_dofs: usize) -> (Vec<usize>, usize) {
    const EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    let dpe = (p + 1) * (p + 2) / 2;
    let ne = p - 1;
    let n_int = dpe - 3 - 3 * ne;

    let n_verts = mesh.n_nodes() as usize;
    let mut edge_ids: HashMap<(u32, u32), usize> = HashMap::new();
    // geo dof id → global id; vertices map to themselves.
    let mut map = vec![usize::MAX; n_geo_dofs];
    for v in 0..n_verts {
        map[v] = v;
    }

    for e in 0..mesh.n_elems() as u32 {
        let ns = mesh.elem_nodes(e);
        let o = e as usize * dpe;
        // Edges, first-touch in local-edge order.
        for (li, &(a, b)) in EDGES.iter().enumerate() {
            let key = (ns[a].min(ns[b]), ns[a].max(ns[b]));
            let is_new = !edge_ids.contains_key(&key);
            let entry = edge_ids.len();
            if is_new {
                edge_ids.insert(key, entry);
                // New edge: record its (p-1) dof mappings from this element's
                // slots (the first touch owns the geometry dof).
                for m in 1..p {
                    let slot = 3 + li * ne + (m - 1);
                    let geo_id = geo_conn[o + slot] as usize;
                    // Global position within the edge (0-based): MFEM orients
                    // the edge from its lower-id vertex; the local position
                    // `m` (1..p-1) counts from `EDGES[li].0`, so from the
                    // other end it is `p - m`.
                    let pos = if ns[a] <= ns[b] { m - 1 } else { p - m - 1 };
                    let g = n_verts + entry * ne + pos;
                    if geo_id >= n_verts {
                        map[geo_id] = g;
                    }
                }
            }
        }
        // Interior dofs, per element in order.
        for k in 0..n_int {
            let slot = 3 + 3 * ne + k;
            let geo_id = geo_conn[o + slot] as usize;
            if geo_id >= n_verts {
                map[geo_id] = n_verts + edge_ids.len() * ne + e as usize * n_int + k;
            }
        }
    }
    (map, n_verts + edge_ids.len() * ne + mesh.n_elems() as usize * n_int)
}

#[test]
fn d173_refined_curved_tri_matches_mfem_file() {
    // Load without marking (the C++ probe loads with refine=0), so the
    // geometry-table slots and the connectivity stay consistent.
    let f = read_mfem_file(PARENT).expect("read parent fixture");
    let parent: Mesh<2> = f.mesh2d.expect("2D parent");
    assert_eq!(parent.n_elems(), 154);
    assert!(parent.geometry.is_some(), "parent must carry curved geometry");
    assert_eq!(parent.geom_order(), 2);

    let fine = refine_uniform(&parent);
    assert_eq!(fine.n_elems(), 616, "154 tris × 4 children");
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 2, "geom_order must survive uniform refinement");
    assert_eq!(g.nodes_per_elem, 6, "(p+1)(p+2)/2 dofs per order-2 triangle");

    // Whole-file comparison against MFEM's own refined output: connectivity
    // sections directly, `nodes` through MFEM's H1_2D dof numbering.
    let want = parse_mesh(CPP_REFINED);

    assert_eq!(fine.n_elems() as usize, want.elems.len(), "element count");
    assert_eq!(fine.n_faces() as usize, want.bdr.len(), "boundary element count");

    for (e, (gtype, conn)) in want.elems.iter().enumerate() {
        assert_eq!(*gtype, 2, "element {e}: geometry type");
        let ns: Vec<u32> = fine.elem_nodes(e as u32).to_vec();
        assert_eq!(ns, *conn, "element {e}: connectivity disagrees");
    }
    for (f, (gtype, conn)) in want.bdr.iter().enumerate() {
        assert_eq!(*gtype, 1, "boundary {f}: geometry type");
        let fs = {
            let o = f * 2;
            vec![fine.face_conn[o], fine.face_conn[o + 1]]
        };
        assert_eq!(fs, *conn, "boundary {f}: connectivity disagrees");
    }

    // `nodes` comparison: build the geo-dof → global-dof map and compare.
    let (map, n_global) = mfem_global_map(&fine, 2, &g.conn, g.n_nodes);
    assert_eq!(n_global, want.nodes.len() / 2, "global dof count");
    let mut unfilled = 0usize;
    let mut max_dn = 0.0_f64;
    for (id, gm) in map.iter().enumerate() {
        if *gm == usize::MAX {
            unfilled += 1;
            continue;
        }
        for c in 0..2 {
            let got = g.coords[id * 2 + c];
            let w = want.nodes[gm * 2 + c];
            max_dn = max_dn.max((got - w).abs());
            assert!(
                (got - w).abs() <= TOL,
                "global dof {gm} (geo {id}, comp {c}): got {got} want {w} (|Δ| {:.3e})",
                (got - w).abs()
            );
        }
    }
    assert_eq!(unfilled, 0, "every geometry dof must claim an MFEM global dof");

    eprintln!(
        "square-disc-p2 rs1: {} elements, {} boundary faces, {} nodes dofs — \
         connectivity byte-matched, nodes max |Δ| {max_dn:e} (tol {TOL:e})",
        fine.n_elems(),
        fine.n_faces(),
        n_global
    );
}
