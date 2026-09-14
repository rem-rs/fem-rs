//! D166 stage-2 acceptance test: uniform refinement of a **curved 2-D quad**
//! mesh must keep its order-`p` geometry — topology *and* MFEM's numbering.
//!
//! Reference: serial MFEM 4.10 on `data/star-q2.mesh` (20 quadratic quads, the
//! classic curved star), refined once with `Mesh::UniformRefinement`.  The
//! parent fixture `star_q2.mesh` is MFEM's own file and `star_q2_r1.mesh` the
//! probe's output, both copied verbatim (8-digit print precision).  Probe
//! (kept in `tmp/d166/quad_ref.cpp`):
//!
//! ```text
//! wsl g++ -std=c++17 -O2 -I$HOME/mfem410_ser tmp/d166/quad_ref.cpp \
//!     $HOME/mfem410_ser/libmfem.a -o $HOME/work/d34/quad_ref
//! wsl $HOME/work/d34/quad_ref $HOME/mfem410_ser/data/star-q2.mesh \
//!     $HOME/work/d34/star-q2-r1.mesh 1
//! ```
//!
//! What this pins for the 2-D path (`refine_uniform` → `refine_uniform_quad4`
//! + `amr::curved_quad`):
//!
//! 1. the refined mesh keeps `geom_order == 2` with `(p+1)²` geometry dofs per
//!    element, shared across elements meeting on a fine edge (MFEM's `nodes`
//!    update), instead of collapsing to per-child independent bilinear quads;
//! 2. the fine vertices sit on the parent geometry (`SetVerticesFromNodes`);
//! 3. the written file — elements, boundary, `nodes` values in MFEM's
//!    `H1_2D_P2` numbering — matches MFEM's output section by section.
//!
//! Tolerance 1e-7: both sides print 8 significant digits, so the ~1e-16
//! arithmetic noise of MFEM's refinement operator is invisible except through
//! re-rounding.

use fem_io::mfem::{read_mfem_file, write_mfem_file};
use fem_mesh::{refine_uniform, Mesh};

const PARENT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/star_q2.mesh");
const CPP_REFINED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/star_q2_r1.mesh");
/// 8-significant-digit print quantum of coordinates of size ~1.
const TOL: f64 = 1e-7;

/// One parsed 2-D `.mesh` file: everything the comparison looks at.
struct Parsed {
    /// `(geom type, connectivity)` per element, in file order.
    elems: Vec<(u32, Vec<u32>)>,
    /// `(geom type, connectivity)` per boundary segment, in file order.
    bdr: Vec<(u32, Vec<u32>)>,
    /// Vertex count (the rows are omitted for curved meshes).
    n_verts: usize,
    /// `nodes` section values exactly as stored in the file (see `ordering`).
    nodes: Vec<f64>,
    /// The `nodes` section's Ordering header (0 = byNODES, 1 = byVDIM).
    ordering: u8,
    /// The `nodes` section's FE collection name (`H1_2D_P2`).
    fec: String,
}

/// Minimal MFEM v1.0 mesh reader for 2-D fixtures.
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
    // MFEM writes the vertex *rows* only for straight-sided meshes; a curved
    // mesh stores its geometry in `nodes` and the section is just the count
    // followed by a blank line (`Mesh::Print`).
    let mut rows = 0usize;
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
        rows += 1;
    }
    // A curved file has `rows == 0` regardless of `n`.
    assert!(rows == 0 || rows == n, "vertex rows must match the count when present");

    // Optional `nodes` section.
    let mut nodes = Vec::new();
    let mut ordering = 1u8;
    let mut fec = String::new();
    for l in it.by_ref() {
        if l.trim() == "nodes" {
            // Header lines: known keys, terminated by a blank line (MFEM's
            // GridFunction::Save) or by the first payload line (fem-rs's
            // writer emits the values right after `Ordering:`).
            loop {
                let h = match it.peek() {
                    Some(l) => l.trim().to_string(),
                    None => break,
                };
                let is_header = h.starts_with("FiniteElement")
                    || h.starts_with("VDim:")
                    || h.starts_with("Ordering:");
                if !h.is_empty() && is_header {
                    it.next();
                    if let Some(name) = h.strip_prefix("FiniteElementCollection:") {
                        fec = name.trim().to_string();
                    }
                    if let Some(o) = h.strip_prefix("Ordering:") {
                        ordering = o.trim().parse().expect("ordering int");
                    }
                } else {
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
    Parsed { elems, bdr, n_verts: n, nodes, ordering, fec }
}

#[test]
fn d166_refined_curved_quad_star_matches_mfem_file() {
    let f = read_mfem_file(PARENT).expect("read parent fixture");
    let parent: Mesh<2> = f.mesh2d.expect("2D parent");
    assert_eq!(parent.n_elems(), 20);
    assert!(parent.geometry.is_some(), "parent must carry curved geometry");
    assert_eq!(parent.geom_order(), 2);

    // Mesh-level invariants of the refinement itself.
    let fine = refine_uniform(&parent);
    let g = fine.geometry.as_ref().expect("refined mesh must keep its geometry");
    assert_eq!(g.order, 2, "geom_order must survive uniform refinement");
    assert_eq!(g.nodes_per_elem, 9, "(p+1)^2 geometry dofs per element");
    assert_eq!(g.conn.len(), fine.n_elems() as usize * 9);
    assert_eq!(fine.n_elems(), 80);
    // MFEM vertex layout: [coarse vertices | edge midpoints | quad centers]
    // (exact counts pinned by the file comparison below).

    // Whole-file comparison: write the refined mesh and diff every section
    // against MFEM's own output.
    let want = parse_mesh(CPP_REFINED);
    let out_path = std::env::temp_dir().join("d166_star_refined.mesh");
    write_mfem_file(&out_path, &fine).expect("write refined mesh");
    let got = parse_mesh(out_path.to_str().expect("temp path"));
    let _ = std::fs::remove_file(&out_path);

    assert_eq!(got.elems.len(), want.elems.len(), "element count");
    assert_eq!(got.bdr.len(), want.bdr.len(), "boundary segment count");
    assert_eq!(got.n_verts, want.n_verts, "vertex count");
    assert_eq!(got.fec, want.fec, "nodes FE collection");
    assert_eq!(got.fec, "H1_2D_P2");
    assert_eq!(got.nodes.len(), want.nodes.len(), "nodes payload length");

    // `star-q2.mesh` carries its `nodes` with `Ordering: 0` (byNODES: all x
    // components, then all y components, one value per line) and MFEM's
    // refinement keeps that layout; fem-rs's writer emits byVDIM.  Transpose
    // the reference to byVDIM before the value comparison.
    let vdim = 2usize;
    let want_byvdim: Vec<f64> = if want.ordering == 0 {
        let ndofs = want.nodes.len() / vdim;
        (0..want.nodes.len())
            .map(|i| {
                let (d, c) = (i / vdim, i % vdim);
                want.nodes[c * ndofs + d]
            })
            .collect()
    } else {
        want.nodes.clone()
    };

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

    let mut max_dn = 0.0_f64;
    for (v, &w) in got.nodes.iter().zip(want_byvdim.iter()) {
        max_dn = max_dn.max((v - w).abs());
    }
    assert!(max_dn <= TOL, "nodes values deviate by {max_dn:e} (tol {TOL:e})");

    eprintln!(
        "star-q2 rs1: 80/80 elements, {} boundary segments matched; max |Δ| nodes {max_dn:e}",
        got.bdr.len()
    );
}
