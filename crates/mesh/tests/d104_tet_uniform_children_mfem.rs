//! D943 — tet uniform refinement must reproduce MFEM 4.10 exactly: per-parent
//! refinement type, child element tables (ids, in emission order), and the
//! refined vertex table (ids + coordinates).
//!
//! Oracle: MFEM 4.10 serial (`$HOME/mfem410_ser`), probe
//! `tmp/d104tet/probe_d104.cpp` — read the mesh, call `UniformRefinement()`
//! twice, and after each step dump the vertex table (ids + coordinates,
//! precision 17) and the element connectivity; *before* each refinement step
//! dump the per-parent refinement type chosen by `UniformRefinement3D_base`'s
//! own `rt_algo = 1` branch (copied verbatim into the probe).  Fixtures
//! (`crates/mesh/tests/data/`, `.mesh.txt` because `*.mesh` is git-ignored):
//!
//! * `d104_beam_tet.mesh.txt` — byte copy of MFEM's `data/beam-tet.mesh`
//!   (48 tets: an 8×1×1 beam, mixed tet shapes); `d104_cart1_tet.mesh.txt` —
//!   MFEM's own `Mesh::Print` of `MakeCartesian3D(1,1,1,TETRAHEDRON)` (6
//!   axis tets), so both sides refine the byte-identical coarse mesh;
//! * `d104_mfem_{beam_tet,cart1_tet}_levels.txt` — the probe dumps
//!   (`LEVEL r` + `V`/`E` rows, `RT` rows attached to the level they
//!   refine).
//!
//! The historical defect (D943, evidence `tmp/d103prol/`): "of beam-tet's 384
//! children only 96 match MFEM".  This pin asserts the *exact* child table of
//! every parent at both refinement levels on both meshes.

use fem_io::mfem::read_mfem;
use fem_mesh::{refine_uniform_3d, tet_select_rt_debug, Mesh};
use std::io::Cursor;

const BEAM_MESH: &str = include_str!("data/d104_beam_tet.mesh.txt");
const CART1_MESH: &str = include_str!("data/d104_cart1_tet.mesh.txt");
const BEAM_ORACLE: &str = include_str!("data/d104_mfem_beam_tet_levels.txt");
const CART1_ORACLE: &str = include_str!("data/d104_mfem_cart1_tet_levels.txt");

/// One probe level: vertex table, element table, and the per-element
/// refinement type the probe computed for this level's elements (empty after
/// the last level — nothing refines further).
#[derive(Debug, Default)]
struct Level {
    verts: Vec<(u32, [f64; 3])>,
    elems: Vec<[u32; 4]>,
    rts: Vec<usize>,
}

/// Parse the probe output: `LEVEL r` blocks of `NV`/`V`/`NE`/`E` rows, each
/// possibly followed by `RT <elem> <rt>` rows (rt of *this* level's elements).
fn parse_levels(text: &str) -> Vec<Level> {
    let mut levels: Vec<Level> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("LEVEL ") {
            assert_eq!(rest.parse::<u32>().unwrap(), levels.len() as u32, "level order");
            levels.push(Level::default());
        } else if let Some(rest) = line.strip_prefix("V ") {
            let mut it = rest.split_whitespace();
            let id = it.next().unwrap().parse::<u32>().unwrap();
            let x = it.next().unwrap().parse::<f64>().unwrap();
            let y = it.next().unwrap().parse::<f64>().unwrap();
            let z = it.next().unwrap().parse::<f64>().unwrap();
            levels.last_mut().unwrap().verts.push((id, [x, y, z]));
        } else if let Some(rest) = line.strip_prefix("E ") {
            let mut it = rest.split_whitespace();
            let id = it.next().unwrap().parse::<u32>().unwrap();
            let v: Vec<u32> = it.map(|s| s.parse::<u32>().unwrap()).collect();
            assert_eq!(v.len(), 4);
            let lvl = levels.last_mut().unwrap();
            assert_eq!(id, lvl.elems.len() as u32, "element emission order");
            lvl.elems.push([v[0], v[1], v[2], v[3]]);
        } else if let Some(rest) = line.strip_prefix("RT ") {
            let mut it = rest.split_whitespace();
            let id = it.next().unwrap().parse::<u32>().unwrap();
            let rt = it.next().unwrap().parse::<usize>().unwrap();
            let lvl = levels.last_mut().unwrap();
            assert_eq!(id as usize, lvl.rts.len(), "rt row order");
            lvl.rts.push(rt);
        }
    }
    levels
}

/// fem-rs side of one probe level: ids + coordinates of the vertex table.
fn rs_verts(mesh: &Mesh<3>) -> Vec<(u32, [f64; 3])> {
    (0..mesh.n_nodes() as u32)
        .map(|id| (id, mesh.coords_of(id)))
        .collect()
}

/// fem-rs side of one probe level: element tables in emission order.
fn rs_elems(mesh: &Mesh<3>) -> Vec<[u32; 4]> {
    (0..mesh.n_elems() as u32)
        .map(|e| {
            let ns = mesh.elem_nodes(e);
            [ns[0], ns[1], ns[2], ns[3]]
        })
        .collect()
}

/// fem-rs side of the probe's `RT` block: the `rt_algo = 1` selection over
/// each element's vertex-difference Jacobian (straight-sided meshes).
fn rs_rts(mesh: &Mesh<3>) -> Vec<usize> {
    (0..mesh.n_elems() as u32)
        .map(|e| tet_select_rt_debug(mesh, mesh.elem_nodes(e)))
        .collect()
}

/// First mismatching element (index, fem-rs table, oracle table), if any.
fn first_elem_diff(rs: &[[u32; 4]], mfem: &[[u32; 4]]) -> Option<(usize, [u32; 4], [u32; 4])> {
    assert_eq!(rs.len(), mfem.len(), "child count");
    for (i, (r, m)) in rs.iter().zip(mfem.iter()).enumerate() {
        if r != m {
            return Some((i, *r, *m));
        }
    }
    None
}

/// First coordinate mismatch (index, fem-rs value, oracle value) at the
/// *bitwise* level (straight-sided refinement midpoints are dyadic — the
/// averages are exact in binary floating point on both sides).
fn first_coord_diff(
    rs: &[(u32, [f64; 3])],
    mfem: &[(u32, [f64; 3])],
) -> Option<(u32, f64, f64)> {
    assert_eq!(rs.len(), mfem.len(), "vertex count");
    for (r, m) in rs.iter().zip(mfem.iter()) {
        assert_eq!(r.0, m.0, "vertex ids");
        for d in 0..3 {
            if r.1[d].to_bits() != m.1[d].to_bits() {
                return Some((r.0, r.1[d], m.1[d]));
            }
        }
    }
    None
}

fn check(name: &str, coarse_text: &str, oracle_text: &str) {
    let levels = parse_levels(oracle_text);
    assert!(levels.len() >= 3, "need levels 0,1,2");
    assert!(!levels[0].rts.is_empty() && !levels[1].rts.is_empty());

    let l0 = read_mfem(Cursor::new(coarse_text)).unwrap().mesh3d.expect("3-D mesh");

    // Coarse read-back: byte-identical input ⇒ identical tables.
    assert_eq!(l0.n_elems(), levels[0].elems.len(), "{name}: coarse NE");
    assert_eq!(l0.n_nodes(), levels[0].verts.len(), "{name}: coarse NV");
    assert!(
        first_elem_diff(&rs_elems(&l0), &levels[0].elems).is_none(),
        "{name}: coarse element table differs from MFEM's LEVEL 0"
    );
    assert!(
        first_coord_diff(&rs_verts(&l0), &levels[0].verts).is_none(),
        "{name}: coarse vertices differ from MFEM's LEVEL 0"
    );

    // Per-parent refinement type, level 0 (drives refinement 0→1).
    assert_eq!(rs_rts(&l0), levels[0].rts, "{name}: rt selection at level 0");

    let l1 = refine_uniform_3d(&l0);
    assert_eq!(l1.n_elems(), levels[1].elems.len(), "{name}: r1 child count");
    assert_eq!(l1.n_nodes(), levels[1].verts.len(), "{name}: r1 NV");
    match first_elem_diff(&rs_elems(&l1), &levels[1].elems) {
        None => {}
        Some((i, r, m)) => panic!(
            "{name}: r1 child {i} (parent {}) table {r:?} != MFEM {m:?}",
            i / 8
        ),
    }
    assert!(
        first_coord_diff(&rs_verts(&l1), &levels[1].verts).is_none(),
        "{name}: r1 vertices differ from MFEM (ids or bitwise coordinates)"
    );

    // Per-parent refinement type, level 1 (drives refinement 1→2) — this is
    // where a wrong level-1 *child order* would show up as a wrong rt: the
    // selection consumes the level-1 vertex order as stored.
    assert_eq!(rs_rts(&l1), levels[1].rts, "{name}: rt selection at level 1");

    let l2 = refine_uniform_3d(&l1);
    assert_eq!(l2.n_elems(), levels[2].elems.len(), "{name}: r2 child count");
    assert_eq!(l2.n_nodes(), levels[2].verts.len(), "{name}: r2 NV");
    match first_elem_diff(&rs_elems(&l2), &levels[2].elems) {
        None => {}
        Some((i, r, m)) => panic!(
            "{name}: r2 child {i} (parent {}) table {r:?} != MFEM {m:?}",
            i / 8
        ),
    }
    assert!(
        first_coord_diff(&rs_verts(&l2), &levels[2].verts).is_none(),
        "{name}: r2 vertices differ from MFEM (ids or bitwise coordinates)"
    );
}

#[test]
fn d104_beam_tet_two_uniform_refinements_match_mfem() {
    check("beam-tet", BEAM_MESH, BEAM_ORACLE);
}

#[test]
fn d104_cart1_tet_two_uniform_refinements_match_mfem() {
    check("cart1-tet", CART1_MESH, CART1_ORACLE);
}
