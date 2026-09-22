//! D526 — `HDivSpace::dof_nodal_coords`: MFEM-nodal coordinates for every
//! H(div) DOF.
//!
//! The D414 `dof_coords` anchors are deliberately *geometric* (eispaced
//! permutation keys, "not MFEM interpolation nodes"); on the 2x2x2 hex cube
//! the k=1 quad-face lattice collapses onto the cube's 26 boundary vertices,
//! aliasing the 96 essential boundary dofs (the D506 lesion).  The accessor
//! tested here reports each DOF at its **MFEM nodal point** instead.
//!
//! MFEM-side reading convention (same as the D505/D506 oracle): the nodal
//! point of a DOF is the slot of the element's `VectorFiniteElement::Nodes`
//! integration rule (`fe->GetNodes()` — H(div) elements' nodal points are
//! *not* `Geometry` nodes), transformed by `ElementTransformation`.
//!
//! Oracles (MFEM 4.10, `$HOME/mfem410_ser`; probe sources + dumps archived
//! under `tmp/d526/`):
//! - `d526_probe.cpp` — dumps `fe->GetNodes()` reference tables and the
//!   physical point of every global dof (`NODE e dof x y z`), plus the
//!   boundary-essential point set (`KEY`, quantised 1e-6 like `key3`).
//!   The pyramid mesh is built in code (`AddPyramid`) with the exact vertex
//!   table of `tmp/d392/one_pyramid.mesh` (the 4.10 text reader segfaults on
//!   that file here).
//! - `wedge_ref.cpp` / `wedge_ref.txt` — `RT_WedgeElement(p).GetNodes()`
//!   for p = 0..=3; the generator behind `wedge_nodal_points` matches every
//!   slot to 1e-14 (`check_wedge.py`).
//! - `data/d526_nodal_keys_mfem.txt` — quantised goldens extracted from the
//!   dumps (sections `#RT0HEXBDR`, `#RT2HEXBDR`, `#PRISM1E0`, `#PRISM1E1`,
//!   `#PYR1E0`).

use std::collections::HashSet;

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::{FESpace, HDivSpace};

/// `data/d526_nodal_keys_mfem.txt` — the probe's quantised point tables.
const GOLDEN: &str = include_str!("data/d526_nodal_keys_mfem.txt");

fn key3(c: &[f64; 3]) -> [i64; 3] {
    [
        (c[0] * 1e6).round() as i64,
        (c[1] * 1e6).round() as i64,
        (c[2] * 1e6).round() as i64,
    ]
}

/// One `#<section>` table of the golden file, sorted (the section header
/// line is `#NAME  description...`).
fn golden(section: &str) -> Vec<[i64; 3]> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in GOLDEN.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('#') {
            inside = rest
                .split_whitespace()
                .next()
                .map(|name| name == section)
                .unwrap_or(false);
            continue;
        }
        if line.is_empty() || !inside {
            continue;
        }
        let t: Vec<i64> = line
            .split_whitespace()
            .map(|tok| tok.parse().expect("golden token"))
            .collect();
        assert_eq!(t.len(), 3, "golden line {line:?}");
        out.push([t[0], t[1], t[2]]);
    }
    assert!(!out.is_empty(), "golden section {section} missing");
    out.sort_unstable();
    out
}

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// The D506 lesion, preserved as the red-proof: the *anchor* key function
/// (`dof_coords`, D414) aliases the 96 essential boundary dofs onto the 26
/// boundary vertices, while `dof_nodal_coords` resolves them to the 96
/// MFEM nodal points.  Before D526 the accessor did not exist and every
/// `dof_coords`-keyed consumer measured 26.
#[test]
fn d526_nodal_keys_resolve_the_26_anchor_aliasing() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 1);
    let all_tags = mesh.unique_boundary_tags();
    let dofs = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
    assert_eq!(dofs.len(), 96, "MFEM ness (RT1 hex cube)");

    let anchors = space.dof_coords();
    let anchor_keys: HashSet<_> = dofs.iter().map(|&d| key3(&anchors[d as usize])).collect();
    assert_eq!(
        anchor_keys.len(),
        26,
        "the D414 anchors alias 96 dofs onto the 26 cube vertices (the lesion)"
    );

    let nodal = space.dof_nodal_coords();
    let nodal_keys: HashSet<_> = dofs.iter().map(|&d| key3(&nodal[d as usize])).collect();
    assert_eq!(nodal_keys.len(), 96, "MFEM nodal points: 96 distinct keys");
}

/// Hex RT2: the 216 boundary dofs (24 x (k+1)^2) hit the probe's 216-key
/// boundary point table exactly.
#[test]
fn d526_hex_rt2_boundary_nodal_keys_match_mfem() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 2);
    let all_tags = mesh.unique_boundary_tags();
    let dofs = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
    assert_eq!(dofs.len(), 216, "MFEM RT2 ness = 216");

    let nodal = space.dof_nodal_coords();
    let mut got: Vec<_> = dofs.iter().map(|&d| key3(&nodal[d as usize])).collect();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got.len(), 216, "216 pairwise-distinct nodal points");
    assert_eq!(got, golden("RT2HEXBDR"), "RT2 boundary keys vs MFEM probe");
}

/// Hex RT0: 24 boundary dofs at the 24 boundary-face centres.
#[test]
fn d526_hex_rt0_boundary_nodal_keys_match_mfem() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 0);
    let all_tags = mesh.unique_boundary_tags();
    let dofs = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
    assert_eq!(dofs.len(), 24, "MFEM RT0 ness = one per boundary face");

    let nodal = space.dof_nodal_coords();
    let mut got: Vec<_> = dofs.iter().map(|&d| key3(&nodal[d as usize])).collect();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got, golden("RT0HEXBDR"), "RT0 face-centre keys vs MFEM probe");
}

/// Hex RT1 interior dofs: the 12 non-face slots of each element sit at the
/// closed x GL(gauss_lobatto) x open x open interior grid — strictly inside
/// the element, off the faces.  Also: every dof of the whole space gets a
/// point distinct from its neighbours where the reference layout demands it.
#[test]
fn d526_hex_rt1_interior_dofs_at_closed_open_grid() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();

    // Collect the per-element interior slots: element slots 24..36 (after
    // the 6 face blocks of 4).  The accessor must place them strictly inside
    // the element box.
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &v in verts {
            let c = mesh.node_coords(v);
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        let dofs = space.element_dofs(e);
        assert_eq!(dofs.len(), 36, "hex RT1: 6x4 faces + 12 interiors");
        for &slot_dof in &dofs[24..36] {
            let p = nodal[slot_dof as usize];
            for d in 0..3 {
                assert!(
                    p[d] > lo[d] + 1e-12 && p[d] < hi[d] - 1e-12,
                    "elem {e} interior dof {slot_dof} at {p:?} outside ({lo:?},{hi:?})"
                );
            }
        }
    }
    // The 12 interior points of the reference layout are the closed {1/2} x
    // open {a, b} x open {a, b} grid = 4 points per axis, 12 with component
    // orientation: per element the 12 interior slots map onto exactly 12
    // distinct dofs with 12 points (4+4+4 grid points, all distinct).
    let mut pts: Vec<_> = {
        let dofs = space.element_dofs(0);
        dofs[24..36].iter().map(|&d| key3(&nodal[d as usize])).collect()
    };
    pts.sort_unstable();
    pts.dedup();
    assert_eq!(pts.len(), 12, "hex RT1 interior nodal grid: 12 distinct points");
}

/// Prism stack (`data/d394_prism_stack.mesh`, 2 wedges) RT1 — the non-hex
/// geometry gate: per-element point sets match the MFEM `NODE` dumps
/// element-for-element (the wedge's nodal table is the one *measured*
/// against 4.10 for p = 0..=3), and the shared tri face {3,4,5} agrees.
#[test]
fn d526_prism_rt1_per_element_points_match_mfem() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();
    assert_eq!(space.n_dofs(), 47, "prism stack RT1 vsize (MFEM)");

    for (e, section) in [(0u32, "PRISM1E0"), (1u32, "PRISM1E1")] {
        let mut got: Vec<_> = space
            .element_dofs(e)
            .iter()
            .map(|&d| key3(&nodal[d as usize]))
            .collect();
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, golden(section), "prism element {e} point set vs MFEM");
    }
}

/// The shared tri face {3,4,5} of the prism stack: both elements map its 3
/// dofs to points strictly inside the shared face — the accessor's
/// cross-element consistency, positively checked (a conflict would have
/// panicked inside `dof_nodal_coords`).
#[test]
fn d526_prism_rt1_shared_tri_face_points_agree() {
    let mesh = load("data/d394_prism_stack.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();

    // The shared face's 3 dofs = the face block of tri {3,4,5}.
    let block = space
        .face_dofs(fem_space::dof_manager::FaceKey::new(3, 4, 5))
        .expect("shared tri face");
    assert_eq!(block.len(), 3);
    // Face verts (ids 3,4,5) = (0,0,1), (1,0,1), (0,1,1): the plane z = 1.
    let plane = |p: [f64; 3]| (p[2] - 1.0).abs();
    for &d in &block {
        let p = nodal[d as usize];
        assert!(
            plane(p) < 1e-9,
            "dof {d} nodal point {p:?} not on the shared tri plane z = 1"
        );
        // Strictly inside the triangle {3,4,5}: x, y > 0 and x + y < 1.
        assert!(
            p[0] > 1e-9 && p[1] > 1e-9 && p[0] + p[1] < 1.0 - 1e-9,
            "dof {d} nodal point {p:?} not strictly inside the shared tri"
        );
    }
}

/// One-pyramid RT1 (`tmp/d392/one_pyramid.mesh`): the element's 28 local
/// slots land on the probe's 28 distinct physical points; the 16 boundary
/// dofs (4 tri faces x 3 + base quad x 4) sit on their faces.
#[test]
fn d526_pyramid_rt1_element_points_match_mfem() {
    let mesh = load("tmp/d392/one_pyramid.mesh");
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();
    assert_eq!(space.n_dofs(), 28, "Fuentes RT1 vsize (MFEM pyr_incode)");

    let mut got: Vec<_> = space
        .element_dofs(0)
        .iter()
        .map(|&d| key3(&nodal[d as usize]))
        .collect();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got, golden("PYR1E0"), "pyramid element-0 point set vs MFEM");

    // Boundary dofs: base-quad dofs on z = 0, apex-face dofs strictly below
    // the apex plane through (0,0,1),(1,0,1)... — simply: every boundary dof
    // point lies on one of the 5 face planes; the base four carry z = 0.
    let all_tags = mesh.unique_boundary_tags();
    let bdr = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
    assert_eq!(bdr.len(), 16);
    let mut base = 0;
    for &d in &bdr {
        let p = nodal[d as usize];
        if p[2].abs() < 1e-12 {
            base += 1;
        }
    }
    assert_eq!(base, 4, "the base quad carries (k+1)^2 = 4 boundary dofs");
}

/// 2-D gate (edges): on a unit-square tri mesh, RT1's 6 edge dofs of each
/// element sit at the two Gauss-Legendre points of their edges and the 2
/// interior components share the element centroid — the
/// `mfem_tri_nodal_dofs` table the accessor maps.
#[test]
fn d526_tri2d_rt1_edges_at_gauss_points() {
    let mesh = Mesh::<2>::unit_square_tri(1);
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();

    let gl = fem_element::quadrature::gauss_legendre_01(2).0;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let c = |i: usize| {
            let p = mesh.node_coords(verts[i]);
            [p[0], p[1]]
        };
        // Want: the GL2 points along the three `TRI_EDGES` listings.
        let mut want: Vec<[i64; 2]> = Vec::new();
        for (p, q) in [(0usize, 1usize), (1, 2), (2, 0)] {
            for &t in &gl {
                let x = [c(p)[0] + t * (c(q)[0] - c(p)[0]), c(p)[1] + t * (c(q)[1] - c(p)[1])];
                want.push([(x[0] * 1e6).round() as i64, (x[1] * 1e6).round() as i64]);
            }
        }
        want.sort_unstable();
        let mut got: Vec<[i64; 2]> = space.element_dofs(e)[..6]
            .iter()
            .map(|&d| {
                let p = nodal[d as usize];
                [(p[0] * 1e6).round() as i64, (p[1] * 1e6).round() as i64]
            })
            .collect();
        got.sort_unstable();
        assert_eq!(got, want, "elem {e}: edge dofs at the GL points of their edges");

        // The 2 interior component dofs share the element centroid.
        let dofs = space.element_dofs(e);
        let (i0, i1) = (dofs[6], dofs[7]);
        assert_eq!(nodal[i0 as usize], nodal[i1 as usize], "centroid, doubled");
        let mut cen = [0.0_f64; 2];
        for &v in verts {
            let p = mesh.node_coords(v);
            cen[0] += p[0] / 3.0;
            cen[1] += p[1] / 3.0;
        }
        assert!((nodal[i0 as usize][0] - cen[0]).abs() < 1e-12);
        assert!((nodal[i0 as usize][1] - cen[1]).abs() < 1e-12);
    }
}
