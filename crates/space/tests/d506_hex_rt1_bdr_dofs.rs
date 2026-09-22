//! D506 — hex RT1 boundary dof block: mechanism determination.
//!
//! The round-54 registration (`tmp/d124/oracle_report.md:21`) says the hex RT1
//! boundary set measures 26 instead of MFEM's 96 with an unexposed quad-face
//! `(k+1)^2` block.  **That mechanism is wrong**, and both halves of it are
//! falsified here:
//!
//! 1. `HDivSpace::face_dofs` *does* return the whole `(k+1)^2` block for a quad
//!    face (`crates/space/src/hdiv.rs:1442`), and
//!    `crates/space/src/constraints/dirichlet.rs:498-522` already walks all four
//!    vertex triplets of a boundary quad until one hits.  On the 2x2x2 hex unit
//!    cube `boundary_dofs_hdiv` returns **96** ids — MFEM 4.10's
//!    `ness_vdof = ness_true = 96` — i.e. 24 boundary quads x 4 dofs.  Nothing
//!    is dropped; there is no key-normalisation defect either (the collector
//!    hits all 24 faces with the *first* triplet alternative, see
//!    `d506_hex_rt1_boundary_blocks_are_the_24_face_blocks`).
//!
//! 2. The `26` is a **key-function artifact**: the D124 oracle keys dofs by
//!    `HDivSpace::dof_coords`, which is the D414 *geometric anchor* convention
//!    (`crates/space/src/hdiv.rs:1477-1488` documents it as "not MFEM's
//!    reference-frame interpolation nodes … a permutation anchor … unique and
//!    face-consistent within one block").  For `k = 1` the quad-face lattice is
//!    the `(k+1)^2` *equispaced* grid = the face's four **corners**, so the 96
//!    face dofs land on the 26 boundary vertices of the cube
//!    (`tmp/d124/femrs_rt1_hex.txt`).  No dof is lost — the key aliases them.
//!
//! The residual capability gap is therefore on the MFEM-nodal side: fem-rs has
//! no accessor for the `FE::Nodes` points of H(div) face dofs (MFEM's
//! `RT_HexahedronElement::Nodes`, the `(k+1)` Gauss-Legendre tensor points per
//! face direction).  This file pins the MFEM truth against fem-rs *without*
//! touching `crates/space/src/hdiv.rs` (lane ownership): the per-face block
//! sizes come from `HDivSpace::face_dofs`, the nodal points are rebuilt from the
//! boundary face rings with the same Gauss-Legendre tensor rule the probe
//! shows.  The accessor gap itself is registered as a debt (D526) for the
//! hdiv.rs lane.
//!
//! Oracle: `tmp/r55/d505_bdr_probe.cpp` (MFEM 4.10 `$HOME/mfem410_ser`,
//! `Mesh::MakeCartesian3D(2,2,2,HEXAHEDRON)`, archived `tmp/r55/out_rt_1.txt`):
//!
//! ```text
//! PROBE family=rt order=1 mesh=2x2x2 ne=8 nbe=24 ndofs=240 vsize=240
//!       truesize=240 ness_vdof=96 ness_true=96 unique_keys=96
//! CLASS entity_dim=1 count=96        # all 96 on quad-face interiors
//! PROBE family=rt order=2 ... ndofs=756 ... ness_vdof=216 ness_true=216
//! ```
//!
//! (MFEM's RT hex `GetNDofs` is 240 at order 1 == fem-rs `HDivSpace::n_dofs`
//! 240 — the space sizes agree; only the face nodal points were unrepresented.)

use std::collections::HashSet;

use fem_element::quadrature::gauss_legendre_01;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::dof_manager::FaceKey;
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;

/// `data/d505_d506_hex_bdr_keys_mfem.txt` — the probe's `DKEY` lines.
const GOLDEN: &str = include_str!("data/d505_d506_hex_bdr_keys_mfem.txt");

const ALL_TAGS: [i32; 6] = [1, 2, 3, 4, 5, 6];

fn key3(c: &[f64; 3]) -> [i64; 3] {
    [
        (c[0] * 1e6).round() as i64,
        (c[1] * 1e6).round() as i64,
        (c[2] * 1e6).round() as i64,
    ]
}

/// One `#<family>` section of the golden table, sorted.
fn golden(family: &str) -> Vec<[i64; 3]> {
    let mut out = Vec::new();
    let mut in_section = false;
    for line in GOLDEN.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('#') {
            if !rest.contains(' ') {
                in_section = rest == family;
            }
            continue;
        }
        if line.is_empty() || !in_section {
            continue;
        }
        let n: Vec<i64> = line
            .split_whitespace()
            .map(|t| t.parse().expect("golden table token"))
            .collect();
        assert_eq!(n.len(), 3, "golden line {line:?}");
        out.push([n[0], n[1], n[2]]);
    }
    assert!(!out.is_empty(), "golden section {family} missing");
    out.sort_unstable();
    out
}

/// The D124 oracle mesh: `Mesh::MakeCartesian3D(2,2,2,HEXAHEDRON)`.
fn hex_cube() -> Mesh<3> {
    Mesh::<3>::unit_cube_hex(2)
}

/// The `(k+1)^2`-dof block of a boundary quad face, through the same four
/// triplets `boundary_dofs_hdiv` tries.
fn face_block(space: &HDivSpace<Mesh<3>>, nodes: &[u32]) -> Option<Vec<u32>> {
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        .iter()
        .find_map(|&(i, j, k)| space.face_dofs(FaceKey::new(nodes[i], nodes[j], nodes[k])))
}

/// Bilinear point of the quad with ring nodes `c0..c3` at `(u, v)`.
fn bilerp(c: &[[f64; 3]; 4], u: f64, v: f64) -> [f64; 3] {
    let mut p = [0.0_f64; 3];
    for d in 0..3 {
        p[d] = (1.0 - u) * (1.0 - v) * c[0][d]
            + u * (1.0 - v) * c[1][d]
            + u * v * c[2][d]
            + (1.0 - u) * v * c[3][d];
    }
    p
}

/// The collector's set is exactly the union of the 24 boundary quads'
/// `(k+1)^2` = 4-dof blocks: no face block is missing (the D506 claim), no
/// block is partial, and the blocks are disjoint.
#[test]
fn d506_hex_rt1_boundary_blocks_are_the_24_face_blocks() {
    let mesh = hex_cube();
    assert_eq!(mesh.n_boundary_faces(), 24);
    let space = HDivSpace::new(mesh.clone(), 1);

    let dofs: HashSet<u32> = boundary_dofs_hdiv(space.mesh(), &space, &ALL_TAGS)
        .into_iter()
        .collect();
    assert_eq!(dofs.len(), 96, "MFEM ness_vdof = 96 = 24 boundary quads x 4");

    let mut union: HashSet<u32> = HashSet::new();
    let mut n_blocks = 0usize;
    for f in 0..mesh.n_boundary_faces() as u32 {
        assert!(ALL_TAGS.contains(&mesh.face_tag(f)));
        let nodes = mesh.face_nodes(f);
        assert_eq!(nodes.len(), 4, "hex boundary face is a quad");
        let block = face_block(&space, nodes)
            .unwrap_or_else(|| panic!("boundary quad {f} {nodes:?}: no (k+1)^2 block"));
        assert_eq!(block.len(), 4, "RT1 quad face block = (k+1)^2 = 4 dofs");
        for &d in &block {
            assert!(
                union.insert(d),
                "dof {d} appears in two boundary face blocks (overlap)"
            );
        }
        n_blocks += 1;
    }
    assert_eq!(n_blocks, 24);
    assert_eq!(
        union, dofs,
        "the collector must be exactly the union of the boundary face blocks"
    );
}

/// Per-dof comparison against MFEM: the 96 essential dofs are one per
/// `(k+1)^2` Gauss-Legendre tensor point of one boundary quad — the nodal
/// points MFEM reports (`RT_HexahedronElement::Nodes`: `(k+1)` GL points per
/// face direction, here 2 x 2 = 4 per face).  Each fem-rs face block has
/// exactly as many dofs as the face has MFEM nodal points, and the point set
/// reproduces the probe dump bitwise (quantised).
#[test]
fn d506_hex_rt1_boundary_nodal_points_match_mfem() {
    let mesh = hex_cube();
    let space = HDivSpace::new(mesh.clone(), 1);

    let (gl, _) = gauss_legendre_01(2); // (k+1) 1-D Gauss-Legendre nodes on [0,1]
    let mut got: Vec<[i64; 3]> = Vec::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        let nodes = mesh.face_nodes(f);
        let block = face_block(&space, &nodes).expect("boundary quad block");
        let mut ring = [[0.0_f64; 3]; 4];
        for (i, &n) in nodes.iter().enumerate() {
            let c = mesh.node_coords(n);
            ring[i] = [c[0], c[1], c[2]];
        }
        let mut pts: Vec<[i64; 3]> = Vec::new();
        for &u in &gl {
            for &v in &gl {
                pts.push(key3(&bilerp(&ring, u, v)));
            }
        }
        pts.sort_unstable();
        pts.dedup();
        assert_eq!(
            pts.len(),
            block.len(),
            "boundary quad {f}: {pts:?} nodal points vs {} dofs",
            block.len()
        );
        got.extend(pts);
    }
    got.sort_unstable();
    got.dedup();
    assert_eq!(got.len(), 96, "24 boundary quads x (k+1)^2 nodal points");

    let want = golden("RT1");
    assert_eq!(want.len(), 96, "MFEM RT1 golden set size");
    assert_eq!(
        got, want,
        "hex RT1 boundary nodal-point set differs from MFEM GetEssentialVDofs"
    );
}

/// `k >= 2` shares the same block rule, `(k+1)^2` dofs per boundary quad:
/// MFEM pins hex RT2 at `ness = 216 = 24 x 9`.
#[test]
fn d506_hex_rt2_boundary_set_size_matches_mfem() {
    let mesh = hex_cube();
    let space = HDivSpace::new(mesh.clone(), 2);
    let dofs = boundary_dofs_hdiv(space.mesh(), &space, &ALL_TAGS);
    assert_eq!(dofs.len(), 216, "hex RT2: 24 x (k+1)^2 = 216 (MFEM probe)");
    for f in 0..mesh.n_boundary_faces() as u32 {
        let nodes = mesh.face_nodes(f);
        let block = face_block(&space, &nodes).expect("boundary quad block");
        assert_eq!(block.len(), 9, "RT2 quad face block = 3x3 = 9 dofs");
    }
}

/// D526 (closed) — `HDivSpace::dof_nodal_coords` is the MFEM-nodal accessor
/// whose absence the file-level docs above registered as the residual gap.
/// Keyed at its own nodal points, the 96 essential boundary dofs land on
/// exactly the probe's 96-key golden table (`golden("RT1")`), point for
/// point — the accessor is not merely set-correct per face but reproduces
/// the very points MFEM's `GetEssentialVDofs`-keyed dump reports.
#[test]
fn d526_hex_rt1_boundary_nodal_accessor_hits_mfem_keys() {
    let mesh = hex_cube();
    let space = HDivSpace::new(mesh.clone(), 1);
    let nodal = space.dof_nodal_coords();
    let dofs = boundary_dofs_hdiv(space.mesh(), &space, &ALL_TAGS);
    assert_eq!(dofs.len(), 96);

    let mut got: Vec<[i64; 3]> = dofs.iter().map(|&d| key3(&nodal[d as usize])).collect();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got.len(), 96, "the 96 nodal points must be pairwise distinct");
    assert_eq!(got, golden("RT1"), "accessor keys vs MFEM GetEssentialVDofs keys");
}
