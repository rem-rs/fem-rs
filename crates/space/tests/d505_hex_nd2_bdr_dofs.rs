//! D505 — hex NDk boundary dof set: the quad-face dofs were invisible to any
//! coordinate key.
//!
//! `boundary_dofs_hcurl` already collected the whole set: for the 2x2x2 hex
//! unit cube at order 2 it returns **192** dof ids, exactly MFEM 4.10's
//! `ness_vdof = ness_true = 192` (`GetEssentialVDofs` over all boundary
//! attributes marks all dofs of the 24 boundary quads: `48 boundary edges x 2`
//! edge dofs + `24 faces x 2k(k-1) = 24 x 4` face-interior dofs).  What was
//! broken is the **mapping from dof to physical point**:
//! `HCurlSpace::dof_coords` filled edge dofs and tet-face dofs but had no
//! hex-quad-face branch, so every one of the 96 face-interior dofs kept the
//! `[0,0,0]` placeholder and the coordinate-keyed oracle of round 54
//! (`tmp/d124/oracle_report.md`, `tmp/d124/femrs_nd2_hex.txt`) measured
//! `96 distinct edge keys + 1 collapsed key = 97` instead of 192.
//!
//! The fix (`crates/space/src/hcurl.rs`, `HCurlSpace::dof_coords`) reads the
//! face-creating element's physical DOF points from the already-computed
//! `quad_face_anchor` table — the same `FE::Nodes` point-value sites MFEM
//! reports (`σ_m(Φ) = Φ(x_m)·t_m`).
//!
//! Oracle: `tmp/r55/d505_bdr_probe.cpp` (MFEM 4.10 `$HOME/mfem410_ser`,
//! `Mesh::MakeCartesian3D(2,2,2,HEXAHEDRON)`, dump archived in
//! `tmp/r55/out_nd_2.txt`):
//!
//! ```text
//! PROBE family=nd order=2 mesh=2x2x2 ne=8 nbe=24 ndofs=300 vsize=300
//!       truesize=300 ness_vdof=192 ness_true=192 unique_keys=192
//! CLASS entity_dim=1 count=96      # 1 interior reference coord = mesh edge
//! CLASS entity_dim=2 count=96      # 2 interior coords      = quad face
//! ```
//!
//! Extended probe runs pinned here as counts only (same probe, same mesh):
//! ND3 `ness=432` (48x3 = 144 edge + 24x12 = 288 face); RT1 `ness=96`,
//! RT2 `ness=216` (see `d506_hex_rt1_bdr_dofs.rs`).

use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::HCurlSpace;

/// `data/d505_d506_hex_bdr_keys_mfem.txt` — the probe's `DKEY` lines.
const GOLDEN: &str = include_str!("data/d505_d506_hex_bdr_keys_mfem.txt");

const ALL_TAGS: [i32; 6] = [1, 2, 3, 4, 5, 6];

/// Quantised dof-coordinate key (the D124 oracle's key function).
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
            // Section markers are `#ND2` / `#RT1`; prose headers contain a space.
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

#[test]
fn d505_hex_nd2_boundary_dof_ids_and_keys_match_mfem() {
    let mesh = hex_cube();
    assert_eq!(mesh.n_boundary_faces(), 24, "hex 2x2x2 boundary quads");
    let nd = HCurlSpace::new(mesh.clone(), 2);

    // 1. The collector's size is MFEM's essential vdof count (not 97 keys).
    let dofs = boundary_dofs_hcurl(&mesh, &nd, &ALL_TAGS);
    assert_eq!(
        dofs.len(),
        192,
        "hex ND2 boundary dof set must be MFEM's 192 \
         (48 boundary edges x 2 + 24 boundary quads x 2k(k-1))"
    );

    // 2. Every dof has its own physical point: no `[0,0,0]` collapse (the D505
    //    symptom was 96 face dofs sharing the origin placeholder).
    let coords = nd.dof_coords();
    let mut keys: Vec<[i64; 3]> = dofs
        .iter()
        .map(|&d| key3(&coords[d as usize]))
        .collect();
    keys.sort_unstable();
    let n_before_dedup = keys.len();
    keys.dedup();
    assert_eq!(
        keys.len(),
        n_before_dedup,
        "distinct dofs must have distinct coordinates (collapsed: {})",
        n_before_dedup - keys.len()
    );
    assert_eq!(
        keys.iter().filter(|k| **k == [0, 0, 0]).count(),
        0,
        "no boundary dof may sit at the [0,0,0] placeholder \
         (HCurlSpace::dof_coords quad-face branch)"
    );

    // 3. Per-dof set comparison against the MFEM 4.10 dump.
    let want = golden("ND2");
    assert_eq!(want.len(), 192, "MFEM ND2 golden set size");
    assert_eq!(
        keys, want,
        "hex ND2 boundary key set differs from MFEM GetEssentialVDofs"
    );
}

/// The edge-dof subset must stay MFEM-exact as well: for ND2 the 96 keys with
/// two unit-cube coordinates are the mesh-edge dofs; the other 96 are the
/// face-interior dofs of the 24 boundary quads (4 each).  This split is the
/// probe's `CLASS entity_dim` breakdown and pins the *composition* of the set,
/// not just its size.
#[test]
fn d505_hex_nd2_boundary_set_composition_matches_mfem() {
    let mesh = hex_cube();
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let dofs = boundary_dofs_hcurl(&mesh, &nd, &ALL_TAGS);
    let coords = nd.dof_coords();
    let mut keys: Vec<[i64; 3]> = dofs.iter().map(|&d| key3(&coords[d as usize])).collect();
    keys.sort_unstable();
    keys.dedup();

    // A dof on a mesh edge of the 2x2x2 grid has two coordinates on grid
    // lines (multiples of 0.5); a face-interior dof has exactly one.
    let n_grid = |k: &[i64; 3]| k.iter().filter(|v| **v % 500_000 == 0).count();
    let n_edge = keys.iter().filter(|k| n_grid(k) >= 2).count();
    let n_face = keys.iter().filter(|k| n_grid(k) == 1).count();
    assert_eq!((n_edge, n_face), (96, 96), "MFEM: 96 edge + 96 face dofs");
}

/// Orders beyond 2 share the same quad-face branch (`2k(k-1)` dofs per face):
/// MFEM pins hex ND3 at `ness = 432` — 48x3 edge + 24x12 face dofs.
#[test]
fn d505_hex_nd3_boundary_set_size_matches_mfem() {
    let mesh = hex_cube();
    let nd = HCurlSpace::new(mesh.clone(), 3);
    let dofs = boundary_dofs_hcurl(&mesh, &nd, &ALL_TAGS);
    assert_eq!(dofs.len(), 432, "hex ND3: 48x3 + 24x12 = 432 (MFEM probe)");
    let coords = nd.dof_coords();
    let mut keys: Vec<[i64; 3]> = dofs.iter().map(|&d| key3(&coords[d as usize])).collect();
    keys.sort_unstable();
    let n = keys.len();
    keys.dedup();
    assert_eq!(keys.len(), n, "ND3 dof coordinates must be distinct");
}
