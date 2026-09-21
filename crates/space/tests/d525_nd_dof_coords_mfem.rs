//! D525 — `HCurlSpace::dof_coords` covers every geometry and every interior
//! dof: the per-dof physical coordinates must equal MFEM 4.10's `FE::Nodes`
//! point-value sites (MFEM `Project_ND` semantics, `σ_j(Φ) = Φ(x_j)·J t_j`).
//!
//! Golden data: `tmp/d525/d525_nd_coords_probe.cpp` against the MFEM 4.10
//! serial tree (`-I$HOME/mfem410 -L$HOME/mfem410_ser`), archived quantised to
//! 1e-6 in `tests/data/d525_nd_coords_mfem.txt` — one `# <mesh> <order>`
//! section per run, one `dof key` line per global dof, in MFEM's global
//! numbering (entity-major: edges, faces, interiors — the D158 layout this
//! space mirrors).  The probe aborts on any cross-element coordinate
//! disagreement (`conflicts=0` in every archived `PROBE` line).
//!
//! Coverage: unit prism / unit pyramid / base-glued pyramid pair / 2x2x1
//! wedge grid / 2x2x2 hex grid / 1x1x1 tet grid at ND2+ND3 (ND1 where the
//! geometry carries face dofs at all), plus the beam-wedge boundary set:
//! MFEM's `GetBoundaryTrueDofs` = 202 = 51x2 edge + 24x4 quad-face + 2x2
//! tri-face dofs — the collector missed the 4 boundary tri-face dofs before
//! D525 (the tet-only `face_anchor` lookup failed on prism faces), which is
//! where the round-55 figure `dofs=198` came from.

use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::HCurlSpace;

const GOLDEN: &str = include_str!("data/d525_nd_coords_mfem.txt");
const BEAMWEDGE_ESS: &str = include_str!("data/d525_beamwedge_nd2_ess.txt");

/// Quantised dof-coordinate key (the D124/D505 oracle key function).
fn key3(c: &[f64; 3]) -> [i64; 3] {
    [
        (c[0] * 1e6).round() as i64,
        (c[1] * 1e6).round() as i64,
        (c[2] * 1e6).round() as i64,
    ]
}

fn load(rel: &str) -> Mesh<3> {
    let path = format!("{}/tests/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

fn load2d(rel: &str) -> Mesh<2> {
    let path = format!("{}/tests/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh2d.unwrap_or_else(|| panic!("{rel} must be a 2-D mesh"))
}

fn load_repo_data(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"))
}

/// `dof id -> quantised MFEM coordinate` of one `# <mesh> <order>` section.
fn golden_section(mesh: &str, order: u8) -> Vec<[i64; 3]> {
    let header = format!("{mesh} {order}");
    let mut out: Vec<[i64; 3]> = Vec::new();
    let mut in_section = false;
    for line in GOLDEN.lines() {
        if let Some(rest) = line.strip_prefix('#') {
            in_section = rest.trim() == header;
            continue;
        }
        if !in_section || line.trim().is_empty() {
            continue;
        }
        let t: Vec<i64> = line
            .split_whitespace()
            .map(|tok| tok.parse().expect("golden token"))
            .collect();
        assert_eq!(t.len(), 4, "golden line {line:?}");
        let dof = t[0] as usize;
        if dof >= out.len() {
            out.resize(dof + 1, [0; 3]);
        }
        out[dof] = [t[1], t[2], t[3]];
    }
    assert!(!out.is_empty(), "golden section {header} missing");
    out
}

fn femrs_keys(space: &HCurlSpace<Mesh<3>>) -> Vec<[i64; 3]> {
    space
        .dof_coords()
        .iter()
        .map(|c| key3(c))
        .collect()
}

fn femrs_keys_2d(space: &HCurlSpace<Mesh<2>>) -> Vec<[i64; 3]> {
    space
        .dof_coords()
        .iter()
        .map(|c| key3(c))
        .collect()
}

/// The full per-dof oracle: same dof count as MFEM, identical coordinate key
/// per global id, and no dof left at the `[0,0,0]` placeholder.
fn assert_per_dof(mesh: &str, order: u8, mesh_file: &str) {
    let space = HCurlSpace::new(load(mesh_file), order);
    let want = golden_section(mesh, order);
    assert_eq!(
        space.n_dofs(),
        want.len(),
        "{mesh} ND{order}: fem-rs ndofs != MFEM ndofs"
    );
    let got = femrs_keys(&space);
    assert_per_dof_keys(mesh, order, &got, &want);
}

/// 2-D variant of [`assert_per_dof`].
fn assert_per_dof_2d(mesh: &str, order: u8, mesh_file: &str) {
    let space = HCurlSpace::new(load2d(mesh_file), order);
    let want = golden_section(mesh, order);
    assert_eq!(
        space.n_dofs(),
        want.len(),
        "{mesh} ND{order}: fem-rs ndofs != MFEM ndofs"
    );
    let got = femrs_keys_2d(&space);
    assert_per_dof_keys(mesh, order, &got, &want);
}

fn assert_per_dof_keys(mesh: &str, order: u8, got: &[[i64; 3]], want: &[[i64; 3]]) {
    let mut mismatches = Vec::new();
    for (g, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        if a != b {
            mismatches.push(format!("  dof {g}: femrs {a:?} vs mfem {b:?}"));
            if mismatches.len() >= 8 {
                break;
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{mesh} ND{order}: {}+ per-dof coordinate mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    let origins = got.iter().filter(|k| **k == [0, 0, 0]).count();
    assert_eq!(origins, 0, "{mesh} ND{order}: dofs at the origin placeholder");
}

#[test]
fn d525_unit_prism_per_dof_coords_match_mfem() {
    assert_per_dof("unit-prism", 1, "data/d525_unit_prism.mesh");
    assert_per_dof("unit-prism", 2, "data/d525_unit_prism.mesh");
    assert_per_dof("unit-prism", 3, "data/d525_unit_prism.mesh");
}

#[test]
fn d525_unit_pyramid_per_dof_coords_match_mfem() {
    assert_per_dof("unit-pyramid", 1, "data/d525_unit_pyramid.mesh");
    assert_per_dof("unit-pyramid", 2, "data/d525_unit_pyramid.mesh");
    assert_per_dof("unit-pyramid", 3, "data/d525_unit_pyramid.mesh");
}

#[test]
fn d525_pyramid_pair_per_dof_coords_match_mfem() {
    // Two base-glued pyramids: exercises shared-face anchors across elements
    // (the probe reports conflicts=0, so MFEM itself is orientation-consistent
    // on the shared quad and its edges).
    assert_per_dof("pyramid-pair", 2, "data/d525_pyramid_pair.mesh");
    assert_per_dof("pyramid-pair", 3, "data/d525_pyramid_pair.mesh");
}

#[test]
fn d525_prism221_per_dof_coords_match_mfem() {
    assert_per_dof("prism221", 1, "data/d525_prism221.mesh");
    assert_per_dof("prism221", 2, "data/d525_prism221.mesh");
    assert_per_dof("prism221", 3, "data/d525_prism221.mesh");
}

#[test]
fn d525_hex222_per_dof_coords_match_mfem() {
    assert_per_dof("hex222", 2, "data/d525_hex222.mesh");
    assert_per_dof("hex222", 3, "data/d525_hex222.mesh");
}

#[test]
fn d525_tet111_per_dof_coords_match_mfem() {
    assert_per_dof("tet111", 2, "data/d525_tet111.mesh");
    assert_per_dof("tet111", 3, "data/d525_tet111.mesh");
}

/// 2-D interiors (tri barycentric points, quad GL/GLL tensor points) share
/// the same MFEM `FE::Nodes` semantics — covered on 2x2 triangle/quad grids.
#[test]
fn d525_2d_tri_quad_per_dof_coords_match_mfem() {
    assert_per_dof_2d("tri22", 2, "data/d525_tri22.mesh");
    assert_per_dof_2d("tri22", 3, "data/d525_tri22.mesh");
    assert_per_dof_2d("quad22", 2, "data/d525_quad22.mesh");
    assert_per_dof_2d("quad22", 3, "data/d525_quad22.mesh");
}

/// The beam-wedge boundary set: 202 dofs (MFEM `ness=202`), every one on its
/// own physical point (round 55 measured 198 dofs collapsing onto 103 keys —
/// 102 edge keys plus the `[0,0,0]` placeholder holding the 96 quad-face
/// dofs), and the key set equal to MFEM's essential set.
#[test]
fn d525_beamwedge_nd2_boundary_set_matches_mfem() {
    let mesh = load_repo_data("data/beam-wedge.mesh");
    let all_tags = mesh.unique_boundary_tags();
    let space = HCurlSpace::new(mesh.clone(), 2);

    let want_ess: Vec<u32> = BEAMWEDGE_ESS
        .split_whitespace()
        .skip(1)
        .map(|t| t.parse().expect("ess id"))
        .collect();
    assert_eq!(want_ess.len(), 202, "MFEM beam-wedge ND2 essential set");

    let dofs = boundary_dofs_hcurl(&mesh, &space, &all_tags);
    assert_eq!(
        dofs.len(),
        202,
        "boundary collector must now include the 4 tri-face dofs (MFEM parity)"
    );
    let coords = space.dof_coords();
    let got: std::collections::BTreeSet<[i64; 3]> =
        dofs.iter().map(|&d| key3(&coords[d as usize])).collect();
    assert_eq!(
        got.iter().filter(|k| **k == [0, 0, 0]).count(),
        0,
        "no boundary dof at the origin placeholder"
    );
    // Triangular-face dofs come in tangent pairs at the SAME physical point
    // (MFEM too: 2 caps x 1 shared point), so the key set has 200 entries,
    // not 202 — the oracle is the MFEM essential set itself.
    let section = golden_section("beamwedge", 2);
    let want: std::collections::BTreeSet<[i64; 3]> =
        want_ess.iter().map(|&g| section[g as usize]).collect();
    assert_eq!(got.len(), want.len(), "distinct boundary key count vs MFEM");
    assert_eq!(got, want, "boundary key set differs from MFEM's");
}
