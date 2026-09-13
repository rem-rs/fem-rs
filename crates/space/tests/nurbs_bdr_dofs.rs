//! `NURBSExtension::GenerateBdrElementDofTable` (D96) against MFEM 4.10: the
//! signed boundary DOF tables of the H¹ (`Mode::H_1`), H(div) (`Mode::H_DIV`)
//! and H(curl) (`Mode::H_CURL`) NURBS spaces.
//!
//! Each fixture is a verbatim dump of a small C++ probe that builds the spaces
//! exactly like `nurbs_ex5`/`nurbs_ex24` do
//! (`NURBSExtension(mesh->NURBSext, order)` plus `NURBS_HDivFECollection` /
//! `NURBS_HCurlFECollection`) and prints
//!
//! * `BEL_H1`/`BEL_HDIV`/`BEL_HCURL` — `FiniteElementSpace::GetBdrElementDofs(b)`
//!   for every mesh boundary element, **with MFEM's sign encoding** (a negative
//!   entry `e` is DOF `-1 - e` with the opposite sign);
//! * `A` — the boundary attributes; `BEFACE` — the boundary element's vertices
//!   and `GetBdrElementFaceIndex` (`fn`, which `Mode::H_DIV`'s sign rule is
//!   written in terms of);
//! * `ELDIV` — the H(div) element DOF rows, and the space sizes.
//!
//! Configurations (MFEM's probe invocation): `square-nurbs.mesh -o 1 -r 0`,
//! `square-nurbs.mesh -o 1 -r 1`, `pipe-nurbs-2d.mesh -o 1 -r 0`,
//! `disc-nurbs.mesh -o 1 -r 0` (5 patches — H¹ and H(curl) only, MFEM's
//! `GetDivExtension` rejects multi-patch meshes), `cube-nurbs.mesh -o 1 -r 0`,
//! `cube-nurbs.mesh -o 1 -r 1` and `cube-nurbs.mesh -o 2 -r 1`.

use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHCurlSpace, NurbsHDivSpace};

const SQUARE: &str = include_str!("../../../data/square-nurbs.mesh");
const PIPE2D: &str = include_str!("../../../data/pipe-nurbs-2d.mesh");
const DISC: &str = include_str!("../../../data/disc-nurbs.mesh");
const CUBE: &str = include_str!("../../../data/cube-nurbs.mesh");
const PIPE3D: &str = include_str!("../../../data/pipe-nurbs.mesh");

const REF_SQUARE_R0: &str = include_str!("data/nurbs_beldof_square_r0_o1_mfem.txt");
const REF_SQUARE_R1: &str = include_str!("data/nurbs_beldof_square_r1_o1_mfem.txt");
const REF_PIPE2D: &str = include_str!("data/nurbs_beldof_pipe2d_r0_o1_mfem.txt");
const REF_DISC: &str = include_str!("data/nurbs_beldof_disc-nurbs_o1_r0_mfem.txt");
const REF_CUBE_R0: &str = include_str!("data/nurbs_beldof_cube-nurbs_o1_r0_mfem.txt");
const REF_CUBE_R1: &str = include_str!("data/nurbs_beldof_cube-nurbs_o1_r1_mfem.txt");
const REF_CUBE_O2: &str = include_str!("data/nurbs_beldof_cube-nurbs_o2_r1_mfem.txt");
const REF_PIPE3D: &str = include_str!("data/nurbs_beldof_pipe3d_o1_r0_mfem.txt");

/// The `BEL_<tag> <b> n=<n> <dofs…>` rows of a fixture, per boundary element.
fn expected(ref_text: &str, tag: &str) -> Vec<Vec<i64>> {
    let prefix = format!("BEL_{tag} ");
    let mut rows: Vec<(usize, Vec<i64>)> = Vec::new();
    for line in ref_text.lines() {
        let Some(rest) = line.strip_prefix(&prefix) else { continue };
        let (b, rest) = rest.split_once(' ').expect("boundary element index");
        let (n, dofs) = rest.split_once(' ').unwrap_or((rest, ""));
        let n: usize = n.strip_prefix("n=").expect("n=").parse().expect("count");
        let vals: Vec<i64> = dofs.split_whitespace().map(|v| v.parse().expect("dof")).collect();
        assert_eq!(vals.len(), n, "fixture row {b} of {tag}");
        rows.push((b.parse().expect("index"), vals));
    }
    assert!(!rows.is_empty(), "fixture has no BEL_{tag} rows");
    rows.sort_by_key(|&(b, _)| b);
    for (i, (b, _)) in rows.iter().enumerate() {
        assert_eq!(i, *b, "fixture rows must be consecutive");
    }
    rows.into_iter().map(|(_, v)| v).collect()
}

/// The number of rows a fixture has for `tag`.
fn count(ref_text: &str, tag: &str) -> usize {
    expected(ref_text, tag).len()
}

fn assert_rows(got: &[Vec<i64>], want: &[Vec<i64>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (b, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{what}: boundary element {b}");
    }
}

#[test]
fn h1_boundary_dofs_square() {
    let s = NurbsFESpace::from_mesh_str(SQUARE, 0, &[1]).expect("space");
    assert_eq!(count(REF_SQUARE_R0, "H1"), 4);
    assert_rows(&s.boundary_dof_table(), &expected(REF_SQUARE_R0, "H1"), "H1 r0");

    let s = NurbsFESpace::from_mesh_str(SQUARE, 1, &[1]).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_SQUARE_R1, "H1"), "H1 r1");

    // `nurbs_ex1`'s order-2 configuration.
    let s = NurbsFESpace::from_mesh_str(SQUARE, 1, &[2]).expect("space");
    assert_eq!(s.n_dofs(), 16);
    assert_eq!(s.boundary_dof_table().len(), 8);
    assert!(s.boundary_dof_table().iter().all(|r| r.len() == 3));
}

#[test]
fn h1_boundary_dofs_pipe2d() {
    let s = NurbsFESpace::from_mesh_str(PIPE2D, 0, &[1]).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_PIPE2D, "H1"), "pipe2d H1");
}

#[test]
fn h1_boundary_dofs_multi_patch_disc() {
    let s = NurbsFESpace::from_mesh_str(DISC, 0, &[1]).expect("space");
    assert_eq!(s.n_dofs(), 25);
    assert_rows(&s.boundary_dof_table(), &expected(REF_DISC, "H1"), "disc H1");
}

#[test]
fn hcurl_boundary_dofs_drops_blocks() {
    // 2-D H(curl): a component's block survives on the entities whose
    // tangential knot-vector order differs from the extension's maximal one.
    let square = NurbsHCurlSpace::from_mesh_str(SQUARE, 0, 1).expect("space");
    assert_rows(&square.boundary_dof_table(), &expected(REF_SQUARE_R0, "HCURL"), "square H(curl)");

    let square = NurbsHCurlSpace::from_mesh_str(SQUARE, 1, 1).expect("space");
    assert_rows(&square.boundary_dof_table(), &expected(REF_SQUARE_R1, "HCURL"), "square r1 H(curl)");

    let pipe = NurbsHCurlSpace::from_mesh_str(PIPE2D, 0, 1).expect("space");
    assert_rows(&pipe.boundary_dof_table(), &expected(REF_PIPE2D, "HCURL"), "pipe2d H(curl)");
}

#[test]
fn hdiv_boundary_dofs_square() {
    let s = NurbsHDivSpace::from_mesh_str(SQUARE, 0, 1).expect("space");
    assert_eq!(s.n_dofs(), 12);
    assert_rows(&s.boundary_dof_table(), &expected(REF_SQUARE_R0, "HDIV"), "square H(div)");

    let s = NurbsHDivSpace::from_mesh_str(SQUARE, 1, 1).expect("space");
    assert_eq!(s.n_dofs(), 24);
    assert_rows(&s.boundary_dof_table(), &expected(REF_SQUARE_R1, "HDIV"), "square r1 H(div)");

    let s = NurbsHDivSpace::from_mesh_str(PIPE2D, 0, 1).expect("space");
    assert_eq!(s.n_dofs(), 24);
    assert_rows(&s.boundary_dof_table(), &expected(REF_PIPE2D, "HDIV"), "pipe2d H(div)");
}

#[test]
fn hdiv_boundary_dofs_cube() {
    let s = NurbsHDivSpace::from_mesh_str(CUBE, 0, 1).expect("space");
    assert_eq!(s.n_dofs(), 36);
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R0, "HDIV"), "cube H(div)");

    let s = NurbsHDivSpace::from_mesh_str(CUBE, 1, 1).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R1, "HDIV"), "cube r1 H(div)");

    let s = NurbsHDivSpace::from_mesh_str(CUBE, 1, 2).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_O2, "HDIV"), "cube o2 H(div)");
}

#[test]
fn hcurl_boundary_dofs_cube() {
    let s = NurbsHCurlSpace::from_mesh_str(CUBE, 0, 1).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R0, "HCURL"), "cube H(curl)");

    let s = NurbsHCurlSpace::from_mesh_str(CUBE, 1, 1).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R1, "HCURL"), "cube r1 H(curl)");
}

/// A 3-D mesh with 8 patches and *generated* boundary elements
/// (`boundary 0` in the mesh file): MFEM's `Mesh::GenerateBoundaryElements`
/// path, whose boundary DOF rows the H¹ table must reproduce as well.
#[test]
fn h1_boundary_dofs_pipe3d() {
    let s = NurbsFESpace::from_mesh_str(PIPE3D, 0, &[1]).expect("space");
    assert_eq!(s.boundary_dof_table().len(), 24);
    assert_rows(&s.boundary_dof_table(), &expected(REF_PIPE3D, "H1"), "pipe3d H1");
}

#[test]
fn h1_boundary_dofs_cube() {
    let s = NurbsFESpace::from_mesh_str(CUBE, 0, &[1]).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R0, "H1"), "cube H1");

    let s = NurbsFESpace::from_mesh_str(CUBE, 1, &[1]).expect("space");
    assert_rows(&s.boundary_dof_table(), &expected(REF_CUBE_R1, "H1"), "cube r1 H1");
}

/// The signed encoding itself: `-1 - dof` marks a DOF whose basis function
/// enters with the opposite sign, and the H(div) table's negative entries are
/// exactly the low-side boundary entities.
#[test]
fn signed_encoding_matches_low_sides() {
    use fem_space::nurbs_extension::unsign_dof;
    let s = NurbsHDivSpace::from_mesh_str(SQUARE, 0, 1).expect("space");
    let table = s.boundary_dof_table();
    let flipped: Vec<bool> = table.iter().map(|r| r.iter().any(|&e| e < 0)).collect();
    assert_eq!(flipped, vec![true, false, true, false]);
    for (b, row) in table.iter().enumerate() {
        for &e in row {
            let (dof, sign) = unsign_dof(e);
            assert!(dof < s.n_dofs(), "row {b}: dof {dof} out of range");
            assert_eq!(sign, if e < 0 { -1 } else { 1 });
        }
    }
    // The element DOF table has no sign encoding.
    assert!(s.element_dofs(0).iter().all(|&d| d < s.n_dofs()));
}
