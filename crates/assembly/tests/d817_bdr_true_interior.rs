//! D817-1 — the "interior-coincident boundary elements get zero assembly"
//! semantics, sunk from the ex9-3D example layer (empty tag set) into the
//! Mesh layer + every bdr-assembly traversal.
//!
//! MFEM ground truth: `Mesh::GetBdrFaceTransformations` (mesh.cpp:1312)
//! answers INVALID for a boundary element whose face is a true interior face
//! (`FaceIsTrueInterior`), and every consumer (`BilinearForm`,
//! `LinearForm`, `NonlinearForm` bdr loops) skips the nullptr — on the
//! periodically identified cube all 54 stored boundary entries are
//! interior-coincident, so C++ ex9 charges no boundary term at all (the
//! round-81 probe: K·u0 equals the no-bdr assembly).
//!
//! The red proof ran on the pre-fix tree: `assemble_advection_boundary_full`
//! over `data/periodic-cube.mesh` with the real tag set produced a non-empty
//! boundary K (outflow faces of the seam charge `ipw·(un+|un|)·φφᵀ`), which
//! MFEM does not assemble.

use fem_assembly::dg::{assemble_advection_boundary_full, DGAdvectionIntegrator};
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_assembly::{Assembler, BoundaryMassIntegrator, face_dofs_p1};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::{FESpace, L2Space};

/// The periodic cube: every one of the 54 stored boundary entries is
/// interior-coincident (MFEM probe true_int=54 of NBE=54), so the DG
/// boundary-trace assembly must charge **nothing** — K empty, RHS zero —
/// exactly like MFEM's nullptr skip.  The volume assembly on the same
/// mesh must stay live (guards against a vacuous pass).
#[test]
fn d817_periodic_cube_bdr_assembly_is_zero_like_mfem() {
    let mfem = read_mfem_file("../../data/periodic-cube.mesh").expect("read periodic cube");
    let mesh = mfem.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.n_boundary_faces(), 54, "NBE of the file");
    let interior_count =
        (0..mesh.n_boundary_faces() as u32).filter(|&f| mesh.bdr_face_true_interior(f)).count();
    assert_eq!(interior_count, 54, "every stored entry is interior-coincident");

    let order = 1u8;
    let space = L2Space::new_with_basis(mesh.clone(), order, fem_space::L2Basis::GaussLobatto);
    let tags = mesh.unique_boundary_tags();
    assert!(!tags.is_empty(), "the stored entries carry real tags");

    let velocity = ConstantVectorCoeff(vec![1.0, 0.0, 0.0]);
    let (k_bdr, rhs_bc) =
        assemble_advection_boundary_full(&space, &velocity, &tags, &|_x| 0.0, order, 3, -1.0);
    assert_eq!(k_bdr.values.len(), 0, "boundary K must be empty (MFEM nullptr skip)");
    assert!(rhs_bc.iter().all(|&v| v == 0.0), "boundary RHS must stay zero");

    // Sanity: the same mesh's volume term DOES assemble (the periodic seam
    // coupling rides on the interior faces), so the zero above is a real
    // skip, not a broken assembler.
    let dg = DGAdvectionIntegrator { velocity, alpha: -1.0 };
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg], 3);
    assert!(k_vol.values.iter().any(|&v| v != 0.0), "volume assembly live");
}

/// Ordinary meshes have no interior-coincident entries, so the skip must be
/// a no-op there: the boundary mass over the unit square equals the perimeter
/// whether or not the mesh also carries synthetic interior-coincident
/// entries, and those entries alone assemble to an empty pattern.
#[test]
fn d817_ordinary_boundary_assembly_unchanged_interior_entry_skipped() {
    let plain = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
    let mut augmented = plain.clone();
    // The internal vertical line x = 0.5: three vertices, two edges, each
    // registered by both adjacent quads.  Append them as boundary entries
    // (tag 9) — interior-coincident by construction.
    let mid: Vec<u32> = (0..augmented.n_nodes() as u32)
        .filter(|&n| (augmented.coords_of(n)[0] - 0.5).abs() < 1e-12)
        .collect();
    assert_eq!(mid.len(), 3, "three vertices on the internal line");
    let mut line = mid;
    line.sort_by_key(|&n| (augmented.coords_of(n)[1] * 1e6).round() as i64);
    for w in line.windows(2) {
        augmented.face_conn.extend_from_slice(&[w[0], w[1]]);
        augmented.face_tags.push(9);
    }
    assert_eq!(augmented.n_faces(), plain.n_faces() + 2);

    let space = fem_space::H1Space::new(plain, 1);
    let fdofs = face_dofs_p1(space.mesh());
    let one = vec![1.0; space.n_dofs()];
    let mat_plain = Assembler::assemble_boundary_bilinear(
        space.n_dofs(),
        space.mesh(),
        &fdofs,
        1,
        &[kappa_integrator()],
        &[1, 2, 3, 4],
        3,
    );
    let total_plain: f64 = (0..space.n_dofs())
        .map(|i| dot_row(&mat_plain, i, &one))
        .sum();
    assert!(
        (total_plain - 4.0).abs() < 1e-12,
        "1ᵀM_Γ1 must equal the unit-square perimeter, got {total_plain}"
    );

    let space_aug = fem_space::H1Space::new(augmented, 1);
    let fdofs_aug = face_dofs_p1(space_aug.mesh());
    // Tag 9 alone: every entry is interior-coincident → empty pattern.
    let mat_nine = Assembler::assemble_boundary_bilinear(
        space_aug.n_dofs(),
        space_aug.mesh(),
        &fdofs_aug,
        1,
        &[kappa_integrator()],
        &[9],
        3,
    );
    assert_eq!(mat_nine.values.len(), 0, "interior-coincident entries are skipped");

    // Real boundary tags: identical total to the plain mesh.
    let mat_aug = Assembler::assemble_boundary_bilinear(
        space_aug.n_dofs(),
        space_aug.mesh(),
        &fdofs_aug,
        1,
        &[kappa_integrator()],
        &[1, 2, 3, 4],
        3,
    );
    let total_aug: f64 = (0..space_aug.n_dofs())
        .map(|i| dot_row(&mat_aug, i, &one))
        .sum();
    assert!((total_aug - total_plain).abs() < 1e-12);
}

fn kappa_integrator() -> &'static dyn fem_assembly::BoundaryBilinearIntegrator {
    use std::sync::OnceLock;
    static K: OnceLock<fem_assembly::BoundaryMassIntegrator> = OnceLock::new();
    K.get_or_init(|| fem_assembly::BoundaryMassIntegrator { kappa: 1.0, bdr_tags: vec![] })
}

fn dot_row(m: &fem_linalg::CsrMatrix<f64>, row: usize, x: &[f64]) -> f64 {
    (m.row_ptr[row]..m.row_ptr[row + 1])
        .map(|p| m.values[p] * x[m.col_idx[p] as usize])
        .sum()
}
