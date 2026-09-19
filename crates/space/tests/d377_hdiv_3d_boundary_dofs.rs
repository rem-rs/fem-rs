//! D377 — 3-D HDiv boundary dofs must expose the whole RT face block.
//!
//! Oracle: the C++ probe `tmp/d392/ess3d_probe.cpp` built against MFEM 4.10
//! (`$HOME/mfem410_ser`, archived output `$HOME/work/d392/d392_probe.out`).
//! The probe loads the same meshes (`Mesh(file, 1, 1)` + one
//! `UniformRefinement()`), builds `RT_FECollection(k, 3)` spaces and prints
//! `GetVSize()` together with `FiniteElementSpace::GetBoundaryTrueDofs`
//! (`fem/fespace.cpp:668` — marks ALL boundary attributes internally).
//! An order-k RT face carries `(k+1)(k+2)/2` dofs on tetrahedra and
//! `(k+1)^2` on hexahedra, so the pinned ess counts are
//!
//! ```text
//! mesh=beam-tet.mesh   k=0 ne=384 vsize=904    ess=272
//! mesh=beam-tet.mesh   k=1 ne=384 vsize=3864   ess=816    (272 faces x 3)
//! mesh=beam-tet.mesh   k=2 ne=384 vsize=10032  ess=1632   (272 faces x 6)
//! mesh=beam-tet.mesh   k=3 ne=384 vsize=20560  ess=2720   (272 faces x 10)
//! mesh=inline-hex.mesh k=0 ne=512 vsize=1728   ess=384
//! mesh=inline-hex.mesh k=1 ne=512 vsize=13056  ess=1536   (384 faces x 4)
//! mesh=inline-hex.mesh k=2 ne=512 vsize=43200  ess=3456   (384 faces x 9)
//! mesh=inline-hex.mesh k=3 ne=512 vsize=101376 ess=6144   (384 faces x 16)
//! ```
//!
//! Before D377 fem-rs returned one dof per boundary face at every order
//! (272 / 384 regardless of k), silently under-constraining RTk ≥ 1 BCs.

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_space::constraints::boundary_dofs_hdiv;
use fem_space::{FESpace, HDivSpace};

/// Load an MFEM mesh file (relative to the repo root) and refine it once,
/// exactly like the C++ probe.
fn load_refined(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = read_mfem_file(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"));
    refine_uniform_3d(&mesh)
}

/// `(mesh file, [(k, MFEM GetVSize, MFEM GetBoundaryTrueDofs count)])`.
/// beam-tet k=3 (ess=2720) is not pinned here: `HDivSpace::validate_order`
/// still caps tet RT at order 2 (no TetRT3 reference element yet) — the C++
/// number is recorded in the module docs for whoever lifts that cap.
const ORACLE: &[(&str, &[(u8, usize, usize)])] = &[
    (
        "data/beam-tet.mesh",
        &[(0, 904, 272), (1, 3864, 816), (2, 10032, 1632)],
    ),
    (
        "data/inline-hex.mesh",
        &[
            (0, 1728, 384),
            (1, 13056, 1536),
            (2, 43200, 3456),
            (3, 101376, 6144),
        ],
    ),
];

#[test]
fn boundary_ess_counts_match_mfem_get_boundary_true_dofs() {
    for &(rel, rows) in ORACLE {
        let mesh = load_refined(rel);
        let all_tags = mesh.unique_boundary_tags();
        assert!(!all_tags.is_empty(), "{rel}: no boundary tags");
        for &(k, vsize, ess) in rows {
            let space = HDivSpace::new(mesh.clone(), k);
            // Bonus pin: the total space size must agree with MFEM's GetVSize.
            assert_eq!(space.n_dofs(), vsize, "{rel} k={k}: GetVSize mismatch");

            let dofs = boundary_dofs_hdiv(space.mesh(), &space, &all_tags);
            assert_eq!(
                dofs.len(), ess,
                "{rel} k={k}: GetBoundaryTrueDofs count mismatch"
            );
            // Dof-set sanity: every id in range, strictly sorted (no dups).
            for &d in &dofs {
                assert!(
                    (d as usize) < space.n_dofs(),
                    "{rel} k={k}: dof {d} out of range"
                );
            }
            assert!(
                dofs.windows(2).all(|w| w[0] < w[1]),
                "{rel} k={k}: essential list not strictly sorted"
            );
        }
    }
}

/// RT0 must keep its historical behaviour: exactly one dof per boundary face
/// (the k=0 columns of the oracle table above).
#[test]
fn rt0_boundary_dofs_equal_boundary_face_count() {
    for &(rel, rows) in ORACLE {
        let (k, vsize, ess) = rows[0];
        assert_eq!(k, 0);
        let mesh = load_refined(rel);
        let space = HDivSpace::new(mesh.clone(), 0);
        assert_eq!(space.n_dofs(), vsize);
        let dofs = boundary_dofs_hdiv(space.mesh(), &space, &mesh.unique_boundary_tags());
        assert_eq!(dofs.len(), ess, "{rel}: RT0 boundary dofs");
    }
}
