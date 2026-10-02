//! D942 — the resident **heterogeneous (mixed-geometry) L² space**
//! ([`MixedL2Space`]) and its prolongation, end to end against MFEM 4.10.
//!
//! The homogeneous [`fem_space::L2Space`] holds one `dofs_per_elem` for the
//! whole mesh, so it cannot represent an L² field on a mesh whose cells mix
//! geometries — which is every refined pyramid mesh (6 pyramid + 4 tet
//! children per parent, D472) and any genuinely mixed coarse mesh.  d103
//! verified the pyramid L² prolongation through a *test-local* stand-in; D942
//! promotes that stand-in into a resident space with MFEM's per-geometry
//! element dispatch and element-major consecutive numbering, and this file
//! verifies it on two cross-geometry grids, in both bases:
//!
//! * **pyramid-refined mesh** (`pyrl2sem*.txt`): inline-pyramid (48 pyramid
//!   parents) → 480 children (288 pyramids + 192 tets).  Semantic truth via
//!   the `pyr_children` refinement point matrices (MFEM's own pyramid
//!   `RefinementOperator` embedding is broken, mesh.cpp:11062 — D944);
//!   GaussLegendre p = 1..2 re-pins the d103 result through the resident
//!   space, GaussLobatto p = 1..2 is new coverage (Fuentes pyramid GLL, whose
//!   z layers MFEM forces open, × tet GLL on the same mixed mesh).
//! * **hex + prism mixed mesh** (`mixhp*.txt`): 1 hexahedron + 1 prism
//!   sharing a quad face → 8 hex + 8 prism children.  MFEM's own
//!   `RefinementOperator` for `L2_FECollection(p, 3, btype)` on the shared
//!   mesh file, dumped and compared entrywise — the direct proof that the
//!   mixed space + builder handle two different 3-D geometries in one space.

#[path = "d105_prolong_util.rs"]
mod util;

use fem_mesh::topology::MeshTopology;
use fem_mesh::refine_uniform_3d;
use fem_space::constraints::prolong::{build_l2_prolongation_matrix, L2ProlongationSpace, MixedL2Space};
use fem_space::L2Basis;

use util::{
    compare_op_coord_keyed, compare_sem_coord_keyed, mesh3, parse_op, parse_sem, row_sums_to_one,
};

const PYRL2SEM_GL_P1: &str = include_str!("data/d105/pyrl2sem_1.txt");
const PYRL2SEM_GL_P2: &str = include_str!("data/d105/pyrl2sem_2.txt");
const PYRL2SEM_GLL_P1: &str = include_str!("data/d105/pyrl2semgll_1.txt");
const PYRL2SEM_GLL_P2: &str = include_str!("data/d105/pyrl2semgll_2.txt");
const MIXHP_GL_P1: &str = include_str!("data/d105/mixhp_1.txt");
const MIXHP_GL_P2: &str = include_str!("data/d105/mixhp_2.txt");
const MIXHP_GLL_P1: &str = include_str!("data/d105/mixhpgll_1.txt");

/// DOF-count pin: the mixed space holds MFEM's per-geometry counts on the
/// pyramid-refined grid — 288·(p+1)³ + 192·(p+1)(p+2)(p+3)/6 (d340 pinned the
/// same number for MFEM's own space on this mesh; the homogeneous L2Space
/// cannot even be constructed here — that refusal is fem-rs D981).
#[test]
fn mixed_l2_space_dof_counts_on_refined_pyramid_mesh() {
    let coarse = mesh3("data/d103/inline-pyramid.mesh");
    let fine = refine_uniform_3d(&coarse);
    assert_eq!(coarse.n_elements(), 48);
    assert_eq!(fine.n_elements(), 480);
    let mut n_pyr = 0usize;
    let mut n_tet = 0usize;
    for e in 0..fine.n_elements() as u32 {
        match fine.element_type(e) {
            fem_mesh::ElementType::Pyramid5 => n_pyr += 1,
            fem_mesh::ElementType::Tet4 => n_tet += 1,
            other => panic!("unexpected child geometry {other:?}"),
        }
    }
    assert_eq!((n_pyr, n_tet), (288, 192), "6 pyramids + 4 tets per parent");
    for p in 1..=2u8 {
        let pp = p as usize;
        let want =
            n_pyr * (pp + 1).pow(3) + n_tet * (pp + 1) * (pp + 2) * (pp + 3) / 6;
        let f = MixedL2Space::new(fine.clone(), p);
        assert_eq!(f.n_dofs(), want, "p={p}: mixed-space fine vsize");
    }
}

/// The pyramid-refined mixed mesh in both bases, against the semantic truth
/// rows (GL re-pins the d103 result through the *resident* space; GLL is the
/// new second-basis coverage).
#[test]
fn mixed_l2_prolongation_on_refined_pyramid_mesh_matches_mfem() {
    let coarse_mesh = mesh3("data/d103/inline-pyramid.mesh");
    let fine_mesh = refine_uniform_3d(&coarse_mesh);
    for (basis, dumps) in [
        (L2Basis::GaussLegendre, &[(1u8, PYRL2SEM_GL_P1), (2, PYRL2SEM_GL_P2)][..]),
        (L2Basis::GaussLobatto, &[(1, PYRL2SEM_GLL_P1), (2, PYRL2SEM_GLL_P2)][..]),
    ] {
        for (p, dump_text) in dumps {
            let p = *p;
            let d = parse_sem(dump_text);
            let c = MixedL2Space::new_with_basis(coarse_mesh.clone(), p, basis);
            let f = MixedL2Space::new_with_basis(fine_mesh.clone(), p, basis);
            assert_eq!(c.n_dofs(), d.csize, "{basis:?} p={p}: coarse vsize vs MFEM CSIZE");
            assert_eq!(f.n_dofs(), d.fsize, "{basis:?} p={p}: fine vsize vs MFEM FSIZE");

            let pmat = build_l2_prolongation_matrix(&c, &f);
            row_sums_to_one(&pmat, &format!("pyr-mixed {basis:?} p={p}"));
            let (_, worst, _) = compare_sem_coord_keyed(
                &d,
                &pmat,
                &f,
                &c,
                &format!("pyr-mixed {basis:?} p={p}"),
            );
            assert!(worst < 5e-12, "{basis:?} p={p}: worst |Δ| = {worst:.3e}");
        }
    }
}

/// The hex + prism mixed mesh: direct `RefinementOperator` parity for the
/// resident mixed space, GL p = 1..2 and GLL p = 1.
#[test]
fn mixed_l2_prolongation_on_hex_prism_mesh_matches_mfem_operator() {
    for (p, basis, dump_text) in [
        (1u8, L2Basis::GaussLegendre, MIXHP_GL_P1),
        (2, L2Basis::GaussLegendre, MIXHP_GL_P2),
        (1, L2Basis::GaussLobatto, MIXHP_GLL_P1),
    ] {
        let coarse = mesh3("data/d105/mix-hex-prism.mesh");
        assert_eq!(coarse.element_type(0), fem_mesh::ElementType::Hex8);
        assert_eq!(coarse.element_type(1), fem_mesh::ElementType::Prism6);
        let fine = refine_uniform_3d(&coarse);
        assert_eq!(fine.n_elements(), 16, "8 hex + 8 prism children");

        let d = parse_op(dump_text);
        let c = MixedL2Space::new_with_basis(coarse.clone(), p, basis);
        let f = MixedL2Space::new_with_basis(fine.clone(), p, basis);
        // MFEM CSIZE = (p+1)³ + (p+1)²(p+2)/2 — the two geometries' counts in
        // one space, the number the homogeneous L2Space cannot hold.
        let pp = p as usize;
        assert_eq!(
            c.n_dofs(),
            (pp + 1).pow(3) + (pp + 1) * (pp + 1) * (pp + 2) / 2,
            "{basis:?} p={p}: mixed coarse vsize"
        );
        assert_eq!(c.n_dofs(), d.csize, "{basis:?} p={p}: coarse vsize vs MFEM CSIZE");
        assert_eq!(f.n_dofs(), d.fsize, "{basis:?} p={p}: fine vsize vs MFEM FSIZE");

        let pmat = build_l2_prolongation_matrix(&c, &f);
        row_sums_to_one(&pmat, &format!("hex+prism {basis:?} p={p}"));
        let (_, worst, _) = compare_op_coord_keyed(
            &d,
            &pmat,
            &f,
            &c,
            &format!("hex+prism {basis:?} p={p}"),
        );
        assert!(worst < 5e-12, "{basis:?} p={p}: worst |Δ| = {worst:.3e}");
    }
}

/// P0 on the mixed mesh: the indicator degeneration is geometry-independent
/// (one DOF per element, children inherit the parent's value).
#[test]
fn mixed_l2_p0_prolongation_is_indicator_on_mixed_mesh() {
    let coarse = mesh3("data/d105/mix-hex-prism.mesh");
    let fine = refine_uniform_3d(&coarse);
    let c = MixedL2Space::new(coarse.clone(), 0);
    let f = MixedL2Space::new(fine.clone(), 0);
    assert_eq!(c.n_dofs(), coarse.n_elements());
    assert_eq!(f.n_dofs(), fine.n_elements());
    let pmat = build_l2_prolongation_matrix(&c, &f);
    for row in 0..pmat.nrows {
        let nnz = pmat.row_ptr[row + 1] - pmat.row_ptr[row];
        assert_eq!(nnz, 1, "row {row}");
        assert_eq!(pmat.values[pmat.row_ptr[row]], 1.0, "row {row}");
    }
}

/// The boundary of the resident space stays explicit: the *homogeneous*
/// [`fem_space::L2Space`] still refuses a prism mesh — its prism arm and
/// mixed-geometry support live in `crates/space/src/l2.rs`, outside this
/// lane's territory, registered as D981.  (On a *hex-first* mixed mesh
/// L2Space does not even refuse: it silently builds an all-hex space — the
/// dispatch keys on element 0's node count only; also recorded in D981.)
#[test]
#[should_panic(expected = "L2Space currently supports")]
fn homogeneous_l2_space_still_refuses_prism_mesh() {
    let coarse = mesh3("data/d103/ref-prism.mesh");
    let _ = fem_space::L2Space::new(coarse, 1);
}
