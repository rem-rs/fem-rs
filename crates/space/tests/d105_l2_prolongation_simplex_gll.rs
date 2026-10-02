//! D941 — GaussLobatto simplex (tri / tet) L² h-refinement prolongation vs
//! MFEM 4.10.
//!
//! MFEM semantics (source-pinned, fe_l2.cpp:570/695 + fe_base.hpp `OpenPoints`
//! + intrules.cpp `CheckOpen`): `L2_FECollection(p, dim, BasisType::GaussLobatto)`
//! puts `L2_TriangleElement`/`L2_TetrahedronElement(p, GaussLobatto)` on
//! simplices — nodes at the barycentric warp of the **closed** GLL points
//! (`OpenPoints` forwards `GetPoints`; "all types can work as open"), dual
//! (nodal) basis.  That coincides with the equispaced nodal lattice only for
//! `p <= 2` — at `p >= 3` the GLL warp points (0.276393…, 0.723607…) differ
//! from 1/3, 2/3, so the equispaced layouts fem-rs's homogeneous
//! [`fem_space::L2Space`] used for its GLL simplex arm (hand-coded per order,
//! registered as D982) diverge from MFEM from `p = 3` on.  The prolongation
//! arm now evaluates the true MFEM elements through the new GLL reference
//! elements `TriL2GL::new_gauss_lobatto` / `TetL2GL::new_gauss_lobatto`.
//!
//! Ground truth (probe `tmp/d105prol/probe_d105.cpp`, dumps
//! `tests/data/d105/`):
//!
//! * **tri** (`trl2*.txt`): MFEM's own `RefinementOperator` for
//!   `L2_FECollection(p, 2, btype)` on the shared 2×2-triangle fixture
//!   (`tri22.mesh`, read by both libraries) applied to the identity — direct
//!   parity, GL as the control and GLL as the D941 subject.
//! * **tet** (`tetsem*.txt`): *semantic* rows — fem-rs's tet uniform
//!   refinement diverges from MFEM 4.10's (D943, mesh layer, out of this
//!   lane's territory), so MFEM's refinement operator cannot be constructed
//!   for the pair.  The probe loads fem-rs's own refined mesh
//!   (`refined-cart1-tet.mesh`, the exact fine mesh the truth was computed
//!   on), finds each fine element's parent by centroid containment, builds
//!   the fine-ref → parent-ref map from the parent-ref images of the child's
//!   four corners (Newton inversion of MFEM's own element transformation) and
//!   evaluates `I = fine_fe.GetTransferMatrix(coarse_fe, isotr)` — exactly
//!   MFEM's `NodalLocalInterpolation` (fe_base.cpp:526) with MFEM's own FEs.

#[path = "d105_prolong_util.rs"]
mod util;

use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_space::constraints::prolong::{build_l2_prolongation_matrix, L2ProlongationSpace, MixedL2Space};
use fem_space::L2Basis;

use util::{
    compare_op_coord_keyed, compare_sem_coord_keyed, mesh2, mesh3, parse_op, parse_sem,
    row_sums_to_one,
};

const TRL2_GL_P1: &str = include_str!("data/d105/trl2_1.txt");
const TRL2_GL_P2: &str = include_str!("data/d105/trl2_2.txt");
const TRL2_GL_P3: &str = include_str!("data/d105/trl2_3.txt");
const TRL2_GLL_P1: &str = include_str!("data/d105/trl2gll_1.txt");
const TRL2_GLL_P2: &str = include_str!("data/d105/trl2gll_2.txt");
const TRL2_GLL_P3: &str = include_str!("data/d105/trl2gll_3.txt");
const TETSEM_GL_P1: &str = include_str!("data/d105/tetsem_1.txt");
const TETSEM_GL_P2: &str = include_str!("data/d105/tetsem_2.txt");
const TETSEM_GL_P3: &str = include_str!("data/d105/tetsem_3.txt");
const TETSEM_GLL_P1: &str = include_str!("data/d105/tetsemgll_1.txt");
const TETSEM_GLL_P2: &str = include_str!("data/d105/tetsemgll_2.txt");
const TETSEM_GLL_P3: &str = include_str!("data/d105/tetsemgll_3.txt");

/// 2-D tri h-refinement prolongation vs MFEM's own `RefinementOperator`
/// (direct dump on the shared coarse mesh file).
fn tri_l2_prolongation_matches(dumps: &[(u8, &str)], basis: L2Basis, tol: f64) {
    for (p, dump_text) in dumps {
        let p = *p;
        let coarse = mesh2("data/d105/tri22.mesh");
        let fine = refine_uniform(&coarse);
        assert_eq!(fine.n_elements(), 4 * coarse.n_elements(), "p={p}: 4-way tri refinement");

        let d = parse_op(dump_text);
        let c = MixedL2Space::new_with_basis(coarse.clone(), p, basis);
        let f = MixedL2Space::new_with_basis(fine.clone(), p, basis);
        assert_eq!(c.n_dofs(), d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        assert_eq!(f.n_dofs(), d.fsize, "p={p}: fine vsize vs MFEM FSIZE");

        let pmat = build_l2_prolongation_matrix(&c, &f);
        row_sums_to_one(&pmat, &format!("tri {basis:?} p={p}"));
        let (_, worst, _) = compare_op_coord_keyed(&d, &pmat, &f, &c, &format!("tri {basis:?} p={p}"));
        assert!(worst < tol, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

#[test]
fn l2_tri_gauss_legendre_prolongation_matches_mfem_operator() {
    tri_l2_prolongation_matches(
        &[(1, TRL2_GL_P1), (2, TRL2_GL_P2), (3, TRL2_GL_P3)],
        L2Basis::GaussLegendre,
        5e-12,
    );
}

/// The D941 subject: GLL warp-point simplex L2, orders 1..3 (p >= 3 is where
/// the GLL warp leaves the equispaced lattice).
#[test]
fn l2_tri_gauss_lobatto_prolongation_matches_mfem_operator() {
    tri_l2_prolongation_matches(
        &[(1, TRL2_GLL_P1), (2, TRL2_GLL_P2), (3, TRL2_GLL_P3)],
        L2Basis::GaussLobatto,
        5e-12,
    );
}

/// 3-D tet h-refinement prolongation against the semantic rows computed on
/// fem-rs's own refined mesh (D943 detour, see the module docs).
fn tet_l2_prolongation_matches(dumps: &[(u8, &str)], basis: L2Basis, tol: f64) {
    for (p, dump_text) in dumps {
        let p = *p;
        let coarse = mesh3("data/d103/cart1-tet.mesh");
        // The exact fine mesh the semantic truth was computed on (fem-rs's own
        // uniform refinement, dumped to file before the probe ran).
        let fine: Mesh<3> = mesh3("data/d105/refined-cart1-tet.mesh");
        assert_eq!(fine.n_elements(), 8 * coarse.n_elements(), "p={p}: 8-way tet refinement");

        let d = parse_sem(dump_text);
        let c = MixedL2Space::new_with_basis(coarse.clone(), p, basis);
        let f = MixedL2Space::new_with_basis(fine.clone(), p, basis);
        assert_eq!(c.n_dofs(), d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        assert_eq!(f.n_dofs(), d.fsize, "p={p}: fine vsize vs MFEM FSIZE");

        let pmat = build_l2_prolongation_matrix(&c, &f);
        row_sums_to_one(&pmat, &format!("tet {basis:?} p={p}"));
        let (_, worst, _) = compare_sem_coord_keyed(&d, &pmat, &f, &c, &format!("tet {basis:?} p={p}"));
        assert!(worst < tol, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

/// GL control: the pre-existing tet arm stays pinned through the new semantic
/// route (indirect MFEM truth instead of the D943-blocked operator dump).
#[test]
fn l2_tet_gauss_legendre_prolongation_matches_mfem_semantics() {
    tet_l2_prolongation_matches(
        &[(1, TETSEM_GL_P1), (2, TETSEM_GL_P2), (3, TETSEM_GL_P3)],
        L2Basis::GaussLegendre,
        5e-12,
    );
}

#[test]
fn l2_tet_gauss_lobatto_prolongation_matches_mfem_semantics() {
    tet_l2_prolongation_matches(
        &[(1, TETSEM_GLL_P1), (2, TETSEM_GLL_P2), (3, TETSEM_GLL_P3)],
        L2Basis::GaussLobatto,
        5e-12,
    );
}
