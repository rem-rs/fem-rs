//! D940 — L² (discontinuous) **prism** h-refinement prolongation vs MFEM 4.10.
//!
//! MFEM truth (WSL `$HOME/mfem410_ser`, probe `tmp/d105prol/probe_d105.cpp`,
//! dumps `tests/data/d105/prl2*.txt`): MFEM's own
//! `FiniteElementSpace::RefinementOperator` for `L2_FECollection(p, 3, btype)`
//! on `ref-prism.mesh` (1 wedge → 8 wedge children; the embedding bookkeeping
//! is correct for prisms, as for hexes) applied to the identity — dumped
//! column-by-column.
//!
//! fem-rs side: [`MixedL2Space`] (the D942 resident heterogeneous L² space —
//! the homogeneous [`fem_space::L2Space`] has no prism arm, D981) on both the
//! coarse and the refined mesh, and `build_l2_prolongation_matrix` with the
//! new prism arm: the coarse basis is MFEM's `L2_WedgeElement(p, btype)`
//! ([`fem_element::lagrange::WedgeL2`], triangle × segment tensor, MFEM's
//! `i ≤ j` layer-major slot order) evaluated at the fine DOF's parent-ref
//! position (the mesh's authoritative prism map inverted, permuted from the
//! mesh's extrusion-first frame to MFEM's `(x, y, z)` wedge frame).
//!
//! Both GL (the collection default, p = 1..3) and GLL (p = 1..3; the p = 3
//! GLL dump added by d118, §4-3b) are pinned;
//! the element-major global numbering and the per-element slot order are
//! pinned through the CDOF/FDOF tables before the entrywise comparison, so
//! the entries compare directly by global index.

#[path = "d105_prolong_util.rs"]
mod util;

use fem_mesh::topology::MeshTopology;
use fem_mesh::refine_uniform_3d;
use fem_space::constraints::prolong::{build_l2_prolongation_matrix, L2ProlongationSpace, MixedL2Space};
use fem_space::L2Basis;

use util::{mesh3, parse_op, row_sums_to_one, value_at};

const PRL2_GL_P1: &str = include_str!("data/d105/prl2_1.txt");
const PRL2_GL_P2: &str = include_str!("data/d105/prl2_2.txt");
const PRL2_GL_P3: &str = include_str!("data/d105/prl2_3.txt");
const PRL2_GLL_P1: &str = include_str!("data/d105/prl2gll_1.txt");
const PRL2_GLL_P2: &str = include_str!("data/d105/prl2gll_2.txt");
// d118 (§4-3b verification depth): MFEM accepts any btype at any p for the
// prism arm (`L2_WedgeElement(p, btype)` = triangle⊗segment, fe_l2.cpp:839;
// `L2_FECollection(p, 3, btype)`, fe_coll.cpp:2340) — round-105 had pinned
// GLL p = 1..2 only; p = 3 closes the GLL column of the supported range.
const PRL2_GLL_P3: &str = include_str!("data/d105/prl2gll_3.txt");

fn prism_l2_prolongation_matches(dumps: &[(u8, &str)], basis: L2Basis) {
    for (p, dump_text) in dumps {
        let p = *p;
        let coarse = mesh3("data/d103/ref-prism.mesh");
        let fine = refine_uniform_3d(&coarse);
        assert_eq!(fine.n_elements(), 8, "p={p}: 8 prism children");
        assert_eq!(fine.element_type(0), fem_mesh::ElementType::Prism6);

        let d = parse_op(dump_text);
        let c = MixedL2Space::new_with_basis(coarse.clone(), p, basis);
        let f = MixedL2Space::new_with_basis(fine.clone(), p, basis);
        assert_eq!(c.n_dofs(), d.csize, "p={p}: coarse vsize vs MFEM CSIZE");
        assert_eq!(f.n_dofs(), d.fsize, "p={p}: fine vsize vs MFEM FSIZE");

        // Element-major consecutive numbering with MFEM's per-element slot
        // order (WedgeL2 is MFEM's constructor verbatim) — this is what turns
        // the entrywise comparison into a full-matrix comparison.
        for (e, want) in d.cdof.iter().enumerate() {
            let got: Vec<u32> = c.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: CDOF element {e}");
        }
        for (e, want) in d.fdof.iter().enumerate() {
            let got: Vec<u32> = f.element_dofs(e as u32).iter().map(|&v| v as u32).collect();
            assert_eq!(got, *want, "p={p}: FDOF element {e}");
        }

        let pmat = build_l2_prolongation_matrix(&c, &f);
        row_sums_to_one(&pmat, &format!("prism {basis:?} p={p}"));

        // global numbering coincides (pinned above), so entries compare directly
        let mut worst = 0.0_f64;
        let mut missing = 0usize;
        for &(i, j, v) in &d.entries {
            let got = value_at(&pmat, i, j);
            if got == 0.0 && v != 0.0 {
                missing += 1;
            }
            worst = worst.max((got - v).abs());
        }
        println!(
            "L2 prism {basis:?} p={p}: {} entries, worst |Δ| = {worst:.3e}, missing {missing}",
            d.entries.len()
        );
        assert_eq!(missing, 0, "p={p}: entries MFEM has that fem-rs lacks");
        assert!(worst < 5e-12, "p={p}: worst |Δ| = {worst:.3e}");
    }
}

#[test]
fn l2_prism_prolongation_matches_mfem_operator_gauss_legendre() {
    prism_l2_prolongation_matches(
        &[(1, PRL2_GL_P1), (2, PRL2_GL_P2), (3, PRL2_GL_P3)],
        L2Basis::GaussLegendre,
    );
}

#[test]
fn l2_prism_prolongation_matches_mfem_operator_gauss_lobatto() {
    prism_l2_prolongation_matches(
        &[(1, PRL2_GLL_P1), (2, PRL2_GLL_P2), (3, PRL2_GLL_P3)],
        L2Basis::GaussLobatto,
    );
}
