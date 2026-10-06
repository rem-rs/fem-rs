//! Embedded (restricted) vector finite element families — MFEM's
//! `ND_R1D` / `ND_R2D` / `RT_R1D` / `RT_R2D` collections
//! (`fem/fe/fe_nd.cpp`, `fem/fe/fe_rt.cpp`, MFEM 4.10).
//!
//! These are the "restricted H(curl)/H(div)" elements of MFEM `ex31` /
//! `ex32p`: a mesh of intrinsic dimension `d ∈ {1, 2}` carrying a vector
//! field with **more components than the mesh dimension** (a 2-D mesh with a
//! 3-component field, or a 1-D mesh with a 3-component field).  The field is
//! NOT a tangential trace of a 3-D field: the in-plane components live on a
//! Nédélec / Raviart–Thomas space of the mesh, the out-of-plane components on
//! a continuous H¹ (ND) or discontinuous L² (RT) space of the mesh, glued
//! into one element-local basis (`dof_map` + `dof2tk`/`dof2nk` tables).
//!
//! Layout conventions (1:1 with MFEM):
//!
//! * reference domains: segment `[0,1]`, triangle `(0,0),(1,0),(0,1)`,
//!   square `[0,1]²` — the fem-rs standard;
//! * element dof numbering (`dof_map`, `Nodes`) is copied from the MFEM
//!   constructors verbatim, including the *negative* `dof_map` entries
//!   (encoding `-1-idx`, sign −1 applied to the tensor basis);
//! * [`EmbeddedSlot`] mirrors each element-local slot's entity association,
//!   which [`fem_space`] consumes to build the collection's global dof
//!   tables (`ND_dof[POINT/SEGMENT/...]` + `DofOrderForOrientation`).
//!
//! The collection-level semantics (dof counts per entity, orientation
//! orderings) live in the space layer (`fem_space::embedded_r2d`); these
//! types are the per-cell local engines.
//!
//! ## Trace collections (`ND_R2D_Trace_FECollection` / `RT_R2D_Trace_FECollection`)
//!
//! MFEM 4.10 wires both as pure constructor mappings onto the same families
//! at `dim − 1` (`fe_coll.cpp`), so no new local element is required:
//!
//! * `ND_R2D_Trace_FECollection(p, dim) : ND_R2D_FECollection(p, dim-1)` —
//!   note MFEM's `GetTraceCollection()` passes the *parent's per-edge dof
//!   count* `2p−1` as the trace order (a quirk of the upstream mapping: the
//!   trace of the order-`p` ND_R2D space is materialized as the order-`2p−1`
//!   `ND_R2D` collection of `dim − 1`);
//! * `RT_R2D_Trace_FECollection(p, dim) : RT_R2D_FECollection(p, dim-1,
//!   INTEGRAL, signs)` — the trace face element is the plain INTEGRAL-map
//!   `L2_SegmentElement(p, ob_type)`, i.e. the parent's edge block.
//!
//! Neither trace collection has a serial-example consumer, so fem-rs models
//! them at the mapping level (this note + the `dim == 1` element arms); a
//! space-layer wrapper can be added on the same pattern as
//! `fem_space::embedded_r2d` when a consumer appears.
//!
//! ## `GetTraceCollection()` upstream truths (D1260/D1261, recorded 2026-10-06)
//!
//! Probed against MFEM 4.10 (`fe_coll.cpp`) and pinned as recorded truth in
//! `fem-space`'s `d117b_rcoll_collection_truth.rs` (+ `d117b_rcoll_ref.txt`):
//!
//! * `ND_R2D_FECollection::GetTraceCollection()` **aborts on default-named
//!   collections for every `dim`** — it tests `nd_name[5]=='_'` but
//!   `"ND_R2D_…"[5]=='D'`, so it falls into `BasisType::GetType('_')`
//!   (`fe_coll.cpp:3325-3343`). Upstream defect (D1260): recorded, protected
//!   against local "fixes" — fem-rs pins the abort, it does not reproduce it.
//! * `RT_R1D_FECollection::GetTraceCollection()` is a plain `MFEM_ABORT`
//!   (`fe_coll.cpp:3217`); and every `dim == 1` R2D variant aborts building
//!   its dim-1 trace (`MFEM_VERIFY(dim==2)` in `RT_R2D_Trace_FECollection`
//!   `fe_coll.cpp:3532`, ND `dim>=1` `fe_coll.cpp:3239`) (D1261).
//! * `ND_R1D_FECollection::GetTraceCollection()` returns NULL; the only
//!   default-named variant that resolves a trace collection is `RT_R2D(p, 2)`
//!   → `RT_R2D_Trace_2D_Pp`.
//!
//! Debt D1262 records the fem-rs stance for the two trace collections
//! themselves: mapping-level only (this section), no space-layer wrapper
//! until a consumer appears (extends D900/D901/D914).

mod nd_r2d;
mod r1d;
mod rt_r2d;

pub use nd_r2d::{NdR2dQuad, NdR2dTri};
pub use r1d::{NdR1dPoint, NdR1dSegment, NdR2dSegment, RtR1dSegment};
pub use rt_r2d::{RtR2dQuad, RtR2dTri};

/// Entity association of one element-local dof slot.
///
/// `Edge(e, k)` / `EdgeScalar(e, k)`: the `k`-th slot (0-based, in slot
/// order) of local edge `e`'s shared block.  The space layer maps `k`
/// through the collection's `DofOrderForOrientation` table
/// (`SegDofOrd[orientation][k]`) onto the global edge dofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedSlot {
    /// Vertex-associated scalar dof (ND_R2D: the z-directed H1 dof;
    /// ND_R1D: the y/z dofs — `c` distinguishes them) at local vertex `v`.
    Vertex(usize, usize),
    /// In-plane tangential (ND) / normal (RT) dof on local edge `e`,
    /// block position `k`.  Carries an orientation sign.
    Edge(usize, usize),
    /// Scalar H1-type dof on local edge `e`, block position `k`
    /// (ND_R2D: z-directed edge dofs).  Orientation-free (site order only).
    EdgeScalar(usize, usize),
    /// Element-interior in-plane dof (element-local, never shared).
    Interior,
    /// Element-interior scalar dof (element-local, never shared).
    InteriorScalar,
}

/// 2×2 affine element Jacobian (intrinsic 2-D mesh element), MFEM
/// `ElementTransformation::Jacobian()` + `Weight()`.
#[derive(Clone, Copy, Debug)]
pub struct Jac2D {
    pub j00: f64,
    pub j01: f64,
    pub j10: f64,
    pub j11: f64,
    /// MFEM `Trans.Weight()` (the signed determinant for affine elements).
    pub det: f64,
}

impl Jac2D {
    /// `[J⁻¹(0,0), J⁻¹(0,1), J⁻¹(1,0), J⁻¹(1,1)]` (MFEM `InverseJacobian()`).
    pub fn inv(&self) -> [f64; 4] {
        let inv_det = 1.0 / self.det;
        [
            self.j11 * inv_det,
            -self.j01 * inv_det,
            -self.j10 * inv_det,
            self.j00 * inv_det,
        ]
    }
}

/// Lagrange basis on an arbitrary 1-D node set at `x`:
/// returns `(values[j], derivs[j])` for `j = 0..nodes.len()`.
pub(crate) fn lagrange_1d(nodes: &[f64], x: f64) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    let mut vals = vec![0.0_f64; n];
    let mut ders = vec![0.0_f64; n];
    for j in 0..n {
        let (mut v, mut d) = (1.0_f64, 0.0_f64);
        for (i, &xi) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }
            let den = nodes[j] - xi;
            let f = (x - xi) / den;
            d = d * f + v / den;
            v *= f;
        }
        vals[j] = v;
        ders[j] = d;
    }
    (vals, ders)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_partition_of_unity_and_delta() {
        let nodes = [0.0, 0.25, 0.6, 1.0];
        let (v, _d) = lagrange_1d(&nodes, 0.3);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-14);
        for (j, &xj) in nodes.iter().enumerate() {
            let (v, _d) = lagrange_1d(&nodes, xj);
            for (i, &vi) in v.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((vi - expect).abs() < 1e-13);
            }
        }
    }
}

#[cfg(test)]
mod d102_delta_tests {
    //! Slot-matrix identity: applying every dof functional to every basis
    //! function at the element's own `Nodes` must give the identity —
    //! the structural 1:1 signature of the MFEM `dof_map`/`dof2tk` tables.
    use super::*;

    fn nd_tri_case(p: usize) {
        let el = NdR2dTri::new(p);
        let n = el.n_dofs();
        let nd_n = p * (p + 2);
        assert_eq!(n, ((3 * p + 1) * (p + 2)) / 2);
        let mut phi = vec![0.0_f64; n * 3];
        for k in 0..n {
            let x = el.nodes()[k];
            el.eval_vshape_ref(&x, &mut phi);
            let tk = el.dof2tk()[k];
            for j in 0..n {
                let expect = if k == j { 1.0 } else { 0.0 };
                let (v, t) = ([phi[j * 3], phi[j * 3 + 1]], [phi[j * 3 + 2]]);
                let m = if tk < 4 {
                    const TK: [[f64; 2]; 4] = [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [0.0, 1.0]];
                    v[0] * TK[tk as usize][0] + v[1] * TK[tk as usize][1]
                } else {
                    t[0]
                };
                assert!(
                    (m - expect).abs() < 1e-11,
                    "NdR2dTri p={p}: M[{k}][{j}] = {m}"
                );
            }
        }
        let _ = nd_n;
    }

    fn rt_tri_case(p: usize) {
        let el = RtR2dTri::new(p);
        let n = el.n_dofs();
        assert_eq!(n, ((p + 1) * (3 * p + 8)) / 2);
        let mut phi = vec![0.0_f64; n * 3];
        for k in 0..n {
            el.eval_vshape_ref(&el.nodes()[k], &mut phi);
            let nk = el.dof2nk()[k];
            for j in 0..n {
                let expect = if k == j { 1.0 } else { 0.0 };
                let m = if nk < 3 {
                    // RT triangle normals (nk_t): (0,-1), (1,1), (-1,0)
                    const NK: [[f64; 2]; 3] = [[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]];
                    phi[j * 3] * NK[nk as usize][0] + phi[j * 3 + 1] * NK[nk as usize][1]
                } else {
                    phi[j * 3 + 2]
                };
                assert!(
                    (m - expect).abs() < 1e-11,
                    "RtR2dTri p={p}: M[{k}][{j}] = {m}"
                );
            }
        }
    }

    fn nd_quad_case(p: usize) {
        let el = NdR2dQuad::new(p);
        let n = el.n_dofs();
        assert_eq!(n, (3 * p + 1) * (p + 1));
        let mut phi = vec![0.0_f64; n * 3];
        for k in 0..n {
            el.eval_vshape_ref(&el.nodes()[k], &mut phi);
            let tk = el.dof2tk()[k];
            for j in 0..n {
                let expect = if k == j { 1.0 } else { 0.0 };
                let m = if tk < 4 {
                    const TK: [[f64; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
                    phi[j * 3] * TK[tk as usize][0] + phi[j * 3 + 1] * TK[tk as usize][1]
                } else {
                    phi[j * 3 + 2]
                };
                assert!(
                    (m - expect).abs() < 1e-11,
                    "NdR2dQuad p={p}: M[{k}][{j}] = {m}"
                );
            }
        }
    }

    fn rt_quad_case(p: usize) {
        let el = RtR2dQuad::new(p);
        let n = el.n_dofs();
        assert_eq!(n, (3 * p + 5) * (p + 1));
        let mut phi = vec![0.0_f64; n * 3];
        for k in 0..n {
            el.eval_vshape_ref(&el.nodes()[k], &mut phi);
            let nk = el.dof2nk()[k];
            for j in 0..n {
                let expect = if k == j { 1.0 } else { 0.0 };
                let m = if nk < 4 {
                    const NK: [[f64; 2]; 4] =
                        [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
                    phi[j * 3] * NK[nk as usize][0] + phi[j * 3 + 1] * NK[nk as usize][1]
                } else {
                    phi[j * 3 + 2]
                };
                assert!(
                    (m - expect).abs() < 1e-11,
                    "RtR2dQuad p={p}: M[{k}][{j}] = {m}"
                );
            }
        }
    }

    #[test]
    fn d102_nd_r2d_tri_slot_matrix_is_identity() {
        for p in 1..=3 {
            nd_tri_case(p);
        }
    }

    #[test]
    fn d102_rt_r2d_tri_slot_matrix_is_identity() {
        for p in 0..=3 {
            rt_tri_case(p);
        }
    }

    #[test]
    fn d102_nd_r2d_quad_slot_matrix_is_identity() {
        for p in 1..=3 {
            nd_quad_case(p);
        }
    }

    #[test]
    fn d102_rt_r2d_quad_slot_matrix_is_identity() {
        for p in 0..=3 {
            rt_quad_case(p);
        }
    }

    #[test]
    fn d102_segment_slot_matrices_are_identity() {
        use crate::embedded::{NdR1dSegment, NdR2dSegment, RtR1dSegment};
        for p in 1..=3usize {
            // ND_R2D segment (vdim 2): x slots tk=0 tangent (1,0), z slots scalar.
            let el = NdR2dSegment::new(p);
            let n = el.n_dofs();
            assert_eq!(n, 2 * p + 1);
            let mut phi = vec![0.0_f64; n * 2];
            for k in 0..n {
                el.eval_vshape_ref(el.nodes()[k], &mut phi);
                for j in 0..n {
                    let expect = if k == j { 1.0 } else { 0.0 };
                    let m = if el.dof2tk()[k] == 0 {
                        phi[j * 2]
                    } else {
                        phi[j * 2 + 1]
                    };
                    assert!((m - expect).abs() < 1e-11);
                }
            }
            // ND_R1D segment (vdim 3): tk 0/1/2.
            let el = NdR1dSegment::new(p);
            let n = el.n_dofs();
            assert_eq!(n, 3 * p + 2);
            let mut phi = vec![0.0_f64; n * 3];
            for k in 0..n {
                el.eval_vshape_ref(el.nodes()[k], &mut phi);
                for j in 0..n {
                    let expect = if k == j { 1.0 } else { 0.0 };
                    let m = phi[j * 3 + el.dof2tk()[k] as usize];
                    assert!((m - expect).abs() < 1e-11);
                }
            }
            // RT_R1D segment (vdim 3): nk 0/1/2.
            let el = RtR1dSegment::new(p);
            let n = el.n_dofs();
            assert_eq!(n, 3 * p + 4);
            let mut phi = vec![0.0_f64; n * 3];
            for k in 0..n {
                el.eval_vshape_ref(el.nodes()[k], &mut phi);
                for j in 0..n {
                    let expect = if k == j { 1.0 } else { 0.0 };
                    let m = phi[j * 3 + el.dof2nk()[k] as usize];
                    assert!((m - expect).abs() < 1e-11);
                }
            }
        }
        // Note: `RT_R2D_SegmentElement` is not ported — it is dead code
        // upstream (no collection wires it; see r1d.rs).
    }
}
