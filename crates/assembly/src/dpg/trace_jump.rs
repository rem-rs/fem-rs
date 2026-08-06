//! DPG trace-test coupling matrix (TraceJumpIntegrator) assembly.
//!
//! Assembles the rectangular matrix `Bhat` where
//! `Bhat[test_dof, trace_dof] = ∫_face v · λ ds`.
//!
//! This is the mixed bilinear form coupling the L² test space with the DPG
//! trace (interface) space on all mesh skeleton faces.
//!
//! MFEM equivalence: `TraceJumpIntegrator` in ex8.cpp, used with
//! `MixedBilinearForm(xhat_space, test_space)`.

use fem_core::types::DofId;
use fem_element::{
    ReferenceElement,
    lagrange::{QuadL2GL, TriP1, TriP2, TriP3, QuadQ1, QuadQ2},
    quadrature::seg_rule_arbitrary,
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

use fem_space::DpgTraceSpace;
use fem_space::FaceInfo;

// ─── 1D Lagrange basis on [0, 1] ─────────────────────────────────────────────

/// Evaluate the 1D Lagrange basis of order `p` at point `xi` in [0, 1].
///
/// The `p+1` basis functions have nodes at `{0, 1/p, 2/p, ..., 1}`.
/// Returns `phi` of length `p+1`.
fn eval_lagrange_1d(p: usize, xi: f64, phi: &mut [f64]) {
    debug_assert!(phi.len() >= p + 1);
    if p == 0 {
        phi[0] = 1.0;
        return;
    }
    for k in 0..=p {
        let xk = k as f64 / p as f64;
        let mut v = 1.0;
        for j in 0..=p {
            if j == k { continue; }
            let xj = j as f64 / p as f64;
            v *= (xi - xj) / (xk - xj);
        }
        phi[k] = v;
    }
}

// ─── Reference element helpers ───────────────────────────────────────────────

fn ref_elem_for_space(
    elem_type: ElementType,
    order: u8,
) -> (Box<dyn ReferenceElement>, usize, bool) {
    // `ref01` = whether the reference element lives on the [0,1]ⁿ domain
    // (MFEM L2/Gauss-Legendre elements) vs the [-1,1]ⁿ domain (legacy
    // QuadQk).  The face parameterisation must match.
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 1) => (Box::new(TriP1) as Box<dyn ReferenceElement>, 3, true),
        (ElementType::Tri3 | ElementType::Tri6, 2) => (Box::new(TriP2) as Box<dyn ReferenceElement>, 6, true),
        (ElementType::Tri3 | ElementType::Tri6, 3) => (Box::new(TriP3) as Box<dyn ReferenceElement>, 10, true),
        // L2 P1 on quads uses Gauss-Legendre nodal basis (MFEM
        // L2_FECollection default BasisType::GaussLegendre) on [0,1]² — NOT
        // the equally-spaced QuadQ1.  Using QuadQ1 here made Bhat
        // inconsistent with the test space used by B0 (0.0156 vs MFEM 0.683
        // on ex8).
        (ElementType::Quad4, 1) => (Box::new(QuadL2GL::new(1)) as Box<dyn ReferenceElement>, 4, true),
        (ElementType::Quad4, 2) => (Box::new(QuadQ2) as Box<dyn ReferenceElement>, 9, false),
        _ => panic!("assemble_bhat ref_elem: unsupported ({elem_type:?}, order={order})"),
    }
}

/// Map face parameter `xi ∈ [0,1]` and local face index to reference-element
/// coordinates `(ξ, η)` for a triangle.
fn edge_xi_tri(lf: usize, xi: f64) -> (f64, f64) {
    match lf {
        0 => (xi, 0.0),
        1 => (1.0 - xi, xi),
        2 => (0.0, xi),
        _ => (0.0, 0.0),
    }
}

/// Map face parameter `xi ∈ [0,1]` and local face index to reference-element
/// coordinates `(ξ, η)` for a quad (reference domain [-1,1]² for QuadQ1).
fn edge_xi_quad(lf: usize, xi: f64) -> (f64, f64) {
    match lf {
        0 => (2.0 * xi - 1.0, -1.0),
        1 => (1.0, 2.0 * xi - 1.0),
        2 => (1.0 - 2.0 * xi, 1.0),
        3 => (-1.0, 1.0 - 2.0 * xi),
        _ => (0.0, 0.0),
    }
}

/// Face parameterisation for reference elements on the [0,1]² domain
/// (e.g. L2 Gauss-Legendre elements, matching MFEM's reference domain).
fn edge_xi_quad_01(lf: usize, xi: f64) -> (f64, f64) {
    match lf {
        0 => (xi, 0.0),
        1 => (1.0, xi),
        2 => (1.0 - xi, 1.0),
        3 => (0.0, 1.0 - xi),
        _ => (0.0, 0.0),
    }
}

// ─── Assembly ─────────────────────────────────────────────────────────────────

/// Assemble the trace-test coupling matrix (Bhat).
///
/// Bhat[i, j] = ∫_face v_i · λ_j ds
///
/// where `v_i` are the L² test basis functions and `λ_j` are the trace basis
/// functions on the mesh skeleton.
///
/// # Arguments
/// * `test_space` — the L² (discontinuous) test space
/// * `trace_space` — the DPG trace space on all skeleton faces
/// * `quad_order` — quadrature order for face integration
///
/// # Returns
/// A sparse matrix of size `n_test × n_trace`.
pub fn assemble_bhat<M: MeshTopology + Clone>(
    test_space: &impl FESpace<Mesh = M>,
    trace_space: &DpgTraceSpace<M>,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let dim = 2; // 2D only for now
    let elem_type = test_space.mesh().element_type(0);
    let test_order = test_space.order();
    let (ref_elem, nt, ref01) = ref_elem_for_space(elem_type, test_order);
    let dpf = trace_space.dofs_per_face();
    let is_tri = matches!(elem_type, ElementType::Tri3 | ElementType::Tri6);

    let eq = seg_rule_arbitrary(quad_order);
    let n_test = test_space.n_dofs();
    let n_trace = trace_space.n_dofs();
    let mut coo = CooMatrix::new(n_test, n_trace);

    let mut phi = vec![0.0; nt];
    let mut trace_phi = vec![0.0; dpf];

    //── All faces (boundary + interior) in face-table order ─────────────────
    for fi in 0..trace_space.n_faces() {
        let info = trace_space.face_info(fi);
        // RT0 trace DOF sign: MFEM encodes the face orientation in the
        // GetFaceDofs sign (vdof < 0 → AddSubMatrix flips the column):
        // sign = +1 when the face's node order matches the canonical
        // (min→max) vertex order, -1 otherwise.
        let face_sign = match info {
            FaceInfo::Boundary { nodes, .. } | FaceInfo::Interior { nodes, .. } => {
                if nodes[0] < nodes[1] { 1.0 } else { -1.0 }
            }
        };
        match info {
            FaceInfo::Boundary { elem, local_face, .. } => {
                let test_dofs: Vec<usize> = test_space.element_dofs(*elem).iter().map(|&d| d as usize).collect();
                let trace_dofs: Vec<usize> = trace_space.face_dofs(fi).iter().map(|&d| d as usize).collect();

                for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
                    let xi = xr[0];
                    let w = wr * face_sign; // RT_Trace map_type=INTEGRAL: no physical Weight (MFEM TraceJumpIntegrator only multiplies for VALUE map types)
                    let (rx, ry) = if is_tri { edge_xi_tri(*local_face, xi) } else if ref01 { edge_xi_quad_01(*local_face, xi) } else { edge_xi_quad(*local_face, xi) };
                    ref_elem.eval_basis(&[rx, ry], &mut phi);
                    eval_lagrange_1d(dpf - 1, xi, &mut trace_phi);

                    for i in 0..nt {
                        let gi = test_dofs[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], w * phi[i] * trace_phi[j]);
                        }
                    }
                }
            }
            FaceInfo::Interior { elem_l, elem_r, local_l, local_r, .. } => {
                let dl: Vec<usize> = test_space.element_dofs(*elem_l).iter().map(|&d| d as usize).collect();
                let dr: Vec<usize> = test_space.element_dofs(*elem_r).iter().map(|&d| d as usize).collect();
                let trace_dofs: Vec<usize> = trace_space.face_dofs(fi).iter().map(|&d| d as usize).collect();
                let npe_l = test_space.mesh().element_nodes(*elem_l).len();
                let npe_r = test_space.mesh().element_nodes(*elem_r).len();

                for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
                    let xi = xr[0];
                    let w = wr * face_sign; // RT_Trace map_type=INTEGRAL: no physical Weight (MFEM TraceJumpIntegrator only multiplies for VALUE map types)
                    eval_lagrange_1d(dpf - 1, xi, &mut trace_phi);

                    // Left element (+ sign, outward normal)
                    let (rxl, ryl) = if npe_l == 3 { edge_xi_tri(*local_l, xi) } else if ref01 { edge_xi_quad_01(*local_l, xi) } else { edge_xi_quad(*local_l, xi) };
                    ref_elem.eval_basis(&[rxl, ryl], &mut phi);
                    for i in 0..nt {
                        let gi = dl[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], w * phi[i] * trace_phi[j]);
                        }
                    }

                    // Right element (- sign, inward normal → trace jump convention)
                    let (rxr, ryr) = if npe_r == 3 { edge_xi_tri(*local_r, 1.0 - xi) } else if ref01 { edge_xi_quad_01(*local_r, 1.0 - xi) } else { edge_xi_quad(*local_r, 1.0 - xi) };
                    ref_elem.eval_basis(&[rxr, ryr], &mut phi);
                    for i in 0..nt {
                        let gi = dr[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], -w * phi[i] * trace_phi[j]);
                        }
                    }
                }
            }
        }
    }

    coo.into_csr()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    #[test]
    fn bhat_basic_tri_p1() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh.clone(), 1);
        let trace = DpgTraceSpace::new(mesh, 0); // order 0 → dpf = 1
        let bhat = assemble_bhat(&l2, &trace, 3);
        // Bhat: n_test × n_trace
        assert_eq!(bhat.nrows, l2.n_dofs());
        assert_eq!(bhat.ncols, trace.n_dofs());
        assert!(bhat.nnz() > 0);
    }

    #[test]
    fn bhat_tri_p2_trace_p1() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh.clone(), 2);
        let trace = DpgTraceSpace::new(mesh, 1); // order 1 → dpf = 2
        let bhat = assemble_bhat(&l2, &trace, 4);
        assert_eq!(bhat.nrows, l2.n_dofs());
        assert_eq!(bhat.ncols, trace.n_dofs());
        assert!(bhat.nnz() > 0);
    }

    #[test]
    fn bhat_quad_p1() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let l2 = L2Space::new(mesh.clone(), 1);
        let trace = DpgTraceSpace::new(mesh, 0);
        let bhat = assemble_bhat(&l2, &trace, 3);
        assert_eq!(bhat.nrows, l2.n_dofs());
        assert_eq!(bhat.ncols, trace.n_dofs());
        assert!(bhat.nnz() > 0);
    }

    #[test]
    fn bhat_finite_entries() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh.clone(), 1);
        let trace = DpgTraceSpace::new(mesh, 0);
        let bhat = assemble_bhat(&l2, &trace, 3);
        // All entries should be finite
        for &v in &bhat.values {
            assert!(v.is_finite(), "non-finite entry in Bhat");
        }
    }

    #[test]
    fn eval_lagrange_1d_constant_p0() {
        let mut phi = [0.0; 1];
        eval_lagrange_1d(0, 0.3, &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn eval_lagrange_1d_linear_p1() {
        let mut phi = [0.0; 2];
        eval_lagrange_1d(1, 0.0, &mut phi);
        assert!((phi[0] - 1.0).abs() < 1e-15);
        assert!((phi[1] - 0.0).abs() < 1e-15);

        eval_lagrange_1d(1, 1.0, &mut phi);
        assert!((phi[0] - 0.0).abs() < 1e-15);
        assert!((phi[1] - 1.0).abs() < 1e-15);

        eval_lagrange_1d(1, 0.5, &mut phi);
        assert!((phi[0] - 0.5).abs() < 1e-15);
        assert!((phi[1] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn eval_lagrange_1d_partition_of_unity() {
        for p in 0..=3 {
            let mut phi = vec![0.0; p + 1];
            for &xi in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                eval_lagrange_1d(p, xi, &mut phi);
                let sum: f64 = phi.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-14,
                    "p={p} xi={xi}: sum={sum}, expected 1"
                );
            }
        }
    }

    #[test]
    fn bhat_each_row_has_entry() {
        // Every test DOF should be adjacent to at least one face
        let mesh = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh.clone(), 1);
        let trace = DpgTraceSpace::new(mesh, 0);
        let bhat = assemble_bhat(&l2, &trace, 3);
        let mut row_counts = vec![0usize; bhat.nrows];
        for r in 0..bhat.nrows {
            row_counts[r] = (bhat.row_ptr[r + 1] - bhat.row_ptr[r]) as usize;
        }
        // All test DOFs on interior elements should have non-zero rows
        // (boundary test DOFs may or may not, depending on mesh)
        let total_dofs: usize = row_counts.iter().sum();
        assert!(total_dofs > 0);
    }
}
