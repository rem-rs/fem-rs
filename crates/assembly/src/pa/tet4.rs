//! Tet4 (linear tetrahedron) partial-assembly for diffusion.
//!
//! 4 nodes, constant reference gradients; the element kernel is the
//! per-quadrature-point metric form `y_i = Σ_q w_q |det J_q| κ(x_q)
//! (J⁻ᵀ∇φ_i)·(J⁻ᵀ∇φ_j) x_j`.
//!
//! # Geometry (D808-4, residual closed in round 78)
//!
//! Round 76 fixed the hex/quad PA kernels to evaluate the mesh's order-`g`
//! isoparametric map (`pa::curved::curved_jacobian`, `geom_order <= 1` taking the
//! verbatim straight path) but left `tet4` alone: its `PaData` held a **single
//! centroid quadrature point**, and one QP cannot represent a varying
//! `det J`/`J⁻ᵀJ⁻¹` even if `J` is evaluated with the curved map.  The data
//! layout is now rule-shaped:
//!
//! * **Straight** (`geom_order <= 1`): `nqp = 1`, the reference-tetra centroid
//!   `(1/4,1/4,1/4)`, through `ElementTransformation::from_simplex_nodes` — the
//!   verbatim pre-fix arithmetic (`|det J|/6` is the centroid rule's weight
//!   `1/6` folded in as a division), so no straight-mesh number moves
//!   (bit-pinned by `crates/assembly/tests/d808r78_tet4_pa_curved.rs`).
//! * **Curved** (`geom_order >= 2`): `nqp = npoints(tet_rule(2p+1))` with `p = 1`
//!   — the assembled path's own rule at the order the sibling PA kernels use
//!   (`prism_pk` takes `tri_rule(2p+1)`, the round-76 hex/quad kernels `2p+1`
//!   points per direction), each QP carrying its own `(J⁻ᵀ, |det J|, κ(x_q))`
//!   from `fem_mesh::transformation::element_jacobian_at`, i.e. *exactly* the
//!   map and rule `Assembler::assemble_bilinear(.., 2p+1)` integrates.  The
//!   gate is the same `curved_jacobian` one the hex/quad family uses.
//!
//! Pinned by `crates/assembly/tests/d808r78_tet4_pa_curved.rs`: on a curved
//! multi-element fixture PA-apply and the assembled SpMV agree to round-off,
//! while the pre-fix (single centroid QP, vertex geometry) route is `O(1e-1)`
//! off; on straight meshes every value is bit-identical to the round-77
//! release.

use crate::pa::types::PaData;
use fem_element::lagrange::TetP1;
use fem_element::quadrature::tet_rule;
use fem_element::reference::QuadratureRule;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::ElementTransformation;

/// The curved branch's quadrature: `tet_rule(2p+1)` at the P1 order `p = 1` —
/// MFEM's `IntRules.Get(Geometry::TETRAHEDRON, 2p+1)`, the rule the assembled
/// path is handed (`Assembler::assemble_bilinear(.., 2p+1)`) and the one the
/// sibling kernels use (`prism_pk`: `tri_rule(2p+1)`; hex/quad: `2p+1` points
/// per direction).
fn curved_tet_rule() -> QuadratureRule {
    tet_rule(3)
}

/// Build PA data for Tet4 diffusion: per element, per quadrature point,
/// `[J⁻ᵀ (row-major), |det J|, κ(x_q)]`.
///
/// `nqp == 1` marks the straight branch (centroid, `|det J|/6` applied by the
/// kernel) and `nqp == npoints(curved_tet_rule())` the curved one (the rule
/// weight is applied by the kernel); see the module docs.
pub fn build_tet4_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
) -> PaData {
    let n_elems = mesh.n_elements();
    let curved = mesh.geom_order() >= 2;
    let rule = curved_tet_rule();
    let nqp = if curved { rule.points.len() } else { 1 };
    let mut pd = PaData::new(n_elems, nqp, 3);

    for e in 0..n_elems {
        if curved {
            for (q, xi) in rule.points.iter().enumerate() {
                let xi3 = [xi[0], xi[1], xi[2]];
                let (jac, xp) = super::curved::curved_jacobian(mesh, e as u32, &xi3)
                    .expect("geom_order >= 2 tet geometry table");
                // `curved_jacobian` is the order-`g` map's `∂x_d/∂ξ_c` (row =
                // reference); `invert_3x3` returns the plain inverse, i.e. the
                // `∇_phys = J⁻ᵀ∇_ref` transform this layout stores.
                let (det, jit) = super::curved::invert_3x3(&jac);
                let qd = pd.elem_qp_mut(e, q);
                for i in 0..3 {
                    for j in 0..3 {
                        qd[i * 3 + j] = jit[i][j];
                    }
                }
                qd[9] = det.abs();
                qd[10] = kappa(&xp);
            }
        } else {
            // Verbatim pre-round-78 straight path: the reference-tetra centroid
            // through `ElementTransformation`, `|det J| / 6` applied in
            // `pa_apply_tet4` exactly as before.
            let nodes = mesh.element_nodes(e as u32);
            let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
            let det_j = tr.det_j().abs();
            let jit = tr.jacobian_inv_t();
            let qd = pd.elem_qp_mut(e, 0);
            for i in 0..3 {
                for j in 0..3 {
                    qd[i * 3 + j] = jit[(i, j)];
                }
            }
            qd[9] = det_j;
            let xp = tr.map_to_physical(&[0.25; 3]);
            qd[10] = kappa(&xp);
        }
    }
    pd
}

/// y += A·x for Tet4 diffusion via PA.
/// Uses the same reference gradients as TetP1 from fem-element.
pub fn pa_apply_tet4(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    let ref_elem = TetP1;

    // Per-QP reference gradients and weights of the rule the data was built
    // with.  The single-QP case is the straight branch: its point is the
    // reference-tetra centroid and its "weight" is the `1/6` the kernel folds
    // into `|det J|` by division (bit-identical to the pre-round-78 arithmetic).
    let curved = curved_tet_rule();
    let qps: Vec<([f64; 3], f64, [f64; 12])> = if pd.nqp == 1 {
        let xi = [0.25_f64; 3];
        let mut g = [0.0_f64; 12];
        ref_elem.eval_grad_basis(&xi, &mut g);
        vec![(xi, 0.0, g)]
    } else {
        assert_eq!(
            pd.nqp,
            curved.points.len(),
            "pa_apply_tet4: PaData.nqp must be 1 (straight) or npoints(tet_rule(3)) (curved)"
        );
        curved
            .points
            .iter()
            .zip(curved.weights.iter())
            .map(|(xi, &w)| {
                let xi3 = [xi[0], xi[1], xi[2]];
                let mut g = [0.0_f64; 12];
                ref_elem.eval_grad_basis(&xi3, &mut g);
                (xi3, w, g)
            })
            .collect()
    };

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let mut xe = [0.0_f64; 4];
        for i in 0..4 {
            xe[i] = x[dofs[i] as usize];
        }
        let mut ye = [0.0_f64; 4];

        for (q, (_xi, wq, grad_ref)) in qps.iter().enumerate() {
            let qd = pd.elem_qp(e, q);
            let j00 = qd[0]; let j01 = qd[1]; let j02 = qd[2];
            let j10 = qd[3]; let j11 = qd[4]; let j12 = qd[5];
            let j20 = qd[6]; let j21 = qd[7]; let j22 = qd[8];
            // Integration weight × κ: the straight branch divides by the
            // reference-tetra measure exactly as the pre-fix kernel did.
            let scale = if pd.nqp == 1 {
                qd[9] / 6.0 * qd[10]
            } else {
                wq * qd[9] * qd[10]
            };

            let gx = [grad_ref[0], grad_ref[3], grad_ref[6], grad_ref[9]];
            let gy = [grad_ref[1], grad_ref[4], grad_ref[7], grad_ref[10]];
            let gz = [grad_ref[2], grad_ref[5], grad_ref[8], grad_ref[11]];

            // Physical gradients ∇φ_i = J⁻ᵀ ∇_ref φ_i.
            let mut pgx = [0.0_f64; 4];
            let mut pgy = [0.0_f64; 4];
            let mut pgz = [0.0_f64; 4];
            for i in 0..4 {
                pgx[i] = j00 * gx[i] + j01 * gy[i] + j02 * gz[i];
                pgy[i] = j10 * gx[i] + j11 * gy[i] + j12 * gz[i];
                pgz[i] = j20 * gx[i] + j21 * gy[i] + j22 * gz[i];
            }

            for i in 0..4 {
                let di = pgx[i] * pgx[i] + pgy[i] * pgy[i] + pgz[i] * pgz[i];
                let mut s = di * xe[i];
                for j in 0..4 {
                    if j == i { continue; }
                    s += (pgx[i] * pgx[j] + pgy[i] * pgy[j] + pgz[i] * pgz[j]) * xe[j];
                }
                ye[i] += scale * s;
            }
        }

        for i in 0..4 { y[dofs[i] as usize] += ye[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;

    #[test]
    fn tet4_pa_finite() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let pd = build_tet4_pa_data(&mesh, &|_| 1.0);
        assert!(pd.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tet4_pa_matches_assembled() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);

        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let space2 = H1Space::new(mesh2, 1);
        let pd = build_tet4_pa_data(space2.mesh(), &|_| 1.0);
        let elem_dofs: Vec<Vec<u32>> = (0..space2.mesh().n_elements() as u32)
            .map(|e| space2.element_dofs(e).to_vec())
            .collect();

        let n = space.n_dofs();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect();

        let mut y_ref = vec![0.0; n];
        mat.spmv(&x, &mut y_ref);
        let mut y_pa = vec![0.0; n];
        pa_apply_tet4(&pd, &elem_dofs, &x, &mut y_pa);

        let max_err: f64 = (0..n).map(|i| (y_pa[i] - y_ref[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-12, "Tet4 PA max error {max_err}");
    }

    /// D808-4-r78: a curved tet mesh takes the multi-QP layout (the rule the
    /// assembled path is handed), a straight one the single-centroid one.
    #[test]
    fn tet4_pa_layout_follows_geom_order() {
        let straight = Mesh::<3>::unit_cube_tet(1);
        assert_eq!(straight.geom_order(), 1);
        assert_eq!(build_tet4_pa_data(&straight, &|_| 1.0).nqp, 1);

        let mut curved = Mesh::<3>::unit_cube_tet(1);
        curved.set_curvature(2);
        assert_eq!(curved.geom_order(), 2);
        let nqp = build_tet4_pa_data(&curved, &|_| 1.0).nqp;
        assert_eq!(nqp, curved_tet_rule().points.len());
        assert!(nqp > 1, "the curved branch must integrate more than one point");
    }
}
