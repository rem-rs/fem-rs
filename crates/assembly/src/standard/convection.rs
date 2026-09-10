//! Convection bilinear form integrator.
//!
//! Computes the element contribution to
//!
//! ```text
//! a(u, v) = ∫_Ω (b · ∇u) v dx
//! ```
//!
//! where `b` is a vector-valued convection velocity field.

use crate::postproc::coefficient::{CoeffCtx, VectorCoeff};
use crate::integrator::{BilinearIntegrator, QpData};

/// Bilinear integrator for the convection operator `(b · ∇u) v`.
///
/// The velocity field `b` is provided as a [`VectorCoeff`].  The resulting
/// matrix is **non-symmetric** (advection biases one direction).
///
/// # Example
/// ```rust,ignore
/// use fem_assembly::standard::ConvectionIntegrator;
/// use fem_assembly::coefficient::ConstantVectorCoeff;
/// // Uniform wind in x-direction
/// let integ = ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) };
/// ```
pub struct ConvectionIntegrator<V: VectorCoeff> {
    /// Convection velocity field.
    pub velocity: V,
}

impl<V: VectorCoeff> BilinearIntegrator for ConvectionIntegrator<V> {
    /// `K_elem[i,j] += w · φᵢ · (b · ∇φⱼ)`
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        let ctx = CoeffCtx::from_qp(
            qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag,
            Some(qp.phi), qp.elem_dofs,
        );

        // Evaluate velocity at this QP.
        let mut b = [0.0_f64; 3];
        self.velocity.eval(&ctx, &mut b[..d]);

        for i in 0..n {
            let phi_i = qp.phi[i];
            for j in 0..n {
                // b · ∇φⱼ
                let mut b_dot_grad_j = 0.0;
                for k in 0..d {
                    b_dot_grad_j += b[k] * qp.grad_phys[j * d + k];
                }
                // MFEM `ConvectionIntegrator::AssembleElementMatrix`:
                //   el.CalcDShape(ip, dshape);
                //   CalcAdjugate(Trans.Jacobian(), adjJ);
                //   vec1 = alpha * Q(ip) * ip.weight;   vec2 = adjJ * vec1;
                //   dshape.Mult(vec2, BdFidxT);         AddMultVWt(shape, BdFidxT, elmat);
                // → K_ij += ip.weight · φᵢ · (b · adjJᵀ∇φⱼ), i.e. the BARE
                // quadrature weight with the |det J| factor carried by the
                // adjugate Jacobian (adjJᵀ = det J · J⁻ᵀ).  `qp.grad_phys` is
                // adjJᵀ∇φ on every assembler path (affine and isoparametric —
                // see `accumulate_volume_bilinear_element`), so `ref_weight`
                // (= ip.weight) is the matching multiplier.  Using
                // `qp.weight` (= ip.weight/|det J|) here would divide by |det J|.
                k_elem[i * n + j] += qp.ref_weight * phi_i * b_dot_grad_j;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::postproc::coefficient::ConstantVectorCoeff;
    use crate::standard::MassIntegrator;
    use fem_mesh::{ElementType, Mesh, MeshTopology};
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    /// `K·x = M·1` for the convection operator with wind `b = (1,0)` and
    /// `u(x) = x`: `∇u = (1,0)` so `b·∇u = 1`, hence
    /// `(K u)_i = ∫ φᵢ dx = (M 1)_i`.
    ///
    /// This is the round-10 regression: the affine simplex branch stored the
    /// true physical gradient `J⁻ᵀ∇φ` together with the bare `ip.weight`, so
    /// every element matrix was scaled by `1/|det J|`.  On `unit_square_tri(2)`
    /// (`|det J| = 1/4` on every element) that gave a 0.75 absolute deviation.
    fn check_k_times_x_equals_m_times_one(mesh: Mesh<2>, quad: u8) -> f64 {
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad);
        let k = Assembler::assemble_bilinear(
            &space,
            &[&ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) }],
            quad,
        );

        // Nodal interpolation of u(x) = x: for P1 H1, DOF i is node i.
        let mut x = vec![0.0_f64; n];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = space.mesh().node_coords(i as u32)[0];
        }
        let ones = vec![1.0_f64; n];

        let mut kx = vec![0.0_f64; n];
        k.spmv(&x, &mut kx);
        let mut m1 = vec![0.0_f64; n];
        m.spmv(&ones, &mut m1);

        let dev = (0..n).map(|i| (kx[i] - m1[i]).abs()).fold(0.0, f64::max);

        // Pre-fix magnitude: the affine branch used to hand ConvectionIntegrator
        // the true gradient J⁻ᵀ∇φ, so on a mesh whose elements all share the same
        // |det J| the old element matrices were exactly K_e/|det J| → the old
        // deviation was |1/|det J| − 1|·max|M·1|.
        let m1_max = m1.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let old_dev = m1_max * (1.0 / affine_det_j(space.mesh()) - 1.0).abs();
        eprintln!(
            "K·x = M·1 on {} elements (n_dofs={n}): max |K·x − M·1| = {dev:.3e} \
             (pre-fix ≈ {old_dev:.3e})",
            space.mesh().n_elements()
        );
        dev
    }

    /// `|det J|` of the first (simplex) element — the factor the old affine
    /// branch got wrong.  Returns 1.0 for the non-affine path (where the old
    /// code was already correct, i.e. the pre-fix deviation was 0).
    fn affine_det_j(mesh: &Mesh<2>) -> f64 {
        let et = mesh.element_type(0);
        if !matches!(et, ElementType::Tri3) {
            return 1.0;
        }
        let nds = mesh.element_nodes(0);
        let c0 = mesh.node_coords(nds[0]);
        let c1 = mesh.node_coords(nds[1]);
        let c2 = mesh.node_coords(nds[2]);
        ((c1[0] - c0[0]) * (c2[1] - c0[1]) - (c1[1] - c0[1]) * (c2[0] - c0[0])).abs()
    }

    /// Affine simplex path (Tri3, `|det J| ≠ 1`) — the branch that lost |det J|.
    #[test]
    fn convection_identity_tri_affine_path() {
        // n = 2 → 8 elements, each with |det J| = 1/8 ≠ 1, so a missing |det J|
        // factor shows up by a factor of 8.
        for &n in &[1usize, 2, 4] {
            let dev = check_k_times_x_equals_m_times_one(Mesh::<2>::unit_square_tri(n), 3);
            assert!(
                dev <= 1e-13,
                "max |K·x − M·1| = {dev:.3e} on unit_square_tri({n}) (expected ≤ 1e-13)"
            );
        }
    }

    /// Tet4 affine path (3-D).
    #[test]
    fn convection_identity_tet_affine_path() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let k = Assembler::assemble_bilinear(
            &space,
            &[&ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0, 0.0]) }],
            3,
        );
        let mut x = vec![0.0_f64; n];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = space.mesh().node_coords(i as u32)[0];
        }
        let ones = vec![1.0_f64; n];
        let mut kx = vec![0.0_f64; n];
        k.spmv(&x, &mut kx);
        let mut m1 = vec![0.0_f64; n];
        m.spmv(&ones, &mut m1);
        let dev = (0..n).map(|i| (kx[i] - m1[i]).abs()).fold(0.0, f64::max);
        assert!(dev <= 1e-13, "max |K·x − M·1| = {dev:.3e} on unit_cube_tet(2)");
    }

    /// Non-affine path (Quad4) must stay correct as well.
    #[test]
    fn convection_identity_quad_isoparametric_path() {
        for &n in &[1usize, 2] {
            let dev = check_k_times_x_equals_m_times_one(Mesh::<2>::unit_square_quad(n), 3);
            assert!(
                dev <= 1e-13,
                "max |K·x − M·1| = {dev:.3e} on unit_square_quad({n}) (expected ≤ 1e-13)"
            );
        }
    }

    /// Sanity: the mesh type used above really is an affine simplex mesh.
    #[test]
    fn tri_mesh_is_affine_simplex() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        assert_eq!(mesh.element_type(0), ElementType::Tri3);
        assert_eq!(mesh.geom_order(), 1);
    }

    /// Convection matrix with uniform b = (1, 0) should be non-symmetric.
    #[test]
    fn convection_is_non_symmetric() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let integ = ConvectionIntegrator {
            velocity: ConstantVectorCoeff(vec![1.0, 0.0]),
        };
        let mat = Assembler::assemble_bilinear(&space, &[&integ], 3);
        let dense = mat.to_dense();
        let n = mat.nrows;
        // Check that at least one (i,j) pair is non-symmetric.
        let mut has_asymmetry = false;
        for i in 0..n {
            for j in 0..n {
                if (dense[i * n + j] - dense[j * n + i]).abs() > 1e-12 {
                    has_asymmetry = true;
                }
            }
        }
        assert!(has_asymmetry, "convection matrix should be non-symmetric");
    }
}
