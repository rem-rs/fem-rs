//! L² error computation for H(div) (Raviart-Thomas) solutions.
//!
//! [`compute_hdiv_l2_error`] evaluates the L² norm of the difference between a
//! discrete H(div) solution and a known exact vector field.
//!
//! Supports both affine (Tri3) and bilinear (Quad4) elements, using the
//! isoparametric Jacobian (per-quadrature-point) for quads.

use fem_element::{
    reference::VectorReferenceElement,
    raviart_thomas::{QuadRT0, TriRT0},
};
use fem_linalg::CsrMatrix;
use fem_mesh::{ElementTransformation, ElementType, Mesh, MeshTopology};
use fem_space::{HDivSpace, fe_space::FESpace};

use crate::{geo_ref_elem_from_mesh, isoparametric_jacobian};

/// Compute ‖F_h − F_exact‖_{L²(Ω)} for an H(div) solution.
///
/// # Arguments
/// * `space` — H(div) finite element space (RT0/RT1)
/// * `uh`    — discrete DOF values (length `space.n_dofs()`)
/// * `exact` — exact vector field `F(x) → [fx, fy]`
///
/// # Panics
/// Panics if the element type is unsupported.
pub fn compute_hdiv_l2_error<F>(
    space: &HDivSpace<Mesh<2>>,
    uh: &[f64],
    exact: F,
) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    let mesh = space.mesh();
    let elem_type = mesh.element_type(0);
    let is_quad = matches!(elem_type, ElementType::Quad4);
    let ref_elem: &dyn VectorReferenceElement = match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => &TriRT0,
        ElementType::Quad4 => &QuadRT0,
        _ => panic!("compute_hdiv_l2_error: unsupported element type {elem_type:?}"),
    };
    let quad = ref_elem.quadrature(6);
    let n_ldofs = ref_elem.n_dofs() as usize;
    let mut ref_phi = vec![0.0; n_ldofs * 2];
    let mut phys_phi = vec![0.0; n_ldofs * 2];
    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let nodes = mesh.element_nodes(e);

        // Pre-build affine transform for tri elements (reused at all q-points).
        let affine_tr = (!is_quad).then(|| ElementTransformation::from_simplex_nodes(mesh, nodes));

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j, xp) = if is_quad {
                let ge = geo_ref_elem_from_mesh(mesh as &dyn MeshTopology, e)
                    .expect("geometry element for quad");
                isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, 2)
            } else {
                let tr = affine_tr.as_ref().unwrap();
                (tr.jacobian().clone(), tr.det_j(), tr.map_to_physical(xi))
            };
            let w = quad.weights[qi] * det_j.abs();

            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            // Contravariant Piola: φ_phys = J · φ_ref / det(J)
            let inv_det = 1.0 / det_j;
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                phys_phi[i * 2]     = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) * inv_det;
                phys_phi[i * 2 + 1] = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) * inv_det;
            }

            let mut fh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                fh[0] += uh[dofs[i]] * phys_phi[i * 2];
                fh[1] += uh[dofs[i]] * phys_phi[i * 2 + 1];
            }

            let fe = exact(&xp);
            let dx = fh[0] - fe[0];
            let dy = fh[1] - fe[1];
            err2 += w * (dx * dx + dy * dy);
        }
    }

    err2.sqrt()
}
