//! L² error computation for H(div) (Raviart-Thomas) solutions.
//!
//! [`compute_hdiv_l2_error`] evaluates the L² norm of the difference between a
//! discrete H(div) solution and a known exact vector field.
//!
//! Supports both affine (Tri3) and bilinear (Quad4) elements, using the
//! isoparametric Jacobian (per-quadrature-point) for quads.

use fem_element::{
    reference::VectorReferenceElement,
    raviart_thomas::{QuadRT0, QuadRT1, TriRT0, TriRT1, TriRT2},
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
    compute_hdiv_l2_error_owned(space, uh, exact, &|_| true)
}

/// Compute ‖F_h − F_exact‖_{L²(Ω_owned)} for an H(div) solution, restricting
/// the element sum to elements for which `owned_pred(elem) == true`.
///
/// Used by the parallel path: each rank integrates only its owned elements
/// and the caller reduces the sum (after `.sqrt()` on the total).
pub fn compute_hdiv_l2_error_owned<F, P>(
    space: &HDivSpace<Mesh<2>>,
    uh: &[f64],
    exact: F,
    owned_pred: &P,
) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
    P: Fn(u32) -> bool,
{
    compute_hdiv_l2_error_owned_q(space, uh, exact, owned_pred, 2 * (space.order() as usize + 1) + 3)
}

/// Variant with an explicit quadrature order (MFEM ex5p uses
/// `max(2, 2*order+1)` for the error integral).
pub fn compute_hdiv_l2_error_owned_q<F, P>(
    space: &HDivSpace<Mesh<2>>,
    uh: &[f64],
    exact: F,
    owned_pred: &P,
    quad_order: usize,
) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
    P: Fn(u32) -> bool,
{
    let mesh = space.mesh();
    let elem_type = mesh.element_type(0);
    let order = space.order();
    let is_quad = matches!(elem_type, ElementType::Quad4);
    let ref_elem: &dyn VectorReferenceElement = match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => &TriRT0,
        (ElementType::Tri3 | ElementType::Tri6, 1) => &TriRT1,
        (ElementType::Tri3 | ElementType::Tri6, 2) => &TriRT2,
        (ElementType::Quad4, 0) => &QuadRT0,
        (ElementType::Quad4, 1) => &QuadRT1,
        _ => panic!("compute_hdiv_l2_error: unsupported (type={elem_type:?}, order={order})"),
    };
    // MFEM GridFunction::ComputeL2Error uses intorder = 2*fe->GetOrder() + 3
    // (gridfunc.cpp).  For RT elements GetOrder() = p + 1 where p is the RT
    // order (RT_QuadrilateralElement(p) → VectorTensorFiniteElement(..., p+1,
    // ...) in fe_rt.cpp), so intorder = 2*(order+1) + 3 = 5 for RT0.  On
    // non-affine (bilinear) quads the integrand is not a polynomial, so the
    // quadrature order changes the value — match MFEM exactly.
    let quad = ref_elem.quadrature(quad_order as u8);
    let n_ldofs = ref_elem.n_dofs() as usize;
    let mut ref_phi = vec![0.0; n_ldofs * 2];
    let mut phys_phi = vec![0.0; n_ldofs * 2];
    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        if !owned_pred(e) {
            continue;
        }
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

/// Compute ‖p_h − p_exact‖_{L²(Ω)} for a scalar L₂ (DG) solution.
///
/// Supports Tri3 (Pk) and Quad4 (Qk) elements — selects the correct
/// reference element based on the mesh element type.
pub fn compute_l2_error_scalar<F>(
    space: &fem_space::L2Space<Mesh<2>>,
    uh: &[f64],
    exact: F,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    compute_l2_error_scalar_owned(space, uh, exact, &|_| true)
}

/// Element-filtered variant of [`compute_l2_error_scalar`] (parallel path:
/// each rank integrates only its owned elements).
pub fn compute_l2_error_scalar_owned<F, P>(
    space: &fem_space::L2Space<Mesh<2>>,
    uh: &[f64],
    exact: F,
    owned_pred: &P,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    P: Fn(u32) -> bool,
{
    compute_l2_error_scalar_owned_q(space, uh, exact, owned_pred, 6)
}

/// Variant of [`compute_l2_error_scalar_owned`] with an explicit quadrature
/// order (MFEM ex5p uses `max(2, 2*order+1)` = 3 for the error integral).
pub fn compute_l2_error_scalar_owned_q<F, P>(
    space: &fem_space::L2Space<Mesh<2>>,
    uh: &[f64],
    exact: F,
    owned_pred: &P,
    quad_order: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    P: Fn(u32) -> bool,
{
    use fem_element::{
        lagrange::{QuadQ1, TriP1},
        ReferenceElement,
    };
    use fem_mesh::{ElementTransformation, ElementType};

    let mesh = space.mesh();
    let elem_type = mesh.element_type(0);
    let ref_elem: &dyn ReferenceElement = match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => &TriP1,
        // L2Space uses the Gauss-Legendre tensor-product basis
        // (QuadL2GL, matching MFEM L2_FECollection), not QuadQ1 (GLL).
        ElementType::Quad4 => &fem_element::lagrange::QuadL2GL::new(1),
        _ => panic!("compute_l2_error_scalar: unsupported element type {elem_type:?}"),
    };
    let quad = ref_elem.quadrature(quad_order as u8);
    let n_ldofs = ref_elem.n_dofs();
    let mut ref_phi = vec![0.0; n_ldofs];
    let mut err2 = 0.0_f64;

    let is_quad = elem_type == ElementType::Quad4;
    let affine_tr = (!is_quad).then(|| ElementTransformation::from_simplex_nodes(mesh, mesh.element_nodes(0)));

    for e in mesh.elem_iter() {
        if !owned_pred(e) {
            continue;
        }
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);

        for (qi, xi) in quad.points.iter().enumerate() {
            // Quad4 needs the isoparametric (bilinear) map, exactly like the
            // H(div) error routine — the affine simplex transform gives wrong
            // det(J) and wrong physical points on quads (p-L² error was
            // 2.4e-2 vs C++ 3.5e-5).
            let (xp, w) = if is_quad {
                let ge = crate::vector_assembler::geo_ref_elem_from_mesh(mesh, e)
                    .expect("geometry element for quad");
                let (jac, det_j, xp) = crate::vector_assembler::isoparametric_jacobian(mesh, nodes, ge.as_ref(), xi, 2);
                (xp, quad.weights[qi] * det_j.abs())
            } else {
                let tr = affine_tr.as_ref().expect("affine transform");
                (tr.map_to_physical(xi), quad.weights[qi] * tr.det_j().abs())
            };
            ref_elem.eval_basis(xi, &mut ref_phi);

            let mut vh = 0.0;
            for i in 0..n_ldofs {
                vh += uh[dofs[i]] * ref_phi[i];
            }
            let ve = exact(&xp);
            err2 += w * (vh - ve) * (vh - ve);
        }
    }

    err2.sqrt()
}
