//! L² error computation for H(div) / H(curl) vector fields and scalar fields,
//! plus the H(div) divergence error (D161).
//!
//! 1:1 with MFEM (`fem/gridfunc.cpp`):
//!
//! * **Scalar** — `GridFunction::ComputeL2Error(Coefficient *exsol[], …)`:
//!   `sqrt( Σ_e |Σ_q ip.weight · T.Weight() · (u_h(x_q) − u_ex(x_q))²| )`,
//!   with the default integration rule
//!   `IntRules.Get(fe->GetGeomType(), 2·fe->GetOrder() + 3)`.
//!   Note MFEM's FE-order convention: Raviart–Thomas elements report
//!   `fe->GetOrder() == p + 1`, so the H(div) default rule is `2(p+1)+3`
//!   (see [`mfem_fe_order`]) while H¹/L²/Nédélec elements report `p`.
//! * **Vector** — `GridFunction::ComputeL2Error(VectorCoefficient &exsol, …)`
//!   (the overload `ex4p`/`ex5p` use).  It evaluates the field through
//!   `GridFunction::GetVectorValues` →
//!   `FiniteElement::CalcVShape(Trans, ·)` →
//!   `VectorFiniteElement::CalcVShape_RT` / `CalcVShape_ND`
//!   (`fem/fe/fe_base.cpp`), i.e. with the Piola maps
//!
//!   ```text
//!   φ_phys(ξ_q) = φ̂(ξ_q) · Jᵀ / Weight()   (H(div), contravariant Piola)
//!   φ_phys(ξ_q) = φ̂(ξ_q) · J⁻¹             (H(curl), covariant Piola)
//!   ```
//!
//!   with `Weight() = sqrt(det(JᵀJ)) = |det J|` for a square Jacobian.  The
//!   physical quadrature weight is `ip.weight · T.Weight()`, i.e.
//!   `quad.weights[q] · |det J|`.
//!
//! The DOF orientation signs (`FESpace::element_signs`, MFEM's negative
//! `GetElementVDofs` entries) and the reference basis are exactly the ones the
//! assembler uses ([`crate::vector_assembler::vec_ref_elem`] +
//! [`crate::vector_assembler::piola_hdiv_basis`] /
//! [`crate::vector_assembler::piola_hcurl_basis`]), so the reconstructed field
//! is the same field the assembled system was solved for:
//! `compute_hdiv_l2_error(space, u, |_| 0)` reproduces `sqrt(uᵀ M u)` from the
//! assembled mass matrix to round-off (see the tests below).
//!
//! # Parallel semantics (owned-element filtering)
//!
//! On a partitioned mesh the *local* mesh is `[owned | ghost]`; integrating
//! every local element counts each ghost element once per rank holding it.
//! MFEM's `ParGridFunction::ComputeL2Error` (`fem/pgridfunc.hpp`) therefore
//! integrates only the **owned** elements of the local mesh (`ParMesh::GetNE()`
//! counts owned elements only) and reduces with `GlobalLpNorm(2, ·)`, i.e.
//! `sqrt(Σ_rank local_err²)`.
//!
//! The three-argument routines here integrate **all local elements** (serial
//! semantics = MFEM `GridFunction::ComputeL2Error` on a serial mesh).
//! Parallel codes pass an owned-element predicate to the `*_filtered` variants
//! — `include` plays the role of MFEM's `elems` marker array — and then apply
//! `sqrt(allreduce_sum(local_err²))` themselves.

use nalgebra::DMatrix;

use fem_mesh::element_jacobian_at;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::{FESpace, SpaceType};

// ─── Helpers ───────────────────────────────────────────────────────────────

/// MFEM's `FiniteElement::GetOrder()` for the elements of `space`.
///
/// MFEM's `ComputeL2Error` uses `2*fe->GetOrder() + 3`, and `fe->GetOrder()` is
/// **not** the `FiniteElementCollection` order for vector elements:
///
/// * Raviart–Thomas (`RT_FECollection(p)`): every element stores `p + 1`
///   (`VectorTensorFiniteElement(..., p + 1, ...)` for Quad/Hex,
///   `VectorFiniteElement(..., p + 1, ...)` for Tri/Tet/Prism — see
///   `fem/fe/fe_rt.cpp`), so H(div) gets `2(p+1)+3 = 2p+5`.
/// * Nédélec (`ND_FECollection(p)`): stores `p`, like the H¹/L² collections.
///
/// Using `2p+3` for H(div) under-integrates the error: e.g. RT0 on quads then
/// integrates with the 2×2 rule (exact to degree 3) while MFEM uses the 3×3
/// rule (exact to degree 5), which measurably changes the result for exact
/// fields with degree-4 content (probe: `u = (x², 0)` on RT0 quads gave
/// `2.6e-16` instead of `4.6585e-3`).
fn mfem_fe_order<S: FESpace>(space: &S, elem_order: u8) -> u8 {
    match space.space_type() {
        SpaceType::HDiv => elem_order.saturating_add(1),
        _ => elem_order,
    }
}

/// MFEM's default `ComputeL2Error` integration order: `2*fe->GetOrder() + 3`
/// (`fem/gridfunc.cpp`), with MFEM's FE-order convention (see
/// [`mfem_fe_order`]).  Callers that pass an explicit `irs` array (e.g.
/// `ex5p.cpp`: `max(2, 2*order+1)`) use the `*_order` variants below.
pub fn default_quad_order<S: FESpace>(space: &S) -> u8 {
    let fe_order = mfem_fe_order(space, space.order()) as u32;
    (2 * fe_order + 3).min(u8::MAX as u32) as u8
}

/// Element filter: `None` includes every local element (serial semantics).
#[inline]
fn included(include: Option<&dyn Fn(u32) -> bool>, e: u32) -> bool {
    match include {
        Some(f) => f(e),
        None => true,
    }
}

/// MFEM `ElementTransformation::{Jacobian, Transform}` at the reference point
/// `xi`: returns `(J, det J, x_phys)`.
///
/// `geo_ref_elem_from_mesh` returns `Some` for tensor-product elements
/// (Quad/Hex/Prism/Pyramid) and for any element of a curved (`geom_order > 1`)
/// mesh — the same split the assembler uses, so the Jacobian agrees with the
/// one the system was assembled with.
fn jacobian_and_point<M: MeshTopology>(
    mesh: &M,
    e: u32,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    match crate::geo_ref_elem_from_mesh(mesh, e) {
        Some(geo) => {
            let nodes = mesh.geometry_nodes(e);
            crate::isoparametric_jacobian(mesh, nodes, geo.as_ref(), xi, dim)
        }
        None => {
            let (jac, xp) = element_jacobian_at(mesh, e, xi, dim);
            let det = jac.determinant();
            (jac, det, xp)
        }
    }
}

// ─── Scalar L² error ───────────────────────────────────────────────────────

fn scalar_l2_error_impl<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let n_elems = mesh.n_elements() as u32;

    let mut err2 = 0.0_f64;
    for e in 0..n_elems {
        if !included(include, e) {
            continue;
        }
        let elem_type = mesh.element_type(e);
        let order = space.element_order(e);
        // The same reference basis the space was assembled with (H¹ vs L²/DG
        // node placement) — mirrors `Assembler::ref_elem_vol_for_space`.
        let re = crate::assembler::ref_elem_vol_for_space(space, elem_type, order);
        let n_ldofs = re.n_dofs();
        let elem_dofs = space.element_dofs(e);
        debug_assert_eq!(elem_dofs.len(), n_ldofs, "element DOF count mismatch");
        let quad = re.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = jacobian_and_point(mesh, e, xi, dim);
            let w = quad.weights[qi] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            let mut uh = 0.0_f64;
            for i in 0..n_ldofs {
                uh += u[elem_dofs[i] as usize] * phi[i];
            }
            let ue = exact(&xp);
            err2 += w * (uh - ue) * (uh - ue);
        }
    }
    err2.max(0.0).sqrt()
}

/// Scalar L² error `‖u_h − u_ex‖_{L²}` over **all local elements** (serial
/// semantics; use the `*_filtered` variants in a partitioned run).
pub fn compute_l2_error_scalar<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    scalar_l2_error_impl(space, u, exact, default_quad_order(space), None)
}

/// [`compute_l2_error_scalar`] with an explicit quadrature order, matching an
/// MFEM caller that passes its own `irs` (e.g. `ex5p.cpp`'s
/// `IntRules.Get(geom, max(2, 2*order+1))`).
pub fn compute_l2_error_scalar_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
) -> f64 {
    scalar_l2_error_impl(space, u, exact, quad_order, None)
}

/// [`compute_l2_error_scalar`] restricted to the elements for which
/// `include(e)` holds.
///
/// `include` plays the role of MFEM's `elems` marker array
/// (`GridFunction::ComputeL2Error(exsol, irs, elems)`); in a partitioned run
/// pass `Some(&|e| mesh.partition().elem_owner[e as usize] == rank)` so ghost
/// elements are not counted once per rank.
pub fn compute_l2_error_scalar_filtered<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    scalar_l2_error_impl(space, u, exact, default_quad_order(space), include)
}

/// [`compute_l2_error_scalar_filtered`] with an explicit quadrature order
/// (same role as the `irs` argument of `GridFunction::ComputeL2Error`).
pub fn compute_l2_error_scalar_filtered_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    scalar_l2_error_impl(space, u, exact, quad_order, include)
}

/// "Owned / quadrature-based" scalar variant: identical to
/// [`compute_l2_error_scalar`] — over **all local elements**.
///
/// The name is kept for API compatibility; it carries no ownership
/// information, so on a partitioned mesh use
/// [`compute_l2_error_scalar_filtered`] instead.
pub fn compute_l2_error_scalar_owned_q<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    compute_l2_error_scalar(space, u, exact)
}

// ─── H(div) / H(curl) vector L² error ──────────────────────────────────────

fn piola_l2_error_impl<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    let space_type = space.space_type();
    assert!(
        matches!(space_type, SpaceType::HDiv | SpaceType::HCurl),
        "piola_l2_error_impl: expected an H(div) or H(curl) space, got {space_type:?}; \
         use compute_l2_error_scalar for scalar spaces"
    );

    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let n_elems = mesh.n_elements() as u32;

    let mut err2 = 0.0_f64;
    for e in 0..n_elems {
        if !included(include, e) {
            continue;
        }
        let elem_type = mesh.element_type(e);
        let order = space.element_order(e);
        // Same reference basis as the assembly (`VectorAssembler::vec_ref_elem`).
        let vre = crate::vector_assembler::vec_ref_elem(space_type, elem_type, dim, order);
        let n_ldofs = vre.n_dofs();
        let elem_dofs = space.element_dofs(e);
        debug_assert_eq!(elem_dofs.len(), n_ldofs, "element DOF count mismatch");
        let signs = space.element_signs(e);
        let quad = vre.quadrature(quad_order);
        let mut ref_bv = vec![0.0_f64; n_ldofs * dim];
        let mut phys_bv = vec![0.0_f64; n_ldofs * dim];
        let mut uh = vec![0.0_f64; dim];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j, xp) = jacobian_and_point(mesh, e, xi, dim);
            let w = quad.weights[qi] * det_j.abs();

            vre.eval_basis_vec(xi, &mut ref_bv);
            match space_type {
                SpaceType::HDiv => {
                    // φ_phys = J φ̂ / det J  (contravariant Piola).  A
                    // degenerate (zero-measure) element would give inf/NaN;
                    // contribute zero there.
                    if det_j.abs() > 1e-80 {
                        crate::vector_assembler::piola_hdiv_basis(
                            &jac, det_j, &ref_bv, &mut phys_bv, n_ldofs, dim,
                        );
                    } else {
                        phys_bv.fill(0.0);
                    }
                }
                _ => {
                    // φ_phys = J⁻ᵀ φ̂  (covariant Piola, H(curl)).
                    let jac_inv_t = jac
                        .clone()
                        .try_inverse()
                        .map(|jit| jit.transpose())
                        .unwrap_or_else(|| DMatrix::<f64>::identity(dim, dim));
                    crate::vector_assembler::piola_hcurl_basis(
                        &jac_inv_t, &ref_bv, &mut phys_bv, n_ldofs, dim,
                    );
                }
            }

            uh.fill(0.0);
            for i in 0..n_ldofs {
                let s = match signs {
                    Some(sg) => sg.get(i).copied().unwrap_or(1.0),
                    None => 1.0,
                };
                let coeff = u[elem_dofs[i] as usize] * s;
                for c in 0..dim {
                    uh[c] += coeff * phys_bv[i * dim + c];
                }
            }

            let ex = exact(&xp);
            for c in 0..dim {
                let d = uh[c] - ex[c];
                err2 += w * d * d;
            }
        }
    }
    err2.max(0.0).sqrt()
}

/// L² error of an H(div) (contravariant Piola) or H(curl) (covariant Piola)
/// vector field: `‖u_h − u_ex‖_{L²}` over **all local elements** (serial
/// semantics; MFEM's vector `GridFunction::ComputeL2Error`).
///
/// For a partitioned run see [`compute_hdiv_l2_error_filtered`].
pub fn compute_hdiv_l2_error<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    piola_l2_error_impl(space, u, exact, default_quad_order(space), None)
}

/// [`compute_hdiv_l2_error`] with an explicit quadrature order (same role as
/// the `irs` argument of `GridFunction::ComputeL2Error`).
pub fn compute_hdiv_l2_error_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
) -> f64 {
    piola_l2_error_impl(space, u, exact, quad_order, None)
}

/// [`compute_hdiv_l2_error`] restricted to the elements for which `include(e)`
/// holds — the parallel building block matching MFEM
/// `ParGridFunction::ComputeL2Error` (owned elements + `sqrt(Σ local²)`).
pub fn compute_hdiv_l2_error_filtered<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    piola_l2_error_impl(space, u, exact, default_quad_order(space), include)
}

/// [`compute_hdiv_l2_error_filtered`] with an explicit quadrature order (same
/// role as the `irs` argument of `GridFunction::ComputeL2Error`).
pub fn compute_hdiv_l2_error_filtered_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    piola_l2_error_impl(space, u, exact, quad_order, include)
}

/// "Owned / quadrature-based" vector variant: identical to
/// [`compute_hdiv_l2_error`] — over **all local elements**.
///
/// The name is kept for API compatibility; it carries no ownership
/// information, so on a partitioned mesh use
/// [`compute_hdiv_l2_error_filtered`] instead.
pub fn compute_hdiv_l2_error_owned_q<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    compute_hdiv_l2_error(space, u, exact)
}

/// Vector variant over **all local elements** (see
/// [`compute_hdiv_l2_error_owned_q`] about the retained name).
pub fn compute_hdiv_l2_error_owned<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    compute_hdiv_l2_error(space, u, exact)
}

// ─── H(div) divergence error (MFEM GridFunction::ComputeDivError) ──────────

fn div_error_impl<S: FESpace>(
    space: &S,
    u: &[f64],
    exact_div: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    let space_type = space.space_type();
    assert!(
        space_type == SpaceType::HDiv,
        "div_error_impl: expected an H(div) space, got {space_type:?}; MFEM's \
         ComputeDivError needs FiniteElement::CalcDivShape, which only the \
         Raviart-Thomas (H(div)) elements implement (ND elements abort there)"
    );

    let mesh = space.mesh();
    let dim = mesh.topological_dim() as usize;
    let n_elems = mesh.n_elements() as u32;

    let mut err2 = 0.0_f64;
    for e in 0..n_elems {
        if !included(include, e) {
            continue;
        }
        let elem_type = mesh.element_type(e);
        let order = space.element_order(e);
        let vre = crate::vector_assembler::vec_ref_elem(space_type, elem_type, dim, order);
        let n_ldofs = vre.n_dofs();
        let elem_dofs = space.element_dofs(e);
        debug_assert_eq!(elem_dofs.len(), n_ldofs, "element DOF count mismatch");
        let signs = space.element_signs(e);
        let quad = vre.quadrature(quad_order);
        let mut ref_div = vec![0.0_f64; n_ldofs];
        let mut phys_div = vec![0.0_f64; n_ldofs];

        // MFEM accumulates per element and adds `fabs(elem_error)` (the
        // negative-quadrature-weights guard, fem/gridfunc.cpp:3648).
        let mut elem_error = 0.0_f64;
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = jacobian_and_point(mesh, e, xi, dim);
            let w = quad.weights[qi] * det_j.abs();

            // div u_h(x) = Σ_i s_i·u_i · div φ̂_i(ξ) / det J — MFEM
            // `GridFunction::GetDivergence`'s RT branch, `(dofs·divshape) /
            // Weight` (fem/gridfunc.cpp:1441-1444; Weight = |det J| = det J
            // for the oriented elements the assembler builds).
            vre.eval_div(xi, &mut ref_div);
            crate::vector_assembler::piola_hdiv_div(det_j, &ref_div, &mut phys_div, n_ldofs);
            let mut duh = 0.0_f64;
            for i in 0..n_ldofs {
                let s = match signs {
                    Some(sg) => sg.get(i).copied().unwrap_or(1.0),
                    None => 1.0,
                };
                duh += u[elem_dofs[i] as usize] * s * phys_div[i];
            }

            let a = duh - exact_div(&xp);
            elem_error += w * a * a;
        }
        err2 += elem_error.abs();
    }
    err2.max(0.0).sqrt()
}

/// Divergence error `‖div u_h − div u_ex‖_{L²}` of an H(div) field over **all
/// local elements** — 1:1 with MFEM `GridFunction::ComputeDivError(exdiv)`
/// (`fem/gridfunc.cpp:3618`): default rule `2·fe->GetOrder() + 3` (with the
/// RT `GetOrder() == p+1` convention, see [`default_quad_order`]), pointwise
/// integrand `(GetDivergence(Tr) − exdiv(Tr))²`, per-element accumulation
/// with the `fabs` guard, `sqrt` at the end.
///
/// H(div) only: MFEM's `GetDivergence` reaches `FiniteElement::CalcDivShape`,
/// which the Nédélec elements do not implement (panics here for non-H(div)
/// vector spaces, like MFEM's `MFEM_ABORT`).
pub fn compute_div_error<S: FESpace>(
    space: &S,
    u: &[f64],
    exact_div: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    div_error_impl(space, u, exact_div, default_quad_order(space), None)
}

/// [`compute_div_error`] with an explicit quadrature order (same role as the
/// `irs` argument of `GridFunction::ComputeDivError`).
pub fn compute_div_error_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact_div: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
) -> f64 {
    div_error_impl(space, u, exact_div, quad_order, None)
}

/// [`compute_div_error`] restricted to the elements for which `include(e)`
/// holds (parallel owned-element filtering; see
/// [`compute_hdiv_l2_error_filtered`]).
pub fn compute_div_error_filtered<S: FESpace>(
    space: &S,
    u: &[f64],
    exact_div: &dyn Fn(&[f64]) -> f64,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    div_error_impl(space, u, exact_div, default_quad_order(space), include)
}

/// [`compute_div_error_filtered`] with an explicit quadrature order.
pub fn compute_div_error_filtered_order<S: FESpace>(
    space: &S,
    u: &[f64],
    exact_div: &dyn Fn(&[f64]) -> f64,
    quad_order: u8,
    include: Option<&dyn Fn(u32) -> bool>,
) -> f64 {
    div_error_impl(space, u, exact_div, quad_order, include)
}

/// Full H(div) norm error `sqrt(‖u_h − u_ex‖²_{L²} + ‖div u_h − div u_ex‖²_{L²})`
/// — 1:1 with MFEM `GridFunction::ComputeHDivError(exsol, exdiv)`
/// (`fem/gridfunc.cpp:3809`): the L² part from
/// [`compute_hdiv_l2_error`], the divergence part from
/// [`compute_div_error`], combined in quadrature.
pub fn compute_hdiv_full_error<S: FESpace>(
    space: &S,
    u: &[f64],
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
    exact_div: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    let l2 = compute_hdiv_l2_error(space, u, exact);
    let div = compute_div_error(space, u, exact_div);
    (l2 * l2 + div * div).sqrt()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{LinearIntegrator, QpData};
    use crate::standard::{MassIntegrator, VectorMassIntegrator};
    use crate::vector_assembler::VectorAssembler;
    use crate::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    use crate::Assembler;
    use fem_linalg::dense::{lu_factor, lu_solve};
    use fem_linalg::CsrMatrix;
    use fem_mesh::Mesh;
    use fem_space::{HCurlSpace, HDivSpace, H1Space, L2Space};

    /// Zero exact vector field (turns the error routine into a norm).
    const ZERO2: fn(&[f64]) -> Vec<f64> = |_| vec![0.0, 0.0];

    /// `b_i = ∫ u_ex · φ_i` — the RHS of the L² projection (MFEM
    /// `LinearForm` + `VectorDomainLFIntegrator`).
    struct VectorRhs(fn(&[f64]) -> Vec<f64>);

    impl VectorLinearIntegrator for VectorRhs {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let ue = (self.0)(qp.x_phys);
            for i in 0..qp.n_dofs {
                let mut dot = 0.0;
                for c in 0..qp.dim {
                    dot += qp.phi_vec[i * qp.dim + c] * ue[c];
                }
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    /// `b_i = ∫ f φ_i` — scalar RHS.
    struct ScalarRhs(fn(&[f64]) -> f64);

    impl LinearIntegrator for ScalarRhs {
        fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
            let fv = (self.0)(qp.x_phys);
            for i in 0..qp.n_dofs {
                f_elem[i] += qp.phys_weight * fv * qp.phi[i];
            }
        }
    }

    /// Direct dense solve of a small CSR system.
    fn dense_solve(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
        let n = a.nrows;
        assert_eq!(n, a.ncols);
        let mut ad = vec![0.0_f64; n * n];
        for row in 0..n {
            for k in a.row_ptr[row]..a.row_ptr[row + 1] {
                ad[row * n + a.col_idx[k] as usize] = a.values[k];
            }
        }
        let mut piv = vec![0usize; n];
        lu_factor(&mut ad, n, &mut piv).expect("lu_factor");
        let mut x = b.to_vec();
        x.resize(n, 0.0);
        lu_solve(&ad, n, &piv, &mut x);
        x
    }

    /// Vector mass matrix on an H(div)/H(curl) space.
    fn vector_mass<S>(space: &S, q: u8) -> CsrMatrix<f64>
    where
        S: FESpace + Sync,
        S::Mesh: MeshTopology + Sync,
    {
        VectorAssembler::assemble_bilinear(
            space,
            &[&VectorMassIntegrator { alpha: 1.0 }],
            q,
        )
    }

    /// `sqrt(uᵀ M u)`.
    fn mat_norm(m: &CsrMatrix<f64>, u: &[f64]) -> f64 {
        let mut mu = vec![0.0; u.len()];
        m.spmv(u, &mut mu);
        let s: f64 = u.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
        s.max(0.0).sqrt()
    }

    /// Pseudo-random but deterministic DOF vector.
    fn pseudo_random(n: usize, seed: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i as f64 + 0.5) * 1.7 + seed).sin() + 0.3 * (i as f64 + seed).cos())
            .collect()
    }

    /// RT0: the error routine must reconstruct exactly the field the assembler
    /// built — Piola map, `|det J|` weight, orientation signs and quadrature
    /// must all agree with `VectorMassIntegrator`.
    #[test]
    fn hdiv_rt0_norm_matches_mass_matrix() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let q = default_quad_order(&space); // 2*0+3 = 3
        let u = pseudo_random(space.n_dofs(), 0.0);

        let err = compute_hdiv_l2_error(&space, &u, &ZERO2);
        let mnorm = mat_norm(&vector_mass(&space, q), &u);
        assert!(
            (err - mnorm).abs() < 1e-11 * mnorm.max(1.0),
            "RT0 ‖u_h‖ mismatch: error-routine {err:.12e} vs mass-matrix {mnorm:.12e}"
        );
    }

    /// RT1 counterpart.
    #[test]
    fn hdiv_rt1_norm_matches_mass_matrix() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 1);
        let q = default_quad_order(&space); // 2*1+3 = 5
        let u = pseudo_random(space.n_dofs(), 0.37);

        let err = compute_hdiv_l2_error(&space, &u, &ZERO2);
        let mnorm = mat_norm(&vector_mass(&space, q), &u);
        assert!(
            (err - mnorm).abs() < 1e-11 * mnorm.max(1.0),
            "RT1 ‖u_h‖ mismatch: error-routine {err:.12e} vs mass-matrix {mnorm:.12e}"
        );
    }

    /// H(curl) ND1 — same consistency check for the covariant Piola branch.
    #[test]
    fn hcurl_nd1_norm_matches_mass_matrix() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let q = default_quad_order(&space);
        let u = pseudo_random(space.n_dofs(), 1.13);

        let err = compute_hdiv_l2_error(&space, &u, &ZERO2);
        let mnorm = mat_norm(&vector_mass(&space, q), &u);
        assert!(
            (err - mnorm).abs() < 1e-11 * mnorm.max(1.0),
            "ND1 ‖u_h‖ mismatch: error-routine {err:.12e} vs mass-matrix {mnorm:.12e}"
        );
    }

    /// Scalar: the error routine matches the assembled mass-matrix norm for
    /// both an H¹ (P2) and an L² (P1) space.
    #[test]
    fn scalar_norm_matches_mass_matrix() {
        let mesh = Mesh::<2>::unit_square_tri(4);

        let h1 = H1Space::new(mesh.clone(), 2);
        let q1 = default_quad_order(&h1);
        let u1 = pseudo_random(h1.n_dofs(), 0.11);
        let m1 = Assembler::assemble_bilinear(&h1, &[&MassIntegrator { rho: 1.0 }], q1);
        let e1 = compute_l2_error_scalar(&h1, &u1, &|_| 0.0);
        let n1 = mat_norm(&m1, &u1);
        assert!(
            (e1 - n1).abs() < 1e-11 * n1.max(1.0),
            "H¹ P2 ‖u_h‖ mismatch: {e1:.12e} vs mass-matrix {n1:.12e}"
        );

        let l2 = L2Space::new(mesh, 1);
        let q2 = default_quad_order(&l2);
        let u2 = pseudo_random(l2.n_dofs(), 0.29);
        let m2 = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], q2);
        let e2 = compute_l2_error_scalar(&l2, &u2, &|_| 0.0);
        let n2 = mat_norm(&m2, &u2);
        assert!(
            (e2 - n2).abs() < 1e-11 * n2.max(1.0),
            "L² P1 ‖u_h‖ mismatch: {e2:.12e} vs mass-matrix {n2:.12e}"
        );
    }

    /// Constant field `(1, 0) ∈ RT0` (RT0 = span{(a x + b, a y + c)}): the L²
    /// projection reproduces it exactly → zero error.
    #[test]
    fn hdiv_rt0_constant_field_is_exact() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        const U: fn(&[f64]) -> Vec<f64> = |_| vec![1.0, 0.0];
        let q = 4u8;

        let m = vector_mass(&space, q);
        let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
        let uh = dense_solve(&m, &b);

        let err = compute_hdiv_l2_error(&space, &uh, &U);
        assert!(err < 1e-12, "constant field error = {err:.3e}, expected 0");
    }

    /// RT1 on triangles: the projection error must equal the independently
    /// assembled Galerkin identity `‖u_h − u_ex‖² = ∫|u_ex|² − uᵀ M u` to
    /// 1e-10 (uses the mass matrix and the RHS only).
    ///
    /// Round-55 note (D492): `u_ex = (x, 2y) ∈ RT1`, so the identity holds
    /// *exactly* — but the right-hand side is a difference of nearly equal
    /// numbers whose rounding noise is one ulp of `5/3` (2.22e-16), five
    /// orders of magnitude above the `‖u_h−u_ex‖² ~ 1e-31` being resolved.
    /// The subtraction must therefore be clamped at its mathematical floor 0
    /// before `sqrt` (see below), otherwise the test is a coin flip on the
    /// last ulp of the mass-matrix accumulation order.  Measured on
    /// `tmp/d492/r55_probe/bin/galerkin.rs`: `5/3 − energy` = `-3 ulp` on
    /// `unit_square_tri(1)`, `-1 ulp` on `unit_square_tri(2)` (this test) and
    /// exactly `0` on `unit_square_tri(4)`.
    #[test]
    fn hdiv_rt1_projection_error_matches_galerkin_identity() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh, 1);
        const U: fn(&[f64]) -> Vec<f64> = |x| vec![x[0], 2.0 * x[1]];
        let q = 5u8;

        let m = vector_mass(&space, q);
        let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
        let uh = dense_solve(&m, &b);

        let err_routine = compute_hdiv_l2_error(&space, &uh, &U);

        // ∫_Ω |u_ex|² = ∫ (x² + 4y²) = 5/3 on the unit square.
        let mut mu = vec![0.0; uh.len()];
        m.spmv(&uh, &mut mu);
        let energy: f64 = uh.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
        let residual_sq = 5.0 / 3.0 - energy;
        // Galerkin orthogonality gives `⟨u_h, u_ex⟩ = ‖u_h‖²`, hence by
        // Cauchy–Schwarz/Bessel `‖u_h‖² = uᵀMu ≤ ‖u_ex‖² = 5/3`: the residual
        // has a **hard upper bound at 0**.  A materially negative value is a
        // real violation of that bound (the mass matrix is not the L² form of
        // the reconstructed field); `-1e-13` is ≈ 450 ulp of 5/3 (spacing
        // 2.22e-16), wide enough never to trip on cancellation noise alone and
        // tight enough to catch a genuine overshoot.
        assert!(
            residual_sq > -1e-13,
            "uᵀ M u exceeds ∫|u_ex|² by {:.3e}",
            -residual_sq
        );
        // Clamp at the mathematical floor `‖u_h − u_ex‖² ≥ 0` before `sqrt`
        // (same idiom as `mat_norm` above): the exact identity holds, but the
        // cancellation above can land the difference one ulp negative and make
        // `sqrt` NaN.  Teeth are unchanged — the bound above pins `energy` from
        // the top and an inconsistent mass matrix moves `err_routine` far above
        // 1e-10 and still fails the assertion below.
        let err_galerkin = residual_sq.max(0.0).sqrt();
        assert!(
            (err_routine - err_galerkin).abs() < 1e-10,
            "RT1 projection error mismatch: routine {err_routine:.14e} vs Galerkin {err_galerkin:.14e}"
        );
    }

    /// Non-representable linear field: the projection error equals the
    /// independently assembled Galerkin identity
    /// `‖u_h − u_ex‖² = ∫|u_ex|² − uᵀ M u` to 1e-10.
    #[test]
    fn hdiv_rt0_projection_error_matches_galerkin_identity() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        // u_ex = (x, 2y): linear but *not* in RT0 (which needs equal x/y
        // linear coefficients).
        const U: fn(&[f64]) -> Vec<f64> = |x| vec![x[0], 2.0 * x[1]];
        let q = 4u8;

        let m = vector_mass(&space, q);
        let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
        let uh = dense_solve(&m, &b);

        let err_routine = compute_hdiv_l2_error(&space, &uh, &U);

        // ∫_Ω |u_ex|² = ∫ (x² + 4y²) = 1/3 + 4/3 = 5/3 on the unit square.
        let mut mu = vec![0.0; uh.len()];
        m.spmv(&uh, &mut mu);
        let energy: f64 = uh.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
        let err_galerkin = (5.0 / 3.0 - energy).sqrt();

        assert!(
            (err_routine - err_galerkin).abs() < 1e-10,
            "projection error mismatch: routine {err_routine:.14e} vs Galerkin {err_galerkin:.14e}"
        );
        assert!(
            (0.05..1.5).contains(&err_routine),
            "expected a non-trivial projection error, got {err_routine:.6e}"
        );
    }

    /// Scalar P0-L² projection of `f = x` on the unit square: the error must
    /// match the analytic Galerkin value `sqrt(1/3 − uᵀ M u)` to 1e-10
    /// (`‖x‖_{L²[0,1]²} = 1/√3`).
    #[test]
    fn scalar_p0_projection_error_is_analytic() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let space = L2Space::new(mesh, 0);
        let q = 4u8;

        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], q);
        let b = Assembler::assemble_linear(&space, &[&ScalarRhs(|x| x[0])], q);
        let uh = dense_solve(&m, &b);

        let err = compute_l2_error_scalar(&space, &uh, &|x| x[0]);

        let mut mu = vec![0.0; uh.len()];
        m.spmv(&uh, &mut mu);
        let energy: f64 = uh.iter().zip(mu.iter()).map(|(a, b)| a * b).sum();
        let expected = (1.0 / 3.0 - energy).sqrt();
        assert!(
            (err - expected).abs() < 1e-10,
            "P0 projection error mismatch: routine {err:.14e} vs analytic {expected:.14e}"
        );
    }

    /// Scalar `f = x²` on the unit square: `‖f‖_{L²} = 1/√5`; the analytic
    /// error can be evaluated in closed form for an affine simplex mesh, but
    /// the simpler check is that a P1-H¹ interpolant of a *linear* field is
    /// exact (zero error).
    #[test]
    fn scalar_h1_linear_interpolant_is_exact() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let f = |x: &[f64]| 2.0 * x[0] - 3.0 * x[1] + 1.0;
        let dofs = space.interpolate(&f).into_vec();
        let err = compute_l2_error_scalar(&space, &dofs, &f);
        assert!(err < 1e-13, "linear interpolant error = {err:.3e}, expected 0");
    }

    /// H(div) constant field on a *quad* mesh (the pex4/pex5 mesh family):
    /// `(1, 0)` and `(0, 1)` are in RT0/RT1 on quads, so the projection is
    /// exact.
    #[test]
    fn hdiv_quad_constant_field_is_exact() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        for order in [0u8, 1u8] {
            let space = HDivSpace::new(mesh.clone(), order);
            const U: fn(&[f64]) -> Vec<f64> = |_| vec![0.0, 1.0];
            let q = 2 * order + 4;

            let m = vector_mass(&space, q);
            let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
            let uh = dense_solve(&m, &b);

            let err = compute_hdiv_l2_error(&space, &uh, &U);
            assert!(
                err < 1e-12,
                "quad RT{order} constant field error = {err:.3e}, expected 0"
            );
        }
    }

    /// The element filter must restrict the integral to the selected elements.
    #[test]
    fn filtered_error_restricts_to_subset() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let n_elems = space.mesh().n_elements() as u32;
        assert!(n_elems >= 4);
        let f = |x: &[f64]| (x[0] * 1.7).sin() * (x[1] * 0.9).cos();
        let dofs = space.interpolate(&f).into_vec();

        let half = n_elems / 2;
        let all = compute_l2_error_scalar(&space, &dofs, &f);
        let sub = compute_l2_error_scalar_filtered(&space, &dofs, &f, Some(&|e| e < half));
        assert!(sub < all, "filtered {sub:.6e} should be < full {all:.6e}");
        assert!(sub > 0.0, "filtered error should be positive, got {sub:.6e}");
    }

    /// `*_filtered(.., None)` must be identical to the unfiltered routine, and
    /// the explicit-order variants must run the same math.
    #[test]
    fn filtered_none_matches_plain_and_order_variant() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let q = default_quad_order(&space);
        let u = pseudo_random(space.n_dofs(), 0.53);
        let exact = |x: &[f64]| vec![x[0], x[1]];

        let plain = compute_hdiv_l2_error(&space, &u, &exact);
        let filtered = compute_hdiv_l2_error_filtered(&space, &u, &exact, None);
        let ordered = compute_hdiv_l2_error_filtered_order(&space, &u, &exact, q, None);
        assert!((plain - filtered).abs() < 1e-15, "filtered(None) differs");
        assert!((plain - ordered).abs() < 1e-15, "explicit-order differs");

        let h1 = H1Space::new(space.mesh().clone(), 1);
        let f = |x: &[f64]| x[0] + x[1];
        let dofs = h1.interpolate(&f).into_vec();
        let s_plain = compute_l2_error_scalar(&h1, &dofs, &f);
        let s_ordered = compute_l2_error_scalar_order(&h1, &dofs, &f, default_quad_order(&h1));
        assert!((s_plain - s_ordered).abs() < 1e-15, "scalar order differs");
    }

    /// MFEM cross-check (WSL MFEM 4.10, harness `tmp/hdiverr/l2err_check.cpp`).
    ///
    /// The global RT DOF numbering differs between MFEM and fem-rs, so a raw
    /// DOF vector cannot be shared.  Both sides therefore build the **exact
    /// L² projection** of the polynomial exact field
    /// `u = (1 + x + 2y + 3xy + x²y², −2 + x − y + xy − 2x²y²)`
    /// onto the same space (`MakeCartesian2D(4,4,QUAD,true)` =
    /// [`Mesh::unit_square_quad`]`(4)`, uniformly refined `nref` times); the
    /// projection — and hence the error — is unique, and every quadrature
    /// involved is exact for these polynomial integrands.
    ///
    /// The MFEM reference values are
    /// `GridFunction::ComputeL2Error(ucoeff)` with MFEM's default rule
    /// `2*fe->GetOrder()+3` (i.e. `2(p+1)+3` for RT_p) and
    /// `GridFunction::ComputeL2Error` of the zero field (= ‖u_ex‖_L²).
    #[test]
    fn mfem_cross_check_quad_rt_l2_error() {
        const U: fn(&[f64]) -> Vec<f64> = |x| {
            let (xi, yi) = (x[0], x[1]);
            let t = xi * xi * yi * yi;
            vec![
                1.0 + xi + 2.0 * yi + 3.0 * xi * yi + t,
                -2.0 + xi - yi + xi * yi - 2.0 * t,
            ]
        };

        // (nref, RT order, DOFs, MFEM err, MFEM norm)
        for (nref, order, ndofs, mfem_err, mfem_norm) in [
            (0usize, 0u8, 40usize, 0.298_904_456_229_935_78, 4.168_999_347_032_277_9),
            (0, 1, 144, 0.004_658_474_953_124_568_1, 4.168_999_347_032_277),
            (2, 0, 544, 0.074_751_944_067_807_469, 4.168_999_347_032_277),
        ] {
            let mut mesh = Mesh::<2>::unit_square_quad(4);
            for _ in 0..nref {
                mesh = fem_mesh::refine_uniform(&mesh);
            }
            let space = HDivSpace::new(mesh, order);
            assert_eq!(space.n_dofs(), ndofs, "RT{order} nref={nref} DOF count");

            // Exact L2 projection (quad order 4 = 3×3 GL is exact here).
            let q = 4u8;
            let m = vector_mass(&space, q);
            let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
            let uh = dense_solve(&m, &b);

            let err = compute_hdiv_l2_error(&space, &uh, &U);
            let norm = compute_hdiv_l2_error(&space, &vec![0.0; ndofs], &U);

            assert!(
                (err - mfem_err).abs() < 1e-10,
                "RT{order} nref={nref}: err {err:.17e} vs MFEM {mfem_err:.17e}"
            );
            assert!(
                (norm - mfem_norm).abs() < 1e-10,
                "RT{order} nref={nref}: norm {norm:.17e} vs MFEM {mfem_norm:.17e}"
            );
        }
    }

    /// On quads the RT0/RT1 spaces contain the affine fields of the form
    /// `(a + b x, c + b y)` (RT_k ⊃ [P_k]² for quads: Q_{k+1,k}×Q_{k,k+1}), so
    /// the L² projection reproduces them and the error is zero.
    #[test]
    fn hdiv_quad_affine_field_is_projection_exact() {
        const U: fn(&[f64]) -> Vec<f64> = |x| vec![1.0 + 2.0 * x[0], -3.0 + 2.0 * x[1]];
        for (nref, label) in [(0usize, "coarse"), (1, "refined")] {
            for order in [0u8, 1u8] {
                let mut mesh = Mesh::<2>::unit_square_quad(4);
                for _ in 0..nref {
                    mesh = fem_mesh::refine_uniform(&mesh);
                }
                let space = HDivSpace::new(mesh, order);
                let q = 2 * order + 4;
                let m = vector_mass(&space, q);
                let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
                let uh = dense_solve(&m, &b);
                let err = compute_hdiv_l2_error(&space, &uh, &U);
                assert!(
                    err < 1e-13,
                    "{label} quad RT{order}: affine field error = {err:.3e}, expected 0"
                );
            }
        }
    }

    // ─── D161: divergence error ─────────────────────────────────────────────

    /// Exact divergence of the D161 cross-check field U below:
    /// `div U = (1 + 3y + 2xy²) + (−1 + x − 4x²y)`.
    const DIV_U: fn(&[f64]) -> f64 =
        |x| (1.0 + 3.0 * x[1] + 2.0 * x[0] * x[1] * x[1]) + (-1.0 + x[0] - 4.0 * x[0] * x[0] * x[1]);

    /// MFEM cross-check (WSL MFEM 4.10, harness
    /// `tmp/d161/probe_div_error.cpp`, output
    /// `tmp/d161/d161_div_err_cpp.txt`).
    ///
    /// Same setup as [`mfem_cross_check_quad_rt_l2_error`]: both sides build
    /// the exact L² projection of the polynomial field
    /// `U = (1 + x + 2y + 3xy + x²y², −2 + x − y + xy − 2x²y²)` on the same
    /// space, then evaluate `GridFunction::ComputeDivError(divU)` with the
    /// default rule `2*fe->GetOrder()+3` (RT: `2(p+1)+3` — exact for these
    /// integrands, so the numbers are quadrature-independent).  The C++
    /// `div_norm` row is `ComputeDivError(divU)` of the zero field, i.e.
    /// `‖div U‖_{L²}` (analytic value `sqrt(31/9) = 1.8559214542766739…`).
    #[test]
    fn mfem_cross_check_quad_rt_div_error() {
        const U: fn(&[f64]) -> Vec<f64> = |x| {
            let (xi, yi) = (x[0], x[1]);
            let t = xi * xi * yi * yi;
            vec![
                1.0 + xi + 2.0 * yi + 3.0 * xi * yi + t,
                -2.0 + xi - yi + xi * yi - 2.0 * t,
            ]
        };

        // (nref, RT order, DOFs, MFEM div_err, MFEM div_norm)
        for (nref, order, ndofs, mfem_div_err, mfem_div_norm) in [
            (0usize, 0u8, 40usize, 2.288_705_473_199_983_41e-1, 1.855_921_454_276_673_99),
            (0, 1, 144, 1.202_813_060_811_759_59e-2, 1.855_921_454_276_673_32),
            (2, 0, 544, 5.765_550_851_273_332_22e-2, 1.855_921_454_276_674_88),
        ] {
            let mut mesh = Mesh::<2>::unit_square_quad(4);
            for _ in 0..nref {
                mesh = fem_mesh::refine_uniform(&mesh);
            }
            let space = HDivSpace::new(mesh, order);
            assert_eq!(space.n_dofs(), ndofs, "RT{order} nref={nref} DOF count");

            // Exact L2 projection (quad order 4 = 3×3 GL is exact here).
            let q = 4u8;
            let m = vector_mass(&space, q);
            let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
            let uh = dense_solve(&m, &b);

            let div_err = compute_div_error(&space, &uh, &DIV_U);
            let div_norm = compute_div_error(&space, &vec![0.0; ndofs], &DIV_U);

            assert!(
                (div_err - mfem_div_err).abs() < 1e-10,
                "RT{order} nref={nref}: div_err {div_err:.17e} vs MFEM {mfem_div_err:.17e}"
            );
            assert!(
                (div_norm - mfem_div_norm).abs() < 1e-10,
                "RT{order} nref={nref}: div_norm {div_norm:.17e} vs MFEM {mfem_div_norm:.17e}"
            );
        }
    }

    /// `(x, 2y)` is in quad RT0 and has `div = 3` — the projection is exact, so
    /// the divergence error against the *nonzero* `div U = 3` must vanish.
    /// (Exercises the exact-divergence evaluation, not just the zero field.)
    #[test]
    fn hdiv_quad_linear_field_div_error_is_zero() {
        const U: fn(&[f64]) -> Vec<f64> = |x| vec![x[0], 2.0 * x[1]];
        const DIV_U: fn(&[f64]) -> f64 = |_| 3.0;

        for order in [0u8, 1u8] {
            let mesh = Mesh::<2>::unit_square_quad(4);
            let space = HDivSpace::new(mesh, order);
            let q = 4u8;
            let m = vector_mass(&space, q);
            let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
            let uh = dense_solve(&m, &b);
            let err = compute_div_error(&space, &uh, &DIV_U);
            assert!(
                err < 1e-12,
                "quad RT{order}: linear field div error = {err:.3e}, expected 0"
            );
        }
    }

    /// Solenoidal-flavoured zero case: the constant field `(1, 0)` is in RT0
    /// (tri and quad) and is divergence-free, so `‖div u_h‖ = 0`.
    #[test]
    fn hdiv_rt0_constant_field_div_error_is_zero() {
        const U: fn(&[f64]) -> Vec<f64> = |_| vec![1.0, 0.0];

        let mesh_q = Mesh::<2>::unit_square_quad(4);
        let mesh_t = Mesh::<2>::unit_square_tri(4);
        for (mesh, label) in [(mesh_q, "quad"), (mesh_t, "tri")] {
            let space = HDivSpace::new(mesh, 0);
            let q = 4u8;
            let m = vector_mass(&space, q);
            let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
            let uh = dense_solve(&m, &b);
            let err = compute_div_error(&space, &uh, &|_| 0.0);
            assert!(
                err < 1e-12,
                "{label} RT0: constant field div error = {err:.3e}, expected 0"
            );
        }
    }

    /// The div error of a nonzero field must be positive (sanity: the
    /// reconstruction actually diverges), and the filtered/order variants must
    /// agree with the plain call.
    #[test]
    fn div_error_variants_agree_and_are_positive() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let space = HDivSpace::new(mesh, 1);
        let q = default_quad_order(&space);
        let u = pseudo_random(space.n_dofs(), 0.71);
        let exact_div = |x: &[f64]| x[0] + x[1];

        let plain = compute_div_error(&space, &u, &exact_div);
        assert!(plain > 1e-3, "expected a non-trivial div error, got {plain:.6e}");

        let filtered = compute_div_error_filtered(&space, &u, &exact_div, None);
        let ordered = compute_div_error_filtered_order(&space, &u, &exact_div, q, None);
        assert!((plain - filtered).abs() < 1e-15, "filtered(None) differs");
        assert!((plain - ordered).abs() < 1e-15, "explicit-order differs");
    }

    /// The composite H(div) norm equals `sqrt(L2² + div²)` of the parts, and a
    /// projection-exact field gives zero (MFEM `ComputeHDivError`).
    #[test]
    fn hdiv_full_error_composition() {
        const U: fn(&[f64]) -> Vec<f64> = |x| vec![1.0 + 2.0 * x[0], -3.0 + 2.0 * x[1]];
        const DIV_U: fn(&[f64]) -> f64 = |_| 4.0; // d/dx(1+2x) + d/dy(−3+2y) = 2 + 2

        let mesh = Mesh::<2>::unit_square_quad(4);
        let space = HDivSpace::new(mesh, 0);
        let q = 4u8;

        let m = vector_mass(&space, q);
        let b = VectorAssembler::assemble_linear(&space, &[&VectorRhs(U)], q);
        let uh = dense_solve(&m, &b);

        let full = compute_hdiv_full_error(&space, &uh, &U, &DIV_U);
        let l2 = compute_hdiv_l2_error(&space, &uh, &U);
        let dv = compute_div_error(&space, &uh, &DIV_U);
        let composed = (l2 * l2 + dv * dv).sqrt();
        assert!(
            (full - composed).abs() < 1e-14,
            "full {full:.17e} vs composed {composed:.17e}"
        );
        assert!(full < 1e-12, "projection-exact field: full H(div) error {full:.3e}");
    }
}
