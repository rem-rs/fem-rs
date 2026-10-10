//! DG elasticity assemblers with full stress-based SIP face terms.
//!
//! Provides the correct DG-SIP linear-elasticity operator:
//!   Volume: ∫ 2μ·ε(u):ε(v) + λ·div(u)·div(v) dx
//!   Interior faces: stress-based SIP (consistency + symmetry + penalty)
//!   Boundary faces: stress-based SIP for weak Dirichlet

use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_mesh::transformation::element_jacobian_at;
use fem_space::fe_space::FESpace;

use super::dg_base::{
    build_face_elem_map, face_point_geom, face_point_geom_3d_face, face_type_of, ref_elem_face,
    FaceGeom, xform_grads,
};
use crate::interior_faces::InteriorFaceList;

/// DG elasticity assembler with full stress-based SIP.
pub struct DgElasticityAssembler;

impl DgElasticityAssembler {
    /// Full-coupling DG-SIP linear-elasticity with stress-based face terms.
    ///
    /// DOF layout: component-major, size = dim * n_scalar.
    ///
    /// Volume: ∫ 2μ·ε(u):ε(v) + λ·div(u)·div(v) dx
    ///   = ∫ μ·(∇u:∇v + ∇u:∇v^T) + λ·I·div(u)·div(v) dx
    ///
    /// Faces: MFEM's `DGElasticityIntegrator`
    ///   `elmat := -elmat + alpha·elmatᵀ + jmat` where the consistency and
    ///   symmetry blocks are the block-diagonal `−A + alpha·Aᵀ` of the
    ///   averaged stress flux and `jmat` is MFEM's penalty
    ///   `jmatcoef = kappa·(nor·nor)·wLM` (see
    ///   [`assemble_interior_face_stress`]).
    ///
    /// `dirichlet_attrs` = list of boundary attributes where Dirichlet BCs are
    /// enforced weakly (the parameter `dir_bdr` in MFEM ex17). Pass an empty
    /// slice for pure natural BC.
    ///
    /// # Quadrature (D805-1)
    ///
    /// `quad_order` is the **face** rule.  MFEM's `DGElasticityIntegrator`
    /// default is `2·max(el1.GetOrder(), el2.GetOrder())` on the face geometry
    /// (`fem/bilininteg.cpp:4122`), which is what callers pass — ex17 passes
    /// `2·order`.
    ///
    /// The **volume** rule is MFEM's `ElasticityIntegrator` default,
    /// `2·Trans.OrderGrad(&el)` (`fem/bilininteg.cpp:3247`; see
    /// [`mfem_elasticity_volume_rule`]).  It is derived internally from the
    /// mesh's geometry order and the element order, so it is *not* taken from
    /// `quad_order`.  On a straight (order-1 geometry) mesh the two coincide:
    /// MFEM's `2·OrderGrad` and the old `quad_order = 2p` select the same
    /// Gauss-Legendre rule at every order (`n = p+1` points per axis either
    /// way) — on a curved mesh they differ and MFEM's own rule is the right one.
    pub fn assemble_sip_elasticity<S: FESpace + Sync>(
        space: &S,
        ifl: &InteriorFaceList,
        lambda_elem: &[f64],
        mu_elem: &[f64],
        kappa: f64,
        alpha: f64,
        dim: usize,
        quad_order: u8,
        dirichlet_attrs: &[i32],
    ) -> CsrMatrix<f64> {
        assert!(dim == 2 || dim == 3);
        let mesh = space.mesh();
        let n_elem = mesh.n_elements() as usize;
        let n_scalar = space.n_dofs();
        let n_total = dim * n_scalar;
        assert_eq!(lambda_elem.len(), n_elem);
        assert_eq!(mu_elem.len(), n_elem);

        let mut coo = CooMatrix::<f64>::new(n_total, n_total);

        // ── 1. Volume ──────────────────────────────────────────────────
        assemble_volume(&mut coo, space, lambda_elem, mu_elem, dim);

        // ── 2. Interior face stress SIP ────────────────────────────────
        let dirichlet_set: std::collections::HashSet<i32> =
            dirichlet_attrs.iter().copied().collect();
        let face_to_elem = build_face_elem_map(mesh, dim);

        for iface in &ifl.faces {
            assemble_interior_face_stress(
                &mut coo,
                mesh,
                space,
                iface.elem_left,
                iface.elem_right,
                &iface.face_nodes,
                lambda_elem,
                mu_elem,
                kappa,
                alpha,
                dim,
                quad_order,
            );
        }

        // ── 3. Boundary face stress SIP (Dirichlet, tagged attributes) ─
        for f in mesh.face_iter() {
            let tag = mesh.face_tag(f);
            if tag == 0 || !dirichlet_set.contains(&tag) {
                continue;
            }
            // D817-1: interior-coincident bdr entries get zero assembly
            // (MFEM `GetBdrFaceTransformations` mesh.cpp:1312 INVALID).
            if mesh.bdr_face_true_interior(f) {
                continue;
            }
            let elem = match face_to_elem.get(&f) {
                Some(&e) => e,
                None => continue,
            };
            assemble_boundary_face_stress(
                &mut coo,
                mesh,
                space,
                f,
                elem,
                lambda_elem,
                mu_elem,
                kappa,
                alpha,
                dim,
                quad_order,
            );
        }

        coo.into_csr()
    }
}

// ─── Volume term: full linear elasticity kernel ────────────────────────────
//
//   K[(a,i),(b,j)] += μ·δᵢⱼ·∇φ_a·∇φ_b + (λ+μ)·∂ᵢφ_a·∂ⱼφ_b
//
//   = μ·(d_i·∇φ_a · d_j·∇φ_b + δᵢⱼ·∇φ_a·∇φ_b) + λ·∂ᵢφ_a·∂ⱼφ_b
//
// In the code we compute block-diagonal μ·∇φ_a·∇φ_b and then add
// μ·∂ⱼφ_a·∂ᵢφ_b (cross) + λ·∂ᵢφ_a·∂ⱼφ_b (div-div), matching
// the decomposition used in `assemble_vol_coupling_per_elem`.

/// MFEM's default quadrature order for `ElasticityIntegrator`'s volume term:
/// `2 * Trans.OrderGrad(&el)` (`fem/bilininteg.cpp:3249-3252`).
///
/// `ElasticityIntegrator` does **not** override `GetDefaultIntegrationRule`
/// (which is what would give the generic `p + p + Trans.OrderW()` rule that
/// `VectorDivergenceIntegrator` and friends use): its own
/// `AssembleElementMatrix` sees `GetIntegrationRule(el, Trans) == NULL` and
/// falls back to `2*Trans.OrderGrad(&el)`, with
/// `IsoparametricTransformation::OrderGrad` (`fem/eltrans.cpp:509-529`)
/// returning
///   * `(g-1)·(dim-1) + (p-1)` for a `Pk` (simplex) geometry map,
///   * `g·(dim-1) + (p-1)`     for a `Qk` (tensor) geometry map,
/// where `g` is the mesh's geometry order and `p` the element's own order.
///
/// On the D805 curved fixture (`g = 3`, `Qk`, `dim = 2`, `p = 1`) that is order
/// 6 → the 4×4 Gauss-Legendre rule, which the probe confirms is exactly what
/// a plain `ElasticityIntegrator` uses there (`[ORDER]` rows of
/// `tmp/d805r76/cpp_truth_default.txt`).
// MFEM: ElasticityIntegrator::AssembleElementMatrix + IsoparametricTransformation::OrderGrad
pub fn mfem_elasticity_volume_rule(
    geom_order: u8,
    elem_order: u8,
    et: ElementType,
    dim: usize,
) -> u8 {
    let g = geom_order as i32;
    let p = elem_order as i32;
    let d = dim as i32;
    let order = match et {
        ElementType::Tri3 | ElementType::Tet4 => (g - 1) * (d - 1) + (p - 1),
        _ => g * (d - 1) + (p - 1),
    };
    (2 * order).clamp(0, u8::MAX as i32) as u8
}

fn assemble_volume<S: FESpace>(
    coo: &mut CooMatrix<f64>,
    space: &S,
    lambda_elem: &[f64],
    mu_elem: &[f64],
    dim: usize,
) {
    let mesh = space.mesh();
    let order = space.order();
    let geom_order = mesh.geom_order();

    for e in mesh.elem_iter() {
        let ei = e as usize;
        let lam = lambda_elem[ei];
        let mu = mu_elem[ei];
        if lam == 0.0 && mu == 0.0 {
            continue;
        }

        let et = mesh.element_type(e);
        let elem_order = space.element_order(e);
        let re: Box<dyn ReferenceElement> =
            fem_space::ref_elem::field_element_for_space(space, et, order);
        let n_l = re.n_dofs();
        // MFEM's `ElasticityIntegrator` volume rule (see the helper above).
        let quad_order = mfem_elasticity_volume_rule(geom_order, elem_order, et, dim);
        let q = re.quadrature(quad_order);

        let dofs: Vec<usize> =
            space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let mut gref = vec![0.0_f64; n_l * dim];
        let mut gphys = vec![0.0_f64; n_l * dim];

        for (qi, xi) in q.points.iter().enumerate() {
            // D799-3: MFEM's `ElasticityIntegrator::AssembleElementMatrix`
            // evaluates the element's **isoparametric** transformation — the
            // order-`g` curved map on a curved mesh, the P1/bilinear map on a
            // straight one (`Trans.Weight()` and `Trans.AdjugateJacobian()`,
            // bilininteg.cpp:1615).  The pre-fix `simplex_jac` used a single
            // centroid Jacobian per element (exact only for affine tets/tris and
            // parallelograms).
            let (jac, _xp) = element_jacobian_at(mesh, e, xi, dim);
            // D696 batch 4: SIGNED — `ip.weight * T.Weight()` (MFEM
            // nonlininteg class); bitwise |det| on valid meshes.
            let det_j = jac.determinant();
            // Degeneracy guard (magnitude test only, D696 batch 4 precedent):
            // skip collapsed cells, never taken on a valid mesh.
            if det_j.abs() < 1e-30 {
                continue;
            }
            let jit = jac
                .try_inverse()
                .unwrap_or_else(|| {
                    eprintln!("  warning: degenerate elasticity element {e}");
                    nalgebra::DMatrix::identity(dim, dim)
                })
                .transpose();
            let w = q.weights[qi] * det_j;
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_l, dim);

            // Full elasticity kernel at this QP
            for a in 0..n_l {
                let ga = |d: usize| -> f64 { gphys[a * dim + d] };
                let ga_dot_gb = |b: usize| -> f64 {
                    (0..dim).map(|d| ga(d) * gphys[b * dim + d]).sum()
                };
                for b in 0..n_l {
                    let nabla_ab = ga_dot_gb(b);
                    for i in 0..dim {
                        // Block-diagonal: μ·δᵢⱼ·∇φ_a·∇φ_b  (i=j only)
                        let row = dofs[a] * dim + i;
                        // i == j term
                        coo.add(row, dofs[b] * dim + i, w * mu * nabla_ab);

                        for j in 0..dim {
                            // Cross + div-div: μ·∂ⱼφ_a·∂ᵢφ_b + λ·∂ᵢφ_a·∂ⱼφ_b
                            let val = mu * gphys[a * dim + j] * gphys[b * dim + i]
                                + lam * gphys[a * dim + i] * gphys[b * dim + j];
                            let col = dofs[b] * dim + j;
                            coo.add(row, col, w * val);
                        }
                    }
                }
            }
        }
    }
}

// ─── Stress-flux helper ────────────────────────────────────────────────────
//
// Compute (σ(φ·e_l)·n)_i at a quadrature point:
//
//   σ(φ·e_l) = λ·∂ₗφ·I + μ·(e_l⊗∇φ + ∇φ⊗e_l)
//   (σ(φ·e_l)·n)_i = λ·∂ₗφ·nᵢ + μ·(∂ᵢφ·nₗ + δᵢₗ·∇φ·n)
//
// Returns `flux[comp]` = (σ(φ·e_l)·n)_comp  for comp = 0..dim-1.
//
// (σ(φ·e_l)·n)_i = λ·∂ₗφ·nᵢ + μ·(∂ᵢφ·nₗ + δᵢₗ·∇φ·n)
//
// where grad[·] = ∂·φ (physical gradient of basis function φ).
fn stress_flux(
    lam: f64,
    mu: f64,
    grad_a: &[f64], // grad[a * dim .. a * dim + dim]
    normal: &[f64],
    l: usize, // component of the basis function
    dim: usize,
) -> Vec<f64> {
    let dl_phi = grad_a[l]; // ∂φ/∂xₗ
    let gdotn: f64 = (0..dim).map(|k| grad_a[k] * normal[k]).sum();
    let mut result = vec![0.0_f64; dim];
    for i in 0..dim {
        let di_phi = grad_a[i]; // ∂φ/∂xᵢ
        let d_il = if i == l { 1.0 } else { 0.0 };
        result[i] = lam * dl_phi * normal[i] + mu * (di_phi * normal[l] + d_il * gdotn);
    }
    result
}
// So: ∂ₗφ = grad[l*dim + l] when indices match... no!
// grad is organized as [basis × component], so:
//   grad[a][d] = ∂φ_a/∂x_d
// So for a fixed basis φ:
//   ∂ₗφ = grad[l]  where l is the gradient component index
//   ∂ᵢφ = grad[i]
//
// grad[a*dim + d] = ∂φ_a/∂x_d
// For basis function 'a' with component 'l': grad[a*dim + l] = ∂φ_a/∂x_l
// For basis function 'a' with component 'i': grad[a*dim + i] = ∂φ_a/∂x_i

// Actually I realize the issue: in the stress flux formula:
//   (σ(φ·e_l)·n)_i = λ·∂ₗφ·nᵢ + μ·(∂ᵢφ·nₗ + δᵢₗ·∇φ·n)
//
// grad = ∇φ, a vector of length dim.
// ∂ₗφ = grad[l]  (the l-th component of the gradient of φ)
// ∂ᵢφ = grad[i]  (the i-th component of the gradient of φ)
// ∇φ·n = Σₖ grad[k]·nₖ

// So for any given basis function φ (at grad index a):
// ∂ₗφ = grad[a*dim + l]
// ∂ᵢφ = grad[a*dim + i]
// ∇φ·n = Σₖ grad[a*dim + k]·nₖ

// Let me rewrite stress_flux correctly:

#[allow(dead_code)]

// ─── Interior face: stress-based SIP ───────────────────────────────────────
//
// For a face between left element (el) and right element (er):
//
//   K += −∫ {σ(u)·n}·[[v]] − α∫ {σ(v)·n}·[[u]] + ∫ (κ/h)[[u]]·[[v]] ds
//
// where {w} = ½(w_L + w_R), [[w]] = w_L − w_R, n = n_L (outward from left).
//
// This gives four blocks (LL, LR, RL, RR), each coupling components i,j
// via the stress tensor.

#[allow(clippy::too_many_arguments)]
fn assemble_interior_face_stress<S: FESpace>(
    coo: &mut CooMatrix<f64>,
    mesh: &S::Mesh,
    space: &S,
    el: u32,
    er: u32,
    face_nodes: &[u32],
    lambda_elem: &[f64],
    mu_elem: &[f64],
    kappa: f64,
    alpha: f64,
    dim: usize,
    quad_order: u8,
) {
    let order = space.order();
    // D814-1: face rule by node count — a hexahedron's quad faces take MFEM's
    // `[0,1]²` tensor rule (`DGElasticityIntegrator`'s inline default
    // `IntRules.Get(face_geom, 2·max(o₁,o₂))`), not the triangular one.
    let face_re = ref_elem_face(face_type_of(face_nodes), order);
    let q_face = face_re.quadrature(quad_order);

    let et_l = mesh.element_type(el);
    let re_l = fem_space::ref_elem::field_element_for_space(space, et_l, order);
    let et_r = mesh.element_type(er);
    let re_r = fem_space::ref_elem::field_element_for_space(space, et_r, order);
    let n_l = re_l.n_dofs();
    let n_r = re_r.n_dofs();

    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    let lam_l = lambda_elem[el as usize];
    let mu_l = mu_elem[el as usize];
    let lam_r = lambda_elem[er as usize];
    let mu_r = mu_elem[er as usize];

    // D1280 route 1: accumulate MFEM's pre-finalization pair over ONE combined
    // (Elem1 ∪ Elem2) dof block — `pre` = the consistency part
    // <{σ(u)·n}, [v]> (MFEM's `elmat` before the finalization) and
    // `jmat_c` = the penalty (`jmat`) — because MFEM's closing transform
    // `elmat := -elmat + α·elmatᵀ + jmat` (bilininteg.cpp) writes the pair
    // (i,j), (j,i) from the SAME `mij = jmat(i,j)`, which makes the final
    // elmat bitwise-symmetric *by construction* (with α = -1, `-aji - aij` and
    // `-aij - aji` are equal by fp addition commutativity).  The four separate
    // per-block accumulations this replaces folded α into each block, so
    // transposed positions were computed through different operand orders and
    // differed by 1 ulp — the residual Symmetry 4.4e-16 in the matrix stats.
    // Combined index: side L = a·dim+i (a < n_l), side R = n_l·dim + b·dim+j.
    let nl_d = n_l * dim;
    let nr_d = n_r * dim;
    let n_comb = nl_d + nr_d;
    let mut pre = vec![0.0_f64; n_comb * n_comb];
    let mut jmat_c = vec![0.0_f64; n_comb * n_comb];

    let mut phi_l = vec![0.0_f64; n_l];
    let mut phi_r = vec![0.0_f64; n_r];
    let mut gref_l = vec![0.0_f64; n_l * dim];
    let mut gref_r = vec![0.0_f64; n_r * dim];
    let mut gphys_l = vec![0.0_f64; n_l * dim];
    let mut gphys_r = vec![0.0_f64; n_r * dim];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        // D799-3: the isoparametric face route (`face_point_geom`, D795-1) —
        // MFEM `DGElasticityIntegrator::AssembleFaceMatrix` composes the face
        // transformation through `Elem1`/`Elem2`'s order-`g` maps
        // (`Trans.SetAllIntPoints` → `Loc1`/`Loc2`, `nor = CalcOrtho(
        // Trans.Jacobian())`, `Trans.Elem1->Weight()`), and its per-QP algorithm
        // (bilininteg.cpp:4012 `AssembleBlock` + :4055) is, with
        // `w = ip.weight/2`, `w1 = w/Trans.Elem1->Weight()`,
        // `nL = w1·λ·nor`, `nM = w1·μ·nor`, `dshape_ps = dshape·adjJ`:
        //   elmat(φ_a·e_i, φ_b·e_j) += shape_a · [ dshape_b·nL_i
        //                                            + δᵢⱼ·dshape_b·nM
        //                                            + dshape_b_j·nM_i ]
        //   jmatcoef = kappa·(nor·nor)·(wL1+2wM1+wL2+2wM2)
        // and then `elmat := -elmat + alpha·elmatᵀ + jmat`.  The factors of
        // `Weight` cancel exactly as in `dg.rs`: `dshape_ps·nM` is
        // `det(J)·w1·μ·(J⁻ᵀ∇φ)·nor = (ip.weight/2)·μ·∇ₓφ·nor`, i.e. the
        // averaged stress flux carried below by `w_f = ipw·|nor|` together with
        // the unit normal; `jmatcoef` is
        // `kappa·|nor|²·ip.weight/2·( (λ+2μ)/det₁ + (λ+2μ)/det₂ )` — MFEM's
        // penalty, which the pre-round-76 `w_f·pen` term was **not** (see the
        // `jmatcoef` block below).
        // D814-1: 3-D faces (triangular or quadrilateral) compose through the
        // same `face_point_geom_3d_face` dispatcher as the other DG drivers.
        let g1 = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, el, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, el, face_nodes, [xi_f[0], xi_f[1]]))
        };
        let g2 = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, er, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, er, face_nodes, [xi_f[0], xi_f[1]]))
        };
        // Face size from the isoparametric face (`nor`'s magnitude = |dX/dξ|;
        // MFEM's `sqrt(nor·nor)`), and the unit normal MFEM's `nor` reduces to.
        let h_f = g1.nor().iter().take(dim).map(|n| n * n).sum::<f64>().sqrt().max(1e-30);
        let normal: Vec<f64> = (0..dim).map(|k| g1.nor()[k] / h_f).collect();
        let w_f = q_face.weights[qi] * h_f;

        re_l.eval_basis(g1.eip(), &mut phi_l);
        re_r.eval_basis(g2.eip(), &mut phi_r);
        re_l.eval_grad_basis(g1.eip(), &mut gref_l);
        re_r.eval_grad_basis(g2.eip(), &mut gref_r);
        xform_grads(g1.jit(), &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(g2.jit(), &gref_r, &mut gphys_r, n_r, dim);

        // D805-1: MFEM's penalty (the `jmat` of `AssembleBlock`), NOT an
        // averaged Lame constant:
        //
        //   w      = ip.weight/2                       (interior face)
        //   wLₖ    = (w / Weightₖ) · λₖ ,  wMₖ = (w / Weightₖ) · μₖ
        //   wLM    = (wL₁ + 2·wM₁) + (wL₂ + 2·wM₂)
        //   jmatcoef = kappa · (nor·nor) · wLM
        //
        // i.e. `kappa·|nor|²·ipw·(1/2)·((λ₁+2μ₁)/W₁ + (λ₂+2μ₂)/W₂)`: the **bare**
        // face quadrature weight `ipw` enters once (through `w = ip.weight/2`),
        // the two elements' `(λ+2μ)/Weight()` are **summed** (MFEM does not
        // average them), and the face measure is carried by `nor·nor` — not by
        // a `1/|nor|` on the coefficient.
        //
        // The pre-fix code instead used `w_f·pen` with `w_f = ipw·|nor|` and
        // `pen = kappa·(λ+2μ)_avg/|nor|`, a different number on any element
        // whose `|nor|²/Weight` ratio is not 1 (41.5% on the D805 fixture; it
        // coincides on a uniform straight square mesh, where `|nor|² = W`).
        //
        // `nor·nor` is formed from the unnormalised `g1.nor` directly (MFEM's
        // `nor*nor`), not from `h_f*h_f`, to stay bit-faithful.
        let nor_dot_nor = g1.nor().iter().take(dim).map(|n| n * n).sum::<f64>();
        let jmatcoef = kappa * nor_dot_nor * q_face.weights[qi] * 0.5
            * ((lam_l + 2.0 * mu_l) / g1.det_j() + (lam_r + 2.0 * mu_r) / g2.det_j());

        // Precompute stress flux for each basis×component on both sides
        // sigma_n_L[a][l][i] = (σ_L(φ_a·e_l)·n)_i
        let mut snl = vec![vec![vec![0.0_f64; dim]; dim]; n_l];
        for a in 0..n_l {
            let ga = &gphys_l[a * dim..(a + 1) * dim];
            for l in 0..dim {
                snl[a][l] = stress_flux(lam_l, mu_l, ga, &normal, l, dim);
            }
        }
        let mut snr = vec![vec![vec![0.0_f64; dim]; dim]; n_r];
        for a in 0..n_r {
            let ga = &gphys_r[a * dim..(a + 1) * dim];
            for l in 0..dim {
                snr[a][l] = stress_flux(lam_r, mu_r, ga, &normal, l, dim);
            }
        }

        // Consistency (pre) and penalty (jmat_c) parts per block, with the
        // block signs of MFEM's initial `elmat = <{σ(u)·n}, [v]>` +
        // `shape2.Neg()` (the jump's -v₂/-u₂ enter through the negated
        // side-2 shapes; α is NOT folded here — the finalization below owns
        // it).  The transposed pair of the final K is formed by the
        // finalization from the SAME two pre entries, which is what keeps it
        // bitwise-symmetric.
        //
        // K_LL (test left a,i; trial left b,j):
        //   pre    += w_f·(+½·σ_L(φ_b·e_j)·n)_i · φ_L_a
        //   jmat_c += jmatcoef·φ_L_a·φ_L_b·δᵢⱼ
        for a in 0..n_l {
            for i in 0..dim {
                let r = a * dim + i;
                for b in 0..n_l {
                    for j in 0..dim {
                        let c = b * dim + j;
                        pre[r * n_comb + c] += w_f * (0.5 * snl[b][j][i] * phi_l[a]);
                        jmat_c[r * n_comb + c] +=
                            jmatcoef * phi_l[a] * phi_l[b] * if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
        }

        // K_LR (test left a,i; trial right b,j): MFEM (1,2) block — row shape
        // positive (jump +v₁), trial flux from Elem2:
        //   pre    += w_f·(+½·σ_R(φ_b·e_j)·n)_i · φ_L_a
        //   jmat_c += -jmatcoef·φ_L_a·φ_R_b·δᵢⱼ   (negated col shape)
        for a in 0..n_l {
            for i in 0..dim {
                let r = a * dim + i;
                for b in 0..n_r {
                    for j in 0..dim {
                        let c = nl_d + b * dim + j;
                        pre[r * n_comb + c] += w_f * (0.5 * snr[b][j][i] * phi_l[a]);
                        jmat_c[r * n_comb + c] +=
                            -jmatcoef * phi_l[a] * phi_r[b] * if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
        }

        // K_RL (test right a,i; trial left b,j): MFEM (2,1) block — row shape
        // negated (jump -v₂), trial flux from Elem1:
        //   pre    += w_f·(-½·σ_L(φ_b·e_j)·n)_i · φ_R_a
        //   jmat_c += -jmatcoef·φ_R_a·φ_L_b·δᵢⱼ
        for a in 0..n_r {
            for i in 0..dim {
                let r = a * dim + i;
                for b in 0..n_l {
                    for j in 0..dim {
                        let c = b * dim + j;
                        pre[(nl_d + r) * n_comb + c] += w_f * (-0.5 * snl[b][j][i] * phi_r[a]);
                        jmat_c[(nl_d + r) * n_comb + c] +=
                            -jmatcoef * phi_r[a] * phi_l[b] * if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
        }

        // K_RR (test right a,i; trial right b,j): MFEM (2,2) block — both
        // shapes negated (elmat: one negation through row_shape; jmat: two
        // negations cancel):
        //   pre    += w_f·(-½·σ_R(φ_b·e_j)·n)_i · φ_R_a
        //   jmat_c += +jmatcoef·φ_R_a·φ_R_b·δᵢⱼ
        for a in 0..n_r {
            for i in 0..dim {
                let r = a * dim + i;
                for b in 0..n_r {
                    for j in 0..dim {
                        let c = b * dim + j;
                        pre[(nl_d + r) * n_comb + (nl_d + c)] +=
                            w_f * (-0.5 * snr[b][j][i] * phi_r[a]);
                        jmat_c[(nl_d + r) * n_comb + (nl_d + c)] +=
                            jmatcoef * phi_r[a] * phi_r[b] * if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
        }
    }

    // MFEM's closing transform, bilininteg.cpp `DGElasticityIntegrator`:
    //
    //   for i, for j < i:  mij = jmat(i,j)
    //                      elmat(i,j) = α·elmat(j,i) - elmat(i,j) + mij
    //                      elmat(j,i) = α·elmat(i,j) - elmat(j,i) + mij
    //   elmat(i,i) = (α-1)·elmat(i,i) + jmat(i,i)
    //
    // ported verbatim (shared `mij`, symmetric-by-construction result).
    let mut comb = vec![0.0_f64; n_comb * n_comb];
    for r in 0..n_comb {
        for c in 0..r {
            let arc = pre[r * n_comb + c];
            let acr = pre[c * n_comb + r];
            let m = jmat_c[r * n_comb + c];
            comb[r * n_comb + c] = alpha * acr - arc + m;
            comb[c * n_comb + r] = alpha * arc - acr + m;
        }
        comb[r * n_comb + r] = (alpha - 1.0) * pre[r * n_comb + r] + jmat_c[r * n_comb + r];
    }

    // D1280 route 1: scatter the finalized combined elmat with MFEM's
    // `AddSubMatrix(vdofs, vdofs, elmat, skip_zeros=1)` semantics —
    // mutual-zero positions are never allocated; one-sided zeros are kept.
    let global = |idx: usize, dofs: &[usize]| -> usize { dofs[idx / dim] * dim + idx % dim };
    for r in 0..n_comb {
        let gd_r = if r < nl_d { global(r, &dofs_l) } else { global(r - nl_d, &dofs_r) };
        for c in 0..n_comb {
            let val = comb[r * n_comb + c];
            if val == 0.0 && comb[c * n_comb + r] == 0.0 {
                continue;
            }
            let gd_c = if c < nl_d { global(c, &dofs_l) } else { global(c - nl_d, &dofs_r) };
            coo.add(gd_r, gd_c, val);
        }
    }
}

// ─── Boundary face: stress-based SIP (weak Dirichlet) ──────────────────────
//
// On a Dirichlet boundary, the formulation is the same as the interior face
// but with the "right" element set to zero (no neighbour):
//
//   {σ(u)·n}·v = σ(u)·n·v     (no average needed)
//   −α·{σ(v)·n}·u = −α·σ(v)·n·u
//   (κ/h)·u·v                     (no jump)
//
// So:
//   K_bdr[(a,i),(b,j)] += −σ_L(φ_b·e_j)·n)_i·φ_L_a
//                          −α·σ_L(φ_a·e_i)·n)_j·φ_L_b
//                          + (κ/h)·φ_L_a·φ_L_b·δᵢⱼ
//
// (no ½ factor since there's no average)
// (the matrix contribution is the "stiffness" part; the RHS from Dirichlet data
//  is assembled separately in the example)

#[allow(clippy::too_many_arguments)]
fn assemble_boundary_face_stress<S: FESpace>(
    coo: &mut CooMatrix<f64>,
    mesh: &S::Mesh,
    space: &S,
    face: u32,
    elem: u32,
    lambda_elem: &[f64],
    mu_elem: &[f64],
    kappa: f64,
    alpha: f64,
    dim: usize,
    quad_order: u8,
) {
    let order = space.order();
    let face_nodes = mesh.face_nodes(face);

    let et = mesh.element_type(elem);
    let re = fem_space::ref_elem::field_element_for_space(space, et, order);
    let n = re.n_dofs();
    let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

    let lam = lambda_elem[elem as usize];
    let mu = mu_elem[elem as usize];

    // D814-1: face rule by node count — a hexahedron's quad faces take MFEM's
    // `[0,1]²` tensor rule, not the triangular one.
    let face_re = ref_elem_face(face_type_of(&face_nodes), order);
    let q_face = face_re.quadrature(quad_order);

    let n_dim = n * dim;
    let mut pre = vec![0.0_f64; n_dim * n_dim];
    let mut jmat_c = vec![0.0_f64; n_dim * n_dim];
    let mut phi = vec![0.0_f64; n];
    let mut gref = vec![0.0_f64; n * dim];
    let mut gphys = vec![0.0_f64; n * dim];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        // D799-3: same isoparametric face route as the interior term.  MFEM's
        // `DGElasticityIntegrator` boundary case is the `ndofs2 == 0` arm of
        // `AssembleFaceMatrix`: `w = ip.weight` (no ½), `w1 = w/Weight₁`,
        // `wLM = wL1 + 2wM1`, `jmatcoef = kappa·(nor·nor)·wLM`.  The
        // consistency/symmetry terms carry the isoparametric face measure
        // `w_f = ipw·|nor|` with the unit normal (MFEM's
        // `nL₁ = (ipw/W₁)·λ·nor` against `dshape_ps = W·∇ₓφ` — the `W` cancels
        // and the flux is the same number), and the penalty below is MFEM's
        // `jmatcoef`, not the pre-round-76 `w_f·pen`.
        //
        // D814-1: 3-D faces compose through `face_point_geom_3d_face`.
        let g1 = if dim == 2 {
            FaceGeom::Face2(face_point_geom(mesh, elem, face_nodes[0], face_nodes[1], xi_f[0]))
        } else {
            FaceGeom::Face3(face_point_geom_3d_face(mesh, elem, &face_nodes, [xi_f[0], xi_f[1]]))
        };
        let h_f = g1.nor().iter().take(dim).map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let normal: Vec<f64> = (0..dim).map(|k| g1.nor()[k] / h_f).collect();
        let w_f = q_face.weights[qi] * h_f;

        re.eval_basis(g1.eip(), &mut phi);
        re.eval_grad_basis(g1.eip(), &mut gref);
        xform_grads(g1.jit(), &gref, &mut gphys, n, dim);

        // D805-1: MFEM's boundary penalty `jmatcoef = kappa·(nor·nor)·wLM` with
        // `w = ip.weight` (the boundary arm does NOT halve `w`), so
        // `kappa·|nor|²·ipw·(λ+2μ)/Weight₁`.  The pre-fix code used
        // `w_f·pen = ipw·|nor|·kappa·(λ+2μ)/|nor| = ipw·kappa·(λ+2μ)`, which
        // drops MFEM's `|nor|²/Weight` factor.
        let nor_dot_nor = g1.nor().iter().take(dim).map(|v| v * v).sum::<f64>();
        let jmatcoef = kappa * nor_dot_nor * q_face.weights[qi] * (lam + 2.0 * mu) / g1.det_j();

        // Precompute stress flux for each basis×component
        let mut sn = vec![vec![vec![0.0_f64; dim]; dim]; n];
        for a in 0..n {
            let ga = &gphys[a * dim..(a + 1) * dim];
            for l in 0..dim {
                sn[a][l] = stress_flux(lam, mu, ga, &normal, l, dim);
            }
        }

        // K_bdr[(a,i),(b,j)] = −σ(φ_b·e_j)·n)_i·φ_a
        //                      −α·σ(φ_a·e_i)·n)_j·φ_b
        //                      + jmatcoef·φ_a·φ_b·δᵢⱼ
        // D1280 route 1: split into the consistency part `pre` (+{σ(u)·n}, v —
        // no ½ on the boundary arm, MFEM `w = ip.weight`) and the penalty
        // `jmat_c`, finalized by the same pairwise transform as the interior
        // face so the block is bitwise-symmetric by construction.
        for a in 0..n {
            for i in 0..dim {
                let r = a * dim + i;
                for b in 0..n {
                    for j in 0..dim {
                        let c = b * dim + j;
                        pre[r * n_dim + c] += w_f * (sn[b][j][i] * phi[a]);
                        jmat_c[r * n_dim + c] +=
                            jmatcoef * phi[a] * phi[b] * if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
        }
    }

    // Same MFEM closing transform as the interior face (shared `mij`,
    // symmetric-by-construction), then the mutual-zero scatter.
    let mut comb = vec![0.0_f64; n_dim * n_dim];
    for r in 0..n_dim {
        for c in 0..r {
            let arc = pre[r * n_dim + c];
            let acr = pre[c * n_dim + r];
            let m = jmat_c[r * n_dim + c];
            comb[r * n_dim + c] = alpha * acr - arc + m;
            comb[c * n_dim + r] = alpha * arc - acr + m;
        }
        comb[r * n_dim + r] = (alpha - 1.0) * pre[r * n_dim + r] + jmat_c[r * n_dim + r];
    }

    scatter_mutual_zero_square(coo, &dofs, &comb, dim, n);
}

// ─── Scatter helpers (D1280 route 1: MFEM AddSubMatrix skip_zeros) ─────────
//
// MFEM assembles each face through a single
// `mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros=1)` over the *combined*
// (Elem1 ∪ Elem2) vdim dof block (fem/bilinearform.cpp:659), whose per-entry
// rule (linalg/sparsemat.cpp:2782) is:
//
//   skip (i,j)  iff  elmat(i,j) == 0.0  &&  elmat(j,i) == 0.0   (mutual zero)
//
// i.e. mutually-zero coupling positions are **never allocated**, while
// one-sided zeros (elmat(i,j)==0 with a nonzero transpose) ARE allocated and
// kept by `FormSystemMatrix`'s `Finalize(0)` (bilinearform.cpp:944 — "remove_zeros
// = 0") as stored zeros.  The previous `scatter` dropped *any* exact zero,
// which neither allocated MFEM's one-sided stored zeros nor mirrors the
// mutual-zero non-allocation — the entry-count half of D1280.

/// Scatter one square (n·dim)×(n·dim) element block with MFEM's mutual-zero
/// skip: `(i,j)` is pushed unless `v_ij == 0.0 && v_ji == 0.0`.
fn scatter_mutual_zero_square(
    coo: &mut CooMatrix<f64>,
    dofs: &[usize],
    block: &[f64],
    dim: usize,
    n: usize,
) {
    let stride = n * dim;
    for a in 0..n {
        for i in 0..dim {
            let r = a * dim + i;
            for b in 0..n {
                for j in 0..dim {
                    let c = b * dim + j;
                    let val = block[r * stride + c];
                    if val == 0.0 && block[c * stride + r] == 0.0 {
                        continue;
                    }
                    coo.add(dofs[r / dim] * dim + r % dim, dofs[c / dim] * dim + c % dim, val);
                }
            }
        }
    }
}

// MFEM: DgElasticityAssembler — stress-based SIP

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    #[test]
    fn dg_elasticity_block_size() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let n = space.n_dofs();
        let lam = vec![1.0; 8];
        let mu = vec![1.0; 8];
        let a = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, &lam, &mu, 20.0, -1.0, 2, 3, &[],
        );
        assert_eq!(a.nrows, 2 * n);
        assert_eq!(a.ncols, 2 * n);
    }

    /// Full stress SIP must be symmetric for α=-1.
    #[test]
    fn dg_elasticity_stress_sip_symmetric() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let n_elem = space.mesh().n_elements() as usize;
        let lam = vec![1.0; n_elem];
        let mu = vec![1.0; n_elem];
        let a = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, &lam, &mu, 20.0, -1.0, 2, 3, &[],
        );
        let n = a.nrows;
        let mut asym = 0.0_f64;
        let mut norm = 0.0_f64;
        for i in 0..n {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[p] as usize;
                let v = a.values[p];
                norm += v * v;
                let vt = a.get(j, i);
                asym += (v - vt) * (v - vt);
            }
        }
        let rel = (asym / (norm + 1e-300)).sqrt();
        assert!(rel < 1e-12, "stress SIP asymmetry rel={rel:.3e}");
    }

    #[test]
    fn dg_elasticity_positive_diagonal() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let n_elem = space.mesh().n_elements() as usize;
        let lam = vec![1.0; n_elem];
        let mu = vec![1.0; n_elem];
        let a = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, &lam, &mu, 20.0, -1.0, 2, 3, &[],
        );
        for i in 0..a.nrows {
            assert!(a.get(i, i) > 0.0, "diagonal[{i}] <= 0");
        }
    }

    /// Two-material test (λ₁=50/μ₁=50 vs λ₂=1/μ₂=1) must differ from uniform.
    #[test]
    fn dg_elasticity_multi_material_differs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let n_elem = space.mesh().n_elements() as usize;

        let lam_uniform = vec![1.0; n_elem];
        let mu_uniform = vec![1.0; n_elem];
        let a_uni = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, &lam_uniform, &mu_uniform, 20.0, -1.0, 2, 3, &[],
        );

        let mut lam_dual = vec![1.0; n_elem];
        let mut mu_dual = vec![1.0; n_elem];
        // Set first element to different material
        lam_dual[0] = 50.0;
        mu_dual[0] = 50.0;
        let a_dual = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, &lam_dual, &mu_dual, 20.0, -1.0, 2, 3, &[],
        );

        let mut diff = 0.0_f64;
        for i in 0..a_uni.nrows {
            for p in a_uni.row_ptr[i]..a_uni.row_ptr[i + 1] {
                let j = a_uni.col_idx[p] as usize;
                diff += (a_uni.values[p] - a_dual.get(i, j)).abs();
            }
        }
        assert!(diff > 1e-6, "multi-material should differ from uniform; diff={diff:.3e}");
    }
}
