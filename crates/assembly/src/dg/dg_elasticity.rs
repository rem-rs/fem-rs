//! DG elasticity assemblers with full stress-based SIP face terms.
//!
//! Provides the correct DG-SIP linear-elasticity operator:
//!   Volume: ∫ 2μ·ε(u):ε(v) + λ·div(u)·div(v) dx
//!   Interior faces: stress-based SIP (consistency + symmetry + penalty)
//!   Boundary faces: stress-based SIP for weak Dirichlet

use std::collections::HashMap;

use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

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
    /// Faces: stress-based SIP
    ///   a(u,v) = −∫ {σ(u)·n}·⟦v⟧ − α∫ {σ(v)·n}·⟦u⟧ + ∫ (κ/h)⟦u⟧·⟦v⟧ ds
    ///
    /// `dirichlet_attrs` = list of boundary attributes where Dirichlet BCs are
    /// enforced weakly (the parameter `dir_bdr` in MFEM ex17). Pass an empty
    /// slice for pure natural BC.
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
        assemble_volume(&mut coo, space, lambda_elem, mu_elem, dim, quad_order);

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

fn assemble_volume<S: FESpace>(
    coo: &mut CooMatrix<f64>,
    space: &S,
    lambda_elem: &[f64],
    mu_elem: &[f64],
    dim: usize,
    quad_order: u8,
) {
    let mesh = space.mesh();
    let order = space.order();

    for e in mesh.elem_iter() {
        let ei = e as usize;
        let lam = lambda_elem[ei];
        let mu = mu_elem[ei];
        if lam == 0.0 && mu == 0.0 {
            continue;
        }

        let et = mesh.element_type(e);
        let re: Box<dyn ReferenceElement> = ref_elem_vol(et, order);
        let n_l = re.n_dofs();
        let q = re.quadrature(quad_order);

        let dofs: Vec<usize> =
            space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        if det_j.abs() < 1e-30 {
            continue;
        }
        let jit = jac.try_inverse().unwrap().transpose();

        let mut gref = vec![0.0_f64; n_l * dim];
        let mut gphys = vec![0.0_f64; n_l * dim];

        for (qi, xi) in q.points.iter().enumerate() {
            let w = q.weights[qi] * det_j.abs();
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
    let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, el, face_nodes, &mut normal);

    let face_re = ref_elem_face(ElementType::Line2, order);
    let q_face = face_re.quadrature(quad_order);

    let et_l = mesh.element_type(el);
    let re_l = ref_elem_vol(et_l, order);
    let et_r = mesh.element_type(er);
    let re_r = ref_elem_vol(et_r, order);
    let n_l = re_l.n_dofs();
    let n_r = re_r.n_dofs();

    let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
    let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();

    let nodes_l = mesh.element_nodes(el);
    let nodes_r = mesh.element_nodes(er);
    let (jac_l, det_l) = simplex_jac(mesh, nodes_l, dim);
    let (jac_r, det_r) = simplex_jac(mesh, nodes_r, dim);
    let jit_l = jac_l.clone().try_inverse().unwrap().transpose();
    let jit_r = jac_r.clone().try_inverse().unwrap().transpose();

    let lam_l = lambda_elem[el as usize];
    let mu_l = mu_elem[el as usize];
    let lam_r = lambda_elem[er as usize];
    let mu_r = mu_elem[er as usize];

    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    // Accumulate 4 blocks: K_LL, K_LR, K_RL, K_RR
    let mut kll = vec![0.0_f64; n_l * n_l * dim * dim];
    let mut klr = vec![0.0_f64; n_l * n_r * dim * dim];
    let mut krl = vec![0.0_f64; n_r * n_l * dim * dim];
    let mut krr = vec![0.0_f64; n_r * n_r * dim * dim];

    let mut phi_l = vec![0.0_f64; n_l];
    let mut phi_r = vec![0.0_f64; n_r];
    let mut gref_l = vec![0.0_f64; n_l * dim];
    let mut gref_r = vec![0.0_f64; n_r * dim];
    let mut gphys_l = vec![0.0_f64; n_l * dim];
    let mut gphys_r = vec![0.0_f64; n_r * dim];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        let w_f = q_face.weights[qi] * h_f;
        let xp: Vec<f64> = (0..dim).map(|i| x0f[i] + (x1f[i] - x0f[i]) * xi_f[0]).collect();

        let xi_l = phys_to_ref(&jac_l, mesh.node_coords(nodes_l[0]), &xp, dim);
        let xi_r = phys_to_ref(&jac_r, mesh.node_coords(nodes_r[0]), &xp, dim);

        re_l.eval_basis(&xi_l, &mut phi_l);
        re_r.eval_basis(&xi_r, &mut phi_r);
        re_l.eval_grad_basis(&xi_l, &mut gref_l);
        re_r.eval_grad_basis(&xi_r, &mut gref_r);
        xform_grads(&jit_l, &gref_l, &mut gphys_l, n_l, dim);
        xform_grads(&jit_r, &gref_r, &mut gphys_r, n_r, dim);

        // MFEM penalty: κ·|nor|²·wLM  with wLM = (wL+2wM)_avg
        // = κ·h_f²/4·ip.weight·½·[(λ₁+2μ₁)/det₁ + (λ₂+2μ₂)/det₂]
        // Our w_f·pen = q_face.w·h_f·pen must match jmatcoef
        // SIP penalty: κ·(λ+2μ)/h_f (standard DG penalty scaling)
        let lam_face = 0.5 * (lam_l + lam_r);
        let mu_face = 0.5 * (mu_l + mu_r);
        let pen = kappa * (lam_face + 2.0 * mu_face) / h_f;

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

        // K_LL[(a,i), (b,j)]:
        //   term1 = −½·σ_L(φ_b·e_j)·n)_i · φ_L_a    (-{σ(u)·n}·[[v]], left u, left v)
        //   term2 = −½·α·σ_L(φ_a·e_i)·n)_j · φ_L_b  (-α·{σ(v)·n}·[[u]], left v, left u)
        //   term3 = (κ/h)·φ_L_a·φ_L_b·δᵢⱼ             (penalty)
        let stride_ll = n_l * dim;
        for a in 0..n_l {
            for i in 0..dim {
                let row_off = a * dim + i;
                for b in 0..n_l {
                    for j in 0..dim {
                        let col_off = b * dim + j;
                        let t1 = -0.5 * snl[b][j][i] * phi_l[a];
                        let t2 = 0.5 * alpha * snl[a][i][j] * phi_l[b];
                        let t3 = pen * phi_l[a] * phi_l[b] * if i == j { 1.0 } else { 0.0 };
                        kll[row_off * stride_ll + col_off] += w_f * (t1 + t2 + t3);
                    }
                }
            }
        }

        // K_LR[(a,i), (b,j)]: test on left (a,i), trial on right (b,j)
        //   [[v]] = -φ_R_b·e_j (left test v=0, right v=φ_R_b·e_j, so [[v]]= -φ_R_b·e_j)
        //   {σ(u)·n}·[[v]]: u on right → only σ_R·n contributes
        //   {σ(u)·n}·[[v]] = ½·σ_R(φ_b·e_j)·n · (-φ_a·e_i) = -½·σ_R(φ_b·e_j)·n_i · φ_a
        //   term1 = −(-½·σ_R(φ_b·e_j)·n_i · φ_a) = +½·σ_R(φ_b·e_j)·n_i · φ_a
        //
        //   {σ(v)·n}·[[u]]: v on left, u on right
        //   {σ(v)·n}·[[u]] = ½·σ_L(φ_a·e_i)·n · φ_R_b·e_j = ½·σ_L(φ_a·e_i)·n_j · φ_R_b  hmm...
        //   Actually, v on left means v_L = φ_a·e_i, v_R = 0
        //   [[u]]: u on right means u_L = 0, u_R = φ_b·e_j, so [[u]] = -φ_b·e_j
        //   {σ(v)·n} = ½·σ_L(φ_a·e_i)·n (only left contributes)
        //   {σ(v)·n}·[[u]] = ½·σ_L(φ_a·e_i)·n · (-φ_R_b·e_j) = -½·σ_L(φ_a·e_i)·n_j · φ_R_b  no...
        //   {σ(v)·n}·[[u]] = Σ_k (σ(v)·n)_k · [[u]]_k
        //   [[u]]_k = -φ_R_b·δⱼₖ
        //   = Σ_k ½·σ_L(φ_a·e_i)·n)_k · (-φ_R_b·δⱼₖ)
        //   = -½·σ_L(φ_a·e_i)·n)_j · φ_R_b
        //   term2 = −α·(-½·σ_L(φ_a·e_i)·n)_j · φ_R_b) = +½·α·σ_L(φ_a·e_i)·n)_j · φ_R_b
        //         = +½·α·snl[a][i][j] · phi_r[b]
        //
        //   penalty: [[u]]·[[v]] = (-φ_R_b·e_j)·(φ_a·e_i) = -φ_a·φ_R_b·δᵢⱼ
        //   term3 = (κ/h)·(-φ_a·φ_R_b·δᵢⱼ) = -(κ/h)·φ_a·φ_R_b·δᵢⱼ
        // K_LR: test on LEFT (a,i), trial on RIGHT (b,j)
        //   v_L=φ_a·e_i, v_R=0 → [[v]]=φ_a·e_i
        //   u_L=0, u_R=φ_b·e_j → [[u]]=-φ_b·e_j
        //   t1 = -{σ(u)·n}·[[v]] = -½·σ_R(φ_b·e_j)·n·φ_a·e_i = -0.5·snr[b][j][i]·φ_a
        //   t2 = +α·{σ(v)·n}·[[u]] = +α·½·σ_L(φ_a·e_i)·n·(-φ_b·e_j) = -0.5·α·snl[a][i][j]·φ_b
        //   t3 = +(κ/h)·[[u]]·[[v]] = +(κ/h)·(-φ_b·e_j)·(φ_a·e_i) = -pen·φ_a·φ_b·δᵢⱼ
        let stride_lr = n_r * dim;
        for a in 0..n_l {
            for i in 0..dim {
                let row_off = a * dim + i;
                for b in 0..n_r {
                    for j in 0..dim {
                        let col_off = b * dim + j;
                        let t1 = -0.5 * snr[b][j][i] * phi_l[a];
                        let t2 = -0.5 * alpha * snl[a][i][j] * phi_r[b];
                        let t3 = -pen * phi_l[a] * phi_r[b] * if i == j { 1.0 } else { 0.0 };
                        klr[row_off * stride_lr + col_off] += w_f * (t1 + t2 + t3);
                    }
                }
            }
        }

        // K_RL[(a,i), (b,j)]: test on right (a,i), trial on left (b,j)
        // By symmetry of the SIP formulation:
        //   term1: +½·σ_L(φ_b·e_j)·n)_i · φ_R_a  (from [[v]] = φ_a·e_R on right)
        //   term2: +½·α·σ_R(φ_a·e_i)·n)_j · φ_L_b
        //   term3: −(κ/h)·φ_R_a·φ_L_b·δᵢⱼ
        let stride_rl = n_l * dim;
        for a in 0..n_r {
            for i in 0..dim {
                let row_off = a * dim + i;
                for b in 0..n_l {
                    for j in 0..dim {
                        let col_off = b * dim + j;
                        let t1 = 0.5 * snl[b][j][i] * phi_r[a];
                        let t2 = 0.5 * alpha * snr[a][i][j] * phi_l[b];
                        let t3 = -pen * phi_r[a] * phi_l[b] * if i == j { 1.0 } else { 0.0 };
                        krl[row_off * stride_rl + col_off] += w_f * (t1 + t2 + t3);
                    }
                }
            }
        }

        // K_RR: test on RIGHT (a,i), trial on RIGHT (b,j)
        //   v_R=φ_a·e_i, v_L=0 → [[v]]=-φ_a·e_i
        //   u_R=φ_b·e_j, u_L=0 → [[u]]=-φ_b·e_j
        //   t1 = -{σ(u)·n}·[[v]] = -½·σ_R(φ_b·e_j)·n·(-φ_a·e_i) = +0.5·snr[b][j][i]·φ_a
        //   t2 = +α·{σ(v)·n}·[[u]] = +α·½·σ_R(φ_a·e_i)·n·(-φ_b·e_j) = -0.5·α·snr[a][i][j]·φ_b
        //   t3 = +(κ/h)·[[u]]·[[v]] = +(κ/h)·(-φ_b·e_j)·(-φ_a·e_i) = +pen·φ_a·φ_b·δᵢⱼ
        let stride_rr = n_r * dim;
        for a in 0..n_r {
            for i in 0..dim {
                let row_off = a * dim + i;
                for b in 0..n_r {
                    for j in 0..dim {
                        let col_off = b * dim + j;
                        let t1 = 0.5 * snr[b][j][i] * phi_r[a];
                        let t2 = -0.5 * alpha * snr[a][i][j] * phi_r[b];
                        let t3 = pen * phi_r[a] * phi_r[b] * if i == j { 1.0 } else { 0.0 };
                        krr[row_off * stride_rr + col_off] += w_f * (t1 + t2 + t3);
                    }
                }
            }
        }
    }

    // Scatter blocks into global matrix
    scatter(coo, &dofs_l, &dofs_l, &kll, dim, n_l, n_l);
    scatter(coo, &dofs_l, &dofs_r, &klr, dim, n_l, n_r);
    scatter(coo, &dofs_r, &dofs_l, &krl, dim, n_r, n_l);
    scatter(coo, &dofs_r, &dofs_r, &krr, dim, n_r, n_r);
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
    let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
    orient_normal_outward(mesh, elem, face_nodes, &mut normal);

    let et = mesh.element_type(elem);
    let re = ref_elem_vol(et, order);
    let n = re.n_dofs();
    let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

    let nodes = mesh.element_nodes(elem);
    let (jac, det_j) = simplex_jac(mesh, nodes, dim);
    if det_j.abs() < 1e-30 {
        return;
    }
    let jit = jac.clone().try_inverse().unwrap().transpose();

    let lam = lambda_elem[elem as usize];
    let mu = mu_elem[elem as usize];

    let face_re = ref_elem_face(ElementType::Line2, order);
    let q_face = face_re.quadrature(quad_order);

    let x0f = mesh.node_coords(face_nodes[0]);
    let x1f = mesh.node_coords(face_nodes[1]);

    let mut kbd = vec![0.0_f64; n * n * dim * dim];
    let mut phi = vec![0.0_f64; n];
    let mut gref = vec![0.0_f64; n * dim];
    let mut gphys = vec![0.0_f64; n * dim];

    for (qi, xi_f) in q_face.points.iter().enumerate() {
        let w_f = q_face.weights[qi] * h_f;
        let xp: Vec<f64> = (0..dim).map(|i| x0f[i] + (x1f[i] - x0f[i]) * xi_f[0]).collect();
        let xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim);

        re.eval_basis(&xi_e, &mut phi);
        re.eval_grad_basis(&xi_e, &mut gref);
        xform_grads(&jit, &gref, &mut gphys, n, dim);

        // MFEM scales penalty by (λ + 2μ) for elasticity
        let pen = kappa * (lam + 2.0 * mu) / h_f;

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
        //                      + (κ/h)·φ_a·φ_b·δᵢⱼ
        let stride = n * dim;
        for a in 0..n {
            for i in 0..dim {
                let row_off = a * dim + i;
                for b in 0..n {
                    for j in 0..dim {
                        let col_off = b * dim + j;
                        let t1 = -sn[b][j][i] * phi[a];
                        let t2 = alpha * sn[a][i][j] * phi[b];
                        let t3 = pen * phi[a] * phi[b] * if i == j { 1.0 } else { 0.0 };
                        kbd[row_off * stride + col_off] += w_f * (t1 + t2 + t3);
                    }
                }
            }
        }
    }

    scatter(coo, &dofs, &dofs, &kbd, dim, n, n);
}

// ─── Scatter helper ────────────────────────────────────────────────────────
//
// Block layout: K_block[a*dim+i][b*dim+j] → global[(dofs_ri[a]*dim+i), (dofs_ci[b]*dim+j)]
// The block is stored flat: [a*dim+i][b*dim+j] at index (a*dim+i)*stride + b*dim+j

fn scatter(
    coo: &mut CooMatrix<f64>,
    dofs_row: &[usize],
    dofs_col: &[usize],
    block: &[f64],
    dim: usize,
    n_row: usize,
    n_col: usize,
) {
    let stride = n_col * dim;
    for a in 0..n_row {
        for i in 0..dim {
            let row_base = dofs_row[a] * dim + i;
            for b in 0..n_col {
                for j in 0..dim {
                    let val = block[(a * dim + i) * stride + b * dim + j];
                    if val != 0.0 {
                        let col_base = dofs_col[b] * dim + j;
                        coo.add(row_base, col_base, val);
                    }
                }
            }
        }
    }
}

// ─── Face-to-element map ───────────────────────────────────────────────────

fn build_face_elem_map<M: MeshTopology>(mesh: &M, dim: usize) -> HashMap<u32, u32> {
    let mut vol_face_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let local_faces = |npe: usize| -> Vec<Vec<usize>> {
        match (npe, dim) {
            (3, 2) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (4, 2) => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]],
            (4, 3) => vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2],
            ],
            _ => vec![],
        }
    };
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let npe = nodes.len();
        for lf in local_faces(npe) {
            let mut key: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            key.sort_unstable();
            vol_face_map.entry(key).or_insert(e);
        }
    }
    let mut result = HashMap::new();
    for f in mesh.face_iter() {
        let fnodes = mesh.face_nodes(f);
        let mut key: Vec<u32> = fnodes.to_vec();
        key.sort_unstable();
        if let Some(&elem) = vol_face_map.get(&key) {
            result.insert(f, elem);
        }
    }
    result
}

// ─── Geometry helpers (2-D only, mirrored from dg.rs) ──────────────────────

fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    (len, vec![-dy / len, dx / len])
}

fn orient_normal_outward<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    normal: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let enodes = mesh.element_nodes(elem);
    let npe = enodes.len();
    let mut centroid = vec![0.0_f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            centroid[d] += c[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= npe as f64;
    }
    let mut midpoint = vec![0.0_f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            midpoint[d] += c[d];
        }
    }
    for d in 0..dim {
        midpoint[d] /= face_nodes.len() as f64;
    }
    let dot: f64 = (0..dim)
        .map(|d| normal[d] * (midpoint[d] - centroid[d]))
        .sum();
    if dot < 0.0 {
        for d in 0..dim {
            normal[d] = -normal[d];
        }
    }
}

fn phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let j_inv = jac
        .clone()
        .try_inverse()
        .expect("degenerate element in phys_to_ref");
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            xi[i] += j_inv[(i, k)] * dx[k];
        }
    }
    xi
}

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3};
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Quad4, 1) => Box::new(fem_element::lagrange::QuadQ1),
        (ElementType::Quad4, 2) => Box::new(fem_element::lagrange::QuadQ2),
        _ => panic!("ref_elem_vol: unsupported ({et:?}, order={order})"),
    }
}

fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{SegP1, SegP2, SegP3};
    match (et, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        _ => panic!("ref_elem_face: unsupported ({et:?}, order={order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            gp[i * dim + j] = (0..dim).map(|k| jit[(j, k)] * gr[i * dim + k]).sum();
        }
    }
}

use nalgebra::DMatrix;

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
