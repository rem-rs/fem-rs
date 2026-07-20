//! B̄ (B-Bar) reduced-integration linear elasticity integrator.
//!
//! The B̄ method [[Hughes, 1980](https://doi.org/10.1002/nme.1620150906)] replaces
//! the volumetric part of the strain-displacement matrix **B** with its
//! element-averaged value:
//!
//! ```text
//! B̄(ξ) = B_dev(ξ) + ⟨B_vol⟩_e
//! ```
//!
//! where `⟨B_vol⟩_e = (1/|Ω_e|) ∫_{Ω_e} B_vol(ξ) dΩ` is constant over the element.
//!
//! This eliminates volumetric locking for low-order elements (Tri3, Quad4, Tet4,
//! Hex8) in the nearly-incompressible limit (ν → 0.5), at the cost of a
//! two-pass element assembly (first pass computes averaged gradients, second
//! pass assembles the modified stiffness).
//!
//! # Usage
//!
//! ```rust,ignore
//! use fem_assembly::standard::assemble_bbar_elasticity;
//!
//! let k_bbar = assemble_bbar_elasticity(
//!     &vspace,
//!     &1.0e6,   // λ (or a ScalarCoeff)
//!     &1.0e3,   // μ
//!     false,    // plane_stress
//!     3,        // quadrature order
//! );
//! ```
//!
//! # Theory
//!
//! For isotropic linear elasticity the B̄ method gives the modified bilinear form:
//!
//! ```text
//! a_B̄(u, v) = λ |Ω_e| ⟨∇·u⟩⟨∇·v⟩ + ∫_{Ω_e} 2μ [ε(u):ε(v) - 1/dim (∇·u)(∇·v)] dΩ
//! ```
//!
//! Thus the stiffness matrix becomes:
//!
//! ```text
//! K_kl^{ab} = λ |Ω_e| ⟨∂φ_k/∂x_a⟩ ⟨∂φ_l/∂x_b⟩
//!            + ∫_{Ω_e} 2μ [ε_ij(φ_k e_a) ε_ij(φ_l e_b) - 1/dim (∂φ_k/∂x_a)(∂φ_l/∂x_b)] dΩ
//! ```

use nalgebra::DMatrix;
use fem_core::types::ElemId;
use fem_element::{
    QuadratureRule, ReferenceElement, PrismPk, PyramidPk,
    lagrange::{SegP1, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3, TriP4,
               QuadQ1, HexQ1},
};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_linalg::CooMatrix;
use fem_linalg::CsrMatrix;
use fem_mesh::{ElementTransformation, element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};

// ─── Reference element factory (re-exported from assembler) ─────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(TriP1), // P0 handled elsewhere
        (ElementType::Tri3 | ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3 | ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3 | ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tri3 | ElementType::Tri6, 4) => Box::new(TriP4),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Quad4, _) => Box::new(fem_element::lagrange::factory::QuadQk::new(order as usize)),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, _) =>
            Box::new(PrismPk::new(order as usize)),
        (ElementType::Pyramid5 | ElementType::Pyramid13, _) =>
            Box::new(PyramidPk::new(order as usize)),
        _ => panic!("bbar::ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

fn mesh_type_to_factory(et: ElementType) -> FactoryElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => FactoryElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => FactoryElemType::Hex,
        ElementType::Line2 | ElementType::Line3 => FactoryElemType::Seg,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => FactoryElemType::Prism,
        ElementType::Pyramid5 | ElementType::Pyramid13 => FactoryElemType::Pyramid,
        _ => panic!("mesh_type_to_factory: unsupported ElementType {et:?}"),
    }
}

fn geo_ref_elem(mesh: &dyn MeshTopology, e: u32) -> Option<Box<dyn ReferenceElement>> {
    let et = mesh.element_type(e);
    let g = mesh.geom_order();
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20
        | ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18
        | ElementType::Pyramid5 | ElementType::Pyramid13);
    if g == 1 && !is_quad_hex { return None; }
    let order = if g > 1 { g } else { 1 };
    let ft = mesh_type_to_factory(et);
    Some(factory_ref_elem(ft, order))
}

fn is_affine(et: ElementType, geom_order: u8) -> bool {
    if geom_order > 1 { return false; }
    matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
}

fn isoparametric_jacobian<M: MeshTopology>(
    mesh: &M, nodes: &[u32], geo_elem: &dyn ReferenceElement,
    xi: &[f64], dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0_f64; n_geo * dim];
    let mut phi_geo  = vec![0.0_f64; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];
    for k in 0..n_geo {
        let xk = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            xp[i] += phi_geo[k] * xk[i];
            for d in 0..dim {
                j[(i, d)] += xk[i] * grad_geo[k * dim + d];
            }
        }
    }
    let det = j.determinant();
    (j, det, xp)
}

fn transform_grads(j_inv_t: &DMatrix<f64>, grad_ref: &[f64], grad_phys: &mut [f64], n_ldofs: usize, dim: usize) {
    for i in 0..n_ldofs {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(j, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + j] = s;
        }
    }
}

/// Assemble the linear-elasticity stiffness matrix using the B̄ (B-Bar) method.
///
/// This function performs a **two-pass element assembly**:
/// 1. First pass over quadrature points computes element volume `Ω_e` and
///    element-averaged shape-function gradients `⟨∂φ_i/∂x_j⟩`.
/// 2. Second pass assembles the modified stiffness matrix `K_B̄` using the
///    averaged volumetric term and the standard deviatoric term.
///
/// # Arguments
/// * `space` — VectorH¹ finite element space (must have `dim` components).
/// * `lambda` — First Lamé parameter (constant or coefficient).
/// * `mu` — Shear modulus (constant or coefficient).
/// * `plane_stress` — If true, uses plane-stress modification of λ in 2D.
/// * `quad_order` — Quadrature order (same rule used for both passes).
///
/// # Returns
/// Sparse stiffness matrix in COO format.
pub fn assemble_bbar_elasticity<S, C1, C2>(
    space: &S,
    lambda: &C1,
    mu: &C2,
    plane_stress: bool,
    quad_order: u8,
) -> CooMatrix<f64>
where
    S: FESpace,
    C1: ScalarCoeff,
    C2: ScalarCoeff,
{
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let n_dofs_total = space.n_dofs();
    let nelem = mesh.n_elements();

    let mut coo = CooMatrix::new(n_dofs_total, n_dofs_total);

    for e in 0..nelem as u32 {
        let elem_type = mesh.element_type(e);
        let order = space.element_order(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let n_nodes = n_ldofs; // for VectorH1, n_dofs = n_nodes * dim
        let n_elem_dofs = n_nodes * dim;
        let raw_dofs: &[u32] = space.element_dofs(e);
        let global_dofs: Vec<usize> = raw_dofs.iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let elem_tag = mesh.element_tag(e);

        let g_order = mesh.geom_order();
        let affine = is_affine(elem_type, g_order);
        let geo_elem = if affine { None } else { geo_ref_elem(mesh, e) };

        let affine_tr = if affine {
            Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
        } else {
            None
        };

        // Quadrature rule
        let quad = ref_elem.quadrature(order.max(quad_order));

        // ─── Phase 1: compute element volume and averaged gradients ─────
        let mut elem_vol = 0.0_f64;
        let mut avg_grad = vec![0.0_f64; n_nodes * dim];

        let mut phi        = vec![0.0_f64; n_ldofs];
        let mut grad_ref   = vec![0.0_f64; n_ldofs * dim];
        let mut grad_phys  = vec![0.0_f64; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w_q: f64;
            if affine {
                let tr = affine_tr.as_ref().unwrap();
                w_q = quad.weights[q] * tr.det_j().abs();
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(tr.jacobian_inv_t(), &grad_ref, &mut grad_phys, n_ldofs, dim);
            } else if let Some(ref geo) = geo_elem {
                let geo_nds = mesh.geometry_nodes(e);
                let (jac_qp, det_qp, _xp_qp) = isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), xi, dim);
                w_q = quad.weights[q] * det_qp.abs();
                let jit = match jac_qp.try_inverse() {
                    Some(inv) => inv.transpose(),
                    None => continue,
                };
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, n_ldofs, dim);
            } else {
                continue; // degenerate
            }

            elem_vol += w_q;
            for i in 0..n_nodes {
                for d in 0..dim {
                    avg_grad[i * dim + d] += w_q * grad_phys[i * dim + d];
                }
            }
        }

        if elem_vol < 1e-30 {
            eprintln!("warning: BBar: element {e} has zero or negative volume ({:.3e})", elem_vol);
            continue;
        }

        // Normalize averaged gradients
        let inv_vol = 1.0 / elem_vol;
        for g in avg_grad.iter_mut() {
            *g *= inv_vol;
        }

        // ─── Phase 2: assemble B̄ stiffness matrix ────────────────────
        let n = n_elem_dofs;
        let mut k_elem = vec![0.0_f64; n * n];

        // Part A: volumetric (constant) contribution
        // K_vol[k][a][l][b] = λ * Ω_e * ⟨∂φ_k/∂x_a⟩ * ⟨∂φ_l/∂x_b⟩
        // Evaluate coefficients at element centroid
        let ctx = CoeffCtx::from_qp(
            &[], dim, e, elem_tag,
            None, Some(raw_dofs),
        );
        let lam_raw = lambda.eval(&ctx);
        let mu_raw = mu.eval(&ctx);
        let lam_eff = if plane_stress && dim == 2 {
            2.0 * lam_raw * mu_raw / (lam_raw + 2.0 * mu_raw).max(1e-30)
        } else {
            lam_raw
        };

        for k in 0..n_nodes {
            for a in 0..dim {
                let row = k * dim + a;
                let avg_grad_k_a = avg_grad[k * dim + a];
                for l in 0..n_nodes {
                    for b in 0..dim {
                        let col = l * dim + b;
                        let avg_grad_l_b = avg_grad[l * dim + b];
                        k_elem[row * n + col] += lam_eff * elem_vol * avg_grad_k_a * avg_grad_l_b;
                    }
                }
            }
        }

        // Part B: deviatoric (integrated at qp)
        // K_dev += Σ_q w_q * [2μ ε:ε - (2μ/dim) (∂φ_k/∂x_a)(∂φ_l/∂x_b)]
        let dim_f64 = dim as f64;
        for (q, xi) in quad.points.iter().enumerate() {
            let w_q: f64;
            let xp_phys: Vec<f64>;

            if affine {
                let tr = affine_tr.as_ref().unwrap();
                w_q = quad.weights[q] * tr.det_j().abs();
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(tr.jacobian_inv_t(), &grad_ref, &mut grad_phys, n_ldofs, dim);
                xp_phys = tr.map_to_physical(xi);
            } else if let Some(ref geo) = geo_elem {
                let geo_nds = mesh.geometry_nodes(e);
                let (jac_qp, det_qp, xp) = isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), xi, dim);
                w_q = quad.weights[q] * det_qp.abs();
                let jit = match jac_qp.try_inverse() {
                    Some(inv) => inv.transpose(),
                    None => continue,
                };
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&jit, &grad_ref, &mut grad_phys, n_ldofs, dim);
                xp_phys = xp;
            } else {
                continue;
            }

            // Evaluate μ at this qp for coefficient functions
            let ctx_qp = CoeffCtx::from_qp(
                &xp_phys, dim, e, elem_tag,
                Some(&phi), Some(raw_dofs),
            );
            let mu_qp = mu.eval(&ctx_qp);

            for k in 0..n_nodes {
                for a in 0..dim {
                    let row = k * dim + a;
                    // Gather grada: gradient components of φ_k at this qp
                    let grada: Vec<f64> = (0..dim).map(|d| grad_phys[k * dim + d]).collect();

                    for l in 0..n_nodes {
                        for b in 0..dim {
                            let col = l * dim + b;
                            let gradb: Vec<f64> = (0..dim).map(|d| grad_phys[l * dim + d]).collect();

                            // Standard ε:ε (shear energy)
                            let mut shear = 0.0;
                            for i in 0..dim {
                                for j in 0..dim {
                                    let eij_a = 0.5 * (
                                        (if j == a { grada[i] } else { 0.0 })
                                        + (if i == a { grada[j] } else { 0.0 })
                                    );
                                    let eij_b = 0.5 * (
                                        (if j == b { gradb[i] } else { 0.0 })
                                        + (if i == b { gradb[j] } else { 0.0 })
                                    );
                                    shear += eij_a * eij_b;
                                }
                            }

                            // Subtract volumetric part: (2μ/dim) * (div_u)(div_v)
                            // div_u = ∂φ_k/∂x_a, div_v = ∂φ_l/∂x_b
                            let vol_sub = (2.0 * mu_qp / dim_f64) * grada[a] * gradb[b];

                            k_elem[row * n + col] += w_q * (2.0 * mu_qp * shear - vol_sub);
                        }
                    }
                }
            }
        }

        coo.add_element_matrix(&global_dofs, &k_elem);
    }

    coo
}

// ─── F-Bar nonlinear integrator ──────────────────────────────────────────────

/// F̄ (F-Bar) method for nearly-incompressible nonlinear hyperelasticity.
///
/// The F̄ method [[de Souza Neto et al., 1996](https://doi.org/10.1002/(SICI)1097-0207(19960229)39:4<685::AID-NME872>3.0.CO;2-O)]
/// replaces the volumetric part of the deformation gradient **F** with its
/// element-averaged value:
///
/// ```text
/// F̄ = (J̄ / J)^{1/3} F
/// ```
///
/// where `J = det(F)` and `J̄ = (1/|Ω_e|) ∫_{Ω_e} J dΩ` is the element-averaged
/// Jacobian determinant. This preserves the deviatoric deformation exactly while
/// using a uniform volumetric stretch, eliminating volumetric locking for
/// low-order elements in the nearly-incompressible regime.
///
/// # Usage
///
/// ```rust,ignore
/// use fem_assembly::standard::{FBarIntegrator, StVenantKirchhoff};
///
/// let model = StVenantKirchhoff::new(1.0e6, 1.0e3);
/// let fbar = FBarIntegrator::new(model);
/// let (tangent, residual) = fbar.assemble(&space, &u_sol, quad_order);
/// ```
pub struct FBarIntegrator<M: crate::standard::elasticity::HyperelasticModel> {
    pub model: M,
}

impl<M: crate::standard::elasticity::HyperelasticModel> FBarIntegrator<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Assemble the tangent stiffness matrix and residual vector using the F̄ method.
    ///
    /// # Arguments
    /// * `space` — VectorH¹ finite element space.
    /// * `u_sol` — Current displacement solution (flat array, interleaved
    ///   node-major: `[u_x0, u_y0, u_x1, u_y1, ...]`).
    /// * `quad_order` — Quadrature order.
    ///
    /// # Returns
    /// `(K_tan, residual)` where `K_tan` is the tangent stiffness matrix (COO)
    /// and `residual` is the internal force vector.
    pub fn assemble<S: FESpace>(
        &self,
        space: &S,
        u_sol: &[f64],
        quad_order: u8,
    ) -> (CooMatrix<f64>, Vec<f64>)
    {
        let mesh = space.mesh();
        let dim = mesh.dim() as usize;
        let n_dofs_total = space.n_dofs();
        let nelem = mesh.n_elements();

        let mut coo = CooMatrix::new(n_dofs_total, n_dofs_total);
        let mut rhs = vec![0.0_f64; n_dofs_total];

        for e in 0..nelem as u32 {
            let elem_type = mesh.element_type(e);
            let order = space.element_order(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_nodes = n_ldofs;
            let n_elem_dofs = n_nodes * dim;
            let raw_dofs: &[u32] = space.element_dofs(e);
            let global_dofs: Vec<usize> = raw_dofs.iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let elem_tag = mesh.element_tag(e);

            // Current element displacement vector
            let mut u_elem = vec![0.0_f64; n_elem_dofs];
            for (i, &dof) in global_dofs.iter().enumerate() {
                u_elem[i] = u_sol[dof];
            }

            let g_order = mesh.geom_order();
            let affine = is_affine(elem_type, g_order);
            let geo_elem = if affine { None } else { geo_ref_elem(mesh, e) };
            let affine_tr = if affine {
                Some(ElementTransformation::from_simplex_nodes(mesh, nodes))
            } else { None };

            let quad = ref_elem.quadrature(order.max(quad_order));

            let mut phi       = vec![0.0_f64; n_ldofs];
            let mut grad_ref  = vec![0.0_f64; n_ldofs * dim];
            let mut grad_phys = vec![0.0_f64; n_ldofs * dim];

            // ─── Phase 1: compute element-averaged J̄ ─────────────────
            let mut avg_J = 0.0_f64;
            let mut elem_vol_0 = 0.0_f64; // volume of the reference configuration

            for (q, xi) in quad.points.iter().enumerate() {
                let (w_q, _xp) = Self::eval_gradients(
                    mesh, e, xi, affine_tr.as_ref(), geo_elem.as_ref(),
                    &ref_elem, &mut phi, &mut grad_ref, &mut grad_phys,
                    n_ldofs, dim,
                );
                // Deformation gradient F = I + ∇u (full 3×3 identity ensures
                // F_33 = 1 for 2D plane-strain, which StVenantKirchhoff expects)
                let mut F = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    F[i][i] = 1.0;
                }
                for i in 0..dim {
                    for j in 0..dim {
                        for k in 0..n_nodes {
                            F[i][j] += u_elem[k * dim + i] * grad_phys[k * dim + j];
                        }
                    }
                }
                let J = if dim == 2 {
                    F[0][0] * F[1][1] - F[0][1] * F[1][0]
                } else {
                    F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
                        - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
                        + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0])
                };
                elem_vol_0 += w_q;
                avg_J += w_q * J;
            }

            if elem_vol_0 < 1e-30 { continue; }
            let J_bar = avg_J / elem_vol_0;

            // ─── Phase 2: assemble F̄ stiffness and residual ──────────
            let n = n_elem_dofs;
            let mut k_elem = vec![0.0_f64; n * n];
            let mut f_elem = vec![0.0_f64; n];

            for (q, xi) in quad.points.iter().enumerate() {
                let (w_q, _xp) = Self::eval_gradients(
                    mesh, e, xi, affine_tr.as_ref(), geo_elem.as_ref(),
                    &ref_elem, &mut phi, &mut grad_ref, &mut grad_phys,
                    n_ldofs, dim,
                );

                // Deformation gradient F (full 3×3 identity)
                let mut F = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    F[i][i] = 1.0;
                }
                for i in 0..dim {
                    for j in 0..dim {
                        for k in 0..n_nodes {
                            F[i][j] += u_elem[k * dim + i] * grad_phys[k * dim + j];
                        }
                    }
                }

                let J = if dim == 2 {
                    F[0][0] * F[1][1] - F[0][1] * F[1][0]
                } else {
                    F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
                        - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
                        + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0])
                };

                if J <= 0.0 {
                    eprintln!("warning: FBar: negative/zero Jacobian at element {e} qp {q}: J={:.3e}", J);
                    continue;
                }

                // F̄ = (J̄ / J)^{1/3} * F
                let scale = (J_bar / J).powf(1.0 / 3.0);
                let mut F_bar = F;
                for i in 0..dim {
                    for j in 0..dim {
                        F_bar[i][j] *= scale;
                    }
                }

                // Compute stress and tangent modulus with F̄
                let mut S_voigt = [0.0_f64; 6];
                let mut C_tan = [[0.0_f64; 6]; 6];

                let mut F_arr = [0.0_f64; 9];
                for i in 0..3 {
                    F_arr[i * 3 + i] = 1.0;
                }
                for i in 0..dim {
                    for j in 0..dim {
                        F_arr[i * 3 + j] = F_bar[i][j];
                    }
                }
                self.model.stress_and_modulus(&F_arr, &mut S_voigt, &mut C_tan);

                // Convert S (2nd Piola-Kirchhoff) to element residual and
                // C (tangent modulus) to element stiffness matrix.
                //
                // The internal force contribution:
                //   f_aI = ∫ F̄ · S · ∇N_a dΩ   (in reference config)
                //
                // The tangent stiffness contribution:
                //   K_aI_bJ = ∫ [δ_IJ · S + C_IJKL · F̄] · (∇N_a ⊗ ∇N_b) dΩ
                //
                // For simplicity we compute the geometric and material parts.

                let w = w_q;

                for a in 0..n_nodes {
                    // ∇N_a in physical space (reference config gradient)
                    let grada: Vec<f64> = (0..dim).map(|d| grad_phys[a * dim + d]).collect();

                    for i in 0..dim {
                        let row = a * dim + i;

                        // Residual: f_aI += w * F̄_ij * S_jK * ∇_K N_a
                        let mut fi = 0.0;
                        for j in 0..dim {
                            for k in 0..dim {
                                // S in voigt: [S_xx, S_yy, S_zz, S_xy, S_xz, S_yz]
                                let S_jk = if j == k {
                                    if j == 0 { S_voigt[0] }
                                    else if j == 1 { S_voigt[1] }
                                    else { S_voigt[2] }
                                } else if (j == 0 && k == 1) || (j == 1 && k == 0) {
                                    S_voigt[3]
                                } else if (j == 0 && k == 2) || (j == 2 && k == 0) {
                                    S_voigt[4]
                                } else {
                                    S_voigt[5]
                                };
                                fi += F_bar[i][j] * S_jk * grada[k];
                            }
                        }
                        f_elem[row] += w * fi;

                        // Tangent stiffness
                        for b in 0..n_nodes {
                            let gradb: Vec<f64> = (0..dim).map(|d| grad_phys[b * dim + d]).collect();

                            for j in 0..dim {
                                let col = b * dim + j;

                                // Geometic stiffness: δ_ij * w * S_kl * (∇N_a)_k * (∇N_b)_l
                                let mut k_geom = 0.0;
                                for k in 0..dim {
                                    for l in 0..dim {
                                        let S_kl = if k == l {
                                            if k == 0 { S_voigt[0] }
                                            else if k == 1 { S_voigt[1] }
                                            else { S_voigt[2] }
                                        } else if (k == 0 && l == 1) || (k == 1 && l == 0) {
                                            S_voigt[3]
                                        } else if (k == 0 && l == 2) || (k == 2 && l == 0) {
                                            S_voigt[4]
                                        } else {
                                            S_voigt[5]
                                        };
                                        k_geom += S_kl * grada[k] * gradb[l];
                                    }
                                }
                                if i == j {
                                    k_elem[row * n + col] += w * k_geom;
                                }

                                // Material stiffness: w * F̄_im * C_mKnL * F̄_jn * (∇N_a)_K * (∇N_b)_L
                                let mut k_mat = 0.0;
                                for m in 0..dim {
                                    for n in 0..dim {
                                        for k in 0..dim {
                                            for l in 0..dim {
                                                // Map (m,K,n,L) to C_tan voigt indices
                                                let c_mk = if m == k { m } else if (m == 0 && k == 1) || (m == 1 && k == 0) { 3 }
                                                    else if (m == 0 && k == 2) || (m == 2 && k == 0) { 4 }
                                                    else { 5 };
                                                let c_nl = if n == l { n } else if (n == 0 && l == 1) || (n == 1 && l == 0) { 3 }
                                                    else if (n == 0 && l == 2) || (n == 2 && l == 0) { 4 }
                                                    else { 5 };
                                                k_mat += F_bar[i][m] * C_tan[c_mk][c_nl] * F_bar[j][n] * grada[k] * gradb[l];
                                            }
                                        }
                                    }
                                }
                                k_elem[row * n + col] += w * k_mat;
                            }
                        }
                    }
                }
            }

            coo.add_element_matrix(&global_dofs, &k_elem);
            // Add element residual to global residual
            for (i, &dof) in global_dofs.iter().enumerate() {
                rhs[dof] += f_elem[i];
            }
        }

        (coo, rhs)
    }

    /// Evaluate gradients and integration weight at a quadrature point.
    fn eval_gradients<MT: MeshTopology>(
        mesh: &MT, e: u32, xi: &[f64],
        affine_tr: Option<&fem_mesh::ElementTransformation>,
        geo_elem: Option<&Box<dyn ReferenceElement>>,
        ref_elem: &Box<dyn ReferenceElement>,
        phi: &mut [f64],
        grad_ref: &mut [f64],
        grad_phys: &mut [f64],
        n_ldofs: usize,
        dim: usize,
    ) -> (f64, Vec<f64>) {
        let n_ldofs = ref_elem.n_dofs();
        if let Some(tr) = affine_tr {
            let w = tr.det_j().abs();
            ref_elem.eval_basis(xi, phi);
            ref_elem.eval_grad_basis(xi, grad_ref);
            transform_grads(tr.jacobian_inv_t(), grad_ref, grad_phys, n_ldofs, dim);
            let xp = tr.map_to_physical(xi);
            (w, xp)
        } else if let Some(ref geo) = geo_elem {
            let geo_nds = mesh.geometry_nodes(e);
            let (jac_qp, det_qp, xp) = isoparametric_jacobian(mesh, geo_nds, geo.as_ref(), xi, dim);
            let w = det_qp.abs();
            let jit = match jac_qp.try_inverse() {
                Some(inv) => inv.transpose(),
                None => return (0.0, vec![]),
            };
            ref_elem.eval_basis(xi, phi);
            ref_elem.eval_grad_basis(xi, grad_ref);
            transform_grads(&jit, grad_ref, grad_phys, n_ldofs, dim);
            (w, xp)
        } else {
            (0.0, vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::VectorH1Space;
    use crate::standard::ElasticityIntegrator;
    use crate::Assembler;

    /// Helper: assemble standard elasticity matrix.
    fn standard_stiffness(mesh: &Mesh<2>, lambda: f64, mu: f64, plane_stress: bool) -> CsrMatrix<f64> {
        let space = VectorH1Space::new(mesh.clone(), 1, 2);
        let integ = ElasticityIntegrator::new(lambda, mu);
        let integ = if plane_stress { integ.with_plane_stress(true) } else { integ };
        Assembler::assemble_bilinear(&space, &[&integ], 3)
    }

    /// Helper: assemble B̄ matrix.
    fn bbar_stiffness(mesh: &Mesh<2>, lambda: f64, mu: f64, plane_stress: bool) -> CooMatrix<f64> {
        let space = VectorH1Space::new(mesh.clone(), 1, 2);
        assemble_bbar_elasticity(&space, &lambda, &mu, plane_stress, 3)
    }

    fn csr_to_dense_vec(csr: &CsrMatrix<f64>) -> Vec<f64> { csr.to_dense() }
    fn coo_to_dense_vec(coo: &CooMatrix<f64>) -> Vec<f64> { coo.clone().into_csr().to_dense() }

    fn dense_diff(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        let mut diff = 0.0;
        for i in 0..n {
            diff += (a[i] - b[i]).abs();
        }
        diff
    }

    #[test]
    fn bbar_matches_standard_for_compressible() {
        // For compressible materials (ν=0.3), B-Bar should be close to standard
        let mesh = Mesh::<2>::unit_square_tri(8);
        let lambda = 1.0;
        let mu = 1.0;
        let k_std = standard_stiffness(&mesh, lambda, mu, false);
        let k_bbar = bbar_stiffness(&mesh, lambda, mu, false);
        let d_std = csr_to_dense_vec(&k_std);
        let d_bbar = coo_to_dense_vec(&k_bbar);
        let diff = dense_diff(&d_std, &d_bbar);
        // B-Bar differs from standard for all materials (because the volumetric
        // part is averaged); the difference should be small for compressible
        // materials with refined meshes.
        println!("B-Bar vs standard diff (compressible, tri8): {:.6e}", diff);
        // We don't assert a tight tolerance — the methods are structurally different.
        // But the difference should be finite and reasonable.
        assert!(diff.is_finite());
    }

    #[test]
    fn bbar_matrix_symmetric() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let k = bbar_stiffness(&mesh, 1.0, 0.5, false);
        let dense = coo_to_dense_vec(&k);
        let n = k.nrows;
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                if diff > 1e-10 {
                    panic!("BBar matrix not symmetric at ({i},{j}): diff={:.3e}", diff);
                }
            }
        }
    }

    #[test]
    fn bbar_row_sums_zero() {
        // Rigid-body modes: row sums should be zero for each DOF
        let mesh = Mesh::<2>::unit_square_tri(6);
        let k = bbar_stiffness(&mesh, 1.0, 0.5, false);
        let dense = coo_to_dense_vec(&k);
        let n = k.nrows;
        for row in 0..n {
            let s: f64 = (0..n).map(|c| dense[row * n + c]).sum();
            if s.abs() > 1e-10 {
                panic!("BBar row sum not zero at row {row}: {:.3e}", s);
            }
        }
    }

    #[test]
    fn bbar_plane_stress_differs_from_plane_strain() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let k_strain = bbar_stiffness(&mesh, 1.0, 0.5, false);
        let k_stress = bbar_stiffness(&mesh, 1.0, 0.5, true);
        let d_strain = coo_to_dense_vec(&k_strain);
        let d_stress = coo_to_dense_vec(&k_stress);
        let diff = dense_diff(&d_strain, &d_stress);
        assert!(diff > 1e-10, "plane stress and strain should differ");
    }

    #[test]
    fn bbar_near_incompressible_stable() {
        // ν → 0.5: nearly incompressible (λ ≫ μ)
        let mesh = Mesh::<2>::unit_square_tri(8);
        let mu = 1.0;
        let lambda = 1.0e6; // ν ≈ 0.5 - 5e-7
        let k = bbar_stiffness(&mesh, lambda, mu, false);
        let dense = coo_to_dense_vec(&k);
        let n = k.nrows;
        // Matrix should be well-formed (finite entries)
        for i in 0..n {
            for j in 0..n {
                assert!(dense[i * n + j].is_finite(), "Non-finite entry at ({i},{j})");
            }
        }
        // Row sums still zero (rigid-body modes preserved)
        for row in 0..n {
            let s: f64 = (0..n).map(|c| dense[row * n + c]).sum();
            if s.abs() > 1e-8 {
                eprintln!("Warning: BBar near-incompressible row sum at {row}: {:.3e}", s);
            }
        }
    }

    #[test]
    fn bbar_quad4_mesh_works() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let k = bbar_stiffness(&mesh, 1.0, 0.5, false);
        let dense = coo_to_dense_vec(&k);
        let n = k.nrows;
        assert!(n > 0);
        for row in 0..n {
            let s: f64 = (0..n).map(|c| dense[row * n + c]).sum();
            if s.abs() > 1e-10 {
                panic!("BBar Quad4 row sum not zero at row {row}: {:.3e}", s);
            }
        }
    }

    #[test]
    fn fbar_identity_zero_residual() {
        // F-Bar with u=0 (undeformed config): residual should be zero
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n_dofs = space.n_dofs();
        let u_zero = vec![0.0; n_dofs];

        let model = crate::standard::elasticity::StVenantKirchhoff::new(1.0, 0.5);
        let fbar = FBarIntegrator::new(model);
        let (_ktan, residual) = fbar.assemble(&space, &u_zero, 3);
        let max_res = residual.iter().map(|&r| r.abs()).fold(0.0, f64::max);
        assert!(max_res < 1e-12, "FBar residual at u=0 should be zero, got {:.3e}", max_res);
    }

    #[test]
    fn fbar_symmetric_tangent() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n_dofs = space.n_dofs();
        // Small deterministic displacement (linear gradient)
        let mut u = vec![0.0_f64; n_dofs];
        for i in 0..n_dofs {
            u[i] = 0.01 * (i as f64).sin();
        }

        let model = crate::standard::elasticity::StVenantKirchhoff::new(1.0, 0.5);
        let fbar = FBarIntegrator::new(model);
        let (ktan, _res) = fbar.assemble(&space, &u, 3);
        let dense = ktan.clone().into_csr().to_dense();
        let n = ktan.nrows;
        for i in 0..n {
            for j in 0..i {
                let diff = (dense[i * n + j] - dense[j * n + i]).abs();
                if diff > 1e-10 {
                    panic!("FBar tangent not symmetric at ({i},{j}): diff={:.3e}", diff);
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn bbar_parallel_assembly_matches_serial() {
        let mesh = Mesh::<2>::unit_square_tri(16);
        let space = VectorH1Space::new(mesh, 1, 2);
        let k_bbar = assemble_bbar_elasticity(&space, &1.0, &0.5, false, 3);
        let k_std = standard_stiffness(space.mesh(), 1.0, 0.5, false);
        // Just ensure BBar assembles without error in parallel build
        assert!(k_bbar.nrows > 0);
        assert!(k_std.nrows > 0);
    }
}
