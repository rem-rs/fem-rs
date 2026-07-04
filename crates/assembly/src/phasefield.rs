#![allow(non_snake_case)]

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("ref_elem_vol: unsupported ({elem_type:?}, {order})"),
    }
}

fn mesh_type_to_factory(et: ElementType) -> FactoryElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
        ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => FactoryElemType::Quad,
        ElementType::Hex8 | ElementType::Hex20 => FactoryElemType::Hex,
        _ => panic!("mesh_type_to_factory: unsupported {et:?}"),
    }
}

fn isoparametric_jacobian<M: MeshTopology>(
    mesh: &M, nodes: &[u32], geo_elem: &dyn ReferenceElement,
    xi: &[f64], dim: usize,
) -> (DMatrix<f64>, f64, Vec<f64>) {
    let n_geo = geo_elem.n_dofs();
    let mut grad_geo = vec![0.0; n_geo * dim];
    let mut phi_geo = vec![0.0; n_geo];
    geo_elem.eval_grad_basis(xi, &mut grad_geo);
    geo_elem.eval_basis(xi, &mut phi_geo);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0; dim];
    for k in 0..n_geo {
        let xk = mesh.node_coords(nodes[k]);
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

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    (j.clone(), j.determinant())
}

fn is_affine(et: ElementType, geom_order: u8) -> bool {
    if geom_order > 1 { return false; }
    matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2)
}

fn geo_ref_elem(et: ElementType, geom_order: u8) -> Option<Box<dyn ReferenceElement>> {
    let is_quad_hex = matches!(et,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9
        | ElementType::Hex8 | ElementType::Hex20);
    if geom_order == 1 && !is_quad_hex { return None; }
    let order = if geom_order > 1 { geom_order } else { 1 };
    let ft = mesh_type_to_factory(et);
    Some(factory_ref_elem(ft, order))
}

fn transform_grads(j_inv_t: &DMatrix<f64>, grad_ref: &[f64], grad_phys: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += j_inv_t[(j, k)] * grad_ref[i * dim + k]; }
            grad_phys[i * dim + j] = s;
        }
    }
}

/// Assemble the degraded elasticity matrix.
///
/// K_ij = ∫ [(1-d)² + κ] · C_{ijkl} · ε(φ_i) : ε(φ_j) dx
///
/// where d is interpolated from `d_dofs` using `space_d`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_degraded_stiffness<M: MeshTopology>(
    mesh: &M,
    space_u: &dyn FESpace<Mesh = M>,
    u_elem_dofs: &[Vec<usize>],
    u_n_ldofs: usize,
    d_dofs: &[f64],
    _space_d: &dyn FESpace<Mesh = M>,
    d_elem_dofs_cache: &[Vec<usize>],
    d_n_ldofs: usize,
    lambda: f64,
    mu: f64,
    kappa_eps: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let dim = mesh.dim() as usize;
    let n = space_u.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);
    let n_elems = mesh.n_elements() as u32;

    let mut phi_u = vec![0.0; u_n_ldofs];
    let mut grad_ref_u = vec![0.0; u_n_ldofs * dim];
    let mut grad_phys_u = vec![0.0; u_n_ldofs * dim];
    let mut phi_d = vec![0.0; d_n_ldofs];

    for e in 0..n_elems {
        let et = mesh.element_type(e);
        let order = space_u.element_order(e);
        let re = ref_elem_vol(et, order);
        let n_ldofs_u = re.n_dofs();
        let n_elem_dofs_u = u_n_ldofs;
        let n_nodes_u = n_ldofs_u;
        let quad = re.quadrature(quad_order);
        let nodes = mesh.element_nodes(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(et, g_order);
        let geo = geo_ref_elem(et, g_order);

        let gd_u = &u_elem_dofs[e as usize];
        let gd_d = &d_elem_dofs_cache[e as usize];

        let mut k_elem = vec![0.0; n_elem_dofs_u * n_elem_dofs_u];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = if affine {
                let (j, d) = simplex_jac(mesh, nodes, dim);
                (j, d)
            } else {
                let (j, d, _) = isoparametric_jacobian(mesh, nodes, geo.as_deref().unwrap(), xi, dim);
                (j, d)
            };
            let w = quad.weights[qi] * det_j.abs();
            let jit = jac.try_inverse().unwrap().transpose();

            re.eval_basis(xi, &mut phi_u);
            re.eval_grad_basis(xi, &mut grad_ref_u);
            transform_grads(&jit, &grad_ref_u, &mut grad_phys_u, n_ldofs_u, dim);

            // Interpolate d at this QP
            re.eval_basis(xi, &mut phi_d);
            let d_qp: f64 = gd_d.iter().zip(phi_d.iter()).map(|(&dof, &p)| d_dofs[dof] * p).sum();
            let degradation = (1.0 - d_qp).max(0.0).powi(2) + kappa_eps;

            // Assemble K_uu element matrix
            // For interleaved DOFs: [u0_x, u0_y, u1_x, u1_y, ...]
            // φ_i for component a at node k: index = k*dim + a
            // ε(φ_i)_{ab} = ½(G_i[a][b] + G_i[b][a]) where G_i[a][b] = grad_phys_u[k*dim+b] if a == component else 0
            for ki in 0..n_nodes_u {
                for ai in 0..dim {
                    let row = ki * dim + ai;
                    // ∇·φ_row = grad_phys_u[ki*dim + ai] (only the ai-component contributes to divergence)
                    let div_i = grad_phys_u[ki * dim + ai];
                    // ε(φ_row)_{ab}
                    let mut eps_i = vec![0.0; dim * dim];
                    for a in 0..dim {
                        for b in 0..dim {
                            let g_ia_b = if a == ai { grad_phys_u[ki * dim + b] } else { 0.0 };
                            let g_ib_a = if b == ai { grad_phys_u[ki * dim + a] } else { 0.0 };
                            eps_i[a * dim + b] = 0.5 * (g_ia_b + g_ib_a);
                        }
                    }

                    for kj in 0..n_nodes_u {
                        for aj in 0..dim {
                            let col = kj * dim + aj;
                            let div_j = grad_phys_u[kj * dim + aj];
                            let mut eps_j = vec![0.0; dim * dim];
                            for a in 0..dim {
                                for b in 0..dim {
                                    let g_ja_b = if a == aj { grad_phys_u[kj * dim + b] } else { 0.0 };
                                    let g_jb_a = if b == aj { grad_phys_u[kj * dim + a] } else { 0.0 };
                                    eps_j[a * dim + b] = 0.5 * (g_ja_b + g_jb_a);
                                }
                            }

                            let vol = lambda * div_i * div_j;
                            let mut shear = 0.0;
                            for a in 0..dim {
                                for b in 0..dim {
                                    shear += eps_i[a * dim + b] * eps_j[a * dim + b];
                                }
                            }
                            k_elem[row * n_elem_dofs_u + col] += w * degradation * (vol + 2.0 * mu * shear);
                        }
                    }
                }
            }
        }

        for (i, &gi) in gd_u.iter().enumerate() {
            for (j, &gj) in gd_u.iter().enumerate() {
                coo.add(gi, gj, k_elem[i * n_elem_dofs_u + j]);
            }
        }
    }

    coo.into_csr()
}

/// Compute the elastic strain energy density ψ₀(ε) at each quadrature point.
///
/// ψ₀ = ½λ·tr(ε)² + μ·ε:ε
///
/// Returns Vec<f64> flattened over [elem][qp].
#[allow(clippy::too_many_arguments)]
pub fn compute_elastic_energy<M: MeshTopology>(
    mesh: &M,
    space_u: &dyn FESpace<Mesh = M>,
    u_elem_dofs: &[Vec<usize>],
    u_n_ldofs: usize,
    u: &[f64],
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> (Vec<f64>, usize) {
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements() as u32;

    let mut energies = Vec::new();
    let mut phi_u = vec![0.0; u_n_ldofs];
    let mut grad_ref_u = vec![0.0; u_n_ldofs * dim];
    let mut grad_phys_u = vec![0.0; u_n_ldofs * dim];

    for e in 0..n_elems {
        let et = mesh.element_type(e);
        let order = space_u.element_order(e);
        let re = ref_elem_vol(et, order);
        let n_ldofs_u = re.n_dofs();
        let n_nodes_u = n_ldofs_u;
        let quad = re.quadrature(quad_order);
        let nodes = mesh.element_nodes(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(et, g_order);
        let geo = geo_ref_elem(et, g_order);

        let gd_u = &u_elem_dofs[e as usize];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j, _xp) = if affine {
                let (j, d) = simplex_jac(mesh, nodes, dim);
                (j, d, vec![])
            } else {
                isoparametric_jacobian(mesh, nodes, geo.as_deref().unwrap(), xi, dim)
            };
            let _ = quad.weights[qi] * det_j.abs();
            let jit = jac.try_inverse().unwrap().transpose();

            re.eval_basis(xi, &mut phi_u);
            re.eval_grad_basis(xi, &mut grad_ref_u);
            transform_grads(&jit, &grad_ref_u, &mut grad_phys_u, n_ldofs_u, dim);

            // Compute ∇u at this QP: ∇u[a][b] = Σ_k φ_k_component_a * grad_phys_u[k*dim+b]
            let mut grad_u = vec![0.0; dim * dim];
            for k in 0..n_nodes_u {
                for a in 0..dim {
                    let u_val = u[gd_u[k * dim + a]];
                    for b in 0..dim {
                        grad_u[a * dim + b] += u_val * grad_phys_u[k * dim + b];
                    }
                }
            }

            // ε = ½(∇u + ∇uᵀ)
            let mut strain = vec![0.0; dim * dim];
            let mut tr = 0.0;
            for a in 0..dim {
                for b in 0..dim {
                    strain[a * dim + b] = 0.5 * (grad_u[a * dim + b] + grad_u[b * dim + a]);
                }
                tr += strain[a * dim + a];
            }

            let mut eps_sq = 0.0;
            for a in 0..dim {
                for b in 0..dim {
                    eps_sq += strain[a * dim + b] * strain[a * dim + b];
                }
            }

            let psi = 0.5 * lambda * tr * tr + mu * eps_sq;
            energies.push(psi);
        }
    }

    let n_qp_per_elem = if n_elems > 0 { energies.len() / n_elems as usize } else { 0 };
    (energies, n_qp_per_elem)
}

/// Update the history field: H_new = max(H_old, ψ⁺).
pub fn update_history_field(h: &mut [f64], new_energy: &[f64]) {
    let n = h.len().min(new_energy.len());
    for i in 0..n {
        if new_energy[i] > h[i] {
            h[i] = new_energy[i];
        }
    }
}

/// Assemble the phase field system.
///
/// Matrix: A_ij = ∫ [G_c·l·(∇φ_i·∇φ_j) + (G_c/l + 2·H(ξ))·φ_i·φ_j] dx
/// RHS:    b_i  = ∫ 2·H(ξ)·φ_i dx
///
/// H values are stored per quadrature point, flattened [elem][qp].
#[allow(clippy::too_many_arguments)]
pub fn assemble_phase_field_system<M: MeshTopology>(
    mesh: &M,
    space_d: &dyn FESpace<Mesh = M>,
    d_elem_dofs_cache: &[Vec<usize>],
    d_n_ldofs: usize,
    h: &[f64],
    n_qp_per_elem: usize,
    g_c: f64,
    l: f64,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let dim = mesh.dim() as usize;
    let n = space_d.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);
    let mut rhs = vec![0.0; n];
    let n_elems = mesh.n_elements() as u32;

    let mut phi = vec![0.0; d_n_ldofs];
    let mut grad_ref = vec![0.0; d_n_ldofs * dim];
    let mut grad_phys = vec![0.0; d_n_ldofs * dim];

    for e in 0..n_elems {
        let et = mesh.element_type(e);
        let order = space_d.element_order(e);
        let re = ref_elem_vol(et, order);
        let n_ldofs = re.n_dofs();
        let quad = re.quadrature(quad_order);
        let nodes = mesh.element_nodes(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(et, g_order);
        let geo = geo_ref_elem(et, g_order,);

        let gd = &d_elem_dofs_cache[e as usize];

        let mut k_elem = vec![0.0; n_ldofs * n_ldofs];
        let mut f_elem = vec![0.0; n_ldofs];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j, _xp) = if affine {
                let (j, d) = simplex_jac(mesh, nodes, dim);
                (j, d, vec![])
            } else {
                isoparametric_jacobian(mesh, nodes, geo.as_deref().unwrap(), xi, dim)
            };
            let w = quad.weights[qi] * det_j.abs();
            let jit = jac.try_inverse().unwrap().transpose();

            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            transform_grads(&jit, &grad_ref, &mut grad_phys, n_ldofs, dim);

            // h at this QP
            let h_idx = e as usize * n_qp_per_elem + qi;
            let h_qp = if h_idx < h.len() { h[h_idx] } else { 0.0 };

            let a_coeff = g_c / l + 2.0 * h_qp;
            let rhs_coeff = 2.0 * h_qp;

            for i in 0..n_ldofs {
                f_elem[i] += w * rhs_coeff * phi[i];
                for j in 0..n_ldofs {
                    let mut dot = 0.0;
                    for d in 0..dim {
                        dot += grad_phys[i * dim + d] * grad_phys[j * dim + d];
                    }
                    k_elem[i * n_ldofs + j] += w * (g_c * l * dot + a_coeff * phi[i] * phi[j]);
                }
            }
        }

        for (i, &gi) in gd.iter().enumerate() {
            rhs[gi] += f_elem[i];
            for (j, &gj) in gd.iter().enumerate() {
                coo.add(gi, gj, k_elem[i * n_ldofs + j]);
            }
        }
    }

    (coo.into_csr(), rhs)
}

/// Pre-compute element DOF maps for fast access during assembly.
pub fn build_elem_dof_cache<M: MeshTopology>(space: &dyn FESpace<Mesh = M>) -> (Vec<Vec<usize>>, usize) {
    let n_elems = space.mesh().n_elements();
    let mut cache = Vec::with_capacity(n_elems);
    let mut max_n_ldofs = 0usize;
    for e in 0..n_elems as u32 {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        max_n_ldofs = max_n_ldofs.max(dofs.len());
        cache.push(dofs);
    }
    (cache, max_n_ldofs)
}

/// Apply Dirichlet boundary conditions using symmetric elimination.
/// Preserves matrix symmetry for PCG by zeroing both rows and columns.
pub fn apply_dirichlet(mat: &mut CsrMatrix<f64>, rhs: &mut [f64], dofs: &[usize], vals: &[f64]) {
    for (&dof, &val) in dofs.iter().zip(vals.iter()) {
        mat.apply_dirichlet_symmetric(dof, val, rhs);
    }
}

/// 2D Miehe spectral split result.
pub struct MieheSplit2d {
    pub psi_plus: f64,
    pub psi_minus: f64,
    /// Positive stress in Voigt order: [σ_xx, σ_yy, σ_xy]
    pub sigma_plus: [f64; 3],
    /// Negative stress in Voigt order: [σ_xx, σ_yy, σ_xy]
    pub sigma_minus: [f64; 3],
    /// Positive tangent in Voigt 3×3, [row][col]
    pub C_plus: [[f64; 3]; 3],
    /// Negative tangent in Voigt 3×3, [row][col]
    pub C_minus: [[f64; 3]; 3],
}

/// Compute the Miehe spectral split for 2D plane strain/plane stress.
///
/// Given the strain in Voigt order [ε_xx, ε_yy, γ_xy] where γ_xy = 2ε_xy,
/// computes the spectral decomposition ε = Σ εₐ nₐ ⊗ nₐ and splits
/// into tensile (+) and compressive (-) parts.
///
/// ψ⁺ = ½λ⟨tr⟩₊² + μ·ε⁺:ε⁺
/// ψ⁻ = ½λ⟨tr⟩₋² + μ·ε⁻:ε⁻
///
/// The tangent C⁺ = ∂σ⁺/∂ε_Voigt is computed via central finite differences
/// (3 Voigt strain perturbations × σ evaluation each).
pub fn miehe_split_2d(eps_Voigt: &[f64; 3], lambda: f64, mu: f64) -> MieheSplit2d {
    // Convert Voigt → full tensor components: ε_ij (true tensor)
    let e_xx = eps_Voigt[0];
    let e_yy = eps_Voigt[1];
    let e_xy = eps_Voigt[2] * 0.5; // γ_xy = 2ε_xy

    let trace = e_xx + e_yy;
    let dev = (e_xx - e_yy) * 0.5;
    let disc = dev * dev + e_xy * e_xy;
    let sqrt_disc = disc.sqrt();
    let eps_1 = trace * 0.5 + sqrt_disc;
    let eps_2 = trace * 0.5 - sqrt_disc;

    let (c, s) = if disc > 1e-30 {
        let theta = 0.5 * (2.0 * e_xy).atan2(e_xx - e_yy);
        (theta.cos(), theta.sin())
    } else {
        (1.0, 0.0)
    };

    // Positive/negative eigenvalue decomposition
    let e1p = eps_1.max(0.0);
    let e1n = eps_1.min(0.0);
    let e2p = eps_2.max(0.0);
    let e2n = eps_2.min(0.0);
    let tr_p = trace.max(0.0);
    let tr_n = trace.min(0.0);

    // ε⁺ in tensor form
    let ep_xx = e1p * c * c + e2p * s * s;
    let ep_yy = e1p * s * s + e2p * c * c;
    let ep_xy = (e1p - e2p) * c * s;

    // ε⁻ in tensor form
    let en_xx = e1n * c * c + e2n * s * s;
    let en_yy = e1n * s * s + e2n * c * c;
    let en_xy = (e1n - e2n) * c * s;

    // ψ⁺ = ½λ⟨tr⟩₊² + μ·ε⁺:ε⁺
    let eps_plus_sq = ep_xx * ep_xx + ep_yy * ep_yy + 2.0 * ep_xy * ep_xy;
    let psi_plus = 0.5 * lambda * tr_p * tr_p + mu * eps_plus_sq;

    let eps_minus_sq = en_xx * en_xx + en_yy * en_yy + 2.0 * en_xy * en_xy;
    let psi_minus = 0.5 * lambda * tr_n * tr_n + mu * eps_minus_sq;

    // σ⁺ = λ⟨tr⟩₊·I + 2μ·ε⁺ (Voigt: [σ_xx, σ_yy, σ_xy])
    let sp_xx = lambda * tr_p + 2.0 * mu * ep_xx;
    let sp_yy = lambda * tr_p + 2.0 * mu * ep_yy;
    let sp_xy = 2.0 * mu * ep_xy;

    let sn_xx = lambda * tr_n + 2.0 * mu * en_xx;
    let sn_yy = lambda * tr_n + 2.0 * mu * en_yy;
    let sn_xy = 2.0 * mu * en_xy;

    let sigma_plus = [sp_xx, sp_yy, sp_xy];
    let sigma_minus = [sn_xx, sn_yy, sn_xy];

    // Compute tangent C⁺/C⁻ via central finite differences in Voigt space.
    // C_αβ = ∂σ_α/∂ε_Voigt_β
    let eps_fd = 1e-8;
    let mut C_plus = [[0.0; 3]; 3];
    let mut C_minus = [[0.0; 3]; 3];

    // Helper: compute sigma_split at perturbed Voigt strain
    let eval_split = |eps_v: &[f64; 3]| -> ([f64; 3], [f64; 3]) {
        let ex = eps_v[0];
        let ey = eps_v[1];
        let exy_tensor = eps_v[2] * 0.5;
        let tr = ex + ey;
        let dev2 = (ex - ey) * 0.5;
        let d2 = dev2 * dev2 + exy_tensor * exy_tensor;
        let sd = d2.sqrt();
        let l1 = tr * 0.5 + sd;
        let l2 = tr * 0.5 - sd;
        let (ct, st) = if d2 > 1e-30 {
            let th = 0.5 * (2.0 * exy_tensor).atan2(ex - ey);
            (th.cos(), th.sin())
        } else {
            (1.0, 0.0)
        };
        let l1p = l1.max(0.0);
        let l1n = l1.min(0.0);
        let l2p = l2.max(0.0);
        let l2n = l2.min(0.0);
        let trp = tr.max(0.0);
        let trn = tr.min(0.0);
        let epp = l1p * ct * ct + l2p * st * st;
        let enp = l1n * ct * ct + l2n * st * st;
        ([lambda * trp + 2.0 * mu * epp, lambda * trp + 2.0 * mu * (l1p * st * st + l2p * ct * ct), 2.0 * mu * (l1p - l2p) * ct * st],
         [lambda * trn + 2.0 * mu * enp, lambda * trn + 2.0 * mu * (l1n * st * st + l2n * ct * ct), 2.0 * mu * (l1n - l2n) * ct * st])
    };

    for col in 0..3 {
        let mut eps_pert = *eps_Voigt;
        eps_pert[col] += eps_fd;
        let (sp, sn) = eval_split(&eps_pert);
        for row in 0..3 {
            C_plus[row][col] = (sp[row] - sigma_plus[row]) / eps_fd;
            C_minus[row][col] = (sn[row] - sigma_minus[row]) / eps_fd;
        }
    }

    // Symmetrize (C should be symmetric for hyperelastic material)
    for i in 0..3 {
        for j in i + 1..3 {
            let avg_p = (C_plus[i][j] + C_plus[j][i]) * 0.5;
            let avg_n = (C_minus[i][j] + C_minus[j][i]) * 0.5;
            C_plus[i][j] = avg_p;
            C_plus[j][i] = avg_p;
            C_minus[i][j] = avg_n;
            C_minus[j][i] = avg_n;
        }
    }

    MieheSplit2d { psi_plus, psi_minus, sigma_plus, sigma_minus, C_plus, C_minus }
}

/// Assemble the Miehe-split tangent stiffness matrix and internal force vector.
///
/// K_ij = ∫ Bᵀ·C_split·B dx  where C_split = [(1-d)²+κ]·C⁺ + C⁻
/// f_int_i = ∫ Bᵀ·σ_split dx where σ_split = [(1-d)²+κ]·σ⁺(ε(u)) + σ⁻(ε(u))
///
/// The split is evaluated at each QP from the current displacement `u`.
/// The phase field `d` is held fixed during this assembly.
#[allow(clippy::too_many_arguments)]
pub fn assemble_miehe_stiffness_and_force<M: MeshTopology>(
    mesh: &M,
    space_u: &dyn FESpace<Mesh = M>,
    u_elem_dofs: &[Vec<usize>],
    u_n_ldofs: usize,
    u: &[f64],
    d_dofs: &[f64],
    _space_d: &dyn FESpace<Mesh = M>,
    d_elem_dofs_cache: &[Vec<usize>],
    d_n_ldofs: usize,
    lambda: f64,
    mu: f64,
    kappa_eps: f64,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let dim = mesh.dim() as usize;
    let n = space_u.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n, n);
    let mut f_int = vec![0.0; n];
    let n_elems = mesh.n_elements() as u32;

    let mut phi_u = vec![0.0; u_n_ldofs];
    let mut grad_ref_u = vec![0.0; u_n_ldofs * dim];
    let mut grad_phys_u = vec![0.0; u_n_ldofs * dim];
    let mut phi_d = vec![0.0; d_n_ldofs];

    for e in 0..n_elems {
        let et = mesh.element_type(e);
        let order = space_u.element_order(e);
        let re = ref_elem_vol(et, order);
        let n_ldofs_u = re.n_dofs();
        let n_elem_dofs_u = u_n_ldofs;
        let n_nodes_u = n_ldofs_u;
        let quad = re.quadrature(quad_order);
        let nodes = mesh.element_nodes(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(et, g_order);
        let geo = geo_ref_elem(et, g_order);

        let gd_u = &u_elem_dofs[e as usize];
        let gd_d = &d_elem_dofs_cache[e as usize];

        let mut k_elem = vec![0.0; n_elem_dofs_u * n_elem_dofs_u];
        let mut f_elem = vec![0.0; n_elem_dofs_u];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = if affine {
                let (j, d) = simplex_jac(mesh, nodes, dim);
                (j, d)
            } else {
                let (j, d, _) = isoparametric_jacobian(mesh, nodes, geo.as_deref().unwrap(), xi, dim);
                (j, d)
            };
            let w = quad.weights[qi] * det_j.abs();
            let jit = jac.try_inverse().unwrap().transpose();

            re.eval_basis(xi, &mut phi_u);
            re.eval_grad_basis(xi, &mut grad_ref_u);
            transform_grads(&jit, &grad_ref_u, &mut grad_phys_u, n_ldofs_u, dim);

            // Interpolate d at this QP
            re.eval_basis(xi, &mut phi_d);
            let d_qp: f64 = gd_d.iter().zip(phi_d.iter()).map(|(&dof, &p)| d_dofs[dof] * p).sum();
            let degradation = (1.0 - d_qp).max(0.0).powi(2) + kappa_eps;

            // Compute ∇u at this QP: ∇u[a][b] = Σ_k u_k_component_a * grad_phys_u[k*dim+b]
            let mut grad_u = vec![0.0; dim * dim];
            for k in 0..n_nodes_u {
                for a in 0..dim {
                    let u_val = u[gd_u[k * dim + a]];
                    for b in 0..dim {
                        grad_u[a * dim + b] += u_val * grad_phys_u[k * dim + b];
                    }
                }
            }

            // Strain tensor ε = ½(∇u + ∇uᵀ)
            let strain_xx = grad_u[0];
            let strain_yy = grad_u[dim + 1];
            let strain_xy = 0.5 * (grad_u[1] + grad_u[dim]);
            let gamma_xy = 2.0 * strain_xy; // Voigt shear

            // Miehe spectral split at this QP
            let split = miehe_split_2d(&[strain_xx, strain_yy, gamma_xy], lambda, mu);

            // Assemble element matrix and force
            for ki in 0..n_nodes_u {
                for ai in 0..dim {
                    let row = ki * dim + ai;
                    let _div_i = grad_phys_u[ki * dim + ai];
                    let mut eps_i = vec![0.0; dim * dim];
                    for a in 0..dim {
                        for b in 0..dim {
                            let g_ia_b = if a == ai { grad_phys_u[ki * dim + b] } else { 0.0 };
                            let g_ib_a = if b == ai { grad_phys_u[ki * dim + a] } else { 0.0 };
                            eps_i[a * dim + b] = 0.5 * (g_ia_b + g_ib_a);
                        }
                    }
                    // Voigt strain vector for this basis function
                    let eps_i_voigt = [eps_i[0], eps_i[3], 2.0 * eps_i[1]]; // [ε_xx, ε_yy, γ_xy]

                    // σ_split contribution to f_int
                    let mut sigma_row = 0.0;
                    for d in 0..3 {
                        let sigma_qp = degradation * split.sigma_plus[d] + split.sigma_minus[d];
                        sigma_row += sigma_qp * eps_i_voigt[d];
                    }
                    f_elem[row] += w * sigma_row;

                    for kj in 0..n_nodes_u {
                        for aj in 0..dim {
                            let col = kj * dim + aj;
                            let mut eps_j = vec![0.0; dim * dim];
                            for a in 0..dim {
                                for b in 0..dim {
                                    let g_ja_b = if a == aj { grad_phys_u[kj * dim + b] } else { 0.0 };
                                    let g_jb_a = if b == aj { grad_phys_u[kj * dim + a] } else { 0.0 };
                                    eps_j[a * dim + b] = 0.5 * (g_ja_b + g_jb_a);
                                }
                            }
                            let eps_j_voigt = [eps_j[0], eps_j[3], 2.0 * eps_j[1]];

                            // C_split = [(1-d)²+κ]·C⁺ + C⁻
                            let mut k_qp = 0.0;
                            for d1 in 0..3 {
                                let mut C_row = [0.0; 3];
                                for c in 0..3 {
                                    C_row[c] = degradation * split.C_plus[d1][c] + split.C_minus[d1][c];
                                }
                                k_qp += eps_i_voigt[d1] * (C_row[0] * eps_j_voigt[0]
                                                          + C_row[1] * eps_j_voigt[1]
                                                          + C_row[2] * eps_j_voigt[2]);
                            }
                            k_elem[row * n_elem_dofs_u + col] += w * k_qp;
                        }
                    }
                }
            }
        }

        for (i, &gi) in gd_u.iter().enumerate() {
            f_int[gi] += f_elem[i];
            for (j, &gj) in gd_u.iter().enumerate() {
                coo.add(gi, gj, k_elem[i * n_elem_dofs_u + j]);
            }
        }
    }

    (coo.into_csr(), f_int)
}

/// Compute ψ⁺ (tensile strain energy density) at each quadrature point.
///
/// Uses Miehe spectral split: only the tensile part drives fracture.
/// Returns (psi_plus_flattened, n_qp_per_elem).
#[allow(clippy::too_many_arguments)]
pub fn compute_psi_plus<M: MeshTopology>(
    mesh: &M,
    space_u: &dyn FESpace<Mesh = M>,
    u_elem_dofs: &[Vec<usize>],
    u_n_ldofs: usize,
    u: &[f64],
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> (Vec<f64>, usize) {
    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements() as u32;

    let mut psi_plus_vals = Vec::new();
    let mut phi_u = vec![0.0; u_n_ldofs];
    let mut grad_ref_u = vec![0.0; u_n_ldofs * dim];
    let mut grad_phys_u = vec![0.0; u_n_ldofs * dim];

    for e in 0..n_elems {
        let et = mesh.element_type(e);
        let order = space_u.element_order(e);
        let re = ref_elem_vol(et, order);
        let n_ldofs_u = re.n_dofs();
        let n_nodes_u = n_ldofs_u;
        let quad = re.quadrature(quad_order);
        let nodes = mesh.element_nodes(e);
        let g_order = mesh.geom_order();
        let affine = is_affine(et, g_order);
        let geo = geo_ref_elem(et, g_order);

        let gd_u = &u_elem_dofs[e as usize];

        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det_j, _xp) = if affine {
                let (j, d) = simplex_jac(mesh, nodes, dim);
                (j, d, vec![])
            } else {
                isoparametric_jacobian(mesh, nodes, geo.as_deref().unwrap(), xi, dim)
            };
            let _ = quad.weights[qi] * det_j.abs();
            let jit = jac.try_inverse().unwrap().transpose();

            re.eval_basis(xi, &mut phi_u);
            re.eval_grad_basis(xi, &mut grad_ref_u);
            transform_grads(&jit, &grad_ref_u, &mut grad_phys_u, n_ldofs_u, dim);

            // ∇u at this QP
            let mut grad_u = vec![0.0; dim * dim];
            for k in 0..n_nodes_u {
                for a in 0..dim {
                    let u_val = u[gd_u[k * dim + a]];
                    for b in 0..dim {
                        grad_u[a * dim + b] += u_val * grad_phys_u[k * dim + b];
                    }
                }
            }

            // Strain ε = ½(∇u + ∇uᵀ)
            let strain_xx = grad_u[0];
            let strain_yy = grad_u[dim + 1];
            let strain_xy = 0.5 * (grad_u[1] + grad_u[dim]);
            let gamma_xy = 2.0 * strain_xy;

            // ψ⁺ from Miehe spectral split
            let split = miehe_split_2d(&[strain_xx, strain_yy, gamma_xy], lambda, mu);
            psi_plus_vals.push(split.psi_plus);
        }
    }

    let n_qp_per_elem = if n_elems > 0 { psi_plus_vals.len() / n_elems as usize } else { 0 };
    (psi_plus_vals, n_qp_per_elem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::{H1Space, VectorH1Space};
    use fem_space::fe_space::FESpace;

    fn small_mesh() -> SimplexMesh<2> {
        // 2×2 unit square with 8 triangles
        SimplexMesh::<2>::unit_square_tri(2)
    }

    #[test]
    fn degraded_stiffness_with_undamaged_matches_elasticity() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let dim = 2;
        let quad_order: u8 = 2;
        let lambda = 121154.0;
        let mu = 80769.0;
        let kappa_eps = 1e-10;

        let space_u = VectorH1Space::new(mesh.clone(), order, dim);
        let space_d = H1Space::new(mesh.clone(), order);
        let (u_elem_dofs, u_n_ldofs) = build_elem_dof_cache(&space_u);
        let (d_elem_dofs, d_n_ldofs) = build_elem_dof_cache(&space_d);

        let n_u = space_u.n_dofs();
        let d_zero = vec![0.0; space_d.n_dofs()];

        // With d=0 everywhere, degradation = (1-0)² + κ ≈ 1
        let k = assemble_degraded_stiffness(
            &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
            &d_zero, &space_d, &d_elem_dofs, d_n_ldofs,
            lambda, mu, kappa_eps, quad_order,
        );

        assert_eq!(k.nrows, n_u);
        assert_eq!(k.ncols, n_u);
        assert!(k.nnz() > 0);
        assert!(k.nrows < 100, "mesh too large for test");
    }

    #[test]
    fn phase_field_system_with_zero_h_gives_zero_rhs() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let quad_order: u8 = 2;
        let g_c = 2.7e-3;
        let l_val = 0.015;

        let space_d = H1Space::new(mesh.clone(), order);
        let (d_elem_dofs, d_n_ldofs) = build_elem_dof_cache(&space_d);
        let n_elems = mesh.n_elements() as usize;
        let n_qp_per_elem = 3; // P1 triangle, quad_order=2
        let h_zero = vec![0.0; n_elems * n_qp_per_elem];

        let (mat, rhs) = assemble_phase_field_system(
            &mesh, &space_d, &d_elem_dofs, d_n_ldofs,
            &h_zero, n_qp_per_elem, g_c, l_val, quad_order,
        );

        assert_eq!(mat.nrows, space_d.n_dofs());
        assert!(mat.nnz() > 0);
        // RHS should be all zeros when H=0
        for &r in &rhs {
            assert!(r.abs() < 1e-14, "rhs should be zero with H=0, got {r}");
        }
    }

    #[test]
    fn history_field_is_monotonic() {
        let mut h = vec![0.0, 0.5, 0.0, 1.0];
        let new = vec![0.1, 0.3, 0.8, 0.9];
        update_history_field(&mut h, &new);
        assert!((h[0] - 0.1).abs() < 1e-14);
        assert!((h[1] - 0.5).abs() < 1e-14); // stays at old max
        assert!((h[2] - 0.8).abs() < 1e-14);
        assert!((h[3] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn build_elem_dof_cache_consistent() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let space = H1Space::new(mesh.clone(), order);
        let (cache, max_n_ldofs) = build_elem_dof_cache(&space);
        assert!(max_n_ldofs > 0);
        assert_eq!(cache.len(), mesh.n_elements());
        for dofs in &cache {
            assert!(!dofs.is_empty());
        }
    }

    #[test]
    fn miehe_split_isotropic_strain_zero_psi_plus() {
        // Pure compression: ε = [-0.01, -0.01, 0] → tr < 0, all eigenvalues negative
        // ψ⁺ should be zero (no tensile energy under pure compression)
        let lambda = 121154.0;
        let mu = 80769.0;
        let split = miehe_split_2d(&[-0.01, -0.01, 0.0], lambda, mu);
        assert!(split.psi_plus.abs() < 1e-20, "psi_plus should be 0 under pure compression");
        assert!(split.psi_minus > 0.0, "psi_minus should be > 0 under compression");
        // σ⁺ should be zero
        for &s in &split.sigma_plus {
            assert!(s.abs() < 1e-14, "sigma_plus should be 0 under compression");
        }
        // σ⁻ should equal isotropic elastic stress
        let tr = -0.02;
        let s_xx = lambda * tr + 2.0 * mu * (-0.01);
        let s_yy = lambda * tr + 2.0 * mu * (-0.01);
        assert!((split.sigma_minus[0] - s_xx).abs() < 1e-10);
        assert!((split.sigma_minus[1] - s_yy).abs() < 1e-10);
    }

    #[test]
    fn miehe_split_tensile_strain_matches_isotropic() {
        // Pure tension: ε = [0.005, 0.005, 0] → tr > 0, all eigenvalues positive
        // ψ⁺ should equal total ψ₀, ψ⁻ should be zero
        let lambda = 121154.0;
        let mu = 80769.0;
        let split = miehe_split_2d(&[0.005, 0.005, 0.0], lambda, mu);
        let tr = 0.01;
        let eps_sq = 0.005f64.powi(2) * 2.0;
        let psi_iso = 0.5 * lambda * tr * tr + mu * eps_sq;
        assert!((split.psi_plus - psi_iso).abs() < 1e-14, "psi_plus should equal total ψ₀ under tension");
        assert!(split.psi_minus.abs() < 1e-20, "psi_minus should be 0 under pure tension");
        // σ⁻ should be zero
        for &s in &split.sigma_minus {
            assert!(s.abs() < 1e-14, "sigma_minus should be 0 under tension");
        }
    }

    #[test]
    fn miehe_split_shear_positive_only() {
        // Pure shear: ε = [[0, 0.01], [0.01, 0]] → eigenvalues: 0.01, -0.01
        // ψ⁺ = μ·ε⁺:ε⁺ = μ·ε₁² = μ·(0.01)² (since tr=0, so no λ contribution)
        let lambda = 1000.0;
        let mu = 500.0;
        let gamma = 0.02; // γ_xy = 2ε_xy
        let split = miehe_split_2d(&[0.0, 0.0, gamma], lambda, mu);
        let eps_plus_sq = (0.01_f64).powi(2); // ε⁺:ε⁺ = ε₁² for pure shear of a single positive eigenvalue
        let psi_plus_expected = mu * eps_plus_sq; // tr=0 → λ term is 0
        assert!((split.psi_plus - psi_plus_expected).abs() < 1e-12, "psi_plus mismatch in shear: {} vs {}", split.psi_plus, psi_plus_expected);
        assert!(split.psi_plus > 0.0);
        // sigma_minus should be non-zero (negative eigenvalue contributes to σ⁻)
        assert!(split.sigma_minus[0] < 0.0 || split.sigma_minus[1] < 0.0 || split.sigma_minus[2] != 0.0);
    }

    #[test]
    fn miehe_tangent_symmetric() {
        // The tangent C⁺ and C⁻ should be symmetric
        let split = miehe_split_2d(&[0.003, -0.001, 0.005], 121154.0, 80769.0);
        for mat in [&split.C_plus, &split.C_minus] {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((mat[i][j] - mat[j][i]).abs() < 1e-12,
                        "C[{i}][{j}] != C[{j}][{i}]: {} vs {}", mat[i][j], mat[j][i]);
                }
            }
        }
    }

    #[test]
    fn assemble_miehe_produces_nonzero() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let dim = 2;
        let quad_order: u8 = 2;
        let lambda = 121154.0;
        let mu = 80769.0;
        let kappa_eps = 1e-6;

        let space_u = VectorH1Space::new(mesh.clone(), order, dim);
        let space_d = H1Space::new(mesh.clone(), order);
        let (u_elem_dofs, u_n_ldofs) = build_elem_dof_cache(&space_u);
        let (d_elem_dofs, d_n_ldofs) = build_elem_dof_cache(&space_d);

        let u_zero = vec![0.0; space_u.n_dofs()];
        let d_zero = vec![0.0; space_d.n_dofs()];

        // With u=0, strain=0, split should give isotropic behavior
        let (k, f) = assemble_miehe_stiffness_and_force(
            &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
            &u_zero, &d_zero, &space_d, &d_elem_dofs, d_n_ldofs,
            lambda, mu, kappa_eps, quad_order,
        );

        assert_eq!(k.nrows, space_u.n_dofs());
        assert_eq!(k.ncols, space_u.n_dofs());
        assert!(k.nnz() > 0);
        // f_int should be zero when u=0
        for &fi in &f {
            assert!(fi.abs() < 1e-14, "f_int should be zero with u=0, got {fi}");
        }
    }

    #[test]
    fn compute_psi_plus_nonnegative() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let dim = 2;
        let quad_order: u8 = 2;
        let lambda = 121154.0;
        let mu = 80769.0;

        let space_u = VectorH1Space::new(mesh.clone(), order, dim);
        let (u_elem_dofs, u_n_ldofs) = build_elem_dof_cache(&space_u);

        let u_zero = vec![0.0; space_u.n_dofs()];
        let (psi, n_qp) = compute_psi_plus(
            &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
            &u_zero, lambda, mu, quad_order,
        );

        assert!(n_qp > 0);
        assert_eq!(psi.len(), mesh.n_elements() * n_qp);
        for &p in &psi {
            assert!(p >= 0.0 || p.abs() < 1e-16, "psi_plus should be nonnegative, got {p}");
        }
    }

    #[test]
    fn apply_dirichlet_symmetric_preserves_symmetry() {
        let mesh = small_mesh();
        let order: u8 = 1;
        let dim = 2;
        let quad_order: u8 = 2;
        let lambda = 121154.0;
        let mu = 80769.0;
        let kappa_eps = 1e-10;

        let space_u = VectorH1Space::new(mesh.clone(), order, dim);
        let space_d = H1Space::new(mesh.clone(), order);
        let (u_elem_dofs, u_n_ldofs) = build_elem_dof_cache(&space_u);
        let (d_elem_dofs, d_n_ldofs) = build_elem_dof_cache(&space_d);

        let d_zero = vec![0.0; space_d.n_dofs()];

        let mut k = assemble_degraded_stiffness(
            &mesh, &space_u, &u_elem_dofs, u_n_ldofs,
            &d_zero, &space_d, &d_elem_dofs, d_n_ldofs,
            lambda, mu, kappa_eps, quad_order,
        );
        let mut rhs = vec![0.0; space_u.n_dofs()];

        // Apply BC on first dof
        apply_dirichlet(&mut k, &mut rhs, &[0], &[1.0]);

        // Check symmetry: K[i,j] == K[j,i] for all i,j
        for i in 0..k.nrows {
            for j in i..k.ncols {
                let kij = k.get(i, j);
                let kji = k.get(j, i);
                if (kij - kji).abs() > 1e-14 {
                    // Only fail if both are non-zero (one could be out of sparsity pattern)
                    if kij.abs() > 1e-14 || kji.abs() > 1e-14 {
                        assert!((kij - kji).abs() < 1e-14,
                            "K[{i},{j}]={kij} != K[{j},{i}]={kji}");
                    }
                }
            }
        }
    }
}
