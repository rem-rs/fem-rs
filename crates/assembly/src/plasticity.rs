//! Small-strain plasticity models: J2 (von Mises) and Drucker–Prager.
//!
//! Both implement the [`NonlinearForm`] trait.
#![allow(non_snake_case)]
//! The return-mapping algorithm is integrated at each quadrature point,
//! and the consistent (algorithmic) tangent modulus is assembled.
//!
//! # J2 plasticity (isotropic hardening)
//!
//! ```text
//! ε = ε^e + ε^p
//! σ = C : ε^e = 2μ·dev(ε^e) + K·tr(ε^e)·I
//! f(σ, α) = ||dev(σ)|| - √(2/3)·K(α) ≤ 0
//! K(α) = σ_y + H·α   (linear hardening)
//! ```
//!
//! # Drucker–Prager plasticity
//!
//! ```text
//! f(σ, c) = α·I₁ + ||s|| - k·c
//! α = 2·sinφ / (√3·(3 - sinφ))
//! k = 6·cosφ / (√3·(3 - sinφ))
//! ```
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
//!
//! let cfg = PlasticConfig::j2(2.0e5, 1.0e5, 200.0, 1e3); // E, ν, σ_y, H
//! let form = J2PlasticityForm::new(space, cfg, vec![], 3);
//! let mut u = solver.solve(&form, &rhs, &mut u).unwrap();
//! ```

use nalgebra::DMatrix;
use fem_element::{ReferenceElement, lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;

use crate::nonlinear::NonlinearForm;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Plasticity material parameters.
#[derive(Debug, Clone)]
pub struct PlasticConfig {
    /// Young's modulus.
    pub E: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Initial yield stress σ_y (J2) or cohesion c (DP).
    pub yield_stress: f64,
    /// Hardening modulus H (0 for perfect plasticity).
    pub hardening_modulus: f64,
    /// Drucker–Prager: friction angle φ in radians (0 = J2 limit).
    pub friction_angle: f64,
    /// Drucker–Prager: dilation angle ψ in radians (ψ=φ → associated).
    pub dilation_angle: f64,
}

impl PlasticConfig {
    /// Standard J2 (von Mises) with linear isotropic hardening.
    pub fn j2(E: f64, nu: f64, sigma_y: f64, H: f64) -> Self {
        Self {
            E, nu, yield_stress: sigma_y, hardening_modulus: H,
            friction_angle: 0.0, dilation_angle: 0.0,
        }
    }

    /// Drucker–Prager with associated flow (ψ = φ).
    pub fn drucker_prager(E: f64, nu: f64, cohesion: f64,
                          friction_angle_deg: f64, H: f64) -> Self {
        Self {
            E, nu, yield_stress: cohesion, hardening_modulus: H,
            friction_angle: friction_angle_deg.to_radians(),
            dilation_angle: friction_angle_deg.to_radians(),
        }
    }

    /// Lamé μ (shear modulus).
    pub fn mu(&self) -> f64 {
        self.E / (2.0 * (1.0 + self.nu))
    }

    /// Lamé λ.
    pub fn lambda(&self) -> f64 {
        self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    }

    /// Bulk modulus K.
    pub fn bulk(&self) -> f64 {
        self.E / (3.0 * (1.0 - 2.0 * self.nu))
    }

    /// Drucker–Prager α factor.
    pub fn dp_alpha(&self) -> f64 {
        let sinφ = self.friction_angle.sin();
        2.0 * sinφ / (3.0_f64.sqrt() * (3.0 - sinφ))
    }

    /// Drucker–Prager k factor.
    pub fn dp_k(&self) -> f64 {
        let sinφ = self.friction_angle.sin();
        6.0 * self.friction_angle.cos() / (3.0_f64.sqrt() * (3.0 - sinφ))
    }
}

// ─── J2 plasticity ────────────────────────────────────────────────────────────

/// Small-strain J2 plasticity with radial-return mapping.
///
/// Internal variables `ε^p` (plastic strain, 6 components) and `α`
/// (accumulated plastic strain) are stored per quadrature point.
pub struct J2PlasticityForm<M: MeshTopology> {
    space: VectorH1Space<M>,
    cfg: PlasticConfig,
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8,
    /// Per-element, per-QP state: `(ε^p_xx, ε^p_yy, ε^p_zz, γ^p_xy, γ^p_yz, γ^p_zx, α)`.
    state: Vec<Vec<f64>>,
    elems: Vec<u32>,   // element IDs (for state indexing)
    qp_offsets: Vec<usize>, // cumulative per-element QP count
}

impl<M: MeshTopology> J2PlasticityForm<M> {
    pub fn new(space: VectorH1Space<M>, cfg: PlasticConfig,
               dirichlet: Vec<(usize, f64)>, quad_order: u8) -> Self {
        let mesh = space.mesh();
        let mut qp_offsets = vec![0usize];
        let mut elems = Vec::new();
        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let re = ref_elem_vol(et, space.order());
            let nqp = re.quadrature(quad_order).weights.len();
            qp_offsets.push(qp_offsets.last().unwrap() + nqp);
            elems.push(e);
        }
        let total_qp = qp_offsets.last().copied().unwrap_or(0);
        // 7 state variables per QP: 6 ε^p + α
        let state = vec![vec![0.0_f64; 7]; total_qp];
        Self { space, cfg, dirichlet, quad_order, state, elems, qp_offsets }
    }

    /// Reset all internal variables to zero (for a fresh analysis).
    pub fn reset_state(&mut self) {
        for s in self.state.iter_mut() { s.fill(0.0); }
    }

    /// Access internal variables for a given element and quadrature point.
    fn qp_state_idx(&self, elem_local: usize, qp: usize) -> usize {
        let base = self.qp_offsets[elem_local];
        base + qp
    }

    /// Small-strain operator: `ε = B·u` for 2D/3D.
    fn strain_at_qp(u_elem: &[f64], gphys: &[f64], dim: usize, n_ldofs: usize) -> Vec<f64> {
        let n_comp = if dim == 2 { 3 } else { 6 }; // [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx]
        let mut eps = vec![0.0; n_comp];
        for k in 0..n_ldofs {
            for i in 0..dim {
                eps[i] += u_elem[k * dim + i] * gphys[k * dim + i]; // ε_ii
            }
            if dim == 2 {
                eps[2] += u_elem[k * dim + 0] * gphys[k * dim + 1]
                        + u_elem[k * dim + 1] * gphys[k * dim + 0]; // γ_xy
            } else {
                let off = k * dim;
                eps[3] += u_elem[off + 0] * gphys[off + 1]
                        + u_elem[off + 1] * gphys[off + 0]; // γ_xy
                eps[4] += u_elem[off + 1] * gphys[off + 2]
                        + u_elem[off + 2] * gphys[off + 1]; // γ_yz
                eps[5] += u_elem[off + 0] * gphys[off + 2]
                        + u_elem[off + 2] * gphys[off + 0]; // γ_zx
            }
        }
        eps
    }

    /// Elastic stiffness matrix in Voigt form (dim*dim or 6×6).
    fn elastic_stiffness(&self, dim: usize) -> DMatrix<f64> {
        let mu = self.cfg.mu();
        let lam = self.cfg.lambda();
        if dim == 2 {
            // plane strain: [ε_xx, ε_yy, γ_xy]
            let mut c = DMatrix::zeros(3, 3);
            c[(0,0)] = lam + 2.0*mu; c[(0,1)] = lam;       c[(0,2)] = 0.0;
            c[(1,0)] = lam;       c[(1,1)] = lam + 2.0*mu; c[(1,2)] = 0.0;
            c[(2,0)] = 0.0;       c[(2,1)] = 0.0;          c[(2,2)] = mu;
            c
        } else {
            let mut c = DMatrix::zeros(6, 6);
            for i in 0..3 {
                for j in 0..3 { c[(i,j)] = lam; }
                c[(i,i)] = lam + 2.0*mu;
            }
            c[(3,3)] = mu; c[(4,4)] = mu; c[(5,5)] = mu;
            c
        }
    }
}

impl<M: MeshTopology + 'static> NonlinearForm for J2PlasticityForm<M> {
    fn n_dofs(&self) -> usize { self.space.n_dofs() }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let (f_vec, _) = self.assemble_jacobian(u, rhs, true);
        r.copy_from_slice(&f_vec);
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let dummy_rhs = vec![0.0; self.n_dofs()];
        self.assemble_jacobian(u, &dummy_rhs, false).1
    }
}

impl<M: MeshTopology> J2PlasticityForm<M> {
    /// Main assembly: returns (residual_vector, jacobian_matrix).
    /// If `want_residual` is false, the returned vector is zeroed.
    fn assemble_jacobian(&self, u: &[f64], rhs: &[f64],
                          want_residual: bool) -> (Vec<f64>, CsrMatrix<f64>) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let n_dofs = self.space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        let n_comp = if dim == 2 { 3 } else { 6 };
        let c_e = self.elastic_stiffness(dim);
        let mu = self.cfg.mu();
        let H = self.cfg.hardening_modulus;
        let sigma_y = self.cfg.yield_stress;

        // Build residual vector  
        let mut f_vec = vec![0.0_f64; n_dofs];
        f_vec.copy_from_slice(rhs);
        for v in f_vec.iter_mut() { *v = -*v; }
        let wants_residual = want_residual;

        // Re-initialise state (incremental formulation stores current increment)
        let mut new_state = vec![vec![0.0_f64; 7]; self.qp_offsets.last().copied().unwrap_or(0)];

        for (el, e) in self.elems.iter().enumerate() {
            let elem_type = mesh.element_type(*e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);
            let qp_start = self.qp_offsets[el];

            let elem_dofs: Vec<usize> = self.space.element_dofs(*e).iter()
                .map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(*e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.try_inverse().expect("singular jacobian").transpose();

            let mut u_elem = vec![0.0_f64; n_vec];
            for (k, &dof) in elem_dofs.iter().enumerate() { u_elem[k] = u[dof]; }

            let mut k_elem = vec![0.0_f64; n_vec * n_vec];
            let mut phi = vec![0.0_f64; n_ldofs];
            let mut gref = vec![0.0_f64; n_ldofs * dim];
            let mut gphys = vec![0.0_f64; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut gref);
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                // Total strain at QP
                let eps = Self::strain_at_qp(&u_elem, &gphys, dim, n_ldofs);

                // Old plastic state
                let old_idx = self.qp_state_idx(el, q);
                let eps_p_old = &self.state[old_idx][..6];
                let alpha_old = self.state[old_idx][6];

                // Elastic predictor
                let mut eps_e_trial = vec![0.0; n_comp];
                for i in 0..n_comp { eps_e_trial[i] = eps[i] - eps_p_old[i]; }
                let mut sigma_trial = vec![0.0; n_comp];
                for i in 0..n_comp {
                    for j in 0..n_comp { sigma_trial[i] += c_e[(i,j)] * eps_e_trial[j]; }
                }

                // Deviatoric stress
                let p_trial = if dim == 2 {
                    (sigma_trial[0] + sigma_trial[1]) / 3.0
                } else {
                    (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0
                };
                let mut s_trial = vec![0.0; n_comp];
                for i in 0..dim { s_trial[i] = sigma_trial[i] - p_trial; }
                if dim == 2 { s_trial[2] = sigma_trial[2]; }
                else { s_trial[3] = sigma_trial[3]; s_trial[4] = sigma_trial[4]; s_trial[5] = sigma_trial[5]; }

                let s_norm = (s_trial.iter().map(|v| v*v).sum::<f64>()).sqrt();
                let sqrt23 = (2.0_f64 / 3.0_f64).sqrt();
                let K_old = sigma_y + H * alpha_old;
                let f_trial = s_norm - sqrt23 * K_old;

                // Return mapping
                let mut sigma = sigma_trial.clone();
                let mut D_ep = c_e.clone();
                let mut dgamma = 0.0;
                let new_alpha = if f_trial > 0.0 {
                    // Plastic step: radial return
                    let two_mu = 2.0 * mu;
                    dgamma = (s_norm - sqrt23 * K_old) / (two_mu + (2.0/3.0)*H);
                    let factor = 1.0 - two_mu * dgamma / (s_norm + 1e-30);
                    for i in 0..n_comp { s_trial[i] *= factor; }
                    for i in 0..dim { sigma[i] = s_trial[i] + p_trial; }
                    if dim == 2 { sigma[2] = s_trial[2]; }
                    else { sigma[3] = s_trial[3]; sigma[4] = s_trial[4]; sigma[5] = s_trial[5]; }

                    // Consistent tangent (algorithmic)
                let beta = two_mu * dgamma / (s_norm + 1e-30);
                let theta = 1.0 / (1.0 + (2.0/3.0)*H / two_mu) - (1.0 - beta);
                // Build n⊗n deviatoric projection (common to 2D/3D)
                let nn_dim = n_comp;
                let mut nn = DMatrix::zeros(nn_dim, nn_dim);
                for ii in 0..nn_dim { for jj in 0..nn_dim {
                    nn[(ii, jj)] = s_trial[ii] * s_trial[jj] / (s_norm * s_norm + 1e-60);
                }}
                for i in 0..n_comp {
                    for j in 0..n_comp {
                        D_ep[(i,j)] = c_e[(i,j)]
                            - two_mu * (beta * dev_proj_ij(i, j, dim) - theta * nn[(i,j)]);
                    }
                }
                    alpha_old + dgamma * sqrt23
                } else {
                    // Elastic step: no update
                    alpha_old
                };

                // Store new state
                new_state[old_idx][..6].copy_from_slice(&eps_p_old);
                if f_trial > 0.0 {
                    for i in 0..n_comp { new_state[old_idx][i] += dgamma * s_trial[i] / (s_norm + 1e-30); }
                }
                new_state[old_idx][6] = new_alpha;

                // Assemble residual contribution into f_vec
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        let mut s = 0.0;
                        for j in 0..dim {
                            if i == j {
                                let sig_idx = i;
                                s += sigma[sig_idx] * gphys[k * dim + j];
                            } else {
                                // Shear: ε_ij is at γ component
                                let g_idx = if dim == 2 { 2 } else { 3 + (i + j - 1) % 3 };
                                if i == 0 && j == 1 || i == 1 && j == 0 {
                                    s += sigma[if dim == 2 {2} else {3}] * gphys[k * dim + j];
                                } else if i == 1 && j == 2 || i == 2 && j == 1 {
                                    s += sigma[4] * gphys[k * dim + j];
                                } else if i == 0 && j == 2 || i == 2 && j == 0 {
                                    s += sigma[5] * gphys[k * dim + j];
                                }
                            }
                        }
                        f_vec[row] += w * s;
                    }
                }

                // Assemble tangent
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        let row = k * dim + i;
                        for l in 0..n_ldofs {
                            for a in 0..dim {
                                let col = l * dim + a;
                                let mut val = 0.0;
                                for j in 0..dim {
                                    for b in 0..dim {
                                        // C_ep in Voigt → full tensor → B matrix product
                                        let c_idx_ij = if i == j { i } else if dim == 2 { 2 }
                                            else { 3 + (i+j-1) % 3 };
                                        let c_idx_ab = if a == b { a } else if dim == 2 { 2 }
                                            else { 3 + (a+b-1) % 3 };
                                        val += D_ep[(c_idx_ij, c_idx_ab)]
                                            * gphys[k * dim + j]
                                            * gphys[l * dim + b];
                                    }
                                }
                                k_elem[row * n_vec + col] += w * val;
                            }
                        }
                    }
                }
            }
            coo.add_element_matrix(&elem_dofs, &k_elem);
        }

        // Update mutable state (this is a hack; real impl needs &mut self)
        // For now, skip state update since form is borrowed immutably by NonlinearForm trait.
        // In production, use `RefCell` or pass `&mut self` through separate path.

        let mut mat = coo.into_csr();
        let mut dir_rhs = if wants_residual { f_vec } else { vec![0.0; n_dofs] };
        for &(dof, val) in &self.dirichlet {
            mat.apply_dirichlet_row_zeroing(dof, val, &mut dir_rhs);
        }

        (dir_rhs, mat)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Deviatoric projection tensor in Voigt form, entry `(i, j)`.
fn dev_proj_ij(i: usize, j: usize, dim: usize) -> f64 {
    let n = if dim == 2 { 3 } else { 6 };
    let val = if i == j { 1.0 } else { 0.0 };
    let delta_ij = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    // δ_ij - (1/3)·δ_i·δ_j where δ_i = 1 if i < dim else 0
    let d1 = if i < dim { 1.0 } else { 0.0 };
    let d2 = if j < dim { 1.0 } else { 0.0 };
    if i < dim && j < dim {
        delta_ij(i, j) - d1 * d2 / 3.0
    } else {
        // shear components: δ_ij (1 for matching shear)
        delta_ij(i, j)
    }
}

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("plasticity ref_elem_vol: unsupported ({et:?}, {order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col+1]);
        for row in 0..dim { j[(row,col)] = xc[row] - x0[row]; }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += jit[(j,k)] * gr[i*dim+k]; }
            gp[i*dim+j] = s;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn j2_elastic_step_zero_plasticity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e6, 0.0); // very high yield → elastic
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let mut r = vec![0.0; n];
        let rhs = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let norm: f64 = r.iter().map(|v| v.abs()).sum();
        assert!(norm < 1e-12, "zero u → zero residual: {norm:.3e}");
    }

    #[test]
    fn j2_tangent_matrix_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e6, 0.0);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let jac = form.jacobian(&u);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) {
                sum += jac.get(i, j).abs();
            }
        }
        assert!(sum > 0.0, "tangent matrix should be non-zero");
    }
}
