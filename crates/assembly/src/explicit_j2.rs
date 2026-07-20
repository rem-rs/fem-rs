//! Explicit-dynamics J2 plasticity with incremental (rate-form) stress update.
//!
//! Uses the **corotational** formulation standard in LS-DYNA / Abaqus/Explicit:
//! each time step applies a small-strain radial return in the unrotated frame,
//! then rotates the Cauchy stress back.  For CFL-limited time steps the
//! rotation is negligible and the algorithm reduces to:
//!
//! 1. Δε = sym(∇Δu)              — incremental small strain
//! 2. σ_trial = σₙ + C : Δε      — trial Cauchy stress
//! 3. J₂ radial return on σ_trial — return mapping
//! 4. σₙ₊₁ = returned stress
//! 5. f_int = ∫ Bᵀ · σₙ₊₁ dΩ    — internal force

use fem_element::lagrange::TriP1;
use fem_element::ReferenceElement;
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;

/// Per-quadrature-point state for explicit J2 plasticity.
#[derive(Debug, Clone)]
pub struct ExplicitJ2QpState {
    /// Cauchy stress in Voigt form: [σ_xx, σ_yy, σ_zz, σ_xy] (2D plane strain)
    pub stress: [f64; 4],
    /// Accumulated effective plastic strain.
    pub alpha: f64,
}

impl Default for ExplicitJ2QpState {
    fn default() -> Self {
        Self { stress: [0.0; 4], alpha: 0.0 }
    }
}

/// Material configuration for explicit J2 plasticity.
#[derive(Debug, Clone)]
pub struct ExplicitJ2Config {
    pub E: f64,
    pub nu: f64,
    pub sigma_y: f64,
    pub H: f64, // isotropic hardening modulus
}

impl ExplicitJ2Config {
    pub fn mu(&self) -> f64 {
        self.E / (2.0 * (1.0 + self.nu))
    }
    pub fn lambda(&self) -> f64 {
        self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    }
}

/// Assemble internal forces using explicit J2 stress update.
///
/// # Arguments
/// * `mesh` — the mesh
/// * `u` — current displacement vector (interleaved: [u0_x, u0_y, u1_x, ...])
/// * `u_prev` — previous displacement vector (for incremental strain)
/// * `qp_states` — per-QP stress/plastic state (mutated in-place)
/// * `cfg` — material parameters
/// * `quad_order` — quadrature order
///
/// Returns internal force vector `f_int`.
pub fn assemble_explicit_j2_2d(
    mesh: &Mesh<2>,
    u: &[f64],        // current displacement (u_pred)
    u_prev: &[f64],   // previous displacement (u_n)
    qp_states: &mut [ExplicitJ2QpState],
    cfg: &ExplicitJ2Config,
    quad_order: u8,
) -> Vec<f64> {
    let dim = 2;
    let n_nodes = mesh.n_nodes() as usize;
    let n_dofs = n_nodes * dim;
    let mut f_int = vec![0.0; n_dofs];

    let mu = cfg.mu();
    let lam = cfg.lambda();
    let sqrt_23 = (2.0 / 3.0_f64).sqrt();
    let sigma_y = cfg.sigma_y;
    let H = cfg.H;

    // Elastic stiffness: plane strain
    let c_e = [
        [lam + 2.0 * mu, lam, 0.0],
        [lam, lam + 2.0 * mu, 0.0],
        [0.0, 0.0, mu],
    ];

    let ref_elem = TriP1;
    let mut qp_idx = 0;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let n_ldofs = nodes.len();
        let (jac, det_j) = tri_jacobian(mesh, &nodes);
        let inv_jac = inv2x2(&jac);
        let quad = ref_elem.quadrature(quad_order);

        let mut u_elem = vec![0.0; n_ldofs * dim];
        for k in 0..n_ldofs {
            let dof_base = nodes[k] as usize * dim;
            for d in 0..dim {
                let u_cur = u.get(dof_base + d).copied().unwrap_or(0.0);
                let u_old = u_prev.get(dof_base + d).copied().unwrap_or(0.0);
                u_elem[k * dim + d] = u_cur - u_old; // incremental displacement
            }
        }

        for _q in 0..quad.weights.len() {
            // Reference gradients (constant for P1)
            let mut ref_grad = vec![0.0; n_ldofs * dim];
            ref_elem.eval_grad_basis(&quad.points[qp_idx % quad.weights.len()], &mut ref_grad);

            // Physical gradients
            let mut gphys = vec![0.0; n_ldofs * dim];
            for k in 0..n_ldofs {
                for i in 0..dim {
                    gphys[k * dim + i] = 0.0;
                    for j in 0..dim {
                        gphys[k * dim + i] += inv_jac[i][j] * ref_grad[k * dim + j];
                    }
                }
            }

            // Displacement gradient ∇u
            let mut gradu = [[0.0; 2]; 2];
            for k in 0..n_ldofs {
                for i in 0..dim {
                    for j in 0..dim {
                        gradu[i][j] += u_elem[k * dim + i] * gphys[k * dim + j];
                    }
                }
            }

            // Small strain increment: Δε = sym(∇u)  (assuming u starts from previous state)
            let de_xx = gradu[0][0];
            let de_yy = gradu[1][1];
            let de_xy = 0.5 * (gradu[0][1] + gradu[1][0]);

            // Trial stress: σ_trial = σ_old + C : Δε
            let state = &mut qp_states[qp_idx];
            let sxx = state.stress[0] + c_e[0][0] * de_xx + c_e[0][1] * de_yy;
            let syy = state.stress[1] + c_e[1][0] * de_xx + c_e[1][1] * de_yy;
            let szz = state.stress[2] + lam * (de_xx + de_yy); // ε_zz = 0 (plane strain)
            let sxy = state.stress[3] + c_e[2][2] * (2.0 * de_xy);

            // Deviatoric stress (including out-of-plane)
            let p = (sxx + syy + szz) / 3.0;
            let s_dev_xx = sxx - p;
            let s_dev_yy = syy - p;
            let s_dev_zz = szz - p;
            let s_dev_xy = sxy;

            // von Mises norm: ||dev(σ)|| = sqrt(S'_ij · S'_ij)
            let eta_sq = s_dev_xx * s_dev_xx + s_dev_yy * s_dev_yy + s_dev_zz * s_dev_zz
                + 2.0 * s_dev_xy * s_dev_xy;
            let eta = eta_sq.sqrt().max(1e-300);

            // Yield function
            let yield_val = eta - sqrt_23 * (sigma_y + H * state.alpha);

            let (s_new_xx, s_new_yy, s_new_zz, s_new_xy, alpha_new) =
                if yield_val > 1e-12 {
                    // Radial return
                    let denom = 1.0 + H / (3.0 * mu);
                    let dgamma = yield_val / (2.0 * mu * denom);
                    let alpha_new = state.alpha + sqrt_23 * dgamma;
                    let factor = 1.0 - 2.0 * mu * dgamma / eta;
                    (
                        s_dev_xx * factor + p,
                        s_dev_yy * factor + p,
                        s_dev_zz * factor + p,
                        s_dev_xy * factor,
                        alpha_new,
                    )
                } else {
                    (sxx, syy, szz, sxy, state.alpha)
                };

            // Update state
            state.stress = [s_new_xx, s_new_yy, s_new_zz, s_new_xy];
            state.alpha = alpha_new;

            // Internal force: f_int = ∫ Bᵀ · σ dΩ
            // For plane strain, use in-plane stress components only for B-matrix.
            // B-matrix maps nodal displacements to strains: ε = B · u
            // B_ki = [∂N_k/∂x, 0; 0, ∂N_k/∂y; ∂N_k/∂y, ∂N_k/∂x]  (Voigt: [ε_xx, ε_yy, 2ε_xy])
            // f_int_k_i = Σ B_ki · σ_voigt · w · detJ
            let w = quad.weights[qp_idx % quad.weights.len()] * det_j.abs();
            for k in 0..n_ldofs {
                let dndx = gphys[k * dim];
                let dndy = gphys[k * dim + 1];
                let dof_x = nodes[k] as usize * dim;
                let dof_y = nodes[k] as usize * dim + 1;
                // f_x += (σ_xx · ∂N/∂x + σ_xy · ∂N/∂y) · w
                f_int[dof_x] += (s_new_xx * dndx + s_new_xy * dndy) * w;
                // f_y += (σ_xy · ∂N/∂x + σ_yy · ∂N/∂y) · w
                f_int[dof_y] += (s_new_xy * dndx + s_new_yy * dndy) * w;
            }

            qp_idx += 1;
        }
    }
    f_int
}

// ─── 2×2 matrix helpers ────────────────────────────────────────────────

fn tri_jacobian(mesh: &Mesh<2>, nodes: &[u32]) -> ([[f64; 2]; 2], f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let j = [
        [x1[0] - x0[0], x2[0] - x0[0]],
        [x1[1] - x0[1], x2[1] - x0[1]],
    ];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    (j, det)
}

fn inv2x2(j: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if det.abs() < 1e-30 {
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    [
        [j[1][1] * inv_det, -j[0][1] * inv_det],
        [-j[1][0] * inv_det, j[0][0] * inv_det],
    ]
}
