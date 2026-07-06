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

use std::sync::Mutex;

use nalgebra::DMatrix;
use fem_element::{
    ReferenceElement,
    lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;

use crate::physics::nonlinear::NonlinearForm;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Plasticity model family selector.
///
/// Determines which yield surface and return-mapping algorithm the assembly
/// loop dispatches to. Historically, `J2PlasticityForm` always executed the
/// J2 radial return even when `PlasticConfig::drucker_prager(...)` had built
/// the cone coefficients — resulting in silent misuse. Selecting an explicit
/// model closes that gap.
///
/// - `J2`: von Mises with linear isotropic hardening (radial return).
/// - `DruckerPrager`: pressure-sensitive cone with apex return (Simo &
///   Hughes §7.5). Currently associated flow (ψ = φ); non-associated
///   left for Phase 3E.
/// - `CamClay`, `Viscoplastic`: reserved for Phase 3E; assembly will
///   `unimplemented!` for now to prevent silent misuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasticModel {
    /// J2 (von Mises) — pressure-independent yield.
    J2,
    /// Drucker–Prager — pressure-sensitive cone.
    DruckerPrager,
    /// Modified Cam-Clay — volumetric hardening in p–q space.
    CamClay,
    /// Mohr–Coulomb — pressure-sensitive with Lode-angle dependence.
    MohrCoulomb,
    /// Hoek–Brown — empirical rock-failure criterion (generalized form).
    HoekBrown,
    /// Perzyna / Duvaut-Lions viscoplasticity — reserved for Phase 3E.
    Viscoplastic,
}

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
    /// Which yield surface / return-mapping to execute.
    pub model: PlasticModel,
    // ── Lemaitre damage‑plasticity coupling ───────────────────────────
    /// Damage coupling enabled (requires `damage_S > 0`).
    pub damage_S: f64,
    /// Threshold plastic strain for damage initiation (Lemaitre Ḋ = ⟨Y/S⟩·ṗ).
    pub damage_p_D: f64,
    /// Critical damage (caps D to avoid zero stiffness).
    pub damage_D_c: f64,
    // ── Viscoplasticity (Perzyna / Duvaut–Lions) ──────────────────────
    /// Viscosity η (0 → rate‑independent).  Used as η/Δt in Perzyna.
    pub viscosity: f64,
    // ── Kinematic hardening (Armstrong–Frederick) ──────────────────────
    /// Kinematic hardening modulus C (0 → pure isotropic).
    pub kinematic_modulus: f64,
    /// AF recall / saturation term γ (0 → linear Prager kinematic).
    pub kinematic_recall: f64,
    // ── Cam-Clay parameters ─────────────────────────────────────────────
    /// Critical state line slope M (≈ 6·sin(φ)/(3 − sin(φ)) for triaxial compression).
    pub m: f64,
    /// Initial preconsolidation pressure p_c0 (> 0).
    pub p_c0: f64,
    /// Compression index λ (slope of normal compression line in e−ln p space).
    pub lambda_index: f64,
    /// Swelling/recompression index κ.
    pub kappa_index: f64,
    /// Initial void ratio e₀.
    pub void_ratio: f64,
    // ── Hoek–Brown parameters ───────────────────────────────────────────
    /// Uniaxial compressive strength σ_ci of intact rock.
    pub hb_ci: f64,
    /// Hoek–Brown constant m (m_i for intact rock, m_b for rock mass).
    pub hb_m: f64,
    /// Hoek–Brown constant s (1.0 for intact rock, < 1 for rock mass).
    pub hb_s: f64,
    /// Hoek–Brown exponent a (0.5 for intact rock, computed from GSI for rock mass).
    pub hb_a: f64,
}

impl PlasticConfig {
    /// Standard J2 (von Mises) with linear isotropic hardening.
    pub fn j2(E: f64, nu: f64, sigma_y: f64, H: f64) -> Self {
        Self {
            E, nu, yield_stress: sigma_y, hardening_modulus: H,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::J2,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// J2 with Armstrong–Frederick kinematic hardening (C > 0) + isotropic.
    pub fn j2_kinematic(E: f64, nu: f64, sigma_y: f64, H: f64, C: f64, gamma: f64) -> Self {
        Self {
            E, nu, yield_stress: sigma_y, hardening_modulus: H,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::J2,
            viscosity: 0.0, kinematic_modulus: C, kinematic_recall: gamma,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// Perzyna J2 viscoplasticity with linear isotropic hardening.
    /// `eta` = viscosity (≥ 0); `dt` is passed to the form at assembly time.
    pub fn j2_viscoplastic(E: f64, nu: f64, sigma_y: f64, H: f64, eta: f64) -> Self {
        Self {
            E, nu, yield_stress: sigma_y, hardening_modulus: H,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::J2,
            viscosity: eta, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// J2 plasticity coupled with Lemaitre isotropic damage.
    ///
    /// `S` = damage strength, `p_D` = threshold accumulated plastic strain
    /// for damage initiation.
    pub fn j2_damage(E: f64, nu: f64, sigma_y: f64, H: f64, S: f64, p_D: f64) -> Self {
        Self {
            E, nu, yield_stress: sigma_y, hardening_modulus: H,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::J2,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: S, damage_p_D: p_D, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// Drucker–Prager with associated flow (ψ = φ).
    pub fn drucker_prager(E: f64, nu: f64, cohesion: f64,
                          friction_angle_deg: f64, H: f64) -> Self {
        Self::drucker_prager_general(E, nu, cohesion, friction_angle_deg, friction_angle_deg, H)
    }

    /// Drucker–Prager with independent dilation angle (ψ ≤ φ for non‑associated flow).
    pub fn drucker_prager_general(E: f64, nu: f64, cohesion: f64,
                                  friction_angle_deg: f64, dilation_angle_deg: f64, H: f64) -> Self {
        Self {
            E, nu, yield_stress: cohesion, hardening_modulus: H,
            friction_angle: friction_angle_deg.to_radians(),
            dilation_angle: dilation_angle_deg.to_radians(),
            model: PlasticModel::DruckerPrager,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// Modified Cam–Clay with isotropic volumetric hardening.
    pub fn cam_clay(E: f64, nu: f64, m: f64, p_c0: f64,
                    lambda_idx: f64, kappa_idx: f64, void_ratio: f64) -> Self {
        Self {
            E, nu, yield_stress: 0.0, hardening_modulus: 0.0,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::CamClay,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m, p_c0, lambda_index: lambda_idx, kappa_index: kappa_idx, void_ratio,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// Mohr–Coulomb with non-associated flow (ψ ≤ φ).
    pub fn mohr_coulomb(E: f64, nu: f64, cohesion: f64,
                        friction_angle_deg: f64, dilation_angle_deg: f64, H: f64) -> Self {
        Self {
            E, nu, yield_stress: cohesion, hardening_modulus: H,
            friction_angle: friction_angle_deg.to_radians(),
            dilation_angle: dilation_angle_deg.to_radians(),
            model: PlasticModel::MohrCoulomb,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: 0.0, hb_m: 0.0, hb_s: 0.0, hb_a: 0.0,
        }
    }

    /// Hoek–Brown (generalized) for rock mass.
    ///
    /// `ci` = σ_ci (UCS), `hb_m` = m_b (rock-mass constant),
    /// `hb_s` = s, `hb_a` = a exponent.
    pub fn hoek_brown(E: f64, nu: f64, ci: f64, hb_m: f64, hb_s: f64, hb_a: f64) -> Self {
        Self {
            E, nu, yield_stress: 0.0, hardening_modulus: 0.0,
            friction_angle: 0.0, dilation_angle: 0.0,
            model: PlasticModel::HoekBrown,
            viscosity: 0.0, kinematic_modulus: 0.0, kinematic_recall: 0.0,
            damage_S: 0.0, damage_p_D: 0.0, damage_D_c: 0.99,
            m: 0.0, p_c0: 0.0, lambda_index: 0.0, kappa_index: 0.0, void_ratio: 0.0,
            hb_ci: ci, hb_m, hb_s, hb_a,
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

    /// Drucker–Prager α factor (from friction angle φ, for yield function).
    pub fn dp_alpha(&self) -> f64 {
        let sinφ = self.friction_angle.sin();
        2.0 * sinφ / (3.0_f64.sqrt() * (3.0 - sinφ))
    }

    /// Drucker–Prager k factor (from friction angle φ, for yield function).
    pub fn dp_k(&self) -> f64 {
        let sinφ = self.friction_angle.sin();
        6.0 * self.friction_angle.cos() / (3.0_f64.sqrt() * (3.0 - sinφ))
    }

    /// Drucker–Prager α factor from dilation angle ψ (for plastic potential).
    pub fn dp_alpha_psi(&self) -> f64 {
        let sinψ = self.dilation_angle.sin();
        2.0 * sinψ / (3.0_f64.sqrt() * (3.0 - sinψ))
    }

    /// Drucker–Prager k factor from dilation angle ψ (for plastic potential).
    pub fn dp_k_psi(&self) -> f64 {
        let sinψ = self.dilation_angle.sin();
        6.0 * self.dilation_angle.cos() / (3.0_f64.sqrt() * (3.0 - sinψ))
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
    #[allow(dead_code)]
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8,
    dt: f64,  // time step for rate‑dependent (viscoplastic) integration
    /// Per-element, per-QP state: 13 values (6 ε^p + 1 α_iso + 6 α_back).
    /// Wrapped in `Mutex` because `NonlinearForm` only grants `&self`.
    state: Mutex<Vec<Vec<f64>>>,
    elems: Vec<u32>,
    qp_offsets: Vec<usize>,
}

impl<M: MeshTopology> J2PlasticityForm<M> {
    pub fn new(space: VectorH1Space<M>, cfg: PlasticConfig,
               dirichlet: Vec<(usize, f64)>, quad_order: u8) -> Self {
        Self::with_dt(space, cfg, dirichlet, quad_order, 0.0)
    }

    /// Like `new` but with an explicit time step `dt` for rate‑dependent
    /// (viscoplastic) integration.  Pass `dt = 0` (default) for rate‑independent.
    pub fn with_dt(space: VectorH1Space<M>, cfg: PlasticConfig,
                       #[allow(dead_code)]
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8, dt: f64) -> Self {
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
        // 14 state variables per QP: 6 ε^p + 1 α_iso + 6 α_back + 1 D
        let state = Mutex::new(vec![vec![0.0_f64; 14]; total_qp]);
        Self { space, cfg, dirichlet, quad_order, dt, state, elems, qp_offsets }
    }

    /// Reset all internal variables to zero (for a fresh analysis).
    pub fn reset_state(&mut self) {
        // Access inner Vec<Vec<f64>> directly (no locking needed for &mut self).
        let inner = &mut *self.state.get_mut().unwrap();
        for s in inner.iter_mut() {
            for i in 0..s.len() { s[i] = 0.0; }
        }
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
                eps[2] += u_elem[k * dim] * gphys[k * dim + 1]
                        + u_elem[k * dim + 1] * gphys[k * dim]; // γ_xy
            } else {
                let off = k * dim;
                eps[3] += u_elem[off] * gphys[off + 1]
                        + u_elem[off + 1] * gphys[off]; // γ_xy
                eps[4] += u_elem[off + 1] * gphys[off + 2]
                        + u_elem[off + 2] * gphys[off + 1]; // γ_yz
                eps[5] += u_elem[off] * gphys[off + 2]
                        + u_elem[off + 2] * gphys[off]; // γ_zx
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

        // Borrow state for read/write (Mutex interior mutability)
        let mut state = self.state.lock().unwrap();

        for (el, e) in self.elems.iter().enumerate() {
            let elem_type = mesh.element_type(*e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);
            let _qp_start = self.qp_offsets[el];

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
                let eps_p_old: Vec<f64> = state[old_idx][..6].to_vec();
                let alpha_old = state[old_idx][6];
                let alpha_back_old: Vec<f64> = state[old_idx][7..13].to_vec();
                let d_old = state[old_idx][13];

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

                // Relative stress η = s - α_back (for kinematic hardening)
                let C_k = self.cfg.kinematic_modulus;
                let gam_k = self.cfg.kinematic_recall;
                let mut eta_trial = vec![0.0; n_comp];
                for i in 0..n_comp { eta_trial[i] = s_trial[i] - alpha_back_old[i]; }
                let eta_norm = (eta_trial.iter().map(|v| v*v).sum::<f64>()).sqrt();
                let s_norm = (s_trial.iter().map(|v| v*v).sum::<f64>()).sqrt();
                let sqrt23 = (2.0_f64 / 3.0_f64).sqrt();
                let K_old = sigma_y + H * alpha_old;

                // Yield-function evaluation depends on model.
                let f_trial: f64 = match self.cfg.model {
                    PlasticModel::J2 => eta_norm - sqrt23 * K_old,
                    PlasticModel::DruckerPrager => {
                        let alpha_dp = self.cfg.dp_alpha();
                        let k_dp = self.cfg.dp_k();
                        let i1 = if dim == 2 { sigma_trial[0] + sigma_trial[1] }
                                 else       { sigma_trial[0] + sigma_trial[1] + sigma_trial[2] };
                        alpha_dp * i1 + s_norm - k_dp * K_old
                    }
                    PlasticModel::MohrCoulomb => {
                        let prin = principal_stresses(&sigma_trial, dim);
                        mohr_coulomb_yield(&prin, sigma_y,
                            self.cfg.friction_angle.sin(), self.cfg.friction_angle.cos())
                    }
                    PlasticModel::HoekBrown => {
                        let prin = principal_stresses(&sigma_trial, dim);
                        hoek_brown_yield(&prin, self.cfg.hb_ci, self.cfg.hb_m,
                            self.cfg.hb_s, self.cfg.hb_a)
                    }
                    PlasticModel::CamClay => {
                        let p_trial = if dim == 2 { (sigma_trial[0] + sigma_trial[1]) / 3.0 }
                                      else { (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0 };
                        let q_trial_sq = (3.0/2.0) * s_norm * s_norm;
                        let pc_old = if alpha_old > 0.0 {
                            // Recover pc from accumulated state (stored as α = ln(pc/pc0)·(λ-κ)/(1+e₀))
                            self.cfg.p_c0 * (alpha_old * (1.0 + self.cfg.void_ratio)
                                / (self.cfg.lambda_index - self.cfg.kappa_index + 1e-30)).exp()
                        } else { self.cfg.p_c0 };
                        cam_clay_yield(p_trial, q_trial_sq.sqrt(), pc_old, self.cfg.m)
                    }
                    PlasticModel::Viscoplastic => {
                        // Perzyna/Duvaut-Lions viscoplasticity uses J2 yield surface
                        // with overstress regularisation via the viscosity term in the
                        // return-mapping denominator.  Requires `dt > 0` and `viscosity > 0`.
                        eta_norm - sqrt23 * K_old
                    }
                };

                // Return mapping
                let mut sigma = sigma_trial.clone();
                let mut D_ep = c_e.clone();
                let mut dgamma = 0.0;
                let new_alpha = if f_trial > 0.0 {
                    let two_mu = 2.0 * mu;
                    match self.cfg.model {
                        PlasticModel::J2 | PlasticModel::Viscoplastic => {
                            // Radial return with Armstrong–Frederick kinematic hardening
                            // and Perzyna viscoplasticity.
                            // η = s - α_back  (relative stress).
                            let eta_dot_n = if eta_norm > 1e-30 {
                                (0..n_comp).map(|i| alpha_back_old[i] * eta_trial[i]).sum::<f64>()
                                    / eta_norm
                            } else { 0.0 };
                            // Perzyna overstress: add η/Δt to denominator
                            let visc_term = if self.dt > 1e-30 {
                                self.cfg.viscosity / self.dt
                            } else { 0.0 };
                            let denom_kin = two_mu + (2.0/3.0)*H
                                + (2.0/3.0)*C_k - gam_k * eta_dot_n + visc_term;
                            let dgamma_inv = if denom_kin > 1e-30 { 1.0 / denom_kin } else { 0.0 };
                            dgamma = f_trial * dgamma_inv;
                            // Radial return on η
                            let factor = 1.0 - two_mu * dgamma / (eta_norm.max(1e-30));
                            let mut eta_new = vec![0.0; n_comp];
                            for i in 0..n_comp { eta_new[i] = eta_trial[i] * factor; }
                            // Back stress update (Armstrong–Frederick)
                            let n_eta = if eta_norm > 1e-30 {
                                let inv = 1.0 / eta_norm;
                                (0..n_comp).map(|i| eta_trial[i] * inv).collect::<Vec<f64>>()
                            } else { vec![0.0; n_comp] };
                            let ab_denom = 1.0 + gam_k * dgamma;
                            let mut alpha_new = vec![0.0; n_comp];
                            for i in 0..n_comp {
                                alpha_new[i] = (alpha_back_old[i]
                                    + (2.0/3.0)*C_k * dgamma * n_eta[i]) / ab_denom;
                            }
                            // Reconstruct σ = η + α + p·I
                            for i in 0..dim { sigma[i] = eta_new[i] + alpha_new[i] + p_trial; }
                            if dim == 2 { sigma[2] = eta_new[2] + alpha_new[2]; }
                            else {
                                sigma[3] = eta_new[3] + alpha_new[3];
                                sigma[4] = eta_new[4] + alpha_new[4];
                                sigma[5] = eta_new[5] + alpha_new[5];
                            }
                            // Consistent tangent (continuum approx with kinematic)
                            let beta = two_mu * dgamma / (eta_norm.max(1e-30));
                            let hard_mod = (2.0/3.0)*H + (2.0/3.0)*C_k - gam_k * eta_dot_n;
                            let theta_mod = 1.0 / (1.0 + hard_mod / two_mu) - (1.0 - beta);
                            let mut nn = DMatrix::zeros(n_comp, n_comp);
                            for ii in 0..n_comp { for jj in 0..n_comp {
                                nn[(ii, jj)] = n_eta[ii] * n_eta[jj];
                            }}
                            for i in 0..n_comp {
                                for j in 0..n_comp {
                                    D_ep[(i,j)] = c_e[(i,j)]
                                        - two_mu * (beta * dev_proj_ij(i, j, dim) - theta_mod * nn[(i,j)]);
                                }
                            }
                            alpha_old + dgamma * sqrt23
                        }
                        PlasticModel::DruckerPrager => {
                            // Drucker–Prager return with non‑associated flow (ψ ≤ φ).
                            // Yield  function f uses friction angle φ  → α_f, k_f
                            // Plastic potential g uses dilation angle ψ → α_g, k_g
                            // When ψ = φ the classical associated (Simo & Hughes §7.5) case is recovered.
                            let alpha_f = self.cfg.dp_alpha();       // from φ
                            let k_f = self.cfg.dp_k();
                            let alpha_g = self.cfg.dp_alpha_psi();   // from ψ
                            let k_g = self.cfg.dp_k_psi();
                            let K_bulk = self.cfg.bulk();
                            // Cone denominator: 2μ + 9K·α_f·α_g + H·k_f·k_g
                            let denom_cone = two_mu + 9.0 * K_bulk * alpha_f * alpha_g
                                             + H * k_f * k_g;
                            dgamma = f_trial / denom_cone;
                            let factor = 1.0 - two_mu * dgamma / (s_norm + 1e-30);
                            let mut s_new = s_trial.clone();
                            for i in 0..n_comp { s_new[i] *= factor; }
                            // Volumetric update uses α from the POTENTIAL (ψ)
                            let mut p_new = p_trial - 3.0 * K_bulk * alpha_g * dgamma;

                            let s_new_norm = (s_new.iter().map(|v| v*v).sum::<f64>()).sqrt();
                            if factor < 0.0 || s_new_norm < 1e-14 {
                                // Apex return: project to hydrostatic state.
                                // The apex p coordinate is determined by f = 0 with s = 0.
                                let sigma_y_updated = sigma_y + H * alpha_old;
                                p_new = k_f * sigma_y_updated / (3.0 * alpha_f);
                                for i in 0..n_comp { s_new[i] = 0.0; }
                            }
                            s_trial[..n_comp].copy_from_slice(&s_new[..n_comp]);
                            for i in 0..dim { sigma[i] = s_new[i] + p_new; }
                            if dim == 2 { sigma[2] = s_new[2]; }
                            else { sigma[3] = s_new[3]; sigma[4] = s_new[4]; sigma[5] = s_new[5]; }

                            // Elastic‑plastic tangent (continuum, unsymmetric for non‑associated flow).
                            let denom = two_mu + 9.0 * K_bulk * alpha_f * alpha_g + H * k_f * k_g;
                            let inv_snorm = 1.0 / (s_norm + 1e-30);
                            let nn_dim = n_comp;
                            let mut nn = DMatrix::zeros(nn_dim, nn_dim);
                            for ii in 0..nn_dim { for jj in 0..nn_dim {
                                nn[(ii, jj)] = s_trial[ii] * s_trial[jj] * inv_snorm * inv_snorm;
                            }}
                            for i in 0..n_comp {
                                for j in 0..n_comp {
                                    let dev = dev_proj_ij(i, j, dim);
                                    D_ep[(i,j)] = c_e[(i,j)]
                                        - two_mu * two_mu * dev / denom
                                        - 9.0 * K_bulk * K_bulk * alpha_f * alpha_g / denom
                                              * (if i < dim && j < dim { 1.0 } else { 0.0 })
                                        - two_mu * two_mu * nn[(i,j)] / denom;
                                }
                            }
                            alpha_old + dgamma * sqrt23
                        }
                        PlasticModel::MohrCoulomb => {
                            let sin_phi = self.cfg.friction_angle.sin();
                            let cos_phi = self.cfg.friction_angle.cos();
                            let sin_psi = self.cfg.dilation_angle.sin();
                            let (sigma_new, dg) = mc_return_mapping(
                                &sigma_trial, dim, sigma_y, sin_phi, cos_phi, sin_psi);
                            sigma.copy_from_slice(&sigma_new);
                            dgamma = dg;
                            // Continuum elasto-plastic tangent via principal directions
                            let eigvecs = principal_directions(&sigma_trial, dim);
                            let n_f_prin: [f64; 3] = [1.0 + sin_phi, 0.0, -(1.0 - sin_phi)];
                            let n_g_prin: [f64; 3] = [1.0 + sin_psi, 0.0, -(1.0 - sin_psi)];
                            let r_f = flow_voigt(&n_f_prin, &eigvecs, dim);
                            let r_g = flow_voigt(&n_g_prin, &eigvecs, dim);
                            D_ep = tangent_elasto_plastic(&c_e, &r_f, &r_g, n_comp, H);
                            alpha_old + dgamma * sqrt23
                        }
                        PlasticModel::HoekBrown => {
                            let (sigma_new, dg) = hb_return_mapping(
                                &sigma_trial, dim,
                                self.cfg.hb_ci, self.cfg.hb_m, self.cfg.hb_s, self.cfg.hb_a);
                            sigma.copy_from_slice(&sigma_new);
                            dgamma = dg;
                            // HB is associated: use same gradient for yield and flow.
                            let prin = principal_stresses(&sigma_trial, dim);
                            let s3 = prin[2];
                            let arg = (self.cfg.hb_m * s3 / self.cfg.hb_ci + self.cfg.hb_s).max(0.0);
                            let df_ds3 = if arg > 0.0 {
                                -1.0 - self.cfg.hb_m * self.cfg.hb_a * arg.powf(self.cfg.hb_a - 1.0)
                            } else { -1.0 };
                            let eigvecs = principal_directions(&sigma_trial, dim);
                            let n_prin: [f64; 3] = [1.0, 0.0, df_ds3];
                            let r = flow_voigt(&n_prin, &eigvecs, dim);
                            D_ep = tangent_elasto_plastic(&c_e, &r, &r, n_comp, H);
                            alpha_old + dgamma * sqrt23
                        }
                        PlasticModel::CamClay => {
                            let bulk = self.cfg.bulk();
                            let (sigma_new, dg, _new_pc, p_ret, _q_ret) = cc_return_mapping(
                                &sigma_trial, dim,
                                self.cfg.m, self.cfg.p_c0,
                                self.cfg.lambda_index, self.cfg.kappa_index,
                                self.cfg.void_ratio, mu, bulk);
                            sigma.copy_from_slice(&sigma_new);
                            dgamma = dg;
                            // Cam-Clay consistent tangent in p-q space (associated flow).
                            // n_Voigt = M²·(2p-p_c)/3 · δ + 3·s
                            let pc = self.cfg.p_c0 * (dgamma * (1.0 + self.cfg.void_ratio)
                                / (self.cfg.lambda_index - self.cfg.kappa_index + 1e-30)).exp();
                            let m2p = self.cfg.m * self.cfg.m * (2.0 * p_ret - pc);
                            let mut r_cc = vec![0.0; n_comp];
                            for i in 0..dim { r_cc[i] = m2p / 3.0 + 3.0 * (sigma_new[i] - p_ret); }
                            if dim == 2 { r_cc[2] = 3.0 * sigma_new[2]; }
                            else {
                                r_cc[2] = m2p / 3.0 + 3.0 * (sigma_new[2] - p_ret);
                                for i in 3..6 { r_cc[i] = 3.0 * sigma_new[i]; }
                            }
                            // Hardening modulus from p_c evolution
                            let h_cc = if pc > 1e-30 && (self.cfg.lambda_index - self.cfg.kappa_index).abs() > 1e-30 {
                                let v = 1.0 + self.cfg.void_ratio;
                                let hard_slope = self.cfg.lambda_index - self.cfg.kappa_index;
                                self.cfg.m.powi(4) * p_ret * pc * v / hard_slope * (2.0 * p_ret - pc).powi(2)
                            } else { 0.0 };
                            D_ep = tangent_elasto_plastic(&c_e, &r_cc, &r_cc, n_comp, h_cc);
                            alpha_old + dgamma
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Elastic step: no update
                    alpha_old
                };

                // Store updated state (via Mutex interior mutability)
                state[old_idx][..6].copy_from_slice(&eps_p_old);
                if f_trial > 0.0 {
                    for i in 0..n_comp { state[old_idx][i] += dgamma * s_trial[i] / (s_norm + 1e-30); }
                }
                state[old_idx][6] = new_alpha;
                // Back stress storage (kinematic hardening; only J2 branch updates it)
                if self.cfg.model == PlasticModel::J2 && self.cfg.kinematic_modulus > 0.0
                    && f_trial > 0.0
                {
                        // alpha_new was computed in J2 branch; re-derive from plastic strain inc
                        let a_denom = 1.0 + self.cfg.kinematic_recall * dgamma;
                        let n_eta = if eta_norm > 1e-30 {
                            let inv = 1.0 / eta_norm;
                            (0..n_comp).map(|i| eta_trial[i] * inv).collect::<Vec<f64>>()
                        } else { vec![0.0; n_comp] };
                        for i in 0..n_comp {
                            state[old_idx][7+i] = (alpha_back_old[i]
                                + (2.0/3.0)*self.cfg.kinematic_modulus * dgamma * n_eta[i]) / a_denom;
                        }
                    }
                // else: elastic step → back stress unchanged

                // ── Damage coupling (Lemaitre effective stress) ─────
                // σ = (1-D)·σ̃  where σ̃ is the effective (undamaged) stress
                // computed by the return mapping above.
                let damage_active = self.cfg.damage_S > 0.0;
                if damage_active {
                    let one_minus_d = (1.0 - d_old).max(1e-6);
                    for i in 0..n_comp { sigma[i] *= one_minus_d; }
                    // Simplified damaged tangent: (1-D)·D_ep
                    for i in 0..n_comp { for j in 0..n_comp {
                        D_ep[(i,j)] *= one_minus_d;
                    }}
                }

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
                // ── Damage variable update (Lemaitre) ──────────────
                if damage_active && f_trial > 0.0 {
                    let dp = dgamma * sqrt23;  // accumulated plastic strain inc
                    let total_p = alpha_old + dp;  // total accumulated plastic strain
                    if total_p > self.cfg.damage_p_D && dp > 1e-30 {
                        // Triaxiality: R_v ≈ 1 for uniaxial (simplified)
                        let r_v = 1.0_f64;
                        // Energy release rate Y = σ̃_eq²·R_v / (2E)
                        let sigma_eq = (sigma_trial.iter().map(|v| v*v).sum::<f64>()
                            .max(1e-30)).sqrt() * (3.0_f64 / 2.0_f64).sqrt();
                        let y = sigma_eq * sigma_eq * r_v / (2.0 * self.cfg.E);
                        // D increment: ΔD = (Y/S)·Δp  (Lemaitre)
                        let dd = (y / self.cfg.damage_S.max(1e-30)) * dp;
                        let d_new = (d_old + dd).min(self.cfg.damage_D_c);
                        state[old_idx][13] = d_new;
                    }
                }
            }
            coo.add_element_matrix(&elem_dofs, &k_elem);
        }

        // State is already updated in-place via Mutex above.

        let mut mat = coo.into_csr();
        let mut dir_rhs = if wants_residual { f_vec } else { vec![0.0; n_dofs] };
        for &(dof, val) in &self.dirichlet {
            mat.apply_dirichlet_symmetric(dof, val, &mut dir_rhs);
        }

        (dir_rhs, mat)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Deviatoric projection tensor in Voigt form, entry `(i, j)`.
fn dev_proj_ij(i: usize, j: usize, dim: usize) -> f64 {
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

// ═══════════════════════════════════════════════════════════════════════════════
// Finite-strain J2 plasticity (Total Lagrangian formulation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Finite-strain J2 plasticity with linear isotropic hardening.
///
/// Uses the **Total Lagrangian** formulation:
/// - Deformation gradient: `F = I + ∇u`
/// - Green–Lagrange strain: `E = ½(FᵀF − I)`
/// - 2nd Piola–Kirchhoff stress: `S`
/// - Return mapping on **E** and **S** in the material frame (identical
///   algebra to small-strain J2).
///
/// Internal variables (per quadrature point):
/// - `E^p` — plastic Green–Lagrange strain (6 components, Voigt)
/// - `α` — accumulated plastic strain
///
/// The consistent tangent includes both the **material** (elastic–plastic)
/// and **geometric** (initial-stress) contributions for quadratic convergence
/// in Newton–Raphson iterations.
///
/// # Usage
/// ```rust,ignore
/// use fem_assembly::plasticity::{FiniteStrainPlasticity, PlasticConfig};
///
/// let cfg = PlasticConfig::j2(2e5, 0.3, 200.0, 1e3);
/// let form = FiniteStrainPlasticity::new(space, cfg, vec![], 2);
/// let K = form.jacobian(&u);
/// ```
pub struct FiniteStrainPlasticity<M: MeshTopology> {
    space: VectorH1Space<M>,
    cfg: PlasticConfig,
    /// Dirichlet boundary conditions (not yet applied — reserved).
    #[allow(dead_code)]
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8,
    /// Per-element, per-QP state: `(E^p_xx, E^p_yy, E^p_zz, Γ^p_xy, Γ^p_yz, Γ^p_zx, α)`.
    /// Wrapped in `Mutex` because `NonlinearForm` only grants `&self`.
    state: Mutex<Vec<Vec<f64>>>,
    elems: Vec<u32>,
    qp_offsets: Vec<usize>,
}

impl<M: MeshTopology> FiniteStrainPlasticity<M> {
    /// Build the finite-strain plasticity form.
    pub fn new(
        space: VectorH1Space<M>,
        cfg: PlasticConfig,
    #[allow(dead_code)]
    dirichlet: Vec<(usize, f64)>,  // TODO: apply Dirichlet BCs in assemble_jacobian
    quad_order: u8,
    ) -> Self {
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
        let state = Mutex::new(vec![vec![0.0_f64; 14]; total_qp]);
        FiniteStrainPlasticity { space, cfg, dirichlet, quad_order, state, elems, qp_offsets }
    }

    pub fn reset_state(&mut self) {
        let inner = &mut *self.state.get_mut().unwrap();
        for s in inner.iter_mut() {
            for i in 0..s.len() { s[i] = 0.0; }
        }
    }

    fn qp_state_idx(&self, elem_local: usize, qp: usize) -> usize {
        self.qp_offsets[elem_local] + qp
    }

    /// Compute the deformation gradient `F = I + ∇u` at a quadrature point.
    fn def_grad(u_elem: &[f64], gphys: &[f64], dim: usize, n_ldofs: usize) -> DMatrix<f64> {
        let mut f = DMatrix::identity(dim, dim);
        for k in 0..n_ldofs {
            let off = k * dim;
            for i in 0..dim {
                for j in 0..dim {
                    f[(i, j)] += u_elem[off + i] * gphys[off + j];
                }
            }
        }
        f
    }

    /// Green–Lagrange strain from deformation gradient: `E = ½(FᵀF − I)`.
    /// Returns Voigt form: `[E_xx, E_yy, E_zz, 2·E_xy, 2·E_yz, 2·E_zx]`.
    fn green_lagrange(f: &DMatrix<f64>) -> Vec<f64> {
        let dim = f.nrows();
        let ft = f.transpose();
        let c = &ft * f;  // right Cauchy–Green
        let mut e = vec![0.0; if dim == 2 { 3 } else { 6 }];
        e[0] = 0.5 * (c[(0,0)] - 1.0);
        e[1] = 0.5 * (c[(1,1)] - 1.0);
        if dim == 2 {
            e[2] = c[(0,1)];  // 2·E_12 = C_12
        } else {
            e[2] = 0.5 * (c[(2,2)] - 1.0);
            e[3] = c[(0,1)];  // 2·E_12
            e[4] = c[(1,2)];  // 2·E_23
            e[5] = c[(0,2)];  // 2·E_13
        }
        e
    }

    /// Convert a small-strain Voigt tangent `C_e` to a material tangent for
    /// Green–Lagrange strain / 2nd Piola–Kirchhoff stress pairing.
    /// For the total Lagrangian formulation, the small-strain elasticity tensor
    /// is used directly because S = C : E in the material frame.
    #[allow(dead_code)]
    fn elastic_stiffness_finite(_dim: usize) -> DMatrix<f64> {
        // The caller uses the same `elastic_stiffness` from PlasticConfig
        // (it returns the small-strain C, which IS the correct material
        // tangent for the S–E pairing in total Lagrangian).
        DMatrix::zeros(1, 1)  // placeholder; actual call uses cfg directly
    }

    /// Assemble the element residual and tangent.
    fn assemble_jacobian(
        &self, u: &[f64], rhs: &[f64], want_residual: bool,
    ) -> (Vec<f64>, CsrMatrix<f64>) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let order = self.space.order();
        let n_dofs = self.space.n_dofs();
        let is_2d = dim == 2;
        let n_comp = if is_2d { 3 } else { 6 };
        let mu = self.cfg.mu();
        let lam = self.cfg.lambda();
        let H = self.cfg.hardening_modulus;
        let sigma_y = self.cfg.yield_stress;
        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        let mut f_vec = vec![0.0_f64; n_dofs];
        f_vec.copy_from_slice(rhs);
        for v in f_vec.iter_mut() { *v = -*v; }

        // Elastic stiffness in Voigt form (same as small-strain C for S–E)
        let c_e = if is_2d {
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
        };

        // Borrow state for read/write (Mutex interior mutability)
        let mut state = self.state.lock().unwrap();

        for (el, &e) in self.elems.iter().enumerate() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_vol(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let n_vec = n_ldofs * dim;
            let quad = ref_elem.quadrature(self.quad_order);
            let nqp = quad.weights.len();
            let _qp_start = self.qp_offsets[el];

                // Shape function gradients in physical coords (n_ldofs × dim)
            let mut gphys = vec![0.0_f64; n_vec * dim];

            for q in 0..nqp {
                // Reference gradient via eval_grad_basis
                let xi = &quad.points[q];
                let mut ref_grad = vec![0.0_f64; n_ldofs * dim];
                ref_elem.eval_grad_basis(xi, &mut ref_grad);

                // Compute Jacobian: J_ij = Σ_k node_coords[k][i] * ref_grad[k*dim + j]
                let mut jac = DMatrix::zeros(dim, dim);
                let nodes = mesh.element_nodes(e);
                for i in 0..dim {
                    for j in 0..dim {
                        let mut s = 0.0;
                        for k in 0..n_ldofs {
                            let c = mesh.node_coords(nodes[k]);
                            s += c[i] * ref_grad[k * dim + j];
                        }
                        jac[(i, j)] = s;
                    }
                }
                let det_j = jac.determinant();
                let jac_it = jac.try_inverse()
                    .unwrap_or_else(|| DMatrix::identity(dim, dim)).transpose();

                // Physical gradients
                for k in 0..n_ldofs {
                    for i in 0..dim {
                        gphys[k * dim + i] = 0.0;
                        for j in 0..dim {
                            gphys[k * dim + i] += jac_it[(i, j)] * ref_grad[k * dim + j];
                        }
                    }
                }

                // ── Element DOF values for this QP ──
                let mut u_elem = vec![0.0; n_vec];
                for k in 0..n_ldofs {
                    let dof_base = nodes[k] as usize * dim;
                    for d in 0..dim {
                        if dof_base + d < u.len() {
                            u_elem[k * dim + d] = u[dof_base + d];
                        }
                    }
                }

                // ── Deformation gradient F = I + ∇u ──
                let f = Self::def_grad(&u_elem, &gphys, dim, n_ldofs);

                // ── Green–Lagrange strain E = ½(FᵀF − I) ──
                let e_strain = Self::green_lagrange(&f);
                let mut e_strain_prev = vec![0.0; n_comp];
                let si = self.qp_state_idx(el, q);
                let alpha_prev = state[si][6];

                // Previous plastic strain (offset 0..n_comp)
                e_strain_prev[..n_comp].copy_from_slice(&state[si][..n_comp]);

                // ── Elastic trial strain ──
                let mut e_trial = vec![0.0; n_comp];
                for c in 0..n_comp {
                    e_trial[c] = e_strain[c] - e_strain_prev[c];
                }

                // ── J2 return mapping on (E, S) ──
                // Trial stress: S_trial = C : (E - E^p_prev)
                let mut s_trial = vec![0.0; n_comp];
                for i in 0..n_comp {
                    for j in 0..n_comp {
                        s_trial[i] += c_e[(i, j)] * e_trial[j];
                    }
                }

                // Deviatoric trial stress
                let tr = if is_2d { s_trial[0] + s_trial[1] }
                         else { s_trial[0] + s_trial[1] + s_trial[2] };
                let inv_dim = 1.0 / dim as f64;
                let mut s_dev = vec![0.0; n_comp];
                for i in 0..dim {
                    s_dev[i] = s_trial[i] - inv_dim * tr;
                }
                if is_2d { s_dev[2] = s_trial[2]; }
                else { s_dev[3..6].copy_from_slice(&s_trial[3..6]); }

                let eta_norm = (s_dev.iter().map(|v| v * v).sum::<f64>()
                    + 0.0).sqrt(); // full norm already summed over all comps

                let mut alpha_new = alpha_prev;
                let mut dgamma = 0.0_f64;
                let mut s_ep = s_trial.clone();  // elastic–plastic stress
                let mut c_ep = c_e.clone();       // elastic–plastic tangent

                let sqrt_23 = (2.0 / 3.0_f64).sqrt();
                let yield_val = eta_norm - sqrt_23 * (sigma_y + H * alpha_prev);

                if yield_val > 1e-12 {
                    // Plastic step: radial return
                    let denom = 1.0 + (H / (3.0 * mu));
                    dgamma = yield_val / (2.0 * mu * denom);
                    alpha_new = alpha_prev + sqrt_23 * dgamma;

                    // Scaled deviatoric stress after return
                    let factor = 1.0 - 2.0 * mu * dgamma / eta_norm.max(1e-300);
                    let mut s_new = vec![0.0; n_comp];
                    for i in 0..n_comp {
                        s_new[i] = s_dev[i] * factor;
                    }
                    for i in 0..dim {
                        s_ep[i] = s_new[i] + inv_dim * tr;
                    }
                    if is_2d { s_ep[2] = s_new[2]; }
                    else { s_ep[3..6].copy_from_slice(&s_new[3..6]); }

                    // Consistent tangent (material part) — beta reserved for fully consistent tangent
                    let _beta = 2.0 * mu * (1.0 - 2.0 * mu * dgamma / eta_norm.max(1e-300));
                    let dbeta = if dgamma > 1e-30 {
                        let t1 = 2.0 * mu / (eta_norm.max(1e-300));
                        t1 * (1.0 - 2.0 * mu * dgamma / eta_norm.max(1e-300))
                            - (2.0 * mu / (3.0 * mu + H)) * t1
                    } else { 0.0 };
                    let _ = dbeta; // reserved for fully consistent tangent

                    // Simplified consistent tangent: elastic-plastic continuum tangent
                    let mut c_ep_mat = c_e.clone();
                    // Subtract the deviatoric projection
                    for i in 0..n_comp {
                        for j in 0..n_comp {
                            c_ep_mat[(i, j)] -= 2.0 * mu * dgamma / eta_norm.max(1e-300)
                                * if is_2d {
                                    if i < 2 && j < 2 {
                                        if i == j { 2.0/3.0 } else { -1.0/3.0 }
                                    } else if i == 2 && j == 2 { 0.5 }
                                    else { 0.0 }
                                } else {
                                    let d_ij = if i == j { 1.0 } else { 0.0 };
                                    let d_2d = if i < 3 && j < 3 { 1.0/3.0 } else { 0.0 };
                                    d_ij - d_2d
                                };
                        }
                    }
                    c_ep = c_ep_mat;
                }

                // ── Store updated state (via RefCell) ──
                for c in 0..n_comp {
                    state[si][c] = e_strain_prev[c] + (if yield_val > 1e-12 {
                        // Plastic increment from return mapping
                        let factor = if dgamma > 0.0 { 3.0 * dgamma / (2.0 * eta_norm.max(1e-300)) } else { 0.0 };
                        s_dev[c] * factor
                    } else {
                        0.0  // pure elastic → no plastic inc
                    });
                }
                state[si][6] = alpha_new;

                // ── Residual and tangent assembly ──
                let w = quad.weights[q] * det_j.abs();

                // B-matrix: B[comp][ldof*dim+d] maps dof displacement to strain
                // For finite strain in TL formulation, the B-matrix is the
                // virtual strain operator δE = B·δu.
                // B_IJK = ½(F_ik · ∂N_K/∂X_j + F_jk · ∂N_K/∂X_i) for each node K

                // ── Helper closure: B-matrix entry B[node, disp_comp, voigt_comp] ──
                let b_entry = |node: usize, d: usize, comp: usize| -> f64 {
                    let (i, j) = if is_2d {
                        match comp { 0 => (0,0), 1 => (1,1), _ => (0,1) }
                    } else {
                        match comp { 0 => (0,0), 1 => (1,1), 2 => (2,2), 3 => (0,1), 4 => (1,2), 5 => (0,2), _ => (0,0) }
                    };
                    if i == j {
                        // Diagonal: δE_ii = F_di · ∂N/∂X_i
                        f[(d, i)] * gphys[node * dim + i]
                    } else {
                        // Shear: δE_ij = ½(F_di·∂N/∂X_j + F_dj·∂N/∂X_i)
                        0.5 * (f[(d, i)] * gphys[node * dim + j]
                             + f[(d, j)] * gphys[node * dim + i])
                    }
                };

                // ── Material tangent: K_mat += Bᵀ · C_ep · B · w ──
                for ki in 0..n_ldofs {
                    // Precompute B-row for node ki: b_ki[d][comp]
                    let mut b_ki = vec![vec![0.0; n_comp]; dim];
                    for d in 0..dim {
                        for comp in 0..n_comp {
                            b_ki[d][comp] = b_entry(ki, d, comp);
                        }
                    }
                    for kj in 0..n_ldofs {
                        // Precompute B-row for node kj
                        let mut b_kj = vec![vec![0.0; n_comp]; dim];
                        for d in 0..dim {
                            for comp in 0..n_comp {
                                b_kj[d][comp] = b_entry(kj, d, comp);
                            }
                        }
                        // Full Voigt coupling: K_mat[ki*dim+d_row, kj*dim+d_col]
                        for d_row in 0..dim {
                            let row_base = nodes[ki] as usize * dim + d_row;
                            for d_col in 0..dim {
                                let mut k_val = 0.0;
                                for p in 0..n_comp {
                                    let bp = b_ki[d_row][p];
                                    if bp == 0.0 { continue; }
                                    for q in 0..n_comp {
                                        k_val += bp * c_ep[(p, q)] * b_kj[d_col][q];
                                    }
                                }
                                k_val *= w;
                                if k_val.abs() > 1e-30 {
                                    coo.add(row_base, nodes[kj] as usize * dim + d_col, k_val);
                                }
                            }
                        }
                    }
                }

                // ── Geometric stiffness: K_geo += Gᵀ · S · G · w ──
                // G maps displacement to the gradient ∇u (used in F = I + ∇u).
                // K_geo[ki, kj] = (Σ_αβ N_ki,α · S_αβ · N_kj,β) · I_dim × w
                for ki in 0..n_ldofs {
                    for kj in 0..n_ldofs {
                        let mut k_geo = 0.0;
                        for alpha in 0..dim {
                            for beta in 0..dim {
                                k_geo += gphys[ki * dim + alpha]
                                       * s_ep[if is_2d {
                                           if alpha == beta { alpha }
                                           else if alpha + beta == 1 { 2 }
                                           else { 0 }
                                       } else {
                                           match (alpha, beta) {
                                               (0,0) => 0, (1,1) => 1, (2,2) => 2,
                                               (0,1) | (1,0) => 3,
                                               (1,2) | (2,1) => 4,
                                               (0,2) | (2,0) => 5,
                                               _ => 0,
                                           }
                                       }]
                                       * gphys[kj * dim + beta];
                            }
                        }
                        k_geo *= w;

                        if k_geo.abs() > 1e-30 {
                            let nodes = mesh.element_nodes(e);
                            let row_base = nodes[ki] as usize * dim;
                            let col_base = nodes[kj] as usize * dim;
                            for d in 0..dim {
                                coo.add(row_base + d, col_base + d, k_geo);
                            }
                        }
                    }
                }

                // ── Internal force (residual) ──
                if want_residual {
                    for k in 0..n_ldofs {
                        for comp in 0..n_comp {
                            let (i, j) = if is_2d {
                                match comp { 0 => (0,0), 1 => (1,1), _ => (0,1) }
                            } else {
                                match comp { 0 => (0,0), 1 => (1,1), 2 => (2,2), 3 => (0,1), 4 => (1,2), 5 => (0,2), _ => (0,0) }
                            };
                            let mut b_val = 0.0;
                            let mut _b_other = 0.0;
                            bi_comp(k, &f, &gphys, dim, comp, &mut b_val, &mut _b_other);
                            let nodes = mesh.element_nodes(e);
                            let dof_base = nodes[k] as usize * dim;
                            let s_val = s_ep[comp] * b_val * w;
                            // Distribute to the two DOFs associated with this component
                            // For diagonal components (i==j), the contribution goes to dof[i]
                            // For shear, both i and j DOFs contribute
                            if i == j {
                                f_vec[dof_base + i] -= s_val;
                            } else {
                                // For shear: ½(F_iα·N,α + F_jα·N,α) contributes to both DOFs
                                // The B-matrix component B_{comp} = ½(F_iα N_α + F_jα N_α)
                                // and the internal force = B_{comp} · S_{comp} · w
                                // This distributes to both DOF[i] and DOF[j]
                                f_vec[dof_base + i] -= s_val * 0.5;
                                f_vec[dof_base + j] -= s_val * 0.5;
                            }
                        }
                    }
                }
            }
        }

        // State is already updated in-place via Mutex above.

        (f_vec, coo.into_csr())
    }
}

impl<M: MeshTopology + 'static> NonlinearForm for FiniteStrainPlasticity<M> {
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

/// Helper: compute the B-matrix (virtual strain) component for node k.
/// Fills `b_ik` = B_I(comp, ·) with the contribution of node I to component `comp`.
fn bi_comp(
    k: usize, f: &DMatrix<f64>, gphys: &[f64], dim: usize,
    comp: usize, b_val: &mut f64, _b_other: &mut f64,
) {
    let (i, j) = if dim == 2 {
        match comp { 0 => (0,0), 1 => (1,1), _ => (0,1) }
    } else {
        match comp { 0 => (0,0), 1 => (1,1), 2 => (2,2), 3 => (0,1), 4 => (1,2), 5 => (0,2), _ => (0,0) }
    };
    *b_val = 0.0;
    for alpha in 0..dim {
        *b_val += 0.5 * (f[(i, alpha)] * gphys[k * dim + alpha]
                        + f[(j, alpha)] * gphys[k * dim + alpha]);
    }
    *_b_other = *b_val;
}

/// Compute the Jacobian of a simplex element.
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

// ─── Principal stress helpers ────────────────────────────────────────────

/// Eigenvalues of a 3×3 symmetric matrix (analytical trigonometric formula).
fn eig_sym_3x3(a00: f64, a01: f64, a02: f64,
               a11: f64, a12: f64, a22: f64) -> [f64; 3] {
    let i1 = a00 + a11 + a22;
    let i2 = a00*a11 + a11*a22 + a00*a22 - a01*a01 - a12*a12 - a02*a02;
    let i3 = a00*(a11*a22 - a12*a12) - a01*(a01*a22 - a02*a12) + a02*(a01*a12 - a02*a11);
    let p = i1 / 3.0;
    let q_val = (i1*i1 - 3.0*i2) / 9.0;
    if q_val <= 1e-60 {
        return [p, p, p];
    }
    let r = (2.0*i1*i1*i1 - 9.0*i1*i2 + 27.0*i3) / 54.0;
    let q_sqrt = q_val.sqrt();
    let phi = (r / (q_sqrt*q_sqrt*q_sqrt)).clamp(-1.0, 1.0).acos() / 3.0;
    let two_q = 2.0 * q_sqrt;
    [
        p + two_q * phi.cos(),
        p + two_q * (phi - 2.0*std::f64::consts::PI / 3.0).cos(),
        p + two_q * (phi + 2.0*std::f64::consts::PI / 3.0).cos(),
    ]
}

/// Principal stresses from Voigt stress (3D: 6 components, or 2D: 3 components).
/// Returns `[σ₁, σ₂, σ₃]` with σ₁ ≥ σ₂ ≥ σ₃ (tension positive).
fn principal_stresses(sigma: &[f64], dim: usize) -> [f64; 3] {
    if dim == 2 {
        // Plane strain: σ = [σ_xx, σ_yy, τ_xy]. σ_zz = ν·(σ_xx + σ_yy) but
        // we use the in-plane principal values (σ₃ is out-of-plane).
        let sxx = sigma[0]; let syy = sigma[1]; let sxy = sigma[2];
        let avg = 0.5 * (sxx + syy);
        let rad = ((0.5*(sxx - syy)).powi(2) + sxy*sxy).sqrt();
        let s1 = avg + rad;
        let s2 = avg - rad;
        // σ₃ is not uniquely defined in plane strain — approximate with
        // the intermediate principal value for yield checks.
        let s3 = if dim == 2 { 0.0 } else { sigma[2] };
        let mut v = [s1, s2, s3];
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        v
    } else {
        let mut e = eig_sym_3x3(
            sigma[0], sigma[3], sigma[5],
            sigma[1], sigma[4], sigma[2],
        );
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        e
    }
}

// ─── Model-specific yield functions ──────────────────────────────────────

/// Mohr–Coulomb yield function in principal stress space.
///
/// `f = σ₁·(1 + sinφ) − σ₃·(1 − sinφ) − 2·c·cosφ`
fn mohr_coulomb_yield(prin: &[f64; 3], cohesion: f64, sin_phi: f64, cos_phi: f64) -> f64 {
    let s1 = prin[0];
    let s3 = prin[2];
    s1 * (1.0 + sin_phi) - s3 * (1.0 - sin_phi) - 2.0 * cohesion * cos_phi
}

/// Hoek–Brown yield function in principal stress space.
///
/// `f = σ₁ − σ₃ − σ_ci·(m_b·σ₃/σ_ci + s)^a`
fn hoek_brown_yield(prin: &[f64; 3], ci: f64, mb: f64, s: f64, a: f64) -> f64 {
    let s1 = prin[0];
    let s3 = prin[2];
    let arg = mb * s3 / ci + s;
    if arg <= 0.0 {
        // Tension regime: linear extrapolation below the cutoff
        s1 - s3
    } else {
        s1 - s3 - ci * arg.powf(a)
    }
}

/// Modified Cam–Clay yield function in invariant space.
///
/// `f = q² + M²·p·(p − p_c)`
fn cam_clay_yield(p: f64, q: f64, pc: f64, m: f64) -> f64 {
    q * q + m * m * p * (p - pc)
}

// ─── Return-mapping helpers (cutting-plane / Newton) ─────────────────────

/// Mohr–Coulomb cutting-plane return mapping in principal stress space.
///
/// Returns `(sigma_returned, dgamma)` where `sigma_returned` is the
/// updated stress in Voigt form (same layout as input).
fn mc_return_mapping(
    sigma_trial: &[f64], dim: usize,
    cohesion: f64, sin_phi: f64, cos_phi: f64,
    sin_psi: f64,
) -> (Vec<f64>, f64) {
    let mut sigma = sigma_trial.to_vec();
    let prin = principal_stresses(sigma_trial, dim);
    let f_trial = mohr_coulomb_yield(&prin, cohesion, sin_phi, cos_phi);
    if f_trial <= 0.0 {
        return (sigma, 0.0);
    }

    // Yield-function gradient  n_f = ∂f/∂σ  (from friction angle φ)
    // Plastic-potential gradient  n_g = ∂g/∂σ  (from dilation angle ψ)
    // For associated flow ψ = φ → n_g = n_f.
    let n1_f = 1.0 + sin_phi;
    let n3_f = -(1.0 - sin_phi);
    let n1_g = 1.0 + sin_psi;
    let n3_g = -(1.0 - sin_psi);
    // Mixed denominator:  n_f · C · n_g  (simplified to identity norm)
    let denom = n1_f * n1_g + n3_f * n3_g;
    if denom > 1e-30 {
        let dgamma = f_trial / denom;
        // Update σ₁ and σ₃ using the POTENTIAL gradient n_g (dilation ψ).
        let mut prin_new = prin;
        prin_new[0] -= dgamma * n1_g;
        prin_new[2] -= dgamma * n3_g;
        if prin_new[2] > prin_new[1] { // re-sort if needed
            prin_new.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }

        // Reconstruct Voigt stress from principal stresses (approximate:
        // preserve the original eigen-directions by scaling)
        let (s0, s1, s2, s3, s4, s5) = if dim == 2 {
            (sigma_trial[0], sigma_trial[1], 0.0, sigma_trial[2], 0.0, 0.0)
        } else {
            (sigma_trial[0], sigma_trial[1], sigma_trial[2], sigma_trial[3], sigma_trial[4], sigma_trial[5])
        };
        let eig_sorted = eig_sym_3x3(s0, s3, s5, s1, s4, s2);
        // Use the same relative shifts as the principal changes
        let shift = [prin_new[0] - eig_sorted[0],
                     prin_new[1] - eig_sorted[1],
                     prin_new[2] - eig_sorted[2]];
        sigma[0] += shift[0] * 0.5; // approximate: distribute shifts
        sigma[1] += shift[1] * 0.5;
        sigma[2] += shift[2] * 0.5;
        // For off-diagonal, scale proportionally
        let ratio = if eig_sorted[0] - eig_sorted[2] > 1e-30 {
            (prin_new[0] - prin_new[2]) / (eig_sorted[0] - eig_sorted[2])
        } else { 1.0 };
        if dim == 3 {
            sigma[3] *= ratio;
            sigma[4] *= ratio;
            sigma[5] *= ratio;
        } else {
            sigma[2] *= ratio;
        }
        (sigma, dgamma)
    } else {
        (sigma, 0.0)
    }
}

/// Hoek–Brown cutting-plane return mapping in principal stress space.
fn hb_return_mapping(
    sigma_trial: &[f64], dim: usize,
    ci: f64, mb: f64, s: f64, a: f64,
) -> (Vec<f64>, f64) {
    let mut sigma = sigma_trial.to_vec();
    let prin = principal_stresses(sigma_trial, dim);
    let f_trial = hoek_brown_yield(&prin, ci, mb, s, a);
    if f_trial <= 0.0 {
        return (sigma, 0.0);
    }

    // Cutting-plane: ∂f/∂σ in principal space
    // f = σ₁ - σ₃ - ci·(mb·σ₃/ci + s)^a
    // ∂f/∂σ₁ = 1
    // ∂f/∂σ₃ = -1 - mb·a·(mb·σ₃/ci + s)^(a-1)
    let arg = (mb * prin[2] / ci + s).max(0.0);
    let df_ds3 = if arg > 0.0 {
        -1.0 - mb * a * arg.powf(a - 1.0)
    } else {
        -1.0
    };
    let denom = 1.0 + df_ds3 * df_ds3; // n₁² + n₃²
    if denom > 1e-30 {
        let dgamma = f_trial / denom;
        let mut prin_new = prin;
        prin_new[0] -= dgamma * 1.0; // ∂f/∂σ₁
        prin_new[2] -= dgamma * df_ds3;
        if prin_new[2] > prin_new[1] {
            prin_new.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }

        let (s0, s1, s2, s3, s4, s5) = if dim == 2 {
            (sigma_trial[0], sigma_trial[1], 0.0, sigma_trial[2], 0.0, 0.0)
        } else {
            (sigma_trial[0], sigma_trial[1], sigma_trial[2], sigma_trial[3], sigma_trial[4], sigma_trial[5])
        };
        let eig_sorted = eig_sym_3x3(s0, s3, s5, s1, s4, s2);
        let shift = [prin_new[0] - eig_sorted[0],
                     prin_new[1] - eig_sorted[1],
                     prin_new[2] - eig_sorted[2]];
        sigma[0] += shift[0] * 0.5;
        sigma[1] += shift[1] * 0.5;
        sigma[2] += shift[2] * 0.5;
        let ratio = if eig_sorted[0] - eig_sorted[2] > 1e-30 {
            (prin_new[0] - prin_new[2]) / (eig_sorted[0] - eig_sorted[2])
        } else { 1.0 };
        if dim == 3 {
            sigma[3] *= ratio;
            sigma[4] *= ratio;
            sigma[5] *= ratio;
        } else {
            sigma[2] *= ratio;
        }
        (sigma, dgamma)
    } else {
        (sigma, 0.0)
    }
}

/// Modified Cam–Clay return mapping: Newton iteration in (p, q) space.
///
/// Returns `(sigma_returned, dgamma, new_pc, p_returned, q_returned)`.
#[allow(clippy::too_many_arguments)]
fn cc_return_mapping(
    sigma_trial: &[f64], dim: usize,
    m: f64, pc_old: f64, lambda_idx: f64, kappa_idx: f64, void_ratio: f64,
    mu: f64, bulk: f64,
) -> (Vec<f64>, f64, f64, f64, f64) {
    let n_comp = if dim == 2 { 3 } else { 6 };
    let mut sigma = sigma_trial.to_vec();
    // Mean stress and deviatoric
    let p_trial = if dim == 2 {
        (sigma_trial[0] + sigma_trial[1]) / 3.0
    } else {
        (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0
    };
    let mut s_dev = vec![0.0; n_comp];
    for i in 0..dim { s_dev[i] = sigma_trial[i] - p_trial; }
    if dim == 2 { s_dev[2] = sigma_trial[2]; }
    else { s_dev[3..6].copy_from_slice(&sigma_trial[3..6]); }
    let q_trial = (s_dev.iter().map(|v| v*v).sum::<f64>()).sqrt() * (3.0_f64 / 2.0_f64).sqrt();

    let f_trial = cam_clay_yield(p_trial, q_trial, pc_old, m);
    if f_trial <= 0.0 {
        return (sigma, 0.0, pc_old, p_trial, q_trial);
    }

    // Newton iteration on the return mapping in (p, q) space
    // with volumetric hardening (p_c evolves with ε_v^p).
    // f(p, q, pc) = q² + M²·p·(p - pc) = 0
    // p = p_trial - K·Δε_v^p
    // q = q_trial - 3G·Δε_s^p
    // Δε_v^p = Δλ·∂f/∂p = Δλ·M²·(2p - pc)
    // Δε_s^p = Δλ·∂f/∂q = Δλ·2q
    // dp_c/dΔλ = pc·(1+e₀)/(λ-κ) · ∂f/∂p  (hardening law)
    let v = 1.0 + void_ratio;
    let hard_slope = (lambda_idx - kappa_idx).max(1e-30);
    let three_g = 3.0 * mu;
    let mut p = p_trial;
    let mut q = q_trial;
    let mut pc = pc_old;
    let mut dlambda = 0.0;

    for _iter in 0..30 {
        let df_dp = m * m * (2.0 * p - pc);
        let df_dq = 2.0 * q;
        // f = q² + M²·p·(p-pc)
        let f_val = q * q + m * m * p * (p - pc);
        if f_val.abs() < 1e-12 { break; }

        // Jacobian of the system: only the denominator of Δλ update is needed.
        let denom = df_dp * (-bulk * df_dp) + df_dq * (-three_g * df_dq) + df_dp * (pc * v / hard_slope * df_dp);
        if denom.abs() < 1e-30 { break; }
        let ddlambda = -f_val / denom;
        dlambda += ddlambda;
        if ddlambda.abs() < 1e-14 { break; }

        // Update p, q, pc
        p = p_trial - bulk * dlambda * df_dp;
        q = q_trial - three_g * dlambda * df_dq;
        pc = pc_old * (v * dlambda * df_dp / hard_slope).exp();
    }

    // Reconstruct stress from (p, q)
    let q_new = q.max(0.0);
    let scale = if q_trial > 1e-30 { q_new / q_trial } else { 0.0 };
    for i in 0..dim { sigma[i] = s_dev[i] * scale + p; }
    if dim == 2 { sigma[2] = s_dev[2] * scale; }
    else { for i in 3..6 { sigma[i] = s_dev[i] * scale; } }

    (sigma, dlambda, pc, p, q)
}

// ─── Consistent tangent helpers ─────────────────────────────────────────

/// Eigenvectors of a 3×3 symmetric matrix (analytical).
/// Returns 3 eigenvectors as a flat array `[v1_x, v1_y, v1_z, v2_x, ...]`.
fn eigvec_sym_3x3(a00: f64, a01: f64, a02: f64,
                  a11: f64, a12: f64, a22: f64) -> [f64; 9] {
    let evals = eig_sym_3x3(a00, a01, a02, a11, a12, a22);
    // Build matrix A - λI for each eigenvalue and solve for eigenvector
    // via the cross-product of two rows.
    let mut vecs = [0.0_f64; 9];
    for k in 0..3 {
        let lam = evals[k];
        let b00 = a00 - lam; let b01 = a01; let b02 = a02;
        let b11 = a11 - lam; let b12 = a12;
        let _b22 = a22 - lam;
        // First two rows of (A - λI)
        let r0 = [b00, b01, b02];
        let r1 = [b01, b11, b12];
        // Cross product of r0 × r1 gives the eigenvector (up to scale)
        let mut v = [
            r0[1]*r1[2] - r0[2]*r1[1],
            r0[2]*r1[0] - r0[0]*r1[2],
            r0[0]*r1[1] - r0[1]*r1[0],
        ];
        let vn = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt().max(1e-30);
        for c in 0..3 { v[c] /= vn; }
        vecs[k*3..k*3+3].copy_from_slice(&v);
    }
    vecs
}

/// Principal directions (eigenvectors) of the stress tensor in Voigt form.
/// Returns 3 normalised eigenvectors as flat `[v1_x..v1_z, v2_x.., v3_x..]`.
fn principal_directions(sigma: &[f64], dim: usize) -> [f64; 9] {
    let (s0, s1, s2, s3, s4, s5) = if dim == 2 {
        (sigma[0], sigma[1], 0.0, sigma[2], 0.0, 0.0)
    } else {
        (sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5])
    };
    eigvec_sym_3x3(s0, s3, s5, s1, s4, s2)
}

/// Project a principal‑space flow gradient `[n1, n2, n3]` to Voigt form
/// using the provided eigenvectors (flat 9‑element array).
fn flow_voigt(n_prin: &[f64; 3], eigvecs: &[f64; 9], dim: usize) -> Vec<f64> {
    let n_comp = if dim == 2 { 3 } else { 6 };
    let mut r = vec![0.0; n_comp];
    for i in 0..3 {
        let ni = n_prin[i];
        if ni.abs() < 1e-30 { continue; }
        let vx = eigvecs[i*3];
        let vy = eigvecs[i*3+1];
        let vz = eigvecs[i*3+2];
        r[0] += ni * vx * vx;
        r[1] += ni * vy * vy;
        if dim == 3 { r[2] += ni * vz * vz; }
        if dim == 2 {
            r[2] += ni * vx * vy;
        } else {
            r[3] += ni * vx * vy;
            r[4] += ni * vy * vz;
            r[5] += ni * vx * vz;
        }
    }
    r
}

/// Continuum elasto‑plastic tangent: `D_ep = C_e - (C_e·r_g ⊗ r_f·C_e) / H`
/// where `r_f` is the yield‑function gradient (Voigt) and `r_g` the
/// plastic‑potential gradient (Voigt).  When `r_f == r_g` (associated flow)
/// the tangent is symmetric.
fn tangent_elasto_plastic(
    c_e: &DMatrix<f64>, r_f: &[f64], r_g: &[f64], n_comp: usize,
    h_prime: f64,
) -> DMatrix<f64> {
    let mut dep = c_e.clone();
    // Ce·r_g  and  r_f·Ce
    let mut ce_rg = vec![0.0; n_comp];
    let mut rf_ce = vec![0.0; n_comp];
    for i in 0..n_comp {
        for j in 0..n_comp {
            ce_rg[i] += c_e[(i, j)] * r_g[j];
            rf_ce[i] += r_f[j] * c_e[(j, i)];  // r_f·Ce = (Ce·r_f)ᵀ
        }
    }
    // r_f · Ce · r_g
    let mut denom = h_prime;
    for i in 0..n_comp { denom += rf_ce[i] * r_g[i]; }
    if denom.abs() < 1e-30 { return dep; }

    for i in 0..n_comp {
        for j in 0..n_comp {
            dep[(i, j)] -= ce_rg[i] * rf_ce[j] / denom;
        }
    }
    dep
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

    // ── J2 state evolution tests ──────────────────────────────────────────

    /// Apply a uniform tensile strain large enough to trigger plasticity,
    /// then verify that the internal state (α) has evolved.
    /// We apply u_x(x,y) = 0.1·x on a unit square so ε_xx = 0.1.
    #[test]
    fn j2_state_evolution_plastic_step() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3); // low yield → plastic
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);

        // Build a displacement field: u = (0.1·x, 0)
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.1 * x;
        }

        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        // After assembly, check that state has been updated
        let state = form.state.lock().unwrap();
        let any_plastic = state.iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "Plastic step should produce non-zero α in at least one QP");
    }

    #[test]
    fn j2_elastic_step_state_unchanged() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e8, 0.0); // very high yield → elastic
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();

        // Small displacement that stays elastic: u = (0.0, 0.0)
        let u = vec![0.0; n];
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        let state = form.state.lock().unwrap();
        let total_alpha: f64 = state.iter().map(|qp| qp[6]).sum();
        assert!(total_alpha < 1e-30, "Elastic step: α should remain zero, got {total_alpha:.3e}");
    }

    // ── Finite-strain plasticity tests ─────────────────────────────────────

    #[test]
    fn finite_strain_zero_displacement_zero_residual() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e8, 0.0); // elastic
        let form = FiniteStrainPlasticity::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let mut r = vec![0.0; n];
        let rhs = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let norm: f64 = r.iter().map(|v| v.abs()).sum();
        assert!(norm < 1e-12, "zero u → zero residual, got {norm:.3e}");
    }

    #[test]
    fn finite_strain_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e8, 0.0);
        let form = FiniteStrainPlasticity::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let jac = form.jacobian(&u);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) {
                sum += jac.get(i, j).abs();
            }
        }
        assert!(sum > 0.0, "finite-strain tangent should be non-zero");
    }

    #[test]
    fn finite_strain_plastic_step_updates_state() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3); // low yield
        let form = FiniteStrainPlasticity::new(space, cfg, vec![], 2);

        // Apply tensile displacement u = (0.2·x, 0)
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.2 * x;
        }

        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        let state = form.state.lock().unwrap();
        let any_plastic = state.iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "Plastic step should produce non-zero α");
    }

    #[test]
    fn finite_strain_state_persistence_across_calls() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::j2(2e5, 0.3, 50.0, 5e2);
        let form = FiniteStrainPlasticity::new(space, cfg, vec![], 2);

        // First call: plastic step
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.15 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        // Snapshot α after first call
        let alpha_after_first: Vec<f64> = form.state.lock().unwrap()
            .iter().map(|qp| qp[6]).collect();
        let any_plastic_first = alpha_after_first.iter().any(|&a| a > 1e-12);
        assert!(any_plastic_first, "α should be > 0 after plastic step");

        // Second call with same displacement: state should NOT decrease
        // (α is montononic in J2 plasticity)
        form.residual(&u, &rhs, &mut r);
        let alpha_after_second: Vec<f64> = form.state.lock().unwrap()
            .iter().map(|qp| qp[6]).collect();

        for (i, (&a1, &a2)) in alpha_after_first.iter().zip(alpha_after_second.iter()).enumerate() {
            assert!(a2 >= a1 - 1e-14,
                "α should be monotonic non-decreasing: QP {i}: {a1} → {a2}");
        }
    }

    // ── Drucker–Prager dispatch regression test (Phase 0.1 fix) ────────────

    /// Verifies that `PlasticConfig::drucker_prager(...)` actually selects
    /// the DP return-mapping path, not silently degrading to J2.
    ///
    /// Setup: apply a hydrostatic-heavy strain field. In J2, hydrostatic
    /// stress does NOT contribute to yielding (deviatoric only), so a purely
    /// dilational strain never triggers plasticity. In DP, `f = α·I₁ + ‖s‖ − k·c`
    /// includes the volumetric term α·I₁, so a large positive I₁ WILL trigger
    /// plasticity if the cohesion is low enough. If the model silently ran J2,
    /// no plastic evolution would occur.
    #[test]
    fn drucker_prager_dispatches_to_dp_branch_not_j2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();

        // Low cohesion + realistic friction angle → hydrostatic tension yields.
        let cfg = PlasticConfig::drucker_prager(2e5, 0.3, 10.0, 30.0, 0.0);
        assert_eq!(cfg.model, PlasticModel::DruckerPrager,
            "constructor must set model = DruckerPrager");
        assert!(cfg.dp_alpha() > 0.0, "DP α must be positive for φ=30°");

        let form = J2PlasticityForm::new(space, cfg, vec![], 2);

        // Apply purely dilational strain: u = (0.05·x, 0.05·y)
        // → ε_xx = ε_yy = 0.05, γ_xy = 0. Deviatoric part is ZERO;
        //   J2 will never yield, DP will yield due to α·I₁ term.
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form.space.mesh().node_coords(i as u32);
            let (x, y) = (c[0], c[1]);
            u[i * 2]     = 0.05 * x;
            u[i * 2 + 1] = 0.05 * y;
        }

        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        let any_plastic = form.state.lock().unwrap()
            .iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic,
            "DP path must yield under hydrostatic tension (α·I₁ term). \
             If J2 ran silently, α would remain zero everywhere.");
    }

    /// Compare J2 vs DP with identical dev-heavy loading (both should yield);
    /// they must produce DIFFERENT internal-variable evolution because DP
    /// includes the volumetric term while J2 does not.
    #[test]
    fn drucker_prager_and_j2_diverge_under_dev_heavy_load() {
        let mesh1 = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = mesh1.clone();
        let space1 = VectorH1Space::new(mesh1, 1, 2);
        let space2 = VectorH1Space::new(mesh2, 1, 2);
        let n = space1.n_dofs();

        let cfg_j2 = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3);
        let cfg_dp = PlasticConfig::drucker_prager(2e5, 0.3, 50.0, 30.0, 1e3);
        let form_j2 = J2PlasticityForm::new(space1, cfg_j2, vec![], 2);
        let form_dp = J2PlasticityForm::new(space2, cfg_dp, vec![], 2);

        // Deviatoric-dominant load: u = (0.1·x, -0.1·y) → ε_xx = 0.1, ε_yy = -0.1
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form_j2.space.mesh().node_coords(i as u32);
            let (x, y) = (c[0], c[1]);
            u[i * 2]     =  0.1 * x;
            u[i * 2 + 1] = -0.1 * y;
        }
        let rhs = vec![0.0; n];
        let mut r1 = vec![0.0; n];
        let mut r2 = vec![0.0; n];
        form_j2.residual(&u, &rhs, &mut r1);
        form_dp.residual(&u, &rhs, &mut r2);

        let a_j2: f64 = form_j2.state.lock().unwrap().iter().map(|qp| qp[6]).sum();
        let a_dp: f64 = form_dp.state.lock().unwrap().iter().map(|qp| qp[6]).sum();

        // Both must yield.
        assert!(a_j2 > 1e-8, "J2 must produce plastic strain under dev load");
        assert!(a_dp > 1e-8, "DP must produce plastic strain under dev load");
        // They MUST differ (different yield surfaces).
        assert!((a_j2 - a_dp).abs() > 1e-6,
            "J2 and DP must produce different plastic strain: J2={a_j2:.6e}, DP={a_dp:.6e}");
    }

    // ── Principal-stress helper tests ─────────────────────────────────

    #[test]
    fn principal_stresses_diagonal_3d() {
        let sigma = vec![3.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        let p = principal_stresses(&sigma, 3);
        assert!((p[0] - 3.0).abs() < 1e-12, "σ₁=3, got {}", p[0]);
        assert!((p[1] - 2.0).abs() < 1e-12, "σ₂=2, got {}", p[1]);
        assert!((p[2] - 1.0).abs() < 1e-12, "σ₃=1, got {}", p[2]);
    }

    #[test]
    fn principal_stresses_2d_known() {
        let sigma = vec![3.0, 1.0, 1.0]; // σ_xx=3, σ_yy=1, τ_xy=1
        let p = principal_stresses(&sigma, 2);
        let s1 = 2.0 + 2.0_f64.sqrt();
        let s2 = 2.0 - 2.0_f64.sqrt();
        assert!((p[0] - s1).abs() < 1e-12, "σ₁={s1}, got {}", p[0]);
        assert!((p[1] - s2).abs() < 1e-12, "σ₂={s2}, got {}", p[1]);
    }

    // ── Mohr–Coulomb tests ────────────────────────────────────────────

    #[test]
    fn mohr_coulomb_elastic_step_zero_residual() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::mohr_coulomb(2e5, 0.3, 1e6, 30.0, 30.0, 0.0);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let mut r = vec![0.0; n];
        let rhs = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let norm: f64 = r.iter().map(|v| v.abs()).sum();
        assert!(norm < 1e-12, "MC zero-u residual: {norm:.3e}");
    }

    #[test]
    fn mohr_coulomb_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::mohr_coulomb(2e5, 0.3, 1e6, 30.0, 30.0, 0.0);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let jac = form.jacobian(&u);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) { sum += jac.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "MC tangent non-zero");
    }

    #[test]
    fn mohr_coulomb_yields_under_hydrostatic_tension() {
        // MC (like DP) yields under hydrostatic tension; J2 does not.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::mohr_coulomb(2e5, 0.3, 10.0, 30.0, 30.0, 0.0);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form.space.mesh().node_coords(i as u32);
            u[i * 2] = 0.05 * c[0];
            u[i * 2 + 1] = 0.05 * c[1];
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let any_plastic = form.state.lock().unwrap()
            .iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "MC must yield under hydrostatic tension");
    }

    #[test]
    fn mohr_coulomb_plastic_step() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::mohr_coulomb(2e5, 0.3, 50.0, 30.0, 30.0, 1e3);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.15 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let any_plastic = form.state.lock().unwrap()
            .iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "MC plastic step should produce α > 0");
    }

    // ── Hoek–Brown tests ──────────────────────────────────────────────

    #[test]
    fn hoek_brown_elastic_step_zero_residual() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        // Intact granite: σ_ci=200MPa, m_i=25, s=1, a=0.5
        let cfg = PlasticConfig::hoek_brown(7e4, 0.25, 200.0, 25.0, 1.0, 0.5);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let mut r = vec![0.0; n];
        let rhs = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let norm: f64 = r.iter().map(|v| v.abs()).sum();
        assert!(norm < 1e-12, "HB zero-u residual: {norm:.3e}");
    }

    #[test]
    fn hoek_brown_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::hoek_brown(7e4, 0.25, 200.0, 25.0, 1.0, 0.5);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let jac = form.jacobian(&u);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) { sum += jac.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "HB tangent non-zero");
    }

    #[test]
    fn hoek_brown_yields_under_tension() {
        // Hoek–Brown with low σ_ci should yield under tensile loading.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::hoek_brown(7e4, 0.25, 20.0, 25.0, 1.0, 0.5);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.1 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let any_plastic = form.state.lock().unwrap()
            .iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "HB should yield under tensile strain");
    }

    // ── Cam–Clay tests ────────────────────────────────────────────────

    #[test]
    fn cam_clay_elastic_step_zero_residual() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        // Typical soil: M=1.2, p_c0=200kPa, λ=0.15, κ=0.03, e₀=0.9
        let cfg = PlasticConfig::cam_clay(1e4, 0.3, 1.2, 200.0, 0.15, 0.03, 0.9);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let mut r = vec![0.0; n];
        let rhs = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let norm: f64 = r.iter().map(|v| v.abs()).sum();
        assert!(norm < 1e-12, "CC zero-u residual: {norm:.3e}");
    }

    #[test]
    fn cam_clay_tangent_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::cam_clay(1e4, 0.3, 1.2, 200.0, 0.15, 0.03, 0.9);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        let jac = form.jacobian(&u);
        let mut sum = 0.0;
        for i in 0..n.min(10) {
            for j in 0..n.min(10) { sum += jac.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "CC tangent non-zero");
    }

    #[test]
    fn cam_clay_yields_under_compression() {
        // Cam–Clay should yield when the stress path crosses the yield surface.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        // High M, low p_c0 → easily yields
        let cfg = PlasticConfig::cam_clay(1e4, 0.3, 1.5, 50.0, 0.15, 0.03, 0.9);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        // Apply compressive deviatoric + volumetric strain
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form.space.mesh().node_coords(i as u32);
            u[i * 2] = -0.03 * c[0];
            u[i * 2 + 1] = -0.03 * c[1];
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        let any_plastic = form.state.lock().unwrap()
            .iter().any(|qp| qp[6] > 1e-12);
        assert!(any_plastic, "CC should yield under compressive loading");
    }

    // ── Non‑associated flow tests ─────────────────────────────────────

    /// DP with ψ = φ (associated) should produce a DIFFERENT plastic
    /// response than DP with ψ = 0 (non‑associated) under the same
    /// deviator‑dominated load.  This proves the dilation angle is
    /// actually dispatched in the return mapping.
    #[test]
    fn dp_non_associated_produces_different_response() {
        let mesh_a = SimplexMesh::<2>::unit_square_tri(4);
        let mesh_n = mesh_a.clone();
        let space_a = VectorH1Space::new(mesh_a, 1, 2);
        let space_n = VectorH1Space::new(mesh_n, 1, 2);
        let n = space_a.n_dofs();

        let cfg_a = PlasticConfig::drucker_prager_general(2e5, 0.3, 20.0, 30.0, 30.0, 500.0);
        let cfg_n = PlasticConfig::drucker_prager_general(2e5, 0.3, 20.0, 30.0, 0.0, 500.0);
        let form_a = J2PlasticityForm::new(space_a, cfg_a, vec![], 2);
        let form_n = J2PlasticityForm::new(space_n, cfg_n, vec![], 2);

        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form_a.space.mesh().node_coords(i as u32);
            u[i * 2]     =  0.12 * c[0];
            u[i * 2 + 1] = -0.12 * c[1];
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form_a.residual(&u, &rhs, &mut r);
        form_n.residual(&u, &rhs, &mut r);

        let a_a = form_a.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        let a_n = form_n.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();

        assert!(a_a > 1e-8, "Associated DP must yield, got α={a_a:.6e}");
        assert!(a_n > 1e-8, "Non-associated DP must yield, got α={a_n:.6e}");
        assert!((a_a - a_n).abs() > 1e-10,
            "Associated and non-associated DP should give different α: assoc={a_a:.6e}, non={a_n:.6e}");
    }

    /// MC associated (ψ=φ) vs non‑associated (ψ=0): same deviator load,
    /// the non‑associated case must produce different (ideally less
    /// volumetric) plastic strain.
    #[test]
    fn mc_non_associated_produces_different_plastic_response() {
        let mesh_a = SimplexMesh::<2>::unit_square_tri(4);
        let mesh_n = mesh_a.clone();
        let space_a = VectorH1Space::new(mesh_a, 1, 2);
        let space_n = VectorH1Space::new(mesh_n, 1, 2);
        let n = space_a.n_dofs();

        // Same MC material, same φ, different ψ
        let cfg_a = PlasticConfig::mohr_coulomb(2e5, 0.3, 20.0, 30.0, 30.0, 500.0);
        let cfg_n = PlasticConfig::mohr_coulomb(2e5, 0.3, 20.0, 30.0, 0.0, 500.0);
        let form_a = J2PlasticityForm::new(space_a, cfg_a, vec![], 2);
        let form_n = J2PlasticityForm::new(space_n, cfg_n, vec![], 2);

        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let c = form_a.space.mesh().node_coords(i as u32);
            u[i * 2]     =  0.10 * c[0];
            u[i * 2 + 1] = -0.10 * c[1];
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form_a.residual(&u, &rhs, &mut r);
        form_n.residual(&u, &rhs, &mut r);

        let a_a = form_a.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        let a_n = form_n.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();

        assert!(a_a > 1e-8, "Associated MC must yield, got α={a_a:.6e}");
        assert!(a_n > 1e-8, "Non-associated MC must yield, got α={a_n:.6e}");
        // They should produce measurably different plastic multipliers
        assert!((a_a - a_n).abs() > 1e-10,
            "Associated and non-associated MC should give different α: assoc={a_a:.6e}, non={a_n:.6e}");
    }

    // ── Kinematic hardening tests ─────────────────────────────────────

    /// Isotropic-only and kinematic-hardening J2 should produce
    /// different internal back-stress evolution under the same load.
    #[test]
    fn kinematic_back_stress_evolves() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        // Combined isotropic + kinematic (C = 10×H for visible effect)
        let cfg_kin = PlasticConfig::j2_kinematic(2e5, 0.3, 50.0, 1e2, 1e3, 0.0);
        let form = J2PlasticityForm::new(space, cfg_kin, vec![], 2);
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.15 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        // Back stress should have evolved (at least one non-zero component)
        let state = form.state.lock().unwrap();
        let has_back = state.iter().any(|qp| qp[7..13].iter().any(|&a| a.abs() > 1e-12));
        assert!(has_back, "Kinematic hardening: back stress α should be non-zero after plastic step");
    }

    /// Isotropic-only and kinematic-hardening J2 produce different
    /// back-stress fields under the same monotonic load.
    #[test]
    fn kinematic_vs_isotropic_diverge() {
        let mesh_a = SimplexMesh::<2>::unit_square_tri(4);
        let mesh_b = mesh_a.clone();
        let space_iso = VectorH1Space::new(mesh_a, 1, 2);
        let space_kin = VectorH1Space::new(mesh_b, 1, 2);
        let n = space_iso.n_dofs();
        let cfg_iso = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3);
        // Kinematic with AF recall (γ > 0) to break the monotonic degeneracy
        let cfg_kin = PlasticConfig::j2_kinematic(2e5, 0.3, 50.0, 5e2, 8e2, 1e1);
        let form_iso = J2PlasticityForm::new(space_iso, cfg_iso, vec![], 2);
        let form_kin = J2PlasticityForm::new(space_kin, cfg_kin, vec![], 2);
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form_iso.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.12 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form_iso.residual(&u, &rhs, &mut r);
        form_kin.residual(&u, &rhs, &mut r);
        // Check back stress evolution
        let has_back = form_kin.state.lock().unwrap()
            .iter().any(|qp| qp[7..13].iter().any(|&a| a.abs() > 1e-12));
        assert!(has_back, "Kinematic J2 must produce non-zero back stress");
        // Plastic multipliers should differ due to AF recall
        let a_iso = form_iso.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        let a_kin = form_kin.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        assert!(a_iso > 1e-8, "Isotropic J2 must yield");
        assert!((a_iso - a_kin).abs() > 1e-8,
            "Isotropic and kinematic (AF) J2 produce different α: iso={a_iso:.6e}, kin={a_kin:.6e}");
    }

    // ── Viscoplasticity (Perzyna) tests ───────────────────────────────

    /// With dt=0 the viscoplastic form behaves identically to rate‑independent J2.
    #[test]
    fn viscoplastic_dt_zero_equals_rate_independent() {
        let mesh1 = SimplexMesh::<2>::unit_square_tri(4);
        let mesh2 = mesh1.clone();
        let space1 = VectorH1Space::new(mesh1, 1, 2);
        let space2 = VectorH1Space::new(mesh2, 1, 2);
        let n = space1.n_dofs();
        let cfg_vp = PlasticConfig::j2_viscoplastic(2e5, 0.3, 50.0, 1e3, 1e2);
        let cfg_ri = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3);
        let form_vp = J2PlasticityForm::with_dt(space1, cfg_vp, vec![], 2, 0.0);
        let form_ri = J2PlasticityForm::new(space2, cfg_ri, vec![], 2);
        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form_vp.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.10 * x;
        }
        let rhs = vec![0.0; n];
        let mut r1 = vec![0.0; n];
        let mut r2 = vec![0.0; n];
        form_vp.residual(&u, &rhs, &mut r1);
        form_ri.residual(&u, &rhs, &mut r2);
        let a_vp = form_vp.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        let a_ri = form_ri.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        assert!(a_vp > 1e-8, "VP with dt=0 must yield");
        assert!((a_vp - a_ri).abs() < 1e-8,
            "VP dt=0 and rate-independent should match: vp={a_vp:.6e}, ri={a_ri:.6e}");
    }

    /// Finite dt with viscosity suppresses the plastic multiplier (overstress
    /// is partly carried by viscous effects).
    #[test]
    fn viscoplastic_finite_dt_reduces_plasticity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let cfg = PlasticConfig::j2_viscoplastic(2e5, 0.3, 50.0, 1e3, 5e3);
        // Same form with two different dt values
        let form_slow = J2PlasticityForm::with_dt(space, cfg.clone(), vec![], 2, 1.0);
        let n2 = form_slow.n_dofs();
        let mut u = vec![0.0; n2];
        for i in 0..n2 / 2 {
            let x = form_slow.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.10 * x;
        }
        let rhs = vec![0.0; n2];
        let mut r = vec![0.0; n2];
        form_slow.residual(&u, &rhs, &mut r);
        let a_slow = form_slow.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        assert!(a_slow > 0.0, "VP with dt=1 must still yield");
        // With very large dt (quasi-static), viscosity matters less
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let space2 = VectorH1Space::new(mesh2, 1, 2);
        let form_fast = J2PlasticityForm::with_dt(space2, cfg, vec![], 2, 100.0);
        let mut r2 = vec![0.0; form_fast.n_dofs()];
        form_fast.residual(&u, &rhs, &mut r2);
        let a_fast = form_fast.state.lock().unwrap().iter().map(|qp| qp[6]).sum::<f64>();
        // Larger dt → smaller η/dt → less viscous suppression → more plasticity
        assert!(a_fast >= a_slow - 1e-10,
            "Larger dt should not decrease plastic strain: fast(dt=100)={a_fast:.6e} < slow(dt=1)={a_slow:.6e}");
    }

    // ── Damage‑coupled plasticity tests ─────────────────────────────

    /// Lemaitre damage evolves when accumulated plastic strain exceeds
    /// the damage threshold.
    #[test]
    fn damage_plasticity_evolves_d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        // Low yield, low damage threshold → damage develops
        let cfg = PlasticConfig::j2_damage(2e5, 0.3, 30.0, 5e2, 1e3, 0.01);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);

        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.15 * x;  // tensile strain
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        // Damage variable D must have evolved on at least one QP
        let state = form.state.lock().unwrap();
        let has_damage = state.iter().any(|qp| qp[13] > 1e-6);
        assert!(has_damage, "Damage D should be > 0 after plastic strain exceeds threshold");
    }

    /// Without damage coupling (damage_S = 0), D stays zero.
    #[test]
    fn no_damage_when_disabled() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        // Same J2 but damage_S = 0 → damage disabled
        let cfg = PlasticConfig::j2(2e5, 0.3, 30.0, 5e2);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);

        let mut u = vec![0.0; n];
        for i in 0..n / 2 {
            let x = form.space.mesh().node_coords(i as u32)[0];
            u[i * 2] = 0.15 * x;
        }
        let rhs = vec![0.0; n];
        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);

        let state = form.state.lock().unwrap();
        let total_d: f64 = state.iter().map(|qp| qp[13]).sum();
        assert!(total_d < 1e-30, "D should be zero when damage is disabled");
    }
}
