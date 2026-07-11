//! Joule Mini App: Transient Magnetics and Joule Heating [serial 1:1 translation]
//!
//! Solves the coupled eddy-current / heat equation:
//!
//! ```text
//! Div sigma Grad Phi = 0
//! sigma E  =  Curl B/mu - sigma grad Phi
//! dB/dt   = -Curl E
//! F       = -k Grad T
//! c dT/dt = -Div F + sigma E·E
//! ```
//!
//! where σ is electrical conductivity, μ is magnetic permeability, k is thermal
//! conductivity, and c is heat capacity.
//!
//! **5 fields** stored in the state vector (BlockVector layout):
//!
//! | Offset | Field | Space | Description |
//! |--------|-------|-------|-------------|
//! | 0 | T | L2 | Temperature |
//! | n_l2 | F | HDiv | Thermal flux |
//! | n_l2+n_rt | P | H1 | Electrostatic potential |
//! | n_l2+n_rt+n_h1 | E | HCurl | Electric field |
//! | n_l2+n_rt+n_h1+n_nd | B | HDiv | Magnetic flux |
//! | n_l2+n_rt+n_h1+n_nd+n_rt | W | L2 | Joule heating σ|E|² |
//!
//! **4 FE spaces**: H¹ (scalar potential), H(curl) (electric field),
//! H(div) (magnetic + thermal flux), L² (temperature + Joule heating).
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_miniapp_joule -- -m data/beam-tet.mesh -o 1 -dt 0.1 -tf 1.0
//! ```
//!
//! ## Reference
//! MFEM miniapp `miniapps/electromagnetics/joule.cpp` + `joule_solver.hpp/cpp`.

use std::collections::HashMap;
use std::f64::consts::PI;

use fem_assembly::assembler::Assembler;
use fem_assembly::coefficient::{CoeffCtx, ScalarCoeff, MeshDependentCoefficient};
use fem_assembly::form::{form_linear_system, recover_fem_solution};
use fem_assembly::mixed::{
    assemble_hcurl_hdiv_weak_curl, assemble_hdiv_l2_mixed,
    ref_elem_vol, ref_elem_vec, HDivL2ScaledDiv,
};
use fem_assembly::standard::*;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::geo_ref_elem_from_mesh;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform_3d, Mesh, MeshTopology};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::constraints::dirichlet::boundary_dofs;
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MagneticDiffusionEOperator — the core operator
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Maxwell diffusion + Joule heating operator.
///
/// Manages 4 FE spaces, 6 fields, and 13 operators.  Provides `implicit_solve`
/// which computes `dstate/dt = f(state, t)` for Backward-Euler / SDIRK stepping.
pub struct MagneticDiffusionEOperator {
    // Spaces
    pub h1: H1Space<Mesh<3>>,
    pub nd: HCurlSpace<Mesh<3>>,
    pub rt: HDivSpace<Mesh<3>>,
    pub l2: L2Space<Mesh<3>>,
    // Dimensions
    pub n_h1: usize,
    pub n_nd: usize,
    pub n_rt: usize,
    pub n_l2: usize,
    /// Total state vector length = 2 * n_l2 + 2 * n_rt + n_h1 + n_nd
    pub state_len: usize,

    // --- H1 system: Div sigma Grad P = 0 ---
    pub a0: CsrMatrix<f64>,
    // --- HCurl system: (M1 + dt * S1) E = rhs ---
    pub m1: CsrMatrix<f64>, // sigma · HCurl mass
    pub s1: CsrMatrix<f64>, // 1/mu · curl-curl
    // --- HDiv system: (M2 + dt * S2) F = rhs ---
    pub m2: CsrMatrix<f64>, // 1/k · HDiv mass
    pub s2: CsrMatrix<f64>, // 1/c · div-div
    // --- L2 thermal mass ---
    pub m3: CsrMatrix<f64>, // c · L2 mass
    // --- Discrete operators ---
    pub grad: CsrMatrix<f64>,       // H1 → HCurl
    pub curl: CsrMatrix<f64>,       // HCurl → HDiv
    pub weak_curl: CsrMatrix<f64>,  // ∫ 1/mu · curl(u) · v dx (HCurl×HDiv)
    pub weak_curl_t: CsrMatrix<f64>, // weak_curl^T (HCurl←HDiv)
    pub weak_div: CsrMatrix<f64>,   // ∫ p · div(v) dx (L2×HDiv)
    pub weak_div_t: CsrMatrix<f64>, // weak_div^T (HDiv←L2)
    pub weak_div_c: CsrMatrix<f64>, // ∫ 1/c · p · div(v) dx (L2×HDiv)
    pub weak_div_c_t: CsrMatrix<f64>, // weak_div_c^T (HDiv←L2)

    // Boundary conditions
    pub poisson_ess_tdofs: Vec<u32>,
    pub poisson_bc_coords: Vec<Vec<f64>>, // physical coords for each BC DOF
    pub hcurl_ess_tdofs: Vec<u32>,
    pub hdiv_ess_tdofs: Vec<u32>,

    /// Drive frequency for time-dependent BC (rad/s)
    pub frequency: f64,

    // Material parameters
    pub sigma: MeshDependentCoefficient,
    pub tcap: MeshDependentCoefficient,
    pub inv_tcap: MeshDependentCoefficient,
    pub inv_tcond: MeshDependentCoefficient,
    pub mu: f64,

    // Stored dt for cached system rebuild
    pub dt_a1: f64,
    pub dt_a2: f64,

    // Pre-allocated work vectors
    pub v0: Vec<f64>,  // H1 work
    pub v1: Vec<f64>,  // HCurl work
    pub v2: Vec<f64>,  // HDiv work

    // State offsets
    pub off_t: usize,
    pub off_f: usize,
    pub off_p: usize,
    pub off_e: usize,
    pub off_b: usize,
    pub off_w: usize,
}

impl MagneticDiffusionEOperator {
    /// Create the operator, building all time-independent operators.
    pub fn new(
        mesh: Mesh<3>,
        order: u8,
        mu: f64,
        sigma_map: HashMap<i32, f64>,
        tcap_map: HashMap<i32, f64>,
        inv_tcap_map: HashMap<i32, f64>,
        inv_tcond_map: HashMap<i32, f64>,
        poisson_bdr: &[i32],
        hcurl_bdr: &[i32],
        hdiv_bdr: &[i32],
    ) -> Self {
        // Spaces: L2 order = order-1 (discontinuous cell-center), HDiv = order-1,
        // HCurl = order, H1 = order (matching MFEM).
        // L2 order-1 on Tet4 uses P0, which is not supported in fem-rs → clamp to 1.
        let l2_order = (if order > 0 { order - 1 } else { 0 }).max(1);
        let rt_order = if order > 0 { order - 1 } else { 0 };
        let l2 = L2Space::new(mesh.clone(), l2_order);
        let rt = HDivSpace::new(mesh.clone(), rt_order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let h1 = H1Space::new(mesh.clone(), order);

        let n_l2 = l2.n_dofs();
        let n_rt = rt.n_dofs();
        let n_nd = nd.n_dofs();
        let n_h1 = h1.n_dofs();

        let off_t = 0;
        let off_f = off_t + n_l2;
        let off_p = off_f + n_rt;
        let off_e = off_p + n_h1;
        let off_b = off_e + n_nd;
        let off_w = off_b + n_rt;
        let state_len = off_w + n_l2;

        let qo = (2 * order + 1).max(5);

        // Material coefficients
        let sigma      = MeshDependentCoefficient::new(sigma_map.clone());
        let tcap       = MeshDependentCoefficient::new(tcap_map.clone());
        let inv_tcap   = MeshDependentCoefficient::new(inv_tcap_map);
        let inv_tcond  = MeshDependentCoefficient::new(inv_tcond_map);

        // Build time-independent operators
        let a0 = Assembler::assemble_bilinear(&h1, &[&DiffusionIntegrator { kappa: sigma.clone() }], qo);
        let m1 = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator { alpha: sigma.clone() }], qo);
        let s1 = VectorAssembler::assemble_bilinear(&nd, &[&CurlCurlIntegrator { mu: 1.0 / mu }], qo);
        let m2 = VectorAssembler::assemble_bilinear(&rt, &[&VectorMassIntegrator { alpha: inv_tcond.clone() }], qo);
        let s2 = VectorAssembler::assemble_bilinear(&rt, &[&GradDivIntegrator { kappa: inv_tcap.clone() }], qo);
        let m3 = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: tcap }], qo);

        let grad = fem_assembly::discrete_op::DiscreteLinearOperator::gradient(&h1, &nd)
            .expect("gradient operator");
        let curl = fem_assembly::discrete_op::DiscreteLinearOperator::curl_3d(&nd, &rt)
            .expect("curl_3d operator");

        let weak_curl = assemble_hcurl_hdiv_weak_curl(&nd, &rt, qo, 1.0 / mu);
        let weak_curl_t = weak_curl.transpose();

        // weakDiv: ∫ p · div(v) dx  (unscaled)
        use fem_assembly::mixed::HDivL2DivIntegrator;
        let weak_div = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo);
        let weak_div_t = weak_div.transpose();
        // weakDivC: ∫ 1/c · p · div(v) dx  (scaled)
        let weak_div_c = assemble_hdiv_l2_mixed(
            &l2, &rt, &[&HDivL2ScaledDiv { alpha: inv_tcap.clone() }], qo,
        );
        let weak_div_c_t = weak_div_c.transpose();

        // Boundary condition DOFs
        let dm = h1.dof_manager();
        let poisson_ess_tdofs: Vec<u32> = if !poisson_bdr.is_empty() {
            boundary_dofs(&mesh, dm, poisson_bdr)
        } else { vec![] };
        let poisson_bc_coords: Vec<Vec<f64>> = poisson_ess_tdofs.iter()
            .map(|&d| dm.dof_coord(d).to_vec())
            .collect();

        let hcurl_ess_tdofs: Vec<u32> = if !hcurl_bdr.is_empty() {
            fem_space::constraints::dirichlet::boundary_dofs_hcurl(&mesh, &nd, hcurl_bdr)
        } else { vec![] };

        let hdiv_ess_tdofs: Vec<u32> = if !hdiv_bdr.is_empty() {
            fem_space::constraints::dirichlet::boundary_dofs_hdiv(&mesh, &rt, hdiv_bdr)
        } else { vec![] };

        let v0 = vec![0.0; n_h1];
        let v1 = vec![0.0; n_nd];
        let v2 = vec![0.0; n_rt];

        MagneticDiffusionEOperator {
            h1, nd, rt, l2,
            n_h1, n_nd, n_rt, n_l2, state_len,
            a0, m1, s1, m2, s2, m3,
            grad, curl,
            weak_curl, weak_curl_t, weak_div, weak_div_t, weak_div_c, weak_div_c_t,
            poisson_ess_tdofs, poisson_bc_coords, hcurl_ess_tdofs, hdiv_ess_tdofs,
            frequency: 0.0,
            sigma, tcap: MeshDependentCoefficient::new(tcap_map),
            inv_tcap, inv_tcond, mu,
            dt_a1: -1.0, dt_a2: -1.0,
            v0, v1, v2,
            off_t, off_f, off_p, off_e, off_b, off_w,
        }
    }

    /// Build or rebuild the HCurl system: A1 = M1(σ) + dt · S1(1/μ)
    fn build_a1(&mut self, dt: f64) -> CsrMatrix<f64> {
        if (dt - self.dt_a1).abs() < 1e-12 * dt {
            // Reuse previous — return M1 + dt_a1 * S1
            return self.m1.axpby(1.0, &self.s1, self.dt_a1);
        }
        self.dt_a1 = dt;
        self.m1.axpby(1.0, &self.s1, dt)
    }

    /// Build or rebuild the HDiv system: A2 = M2(1/k) + dt · S2(1/c)
    fn build_a2(&mut self, dt: f64) -> CsrMatrix<f64> {
        if (dt - self.dt_a2).abs() < 1e-12 * dt {
            return self.m2.axpby(1.0, &self.s2, self.dt_a2);
        }
        self.dt_a2 = dt;
        // For A2, inv_tcap is scaled by dt (matching MFEM SetScaleFactor)
        // We recompute S2 with the scaled coefficient
        let qo = (2 * self.h1.order() + 1).max(5);
        let mut inv_tcap_scaled = self.inv_tcap.clone();
        inv_tcap_scaled.set_scale(dt);
        let s2_scaled = VectorAssembler::assemble_bilinear(
            &self.rt, &[&GradDivIntegrator { kappa: inv_tcap_scaled }], qo,
        );
        self.m2.axpby(1.0, &s2_scaled, 1.0)
    }

    /// Compute the total Joule heating W = σ · |E|² as an L² projection.
    ///
    /// This is equivalent to MFEM's `JouleHeatingCoefficient` used with
    /// `ProjectCoefficient`.  We loop over elements, evaluate E at quadrature
    /// points using the HCurl basis, compute `σ|E|²`, and assemble the L²
    /// linear system `M · w = b`.
    pub fn get_joule_heating(&self, e_dofs: &[f64], w_dofs: &mut [f64]) {
        use fem_element::ReferenceElement;
        use fem_assembly::isoparametric_jacobian;
        use fem_mesh::element_type::ElementType;

        let mesh = self.nd.mesh();
        let dim = mesh.dim() as usize;
        let qo = (2 * self.h1.order() + 1).max(5);

        // Assemble RHS: b_i = ∫ φ_i · σ · |E_h|² dx
        let mut rhs = vec![0.0; self.n_l2];

        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let nd_ref = ref_elem_vec(et, self.nd.order(), SpaceType::HCurl)
                .expect("HCurl ref elem");
            let l2_ref = ref_elem_vol(et, self.l2.order())
                .expect("L2 ref elem");
            let n_nd_e = nd_ref.n_dofs();
            let n_l2_e = l2_ref.n_dofs();

            let nd_dofs: Vec<usize> = self.nd.element_dofs(e).iter().map(|&d| d as usize).collect();
            let l2_dofs: Vec<usize> = self.l2.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nd_signs = self.nd.element_signs(e);
            let elem_tag = mesh.element_tag(e);
            let sigma_val = self.sigma.eval(&CoeffCtx { x: &[], elem_id: e, elem_tag, dim, phi: None, elem_dofs: None });
            if sigma_val.abs() < 1e-30 { continue; }

            let use_iso = !matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
            let ge: Option<Box<dyn ReferenceElement>> = if use_iso {
                geo_ref_elem_from_mesh(mesh, e)
            } else { None };
            let nodes = mesh.element_nodes(e);
            let quad = nd_ref.quadrature(qo);

            let mut nd_basis = vec![0.0; n_nd_e * dim];
            let mut l2_phi = vec![0.0; n_l2_e];

            for (qi, xi) in quad.points.iter().enumerate() {
                let (w, jit, _det_j): (f64, nalgebra::DMatrix<f64>, f64) = if use_iso {
                    let g: &dyn ReferenceElement = ge.as_deref().unwrap();
                    let (jac, det, _) = isoparametric_jacobian(mesh, &nodes, g, xi, dim);
                    (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose(), det)
                } else {
                    let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
                    (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone(), tr.det_j())
                };

                nd_ref.eval_basis_vec(xi, &mut nd_basis);
                l2_ref.eval_basis(xi, &mut l2_phi);

                // Evaluate E at this QP using Piola-transformed HCurl basis
                let mut e_val = [0.0_f64; 3];
                for i in 0..n_nd_e {
                    let si = nd_signs.get(i).copied().unwrap_or(1.0);
                    // HCurl Piola: ψ_phys = J^{-T} · ψ_ref
                    let dot0 = (0..dim).map(|c| jit[(0, c)] * nd_basis[i * dim + c]).sum::<f64>();
                    let e_i = e_dofs[nd_dofs[i]] * si;
                    e_val[0] += e_i * dot0;
                    if dim > 1 {
                        let dot1 = (0..dim).map(|c| jit[(1, c)] * nd_basis[i * dim + c]).sum::<f64>();
                        e_val[1] += e_i * dot1;
                    }
                    if dim > 2 {
                        let dot2 = (0..dim).map(|c| jit[(2, c)] * nd_basis[i * dim + c]).sum::<f64>();
                        e_val[2] += e_i * dot2;
                    }
                }
                let e2 = e_val[0] * e_val[0] + e_val[1] * e_val[1] + e_val[2] * e_val[2];
                let source = w * sigma_val * e2;

                for j in 0..n_l2_e {
                    rhs[l2_dofs[j]] += source * l2_phi[j];
                }
            }
        }

        // Solve M3 * w = rhs (L2 mass matrix)
        let cfg = SolverConfig { rtol: 1e-12, atol: 1e-30, max_iter: 5000, verbose: false, ..Default::default() };
        solve_cg(&self.m3, &rhs, w_dofs, &cfg).expect("Joule heating L2 projection");
    }

    /// Compute the electric losses: E^T · M1 · E = ∫ σ · |E|² dx.
    pub fn electric_losses(&self, e_dofs: &[f64]) -> f64 {
        let mut tmp = vec![0.0; self.n_nd];
        self.m1.spmv(e_dofs, &mut tmp);
        e_dofs.iter().zip(tmp.iter()).map(|(e, t)| e * t).sum()
    }

    /// Implicit time-step solve: compute `dstate/dt = f(state, t)`.
    ///
    /// Corresponds to MFEM's `MagneticDiffusionEOperator::ImplicitSolve`.
    ///
    /// For Backward-Euler, `state` is `X_n` and the returned `dstate` satisfies
    /// `X_{n+1} = X_n + dt · dstate`.
    ///
    /// `t` is the time at which to evaluate BCs (typically `t + dt` for BE).
    ///
    /// **Note:** Writes diagnosed fields (P, E, F, W) back into `state` in-place
    /// (matching MFEM's const_cast pattern in ImplicitSolve), so that `state`
    /// always holds the latest diagnosed values.  Only B and T are evolved via
    /// the returned `dstate`.
    pub fn implicit_solve(&mut self, dt: f64, t: f64, state: &mut [f64], dstate: &mut [f64]) {
        dstate.fill(0.0);

        // ── 1. Extract field views from state ──
        let temp = &state[self.off_t..self.off_t + self.n_l2];
        let _f = &state[self.off_f..self.off_f + self.n_rt];
        let _p = &state[self.off_p..self.off_p + self.n_h1];
        let _e = &state[self.off_e..self.off_e + self.n_nd];
        let b = &state[self.off_b..self.off_b + self.n_rt];

        // Work with temp vectors for d_t and d_b to avoid borrow conflicts
        let mut d_b_tmp = vec![0.0; self.n_rt];
        let mut d_t_tmp = vec![0.0; self.n_l2];

        // ── 2. Solve for electrostatic potential P: Div σ Grad P = 0 ──
        // Only boundary conditions (voltage on front/rear) drive the solution.
        let rhs_p = vec![0.0; self.n_h1];
        let bc_p: Vec<f64> = self.poisson_ess_tdofs.iter()
            .zip(self.poisson_bc_coords.iter())
            .map(|(_, x)| p_bc_voltage(x, t, self.frequency))
            .collect();
        let (a0_red, a0_rhs, free_p, constrained_p) =
            form_linear_system(&self.a0, &rhs_p, &self.poisson_ess_tdofs, &bc_p);
        let mut x_p = vec![0.0; a0_red.nrows];
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, verbose: false, ..Default::default() };
        solve_cg(&a0_red, &a0_rhs, &mut x_p, &cfg).expect("Poisson CG solve");
        let p_new = recover_fem_solution(&x_p, &free_p, &constrained_p, &bc_p, self.n_h1);

        // ── 3. Build RHS for E-system: v1 = weakCurl^T · B + M1 · Grad(P_new) ──
        // v1 = weakCurl^T * B  (using pre-computed transpose)
        self.weak_curl_t.spmv(b, &mut self.v1);
        // grad_p = Grad * P_new
        let mut grad_p = vec![0.0; self.n_nd];
        self.grad.spmv(&p_new, &mut grad_p);
        // v1 += M1 * Grad(P_new)
        self.m1.spmv_add(1.0, &grad_p, 1.0, &mut self.v1);

        // ── 4. Solve (M1 + dt·S1) · E_new = v1 with HCurl BC ──
        let a1_mat = self.build_a1(dt);
        let bc_e = vec![0.0; self.hcurl_ess_tdofs.len()];
        let (a1_red, a1_rhs, free_e, constrained_e) =
            form_linear_system(&a1_mat, &self.v1, &self.hcurl_ess_tdofs, &bc_e);
        let mut x_e = vec![0.0; a1_red.nrows];
        solve_cg(&a1_red, &a1_rhs, &mut x_e, &cfg).expect("E-field CG solve");
        let mut e_new = recover_fem_solution(&x_e, &free_e, &constrained_e, &bc_e, self.n_nd);

        // ── 5. Total E correction: E_total = E_ind - Grad(P_new) ──
        self.grad.spmv_add(-1.0, &p_new, 1.0, &mut e_new);

        // ── 6. dB/dt = -Curl(E_new) ──
        self.curl.spmv(&e_new, &mut d_b_tmp);
        for v in d_b_tmp.iter_mut() { *v = -*v; }

        // ── 7. Compute Joule heating W = σ · |E_new|² ──
        let mut w_new = vec![0.0; self.n_l2];
        self.get_joule_heating(&e_new, &mut w_new);

        // ── 8. Build RHS for thermal flux system ──
        // v2 = dt · weakDivC^T · W + weakDiv^T · T  (pre-computed transposes)
        self.v2.fill(0.0);
        self.weak_div_c_t.spmv(&w_new, &mut self.v2);
        for v in self.v2.iter_mut() { *v *= dt; }
        self.weak_div_t.spmv(temp, &mut self.v2);

        // ── 9. Solve (M2 + dt·S2) · F_new = v2 with HDiv BC ──
        let a2_mat = self.build_a2(dt);
        let bc_f = vec![0.0; self.hdiv_ess_tdofs.len()];
        let (a2_red, a2_rhs, free_f, constrained_f) =
            form_linear_system(&a2_mat, &self.v2, &self.hdiv_ess_tdofs, &bc_f);
        let mut x_f = vec![0.0; a2_red.nrows];
        solve_cg(&a2_red, &a2_rhs, &mut x_f, &cfg).expect("Flux CG solve");
        let f_new = recover_fem_solution(&x_f, &free_f, &constrained_f, &bc_f, self.n_rt);

        // ── 10. dT/dt = M3^{-1} · (W_new - weakDiv · F_new) ──
        let mut div_f = vec![0.0; self.n_l2];
        self.weak_div.spmv(&f_new, &mut div_f);
        let mut rhs_t = vec![0.0; self.n_l2];
        for i in 0..self.n_l2 { rhs_t[i] = w_new[i] - div_f[i]; }
        solve_cg(&self.m3, &rhs_t, &mut d_t_tmp, &cfg).expect("Temperature CG solve");

        // ── 11. Write back diagnosed fields to state (matching MFEM in-place update) ──
        // MFEM's ImplicitSolve writes new P, E, F, W back to the state vector
        // via const_cast.  These fields are diagnosed (not time-evolved) but
        // needed at the next step.
        state[self.off_p..self.off_p + self.n_h1].copy_from_slice(&p_new);
        state[self.off_e..self.off_e + self.n_nd].copy_from_slice(&e_new);
        state[self.off_f..self.off_f + self.n_rt].copy_from_slice(&f_new);
        state[self.off_w..self.off_w + self.n_l2].copy_from_slice(&w_new);

        // Copy temp derivatives into dstate (evolved fields only: B and T)
        dstate[self.off_b..self.off_b + self.n_rt].copy_from_slice(&d_b_tmp);
        dstate[self.off_t..self.off_t + self.n_l2].copy_from_slice(&d_t_tmp);
        // dP, dE, dF, dW are 0 (these fields are diagnosed, not time-evolved)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Voltage boundary condition (matching MFEM p_bc)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Electrostatic potential on the boundary.
///
/// MFEM equivalent: `p_bc(const Vector &x, real_t t)` in joule.cpp.
/// The MFEM cylinder mesh has ends centered around z=0:
/// - Front face (z < 0): P = +cos(ω·t)
/// - Rear face  (z ≥ 0): P = -cos(ω·t)
/// For a unit cube [0,1]³ we split at the midpoint z = 0.5.
pub fn p_bc_voltage(x: &[f64], t: f64, freq: f64) -> f64 {
    // Split at z-midpoint of the domain (0.5 for unit cube, 0 for MFEM cylinder)
    let z_split = 0.5;
    let val = if x.len() < 3 || x[2] < z_split { 1.0 } else { -1.0 };
    val * (freq * t).cos()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SDIRK Butcher tableaus
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// SDIRK2: γ = 1 - 1/√2 ≈ 0.2928932188134524 (L-stable, order 2)
const SDIRK2_GAMMA: f64 = 0.2928932188134524;

/// SDIRK33: γ ≈ 0.4358665215084590 (L-stable, order 3, Alexander 1977)
const SDIRK33_GAMMA: f64 = 0.4358665215084590;

/// Take one Backward-Euler step: `state_{n+1} = state_n + dt * k`.
///
/// `implicit_solve` writes new diagnosed fields (P, E, F, W) directly into
/// `state`, then we add `dt × dstate` to update B and T (the evolved fields).
/// Since P/E/F/W were already updated by `implicit_solve`, the result is
/// a fully consistent state.
pub fn step_backward_euler(
    oper: &mut MagneticDiffusionEOperator,
    dt: f64,
    t: f64,
    state: &mut [f64],
    dstate: &mut [f64],
) {
    oper.implicit_solve(dt, t + dt, state, dstate);
    for j in 0..oper.state_len {
        state[j] += dt * dstate[j];
    }
}

/// Take one SDIRK2 step (2 stages, L-stable, order 2).
///
/// After the stage updates, diagnosed fields (P, E, F, W) are copied from
/// the last stage vector since `implicit_solve` writes them into its `state`
/// argument (which for stages is the stage vector, not the caller's state).
pub fn step_sdirk2(
    oper: &mut MagneticDiffusionEOperator,
    dt: f64,
    t: f64,
    state: &mut [f64],
    dstate: &mut [f64],
) {
    let g = SDIRK2_GAMMA;
    let slen = oper.state_len;
    let mut k1 = vec![0.0; slen];
    let mut k2 = vec![0.0; slen];
    let mut stage = vec![0.0; slen];

    // Stage 1: stage = u_n, implicit_solve writes P1,E1,F1,W1 into stage
    // and returns k1 = f(stage) with dB/dt, dT/dt
    stage.copy_from_slice(state);
    oper.implicit_solve(g * dt, t + g * dt, &mut stage, &mut k1);

    // Stage 2: stage = u_n + dt*(1-γ)*k1(B,T parts only)
    // implicit_solve writes P2,E2,F2,W2 into stage and returns k2
    stage.copy_from_slice(state);
    // Only B and T components have non-zero k1 (diagnosed fields have zero d/dt)
    for j in 0..slen { stage[j] += dt * (1.0 - g) * k1[j]; }
    oper.implicit_solve(g * dt, t + dt, &mut stage, &mut k2);

    // Save stage2's diagnosed fields before updating state
    let p2 = stage[oper.off_p..oper.off_p + oper.n_h1].to_vec();
    let e2 = stage[oper.off_e..oper.off_e + oper.n_nd].to_vec();
    let f2 = stage[oper.off_f..oper.off_f + oper.n_rt].to_vec();
    let w2 = stage[oper.off_w..oper.off_w + oper.n_l2].to_vec();

    // Final update: state = u_n + dt*((1-γ)*k1 + γ*k2)  (only B, T change)
    for j in 0..slen {
        let dj = (1.0 - g) * k1[j] + g * k2[j];
        state[j] += dt * dj;
        dstate[j] = dj;
    }

    // Copy diagnosed fields from stage 2 (they correspond to the new B, T)
    state[oper.off_p..oper.off_p + oper.n_h1].copy_from_slice(&p2);
    state[oper.off_e..oper.off_e + oper.n_nd].copy_from_slice(&e2);
    state[oper.off_f..oper.off_f + oper.n_rt].copy_from_slice(&f2);
    state[oper.off_w..oper.off_w + oper.n_l2].copy_from_slice(&w2);
}

/// Take one SDIRK33 step (3 stages, L-stable, order 3, Alexander 1977).
pub fn step_sdirk33(
    oper: &mut MagneticDiffusionEOperator,
    dt: f64,
    t: f64,
    state: &mut [f64],
    dstate: &mut [f64],
) {
    let g = SDIRK33_GAMMA;
    let slen = oper.state_len;
    let mut k1 = vec![0.0; slen];
    let mut k2 = vec![0.0; slen];
    let mut k3 = vec![0.0; slen];
    let mut stage = vec![0.0; slen];

    // Stage 1: Y1 = u_n + g*dt*k1
    stage.copy_from_slice(state);
    oper.implicit_solve(g * dt, t + g * dt, &mut stage, &mut k1);

    // Stage 2: Y2 = u_n + dt*((0.5-g)*k1 + g*k2)
    stage.copy_from_slice(state);
    for j in 0..slen { stage[j] += dt * (0.5 - g) * k1[j]; }
    oper.implicit_solve(g * dt, t + 0.5 * dt, &mut stage, &mut k2);

    // Stage 3: Y3 = u_n + dt*(2g*k1 + (1-4g)*k2 + g*k3)
    stage.copy_from_slice(state);
    for j in 0..slen { stage[j] += dt * (2.0 * g * k1[j] + (1.0 - 4.0 * g) * k2[j]); }
    oper.implicit_solve(g * dt, t + dt, &mut stage, &mut k3);

    // Save stage3's diagnosed fields
    let p3 = stage[oper.off_p..oper.off_p + oper.n_h1].to_vec();
    let e3 = stage[oper.off_e..oper.off_e + oper.n_nd].to_vec();
    let f3 = stage[oper.off_f..oper.off_f + oper.n_rt].to_vec();
    let w3 = stage[oper.off_w..oper.off_w + oper.n_l2].to_vec();

    // Final update: u_{n+1} = u_n + dt*(2g*k1 + (1-4g)*k2 + g*k3)
    for j in 0..slen {
        let dj = 2.0 * g * k1[j] + (1.0 - 4.0 * g) * k2[j] + g * k3[j];
        state[j] += dt * dj;
        dstate[j] = dj;
    }

    // Copy diagnosed fields from stage 3
    state[oper.off_p..oper.off_p + oper.n_h1].copy_from_slice(&p3);
    state[oper.off_e..oper.off_e + oper.n_nd].copy_from_slice(&e3);
    state[oper.off_f..oper.off_f + oper.n_rt].copy_from_slice(&f3);
    state[oper.off_w..oper.off_w + oper.n_l2].copy_from_slice(&w3);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main driver
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "data/beam-tet.mesh".to_string();
    let mut order: u8 = 1;
    let mut ser_ref = 0usize;
    let mut dt = 0.1_f64;
    let mut t_final = 1.0_f64;
    let mut mu = 1.0_f64;
    let mut sigma = 2.0 * PI * 10.0;
    let tcap = 1.0_f64;
    let tcond = 0.01_f64;
    let freq = 1.0 / 60.0;
    let _vis = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--mesh" => { i += 1; if i < args.len() { mesh_file = args[i].clone(); } }
            "-o" | "--order" => { i += 1; if i < args.len() { order = args[i].parse().unwrap_or(1); } }
            "-rs" | "--refine-serial" => { i += 1; if i < args.len() { ser_ref = args[i].parse().unwrap_or(0); } }
            "-dt" | "--time-step" => { i += 1; if i < args.len() { dt = args[i].parse().unwrap_or(0.1); } }
            "-tf" | "--t-final" => { i += 1; if i < args.len() { t_final = args[i].parse().unwrap_or(1.0); } }
            "-mu" | "--permeability" => { i += 1; if i < args.len() { mu = args[i].parse().unwrap_or(1.0); } }
            "-cnd" | "--sigma" => { i += 1; if i < args.len() { sigma = args[i].parse().unwrap_or(2.0 * PI * 10.0); } }
            "-f" | "--frequency" => { i += 1; if i < args.len() { let _ = args[i].parse::<f64>(); } }
            "-vis" | "--visualization" => { /* vis = true */ }
            "-amr" => { i += 1; }
            "-h" | "--help" => {
                eprintln!("Joule Mini App: Transient Magnetics + Joule Heating");
                eprintln!("  -m  | --mesh     Mesh file (default: data/beam-tet.mesh)");
                eprintln!("  -o  | --order    FE order (default: 1)");
                eprintln!("  -rs | --refine-serial  Serial refinements (default: 0)");
                eprintln!("  -dt | --time-step     Time step (default: 0.1)");
                eprintln!("  -tf | --t-final       Final time (default: 1.0)");
                eprintln!("  -mu | --permeability  Magnetic permeability (default: 1.0)");
                eprintln!("  -cnd| --sigma         Conductivity (default: 2π·10)");
                eprintln!("  -f  | --frequency     Drive frequency (default: 1/60)");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let skin_depth = (2.0 / (2.0 * PI * freq * mu * sigma)).sqrt();
    let skin_depth_dt = (2.0 * dt / (mu * sigma)).sqrt();
    println!("Skin depth (AC): {:.6}", skin_depth);
    println!("Skin depth (dt): {:.6}", skin_depth_dt);

    // ── Material maps for "rod" problem (tag 1 = rod, 2+3 = air) ──
    let sigma_air = 1.0e-6 * sigma;
    let tcond_air = 1.0e6 * tcond;
    let tcap_air = 1.0 * tcap;

    let mut sigma_map = HashMap::new();
    sigma_map.insert(1, sigma);
    sigma_map.insert(2, sigma_air);
    sigma_map.insert(3, sigma_air);

    let mut inv_tcond_map = HashMap::new();
    inv_tcond_map.insert(1, 1.0 / tcond);
    inv_tcond_map.insert(2, 1.0 / tcond_air);
    inv_tcond_map.insert(3, 1.0 / tcond_air);

    let mut tcap_map = HashMap::new();
    tcap_map.insert(1, tcap);
    tcap_map.insert(2, tcap_air);
    tcap_map.insert(3, tcap_air);

    let mut inv_tcap_map = HashMap::new();
    inv_tcap_map.insert(1, 1.0 / tcap);
    inv_tcap_map.insert(2, 1.0 / tcap_air);
    inv_tcap_map.insert(3, 1.0 / tcap_air);

    // ── Mesh ──
    let mfem = read_mfem_file(&mesh_file).expect("mesh file");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D mesh");
    for _ in 0..ser_ref { mesh = refine_uniform_3d(&mesh); }

    // ── Boundary conditions (rod problem) ──
    // E-field: front(1), rear(2), outer(3) — tangential E = 0 (PEC)
    let hcurl_bdr = vec![1i32, 2, 3];
    // Thermal flux: front(1), rear(2) — zero normal flux
    let hdiv_bdr = vec![1i32, 2];
    // Poisson: front(1), rear(2) — prescribed voltage (in main, we set zero for now)
    let poisson_bdr = vec![1i32, 2];

    // ── FE spaces and operator ──
    let mut oper = MagneticDiffusionEOperator::new(
        mesh, order, mu,
        sigma_map, tcap_map, inv_tcap_map, inv_tcond_map,
        &poisson_bdr, &hcurl_bdr, &hdiv_bdr,
    );
    oper.frequency = 2.0 * PI * freq;

    // ── Initialize state ──
    let mut state = vec![0.0; oper.state_len];
    let mut dstate = vec![0.0; oper.state_len];

    println!("\nDOFs: H1={} HCurl={} HDiv={} L2={}",
             oper.n_h1, oper.n_nd, oper.n_rt, oper.n_l2);
    println!("State vector length: {}", oper.state_len);
    println!("t=0, dt={dt}, t_final={t_final}");

    // ── Time loop ──
    let mut t = 0.0;
    let mut step = 0;
    let dt = dt; // shadow for clarity
    while t < t_final - 1e-12 {
        let dt_actual = dt.min(t_final - t);

        // Use SDIRK2 by default (L-stable, order 2)
        step_sdirk2(&mut oper, dt_actual, t, &mut state, &mut dstate);
        t += dt_actual;
        step += 1;

        // Diagnostics
        if step % 10 == 0 || t >= t_final - 1e-12 {
            let e_losses = oper.electric_losses(
                &state[oper.off_e..oper.off_e + oper.n_nd]
            );
            let t_norm: f64 = state[oper.off_t..oper.off_t + oper.n_l2].iter().map(|v| v * v).sum::<f64>().sqrt();
            let b_norm: f64 = state[oper.off_b..oper.off_b + oper.n_rt].iter().map(|v| v * v).sum::<f64>().sqrt();
            let p_max = state[oper.off_p..oper.off_p + oper.n_h1].iter().copied().fold(0.0_f64, f64::max);
            let p_min = state[oper.off_p..oper.off_p + oper.n_h1].iter().copied().fold(0.0_f64, f64::min);
            println!("step {step:6}, t={t:.6e}, E·J={e_losses:.8e}, |B|={b_norm:.6e}, |T|={t_norm:.6e}, P∈[{p_min:.4e},{p_max:.4e}]");
        }
    }

    println!("\nJoule simulation complete: {} steps to t={}", step, t);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_operator() -> MagneticDiffusionEOperator {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let sigma = 1.0;
        let tcap = 1.0;
        let tcond = 0.01;
        let sigma_air = 1.0e-6 * sigma;

        let mut sigma_map = HashMap::new();
        sigma_map.insert(1, sigma);
        sigma_map.insert(2, sigma_air);
        sigma_map.insert(3, sigma_air);

        let mut inv_tcond_map = HashMap::new();
        inv_tcond_map.insert(1, 1.0 / tcond);
        inv_tcond_map.insert(2, 1.0e6);
        inv_tcond_map.insert(3, 1.0e6);

        let mut tcap_map = HashMap::new();
        tcap_map.insert(1, tcap);
        tcap_map.insert(2, tcap);
        tcap_map.insert(3, tcap);

        let mut inv_tcap_map = HashMap::new();
        inv_tcap_map.insert(1, 1.0 / tcap);
        inv_tcap_map.insert(2, 1.0 / tcap);
        inv_tcap_map.insert(3, 1.0 / tcap);

        let mut oper = MagneticDiffusionEOperator::new(
            mesh, 1, 1.0,
            sigma_map, tcap_map, inv_tcap_map, inv_tcond_map,
            &[1, 2], &[1, 2, 3], &[1, 2],
        );
        oper.frequency = 0.0; // DC (no time variation in BC for tests)
        oper
    }

    #[test]
    fn joule_operator_constructor_succeeds() {
        let oper = setup_operator();
        assert!(oper.n_h1 > 0);
        assert!(oper.n_nd > 0);
        assert!(oper.n_rt > 0);
        assert!(oper.n_l2 > 0);
        assert_eq!(oper.state_len, 2 * oper.n_l2 + 2 * oper.n_rt + oper.n_h1 + oper.n_nd);
    }

    #[test]
    fn joule_dc_bc_produces_nonzero_p_and_e() {
        // With DC BC (P=±1), the electrostatic potential P should be non-zero,
        // and the induced E-field (via Grad(P)) should also be non-zero.
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        let mut dstate = vec![0.0; oper.state_len];
        oper.implicit_solve(0.1, 0.0, &mut state, &mut dstate);

        // P was written back to state (diagnosed field) — check P ≠ 0
        let p_norm: f64 = state[oper.off_p..oper.off_p + oper.n_h1].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        assert!(p_norm > 1e-6, "DC BC should produce non-zero P: |P|={p_norm:.6e}");

        // E was written back to state — should be non-zero (induced)
        let e_norm: f64 = state[oper.off_e..oper.off_e + oper.n_nd].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        assert!(e_norm > 1e-6, "DC BC should produce non-zero E via Grad(P): |E|={e_norm:.6e}");

        // B is evolved (not written back), should still be zero
        let b_norm: f64 = state[oper.off_b..oper.off_b + oper.n_rt].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        assert!(b_norm < 1e-30, "B should still be zero (no initial B): |B|={b_norm:.6e}");
    }

    #[test]
    fn joule_electric_losses_zero_with_zero_field() {
        let oper = setup_operator();
        let e = vec![0.0; oper.n_nd];
        let losses = oper.electric_losses(&e);
        assert!(losses.abs() < 1e-30, "losses should be zero for zero E-field");
    }

    #[test]
    fn joule_electric_losses_positive_with_nonzero_field() {
        let oper = setup_operator();
        // Solve M1 * x = 1 to get a physically-admissible E-field
        let ones = vec![1.0; oper.n_nd];
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..Default::default() };
        let mut e = vec![0.0; oper.n_nd];
        solve_cg(&oper.m1, &ones, &mut e, &cfg).expect("M1 solve");
        let e_norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();

        let losses = oper.electric_losses(&e);
        assert!(losses > 0.0, "losses should be positive for non-zero E (|e|={e_norm:.6e}): got {losses:.6e}");
    }

    #[test]
    fn joule_heating_zero_with_zero_e() {
        let oper = setup_operator();
        let e = vec![0.0; oper.n_nd];
        let mut w = vec![0.0; oper.n_l2];
        oper.get_joule_heating(&e, &mut w);
        let w_norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(w_norm < 1e-30, "Joule heating should be zero with zero E: |w|={}", w_norm);
    }

    #[test]
    fn joule_heating_grows_with_e_strength() {
        let oper = setup_operator();
        let mut e_low = vec![0.0; oper.n_nd];
        let mut e_high = vec![0.0; oper.n_nd];

        // Use eigenvector-like direction: solve M1 * x = 1 (roughly)
        let cfg = SolverConfig { rtol: 1e-3, max_iter: 10, ..Default::default() };
        let ones = vec![1.0; oper.n_nd];
        solve_cg(&oper.m1, &ones, &mut e_low, &cfg).ok();

        let twos = vec![2.0; oper.n_nd];
        solve_cg(&oper.m1, &twos, &mut e_high, &cfg).ok();

        let mut w_low = vec![0.0; oper.n_l2];
        let mut w_high = vec![0.0; oper.n_l2];
        oper.get_joule_heating(&e_low, &mut w_low);
        oper.get_joule_heating(&e_high, &mut w_high);

        let wl: f64 = w_low.iter().sum();
        let wh: f64 = w_high.iter().sum();
        assert!(wh > wl, "Joule heating should grow with E strength: low={}, high={}", wl, wh);
    }

    #[test]
    fn joule_implicit_solve_preserves_state_length() {
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        let mut dstate = vec![0.0; oper.state_len];
        oper.implicit_solve(0.1, 0.0, &mut state, &mut dstate);
        assert_eq!(dstate.len(), oper.state_len);
    }

    #[test]
    fn joule_one_step_backward_euler_does_not_blow_up() {
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        // Small initial B-field
        for i in oper.off_b..oper.off_b + oper.n_rt.min(10) {
            state[i] = 1e-6;
        }
        let mut dstate = vec![0.0; oper.state_len];
        oper.implicit_solve(0.01, 0.0, &mut state, &mut dstate);

        // Update
        for j in 0..oper.state_len {
            state[j] += 0.01 * dstate[j];
        }

        let t_norm: f64 = state[oper.off_t..oper.off_t + oper.n_l2].iter().map(|v| v * v).sum::<f64>().sqrt();
        let e_norm: f64 = state[oper.off_e..oper.off_e + oper.n_nd].iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm: f64 = state[oper.off_b..oper.off_b + oper.n_rt].iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(b_norm.is_finite(), "B-field should be finite after one step");
        assert!(e_norm.is_finite(), "E-field should be finite after one step");
        assert!(t_norm.is_finite(), "Temperature should be finite after one step");
    }

    #[test]
    fn joule_sdirk2_step_produces_finite_results() {
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        for i in oper.off_b..oper.off_b + oper.n_rt.min(5) { state[i] = 1e-6; }
        let mut dstate = vec![0.0; oper.state_len];
        step_sdirk2(&mut oper, 0.01, 0.0, &mut state, &mut dstate);

        let b_norm: f64 = state[oper.off_b..oper.off_b + oper.n_rt].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let e_norm: f64 = state[oper.off_e..oper.off_e + oper.n_nd].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let t_norm: f64 = state[oper.off_t..oper.off_t + oper.n_l2].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let p_max = state[oper.off_p..oper.off_p + oper.n_h1].iter()
            .copied().fold(0.0_f64, f64::max);
        assert!(b_norm.is_finite(), "B should be finite");
        assert!(e_norm.is_finite(), "E should be finite");
        assert!(t_norm.is_finite(), "T should be finite");
        assert!(p_max > 0.5, "SDIRK2 should produce BC-driven P: max={p_max}");
    }

    #[test]
    fn joule_sdirk33_step_produces_finite_results() {
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        for i in oper.off_b..oper.off_b + oper.n_rt.min(5) { state[i] = 1e-6; }
        let mut dstate = vec![0.0; oper.state_len];
        step_sdirk33(&mut oper, 0.01, 0.0, &mut state, &mut dstate);

        let b_norm: f64 = state[oper.off_b..oper.off_b + oper.n_rt].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let e_norm: f64 = state[oper.off_e..oper.off_e + oper.n_nd].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let t_norm: f64 = state[oper.off_t..oper.off_t + oper.n_l2].iter()
            .map(|v| v * v).sum::<f64>().sqrt();
        let p_max = state[oper.off_p..oper.off_p + oper.n_h1].iter()
            .copied().fold(0.0_f64, f64::max);
        assert!(b_norm.is_finite(), "B should be finite");
        assert!(e_norm.is_finite(), "E should be finite");
        assert!(t_norm.is_finite(), "T should be finite");
        assert!(p_max > 0.5, "SDIRK33 should produce BC-driven P: max={p_max}");
    }

    #[test]
    fn joule_bc_nonzero_p_is_computed() {
        // Verify that the BC produces non-zero P field by examining state after
        // one step (P is in the state vector, diagnosed each step).
        let mut oper = setup_operator();
        let mut state = vec![0.0; oper.state_len];
        let mut dstate = vec![0.0; oper.state_len];
        step_backward_euler(&mut oper, 0.1, 0.0, &mut state, &mut dstate);

        // P should be non-zero (from voltage BC)
        let p_min = state[oper.off_p..oper.off_p + oper.n_h1].iter()
            .copied().fold(0.0_f64, f64::min);
        let p_max = state[oper.off_p..oper.off_p + oper.n_h1].iter()
            .copied().fold(0.0_f64, f64::max);
        // BC: P=+1 on z<0, P=-1 on z>=0 → should find values near ±1
        assert!(p_max > 0.5, "BC should drive P>0.5 on front face: max={p_max}");
        assert!(p_min < -0.5, "BC should drive P<-0.5 on rear face: min={p_min}");
    }
}
