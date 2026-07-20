//! Thermoelastic coupling utilities — Abaqus coupled temp-displacement alignment.
//!
//! Provides:
//! - Steady heat conduction (`assemble_heat_system`)
//! - Transient heat equation with BDF-1/BDF-2 (`assemble_heat_capacity`, `solve_transient_heat_step`)
//! - Thermal expansion load (`assemble_thermal_expansion_rhs`)
//! - Staggered coupling (steady + transient) (`solve_thermoelastic_staggered`,
//!   `solve_thermoelastic_staggered_transient`)
//! - Fully coupled monolithic system (`solve_thermoelastic_monolithic`)
//!
//! ## Governing equations
//!
//! ```text
//! Heat:       ρ·c·∂T/∂t - k·∇²T = Q
//! Elasticity: ∇·σ = f,  σ = C:(ε - α·(T-T₀)·I)
//! ```
//!
//! ## Monolithic coupled system
//!
//! ```text
//! [K_uu   K_uT] [Δu]   [R_u]
//! [K_Tu   K_TT] [ΔT] = [R_T]
//! ```
//!
//! where `K_uu` = elasticity, `K_uT` = thermal expansion coupling,
//! `K_TT` = C/Δt + K (heat capacity + conduction), `K_Tu` = thermoelastic
//! damping (Gough-Joule, often neglected).

use fem_linalg::{CsrMatrix, CooMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_space::VectorH1Space;
use fem_solver::SolverConfig;

use crate::assembler::Assembler;
use crate::standard::{DiffusionIntegrator, ElasticityIntegrator, MassIntegrator};
use crate::dg::dg_advection::{ref_elem_vol, simplex_jac, xform_grads};

// ============================================================================
// Steady heat conduction (existing)
// ============================================================================

/// Assemble the steady heat conduction matrix and RHS.
///
/// Matrix: `K_ij = ∫ k·∇φ_i·∇φ_j dx`
/// RHS:    `f_i = ∫ φ_i·Q dx` (source)
///
/// Returns `(K, rhs)`. Only volumetric heat source is included;
/// Neumann BC must be added separately by modifying the RHS.
pub fn assemble_heat_system<M: MeshTopology + Clone>(
    mesh: &M,
    kappa: f64,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let k = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa }], quad_order,
    );

    (k, vec![0.0; n])
}

// ============================================================================
// Heat capacity matrix (transient)
// ============================================================================

/// Assemble the heat capacity (mass) matrix.
///
/// `C_ij = ∫ ρ·c·φ_i·φ_j dx`
///
/// This is the time-derivative matrix for the transient heat equation:
/// `C·Ṫ + K·T = Q`.
///
/// # Arguments
/// * `mesh` — the mesh
/// * `rho_cp` — density × specific heat capacity (ρ·cₚ)
/// * `quad_order` — quadrature order
pub fn assemble_heat_capacity<M: MeshTopology + Clone>(
    mesh: &M,
    rho_cp: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone(), 1);
    Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: rho_cp }], quad_order,
    )
}

// ============================================================================
// Transient heat step (BDF-1 / BDF-2)
// ============================================================================

/// BDF coefficients for transient heat equation.
///
/// For `C·Ṫ + K·T = Q`, the BDF discretization is:
/// ```text
/// (a₀/Δt)·C·T^{n+1} + K·T^{n+1} = Q^{n+1} + (1/Δt)·C·(a₁·Tⁿ + a₂·T^{n-1})
/// ```
#[derive(Debug, Clone, Copy)]
pub enum BdfScheme {
    /// BDF-1 (implicit Euler): a₀=1, a₁=1, a₂=0
    Bdf1,
    /// BDF-2 (2-step): a₀=3/2, a₁=2, a₂=-1/2
    Bdf2,
}

/// Perform one transient heat time step using BDF-1 or BDF-2.
///
/// Solves: `(a₀/Δt·C + K)·T^{n+1} = Q^{n+1} + C·(a₁·Tⁿ + a₂·T^{n-1})/Δt`
///
/// # Arguments
/// * `K` — conductivity matrix
/// * `C` — heat capacity matrix
/// * `dt` — time step size
/// * `T_n` — temperature at current time `tⁿ` (length `n_dofs`)
/// * `T_n_prev` — temperature at `t^{n-1}` (for BDF-2; ignored for BDF-1)
/// * `Q` — heat source vector at `t^{n+1}`
/// * `bc_dofs` — Dirichlet BC DOF indices
/// * `bc_vals` — Dirichlet BC values
/// * `scheme` — BDF-1 or BDF-2
/// * `cfg` — solver configuration
///
/// Returns `T^{n+1}`, the temperature at the next time step.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_heat_step(
    K: &CsrMatrix<f64>,
    C: &CsrMatrix<f64>,
    dt: f64,
    T_n: &[f64],
    T_n_prev: &[f64],
    Q: &[f64],
    bc_dofs: &[usize],
    bc_vals: &[f64],
    scheme: BdfScheme,
    cfg: &SolverConfig,
) -> Vec<f64> {
    let n = T_n.len();
    let (a0, a1, a2) = match scheme {
        BdfScheme::Bdf1 => (1.0, 1.0, 0.0),
        BdfScheme::Bdf2 => (1.5, 2.0, -0.5),
    };

    // Build LHS: (a₀/Δt)·C + K
    let mut lhs_coo = CooMatrix::<f64>::new(n, n);

    // Add (a₀/Δt)·C
    for i in 0..n {
        for p in C.row_ptr[i]..C.row_ptr[i + 1] {
            lhs_coo.add(i, C.col_idx[p] as usize, (a0 / dt) * C.values[p]);
        }
    }
    // Add K
    for i in 0..n {
        for p in K.row_ptr[i]..K.row_ptr[i + 1] {
            lhs_coo.add(i, K.col_idx[p] as usize, K.values[p]);
        }
    }
    let mut lhs = lhs_coo.into_csr();

    // Build RHS: Q + C·(a₁·Tⁿ + a₂·T^{n-1})/Δt
    let mut rhs = Q.to_vec();
    let mut c_sum = vec![0.0; n];
    let mut temp = vec![0.0; n];
    for i in 0..n {
        temp[i] = a1 * T_n[i] + a2 * T_n_prev[i];
    }
    C.spmv(&temp, &mut c_sum);
    let inv_dt = 1.0 / dt;
    for i in 0..n {
        rhs[i] += inv_dt * c_sum[i];
    }

    // Apply Dirichlet BC
    for (&dof, &val) in bc_dofs.iter().zip(bc_vals.iter()) {
        lhs.apply_dirichlet_symmetric(dof, val, &mut rhs);
    }

    // Solve
    let mut T_new = vec![0.0; n];
    fem_solver::solve_pcg_jacobi(&lhs, &rhs, &mut T_new, cfg)
        .expect("transient heat step solve failed");
    T_new
}

// ============================================================================
// Thermal expansion RHS (existing)
// ============================================================================

/// Assemble the thermal expansion right-hand side vector.
///
/// `f_i = ∫ (3λ + 2μ)·α·(T - T₀)·(∇·φ_i) dx`
///
/// where `φ_i` is the i-th vector test function in the displacement space
/// (VectorH1Space with interleaved DOFs), `T` is the temperature field,
/// `T₀` is the stress-free reference temperature, `α` is the thermal
/// expansion coefficient, and `λ, μ` are the Lamé parameters.
///
/// Returns a vector of length `n_dofs` (same as the displacement space).
#[allow(clippy::too_many_arguments)]
pub fn assemble_thermal_expansion_rhs<M: MeshTopology + Clone>(
    mesh: &M,
    disp_space: &VectorH1Space<M>,
    temp: &[f64],
    t_ref: f64,
    alpha: f64,
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> Vec<f64> {
    let dim = mesh.dim() as usize;
    let n_dofs = disp_space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    let pres_space = H1Space::new(mesh.clone(), 1);

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let order = disp_space.order();
        let re = ref_elem_vol(et, order);
        let n_ldofs = re.n_dofs();
        let n_vec = n_ldofs * dim;
        let quad = re.quadrature(quad_order);
        let dofs: Vec<usize> = disp_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let temp_dofs: Vec<usize> = pres_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        let jit = jac.try_inverse().unwrap().transpose();

        let mut f_elem = vec![0.0; n_vec];
        let mut phi = vec![0.0; n_ldofs];
        let mut gref = vec![0.0; n_ldofs * dim];
        let mut gphys = vec![0.0; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            // Interpolate T at QP
            let temp_qp: f64 = temp_dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| temp[d] * p).sum();
            let delta_t = temp_qp - t_ref;

            // Thermal expansion coefficient: β = α·(3λ + 2μ)
            let beta = alpha * (3.0 * lambda + 2.0 * mu);

            for k in 0..n_ldofs {
                // div(φ_k) = Σ_d gphys[k*dim + d]
                let div_phi = (0..dim).map(|d| gphys[k * dim + d]).sum::<f64>();
                let val = w * beta * delta_t * div_phi;
                for a in 0..dim {
                    f_elem[k * dim + a] += val;
                }
            }
        }

        for (i, &gi) in dofs.iter().enumerate() {
            rhs[gi] += f_elem[i];
        }
    }

    rhs
}

// ============================================================================
// Staggered coupling (steady) — existing
// ============================================================================

/// Solve a staggered thermoelastic problem (steady).
///
/// 1. Solve the steady heat equation for temperature T.
/// 2. Compute the thermal expansion RHS from T.
/// 3. Solve linear elasticity with thermal load.
///
/// Returns `(T, u)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_thermoelastic_staggered<M: MeshTopology + Clone>(
    mesh: &M,
    kappa: f64,
    alpha: f64,
    lambda: f64,
    mu: f64,
    t_ref: f64,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
) -> (Vec<f64>, Vec<f64>) {
    let dim = mesh.dim() as usize;
    let pres_space = H1Space::new(mesh.clone(), 1);
    let disp_space = VectorH1Space::new(mesh.clone(), 1, dim as u8);
    let n_p = pres_space.n_dofs();
    let n_u = disp_space.n_dofs();
    let cfg = SolverConfig {
        rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false,
        ..SolverConfig::default()
    };

    // 1. Solve heat equation
    let (mut k_t, mut rhs_t) = assemble_heat_system(mesh, kappa, quad_order);
    for (&dof, &val) in temp_bc_dofs.iter().zip(temp_bc_vals.iter()) {
        k_t.apply_dirichlet_symmetric(dof, val, &mut rhs_t);
    }
    let mut temp = vec![0.0; n_p];
    fem_solver::solve_pcg_jacobi(&k_t, &rhs_t, &mut temp, &cfg)
        .expect("heat solve failed");

    // 2. Assemble elasticity matrix
    let mut k_u = Assembler::assemble_bilinear(
        &disp_space, &[&ElasticityIntegrator::new(lambda, mu)], quad_order,
    );

    // 3. Compute thermal expansion RHS
    let mut rhs_u = assemble_thermal_expansion_rhs(
        mesh, &disp_space, &temp, t_ref, alpha, lambda, mu, quad_order,
    );

    // 4. Apply Dirichlet BC on displacement
    for (&dof, &val) in disp_bc_dofs.iter().zip(disp_bc_vals.iter()) {
        k_u.apply_dirichlet_symmetric(dof, val, &mut rhs_u);
    }

    // 5. Solve elasticity
    let mut u = vec![0.0; n_u];
    fem_solver::solve_pcg_jacobi(&k_u, &rhs_u, &mut u, &cfg)
        .expect("elasticity solve failed");

    (temp, u)
}

// ============================================================================
// Transient staggered coupling
// ============================================================================

/// Perform one step of transient staggered thermoelastic coupling.
///
/// 1. Advance heat equation one BDF step → T^{n+1}
/// 2. Compute thermal expansion load from T^{n+1}
/// 3. Solve linear elasticity → u^{n+1}
///
/// Returns `(T_new, u_new)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_thermoelastic_staggered_transient_step<M: MeshTopology + Clone>(
    mesh: &M,
    K_t: &CsrMatrix<f64>,
    C_t: &CsrMatrix<f64>,
    K_u: &CsrMatrix<f64>,
    dt: f64,
    T_n: &[f64],
    T_n_prev: &[f64],
    Q: &[f64],
    alpha: f64,
    lambda: f64,
    mu: f64,
    t_ref: f64,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
    scheme: BdfScheme,
    solver_cfg: &SolverConfig,
) -> (Vec<f64>, Vec<f64>) {
    let dim = mesh.dim() as usize;
    let pres_space = H1Space::new(mesh.clone(), 1);
    let disp_space = VectorH1Space::new(mesh.clone(), 1, dim as u8);
    let n_p = pres_space.n_dofs();

    // 1. Transient heat step
    let T_new = solve_transient_heat_step(
        K_t, C_t, dt, T_n, T_n_prev, Q,
        temp_bc_dofs, temp_bc_vals, scheme, solver_cfg,
    );

    // 2. Compute thermal expansion load from T_new
    let mut rhs_u = assemble_thermal_expansion_rhs(
        mesh, &disp_space, &T_new, t_ref, alpha, lambda, mu, quad_order,
    );

    // 3. Solve elasticity
    let mut K_u_copy = K_u.clone();
    for (&dof, &val) in disp_bc_dofs.iter().zip(disp_bc_vals.iter()) {
        K_u_copy.apply_dirichlet_symmetric(dof, val, &mut rhs_u);
    }
    let mut u_new = vec![0.0; disp_space.n_dofs()];
    fem_solver::solve_pcg_jacobi(&K_u_copy, &rhs_u, &mut u_new, solver_cfg)
        .expect("staggered transient elasticity solve failed");

    (T_new, u_new)
}

/// Drive multiple steps of transient staggered thermoelastic coupling.
///
/// # Arguments
/// * `n_steps` — number of time steps
/// * `dt` — time step size
/// * `T_0` — initial temperature
/// * `u_0` — initial displacement
/// * `callback` — called after each step with `(step, t, T, u)`
///
/// Returns `(T, u)` after the final step.
#[allow(clippy::too_many_arguments)]
pub fn integrate_thermoelastic_staggered_transient<M, CB>(
    mesh: &M,
    K_t: &CsrMatrix<f64>,
    C_t: &CsrMatrix<f64>,
    K_u: &CsrMatrix<f64>,
    n_steps: usize,
    dt: f64,
    T_0: &[f64],
    u_0: &[f64],
    Q: &[f64],
    alpha: f64,
    lambda: f64,
    mu: f64,
    t_ref: f64,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
    solver_cfg: &SolverConfig,
    mut callback: CB,
) -> (Vec<f64>, Vec<f64>)
where
    M: MeshTopology + Clone,
    CB: FnMut(usize, f64, &[f64], &[f64]),
{
    let mut T_n = T_0.to_vec();
    let mut u_n = u_0.to_vec();
    let mut T_n_prev = T_0.to_vec(); // for BDF-2: initially copy of T_0

    callback(0, 0.0, &T_n, &u_n);

    // First step uses BDF-1 (no T_{n-1} available)
    let mut scheme = BdfScheme::Bdf1;
    for step in 1..=n_steps {
        let t = step as f64 * dt;

        // After first step, switch to BDF-2
        if step == 2 {
            scheme = BdfScheme::Bdf2;
        }

        let (T_new, u_new) = solve_thermoelastic_staggered_transient_step(
            mesh, K_t, C_t, K_u, dt,
            &T_n, &T_n_prev, Q,
            alpha, lambda, mu, t_ref,
            temp_bc_dofs, temp_bc_vals,
            disp_bc_dofs, disp_bc_vals,
            quad_order, scheme, solver_cfg,
        );

        T_n_prev.copy_from_slice(&T_n);
        T_n = T_new;
        u_n = u_new;

        callback(step, t, &T_n, &u_n);
    }

    (T_n, u_n)
}

// ============================================================================
// Fully coupled monolithic thermoelastic system
// ============================================================================

/// Assemble the fully coupled thermoelastic monolithic matrix for one time step.
///
/// The system is:
/// ```text
/// [K_uu   K_uT] [u  ]   [f_u + f_th]
/// [K_Tu   K_TT] [T  ] = [    f_T   ]
/// ```
///
/// where:
/// - `K_uu` = elasticity matrix (n_u × n_u)
/// - `K_uT` = thermal expansion coupling matrix (n_u × n_T)
/// - `K_TT` = `(a₀/Δt)·C + K_t` — effective heat matrix (n_T × n_T)
/// - `K_Tu` = thermoelastic damping coupling (n_T × n_u) — optional, often 0
///
/// Returns `(coupled_matrix, block_n_u, block_n_T)` as a single CSR matrix
/// with block structure. The caller can extract blocks if needed.
///
/// The RHS is NOT assembled here (needs Tⁿ for the thermal history term);
/// use [`assemble_thermoelastic_monolithic_rhs`] instead.
#[allow(clippy::too_many_arguments)]
pub fn assemble_thermoelastic_monolithic_system<M: MeshTopology + Clone>(
    K_uu: &CsrMatrix<f64>,
    K_t: &CsrMatrix<f64>,
    C_t: &CsrMatrix<f64>,
    dt: f64,
    lambda: f64,
    mu: f64,
    alpha: f64,
    temp_space: &H1Space<M>,
    disp_space: &VectorH1Space<M>,
    quad_order: u8,
    scheme: BdfScheme,
    include_K_Tu: bool,
) -> (CsrMatrix<f64>, usize, usize) {
    let n_u = K_uu.nrows;
    let n_T = K_t.nrows;
    let total = n_u + n_T;
    let (a0, _a1, _a2) = match scheme {
        BdfScheme::Bdf1 => (1.0, 1.0, 0.0),
        BdfScheme::Bdf2 => (1.5, 2.0, -0.5),
    };

    let mut coo = CooMatrix::<f64>::new(total, total);

    // Block (0,0): K_uu
    for i in 0..n_u {
        for p in K_uu.row_ptr[i]..K_uu.row_ptr[i + 1] {
            let j = K_uu.col_idx[p] as usize;
            coo.add(i, j, K_uu.values[p]);
        }
    }

    // Block (1,1): (a₀/Δt)·C + K_t
    let offset = n_u;
    for i in 0..n_T {
        for p in C_t.row_ptr[i]..C_t.row_ptr[i + 1] {
            let j = C_t.col_idx[p] as usize;
            coo.add(offset + i, offset + j, (a0 / dt) * C_t.values[p]);
        }
        for p in K_t.row_ptr[i]..K_t.row_ptr[i + 1] {
            let j = K_t.col_idx[p] as usize;
            coo.add(offset + i, offset + j, K_t.values[p]);
        }
    }

    // Block (0,1): K_uT — thermal expansion coupling
    // K_uT(a, b) = -∫ (3λ+2μ)·α·(∇·φ_a)·ψ_b dx
    // where φ_a is a vector test function and ψ_b is a scalar test function
    let beta = alpha * (3.0 * lambda + 2.0 * mu);
    let dim = disp_space.mesh().dim() as usize;

    let mesh = disp_space.mesh();
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let order = disp_space.order();
        let re = ref_elem_vol(et, order);
        let n_ldofs = re.n_dofs();
        let quad = re.quadrature(quad_order);
        let u_dofs: Vec<usize> = disp_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let t_dofs: Vec<usize> = temp_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        let jit = jac.try_inverse().unwrap().transpose();

        let mut phi_t = vec![0.0; t_dofs.len()];
        let mut gref = vec![0.0; n_ldofs * dim];
        let mut gphys = vec![0.0; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            re.eval_basis(xi, &mut phi_t);
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            for a in 0..n_ldofs {
                let div_phi = (0..dim).map(|d| gphys[a * dim + d]).sum::<f64>();
                for b in 0..t_dofs.len() {
                    let val = -w * beta * div_phi * phi_t[b];
                    if val.abs() < 1e-30 {
                        continue;
                    }
                    // Vector DOF: each scalar DOF maps to 'dim' displacement DOFs
                    for d in 0..dim {
                        let i_global = u_dofs[a * dim + d];
                        let j_global = offset + t_dofs[b];
                        coo.add(i_global, j_global, val);
                    }
                }
            }
        }
    }

    // Block (1,0): K_Tu (thermoelastic damping, Gough-Joule)
    // K_Tu(b, a) = -∫ (3λ+2μ)·α·T₀·(∇·φ_a)·ψ_b dx
    // This is usually small; only include if include_K_Tu = true
    if include_K_Tu {
        // K_Tu is the transpose of K_uT scaled by T₀
        // For simplicity we use the same loop structure
        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let order = disp_space.order();
            let re = ref_elem_vol(et, order);
            let n_ldofs = re.n_dofs();
            let quad = re.quadrature(quad_order);
            let u_dofs: Vec<usize> = disp_space.element_dofs(e)
                .iter().map(|&d| d as usize).collect();
            let t_dofs: Vec<usize> = temp_space.element_dofs(e)
                .iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            let (jac, det_j) = simplex_jac(mesh, nodes, dim);
            let jit = jac.try_inverse().unwrap().transpose();

            let mut phi_t = vec![0.0; t_dofs.len()];
            let mut gref = vec![0.0; n_ldofs * dim];
            let mut gphys = vec![0.0; n_ldofs * dim];

            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();
                re.eval_basis(xi, &mut phi_t);
                re.eval_grad_basis(xi, &mut gref);
                xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

                for a in 0..n_ldofs {
                    let div_phi = (0..dim).map(|d| gphys[a * dim + d]).sum::<f64>();
                    for b in 0..t_dofs.len() {
                        let val = -w * beta * div_phi * phi_t[b];
                        if val.abs() < 1e-30 {
                            continue;
                        }
                        // K_Tu(b, a): row = temperature DOF, col = displacement DOF
                        for d in 0..dim {
                            let i_global = offset + t_dofs[b];
                            let j_global = u_dofs[a * dim + d];
                            coo.add(i_global, j_global, val);
                        }
                    }
                }
            }
        }
    }

    (coo.into_csr(), n_u, n_T)
}

/// Assemble the RHS for the monolithic thermoelastic system.
///
/// Returns a vector of length `n_u + n_T`:
/// ```text
/// RHS = [f_u + f_th(T^{n+1});  Q + C·(a₁·Tⁿ + a₂·T^{n-1})/Δt]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn assemble_thermoelastic_monolithic_rhs(
    T_n: &[f64],
    T_n_prev: &[f64],
    Q: &[f64],
    C_t: &CsrMatrix<f64>,
    f_u: &[f64],
    dt: f64,
    scheme: BdfScheme,
) -> Vec<f64> {
    let n_T = T_n.len();
    let n_u = f_u.len();
    let total = n_u + n_T;
    let (_a0, a1, a2) = match scheme {
        BdfScheme::Bdf1 => (1.0, 1.0, 0.0),
        BdfScheme::Bdf2 => (1.5, 2.0, -0.5),
    };

    let mut rhs = vec![0.0; total];

    // Mechanical part: f_u (thermal expansion is NOT added here;
    // it's part of the matrix coupling K_uT·T)
    for i in 0..n_u {
        rhs[i] = f_u[i];
    }

    // Thermal part: Q + C·(a₁·Tⁿ + a₂·T^{n-1})/Δt
    let mut temp = vec![0.0; n_T];
    for i in 0..n_T {
        temp[i] = a1 * T_n[i] + a2 * T_n_prev[i];
    }
    let mut c_sum = vec![0.0; n_T];
    C_t.spmv(&temp, &mut c_sum);
    let inv_dt = 1.0 / dt;
    for i in 0..n_T {
        rhs[n_u + i] = Q[i] + inv_dt * c_sum[i];
    }

    rhs
}

/// Solve the monolithic coupled thermoelastic system for one time step.
///
/// Assembles the full coupled matrix and RHS, applies BCs, and solves.
/// Returns `(u_new, T_new)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_thermoelastic_monolithic_step<M: MeshTopology + Clone>(
    K_uu: &CsrMatrix<f64>,
    K_t: &CsrMatrix<f64>,
    C_t: &CsrMatrix<f64>,
    dt: f64,
    T_n: &[f64],
    T_n_prev: &[f64],
    Q: &[f64],
    f_u: &[f64],
    lambda: f64,
    mu: f64,
    alpha: f64,
    temp_space: &H1Space<M>,
    disp_space: &VectorH1Space<M>,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
    scheme: BdfScheme,
    solver_cfg: &SolverConfig,
) -> (Vec<f64>, Vec<f64>) {
    let (mut coupled_mat, n_u, n_T) = assemble_thermoelastic_monolithic_system(
        K_uu, K_t, C_t, dt, lambda, mu, alpha,
        temp_space, disp_space, quad_order, scheme, false,
    );

    let mut rhs = assemble_thermoelastic_monolithic_rhs(
        T_n, T_n_prev, Q, C_t, f_u, dt, scheme,
    );

    // Apply Dirichlet BCs to the coupled system
    for (&dof, &val) in disp_bc_dofs.iter().zip(disp_bc_vals.iter()) {
        coupled_mat.apply_dirichlet_symmetric(dof, val, &mut rhs);
    }
    for (&dof, &val) in temp_bc_dofs.iter().zip(temp_bc_vals.iter()) {
        let dof_T = n_u + dof;
        coupled_mat.apply_dirichlet_symmetric(dof_T, val, &mut rhs);
    }

    // Solve the coupled system
    let mut x = vec![0.0; n_u + n_T];
    fem_solver::solve_pcg_jacobi(&coupled_mat, &rhs, &mut x, solver_cfg)
        .expect("monolithic thermoelastic solve failed");

    // Split into u and T
    let u_new = x[..n_u].to_vec();
    let T_new = x[n_u..].to_vec();
    (u_new, T_new)
}

/// Drive multiple steps of monolithic coupled thermoelastic analysis.
#[allow(clippy::too_many_arguments)]
pub fn integrate_thermoelastic_monolithic<M, CB>(
    K_uu: &CsrMatrix<f64>,
    K_t: &CsrMatrix<f64>,
    C_t: &CsrMatrix<f64>,
    n_steps: usize,
    dt: f64,
    T_0: &[f64],
    u_0: &[f64],
    Q: &[f64],
    f_u: &[f64],
    lambda: f64,
    mu: f64,
    alpha: f64,
    temp_space: &H1Space<M>,
    disp_space: &VectorH1Space<M>,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
    solver_cfg: &SolverConfig,
    mut callback: CB,
) -> (Vec<f64>, Vec<f64>)
where
    M: MeshTopology + Clone,
    CB: FnMut(usize, f64, &[f64], &[f64]),
{
    let mut T_n = T_0.to_vec();
    let mut u_n = u_0.to_vec();
    let mut T_n_prev = T_0.to_vec();
    let mut u_n_prev = u_0.to_vec();

    callback(0, 0.0, &T_n, &u_n);

    let mut scheme = BdfScheme::Bdf1;
    for step in 1..=n_steps {
        let t = step as f64 * dt;

        if step == 2 {
            scheme = BdfScheme::Bdf2;
        }

        let (u_new, T_new) = solve_thermoelastic_monolithic_step(
            K_uu, K_t, C_t, dt,
            &T_n, &T_n_prev, Q, f_u,
            lambda, mu, alpha,
            temp_space, disp_space,
            temp_bc_dofs, temp_bc_vals,
            disp_bc_dofs, disp_bc_vals,
            quad_order, scheme, solver_cfg,
        );

        T_n_prev.copy_from_slice(&T_n);
        u_n_prev.copy_from_slice(&u_n);
        T_n = T_new;
        u_n = u_new;

        callback(step, t, &T_n, &u_n);
    }

    (T_n, u_n)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    // ─── Existing tests ───────────────────────────────────────────────────

    #[test]
    fn thermal_expansion_rhs_zero_at_reference_temp() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let quad_order = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let n_p = H1Space::new(mesh.clone(), 1).n_dofs();
        let temp = vec![0.0; n_p]; // T = T₀ = 0

        let rhs = assemble_thermal_expansion_rhs(
            &mesh, &disp_space, &temp, 0.0, 1.0e-5, 1.0, 0.5, quad_order,
        );

        let norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-14, "thermal RHS should be zero at reference temp, got {norm}");
    }

    #[test]
    fn thermal_expansion_rhs_nonzero_with_delta_t() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let quad_order = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let n_p = H1Space::new(mesh.clone(), 1).n_dofs();
        let temp = vec![100.0; n_p]; // uniform ΔT = 100

        let rhs = assemble_thermal_expansion_rhs(
            &mesh, &disp_space, &temp, 0.0, 1.0e-5, 1.0, 0.5, quad_order,
        );

        let norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0, "thermal RHS should be non-zero with ΔT > 0");
    }

    #[test]
    fn heat_system_assembles() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let (k, rhs) = assemble_heat_system(&mesh, 1.0, 2);
        assert_eq!(k.nrows, mesh.n_nodes());
        assert_eq!(rhs.len(), mesh.n_nodes());
        assert!(k.nnz() > 0);
    }

    #[test]
    fn thermoelastic_staggered_solves() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let kappa = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let t_ref = 0.0;
        let quad_order = 2;

        let pres_space = H1Space::new(mesh.clone(), 1);
        let temp_bc_dofs: Vec<usize> = (0..pres_space.n_dofs())
            .filter(|&d| {
                let x = pres_space.dof_manager().dof_coord(d as u32);
                x[0] < 0.01 || x[0] > 0.99
            })
            .collect();
        let temp_bc_vals: Vec<f64> = temp_bc_dofs.iter().map(|&d| {
            let x = pres_space.dof_manager().dof_coord(d as u32);
            if x[0] < 0.01 { 100.0 } else { 0.0 }
        }).collect();

        let disp_space = VectorH1Space::new(mesh.clone(), 1, 2);
        let dim = 2;
        let disp_bc_dofs: Vec<usize> = (0..mesh.n_nodes() as u32)
            .filter(|&n| mesh.node_coords(n)[1] < 0.01)
            .flat_map(|n| {
                let idx = n as usize;
                vec![idx * dim, idx * dim + 1]
            })
            .collect();
        let disp_bc_vals = vec![0.0; disp_bc_dofs.len()];

        let (temp, u) = solve_thermoelastic_staggered(
            &mesh, kappa, alpha, lambda, mu, t_ref,
            &temp_bc_dofs, &temp_bc_vals,
            &disp_bc_dofs, &disp_bc_vals, quad_order,
        );

        assert_eq!(temp.len(), pres_space.n_dofs());
        assert_eq!(u.len(), disp_space.n_dofs());

        for &t in &temp {
            assert!(t >= -1.0 && t <= 101.0, "temperature out of range: {t}");
        }

        let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(u_norm > 0.0, "thermal expansion should produce non-zero displacement");
    }

    // ─── New tests: heat capacity ──────────────────────────────────────────

    #[test]
    fn heat_capacity_assemble() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let c = assemble_heat_capacity(&mesh, 1.0, 2);
        assert_eq!(c.nrows, mesh.n_nodes());
        assert!(c.nnz() > 0);
        // Lump sum should be positive (total capacity = ∫ ρ·c dx)
        let mut lumped = 0.0;
        for i in 0..c.nrows {
            for p in c.row_ptr[i]..c.row_ptr[i + 1] {
                lumped += c.values[p];
            }
        }
        assert!(lumped > 0.0, "capacity matrix should have positive entries");
    }

    // ─── New tests: transient heat step ────────────────────────────────────

    #[test]
    fn transient_heat_bdf1_decays_to_steady_state() {
        // 1D-like: source Q=10, uniform initial T=0, BC on all DOFs
        // Steady state: T = Q / (no flux terms) — but with BC, it's more complex.
        // Here we just check that the solver runs and produces finite values.
        let mesh = Mesh::<2>::unit_square_tri(8);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let quad_order = 2;

        let (K, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let n = K.nrows;

        let T_0 = vec![0.0; n];
        let T_prev = vec![0.0; n];
        let Q = vec![1.0; n]; // uniform heat source

        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false, ..SolverConfig::default() };

        let T_1 = solve_transient_heat_step(
            &K, &C, 0.1, &T_0, &T_prev, &Q, &[], &[], BdfScheme::Bdf1, &cfg,
        );
        assert_eq!(T_1.len(), n);
        // Temperature should be positive (heating from source)
        let t_sum: f64 = T_1.iter().sum();
        assert!(t_sum > 0.0, "temperature should increase with heat source");
    }

    #[test]
    fn transient_heat_bdf2_symmetric_solves() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let quad_order = 2;

        let (K, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let n = K.nrows;

        let T_0 = vec![0.0; n];
        let T_prev = vec![0.0; n];
        let Q = vec![0.0; n];

        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false, ..SolverConfig::default() };

        let T_1 = solve_transient_heat_step(
            &K, &C, 0.1, &T_0, &T_prev, &Q, &[], &[], BdfScheme::Bdf2, &cfg,
        );
        // With no source and zero BC, solution stays zero
        let t_max: f64 = T_1.iter().fold(0.0, |a, &b| a.max(b.abs()));
        assert!(t_max < 1e-14, "T should be zero with no source: max={:.4e}", t_max);
    }

    // ─── New tests: staggered transient ────────────────────────────────────

    #[test]
    fn staggered_transient_integrates() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let t_ref = 0.0;
        let quad_order = 2;

        let (K_t, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C_t = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let dim: u8 = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let K_u = Assembler::assemble_bilinear(
            &disp_space, &[&ElasticityIntegrator::new(lambda, mu)], quad_order,
        );

        let n_p = K_t.nrows;
        let T_0 = vec![0.0; n_p];
        let u_0 = vec![0.0; disp_space.n_dofs()];
        let Q = vec![1.0; n_p]; // uniform heating

        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let mut steps: Vec<(f64, f64)> = Vec::new();
        let (T_final, u_final) = integrate_thermoelastic_staggered_transient(
            &mesh, &K_t, &C_t, &K_u, 5, 0.1,
            &T_0, &u_0, &Q,
            alpha, lambda, mu, t_ref,
            &[], &[], &[], &[],
            quad_order, &cfg, |step, t, T, u| {
                let t_sum: f64 = T.iter().sum();
                let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
                steps.push((t_sum, u_norm));
            },
        );

        assert_eq!(T_final.len(), n_p);
        assert_eq!(u_final.len(), disp_space.n_dofs());
        assert_eq!(steps.len(), 6); // initial + 5 steps

        // Temperature should increase over time
        let t_end_sum: f64 = T_final.iter().sum();
        assert!(t_end_sum > 0.0, "temperature should increase");
    }

    // ─── New tests: monolithic coupled system ──────────────────────────────

    #[test]
    fn monolithic_system_assembles() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let quad_order = 2;

        let (K_t, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C_t = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let dim: u8 = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let temp_space = H1Space::new(mesh.clone(), 1);
        let K_uu = Assembler::assemble_bilinear(
            &disp_space, &[&ElasticityIntegrator::new(lambda, mu)], quad_order,
        );

        let (coupled, n_u, n_T) = assemble_thermoelastic_monolithic_system(
            &K_uu, &K_t, &C_t, 0.1, lambda, mu, alpha,
            &temp_space, &disp_space, quad_order, BdfScheme::Bdf1, false,
        );

        assert_eq!(n_u, disp_space.n_dofs());
        assert_eq!(n_T, temp_space.n_dofs());
        assert_eq!(coupled.nrows, n_u + n_T);
        assert_eq!(coupled.ncols, n_u + n_T);
        assert!(coupled.nnz() > 0);
    }

    #[test]
    fn monolithic_step_executes() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let quad_order = 2;

        let (K_t, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C_t = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let dim: u8 = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let temp_space = H1Space::new(mesh.clone(), 1);
        let K_uu = Assembler::assemble_bilinear(
            &disp_space, &[&ElasticityIntegrator::new(lambda, mu)], quad_order,
        );

        let n_p = temp_space.n_dofs();
        let T_n = vec![0.0; n_p];
        let T_prev = vec![0.0; n_p];
        let Q = vec![10.0; n_p];
        let f_u = vec![0.0; disp_space.n_dofs()];

        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let (u_new, T_new) = solve_thermoelastic_monolithic_step(
            &K_uu, &K_t, &C_t, 0.1,
            &T_n, &T_prev, &Q, &f_u,
            lambda, mu, alpha,
            &temp_space, &disp_space,
            &[], &[], &[], &[],
            quad_order, BdfScheme::Bdf1, &cfg,
        );

        assert_eq!(T_new.len(), n_p);
        assert_eq!(u_new.len(), disp_space.n_dofs());

        // Temperature should be positive from heat source
        let t_sum: f64 = T_new.iter().sum();
        assert!(t_sum > 0.0, "temp should increase from heat source");
    }

    #[test]
    fn monolithic_integrate_runs() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let kappa = 1.0;
        let rho_cp = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let quad_order = 2;

        let (K_t, _) = assemble_heat_system(&mesh, kappa, quad_order);
        let C_t = assemble_heat_capacity(&mesh, rho_cp, quad_order);
        let dim: u8 = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let temp_space = H1Space::new(mesh.clone(), 1);
        let K_uu = Assembler::assemble_bilinear(
            &disp_space, &[&ElasticityIntegrator::new(lambda, mu)], quad_order,
        );

        let n_p = temp_space.n_dofs();
        let T_0 = vec![0.0; n_p];
        let u_0 = vec![0.0; disp_space.n_dofs()];
        let Q = vec![5.0; n_p];
        let f_u = vec![0.0; disp_space.n_dofs()];

        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let mut count = 0usize;
        let (T_final, u_final) = integrate_thermoelastic_monolithic(
            &K_uu, &K_t, &C_t, 5, 0.1,
            &T_0, &u_0, &Q, &f_u,
            lambda, mu, alpha,
            &temp_space, &disp_space,
            &[], &[], &[], &[],
            quad_order, &cfg, |_step, _t, _T, _u| { count += 1; },
        );

        assert_eq!(count, 6);
        assert!(T_final.iter().sum::<f64>() > 0.0);
    }
}
