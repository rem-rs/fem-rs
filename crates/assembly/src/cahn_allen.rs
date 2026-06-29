//! Cahn-Hilliard and Allen-Cahn phase field solvers.
//!
//! Both use semi-implicit (IMEX) time stepping with linear terms implicit
//! and the nonlinear double-well potential explicit.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_solver::{SolverConfig, solve_cg};
use fem_space::fe_space::FESpace;
use crate::assembler::Assembler;
use crate::standard::DiffusionIntegrator;
use crate::standard::MassIntegrator;

/// Configuration for the Cahn-Hilliard solver.
#[derive(Debug, Clone)]
pub struct CahnHilliardConfig {
    pub mobility: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub t_max: f64,
    pub output_interval: usize,
    pub solver_cfg: SolverConfig,
}

impl Default for CahnHilliardConfig {
    fn default() -> Self {
        CahnHilliardConfig {
            mobility: 1.0, epsilon: 0.02, dt: 1e-5, t_max: 0.01,
            output_interval: 0,
            solver_cfg: SolverConfig { rtol: 1e-10, max_iter: 1000, ..SolverConfig::default() },
        }
    }
}

pub struct CahnHilliardResult {
    pub c: Vec<f64>,
    pub mu: Vec<f64>,
    pub free_energy: Vec<f64>,
    pub times: Vec<f64>,
}

/// Build the system matrix M + dt·mob·ε²·K from mass and stiffness matrices.
fn build_cahn_hilliard_sys(mass: &CsrMatrix<f64>, stiff: &CsrMatrix<f64>, n: usize, dt: f64, mob: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    let md = mass.diagonal();
    for i in 0..n {
        let start = stiff.row_ptr[i];
        let end = stiff.row_ptr[i + 1];
        for k in start..end {
            let j = stiff.col_idx[k] as usize;
            let k_val = stiff.values[k];
            let m_val = if i == j { md[i] } else { 0.0 };
            coo.add(i, j, m_val + dt * mob * k_val);
        }
    }
    coo.into_csr()
}

/// Solve the Cahn-Hilliard equation with IMEX time stepping.
pub fn solve_cahn_hilliard<M: MeshTopology + Clone + Send + Sync>(
    mesh: &M, order: u8, c0: Vec<f64>, quad_order: u8, cfg: &CahnHilliardConfig,
) -> CahnHilliardResult {
    let space = fem_space::H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    let eps2 = cfg.epsilon * cfg.epsilon;
    let mob = cfg.mobility;

    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order);
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: eps2 }], quad_order);
    let sys_mat = build_cahn_hilliard_sys(&mass, &stiff, n_dofs, cfg.dt, mob);

    let mut c = c0;
    let mut mu = vec![0.0; n_dofs];
    let mut free_energy = Vec::new();
    let mut times = Vec::new();
    let n_steps = (cfg.t_max / cfg.dt).ceil() as usize;

    for step in 0..n_steps {
        let t = (step + 1) as f64 * cfg.dt;

        // RHS = M·c^n + dt·mob·K·c^n - dt·mob·M·(c³-c)
        let mut mass_c = vec![0.0; n_dofs];
        mass.spmv(&c, &mut mass_c);
        let mut kc = vec![0.0; n_dofs];
        stiff.spmv(&c, &mut kc);
        let mut nl = vec![0.0; n_dofs];
        for i in 0..n_dofs { nl[i] = c[i] * c[i] * c[i] - c[i]; }
        let mut m_nl = vec![0.0; n_dofs];
        mass.spmv(&nl, &mut m_nl);

        let mut rhs = vec![0.0; n_dofs];
        for i in 0..n_dofs {
            rhs[i] = mass_c[i] + cfg.dt * mob * kc[i] - cfg.dt * mob * m_nl[i];
        }

        solve_cg(&sys_mat, &rhs, &mut c, &cfg.solver_cfg).ok();

        // μ = f₀'(c) - ε²∇²c  via M·μ = M·(c³-c) + K·c
        for i in 0..n_dofs { nl[i] = c[i] * c[i] * c[i] - c[i]; }
        mass.spmv(&nl, &mut m_nl);
        stiff.spmv(&c, &mut kc);
        let mut mu_rhs = vec![0.0; n_dofs];
        for i in 0..n_dofs { mu_rhs[i] = m_nl[i] + kc[i]; }
        solve_cg(&mass, &mu_rhs, &mut mu, &cfg.solver_cfg).ok();

        let energy = compute_cahn_hilliard_energy_raw(&mass, &stiff, &c, cfg.epsilon);
        free_energy.push(energy);
        times.push(t);

        if cfg.output_interval > 0 && step % cfg.output_interval == 0 {
            eprintln!("CH step {step}: energy={energy:.6e}");
        }
    }

    CahnHilliardResult { c, mu, free_energy, times }
}

/// Estimate free energy using assembled operators: E = ¼∫(1-c²)² + ε²/2·cᵀK·c
fn compute_cahn_hilliard_energy_raw(stiff: &CsrMatrix<f64>, _mass: &CsrMatrix<f64>, c: &[f64], _epsilon: f64) -> f64 {
    let mut kc = vec![0.0; c.len()];
    stiff.spmv(c, &mut kc);
    let grad_energy: f64 = c.iter().zip(kc.iter()).map(|(&ci, &kci)| ci * kci).sum();
    let mut well_energy = 0.0;
    for &ci in c {
        let c2 = ci * ci;
        well_energy += 0.25 * (1.0 - c2) * (1.0 - c2);
    }
    // Gradient energy is already scaled by ε² via stiff = ε²·K
    // Well energy uses nodal quadrature as a rough approximation
    0.5 * grad_energy + well_energy
}

// ─── Allen-Cahn ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AllenCahnConfig {
    pub l_factor: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub t_max: f64,
    pub output_interval: usize,
    pub solver_cfg: SolverConfig,
}

impl Default for AllenCahnConfig {
    fn default() -> Self {
        AllenCahnConfig {
            l_factor: 1.0, epsilon: 0.02, dt: 1e-5, t_max: 0.01,
            output_interval: 0,
            solver_cfg: SolverConfig { rtol: 1e-10, max_iter: 1000, ..SolverConfig::default() },
        }
    }
}

pub struct AllenCahnResult {
    pub c: Vec<f64>,
    pub energy: Vec<f64>,
    pub times: Vec<f64>,
}

/// Solve Allen-Cahn with IMEX time stepping.
pub fn solve_allen_cahn<M: MeshTopology + Clone + Send + Sync>(
    mesh: &M, order: u8, c0: Vec<f64>, quad_order: u8, cfg: &AllenCahnConfig,
) -> AllenCahnResult {
    let space = fem_space::H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad_order);
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: cfg.epsilon * cfg.epsilon }], quad_order);
    let sys_mat = build_cahn_hilliard_sys(&mass, &stiff, n_dofs, cfg.dt, cfg.l_factor);

    let mut c = c0;
    let mut energy = Vec::new();
    let mut times = Vec::new();
    let n_steps = (cfg.t_max / cfg.dt).ceil() as usize;

    for step in 0..n_steps {
        let t = (step + 1) as f64 * cfg.dt;
        let mut mass_c = vec![0.0; n_dofs];
        mass.spmv(&c, &mut mass_c);
        let mut nl = vec![0.0; n_dofs];
        for i in 0..n_dofs { nl[i] = c[i] * c[i] * c[i] - c[i]; }
        let mut m_nl = vec![0.0; n_dofs];
        mass.spmv(&nl, &mut m_nl);
        let mut rhs = vec![0.0; n_dofs];
        for i in 0..n_dofs { rhs[i] = mass_c[i] - cfg.dt * cfg.l_factor * m_nl[i]; }
        solve_cg(&sys_mat, &rhs, &mut c, &cfg.solver_cfg).ok();

        let eng = compute_cahn_hilliard_energy_raw(&stiff, &mass, &c, cfg.epsilon);
        energy.push(eng);
        times.push(t);
        if cfg.output_interval > 0 && step % cfg.output_interval == 0 {
            eprintln!("AC step {step}: energy={eng:.6e}");
        }
    }
    AllenCahnResult { c, energy, times }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn cahn_hilliard_solver_runs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let order = 1;
        let quad = order + 1;
        let space = fem_space::H1Space::new(mesh.clone(), order);
        let n = space.n_dofs();
        let mut c0 = vec![0.5; n];
        for i in 0..n { c0[i] += 0.1 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos(); }

        let cfg = CahnHilliardConfig { epsilon: 0.2, dt: 1e-6, t_max: 2e-6, ..Default::default() };
        let r = solve_cahn_hilliard(&mesh, order, c0, quad, &cfg);
        assert!(r.free_energy.len() >= 2, "should have at least 2 time steps");
        for v in &r.free_energy { assert!(v.is_finite(), "non-finite energy"); }
    }

    #[test]
    fn allen_cahn_solver_runs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(6);
        let order = 1; let quad = order + 1;
        let n = fem_space::H1Space::new(mesh.clone(), order).n_dofs();
        let mut c0 = vec![0.0; n];
        for i in 0..n { c0[i] = (std::f64::consts::PI * i as f64 / n as f64).sin(); }

        let cfg = AllenCahnConfig { epsilon: 0.2, dt: 1e-6, t_max: 2e-6, ..Default::default() };
        let r = solve_allen_cahn(&mesh, order, c0, quad, &cfg);
        assert!(r.energy.len() >= 2, "should have at least 2 time steps");
        for v in &r.energy { assert!(v.is_finite(), "non-finite energy"); }
    }
}
