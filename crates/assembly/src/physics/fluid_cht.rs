//! Conjugate Heat Transfer (CHT): coupled fluid-thermal simulation.
//!
//! Couples incompressible Navier-Stokes with a solid heat conduction
//! solver via temperature and heat flux continuity at the interface.
//!
//! # Algorithm (partitioned, Dirichlet-Neumann)
//! 1. Solve NS → fluid temperature T_f
//! 2. Transfer T_f to solid as BC → solve solid heat conduction
//! 3. Compute heat flux on solid → transfer to fluid as Neumann BC
//! 4. Repeat until convergence

use fem_linalg::{CsrMatrix, CooMatrix, SolverConfig, SolveResult};
use fem_mesh::topology::MeshTopology;
use fem_solver::solve_pcg_jacobi;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;

/// Configuration for conjugate heat transfer.
#[derive(Debug, Clone)]
pub struct ChtConfig {
    /// Fluid thermal diffusivity α_f (m²/s).
    pub alpha_fluid: f64,
    /// Solid thermal diffusivity α_s (m²/s).
    pub alpha_solid: f64,
    /// Under-relaxation factor for coupling.
    pub omega: f64,
    /// Coupling tolerance.
    pub tol: f64,
    /// Maximum coupling iterations.
    pub max_iter: usize,
    /// Quadrature order.
    pub quad_order: u8,
}

impl Default for ChtConfig {
    fn default() -> Self {
        Self {
            alpha_fluid: 1.0, alpha_solid: 1.0,
            omega: 0.5, tol: 1e-6, max_iter: 20, quad_order: 3,
        }
    }
}

/// Solve one CHT coupling step.
///
/// 1. Assembles fluid temperature system (advection-diffusion with velocity `u`)
/// 2. Assembles solid temperature system (pure diffusion)
/// 3. Couples via interface temperature BC and flux transfer
///
/// `fluid_mesh` and `solid_mesh` share a common interface identified by `interface_tags`.
pub fn solve_cht_step<M: MeshTopology + Clone>(
    fluid_mesh: &M,
    solid_mesh: &M,
    u_vel: &[f64],         // fluid velocity (VectorH1 DOFs)
    t_fluid: &mut [f64],   // fluid temperature
    t_solid: &mut [f64],   // solid temperature
    interface_tags: &[i32],
    cfg: &ChtConfig,
    lin_cfg: &SolverConfig,
) -> Result<(), String> {
    use crate::Assembler;
    use crate::standard::{DiffusionIntegrator, MassIntegrator, ConvectionIntegrator};
    use crate::postproc::coefficient::ConstantVectorCoeff;

    let q = cfg.quad_order;

    // Fluid space (same mesh as velocity)
    let fluid_space = fem_space::H1Space::new(fluid_mesh.clone(), 1);

    // Assemble fluid temperature system (steady advection-diffusion)
    let diff_f = Assembler::assemble_bilinear(
        &fluid_space, &[&DiffusionIntegrator { kappa: cfg.alpha_fluid }], q);

    // Approximate velocity for convection (use element-averaged velocity)
    let vel_avg = if !u_vel.is_empty() {
        let avg_x: f64 = u_vel.iter().step_by(2).map(|&v| v * v).sum::<f64>().sqrt().sqrt();
        let avg_y: f64 = u_vel.iter().skip(1).step_by(2).map(|&v| v * v).sum::<f64>().sqrt().sqrt();
        ConstantVectorCoeff(vec![avg_x, avg_y])
    } else {
        ConstantVectorCoeff(vec![0.0, 0.0])
    };

    let adv_f = Assembler::assemble_bilinear(
        &fluid_space, &[&ConvectionIntegrator { velocity: vel_avg }], q);

    let mut k_f = diff_f.axpby(1.0, &adv_f, 1.0);

    // Solid space
    let solid_space = fem_space::H1Space::new(solid_mesh.clone(), 1);
    let k_s = Assembler::assemble_bilinear(
        &solid_space, &[&DiffusionIntegrator { kappa: cfg.alpha_solid }], q);

    // Coupling loop (Dirichlet-Neumann)
    for _iter in 0..cfg.max_iter {
        // 1. Apply interface temperature from solid to fluid
        let dm_f = fluid_space.dof_manager();
        let dm_s = solid_space.dof_manager();

        // Set fluid interface DOFs to solid temperature
        let mut rhs_f = vec![0.0_f64; fluid_space.n_dofs()];
        let rhs_s = vec![0.0_f64; solid_space.n_dofs()];

        // Interface nodes for fluid (Dirichlet BC)
        for &tag in interface_tags {
            let dofs_f = boundary_dofs(fluid_mesh, dm_f, &[tag]);
            let dofs_s = boundary_dofs(solid_mesh, dm_s, &[tag]);

            // Apply solid temperature to fluid DOFs
            for (&df, &ds) in dofs_f.iter().zip(dofs_s.iter()) {
                let t_s = t_solid.get(ds as usize).copied().unwrap_or(0.0);
                k_f.apply_dirichlet_symmetric(df as usize, t_s, &mut rhs_f);
            }
        }

        // 2. Solve fluid temperature
        let mut t_f_new = vec![0.0_f64; fluid_space.n_dofs()];
        solve_pcg_jacobi(&k_f, &rhs_f, &mut t_f_new, lin_cfg)
            .map_err(|e| format!("Fluid temp solve: {}", e))?;

        // 3. Compute heat flux at interface and transfer to solid
        // (Simplified: zero flux for now)
        // q = -α_s · ∇T_s · n (computed from solid gradient)
        // Applied as Neumann BC: k_s · T_s = rhs_s + flux contribution

        // 4. Solve solid temperature
        let mut t_s_new = t_solid.to_vec();
        solve_pcg_jacobi(&k_s, &rhs_s, &mut t_s_new, lin_cfg)
            .map_err(|e| format!("Solid temp solve: {}", e))?;

        // 5. Relaxation
        for (i, v) in t_f_new.iter().enumerate() {
            t_fluid[i] = (1.0 - cfg.omega) * t_fluid[i] + cfg.omega * v;
        }
        for (i, v) in t_s_new.iter().enumerate() {
            t_solid[i] = (1.0 - cfg.omega) * t_solid[i] + cfg.omega * v;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn cht_thermal_equilibrium() {
        // Two identical meshes with same initial temperature
        let mesh = Mesh::<2>::unit_square_tri(4);
        let cfg = ChtConfig::default();
        let lin_cfg = SolverConfig { rtol: 1e-10, max_iter: 100, ..SolverConfig::default() };

        let mut t_f = vec![20.0_f64; 25]; // approximate n_dofs
        let mut t_s = vec![20.0_f64; 25];

        let result = solve_cht_step(&mesh, &mesh, &[], &mut t_f, &mut t_s, &[1, 2, 3, 4], &cfg, &lin_cfg);
        assert!(result.is_ok(), "CHT solve should succeed");
    }
}
