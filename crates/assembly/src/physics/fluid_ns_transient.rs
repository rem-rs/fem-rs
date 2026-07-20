//! Transient incompressible Navier-Stokes driver with BDF time integration.
//!
//! Wraps the saddle-point Oseen system into a reusable time-stepping loop
//! supporting BDF-1 (implicit Euler) and BDF-2.
//!
//! # Usage
//! ```rust,ignore
//! let mut driver = TransientNsDriver::new(mesh, 2, 1, NsConfig { nu: 0.01, .. });
//! driver.set_dt(0.01);
//! for step in 0..100 {
//!     driver.step(&mut u, &mut p, &apply_bcs)?;
//! }
//! ```

use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use crate::physics::navier_stokes::assemble_convection_matrix;
use crate::physics::fluid_cfd::NsConfig;
use super::navier_stokes::assemble_divergence_matrix;
use super::super::Assembler;
use crate::standard::MassIntegrator;

/// Transient NS driver with BDF-1/BDF-2 time integration.
#[allow(dead_code)]
pub struct TransientNsDriver<M: MeshTopology + Clone> {
    vel_space: fem_space::VectorH1Space<M>,
    pres_space: fem_space::H1Space<M>,
    config: NsConfig,
    dt: f64,
    // Time-independent operators
    m_v: CsrMatrix<f64>,    // velocity mass matrix
    a_diff: CsrMatrix<f64>, // viscous diffusion
    b: CsrMatrix<f64>,      // divergence
    bt: CsrMatrix<f64>,     // gradient
    // State history (for BDF-2)
    u_prev: Vec<f64>,
}

impl<M: MeshTopology + Clone> TransientNsDriver<M> {
    /// Create a new transient NS driver.
    pub fn new(mesh: M, vel_order: u8, pres_order: u8, config: NsConfig) -> Self {
        let vel_space = fem_space::VectorH1Space::new(mesh.clone(), vel_order, mesh.dim() as u8);
        let pres_space = fem_space::H1Space::new(mesh, pres_order);
        let q = config.quad_order;
        let n_v = vel_space.n_dofs();
        let n_s = vel_space.n_scalar_dofs();

        // Viscous diffusion block
        let a_diff = crate::physics::navier_stokes::assemble_oseen_block(
            &vel_space, &vec![0.0; n_v], config.nu, q);

        // Divergence / gradient
        let b = assemble_divergence_matrix(&vel_space, pres_space.mesh(), q);
        let bt = b.transpose();

        // Velocity mass (block-diagonal from scalar mass)
        // Build on the velocity scalar space (same order as vel_space)
        let scalar_space = fem_space::H1Space::new(vel_space.mesh().clone(), vel_order);
        let scalar_mass = Assembler::assemble_bilinear(
            &scalar_space, &[&MassIntegrator { rho: 1.0 }], q);
        let mut m_coo = CooMatrix::new(n_v, n_v);
        for c in 0..2 {
            let off = c * n_s;
            for r in 0..n_s {
                for k in scalar_mass.row_ptr[r]..scalar_mass.row_ptr[r + 1] {
                    let col = scalar_mass.col_idx[k] as usize;
                    m_coo.add(off + r, off + col, scalar_mass.values[k]);
                }
            }
        }
        let m_v = m_coo.into_csr();

        Self {
            vel_space, pres_space, config,
            dt: 0.01,
            m_v, a_diff, b, bt,
            u_prev: vec![0.0; n_v],
        }
    }

    /// Set time step size.
    pub fn set_dt(&mut self, dt: f64) { self.dt = dt; }

    /// Get spaces.
    pub fn vel_space(&self) -> &fem_space::VectorH1Space<M> { &self.vel_space }
    pub fn pres_space(&self) -> &fem_space::H1Space<M> { &self.pres_space }
    pub fn n_vel(&self) -> usize { self.vel_space.n_dofs() }
    pub fn n_pres(&self) -> usize { self.pres_space.n_dofs() }

    /// Perform one BDF time step.
    ///
    /// Assembles the Oseen system with BDF time derivative:
    /// `(α₀/Δt)·M·u^{n+1} + ν·A·u^{n+1} + C(u)·u^{n+1} + G·p = (1/Δt)·(α₁·uⁿ + α₂·u^{n-1})`
    ///
    /// `apply_bcs` is a closure that applies Dirichlet BCs to `(mat, rhs)`.
    pub fn step(
        &mut self,
        u: &mut [f64],
        p: &mut [f64],
        apply_bcs: &dyn Fn(&mut CsrMatrix<f64>, &mut [f64], &CsrMatrix<f64>, &[f64]),
    ) -> Result<(), String> {
        let n_v = self.n_vel();
        let n_p = self.n_pres();

        // BDF coefficients
        let (a0, a1, a2) = if self.u_prev.iter().any(|&v| v != 0.0) {
            // BDF-2: (3/2·u^{n+1} - 2·uⁿ + 1/2·u^{n-1}) / Δt
            (1.5, -2.0, 0.5)
        } else {
            // BDF-1 (first step): (u^{n+1} - uⁿ) / Δt
            (1.0, -1.0, 0.0)
        };
        let inv_dt = 1.0 / self.dt;

        // Convection at current velocity (Picard linearization)
        let c_conv = assemble_convection_matrix(&self.vel_space, u, self.config.quad_order);

        // Assemble Oseen operator: (α₀/Δt)·M + ν·A + C(u)
        let mut coo = CooMatrix::new(n_v, n_v);
        for r in 0..n_v {
            for k in self.m_v.row_ptr[r]..self.m_v.row_ptr[r + 1] {
                coo.add(r, self.m_v.col_idx[k] as usize, a0 * inv_dt * self.m_v.values[k]);
            }
            for k in self.a_diff.row_ptr[r]..self.a_diff.row_ptr[r + 1] {
                coo.add(r, self.a_diff.col_idx[k] as usize, self.a_diff.values[k]);
            }
            for k in c_conv.row_ptr[r]..c_conv.row_ptr[r + 1] {
                coo.add(r, c_conv.col_idx[k] as usize, c_conv.values[k]);
            }
        }
        let mut a_mat = coo.into_csr();

        // RHS = (a1·uⁿ + a2·u^{n-1}) / Δt  (from mass)
        let mut rhs = vec![0.0_f64; n_v];
        for r in 0..n_v {
            let mut sum = 0.0_f64;
            for k in self.m_v.row_ptr[r]..self.m_v.row_ptr[r + 1] {
                let col = self.m_v.col_idx[k] as usize;
                sum += self.m_v.values[k] * (a1 * u[col] + a2 * self.u_prev[col]);
            }
            rhs[r] = sum * inv_dt;
        }

        // Apply BCs
        apply_bcs(&mut a_mat, &mut rhs, &self.b, &[]);

        // Build saddle-point system
        let n_total = n_v + n_p;
        let mut sys_coo = CooMatrix::new(n_total, n_total);
        for r in 0..n_v {
            for k in a_mat.row_ptr[r]..a_mat.row_ptr[r + 1] {
                sys_coo.add(r, a_mat.col_idx[k] as usize, a_mat.values[k]);
            }
        }
        for r in 0..n_p {
            sys_coo.add(n_v + r, n_v + r, 1e-6); // pressure regularization
            for k in self.b.row_ptr[r]..self.b.row_ptr[r + 1] {
                let col = self.b.col_idx[k] as usize;
                let v = self.b.values[k];
                sys_coo.add(n_v + r, col, v);     // B
                sys_coo.add(col, n_v + r, v);     // B^T
            }
        }
        let sys = sys_coo.into_csr();

        let mut rhs_flat = vec![0.0_f64; n_total];
        rhs_flat[..n_v].copy_from_slice(&rhs);

        // Save previous state for BDF-2
        self.u_prev.copy_from_slice(u);

        // Solve saddle-point with GMRES
        let mut x = vec![0.0_f64; n_total];
        let lin_cfg = SolverConfig {
            rtol: 1e-6, max_iter: 500, verbose: false,
            ..SolverConfig::default()
        };
        match fem_solver::solve_gmres(&sys, &rhs_flat, &mut x, 50, &lin_cfg) {
            Ok(_res) => {
                u.copy_from_slice(&x[..n_v]);
                p.copy_from_slice(&x[n_v..]);
                Ok(())
            }
            Err(e) => Err(format!("GMRES failed: {}", e)),
        }
    }

    /// Run multiple steps.
    pub fn integrate(
        &mut self,
        n_steps: usize,
        u: &mut [f64],
        p: &mut [f64],
        apply_bcs: &dyn Fn(&mut CsrMatrix<f64>, &mut [f64], &CsrMatrix<f64>, &[f64]),
        report: &dyn Fn(usize, f64, &[f64], usize),
    ) -> Result<(), String> {
        for step in 0..n_steps {
            let t = step as f64 * self.dt;
            self.step(u, p, apply_bcs)?;
            report(step, t, u, 0);
        }
        Ok(())
    }
}
