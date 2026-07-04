//! Incremental plasticity driver with Newton–Raphson load stepping.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
//! use fem_assembly::plasticity_driver::{PlasticityDriver, DriverConfig, DriverResult};
//! use fem_assembly::nonlinear::{NewtonConfig, NewtonSolver};
//! use fem_mesh::SimplexMesh;
//! use fem_space::vector_h1::VectorH1Space;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(4);
//! let space = VectorH1Space::new(mesh, 1, 2);
//! let cfg = PlasticConfig::j2(2e5, 0.3, 200.0, 1e3);
//! let form = J2PlasticityForm::new(space, cfg, vec![], 2);
//! let rhs = vec![0.0; form.n_dofs()];  // external load
//!
//! let mut u = vec![0.0; form.n_dofs()];
//! let driver = PlasticityDriver::new(DriverConfig::default().n_steps(5));
//! let result = driver.solve(&form, &rhs, &mut u).unwrap();
//! println!("Converged in {} steps", result.steps.len());
//! ```

use crate::nonlinear::{NewtonConfig, NewtonSolver};

/// Configuration for the plasticity load‑stepping driver.
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// Per‑step Newton configuration.
    pub newton: NewtonConfig,
    /// Number of **equal** load increments (ignored if `load_factors` is set).
    pub n_steps: usize,
    /// Custom load‑factor schedule (cumulative, must end at 1.0).
    /// When `None`, `n_steps` uniform divisions are used.
    pub load_factors: Option<Vec<f64>>,
    /// Print a summary per load step.
    pub verbose: bool,
}

impl DriverConfig {
    /// Quick builder: set the number of uniform load steps.
    pub fn n_steps(mut self, n: usize) -> Self { self.n_steps = n; self }
    /// Quick builder: set a custom load‑factor schedule.
    pub fn load_factors(mut self, lf: Vec<f64>) -> Self { self.load_factors = Some(lf); self }
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            newton: NewtonConfig {
                atol: 1e-8, rtol: 1e-6, max_iter: 30, line_search: true,
                ..NewtonConfig::default()
            },
            n_steps: 10,
            load_factors: None,
            verbose: true,
        }
    }
}

/// One load‑step record.
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// Load factor at this step.
    pub load_factor: f64,
    /// Norm of the converged displacement.
    pub u_norm: f64,
    /// Norm of the converged residual.
    pub residual_norm: f64,
    /// Number of Newton iterations taken.
    pub newton_iters: usize,
    /// Whether the Newton solve converged.
    pub converged: bool,
}

/// Overall result of a driven plasticity solve.
#[derive(Debug, Clone)]
pub struct DriverResult {
    /// Record for each completed load step.
    pub steps: Vec<StepRecord>,
    /// Displacement at each load step (one slice per step, length n_dofs).
    pub u_history: Vec<Vec<f64>>,
    /// True if ALL steps converged.
    pub converged: bool,
}

/// Incremental plasticity driver.
///
/// Applies the external load in a user‑specified number of increments
/// (or a custom load‑factor schedule) and runs `NewtonSolver` at each step.
/// The plasticity internal state (stored inside the form via `Mutex`) persists
/// across increments automatically.
pub struct PlasticityDriver {
    cfg: DriverConfig,
    solver: NewtonSolver,
}

impl PlasticityDriver {
    pub fn new(cfg: DriverConfig) -> Self {
        let solver = NewtonSolver::new(cfg.newton.clone());
        Self { cfg, solver }
    }

    /// Run the full load–controlled plasticity analysis.
    ///
    /// - `form` — the plasticity form (implements `NonlinearForm`; state
    ///   persists across steps via interior mutability).
    /// - `rhs_base` — the **full** external load vector (the driver scales
    ///   it by the current load factor at each step).
    /// - `u` — initial guess (typically zero); on return contains the
    ///   converged solution at the final load level.
    pub fn solve(
        &self,
        form: &dyn crate::nonlinear::NonlinearForm,
        rhs_base: &[f64],
        u: &mut [f64],
    ) -> Result<DriverResult, DriverResult> {
        let n = form.n_dofs();
        assert_eq!(rhs_base.len(), n);
        assert_eq!(u.len(), n);

        // Build the load‑factor schedule
        let load_factors: Vec<f64> = match &self.cfg.load_factors {
            Some(lf) => lf.clone(),
            None => {
                (1..=self.cfg.n_steps)
                    .map(|k| k as f64 / self.cfg.n_steps as f64)
                    .collect()
            }
        };

        if load_factors.is_empty() || (load_factors.last().copied().unwrap_or(0.0) - 1.0).abs() > 1e-12 {
            return Err(DriverResult {
                steps: vec![],
                u_history: vec![],
                converged: false,
            });
        }

        let mut steps = Vec::with_capacity(load_factors.len());
        let mut u_history = Vec::with_capacity(load_factors.len());

        for (i, &lam) in load_factors.iter().enumerate() {
            if self.cfg.verbose {
                eprintln!("[Driver] step {}/{}  λ = {:.6}", i + 1, load_factors.len(), lam);
            }

            // Scale the RHS
            let rhs: Vec<f64> = if (lam - 1.0).abs() < 1e-14 {
                rhs_base.to_vec()
            } else {
                rhs_base.iter().map(|&r| lam * r).collect()
            };

            // Newton solve
            let result = self.solver.solve(form, &rhs, u);

            let converged = result.is_ok();
            let nr = result.unwrap_or_else(|e| e);

            let u_norm = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let mut r = vec![0.0; n];
            form.residual(u, &rhs, &mut r);
            let res_norm = r.iter().map(|&v| v * v).sum::<f64>().sqrt();

            steps.push(StepRecord {
                load_factor: lam,
                u_norm,
                residual_norm: res_norm,
                newton_iters: nr.iterations,
                converged,
            });
            u_history.push(u.to_vec());

            if !converged {
                if self.cfg.verbose {
                    eprintln!("[Driver] step {} did NOT converge ({} iters, ‖F‖={:.3e})",
                        i + 1, nr.iterations, res_norm);
                }
                // Return partial result
                return Err(DriverResult { steps, u_history, converged: false });
            }
        }

        Ok(DriverResult { steps, u_history, converged: true })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::NonlinearForm;
    use crate::plasticity::{J2PlasticityForm, PlasticConfig};
    use fem_mesh::topology::MeshTopology;
    use fem_mesh::SimplexMesh;
    use fem_space::vector_h1::VectorH1Space;

    /// Elastic J2: the driver should converge in one Newton iteration per step
    /// (zero residual at each converged step).
    #[test]
    fn driver_elastic_converges() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        // High yield → elastic
        let cfg = PlasticConfig::j2(2e5, 0.3, 1e8, 0.0);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let u = vec![0.0; n];
        // Dummy RHS (zero)
        let rhs = vec![0.0; n];
        let driver = PlasticityDriver::new(DriverConfig::default().n_steps(3));
        let result = driver.solve(&form, &rhs, &mut u.clone()).unwrap();
        assert!(result.converged, "Driver should converge for elastic J2");
        assert_eq!(result.steps.len(), 3, "Should have 3 load steps");
        for (i, step) in result.steps.iter().enumerate() {
            assert!(step.converged, "Step {i} should converge");
        }
    }

    /// Plastic J2: driver correctly manages load stepping and state
    /// persistence (basic acceptance test with zero load).
    #[test]
    fn driver_plastic_zero_load_converges() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let cfg = PlasticConfig::j2(2e5, 0.3, 50.0, 1e3);
        let form = J2PlasticityForm::new(space, cfg, vec![], 2);
        let n = form.n_dofs();
        let mut u = vec![0.0; n];
        let rhs = vec![0.0; n];
        let driver = PlasticityDriver::new(DriverConfig::default().n_steps(2));
        let result = driver.solve(&form, &rhs, &mut u).unwrap();
        assert!(result.converged, "Driver should converge for zero load");
        assert_eq!(result.steps.len(), 2);
        assert!(result.steps.last().unwrap().u_norm < 1e-12);
        // The form was evaluated (residual computed) without panicking —
        // this validates the driver integration with NonlinearForm.
    }
}
