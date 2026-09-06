//! SurfaceCurrent — port of MFEM `SurfaceCurrent` from `tesla_solver.hpp/cpp`.
//!
//! C++ reference:
//! ```cpp
//! SurfaceCurrent(ParFiniteElementSpace &H1FESpace, ParDiscreteGradOperator &Grad,
//!                Array<int> &kbcs, Array<int> &vbcs, Vector &vbcv);
//! void ComputeSurfaceCurrent(ParGridFunction &k);
//! ```
//!
//! ## Algorithm (C++ MFEM)
//! 1. Build `s0_` = ParBilinearForm(H1, DiffusionIntegrator) over all boundaries
//! 2. `ess_bdr_tdofs` = H1.GetEssentialTrueDofs(vbcs → Dirichlet)
//! 3. `psi_ = 0`; for each `vbcs[i]`: ProjectBdrCoefficient(Constant(vbcv[i]))
//! 4. s0_.FormLinearSystem(ess_bdr_tdofs, psi_, rhs_, S0_, Psi_, RHS_)
//! 5. PCG+AMG solve: pcg_(S0_) → Psi_
//! 6. s0_.RecoverFEMSolution(Psi_, rhs_, psi_)
//! 7. grad_.Mult(*psi_, k)  // k = ∇ψ
//! 8. k.ProjectBdrCoefficientTangent(Zero, non_k_bdr_)  // zero on non-k boundaries
//!
//! ## Rust mapping (requires integration with fem-parallel)
//! - `ParBilinearForm` → TBD (mixed assembler or new par_bilinear_form module)
//! - `ParDiscreteGradOperator` → `ParDiscreteLinearOperator::gradient`
//! - `ProjectBdrCoefficient` → `GridFunction::project_bdr_coefficient`
//! - `ProjectBdrCoefficientTangent` → TBD (postproc/grid_function.rs has tangent projection)
//! - `PCG+AMG` → `par_solve_pcg_amg` (par_amg.rs)

/// Surface current boundary condition configuration.
#[derive(Debug, Clone)]
pub struct SurfaceCurrentConfig {
    /// Boundary attributes where surface current is applied (1-indexed)
    pub kbcs: Vec<i32>,
    /// Boundary attributes where voltage is applied (1-indexed)
    pub vbcs: Vec<i32>,
    /// Voltage values corresponding to `vbcs`
    pub vbcv: Vec<f64>,
}

impl SurfaceCurrentConfig {
    pub fn new(kbcs: Vec<i32>, vbcs: Vec<i32>, vbcv: Vec<f64>) -> Self {
        Self { kbcs, vbcs, vbcv }
    }

    /// Validate: kbcs, vbcs, vbcv must have compatible lengths.
    pub fn validate(&self) -> Result<(), String> {
        if self.vbcs.len() != self.vbcv.len() {
            return Err(format!(
                "vbcs ({}) and vbcv ({}) length mismatch",
                self.vbcs.len(),
                self.vbcv.len()
            ));
        }
        if self.kbcs.is_empty() {
            return Err("kbcs must be non-empty for surface current BC".into());
        }
        Ok(())
    }

    /// Build the `non_k_bdr` marker array (C++ `non_k_bdr_`).
    ///
    /// Returns a `Vec<bool>` where `true` means the boundary attribute is
    /// NOT a surface current boundary (i.e., tangential component forced to zero).
    pub fn non_k_boundary_marker(&self, n_bdr_tags: usize) -> Vec<bool> {
        let mut marker = vec![true; n_bdr_tags];
        for &attr in &self.kbcs {
            if attr > 0 && attr as usize <= n_bdr_tags {
                marker[attr as usize - 1] = false;
            }
        }
        marker
    }

    /// Essential boundary marker: all boundaries except kbcs get Dirichlet.
    pub fn essential_boundary_marker(&self, n_bdr_tags: usize) -> Vec<bool> {
        let mut marker = vec![true; n_bdr_tags];
        for &attr in &self.kbcs {
            if attr > 0 && attr as usize <= n_bdr_tags {
                marker[attr as usize - 1] = false; // kbcs are NOT essential
            }
        }
        marker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        let cfg = SurfaceCurrentConfig::new(vec![1], vec![2], vec![1.0]);
        assert!(cfg.validate().is_ok());

        let bad = SurfaceCurrentConfig::new(vec![1], vec![2, 3], vec![1.0]);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn non_k_marker() {
        let cfg = SurfaceCurrentConfig::new(vec![1, 3], vec![2], vec![0.5]);
        let marker = cfg.non_k_boundary_marker(4);
        assert_eq!(marker, vec![false, true, false, true]);
    }

    #[test]
    fn ess_marker() {
        let cfg = SurfaceCurrentConfig::new(vec![2], vec![1], vec![1.0]);
        let marker = cfg.essential_boundary_marker(3);
        assert_eq!(marker, vec![true, false, true]);
    }
}
