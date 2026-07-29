//! Threshold-based AMR refiner and derefiner.
//!
//! Mirrors MFEM `ThresholdRefiner` + `ThresholdDerefiner`:
//!
//! ```rust,ignore
//! let mut refiner = ThresholdRefiner::new(false); // false = ZZ, true = Kelly
//! refiner.set_local_error_goal(0.005);
//! refiner.set_nc_limit(3);
//!
//! refiner.apply(&mut mesh, &mut nc_state, &gf, &integrator);
//! if refiner.stop() { break; }
//!
//! let mut derefiner = ThresholdDerefiner::new();
//! derefiner.set_threshold(0.15 * 0.005);
//! derefiner.apply(&mut mesh, &mut nc_state, &refiner);
//! ```

use fem_core::ElemId;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_mesh::amr::{HangingNodeConstraint, NcState2D};
use fem_space::fe_space::FESpace;

use crate::postproc::grid_function::GridFunction;
use crate::postproc::error_estimate::{threshold_mark, kelly_estimator};
use crate::postproc::flux_recovery::{zz_estimator_mfem, FluxRecovery};

/// Threshold-based AMR refiner — MFEM `ThresholdRefiner` equivalent.
///
/// Manages error estimation, threshold marking, and NC refinement.
/// The last-computed error indicators and marked-element list are exposed
/// for the `ThresholdDerefiner`.
pub struct ThresholdRefiner {
    local_err_goal: f64,
    nc_limit: u32,
    use_kelly: bool,
    /// Number of elements refined in the last `apply` call; `0` → `stop() == true`.
    last_marked_count: usize,
    /// Per-element error from the last `apply` call.
    pub eta: Vec<f64>,
    /// Elements marked in the last `apply`, in ascending order.
    pub last_marked: Vec<ElemId>,
    /// Hanging-node constraints from the last NC refinement.
    pub constraints: Vec<HangingNodeConstraint>,
}

impl ThresholdRefiner {
    /// `use_kelly`: `false` → ZZ estimator, `true` → Kelly estimator.
    pub fn new(use_kelly: bool) -> Self {
        ThresholdRefiner {
            local_err_goal: 0.0,
            nc_limit: 0,
            use_kelly,
            last_marked_count: 0,
            eta: Vec::new(),
            last_marked: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Set the local-error threshold: elements with `η > goal` are refined.
    pub fn set_local_error_goal(&mut self, goal: f64) { self.local_err_goal = goal; }

    /// Maximum non-conforming refinement level difference (0 = no limit).
    pub fn set_nc_limit(&mut self, limit: u32) { self.nc_limit = limit; }

    /// Reset internal state (call at the start of each time step).
    pub fn reset(&mut self) {
        self.last_marked_count = 0;
        self.eta.clear();
        self.last_marked.clear();
    }

    /// Whether the last `apply` refined no elements.
    pub fn stop(&self) -> bool { self.last_marked_count == 0 }

    /// Run the error estimator, mark elements, and apply NC refinement.
    ///
    /// On return, `self.eta` contains the per-element error, `self.constraints`
    /// the updated hanging-node constraints, and `*mesh` is the refined mesh.
    pub fn apply<M: MeshTopology, S: FESpace<Mesh = M>, F: FluxRecovery>(
        &mut self,
        mesh: &mut Mesh<2>,
        nc_state: &mut dyn NcState2D,
        gf: &GridFunction<'_, S>,
        integrator: &F,
    ) {
        // ── 1. Error estimation ────────────────────────────────────────────
        let indicators = if self.use_kelly {
            kelly_estimator(gf)
        } else {
            zz_estimator_mfem(gf, integrator)
        };
        self.eta = indicators.eta;

        // ── 2. Threshold marking ───────────────────────────────────────────
        let marked = threshold_mark(&self.eta, self.local_err_goal);

        // ── 3. NC refinement ───────────────────────────────────────────────
        if marked.is_empty() {
            self.last_marked_count = 0;
            self.last_marked.clear();
            return;
        }

        let (new_mesh, constraints, _midpoint_map) =
            nc_state.refine(mesh, &marked, self.nc_limit);
        *mesh = new_mesh;
        self.constraints = constraints;
        self.last_marked_count = marked.len();
        self.last_marked = marked;
    }
}

/// Threshold-based derefiner — MFEM `ThresholdDerefiner` equivalent.
///
/// Coarsens element groups whose children's aggregated error (sum, matching
/// C++ default `op=1`) is below the threshold.
pub struct ThresholdDerefiner {
    threshold: f64,
}

impl ThresholdDerefiner {
    pub fn new() -> Self { ThresholdDerefiner { threshold: 0.0 } }

    /// Elements whose children's aggregate error falls below `thresh` are coarsened.
    pub fn set_threshold(&mut self, thresh: f64) { self.threshold = thresh; }

    /// Apply selective derefinement using `refiner.eta` and `refiner.last_marked`.
    ///
    /// The refined mesh has `old_n_elems + 3 × last_marked.len()` elements.
    /// For each marked old index `e`, the 4 children start at
    /// `e + 3 × marked_before(e)` in the eta array.
    ///
    /// On return, `*mesh` and `refiner.constraints` are updated.
    pub fn apply(
        &mut self,
        mesh: &mut Mesh<2>,
        nc_state: &mut dyn NcState2D,
        refiner: &mut ThresholdRefiner,
    ) {
        if !nc_state.can_derefine() || refiner.eta.is_empty() || self.threshold <= 0.0 {
            return;
        }
        // Restore the pre-refinement mesh.
        let Some((old_mesh, _old_constraints)) = nc_state.derefine_last() else { return };
        // Compute refined_before for each marked old element.
        let mut keep_refined: Vec<ElemId> = Vec::new();
        let mut refined_before: usize = 0;
        for &old_e in &refiner.last_marked {
            let child_start = old_e as usize + 3 * refined_before;
            refined_before += 1;
            // Aggregate child errors (sum, matching C++ default op=1).
            let child_sum: f64 = refiner.eta[child_start..child_start + 4].iter().sum();
            if child_sum >= self.threshold {
                keep_refined.push(old_e);
            }
        }
        let n_coarsened = refiner.last_marked.len() - keep_refined.len();
        if n_coarsened > 0 {
            // Re-refine with the keep set (partial coarsening).
            let (new_mesh, constraints, _) = nc_state.refine(&old_mesh, &keep_refined, 0);
            *mesh = new_mesh;
            refiner.constraints = constraints;
        } else {
            // No elements to coarsen: restore the full refinement.
            let (restored_mesh, constraints, _) =
                nc_state.refine(&old_mesh, &refiner.last_marked, 0);
            *mesh = restored_mesh;
            refiner.constraints = constraints;
        }
    }
}
