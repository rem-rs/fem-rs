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
use crate::postproc::flux_recovery::{zz_estimator_mfem, zz_estimator_mfem_nc, FluxRecovery};

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
        dof_constraints: Option<&[HangingNodeConstraint]>,
    ) {
        // ── 1. Error estimation ────────────────────────────────────────────
        // On NC (non-conforming) meshes the flux space carries hanging-node
        // constraints; MFEM's SumFluxAndCount propagates them, so use the NC
        // variant (matches ex15: EnsureNCMesh → GeneralRefinement → NC path).
        // For P2 spaces the flux-space constraints must be the DOF-level P2
        // constraints (vertex-view ids), NOT the mesh-level P1 constraints
        // (physical node ids) — the latter index the averaged flux array with
        // physical ids and corrupt the estimator on multi-level NC meshes.
        let constraints = dof_constraints.unwrap_or_else(|| nc_state.constraints());
        let indicators = if self.use_kelly {
            kelly_estimator(gf)
        } else {
            zz_estimator_mfem_nc(gf, integrator, constraints)
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
/// Coarsens element groups (parents whose 4 children are all leaves) whose
/// children's aggregated error (sum, matching C++ default `op=1`) is below
/// the threshold.  Mirrors MFEM `Mesh::DerefineByError` +
/// `NCMesh::GetDerefinementTable` + `Derefine`.
pub struct ThresholdDerefiner {
    threshold: f64,
}

impl ThresholdDerefiner {
    pub fn new() -> Self { ThresholdDerefiner { threshold: 0.0 } }

    /// Elements whose children's aggregate error falls below `thresh` are coarsened.
    pub fn set_threshold(&mut self, thresh: f64) { self.threshold = thresh; }

    /// Apply selective derefinement using `refiner.eta` (errors on the current
    /// mesh).  On return, `*mesh` and `refiner.constraints` are updated.
    ///
    /// Returns `true` if at least one group was coarsened (MFEM
    /// `ThresholdDerefiner::ApplyImpl` returns `CONTINUE + DEREFINED`).
    pub fn apply(
        &mut self,
        mesh: &mut Mesh<2>,
        nc_state: &mut dyn NcState2D,
        refiner: &mut ThresholdRefiner,
    ) -> bool {
        if self.threshold <= 0.0 || refiner.eta.is_empty() { return false; }

        // MFEM Mesh::NonconformingDerefinement: for each derefinement-table
        // group, aggregate the child errors (op=1: sum) and coarsen the group
        // when the aggregate is below the threshold.  A group is only
        // coarsened when ALL its children are leaves of the current mesh.
        let groups = nc_state.deref_groups();
        if groups.is_empty() { return false; }

        let mut to_derefine: Vec<usize> = Vec::new();
        for &g in &groups {
            let children = nc_state.deref_group_children(g);
            if children.iter().any(|&c| c as usize >= refiner.eta.len()) { continue; }
            let agg: f64 = children.iter().map(|&c| refiner.eta[c as usize]).sum();
            if agg < self.threshold {
                to_derefine.push(g);
            }
        }
        if to_derefine.is_empty() { return false; }

        let Some(new_mesh) = nc_state.derefine_groups(mesh, &to_derefine) else {
            return false;
        };
        *mesh = new_mesh;
        refiner.constraints = nc_state.constraints().to_vec();
        true
    }
}
