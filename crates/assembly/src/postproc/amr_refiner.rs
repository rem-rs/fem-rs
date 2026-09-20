//! Threshold-based AMR refiner and derefiner.
//!
//! Mirrors MFEM `ThresholdRefiner` + `ThresholdDerefiner`
//! (`mesh/mesh_operators.{hpp,cpp}`), including the full marking-threshold
//! parameter family (D404/D497):
//!
//! ```rust,ignore
//! let mut refiner = ThresholdRefiner::new(false); // false = ZZ, true = Kelly
//! refiner.set_total_error_fraction(0.0);          // ex15: purely local threshold
//! refiner.set_local_error_goal(0.005);
//! refiner.set_nc_limit(3);
//!
//! refiner.apply(&mut mesh, &mut nc_state, &gf, &integrator, None);
//! if refiner.stop() { break; }
//!
//! let mut derefiner = ThresholdDerefiner::new();
//! derefiner.set_threshold(0.15 * 0.005);
//! derefiner.apply(&mut mesh, &mut nc_state, &refiner);
//! ```
//!
//! # Marking threshold (MFEM `ThresholdRefiner::MarkWithoutRefining`)
//!
//! ```text
//! total_err = (Σ_i η_i^p)^(1/p)               (p = total_norm_p; p = ∞ → max η_i)
//! threshold = max(total_err · total_fraction · N^(−1/p), local_err_goal)   p < ∞
//! threshold = max(total_err · total_fraction,           local_err_goal)   p = ∞
//! ```
//! and every element with `η_i > threshold` is marked (`threshold_mark`).
//! `total_norm_p`/`total_fraction`/`local_err_goal` default to ∞/0.5/0 exactly
//! as in MFEM's constructor, so the default rule is `η > 0.5·‖η‖_∞`
//! (`total_norm_p = 2` gives the RMS rule `η > 0.5·RMS(η)` — the same rule as
//! [`crate::postproc::error_estimate::ElementIndicators::rms_mark`]).

use fem_core::ElemId;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_mesh::amr::{HangingNodeConstraint, NcState2D, QuadRefineDir};
use fem_space::fe_space::FESpace;

use crate::postproc::grid_function::GridFunction;
use crate::postproc::error_estimate::{
    threshold_mark, AnisotropicErrorEstimator, ElementIndicators, kelly_estimator,
};
use crate::postproc::flux_recovery::{zz_estimator_mfem_nc, FluxRecovery};

/// MFEM `Vector::Normlp` analogue: the discrete p-norm `(Σ|v_i|^p)^(1/p)`.
///
/// Mirrors MFEM's special branches `p == 1` → `Norml1`, `p == 2` → `Norml2`
/// (so the p = 2 total error is bit-identical to
/// [`crate::postproc::error_estimate::ElementIndicators::total_error`]), the
/// direct `pow` sum for other finite `p` (MFEM scales on the fly to avoid
/// overflow; indicator magnitudes are O(1) so the unscaled sum is safe), and
/// the max for `p = ∞`.
fn p_norm(v: &[f64], p: f64) -> f64 {
    debug_assert!(p > 0.0, "p_norm: MFEM Vector::Normlp requires p > 0");
    if p == 1.0 {
        v.iter().map(|x| x.abs()).sum()
    } else if p == 2.0 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    } else if p.is_finite() {
        v.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
    } else {
        v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
    }
}

/// Threshold-based AMR refiner — MFEM `ThresholdRefiner` equivalent.
///
/// Manages error estimation, threshold marking, and NC refinement.
/// The last-computed error indicators and marked-element list are exposed
/// for the `ThresholdDerefiner`.
pub struct ThresholdRefiner {
    /// MFEM `total_norm_p`: exponent of the discrete p-norm used for the
    /// total error (default ∞).
    total_norm_p: f64,
    /// MFEM `total_err_goal`: stop when `total_err <= total_err_goal`.
    total_err_goal: f64,
    /// MFEM `total_fraction`: fraction of the total error in the threshold
    /// (default 0.5; 0 ⇒ purely local-threshold marking).
    total_fraction: f64,
    /// MFEM `local_err_goal`: **floor** on the computed threshold — NOT the
    /// threshold itself (pre-D497 this field *was* the whole threshold; the
    /// MFEM `max(·, local_err_goal)` composition is now in place).
    local_err_goal: f64,
    /// MFEM `max_elements`: stop when the mesh has at least this many elements.
    max_elements: u64,
    /// Threshold used in the last `apply`/`apply_aniso` (MFEM `GetThreshold`).
    threshold: f64,
    nc_limit: u32,
    use_kelly: bool,
    /// Number of elements refined in the last `apply` call; `0` → `stop() == true`.
    last_marked_count: usize,
    /// MFEM `ThresholdRefiner::non_conforming` flag (`mesh/mesh_operators.hpp`):
    /// `1` after [`Self::prefer_nonconforming_refinement`], `-1` after
    /// [`Self::prefer_conforming_refinement`] (the constructor default, as in
    /// MFEM's constructor `non_conforming = -1`).  MFEM's
    /// `Mesh::GeneralRefinement` overrides the flag to *non-conforming* whenever
    /// the mesh already carries an `NCMesh` — ex15.cpp:142
    /// (`mesh.EnsureNCMesh(true)`) always arranges that, so the flag never
    /// diverts ex15.  [`Self::apply`] mirrors the same override (its refinement
    /// engine *is* the `NcState2D` NC machinery), so the flag is recorded for
    /// API parity and surfaced by [`Self::non_conforming`]; MFEM's conforming
    /// red-green branch (`Mesh::LocalRefinement`) is not ported.
    non_conforming: i32,
    /// Per-element error from the last `apply` call.
    pub eta: Vec<f64>,
    /// Elements marked in the last `apply`, in ascending order.
    pub last_marked: Vec<ElemId>,
    /// Hanging-node constraints from the last NC refinement.
    pub constraints: Vec<HangingNodeConstraint>,
}

impl ThresholdRefiner {
    /// `use_kelly`: `false` → ZZ estimator, `true` → Kelly estimator.
    ///
    /// Defaults mirror MFEM's constructor: `total_norm_p = ∞`,
    /// `total_err_goal = 0`, `total_fraction = 0.5`, `local_err_goal = 0`,
    /// `max_elements` unlimited, `nc_limit = 0` — i.e. the default rule marks
    /// `η > 0.5·‖η‖_∞`.
    pub fn new(use_kelly: bool) -> Self {
        ThresholdRefiner {
            total_norm_p: f64::INFINITY,
            total_err_goal: 0.0,
            total_fraction: 0.5,
            local_err_goal: 0.0,
            max_elements: u64::MAX,
            threshold: 0.0,
            nc_limit: 0,
            use_kelly,
            last_marked_count: 0,
            non_conforming: -1,
            eta: Vec::new(),
            last_marked: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// MFEM `SetTotalErrorNormP`: exponent p of the discrete p-norm used to
    /// compute the total error from the local element errors (`∞` = max-norm).
    pub fn set_total_error_norm_p(&mut self, norm_p: f64) { self.total_norm_p = norm_p; }

    /// MFEM `SetTotalErrorGoal`: stop when `total_err <= total_err_goal`.
    pub fn set_total_error_goal(&mut self, err_goal: f64) { self.total_err_goal = err_goal; }

    /// MFEM `SetTotalErrorFraction`: fraction of the total error in the
    /// threshold (default 0.5).  `0` ⇒ `total_err` is ignored and the marking
    /// is purely local (`threshold = local_err_goal`) — the MFEM ex15
    /// configuration.
    pub fn set_total_error_fraction(&mut self, fraction: f64) { self.total_fraction = fraction; }

    /// Set the local-error floor: MFEM `SetLocalErrorGoal`.  The marking
    /// threshold is `max(total_err·fraction·N^(−1/p), goal)`, so the goal acts
    /// as a *lower bound* on the threshold, never as a standalone rule.
    pub fn set_local_error_goal(&mut self, goal: f64) { self.local_err_goal = goal; }

    /// MFEM `SetMaxElements`: stop when the mesh has `>= max_elem` elements.
    pub fn set_max_elements(&mut self, max_elem: u64) { self.max_elements = max_elem; }

    /// Maximum non-conforming refinement level difference (0 = no limit).
    pub fn set_nc_limit(&mut self, limit: u32) { self.nc_limit = limit; }

    /// MFEM `ThresholdRefiner::PreferNonconformingRefinement` (`non_conforming
    /// = 1`): use nonconforming refinement, if possible.  See the field doc
    /// for the `GeneralRefinement` override that makes the NC engine win
    /// whenever an NC state is in play (ex15 always).
    pub fn prefer_nonconforming_refinement(&mut self) { self.non_conforming = 1; }

    /// MFEM `ThresholdRefiner::PreferConformingRefinement` (`non_conforming =
    /// -1`, the constructor default): use conforming refinement, if possible.
    /// ex15.cpp:235 calls this — with no effect there, because ex15.cpp:142
    /// `EnsureNCMesh` forces the NC path (see the field doc).
    pub fn prefer_conforming_refinement(&mut self) { self.non_conforming = -1; }

    /// The MFEM `non_conforming` preference flag: `-1` = prefer conforming
    /// (default), `1` = prefer non-conforming.
    pub fn non_conforming(&self) -> i32 { self.non_conforming }

    /// MFEM `GetThreshold`: the threshold used in the last `apply` call
    /// (`0` before the first call / on early-STOP paths).
    pub fn threshold(&self) -> f64 { self.threshold }

    /// Reset internal state (call at the start of each time step).
    pub fn reset(&mut self) {
        self.last_marked_count = 0;
        self.eta.clear();
        self.last_marked.clear();
    }

    /// Whether the last `apply` refined no elements (MFEM `Stop()`: true on
    /// any STOP criterion — `max_elements`, `total_err_goal`, or no marks).
    pub fn stop(&self) -> bool { self.last_marked_count == 0 }

    /// MFEM `ThresholdRefiner::MarkWithoutRefining` threshold computation:
    /// `max(total_err · fraction · N^(−1/p), local_err_goal)` for finite `p`,
    /// `max(total_err · fraction, local_err_goal)` for `p = ∞`.
    ///
    /// `ne` is the element count of the *current* mesh (MFEM `GetGlobalNE`).
    fn compute_threshold(&self, total_err: f64, ne: usize) -> f64 {
        if self.total_norm_p.is_infinite() {
            (total_err * self.total_fraction).max(self.local_err_goal)
        } else {
            (total_err * self.total_fraction
                * (ne as f64).powf(-1.0 / self.total_norm_p))
                .max(self.local_err_goal)
        }
    }

    /// MFEM `MarkWithoutRefining` prologue: the `max_elements` STOP and the
    /// `total_err_goal` STOP.  Returns `true` when a STOP criterion fired
    /// (caller must return without refining; `stop()` then reports true).
    fn stop_before_marking(&mut self, mesh: &Mesh<2>) -> bool {
        // MFEM sets `threshold = 0.0` at the top of MarkWithoutRefining.
        self.threshold = 0.0;
        // STOP: num_elements >= max_elements (no estimation is performed).
        if mesh.n_elements() as u64 >= self.max_elements {
            self.last_marked_count = 0;
            self.last_marked.clear();
            return true;
        }
        false
    }

    /// MFEM `MarkWithoutRefining` epilogue: `total_err_goal` STOP check +
    /// threshold + `η > threshold` marking.  Returns an empty list on STOP.
    fn mark_mfem(&mut self) -> Vec<u32> {
        let total_err = p_norm(&self.eta, self.total_norm_p);
        // STOP: total_err <= total_err_goal.
        if total_err <= self.total_err_goal {
            self.last_marked_count = 0;
            self.last_marked.clear();
            return Vec::new();
        }
        self.threshold = self.compute_threshold(total_err, self.eta.len());
        // MFEM: `local_err(el) > threshold` (strict).
        threshold_mark(&self.eta, self.threshold)
    }

    /// Run the error estimator, mark elements, and apply NC refinement.
    ///
    /// On return, `self.eta` contains the per-element error, `self.threshold`
    /// the marking threshold, `self.constraints` the updated hanging-node
    /// constraints, and `*mesh` is the refined mesh.
    pub fn apply<M: MeshTopology, S: FESpace<Mesh = M>, F: FluxRecovery>(
        &mut self,
        mesh: &mut Mesh<2>,
        nc_state: &mut dyn NcState2D,
        gf: &GridFunction<'_, S>,
        integrator: &F,
        dof_constraints: Option<&[HangingNodeConstraint]>,
    ) {
        if self.stop_before_marking(mesh) { return; }

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

        // ── 2. Threshold marking (MFEM MarkWithoutRefining) ────────────────
        let marked = self.mark_mfem();
        if marked.is_empty() {
            // MFEM ApplyImpl: `num_marked_elements == 0 → STOP` (covers both
            // the `total_err_goal` STOP set inside `mark_mfem` and the
            // natural no-element-above-threshold case).
            self.last_marked_count = 0;
            self.last_marked.clear();
            return;
        }

        // ── 3. NC refinement ───────────────────────────────────────────────
        let (new_mesh, constraints, _midpoint_map) =
            nc_state.refine(mesh, &marked, self.nc_limit);
        *mesh = new_mesh;
        self.constraints = constraints;
        self.last_marked_count = marked.len();
        self.last_marked = marked;
    }

    /// Anisotropic threshold refinement (MFEM `ThresholdRefiner` driven by an
    /// `AnisotropicErrorEstimator`, 2D Quad4 meshes): marked elements are cut
    /// along the dominant flux-error direction — flag bit `k` set ⇒
    /// `QuadRefineDir` X/Y, both bits ⇒ the 4-way split.  Marking uses the
    /// same MFEM threshold family as [`ThresholdRefiner::apply`]; in MFEM the
    /// aniso estimator only overrides each `Refinement`'s *type*
    /// (`MarkWithoutRefining`: `ref.SetType(aniso_flags[ref.index])`).
    ///
    /// The refinement goes through `refine_nonconforming_quad_aniso`, which
    /// returns the hanging-node constraints but does not extend the
    /// derefinement tree of a `NcState2D`; aniso splits followed by
    /// derefinement are therefore not supported (a fem-mesh port gap, not an
    /// estimator-interface one).
    pub fn apply_aniso(
        &mut self,
        mesh: &mut Mesh<2>,
        indicators: &ElementIndicators,
    ) {
        if self.stop_before_marking(mesh) { return; }
        self.eta = indicators.eta.clone();
        let marked: Vec<(ElemId, QuadRefineDir)> = self
            .mark_mfem()
            .into_iter()
            .map(|e| {
                let flags = indicators.get_anisotropic_flags();
                let f = flags.get(e as usize).copied().unwrap_or(0);
                let dir = match (f & 1 != 0, f & 2 != 0) {
                    (true, false) => QuadRefineDir::X,
                    (false, true) => QuadRefineDir::Y,
                    _ => QuadRefineDir::Both,
                };
                (e, dir)
            })
            .collect();
        if marked.is_empty() {
            self.last_marked_count = 0;
            self.last_marked.clear();
            return;
        }
        let (new_mesh, constraints) =
            fem_mesh::amr::refine_nonconforming_quad_aniso(mesh, &marked, None);
        *mesh = new_mesh;
        self.constraints = constraints;
        self.last_marked_count = marked.len();
        self.last_marked = marked.iter().map(|&(e, _)| e).collect();
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
    nc_limit: u32,
    /// MFEM `ThresholdDerefiner::op` — child-error aggregation
    /// (`Mesh::AggregateError`): `0` = min, `1` = sum (constructor default),
    /// `2` = max.
    op: i32,
}

impl ThresholdDerefiner {
    /// MFEM constructor defaults: `threshold = 0`, `nc_limit = 0`, `op = 1`.
    pub fn new() -> Self { ThresholdDerefiner { threshold: 0.0, nc_limit: 0, op: 1 } }

    /// Elements whose children's aggregate error falls below `thresh` are coarsened.
    pub fn set_threshold(&mut self, thresh: f64) { self.threshold = thresh; }

    /// MFEM `ThresholdDerefiner::SetOp`: aggregation of the children's errors
    /// in a derefinement group — `0` = min, `1` = sum (default), `2` = max
    /// (`Mesh::AggregateError`).
    pub fn set_op(&mut self, op: i32) {
        assert!(matches!(op, 0 | 1 | 2), "ThresholdDerefiner::set_op: invalid op {op} (MFEM AggregateError: 0=min, 1=sum, 2=max)");
        self.op = op;
    }

    /// `Mesh::AggregateError(elem_error, fine, nfine, op)` over the group's
    /// children (all of which must be valid `eta` indices — the caller
    /// filters otherwise).
    fn aggregate(&self, children: &[ElemId], eta: &[f64]) -> f64 {
        let mut it = children.iter().map(|&c| eta[c as usize]);
        let Some(mut error) = it.next() else { return 0.0 };
        for err_fine in it {
            match self.op {
                0 => error = error.min(err_fine),
                1 => error += err_fine,
                _ => error = error.max(err_fine),
            }
        }
        error
    }

    /// Maximum NC level difference between adjacent elements after
    /// derefinement (0 = unlimited).  MFEM `ThresholdDerefiner::SetNCLimit`;
    /// ex15.cpp calls `derefiner.SetNCLimit(nc_limit)` (3).
    pub fn set_nc_limit(&mut self, limit: u32) { self.nc_limit = limit; }

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
        let mut nc_filtered = 0usize;
        let mut agg_below = 0usize;
        for &g in &groups {
            // MFEM CheckDerefinementNCLevel: skip groups whose coarsening
            // would exceed the max NC level between adjacent elements.
            if self.nc_limit > 0 && !nc_state.deref_group_nc_ok(g, self.nc_limit, mesh) {
                nc_filtered += 1;
                continue;
            }
            let children = nc_state.deref_group_children(g);
            if children.iter().any(|&c| c as usize >= refiner.eta.len()) { continue; }
            let agg = self.aggregate(&children, &refiner.eta);
            if agg < self.threshold {
                to_derefine.push(g);
                agg_below += 1;
            }
        }
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            let mut aggs: Vec<f64> = Vec::new();
            for &g in &groups {
                if self.nc_limit > 0 && !nc_state.deref_group_nc_ok(g, self.nc_limit, mesh) { continue; }
                let children = nc_state.deref_group_children(g);
                if children.iter().any(|&c| c as usize >= refiner.eta.len()) { continue; }
                aggs.push(self.aggregate(&children, &refiner.eta));
            }
            aggs.sort_by(|a, b| b.partial_cmp(a).unwrap());
            eprintln!("DBG derefiner: groups={} nc_filtered={nc_filtered} agg_below={agg_below} chosen={} top5-agg={:?}",
                groups.len(), to_derefine.len(),
                aggs.iter().take(5).map(|a| format!("{a:.6e}")).collect::<Vec<_>>());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::DiffusionIntegrator;
    use fem_mesh::amr::NCStateQuad;
    use fem_space::H1Space;

    type M2 = Mesh<2>;

    /// One derefinable quad group with child errors 1, 2, 3, 4.  The fixture
    /// mesh is the single-element `unit_square_quad(1)`; splitting element 0
    /// yields 4 elements (one tree group with children [0, 1, 2, 3]), and
    /// coarsening returns to 1.
    fn deref_fixture() -> (M2, NCStateQuad, usize) {
        let mesh = M2::unit_square_quad(1);
        let mut nc_state = NCStateQuad::new();
        let (mesh, _cons, _) = nc_state.refine(&mesh, &[0], 0);
        let ne = mesh.n_elements();
        assert_eq!(ne, 4);
        (mesh, nc_state, ne)
    }

    /// eta values: 1, 2, 3, 4 on the group's children (in the order the tree
    /// reports them).
    fn synthetic_eta(nc_state: &NCStateQuad, ne: usize) -> Vec<f64> {
        let mut eta = vec![100.0_f64; ne];
        let children = nc_state.deref_group_children(nc_state.deref_groups()[0]);
        for (k, &c) in children.iter().enumerate() {
            eta[c as usize] = (k + 1) as f64;
        }
        eta
    }

    /// D500 — `ThresholdDerefiner::SetOp`: op 0/1/2 aggregate a group as
    /// min/sum/max exactly like MFEM `Mesh::AggregateError`.  With children
    /// (1, 2, 3, 4): min = 1, sum = 10, max = 4, so a threshold of 2 fires
    /// only for min, 4.5 only for max, and 10.5 for sum.
    #[test]
    fn d500_derefiner_set_op_selects_aggregation() {
        // (a) op = 0 (min): 1 < 2 ⇒ deref (back to the single parent quad).
        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(2.0);
        derefiner.set_op(0);
        assert!(derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), 1, "min(1,2,3,4)=1 < 2 must coarsen");

        // (b) op = 1 (sum): 10 ≥ 2 ⇒ no deref.
        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(2.0);
        derefiner.set_op(1);
        assert!(!derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), ne, "sum=10 must not coarsen at 2");

        // (c) op = 2 (max): 4 ≥ 2 stays; 4 < 5 coarsens.
        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(2.0);
        derefiner.set_op(2);
        assert!(!derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), ne, "max=4 must not coarsen at 2");

        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(5.0);
        derefiner.set_op(2);
        assert!(derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), 1, "max=4 < 5 must coarsen");

        // (d) sum fires at 10.5 (completes the sum branch through set_op).
        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(10.5);
        derefiner.set_op(1);
        assert!(derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), 1, "sum=10 < 10.5 must coarsen");
    }

    /// D500 — the default op is MFEM's constructor default `op = 1` (sum): at
    /// a threshold between max (4) and sum (10) the default must *not* fire,
    /// i.e. the pre-D500 hard-coded sum behaviour is bit-identical.
    #[test]
    fn d500_derefiner_default_op_is_sum() {
        let (mut mesh, mut nc_state, ne) = deref_fixture();
        let mut refiner = ThresholdRefiner::new(false);
        refiner.eta = synthetic_eta(&nc_state, ne);
        let mut derefiner = ThresholdDerefiner::new();
        derefiner.set_threshold(4.5);
        assert!(!derefiner.apply(&mut mesh, &mut nc_state, &mut refiner));
        assert_eq!(mesh.n_elements(), ne, "default op = sum (10) must stay above 4.5");
    }

    /// D500 — the `non_conforming` flag mirrors MFEM: constructor default
    /// `-1`, `PreferNonconformingRefinement` → 1, `PreferConformingRefinement`
    /// → −1 (ex15.cpp:235), and — per MFEM `GeneralRefinement`'s
    /// `if (ncmesh) nonconforming = 1` override — `apply` keeps refining
    /// through the NC engine even with the conforming preference set.
    #[test]
    fn d500_refiner_non_conforming_flag_parity() {
        let mesh = M2::unit_square_tri(2);
        let space = H1Space::new(mesh.clone(), 1);
        let d = space.interpolate(&|x: &[f64]| (std::f64::consts::PI * x[0]).sin());
        let gf = GridFunction::new(&space, d.as_slice().to_vec());
        let int = DiffusionIntegrator { kappa: 1.0 };

        let mut r = ThresholdRefiner::new(false);
        assert_eq!(r.non_conforming(), -1, "constructor default = -1");
        r.prefer_nonconforming_refinement();
        assert_eq!(r.non_conforming(), 1);
        r.prefer_conforming_refinement();
        assert_eq!(r.non_conforming(), -1, "ex15.cpp:235 preference");

        // The conforming preference does not divert the NC engine (MFEM
        // GeneralRefinement override for meshes with NC state): the mesh
        // grows through `nc_state.refine` and the derefinement tree becomes
        // available (rollback = the NC engine's witness).
        let ne0 = mesh.n_elements();
        let mut m = mesh;
        let mut nc = fem_mesh::amr::NCState::new();
        r.set_total_error_fraction(0.0);
        r.set_local_error_goal(1.0e-3);
        r.apply(&mut m, &mut nc, &gf, &int, Some(&[]));
        assert!(!r.stop(), "marks expected on the MMS field");
        assert!(m.n_elements() > ne0, "apply must refine through the NC engine");
        assert!(nc.can_derefine(), "apply must drive the NC tree (rollback available)");
    }
}
