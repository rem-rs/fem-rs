//! # Forest performance layer (Phase 7)
//!
//! Caching, incremental update, batch grouping, and benchmarking for AMR
//! mesh operations.  Reduces redundant computation during the adapt→solve
//! cycle by tracking which elements have changed.

use fem_core::ElemId;
use crate::Mesh;

// ═════════════════════════════════════════════════════════════════════════════
//  Element cache (Task 7.1 Step 1)
// ═════════════════════════════════════════════════════════════════════════════

/// Per-element cached geometric / matrix data for matrix-free operators.
///
/// Populated once after a mesh change, then reused across multiple
/// operator applications until the next change.
#[derive(Debug, Clone)]
pub struct ElementCache2D {
    /// Jacobian determinant at element centroid (|J|).
    pub det_j: Vec<f64>,
    /// Inverse Jacobian transpose, flattened `[jit_xx, jit_xy, jit_yx, jit_yy]`.
    pub jit: Vec<[f64; 4]>,
    /// Element area.
    pub area: Vec<f64>,
    /// Whether each entry is valid.
    pub valid: Vec<bool>,
    /// Number of elements the cache was built for.
    pub n_elems: usize,
}

impl ElementCache2D {
    /// Build a new cache for the given mesh.  Computes all entries.
    pub fn build(mesh: &Mesh<2>) -> Self {
        let n_elems = mesh.n_elems();
        let mut det_j = Vec::with_capacity(n_elems);
        let mut jit = Vec::with_capacity(n_elems);
        let mut area = Vec::with_capacity(n_elems);
        let mut valid = Vec::with_capacity(n_elems);

        let is_quad = mesh.element_type_at(0) == crate::element_type::ElementType::Quad4;

        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let c = |i: usize| mesh.coords_of(ns[i]);

            let (det, jit_arr, a) = if is_quad {
                let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
                let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
                let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
                let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
                let d = j00 * j11 - j01 * j10;
                let id = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
                let a = 0.5 * ((c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1])
                              - (c(1)[0]*c(0)[1] + c(2)[0]*c(1)[1] + c(3)[0]*c(2)[1] + c(0)[0]*c(3)[1])).abs();
                (d, [j11*id, -j10*id, -j01*id, j00*id], a)
            } else {
                let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
                let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
                let d = j00 * j11 - j01 * j10;
                let id = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
                let a = 0.5 * d.abs();
                (d, [j11*id, -j10*id, -j01*id, j00*id], a)
            };

            det_j.push(det);
            jit.push(jit_arr);
            area.push(a);
            valid.push(true);
        }

        ElementCache2D { det_j, jit, area, valid, n_elems }
    }

    /// Invalidate cache entries for elements that have been replaced.
    /// Call this after refinement or coarsening changes the mesh.
    pub fn invalidate(&mut self, changed_elems: &[ElemId]) {
        for &e in changed_elems {
            if (e as usize) < self.valid.len() {
                self.valid[e as usize] = false;
            }
        }
    }

    /// Rebuild invalidated entries from the mesh (incremental update).
    pub fn rebuild_invalid(&mut self, mesh: &Mesh<2>) {
        let is_quad = mesh.element_type_at(0) == crate::element_type::ElementType::Quad4;
        for e in 0..self.n_elems as ElemId {
            if !self.valid[e as usize] {
                let ns = mesh.elem_nodes(e);
                let c = |i: usize| mesh.coords_of(ns[i]);

                let (det, jit_arr, a) = if is_quad {
                    let j00 = 0.25 * (-c(0)[0] + c(1)[0] + c(2)[0] - c(3)[0]);
                    let j01 = 0.25 * (-c(0)[0] - c(1)[0] + c(2)[0] + c(3)[0]);
                    let j10 = 0.25 * (-c(0)[1] + c(1)[1] + c(2)[1] - c(3)[1]);
                    let j11 = 0.25 * (-c(0)[1] - c(1)[1] + c(2)[1] + c(3)[1]);
                    let d = j00 * j11 - j01 * j10;
                    let id = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
                    let a = 0.5 * ((c(0)[0]*c(1)[1] + c(1)[0]*c(2)[1] + c(2)[0]*c(3)[1] + c(3)[0]*c(0)[1])
                                  - (c(1)[0]*c(0)[1] + c(2)[0]*c(1)[1] + c(3)[0]*c(2)[1] + c(0)[0]*c(3)[1])).abs();
                    (d, [j11*id, -j10*id, -j01*id, j00*id], a)
                } else {
                    let j00 = c(1)[0] - c(0)[0]; let j01 = c(2)[0] - c(0)[0];
                    let j10 = c(1)[1] - c(0)[1]; let j11 = c(2)[1] - c(0)[1];
                    let d = j00 * j11 - j01 * j10;
                    let id = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
                    let a = 0.5 * d.abs();
                    (d, [j11*id, -j10*id, -j01*id, j00*id], a)
                };
                self.det_j[e as usize] = det;
                self.jit[e as usize] = jit_arr;
                self.area[e as usize] = a;
                self.valid[e as usize] = true;
            }
        }
    }

    /// Get cached inverse Jacobian transpose for element `e`.
    #[inline]
    pub fn get_jit(&self, e: ElemId) -> [f64; 4] {
        self.jit[e as usize]
    }

    /// Get cached Jacobian determinant for element `e`.
    #[inline]
    pub fn get_det_j(&self, e: ElemId) -> f64 {
        self.det_j[e as usize]
    }

    /// Get cached element area.
    #[inline]
    pub fn get_area(&self, e: ElemId) -> f64 {
        self.area[e as usize]
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Incremental update tracker (Task 7.1 Step 2)
// ═════════════════════════════════════════════════════════════════════════════

/// Tracks which elements have changed since the last computation,
/// supporting incremental updates for AMR cycles.
#[derive(Debug, Clone)]
pub struct IncrementalTracker {
    /// Dirty flags per element.
    pub dirty: Vec<bool>,
    /// Generation counter (incremented on each refinement/coarsening).
    pub generation: u64,
}

impl IncrementalTracker {
    /// Create a new tracker for `n_elems` elements.
    pub fn new(n_elems: usize) -> Self {
        Self {
            dirty: vec![true; n_elems],
            generation: 0,
        }
    }

    /// Mark a set of elements as dirty (changed by refinement/coarsening).
    pub fn mark_dirty(&mut self, elems: &[ElemId]) {
        for &e in elems {
            if (e as usize) < self.dirty.len() {
                self.dirty[e as usize] = true;
            }
        }
    }

    /// Mark all elements clean (computation done).
    pub fn mark_all_clean(&mut self) {
        self.dirty.iter_mut().for_each(|v| *v = false);
    }

    /// Get all dirty element IDs.
    pub fn dirty_elems(&self) -> Vec<ElemId> {
        self.dirty.iter().enumerate()
            .filter(|(_, &d)| d)
            .map(|(i, _)| i as ElemId)
            .collect()
    }

    /// Whether any elements are dirty.
    pub fn has_dirty(&self) -> bool {
        self.dirty.iter().any(|&d| d)
    }

    /// Resize the tracker (called after mesh changes element count).
    pub fn resize(&mut self, new_n_elems: usize) {
        self.dirty.resize(new_n_elems, true);
        self.generation += 1;
    }

    /// Mark children as dirty after a refinement operation.
    /// `parent_elem` is the original element that was split, `children`
    /// are the new child element IDs.
    pub fn mark_refined(&mut self, parent_elem: ElemId, children: &[ElemId]) {
        if (parent_elem as usize) < self.dirty.len() {
            self.dirty[parent_elem as usize] = false; // parent no longer exists
        }
        for &c in children {
            if (c as usize) < self.dirty.len() {
                self.dirty[c as usize] = true;
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  GPU-ready batch grouping (Task 7.1 Step 3)
// ═════════════════════════════════════════════════════════════════════════════

/// A batch of active elements grouped for GPU / vectorised processing.
#[derive(Debug, Clone)]
pub struct ElemBatch {
    /// Contiguous element IDs in this batch.
    pub elems: Vec<ElemId>,
    /// Batch index.
    pub batch_idx: usize,
}

/// Group active elements into fixed-size batches for GPU offloading.
///
/// Each batch contains up to `batch_size` contiguous elements (in memory
/// order), ensuring coalesced access to cached element data.
pub fn build_batches(n_active: usize, batch_size: usize) -> Vec<ElemBatch> {
    if n_active == 0 || batch_size == 0 {
        return Vec::new();
    }

    let n_batches = (n_active + batch_size - 1) / batch_size;
    let mut batches = Vec::with_capacity(n_batches);

    for b in 0..n_batches {
        let start = b * batch_size;
        let end = (start + batch_size).min(n_active);
        let elems: Vec<ElemId> = (start..end).map(|i| i as ElemId).collect();
        batches.push(ElemBatch { elems, batch_idx: b });
    }

    batches
}

/// Filter active elements by a criterion.  Returns indices of elements
/// that pass the filter (e.g., elements above an error threshold).
pub fn filter_active(
    eta: &[f64],
    threshold: f64,
) -> Vec<ElemId> {
    eta.iter().enumerate()
        .filter(|(_, &v)| v > threshold)
        .map(|(i, _)| i as ElemId)
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  AMR cycle benchmark (Task 7.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Timing statistics for each phase of an AMR cycle.
#[derive(Debug, Clone, Default)]
pub struct AmrCycleTiming {
    pub error_estimation_ms: f64,
    pub marking_ms: f64,
    pub refinement_ms: f64,
    pub total_ms: f64,
    pub n_elems_before: usize,
    pub n_elems_after: usize,
}

/// Run a benchmark of one full AMR cycle and return timing.
///
/// Measures error estimation, marking, and refinement phases separately.
/// This provides a baseline for optimisation (Task 7.2 Step 3).
pub fn benchmark_amr_cycle<F1, F2, F3>(
    mesh: &Mesh<2>,
    u: &[f64],
    estimator_fn: F1,
    marker_fn: F2,
    refine_fn: F3,
) -> AmrCycleTiming
where
    F1: Fn(&Mesh<2>, &[f64]) -> Vec<f64>,
    F2: Fn(&[f64]) -> Vec<ElemId>,
    F3: Fn(&Mesh<2>, &[ElemId]) -> (Mesh<2>, Vec<crate::amr::HangingNodeConstraint>),
{
    let n_before = mesh.n_elems();
    let t0 = std::time::Instant::now();

    let t1 = std::time::Instant::now();
    let eta = estimator_fn(mesh, u);
    let t_est = t1.elapsed();

    let t2 = std::time::Instant::now();
    let marked = marker_fn(&eta);
    let t_mark = t2.elapsed();

    let t3 = std::time::Instant::now();
    let (new_mesh, _constraints) = refine_fn(mesh, &marked);
    let t_ref = t3.elapsed();

    let total = t0.elapsed();

    AmrCycleTiming {
        error_estimation_ms: t_est.as_secs_f64() * 1000.0,
        marking_ms: t_mark.as_secs_f64() * 1000.0,
        refinement_ms: t_ref.as_secs_f64() * 1000.0,
        total_ms: total.as_secs_f64() * 1000.0,
        n_elems_before: n_before,
        n_elems_after: new_mesh.n_elems(),
    }
}

/// Run a multi-cycle AMR benchmark and report aggregate statistics.
pub fn benchmark_amr_multicycle<F1, F2, F3>(
    initial_mesh: &Mesh<2>,
    u: &[f64],
    n_cycles: usize,
    estimator_fn: &F1,
    marker_fn: &F2,
    refine_fn: &F3,
) -> Vec<AmrCycleTiming>
where
    F1: Fn(&Mesh<2>, &[f64]) -> Vec<f64>,
    F2: Fn(&[f64]) -> Vec<ElemId>,
    F3: Fn(&Mesh<2>, &[ElemId]) -> (Mesh<2>, Vec<crate::amr::HangingNodeConstraint>),
{
    let mut mesh = initial_mesh.clone();
    let mut u = u.to_vec();
    let mut results = Vec::with_capacity(n_cycles);

    for _cycle in 0..n_cycles {
        let timing = benchmark_amr_cycle(&mesh, &u, estimator_fn, marker_fn, refine_fn);
        results.push(timing);

        // Update mesh and solution for next cycle
        let (new_mesh, _) = refine_fn(&mesh, &[]); // dummy refinement
        mesh = new_mesh;
        // Prolongate solution (simplified: copy existing values)
        let new_n = mesh.n_nodes();
        u.resize(new_n, 0.0);
    }

    results
}

/// Compute the total CPU time and element throughput for a benchmark run.
pub fn benchmark_summary(timings: &[AmrCycleTiming]) -> (f64, f64) {
    let total_time: f64 = timings.iter().map(|t| t.total_ms).sum();
    let total_elems: usize = timings.iter().map(|t| t.n_elems_after).sum();
    let throughput = if total_time > 0.0 {
        total_elems as f64 / (total_time / 1000.0) // elems/second
    } else {
        0.0
    };
    (total_time, throughput)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;
    use fem_core::NodeId;

    // ── Element cache tests ───────────────────────────────────────────────

    #[test]
    fn cache_build_for_tri3_mesh() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let cache = ElementCache2D::build(&mesh);
        assert_eq!(cache.n_elems, mesh.n_elems());
        assert!(cache.valid.iter().all(|&v| v));
        // All Jacobian determinants should be positive for a valid mesh
        assert!(cache.det_j.iter().all(|&d| d > 0.0));
        // All areas should be positive
        assert!(cache.area.iter().all(|&a| a > 0.0));
    }

    #[test]
    fn cache_build_for_quad4_mesh() {
        let mesh = Mesh::<2>::unit_square_quad(4);
        let cache = ElementCache2D::build(&mesh);
        assert_eq!(cache.n_elems, mesh.n_elems());
        assert!(cache.det_j.iter().all(|&d| d > 0.0));
        // Regular squares have area = 1/n_elems
        let expected_area = 1.0 / mesh.n_elems() as f64;
        let avg_area: f64 = cache.area.iter().sum::<f64>() / cache.area.len() as f64;
        assert!((avg_area - expected_area).abs() < 0.01);
    }

    #[test]
    fn cache_invalidate_and_rebuild() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mut cache = ElementCache2D::build(&mesh);
        assert!(cache.valid.iter().all(|&v| v));

        // Invalidate element 0
        cache.invalidate(&[0]);
        assert!(!cache.valid[0]);

        // Rebuild invalidated entries
        cache.rebuild_invalid(&mesh);
        assert!(cache.valid.iter().all(|&v| v));
        assert!(cache.det_j[0] > 0.0);
    }

    // ── Incremental tracker tests ─────────────────────────────────────────

    #[test]
    fn tracker_starts_all_dirty() {
        let t = IncrementalTracker::new(10);
        assert!(t.has_dirty());
        assert_eq!(t.dirty_elems().len(), 10);
    }

    #[test]
    fn tracker_mark_clean() {
        let mut t = IncrementalTracker::new(5);
        t.mark_all_clean();
        assert!(!t.has_dirty());
        assert!(t.dirty_elems().is_empty());
    }

    #[test]
    fn tracker_mark_dirty_and_resize() {
        let mut t = IncrementalTracker::new(5);
        t.mark_all_clean();
        t.mark_dirty(&[1, 3]);
        let dirty = t.dirty_elems();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&1));
        assert!(dirty.contains(&3));

        t.resize(10);
        assert_eq!(t.dirty.len(), 10);
        // New elements are dirty by default
        assert!(t.has_dirty());
    }

    #[test]
    fn tracker_mark_refined() {
        let mut t = IncrementalTracker::new(20);
        t.mark_all_clean();
        // Simulate: element 0 refined into children 10-13
        t.mark_refined(0, &[10, 11, 12, 13]);
        // Element 0 is now inactive (false)
        assert!(!t.dirty[0]);
        // Children should be dirty
        let dirty = t.dirty_elems();
        assert!(dirty.contains(&10), "Child 10 should be dirty");
        assert!(dirty.contains(&11), "Child 11 should be dirty");
        // Parent should not be dirty
        assert!(!dirty.contains(&0), "Parent 0 should not be dirty");
    }

    // ── Batch tests ───────────────────────────────────────────────────────

    #[test]
    fn batch_single_group() {
        let batches = build_batches(10, 100);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].elems.len(), 10);
    }

    #[test]
    fn batch_multiple_groups() {
        let batches = build_batches(100, 32);
        assert_eq!(batches.len(), 4); // ceil(100/32) = 4
        assert_eq!(batches[0].elems.len(), 32);
        assert_eq!(batches[3].elems.len(), 4); // last batch: 100-96=4
    }

    #[test]
    fn batch_empty() {
        let batches = build_batches(0, 32);
        assert!(batches.is_empty());
    }

    #[test]
    fn filter_active_basic() {
        let eta = vec![0.1, 0.5, 0.05, 0.8, 0.01];
        let active = filter_active(&eta, 0.2);
        assert_eq!(active, vec![1, 3]); // indices 1 (0.5) and 3 (0.8)
    }

    // ── Benchmark tests ───────────────────────────────────────────────────

    #[test]
    fn benchmark_single_cycle_runs() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0]*c[0] + c[1]
        }).collect();

        let timing = benchmark_amr_cycle(
            &mesh, &u,
            |m, u| crate::amr::estimators::zz_estimator(m, u),
            |eta| crate::amr::estimators::dorfler_mark(eta, 0.3),
            |m, marked| {
                crate::amr::refine_nonconforming(m, marked, None)
            },
        );

        assert!(timing.total_ms >= 0.0);
        assert!(timing.n_elems_before > 0);
        assert!(timing.error_estimation_ms >= 0.0);
    }

    #[test]
    fn summary_produces_reasonable_values() {
        let timings = vec![
            AmrCycleTiming { total_ms: 100.0, n_elems_after: 1000, ..Default::default() },
            AmrCycleTiming { total_ms: 200.0, n_elems_after: 2000, ..Default::default() },
        ];

        let (total, throughput) = benchmark_summary(&timings);
        assert!((total - 300.0).abs() < 1e-10);
        assert!(throughput > 0.0);
    }

    #[test]
    fn dorfler_mark_gives_subset_of_elements() {
        let eta = vec![0.1, 0.9, 0.05, 0.8, 0.02, 0.7];
        let marked = crate::amr::estimators::dorfler_mark(&eta, 0.5);
        assert!(marked.len() <= eta.len());
        assert!(marked.len() > 0);
        // Marked elements should be the ones with largest eta
        for &e in &marked {
            assert!(eta[e as usize] >= 0.05);
        }
    }
}
