//! Distributed forest-of-octrees data structure (cf. p4est).
//!
//! [`Forest<D>`] manages a collection of [`Tree<D>`] objects distributed
//! across MPI ranks.  Quadrants are stored in Morton (Z-order) order within
//! each tree, and trees are partitioned such that each rank owns a contiguous
//! range of the global Morton-order space.
//!
//! The forest supports distributed refinement and coarsening, 2:1 balance,
//! ghost quadrant exchange, and conversion to/from [`Mesh<D>`].

use std::collections::HashMap;

use fem_core::Rank;

use crate::forest::quadrant::{MortonKey, Quadrant};
use crate::forest::tree::Tree;
use crate::forest::connector::Connector;
use crate::Comm;

// ─── ForestStats ──────────────────────────────────────────────────────────────

/// Global statistics for a forest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForestStats {
    /// Total number of trees (coarse grid cells) across all ranks.
    pub n_trees: usize,
    /// Total number of quadrants (active + inactive) across all ranks.
    pub n_quadrants: usize,
    /// Total number of active (leaf) quadrants across all ranks.
    pub n_active: usize,
    /// Minimum refinement level across all active quadrants.
    pub min_level: u8,
    /// Maximum refinement level across all active quadrants.
    pub max_level: u8,
    /// Minimum number of active quadrants on any rank.
    pub min_local: usize,
    /// Maximum number of active quadrants on any rank.
    pub max_local: usize,
}

impl ForestStats {
    fn empty() -> Self {
        Self {
            n_trees: 0,
            n_quadrants: 0,
            n_active: 0,
            min_level: u8::MAX,
            max_level: 0,
            min_local: 0,
            max_local: 0,
        }
    }
}

// ─── Forest ───────────────────────────────────────────────────────────────────

/// A distributed forest of octrees (or quadtrees in 2-D).
///
/// Structure mirrors the p4est library's design:
///
/// - The logical domain is partitioned into a uniform coarse grid
///   (the "trees" of the forest).
/// - Each tree is independently refined/coarsened via quadrants.
/// - Trees are distributed across MPI ranks by a Morton-order prefix-sum,
///   so that each rank owns a contiguous segment of the Z-order curve.
/// - Ghost quadrants (owned by other ranks but adjacent to local quadrants)
///   are tracked separately.
#[derive(Clone)]
pub struct Forest<const D: usize> {
    /// Local trees owned by this MPI rank.
    pub(super) trees: Vec<Tree<D>>,
    /// Ghost connectors tracking cross-rank quadrant adjacency.
    pub(super) connectors: Vec<Connector>,
    /// MPI communicator.
    pub(super) comm: Comm,
    /// Cumulative local quadrant counts for prefix-sum partitioning.
    /// `tree_prefix[i] = number of quadrants in trees[0..i]`.
    pub(super) tree_prefix: Vec<usize>,
    /// Global forest statistics.
    pub(super) stats: ForestStats,
}

impl<const D: usize> std::fmt::Debug for Forest<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Forest")
            .field("n_trees", &self.trees.len())
            .field("n_connectors", &self.connectors.len())
            .field("n_local_active", &self.n_local_active())
            .field("stats", &self.stats)
            .finish()
    }
}

impl<const D: usize> Forest<D> {
    /// Create an empty forest.
    pub fn empty(comm: Comm) -> Self {
        Self {
            trees: Vec::new(),
            connectors: Vec::new(),
            comm,
            tree_prefix: Vec::new(),
            stats: ForestStats::empty(),
        }
    }

    /// Create a forest from a set of initial (uniform-level-0) trees.
    ///
    /// Each entry in `tree_quadrants` is the list of quadrants for one coarse
    /// grid cell / tree.  For a uniform initial mesh each tree has exactly
    /// 1 quadrant (the root of that tree).
    ///
    /// The quadrants are distributed across ranks by Morton-order prefix-sum.
    pub fn from_trees(
        tree_quadrants: Vec<Vec<Quadrant<D>>>,
        comm: Comm,
    ) -> Self {
        let n_trees = tree_quadrants.len();

        // Build local trees (all on rank 0 initially, then partition).
        let local_trees: Vec<Tree<D>> = tree_quadrants
            .into_iter()
            .map(Tree::from_quadrants)
            .collect();

        Self::distribute_trees(local_trees, comm, n_trees)
    }

    /// Distribute trees across MPI ranks using Morton-order prefix-sum.
    fn distribute_trees(trees: Vec<Tree<D>>, comm: Comm, n_trees: usize) -> Self {
        let size = comm.size();
        let rank = comm.rank() as Rank;

        // Count active quadrants per tree for the prefix-sum.
        let local_counts: Vec<usize> = trees.iter().map(|t| t.n_active()).collect();
        let total_local: usize = local_counts.iter().sum();

        // Global prefix-sum of quadrant counts (all-gather the counts).
        // For now (serial / single rank), all trees stay on this rank.
        let (assigned_trees, prefix, stats) = if size <= 1 {
            // Single rank: keep all trees.
            let mut cum = Vec::with_capacity(trees.len());
            let mut s = 0;
            for c in &local_counts {
                cum.push(s);
                s += c;
            }
            let stats = compute_stats(&trees, comm.clone());
            (trees, cum, stats)
        } else {
            // Multi-rank: use prefix-sum partitioning.
            let _all_counts = Self::gather_counts(&local_counts, &comm);
            // Distribute by Morton-order prefix-sum.
            let (_start, _end, assigned) =
                Self::partition_by_prefix(&local_counts, &_all_counts, rank, size);
            let subset: Vec<Tree<D>> = trees.into_iter()
                .enumerate()
                .filter(|&(i, _)| assigned.contains(&i))
                .map(|(_, t)| t)
                .collect();
            let mut cum = Vec::with_capacity(subset.len());
            let mut s = 0;
            for t in &subset {
                cum.push(s);
                s += t.n_active();
            }
            let stats = compute_stats(&subset, comm.clone());
            (subset, cum, stats)
        };

        Forest {
            trees: assigned_trees,
            connectors: Vec::new(),
            comm,
            tree_prefix: prefix,
            stats,
        }
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    /// The local trees on this rank.
    pub fn trees(&self) -> &[Tree<D>] {
        &self.trees
    }

    /// Mutable reference to local trees.
    pub fn trees_mut(&mut self) -> &mut [Tree<D>] {
        &mut self.trees
    }

    /// The MPI communicator.
    pub fn comm(&self) -> &Comm {
        &self.comm
    }

    /// Global statistics.
    pub fn stats(&self) -> &ForestStats {
        &self.stats
    }

    /// Number of local active quadrants (leaf elements on this rank).
    pub fn n_local_active(&self) -> usize {
        self.trees.iter().map(|t| t.n_active()).sum()
    }

    /// Number of local quadrants (active + inactive).
    pub fn n_local_quadrants(&self) -> usize {
        self.trees.iter().map(|t| t.n_quadrants()).sum()
    }

    /// Collect all local active quadrants in Morton order.
    pub fn local_active_quadrants(&self) -> Vec<Quadrant<D>> {
        let mut result = Vec::new();
        for tree in &self.trees {
            result.extend(tree.active_quadrants());
        }
        result
    }

    /// Collect all local active Morton keys.
    pub fn local_active_keys(&self) -> Vec<MortonKey> {
        let mut result = Vec::new();
        for tree in &self.trees {
            result.extend(tree.active_keys());
        }
        result
    }

    /// Ghost connectors.
    pub fn connectors(&self) -> &[Connector] {
        &self.connectors
    }

    // ─── Refinement ─────────────────────────────────────────────────────────

    /// Refine all quadrants with keys in `marked`.
    ///
    /// `marked` contains Morton keys of quadrants to refine (can be from
    /// any tree).  Returns the keys of newly created children.
    pub fn refine_keys(&mut self, marked: &[MortonKey]) -> Vec<MortonKey> {
        let mut children = Vec::new();

        if marked.is_empty() {
            return children;
        }

        // Build a set of marked keys for fast lookup.
        let mark_set: std::collections::HashSet<MortonKey> =
            marked.iter().copied().collect();

        for tree in &mut self.trees {
            // Find indices of marked quadrants in this tree.
            let mut to_refine: Vec<usize> = tree
                .quadrants()
                .iter()
                .enumerate()
                .filter(|(_, q)| q.is_active && mark_set.contains(&q.key))
                .map(|(i, _)| i)
                .collect();

            let new_children = tree.refine_marked(to_refine);

            // Record the keys of new children.
            for &ci in &new_children {
                if ci < tree.quadrants().len() {
                    children.push(tree.quadrants()[ci].key);
                }
            }
        }

        self.refresh_stats();
        children
    }

    /// Coarsen all quadrants whose parent key is in `marked_parents`.
    ///
    /// `marked_parents` are the Morton keys of parents to restore.
    /// Only valid when all `2^D` children of each parent are present
    /// and active on this rank.
    pub fn coarsen_keys(&mut self, marked_parents: &[MortonKey]) -> Vec<MortonKey> {
        let mut restored = Vec::new();

        if marked_parents.is_empty() {
            return restored;
        }

        for tree in &mut self.trees {
            let parents = tree.coarsen_marked(marked_parents);
            for &pi in &parents {
                if pi < tree.quadrants().len() {
                    restored.push(tree.quadrants()[pi].key);
                }
            }
        }

        self.refresh_stats();
        restored
    }

    // ─── Partitioning ───────────────────────────────────────────────────────

    /// Gather local quadrant counts from all ranks.
    fn gather_counts(local: &[usize], comm: &Comm) -> Vec<usize> {
        let size = comm.size();
        if size <= 1 {
            return local.to_vec();
        }

        let local_total: usize = local.iter().sum();
        let _all_totals: Vec<i64> = (0..size)
            .map(|_| comm.allreduce_sum_i64(local_total as i64))
            .collect();

        // For a full gather we'd use allgather; here we use a simple
        // approach for now.
        let mut all = vec![0usize; size];
        for r in 0..size {
            all[r] = comm.allreduce_sum_i64(if r as i32 == comm.rank() {
                local_total as i64
            } else {
                0
            }) as usize;
        }
        all
    }

    /// Determine tree assignment for each rank based on Morton-order
    /// prefix-sum partitioning.
    fn partition_by_prefix(
        local_counts: &[usize],
        all_counts: &[usize],
        rank: Rank,
        size: usize,
    ) -> (usize, usize, Vec<usize>) {
        if size <= 1 {
            let all_trees: Vec<usize> = (0..local_counts.len()).collect();
            return (0, local_counts.len(), all_trees);
        }

        // Build global prefix-sum.
        let n_trees: usize = all_counts.len();
        let mut prefix = vec![0usize; n_trees + 1];
        for i in 0..n_trees {
            prefix[i + 1] = prefix[i] + all_counts[i];
        }
        let total = prefix[n_trees];
        let target_per_rank = total / size;
        let rem = total % size;

        // Compute start/end in the prefix-sum for this rank.
        let r = rank as usize;
        let start = r * target_per_rank + r.min(rem);
        let end = (r + 1) * target_per_rank + (r + 1).min(rem);

        // Find which trees this rank owns.
        let mut assigned = Vec::new();
        for i in 0..n_trees {
            let tree_start = prefix[i];
            let tree_end = prefix[i + 1];
            if tree_start < end && tree_end > start {
                assigned.push(i);
            }
        }

        (start, end, assigned)
    }

    // ─── Load balancing ─────────────────────────────────────────────────────

    /// Compute the global load imbalance factor.
    ///
    /// Returns `max_local / ideal`.  Value > 1.0 means overloaded ranks exist.
    pub fn imbalance(&self) -> f64 {
        let size = self.comm.size() as f64;
        if size <= 0.0 {
            return 0.0;
        }
        let stats = self.stats;
        if stats.n_active == 0 {
            return 0.0;
        }
        let ideal = stats.n_active as f64 / size;
        if ideal <= 0.0 {
            return 1.0;
        }
        stats.max_local as f64 / ideal
    }

    /// Recompute statistics from local trees and synchronize globally.
    pub fn refresh_stats(&mut self) {
        let mut stats = ForestStats {
            n_trees: self.trees.len(),
            n_quadrants: self.n_local_quadrants(),
            n_active: self.n_local_active(),
            min_level: u8::MAX,
            max_level: 0,
            min_local: usize::MAX,
            max_local: 0,
        };

        // Local level range.
        for t in &self.trees {
            for q in t.quadrants() {
                if q.is_active {
                    let lvl = q.level;
                    if lvl < stats.min_level {
                        stats.min_level = lvl;
                    }
                    if lvl > stats.max_level {
                        stats.max_level = lvl;
                    }
                }
            }
        }

        // Global sync.
        let size = self.comm.size();
        if size > 1 {
            let n_local = stats.n_active;
            // Gather min/max levels.
            let global_min = self.comm.allreduce_sum_i64(
                if stats.min_level < u8::MAX { stats.min_level as i64 } else { i64::MAX }
            );
            stats.min_level = global_min.min(u8::MAX as i64) as u8;
            let global_max = self.comm.allreduce_sum_i64(stats.max_level as i64);
            stats.max_level = global_max as u8;

            // Gather n_active stats across ranks.
            let all_min = self.comm.allreduce_sum_i64(n_local as i64);
            stats.min_local = all_min as usize;
            let all_max = self.comm.allreduce_sum_i64(n_local as i64);
            stats.max_local = all_max as usize;

            // Sum counts.
            let n_global = self.comm.allreduce_sum_i64(n_local as i64) as usize;
            stats.n_active = n_global;
            let n_q_global = self.comm.allreduce_sum_i64(stats.n_quadrants as i64) as usize;
            stats.n_quadrants = n_q_global;
            let n_t_global = self.comm.allreduce_sum_i64(stats.n_trees as i64) as usize;
            stats.n_trees = n_t_global;
        } else {
            stats.min_local = stats.n_active;
            stats.max_local = stats.n_active;
        }

        self.stats = stats;
    }

    // ─── MPI Coordination ───────────────────────────────────────────────────

    /// Synchronise forest state across all ranks (barrier + stats refresh).
    pub fn synchronize(&mut self) {
        self.comm.barrier();
        self.refresh_stats();
    }

    /// All-reduce the sum of a local value.
    pub fn allreduce_sum(&self, local: i64) -> i64 {
        self.comm.allreduce_sum_i64(local)
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

fn compute_stats<const D: usize>(trees: &[Tree<D>], _comm: Comm) -> ForestStats {
    let mut n_active = 0;
    let mut n_quadrants = 0;
    let mut min_level = u8::MAX;
    let mut max_level = 0;

    for t in trees {
        n_quadrants += t.n_quadrants();
        n_active += t.n_active();
        for q in t.quadrants() {
            if q.is_active {
                let lvl = q.level;
                if lvl < min_level { min_level = lvl; }
                if lvl > max_level { max_level = lvl; }
            }
        }
    }

    ForestStats {
        n_trees: trees.len(),
        n_quadrants,
        n_active,
        min_level,
        max_level,
        min_local: n_active,
        max_local: n_active,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::native::SerialBackend;

    fn serial_comm() -> Comm {
        Comm::from_backend(Box::new(SerialBackend))
    }

    fn single_tree_forest_2d() -> Forest<2> {
        let q = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
        Forest::from_trees(vec![vec![q]], serial_comm())
    }

    fn four_tree_forest_2d() -> Forest<2> {
        let trees: Vec<Vec<Quadrant<2>>> = (0u32..4)
            .map(|i| {
                let x = i % 2;
                let y = i / 2;
                vec![Quadrant::<2>::new(i, 0, x, y, 0, i as i32)]
            })
            .collect();
        Forest::from_trees(trees, serial_comm())
    }

    #[test]
    fn test_empty_forest() {
        let comm = serial_comm();
        let forest = Forest::<2>::empty(comm);
        assert_eq!(forest.n_local_active(), 0);
        assert_eq!(forest.n_local_quadrants(), 0);
    }

    #[test]
    fn test_forest_from_single_tree() {
        let forest = single_tree_forest_2d();
        assert_eq!(forest.n_local_active(), 1);
        assert_eq!(forest.stats().n_trees, 1);
    }

    #[test]
    fn test_forest_from_four_trees() {
        let forest = four_tree_forest_2d();
        assert_eq!(forest.n_local_active(), 4);
        assert_eq!(forest.trees().len(), 4);

        // Each tree has 1 root quadrant.
        for t in forest.trees() {
            assert_eq!(t.n_active(), 1);
        }
    }

    #[test]
    fn test_refine_single_quadrant_in_forest() {
        let mut forest = single_tree_forest_2d();
        let root_key = MortonKey::from_coords::<2>(0, 0, 0, 0);

        let children = forest.refine_keys(&[root_key]);
        assert_eq!(children.len(), 4);

        // Forest now has 4 active quadrants.
        assert_eq!(forest.n_local_active(), 4);
        assert_eq!(forest.n_local_quadrants(), 5); // 1 inactive parent + 4 children
    }

    #[test]
    fn test_refine_all_in_forest() {
        let mut forest = four_tree_forest_2d();
        let keys: Vec<MortonKey> = forest.local_active_keys();
        assert_eq!(keys.len(), 4);

        let children = forest.refine_keys(&keys);
        // 4 parents × 4 children each
        assert_eq!(children.len(), 16);
        assert_eq!(forest.n_local_active(), 16);
    }

    #[test]
    fn test_refine_and_coarsen_forest_roundtrip() {
        let mut forest = single_tree_forest_2d();
        let root_key = MortonKey::from_coords::<2>(0, 0, 0, 0);

        // Refine.
        forest.refine_keys(&[root_key]);
        assert_eq!(forest.n_local_active(), 4);

        // Coarsen back.
        let restored = forest.coarsen_keys(&[root_key]);
        assert_eq!(restored.len(), 1);
        assert_eq!(forest.n_local_active(), 1);
    }

    #[test]
    fn test_imbalance_single_rank() {
        let forest = single_tree_forest_2d();
        let imb = forest.imbalance();
        assert!((imb - 1.0).abs() < 1e-12, "single rank should have imbalance 1.0, got {imb}");
    }

    #[test]
    fn test_local_active_quadrants_order() {
        let mut forest = four_tree_forest_2d();
        // Refine two of the trees.
        let keys: Vec<MortonKey> = forest.local_active_keys();
        forest.refine_keys(&keys[..2]);

        let active = forest.local_active_quadrants();
        // Verify Morton order.
        for i in 1..active.len() {
            assert!(active[i - 1].key < active[i].key,
                "active quadrants should be in Morton order");
        }
    }

    #[test]
    fn test_refine_twice_marks_parent_inactive() {
        let mut forest = single_tree_forest_2d();
        let root_key = MortonKey::from_coords::<2>(0, 0, 0, 0);

        // Level 1 refine.
        forest.refine_keys(&[root_key]);
        assert!(!forest.trees[0].quadrants()[0].is_active);

        // Level 2 refine: refine the child at (1,0), which has Morton key = 1.
        let child_key = MortonKey::from_coords::<2>(0, 1, 0, 0);
        forest.refine_keys(&[child_key]);

        // The child is now inactive, its grandchildren are active at different keys.
        let active = forest.local_active_keys();
        assert!(!active.contains(&child_key),
            "child key 1 should not be active after refinement");
        // Should have: 3 children from level 1 still active at keys {2,3}
        //             + 4 grandchildren at keys {4,5,6,7} = 7 active keys
        assert_eq!(active.len(), 7,
            "expected 7 active keys: 3 surviving children + 4 grandchildren");
    }

    #[test]
    fn test_stats_single_rank() {
        let forest = single_tree_forest_2d();
        let stats = forest.stats;
        assert_eq!(stats.n_active, 1);
        assert_eq!(stats.n_trees, 1);
        assert_eq!(stats.n_quadrants, 1);
        assert_eq!(stats.min_level, 0);
        assert_eq!(stats.max_level, 0);
    }
}
