//! Local tree of quadrants on a single MPI rank.
//!
//! Each [`Tree<D>`] stores a sorted sequence of quadrants belonging to one
//! partition of the Morton-order space.  The tree supports local refinement
//! and coarsening, cumulative prefix-sum queries for distributed partitioning,
//! and lookup by Morton key.

use crate::forest::quadrant::{MortonKey, Quadrant};

/// A local tree of quadrants (one per "coarse" grid cell or per partition).
///
/// Each tree stores its quadrants in strict Morton-key order so that a global
/// Z-order traversal across all trees on all ranks is a simple concatenation
/// of each rank's quadrant sequence.
#[derive(Debug, Clone)]
pub struct Tree<const D: usize> {
    /// Quadrants sorted by Morton key.
    quadrants: Vec<Quadrant<D>>,
    /// Cumulative element counts for prefix-sum based partitioning.
    /// `cumulative[i] = sum_{j < i} n_active_quadrants_in_tree(j)`.
    /// Used only in the forest-level partitioning; may be empty.
    #[allow(dead_code)]
    pub(super) cumulative: Vec<usize>,
}

impl<const D: usize> Tree<D> {
    /// Create an empty tree.
    pub fn empty() -> Self {
        Self {
            quadrants: Vec::new(),
            cumulative: Vec::new(),
        }
    }

    /// Create a tree from a pre-sorted set of quadrants.
    ///
    /// Ensures quadrants are sorted by Morton key.
    pub fn from_quadrants(mut quadrants: Vec<Quadrant<D>>) -> Self {
        quadrants.sort();
        Self {
            quadrants,
            cumulative: Vec::new(),
        }
    }

    /// The number of quadrants (active and inactive) in this tree.
    pub fn n_quadrants(&self) -> usize {
        self.quadrants.len()
    }

    /// Number of active (leaf) quadrants in this tree.
    pub fn n_active(&self) -> usize {
        self.quadrants.iter().filter(|q| q.is_active).count()
    }

    /// Reference to all quadrants.
    pub fn quadrants(&self) -> &[Quadrant<D>] {
        &self.quadrants
    }

    /// Mutable reference to all quadrants.
    pub fn quadrants_mut(&mut self) -> &mut [Quadrant<D>] {
        &mut self.quadrants
    }

    /// Index of a quadrant by its Morton key (binary search).
    pub fn find(&self, key: &MortonKey) -> Option<usize> {
        self.quadrants.binary_search_by_key(key, |q| q.key).ok()
    }

    /// The Morton-key range of this tree's quadrants.
    /// Returns `(first_key, last_key)` if non-empty.
    pub fn key_range(&self) -> Option<(MortonKey, MortonKey)> {
        self.quadrants.first().map(|f| {
            let l = self.quadrants.last().unwrap();
            (f.key, l.key)
        })
    }

    /// Refine a single quadrant by its index in the tree.
    ///
    /// The quadrant is marked inactive and its children are inserted,
    /// maintaining Morton-key order.  Returns the indices of the new children.
    pub fn refine_at(&mut self, index: usize) -> Vec<usize> {
        let children = self.quadrants[index].refine();
        let child_count = children.len();

        // Insert children after the parent position, preserving Morton order.
        // Since children all have keys > parent_key and < next_sibling_key,
        // we can insert them at index+1.
        let insert_pos = index + 1;
        let child_indices: Vec<usize> = (0..child_count)
            .map(|i| insert_pos + i)
            .collect();

        // Insert children in reverse so that the final order is correct.
        // Actually, since `splice` replaces a range, we insert at once.
        let tail = self.quadrants.split_off(insert_pos);
        self.quadrants.extend(children);
        self.quadrants.extend(tail);

        child_indices
    }

    /// Refine all quadrants that are marked and active.
    ///
    /// `marked` is a list of indices into `self.quadrants` that should be
    /// refined.  Returns the indices of newly created children.
    ///
    /// Processes marks in **descending** index order, then re-sorts the
    /// quadrant list to maintain Morton-key ordering (children's keys may
    /// sort after subsequent elements due to cross-level coordinate overlap).
    pub fn refine_marked(&mut self, mut marked: Vec<usize>) -> Vec<usize> {
        // Only active quadrants can be refined.
        marked.retain(|&i| i < self.quadrants.len() && self.quadrants[i].is_active);
        marked.sort_unstable_by(|a, b| b.cmp(a)); // descending

        for &idx in &marked {
            self.refine_at(idx);
        }

        // Re-sort to maintain Morton-key ordering after all refinements.
        self.quadrants.sort_by(|a, b| a.key.cmp(&b.key).then_with(|| a.level.cmp(&b.level)));

        // Return indices of new children (approximate: active quadrants
        // with the highest levels).
        let max_level = self.quadrants.iter().map(|q| q.level).max().unwrap_or(0);
        let child_indices: Vec<usize> = self.quadrants
            .iter()
            .enumerate()
            .filter(|(_, q)| q.is_active && q.level == max_level)
            .map(|(i, _)| i)
            .collect();
        child_indices
    }

    /// Coarsen a group of `2^D` sibling quadrants into their parent.
    ///
    /// All must be active siblings forming a complete set.  Returns the
    /// index of the newly created parent quadrant, or `None` if the
    /// siblings don't form a valid coarsening set.
    pub fn coarsen_siblings(&mut self, indices: &[usize]) -> Option<usize> {
        if indices.len() != (1 << D) {
            return None;
        }

        // Collect sibling quadrants.
        let siblings: Vec<Quadrant<D>> = indices.iter()
            .map(|&i| self.quadrants[i].clone())
            .collect();

        // Verify all are active and share the same parent.
        let parent_key = siblings[0].key.parent::<D>(siblings[0].level);
        if !siblings.iter().all(|q| q.is_active && q.key.parent::<D>(q.level) == parent_key) {
            return None;
        }

        let parent = Quadrant::<D>::coarsen(&siblings)?;

        // Remove siblings and insert parent at the first sibling's position.
        let first = indices[0];
        // Remove in descending order.
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &sorted_indices {
            self.quadrants.remove(idx);
        }
        self.quadrants.insert(first, parent);

        Some(first)
    }

    /// Coarsen all complete sets of siblings that are marked.
    ///
    /// `marked_parents` are the Morton keys of the parent quadrants to
    /// coarsen back to.  Returns the indices of newly created parent quadrants.
    pub fn coarsen_marked(&mut self, marked_parents: &[MortonKey]) -> Vec<usize> {
        let mut new_parents = Vec::new();

        for &parent_key in marked_parents {
            // Find all children of this parent quadrant.
            // Children have key = parent_key with an appended child index.
            // The children of a quadrant are those 2^D quadrants whose
            // `key.parent::<D>() == parent_key`.
            let child_indices: Vec<usize> = (0..self.quadrants.len())
                .filter(|&i| {
                    let q = &self.quadrants[i];
                    q.is_active
                        && q.key.parent::<D>(q.level) == parent_key
                })
                .collect();

            if child_indices.len() == (1 << D) {
                if let Some(idx) = self.coarsen_siblings(&child_indices) {
                    new_parents.push(idx);
                }
            }
        }

        new_parents
    }

    /// Collect all active quadrant keys in Morton order.
    pub fn active_keys(&self) -> Vec<MortonKey> {
        self.quadrants
            .iter()
            .filter(|q| q.is_active)
            .map(|q| q.key)
            .collect()
    }

    /// Collect all active quadrants (cloned).
    pub fn active_quadrants(&self) -> Vec<Quadrant<D>> {
        self.quadrants
            .iter()
            .filter(|q| q.is_active)
            .cloned()
            .collect()
    }

    /// Check internal invariants:
    /// - Quadrants are sorted by Morton key.
    /// - No duplicate (key, level) pairs among active quadrants.
    pub fn check_invariants(&self) -> bool {
        let mut prev_key: Option<MortonKey> = None;
        for q in &self.quadrants {
            if let Some(pk) = prev_key {
                // Keys must be non-decreasing to support binary search.
                // Equal keys are allowed (different levels at same spatial pos).
                if q.key < pk {
                    return false;
                }
            }
            prev_key = Some(q.key);
        }

        // Check for duplicate (key, level) pairs among active quadrants.
        let mut seen: std::collections::HashSet<(MortonKey, u8)> = std::collections::HashSet::new();
        for q in &self.quadrants {
            if q.is_active {
                if !seen.insert((q.key, q.level)) {
                    return false; // duplicate active (key, level)
                }
            }
        }
        true
    }
}

// ─── Iterator: quadrants in Morton order ──────────────────────────────────────

impl<const D: usize> IntoIterator for Tree<D> {
    type Item = Quadrant<D>;
    type IntoIter = std::vec::IntoIter<Quadrant<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.quadrants.into_iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a Tree<D> {
    type Item = &'a Quadrant<D>;
    type IntoIter = std::slice::Iter<'a, Quadrant<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.quadrants.iter()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::quadrant::Quadrant;

    fn make_single_tree_2d() -> Tree<2> {
        let q = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
        Tree::from_quadrants(vec![q])
    }

    fn make_4x4_tree_2d() -> Tree<2> {
        let mut qs = Vec::new();
        for y in 0u32..4 {
            for x in 0u32..4 {
                qs.push(Quadrant::<2>::new(0, 2, x, y, 0, 0));
            }
        }
        Tree::from_quadrants(qs)
    }

    #[test]
    fn test_empty_tree() {
        let tree = Tree::<2>::empty();
        assert_eq!(tree.n_quadrants(), 0);
        assert_eq!(tree.n_active(), 0);
    }

    #[test]
    fn test_tree_from_quadrants_is_sorted() {
        let qs = vec![
            Quadrant::<2>::new(0, 2, 1, 0, 0, 0),
            Quadrant::<2>::new(0, 2, 0, 0, 0, 0),
        ];
        let tree = Tree::from_quadrants(qs.clone());
        assert!(tree.check_invariants());
        // The tree should have sorted them.
        assert!(tree.quadrants[0].key < tree.quadrants[1].key);
    }

    #[test]
    fn test_refine_single_quadrant() {
        let mut tree = make_single_tree_2d();
        assert_eq!(tree.n_active(), 1);

        let children = tree.refine_at(0);
        assert_eq!(children.len(), 4);

        // After refinement: parent inactive + 4 children = 5 quadrants total.
        assert_eq!(tree.n_quadrants(), 5);
        assert_eq!(tree.n_active(), 4);

        assert!(!tree.quadrants[0].is_active);
        for i in 1..5 {
            assert!(tree.quadrants[i].is_active);
            assert_eq!(tree.quadrants[i].level, 1);
        }
    }

    #[test]
    fn test_refine_multiple_quadrants() {
        let mut tree = make_4x4_tree_2d();
        let n_initial = tree.n_quadrants();
        assert_eq!(n_initial, 16);

        // Refine first 4 quadrants (in Morton order).
        let children = tree.refine_marked(vec![0, 1, 2, 3]);
        // Each refined quadrant produces 4 children (parent stays inactive).
        assert_eq!(children.len(), 16);
        assert_eq!(tree.n_quadrants(), 32); // 16 + 4*4 = 32

        // The 4 original parents are now inactive (find by level+key).
        // Since multiple refined generations may share a key, we scan all quadrants.
        let n_inactive = tree.quadrants().iter().filter(|q| !q.is_active).count();
        assert_eq!(n_inactive, 4, "4 original parents should be inactive");
        // Verify Morton ordering: keys must be non-decreasing.
        let mut prev: Option<MortonKey> = None;
        for q in tree.quadrants() {
            if let Some(pk) = prev {
                assert!(q.key >= pk,
                    "key ordering violation: {:?} < {:?}", q.key, pk);
            }
            prev = Some(q.key);
        }
        // Verify no duplicate (key, level) among active quadrants.
        let mut seen: std::collections::HashSet<(u64, u8)> = std::collections::HashSet::new();
        for q in tree.quadrants() {
            if q.is_active {
                let id = (q.key.0, q.level);
                assert!(seen.insert(id),
                    "duplicate active (key, level): key={:?}, level={}", q.key, q.level);
            }
        }
    }

    #[test]
    fn test_coarsen_siblings() {
        let mut tree = Tree::<2>::empty();
        // Create a parent and 4 children at level 1.
        let mut parent = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
        parent.is_active = false; // parent inactive
        tree.quadrants.push(parent);
        // Children in Morton order: (0,0), (1,0), (0,1), (1,1)
        for &(x, y) in &[(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
            tree.quadrants.push(Quadrant::<2>::new(0, 1, x, y, 0, 0));
        }
        assert_eq!(tree.n_quadrants(), 5);

        // Coarsen children 1..5 (indices 1-4).
        let result = tree.coarsen_siblings(&[1, 2, 3, 4]);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 1);

        // After coarsening: parent (inactive) + 1 new parent = 2 quadrants.
        assert_eq!(tree.n_quadrants(), 2);
        // The new parent is active, at level 0.
        assert!(tree.quadrants[1].is_active);
        assert_eq!(tree.quadrants[1].level, 0);
    }

    #[test]
    fn test_refine_and_coarsen_roundtrip() {
        let mut tree = make_single_tree_2d();

        // Refine.
        let _children = tree.refine_at(0);
        assert_eq!(tree.n_active(), 4);

        // Coarsen back.
        let parent_key = MortonKey::from_coords::<2>(0, 0, 0, 0);
        let new_parents = tree.coarsen_marked(&[parent_key]);
        assert_eq!(new_parents.len(), 1);

        // Back to 1 active quadrant (the root).
        assert_eq!(tree.n_active(), 1);
    }

    #[test]
    fn test_keys_in_sorted_order() {
        let tree = make_4x4_tree_2d();
        let keys = tree.active_keys();
        for i in 1..keys.len() {
            assert!(keys[i - 1] < keys[i], "keys should be strictly increasing");
        }
    }

    #[test]
    fn test_find_quadrant() {
        let tree = make_4x4_tree_2d();
        let key = MortonKey::from_coords::<2>(0, 1, 2, 0);
        let idx = tree.find(&key);
        assert!(idx.is_some());
        assert_eq!(tree.quadrants[idx.unwrap()].x(), 1);
        assert_eq!(tree.quadrants[idx.unwrap()].y(), 2);
    }
}
