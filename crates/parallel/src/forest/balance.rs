//! 2:1 balance constraint enforcement for the forest.
//!
//! After refinement, the 2:1 balance rule requires that no two adjacent active
//! quadrants differ by more than one refinement level.  This module provides
//! the closure computation that identifies additional quadrants that must be
//! refined to restore balance.
//!
//! Unlike the mesh-based [`RefinementTree::enforce_2to1_balance`], this
//! implementation operates on logical coordinates within the forest and does
//! not require a [`Mesh`] — it uses purely coordinate-based neighbour lookups.

use std::collections::{HashMap, HashSet};

use super::quadrant::{MortonKey, neighbour_key};
use super::tree::Tree;

/// Compute the 2:1 balance closure for a set of quadrants marked for refinement.
///
/// Given a set of currently marked Morton keys, returns the expanded set that
/// satisfies 2:1 balance — any quadrant that has a neighbour more than one
/// level finer must itself be refined.
///
/// # Arguments
///
/// * `marked` — Morton keys already marked for refinement.
/// * `active_map` — map from Morton key to refinement level for all active
///   quadrants in the forest (local + ghost).
///
/// # Returns
///
/// A `HashSet` of Morton keys that must be refined to satisfy 2:1 balance.
pub fn balance_closure_2to1<const D: usize>(
    marked: &HashSet<MortonKey>,
    active_map: &HashMap<MortonKey, u8>,
) -> HashSet<MortonKey> {
    let mut closure: HashSet<MortonKey> = marked.iter().copied().collect();
    let mut changed = true;

    while changed {
        changed = false;
        // Collect candidates for each iteration.
        let candidates: Vec<MortonKey> = closure.iter().copied().collect();

        for key in &candidates {
            let level = active_map.get(key).copied().unwrap_or(0);

            // Check all face neighbours at the same level.
            let n_directions = if D == 2 { 4 } else { 6 };
            for dir in 0..n_directions {
                if let Some(nb_key) = neighbour_key::<D>(key, level, dir) {
                    // Does the neighbour exist at the same level?
                    if let Some(&nb_level) = active_map.get(&nb_key) {
                        // Neighbor exists at some level.
                        if nb_level >= level {
                            // Fine: neighbor is at same or finer level.
                            continue;
                        }
                        // Neighbor is coarser: if level - nb_level > 1, we need balance.
                        if level - nb_level > 1 {
                            // Mark the coarser neighbor for refinement.
                            if !closure.contains(&nb_key) {
                                closure.insert(nb_key);
                                changed = true;
                            }
                        }
                    } else {
                        // No quadrant at this exact key position.
                        // The neighbour might be at a coarser level (a larger
                        // quadrant spanning this position).  Walk up the ancestor
                        // chain until we find one.
                        let mut ancestor_level = level;
                        let mut ancestor_key = key.parent::<D>(level);
                        ancestor_level -= 1;
                        loop {
                            if let Some(&anc_level) = active_map.get(&ancestor_key) {
                                if level - anc_level > 1 {
                                    if !closure.contains(&ancestor_key) {
                                        closure.insert(ancestor_key);
                                        changed = true;
                                    }
                                }
                                break;
                            }
                            if ancestor_level == 0 {
                                break;
                            }
                            ancestor_level -= 1;
                            ancestor_key = ancestor_key.parent::<D>(ancestor_level);
                        }
                    }
                }
            }
        }
    }

    closure
}

/// Compute the 2:1 balance closure on a local tree, given a global active map.
///
/// Convenience wrapper that extracts the active map from the provided trees
/// and ghost information, calls [`balance_closure_2to1`], and returns the
/// expanded set of keys to refine locally.
pub fn enforce_2to1_local<const D: usize>(
    trees: &[Tree<D>],
    ghosts: &[super::Connector],
    marked: &[MortonKey],
) -> Vec<MortonKey> {
    // Build active map from local trees.
    let mut active_map: HashMap<MortonKey, u8> = HashMap::new();
    for t in trees {
        for q in t.quadrants() {
            if q.is_active {
                active_map.insert(q.key, q.level);
            }
        }
    }

    // Add ghost quadrants (which are active by definition).
    for conn in ghosts {
        for &(key, _owner, ghost_level) in &conn.ghosts {
            active_map.entry(key).or_insert(ghost_level);
        }
    }

    let marked_set: HashSet<MortonKey> = marked.iter().copied().collect();
    let closure = balance_closure_2to1::<D>(&marked_set, &active_map);

    let mut result: Vec<MortonKey> = closure.into_iter().collect();
    result.sort();
    result
}

/// Enforce 2:1 balance on the forest (local trees + ghosts).
///
/// `marked` are the Morton keys the user wants to refine.  Returns the keys
/// that need refinement to satisfy 2:1 balance (superset of `marked`).
pub fn forest_balance_2to1<const D: usize>(
    forest: &super::Forest<D>,
    marked: &[MortonKey],
) -> Vec<MortonKey> {
    enforce_2to1_local::<D>(forest.trees(), forest.connectors(), marked)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::quadrant::Quadrant;

    fn make_4x4_active_map() -> HashMap<MortonKey, u8> {
        let mut map = HashMap::new();
        for y in 0u32..4 {
            for x in 0u32..4 {
                let key = MortonKey::from_coords::<2>(0, x, y, 0);
                map.insert(key, 2);
            }
        }
        map
    }

    #[test]
    fn test_balance_closure_empty() {
        let marked = HashSet::new();
        let active_map = make_4x4_active_map();
        let result = balance_closure_2to1::<2>(&marked, &active_map);
        assert!(result.is_empty(), "empty marked → empty closure");
    }

    #[test]
    fn test_balance_closure_no_op() {
        // All at level 2, refining one element doesn't require cascade
        // because neighbours are at the same level.
        let active_map = make_4x4_active_map();
        let key = MortonKey::from_coords::<2>(0, 1, 1, 0);
        let mut marked = HashSet::new();
        marked.insert(key);

        let result = balance_closure_2to1::<2>(&marked, &active_map);
        // Only the marked element should be in the closure.
        assert_eq!(result.len(), 1);
        assert!(result.contains(&key));
    }

    #[test]
    fn test_balance_cascade_level_gap() {
        // Set up a scenario where quadrant at (0,0) level 2 neighbours
        // a quadrant at level 0.  Gap of 2 > 1 → neighbour must be refined.
        let mut active_map = HashMap::new();
        // A level-2 quadrant at (0,0)
        active_map.insert(MortonKey::from_coords::<2>(0, 0, 0, 0), 2);
        // A level-0 quadrant covering the rest (neighbour at x=1 side)
        active_map.insert(MortonKey::from_coords::<2>(0, 0, 0, 0), 0);

        let mut marked = HashSet::new();
        marked.insert(MortonKey::from_coords::<2>(0, 0, 0, 0));

        let result = balance_closure_2to1::<2>(&marked, &active_map);
        // The level-0 quadrant should be added to the closure.
        assert!(result.contains(&MortonKey::from_coords::<2>(0, 0, 0, 0)),
            "coarse neighbour must be added");
    }

    #[test]
    fn test_balance_no_cascade_when_gap_is_one() {
        // Level 3 quadrant neighbour at level 2 → gap of 1, OK.
        let mut active_map = HashMap::new();
        active_map.insert(MortonKey::from_coords::<2>(0, 4, 4, 0), 3);
        // Neighbour at level 2 → should NOT be added.
        active_map.insert(MortonKey::from_coords::<2>(0, 2, 2, 0), 2);

        let mut marked = HashSet::new();
        marked.insert(MortonKey::from_coords::<2>(0, 4, 4, 0));

        let result = balance_closure_2to1::<2>(&marked, &active_map);
        assert_eq!(result.len(), 1, "no cascade needed for gap of 1");
    }

    #[test]
    fn test_balance_3d() {
        let mut active_map = HashMap::new();
        // Level 2 quadrant.
        active_map.insert(MortonKey::from_coords::<3>(0, 0, 0, 0), 2);
        // Level 0 quadrant covering the whole space.
        active_map.insert(MortonKey::from_coords::<3>(0, 0, 0, 0), 0);

        let mut marked = HashSet::new();
        marked.insert(MortonKey::from_coords::<3>(0, 0, 0, 0));

        let result = balance_closure_2to1::<3>(&marked, &active_map);
        assert!(result.contains(&MortonKey::from_coords::<3>(0, 0, 0, 0)),
            "3-D 2:1 balance should catch level gap");
    }

    #[test]
    fn test_enforce_2to1_local() {
        // Build a simple tree with one level-0 and 4 level-2 quadrants.
        let trees = vec![Tree::from_quadrants(vec![
            Quadrant::<2>::new(0, 0, 0, 0, 0, 0),
        ])];

        let marked = vec![MortonKey::from_coords::<2>(0, 0, 0, 0)];
        let result = enforce_2to1_local::<2>(&trees, &[], &marked);
        assert_eq!(result.len(), 1, "single quadrant, no neighbours → no cascade");
    }
}
