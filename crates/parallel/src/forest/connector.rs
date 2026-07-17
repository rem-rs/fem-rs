//! Ghost quadrant and connector system for cross-rank adjacency.
//!
//! When a forest is partitioned across MPI ranks, quadrants on one rank may
//! be adjacent to quadrants owned by another rank.  These "ghost" quadrants
//! are needed for:
//!
//! - **2:1 balance** — detecting level gaps that cross rank boundaries.
//! - **Flux computation** — evaluating face integrals at partition boundaries.
//! - **Solution transfer** — interpolating solution values at shared interfaces.
//!
//! The [`Connector<D>`] type tracks the set of ghost quadrants for each
//! neighbouring rank.

use std::collections::HashSet;

use crate::forest::quadrant::{MortonKey, Quadrant, neighbour_key};
use crate::forest::tree::Tree;
use crate::forest::forest::Forest;
use crate::Comm;

/// A ghost information container for inter-rank communication.
///
/// Each ghost entry records the Morton key of a quadrant owned by another
/// rank, along with the owning rank's ID and the quadrant's refinement level.
#[derive(Debug, Clone)]
pub struct Connector {
    /// List of ghost quadrant keys with their owner ranks and levels.
    pub ghosts: Vec<(MortonKey, i32, u8)>,
    /// Number of locally owned quadrants (for reference).
    pub n_local: usize,
}

impl Connector {
    /// Create an empty connector.
    pub fn empty() -> Self {
        Self {
            ghosts: Vec::new(),
            n_local: 0,
        }
    }

    /// Number of ghost quadrants.
    pub fn n_ghosts(&self) -> usize {
        self.ghosts.len()
    }

    /// Check whether a key belongs to a ghost quadrant.
    pub fn is_ghost(&self, key: &MortonKey) -> bool {
        self.ghosts.iter().any(|(k, _, _)| k == key)
    }

    /// Get the owner of a ghost quadrant, if present.
    pub fn owner(&self, key: &MortonKey) -> Option<i32> {
        self.ghosts.iter().find(|(k, _, _)| k == key).map(|(_, o, _)| *o)
    }
}

/// Build connectors for a forest by finding cross-rank quadrant adjacencies.
///
/// For each local active quadrant, this function checks its face neighbours
/// at the same level.  If a neighbour's key is not in the local active set
/// and is not already in the ghost set, it is added as a ghost.
///
/// This is a simplified implementation that works well for serial/single-rank
/// testing.  Multi-rank operation requires MPI-based discovery of which rank
/// owns each ghost quadrant (to be extended later).
pub fn build_connectors<const D: usize>(
    trees: &[Tree<D>],
    comm: &Comm,
) -> Vec<Connector> {
    let size = comm.size();

    // For single rank, no ghosts needed.
    if size <= 1 {
        return Vec::new();
    }

    // Build set of local active keys for fast lookup,
    // and collect quadrants with their levels for neighbour computation.
    let local_active: HashSet<MortonKey> = trees
        .iter()
        .flat_map(|t| t.active_keys())
        .collect();
    let active_quadrants: Vec<Quadrant<D>> = trees
        .iter()
        .flat_map(|t| t.active_quadrants())
        .collect();

    // Discover ghost keys: neighbours of local quadrants that are not local.
    let mut ghost_keys: HashSet<MortonKey> = HashSet::new();

    for q in &active_quadrants {
        let n_directions = if D == 2 { 4 } else { 6 };
        for dir in 0..n_directions {
            if let Some(nb_key) = neighbour_key::<D>(&q.key, q.level, dir) {
                if !local_active.contains(&nb_key) {
                    ghost_keys.insert(nb_key);
                }
            }
        }
    }

    // For now (simplified), all ghosts are assigned an owner of -1
    // (meaning owner unknown, to be resolved via MPI in a future extension).
    // The ghosts are still usable for balance checks and local operations.
    // Ghost level is set to 0 by default (simplification for single-rank usage).
    let ghosts: Vec<(MortonKey, i32, u8)> = ghost_keys
        .into_iter()
        .map(|k| (k, -1, 0))
        .collect();

    let n_local = local_active.len();
    vec![Connector { ghosts, n_local }]
}

/// Update the Forest's ghost connectors after refinement or coarsening.
pub fn update_connectors<const D: usize>(forest: &mut Forest<D>) {
    let connectors = build_connectors(forest.trees(), forest.comm());
    forest.connectors = connectors;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::native::SerialBackend;

    fn serial_comm() -> Comm {
        Comm::from_backend(Box::new(SerialBackend))
    }

    #[test]
    fn test_empty_connector() {
        let conn = Connector::empty();
        assert_eq!(conn.n_ghosts(), 0);
        assert_eq!(conn.n_local, 0);
        assert!(!conn.is_ghost(&MortonKey::ROOT));
    }

    #[test]
    fn test_single_rank_no_ghosts() {
        let trees = vec![Tree::from_quadrants(vec![
            Quadrant::<2>::new(0, 0, 0, 0, 0, 0),
        ])];
        let conns = build_connectors::<2>(&trees, &serial_comm());
        assert!(conns.is_empty(), "single rank should have no ghosts");
    }

    #[test]
    fn test_ghost_discovery_for_refined_quadrant() {
        // Build a single tree with all quadrants at level 1 covering a 2x2 grid.
        let mut qs = Vec::new();
        for (x, y) in &[(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
            qs.push(Quadrant::<2>::new(0, 1, *x, *y, 0, 0));
        }
        let trees = vec![Tree::from_quadrants(qs)];

        // With a single rank, still no ghosts.
        let conns = build_connectors::<2>(&trees, &serial_comm());
        assert!(conns.is_empty(), "all on one rank → no ghosts");
    }

    #[test]
    fn test_connector_owner_lookup() {
        let mut conn = Connector::empty();
        let key = MortonKey::from_coords::<2>(0, 1, 1, 0);
        conn.ghosts.push((key, 3, 0));
        assert!(conn.is_ghost(&key));
        assert_eq!(conn.owner(&key), Some(3));
        assert!(!conn.is_ghost(&MortonKey::ROOT));
    }
}
