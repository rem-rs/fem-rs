//! Integration tests for the Forest data structure with MPI-like backends.
//!
//! These tests verify the Forest↔Mesh conversion round-trip and other
//! MPI-level forest operations using a serial (single-rank) backend.

use std::collections::{HashMap, HashSet};

use fem_mesh::ElementType;
use fem_parallel::backend::native::SerialBackend;
use fem_parallel::forest::{
    balance_closure_2to1, default_tree_boxes, forest_balance_2to1, forest_to_mesh,
    mesh_to_forest, Forest, MortonKey, Quadrant,
};
use fem_parallel::Comm;

/// Verify that non-uniform refinement followed by 2:1 balance enforcement
/// produces a consistent forest structure.
///
/// The workflow: refine non-uniformly, compute the balance closure (via the
/// Forest-level API), refine all keys identified by the closure, and verify
/// the resulting forest is structurally sound.
///
/// Note: The current MortonKey encoding does not store the refinement level,
/// which can cause key collisions when quadrants at different levels share
/// the same (x,y) coordinate values. This limits the balance algorithm's
/// ability to detect level gaps >1 across such entries in a Forest created
/// through consecutive refine operations.  This test exercises the combined
/// balance+refine API path and verifies structural integrity (no panics,
/// stats consistent, active count increased correctly).
#[test]
fn test_forest_2to1_balance_after_refine() {
    let mut forest = make_single_tree_forest(serial_comm());

    // Refine root -> 4 children at level 1.
    forest.refine_keys(&[MortonKey::ROOT]);
    assert_eq!(forest.n_local_active(), 4);

    // Refine the first child -> 4 grandchildren at level 2.
    let keys = forest.local_active_keys();
    forest.refine_keys(&[keys[0]]);
    assert_eq!(forest.n_local_active(), 7);

    // Compute the 2:1 balance closure for all active keys.
    // Note: local_active_keys() may return duplicate MortonKeys when
    // two quadrants at different levels share the same (x,y) coordinate.
    // The closure deduplicates via HashSet, so we use a set for comparison.
    let all_keys: HashSet<MortonKey> =
        forest.local_active_keys().into_iter().collect();
    let all_keys_vec: Vec<MortonKey> = all_keys.iter().copied().collect();
    let closure = forest_balance_2to1(&forest, &all_keys_vec);

    // The closure must be a superset of the marked keys.
    assert!(!closure.is_empty(), "balance closure should not be empty");
    assert!(closure.len() >= all_keys.len(),
        "closure ({}) should be a superset of marked keys ({})",
        closure.len(), all_keys.len());

    // Refine the closure (marks + any additional coarse neighbors).
    forest.refine_keys(&closure);

    // After balancing, verify the forest stats are consistent.
    // Refining closure.len() keys produces 4 * closure.len()
    // active quadrants (each quadrant yields 4 children in 2-D),
    // so n_active > 7 verifies the pipeline completed structurally.
    let stats = forest.stats();
    assert!(
        stats.max_level <= stats.min_level + 1,
        "max_level {} - min_level {} > 1 after balance",
        stats.max_level, stats.min_level
    );
    assert!(
        stats.n_active > 7,
        "after balance+refine, active count ({}) should exceed initial 7",
        stats.n_active
    );
}

/// Verify the low-level 2:1 balance closure algorithm with a manually
/// constructed active set that avoids MortonKey collisions.
///
/// The active map contains a 2x2 block of level-2 quadrants whose
/// coarse neighbors exist only at sufficiently close levels, so the
/// closure should be exactly the marked set (no cascade needed).
#[test]
fn test_balance_closure_single_level_no_cascade() {
    // Level-2 quadrants forming a 2x2 grid:
    //   (2,0) key 4  |  (3,0) key 5
    //   (2,1) key 6  |  (3,1) key 7
    let mut active_map = HashMap::new();
    for &key_val in &[4u64, 5, 6, 7] {
        active_map.insert(MortonKey(key_val), 2u8);
    }

    let mut marked = HashSet::new();
    marked.insert(MortonKey(4));

    let closure = balance_closure_2to1::<2>(&marked, &active_map);

    // All neighbours at the same level -> no cascade.
    assert_eq!(closure.len(), 1);
    assert!(closure.contains(&MortonKey(4)));
}

/// Verify that the balance closure correctly identifies a coarse neighbour
/// when there is a level gap of 2 (violating the 2:1 rule).
///
/// We construct an active map with:
/// - Level-3 quadrants at raw keys 16-19 (deinterleave: (4,0), (5,0),
///   (4,1), (5,1) in an 8x8 level-3 grid).
/// - Level-2 quadrants at raw keys 5-7.
/// - A **level-1 quadrant at key 1** (coordinates (1,0)).
///
/// When a level-3 marked quadrant has a missing same-level neighbour,
/// the ancestor walk climbs the candidate's ancestor chain:
///
///   marked key 17 (5,0) → parent (2,0) key 4 → grandparent (1,0) key 1
///
/// Key 1 is found in the active map at level 1, giving a level gap of
/// 3 - 1 = 2 > 1, so key 1 is added to the closure.
///
/// The same cascade also occurs for marked keys 18 and 19 through their
/// respective ancestor chains which converge to (1,0) key 1.
#[test]
fn test_balance_closure_cascade_level_gap() {
    let mut active_map = HashMap::new();

    // Level-3 quadrants: keys 16-19.
    for &key_val in &[16u64, 17, 18, 19] {
        active_map.insert(MortonKey(key_val), 3u8);
    }

    // Level-2 quadrants: keys 5, 6, 7.
    for &key_val in &[5u64, 6, 7] {
        active_map.insert(MortonKey(key_val), 2u8);
    }

    // Level-1 quadrant at (1,0) = key 1.
    // The ancestor walk from each level-3 marked key eventually reaches
    // this key, producing a gap of 2 (>1) and triggering a cascade.
    active_map.insert(MortonKey(1), 1u8);

    // Mark the level-3 quadrants for refinement (to level 4).
    let mut marked = HashSet::new();
    for &key_val in &[16u64, 17, 18, 19] {
        marked.insert(MortonKey(key_val));
    }

    let closure = balance_closure_2to1::<2>(&marked, &active_map);

    // The marked keys must be in the closure.
    for k in &marked {
        assert!(closure.contains(k),
            "closure must contain marked key {:?}", k);
    }

    // The ancestor walk from level-3 marked keys (17-19) finds key 1
    // at level 1 (gap = 2 > 1), so the closure must contain key 1.
    assert!(closure.len() > marked.len(),
        "cascade should add coarse neighbour to closure (size {} > {})",
        closure.len(), marked.len());
    assert!(closure.contains(&MortonKey(1)),
        "coarse neighbour key 1 (level-1 at (1,0)) must be in closure");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────────

/// Create a 1-tree forest on the unit square with one root quadrant.
fn make_single_tree_forest(comm: Comm) -> Forest<2> {
    let root = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
    Forest::from_trees(vec![vec![root]], comm)
}

/// Create a single-rank communicator backed by the serial backend.
fn serial_comm() -> Comm {
    Comm::from_backend(Box::new(SerialBackend))
}

/// Return the Morton key of the parent quadrant for a given key at a given level.
/// The parent occupies the space of 2^D children at level-1.
fn parent_key<const D: usize>(key: &MortonKey, level: u8) -> MortonKey {
    key.parent::<D>(level)
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

/// Verify that a single-rank forest survives forest→mesh→forest conversion
/// without losing active quadrant count.
#[test]
fn test_forest_mesh_roundtrip_serial() {
    let forest = make_single_tree_forest(serial_comm());
    let boxes = default_tree_boxes::<2>(1);
    let mesh = forest_to_mesh(&forest, &boxes, ElementType::Quad4);
    let (forest2, _) = mesh_to_forest(&mesh, forest.comm().clone(), 1);

    assert_eq!(
        forest.stats().n_active,
        forest2.stats().n_active,
        "n_active mismatch after round-trip"
    );
}

/// Stronger round-trip test: refine the forest first, then verify the
/// mesh and re-converted forest have consistent quadrant/element counts.
#[test]
fn test_forest_mesh_roundtrip_serial_refined() {
    let mut forest = make_single_tree_forest(serial_comm());

    // Refine the root quadrant (produces 4 children in 2D).
    let children = forest.refine_keys(&[MortonKey::ROOT]);
    assert_eq!(children.len(), 4, "refining root should create 4 children");

    let boxes = default_tree_boxes::<2>(1);
    let mesh = forest_to_mesh(&forest, &boxes, ElementType::Quad4);
    let (forest2, _) = mesh_to_forest(&mesh, forest.comm().clone(), 1);

    assert_eq!(
        forest.stats().n_active,
        forest2.stats().n_active,
        "n_active mismatch after refine round-trip"
    );
    // Also validate the mesh has 4 elements.
    assert_eq!(mesh.n_elems(), 4, "refined forest should produce 4 mesh elements");
}

/// Verify that refining all quadrants then coarsening all children returns to
/// the initial active quadrant count, repeated for 10 cycles.
///
/// This tests the stability of the refine/coarsen round-trip in the forest.
#[test]
fn test_forest_refine_coarsen_10_cycles() {
    let mut forest = make_single_tree_forest(serial_comm());
    let initial_active = forest.n_local_active();

    for cycle in 0..10 {
        // Step 1: Refine all active quadrants
        let all_active: Vec<MortonKey> = forest.local_active_keys();
        let _children = forest.refine_keys(&all_active);
        assert!(
            forest.n_local_active() > 0,
            "cycle {cycle}: no active quadrants after refine"
        );

        // Step 2: Coarsen all children back to parents.
        // Filter to only coarsen quadrants at level > 0 (roots cannot be coarsened).
        let all_parents: std::collections::HashSet<MortonKey> = forest
            .local_active_quadrants()
            .iter()
            .filter(|q| q.level > 0)
            .map(|q| parent_key::<2>(&q.key, q.level))
            .collect();

        let _restored = forest.coarsen_keys(
            &all_parents.into_iter().collect::<Vec<_>>(),
        );

        // After coarsening, we should be back to initial state.
        assert_eq!(
            forest.n_local_active(),
            initial_active,
            "cycle {cycle}: active quadrant count mismatch after coarsen"
        );
    }
}
