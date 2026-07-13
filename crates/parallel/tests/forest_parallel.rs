//! Integration tests for the Forest data structure with MPI-like backends.
//!
//! These tests verify the Forest↔Mesh conversion round-trip and other
//! MPI-level forest operations using a serial (single-rank) backend.

use fem_mesh::ElementType;
use fem_parallel::backend::native::SerialBackend;
use fem_parallel::forest::{
    default_tree_boxes, forest_to_mesh, mesh_to_forest, Forest, MortonKey, Quadrant,
};
use fem_parallel::Comm;

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
