//! # P4est-style distributed forest for parallel AMR
//!
//! This module implements the core data structures for a distributed
//! forest-of-octrees (cf. p4est / deal.II), providing:
//!
//! - **Quadrant** — logical cells with Morton (Z-order) encoding and
//!   hierarchical split/merge.
//! - **Tree** — a local collection of quadrants owned by one MPI rank,
//!   stored in Morton order.
//! - **Forest** — the distributed forest across MPI ranks, with Morton-order
//!   prefix-sum partitioning.
//! - **Balance** — 2:1 balance constraint enforcement using logical
//!   coordinate adjacency.
//! - **Connector** — ghost quadrant tracking for cross-rank neighbour
//!   information.
//! - **Convert** — conversion between the logical forest representation
//!   and the physical [`Mesh<D>`] used by FEM assembly.
//! - **Solution** — solution vectors living directly on the forest's corner
//!   nodes, with prolongation and restriction.
//!
//! ## Design
//!
//! The forest treats the domain as a coarse grid of "trees", each of which
//! is independently refined via quadrant splits.  Quadrants are stored in
//! Morton (Z-order) key order so that:
//!
//! 1. A global Z-order traversal is a simple concatenation of each rank's
//!    local sequence.
//! 2. Neighbours in physical space are close in the Morton ordering
//!    (good spatial locality).
//! 3. Prefix-sum partitioning over the Morton order gives well-balanced
//!    partitions.
//!
//! ## Status
//!
//! This is Phase 1 of the AMR → deal.II alignment plan.  The implementation
//! supports serial (single-rank) operation.  Multi-rank MPI communication
//! for ghost discovery and load balancing will be extended in future phases.
//!
//! Files in this module:
//!
//! | File | Contents |
//! |------|----------|
//! | `quadrant.rs` | [`Quadrant<D>`], [`MortonKey`], split/merge, neighbour lookup |
//! | `tree.rs` | [`Tree<D>`] — local quadrant collection with refine/coarsen |
//! | `forest.rs` | [`Forest<D>`] — distributed forest with partitioning |
//! | `balance.rs` | 2:1 balance closure |
//! | `connector.rs` | Ghost quadrant connectors |
//! | `convert.rs` | Forest↔Mesh conversion |
//! | `solution.rs` | Solution vectors on the forest |

pub mod quadrant;
pub mod tree;
pub mod forest;
pub mod balance;
pub mod connector;
pub mod convert;
pub mod solution;

// Re-exports.
pub use quadrant::{MortonKey, Quadrant, neighbour_key};
pub use tree::Tree;
pub use forest::{Forest, ForestStats};
pub use balance::{balance_closure_2to1, enforce_2to1_local, forest_balance_2to1};
pub use connector::{Connector, build_connectors, update_connectors};
pub use convert::{TreeBoundingBox, forest_to_mesh, mesh_to_forest, default_tree_boxes};
pub use solution::ForestSolution;
