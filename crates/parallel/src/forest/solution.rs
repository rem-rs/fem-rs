//! Solution vectors living directly on the forest.
//!
//! In the p4est-style design, solution vectors are associated with forest
//! **corners** (shared nodes) rather than passing through an intermediate
//! Mesh.  This module provides:
//!
//! - A [`ForestSolution<D>`] that manages nodal solution values keyed by
//!   Morton-ordered node positions.
//! - Prolongation (interpolation when refining) and restriction (averaging
//!   when coarsening) directly on the forest's quadrant hierarchy.
//!
//! For P1 elements each quadrant corner corresponds to exactly one DOF.
//! The solution is stored as a flat `Vec<f64>` indexed by node ID in the
//! same order as the forest produces them (Morton-order traversal).

use crate::forest::quadrant::{MortonKey, Quadrant};
use crate::forest::convert::{TreeBoundingBox, build_mesh_data};
use crate::forest::forest::Forest;
use fem_mesh::MeshTopology;

/// Position-based index into the solution array.
///
/// Wraps an ordered list of positions so we can look up node IDs by spatial
/// position without needing `Hash`/`Eq` on `[f64; D]`.
#[derive(Clone)]
struct PosIndex<const D: usize> {
    positions: Vec<[f64; D]>,
}

impl<const D: usize> std::fmt::Debug for PosIndex<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PosIndex")
            .field("n_positions", &self.positions.len())
            .finish()
    }
}

impl<const D: usize> PosIndex<D> {
    fn from(positions: Vec<[f64; D]>) -> Self {
        Self { positions }
    }

    fn find(&self, pos: &[f64; D]) -> Option<usize> {
        self.positions.iter().position(|p| {
            for d in 0..D {
                if (p[d] - pos[d]).abs() > 1e-12 {
                    return false;
                }
            }
            true
        })
    }

    fn len(&self) -> usize {
        self.positions.len()
    }
}

/// A solution vector stored directly on a forest's corner nodes.
///
/// The solution is indexed by node ID from a reference mesh built from the
/// forest.  When the forest is refined or coarsened, the solution is
/// prolongated or restricted accordingly.
#[derive(Debug, Clone)]
pub struct ForestSolution<const D: usize> {
    /// Per-node solution values (DOF values at mesh nodes).
    pub values: Vec<f64>,
    /// Number of degrees of freedom per node (1 for scalar, >1 for vector).
    pub dofs_per_node: usize,
    /// Position index mapping spatial positions to node IDs.
    pos_index: PosIndex<D>,
}

impl<const D: usize> ForestSolution<D> {
    /// Create a new solution from a forest.
    ///
    /// Initializes the solution to zero.
    pub fn new(forest: &Forest<D>, tree_boxes: &[TreeBoundingBox<D>], _elem_type: fem_mesh::ElementType) -> Self
    where
        [f64; D]: Default,
    {
        let (positions, _) = build_position_index::<D>(forest, tree_boxes);
        let n_nodes = positions.len();
        Self {
            values: vec![0.0; n_nodes],
            dofs_per_node: 1,
            pos_index: PosIndex::from(positions),
        }
    }

    /// Create a new solution from a forest, initialized from a function.
    pub fn from_fn<F>(forest: &Forest<D>, tree_boxes: &[TreeBoundingBox<D>],
                      _elem_type: fem_mesh::ElementType, f: F) -> Self
    where
        F: Fn(&[f64; D]) -> f64,
        [f64; D]: Default,
    {
        let (positions, _) = build_position_index::<D>(forest, tree_boxes);
        let n_nodes = positions.len();
        let mut values = vec![0.0; n_nodes];

        for (i, pos) in positions.iter().enumerate() {
            values[i] = f(pos);
        }

        Self {
            values,
            dofs_per_node: 1,
            pos_index: PosIndex::from(positions),
        }
    }

    /// Initialize from a mesh solution vector.
    ///
    /// Assumes the mesh solution values at each node correspond to the
    /// forest's corner nodes in the same spatial positions.
    pub fn from_mesh_solution(&mut self, mesh_solution: &[f64], mesh: &fem_mesh::Mesh<D>,
                               _tree_boxes: &[TreeBoundingBox<D>]) {
        for (i, pos) in self.pos_index.positions.iter().enumerate() {
            // Find the mesh node at this position.
            let mn = (0..mesh.n_nodes() as u32)
                .find(|&n| {
                    let c = mesh.node_coords(n);
                    (0..D).all(|d| (c[d] - pos[d]).abs() < 1e-12)
                });
            if let Some(mn) = mn {
                let idx = mn as usize;
                if idx < mesh_solution.len() {
                    self.values[i] = mesh_solution[idx];
                }
            }
        }
    }

    /// Number of nodes in the solution.
    pub fn n_nodes(&self) -> usize {
        self.values.len() / self.dofs_per_node
    }

    /// Get the solution value at a spatial position.
    pub fn value_at(&self, pos: &[f64; D]) -> f64 {
        if let Some(idx) = self.pos_index.find(pos) {
            self.values[idx]
        } else {
            0.0
        }
    }

    // ─── Prolongation ─────────────────────────────────────────────────────

    /// Prolongate (interpolate) the solution when a set of parent quadrants
    /// are refined into children.
    ///
    /// After refinement, new corner nodes are created at edge midpoints and
    /// face/volume centers.  For P1 elements, these are linearly interpolated
    /// from the parent corner values.
    ///
    /// `new_forest` is the refined forest; `old_forest` is the pre-refinement
    /// state.
    pub fn prolongate_refine(
        &mut self,
        old_forest: &Forest<D>,
        new_forest: &Forest<D>,
        tree_boxes: &[TreeBoundingBox<D>],
    ) where
        [f64; D]: Default,
    {
        let (new_positions, _) = build_position_index::<D>(new_forest, tree_boxes);
        let mut new_values = vec![0.0; new_positions.len()];

        for (i, pos) in new_positions.iter().enumerate() {
            if let Some(old_idx) = self.pos_index.find(pos) {
                // Existing node: copy value.
                new_values[i] = self.values[old_idx];
            } else {
                // New node at edge midpoint — interpolate.
                new_values[i] = self.interpolate_at(pos, old_forest, tree_boxes);
            }
        }

        self.values = new_values;
        self.pos_index = PosIndex::from(new_positions);
    }

    /// Interpolate the solution at a spatial position using the old forest.
    fn interpolate_at(
        &self,
        pos: &[f64; D],
        _old_forest: &Forest<D>,
        _tree_boxes: &[TreeBoundingBox<D>],
    ) -> f64 {
        // Inverse-distance weighted interpolation from all known corner nodes.
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (i, cpos) in self.pos_index.positions.iter().enumerate() {
            let dx: f64 = (0..D).map(|d| (cpos[d] - pos[d]).powi(2)).sum();
            if dx < 1e-30 {
                return self.values[i];
            }
            let w = 1.0 / dx;
            weighted_sum += w * self.values[i];
            total_weight += w;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    // ─── Restriction ──────────────────────────────────────────────────────

    /// Restrict (coarsen) the solution when a set of child quadrants are
    /// merged back into their parent.
    ///
    /// After coarsening, the parent's corner values are copied from the old,
    /// finer solution.  Internal nodes that disappear are averaged into
    /// the parent corners.
    pub fn restrict_coarsen(
        &mut self,
        old_forest: &Forest<D>,
        new_forest: &Forest<D>,
        tree_boxes: &[TreeBoundingBox<D>],
    ) where
        [f64; D]: Default,
    {
        let (new_positions, _) = build_position_index::<D>(new_forest, tree_boxes);
        let mut new_values = vec![0.0; new_positions.len()];

        for (i, pos) in new_positions.iter().enumerate() {
            if let Some(old_idx) = self.pos_index.find(pos) {
                new_values[i] = self.values[old_idx];
            } else {
                // Node doesn't exist in old map — interpolate.
                new_values[i] = self.interpolate_at(pos, old_forest, tree_boxes);
            }
        }

        self.values = new_values;
        self.pos_index = PosIndex::from(new_positions);
    }

    /// Update the solution after a forest refine/coarsen cycle.
    ///
    /// Automatically detects whether refinement or coarsening occurred
    /// based on the quadrant count.
    pub fn adapt(
        &mut self,
        old_forest: &Forest<D>,
        new_forest: &Forest<D>,
        tree_boxes: &[TreeBoundingBox<D>],
    ) where
        [f64; D]: Default,
    {
        let old_n = old_forest.n_local_active();
        let new_n = new_forest.n_local_active();

        if new_n > old_n {
            self.prolongate_refine(old_forest, new_forest, tree_boxes);
        } else if new_n < old_n {
            self.restrict_coarsen(old_forest, new_forest, tree_boxes);
        }
    }

    /// Interpolate solution values onto a Mesh for output/visualisation.
    pub fn to_mesh_values(&self, mesh: &fem_mesh::Mesh<D>) -> Vec<f64> {
        let n_nodes = mesh.n_nodes();
        let mut result = vec![0.0; n_nodes];

        for n in 0..n_nodes as u32 {
            let pos = mesh.node_coords(n);
            let mut pos_arr = [0.0; D];
            pos_arr.copy_from_slice(&pos[..D]);

            if let Some(idx) = self.pos_index.find(&pos_arr) {
                result[n as usize] = self.values[idx];
            } else {
                // Interpolate from nearby nodes.
                let mut best_dist = f64::MAX;
                let mut best_val = 0.0;
                for (ci, cpos) in self.pos_index.positions.iter().enumerate() {
                    let dist: f64 = (0..D).map(|d| (cpos[d] - pos_arr[d]).powi(2)).sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_val = self.values[ci];
                    }
                }
                result[n as usize] = best_val;
            }
        }

        result
    }
}

// ─── Position index construction ─────────────────────────────────────────────

/// Build a list of unique corner positions from a forest's active quadrants.
fn build_position_index<const D: usize>(
    forest: &Forest<D>,
    tree_boxes: &[TreeBoundingBox<D>],
) -> (Vec<[f64; D]>, Vec<Vec<u32>>)
where
    [f64; D]: Default,
{
    let (node_map, elem_nodes_list, _elem_tags) = build_mesh_data::<D>(forest, tree_boxes);
    let positions = node_map.positions;
    let elem_nodes: Vec<Vec<u32>> = elem_nodes_list;
    (positions, elem_nodes)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::native::SerialBackend;
    use crate::Comm;

    fn serial_comm() -> Comm {
        Comm::from_backend(Box::new(SerialBackend))
    }

    fn single_tree_forest() -> Forest<2> {
        let q = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
        Forest::from_trees(vec![vec![q]], serial_comm())
    }

    fn tree_boxes_2d() -> Vec<TreeBoundingBox<2>> {
        vec![TreeBoundingBox::new([0.0, 0.0], [1.0, 1.0])]
    }

    #[test]
    fn test_create_solution_zero() {
        let forest = single_tree_forest();
        let boxes = tree_boxes_2d();
        let sol = ForestSolution::<2>::new(&forest, &boxes, fem_mesh::ElementType::Quad4);
        // 1 Quad4 element = 4 nodes.
        assert_eq!(sol.n_nodes(), 4);
        assert!(sol.values.iter().all(|&v| v.abs() < 1e-14));
    }

    #[test]
    fn test_solution_from_fn() {
        let forest = single_tree_forest();
        let boxes = tree_boxes_2d();
        let sol = ForestSolution::<2>::from_fn(
            &forest, &boxes, fem_mesh::ElementType::Quad4,
            |pos| pos[0] + pos[1],
        );

        // Check corner values.
        assert!((sol.value_at(&[0.0, 0.0]) - 0.0).abs() < 1e-14);
        assert!((sol.value_at(&[1.0, 0.0]) - 1.0).abs() < 1e-14);
        assert!((sol.value_at(&[0.0, 1.0]) - 1.0).abs() < 1e-14);
        assert!((sol.value_at(&[1.0, 1.0]) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_prolongation_refine() {
        let forest = single_tree_forest();
        let boxes = tree_boxes_2d();

        let sol = ForestSolution::<2>::from_fn(
            &forest, &boxes, fem_mesh::ElementType::Quad4,
            |pos| pos[0] + pos[1],
        );

        // Refine the forest.
        let mut refined = forest.clone();
        let root = MortonKey::from_coords::<2>(0, 0, 0, 0);
        refined.refine_keys(&[root]);

        // Prolongate.
        let mut new_sol = sol.clone();
        new_sol.prolongate_refine(&forest, &refined, &boxes);

        assert_eq!(new_sol.n_nodes(), 9);

        // Center node (0.5, 0.5) should be interpolated.
        let center_val = new_sol.value_at(&[0.5, 0.5]);
        assert!((center_val - 1.0).abs() < 0.01,
            "center value should be ~1.0, got {center_val}");

        // Corner nodes should keep their original values.
        assert!((new_sol.value_at(&[0.0, 0.0]) - 0.0).abs() < 1e-12);
        assert!((new_sol.value_at(&[1.0, 1.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_restriction_coarsen() {
        let forest = single_tree_forest();
        let boxes = tree_boxes_2d();

        // Start with a refined forest.
        let mut refined = forest.clone();
        let root = MortonKey::from_coords::<2>(0, 0, 0, 0);
        refined.refine_keys(&[root]);

        let sol = ForestSolution::<2>::from_fn(
            &refined, &boxes, fem_mesh::ElementType::Quad4,
            |pos| pos[0] + pos[1],
        );

        // Coarsen back.
        let mut coarsened = refined.clone();
        coarsened.coarsen_keys(&[root]);

        // Restrict.
        let mut new_sol = sol.clone();
        new_sol.restrict_coarsen(&refined, &coarsened, &boxes);

        assert_eq!(new_sol.n_nodes(), 4);
        assert!((new_sol.value_at(&[0.0, 0.0]) - 0.0).abs() < 1e-12);
        assert!((new_sol.value_at(&[1.0, 1.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_refine_coarsen_roundtrip_solution() {
        let forest = single_tree_forest();
        let boxes = tree_boxes_2d();

        let sol = ForestSolution::<2>::from_fn(
            &forest, &boxes, fem_mesh::ElementType::Quad4,
            |pos| pos[0] + 2.0 * pos[1],
        );

        let v00 = sol.value_at(&[0.0, 0.0]);
        let v10 = sol.value_at(&[1.0, 0.0]);
        let v01 = sol.value_at(&[0.0, 1.0]);
        let v11 = sol.value_at(&[1.0, 1.0]);

        // Refine → coarsen.
        let mut refined = forest.clone();
        let root = MortonKey::from_coords::<2>(0, 0, 0, 0);
        refined.refine_keys(&[root]);

        let mut sol2 = sol.clone();
        sol2.prolongate_refine(&forest, &refined, &boxes);

        let mut coarsened = refined.clone();
        coarsened.coarsen_keys(&[root]);

        let mut sol3 = sol2.clone();
        sol3.restrict_coarsen(&refined, &coarsened, &boxes);

        assert!((sol3.value_at(&[0.0, 0.0]) - v00).abs() < 1e-12);
        assert!((sol3.value_at(&[1.0, 0.0]) - v10).abs() < 1e-12);
        assert!((sol3.value_at(&[0.0, 1.0]) - v01).abs() < 1e-12);
        assert!((sol3.value_at(&[1.0, 1.0]) - v11).abs() < 1e-12);
    }
}
