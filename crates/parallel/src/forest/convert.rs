//! Forest↔Mesh conversion layer.
//!
//! Converts between the logical forest representation and the physical
//! [`Mesh<D>`] representation used by the FEM assembly.
//!
//! The conversion maps each active quadrant's logical coordinates to physical
//! node coordinates via an affine transformation of the tree's bounding box.

use fem_mesh::{Mesh, MeshTopology};
use fem_core::{ElemId, NodeId};

use crate::forest::quadrant::{MortonKey, Quadrant};
use crate::forest::forest::Forest;

// ─── Bounding box per tree ───────────────────────────────────────────────────

/// Describes the physical bounding box of a coarse tree cell.
#[derive(Debug, Clone, Copy)]
pub struct TreeBoundingBox<const D: usize> {
    /// Lower corner of the bounding box in physical coordinates.
    pub origin: [f64; D],
    /// Extent (size) of the bounding box in each dimension.
    pub extent: [f64; D],
}

impl<const D: usize> TreeBoundingBox<D> {
    /// Create a bounding box from origin and extent.
    pub fn new(origin: [f64; D], extent: [f64; D]) -> Self {
        Self { origin, extent }
    }

    /// Create a unit-square bounding box for a single tree.
    pub fn unit_square() -> Self where [f64; D]: Default {
        Self {
            origin: [0.0; D],
            extent: [1.0; D],
        }
    }

    /// Map a logical quadrant coordinate to physical coordinates.
    ///
    /// A quadrant at level `L` with logical coordinates `(x, y)` occupies
    /// the physical region:
    ///   `origin + extent * (x / 2^L, y / 2^L, ...)`
    /// to
    ///   `origin + extent * ((x+1) / 2^L, (y+1) / 2^L, ...)`
    pub fn quadrant_corner(&self, key: &MortonKey, level: u8) -> [f64; D] {
        let (lx, ly, lz) = key.to_coords::<D>();
        let inv_2l = 1.0 / (1u64 << level) as f64;

        let mut corner = self.origin;
        corner[0] += self.extent[0] * lx as f64 * inv_2l;
        if D > 1 {
            corner[1] += self.extent[1] * ly as f64 * inv_2l;
        }
        if D > 2 {
            corner[2] += self.extent[2] * lz as f64 * inv_2l;
        }
        corner
    }

    /// Compute the physical size of a quadrant at a given level.
    pub fn quadrant_size(&self, level: u8) -> [f64; D] {
        let inv_2l = 1.0 / (1u64 << level) as f64;
        let mut size = [0.0; D];
        for d in 0..D {
            size[d] = self.extent[d] * inv_2l;
        }
        size
    }
}

// ─── Default bounding boxes ───────────────────────────────────────────────────

/// Default bounding boxes for a unit-square / unit-cube forest with
/// `n_trees` regular subdivision along each axis.
pub fn default_tree_boxes<const D: usize>(n_trees_per_dim: usize) -> Vec<TreeBoundingBox<D>> {
    let n_total = n_trees_per_dim.pow(D as u32);
    let inv = 1.0 / n_trees_per_dim as f64;

    let mut boxes = Vec::with_capacity(n_total);
    for idx in 0..n_total {
        let mut origin = [0.0; D];
        let mut remaining = idx;
        for d in 0..D {
            let coord = remaining % n_trees_per_dim;
            origin[d] = coord as f64 * inv;
            remaining /= n_trees_per_dim;
        }
        boxes.push(TreeBoundingBox {
            origin,
            extent: [inv; D],
        });
    }
    boxes
}

// ─── Node map helper (avoids f64 Hash/Eq issues) ────────────────────────────

/// A position → NodeId map built from [f64; D] positions.
/// Uses linear search, suitable for the modest node counts in a mesh conversion.
pub(crate) struct NodeMap<const D: usize> {
    pub(crate) positions: Vec<[f64; D]>,
    coords: Vec<f64>,
    next_id: NodeId,
}

impl<const D: usize> NodeMap<D> {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            coords: Vec::new(),
            next_id: 0,
        }
    }

    #[allow(dead_code)]
    fn n_nodes(&self) -> usize {
        self.next_id as usize
    }

    /// Find or create a node at the given position.
    fn get_or_create(&mut self, pos: [f64; D]) -> NodeId {
        for (i, p) in self.positions.iter().enumerate() {
            if pos_equal::<D>(p, &pos) {
                return i as NodeId;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.positions.push(pos);
        self.coords.extend_from_slice(&pos);
        id
    }

    /// Get position of a node ID (for testing).
    #[allow(dead_code)]
    fn position_of(&self, id: NodeId) -> Option<&[f64; D]> {
        self.positions.get(id as usize)
    }
}

fn pos_equal<const D: usize>(a: &[f64; D], b: &[f64; D]) -> bool {
    for d in 0..D {
        if (a[d] - b[d]).abs() > 1e-14 {
            return false;
        }
    }
    true
}

// ─── Forest to Mesh ───────────────────────────────────────────────────────────

/// Collect all corner positions from active quadrants of a forest.
///
/// Returns the unique positions and, for each active quadrant, the node IDs
/// of its corners.
pub(crate) fn build_mesh_data<const D: usize>(
    forest: &Forest<D>,
    tree_boxes: &[TreeBoundingBox<D>],
) -> (NodeMap<D>, Vec<Vec<NodeId>>, Vec<i32>)
where
    [f64; D]: Default,
{
    let mut node_map = NodeMap::<D>::new();
    let mut elem_nodes_list: Vec<Vec<NodeId>> = Vec::new();
    let mut elem_tags = Vec::new();

    for (tree_idx, tree) in forest.trees().iter().enumerate() {
        let bbox = if tree_idx < tree_boxes.len() {
            &tree_boxes[tree_idx]
        } else {
            &TreeBoundingBox::unit_square()
        };

        for q in tree.quadrants() {
            if !q.is_active {
                continue;
            }

            let qsize = bbox.quadrant_size(q.level);
            let corner = bbox.quadrant_corner(&q.key, q.level);

            // Generate corner nodes: iterate over all 2^D corners.
            let n_corners = 1 << D;
            let mut elem_nodes = Vec::with_capacity(n_corners);
            for ci in 0..n_corners {
                let mut pos = corner;
                for d in 0..D {
                    if (ci >> d) & 1 != 0 {
                        pos[d] += qsize[d];
                    }
                }
                elem_nodes.push(node_map.get_or_create(pos));
            }

            elem_nodes_list.push(elem_nodes);
            elem_tags.push(q.tag);
        }
    }

    (node_map, elem_nodes_list, elem_tags)
}

/// Convert a forest (and bounding boxes) to a Mesh for FEM assembly.
///
/// Each active quadrant is converted to a mesh element with the appropriate
/// element type (Quad4 for D=2, Hex8 for D=3).  Shared nodes between adjacent
/// quadrants are merged using position comparison.
///
/// # Arguments
///
/// * `forest` — The forest to convert.
/// * `tree_boxes` — Physical bounding boxes for each tree in the forest,
///   indexed by tree order.
/// * `elem_type` — The element type to use for mesh elements.
///
/// # Returns
///
/// A `Mesh<D>` containing all active quadrants as elements.
pub fn forest_to_mesh<const D: usize>(
    forest: &Forest<D>,
    tree_boxes: &[TreeBoundingBox<D>],
    elem_type: fem_mesh::ElementType,
) -> Mesh<D>
where
    [f64; D]: Default,
{
    let (node_map, elem_nodes_list, elem_tags) = build_mesh_data::<D>(forest, tree_boxes);

    // Build flat connectivity.
    let npe = match D {
        2 => 4,  // Quad4
        3 => 8,  // Hex8
        _ => 0,
    };
    let mut conn = Vec::with_capacity(elem_nodes_list.len() * npe);
    for nodes in &elem_nodes_list {
        conn.extend_from_slice(nodes);
    }

    // Determine face type.
    let face_type = match D {
        2 => fem_mesh::ElementType::Line2,
        3 => fem_mesh::ElementType::Quad4,
        _ => fem_mesh::ElementType::Line2,
    };

    Mesh {
        coords: node_map.coords,
        conn,
        elem_tags,
        elem_type,
        face_conn: Vec::new(),
        face_tags: Vec::new(),
        face_type,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(),
        edge_to_elem: Vec::new(),
        geometry: None,
    }
}

// ─── Mesh to Forest ───────────────────────────────────────────────────────────

/// Build a forest from an existing uniform mesh.
///
/// Each element of the mesh becomes a level-0 tree with a single root quadrant.
/// The bounding boxes are derived from the mesh's element coordinates.
///
/// The forest is then partitioned across MPI ranks.
pub fn mesh_to_forest<const D: usize>(
    mesh: &Mesh<D>,
    comm: crate::Comm,
    n_trees_per_dim: usize,
) -> (Forest<D>, Vec<TreeBoundingBox<D>>)
where
    [f64; D]: Default,
{
    let n_trees = n_trees_per_dim.pow(D as u32);
    let mut tree_quadrants: Vec<Vec<Quadrant<D>>> = Vec::with_capacity(n_trees);
    let tree_boxes = default_tree_boxes::<D>(n_trees_per_dim);

    for _ in 0..n_trees {
        tree_quadrants.push(Vec::new());
    }

    let n_elems = mesh.n_elems();
    for e in 0..n_elems as ElemId {
        let centroid = element_centroid(mesh, e);
        let tree_idx = find_enclosing_tree::<D>(&centroid, &tree_boxes)
            .unwrap_or(0);

        let q = Quadrant::<D>::new(
            tree_idx as u32,
            0,
            (tree_idx % n_trees_per_dim) as u32,
            ((tree_idx / n_trees_per_dim) % n_trees_per_dim) as u32,
            (tree_idx / (n_trees_per_dim * n_trees_per_dim)) as u32,
            mesh.elem_tags[e as usize],
        );
        tree_quadrants[tree_idx].push(q);
    }

    let forest = Forest::from_trees(tree_quadrants, comm);
    (forest, tree_boxes)
}

/// Compute the centroid of a mesh element.
fn element_centroid<const D: usize>(mesh: &Mesh<D>, elem: ElemId) -> [f64; D] {
    let nodes = mesh.elem_nodes(elem);
    let npe = nodes.len();
    let mut centroid = [0.0; D];
    for &n in nodes {
        let coords = mesh.node_coords(n);
        for d in 0..D {
            centroid[d] += coords[d];
        }
    }
    for d in 0..D {
        centroid[d] /= npe as f64;
    }
    centroid
}

/// Find the tree whose bounding box contains a given point.
fn find_enclosing_tree<const D: usize>(
    point: &[f64; D],
    boxes: &[TreeBoundingBox<D>],
) -> Option<usize> {
    for (i, bbox) in boxes.iter().enumerate() {
        let mut inside = true;
        for d in 0..D {
            if point[d] < bbox.origin[d] || point[d] > bbox.origin[d] + bbox.extent[d] {
                inside = false;
                break;
            }
        }
        if inside {
            return Some(i);
        }
    }
    None
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

    #[test]
    fn test_quadrant_corner() {
        let bbox = TreeBoundingBox::new([0.0, 0.0], [1.0, 1.0]);

        // Root quadrant (level 0) corner is at origin.
        let key = MortonKey::from_coords::<2>(0, 0, 0, 0);
        let corner = bbox.quadrant_corner(&key, 0);
        assert!((corner[0] - 0.0).abs() < 1e-14);
        assert!((corner[1] - 0.0).abs() < 1e-14);

        // Level 2 quadrant at (1, 1): corner at (0.25, 0.25).
        let key = MortonKey::from_coords::<2>(0, 1, 1, 0);
        let corner = bbox.quadrant_corner(&key, 2);
        assert!((corner[0] - 0.25).abs() < 1e-14);
        assert!((corner[1] - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_forest_to_mesh_single_quadrant() {
        let forest = single_tree_forest();
        let boxes = vec![TreeBoundingBox::new([0.0, 0.0], [1.0, 1.0])];
        let mesh = forest_to_mesh(&forest, &boxes, fem_mesh::ElementType::Quad4);

        // One element (Quad4) with 4 nodes.
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.conn.len(), 4);
    }

    #[test]
    fn test_forest_to_mesh_refined() {
        let mut forest = single_tree_forest();
        let root = MortonKey::from_coords::<2>(0, 0, 0, 0);
        forest.refine_keys(&[root]);

        let boxes = vec![TreeBoundingBox::new([0.0, 0.0], [1.0, 1.0])];
        let mesh = forest_to_mesh(&forest, &boxes, fem_mesh::ElementType::Quad4);

        // 4 children = 4 elements.
        assert_eq!(mesh.n_elems(), 4);
        // Should have 9 nodes: 2x2 grid of children = 3x3 grid of nodes.
        assert_eq!(mesh.n_nodes(), 9);
    }

    #[test]
    fn test_forest_to_mesh_multiple_trees() {
        let trees: Vec<Vec<Quadrant<2>>> = (0u32..4)
            .map(|i| {
                vec![Quadrant::<2>::new(i, 0, 0, 0, 0, i as i32)]
            })
            .collect();
        let forest = Forest::from_trees(trees, serial_comm());
        let boxes = default_tree_boxes::<2>(2);
        let mesh = forest_to_mesh(&forest, &boxes, fem_mesh::ElementType::Quad4);

        // 4 trees × 1 quadrant each = 4 elements.
        assert_eq!(mesh.n_elems(), 4);
        // 3×3 grid of nodes = 9 nodes (shared at boundaries).
        assert_eq!(mesh.n_nodes(), 9);
    }

    #[test]
    fn test_forest_to_mesh_shared_nodes() {
        let mut forest = single_tree_forest();
        let root = MortonKey::from_coords::<2>(0, 0, 0, 0);
        forest.refine_keys(&[root]);

        let boxes = vec![TreeBoundingBox::new([0.0, 0.0], [1.0, 1.0])];
        let mesh = forest_to_mesh(&forest, &boxes, fem_mesh::ElementType::Quad4);

        // The center node (0.5, 0.5) should be shared by all 4 elements.
        let center_coords = [0.5, 0.5];
        let elems_with_center = (0..mesh.n_elems() as ElemId)
            .filter(|&e| {
                let nodes = mesh.elem_nodes(e);
                nodes.iter().any(|&n| {
                    let c = mesh.node_coords(n);
                    (c[0] - center_coords[0]).abs() < 1e-14
                        && (c[1] - center_coords[1]).abs() < 1e-14
                })
            })
            .count();
        assert_eq!(elems_with_center, 4,
            "all 4 elements should share the center node");
    }

    #[test]
    fn test_default_tree_boxes_2d() {
        let boxes = default_tree_boxes::<2>(2);
        assert_eq!(boxes.len(), 4);
        assert!((boxes[0].origin[0] - 0.0).abs() < 1e-14);
        assert!((boxes[0].origin[1] - 0.0).abs() < 1e-14);
        assert!((boxes[1].origin[0] - 0.5).abs() < 1e-14);
        assert!((boxes[1].origin[1] - 0.0).abs() < 1e-14);
        assert!((boxes[3].origin[0] - 0.5).abs() < 1e-14);
        assert!((boxes[3].origin[1] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_forest_to_mesh_3d() {
        let q = Quadrant::<3>::new(0, 0, 0, 0, 0, 0);
        let forest = Forest::from_trees(vec![vec![q]], serial_comm());
        let boxes = vec![TreeBoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
        let mesh = forest_to_mesh(&forest, &boxes, fem_mesh::ElementType::Hex8);

        assert_eq!(mesh.n_elems(), 1);
        // Hex8: 8 nodes.
        assert_eq!(mesh.n_nodes(), 8);
    }

    #[test]
    fn test_find_enclosing_tree() {
        let boxes = default_tree_boxes::<2>(2);
        // Point in the first quadrant.
        let pt = [0.25, 0.25];
        let idx = find_enclosing_tree::<2>(&pt, &boxes);
        assert_eq!(idx, Some(0));

        // Point in the fourth quadrant.
        let pt = [0.75, 0.75];
        let idx = find_enclosing_tree::<2>(&pt, &boxes);
        assert_eq!(idx, Some(3));
    }
}
