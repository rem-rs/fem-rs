//! # H-matrix infrastructure (Phase 7)
//!
//! Provides the geometric data structures needed for H-matrix / BEM / fast
//! direct solvers:
//!
//! - [`BoundingBox`]      — axis-aligned bounding box in R^D
//! - [`ClusterTree`]      — recursive binary spatial partitioning of a DOF set
//! - [`BlockClusterTree`] — product tree of two cluster trees with admissibility
//!   labelling
//! - [`Admissibility`]    — pluggable admissibility criterion
//!
//! ## Design notes
//!
//! This module only builds the *tree structures*; it does **not** implement ACA,
//! low-rank compression, or H-matrix arithmetic.  That is intentional: the trees
//! are a reusable foundation for BEM, H²-matrix solvers, or FMM-based preconditioners.
//!
//! ### Complexity targets
//! | Operation             | Complexity           |
//! |-----------------------|----------------------|
//! | `ClusterTree::build`  | O(n log n)           |
//! | `BlockClusterTree`    | O(n log² n) pairs    |
//! | Memory (admissible)   | O(n log n)           |
//!
//! ### Example
//! ```rust
//! use fem_linalg::hmatrix::{ClusterTree, BlockClusterTree, StandardAdmissibility};
//!
//! // 2-D point cloud
//! let pts: Vec<[f64; 2]> = (0..64).map(|i| {
//!     let t = i as f64 / 64.0 * std::f64::consts::TAU;
//!     [t.cos(), t.sin()]
//! }).collect();
//!
//! let row_tree = ClusterTree::build(&pts, 8);
//! let col_tree = ClusterTree::build(&pts, 8);
//! let bct = BlockClusterTree::build(&row_tree, &col_tree,
//!     &StandardAdmissibility { eta: 2.0 });
//!
//! println!("admissible blocks  : {}", bct.admissible_count());
//! println!("inadmissible blocks: {}", bct.inadmissible_count());
//! ```

// ─── BoundingBox ─────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in R^D.
///
/// `D` is a compile-time constant dimension (1, 2, or 3 are typical).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox<const D: usize> {
    /// Component-wise minimum corner.
    pub min: [f64; D],
    /// Component-wise maximum corner.
    pub max: [f64; D],
}

impl<const D: usize> BoundingBox<D> {
    /// Construct a bounding box that contains a single point.
    #[inline]
    pub fn from_point(p: &[f64; D]) -> Self {
        BoundingBox { min: *p, max: *p }
    }

    /// Expand this box to also contain `p`.
    #[inline]
    pub fn expand(&mut self, p: &[f64; D]) {
        for d in 0..D {
            if p[d] < self.min[d] { self.min[d] = p[d]; }
            if p[d] > self.max[d] { self.max[d] = p[d]; }
        }
    }

    /// Merge two boxes (smallest containing box).
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for d in 0..D {
            if other.min[d] < out.min[d] { out.min[d] = other.min[d]; }
            if other.max[d] > out.max[d] { out.max[d] = other.max[d]; }
        }
        out
    }

    /// Diameter: maximum side length across all dimensions.
    #[inline]
    pub fn diameter(&self) -> f64 {
        (0..D).map(|d| self.max[d] - self.min[d])
              .fold(0.0_f64, f64::max)
    }

    /// Minimum distance between this box and `other`.
    ///
    /// Returns 0 if the boxes overlap.
    pub fn dist_to(&self, other: &Self) -> f64 {
        let mut d2 = 0.0_f64;
        for k in 0..D {
            let gap = f64::max(0.0, f64::max(self.min[k] - other.max[k],
                                              other.min[k] - self.max[k]));
            d2 += gap * gap;
        }
        d2.sqrt()
    }

    /// Index of the longest dimension (used for median split).
    pub fn longest_dim(&self) -> usize {
        (0..D).max_by(|&a, &b| {
            let la = self.max[a] - self.min[a];
            let lb = self.max[b] - self.min[b];
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
        }).unwrap_or(0)
    }
}

impl<const D: usize> BoundingBox<D> {
    /// Build the tightest bounding box for a set of points (index subset).
    pub fn for_indices(points: &[[f64; D]], indices: &[usize]) -> Option<Self> {
        let mut it = indices.iter().copied();
        let first = it.next()?;
        let mut bbox = BoundingBox::from_point(&points[first]);
        for i in it {
            bbox.expand(&points[i]);
        }
        Some(bbox)
    }
}

// ─── ClusterTree ─────────────────────────────────────────────────────────────

/// A node in the cluster tree.
///
/// Leaf nodes own a contiguous slice `[dof_start, dof_end)` of the
/// **permuted** DOF index array stored in [`ClusterTree::perm`].
/// Interior nodes own the union of their children's slices.
#[derive(Debug, Clone)]
pub struct ClusterNode<const D: usize> {
    /// Bounding box of all DOFs in this cluster.
    pub bbox:      BoundingBox<D>,
    /// Index range `[start, end)` into [`ClusterTree::perm`].
    pub dof_start: usize,
    pub dof_end:   usize,
    /// Child node indices (0 or 2 children; empty = leaf).
    pub children:  Vec<usize>,
}

impl<const D: usize> ClusterNode<D> {
    /// Number of DOFs in this cluster.
    #[inline]
    pub fn size(&self) -> usize { self.dof_end - self.dof_start }

    /// True if this is a leaf node.
    #[inline]
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }
}

/// Binary cluster tree built by recursive coordinate bisection.
///
/// Points are split along the longest bounding-box dimension at the median.
/// Recursion stops when a cluster contains at most `leaf_size` DOFs.
///
/// The permuted DOF ordering is stored in [`ClusterTree::perm`]; every leaf
/// `dof_start..dof_end` slice is contiguous in that array.
#[derive(Debug)]
pub struct ClusterTree<const D: usize> {
    /// All cluster nodes in BFS order (root = index 0).
    pub nodes: Vec<ClusterNode<D>>,
    /// Permutation of original DOF indices (length = number of DOFs).
    pub perm:  Vec<usize>,
}

impl<const D: usize> ClusterTree<D> {
    /// Build a cluster tree for `points` with leaves of at most `leaf_size` DOFs.
    ///
    /// # Panics
    /// Panics if `points` is empty or `leaf_size == 0`.
    pub fn build(points: &[[f64; D]], leaf_size: usize) -> Self {
        assert!(!points.is_empty(), "ClusterTree: empty point set");
        assert!(leaf_size > 0, "ClusterTree: leaf_size must be positive");

        let n = points.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut nodes: Vec<ClusterNode<D>> = Vec::with_capacity(2 * n / leaf_size + 1);

        // Recursive split stored iteratively via an explicit stack.
        // Each stack entry: (node_index_placeholder, start, end)
        // We first push a placeholder for the root, then process.
        let root_bbox = BoundingBox::for_indices(points, &perm).unwrap();
        nodes.push(ClusterNode {
            bbox:      root_bbox,
            dof_start: 0,
            dof_end:   n,
            children:  vec![],
        });

        let mut stack: Vec<usize> = vec![0]; // node indices to split

        while let Some(node_idx) = stack.pop() {
            let start = nodes[node_idx].dof_start;
            let end   = nodes[node_idx].dof_end;
            let size  = end - start;

            if size <= leaf_size {
                // Leaf: no further split needed.
                continue;
            }

            // Split along longest dimension at the median.
            let dim = nodes[node_idx].bbox.longest_dim();
            let slice = &mut perm[start..end];
            slice.sort_unstable_by(|&a, &b| {
                points[a][dim].partial_cmp(&points[b][dim])
                              .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mid = start + size / 2;

            // Build bounding boxes for the two halves.
            let left_bbox  = BoundingBox::for_indices(points, &perm[start..mid]).unwrap();
            let right_bbox = BoundingBox::for_indices(points, &perm[mid..end]).unwrap();

            let left_idx  = nodes.len();
            let right_idx = left_idx + 1;

            nodes.push(ClusterNode {
                bbox:      left_bbox,
                dof_start: start,
                dof_end:   mid,
                children:  vec![],
            });
            nodes.push(ClusterNode {
                bbox:      right_bbox,
                dof_start: mid,
                dof_end:   end,
                children:  vec![],
            });

            nodes[node_idx].children = vec![left_idx, right_idx];

            stack.push(right_idx);
            stack.push(left_idx);
        }

        ClusterTree { nodes, perm }
    }

    /// Root node.
    #[inline]
    pub fn root(&self) -> &ClusterNode<D> { &self.nodes[0] }

    /// Number of nodes in the tree.
    #[inline]
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Maximum depth of the tree (root = depth 0).
    pub fn max_depth(&self) -> usize {
        fn depth_of<const D: usize>(nodes: &[ClusterNode<D>], i: usize, d: usize) -> usize {
            if nodes[i].is_leaf() { return d; }
            nodes[i].children.iter()
                .map(|&c| depth_of(nodes, c, d + 1))
                .max()
                .unwrap_or(d)
        }
        depth_of(&self.nodes, 0, 0)
    }

    /// All leaf node indices (in tree order).
    pub fn leaf_indices(&self) -> Vec<usize> {
        self.nodes.iter().enumerate()
            .filter(|(_, n)| n.is_leaf())
            .map(|(i, _)| i)
            .collect()
    }
}

// ─── Admissibility ───────────────────────────────────────────────────────────

/// Pluggable admissibility criterion for H-matrix block clustering.
///
/// Implement this trait to change when a `(row_cluster, col_cluster)` pair
/// is treated as a low-rank (admissible) block vs. a dense block.
pub trait Admissibility<const D: usize> {
    /// Return `true` if the pair `(row, col)` is *admissible* (should be
    /// approximated as a low-rank block).
    fn is_admissible(&self, row: &ClusterNode<D>, col: &ClusterNode<D>) -> bool;
}

/// Standard η-admissibility: `min(diam_row, diam_col) ≤ η · dist(row, col)`.
///
/// * `η = 2.0` is a common default (larger η = more admissible blocks).
/// * Requires `dist > 0`; overlapping boxes are always inadmissible.
#[derive(Debug, Clone, Copy)]
pub struct StandardAdmissibility {
    /// Admissibility parameter η > 0.
    pub eta: f64,
}

impl<const D: usize> Admissibility<D> for StandardAdmissibility {
    fn is_admissible(&self, row: &ClusterNode<D>, col: &ClusterNode<D>) -> bool {
        let dist = row.bbox.dist_to(&col.bbox);
        if dist == 0.0 { return false; }
        let min_diam = f64::min(row.bbox.diameter(), col.bbox.diameter());
        min_diam <= self.eta * dist
    }
}

/// Strict admissibility: admissible only when the two boxes are well-separated
/// (dist ≥ diam_max).  Produces fewer admissible blocks but larger low-rank
/// ranks per block.
#[derive(Debug, Clone, Copy)]
pub struct StrictAdmissibility;

impl<const D: usize> Admissibility<D> for StrictAdmissibility {
    fn is_admissible(&self, row: &ClusterNode<D>, col: &ClusterNode<D>) -> bool {
        let dist = row.bbox.dist_to(&col.bbox);
        if dist == 0.0 { return false; }
        let max_diam = f64::max(row.bbox.diameter(), col.bbox.diameter());
        dist >= max_diam
    }
}

// ─── BlockClusterTree ────────────────────────────────────────────────────────

/// Classification of a `(row, col)` cluster pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Low-rank (far-field) block: can be approximated by ACA / SVD.
    Admissible,
    /// Dense (near-field) block: must be stored or computed exactly.
    Inadmissible,
}

/// A node in the block cluster tree.
#[derive(Debug, Clone)]
pub struct BlockClusterNode {
    /// Index into the *row* tree's node array.
    pub row_node: usize,
    /// Index into the *col* tree's node array.
    pub col_node: usize,
    /// Whether this block is admissible (low-rank) or inadmissible (dense).
    pub kind:     BlockKind,
    /// Child block indices (empty for leaf blocks).
    pub children: Vec<usize>,
}

impl BlockClusterNode {
    /// True if this is a leaf block.
    #[inline]
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }
}

/// Block cluster tree: product of two [`ClusterTree`]s with admissibility labels.
///
/// The tree is built by recursively splitting both row and column clusters
/// until a pair is admissible or both are leaves.
pub struct BlockClusterTree<const D: usize> {
    /// All block-cluster nodes.
    pub blocks: Vec<BlockClusterNode>,
    /// Reference to the row cluster tree (same lifetime as self; here we store
    /// node counts for verification).
    pub row_nodes: usize,
    pub col_nodes: usize,
}

impl<const D: usize> BlockClusterTree<D> {
    /// Build the block cluster tree.
    ///
    /// * `row_tree` — cluster tree for the row index set
    /// * `col_tree` — cluster tree for the column index set
    /// * `adm`      — admissibility criterion
    pub fn build<A>(
        row_tree: &ClusterTree<D>,
        col_tree: &ClusterTree<D>,
        adm:      &A,
    ) -> Self
    where
        A: Admissibility<D>,
    {
        let mut blocks: Vec<BlockClusterNode> = Vec::new();

        // Iterative BFS build via explicit stack.
        // Stack entries: (row_node_idx, col_node_idx, parent_block_idx_opt)
        let mut stack: Vec<(usize, usize, Option<usize>)> = vec![(0, 0, None)];

        while let Some((ri, ci, parent)) = stack.pop() {
            let rn = &row_tree.nodes[ri];
            let cn = &col_tree.nodes[ci];

            let kind = if adm.is_admissible(rn, cn) {
                BlockKind::Admissible
            } else {
                BlockKind::Inadmissible
            };

            let block_idx = blocks.len();
            blocks.push(BlockClusterNode {
                row_node: ri,
                col_node: ci,
                kind,
                children: vec![],
            });

            if let Some(p) = parent {
                blocks[p].children.push(block_idx);
            }

            // Recurse only if inadmissible and not both leaves.
            if kind == BlockKind::Inadmissible && !(rn.is_leaf() && cn.is_leaf()) {
                match (rn.is_leaf(), cn.is_leaf()) {
                    (false, false) => {
                        // Split both.
                        for &r_child in &row_tree.nodes[ri].children.clone() {
                            for &c_child in &col_tree.nodes[ci].children.clone() {
                                stack.push((r_child, c_child, Some(block_idx)));
                            }
                        }
                    }
                    (true, false) => {
                        // Row is leaf, split col only.
                        for &c_child in &col_tree.nodes[ci].children.clone() {
                            stack.push((ri, c_child, Some(block_idx)));
                        }
                    }
                    (false, true) => {
                        // Col is leaf, split row only.
                        for &r_child in &row_tree.nodes[ri].children.clone() {
                            stack.push((r_child, ci, Some(block_idx)));
                        }
                    }
                    (true, true) => { /* both leaves, inadmissible dense block */ }
                }
            }
        }

        BlockClusterTree {
            blocks,
            row_nodes: row_tree.node_count(),
            col_nodes: col_tree.node_count(),
        }
    }

    /// Number of admissible (low-rank) leaf blocks.
    pub fn admissible_count(&self) -> usize {
        self.leaf_blocks().filter(|b| b.kind == BlockKind::Admissible).count()
    }

    /// Number of inadmissible (dense) leaf blocks.
    pub fn inadmissible_count(&self) -> usize {
        self.leaf_blocks().filter(|b| b.kind == BlockKind::Inadmissible).count()
    }

    /// Iterator over leaf blocks.
    pub fn leaf_blocks(&self) -> impl Iterator<Item = &BlockClusterNode> {
        self.blocks.iter().filter(|b| b.is_leaf())
    }

    /// Total number of blocks (including internal nodes).
    #[inline]
    pub fn block_count(&self) -> usize { self.blocks.len() }

    /// Root block (row 0, col 0).
    #[inline]
    pub fn root(&self) -> &BlockClusterNode { &self.blocks[0] }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Generate n uniformly-spaced points on the unit circle.
    fn circle_points(n: usize) -> Vec<[f64; 2]> {
        (0..n).map(|i| {
            let t = i as f64 / n as f64 * TAU;
            [t.cos(), t.sin()]
        }).collect()
    }

    /// Uniform grid of n×n points in [0,1]².
    fn grid_points_2d(n: usize) -> Vec<[f64; 2]> {
        let mut pts = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                pts.push([i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]);
            }
        }
        pts
    }

    // ── BoundingBox tests ─────────────────────────────────────────────────────

    #[test]
    fn bbox_from_point_is_degenerate() {
        let bb = BoundingBox::from_point(&[1.0_f64, 2.0]);
        assert_eq!(bb.min, [1.0, 2.0]);
        assert_eq!(bb.max, [1.0, 2.0]);
        assert_eq!(bb.diameter(), 0.0);
    }

    #[test]
    fn bbox_expand_covers_all_points() {
        let mut bb = BoundingBox::from_point(&[0.0_f64, 0.0]);
        bb.expand(&[3.0, 0.0]);
        bb.expand(&[0.0, 4.0]);
        assert_eq!(bb.min, [0.0, 0.0]);
        assert_eq!(bb.max, [3.0, 4.0]);
        assert_eq!(bb.diameter(), 4.0); // max side
    }

    #[test]
    fn bbox_merge_is_commutative() {
        let a = BoundingBox { min: [0.0_f64, 0.0], max: [1.0, 2.0] };
        let b = BoundingBox { min: [0.5_f64, 1.0], max: [3.0, 1.5] };
        let ab = a.merge(&b);
        let ba = b.merge(&a);
        assert_eq!(ab.min, ba.min);
        assert_eq!(ab.max, ba.max);
        assert_eq!(ab.min, [0.0, 0.0]);
        assert_eq!(ab.max, [3.0, 2.0]);
    }

    #[test]
    fn bbox_dist_separated() {
        let a = BoundingBox { min: [0.0_f64, 0.0], max: [1.0, 1.0] };
        let b = BoundingBox { min: [3.0_f64, 0.0], max: [4.0, 1.0] };
        let d = a.dist_to(&b);
        assert!((d - 2.0).abs() < 1e-14, "expected dist=2, got {d}");
    }

    #[test]
    fn bbox_dist_overlapping_is_zero() {
        let a = BoundingBox { min: [0.0_f64, 0.0], max: [2.0, 2.0] };
        let b = BoundingBox { min: [1.0_f64, 1.0], max: [3.0, 3.0] };
        assert_eq!(a.dist_to(&b), 0.0);
    }

    #[test]
    fn bbox_longest_dim() {
        let bb = BoundingBox { min: [0.0_f64, 0.0, 0.0], max: [1.0, 5.0, 3.0] };
        assert_eq!(bb.longest_dim(), 1); // dim 1 has length 5
    }

    // ── ClusterTree tests ─────────────────────────────────────────────────────

    #[test]
    fn cluster_tree_leaf_size_respected() {
        let pts = circle_points(64);
        let leaf_size = 8;
        let tree = ClusterTree::build(&pts, leaf_size);
        for node in tree.nodes.iter().filter(|n| n.is_leaf()) {
            assert!(node.size() <= leaf_size,
                "leaf has {} DOFs, expected ≤ {}", node.size(), leaf_size);
        }
    }

    #[test]
    fn cluster_tree_perm_is_permutation() {
        let pts = circle_points(64);
        let tree = ClusterTree::build(&pts, 8);
        let mut sorted = tree.perm.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..pts.len()).collect();
        assert_eq!(sorted, expected, "perm is not a permutation of 0..n");
    }

    #[test]
    fn cluster_tree_dof_ranges_partition_root() {
        let n = 64;
        let pts = circle_points(n);
        let tree = ClusterTree::build(&pts, 8);
        // All leaf ranges together cover [0, n) without overlap.
        let mut covered = vec![false; n];
        for node in tree.nodes.iter().filter(|n| n.is_leaf()) {
            for idx in node.dof_start..node.dof_end {
                assert!(!covered[idx], "DOF {idx} covered twice");
                covered[idx] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "some DOFs not covered by any leaf");
    }

    #[test]
    fn cluster_tree_root_covers_all_dofs() {
        let pts = grid_points_2d(8); // 64 points
        let tree = ClusterTree::build(&pts, 4);
        assert_eq!(tree.root().dof_start, 0);
        assert_eq!(tree.root().dof_end, pts.len());
    }

    #[test]
    fn cluster_tree_depth_is_logarithmic() {
        let n = 128;
        let pts = circle_points(n);
        let tree = ClusterTree::build(&pts, 4);
        let depth = tree.max_depth();
        // Expect depth ≤ ceil(log2(n/leaf_size)) + 1 ≈ 6
        assert!(depth <= 10, "tree depth {depth} unexpectedly large");
        assert!(depth >= 3,  "tree depth {depth} unexpectedly small");
    }

    #[test]
    fn cluster_tree_1d_monotone_split() {
        // 1-D points: split must produce monotone coordinate halves.
        let pts: Vec<[f64; 1]> = (0..16).map(|i| [i as f64]).collect();
        let tree = ClusterTree::build(&pts, 4);
        // Every non-root node's bbox should be contained in its parent's.
        for (i, node) in tree.nodes.iter().enumerate() {
            for &c in &node.children {
                assert!(tree.nodes[c].bbox.min[0] >= node.bbox.min[0] - 1e-14);
                assert!(tree.nodes[c].bbox.max[0] <= node.bbox.max[0] + 1e-14);
                let _ = i; // suppress unused
            }
        }
    }

    #[test]
    fn cluster_tree_single_point_is_leaf() {
        let pts = vec![[0.5_f64, 0.5]];
        let tree = ClusterTree::build(&pts, 4);
        assert_eq!(tree.node_count(), 1);
        assert!(tree.root().is_leaf());
    }

    // ── Admissibility tests ───────────────────────────────────────────────────

    #[test]
    fn standard_admissibility_well_separated() {
        let adm = StandardAdmissibility { eta: 2.0 };
        let row: ClusterNode<2> = ClusterNode {
            bbox:      BoundingBox { min: [0.0, 0.0], max: [1.0, 1.0] },
            dof_start: 0, dof_end: 4, children: vec![],
        };
        let col: ClusterNode<2> = ClusterNode {
            bbox:      BoundingBox { min: [5.0, 0.0], max: [6.0, 1.0] },
            dof_start: 0, dof_end: 4, children: vec![],
        };
        // diam_row=1, dist=4 → 1 ≤ 2·4=8 → admissible
        assert!(adm.is_admissible(&row, &col));
    }

    #[test]
    fn standard_admissibility_overlapping_is_inadmissible() {
        let adm = StandardAdmissibility { eta: 2.0 };
        let row: ClusterNode<2> = ClusterNode {
            bbox:      BoundingBox { min: [0.0, 0.0], max: [2.0, 2.0] },
            dof_start: 0, dof_end: 4, children: vec![],
        };
        let col: ClusterNode<2> = ClusterNode {
            bbox:      BoundingBox { min: [1.0, 1.0], max: [3.0, 3.0] },
            dof_start: 0, dof_end: 4, children: vec![],
        };
        assert!(!adm.is_admissible(&row, &col));
    }

    #[test]
    fn strict_admissibility_subset_of_standard() {
        // Every strictly-admissible pair should also be standard-admissible.
        let pts = circle_points(32);
        let tree = ClusterTree::build(&pts, 4);
        let strict = StrictAdmissibility;
        let standard = StandardAdmissibility { eta: 1.0 };
        for i in 0..tree.nodes.len() {
            for j in 0..tree.nodes.len() {
                if strict.is_admissible(&tree.nodes[i], &tree.nodes[j]) {
                    assert!(standard.is_admissible(&tree.nodes[i], &tree.nodes[j]),
                        "strict-adm pair ({i},{j}) not standard-adm");
                }
            }
        }
    }

    // ── BlockClusterTree tests ────────────────────────────────────────────────

    #[test]
    fn bct_leaf_blocks_cover_all_dof_pairs() {
        let pts = circle_points(32);
        let tree = ClusterTree::build(&pts, 4);
        let bct = BlockClusterTree::build(&tree, &tree, &StandardAdmissibility { eta: 2.0 });

        // Sum of (row_size × col_size) over all leaf blocks must equal n².
        let n = pts.len();
        let total: usize = bct.leaf_blocks().map(|b| {
            let rs = tree.nodes[b.row_node].size();
            let cs = tree.nodes[b.col_node].size();
            rs * cs
        }).sum();
        assert_eq!(total, n * n,
            "leaf blocks cover {total} pairs, expected {}", n * n);
    }

    #[test]
    fn bct_has_admissible_and_inadmissible_blocks() {
        let pts = circle_points(64);
        let tree = ClusterTree::build(&pts, 4);
        let bct = BlockClusterTree::build(&tree, &tree, &StandardAdmissibility { eta: 2.0 });
        assert!(bct.admissible_count() > 0,   "no admissible blocks");
        assert!(bct.inadmissible_count() > 0, "no inadmissible blocks");
    }

    #[test]
    fn bct_more_admissible_dof_pairs_with_larger_eta() {
        // Larger η means more DOF pairs are treated as low-rank (admissible).
        // We compare the sum of (row_size × col_size) over admissible leaf blocks.
        let pts = grid_points_2d(8); // 64 points
        let tree = ClusterTree::build(&pts, 4);

        let adm_dof_pairs = |eta: f64| -> usize {
            let bct = BlockClusterTree::build(&tree, &tree,
                &StandardAdmissibility { eta });
            bct.leaf_blocks()
                .filter(|b| b.kind == BlockKind::Admissible)
                .map(|b| tree.nodes[b.row_node].size() * tree.nodes[b.col_node].size())
                .sum::<usize>()
        };

        let pairs_tight = adm_dof_pairs(0.5);
        let pairs_loose = adm_dof_pairs(4.0);
        assert!(pairs_loose >= pairs_tight,
            "larger eta should cover at least as many DOF pairs admissibly \
             (loose={pairs_loose}, tight={pairs_tight})");
    }

    #[test]
    fn bct_root_is_full_matrix() {
        let pts = circle_points(16);
        let tree = ClusterTree::build(&pts, 4);
        let bct = BlockClusterTree::build(&tree, &tree, &StandardAdmissibility { eta: 2.0 });
        let root = bct.root();
        assert_eq!(root.row_node, 0);
        assert_eq!(root.col_node, 0);
    }

    #[test]
    fn bct_3d_points_build_succeeds() {
        let pts: Vec<[f64; 3]> = (0..64).map(|i| {
            let t = i as f64 / 64.0 * TAU;
            [t.cos(), t.sin(), i as f64 / 64.0]
        }).collect();
        let tree = ClusterTree::build(&pts, 8);
        let bct = BlockClusterTree::build(&tree, &tree, &StandardAdmissibility { eta: 2.0 });
        assert!(bct.block_count() > 0);
        assert!(bct.admissible_count() + bct.inadmissible_count() > 0);
    }

    #[test]
    fn bct_asymmetric_row_col_trees() {
        // Row tree from 32 circle points, col tree from 16 circle points.
        let row_pts = circle_points(32);
        let col_pts = circle_points(16);
        let row_tree = ClusterTree::build(&row_pts, 4);
        let col_tree = ClusterTree::build(&col_pts, 4);
        let bct = BlockClusterTree::build(&row_tree, &col_tree,
            &StandardAdmissibility { eta: 2.0 });
        // Leaf block DOF coverage: sum(rs*cs) = n_row * n_col
        let total: usize = bct.leaf_blocks().map(|b| {
            row_tree.nodes[b.row_node].size() * col_tree.nodes[b.col_node].size()
        }).sum();
        assert_eq!(total, row_pts.len() * col_pts.len());
    }
}
