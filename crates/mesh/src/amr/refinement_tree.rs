//! Lightweight refinement tree for tracking parent-child relationships
//! across multiple levels of non-conforming AMR.
//!
//! This type is designed to work alongside any NCState variant (NCState,
//! NCState3D, NCStateQuad, NCStateHex, NCStatePrism, NCStatePyramid) to
//! provide ancestor/sibling/element-level queries after multi-level refinement.
//!
//! The tree uses snapshot-based history (mirroring NCState's own snapshot
//! approach) so that `record_derefine` correctly restores the previous
//! refinement topology — including the identity reshuffle that happens when
//! children (whose `ElemId` positions may overlap with parent‑mesh positions)
//! are removed and original elements are restored.

use std::collections::{HashMap, HashSet};
use fem_core::{ElemId, NodeId};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::bisect::edge_key;

// ─── Helpers ────────────────────────────────────────────────────────────────────

/// Local (apex‑independent) edge index pairs for each supported element type.
/// The returned slice contains each edge of a reference element as a pair of
/// **local** node indices, suitable for iterating over `mesh.elem_nodes(e)`.
fn local_edges(et: ElementType) -> &'static [(usize, usize)] {
    match et {
        ElementType::Tri3 => &[
            (0, 1), (1, 2), (2, 0),
        ],
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => &[
            (0, 1), (1, 2), (2, 3), (3, 0),
        ],
        ElementType::Tet4 | ElementType::Tet10 => &[
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3),
        ],
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => &[
            // bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            // top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            // vertical
            (0, 4), (1, 5), (2, 6), (3, 7),
        ],
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => &[
            // bottom triangle
            (0, 1), (1, 2), (2, 0),
            // vertical
            (0, 3), (1, 4), (2, 5),
            // top triangle
            (3, 4), (4, 5), (5, 3),
        ],
        ElementType::Pyramid5 | ElementType::Pyramid13 => &[
            // base
            (0, 1), (1, 2), (2, 3), (3, 0),
            // apex
            (0, 4), (1, 4), (2, 4), (3, 4),
        ],
        _ => &[],
    }
}

// ─── Snapshot ───────────────────────────────────────────────────────────────────

/// A snapshot of the refinement tree taken before a refinement step, allowing
/// exact rollback during `record_derefine`.
#[derive(Debug, Clone)]
struct RefinementTreeSnapshot {
    parent:   HashMap<ElemId, ElemId>,
    children: HashMap<ElemId, Vec<ElemId>>,
    level:    HashMap<ElemId, u8>,
}

// ─── RefinementTree ─────────────────────────────────────────────────────────────

/// Tracks parent–child relationships across refinement levels.
///
/// **Usage**
///
/// ```rust,ignore
/// let mut tree = RefinementTree::new();
/// // Optionally initialise level 0 for all initial elements:
/// tree.init(n_elems);
///
/// // After every NCState::refine call:
/// let (new_mesh, ..) = nc_state.refine(&mesh, &marked);
/// tree.record_refine(mesh.n_elems(), &marked, 4 /* Tri3/Quad4 */);
///
/// // After every NCState::derefine_last call:
/// if let Some((old_mesh, ..)) = nc_state.derefine_last() {
///     tree.record_derefine();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RefinementTree {
    /// Maps child element ID → parent element ID.
    parent: HashMap<ElemId, ElemId>,
    /// Maps parent element ID → child element IDs.
    children: HashMap<ElemId, Vec<ElemId>>,
    /// Maps element ID → refinement level (0 = initial mesh).
    level: HashMap<ElemId, u8>,
    /// History snapshots for rollback-based derefinement.
    snapshots: Vec<RefinementTreeSnapshot>,
}

impl RefinementTree {
    /// Create an empty refinement tree.
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            children: HashMap::new(),
            level: HashMap::new(),
            snapshots: Vec::new(),
        }
    }

    /// Initialise level 0 for all initial mesh elements (idempotent).
    ///
    /// Call this once after constructing the initial (conforming) mesh.
    pub fn init(&mut self, n_elems: usize) {
        for e in 0..n_elems as ElemId {
            self.level.entry(e).or_insert(0);
        }
    }

    /// Whether at least one rollback (derefine) step is available.
    pub fn can_derefine(&self) -> bool {
        !self.snapshots.is_empty()
    }

    // ─── Recording refinement ───────────────────────────────────────────────

    /// Record that a refinement step was performed.
    ///
    /// * `n_elems_before` — element count in the mesh *before* refinement.
    /// * `marked`         — elements that were refined.
    /// * `n_children`     — number of children per refined element
    ///   (e.g. 4 for Tri3/Quad4, 8 for Tet4/Hex8).
    ///
    /// Children are assigned `level = parent_level + 1`.  The tree takes a
    /// snapshot before recording so that `record_derefine` can roll back.
    pub fn record_refine(&mut self, n_elems_before: usize, marked: &[ElemId], n_children: usize) {
        // Push snapshot for rollback.
        self.snapshots.push(RefinementTreeSnapshot {
            parent:   self.parent.clone(),
            children: self.children.clone(),
            level:    self.level.clone(),
        });

        if marked.is_empty() || n_children == 0 {
            return;
        }

        let marked_set: HashSet<ElemId> = marked.iter().copied().collect();

        // ── Prefix-sum: number of marked elements *before* each position ──
        let mut marked_before = vec![0usize; n_elems_before];
        let mut count = 0usize;
        for e in 0..n_elems_before as ElemId {
            marked_before[e as usize] = count;
            if marked_set.contains(&e) {
                count += 1;
            }
        }

        // ── Pre-compute carry-forward levels ────────────────────────
        // Save old levels for unrefined elements *before* the children loop
        // overwrites level entries at shared positions.
        let extra_per_refine = n_children.saturating_sub(1);
        let mut carry_levels: Vec<(ElemId, u8)> = Vec::new();
        for e in 0..n_elems_before as ElemId {
            if !marked_set.contains(&e) {
                let old_level = self.level.get(&e).copied().unwrap_or(0);
                let new_pos = e as usize + marked_before[e as usize] * extra_per_refine;
                carry_levels.push((new_pos as ElemId, old_level));
            }
        }

        // ── Record parent → children and child → parent ─────────────────
        for &e in marked {
            let parent_level = self.level.get(&e).copied().unwrap_or(0);
            let start = e as usize + marked_before[e as usize] * extra_per_refine;

            let children: Vec<ElemId> = (0..n_children)
                .map(|i| (start + i) as ElemId)
                .collect();

            self.children.insert(e, children.clone());
            for &child in &children {
                // Remove any stale parent entry from a previous refinement
                // level, so that the new relationship (or absence thereof
                // for self-loops) is authoritative.
                self.parent.remove(&child);

                // When a child occupies the same ElemId as its parent
                // (common in multi-level refinement), skip the parent map
                // entry to avoid a self-loop.  The caller must use the
                // snapshot history to trace ancestry across levels.
                if child != e {
                    self.parent.insert(child, e);
                }
                self.level.insert(child, parent_level + 1);
            }
        }

        // ── Propagate levels for carry-forward (unrefined) elements ──
        // An unrefined element at old position `e` shifts to a new position
        // based on how many refined elements (and their extra children) preceded
        // it.  We copy its old level to the new position so that multi-level
        // refinement doesn't lose the level for later queries and balance checks.
        for (pos, level) in carry_levels {
            self.level.insert(pos, level);
        }
    }

    /// Roll back the most recent refinement step.
    ///
    /// Returns `true` if a snapshot was restored, `false` if no history exists.
    pub fn record_derefine(&mut self) -> bool {
        if let Some(snap) = self.snapshots.pop() {
            self.parent   = snap.parent;
            self.children = snap.children;
            self.level    = snap.level;
            true
        } else {
            false
        }
    }

    // ─── Query methods ─────────────────────────────────────────────────────

    /// Parent of `elem`, or `None` if it is a root (level‑0) element.
    pub fn parent_of(&self, elem: ElemId) -> Option<ElemId> {
        self.parent.get(&elem).copied()
    }

    /// Siblings of `elem` (other children of the same parent).
    pub fn siblings_of(&self, elem: ElemId) -> Vec<ElemId> {
        // First try direct parent lookup
        if let Some(&p) = self.parent.get(&elem) {
            if let Some(kids) = self.children.get(&p) {
                return kids.iter().filter(|&&k| k != elem).copied().collect();
            }
        }
        // Fallback: scan all parents for one whose children include this elem
        // (needed when a child shares its parent's ElemId, common in multi-level
        //  refinement where parent_of returns None to avoid a self-loop).
        for (&_p, kids) in &self.children {
            if kids.contains(&elem) {
                return kids.iter().filter(|&&k| k != elem).copied().collect();
            }
        }
        Vec::new()
    }

    /// Refinement level of `elem` (0 = initial mesh element).
    ///
    /// Returns 0 for elements not present in the tree (this is safe when the
    /// tree has been kept in sync with the mesh).
    pub fn level(&self, elem: ElemId) -> u8 {
        self.level.get(&elem).copied().unwrap_or(0)
    }

    /// Number of children of `elem` (0 if `elem` was never refined).
    pub fn n_children(&self, elem: ElemId) -> usize {
        self.children.get(&elem).map_or(0, Vec::len)
    }

    /// Number of children produced by a single isotropic refinement in `dim`
    /// dimensions (2⁰ = 1, 2¹ = 2, 2² = 4, 2³ = 8, …).
    pub const fn n_children_per_refine(dim: usize) -> usize {
        1 << dim
    }

    /// Returns the number of recorded refinement levels (depth).
    pub fn depth(&self) -> usize {
        self.snapshots.len()
    }

    // ─── 2:1 balance closure ───────────────────────────────────────────────

    /// Compute the closure of `marked` under the **2:1 balance** rule.
    ///
    /// An element `e` at level `L` that is being refined (i.e. belongs to the
    /// closure set) forces its edge‑neighbour `n` to also be refined whenever
    /// `level(n) < level(e)`.  The check is iterated until the set stabilises.
    ///
    /// This guarantees that after refinement no two adjacent elements differ
    /// by more than one refinement level.
    ///
    /// The returned vector is sorted for deterministic iteration by callers.
    ///
    /// **Note:** the method uses *edge* adjacency, which is slightly
    /// conservative for 3‑D (it considers edge‑neighbours as well as
    /// face‑neighbours), but still correct — it simply may mark extra
    /// elements that are not strictly required.
    pub fn enforce_2to1_balance<const D: usize>(
        &self,
        mesh: &Mesh<D>,
        marked: &[ElemId],
    ) -> Vec<ElemId> {
        if marked.is_empty() {
            return Vec::new();
        }

        let edges = local_edges(mesh.elem_type);
        if edges.is_empty() {
            // Unsupported element type: return marked as-is, unchanged.
            let mut out = marked.to_vec();
            out.sort();
            out.dedup();
            return out;
        }

        // ── Build global edge → element adjacency ──────────────────────
        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..mesh.n_elems() as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(i, j) in edges {
                let key = edge_key(ns[i], ns[j]);
                edge_elems.entry(key).or_default().push(e);
            }
        }

        // ── Iterative closure ─────────────────────────────────────────
        let mut closure: HashSet<ElemId> = marked.iter().copied().collect();
        let mut changed = true;
        while changed {
            changed = false;
            // Collect candidates *before* mutating closure.
            let candidates: Vec<ElemId> = closure.iter().copied().collect();
            for &e in &candidates {
                let level_e = self.level(e);
                let ns = mesh.elem_nodes(e);
                for &(i, j) in edges {
                    let key = edge_key(ns[i], ns[j]);
                    if let Some(neighbors) = edge_elems.get(&key) {
                        for &n in neighbors {
                            if n == e || closure.contains(&n) {
                                continue;
                            }
                            if self.level(n) < level_e {
                                closure.insert(n);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        let mut result: Vec<ElemId> = closure.into_iter().collect();
        result.sort();
        result
    }
}

impl Default for RefinementTree {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Mesh;

    // ─── Helper: small quad mesh ──────────────────────────────────────────

    /// A 2×2-element Quad4 mesh on [0,1]×[0,1]:
    ///   E0: (0,0)  E1: (1,0)
    ///   E2: (0,1)  E3: (1,1)   (each split into 4 quads)
    fn unit_square_quad_2x2() -> Mesh<2> {
        let coords = vec![
            0.0, 0.0,    // 0
            0.5, 0.0,    // 1
            1.0, 0.0,    // 2
            0.0, 0.5,    // 3
            0.5, 0.5,    // 4
            1.0, 0.5,    // 5
            0.0, 1.0,    // 6
            0.5, 1.0,    // 7
            1.0, 1.0,    // 8
        ];
        let conn = vec![
            0, 1, 4, 3,  // E0
            1, 2, 5, 4,  // E1
            3, 4, 7, 6,  // E2
            4, 5, 8, 7,  // E3
        ];
        let tags = vec![0; 4];
        let face_conn = vec![0,1, 1,2, 2,5, 5,8, 8,7, 7,6, 6,3, 3,0];
        let face_tags = vec![0; 8];
        Mesh::uniform(coords, conn, tags, ElementType::Quad4, face_conn, face_tags, ElementType::Line2)
    }

    /// A small Tri3 mesh: 2 triangles forming a unit square.
    fn unit_square_tri_2() -> Mesh<2> {
        let coords = vec![
            0.0, 0.0, // 0
            1.0, 0.0, // 1
            0.0, 1.0, // 2
            1.0, 1.0, // 3
        ];
        let conn = vec![
            0, 1, 2,  // E0: lower-left
            1, 3, 2,  // E1: upper-right
        ];
        let tags = vec![0; 2];
        let face_conn = vec![0,1, 1,3, 3,2, 2,0];
        let face_tags = vec![0; 4];
        Mesh::uniform(coords, conn, tags, ElementType::Tri3, face_conn, face_tags, ElementType::Line2)
    }

    // ─── Tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_tree_queries() {
        let tree = RefinementTree::new();
        assert_eq!(tree.parent_of(0), None);
        assert!(tree.siblings_of(0).is_empty());
        assert_eq!(tree.level(42), 0);
        assert_eq!(tree.n_children(0), 0);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_init_sets_level_zero() {
        let mut tree = RefinementTree::new();
        tree.init(5);
        for e in 0..5 {
            assert_eq!(tree.level(e as ElemId), 0, "element {e} should be level 0");
        }
        assert_eq!(tree.level(999), 0); // not initialised, but safe default
    }

    #[test]
    fn test_single_refine_tri3() {
        let mut tree = RefinementTree::new();
        tree.init(2); // 2 initial Tri3 elements
        let marked = [0u32]; // refine only element 0
        tree.record_refine(2, &marked, 4); // Tri3 → 4 children

        // Element 0 was refined: 4 children at positions 0, 1, 2, 3
        assert_eq!(tree.n_children(0), 4);
        let kids = tree.children.get(&0).unwrap();
        assert_eq!(kids.len(), 4);
        assert_eq!(kids[0], 0);
        assert_eq!(kids[1], 1);
        assert_eq!(kids[2], 2);
        assert_eq!(kids[3], 3);

        // Each child has level = 1.  The child at position 0 has the
        // same ElemId as the parent (0) → parent entry is skipped to
        // avoid a self-loop.
        for &c in kids {
            assert_eq!(tree.level(c), 1);
        }
        // Children at positions 1-3 have explicit parent = 0.
        // Child at position 0 shares the parent's ElemId → no entry.
        assert_eq!(tree.parent_of(1), Some(0));
        assert_eq!(tree.parent_of(2), Some(0));
        assert_eq!(tree.parent_of(3), Some(0));
        assert_eq!(tree.parent_of(0), None); // self-loop avoided

        // Element 1 (unrefined) carried over to position 4
        assert_eq!(tree.level(4), 0);
        assert_eq!(tree.n_children(1), 0);

        // Sibling relationships: children at 0, 1, 2, 3 are siblings.
        // All have parent = 0 except element 0 (self-loop avoided).
        let sib = tree.siblings_of(2);
        assert_eq!(sib.len(), 3);
        for &c in &[0u32, 1, 3] {
            assert!(sib.contains(&c), "sibling {c} should be in set");
        }
        assert!(!sib.contains(&2)); // exclude self
    }

    #[test]
    fn test_refine_only_second_element() {
        let mut tree = RefinementTree::new();
        tree.init(4);
        // Refine element 3 only.
        tree.record_refine(4, &[3u32], 4);

        // Element 3's children start at position 3 (no marked elements before it)
        assert_eq!(tree.n_children(3), 4);
        let kids = tree.children.get(&3).unwrap();
        assert_eq!(&kids[..], &[3, 4, 5, 6]); // [3, 4, 5, 6]

        // Old elements that were after 3 in the original mesh shift:
        // element at position 3 → 4 children → old elements 4..3 shift by 3 positions
        // Actually, old element at pos 3 was refined, so elements 0-2 stay,
        // elements 3 → 4 kids, no old elements after pos 3 in a 4-element mesh.

        // Let's verify: n_elems_before=4, marked=[3], n_children=4
        // marked_before: [0, 0, 0, 0] (nothing before position 3 is marked)
        // Children of 3 start at 3 + 0 = 3
        // Updated mesh has 4 + 3 = 7 elements
        assert_eq!(tree.children.get(&3).unwrap(), &[3, 4, 5, 6]);
    }

    #[test]
    fn test_multi_element_refine() {
        let mut tree = RefinementTree::new();
        tree.init(10);

        // Mark elements 1, 3, 5 for refinement.
        let marked = [1u32, 3, 5];
        tree.record_refine(10, &marked, 4);

        // Element 1: 0 marked before → children at 1, 2, 3, 4
        assert_eq!(tree.children.get(&1).unwrap(), &[1, 2, 3, 4]);
        // Child at position 1 shares the parent's ElemId → parent entry
        // skipped to avoid self-loop.  Children 2, 3, 4 have explicit parent = 1.
        assert_eq!(tree.parent_of(2), Some(1));
        assert_eq!(tree.parent_of(3), Some(1));
        assert_eq!(tree.parent_of(4), Some(1));
        for c in &[1u32, 2, 3, 4] {
            assert_eq!(tree.level(*c), 1);
        }

        // Element 3: 1 marked before (element 1) → children at 3+3=6 → 6,7,8,9
        assert_eq!(tree.children.get(&3).unwrap(), &[6, 7, 8, 9]);

        // Element 5: 2 marked before → children at 5+6=11 → 11,12,13,14
        assert_eq!(tree.children.get(&5).unwrap(), &[11, 12, 13, 14]);

        // Unrefined elements (carry-forwards) are at their new positions.
        // Old element 0 → position 0, old element 2 → position 5,
        // old element 4 → position 10, etc.
        assert_eq!(tree.level(0), 0);
        assert_eq!(tree.level(5), 0); // carry-forward of old element 2
        assert_eq!(tree.level(10), 0); // carry-forward of old element 4, defaults to 0

        // Total refined elements count
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn test_multi_level_refine() {
        let mut tree = RefinementTree::new();
        tree.init(4);

        // Level 1: refine element 0
        let marked1 = [0u32];
        tree.record_refine(4, &marked1, 4);
        // Children of 0: at 0,1,2,3 (all level 1)
        // Mesh now has 4 + 3 = 7 elements
        assert_eq!(tree.n_children(0), 4);
        for c in 0..4u32 { assert_eq!(tree.level(c), 1); }

        // Level 2: refine child 2 (which is at position 2, level 1)
        //   n_elems_before = 7
        let marked2 = [2u32];
        tree.record_refine(7, &marked2, 4);
        // marked_before: [0, 0, 0, 0, 0, 0, 0]
        // Children of 2 start at 2 + 0 = 2, positions 2,3,4,5
        let grandkids = tree.children.get(&2).unwrap();
        assert_eq!(grandkids, &[2, 3, 4, 5]);
        for &c in grandkids {
            assert_eq!(tree.level(c), 2); // level 1 + 1 = 2
            // Child at position 2 shares the parent's ElemId → self-loop avoided.
            if c != 2 {
                assert_eq!(tree.parent_of(c), Some(2));
            }
        }
        assert_eq!(tree.parent_of(2), None); // self-loop avoided

        // Total depth (number of refine steps recorded)
        assert_eq!(tree.depth(), 2);

        // After two refinement steps, the element at position 2 is:
        //   1st refine: child 2 of parent 0 (level 1)
        //   2nd refine: child of the old position-2 element (level 2)
        //
        // The old position-2 element was refined → children at [2,3,4,5].
        // But the parent and child at position 2 share the same ElemId,
        // so parent_of(2) returns None to avoid a self-loop.
        // The children map (historical: "old" elem 2 had children) is intact.
        assert_eq!(tree.parent_of(2), None);
        assert_eq!(tree.n_children(2), 4);

        // Siblings of grandchild 3: children of old element 2, excluding 3
        let sib = tree.siblings_of(3);
        assert_eq!(sib.len(), 3);
        // The children list includes [2,3,4,5]; element 2 is the
        // self-reference position and has no parent entry, but still
        // appears as a sibling from the children map.
        assert!(sib.contains(&2));
        assert!(sib.contains(&4));
        assert!(sib.contains(&5));
    }

    #[test]
    fn test_derefine_restores_state() {
        let mut tree = RefinementTree::new();
        tree.init(6);

        // Refine elements 1 and 4
        tree.record_refine(6, &[1u32, 4], 4);

        // Verify children exist
        assert_eq!(tree.n_children(1), 4);
        assert_eq!(tree.n_children(4), 4);
        assert_eq!(tree.depth(), 1);

        // Derefine
        assert!(tree.record_derefine());

        // Children are gone
        assert_eq!(tree.n_children(1), 0);
        assert_eq!(tree.n_children(4), 0);
        for child in [1u32,2,3,4,7,8,9,10] {
            assert_eq!(tree.parent_of(child), None);
        }

        // Original elements back to level 0
        for e in 0..6u32 {
            assert_eq!(tree.level(e), 0);
        }
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_derefine_on_empty_history_returns_false() {
        let mut tree = RefinementTree::new();
        assert!(!tree.record_derefine());
    }

    #[test]
    fn test_n_children_per_refine() {
        assert_eq!(RefinementTree::n_children_per_refine(0), 1);
        assert_eq!(RefinementTree::n_children_per_refine(1), 2);
        assert_eq!(RefinementTree::n_children_per_refine(2), 4);
        assert_eq!(RefinementTree::n_children_per_refine(3), 8);
    }

    // ─── 2:1 balance tests ─────────────────────────────────────────────────

    #[test]
    fn test_balance_no_op_when_empty() {
        let mesh = unit_square_quad_2x2();
        let tree = RefinementTree::new();
        let result = tree.enforce_2to1_balance(&mesh, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_balance_single_element_no_neighbor_violation() {
        // 4 elements, all at level 0.  Refining any one element doesn't
        // create a 2:1 violation because all neighbours are at level 0 too.
        // After refinement, children at level 1 are adjacent to neighbours
        // at level 0 — gap = 1, which is OK.
        let mesh = unit_square_quad_2x2();
        let mut tree = RefinementTree::new();
        tree.init(4);

        let result = tree.enforce_2to1_balance(&mesh, &[0u32]);
        // Only element 0 should be in the closure.
        assert_eq!(result, vec![0u32], "no extra elements needed");
    }

    #[test]
    fn test_balance_neighbor_at_lower_level_triggers_marking() {
        // Create an artificial situation where element 2 has had its
        // neighbour (element 0) refined twice, creating a gap of 2.
        let mesh = unit_square_quad_2x2();
        let mut tree = RefinementTree::new();
        tree.init(4);

        // Manually set up: element 0 at level 2, rest at level 0.
        // After a hypothetical earlier refinement, element 0 was refined
        // twice, giving it level 2.  Elements 1-3 are at level 0.
        tree.level.insert(0, 2u8);
        tree.level.insert(1, 0u8);
        tree.level.insert(2, 0u8);
        tree.level.insert(3, 0u8);

        // Now mark element 0 for refinement.  Its children will be at
        // level 3.  Neighbour element 1 (level 0) would be adjacent to
        // level-3 children — gap = 3 > 1 → element 1 needs refining.
        let result = tree.enforce_2to1_balance(&mesh, &[0u32]);
        assert!(result.contains(&0), "element 0 must be in closure");
        assert!(result.contains(&1), "neighbour element 1 must be in closure");
        // Elements 2 and 3 are not neighbours of 0, so they should not
        // be in the closure (they don't share an edge with 0 in this mesh).
    }

    #[test]
    fn test_balance_cascade() {
        // Neighbour-of-neighbour cascade:
        // Element 0 at level 2 triggers element 1 (level 0).
        // Element 1 then triggers its neighbours 2 and 3 (level 0).
        let mesh = unit_square_quad_2x2();
        let mut tree = RefinementTree::new();
        tree.init(4);

        tree.level.insert(0, 2u8); // element 0 at level 2
        // others stay at level 0

        let result = tree.enforce_2to1_balance(&mesh, &[0u32]);
        // E0 at level 2 forces its edge-neighbours E1 and E2 (level 0 < 2)
        // into the closure.  E1/E2 are themselves at level 0, so they cannot
        // force their own neighbours (0 < 0 is false), leaving E3 out.
        // Closure = {0, 1, 2}.
        assert_eq!(result, vec![0u32, 1, 2], "0→1,2 cascade only; E3 unreachable from level-0 E1/E2");
    }

    #[test]
    fn test_balance_tri3_edges() {
        // Tri3 mesh with 2 elements. Refine one, neighbour at same level = no cascade.
        let mesh = unit_square_tri_2();
        let mut tree = RefinementTree::new();
        tree.init(2);

        // Mark element 0 for refinement.
        let result = tree.enforce_2to1_balance(&mesh, &[0u32]);
        assert_eq!(result, vec![0u32], "Tri3 same-level refinement needs no cascade");
    }

    #[test]
    fn test_balance_tri3_cascade() {
        let mesh = unit_square_tri_2();
        let mut tree = RefinementTree::new();
        tree.init(2);

        // Element 0 at level 2, neighbour element 1 at level 0.
        tree.level.insert(0, 2u8);
        tree.level.insert(1, 0u8);

        let result = tree.enforce_2to1_balance(&mesh, &[0u32]);
        // Element 1 is adjacent to element 0 (they share edge 1-2).
        // So both elements should be in the closure.
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_balance_marked_returns_sorted() {
        let mesh = unit_square_quad_2x2();
        let tree = RefinementTree::new();
        // All at level 0, no cascade needed, but verify sorting.
        let result = tree.enforce_2to1_balance(&mesh, &[3u32, 1u32, 0u32]);
        assert_eq!(result, vec![0u32, 1, 3], "result should be sorted");
    }

    #[test]
    fn test_init_idempotent() {
        let mut tree = RefinementTree::new();
        tree.init(3);
        // Manually set element 1 to level 2.
        tree.level.insert(1, 2);
        // Calling init again should not overwrite existing levels.
        tree.init(3);
        assert_eq!(tree.level(0), 0);
        assert_eq!(tree.level(1), 2, "existing level should not be overwritten");
        assert_eq!(tree.level(2), 0);
    }

    #[test]
    fn test_siblings_ordering() {
        let mut tree = RefinementTree::new();
        tree.init(10);

        // Refine elements 1 and 6.
        tree.record_refine(10, &[1u32, 6], 4);

        // Children of element 1: [1, 2, 3, 4]
        // Children of element 6: start at 6 + 3 = 9 → [9, 10, 11, 12]

        // Siblings of child 2 should be [1, 3, 4] (excluding 2)
        let sib2 = tree.siblings_of(2);
        assert_eq!(sib2.len(), 3);
        assert!(sib2.contains(&1));
        assert!(sib2.contains(&3));
        assert!(sib2.contains(&4));
        assert!(!sib2.contains(&2));

        // Sibling of child 10 should be [9, 11, 12]
        let sib10 = tree.siblings_of(10);
        assert_eq!(sib10.len(), 3);
        assert!(sib10.contains(&9));
        assert!(sib10.contains(&11));
        assert!(sib10.contains(&12));
    }

    #[test]
    fn test_multi_level_depth_tracking() {
        let mut tree = RefinementTree::new();
        assert_eq!(tree.depth(), 0);

        tree.init(4);
        tree.record_refine(4, &[0u32], 4);
        assert_eq!(tree.depth(), 1);

        tree.record_refine(7, &[2u32], 4);
        assert_eq!(tree.depth(), 2);

        tree.record_derefine();
        assert_eq!(tree.depth(), 1);

        tree.record_derefine();
        assert_eq!(tree.depth(), 0);

        // No more history
        assert!(!tree.record_derefine());
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_derefine_multiple_steps() {
        let mut tree = RefinementTree::new();
        tree.init(8);

        // Level 1: refine elements 2, 5
        tree.record_refine(8, &[2u32, 5], 4);
        let depth1 = tree.depth();
        assert_eq!(depth1, 1);

        // Level 2: refine specific children from level 1
        // Children of 2: [2, 3, 4, 5]
        // Children of 5: start at 5 + 3 = 8 → [8, 9, 10, 11]
        // Mesh now has 8 + 2*3 = 14 elements
        tree.record_refine(14, &[3u32, 10], 4);
        assert_eq!(tree.depth(), 2);

        // Verify grandchildren exist.  Element 3 was refined -> children [3,4,5,6].
        // Child 3 shares the parent ElemId so parent_of(3) returns None,
        // but children 4,5,6 have explicit parent = 3.
        assert_eq!(tree.n_children(3), 4);
        assert_eq!(tree.parent_of(4), Some(3));

        // Derefine one step
        assert!(tree.record_derefine());
        assert_eq!(tree.depth(), 1);

        // After derefine, grandchildren related to element 3 and 10 are gone
        assert_eq!(tree.n_children(3), 0, "children of element 3 are gone");
        assert_eq!(tree.n_children(10), 0, "children of element 10 are gone");

        // But level-1 relationships survive
        assert_eq!(tree.n_children(2), 4);
        assert_eq!(tree.n_children(5), 4);

        // Derefine back to initial
        assert!(tree.record_derefine());
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.n_children(2), 0);
        assert_eq!(tree.n_children(5), 0);
    }

    #[test]
    fn test_can_derefine() {
        let mut tree = RefinementTree::new();
        assert!(!tree.can_derefine());
        tree.init(2);
        tree.record_refine(2, &[0u32], 4);
        assert!(tree.can_derefine());
        tree.record_derefine();
        assert!(!tree.can_derefine());
    }
}
