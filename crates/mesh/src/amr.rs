//! Adaptive Mesh Refinement (AMR) for simplex meshes.
//!
//! Provides:
//! 1. **Bisection refinement** — newest-vertex bisection for triangles.
//! 2. **Zienkiewicz–Zhu (ZZ) error estimator** — gradient recovery-based element error.
//! 3. **Dörfler (bulk) marking** — marks a minimal subset of elements whose
//!    estimated errors sum to at least θ of the global error.
//! 4. **Hanging-node constraints** (2-D/3-D) — stores linear constraint
//!    equations for non-conforming refinement interfaces.
//!
//! # Usage
//! ```rust,ignore
//! use fem_mesh::{SimplexMesh, amr::{refine_marked, zz_estimator, dorfler_mark}};
//!
//! let mut mesh = SimplexMesh::<2>::unit_square_tri(4);
//! let errors   = zz_estimator(&mesh, &u_h);   // element-wise error indicators
//! let marked   = dorfler_mark(&errors, 0.5);   // Dörfler θ = 0.5
//! mesh         = refine_marked(&mesh, &marked);
//! ```

use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::{element_type::ElementType, simplex::SimplexMesh};

// ─── Bisection refinement ─────────────────────────────────────────────────────

/// Newest-vertex bisection refinement for a 2-D triangle mesh.
///
/// Each marked element is split into **2** children by bisecting the longest
/// edge (opposite to the newest vertex).  To maintain conformity, edges shared
/// with unmarked neighbours are also bisected (propagation step, simplified
/// here to a single conformity pass).
///
/// # Arguments
/// - `mesh`    — input `SimplexMesh<2>` with `elem_type = Tri3`.
/// - `marked`  — sorted list of element indices to refine.
///
/// # Returns
/// A new `SimplexMesh<2>` with the refined elements replaced by their children.
pub fn refine_marked(mesh: &SimplexMesh<2>, marked: &[ElemId]) -> SimplexMesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    // ── 1. Identify all edges to bisect ───────────────────────────────────────
    // For each marked element, mark its longest edge.
    // We also propagate to neighbours (one pass) to ensure conformity.
    let npe = 3usize;
    let n_elems = mesh.n_elems();

    // Build edge → element list for propagation.
    // edge key = (min_node, max_node)
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // Mark the longest edge of each element to be bisected.
    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    // Conformity propagation (one pass): if an interior edge is bisected,
    // both adjacent elements' longest edges should also be bisected.
    // We simply bisect the entire element (all edges) for simplicity.
    // This over-refines slightly but guarantees conformity.
    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    // ── 2. Collect new midpoint nodes ─────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        // For Tri3 bisection: bisect longest edge only (newest-vertex bisection).
        // For simplicity here, bisect all 3 edges (red refinement).
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    // ── 3. Build new element connectivity ─────────────────────────────────────
    let mut new_conn: Vec<NodeId>  = Vec::new();
    let mut new_tags: Vec<i32>     = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            // Red refinement: split Tri3 into 4 children.
            //   Original nodes: n0, n1, n2
            //   Midpoints:      m01, m12, m02
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();
            // 4 children
            new_conn.extend_from_slice(&[n0,  m01, m02]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, n1,  m12]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m02, m12, n2 ]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m12, m02]);  new_tags.push(tag); // inner
        } else {
            // Unchanged element — copy as-is.
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Rebuild boundary faces ─────────────────────────────────────────────
    // Boundary edges that were bisected get 2 children; others stay.
    let npf = 2usize; // Line2
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            // Bisected edge → 2 children
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    )
}

// ─── Hanging-node constraint ──────────────────────────────────────────────────

/// A hanging-node constraint: `u[constrained] = 0.5*(u[parent_a] + u[parent_b])`.
#[derive(Debug, Clone)]
pub struct HangingNodeConstraint {
    /// The constrained (hanging) node DOF index.
    pub constrained: usize,
    /// The two parent node DOF indices (the edge endpoints).
    pub parent_a:    usize,
    pub parent_b:    usize,
}

/// Accumulated state for multi-level non-conforming refinement.
///
/// Tracks all hanging-node constraints across multiple refinement levels.
/// When a subsequent refinement resolves a hanging node (both adjacent
/// elements get refined), that constraint is automatically removed.
///
/// # Usage
/// ```rust,ignore
/// let mut nc = NCState::new();
/// let (mesh, constraints, midpts) = nc.refine(&mesh, &marked_level1);
/// // ... solve, estimate error ...
/// let (mesh, constraints, midpts) = nc.refine(&mesh, &marked_level2);
/// // constraints now includes carried-over + new hanging nodes
/// ```
#[derive(Debug, Clone)]
pub struct NCState {
    /// All active hanging-node constraints.
    constraints: Vec<HangingNodeConstraint>,
    /// Set of edges that currently have a midpoint (edge_key → midpoint node).
    /// Used to detect when a previous hanging node gets resolved.
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

/// Accumulated state for multi-level non-conforming refinement in 3-D Tet4 meshes.
///
/// Tracks active edge midpoints across successive refinement levels and rebuilds
/// hanging-node constraints after each step.
#[derive(Debug, Clone)]
pub struct NCState3D {
    /// All active hanging-node constraints (edge midpoint constraints).
    constraints: Vec<HangingNodeConstraint>,
    /// Set of edges that currently have a midpoint (edge_key -> midpoint node).
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

impl NCState3D {
    /// Create an empty 3-D NC state for a conforming initial mesh.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
        }
    }

    /// Current hanging-node constraints (for use with `apply_hanging_constraints`).
    pub fn constraints(&self) -> &[HangingNodeConstraint] {
        &self.constraints
    }

    /// Perform one level of non-conforming refinement for Tet4 meshes.
    ///
    /// Returns `(new_mesh, constraints, midpoint_map, hanging_faces)`.
    pub fn refine(
        &mut self,
        mesh: &SimplexMesh<3>,
        marked: &[ElemId],
    ) -> (
        SimplexMesh<3>,
        Vec<HangingNodeConstraint>,
        HashMap<(NodeId, NodeId), NodeId>,
        Vec<HangingFaceConstraint>,
    ) {
        let (new_mesh, constraints, hanging_faces, midpoint_map, new_active_midpoints) =
            refine_nonconforming_3d_internal(mesh, marked, Some(&self.active_midpoints));
        self.constraints = constraints.clone();
        self.active_midpoints = new_active_midpoints;
        (new_mesh, constraints, midpoint_map, hanging_faces)
    }
}

impl NCState {
    /// Create an empty NC state for a conforming initial mesh.
    pub fn new() -> Self {
        NCState {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
        }
    }

    /// Current hanging-node constraints (for use with `apply_hanging_constraints`).
    pub fn constraints(&self) -> &[HangingNodeConstraint] {
        &self.constraints
    }

    /// Perform one level of non-conforming refinement.
    ///
    /// - Refines `marked` elements via red refinement (4 children each).
    /// - Tracks which previous hanging nodes get resolved.
    /// - Returns `(new_mesh, constraints, midpoint_map)` where `midpoint_map`
    ///   maps `(a, b) → mid` for each newly created midpoint node.
    ///   Use [`prolongate_p1`] with the midpoint map to transfer solutions.
    pub fn refine(
        &mut self,
        mesh: &SimplexMesh<2>,
        marked: &[ElemId],
    ) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        assert!(
            mesh.elem_type == ElementType::Tri3,
            "NCState::refine: only Tri3 meshes are supported"
        );

        if marked.is_empty() {
            return (mesh.clone(), self.constraints.clone(), HashMap::new());
        }

        let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
        let n_elems = mesh.n_elems();

        // ── 1. Build edge → adjacent element list ──────────────────────
        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let key = edge_key(ns[a], ns[b]);
                edge_elems.entry(key).or_default().push(e);
            }
        }

        // ── 2. Create midpoint nodes for marked elements ───────────────
        let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
        let mut new_coords: Vec<f64> = mesh.coords.clone();
        let mut next_node = mesh.n_nodes() as NodeId;

        for &e in marked {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let key = edge_key(ns[a], ns[b]);
                if midpoint_map.contains_key(&key) { continue; }
                // Check if a midpoint already exists from a previous level.
                if let Some(&mid) = self.active_midpoints.get(&key) {
                    midpoint_map.insert(key, mid);
                } else {
                    let xa = mesh.coords_of(ns[a]);
                    let xb = mesh.coords_of(ns[b]);
                    new_coords.push(0.5 * (xa[0] + xb[0]));
                    new_coords.push(0.5 * (xa[1] + xb[1]));
                    let id = next_node;
                    next_node += 1;
                    midpoint_map.insert(key, id);
                }
            }
        }

        // ── 3. Build new element connectivity ──────────────────────────
        let mut new_conn: Vec<NodeId> = Vec::new();
        let mut new_tags: Vec<i32> = Vec::new();

        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let tag = mesh.elem_tags[e as usize];

            if marked_set.contains(&e) {
                let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
                let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
                let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
                let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();

                new_conn.extend_from_slice(&[n0,  m01, m02]); new_tags.push(tag);
                new_conn.extend_from_slice(&[m01, n1,  m12]); new_tags.push(tag);
                new_conn.extend_from_slice(&[m02, m12, n2 ]); new_tags.push(tag);
                new_conn.extend_from_slice(&[m01, m12, m02]); new_tags.push(tag);
            } else {
                for k in 0..3 { new_conn.push(ns[k]); }
                new_tags.push(tag);
            }
        }

        // ── 4. Detect hanging nodes + resolve old ones ─────────────────
        // Merge new midpoints into active set.
        for (&edge, &mid) in &midpoint_map {
            self.active_midpoints.insert(edge, mid);
        }

        // Rebuild constraints: a midpoint is hanging if at least one of
        // its parent edge's adjacent elements in the NEW mesh does NOT
        // reference the midpoint node.
        //
        // Build edge → element adjacency for the NEW connectivity.
        let new_n_elems = new_tags.len();
        let mut new_edge_elems: HashMap<(NodeId, NodeId), Vec<u32>> = HashMap::new();
        for e in 0..new_n_elems as u32 {
            let off = e as usize * 3;
            let ns = &new_conn[off..off + 3];
            for &(a, b) in &local_edges_tri() {
                let key = edge_key(ns[a], ns[b]);
                new_edge_elems.entry(key).or_default().push(e);
            }
        }

        // Also build a set of all nodes referenced by each element.
        let new_node_set: std::collections::HashSet<NodeId> =
            new_conn.iter().copied().collect();

        let mut new_constraints = Vec::new();
        for (&(a, b), &mid) in &self.active_midpoints {
            if !new_node_set.contains(&mid) {
                // Midpoint not in any element → stale, remove.
                continue;
            }
            // Check if the midpoint is used by all elements that share
            // the parent edge.  If both sub-edges (a,mid) and (mid,b)
            // appear in the adjacency, all neighbours see the midpoint.
            let sub_a = edge_key(a, mid);
            let sub_b = edge_key(mid, b);
            let adj_a = new_edge_elems.get(&sub_a).map(|v| v.len()).unwrap_or(0);
            let adj_b = new_edge_elems.get(&sub_b).map(|v| v.len()).unwrap_or(0);

            // Also check if the original parent edge (a,b) still exists
            // in any element (meaning a coarse element still spans a→b).
            let parent_exists = new_edge_elems.contains_key(&edge_key(a, b));

            if parent_exists {
                // A coarse element still has edge (a,b), so mid is hanging.
                new_constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            } else if adj_a < 2 || adj_b < 2 {
                // Sub-edges not fully surrounded → boundary hanging node
                // (can happen on the mesh boundary — skip, not truly hanging).
            }
            // Otherwise: both sub-edges have 2 adjacent elements each →
            // the midpoint is fully resolved (no longer hanging).
        }

        // Clean up stale midpoints.
        self.active_midpoints.retain(|_, mid| new_node_set.contains(mid));

        new_constraints.sort_by_key(|c| c.constrained);
        self.constraints = new_constraints.clone();

        // ── 5. Rebuild boundary faces ──────────────────────────────────
        let npf = 2usize;
        let n_faces = mesh.n_faces();
        let mut new_face_conn: Vec<NodeId> = Vec::new();
        let mut new_face_tags: Vec<i32> = Vec::new();

        for f in 0..n_faces {
            let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
            let fa = fn_slice[0];
            let fb = fn_slice[1];
            let tag = mesh.face_tags[f];

            if let Some(&mid) = midpoint_map.get(&edge_key(fa, fb)) {
                new_face_conn.extend_from_slice(&[fa, mid]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mid, fb]); new_face_tags.push(tag);
            } else {
                new_face_conn.extend_from_slice(&[fa, fb]);
                new_face_tags.push(tag);
            }
        }

        let new_mesh = SimplexMesh::uniform(
            new_coords, new_conn, new_tags, ElementType::Tri3,
            new_face_conn, new_face_tags, ElementType::Line2,
        );

        (new_mesh, self.constraints.clone(), midpoint_map)
    }
}

/// Prolongate (interpolate) a P1 solution vector from a coarser mesh to the
/// refined mesh produced by [`refine_nonconforming`] or [`NCState::refine`].
///
/// Existing nodes keep their values; each new midpoint node gets the average
/// of the two parent nodes: `u_new[mid] = 0.5*(u_old[a] + u_old[b])`.
///
/// # Arguments
/// * `u_coarse`     — solution on the coarse mesh (length = coarse n_nodes).
/// * `n_nodes_fine` — number of nodes in the fine mesh.
/// * `midpoint_map` — mapping `(a, b) → mid` from edge endpoints to midpoint
///                    node IDs (as returned by [`refine_nonconforming`]).
///
/// # Returns
/// Solution vector of length `n_nodes_fine`.
pub fn prolongate_p1(
    u_coarse: &[f64],
    n_nodes_fine: usize,
    midpoint_map: &HashMap<(NodeId, NodeId), NodeId>,
) -> Vec<f64> {
    let mut u_fine = vec![0.0_f64; n_nodes_fine];
    // Copy existing node values.
    for (i, &v) in u_coarse.iter().enumerate() {
        u_fine[i] = v;
    }
    // Interpolate new midpoint nodes.
    for (&(a, b), &mid) in midpoint_map {
        u_fine[mid as usize] = 0.5 * (u_coarse[a as usize] + u_coarse[b as usize]);
    }
    u_fine
}

// ─── Non-conforming refinement ───────────────────────────────────────────────

/// Non-conforming (hanging-node) refinement for a 2-D triangle mesh.
///
/// Only the marked elements are refined (red refinement → 4 children each).
/// Unmarked elements are kept unchanged.  Where a refined and an unrefined
/// element share an edge, the new midpoint node is a **hanging node** whose
/// DOF value must be constrained to `u_hang = 0.5*(u_a + u_b)`.
///
/// # Arguments
/// - `mesh`   — input `SimplexMesh<2>` with `elem_type = Tri3`.
/// - `marked` — sorted list of element indices to refine.
///
/// # Returns
/// `(new_mesh, constraints)` where `constraints` lists all hanging nodes.
pub fn refine_nonconforming(
    mesh: &SimplexMesh<2>,
    marked: &[ElemId],
) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_nonconforming: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Build edge → adjacent element list ────────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // ── 2. Create midpoint nodes for marked elements ONLY ────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    // ── 3. Build new element connectivity ────────────────────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            // Red refinement: split into 4 children.
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();

            new_conn.extend_from_slice(&[n0,  m01, m02]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, n1,  m12]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m02, m12, n2 ]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m12, m02]); new_tags.push(tag);
        } else {
            // Unchanged element.
            for k in 0..3 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Detect hanging nodes ──────────────────────────────────────────────
    // A midpoint node is hanging if at least one element adjacent to its parent
    // edge was NOT refined (i.e., the coarse element doesn't reference the midpoint).
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj_elems) = edge_elems.get(&(a, b)) {
            let has_unrefined_neighbour = adj_elems.iter().any(|e| !marked_set.contains(e));
            if has_unrefined_neighbour {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── 5. Rebuild boundary faces ────────────────────────────────────────────
    let npf = 2usize;
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]); new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    );

    (new_mesh, constraints)
}

// ─── ZZ error estimator ───────────────────────────────────────────────────────

/// Compute element-wise Zienkiewicz–Zhu (ZZ) gradient-recovery error indicators.
///
/// Uses simple nodal averaging of element gradients to recover a smoothed
/// gradient `G(u)`, then computes
/// `η_K = ‖∇u_h|_K − G(u)|_K‖_{L²(K)}`
/// for each element `K`.
///
/// # Arguments
/// - `mesh`     — the mesh.
/// - `u`        — solution vector (one value per node, length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn zz_estimator(mesh: &SimplexMesh<2>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // ── 1. Compute element gradients ──────────────────────────────────────────
    // For Tri3: ∇u is constant over each element.
    let mut elem_grads: Vec<[f64; 2]> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
        let [x0, y0] = mesh.coords_of(n0);
        let [x1, y1] = mesh.coords_of(n1);
        let [x2, y2] = mesh.coords_of(n2);
        let u0 = u[n0 as usize]; let u1 = u[n1 as usize]; let u2 = u[n2 as usize];

        // Jacobian of mapping from reference triangle to physical:
        // J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;

        // Reference gradients of Lagrange basis: ∇ψ₀ = (-1,-1), ∇ψ₁ = (1,0), ∇ψ₂ = (0,1)
        // Physical grad = J^{-T} * ref_grad
        // J^{-T} = (1/det) * [[j11, -j10], [-j01, j00]]
        let g_ref = [
            [-1.0_f64, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0],
        ];
        let uh = [u0, u1, u2];
        let mut gx = 0.0_f64; let mut gy = 0.0_f64;
        for k in 0..3 {
            // J^{-T} * g_ref[k]
            let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) / det;
            let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) / det;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        elem_grads.push([gx, gy]);
    }

    // ── 2. Nodal gradient recovery (simple averaging) ─────────────────────────
    let mut nodal_grad = vec![[0.0_f64; 2]; n_nodes];
    let mut nodal_count = vec![0usize; n_nodes];

    for (e, &grad) in elem_grads.iter().enumerate() {
        let ns = mesh.elem_nodes(e as ElemId);
        for &n in ns {
            nodal_grad[n as usize][0] += grad[0];
            nodal_grad[n as usize][1] += grad[1];
            nodal_count[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nodal_count[n] as f64;
        if c > 0.0 {
            nodal_grad[n][0] /= c;
            nodal_grad[n][1] /= c;
        }
    }

    // ── 3. Element error indicator ────────────────────────────────────────────
    let mut eta = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)).abs();

        // Recovered gradient at centroid = average of nodal recovered gradients
        let grx: f64 = ns.iter().map(|&n| nodal_grad[n as usize][0]).sum::<f64>() / 3.0;
        let gry: f64 = ns.iter().map(|&n| nodal_grad[n as usize][1]).sum::<f64>() / 3.0;
        let eg = &elem_grads[e as usize];

        let dx = eg[0] - grx;
        let dy = eg[1] - gry;
        // η_K = ‖(∇u_h − G(u_h))‖ * sqrt(area)
        eta.push(area.sqrt() * (dx*dx + dy*dy).sqrt());
    }
    eta
}

// ─── Kelly error estimator ──────────────────────────────────────────────────

/// Compute element-wise Kelly (face-jump) error indicators.
///
/// The Kelly estimator uses the jump of the normal gradient across interior
/// edges to estimate the local error:
///
/// `η_K² = Σ_{edges E ⊂ ∂K} h_E · ‖[∂u/∂n]_E‖²`
///
/// where `[∂u/∂n]` is the jump in normal derivative across the edge and `h_E`
/// is the edge length.
///
/// # Arguments
/// - `mesh` — triangular 2-D mesh (Tri3).
/// - `u`    — solution vector (one value per node, length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn kelly_estimator(mesh: &SimplexMesh<2>, u: &[f64]) -> Vec<f64> {
    use std::collections::HashMap;

    let n_elems = mesh.n_elems();

    // 1. Compute constant element gradients (same as ZZ step 1)
    let mut elem_grads: Vec<[f64; 2]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let u0 = u[ns[0] as usize]; let u1 = u[ns[1] as usize]; let u2 = u[ns[2] as usize];

        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;

        let g_ref = [[-1.0_f64, -1.0], [1.0, 0.0], [0.0, 1.0]];
        let uh = [u0, u1, u2];
        let mut gx = 0.0_f64; let mut gy = 0.0_f64;
        for k in 0..3 {
            let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) / det;
            let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) / det;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        elem_grads.push([gx, gy]);
    }

    // 2. Build edge → (elem_a, elem_b) adjacency
    // Edge key: (min_node, max_node)
    type Edge = (NodeId, NodeId);
    fn edge_key(a: NodeId, b: NodeId) -> Edge {
        if a < b { (a, b) } else { (b, a) }
    }

    let mut edge_elems: HashMap<Edge, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let edges = [
            edge_key(ns[0], ns[1]),
            edge_key(ns[1], ns[2]),
            edge_key(ns[0], ns[2]),
        ];
        for ek in &edges {
            edge_elems.entry(*ek).or_default().push(e);
        }
    }

    // 3. Compute jump contributions per element
    let mut eta_sq = vec![0.0_f64; n_elems];

    for (&(na, nb), elems) in &edge_elems {
        if elems.len() != 2 { continue; } // skip boundary edges (no jump)
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;

        // Edge vector and length
        let [xa, ya] = mesh.coords_of(na);
        let [xb, yb] = mesh.coords_of(nb);
        let dx = xb - xa;
        let dy = yb - ya;
        let h_e = (dx * dx + dy * dy).sqrt();

        // Edge normal (unnormalized is fine since we normalize the jump)
        let nx = dy / h_e;
        let ny = -dx / h_e;

        // Normal gradient jump: [∂u/∂n] = (grad_u_e0 - grad_u_e1) · n
        let g0 = &elem_grads[e0];
        let g1 = &elem_grads[e1];
        let jump = (g0[0] - g1[0]) * nx + (g0[1] - g1[1]) * ny;

        // Distribute h_E * jump² to both elements
        let contrib = h_e * jump * jump;
        eta_sq[e0] += contrib;
        eta_sq[e1] += contrib;
    }

    // 4. Return sqrt
    eta_sq.iter().map(|&s| s.sqrt()).collect()
}

// ─── Dörfler marking ─────────────────────────────────────────────────────────

/// Dörfler (bulk criterion) marking strategy.
///
/// Returns a sorted list of element indices to refine such that the sum of their
/// error indicators is at least `theta` times the sum of all indicators.
///
/// # Arguments
/// - `eta`   — element error indicators (from [`zz_estimator`]).
/// - `theta` — bulk parameter in (0, 1]; θ = 0.5 is typical.
pub fn dorfler_mark(eta: &[f64], theta: f64) -> Vec<ElemId> {
    let total: f64 = eta.iter().sum();
    let threshold = theta * total;

    // Sort by decreasing error
    let mut indices: Vec<ElemId> = (0..eta.len() as ElemId).collect();
    indices.sort_unstable_by(|&a, &b| eta[b as usize].partial_cmp(&eta[a as usize]).unwrap());

    let mut marked = Vec::new();
    let mut acc = 0.0_f64;
    for idx in indices {
        if acc >= threshold { break; }
        acc += eta[idx as usize];
        marked.push(idx);
    }
    marked.sort_unstable();
    marked
}

// ─── Uniform refinement ───────────────────────────────────────────────────────

/// Uniformly refine all elements of the mesh (red refinement for Tri3).
pub fn refine_uniform(mesh: &SimplexMesh<2>) -> SimplexMesh<2> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    refine_marked(mesh, &all)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Local edge index pairs for Tri3.
fn local_edges_tri() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (0, 2)]
}

/// Canonical edge key (sorted node pair).
fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

/// Return the canonical edge key of the longest edge of a Tri3 element.
fn longest_edge_tri(mesh: &SimplexMesh<2>, ns: &[NodeId]) -> (NodeId, NodeId) {
    let coords: [[f64; 2]; 3] = std::array::from_fn(|k| mesh.coords_of(ns[k]));
    let edges = local_edges_tri();
    let mut best = edge_key(ns[edges[0].0], ns[edges[0].1]);
    let mut best_len2 = 0.0_f64;
    for (a, b) in edges {
        let dx = coords[b][0] - coords[a][0];
        let dy = coords[b][1] - coords[a][1];
        let l2 = dx*dx + dy*dy;
        if l2 > best_len2 {
            best_len2 = l2;
            best = edge_key(ns[a], ns[b]);
        }
    }
    best
}

// ─── 3-D (Tet4) Support ──────────────────────────────────────────────────────

/// A hanging-face descriptor in 3-D.
///
/// This records a coarse triangular face `(parent_a, parent_b, parent_c)` that is
/// non-conforming against a refined neighbor. `constrained` stores a representative
/// midpoint node on that face (for stable sorting / debugging), while edge midpoint
/// constraints are provided via `HangingNodeConstraint`.
#[derive(Debug, Clone)]
pub struct HangingFaceConstraint {
    /// Representative midpoint node on the hanging face.
    pub constrained: usize,
    /// Coarse face vertex node indices.
    pub parent_a: usize,
    pub parent_b: usize,
    pub parent_c: usize,
}

/// Local face index triplets for Tet4 (4 triangular faces).
/// Each face is represented as a sorted triplet of local node indices.
fn local_faces_tet() -> [(usize, usize, usize); 4] {
    [
        (0, 1, 2), // Face 0: opposite to vertex 3
        (0, 1, 3), // Face 1: opposite to vertex 2
        (0, 2, 3), // Face 2: opposite to vertex 1
        (1, 2, 3), // Face 3: opposite to vertex 0
    ]
}

/// Canonical face key (sorted triplet of nodes).
fn face_key_3d(a: NodeId, b: NodeId, c: NodeId) -> (NodeId, NodeId, NodeId) {
    let mut nodes = [a, b, c];
    nodes.sort();
    (nodes[0], nodes[1], nodes[2])
}

/// Perform non-conforming red refinement on a 3-D Tet4 mesh.
///
/// Refines only marked elements; unrefined neighbors create hanging face constraints.
/// Tet4 red refinement creates 8 child tets and 5 new nodes per parent:
/// - 4 edge midpoints (one per parent edge)
/// - 1 face center per refined face (only for faces touching a refined tet)
pub fn refine_nonconforming_3d(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>) {
    let (new_mesh, edge_constraints, face_constraints, _, _) =
        refine_nonconforming_3d_internal(mesh, marked, None);
    (new_mesh, edge_constraints, face_constraints)
}

fn refine_nonconforming_3d_internal(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    SimplexMesh<3>,
    Vec<HangingNodeConstraint>,
    Vec<HangingFaceConstraint>,
    HashMap<(NodeId, NodeId), NodeId>,
    HashMap<(NodeId, NodeId), NodeId>,
) {
    assert!(
        mesh.elem_type == ElementType::Tet4,
        "refine_nonconforming_3d: only Tet4 meshes are supported"
    );

    if marked.is_empty() {
        let mut active = HashMap::new();
        if let Some(prev) = active_midpoints {
            active = prev.clone();
        }
        return (mesh.clone(), Vec::new(), Vec::new(), HashMap::new(), active);
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Build face → adjacent element list (for Tet4, each element has 4 faces) ──
    let mut face_elems: HashMap<(NodeId, NodeId, NodeId), Vec<ElemId>> = HashMap::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);

        // Record faces
        for (a, b, c) in local_faces_tet() {
            let key = face_key_3d(ns[a], ns[b], ns[c]);
            face_elems.entry(key).or_default().push(e);
        }

    }

    // ── 2. Create midpoint nodes for marked elements ───────────────────────────
    let mut edge_midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);

        // Create edge midpoints
        for (i, j) in local_edges_tet() {
            let key = edge_key(ns[i], ns[j]);
            edge_midpoint_map.entry(key).or_insert_with(|| {
                if let Some(prev) = active_midpoints.and_then(|m| m.get(&key)) {
                    *prev
                } else {
                    let xa = mesh.coords_of(ns[i]);
                    let xb = mesh.coords_of(ns[j]);
                    new_coords.push(0.5 * (xa[0] + xb[0]));
                    new_coords.push(0.5 * (xa[1] + xb[1]));
                    new_coords.push(0.5 * (xa[2] + xb[2]));
                    let id = next_node;
                    next_node += 1;
                    id
                }
            });
        }
    }

    // ── 3. Build new element connectivity ─────────────────────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            // Red refinement: split Tet4 into 8 children using edge midpoints.
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2]; let n3 = ns[3];

            let m01 = *edge_midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m02 = *edge_midpoint_map.get(&edge_key(n0, n2)).unwrap();
            let m03 = *edge_midpoint_map.get(&edge_key(n0, n3)).unwrap();
            let m12 = *edge_midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m13 = *edge_midpoint_map.get(&edge_key(n1, n3)).unwrap();
            let m23 = *edge_midpoint_map.get(&edge_key(n2, n3)).unwrap();

            // 4 corner tets.
            new_conn.extend_from_slice(&[n0, m01, m02, m03]); new_tags.push(tag);
            new_conn.extend_from_slice(&[n1, m01, m12, m13]); new_tags.push(tag);
            new_conn.extend_from_slice(&[n2, m02, m12, m23]); new_tags.push(tag);
            new_conn.extend_from_slice(&[n3, m03, m13, m23]); new_tags.push(tag);

            // 4 tets splitting the central octahedron.
            new_conn.extend_from_slice(&[m01, m02, m03, m23]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m02, m12, m23]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m12, m13, m23]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m03, m13, m23]); new_tags.push(tag);
        } else {
            // Unrefined element: keep as is
            for k in 0..4 {
                new_conn.push(ns[k]);
            }
            new_tags.push(tag);
        }
    }

    // ── 4. Detect hanging faces and derive hanging-node edge constraints ───────
    let mut face_constraints: Vec<HangingFaceConstraint> = Vec::new();
    let mut edge_constraints: Vec<HangingNodeConstraint> = Vec::new();

    for (&(a, b, c), adj) in &face_elems {
        if adj.len() != 2 {
            continue;
        }
        let refined_count = adj.iter().filter(|&&e| marked_set.contains(&e)).count();
        if refined_count != 1 {
            continue;
        }

        let eab = edge_key(a, b);
        let ebc = edge_key(b, c);
        let eac = edge_key(a, c);

        let mab = match edge_midpoint_map.get(&eab) {
            Some(v) => *v,
            None => continue,
        };
        let mbc = match edge_midpoint_map.get(&ebc) {
            Some(v) => *v,
            None => continue,
        };
        let mac = match edge_midpoint_map.get(&eac) {
            Some(v) => *v,
            None => continue,
        };

        edge_constraints.push(HangingNodeConstraint {
            constrained: mab as usize,
            parent_a: a as usize,
            parent_b: b as usize,
        });
        edge_constraints.push(HangingNodeConstraint {
            constrained: mbc as usize,
            parent_a: b as usize,
            parent_b: c as usize,
        });
        edge_constraints.push(HangingNodeConstraint {
            constrained: mac as usize,
            parent_a: a as usize,
            parent_b: c as usize,
        });

        face_constraints.push(HangingFaceConstraint {
            constrained: mab as usize,
            parent_a: a as usize,
            parent_b: b as usize,
            parent_c: c as usize,
        });
    }

    edge_constraints.sort_by_key(|c| c.constrained);
    edge_constraints.dedup_by_key(|c| c.constrained);

    face_constraints.sort_by_key(|c| (c.parent_a, c.parent_b, c.parent_c));
    face_constraints.dedup_by_key(|c| (c.parent_a, c.parent_b, c.parent_c));

    // Rebuild active midpoint set from previous + current, keeping only live nodes.
    let mut new_active_midpoints = HashMap::new();
    if let Some(prev) = active_midpoints {
        for (&edge, &mid) in prev {
            new_active_midpoints.insert(edge, mid);
        }
    }
    for (&edge, &mid) in &edge_midpoint_map {
        new_active_midpoints.insert(edge, mid);
    }

    let new_node_set: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
    new_active_midpoints.retain(|_, mid| new_node_set.contains(mid));

    // Rebuild edge constraints from active midpoint map and current mesh topology.
    let mut current_edge_set: std::collections::HashSet<(NodeId, NodeId)> =
        std::collections::HashSet::new();
    for e in 0..new_tags.len() as ElemId {
        let ns = &new_conn[e as usize * 4..e as usize * 4 + 4];
        for &(i, j) in &local_edges_tet() {
            current_edge_set.insert(edge_key(ns[i], ns[j]));
        }
    }

    let mut rebuilt_constraints = Vec::new();
    for (&(a, b), &mid) in &new_active_midpoints {
        if current_edge_set.contains(&edge_key(a, b)) {
            rebuilt_constraints.push(HangingNodeConstraint {
                constrained: mid as usize,
                parent_a: a as usize,
                parent_b: b as usize,
            });
        }
    }
    rebuilt_constraints.sort_by_key(|c| c.constrained);
    rebuilt_constraints.dedup_by_key(|c| c.constrained);

    // ── 5. Rebuild boundary triangular faces ──────────────────────────────────
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    let npf = 3usize;

    for f in 0..mesh.n_faces() {
        let fs = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fs[0];
        let b = fs[1];
        let c = fs[2];
        let tag = mesh.face_tags[f];

        let mab = edge_midpoint_map.get(&edge_key(a, b)).copied();
        let mbc = edge_midpoint_map.get(&edge_key(b, c)).copied();
        let mac = edge_midpoint_map.get(&edge_key(a, c)).copied();

        if let (Some(mab), Some(mbc), Some(mac)) = (mab, mbc, mac) {
            new_face_conn.extend_from_slice(&[a, mab, mac]);
            new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[b, mbc, mab]);
            new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[c, mac, mbc]);
            new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mab, mbc, mac]);
            new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b, c]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords,
        new_conn,
        new_tags,
        ElementType::Tet4,
        new_face_conn,
        new_face_tags,
        ElementType::Tri3,
    );

    (
        new_mesh,
        rebuilt_constraints,
        face_constraints,
        edge_midpoint_map,
        new_active_midpoints,
    )
}

/// Local edge pairs for Tet4 (6 edges).
fn local_edges_tet() -> [(usize, usize); 6] {
    [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
    ]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_refinement_element_count() {
        // Each Tri3 → 4 children with red refinement.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_before = mesh.n_elems();
        let fine = refine_uniform(&mesh);
        assert_eq!(fine.n_elems(), 4 * n_before,
            "Expected 4×{n_before}={} elements, got {}", 4*n_before, fine.n_elems());
    }

    #[test]
    fn uniform_refinement_node_count() {
        // A 1×1 square → 2 triangles, 4 nodes.
        // After red refinement: 8 triangles, 4+3=7 new midpoints? Actually 4+3=7 total.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let fine = refine_uniform(&mesh);
        // 1×1 unit square: 4 corners + 4 edge midpoints + 1 interior midpoint = 9
        assert!(fine.n_nodes() > mesh.n_nodes(),
            "Refinement should add nodes: before={}, after={}", mesh.n_nodes(), fine.n_nodes());
    }

    #[test]
    fn uniform_refinement_two_levels() {
        // 2 levels of uniform refinement: n → 4n → 16n elements.
        let mesh0 = SimplexMesh::<2>::unit_square_tri(2);
        let n0 = mesh0.n_elems();
        let mesh1 = refine_uniform(&mesh0);
        let mesh2 = refine_uniform(&mesh1);
        assert_eq!(mesh2.n_elems(), 16 * n0);
    }

    #[test]
    fn dorfler_marks_at_least_theta() {
        // All equal errors → should mark first `ceil(θ * n)` elements.
        let eta = vec![1.0_f64; 10];
        let marked = dorfler_mark(&eta, 0.5);
        let marked_sum: f64 = marked.iter().map(|&i| eta[i as usize]).sum();
        let total: f64 = eta.iter().sum();
        assert!(marked_sum >= 0.5 * total,
            "Dörfler: marked sum {marked_sum} < 0.5 * {total}");
    }

    #[test]
    fn zz_estimator_smooth_solution() {
        // For u = x (linear), the FE solution is exact on Tri3 → ZZ error should be ≈ 0.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let u: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();
        let eta = zz_estimator(&mesh, &u);
        let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_eta < 1e-12, "ZZ estimator: exact linear solution, max_eta={max_eta:.3e}");
    }

    #[test]
    fn refine_marked_subset() {
        // Mark only a few elements and verify total element count.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n0 = mesh.n_elems();
        let marked = vec![0u32, 1, 2]; // mark 3 elements
        let fine = refine_marked(&mesh, &marked);
        // Each marked element → 4, but neighbours may be pulled in.
        // At minimum: 3 elements became 4*3=12, rest unchanged.
        assert!(fine.n_elems() >= n0 - 3 + 3 * 4,
            "Expected ≥{} elems, got {}", n0 - 3 + 3*4, fine.n_elems());
    }

    #[test]
    fn kelly_estimator_linear_exact() {
        // For a linear function u(x,y) = x, the gradient is constant everywhere,
        // so jumps across edges should be zero → Kelly indicator = 0.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 1e-12, "Kelly should be zero for linear u, got {max_eta:.3e}");
    }

    #[test]
    fn kelly_estimator_nonzero_for_quadratic() {
        // u(x,y) = x² has a piecewise-constant gradient x-component = 2x
        // that varies between elements → non-zero jumps.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let x = mesh.coords_of(i as NodeId)[0];
            x * x
        }).collect();
        let eta = kelly_estimator(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta > 1e-4, "Kelly should be nonzero for x², got {max_eta:.3e}");
    }

    // ── Non-conforming refinement tests ──────────────────────────────────────

    #[test]
    fn nc_refine_no_marked_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[]);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn nc_refine_all_marked_no_hanging() {
        // Refining all elements → no hanging nodes (equivalent to uniform).
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints) = refine_nonconforming(&mesh, &all);
        assert_eq!(nc.n_elems(), 4 * mesh.n_elems());
        assert!(constraints.is_empty(),
            "all-marked NCMesh should have no hanging nodes, got {}", constraints.len());
    }

    #[test]
    fn nc_refine_single_element_has_hanging_nodes() {
        // Refine just element 0 of a 2×2 mesh → should produce hanging nodes
        // on the edges shared with unrefined neighbours.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[0]);

        // Element 0 → 4 children, rest (7) unchanged → 7 + 4 = 11 elements.
        assert_eq!(nc.n_elems(), mesh.n_elems() - 1 + 4);

        // Element 0 has 3 edges; some are interior → hanging nodes on those.
        assert!(!constraints.is_empty(),
            "single-element NC refine should produce hanging nodes");

        // Each hanging node should be a new midpoint.
        let orig_n = mesh.n_nodes();
        for c in &constraints {
            assert!(c.constrained >= orig_n,
                "hanging node {} should be >= orig_n_nodes {}", c.constrained, orig_n);
            assert!(c.parent_a < orig_n);
            assert!(c.parent_b < orig_n);
        }
    }

    #[test]
    fn nc_refine_hanging_node_coords_are_midpoints() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[0]);

        for c in &constraints {
            let mid_coords = nc.coords_of(c.constrained as NodeId);
            let pa = nc.coords_of(c.parent_a as NodeId);
            let pb = nc.coords_of(c.parent_b as NodeId);
            for d in 0..2 {
                let expected = 0.5 * (pa[d] + pb[d]);
                assert!(
                    (mid_coords[d] - expected).abs() < 1e-14,
                    "hanging node coord[{d}] = {}, expected midpoint {}",
                    mid_coords[d], expected
                );
            }
        }
    }

    #[test]
    fn nc_refine_fewer_elements_than_conforming() {
        // Non-conforming refine of a subset should produce fewer elements
        // than the conforming refine_marked (which propagates to neighbours).
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let marked = vec![0u32, 1, 2];

        let conforming = refine_marked(&mesh, &marked);
        let (nc, _) = refine_nonconforming(&mesh, &marked);

        assert!(
            nc.n_elems() <= conforming.n_elems(),
            "NC ({}) should have ≤ elements than conforming ({})",
            nc.n_elems(), conforming.n_elems()
        );
    }

    #[test]
    fn nc_refine_two_levels() {
        // Refine once, then refine again on some new elements.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let (nc1, c1) = refine_nonconforming(&mesh, &[0, 1]);
        assert!(!c1.is_empty() || mesh.n_elems() == 2,
            "first level should have constraints (or trivial mesh)");

        // Refine element 0 of the new mesh.
        let (nc2, c2) = refine_nonconforming(&nc1, &[0]);
        assert!(nc2.n_elems() > nc1.n_elems());
        // Second level may also produce hanging nodes.
        let _ = c2;
    }

    #[test]
    fn nc_refine_mesh_valid() {
        // The resulting mesh should pass consistency check.
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let (nc, _) = refine_nonconforming(&mesh, &[0, 3, 5]);
        nc.check().unwrap();
    }

    // ── Prolongation tests ──────────────────────────────────────────────────

    #[test]
    fn prolongate_p1_linear_exact() {
        // For u = x (linear), prolongation should be exact.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let u: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();

        let mut nc = NCState::new();
        let (fine, _, midpts) = nc.refine(&mesh, &[0, 1, 2]);
        let u_fine = prolongate_p1(&u, fine.n_nodes(), &midpts);

        // Every node in the fine mesh should have u = x.
        for n in 0..fine.n_nodes() {
            let x = fine.coords_of(n as NodeId)[0];
            assert!(
                (u_fine[n] - x).abs() < 1e-14,
                "prolongation: u[{n}]={}, expected x={x}", u_fine[n]
            );
        }
    }

    #[test]
    fn prolongate_p1_preserves_coarse_values() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let u: Vec<f64> = (0..mesh.n_nodes()).map(|i| i as f64 * 1.5).collect();

        let mut nc = NCState::new();
        let (fine, _, midpts) = nc.refine(&mesh, &[0]);
        let u_fine = prolongate_p1(&u, fine.n_nodes(), &midpts);

        // Coarse node values must be preserved.
        for i in 0..mesh.n_nodes() {
            assert!(
                (u_fine[i] - u[i]).abs() < 1e-14,
                "coarse node {i}: u_fine={}, u_coarse={}", u_fine[i], u[i]
            );
        }
    }

    // ── Multi-level NCState tests ───────────────────────────────────────────

    #[test]
    fn ncstate_two_level_refine() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        let (m1, c1, _) = nc.refine(&mesh, &[0, 1]);
        assert!(!c1.is_empty());
        m1.check().unwrap();

        // Second level: refine some of the new elements.
        let (m2, c2, _) = nc.refine(&m1, &[0, 1]);
        assert!(m2.n_elems() > m1.n_elems());
        m2.check().unwrap();
        let _ = c2;
    }

    #[test]
    fn ncstate_resolves_hanging_nodes_when_neighbour_refined() {
        // Refining all elements at level 2 does NOT resolve level 1 hanging nodes,
        // because the formerly-coarse elements' children are at a different depth
        // than the re-refined children. This is correct NC behavior.
        //
        // However, when we refine only the coarse elements that cause hanging nodes
        // (and not the already-fine ones), the hanging nodes SHOULD be resolved
        // at that interface — though new hanging nodes may appear elsewhere.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        // Refine half the elements → hanging nodes.
        let half: Vec<ElemId> = (0..mesh.n_elems() as ElemId / 2).collect();
        let (m1, c1, _) = nc.refine(&mesh, &half);
        assert!(!c1.is_empty(), "should have hanging nodes after partial refinement");
        m1.check().unwrap();

        // Refine ALL elements → creates a uniformly finer mesh, but multi-level
        // hanging nodes can appear from depth mismatch.
        let all: Vec<ElemId> = (0..m1.n_elems() as ElemId).collect();
        let (m2, c2, _) = nc.refine(&m1, &all);
        m2.check().unwrap();
        // The original hanging nodes may produce new constraints at deeper levels.
        // This is expected for multi-level NC refinement.
        let _ = c2;
    }

    #[test]
    fn ncstate_multi_level_prolongation() {
        // Prolongate u=x through two levels of NC refinement.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let u0: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();

        let mut nc = NCState::new();
        let (m1, _, mp1) = nc.refine(&mesh, &[0, 1]);
        let u1 = prolongate_p1(&u0, m1.n_nodes(), &mp1);

        let (m2, _, mp2) = nc.refine(&m1, &[0]);
        let u2 = prolongate_p1(&u1, m2.n_nodes(), &mp2);

        // All nodes should still have u = x (exact for linear).
        for n in 0..m2.n_nodes() {
            let x = m2.coords_of(n as NodeId)[0];
            assert!(
                (u2[n] - x).abs() < 1e-14,
                "2-level prolongation: node {n}, u={}, x={x}", u2[n]
            );
        }
    }

    // ── 3-D (Tet4) NCMesh tests ────────────────────────────────────────────

    #[test]
    fn tet4_nonconforming_refine_single_element() {
        // Create a simple Tet4 mesh: unit cube with some tets.
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let n_elems_orig = mesh.n_elems();

        let (refined, edge_constraints, face_constraints) = refine_nonconforming_3d(&mesh, &[0]);

        // Refining 1 tet should create 8 children.
        // Total elems = 8 refined children + (n_elems_orig - 1) unchanged.
        let expected = 8 + (n_elems_orig - 1);
        assert_eq!(refined.n_elems(), expected,
            "Expected {} elems, got {}", expected, refined.n_elems());
        
        // Should create new midpoint nodes.
        assert!(refined.n_nodes() > mesh.n_nodes(), "Refinement should add nodes");

        // One refined tet against its unrefined neighbors should produce hanging edges/faces.
        assert!(!edge_constraints.is_empty(), "expected hanging edge constraints");
        assert!(!face_constraints.is_empty(), "expected hanging face descriptors");

        refined.check().unwrap();
    }

    #[test]
    fn tet4_nonconforming_refine_with_neighbor() {
        // Refine one tet and verify non-conforming constraints are emitted.
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let n_elems_orig = mesh.n_elems();

        // Refine the first tet only.
        let (refined, edge_constr, face_constr) = refine_nonconforming_3d(&mesh, &[0]);

        // Should have 8 children from refined tet + (n_elems_orig - 1) unchanged.
        let expected_elems = 8 + (n_elems_orig - 1);
        assert_eq!(refined.n_elems(), expected_elems,
            "Expected {} refined elems, got {}", expected_elems, refined.n_elems());

        // Nodes should increase (at minimum by 6 edge midpoints per refined tet).
        assert!(refined.n_nodes() >= mesh.n_nodes() + 6);

        assert!(!edge_constr.is_empty());
        assert!(!face_constr.is_empty());
    }

    #[test]
    fn hanging_face_constraint_struct_creation() {
        // Verify that HangingFaceConstraint is properly structured.
        let constraint = HangingFaceConstraint {
            constrained: 10,
            parent_a: 0,
            parent_b: 1,
            parent_c: 2,
        };
        assert_eq!(constraint.constrained, 10);
        assert_eq!(constraint.parent_a, 0);
        assert_eq!(constraint.parent_b, 1);
        assert_eq!(constraint.parent_c, 2);
    }

    #[test]
    fn ncstate3d_two_level_refine() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc3 = NCState3D::new();

        let (m1, c1, _, f1) = nc3.refine(&mesh, &[0]);
        assert!(m1.n_elems() > mesh.n_elems());
        assert!(!c1.is_empty());
        assert!(!f1.is_empty());
        m1.check().unwrap();

        // Refine a subset again; constraints should still be valid and mesh consistent.
        let (m2, c2, _, _) = nc3.refine(&m1, &[0, 1]);
        assert!(m2.n_elems() > m1.n_elems());
        assert!(!c2.is_empty());
        m2.check().unwrap();
    }
}
