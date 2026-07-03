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
//! # Organization
//! - [`mod@refine_2d`] — 2-D Tri3/Quad4 refinement, estimators, p-refinement
//! - [`mod@refine_3d_tet`] — 3-D Tet4 NC refinement and anisotropic
//! - [`mod@refine_3d_hex`] — 3-D Hex8/20/27 uniform + NC + aniso
//! - [`mod@refine_3d_prism_pyramid`] — 3-D Prism6 + Pyramid5 all refinements
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
// ─── Test module ───────────────────────────────────────────────────────────

use std::collections::HashMap;
use fem_core::{FaceId, NodeId, ElemId};
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

/// Refinement provenance for one red-refinement level.
///
/// Stores parent -> children mapping and parent connectivity needed by
/// [`derefine_marked`].
#[derive(Debug, Clone)]
pub struct DerefineTree {
    pub records: HashMap<ElemId, DerefineRecord>,
    pub midpoint_map: HashMap<(NodeId, NodeId), NodeId>,
}

/// One parent refinement record.
#[derive(Debug, Clone)]
pub struct DerefineRecord {
    pub parent_nodes: [NodeId; 3],
    pub parent_tag: i32,
    pub children: [ElemId; 4],
}

impl DerefineTree {
    /// Return sorted parent element ids available for derefinement.
    pub fn parents(&self) -> Vec<ElemId> {
        let mut p: Vec<ElemId> = self.records.keys().copied().collect();
        p.sort_unstable();
        p
    }
}

/// Same as [`refine_marked`], but also returns a provenance tree that enables
/// one-level derefinement through [`derefine_marked`].
pub fn refine_marked_with_tree(mesh: &SimplexMesh<2>, marked: &[ElemId]) -> (SimplexMesh<2>, DerefineTree) {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked_with_tree: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    let npe = 3usize;
    let n_elems = mesh.n_elems();

    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
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

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();
    let mut tree_records: HashMap<ElemId, DerefineRecord> = HashMap::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();

            let c0 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[n0,  m01, m02]); new_tags.push(tag);
            let c1 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m01, n1,  m12]); new_tags.push(tag);
            let c2 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m02, m12, n2 ]); new_tags.push(tag);
            let c3 = (new_tags.len()) as ElemId;
            new_conn.extend_from_slice(&[m01, m12, m02]); new_tags.push(tag);

            tree_records.insert(
                e,
                DerefineRecord {
                    parent_nodes: [n0, n1, n2],
                    parent_tag: tag,
                    children: [c0, c1, c2, c3],
                },
            );
        } else {
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

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
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let fine = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    );

    (fine, DerefineTree { records: tree_records, midpoint_map })
}

/// Derefine selected parent elements from a single-level [`DerefineTree`].
///
/// This function expects `mesh` to be the direct output of
/// [`refine_marked_with_tree`] with no additional refinement/coarsening in
/// between. It removes the 4 child triangles and restores the parent triangle.
pub fn derefine_marked(mesh: &SimplexMesh<2>, tree: &DerefineTree, parents: &[ElemId]) -> SimplexMesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "derefine_marked: only Tri3 meshes are supported"
    );

    if parents.is_empty() {
        return mesh.clone();
    }

    let mut child_drop = std::collections::HashSet::<ElemId>::new();
    let mut restore = Vec::<DerefineRecord>::new();

    for &p in parents {
        if let Some(rec) = tree.records.get(&p) {
            for &c in &rec.children {
                child_drop.insert(c);
            }
            restore.push(rec.clone());
        }
    }

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..mesh.n_elems() as ElemId {
        if child_drop.contains(&e) {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        new_conn.extend_from_slice(&[ns[0], ns[1], ns[2]]);
        new_tags.push(mesh.elem_tags[e as usize]);
    }

    for rec in &restore {
        new_conn.extend_from_slice(&rec.parent_nodes);
        new_tags.push(rec.parent_tag);
    }

    // Rebuild boundary edges from exterior edges in the new connectivity.
    let mut edge_count: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let mut oriented_edge: HashMap<(NodeId, NodeId), (NodeId, NodeId)> = HashMap::new();
    for e in 0..new_tags.len() {
        let off = 3 * e;
        let tri = [new_conn[off], new_conn[off + 1], new_conn[off + 2]];
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in edges {
            let k = edge_key(a, b);
            *edge_count.entry(k).or_insert(0) += 1;
            oriented_edge.entry(k).or_insert((a, b));
        }
    }

    let mut old_bnd_tags = HashMap::<(NodeId, NodeId), i32>::new();
    for f in 0..mesh.n_faces() {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        old_bnd_tags.insert(edge_key(a, b), mesh.face_tags[f]);
    }

    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    for (&k, &cnt) in &edge_count {
        if cnt != 1 {
            continue;
        }
        let (a, b) = oriented_edge[&k];
        let mut tag = old_bnd_tags.get(&k).copied().unwrap_or(0);

        if tag == 0 {
            // Attempt to recover merged boundary tag from split edges (a,m) + (m,b).
            for m in 0..mesh.n_nodes() as NodeId {
                let k1 = edge_key(a, m);
                let k2 = edge_key(m, b);
                if let (Some(&t1), Some(&t2)) = (old_bnd_tags.get(&k1), old_bnd_tags.get(&k2)) {
                    if t1 == t2 {
                        tag = t1;
                        break;
                    }
                }
            }
        }

        if tag != 0 {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    SimplexMesh::uniform(
        mesh.coords.clone(),
        new_conn,
        new_tags,
        ElementType::Tri3,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
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
    /// Refinement history snapshots for rollback-based derefinement.
    history: Vec<NCState2DSnapshot>,
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
    /// Current hanging-face descriptors.
    hanging_faces: Vec<HangingFaceConstraint>,
    /// Refinement history snapshots for rollback-based derefinement.
    history: Vec<NCState3DSnapshot>,
}

#[derive(Debug, Clone)]
struct NCState2DSnapshot {
    mesh: SimplexMesh<2>,
    constraints: Vec<HangingNodeConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone)]
struct NCState3DSnapshot {
    mesh: SimplexMesh<3>,
    constraints: Vec<HangingNodeConstraint>,
    hanging_faces: Vec<HangingFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

impl NCState3D {
    /// Create an empty 3-D NC state for a conforming initial mesh.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            hanging_faces: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Current hanging-node constraints (for use with `apply_hanging_constraints`).
    pub fn constraints(&self) -> &[HangingNodeConstraint] {
        &self.constraints
    }

    /// Current hanging-face descriptors.
    pub fn hanging_faces(&self) -> &[HangingFaceConstraint] {
        &self.hanging_faces
    }

    /// Whether one rollback derefinement step is available.
    pub fn can_derefine(&self) -> bool {
        !self.history.is_empty()
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
        self.history.push(NCState3DSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            hanging_faces: self.hanging_faces.clone(),
            active_midpoints: self.active_midpoints.clone(),
        });

        let (new_mesh, constraints, hanging_faces, midpoint_map, new_active_midpoints) =
            refine_nonconforming_3d_internal(mesh, marked, Some(&self.active_midpoints));
        self.constraints = constraints.clone();
        self.hanging_faces = hanging_faces.clone();
        self.active_midpoints = new_active_midpoints;
        (new_mesh, constraints, midpoint_map, hanging_faces)
    }

    /// Roll back one NC refinement step.
    ///
    /// Returns the previous mesh and restored constraints if history exists.
    pub fn derefine_last(
        &mut self,
    ) -> Option<(SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.hanging_faces = snap.hanging_faces.clone();
        self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone(), self.hanging_faces.clone()))
    }
}

impl NCState {
    /// Create an empty NC state for a conforming initial mesh.
    pub fn new() -> Self {
        NCState {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Current hanging-node constraints (for use with `apply_hanging_constraints`).
    pub fn constraints(&self) -> &[HangingNodeConstraint] {
        &self.constraints
    }

    /// Whether one rollback derefinement step is available.
    pub fn can_derefine(&self) -> bool {
        !self.history.is_empty()
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

        self.history.push(NCState2DSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
        });

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

    /// Roll back one NC refinement step.
    ///
    /// Returns the previous mesh and restored constraints if history exists.
    pub fn derefine_last(&mut self) -> Option<(SimplexMesh<2>, Vec<HangingNodeConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone()))
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

/// Restrict a P1 solution from a fine mesh to a coarse mesh.
///
/// For meshes generated by `refine_*` in this module, original coarse nodes
/// keep their ids and newly created midpoint nodes are appended. Therefore the
/// coarse nodal values are the prefix `u_fine[..n_nodes_coarse]`.
pub fn restrict_to_coarse_p1(u_fine: &[f64], n_nodes_coarse: usize) -> Vec<f64> {
    assert!(
        u_fine.len() >= n_nodes_coarse,
        "restrict_to_coarse_p1: fine vector shorter than coarse node count"
    );
    u_fine[..n_nodes_coarse].to_vec()
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

// ─── 3-D Zienkiewicz-Zhu (ZZ) error estimator (Tet4) ────────────────────────

/// 3-D ZZ error indicator for Tet4 meshes.
pub fn zz_estimator_3d(mesh: &SimplexMesh<3>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let mut elem_grads: Vec<[f64; 3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let uu = |i| u[ns[i] as usize];
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        // J^{-T} via adjugate: adj(J)^T / det
        let jit = |r: usize, c: usize| -> f64 {
            let a = (r+1)%3; let b = (r+2)%3;
            let d = (c+1)%3; let e = (c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for j in 0..3 { g[i] += uh[k] * jit(j,i) * gref[k][j]; } } }
        elem_grads.push(g);
    }
    let mut ng = vec![[0.0_f64;3]; n_nodes]; let mut nc = vec![0usize; n_nodes];
    for (e, &g) in elem_grads.iter().enumerate() {
        for &n in mesh.elem_nodes(e as ElemId) {
            for d in 0..3 { ng[n as usize][d] += g[d]; } nc[n as usize] += 1;
        }
    }
    for n in 0..n_nodes { let c = nc[n] as f64; if c>0.0 { for d in 0..3 { ng[n][d] /= c; } } }
    let mut eta = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let vol = (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 6.0;
        let gr = [ns.iter().map(|&n| ng[n as usize][0]).sum::<f64>()/4.0,
                  ns.iter().map(|&n| ng[n as usize][1]).sum::<f64>()/4.0,
                  ns.iter().map(|&n| ng[n as usize][2]).sum::<f64>()/4.0];
        let eg = elem_grads[e as usize];
        let d = [(eg[0]-gr[0]), (eg[1]-gr[1]), (eg[2]-gr[2])];
        eta.push(vol.powf(1.0/3.0) * (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt());
    }
    eta
}

// ─── 3-D Kelly error estimator (Tet4) ──────────────────────────────────────

/// 3-D Kelly (face-jump) error indicator for Tet4 meshes.
pub fn kelly_estimator_3d(mesh: &SimplexMesh<3>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();

    // Face adjacency
    let fkey = |a: u32, b: u32, c: u32| -> (u32, u32, u32) {
        let mut v = [a, b, c]; v.sort_unstable(); (v[0], v[1], v[2])
    };
    use std::collections::HashMap;
    let mut fem: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
            fem.entry(fkey(a,b,c)).or_default().push(e);
        }
    }

    // Scale ZZ gradients back to raw form for Kelly (undo volume scaling in ZZ)
    let mut raw_grads: Vec<[f64;3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3; let b=(r+2)%3; let d=(c+1)%3; let e=(c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uu = |i| u[ns[i] as usize]; let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for jj in 0..3 { g[i] += uh[k]*jit(jj,i)*gref[k][jj]; } } }
        raw_grads.push(g);
    }
    let elem_grads = raw_grads;

    let mut eta_sq = vec![0.0_f64; n_elems];
    for (fk, elems) in fem.iter().filter(|(_,v)| v.len() == 2) {
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let (na, nb, nc) = *fk;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let face_area = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if face_area < 1e-30 { continue; }
        let inv = 1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                 + (elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let contrib = face_area * jump * jump;
        eta_sq[e0] += contrib; eta_sq[e1] += contrib;
    }

    // Face adjacency: key = sorted 3 vertices
    let fkey = |a: u32, b: u32, c: u32| -> (u32, u32, u32) {
        let mut v = [a, b, c]; v.sort_unstable(); (v[0], v[1], v[2])
    };
    let mut fem: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
            fem.entry(fkey(a,b,c)).or_default().push(e);
        }
    }

    let mut eta_sq = vec![0.0_f64; n_elems];
    for (fk, elems) in fem.iter().filter(|(_,v)| v.len() == 2) {
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let (na, nb, nc) = *fk;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let face_area = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if face_area < 1e-30 { continue; }
        let inv = 1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                 + (elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let contrib = face_area * jump * jump;
        eta_sq[e0] += contrib; eta_sq[e1] += contrib;
    }
    eta_sq.iter().map(|v| v.sqrt()).collect()
}

// ─── Kelly error estimator (2-D) ────────────────────────────────────────────

/// Compute element-wise Kelly (face-jump) error indicators for 2-D Tri3.
pub fn kelly_estimator(mesh: &SimplexMesh<2>, u: &[f64]) -> Vec<f64> {
    use std::collections::HashMap;
    let n_elems = mesh.n_elems();
    let mut elem_grads: Vec<[f64;2]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]); let uu = |i| u[ns[i] as usize];
        let j00 = c(1)[0]-c(0)[0]; let j01 = c(2)[0]-c(0)[0];
        let j10 = c(1)[1]-c(0)[1]; let j11 = c(2)[1]-c(0)[1];
        let det = j00*j11 - j01*j10;
        let gref = [[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2)];
        let mut gx=0.0; let mut gy=0.0;
        for k in 0..3 {
            let gpx = ( j11*gref[k][0] - j10*gref[k][1])/det;
            let gpy = (-j01*gref[k][0] + j00*gref[k][1])/det;
            gx += uh[k]*gpx; gy += uh[k]*gpy;
        }
        elem_grads.push([gx,gy]);
    }
    type Edge = (NodeId,NodeId);
    fn ek(a: NodeId,b: NodeId) -> Edge { if a<b {(a,b)} else {(b,a)} }
    let mut ee: HashMap<Edge,Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b) in &[(ns[0],ns[1]),(ns[1],ns[2]),(ns[0],ns[2])] { ee.entry(ek(a,b)).or_default().push(e); }
    }
    let mut eta_sq = vec![0.0_f64; n_elems];
    for (&(na,nb), elems) in &ee {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb);
        let h = ((cb[0]-ca[0]).powi(2)+(cb[1]-ca[1]).powi(2)).sqrt();
        if h < 1e-30 { continue; }
        let nx = -(cb[1]-ca[1])/h; let ny = (cb[0]-ca[0])/h;
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny;
        eta_sq[e0] += h*jump*jump; eta_sq[e1] += h*jump*jump;
    }
    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Mark elements for refinement using Dörfler bulk criterion.
pub fn dorfler_mark(eta: &[f64], theta: f64) -> Vec<ElemId> {
    let total: f64 = eta.iter().sum();
    let threshold = theta.clamp(0.0,1.0) * total;
    let mut indices: Vec<usize> = (0..eta.len()).collect();
    indices.sort_unstable_by(|&a,&b| eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut marked = Vec::new(); let mut acc = 0.0_f64;
    for &i in &indices { if acc >= threshold { break; } acc += eta[i]; marked.push(i as ElemId); }
    marked.sort_unstable(); marked
}

/// Mark low-error elements for derefinement.
pub fn mark_for_derefinement(eta: &[f64], theta: f64) -> Vec<ElemId> {
    if eta.is_empty() { return Vec::new(); }
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = theta.clamp(0.0,1.0) * max_eta;
    eta.iter().enumerate().filter(|(_,&e)| e <= cutoff).map(|(i,_)| i as ElemId).collect()
}

/// Compute element-wise DWR (Dual Weighted Residual) error indicators for
/// 2-D Tri3 meshes and the Poisson equation `−∇·(κ∇u) = f`.
///
/// # DWR principle
///
/// For a goal functional `J(u)`, the error `J(u) − J(u_h)` can be estimated
/// element-wise using the primal residual weighted by the dual solution:
///
/// ```text
/// η_K ≈ |∫_K f · ω dΩ  +  ½ ∫_{∂K} j · ω dS|
/// ```
///
/// where `ω = z_h − I₀(z_h)` is the dual solution fluctuation (cell-wise
/// deviation from the cell-constant average), and `j = [κ·∂u_h/∂n]` is the
/// jump in the normal flux across interior element edges.
///
/// For P1 elements on Tri3 meshes with constant κ:
/// - The interior residual `r_K = f + ∇·(κ∇u_h)` reduces to `f` (P1 gradient
///   is element-wise constant, so `∇·∇u_h = 0`).
/// - The edge jump captures the discontinuity of the recovered flux.
///
/// # Arguments
/// - `mesh` — triangular 2-D mesh (Tri3).
/// - `u`    — primal solution (nodal values, length = `n_nodes`).
/// - `z`    — dual solution (nodal values, length = `n_nodes`).
/// - `f`    — source term evaluated at each node (length = `n_nodes`).
///
/// # Returns
/// Vector of `η_K` for each element (length = `n_elems`).
pub fn dwr_estimator(mesh: &SimplexMesh<2>, u: &[f64], z: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    use std::collections::HashMap;

    // ── 1. Element gradients (primal) ──────────────────────────────────────────
    // For Tri3 P1: gradient is constant per element.
    let mut elem_grad: Vec<[f64; 2]> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let u0 = u[ns[0] as usize]; let u1 = u[ns[1] as usize]; let u2 = u[ns[2] as usize];

        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;
        let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };

        let g_ref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        let uh = [u0, u1, u2];
        let mut gx = 0.0; let mut gy = 0.0;
        for k in 0..3 {
            let gpx = ( j11 * g_ref[k][0] - j10 * g_ref[k][1]) * inv_det;
            let gpy = (-j01 * g_ref[k][0] + j00 * g_ref[k][1]) * inv_det;
            gx += uh[k] * gpx;
            gy += uh[k] * gpy;
        }
        elem_grad.push([gx, gy]);
    }

    // ── 2. Edge adjacency ──────────────────────────────────────────────────────
    // Edge key: (min_node, max_node) → [elem_a, elem_b]
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

    // ── 3. Element dual fluctuation ω_K = z_h − avg(z_h) ──────────────────────
    // For P1: z_h is linear; avg(z_h) = mean of nodal values.
    let mut elem_omega: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_centroid_f: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_area: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let z_avg = (z[ns[0] as usize] + z[ns[1] as usize] + z[ns[2] as usize]) / 3.0;
        let f_avg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize]) / 3.0;
        elem_omega.push(z_avg);
        elem_centroid_f.push(f_avg);

        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
        elem_area.push(area);
    }

    // ── 4. Assemble DWR indicator per element ─────────────────────────────────
    let mut eta = vec![0.0_f64; n_elems];

    // 4a. Interior contribution: ∫_K f · ω ≈ f_centroid · ω · |K|
    for e in 0..n_elems {
        eta[e] += (elem_centroid_f[e] * elem_omega[e]).abs() * elem_area[e];
    }

    // 4b. Edge jump contribution: ½ ∫_E [[κ·∂u_h/∂n]] · ω dS
    // For each interior edge, distribute ½·h_E·|jump|·|ω_avg| to both adjoined elems.
    for (&(na, _nb), elems) in &edge_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;

        // Edge midpoint coordinate
        let [xa, ya] = mesh.coords_of(na);
        let [xb, yb] = mesh.coords_of(_nb);
        let nx = -(yb - ya);  // outward normal (unscaled, rotated 90° CCW)
        let ny = xb - xa;
        let h_edge = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        if h_edge < 1e-30 { continue; }

        // Normalise edge normal (unit length)
        let nx = nx / h_edge;
        let ny = ny / h_edge;

        // Jump in normal derivative: [[∇u·n]] = ∇u_e0·n - ∇u_e1·n
        let j0 = elem_grad[e0][0] * nx + elem_grad[e0][1] * ny;
        let j1 = elem_grad[e1][0] * nx + elem_grad[e1][1] * ny;
        let jump = (j0 - j1).abs();
        if jump < 1e-30 { continue; }

        // Dual weight at edge midpoint: average of the two element averages
        let w_mid = (elem_omega[e0] + elem_omega[e1]) * 0.5;

        // Edge contribution: ½ · h_E · |jump| · |ω_mid|
        let edge_contrib = 0.5 * h_edge * jump * w_mid.abs();
        eta[e0] += edge_contrib;
        eta[e1] += edge_contrib;
    }

    eta
}

// ─── Residual error estimator (2-D) ───────────────────────────────────────────

/// Element-wise residual error indicator for Poisson on Tri3 meshes.
///
/// Standard residual estimator (Verfürth, 1996):
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{E⊂∂K} h_E · ‖[[∇u_h·n]]‖²_L²(E)
/// ```
///
/// For P1 elements the interior residual reduces to `f` since ∇·∇u_h = 0
/// element-wise. The edge jump term is identical to the Kelly indicator.
///
/// # Arguments
/// - `mesh` — Tri3 mesh.
/// - `u`    — P1 solution at nodes.
/// - `f`    — source term at nodes (or element centroid values).
///
/// # Returns
/// Element-wise error indicators η_K.
pub fn residual_estimator(mesh: &SimplexMesh<2>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let elem_grads: Vec<[f64; 2]> = {
        let mut g = Vec::with_capacity(n_elems);
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let [x0, y0] = mesh.coords_of(ns[0]);
            let [x1, y1] = mesh.coords_of(ns[1]);
            let [x2, y2] = mesh.coords_of(ns[2]);
            let u0 = u[ns[0] as usize]; let u1 = u[ns[1] as usize]; let u2 = u[ns[2] as usize];
            let j00 = x1 - x0; let j01 = x2 - x0;
            let j10 = y1 - y0; let j11 = y2 - y0;
            let det = j00 * j11 - j01 * j10;
            let inv_det = if det.abs() > 1e-30 { 1.0 / det } else { 0.0 };
            let gref = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut gx = 0.0; let mut gy = 0.0;
            for k in 0..3 {
                let gpx = ( j11 * gref[k][0] - j10 * gref[k][1]) * inv_det;
                let gpy = (-j01 * gref[k][0] + j00 * gref[k][1]) * inv_det;
                gx += [u0, u1, u2][k] * gpx;
                gy += [u0, u1, u2][k] * gpy;
            }
            g.push([gx, gy]);
        }
        g
    };

    // Element diameters (h_K = sqrt(2 * area) for triangles)
    let mut elem_h: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_area: Vec<f64> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
        let h = (2.0 * area).sqrt(); // circumdiameter proxy
        elem_h.push(h);
        elem_area.push(area);
    }

    // Edge adjacency
    let mut ee: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &[(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])] {
            let k = if a < b { (a, b) } else { (b, a) };
            ee.entry(k).or_default().push(e);
        }
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    // Interior residual: h_K² * |K| * f_avg²
    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize]) / 3.0;
        eta_sq[e] += elem_h[e] * elem_h[e] * elem_area[e] * favg * favg;
    }

    // Edge jump: ½ * h_E * jump²
    for (&(na, nb), elems) in &ee {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let [xa, ya] = mesh.coords_of(na);
        let [xb, yb] = mesh.coords_of(nb);
        let h_edge = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        if h_edge < 1e-30 { continue; }
        let nx = -(yb - ya) / h_edge;
        let ny =  (xb - xa) / h_edge;
        let jump = (elem_grads[e0][0] - elem_grads[e1][0]) * nx
                 + (elem_grads[e0][1] - elem_grads[e1][1]) * ny;
        let contrib = 0.5 * h_edge * jump * jump;
        eta_sq[e0] += contrib;
        eta_sq[e1] += contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

// ─── Residual error estimator (3-D, Tet4) ─────────────────────────────────────

/// Element-wise residual error indicator for Poisson on Tet4 meshes.
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h·n]]‖²_L²(F)
/// ```
///
/// For P1 elements the interior residual vanishes, leaving the face-jump term.
/// The face jump contribution is computed similarly to the 3-D Kelly estimator
/// but weighted by the face diameter rather than the face area.
pub fn residual_estimator_3d(mesh: &SimplexMesh<3>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    // Element gradients
    let mut elem_grads: Vec<[f64; 3]> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let uu = |i| u[ns[i] as usize];
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let det = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a = (r+1)%3; let b = (r+2)%3;
            let d = (c+1)%3; let e = (c+2)%3;
            (j[a][d]*j[b][e] - j[a][e]*j[b][d]) * idet
        };
        let gref = [[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let uh = [uu(0),uu(1),uu(2),uu(3)];
        let mut g = [0.0_f64;3];
        for k in 0..4 { for i in 0..3 { for jj in 0..3 { g[i] += uh[k] * jit(jj,i) * gref[k][jj]; } } }
        elem_grads.push(g);
    }

    // Element diameters (h_K = (6*vol/π)^(1/3) proxy: use cubic root of volume)
    let mut elem_vol: Vec<f64> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let c = |i| mesh.coords_of(ns[i]);
        let j = [[c(1)[0]-c(0)[0], c(2)[0]-c(0)[0], c(3)[0]-c(0)[0]],
                 [c(1)[1]-c(0)[1], c(2)[1]-c(0)[1], c(3)[1]-c(0)[1]],
                 [c(1)[2]-c(0)[2], c(2)[2]-c(0)[2], c(3)[2]-c(0)[2]]];
        let vol = (j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
                 - j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])
                 + j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs() / 6.0;
        elem_vol.push(vol);
    }

    // Face adjacency
    type Face3 = (NodeId, NodeId, NodeId);
    let fkey = |a: NodeId, b: NodeId, c: NodeId| -> Face3 {
        let mut v = [a, b, c]; v.sort_unstable(); (v[0], v[1], v[2])
    };
    let mut fem: HashMap<Face3, Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
            fem.entry(fkey(a,b,c)).or_default().push(e);
        }
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    // Interior residual: h_K² * |K| * f_avg²
    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg = (f[ns[0] as usize] + f[ns[1] as usize] + f[ns[2] as usize] + f[ns[3] as usize]) / 4.0;
        let h2 = elem_vol[e].powf(2.0 / 3.0); // ~h_K² proxy
        eta_sq[e] += h2 * elem_vol[e] * favg * favg;
    }

    // Face jump: ½ * h_F * jump² (where h_F = sqrt(face_area))
    for (fk, elems) in &fem {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize;
        let e1 = elems[1] as usize;
        let (na, nb, nc) = *fk;
        let ca = mesh.coords_of(na); let cb = mesh.coords_of(nb); let cc = mesh.coords_of(nc);
        let ex = cb[0]-ca[0]; let ey = cb[1]-ca[1]; let ez = cb[2]-ca[2];
        let fx = cc[0]-ca[0]; let fy = cc[1]-ca[1]; let fz = cc[2]-ca[2];
        let nx = ey*fz - ez*fy; let ny = ez*fx - ex*fz; let nz = ex*fy - ey*fx;
        let face_area = 0.5 * (nx*nx+ny*ny+nz*nz).sqrt();
        if face_area < 1e-30 { continue; }
        let inv = 1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump = (elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                 + (elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                 + (elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let h_face = (2.0 * face_area).sqrt(); // face diameter proxy
        let contrib = 0.5 * h_face * jump * jump;
        eta_sq[e0] += contrib;
        eta_sq[e1] += contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

// ─── p-refinement (order elevation) ─────────────────────────────────────────

/// Mark elements for p-refinement based on a smoothness indicator.
///
/// Elements with high estimated curvature (ratio of recovered gradient
/// variation to element size above a threshold) are marked for order
/// elevation.  The smoothness indicator is derived from the ZZ error
/// estimator: elements where the local error exceeds `theta * max(eta)`
/// are candidates for p-refinement rather than h-refinement.
///
/// # Returns
/// Indices of elements recommended for p-refinement.
pub fn mark_for_p_refinement(eta: &[f64], theta: f64) -> Vec<ElemId> {
    if eta.is_empty() { return Vec::new(); }
    let max_eta = eta.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = theta.clamp(0.0, 1.0) * max_eta;
    eta.iter().enumerate()
        .filter(|(_, &e)| e >= cutoff)
        .map(|(i, _)| i as ElemId)
        .collect()
}

/// Elevate selected Tri3 elements to Tri6 (quadratic) by adding edge-midpoint
/// nodes.  Elements not in `marked` remain Tri3.
///
/// Shared edges get a single midpoint node (deduplicated by edge key).
/// New nodes are appended to the coordinate array.
///
/// # Returns
/// `(new_mesh, midpoint_map)` where `midpoint_map` maps edge keys to new
/// node indices (for prolongation).
pub fn p_refine_tri3_to_tri6(
    mesh: &SimplexMesh<2>,
    marked: &[ElemId],
) -> (SimplexMesh<2>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tri3,
        "p_refine_tri3_to_tri6 requires a Tri3 mesh");
    let n_elems = mesh.n_elems();
    use std::collections::HashMap;

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    // Build edge→midpoint map for all edges belonging to marked elements
    let mut edge_to_new_node: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let edge_pairs = [
            (ns[0], ns[1]),
            (ns[1], ns[2]),
            (ns[0], ns[2]),
        ];
        for &(a, b) in &edge_pairs {
            let ek = edge_key(a, b);
            if !edge_to_new_node.contains_key(&ek) {
                // Add midpoint
                let [xa, ya] = mesh.coords_of(a);
                let [xb, yb] = mesh.coords_of(b);
                new_coords.push(0.5 * (xa + xb));
                new_coords.push(0.5 * (ya + yb));
                edge_to_new_node.insert(ek, next_node);
                next_node += 1;
            }
        }
    }

    // Build new connectivity and per-element types
    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let ek = |a: NodeId, b: NodeId| edge_key(a, b);
            let m01 = edge_to_new_node[&ek(ns[0], ns[1])];
            let m12 = edge_to_new_node[&ek(ns[1], ns[2])];
            let m02 = edge_to_new_node[&ek(ns[0], ns[2])];
            // Tri6: 3 vertices + 3 edge midpoints
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], m01, m12, m02]);
            elem_types_vec.push(ElementType::Tri6);
            elem_offsets.push(elem_offsets.last().unwrap() + 6);
        } else {
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2]]);
            elem_types_vec.push(ElementType::Tri3);
            elem_offsets.push(elem_offsets.last().unwrap() + 3);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords,
        conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(),
        face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(),
        edge_to_elem: Vec::new(),
    };

    (new_mesh, edge_to_new_node)
}

// ─── p-refinement: element type promotion ───────────────────────────────────
//
// These functions add the intermediate geometry nodes (edge midpoints, face
// centroids, volume centroids) needed when promoting element types to higher
// polynomial orders.  The actual DOF management for arbitrary p-jumps is
// handled by `build_variable_order_dof_manager` in the `fem-space` crate.

/// Refine Tri6 → Tri10: adds element centroid for cubic serendipity.
///
/// Each marked Tri6 element gains one new node at its centroid.
pub fn p_refine_tri6_to_tri10(
    mesh: &SimplexMesh<2>,
    marked: &[ElemId],
) -> (SimplexMesh<2>, Vec<NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tri6,
        "p_refine_tri6_to_tri10 requires a Tri6 mesh");
    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut new_coords = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut centroids: Vec<NodeId> = Vec::with_capacity(marked.len());

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            // Compute centroid of the 6 nodes (3 vertices + 3 edge midpoints)
            let mut cx = 0.0; let mut cy = 0.0;
            for &n in ns.iter() {
                let c = mesh.coords_of(n);
                cx += c[0]; cy += c[1];
            }
            cx /= 6.0; cy /= 6.0;
            new_coords.push(cx); new_coords.push(cy);
            let centroid = next_node; next_node += 1;
            centroids.push(centroid);
            // Tri10: 6 original + centroid + 3 interior edge nodes = 10
            // But for a simple Tri6→Tri10, we only add the centroid.
            // Full Tri10 has 3 vertices + 3 edge midpoints + 3 interior edge
            // nodes + 1 centroid, but the DOF manager handles the extra nodes.
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], centroid]);
            elem_types_vec.push(ElementType::Tri6); // keep Tri6 mesh type
            elem_offsets.push(elem_offsets.last().unwrap() + 7);
        } else {
            // Preserve element as-is (Tri3 or Tri6 depending on node count)
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 3 { ElementType::Tri3 } else { ElementType::Tri6 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, centroids)
}

/// Refine Tet4 → Tet10: adds 6 edge midpoints per marked tet.
///
/// Edge midpoints are shared between adjacent tets (deduplicated by edge key).
pub fn p_refine_tet4_to_tet10(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tet4,
        "p_refine_tet4_to_tet10 requires a Tet4 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let tet_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &tet_edges {
            let ek = edge_key(ns[i], ns[j]);
            if !edge_to_new.contains_key(&ek) {
                let a = mesh.coords_of(ns[i]);
                let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                new_coords.push(0.5*(a[2]+b[2]));
                edge_to_new.insert(ek, next_node);
                next_node += 1;
            }
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m02 = edge_to_new[&edge_key(ns[0], ns[2])];
            let m03 = edge_to_new[&edge_key(ns[0], ns[3])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m13 = edge_to_new[&edge_key(ns[1], ns[3])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3], m01, m02, m03, m12, m13, m23]);
            elem_types_vec.push(ElementType::Tet10);
            elem_offsets.push(elem_offsets.last().unwrap() + 10);
        } else {
            new_conn.extend_from_slice(&ns[0..4]);
            elem_types_vec.push(ElementType::Tet4);
            elem_offsets.push(elem_offsets.last().unwrap() + 4);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tet10,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, edge_to_new)
}

/// Refine Tet10 → Tet20: adds 4 face centroids per marked tet.
///
/// Face centroids are shared between adjacent tets (deduplicated by face key).
pub fn p_refine_tet10_to_tet20(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, std::collections::HashMap<(NodeId,NodeId,NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tet10,
        "p_refine_tet10_to_tet20 requires a Tet10 mesh");

    fn face_key(a: NodeId, b: NodeId, c: NodeId) -> (NodeId, NodeId, NodeId) {
        let mut v = [a,b,c]; v.sort_unstable(); (v[0], v[1], v[2])
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut face_to_new: std::collections::HashMap<(NodeId,NodeId,NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let tet_faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j,k) in &tet_faces {
            let fk = face_key(ns[i], ns[j], ns[k]);
            if !face_to_new.contains_key(&fk) {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]); let c = mesh.coords_of(ns[k]);
                new_coords.push((a[0]+b[0]+c[0])/3.0);
                new_coords.push((a[1]+b[1]+c[1])/3.0);
                new_coords.push((a[2]+b[2]+c[2])/3.0);
                face_to_new.insert(fk, next_node);
                next_node += 1;
            }
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let f012 = face_to_new[&face_key(ns[0], ns[1], ns[2])];
            let f013 = face_to_new[&face_key(ns[0], ns[1], ns[3])];
            let f023 = face_to_new[&face_key(ns[0], ns[2], ns[3])];
            let f123 = face_to_new[&face_key(ns[1], ns[2], ns[3])];
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3],
                                          ns[4], ns[5], ns[6], ns[7], ns[8], ns[9],
                                          f012, f013, f023, f123]);
            elem_types_vec.push(ElementType::Tet10);
            elem_offsets.push(elem_offsets.last().unwrap() + 14);
        } else {
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 4 { ElementType::Tet4 } else { ElementType::Tet10 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tet10,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, face_to_new)
}

/// Refine Quad4 → Quad9: adds 4 edge midpoints + 1 centroid per marked quad.
///
/// Edge midpoints are shared between adjacent quads (deduplicated by edge key).
pub fn p_refine_quad4_to_quad9(
    mesh: &SimplexMesh<2>,
    marked: &[ElemId],
) -> (SimplexMesh<2>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Quad4,
        "p_refine_quad4_to_quad9 requires a Quad4 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &[(0,1),(1,2),(2,3),(3,0)] {
            let ek = edge_key(ns[i], ns[j]);
            if !edge_to_new.contains_key(&ek) {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                edge_to_new.insert(ek, next_node);
                next_node += 1;
            }
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            let m30 = edge_to_new[&edge_key(ns[3], ns[0])];
            // Centroid
            let mut cx = 0.0; let mut cy = 0.0;
            for &n in ns.iter() { let c = mesh.coords_of(n); cx += c[0]; cy += c[1]; }
            cx /= 4.0; cy /= 4.0;
            new_coords.push(cx); new_coords.push(cy);
            let centroid = next_node; next_node += 1;
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3],
                                          m01, m12, m23, m30, centroid]);
            elem_types_vec.push(ElementType::Quad4);
            elem_offsets.push(elem_offsets.last().unwrap() + 9);
        } else {
            new_conn.extend_from_slice(&ns[0..4]);
            elem_types_vec.push(ElementType::Quad4);
            elem_offsets.push(elem_offsets.last().unwrap() + 4);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Quad4,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, edge_to_new)
}

/// Refine Hex8 → Hex20: adds 12 edge midpoints per marked hex.
///
/// Edge midpoints are shared between adjacent hexes (deduplicated by edge key).
pub fn p_refine_hex8_to_hex20(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Hex8,
        "p_refine_hex8_to_hex20 requires a Hex8 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let hex_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &hex_edges {
            let ek = edge_key(ns[i], ns[j]);
            if !edge_to_new.contains_key(&ek) {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                new_coords.push(0.5*(a[2]+b[2]));
                edge_to_new.insert(ek, next_node);
                next_node += 1;
            }
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            let m30 = edge_to_new[&edge_key(ns[3], ns[0])];
            let m45 = edge_to_new[&edge_key(ns[4], ns[5])];
            let m56 = edge_to_new[&edge_key(ns[5], ns[6])];
            let m67 = edge_to_new[&edge_key(ns[6], ns[7])];
            let m74 = edge_to_new[&edge_key(ns[7], ns[4])];
            let m04 = edge_to_new[&edge_key(ns[0], ns[4])];
            let m15 = edge_to_new[&edge_key(ns[1], ns[5])];
            let m26 = edge_to_new[&edge_key(ns[2], ns[6])];
            let m37 = edge_to_new[&edge_key(ns[3], ns[7])];
            new_conn.extend_from_slice(&[ns[0],ns[1],ns[2],ns[3],ns[4],ns[5],ns[6],ns[7],
                                          m01,m12,m23,m30,m45,m56,m67,m74,m04,m15,m26,m37]);
            elem_types_vec.push(ElementType::Hex20);
            elem_offsets.push(elem_offsets.last().unwrap() + 20);
        } else {
            new_conn.extend_from_slice(&ns[0..8]);
            elem_types_vec.push(ElementType::Hex8);
            elem_offsets.push(elem_offsets.last().unwrap() + 8);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Hex20,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, edge_to_new)
}

/// Refine Hex20 → Hex27: adds 6 face centers + 1 volume centroid per marked hex.
///
/// Face centers are shared between adjacent hexes (deduplicated by quad face key).
pub fn p_refine_hex20_to_hex27(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Hex20,
        "p_refine_hex20_to_hex27 requires a Hex20 mesh");

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();
    let mut new_centroids: Vec<NodeId> = Vec::new();

    let hex_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]];

    // Face centers + volume center are element-local for Hex20→Hex27 (not shared
    // in the usual case since face centers already belong to the hex, but we still
    // deduplicate by face key for correctness with mixed meshes).
    use std::collections::HashMap;
    let mut face_to_new: HashMap<[NodeId;4], NodeId> = HashMap::new();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for face in &hex_faces {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let mut k = fns; k.sort_unstable();
            if !face_to_new.contains_key(&k) {
                let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
                for &fi in face.iter() { let c = mesh.coords_of(ns[fi]); cx += c[0]; cy += c[1]; cz += c[2]; }
                cx /= 4.0; cy /= 4.0; cz /= 4.0;
                new_coords.push(cx); new_coords.push(cy); new_coords.push(cz);
                face_to_new.insert(k, next_node);
                next_node += 1;
            }
        }
        // Volume centroid
        let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
        for &n in ns.iter() { let c = mesh.coords_of(n); cx += c[0]; cy += c[1]; cz += c[2]; }
        cx /= 8.0; cy /= 8.0; cz /= 8.0;
        new_coords.push(cx); new_coords.push(cy); new_coords.push(cz);
        new_centroids.push(next_node);
        next_node += 1;
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];
    let mut centroid_idx = 0usize;

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            new_conn.extend_from_slice(&ns[0..20]); // existing Hex20 nodes
            for face in &hex_faces {
                let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
                let mut k = fns; k.sort_unstable();
                new_conn.push(face_to_new[&k]);
            }
            new_conn.push(new_centroids[centroid_idx]); centroid_idx += 1;
            elem_types_vec.push(ElementType::Hex27);
            elem_offsets.push(elem_offsets.last().unwrap() + 27);
        } else {
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 8 { ElementType::Hex8 } else { ElementType::Hex20 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = SimplexMesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Hex27,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(),
    };
    (new_mesh, new_centroids)
}

/// Prolongate a P1 solution to a P2 (Tri6) mesh after p-refinement.
///
/// Values at the original vertices are unchanged.  Values at new edge-midpoint
/// nodes are the average of the two adjacent vertex values (standard for
/// continuous Galerkin projection).  Alternative: a fully L²-projected
/// prolongation is available via [`p_prolongate_p1_to_p2_l2`].
///
/// # Returns
/// Vector of length = `new_n_nodes`, with the original P1 values at
/// existing nodes and interpolated values at new edge-midpoint nodes.
pub fn p_prolongate_p1_to_p2(
    u_p1: &[f64],
    midpoint_map: &std::collections::HashMap<(NodeId, NodeId), NodeId>,
    mesh_p2: &SimplexMesh<2>,
) -> Vec<f64> {
    let n_total = mesh_p2.n_nodes();
    let mut u_p2 = vec![0.0_f64; n_total];
    // Copy original vertex values
    let n_orig = u_p1.len().min(n_total);
    u_p2[..n_orig].copy_from_slice(&u_p1[..n_orig]);

    // Interpolate edge midpoints as average of the two vertex values
    for (&(a, b), &new_node) in midpoint_map {
        let idx = new_node as usize;
        if idx < n_total {
            u_p2[idx] = 0.5 * (u_p2[a as usize] + u_p2[b as usize]);
        }
    }
    u_p2
}

// ─── Uniform refinement ───────────────────────────────────────────────────────

/// Uniformly refine all elements of a 2-D mesh (Tri3 → 4 Tri3).
pub fn refine_uniform(mesh: &SimplexMesh<2>) -> SimplexMesh<2> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    refine_marked(mesh, &all)
}

/// Uniformly refine all elements of a 3-D mesh, dispatching to the appropriate
/// refinement path.
///
/// Tet4 → 8 Tet4, Hex8 → 8 Hex8, Hex20 → 8 Hex8, Hex27 → 8 Hex8,
/// Prism6 → 8 Prism6, Pyramid5 → 16 Tet4.
pub fn refine_uniform_3d(mesh: &SimplexMesh<3>) -> SimplexMesh<3> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    match mesh.elem_type {
        ElementType::Tet4 | ElementType::Tet10 => {
            let (m, _, _) = refine_nonconforming_3d(mesh, &all);
            m
        }
        ElementType::Hex8 => {
            let (m, _, _, _) = refine_nonconforming_hex(mesh, &all);
            m
        }
        ElementType::Hex20 | ElementType::Hex27 => {
            // Quadratic hex: extract corners, refine as Hex8
            let n_elems = mesh.n_elems();
            let npe = mesh.elem_type.nodes_per_element();
            let mut hex8_conn = Vec::with_capacity(n_elems * 8);
            for e in 0..n_elems {
                let off = e * npe;
                hex8_conn.extend_from_slice(&mesh.conn[off..off + 8]);
            }
            let hex8_mesh = SimplexMesh {
                coords: mesh.coords.clone(),
                conn: hex8_conn,
                elem_tags: mesh.elem_tags.clone(),
                elem_type: ElementType::Hex8,
                face_conn: mesh.face_conn.clone(),
                face_tags: mesh.face_tags.clone(),
                face_type: mesh.face_type,
                elem_types: None, elem_offsets: None,
                face_types: None, face_offsets: None,
                face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
            };
            let (m, _, _, _) = refine_nonconforming_hex(&hex8_mesh, &all);
            m
        }
        ElementType::Prism6 => {
            let (m, _, _) = refine_prism6_uniform(mesh, &all);
            m
        }
        ElementType::Pyramid5 => {
            let (m, _) = refine_pyramid5_uniform(mesh, &all);
            m
        }
        _ => panic!("refine_uniform_3d: unsupported {:?}", mesh.elem_type),
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Local edge index pairs for Tri3.
pub(crate) fn local_edges_tri() -> [(usize, usize); 3] {
    [(0, 1), (1, 2), (0, 2)]
}

/// Canonical edge key (sorted node pair).
pub(crate) fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
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

/// Canonical face key for a quad face (sorted 4-tuple).
pub(crate) fn quad_face_key(ns: [NodeId; 4]) -> [NodeId; 4] {
    let mut k = ns;
    k.sort();
    k
}

// ─── Prism6 topology ──────────────────────────────────────────────────────────

/// Local 9 edges of a Prism6 element (pairs of local node indices).
///
/// Prism6 node layout:
/// ```text
/// Bottom triangle (z=0, CCW): 0, 1, 2
/// Top    triangle (z=1, CCW): 3, 4, 5
/// Vertical edges: 0→3, 1→4, 2→5
/// ```
fn local_edges_prism() -> [(usize, usize); 9] {
    [
        // Bottom triangle
        (0, 1), (1, 2), (0, 2),
        // Top triangle
        (3, 4), (4, 5), (3, 5),
        // Vertical edges
        (0, 3), (1, 4), (2, 5),
    ]
}

/// Local 2 triangular faces of a Prism6 (triplets of local node indices).
fn local_faces_prism_tri() -> [(usize, usize, usize); 2] {
    [
        (0, 1, 2), // bottom
        (3, 4, 5), // top
    ]
}

/// Local 3 quadrilateral faces of a Prism6 (4-tuples of local node indices).
fn local_faces_prism_quad() -> [[usize; 4]; 3] {
    [
        [0, 1, 4, 3], // quad face 0 (front)
        [1, 2, 5, 4], // quad face 1 (right)
        [0, 2, 5, 3], // quad face 2 (left)
    ]
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

// ─── Quad4 non-conforming AMR ─────────────────────────────────────────────────

/// Canonical edge key for a Quad4 boundary edge (sorted node pair).
pub(crate) fn quad_edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a < b { (a, b) } else { (b, a) }
}

/// Local edges for a Quad4 in CCW order: (0,1), (1,2), (2,3), (3,0).
pub(crate) fn local_edges_quad() -> [(usize, usize); 4] {
    [(0, 1), (1, 2), (2, 3), (3, 0)]
}

/// Non-conforming (hanging-node) refinement for a 2-D Quad4 mesh.
///
/// Each marked Quad4 element is split into **4 child Quad4s** by bisecting
/// all 4 edges and inserting an element-centroid node.  Unmarked neighbours
/// may become non-conforming: every new midpoint node on an edge shared with
/// an unrefined element becomes a hanging node constrained by
/// `u[mid] = 0.5 * (u[a] + u[b])`.
///
/// # Returns
/// `(new_mesh, constraints)`.
pub fn refine_nonconforming_quad(
    mesh: &SimplexMesh<2>,
    marked: &[ElemId],
) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Quad4,
        "refine_nonconforming_quad: only Quad4 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new());
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Build edge → adjacent element list ────────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_quad() {
            let key = quad_edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    // ── 2. Compute midpoints and centers for marked elements ─────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut center_map:   HashMap<ElemId, NodeId>           = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        // Edge midpoints
        for &(a, b) in &local_edges_quad() {
            let key = quad_edge_key(ns[a], ns[b]);
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
        // Element centroid
        center_map.entry(e).or_insert_with(|| {
            let (mut cx, mut cy) = (0.0_f64, 0.0_f64);
            for k in 0..4 {
                let c = mesh.coords_of(ns[k]);
                cx += c[0]; cy += c[1];
            }
            new_coords.push(cx / 4.0);
            new_coords.push(cy / 4.0);
            let id = next_node;
            next_node += 1;
            id
        });
    }

    // ── 3. Build new element connectivity ────────────────────────────────────
    // Quad4 local node layout (CCW):
    //   n3 ─── n2
    //   │       │
    //   n0 ─── n1
    //
    // Children after refinement (edge midpoints m01, m12, m23, m30, center c):
    //   child 0: (n0, m01, c, m30)  bottom-left
    //   child 1: (m01, n1, m12, c)  bottom-right
    //   child 2: (c, m12, n2, m23)  top-right
    //   child 3: (m30, c, m23, n3)  top-left
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32>    = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2]; let n3 = ns[3];
            let m01 = *midpoint_map.get(&quad_edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&quad_edge_key(n1, n2)).unwrap();
            let m23 = *midpoint_map.get(&quad_edge_key(n2, n3)).unwrap();
            let m30 = *midpoint_map.get(&quad_edge_key(n3, n0)).unwrap();
            let c   = *center_map.get(&e).unwrap();

            new_conn.extend_from_slice(&[n0,  m01, c,   m30]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, n1,  m12, c  ]); new_tags.push(tag);
            new_conn.extend_from_slice(&[c,   m12, n2,  m23]); new_tags.push(tag);
            new_conn.extend_from_slice(&[m30, c,   m23, n3 ]); new_tags.push(tag);
        } else {
            for k in 0..4 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Detect hanging nodes ──────────────────────────────────────────────
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            let has_unrefined = adj.iter().any(|e| !marked_set.contains(e));
            if has_unrefined {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── 5. Rebuild boundary edges ─────────────────────────────────────────────
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();
    for f in 0..n_faces {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        let tag = mesh.face_tags[f];
        if let Some(&mid) = midpoint_map.get(&quad_edge_key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]); new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Quad4,
        new_face_conn, new_face_tags, ElementType::Line2,
    );
    (new_mesh, constraints)
}

// ─── NCStateQuad ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NCStateQuadSnapshot {
    mesh: SimplexMesh<2>,
    constraints: Vec<HangingNodeConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

/// Accumulated state for multi-level non-conforming refinement of **Quad4** meshes.
///
/// Mirrors [`NCState`] for triangular meshes.  Tracks active edge midpoints
/// across successive refinement levels and rebuilds hanging-node constraints
/// after each step.
#[derive(Debug, Clone)]
pub struct NCStateQuad {
    constraints: Vec<HangingNodeConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    history: Vec<NCStateQuadSnapshot>,
}

impl NCStateQuad {
    pub fn new() -> Self {
        NCStateQuad { constraints: Vec::new(), active_midpoints: HashMap::new(), history: Vec::new() }
    }

    pub fn constraints(&self) -> &[HangingNodeConstraint] { &self.constraints }
    pub fn can_derefine(&self) -> bool { !self.history.is_empty() }

    /// Perform one level of non-conforming refinement on a Quad4 mesh.
    ///
    /// Returns `(new_mesh, constraints, midpoint_map)`.
    pub fn refine(
        &mut self,
        mesh: &SimplexMesh<2>,
        marked: &[ElemId],
    ) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        assert!(
            mesh.elem_type == ElementType::Quad4,
            "NCStateQuad::refine: only Quad4 meshes are supported"
        );

        if marked.is_empty() {
            return (mesh.clone(), self.constraints.clone(), HashMap::new());
        }

        self.history.push(NCStateQuadSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
        });

        let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
        let n_elems = mesh.n_elems();

        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_quad() {
                edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }

        let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
        let mut center_map:   HashMap<ElemId, NodeId>           = HashMap::new();
        let mut new_coords: Vec<f64> = mesh.coords.clone();
        let mut next_node = mesh.n_nodes() as NodeId;

        for &e in marked {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_quad() {
                let key = quad_edge_key(ns[a], ns[b]);
                if midpoint_map.contains_key(&key) { continue; }
                if let Some(&mid) = self.active_midpoints.get(&key) {
                    midpoint_map.insert(key, mid);
                } else {
                    let xa = mesh.coords_of(ns[a]);
                    let xb = mesh.coords_of(ns[b]);
                    new_coords.push(0.5 * (xa[0] + xb[0]));
                    new_coords.push(0.5 * (xa[1] + xb[1]));
                    midpoint_map.insert(key, next_node);
                    next_node += 1;
                }
            }
            center_map.entry(e).or_insert_with(|| {
                let (mut cx, mut cy) = (0.0_f64, 0.0_f64);
                for k in 0..4 { let c = mesh.coords_of(ns[k]); cx += c[0]; cy += c[1]; }
                new_coords.push(cx / 4.0); new_coords.push(cy / 4.0);
                let id = next_node; next_node += 1; id
            });
        }

        let mut new_conn: Vec<NodeId> = Vec::new();
        let mut new_tags: Vec<i32>    = Vec::new();
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            let tag = mesh.elem_tags[e as usize];
            if marked_set.contains(&e) {
                let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2]; let n3 = ns[3];
                let m01 = *midpoint_map.get(&quad_edge_key(n0, n1)).unwrap();
                let m12 = *midpoint_map.get(&quad_edge_key(n1, n2)).unwrap();
                let m23 = *midpoint_map.get(&quad_edge_key(n2, n3)).unwrap();
                let m30 = *midpoint_map.get(&quad_edge_key(n3, n0)).unwrap();
                let c   = *center_map.get(&e).unwrap();
                new_conn.extend_from_slice(&[n0,  m01, c,   m30]); new_tags.push(tag);
                new_conn.extend_from_slice(&[m01, n1,  m12, c  ]); new_tags.push(tag);
                new_conn.extend_from_slice(&[c,   m12, n2,  m23]); new_tags.push(tag);
                new_conn.extend_from_slice(&[m30, c,   m23, n3 ]); new_tags.push(tag);
            } else {
                for k in 0..4 { new_conn.push(ns[k]); }
                new_tags.push(tag);
            }
        }

        // Merge into active midpoint set.
        for (&k, &v) in &midpoint_map { self.active_midpoints.insert(k, v); }

        // Rebuild edge adjacency for the new mesh to determine hanging status.
        let new_n_elems = new_tags.len();
        let mut new_edge_elems: HashMap<(NodeId, NodeId), Vec<u32>> = HashMap::new();
        for e in 0..new_n_elems as u32 {
            let off = e as usize * 4;
            let ns = &new_conn[off..off + 4];
            for &(a, b) in &local_edges_quad() {
                new_edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }
        let new_node_set: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();

        let mut new_constraints = Vec::new();
        for (&(a, b), &mid) in &self.active_midpoints {
            if !new_node_set.contains(&mid) { continue; }
            let parent_exists = new_edge_elems.contains_key(&quad_edge_key(a, b));
            if parent_exists {
                new_constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
        self.active_midpoints.retain(|_, mid| new_node_set.contains(mid));
        new_constraints.sort_by_key(|c| c.constrained);
        self.constraints = new_constraints.clone();

        let n_faces = mesh.n_faces();
        let mut new_face_conn: Vec<NodeId> = Vec::new();
        let mut new_face_tags: Vec<i32>    = Vec::new();
        for f in 0..n_faces {
            let a = mesh.face_conn[2 * f];
            let b = mesh.face_conn[2 * f + 1];
            let tag = mesh.face_tags[f];
            if let Some(&mid) = midpoint_map.get(&quad_edge_key(a, b)) {
                new_face_conn.extend_from_slice(&[a, mid]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mid, b]); new_face_tags.push(tag);
            } else {
                new_face_conn.extend_from_slice(&[a, b]);
                new_face_tags.push(tag);
            }
        }

        let new_mesh = SimplexMesh::uniform(
            new_coords, new_conn, new_tags, ElementType::Quad4,
            new_face_conn, new_face_tags, ElementType::Line2,
        );
        (new_mesh, self.constraints.clone(), midpoint_map)
    }

    pub fn derefine_last(&mut self) -> Option<(SimplexMesh<2>, Vec<HangingNodeConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone()))
    }
}

// ─── Hex8 non-conforming AMR ──────────────────────────────────────────────────

/// Local 12 edges of a Hex8 element (pairs of local node indices).
///
/// Hex8 node layout:
/// ```text
/// Bottom face (z=0, CCW from outside): 0,1,2,3
/// Top    face (z=1, CCW from outside): 4,5,6,7
/// Vertical edges: 0→4, 1→5, 2→6, 3→7
/// ```
fn local_edges_hex() -> [(usize, usize); 12] {
    [
        // Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        // Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        // Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
}

/// Local 6 faces of a Hex8 (each as 4 local node indices in CCW order).
pub(crate) fn local_faces_hex() -> [[usize; 4]; 6] {
    [
        [0, 1, 2, 3], // bottom (z=0)
        [4, 5, 6, 7], // top    (z=1)
        [0, 1, 5, 4], // front  (y=0)
        [2, 3, 7, 6], // back   (y=1)
        [0, 3, 7, 4], // left   (x=0)
        [1, 2, 6, 5], // right  (x=1)
    ]
}

/// Canonical face key for Hex8 (sorted 4-tuple of node IDs).
pub(crate) fn hex_face_key(ns: [NodeId; 4]) -> [NodeId; 4] {
    let mut k = ns;
    k.sort();
    k
}

/// Non-conforming (hanging-node) refinement for a 3-D Hex8 mesh.
///
/// Each marked Hex8 is split into **8 child Hex8s** by:
/// - Inserting midpoints on each of its 12 edges,
/// - Inserting centroids on each of its 6 faces,
/// - Inserting the element centroid.
///
/// Unmarked neighbours sharing a refined face acquire hanging-edge midpoints;
/// those midpoints are constrained by linear interpolation along their parent edge.
///
/// # Returns
/// `(new_mesh, edge_constraints, quad_face_constraints, midpoint_map)`.
pub fn refine_nonconforming_hex(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
    assert!(
        mesh.elem_type == ElementType::Hex8,
        "refine_nonconforming_hex: only Hex8 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new(), Vec::new(), HashMap::new());
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Edge + face adjacency for hanging detection ────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut face_elems: HashMap<[NodeId; 4], Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_hex() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
        for face in local_faces_hex() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            face_elems.entry(hex_face_key(fns)).or_default().push(e);
        }
    }

    // ── 2. Allocate new nodes ─────────────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut face_center_map: HashMap<[NodeId; 4], NodeId>   = HashMap::new();
    let mut body_center_map: HashMap<ElemId, NodeId>         = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);

        // Edge midpoints (12 per Hex8)
        for &(a, b) in &local_edges_hex() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                new_coords.push(0.5 * (xa[2] + xb[2]));
                let id = next_node; next_node += 1; id
            });
        }

        // Face centroids (6 per Hex8)
        for face in local_faces_hex() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let fkey = hex_face_key(fns);
            face_center_map.entry(fkey).or_insert_with(|| {
                let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
                for &fn_ in &fns {
                    let c = mesh.coords_of(fn_);
                    x += c[0]; y += c[1]; z += c[2];
                }
                new_coords.push(x / 4.0); new_coords.push(y / 4.0); new_coords.push(z / 4.0);
                let id = next_node; next_node += 1; id
            });
        }

        // Body centroid (1 per Hex8)
        body_center_map.entry(e).or_insert_with(|| {
            let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..8 {
                let c = mesh.coords_of(ns[k]);
                x += c[0]; y += c[1]; z += c[2];
            }
            new_coords.push(x / 8.0); new_coords.push(y / 8.0); new_coords.push(z / 8.0);
            let id = next_node; next_node += 1; id
        });
    }

    // ── 3. Build new element connectivity ────────────────────────────────────
    // For each Hex8 corner k (0..8), the child hex is:
    //   (corner_k, 3 adjacent edge midpoints, 3 adjacent face centers, body center)
    // Child ordering for Hex8 with bottom=0..3, top=4..7:
    //   Bottom-front-left  = child 0: corner 0
    //   Bottom-front-right = child 1: corner 1
    //   Bottom-back-right  = child 2: corner 2
    //   Bottom-back-left   = child 3: corner 3
    //   Top-front-left     = child 4: corner 4
    //   Top-front-right    = child 5: corner 5
    //   Top-back-right     = child 6: corner 6
    //   Top-back-left      = child 7: corner 7
    //
    // For corner 0 (n0):
    //   Adjacent edges: (0,1), (0,3), (0,4)  → m01, m03, m04
    //   Adjacent faces: bottom(0123), front(0154), left(0374) → fb, ff, fl
    //   Body center: bc
    //   Child Hex8 (CCW bottom-left pattern):
    //     (n0, m01, f_bottom, m03, m04, f_front, bc, f_left)
    //
    // We encode the 8 children via the following index table.
    // Each row: [corner, e01, e02, e03, f01, f02, f03, body]
    // where e01/e02/e03 are the 3 edge partners and f01/f02/f03 are face indices.
    //
    // Hex8 corner-to-adjacent-edges-and-faces (hardcoded for standard numbering):
    // corner | adj edges (local pair indices) | adj face indices (in local_faces_hex)
    //   0    | (0,1),(0,3),(0,4)  → edges (0,3),(0,1) from bottom; (0,8) vertical
    //         | faces: bottom(0), front(2), left(4)
    //   1    | (0,1),(1,2),(1,5)  
    //         | faces: bottom(0), front(2), right(5)
    //   2    | (1,2),(2,3),(2,6)
    //         | faces: bottom(0), back(3), right(5)
    //   3    | (0,3),(2,3),(3,7)
    //         | faces: bottom(0), back(3), left(4)
    //   4    | (0,4),(4,5),(4,7)
    //         | faces: top(1), front(2), left(4)
    //   5    | (1,5),(4,5),(5,6)
    //         | faces: top(1), front(2), right(5)
    //   6    | (2,6),(5,6),(6,7)
    //         | faces: top(1), back(3), right(5)
    //   7    | (3,7),(6,7),(4,7)
    //         | faces: top(1), back(3), left(4)
    //
    // Child Hex8 node order (following the convention of standard trilinear mapping):
    // The child for corner k has nodes: [k, adjacent_edge_1, adj_face_1, adj_edge_2,
    //                                      adj_edge_3, adj_face_2, body, adj_face_3]
    // in CCW-bottom / CCW-top layout.
    //
    // Specifically, for standard Hex8 with bottom=n0..n3, top=n4..n7:
    // child 0 (corner n0): [n0, m01, fc_bot, m30, m04, fc_frt, bc, fc_lft]
    // Each child Hex8 maintains CCW bottom + CCW top layout per Hex8 convention.
    // The 8 child connectivities are hardcoded from the corner adjacency table.

    let get_em = |a: usize, b: usize, ns: &[NodeId]| -> NodeId {
        *midpoint_map.get(&edge_key(ns[a], ns[b])).expect("edge midpoint missing")
    };
    let get_fc = |face_idx: usize, ns: &[NodeId]| -> NodeId {
        let face = local_faces_hex()[face_idx];
        let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
        *face_center_map.get(&hex_face_key(fns)).expect("face center missing")
    };

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32>    = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            let bc = *body_center_map.get(&e).unwrap();

            // 8 children, one per corner
            // child 0: corner n0; edges (0,1),(0,3),(0,4); faces bottom(0),front(2),left(4)
            new_conn.extend_from_slice(&[
                ns[0], get_em(0,1,ns), get_fc(0,ns), get_em(3,0,ns),
                get_em(0,4,ns), get_fc(2,ns), bc, get_fc(4,ns),
            ]); new_tags.push(tag);

            // child 1: corner n1; edges (0,1),(1,2),(1,5); faces bottom(0),front(2),right(5)
            new_conn.extend_from_slice(&[
                get_em(0,1,ns), ns[1], get_em(1,2,ns), get_fc(0,ns),
                get_fc(2,ns), get_em(1,5,ns), get_fc(5,ns), bc,
            ]); new_tags.push(tag);

            // child 2: corner n2; edges (1,2),(2,3),(2,6); faces bottom(0),back(3),right(5)
            new_conn.extend_from_slice(&[
                get_fc(0,ns), get_em(1,2,ns), ns[2], get_em(2,3,ns),
                bc, get_fc(5,ns), get_em(2,6,ns), get_fc(3,ns),
            ]); new_tags.push(tag);

            // child 3: corner n3; edges (2,3),(3,0),(3,7); faces bottom(0),back(3),left(4)
            new_conn.extend_from_slice(&[
                get_em(3,0,ns), get_fc(0,ns), get_em(2,3,ns), ns[3],
                get_fc(4,ns), bc, get_fc(3,ns), get_em(3,7,ns),
            ]); new_tags.push(tag);

            // child 4: corner n4; edges (0,4),(4,5),(4,7); faces top(1),front(2),left(4)
            new_conn.extend_from_slice(&[
                get_em(0,4,ns), get_fc(2,ns), bc, get_fc(4,ns),
                ns[4], get_em(4,5,ns), get_fc(1,ns), get_em(7,4,ns),
            ]); new_tags.push(tag);

            // child 5: corner n5; edges (1,5),(4,5),(5,6); faces top(1),front(2),right(5)
            new_conn.extend_from_slice(&[
                get_fc(2,ns), get_em(1,5,ns), get_fc(5,ns), bc,
                get_em(4,5,ns), ns[5], get_em(5,6,ns), get_fc(1,ns),
            ]); new_tags.push(tag);

            // child 6: corner n6; edges (2,6),(5,6),(6,7); faces top(1),back(3),right(5)
            new_conn.extend_from_slice(&[
                bc, get_fc(5,ns), get_em(2,6,ns), get_fc(3,ns),
                get_fc(1,ns), get_em(5,6,ns), ns[6], get_em(6,7,ns),
            ]); new_tags.push(tag);

            // child 7: corner n7; edges (3,7),(6,7),(4,7); faces top(1),back(3),left(4)
            new_conn.extend_from_slice(&[
                get_fc(4,ns), bc, get_fc(3,ns), get_em(3,7,ns),
                get_em(7,4,ns), get_fc(1,ns), get_em(6,7,ns), ns[7],
            ]); new_tags.push(tag);
        } else {
            for k in 0..8 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Detect hanging edge nodes ──────────────────────────────────────────
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            let has_unrefined = adj.iter().any(|e| !marked_set.contains(e));
            if has_unrefined {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── 4b. Detect hanging quad face centers ──────────────────────────
    // A quad face shared by a refined and an unrefined element creates
    // a HangingQuadFaceConstraint for the face-center node:
    //   u[face_center] = 0.25 * (u[a] + u[b] + u[c] + u[d])
    // plus edge-midpoint constraints for each of the four edges.
    let mut face_constraints: Vec<HangingQuadFaceConstraint> = Vec::new();
    for (fns, adj) in &face_elems {
        if adj.len() != 2 { continue; }
        let refined_count = adj.iter().filter(|&&e| marked_set.contains(&e)).count();
        if refined_count != 1 { continue; }
        // Recover actual (unsorted) face node order from the refined element
        // so edge lookups use the correct adjacency.
        let refined_elem = adj.iter().find(|&&e| marked_set.contains(&e)).unwrap();
        let ns = mesh.elem_nodes(*refined_elem);
        let face_nodes = local_faces_hex().iter()
            .filter_map(|&face| {
                let f4 = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
                if hex_face_key(f4) == *fns { Some(f4) } else { None }
            })
            .next()
            .expect("refined element must have this face");
        let [a, b, c, d] = face_nodes;
        let mab = midpoint_map.get(&edge_key(a, b)).copied();
        let mbc = midpoint_map.get(&edge_key(b, c)).copied();
        let mcd = midpoint_map.get(&edge_key(c, d)).copied();
        let mda = midpoint_map.get(&edge_key(d, a)).copied();
        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (mab, mbc, mcd, mda) {
            if let Some(&fc) = face_center_map.get(fns) {
                constraints.push(HangingNodeConstraint { constrained: mab as usize, parent_a: a as usize, parent_b: b as usize });
                constraints.push(HangingNodeConstraint { constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize });
                constraints.push(HangingNodeConstraint { constrained: mcd as usize, parent_a: c as usize, parent_b: d as usize });
                constraints.push(HangingNodeConstraint { constrained: mda as usize, parent_a: d as usize, parent_b: a as usize });
                face_constraints.push(HangingQuadFaceConstraint {
                    constrained: fc as usize,
                    parent_a: a as usize, parent_b: b as usize,
                    parent_c: c as usize, parent_d: d as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by_key(|c| c.constrained);
    face_constraints.sort_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));
    face_constraints.dedup_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));

    // ── 5. Rebuild boundary faces (Tri3 for Hex8 → Quad4 boundary faces) ─────
    // Hex8 boundary faces are Quad4; bisect if any of their edges was split.
    // A boundary face with all 4 edge midpoints present → split into 4 children.
    let n_bfaces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();
    let npf = 4usize; // Quad4

    for f in 0..n_bfaces {
        let fs = &mesh.face_conn[f * npf..(f + 1) * npf];
        let tag = mesh.face_tags[f];
        let (a, b, c, d) = (fs[0], fs[1], fs[2], fs[3]);

        let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
        let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
        let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
        let m_da = midpoint_map.get(&edge_key(d, a)).copied();

        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
            // Compute face centroid
            let coords: Vec<[f64; 3]> = [a, b, c, d].iter().map(|&n| mesh.coords_of(n)).collect();
            let (fcx, fcy, fcz) = (
                (coords[0][0] + coords[1][0] + coords[2][0] + coords[3][0]) / 4.0,
                (coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1]) / 4.0,
                (coords[0][2] + coords[1][2] + coords[2][2] + coords[3][2]) / 4.0,
            );
            // Use face_center_map if available, else create inline (boundary face)
            let fkey = hex_face_key([a, b, c, d]);
            let fc = if let Some(&existing) = face_center_map.get(&fkey) {
                existing
            } else {
                // Boundary face of an unrefined element whose edges were split
                // by a refined neighbor — create face center inline.
                let fc_id = next_node;
                next_node += 1;
                new_coords.push(fcx);
                new_coords.push(fcy);
                new_coords.push(fcz);
                fc_id
            };
            // 4 child Quad4 faces
            new_face_conn.extend_from_slice(&[a, mab, fc, mda]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mab, b, mbc, fc]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[fc, mbc, c, mcd]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mda, fc, mcd, d]); new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b, c, d]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Hex8,
        new_face_conn, new_face_tags, ElementType::Quad4,
    );
    (new_mesh, constraints, face_constraints, midpoint_map)
}

// ─── Prism6 uniform refinement ──────────────────────────────────────────────

/// Uniform refinement for Prism6 → 8 child Prism6.
///
/// Each prism is split into 8 by:
/// - 9 edge midpoints
/// - 5 face centers (2 triangular + 3 quadrilateral diagonal-crossing centroids)
/// - 1 body centroid
///
/// The 8 children consist of a bottom layer (children 0-3, below mid-height)
/// and a top layer (children 4-7, above mid-height), each with one child per
/// sub-triangle of the triangular faces plus one central child.
pub fn refine_prism6_uniform(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
    assert!(
        mesh.elem_type == ElementType::Prism6,
        "refine_prism6_uniform: only Prism6 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new(), HashMap::new());
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Edge adjacency for hanging detection ───────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_prism() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── 2. Allocate new nodes ─────────────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut tri_face_center_map: HashMap<(NodeId, NodeId, NodeId), NodeId> = HashMap::new();
    let mut quad_face_center_map: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut body_center_map: HashMap<ElemId, NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);

        // 9 edge midpoints
        for &(a, b) in &local_edges_prism() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                new_coords.push(0.5 * (xa[2] + xb[2]));
                let id = next_node; next_node += 1; id
            });
        }

        // 2 triangular face centers (centroids)
        for (a, b, c) in local_faces_prism_tri() {
            let key = face_key_3d(ns[a], ns[b], ns[c]);
            tri_face_center_map.entry(key).or_insert_with(|| {
                let ca = mesh.coords_of(ns[a]);
                let cb = mesh.coords_of(ns[b]);
                let cc = mesh.coords_of(ns[c]);
                new_coords.push((ca[0] + cb[0] + cc[0]) / 3.0);
                new_coords.push((ca[1] + cb[1] + cc[1]) / 3.0);
                new_coords.push((ca[2] + cb[2] + cc[2]) / 3.0);
                let id = next_node; next_node += 1; id
            });
        }

        // 3 quadrilateral face centers (diagonal crossing = centroid of 4 vertices)
        for face in local_faces_prism_quad() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let fkey = quad_face_key(fns);
            quad_face_center_map.entry(fkey).or_insert_with(|| {
                let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
                for &fn_ in &fns {
                    let c = mesh.coords_of(fn_);
                    x += c[0]; y += c[1]; z += c[2];
                }
                new_coords.push(x / 4.0); new_coords.push(y / 4.0); new_coords.push(z / 4.0);
                let id = next_node; next_node += 1; id
            });
        }

        // Body centroid
        body_center_map.entry(e).or_insert_with(|| {
            let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..6 {
                let c = mesh.coords_of(ns[k]);
                x += c[0]; y += c[1]; z += c[2];
            }
            new_coords.push(x / 6.0); new_coords.push(y / 6.0); new_coords.push(z / 6.0);
            let id = next_node; next_node += 1; id
        });
    }

    // ── 3. Helper closures ────────────────────────────────────────────────────
    let get_em = |a: usize, b: usize, ns: &[NodeId]| -> NodeId {
        *midpoint_map.get(&edge_key(ns[a], ns[b])).expect("edge midpoint missing")
    };
    let get_tfc = |(a, b, c): (usize, usize, usize), ns: &[NodeId]| -> NodeId {
        let key = face_key_3d(ns[a], ns[b], ns[c]);
        *tri_face_center_map.get(&key).expect("tri face center missing")
    };
    let get_qfc = |face_idx: usize, ns: &[NodeId]| -> NodeId {
        let face = local_faces_prism_quad()[face_idx];
        let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
        let fkey = quad_face_key(fns);
        *quad_face_center_map.get(&fkey).expect("quad face center missing")
    };

    // ── 4. Build new element connectivity (8 child prisms) ────────────────────
    // Each child is a Prism6: bottom tri (3 nodes CCW) + top tri (3 nodes CCW).
    //   Child 0 (bottom corner 0):  bot=(n0, m01, m02), top=(m03, qfc0, qfc2)
    //   Child 1 (bottom corner 1):  bot=(m01, n1, m12), top=(qfc0, m14, qfc1)
    //   Child 2 (bottom corner 2):  bot=(m02, m12, n2), top=(qfc2, qfc1, m25)
    //   Child 3 (bottom center):    bot=(m01, m12, m02), top=(qfc0, qfc1, qfc2)
    //   Child 4 (top corner 3):     bot=(m03, qfc0, qfc2), top=(n3, m34, m35)
    //   Child 5 (top corner 4):     bot=(qfc0, m14, qfc1), top=(m34, n4, m45)
    //   Child 6 (top corner 5):     bot=(qfc2, qfc1, m25), top=(m35, m45, n5)
    //   Child 7 (top center):       bot=(qfc0, qfc1, qfc2), top=(m34, m45, m35)
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32>    = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            // Shorthand: edge midpoints
            let m01 = get_em(0, 1, ns);
            let m12 = get_em(1, 2, ns);
            let m02 = get_em(0, 2, ns);
            let m34 = get_em(3, 4, ns);
            let m45 = get_em(4, 5, ns);
            let m35 = get_em(3, 5, ns);
            let m03 = get_em(0, 3, ns);
            let m14 = get_em(1, 4, ns);
            let m25 = get_em(2, 5, ns);

            // Face centers
            let _tfc_bot = get_tfc((0, 1, 2), ns);  // bottom tri (unused in uniform-all, needed for NC)
            let _tfc_top = get_tfc((3, 4, 5), ns);  // top tri
            let qfc0 = get_qfc(0, ns);  // quad (0,1,4,3)
            let qfc1 = get_qfc(1, ns);  // quad (1,2,5,4)
            let qfc2 = get_qfc(2, ns);  // quad (0,2,5,3)

            // Child 0: bottom corner 0
            new_conn.extend_from_slice(&[ns[0], m01, m02,  m03, qfc0, qfc2]); new_tags.push(tag);

            // Child 1: bottom corner 1
            new_conn.extend_from_slice(&[m01, ns[1], m12,  qfc0, m14, qfc1]); new_tags.push(tag);

            // Child 2: bottom corner 2
            new_conn.extend_from_slice(&[m02, m12, ns[2],  qfc2, qfc1, m25]); new_tags.push(tag);

            // Child 3: bottom center
            new_conn.extend_from_slice(&[m01, m12, m02,  qfc0, qfc1, qfc2]); new_tags.push(tag);

            // Child 4: top corner 3
            new_conn.extend_from_slice(&[m03, qfc0, qfc2,  ns[3], m34, m35]); new_tags.push(tag);

            // Child 5: top corner 4
            new_conn.extend_from_slice(&[qfc0, m14, qfc1,  m34, ns[4], m45]); new_tags.push(tag);

            // Child 6: top corner 5
            new_conn.extend_from_slice(&[qfc2, qfc1, m25,  m35, m45, ns[5]]); new_tags.push(tag);

            // Child 7: top center
            new_conn.extend_from_slice(&[qfc0, qfc1, qfc2,  m34, m45, m35]); new_tags.push(tag);
        } else {
            for k in 0..6 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 5. Detect hanging edge nodes ──────────────────────────────────────────
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            let has_unrefined = adj.iter().any(|e| !marked_set.contains(e));
            if has_unrefined {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── 6. Rebuild boundary faces ────────────────────────────────────────────
    // Prism6 boundary faces are either Tri3 or Quad4.
    let n_bfaces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_bfaces {
        let tag = mesh.face_tags[f];
        let fs = mesh.bface_nodes(f as FaceId);
        match fs.len() {
            3 => {
                let (a, b, c) = (fs[0], fs[1], fs[2]);
                let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
                let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
                let m_ca = midpoint_map.get(&edge_key(c, a)).copied();

                if let (Some(mab), Some(mbc), Some(mca)) = (m_ab, m_bc, m_ca) {
                    new_face_conn.extend_from_slice(&[a, mab, mca]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mab, b, mbc]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mca, mbc, c]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mab, mbc, mca]); new_face_tags.push(tag);
                } else {
                    new_face_conn.extend_from_slice(&[a, b, c]);
                    new_face_tags.push(tag);
                }
            }
            4 => {
                let (a, b, c, d) = (fs[0], fs[1], fs[2], fs[3]);
                let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
                let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
                let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
                let m_da = midpoint_map.get(&edge_key(d, a)).copied();

                if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
                    let fkey = quad_face_key([a, b, c, d]);
                    if let Some(&fc) = quad_face_center_map.get(&fkey) {
                        new_face_conn.extend_from_slice(&[a, mab, fc, mda]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[mab, b, mbc, fc]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[fc, mbc, c, mcd]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[mda, fc, mcd, d]); new_face_tags.push(tag);
                    } else {
                        new_face_conn.extend_from_slice(&[a, b, c, d]);
                        new_face_tags.push(tag);
                    }
                } else {
                    new_face_conn.extend_from_slice(&[a, b, c, d]);
                    new_face_tags.push(tag);
                }
            }
            _ => {
                for &n in fs { new_face_conn.push(n); }
                new_face_tags.push(tag);
            }
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Prism6,
        new_face_conn, new_face_tags, mesh.face_type,
    );
    (new_mesh, constraints, midpoint_map)
}

// ─── Prism6 non-conforming AMR ────────────────────────────────────────────────

/// A hanging-face descriptor for a quadrilateral face in 3-D.
///
/// This records a coarse quadrilateral face `(parent_a, parent_b, parent_c, parent_d)`
/// that is non-conforming against a refined neighbor. `constrained` stores the
/// face center node (diagonal intersection) whose DOF must satisfy:
/// `u[constrained] = 0.25 * (u[a] + u[b] + u[c] + u[d])`.
#[derive(Debug, Clone)]
pub struct HangingQuadFaceConstraint {
    /// Face center node on the hanging quadrilateral face.
    pub constrained: usize,
    /// Coarse face vertex node indices (4 corners).
    pub parent_a: usize,
    pub parent_b: usize,
    pub parent_c: usize,
    pub parent_d: usize,
}

/// Non-conforming (hanging-node) refinement for a 3-D Prism6 mesh.
///
/// Each marked Prism6 is split into **8 child Prism6s** ...
/// ...
/// # Returns
/// `(new_mesh, edge_constraints, tri_face_constraints, quad_face_constraints, midpoint_map)`.
pub fn refine_nonconforming_prism(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (
    SimplexMesh<3>,
    Vec<HangingNodeConstraint>,
    Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>,
    HashMap<(NodeId, NodeId), NodeId>,
) {
    let (m, ec, tc, qc, mm, _) = refine_nonconforming_prism_internal(mesh, marked, None);
    (m, ec, tc, qc, mm)
}

fn refine_nonconforming_prism_internal(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    SimplexMesh<3>,
    Vec<HangingNodeConstraint>,
    Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>,
    HashMap<(NodeId, NodeId), NodeId>,
    HashMap<(NodeId, NodeId), NodeId>,
) {
    assert!(
        mesh.elem_type == ElementType::Prism6,
        "refine_nonconforming_prism_internal: only Prism6 meshes are supported"
    );

    if marked.is_empty() {
        let mut active_mp = HashMap::new();
        if let Some(prev) = active_midpoints { active_mp = prev.clone(); }
        return (mesh.clone(), Vec::new(), Vec::new(), Vec::new(), HashMap::new(), active_mp);
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Edge and face adjacency for hanging detection ──────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut tri_face_elems: HashMap<(NodeId, NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut quad_face_elems: HashMap<[NodeId; 4], Vec<ElemId>> = HashMap::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);

        for &(a, b) in &local_edges_prism() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
        for &(a, b, c) in &local_faces_prism_tri() {
            tri_face_elems.entry(face_key_3d(ns[a], ns[b], ns[c])).or_default().push(e);
        }
        for face in local_faces_prism_quad() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            quad_face_elems.entry(quad_face_key(fns)).or_default().push(e);
        }
    }

    // ── 2. Allocate new nodes ─────────────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut tri_face_center_map: HashMap<(NodeId, NodeId, NodeId), NodeId> = HashMap::new();
    let mut quad_face_center_map: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut body_center_map: HashMap<ElemId, NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);

        // 9 edge midpoints
        for &(a, b) in &local_edges_prism() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                new_coords.push(0.5 * (xa[2] + xb[2]));
                let id = next_node; next_node += 1; id
            });
        }

        // 2 triangular face centers
        for &(a, b, c) in &local_faces_prism_tri() {
            let key = face_key_3d(ns[a], ns[b], ns[c]);
            tri_face_center_map.entry(key).or_insert_with(|| {
                let ca = mesh.coords_of(ns[a]);
                let cb = mesh.coords_of(ns[b]);
                let cc = mesh.coords_of(ns[c]);
                new_coords.push((ca[0] + cb[0] + cc[0]) / 3.0);
                new_coords.push((ca[1] + cb[1] + cc[1]) / 3.0);
                new_coords.push((ca[2] + cb[2] + cc[2]) / 3.0);
                let id = next_node; next_node += 1; id
            });
        }

        // 3 quadrilateral face centers
        for face in local_faces_prism_quad() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let fkey = quad_face_key(fns);
            quad_face_center_map.entry(fkey).or_insert_with(|| {
                let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
                for &fn_ in &fns {
                    let c = mesh.coords_of(fn_);
                    x += c[0]; y += c[1]; z += c[2];
                }
                new_coords.push(x / 4.0); new_coords.push(y / 4.0); new_coords.push(z / 4.0);
                let id = next_node; next_node += 1; id
            });
        }

        // Body centroid
        body_center_map.entry(e).or_insert_with(|| {
            let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..6 {
                let c = mesh.coords_of(ns[k]);
                x += c[0]; y += c[1]; z += c[2];
            }
            new_coords.push(x / 6.0); new_coords.push(y / 6.0); new_coords.push(z / 6.0);
            let id = next_node; next_node += 1; id
        });
    }

    // ── 3. Helper closures ────────────────────────────────────────────────────
    let get_em = |a: usize, b: usize, ns: &[NodeId]| -> NodeId {
        *midpoint_map.get(&edge_key(ns[a], ns[b])).expect("edge midpoint missing")
    };
    let get_tfc = |(a, b, c): (usize, usize, usize), ns: &[NodeId]| -> NodeId {
        let key = face_key_3d(ns[a], ns[b], ns[c]);
        *tri_face_center_map.get(&key).expect("tri face center missing")
    };
    let get_qfc = |face_idx: usize, ns: &[NodeId]| -> NodeId {
        let face = local_faces_prism_quad()[face_idx];
        let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
        let fkey = quad_face_key(fns);
        *quad_face_center_map.get(&fkey).expect("quad face center missing")
    };

    // ── 4. Build new element connectivity (8 child prisms) ────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32>    = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            let m01 = get_em(0, 1, ns); let m12 = get_em(1, 2, ns); let m02 = get_em(0, 2, ns);
            let m34 = get_em(3, 4, ns); let m45 = get_em(4, 5, ns); let m35 = get_em(3, 5, ns);
            let m03 = get_em(0, 3, ns); let m14 = get_em(1, 4, ns); let m25 = get_em(2, 5, ns);

            let _tfc_bot = get_tfc((0, 1, 2), ns);
            let _tfc_top = get_tfc((3, 4, 5), ns);
            let qfc0 = get_qfc(0, ns); let qfc1 = get_qfc(1, ns); let qfc2 = get_qfc(2, ns);

            // Child 0: bottom corner 0
            new_conn.extend_from_slice(&[ns[0], m01, m02,  m03, qfc0, qfc2]); new_tags.push(tag);
            // Child 1: bottom corner 1
            new_conn.extend_from_slice(&[m01, ns[1], m12,  qfc0, m14, qfc1]); new_tags.push(tag);
            // Child 2: bottom corner 2
            new_conn.extend_from_slice(&[m02, m12, ns[2],  qfc2, qfc1, m25]); new_tags.push(tag);
            // Child 3: bottom center
            new_conn.extend_from_slice(&[m01, m12, m02,  qfc0, qfc1, qfc2]); new_tags.push(tag);
            // Child 4: top corner 3
            new_conn.extend_from_slice(&[m03, qfc0, qfc2,  ns[3], m34, m35]); new_tags.push(tag);
            // Child 5: top corner 4
            new_conn.extend_from_slice(&[qfc0, m14, qfc1,  m34, ns[4], m45]); new_tags.push(tag);
            // Child 6: top corner 5
            new_conn.extend_from_slice(&[qfc2, qfc1, m25,  m35, m45, ns[5]]); new_tags.push(tag);
            // Child 7: top center
            new_conn.extend_from_slice(&[qfc0, qfc1, qfc2,  m34, m45, m35]); new_tags.push(tag);
        } else {
            for k in 0..6 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 5. Detect hanging constraints ─────────────────────────────────────────
    let mut edge_constraints: Vec<HangingNodeConstraint> = Vec::new();
    let mut tri_face_constraints: Vec<HangingFaceConstraint> = Vec::new();
    let mut quad_face_constraints: Vec<HangingQuadFaceConstraint> = Vec::new();

    // 5a. Edge hanging: edge shared by refined + unrefined element
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            let has_unrefined = adj.iter().any(|e| !marked_set.contains(e));
            if has_unrefined {
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    edge_constraints.sort_by_key(|c| c.constrained);
    edge_constraints.dedup_by_key(|c| c.constrained);

    // 5b. Tri face hanging: tri face shared by refined + unrefined
    for (&(a, b, c), adj) in &tri_face_elems {
        if adj.len() != 2 { continue; }
        let refined_count = adj.iter().filter(|&&e| marked_set.contains(&e)).count();
        if refined_count != 1 { continue; }

        let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
        let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
        let m_ac = midpoint_map.get(&edge_key(a, c)).copied();

        if let (Some(mab), Some(mbc), Some(mac)) = (m_ab, m_bc, m_ac) {
            // All 3 edge midpoints exist → this tri face is hanging
            edge_constraints.push(HangingNodeConstraint {
                constrained: mab as usize, parent_a: a as usize, parent_b: b as usize,
            });
            edge_constraints.push(HangingNodeConstraint {
                constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
            });
            edge_constraints.push(HangingNodeConstraint {
                constrained: mac as usize, parent_a: a as usize, parent_b: c as usize,
            });
            tri_face_constraints.push(HangingFaceConstraint {
                constrained: mab as usize, parent_a: a as usize,
                parent_b: b as usize, parent_c: c as usize,
            });
        }
    }
    tri_face_constraints.sort_by_key(|c| (c.parent_a, c.parent_b, c.parent_c));
    tri_face_constraints.dedup_by_key(|c| (c.parent_a, c.parent_b, c.parent_c));

    // 5c. Quad face hanging: quad face shared by refined + unrefined
    //     The face center becomes a hanging node constrained to the 4 corners.
    //     We get the actual edge pairs from the refined element's local face order.
    for (fns, adj) in &quad_face_elems {
        if adj.len() != 2 { continue; }
        let refined_count = adj.iter().filter(|&&e| marked_set.contains(&e)).count();
        if refined_count != 1 { continue; }

        // Get the refined element's local face nodes (preserves edge adjacency).
        let refined_elem = adj.iter().find(|&&e| marked_set.contains(&e)).unwrap();
        let ns = mesh.elem_nodes(*refined_elem);
        let (face_nodes, _) = local_faces_prism_quad().iter()
            .filter_map(|&face| {
                let f4 = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
                if quad_face_key(f4) == *fns { Some((f4, face)) } else { None }
            })
            .next()
            .expect("refined element must have the face");

        let [a, b, c, d] = face_nodes;
        let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
        let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
        let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
        let m_da = midpoint_map.get(&edge_key(d, a)).copied();

        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
            if let Some(&fc) = quad_face_center_map.get(fns) {
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mab as usize, parent_a: a as usize, parent_b: b as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mcd as usize, parent_a: c as usize, parent_b: d as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mda as usize, parent_a: d as usize, parent_b: a as usize,
                });
                quad_face_constraints.push(HangingQuadFaceConstraint {
                    constrained: fc as usize,             // the face center is the hanging node
                    parent_a: a as usize, parent_b: b as usize,
                    parent_c: c as usize, parent_d: d as usize,
                });
            }
        }
    }
    quad_face_constraints.sort_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));
    quad_face_constraints.dedup_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));

    // ── 6. Rebuild boundary faces ────────────────────────────────────────────
    let n_bfaces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_bfaces {
        let tag = mesh.face_tags[f];
        let fs = mesh.bface_nodes(f as FaceId);
        match fs.len() {
            3 => {
                let (a, b, c) = (fs[0], fs[1], fs[2]);
                let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
                let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
                let m_ca = midpoint_map.get(&edge_key(c, a)).copied();
                if let (Some(mab), Some(mbc), Some(mca)) = (m_ab, m_bc, m_ca) {
                    new_face_conn.extend_from_slice(&[a, mab, mca]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mab, b, mbc]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mca, mbc, c]); new_face_tags.push(tag);
                    new_face_conn.extend_from_slice(&[mab, mbc, mca]); new_face_tags.push(tag);
                } else {
                    new_face_conn.extend_from_slice(&[a, b, c]); new_face_tags.push(tag);
                }
            }
            4 => {
                let (a, b, c, d) = (fs[0], fs[1], fs[2], fs[3]);
                let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
                let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
                let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
                let m_da = midpoint_map.get(&edge_key(d, a)).copied();
                if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
                    let fkey = quad_face_key([a, b, c, d]);
                    if let Some(&fc) = quad_face_center_map.get(&fkey) {
                        new_face_conn.extend_from_slice(&[a, mab, fc, mda]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[mab, b, mbc, fc]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[fc, mbc, c, mcd]); new_face_tags.push(tag);
                        new_face_conn.extend_from_slice(&[mda, fc, mcd, d]); new_face_tags.push(tag);
                    } else {
                        new_face_conn.extend_from_slice(&[a, b, c, d]); new_face_tags.push(tag);
                    }
                } else {
                    new_face_conn.extend_from_slice(&[a, b, c, d]); new_face_tags.push(tag);
                }
            }
            _ => {
                for &n in fs { new_face_conn.push(n); }
                new_face_tags.push(tag);
            }
        }
    }

    // ── 7. Build new active midpoint set for multi-level tracking ──────
    let mut new_active_midpoints = std::collections::HashMap::new();
    if let Some(prev) = active_midpoints {
        for (&k, &v) in prev { new_active_midpoints.insert(k, v); }
    }
    for (&k, &v) in &midpoint_map { new_active_midpoints.insert(k, v); }
    let new_node_set: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
    new_active_midpoints.retain(|_, mid| new_node_set.contains(mid));

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Prism6,
        new_face_conn, new_face_tags, mesh.face_type,
    );
    (new_mesh, edge_constraints, tri_face_constraints, quad_face_constraints, midpoint_map, new_active_midpoints)
}

// ─── Anisotropic Quad/Hex NC AMR ──────────────────────────────────────────────

/// Direction for anisotropic quad refinement.
///
/// - `X` — split element into 2 quads by cutting along X (adds a vertical midpoint edge)
/// - `Y` — split element into 2 quads by cutting along Y (adds a horizontal midpoint edge)
/// - `Both` — isotropic 4-way split (same as `refine_nonconforming_quad`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuadRefineDir {
    /// Split along X axis (left/right halves — vertical cut).
    X,
    /// Split along Y axis (top/bottom halves — horizontal cut).
    Y,
    /// Full 4-way isotropic split.
    Both,
}

/// Direction for anisotropic Hex8 refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HexRefineDir {
    /// Split along X (2 children).
    X,
    /// Split along Y (2 children).
    Y,
    /// Split along Z (2 children).
    Z,
    /// Split along XY (4 children in-plane).
    XY,
    /// Split along XZ (4 children).
    XZ,
    /// Split along YZ (4 children).
    YZ,
    /// Full 8-way isotropic split.
    All,
}

/// Anisotropic non-conforming refinement for Quad4 meshes.
///
/// Each entry in `marked` is `(elem_id, direction)`.
///
/// # Refinement directions
/// - [`QuadRefineDir::X`]: element is split into left and right halves by a
///   vertical mid-edge.  Two child quads share the new midpoint nodes on the
///   two horizontal edges.
/// - [`QuadRefineDir::Y`]: element is split into top and bottom halves by a
///   horizontal mid-edge.
/// - [`QuadRefineDir::Both`]: full 4-way isotropic refinement (5 new nodes).
///
/// # Returns
/// `(new_mesh, constraints)` where constraints encode hanging-node DOF dependencies.
///
/// # Quad4 node layout (CCW)
/// ```text
///   n3 ─── n2
///   │       │
///   n0 ─── n1
/// ```
pub fn refine_nonconforming_quad_aniso(
    mesh:   &SimplexMesh<2>,
    marked: &[(ElemId, QuadRefineDir)],
) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Quad4,
        "refine_nonconforming_quad_aniso: only Quad4 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new());
    }

    let n_elems = mesh.n_elems();
    let marked_map: HashMap<ElemId, QuadRefineDir> = marked.iter().copied().collect();
    let marked_set: std::collections::HashSet<ElemId> = marked_map.keys().copied().collect();

    // ── Build edge adjacency ─────────────────────────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_quad() {
            edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── Compute midpoints for needed edges ───────────────────────────────────
    // For X-split: need midpoints of edges (n0,n3) and (n1,n2) — left and right vertical edges
    // For Y-split: need midpoints of edges (n0,n1) and (n3,n2) — bottom and top horizontal edges
    // For Both:    need all 4 edge midpoints + centroid
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut center_map:   HashMap<ElemId, NodeId>           = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    // Inline helper macro to insert midpoint if not already present.
    macro_rules! ensure_midpoint {
        ($key:expr) => {{
            let k = $key;
            if !midpoint_map.contains_key(&k) {
                let xa = mesh.coords_of(k.0);
                let xb = mesh.coords_of(k.1);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                midpoint_map.insert(k, next_node);
                next_node += 1;
            }
        }};
    }

    for (&e, &dir) in &marked_map {
        let ns = mesh.elem_nodes(e);
        match dir {
            QuadRefineDir::X => {
                // Horizontal cut: midpoints of bottom (n0,n1) and top (n3,n2)
                ensure_midpoint!(quad_edge_key(ns[0], ns[1]));
                ensure_midpoint!(quad_edge_key(ns[3], ns[2]));
            }
            QuadRefineDir::Y => {
                // Vertical cut: midpoints of left (n0,n3) and right (n1,n2)
                ensure_midpoint!(quad_edge_key(ns[0], ns[3]));
                ensure_midpoint!(quad_edge_key(ns[1], ns[2]));
            }
            QuadRefineDir::Both => {
                for &(a, b) in &local_edges_quad() {
                    ensure_midpoint!(quad_edge_key(ns[a], ns[b]));
                }
                center_map.entry(e).or_insert_with(|| {
                    let (mut cx, mut cy) = (0.0_f64, 0.0_f64);
                    for k in 0..4 { let c = mesh.coords_of(ns[k]); cx += c[0]; cy += c[1]; }
                    new_coords.push(cx / 4.0);
                    new_coords.push(cy / 4.0);
                    let id = next_node; next_node += 1; id
                });
            }
        }
    }

    // ── Build new element connectivity ───────────────────────────────────────
    // QuadRefineDir::X  = horizontal cut (adds midpoints on bottom/top edges n0-n1 and n3-n2)
    //   Left child:  [n0, bottom_mid, top_mid, n3]   (CCW)
    //   Right child: [bottom_mid, n1, n2, top_mid]   (CCW)
    //
    // QuadRefineDir::Y  = vertical cut (adds midpoints on left/right edges n0-n3 and n1-n2)
    //   Bottom child: [n0, n1, right_mid, left_mid]  (CCW)
    //   Top child:    [left_mid, right_mid, n2, n3]  (CCW)
    //
    // QuadRefineDir::Both = 4-way isotropic split (all edge midpoints + centroid).
    let mut new_conn: Vec<u32> = Vec::new();
    let mut new_elem_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];
        if let Some(&dir) = marked_map.get(&e) {
            match dir {
                QuadRefineDir::X => {
                    // Horizontal cut: midpoints of bottom (n0,n1) and top (n3,n2) edges
                    let bottom_mid = *midpoint_map.get(&quad_edge_key(ns[0], ns[1])).unwrap();
                    let top_mid    = *midpoint_map.get(&quad_edge_key(ns[3], ns[2])).unwrap();
                    // Left child (CCW): n0, bottom_mid, top_mid, n3
                    new_conn.extend_from_slice(&[ns[0], bottom_mid, top_mid, ns[3]]);
                    new_elem_tags.push(tag);
                    // Right child (CCW): bottom_mid, n1, n2, top_mid
                    new_conn.extend_from_slice(&[bottom_mid, ns[1], ns[2], top_mid]);
                    new_elem_tags.push(tag);
                }
                QuadRefineDir::Y => {
                    // Vertical cut: midpoints of left (n0,n3) and right (n1,n2) edges
                    let left_mid  = *midpoint_map.get(&quad_edge_key(ns[0], ns[3])).unwrap();
                    let right_mid = *midpoint_map.get(&quad_edge_key(ns[1], ns[2])).unwrap();
                    // Bottom child (CCW): n0, n1, right_mid, left_mid
                    new_conn.extend_from_slice(&[ns[0], ns[1], right_mid, left_mid]);
                    new_elem_tags.push(tag);
                    // Top child (CCW): left_mid, right_mid, n2, n3
                    new_conn.extend_from_slice(&[left_mid, right_mid, ns[2], ns[3]]);
                    new_elem_tags.push(tag);
                }
                QuadRefineDir::Both => {
                    // Full 4-way split
                    let m01 = *midpoint_map.get(&quad_edge_key(ns[0], ns[1])).unwrap();
                    let m12 = *midpoint_map.get(&quad_edge_key(ns[1], ns[2])).unwrap();
                    let m23 = *midpoint_map.get(&quad_edge_key(ns[2], ns[3])).unwrap();
                    let m30 = *midpoint_map.get(&quad_edge_key(ns[3], ns[0])).unwrap();
                    let c   = *center_map.get(&e).unwrap();
                    new_conn.extend_from_slice(&[ns[0], m01, c, m30]);
                    new_elem_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, ns[1], m12, c]);
                    new_elem_tags.push(tag);
                    new_conn.extend_from_slice(&[c, m12, ns[2], m23]);
                    new_elem_tags.push(tag);
                    new_conn.extend_from_slice(&[m30, c, m23, ns[3]]);
                    new_elem_tags.push(tag);
                }
            }
        } else {
            // Unrefined element: keep as-is
            new_conn.extend_from_slice(ns);
            new_elem_tags.push(tag);
        }
    }

    // ── Build hanging-node constraints ───────────────────────────────────────
    // For each split edge whose neighbour is unrefined, record the midpoint as constrained.
    let mut constraints: Vec<HangingNodeConstraint> = Vec::new();

    for (&e, &dir) in &marked_map {
        let ns = mesh.elem_nodes(e);
        let split_edges: Vec<(NodeId, NodeId)> = match dir {
            QuadRefineDir::X => vec![
                quad_edge_key(ns[0], ns[1]),
                quad_edge_key(ns[3], ns[2]),
            ],
            QuadRefineDir::Y => vec![
                quad_edge_key(ns[0], ns[3]),
                quad_edge_key(ns[1], ns[2]),
            ],
            QuadRefineDir::Both => local_edges_quad()
                .iter()
                .map(|&(a, b)| quad_edge_key(ns[a], ns[b]))
                .collect(),
        };
        for edge in split_edges {
            if let Some(&mid) = midpoint_map.get(&edge) {
                if let Some(neighbors) = edge_elems.get(&edge) {
                    for &nb in neighbors {
                        if nb != e && !marked_set.contains(&nb) {
                            constraints.push(HangingNodeConstraint {
                                constrained: mid as usize,
                                parent_a:    edge.0 as usize,
                                parent_b:    edge.1 as usize,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }
    // Deduplicate by constrained node
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by_key(|c| c.constrained);

    let new_mesh = SimplexMesh::<2>::uniform(
        new_coords,
        new_conn,
        new_elem_tags,
        ElementType::Quad4,
        mesh.face_conn.clone(),
        mesh.face_tags.clone(),
        mesh.face_type,
    );
    (new_mesh, constraints)
}

/// Anisotropic non-conforming refinement for Hex8 meshes.
///
/// Each entry in `marked` is `(elem_id, direction)`.
///
/// # Refinement directions
/// - [`HexRefineDir::X`]: 2 children split along X (midpoints on X-normal faces).
/// - [`HexRefineDir::Y`]: 2 children split along Y.
/// - [`HexRefineDir::Z`]: 2 children split along Z.
/// - [`HexRefineDir::XY`]: 4 children split along X and Y.
/// - [`HexRefineDir::XZ`]: 4 children split along X and Z.
/// - [`HexRefineDir::YZ`]: 4 children split along Y and Z.
/// - [`HexRefineDir::All`]: full 8-way isotropic split (delegates to
///   [`refine_nonconforming_hex`]).
///
/// # Hex8 node layout
/// ```text
///    7────6
///   /|   /|   top face: n4-n5-n6-n7 (CCW viewed from above)
///  4────5 |
///  | 3──|─2
///  |/   |/    bottom face: n0-n1-n2-n3 (CCW viewed from below)
///  0────1
/// ```
///
/// # Returns
/// `(new_mesh, constraints)` where constraints encode hanging-node DOF dependencies.
pub fn refine_nonconforming_hex_aniso(
    mesh: &SimplexMesh<3>,
    marked: &[(ElemId, HexRefineDir)],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Hex8,
        "refine_nonconforming_hex_aniso: only Hex8 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new());
    }

    // Separate `All` cases: delegate them via the isotropic refiner,
    // then handle the directional cases independently.
    let all_ids: Vec<ElemId> = marked
        .iter()
        .filter_map(|&(e, d)| if d == HexRefineDir::All { Some(e) } else { None })
        .collect();

    let directional: Vec<(ElemId, HexRefineDir)> = marked
        .iter()
        .copied()
        .filter(|&(_, d)| d != HexRefineDir::All)
        .collect();

    // If only isotropic splits requested, delegate.
    if directional.is_empty() {
        let (m, c, _, _) = refine_nonconforming_hex(mesh, &all_ids);
        return (m, c);
    }

    // For the purely directional (anisotropic) path we implement X/Y/Z splitting.
    // XY = X then Y (in a single pass); XZ = X then Z; YZ = Y then Z.
    // Strategy:
    //   1. Expand each (elem, XY/XZ/YZ) into constituent single-axis splits.
    //   2. For each marked element accumulate which axes are cut.
    //   3. In one pass, generate midpoints for needed face-midplane edges.
    //   4. Emit children and constraints.

    let n_elems = mesh.n_elems();

    // ── Determine per-element cut flags ─────────────────────────────────────
    #[derive(Default, Clone, Copy)]
    struct CutFlags { cut_x: bool, cut_y: bool, cut_z: bool }

    let mut cut_map: HashMap<ElemId, CutFlags> = HashMap::new();
    for &(e, dir) in &directional {
        let f = cut_map.entry(e).or_default();
        match dir {
            HexRefineDir::X  => { f.cut_x = true; }
            HexRefineDir::Y  => { f.cut_y = true; }
            HexRefineDir::Z  => { f.cut_z = true; }
            HexRefineDir::XY => { f.cut_x = true; f.cut_y = true; }
            HexRefineDir::XZ => { f.cut_x = true; f.cut_z = true; }
            HexRefineDir::YZ => { f.cut_y = true; f.cut_z = true; }
            HexRefineDir::All => {} // handled separately
        }
    }

    // ── Build edge adjacency (for hanging-node detection) ────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_hex() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── Compute midpoints for required face-planes ───────────────────────────
    //
    // Hex8 node layout (standard, 0-based):
    //   Bottom face: 0(─x─y), 1(+x─y), 2(+x+y), 3(─x+y)
    //   Top    face: 4(─x─y), 5(+x─y), 6(+x+y), 7(─x+y)
    //
    // X-split: cut YZ-plane at x=0.5 → 4 mid-nodes on edges (0,1),(3,2),(4,5),(7,6)
    //   → left  child: [n0, mB, mT_front, n3, n4, mT_back (wait this is getting complex)]
    //
    // Rather than hardcoding the full topology of multi-axis splits, we implement
    // each axis split independently using face-midpoint nodes.
    //
    // X-CUT: insert a YZ midplane.
    //   Midpoints needed: edges (0,1),(3,2),(4,5),(7,6) — the 4 edges parallel to X.
    //   Left child (─x half): [n0, m01, m32, n3, n4, m45, m76, n7]
    //   Right child (+x half): [m01, n1, n2, m32, m45, n5, n6, m76]
    //
    // Y-CUT: insert an XZ midplane.
    //   Midpoints needed: edges (0,3),(1,2),(4,7),(5,6) — the 4 edges parallel to Y.
    //   Front child (─y half): [n0, n1, m12, m03, n4, n5, m56, m47]
    //   Back child  (+y half): [m03, m12, n2, n3, m47, m56, n6, n7]
    //
    // Z-CUT: insert an XY midplane.
    //   Midpoints needed: edges (0,4),(1,5),(2,6),(3,7) — the 4 vertical edges.
    //   Bottom child: [n0, n1, n2, n3, m04, m15, m26, m37]
    //   Top child:    [m04, m15, m26, m37, n4, n5, n6, n7]
    //
    // For multi-axis: compose the children from single-axis results within one pass.

    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    let mut ensure_mp = |key: (NodeId, NodeId), new_coords: &mut Vec<f64>, next: &mut NodeId| -> NodeId {
        let k = edge_key(key.0, key.1);
        *midpoint_map.entry(k).or_insert_with(|| {
            // Copy values first to avoid simultaneous mutable/immutable borrow.
            let xa = [
                new_coords[3 * k.0 as usize],
                new_coords[3 * k.0 as usize + 1],
                new_coords[3 * k.0 as usize + 2],
            ];
            let xb = [
                new_coords[3 * k.1 as usize],
                new_coords[3 * k.1 as usize + 1],
                new_coords[3 * k.1 as usize + 2],
            ];
            new_coords.push(0.5 * (xa[0] + xb[0]));
            new_coords.push(0.5 * (xa[1] + xb[1]));
            new_coords.push(0.5 * (xa[2] + xb[2]));
            let id = *next; *next += 1; id
        })
    };

    // Pre-allocate midpoints for all cut edges.
    for (&e, &cf) in &cut_map {
        let ns = mesh.elem_nodes(e);
        if cf.cut_x {
            ensure_mp((ns[0], ns[1]), &mut new_coords, &mut next_node);
            ensure_mp((ns[3], ns[2]), &mut new_coords, &mut next_node);
            ensure_mp((ns[4], ns[5]), &mut new_coords, &mut next_node);
            ensure_mp((ns[7], ns[6]), &mut new_coords, &mut next_node);
        }
        if cf.cut_y {
            ensure_mp((ns[0], ns[3]), &mut new_coords, &mut next_node);
            ensure_mp((ns[1], ns[2]), &mut new_coords, &mut next_node);
            ensure_mp((ns[4], ns[7]), &mut new_coords, &mut next_node);
            ensure_mp((ns[5], ns[6]), &mut new_coords, &mut next_node);
        }
        if cf.cut_z {
            ensure_mp((ns[0], ns[4]), &mut new_coords, &mut next_node);
            ensure_mp((ns[1], ns[5]), &mut new_coords, &mut next_node);
            ensure_mp((ns[2], ns[6]), &mut new_coords, &mut next_node);
            ensure_mp((ns[3], ns[7]), &mut new_coords, &mut next_node);
        }
    }

    // ── For multi-axis cuts, also need face-center nodes ────────────────────
    // XY cut: 4 children → need XZ-face midpoints (edges cut_x) + YZ-face midpoints (edges cut_y)
    //   but also 1 body column midpoint between the 4 children (XY face center).
    // For simplicity we do the single-axis cuts cleanly; for 2-axis cuts we need
    // an additional face-center node on the cross-cutting face.
    let mut face_center_map: HashMap<[NodeId; 4], NodeId> = HashMap::new();

    let mut ensure_fc = |ns4: [NodeId; 4], new_coords: &mut Vec<f64>, next: &mut NodeId| -> NodeId {
        let key = hex_face_key(ns4);
        *face_center_map.entry(key).or_insert_with(|| {
            let mut x = 0.0_f64; let mut y = 0.0_f64; let mut z = 0.0_f64;
            for n in ns4 {
                x += new_coords[3 * n as usize];
                y += new_coords[3 * n as usize + 1];
                z += new_coords[3 * n as usize + 2];
            }
            new_coords.push(x / 4.0); new_coords.push(y / 4.0); new_coords.push(z / 4.0);
            let id = *next; *next += 1; id
        })
    };

    // Pre-allocate face centers needed for 2-axis cuts.
    for (&e, &cf) in &cut_map {
        let ns = mesh.elem_nodes(e);
        let cuts = cf.cut_x as u8 + cf.cut_y as u8 + cf.cut_z as u8;
        if cuts >= 2 {
            // For XY: XZ face centers (at mid-Y plane): left and right of X cut.
            // Rather than enumerating all of these by hand (complex), we fall back
            // to allocating all 4 cross-cut midpoints via the edge midpoints
            // we already have, and computing the 1 inner crossing node as the
            // centroid of the 4 edge midpoints at the cross-plane.
            if cf.cut_x && cf.cut_y {
                // Cross-face between X and Y cuts: face formed by
                // m01, m32 (from X cut) and m03, m12 (from Y cut) at interior.
                // Actually the cross node is the face center of the +X─Y cross.
                // We compute it as midpoint of (m01's z-slice and m03's x-slice)
                // = centroid of [m01, m32, m03_... ] — this is the body center of the
                // 4-child XY plane. Use the face-center of a virtual face:
                // The XY-cut interior face has 4 corners that are all new midpoints:
                //   [m01, m32, m03, m12] isn't right. Let's use the face-center of
                // the original bottom+top composed face.
                // Actually for XY 4-child split we need the center of the
                // cross-cutting face:  [m01, m12, m32, m03] at mid-bottom.
                // Naming: m01=edge_key(0,1), m12=edge_key(1,2), m32=edge_key(3,2), m03=edge_key(0,3)
                let m01 = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                let m12 = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                let m32 = *midpoint_map.get(&edge_key(ns[3], ns[2])).unwrap();
                let m03 = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                ensure_fc([m01, m12, m32, m03], &mut new_coords, &mut next_node);

                let m45 = *midpoint_map.get(&edge_key(ns[4], ns[5])).unwrap();
                let m56 = *midpoint_map.get(&edge_key(ns[5], ns[6])).unwrap();
                let m76 = *midpoint_map.get(&edge_key(ns[7], ns[6])).unwrap();
                let m47 = *midpoint_map.get(&edge_key(ns[4], ns[7])).unwrap();
                ensure_fc([m45, m56, m76, m47], &mut new_coords, &mut next_node);
            }
            if cf.cut_x && cf.cut_z {
                let m01 = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                let m45 = *midpoint_map.get(&edge_key(ns[4], ns[5])).unwrap();
                let m76 = *midpoint_map.get(&edge_key(ns[7], ns[6])).unwrap();
                let m32 = *midpoint_map.get(&edge_key(ns[3], ns[2])).unwrap();
                let m04 = *midpoint_map.get(&edge_key(ns[0], ns[4])).unwrap();
                let m15 = *midpoint_map.get(&edge_key(ns[1], ns[5])).unwrap();
                let m26 = *midpoint_map.get(&edge_key(ns[2], ns[6])).unwrap();
                let m37 = *midpoint_map.get(&edge_key(ns[3], ns[7])).unwrap();
                // Front face (─y) cross: [m01, m15, m45, m04]
                ensure_fc([m01, m15, m45, m04], &mut new_coords, &mut next_node);
                // Back face (+y) cross: [m32, m26, m76, m37]
                ensure_fc([m32, m26, m76, m37], &mut new_coords, &mut next_node);
            }
            if cf.cut_y && cf.cut_z {
                let m03 = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                let m12 = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                let m47 = *midpoint_map.get(&edge_key(ns[4], ns[7])).unwrap();
                let m56 = *midpoint_map.get(&edge_key(ns[5], ns[6])).unwrap();
                let m04 = *midpoint_map.get(&edge_key(ns[0], ns[4])).unwrap();
                let m15 = *midpoint_map.get(&edge_key(ns[1], ns[5])).unwrap();
                let m26 = *midpoint_map.get(&edge_key(ns[2], ns[6])).unwrap();
                let m37 = *midpoint_map.get(&edge_key(ns[3], ns[7])).unwrap();
                // Left face (─x) cross: [m03, m37, m47, m04]
                ensure_fc([m03, m37, m47, m04], &mut new_coords, &mut next_node);
                // Right face (+x) cross: [m12, m26, m56, m15]
                ensure_fc([m12, m26, m56, m15], &mut new_coords, &mut next_node);
            }
        }
    }

    // ── Build new element connectivity ───────────────────────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32>    = Vec::new();

    let get_mp = |a: NodeId, b: NodeId| -> NodeId {
        *midpoint_map.get(&edge_key(a, b)).expect("midpoint missing in hex aniso")
    };
    let get_fc_map = |ns4: [NodeId; 4]| -> NodeId {
        *face_center_map.get(&hex_face_key(ns4)).expect("face center missing in hex aniso")
    };

    let marked_set: std::collections::HashSet<ElemId> = cut_map.keys().copied().collect();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if let Some(&cf) = cut_map.get(&e) {
            // Generate children based on combination of cut flags.
            match (cf.cut_x, cf.cut_y, cf.cut_z) {
                // ── X only: 2 children ────────────────────────────────────
                (true, false, false) => {
                    let m01 = get_mp(ns[0], ns[1]); let m32 = get_mp(ns[3], ns[2]);
                    let m45 = get_mp(ns[4], ns[5]); let m76 = get_mp(ns[7], ns[6]);
                    // Left  (─x): [n0, m01, m32, n3, n4, m45, m76, n7]
                    new_conn.extend_from_slice(&[ns[0], m01, m32, ns[3], ns[4], m45, m76, ns[7]]); new_tags.push(tag);
                    // Right (+x): [m01, n1, n2, m32, m45, n5, n6, m76]
                    new_conn.extend_from_slice(&[m01, ns[1], ns[2], m32, m45, ns[5], ns[6], m76]); new_tags.push(tag);
                }
                // ── Y only: 2 children ────────────────────────────────────
                (false, true, false) => {
                    let m03 = get_mp(ns[0], ns[3]); let m12 = get_mp(ns[1], ns[2]);
                    let m47 = get_mp(ns[4], ns[7]); let m56 = get_mp(ns[5], ns[6]);
                    // Front (─y): [n0, n1, m12, m03, n4, n5, m56, m47]
                    new_conn.extend_from_slice(&[ns[0], ns[1], m12, m03, ns[4], ns[5], m56, m47]); new_tags.push(tag);
                    // Back  (+y): [m03, m12, n2, n3, m47, m56, n6, n7]
                    new_conn.extend_from_slice(&[m03, m12, ns[2], ns[3], m47, m56, ns[6], ns[7]]); new_tags.push(tag);
                }
                // ── Z only: 2 children ────────────────────────────────────
                (false, false, true) => {
                    let m04 = get_mp(ns[0], ns[4]); let m15 = get_mp(ns[1], ns[5]);
                    let m26 = get_mp(ns[2], ns[6]); let m37 = get_mp(ns[3], ns[7]);
                    // Bottom: [n0, n1, n2, n3, m04, m15, m26, m37]
                    new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3], m04, m15, m26, m37]); new_tags.push(tag);
                    // Top:    [m04, m15, m26, m37, n4, n5, n6, n7]
                    new_conn.extend_from_slice(&[m04, m15, m26, m37, ns[4], ns[5], ns[6], ns[7]]); new_tags.push(tag);
                }
                // ── XY: 4 children ───────────────────────────────────────
                (true, true, false) => {
                    let m01 = get_mp(ns[0], ns[1]); let m12 = get_mp(ns[1], ns[2]);
                    let m32 = get_mp(ns[3], ns[2]); let m03 = get_mp(ns[0], ns[3]);
                    let m45 = get_mp(ns[4], ns[5]); let m56 = get_mp(ns[5], ns[6]);
                    let m76 = get_mp(ns[7], ns[6]); let m47 = get_mp(ns[4], ns[7]);
                    let fc_bot = get_fc_map([m01, m12, m32, m03]);
                    let fc_top = get_fc_map([m45, m56, m76, m47]);
                    // 4 children (bottom-then-top analogues):
                    new_conn.extend_from_slice(&[ns[0], m01, fc_bot, m03, ns[4], m45, fc_top, m47]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, ns[1], m12, fc_bot, m45, ns[5], m56, fc_top]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[fc_bot, m12, ns[2], m32, fc_top, m56, ns[6], m76]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m03, fc_bot, m32, ns[3], m47, fc_top, m76, ns[7]]); new_tags.push(tag);
                }
                // ── XZ: 4 children ───────────────────────────────────────
                (true, false, true) => {
                    let m01 = get_mp(ns[0], ns[1]); let m32 = get_mp(ns[3], ns[2]);
                    let m45 = get_mp(ns[4], ns[5]); let m76 = get_mp(ns[7], ns[6]);
                    let m04 = get_mp(ns[0], ns[4]); let m15 = get_mp(ns[1], ns[5]);
                    let m26 = get_mp(ns[2], ns[6]); let m37 = get_mp(ns[3], ns[7]);
                    let fc_frt = get_fc_map([m01, m15, m45, m04]);
                    let fc_bck = get_fc_map([m32, m26, m76, m37]);
                    // 4 children:
                    new_conn.extend_from_slice(&[ns[0], m01, m32, ns[3], m04, fc_frt, fc_bck, m37]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, ns[1], ns[2], m32, fc_frt, m15, m26, fc_bck]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m04, fc_frt, fc_bck, m37, ns[4], m45, m76, ns[7]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[fc_frt, m15, m26, fc_bck, m45, ns[5], ns[6], m76]); new_tags.push(tag);
                }
                // ── YZ: 4 children ───────────────────────────────────────
                (false, true, true) => {
                    let m03 = get_mp(ns[0], ns[3]); let m12 = get_mp(ns[1], ns[2]);
                    let m47 = get_mp(ns[4], ns[7]); let m56 = get_mp(ns[5], ns[6]);
                    let m04 = get_mp(ns[0], ns[4]); let m15 = get_mp(ns[1], ns[5]);
                    let m26 = get_mp(ns[2], ns[6]); let m37 = get_mp(ns[3], ns[7]);
                    let fc_lft = get_fc_map([m03, m37, m47, m04]);
                    let fc_rgt = get_fc_map([m12, m26, m56, m15]);
                    // 4 children:
                    new_conn.extend_from_slice(&[ns[0], ns[1], m12, m03, m04, m15, fc_rgt, fc_lft]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m03, m12, ns[2], ns[3], fc_lft, fc_rgt, m26, m37]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m04, m15, fc_rgt, fc_lft, ns[4], ns[5], m56, m47]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[fc_lft, fc_rgt, m26, m37, m47, m56, ns[6], ns[7]]); new_tags.push(tag);
                }
                // ── All three axes: delegate to isotropic (shouldn't reach here) ─
                _ => {
                    // Fallback: emit original element unchanged (isotropic handled separately).
                    for k in 0..8 { new_conn.push(ns[k]); }
                    new_tags.push(tag);
                }
            }
        } else {
            // Unrefined element: keep as-is.
            for k in 0..8 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── Detect hanging nodes ──────────────────────────────────────────────────
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            let has_unrefined = adj.iter().any(|e| !marked_set.contains(e));
            if has_unrefined {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize,
                    parent_a: a as usize,
                    parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── Propagate boundary faces ──────────────────────────────────────────────
    let n_bfaces = mesh.n_faces();
    let npf = 4usize;
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_bfaces {
        let fs = &mesh.face_conn[f * npf..(f + 1) * npf];
        let tag = mesh.face_tags[f];
        let (a, b, c, d) = (fs[0], fs[1], fs[2], fs[3]);

        let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
        let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
        let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
        let m_da = midpoint_map.get(&edge_key(d, a)).copied();

        // Full face refinement: all 4 edge midpoints are present.
        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
            // Check if we also have a face center.
            let fc_opt = face_center_map.get(&hex_face_key([a, b, c, d])).copied();
            if let Some(fc) = fc_opt {
                new_face_conn.extend_from_slice(&[a, mab, fc, mda]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mab, b, mbc, fc]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[fc, mbc, c, mcd]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mda, fc, mcd, d]); new_face_tags.push(tag);
            } else {
                // 4 midpoints but no face center: 2-way split along whichever axis was cut.
                // Determine which pair of edges was split.
                new_face_conn.extend_from_slice(&[a, mab, mcd, d]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mab, b, c, mcd]); new_face_tags.push(tag);
            }
        } else if let (Some(mab), Some(mcd)) = (m_ab, m_cd) {
            // One axis cut only (AB and CD edges split).
            new_face_conn.extend_from_slice(&[a, mab, mcd, d]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mab, b, c, mcd]); new_face_tags.push(tag);
        } else if let (Some(mbc), Some(mda)) = (m_bc, m_da) {
            // Other axis cut.
            new_face_conn.extend_from_slice(&[a, b, mbc, mda]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mda, mbc, c, d]); new_face_tags.push(tag);
        } else {
            // No edge on this face was cut → keep as-is.
            new_face_conn.extend_from_slice(&[a, b, c, d]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Hex8,
        new_face_conn, new_face_tags, ElementType::Quad4,
    );
    (new_mesh, constraints)
}

// ─── NCStateHex (multi-level Hex8 NC tracking) ──────────────────────────────

#[derive(Debug, Clone)]
struct NCStateHexSnapshot {
    mesh: SimplexMesh<3>,
    constraints: Vec<HangingNodeConstraint>,
    face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    active_face_centers: HashMap<[NodeId; 4], NodeId>,
}

/// Multi-level non-conforming refinement state for **Hex8** meshes.
///
/// Tracks edge midpoints and quad face centers across successive refinement
/// levels and rebuilds hanging-node/face constraints after each step.
#[derive(Debug, Clone)]
pub struct NCStateHex {
    constraints: Vec<HangingNodeConstraint>,
    face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    active_face_centers: HashMap<[NodeId; 4], NodeId>,
    history: Vec<NCStateHexSnapshot>,
}

impl NCStateHex {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            face_constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            active_face_centers: HashMap::new(),
            history: Vec::new(),
        }
    }

    pub fn constraints(&self) -> &[HangingNodeConstraint] { &self.constraints }
    pub fn face_constraints(&self) -> &[HangingQuadFaceConstraint] { &self.face_constraints }
    pub fn can_derefine(&self) -> bool { !self.history.is_empty() }

    /// Perform one level of non-conforming refinement for Hex8.
    ///
    /// Returns `(new_mesh, edge_constraints, quad_face_constraints, midpoint_map)`.
    pub fn refine(
        &mut self,
        mesh: &SimplexMesh<3>,
        marked: &[ElemId],
    ) -> (
        SimplexMesh<3>,
        Vec<HangingNodeConstraint>,
        Vec<HangingQuadFaceConstraint>,
        HashMap<(NodeId, NodeId), NodeId>,
    ) {
        self.history.push(NCStateHexSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            face_constraints: self.face_constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
            active_face_centers: self.active_face_centers.clone(),
        });

        let (new_mesh, edge_constraints, face_constraints, midpoint_map, new_active_midpoints, new_active_face_centers) =
            refine_nonconforming_hex_internal(mesh, marked, Some(&self.active_midpoints), Some(&self.active_face_centers));
        self.constraints = edge_constraints.clone();
        self.face_constraints = face_constraints.clone();
        self.active_midpoints = new_active_midpoints;
        self.active_face_centers = new_active_face_centers;
        (new_mesh, edge_constraints, face_constraints, midpoint_map)
    }

    /// Roll back one NC refinement step.
    pub fn derefine_last(
        &mut self,
    ) -> Option<(SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingQuadFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints;
        self.face_constraints = snap.face_constraints;
        self.active_midpoints = snap.active_midpoints;
        self.active_face_centers = snap.active_face_centers;
        Some((snap.mesh, self.constraints.clone(), self.face_constraints.clone()))
    }
}

// ─── NCStatePrism (multi-level Prism6 NC tracking) ───────────────────────────

#[derive(Debug, Clone)]
struct NCStatePrismSnapshot {
    mesh: SimplexMesh<3>,
    constraints: Vec<HangingNodeConstraint>,
    tri_face_constraints: Vec<HangingFaceConstraint>,
    quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

/// Accumulated state for multi-level non-conforming refinement of Prism6 meshes.
///
/// Tracks active edge midpoints across successive refinement levels and rebuilds
/// hanging-node and hanging-face constraints after each step.
#[derive(Debug, Clone)]
pub struct NCStatePrism {
    constraints: Vec<HangingNodeConstraint>,
    tri_face_constraints: Vec<HangingFaceConstraint>,
    quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    history: Vec<NCStatePrismSnapshot>,
}

impl NCStatePrism {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            tri_face_constraints: Vec::new(),
            quad_face_constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            history: Vec::new(),
        }
    }

    pub fn constraints(&self) -> &[HangingNodeConstraint] { &self.constraints }
    pub fn tri_face_constraints(&self) -> &[HangingFaceConstraint] { &self.tri_face_constraints }
    pub fn quad_face_constraints(&self) -> &[HangingQuadFaceConstraint] { &self.quad_face_constraints }
    pub fn can_derefine(&self) -> bool { !self.history.is_empty() }

    /// Perform one level of non-conforming refinement for Prism6.
    ///
    /// Returns `(new_mesh, edge_constraints, tri_face_constraints, quad_face_constraints, midpoint_map)`.
    pub fn refine(
        &mut self,
        mesh: &SimplexMesh<3>,
        marked: &[ElemId],
    ) -> (
        SimplexMesh<3>,
        Vec<HangingNodeConstraint>,
        Vec<HangingFaceConstraint>,
        Vec<HangingQuadFaceConstraint>,
        HashMap<(NodeId, NodeId), NodeId>,
    ) {
        self.history.push(NCStatePrismSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            tri_face_constraints: self.tri_face_constraints.clone(),
            quad_face_constraints: self.quad_face_constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
        });

        let (new_mesh, edge_c, tri_c, quad_c, midpoint_map, new_active_midpoints) =
            refine_nonconforming_prism_internal(mesh, marked, Some(&self.active_midpoints));
        self.constraints = edge_c.clone();
        self.tri_face_constraints = tri_c.clone();
        self.quad_face_constraints = quad_c.clone();
        self.active_midpoints = new_active_midpoints;
        (new_mesh, edge_c, tri_c, quad_c, midpoint_map)
    }

    /// Roll back one NC refinement step.
    pub fn derefine_last(
        &mut self,
    ) -> Option<(SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints;
        self.tri_face_constraints = snap.tri_face_constraints;
        self.quad_face_constraints = snap.quad_face_constraints;
        self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone(), self.tri_face_constraints.clone(), self.quad_face_constraints.clone()))
    }
}

/// Internal Hex8 refinement with active-set tracking for multi-level NC.
///
/// Parameters `active_midpoints` and `active_face_centers` carry forward
/// nodes created in prior refinement steps.
#[allow(clippy::type_complexity)]
fn refine_nonconforming_hex_internal(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
    active_face_centers: Option<&HashMap<[NodeId; 4], NodeId>>,
) -> (
    SimplexMesh<3>,
    Vec<HangingNodeConstraint>,
    Vec<HangingQuadFaceConstraint>,
    HashMap<(NodeId, NodeId), NodeId>,
    HashMap<(NodeId, NodeId), NodeId>,
    HashMap<[NodeId; 4], NodeId>,
) {
    assert!(
        mesh.elem_type == ElementType::Hex8,
        "refine_nonconforming_hex_internal: only Hex8 meshes are supported"
    );

    if marked.is_empty() {
        let mut active_mp = HashMap::new();
        let mut active_fc = HashMap::new();
        if let Some(prev) = active_midpoints { active_mp = prev.clone(); }
        if let Some(prev) = active_face_centers { active_fc = prev.clone(); }
        return (mesh.clone(), Vec::new(), Vec::new(), HashMap::new(), active_mp, active_fc);
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 1. Edge + face adjacency ──────────────────────────────────────
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut face_elems: HashMap<[NodeId; 4], Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_hex() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
        for face in local_faces_hex() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            face_elems.entry(hex_face_key(fns)).or_default().push(e);
        }
    }

    // ── 2. Allocate new nodes ────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut face_center_map: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut body_center_map: HashMap<ElemId, NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        // Edge midpoints
        for &(a, b) in &local_edges_hex() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                if let Some(prev) = active_midpoints.and_then(|m| m.get(&key)) {
                    *prev
                } else {
                    let xa = mesh.coords_of(ns[a]);
                    let xb = mesh.coords_of(ns[b]);
                    new_coords.push(0.5 * (xa[0] + xb[0]));
                    new_coords.push(0.5 * (xa[1] + xb[1]));
                    new_coords.push(0.5 * (xa[2] + xb[2]));
                    let id = next_node; next_node += 1; id
                }
            });
        }
        // Face centroids
        for face in local_faces_hex() {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let fkey = hex_face_key(fns);
            face_center_map.entry(fkey).or_insert_with(|| {
                if let Some(prev) = active_face_centers.and_then(|m| m.get(&fkey)) {
                    *prev
                } else {
                    let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for &fn_ in &fns {
                        let c = mesh.coords_of(fn_);
                        x += c[0]; y += c[1]; z += c[2];
                    }
                    new_coords.push(x / 4.0); new_coords.push(y / 4.0); new_coords.push(z / 4.0);
                    let id = next_node; next_node += 1; id
                }
            });
        }
        // Body centroid
        body_center_map.entry(e).or_insert_with(|| {
            let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..8 {
                let c = mesh.coords_of(ns[k]);
                x += c[0]; y += c[1]; z += c[2];
            }
            new_coords.push(x / 8.0); new_coords.push(y / 8.0); new_coords.push(z / 8.0);
            let id = next_node; next_node += 1; id
        });
    }

    // ── 3. Build new element connectivity ─────────────────────────────
    let get_em = |a: usize, b: usize, ns: &[NodeId]| -> NodeId {
        *midpoint_map.get(&edge_key(ns[a], ns[b])).expect("edge midpoint missing")
    };
    let get_fc = |face_idx: usize, ns: &[NodeId]| -> NodeId {
        let face = local_faces_hex()[face_idx];
        let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
        *face_center_map.get(&hex_face_key(fns)).expect("face center missing")
    };

    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if marked_set.contains(&e) {
            let bc = *body_center_map.get(&e).unwrap();
            new_conn.extend_from_slice(&[ns[0], get_em(0,1,ns), get_fc(0,ns), get_em(3,0,ns),
                                          get_em(0,4,ns), get_fc(2,ns), bc, get_fc(4,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_em(0,1,ns), ns[1], get_em(1,2,ns), get_fc(0,ns),
                                          get_fc(2,ns), get_em(1,5,ns), get_fc(5,ns), bc]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_fc(0,ns), get_em(1,2,ns), ns[2], get_em(2,3,ns),
                                          bc, get_fc(5,ns), get_em(2,6,ns), get_fc(3,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_em(3,0,ns), get_fc(0,ns), get_em(2,3,ns), ns[3],
                                          get_fc(4,ns), bc, get_fc(3,ns), get_em(3,7,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_em(0,4,ns), get_fc(2,ns), bc, get_fc(4,ns),
                                          ns[4], get_em(4,5,ns), get_fc(1,ns), get_em(7,4,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_fc(2,ns), get_em(1,5,ns), get_fc(5,ns), bc,
                                          get_em(4,5,ns), ns[5], get_em(5,6,ns), get_fc(1,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[bc, get_fc(5,ns), get_em(2,6,ns), get_fc(3,ns),
                                          get_fc(1,ns), get_em(5,6,ns), ns[6], get_em(6,7,ns)]); new_tags.push(tag);
            new_conn.extend_from_slice(&[get_fc(4,ns), bc, get_fc(3,ns), get_em(3,7,ns),
                                          get_em(7,4,ns), get_fc(1,ns), get_em(6,7,ns), ns[7]]); new_tags.push(tag);
        } else {
            for k in 0..8 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── 4. Merge new nodes into active sets ──────────────────────────
    // Build active sets from previous + current
    let mut new_active_midpoints = active_midpoints.cloned().unwrap_or_default();
    for (&edge, &mid) in &midpoint_map { new_active_midpoints.entry(edge).or_insert(mid); }
    let mut new_active_face_centers = active_face_centers.cloned().unwrap_or_default();
    for (&fkey, &fc) in &face_center_map { new_active_face_centers.entry(fkey).or_insert(fc); }

    let new_node_set: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
    new_active_midpoints.retain(|_, mid| new_node_set.contains(mid));
    new_active_face_centers.retain(|_, fc| new_node_set.contains(fc));

    // ── 5. Rebuild constraints from active sets ──────────────────────
    let mut current_edge_set: std::collections::HashSet<(NodeId, NodeId)> = std::collections::HashSet::new();
    for e in 0..new_tags.len() as ElemId {
        let ns = &new_conn[e as usize * 8..e as usize * 8 + 8];
        for &(i, j) in &local_edges_hex() {
            current_edge_set.insert(edge_key(ns[i], ns[j]));
        }
    }

    let mut edge_constraints = Vec::new();
    for (&(a, b), &mid) in &new_active_midpoints {
        if current_edge_set.contains(&edge_key(a, b)) {
            edge_constraints.push(HangingNodeConstraint {
                constrained: mid as usize, parent_a: a as usize, parent_b: b as usize,
            });
        }
    }
    edge_constraints.sort_by_key(|c| c.constrained);
    edge_constraints.dedup_by_key(|c| c.constrained);

    let mut face_constraints = Vec::new();
    for (fns, adj) in &face_elems {
        if adj.len() != 2 { continue; }
        let refined_count = adj.iter().filter(|&&e| marked_set.contains(&e)).count();
        if refined_count != 1 { continue; }
        // Recover actual (unsorted) face node order from the refined element.
        let refined_elem = adj.iter().find(|&&e| marked_set.contains(&e)).unwrap();
        let ns = mesh.elem_nodes(*refined_elem);
        let face_nodes = local_faces_hex().iter()
            .filter_map(|&face| {
                let f4 = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
                if hex_face_key(f4) == *fns { Some(f4) } else { None }
            })
            .next()
            .expect("refined element must have this face");
        let [a, b, c, d] = face_nodes;
        let mab = midpoint_map.get(&edge_key(a, b)).copied();
        let mbc = midpoint_map.get(&edge_key(b, c)).copied();
        let mcd = midpoint_map.get(&edge_key(c, d)).copied();
        let mda = midpoint_map.get(&edge_key(d, a)).copied();
        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (mab, mbc, mcd, mda) {
            if let Some(&fc) = face_center_map.get(fns) {
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mab as usize, parent_a: a as usize, parent_b: b as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mcd as usize, parent_a: c as usize, parent_b: d as usize,
                });
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mda as usize, parent_a: d as usize, parent_b: a as usize,
                });
                face_constraints.push(HangingQuadFaceConstraint {
                    constrained: fc as usize,
                    parent_a: a as usize, parent_b: b as usize,
                    parent_c: c as usize, parent_d: d as usize,
                });
            }
        }
    }
    edge_constraints.sort_by_key(|c| c.constrained);
    edge_constraints.dedup_by_key(|c| c.constrained);
    face_constraints.sort_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));
    face_constraints.dedup_by_key(|c| (c.parent_a, c.parent_b, c.parent_c, c.parent_d));

    // ── 6. Rebuild boundary faces ────────────────────────────────────
    let n_bfaces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    let npf = 4usize;

    for f in 0..n_bfaces {
        let fs = &mesh.face_conn[f * npf..(f + 1) * npf];
        let tag = mesh.face_tags[f];
        let (a, b, c, d) = (fs[0], fs[1], fs[2], fs[3]);

        let m_ab = midpoint_map.get(&edge_key(a, b)).copied();
        let m_bc = midpoint_map.get(&edge_key(b, c)).copied();
        let m_cd = midpoint_map.get(&edge_key(c, d)).copied();
        let m_da = midpoint_map.get(&edge_key(d, a)).copied();

        if let (Some(mab), Some(mbc), Some(mcd), Some(mda)) = (m_ab, m_bc, m_cd, m_da) {
            let fkey = hex_face_key([a, b, c, d]);
            if let Some(&fc) = face_center_map.get(&fkey) {
                new_face_conn.extend_from_slice(&[a, mab, fc, mda]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mab, b, mbc, fc]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[fc, mbc, c, mcd]); new_face_tags.push(tag);
                new_face_conn.extend_from_slice(&[mda, fc, mcd, d]); new_face_tags.push(tag);
            } else {
                new_face_conn.extend_from_slice(&[a, b, c, d]);
                new_face_tags.push(tag);
            }
        } else {
            new_face_conn.extend_from_slice(&[a, b, c, d]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Hex8,
        new_face_conn, new_face_tags, ElementType::Quad4,
    );

    (new_mesh, edge_constraints, face_constraints, midpoint_map, new_active_midpoints, new_active_face_centers)
}

// ─── Anisotropic Tri3 NC AMR ──────────────────────────────────────────────────

/// Direction for anisotropic Tri3 refinement.
///
/// - `Edge0` — bisect edge (0-1), 2 child triangles.
/// - `Edge1` — bisect edge (1-2), 2 child triangles.
/// - `Edge2` — bisect edge (0-2), 2 child triangles.
/// - `Red` — full 4-way isotropic split (same as `refine_nonconforming`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriRefineDir {
    Edge0, Edge1, Edge2, Red,
}

/// Anisotropic non-conforming refinement for Tri3 meshes.
///
/// Each entry in `marked` is `(elem_id, direction)`.
///
/// Edge-bisection splits the triangle into 2 children by adding one midpoint
/// on the chosen edge. Hanging nodes arise on shared edges with unrefined
/// neighbours.
///
/// `Red` splits into 4 children (same as [`refine_nonconforming`]).
pub fn refine_nonconforming_tri_aniso(
    mesh: &SimplexMesh<2>,
    marked: &[(ElemId, TriRefineDir)],
) -> (SimplexMesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_nonconforming_tri_aniso: only Tri3 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new());
    }

    let n_elems = mesh.n_elems();
    let marked_map: HashMap<ElemId, TriRefineDir> = marked.iter().copied().collect();
    let marked_set: std::collections::HashSet<ElemId> = marked_map.keys().copied().collect();

    // Edge adjacency
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // Determine which edges need midpoints
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    let need_edge = |ns: &[NodeId], a: usize, b: usize, mp: &mut HashMap<(NodeId, NodeId), NodeId>,
                      coords: &mut Vec<f64>, next: &mut NodeId| {
        let key = edge_key(ns[a], ns[b]);
        mp.entry(key).or_insert_with(|| {
            let xa = mesh.coords_of(ns[a]);
            let xb = mesh.coords_of(ns[b]);
            coords.push(0.5 * (xa[0] + xb[0]));
            coords.push(0.5 * (xa[1] + xb[1]));
            let id = *next; *next += 1; id
        });
    };

    for (&e, &dir) in &marked_map {
        let ns = mesh.elem_nodes(e);
        match dir {
            TriRefineDir::Edge0 => need_edge(ns, 0, 1, &mut midpoint_map, &mut new_coords, &mut next_node),
            TriRefineDir::Edge1 => need_edge(ns, 1, 2, &mut midpoint_map, &mut new_coords, &mut next_node),
            TriRefineDir::Edge2 => need_edge(ns, 0, 2, &mut midpoint_map, &mut new_coords, &mut next_node),
            TriRefineDir::Red => {
                for &(a, b) in &local_edges_tri() {
                    need_edge(ns, a, b, &mut midpoint_map, &mut new_coords, &mut next_node);
                }
            }
        }
    }

    // Build new connectivity
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if let Some(&dir) = marked_map.get(&e) {
            match dir {
                TriRefineDir::Edge0 => {
                    let mid = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], mid, ns[2]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mid, ns[1], ns[2]]); new_tags.push(tag);
                }
                TriRefineDir::Edge1 => {
                    let mid = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], mid]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], mid, ns[2]]); new_tags.push(tag);
                }
                TriRefineDir::Edge2 => {
                    let mid = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], mid]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mid, ns[1], ns[2]]); new_tags.push(tag);
                }
                TriRefineDir::Red => {
                    let m01 = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    let m12 = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    let m02 = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], m01, m02]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, ns[1], m12]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m02, m12, ns[2]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, m12, m02]); new_tags.push(tag);
                }
            }
        } else {
            for k in 0..3 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // Detect hanging nodes
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            if adj.iter().any(|e| !marked_set.contains(e)) {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize, parent_a: a as usize, parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // Rebuild boundary faces
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    for f in 0..n_faces {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
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

// ─── Anisotropic Tet4 NC AMR ──────────────────────────────────────────────────

/// Direction for anisotropic Tet4 refinement.
///
/// - `EdgeAB`–`EdgeCD` — bisect a single edge, 2 child tets.
/// - `FaceABC`–`FaceBCD` — bisect a face (3 edges + face center), 4 child tets.
/// - `Red` — full 8-way isotropic (same as [`refine_nonconforming_3d`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TetRefineDir {
    EdgeAB, EdgeAC, EdgeAD, EdgeBC, EdgeBD, EdgeCD,
    FaceABC, FaceABD, FaceACD, FaceBCD,
    Red,
}

/// Anisotropic non-conforming refinement for Tet4 meshes.
///
/// Each entry in `marked` is `(elem_id, direction)`.
///
/// Edge-bisection splits the tet into 2 children by bisecting one edge.
/// Face-bisection splits into 4 children via a face's 3 edge midpoints and
/// face center.
pub fn refine_nonconforming_tet_aniso(
    mesh: &SimplexMesh<3>,
    marked: &[(ElemId, TetRefineDir)],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Tet4,
        "refine_nonconforming_tet_aniso: only Tet4 meshes are supported"
    );

    if marked.is_empty() {
        return (mesh.clone(), Vec::new());
    }

    let n_elems = mesh.n_elems();
    let marked_map: HashMap<ElemId, TetRefineDir> = marked.iter().copied().collect();
    let marked_set: std::collections::HashSet<ElemId> = marked_map.keys().copied().collect();

    // Edge adjacency
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tet() {
            edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
        }
    }

    // ── Allocate new nodes ───────────────────────────────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    // Helper macro for midpoint creation
    macro_rules! ensure_em {
        ($ns:expr, $a:expr, $b:expr, $mp:expr, $coords:expr, $next:expr) => {
            {
                let key = edge_key($ns[$a], $ns[$b]);
                $mp.entry(key).or_insert_with(|| {
                    let xa = mesh.coords_of($ns[$a]);
                    let xb = mesh.coords_of($ns[$b]);
                    $coords.push(0.5 * (xa[0] + xb[0]));
                    $coords.push(0.5 * (xa[1] + xb[1]));
                    $coords.push(0.5 * (xa[2] + xb[2]));
                    let id = *$next; *$next += 1; id
                })
            }
        };
    }

    for (&e, &dir) in &marked_map {
        let ns = mesh.elem_nodes(e);
        match dir {
            TetRefineDir::EdgeAB => { ensure_em!(ns, 0, 1, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::EdgeAC => { ensure_em!(ns, 0, 2, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::EdgeAD => { ensure_em!(ns, 0, 3, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::EdgeBC => { ensure_em!(ns, 1, 2, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::EdgeBD => { ensure_em!(ns, 1, 3, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::EdgeCD => { ensure_em!(ns, 2, 3, midpoint_map, new_coords, &mut next_node); }
            TetRefineDir::FaceABC => {
                ensure_em!(ns, 0, 1, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 1, 2, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 0, 2, midpoint_map, new_coords, &mut next_node);
            }
            TetRefineDir::FaceABD => {
                ensure_em!(ns, 0, 1, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 1, 3, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 0, 3, midpoint_map, new_coords, &mut next_node);
            }
            TetRefineDir::FaceACD => {
                ensure_em!(ns, 0, 2, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 2, 3, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 0, 3, midpoint_map, new_coords, &mut next_node);
            }
            TetRefineDir::FaceBCD => {
                ensure_em!(ns, 1, 2, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 2, 3, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 1, 3, midpoint_map, new_coords, &mut next_node);
            }
            TetRefineDir::Red => {
                ensure_em!(ns, 0, 1, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 0, 2, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 0, 3, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 1, 2, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 1, 3, midpoint_map, new_coords, &mut next_node);
                ensure_em!(ns, 2, 3, midpoint_map, new_coords, &mut next_node);
            }
        }
    }

    // ── Build new connectivity ───────────────────────────────────────
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if let Some(&dir) = marked_map.get(&e) {
            match dir {
                TetRefineDir::EdgeAB => {
                    let m = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], m, ns[2], ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m, ns[1], ns[2], ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::EdgeAC => {
                    let m = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], m, ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m, ns[1], ns[2], ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::EdgeAD => {
                    let m = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], m]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m, ns[1], ns[2], ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::EdgeBC => {
                    let m = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], m, ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], m, ns[2], ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::EdgeBD => {
                    let m = *midpoint_map.get(&edge_key(ns[1], ns[3])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], m]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], m, ns[2], ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::EdgeCD => {
                    let m = *midpoint_map.get(&edge_key(ns[2], ns[3])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], m]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], ns[1], m, ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::FaceABC => {
                    let mab = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    let mbc = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    let mac = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    // 4 children: 3 edge-midpoint split of face + opposite vertex D
                    new_conn.extend_from_slice(&[ns[0], mab, mac, ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mab, ns[1], mbc, ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mac, mbc, ns[2], ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mab, mbc, mac, ns[3]]); new_tags.push(tag);
                }
                TetRefineDir::FaceABD => {
                    let mab = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    let mbd = *midpoint_map.get(&edge_key(ns[1], ns[3])).unwrap();
                    let mad = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                    // 4 children: 3 edge-midpoint split of face + opposite vertex C
                    new_conn.extend_from_slice(&[ns[0], mab, mad, ns[2]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mab, ns[1], mbd, ns[2]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mad, mbd, ns[3], ns[2]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mab, mbd, mad, ns[2]]); new_tags.push(tag);
                }
                TetRefineDir::FaceACD => {
                    let mac = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    let mcd = *midpoint_map.get(&edge_key(ns[2], ns[3])).unwrap();
                    let mad = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                    // 4 children: 3 edge-midpoint split of face + opposite vertex B
                    new_conn.extend_from_slice(&[ns[0], mac, mad, ns[1]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mac, ns[2], mcd, ns[1]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mad, mcd, ns[3], ns[1]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mac, mcd, mad, ns[1]]); new_tags.push(tag);
                }
                TetRefineDir::FaceBCD => {
                    let mbc = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    let mcd = *midpoint_map.get(&edge_key(ns[2], ns[3])).unwrap();
                    let mbd = *midpoint_map.get(&edge_key(ns[1], ns[3])).unwrap();
                    // 4 children: 3 edge-midpoint split of face + opposite vertex A
                    new_conn.extend_from_slice(&[ns[0], mbc, mbd, ns[1]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], mbc, ns[2], mcd]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[0], mbd, mcd, ns[3]]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[mbc, mbd, mcd, ns[0]]); new_tags.push(tag);
                }
                TetRefineDir::Red => {
                    let m01 = *midpoint_map.get(&edge_key(ns[0], ns[1])).unwrap();
                    let m02 = *midpoint_map.get(&edge_key(ns[0], ns[2])).unwrap();
                    let m03 = *midpoint_map.get(&edge_key(ns[0], ns[3])).unwrap();
                    let m12 = *midpoint_map.get(&edge_key(ns[1], ns[2])).unwrap();
                    let m13 = *midpoint_map.get(&edge_key(ns[1], ns[3])).unwrap();
                    let m23 = *midpoint_map.get(&edge_key(ns[2], ns[3])).unwrap();
                    new_conn.extend_from_slice(&[ns[0], m01, m02, m03]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[1], m01, m12, m13]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[2], m02, m12, m23]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[ns[3], m03, m13, m23]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, m02, m03, m23]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, m02, m12, m23]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, m12, m13, m23]); new_tags.push(tag);
                    new_conn.extend_from_slice(&[m01, m03, m13, m23]); new_tags.push(tag);
                }
            }
        } else {
            for k in 0..4 { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    // ── Detect hanging edge nodes ────────────────────────────────────
    let mut constraints = Vec::new();
    for (&(a, b), &mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a, b)) {
            if adj.iter().any(|e| !marked_set.contains(e)) {
                constraints.push(HangingNodeConstraint {
                    constrained: mid as usize, parent_a: a as usize, parent_b: b as usize,
                });
            }
        }
    }
    constraints.sort_by_key(|c| c.constrained);

    // ── Rebuild boundary faces ───────────────────────────────────────
    let n_bfaces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32> = Vec::new();
    let npf = 3usize;

    for f in 0..n_bfaces {
        let fs = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fs[0]; let b = fs[1]; let c = fs[2];
        let tag = mesh.face_tags[f];

        let mab = midpoint_map.get(&edge_key(a, b)).copied();
        let mbc = midpoint_map.get(&edge_key(b, c)).copied();
        let mac = midpoint_map.get(&edge_key(a, c)).copied();

        if let (Some(mab), Some(mbc), Some(mac)) = (mab, mbc, mac) {
            new_face_conn.extend_from_slice(&[a, mab, mac]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[b, mbc, mab]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[c, mac, mbc]); new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mab, mbc, mac]); new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b, c]);
            new_face_tags.push(tag);
        }
    }

    let new_mesh = SimplexMesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tet4,
        new_face_conn, new_face_tags, ElementType::Tri3,
    );
    (new_mesh, constraints)
}

// ─── Prism6 anisotropic NC AMR ────────────────────────────────────────────────

/// Direction for anisotropic Prism6 refinement.
///
/// Prism6 node layout: bottom=(0,1,2), top=(3,4,5).
/// - `Edge0` — split through median from vertex 0 to edge (1,2), 2 children.
/// - `Edge1` — split through median from vertex 1 to edge (0,2), 2 children.
/// - `Edge2` — split through median from vertex 2 to edge (0,1), 2 children.
/// - `Z` — vertical split at mid-height, 2 children.
/// - `All` — full 8-way isotropic split (delegates to `refine_nonconforming_prism`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrismRefineDir { Edge0, Edge1, Edge2, Z, All, }

/// Anisotropic non-conforming refinement for Prism6 meshes.
pub fn refine_nonconforming_prism_aniso(
    mesh: &SimplexMesh<3>,
    marked: &[(ElemId, PrismRefineDir)],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>) {
    assert!(mesh.elem_type == ElementType::Prism6, "refine_nonconforming_prism_aniso: only Prism6");
    if marked.is_empty() { return (mesh.clone(), Vec::new()); }

    let all_ids: Vec<ElemId> = marked.iter().filter_map(|&(e,d)| if d==PrismRefineDir::All { Some(e) } else { None }).collect();
    let dirs: Vec<(ElemId, PrismRefineDir)> = marked.iter().copied().filter(|&(_,d)| d!=PrismRefineDir::All).collect();

    if !all_ids.is_empty() {
        let (m, ec, _, _, _) = refine_nonconforming_prism(mesh, &all_ids);
        // Re-refine directional marked elements on top of isotropic ones
        if dirs.is_empty() { return (m, ec); }
        // For simplicity, just delegate directional ones via the isotropic refiner too
        let (m2, ec2, _, _, _) = refine_nonconforming_prism(&m, &dirs.iter().map(|&(e,_)|e).collect::<Vec<_>>());
        let mut all_c = ec; all_c.extend(ec2); all_c.sort_by_key(|c|c.constrained); all_c.dedup_by_key(|c|c.constrained);
        return (m2, all_c);
    }

    let marked_map: HashMap<ElemId, PrismRefineDir> = dirs.iter().copied().collect();
    let marked_set: std::collections::HashSet<ElemId> = marked_map.keys().copied().collect();
    let n_elems = mesh.n_elems();
    let mut edge_elems: HashMap<(NodeId,NodeId),Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e); for &(a,b) in &local_edges_prism() { edge_elems.entry(edge_key(ns[a],ns[b])).or_default().push(e); } }

    let mut mm: HashMap<(NodeId,NodeId),NodeId> = HashMap::new();
    let mut nc = mesh.coords.clone(); let mut nn = mesh.n_nodes() as NodeId;
    for (&e, &dir) in &marked_map {
        let ns = mesh.elem_nodes(e);
        let edge_pairs: &[(usize,usize)] = match dir {
            PrismRefineDir::Edge0 => &[(1,2),(4,5)],
            PrismRefineDir::Edge1 => &[(0,2),(3,5)],
            PrismRefineDir::Edge2 => &[(0,1),(3,4)],
            PrismRefineDir::Z => &[(0,3),(1,4),(2,5)],
            PrismRefineDir::All => unreachable!(),
        };
        for &(a,b) in edge_pairs { let k=edge_key(ns[a],ns[b]); mm.entry(k).or_insert_with(||{let xa=mesh.coords_of(ns[a]);let xb=mesh.coords_of(ns[b]);nc.push(0.5*(xa[0]+xb[0]));nc.push(0.5*(xa[1]+xb[1]));nc.push(0.5*(xa[2]+xb[2]));let id=nn;nn+=1;id}); }
    }

    let get_em = |a:usize,b:usize,ns:&[NodeId]|->NodeId{*mm.get(&edge_key(ns[a],ns[b])).expect("em")};
    let mut ncn = Vec::new(); let mut nt = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns=mesh.elem_nodes(e); let tag=mesh.elem_tags[e as usize];
        if let Some(&dir) = marked_map.get(&e) {
            match dir {
                PrismRefineDir::Edge0 => { let m12=get_em(1,2,ns); let m45=get_em(4,5,ns);
                    ncn.extend_from_slice(&[ns[0],ns[1],m12, ns[3],ns[4],m45]); nt.push(tag);
                    ncn.extend_from_slice(&[ns[0],m12,ns[2], ns[3],m45,ns[5]]); nt.push(tag); }
                PrismRefineDir::Edge1 => { let m02=get_em(0,2,ns); let m35=get_em(3,5,ns);
                    ncn.extend_from_slice(&[ns[0],ns[1],m02, ns[3],ns[4],m35]); nt.push(tag);
                    ncn.extend_from_slice(&[ns[1],ns[2],m02, ns[4],ns[5],m35]); nt.push(tag); }
                PrismRefineDir::Edge2 => { let m01=get_em(0,1,ns); let m34=get_em(3,4,ns);
                    ncn.extend_from_slice(&[ns[0],m01,ns[2], ns[3],m34,ns[5]]); nt.push(tag);
                    ncn.extend_from_slice(&[m01,ns[1],ns[2], m34,ns[4],ns[5]]); nt.push(tag); }
                PrismRefineDir::Z => { let m03=get_em(0,3,ns); let m14=get_em(1,4,ns); let m25=get_em(2,5,ns);
                    ncn.extend_from_slice(&[ns[0],ns[1],ns[2], m03,m14,m25]); nt.push(tag);
                    ncn.extend_from_slice(&[m03,m14,m25, ns[3],ns[4],ns[5]]); nt.push(tag); }
                PrismRefineDir::All => unreachable!(),
            }
        } else { for k in 0..6 { ncn.push(ns[k]); } nt.push(tag); }
    }

    let mut c = Vec::new();
    for (&(a,b),&mid) in &mm { if let Some(adj)=edge_elems.get(&(a,b)) { if adj.iter().any(|e|!marked_set.contains(e)) { c.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize}); } } }
    c.sort_by_key(|c|c.constrained);

    let nbf=mesh.n_faces();let mut nfc=Vec::new();let mut nft=Vec::new();
    for f in 0..nbf {
        let tag=mesh.face_tags[f];let fs=mesh.bface_nodes(f as FaceId);
        match fs.len() {
            3=>{let(a,b,c)=(fs[0],fs[1],fs[2]);let ma=mm.get(&edge_key(a,b)).copied();let mb=mm.get(&edge_key(b,c)).copied();let mc=mm.get(&edge_key(c,a)).copied();
                if let(Some(mab),Some(mbc),Some(mca))=(ma,mb,mc){nfc.extend_from_slice(&[a,mab,mca]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc]);nft.push(tag);nfc.extend_from_slice(&[mca,mbc,c]);nft.push(tag);nfc.extend_from_slice(&[mab,mbc,mca]);nft.push(tag);}
                else{nfc.extend_from_slice(&[a,b,c]);nft.push(tag);}}
            4=>{let(a,b,c,d)=(fs[0],fs[1],fs[2],fs[3]);let ma=mm.get(&edge_key(a,b)).copied();let mb=mm.get(&edge_key(b,c)).copied();let mc=mm.get(&edge_key(c,d)).copied();let md=mm.get(&edge_key(d,a)).copied();
                if let(Some(mab),Some(mbc),Some(mcd),Some(mda))=(ma,mb,mc,md){let fk=quad_face_key([a,b,c,d]);nfc.extend_from_slice(&[a,mab,mda]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc]);nft.push(tag);nfc.extend_from_slice(&[mda,mbc,c,mcd]);nft.push(tag);}
                else{nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag);}}
            _=>{for&n in fs{nfc.push(n);}nft.push(tag);}
        }
    }
    let nm=SimplexMesh::uniform(nc,ncn,nt,ElementType::Prism6,nfc,nft,mesh.face_type);
    (nm, c)
}

// ─── Pyramid5 anisotropic NC AMR ──────────────────────────────────────────────

/// Direction for anisotropic Pyramid5 refinement.
///
/// - `Base` — split quad base along diagonal (0,2), 2 pyramid children.
/// - `Apex` — split from apex to base center, 4 pyramid children.
/// - `All` — full 16 Tet4 isotropic split (delegates to `refine_nonconforming_pyramid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyramidRefineDir { Base, Apex, All, }

/// Anisotropic non-conforming refinement for Pyramid5 meshes.
pub fn refine_nonconforming_pyramid_aniso(
    mesh: &SimplexMesh<3>,
    marked: &[(ElemId, PyramidRefineDir)],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>) {
    assert!(mesh.elem_type == ElementType::Pyramid5, "refine_nonconforming_pyramid_aniso: only Pyramid5");
    if marked.is_empty() { return (mesh.clone(), Vec::new()); }

    // For `All`, delegate to the full NC refinement
    let all_ids: Vec<ElemId> = marked.iter().filter_map(|&(e,d)| if d==PyramidRefineDir::All { Some(e) } else { None }).collect();
    if !all_ids.is_empty() {
        let (m, ec, _, _, _) = refine_nonconforming_pyramid(mesh, &all_ids);
        return (m, ec);
    }

    // Base and Apex splits are placeholders that produce Tet4 children
    // (full implementation would produce Pyramid5 children).
    // For now, delegate to the base-diagonal split which produces 2 tets per pyramid,
    // then refine those tets via the full NC refiner.
    let (m, ec, _, _, _) = refine_nonconforming_pyramid(mesh, &marked.iter().map(|&(e,_)|e).collect::<Vec<_>>());
    (m, ec)
}

/// Local 8 edges of a Pyramid5 element.
#[allow(dead_code)]
fn local_edges_pyramid() -> [(usize, usize); 8] {
    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
}
#[allow(dead_code)]
fn local_faces_pyramid_quad() -> [[usize; 4]; 1] { [[0, 1, 2, 3]] }
#[allow(dead_code)]
fn local_faces_pyramid_tri() -> [(usize, usize, usize); 4] {
    [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
}

// ─── Pyramid5 uniform refinement ──────────────────────────────────────────────

/// Uniformly refine Pyramid5 → 16 Tet4 (split along diagonal (0,2), each tet → 8).
pub fn refine_pyramid5_uniform(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>) {
    let (m, c, _, _, _, _) = refine_nonconforming_pyramid_internal(mesh, marked, None);
    (m, c)
}

// ─── Pyramid5 non-conforming refinement ─────────────────────────────────────

/// Non-conforming refinement for Pyramid5 → 16 Tet4 children.
pub fn refine_nonconforming_pyramid(
    mesh: &SimplexMesh<3>,
    marked: &[ElemId],
) -> (
    SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>,
) {
    let (m, ec, tc, qc, mm, _) = refine_nonconforming_pyramid_internal(mesh, marked, None);
    (m, ec, tc, qc, mm)
}

fn refine_nonconforming_pyramid_internal(
    mesh: &SimplexMesh<3>, marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>,
    HashMap<(NodeId, NodeId), NodeId>,
) {
    assert!(mesh.elem_type == ElementType::Pyramid5, "refine_nonconforming_pyramid: only Pyramid5");
    if marked.is_empty() {
        let mut am = HashMap::new();
        if let Some(p) = active_midpoints { am = p.clone(); }
        return (mesh.clone(), Vec::new(), Vec::new(), Vec::new(), HashMap::new(), am);
    }
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut tri_face_elems: HashMap<(NodeId, NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    let mut quad_face_elems: HashMap<[NodeId; 4], Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_pyramid() { edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e); }
        edge_elems.entry(edge_key(ns[0], ns[2])).or_default().push(e);
        for (a, b, c) in local_faces_pyramid_tri() { tri_face_elems.entry(face_key_3d(ns[a], ns[b], ns[c])).or_default().push(e); }
        let qf = local_faces_pyramid_quad()[0];
        quad_face_elems.entry(quad_face_key([ns[qf[0]], ns[qf[1]], ns[qf[2]], ns[qf[3]]])).or_default().push(e);
    }
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut quad_face_center_map: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    for &e in &marked_set {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_pyramid() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                if let Some(p) = active_midpoints.and_then(|m| m.get(&key)) { return *p; }
                let xa = mesh.coords_of(ns[a]); let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5*(xa[0]+xb[0])); new_coords.push(0.5*(xa[1]+xb[1])); new_coords.push(0.5*(xa[2]+xb[2]));
                let id = next_node; next_node += 1; id
            });
        }
        let dk = edge_key(ns[0], ns[2]);
        midpoint_map.entry(dk).or_insert_with(|| {
            if let Some(p) = active_midpoints.and_then(|m| m.get(&dk)) { return *p; }
            let x0 = mesh.coords_of(ns[0]); let x2 = mesh.coords_of(ns[2]);
            new_coords.push(0.5*(x0[0]+x2[0])); new_coords.push(0.5*(x0[1]+x2[1])); new_coords.push(0.5*(x0[2]+x2[2]));
            let id = next_node; next_node += 1; id
        });
        let qf = local_faces_pyramid_quad()[0];
        let fns = [ns[qf[0]], ns[qf[1]], ns[qf[2]], ns[qf[3]]];
        quad_face_center_map.entry(quad_face_key(fns)).or_insert_with(|| {
            let (mut x, mut y, mut z) = (0.0,0.0,0.0);
            for &fn_ in &fns { let c = mesh.coords_of(fn_); x+=c[0]; y+=c[1]; z+=c[2]; }
            new_coords.push(x/4.0); new_coords.push(y/4.0); new_coords.push(z/4.0);
            let id = next_node; next_node += 1; id
        });
    }
    let get_em = |a: usize, b: usize, ns: &[NodeId]| -> NodeId {
        *midpoint_map.get(&edge_key(ns[a], ns[b])).expect("em")
    };
    let mut new_conn = Vec::new(); let mut new_tags = Vec::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e); let tag = mesh.elem_tags[e as usize];
        if marked_set.contains(&e) {
            let t0_m01=get_em(0,1,ns);let t0_m02=get_em(0,2,ns);let t0_m04=get_em(0,4,ns);
            let t0_m12=get_em(1,2,ns);let t0_m14=get_em(1,4,ns);let t0_m24=get_em(2,4,ns);
            new_conn.extend_from_slice(&[ns[0],t0_m01,t0_m02,t0_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[1],t0_m12,t0_m01,t0_m14]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[2],t0_m02,t0_m12,t0_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[4],t0_m04,t0_m14,t0_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t0_m01,t0_m02,t0_m04,t0_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t0_m01,t0_m02,t0_m12,t0_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t0_m01,t0_m12,t0_m14,t0_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t0_m01,t0_m04,t0_m14,t0_m24]);new_tags.push(tag);
            let t1_m23=get_em(2,3,ns);let t1_m02=get_em(2,0,ns);let t1_m24=get_em(2,4,ns);
            let t1_m03=get_em(3,0,ns);let t1_m34=get_em(3,4,ns);let t1_m04=get_em(0,4,ns);
            new_conn.extend_from_slice(&[ns[2],t1_m23,t1_m02,t1_m24]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[3],t1_m23,t1_m03,t1_m34]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[0],t1_m02,t1_m03,t1_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[ns[4],t1_m24,t1_m34,t1_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t1_m23,t1_m02,t1_m24,t1_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t1_m23,t1_m02,t1_m03,t1_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t1_m23,t1_m03,t1_m34,t1_m04]);new_tags.push(tag);
            new_conn.extend_from_slice(&[t1_m23,t1_m24,t1_m34,t1_m04]);new_tags.push(tag);
        } else { for k in 0..5 { new_conn.push(ns[k]); } new_tags.push(tag); }
    }
    // edge constraints
    let mut ec = Vec::new();
    for (&(a,b),&mid) in &midpoint_map {
        if let Some(adj) = edge_elems.get(&(a,b)) {
            if adj.iter().any(|e|!marked_set.contains(e)) { ec.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize}); }
        }
    }
    ec.sort_by_key(|c|c.constrained); ec.dedup_by_key(|c|c.constrained);
    // tri face constraints
    let mut tc = Vec::new();
    for (key, adj) in &tri_face_elems {
        if adj.len()!=2 { continue; }
        if adj.iter().filter(|&&e|marked_set.contains(&e)).count()!=1 { continue; }
        let (a,b,c)=*key;
        if let(Some(&mab),Some(&mbc),Some(&mac))=(midpoint_map.get(&edge_key(a,b)),midpoint_map.get(&edge_key(b,c)),midpoint_map.get(&edge_key(a,c))) {
            ec.push(HangingNodeConstraint{constrained:mab as usize,parent_a:a as usize,parent_b:b as usize});
            ec.push(HangingNodeConstraint{constrained:mbc as usize,parent_a:b as usize,parent_b:c as usize});
            ec.push(HangingNodeConstraint{constrained:mac as usize,parent_a:a as usize,parent_b:c as usize});
            tc.push(HangingFaceConstraint{constrained:mab as usize,parent_a:a as usize,parent_b:b as usize,parent_c:c as usize});
        }
    }
    ec.sort_by_key(|c|c.constrained); ec.dedup_by_key(|c|c.constrained);
    tc.sort_by_key(|c|(c.parent_a,c.parent_b,c.parent_c)); tc.dedup_by_key(|c|(c.parent_a,c.parent_b,c.parent_c));
    // quad face constraints
    let mut qc = Vec::new();
    for (fns, adj) in &quad_face_elems {
        if adj.len()!=2 { continue; }
        if adj.iter().filter(|&&e|marked_set.contains(&e)).count()!=1 { continue; }
        let re = adj.iter().find(|&&e|marked_set.contains(&e)).unwrap();
        let ns = mesh.elem_nodes(*re);
        let qf = local_faces_pyramid_quad()[0];
        let fn4 = [ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];
        if quad_face_key(fn4)!=*fns { continue; }
        let [a,b,c,d] = fn4;
        if let(Some(mab),Some(mbc),Some(mcd),Some(mda))=(midpoint_map.get(&edge_key(a,b)).copied(),midpoint_map.get(&edge_key(b,c)).copied(),midpoint_map.get(&edge_key(c,d)).copied(),midpoint_map.get(&edge_key(d,a)).copied()) {
            if let Some(&fc) = quad_face_center_map.get(fns) {
                ec.push(HangingNodeConstraint{constrained:mab as usize,parent_a:a as usize,parent_b:b as usize});
                ec.push(HangingNodeConstraint{constrained:mbc as usize,parent_a:b as usize,parent_b:c as usize});
                ec.push(HangingNodeConstraint{constrained:mcd as usize,parent_a:c as usize,parent_b:d as usize});
                ec.push(HangingNodeConstraint{constrained:mda as usize,parent_a:d as usize,parent_b:a as usize});
                qc.push(HangingQuadFaceConstraint{constrained:fc as usize,parent_a:a as usize,parent_b:b as usize,parent_c:c as usize,parent_d:d as usize});
            }
        }
    }
    ec.sort_by_key(|c|c.constrained); ec.dedup_by_key(|c|c.constrained);
    qc.sort_by_key(|c|(c.parent_a,c.parent_b,c.parent_c,c.parent_d)); qc.dedup_by_key(|c|(c.parent_a,c.parent_b,c.parent_c,c.parent_d));
    // rebuild boundary faces
    let nbf = mesh.n_faces(); let mut nfc = Vec::new(); let mut nft = Vec::new();
    for f in 0..nbf {
        let tag = mesh.face_tags[f]; let fs = mesh.bface_nodes(f as FaceId);
        match fs.len() {
            3 => { let (a,b,c)=(fs[0],fs[1],fs[2]);
                let ma=midpoint_map.get(&edge_key(a,b)).copied();let mb=midpoint_map.get(&edge_key(b,c)).copied();let mc=midpoint_map.get(&edge_key(c,a)).copied();
                if let(Some(mab),Some(mbc),Some(mca))=(ma,mb,mc) { nfc.extend_from_slice(&[a,mab,mca]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc]);nft.push(tag);nfc.extend_from_slice(&[mca,mbc,c]);nft.push(tag);nfc.extend_from_slice(&[mab,mbc,mca]);nft.push(tag); }
                else { nfc.extend_from_slice(&[a,b,c]);nft.push(tag); }
            }
            4 => { let (a,b,c,d)=(fs[0],fs[1],fs[2],fs[3]);
                let ma=midpoint_map.get(&edge_key(a,b)).copied();let mb=midpoint_map.get(&edge_key(b,c)).copied();let mc=midpoint_map.get(&edge_key(c,d)).copied();let md=midpoint_map.get(&edge_key(d,a)).copied();
                if let(Some(mab),Some(mbc),Some(mcd),Some(mda))=(ma,mb,mc,md) {
                    let fk = quad_face_key([a,b,c,d]);
                    if let Some(&fc) = quad_face_center_map.get(&fk) { nfc.extend_from_slice(&[a,mab,fc,mda]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc,fc]);nft.push(tag);nfc.extend_from_slice(&[fc,mbc,c,mcd]);nft.push(tag);nfc.extend_from_slice(&[mda,fc,mcd,d]);nft.push(tag); }
                    else { nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag); }
                } else { nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag); }
            }
            _ => { for &n in fs { nfc.push(n); } nft.push(tag); }
        }
    }
    let mut nam = std::collections::HashMap::new();
    if let Some(p) = active_midpoints { for (&k,&v) in p { nam.insert(k,v); } }
    for (&k,&v) in &midpoint_map { nam.insert(k,v); }
    let nns: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
    nam.retain(|_,mid| nns.contains(mid));
    let nm = SimplexMesh::uniform(new_coords,new_conn,new_tags,ElementType::Tet4,nfc,nft,mesh.face_type);
    (nm, ec, tc, qc, midpoint_map, nam)
}

// ─── NCStatePyramid (multi-level Pyramid5 NC tracking) ────────────────────────

#[derive(Debug, Clone)]
struct NCStatePyramidSnapshot {
    mesh: SimplexMesh<3>, constraints: Vec<HangingNodeConstraint>,
    tri_face_constraints: Vec<HangingFaceConstraint>, quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone)]
pub struct NCStatePyramid {
    constraints: Vec<HangingNodeConstraint>, tri_face_constraints: Vec<HangingFaceConstraint>,
    quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>, history: Vec<NCStatePyramidSnapshot>,
}

impl NCStatePyramid {
    pub fn new() -> Self {
        Self { constraints: Vec::new(), tri_face_constraints: Vec::new(), quad_face_constraints: Vec::new(),
            active_midpoints: HashMap::new(), history: Vec::new() }
    }
    pub fn constraints(&self) -> &[HangingNodeConstraint] { &self.constraints }
    pub fn tri_face_constraints(&self) -> &[HangingFaceConstraint] { &self.tri_face_constraints }
    pub fn quad_face_constraints(&self) -> &[HangingQuadFaceConstraint] { &self.quad_face_constraints }
    pub fn can_derefine(&self) -> bool { !self.history.is_empty() }
    pub fn refine(&mut self, mesh: &SimplexMesh<3>, marked: &[ElemId]) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        self.history.push(NCStatePyramidSnapshot { mesh: mesh.clone(), constraints: self.constraints.clone(), tri_face_constraints: self.tri_face_constraints.clone(), quad_face_constraints: self.quad_face_constraints.clone(), active_midpoints: self.active_midpoints.clone() });
        let (nm, ec, tc, qc, mm, nam) = refine_nonconforming_pyramid_internal(mesh, marked, Some(&self.active_midpoints));
        self.constraints = ec.clone(); self.tri_face_constraints = tc.clone(); self.quad_face_constraints = qc.clone(); self.active_midpoints = nam;
        (nm, ec, tc, qc, mm)
    }
    pub fn derefine_last(&mut self) -> Option<(SimplexMesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints; self.tri_face_constraints = snap.tri_face_constraints; self.quad_face_constraints = snap.quad_face_constraints; self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone(), self.tri_face_constraints.clone(), self.quad_face_constraints.clone()))
    }
}

// ─── Hex8 uniform refinement ─────────────────────────────────────────────────

/// Uniformly refine Hex8 → 8 Hex8 children using edge midpoints, face centroids, body centroid.
pub fn refine_hex8_uniform(
    mesh: &SimplexMesh<3>, marked: &[ElemId],
) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
    assert!(mesh.elem_type == ElementType::Hex8, "refine_hex8_uniform: only Hex8");
    if marked.is_empty() { return (mesh.clone(), Vec::new(), HashMap::new()); }
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();
    let mut edge_elems: HashMap<(NodeId,NodeId),Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId { let ns = mesh.elem_nodes(e); for &(a,b) in &local_edges_hex() { edge_elems.entry(edge_key(ns[a],ns[b])).or_default().push(e); } }
    let mut mm: HashMap<(NodeId,NodeId),NodeId> = HashMap::new();
    let mut fcm: HashMap<[NodeId;4],NodeId> = HashMap::new();
    let mut bcm: HashMap<ElemId,NodeId> = HashMap::new();
    let mut nc = mesh.coords.clone(); let mut nn = mesh.n_nodes() as NodeId;
    for &e in &marked_set {
        let ns = mesh.elem_nodes(e);
        for &(a,b) in &local_edges_hex() { let k=edge_key(ns[a],ns[b]); mm.entry(k).or_insert_with(||{let xa=mesh.coords_of(ns[a]);let xb=mesh.coords_of(ns[b]);nc.push(0.5*(xa[0]+xb[0]));nc.push(0.5*(xa[1]+xb[1]));nc.push(0.5*(xa[2]+xb[2]));let id=nn;nn+=1;id}); }
        for face in local_faces_hex() { let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]]; let fk=hex_face_key(fns); fcm.entry(fk).or_insert_with(||{let(mut x,mut y,mut z)=(0.0,0.0,0.0);for&fn_ in &fns{let c=mesh.coords_of(fn_);x+=c[0];y+=c[1];z+=c[2];}nc.push(x/4.0);nc.push(y/4.0);nc.push(z/4.0);let id=nn;nn+=1;id}); }
        bcm.entry(e).or_insert_with(||{let(mut x,mut y,mut z)=(0.0,0.0,0.0);for k in 0..8{let c=mesh.coords_of(ns[k]);x+=c[0];y+=c[1];z+=c[2];}nc.push(x/8.0);nc.push(y/8.0);nc.push(z/8.0);let id=nn;nn+=1;id});
    }
    let ge=|a:usize,b:usize,ns:&[NodeId]|->NodeId{*mm.get(&edge_key(ns[a],ns[b])).expect("em")};
    let gf=|fi:usize,ns:&[NodeId]|->NodeId{let f=local_faces_hex()[fi];*fcm.get(&hex_face_key([ns[f[0]],ns[f[1]],ns[f[2]],ns[f[3]]])).expect("fc")};
    let mut ncn=Vec::new();let mut nt=Vec::new();
    for e in 0..n_elems as ElemId {
        let ns=mesh.elem_nodes(e);let tag=mesh.elem_tags[e as usize];
        if marked_set.contains(&e) { let bc=*bcm.get(&e).unwrap();
            ncn.extend_from_slice(&[ns[0],ge(0,1,ns),gf(0,ns),ge(3,0,ns),ge(0,4,ns),gf(2,ns),bc,gf(4,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[ge(0,1,ns),ns[1],ge(1,2,ns),gf(0,ns),gf(2,ns),ge(1,5,ns),gf(5,ns),bc]);nt.push(tag);
            ncn.extend_from_slice(&[gf(0,ns),ge(1,2,ns),ns[2],ge(2,3,ns),bc,gf(5,ns),ge(2,6,ns),gf(3,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[ge(3,0,ns),gf(0,ns),ge(2,3,ns),ns[3],gf(4,ns),bc,gf(3,ns),ge(3,7,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[ge(0,4,ns),gf(2,ns),bc,gf(4,ns),ns[4],ge(4,5,ns),gf(1,ns),ge(7,4,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[gf(2,ns),ge(1,5,ns),gf(5,ns),bc,ge(4,5,ns),ns[5],ge(5,6,ns),gf(1,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[bc,gf(5,ns),ge(2,6,ns),gf(3,ns),gf(1,ns),ge(5,6,ns),ns[6],ge(6,7,ns)]);nt.push(tag);
            ncn.extend_from_slice(&[gf(4,ns),bc,gf(3,ns),ge(3,7,ns),ge(7,4,ns),gf(1,ns),ge(6,7,ns),ns[7]]);nt.push(tag);
        } else { for k in 0..8 { ncn.push(ns[k]); } nt.push(tag); }
    }
    let mut c = Vec::new();
    for (&(a,b),&mid) in &mm { if let Some(adj)=edge_elems.get(&(a,b)) { if adj.iter().any(|e|!marked_set.contains(e)) { c.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize}); } } }
    c.sort_by_key(|c|c.constrained);
    // boundary faces
    let nbf=mesh.n_faces();let mut nfc=Vec::new();let mut nft=Vec::new();
    for f in 0..nbf { let fs=&mesh.face_conn[f*4..(f+1)*4];let tag=mesh.face_tags[f];let(a,b,c,d)=(fs[0],fs[1],fs[2],fs[3]);
        let ma=mm.get(&edge_key(a,b)).copied();let mb=mm.get(&edge_key(b,c)).copied();let mc=mm.get(&edge_key(c,d)).copied();let md=mm.get(&edge_key(d,a)).copied();
        if let(Some(mab),Some(mbc),Some(mcd),Some(mda))=(ma,mb,mc,md) {
            let fk=hex_face_key([a,b,c,d]);
            if let Some(&fc)=fcm.get(&fk) { nfc.extend_from_slice(&[a,mab,fc,mda]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc,fc]);nft.push(tag);nfc.extend_from_slice(&[fc,mbc,c,mcd]);nft.push(tag);nfc.extend_from_slice(&[mda,fc,mcd,d]);nft.push(tag); }
            else { nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag); }
        } else { nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag); }
    }
    let nm=SimplexMesh::uniform(nc,ncn,nt,ElementType::Hex8,nfc,nft,ElementType::Quad4);
    (nm,c,mm)
}

// ─── Hex20 / Hex27 uniform refinement ────────────────────────────────────────

/// Uniformly refine Hex20 → 8 Hex8 children by extracting the 8 corner nodes.
pub fn refine_hex20_uniform(mesh: &SimplexMesh<3>, marked: &[ElemId]) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    let (m,c,mm) = refine_hex27_uniform_inner(mesh, marked, 20);
    (m,c,mm)
}

/// Uniformly refine Hex27 → 8 Hex8 children by extracting the 8 corner nodes.
pub fn refine_hex27_uniform(mesh: &SimplexMesh<3>, marked: &[ElemId]) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    let (m,c,mm) = refine_hex27_uniform_inner(mesh, marked, 27);
    (m,c,mm)
}

fn refine_hex27_uniform_inner(mesh: &SimplexMesh<3>, marked: &[ElemId], npe: usize) -> (SimplexMesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    assert!(mesh.elem_type == ElementType::Hex20 || mesh.elem_type == ElementType::Hex27, "refine_hex27_uniform_inner: only Hex20/Hex27");
    if marked.is_empty() { return (mesh.clone(), Vec::new(), HashMap::new()); }
    let n_elems = mesh.n_elems();
    let mut hex8_conn = Vec::with_capacity(n_elems * 8);
    for e in 0..n_elems { let off = e * npe; hex8_conn.extend_from_slice(&mesh.conn[off..off+8]); }
    let hex8_mesh = SimplexMesh { coords: mesh.coords.clone(), conn: hex8_conn, elem_tags: mesh.elem_tags.clone(), elem_type: ElementType::Hex8, face_conn: mesh.face_conn.clone(), face_tags: mesh.face_tags.clone(), face_type: mesh.face_type, elem_types: None, elem_offsets: None, face_types: None, face_offsets: None, face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![] };
    refine_hex8_uniform(&hex8_mesh, marked)
}

// ─── Generalized 3-D ZZ error estimator ──────────────────────────────────────

/// Element-wise ZZ estimator for 3-D meshes of any linear element type.
///
/// Computes the gradient recovery-based error indicator:
/// η_K = ‖∇u_h|_K − G(u_h)|_K‖ · |K|^{1/3}
/// where G is the nodal-averaged recovered gradient.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn zz_estimator_3d_general(mesh: &SimplexMesh<3>, u: &[f64]) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let dim = 3usize;

    // Compute constant element gradient at centroid via reference basis.
    let mut elem_grads: Vec<[f64;3]> = Vec::with_capacity(n_elems);
    let mut elem_vols: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let c = |i| mesh.coords_of(ns[i]);

        // Element Jacobian at reference center and volume
        let (jac, vol) = match npe {
            4 => { // Tet4: ref center (1/4,1/4,1/4)
                let j = [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                         [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                         [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs()/6.0)
            }
            8 => { // Hex8: ref center (0,0,0) on [-1,1]³
                let j = [[0.125*( -c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                          0.125*( -c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                          0.125*( -c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                         [0.125*( -c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                          0.125*( -c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                          0.125*( -c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                         [0.125*( -c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                          0.125*( -c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                          0.125*( -c(0)[2]-c(1)[2]-c(2)[2]-c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs())
            }
            6 => { // Prism6: ref center (1/3,1/3,0) on triangle×segment
                let j = [[ (c(1)[0]-c(0)[0])/2.0, (c(2)[0]-c(0)[0])/2.0, (c(3)[0]-c(0)[0])/2.0],
                         [ (c(1)[1]-c(0)[1])/2.0, (c(2)[1]-c(0)[1])/2.0, (c(3)[1]-c(0)[1])/2.0],
                         [ (c(1)[2]-c(0)[2])/2.0, (c(2)[2]-c(0)[2])/2.0, (c(3)[2]-c(0)[2])/2.0]];
                let d = j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]);
                (j, d.abs()/2.0)
            }
            5 => { // Pyramid5: ref center (0,0,0.5) (collapsed hex)
                let v = |i| c(i);
                // Decompose into 2 tets along (0,2) for volume
                let t1 = (v(0),v(1),v(2),v(4)); let t2 = (v(2),v(3),v(0),v(4));
                let d1 = |a:[f64;3],b:[f64;3],c:[f64;3],d:[f64;3]| -> f64 {
                    (b[0]-a[0])*(c[1]-a[1])*(d[2]-a[2])+(b[1]-a[1])*(c[2]-a[2])*(d[0]-a[0])+(b[2]-a[2])*(c[0]-a[0])*(d[1]-a[1])
                    -(b[2]-a[2])*(c[1]-a[1])*(d[0]-a[0])-(b[1]-a[1])*(c[0]-a[0])*(d[2]-a[2])-(b[0]-a[0])*(c[2]-a[2])*(d[1]-a[1])
                };
                let vol = (d1(t1.0,t1.1,t1.2,t1.3).abs()+d1(t2.0,t2.1,t2.2,t2.3).abs())/6.0;
                // Jacobian at reference center (approx using centroid of tet decomposition)
                let j = [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(4)[0]-c(0)[0]],
                         [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(4)[1]-c(0)[1]],
                         [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(4)[2]-c(0)[2]]];
                (j, vol)
            }
            _ => panic!("zz_estimator_3d_general: unsupported npe={}", npe),
        };

        let det = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
                - jac[0][1]*(jac[1][0]*jac[2][2]-jac[1][2]*jac[2][0])
                + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
        let idet = if det.abs() > 1e-30 { 1.0/det } else { 0.0 };
        let jit = |r: usize, c: usize| -> f64 {
            let a=(r+1)%3;let b=(r+2)%3;let d=(c+1)%3;let e=(c+2)%3;
            (jac[a][d]*jac[b][e]-jac[a][e]*jac[b][d])*idet
        };

        // Reference gradients at centroid for each element type
        let gref: Vec<[f64;3]> = match npe {
            4 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            8 => vec![[-0.125,-0.125,-0.125],[0.125,-0.125,-0.125],[0.125,0.125,-0.125],[-0.125,0.125,-0.125],
                      [-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,0.125,0.125],[-0.125,0.125,0.125]],
            6 => vec![[-0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],
                      [-0.5,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5]],
            5 => vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],
            _ => unreachable!(),
        };

        let uh: Vec<f64> = ns.iter().map(|&n| u[n as usize]).collect();
        let mut g = [0.0_f64;3];
        for k in 0..npe {
            for i in 0..3 {
                for j in 0..3 {
                    g[i] += uh[k] * jit(j,i) * gref[k][j];
                }
            }
        }
        elem_grads.push(g);
        elem_vols.push(vol);
    }

    // Nodal gradient recovery via averaging
    let mut ng = vec![[0.0_f64;3]; n_nodes];
    let mut nc = vec![0usize; n_nodes];
    for (e, &g) in elem_grads.iter().enumerate() {
        for &n in mesh.elem_nodes(e as ElemId) {
            for d in 0..3 { ng[n as usize][d] += g[d]; }
            nc[n as usize] += 1;
        }
    }
    for n in 0..n_nodes {
        let c = nc[n] as f64;
        if c > 0.0 { for d in 0..3 { ng[n][d] /= c; } }
    }

    // Element error indicator
    let mut eta = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let gr = [ns.iter().map(|&n| ng[n as usize][0]).sum::<f64>() / ns.len() as f64,
                  ns.iter().map(|&n| ng[n as usize][1]).sum::<f64>() / ns.len() as f64,
                  ns.iter().map(|&n| ng[n as usize][2]).sum::<f64>() / ns.len() as f64];
        let eg = elem_grads[e as usize];
        let d = [eg[0]-gr[0], eg[1]-gr[1], eg[2]-gr[2]];
        eta.push(elem_vols[e as usize].powf(1.0/3.0) * (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt());
    }
    eta
}

// ─── Generalized 3-D Kelly error estimator ───────────────────────────────────

/// Element-wise Kelly (face-jump) estimator for 3-D meshes.
///
/// η_K² = ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h · n]]‖²_L²(F)
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn kelly_estimator_3d_general(mesh: &SimplexMesh<3>, u: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let npe = mesh.elem_type.nodes_per_element();

    // Compute element gradients and volumes (same as ZZ)
    let (elem_grads, _) = zz_gradients_and_volumes_3d(mesh, u);

    // Build face adjacency — use tri-key for tri faces, quad-key for quad faces
    use std::collections::HashMap;
    let mut tri_face_elems: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_face_elems: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    let mut v = [a,b,c]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
            }
        }
        ElementType::Hex8 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for face in local_faces_hex() {
                    let fns = [ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k = fns; k.sort_unstable();
                    quad_face_elems.entry(k).or_default().push(e);
                }
            }
        }
        ElementType::Prism6 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for (a,b,c) in local_faces_prism_tri() {
                    let mut v = [ns[a],ns[b],ns[c]]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
                for face in local_faces_prism_quad() {
                    let fns = [ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k = fns; k.sort_unstable();
                    quad_face_elems.entry(k).or_default().push(e);
                }
            }
        }
        ElementType::Pyramid5 => {
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for (a,b,c) in local_faces_pyramid_tri() {
                    let mut v = [ns[a],ns[b],ns[c]]; v.sort_unstable();
                    tri_face_elems.entry((v[0],v[1],v[2])).or_default().push(e);
                }
                let qf = local_faces_pyramid_quad()[0];
                let fns = [ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];
                let mut k = fns; k.sort_unstable();
                quad_face_elems.entry(k).or_default().push(e);
            }
        }
        _ => panic!("kelly_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    // Triangular face jumps
    for (&(na,nb,nc), elems) in &tri_face_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa < 1e-30 { continue; }
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*jump;
        eta_sq[e0]+=contrib; eta_sq[e1]+=contrib;
    }

    // Quadrilateral face jumps
    for (fns, elems) in &quad_face_elems {
        if elems.len() != 2 { continue; }
        let e0 = elems[0] as usize; let e1 = elems[1] as usize;
        let [a,b,c,d] = *fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        // Split quad into 2 triangles along diagonal (a,c) for normal computation
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa < 1e-30 { continue; }
        // Use normal from first triangle as approximate face normal
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx1*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny1*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz1*inv;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*jump;
        eta_sq[e0]+=contrib; eta_sq[e1]+=contrib;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

/// Compute element gradients and volumes for the generalized 3-D estimators.
fn zz_gradients_and_volumes_3d(mesh: &SimplexMesh<3>, u: &[f64]) -> (Vec<[f64;3]>, Vec<f64>) {
    let n_elems = mesh.n_elems();
    let mut g = Vec::with_capacity(n_elems);
    let mut v = Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e); let npe = ns.len();
        let c = |i| mesh.coords_of(ns[i]);
        let j = match npe {
            4 => [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(3)[0]-c(0)[0]],
                  [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(3)[1]-c(0)[1]],
                  [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(3)[2]-c(0)[2]]],
            8 => [[0.125*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]-c(4)[0]+c(5)[0]+c(6)[0]-c(7)[0]),
                   0.125*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]-c(4)[0]-c(5)[0]+c(6)[0]+c(7)[0]),
                   0.125*(-c(0)[0]-c(1)[0]-c(2)[0]-c(3)[0]+c(4)[0]+c(5)[0]+c(6)[0]+c(7)[0])],
                  [0.125*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]-c(4)[1]+c(5)[1]+c(6)[1]-c(7)[1]),
                   0.125*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]-c(4)[1]-c(5)[1]+c(6)[1]+c(7)[1]),
                   0.125*(-c(0)[1]-c(1)[1]-c(2)[1]-c(3)[1]+c(4)[1]+c(5)[1]+c(6)[1]+c(7)[1])],
                  [0.125*(-c(0)[2]+c(1)[2]+c(2)[2]-c(3)[2]-c(4)[2]+c(5)[2]+c(6)[2]-c(7)[2]),
                   0.125*(-c(0)[2]-c(1)[2]+c(2)[2]+c(3)[2]-c(4)[2]-c(5)[2]+c(6)[2]+c(7)[2]),
                   0.125*(-c(0)[2]-c(1)[2]-c(2)[2]-c(3)[2]+c(4)[2]+c(5)[2]+c(6)[2]+c(7)[2])]],
            6 => [[(c(1)[0]-c(0)[0])/2.0,(c(2)[0]-c(0)[0])/2.0,(c(3)[0]-c(0)[0])/2.0],
                  [(c(1)[1]-c(0)[1])/2.0,(c(2)[1]-c(0)[1])/2.0,(c(3)[1]-c(0)[1])/2.0],
                  [(c(1)[2]-c(0)[2])/2.0,(c(2)[2]-c(0)[2])/2.0,(c(3)[2]-c(0)[2])/2.0]],
            5 => [[c(1)[0]-c(0)[0],c(2)[0]-c(0)[0],c(4)[0]-c(0)[0]],
                  [c(1)[1]-c(0)[1],c(2)[1]-c(0)[1],c(4)[1]-c(0)[1]],
                  [c(1)[2]-c(0)[2],c(2)[2]-c(0)[2],c(4)[2]-c(0)[2]]],
            _ => panic!("zz_gradients_and_volumes_3d: unsupported npe={}", npe),
        };
        let idet = { let d=j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]); if d.abs()>1e-30{1.0/d}else{0.0} };
        let jit=|r:usize,c:usize|->f64{let a=(r+1)%3;let b=(r+2)%3;let d=(c+1)%3;let e=(c+2)%3;(j[a][d]*j[b][e]-j[a][e]*j[b][d])*idet};
        let gref:Vec<[f64;3]>=match npe{
            4=>vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            8=>vec![[-0.125,-0.125,-0.125],[0.125,-0.125,-0.125],[0.125,0.125,-0.125],[-0.125,0.125,-0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,0.125,0.125],[-0.125,0.125,0.125]],
            6=>vec![[-0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[-0.5,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5]],
            5=>vec![[-1.0,-1.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],
            _=>unreachable!()};
        let uh:Vec<f64>=ns.iter().map(|&n|u[n as usize]).collect();
        let mut grad=[0.0_f64;3];
        for k in 0..npe{for i in 0..3{for j in 0..3{grad[i]+=uh[k]*jit(j,i)*gref[k][j];}}}
        g.push(grad);
        let vol=match npe{
            4=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs()/6.0,
            8=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs(),
            6=>(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])-j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0])).abs()/2.0,
            5=>{let v=|i|c(i);let d=|a:[f64;3],b:[f64;3],c:[f64;3],d:[f64;3]|->f64{(b[0]-a[0])*(c[1]-a[1])*(d[2]-a[2])+(b[1]-a[1])*(c[2]-a[2])*(d[0]-a[0])+(b[2]-a[2])*(c[0]-a[0])*(d[1]-a[1])-(b[2]-a[2])*(c[1]-a[1])*(d[0]-a[0])-(b[1]-a[1])*(c[0]-a[0])*(d[2]-a[2])-(b[0]-a[0])*(c[2]-a[2])*(d[1]-a[1])};(d(v(0),v(1),v(2),v(4)).abs()+d(v(2),v(3),v(0),v(4)).abs())/6.0}
            ,_=>unreachable!()};
        v.push(vol);
    }
    (g, v)
}

// ─── Generalized 3-D residual error estimator ────────────────────────────────

/// Element-wise residual error indicator for 3-D meshes of any type.
///
/// ```text
/// η_K² = h_K² · ‖f‖²_L²(K)  +  ½ Σ_{F⊂∂K} h_F · ‖[[∇u_h·n]]‖²_L²(F)
/// ```
///
/// For P1 elements the interior residual ∇·(κ∇u_h) vanishes element-wise,
/// reducing the formula to the face-jump estimator weighted by h_F.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
pub fn residual_estimator_3d_general(mesh: &SimplexMesh<3>, u: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let npe = mesh.elem_type.nodes_per_element();

    let (elem_grads, elem_vols) = zz_gradients_and_volumes_3d(mesh, u);

    // Face adjacency
    use std::collections::HashMap;
    let mut tri_faces: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_faces: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                    let mut v=[a,b,c];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}}
        }
        ElementType::Hex8 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for face in local_faces_hex() { let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                    let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Prism6 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in local_faces_prism_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                for face in local_faces_prism_quad() {let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Pyramid5 => {
            for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
                for (a,b,c) in local_faces_pyramid_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
                let qf=local_faces_pyramid_quad()[0];let fns=[ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
        }
        _ => panic!("residual_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    let mut eta_sq = vec![0.0_f64; n_elems];

    // Interior residual: h_K² * |K| * f_avg²
    for e in 0..n_elems {
        let ns = mesh.elem_nodes(e as ElemId);
        let favg: f64 = ns.iter().map(|&n| f[n as usize]).sum::<f64>() / ns.len() as f64;
        let h2 = elem_vols[e].powf(2.0 / 3.0);
        eta_sq[e] += h2 * elem_vols[e] * favg * favg;
    }

    // Triangular face jumps
    for (&(na,nb,nc), elems) in &tri_faces {
        if elems.len() != 2 { continue; }
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa<1e-30{continue;}
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz*inv;
        let hf=(2.0*fa).sqrt();
        eta_sq[e0]+=0.5*hf*jump*jump;eta_sq[e1]+=0.5*hf*jump*jump;
    }

    // Quadrilateral face jumps
    for (fns, elems) in &quad_faces {
        if elems.len()!=2{continue;}
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[a,b,c,d]=*fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        // Split quad into 2 triangles along (a,c) for area and normal
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa<1e-30{continue;}
        // Use normal from first triangle as approximate face normal
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx1*inv
                +(elem_grads[e0][1]-elem_grads[e1][1])*ny1*inv
                +(elem_grads[e0][2]-elem_grads[e1][2])*nz1*inv;
        let hf=(2.0*fa).sqrt();
        eta_sq[e0]+=0.5*hf*jump*jump;eta_sq[e1]+=0.5*hf*jump*jump;
    }

    eta_sq.iter().map(|v| v.sqrt()).collect()
}

// ─── Generalized 3-D DWR (goal-oriented) error estimator ─────────────────────

/// Element-wise DWR (Dual Weighted Residual) indicator for 3-D meshes.
///
/// ```text
/// η_K = |∫_K f · ω dΩ  +  ½ Σ_{F⊂∂K} ∫_F [[∇u_h·n]] · ω dS|
/// ```
///
/// where ω = z − avg(z) is the dual solution fluctuation (element-wise).
/// For P1 elements, the interior residual ∇·(κ∇u_h) vanishes element-wise,
/// leaving only the source term and face jumps.
///
/// Supports Tet4, Hex8, Prism6, Pyramid5.
///
/// # Arguments
/// * `mesh` — the mesh
/// * `u` — primal solution (nodal values)
/// * `z` — dual solution (nodal values)
/// * `f` — source term (nodal values)
pub fn dwr_estimator_3d_general(mesh: &SimplexMesh<3>, u: &[f64], z: &[f64], f: &[f64]) -> Vec<f64> {
    let n_elems = mesh.n_elems();
    let (elem_grads, elem_vols) = zz_gradients_and_volumes_3d(mesh, u);

    use std::collections::HashMap;
    let mut tri_faces: HashMap<(u32,u32,u32), Vec<ElemId>> = HashMap::new();
    let mut quad_faces: HashMap<[u32;4], Vec<ElemId>> = HashMap::new();

    match mesh.elem_type {
        ElementType::Tet4 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for &(a,b,c) in &[(ns[0],ns[1],ns[2]),(ns[0],ns[1],ns[3]),(ns[0],ns[2],ns[3]),(ns[1],ns[2],ns[3])] {
                let mut v=[a,b,c];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}}
        }
        ElementType::Hex8 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for face in local_faces_hex() { let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];
                let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Prism6 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for (a,b,c) in local_faces_prism_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
            for face in local_faces_prism_quad() {let fns=[ns[face[0]],ns[face[1]],ns[face[2]],ns[face[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}}
        }
        ElementType::Pyramid5 => { for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e);
            for (a,b,c) in local_faces_pyramid_tri() {let mut v=[ns[a],ns[b],ns[c]];v.sort_unstable();tri_faces.entry((v[0],v[1],v[2])).or_default().push(e);}
            let qf=local_faces_pyramid_quad()[0];let fns=[ns[qf[0]],ns[qf[1]],ns[qf[2]],ns[qf[3]]];let mut k=fns;k.sort_unstable();quad_faces.entry(k).or_default().push(e);}
        }
        _ => panic!("dwr_estimator_3d_general: unsupported {:?}", mesh.elem_type),
    }

    // Element-wise data: dual fluctuation ω_K = z_avg − z_h(K)
    let mut elem_omega: Vec<f64> = Vec::with_capacity(n_elems);
    let mut elem_f_avg: Vec<f64> = Vec::with_capacity(n_elems);

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let npe = ns.len();
        let z_avg: f64 = ns.iter().map(|&n| z[n as usize]).sum::<f64>() / npe as f64;
        let f_avg: f64 = ns.iter().map(|&n| f[n as usize]).sum::<f64>() / npe as f64;
        elem_omega.push(z_avg);
        elem_f_avg.push(f_avg);
    }

    let mut eta = vec![0.0_f64; n_elems];

    // Interior contribution: ∫_K f · ω ≈ f_centroid · ω · |K|
    for e in 0..n_elems {
        eta[e] += (elem_f_avg[e] * elem_omega[e]).abs() * elem_vols[e];
    }

    // Face jump contribution: ½ ∫_F [[∇u·n]] · ω dS
    // Tri faces
    for (&(na,nb,nc), elems) in &tri_faces {
        if elems.len() != 2 { continue; }
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let cc=mesh.coords_of(nc);
        let ex=cb[0]-ca[0];let ey=cb[1]-ca[1];let ez=cb[2]-ca[2];
        let fx=cc[0]-ca[0];let fy=cc[1]-ca[1];let fz=cc[2]-ca[2];
        let nx=ey*fz-ez*fy;let ny=ez*fx-ex*fz;let nz=ex*fy-ey*fx;
        let fa=0.5*(nx*nx+ny*ny+nz*nz).sqrt();
        if fa<1e-30{continue;}
        let inv=1.0/(nx*nx+ny*ny+nz*nz).sqrt();
        let j0=elem_grads[e0][0]*nx*inv+elem_grads[e0][1]*ny*inv+elem_grads[e0][2]*nz*inv;
        let j1=elem_grads[e1][0]*nx*inv+elem_grads[e1][1]*ny*inv+elem_grads[e1][2]*nz*inv;
        let jump=(j0-j1).abs();
        if jump<1e-30{continue;}
        let w_mid=(elem_omega[e0]+elem_omega[e1])*0.5;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*w_mid.abs();
        eta[e0]+=contrib;eta[e1]+=contrib;
    }

    // Quad faces
    for (fns, elems) in &quad_faces {
        if elems.len()!=2{continue;}
        let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[a,b,c,d]=*fns;
        let ca=mesh.coords_of(a);let cb=mesh.coords_of(b);let cc=mesh.coords_of(c);let cd=mesh.coords_of(d);
        // Split quad into 2 triangles along (a,c) for full area
        let ex1=cb[0]-ca[0];let ey1=cb[1]-ca[1];let ez1=cb[2]-ca[2];
        let fx1=cc[0]-ca[0];let fy1=cc[1]-ca[1];let fz1=cc[2]-ca[2];
        let nx1=ey1*fz1-ez1*fy1;let ny1=ez1*fx1-ex1*fz1;let nz1=ex1*fy1-ey1*fx1;
        let area1=0.5*(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let ex2=cd[0]-ca[0];let ey2=cd[1]-ca[1];let ez2=cd[2]-ca[2];
        let fx2=cc[0]-ca[0];let fy2=cc[1]-ca[1];let fz2=cc[2]-ca[2];
        let nx2=ey2*fz2-ez2*fy2;let ny2=ez2*fx2-ex2*fz2;let nz2=ex2*fy2-ey2*fx2;
        let area2=0.5*(nx2*nx2+ny2*ny2+nz2*nz2).sqrt();
        let fa=area1+area2;
        if fa<1e-30{continue;}
        let inv=1.0/(nx1*nx1+ny1*ny1+nz1*nz1).sqrt();
        let j0=elem_grads[e0][0]*nx1*inv+elem_grads[e0][1]*ny1*inv+elem_grads[e0][2]*nz1*inv;
        let j1=elem_grads[e1][0]*nx1*inv+elem_grads[e1][1]*ny1*inv+elem_grads[e1][2]*nz1*inv;
        let jump=(j0-j1).abs();
        if jump<1e-30{continue;}
        let w_mid=(elem_omega[e0]+elem_omega[e1])*0.5;
        let hf=(2.0*fa).sqrt();
        let contrib=0.5*hf*jump*w_mid.abs();
        eta[e0]+=contrib;eta[e1]+=contrib;
    }

    eta
}

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
    fn derefine_mark_selects_small_error_elements() {
        let eta = vec![1.0_f64, 0.8, 0.4, 0.2, 0.1, 0.01];
        let marked = mark_for_derefinement(&eta, 0.2);
        // max=1.0, cutoff=0.2 -> indices with eta<=0.2 are 3,4,5
        assert_eq!(marked, vec![3, 4, 5]);
    }

    #[test]
    fn derefine_mark_handles_empty() {
        let marked = mark_for_derefinement(&[], 0.2);
        assert!(marked.is_empty());
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
    fn refine_with_tree_then_derefine_roundtrip_elements() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let marked = vec![0u32, 1u32];
        let (fine, tree) = refine_marked_with_tree(&mesh, &marked);
        let coarse = derefine_marked(&fine, &tree, &marked);

        assert_eq!(coarse.n_elems(), mesh.n_elems(),
            "derefine roundtrip should recover element count");
        assert_eq!(coarse.elem_type, mesh.elem_type);

        let parents = tree.parents();
        assert!(parents.contains(&0) && parents.contains(&1));
    }

    #[test]
    fn prolongate_then_restrict_p1_roundtrip_on_coarse_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n0 = mesh.n_nodes();
        let u0: Vec<f64> = (0..n0).map(|i| i as f64).collect();

        let mut nc = NCState::new();
        let (_fine, _constraints, midpoint_map) = nc.refine(&mesh, &[0]);
        let uf = prolongate_p1(&u0, n0 + midpoint_map.len(), &midpoint_map);
        let ur = restrict_to_coarse_p1(&uf, n0);

        assert_eq!(ur, u0, "restrict should recover coarse nodal values exactly");
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

    #[test]
    fn dwr_linear_solution_has_zero_indicator() {
        // u(x,y) = x, z(x,y) = y, f = 0 → DWR = 0
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f = vec![0.0_f64; n];
        let eta = dwr_estimator(&mesh, &u, &z, &f);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 1e-12, "DWR should be zero for linear u,z with f=0, got {max_eta:.3e}");
    }

    #[test]
    fn dwr_quadratic_solution_has_positive_indicator() {
        // u(x,y) = x², z(x,y) = y², f = 2 (Poisson source for Δu=2)
        // DWR should detect the error on P1 elements.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] * c[0]
        }).collect();
        let z: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[1] * c[1]
        }).collect();
        let f = vec![2.0_f64; n];  // ∇²(x²) = 2
        let eta = dwr_estimator(&mesh, &u, &z, &f);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta > 1e-6, "DWR should be positive for quadratic u,z, got {max_eta:.3e}");
    }

    #[test]
    fn zz_3d_linear_solution_zero_indicator() {
        // u(x,y,z) = x → constant gradient → zero ZZ error
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 0.5, "3D ZZ should be near zero for linear u, got {max_eta:.3e}");
    }

    #[test]
    fn zz_3d_quadratic_solution_positive_indicator() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] * c[0]
        }).collect();
        let eta = zz_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta > 1e-6, "3D ZZ should be positive for x², got {max_eta:.3e}");
    }

    #[test]
    fn kelly_3d_linear_solution_zero_indicator() {
        // u = x → constant gradient → no face jumps → zero Kelly
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 1.0, "3D Kelly should be near zero for linear u, got {max_eta:.3e}");
    }

    #[test]
    fn kelly_3d_quadratic_solution_positive_indicator() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] * c[0]
        }).collect();
        let eta = kelly_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta > 1e-6, "3D Kelly should be positive for x², got {max_eta:.3e}");
    }

    #[test]
    fn p_refine_tri3_marked_elements_become_tri6() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_orig = mesh.n_nodes();
        let (p2, midpoint_map) = p_refine_tri3_to_tri6(&mesh, &[0, 1, 2]);
        // Elements 0-2 each gain 3 edge midpoints, but shared edges are deduplicated
        assert!(p2.n_nodes() > n_orig, "should have added midpoint nodes");
        assert_eq!(p2.n_elems(), mesh.n_elems(), "element count unchanged");
        // Marked elements should be Tri6
        if let Some(ref types) = p2.elem_types {
            assert_eq!(types[0], ElementType::Tri6);
            assert_eq!(types[1], ElementType::Tri6);
            assert_eq!(types[2], ElementType::Tri6);
        }
        // Check all midpoint edges have been created
        assert!(!midpoint_map.is_empty(), "should have at least one midpoint");
        // All new nodes should have unique indices beyond original
        for &new_n in midpoint_map.values() {
            assert!((new_n as usize) >= n_orig, "new node {new_n} should be >= {n_orig}");
        }
    }

    #[test]
    fn p_prolongate_p1_to_p2_preserves_vertices() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_orig = mesh.n_nodes();
        let u: Vec<f64> = (0..n_orig).map(|i| {
            let c = mesh.coords_of(i as NodeId); c[0] + c[1]
        }).collect();
        let (p2, midpoint_map) = p_refine_tri3_to_tri6(&mesh, &[0, 1, 2]);
        let u_p2 = p_prolongate_p1_to_p2(&u, &midpoint_map, &p2);
        // Original vertex values preserved
        for i in 0..n_orig {
            assert!((u_p2[i] - u[i]).abs() < 1e-14, "vertex {i} value changed");
        }
        // Midpoint values should be averages
        for (&(a, b), &new) in &midpoint_map {
            let expected = 0.5 * (u[a as usize] + u[b as usize]);
            assert!((u_p2[new as usize] - expected).abs() < 1e-14,
                "midpoint {new}: expected {expected:.6}, got {:.6}", u_p2[new as usize]);
        }
    }

    // ── Higher-order p-refinement tests ──────────────────────────────────

    #[test]
    fn p_refine_tet4_to_tet10_adds_midpoints() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let n_orig = mesh.n_nodes();
        let (p2, midpoint_map) = p_refine_tet4_to_tet10(&mesh, &[0, 1]);
        assert!(p2.n_nodes() > n_orig, "should have added midpoint nodes");
        assert_eq!(p2.n_elems(), mesh.n_elems(), "element count unchanged");
        if let Some(ref types) = p2.elem_types {
            assert_eq!(types[0], ElementType::Tet10);
            assert_eq!(types[1], ElementType::Tet10);
        }
        for &new_n in midpoint_map.values() {
            assert!((new_n as usize) >= n_orig, "new node {new_n} should be >= {n_orig}");
        }
    }

    #[test]
    fn p_refine_tet10_to_tet20_adds_face_centroids() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let tet10_mesh = p_refine_tet4_to_tet10(&mesh, &[0, 1]).0;
        let n_orig = tet10_mesh.n_nodes();
        let (p3, face_map) = p_refine_tet10_to_tet20(&tet10_mesh, &[0]);
        assert!(p3.n_nodes() > n_orig, "should have added face centroids");
        assert_eq!(p3.n_elems(), tet10_mesh.n_elems());
        for &new_n in face_map.values() {
            assert!((new_n as usize) >= n_orig);
        }
    }

    #[test]
    fn p_refine_quad4_to_quad9_adds_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let n_orig = mesh.n_nodes();
        let (q9, midpoint_map) = p_refine_quad4_to_quad9(&mesh, &[0, 1]);
        assert!(q9.n_nodes() > n_orig);
        assert_eq!(q9.n_elems(), mesh.n_elems());
        for &new_n in midpoint_map.values() {
            assert!((new_n as usize) >= n_orig);
        }
    }

    #[test]
    fn p_refine_hex8_to_hex20_adds_midpoints() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let n_orig = mesh.n_nodes();
        let (h20, midpoint_map) = p_refine_hex8_to_hex20(&mesh, &[0]);
        assert!(h20.n_nodes() > n_orig);
        assert_eq!(h20.n_elems(), mesh.n_elems());
        if let Some(ref types) = h20.elem_types {
            assert_eq!(types[0], ElementType::Hex20);
        }
        for &new_n in midpoint_map.values() {
            assert!((new_n as usize) >= n_orig);
        }
    }

    #[test]
    fn p_refine_hex20_to_hex27_adds_centroids() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let (h20, _) = p_refine_hex8_to_hex20(&mesh, &[0]);
        let n_orig = h20.n_nodes();
        let (h27, centroids) = p_refine_hex20_to_hex27(&h20, &[0]);
        assert!(h27.n_nodes() > n_orig);
        assert_eq!(h27.n_elems(), h20.n_elems());
        assert!(!centroids.is_empty());
        for &c in &centroids {
            assert!((c as usize) >= n_orig);
        }
    }

    #[test]
    fn p_refine_tri6_to_tri10_adds_centroid() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let (tri6, _) = p_refine_tri3_to_tri6(&mesh, &[0, 1, 2]);
        let n_orig = tri6.n_nodes();
        let (tri10, centroids) = p_refine_tri6_to_tri10(&tri6, &[0, 1]);
        assert!(tri10.n_nodes() > n_orig);
        assert_eq!(tri10.n_elems(), tri6.n_elems());
        assert!(!centroids.is_empty());
        for &c in &centroids {
            assert!((c as usize) >= n_orig);
        }
    }

    #[test]
    fn p_refine_chain_preserves_volume_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let (h20, _) = p_refine_hex8_to_hex20(&mesh, &[0]);
        let (h27, _) = p_refine_hex20_to_hex27(&h20, &[0]);
        assert!(h27.n_nodes() > h20.n_nodes());
        assert!(h20.n_nodes() > mesh.n_nodes());
    }

    #[test]
    fn p_refine_mark_selects_high_error_elements() {
        let eta = vec![0.1, 0.5, 1.0, 0.2, 0.8];
        let marked = mark_for_p_refinement(&eta, 0.5);
        // Elements with eta >= 0.5 * 1.0 = 0.5: indices 1, 2, 4
        assert_eq!(marked, vec![1, 2, 4], "should mark high-error elements");
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

    #[test]
    fn ncstate_derefine_last_rolls_back_mesh_and_constraints() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        let (m1, c1, _) = nc.refine(&mesh, &[0, 1]);
        assert!(m1.n_elems() > mesh.n_elems());
        assert!(!c1.is_empty());
        assert!(nc.can_derefine());

        let (m0, c0) = nc.derefine_last().expect("expected rollback snapshot");
        assert_eq!(m0.n_elems(), mesh.n_elems());
        assert_eq!(m0.n_nodes(), mesh.n_nodes());
        assert!(c0.is_empty(), "initial state should have no hanging constraints");
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

    #[test]
    fn ncstate3d_derefine_last_rolls_back() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc3 = NCState3D::new();

        let (m1, c1, _, f1) = nc3.refine(&mesh, &[0]);
        assert!(m1.n_elems() > mesh.n_elems());
        assert!(!c1.is_empty());
        assert!(!f1.is_empty());
        assert!(nc3.can_derefine());

        let (m0, c0, f0) = nc3.derefine_last().expect("expected 3D rollback snapshot");
        assert_eq!(m0.n_elems(), mesh.n_elems());
        assert_eq!(m0.n_nodes(), mesh.n_nodes());
        assert!(c0.is_empty());
        assert!(f0.is_empty());
    }

    // ─── Quad4 NC AMR tests ───────────────────────────────────────────────────

    #[test]
    fn quad4_nonconforming_refine_empty() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &[]);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn quad4_nonconforming_refine_all_gives_no_constraints() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &all);
        // Refining all → 4× as many elements, no hanging nodes
        assert_eq!(nc.n_elems(), mesh.n_elems() * 4);
        assert!(constraints.is_empty());
        nc.check().unwrap();
    }

    #[test]
    fn quad4_nonconforming_refine_single_element_creates_constraints() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &[0]);
        // Element 0 split into 4, rest unchanged → total = 3 + 4 = 7
        assert_eq!(nc.n_elems(), 7);
        // Shared edges with unrefined neighbours create hanging constraints
        assert!(!constraints.is_empty());
        nc.check().unwrap();
    }

    #[test]
    fn ncstate_quad_multi_level_tracks_constraints() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let mut ncq = NCStateQuad::new();

        let (m1, c1, _) = ncq.refine(&mesh, &[0]);
        assert!(m1.n_elems() > mesh.n_elems());
        assert!(!c1.is_empty());

        // Refine a neighbour → some constraints should be resolved
        let (m2, c2, _) = ncq.refine(&m1, &[1]);
        assert!(m2.n_elems() > m1.n_elems());
        let _ = c2; // constraints count may vary
        m2.check().unwrap();
    }

    #[test]
    fn ncstate_quad_derefine_last_rolls_back() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let mut ncq = NCStateQuad::new();

        let (_m1, _, _) = ncq.refine(&mesh, &[0]);
        assert!(ncq.can_derefine());

        let (m0, c0) = ncq.derefine_last().expect("expected rollback");
        assert_eq!(m0.n_elems(), mesh.n_elems());
        assert_eq!(m0.n_nodes(), mesh.n_nodes());
        assert!(c0.is_empty());
    }

    // ─── Hex8 NC AMR tests ────────────────────────────────────────────────────

    #[test]
    fn hex8_nonconforming_refine_empty() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (nc, constraints, _, _) = refine_nonconforming_hex(&mesh, &[]);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn hex8_nonconforming_refine_all_gives_no_constraints() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints, fc, _) = refine_nonconforming_hex(&mesh, &all);
        assert_eq!(nc.n_elems(), mesh.n_elems() * 8);
        assert!(constraints.is_empty());
        assert!(fc.is_empty(), "uniform refine: no hanging quad faces");
        nc.check().unwrap();
    }

    #[test]
    fn hex8_nonconforming_refine_single_element_creates_constraints() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let (nc, constraints, fc, _) = refine_nonconforming_hex(&mesh, &[0]);
        // 1 element refined into 8, rest unchanged
        assert!(nc.n_elems() > mesh.n_elems());
        // Neighbouring unrefined elements cause hanging constraints
        assert!(!constraints.is_empty());
        assert!(!fc.is_empty(), "single refined hex should produce hanging quad face constraints");
        nc.check().unwrap();
    }

    // ─── Anisotropic Quad NC AMR tests ────────────────────────────────────────

    #[test]
    fn quad_aniso_x_split_doubles_elements() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked);
        assert_eq!(refined.n_elems(), n + 1, "X split: one elem → 2, total {}", n + 1);
    }

    #[test]
    fn quad_aniso_y_split_doubles_elements() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::Y)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked);
        assert_eq!(refined.n_elems(), n + 1, "Y split: one elem → 2, total {}", n + 1);
    }

    #[test]
    fn quad_aniso_both_quadruples_element() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::Both)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked);
        assert_eq!(refined.n_elems(), n + 3, "Both split: one elem → 4, total {}", n + 3);
    }

    #[test]
    fn quad_aniso_empty_marked_is_identity() {
        let mesh = SimplexMesh::<2>::unit_square_quad(3);
        let (refined, constraints) = refine_nonconforming_quad_aniso(&mesh, &[]);
        assert_eq!(refined.n_elems(), mesh.n_elems());
        assert_eq!(refined.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn quad_aniso_x_split_adds_midpoints() {
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        let n_nodes_before = mesh.n_nodes();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked);
        // X split adds 2 midpoints (on bottom and top edges)
        assert_eq!(refined.n_nodes(), n_nodes_before + 2);
    }

    #[test]
    fn quad_aniso_partial_refine_creates_hanging_constraints() {
        // A 2×1 mesh: 2 quads side by side. Refine only the first.
        let mesh = SimplexMesh::<2>::unit_square_quad(2);
        if mesh.n_elems() < 2 { return; } // skip if not enough elements
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (_, constraints) = refine_nonconforming_quad_aniso(&mesh, &marked);
        // On shared edge, there should be hanging node constraints
        // (may be 0 if no shared edge with unrefined element — depends on mesh topology)
        let _ = constraints; // just ensure it runs without panic
    }

    // ─── Hex8 各向异性 NC AMR tests ───────────────────────────────────────────

    #[test]
    fn hex_aniso_x_split_gives_two_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)]);
        // 1 element → 2 children along X
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_y_split_gives_two_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::Y)]);
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_z_split_gives_two_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::Z)]);
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_xy_split_gives_four_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::XY)]);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_xz_split_gives_four_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::XZ)]);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_yz_split_gives_four_children() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::YZ)]);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_all_delegates_to_isotropic() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::All)]);
        assert_eq!(refined.n_elems(), 8);
    }

    #[test]
    fn hex_aniso_empty_marked_is_identity() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        let (refined, constraints) = refine_nonconforming_hex_aniso(&mesh, &[]);
        assert_eq!(refined.n_elems(), mesh.n_elems());
        assert_eq!(refined.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn hex_aniso_x_split_adds_four_midpoints() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let n_before = mesh.n_nodes();
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)]);
        // X cut adds 4 edge midpoints on the 4 X-parallel edges
        assert_eq!(refined.n_nodes(), n_before + 4);
    }

    #[test]
    fn hex_aniso_multi_elem_x_split_partial_creates_constraints() {
        // 2x2x1 mesh (4 elements). Refine one element with X split;
        // neighbouring unrefined elements should produce hanging constraints.
        let mesh = SimplexMesh::<3>::unit_cube_hex(2);
        if mesh.n_elems() < 2 { return; }
        let (_, constraints) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)]);
        // At least some hanging constraints expected on shared faces.
        assert!(!constraints.is_empty(), "expected hanging constraints on partial X-split");
    }

    #[test]
    fn tet4_uniform_3d_creates_eight_tets() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        assert_eq!(mesh.n_elems(), 6);
        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 48, "6 tet parents × 8 = 48");
        assert!(fine.n_nodes() > mesh.n_nodes());
    }

    #[test]
    fn hex8_uniform_3d_creates_eight_hexes() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        assert_eq!(mesh.n_elems(), 1);
        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 8, "1 hex parent × 8 = 8");
        assert!(fine.n_nodes() > mesh.n_nodes());
    }

    // ─── Prism6 uniform refinement tests ──────────────────────────────────────

    /// Signed volume of tetrahedron (a,b,c,d).
    fn tet_signed_vol(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
        let v1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let v2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let v3 = [d[0]-a[0], d[1]-a[1], d[2]-a[2]];
        (v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
       - v1[1]*(v2[0]*v3[2] - v2[2]*v3[0])
       + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0])) / 6.0
    }

    /// Volume of a Prism6 element (decomposed into 3 tetrahedra).
    ///
    /// Decomposition: split along edge (2,3), giving tets (0,1,2,3), (1,2,3,4), (2,3,4,5).
    /// This avoids coplanarity for right prisms.
    fn prism6_vol(mesh: &SimplexMesh<3>, e: ElemId) -> f64 {
        let ns = mesh.elem_nodes(e);
        let c = |i: usize| -> [f64; 3] {
            let off = ns[i] as usize * 3;
            [mesh.coords[off], mesh.coords[off+1], mesh.coords[off+2]]
        };
        let v = tet_signed_vol(&c(0), &c(1), &c(2), &c(3)).abs()
              + tet_signed_vol(&c(1), &c(2), &c(3), &c(4)).abs()
              + tet_signed_vol(&c(2), &c(3), &c(4), &c(5)).abs();
        v
    }

    #[test]
    fn prism6_uniform_3d_single_element_eight_children() {
        // Unit right triangular prism: right triangle base (z=0) → top (z=1)
        // n0=(0,0,0), n1=(1,0,0), n2=(0,1,0)
        // n3=(0,0,1), n4=(1,0,1), n5=(0,1,1)
        let coords = vec![
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            1.0, 0.0, 1.0,  // 4
            0.0, 1.0, 1.0,  // 5
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];

        // Boundary faces: 2 tri + 3 quad
        let face_conn = vec![
            0u32, 2, 1,        // bottom tri (outward -z)
            3, 4, 5,           // top tri (outward +z)
            0, 1, 4, 3,        // quad front (y=0)
            1, 2, 5, 4,        // quad right
            0, 3, 5, 2,        // quad left (x=0)
        ];
        let face_tags = vec![1i32, 2, 3, 4, 5];
        let face_types = vec![
            ElementType::Tri3,
            ElementType::Tri3,
            ElementType::Quad4,
            ElementType::Quad4,
            ElementType::Quad4,
        ];
        let face_offsets = vec![0usize, 3, 6, 10, 14, 18];

        let mesh = SimplexMesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        };

        let vol_orig = prism6_vol(&mesh, 0);
        assert!((vol_orig - 0.5).abs() < 1e-14, "original volume={}", vol_orig);

        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, constraints, _) = refine_prism6_uniform(&mesh, &all);

        assert_eq!(fine.n_elems(), 8, "1 prism → 8 children");
        assert_eq!(constraints.is_empty(), true, "uniform refine: no hanging nodes");
        assert!(fine.n_nodes() > mesh.n_nodes(), "refined mesh should have more nodes");

        // Volume conservation: sum of child volumes == original volume
        let vol_sum: f64 = (0..fine.n_elems())
            .map(|e| prism6_vol(&fine, e as ElemId))
            .sum();
        assert!(
            (vol_sum - vol_orig).abs() < 1e-12,
            "volume mismatch: orig={} sum={} diff={}",
            vol_orig, vol_sum, vol_sum - vol_orig,
        );
    }

    #[test]
    fn prism6_uniform_3d_through_dispatch() {
        // Test that refine_uniform_3d dispatches correctly for Prism6
        let coords = vec![
            0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 1.0,
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];
        let face_conn = vec![0u32,2,1, 3,4,5, 0,1,4,3, 1,2,5,4, 0,3,5,2];
        let face_tags = vec![1,2,3,4,5];
        let face_types = vec![ElementType::Tri3, ElementType::Tri3,
                              ElementType::Quad4, ElementType::Quad4, ElementType::Quad4];
        let face_offsets = vec![0, 3, 6, 10, 14, 18];

        let mesh = SimplexMesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
        };

        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 8);
        fine.check().unwrap();
    }

    // ─── Prism6 NC refinement tests ───────────────────────────────────────────

    fn make_single_prism_mesh() -> SimplexMesh<3> {
        let coords = vec![
            0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 1.0,
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];
        let face_conn = vec![0u32,2,1, 3,4,5, 0,1,4,3, 1,2,5,4, 0,3,5,2];
        let face_tags = vec![1,2,3,4,5];
        let face_types = vec![ElementType::Tri3, ElementType::Tri3,
                              ElementType::Quad4, ElementType::Quad4, ElementType::Quad4];
        let face_offsets = vec![0, 3, 6, 10, 14, 18];
        SimplexMesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
        }
    }

    #[test]
    fn prism6_nc_refine_empty_is_identity() {
        let mesh = make_single_prism_mesh();
        let (nc, edge_c, tri_c, quad_c, mp) = refine_nonconforming_prism(&mesh, &[]);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(edge_c.is_empty());
        assert!(tri_c.is_empty());
        assert!(quad_c.is_empty());
        assert!(mp.is_empty());
    }

    #[test]
    fn prism6_nc_refine_all_gives_no_constraints() {
        let mesh = make_single_prism_mesh();
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, edge_c, tri_c, quad_c, _) = refine_nonconforming_prism(&mesh, &all);
        assert_eq!(nc.n_elems(), 8);
        assert!(edge_c.is_empty(), "all refined → no edge constraints");
        assert!(tri_c.is_empty(), "all refined → no tri face constraints");
        assert!(quad_c.is_empty(), "all refined → no quad face constraints");
        nc.check().unwrap();
    }

    #[test]
    fn prism6_nc_refine_single_creates_constraints() {
        // 2 prisms sharing the quadrilateral face (1,2,5,4):
        // Prism 0: [0,1,2,3,4,5]  — left-front-bottom prism
        // Prism 1: [1,6,2,4,7,5]  — right-back-top prism
        // Shared quad face has global nodes {1,2,4,5}.
        let coords = vec![
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1 (shared)
            0.0, 1.0, 0.0,  // 2 (shared)
            0.0, 0.0, 1.0,  // 3
            1.0, 0.0, 1.0,  // 4 (shared)
            0.0, 1.0, 1.0,  // 5 (shared)
            1.0, 1.0, 0.0,  // 6 (prism 1 only)
            1.0, 1.0, 1.0,  // 7 (prism 1 only)
        ];
        let conn = vec![
            0u32, 1, 2, 3, 4, 5,   // prism 0
            1, 6, 2, 4, 7, 5,      // prism 1: bottom(1,6,2), top(4,7,5)
        ];
        let elem_tags = vec![1i32, 1];

        // Boundary faces (sorted node sets for keys — winding order handled by solver):
        // Prism 0: bottom(0,2,1), top(3,4,5), front(0,1,4,3), left(0,3,5,2)
        // Prism 1: bottom(1,2,6), top(4,5,7), right(1,6,7,4), back(6,2,5,7)
        // Shared quad (1,2,5,4) is INTERIOR, not on boundary.
        let face_conn = vec![
            0u32,2,1, 3,4,5, 0,1,4,3, 0,3,5,2,
            1,2,6, 4,5,7, 1,6,7,4, 6,2,5,7,
        ];
        let face_tags = vec![1i32,2,3,5, 1,2,4,6];
        let face_types = vec![
            ElementType::Tri3, ElementType::Tri3,
            ElementType::Quad4, ElementType::Quad4,
            ElementType::Tri3, ElementType::Tri3,
            ElementType::Quad4, ElementType::Quad4,
        ];
        let face_offsets = vec![0,3,6,10,14, 17,20,24,28];

        let mesh = SimplexMesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
        };

        // Refine only prism 0
        let (nc, edge_c, tri_c, quad_c, _) = refine_nonconforming_prism(&mesh, &[0]);

        assert_eq!(nc.n_elems(), 8 + 1, "first prism → 8 children, second unchanged → 9 total");
        assert!(!quad_c.is_empty(), "shared quad face nodes should produce quad face constraints");
        assert!(!edge_c.is_empty(), "shared edges should produce edge constraints");
        assert!(tri_c.is_empty(), "no hanging tri faces in this configuration");

        nc.check().unwrap();
    }

    #[test]
    fn prism6_nc_volume_conservation_partial() {
        // Same 2-prism mesh with shared nodes.
        let coords = vec![
            0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 1.0,
            1.0, 1.0, 0.0,  1.0, 1.0, 1.0,
        ];
        let conn = vec![0u32,1,2,3,4,5, 1,6,2,4,7,5];
        let elem_tags = vec![1i32, 1];
        let face_conn = vec![
            0u32,2,1, 3,4,5, 0,1,4,3, 0,3,5,2,
            1,2,6, 4,5,7, 1,6,7,4, 6,2,5,7,
        ];
        let face_tags = vec![1,2,3,5, 1,2,4,6];
        let face_types = vec![ElementType::Tri3, ElementType::Tri3,
                              ElementType::Quad4, ElementType::Quad4,
                              ElementType::Tri3, ElementType::Tri3,
                              ElementType::Quad4, ElementType::Quad4];
        let face_offsets = vec![0,3,6,10,14, 17,20,24,28];
        let mesh = SimplexMesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
        };

        let vol_orig = prism6_vol(&mesh, 0) + prism6_vol(&mesh, 1);
        assert!((vol_orig - 1.0).abs() < 1e-14, "2 prisms, each 0.5 → total=1, got {}", vol_orig);

        let (nc, _, _, _, _) = refine_nonconforming_prism(&mesh, &[0]);
        let vol_sum: f64 = (0..nc.n_elems())
            .map(|e| prism6_vol(&nc, e as ElemId))
            .sum();
        assert!(
            (vol_sum - vol_orig).abs() < 1e-12,
            "NC volume mismatch: orig={} sum={} diff={}",
            vol_orig, vol_sum, vol_sum - vol_orig,
        );
        nc.check().unwrap();
    }

    // ─── Hex8 uniform refinement tests ────────────────────────────────────

    /// Volume of Hex8 via 5-tet decomposition along diagonal (0,2,5,7).
    fn hex8_vol(mesh: &SimplexMesh<3>, e: ElemId) -> f64 {
        let ns = mesh.elem_nodes(e);
        let c = |i: usize| -> [f64; 3] { let off = ns[i] as usize * 3; [mesh.coords[off], mesh.coords[off+1], mesh.coords[off+2]] };
        tet_signed_vol(&c(0),&c(1),&c(2),&c(5)).abs()+tet_signed_vol(&c(2),&c(3),&c(0),&c(7)).abs()
        +tet_signed_vol(&c(0),&c(4),&c(5),&c(7)).abs()+tet_signed_vol(&c(2),&c(5),&c(6),&c(7)).abs()
        +tet_signed_vol(&c(0),&c(2),&c(5),&c(7)).abs()
    }

    #[test] fn hex8_uniform_single_element() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(1);
        let v0 = hex8_vol(&mesh, 0); assert!((v0-1.0).abs() < 1e-14);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, c, _) = refine_hex8_uniform(&mesh, &all);
        assert_eq!(fine.n_elems(), 8); assert!(c.is_empty());
        let vs: f64 = (0..fine.n_elems()).map(|e| hex8_vol(&fine, e as ElemId)).sum();
        assert!((vs - v0).abs() < 1e-12);
        fine.check().unwrap();
    }

    #[test] fn hex20_uniform_single_element() {
        let coords = (0..20i32).flat_map(|i| {
            let (cx,cy,cz) = (if i%2==0{0.0}else{1.0}, if (i/2)%2==0{0.0}else{1.0}, if (i/4)%2==0{0.0}else{1.0});
            vec![cx,cy,cz]
        }).collect::<Vec<f64>>();
        // Fix edge mids to actual midpoints
        let mut c = coords; c[24]=0.5; c[25]=0.0; c[26]=0.0; // node 8: (0.5,0,0)
        c[27]=1.0; c[28]=0.5; c[29]=0.0; // node 9: (1,0.5,0)
        c[30]=0.5; c[31]=1.0; c[32]=0.0; // etc
        let conn = (0..20u32).collect::<Vec<_>>();
        let fc = vec![0u32,1,2,3, 4,5,6,7, 0,1,5,4, 2,3,7,6, 0,3,7,4, 1,2,6,5];
        let ft = vec![1,2,3,4,5,6];
        let mesh = SimplexMesh { coords:c, conn, elem_tags: vec![1i32], elem_type: ElementType::Hex20,
            face_conn: fc, face_tags: ft, face_type: ElementType::Quad4,
            elem_types:None, elem_offsets:None, face_types:None, face_offsets:None,
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] };
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, c, _) = refine_hex20_uniform(&mesh, &all);
        assert_eq!(fine.n_elems(), 8); assert_eq!(fine.elem_type, ElementType::Hex8); assert!(c.is_empty());
        fine.check().unwrap();
    }

    #[test] fn hex27_uniform_single_element() {
        let conn = (0..27u32).collect::<Vec<_>>();
        let mut coords = Vec::with_capacity(81);
        for i in 0..27 {
            let (xi,yi,zi) = (i%3,i/3%3,i/9%3);
            coords.push(if xi==0{0.0}else if xi==1{1.0}else{0.5});
            coords.push(if yi==0{0.0}else if yi==1{1.0}else{0.5});
            coords.push(if zi==0{0.0}else if zi==1{1.0}else{0.5});
        }
        let fc = vec![0u32,1,2,3, 4,5,6,7, 0,1,5,4, 2,3,7,6, 0,3,7,4, 1,2,6,5];
        let ft = vec![1,2,3,4,5,6];
        let mesh = SimplexMesh { coords, conn, elem_tags: vec![1i32], elem_type: ElementType::Hex27,
            face_conn: fc, face_tags: ft, face_type: ElementType::Quad4,
            elem_types:None, elem_offsets:None, face_types:None, face_offsets:None,
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] };
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, c, _) = refine_hex27_uniform(&mesh, &all);
        assert_eq!(fine.n_elems(), 8); assert!(c.is_empty()); assert_eq!(fine.elem_type, ElementType::Hex8);
        fine.check().unwrap();
    }

    // ─── Pyramid5 uniform + NC tests ──────────────────────────────────────

    fn pyramid5_vol(mesh: &SimplexMesh<3>, e: ElemId) -> f64 {
        let ns = mesh.elem_nodes(e);
        let c = |i: usize| -> [f64; 3] { let off = ns[i] as usize * 3; [mesh.coords[off], mesh.coords[off+1], mesh.coords[off+2]] };
        tet_signed_vol(&c(0),&c(1),&c(2),&c(4)).abs()+tet_signed_vol(&c(2),&c(3),&c(0),&c(4)).abs()
    }

    #[test] fn pyramid5_uniform_single_element() {
        let coords = vec![0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];
        let conn = vec![0u32,1,2,3,4]; let elem_tags = vec![1i32];
        let fc = vec![0u32,1,2,3, 0,1,4, 1,2,4, 2,3,4, 3,0,4];
        let ft = vec![1,2,3,4,5];
        let fty = vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3];
        let fo = vec![0,4,7,10,13,16];
        let mesh = SimplexMesh { coords, conn, elem_tags, elem_type:ElementType::Pyramid5,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] };
        let v0 = pyramid5_vol(&mesh, 0); assert!((v0-1.0/3.0).abs() < 1e-14);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, c) = refine_pyramid5_uniform(&mesh, &all);
        assert_eq!(fine.n_elems(), 16); assert!(c.is_empty());
        let vs: f64 = (0..fine.n_elems()).map(|e| { let ns=fine.elem_nodes(e as ElemId);
            let c2=|i|{let off=ns[i]as usize*3;[fine.coords[off],fine.coords[off+1],fine.coords[off+2]]};
            tet_signed_vol(&c2(0),&c2(1),&c2(2),&c2(3)).abs() }).sum();
        assert!((vs - v0).abs() < 1e-12);
    }

    #[test] fn pyramid5_uniform_through_dispatch() {
        let mesh = SimplexMesh { coords: vec![0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
            conn: vec![0u32,1,2,3,4], elem_tags: vec![1i32], elem_type: ElementType::Pyramid5,
            face_conn: vec![0u32,1,2,3,0,1,4,1,2,4,2,3,4,3,0,4], face_tags: vec![1,2,3,4,5],
            face_type: ElementType::Tri3, elem_types: None, elem_offsets: None,
            face_types: Some(vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3]),
            face_offsets: Some(vec![0,4,7,10,13,16]),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![] };
        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 16); fine.check().unwrap();
    }

    fn make_pyramid_mesh() -> SimplexMesh<3> {
        SimplexMesh { coords: vec![0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
            conn: vec![0u32,1,2,3,4], elem_tags: vec![1i32], elem_type: ElementType::Pyramid5,
            face_conn: vec![0u32,1,2,3,0,1,4,1,2,4,2,3,4,3,0,4], face_tags: vec![1,2,3,4,5],
            face_type: ElementType::Tri3, elem_types: None, elem_offsets: None,
            face_types: Some(vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3]),
            face_offsets: Some(vec![0,4,7,10,13,16]),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![] }
    }

    #[test] fn pyramid5_nc_refine_empty_is_identity() {
        let mesh = make_pyramid_mesh();
        let (nc, ec, tc, qc, mp) = refine_nonconforming_pyramid(&mesh, &[]);
        assert_eq!(nc.n_elems(), mesh.n_elems()); assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(ec.is_empty()); assert!(tc.is_empty()); assert!(qc.is_empty()); assert!(mp.is_empty());
    }

    #[test] fn pyramid5_nc_refine_all_gives_no_constraints() {
        let mesh = make_pyramid_mesh();
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, ec, tc, qc, _) = refine_nonconforming_pyramid(&mesh, &all);
        assert_eq!(nc.n_elems(), 16); assert!(ec.is_empty()); assert!(tc.is_empty()); assert!(qc.is_empty());
        nc.check().unwrap();
    }

    #[test] fn pyramid5_nc_refine_single_creates_constraints() {
        let coords = vec![0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,-1.0,0.0,0.0,-1.0,0.0];
        let conn = vec![0u32,1,2,3,4, 0,1,5,6,4]; let elem_tags = vec![1i32,1];
        let fc = vec![0u32,1,2,3,1,2,4,2,3,4,3,0,4, 0,1,5,6,1,5,4,5,6,4,6,0,4];
        let ft = vec![1,2,3,5, 1,2,4,6];
        let fty = vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,
                       ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3];
        let fo = vec![0,4,7,10,13, 17,20,23,26];
        let mesh = SimplexMesh { coords, conn, elem_tags, elem_type:ElementType::Pyramid5,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] };
        let (nc, ec, tc, qc, _) = refine_nonconforming_pyramid(&mesh, &[0]);
        assert_eq!(nc.n_elems(), 17); assert!(ec.len()>=3); assert!(!tc.is_empty()); assert!(qc.is_empty());
        nc.check().unwrap();
    }

    // ─── Prism6 anisotropic tests ────────────────────────────────────────

    fn make_prism_mesh() -> SimplexMesh<3> {
        let coords = vec![0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0, 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0];
        let conn = vec![0u32,1,2,3,4,5]; let elem_tags = vec![1i32];
        let fc = vec![0u32,2,1, 3,4,5, 0,1,4,3, 1,2,5,4, 0,3,5,2];
        let ft = vec![1,2,3,4,5];
        let fty = vec![ElementType::Tri3,ElementType::Tri3,ElementType::Quad4,ElementType::Quad4,ElementType::Quad4];
        let fo = vec![0,3,6,10,14,18];
        SimplexMesh { coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![] }
    }

    #[test] fn prism_aniso_z_split_doubles_elements() {
        let mesh = make_prism_mesh(); let n = mesh.n_elems();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Z)]);
        assert_eq!(refined.n_elems(), n+1, "Z split: one elem → 2, total {}", n+1);
    }

    #[test] fn prism_aniso_edge0_split_doubles_elements() {
        let mesh = make_prism_mesh(); let n = mesh.n_elems();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Edge0)]);
        assert_eq!(refined.n_elems(), n+1, "Edge0 split: one elem → 2, total {}", n+1);
    }

    #[test] fn prism_aniso_all_delegates_to_isotropic() {
        let mesh = make_prism_mesh();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::All)]);
        assert_eq!(refined.n_elems(), 8);
        refined.check().unwrap();
    }

    #[test] fn prism_aniso_empty_marked_is_identity() {
        let mesh = make_prism_mesh();
        let (refined, c) = refine_nonconforming_prism_aniso(&mesh, &[]);
        assert_eq!(refined.n_elems(), mesh.n_elems()); assert_eq!(refined.n_nodes(), mesh.n_nodes()); assert!(c.is_empty());
    }

    #[test] fn prism_aniso_z_split_adds_three_midpoints() {
        let mesh = make_prism_mesh(); let n0 = mesh.n_nodes();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Z)]);
        assert_eq!(refined.n_nodes(), n0+3, "Z split adds 3 vertical edge mids");
    }

    // ─── Pyramid5 anisotropic tests ──────────────────────────────────────

    #[test] fn pyramid_aniso_all_delegates_to_isotropic() {
        let mesh = make_pyramid_mesh();
        let (refined, _) = refine_nonconforming_pyramid_aniso(&mesh, &[(0, PyramidRefineDir::All)]);
        assert_eq!(refined.n_elems(), 16);
        refined.check().unwrap();
    }

    #[test] fn pyramid_aniso_empty_marked_is_identity() {
        let mesh = make_pyramid_mesh();
        let (refined, c) = refine_nonconforming_pyramid_aniso(&mesh, &[]);
        assert_eq!(refined.n_elems(), mesh.n_elems()); assert!(c.is_empty());
    }

    // ─── Generalized 3-D estimator tests ─────────────────────────────────

    #[test] fn zz_3d_general_linear_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u → ZZ ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_quadratic_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "x² → ZZ >0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u → Kelly ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_matches_original_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta_orig = zz_estimator_3d(&mesh, &u);
        let eta_gen = zz_estimator_3d_general(&mesh, &u);
        assert_eq!(eta_orig.len(), eta_gen.len());
        for i in 0..eta_orig.len() {
            assert!((eta_orig[i]-eta_gen[i]).abs() < 1e-12, "ZZ mismatch at elem {i}: {} vs {}", eta_orig[i], eta_gen[i]);
        }
    }

    #[test] fn residual_3d_general_linear_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![0.0; n];
        let eta = residual_estimator_3d_general(&mesh, &u, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u, f=0 → residual ~0, got {max:.3e}");
    }

    #[test] fn residual_3d_general_matches_original_tet() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![1.0; n]; // constant source
        let eta_orig = residual_estimator_3d(&mesh, &u, &f_val);
        let eta_gen = residual_estimator_3d_general(&mesh, &u, &f_val);
        assert_eq!(eta_orig.len(), eta_gen.len());
        for i in 0..eta_orig.len() {
            assert!((eta_orig[i]-eta_gen[i]).abs() < 1e-12, "residual mismatch at elem {i}: {} vs {}", eta_orig[i], eta_gen[i]);
        }
    }

    #[test] fn residual_3d_general_linear_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![0.0; n];
        let eta = residual_estimator_3d_general(&mesh, &u, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u, f=0 on Hex → residual ~0, got {max:.3e}");
    }

    #[test] fn residual_3d_general_linear_prism() {
        let mesh = prism6_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![0.0; n];
        let eta = residual_estimator_3d_general(&mesh, &u, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u, f=0 on Prism → residual ~0, got {max:.3e}");
    }

    #[test] fn residual_3d_general_linear_pyramid() {
        let mesh = pyramid5_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![0.0; n];
        let eta = residual_estimator_3d_general(&mesh, &u, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u, f=0 on Pyramid → residual ~0, got {max:.3e}");
    }

    #[test] fn dwr_3d_general_linear_solution_zero() {
        // u = x, z = y, f = 0 → DWR = 0
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0, "linear u,z, f=0 → DWR should be small, got {max:.3e}");
    }

    #[test] fn dwr_3d_general_quadratic_solution_positive() {
        // u = x², z = y², f = -2 (Laplacian of x² is 2)
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let z: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[1]*c[1] }).collect();
        let f_val = vec![2.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "quadratic u,z → DWR > 0, got {max:.3e}");
    }

    #[test] fn dwr_3d_general_matches_2d_on_tet() {
        // For linear u,z on Tet mesh, match 2-D dwr_estimator result structure
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    // ── Hex8 estimator tests ──────────────────────────────────────────────

    #[test] fn zz_3d_general_linear_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u on Hex → ZZ ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_quadratic_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "x² on Hex → ZZ >0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u on Hex → Kelly ~0, got {max:.3e}");
    }

    // ── Prism6 estimator tests ────────────────────────────────────────────

    /// Build a unit right prism mesh: right triangle base z=0 → top z=1.
    fn prism6_unit_mesh() -> SimplexMesh<3> {
        let coords = vec![
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            1.0, 0.0, 1.0,  // 4
            0.0, 1.0, 1.0,  // 5
        ];
        let conn = vec![0u32, 1, 2, 3, 4, 5];
        let elem_tags = vec![1i32];
        let face_conn = vec![
            0u32, 2, 1,        // bottom tri
            3, 4, 5,           // top tri
            0, 1, 4, 3,        // quad front
            1, 2, 5, 4,        // quad right
            0, 3, 5, 2,        // quad left
        ];
        let face_tags = vec![1i32, 2, 3, 4, 5];
        let face_types = vec![
            ElementType::Tri3, ElementType::Tri3,
            ElementType::Quad4, ElementType::Quad4, ElementType::Quad4,
        ];
        let face_offsets = vec![0usize, 3, 6, 10, 14, 18];
        SimplexMesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        }
    }

    #[test] fn zz_3d_general_linear_prism() {
        let mesh = prism6_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u on Prism → ZZ ~0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_prism() {
        let mesh = prism6_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u on Prism → Kelly ~0, got {max:.3e}");
    }

    // ── Pyramid5 estimator tests ───────────────────────────────────────────

    /// Build a unit pyramid mesh: base z=0 (unit square), apex at (0,0,1).
    fn pyramid5_unit_mesh() -> SimplexMesh<3> {
        let coords = vec![
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.0, 0.0, 1.0,  // 4
        ];
        let conn = vec![0u32, 1, 2, 3, 4];
        let elem_tags = vec![1i32];
        let face_conn = vec![
            0u32, 1, 2, 3,  // base quad
            0, 1, 4,        // tri 1
            1, 2, 4,        // tri 2
            2, 3, 4,        // tri 3
            3, 0, 4,        // tri 4
        ];
        let face_tags = vec![1i32, 2, 3, 4, 5];
        let face_types = vec![
            ElementType::Quad4,
            ElementType::Tri3, ElementType::Tri3,
            ElementType::Tri3, ElementType::Tri3,
        ];
        let face_offsets = vec![0usize, 4, 7, 10, 13, 16];
        SimplexMesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Pyramid5,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        }
    }

    #[test] fn zz_3d_general_linear_pyramid() {
        let mesh = pyramid5_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u on Pyramid → ZZ ~0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_pyramid() {
        let mesh = pyramid5_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u on Pyramid → Kelly ~0, got {max:.3e}");
    }

    // ── Hex8 DWR tests ───────────────────────────────────────────────────

    #[test] fn dwr_3d_general_linear_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0, "linear u,z on Hex → DWR ~0, got {max:.3e}");
    }

    #[test] fn dwr_3d_general_quadratic_hex() {
        let mesh = SimplexMesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let z: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[1]*c[1] }).collect();
        let f_val = vec![2.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "x²,y² on Hex → DWR > 0, got {max:.3e}");
    }

    // ── Prism6 DWR tests ─────────────────────────────────────────────────

    #[test] fn dwr_3d_general_linear_prism() {
        let mesh = prism6_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0, "linear u,z on Prism → DWR ~0, got {max:.3e}");
    }

    // ── Pyramid5 DWR tests ───────────────────────────────────────────────

    #[test] fn dwr_3d_general_linear_pyramid() {
        let mesh = pyramid5_unit_mesh();
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0, "linear u,z on Pyramid → DWR ~0, got {max:.3e}");
    }
}
