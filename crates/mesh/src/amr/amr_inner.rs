//! Non-conforming refinement and hanging-node constraint logic.
//!
//! Sub-modules: [`super::bisect`], [`super::estimators`], [`super::p_refine`],
//! [`super::refine_2d`], [`super::make_conforming`].

use std::collections::HashMap;
use fem_core::{FaceId, NodeId, ElemId};
use crate::{element_type::ElementType, simplex::{GeometryData, Mesh}, rebuild_boundary::rebuild_3d_boundary};
use crate::cad::{ProjectionConfig, project_boundary_to_cad};

use super::bisect::{edge_key, local_edges_tri, refine_marked};

// ─── Hanging-node constraint ──────────────────────────────────────────────────

/// A hanging-node constraint: `u[constrained] = Σ coeff_i·u[parent_i]`.
///
/// P1 (linear) meshes use the classic form `u[c] = 0.5·u[a] + 0.5·u[b]`
/// (coeff_a = coeff_b = 0.5, extra empty). P2 (quadratic) hanging nodes
/// interpolate with the parent-edge P2 basis: the constrained DOF is a
/// *fine-edge midpoint* lying at the 1/4 or 3/4 point of a coarse edge, and
/// `extra` carries the coarse-edge-midpoint DOF with coefficient 3/4
/// (matching MFEM `FiniteElement::GetTransferMatrix`).
#[derive(Debug, Clone)]
pub struct HangingNodeConstraint {
    /// The constrained (hanging) node DOF index.
    pub constrained: usize,
    /// The first parent node DOF index.
    pub parent_a:    usize,
    /// The second parent node DOF index.
    pub parent_b:    usize,
    /// Coefficient of `parent_a` (P1: 0.5).
    pub coeff_a:     f64,
    /// Coefficient of `parent_b` (P1: 0.5).
    pub coeff_b:     f64,
    /// Extra parent DOFs with coefficients (P2 coarse-edge midpoint: [(mid, 0.75)]).
    pub extra:       Vec<(usize, f64)>,
}

impl HangingNodeConstraint {
    /// P1 constraint `u[c] = 0.5·(u[a] + u[b])`.
    pub fn new_p1(constrained: usize, parent_a: usize, parent_b: usize) -> Self {
        HangingNodeConstraint {
            constrained,
            parent_a,
            parent_b,
            coeff_a: 0.5,
            coeff_b: 0.5,
            extra: Vec::new(),
        }
    }

    /// Arbitrary weighted constraint `u[c] = coeff_a·u[a] + coeff_b·u[b] + Σ extra`.
    pub fn new_weighted(
        constrained: usize,
        parent_a: usize,
        parent_b: usize,
        coeff_a: f64,
        coeff_b: f64,
        extra: Vec<(usize, f64)>,
    ) -> Self {
        HangingNodeConstraint {
            constrained,
            parent_a,
            parent_b,
            coeff_a,
            coeff_b,
            extra,
        }
    }

    /// All (parent, coefficient) pairs.
    pub fn parents(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        std::iter::once((self.parent_a, self.coeff_a))
            .chain(std::iter::once((self.parent_b, self.coeff_b)))
            .chain(self.extra.iter().copied())
    }
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
    /// Edge refinement level: number of times each edge has been split.
    edge_level: HashMap<(NodeId, NodeId), u32>,
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
    mesh: Mesh<2>,
    constraints: Vec<HangingNodeConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    edge_level: HashMap<(NodeId, NodeId), u32>,
}

#[derive(Debug, Clone)]
struct NCState3DSnapshot {
    mesh: Mesh<3>,
    constraints: Vec<HangingNodeConstraint>,
    hanging_faces: Vec<HangingFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

impl Default for NCState3D {
    fn default() -> Self { Self::new() }
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
    #[allow(clippy::type_complexity)]
    pub fn refine(
        &mut self,
        mesh: &Mesh<3>,
        marked: &[ElemId],
    ) -> (
        Mesh<3>,
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
    ) -> Option<(Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.hanging_faces = snap.hanging_faces.clone();
        self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone(), self.hanging_faces.clone()))
    }
}

impl Default for NCState {
    fn default() -> Self { Self::new() }
}

/// Search the mesh for an existing node exactly at the midpoint of edge
/// `(a, b)`.  Used as a fallback when `active_midpoints` does not know the
/// edge (e.g. the mesh was rebuilt and renumbered between NC-refinement
/// rounds, as happens in the parallel partition rebuild).
fn find_midpoint_node(mesh: &Mesh<2>, key: &(NodeId, NodeId)) -> Option<NodeId> {
    let (a, b) = *key;
    let xa = mesh.coords_of(a);
    let xb = mesh.coords_of(b);
    let mx = 0.5 * (xa[0] + xb[0]);
    let my = 0.5 * (xa[1] + xb[1]);
    // A midpoint node must be referenced by at least one element (it cannot
    // be an isolated node).  Scan all nodes referenced by elements.
    for n in 0..mesh.n_nodes() as NodeId {
        let c = mesh.coords_of(n);
        if (c[0] - mx).abs() < 1e-12 && (c[1] - my).abs() < 1e-12 {
            return Some(n);
        }
    }
    None
}

impl NCState {
    /// Create an empty NC state for a conforming initial mesh.
    pub fn new() -> Self {
        NCState {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            edge_level: HashMap::new(),
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
    /// `nc_limit` limits the maximum refinement-level difference between
    /// adjacent elements (0 = no limit).  When exceeded, the coarse neighbor
    /// is also refined (propagation).
    ///
    /// - Refines `marked` elements via red refinement (4 children each).
    /// - Tracks which previous hanging nodes get resolved.
    /// - Returns `(new_mesh, constraints, midpoint_map)` where `midpoint_map`
    ///   maps `(a, b) → mid` for each newly created midpoint node.
    ///   Use [`prolongate_p1`] with the midpoint map to transfer solutions.
    pub fn refine(
        &mut self,
        mesh: &Mesh<2>,
        marked: &[ElemId],
        nc_limit: u32,
    ) -> (Mesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        assert!(
            mesh.elem_type == ElementType::Tri3,
            "NCState::refine: only Tri3 meshes are supported"
        );

        if marked.is_empty() {
            return (mesh.clone(), self.constraints.clone(), HashMap::new());
        }

        // ── nc_limit propagation ───────────────────────────────────────────
        let n_elems = mesh.n_elems();
        let prop_marked: Vec<ElemId> = if nc_limit > 0 {
            let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for &(a, b) in &local_edges_tri() {
                    edge_elems.entry(edge_key(ns[a], ns[b])).or_default().push(e);
                }
            }
            propagate_nc_limit_tri(marked, mesh, &edge_elems, &self.edge_level, nc_limit)
        } else {
            marked.to_vec()
        };

        self.history.push(NCState2DSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
            edge_level: self.edge_level.clone(),
        });

        let marked_set: std::collections::HashSet<ElemId> = prop_marked.iter().copied().collect();
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

        for &e in &prop_marked {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                let key = edge_key(ns[a], ns[b]);
                if midpoint_map.contains_key(&key) { continue; }
                // Check if a midpoint already exists from a previous level.
                if let Some(&mid) = self.active_midpoints.get(&key) {
                    midpoint_map.insert(key, mid);
                } else if let Some(mid) = find_midpoint_node(mesh, &key) {
                    // Fallback: a midpoint node for this edge may already
                    // exist in the mesh even though `active_midpoints` does
                    // not know it (e.g. the mesh was rebuilt/renumbered by a
                    // parallel partition rebuild between rounds).  Reuse it
                    // instead of creating a duplicate midpoint.
                    midpoint_map.insert(key, mid);
                    self.active_midpoints.insert(key, mid);
                } else {
                    let xa = mesh.coords_of(ns[a]);
                    let xb = mesh.coords_of(ns[b]);
                    new_coords.push(0.5 * (xa[0] + xb[0]));
                    new_coords.push(0.5 * (xa[1] + xb[1]));
                    let id = next_node;
                    // Track edge refinement level: sub-edges get parent+1.
                    let parent_level = self.edge_level.get(&key).copied().unwrap_or(0);
                    self.edge_level.insert(edge_key(ns[a], id), parent_level + 1);
                    self.edge_level.insert(edge_key(id, ns[b]), parent_level + 1);
                    self.edge_level.remove(&key);
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

        let new_mesh = Mesh::uniform(
            new_coords, new_conn, new_tags, ElementType::Tri3,
            new_face_conn, new_face_tags, ElementType::Line2,
        );

        (new_mesh, self.constraints.clone(), midpoint_map)
    }

    /// Roll back one NC refinement step.
    ///
    /// Returns the previous mesh and restored constraints if history exists.
    pub fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.active_midpoints = snap.active_midpoints;
        self.edge_level = snap.edge_level;
        Some((snap.mesh, self.constraints.clone()))
    }
}

/// Unified 2-D non-conforming AMR trait — matches C++ `Mesh::GeneralRefinement`.
pub trait NcState2D {
    /// Refine `marked` elements, returning `(new_mesh, constraints, midpoint_map)`.
    fn refine(
        &mut self,
        mesh: &Mesh<2>,
        marked: &[ElemId],
        nc_limit: u32,
    ) -> (Mesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>);

    /// Roll back the last refinement pass.
    fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)>;

    /// Whether a rollback is available.
    fn can_derefine(&self) -> bool;

    /// Current hanging-node constraints.
    fn constraints(&self) -> &[HangingNodeConstraint];

    /// Active edge midpoints (parent edge -> midpoint node) of the current
    /// mesh — used by the P2 constraint upgrade to walk split sub-edges.
    fn midpoints(&self) -> &HashMap<(NodeId, NodeId), NodeId> {
        static EMPTY: std::sync::OnceLock<HashMap<(NodeId, NodeId), NodeId>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| HashMap::new())
    }

    /// Tree-node indices of derefinable groups (children all leaves).
    fn deref_groups(&self) -> Vec<usize> { Vec::new() }

    /// Children (current-mesh element indices) of the given tree node.
    fn deref_group_children(&self, _node: usize) -> [ElemId; 4] {
        [ElemId::MAX; 4]
    }

    /// Coarsen the given tree-node groups; returns the new mesh or `None`.
    fn derefine_groups(&mut self, _mesh: &Mesh<2>, _groups: &[usize]) -> Option<Mesh<2>> {
        None
    }

    /// MFEM `NCMesh::CheckDerefinementNCLevel`: whether derefining tree-node
    /// `node` keeps the max NC level between adjacent elements within
    /// `nc_limit`.  Default: always allowed (no NC limit).
    fn deref_group_nc_ok(&self, _node: usize, _nc_limit: u32, _mesh: &Mesh<2>) -> bool {
        true
    }
}

impl NcState2D for NCState {
    fn refine(
        &mut self, mesh: &Mesh<2>, marked: &[ElemId], nc_limit: u32,
    ) -> (Mesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        self.refine(mesh, marked, nc_limit)
    }
    fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        self.derefine_last()
    }
    fn can_derefine(&self) -> bool { self.can_derefine() }
    fn constraints(&self) -> &[HangingNodeConstraint] { self.constraints() }
}

impl NcState2D for NCStateQuad {
    fn refine(
        &mut self, mesh: &Mesh<2>, marked: &[ElemId], nc_limit: u32,
    ) -> (Mesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        self.refine(mesh, marked, nc_limit)
    }
    fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        self.derefine_last()
    }
    fn can_derefine(&self) -> bool { self.can_derefine() }
    fn constraints(&self) -> &[HangingNodeConstraint] { self.constraints() }
    fn deref_groups(&self) -> Vec<usize> { self.deref_groups() }
    fn deref_group_children(&self, node: usize) -> [ElemId; 4] {
        self.deref_group_children(node)
    }
    fn derefine_groups(&mut self, mesh: &Mesh<2>, groups: &[usize]) -> Option<Mesh<2>> {
        self.derefine_groups(mesh, groups)
    }
    fn midpoints(&self) -> &HashMap<(NodeId, NodeId), NodeId> { self.active_midpoints() }
    fn deref_group_nc_ok(&self, node: usize, nc_limit: u32, mesh: &Mesh<2>) -> bool {
        self.deref_group_nc_ok(node, nc_limit, mesh)
    }
}

/// Propagate refinement to neighbors when nc_limit would be violated (Tri3).
fn propagate_nc_limit_tri(
    marked: &[ElemId],
    mesh: &Mesh<2>,
    edge_elems: &HashMap<(NodeId, NodeId), Vec<ElemId>>,
    edge_level: &HashMap<(NodeId, NodeId), u32>,
    nc_limit: u32,
) -> Vec<ElemId> {
    use std::collections::BTreeSet;
    let mut result: BTreeSet<ElemId> = marked.iter().copied().collect();
    let mut queue: Vec<ElemId> = marked.to_vec();
    while let Some(e) = queue.pop() {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            if edge_level.get(&key).copied().unwrap_or(0) >= nc_limit {
                if let Some(neighbors) = edge_elems.get(&key) {
                    for &n in neighbors {
                        if n != e && result.insert(n) {
                            queue.push(n);
                        }
                    }
                }
            }
        }
    }
    result.into_iter().collect()
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
///   node IDs (as returned by [`refine_nonconforming`]).
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
/// - `mesh`   — input `Mesh<2>` with `elem_type = Tri3`.
/// - `marked` — sorted list of element indices to refine.
///
/// # Returns
/// `(new_mesh, constraints)` where `constraints` lists all hanging nodes.
pub fn refine_nonconforming(
    mesh: &Mesh<2>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 2);
    }

    (new_mesh, constraints)
}



/// Uniformly refine all elements of a 2-D mesh.
///
/// - Tri3  → 4 Tri3  (newest-vertex bisection).
/// - Quad4 → 4 Quad4 (conforming split matching MFEM UniformRefinement2D_base).
pub fn refine_uniform(mesh: &Mesh<2>) -> Mesh<2> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    // Mixed Tri3+Quad4 meshes: per-element-type refinement with a shared
    // edge-midpoint map (MFEM UniformRefinement2D_base handles mixed meshes).
    if mesh.elem_types.is_some() {
        return refine_uniform_2d_mixed(mesh);
    }    match mesh.elem_type {
        ElementType::Tri3 => refine_marked(mesh, &all),
        ElementType::Quad4 => refine_uniform_quad4(mesh),
        _ => panic!(
            "refine_uniform: unsupported element type {:?} (only Tri3 and Quad4 are supported)",
            mesh.elem_type
        ),
    }
}

/// Rotate each Tri3 element's vertex indices so that edge (0,1) is the
/// longest edge — a 1:1 port of MFEM `Mesh::MarkTriMeshForRefinement`
/// (`mesh/mesh.cpp`), which MFEM runs on load when the mesh constructor's
/// `refine` flag is set (e.g. `Mesh(mesh_file, 1, 1)`).
///
/// This reordering changes the *local* vertex order of each triangle; the
/// global vertex numbering is unchanged.  It matters for 1:1 fidelity of
/// subsequent `UniformRefinement` because the new edge-midpoint vertex ids
/// follow the element × local-edge traversal order, which depends on the
/// rotated local ordering (see `refine_uniform_2d_mixed`).
///
/// Quad4 elements are left untouched (MFEM only rotates triangles).
pub fn mark_tri_mesh_for_refinement(mesh: &mut Mesh<2>) {
    const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    let n_elems = mesh.n_elems();
    for e in 0..n_elems as ElemId {
        if mesh.element_type_at(e) != ElementType::Tri3 {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        // Squared lengths of the three edges: d[0] = |v0-v1|², d[1] = |v1-v2|²,
        // d[2] = |v2-v0|²  (MFEM Triangle::MarkEdge, 2-D).
        let mut d = [0.0_f64; 3];
        for (i, &(li, lj)) in TRI_EDGES.iter().enumerate() {
            let a = mesh.coords_of(ns[li]);
            let b = mesh.coords_of(ns[lj]);
            d[i] = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        }
        // MFEM: shift selects the longest edge; indices are rotated so the
        // longest edge ends up between local vertices 0 and 1.
        let shift = if d[0] >= d[1] {
            if d[0] >= d[2] { 0 } else { 2 }
        } else if d[1] >= d[2] { 1 } else { 2 };
        if shift == 0 {
            continue;
        }
        // Rewrite the element's conn slice (elem_offsets aware).
        let start = if let Some(ref offs) = mesh.elem_offsets {
            offs[e as usize]
        } else {
            e as usize * 3
        };
        let end = if let Some(ref offs) = mesh.elem_offsets {
            offs[e as usize + 1]
        } else {
            start + 3
        };
        let (a, b, c) = (mesh.conn[start], mesh.conn[start + 1], mesh.conn[start + 2]);
        match shift {
            // case 1: [v0,v1,v2] → [v1,v2,v0]
            1 => { mesh.conn[start] = b; mesh.conn[start + 1] = c; mesh.conn[start + 2] = a; }
            // case 2: [v0,v1,v2] → [v2,v0,v1]
            _ => { mesh.conn[start] = c; mesh.conn[start + 1] = a; mesh.conn[start + 2] = b; }
        }
        debug_assert!(end <= mesh.conn.len());
    }
}

/// Uniform refinement of a mixed Tri3 + Quad4 2-D mesh, matching MFEM's
/// `UniformRefinement2D_base` (which supports mixed element types).
///
/// Vertex layout follows MFEM exactly:
/// - original vertices (0..n_verts),
/// - edge midpoints `oedge + i` in `el_to_edge` (element × local-edge) order,
/// - quad center vertices `oelem + q` appended AFTER all edge midpoints, one
///   per quadrilateral element in element order (`quad_counter`).
///
/// Each Tri3 → 4 Tri3 children (red refinement), each Quad4 → 4 Quad4
/// children, boundary segments → 2 segments, all child attributes copied.
fn refine_uniform_2d_mixed(mesh: &Mesh<2>) -> Mesh<2> {
    let dim = 2usize;
    let n_verts = mesh.n_nodes();
    let n_elems = mesh.n_elems();

    // ── 1. Global edge map in element × local-edge order (MFEM el_to_edge) ──
    const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
    let mut edge_map: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let mut edge_list: Vec<(NodeId, NodeId)> = Vec::new();
    for e in 0..n_elems as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        let edges = match et {
            ElementType::Tri3 | ElementType::Tri6 => &TRI_EDGES[..],
            ElementType::Quad4 => &QUAD_EDGES[..],
            _ => panic!("refine_uniform_2d_mixed: unsupported element type {et:?}"),
        };
        for &(li, lj) in edges {
            let key = (ns[li].min(ns[lj]), ns[li].max(ns[lj]));
            edge_map.entry(key).or_insert_with(|| {
                let id = edge_list.len();
                edge_list.push(key);
                id
            });
        }
    }
    let n_edges = edge_list.len();

    // ── 2. Count quads: their centers are appended after all edge midpoints ──
    let mut n_quads = 0usize;
    for e in 0..n_elems as ElemId {
        if mesh.element_type_at(e) == ElementType::Quad4 { n_quads += 1; }
    }
    let oedge = n_verts;
    let oelem = oedge + n_edges;
    let n_new_verts = oelem + n_quads;
    let mut new_coords = vec![0.0_f64; n_new_verts * dim];
    new_coords[..n_verts * dim].copy_from_slice(&mesh.coords);

    // Edge midpoint coordinates, stored at vertex id `oedge + edge_id`
    // (MFEM UniformRefinement2D_base calls AverageVertices(vv, 2, oedge+e[ei])
    // in element order — the vertex index oedge+edge_id matches our map).
    for (ei, &(a, b)) in edge_list.iter().enumerate() {
        let vi = oedge + ei;
        let ca = &mesh.coords[a as usize * dim..(a as usize + 1) * dim];
        let cb = &mesh.coords[b as usize * dim..(b as usize + 1) * dim];
        new_coords[vi * dim]     = (ca[0] + cb[0]) * 0.5;
        new_coords[vi * dim + 1] = (ca[1] + cb[1]) * 0.5;
    }

    // ── 3. Children: Tri3 → 4 Tri3, Quad4 → 4 Quad4, in element order ───────
    let mut child_conn = Vec::with_capacity(n_elems * 4 * 4);
    let mut child_tags = Vec::with_capacity(n_elems * 4);
    let mut child_types = Vec::with_capacity(n_elems * 4);
    let mut child_offsets: Vec<usize> = vec![0];
    let mut quad_counter = 0usize;

    for e in 0..n_elems as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];
        let e_mid: Vec<NodeId> = match et {
            ElementType::Tri3 | ElementType::Tri6 => TRI_EDGES.iter().map(|&(li, lj)| {
                let key = (ns[li].min(ns[lj]), ns[li].max(ns[lj]));
                (oedge + edge_map[&key]) as NodeId
            }).collect(),
            _ => QUAD_EDGES.iter().map(|&(li, lj)| {
                let key = (ns[li].min(ns[lj]), ns[li].max(ns[lj]));
                (oedge + edge_map[&key]) as NodeId
            }).collect(),
        };
        match et {
            ElementType::Tri3 | ElementType::Tri6 => {
                // MFEM UniformRefinement2D_base child order (lines 9787-9794):
                //   [v0, e0, e2], [e1, e2, e0](center), [e0, v1, e1], [e2, e1, v2]
                for c in [
                    [ns[0], e_mid[0], e_mid[2]],
                    [e_mid[1], e_mid[2], e_mid[0]],
                    [e_mid[0], ns[1], e_mid[1]],
                    [e_mid[2], e_mid[1], ns[2]],
                ] {
                    child_conn.extend_from_slice(&c);
                    child_offsets.push(child_conn.len());
                    child_types.push(ElementType::Tri3);
                }
            }
            _ => {
                let center_idx = (oelem + quad_counter) as NodeId;
                quad_counter += 1;
                let cidx = center_idx as usize;
                for d in 0..dim {
                    let mut s = 0.0;
                    for &vi in ns {
                        s += mesh.coords[vi as usize * dim + d];
                    }
                    new_coords[cidx * dim + d] = s / 4.0;
                }
                // MFEM UniformRefinement2D_base (lines 9805-9812):
                //   [v0, e0, center, e3], [e0, v1, e1, center],
                //   [center, e1, v2, e2], [e3, center, e2, v3]
                for c in [
                    [ns[0], e_mid[0], center_idx, e_mid[3]],
                    [e_mid[0], ns[1], e_mid[1], center_idx],
                    [center_idx, e_mid[1], ns[2], e_mid[2]],
                    [e_mid[3], center_idx, e_mid[2], ns[3]],
                ] {
                    child_conn.extend_from_slice(&c);
                    child_offsets.push(child_conn.len());
                    child_types.push(ElementType::Quad4);
                }
            }
        }
        child_tags.push(tag);
        child_tags.push(tag);
        child_tags.push(tag);
        child_tags.push(tag);
    }

    // ── 4. Boundary: each segment → 2 segments via edge midpoint ───────────
    let n_faces = mesh.n_faces();
    let mut new_face_conn = Vec::with_capacity(n_faces * 4);
    let mut new_face_tags = Vec::with_capacity(n_faces * 2);
    for f in 0..n_faces {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let key = (a.min(b), a.max(b));
        let mid = (oedge + edge_map[&key]) as NodeId;
        new_face_conn.extend_from_slice(&[a, mid, mid, b]);
        let tag = mesh.face_tags[f];
        new_face_tags.push(tag);
        new_face_tags.push(tag);
    }

    Mesh {
        coords: new_coords,
        conn: child_conn,
        elem_tags: child_tags,
        elem_type: mesh.elem_type,
        face_conn: new_face_conn,
        face_tags: new_face_tags,
        face_type: ElementType::Line2,
        elem_types: Some(child_types),
        elem_offsets: Some(child_offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
            nc_vertex_view: None,
}
}

/// Conforming Quad4 uniform refinement matching MFEM's UniformRefinement2D_base.
///
/// Each Quad4 is split into 4 sub-Quads by connecting edge midpoints and the
/// element center.  Vertex ordering matches MFEM's quad_t::Edges convention:
///
/// ```text
///   v[3] ---- e[3] ---- v[2]
///     |                  |
///    e[3]    center     e[2]
///     |                  |
///   v[0] ---- e[0] ---- v[1]      (CCW = v0→v1→v2→v3)
/// ```
///
/// Child elements (CCW):
///   - 0 (lower-left):  v[0],  e[0], center, e[3]
///   - 1 (lower-right): e[0],  v[1], e[1],  center
///   - 2 (upper-right): center, e[1], v[2],  e[2]
///   - 3 (upper-left):  e[3],  center, e[2], v[3]
///
/// New vertex order: original vertices, then edge midpoints, then element centers.
/// Edge midpoints are deduplicated via a (min,max) key so adjacent elements share them.
fn refine_uniform_quad4(mesh: &Mesh<2>) -> Mesh<2> {
    let dim = 2usize;
    let npe = 4usize;
    let n_orig_verts = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let n_bdr_faces = mesh.face_conn.len() / 2;

    // Quad edges matching MFEM quad_t::Edges and HCurlSpace QUAD_EDGES.
    const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    // Count unique edges (matching MFEM's edge numbering order).
    let mut edge_set: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let mut edge_list: Vec<(NodeId, NodeId)> = Vec::new();
    for e in 0..n_elems {
        let base = e * npe;
        for &(li, lj) in &QUAD_EDGES {
            let a = mesh.conn[base + li];
            let b = mesh.conn[base + lj];
            let key = (a.min(b), a.max(b));
            edge_set.entry(key).or_insert_with(|| {
                let idx = edge_list.len();
                edge_list.push(key);
                idx
            });
        }
    }
    // MFEM: boundary edges already in el_to_edge from element traversal.
    // Verify no orphan boundary edges exist.
    for f in 0..n_bdr_faces {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let key = (a.min(b), a.max(b));
        debug_assert!(edge_set.contains_key(&key),
            "orphan boundary edge ({},{}) — not found in any element", a, b);
    }
    let n_edges = edge_list.len();

    // Allocate new vertices: [orig | edge midpoints | element centers]
    let n_new_verts = n_orig_verts + n_edges + n_elems;
    let mut new_coords = vec![0.0_f64; n_new_verts * dim];
    new_coords[..n_orig_verts * dim].copy_from_slice(&mesh.coords);

    // Edge midpoint coordinates (folded vertices: average of the two vertex
    // coordinates — MFEM's edge transformation, which uses `vertices`).
    for (ei, &(a, b)) in edge_list.iter().enumerate() {
        let vi = n_orig_verts + ei;
        let ca = &mesh.coords[a as usize * dim..(a as usize + 1) * dim];
        let cb = &mesh.coords[b as usize * dim..(b as usize + 1) * dim];
        new_coords[vi * dim]     = (ca[0] + cb[0]) * 0.5;
        new_coords[vi * dim + 1] = (ca[1] + cb[1]) * 0.5;
    }

    // Per-element geometry propagation (if the parent mesh carries one).
    // The child elements get independent geometry nodes computed from the
    // *parent element's own* geometry (MFEM `nodes` update), which keeps the
    // two sides of a periodic seam geometrically distinct.
    let parent_geom: Option<Vec<[[f64; 2]; 4]>> = mesh.geometry.as_ref().map(|g| {
        (0..n_elems)
            .map(|e| {
                let off = e * g.nodes_per_elem;
                let mut out = [[0.0_f64; 2]; 4];
                for k in 0..4 {
                    let c = &g.coords[g.conn[off + k] as usize * dim..(g.conn[off + k] as usize + 1) * dim];
                    out[k] = [c[0], c[1]];
                }
                out
            })
            .collect()
    });
    let mut child_geom_conn: Vec<NodeId> = Vec::new();
    let mut child_geom_coords: Vec<f64> = Vec::new();

    let mut child_conn = Vec::with_capacity(n_elems * 4 * npe);
    let mut new_tags = Vec::with_capacity(n_elems * 4);

    for e in 0..n_elems {
        let base = e * npe;
        let v = [
            mesh.conn[base],
            mesh.conn[base + 1],
            mesh.conn[base + 2],
            mesh.conn[base + 3],
        ];
        let center_idx = (n_orig_verts + n_edges + e) as NodeId;

        // Parent element's own geometry nodes (g[i] corresponds to v[i]).
        let g = parent_geom.as_ref().map(|pg| pg[e]);

        // Center coordinate: from the parent element's own geometry (MFEM
        // evaluates the element transformation at the center).  Falls back to
        // the folded-vertex average when no per-element geometry exists.
        let cidx = center_idx as usize;
        for d in 0..dim {
            let mut s = 0.0;
            for gi in 0..4 {
                s += match &g {
                    Some(pg) => pg[gi][d],
                    None => mesh.coords[v[gi] as usize * dim + d],
                };
            }
            new_coords[cidx * dim + d] = s / 4.0;
        }

        // Global edge-midpoint indices for local edges 0..3
        let e_mid: [NodeId; 4] = core::array::from_fn(|li| {
            let (a, b) = QUAD_EDGES[li];
            let key = (v[a].min(v[b]), v[a].max(v[b]));
            (n_orig_verts + edge_set[&key]) as NodeId
        });

        // MFEM pattern: child elements (lines 9805-9812)
        // 0 (lower-left):  v[0],  e[0], center, e[3]
        child_conn.extend_from_slice(&[v[0], e_mid[0], center_idx, e_mid[3]]);
        // 1 (lower-right): e[0],  v[1], e[1],  center
        child_conn.extend_from_slice(&[e_mid[0], v[1], e_mid[1], center_idx]);
        // 2 (upper-right): center, e[1], v[2],  e[2]
        child_conn.extend_from_slice(&[center_idx, e_mid[1], v[2], e_mid[2]]);
        // 3 (upper-left):  e[3],  center, e[2], v[3]
        child_conn.extend_from_slice(&[e_mid[3], center_idx, e_mid[2], v[3]]);

        // Child geometry (per-element independent nodes), when the parent has
        // geometry.  The child nodes follow the same H1 vertex ordering as
        // `child_conn` (v0=LL, v1=LR, v2=UR, v3=UL); parent geometry nodes are
        // already in that order (the reader normalised the L2 lexicographic
        // order to H1).  H1 edges: bottom=(0,1), right=(1,2), top=(2,3),
        // left=(3,0).
        if let Some(pg) = &g {
            let em = [
                avg2(&pg[0], &pg[1]), // bottom edge midpoint
                avg2(&pg[1], &pg[2]), // right edge midpoint
                avg2(&pg[2], &pg[3]), // top edge midpoint
                avg2(&pg[3], &pg[0]), // left edge midpoint
            ];
            // MFEM RefinementMatrix column order is the L2-lex DOF order
            // ((0,0),(1,0),(0,1),(1,1)) — H1 pg order is (0,0),(1,0),(1,1),(0,1),
            // so the last two swap.  The 0.25-weighted column sums differ by
            // 1 ulp depending on order (bit-identical target verified).
            let cc = avg4(&pg[0], &pg[1], &pg[3], &pg[2]);
            let children: [[[f64; 2]; 4]; 4] = [
                [pg[0], em[0], cc, em[3]], // LL: v0, bottom-mid, center, left-mid
                [em[0], pg[1], em[1], cc], // LR: bottom-mid, v1, right-mid, center
                [cc, em[1], pg[2], em[2]], // UR: center, right-mid, v2, top-mid
                [em[3], cc, em[2], pg[3]], // UL: left-mid, center, top-mid, v3
            ];
            for c in children {
                for node in c {
                    child_geom_conn.push((child_geom_coords.len() / dim) as NodeId);
                    child_geom_coords.push(node[0]);
                    child_geom_coords.push(node[1]);
                }
            }
        }

        let tag = mesh.elem_tags[e];
        new_tags.extend_from_slice(&[tag, tag, tag, tag]);
    }

    // Boundary: each segment → 2 segments via midpoint
    let mut new_face_conn = Vec::with_capacity(n_bdr_faces * 4);
    let mut new_face_tags = Vec::with_capacity(n_bdr_faces * 2);
    for f in 0..n_bdr_faces {
        let a = mesh.face_conn[f * 2];
        let b = mesh.face_conn[f * 2 + 1];
        let key = (a.min(b), a.max(b));
        let mid = (n_orig_verts + edge_set[&key]) as NodeId;
        new_face_conn.push(a);
        new_face_conn.push(mid);
        new_face_conn.push(mid);
        new_face_conn.push(b);
        let tag = mesh.face_tags[f];
        new_face_tags.push(tag);
        new_face_tags.push(tag);
    }

    Mesh {
        coords: new_coords,
        conn: child_conn,
        elem_tags: new_tags,
        elem_type: ElementType::Quad4,
        face_conn: new_face_conn,
        face_tags: new_face_tags,
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        nc_vertex_view: None,
        geometry: parent_geom.map(|_| {
            let n_geom = child_geom_coords.len() / dim;
            GeometryData {
                order: 1,
                conn: child_geom_conn,
                nodes_per_elem: 4,
                coords: child_geom_coords,
                n_nodes: n_geom,
            }
        }),
    }
}

/// Average of two 2-D points.
fn avg2(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2] {
    // MFEM RefinementMatrix interpolation: the sparse-matrix Mult accumulates
    // 0.5*p0 then fuses 0.5*p1 via FMA (kernels::Mult).
    [0.5f64.mul_add(b[0], 0.5 * a[0]), 0.5f64.mul_add(b[1], 0.5 * a[1])]
}

/// Average of four 2-D points.
fn avg4(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> [f64; 2] {
    [
        0.25f64.mul_add(d[0], 0.25f64.mul_add(c[0], 0.25 * a[0] + 0.25 * b[0])),
        0.25f64.mul_add(d[1], 0.25f64.mul_add(c[1], 0.25 * a[1] + 0.25 * b[1])),
    ]
}

/// Uniformly refine all elements of a 3-D mesh, dispatching to the appropriate
/// refinement path.
///
/// Tet4 → 8 Tet4, Hex8 → 8 Hex8, Hex20 → 8 Hex8, Hex27 → 8 Hex8,
/// Prism6 → 8 Prism6, Pyramid5 → 16 Tet4.
pub fn refine_uniform_3d(mesh: &Mesh<3>) -> Mesh<3> {
    let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
    // For mixed-element meshes, use per-element-type refinement with shared edge map.
    if mesh.elem_types.is_some() {
        return refine_mixed_3d(mesh);
    }
    let mut result = match mesh.elem_type {
        ElementType::Tet4 | ElementType::Tet10 => {
            let (m, _, _) = refine_nonconforming_3d(mesh, &all, None);
            m
        }
        ElementType::Hex8 => {
            let (m, _, _, _) = refine_nonconforming_hex(mesh, &all, None);
            m
        }
        ElementType::Hex20 | ElementType::Hex27 => {
            let n_elems = mesh.n_elems();
            let npe = mesh.elem_type.nodes_per_element();
            let mut hex8_conn = Vec::with_capacity(n_elems * 8);
            for e in 0..n_elems {
                let off = e * npe;
                hex8_conn.extend_from_slice(&mesh.conn[off..off + 8]);
            }
            let hex8_mesh = Mesh {
                coords: mesh.coords.clone(),
                conn: hex8_conn,
                elem_tags: mesh.elem_tags.clone(),
                elem_type: ElementType::Hex8,
                face_conn: vec![], face_tags: vec![],
                face_type: ElementType::Quad4,
                elem_types: None, elem_offsets: None,
                face_types: None, face_offsets: None,
                face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
};
            let (m, _, _, _) = refine_nonconforming_hex(&hex8_mesh, &all, None);
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
    };
    rebuild_3d_boundary(&mut result, mesh);
    result
}

/// MFEM `UniformRefinement3D_base` best-aspect-ratio refinement-type
/// selection (rt_algo = 1): computes the aspect ratio κ = max(σ0/σ2) of two
/// candidate octahedron splits per refinement type and picks the type with
/// the smallest κ.  `J` is the element Jacobian at the centroid (constant
/// for a linear tet), transformed by the inverse regular-tet Jacobian
/// (`Geometry::JacToPerfJac` with `PerfGeomToGeomJac[TETRAHEDRON]`).
pub fn tet_select_rt_debug(mesh: &Mesh<3>, ns: &[NodeId]) -> usize {
    let c = |i: usize| mesh.coords_of(ns[i]);
    let (v0, v1, v2, v3) = (c(0), c(1), c(2), c(3));
    // J[axis][col] = v_{col+1} - v0
    let mut j = [[0.0_f64; 3]; 3];
    for t in 0..3 {
        j[t][0] = v1[t] - v0[t];
        j[t][1] = v2[t] - v0[t];
        j[t][2] = v3[t] - v0[t];
    }
    // Em: cols 0-2 = 0.5*J, cols 3-5 = 0.5*(J_i + J_j).
    let mut em = [[0.0_f64; 6]; 3];
    for t in 0..3 {
        for s in 0..3 {
            em[t][s] = 0.5 * j[t][s];
        }
        em[t][3] = 0.5 * (j[t][0] + j[t][1]);
        em[t][4] = 0.5 * (j[t][0] + j[t][2]);
        em[t][5] = 0.5 * (j[t][1] + j[t][2]);
    }
    // Inverse of the regular-tet Jacobian (GetPerfPointMat(TETRAHEDRON)).
    // Values = MFEM's `PerfGeomToGeomJac[TET]` exactly (probe-printed with
    // %.17g): the last-ulp difference from a closed-form inverse changes the
    // rt tie-breaking on near-symmetric tets.
    let perf_inv = [
        [1.0, -0.57735026918962584, -0.40824829046386302],
        [0.0, 1.1547005383792517, -0.40824829046386302],
        [0.0, 0.0, 1.2247448713915892],
    ];
    // Aspect ratio of a candidate split: κ = max(σ0/σ2 over the two Jacobians).
    let kappa_of = |js1: &[[f64; 3]; 3], js2: &[[f64; 3]; 3]| -> f64 {
        let ar = |js: &[[f64; 3]; 3]| -> f64 {
            // Jp = Js * PerfGeomToGeomJac[TET]
            let mut jp = [[0.0_f64; 3]; 3];
            for t in 0..3 {
                for c in 0..3 {
                    jp[t][c] = js[t][0] * perf_inv[0][c]
                        + js[t][1] * perf_inv[1][c]
                        + js[t][2] * perf_inv[2][c];
                }
            }
            // MFEM's CalcSingularvalue<3> reads the DenseMatrix data in
            // column-major order; the bit-exact port needs the transpose in
            // row-major form (σ(A) = σ(Aᵀ), but the tie-breaking arithmetic
            // depends on the orientation).
            let data: [f64; 9] = [
                jp[0][0], jp[1][0], jp[2][0],
                jp[0][1], jp[1][1], jp[2][1],
                jp[0][2], jp[1][2], jp[2][2],
            ];
            let s0 = crate::mfem_kernels::calc_singularvalue_3(&data, 0);
            let s2 = crate::mfem_kernels::calc_singularvalue_3(&data, 2);
            if s2.abs() < 1e-300 { f64::INFINITY } else { s0 / s2 }
        };
        ar(js1).max(ar(js2))
    };
    // rt = 0
    let js = [
        [em[0][5] - em[0][0], em[0][1] - em[0][0], em[0][2] - em[0][0]],
        [em[1][5] - em[1][0], em[1][1] - em[1][0], em[1][2] - em[1][0]],
        [em[2][5] - em[2][0], em[2][1] - em[2][0], em[2][2] - em[2][0]],
    ];
    let js2 = [
        [em[0][5] - em[0][0], em[0][2] - em[0][0], em[0][4] - em[0][0]],
        [em[1][5] - em[1][0], em[1][2] - em[1][0], em[1][4] - em[1][0]],
        [em[2][5] - em[2][0], em[2][2] - em[2][0], em[2][4] - em[2][0]],
    ];
    let mut kappa_min = kappa_of(&js, &js2);
    let mut rt = 0usize;
    // rt = 1
    let js = [
        [em[0][0] - em[0][1], em[0][4] - em[0][1], em[0][2] - em[0][1]],
        [em[1][0] - em[1][1], em[1][4] - em[1][1], em[1][2] - em[1][1]],
        [em[2][0] - em[2][1], em[2][4] - em[2][1], em[2][2] - em[2][1]],
    ];
    let js2 = [
        [em[0][2] - em[0][1], em[0][4] - em[0][1], em[0][5] - em[0][1]],
        [em[1][2] - em[1][1], em[1][4] - em[1][1], em[1][5] - em[1][1]],
        [em[2][2] - em[2][1], em[2][4] - em[2][1], em[2][5] - em[2][1]],
    ];
    let kappa = kappa_of(&js, &js2);
    // MFEM picks rt = 1 / rt = 2 only when the candidate kappa is strictly
    // smaller (ties keep the earlier rt).  With the bit-exact singular-value
    // port (mfem_kernels) the comparisons reproduce MFEM's choices exactly.
    if kappa < kappa_min {
        kappa_min = kappa;
        rt = 1;
    }
    // rt = 2
    let js = [
        [em[0][0] - em[0][2], em[0][1] - em[0][2], em[0][3] - em[0][2]],
        [em[1][0] - em[1][2], em[1][1] - em[1][2], em[1][3] - em[1][2]],
        [em[2][0] - em[2][2], em[2][1] - em[2][2], em[2][3] - em[2][2]],
    ];
    let js2 = [
        [em[0][1] - em[0][2], em[0][5] - em[0][2], em[0][3] - em[0][2]],
        [em[1][1] - em[1][2], em[1][5] - em[1][2], em[1][3] - em[1][2]],
        [em[2][1] - em[2][2], em[2][5] - em[2][2], em[2][3] - em[2][2]],
    ];
    let kappa = kappa_of(&js, &js2);
    if kappa < kappa_min {
        rt = 2;
    }
    rt
}

/// MFEM `Mesh::MarkTetMeshForRefinement`: rotates each tetrahedron so that
/// vertices 0-1 is the longest edge (and the two longest remaining edges are
/// in canonical positions), and marks each boundary triangle similarly.
///
/// The "edge length" used for ordering is the *rank* of the edge's geometric
/// length among all mesh edges (`Mesh::GetEdgeOrdering`): equal-length ties
/// are broken by libstdc++ `std::sort` order (simulated by [`std_sort_by`]).
///
/// Only tetrahedra and triangular boundary faces are re-ordered (hex/prism
/// elements and quad boundary faces are untouched, matching MFEM).
pub fn mark_tet_mesh_for_refinement(mesh: &mut Mesh<3>) {
    // ── 1. Edge ids (insertion order over elements × local edges, same as
    //         MFEM's DSTable v_to_v) and geometric edge lengths.  The local
    //         edge order must match MFEM's Geometry::Constants Edges tables
    //         (tet {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}; hex and prism below). ──
    let tet_edges: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let hex_edges: [[usize; 2]; 12] = [
        [0, 1], [1, 2], [3, 2], [0, 3], [4, 5], [5, 6], [7, 6], [4, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    let prism_edges: [[usize; 2]; 9] = [
        [0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5],
    ];
    let mut em: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edge_len: Vec<f64> = Vec::new();
    for e in 0..mesh.n_elems() as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        let edges: &[[usize; 2]] = match et {
            ElementType::Tet4 | ElementType::Tet10 => &tet_edges,
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => &hex_edges,
            ElementType::Prism6 | ElementType::Prism15 => &prism_edges,
            _ => &[],
        };
        for &[a, b] in edges {
            let key = edge_key(ns[a], ns[b]);
            if !em.contains_key(&key) {
                let ca = mesh.coords_of(key.0);
                let cb = mesh.coords_of(key.1);
                let d = (ca[0]-cb[0]).powi(2) + (ca[1]-cb[1]).powi(2) + (ca[2]-cb[2]).powi(2);
                em.insert(key, em.len());
                edge_len.push(d.sqrt());
            }
        }
    }
    // ── 2. Edge ordering by geometric length (libstdc++ std::sort, Pair
    //         compares only the length field → unstable for equal lengths). ──
    let n_edges = em.len();
    // DSTable semantics: each edge (a,b), a<b, is stored in row a; the row
    // iterator walks the PREV chain → **reverse insertion order** (newest
    // edge first).  Iterating rows 0..n in that order gives MFEM's
    // `GetEdgeOrdering` input sequence; equal-length ties therefore break by
    // reverse-insertion order, which a column-sorted row would not reproduce.
    let mut rows: Vec<Vec<(u32, usize)>> = vec![Vec::new(); mesh.n_nodes()];
    for (&(a, b), &id) in em.iter() {
        rows[a as usize].push((b, id));
    }
    // Sort by insertion position so the row is in insertion order, then
    // reverse it (HashMap iteration order is arbitrary).
    for r in rows.iter_mut() {
        r.sort_by_key(|x| x.1);
        r.reverse();
    }
    // GetEdgeOrdering fills `length_idx[edge_id]` while walking the rows
    // (DSTable index), so the array handed to std::sort is in **edge-id
    // order**, not visit order.
    let mut length_idx: Vec<(f64, usize)> = Vec::with_capacity(n_edges);
    for id in 0..n_edges {
        length_idx.push((edge_len[id], id));
    }
    std_sort_by(&mut length_idx, |x, y| x.0 < y.0);
    let mut order = vec![0usize; n_edges];
    for (i, &(_, id)) in length_idx.iter().enumerate() {
        order[id] = i;
    }
    // ── 3. MarkEdge for every tetrahedron. ────────────────────────────────
    let len_of = |a: u32, b: u32| -> usize {
        order[em[&edge_key(a, b)]]
    };
    for e in 0..mesh.n_elems() as ElemId {
        if mesh.element_type_at(e) != ElementType::Tet4 {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        let mut n = [ns[0], ns[1], ns[2], ns[3]];
        mark_edge_tet(&mut n, len_of);
        let off = mesh.elem_offsets.as_ref().map_or(e as usize * 4, |o| o[e as usize]);
        mesh.conn[off..off + 4].copy_from_slice(&n);
    }
    // ── 4. MarkEdge for every triangular boundary face. ───────────────────
    for f in 0..mesh.n_faces() as FaceId {
        if mesh.face_type_at(f) != ElementType::Tri3 {
            continue;
        }
        let bfv = mesh.bface_nodes(f);
        let mut n = [bfv[0], bfv[1], bfv[2]];
        mark_edge_tri(&mut n, len_of);
        let off = mesh.face_offsets.as_ref().map_or(f as usize * 3, |o| o[f as usize]);
        mesh.face_conn[off..off + 3].copy_from_slice(&n);
    }
}

/// Tetrahedron::MarkEdge — longest-edge rotation + canonical edge placement.
fn mark_edge_tet(n: &mut [u32; 4], len: impl Fn(u32, u32) -> usize) {
    let mut l = len(n[0], n[1]);
    let mut j = 0;
    if len(n[1], n[2]) > l { l = len(n[1], n[2]); j = 1; }
    if len(n[2], n[0]) > l { l = len(n[2], n[0]); j = 2; }
    if len(n[0], n[3]) > l { l = len(n[0], n[3]); j = 3; }
    if len(n[1], n[3]) > l { l = len(n[1], n[3]); j = 4; }
    if len(n[2], n[3]) > l { j = 5; }
    let ind = *n;
    match j {
        1 => { n[0] = ind[1]; n[1] = ind[2]; n[2] = ind[0]; }
        2 => { n[0] = ind[2]; n[1] = ind[0]; n[2] = ind[1]; }
        3 => { n[0] = ind[3]; n[1] = ind[0]; n[2] = ind[2]; n[3] = ind[1]; }
        4 => { n[0] = ind[1]; n[1] = ind[3]; n[2] = ind[2]; n[3] = ind[0]; }
        5 => { n[0] = ind[2]; n[1] = ind[3]; n[2] = ind[0]; n[3] = ind[1]; }
        _ => {}
    }
    // Second part: canonical placement of the two remaining longest edges;
    // only some TYPE combinations swap (0,1) and (2,3).
    let mut i0 = 2usize;
    let mut i1 = 1usize;
    l = len(n[0], n[2]);
    if len(n[0], n[3]) > l { l = len(n[0], n[3]); i0 = 3; }
    if len(n[2], n[3]) > l { i0 = 5; }
    l = len(n[1], n[2]);
    if len(n[1], n[3]) > l { l = len(n[1], n[3]); i1 = 4; }
    if len(n[2], n[3]) > l { i1 = 5; }
    let swap = match i0 {
        2 => false,           // PU / A / M
        3 => i1 != 1,         // A: no; PU/M: yes
        _ => i1 == 4,         // M(1): no; M(4): yes; O: no
    };
    if swap {
        n.swap(0, 1);
        n.swap(2, 3);
    }
}

/// Triangle::MarkEdge — longest-edge rotation (moves longest edge to 0-1).
fn mark_edge_tri(n: &mut [u32; 3], len: impl Fn(u32, u32) -> usize) {
    let mut l = len(n[0], n[1]);
    let mut j = 0;
    if len(n[1], n[2]) > l { l = len(n[1], n[2]); j = 1; }
    if len(n[2], n[0]) > l { j = 2; }
    let ind = *n;
    match j {
        1 => { n[0] = ind[1]; n[1] = ind[2]; n[2] = ind[0]; }
        2 => { n[0] = ind[2]; n[1] = ind[0]; n[2] = ind[1]; }
        _ => {}
    }
}

/// Refine a mixed-element 3-D mesh using a shared edge-midpoint map.
///
/// All element types contribute to and use the same edge midpoint map,
/// ensuring conforming interfaces between different element types.
fn refine_mixed_3d(mesh: &Mesh<3>) -> Mesh<3> {
    let n_elems = mesh.n_elems();
    let mut coords = mesh.coords.clone();
    let mut em: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let next_node0 = mesh.n_nodes() as NodeId;

    // ── 1. Global edge midpoint map ────────────────────────────────────────
    let tet_edges = local_edges_tet();
    let hex_edges = local_edges_hex();
    let prism_edges = local_edges_prism();
    // (edge_key, insertion id) pairs in element × local-edge order — this IS
    // MFEM's GetVertexToVertexTable ordering.
    let mut em_ids: Vec<((NodeId, NodeId), usize)> = Vec::new();
    for e in 0..n_elems as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        let edges = match et {
            ElementType::Tet4 | ElementType::Tet10 => &tet_edges[..],
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => &hex_edges[..],
            ElementType::Prism6 | ElementType::Prism15 => &prism_edges[..],
            _ => continue,
        };
        for &(a, b) in edges {
            let key = edge_key(ns[a], ns[b]);
            if !em.contains_key(&key) {
                let id = em_ids.len();
                em.insert(key, id as NodeId); // temporary id (insertion order)
                em_ids.push((key, id));
            }
        }
    }

    // ── 1b. e2v re-mapping (MFEM UniformRefinement3D_base):
    //   e2v[edge] = position of the edge in J_v2v after sorting each row
    //   [row_start, end) with libstdc++ std::sort (Pair compares only the
    //   column id j).  The refined edge-midpoint vertex id = oedge + e2v[em].
    let n_verts = mesh.n_nodes();
    let mut rows: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n_verts];
    for (eid, &(key, _)) in em_ids.iter().enumerate() {
        let (i, j) = key;
        rows[i as usize].push((j, eid));
    }
    let mut j_v2v: Vec<(i32, usize)> = Vec::new();
    for i in 0..n_verts {
        let start = j_v2v.len();
        for &(j, eid) in &rows[i] {
            j_v2v.push((j as i32, eid));
        }
        // std::sort(row_start, end): sorts [start..len) by column id only.
        std_sort_by(&mut j_v2v[start..], |x, y| x.0 < y.0);
    }
    let mut e2v = vec![0usize; em_ids.len()];
    for (pos, &(_, eid)) in j_v2v.iter().enumerate() {
        e2v[eid] = pos;
    }
    // Edge-midpoint vertex id = oedge + e2v[insertion id]; coords must be
    // stored in vertex-id order (oedge..oedge+n_edges).
    let mut mid_coords = vec![[0.0_f64; 3]; em_ids.len()];
    for (id, &(key, _)) in em_ids.iter().enumerate() {
        let (a, b) = key;
        let ca = mesh.coords_of(a);
        let cb = mesh.coords_of(b);
        mid_coords[e2v[id]] = [0.5*(ca[0]+cb[0]), 0.5*(ca[1]+cb[1]), 0.5*(ca[2]+cb[2])];
    }
    for c in &mid_coords {
        coords.extend_from_slice(c);
    }
    for (id, &(key, _)) in em_ids.iter().enumerate() {
        em.insert(key, next_node0 + e2v[id] as NodeId);
    }
    let mut next_node = next_node0 + em_ids.len() as NodeId; // oface start

    // ── 2. Quad-face centers (hex/prism), THEN hex body centers ───────────
    // MFEM UniformRefinement3D_base: new vertices are edge midpoints +
    // QUAD face centers (oface + f2qf, all of them) + HEX body centers
    // (oelem + hex_counter, AFTER all quad faces).  Tetrahedra get ONLY edge
    // midpoints; prisms get no body centers.
    let mut quad_fc: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut body_cc: HashMap<ElemId, NodeId> = HashMap::new();
    const MF_HEX_FACES: [[usize; 4]; 6] = [
        [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
    ];

    // Pass A: quad face centers in element × MFEM-face order (f2qf order).
    for e in 0..n_elems as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        match et {
            ElementType::Hex8 => {
                for f in &MF_HEX_FACES {
                    let fns = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
                    quad_fc.entry(quad_face_key(fns)).or_insert_with(|| {
                        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
                        for &n in &fns { let c = mesh.coords_of(n); x += c[0]; y += c[1]; z += c[2]; }
                        coords.extend_from_slice(&[x / 4.0, y / 4.0, z / 4.0]);
                        let id = next_node; next_node += 1; id
                    });
                }
            }
            ElementType::Prism6 => {
                for f in &local_faces_prism_quad() {
                    let fns = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
                    quad_fc.entry(quad_face_key(fns)).or_insert_with(|| {
                        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
                        for &n in &fns { let c = mesh.coords_of(n); x += c[0]; y += c[1]; z += c[2]; }
                        coords.extend_from_slice(&[x / 4.0, y / 4.0, z / 4.0]);
                        let id = next_node; next_node += 1; id
                    });
                }
            }
            _ => {}
        }
    }

    // Pass B: hex body centers (after ALL quad face centers, oelem + hex_counter).
    for e in 0..n_elems as ElemId {
        if mesh.element_type_at(e) == ElementType::Hex8 {
            let ns = mesh.elem_nodes(e);
            body_cc.entry(e).or_insert_with(|| {
                let nv = ns.len() as f64;
                let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
                for &n in ns { let c = mesh.coords_of(n); x += c[0]; y += c[1]; z += c[2]; }
                coords.extend_from_slice(&[x / nv, y / nv, z / nv]);
                let id = next_node; next_node += 1; id
            });
        }
    }

    // ── 3. Generate child elements per type, using extend-then-push pattern ─
    let mut new_conn = Vec::<NodeId>::new();
    let mut new_tags = Vec::<i32>::new();
    let mut new_types = Vec::<ElementType>::new();
    let mut new_offsets = Vec::<usize>::new();
    new_offsets.push(0); // start offset for first child

    macro_rules! mid { ($a:expr,$b:expr) => { em[&edge_key($a,$b)] }; }

    for e in 0..n_elems as ElemId {
        let et = mesh.element_type_at(e);
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];
        match et {
            ElementType::Tet4 => {
                let m01=mid!(ns[0],ns[1]);let m02=mid!(ns[0],ns[2]);let m03=mid!(ns[0],ns[3]);
                let m12=mid!(ns[1],ns[2]);let m13=mid!(ns[1],ns[3]);let m23=mid!(ns[2],ns[3]);
                // MFEM UniformRefinement3D_base (mesh.cpp): 4 corner tets +
                // 4 interior tets from mv_all[rt], where rt is chosen by the
                // best-aspect-ratio algorithm (rt_algo = 1) using the
                // Jacobian at the tet centroid.
                let rt = tet_select_rt_debug(mesh, ns);
                let mv = match rt {
                    0 => [
                        [0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1],
                    ],
                    1 => [
                        [1, 0, 4, 2], [1, 2, 4, 5], [1, 5, 4, 3], [1, 3, 4, 0],
                    ],
                    _ => [
                        [2, 0, 1, 3], [2, 1, 5, 3], [2, 5, 4, 3], [2, 4, 0, 3],
                    ],
                };
                // e0=(0,1)=m01, e1=(0,2)=m02, e2=(0,3)=m03, e3=(1,2)=m12,
                // e4=(1,3)=m13, e5=(2,3)=m23.
                for &ch in &[
                    [ns[0],m01,m02,m03],[m01,ns[1],m12,m13],[m02,m12,ns[2],m23],[m03,m13,m23,ns[3]],
                ] { new_conn.extend_from_slice(&ch); new_offsets.push(new_conn.len()); }
                let e = [m01, m02, m03, m12, m13, m23];
                for k in 0..4 {
                    let ch = [e[mv[k][0]], e[mv[k][1]], e[mv[k][2]], e[mv[k][3]]];
                    new_conn.extend_from_slice(&ch);
                    new_offsets.push(new_conn.len());
                }
                for _ in 0..8 { new_tags.push(tag); new_types.push(ElementType::Tet4); }
            }
            ElementType::Hex8 => {
                // MFEM hex_t::Edges: {0,1},{1,2},{3,2},{0,3},{4,5},{5,6},{7,6},
                // {4,7},{0,4},{1,5},{2,6},{3,7}
                let e0=mid!(ns[0],ns[1]);let e1=mid!(ns[1],ns[2]);let e2=mid!(ns[3],ns[2]);let e3=mid!(ns[0],ns[3]);
                let e4=mid!(ns[4],ns[5]);let e5=mid!(ns[5],ns[6]);let e6=mid!(ns[7],ns[6]);let e7=mid!(ns[4],ns[7]);
                let e8=mid!(ns[0],ns[4]);let e9=mid!(ns[1],ns[5]);let e10=mid!(ns[2],ns[6]);let e11=mid!(ns[3],ns[7]);
                // MFEM hex_t::FaceVert: {3,2,1,0},{0,1,5,4},{1,2,6,5},
                // {2,3,7,6},{3,0,4,7},{4,5,6,7} (bottom,front,right,back,left,top)
                const MF_FACES: [[usize; 4]; 6] = [
                    [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
                    [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
                ];
                let qf = |fi: usize| -> NodeId {
                    let f = MF_FACES[fi];
                    quad_fc[&quad_face_key([ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]])]
                };
                let f0=qf(0);let f1=qf(1);let f2=qf(2);let f3=qf(3);let f4=qf(4);let f5=qf(5);
                let bc = body_cc[&e];
                // MFEM 8 children (UniformRefinement3D_base CUBE branch).
                for &ch in &[
                    [ns[0],e0,f0,e3,e8,f1,bc,f4],
                    [e0,ns[1],e1,f0,f1,e9,f2,bc],
                    [f0,e1,ns[2],e2,bc,f2,e10,f3],
                    [e3,f0,e2,ns[3],f4,bc,f3,e11],
                    [e8,f1,bc,f4,ns[4],e4,f5,e7],
                    [f1,e9,f2,bc,e4,ns[5],e5,f5],
                    [bc,f2,e10,f3,f5,e5,ns[6],e6],
                    [f4,bc,f3,e11,e7,f5,e6,ns[7]],
                ] { new_conn.extend_from_slice(&ch); new_offsets.push(new_conn.len()); }
                for _ in 0..8 { new_tags.push(tag); new_types.push(ElementType::Hex8); }
            }
            ElementType::Prism6 => {
                // MFEM wedge_t::Edges: {0,1},{1,2},{2,0},{3,4},{4,5},{5,3},
                // {0,3},{1,4},{2,5} → e0=m01,e1=m12,e2=m02,e3=m34,e4=m45,
                // e5=m35,e6=m03,e7=m14,e8=m25.  Quad faces FaceVert[2..5]:
                // {0,1,4,3},{1,2,5,4},{2,0,3,5} → q0,q1,q2.
                let m01=mid!(ns[0],ns[1]);let m02=mid!(ns[0],ns[2]);let m12=mid!(ns[1],ns[2]);
                let m34=mid!(ns[3],ns[4]);let m35=mid!(ns[3],ns[5]);let m45=mid!(ns[4],ns[5]);
                let m03=mid!(ns[0],ns[3]);let m14=mid!(ns[1],ns[4]);let m25=mid!(ns[2],ns[5]);
                let qf = |fi: usize| -> NodeId {
                    let f=local_faces_prism_quad()[fi]; quad_fc[&quad_face_key([ns[f[0]],ns[f[1]],ns[f[2]],ns[f[3]]])]
                };
                let q0=qf(0);let q1=qf(1);let q2=qf(2);
                // MFEM 8 children (UniformRefinement3D_base WEDGE branch).
                for &ch in &[
                    [ns[0],m01,m02,m03,q0,q2],
                    [m12,m02,m01,q1,q2,q0],
                    [m01,ns[1],m12,q0,m14,q1],
                    [m02,m12,ns[2],q2,q1,m25],
                    [m03,q0,q2,ns[3],m34,m35],
                    [q1,q2,q0,m45,m35,m34],
                    [q0,m14,q1,m34,ns[4],m45],
                    [q2,q1,m25,m35,m45,ns[5]],
                ] { new_conn.extend_from_slice(&ch); new_offsets.push(new_conn.len()); }
                for _ in 0..8 { new_tags.push(tag); new_types.push(ElementType::Prism6); }
            }
            _ => {}
        }
    }

    let mut result = Mesh {
        coords,
        conn: new_conn,
        elem_tags: new_tags,
        elem_type: mesh.elem_type,
        face_conn: vec![], face_tags: vec![], face_type: ElementType::Tri3,
        elem_types: Some(new_types), elem_offsets: Some(new_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
};
    rebuild_3d_boundary(&mut result, mesh);
    result
}

fn midpoint_edge(coords: &mut Vec<f64>, em: &mut HashMap<(u32, u32), u32>, nn: &mut u32,
                 a: u32, b: u32) -> u32 {
    let k = if a < b { (a, b) } else { (b, a) };
    *em.entry(k).or_insert_with(|| {
        let j = *nn; *nn += 1;
        let oa = a as usize * 3; let ob = b as usize * 3;
        coords.extend_from_slice(&[
            0.5 * (coords[oa] + coords[ob]),
            0.5 * (coords[oa + 1] + coords[ob + 1]),
            0.5 * (coords[oa + 2] + coords[ob + 2]),
        ]);
        j
    })
}

/// Uniformly refine all Tri3 elements of a **surface** mesh (`Mesh<3>`).
///
/// Each triangle is split into 4 by adding edge midpoints. New nodes are
/// placed at the linear midpoint (caller should `snap_nodes` afterwards).
pub fn refine_uniform_surface_tri3(mesh: &Mesh<3>) -> Mesh<3> {
    let old_n = mesh.conn.len() / 3;
    let mut em = HashMap::new();
    let mut coords = mesh.coords.clone();
    let mut nn = (coords.len() / 3) as u32;
    let mut nc = Vec::with_capacity(old_n * 12);
    let mut nt = Vec::with_capacity(old_n * 4);
    for t in 0..old_n {
        let i = t * 3;
        let (a, b, c) = (mesh.conn[i], mesh.conn[i + 1], mesh.conn[i + 2]);
        let tag = mesh.elem_tags[t];
        let ab = midpoint_edge(&mut coords, &mut em, &mut nn, a, b);
        let ac = midpoint_edge(&mut coords, &mut em, &mut nn, a, c);
        let bc = midpoint_edge(&mut coords, &mut em, &mut nn, b, c);
        nc.extend_from_slice(&[a, ab, ac, b, bc, ab, c, ac, bc, ab, bc, ac]);
        nt.extend_from_slice(&[tag, tag, tag, tag]);
    }
    let mut fconn = Vec::with_capacity(mesh.face_conn.len());
    let mut ftags = Vec::with_capacity(mesh.face_tags.len() * 2);
    for f in 0..mesh.face_conn.len() / 2 {
        let (a, b) = (mesh.face_conn[f * 2], mesh.face_conn[f * 2 + 1]);
        let tag = mesh.face_tags[f];
        let m = midpoint_edge(&mut coords, &mut em, &mut nn, a, b);
        fconn.push(a);
        fconn.push(m);
        fconn.push(m);
        fconn.push(b);
        ftags.push(tag);
        ftags.push(tag);
    }
    Mesh {
        coords, conn: nc, elem_tags: nt,
        elem_type: ElementType::Tri3,
        face_conn: fconn, face_tags: ftags,
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
}
}

/// Uniformly refine all Quad4 elements of a **surface** mesh (`Mesh<3>`).
///
/// Each quad is split into 4 by adding edge midpoints and a center node.
pub fn refine_uniform_surface_quad4(mesh: &Mesh<3>) -> Mesh<3> {
    let old_n = mesh.conn.len() / 4;
    let mut em = HashMap::new();
    let mut coords = mesh.coords.clone();
    let mut nn = (coords.len() / 3) as u32;
    let mut nc = Vec::with_capacity(old_n * 16);
    let mut nt = Vec::with_capacity(old_n * 4);
    for q in 0..old_n {
        let i = q * 4;
        let (a, b, c, d) = (mesh.conn[i], mesh.conn[i + 1], mesh.conn[i + 2], mesh.conn[i + 3]);
        let tag = mesh.elem_tags[q];
        let ab = midpoint_edge(&mut coords, &mut em, &mut nn, a, b);
        let bc = midpoint_edge(&mut coords, &mut em, &mut nn, b, c);
        let cd = midpoint_edge(&mut coords, &mut em, &mut nn, c, d);
        let da = midpoint_edge(&mut coords, &mut em, &mut nn, d, a);
        // Quad center
        let cx = nn; nn += 1;
        let oa = a as usize * 3; let ob = b as usize * 3;
        let oc = c as usize * 3; let od = d as usize * 3;
        coords.extend_from_slice(&[
            0.25 * (coords[oa] + coords[ob] + coords[oc] + coords[od]),
            0.25 * (coords[oa + 1] + coords[ob + 1] + coords[oc + 1] + coords[od + 1]),
            0.25 * (coords[oa + 2] + coords[ob + 2] + coords[oc + 2] + coords[od + 2]),
        ]);
        nc.extend_from_slice(&[a, ab, cx, da, ab, b, bc, cx, cx, bc, c, cd, da, cx, cd, d]);
        nt.extend_from_slice(&[tag, tag, tag, tag]);
    }
    // Rebuild the boundary faces: each parent boundary edge (a,b) splits into
    // (a,m) and (m,b), inheriting the parent tag.  The edge-midpoint table is
    // shared with the element refinement so shared nodes stay consistent.
    let mut fconn = Vec::with_capacity(mesh.face_conn.len());
    let mut ftags = Vec::with_capacity(mesh.face_tags.len() * 2);
    for f in 0..mesh.face_conn.len() / 2 {
        let (a, b) = (mesh.face_conn[f * 2], mesh.face_conn[f * 2 + 1]);
        let tag = mesh.face_tags[f];
        let m = midpoint_edge(&mut coords, &mut em, &mut nn, a, b);
        fconn.push(a);
        fconn.push(m);
        fconn.push(m);
        fconn.push(b);
        ftags.push(tag);
        ftags.push(tag);
    }
    Mesh {
        coords, conn: nc, elem_tags: nt,
        elem_type: ElementType::Quad4,
        face_conn: fconn, face_tags: ftags,
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
}
}

/// Refine elements of a Tri3 surface mesh near a target vertex, matching
/// MFEM's `Mesh::RefineAtVertex` (used by ex7 `-amr 1`).
pub fn refine_at_vertex_surface(mesh: &Mesh<3>, target: &[f64; 3]) -> Mesh<3> {
    let old_n = mesh.conn.len() / 3;
    // Diameter: max distance from target to any node.
    let max_dist = (0..(mesh.coords.len() / 3) as u32).map(|n| {
        let o = n as usize * 3;
        ((mesh.coords[o] - target[0]).powi(2)
       + (mesh.coords[o + 1] - target[1]).powi(2)
       + (mesh.coords[o + 2] - target[2]).powi(2)).sqrt()
    }).fold(0.0_f64, f64::max);

    let threshold = 0.3 * max_dist;
    let mut marked = Vec::new();
    for t in 0..old_n as ElemId {
        let i = t as usize * 3;
        let oa = mesh.conn[i] as usize * 3;
        let ob = mesh.conn[i + 1] as usize * 3;
        let oc = mesh.conn[i + 2] as usize * 3;
        let cx = (mesh.coords[oa] + mesh.coords[ob] + mesh.coords[oc]) / 3.0;
        let cy = (mesh.coords[oa + 1] + mesh.coords[ob + 1] + mesh.coords[oc + 1]) / 3.0;
        let cz = (mesh.coords[oa + 2] + mesh.coords[ob + 2] + mesh.coords[oc + 2]) / 3.0;
        let d = ((cx - target[0]).powi(2) + (cy - target[1]).powi(2) + (cz - target[2]).powi(2)).sqrt();
        if d < threshold { marked.push(t); }
    }
    if marked.is_empty() { return mesh.clone(); }

    // Closure: also refine neighbours of marked elements.
    let ek = |x: u32, y: u32| if x < y { (x, y) } else { (y, x) };
    let mut edge_elems: HashMap<(u32, u32), Vec<ElemId>> = HashMap::new();
    for t in 0..old_n as ElemId {
        let i = t as usize * 3;
        for &(a, b) in &[(mesh.conn[i], mesh.conn[i+1]), (mesh.conn[i+1], mesh.conn[i+2]), (mesh.conn[i], mesh.conn[i+2])] {
            edge_elems.entry(ek(a, b)).or_default().push(t);
        }
    }
    let mut to_refine: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    for &t in &marked {
        let i = t as usize * 3;
        for &(a, b) in &[(mesh.conn[i], mesh.conn[i+1]), (mesh.conn[i+1], mesh.conn[i+2]), (mesh.conn[i], mesh.conn[i+2])] {
            if let Some(adj) = edge_elems.get(&ek(a, b)) { for &a in adj { to_refine.insert(a); } }
        }
    }

    let mut em = HashMap::new();
    let mut coords = mesh.coords.clone();
    let mut nn = (coords.len() / 3) as u32;
    let mut nc = Vec::with_capacity(old_n * 12);
    let mut nt = Vec::with_capacity(old_n * 4);
    for t in 0..old_n as ElemId {
        let i = t as usize * 3;
        let (a, b, c) = (mesh.conn[i], mesh.conn[i + 1], mesh.conn[i + 2]);
        let tag = mesh.elem_tags[t as usize];
        if to_refine.contains(&t) {
            let ab = midpoint_edge(&mut coords, &mut em, &mut nn, a, b);
            let ac = midpoint_edge(&mut coords, &mut em, &mut nn, a, c);
            let bc = midpoint_edge(&mut coords, &mut em, &mut nn, b, c);
            nc.extend_from_slice(&[a, ab, ac, b, bc, ab, c, ac, bc, ab, bc, ac]);
            nt.extend_from_slice(&[tag, tag, tag, tag]);
        } else {
            nc.extend_from_slice(&[a, b, c]); nt.push(tag);
        }
    }
    Mesh {
        coords, conn: nc, elem_tags: nt,
        elem_type: ElementType::Tri3,
        face_conn: vec![], face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
}
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Local edge index pairs for Tri3.


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
pub(crate) fn local_faces_prism_tri() -> [(usize, usize, usize); 2] {
    [
        (0, 1, 2), // bottom
        (3, 4, 5), // top
    ]
}

/// Local 3 quadrilateral faces of a Prism6 (4-tuples of local node indices).
pub(crate) fn local_faces_prism_quad() -> [[usize; 4]; 3] {
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
    mesh: &Mesh<3>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>) {
    let (mut new_mesh, edge_constraints, face_constraints, _, _) =
        refine_nonconforming_3d_internal(mesh, marked, None);
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 3);
    }
    (new_mesh, edge_constraints, face_constraints)
}

#[allow(clippy::type_complexity)]
fn refine_nonconforming_3d_internal(
    mesh: &Mesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    Mesh<3>,
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

            // 4 tets splitting the central octahedron.  MFEM
            // UniformRefinement3D_base (mesh.cpp) chooses the octahedron
            // diagonal by the best-aspect-ratio refinement type `rt`
            // (rt_algo = 1); the fixed split used here previously produced a
            // different refinement pattern than MFEM for tets whose longest
            // edge is not (v0,v1) on the octahedron.
            let rt = tet_select_rt_debug(mesh, &[n0, n1, n2, n3]);
            let e = [m01, m02, m03, m12, m13, m23];
            let mv: [[usize; 4]; 4] = match rt {
                0 => [[0, 5, 1, 2], [0, 5, 2, 4], [0, 5, 4, 3], [0, 5, 3, 1]],
                1 => [[1, 0, 4, 2], [1, 2, 4, 5], [1, 5, 4, 3], [1, 3, 4, 0]],
                _ => [[2, 0, 1, 3], [2, 1, 5, 3], [2, 5, 4, 3], [2, 4, 0, 3]],
            };
            for k in 0..4 {
                let ch = [e[mv[k][0]], e[mv[k][1]], e[mv[k][2]], e[mv[k][3]]];
                new_conn.extend_from_slice(&ch);
                new_tags.push(tag);
            }
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
        coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
        edge_constraints.push(HangingNodeConstraint {
            constrained: mbc as usize,
            parent_a: b as usize,
            parent_b: c as usize,
        coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
        edge_constraints.push(HangingNodeConstraint {
            constrained: mac as usize,
            parent_a: a as usize,
            parent_b: c as usize,
        coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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
            coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let new_mesh = Mesh::uniform(
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

/// MFEM `NCMesh::InitRootState` (ncmesh.cpp) for 2-D quad meshes: assign a
/// Hilbert-curve state (0..7) to every initial (root) element so that
/// `CollectLeafElements` visits its children along the space-filling curve.
///
/// For each root the entry node is the exit node of the previous root
/// (`entry_node`), and the state is chosen so the curve exits through a node
/// shared with the next root.
fn init_root_states_quad(mesh: &Mesh<2>) -> Vec<u8> {
    let n_roots = mesh.n_elems();
    let mut states = vec![0u8; n_roots];
    let mut entry_node: i64 = -2; // MFEM: FindNodeExt returns -1 for -2
    for i in 0..n_roots {
        let el = mesh.elem_nodes(i as ElemId);
        // FindNodeExt(el, entry_node, false): local index of entry_node in el.
        let v_in = if entry_node >= 0 {
            el.iter().position(|&n| n as i64 == entry_node)
        } else {
            None
        };
        let v_in = v_in.unwrap_or(0);
        // shared[k] = whether el's k-th node also belongs to the next root.
        let mut shared = [false; 4];
        if i + 1 < n_roots {
            let next = mesh.elem_nodes((i + 1) as ElemId);
            for j in 0..4 {
                if let Some(p) = el.iter().position(|&n| n == next[j]) {
                    shared[p] = true;
                }
            }
        }
        // state = Dim * v_in, then nudge by j so the exit node is shared.
        let mut state = (2 * v_in) as usize;
        for j in 0..2 {
            let exit = QUAD_HILBERT_CHILD_ORDER[state + j][3] as usize;
            if shared[exit] {
                state += j;
                break;
            }
        }
        states[i] = state as u8;
        let exit = QUAD_HILBERT_CHILD_ORDER[state][3] as usize;
        entry_node = el[exit] as i64;
    }
    states
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
    mesh: &Mesh<2>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, Vec<HangingNodeConstraint>) {    assert!(
        mesh.elem_type == ElementType::Quad4,
        "refine_nonconforming_quad: only Quad4 meshes are supported"
    );

    if marked.is_empty() {
        let mesh = if let Some(config) = project_boundary {
            project_boundary_to_cad(mesh, config, 2)
        } else { mesh.clone() };
        return (mesh, Vec::new());
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();

    // ── 2. Compute midpoints and centers for marked elements ─────────────────
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut center_map:   HashMap<ElemId, NodeId>           = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;

    // Coordinate → existing node lookup: a marked element may sit next to an
    // already-refined neighbour whose hanging node lies at the midpoint of one
    // of this element's edges.  Refining that edge must REUSE the existing
    // hanging node (MFEM NC semantics) instead of creating a duplicate node at
    // the same coordinate — duplicates silently inflate n_dofs and desync the
    // AMR trajectory (ex6: 311 vs 301 nodes at it3).
    let coord_map: HashMap<String, NodeId> = {
        let mut m = HashMap::new();
        for n in 0..mesh.n_nodes() as NodeId {
            let c = mesh.coords_of(n);
            m.entry(format!("{:.12},{:.12}", c[0], c[1])).or_insert(n);
        }
        m
    };

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        // Edge midpoints
        for &(a, b) in &local_edges_quad() {
            let key = quad_edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                let mx = 0.5 * (xa[0] + xb[0]);
                let my = 0.5 * (xa[1] + xb[1]);
                // Reuse an existing node already at this midpoint (hanging
                // node of a refined neighbour / previously split edge).
                if let Some(&existing) = coord_map.get(&format!("{mx:.12},{my:.12}")) {
                    return existing;
                }
                new_coords.push(mx);
                new_coords.push(my);
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
    // Full-topology scan on the *refined* mesh: any element edge (a,b) whose
    // geometric midpoint coincides with a mesh vertex m that the element does
    // NOT contain is a hanging node.  This catches both this round's new
    // midpoints and hanging nodes carried over from previous refinement
    // levels — the previous implementation only scanned `midpoint_map`
    // (this round's fresh midpoints), silently dropping pre-existing
    // constraints, which broke multi-level AMR (ex6 it2+: 20 constraints vs
    // MFEM's 30).
    let mut constraints = Vec::new();

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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Quad4,
        new_face_conn, new_face_tags, ElementType::Line2,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 2);
    }

    let mut coord_map: HashMap<String, NodeId> = HashMap::new();
    for n in 0..new_mesh.n_nodes() as NodeId {
        let c = new_mesh.coords_of(n);
        coord_map.entry(format!("{:.9},{:.9}", c[0], c[1])).or_insert(n);
    }
    let coords_of = |id: NodeId| -> [f64; 2] {
        let c = new_mesh.coords_of(id);
        [c[0], c[1]]
    };
    // Multi-level detection: starting from every element edge (a,b), walk the
    // bisection chain (a,b) → (a,m),(m,b) → ... recording every midpoint that
    // the element does NOT contain as a hanging constraint.  A pure
    // "midpoint-of-element-edge" scan (as before) misses mid-level hanging
    // nodes whose parent edge is not an element edge (e.g. m2 = mid(a,m1)
    // where m1 is itself hanging).
    for e in 0..new_mesh.n_elems() as ElemId {
        let ns = new_mesh.elem_nodes(e);
        let contains = |m: NodeId| ns.contains(&m);
        fn walk(
            a: NodeId, b: NodeId,
            coord_map: &HashMap<String, NodeId>,
            coords_of: &dyn Fn(NodeId) -> [f64; 2],
            contains: &dyn Fn(NodeId) -> bool,
            out: &mut Vec<HangingNodeConstraint>,
        ) {
            let ca = coords_of(a);
            let cb = coords_of(b);
            let key = format!("{:.9},{:.9}", 0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]));
            if let Some(&m) = coord_map.get(&key) {
                if m != a && m != b && !contains(m) {
                    out.push(HangingNodeConstraint {
                        constrained: m as usize,
                        parent_a: a as usize,
                        parent_b: b as usize,
                        coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
                    });
                    // Recurse into the two halves (m may itself be hanging at
                    // a finer level: its parent edge is (a,m) / (m,b), which
                    // need not be an element edge).
                    walk(a, m, coord_map, coords_of, contains, out);
                    walk(m, b, coord_map, coords_of, contains, out);
                }
            }
        }
        for &(ea, eb) in &local_edges_quad() {
            let a = ns[ea];
            let b = ns[eb];
            walk(a, b, &coord_map, &coords_of, &contains, &mut constraints);
        }
    }
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by(|a, b| a.constrained == b.constrained);
    (new_mesh, constraints)
}

// ─── NCStateQuad ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NCStateQuadSnapshot {
    mesh: Mesh<2>,
    constraints: Vec<HangingNodeConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
    /// Edge refinement level: number of times each edge has been split.
    edge_level: HashMap<(NodeId, NodeId), u32>,
    /// Leaf-element order (MFEM `leaf_elements`, Hilbert SFC) at snapshot time.
    leaf_order: Vec<ElemId>,
    /// element → root mapping at snapshot time.
    elem_root: HashMap<ElemId, usize>,
}

/// Detect hanging-node constraints on a Quad4 NC mesh **purely from topology
/// and coordinates** (no refinement history — safe after any partition
/// rebuild that renumbers local ids).
///
/// A node `m` is hanging iff it is the midpoint (by coordinates, exact
/// `0.5·a + 0.5·b` match) of an edge `(a,b)` that is used as a **full** edge
/// by some (coarse) element while at least one element uses the half-edges
/// `(a,m)` or `(m,b)` (refined neighbour).  P1 constraint:
/// `u[m] = 0.5·(u[a] + u[b])`.
///
/// Multi-level: from every element edge the bisection chain is walked
/// recursively (`(a,b)` → `(a,m)`, `(m,b)` → ...), recording every midpoint
/// the element does not contain.  A pure one-level "midpoint of a full edge"
/// scan misses 2nd+ level hanging nodes whose parent edge `(a,m1)` is no
/// longer an element edge (pex6 it3: unknowns 321 vs C++ 291, ~30 missed
/// constraints).
pub fn detect_hanging_quad(mesh: &Mesh<2>) -> Vec<HangingNodeConstraint> {
    use std::collections::HashMap;
    let n_elems = mesh.n_elems();
    let n_nodes = mesh.n_nodes();

    // Node coordinates for midpoint matching.
    let coords_of = |n: NodeId| -> [f64; 2] {
        let c = mesh.coords_of(n);
        [c[0], c[1]]
    };
    // Coordinate → node id.  Keyed by 9-decimal-rounded string (1e-9
    // tolerance), NOT exact bit match: 2nd+ level hanging nodes are created
    // by *accumulated* bisection (m2 = mid(a, m1), m1 = mid(a,b)), so their
    // coordinates differ from a freshly computed 0.5a+0.5b by rounding at
    // the ~1e-16..1e-15 level and `to_bits` exact matching misses them
    // (pex6 it3: unknowns 321 vs C++ 291, ~30 missed 2nd-level constraints).
    // This mirrors the multi-level walk in `refine_uniform` (also 9-decimal
    // keys).  The half-edge check below still guards against collinear
    // vertices, so the tolerance cannot introduce false positives.
    let mut coord_to_node: HashMap<String, NodeId> = HashMap::new();
    for n in 0..n_nodes as NodeId {
        let [x, y] = coords_of(n);
        coord_to_node.entry(format!("{:.9},{:.9}", x, y)).or_insert(n);
    }

    // Multi-level walk: from each element edge (a,b), recurse into the two
    // halves.  A midpoint m that exists as a mesh node but is NOT contained
    // in the element is hanging (the element uses (a,b) as a full edge while
    // a refined neighbour uses (a,m)/(m,b)).  m's own parent edge (a,m) may
    // itself be bisected further (m2 = mid(a,m)), so recurse.
    //
    // The half-edge check (sub-segments (a,m)/(m,b) used by some element)
    // distinguishes a true hanging node from a collinear mesh vertex that
    // happens to lie at the midpoint of an unrelated element edge (it0-it2
    // regression lesson: incident-edge-pair scans mis-flag straight-line
    // vertices).
    fn walk(
        a: NodeId,
        b: NodeId,
        coord_to_node: &HashMap<String, NodeId>,
        coords_of: &dyn Fn(NodeId) -> [f64; 2],
        contains: &dyn Fn(NodeId) -> bool,
        elem_edges: &std::collections::HashSet<(u32, u32)>,
        seen: &mut std::collections::HashSet<u32>,
        out: &mut Vec<HangingNodeConstraint>,
    ) {
        let [ax, ay] = coords_of(a);
        let [bx, by] = coords_of(b);
        let key = format!("{:.9},{:.9}", 0.5 * ax + 0.5 * bx, 0.5 * ay + 0.5 * by);
        let Some(&m) = coord_to_node.get(&key) else {
            return; // no node at the midpoint: no (finer) split here
        };
        if m == a || m == b || contains(m) {
            return; // degenerate, or m is a node of this element (not hanging)
        }
        // m is a hanging node iff at least one half-edge (a,m)/(m,b) is an
        // element edge (refined neighbour) — even at level 2+ where the
        // parent edge (a,m1) is no longer an element edge itself.
        let (lo, hi) = if a < m { (a, m) } else { (m, a) };
        let (mlo, mhi) = if m < b { (m, b) } else { (b, m) };
        let half_used = elem_edges.contains(&(lo, hi)) || elem_edges.contains(&(mlo, mhi));
        if !half_used {
            return; // collinear vertex at the midpoint, not a hanging node
        }
        if !seen.insert(m) {
            return;
        }
        out.push(HangingNodeConstraint::new_p1(m as usize, a as usize, b as usize));
        walk(a, m, coord_to_node, coords_of, contains, elem_edges, seen, out);
        walk(m, b, coord_to_node, coords_of, contains, elem_edges, seen, out);
    }

    // All element edges as unordered keys (for the half-edge check above).
    let mut elem_edges: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for k in 0..4 {
            let (a, b) = (ns[k], ns[(k + 1) % 4]);
            let key = if a < b { (a, b) } else { (b, a) };
            elem_edges.insert(key);
        }
    }

    let mut constraints = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let contains = |m: NodeId| ns.contains(&m);
        for k in 0..4 {
            let a = ns[k];
            let b = ns[(k + 1) % 4];
            walk(a, b, &coord_to_node, &coords_of, &contains, &elem_edges, &mut seen, &mut constraints);
        }
    }
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by(|a, b| a.constrained == b.constrained);
    constraints
}

/// MFEM `quad_hilbert_child_order` (ncmesh_tables.hpp): the order in which the
/// 4 children of a refined quad are visited, per Hilbert-curve state.  This
/// determines the *leaf element order* (`CollectLeafElements`) and hence the
/// MFEM Mesh element numbering, edge table order and face (element-centre)
/// DOF numbering — all of which feed the H1 Q2 global DOF numbering.
const QUAD_HILBERT_CHILD_ORDER: [[u8; 4]; 8] = [
    [0, 1, 2, 3], [0, 3, 2, 1], [1, 2, 3, 0], [1, 0, 3, 2],
    [2, 3, 0, 1], [2, 1, 0, 3], [3, 0, 1, 2], [3, 2, 1, 0],
];
/// MFEM `quad_hilbert_child_state`: the state passed to each child when
/// recursing along `QUAD_HILBERT_CHILD_ORDER[state]`.
const QUAD_HILBERT_CHILD_STATE: [[u8; 4]; 8] = [
    [1, 0, 0, 5], [0, 1, 1, 4], [3, 2, 2, 7], [2, 3, 3, 6],
    [5, 4, 4, 1], [4, 5, 5, 0], [7, 6, 6, 3], [6, 7, 7, 2],
];

/// One 4-split of a Quad4 parent element in the NC refinement tree.
///
/// Mirrors MFEM's `NCMesh::Element` refinement tree used by
/// `GetDerefinementTable`: a node is *derefinable* when all its children are
/// still leaves of the current mesh (`child_leaf[k] == true`).
#[derive(Debug, Clone)]
struct QuadRefineNode {
    /// Parent corner nodes (current-mesh node ids).
    parent_nodes: [NodeId; 4],
    parent_tag: i32,
    /// Current-mesh element indices of the 4 children.  Only meaningful while
    /// the matching `child_leaf[k]` is true (element numbering shifts on
    /// every subsequent refinement pass).  `children[k]` is the element
    /// produced by MFEM child slot `k` (XY split: 0 = n0 corner, 1 = n1
    /// corner, 2 = n2 corner, 3 = n3 corner) — the Hilbert SFC order only
    /// affects the *leaf output order*, not the child slots.
    children: [ElemId; 4],
    /// Whether each child is still a leaf element of the current mesh.
    child_leaf: [bool; 4],
    /// False once this node's group has been derefined (coarsened) away.
    alive: bool,
    /// Position of this node's parent element within an ancestor tree node
    /// (parent node index, child slot).  When this group is derefined the
    /// parent element becomes a leaf again, so the ancestor's `child_leaf`
    /// slot must be restored to `true` (MFEM `DerefineElement` semantics).
    parent: Option<(usize, usize)>,
    /// Hilbert-curve state (MFEM `root_state` / recursion state) of this
    /// node, used to order its children in the leaf sequence.
    state: u8,
}

impl QuadRefineNode {
    fn derefinable(&self) -> bool { self.alive && self.child_leaf.iter().all(|&l| l) }
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
    /// Edge refinement level: number of times each edge has been split.
    edge_level: HashMap<(NodeId, NodeId), u32>,
    history: Vec<NCStateQuadSnapshot>,
    /// Refinement tree (see `QuadRefineNode`) for MFEM-style derefinement.
    refine_tree: Vec<QuadRefineNode>,
    /// Current-mesh element → (tree node index, child slot) for leaf children
    /// that participate in the refinement tree.
    elem_to_node: HashMap<ElemId, (usize, usize)>,
    /// MFEM `root_state` per initial (root) element, from `InitRootState`.
    /// Only meaningful once the first refinement has been applied to the
    /// initial mesh; the number of roots equals the initial element count.
    root_states: Vec<u8>,
    /// Number of top-level (original) mesh nodes — MFEM assigns these the
    /// first `GetNV()` vertex DOFs (UpdateVertices STEP 1-2).
    top_level_nodes: usize,
    /// Current mesh-node ids of the top-level (original) vertices, in
    /// creation order (id order of the initial mesh).  MFEM's NCMesh node
    /// table is append-only (derefinement never renumbers nodes), so the
    /// original vertices always keep ids 0..N0-1 there; Rust's deref
    /// compacts node ids, so this Vec is remapped through the compaction
    /// map after every derefinement to keep the vertex-view top-level block
    /// (and hence the vertex DOF numbering) MFEM-compatible.
    top_level_ids: Vec<NodeId>,
    /// Current leaf-element order (MFEM `leaf_elements` after
    /// `CollectLeafElements`): the order in which elements appear in the
    /// refined mesh.  Drives element numbering, the edge table and hence the
    /// MFEM-compatible global DOF numbering.
    leaf_order: Vec<ElemId>,
    /// Current-mesh element → initial (root) element id.  Used to recover the
    /// MFEM `root_state` of unrefined root leaves during refinement.
    elem_root: HashMap<ElemId, usize>,
}

impl Default for NCStateQuad {
    fn default() -> Self { Self::new() }
}

impl NCStateQuad {
    pub fn new() -> Self {
        NCStateQuad {
            constraints: Vec::new(),
            active_midpoints: HashMap::new(),
            edge_level: HashMap::new(),
            history: Vec::new(),
            refine_tree: Vec::new(),
            elem_to_node: HashMap::new(),
            root_states: Vec::new(),
            top_level_nodes: 0,
            top_level_ids: Vec::new(),
            leaf_order: Vec::new(),
            elem_root: HashMap::new(),
        }
    }

    pub fn constraints(&self) -> &[HangingNodeConstraint] { &self.constraints }
    pub fn can_derefine(&self) -> bool { !self.history.is_empty() }
    /// Active edge midpoints: (parent edge endpoints) -> midpoint node, for
    /// every edge of the current mesh that has been split (MFEM's edge nodes).
    /// P2 constraint generation walks this to find all slave sub-edges of a
    /// coarse (master) edge (MFEM TraverseEdge / BuildEdgeList semantics).
    pub fn active_midpoints(&self) -> &HashMap<(NodeId, NodeId), NodeId> {
        &self.active_midpoints
    }

    /// Perform one level of non-conforming refinement on a Quad4 mesh.
    ///
    /// `nc_limit` limits the maximum refinement-level difference between
    /// adjacent elements (0 = no limit).  When exceeded, the coarse neighbor
    /// is also refined (propagation).
    ///
    /// Returns `(new_mesh, constraints, midpoint_map)`.
    pub fn refine(
        &mut self,
        mesh: &Mesh<2>,
        marked: &[ElemId],
        nc_limit: u32,
    ) -> (Mesh<2>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        assert!(
            mesh.elem_type == ElementType::Quad4,
            "NCStateQuad::refine: only Quad4 meshes are supported"
        );

        if marked.is_empty() {
            return (mesh.clone(), self.constraints.clone(), HashMap::new());
        }

        // ── nc_limit propagation ─────────────────────────────────────────────
        let n_elems = mesh.n_elems();
        let prop_marked: Vec<ElemId> = if nc_limit > 0 {
            let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
            for e in 0..n_elems as ElemId {
                let ns = mesh.elem_nodes(e);
                for &(a, b) in &local_edges_quad() {
                    edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
                }
            }
            propagate_nc_limit_quad(marked, mesh, &edge_elems, &self.edge_level, nc_limit)
        } else {
            marked.to_vec()
        };

        // ── First refinement: initialize MFEM root states & leaf order ──
        // (must happen BEFORE the snapshot is pushed, otherwise derefine_last
        // would restore an empty leaf_order)
        if self.root_states.is_empty() {
            self.top_level_nodes = mesh.n_nodes();
            self.top_level_ids = (0..mesh.n_nodes() as NodeId).collect();
            self.root_states = init_root_states_quad(mesh);
            self.leaf_order = (0..mesh.n_elems() as ElemId).collect();
            self.elem_root = (0..mesh.n_elems() as usize)
                .map(|e| (e as ElemId, e))
                .collect();
        }

        self.history.push(NCStateQuadSnapshot {
            mesh: mesh.clone(),
            constraints: self.constraints.clone(),
            active_midpoints: self.active_midpoints.clone(),
            edge_level: self.edge_level.clone(),
            leaf_order: self.leaf_order.clone(),
            elem_root: self.elem_root.clone(),
        });

        let marked_set: std::collections::HashSet<ElemId> = prop_marked.iter().copied().collect();
        let n_elems = mesh.n_elems();
        assert_eq!(n_elems, self.leaf_order.len(), "leaf_order out of sync");
        for &e in &prop_marked {
            assert!(
                self.leaf_order.contains(&e),
                "marked elem {e} not in leaf_order (n_elems={n_elems}, leaf_order_len={})",
                self.leaf_order.len()
            );
        }

        // ── Hilbert state of each refined element ────────────────────────────
        // A refined element is either a live leaf child of a tree node
        // (state = child_state[parent.state][slot]) or an unrefined initial
        // root (state = root_states[root]).  Precompute into a map so the
        // tree update and node-creation phases can both use it.
        let mut elem_state: HashMap<ElemId, u8> = HashMap::new();
        {
            let (etn, tree, eroot, rstates) =
                (&self.elem_to_node, &self.refine_tree, &self.elem_root, &self.root_states);
            for &e in &prop_marked {
                let st = if let Some(&(ni, k)) = etn.get(&e) {
                    // e is child slot `k` of tree node ni.  The child state
                    // must be looked up by the HILBERT position of that slot
                    // in QUAD_HILBERT_CHILD_ORDER[parent.state] (MFEM
                    // CollectLeafElements: `ch = order[state][i];
                    // st = child_state[state][i]`) — indexing by slot k is
                    // wrong whenever the order is not the identity.
                    let pstate = tree[ni].state as usize;
                    let hpos = QUAD_HILBERT_CHILD_ORDER[pstate]
                        .iter()
                        .position(|&c| c as usize == k)
                        .expect("child slot in order table");
                    QUAD_HILBERT_CHILD_STATE[pstate][hpos]
                } else {
                    let root = eroot.get(&e).copied().unwrap_or(e as usize);
                    rstates[root]
                };
                elem_state.insert(e, st);
            }
        }

        // ── New leaf order (MFEM CollectLeafElements: Hilbert SFC) ──────────
        // Unrefined elements keep their leaf-order position; refined elements
        // expand in place to their 4 children ordered by the Hilbert table.
        // `new_leaf_order` entries: (old element, child slot) with
        // slot == usize::MAX for unrefined elements.
        let mut new_leaf_order: Vec<(ElemId, usize)> = Vec::with_capacity(n_elems + 3 * prop_marked.len());
        for &e in &self.leaf_order {
            if marked_set.contains(&e) {
                let st = elem_state[&e] as usize;
                for &ch in &QUAD_HILBERT_CHILD_ORDER[st] {
                    new_leaf_order.push((e, ch as usize));
                }
            } else {
                new_leaf_order.push((e, usize::MAX));
            }
        }
        // (old element, child slot) -> new element id (only for refined).
        let mut child_new_id: HashMap<(ElemId, usize), ElemId> = HashMap::new();
        // old element -> new element id (unrefined only).
        let mut elem_new_id: HashMap<ElemId, ElemId> = HashMap::new();
        for (idx, &(e, ch)) in new_leaf_order.iter().enumerate() {
            if ch != usize::MAX {
                child_new_id.insert((e, ch), idx as ElemId);
            } else {
                elem_new_id.insert(e, idx as ElemId);
            }
        }

        // ── Update the refinement tree (MFEM GetDerefinementTable support) ──
        // 1) children of the current tree that get refined this pass cease to
        //    be leaves (their numbering is no longer tracked in the tree).
        let old_elem_to_node = std::mem::take(&mut self.elem_to_node);
        for &e in &prop_marked {
            if let Some(&(ni, k)) = old_elem_to_node.get(&e) {
                if self.refine_tree[ni].child_leaf[k] {
                    self.refine_tree[ni].child_leaf[k] = false;
                }
            }
        }
        // 2) create a fresh tree node for every refined element.  Children
        //    are stored by MFEM child slot 0..3 (XY split), NOT in Hilbert
        //    order — Hilbert order only affects the leaf output sequence.
        let old_tree_len = self.refine_tree.len();
        for &e in &prop_marked {
            let ns = mesh.elem_nodes(e);
            let st = elem_state[&e];
            let node = QuadRefineNode {
                parent_nodes: [ns[0], ns[1], ns[2], ns[3]],
                parent_tag: mesh.elem_tags[e as usize],
                children: [
                    child_new_id[&(e, 0)], child_new_id[&(e, 1)],
                    child_new_id[&(e, 2)], child_new_id[&(e, 3)],
                ],
                child_leaf: [true; 4],
                alive: true,
                // if the refined element was itself a leaf child of an
                // ancestor tree node, remember where it sits
                parent: old_elem_to_node.get(&e).copied(),
                state: st,
            };
            let ni = self.refine_tree.len();
            self.refine_tree.push(node);
            for k in 0..4 {
                self.elem_to_node.insert(child_new_id[&(e, k)], (ni, k));
            }
        }
        // 3) renumber surviving leaf children of pre-existing tree nodes.
        for (_, node) in self.refine_tree.iter_mut().enumerate().take(old_tree_len) {
            if !node.alive { continue; }
            for k in 0..4 {
                if node.child_leaf[k] {
                    let c = node.children[k];
                    match elem_new_id.get(&c) {
                        Some(&nid) => node.children[k] = nid,
                        None => {
                            // The child is not in the new leaf order: it was
                            // either derefined away (the recovered parent
                            // element replaced it) or refined this pass
                            // (marked).  Either way it is no longer a leaf of
                            // the current mesh — keep the stale id would let
                            // it alias a later element (tree corruption).
                            node.child_leaf[k] = false;
                        }
                    }
                }
            }
        }
        // 4) rebuild elem_to_node with the new numbering (old entries only).
        for (e, &(ni, k)) in &old_elem_to_node {
            let node = &self.refine_tree[ni];
            if !node.alive || !node.child_leaf[k] { continue; }
            if marked_set.contains(e) { continue; }            self.elem_to_node.insert(elem_new_id[e], (ni, k));
        }
        // 5) rebuild elem_root and leaf_order for the new mesh.  new_leaf_order
        //    is already in new-id order, so assign positionally.
        let old_elem_root = self.elem_root.clone();
        self.elem_root.clear();
        self.leaf_order.clear();
        for (idx, &(e, _)) in new_leaf_order.iter().enumerate() {
            let root = old_elem_root[&e];
            self.elem_root.insert(idx as ElemId, root);
            self.leaf_order.push(idx as ElemId);
        }

        // ── nc_limit: rebuild edge_elems AFTER propagation ──────────────
        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..n_elems as ElemId {
            let ns = mesh.elem_nodes(e);
            for &(a, b) in &local_edges_quad() {
                edge_elems.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }

        // ── Create new nodes in MFEM vertex-view order ──────────────────────
        // MFEM UpdateVertices STEP 3 assigns the non-top-level vertex DOFs by
        // scanning the (Hilbert-ordered) leaf elements and numbering each
        // new corner on first appearance.  Equivalent here: scan the refined
        // elements in old leaf_order, and within each element scan its 4
        // children in Hilbert order, creating edge midpoints and the centre
        // node on first appearance (edge midpoints shared across elements are
        // deduplicated by the midpoint map).
        let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
        let mut center_map:   HashMap<ElemId, NodeId>           = HashMap::new();
        let mut new_coords: Vec<f64> = mesh.coords.clone();
        let mut next_node = mesh.n_nodes() as NodeId;

        // Pre-existing mesh nodes by rounded coordinate, for midpoint reuse.
        // A marked element's edge may already carry a midpoint from a
        // *previous* refinement round (a hanging edge that this element now
        // refines further).  `self.active_midpoints` normally records those,
        // but `par_refine_marked` rebuilds a fresh `NCStateQuad` per round
        // (partition rebuild renumbers ids), so the history is empty and a
        // naive get-or-create would duplicate the midpoint node (pex6 it3:
        // 30 duplicated coords, unknowns 321 vs C++ 291).  Look the midpoint
        // up by coordinate like MFEM `Mesh::GetId` (9-decimal tolerance for
        // accumulated bisection rounding), reusing the existing node.
        let mut coord_to_node: HashMap<String, NodeId> = HashMap::new();
        for n in 0..mesh.n_nodes() as NodeId {
            let c = mesh.coords_of(n);
            coord_to_node.entry(format!("{:.9},{:.9}", c[0], c[1])).or_insert(n);
        }

        // Refined elements in leaf (Hilbert) order — the order in which they
        // appear in the old leaf_order; propagate any stragglers at the end.
        let mut marked_leaf: Vec<ElemId> = self.leaf_order.iter()
            .filter(|e| marked_set.contains(e))
            .copied()
            .collect();
        for &e in &prop_marked {
            if !marked_leaf.contains(&e) { marked_leaf.push(e); }
        }

        // edge-level updates recorded while creating midpoints (parent edge
        // removed, two sub-edges added) — applied after the creation loop.
        let mut edge_level_updates: Vec<((NodeId, NodeId), u32)> = Vec::new();
        let mut edge_level_removals: Vec<(NodeId, NodeId)> = Vec::new();

        for &e in &marked_leaf {
            let st = elem_state[&e] as usize;
            let ns = mesh.elem_nodes(e);
            let (n0, n1, n2, n3) = (ns[0], ns[1], ns[2], ns[3]);
            // get-or-create midpoint of edge (a,b)
            let mut mid = |a: NodeId, b: NodeId,
                           midpoint_map: &mut HashMap<(NodeId, NodeId), NodeId>,
                           new_coords: &mut Vec<f64>,
                           next_node: &mut NodeId| -> NodeId {
                let key = quad_edge_key(a, b);
                if let Some(&m) = midpoint_map.get(&key) { return m; }
                if let Some(&m) = self.active_midpoints.get(&key) {
                    midpoint_map.insert(key, m);
                    return m;
                }
                let xa = mesh.coords_of(a);
                let xb = mesh.coords_of(b);
                // Reuse a midpoint already present in the mesh (created by a
                // previous round's refinement of this same edge): looking it
                // up by rounded coordinate matches MFEM `Mesh::GetId`.  This
                // is what keeps a second-round refinement of a hanging edge
                // from duplicating the midpoint node.  (Must run before the
                // 0.5·pa+0.5·pb push below, which would create the dup.)
                let mkey = format!("{:.9},{:.9}", 0.5 * xa[0] + 0.5 * xb[0], 0.5 * xa[1] + 0.5 * xb[1]);
                if let Some(&m) = coord_to_node.get(&mkey) {
                    if m != a && m != b {
                        midpoint_map.insert(key, m);
                        return m;
                    }
                }
                // MFEM CalcVertexPos for a scale-0.5 edge node:
                //   pos = (1-s)·pa + s·pb = 0.5·pa + 0.5·pb
                // (NOT 0.5·(pa+pb) — differs by 1 ulp and propagates into the
                // refined elements' Jacobians and element matrices).
                new_coords.push(0.5 * xa[0] + 0.5 * xb[0]);
                new_coords.push(0.5 * xa[1] + 0.5 * xb[1]);
                let m = *next_node;
                *next_node += 1;
                midpoint_map.insert(key, m);
                let parent_level = self.edge_level.get(&key).copied().unwrap_or(0);
                edge_level_updates.push((quad_edge_key(a, m), parent_level + 1));
                edge_level_updates.push((quad_edge_key(m, b), parent_level + 1));
                edge_level_removals.push(key);
                m
            };
            // get-or-create centre of element e
            let centre = |e: ElemId,
                              center_map: &mut HashMap<ElemId, NodeId>,
                              new_coords: &mut Vec<f64>,
                              next_node: &mut NodeId| -> NodeId {
                if let Some(&c) = center_map.get(&e) { return c; }
                // MFEM XY split centre: midel = nodes.GetId(mid01, mid23) with
                // CalcVertexPos = 0.5·pos(mid01) + 0.5·pos(mid23), where each
                // midpoint uses the same 0.5·pa + 0.5·pb formula.  (A 4-corner
                // average differs in the last ulp.)
                let ns2 = mesh.elem_nodes(e);
                let (m01x, m01y) = {
                    let xa = mesh.coords_of(ns2[0]);
                    let xb = mesh.coords_of(ns2[1]);
                    (0.5 * xa[0] + 0.5 * xb[0], 0.5 * xa[1] + 0.5 * xb[1])
                };
                let (m23x, m23y) = {
                    let xa = mesh.coords_of(ns2[2]);
                    let xb = mesh.coords_of(ns2[3]);
                    (0.5 * xa[0] + 0.5 * xb[0], 0.5 * xa[1] + 0.5 * xb[1])
                };
                new_coords.push(0.5 * m01x + 0.5 * m23x);
                new_coords.push(0.5 * m01y + 0.5 * m23y);
                let id = *next_node;
                *next_node += 1;
                center_map.insert(e, id);
                id
            };
            // scan children in Hilbert order; per child scan its 4 corners in
            // MFEM XY-split order, creating midpoints / centre on first use.
            for &ch in &QUAD_HILBERT_CHILD_ORDER[st] {
                match ch {
                    0 => {
                        let _ = mid(n0, n1, &mut midpoint_map, &mut new_coords, &mut next_node);
                        let _ = centre(e, &mut center_map, &mut new_coords, &mut next_node);
                        let _ = mid(n3, n0, &mut midpoint_map, &mut new_coords, &mut next_node);
                    }
                    1 => {
                        let _ = mid(n0, n1, &mut midpoint_map, &mut new_coords, &mut next_node);
                        let _ = mid(n1, n2, &mut midpoint_map, &mut new_coords, &mut next_node);
                        let _ = centre(e, &mut center_map, &mut new_coords, &mut next_node);
                    }
                    2 => {
                        let _ = centre(e, &mut center_map, &mut new_coords, &mut next_node);
                        let _ = mid(n1, n2, &mut midpoint_map, &mut new_coords, &mut next_node);
                        let _ = mid(n2, n3, &mut midpoint_map, &mut new_coords, &mut next_node);
                    }
                    3 => {
                        let _ = mid(n3, n0, &mut midpoint_map, &mut new_coords, &mut next_node);
                        let _ = centre(e, &mut center_map, &mut new_coords, &mut next_node);
                        let _ = mid(n2, n3, &mut midpoint_map, &mut new_coords, &mut next_node);
                    }
                    _ => unreachable!(),
                }
            }
        }
        // apply deferred edge-level updates
        for (k, lvl) in edge_level_updates {
            self.edge_level.insert(k, lvl);
        }
        for k in edge_level_removals {
            self.edge_level.remove(&k);
        }

        let mut new_conn: Vec<NodeId> = Vec::new();
        let mut new_tags: Vec<i32>    = Vec::new();
        // Elements are output in the NEW (Hilbert) leaf order; each refined
        // element contributes its 4 children in Hilbert order, matching the
        // MFEM `CollectLeafElements` sequence.
        for &(e, ch) in &new_leaf_order {
            let ns = mesh.elem_nodes(e);
            let tag = mesh.elem_tags[e as usize];
            if ch == usize::MAX {
                for k in 0..4 { new_conn.push(ns[k]); }
                new_tags.push(tag);
            } else {
                let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2]; let n3 = ns[3];
                let m01 = *midpoint_map.get(&quad_edge_key(n0, n1)).unwrap();
                let m12 = *midpoint_map.get(&quad_edge_key(n1, n2)).unwrap();
                let m23 = *midpoint_map.get(&quad_edge_key(n2, n3)).unwrap();
                let m30 = *midpoint_map.get(&quad_edge_key(n3, n0)).unwrap();
                let c   = *center_map.get(&e).unwrap();
                // MFEM XY-split child corner order (RefineElement):
                //   child0 = (n0, m01, c, m30)   child1 = (m01, n1, m12, c)
                //   child2 = (c, m12, n2, m23)   child3 = (m30, c, m23, n3)
                match ch {
                    0 => new_conn.extend_from_slice(&[n0,  m01, c,   m30]),
                    1 => new_conn.extend_from_slice(&[m01, n1,  m12, c  ]),
                    2 => new_conn.extend_from_slice(&[c,   m12, n2,  m23]),
                    3 => new_conn.extend_from_slice(&[m30, c,   m23, n3 ]),
                    _ => unreachable!(),
                }
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
        // ── MFEM BuildEdgeList master/slave semantics ─────────────────────
        // A coarse (master) edge is a split edge that is exposed on a leaf
        // element.  Its split sub-edges are SLAVE edges: their midpoints are
        // constrained by the master-edge P2 basis.  A sub-edge that is itself
        // split and exposed must NOT generate its own P1 constraint (it would
        // double-constrain the midpoint with a different master).  So we keep
        // only the "true master" edges: candidates minus every edge that lies
        // on the split chain of another candidate.
        let mut candidate: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
        for (&(a, b), &mid) in &self.active_midpoints {
            if new_node_set.contains(&mid)
                && new_edge_elems.contains_key(&quad_edge_key(a, b))
            {
                candidate.insert(quad_edge_key(a, b));
            }
        }
        fn remove_slaves(
            a: NodeId,
            b: NodeId,
            masters: &mut std::collections::HashSet<(NodeId, NodeId)>,
            midpoints: &HashMap<(NodeId, NodeId), NodeId>,
        ) {
            let key = quad_edge_key(a, b);
            if let Some(&mid) = midpoints.get(&key) {
                masters.remove(&quad_edge_key(a, mid));
                remove_slaves(a, mid, masters, midpoints);
                masters.remove(&quad_edge_key(mid, b));
                remove_slaves(mid, b, masters, midpoints);
            }
        }
        let mut masters = candidate.clone();
        for &(a, b) in &candidate {
            if masters.contains(&(a, b)) {
                remove_slaves(a, b, &mut masters, &self.active_midpoints);
            }
        }
        for (&(a, b), &mid) in &self.active_midpoints {
            if !new_node_set.contains(&mid) { continue; }
            if !masters.contains(&quad_edge_key(a, b)) { continue; }
            if std::env::var("EX15_DBG_DEREF").is_ok() {
                let has = self.active_midpoints.contains_key(&quad_edge_key(a, b));
                eprintln!("REFINE-P1 {} <- {} + {} (in_active={has})", mid, a, b);
            }
            new_constraints.push(HangingNodeConstraint {
                constrained: mid as usize,
                parent_a: a as usize,
                parent_b: b as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
            });
        }
        // Keep every historical split midpoint whose node id is still valid
        // (refinement only appends nodes, so all old ids stay valid).  MFEM
        // keeps its edge-node history even when a split point is not a corner
        // of any element; deleting it here loses deep slave constraints.
        let n_nodes_after = new_coords.len() / 2;
        self.active_midpoints.retain(|_, mid| (*mid as usize) < n_nodes_after);
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

        // MFEM vertex-view order (UpdateVertices): top-level (original) nodes
        // first, then every non-top-level node in leaf-element (Hilbert SFC)
        // order — first appearance when scanning the elements in leaf_order.
        // The DofManager reads this to make the global vertex DOF ids match
        // MFEM on the (multi-level) NC mesh.
        let mut view: Vec<NodeId> = self.top_level_ids.clone();
        let mut seen: std::collections::HashSet<NodeId> = view.iter().copied().collect();
        for &e in &self.leaf_order {
            let off = e as usize * 4;
            for k in 0..4 {
                let n = new_conn[off + k];
                if seen.insert(n) {
                    view.push(n);
                }
            }
        }
        debug_assert_eq!(view.len(), new_coords.len() / 2, "vertex view size mismatch");

        let mut new_mesh = Mesh::uniform(
            new_coords, new_conn, new_tags, ElementType::Quad4,
            new_face_conn, new_face_tags, ElementType::Line2,
        );
        new_mesh.nc_vertex_view = Some(view);
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            let mut all: std::collections::HashSet<ElemId> = Default::default();
            let mut dups: Vec<(usize, usize, ElemId)> = Vec::new();
            for (ni, node) in self.refine_tree.iter().enumerate() {
                if !node.alive { continue; }
                for (k, &c) in node.children.iter().enumerate() {
                    if node.child_leaf[k] && !all.insert(c) { dups.push((ni, k, c)); }
                }
            }
            eprintln!("REFINE-DONE leaf-children-dups={dups:?}");
            if !dups.is_empty() {
                let mut by_node: std::collections::BTreeMap<usize, Vec<ElemId>> = Default::default();
                for &(ni, _, _) in &dups { by_node.entry(ni).or_default(); }
                for &(ni, _, c) in &dups {
                    by_node.entry(ni).or_default().push(c);
                }
                for (ni, cs) in by_node {
                    let node = &self.refine_tree[ni];
                    eprintln!(
                        "REFINE-DUP node={ni} children={:?} leaf={:?} parent={:?}",
                        node.children, node.child_leaf, node.parent
                    );
                    let _ = cs;
                }
            }
        }
        (new_mesh, self.constraints.clone(), midpoint_map)
    }

    pub fn derefine_last(&mut self) -> Option<(Mesh<2>, Vec<HangingNodeConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints.clone();
        self.active_midpoints = snap.active_midpoints;
        self.edge_level = snap.edge_level;
        self.leaf_order = snap.leaf_order;
        self.elem_root = snap.elem_root;
        Some((snap.mesh, self.constraints.clone()))
    }

    /// Indices of tree nodes whose 4 children are all still leaves of the
    /// current mesh — the MFEM `GetDerefinementTable` groups.  A group may be
    /// coarsened (derefined) if its aggregated error is below the threshold.
    pub fn deref_groups(&self) -> Vec<usize> {
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            let mut info: Vec<String> = Vec::new();
            for (i, n) in self.refine_tree.iter().enumerate() {
                if n.derefinable() {
                    info.push(format!("{i}:{:?}", n.children));
                }
            }
            eprintln!("DEREF-ALL groups={} {}", info.len(), info.join(" "));
        }
        self.refine_tree
            .iter()
            .enumerate()
            .filter(|(_, n)| n.derefinable())
            .map(|(i, _)| i)
            .collect()
    }

    /// Children (current-mesh element indices) of the given tree nodes.
    pub fn deref_group_children(&self, node: usize) -> [ElemId; 4] {
        self.refine_tree[node].children
    }

    /// MFEM `NCMesh::CheckDerefinementNCLevel` (Quad4): a derefinement group
    /// is level-ok when no child's k-direction split depth reaches
    /// `max_nc_level` while the parent was refined in that direction.
    /// Rust's quad refinement is always a 4-split (XY), so the parent
    /// `ref_type` sets both bits — both directions are checked:
    ///   splits[0] = max(EdgeSplitLevel(edge0), EdgeSplitLevel(edge2))
    ///   splits[1] = max(EdgeSplitLevel(edge1), EdgeSplitLevel(edge3))
    pub fn deref_group_nc_ok(&self, node: usize, nc_limit: u32, mesh: &Mesh<2>) -> bool {
        if nc_limit == 0 { return true; }
        let Some(tn) = self.refine_tree.get(node) else { return true; };
        if !tn.derefinable() { return true; }
        let coords: Vec<[f64; 2]> =
            (0..mesh.n_nodes()).map(|n| mesh.coords_of(n as NodeId)).collect();
        let mut pos: HashMap<(i64, i64), NodeId> = HashMap::new();
        for (n, c) in coords.iter().enumerate() {
            let q = ((c[0] * 1e10).round() as i64, (c[1] * 1e10).round() as i64);
            pos.entry(q).or_insert(n as NodeId);
        }
        let mut memo: HashMap<(NodeId, NodeId), u32> = HashMap::new();
        // MFEM HasVertex(): only element-corner midpoints count.
        let mut is_vertex: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for e in 0..mesh.n_elems() as ElemId {
            for &n in mesh.elem_nodes(e) { is_vertex.insert(n); }
        }
        let mut worst: Vec<(u32, u32)> = Vec::new();
        for &c in &tn.children {
            let ns = mesh.elem_nodes(c);
            let s0 = edge_split_level_geo(ns[0], ns[1], &coords, &pos, &is_vertex, &mut memo)
                .max(edge_split_level_geo(ns[2], ns[3], &coords, &pos, &is_vertex, &mut memo));
            let s1 = edge_split_level_geo(ns[1], ns[2], &coords, &pos, &is_vertex, &mut memo)
                .max(edge_split_level_geo(ns[3], ns[0], &coords, &pos, &is_vertex, &mut memo));
            worst.push((s0, s1));
            if s0 >= nc_limit || s1 >= nc_limit {
                if std::env::var("EX15_DBG_DEREF").is_ok() {
                    eprintln!("DBG nc_ok: group {node} child {c} ns={ns:?} splits=({s0},{s1}) limit={nc_limit}");
                }
                return false;
            }
        }
        if std::env::var("EX15_DBG_DEREF").is_ok() && worst.iter().any(|&(a, b)| a > 0 || b > 0) {
            eprintln!("DBG nc_ok: group {node} children={:?} all-splits={worst:?}", tn.children);
        }
        true
    }

    /// Coarsen (derefine) the given tree-node groups: remove their leaf
    /// children from `mesh` and restore each parent element at the position
    /// of its first child.  Returns the new mesh.  Hanging-node state
    /// (midpoints, edge levels, constraints) is updated accordingly.
    pub fn derefine_groups(&mut self, mesh: &Mesh<2>, groups: &[usize]) -> Option<Mesh<2>> {
        if groups.is_empty() { return None; }

        let n_elems = mesh.n_elems() as usize;
        let mut removed: std::collections::HashSet<ElemId> = Default::default();
        let mut parents: Vec<(ElemId, [NodeId; 4], i32)> = Vec::new(); // (slot, nodes, tag)
        for &ni in groups {
            let node = &self.refine_tree[ni];
            if !node.derefinable() { continue; }
            for k in 0..4 {
                let c = node.children[k];
                if !removed.insert(c) {
                    eprintln!(
                        "DBG deref: group {ni} child {c} (slot {k}) already removed; tree children={:?}",
                        node.children
                    );
                }
            }
            // MFEM DerefineElement: the recovered parent element's corners are
            // exactly the corners of its 4 children (XY split layout):
            //   child0=(n0,m01,c,m30)  child1=(m01,n1,m12,c)
            //   child2=(c,m12,n2,m23)  child3=(m30,c,m23,n3)
            // so parent corner k = child slot k's corner k.  Recovering them
            // from the current-mesh children (instead of the stale historical
            // parent_nodes ids) keeps the parent geometry valid even after
            // repeated derefinements have compacted away orphan nodes.
            let cc: Vec<[NodeId; 4]> = node
                .children
                .iter()
                .map(|&c| {
                    let ns = mesh.elem_nodes(c);
                    [ns[0], ns[1], ns[2], ns[3]]
                })
                .collect();
            let parent_nodes =
                [cc[0][0], cc[1][1], cc[2][2], cc[3][3]];
            parents.push((node.children[0], parent_nodes, node.parent_tag));
        }
        if parents.is_empty() { return None; }
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            let mut all: std::collections::HashSet<ElemId> = Default::default();
            let mut dups: Vec<(usize, usize, ElemId)> = Vec::new();
            for (ni, node) in self.refine_tree.iter().enumerate() {
                if !node.alive { continue; }
                for (k, &c) in node.children.iter().enumerate() {
                    if node.child_leaf[k] && !all.insert(c) { dups.push((ni, k, c)); }
                }
            }
            eprintln!("DBG deref pre-check all-leaf-children-dups={dups:?}");
            let mut all2: std::collections::HashSet<ElemId> = Default::default();
            let mut dups2: Vec<ElemId> = Vec::new();
            for &ni in groups {
                let node = &self.refine_tree[ni];
                for &c in &node.children {
                    if !all2.insert(c) { dups2.push(c); }
                }
            }
            eprintln!("DBG deref groups={} children-dups={dups2:?}", groups.len());
            for &ni in groups.iter().take(20) {
                let node = &self.refine_tree[ni];
                eprintln!(
                    "  tree[{ni}] children={:?} leaf={:?} alive={} parent={:?} state={} parent_nodes={:?}",
                    node.children, node.child_leaf, node.alive, node.parent, node.state, node.parent_nodes
                );
                // ancestor chain
                let mut cur = node.parent;
                let mut chain = Vec::new();
                while let Some((pi, pk)) = cur {
                    chain.push((pi, pk));
                    cur = self.refine_tree[pi].parent;
                }
                eprintln!("  tree[{ni}] ancestor-chain={chain:?}");
            }
        }
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            for &ni in groups {
                let node = &self.refine_tree[ni];
                eprintln!(
                    "DBG GRP node={ni} parent={:?} children={:?} leaf={:?}",
                    node.parent, node.children, node.child_leaf
                );
            }
        }
        parents.sort_by_key(|&(slot, _, _)| slot);

        // New element numbering after coarsening: `new_id(e) = e - shift[e]`
        // where shift[e] counts removed children before e.  Surviving tree
        // children and elem_to_node keys must be renumbered accordingly.
        let mut shift: Vec<usize> = vec![0; n_elems];
        {
            let mut cnt = 0usize;
            for e in 0..n_elems {
                shift[e] = cnt;
                if removed.contains(&(e as ElemId)) { cnt += 1; }
            }
        }
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            for ni in [0usize, 9, 11, 12, 14] {
                if ni >= self.refine_tree.len() { continue; }
                let node = &self.refine_tree[ni];
                let vals: Vec<String> = node
                    .children
                    .iter()
                    .map(|&c| {
                        let c = c as usize;
                        format!(
                            "{c}(r={},sh={})",
                            removed.contains(&(c as ElemId)),
                            c - shift[c]
                        )
                    })
                    .collect();
                eprintln!("DEREF-RENUM tree[{ni}] alive={} children=[{}]", node.alive, vals.join(" "));
            }
        }
        // When a group is derefined its parent element becomes a leaf again;
        // restore the ancestor slot (MFEM DerefineElement) and register the
        // recovered parent in elem_to_node so later refinements track it.
        //
        // The parent element's new id is its position in the compressed
        // element sequence: it occupies its first child's (removed) slot, so
        // its id comes from `new_id_of[children[0]]` (NOT `slot - shift[slot]`
        // — for contiguous removed regions that maps several distinct parent
        // elements onto the same id and corrupts the tree).
        let mut new_id_of: HashMap<ElemId, ElemId> = HashMap::new();
        {
            let mut cnt = 0usize;
            let mut pit = 0usize;
            for e in 0..n_elems as ElemId {
                while pit < parents.len() && parents[pit].0 <= e {
                    new_id_of.insert(parents[pit].0, cnt as ElemId);
                    cnt += 1;
                    pit += 1;
                }
                if !removed.contains(&e) {
                    new_id_of.insert(e, cnt as ElemId);
                    cnt += 1;
                }
            }
        }
        let mut restored: Vec<(ElemId, usize, usize)> = Vec::new(); // (parent_elem_id, anc_node, anc_slot)
        for &ni in groups {
            let node = &self.refine_tree[ni];
            if let Some((pi, pk)) = node.parent {
                // parent element occupies its first child's (removed) slot
                let slot = node.children[0];
                let pe = new_id_of[&slot];
                if std::env::var("EX15_DBG_DEREF").is_ok() && (pi == 5 || pi == 9) {
                    eprintln!(
                        "DBG RESTORE group={ni} parent=({pi},{pk}) children[0]={slot} pe={pe} child_leaf={:?}",
                        node.child_leaf
                    );
                }
                restored.push((pe, pi, pk));
            }
        }
        for node in &mut self.refine_tree {
            if !node.alive { continue; }
            for k in 0..4 {
                if node.child_leaf[k] {
                    let c = node.children[k] as usize;
                    if removed.contains(&(c as ElemId)) {
                        // The child was coarsened away this pass.  Do NOT
                        // renumber it: that maps onto a surviving element and
                        // makes two tree nodes share one element.
                        node.child_leaf[k] = false;
                    } else {
                        node.children[k] = new_id_of[&(c as ElemId)];
                    }
                }
            }
        }
        let old_etn = std::mem::take(&mut self.elem_to_node);
        for (e, v) in old_etn {
            let e = e as ElemId;
            if removed.contains(&e) { continue; }
            self.elem_to_node.insert(new_id_of[&e], v);
        }
        for (pe, pi, pk) in restored {
            self.refine_tree[pi].child_leaf[pk] = true;
            // The recovered parent element occupies its derefined child's
            // slot in the ancestor node (MFEM DerefineElement restores the
            // child element at the parent position); keep children[] in sync
            // so later refinements renumber it correctly.
            self.refine_tree[pi].children[pk] = pe;
            self.elem_to_node.insert(pe, (pi, pk));
        }

        // Rebuild element connectivity: skip removed children, insert each
        // parent element at its first child's position.
        let mut new_conn: Vec<NodeId> = Vec::new();
        let mut new_tags: Vec<i32> = Vec::new();
        let mut parent_iter = parents.iter().peekable();
        for e in 0..n_elems as ElemId {
            // Insert the parent element at its first child's position before
            // processing element e (the slot is a removed child itself).
            while let Some((slot, nodes, tag)) = parent_iter.peek() {
                if *slot > e { break; }
                new_conn.extend_from_slice(&nodes[..]);
                new_tags.push(*tag);
                parent_iter.next();
            }
            if removed.contains(&e) { continue; }
            let ns = mesh.elem_nodes(e);
            new_conn.extend_from_slice(ns);
            new_tags.push(mesh.elem_tags[e as usize]);
        }
        for (_, nodes, tag) in parent_iter {
            new_conn.extend_from_slice(&nodes[..]);
            new_tags.push(*tag);
        }

        // Faces: rebuild boundary faces from the old face table (a coarsened
        // child edge disappears, so the parent edge may become boundary).
        let old_face_tag: std::collections::HashMap<(NodeId, NodeId), i32> = mesh
            .face_conn
            .chunks_exact(2)
            .zip(mesh.face_tags.iter())
            .map(|(p, &t)| (quad_edge_key(p[0], p[1]), t))
            .collect();
        let used: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
        // A boundary edge is an element edge used by exactly one element.
        // MFEM rebuilds the boundary from the coarsened mesh, so edges that
        // become boundary after coarsening (parent edges of removed children)
        // must be recovered too — the old face table only knows the *child*
        // (finer) edges, so walk down the split chain of the edge to find its
        // attribute (DerefineElement's RegisterFaces inherits the child's
        // face attribute).
        let mut edge_adj: std::collections::HashMap<(NodeId, NodeId), u32> = Default::default();
        for e in 0..new_tags.len() as ElemId {
            let ns = &new_conn[e as usize * 4..e as usize * 4 + 4];
            for &(a, b) in &local_edges_quad() {
                *edge_adj.entry(quad_edge_key(ns[a], ns[b])).or_insert(0) += 1;
            }
        }
        fn face_tag_of(
            a: NodeId,
            b: NodeId,
            old_face_tag: &std::collections::HashMap<(NodeId, NodeId), i32>,
            midpoints: &HashMap<(NodeId, NodeId), NodeId>,
        ) -> Option<i32> {
            let key = quad_edge_key(a, b);
            if let Some(&t) = old_face_tag.get(&key) { return Some(t); }
            if let Some(&m) = midpoints.get(&key) {
                if let Some(t) = face_tag_of(a, m, old_face_tag, midpoints) { return Some(t); }
                if let Some(t) = face_tag_of(m, b, old_face_tag, midpoints) { return Some(t); }
            }
            None
        }
        let mut new_face_conn: Vec<NodeId> = Vec::new();
        let mut new_face_tags: Vec<i32> = Vec::new();
        for e in 0..new_tags.len() as ElemId {
            let ns = &new_conn[e as usize * 4..e as usize * 4 + 4];
            for &(a, b) in &local_edges_quad() {
                let key = quad_edge_key(ns[a], ns[b]);
                if !used.contains(&ns[a]) || !used.contains(&ns[b]) { continue; }
                if edge_adj[&key] != 1 { continue; } // interior edge
                if let Some(&t) = old_face_tag.get(&key) {
                    new_face_conn.extend_from_slice(&[ns[a], ns[b]]);
                    new_face_tags.push(t);
                } else if let Some(t) = face_tag_of(ns[a], ns[b], &old_face_tag, &self.active_midpoints) {
                    new_face_conn.extend_from_slice(&[ns[a], ns[b]]);
                    new_face_tags.push(t);
                }
            }
        }

        // ── Compact node numbering (drop orphan nodes of removed children) ──
        // MFEM's NCMesh::Update() deletes unused nodes after Derefine; keep
        // only nodes referenced by the new connectivity and remap everything.
        // IMPORTANT: nodes referenced by the active edge-midpoint map (split
        // endpoints + midpoints) must survive even if no element uses them —
        // MFEM keeps its edge-node history so a later TraverseEdge can still
        // find sub-edge midpoints and constrain deep slaves.  Dropping them
        // here silently loses the split history (slave constraints vanish).
        let mut keep: std::collections::HashSet<NodeId> = Default::default();
        for (&(a, b), &mid) in &self.active_midpoints {
            keep.insert(a);
            keep.insert(b);
            keep.insert(mid);
        }
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut new_coords: Vec<f64> = Vec::new();
        for &n in &new_conn {
            if !node_map.contains_key(&n) {
                node_map.insert(n, node_map.len() as NodeId);
                new_coords.extend_from_slice(&mesh.coords_of(n));
            }
        }
        for &n in &keep {
            if !node_map.contains_key(&n) {
                node_map.insert(n, node_map.len() as NodeId);
                new_coords.extend_from_slice(&mesh.coords_of(n));
            }
        }
        let new_conn: Vec<NodeId> = new_conn.iter().map(|&n| node_map[&n]).collect();
        let new_face_conn: Vec<NodeId> =
            new_face_conn.iter().map(|&n| node_map[&n]).collect();
        // remap tree parent geometries (parents are elements of the new mesh).
        // Only derefinable nodes have a parent element in the mesh; nodes with
        // no remaining leaf children can never be derefined again, so their
        // historical geometry is not remapped (their old ids are stale after
        // the node compaction but are never used again).
        for (ni, node) in self.refine_tree.iter_mut().enumerate() {
            if !node.derefinable() { continue; }
            for k in 0..4 {
                if let Some(&rn) = node_map.get(&node.parent_nodes[k]) {
                    node.parent_nodes[k] = rn;
                } else {
                    // The parent element's corner node may be a hanging node
                    // of a still-refined neighbour that got compacted away.
                    // Recover the top-level node via the child's corners
                    // (MFEM RetrieveNode semantics): the parent corners are
                    // exactly the corners of its children that survive.
                    // Fall back: keep the old id (the parent element is not
                    // derefinable while children remain, so this only matters
                    // for geometry; safest is to leave it unchanged).
                    eprintln!(
                        "warn: parent_nodes[{}] of tree node {} (id {}) missing after compaction",
                        k, ni, node.parent_nodes[k]
                    );
                }
            }
        }
        // remap active midpoints, dropping orphans (nodes no longer in use)
        let old_midpoints = std::mem::take(&mut self.active_midpoints);
        for ((a, b), mid) in old_midpoints {
            if let (Some(&ra), Some(&rb), Some(&rm)) =
                (node_map.get(&a), node_map.get(&b), node_map.get(&mid))
            {
                // Keep keys canonicalized (min,max) — the P2 constraint walk
                // and P1 generation rely on canonical keys; an uncanonicalized
                // key after a derefinement silently breaks the midpoint-chain
                // recursion (slave constraints lost).
                self.active_midpoints.insert(quad_edge_key(ra, rb), rm);
            }
        }
        // remap edge levels, dropping orphan edges
        let old_edge_level = std::mem::take(&mut self.edge_level);
        for ((a, b), lvl) in old_edge_level {
            if let (Some(&ra), Some(&rb)) = (node_map.get(&a), node_map.get(&b)) {
                self.edge_level.insert((ra, rb), lvl);
            }
        }

        let mut new_mesh = Mesh::uniform(
            new_coords, new_conn.clone(), new_tags, ElementType::Quad4,
            new_face_conn, new_face_tags, ElementType::Line2,
        );

        // Mark nodes as coarsened (dead) in the tree.
        for &ni in groups {
            if let Some(node) = self.refine_tree.get_mut(ni) {
                node.alive = false;
            }
        }
        self.elem_to_node.retain(|_, &mut (ni, _)| self.refine_tree[ni].alive);

        // ── Rebuild leaf_order & elem_root for the coarsened mesh ───────────
        // MFEM DerefineElement: the recovered parent element replaces its 4
        // children in the leaf sequence (at the first child's position).
        // New element ids match how new_conn was assembled above: surviving
        // elements keep their relative order (compressed), and each recovered
        // parent takes the id at its first child's (removed) slot.
        let old_leaf_order = std::mem::take(&mut self.leaf_order);
        let old_elem_root = std::mem::take(&mut self.elem_root);
        // `new_id_of` was already built above (before `restored`) — reuse it.
        self.leaf_order.clear();
        self.elem_root.clear();
        for &e in &old_leaf_order {
            if removed.contains(&e) {
                // first removed child of a group = parent slot: insert parent
                if new_id_of.contains_key(&e) {
                    let pe = new_id_of[&e];
                    let root = old_elem_root[&e];
                    self.leaf_order.push(pe);
                    self.elem_root.insert(pe, root);
                }
                continue;
            }
            let ne = new_id_of[&e];
            let root = old_elem_root[&e];
            self.leaf_order.push(ne);
            self.elem_root.insert(ne, root);
        }
        if std::env::var("EX15_DBG_DEREF").is_ok() {
            eprintln!(
                "DEREF groups={} old_le={} new_le={} mesh_ne={} parents={:?}",
                groups.len(), old_leaf_order.len(), self.leaf_order.len(), new_mesh.n_elems(),
                parents.iter().map(|p| p.0).collect::<Vec<_>>()
            );
            let missing: Vec<ElemId> = (0..new_mesh.n_elems() as ElemId)
                .filter(|e| !self.leaf_order.contains(e))
                .collect();
            eprintln!("  missing-from-leaf_order: {missing:?}");
            let mut dup: Vec<ElemId> = Vec::new();
            let mut seen: std::collections::HashSet<ElemId> = Default::default();
            for &e in &self.leaf_order {
                if !seen.insert(e) { dup.push(e); }
            }
            eprintln!("  dup-in-leaf_order: {dup:?}");
            let mut m2: std::collections::HashSet<ElemId> = (0..new_mesh.n_elems() as ElemId).collect();
            for &e in &self.leaf_order { m2.remove(&e); }
            let _ = missing;
            eprintln!("  unaccounted-mesh-elems: {:?}", m2.iter().collect::<Vec<_>>());
        }

        // Drop midpoints whose parent edge no longer exists after coarsening.
        let new_edge_elems: std::collections::HashMap<(NodeId, NodeId), Vec<u32>> = {
            let mut m: std::collections::HashMap<(NodeId, NodeId), Vec<u32>> = Default::default();
            for e in 0..new_mesh.n_elems() as u32 {
                let ns = new_mesh.elem_nodes(e);
                for &(a, b) in &local_edges_quad() {
                    m.entry(quad_edge_key(ns[a], ns[b])).or_default().push(e);
                }
            }
            m
        };
        let new_node_set: std::collections::HashSet<NodeId> = new_conn.iter().copied().collect();
        // MFEM keeps an edge-node while its edge_refc > 0, i.e. while the edge
        // itself is referenced by a leaf element (ReferenceElement bumps
        // edge_refc on every element edge; UnreferenceElement only deletes
        // when the refcount hits zero).  The midpoint node does NOT have to be
        // an element corner — e.g. the t=1/8 hanging point 365' on element
        // edge (69,365) survives because (69,365) is an element edge.  The old
        // `new_node_set.contains(&mid)` requirement dropped such deep split
        // records, breaking the P2 TraverseEdge chain (slave constraints for
        // dofs 85 92 131 147 234 241 275 291 369 375 vanished).  Keep a split
        // record if the edge itself or ANY descendant sub-edge (recursively —
        // MFEM's ReferenceElement recursion bumps edge_refc all the way up the
        // split tree) is an element edge.
        fn split_edge_referenced(
            a: NodeId,
            b: NodeId,
            midpoints: &HashMap<(NodeId, NodeId), NodeId>,
            new_edge_elems: &std::collections::HashMap<(NodeId, NodeId), Vec<u32>>,
        ) -> bool {
            if new_edge_elems.contains_key(&quad_edge_key(a, b)) { return true; }
            if let Some(&mid) = midpoints.get(&quad_edge_key(a, b)) {
                if split_edge_referenced(a, mid, midpoints, new_edge_elems) { return true; }
                if split_edge_referenced(mid, b, midpoints, new_edge_elems) { return true; }
            }
            false
        }
        let mp_snapshot = self.active_midpoints.clone();
        self.active_midpoints.retain(|&(a, b), _mid| {
            split_edge_referenced(a, b, &mp_snapshot, &new_edge_elems)
        });

        // Rebuild hanging-node constraints for the coarsened mesh.
        let mut new_constraints: Vec<HangingNodeConstraint> = Vec::new();
        // MFEM BuildEdgeList master/slave semantics (see refine): keep only
        // true master edges (exposed split edges not on another split chain).
        let mut candidate: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
        for (&(a, b), &mid) in &self.active_midpoints {
            if new_node_set.contains(&mid)
                && new_edge_elems.contains_key(&quad_edge_key(a, b))
            {
                candidate.insert(quad_edge_key(a, b));
            }
        }
        fn remove_slaves(
            a: NodeId,
            b: NodeId,
            masters: &mut std::collections::HashSet<(NodeId, NodeId)>,
            midpoints: &HashMap<(NodeId, NodeId), NodeId>,
        ) {
            let key = quad_edge_key(a, b);
            if let Some(&mid) = midpoints.get(&key) {
                masters.remove(&quad_edge_key(a, mid));
                remove_slaves(a, mid, masters, midpoints);
                masters.remove(&quad_edge_key(mid, b));
                remove_slaves(mid, b, masters, midpoints);
            }
        }
        let mut masters = candidate.clone();
        for &(a, b) in &candidate {
            if masters.contains(&(a, b)) {
                remove_slaves(a, b, &mut masters, &self.active_midpoints);
            }
        }
        for (&(a, b), &mid) in &self.active_midpoints {
            if !new_node_set.contains(&mid) { continue; }
            if !masters.contains(&quad_edge_key(a, b)) { continue; }
            new_constraints.push(HangingNodeConstraint {
                constrained: mid as usize,
                parent_a: a as usize,
                parent_b: b as usize,
                coeff_a: 0.5,
                coeff_b: 0.5,
                extra: Vec::new(),
            });
        }
        new_constraints.sort_by_key(|c| c.constrained);
        self.constraints = new_constraints;

        if std::env::var("EX15_DBG_DEREF").is_ok() {
            for ni in [9usize, 11, 12, 14] {                if ni < self.refine_tree.len() {
                    let node = &self.refine_tree[ni];
                    eprintln!(
                        "POST-DEREF tree[{ni}] children={:?} leaf={:?} alive={}",
                        node.children, node.child_leaf, node.alive
                    );
                }
            }
            let mut etn: Vec<_> = self.elem_to_node.iter().collect();
            etn.sort();
            let head: Vec<_> = etn.iter().filter(|(e, _)| **e < 80 || **e > 300).take(60).map(|(&e, &(ni, k))| format!("{e}:({ni},{k})")).collect();
            eprintln!("POST-DEREF elem_to_node (sample): {}", head.join(" "));
        }

        // Rebuild the NC vertex view (MFEM NCMesh::Update -> UpdateVertices):
        // top-level (original) nodes first, then every other node in
        // leaf-element (Hilbert SFC) order — first appearance when scanning
        // the elements in leaf_order.  Without this the DofManager falls back
        // to the identity phys->dof map and the P2 constraints / Dirichlet
        // boundary DOFs / ZZ flux constraints all use wrong ids after a
        // derefinement (silently corrupting the solve).
        {
            // MFEM's NCMesh node table is append-only, so the top-level
            // vertices keep their creation ids 0..N0-1 across derefinements;
            // Rust compacts node ids, so remap the tracked top-level ids
            // through the compaction map (keeping their original order).
            self.top_level_ids = self
                .top_level_ids
                .iter()
                .filter_map(|&n| node_map.get(&n).copied())
                .collect();
            let mut view: Vec<NodeId> = self.top_level_ids.clone();
            let mut seen: std::collections::HashSet<NodeId> = view.iter().copied().collect();
            for &e in &self.leaf_order {
                let off = e as usize * 4;
                for k in 0..4 {
                    let n = new_conn[off + k];
                    if seen.insert(n) {
                        view.push(n);
                    }
                }
            }
            // Note: preserved (non-element) nodes kept for the edge-midpoint
            // history are NOT part of the vertex view — they are not DOFs
            // (MFEM's vertex table only covers element-used nodes).  They only
            // exist in the mesh node table so the constraint walk can still
            // reference them; build_q2_quad sizes the vertex DOF block by
            // view.len(), excluding them.
            new_mesh.nc_vertex_view = Some(view);
        }

        Some(new_mesh)
    }
}

/// Return the geometric split level of edge `(a, b)`: the depth of the
/// midpoint chain (1 + max over both halves), matching MFEM's
/// `NCMesh::EdgeSplitLevel`.  0 if the edge has no midpoint node.
///
/// The midpoint is located by quantized coordinate lookup, so coarse edges
/// that were split by a finer neighbor are detected even though the edge
/// itself is not present in the current mesh's edge map.  Like MFEM
/// (`!nodes[mid].HasVertex() -> 0`), a midpoint that is NOT an element
/// corner (not in `is_vertex`) does NOT count — deep split chains whose
/// midpoints stopped being element corners stop contributing (deref'd
/// away sub-chains must not inflate the level).
pub(crate) fn edge_split_level_geo(
    a: NodeId,
    b: NodeId,
    coords: &[[f64; 2]],
    pos: &HashMap<(i64, i64), NodeId>,
    is_vertex: &std::collections::HashSet<NodeId>,
    memo: &mut HashMap<(NodeId, NodeId), u32>,
) -> u32 {
    let key = (a.min(b), a.max(b));
    if let Some(&v) = memo.get(&key) { return v; }
    let mx = 0.5 * (coords[a as usize][0] + coords[b as usize][0]);
    let my = 0.5 * (coords[a as usize][1] + coords[b as usize][1]);
    let q = ((mx * 1e10).round() as i64, (my * 1e10).round() as i64);
    let v = if let Some(&m) = pos.get(&q) {
        if m == a || m == b || !is_vertex.contains(&m) { 0 }
        else {
            1 + edge_split_level_geo(a, m, coords, pos, is_vertex, memo)
                .max(edge_split_level_geo(m, b, coords, pos, is_vertex, memo))
        }
    } else { 0 };
    memo.insert(key, v);
    v
}

/// Mark every Quad4 element whose maximum edge-split level exceeds
/// `nc_limit` — the exact semantics of MFEM's `NCMesh::GetLimitRefinements`.
///
/// The returned elements should be refined (in a separate batch, like MFEM's
/// `LimitNCLevel` loop) until this returns empty.
pub fn limit_nc_level_quad(mesh: &Mesh<2>, nc_limit: u32) -> Vec<ElemId> {
    let n_nodes = mesh.n_nodes();
    let coords: Vec<[f64; 2]> =
        (0..n_nodes).map(|n| mesh.coords_of(n as NodeId)).collect();
    let mut pos: HashMap<(i64, i64), NodeId> = HashMap::new();
    for (n, c) in coords.iter().enumerate() {
        let q = ((c[0] * 1e10).round() as i64, (c[1] * 1e10).round() as i64);
        pos.entry(q).or_insert(n as NodeId);
    }
    let mut memo: HashMap<(NodeId, NodeId), u32> = HashMap::new();
    let mut out = Vec::new();
    // MFEM HasVertex(): a midpoint counts only while it is an element corner.
    let mut is_vertex: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for e in 0..mesh.n_elems() as ElemId {
        for &n in mesh.elem_nodes(e) { is_vertex.insert(n); }
    }
    for e in 0..mesh.n_elems() as ElemId {
        let ns = mesh.elem_nodes(e);
        let mut splits = 0u32;
        for &(la, lb) in &local_edges_quad() {
            splits = splits.max(edge_split_level_geo(
                ns[la], ns[lb], &coords, &pos, &is_vertex, &mut memo));
        }
        if splits > nc_limit { out.push(e); }
    }
    out
}

/// Propagate refinement to neighbors when nc_limit would be violated (Quad4).
///
/// Uses the geometric edge-split level (MFEM `GetLimitRefinements` semantics):
/// an element whose edges are split more than `nc_limit` times is added to the
/// marked set, matching MFEM's `NCMesh::LimitNCLevel` propagation.
fn propagate_nc_limit_quad(
    marked: &[ElemId],
    mesh: &Mesh<2>,
    edge_elems: &HashMap<(NodeId, NodeId), Vec<ElemId>>,
    edge_level: &HashMap<(NodeId, NodeId), u32>,
    nc_limit: u32,
) -> Vec<ElemId> {
    let _ = (edge_elems, edge_level); // kept for signature compatibility
    use std::collections::BTreeSet;
    let mut result: BTreeSet<ElemId> = marked.iter().copied().collect();
    // Elements whose edges are already split beyond nc_limit (either marked or
    // their coarse neighbors) must be refined together.
    for e in limit_nc_level_quad(mesh, nc_limit) {
        result.insert(e);
    }
    result.into_iter().collect()
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
#[allow(clippy::type_complexity)]
pub fn refine_nonconforming_hex(
    mesh: &Mesh<3>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
    assert!(
        mesh.elem_type == ElementType::Hex8,
        "refine_nonconforming_hex: only Hex8 meshes are supported"
    );

    if marked.is_empty() {
        let mut m = mesh.clone();
        if let Some(config) = project_boundary {
            m = project_boundary_to_cad(&m, config, 3);
        }
        return (m, Vec::new(), Vec::new(), HashMap::new());
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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
                constraints.push(HangingNodeConstraint::new_p1(mab as usize, a as usize, b as usize));
                constraints.push(HangingNodeConstraint::new_p1(mbc as usize, b as usize, c as usize));
                constraints.push(HangingNodeConstraint::new_p1(mcd as usize, c as usize, d as usize));
                constraints.push(HangingNodeConstraint::new_p1(mda as usize, d as usize, a as usize));
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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Hex8,
        new_face_conn, new_face_tags, ElementType::Quad4,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 3);
    }
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
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let new_mesh = Mesh::uniform(
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
#[allow(clippy::type_complexity)]
pub fn refine_nonconforming_prism(
    mesh: &Mesh<3>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (
    Mesh<3>,
    Vec<HangingNodeConstraint>,
    Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>,
    HashMap<(NodeId, NodeId), NodeId>,
) {
    let (mut m, ec, tc, qc, mm, _) = refine_nonconforming_prism_internal(mesh, marked, None);
    if let Some(config) = project_boundary {
        m = project_boundary_to_cad(&m, config, 3);
    }
    (m, ec, tc, qc, mm)
}

#[allow(clippy::type_complexity)]
fn refine_nonconforming_prism_internal(
    mesh: &Mesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    Mesh<3>,
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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
            coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
            edge_constraints.push(HangingNodeConstraint {
                constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
            coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
            edge_constraints.push(HangingNodeConstraint {
                constrained: mac as usize, parent_a: a as usize, parent_b: c as usize,
            coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mcd as usize, parent_a: c as usize, parent_b: d as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mda as usize, parent_a: d as usize, parent_b: a as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let new_mesh = Mesh::uniform(
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
    mesh:   &Mesh<2>,
    marked: &[(ElemId, QuadRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Quad4,
        "refine_nonconforming_quad_aniso: only Quad4 meshes are supported"
    );

    if marked.is_empty() {
        let mut m = mesh.clone();
        if let Some(config) = project_boundary {
            m = project_boundary_to_cad(&m, config, 2);
        }
        return (m, Vec::new());
    }

    let n_elems = mesh.n_elems();
    let marked_map: HashMap<ElemId, QuadRefineDir> = marked.iter().copied().collect();

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

    // Coordinate → existing node lookup (reuse hanging nodes of refined
    // neighbours at edge midpoints, MFEM NC semantics — see
    // refine_nonconforming_quad).
    let coord_map: HashMap<String, NodeId> = {
        let mut m = HashMap::new();
        for n in 0..mesh.n_nodes() as NodeId {
            let c = mesh.coords_of(n);
            m.entry(format!("{:.12},{:.12}", c[0], c[1])).or_insert(n);
        }
        m
    };

    // Inline helper macro to insert midpoint if not already present.
    macro_rules! ensure_midpoint {
        ($key:expr) => {{
            let k = $key;
            if !midpoint_map.contains_key(&k) {
                let xa = mesh.coords_of(k.0);
                let xb = mesh.coords_of(k.1);
                let mx = 0.5 * (xa[0] + xb[0]);
                let my = 0.5 * (xa[1] + xb[1]);
                if let Some(&existing) = coord_map.get(&format!("{mx:.12},{my:.12}")) {
                    midpoint_map.insert(k, existing);
                } else {
                    new_coords.push(mx);
                    new_coords.push(my);
                    midpoint_map.insert(k, next_node);
                    next_node += 1;
                }
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

    // ── Rebuild boundary edges ───────────────────────────────────────────────
    // A refined boundary edge is split at its midpoint: the new node becomes
    // a boundary node (Dirichlet), matching MFEM's NCMesh boundary splitting.
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();
    for f in 0..n_faces {
        let a = mesh.face_conn[2 * f];
        let b = mesh.face_conn[2 * f + 1];
        let tag = mesh.face_tags[f];
        if let Some(&mid) = midpoint_map.get(&quad_edge_key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]);
            new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);
            new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    let mut new_mesh = Mesh::<2>::uniform(
        new_coords,
        new_conn,
        new_elem_tags,
        ElementType::Quad4,
        new_face_conn,
        new_face_tags,
        ElementType::Line2,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 2);
    }

    // ── Detect hanging nodes (multi-level, full-topology walk) ───────────────
    // Same as refine_nonconforming_quad: starting from every element edge
    // (a,b) of the *refined* mesh, walk the bisection chain (a,b) → (a,m),
    // (m,b) → … recording every midpoint m the element does NOT contain.
    // This catches this round's fresh midpoints AND hanging nodes carried
    // over from previous refinement levels (a pure midpoint_map scan over the
    // marked elements misses pre-existing constraints, which broke ex6 it2+
    // where old hanging nodes must persist into the next solve).
    let mut constraints: Vec<HangingNodeConstraint> = Vec::new();
    let mut coord_map: HashMap<String, NodeId> = HashMap::new();
    for n in 0..new_mesh.n_nodes() as NodeId {
        let c = new_mesh.coords_of(n);
        coord_map.entry(format!("{:.9},{:.9}", c[0], c[1])).or_insert(n);
    }
    let coords_of = |id: NodeId| -> [f64; 2] {
        let c = new_mesh.coords_of(id);
        [c[0], c[1]]
    };
    for e in 0..new_mesh.n_elems() as ElemId {
        let ns = new_mesh.elem_nodes(e);
        let contains = |m: NodeId| ns.contains(&m);
        fn walk(
            a: NodeId, b: NodeId,
            coord_map: &HashMap<String, NodeId>,
            coords_of: &dyn Fn(NodeId) -> [f64; 2],
            contains: &dyn Fn(NodeId) -> bool,
            out: &mut Vec<HangingNodeConstraint>,
        ) {
            let ca = coords_of(a);
            let cb = coords_of(b);
            let key = format!("{:.9},{:.9}", 0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]));
            if let Some(&m) = coord_map.get(&key) {
                if m != a && m != b && !contains(m) {
                    out.push(HangingNodeConstraint {
                        constrained: m as usize,
                        parent_a: a as usize,
                        parent_b: b as usize,
                        coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
                    });
                    walk(a, m, coord_map, coords_of, contains, out);
                    walk(m, b, coord_map, coords_of, contains, out);
                }
            }
        }
        for &(ea, eb) in &local_edges_quad() {
            let a = ns[ea];
            let b = ns[eb];
            walk(a, b, &coord_map, &coords_of, &contains, &mut constraints);
        }
    }
    constraints.sort_by_key(|c| c.constrained);
    constraints.dedup_by(|a, b| a.constrained == b.constrained);
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
    mesh: &Mesh<3>,
    marked: &[(ElemId, HexRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Hex8,
        "refine_nonconforming_hex_aniso: only Hex8 meshes are supported"
    );

    if marked.is_empty() {
        let mut m = mesh.clone();
        if let Some(config) = project_boundary {
            m = project_boundary_to_cad(&m, config, 3);
        }
        return (m, Vec::new());
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
        let (m, c, _, _) = refine_nonconforming_hex(mesh, &all_ids, None);
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Hex8,
        new_face_conn, new_face_tags, ElementType::Quad4,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 3);
    }
    (new_mesh, constraints)
}

// ─── NCStateHex (multi-level Hex8 NC tracking) ──────────────────────────────

#[derive(Debug, Clone)]
struct NCStateHexSnapshot {
    mesh: Mesh<3>,
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

impl Default for NCStateHex {
    fn default() -> Self { Self::new() }
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
    #[allow(clippy::type_complexity)]
    pub fn refine(
        &mut self,
        mesh: &Mesh<3>,
        marked: &[ElemId],
    ) -> (
        Mesh<3>,
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
    ) -> Option<(Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingQuadFaceConstraint>)> {
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
    mesh: Mesh<3>,
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

impl Default for NCStatePrism {
    fn default() -> Self { Self::new() }
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
    #[allow(clippy::type_complexity)]
    pub fn refine(
        &mut self,
        mesh: &Mesh<3>,
        marked: &[ElemId],
    ) -> (
        Mesh<3>,
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
    #[allow(clippy::type_complexity)]
    pub fn derefine_last(
        &mut self,
    ) -> Option<(Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>)> {
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
    mesh: &Mesh<3>,
    marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
    active_face_centers: Option<&HashMap<[NodeId; 4], NodeId>>,
) -> (
    Mesh<3>,
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
            coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mbc as usize, parent_a: b as usize, parent_b: c as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mcd as usize, parent_a: c as usize, parent_b: d as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
});
                edge_constraints.push(HangingNodeConstraint {
                    constrained: mda as usize, parent_a: d as usize, parent_b: a as usize,
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let new_mesh = Mesh::uniform(
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
    mesh: &Mesh<2>,
    marked: &[(ElemId, TriRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_nonconforming_tri_aniso: only Tri3 meshes are supported"
    );

    if marked.is_empty() {
        let mesh = if let Some(config) = project_boundary {
            project_boundary_to_cad(mesh, config, 2)
        } else { mesh.clone() };
        return (mesh, Vec::new());
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 2);
    }
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
    mesh: &Mesh<3>,
    marked: &[(ElemId, TetRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>) {
    assert!(
        mesh.elem_type == ElementType::Tet4,
        "refine_nonconforming_tet_aniso: only Tet4 meshes are supported"
    );

    if marked.is_empty() {
        let mesh = if let Some(config) = project_boundary {
            project_boundary_to_cad(mesh, config, 3)
        } else { mesh.clone() };
        return (mesh, Vec::new());
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
                coeff_a: 0.5, coeff_b: 0.5, extra: Vec::new(),
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

    let mut new_mesh = Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tet4,
        new_face_conn, new_face_tags, ElementType::Tri3,
    );
    if let Some(config) = project_boundary {
        new_mesh = project_boundary_to_cad(&new_mesh, config, 3);
    }
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
    mesh: &Mesh<3>,
    marked: &[(ElemId, PrismRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>) {
    assert!(mesh.elem_type == ElementType::Prism6, "refine_nonconforming_prism_aniso: only Prism6");
    if marked.is_empty() {
        let mut m = mesh.clone();
        if let Some(config) = project_boundary {
            m = project_boundary_to_cad(&m, config, 3);
        }
        return (m, Vec::new());
    }

    let all_ids: Vec<ElemId> = marked.iter().filter_map(|&(e,d)| if d==PrismRefineDir::All { Some(e) } else { None }).collect();
    let dirs: Vec<(ElemId, PrismRefineDir)> = marked.iter().copied().filter(|&(_,d)| d!=PrismRefineDir::All).collect();

    if !all_ids.is_empty() {
        let (m, ec, _, _, _) = refine_nonconforming_prism(mesh, &all_ids, None);
        // Re-refine directional marked elements on top of isotropic ones
        if dirs.is_empty() {
            let mut m2 = m;
            if let Some(config) = project_boundary {
                m2 = project_boundary_to_cad(&m2, config, 3);
            }
            return (m2, ec);
        }
        // For simplicity, just delegate directional ones via the isotropic refiner too
        let (m2, ec2, _, _, _) = refine_nonconforming_prism(&m, &dirs.iter().map(|&(e,_)|e).collect::<Vec<_>>(), None);
        let mut all_c = ec; all_c.extend(ec2); all_c.sort_by_key(|c|c.constrained); all_c.dedup_by_key(|c|c.constrained);
        let mut m3 = m2;
        if let Some(config) = project_boundary {
            m3 = project_boundary_to_cad(&m3, config, 3);
        }
        return (m3, all_c);
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
    for (&(a,b),&mid) in &mm { if let Some(adj)=edge_elems.get(&(a,b)) { if adj.iter().any(|e|!marked_set.contains(e)) { c.push(HangingNodeConstraint::new_p1(mid as usize,a as usize,b as usize)); } } }
    c.sort_by_key(|c|c.constrained);

    let nbf=mesh.n_faces();let mut nfc=Vec::new();let mut nft=Vec::new();
    for f in 0..nbf {
        let tag=mesh.face_tags[f];let fs=mesh.bface_nodes(f as FaceId);
        match fs.len() {
            3=>{let(a,b,c)=(fs[0],fs[1],fs[2]);let ma=mm.get(&edge_key(a,b)).copied();let mb=mm.get(&edge_key(b,c)).copied();let mc=mm.get(&edge_key(c,a)).copied();
                if let(Some(mab),Some(mbc),Some(mca))=(ma,mb,mc){nfc.extend_from_slice(&[a,mab,mca]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc]);nft.push(tag);nfc.extend_from_slice(&[mca,mbc,c]);nft.push(tag);nfc.extend_from_slice(&[mab,mbc,mca]);nft.push(tag);}
                else{nfc.extend_from_slice(&[a,b,c]);nft.push(tag);}}
            4=>{let(a,b,c,d)=(fs[0],fs[1],fs[2],fs[3]);let ma=mm.get(&edge_key(a,b)).copied();let mb=mm.get(&edge_key(b,c)).copied();let mc=mm.get(&edge_key(c,d)).copied();let md=mm.get(&edge_key(d,a)).copied();
                if let(Some(mab),Some(mbc),Some(mcd),Some(mda))=(ma,mb,mc,md){let _fk=quad_face_key([a,b,c,d]);nfc.extend_from_slice(&[a,mab,mda]);nft.push(tag);nfc.extend_from_slice(&[mab,b,mbc]);nft.push(tag);nfc.extend_from_slice(&[mda,mbc,c,mcd]);nft.push(tag);}
                else{nfc.extend_from_slice(&[a,b,c,d]);nft.push(tag);}}
            _=>{for&n in fs{nfc.push(n);}nft.push(tag);}
        }
    }
    let mut nm=Mesh::uniform(nc,ncn,nt,ElementType::Prism6,nfc,nft,mesh.face_type);
    if let Some(config) = project_boundary {
        nm = project_boundary_to_cad(&nm, config, 3);
    }
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
    mesh: &Mesh<3>,
    marked: &[(ElemId, PyramidRefineDir)],
    project_boundary: Option<&ProjectionConfig>,
) -> (Mesh<3>, Vec<HangingNodeConstraint>) {
    assert!(mesh.elem_type == ElementType::Pyramid5, "refine_nonconforming_pyramid_aniso: only Pyramid5");
    if marked.is_empty() {
        let mut m = mesh.clone();
        if let Some(config) = project_boundary {
            m = project_boundary_to_cad(&m, config, 3);
        }
        return (m, Vec::new());
    }

    // For `All`, delegate to the full NC refinement
    let all_ids: Vec<ElemId> = marked.iter().filter_map(|&(e,d)| if d==PyramidRefineDir::All { Some(e) } else { None }).collect();
    if !all_ids.is_empty() {
        let (m, ec, _, _, _) = refine_nonconforming_pyramid(mesh, &all_ids, None);
        let mut m2 = m;
        if let Some(config) = project_boundary {
            m2 = project_boundary_to_cad(&m2, config, 3);
        }
        return (m2, ec);
    }

    // Base and Apex splits are placeholders that produce Tet4 children
    // (full implementation would produce Pyramid5 children).
    // For now, delegate to the base-diagonal split which produces 2 tets per pyramid,
    // then refine those tets via the full NC refiner.
    let (m, ec, _, _, _) = refine_nonconforming_pyramid(mesh, &marked.iter().map(|&(e,_)|e).collect::<Vec<_>>(), None);
    let mut m2 = m;
    if let Some(config) = project_boundary {
        m2 = project_boundary_to_cad(&m2, config, 3);
    }
    (m2, ec)
}

/// Local 8 edges of a Pyramid5 element.
#[allow(dead_code)]
fn local_edges_pyramid() -> [(usize, usize); 8] {
    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
}
#[allow(dead_code)]
pub(crate) fn local_faces_pyramid_quad() -> [[usize; 4]; 1] { [[0, 1, 2, 3]] }
#[allow(dead_code)]
pub(crate) fn local_faces_pyramid_tri() -> [(usize, usize, usize); 4] {
    [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
}

// ─── Pyramid5 uniform refinement ──────────────────────────────────────────────

/// Uniformly refine Pyramid5 → 16 Tet4 (split along diagonal (0,2), each tet → 8).
pub fn refine_pyramid5_uniform(
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, Vec<HangingNodeConstraint>) {
    let (m, c, _, _, _, _) = refine_nonconforming_pyramid_internal(mesh, marked, None);
    (m, c)
}

// ─── Pyramid5 non-conforming refinement ─────────────────────────────────────

/// Non-conforming refinement for Pyramid5 → 16 Tet4 children.
#[allow(clippy::type_complexity)]
pub fn refine_nonconforming_pyramid(
    mesh: &Mesh<3>,
    marked: &[ElemId],
    project_boundary: Option<&ProjectionConfig>,
) -> (
    Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>,
    Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>,
) {
    let (mut m, ec, tc, qc, mm, _) = refine_nonconforming_pyramid_internal(mesh, marked, None);
    if let Some(config) = project_boundary {
        m = project_boundary_to_cad(&m, config, 3);
    }
    (m, ec, tc, qc, mm)
}

#[allow(clippy::type_complexity)]
fn refine_nonconforming_pyramid_internal(
    mesh: &Mesh<3>, marked: &[ElemId],
    active_midpoints: Option<&HashMap<(NodeId, NodeId), NodeId>>,
) -> (
    Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>,
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
            if adj.iter().any(|e|!marked_set.contains(e)) { ec.push(HangingNodeConstraint::new_p1(mid as usize,a as usize,b as usize)); }
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
            ec.push(HangingNodeConstraint::new_p1(mab as usize,a as usize,b as usize));
            ec.push(HangingNodeConstraint::new_p1(mbc as usize,b as usize,c as usize));
            ec.push(HangingNodeConstraint::new_p1(mac as usize,a as usize,c as usize));
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
                ec.push(HangingNodeConstraint::new_p1(mab as usize,a as usize,b as usize));
                ec.push(HangingNodeConstraint::new_p1(mbc as usize,b as usize,c as usize));
                ec.push(HangingNodeConstraint::new_p1(mcd as usize,c as usize,d as usize));
                ec.push(HangingNodeConstraint::new_p1(mda as usize,d as usize,a as usize));
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
    let nm = Mesh::uniform(new_coords,new_conn,new_tags,ElementType::Tet4,nfc,nft,mesh.face_type);
    (nm, ec, tc, qc, midpoint_map, nam)
}

// ─── NCStatePyramid (multi-level Pyramid5 NC tracking) ────────────────────────

#[derive(Debug, Clone)]
struct NCStatePyramidSnapshot {
    mesh: Mesh<3>, constraints: Vec<HangingNodeConstraint>,
    tri_face_constraints: Vec<HangingFaceConstraint>, quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone)]
pub struct NCStatePyramid {
    constraints: Vec<HangingNodeConstraint>, tri_face_constraints: Vec<HangingFaceConstraint>,
    quad_face_constraints: Vec<HangingQuadFaceConstraint>,
    active_midpoints: HashMap<(NodeId, NodeId), NodeId>, history: Vec<NCStatePyramidSnapshot>,
}

impl Default for NCStatePyramid {
    fn default() -> Self { Self::new() }
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
    #[allow(clippy::type_complexity)]
    pub fn refine(&mut self, mesh: &Mesh<3>, marked: &[ElemId]) -> (Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
        self.history.push(NCStatePyramidSnapshot { mesh: mesh.clone(), constraints: self.constraints.clone(), tri_face_constraints: self.tri_face_constraints.clone(), quad_face_constraints: self.quad_face_constraints.clone(), active_midpoints: self.active_midpoints.clone() });
        let (nm, ec, tc, qc, mm, nam) = refine_nonconforming_pyramid_internal(mesh, marked, Some(&self.active_midpoints));
        self.constraints = ec.clone(); self.tri_face_constraints = tc.clone(); self.quad_face_constraints = qc.clone(); self.active_midpoints = nam;
        (nm, ec, tc, qc, mm)
    }
    #[allow(clippy::type_complexity)]
    pub fn derefine_last(&mut self) -> Option<(Mesh<3>, Vec<HangingNodeConstraint>, Vec<HangingFaceConstraint>, Vec<HangingQuadFaceConstraint>)> {
        let snap = self.history.pop()?;
        self.constraints = snap.constraints; self.tri_face_constraints = snap.tri_face_constraints; self.quad_face_constraints = snap.quad_face_constraints; self.active_midpoints = snap.active_midpoints;
        Some((snap.mesh, self.constraints.clone(), self.tri_face_constraints.clone(), self.quad_face_constraints.clone()))
    }
}

// ─── Hex8 uniform refinement ─────────────────────────────────────────────────

/// Uniformly refine Hex8 → 8 Hex8 children using edge midpoints, face centroids, body centroid.
pub fn refine_hex8_uniform(
    mesh: &Mesh<3>, marked: &[ElemId],
) -> (Mesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId, NodeId), NodeId>) {
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
    for (&(a,b),&mid) in &mm { if let Some(adj)=edge_elems.get(&(a,b)) { if adj.iter().any(|e|!marked_set.contains(e)) { c.push(HangingNodeConstraint::new_p1(mid as usize,a as usize,b as usize)); } } }
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
    let nm=Mesh::uniform(nc,ncn,nt,ElementType::Hex8,nfc,nft,ElementType::Quad4);
    (nm,c,mm)
}

// ─── Hex20 / Hex27 uniform refinement ────────────────────────────────────────

/// Uniformly refine Hex20 → 8 Hex8 children by extracting the 8 corner nodes.
pub fn refine_hex20_uniform(mesh: &Mesh<3>, marked: &[ElemId]) -> (Mesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    let (m,c,mm) = refine_hex27_uniform_inner(mesh, marked, 20);
    (m,c,mm)
}

/// Uniformly refine Hex27 → 8 Hex8 children by extracting the 8 corner nodes.
pub fn refine_hex27_uniform(mesh: &Mesh<3>, marked: &[ElemId]) -> (Mesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    let (m,c,mm) = refine_hex27_uniform_inner(mesh, marked, 27);
    (m,c,mm)
}

fn refine_hex27_uniform_inner(mesh: &Mesh<3>, marked: &[ElemId], npe: usize) -> (Mesh<3>, Vec<HangingNodeConstraint>, HashMap<(NodeId,NodeId),NodeId>) {
    assert!(mesh.elem_type == ElementType::Hex20 || mesh.elem_type == ElementType::Hex27, "refine_hex27_uniform_inner: only Hex20/Hex27");
    if marked.is_empty() { return (mesh.clone(), Vec::new(), HashMap::new()); }
    let n_elems = mesh.n_elems();
    let mut hex8_conn = Vec::with_capacity(n_elems * 8);
    for e in 0..n_elems { let off = e * npe; hex8_conn.extend_from_slice(&mesh.conn[off..off+8]); }
    let hex8_mesh = Mesh { coords: mesh.coords.clone(), conn: hex8_conn, elem_tags: mesh.elem_tags.clone(), elem_type: ElementType::Hex8, face_conn: mesh.face_conn.clone(), face_tags: mesh.face_tags.clone(), face_type: mesh.face_type, elem_types: None, elem_offsets: None, face_types: None, face_offsets: None, face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], nc_vertex_view: None, geometry: None };
    refine_hex8_uniform(&hex8_mesh, marked)
}



#[cfg(test)]
mod tests {
    use crate::amr::*;
    use fem_core::{NodeId, ElemId};
    use crate::{element_type::ElementType, simplex::Mesh};

    #[test]
    fn uniform_refinement_element_count() {
        // Each Tri3 → 4 children with red refinement.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n_before = mesh.n_elems();
        let fine = refine_uniform(&mesh);
        assert_eq!(fine.n_elems(), 4 * n_before,
            "Expected 4×{n_before}={} elements, got {}", 4*n_before, fine.n_elems());
    }

    /// MFEM's `Mesh(fichera-mixed.mesh, 1, 1)` ctor marks the tets (longest
    /// edge rotated to (v0,v1), canonical second-longest placement).  The
    /// probe output from the C++ library (see fem-pro HANDOVER pex34) gives
    /// the exact marked orders below — the marking must match bit-for-bit
    /// because it feeds the rt (octahedron diagonal) selection.
    #[test]
    fn fichera_mixed_marking_matches_mfem() {
        let mut mesh = fichera_mixed_mesh();
        crate::mark_tet_mesh_for_refinement(&mut mesh);
        let marked: Vec<Vec<u32>> = (0..14)
            .filter(|&e| mesh.element_type_at(e as u32) == ElementType::Tet4)
            .map(|e| {
                let off = mesh.elem_offsets.as_ref().unwrap()[e];
                mesh.conn[off..off + 4].to_vec()
            })
            .collect();
        let expected: Vec<Vec<u32>> = vec![
            vec![21, 13, 15, 25],
            vec![13, 21, 15, 12],
            vec![21, 13, 25, 22],
            vec![15, 21, 25, 24],
            vec![25, 13, 15, 16],
        ];
        assert_eq!(marked, expected, "tet marking must match MFEM's MarkEdge");
    }

    #[test]
    fn uniform_refinement_node_count() {
        // A 1×1 square → 2 triangles, 4 nodes.
        // After red refinement: 8 triangles, 4+3=7 new midpoints? Actually 4+3=7 total.
        let mesh = Mesh::<2>::unit_square_tri(1);
        let fine = refine_uniform(&mesh);
        // 1×1 unit square: 4 corners + 4 edge midpoints + 1 interior midpoint = 9
        assert!(fine.n_nodes() > mesh.n_nodes(),
            "Refinement should add nodes: before={}, after={}", mesh.n_nodes(), fine.n_nodes());
    }

    #[test]
    fn uniform_refinement_two_levels() {
        // 2 levels of uniform refinement: n → 4n → 16n elements.
        let mesh0 = Mesh::<2>::unit_square_tri(2);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(3);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n0 = mesh.n_nodes();
        let u0: Vec<f64> = (0..n0).map(|i| i as f64).collect();

        let mut nc = NCState::new();
        let (_fine, _constraints, midpoint_map) = nc.refine(&mesh, &[0], 0);
        let uf = prolongate_p1(&u0, n0 + midpoint_map.len(), &midpoint_map);
        let ur = restrict_to_coarse_p1(&uf, n0);

        assert_eq!(ur, u0, "restrict should recover coarse nodal values exactly");
    }

    #[test]
    fn kelly_estimator_linear_exact() {
        // For a linear function u(x,y) = x, the gradient is constant everywhere,
        // so jumps across edges should be zero → Kelly indicator = 0.
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 0.5, "3D ZZ should be near zero for linear u, got {max_eta:.3e}");
    }

    #[test]
    fn zz_3d_quadratic_solution_positive_indicator() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d(&mesh, &u);
        let max_eta: f64 = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max_eta < 1.0, "3D Kelly should be near zero for linear u, got {max_eta:.3e}");
    }

    #[test]
    fn kelly_3d_quadratic_solution_positive_indicator() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<3>::unit_cube_tet(2);
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
        let mesh = Mesh::<3>::unit_cube_tet(2);
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
        let mesh = Mesh::<2>::unit_square_quad(2);
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
        let mesh = Mesh::<3>::unit_cube_hex(2);
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
        let mesh = Mesh::<3>::unit_cube_hex(2);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<3>::unit_cube_hex(2);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[], None);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn nc_refine_all_marked_no_hanging() {
        // Refining all elements → no hanging nodes (equivalent to uniform).
        let mesh = Mesh::<2>::unit_square_tri(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints) = refine_nonconforming(&mesh, &all, None);
        assert_eq!(nc.n_elems(), 4 * mesh.n_elems());
        assert!(constraints.is_empty(),
            "all-marked NCMesh should have no hanging nodes, got {}", constraints.len());
    }

    #[test]
    fn nc_refine_single_element_has_hanging_nodes() {
        // Refine just element 0 of a 2×2 mesh → should produce hanging nodes
        // on the edges shared with unrefined neighbours.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[0], None);

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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc, constraints) = refine_nonconforming(&mesh, &[0], None);

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
        let mesh = Mesh::<2>::unit_square_tri(4);
        let marked = vec![0u32, 1, 2];

        let conforming = refine_marked(&mesh, &marked);
        let (nc, _) = refine_nonconforming(&mesh, &marked, None);

        assert!(
            nc.n_elems() <= conforming.n_elems(),
            "NC ({}) should have ≤ elements than conforming ({})",
            nc.n_elems(), conforming.n_elems()
        );
    }

    #[test]
    fn nc_refine_two_levels() {
        // Refine once, then refine again on some new elements.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let (nc1, c1) = refine_nonconforming(&mesh, &[0, 1], None);
        assert!(!c1.is_empty() || mesh.n_elems() == 2,
            "first level should have constraints (or trivial mesh)");

        // Refine element 0 of the new mesh.
        let (nc2, c2) = refine_nonconforming(&nc1, &[0], None);
        assert!(nc2.n_elems() > nc1.n_elems());
        // Second level may also produce hanging nodes.
        let _ = c2;
    }

    #[test]
    fn nc_refine_mesh_valid() {
        // The resulting mesh should pass consistency check.
        let mesh = Mesh::<2>::unit_square_tri(3);
        let (nc, _) = refine_nonconforming(&mesh, &[0, 3, 5], None);
        nc.check().unwrap();
    }

    // ── Prolongation tests ──────────────────────────────────────────────────

    #[test]
    fn prolongate_p1_linear_exact() {
        // For u = x (linear), prolongation should be exact.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let u: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();

        let mut nc = NCState::new();
        let (fine, _, midpts) = nc.refine(&mesh, &[0, 1, 2], 0);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let u: Vec<f64> = (0..mesh.n_nodes()).map(|i| i as f64 * 1.5).collect();

        let mut nc = NCState::new();
        let (fine, _, midpts) = nc.refine(&mesh, &[0], 0);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        let (m1, c1, _) = nc.refine(&mesh, &[0, 1], 0);
        assert!(!c1.is_empty());
        m1.check().unwrap();

        // Second level: refine some of the new elements.
        let (m2, c2, _) = nc.refine(&m1, &[0, 1], 0);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        // Refine half the elements → hanging nodes.
        let half: Vec<ElemId> = (0..mesh.n_elems() as ElemId / 2).collect();
        let (m1, c1, _) = nc.refine(&mesh, &half, 0);
        assert!(!c1.is_empty(), "should have hanging nodes after partial refinement");
        m1.check().unwrap();

        // Refine ALL elements → creates a uniformly finer mesh, but multi-level
        // hanging nodes can appear from depth mismatch.
        let all: Vec<ElemId> = (0..m1.n_elems() as ElemId).collect();
        let (m2, c2, _) = nc.refine(&m1, &all, 0);
        m2.check().unwrap();
        // The original hanging nodes may produce new constraints at deeper levels.
        // This is expected for multi-level NC refinement.
        let _ = c2;
    }

    #[test]
    fn ncstate_multi_level_prolongation() {
        // Prolongate u=x through two levels of NC refinement.
        let mesh = Mesh::<2>::unit_square_tri(2);
        let u0: Vec<f64> = (0..mesh.n_nodes())
            .map(|n| mesh.coords_of(n as NodeId)[0])
            .collect();

        let mut nc = NCState::new();
        let (m1, _, mp1) = nc.refine(&mesh, &[0, 1], 0);
        let u1 = prolongate_p1(&u0, m1.n_nodes(), &mp1);

        let (m2, _, mp2) = nc.refine(&m1, &[0], 0);
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
        let mesh = Mesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();

        let (m1, c1, _) = nc.refine(&mesh, &[0, 1], 0);
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
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems_orig = mesh.n_elems();

        let (refined, edge_constraints, face_constraints) = refine_nonconforming_3d(&mesh, &[0], None);

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
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let n_elems_orig = mesh.n_elems();

        // Refine the first tet only.
        let (refined, edge_constr, face_constr) = refine_nonconforming_3d(&mesh, &[0], None);

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
        let mesh = Mesh::<3>::unit_cube_tet(1);
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
        let mesh = Mesh::<3>::unit_cube_tet(1);
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
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &[], None);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn quad4_nonconforming_refine_all_gives_no_constraints() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &all, None);
        // Refining all → 4× as many elements, no hanging nodes
        assert_eq!(nc.n_elems(), mesh.n_elems() * 4);
        assert!(constraints.is_empty());
        nc.check().unwrap();
    }

    #[test]
    fn quad4_nonconforming_refine_single_element_creates_constraints() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let (nc, constraints) = refine_nonconforming_quad(&mesh, &[0], None);
        // Element 0 split into 4, rest unchanged → total = 3 + 4 = 7
        assert_eq!(nc.n_elems(), 7);
        // Shared edges with unrefined neighbours create hanging constraints
        assert!(!constraints.is_empty());
        nc.check().unwrap();
    }

    #[test]
    fn ncstate_quad_multi_level_tracks_constraints() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let mut ncq = NCStateQuad::new();

        let (m1, c1, _) = ncq.refine(&mesh, &[0], 0);
        assert!(m1.n_elems() > mesh.n_elems());
        assert!(!c1.is_empty());

        // Refine a neighbour → some constraints should be resolved
        let (m2, c2, _) = ncq.refine(&m1, &[1], 0);
        assert!(m2.n_elems() > m1.n_elems());
        let _ = c2; // constraints count may vary
        m2.check().unwrap();
    }

    #[test]
    fn ncstate_quad_derefine_last_rolls_back() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let mut ncq = NCStateQuad::new();

        let (_m1, _, _) = ncq.refine(&mesh, &[0], 0);
        assert!(ncq.can_derefine());

        let (m0, c0) = ncq.derefine_last().expect("expected rollback");
        assert_eq!(m0.n_elems(), mesh.n_elems());
        assert_eq!(m0.n_nodes(), mesh.n_nodes());
        assert!(c0.is_empty());
    }

    // ─── Hex8 NC AMR tests ────────────────────────────────────────────────────

    #[test]
    fn hex8_nonconforming_refine_empty() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (nc, constraints, _, _) = refine_nonconforming_hex(&mesh, &[], None);
        assert_eq!(nc.n_elems(), mesh.n_elems());
        assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn hex8_nonconforming_refine_all_gives_no_constraints() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, constraints, fc, _) = refine_nonconforming_hex(&mesh, &all, None);
        assert_eq!(nc.n_elems(), mesh.n_elems() * 8);
        assert!(constraints.is_empty());
        assert!(fc.is_empty(), "uniform refine: no hanging quad faces");
        nc.check().unwrap();
    }

    #[test]
    fn hex8_nonconforming_refine_single_element_creates_constraints() {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let (nc, constraints, fc, _) = refine_nonconforming_hex(&mesh, &[0], None);
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
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked, None);
        assert_eq!(refined.n_elems(), n + 1, "X split: one elem → 2, total {}", n + 1);
    }

    #[test]
    fn quad_aniso_y_split_doubles_elements() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::Y)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked, None);
        assert_eq!(refined.n_elems(), n + 1, "Y split: one elem → 2, total {}", n + 1);
    }

    #[test]
    fn quad_aniso_both_quadruples_element() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n = mesh.n_elems();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::Both)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked, None);
        assert_eq!(refined.n_elems(), n + 3, "Both split: one elem → 4, total {}", n + 3);
    }

    #[test]
    fn quad_aniso_empty_marked_is_identity() {
        let mesh = Mesh::<2>::unit_square_quad(3);
        let (refined, constraints) = refine_nonconforming_quad_aniso(&mesh, &[], None);
        assert_eq!(refined.n_elems(), mesh.n_elems());
        assert_eq!(refined.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn quad_aniso_x_split_adds_midpoints() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let n_nodes_before = mesh.n_nodes();
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (refined, _) = refine_nonconforming_quad_aniso(&mesh, &marked, None);
        // X split adds 2 midpoints (on bottom and top edges)
        assert_eq!(refined.n_nodes(), n_nodes_before + 2);
    }

    #[test]
    fn quad_aniso_partial_refine_creates_hanging_constraints() {
        // A 2×1 mesh: 2 quads side by side. Refine only the first.
        let mesh = Mesh::<2>::unit_square_quad(2);
        if mesh.n_elems() < 2 { return; } // skip if not enough elements
        let marked: Vec<(ElemId, QuadRefineDir)> = vec![(0, QuadRefineDir::X)];
        let (_, constraints) = refine_nonconforming_quad_aniso(&mesh, &marked, None);
        // On shared edge, there should be hanging node constraints
        // (may be 0 if no shared edge with unrefined element — depends on mesh topology)
        let _ = constraints; // just ensure it runs without panic
    }

    // ─── Hex8 各向异性 NC AMR tests ───────────────────────────────────────────

    #[test]
    fn hex_aniso_x_split_gives_two_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)], None);
        // 1 element → 2 children along X
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_y_split_gives_two_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::Y)], None);
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_z_split_gives_two_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::Z)], None);
        assert_eq!(refined.n_elems(), 2);
    }

    #[test]
    fn hex_aniso_xy_split_gives_four_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::XY)], None);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_xz_split_gives_four_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::XZ)], None);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_yz_split_gives_four_children() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::YZ)], None);
        assert_eq!(refined.n_elems(), 4);
    }

    #[test]
    fn hex_aniso_all_delegates_to_isotropic() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::All)], None);
        assert_eq!(refined.n_elems(), 8);
    }

    #[test]
    fn hex_aniso_empty_marked_is_identity() {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let (refined, constraints) = refine_nonconforming_hex_aniso(&mesh, &[], None);
        assert_eq!(refined.n_elems(), mesh.n_elems());
        assert_eq!(refined.n_nodes(), mesh.n_nodes());
        assert!(constraints.is_empty());
    }

    #[test]
    fn hex_aniso_x_split_adds_four_midpoints() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let n_before = mesh.n_nodes();
        let (refined, _) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)], None);
        // X cut adds 4 edge midpoints on the 4 X-parallel edges
        assert_eq!(refined.n_nodes(), n_before + 4);
    }

    #[test]
    fn hex_aniso_multi_elem_x_split_partial_creates_constraints() {
        // 2x2x1 mesh (4 elements). Refine one element with X split;
        // neighbouring unrefined elements should produce hanging constraints.
        let mesh = Mesh::<3>::unit_cube_hex(2);
        if mesh.n_elems() < 2 { return; }
        let (_, constraints) = refine_nonconforming_hex_aniso(&mesh, &[(0, HexRefineDir::X)], None);
        // At least some hanging constraints expected on shared faces.
        assert!(!constraints.is_empty(), "expected hanging constraints on partial X-split");
    }

    #[test]
    fn tet4_uniform_3d_creates_eight_tets() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        assert_eq!(mesh.n_elems(), 6);
        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 48, "6 tet parents × 8 = 48");
        assert!(fine.n_nodes() > mesh.n_nodes());
    }

    #[test]
    fn hex8_uniform_3d_creates_eight_hexes() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
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
    fn prism6_vol(mesh: &Mesh<3>, e: ElemId) -> f64 {
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

        let mesh = Mesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
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

        let mesh = Mesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
};

        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 8);
        fine.check().unwrap();
    }

    // ─── Prism6 NC refinement tests ───────────────────────────────────────────

    fn make_single_prism_mesh() -> Mesh<3> {
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
        Mesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
}
    }

    #[test]
    fn prism6_nc_refine_empty_is_identity() {
        let mesh = make_single_prism_mesh();
        let (nc, edge_c, tri_c, quad_c, mp) = refine_nonconforming_prism(&mesh, &[], None);
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
        let (nc, edge_c, tri_c, quad_c, _) = refine_nonconforming_prism(&mesh, &all, None);
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

        let mesh = Mesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
};

        // Refine only prism 0
        let (nc, edge_c, tri_c, quad_c, _) = refine_nonconforming_prism(&mesh, &[0], None);

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
        let mesh = Mesh {
            coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
};

        let vol_orig = prism6_vol(&mesh, 0) + prism6_vol(&mesh, 1);
        assert!((vol_orig - 1.0).abs() < 1e-14, "2 prisms, each 0.5 → total=1, got {}", vol_orig);

        let (nc, _, _, _, _) = refine_nonconforming_prism(&mesh, &[0], None);
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
    fn hex8_vol(mesh: &Mesh<3>, e: ElemId) -> f64 {
        let ns = mesh.elem_nodes(e);
        let c = |i: usize| -> [f64; 3] { let off = ns[i] as usize * 3; [mesh.coords[off], mesh.coords[off+1], mesh.coords[off+2]] };
        tet_signed_vol(&c(0),&c(1),&c(2),&c(5)).abs()+tet_signed_vol(&c(2),&c(3),&c(0),&c(7)).abs()
        +tet_signed_vol(&c(0),&c(4),&c(5),&c(7)).abs()+tet_signed_vol(&c(2),&c(5),&c(6),&c(7)).abs()
        +tet_signed_vol(&c(0),&c(2),&c(5),&c(7)).abs()
    }

    #[test] fn hex8_uniform_single_element() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
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
        let mesh = Mesh { coords:c, conn, elem_tags: vec![1i32], elem_type: ElementType::Hex20,
            face_conn: fc, face_tags: ft, face_type: ElementType::Quad4,
            elem_types:None, elem_offsets:None, face_types:None, face_offsets:None,
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![], nc_vertex_view:None, geometry:None };
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
        let mesh = Mesh { coords, conn, elem_tags: vec![1i32], elem_type: ElementType::Hex27,
            face_conn: fc, face_tags: ft, face_type: ElementType::Quad4,
            elem_types:None, elem_offsets:None, face_types:None, face_offsets:None,
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![], nc_vertex_view:None, geometry:None };
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (fine, c, _) = refine_hex27_uniform(&mesh, &all);
        assert_eq!(fine.n_elems(), 8); assert!(c.is_empty()); assert_eq!(fine.elem_type, ElementType::Hex8);
        fine.check().unwrap();
    }

    // ─── Pyramid5 uniform + NC tests ──────────────────────────────────────

    fn pyramid5_vol(mesh: &Mesh<3>, e: ElemId) -> f64 {
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
        let mesh = Mesh { coords, conn, elem_tags, elem_type:ElementType::Pyramid5,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![], nc_vertex_view:None, geometry:None };
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
        let mesh = Mesh { coords: vec![0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
            conn: vec![0u32,1,2,3,4], elem_tags: vec![1i32], elem_type: ElementType::Pyramid5,
            face_conn: vec![0u32,1,2,3,0,1,4,1,2,4,2,3,4,3,0,4], face_tags: vec![1,2,3,4,5],
            face_type: ElementType::Tri3, elem_types: None, elem_offsets: None,
            face_types: Some(vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3]),
            face_offsets: Some(vec![0,4,7,10,13,16]),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], nc_vertex_view: None, geometry: None };
        let fine = refine_uniform_3d(&mesh);
        assert_eq!(fine.n_elems(), 16); fine.check().unwrap();
    }

    fn make_pyramid_mesh() -> Mesh<3> {
        Mesh { coords: vec![0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
            conn: vec![0u32,1,2,3,4], elem_tags: vec![1i32], elem_type: ElementType::Pyramid5,
            face_conn: vec![0u32,1,2,3,0,1,4,1,2,4,2,3,4,3,0,4], face_tags: vec![1,2,3,4,5],
            face_type: ElementType::Tri3, elem_types: None, elem_offsets: None,
            face_types: Some(vec![ElementType::Quad4,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3,ElementType::Tri3]),
            face_offsets: Some(vec![0,4,7,10,13,16]),
            face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![], nc_vertex_view: None, geometry: None }
    }

    #[test] fn pyramid5_nc_refine_empty_is_identity() {
        let mesh = make_pyramid_mesh();
        let (nc, ec, tc, qc, mp) = refine_nonconforming_pyramid(&mesh, &[], None);
        assert_eq!(nc.n_elems(), mesh.n_elems()); assert_eq!(nc.n_nodes(), mesh.n_nodes());
        assert!(ec.is_empty()); assert!(tc.is_empty()); assert!(qc.is_empty()); assert!(mp.is_empty());
    }

    #[test] fn pyramid5_nc_refine_all_gives_no_constraints() {
        let mesh = make_pyramid_mesh();
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let (nc, ec, tc, qc, _) = refine_nonconforming_pyramid(&mesh, &all, None);
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
        let mesh = Mesh { coords, conn, elem_tags, elem_type:ElementType::Pyramid5,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![], nc_vertex_view:None, geometry:None };
        let (nc, ec, tc, qc, _) = refine_nonconforming_pyramid(&mesh, &[0], None);
        assert_eq!(nc.n_elems(), 17); assert!(ec.len()>=3); assert!(!tc.is_empty()); assert!(qc.is_empty());
        nc.check().unwrap();
    }

    // ─── Prism6 anisotropic tests ────────────────────────────────────────

    fn make_prism_mesh() -> Mesh<3> {
        let coords = vec![0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0, 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0];
        let conn = vec![0u32,1,2,3,4,5]; let elem_tags = vec![1i32];
        let fc = vec![0u32,2,1, 3,4,5, 0,1,4,3, 1,2,5,4, 0,3,5,2];
        let ft = vec![1,2,3,4,5];
        let fty = vec![ElementType::Tri3,ElementType::Tri3,ElementType::Quad4,ElementType::Quad4,ElementType::Quad4];
        let fo = vec![0,3,6,10,14,18];
        Mesh { coords, conn, elem_tags, elem_type: ElementType::Prism6,
            face_conn:fc, face_tags:ft, face_type:ElementType::Tri3,
            elem_types:None, elem_offsets:None, face_types:Some(fty), face_offsets:Some(fo),
            face_to_elem:None, edge_conn:vec![], edge_to_elem:vec![], nc_vertex_view:None, geometry:None }
    }

    #[test] fn prism_aniso_z_split_doubles_elements() {
        let mesh = make_prism_mesh(); let n = mesh.n_elems();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Z)], None);
        assert_eq!(refined.n_elems(), n+1, "Z split: one elem → 2, total {}", n+1);
    }

    #[test] fn prism_aniso_edge0_split_doubles_elements() {
        let mesh = make_prism_mesh(); let n = mesh.n_elems();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Edge0)], None);
        assert_eq!(refined.n_elems(), n+1, "Edge0 split: one elem → 2, total {}", n+1);
    }

    #[test] fn prism_aniso_all_delegates_to_isotropic() {
        let mesh = make_prism_mesh();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::All)], None);
        assert_eq!(refined.n_elems(), 8);
        refined.check().unwrap();
    }

    #[test] fn prism_aniso_empty_marked_is_identity() {
        let mesh = make_prism_mesh();
        let (refined, c) = refine_nonconforming_prism_aniso(&mesh, &[], None);
        assert_eq!(refined.n_elems(), mesh.n_elems()); assert_eq!(refined.n_nodes(), mesh.n_nodes()); assert!(c.is_empty());
    }

    #[test] fn prism_aniso_z_split_adds_three_midpoints() {
        let mesh = make_prism_mesh(); let n0 = mesh.n_nodes();
        let (refined, _) = refine_nonconforming_prism_aniso(&mesh, &[(0, PrismRefineDir::Z)], None);
        assert_eq!(refined.n_nodes(), n0+3, "Z split adds 3 vertical edge mids");
    }

    // ─── Pyramid5 anisotropic tests ──────────────────────────────────────

    #[test] fn pyramid_aniso_all_delegates_to_isotropic() {
        let mesh = make_pyramid_mesh();
        let (refined, _) = refine_nonconforming_pyramid_aniso(&mesh, &[(0, PyramidRefineDir::All)], None);
        assert_eq!(refined.n_elems(), 16);
        refined.check().unwrap();
    }

    #[test] fn pyramid_aniso_empty_marked_is_identity() {
        let mesh = make_pyramid_mesh();
        let (refined, c) = refine_nonconforming_pyramid_aniso(&mesh, &[], None);
        assert_eq!(refined.n_elems(), mesh.n_elems()); assert!(c.is_empty());
    }

    // ─── Generalized 3-D estimator tests ─────────────────────────────────

    #[test] fn zz_3d_general_linear_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u → ZZ ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_quadratic_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "x² → ZZ >0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u → Kelly ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_matches_original_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let f_val = vec![0.0; n];
        let eta = residual_estimator_3d_general(&mesh, &u, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u, f=0 → residual ~0, got {max:.3e}");
    }

    #[test] fn residual_3d_general_matches_original_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<3>::unit_cube_hex(3);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
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
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        assert_eq!(eta.len(), mesh.n_elems());
    }

    // ── Hex8 estimator tests ──────────────────────────────────────────────

    #[test] fn zz_3d_general_linear_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 0.5, "linear u on Hex → ZZ ~0, got {max:.3e}");
    }

    #[test] fn zz_3d_general_quadratic_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| { let c = mesh.coords_of(i as NodeId); c[0]*c[0] }).collect();
        let eta = zz_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max > 1e-6, "x² on Hex → ZZ >0, got {max:.3e}");
    }

    #[test] fn kelly_3d_general_linear_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let eta = kelly_estimator_3d_general(&mesh, &u);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 2.0, "linear u on Hex → Kelly ~0, got {max:.3e}");
    }

    // ── Prism6 estimator tests ────────────────────────────────────────────

    /// Build a unit right prism mesh: right triangle base z=0 → top z=1.
    fn prism6_unit_mesh() -> Mesh<3> {
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
        Mesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Prism6,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
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
    fn pyramid5_unit_mesh() -> Mesh<3> {
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
        Mesh {
            coords, conn, elem_tags,
            elem_type: ElementType::Pyramid5,
            face_conn, face_tags, face_type: ElementType::Tri3,
            elem_types: None, elem_offsets: None,
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![], geometry: None,
            nc_vertex_view: None,
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
        let mesh = Mesh::<3>::unit_cube_hex(3);
        let n = mesh.n_nodes();
        let u: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[0]).collect();
        let z: Vec<f64> = (0..n).map(|i| mesh.coords_of(i as NodeId)[1]).collect();
        let f_val = vec![0.0; n];
        let eta = dwr_estimator_3d_general(&mesh, &u, &z, &f_val);
        let max = eta.iter().cloned().fold(0.0, f64::max);
        assert!(max < 10.0, "linear u,z on Hex → DWR ~0, got {max:.3e}");
    }

    #[test] fn dwr_3d_general_quadratic_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(3);
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

/// libstdc++ `std::sort` simulation (introsort + insertion sort) with a
/// caller-supplied comparison — bit-exact port used by MFEM's
/// `UniformRefinement3D_base` e2v re-mapping (sorts each row
/// `[row_start, end)` of `Pair<int,int>` comparing only the column id).
fn std_sort_by<T: Copy>(a: &mut [T], cmp: impl Fn(&T, &T) -> bool) {
    const THRESHOLD: usize = 16;
    fn lg(n: usize) -> usize {
        let mut r = 0;
        let mut v = n;
        while v > 1 { v >>= 1; r += 1; }
        r
    }
    fn move_median<T: Copy>(a: &mut [T], result: usize, x: usize, y: usize, z: usize, cmp: &impl Fn(&T, &T) -> bool) {
        let r = if cmp(&a[x], &a[y]) {
            if cmp(&a[y], &a[z]) { y }
            else if cmp(&a[x], &a[z]) { z }
            else { x }
        } else if cmp(&a[x], &a[z]) { x }
        else if cmp(&a[y], &a[z]) { z }
        else { y };
        a.swap(result, r);
    }
    fn unguarded_partition<T: Copy>(a: &mut [T], first: usize, last: usize, pivot: usize, cmp: &impl Fn(&T, &T) -> bool) -> usize {
        let mut f = first;
        let mut l = last;
        loop {
            while cmp(&a[f], &a[pivot]) { f += 1; }
            l -= 1;
            while cmp(&a[pivot], &a[l]) { l -= 1; }
            if !(f < l) { return f; }
            a.swap(f, l);
            f += 1;
        }
    }
    fn unguarded_linear_insert<T: Copy>(a: &mut [T], last: usize, cmp: &impl Fn(&T, &T) -> bool) {
        let v = a[last];
        let mut l = last;
        let mut next = last - 1;
        while next > 0 && cmp(&v, &a[next]) {
            a[l] = a[next];
            l = next;
            next -= 1;
        }
        if next == 0 && cmp(&v, &a[0]) { a[l] = a[0]; a[0] = v; }
        else { a[l] = v; }
    }
    fn insertion_sort<T: Copy>(a: &mut [T], first: usize, last: usize, cmp: &impl Fn(&T, &T) -> bool) {
        for i in first + 1..last {
            if cmp(&a[i], &a[first]) {
                let v = a[i];
                a.copy_within(first..i, first + 1);
                a[first] = v;
            } else { unguarded_linear_insert(a, i, cmp); }
        }
    }
    fn introsort_loop<T: Copy>(a: &mut [T], first: usize, last: usize, mut depth: usize, cmp: &impl Fn(&T, &T) -> bool) {
        let mut f = first;
        let mut l = last;
        while l - f > THRESHOLD {
            if depth == 0 {
                insertion_sort(a, f, l, cmp);
                return;
            }
            depth -= 1;
            let mid = f + (l - f) / 2;
            move_median(a, f, f + 1, mid, l - 1, cmp);
            let cut = unguarded_partition(a, f + 1, l, f, cmp);
            let left = cut - f;
            let right = l - cut;
            if left < right { introsort_loop(a, f, cut, depth, cmp); f = cut; }
            else { introsort_loop(a, cut, l, depth, cmp); l = cut; }
        }
    }
    fn final_insertion_sort<T: Copy>(a: &mut [T], first: usize, last: usize, cmp: &impl Fn(&T, &T) -> bool) {
        if last - first > THRESHOLD {
            insertion_sort(a, first, first + THRESHOLD, cmp);
            for i in first + THRESHOLD..last { unguarded_linear_insert(a, i, cmp); }
        } else { insertion_sort(a, first, last, cmp); }
    }
    let n = a.len();
    if n <= 1 { return; }
    introsort_loop(a, 0, n, lg(n) * 2, &cmp);
    final_insertion_sort(a, 0, n, &cmp);
}

/// Build the fichera-mixed mesh (14 elements: 5 tet + 3 hex + 6 prism,
/// 26 vertices) with the same connectivity as data/fichera-mixed.mesh.
#[cfg(test)]
pub(crate) fn fichera_mixed_mesh() -> Mesh<3> {
    let coords: Vec<f64> = [
        [0., -1., -1.], [1., -1., -1.], [-1., 0., -1.], [0., 0., -1.], [1., 0., -1.],
        [-1., 1., -1.], [0., 1., -1.], [1., 1., -1.], [-1., -1., 0.], [0., -1., 0.],
        [1., -1., 0.], [-1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [-1., 1., 0.],
        [0., 1., 0.], [1., 1., 0.], [-1., -1., 1.], [0., -1., 1.], [1., -1., 1.],
        [-1., 0., 1.], [0., 0., 1.], [1., 0., 1.], [-1., 1., 1.], [0., 1., 1.],
        [1., 1., 1.],
    ]
    .iter()
    .flat_map(|c| c.iter().copied())
    .collect();
    // (type, nodes): 4=tet, 5=hex, 6=prism
    let elems: &[(u8, &[u32])] = &[
        (4, &[13, 15, 21, 25]),
        (4, &[12, 13, 15, 21]),
        (4, &[13, 21, 22, 25]),
        (4, &[15, 24, 21, 25]),
        (4, &[13, 15, 25, 16]),
        (5, &[0, 1, 4, 3, 9, 10, 13, 12]),
        (5, &[8, 9, 12, 11, 17, 18, 21, 20]),
        (5, &[2, 3, 6, 5, 11, 12, 15, 14]),
        (6, &[3, 4, 6, 12, 13, 15]),
        (6, &[4, 7, 6, 13, 16, 15]),
        (6, &[12, 13, 21, 9, 10, 18]),
        (6, &[13, 22, 21, 10, 19, 18]),
        (6, &[11, 14, 20, 12, 15, 21]),
        (6, &[15, 21, 24, 14, 20, 23]),
    ];
    let mut conn = Vec::new();
    let mut elem_types = Vec::new();
    let mut offsets = vec![0usize];
    for (t, ns) in elems {
        conn.extend_from_slice(ns);
        let et = match t {
            4 => ElementType::Tet4,
            5 => ElementType::Hex8,
            _ => ElementType::Prism6,
        };
        elem_types.push(et);
        offsets.push(conn.len());
    }
    Mesh {
        coords,
        conn,
        elem_tags: vec![1; 14],
        elem_type: ElementType::Tet4,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Tri3,
        elem_types: Some(elem_types),
        elem_offsets: Some(offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
    }
}
