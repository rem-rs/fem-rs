//! Curved (high-order) geometry support for uniform Hex8 refinement.
//!
//! MFEM refines a curved mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement3D_base` + `Mesh::UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh;
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions
//!    (`Mesh::SetVerticesFromNodes`): a fine vertex sitting at a coarse Qk
//!    dof point receives that dof's value *exactly*.
//!
//! Step 2 is where the geometry actually comes from.  MFEM builds the fine
//! `nodes` through `GridFunction::Update` with the refinement operator
//! (`FiniteElementSpace::Update`): for every *fine* element it takes the
//! child's embedded sub-cell (`CoarseFineTransformations::embeddings` +
//! `point_matrices`, whose affine maps are `parent_ref = origin + 0.5·child_ref`
//! with `origin ∈ {0, 0.5}³` and *no* mirroring) and evaluates the coarse
//! element's own FE at the fine element's dof reference points mapped into the
//! parent frame (`FiniteElement::GetLocalInterpolation`:
//! `I(k, j) = shape_j(child_map(node_k))`).  So the fine geometry is a
//! **per-child evaluation of the parent's nodal field**, not a transport with
//! vertex snapping; `Mesh::SetVerticesFromNodes` only copies the vertex dofs
//! back into `vertices` afterwards.
//!
//! Skipping it makes every new vertex a straight average of the corner
//! coordinates, which diverges from MFEM on any non-affine mesh (e.g. the
//! curved cylinder `multidomain-hex.mesh`: only 96/480 refined children
//! matched MFEM's corner sets).
//!
//! This module implements that evaluation for hexahedral `nodes` geometry of
//! **any order** `p ≥ 2`:
//! - [`HexQkGeometry`] reads an element's `(p+1)³` geometry dofs and provides
//!   the parent-field evaluation used both by the refinement kernels (the
//!   coordinates of the new edge-midpoint / face-center / body-center
//!   vertices) and by [`build_refined_hex_geometry`];
//! - [`build_refined_hex_geometry`] constructs the refined mesh's own
//!   [`GeometryData`]: fine vertices are the fine mesh nodes; every other
//!   geometry dof is the parent field evaluated at its reference point, keyed
//!   by the fine entity (edge / face / element interior) so elements sharing
//!   an entity share the dof — exactly MFEM's first-touch-wins
//!   (`GridFunction::Update`'s `mark` array).
//!
//! The positional layout of the `(p+1)³` geometry dofs is **not** hard-coded:
//! it is `HexQk::new(p).dof_coords()` (crates/element), the same table
//! `DofManager::build_q2_hex` / `build_pk_hex` (crates/space) and `fem-io`'s
//! `nodes` reader use, so the element, the global numbering and this module
//! cannot drift apart.  For `p == 2` that table is `LEGACY_P2_SLOTS`, which
//! reproduces the previous hand-written Q2 layout
//! (`vertices → 12 edges → 6 faces → body`) slot for slot, so the whole
//! order-2 path is unchanged bit for bit.

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// Reference coordinates ([0,1]³) of the 8 vertices of a Hex8 in **MFEM
/// `Geometry::Constants<CUBE>` order**: `0=(0,0,0) 1=(1,0,0) 2=(1,1,0)
/// 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)`.
///
/// This is the order `HexQk::dof_coords` (crates/element) and
/// `local_faces_hex` used as "bottom = 0,1,2,3 at z=0". It is **not** the
/// bitwise `(v&1, (v>>1)&1, (v>>2)&1)` encoding: that one silently exchanges
/// vertices 2↔3 and 6↔7 (an x-mirror of the y=1 half). Getting this table
/// wrong scrambles the reference points of every non-vertex dof *and* the
/// child-octant origin below, so a refined order-2 hex mesh stops
/// reproducing its parent geometry (D111: `data/cube.mesh -o 2 -rs 1` had
/// min det(J) = −1.32 instead of +0.00195).
const MFEM_HEX_VERTS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Index-space corner of local vertex `v` — each component is `0` or `p`
/// (`p` being the geometry order, i.e. the last tensor index).
fn vert_index(v: usize, p: usize) -> [usize; 3] {
    [
        usize::from(MFEM_HEX_VERTS[v][0] > 0.5) * p,
        usize::from(MFEM_HEX_VERTS[v][1] > 0.5) * p,
        usize::from(MFEM_HEX_VERTS[v][2] > 0.5) * p,
    ]
}

/// The 1-D Lagrange basis of the ascending nodes `nodes` evaluated at `x`
/// (`L_i(x) = Π_{j≠i} (x − x_j)/(x_i − x_j)`).
///
/// The product form is what makes the interpolation *exact at the nodes*:
/// `L_i(x_i)` is a product of exactly-1 ratios and `L_i(x_j)`, `j ≠ i`, has
/// an exactly-zero factor.  A point that coincides with a geometry dof is
/// therefore read back bit for bit, matching MFEM's `SetVerticesFromNodes`
/// picks.
fn lagrange(nodes: &[f64], i: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (j, &xj) in nodes.iter().enumerate() {
        if j != i {
            v *= (x - xj) / (nodes[i] - xj);
        }
    }
    v
}

/// The `(axis, positive?)` of the step `from → to` between two index-space
/// corners that differ in exactly one component.
fn axis_step(from: [usize; 3], to: [usize; 3]) -> (usize, bool) {
    let d = (0..3).find(|&d| from[d] != to[d]).expect("corners must differ in one axis");
    (d, to[d] > from[d])
}

/// Read-only view of an order-`p` (`p ≥ 2`) hex [`GeometryData`] attached to
/// a mesh.
pub(crate) struct HexQkGeometry<'a> {
    mesh: &'a Mesh<3>,
    geo: &'a GeometryData,
    /// Geometry order `p`.
    order: usize,
    /// Geometry dofs per element, `(p+1)³` (`GeometryData::nodes_per_elem`).
    dpe: usize,
    /// The `p+1` one-dimensional dof nodes on `[0,1]`, ascending.
    nodes: Vec<f64>,
    /// Tensor index `(ix, iy, iz)` of each local dof position, in the
    /// geometry FE's own local order (`HexQk::dof_coords`).
    idx: Vec<[usize; 3]>,
    /// Reference coordinates ([0,1]³) of each local dof position.
    ref01: Vec<[f64; 3]>,
}

impl<'a> HexQkGeometry<'a> {
    /// Returns `Some` iff `mesh` is a Hex8 mesh carrying order-`p ≥ 2`
    /// geometry with `(p+1)³` dofs per element (as built by the MFEM reader
    /// for curved `nodes` meshes).
    pub(crate) fn new(mesh: &'a Mesh<3>) -> Option<Self> {
        let geo = mesh.geometry.as_ref()?;
        let order = geo.order as usize;
        if order < 2 {
            // Order 1 (or absent) geometry: the straight averaging the
            // refinement kernels already do *is* the interpolation.
            return None;
        }
        let dpe = (order + 1).pow(3);
        if geo.nodes_per_elem != dpe {
            return None;
        }
        if geo.conn.len() < mesh.n_elems() * dpe {
            return None;
        }

        // The single source of truth for the positional layout: the element's
        // own dof table (`HexQk` on `[-1,1]³`), mapped onto the `[0,1]³`
        // reference cube the mesh kernels use.
        let fe = ElementType::Hex8.ref_elem(order as u8);
        let dof_coords = fe.dof_coords();
        if dof_coords.len() != dpe {
            return None;
        }
        let ref01: Vec<[f64; 3]> = dof_coords
            .iter()
            .map(|c| [0.5 * (c[0] + 1.0), 0.5 * (c[1] + 1.0), 0.5 * (c[2] + 1.0)])
            .collect();
        let mut nodes: Vec<f64> = ref01.iter().map(|r| r[0]).collect();
        nodes.sort_by(|a, b| a.partial_cmp(b).expect("finite dof nodes"));
        nodes.dedup();
        if nodes.len() != order + 1 {
            return None;
        }
        let mut idx = Vec::with_capacity(dpe);
        for r in &ref01 {
            let mut t = [0usize; 3];
            for (d, td) in t.iter_mut().enumerate() {
                *td = nodes.iter().position(|&n| n == r[d]).expect("dof node must exist");
            }
            idx.push(t);
        }

        Some(HexQkGeometry { mesh, geo, order, dpe, nodes, idx, ref01 })
    }

    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * self.dpe;
        &self.geo.conn[o..o + self.dpe]
    }

    fn dof_xyz(&self, dof: NodeId) -> [f64; 3] {
        let o = dof as usize * 3;
        [self.geo.coords[o], self.geo.coords[o + 1], self.geo.coords[o + 2]]
    }

    /// Evaluate the parent element's order-`p` geometry field at the `[0,1]³`
    /// reference point `xi` — the tensor Lagrange interpolation of the
    /// element's `(p+1)³` geometry dofs, accumulated in local dof order.
    fn eval_at(&self, e: ElemId, xi: [f64; 3]) -> [f64; 3] {
        let p = self.order;
        let bx: Vec<f64> = (0..=p).map(|i| lagrange(&self.nodes, i, xi[0])).collect();
        let by: Vec<f64> = (0..=p).map(|i| lagrange(&self.nodes, i, xi[1])).collect();
        let bz: Vec<f64> = (0..=p).map(|i| lagrange(&self.nodes, i, xi[2])).collect();
        let dofs = self.elem_dofs(e);
        let mut out = [0.0_f64; 3];
        for (o, t) in self.idx.iter().enumerate() {
            let w = bx[t[0]] * by[t[1]] * bz[t[2]];
            let x = self.dof_xyz(dofs[o]);
            for (k, ok) in out.iter_mut().enumerate() {
                *ok += w * x[k];
            }
        }
        out
    }

    /// Centroid ([0,1]³) of the four corners of a local face.
    fn centroid(&self, f: &[usize; 4]) -> [f64; 3] {
        let mut c = [0.0_f64; 3];
        for &v in f {
            let r = MFEM_HEX_VERTS[v];
            for (k, ck) in c.iter_mut().enumerate() {
                *ck += r[k];
            }
        }
        [c[0] / 4.0, c[1] / 4.0, c[2] / 4.0]
    }

    /// Coordinate of the new fine vertex at the midpoint of local edge `li`
    /// (index into `local_edges_hex`): the parent geometry field evaluated at
    /// the edge midpoint.  For an order-2 parent that point *is* the parent's
    /// mid-edge geometry dof and the evaluation reads it back exactly (all
    /// basis weights are 0/1) — MFEM's `SetVerticesFromNodes` pick.
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 3] {
        let (a, b) = crate::amr::amr_inner::local_edges_hex()[li];
        let (ra, rb) = (MFEM_HEX_VERTS[a], MFEM_HEX_VERTS[b]);
        self.eval_at(
            e,
            [0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1]), 0.5 * (ra[2] + rb[2])],
        )
    }

    /// Coordinate of the new fine vertex at the center of local face `fi`
    /// (index into `local_faces_hex`): the parent field at the face centroid.
    pub(crate) fn face_pick(&self, e: ElemId, fi: usize) -> [f64; 3] {
        let f = crate::amr::amr_inner::local_faces_hex()[fi];
        self.eval_at(e, self.centroid(&f))
    }

    /// Coordinate of the face-center vertex of the face of element `e` whose
    /// four corners are `fns` (any order). Returns `None` if `fns` is not a
    /// face of `e`.
    pub(crate) fn face_pick_by_nodes(&self, e: ElemId, fns: [NodeId; 4]) -> Option<[f64; 3]> {
        let ns = self.mesh.elem_nodes(e);
        let mut want = fns;
        want.sort_unstable();
        for f in crate::amr::amr_inner::local_faces_hex().iter() {
            let mut key = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
            key.sort_unstable();
            if key == want {
                return Some(self.eval_at(e, self.centroid(f)));
            }
        }
        None
    }

    /// Coordinate of the body-center vertex of element `e`: the parent field
    /// at the reference cube center.
    pub(crate) fn body_pick(&self, e: ElemId) -> [f64; 3] {
        self.eval_at(e, [0.5, 0.5, 0.5])
    }
}

/// One entry per *fine* element: `(parent element, child index)`. Child
/// index `IDENTITY` marks an unrefined element copied through unchanged
/// (its fine element *is* the parent element: reference positions map with
/// scale 1 instead of 1/2, and every evaluation lands exactly on a parent
/// dof point).
pub(crate) const IDENTITY: u8 = 8;

/// Key of a shared fine edge dof: the fine edge (its sorted node pair) plus
/// the dof's index within the edge, measured from its lower-id node.
type EdgeKey = (NodeId, NodeId, usize);
/// Key of a shared fine face dof: the fine face (its sorted node quad) plus
/// the dof's two in-face indices in the face's canonical frame (`j1` away
/// from the minimum-id corner towards its smaller-id neighbour, `j2` towards
/// the other one).
type FaceKey = ([NodeId; 4], usize, usize);

/// Build the refined mesh's [`GeometryData`] from the parent's order-`p` hex
/// geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates,
/// where every new vertex already carries the parent-field coordinate
/// computed by the refinement kernels).  `fine_parent` maps each fine element
/// to its parent element and child index (see [`IDENTITY`]).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_
/// nodes` are the fine mesh nodes (fine vertex coordinates), followed by the
/// edge/face/interior dofs created on first touch, keyed by the fine node
/// sets so elements sharing an entity share the dof.  Values are the parent
/// order-`p` field evaluated at the corresponding reference point — read back
/// exactly wherever the point coincides with a parent dof (all weights are
/// exactly 0/1; in particular every dof of an identity element) and the
/// standard order-`p` interpolation otherwise, i.e. MFEM's
/// `GetLocalInterpolation` with its first-touch `mark` rule.
pub(crate) fn build_refined_hex_geometry(
    parent: &Mesh<3>,
    fine: &Mesh<3>,
    fine_parent: &[(ElemId, u8)],
) -> Option<GeometryData> {
    let pq = HexQkGeometry::new(parent)?;
    let p = pq.order;
    let dpe = pq.dpe;
    debug_assert_eq!(fine.n_elems(), fine_parent.len());

    // Index-space corner of each local vertex, used to classify a dof.
    let vix: Vec<[usize; 3]> = (0..8).map(|v| vert_index(v, p)).collect();
    let edges = crate::amr::amr_inner::local_edges_hex();
    let faces = crate::amr::amr_inner::local_faces_hex();

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<EdgeKey, NodeId> = HashMap::new();
    let mut face_dofs: HashMap<FaceKey, NodeId> = HashMap::new();
    let mut vol_dofs: HashMap<(ElemId, usize), NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * dpe);
    for (fe, &(pe, child)) in fine_parent.iter().enumerate() {
        let fe = fe as ElemId;
        let ns = fine.elem_nodes(fe);
        // Fine-ref → parent-ref affine map: parent = origin + scale * fine.
        // `child` is the *MFEM hex corner index* of the child's own corner
        // (see the child table in `refine_nonconforming_hex`), so its octant
        // origin is `0.5 * MFEM_HEX_VERTS[child]` (D111: the bitwise
        // `((child>>k)&1)` encoding picks the wrong octant for corners
        // 2,3,6,7 and corrupts every dof evaluated from that child).
        let (origin, scale) = if child == IDENTITY {
            ([0.0_f64; 3], 1.0_f64)
        } else {
            let v = MFEM_HEX_VERTS[child as usize];
            ([0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], 0.5)
        };

        for o in 0..dpe {
            let r = pq.ref01[o];
            let xi = [
                origin[0] + scale * r[0],
                origin[1] + scale * r[1],
                origin[2] + scale * r[2],
            ];
            let t = pq.idx[o];
            let on_bnd = [t[0] == 0 || t[0] == p, t[1] == 0 || t[1] == p, t[2] == 0 || t[2] == p];
            let n_bnd = on_bnd.iter().filter(|&&b| b).count();

            let id = if n_bnd == 3 {
                // Vertex: the fine mesh node itself.
                let v = (0..8).find(|&v| vix[v] == t).expect("hex vertex slot");
                ns[v]
            } else if n_bnd == 2 {
                // Edge dof: shared by every element meeting on that fine edge.
                let av = (0..3).find(|&d| !on_bnd[d]).expect("hex edge axis");
                let (a, b) = *edges
                    .iter()
                    .find(|&&(a, b)| {
                        let (va, vb) = (vix[a], vix[b]);
                        va[av] != vb[av]
                            && (0..3).filter(|&d| d != av).all(|d| va[d] == vb[d] && va[d] == t[d])
                    })
                    .expect("hex edge slot");
                // Measure the within-edge index from one of its fine nodes so
                // that both elements name the same dof; the 1-D node set is
                // symmetric, so index `m` from the low end is `p - m` from the
                // high end.
                let (lo, hi) = if vix[a][av] == 0 { (a, b) } else { (b, a) };
                let (n0, n1) = (ns[lo], ns[hi]);
                let key = if n0 <= n1 { (n0, n1, t[av]) } else { (n1, n0, p - t[av]) };
                *edge_dofs.entry(key).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                })
            } else if n_bnd == 1 {
                // Face dof: shared by every element meeting on that fine face.
                // The local face is found through its fixed axis; the two
                // in-face indices are then measured from the face's canonical
                // frame (its minimum-id corner, then the smaller-id of its two
                // in-face neighbours), which every element computes the same
                // way.
                let ax = (0..3).find(|&d| on_bnd[d]).expect("hex face axis");
                let f = *faces
                    .iter()
                    .find(|f| f.iter().all(|&v| vix[v][ax] == t[ax]))
                    .expect("hex face slot");
                let ids = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
                let k = (0..4).min_by_key(|&i| ids[i]).expect("non-empty face");
                let (nl, nr) = (f[(k + 3) % 4], f[(k + 1) % 4]);
                let (second, other) = if ns[nl] <= ns[nr] { (nl, nr) } else { (nr, nl) };
                let (d1, s1) = axis_step(vix[f[k]], vix[second]);
                let (d2, s2) = axis_step(vix[f[k]], vix[other]);
                let j1 = if s1 { t[d1] } else { p - t[d1] };
                let j2 = if s2 { t[d2] } else { p - t[d2] };
                let mut key4 = ids;
                key4.sort_unstable();
                *face_dofs.entry((key4, j1, j2)).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                })
            } else {
                // Interior dof: one per fine element and local position.
                *vol_dofs.entry((fe, o)).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                })
            };
            conn.push(id);
        }
    }

    Some(GeometryData {
        order: p as u8,
        conn,
        nodes_per_elem: dpe,
        coords: geo_coords,
        n_nodes: next_dof as usize,
    })
}
