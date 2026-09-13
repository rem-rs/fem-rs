//! Curved (high-order) geometry support for uniform Hex8 refinement.
//!
//! MFEM refines a curved mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement3D_base` + `Mesh::UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh;
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions
//!    (`Mesh::SetVerticesFromNodes`): a fine vertex sitting at a coarse Qk
//!    dof point receives that dof's value *exactly* (the refinement operator
//!    evaluates the coarse nodal basis at the child reference corners, which
//!    coincide with coarse dof points, so all basis weights are 0/1).
//!
//! Skipping step 2 makes every new vertex a straight average of the corner
//! coordinates, which diverges from MFEM on any non-affine mesh (e.g. the
//! curved cylinder `multidomain-hex.mesh`: only 96/480 refined children
//! matched MFEM's corner sets).
//!
//! This module implements the equivalent of step 2 for order-2 (Q2) hex
//! geometry:
//! - [`HexQ2Geometry`] reads the parent element's 27 geometry dofs and
//!   provides the exact pick of the mid-edge / face-center / body-center dof
//!   coordinates for the refinement kernels;
//! - [`build_refined_hex_geometry`] constructs the refined mesh's own
//!   [`GeometryData`] (fine vertices are the fine mesh nodes; fine
//!   edge/face/interior dofs are interpolated from the parent Q2 field at
//!   the corresponding reference points), so repeated refinements and
//!   downstream curved assembly keep the high-order geometry.

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};

use crate::simplex::{GeometryData, Mesh};

/// Per-element *positional* layout of the 27 geometry dofs of a Q2 hex, as
/// stored in [`GeometryData::conn`].
///
/// This must mirror `DofManager::build_q2_hex` (crates/space), which mirrors
/// the `HexQk` reference basis; `build_h1_geometry` (crates/io) writes the
/// MFEM `nodes` grid-function values into exactly these positions.
///
/// Positions 0..8 are the element vertices (element connectivity order),
/// 8..20 the mid-edge dofs in [`GEO_EDGES`] order, 20..26 the face-center
/// dofs in [`GEO_FACES`] order and 26 the body-center dof.
const GEO_EDGES: [(usize, usize); 12] = [
    (1, 5), (2, 6), (3, 7), (0, 4), (0, 3), (1, 2),
    (5, 6), (4, 7), (0, 1), (3, 2), (7, 6), (4, 5),
];
const GEO_FACES: [[usize; 4]; 6] = [
    [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4],
    [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7],
];
const HEX_Q2_DPE: usize = 27;

/// Reference coordinates ([0,1]³) of the 8 vertices of a Hex8 in **MFEM
/// `Geometry::Constants<CUBE>` order**: `0=(0,0,0) 1=(1,0,0) 2=(1,1,0)
/// 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)`.
///
/// This is the order `HexQ2::dof_coords` (crates/element) and
/// `local_faces_hex` used as "bottom = 0,1,2,3 at z=0". It is **not** the
/// bitwise `(v&1, (v>>1)&1, (v>>2)&1)` encoding: that one silently exchanges
/// vertices 2↔3 and 6↔7 (an x-mirror of the y=1 half). Getting this table
/// wrong scrambles the reference points of dofs 8..26 in `q2_eval` *and* the
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

/// Geometry-position index (8..20) of the mid-edge dof of the local hex edge
/// `(a, b)` (vertex-index pair, order-insensitive).
fn geo_edge_pos(a: usize, b: usize) -> usize {
    GEO_EDGES
        .iter()
        .position(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        .expect("local hex edge must exist in Q2 geometry layout")
}

/// Geometry-position index (20..26) of the face-center dof of the local hex
/// face whose corner vertex indices are `q` (order-insensitive).
fn geo_face_pos(q: &[usize; 4]) -> usize {
    let mut s: Vec<usize> = q.to_vec();
    s.sort_unstable();
    GEO_FACES
        .iter()
        .position(|f| {
            let mut t: Vec<usize> = f.to_vec();
            t.sort_unstable();
            t == s
        })
        .expect("local hex face must exist in Q2 geometry layout")
}

/// Reference coordinates ([0,1]³) of geometry dof position `p`.
///
/// **D111**: the vertex table is MFEM's `Geometry::Constants<CUBE>` order —
/// `0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1)
/// 7=(0,1,1)` — as spelled out by `HexQ2::dof_coords` (crates/element) and
/// `local_faces_hex` (bottom = 0,1,2,3 at z=0). It is *not* the bitwise
/// `(v&1, (v>>1)&1, (v>>2)&1)` encoding: that one silently exchanges
/// vertices 2↔3 and 6↔7 (an x-mirror of the y=1 half), which mis-assigns
/// the reference points of edge/face/body dofs 8..26 and makes `q2_eval`
/// evaluate a *permuted* Q2 field — the refined child geometry then no
/// longer reproduces the parent's (even affine!) geometry and its Jacobian
/// can fold.
fn geo_pos_ref(p: usize) -> [f64; 3] {
    let vert_ref = |v: usize| MFEM_HEX_VERTS[v];
    match p {
        0..8 => vert_ref(p),
        8..20 => {
            let (a, b) = GEO_EDGES[p - 8];
            let (ra, rb) = (vert_ref(a), vert_ref(b));
            [
                0.5 * (ra[0] + rb[0]),
                0.5 * (ra[1] + rb[1]),
                0.5 * (ra[2] + rb[2]),
            ]
        }
        20..26 => {
            let mut c = [0.0_f64; 3];
            for &v in &GEO_FACES[p - 20] {
                let r = vert_ref(v);
                for (k, ck) in c.iter_mut().enumerate() {
                    *ck += r[k];
                }
            }
            [c[0] / 4.0, c[1] / 4.0, c[2] / 4.0]
        }
        _ => [0.5, 0.5, 0.5],
    }
}

/// 1-D Q2 Lagrange basis: node `j` (at t = 0, 0.5, 1) evaluated at `t`.
/// At the nodes the weights are exactly 0/1, so evaluations landing on dof
/// points are exact picks.
fn q2_basis(j: usize, t: f64) -> f64 {
    match j {
        0 => (t - 0.5) * (t - 1.0) / 0.5,
        1 => t * (1.0 - t) * 4.0,
        _ => t * (t - 0.5) / 0.5,
    }
}

/// Evaluate the parent element's Q2 geometry field (the element's 27 dof
/// coordinates in positional layout) at reference point `xi` ∈ [0,1]³.
fn q2_eval(parent_xyz: &[[f64; 3]; HEX_Q2_DPE], xi: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0_f64; 3];
    for (p, x) in parent_xyz.iter().enumerate() {
        let r = geo_pos_ref(p);
        let mut w = 1.0_f64;
        for (k, xik) in xi.iter().enumerate() {
            let node = (r[k] * 2.0).round() as usize; // r[k] ∈ {0, 0.5, 1}
            w *= q2_basis(node, *xik);
        }
        for (ok, ok_val) in out.iter_mut().enumerate() {
            *ok_val += w * x[ok];
        }
    }
    out
}

/// Read-only view of an order-2 hex [`GeometryData`] attached to a mesh.
pub(crate) struct HexQ2Geometry<'a> {
    mesh: &'a Mesh<3>,
    geo: &'a GeometryData,
}

impl<'a> HexQ2Geometry<'a> {
    /// Returns `Some` iff `mesh` is a Hex8 mesh carrying order-2 geometry
    /// with 27 dofs per element (as built by the MFEM reader for curved
    /// `nodes` meshes).
    pub(crate) fn new(mesh: &'a Mesh<3>) -> Option<Self> {
        let geo = mesh.geometry.as_ref()?;
        if geo.order != 2 || geo.nodes_per_elem != HEX_Q2_DPE {
            return None;
        }
        if geo.conn.len() < mesh.n_elems() * HEX_Q2_DPE {
            return None;
        }
        Some(HexQ2Geometry { mesh, geo })
    }

    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * HEX_Q2_DPE;
        &self.geo.conn[o..o + HEX_Q2_DPE]
    }

    fn dof_xyz(&self, dof: NodeId) -> [f64; 3] {
        let o = dof as usize * 3;
        [self.geo.coords[o], self.geo.coords[o + 1], self.geo.coords[o + 2]]
    }

    /// Exact coordinate of the fine vertex at the midpoint of local edge
    /// `li` (index into `local_edges_hex`): the parent's mid-edge geometry
    /// dof value (MFEM `SetVerticesFromNodes` pick).
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 3] {
        let (a, b) = crate::amr::amr_inner::local_edges_hex()[li];
        let pos = 8 + geo_edge_pos(a, b);
        self.dof_xyz(self.elem_dofs(e)[pos])
    }

    /// Exact coordinate of the fine vertex at the center of local face `fi`
    /// (index into `local_faces_hex`): the parent's face geometry dof value.
    pub(crate) fn face_pick(&self, e: ElemId, fi: usize) -> [f64; 3] {
        let f = crate::amr::amr_inner::local_faces_hex()[fi];
        let pos = 20 + geo_face_pos(&f);
        self.dof_xyz(self.elem_dofs(e)[pos])
    }

    /// Exact coordinate of the face-center dof of the face of element `e`
    /// whose four corners are `fns` (any order). Returns `None` if `fns` is
    /// not a face of `e`.
    pub(crate) fn face_pick_by_nodes(&self, e: ElemId, fns: [NodeId; 4]) -> Option<[f64; 3]> {
        let ns = self.mesh.elem_nodes(e);
        let mut want = fns;
        want.sort_unstable();
        for f in crate::amr::amr_inner::local_faces_hex().iter() {
            let q = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
            let mut key = q;
            key.sort_unstable();
            if key == want {
                let pos = 20 + geo_face_pos(f);
                return Some(self.dof_xyz(self.elem_dofs(e)[pos]));
            }
        }
        None
    }

    /// Exact coordinate of the body-center geometry dof of element `e`.
    pub(crate) fn body_pick(&self, e: ElemId) -> [f64; 3] {
        self.dof_xyz(self.elem_dofs(e)[26])
    }

    fn parent_field(&self, e: ElemId) -> [[f64; 3]; HEX_Q2_DPE] {
        let dofs = self.elem_dofs(e);
        std::array::from_fn(|p| self.dof_xyz(dofs[p]))
    }
}

/// One entry per *fine* element: `(parent element, child index)`. Child
/// index `IDENTITY` marks an unrefined element copied through unchanged
/// (its fine element *is* the parent element: reference positions map with
/// scale 1 instead of 1/2, and every dof evaluation lands exactly on a
/// parent dof point).
pub(crate) const IDENTITY: u8 = 8;

/// Build the refined mesh's [`GeometryData`] from the parent's order-2 hex
/// geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates,
/// where every new vertex already carries its MFEM pick coordinate).
/// `fine_parent` maps each fine element to its parent element and child
/// index (see [`IDENTITY`]).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_
/// nodes` are the fine mesh nodes (fine vertex coordinates), followed by the
/// edge/face/body dofs created on first touch, keyed by the fine node sets
/// so elements sharing an entity share the dof. Values are evaluated from
/// the parent Q2 field at the corresponding reference point — an exact pick
/// of the parent dof value whenever the point coincides with a parent dof
/// (all weights are exactly 0/1; in particular every do of an identity
/// element), and the standard Q2 interpolation otherwise (MFEM's
/// `RefinementOperator`).
pub(crate) fn build_refined_hex_geometry(
    parent: &Mesh<3>,
    fine: &Mesh<3>,
    fine_parent: &[(ElemId, u8)],
) -> Option<GeometryData> {
    let pq = HexQ2Geometry::new(parent)?;
    debug_assert_eq!(fine.n_elems(), fine_parent.len());

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<[NodeId; 2], NodeId> = HashMap::new();
    let mut face_dofs: HashMap<[NodeId; 4], NodeId> = HashMap::new();
    let mut vol_dofs: HashMap<ElemId, NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * HEX_Q2_DPE);
    for (fe, &(pe, child)) in fine_parent.iter().enumerate() {
        let ns = fine.elem_nodes(fe as ElemId);
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
        let field = pq.parent_field(pe);
        let to_parent = |fine_ref: [f64; 3]| {
            [
                origin[0] + scale * fine_ref[0],
                origin[1] + scale * fine_ref[1],
                origin[2] + scale * fine_ref[2],
            ]
        };

        // Vertices (positions 0..8): the fine mesh nodes themselves.
        for &n in ns.iter().take(8) {
            conn.push(n);
        }
        // Mid-edge dofs (positions 8..20).
        for k in 0..12 {
            let (a, b) = GEO_EDGES[k];
            let mut key = [ns[a], ns[b]];
            key.sort_unstable();
            let xi = to_parent(geo_pos_ref(8 + k));
            let dof = *edge_dofs.entry(key).or_insert_with(|| {
                let x = q2_eval(&field, xi);
                geo_coords.extend_from_slice(&x);
                let id = next_dof;
                next_dof += 1;
                id
            });
            conn.push(dof);
        }
        // Face-center dofs (positions 20..26).
        for k in 0..6 {
            let f = GEO_FACES[k];
            let mut key = [ns[f[0]], ns[f[1]], ns[f[2]], ns[f[3]]];
            key.sort_unstable();
            let xi = to_parent(geo_pos_ref(20 + k));
            let dof = *face_dofs.entry(key).or_insert_with(|| {
                let x = q2_eval(&field, xi);
                geo_coords.extend_from_slice(&x);
                let id = next_dof;
                next_dof += 1;
                id
            });
            conn.push(dof);
        }
        // Body-center dof (position 26): one per fine element.
        let xi = to_parent(geo_pos_ref(26));
        let dof = *vol_dofs.entry(fe as ElemId).or_insert_with(|| {
            let x = q2_eval(&field, xi);
            geo_coords.extend_from_slice(&x);
            let id = next_dof;
            next_dof += 1;
            id
        });
        conn.push(dof);
    }

    Some(GeometryData {
        order: 2,
        conn,
        nodes_per_elem: HEX_Q2_DPE,
        coords: geo_coords,
        n_nodes: next_dof as usize,
    })
}
