//! Curved (high-order) geometry support for uniform Prism6 (wedge)
//! refinement — the wedge analogue of [`super::curved_hex`] and
//! [`super::curved_quad`].
//!
//! MFEM refines a curved 3-D mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement3D_base` + `UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh —
//!    for a wedge this means the 9 edge midpoints and the 3 **quadrilateral**
//!    face centers (MFEM's 8-child wedge split uses no triangular face centers
//!    and no body center, so a pure-wedge mesh gains exactly 12 vertices per
//!    element);
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions: for every *fine* element
//!    the child's embedding (`CoarseFineTransformations::point_matrices`
//!    `pri_children`, an affine map `parent_ref = origin + M·child_ref` with
//!    no mirroring) is applied to the child's dof reference points and the
//!    coarse element's own FE is evaluated there, so the fine geometry is the
//!    **per-child evaluation of the parent's nodal field**, with dofs shared
//!    between elements meeting on the same fine entity (MFEM's first-touch
//!    `mark` rule).
//!
//! The positional layout of the `(p+1)²(p+2)/2` geometry dofs is the element's
//! own table ([`PrismPk`] — the layer-major Gauss-Lobatto lattice round 34's
//! D164 pinned), the same table `fem_io`'s `nodes` writer pairs the mesh table
//! with, so element, numbering and this module cannot drift apart.  Note the
//! wedge reference conventions differ: `PrismPk` stacks the **extrusion
//! first** (`(ξ, η, ζ)` = (layer, triangle x, triangle y)) while MFEM's
//! `PRISM` reference wedge is `(x, y, z)` with `z` the extrusion —
//! [`MFEM_PRI_CHILDREN`] is stored in MFEM's convention and permuted on use.
//!
//! MFEM's child order for the 8 wedge children (the order they enter
//! `new_elements`, and hence the fine mesh's element order) is *corner 0,
//! center, corner 1, corner 2* per layer — unlike the historical fem-rs
//! straight-side order (corners first, center last), which stays untouched for
//! straight-sided meshes; the MFEM order and MFEM's canonical new-vertex ids
//! (`MfemPrismRefineIds`, called from `amr_inner`) are used only where a
//! written file pins them down: geometry present **and** uniform refinement.

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};
use fem_element::lagrange::PrismPk;
use fem_element::ReferenceElement;

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// The six wedge vertices in `PrismPk`'s reference convention
/// `(ξ, η, ζ)` = (extrusion, triangle x, triangle y): vertices 0-2 form the
/// `ξ = 0` triangle `(0,0) (1,0) (0,1)`, vertices 3-5 the `ξ = 1` one.
const VERT_PK: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
];

/// Local edges of a wedge as vertex pairs, in MFEM
/// `Geometry::Constants<PRISM>::Edges` order: bottom triangle, top triangle,
/// verticals.  (`amr_inner::local_edges_prism` enumerates the same undirected
/// edges in the same order.)
const EDGES: [(usize, usize); 9] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
];

/// The triangle edge each quadrilateral face spans, as `(from, to)` vertex
/// pairs: face `f` (`amr_inner::local_faces_prism_quad()[f]`) lies over
/// triangle edge `f`.
const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// MFEM `UniformRefinement3D_base`'s `pri_children` point matrices
/// (`mesh/mesh.cpp`, `A = 0, B = 0.5, C = 1`): 8 children, each 6 reference
/// points in **MFEM `(x, y, z)`** coordinates — the images of the child's
/// reference wedge vertices in the parent's reference wedge.  Child order =
/// `CoarseFineTr` matrix index = fine element order: corner 0, center, corner
/// 1, corner 2 (lower layer), then corner 3, center, corner 4, corner 5
/// (upper layer).
const MFEM_PRI_CHILDREN: [[[f64; 3]; 6]; 8] = [
    // corner 0, lower half
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    // center, lower half (rotated frame: the child's extrusion axis runs
    // across the parent triangle)
    [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
    // corner 1, lower half
    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
    // corner 2, lower half
    [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 1.0, 0.5]],
    // corner 3 (above corner 0), upper half
    [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]],
    // center, upper half
    [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0]],
    // corner 4, upper half
    [[0.5, 0.0, 0.5], [1.0, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 1.0]],
    // corner 5, upper half
    [[0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0]],
];

/// Affine map `parent_ref = origin + M·child_ref` of child `child`, in
/// `PrismPk`'s `(ξ, η, ζ)` convention (MFEM stores `(x, y, z)` with the
/// extrusion last; `PrismPk` has it first, so the point matrices are permuted
/// `(x, y, z) → (z, x, y)` on use).
fn child_map(child: usize) -> ([f64; 3], [[f64; 3]; 3]) {
    let pts = &MFEM_PRI_CHILDREN[child];
    // MFEM (x, y, z) → pk (ξ, η, ζ) = (z, x, y).
    let pk = |p: [f64; 3]| [p[2], p[0], p[1]];
    let origin = pk(pts[0]);
    let mut m = [[0.0_f64; 3]; 3];
    // Image of the pk basis axes: ξ ← MFEM z axis (pts[3] − pts[0]),
    // η ← MFEM x axis (pts[1] − pts[0]), ζ ← MFEM y axis (pts[2] − pts[0]).
    for (k, o) in origin.iter().enumerate() {
        m[k][0] = pk(pts[3])[k] - o;
        m[k][1] = pk(pts[1])[k] - o;
        m[k][2] = pk(pts[2])[k] - o;
    }
    (origin, m)
}

/// Child index of an **unrefined** element copied through unchanged (its fine
/// element *is* the parent element: the reference map is the identity and
/// every evaluation lands exactly on a parent dof point).
pub(crate) const IDENTITY: u8 = 8;

/// Read-only view of an order-`p` (`p ≥ 2`) wedge [`GeometryData`] attached to
/// a Prism6 mesh.
///
/// Two slot orders exist in the wild and are told apart on construction:
/// `Mesh::set_curvature` builds the table in [`PrismPk`]'s **layer-major**
/// order, while `fem_io`'s reader (`DofManager::build_prism_h1`) builds it in
/// **MFEM's H1 entity order** (`H1PrismPk`: vertices → edges → triangular
/// faces → quadrilateral faces → interior).  The two are permutations of the
/// same lattice, so an H1-ordered table is converted to the layer-major view
/// once, via `H1PrismPk::layer_perm`, and everything downstream speaks
/// layer-major.
pub(crate) struct PrismPkGeometry<'a> {
    geo: &'a GeometryData,
    fe: PrismPk,
    /// Geometry order `p`.
    pub(crate) order: usize,
    /// Geometry dofs per element, `(p+1)²(p+2)/2` (`GeometryData::nodes_per_elem`).
    pub(crate) dpe: usize,
    /// Layer-major view of the connectivity: `None` when the table is already
    /// layer-major (`set_curvature`), otherwise the permuted copy of an
    /// H1-ordered (`reader`) table.
    layer_conn: Option<Vec<NodeId>>,
}

impl<'a> PrismPkGeometry<'a> {
    /// Returns `Some` iff `mesh` is a Prism6 mesh carrying order-`p ≥ 2`
    /// geometry with `(p+1)²(p+2)/2` dofs per element (as built by the MFEM
    /// reader / `set_curvature` for curved `nodes` meshes).
    pub(crate) fn new(mesh: &'a Mesh<3>) -> Option<Self> {
        if mesh.elem_type != ElementType::Prism6 {
            return None;
        }
        let geo = mesh.geometry.as_ref()?;
        let order = geo.order as usize;
        if order < 2 {
            // Order 1 (or absent) geometry: the straight averaging the
            // refinement kernels already do *is* the interpolation.
            return None;
        }
        let dpe = (order + 1) * (order + 1) * (order + 2) / 2;
        if geo.nodes_per_elem != dpe || geo.conn.len() < mesh.n_elems() * dpe {
            return None;
        }
        let fe = PrismPk::new(order);
        if fe.n_dofs() != dpe {
            return None;
        }
        // Slot-order detection: an H1-entity-ordered table starts every
        // element's row with the element's six *vertex* ids, while a
        // layer-major table's slots 3..6 hold bottom-face edge/interior dofs
        // whose ids never collide with vertex ids (both numberings place the
        // vertices first).  Vertex ids and dof ids are disjoint in both.
        let h1_ordered = (0..mesh.n_elems() as ElemId).all(|e| {
            let ns = mesh.elem_nodes(e);
            let o = e as usize * dpe;
            (0..6).all(|v| geo.conn[o + v] == ns[v])
        });
        let layer_conn = if h1_ordered {
            // H1 slot `m` holds the layer slot `perm[m]`'s dof: invert into a
            // layer-major row per element.
            let perm = fem_element::lagrange::H1PrismPk::new(order).layer_perm().to_vec();
            debug_assert_eq!(perm.len(), dpe);
            let mut out = geo.conn.clone();
            for e in 0..mesh.n_elems() as usize {
                let o = e * dpe;
                for (m, &s) in perm.iter().enumerate() {
                    out[o + s] = geo.conn[o + m];
                }
            }
            Some(out)
        } else {
            None
        };
        Some(PrismPkGeometry { geo, fe, order, dpe, layer_conn })
    }

    /// The layer-major slot dofs of element `e`.
    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * self.dpe;
        match &self.layer_conn {
            Some(c) => &c[o..o + self.dpe],
            None => &self.geo.conn[o..o + self.dpe],
        }
    }

    /// Evaluate the parent element's order-`p` geometry field at the `PrismPk`
    /// reference point `xi` — the interpolation of the element's geometry dofs
    /// with the element's own nodal basis (exact δ at the dof points, so every
    /// evaluation at a parent dof reads the stored value back bit for bit).
    pub(crate) fn eval_at(&self, e: ElemId, xi: [f64; 3]) -> [f64; 3] {
        let mut w = vec![0.0_f64; self.dpe];
        self.fe.eval_basis(&xi, &mut w);
        let dofs = self.elem_dofs(e);
        let mut out = [0.0_f64; 3];
        for (s, &w) in w.iter().enumerate() {
            if w != 0.0 {
                let d = dofs[s] as usize;
                out[0] += w * self.geo.coords[3 * d];
                out[1] += w * self.geo.coords[3 * d + 1];
                out[2] += w * self.geo.coords[3 * d + 2];
            }
        }
        out
    }

    /// Coordinate of the new fine vertex at the midpoint of local edge `li`
    /// (index into [`EDGES`]): the parent geometry field evaluated at the edge
    /// midpoint.
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 3] {
        let (a, b) = EDGES[li];
        let (ra, rb) = (VERT_PK[a], VERT_PK[b]);
        self.eval_at(e, [0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1]), 0.5 * (ra[2] + rb[2])])
    }

    /// Coordinate of the new fine vertex at the center of quadrilateral face
    /// `fi` (index into `amr_inner::local_faces_prism_quad()`): the parent
    /// field at the face centroid (the average of its four corners'
    /// reference coordinates — the position MFEM's `AverageVertices(vv, 4, …)`
    /// new vertex is refined from).
    pub(crate) fn quad_face_pick(&self, e: ElemId, fi: usize) -> [f64; 3] {
        let face = crate::amr::amr_inner::local_faces_prism_quad()[fi];
        let mut c = [0.0_f64; 3];
        for &v in &face {
            let r = VERT_PK[v];
            for (k, ck) in c.iter_mut().enumerate() {
                *ck += r[k];
            }
        }
        self.eval_at(e, [c[0] / 4.0, c[1] / 4.0, c[2] / 4.0])
    }

    /// Coordinate of the centroid of triangular face `layer` (0 = bottom,
    /// 1 = top) of wedge `e`: the parent field at the average of the three
    /// corners' reference coordinates.
    pub(crate) fn tri_face_pick(&self, e: ElemId, layer: usize) -> [f64; 3] {
        let mut c = [0.0_f64; 3];
        for &v in &VERT_PK[3 * layer..3 * layer + 3] {
            for (k, ck) in c.iter_mut().enumerate() {
                *ck += v[k];
            }
        }
        self.eval_at(e, [c[0] / 3.0, c[1] / 3.0, c[2] / 3.0])
    }

    /// Coordinate of the centroid of wedge `e` (unused by the wedge split but
    /// kept for callers that allocate a body vertex): the parent field at the
    /// average of the six corners' reference coordinates.
    pub(crate) fn body_pick(&self, e: ElemId) -> [f64; 3] {
        let mut c = [0.0_f64; 3];
        for r in VERT_PK {
            for (k, ck) in c.iter_mut().enumerate() {
                *ck += r[k];
            }
        }
        self.eval_at(e, [c[0] / 6.0, c[1] / 6.0, c[2] / 6.0])
    }
}

/// Classification of a triangle dof of the [`PrismPk`] lattice — MFEM
/// `H1_TriangleElement`'s node layout (`fem/fe/fe_h1.cpp`): corners, then the
/// three edges (counted from their first vertex), then the interior at the
/// `w`-normalised GLL points `(cp[i]/w, cp[j]/w)`, `j` outer / `i` inner.
#[derive(Clone, Copy, Debug)]
enum TriSlot {
    Corner(usize),
    /// Edge `0..3` (`(0,1) (1,2) (2,0)`), position `m` (1..p) from the edge's
    /// first vertex.
    Edge(usize, usize),
    /// Interior at GLL tensor indices `(i, j)`, `i, j ≥ 1`, `i + j ≤ p − 1`.
    Interior(usize, usize),
}

/// The triangle slot of every [`PrismPk`] layer dof `t` (order `p`).
fn tri_slots(p: usize) -> Vec<TriSlot> {
    let mut out = Vec::with_capacity((p + 1) * (p + 2) / 2);
    out.push(TriSlot::Corner(0));
    out.push(TriSlot::Corner(1));
    out.push(TriSlot::Corner(2));
    for m in 1..p {
        out.push(TriSlot::Edge(0, m));
    }
    for m in 1..p {
        out.push(TriSlot::Edge(1, m));
    }
    for m in 1..p {
        out.push(TriSlot::Edge(2, m));
    }
    for j in 1..p {
        for i in 1..(p - j) {
            out.push(TriSlot::Interior(i, j));
        }
    }
    out
}

/// Where a slot of the [`PrismPk`] layer-major table sits on the wedge.  Slot
/// `s` is layer `k = s / n_tri` (ξ = the `k`-th Gauss-Lobatto point) holding
/// triangle dof `t = s % n_tri`.
#[derive(Clone, Copy, Debug)]
enum SlotKind {
    /// Wedge vertex (local index 0..6).
    Vertex(usize),
    /// Wedge edge (`EDGES` index) at the within-edge Gauss-Lobatto position
    /// `m` (1..p), counted from `EDGES[e].0`.
    Edge { e: usize, m: usize },
    /// Triangular face dof: face `layer` (0 = bottom, 1 = top), triangle dof
    /// `t` (an interior triangle dof).
    TriFace { layer: usize, t: usize },
    /// Quadrilateral face dof: face `f` over triangle edge `f`, `a` the
    /// within-edge Gauss-Lobatto position from `TRI_EDGES[f].0`, `b` the
    /// layer index (both 1..p).
    QuadFace { f: usize, a: usize, b: usize },
    /// Element interior.
    Interior,
}

/// The [`SlotKind`] of the layer-`k` slot holding triangle dof `t` of an
/// order-`p` wedge.
fn slot_kind(p: usize, k: usize, t: usize, tri: &[TriSlot]) -> SlotKind {
    match tri[t] {
        TriSlot::Corner(c) => {
            if k == 0 {
                SlotKind::Vertex(c)
            } else if k == p {
                SlotKind::Vertex(c + 3)
            } else {
                // Vertical edge `6 + c` runs from bottom vertex `c` upward,
                // so the layer index *is* the within-edge position.
                SlotKind::Edge { e: 6 + c, m: k }
            }
        }
        TriSlot::Edge(e, m) => {
            if k == 0 {
                SlotKind::Edge { e, m }
            } else if k == p {
                // Top edge block: same orientation as the bottom edge
                // (`EDGES[3..6]` start at the top vertex above `EDGES[e].0`).
                SlotKind::Edge { e: e + 3, m }
            } else {
                SlotKind::QuadFace { f: e, a: m, b: k }
            }
        }
        TriSlot::Interior(..) => {
            if k == 0 {
                SlotKind::TriFace { layer: 0, t }
            } else if k == p {
                SlotKind::TriFace { layer: 1, t }
            } else {
                SlotKind::Interior
            }
        }
    }
}

/// Key of a shared fine edge dof: the fine edge (its sorted node pair) plus
/// the dof's index within the edge, measured from its lower-id node.
type EdgeKey = (NodeId, NodeId, usize);
/// Key of a shared fine triangular-face dof: the fine face (sorted corner
/// nodes) plus the dof's GLL index triple in the face's canonical frame
/// (canonical corner 0 = minimum-id corner, corner 1 = the smaller-id of the
/// remaining two).  `u[q]` is the dof's GLL index associated with canonical
/// corner `q` (`u₀ + u₁ + u₂ = p`).
type TriFaceKey = ([NodeId; 3], usize, usize, usize);
/// Key of a shared fine quadrilateral-face dof: the fine face (sorted corner
/// nodes) plus the dof's two in-face indices in the face's canonical frame
/// (minimum-id corner, first axis towards the smaller-id neighbour).
type QuadFaceKey = ([NodeId; 4], usize, usize);

/// Build the refined mesh's [`GeometryData`] from the parent's order-`p` wedge
/// geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates, where
/// every new vertex already carries the parent-field coordinate computed by
/// the refinement kernels).  `fine_parent` maps each fine element to its
/// parent element and child index — the index into MFEM's `pri_children`
/// tables *in fine element order* ([`IDENTITY`] for an unrefined element).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_nodes`
/// are the fine mesh nodes (fine vertex coordinates), followed by the
/// edge/face/interior dofs created on first touch, keyed by the fine node sets
/// so elements sharing an entity share the dof.  Values are the parent
/// order-`p` field evaluated at the corresponding reference point — read back
/// exactly wherever the point coincides with a parent dof (in particular every
/// dof of an identity element) and the standard order-`p` interpolation
/// otherwise, i.e. MFEM's `GetLocalInterpolation` with its first-touch `mark`
/// rule.
pub(crate) fn build_refined_prism_geometry(
    parent: &Mesh<3>,
    fine: &Mesh<3>,
    fine_parent: &[(ElemId, u8)],
) -> Option<GeometryData> {
    let pq = PrismPkGeometry::new(parent)?;
    let p = pq.order;
    let dpe = pq.dpe;
    debug_assert_eq!(fine.n_elems(), fine_parent.len());

    // Gauss-Lobatto points on [0,1] — the 1-D lattice the classification is
    // expressed in (same points `PrismPk` places its layers/edges on); only
    // the debug ground check below needs them.
    #[cfg(debug_assertions)]
    let (g, _w) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
    #[cfg(debug_assertions)]
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();

    // The child's dof positions in its own reference wedge (`PrismPk`
    // layer-major) and their classification.
    let fe_coords: Vec<[f64; 3]> =
        pq.fe.dof_coords().iter().map(|c| [c[0], c[1], c[2]]).collect();
    let n_tri = (p + 1) * (p + 2) / 2;
    let tri = tri_slots(p);
    debug_assert_eq!(tri.len(), n_tri);
    debug_assert_eq!(fe_coords.len(), dpe);
    let kinds: Vec<SlotKind> = (0..dpe)
        .map(|s| slot_kind(p, s / n_tri, s % n_tri, &tri))
        .collect();
    // Ground check (debug builds): the classification must place every slot at
    // the coordinates `PrismPk` itself reports, or the entity keys below would
    // be built on a misread lattice.
    #[cfg(debug_assertions)]
    for (s, kind) in kinds.iter().enumerate() {
        let r = &fe_coords[s];
        let want: [f64; 3] = match *kind {
            SlotKind::Vertex(v) => VERT_PK[v],
            SlotKind::Edge { e, m } => {
                let (a, b) = EDGES[e];
                let (ra, rb) = (VERT_PK[a], VERT_PK[b]);
                let t = cp[m];
                [
                    ra[0] + t * (rb[0] - ra[0]),
                    ra[1] + t * (rb[1] - ra[1]),
                    ra[2] + t * (rb[2] - ra[2]),
                ]
            }
            SlotKind::TriFace { layer, t } => {
                let TriSlot::Interior(i, j) = tri[t] else {
                    unreachable!("tri-face dofs are interior triangle dofs");
                };
                let w = cp[i] + cp[j] + cp[p - i - j];
                [cp[p * layer], cp[i] / w, cp[j] / w]
            }
            SlotKind::QuadFace { f, a, b } => {
                let (v0, v1) = TRI_EDGES[f];
                let (r0, r1) = (VERT_PK[v0], VERT_PK[v1]);
                let t = cp[a];
                [cp[b], r0[1] + t * (r1[1] - r0[1]), r0[2] + t * (r1[2] - r0[2])]
            }
            SlotKind::Interior => {
                let TriSlot::Interior(i, j) = tri[s % n_tri] else {
                    unreachable!("interior slots hold interior triangle dofs");
                };
                let w = cp[i] + cp[j] + cp[p - i - j];
                [cp[s / n_tri], cp[i] / w, cp[j] / w]
            }
        };
        for d in 0..3 {
            assert!(
                (r[d] - want[d]).abs() < 1e-12,
                "prism slot {s} classified {kind:?} but sits at {r:?} (expected {want:?})"
            );
        }
    }

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<EdgeKey, NodeId> = HashMap::new();
    let mut triface_dofs: HashMap<TriFaceKey, NodeId> = HashMap::new();
    let mut quadface_dofs: HashMap<QuadFaceKey, NodeId> = HashMap::new();
    let mut vol_dofs: HashMap<(ElemId, usize), NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * dpe);
    for (fe_i, &(pe, child)) in fine_parent.iter().enumerate() {
        let fe = fe_i as ElemId;
        let ns = fine.elem_nodes(fe);
        let (origin, m) = if child == IDENTITY {
            ([0.0_f64; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        } else {
            child_map(child as usize)
        };

        for (s, kind) in kinds.iter().enumerate() {
            let r = &fe_coords[s];
            let xi = [
                origin[0] + m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
                origin[1] + m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
                origin[2] + m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2],
            ];
            let id = match *kind {
                SlotKind::Vertex(v) => ns[v],
                SlotKind::Edge { e, m: pos } => {
                    let (a, b) = EDGES[e];
                    // Measure the within-edge index from the lower-id endpoint
                    // so both elements name the same dof; the GLL set is
                    // symmetric, so the dof at GLL position `pos` from one
                    // end sits at position `p − pos` from the other (the hex
                    // builder's tensor-index convention: both sides speak in
                    // `1..p−1`).
                    let (n0, n1) = (ns[a], ns[b]);
                    let key = if n0 <= n1 {
                        (n0, n1, pos)
                    } else {
                        (n1, n0, p - pos)
                    };
                    *edge_dofs.entry(key).or_insert_with(|| {
                        let x = pq.eval_at(pe, xi);
                        geo_coords.extend_from_slice(&x);
                        let id = next_dof;
                        next_dof += 1;
                        id
                    })
                }
                SlotKind::TriFace { layer, t } => {
                    // The dof's GLL index triple w.r.t. the element's own
                    // triangle frame: corner 0 ↔ (p−i−j), corner 1 ↔ i,
                    // corner 2 ↔ j (the barycentric GLL indices).
                    let (i, j) = match tri[t] {
                        TriSlot::Interior(i, j) => (i, j),
                        _ => unreachable!("tri face dofs are interior triangle dofs"),
                    };
                    let u = [p - i - j, i, j];
                    // Canonical frame of the fine face: corner 0 = minimum-id
                    // node, corner 1 = the smaller-id of the remaining two —
                    // both elements compute the same frame from the same ids.
                    let base = 3 * layer;
                    let ids = [ns[base], ns[base + 1], ns[base + 2]];
                    let k0 = (0..3).min_by_key(|&q| ids[q]).expect("non-empty");
                    let mut others = [(NodeId::MAX, 0usize); 2];
                    let mut n = 0;
                    for q in 0..3 {
                        if q != k0 {
                            others[n] = (ids[q], q);
                            n += 1;
                        }
                    }
                    others.sort_unstable_by_key(|&(id, _)| id);
                    let mut key3 = ids;
                    key3.sort_unstable();
                    *triface_dofs
                        .entry((key3, u[k0], u[others[0].1], u[others[1].1]))
                        .or_insert_with(|| {
                            let x = pq.eval_at(pe, xi);
                            geo_coords.extend_from_slice(&x);
                            let id = next_dof;
                            next_dof += 1;
                            id
                        })
                }
                SlotKind::QuadFace { f, a, b } => {
                    // Face corners in the in-face frame: (a from
                    // TRI_EDGES[f].0 → .1, b from the bottom layer), cycle
                    // order (0,0) (1,0) (1,1) (0,1).
                    let (v0, v1) = TRI_EDGES[f];
                    let cyc = [ns[v0], ns[v1], ns[v1 + 3], ns[v0 + 3]];
                    let pos = [[0usize, 0], [1, 0], [1, 1], [0, 1]];
                    let k0 = (0..4).min_by_key(|&q| cyc[q]).expect("non-empty");
                    let nx = (k0 + 1) % 4;
                    let pv = (k0 + 3) % 4;
                    let (sec, oth) = if cyc[nx] <= cyc[pv] { (nx, pv) } else { (pv, nx) };
                    let step = |q: usize| -> (usize, bool) {
                        if pos[q][0] != pos[k0][0] {
                            (0, pos[q][0] > pos[k0][0])
                        } else {
                            (1, pos[q][1] > pos[k0][1])
                        }
                    };
                    let (d1, s1) = step(sec);
                    let (d2, s2) = step(oth);
                    let j1 = if d1 == 0 {
                        if s1 { a } else { p - a }
                    } else if s1 {
                        b
                    } else {
                        p - b
                    };
                    let j2 = if d2 == 0 {
                        if s2 { a } else { p - a }
                    } else if s2 {
                        b
                    } else {
                        p - b
                    };
                    let mut key4 = cyc;
                    key4.sort_unstable();
                    *quadface_dofs.entry((key4, j1, j2)).or_insert_with(|| {
                        let x = pq.eval_at(pe, xi);
                        geo_coords.extend_from_slice(&x);
                        let id = next_dof;
                        next_dof += 1;
                        id
                    })
                }
                SlotKind::Interior => *vol_dofs.entry((fe, s)).or_insert_with(|| {
                    let x = pq.eval_at(pe, xi);
                    geo_coords.extend_from_slice(&x);
                    let id = next_dof;
                    next_dof += 1;
                    id
                }),
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
