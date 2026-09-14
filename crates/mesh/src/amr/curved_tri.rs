//! Curved (high-order) geometry support for uniform Tri3 refinement — the
//! triangle analogue of [`super::curved_hex`], [`super::curved_quad`] and
//! [`super::curved_prism`].
//!
//! MFEM refines a curved 2-D mesh (`nodes` grid function present) in two steps
//! (`Mesh::UniformRefinement2D_base` + `UpdateNodes`):
//!
//! 1. the straight `vertices` array is averaged exactly like a linear mesh
//!    (vertex layout `[coarse vertices | edge midpoints]`, edge ids first-touch
//!    in element × local-edge order — `refine_marked` already implements that
//!    numbering, so unlike the wedge case no id gate is needed);
//! 2. because `update_nodes` is set for curved meshes, the refined `nodes`
//!    grid function replaces the vertex positions: for every *fine* element
//!    the child's embedded sub-triangle (`tri_children` point matrices, an
//!    affine map `parent_ref = origin + M·child_ref`, the center child
//!    rotated) is applied to the fine dof reference points and the coarse
//!    element's own FE is evaluated there, so the fine geometry is the
//!    **per-child evaluation of the parent's nodal field**, with dofs shared
//!    between elements meeting on the same fine edge (MFEM's first-touch
//!    `mark` rule).
//!
//! The positional layout of the `(p+1)(p+2)/2` geometry dofs is the element's
//! own table ([`H1TriPk`] — MFEM `H1_TriangleElement`'s Gauss-Lobatto lattice:
//! corners, the three edges counted from their first vertex, then the
//! interior), the same layout `Mesh::set_curvature`'s 2-D triangle path writes
//! and `fem_io`'s writer pairs the table with, so element, numbering and this
//! module cannot drift apart.
//!
//! Refinement of a **mixed** 2-D mesh never sees this module today: no code
//! path can build a curved mixed Tri3+Quad4 mesh (`set_curvature` refuses
//! mixed meshes; the reader reads mixed meshes straight-sided), so
//! `refine_uniform_2d_mixed` keeps its `geometry: None` (unreachable for
//! curved parents).

use std::collections::HashMap;

use fem_core::{ElemId, NodeId};
use fem_element::lagrange::H1TriPk;
use fem_element::ReferenceElement;

use crate::element_type::ElementType;
use crate::simplex::{GeometryData, Mesh};

/// Reference coordinates ([0,1]²) of the three vertices of a Tri3 in MFEM
/// `Geometry::Constants<TRIANGLE>` order: `0=(0,0) 1=(1,0) 2=(0,1)`.
const VERTS: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

/// Local edges of a triangle as `(from, to)` vertex pairs, in MFEM
/// `Constants<TRIANGLE>::Edges` order.
const EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

/// MFEM `UniformRefinement2D_base`'s `tri_children` point matrices
/// (`mesh/mesh.cpp`, `A = 0, B = 0.5, C = 1`): 4 children, each 3 reference
/// points — the images of the child's reference triangle vertices in the
/// parent's reference triangle.  Child order = embedding matrix index = fine
/// element order: corner 0, center, corner 1, corner 2 (the center child's
/// frame is rotated, like the wedge's center children).
const MFEM_TRI_CHILDREN: [[[f64; 2]; 3]; 4] = [
    [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
    [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]],
    [[0.5, 0.0], [1.0, 0.0], [0.5, 0.5]],
    [[0.0, 0.5], [0.5, 0.5], [0.0, 1.0]],
];

/// Affine map `parent_ref = origin + M·child_ref` of child `child`.
fn child_map(child: usize) -> ([f64; 2], [[f64; 2]; 2]) {
    let pts = &MFEM_TRI_CHILDREN[child];
    let origin = pts[0];
    let mut m = [[0.0_f64; 2]; 2];
    for (k, o) in origin.iter().enumerate() {
        m[k][0] = pts[1][k] - o;
        m[k][1] = pts[2][k] - o;
    }
    (origin, m)
}

/// Child index of an **unrefined** element copied through unchanged (its fine
/// element *is* the parent element: the reference map is the identity and
/// every evaluation lands exactly on a parent dof point).
pub(crate) const IDENTITY: u8 = 4;

/// Read-only view of an order-`p` (`p ≥ 2`) triangle [`GeometryData`] attached
/// to a 2-D Tri3 mesh (as built by `Mesh::set_curvature`'s 2-D triangle path).
pub(crate) struct TriPkGeometry<'a> {
    geo: &'a GeometryData,
    fe: H1TriPk,
    /// Geometry order `p`.
    pub(crate) order: usize,
    /// Geometry dofs per element, `(p+1)(p+2)/2` (`GeometryData::nodes_per_elem`).
    pub(crate) dpe: usize,
}

impl<'a> TriPkGeometry<'a> {
    /// Returns `Some` iff `mesh` is a Tri3 mesh carrying order-`p ≥ 2`
    /// geometry with `(p+1)(p+2)/2` dofs per element in H1TriPk's slot order.
    pub(crate) fn new(mesh: &'a Mesh<2>) -> Option<Self> {
        if mesh.elem_type != ElementType::Tri3 {
            return None;
        }
        let geo = mesh.geometry.as_ref()?;
        let order = geo.order as usize;
        if order < 2 {
            // Order 1 (or absent) geometry: the straight averaging the
            // refinement kernels already do *is* the interpolation.
            return None;
        }
        let dpe = (order + 1) * (order + 2) / 2;
        if geo.nodes_per_elem != dpe || geo.conn.len() < mesh.n_elems() * dpe {
            return None;
        }
        let fe = H1TriPk::new(order);
        if fe.n_dofs() != dpe {
            return None;
        }
        Some(TriPkGeometry { geo, fe, order, dpe })
    }

    fn elem_dofs(&self, e: ElemId) -> &[NodeId] {
        let o = e as usize * self.dpe;
        &self.geo.conn[o..o + self.dpe]
    }

    /// Evaluate the parent element's order-`p` geometry field at the reference
    /// triangle point `xi` — the interpolation of the element's geometry dofs
    /// with the element's own nodal basis (exact δ at the dof points, so every
    /// evaluation at a parent dof reads the stored value back bit for bit).
    pub(crate) fn eval_at(&self, e: ElemId, xi: [f64; 2]) -> [f64; 2] {
        let mut w = vec![0.0_f64; self.dpe];
        self.fe.eval_basis(&xi, &mut w);
        let dofs = self.elem_dofs(e);
        let mut out = [0.0_f64; 2];
        for (s, &w) in w.iter().enumerate() {
            if w != 0.0 {
                let d = dofs[s] as usize;
                out[0] += w * self.geo.coords[2 * d];
                out[1] += w * self.geo.coords[2 * d + 1];
            }
        }
        out
    }

    /// Coordinate of the new fine vertex at the midpoint of local edge `li`
    /// (index into [`EDGES`]): the parent geometry field evaluated at the edge
    /// midpoint.  For an order-2 parent that point *is* the parent's mid-edge
    /// geometry dof and the evaluation reads it back exactly (all basis
    /// weights are 0/1) — MFEM's `SetVerticesFromNodes` pick.
    pub(crate) fn edge_pick(&self, e: ElemId, li: usize) -> [f64; 2] {
        let (a, b) = EDGES[li];
        let (ra, rb) = (VERTS[a], VERTS[b]);
        self.eval_at(e, [0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1])])
    }
}

/// One slot of the [`H1TriPk`] table.
#[derive(Clone, Copy, Debug)]
enum SlotKind {
    Corner(usize),
    /// Edge `0..3` ([`EDGES`]), GLL position `m` (1..p) counted from the
    /// edge's first vertex.
    Edge { e: usize, m: usize },
    /// Element interior (the slot's index, private per fine element).
    Interior,
}

/// The [`SlotKind`] of every [`H1TriPk`] slot of an order-`p` triangle
/// (H1TriPk's documented layout: corners, then the three edge blocks counted
/// from their first vertex, then the interior `j`-outer `i`-inner).
fn slot_kinds(p: usize) -> Vec<SlotKind> {
    let mut out = vec![SlotKind::Corner(0), SlotKind::Corner(1), SlotKind::Corner(2)];
    for e in 0..3 {
        for m in 1..p {
            out.push(SlotKind::Edge { e, m });
        }
    }
    while out.len() < (p + 1) * (p + 2) / 2 {
        out.push(SlotKind::Interior);
    }
    out
}

/// Key of a shared fine edge dof: the fine edge (its sorted node pair) plus
/// the dof's GLL index within the edge, measured from its lower-id node (both
/// sides speak in `1..p−1`; the GLL set is symmetric, so the dof at position
/// `m` from one end sits at `p − m` from the other).
type EdgeKey = (NodeId, NodeId, usize);

/// Build the refined mesh's [`GeometryData`] from the parent's order-`p`
/// triangle geometry.
///
/// `fine` is the refined mesh (connectivity + final vertex coordinates, where
/// every new vertex already carries the parent-field coordinate computed by
/// the refinement kernels).  `fine_parent` maps each fine element to its
/// parent element and child index — the index into MFEM's `tri_children`
/// tables *in fine element order* ([`IDENTITY`] for an unrefined element).
///
/// The fine geometry dof ids occupy a fresh index space: ids `0..n_fine_nodes`
/// are the fine mesh nodes (fine vertex coordinates), followed by the edge
/// dofs created on first touch (keyed by the fine node pair so elements
/// sharing an edge share the dof) and one interior dof set per fine element.
/// Values are the parent order-`p` field evaluated at the corresponding
/// reference point — read back exactly wherever the point coincides with a
/// parent dof (in particular every dof of an identity element) and the
/// standard order-`p` interpolation otherwise, i.e. MFEM's
/// `GetLocalInterpolation` with its first-touch `mark` rule.
pub(crate) fn build_refined_tri_geometry(
    parent: &Mesh<2>,
    fine: &Mesh<2>,
    fine_parent: &[(ElemId, u8)],
) -> Option<GeometryData> {
    let pq = TriPkGeometry::new(parent)?;
    let p = pq.order;
    let dpe = pq.dpe;
    debug_assert_eq!(fine.n_elems(), fine_parent.len());

    let fe_coords: Vec<[f64; 2]> =
        pq.fe.dof_coords().iter().map(|c| [c[0], c[1]]).collect();
    debug_assert_eq!(fe_coords.len(), dpe);
    let kinds = slot_kinds(p);
    debug_assert_eq!(kinds.len(), dpe);

    let n_fine_nodes = fine.n_nodes() as NodeId;
    let mut geo_coords: Vec<f64> = fine.coords.clone();
    let mut next_dof = n_fine_nodes;
    let mut edge_dofs: HashMap<EdgeKey, NodeId> = HashMap::new();
    let mut vol_dofs: HashMap<(ElemId, usize), NodeId> = HashMap::new();

    let mut conn: Vec<NodeId> = Vec::with_capacity(fine.n_elems() * dpe);
    for (fe_i, &(pe, child)) in fine_parent.iter().enumerate() {
        let fe = fe_i as ElemId;
        let ns = fine.elem_nodes(fe);
        let (origin, m) = if child == IDENTITY {
            ([0.0_f64; 2], [[1.0, 0.0], [0.0, 1.0]])
        } else {
            child_map(child as usize)
        };

        for (s, kind) in kinds.iter().enumerate() {
            let r = &fe_coords[s];
            let xi = [
                origin[0] + m[0][0] * r[0] + m[0][1] * r[1],
                origin[1] + m[1][0] * r[0] + m[1][1] * r[1],
            ];
            let id = match *kind {
                SlotKind::Corner(c) => ns[c],
                SlotKind::Edge { e, m: pos } => {
                    let (a, b) = EDGES[e];
                    let (n0, n1) = (ns[a], ns[b]);
                    let key = if n0 <= n1 { (n0, n1, pos) } else { (n1, n0, p - pos) };
                    *edge_dofs.entry(key).or_insert_with(|| {
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
