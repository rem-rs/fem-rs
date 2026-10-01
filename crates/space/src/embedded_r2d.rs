//! Embedded (restricted) H(curl)/H(div) spaces — the space-layer counterpart
//! of MFEM's `ND_R2D_FECollection` / `RT_R2D_FECollection`
//! (`fem/fe_coll.cpp`, MFEM 4.10): a 2-D mesh (intrinsic dimension 2,
//! coordinates 2- or 3-component) carrying a **3-component** vector field —
//! MFEM `FiniteElementSpace(mesh, ND_R2D_FECollection(order, 2))` as used by
//! `examples/ex31.cpp` / `ex32p.cpp`.
//!
//! ## Global dof layout (1:1 with MFEM, pinned by `tmp/d102r2d/probe_space`)
//!
//! entity-major, exactly MFEM's `FiniteElementSpace` numbering:
//!
//! 1. **vertex dofs** — ND only: one z-directed (H¹-type) dof per mesh
//!    vertex (`ND_dof[POINT] = 1`); RT has none (`RT_dof[POINT] = 0`);
//! 2. **edge dofs** — one block per mesh edge in first-encounter order
//!    (`EdgeKey` walk over the elements): `ND_dof[SEGMENT] = 2p−1`
//!    (p in-plane, orientation-signed, then p−1 z-directed scalars) for ND;
//!    `RT_dof[SEGMENT] = p+1` (orientation-signed) for RT;
//! 3. **element-interior dofs** — per cell in element order (the collection's
//!    cell-geometry dof counts).
//!
//! Edge blocks map onto shared edges through the collection's
//! `DofOrderForOrientation(SEGMENT, or)` tables (`SegDofOrd`): a negatively
//! oriented element edge reverses (and sign-flips) its in-plane slots, the
//! scalar slots only reverse.
//!
//! ## Deliberately *not* a `FESpace` impl
//!
//! [`crate::FESpace`] would route these spaces into
//! `fem_assembly::VectorAssembler`, whose dispatch table pairs
//! `(HCurl|HDiv, cell, dim, order)` with the *regular* ND/RT elements — for
//! the embedded collections that element has a different dof count and a
//! hybrid slot layout, i.e. silent misassembly.  The space therefore exposes
//! its own (inherent) accessors and expects consumers to assemble through the
//! `fem_element::embedded` elements, like the ported `ex31` does.
//!
//! ## Cell coverage
//!
//! Tri3/Tri6 and Quad4/Quad8/Quad9 rows (the dof tables are reference-element
//! data; curved geometry enters only through the Jacobian callback of
//! [`HCurlR2dSpace::project_with_jac`]).  The `dim == 1` arms of the
//! collections (`ND_R2D_FECollection(p, 1)` on segment meshes, cell element
//! [`NdR2dSegment`]; `RT_R2D_FECollection(p, 1)`, plain INTEGRAL-map L2
//! segment) are covered at the element layer.
//!
//! [`NdR2dSegment`]: fem_element::embedded::NdR2dSegment

use std::collections::{BTreeSet, HashMap};

use fem_core::types::DofId;
use fem_element::embedded::{
    EmbeddedSlot, Jac2D, NdR2dQuad, NdR2dTri, RtR2dQuad, RtR2dTri,
};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::EdgeKey;

/// MFEM `Geometry::Constants<TRIANGLE>::Edges` (local edge vertex pairs).
const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
/// MFEM `Geometry::Constants<QUADRILATERAL>::Edges`.
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

fn local_edges_of(et: ElementType) -> &'static [(usize, usize)] {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => &TRI_EDGES,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => &QUAD_EDGES,
        other => panic!("embedded_r2d space: unsupported cell type {other:?}"),
    }
}

/// Per-space dof layout: flat slot arrays + CSR offsets + total.
struct Layout {
    dofs_flat: Vec<DofId>,
    signs_flat: Vec<f64>,
    elem_offsets: Vec<usize>,
    n_dofs: usize,
}

/// `SegDofOrd` resolution for one slot of an edge block.
///
/// Returns `(global offset within the block, sign)`.  `k` is the slot's
/// position within the block, `p` the collection's RT/ND index, `nd` selects
/// the ND (`2p−1` slots: first `p` in-plane, then `p−1` scalars) vs RT
/// (`p+1` in-plane) table.
fn seg_dof_ord(nd: bool, p: usize, k: usize, positive: bool) -> (usize, f64) {
    if positive {
        (k, 1.0)
    } else if nd {
        if k < p {
            (p - 1 - k, -1.0)
        } else {
            (3 * p - 2 - k, 1.0)
        }
    } else {
        (p - k, -1.0)
    }
}

/// H(curl) space of MFEM's `ND_R2D_FECollection(order, 2)`: the restricted
/// 3-component Nédélec field of `ex31` — in-plane components on the regular
/// ND space of the surface, the out-of-plane z component on the continuous
/// H¹ space of the surface.
pub struct HCurlR2dSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    layout: Layout,
    edge_base: HashMap<EdgeKey, DofId>,
    interior_base: Vec<usize>,
    tri: NdR2dTri,
    quad: NdR2dQuad,
}

impl<M: MeshTopology> HCurlR2dSpace<M> {
    /// `FiniteElementSpace(mesh, ND_R2D_FECollection(order, 2))`.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!(order >= 1, "ND_R2D_FECollection requires order >= 1");
        let p = order as usize;
        let n_elems = mesh.n_elements();

        // Entity-major numbering: vertex z dofs, then edge blocks (2p-1 each),
        // then per-element interiors.
        let mut edge_base: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next: DofId = mesh.n_nodes() as DofId;
        for e in 0..n_elems as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in local_edges_of(mesh.element_type(e)) {
                let key = EdgeKey::new(verts[li], verts[lj]);
                if let std::collections::hash_map::Entry::Vacant(vac) = edge_base.entry(key) {
                    vac.insert(next);
                    next += (2 * p - 1) as DofId;
                }
            }
        }
        let n_edge_dofs = next as usize - mesh.n_nodes();
        let tri = NdR2dTri::new(p);
        let quad = NdR2dQuad::new(p);
        let n_interior = |et: ElementType| match et {
            ElementType::Tri3 | ElementType::Tri6 => tri.n_dofs() - 3 - 3 * (2 * p - 1),
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                quad.n_dofs() - 4 - 4 * (2 * p - 1)
            }
            other => panic!("HCurlR2dSpace: unsupported cell type {other:?}"),
        };
        let mut interior_base = Vec::with_capacity(n_elems);
        let mut acc = mesh.n_nodes() + n_edge_dofs;
        for e in 0..n_elems as u32 {
            interior_base.push(acc);
            acc += n_interior(mesh.element_type(e));
        }
        let n_dofs = acc;

        // Slot tables (MFEM local slot order) via the element engines.
        let mut dofs_flat: Vec<DofId> = Vec::new();
        let mut signs_flat: Vec<f64> = Vec::new();
        let mut elem_offsets = Vec::with_capacity(n_elems + 1);
        elem_offsets.push(0usize);
        for e in 0..n_elems as u32 {
            let et = mesh.element_type(e);
            let verts = mesh.element_nodes(e);
            let local_edges = local_edges_of(et);
            let slots: &[EmbeddedSlot] = match et {
                ElementType::Tri3 | ElementType::Tri6 => tri.slots(),
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => quad.slots(),
                other => panic!("HCurlR2dSpace: unsupported cell type {other:?}"),
            };
            // per-edge running position within the block
            let mut block_pos = vec![0usize; local_edges.len()];
            let mut interior = interior_base[e as usize];
            for slot in slots.iter() {
                let (dof, sign) = match *slot {
                    EmbeddedSlot::Vertex(v, _) => (verts[v] as DofId, 1.0),
                    EmbeddedSlot::Edge(ei, _) | EmbeddedSlot::EdgeScalar(ei, _) => {
                        let (a, b) = local_edges[ei];
                        let (va, vb) = (verts[a], verts[b]);
                        let base = edge_base[&EdgeKey::new(va, vb)];
                        let positive = va < vb;
                        let k = block_pos[ei];
                        block_pos[ei] += 1;
                        let (off, s) = seg_dof_ord(true, p, k, positive);
                        (base + off as DofId, s)
                    }
                    EmbeddedSlot::Interior | EmbeddedSlot::InteriorScalar => {
                        let d = interior as DofId;
                        interior += 1;
                        (d, 1.0)
                    }
                };
                dofs_flat.push(dof);
                signs_flat.push(sign);
            }
            debug_assert_eq!(block_pos.iter().sum::<usize>(), local_edges.len() * (2 * p - 1));
            elem_offsets.push(dofs_flat.len());
        }

        HCurlR2dSpace {
            mesh,
            order,
            layout: Layout { dofs_flat, signs_flat, elem_offsets, n_dofs },
            edge_base,
            interior_base,
            tri,
            quad,
        }
    }

    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    /// Collection order `p` (`ND_R2D_FECollection(p, 2)`, `GetOrder() = p`).
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Total dofs (`FiniteElementSpace::GetVSize()`).
    pub fn n_dofs(&self) -> usize {
        self.layout.n_dofs
    }

    /// Global dof ids for element `e`, in the MFEM local slot order.
    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        let (s, t) = (self.layout.elem_offsets[elem as usize], self.layout.elem_offsets[elem as usize + 1]);
        &self.layout.dofs_flat[s..t]
    }

    /// Orientation signs for element `e`'s slots (±1.0).
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let (s, t) = (self.layout.elem_offsets[elem as usize], self.layout.elem_offsets[elem as usize + 1]);
        &self.layout.signs_flat[s..t]
    }

    /// Global dof id of the vertex z-dof at mesh vertex `v`.
    pub fn vertex_dof(&self, v: u32) -> DofId {
        v as DofId
    }

    /// Base global id of an edge's dof block (`2p−1` consecutive dofs).
    pub fn edge_block(&self, e: EdgeKey) -> Option<DofId> {
        self.edge_base.get(&e).copied()
    }

    /// Dofs whose entity touches a boundary face with a tag in `tags` —
    /// the endpoint vertex z-dofs plus the full block of every boundary edge
    /// (MFEM `GetEssentialTrueDofs` on all-boundary data: the interior-edge
    /// dofs stay free, the boundary edges' scalar z slots are essential too).
    pub fn boundary_dofs(&self, tags: &[i32]) -> Vec<DofId> {
        let mut set: BTreeSet<DofId> = BTreeSet::new();
        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !tags.contains(&self.mesh.face_tag(f)) {
                continue;
            }
            let nds = self.mesh.face_nodes(f);
            if nds.len() < 2 {
                continue;
            }
            for i in 0..nds.len() {
                let a = nds[i];
                let b = nds[(i + 1) % nds.len()];
                set.insert(a as DofId);
                set.insert(b as DofId);
                if let Some(&base) = self.edge_base.get(&EdgeKey::new(a, b)) {
                    for off in 0..(2 * self.order as usize - 1) {
                        set.insert(base + off as DofId);
                    }
                }
            }
        }
        set.into_iter().collect()
    }

    /// MFEM `GridFunction::ProjectCoefficient` for the collection:
    /// per element slot `k`, `dof_k = vc(x_k) · (J t̂_k)` for in-plane slots
    /// (`t̂_k` = the slot's `dof2tk` tangent) and `dof_k = vc_z(x_k)` for the
    /// z-directed slots, scattered through the orientation signs.
    ///
    /// `jac(elem, xi)` must return the element's intrinsic 2-D Jacobian and
    /// the physical image of `xi` (affine or isoparametric — the space is
    /// geometry-agnostic).
    pub fn project_with_jac(
        &self,
        f: &dyn Fn(&[f64]) -> [f64; 3],
        jac: &dyn Fn(u32, [f64; 2]) -> (Jac2D, [f64; 2]),
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.layout.n_dofs];
        for e in 0..self.mesh.n_elements() as u32 {
            let et = self.mesh.element_type(e);
            let el_dofs = self.element_dofs(e);
            let el_signs = self.element_signs(e);
            match et {
                ElementType::Tri3 | ElementType::Tri6 => {
                    let el = &self.tri;
                    for (k, site) in el.nodes().iter().enumerate() {
                        let (j, x) = jac(e, *site);
                        let vc = f(&x);
                        let tk = el.dof2tk()[k];
                        let val = if tk == 4 {
                            vc[2]
                        } else {
                            let t = el.tangent(tk);
                            let tx = [j.j00 * t[0] + j.j01 * t[1], j.j10 * t[0] + j.j11 * t[1]];
                            vc[0] * tx[0] + vc[1] * tx[1]
                        };
                        let g = el_dofs[k] as usize;
                        out[g] = el_signs[k] * val;
                    }
                }
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                    let el = &self.quad;
                    for (k, site) in el.nodes().iter().enumerate() {
                        let (j, x) = jac(e, *site);
                        let vc = f(&x);
                        let tk = el.dof2tk()[k];
                        let val = if tk == 4 {
                            vc[2]
                        } else {
                            let t = el.tangent(tk);
                            let tx = [j.j00 * t[0] + j.j01 * t[1], j.j10 * t[0] + j.j11 * t[1]];
                            vc[0] * tx[0] + vc[1] * tx[1]
                        };
                        let g = el_dofs[k] as usize;
                        out[g] = el_signs[k] * val;
                    }
                }
                other => panic!("HCurlR2dSpace: unsupported cell type {other:?}"),
            }
        }
        out
    }
}

/// H(div) space of MFEM's `RT_R2D_FECollection(p, 2)` (RT index `p ≥ 0`,
/// `GetOrder() = p+1`): the restricted 3-component flux field of `ex32p` —
/// in-plane components on the regular RT space of the surface, the
/// out-of-plane z component on the discontinuous L² space of the surface.
pub struct HDivR2dSpace<M: MeshTopology> {
    mesh: M,
    rt_order: u8,
    layout: Layout,
    edge_base: HashMap<EdgeKey, DofId>,
    interior_base: Vec<usize>,
    tri: RtR2dTri,
    quad: RtR2dQuad,
}

impl<M: MeshTopology> HDivR2dSpace<M> {
    /// `FiniteElementSpace(mesh, RT_R2D_FECollection(p, 2))`.
    pub fn new(mesh: M, rt_order: u8) -> Self {
        let p = rt_order as usize;
        let n_elems = mesh.n_elements();

        let mut edge_base: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next: DofId = 0;
        for e in 0..n_elems as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in local_edges_of(mesh.element_type(e)) {
                let key = EdgeKey::new(verts[li], verts[lj]);
                if let std::collections::hash_map::Entry::Vacant(vac) = edge_base.entry(key) {
                    vac.insert(next);
                    next += (p + 1) as DofId;
                }
            }
        }
        let n_edge_dofs = next as usize;
        let tri = RtR2dTri::new(p);
        let quad = RtR2dQuad::new(p);
        let n_interior = |et: ElementType| match et {
            ElementType::Tri3 | ElementType::Tri6 => tri.n_dofs() - 3 * (p + 1),
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                quad.n_dofs() - 4 * (p + 1)
            }
            other => panic!("HDivR2dSpace: unsupported cell type {other:?}"),
        };
        let mut interior_base = Vec::with_capacity(n_elems);
        let mut acc = n_edge_dofs;
        for e in 0..n_elems as u32 {
            interior_base.push(acc);
            acc += n_interior(mesh.element_type(e));
        }
        let n_dofs = acc;

        let mut dofs_flat: Vec<DofId> = Vec::new();
        let mut signs_flat: Vec<f64> = Vec::new();
        let mut elem_offsets = Vec::with_capacity(n_elems + 1);
        elem_offsets.push(0usize);
        for e in 0..n_elems as u32 {
            let et = mesh.element_type(e);
            let verts = mesh.element_nodes(e);
            let local_edges = local_edges_of(et);
            let slots: &[EmbeddedSlot] = match et {
                ElementType::Tri3 | ElementType::Tri6 => tri.slots(),
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => quad.slots(),
                other => panic!("HDivR2dSpace: unsupported cell type {other:?}"),
            };
            let mut block_pos = vec![0usize; local_edges.len()];
            let mut interior = interior_base[e as usize];
            for slot in slots.iter() {
                let (dof, sign) = match *slot {
                    EmbeddedSlot::Edge(ei, _) => {
                        let (a, b) = local_edges[ei];
                        let (va, vb) = (verts[a], verts[b]);
                        let base = edge_base[&EdgeKey::new(va, vb)];
                        let k = block_pos[ei];
                        block_pos[ei] += 1;
                        let (off, s) = seg_dof_ord(false, p, k, va < vb);
                        (base + off as DofId, s)
                    }
                    EmbeddedSlot::Interior | EmbeddedSlot::InteriorScalar => {
                        let d = interior as DofId;
                        interior += 1;
                        (d, 1.0)
                    }
                    EmbeddedSlot::Vertex(..) | EmbeddedSlot::EdgeScalar(..) => {
                        unreachable!("RT_R2D elements carry no vertex/edge-scalar slots")
                    }
                };
                dofs_flat.push(dof);
                signs_flat.push(sign);
            }
            debug_assert_eq!(block_pos.iter().sum::<usize>(), local_edges.len() * (p + 1));
            elem_offsets.push(dofs_flat.len());
        }

        HDivR2dSpace {
            mesh,
            rt_order,
            layout: Layout { dofs_flat, signs_flat, elem_offsets, n_dofs },
            edge_base,
            interior_base,
            tri,
            quad,
        }
    }

    pub fn mesh(&self) -> &M {
        &self.mesh
    }

    /// RT index `p` (`RT_R2D_FECollection(p, 2)`; `GetOrder() = p + 1`).
    pub fn rt_order(&self) -> u8 {
        self.rt_order
    }

    pub fn n_dofs(&self) -> usize {
        self.layout.n_dofs
    }

    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        let (s, t) = (self.layout.elem_offsets[elem as usize], self.layout.elem_offsets[elem as usize + 1]);
        &self.layout.dofs_flat[s..t]
    }

    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let (s, t) = (self.layout.elem_offsets[elem as usize], self.layout.elem_offsets[elem as usize + 1]);
        &self.layout.signs_flat[s..t]
    }

    /// Base global id of an edge's dof block (`p+1` consecutive dofs).
    pub fn edge_block(&self, e: EdgeKey) -> Option<DofId> {
        self.edge_base.get(&e).copied()
    }

    /// Dofs on boundary faces with a tag in `tags`: the full block of every
    /// boundary edge (RT_R2D carries no vertex dofs).
    pub fn boundary_dofs(&self, tags: &[i32]) -> Vec<DofId> {
        let mut set: BTreeSet<DofId> = BTreeSet::new();
        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !tags.contains(&self.mesh.face_tag(f)) {
                continue;
            }
            let nds = self.mesh.face_nodes(f);
            if nds.len() < 2 {
                continue;
            }
            for i in 0..nds.len() {
                let a = nds[i];
                let b = nds[(i + 1) % nds.len()];
                if let Some(&base) = self.edge_base.get(&EdgeKey::new(a, b)) {
                    for off in 0..(self.rt_order as usize + 1) {
                        set.insert(base + off as DofId);
                    }
                }
            }
        }
        set.into_iter().collect()
    }

    /// MFEM `RT_R2D_FiniteElement::Project`: per element slot `k`,
    /// `dof_k = n̂_k^T adj(J) vc(x_k) + Weight · vc_z(x_k) · [z slot]`.
    pub fn project_with_jac(
        &self,
        f: &dyn Fn(&[f64]) -> [f64; 3],
        jac: &dyn Fn(u32, [f64; 2]) -> (Jac2D, [f64; 2]),
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.layout.n_dofs];
        for e in 0..self.mesh.n_elements() as u32 {
            let et = self.mesh.element_type(e);
            let el_dofs = self.element_dofs(e);
            let el_signs = self.element_signs(e);
            match et {
                ElementType::Tri3 | ElementType::Tri6 => {
                    let el = &self.tri;
                    for (k, site) in el.nodes().iter().enumerate() {
                        let (j, x) = jac(e, *site);
                        let vc = f(&x);
                        let nk = el.dof2nk()[k];
                        let val = if nk == 3 {
                            j.det * vc[2]
                        } else {
                            let n = el.normal(nk);
                            let ajv = [
                                j.j11 * vc[0] - j.j01 * vc[1],
                                -j.j10 * vc[0] + j.j00 * vc[1],
                            ];
                            n[0] * ajv[0] + n[1] * ajv[1]
                        };
                        let g = el_dofs[k] as usize;
                        out[g] = el_signs[k] * val;
                    }
                }
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                    let el = &self.quad;
                    for (k, site) in el.nodes().iter().enumerate() {
                        let (j, x) = jac(e, *site);
                        let vc = f(&x);
                        let nk = el.dof2nk()[k];
                        let val = if nk == 4 {
                            j.det * vc[2]
                        } else {
                            let n = el.normal(nk);
                            let ajv = [
                                j.j11 * vc[0] - j.j01 * vc[1],
                                -j.j10 * vc[0] + j.j00 * vc[1],
                            ];
                            n[0] * ajv[0] + n[1] * ajv[1]
                        };
                        let g = el_dofs[k] as usize;
                        out[g] = el_signs[k] * val;
                    }
                }
                other => panic!("HDivR2dSpace: unsupported cell type {other:?}"),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `SegDofOrd` tables from `fe_coll.cpp` (ND_R2D / RT_R2D), for the
    /// pin test and as readable documentation.
    #[test]
    fn seg_dof_ord_matches_mfem_tables() {
        // ND_R2D p=2: 3 slots, negative orientation:
        //   in-plane reversed+flipped, scalar reversed-only.
        assert_eq!(seg_dof_ord(true, 2, 0, false), (1, -1.0));
        assert_eq!(seg_dof_ord(true, 2, 1, false), (0, -1.0));
        assert_eq!(seg_dof_ord(true, 2, 2, false), (2, 1.0));
        // positive orientation: identity
        assert_eq!(seg_dof_ord(true, 2, 0, true), (0, 1.0));
        assert_eq!(seg_dof_ord(true, 2, 2, true), (2, 1.0));
        // RT_R2D p=1: 2 slots, negative: reversed+flipped.
        assert_eq!(seg_dof_ord(false, 1, 0, false), (1, -1.0));
        assert_eq!(seg_dof_ord(false, 1, 1, false), (0, -1.0));
    }
}
