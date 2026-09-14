//! Arbitrary-order Lagrange element on the reference triangular prism.
//!
//! Reference domain: (ξ, η, ζ) where (η, ζ) ∈ unit triangle `(0,0),(1,0),(0,1)`
//! and ξ ∈ [0,1] (extrusion direction). Volume = 0.5.
//!
//! The basis is the tensor product of a 1-D Lagrange basis on [0,1] in ξ and
//! the triangular Lagrange basis in (η, ζ) (identical to
//! [`H1TriPk`](super::factory::H1TriPk)).  Both factors place their nodes on
//! the **closed Gauss-Lobatto points** (`Poly_1D::ClosedPoints`), which is
//! MFEM `H1_WedgeElement`'s node placement (`fem/fe/fe_h1.cpp:863`:
//! `Nodes.IntPoint(i) = (t_Nodes[t_dof[i]].x, t_Nodes[t_dof[i]].y,
//! s_Nodes[s_dof[i]].x)` with `t_Nodes`/`s_Nodes` the `H1_TriangleElement` /
//! `H1_SegmentElement` node tables).  The equispaced placement this element
//! used before (D164) agrees with MFEM only for `p ≤ 2`; from `p = 3` on the
//! GLL points (`0.276393…` / `0.723607…`) differ from `1/3` / `2/3` and every
//! curved prism geometry evaluated with the equispaced lattice was a different
//! order-`p` interpolant than MFEM's.
//!
//! DOF count: `(p+1) × (p+1)(p+2)/2`.

use crate::lagrange::factory::{dlag1d_on, lag1d_on, H1TriPk};
use crate::quadrature::prism_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

/// MFEM `Geometry::Constants<Geometry::PRISM>::Edges` (`fem/geom.cpp`): local
/// edge `k` runs from local vertex `EDGES[k][0]` to `EDGES[k][1]`.  The `H1`
/// wedge's edge blocks follow this order (bottom triangle edges 0-2, top
/// triangle edges 3-5, vertical edges 6-8).
pub const PRISM_EDGES: [[usize; 2]; 9] = [
    [0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5],
];

/// One local dof of MFEM's `H1_WedgeElement`: which entity it lives on and
/// where on that entity (the constructor's slot fill order,
/// `fem/fe/fe_h1.cpp:863`).  This is MFEM's **entity** layout
/// `[v0…v5 | e01 e12 e20 e34 e45 e53 e03 e14 e25 | tri0 tri1 | q0 q1 q2 |
/// interior]` — as opposed to [`PrismPk`]'s layer-major order of the *same*
/// Gauss-Lobatto lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H1PrismSlot {
    /// Local vertex `0..6` (bottom triangle 0-2, top triangle 3-5).
    Vertex(usize),
    /// Local edge [`PRISM_EDGES`] index and the 0-based position counted from
    /// the edge's first vertex (the edge's own local orientation).
    Edge(usize, usize),
    /// Local triangular face (0 = bottom, 1 = top) and the running index of
    /// the face dof in the element's *local* triangle orientation
    /// (`H1_TriangleElement` interior order, j-outer i-inner).
    TriFace(usize, usize),
    /// Local quadrilateral face (2..4, [`PRISM_EDGES`]-adjacent
    /// `(0,1,4,3) (1,2,5,4) (2,0,3,5)`) and the 1-based in-face Gauss-Lobatto
    /// indices `(i, j)`: `i` along `FaceVert[0] → FaceVert[1]`, `j` along
    /// `FaceVert[0] → FaceVert[3]` of the *local* face.
    QuadFace(usize, usize, usize),
    /// Element-interior dof, in MFEM's order (layer `k = 1..p-1` outer, the
    /// triangle interior in the bottom face's running order within a layer).
    Interior(usize),
}

/// MFEM's `H1_WedgeElement(p)` slot table, in slot order (D168 ground truth,
/// verified against `FiniteElementSpace::GetElementDofs` +
/// `FiniteElement::GetNodes` on `MakeCartesian3D` wedge stacks, probe
/// `tmp/a34_prism_h1_probe.cpp`).
///
/// Mirrors the constructor's `t_dof`/`s_dof` fill order: vertices, then the
/// nine edge blocks, then the two triangular faces (the bottom face's dof
/// index is permuted relative to the top's — MFEM is self-consistent because
/// the two faces' `FaceVert` orders are transposed the same way), then the
/// three quadrilateral faces, then the interior.
pub fn h1_prism_slots(p: usize) -> Vec<H1PrismSlot> {
    assert!(p >= 1, "h1_prism_slots: order must be >= 1");
    let ne = p - 1;
    let nt = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
    let nq = ne * ne;
    let mut out: Vec<H1PrismSlot> = Vec::with_capacity((p + 1) * (p + 1) * (p + 2) / 2);    // Vertices: `t_dof = (0,1,2,0,1,2)`, `s_dof = (0,0,0,1,1,1)`.
    for v in 0..6 {
        out.push(H1PrismSlot::Vertex(v));
    }
    // Edges: slot `5 + kk*ne + i` (i = 1..p-1) — the constructor *fills* in
    // (i, kk) order but *places* edge kk's `ne` dofs contiguously at slots
    // `6 + kk*ne …`, position `i` counted from the edge's first vertex.
    for s_e in 0..9 * ne {
        let kk = s_e / ne;
        let i = s_e % ne + 1;
        out.push(H1PrismSlot::Edge(kk, i - 1));
    }
    // Triangular faces: the constructor fills bottom/top together per (j, i)
    // but *places* the bottom face's `nt` dofs at slots `6 + 9ne …` and the
    // top face's at `6 + 9ne + nt …` (each block in the running (j, i) order;
    // the bottom's `t_dof` is the permuted triangle interior index
    // `l = j - p + ((2p-1-i)·i)/2`, the top's the running index).
    let mut k = 0usize;
    for _ in 0..nt {
        out.push(H1PrismSlot::TriFace(0, k));
        k += 1;
    }
    for k in 0..nt {
        out.push(H1PrismSlot::TriFace(1, k));
    }
    // Quadrilateral faces: f = 0..3, slot `6 + 9ne + 2nt + f*nq + (j-1)*ne + (i-1)`.
    for f in 0..3 {
        for j in 1..p {
            for i in 1..p {
                out.push(H1PrismSlot::QuadFace(2 + f, i, j));
            }
        }
    }
    // Interior: layer kk = 1..p-1, the triangle interior in the bottom face's
    // running order within a layer.
    let mut m = 0usize;
    for _kk in 1..p {
        for _j in 1..p {
            for _i in 1..(p - _j) {
                out.push(H1PrismSlot::Interior(m));
                m += 1;
            }
        }
    }
    debug_assert_eq!(out.len(), 6 + 9 * ne + 2 * nt + 3 * nq + nt * ne);
    debug_assert_eq!(out.len(), (p + 1) * (p + 1) * (p + 2) / 2);
    out
}

/// MFEM `H1_WedgeElement(p)` clone — the [`PrismPk`] Gauss-Lobatto lattice in
/// MFEM's **entity** slot order (see [`h1_prism_slots`]).
///
/// This is the element `H1_FECollection(p, 3)` puts on wedge cells, and the
/// layout `FiniteElementSpace::GetElementDofs` returns: vertices → edges →
/// triangular faces → quadrilateral faces → interior, each block oriented by
/// the element's own local entity orientation.  It is *not* a different
/// function space from [`PrismPk`]: slot `m` of this element **is** slot
/// `perm[m]` of `PrismPk` (same shape function, same node), so `PrismPk` (the
/// layer order the mesh geometry table is written in) and this element (the
/// entity order the H¹ space is numbered in) always describe the same field.
pub struct H1PrismPk {
    inner: std::sync::Arc<H1PrismPkInner>,
}

struct H1PrismPkInner {
    order: usize,
    /// Layer-major GLL element providing the basis (and the exact-nodal
    /// shortcut) on the shared lattice.
    layer: PrismPk,
    /// H1 slot `m` = layer slot `perm[m]`.
    perm: Vec<usize>,
    /// Slot reference points, H1 slot order.
    nodes: Vec<[f64; 3]>,
}

impl H1PrismPk {
    /// Build (or fetch from the per-order cache) the element.
    pub fn new(p: usize) -> Self {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<H1PrismPkInner>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let inner = {
            let mut m = cache.lock().expect("H1PrismPk cache poisoned");
            m.entry(p).or_insert_with(|| Arc::new(h1_prism_pk_build(p))).clone()
        };
        Self { inner }
    }

    /// MFEM's `H1_WedgeElement(p)` slot table — see [`h1_prism_slots`].
    pub fn slot_labels(p: usize) -> Vec<H1PrismSlot> {
        h1_prism_slots(p)
    }

    /// The layer-major slot of [`PrismPk`] that holds slot `m`'s dof.
    pub fn layer_perm(&self) -> &[usize] {
        &self.inner.perm
    }
}

/// The `H1_SegmentElement` node table as layer (cp) indices:
/// `(cp[0], cp[p], cp[1], …, cp[p-1])`.
fn seg_node_layer(s: usize, p: usize) -> usize {
    match s {
        0 => 0,
        1 => p,
        s => s - 1,
    }
}

fn h1_prism_pk_build(p: usize) -> H1PrismPkInner {
    let layer = PrismPk::new(p);
    let n_tri = layer.n_dofs() / (p + 1);
    let layer_coords = layer.dof_coords();
    let tri: Vec<[f64; 2]> = layer_coords
        .iter()
        .take(n_tri)
        .map(|c| [c[1], c[2]])
        .collect();
    let slots = h1_prism_slots(p);
    // `t_dof` (triangle node index) and `s_dof` (segment node index) of every
    // slot, from the constructor's fill order.
    let mut perm = Vec::with_capacity(slots.len());
    let mut nodes = Vec::with_capacity(slots.len());
    let ne = p - 1;
    let nt = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
    // Running (t_dof, s_dof) per slot, reconstructed in the same order.
    let mut ts: Vec<(usize, usize)> = Vec::with_capacity(slots.len());
    for &(t, s) in [(0usize, 0usize), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)].iter() {
        ts.push((t, s));
    }
    // Edges: slot `5 + kk*ne + i` — edge `kk` contiguous, position `i` within
    // the block (see `h1_prism_slots`).
    for s_e in 0..9 * ne {
        let kk = s_e / ne;
        let i = s_e % ne + 1;
        let (t, s) = if kk < 6 { (2 + (kk % 3) * ne + i, kk / 3) } else { (kk - 6, i + 1) };
        ts.push((t, s));
    }
    let mut bottoms: Vec<(usize, usize)> = Vec::with_capacity(nt);
    let mut tops: Vec<(usize, usize)> = Vec::with_capacity(nt);
    let mut k = 0usize;
    for j in 1..p {
        for i in 1..(p - j) {
            // MFEM computes `l` in (possibly negative) int arithmetic.
            let l = (j as i64) - (p as i64) + (((2 * p - 1 - i) * i) / 2) as i64;
            debug_assert!((0..nt as i64).contains(&l));
            bottoms.push((3 * p + l as usize, 0));
            tops.push((3 * p + k, 1));
            k += 1;
        }
    }
    debug_assert_eq!(k, nt);
    ts.extend(bottoms);
    ts.extend(tops);
    for _f in 0..3 {
        for j in 1..p {
            for i in 1..p {
                ts.push((2 + _f * ne + i, 1 + j));
            }
        }
    }
    for kk in 1..p {
        let mut l = 0usize;
        for _j in 1..p {
            for _i in 1..(p - _j) {
                ts.push((3 * p + l, 1 + kk));
                l += 1;
            }
        }
    }
    debug_assert_eq!(ts.len(), slots.len());
    for (i, &(t, s)) in ts.iter().enumerate() {
        let layer_k = seg_node_layer(s, p);
        perm.push(layer_k * n_tri + t);
        nodes.push([layer_coords[perm[i]][0], tri[t][0], tri[t][1]]);
    }
    H1PrismPkInner { order: p, layer, perm, nodes }
}

impl ReferenceElement for H1PrismPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.inner.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.inner.nodes.len()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let mut layer_vals = vec![0.0_f64; self.inner.layer.n_dofs()];
        self.inner.layer.eval_basis(xi, &mut layer_vals);
        for (m, &src) in self.inner.perm.iter().enumerate() {
            values[m] = layer_vals[src];
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let mut layer_grads = vec![0.0_f64; self.inner.layer.n_dofs() * 3];
        self.inner.layer.eval_grad_basis(xi, &mut layer_grads);
        for (m, &src) in self.inner.perm.iter().enumerate() {
            grads[m * 3] = layer_grads[src * 3];
            grads[m * 3 + 1] = layer_grads[src * 3 + 1];
            grads[m * 3 + 2] = layer_grads[src * 3 + 2];
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        prism_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}


/// Arbitrary-order Lagrange element on the reference triangular prism, on
/// MFEM `H1_WedgeElement`'s **Gauss-Lobatto** node lattice.
///
/// DOF ordering: layer-by-layer, where layer `k` (ξ = `cp[k]`, the `k`-th
/// closed Gauss-Lobatto point on [0,1]) contains the full set of triangular
/// DOFs.  Within each layer, ordering follows [`H1TriPk`] (MFEM's triangle
/// node order: vertices, then the `ζ = 0` edge, the hypotenuse, the `η = 0`
/// edge, then the interior).  This is *not* MFEM's slot order —
/// `H1_WedgeElement` orders its (identical) lattice by entity (vertices →
/// edges → faces → interior, see `fem_io`'s `prism_h1_slots`); the layer-major
/// order is the contract between this element and every consumer that reads
/// the mesh geometry table (`Mesh::set_curvature`'s prism path,
/// `element_jacobian`, `crates/assembly`'s `geo_ref_elem`,
/// `curved::CurvedMesh`, `fem-io`'s writer), which is pinned by
/// `crates/mesh/tests/d152_prism_curvature.rs::prism_pk_slot_order_is_frozen`.
pub struct PrismPk {
    order: usize,
    /// GLL triangle factor (MFEM `H1_TriangleElement`): nodes *and* basis.
    tri: H1TriPk,
    n_tri: usize,
    /// Closed Gauss-Lobatto points on [0,1], ascending (`cp[0] = 0`,
    /// `cp[p] = 1`) — MFEM `Poly_1D::ClosedPoints(p)`.
    cp: Vec<f64>,
}

impl PrismPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let tri = H1TriPk::new(p);
        let n_tri = tri.n_dofs();
        let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
        let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        Self { order: p, tri, n_tri, cp }
    }

    fn dof_index(&self, k: usize, tri_dof: usize) -> usize {
        k * self.n_tri + tri_dof
    }

    /// The dof sitting exactly at `xi` (bit-identical coordinates), if any.
    ///
    /// A nodal basis is δ at its own nodes; the 1-D factor (`lag1d_on`)
    /// returns that exactly, but the triangle factor's Vandermonde-inverse
    /// evaluation carries `O(1e-14)` noise *at the nodes themselves* (`p = 4`
    /// monomial Vandermonde), which fem-io's prism `nodes` writer — whose
    /// interpolation matrix is the identity once the geometry lattice matches
    /// (D164) — would multiply through into every written value.  Callers that
    /// evaluate at a dof's own reference point (the writer, and any consumer
    /// passing stored `dof_coords()` back in) get the exact δ instead.
    fn node_at(&self, xi: &[f64]) -> Option<usize> {
        let k = self.cp.iter().position(|&c| c == xi[0])?;
        let tri = self.tri.dof_coords();
        let t = tri
            .iter()
            .position(|c| c[0] == xi[1] && c[1] == xi[2])?;
        Some(self.dof_index(k, t))
    }
}

impl ReferenceElement for PrismPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        (self.order + 1) * self.n_tri
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        // `φ_{k,t}(ξ, η, ζ) = σ_k(ξ) · τ_t(η, ζ)` with `σ` the 1-D Lagrange
        // basis on the GLL points and `τ` the GLL triangle basis — the unique
        // nodal basis on the tensor lattice, i.e. MFEM's `H1_WedgeElement::
        // CalcShape` (`shape[i] = t_shape[t_dof[i]]·s_shape[s_dof[i]]`).
        if let Some(dof) = self.node_at(xi) {
            values[..self.n_dofs()].fill(0.0);
            values[dof] = 1.0;
            return;
        }
        let mut t_vals = vec![0.0_f64; self.n_tri];
        self.tri.eval_basis(&xi[1..3], &mut t_vals);
        let mut s_vals = vec![0.0_f64; self.order + 1];
        lag1d_on(&self.cp, xi[0], &mut s_vals);
        for k in 0..=self.order {
            let sv = s_vals[k];
            for t in 0..self.n_tri {
                values[self.dof_index(k, t)] = sv * t_vals[t];
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let mut t_vals = vec![0.0_f64; self.n_tri];
        let mut t_grads = vec![0.0_f64; self.n_tri * 2];
        self.tri.eval_basis(&xi[1..3], &mut t_vals);
        self.tri.eval_grad_basis(&xi[1..3], &mut t_grads);
        let mut s_vals = vec![0.0_f64; self.order + 1];
        let mut ds_vals = vec![0.0_f64; self.order + 1];
        lag1d_on(&self.cp, xi[0], &mut s_vals);
        dlag1d_on(&self.cp, xi[0], &mut ds_vals);
        // `∂ξ` hits only the segment factor; `∂η, ∂ζ` only the triangle factor.
        for k in 0..=self.order {
            for t in 0..self.n_tri {
                let dof = self.dof_index(k, t);
                grads[dof * 3] = ds_vals[k] * t_vals[t];
                grads[dof * 3 + 1] = s_vals[k] * t_grads[t * 2];
                grads[dof * 3 + 2] = s_vals[k] * t_grads[t * 2 + 1];
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        prism_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let tri = self.tri.dof_coords();
        let mut coords = Vec::with_capacity(self.n_dofs());
        for k in 0..=self.order {
            for tc in tri.iter() {
                coords.push(vec![self.cp[k], tc[0], tc[1]]);
            }
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `H1PrismPk` p=2 slot positions must equal MFEM's
    /// `H1_WedgeElement::Nodes` (probe `tmp/a34_prism_h1_probe.cpp`, p=2:
    /// entity order `[6 v | 9 e | 3 q]`, quad-face slots at the side-quad
    /// centres).
    #[test]
    fn h1_prism_pk_p2_slots_are_mfems() {
        let fe = H1PrismPk::new(2);
        let rc = fe.dof_coords();
        assert_eq!(rc.len(), 18);
        let want: [[f64; 3]; 18] = [
            // (fem-rs convention: (ξ, η, ζ) = (MFEM z, MFEM x, MFEM y))
            // vertices
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
            // edges: bottom e01 e12 e20, top e01 e12 e20, verticals e03 e14 e25
            [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5],
            [1.0, 0.5, 0.0], [1.0, 0.5, 0.5], [1.0, 0.0, 0.5],
            [0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0],
            // quad faces (0,1,4,3), (1,2,5,4), (2,0,3,5) centres
            [0.5, 0.5, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.5],
        ];
        for (s, (got, want)) in rc.iter().zip(&want).enumerate() {
            for d in 0..3 {
                assert!(
                    (got[d] - want[d]).abs() < 1e-14,
                    "H1PrismPk p=2 slot {s}: {got:?} != {want:?}"
                );
            }
        }
        // ...and the slot kinds are MFEM's.
        let slots = h1_prism_slots(2);
        assert_eq!(slots[3], H1PrismSlot::Vertex(3));
        assert_eq!(slots[14], H1PrismSlot::Edge(8, 0));
        assert_eq!(slots[15], H1PrismSlot::QuadFace(2, 1, 1));
    }

    /// The basis is nodal at its own (GLL) nodes and has the partition of
    /// unity — the two properties every consumer of the H¹ wedge relies on.
    #[test]
    fn h1_prism_pk_basis_is_nodal_and_partition_of_unity() {
        for &p in &[1usize, 2, 3, 4] {
            let fe = H1PrismPk::new(p);
            let n = fe.n_dofs();
            let rc = fe.dof_coords();
            let mut vals = vec![0.0_f64; n];
            for (s, xi) in rc.iter().enumerate() {
                fe.eval_basis(xi, &mut vals);
                for (k, &v) in vals.iter().enumerate() {
                    let want: f64 = if k == s { 1.0 } else { 0.0 };
                    assert!(
                        (v - want).abs() < 1e-12,
                        "p={p}: basis {k} at dof {s}'s node = {v} (want {want})"
                    );
                }
            }
            for xi in [[0.3, 0.2, 0.1], [0.6, 0.15, 0.25], [0.11, 0.4, 0.05]] {
                fe.eval_basis(&xi, &mut vals);
                let sum: f64 = vals.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "p={p}: partition of unity at {xi:?} = {sum}"
                );
            }
            // Same functions as PrismPk, different slot order.
            let layer = PrismPk::new(p);
            let mut layer_vals = vec![0.0_f64; n];
            layer.eval_basis(&[0.3, 0.2, 0.1], &mut layer_vals);
            fe.eval_basis(&[0.3, 0.2, 0.1], &mut vals);
            let perm = fe.layer_perm();
            for (m, &src) in perm.iter().enumerate() {
                assert!((vals[m] - layer_vals[src]).abs() < 1e-13);
            }
        }
    }
}
