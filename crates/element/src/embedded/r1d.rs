//! Segment / point variants of the embedded (restricted) families — MFEM 4.10
//! `ND_R2D_SegmentElement` (the `dim == 1` arm of the ND_R2D collection) and
//! `ND_R1D_SegmentElement` / `RT_R1D_SegmentElement` / `ND_R1D_PointElement`
//! (the `ND_R1D` / `RT_R1D` collections on intrinsic 1-D meshes).
//! Reference segment `[0,1]`.
//!
//! Component conventions (1:1 with the MFEM element `vdim`):
//!
//! | type | vdim | components |
//! |---|---|---|
//! | [`NdR2dSegment`] | 2 | in-plane x (ND), z (H1-type) |
//! | [`NdR1dSegment`] | 3 | x (ND), y (H1-type), z (H1-type) |
//! | [`RtR1dSegment`] | 3 | x normal (RT), y (L2-type), z (L2-type) |
//! | [`NdR1dPoint`]   | 2 | y, z point dofs |
//!
//! MFEM's `RT_R2D_SegmentElement` is deliberately **not** ported: it is not a
//! member of any collection (`RT_R2D_FECollection(p, 1)` uses the plain
//! INTEGRAL-map `L2_SegmentElement`; see the note in the RT section below).

use super::{lagrange_1d, EmbeddedSlot};
use crate::quadrature::{gauss_legendre_01_arbitrary, gauss_lobatto_01_arbitrary};

// ─── ND families ────────────────────────────────────────────────────────────

/// MFEM `ND_R2D_SegmentElement(p)` — `2p+1` dofs, `vdim = 2`, 1 dof per
/// vertex (z-directed), the in-plane ND part + z H1 part on the cell.
pub struct NdR2dSegment {
    p: usize,
    dof_map: Vec<i32>,
    dof2tk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<f64>,
    cp: Vec<f64>,
    op: Vec<f64>,
}

impl NdR2dSegment {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "NdR2dSegment requires order >= 1 (MFEM VERIFY)");
        let dof = 2 * p + 1;
        let (cp, _) = gauss_lobatto_01_arbitrary(p + 1);
        let (op, _) = gauss_legendre_01_arbitrary(p);
        let mut dof_map = vec![0_i32; dof];
        let mut dof2tk = vec![0_u8; dof];
        let mut slots = vec![EmbeddedSlot::Interior; dof];
        let mut nodes = vec![0.0_f64; dof];
        let mut o = 0usize;
        // (0): z-directed vertex dof
        dof_map[p] = o as i32;
        dof2tk[o] = 1;
        slots[o] = EmbeddedSlot::Vertex(0, 0);
        nodes[o] = cp[0];
        o += 1;
        // (1): z-directed vertex dof
        dof_map[2 * p] = o as i32;
        dof2tk[o] = 1;
        slots[o] = EmbeddedSlot::Vertex(1, 0);
        nodes[o] = cp[p];
        o += 1;
        // interior x-components
        for i in 0..p {
            dof_map[i] = o as i32;
            dof2tk[o] = 0;
            nodes[o] = op[i];
            o += 1;
        }
        // interior z-components
        for i in 1..p {
            dof_map[p + i] = o as i32;
            dof2tk[o] = 1;
            nodes[o] = cp[i];
            o += 1;
        }
        debug_assert_eq!(o, dof, "ND_R2D_Segment slot count mismatch");
        NdR2dSegment { p, dof_map, dof2tk, slots, nodes, cp, op }
    }

    pub fn order(&self) -> usize {
        self.p
    }
    pub fn n_dofs(&self) -> usize {
        2 * self.p + 1
    }
    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }
    pub fn dof2tk(&self) -> &[u8] {
        &self.dof2tk
    }
    /// dof sites along the reference segment (MFEM `FE::Nodes`).
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Reference `CalcVShape` — `out` length `n_dofs() * 2`
    /// (component 0 = in-plane, component 1 = z).
    pub fn eval_vshape_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (cx, _dcx) = lagrange_1d(&self.cp, x);
        let (ox, _dox) = lagrange_1d(&self.op, x);
        for i in 0..p {
            let idx = self.dof_map[i] as usize;
            out[idx * 2] = ox[i];
            out[idx * 2 + 1] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[p + i] as usize;
            out[idx * 2] = 0.0;
            out[idx * 2 + 1] = cx[i];
        }
    }

    /// Reference `CalcCurlShape` — `n_dofs()` (scalar, `cdim = 1`).
    pub fn eval_curl_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (_cx, dcx) = lagrange_1d(&self.cp, x);
        for i in 0..=p {
            let idx = self.dof_map[p + i] as usize;
            out[idx] = -dcx[i];
        }
        for i in 0..p {
            let idx = self.dof_map[i] as usize;
            out[idx] = 0.0;
        }
    }

    /// Physical `CalcVShape(Trans, shape)`: the in-plane component through
    /// `J⁻¹(0,0)` (MFEM multiplies column 0 only).
    pub fn eval_vshape_phys(&self, x: f64, j_inv_00: f64, out: &mut [f64]) {
        self.eval_vshape_ref(x, out);
        for k in 0..self.n_dofs() {
            out[k * 2] *= j_inv_00;
        }
    }
}

/// MFEM `ND_R1D_SegmentElement(p)` — `3p+2` dofs, `vdim = 3` (x ND component
/// + y/z H1-type components), 2 dofs per vertex.
pub struct NdR1dSegment {
    p: usize,
    dof_map: Vec<i32>,
    dof2tk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<f64>,
    cp: Vec<f64>,
    op: Vec<f64>,
}

impl NdR1dSegment {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "NdR1dSegment requires order >= 1 (MFEM VERIFY)");
        let dof = 3 * p + 2;
        let (cp, _) = gauss_lobatto_01_arbitrary(p + 1);
        let (op, _) = gauss_legendre_01_arbitrary(p);
        let mut dof_map = vec![0_i32; dof];
        let mut dof2tk = vec![0_u8; dof];
        let mut slots = vec![EmbeddedSlot::Interior; dof];
        let mut nodes = vec![0.0_f64; dof];
        let mut o = 0usize;
        // (0): y- and z-directed vertex dofs
        dof_map[p] = o as i32;
        dof2tk[o] = 1;
        slots[o] = EmbeddedSlot::Vertex(0, 0);
        nodes[o] = cp[0];
        o += 1;
        dof_map[2 * p + 1] = o as i32;
        dof2tk[o] = 2;
        slots[o] = EmbeddedSlot::Vertex(0, 1);
        nodes[o] = cp[0];
        o += 1;
        // (1): y- and z-directed vertex dofs
        dof_map[2 * p] = o as i32;
        dof2tk[o] = 1;
        slots[o] = EmbeddedSlot::Vertex(1, 0);
        nodes[o] = cp[p];
        o += 1;
        dof_map[3 * p + 1] = o as i32;
        dof2tk[o] = 2;
        slots[o] = EmbeddedSlot::Vertex(1, 1);
        nodes[o] = cp[p];
        o += 1;
        // interior x-components
        for i in 0..p {
            dof_map[i] = o as i32;
            dof2tk[o] = 0;
            nodes[o] = op[i];
            o += 1;
        }
        // interior y-components
        for i in 1..p {
            dof_map[p + i] = o as i32;
            dof2tk[o] = 1;
            nodes[o] = cp[i];
            o += 1;
        }
        // interior z-components
        for i in 1..p {
            dof_map[2 * p + 1 + i] = o as i32;
            dof2tk[o] = 2;
            nodes[o] = cp[i];
            o += 1;
        }
        debug_assert_eq!(o, dof, "ND_R1D_Segment slot count mismatch");
        NdR1dSegment { p, dof_map, dof2tk, slots, nodes, cp, op }
    }

    pub fn order(&self) -> usize {
        self.p
    }
    pub fn n_dofs(&self) -> usize {
        3 * self.p + 2
    }
    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }
    pub fn dof2tk(&self) -> &[u8] {
        &self.dof2tk
    }
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Reference `CalcVShape` — `out` length `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (cx, _dcx) = lagrange_1d(&self.cp, x);
        let (ox, _dox) = lagrange_1d(&self.op, x);
        for i in 0..p {
            let idx = self.dof_map[i] as usize;
            out[idx * 3] = ox[i];
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[p + i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = cx[i];
            out[idx * 3 + 2] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[2 * p + 1 + i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = cx[i];
        }
    }

    /// Reference `CalcCurlShape` — `n_dofs() * 3` (`cdim = 3`).
    pub fn eval_curl_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (_cx, dcx) = lagrange_1d(&self.cp, x);
        for i in 0..p {
            let idx = self.dof_map[i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[p + i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = dcx[i];
        }
        for i in 0..=p {
            let idx = self.dof_map[2 * p + 1 + i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = -dcx[i];
            out[idx * 3 + 2] = 0.0;
        }
    }

    /// Physical `CalcVShape(Trans, shape)`: the x component through
    /// `J⁻¹(0,0)`.
    pub fn eval_vshape_phys(&self, x: f64, j_inv_00: f64, out: &mut [f64]) {
        self.eval_vshape_ref(x, out);
        for k in 0..self.n_dofs() {
            out[k * 3] *= j_inv_00;
        }
    }
}

/// MFEM `ND_R1D_PointElement(p)` — 2 dofs (y, z) at the vertex, `vdim = 2`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NdR1dPoint;

impl NdR1dPoint {
    pub fn n_dofs(&self) -> usize {
        2
    }
    /// Reference `CalcVShape` — identity `[[1, 0], [0, 1]]`.
    pub fn eval_vshape_ref(&self, _x: f64, out: &mut [f64]) {
        out[0] = 1.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 1.0;
    }
}

// ─── RT families ────────────────────────────────────────────────────────────
//
// Note (d102): MFEM's `RT_R2D_SegmentElement` (`fe_rt.cpp`) is **not** wired
// into any collection — `RT_R2D_FECollection(p, 1)` uses the plain
// INTEGRAL-map `L2_SegmentElement(p, ob_type)` as its cell element, and the
// `RT_R2D_Trace_FECollection` trace resolves to the same L2 segment.  The
// upstream class is dead code with an internally inconsistent `order = p+1`
// loop bound, so it is deliberately not ported here (dead-code zero
// tolerance).  The `dim == 1` arm of the RT_R2D collection therefore only
// needs the L2 segment, which the space layer takes from
// `lagrange::factory`'s segment L2 family.

/// MFEM `RT_R1D_SegmentElement(p)` — `3p+4` dofs, `vdim = 3` (x normal RT
/// component + y/z L2-type components), 1 vertex dof (x at the endpoints).
/// Cell element of `RT_R1D_FECollection(p, 1)`.
pub struct RtR1dSegment {
    p: usize,
    dof_map: Vec<i32>,
    dof2nk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<f64>,
    cp: Vec<f64>,
    op: Vec<f64>,
}

impl RtR1dSegment {
    pub fn new(p: usize) -> Self {
        let dof = 3 * p + 4;
        let (cp, _) = gauss_lobatto_01_arbitrary(p + 2);
        let (op, _) = gauss_legendre_01_arbitrary(p + 1);
        let mut dof_map = vec![0_i32; dof];
        let mut dof2nk = vec![0_u8; dof];
        let mut slots = vec![EmbeddedSlot::Interior; dof];
        let mut nodes = vec![0.0_f64; dof];
        let mut o = 0usize;
        // (0): x-directed vertex dof
        dof_map[0] = o as i32;
        dof2nk[o] = 0;
        slots[o] = EmbeddedSlot::Vertex(0, 0);
        nodes[o] = cp[0];
        o += 1;
        // (1): x-directed vertex dof
        dof_map[p + 1] = o as i32;
        dof2nk[o] = 0;
        slots[o] = EmbeddedSlot::Vertex(1, 0);
        nodes[o] = cp[p + 1];
        o += 1;
        // interior x-components
        for i in 1..=p {
            dof_map[i] = o as i32;
            dof2nk[o] = 0;
            nodes[o] = cp[i];
            o += 1;
        }
        // interior y-components
        for i in 0..=p {
            dof_map[p + i + 2] = o as i32;
            dof2nk[o] = 1;
            nodes[o] = op[i];
            o += 1;
        }
        // interior z-components
        for i in 0..=p {
            dof_map[2 * p + 3 + i] = o as i32;
            dof2nk[o] = 2;
            nodes[o] = op[i];
            o += 1;
        }
        debug_assert_eq!(o, dof, "RT_R1D_Segment slot count mismatch");
        RtR1dSegment { p, dof_map, dof2nk, slots, nodes, cp, op }
    }

    /// RT index `p` (element order `p + 1`).
    pub fn order(&self) -> usize {
        self.p
    }
    pub fn n_dofs(&self) -> usize {
        3 * self.p + 4
    }
    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }
    pub fn dof2nk(&self) -> &[u8] {
        &self.dof2nk
    }
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Reference `CalcVShape` — `out` length `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (cx, _dcx) = lagrange_1d(&self.cp, x);
        let (ox, _dox) = lagrange_1d(&self.op, x);
        for i in 0..=p + 1 {
            let idx = self.dof_map[i] as usize;
            out[idx * 3] = cx[i];
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[p + i + 2] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = ox[i];
            out[idx * 3 + 2] = 0.0;
        }
        for i in 0..=p {
            let idx = self.dof_map[2 * p + 3 + i] as usize;
            out[idx * 3] = 0.0;
            out[idx * 3 + 1] = 0.0;
            out[idx * 3 + 2] = ox[i];
        }
    }

    /// Reference `CalcDivShape` — `n_dofs()` (x-component derivative only).
    pub fn eval_div_ref(&self, x: f64, out: &mut [f64]) {
        let p = self.p;
        let (_cx, dcx) = lagrange_1d(&self.cp, x);
        for i in 0..=p + 1 {
            let idx = self.dof_map[i] as usize;
            out[idx] = dcx[i];
        }
        for i in 0..=p {
            let idx = self.dof_map[p + i + 2] as usize;
            out[idx] = 0.0;
            let idx = self.dof_map[2 * p + 3 + i] as usize;
            out[idx] = 0.0;
        }
    }

    /// Physical `CalcVShape(Trans, shape)`: the x component through
    /// `J(0,0)/Weight` (= 1 for a 1-D transform; kept for 1:1 form).
    pub fn eval_vshape_phys(&self, x: f64, j00_over_w: f64, out: &mut [f64]) {
        self.eval_vshape_ref(x, out);
        for k in 0..self.n_dofs() {
            out[k * 3] *= j00_over_w;
        }
    }
}
