//! `RT_R2D_TriangleElement` / `RT_R2D_QuadrilateralElement` (MFEM 4.10
//! `fem/fe/fe_rt.cpp`) — 3-component H(div) fields on intrinsic 2-D meshes:
//! in-plane components on the regular Raviart–Thomas triangle/quad of order
//! `p`, the out-of-plane (z) component on the discontinuous L² space of
//! order `p` (INTEGRAL map).
//!
//! Physical transform (MFEM `RT_R2D_FiniteElement::CalcVShape(Trans, …)`):
//! the in-plane reference components transform with `J / Weight` (RT
//! contravariant Piola), the z component is untouched.

use super::{lagrange_1d, EmbeddedSlot, Jac2D};
use crate::lagrange::factory::TriL2GL;
use crate::quadrature::{gauss_legendre_01_arbitrary, gauss_lobatto_01_arbitrary};
use crate::raviart_thomas::TriRTk;
use crate::reference::{ReferenceElement, VectorReferenceElement};

/// z-part shape engine: L² triangle of order `p` (MFEM
/// `L2_TriangleElement(p, GaussLegendre)`), with the order-0 case handled
/// locally (a single constant shape at the barycenter — `TriL2GL::new`
/// serves `p >= 1`).
enum L2Tri {
    Const0,
    Pk(TriL2GL),
}

impl L2Tri {
    fn new(p: usize) -> Self {
        if p == 0 {
            L2Tri::Const0
        } else {
            L2Tri::Pk(TriL2GL::new(p))
        }
    }
    fn n_dofs(&self) -> usize {
        match self {
            L2Tri::Const0 => 1,
            L2Tri::Pk(e) => e.n_dofs(),
        }
    }
    fn eval(&self, xi: &[f64], out: &mut [f64]) {
        match self {
            L2Tri::Const0 => out[0] = 1.0,
            L2Tri::Pk(e) => e.eval_basis(xi, out),
        }
    }
    fn nodes(&self) -> Vec<[f64; 2]> {
        match self {
            L2Tri::Const0 => vec![[1.0 / 3.0; 2]],
            L2Tri::Pk(e) => e.dof_coords().iter().map(|p| [p[0], p[1]]).collect(),
        }
    }
}

// ─── Triangle ───────────────────────────────────────────────────────────────

/// MFEM `RT_R2D_TriangleElement(p)` — `((p+1)(3p+8))/2` dofs, from the
/// regular RT triangle `RT_FE(p)` (in-plane) and the L2 triangle `L2_FE(p)`
/// (z part), `nk_t = {0,-1,0, 1,1,0, -1,0,0, 0,0,1}`.
pub struct RtR2dTri {
    p: usize,
    rt: TriRTk,
    l2: L2Tri,
    dof_map: Vec<i32>,
    dof2nk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<[f64; 2]>,
}

impl RtR2dTri {
    pub fn new(p: usize) -> Self {
        let dof = ((p + 1) * (3 * p + 8)) / 2;
        let rt = TriRTk::new(p);
        let l2 = L2Tri::new(p);
        debug_assert_eq!(rt.n_dofs(), 3 * (p + 1) + p * (p + 1));
        debug_assert_eq!(l2.n_dofs(), (p + 1) * (p + 2) / 2);

        let mut dof_map = vec![0_i32; dof];
        let mut dof2nk = vec![0_u8; dof];
        let mut slots = vec![EmbeddedSlot::Interior; dof];
        let (mut o, mut r, mut l) = (0usize, 0usize, 0usize);
        // Three edges: p+1 in-plane (RT_FE) dofs each.
        for e in 0..3usize {
            for i in 0..=p {
                dof_map[o] = r as i32;
                r += 1;
                dof2nk[o] = e as u8;
                slots[o] = EmbeddedSlot::Edge(e, i);
                o += 1;
            }
        }
        // Interior dofs in the plane (RT_FE pairs: x-directed then z-tangent).
        if p >= 1 {
            for j in 0..p {
                for _i in 0..(p - j) {
                    dof_map[o] = r as i32;
                    r += 1;
                    dof2nk[o] = 0;
                    slots[o] = EmbeddedSlot::Interior;
                    o += 1;
                    dof_map[o] = r as i32;
                    r += 1;
                    dof2nk[o] = 2;
                    slots[o] = EmbeddedSlot::Interior;
                    o += 1;
                }
            }
        }
        // Interior z-directed dofs (L2_FE).
        for j in 0..=p {
            for _i in 0..=(p - j) {
                dof_map[o] = -1 - l as i32;
                l += 1;
                dof2nk[o] = 3;
                slots[o] = EmbeddedSlot::InteriorScalar;
                o += 1;
            }
        }
        debug_assert_eq!(r, rt.n_dofs(), "RT_R2D_Triangle incorrect number of RT dofs");
        debug_assert_eq!(l, l2.n_dofs(), "RT_R2D_Triangle incorrect number of L2 dofs");
        debug_assert_eq!(o, dof, "RT_R2D_Triangle incorrect number of dofs");

        let rt_nodes: Vec<[f64; 2]> =
            rt.dof_coords().iter().map(|p| [p[0], p[1]]).collect();
        let l2_nodes = l2.nodes();
        let mut nodes = vec![[0.0_f64; 2]; dof];
        for (i, st) in nodes.iter_mut().enumerate() {
            let idx = dof_map[i];
            *st = if idx >= 0 {
                rt_nodes[idx as usize]
            } else {
                l2_nodes[(-idx - 1) as usize]
            };
        }

        RtR2dTri { p, rt, l2, dof_map, dof2nk, slots, nodes }
    }

    /// RT index `p` (MFEM `RT_R2D_FECollection(p, 2)`; the collection's
    /// `GetOrder()` is `p + 1`).
    pub fn order(&self) -> usize {
        self.p
    }

    pub fn n_dofs(&self) -> usize {
        ((self.p + 1) * (3 * self.p + 8)) / 2
    }

    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }

    /// Reference normals per slot (`dof2nk`): 0/1/2 = in-plane edge normal
    /// index, 3 = z-directed.
    pub fn dof2nk(&self) -> &[u8] {
        &self.dof2nk
    }

    /// The `nk` table entry as a 2-D reference vector (the z axis is
    /// implicit: z-directed slots carry `n3(2) = 1`).
    pub fn normal(&self, nk: u8) -> [f64; 2] {
        const NK_T: [[f64; 2]; 3] = [[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]];
        NK_T[nk as usize]
    }

    pub fn nodes(&self) -> &[[f64; 2]] {
        &self.nodes
    }

    /// Reference `CalcVShape(ip, shape)` — `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, xi: &[f64], out: &mut [f64]) {
        let n = self.rt.n_dofs();
        let mut rt_shape = vec![0.0_f64; n * 2];
        let mut l2_shape = vec![0.0_f64; self.l2.n_dofs()];
        self.rt.eval_basis_vec(xi, &mut rt_shape);
        self.l2.eval(xi, &mut l2_shape);
        for k in 0..self.n_dofs() {
            let idx = self.dof_map[k];
            let (sx, sy, sz) = if idx >= 0 {
                (rt_shape[idx as usize * 2], rt_shape[idx as usize * 2 + 1], 0.0)
            } else {
                (0.0, 0.0, l2_shape[(-idx - 1) as usize])
            };
            out[k * 3] = sx;
            out[k * 3 + 1] = sy;
            out[k * 3 + 2] = sz;
        }
    }

    /// Reference `CalcDivShape(ip, div_shape)` — `n_dofs()`
    /// (z-directed slots have zero divergence).
    pub fn eval_div_ref(&self, xi: &[f64], out: &mut [f64]) {
        let n = self.rt.n_dofs();
        let mut rt_dshape = vec![0.0_f64; n];
        self.rt.eval_div(xi, &mut rt_dshape);
        for k in 0..self.n_dofs() {
            let idx = self.dof_map[k];
            out[k] = if idx >= 0 { rt_dshape[idx as usize] } else { 0.0 };
        }
    }

    /// Physical `CalcVShape(Trans, shape)` — in-plane columns through
    /// `J / Weight` (RT Piola).
    pub fn eval_vshape_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_vshape_ref(xi, out);
        // MFEM: columns 0/1 through `J`, then `shape *= 1 / Weight` — the whole
        // matrix, z column included.
        let w = 1.0 / jac.det;
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = (sx * jac.j00 + sy * jac.j01) * w;
            out[k * 3 + 1] = (sx * jac.j10 + sy * jac.j11) * w;
            out[k * 3 + 2] = sz * w;
        }
    }
}

// ─── Quadrilateral ──────────────────────────────────────────────────────────

/// MFEM `RT_R2D_QuadrilateralElement(p, GaussLobatto, GaussLegendre)` —
/// `(3p+5)(p+1)` dofs from the 1-D closed (order `p+1`) and open (order `p`)
/// bases, `nk_q = {0,-1,0, 1,0,0, 0,1,0, -1,0,0, 0,0,1}`.
pub struct RtR2dQuad {
    p: usize,
    dof_map: Vec<i32>,
    dof2nk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<[f64; 2]>,
    cp: Vec<f64>,
    op: Vec<f64>,
}

impl RtR2dQuad {
    pub fn new(p: usize) -> Self {
        let dof = (3 * p + 5) * (p + 1);
        let dofx = (p + 1) * (p + 2);
        let dofxy = 2 * dofx;
        let (cp, _) = gauss_lobatto_01_arbitrary(p + 2);
        let (op, _) = gauss_legendre_01_arbitrary(p + 1);

        let mut dof_map = vec![0_i32; dof];
        let mut o = 0usize;
        // edges (in-plane; the y-block bottom row / x-block side columns)
        for i in 0..=p {
            dof_map[dofx + i] = {
                let v = o as i32;
                o += 1;
                v
            }; // (0,1)
        }
        for i in 0..=p {
            dof_map[(p + 1) + i * (p + 2)] = {
                let v = o as i32;
                o += 1;
                v
            }; // (1,2)
        }
        for i in 0..=p {
            dof_map[dofx + (p - i) + (p + 1) * (p + 1)] = {
                let v = o as i32;
                o += 1;
                v
            }; // (2,3)
        }
        for i in 0..=p {
            dof_map[(p - i) * (p + 2)] = {
                let v = o as i32;
                o += 1;
                v
            }; // (3,0)
        }
        // interior
        for j in 0..=p {
            for i in 1..=p {
                dof_map[i + j * (p + 2)] = {
                    let v = o as i32;
                    o += 1;
                    v
                }; // x
            }
        }
        for j in 1..=p {
            for i in 0..=p {
                dof_map[dofx + i + j * (p + 1)] = {
                    let v = o as i32;
                    o += 1;
                    v
                }; // y
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                dof_map[dofxy + i + j * (p + 1)] = {
                    let v = o as i32;
                    o += 1;
                    v
                }; // z
            }
        }
        debug_assert_eq!(o, dof, "RT_R2D_Quad slot count mismatch");

        // dof orientations: x-components flip on the left half of each row
        // (plus the middle column of the top half when p is odd); the
        // y-components flip on the bottom rows (plus the middle row's left
        // half when p is odd).
        let flip = |m: &mut i32| {
            *m = -1 - *m;
        };
        for j in 0..=p {
            for i in 0..=(p / 2) {
                flip(&mut dof_map[i + j * (p + 2)]);
            }
        }
        if p % 2 == 1 {
            for j in (p / 2 + 1)..=p {
                flip(&mut dof_map[(p / 2 + 1) + j * (p + 2)]);
            }
        }
        for j in 0..=(p / 2) {
            for i in 0..=p {
                flip(&mut dof_map[dofx + i + j * (p + 1)]);
            }
        }
        if p % 2 == 1 {
            for i in 0..=(p / 2) {
                flip(&mut dof_map[dofx + i + (p / 2 + 1) * (p + 1)]);
            }
        }

        // Slot table (slot order = the `o` order above).
        let mut slots = Vec::with_capacity(dof);
        for e in 0..4usize {
            for i in 0..=p {
                slots.push(EmbeddedSlot::Edge(e, i));
            }
        }
        for _ in 0..(p * (p + 1)) {
            slots.push(EmbeddedSlot::Interior); // x
        }
        for _ in 0..(p * (p + 1)) {
            slots.push(EmbeddedSlot::Interior); // y
        }
        for _ in 0..((p + 1) * (p + 1)) {
            slots.push(EmbeddedSlot::InteriorScalar); // z
        }
        debug_assert_eq!(slots.len(), dof);

        // dof2nk + Nodes: walk the tensor positions.
        let mut dof2nk = vec![0_u8; dof];
        let mut nodes = vec![[0.0_f64; 2]; dof];
        let mut o = 0usize;
        for j in 0..=p {
            for i in 0..=(p + 1) {
                let raw = dof_map[o];
                let idx = if raw < 0 { (-raw - 1) as usize } else { raw as usize };
                dof2nk[idx] = if raw < 0 { 3 } else { 1 };
                nodes[idx] = [cp[i], op[j]];
                o += 1;
            }
        }
        for j in 0..=(p + 1) {
            for i in 0..=p {
                let raw = dof_map[o];
                let idx = if raw < 0 { (-raw - 1) as usize } else { raw as usize };
                dof2nk[idx] = if raw < 0 { 0 } else { 2 };
                nodes[idx] = [op[i], cp[j]];
                o += 1;
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                let idx = if dof_map[o] < 0 { (-dof_map[o] - 1) as usize } else { dof_map[o] as usize };
                dof2nk[idx] = 4;
                nodes[idx] = [op[i], op[j]];
                o += 1;
            }
        }
        debug_assert_eq!(o, dof);

        RtR2dQuad { p, dof_map, dof2nk, slots, nodes, cp, op }
    }

    /// RT index `p` (element order is `p + 1`).
    pub fn order(&self) -> usize {
        self.p
    }

    pub fn n_dofs(&self) -> usize {
        (3 * self.p + 5) * (self.p + 1)
    }

    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }

    pub fn dof2nk(&self) -> &[u8] {
        &self.dof2nk
    }

    /// The `nk_q` table entry as a 2-D reference vector (axes 0..3; the
    /// z axis carries `n3(2) = 1`).
    pub fn normal(&self, nk: u8) -> [f64; 2] {
        const NK_Q: [[f64; 2]; 4] =
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        NK_Q[nk as usize]
    }

    pub fn nodes(&self) -> &[[f64; 2]] {
        &self.nodes
    }

    /// Reference `CalcVShape(ip, shape)` — `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, xi: &[f64], out: &mut [f64]) {
        let pp1 = self.p + 1;
        let (cx, _dcx) = lagrange_1d(&self.cp, xi[0]);
        let (ox, _dox) = lagrange_1d(&self.op, xi[0]);
        let (cy, _dcy) = lagrange_1d(&self.cp, xi[1]);
        let (oy, _doy) = lagrange_1d(&self.op, xi[1]);
        let mut o = 0usize;
        for j in 0..pp1 {
            for i in 0..=pp1 {
                let (idx, s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx * 3] = s * cx[i] * oy[j];
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = 0.0;
            }
        }
        for j in 0..=pp1 {
            for i in 0..pp1 {
                let (idx, s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = s * ox[i] * cy[j];
                out[idx * 3 + 2] = 0.0;
            }
        }
        for j in 0..pp1 {
            for i in 0..pp1 {
                let (idx, _s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = ox[i] * oy[j];
            }
        }
    }

    /// Reference `CalcDivShape(ip, divshape)` — `n_dofs()`.
    pub fn eval_div_ref(&self, xi: &[f64], out: &mut [f64]) {
        let pp1 = self.p + 1;
        let (_cx, dcx) = lagrange_1d(&self.cp, xi[0]);
        let (ox, _dox) = lagrange_1d(&self.op, xi[0]);
        let (_cy, dcy) = lagrange_1d(&self.cp, xi[1]);
        let (oy, _doy) = lagrange_1d(&self.op, xi[1]);
        let mut o = 0usize;
        for j in 0..pp1 {
            for i in 0..=pp1 {
                let (idx, s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx] = s * dcx[i] * oy[j];
            }
        }
        for j in 0..=pp1 {
            for i in 0..pp1 {
                let (idx, s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx] = s * ox[i] * dcy[j];
            }
        }
        for _j in 0..pp1 {
            for _i in 0..pp1 {
                let (idx, _s) = resolve(self.dof_map[o]);
                o += 1;
                out[idx] = 0.0;
            }
        }
    }

    /// Physical `CalcVShape(Trans, shape)` — in-plane columns through
    /// `J / Weight`.
    pub fn eval_vshape_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_vshape_ref(xi, out);
        // MFEM: columns 0/1 through `J`, then `shape *= 1 / Weight` — the whole
        // matrix, z column included.
        let w = 1.0 / jac.det;
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = (sx * jac.j00 + sy * jac.j01) * w;
            out[k * 3 + 1] = (sx * jac.j10 + sy * jac.j11) * w;
            out[k * 3 + 2] = sz * w;
        }
    }
}

/// Resolve a `dof_map` entry: `(slot, sign)` with the `-1-idx` encoding.
#[inline]
fn resolve(raw: i32) -> (usize, f64) {
    if raw < 0 {
        ((-raw - 1) as usize, -1.0)
    } else {
        (raw as usize, 1.0)
    }
}
