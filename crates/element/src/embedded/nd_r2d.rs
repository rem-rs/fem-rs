//! `ND_R2D_TriangleElement` / `ND_R2D_QuadrilateralElement` (MFEM 4.10
//! `fem/fe/fe_nd.cpp`) — 3-component H(curl) fields on intrinsic 2-D meshes:
//! in-plane components on the regular Nédélec triangle/quad of order `p`,
//! the out-of-plane (z) component on the continuous H¹ space of order `p`.
//!
//! Physical transforms (MFEM `CalcVShape(Trans, …)` / `CalcPhysCurlShape`):
//! the in-plane reference components transform with `J⁻¹` (ND covariant),
//! the z component is untouched; the physical curl of the in-plane part is
//! `curl_ref · J / Weight`, the z part's is the plain `∇z` rotated by `J⁻¹`
//! (both folded into the same 3-component output exactly as MFEM does).

use super::{lagrange_1d, EmbeddedSlot, Jac2D};
use crate::lagrange::factory::H1TriPk;
use crate::nedelec::TriNDk;
use crate::quadrature::{gauss_legendre_01_arbitrary, gauss_lobatto_01_arbitrary};
use crate::reference::{ReferenceElement, VectorReferenceElement};

// ─── Triangle ───────────────────────────────────────────────────────────────

/// In-plane ND engine of [`NdR2dTri`].  `p >= 2` reuses the crate's
/// `TriNDk` (the same u-basis + interpolation-matrix construction as MFEM's
/// `ND_TriangleElement`); `p == 1` is built here with MFEM's exact
/// u-functions (constant x, constant y, rotation about the centroid) — the
/// crate's `TriNDk` uses the Whitney basis for `k = 1`, which spans the same
/// space but is *not* the MFEM basis (the delta functionals differ), so the
/// 1:1 port cannot delegate the `p == 1` case.
enum NdTriEngine {
    P1 { ti: [[f64; 3]; 3] },
    Pk(TriNDk),
}

impl NdTriEngine {
    fn n_dofs(&self) -> usize {
        match self {
            NdTriEngine::P1 { .. } => 3,
            NdTriEngine::Pk(e) => e.n_dofs(),
        }
    }
    fn eval_basis_vec(&self, xi: &[f64], out: &mut [f64]) {
        match self {
            NdTriEngine::Pk(e) => e.eval_basis_vec(xi, out),
            NdTriEngine::P1 { ti } => {
                // u = [(1,0), (0,1), (y-c, -(x-c))], c = 1/3.
                let (x, y) = (xi[0], xi[1]);
                const C: f64 = 1.0 / 3.0;
                let u = [[1.0, 0.0], [0.0, 1.0], [y - C, -(x - C)]];
                for (j, row) in ti.iter().enumerate() {
                    let mut vx = 0.0;
                    let mut vy = 0.0;
                    for (m, um) in u.iter().enumerate() {
                        vx += row[m] * um[0];
                        vy += row[m] * um[1];
                    }
                    out[j * 2] = vx;
                    out[j * 2 + 1] = vy;
                }
            }
        }
    }
    fn eval_curl(&self, xi: &[f64], out: &mut [f64]) {
        match self {
            NdTriEngine::Pk(e) => e.eval_curl(xi, out),
            NdTriEngine::P1 { ti } => {
                // curl(u_0) = curl(u_1) = 0; curl(u_2) = ∂_x(-(x-c)) - ∂_y(y-c) = -2.
                let curlu = [0.0, 0.0, -2.0];
                for (j, row) in ti.iter().enumerate() {
                    let mut s = 0.0;
                    for (m, &cu) in curlu.iter().enumerate() {
                        s += row[m] * cu;
                    }
                    out[j] = s;
                }
            }
        }
    }
}

/// 3×3 matrix inverse by Gauss-Jordan with partial pivoting
/// (returns `None` for a singular matrix).
fn inv3(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut m = a;
    let mut inv = [
        [1.0_f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for col in 0..3 {
        let (mut best, mut best_v) = (col, m[col][col].abs());
        for r in col + 1..3 {
            if m[r][col].abs() > best_v {
                best = r;
                best_v = m[r][col].abs();
            }
        }
        if best_v < 1e-300 {
            return None;
        }
        m.swap(col, best);
        inv.swap(col, best);
        let piv = m[col][col];
        for j in 0..3 {
            m[col][j] /= piv;
            inv[col][j] /= piv;
        }
        for r in 0..3 {
            if r != col {
                let f = m[r][col];
                for j in 0..3 {
                    m[r][j] -= f * m[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

/// MFEM `ND_R2D_TriangleElement(p)` — `((3p+1)(p+2))/2` dofs, built from the
/// regular ND triangle `ND_FE(p)` (in-plane) and the H1 triangle `H1_FE(p)`
/// (z part), `tk_t = {1,0,0, -1,1,0, 0,-1,0, 0,1,0, 0,0,1}`.
pub struct NdR2dTri {
    p: usize,
    nd: NdTriEngine,
    h1: H1TriPk,
    dof_map: Vec<i32>,
    dof2tk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<[f64; 2]>,
}

impl NdR2dTri {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "NdR2dTri requires order >= 1 (MFEM VERIFY)");
        let dof = ((3 * p + 1) * (p + 2)) / 2;
        let nd = if p == 1 {
            // MFEM ND_TriangleElement(1): nodes (0.5,0), (0.5,0.5), (0,0.5)
            // with tangents tk = (1,0), (-1,1), (0,-1); u-functions
            // [(1,0), (0,1), (y-c, -(x-c))]; T[k][m] = functional_k(u_m),
            // shapes = T⁻¹ · u.
            const C: f64 = 1.0 / 3.0;
            let nd_nodes: [[f64; 2]; 3] = [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
            let tk: [[f64; 2]; 3] = [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]];
            let mut t = [[0.0_f64; 3]; 3];
            for (k, (xk, tkk)) in nd_nodes.iter().zip(tk.iter()).enumerate() {
                let (x, y) = (xk[0], xk[1]);
                let us: [[f64; 2]; 3] = [[1.0, 0.0], [0.0, 1.0], [y - C, -(x - C)]];
                for (m, um) in us.iter().enumerate() {
                    t[k][m] = um[0] * tkk[0] + um[1] * tkk[1];
                }
            }
            let ti = inv3(t).expect("ND_R2D_Tri p=1: singular interpolation matrix");
            // MFEM's shapes = Ti · u with Ti = (Tᵀ)⁻¹ so that
            // functional_k(φ_j) = Σ_n Ti[j][n]·T[k][n] = (T·Tiᵀ)[k][j] = δ.
            let ti_tr = [
                [ti[0][0], ti[1][0], ti[2][0]],
                [ti[0][1], ti[1][1], ti[2][1]],
                [ti[0][2], ti[1][2], ti[2][2]],
            ];
            NdTriEngine::P1 { ti: ti_tr }
        } else {
            NdTriEngine::Pk(TriNDk::new(p))
        };
        let h1 = H1TriPk::new(p);
        debug_assert_eq!(nd.n_dofs(), p * (p + 2));
        debug_assert_eq!(h1.n_dofs(), (p + 1) * (p + 2) / 2);

        let mut dof_map = vec![0_i32; dof];
        let mut dof2tk = vec![0_u8; dof];
        let mut slots = vec![EmbeddedSlot::Interior; dof];
        let (mut o, mut n, mut h) = (0usize, 0usize, 0usize);

        // Three nodes (z-directed H1 dofs at the vertices).
        for v in 0..3 {
            dof_map[o] = -1 - h as i32;
            h += 1;
            dof2tk[o] = 4;
            slots[o] = EmbeddedSlot::Vertex(v, 0);
            o += 1;
        }
        // Three edges: p in-plane (ND_FE) dofs then p-1 z-directed (H1) dofs.
        for e in 0..3usize {
            for i in 0..p {
                dof_map[o] = n as i32;
                n += 1;
                dof2tk[o] = e as u8;
                slots[o] = EmbeddedSlot::Edge(e, i);
                o += 1;
            }
            for i in 0..p.saturating_sub(1) {
                dof_map[o] = -1 - h as i32;
                h += 1;
                dof2tk[o] = 4;
                slots[o] = EmbeddedSlot::EdgeScalar(e, i);
                o += 1;
            }
        }
        // Interior dofs in the plane (ND_FE pairs: x-directed then y-directed).
        if p >= 2 {
            for j in 0..=(p - 2) {
                for _i in 0..=(p - 2 - j) {
                    dof_map[o] = n as i32;
                    n += 1;
                    dof2tk[o] = 0;
                    slots[o] = EmbeddedSlot::Interior;
                    o += 1;
                    dof_map[o] = n as i32;
                    n += 1;
                    dof2tk[o] = 3;
                    slots[o] = EmbeddedSlot::Interior;
                    o += 1;
                }
            }
        }
        // Interior z-directed dofs (H1).
        if p >= 2 {
            for j in 0..(p - 1) {
                for _i in 0..(p - 2 - j) {
                    dof_map[o] = -1 - h as i32;
                    h += 1;
                    dof2tk[o] = 4;
                    slots[o] = EmbeddedSlot::InteriorScalar;
                    o += 1;
                }
            }
        }
        debug_assert_eq!(n, nd.n_dofs(), "ND_R2D_Triangle incorrect number of ND dofs");
        debug_assert_eq!(h, h1.n_dofs(), "ND_R2D_Triangle incorrect number of H1 dofs");
        debug_assert_eq!(o, dof, "ND_R2D_Triangle incorrect number of dofs");

        // Nodes: ND_FE sites for in-plane slots, H1_FE sites for z slots.
        let nd_nodes: Vec<[f64; 2]> = match &nd {
            NdTriEngine::P1 { .. } => vec![[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
            NdTriEngine::Pk(e) => e.dof_coords().iter().map(|p| [p[0], p[1]]).collect(),
        };
        let h1_nodes: Vec<[f64; 2]> =
            h1.dof_coords().iter().map(|p| [p[0], p[1]]).collect();
        let mut nodes = vec![[0.0_f64; 2]; dof];
        for (i, nd_st) in nodes.iter_mut().enumerate() {
            let idx = dof_map[i];
            *nd_st = if idx >= 0 {
                nd_nodes[idx as usize]
            } else {
                h1_nodes[(-idx - 1) as usize]
            };
        }

        NdR2dTri { p, nd, h1, dof_map, dof2tk, slots, nodes }
    }

    pub fn order(&self) -> usize {
        self.p
    }

    pub fn n_dofs(&self) -> usize {
        ((3 * self.p + 1) * (self.p + 2)) / 2
    }

    /// Element-local slot → entity association (MFEM dof table semantics).
    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }

    /// Reference tangents per slot (`dof2tk`): 0/1/2/3 = in-plane tangent
    /// index, 4 = z-directed.
    pub fn dof2tk(&self) -> &[u8] {
        &self.dof2tk
    }

    /// The `tk` table entry `tk` as a 2-D reference vector (the z axis is
    /// implicit: `dof2tk == 4` carries `t3(2) = 1` in the MFEM formulas).
    pub fn tangent(&self, tk: u8) -> [f64; 2] {
        const TK_T: [[f64; 2]; 4] =
            [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [0.0, 1.0]];
        TK_T[tk as usize]
    }

    /// dof sites (MFEM `FE::Nodes`) on the reference triangle.
    pub fn nodes(&self) -> &[[f64; 2]] {
        &self.nodes
    }

    /// Reference-domain `CalcVShape(ip, shape)` — `out` length `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, xi: &[f64], out: &mut [f64]) {
        let n = self.nd.n_dofs();
        let mut nd_shape = vec![0.0_f64; n * 2];
        let mut h1_shape = vec![0.0_f64; self.h1.n_dofs()];
        self.nd.eval_basis_vec(xi, &mut nd_shape);
        self.h1.eval_basis(xi, &mut h1_shape);
        for k in 0..self.n_dofs() {
            let idx = self.dof_map[k];
            let (sx, sy, sz) = if idx >= 0 {
                (nd_shape[idx as usize * 2], nd_shape[idx as usize * 2 + 1], 0.0)
            } else {
                (0.0, 0.0, h1_shape[(-idx - 1) as usize])
            };
            out[k * 3] = sx;
            out[k * 3 + 1] = sy;
            out[k * 3 + 2] = sz;
        }
    }

    /// Reference-domain `CalcCurlShape(ip, curl_shape)` — `n_dofs() * 3`.
    pub fn eval_curl_ref(&self, xi: &[f64], out: &mut [f64]) {
        let n = self.nd.n_dofs();
        let mut nd_dshape = vec![0.0_f64; n];
        let mut h1_dshape = vec![0.0_f64; self.h1.n_dofs() * 2];
        self.nd.eval_curl(xi, &mut nd_dshape);
        self.h1.eval_grad_basis(xi, &mut h1_dshape);
        for k in 0..self.n_dofs() {
            let idx = self.dof_map[k];
            let (cx, cy, cz) = if idx >= 0 {
                (0.0, 0.0, nd_dshape[idx as usize])
            } else {
                let j = (-idx - 1) as usize;
                (h1_dshape[j * 2 + 1], -h1_dshape[j * 2], 0.0)
            };
            out[k * 3] = cx;
            out[k * 3 + 1] = cy;
            out[k * 3 + 2] = cz;
        }
    }

    /// Physical `CalcVShape(Trans, shape)` — in-plane columns through `J⁻¹`.
    pub fn eval_vshape_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_vshape_ref(xi, out);
        let ji = jac.inv();
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = sx * ji[0] + sy * ji[2];
            out[k * 3 + 1] = sx * ji[1] + sy * ji[3];
            out[k * 3 + 2] = sz;
        }
    }

    /// Physical `CalcPhysCurlShape(Trans, curl_shape)` — `n_dofs() * 3`.
    /// MFEM scales columns 0/1 with `J` and then the whole matrix with
    /// `1 / Weight` (`curl_shape *= 1.0 / Trans.Weight()`), i.e. the z
    /// column is scaled too.
    pub fn eval_curl_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_curl_ref(xi, out);
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = (sx * jac.j00 + sy * jac.j01) / jac.det;
            out[k * 3 + 1] = (sx * jac.j10 + sy * jac.j11) / jac.det;
            out[k * 3 + 2] = sz / jac.det;
        }
    }
}

// ─── Quadrilateral ──────────────────────────────────────────────────────────

/// MFEM `ND_R2D_QuadrilateralElement(p, GaussLobatto, GaussLegendre)` —
/// `(3p+1)(p+1)` dofs built directly from the 1-D closed (order `p`,
/// Gauss–Lobatto) and open (order `p−1`, Gauss–Legendre) bases,
/// `tk_q = {1,0,0, 0,1,0, -1,0,0, 0,-1,0, 0,0,1}`.
pub struct NdR2dQuad {
    p: usize,
    dof_map: Vec<i32>,
    dof2tk: Vec<u8>,
    slots: Vec<EmbeddedSlot>,
    nodes: Vec<[f64; 2]>,
    cp: Vec<f64>,
    op: Vec<f64>,
}

impl NdR2dQuad {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "NdR2dQuad requires order >= 1 (MFEM VERIFY)");
        let dof = (3 * p + 1) * (p + 1);
        let dofx = p * (p + 1);
        let dofxy = 2 * dofx;
        let (cp, _) = gauss_lobatto_01_arbitrary(p + 1);
        let (op, _) = gauss_legendre_01_arbitrary(p);

        let mut dof_map = vec![0_i32; dof];
        let mut o = 0usize;
        fn take(o: &mut usize) -> i32 {
            let v = *o as i32;
            *o += 1;
            v
        }
        // nodes (z-directed H1 dofs at the four vertices)
        dof_map[dofxy] = take(&mut o); // (0)
        dof_map[dofxy + p] = take(&mut o); // (1)
        dof_map[dof - 1] = take(&mut o); // (2)
        dof_map[dof - p - 1] = take(&mut o); // (3)
        // edges: in-plane blocks carry the negative `-1-slot` encoding on the
        // two edges whose local direction opposes the tensor direction.
        for i in 0..p {
            dof_map[i] = take(&mut o); // (0,1) x-directed
        }
        for i in 1..p {
            dof_map[dofxy + i] = take(&mut o); // (0,1) z-directed
        }
        for j in 0..p {
            dof_map[dofx + p + j * (p + 1)] = take(&mut o); // (1,2) y-directed
        }
        for j in 1..p {
            dof_map[dofxy + p + j * (p + 1)] = take(&mut o); // (1,2) z-directed
        }
        for i in 0..p {
            dof_map[(p - 1 - i) + p * p] = -1 - take(&mut o); // (2,3) x (flipped)
        }
        for i in 1..p {
            dof_map[dofxy + (p - i) + p * (p + 1)] = take(&mut o); // (2,3) z-directed
        }
        for j in 0..p {
            dof_map[dofx + (p - 1 - j) * (p + 1)] = -1 - take(&mut o); // (3,0) y (flipped)
        }
        for j in 1..p {
            dof_map[dofxy + (p - j) * (p + 1)] = take(&mut o); // (3,0) z-directed
        }
        // interior
        for j in 1..p {
            for i in 0..p {
                dof_map[i + j * p] = take(&mut o); // x
            }
        }
        for j in 0..p {
            for i in 1..p {
                dof_map[dofx + i + j * (p + 1)] = take(&mut o); // y
            }
        }
        for j in 1..p {
            for i in 1..p {
                dof_map[dofxy + i + j * (p + 1)] = take(&mut o); // z
            }
        }
        debug_assert_eq!(o, dof, "ND_R2D_Quad slot count mismatch");

        // Slot table (slot order = the `o` order above) and dof2tk.
        let mut slots = Vec::with_capacity(dof);
        for v in 0..4 {
            slots.push(EmbeddedSlot::Vertex(v, 0));
        }
        for i in 0..p {
            slots.push(EmbeddedSlot::Edge(0, i));
        }
        for i in 0..p - 1 {
            slots.push(EmbeddedSlot::EdgeScalar(0, i));
        }
        for j in 0..p {
            slots.push(EmbeddedSlot::Edge(1, j));
        }
        for j in 0..p - 1 {
            slots.push(EmbeddedSlot::EdgeScalar(1, j));
        }
        for i in 0..p {
            slots.push(EmbeddedSlot::Edge(2, i));
        }
        for i in 0..p - 1 {
            slots.push(EmbeddedSlot::EdgeScalar(2, i));
        }
        for j in 0..p {
            slots.push(EmbeddedSlot::Edge(3, j));
        }
        for j in 0..p - 1 {
            slots.push(EmbeddedSlot::EdgeScalar(3, j));
        }
        for _ in 0..(2 * p * (p - 1)) {
            slots.push(EmbeddedSlot::Interior);
        }
        for _ in 0..((p - 1) * (p - 1)) {
            slots.push(EmbeddedSlot::InteriorScalar);
        }
        debug_assert_eq!(slots.len(), dof);

        // dof2tk + Nodes: walk the tensor positions (the CalcVShape order),
        // resolving each slot through dof_map (negatives flip the sign).
        let mut dof2tk = vec![0_u8; dof];
        let mut nodes = vec![[0.0_f64; 2]; dof];
        let mut o = 0usize;
        for j in 0..=p {
            for i in 0..p {
                let raw = dof_map[o];
                let idx = if raw < 0 { (-raw - 1) as usize } else { raw as usize };
                dof2tk[idx] = if raw < 0 { 2 } else { 0 };
                nodes[idx] = [op[i], cp[j]];
                o += 1;
            }
        }
        for j in 0..p {
            for i in 0..=p {
                let raw = dof_map[o];
                let idx = if raw < 0 { (-raw - 1) as usize } else { raw as usize };
                dof2tk[idx] = if raw < 0 { 3 } else { 1 };
                nodes[idx] = [cp[i], op[j]];
                o += 1;
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                let raw = dof_map[o];
                let idx = if raw < 0 { (-raw - 1) as usize } else { raw as usize };
                dof2tk[idx] = 4;
                nodes[idx] = [cp[i], cp[j]];
                o += 1;
            }
        }
        debug_assert_eq!(o, dof);

        NdR2dQuad { p, dof_map, dof2tk, slots, nodes, cp, op }
    }

    pub fn order(&self) -> usize {
        self.p
    }

    pub fn n_dofs(&self) -> usize {
        (3 * self.p + 1) * (self.p + 1)
    }

    pub fn slots(&self) -> &[EmbeddedSlot] {
        &self.slots
    }

    pub fn dof2tk(&self) -> &[u8] {
        &self.dof2tk
    }

    /// The `tk_q` table entry as a 2-D reference vector (axes 0..3; the
    /// z axis carries `t3(2) = 1`).
    pub fn tangent(&self, tk: u8) -> [f64; 2] {
        const TK_Q: [[f64; 2]; 4] =
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        TK_Q[tk as usize]
    }

    pub fn nodes(&self) -> &[[f64; 2]] {
        &self.nodes
    }

    /// Reference `CalcVShape(ip, shape)` — `n_dofs() * 3`.
    pub fn eval_vshape_ref(&self, xi: &[f64], out: &mut [f64]) {
        let p = self.p;
        let (cx, _dcx) = lagrange_1d(&self.cp, xi[0]);
        let (ox, _dox) = lagrange_1d(&self.op, xi[0]);
        let (cy, _dcy) = lagrange_1d(&self.cp, xi[1]);
        let (oy, _doy) = lagrange_1d(&self.op, xi[1]);
        let mut o = 0usize;
        for j in 0..=p {
            for i in 0..p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, s) = resolve(raw);
                out[idx * 3] = s * ox[i] * cy[j];
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = 0.0;
            }
        }
        for j in 0..p {
            for i in 0..=p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, s) = resolve(raw);
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = s * cx[i] * oy[j];
                out[idx * 3 + 2] = 0.0;
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, _s) = resolve(raw);
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = cx[i] * cy[j];
            }
        }
    }

    /// Reference `CalcCurlShape(ip, curl_shape)` — `n_dofs() * 3`.
    pub fn eval_curl_ref(&self, xi: &[f64], out: &mut [f64]) {
        let p = self.p;
        let (cx, dcx) = lagrange_1d(&self.cp, xi[0]);
        let (ox, _dox) = lagrange_1d(&self.op, xi[0]);
        let (cy, dcy) = lagrange_1d(&self.cp, xi[1]);
        let (oy, _doy) = lagrange_1d(&self.op, xi[1]);
        let mut o = 0usize;
        for j in 0..=p {
            for i in 0..p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, s) = resolve(raw);
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = -s * ox[i] * dcy[j];
            }
        }
        for j in 0..p {
            for i in 0..=p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, s) = resolve(raw);
                out[idx * 3] = 0.0;
                out[idx * 3 + 1] = 0.0;
                out[idx * 3 + 2] = s * dcx[i] * oy[j];
            }
        }
        for j in 0..=p {
            for i in 0..=p {
                let raw = self.dof_map[o];
                o += 1;
                let (idx, _s) = resolve(raw);
                out[idx * 3] = cx[i] * dcy[j];
                out[idx * 3 + 1] = -dcx[i] * cy[j];
                out[idx * 3 + 2] = 0.0;
            }
        }
    }

    /// Physical `CalcVShape(Trans, shape)` — in-plane columns through `J⁻¹`.
    pub fn eval_vshape_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_vshape_ref(xi, out);
        let ji = jac.inv();
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = sx * ji[0] + sy * ji[2];
            out[k * 3 + 1] = sx * ji[1] + sy * ji[3];
            out[k * 3 + 2] = sz;
        }
    }

    /// Physical `CalcPhysCurlShape(Trans, curl_shape)` — `n_dofs() * 3`.
    /// MFEM scales columns 0/1 with `J` and then the whole matrix with
    /// `1 / Weight`, i.e. the z column is scaled too.
    pub fn eval_curl_phys(&self, xi: &[f64], jac: &Jac2D, out: &mut [f64]) {
        self.eval_curl_ref(xi, out);
        for k in 0..self.n_dofs() {
            let (sx, sy, sz) = (out[k * 3], out[k * 3 + 1], out[k * 3 + 2]);
            out[k * 3] = (sx * jac.j00 + sy * jac.j01) / jac.det;
            out[k * 3 + 1] = (sx * jac.j10 + sy * jac.j11) / jac.det;
            out[k * 3 + 2] = sz / jac.det;
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
