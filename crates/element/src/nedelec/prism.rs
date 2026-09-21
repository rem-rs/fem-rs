//! Nedelec-I H(curl) element on the reference triangular prism.
//!
//! Reference coordinates: (xi, eta, zeta) where xi in [0,1] (extrusion),
//! (eta, zeta) in unit triangle.
//!
//! Provides PrismND1 (order-1 barycentric Whitney forms) and PrismNDk
//! (1:1 port of MFEM `ND_WedgeElement`).

use crate::gll_basis::gll_nodes;
use crate::nedelec::tri_ndk::{chebyshev_d, invert_dense, TriNDk};
use crate::quadrature::{gauss_legendre_01, prism_rule};
use crate::reference::{QuadratureRule, VectorReferenceElement};

// ─── Barycentric coordinates (needed by PrismND1) ────────────────────────────

fn barycentric(xi: f64, eta: f64, zeta: f64) -> ([f64; 6], [[f64; 3]; 6]) {
    let a = 1.0 - xi;
    let b = 1.0 - eta - zeta;
    let lam = [a * b, a * eta, a * zeta, xi * b, xi * eta, xi * zeta];
    let grad = [
        [-b, -a, -a],
        [-eta, a, 0.0],
        [-zeta, 0.0, a],
        [b, -xi, -xi],
        [eta, xi, 0.0],
        [zeta, 0.0, xi],
    ];
    (lam, grad)
}

fn edge_geom(edge: usize) -> ([f64; 3], [f64; 3]) {
    // MFEM `Constants<Geometry::PRISM>::Edges` ordering:
    // (0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5).
    match edge {
        0 => ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // (0,1)
        1 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), // (1,2)
        2 => ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]), // (2,0)
        3 => ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]), // (3,4)
        4 => ([1.0, 1.0, 0.0], [1.0, 0.0, 1.0]), // (4,5)
        5 => ([1.0, 0.0, 1.0], [1.0, 0.0, 0.0]), // (5,3)
        6 => ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), // (0,3)
        7 => ([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]), // (1,4)
        _ => ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0]), // (2,5)
    }
}

// ─── PrismND1 (original barycentric Whitney 1-forms) ─────────────────────────

pub struct PrismND1;

impl VectorReferenceElement for PrismND1 {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        1
    }
    fn n_dofs(&self) -> usize {
        9
    }
    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        // Reference coordinates (Rust): (ξ, η, ζ) with ξ the extrusion and
        // (η, ζ) the unit triangle.  MFEM's wedge uses (x,y,z) with the
        // triangle in (x,y) and extrusion z — the Rust axis order is
        // (ξ,η,ζ) = (z,x,y).  Basis = tensor product
        //   NDTriangle(η,ζ) ⊗ H1Segment(ξ)  for the 6 triangle edges,
        //   H1Triangle(η,ζ) ⊗ NDSegment     for the 3 vertical edges.
        let (xi_, eta, zeta) = (xi[0], xi[1], xi[2]);
        let lam0 = 1.0 - eta - zeta; // H1 triangle basis
        let lam1 = eta;
        let lam2 = zeta;
        // NDTriangle edge bases (Whitney 1-forms, Rust TriND1, with edge 2
        // oriented (2,0) per MFEM ND_TriangleElement — opposite to Rust's
        // TriND1 (0,2) convention, hence the sign flip on e2):
        //   e0 (0,1): (1−ζ, η),  e1 (1,2): (−ζ, η),  e2 (2,0): (−ζ, η−1)
        let tri_x = [1.0 - zeta, -zeta, -zeta];
        let tri_y = [eta, eta, eta - 1.0];
        for e in 0..3 {
            // layer 0 (ξ=0) and layer 1 (ξ=1)
            let s0 = 1.0 - xi_;
            let s1 = xi_;
            // 3-D components in Rust order (ξ, η, ζ) = MFEM (z, x, y):
            // triangle-edge dofs live in the (η,ζ) plane → components 1,2.
            values[e * 3] = 0.0;
            values[e * 3 + 1] = tri_x[e] * s0;
            values[e * 3 + 2] = tri_y[e] * s0;
            values[(e + 3) * 3] = 0.0;
            values[(e + 3) * 3 + 1] = tri_x[e] * s1;
            values[(e + 3) * 3 + 2] = tri_y[e] * s1;
        }
        // vertical edges: (λ_k, 0, 0) in (ξ,η,ζ) components.
        values[6 * 3] = lam0;
        values[6 * 3 + 1] = 0.0;
        values[6 * 3 + 2] = 0.0;
        values[7 * 3] = lam1;
        values[7 * 3 + 1] = 0.0;
        values[7 * 3 + 2] = 0.0;
        values[8 * 3] = lam2;
        values[8 * 3 + 1] = 0.0;
        values[8 * 3 + 2] = 0.0;
    }
    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let (xi_, eta, zeta) = (xi[0], xi[1], xi[2]);
        // MFEM ND_WedgeElement::CalcCurlShape in Rust (ξ,η,ζ) = (z,x,y) order:
        //   tri-edge dof: curl = (tn_curl·s, −tn_y·ds, tn_x·ds)
        //   vertical dof: curl = (0, ∂λ/∂ζ, −∂λ/∂η)
        let tri_x = [1.0 - zeta, -zeta, -zeta];
        let tri_y = [eta, eta, eta - 1.0];
        let tri_curl = [2.0, 2.0, 2.0]; // 2-D curls (e2 oriented (2,0))
        for e in 0..3 {
            for (k, sgn) in [(0usize, 1.0f64), (3, -1.0)].iter() {
                let s = if *sgn > 0.0 { 1.0 - xi_ } else { xi_ };
                let ds = if *sgn > 0.0 { -1.0 } else { 1.0 };
                let i = e + k;
                curl_vals[i * 3] = tri_curl[e] * s;
                curl_vals[i * 3 + 1] = -tri_y[e] * ds;
                curl_vals[i * 3 + 2] = tri_x[e] * ds;
            }
        }
        // vertical edges: ∇λ₀ = (−1,−1), ∇λ₁ = (1,0), ∇λ₂ = (0,1) in (η,ζ).
        let dlam = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        for k in 0..3 {
            let i = 6 + k;
            curl_vals[i * 3] = 0.0;
            curl_vals[i * 3 + 1] = dlam[k][1]; // ∂λ/∂ζ
            curl_vals[i * 3 + 2] = -dlam[k][0]; // −∂λ/∂η
        }
    }
    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        prism_rule(order)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.5],
            vec![0.0, 0.5, 0.5],
            vec![1.0, 0.5, 0.0],
            vec![1.0, 0.0, 0.5],
            vec![1.0, 0.5, 0.5],
            vec![0.5, 0.0, 0.0],
            vec![0.5, 1.0, 0.0],
            vec![0.5, 0.0, 1.0],
        ]
    }
}

// ─── PrismNDk (1:1 port of MFEM `ND_WedgeElement`) ──────────────────────────

/// One element-local DOF of the MFEM `ND_WedgeElement` slot table:
/// `(t_dof, s_dof, dof2tk)` — `t_dof` indexes the triangular sub-element
/// (`dof2tk != 3`: the Nédélec triangle; `dof2tk == 3`: the H¹ triangle),
/// `s_dof` the extruded sub-segment (H¹ for the in-plane tangents, Nédélec
/// for the vertical `dof2tk == 3`), and `dof2tk` the reference tangent of
/// MFEM's wedge `tk` table.
type WedgeSlot = (u16, u16, u8);

/// MFEM `ND_WedgeElement` wedge `tk` table, already mapped to the fem-rs
/// reference frame `(ξ,η,ζ) = (z,x,y)` (MFEM `(x,y,z)` components rotate as
/// `(z,x,y)`):
///   tk0 `(1,0,0)` → `(0,1,0)`, tk1 `(−1,1,0)` → `(0,−1,1)`,
///   tk2 `(0,−1,0)` → `(0,0,−1)`, tk3 `(0,0,1)` → `(1,0,0)`,
///   tk4 `(0,1,0)` → `(0,0,1)`.
const WEDGE_TK_RUST: [[f64; 3]; 5] = [
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// The slot enumeration of MFEM `ND_WedgeElement::ND_WedgeElement`
/// (`fe_nd.cpp`): `9p` edge slots (bottom triangle edges (0,1) (1,2) (2,0),
/// top (3,4) (4,5) (5,3), vertical (0,3) (1,4) (2,5)), the bottom tri face
/// (0,2,1) and top tri face (3,4,5) — two tangent slots per point, the three
/// quad faces (0,1,4,3), (1,2,5,4), (2,0,3,5) — an in-plane-tangent block
/// then a vertical-tangent block — and the interior (in-plane pairs on the
/// closed layers, then vertical slots on the open layers).
fn wedge_slot_table(p: usize) -> Vec<WedgeSlot> {
    let pm1 = p - 1;
    let pm2 = p.saturating_sub(2);
    let mut t: Vec<WedgeSlot> = Vec::with_capacity(3 * p * (p + 1) * (p + 2) / 2);
    // edges
    for i in 0..p {
        t.push((i as u16, 0, 0));
    }
    for i in 0..p {
        t.push(((p + i) as u16, 0, 1));
    }
    for i in 0..p {
        t.push(((2 * p + i) as u16, 0, 2));
    }
    for i in 0..p {
        t.push((i as u16, 1, 0));
    }
    for i in 0..p {
        t.push(((p + i) as u16, 1, 1));
    }
    for i in 0..p {
        t.push(((2 * p + i) as u16, 1, 2));
    }
    for i in 0..p {
        t.push((0, i as u16, 3));
    }
    for i in 0..p {
        t.push((1, i as u16, 3));
    }
    for i in 0..p {
        t.push((2, i as u16, 3));
    }
    if p >= 2 {
        // bottom tri face (0,2,1): the ND-triangle interior point with index
        // `l = j + (2p−1−i)·i/2`, tk4 slot first then tk0.
        for j in 0..=pm2 {
            for i in 0..=(pm2 - j) {
                let l = j + (2 * p - 1 - i) * i / 2;
                t.push(((3 * p + 2 * l + 1) as u16, 0, 4));
                t.push(((3 * p + 2 * l) as u16, 0, 0));
            }
        }
        // top tri face (3,4,5): tk0 slot first then tk4.
        let mut m = 0;
        for _j in 0..=pm2 {
            for _i in 0..=(pm2 - _j) {
                t.push(((3 * p + m) as u16, 1, 0));
                m += 1;
                t.push(((3 * p + m) as u16, 1, 4));
                m += 1;
            }
        }
        // quad face (0,1,4,3)
        for j in 2..=p {
            for i in 0..p {
                t.push((i as u16, j as u16, 0));
            }
        }
        for j in 0..p {
            for i in 0..pm1 {
                t.push(((3 + i) as u16, j as u16, 3));
            }
        }
        // quad face (1,2,5,4)
        for j in 2..=p {
            for i in 0..p {
                t.push(((p + i) as u16, j as u16, 1));
            }
        }
        for j in 0..p {
            for i in 0..pm1 {
                t.push(((p + 2 + i) as u16, j as u16, 3));
            }
        }
        // quad face (2,0,3,5)
        for j in 2..=p {
            for i in 0..p {
                t.push(((2 * p + i) as u16, j as u16, 2));
            }
        }
        for j in 0..p {
            for i in 0..pm1 {
                t.push(((2 * p + 1 + i) as u16, j as u16, 3));
            }
        }
        // interior: in-plane pairs on the closed layers
        for k in 2..=p {
            let mut l = 0;
            for _j in 0..=pm2 {
                for _i in 0..=(pm2 - _j) {
                    t.push(((3 * p + l) as u16, k as u16, 0));
                    l += 1;
                    t.push(((3 * p + l) as u16, k as u16, 4));
                    l += 1;
                }
            }
        }
        // interior: vertical slots on the open layers (H1-tri interior points)
        for k in 0..p {
            let mut l = 0;
            for _j in 0..pm2 {
                for _i in 0..(pm2 - _j) {
                    t.push(((3 * p + l) as u16, k as u16, 3));
                    l += 1;
                }
            }
        }
    }
    t
}

/// Lagrange basis (values + derivatives) on the arbitrary ascending node set
/// `nodes` evaluated at `x` — the shared 1-D factor of the H¹ segment
/// (Gauss-Lobatto nodes) and the Nédélec segment (Gauss-Legendre nodes).
fn lagrange_eval(nodes: &[f64], x: f64) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    let mut w = vec![1.0_f64; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                w[j] /= nodes[j] - nodes[k];
            }
        }
    }
    let exact = (0..n).find(|&m| x == nodes[m]);
    let mut v = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    if let Some(m) = exact {
        v[m] = 1.0;
        let a0: f64 = (0..n)
            .filter(|&k| k != m)
            .map(|k| 1.0 / (nodes[m] - nodes[k]))
            .sum();
        d[m] = a0;
        for j in 0..n {
            if j != m {
                d[j] = (w[j] / w[m]) / (nodes[m] - nodes[j]);
            }
        }
    } else {
        let mut lam = vec![0.0_f64; n];
        let mut dlam = vec![0.0_f64; n];
        let (mut s, mut ds) = (0.0_f64, 0.0_f64);
        for j in 0..n {
            let t = x - nodes[j];
            lam[j] = w[j] / t;
            dlam[j] = -w[j] / (t * t);
            s += lam[j];
            ds += dlam[j];
        }
        for j in 0..n {
            v[j] = lam[j] / s;
            d[j] = (dlam[j] - v[j] * ds) / s;
        }
    }
    (v, d)
}

/// MFEM `H1_TriangleElement` GLL node set (ascending `[0,1]`).
fn wedge_gll(p: usize) -> Vec<f64> {
    gll_nodes(p).iter().map(|&x| 0.5 * (x + 1.0)).collect()
}

/// The H¹ triangle nodes in MFEM entity order (`fe_h1.cpp`): vertices, then
/// the edge interiors — (0,1) ascending, (1,2), (2,0) descending — then the
/// barycentric GLL interior points.
fn h1_tri_nodes(p: usize, cp: &[f64]) -> Vec<[f64; 2]> {
    let mut nodes = vec![[cp[0], cp[0]], [cp[p], cp[0]], [cp[0], cp[p]]];
    if p >= 2 {
        for i in 1..p {
            nodes.push([cp[i], cp[0]]);
        }
        for i in 1..p {
            nodes.push([cp[p - i], cp[i]]);
        }
        for i in 1..p {
            nodes.push([cp[0], cp[p - i]]);
        }
        for j in 1..p {
            for i in 1..(p - j) {
                let w = cp[i] + cp[j] + cp[p - i - j];
                nodes.push([cp[i] / w, cp[j] / w]);
            }
        }
    }
    nodes
}

/// `(point, tangent)` of every local DOF in MFEM's reference frame — the
/// single source of truth behind [`PrismNDk::mfem_layout_points`],
/// [`VectorReferenceElement::dof_coords`] (frame-permuted) and
/// [`PrismNDk::dof_tangents`].  Node of slot `(t_dof, s_dof, tk)`:
/// `tk != 3` → the Nédélec-triangle point `t_dof` on the layer
/// `s1_nodes[s_dof]` (H¹ segment, endpoints first); `tk == 3` → the H¹-tri
/// point `t_dof` at the Nédélec-segment layer `sn_nodes[s_dof]`.
fn wedge_layout(p: usize, slots: &[WedgeSlot]) -> Vec<([f64; 3], [f64; 3])> {
    let cp = wedge_gll(p);
    // H1 segment node order (fe_h1.cpp): endpoints first, then the interior
    // GLL points ascending.
    let s1_nodes: Vec<f64> = [cp[0], cp[p]]
        .into_iter()
        .chain(cp[1..p].iter().copied())
        .collect();
    let sn_nodes = gauss_legendre_01(p).0;
    let tn_nodes: Vec<[f64; 2]> = TriNDk::new(p)
        .dof_coords()
        .iter()
        .map(|c| [c[0], c[1]])
        .collect();
    let t1_nodes = h1_tri_nodes(p, &cp);
    slots
        .iter()
        .map(|&(td, sd, tk)| {
            let (td, sd) = (td as usize, sd as usize);
            // fem-rs frame (ξ,η,ζ) = (z,x,y): extrusion first, tri plane last.
            let node = if tk != 3 {
                [s1_nodes[sd], tn_nodes[td][0], tn_nodes[td][1]]
            } else {
                [sn_nodes[sd], t1_nodes[td][0], t1_nodes[td][1]]
            };
            (node, WEDGE_TK_RUST[tk as usize])
        })
        .collect()
}

/// MFEM-faithful arbitrary-order Nédélec-I element on the reference prism
/// (1:1 port of MFEM `ND_WedgeElement(p, GaussLobatto, GaussLegendre)`,
/// `fe_nd.cpp`).
///
/// Frame: the fem-rs reference `(ξ, η, ζ)` carries the extrusion in `ξ` and
/// the unit triangle in `(η, ζ)` — i.e. `(ξ,η,ζ) = (z,x,y)` of MFEM's wedge,
/// the frame the assembly geometry element (`PrismPk`) uses.  Basis vectors
/// rotate with the frame (`(v_ξ,v_η,v_ζ)_RUST = (v_z,v_x,v_y)_MFEM`); the
/// slot order is exactly MFEM's `dof_map` enumeration ([`wedge_slot_table`]).
///
/// The DOF functionals are MFEM's nodal point values `σ_i(Φ) = Φ(x_i)·(J tk_i)`
/// (`fe_base.cpp::Project_ND`) at the `FE::Nodes` points [`dof_coords`] with
/// the *unnormalized* reference tangents [`dof_tangents`] — on the shared
/// `[0,1]³` reference domain there is no pull-back rescaling, the reference
/// basis *is* MFEM's.
pub struct PrismNDk {
    order: usize,
    slots: Vec<WedgeSlot>,
    /// H¹-triangle entity-ordered basis via `Ti` (`T` = lex tensor values at
    /// the entity-ordered nodes, MFEM `H1_TriangleElement`).
    h1_tri_ti: Vec<f64>,
    ndtri: TriNDk,
    layout: Vec<([f64; 3], [f64; 3])>,
}

impl PrismNDk {
    pub fn new(order: usize) -> Self {
        assert!(order >= 1, "PrismNDk: order >= 1");
        let slots = wedge_slot_table(order);
        let layout = wedge_layout(order, &slots);
        // H1 triangle interpolation matrix (lex-ordered tensor values at the
        // entity-ordered nodes) and its inverse.
        let p = order;
        let cp = wedge_gll(p);
        let t1_nodes = h1_tri_nodes(p, &cp);
        let n1 = (p + 1) * (p + 2) / 2;
        let mut t = vec![0.0_f64; n1 * n1];
        for (k, nd) in t1_nodes.iter().enumerate() {
            // MFEM `poly1d.CalcBasis` is the **Chebyshev** basis
            // (`T_j(2x−1)`), not the nodal Lagrange — the raw lex products
            // built from it are not individually nodal, which is exactly why
            // the `Ti` interpolation inverse is needed.
            let (sx, _) = chebyshev_d(p, nd[0]);
            let (sy, _) = chebyshev_d(p, nd[1]);
            let (sl, _) = chebyshev_d(p, 1.0 - nd[0] - nd[1]);
            let mut o = 0;
            for j in 0..=p {
                for i in 0..=(p - j) {
                    // MFEM stores T[o][k] = lex value o at the entity node k
                    // and factors `Ti = T⁻¹`; entity-ordered values are then
                    // `Ti · lex(pt)`.
                    t[o * n1 + k] = sx[i] * sy[j] * sl[p - i - j];
                    o += 1;
                }
            }
        }
        let h1_tri_ti = invert_dense(n1, &t, "PrismNDk H1 tri");
        PrismNDk {
            order,
            slots,
            h1_tri_ti,
            ndtri: TriNDk::new(order),
            layout,
        }
    }

    /// Reference tangents `t̂_i` of every local DOF in the fem-rs frame
    /// `(ξ,η,ζ) = (z,x,y)` — MFEM's `tk[dof2tk[i]]` rotated; the physical dual
    /// tangent is `J·t̂` (`fe_base.cpp::Project_ND`).
    pub fn dof_tangents(&self) -> Vec<[f64; 3]> {
        self.slots
            .iter()
            .map(|&(_, _, tk)| WEDGE_TK_RUST[tk as usize])
            .collect()
    }

    /// The sub-element shape values at `xi` (fem-rs frame): the
    /// Nédélec-triangle values/curls, the H¹-triangle values/gradients
    /// (entity order), the H¹-segment values/derivatives (H¹ dof order) and
    /// the Nédélec-segment values.
    #[allow(clippy::type_complexity)]
    fn sub_shapes(
        &self,
        xi: &[f64],
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let p = self.order;
        let (x, y, z) = (xi[1], xi[2], xi[0]); // MFEM (x,y,z) = (η,ζ,ξ)
        let n_t = self.ndtri.n_dofs();
        let mut tn = vec![0.0_f64; 2 * n_t];
        self.ndtri.eval_basis_vec(&[x, y], &mut tn);
        let mut tn_curl = vec![0.0_f64; n_t];
        self.ndtri.eval_curl(&[x, y], &mut tn_curl);
        // H1 triangle: entity-ordered values and gradients from the Chebyshev
        // lex tensor products through `Ti` (MFEM `H1_TriangleElement`).
        let n1 = (p + 1) * (p + 2) / 2;
        let cp = wedge_gll(p);
        let (sx, dx) = chebyshev_d(p, x);
        let (sy, dy) = chebyshev_d(p, y);
        let (sl, dl) = chebyshev_d(p, 1.0 - x - y);
        let mut lex = vec![0.0_f64; n1];
        let mut lex_dx = vec![0.0_f64; n1];
        let mut lex_dy = vec![0.0_f64; n1];
        let mut o = 0;
        for j in 0..=p {
            for i in 0..=(p - j) {
                let l = p - i - j;
                lex[o] = sx[i] * sy[j] * sl[l];
                lex_dx[o] = dx[i] * sy[j] * sl[l] - sx[i] * sy[j] * dl[l];
                lex_dy[o] = sx[i] * dy[j] * sl[l] - sx[i] * sy[j] * dl[l];
                o += 1;
            }
        }
        let mut t1 = vec![0.0_f64; n1];
        let mut t1_dx = vec![0.0_f64; n1];
        let mut t1_dy = vec![0.0_f64; n1];
        for k in 0..n1 {
            for b in 0..n1 {
                let c = self.h1_tri_ti[k * n1 + b];
                t1[k] += c * lex[b];
                t1_dx[k] += c * lex_dx[b];
                t1_dy[k] += c * lex_dy[b];
            }
        }
        // H1 segment in H1 dof order (endpoints first), Nédélec segment
        // ascending.
        let s1_nodes: Vec<f64> = [cp[0], cp[p]]
            .into_iter()
            .chain(cp[1..p].iter().copied())
            .collect();
        let (sa, da) = lagrange_eval(&cp, z);
        let mut s1 = vec![0.0_f64; p + 1];
        let mut s1_d = vec![0.0_f64; p + 1];
        for (j, &node) in s1_nodes.iter().enumerate() {
            // The H1 node set is the GLL set: match the ascending index by
            // position (endpoints cp[0]/cp[p] included).
            let idx = cp.iter().position(|&c| c == node).unwrap();
            s1[j] = sa[idx];
            s1_d[j] = da[idx];
        }
        let eop = gauss_legendre_01(p).0;
        let (sn, _) = lagrange_eval(&eop, z);
        (tn, tn_curl, t1, t1_dx, t1_dy, s1, s1_d, sn)
    }

    /// Reference points of the MFEM `ND_WedgeElement(p)` nodal layout
    /// (`FE::Nodes`, one entry per element slot) in MFEM's wedge frame —
    /// the triangle {(0,0),(1,0),(0,1)} in (x,y) and z ∈ [0,1] — the layout
    /// the HCurlSpace prism slot tables mirror (D525).  Derived from the same
    /// slot table as the basis ([`wedge_layout`]), so the element cannot
    /// drift from its own DOF layout; `mfem_layout_points()[i]` is the frame
    /// permutation `(η,ζ,ξ)` of `dof_coords()[i]`.
    pub fn mfem_layout_points(&self) -> Vec<[f64; 3]> {
        self.layout
            .iter()
            .map(|(x, _)| [x[1], x[2], x[0]])
            .collect()
    }
}

impl VectorReferenceElement for PrismNDk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * self.order * (self.order + 1) * (self.order + 2) / 2
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        if self.order == 1 {
            // `ND_WedgeElement(1)` is the Whitney wedge = `PrismND1` slot for
            // slot (the fem-rs `TriNDk(1)` shortcut used by the general path
            // below keeps the historical 2-D ND1 flip on tri edge (2,0),
            // which MFEM's p = 1 does not have).
            PrismND1.eval_basis_vec(xi, values);
            return;
        }
        let (tn, _tn_curl, t1, _t1_dx, _t1_dy, s1, _s1_d, sn) = self.sub_shapes(xi);
        for (i, &(td, sd, tk)) in self.slots.iter().enumerate() {
            let (td, sd) = (td as usize, sd as usize);
            // MFEM (vx,vy,vz) → fem-rs components (ξ,η,ζ) = (z,x,y).
            let (vx, vy, vz);
            if tk != 3 {
                vx = tn[2 * td] * s1[sd];
                vy = tn[2 * td + 1] * s1[sd];
                vz = 0.0;
            } else {
                vx = 0.0;
                vy = 0.0;
                vz = t1[td] * sn[sd];
            }
            values[3 * i] = vz;
            values[3 * i + 1] = vx;
            values[3 * i + 2] = vy;
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        if self.order == 1 {
            PrismND1.eval_curl(xi, curl_vals);
            return;
        }
        // The curl differentiates only the in-plane/vertical 1-D factors and
        // the Nédélec-triangle scalar curl; the H¹-triangle *values* `t1`
        // are not needed here.
        let (tn, tn_curl, _t1, t1_dx, t1_dy, s1, s1_d, sn) = self.sub_shapes(xi);
        for (i, &(td, sd, tk)) in self.slots.iter().enumerate() {
            let (td, sd) = (td as usize, sd as usize);
            // MFEM ND_WedgeElement::CalcCurlShape.
            let (cx, cy, cz);
            if tk != 3 {
                cx = -tn[2 * td + 1] * s1_d[sd];
                cy = tn[2 * td] * s1_d[sd];
                cz = tn_curl[td] * s1[sd];
            } else {
                cx = t1_dy[td] * sn[sd];
                cy = -t1_dx[td] * sn[sd];
                cz = 0.0;
            }
            curl_vals[3 * i] = cz;
            curl_vals[3 * i + 1] = cx;
            curl_vals[3 * i + 2] = cy;
        }
    }

    fn eval_div(&self, _xi: &[f64], div_vals: &mut [f64]) {
        for v in div_vals.iter_mut() {
            *v = 0.0;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        prism_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.layout.iter().map(|(x, _)| x.to_vec()).collect()
    }
}

// MFEM 4.10 truth for the ND prism elements (golden dump generated by
// `tmp/d546/gen_dump.py` from `tmp/d546/d548_nd_prism_pyra_probe.cpp`).
#[cfg(test)]
mod mfem_dump {
    include!("prism/prism_mfem_dump.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MFEM `ND_WedgeElement` layout table: `3p(p+1)(p+2)/2` slots for
    /// every order (90 at p = 3), now the element's own dimension.
    #[test]
    fn mfem_layout_point_counts() {
        let want = |p: usize| 3 * p * (p + 1) * (p + 2) / 2;
        for p in 1..=4usize {
            assert_eq!(PrismNDk::new(p).n_dofs(), want(p));
            assert_eq!(PrismNDk::new(p).mfem_layout_points().len(), want(p));
        }
    }

    #[test]
    fn prism_nd1_n_dofs() {
        assert_eq!(PrismND1.n_dofs(), 9);
    }
    #[test]
    fn prism_nd1_basis_finite() {
        let mut v = [0.0; 27];
        PrismND1.eval_basis_vec(&[0.2, 0.3, 0.1], &mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn prism_nd1_curl_finite() {
        let mut c = [0.0; 27];
        PrismND1.eval_curl(&[0.2, 0.3, 0.1], &mut c);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    /// PrismNDk(1) is MFEM `ND_WedgeElement(1)` — the same slot order and
    /// basis as PrismND1 (Whitney).
    #[test]
    fn prism_ndk_k1_same_dimension() {
        assert_eq!(PrismND1.n_dofs(), PrismNDk::new(1).n_dofs());
        let vk = PrismNDk::new(1);
        let mut vals = [0.0; 27];
        vk.eval_basis_vec(&[0.2, 0.3, 0.1], &mut vals);
        assert!(vals.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn prism_ndk_k2_basis_finite() {
        let ndk = PrismNDk::new(2);
        let mut v = vec![0.0; ndk.n_dofs() * 3];
        ndk.eval_basis_vec(&[0.2, 0.3, 0.1], &mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// Per-slot MFEM 4.10 parity (`d548_out.txt` truth): for p = 1..3 the
    /// fem-rs basis and curl at the two probe points equal MFEM's
    /// `CalcVShape`/`CalcCurlShape` rows slot by slot — the frame map
    /// `(ξ,η,ζ) = (z,x,y)` turns fem-rs components into MFEM's `(vx,vy,vz)`.
    #[test]
    fn prism_ndk_matches_mfem_410_probe() {
        // MFEM (x,y,z) sample points; fem-rs evaluates at (z,x,y).
        let pts = [[0.621, 0.137, 0.413], [0.53, 0.71, 0.22]];
        for p in 1..=3usize {
            let (ndofs, vc): (usize, [&[[f64; 6]]; 2]) = match p {
                1 => (
                    mfem_dump::NDOFS_1,
                    [&mfem_dump::VC_1_0, &mfem_dump::VC_1_1],
                ),
                2 => (
                    mfem_dump::NDOFS_2,
                    [&mfem_dump::VC_2_0, &mfem_dump::VC_2_1],
                ),
                _ => (
                    mfem_dump::NDOFS_3,
                    [&mfem_dump::VC_3_0, &mfem_dump::VC_3_1],
                ),
            };
            let e = PrismNDk::new(p);
            assert_eq!(e.n_dofs(), ndofs);
            let n = e.n_dofs();
            let mut v = vec![0.0_f64; n * 3];
            let mut c = vec![0.0_f64; n * 3];
            for (q, xi) in pts.iter().enumerate() {
                e.eval_basis_vec(xi, &mut v);
                e.eval_curl(xi, &mut c);
                let golden = vc[q];
                for i in 0..n {
                    // fem-rs (ξ,η,ζ) = MFEM (z,x,y)
                    let want_v = [golden[i][2], golden[i][0], golden[i][1]];
                    let want_c = [golden[i][5], golden[i][3], golden[i][4]];
                    for d in 0..3 {
                        assert!(
                            (v[i * 3 + d] - want_v[d]).abs() < 5e-12,
                            "p={p} q={q} V[{i}][{d}]: {} vs {}",
                            v[i * 3 + d],
                            want_v[d]
                        );
                        assert!(
                            (c[i * 3 + d] - want_c[d]).abs() < 5e-12,
                            "p={p} q={q} C[{i}][{d}]: {} vs {}",
                            c[i * 3 + d],
                            want_c[d]
                        );
                    }
                }
            }
        }
    }

    /// `FE::Nodes` parity: the frame-permuted layout equals the probe's node
    /// table slot by slot, and the reference tangents equal the probe's
    /// unit-vector `Project` tangents rotated into the fem-rs frame.
    #[test]
    fn prism_ndk_nodes_and_tangents_match_mfem_410_probe() {
        for p in 1..=3usize {
            let (nodes_mfem, tk_mfem): (&[[f64; 3]], &[[f64; 3]]) = match p {
                1 => (&mfem_dump::NODES_1, &mfem_dump::TK_1),
                2 => (&mfem_dump::NODES_2, &mfem_dump::TK_2),
                _ => (&mfem_dump::NODES_3, &mfem_dump::TK_3),
            };
            let e = PrismNDk::new(p);
            let n = e.n_dofs();
            let nodes = e.mfem_layout_points();
            let tks = e.dof_tangents();
            for i in 0..n {
                for d in 0..3 {
                    assert!(
                        (nodes[i][d] - nodes_mfem[i][d]).abs() < 5e-14,
                        "p={p} NODES[{i}][{d}]"
                    );
                    // MFEM tk (tx,ty,tz) → fem-rs (tz,tx,ty).
                    let want_tk = [tk_mfem[i][2], tk_mfem[i][0], tk_mfem[i][1]];
                    assert!(
                        (tks[i][d] - want_tk[d]).abs() < 5e-14,
                        "p={p} TK[{i}][{d}]"
                    );
                }
            }
        }
    }

    /// Nodal property: `σ_j(Φ_i) = Φ(x_j)·t̂_j = δ_ij` with the element's own
    /// `(dof_coords, dof_tangents)` — MFEM's `Project_ND` duals.
    #[test]
    fn prism_ndk_dof_functionals_are_point_values() {
        for p in 1..=4usize {
            let e = PrismNDk::new(p);
            let n = e.n_dofs();
            let coords = e.dof_coords();
            let tks = e.dof_tangents();
            let mut v = vec![0.0_f64; n * 3];
            for j in 0..n {
                e.eval_basis_vec(&coords[j], &mut v);
                for i in 0..n {
                    let s = v[i * 3] * tks[j][0]
                        + v[i * 3 + 1] * tks[j][1]
                        + v[i * 3 + 2] * tks[j][2];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-10,
                        "p={p}: σ_{j}(Φ_{i}) = {s} (want {want})"
                    );
                }
            }
        }
    }
}
