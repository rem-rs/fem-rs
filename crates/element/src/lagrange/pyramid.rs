//! Arbitrary-order Lagrange element on the reference pyramid.
//!
//! Reference pyramid: vertices (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1).
//! Domain: x ∈ [0, 1-z], y ∈ [0, 1-z], z ∈ [0,1]. Volume = 1/3.
//!
//! Uses the collapsed-coordinate formulation:
//! - Collapsed coordinates: r = x/(1-z), s = y/(1-z), t = z, all in [0,1].
//! - Nodes are placed at equispaced grid points: (r,s,t) = (i/p, j/p, k/p) where
//!   i,j ≤ p-k. Total DOFs = Σ_{k=0}^{p} (p-k+1)² = (p+1)(p+2)(2p+3)/6.
//!
//! Basis: φ(x,y,z) = L_k(t) · L_i^{(p-k)}(r) · L_j^{(p-k)}(s)
//! where L_n^{(d)} is the standard degree-d Lagrange polynomial through
//! equispaced nodes on [0,1].

use crate::quadrature::pyramid_rule;
use crate::reference::{QuadratureRule, ReferenceElement};

fn lagrange_1d_val(i: usize, degree: usize, xi: f64) -> f64 {
    if degree == 0 {
        return 1.0;
    }
    let d = degree as f64;
    let t = d * xi;
    let mut val = 1.0;
    let tn = i as f64;
    for m in 0..=degree {
        if m != i {
            val *= (t - m as f64) / (tn - m as f64);
        }
    }
    val
}

fn lagrange_1d_deriv(i: usize, degree: usize, xi: f64) -> f64 {
    if degree == 0 {
        return 0.0;
    }
    let d = degree as f64;
    let t = d * xi;
    let mut sum = 0.0;
    let tn = i as f64;
    for k in 0..=degree {
        if k != i {
            let mut term = 1.0;
            for m in 0..=degree {
                if m != i && m != k {
                    term *= (t - m as f64) / (tn - m as f64);
                }
            }
            sum += term / (tn - k as f64);
        }
    }
    d * sum
}

/// Pyramid basis family selection (MFEM `ScalarPyramid::DefaultType`, i.e.
/// `H1_FECollection`'s `pyr_type` argument, `fem/fe/fe_pyramid.hpp:23`).
///
/// * `Fuentes` (`pyr_type = 1`) — [`super::pyramid_fuentes::H1FuentesPyramidPk`],
///   `p(p²+3)+1` DOFs.  This is MFEM's **default** (`ScalarPyramid::DefaultType
///   = 1`), and therefore fem-rs's default too (D347): `H1Space::new`,
///   `DofManager::new`, `ref_elem_vol_h1` and `Mesh::set_curvature` all select
///   this family unless the caller asks for [`Bergot`](Self::Bergot).
/// * `Bergot` (`pyr_type = 0`) — [`H1PyramidPk`], `(p+1)(p+2)(2p+3)/6` DOFs.
///   Preserved as an explicit, fully supported opt-out
///   (`space::H1Space::with_pyramid_basis`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyramidBasisType {
    /// Bernardi-Boggs-Fluery type (collapsed coordinates, GLL-barycentric
    /// nodes) — [`H1PyramidPk`]; MFEM's `pyr_type = 0`.  Only conforming with
    /// MFEM's H¹ tetrahedron/wedge on the pyramid's triangular faces up to
    /// `p = 2` (from `p = 3` the tri-face lattice differs, measured in D347).
    Bergot,
    /// Fuentes-Keith-Demkowicz type (exact sequence) —
    /// [`super::pyramid_fuentes::H1FuentesPyramidPk`]; MFEM's `pyr_type = 1`
    /// and `ScalarPyramid::DefaultType`.
    #[default]
    Fuentes,
}

/// The pyramid H¹ reference element of the given family and order — the two
/// arms of MFEM's `pyr_type` switch (`fe_coll.cpp:1976-1985`).
///
/// `PyramidBasisType::default()` (Fuentes) is what every "default" path in
/// fem-rs uses since D347: `fem_assembly::assembler::ref_elem_vol_h1` (via
/// `FESpace::pyramid_basis`) and
/// `fem_space::dof_manager::DofManager::build_pyramid_pk`.
pub fn h1_pyramid_element(p: usize, basis: PyramidBasisType) -> Box<dyn ReferenceElement> {
    match basis {
        PyramidBasisType::Bergot => Box::new(H1PyramidPk::new(p)),
        PyramidBasisType::Fuentes => {
            Box::new(super::pyramid_fuentes::H1FuentesPyramidPk::new(p))
        }
    }
}

/// Arbitrary-order Lagrange element on the reference pyramid.
///
/// DOF ordering: layer-by-layer from base (k=0) to apex (k=p).
/// Within each layer, row-major ordering over the (p-k+1)×(p-k+1) grid.
///
/// This is the legacy *equispaced collapsed-lattice* element, not one of
/// MFEM's two families: it agrees with [`H1PyramidPk`] (Bergot) only up to
/// `p = 2`, and with neither family beyond.  It is still the DG/L2 pyramid
/// element of `ref_elem_vol*` (the H¹ path uses
/// [`h1_pyramid_element`]) — the L2 default is Fuentes in MFEM (D325).
///
/// # D306 — L2 pyramid family gap (documented, **not** fixed)
///
/// MFEM picks its L2 pyramid element with the same `ScalarPyramid::DefaultType`
/// switch as the H1 one (`fem/fe/fe_pyramid.hpp:23`), and the default family is
/// *Fuentes*.  Neither MFEM arm is this element:
///
/// | element | nodes | DOFs (p = 1, 2, 3) |
/// |---|---|---|
/// | MFEM default `L2_FuentesPyramidElement(p, btype)` (`fe_l2.cpp:927`, `FunctionSpace::Uk`) | `(op[i]·(1 − a·op[k]), op[j]·(1 − a·op[k]), a·op[k])` with `op = Poly_1D::OpenPoints(p, btype)` and `a = 1` for an open btype (the `L2_FECollection` default `GaussLegendre`, `fe_coll.hpp:384`) / `a =` the largest `GaussLegendre` node for a closed one | `(p+1)³`: 8, 27, 64 |
/// | MFEM alternative `L2_BergotPyramidElement(p, btype)` (`fe_l2.cpp:1078`, `FunctionSpace::Pk`) | Fuentes-style barycentric lattice `(op[i]·(op[j] + op[p−j−k])/w, op[j]·(op[i] + op[p−i−k])/w, op[k]·op[p−k]/w)`, `w = wik·wjk·op[p−k]`, apex limit `w < apex_tol` → `(0, 0, 1)` | `(p+1)(p+2)(2p+3)/6` — **this element's count**: 5, 14, 30 |
/// | this element | equispaced **closed** layer lattice `(i/p, j/p, k/p)` | `(p+1)(p+2)(2p+3)/6`: 5, 14, 30 |
///
/// Consequences (measured facts, no divergence in DOF counts by accident):
/// `fem_assembly::assembler::ref_elem_vol_l2` routes `Pyramid5` to *this*
/// element (`ref_elem_vol`'s pyramid arm), so on a pyramid cell the L2 DOF
/// count already differs from MFEM's default (`(p+1)³` vs
/// `(p+1)(p+2)(2p+3)/6`), and against the Bergot-L2 alternative the *nodes*
/// differ (open-point barycentric lattice vs equispaced closed lattice), which
/// changes the nodal values of every interpolant/projector.  In addition
/// `fem_space::L2Space::new_with_basis` rejects `Pyramid5` outright
/// (`crates/space/src/l2.rs`, "*L2Space currently supports Tri3/Quad4 (2D) and
/// Tet4/Hex8 (3D)*"), so fem-rs has no L2 pyramid **space**: the element is
/// reachable only through the assembly-time `ref_elem_vol*` lookups.
///
/// Porting surface for D340 (both MFEM arms share it): the Fuentes pyramid
/// helpers `mu0/mu1` (`fe_pyramid.cpp`), `Poly_1D::CalcHomogenizedScaLegendre`
/// (`fe_base.cpp`), the `OpenPoints`/`ClosedPoints` 1-D tables, a Vandermonde
/// inverse (`DenseMatrix T; Ti.Factor(T)`, 1:1 with [`H1PyramidPk`]'s pattern),
/// plus — for the Fuentes arm — the `(p+1)³` DOF numbering
/// (`o = k(p+1)² + j(p+1) + i`) and L2-space layer support.  MFEM notes in
/// `fe_l2.cpp:940` that the Fuentes-L2 basis is *not independent* on closed
/// interpolation points for `p ≥ 1`, so closed requests are forced open in `z`.
pub struct PyramidPk {
    order: usize,
    layer_offset: Vec<usize>,
}

/// Integer collapsed-lattice labels `(i, j, k)` of MFEM
/// `H1_BergotPyramidElement(p)`'s DOF slots in **entity order** (D191/D299):
/// the layout `FiniteElementSpace::GetElementDofs` returns on a pyramid.
///
/// `0 ≤ k ≤ p` is the layer, `i, j ≥ 0` with `i + j ≤ p − k` the in-layer
/// collapsed indices along `v0→v1` / `v0→v3`.  The slot order is
///
/// * slots 0–4: vertices `(0,0,0) (p,0,0) (p,p,0) (0,p,0) (0,0,p)`;
/// * 8 edge blocks in MFEM's PYRAMID edge-table order and direction
///   `(0,1) (1,2) (3,2) (0,3) (0,4) (1,4) (2,4) (3,4)` (blocks 2/3 run
///   `v3→v2` and `v0→v3`);
/// * quad base-face block in the H1(quad) interior order (`j` outer, `i`
///   fastest);
/// * 4 tri side-face blocks in face order `(0,1,4) (1,2,4) (2,3,4) (3,0,4)`,
///   each in the H1(tri) interior order of that face;
/// * interior block: `k` outer, then `j`, `i` fastest.
///
/// Dumped from MFEM 4.10 `H1_FECollection(p, 3, GaussLobatto, pyr_type=0)`
/// (probe `tmp/d191/pyr_h1_probe.cpp`).
pub fn h1_pyramid_slot_labels(p: usize) -> Vec<[usize; 3]> {
    let mut slots: Vec<[usize; 3]> = Vec::with_capacity((p + 1) * (p + 2) * (2 * p + 3) / 6);
    slots.push([0, 0, 0]);
    slots.push([p, 0, 0]);
    slots.push([p, p, 0]);
    slots.push([0, p, 0]);
    slots.push([0, 0, p]);
    if p >= 2 {
        let corners = [[0, 0, 0], [p, 0, 0], [p, p, 0], [0, p, 0], [0, 0, p]];
        let edge_pairs = [[0usize, 1], [1, 2], [3, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]];
        for &[a, b] in &edge_pairs {
            let (ea, eb) = (corners[a], corners[b]);
            for q in 1..p {
                let pt = [0usize, 1, 2].map(|d| {
                    // Signed lerp: the apex edges decrease a coordinate.
                    let e = (ea[d] as isize) * (p - q) as isize
                        + (eb[d] as isize) * q as isize;
                    (e / p as isize) as usize
                });
                slots.push(pt);
            }
        }
        // Quad base face: j (v0→v3) outer, i (v0→v1) fastest.
        for j in 1..p {
            for i in 1..p {
                slots.push([i, j, 0]);
            }
        }
    }
    if p >= 3 {
        // Tri face (0,1,4): rows parallel to the base edge (z/k outer).
        for k in 1..=p - 2 {
            for i in 1..=p - 1 - k {
                slots.push([i, 0, k]);
            }
        }
        // Tri face (1,2,4).
        for k in 1..=p - 2 {
            for j in 1..=p - 1 - k {
                slots.push([p - k, j, k]);
            }
        }
        // Tri face (2,3,4): columns across the base edge.
        for i in 1..=p - 2 {
            for k in 1..=p - 1 - i {
                slots.push([i, p - k, k]);
            }
        }
        // Tri face (3,0,4).
        for j in 1..=p - 2 {
            for k in 1..=p - 1 - j {
                slots.push([0, j, k]);
            }
        }
        // Interior: k (layer) outer, j outer, i fastest.
        for k in 1..=p - 2 {
            for j in 1..=p - 1 - k {
                for i in 1..=p - 1 - k {
                    slots.push([i, j, k]);
                }
            }
        }
    }
    slots
}

/// Reference position of the collapsed-lattice label `l = (i, j, k)` in MFEM
/// `H1_BergotPyramidElement(p)`: the **GLL-barycentric** node placement
/// (`fe/fe_h1.cpp`, `H1_BergotPyramidElement::H1_BergotPyramidElement`).
///
/// Base-layer nodes sit at the 1-D closed Gauss–Lobatto points `cp` on
/// `[0,1]`; tri-face and interior nodes are the barycentric combinations of
/// the `cp` values that MFEM's constructor computes.  The formulas below are
/// 1:1 ports of that constructor (each face keeps MFEM's own expression, so
/// the floating-point results match the C++ element bit for bit).
fn h1_pyramid_node_position(p: usize, cp: &[f64], l: [usize; 3]) -> [f64; 3] {
    let [i, j, k] = l;
    if k == 0 {
        // Base plane: vertices, base edges and the quad face all sit at
        // plain tensor `cp` values (cp[0] = 0 exactly).
        return [cp[i], cp[j], cp[0]];
    }
    if (i, j, k) == (0, 0, p) {
        return [cp[0], cp[0], cp[p]]; // apex
    }
    // Apex edges (checked before the tri faces: e.g. an edge (2,4) label
    // also satisfies the tri-face conditions).
    if i == 0 && j == 0 {
        return [cp[0], cp[0], cp[k]]; // (0,4)
    }
    if j == 0 && i + k == p {
        return [cp[i], cp[0], cp[k]]; // (1,4)
    }
    if i == j && i + k == p {
        return [cp[i], cp[j], cp[k]]; // (2,4)
    }
    if i == 0 && j + k == p {
        return [cp[0], cp[j], cp[k]]; // (3,4)
    }
    // Triangular side faces (MFEM's `w`-normalised barycentric placement).
    if j == 0 {
        // (0,1,4)
        let w = cp[i] + cp[k] + cp[p - i - k];
        return [cp[i] / w, cp[0], cp[k] / w];
    }
    if i == p - k {
        // (1,2,4)
        let w = cp[j] + cp[k] + cp[p - j - k];
        return [1.0 - cp[k] / w, cp[j] / w, cp[k] / w];
    }
    if j == p - k {
        // (2,3,4)
        let w = cp[i] + cp[k] + cp[p - i - k];
        return [cp[i] / w, 1.0 - cp[k] / w, cp[k] / w];
    }
    if i == 0 {
        // (3,0,4)
        let w = cp[j] + cp[k] + cp[p - j - k];
        return [cp[0], cp[j] / w, cp[k] / w];
    }
    // Interior: MFEM's double-barycentric placement.
    let wjk = cp[j] + cp[k] + cp[p - j - k];
    let wik = cp[i] + cp[k] + cp[p - i - k];
    let w = wik * wjk * cp[p - k];
    [
        cp[i] * (cp[j] + cp[p - j - k]) / w,
        cp[j] * (cp[i] + cp[p - i - k]) / w,
        cp[k] * cp[p - k] / w,
    ]
}

/// Shifted Legendre polynomials `P̃_n(x) = P_n(2x−1)` on `[0,1]` and their
/// `x`-derivatives (MFEM `Poly_1D::CalcLegendre` verbatim).
///
/// Shared with the Fuentes family ([`super::pyramid_fuentes`]), which feeds it
/// through `CalcScaledLegendre`.
pub(crate) fn calc_legendre_d(p: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0; p + 1];
    let mut d = vec![0.0; p + 1];
    let z;
    u[0] = 1.0;
    d[0] = 0.0;
    if p == 0 {
        return (u, d);
    }
    u[1] = 2.0 * x - 1.0;
    z = u[1];
    d[1] = 2.0;
    for n in 1..p {
        u[n + 1] = ((2 * n + 1) as f64 * z * u[n] - n as f64 * u[n - 1]) / (n + 1) as f64;
        d[n + 1] = (4 * n + 2) as f64 * u[n] + d[n - 1];
    }
    (u, d)
}

/// Shifted Jacobi polynomials `P_n^{(α,0)}(2x−t)` with their `x`- and
/// `t`-derivatives (MFEM `FuentesPyramid::CalcScaledJacobi`, `fe_pyramid.cpp:613`).
///
/// `H1_BergotPyramidElement` only needs the first two outputs (it evaluates at
/// `t = 1`); the Fuentes pyramid ([`super::pyramid_fuentes`]) needs `dudt` as
/// well, so the single implementation lives here and the Bergot path drops the
/// third output.
pub(crate) fn calc_scaled_jacobi(
    p: usize,
    alpha: f64,
    x: f64,
    t: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = vec![0.0; p + 1];
    let mut dudx = vec![0.0; p + 1];
    let mut dudt = vec![0.0; p + 1];
    u[0] = 1.0;
    dudx[0] = 0.0;
    dudt[0] = 0.0;
    if p >= 1 {
        u[1] = (2.0 + alpha) * x - t;
        dudx[1] = 2.0 + alpha;
        dudt[1] = -1.0;
    }
    for i in 2..=p {
        let a = 2.0 * i as f64 * (alpha + i as f64) * (2.0 * i as f64 + alpha - 2.0);
        let b = 2.0 * i as f64 + alpha - 1.0;
        let c = (2.0 * i as f64 + alpha) * (2.0 * i as f64 + alpha - 2.0);
        let d = 2.0 * (alpha + i as f64 - 1.0) * (i - 1) as f64 * (2.0 * i as f64 + alpha);
        u[i] = (b * (c * (2.0 * x - t) + alpha * alpha * t) * u[i - 1] - d * t * t * u[i - 2]) / a;
        dudx[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudx[i - 1]
                        + 2.0 * c * u[i - 1])
                   - d * t * t * dudx[i - 2]) / a;
        dudt[i] = (b * ((c * (2.0 * x - t) + alpha * alpha * t) * dudt[i - 1]
                        + (alpha * alpha - c) * u[i - 1])
                   - d * t * t * dudt[i - 2] - 2.0 * d * t * u[i - 2]) / a;
    }
    (u, dudx, dudt)
}

/// Bergot raw expansion index list: `(i, j, k)` with `k ≤ p − max(i, j)`, in
/// MFEM's enumeration order (`i` outer, `j`, `k`).
fn bergot_lex(p: usize) -> Vec<(usize, usize, usize)> {
    let mut lex = Vec::with_capacity((p + 1) * (p + 2) * (2 * p + 3) / 6);
    for i in 0..=p {
        for j in 0..=p {
            let maxij = i.max(j);
            for k in 0..=p - maxij {
                lex.push((i, j, k));
            }
        }
    }
    lex
}

/// MFEM `H1_BergotPyramidElement(p)` clone on the reference pyramid
/// `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)` — `(p+1)(p+2)(2p+3)/6` DOFs in
/// MFEM's **entity** slot order ([`h1_pyramid_slot_labels`]) at the
/// **GLL-barycentric** node positions ([`h1_pyramid_node_position`]).
///
/// Basis: the nodal basis on those nodes, built exactly like MFEM builds it —
/// the raw collapsed expansion
///
/// ```text
/// u_o = L_i(r) · L_j(s) · P_k^{(2(max(i,j)+1),0)}(z) · (1−z)^max(i,j)
///       r = x/(1−z), s = y/(1−z), o = (i, j, k) in `bergot_lex` order
/// ```
///
/// (`L` shifted Legendre, `P` shifted Jacobi) is evaluated at every node to
/// form the Vandermonde `T(o, m)`, and `T⁻¹` maps `u` to the shape functions
/// (MFEM's `Ti.Factor(T)` / `Ti.Mult(u, shape)`).  At the apex
/// (`|z−1| < 1e-8`) the analytic limits of `u` / `∂u` are used, as in MFEM's
/// `CalcShape`/`CalcDShape`.
///
/// This is MFEM's `H1_FECollection(p, 3, GaussLobatto, pyr_type=0)` pyramid —
/// available in fem-rs through the explicit
/// [`PyramidBasisType::Bergot`] opt-out (`H1Space::with_pyramid_basis`,
/// `Mesh::set_curvature_with_pyramid_basis`); since D347 the *default* is
/// Fuentes.  It differs from the equispaced [`PyramidPk`] (same slot
/// arrangement) from p = 3 on, where the GLL-barycentric nodes leave the
/// equispaced lattice; and from MFEM's *default* Fuentes pyramid
/// (`pyr_type=1`, `p(p²+3)+1` DOFs,
/// [`super::pyramid_fuentes::H1FuentesPyramidPk`]).
pub struct H1PyramidPk {
    inner: std::sync::Arc<H1PyramidPkInner>,
}

struct H1PyramidPkInner {
    order: usize,
    /// Slot reference positions in entity order (GLL-barycentric).
    nodes: Vec<[f64; 3]>,
    lex: Vec<(usize, usize, usize)>,
    /// `φ_m = Σ_o ti[m·n + o] · u_o(x)` — row-major `T⁻¹`.
    ti: Vec<f64>,
}

impl H1PyramidPk {
    /// Build (or fetch from the per-order cache) the element.
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be >= 1");
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<H1PyramidPkInner>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let inner = {
            let mut m = cache.lock().expect("H1PyramidPk cache poisoned");
            m.entry(p).or_insert_with(|| Arc::new(h1_pyramid_pk_build(p))).clone()
        };
        Self { inner }
    }

    /// MFEM's `H1_BergotPyramidElement(p)` slot table — see
    /// [`h1_pyramid_slot_labels`].
    pub fn slot_labels(p: usize) -> Vec<[usize; 3]> {
        h1_pyramid_slot_labels(p)
    }
}

fn h1_pyramid_pk_build(p: usize) -> H1PyramidPkInner {
    // Closed Gauss–Lobatto points on [0,1] (MFEM `poly1d.ClosedPoints(p,
    // GaussLobatto)` — the `btype` MFEM's `H1_BergotPyramidElement` uses).
    let (g, _w) = crate::quadrature::gauss_lobatto_arbitrary(p + 1);
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    let labels = h1_pyramid_slot_labels(p);
    let nodes: Vec<[f64; 3]> = labels
        .iter()
        .map(|&l| h1_pyramid_node_position(p, &cp, l))
        .collect();
    let lex = bergot_lex(p);
    let n = nodes.len();
    debug_assert_eq!(lex.len(), n);
    let mut t = nalgebra::DMatrix::<f64>::zeros(n, n);
    let apex_tol = 1e-8_f64;
    for (m, node) in nodes.iter().enumerate() {
        let (x, y, z) = (node[0], node[1], node[2]);
        let (u, _du) = bergot_raw(p, &lex, x, y, z, apex_tol);
        for (o, _) in lex.iter().enumerate() {
            t[(o, m)] = u[o];
        }
    }
    let ti_m = t.try_inverse().expect("H1PyramidPk: singular Vandermonde matrix");
    let mut ti = vec![0.0; n * n];
    for m in 0..n {
        for o in 0..n {
            ti[m * n + o] = ti_m[(m, o)];
        }
    }
    H1PyramidPkInner { order: p, nodes, lex, ti }
}

/// Evaluate MFEM's Bergot raw expansion `u` (and optionally its gradients,
/// when `du` is `Some`) at `(x, y, z)`, including the apex-limit paths of
/// `H1_BergotPyramidElement::CalcShape`/`CalcDShape`.
fn bergot_raw(
    p: usize,
    lex: &[(usize, usize, usize)],
    x: f64,
    y: f64,
    z: f64,
    apex_tol: f64,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let n = lex.len();
    let mut u = vec![0.0; n];
    let mut du = Some(vec![0.0; n * 3]);
    if (z - 1.0).abs() < apex_tol {
        // Apex limits along the centre line (MFEM's precomputed polynomials).
        for (o, &(i, j, k)) in lex.iter().enumerate() {
            let k = k as f64;
            if i == 0 && j == 0 {
                u[o] = ((k + 3.0) * k + 2.0) / 2.0;
                du.as_mut().unwrap()[o * 3 + 2] =
                    (((k + 6.0) * k + 11.0) * k + 6.0) * k / 6.0;
            } else if i == 1 && j == 0 {
                du.as_mut().unwrap()[o * 3] =
                    ((((k + 10.0) * k + 35.0) * k + 50.0) * k + 24.0) / 24.0;
            } else if i == 0 && j == 1 {
                du.as_mut().unwrap()[o * 3 + 1] =
                    ((((k + 10.0) * k + 35.0) * k + 50.0) * k + 24.0) / 24.0;
            }
        }
        return (u, du);
    }
    let r = if z < 1.0 { x / (1.0 - z) } else { 0.0 };
    let s = if z < 1.0 { y / (1.0 - z) } else { 0.0 };
    let (lx, dlx) = calc_legendre_d(p, r);
    let (ly, dly) = calc_legendre_d(p, s);
    // The z-Jacobi factor only depends on m = max(i, j): precompute per m.
    let mut js: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(p + 1);
    for m in 0..=p {
        let (u_m, du_m, _du_dt) = calc_scaled_jacobi(p - m, 2.0 * (m as f64 + 1.0), z, 1.0);
        js.push((u_m, du_m));
    }
    let one_minus_z = 1.0 - z;
    for (o, &(i, j, k)) in lex.iter().enumerate() {
        let m = i.max(j);
        let (jz, djz) = &js[m];
        let omz_m = one_minus_z.powi(m as i32);
        u[o] = lx[i] * ly[j] * jz[k] * omz_m;
        if let Some(du) = du.as_mut() {
            let omz_m1 = one_minus_z.powi(m as i32 - 1);
            let omz_m2 = one_minus_z.powi(m as i32 - 2);
            du[o * 3] = dlx[i] * ly[j] * jz[k] * omz_m1;
            du[o * 3 + 1] = lx[i] * dly[j] * jz[k] * omz_m1;
            du[o * 3 + 2] = lx[i] * ly[j] * djz[k] * omz_m
                + (x * dlx[i] * ly[j] + y * lx[i] * dly[j]) * jz[k] * omz_m2
                - if m > 0 {
                    m as f64 * lx[i] * ly[j] * jz[k] * omz_m1
                } else {
                    0.0
                };
        }
    }
    (u, du)
}

impl ReferenceElement for H1PyramidPk {
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
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let n = self.inner.nodes.len();
        let (u, _) = bergot_raw(self.inner.order, &self.inner.lex, x, y, z, 1e-8);
        for m in 0..n {
            let mut acc = 0.0;
            for o in 0..n {
                acc += self.inner.ti[m * n + o] * u[o];
            }
            values[m] = acc;
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let (x, y, z) = (xi[0], xi[1], xi[2]);
        let n = self.inner.nodes.len();
        let (_, du) = bergot_raw(self.inner.order, &self.inner.lex, x, y, z, 1e-8);
        let du = du.expect("bergot_raw always returns gradients");
        for m in 0..n {
            let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
            for o in 0..n {
                let c = self.inner.ti[m * n + o];
                gx += c * du[o * 3];
                gy += c * du[o * 3 + 1];
                gz += c * du[o * 3 + 2];
            }
            grads[m * 3] = gx;
            grads[m * 3 + 1] = gy;
            grads[m * 3 + 2] = gz;
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        self.inner.nodes.iter().map(|c| vec![c[0], c[1], c[2]]).collect()
    }
}

impl PyramidPk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "order must be ≥ 1");
        let mut layer_offset = Vec::with_capacity(p + 2);
        let mut off = 0usize;
        for k in 0..=p {
            layer_offset.push(off);
            let n = p - k + 1;
            off += n * n;
        }
        layer_offset.push(off);
        Self {
            order: p,
            layer_offset,
        }
    }

    fn layer_n(&self, k: usize) -> usize {
        self.order - k + 1
    }

    fn n_dofs_total(&self) -> usize {
        *self.layer_offset.last().unwrap()
    }

    fn dof_index(&self, k: usize, i: usize, j: usize) -> usize {
        let n = self.layer_n(k);
        self.layer_offset[k] + j * n + i
    }

    #[allow(dead_code)]
    fn for_each_dof<F: FnMut(usize, usize, usize, usize)>(&self, mut f: F) {
        let p = self.order;
        for k in 0..=p {
            let n = p - k + 1;
            for j in 0..n {
                for i in 0..n {
                    f(k, i, j, self.dof_index(k, i, j));
                }
            }
        }
    }
}

impl ReferenceElement for PyramidPk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        self.n_dofs_total()
    }

    fn eval_basis(&self, xi: &[f64], values: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let p = self.order;

        if (z - 1.0).abs() < 1e-14 {
            for v in values.iter_mut() {
                *v = 0.0;
            }
            let apex = self.dof_index(p, 0, 0);
            values[apex] = 1.0;
            return;
        }

        let inv_one_minus_z = 1.0 / (1.0 - z);
        let r = x * inv_one_minus_z;
        let s = y * inv_one_minus_z;

        for k in 0..=p {
            let lz = lagrange_1d_val(k, p, z);
            let layer_deg = p - k;
            let n = layer_deg + 1;
            for j in 0..n {
                let ls = lagrange_1d_val(j, layer_deg, s);
                for i in 0..n {
                    let lr = lagrange_1d_val(i, layer_deg, r);
                    values[self.dof_index(k, i, j)] = lz * lr * ls;
                }
            }
        }
    }

    fn eval_grad_basis(&self, xi: &[f64], grads: &mut [f64]) {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let p = self.order;

        if (z - 1.0).abs() < 1e-14 {
            for g in grads.iter_mut() {
                *g = 0.0;
            }
            return;
        }

        let inv_one_minus_z = 1.0 / (1.0 - z);

        for k in 0..=p {
            let lz = lagrange_1d_val(k, p, z);
            let dlz = lagrange_1d_deriv(k, p, z);
            let layer_deg = p - k;
            let n = layer_deg + 1;

            let (mut lr, mut ls) = if n > 1 {
                (vec![0.0; n], vec![0.0; n])
            } else {
                // Special case for apex layer (n=1): constant basis functions
                // pre-evaluated lr/ls arrays
                let mut lr = Vec::with_capacity(1);
                let mut ls = Vec::with_capacity(1);
                lr.push(1.0);
                ls.push(1.0);
                (lr, ls)
            };

            let r = x * inv_one_minus_z;
            let s_val = y * inv_one_minus_z;

            for i in 0..n {
                lr[i] = lagrange_1d_val(i, layer_deg, r);
            }
            for j in 0..n {
                ls[j] = lagrange_1d_val(j, layer_deg, s_val);
            }

            let (mut dlr, mut dls) = if n > 1 {
                (vec![0.0; n], vec![0.0; n])
            } else {
                (vec![0.0; 1], vec![0.0; 1])
            };

            for i in 0..n {
                dlr[i] = lagrange_1d_deriv(i, layer_deg, r);
            }
            for j in 0..n {
                dls[j] = lagrange_1d_deriv(j, layer_deg, s_val);
            }

            for j in 0..n {
                for i in 0..n {
                    let dof = self.dof_index(k, i, j);
                    let lr_i = lr[i];
                    let ls_j = ls[j];
                    let dlr_i = dlr[i];
                    let dls_j = dls[j];

                    grads[dof * 3] = lz * dlr_i * ls_j * inv_one_minus_z;
                    grads[dof * 3 + 1] = lz * lr_i * dls_j * inv_one_minus_z;
                    grads[dof * 3 + 2] = dlz * lr_i * ls_j
                        + lz * dlr_i * ls_j * x * inv_one_minus_z * inv_one_minus_z
                        + lz * lr_i * dls_j * y * inv_one_minus_z * inv_one_minus_z;
                }
            }
        }
    }

    fn quadrature(&self, order: u8) -> QuadratureRule {
        pyramid_rule(order)
    }

    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order;
        let mut coords = Vec::with_capacity(self.n_dofs_total());
        for k in 0..=p {
            let z = k as f64 / p as f64;
            let n = p - k + 1;
            for j in 0..n {
                let y = j as f64 / p as f64;
                for i in 0..n {
                    let x = i as f64 / p as f64;
                    coords.push(vec![x, y, z]);
                }
            }
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_pou(elem: &dyn ReferenceElement) {
        let order = elem.order();
        let rule = elem.quadrature((2 * order + 2).min(15));
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "POU failed at {:?}: sum={s}", pt);
        }
    }

    fn check_grad_zero(elem: &dyn ReferenceElement) {
        let dim = elem.dim() as usize;
        let order = elem.order();
        let rule = elem.quadrature((2 * order + 2).min(15));
        let mut g = vec![0.0_f64; elem.n_dofs() * dim];
        for pt in &rule.points {
            elem.eval_grad_basis(pt, &mut g);
            for d in 0..dim {
                let s: f64 = (0..elem.n_dofs()).map(|i| g[i * dim + d]).sum();
                assert!(s.abs() < 1e-10, "grad sum d={d} = {s} at {:?}", pt);
            }
        }
    }

    fn check_nodal_interp(elem: &dyn ReferenceElement) {
        let coords = elem.dof_coords();
        let n = elem.n_dofs();
        let mut phi = vec![0.0_f64; n];
        for (i, coord) in coords.iter().enumerate() {
            elem.eval_basis(coord, &mut phi);
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (phi[j] - expected).abs() < 1e-10,
                    "nodal interp: node {i}, basis {j}: expected {expected}, got {}",
                    phi[j]
                );
            }
        }
    }

    #[test]
    fn pyramid_pou() {
        for p in 1..=4 {
            check_pou(&PyramidPk::new(p));
        }
    }
    #[test]
    fn pyramid_grad_zero() {
        for p in 1..=4 {
            check_grad_zero(&PyramidPk::new(p));
        }
    }
    #[test]
    fn pyramid_nodal_interp() {
        for p in 1..=4 {
            check_nodal_interp(&PyramidPk::new(p));
        }
    }
    #[test]
    fn pyramid_n_dofs() {
        assert_eq!(PyramidPk::new(1).n_dofs(), 5);
        assert_eq!(PyramidPk::new(2).n_dofs(), 14);
        assert_eq!(PyramidPk::new(3).n_dofs(), 30);
        // (p+1)(p+2)(2p+3)/6
        assert_eq!(PyramidPk::new(4).n_dofs(), 55);
    }

    #[test]
    fn pyramid_pk_gradient_fd() {
        let h = 1e-7;
        for p in 1..=4 {
            let elem = PyramidPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) = (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * 3],
            );
            let test_pts: &[[f64; 3]] = if p == 1 {
                &[[0.3, 0.2, 0.1]]
            } else {
                &[[0.2, 0.3, 0.15], [0.4, 0.1, 0.25]]
            };
            for pt in test_pts {
                let (x, y, z) = (pt[0], pt[1], pt[2]);
                if x + z > 0.95 || y + z > 0.95 {
                    continue;
                }
                if z > 0.8 {
                    continue;
                }
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x + h, y, z], &mut vx);
                elem.eval_basis(&[x, y + h, z], &mut vy);
                elem.eval_basis(&[x, y, z + h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 3] - fd_x).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i} gx"
                    );
                    assert!(
                        (grads[i * 3 + 1] - fd_y).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i} gy"
                    );
                    assert!(
                        (grads[i * 3 + 2] - fd_z).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i} gz"
                    );
                }
            }
        }
    }

    #[test]
    fn pyramid_basis_type_family_map() {
        // The two arms of MFEM's `pyr_type` switch (`fe_coll.cpp:1976`), with
        // `Default` = Fuentes = `ScalarPyramid::DefaultType` (D347).
        assert_eq!(PyramidBasisType::default(), PyramidBasisType::Fuentes);
        assert_eq!(
            h1_pyramid_element(2, PyramidBasisType::Bergot).n_dofs(),
            14, // (p+1)(p+2)(2p+3)/6
        );
        assert_eq!(
            h1_pyramid_element(2, PyramidBasisType::Fuentes).n_dofs(),
            15, // p(p²+3)+1
        );
        assert_eq!(
            h1_pyramid_element(4, PyramidBasisType::Fuentes).n_dofs(),
            77
        );
        // The default arm is the Fuentes element.
        assert_eq!(
            h1_pyramid_element(3, PyramidBasisType::default()).n_dofs(),
            37
        );
    }

    // ── H1PyramidPk (MFEM H1_BergotPyramidElement) ──────────────────────────

    fn h1_pyr_check_pou(elem: &H1PyramidPk) {
        let order = elem.order();
        let rule = elem.quadrature((2 * order + 2).min(15));
        let mut phi = vec![0.0_f64; elem.n_dofs()];
        for pt in &rule.points {
            elem.eval_basis(pt, &mut phi);
            let s: f64 = phi.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "POU failed at {:?}: sum={s}", pt);
        }
    }

    /// MFEM's Bergot node placement at p = 3: probe-encoded GLL-barycentric
    /// positions (`tmp/d191/probe_p3.txt`, `== bergot p=3`, `enode` table).
    #[test]
    fn h1_pyramid_pk_p3_positions_match_mfem_probe() {
        let elem = H1PyramidPk::new(3);
        let coords = elem.dof_coords();
        let want: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.27639320225002106, 0.0, 0.0],
            [0.72360679774997894, 0.0, 0.0],
            [1.0, 0.27639320225002106, 0.0],
            [1.0, 0.72360679774997894, 0.0],
            [0.27639320225002106, 1.0, 0.0],
            [0.72360679774997894, 1.0, 0.0],
            [0.0, 0.27639320225002106, 0.0],
            [0.0, 0.72360679774997894, 0.0],
            [0.0, 0.0, 0.27639320225002106],
            [0.0, 0.0, 0.72360679774997894],
            [0.72360679774997894, 0.0, 0.27639320225002106],
            [0.27639320225002106, 0.0, 0.72360679774997894],
            [0.72360679774997894, 0.72360679774997894, 0.27639320225002106],
            [0.27639320225002106, 0.27639320225002106, 0.72360679774997894],
            [0.0, 0.72360679774997894, 0.27639320225002106],
            [0.0, 0.27639320225002106, 0.72360679774997894],
            [0.27639320225002106, 0.27639320225002106, 0.0],
            [0.72360679774997894, 0.27639320225002106, 0.0],
            [0.27639320225002106, 0.72360679774997894, 0.0],
            [0.72360679774997894, 0.72360679774997894, 0.0],
            [0.33333333333333331, 0.0, 0.33333333333333331],
            [0.66666666666666674, 0.33333333333333331, 0.33333333333333331],
            [0.33333333333333331, 0.66666666666666674, 0.33333333333333331],
            [0.0, 0.33333333333333331, 0.33333333333333331],
            [0.30710355805557898, 0.30710355805557898, 0.40200377652776603],
        ];
        assert_eq!(coords.len(), want.len());
        for (m, w) in want.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (coords[m][d] - w[d]).abs() < 1e-15,
                    "p3 node {m} axis {d}: got {}, want {}",
                    coords[m][d],
                    w[d],
                );
            }
        }
        // The MFEM slot table must reproduce the probe's entity blocks: the
        // p = 3 tri-face slots 25..29 hold dofs 25..29 (identity here), and
        // the edge-2 block runs v3→v2 (slot 9 carries the cp[1] node).
        let labels = H1PyramidPk::slot_labels(3);
        assert_eq!(labels[9], [1, 3, 0]);
        assert_eq!(labels[10], [2, 3, 0]);
        assert_eq!(labels[21], [1, 1, 0]);
        assert_eq!(labels[25], [1, 0, 1]);
        assert_eq!(labels[29], [1, 1, 1]);
    }

    /// Counts `(p+1)(p+2)(2p+3)/6` and slot-table size agree, p = 1..10.
    #[test]
    fn h1_pyramid_pk_counts() {
        for p in 1..=10usize {
            let n = (p + 1) * (p + 2) * (2 * p + 3) / 6;
            assert_eq!(H1PyramidPk::new(p).n_dofs(), n, "p={p}");
            assert_eq!(h1_pyramid_slot_labels(p).len(), n, "p={p}");
            assert_eq!(bergot_lex(p).len(), n, "p={p}");
        }
    }

    /// Nodal property φ_m(node_l) = δ_ml — the construction is a Vandermonde
    /// inverse over MFEM's Legendre–Jacobi raw expansion, so conditioning is
    /// the same fact MFEM lives with (measured: ≤ 4e-11 residual through
    /// p = 8; p = 9/10 degrade like MFEM's own LU would).
    #[test]
    fn h1_pyramid_pk_nodal_interp() {
        for p in 1..=8usize {
            let elem = H1PyramidPk::new(p);
            let coords = elem.dof_coords();
            let n = elem.n_dofs();
            let mut phi = vec![0.0_f64; n];
            let tol = if p <= 4 { 1e-12 } else { 1e-9 };
            for (l, node) in coords.iter().enumerate() {
                elem.eval_basis(node, &mut phi);
                for (m, v) in phi.iter().enumerate() {
                    let target = if l == m { 1.0 } else { 0.0 };
                    assert!(
                        (v - target).abs() < tol,
                        "p={p}: nodal property failed at node {l}, basis {m}: {v}"
                    );
                }
            }
        }
    }

    /// Partition of unity and constant-annihilating gradients on the pyramid
    /// interior (p = 1..4).
    #[test]
    fn h1_pyramid_pk_pou_and_grad_sum() {
        for p in 1..=4usize {
            let elem = H1PyramidPk::new(p);
            h1_pyr_check_pou(&elem);
            let order = elem.order();
            let rule = elem.quadrature((2 * order + 2).min(15));
            let n = elem.n_dofs();
            let mut g = vec![0.0_f64; n * 3];
            for pt in &rule.points {
                if (pt[2] - 1.0).abs() < 1e-8 {
                    continue; // apex limit: gradients are one-sided there
                }
                elem.eval_grad_basis(pt, &mut g);
                for d in 0..3 {
                    let s: f64 = (0..n).map(|i| g[i * 3 + d]).sum();
                    assert!(s.abs() < 1e-9, "p={p} grad sum d={d} = {s} at {pt:?}");
                }
            }
        }
    }

    /// Apex evaluation returns the apex-vertex indicator (slot 4) — computed
    /// through MFEM's limit path, so this also validates `T⁻¹ u_limit`.
    #[test]
    fn h1_pyramid_pk_apex_eval() {
        for p in 1..=6usize {
            let elem = H1PyramidPk::new(p);
            let n = elem.n_dofs();
            let mut phi = vec![0.0_f64; n];
            elem.eval_basis(&[0.25, 0.25, 1.0], &mut phi);
            for (m, v) in phi.iter().enumerate() {
                let target = if m == 4 { 1.0 } else { 0.0 };
                assert!(
                    (v - target).abs() < 1e-10,
                    "p={p}: apex phi[{m}] = {v} (want {target})"
                );
            }
        }
    }

    /// Gradient finite-difference check on the interior (the chain-rule
    /// differentiation of the collapsed expansion).
    #[test]
    fn h1_pyramid_pk_gradient_fd() {
        let h = 1e-7;
        for p in 2..=4usize {
            let elem = H1PyramidPk::new(p);
            let n = elem.n_dofs();
            let (mut vc, mut vx, mut vy, mut vz, mut grads) =
                (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n * 3]);
            // Stay clear of the singular faces (z ≤ 0.8 keeps r, s ≤ 1).
            for &(x, y, z) in [(0.2, 0.3, 0.15), (0.4, 0.1, 0.25)].iter() {
                elem.eval_basis(&[x, y, z], &mut vc);
                elem.eval_basis(&[x + h, y, z], &mut vx);
                elem.eval_basis(&[x, y + h, z], &mut vy);
                elem.eval_basis(&[x, y, z + h], &mut vz);
                elem.eval_grad_basis(&[x, y, z], &mut grads);
                for i in 0..n {
                    let fd_x = (vx[i] - vc[i]) / h;
                    let fd_y = (vy[i] - vc[i]) / h;
                    let fd_z = (vz[i] - vc[i]) / h;
                    assert!(
                        (grads[i * 3] - fd_x).abs() < 1e-4
                            && (grads[i * 3 + 1] - fd_y).abs() < 1e-4
                            && (grads[i * 3 + 2] - fd_z).abs() < 1e-4,
                        "p={p} ({x},{y},{z}) i={i}: fd=({fd_x},{fd_y},{fd_z}) an=({},{},{})",
                        grads[i * 3], grads[i * 3 + 1], grads[i * 3 + 2],
                    );
                }
            }
        }
    }
}
