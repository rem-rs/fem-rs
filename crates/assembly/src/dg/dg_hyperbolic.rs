//! DG time-domain operator for hyperbolic conservation laws.
//!
//! Provides [`FluxFunction`] trait, [`EulerFlux`], [`EulerFlux3`],
//! [`RusanovFlux`], and [`DgHyperbolicConservationLaws`].
//!
//! ## Reference
//! MFEM examples/ex18.hpp — DGHyperbolicConservationLaws

use nalgebra as na;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use fem_element::reference::{QuadratureRule, ReferenceElement};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::lagrange::factory::TriPk;
use fem_element::lagrange::tri::{TriP1};
use fem_element::lagrange::QuadL2GL;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::element_jacobian_at;

use super::dg_base::{
    face_point_geom, face_point_geom_3d_face, face_type_of, ref_elem_face, ref_elem_vol,
};

/// Element shape for dispatching Tri3/Quad4 (2-D) and Tet4/Hex8 (3-D) code
/// paths.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ElemShape { Tri, Quad, Tet, Hex }/// Physical flux function for hyperbolic conservation laws.
pub trait FluxFunction: Send + Sync {
    fn num_equations(&self) -> usize;
    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]);
    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64;
    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64>;
}

// ─── EulerFlux ──────────────────────────────────────────────────────────────────

/// Convert conserved variables to primitive variables.
fn cons_to_prim(q: &[f64], gamma: f64) -> (f64, f64, f64, f64) {
    let rho = q[0].max(1e-14);
    let u = q[1] / rho;
    let v = q[2] / rho;
    let ke = 0.5 * rho * (u * u + v * v);
    let p = ((gamma - 1.0) * (q[3] - ke)).max(1e-14);
    (rho, u, v, p)
}

/// Convert primitive variables to conserved variables.
#[allow(dead_code)]
fn prim_to_cons(rho: f64, u: f64, v: f64, p: f64, gamma: f64) -> [f64; 4] {
    let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    [rho, rho * u, rho * v, e]
}

/// 2-D compressible Euler flux (4 equations).
///
/// Conserved variables: [ρ, ρu, ρv, E]
/// γ (specific heat ratio) defaults to 1.4 (air).
pub struct EulerFlux {
    pub gamma: f64,
}

impl Default for EulerFlux {
    fn default() -> Self {
        Self { gamma: 1.4 }
    }
}

impl FluxFunction for EulerFlux {
    fn num_equations(&self) -> usize {
        4
    }

    fn compute_flux(&self, state: &[f64], _point: &[f64], flux_out: &mut [f64]) {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        // flux_out interleaved by dim then equation:
        //   [F_x[ρ], F_y[ρ], F_x[ρu], F_y[ρu], F_x[ρv], F_y[ρv], F_x[E], F_y[E]]
        let E = state[3];
        flux_out[0] = state[1];                         // F_x[ρ]:  ρu
        flux_out[1] = state[2];                         // F_y[ρ]:  ρv
        flux_out[2] = rho * u * u + p;                  // F_x[ρu]: ρu² + p
        flux_out[3] = rho * u * v;                      // F_y[ρu]: ρuv
        flux_out[4] = rho * u * v;                      // F_x[ρv]: ρuv
        flux_out[5] = rho * v * v + p;                  // F_y[ρv]: ρv² + p
        flux_out[6] = u * (E + p);                      // F_x[E]:  u(E + p)
        flux_out[7] = v * (E + p);                      // F_y[E]:  v(E + p)
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        let (rho, u, v, p) = cons_to_prim(state, self.gamma);
        let a = (self.gamma * p / rho).sqrt();
        // MFEM EulerFlux::ComputeFluxDotN (fem/hyperbolic.cpp): the maximum
        // characteristic speed is the NORMAL fluid speed |u·n|/|n| plus the
        // sound speed — NOT the full speed |u| + a.
        let un = u * normal[0] + v * normal[1];
        let nnorm = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        let speed = un.abs() / nnorm;
        speed + a
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        rusanov_combine(self, ql, qr, normal)
    }
}

// ─── RusanovFlux ────────────────────────────────────────────────────────────────

/// Rusanov (local Lax-Friedrichs) numerical flux.
///
/// Wraps any `FluxFunction` — delegates `compute_flux` and `max_speed`
/// to the inner function, and `numerical_flux` calls `inner.numerical_flux`.
pub struct RusanovFlux<F: FluxFunction> {
    pub inner: F,
}

impl<F: FluxFunction> FluxFunction for RusanovFlux<F> {
    fn num_equations(&self) -> usize {
        self.inner.num_equations()
    }

    fn compute_flux(&self, state: &[f64], point: &[f64], flux_out: &mut [f64]) {
        self.inner.compute_flux(state, point, flux_out);
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        self.inner.max_speed(state, normal)
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        self.inner.numerical_flux(ql, qr, normal)
    }
}

/// The shared Rusanov (local Lax-Friedrichs) combination — the single
/// arithmetic source for every [`FluxFunction`]'s `numerical_flux`:
///
/// ```text
/// f̂ = ½(F(qL)·n̂ + F(qR)·n̂) − ½·c·(qR − qL),   c = max speed over qL, qR
/// ```
///
/// The caller folds the face measure `|nor|` into the quadrature weight
/// (D799-3), so the combination receives the **unit** normal `n̂` and
/// `w·f̂ = ipw·(½(F·nor_L + F·nor_R) − ½·c·|nor|·(qR − qL))` — MFEM's
/// `RusanovFlux::Eval` (hyperbolic.cpp:763) with its `scaledMaxE` folded out
/// of the first half and into the weight.
fn rusanov_combine(ff: &dyn FluxFunction, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
    let neq = ff.num_equations();
    let dim = normal.len();
    let zero = vec![0.0_f64; dim];
    let mut fl = vec![0.0_f64; neq * dim];
    let mut fr = vec![0.0_f64; neq * dim];
    ff.compute_flux(ql, &zero, &mut fl);
    ff.compute_flux(qr, &zero, &mut fr);
    // F_n = Σ_d F[eq, d] · n̂_d  (component-wise for each equation)
    let mut fnl = vec![0.0_f64; neq];
    let mut fnr = vec![0.0_f64; neq];
    for eq in 0..neq {
        for d in 0..dim {
            fnl[eq] += fl[eq * dim + d] * normal[d];
            fnr[eq] += fr[eq * dim + d] * normal[d];
        }
    }
    let c = ff.max_speed(ql, normal).max(ff.max_speed(qr, normal));
    // ½(F_n(L) + F_n(R)) - ½·c·(qR - qL)
    let mut f = vec![0.0_f64; neq];
    for eq in 0..neq {
        f[eq] = 0.5 * (fnl[eq] + fnr[eq]) - 0.5 * c * (qr[eq] - ql[eq]);
    }
    f
}

// ─── EulerFlux3 ─────────────────────────────────────────────────────────────────

/// 3-D conserved → primitive conversion (ρ, u, v, w, p) — MFEM
/// `EulerFlux::ComputeFlux`'s primitives (hyperbolic.cpp:1277):
/// `ke = ½|m|²/ρ`, `p = (γ−1)(E − ke)`.
fn cons_to_prim3(q: &[f64], gamma: f64) -> (f64, f64, f64, f64, f64) {
    let rho = q[0].max(1e-14);
    let u = q[1] / rho;
    let v = q[2] / rho;
    let w = q[3] / rho;
    let ke = 0.5 * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) / rho;
    let p = ((gamma - 1.0) * (q[4] - ke)).max(1e-14);
    (rho, u, v, w, p)
}

/// 3-D compressible Euler flux (5 equations) — the `dim = 3` arm of MFEM's
/// dim-generic `EulerFlux` (ex18 constructs `EulerFlux(mesh.Dimension(), γ)`;
/// the 2-D [`EulerFlux`] here predates the D816-2 3-D operator arms).
///
/// Conserved variables: [ρ, ρu, ρv, ρw, E], flux layout `flux_out[eq·3 + d]`.
/// γ (specific heat ratio) defaults to 1.4 (air).
pub struct EulerFlux3 {
    pub gamma: f64,
}

impl Default for EulerFlux3 {
    fn default() -> Self {
        Self { gamma: 1.4 }
    }
}

impl FluxFunction for EulerFlux3 {
    fn num_equations(&self) -> usize {
        5
    }

    fn compute_flux(&self, state: &[f64], _point: &[f64], flux_out: &mut [f64]) {
        // MFEM EulerFlux::ComputeFlux (hyperbolic.cpp:1277), dim = 3:
        //   F = [m; m⊗m/ρ + pI; m·H],  H = (E + p)/ρ
        let rho = state[0].max(1e-14);
        let m = [state[1], state[2], state[3]];
        let energy = state[4];
        let ke = 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / rho;
        let p = ((self.gamma - 1.0) * (energy - ke)).max(1e-14);
        let h = (energy + p) / rho;
        for d in 0..3 {
            flux_out[0 * 3 + d] = m[d]; // F[ρ] = m
            for i in 0..3 {
                flux_out[(1 + i) * 3 + d] = m[i] * m[d] / rho; // ρu uᵀ
            }
            flux_out[(1 + d) * 3 + d] += p; // + p on the diagonal
            flux_out[4 * 3 + d] = m[d] * h; // F[E] = m·H
        }
    }

    fn max_speed(&self, state: &[f64], normal: &[f64]) -> f64 {
        // MFEM EulerFlux::ComputeFluxDotN's speed (hyperbolic.cpp:1326): the
        // NORMAL fluid speed |u·n̂| plus the sound speed √(γp/ρ).
        let (_rho, u, v, w, p) = cons_to_prim3(state, self.gamma);
        let un = u * normal[0] + v * normal[1] + w * normal[2];
        let nnorm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
            .sqrt()
            .max(1e-30);
        let a = (self.gamma * p / _rho).sqrt();
        un.abs() / nnorm + a
    }

    fn numerical_flux(&self, ql: &[f64], qr: &[f64], normal: &[f64]) -> Vec<f64> {
        rusanov_combine(self, ql, qr, normal)
    }
}

// ─── InteriorFace ─────────────────────────────────────────────────────────────

/// Face data for an interior face shared by two elements.
///
/// D799-3: the geometry is the **isoparametric** face geometry of MFEM's
/// `HyperbolicFormIntegrator::AssembleFaceVector` (hyperbolic.cpp:177) — per
/// face quadrature point, `Tr.SetAllIntPoints(&ip)` +
/// `nor = CalcOrtho(Tr.Jacobian())`.  `nor_qp` holds the *unit* normals
/// outward from `elem_l` and `qp_weights` the matching measures `ip.weight·|nor|`
/// (MFEM carries `|nor|` inside `nor` and uses the bare `ip.weight`; the product
/// `qp_weights[q]·nor_qp[q]` is exactly `ip.weight·nor`), so a straight edge
/// gives the same numbers as the pre-fix chord route.
#[allow(dead_code)]
struct InteriorFace {
    elem_l: usize,
    elem_r: usize,
    /// Unit outward normal from `elem_l` at each face quadrature point.
    /// 2-D edges carry `[nx, ny, 0]`; 3-D faces the full `[nx, ny, nz]`.
    nor_qp: Vec<[f64; 3]>,
    /// The element reference point (MFEM `GetElement1IntPoint()`): 2-D
    /// `[ξ, η]` padded with `0`; 3-D `[ξ, η, ζ]`.
    qp_ref_l: Vec<[f64; 3]>,
    qp_ref_r: Vec<[f64; 3]>,
    qp_weights: Vec<f64>,
    basis_l: Vec<Vec<f64>>,
    basis_r: Vec<Vec<f64>>,
}

// ─── BoundaryFace ─────────────────────────────────────────────────────────────

/// Face data for a boundary face (same isoparametric convention as
/// `InteriorFace`; the mirror-wall reflection needs a *unit* normal).
#[allow(dead_code)]
struct BoundaryFace {
    elem: usize,
    /// Unit outward normal from `elem` at each face quadrature point
    /// (`[nx, ny, 0]` in 2-D).
    nor_qp: Vec<[f64; 3]>,
    qp_ref: Vec<[f64; 3]>,
    qp_weights: Vec<f64>,
    basis: Vec<Vec<f64>>,
}

// ─── DgHyperbolicConservationLaws ─────────────────────────────────────────────

/// DG time-domain operator for hyperbolic conservation laws on triangle meshes.
///
/// Provides element-wise inverse mass matrix, weak divergence, and face-based
/// flux assembly for hyperbolic systems (e.g., Euler equations).
pub struct DgHyperbolicConservationLaws {
    n_elems: usize,
    dofs_per_elem: usize,
    n_eq: usize,
    dim: usize,
    total_dofs: usize,
    invmass: Vec<na::DMatrix<f64>>,
    /// Preassembled weak divergence ∫φ_i·∇φ_j (trial space ByDim layout:
    /// column j*dim+d). Used by `mult` when `preassemble_weakdiv` is set
    /// (MFEM ex18.hpp ComputeWeakDivergence / AddMult_a_ABt path).
    weakdiv: Vec<na::DMatrix<f64>>,
    /// Per-element, per-QP **isoparametric** volume geometry `(det J, J⁻ᵀ)` for
    /// the matrix-free volume term (MFEM `Tr.Weight()` + `CalcPhysDShape`).
    /// `J⁻ᵀ` row-major in the top-left `dim×dim` block.  Empty when
    /// `preassemble_weakdiv` (that path never touches it).
    vol_qp_geom: Vec<Vec<(f64, [[f64; 3]; 3])>>,
    ref_elem: Box<dyn ReferenceElement>,
    elem_shape: ElemShape,
    flux: Box<dyn FluxFunction>,
    interior_faces: Vec<InteriorFace>,
    boundary_faces: Vec<BoundaryFace>,
    max_char_speed: std::cell::Cell<f64>,
    z: RefCell<Vec<f64>>,
    preassemble_weakdiv: bool,
}

// ─── Helper functions ─────────────────────────────────────────────────────────

fn make_ref_elem(mesh: &dyn MeshTopology, order: u8) -> (Box<dyn ReferenceElement>, ElemShape) {
    // D816-2: the 3-D arms take MFEM `DG_FECollection(order, dim)` = the L2
    // collection with GaussLegendre nodes through the shared `ref_elem_vol`
    // single source ([`TetL2GL`]/[`HexL2GL`]); the 2-D arms keep their
    // historical (round-40 / D269) element choices.
    match mesh.element_type(0) {
        ElementType::Quad4 => {
            assert_eq!(order, 1, "Quad4 only supports order=1 currently");
            // MFEM DG_FECollection(order, dim, BasisType::GaussLegendre) uses
            // the Gauss-Legendre nodal basis on [0,1]² — NOT the equally
            // spaced QuadQ1.  With GL nodes the mass matrix is diagonal
            // (C++ invmass = 36·I on this mesh; QuadQ1 gave a full 144/-72
            // matrix → 10× larger dudt → NaN).
            (Box::new(QuadL2GL::new(1)), ElemShape::Quad)
        }
        ElementType::Tet4 => (ref_elem_vol(ElementType::Tet4, order), ElemShape::Tet),
        ElementType::Hex8 => (ref_elem_vol(ElementType::Hex8, order), ElemShape::Hex),
        _ => {
            match order {
                1 => (Box::new(TriP1), ElemShape::Tri),
                2 => (Box::new(TriPk::new(2)), ElemShape::Tri),
                3 => (Box::new(TriPk::new(3)), ElemShape::Tri),
                _ => (Box::new(TriP1), ElemShape::Tri),
            }
        }
    }
}

/// Element Jacobian at a quadrature point, **SIGNED** — D696 batch 4: the
/// returned det is the quadrature measure consumed as `w_q·detJ` (MFEM
/// hyperbolic.cpp:103/165 `ip.weight * Tr.Weight()`, signed).
///
/// D799-3: MFEM's `Tr.Weight()`/`Tr.Jacobian()` are the element's
/// **isoparametric** transformation — the order-`g` curved map on a curved
/// mesh, the P1/bilinear map on a straight one.  The pre-fix helpers built the
/// Jacobian from raw corner differences (`tri3_jac_at_qp`) or from a single
/// centroid bilinear value (`quad4_jac_at_qp` + `elem_centroid_jac`), which
/// agreed with MFEM only on affine tets and parallelograms.  Returns
/// `(detJ, J^{-T})` with `J^{-T}` row-major in the top-left `dim×dim` block
/// (`jit[d][k]`), identity-padded (D816-2 added the 3-D adjugate arm; the 2-D
/// arm keeps its historical explicit formula, so the 2-D path is
/// bit-compatible with the pre-D816 `[f64; 4]` layout).
// MFEM: ElementTransformation::Weight + Jacobian
fn elem_jac_at_qp(mesh: &dyn MeshTopology, elem: u32, xi: &[f64], dim: usize) -> (f64, [[f64; 3]; 3]) {
    let (jac, _xp) = element_jacobian_at(mesh, elem, xi, dim);
    let det = jac.determinant();
    let inv_det = 1.0 / det;
    if dim == 2 {
        // J^{-1} = (1/det)·[[J22,-J12],[-J21,J11]] for the 2×2 case, then
        // transposed: rows of J^{-T} = [J11/det, -J10/det], [-J01/det, J00/det].
        let mut jit = [[0.0_f64; 3]; 3];
        jit[0][0] = jac[(1, 1)] * inv_det;
        jit[0][1] = -jac[(1, 0)] * inv_det;
        jit[1][0] = -jac[(0, 1)] * inv_det;
        jit[1][1] = jac[(0, 0)] * inv_det;
        jit[2][2] = 1.0;
        (det, jit)
    } else {
        // 3-D adjugate: J^{-T}(d,k) = J^{-1}(k,d) = C_{d,k}/det, where
        // C_{d,k} = (−1)^{d+k}·M_{d,k} is the (d,k) COFACTOR of J and M_{d,k}
        // deletes row d and column k.  (D816-2 red-herring note: the first cut
        // walked the remaining rows/cols in (d+1,d+2) cyclic order — a SWAP
        // when d = 1 or k = 1, flipping that minor's sign; the remaining
        // indices must be in increasing order for the (−1)^{d+k} sign.)
        let g = |r: usize, c: usize| jac[(r, c)];
        let mut jit = [[0.0_f64; 3]; 3];
        for d in 0..3 {
            for k in 0..3 {
                let (d1, d2) = ((d + 1) % 3, (d + 2) % 3);
                let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
                let (d1, d2) = if d1 < d2 { (d1, d2) } else { (d2, d1) };
                let (k1, k2) = if k1 < k2 { (k1, k2) } else { (k2, k1) };
                let minor = g(d1, k1) * g(d2, k2) - g(d1, k2) * g(d2, k1);
                let sign = if (d + k) % 2 == 0 { 1.0 } else { -1.0 };
                jit[d][k] = sign * minor * inv_det;
            }
        }
        (det, jit)
    }
}

/// Compute element-wise inverse mass matrix M_e⁻¹ in physical space.
/// M_e[i,j] = Σ_q w_q · detJ(ξ_q) · φ_i(ξ_q) · φ_j(ξ_q)   (detJ signed — D696)
fn compute_inv_mass(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let _ = shape;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut invmass = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let mut m = na::DMatrix::<f64>::zeros(dp, dp);
        for q in 0..n_qp {
            let xi = &qr.points[q];
            let det_j = elem_jac_at_qp(mesh, e as u32, xi, dim).0;
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            for i in 0..dp {
                for j in 0..dp {
                    m[(i, j)] += w * phi[i] * phi[j];
                }
            }
        }
        let chol = m.cholesky().expect("Mass matrix must be SPD");
        invmass.push(chol.inverse());
    }
    invmass
}

/// Compute element-wise weak divergence matrix in physical space.
/// weakdiv[e][i, j*dim + d] = Σ_q w_q · detJ(ξ_q) · φ_i(ξ_q) · (J^{-T}(ξ_q) · ∇ξ_φ_j)_d
fn compute_weak_div(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement, n_elems: usize, shape: ElemShape) -> Vec<na::DMatrix<f64>> {
    let dp = ref_elem.n_dofs();
    let dim = mesh.dim() as usize;
    let _ = shape;
    let q_order = 2 * ref_elem.order();
    let qr = ref_elem.quadrature(q_order);
    let n_qp = qr.n_points();
    let mut phi = vec![0.0; dp];
    let mut gphi = vec![0.0; dp * dim];
    let mut weakdiv = Vec::with_capacity(n_elems);
    for e in 0..n_elems {
        let mut wd = na::DMatrix::<f64>::zeros(dp, dp * dim);
        for q in 0..n_qp {
            let xi = &qr.points[q];
            let (det_j, jit) = elem_jac_at_qp(mesh, e as u32, xi, dim);
            let w = qr.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut gphi);
            for i in 0..dp {
                let mut gphys_i = [0.0_f64; 3];
                for d in 0..dim {
                    for k in 0..dim {
                        gphys_i[d] += jit[d][k] * gphi[i * dim + k];
                    }
                }
                for j in 0..dp {
                    for d in 0..dim {
                        wd[(i, j * dim + d)] += w * phi[j] * gphys_i[d];
                    }
                }
            }
        }
        weakdiv.push(wd);
    }
    weakdiv
}

/// Tri3 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2 —
/// listed counter-clockwise, so `(a, b)` runs along the element's own edge.
const TRI3_FACES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Quad4 face patterns: [local_node_a, local_node_b] for faces 0, 1, 2, 3 —
/// counter-clockwise, like [`TRI3_FACES`].
const QUAD4_FACES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

/// Tet4 face patterns: the canonical `FaceVert[lf]` cycles of MFEM's
/// `Geometry::Constants<TETRAHEDRON>` (fem/geom.cpp:987) — outward-oriented.
const TET4_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// Hex8 face patterns: the canonical `FaceVert[lf]` cycles of MFEM's
/// `Geometry::Constants<CUBE>` (fem/geom.cpp:1032) in the crate's `Hex8`
/// vertex order — outward-oriented.
const HEX8_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

/// Local face patterns of the element's shape.
fn shape_faces(shape: ElemShape) -> &'static [[usize; 2]] {
    match shape {
        ElemShape::Quad => &QUAD4_FACES,
        ElemShape::Tri => &TRI3_FACES,
        _ => panic!("shape_faces: 2-D edges requested for shape {shape:?}"),
    }
}

/// Per-quadrature-point **isoparametric** face geometry of element `elem`'s
/// local face `lf`, composed through that element's own map — MFEM's
/// `FaceElementTransformations` (`Tr.SetAllIntPoints(&ip)` +
/// `nor = CalcOrtho(Tr.Jacobian())`, hyperbolic.cpp:232-255).
///
/// Returns `(unit outward normals, measures ipw·|nor|, element reference
/// points)`.  MFEM keeps `|nor|` inside its `nor` and integrates with the bare
/// `ip.weight`; folding the magnitude into the weight (`ipw·|nor|`, with the
/// `[0,1]` face rule) is the same number and leaves a unit normal for the
/// flux and for the reflecting-wall reflection.  `reverse` flips the face
/// parameterisation (the neighbour's edge runs the other way), keeping both
/// elements sampled at the **same** physical face point.
///
/// On a straight edge `|nor|` is the chord length, so every value below is the
/// pre-fix one; on a curved edge it follows the isoparametric edge.
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho
fn face_qp_geom(
    mesh: &dyn MeshTopology,
    elem: u32,
    lf: usize,
    pts: &[f64],
    wts: &[f64],
    reverse: bool,
) -> (Vec<[f64; 3]>, Vec<f64>, Vec<[f64; 3]>) {
    let shape = if mesh.element_nodes(elem).len() == 4 { ElemShape::Quad } else { ElemShape::Tri };
    let en = mesh.element_nodes(elem);
    let (ia, ib) = (shape_faces(shape)[lf][0], shape_faces(shape)[lf][1]);
    let (na, nb) = (en[ia], en[ib]);
    let mut nors = Vec::with_capacity(pts.len());
    let mut ws = Vec::with_capacity(pts.len());
    let mut eips = Vec::with_capacity(pts.len());
    for q in 0..pts.len() {
        let t = if reverse { 1.0 - pts[q] } else { pts[q] };
        let g = face_point_geom(mesh, elem, na, nb, t);
        let nrm = (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1]).sqrt().max(1e-30);
        // D816-2: normals/element points carry a `0` z-component so both
        // dimensional paths share one face-data layout.
        nors.push([g.nor[0] / nrm, g.nor[1] / nrm, 0.0]);
        ws.push(wts[q] * nrm);
        eips.push([g.eip[0], g.eip[1], 0.0]);
    }
    (nors, ws, eips)
}

/// Per-quadrature-point **isoparametric** face geometry of element `elem`'s
/// 3-D face whose node list is `fnodes` — the 3-D counterpart of
/// [`face_qp_geom`], composed through that element's own map via
/// `face_point_geom_3d_face` (MFEM's `FaceElementTransformations`:
/// `Tr.SetAllIntPoints(&ip)` + `nor = CalcOrtho(Tr.Jacobian())`).
///
/// Returns `(unit outward normals, measures ipw·|nor|, element reference
/// points)` exactly like the 2-D helper; `|nor|` is the face area element
/// folded into the weight.  `fnodes` must be the face's canonical node list —
/// the FIRST owning element's `FaceVert[lf]` cycle ([`TET4_FACES`] /
/// [`HEX8_FACES`]) — so both neighbours are composed at the same ξ.
// MFEM: FaceElementTransformations::Jacobian + CalcOrtho
fn face_qp_geom_3d(
    mesh: &dyn MeshTopology,
    elem: u32,
    fnodes: &[u32],
    rule: &QuadratureRule,
) -> (Vec<[f64; 3]>, Vec<f64>, Vec<[f64; 3]>) {
    let n_qp = rule.n_points();
    let mut nors = Vec::with_capacity(n_qp);
    let mut ws = Vec::with_capacity(n_qp);
    let mut eips = Vec::with_capacity(n_qp);
    for q in 0..n_qp {
        let xi = &rule.points[q];
        let g = face_point_geom_3d_face(mesh, elem, fnodes, [xi[0], xi[1]]);
        let nrm =
            (g.nor[0] * g.nor[0] + g.nor[1] * g.nor[1] + g.nor[2] * g.nor[2]).sqrt().max(1e-30);
        nors.push([g.nor[0] / nrm, g.nor[1] / nrm, g.nor[2] / nrm]);
        ws.push(rule.weights[q] * nrm);
        eips.push(g.eip);
    }
    (nors, ws, eips)
}

/// Build interior and boundary face structures.
///
/// 2-D: Tri3 (3 faces) and Quad4 (4 faces) meshes, with periodic pairing.
/// D816-2: 3-D Tet4 (4 triangular faces) and Hex8 (6 quadrilateral faces)
/// meshes via [`build_faces_3d`].
fn build_faces(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement) -> (Vec<InteriorFace>, Vec<BoundaryFace>) {
    if mesh.dim() == 3 {
        build_faces_3d(mesh, ref_elem)
    } else {
        build_faces_2d(mesh, ref_elem)
    }
}

/// Build the 3-D interior and boundary face structures.
///
/// D816-2: the pairing and the face parameterisation follow MFEM's
/// `Mesh::GenerateFaces` (mesh.cpp:8793, `AddTriangleFaceElement` /
/// `AddQuadFaceElement`, mesh.cpp:8713/8741): scanning elements in order,
/// each local face is keyed by its sorted node set; the **first** element
/// touching a face becomes MFEM's `Elem1`, and its canonical `FaceVert[lf]`
/// cycle ([`TET4_FACES`] / [`HEX8_FACES`]) is the face's node list — the same
/// parameterisation MFEM's generated face element carries, so both neighbours
/// compose at the same ξ through `face_point_geom_3d_face`.  The face
/// quadrature rule is MFEM's `IntRules.Get(face_geometry, 2·order)`
/// (`HyperbolicFormIntegrator::AssembleFaceVector` with `IntOrderOffset = 0`;
/// the rule's geometry comes from `face_type_of`, i.e. the face's node count).
/// The 2-D periodic-pairing heuristic has no 3-D counterpart (ex18's periodic
/// problems are 2-D only); every unpaired face is a reflecting wall.
fn build_faces_3d(
    mesh: &dyn MeshTopology,
    ref_elem: &dyn ReferenceElement,
) -> (Vec<InteriorFace>, Vec<BoundaryFace>) {
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements() as u32;
    let face_cycles: &[&[usize]] = match mesh.element_type(0) {
        ElementType::Tet4 => {
            &[&TET4_FACES[0], &TET4_FACES[1], &TET4_FACES[2], &TET4_FACES[3]]
        }
        ElementType::Hex8 => {
            &[
                &HEX8_FACES[0],
                &HEX8_FACES[1],
                &HEX8_FACES[2],
                &HEX8_FACES[3],
                &HEX8_FACES[4],
                &HEX8_FACES[5],
            ]
        }
        other => panic!(
            "DgHyperbolicConservationLaws: unsupported 3-D element type {other:?} \
             (Tet4/Hex8 only; ex18 defines no pyramid problem)"
        ),
    };
    let face_order = 2 * ref_elem.order();

    // MFEM's generated-face table: sorted node key -> (first owner, face list
    // index into `faces3`).
    struct Face3 {
        /// The face's canonical node list: the first owner's `FaceVert[lf]`
        /// cycle — identical to MFEM's `faces[gf]` vertex order.
        nodes: Vec<u32>,
        elem_l: u32,
        elem_r: Option<u32>,
    }
    let mut faces3: Vec<Face3> = Vec::new();
    let mut key_idx: HashMap<Vec<u32>, usize> = HashMap::new();
    for e in 0..n_elems {
        let en = mesh.element_nodes(e);
        for cyc in face_cycles {
            let nodes: Vec<u32> = cyc.iter().map(|&k| en[k]).collect();
            let mut key = nodes.clone();
            key.sort_unstable();
            match key_idx.get(&key) {
                None => {
                    key_idx.insert(key, faces3.len());
                    faces3.push(Face3 { nodes, elem_l: e, elem_r: None });
                }
                Some(&i) => faces3[i].elem_r = Some(e),
            }
        }
    }

    let mut interior = Vec::new();
    let mut boundary = Vec::new();
    let mut phi = vec![0.0; dp];
    for f in &faces3 {
        // The face's own rule: TRIANGLE for a tet's face, SQUARE ([0,1]²
        // tensor Gauss) for a hex's — by the face's node count.
        let ftype = face_type_of(&f.nodes);
        let rule = ref_elem_face(ftype, ref_elem.order()).quadrature(face_order);
        let n_qp = rule.n_points();
        // D799-3 route: the per-QP isoparametric geometry of each side; the
        // flux uses Elem1's (the left element's) `nor`, exactly as MFEM's
        // `AssembleFaceVector` does with `Tr` built from Elem1.
        let (nor_l, w_l, ref_l) = face_qp_geom_3d(mesh, f.elem_l, &f.nodes, &rule);
        if let Some(elem_r) = f.elem_r {
            let (_nor_r, _w_r, ref_r) = face_qp_geom_3d(mesh, elem_r, &f.nodes, &rule);
            let mut basis_l = Vec::with_capacity(n_qp);
            let mut basis_r = Vec::with_capacity(n_qp);
            for q in 0..n_qp {
                ref_elem.eval_basis(&ref_l[q], &mut phi);
                basis_l.push(phi.clone());
                ref_elem.eval_basis(&ref_r[q], &mut phi);
                basis_r.push(phi.clone());
            }
            interior.push(InteriorFace {
                elem_l: f.elem_l as usize,
                elem_r: elem_r as usize,
                nor_qp: nor_l,
                qp_ref_l: ref_l,
                qp_ref_r: ref_r,
                qp_weights: w_l,
                basis_l,
                basis_r,
            });
        } else {
            let mut basis = Vec::with_capacity(n_qp);
            for q in 0..n_qp {
                ref_elem.eval_basis(&ref_l[q], &mut phi);
                basis.push(phi.clone());
            }
            boundary.push(BoundaryFace {
                elem: f.elem_l as usize,
                nor_qp: nor_l,
                qp_ref: ref_l,
                qp_weights: w_l,
                basis,
            });
        }
    }
    (interior, boundary)
}

/// 2-D interior/boundary face builder (Tri3/Quad4 edges, with periodic
/// pairing) — unchanged by D816-2 apart from the shared `[f64; 3]` face-data
/// layout (z padded with 0).
fn build_faces_2d(mesh: &dyn MeshTopology, ref_elem: &dyn ReferenceElement) -> (Vec<InteriorFace>, Vec<BoundaryFace>) {
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements() as u32;
    let n_qp = ((2 * ref_elem.order() + 1) as usize).min(4).max(1);
    let (face_pts, face_wts) = gauss_legendre_01(n_qp);

    // Detect element type from first element's node count
    let shape = if mesh.element_nodes(0).len() == 4 { ElemShape::Quad } else { ElemShape::Tri };
    let face_patterns: &[[usize; 2]] = shape_faces(shape);

    // Record all element edges
    struct ElemEdge {
        nodes: [u32; 2],
        elem: u32,
        local_face: u8,
    }
    let mut edge_list: Vec<ElemEdge> = Vec::new();
    for e in 0..n_elems {
        let enodes = mesh.element_nodes(e);
        for (lf, &[na, nb]) in face_patterns.iter().enumerate() {
            let (n0, n1) = (enodes[na].min(enodes[nb]), enodes[na].max(enodes[nb]));
            edge_list.push(ElemEdge {
                nodes: [n0, n1],
                elem: e,
                local_face: lf as u8,
            });
        }
    }

    // Group by sorted node pair
    let mut edge_map: HashMap<(u32, u32), Vec<(u32, u8)>> = HashMap::new();
    for ee in &edge_list {
        edge_map
            .entry((ee.nodes[0], ee.nodes[1]))
            .or_default()
            .push((ee.elem, ee.local_face));
    }

    let mut interior = Vec::new();
    let mut boundary = Vec::new();
    let mut visited = HashSet::new();
    let mut unpaired: Vec<(u32, u8)> = Vec::new();

    for ee in &edge_list {
        let key = (ee.nodes[0], ee.nodes[1]);
        if visited.contains(&(ee.elem, ee.local_face, key)) {
            continue;
        }
        let entries = &edge_map[&key];
        if entries.len() == 2 {
            let (e0, f0) = entries[0];
            let (e1, f1) = entries[1];
            visited.insert((e0, f0, key));
            visited.insert((e1, f1, key));
            let (elem_l, face_l, elem_r, face_r) = if e0 < e1 {
                (e0 as usize, f0, e1 as usize, f1)
            } else {
                (e1 as usize, f1, e0 as usize, f0)
            };

            // D799-3: the per-QP isoparametric face geometry of each side.  The
            // flux uses Elem1's (the left element's) `nor`, exactly as MFEM's
            // `AssembleFaceVector` does with `Tr` built from Elem1.
            let (nor_l, w_l, ref_l) =
                face_qp_geom(mesh, elem_l as u32, face_l as usize, &face_pts, &face_wts, false);
            // Orientation of the neighbour's edge relative to the left's: the
            // neighbour's own local face starts at a different node when the two
            // edges run in opposite directions.
            let l_first = mesh.element_nodes(elem_l as u32)[face_patterns[face_l as usize][0]];
            let r_first = mesh.element_nodes(elem_r as u32)[face_patterns[face_r as usize][0]];
            let reverse_r = l_first != r_first;
            let (_nor_r, _w_r, ref_r) =
                face_qp_geom(mesh, elem_r as u32, face_r as usize, &face_pts, &face_wts, reverse_r);

            let mut basis_l = Vec::with_capacity(n_qp);
            let mut basis_r = Vec::with_capacity(n_qp);
            let mut phi = vec![0.0; dp];
            for q in 0..n_qp {
                ref_elem.eval_basis(&ref_l[q], &mut phi);
                basis_l.push(phi.clone());
                ref_elem.eval_basis(&ref_r[q], &mut phi);
                basis_r.push(phi.clone());
            }

            interior.push(InteriorFace {
                elem_l,
                elem_r,
                nor_qp: nor_l,
                qp_ref_l: ref_l,
                qp_ref_r: ref_r,
                qp_weights: w_l,
                basis_l,
                basis_r,
            });
        } else {
            let (elem, lf) = entries[0];
            if !visited.contains(&(elem, lf, key)) {
                visited.insert((elem, lf, key));
                unpaired.push((elem, lf));
            }
        }
    }

    // Boundary face geometry for all unpaired edges.  `pair_normal` is the
    // chord normal of the edge's two **corner** nodes — used only by the
    // periodic-pairing heuristic below (a direction test), never as a measure;
    // the assembled flux uses the isoparametric per-QP normals.
    struct BoundInfo { elem: u32, lf: u8, pair_normal: [f64; 2] }
    let mut bound_info: Vec<BoundInfo> = Vec::new();
    for &(elem, lf) in &unpaired {
        let enodes = mesh.element_nodes(elem);
        let lfp = face_patterns[lf as usize];
        let (na, nb) = (enodes[lfp[0]], enodes[lfp[1]]);
        let pa = mesh.node_coords(na);
        let pb = mesh.node_coords(nb);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let length = (dx * dx + dy * dy).sqrt().max(1e-30);
        bound_info.push(BoundInfo { elem, lf, pair_normal: [dy / length, -dx / length] });
    }

    // Detect periodic pairs among boundary faces
    let periodic_idx: Vec<(usize, usize)> = {
        let unpaired_with_normals: Vec<(u32, u8, [f64;2])> = bound_info.iter()
            .map(|b| (b.elem, b.lf, b.pair_normal)).collect();
        detect_periodic_pairs(&unpaired_with_normals)
    };

    // Track which bound_info entries are consumed by periodic pairing
    let mut used_by_periodic = vec![false; bound_info.len()];
    for &(i, j) in &periodic_idx {
        used_by_periodic[i] = true;
        used_by_periodic[j] = true;
        // Create periodic interior face from pair (i, j)
        let bi = &bound_info[i];
        let bj = &bound_info[j];
        let (elem_l, elem_r, face_l, face_r) = {
            // Use the element with smaller index as L, normal outward from L
            if bi.elem < bj.elem {
                (bi.elem as usize, bj.elem as usize, bi.lf, bj.lf)
            } else {
                (bj.elem as usize, bi.elem as usize, bj.lf, bi.lf)
            }
        };
        let (nor_l, w_l, ref_l) =
            face_qp_geom(mesh, elem_l as u32, face_l as usize, &face_pts, &face_wts, false);
        // For a periodic pair the two faces are opposite sides of the domain, so
        // the neighbour's face quadrature points are mirrored.
        let (_nor_r, _w_r, ref_r) =
            face_qp_geom(mesh, elem_r as u32, face_r as usize, &face_pts, &face_wts, true);
        let mut basis_l = Vec::with_capacity(n_qp);
        let mut basis_r = Vec::with_capacity(n_qp);
        let mut phi_l = vec![0.0; dp];
        let mut phi_r = vec![0.0; dp];
        for q in 0..n_qp {
            ref_elem.eval_basis(&ref_l[q], &mut phi_l);
            basis_l.push(phi_l.clone());
            ref_elem.eval_basis(&ref_r[q], &mut phi_r);
            basis_r.push(phi_r.clone());
        }
        interior.push(InteriorFace {
            elem_l, elem_r,
            nor_qp: nor_l,
            qp_ref_l: ref_l, qp_ref_r: ref_r, qp_weights: w_l,
            basis_l, basis_r,
        });
    }

    // Remaining unpaired boundary faces become boundary faces
    for (idx, bi) in bound_info.iter().enumerate() {
        if used_by_periodic[idx] { continue; }
        let (nor_qp, qp_w, qp_ref) =
            face_qp_geom(mesh, bi.elem, bi.lf as usize, &face_pts, &face_wts, false);
        let mut basis = Vec::with_capacity(n_qp);
        let mut phi = vec![0.0; dp];
        for q in 0..n_qp {
            ref_elem.eval_basis(&qp_ref[q], &mut phi);
            basis.push(phi.clone());
        }
        boundary.push(BoundaryFace {
            elem: bi.elem as usize,
            nor_qp,
            qp_ref,
            qp_weights: qp_w,
            basis,
        });
    }

    (interior, boundary)
}

/// Detect periodic face pairs from a list of boundary face candidates.
///
/// Groups boundary faces by their normal direction, then pairs faces from
/// opposite sides (normals that are negatives of each other).  This handles
/// the standard periodic-square.mesh case.
fn detect_periodic_pairs(unpaired: &[(u32, u8, [f64;2])]) -> Vec<(usize, usize)> {
    // Group by normal direction (quantized to ±x, ±y).
    // For a square mesh periodic in both directions, opposite sides
    // have normals that are exact opposites.
    let eps = 1e-10_f64;
    let mut pairs = Vec::new();
    let n = unpaired.len();
    let mut used = vec![false; n];
    for i in 0..n {
        if used[i] { continue; }
        let ni = &unpaired[i].2;
        for j in (i+1)..n {
            if used[j] { continue; }
            let nj = &unpaired[j].2;
            // Check if normals are opposites: ni ≈ -nj
            if (ni[0] + nj[0]).abs() < eps && (ni[1] + nj[1]).abs() < eps {
                pairs.push((i, j));
                used[i] = true;
                used[j] = true;
                break;
            }
        }
    }
    pairs
}


// ─── DgHyperbolicConservationLaws impl ────────────────────────────────────────

impl DgHyperbolicConservationLaws {
    /// Construct a new DG hyperbolic conservation law operator.
    pub fn new(
        mesh: &dyn MeshTopology,
        order: u8,
        flux_fn: Box<dyn FluxFunction>,
        preassemble_weakdiv: bool,
    ) -> Self {
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        let n_eq = flux_fn.num_equations();
        let (ref_elem, elem_shape) = make_ref_elem(mesh, order);
        let dofs_per_elem = ref_elem.n_dofs();
        let total_dofs = n_elems * dofs_per_elem * n_eq;
        let invmass = compute_inv_mass(mesh, &*ref_elem, n_elems, elem_shape);
        let weakdiv = if preassemble_weakdiv {
            compute_weak_div(mesh, &*ref_elem, n_elems, elem_shape)
        } else {
            Vec::new()
        };
        let (interior_faces, boundary_faces) = build_faces(mesh, &*ref_elem);
        // Matrix-free volume term: per-QP isoparametric geometry, built once.
        let vol_qp_geom = if preassemble_weakdiv {
            Vec::new()
        } else {
            let qr = ref_elem.quadrature(2 * ref_elem.order());
            (0..n_elems as u32)
                .map(|e| {
                    qr.points
                        .iter()
                        .map(|xi| elem_jac_at_qp(mesh, e, xi, dim))
                        .collect()
                })
                .collect()
        };
        Self {
            n_elems,
            dofs_per_elem,
            n_eq,
            dim,
            total_dofs,
            invmass,
            weakdiv,
            vol_qp_geom,
            ref_elem,
            elem_shape,
            flux: flux_fn,
            interior_faces,
            boundary_faces,
            max_char_speed: std::cell::Cell::new(0.0),
            z: RefCell::new(vec![0.0; total_dofs]),
            preassemble_weakdiv,
        }
    }

    /// Total number of degrees of freedom.
    pub fn n_dofs(&self) -> usize {
        self.total_dofs
    }

    /// Maximum characteristic speed across all faces (updated during Mult).
    pub fn max_char_speed(&self) -> f64 {
        self.max_char_speed.get()
    }

    /// Compute the DG update: dudt = M⁻¹ (face_fluxes - Div·F(u)).
    ///
    /// `u` is the solution vector in **byNODES (DOF-major)** layout:
    /// `index = e * dp * nq + i * nq + eq`
    /// where `e` = element, `i` = local DOF, `eq` = equation index.
    ///
    /// Algorithm:
    /// 1. Reset workspace `z` and `max_char_speed`.
    /// 2. Interior faces: add numerical flux contribution (±f_hat).
    /// 3. Boundary faces: reflecting wall BC (mirror normal velocity).
    /// 4. Volume term: `z += weakdiv[e] · F_col` (if preassembled).
    /// 5. Apply inverse mass: `dudt_e = invmass[e] · z_e`.
    pub fn mult(&self, u: &[f64], dudt: &mut [f64]) {
        let dp = self.dofs_per_elem;
        let nq = self.n_eq;

        // 1-4. The pre-mass residual (MFEM's NonlinearForm result `z`).
        let mut z = self.z.borrow_mut();
        self.residual_into(u, &mut z);

        // 5. Apply inverse mass matrix
        for e in 0..self.n_elems {
            let base = e * dp * nq;
            let inv = &self.invmass[e];
            for eq in 0..nq {
                let mut zcol = na::DVector::<f64>::zeros(dp);
                for i in 0..dp {
                    zcol[i] = z[base + i * nq + eq];
                }
                let ycol = inv * zcol;
                for i in 0..dp {
                    dudt[base + i * nq + eq] = ycol[i];
                }
            }
        }
    }

    /// The pre-mass residual of [`mult`] — MFEM's `NonlinearForm` result
    /// `z = -⟨f̂(u_h, n), [[v]]⟩_e ± (F(u_h), ∇v)` (ex18.hpp `Mult`'s `z`
    /// before the element-wise M⁻¹), in the same layout as `mult`'s vectors.
    /// Resets and updates `max_char_speed`.
    pub fn mult_residual(&self, u: &[f64], z: &mut [f64]) {
        assert_eq!(z.len(), self.total_dofs, "z must have the operator's dof count");
        self.residual_into(u, z);
    }

    /// Steps 1-4 of [`mult`]: reset `max_char_speed`, zero `z`, accumulate the
    /// interior-face fluxes, the reflecting-wall boundary fluxes and the
    /// volume term.
    fn residual_into(&self, u: &[f64], z: &mut [f64]) {
        let dp = self.dofs_per_elem;
        let nq = self.n_eq;
        let dim = self.dim;

        // 1. Reset workspace
        z.fill(0.0);
        self.max_char_speed.set(0.0);

        // 2. Interior face flux contributions.
        // MFEM `HyperbolicFormIntegrator::AssembleFaceVector`
        // (hyperbolic.cpp:177-270) per face quadrature point:
        //   Tr.SetAllIntPoints(&ip); CalcOrtho(Tr.Jacobian(), nor);
        //   fluxN = numFlux.Eval(state1, state2, nor, Tr)   // nor NOT unit
        //   elvect1 -= ip.weight·sign·shape1·fluxN ;  elvect2 += ip.weight·sign·shape2·fluxN
        // with the `[0,1]` face rule (`Σw = 1/2` on a triangle).  `nor` enters
        // the flux linearly, so carrying a unit normal plus `qp_weights = ipw·|nor|`
        // is the same product; `nor_qp` is that unit normal, per quadrature
        // point and outward from Elem1.
        let mut uL = vec![0.0; nq];
        let mut uR = vec![0.0; nq];
        for face in &self.interior_faces {
            let baseL = face.elem_l * dp * nq;
            let baseR = face.elem_r * dp * nq;
            for q in 0..face.qp_weights.len() {
                uL.fill(0.0);
                uR.fill(0.0);
                for eq in 0..nq {
                    for i in 0..dp {
                        uL[eq] += face.basis_l[q][i] * u[baseL + i * nq + eq];
                        uR[eq] += face.basis_r[q][i] * u[baseR + i * nq + eq];
                    }
                }
                let nor = &face.nor_qp[q];
                let cL = self.flux.max_speed(&uL, nor);
                let cR = self.flux.max_speed(&uR, nor);
                let c = cL.max(cR);
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &uR, nor);
                let w = face.qp_weights[q];
                // Form 2: face = -ĝ·[[v]] = +ĝ·v_L - ĝ·v_R
                for eq in 0..nq {
                    let fw = w * f_hat[eq];
                    for i in 0..dp {
                        z[baseL + i * nq + eq] -= fw * face.basis_l[q][i];
                        z[baseR + i * nq + eq] += fw * face.basis_r[q][i];
                    }
                }
            }
        }

        // 3. Boundary faces (reflecting wall BC).  The mirror is built from the
        // **unit** normal (a scaled normal would not reflect), which is why the
        // face data keeps `nor_qp` normalised and the measure separate.
        // D816-2: dim-generic — the Euler state layout (nq = dim+2) mirrors
        // its `dim` momentum components, which covers the 2-D (nq = 4) and
        // 3-D (nq = 5) arms with the same arithmetic.
        if !self.boundary_faces.is_empty() {
            assert_eq!(
                nq,
                dim + 2,
                "reflecting-wall BC assumes the Euler state layout (nq = dim + 2), got nq = {nq}"
            );
        }
        let mut u_mirror = vec![0.0; nq];
        for face in &self.boundary_faces {
            let base = face.elem * dp * nq;
            for q in 0..face.qp_weights.len() {
                uL.fill(0.0);
                for eq in 0..nq {
                    for i in 0..dp {
                        uL[eq] += face.basis[q][i] * u[base + i * nq + eq];
                    }
                }
                let nor = &face.nor_qp[q];
                let mut vn = 0.0_f64;
                for d in 0..dim {
                    vn += uL[1 + d] * nor[d];
                }
                u_mirror[0] = uL[0];
                for d in 0..dim {
                    u_mirror[1 + d] = uL[1 + d] - 2.0 * vn * nor[d];
                }
                u_mirror[nq - 1] = uL[nq - 1];
                let c = self.flux.max_speed(&uL, nor)
                    .max(self.flux.max_speed(&u_mirror, nor));
                if c > self.max_char_speed.get() { self.max_char_speed.set(c); }
                let f_hat = self.flux.numerical_flux(&uL, &u_mirror, nor);
                let w = face.qp_weights[q];
                // Form 2: boundary face = -ĝ·v (only L side, no R element)
                for eq in 0..nq {
                    let fw = w * f_hat[eq];
                    for i in 0..dp {
                        z[base + i * nq + eq] -= fw * face.basis[q][i];
                    }
                }
            }
        }

        // 4. Volume term (Form 2: +∫ F·∇v).
        if self.preassemble_weakdiv {
            // Preassembled weak divergence × node flux — MFEM
            // DGHyperbolicConservationLaws::Mult weakdiv path
            // (ex18.hpp ComputeWeakDivergence): F(u) is evaluated at each
            // DOF *node* j, then contracted with the preassembled
            // ∫φ_i·∇φ_j matrix (trial space ByDim):
            //   z(r, eq) += Σ_j Σ_d weakdiv[r, j*dim+d] · F(u_j)[eq, d]
            // (cf. mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat)).
            // For nonlinear fluxes this is NOT equivalent to quadrature of
            // F(u(x_q)) — on periodic meshes the per-QP form fails to cancel
            // against the face fluxes and goes NaN.
            let mut flux_node = vec![0.0; nq * dim];
            let zero_pt = vec![0.0_f64; dim];
            for e in 0..self.n_elems {
                let base = e * dp * nq;
                let wd = &self.weakdiv[e];
                let mut state = vec![0.0; nq];
                for j in 0..dp {
                    for eq in 0..nq {
                        state[eq] = u[base + j * nq + eq];
                    }
                    self.flux.compute_flux(&state, &zero_pt, &mut flux_node);
                    for eq in 0..nq {
                        for d in 0..dim {
                            let f = flux_node[eq * dim + d];
                            if f != 0.0 {
                                let col = j * dim + d;
                                for r in 0..dp {
                                    z[base + r * nq + eq] += f * wd[(r, col)];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Matrix-free fallback (MFEM non-preassembled path): direct
            // quadrature of F(u(x_q))·∇φ_i.
            let q_order = 2 * self.ref_elem.order();
            let qr = self.ref_elem.quadrature(q_order);
            let n_vol_qp = qr.n_points();
            let mut phi = vec![0.0; dp];
            let mut gphi = vec![0.0; dp * dim];
            let mut state_qp = vec![0.0; nq];
            let mut flux_qp = vec![0.0; nq * dim];
            let zero_pt = vec![0.0_f64; dim];
            for e in 0..self.n_elems {
                let base = e * dp * nq;
                let geom = &self.vol_qp_geom[e];
                for q in 0..n_vol_qp {
                    // D799-3: MFEM's `AssembleElementVector` uses the element's
                    // **isoparametric** transformation per quadrature point
                    // (`Tr.SetIntPoint(&ip)`, hyperbolic.cpp:44-105:
                    // `w = ip.weight·Tr.Weight()`, `dshape = CalcPhysDShape(Tr)`
                    // = J⁻ᵀ∇_ref), not one centroid Jacobian for the whole
                    // element.  The table was built in `new`.
                    let xi = &qr.points[q];
                    let (det_j, jit) = geom[q];
                    let w = qr.weights[q] * det_j;
                    self.ref_elem.eval_basis(xi, &mut phi);
                    // Interpolate u to QP
                    state_qp.fill(0.0);
                    for eq in 0..nq {
                        for i in 0..dp {
                            state_qp[eq] += phi[i] * u[base + i * nq + eq];
                        }
                    }
                    // Compute physical flux at QP
                    self.flux.compute_flux(&state_qp, &zero_pt, &mut flux_qp);
                    // Evaluate physical gradient of test functions at this QP
                    self.ref_elem.eval_grad_basis(xi, &mut gphi);
                    for i in 0..dp {
                        // ∇x_φ_i = J^{-T} · ∇ξ_φ_i  (row d of J⁻ᵀ)
                        let mut gphys = [0.0_f64; 3];
                        for d in 0..dim {
                            for k in 0..dim {
                                gphys[d] += jit[d][k] * gphi[i * dim + k];
                            }
                        }
                        // z[e,i,eq] += w · Σ_d F[eq,d]·∂_d φ_i  (signed — D696)
                        for eq in 0..nq {
                            let mut acc = 0.0_f64;
                            for d in 0..dim {
                                acc += flux_qp[eq * dim + d] * gphys[d];
                            }
                            z[base + i * nq + eq] += w * acc;
                        }
                    }
                }
            }
        }
    }
}
