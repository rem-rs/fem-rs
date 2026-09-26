//! Prism Pk partial assembly on the wedge's tensor structure.
//!
//! MFEM's `H1_WedgeElement` is a tensor product of a 1-D `[0,1]` factor and a
//! triangle factor ([`H1TriPk`], MFEM `H1_TriangleElement`), so the diffusion
//! form on a prism can be applied with a two-factor sum-factorization instead
//! of a dense element matrix:
//!
//! 1. contract the layer (ξ) index of the element vector for every axial
//!    quadrature point (values and ξ-derivatives),
//! 2. at each full quadrature point `(ξ_q, η_t, ζ_t)`, contract the triangle
//!    factor to get the reference gradient `(∂_ξ u, ∂_η u, ∂_ζ u)`, map it
//!    through the per-QP metric and scatter it back into the triangle factor,
//! 3. contract the layer index back.
//!
//! **Geometry (D770).**  Each quadrature point carries the full symmetric
//! metric `W = w_q · detJ · κ(x_q) · J⁻ᵀ·J⁻¹` (6 values).  The previous version
//! stored `J⁻ᵀ` and used only the *first* QP's `|detJ|` and `κ` as one scalar
//! factor outside the Kronecker sum, with the Jacobian itself taken from a
//! **finite-difference** (ε = 1e-6) probe of the mapping.  That is only correct
//! for a prism whose map is a similarity (`J⁻ᵀJ⁻¹ ∝ I`): every sheared,
//! stretched or twisted prism — i.e. every prism mesh that is not made of
//! identical unit prisms — was silently integrated with the wrong metric (a
//! stretched prism is off by the anisotropy ratio, a twisted one by O(1)).
//!
//! **Curved prisms (D783).**  The geometry of a *straight* prism is the
//! 6-vertex trilinear map ([`PrismGeom`], the P1 vertex table).  A mesh
//! carrying a **high-order geometry table** (`geom_order ≥ 2`:
//! [`Mesh::set_curvature`] or the reader's `nodes` table, e.g. a curved
//! Prism15/18 element) is instead evaluated with the mesh crate's order-`g`
//! isoparametric map ([`fem_mesh::element_jacobian_at`] →
//! `PrismPk(g)` over `Mesh::geometry_nodes`), i.e. *exactly* the map the
//! assembled path evaluates (`assembler::geo_ref_elem` /
//! `geo_ref_elem_from_mesh` return the same `PrismPk(g)` for `g > 1`).
//! Before D783 every prism was approximated by its six straight-edged
//! vertices, so a curved element was integrated with the wrong metric while
//! the assembled matrix used the curved one (the D732/D734 class of gap; on
//! this file's fixture the pre-fix deviation is O(1e-1) relative).
//! The straight branch (`geom_order ≤ 1`) keeps [`PrismGeom`]'s analytic
//! trilinear Jacobian bit for bit (the mesh crate's D715/D787 gating pattern).
//! Pinned by `crates/assembly/tests/d808_prism_pa_multi.rs`.
//!
//! [`Mesh::set_curvature`]: fem_mesh::Mesh::set_curvature
//!
//! The quadrature is the assembled path's own: `p+1` Gauss-Legendre points on
//! `[0,1]` (the 1-D factor of MFEM's `IntRules.Get(Geometry::PRISM, 2p+1)`,
//! which is `seg_rule(2p+1)`) times `tri_rule(2p+1)`.  PA and assembly
//! therefore evaluate the same integrand at the same points, and agree to
//! round-off on *any* prism — affine, sheared or twisted (pinned in
//! `prism_pa_all_orders_match_assembled` and
//! `crates/assembly/tests/d770_prism_pa_geometry.rs`).
//!
//! Known limitation (residual, registered): the mapping used for the geometry
//! is the 6-vertex trilinear one, i.e. the mesh's P1 geometry.  A prism mesh
//! carrying a *high-order* geometry table (curved prisms, Prism15/18) is
//! approximated by its straight-edged vertices here, while the assembled path
//! uses the curved map — the same class of gap as D732/D734.
//!
//! **D783 (round 75) closes that residual for curved meshes**: when the mesh
//! carries a geometry table of order `g ≥ 2` ([`Mesh::set_curvature`], the
//! reader's `nodes` table), the per-QP `(J, x)` come from
//! [`fem_mesh::element_jacobian_at`] — the mesh crate's order-`g`
//! isoparametric `PrismPk(g)` path, i.e. *exactly* the map the assembled path
//! evaluates (`assembler::geo_ref_elem` / `geo_ref_elem_from_mesh` return the
//! same `PrismPk(g)` for `g > 1`).  The straight branch (`geom_order ≤ 1`)
//! still uses [`PrismGeom`]'s analytic trilinear Jacobian, so every straight
//! prism keeps its previous values bit for bit (the D715/D787 gating pattern).
//! Pinned by `crates/assembly/tests/d808_prism_pa_multi.rs` (curved and
//! straight, multi-element, shared DOFs).
//!
//! [`Mesh::set_curvature`]: fem_mesh::Mesh::set_curvature

use crate::pa::types::PaData;
use fem_element::lagrange::H1TriPk;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;

// ─── 1D Lagrange basis (Gauss-Lobatto nodes on [0,1]) ─────────────────────

/// The closed Gauss-Lobatto points on `[0,1]` — the 1-D factor of
/// `PrismPk`/MFEM `H1_WedgeElement` (D164), taken from the shared `[0,1]` table
/// (`MFEM`'s `poly1d.ClosedPoints`).
///
/// D729: this used to map the `[-1,1]` table through `0.5·(x+1)`; above
/// `p = 4` that double rounding (MFEM stores the `[0,1]` node, this crate's
/// `gauss_lobatto_arbitrary` maps it back to `[-1,1]`, then the map re-applies
/// the affine transform) differs from MFEM's stored node by 1–2 ulp.
fn gll_closed_points(p: usize) -> Vec<f64> {
    fem_element::quadrature::gauss_lobatto_01_arbitrary(p + 1).0
}

/// All 1-D Lagrange basis values on `nodes` at `x` (nodes are distinct).
fn lag1d_all(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut out = vec![0.0_f64; n];
    for (i, v) in out.iter_mut().enumerate() {
        let mut acc = 1.0;
        for (m, xm) in nodes.iter().enumerate() {
            if m != i {
                acc *= (x - xm) / (nodes[i] - xm);
            }
        }
        *v = acc;
    }
    out
}

/// All 1-D Lagrange basis derivatives on `nodes` at `x`.
fn dlag1d_all(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut out = vec![0.0_f64; n];
    for (i, v) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, xj) in nodes.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut term = 1.0 / (nodes[i] - xj);
            for (m, xm) in nodes.iter().enumerate() {
                if m != i && m != j {
                    term *= (x - xm) / (nodes[i] - xm);
                }
            }
            acc += term;
        }
        *v = acc;
    }
    out
}

/// Gauss-Legendre quadrature on `[0,1]`; delegates to the crate's shared MFEM
/// tables ([`fem_element::quadrature::gauss_legendre_01`] for `n <= 5`, MFEM's
/// Newton generator above).
///
/// D729: the previous private table was 15-digit literals for `n <= 6` (1–4 ulp
/// from MFEM), a `[-1,1]`-remap for `n = 7`, and a **trapezoidal** `1/n`-weight
/// fallback for every `n >= 8` — which is the `2p+1` rule of `p >= 4`, i.e. the
/// whole `S₁`/`M₁` Kronecker factor pair was built from a non-Gauss rule there
/// (PA vs assembled: 7.7e-1 absolute at `p = 4`).
fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n <= 5 {
        fem_element::quadrature::gauss_legendre_01(n)
    } else {
        fem_element::quadrature::gauss_legendre_01_arbitrary(n)
    }
}

// ─── Physical-to-reference Jacobian for prism ───────────────────────────────

/// Build geometry data for a prism element: physical coordinates, Jacobian.
struct PrismGeom {
    /// Physical coords of the 6 vertices [bottom tri, top tri].
    v: [[f64; 3]; 6],
}

impl PrismGeom {
    fn from_mesh<M: MeshTopology>(mesh: &M, e: u32) -> Self {
        let ns = mesh.element_nodes(e);
        let mut v = [[0.0; 3]; 6];
        for i in 0..6.min(ns.len()) {
            let c = mesh.node_coords(ns[i]);
            v[i] = [c[0], c[1], c[2]];
        }
        PrismGeom { v }
    }

    /// Map reference (ξ, η, ζ) → physical (x, y, z) via trilinear prism mapping.
    fn map(&self, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
        let xi0 = 1.0 - xi;
        let (n0, n1, n2) = (0usize, 1, 2);
        let (n3, n4, n5) = (3usize, 4, 5);
        let lam0 = 1.0 - eta - zeta;
        [
            xi0 * (lam0 * self.v[n0][0] + eta * self.v[n1][0] + zeta * self.v[n2][0])
                + xi * (lam0 * self.v[n3][0] + eta * self.v[n4][0] + zeta * self.v[n5][0]),
            xi0 * (lam0 * self.v[n0][1] + eta * self.v[n1][1] + zeta * self.v[n2][1])
                + xi * (lam0 * self.v[n3][1] + eta * self.v[n4][1] + zeta * self.v[n5][1]),
            xi0 * (lam0 * self.v[n0][2] + eta * self.v[n1][2] + zeta * self.v[n2][2])
                + xi * (lam0 * self.v[n3][2] + eta * self.v[n4][2] + zeta * self.v[n5][2]),
        ]
    }

    /// **Analytic** Jacobian of the trilinear prism map (D770 — the previous
    /// version differenced `map` with `eps = 1e-6`): row `c` is `∂x/∂ξ_c` for
    /// `ξ = (ξ, η, ζ)`, i.e.
    /// `x = (1-ξ)·B + ξ·T` with `B = λ₀v₀ + ηv₁ + ζv₂`,
    /// `T = λ₀v₃ + ηv₄ + ζv₅`, `λ₀ = 1-η-ζ`, hence
    /// `∂_ξ x = T - B`, `∂_η x = (1-ξ)(v₁-v₀) + ξ(v₄-v₃)`,
    /// `∂_ζ x = (1-ξ)(v₂-v₀) + ξ(v₅-v₃)`.
    fn jacobian(&self, xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 3] {
        let lam0 = 1.0 - eta - zeta;
        let xi0 = 1.0 - xi;
        let mut j = [[0.0_f64; 3]; 3];
        for d in 0..3 {
            let b = lam0 * self.v[0][d] + eta * self.v[1][d] + zeta * self.v[2][d];
            let t = lam0 * self.v[3][d] + eta * self.v[4][d] + zeta * self.v[5][d];
            j[0][d] = t - b;
            j[1][d] = xi0 * (self.v[1][d] - self.v[0][d]) + xi * (self.v[4][d] - self.v[3][d]);
            j[2][d] = xi0 * (self.v[2][d] - self.v[0][d]) + xi * (self.v[5][d] - self.v[3][d]);
        }
        j
    }
}

/// Determinant and inverse of a 3×3 Jacobian whose rows are `∂x/∂ξ_c`.
///
/// Returns `(det, Jinv)` with `Jinv[r][c] = ∂ξ_r/∂x_c` (the true inverse; the
/// old code clamped to `max(det, 1e-30)` and took `|det|`, which quietly
/// produced `1e30` entries for inverted prisms instead of the negative-signed
/// operator the assembled path assembles — D679's signed convention).
fn invert_3x3(j: &[[f64; 3]; 3]) -> (f64, [[f64; 3]; 3]) {
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    let inv = 1.0 / det;
    let c = [
        [
            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) * inv,
            (j[0][2] * j[2][1] - j[0][1] * j[2][2]) * inv,
            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) * inv,
        ],
        [
            (j[1][2] * j[2][0] - j[1][0] * j[2][2]) * inv,
            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * inv,
            (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * inv,
        ],
        [
            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) * inv,
            (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * inv,
            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * inv,
        ],
    ];
    (det, c)
}

/// Symmetric components of `J⁻ᵀ·J⁻¹` (reference metric), in the storage order
/// `[m_ξξ, m_ξη, m_ξζ, m_ηη, m_ηζ, m_ζζ]`.
fn ref_metric(jinv: &[[f64; 3]; 3]) -> [f64; 6] {
    let col = |c: usize| [jinv[0][c], jinv[1][c], jinv[2][c]];
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    [
        dot(c0, c0),
        dot(c0, c1),
        dot(c0, c2),
        dot(c1, c1),
        dot(c1, c2),
        dot(c2, c2),
    ]
}

/// Number of geometry values stored per quadrature point: the six symmetric
/// components of `W = w_q · detJ · κ(x_q) · J⁻ᵀ·J⁻¹` (D770).
pub const PA_GEOM: usize = 6;

/// The per-QP `(J, x_phys)` of a **curved** prism (D783), from the mesh's own
/// order-`g` isoparametric geometry table.
///
/// Returns `None` for a straight mesh (`geom_order ≤ 1`), where the caller
/// keeps [`PrismGeom`]'s analytic trilinear map bit for bit.  The geometry
/// element and node table are the mesh crate's
/// [`fem_mesh::element_jacobian_at`] choice — the same `PrismPk(g)` and
/// per-element geometry row the assembled path uses
/// (`assembler::geo_ref_elem` / `assembler::geo_ref_elem_from_mesh`), so a
/// curved prism's PA and assembly integrate the same map at the same
/// quadrature points.
///
/// The transpose and the straight-mesh gating live in
/// [`super::curved::curved_jacobian`], shared with the hex/quad kernels
/// (D808-4).
fn curved_prism_jacobian<M: MeshTopology>(
    mesh: &M,
    e: u32,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> Option<([[f64; 3]; 3], [f64; 3])> {
    super::curved::curved_jacobian(mesh, e, &[xi, eta, zeta])
}

// ─── PA data build ─────────────────────────────────────────────────────────

/// Build PA data for Prism Pk diffusion: per-element `PaData` with geometry info.
///
/// Stores the six symmetric components of `w_q·detJ·κ·J⁻ᵀJ⁻¹` at every
/// quadrature point of the assembled path's rule (`p+1` Gauss-Legendre points
/// in ξ × `tri_rule(2p+1)`), computed from the element's analytic trilinear
/// Jacobian (D770; the old data carried `J⁻ᵀ`, `|detJ|` and `κ` of a
/// finite-difference Jacobian and the apply used only the first QP's scalars).
/// A curved mesh (`geom_order ≥ 2`) takes the mesh's order-`g` isoparametric
/// map instead of the 6 vertices (D783 — see the module docs).
pub fn build_prism_pk_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
    p: usize,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nq_1d = p + 1; // axial quadrature points, the factor of the 2p+1 rule
    let tri_ref = H1TriPk::new(p);
    let tri_rule = tri_ref.quadrature((2 * p + 1).min(15) as u8);
    let nq_tri = tri_rule.points.len();
    let nqp = nq_1d * nq_tri;
    let mut pd = PaData::new(n_elems, nqp, PA_GEOM);

    let (xi_qpts, xi_wts) = gauss_legendre_1d(nq_1d);
    // D783: a curved mesh (`geom_order ≥ 2`) takes its geometry from the
    // mesh's order-`g` isoparametric table instead of the 6 vertices.
    let curved = mesh.geom_order() >= 2;

    for e in 0..n_elems {
        let geom = if curved { None } else { Some(PrismGeom::from_mesh(mesh, e as u32)) };

        for (qxi, &xi) in xi_qpts.iter().enumerate() {
            for (qtri, tri_qp) in tri_rule.points.iter().enumerate() {
                let qi = qxi * nq_tri + qtri;
                let eta = tri_qp[0];
                let zeta = tri_qp[1];

                let (jac, xc) = match geom.as_ref() {
                    Some(g) => (g.jacobian(xi, eta, zeta), g.map(xi, eta, zeta)),
                    None => curved_prism_jacobian(mesh, e as u32, xi, eta, zeta)
                        .expect("geom_order ≥ 2 prism geometry table"),
                };
                let (det, jinv) = invert_3x3(&jac);
                let metric = ref_metric(&jinv);

                let scale = xi_wts[qxi] * tri_rule.weights[qtri] * det * kappa(&xc);

                let qd = pd.elem_qp_mut(e, qi);
                for c in 0..PA_GEOM {
                    qd[c] = metric[c] * scale;
                }
            }
        }
    }
    pd
}

// ─── PA apply (tensor sum-factorization) ───────────────────────────────────

/// Apply the prism stiffness matrix with the wedge's two-factor
/// sum-factorization: `y += A·x`.
///
/// `elem_dofs` are the *space's* element dofs, which since D168 follow MFEM's
/// `H1_WedgeElement` entity order, while the factors act in [`PrismPk`]'s
/// layer-major order — the application permutes through [`H1PrismPk`]'s slot
/// table (`slot m of the H1 wedge = layer slot `perm[m]` of `PrismPk`).
pub fn pa_apply_prism_pk(
    pd: &PaData,
    elem_dofs: &[Vec<u32>],
    p: usize,
    x: &[f64],
    y: &mut [f64],
) {
    let n_tri = (p + 1) * (p + 2) / 2;
    let np1 = p + 1;
    let n_loc = np1 * n_tri;

    // entity slot ↔ layer slot permutation (`H1PrismPk` = `PrismPk` permuted).
    let perm = fem_element::lagrange::H1PrismPk::new(p).layer_perm().to_vec();
    let mut inv = vec![0usize; n_loc];
    for (m, &ls) in perm.iter().enumerate() {
        inv[ls] = m;
    }

    // Layer factor: GLL Lagrange values `b[q*np1+i]` and derivatives
    // `g[q*np1+i]` at the `p+1` axial rule points (the rule of the data build).
    let nq_1d = np1;
    let (xi_q, _) = gauss_legendre_1d(nq_1d);
    let nodes = gll_closed_points(p);
    let mut b = vec![0.0; nq_1d * np1];
    let mut g = vec![0.0; nq_1d * np1];
    for (q, &xq) in xi_q.iter().enumerate() {
        let v = lag1d_all(&nodes, xq);
        let d = dlag1d_all(&nodes, xq);
        for i in 0..np1 {
            b[q * np1 + i] = v[i];
            g[q * np1 + i] = d[i];
        }
    }

    // Triangle factor: values `tv[t*n_tri+i]`, gradients
    // `tg[t*n_tri*2 + 2 i + d]` at the same triangle rule as the data build.
    let tri = H1TriPk::new(p);
    let tri_rule = tri.quadrature((2 * p + 1).min(15) as u8);
    let ntq = tri_rule.points.len();
    let mut tv = vec![0.0; ntq * n_tri];
    let mut tg = vec![0.0; ntq * n_tri * 2];
    for (t, pt) in tri_rule.points.iter().enumerate() {
        tri.eval_basis(pt, &mut tv[t * n_tri..(t + 1) * n_tri]);
        tri.eval_grad_basis(pt, &mut tg[t * n_tri * 2..(t + 1) * n_tri * 2]);
    }
    debug_assert_eq!(nq_1d * ntq, pd.nqp);

    // Scratch: `ue`/`ye` are [tri dof × layer], `s0`/`s1`/`f1`/`f2` are
    // [axial qp × tri dof].
    let mut ue = vec![0.0; n_loc];
    let mut ye = vec![0.0; n_loc];
    let mut s0 = vec![0.0; nq_1d * n_tri];
    let mut s1 = vec![0.0; nq_1d * n_tri];
    let mut f1 = vec![0.0; nq_1d * n_tri];
    let mut f2 = vec![0.0; nq_1d * n_tri];

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < n_loc {
            continue;
        }
        // Gather: layer slot `l` of tri dof `i` reads entity slot inv[l*n_tri+i].
        for l in 0..np1 {
            for i in 0..n_tri {
                ue[i * np1 + l] = x[dofs[inv[l * n_tri + i]] as usize];
            }
        }

        // Phase 1: contract the layer factor (values and ξ-derivatives).
        for q in 0..nq_1d {
            for i in 0..n_tri {
                let mut a0 = 0.0;
                let mut a1 = 0.0;
                for l in 0..np1 {
                    let u = ue[i * np1 + l];
                    a0 += u * b[q * np1 + l];
                    a1 += u * g[q * np1 + l];
                }
                s0[q * n_tri + i] = a0;
                s1[q * n_tri + i] = a1;
            }
        }

        // Phase 2: at each full qp form the reference gradient, apply the
        // metric `W`, and scatter the fluxes back into the triangle factor.
        f1.iter_mut().for_each(|v| *v = 0.0);
        f2.iter_mut().for_each(|v| *v = 0.0);
        for q in 0..nq_1d {
            for t in 0..ntq {
                let qi = q * ntq + t;
                debug_assert!(qi < pd.nqp);
                let w = pd.elem_qp(e, qi);
                let (w00, w01, w02, w11, w12, w22) = (w[0], w[1], w[2], w[3], w[4], w[5]);

                let mut u_xi = 0.0;
                let mut u_eta = 0.0;
                let mut u_zeta = 0.0;
                for i in 0..n_tri {
                    u_xi += s1[q * n_tri + i] * tv[t * n_tri + i];
                    u_eta += s0[q * n_tri + i] * tg[t * n_tri * 2 + i * 2];
                    u_zeta += s0[q * n_tri + i] * tg[t * n_tri * 2 + i * 2 + 1];
                }
                let f_xi = w00 * u_xi + w01 * u_eta + w02 * u_zeta;
                let f_eta = w01 * u_xi + w11 * u_eta + w12 * u_zeta;
                let f_zeta = w02 * u_xi + w12 * u_eta + w22 * u_zeta;

                for i in 0..n_tri {
                    f1[q * n_tri + i] += tv[t * n_tri + i] * f_xi;
                    f2[q * n_tri + i] += tg[t * n_tri * 2 + i * 2] * f_eta
                        + tg[t * n_tri * 2 + i * 2 + 1] * f_zeta;
                }
            }
        }

        // Phase 3: contract the layer factor back.
        for i in 0..n_tri {
            for l in 0..np1 {
                let mut v = 0.0;
                for q in 0..nq_1d {
                    v += g[q * np1 + l] * f1[q * n_tri + i] + b[q * np1 + l] * f2[q * n_tri + i];
                }
                ye[i * np1 + l] = v;
            }
        }

        // Scatter back through the entity↔layer permutation.
        for l in 0..np1 {
            for i in 0..n_tri {
                y[dofs[inv[l * n_tri + i]] as usize] += ye[i * np1 + l];
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    fn make_prism_mesh() -> Mesh<3> {
        Mesh::<3>::uniform(
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
            ],
            vec![0u32, 1, 2, 3, 4, 5],
            vec![1i32],
            fem_mesh::ElementType::Prism6,
            vec![], vec![],
            fem_mesh::ElementType::Tri3,
        )
    }

    #[test]
    fn prism_pa_apply_is_finite() {
        let mesh = make_prism_mesh();
        let space = H1Space::new(mesh.clone(), 2);
        let n = space.n_dofs();
        let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, 2);

        let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
        for e in 0..mesh.n_elems() {
            let d = space.element_dofs(e as u32);
            elem_dofs.push(d.to_vec());
        }

        let x = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        pa_apply_prism_pk(&pd, &elem_dofs, 2, &x, &mut y);
        assert!(y.iter().all(|&v| v.is_finite()), "PA apply produced non-finite values");
        assert!(y.iter().any(|&v| v.abs() > 0.0), "PA apply produced all zeros");
    }

    #[test]
    fn prism_pa_p2_matches_assembled() {
        let mesh = make_prism_mesh();
        let p = 2;
        let space = H1Space::new(mesh.clone(), p as u8);
        let n = space.n_dofs();

        // Assembled matrix via standard diffusion integrator
        let a_assembled = crate::Assembler::assemble_bilinear(
            &space, &[&crate::standard::DiffusionIntegrator { kappa: 1.0 }], 2 * p as u8 + 1,
        );

        // PA apply
        let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
        let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
        for e in 0..mesh.n_elems() as u32 {
            elem_dofs.push(space.element_dofs(e as u32).to_vec());
        }

        // Non-constant field: a constant x makes A·x ≈ 0 on both sides, which
        // cannot catch a wrong entity↔layer permutation (D168).
        let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
        let mut y_pa = vec![0.0_f64; n];
        pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);

        let mut y_asm = vec![0.0_f64; n];
        a_assembled.spmv(&x, &mut y_asm);

        let max_err: f64 = y_pa.iter().zip(y_asm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let pa_nrm: f64 = y_pa.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        // PA and assembly share the rule and the (analytic) geometry, so the
        // only difference left is summation order (D770).
        assert!(
            max_err < 1e-12,
            "Prism P{p} PA vs assembled max abs err = {max_err:.3e} (pa_nrm={pa_nrm:.3e})"
        );
    }

    /// D729 — the axial 1-D rule must be the shared MFEM `[0,1]` table the rest
    /// of the library uses (`gauss_legendre_01` for `n <= 5`, MFEM's Newton
    /// generator above), not a private table.
    ///
    /// The old local `gauss_legendre_1d` was a 15-digit literal table for
    /// `n <= 6`, `0.5·(x+1)`-remapped `[-1,1]` nodes for `n = 7`, and — for
    /// every `n >= 8`, i.e. the `2p+1` rule of `p >= 4` — a **trapezoidal**
    /// fallback with weights `1/n`, which is not a Gauss rule at all.
    #[test]
    fn prism_axial_rules_are_the_mfem_unit_interval_tables() {
        for n in 2..=12usize {
            let (p, w) = gauss_legendre_1d(n);
            let (mp, mw) = if n <= 5 {
                fem_element::quadrature::gauss_legendre_01(n)
            } else {
                fem_element::quadrature::gauss_legendre_01_arbitrary(n)
            };
            assert_eq!(p.len(), mp.len(), "n={n}: node count");
            for i in 0..n {
                assert_eq!(
                    p[i].to_bits(),
                    mp[i].to_bits(),
                    "n={n} node {i}: {:.17e} vs MFEM {:.17e}",
                    p[i],
                    mp[i]
                );
                assert_eq!(
                    w[i].to_bits(),
                    mw[i].to_bits(),
                    "n={n} weight {i}: {:.17e} vs MFEM {:.17e}",
                    w[i],
                    mw[i]
                );
            }
            // The rule integrates on [0,1]: Σw = 1 and every node is inside.
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "n={n}: Σw = {sum}");
            assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)), "n={n}: node outside [0,1]");
        }
    }

    /// D729 — the axial nodes of the tensor factors are MFEM's `[0,1]`
    /// Gauss-Lobatto nodes (`poly1d.ClosedPoints`), not `0.5·(x+1)` images of
    /// the `[-1,1]` table (the double rounding is a 1-ulp shim for `p >= 5`).
    #[test]
    fn prism_axial_gll_nodes_are_the_unit_interval_table() {
        for n in 2..=8usize {
            let got = gll_closed_points(n - 1);
            let want = fem_element::quadrature::gauss_lobatto_01_arbitrary(n).0;
            assert_eq!(got.len(), want.len(), "n={n}");
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "n={n} node {i}: {:.17e} vs MFEM {:.17e}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// D770 — the per-QP metric must be the analytic trilinear one: the
    /// finite-difference Jacobian it replaced (`eps = 1e-6`) agreed with the
    /// analytic Jacobian only to ~1e-10, and the apply used the first QP's
    /// scalars only.  On the unit prism the analytic Jacobian is exactly `I`.
    #[test]
    fn prism_geometry_is_analytic() {
        let mesh = make_prism_mesh();
        let geom = PrismGeom::from_mesh(&mesh, 0);
        // Unit prism: ∂x/∂ξ = (0,0,1), ∂x/∂η = (1,0,0), ∂x/∂ζ = (0,1,0).  The
        // corner points are exact bitwise; interior points carry the rounded
        // `λ₀ = 1-η-ζ` sum (a plain arithmetic fact, not an FD error — the
        // finite-difference Jacobian this replaced was off by ~1e-10 *relative*
        // everywhere).
        let want: [[f64; 3]; 3] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for (xi, eta, zeta) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] {
            let j = geom.jacobian(xi, eta, zeta);
            for c in 0..3 {
                for d in 0..3 {
                    assert_eq!(
                        j[c][d].to_bits(),
                        want[c][d].to_bits(),
                        "J[{c}][{d}] at ({xi},{eta},{zeta}) = {}",
                        j[c][d]
                    );
                }
            }
            let (det, jinv) = invert_3x3(&j);
            assert_eq!(det, 1.0);
            // `jinv` is the standard matrix inverse of `jac`; its *columns* are
            // `∂ξ_r/∂x_d` (the rows of `A = (∂ξ/∂x)`, see `ref_metric`).
            for r in 0..3 {
                for d in 0..3 {
                    let mut s = 0.0;
                    for k in 0..3 {
                        s += j[r][k] * jinv[k][d];
                    }
                    assert!((s - if r == d { 1.0 } else { 0.0 }).abs() < 1e-15);
                }
            }
            let m = ref_metric(&jinv);
            assert_eq!(m, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        }
        for (xi, eta, zeta) in [(0.3, 0.2, 0.5), (0.5, 1.0 / 3.0, 1.0 / 3.0)] {
            let j = geom.jacobian(xi, eta, zeta);
            for c in 0..3 {
                for d in 0..3 {
                    assert!(
                        (j[c][d] - want[c][d]).abs() < 1e-15,
                        "J[{c}][{d}] at ({xi},{eta},{zeta}) = {} vs {}",
                        j[c][d],
                        want[c][d]
                    );
                }
            }
            let (det, jinv) = invert_3x3(&j);
            assert!((det - 1.0).abs() < 1e-15, "detJ = {det}");
            let m = ref_metric(&jinv);
            for (k, v) in m.iter().enumerate() {
                let w = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0][k];
                assert!((v - w).abs() < 1e-15, "metric[{k}] = {v} vs {w}");
            }
        }
    }

    /// D770 — the PA Kronecker apply must reproduce the assembled diffusion at
    /// every order (the D729 trapezoidal-fallback defect) and now with the
    /// exact per-QP geometry, so the residual is round-off, not 1e-10
    /// finite-difference noise.
    #[test]
    fn prism_pa_all_orders_match_assembled() {
        let mesh = make_prism_mesh();
        for p in [2usize, 3, 4, 5] {
            let space = H1Space::new(mesh.clone(), p as u8);
            let n = space.n_dofs();
            let a_assembled = crate::Assembler::assemble_bilinear(
                &space,
                &[&crate::standard::DiffusionIntegrator { kappa: 1.0 }],
                2 * p as u8 + 1,
            );
            let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
            let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
            for e in 0..mesh.n_elems() as u32 {
                elem_dofs.push(space.element_dofs(e as u32).to_vec());
            }
            let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
            let mut y_pa = vec![0.0_f64; n];
            pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);
            let mut y_asm = vec![0.0_f64; n];
            a_assembled.spmv(&x, &mut y_asm);
            let max_err: f64 = y_pa
                .iter()
                .zip(y_asm.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let pa_nrm: f64 = y_pa.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            println!("prism P{p}: PA vs assembled max abs err = {max_err:.3e} (pa_nrm={pa_nrm:.3e})");
            assert!(
                max_err < 1e-12,
                "Prism P{p} PA vs assembled max abs err = {max_err:.3e} (pa_nrm={pa_nrm:.3e})"
            );
        }
    }
}
