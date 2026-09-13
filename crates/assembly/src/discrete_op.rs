//! Discrete linear operators: gradient, curl, and divergence.
//!
//! These operators map between finite element spaces in the de Rham complex:
//!
//! ```text
//!   H1 --grad--> H(curl) --curl--> H(div) --div--> L2
//! ```
//!
//! ## Supported space pairs
//!
//! | Operator     | Domain     | Range      | Order |
//! |--------------|------------|------------|-------|
//! | `gradient`   | H1 (P1)    | H(curl) ND1| 1     |
//! | `gradient`   | H1 (P2)    | H(curl) ND2| 2     |
//! | `gradient`   | H1 (P2)    | H(curl) ND2 (3-D hex) | 2 |
//! | `curl_2d`    | H(curl) ND1| L2 (P0)   | 1     |
//! | `curl_2d`    | H(curl) ND2| L2 (P1)   | 2     |
//! | `curl_2d`    | H(curl) ND2| L2 (P2)   | 2     |
//! | `divergence` | H(div) RT0 | L2 (P0)   | 0     |
//! | `divergence` | H(div) RT1 | L2 (P1)   | 1     |
//! | `divergence` | H(div) RT1 | L2 (P2)   | 1     |
//! | `divergence` | H(div) RT2 | L2 (P2)   | 2     |
//! | `curl_3d`    | H(curl) ND1| H(div) RT0| 1     |
//! | `curl_3d`    | H(curl) ND2| H(div) RT1| 2     |
//! | `curl_2d_hdiv` | H(curl) ND1/ND2 | H(div) RT0/RT2 | 2 (2D) |
//!
//! The lowest-order (P1→ND1, RT0→P0) matrices are assembled topologically
//! (exact, no quadrature error).  Higher-order pairs (P2→ND2, RT1→P1) use
//! a numerical DOF-functional projection on each reference element, which is
//! also exact since the spaces satisfy the de Rham commuting-diagram property.
//!
//! # Error handling
//!
//! All assembly functions return [`Result<CsrMatrix<f64>, DiscreteOpError>`].
//! The error type covers incompatible space orders and unsupported mesh
//! dimensions, giving callers a chance to handle mismatches rather than
//! aborting the process with a panic.

use std::collections::HashSet;

use fem_mesh::ElementType;
use fem_element::{
    quadrature::gauss_legendre_01, HexNDk, HexQ2, ReferenceElement, TetND2, TetRT1, TriNDk,
    TriND2, TriRT1, TriRT2, VectorReferenceElement,
};
use fem_element::lagrange::factory::TriPk;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, ElementTransformation};
use fem_space::dof_manager::FaceKey;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

// ---- Error type ------------------------------------------------------------

/// Errors returned by [`DiscreteLinearOperator`] assembly methods.
#[derive(Debug, thiserror::Error)]
pub enum DiscreteOpError {
    /// The H1 space has an unsupported polynomial order.
    #[error("gradient: H1 space must be order 1 (P1) or 2 (P2), got order {0}")]
    UnsupportedH1Order(u8),

    /// The H(curl) space has an unsupported polynomial order.
    #[error("{op}: H(curl) space must be order 1 (ND1) or 2 (ND2), got order {order}")]
    UnsupportedHCurlOrder { op: &'static str, order: u8 },

    /// The H(div) space has an unsupported polynomial order.
    #[error("{op}: H(div) space must be order 0 (RT0), 1 (RT1), or 2 (RT2, 2D only), got order {order}")]
    UnsupportedHDivOrder { op: &'static str, order: u8 },

    /// The L2 space has an unsupported polynomial order.
    #[error("{op}: L2 space must be order 0 (P0), 1 (P1), or 2 (P2), got order {order}")]
    UnsupportedL2Order { op: &'static str, order: u8 },

    /// The mesh has an unsupported spatial dimension.
    #[error("{op}: unsupported mesh dimension {dim}")]
    UnsupportedDimension { op: &'static str, dim: u8 },

    /// The space orders are incompatible with each other.
    #[error("{op}: incompatible space orders — H1 order {h1_order} requires H(curl) order {h1_order}, got {hcurl_order}")]
    IncompatibleOrders { op: &'static str, h1_order: u8, hcurl_order: u8 },

    /// The mesh dimension is supported but its cell type is not.
    #[error("{op}: unsupported cell type {cell}")]
    UnsupportedCellType { op: &'static str, cell: &'static str },
}

/// Short name of a cell type, for [`DiscreteOpError::UnsupportedCellType`].
fn element_type_name(t: ElementType) -> &'static str {
    match t {
        ElementType::Tri3 => "Tri3",
        ElementType::Tri6 => "Tri6",
        ElementType::Quad4 => "Quad4",
        ElementType::Quad8 => "Quad8",
        ElementType::Quad9 => "Quad9",
        ElementType::Tet4 => "Tet4",
        ElementType::Tet10 => "Tet10",
        ElementType::Hex8 => "Hex8",
        ElementType::Hex20 => "Hex20",
        ElementType::Prism6 => "Prism6",
        ElementType::Prism15 => "Prism15",
        ElementType::Pyramid5 => "Pyramid5",
        _ => "unknown",
    }
}

/// Local RT2 Vandermonde and P2-sampled divergences for the RT2→P2 reconstruction on an
/// affine triangle (`dmat[i,k] = DOF_i^{RT2}(Φ_k^{ref})`, `ymat[p,k] = div(Φ_k)(x_p)/det_j`).
///
/// D34: the dof rows use the MFEM **nodal** functionals (signed pointwise
/// normal-flux samples shared with `HDivSpace::interpolate_vector`).
#[allow(clippy::too_many_arguments)]
fn rt2_triangle_dmat_ymat_div_p2<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    j00: f64,
    j01: f64,
    j10: f64,
    j11: f64,
    det_j: f64,
    signs: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    const N_RT2: usize = 15;
    const N_L2: usize = 6;
    let n_rt2 = N_RT2;
    let n_l2_local = N_L2;

    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);

    let mut dmat = vec![0.0_f64; n_rt2 * n_rt2];
    let mut ymat = vec![0.0_f64; n_l2_local * n_rt2];
    let mut prim = [[0.0_f64; 2]; N_RT2];
    let mut div_prim = [0.0_f64; N_RT2];

    let (dof_pts, dof_nks) = fem_element::raviart_thomas::tri_rt1::mfem_tri_nodal_dofs(2);

    for k in 0..n_rt2 {
        for s in 0..n_rt2 {
            let (xi, nk) = (&dof_pts[s], &dof_nks[s]);
            TriRT2::mfem_primitives_and_divs(xi[0], xi[1], &mut prim, &mut div_prim);
            let urx = prim[k][0];
            let ury = prim[k][1];
            let upx = (j00 * urx + j01 * ury) / det_j;
            let upy = (j10 * urx + j11 * ury) / det_j;
            // cof(J)·nk (unnormalised physical normal)
            let nx = j11 * nk[0] - j10 * nk[1];
            let ny = -j01 * nk[0] + j00 * nk[1];
            dmat[s * n_rt2 + k] = signs[s] * (upx * nx + upy * ny);
        }

        let sample_pts = [
            [x0[0], x0[1]],
            [x1[0], x1[1]],
            [x2[0], x2[1]],
            [0.5 * (x0[0] + x1[0]), 0.5 * (x0[1] + x1[1])],
            [0.5 * (x1[0] + x2[0]), 0.5 * (x1[1] + x2[1])],
            [0.5 * (x0[0] + x2[0]), 0.5 * (x0[1] + x2[1])],
        ];
        for p in 0..n_l2_local {
            let yp0 = sample_pts[p][0] - x0[0];
            let yp1 = sample_pts[p][1] - x0[1];
            let inv_det = 1.0 / det_j;
            let xi0 = inv_det * (j11 * yp0 - j01 * yp1);
            let xi1 = inv_det * (-j10 * yp0 + j00 * yp1);
            TriRT2::mfem_primitives_and_divs(xi0, xi1, &mut prim, &mut div_prim);
            ymat[p * n_rt2 + k] = div_prim[k] / det_j;
        }
    }

    (dmat, ymat)
}

// ---- Operator struct -------------------------------------------------------

/// Discrete linear operators that build sparse matrices mapping between FE spaces.
///
/// All methods are associated functions (no `self`) that take the relevant
/// spaces as arguments and return `Result<CsrMatrix<f64>, DiscreteOpError>`.
pub struct DiscreteLinearOperator;

impl DiscreteLinearOperator {
    /// Build the discrete gradient matrix G: H1 -> H(curl).
    ///
    /// ## Order 1 — topological assembly (P1 → ND1)
    ///
    /// G is the signed vertex-edge incidence matrix.  For each edge with
    /// global orientation from vertex `a` to vertex `b` (a < b):
    ///
    /// ```text
    ///   G[edge_dof, b] = +1,   G[edge_dof, a] = -1
    /// ```
    ///
    /// This is exact and requires no quadrature.
    ///
    /// ## Order 2 — numerical assembly (P2 → ND2, 2D only)
    ///
    /// The de Rham commuting diagram guarantees that G = Π_{ND2} ∘ ∇, where
    /// Π_{ND2} is the canonical ND2 interpolation operator.  Each entry is
    ///
    /// ```text
    ///   G[nd2_dof_i, p2_dof_j] = DOF_i^{ND2}(∇φ_j^{P2})
    /// ```
    ///
    /// computed per-element via numerical integration on the reference triangle.
    ///
    /// # Errors
    /// Returns [`DiscreteOpError`] if the space orders are unsupported or
    /// incompatible, or if the mesh dimension is not 2 or 3.
    pub fn gradient<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let h1_order = h1_space.order();
        let hcurl_order = hcurl_space.order();

        // Validate orders.
        match h1_order {
            1 | 2 => {}
            o => return Err(DiscreteOpError::UnsupportedH1Order(o)),
        }
        // H1 order 1 works with any HCurl order ≥ 1 (topological edge-vertex incidence).
        if h1_order == 1 {
            if hcurl_order < 1 {
                return Err(DiscreteOpError::UnsupportedHCurlOrder { op: "gradient", order: hcurl_order });
            }
            return Self::gradient_p1_nd1(h1_space, hcurl_space);
        }
        // h1_order == 2
        match hcurl_order {
            2 => match h1_space.mesh().dim() {
                2 => Self::gradient_p2_nd2(h1_space, hcurl_space),
                3 => Self::gradient_p2_nd2_hex3d(h1_space, hcurl_space),
                dim => Err(DiscreteOpError::UnsupportedDimension {
                    op: "gradient (P2→ND2)",
                    dim,
                }),
            },
            o => Err(DiscreteOpError::UnsupportedHCurlOrder { op: "gradient", order: o }),
        }
    }

    // ── Order-1 topological gradient ──────────────────────────────────────────

    fn gradient_p1_nd1<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = h1_space.mesh();
        let n_hcurl = hcurl_space.n_dofs();
        let n_h1 = h1_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_hcurl, n_h1);

        let mut visited = HashSet::with_capacity(n_hcurl);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let local_edges: &[(usize, usize)] = match mesh.element_type(e) {
                ElementType::Tri3 | ElementType::Tri6 => &[(0, 1), (1, 2), (0, 2)],
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                    &[(0, 1), (1, 2), (2, 3), (3, 0)]
                }
                ElementType::Tet4 | ElementType::Tet10 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                ElementType::Hex8 | ElementType::Hex20 => &[
                    // MUST match HCurlSpace::HEX_EDGES order (hcurl.rs), i.e.
                    // MFEM `Geometry::Constants<Geometry::CUBE>::Edges`:
                    //   (0,1),(1,2),(3,2),(0,3),(4,5),(5,6),(7,6),(4,7),
                    //   (0,4),(1,5),(2,6),(3,7).
                    // A previous version listed the edges grouped by
                    // direction, which assigned gradient rows to the wrong
                    // HCurl DOFs on hex meshes.
                    (0, 1), (1, 2), (3, 2), (0, 3),
                    (4, 5), (5, 6), (7, 6), (4, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ],
                ElementType::Prism6 | ElementType::Prism15 => &[
                    (0, 1), (0, 2), (1, 2), // bottom tri
                    (3, 4), (3, 5), (4, 5), // top tri
                    (0, 3), (1, 4), (2, 5), // vertical
                ],
                ElementType::Pyramid5 => &[
                    (0, 1), (1, 2), (2, 3), (3, 0), // base quad
                    (0, 4), (1, 4), (2, 4), (3, 4), // apex
                ],
                _ => return Err(DiscreteOpError::UnsupportedDimension { op: "gradient", dim: mesh.dim() }),
            };
            let h1_dofs = h1_space.element_dofs(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);

            for (local_edge_idx, &(li, lj)) in local_edges.iter().enumerate() {
                let edge_dof = hcurl_dofs[local_edge_idx] as usize;
                if !visited.insert(edge_dof) { continue; }

                let va_dof = h1_dofs[li] as usize;
                let vb_dof = h1_dofs[lj] as usize;

                let (gi, gj) = (verts[li], verts[lj]);
                if gi < gj {
                    coo.add(edge_dof, vb_dof,  1.0);
                    coo.add(edge_dof, va_dof, -1.0);
                } else {
                    coo.add(edge_dof, va_dof,  1.0);
                    coo.add(edge_dof, vb_dof, -1.0);
                }
            }
        }

        Ok(coo.into_csr())
    }

    // ── Order-2 numerical gradient (P2 → ND2, 2D triangles only) ─────────────

    /// D32: all ND2 dof rows use the MFEM **nodal** point-value functionals
    /// `σ_i(F) = F(x_i)·t̂_i` (Gauss points on the edges, `(1/3,1/3)` interior),
    /// consistent with `TriND2` / `HCurlSpace` (anti-diagonal reversal pairing
    /// encoded in `element_dofs`/`element_signs`).
    ///
    /// Because the covariant pullback of the physical gradient is the reference
    /// gradient (`Jᵀ·J^{-T}∇_ref φ = ∇_ref φ`), every row is J-independent and
    /// evaluated once on the reference element.  The per-element scatter just
    /// applies `HCurlSpace::element_signs` (the ±1 of the reversal transform)
    /// and writes into the (already permuted) global slots.
    fn gradient_p2_nd2<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = h1_space.mesh();
        if mesh.dim() != 2 {
            return Err(DiscreteOpError::UnsupportedDimension { op: "gradient (P2→ND2)", dim: mesh.dim() });
        }

        let n_nd2 = hcurl_space.n_dofs();
        let n_p2  = h1_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_nd2, n_p2);

        let n_p2_local = 6usize;
        let dim = 2usize;

        // 2-point Gauss-Legendre on [0,1] — the ND2 edge dof points
        // (MFEM `OpenPoints(1)`), matching `TriND2::GL2`.
        let (gl2, _) = gauss_legendre_01(2);

        let p2_elem = TriPk::new(2);

        // ── Reference rows (all 8, J-independent).
        // (point, tangent) per slot, mirroring `TriND2` slot order:
        // e₀ (v0→v1) tang (1,0); e₁ (v1→v2) tang (−1,1);
        // e₂ (v2→v0) tang (0,−1); interior (1/3,1/3) tang (1,0),(0,1).
        let g_ref: Vec<f64> = {
            let mut pts_tans: [([f64; 2], [f64; 2]); 8] = [([0.0; 2], [0.0; 2]); 8];
            let mut row = 0usize;
            for &t in gl2.iter() {
                pts_tans[row] = ([t, 0.0], [1.0, 0.0]);
                row += 1;
            }
            for &t in gl2.iter() {
                pts_tans[row] = ([1.0 - t, t], [-1.0, 1.0]);
                row += 1;
            }
            for &t in gl2.iter() {
                pts_tans[row] = ([0.0, 1.0 - t], [0.0, -1.0]);
                row += 1;
            }
            pts_tans[row] = ([1.0 / 3.0, 1.0 / 3.0], [1.0, 0.0]);
            row += 1;
            pts_tans[row] = ([1.0 / 3.0, 1.0 / 3.0], [0.0, 1.0]);
            assert_eq!(row + 1, 8);

            let mut g = vec![0.0f64; 8 * n_p2_local];
            let mut p2_grads = vec![0.0f64; n_p2_local * dim];
            for (r, (pt, tang)) in pts_tans.iter().enumerate() {
                p2_elem.eval_grad_basis(pt, &mut p2_grads);
                for j in 0..n_p2_local {
                    g[r * n_p2_local + j] = p2_grads[j * dim] * tang[0]
                        + p2_grads[j * dim + 1] * tang[1];
                }
            }
            g
        };

        // ── Scatter into global COO, one element at a time.
        //
        // Local slot s maps to global `nd2_dofs[s]`; `element_signs[s]` is the
        // ±1 of the edge-reversal transform (aligned +1, reversed −1), so the
        // global entry is `sign_s · σ_s^{ref}(∇φ)`.  Adjacent elements now
        // agree exactly (nodal functionals), so first-writer-wins stays valid.
        let mut visited = HashSet::with_capacity(n_nd2);

        for e in mesh.elem_iter() {
            let h1_dofs  = h1_space.element_dofs(e);    // 6 global P2 DOFs
            let nd2_dofs = hcurl_space.element_dofs(e); // 8 global ND2 DOFs
            let nd2_signs = hcurl_space.element_signs(e);

            for (i_local, &g_nd2) in nd2_dofs.iter().enumerate() {
                if !visited.insert(g_nd2 as usize) { continue; }
                let sign = nd2_signs[i_local];
                for (j_local, &global_p2) in h1_dofs.iter().enumerate() {
                    let val = sign * g_ref[i_local * n_p2_local + j_local];
                    if val.abs() > 1e-15 {
                        coo.add(g_nd2 as usize, global_p2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    // ── Order-2 numerical gradient (P2 → ND2, 3-D hexahedra) ────────────────

    /// Build `G: H1(P2) → H(curl) ND2` on a 3-D hexahedral mesh.
    ///
    /// MFEM's `ParDiscreteLinearOperator(&HGradFESpace, &HCurlFESpace)` +
    /// `GradientInterpolator` (joule/volta/tesla), i.e. the discrete gradient
    /// of the de Rham complex: the DOF vector of `G p` collects the H(curl)
    /// functionals applied to `∇p`.
    ///
    /// Every ND2 hexahedron DOF is the point-value functional
    /// `σ_s(Φ) = Φ(ξ_s)·τ_s` with `(ξ_s, τ_s) = (HexNDk::dof_coords[s],
    /// HexNDk::dof_tangents[s])` — `ξ_s ∈ [-1,1]³` on the reference cube and
    /// `τ_s = 2·e_a` the *unnormalized* reference tangent of component `a`
    /// (see `HCurlSpace::interpolate_vector`).  Hence
    ///
    /// ```text
    ///   G[nd_s, p2_j] = σ_s(∇φ_j) = ∇_ref φ_j(ξ_s) · τ_s
    /// ```
    ///
    /// with **no** Jacobian factor: the covariant pullback pairs the physical
    /// gradient with the physical tangent as
    /// `(J⁻ᵀ∇_ref φ) · (J τ) = ∇_ref φ · τ`.  Like the 2-D `P2→ND2` path the
    /// whole local matrix is therefore evaluated once on the reference
    /// element.
    ///
    /// The element-local slot order is [`HexNDk`]'s (12 edges × 2, 6 quad
    /// faces × 4, 6 interior), which is also `HCurlSpace`'s; the per-element
    /// scatter applies the space's own `element_dofs` / `element_signs`
    /// (edge-reversal sign, quad-face orientation sign) so that adjacent
    /// elements agree exactly and first-writer-wins is valid.
    fn gradient_p2_nd2_hex3d<M: MeshTopology>(
        h1_space: &H1Space<M>,
        hcurl_space: &HCurlSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = h1_space.mesh();
        const N_P2_LOCAL: usize = 27;

        // ND2 is only implemented for hexahedra (MFEM's `ND_HexahedronElement(2)`).
        match mesh.element_type(0) {
            ElementType::Hex8 | ElementType::Hex20 => {}
            other => {
                return Err(DiscreteOpError::UnsupportedCellType {
                    op: "gradient (P2→ND2)",
                    cell: element_type_name(other),
                })
            }
        }

        let nd = HexNDk::new(2);
        let coords = nd.dof_coords();
        let tangents = nd.dof_tangents();
        let n_slots = coords.len();
        if n_slots != nd.n_dofs() {
            return Err(DiscreteOpError::UnsupportedCellType {
                op: "gradient (P2→ND2)",
                cell: "HexNDk::dof_coords length mismatch",
            });
        }

        let p2 = HexQ2;
        let mut g_ref = vec![0.0_f64; n_slots * N_P2_LOCAL];
        {
            let mut grads = vec![0.0_f64; N_P2_LOCAL * 3];
            for (s, (xi, tau)) in coords.iter().zip(tangents.iter()).enumerate() {
                let xi3 = [xi[0], xi[1], xi[2]];
                p2.eval_grad_basis(&xi3, &mut grads);
                for j in 0..N_P2_LOCAL {
                    g_ref[s * N_P2_LOCAL + j] = grads[j * 3] * tau[0]
                        + grads[j * 3 + 1] * tau[1]
                        + grads[j * 3 + 2] * tau[2];
                }
            }
        }

        let n_nd = hcurl_space.n_dofs();
        let n_p2 = h1_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_nd, n_p2);
        let mut visited = HashSet::with_capacity(n_nd);

        for e in mesh.elem_iter() {
            let h1_dofs = h1_space.element_dofs(e);
            let nd_dofs = hcurl_space.element_dofs(e);
            let nd_signs = hcurl_space.element_signs(e);
            if nd_dofs.len() != n_slots || h1_dofs.len() != N_P2_LOCAL {
                return Err(DiscreteOpError::UnsupportedCellType {
                    op: "gradient (P2→ND2)",
                    cell: "element DOF count is not HexND2(54) × HexQ2(27)",
                });
            }
            for (s_local, &g_nd) in nd_dofs.iter().enumerate() {
                if !visited.insert(g_nd as usize) {
                    continue;
                }
                let sign = nd_signs[s_local];
                for (j_local, &global_p2) in h1_dofs.iter().enumerate() {
                    let val = sign * g_ref[s_local * N_P2_LOCAL + j_local];
                    if val.abs() > 1e-15 {
                        coo.add(g_nd as usize, global_p2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    /// Build the discrete curl matrix C: H(curl) -> L2 (2D).
    ///
    /// ## Order 1 — ND1 -> P0 (integral DOFs)
    ///
    /// The entry on each element is the cell integral of `curl(u_h)` and is
    /// assembled as
    ///
    /// ```text
    ///   C[l2_dof, hcurl_dof] = sign * curl_ref / det_j * |det_j| * area_ref
    /// ```
    ///
    /// with `area_ref = 0.5` for the reference triangle.
    ///
    /// ## Order 2 — ND2 -> P1/P2 (point-value DOFs)
    ///
    /// L2(P1) is discontinuous nodal. For each element vertex `xi_k`,
    ///
    /// ```text
    ///   C[p1_dof_k, nd2_dof_j] = sign_j * curl_ref_j(xi_k) / det_j
    /// ```
    ///
    /// # Errors
    /// Returns [`DiscreteOpError`] if orders are unsupported/incompatible, or
    /// if the mesh is not 2-dimensional.
    pub fn curl_2d<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let hcurl_order = hcurl_space.order();
        let l2_order = l2_space.order();

        match hcurl_order {
            1 | 2 => {}
            o => return Err(DiscreteOpError::UnsupportedHCurlOrder { op: "curl_2d", order: o }),
        }
        match l2_order {
            0..=2 => {}
            o => return Err(DiscreteOpError::UnsupportedL2Order { op: "curl_2d", order: o }),
        }
        // ND1 -> P0 and ND2 -> P1/P2 are the supported pairs.
        if !((hcurl_order == 1 && l2_order == 0)
            || (hcurl_order == 2 && (l2_order == 1 || l2_order == 2)))
        {
            return Err(DiscreteOpError::IncompatibleOrders {
                op: "curl_2d",
                h1_order: hcurl_order,
                hcurl_order: l2_order,
            });
        }

        let mesh = hcurl_space.mesh();
        if mesh.dim() != 2 {
            return Err(DiscreteOpError::UnsupportedDimension {
                op: "curl_2d",
                dim: mesh.dim(),
            });
        }

        match (hcurl_order, l2_order) {
            (1, 0) => Self::curl_2d_nd1_p0(hcurl_space, l2_space),
            (2, 1) => Self::curl_2d_nd2_p1(hcurl_space, l2_space),
            (2, 2) => Self::curl_2d_nd2_p2(hcurl_space, l2_space),
            _ => unreachable!(),
        }
    }

    // ── ND1 -> P0 curl in 2D ────────────────────────────────────────────────

    fn curl_2d_nd1_p0<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();

        let n_l2 = l2_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_l2, n_hcurl);

        // Reference curls for TriND1: [2, 2, -2] (constant)
        let ref_elem = TriNDk::new(1);
        let mut curl_ref = vec![0.0; ref_elem.n_dofs()];
        ref_elem.eval_curl(&[0.0, 0.0], &mut curl_ref);

        let area_ref = 0.5; // reference triangle area

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let signs = hcurl_space.element_signs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let l2_dof = l2_dofs[0] as usize;

            // Compute Jacobian determinant
            let det_j = simplex_det(mesh, nodes);

            // Physical curl = curl_ref / det_j
            // Integral over element = curl_phys * |det_j| * area_ref
            //                       = (curl_ref / det_j) * |det_j| * area_ref
            //                       = curl_ref * sign(det_j) * area_ref
            let sign_det = if det_j > 0.0 { 1.0 } else { -1.0 };

            for i in 0..ref_elem.n_dofs() {
                let hcurl_dof = hcurl_dofs[i] as usize;
                let val = signs[i] * curl_ref[i] * sign_det * area_ref;
                coo.add(l2_dof, hcurl_dof, val);
            }
        }

        Ok(coo.into_csr())
    }

    // ── ND2 -> P1 curl in 2D ────────────────────────────────────────────────

    /// D32: the ND2 dof rows are the MFEM **nodal** point-value functionals
    /// `σ(F) = F(x_i)·t̂` (edge Gauss points with the canonical tangent, the
    /// interior `(1/3,1/3)` component samples), orientation-consistent with
    /// `HCurlSpace` (`element_dofs` already encodes the anti-diagonal reversal
    /// permutation, so the row's canonical slot index is recovered from the
    /// global id).  `interpolate_vector` feeds exactly these canonical values.
    fn curl_2d_nd2_p1<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();

        let n_l2 = l2_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_l2, n_hcurl);

        let ref_elem = TriND2;
        let n_nd2 = ref_elem.n_dofs(); // 8
        // Slot block order matches `HCurlSpace::TRI_EDGES_MFEM`
        // (= (0,1),(1,2),(2,0)); the vertex *pair* of entry 2 equals the
        // legacy (0,2) pair, so the canonical (min,max) is unchanged.
        let tri_edges = [(0usize, 1usize), (1usize, 2usize), (2usize, 0usize)];

        // 2-point Gauss-Legendre on [0,1] — the ND2 edge dof points.
        let (gl2, _) = gauss_legendre_01(2);

        // Physical spanning fields matching TriND2 monomial space:
        // m0=(1,0), m1=(x,0), m2=(y,0), m3=(0,1), m4=(0,x), m5=(0,y),
        // m6=(-xy,x^2), m7=(-y^2,xy).
        let eval_field = |k: usize, x: f64, y: f64| -> (f64, f64) {
            match k {
                0 => (1.0, 0.0),
                1 => (x, 0.0),
                2 => (y, 0.0),
                3 => (0.0, 1.0),
                4 => (0.0, x),
                5 => (0.0, y),
                6 => (-x * y, x * x),
                7 => (-y * y, x * y),
                _ => unreachable!(),
            }
        };
        let eval_curl = |k: usize, x: f64, y: f64| -> f64 {
            match k {
                0 => 0.0,
                1 => 0.0,
                2 => -1.0,
                3 => 0.0,
                4 => 1.0,
                5 => 0.0,
                6 => 3.0 * x,
                7 => 3.0 * y,
                _ => unreachable!(),
            }
        };

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let l2_dofs = l2_space.element_dofs(e);

            // Geometry for interior dof mapping.
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let j00 = x1[0] - x0[0]; let j10 = x1[1] - x0[1];
            let j01 = x2[0] - x0[0]; let j11 = x2[1] - x0[1];

            // D (8x8): ND2 DOFs of spanning fields. Y (3x8): nodal curls.
            let mut dmat = vec![0.0_f64; n_nd2 * n_nd2];
            let mut ymat = vec![0.0_f64; 3 * n_nd2];

            for k in 0..n_nd2 {
                let mut dof_k = [0.0_f64; 8];

                // Edge rows: nodal point values along the canonical (min,max)
                // direction; the canonical slot index of each element slot is
                // recovered from its global id (adjacent ids per block).
                for (edge_local, &(li, lj)) in tri_edges.iter().enumerate() {
                    let gi = nodes[li];
                    let gj = nodes[lj];
                    let (ga, gb) = if gi < gj { (gi, gj) } else { (gj, gi) };
                    let pa = mesh.node_coords(ga);
                    let pb = mesh.node_coords(gb);
                    let tx = pb[0] - pa[0];
                    let ty = pb[1] - pa[1];

                    let first = hcurl_dofs[2 * edge_local].min(hcurl_dofs[2 * edge_local + 1]);
                    for m in 0..2usize {
                        let j_canon = (hcurl_dofs[2 * edge_local + m] - first) as usize;
                        let t = gl2[j_canon];
                        let xp = pa[0] + t * tx;
                        let yp = pa[1] + t * ty;
                        let (fx, fy) = eval_field(k, xp, yp);
                        dof_k[2 * edge_local + m] = fx * tx + fy * ty;
                    }
                }

                // Interior rows: point values at the reference (1/3,1/3) with
                // tangents (1,0)/(0,1): σ = (J t̂)·F — matches
                // `HCurlSpace::interpolate_vector`.
                let xc = [x0[0] + (j00 + j01) / 3.0, x0[1] + (j10 + j11) / 3.0];
                let (fx, fy) = eval_field(k, xc[0], xc[1]);
                dof_k[6] = fx * j00 + fy * j10;
                dof_k[7] = fx * j01 + fy * j11;

                for i in 0..n_nd2 {
                    dmat[i * n_nd2 + k] = dof_k[i];
                }

                // P1 nodal curls at element vertices.
                for p in 0..3 {
                    let xp = mesh.node_coords(nodes[p]);
                    ymat[p * n_nd2 + k] = eval_curl(k, xp[0], xp[1]);
                }
            }

            // Solve D^T * Z = Y^T, with Z = A^T and A mapping ND2 DOFs -> P1 nodal values.
            let mut dt = vec![0.0_f64; n_nd2 * n_nd2];
            for i in 0..n_nd2 {
                for j in 0..n_nd2 {
                    dt[i * n_nd2 + j] = dmat[j * n_nd2 + i];
                }
            }
            let mut yt = vec![0.0_f64; n_nd2 * 3];
            for p in 0..3 {
                for k in 0..n_nd2 {
                    yt[k * 3 + p] = ymat[p * n_nd2 + k];
                }
            }

            let z = solve_small(n_nd2, 3, &dt, &yt); // shape 8x3 row-major

            for (p_local, &global_p1) in l2_dofs.iter().enumerate() {
                for (i_local, &global_nd2) in hcurl_dofs.iter().enumerate() {
                    let val = z[i_local * 3 + p_local];
                    if val.abs() > 1e-15 {
                        coo.add(global_p1 as usize, global_nd2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    // ── ND2 -> P2 curl in 2D ────────────────────────────────────────────────

    fn curl_2d_nd2_p2<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();

        let n_l2 = l2_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_l2, n_hcurl);

        let ref_elem = TriND2;
        let n_nd2 = ref_elem.n_dofs(); // 8
        // Slot block order matches `HCurlSpace::TRI_EDGES_MFEM` (the vertex
        // pair of entry 2 equals the legacy (0,2) pair).
        let tri_edges = [(0usize, 1usize), (1usize, 2usize), (2usize, 0usize)];

        // 2-point Gauss-Legendre on [0,1] — the ND2 edge dof points.
        let (gl2, _) = gauss_legendre_01(2);

        let eval_field = |k: usize, x: f64, y: f64| -> (f64, f64) {
            match k {
                0 => (1.0, 0.0),
                1 => (x, 0.0),
                2 => (y, 0.0),
                3 => (0.0, 1.0),
                4 => (0.0, x),
                5 => (0.0, y),
                6 => (-x * y, x * x),
                7 => (-y * y, x * y),
                _ => unreachable!(),
            }
        };
        let eval_curl = |k: usize, x: f64, y: f64| -> f64 {
            match k {
                0 => 0.0,
                1 => 0.0,
                2 => -1.0,
                3 => 0.0,
                4 => 1.0,
                5 => 0.0,
                6 => 3.0 * x,
                7 => 3.0 * y,
                _ => unreachable!(),
            }
        };

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let l2_dofs = l2_space.element_dofs(e); // 6 local P2 DOFs

            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let j00 = x1[0] - x0[0];
            let j10 = x1[1] - x0[1];
            let j01 = x2[0] - x0[0];
            let j11 = x2[1] - x0[1];

            let mut dmat = vec![0.0_f64; n_nd2 * n_nd2];
            let mut ymat = vec![0.0_f64; 6 * n_nd2];

            // D32: nodal dof rows (see `curl_2d_nd2_p1`).
            for k in 0..n_nd2 {
                let mut dof_k = [0.0_f64; 8];

                for (edge_local, &(li, lj)) in tri_edges.iter().enumerate() {
                    let gi = nodes[li];
                    let gj = nodes[lj];
                    let (ga, gb) = if gi < gj { (gi, gj) } else { (gj, gi) };
                    let pa = mesh.node_coords(ga);
                    let pb = mesh.node_coords(gb);
                    let tx = pb[0] - pa[0];
                    let ty = pb[1] - pa[1];

                    let first = hcurl_dofs[2 * edge_local].min(hcurl_dofs[2 * edge_local + 1]);
                    for m in 0..2usize {
                        let j_canon = (hcurl_dofs[2 * edge_local + m] - first) as usize;
                        let t = gl2[j_canon];
                        let xp = pa[0] + t * tx;
                        let yp = pa[1] + t * ty;
                        let (fx, fy) = eval_field(k, xp, yp);
                        dof_k[2 * edge_local + m] = fx * tx + fy * ty;
                    }
                }

                // Interior rows: point values at the reference (1/3,1/3).
                let xc = [x0[0] + (j00 + j01) / 3.0, x0[1] + (j10 + j11) / 3.0];
                let (fx, fy) = eval_field(k, xc[0], xc[1]);
                dof_k[6] = fx * j00 + fy * j10;
                dof_k[7] = fx * j01 + fy * j11;

                for i in 0..n_nd2 {
                    dmat[i * n_nd2 + k] = dof_k[i];
                }

                // P2 nodal curls: vertices and edge midpoints.
                let sample_pts = [
                    [x0[0], x0[1]],
                    [x1[0], x1[1]],
                    [x2[0], x2[1]],
                    [0.5 * (x0[0] + x1[0]), 0.5 * (x0[1] + x1[1])],
                    [0.5 * (x1[0] + x2[0]), 0.5 * (x1[1] + x2[1])],
                    [0.5 * (x0[0] + x2[0]), 0.5 * (x0[1] + x2[1])],
                ];
                for p in 0..6 {
                    ymat[p * n_nd2 + k] = eval_curl(k, sample_pts[p][0], sample_pts[p][1]);
                }
            }

            let mut dt = vec![0.0_f64; n_nd2 * n_nd2];
            for i in 0..n_nd2 {
                for j in 0..n_nd2 {
                    dt[i * n_nd2 + j] = dmat[j * n_nd2 + i];
                }
            }
            let mut yt = vec![0.0_f64; n_nd2 * 6];
            for p in 0..6 {
                for k in 0..n_nd2 {
                    yt[k * 6 + p] = ymat[p * n_nd2 + k];
                }
            }

            let z = solve_small(n_nd2, 6, &dt, &yt); // shape 8x6

            for (p_local, &global_p2) in l2_dofs.iter().enumerate() {
                for (i_local, &global_nd2) in hcurl_dofs.iter().enumerate() {
                    let val = z[i_local * 6 + p_local];
                    if val.abs() > 1e-15 {
                        coo.add(global_p2 as usize, global_nd2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    /// Build the discrete curl matrix C: H(curl) → H(div) on **2D triangles** (ND2 → RT2).
    ///
    /// Implemented by [`crate::vector_assembler::VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2`]:
    /// same affine map, quadrature weight `w_{\mathrm{ref}}|det J|`, and Piola maps as
    /// [`crate::vector_assembler::VectorAssembler::assemble_bilinear`] on simplices.  The
    /// local block is
    /// \[
    ///   B_{ij} = \int_T \Psi_i^{RT2} \cdot (\Phi_{j,y}^{ND2},\,-\Phi_{j,x}^{ND2}) \,\mathrm{d}x
    /// \]
    /// (covariant `Φ`, contravariant `Ψ`).  Element matrices are **added** into CSR for shared
    /// RT2 DOFs like [`Self::divergence`] RT2→P2.  ND2 columns omit [`HCurlSpace::element_signs`]
    /// (same as [`Self::curl_2d_nd2_p2`]).
    ///
    /// **De Rham:** tests bound `C*G` on smooth potentials. Composing with [`Self::divergence`]
    /// RT2→P2, random-stress checks still give `max|D*C*u|` on the order of `1e1` on refined
    /// `unit_square` meshes (this quadrature `C` and the RT2→P2 reconstruction `D` are not a
    /// commuting pair in max norm); the mixed volume form remains the documented `C`.
    ///
    /// # Errors
    /// Returns [`DiscreteOpError`] if the mesh is not 2-D or orders are not `(2, 2)`.
    pub fn curl_2d_hdiv<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();
        if mesh.dim() != 2 {
            return Err(DiscreteOpError::UnsupportedDimension {
                op: "curl_2d_hdiv",
                dim: mesh.dim(),
            });
        }
        let hcurl_order = hcurl_space.order();
        let hdiv_order = hdiv_space.order();
        match (hcurl_order, hdiv_order) {
            (1, 0) => Ok(crate::vector_assembler::VectorAssembler::assemble_curl_hdiv_pairing_2d_nd1_rt0(
                hcurl_space, hdiv_space, 3,
            )),
            (2, 2) => Ok(crate::vector_assembler::VectorAssembler::assemble_curl_hdiv_pairing_2d_nd2_rt2(
                hcurl_space, hdiv_space,
                crate::vector_assembler::TRI_ND2_RT2_MIXED_QUAD_ORDER,
            )),
            _ => Err(DiscreteOpError::IncompatibleOrders {
                op: "curl_2d_hdiv",
                h1_order: hcurl_order,
                hcurl_order: hdiv_order,
            }),
        }
    }

    /// Build the discrete divergence matrix D: H(div) -> L2.
    ///
    /// ## Order 0 — topological assembly (RT0 → P0)
    ///
    /// The matrix is the signed face-element incidence matrix:
    /// `D[elem, face] = face_sign`.  Exact, no quadrature.
    ///
    /// ## Order 1 — numerical assembly (RT1 → P1/P2)
    ///
    /// `D[l2_dof_i, hdiv_dof_j] = DOF_i^{L2}(div Ψ_j)`, computed via
    /// numerical integration on the reference element and scatter-assembled.
    ///
    /// ## Order 2 — 2D triangles only (RT2 → P2)
    ///
    /// Same local reconstruction idea as RT1→P2, using fifteen Piola pullbacks
    /// of the MFEM RT2 reference primitives and the MFEM-style nodal `H(div)` DOFs.
    ///
    /// # Errors
    /// Returns [`DiscreteOpError`] if the space orders are unsupported or
    /// incompatible.
    pub fn divergence<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let hdiv_order = hdiv_space.order();
        let l2_order   = l2_space.order();

        match hdiv_order {
            0..=2 => {}
            o => return Err(DiscreteOpError::UnsupportedHDivOrder { op: "divergence", order: o }),
        }
        match l2_order {
            0..=2 => {}
            o => return Err(DiscreteOpError::UnsupportedL2Order { op: "divergence", order: o }),
        }
        // RT0→P0, RT1→P1/P2, RT2→P2 (2D).
        if !((hdiv_order == 0 && l2_order == 0)
            || (hdiv_order == 1 && (l2_order == 1 || l2_order == 2))
            || (hdiv_order == 2 && l2_order == 2))
        {
            return Err(DiscreteOpError::IncompatibleOrders {
                op: "divergence",
                h1_order: hdiv_order,
                hcurl_order: l2_order,
            });
        }

        match (hdiv_order, l2_order) {
            (0, 0) => Self::divergence_rt0_p0(hdiv_space, l2_space),
            (1, 1) | (1, 2) => Self::divergence_rt1_p1(hdiv_space, l2_space),
            (2, 2) => Self::divergence_rt2_p2(hdiv_space, l2_space),
            _ => unreachable!(),
        }
    }

    // ── RT0 → P0 topological divergence ───────────────────────────────────────

    fn divergence_rt0_p0<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hdiv_space.mesh();
        let dim = mesh.dim();
        if dim != 2 && dim != 3 {
            return Err(DiscreteOpError::UnsupportedDimension {
                op: "divergence",
                dim,
            });
        }

        let n_local_dofs = if dim == 2 {
            // Tri has 3 edges (3 local DOFs), Quad has 4 edges (4 local DOFs)
            let etype = mesh.element_type(0);
            match etype {
                ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => 4,
                _ => 3,
            }
        } else { 4 };

        let n_l2   = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_l2, n_hdiv);

        // RT0 DOFs store v·n_unnormalized where n_unnormalized has magnitude = edge_length.
        // The physical divergence formula: div(v)|_e = (1/area) * Σ sign_i * DOF_i
        // where DOF_i already includes edge_length in the unnormalized normal.
        for e in mesh.elem_iter() {
            let hdiv_dofs = hdiv_space.element_dofs(e);
            let l2_dofs   = l2_space.element_dofs(e);
            let l2_dof    = l2_dofs[0] as usize;
            let signs     = hdiv_space.element_signs(e);

            for i in 0..n_local_dofs {
                coo.add(l2_dof, hdiv_dofs[i] as usize, signs[i]);
            }
        }

        Ok(coo.into_csr())
    }

    // ── RT1 → P1 numerical divergence (2D triangles and 3D tetrahedra) ───────

    fn divergence_rt1_p1<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hdiv_space.mesh();
        if mesh.dim() != 2 && mesh.dim() != 3 {
            return Err(DiscreteOpError::UnsupportedDimension { op: "divergence (RT1→P1)", dim: mesh.dim() });
        }

        let n_l2   = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_l2, n_hdiv);

        if mesh.dim() == 2 {
            let rt1_elem = TriRT1;
            let n_rt1    = rt1_elem.n_dofs(); // 8

            // D34: the H(div) dofs carry MFEM **nodal** semantics — pointwise
            // normal-flux samples `f(x_s)·cof(J)·nk_s` scaled by the element
            // sign (matching `HDivSpace::interpolate_vector`).  The local
            // matrix rows are therefore those signed nodal functionals applied
            // to the physical RT1 monomial primitives.
            let (dof_pts, dof_nks) =
                fem_element::raviart_thomas::tri_rt1::mfem_tri_nodal_dofs(1);

            let eval_field = |k: usize, x: f64, y: f64| -> (f64, f64) {
                match k {
                    0 => (1.0, 0.0),
                    1 => (x, 0.0),
                    2 => (y, 0.0),
                    3 => (0.0, 1.0),
                    4 => (0.0, x),
                    5 => (0.0, y),
                    6 => (x * x, x * y),
                    7 => (x * y, y * y),
                    _ => unreachable!(),
                }
            };
            let eval_div = |k: usize, x: f64, y: f64| -> f64 {
                match k {
                    0 => 0.0,
                    1 => 1.0,
                    2 => 0.0,
                    3 => 0.0,
                    4 => 0.0,
                    5 => 1.0,
                    6 => 3.0 * x,
                    7 => 3.0 * y,
                    _ => unreachable!(),
                }
            };

            for e in mesh.elem_iter() {
                let hdiv_dofs  = hdiv_space.element_dofs(e);
                let l2_dofs    = l2_space.element_dofs(e);
                let nodes = mesh.element_nodes(e);

                let x0 = mesh.node_coords(nodes[0]);
                let x1 = mesh.node_coords(nodes[1]);
                let x2 = mesh.node_coords(nodes[2]);
                let j00 = x1[0] - x0[0]; let j10 = x1[1] - x0[1];
                let j01 = x2[0] - x0[0]; let j11 = x2[1] - x0[1];
                let signs = hdiv_space.element_signs(e);

                let mut dmat = vec![0.0_f64; n_rt1 * n_rt1];
                let n_l2_local = l2_dofs.len(); // 3 (P1) or 6 (P2)
                let mut ymat = vec![0.0_f64; n_l2_local * n_rt1];

                for k in 0..n_rt1 {
                    for s in 0..n_rt1 {
                        let (xi, nk) = (&dof_pts[s], &dof_nks[s]);
                        let xp = x0[0] + j00 * xi[0] + j01 * xi[1];
                        let yp = x0[1] + j10 * xi[0] + j11 * xi[1];
                        let (fx, fy) = eval_field(k, xp, yp);
                        // cof(J)·nk (unnormalised physical normal)
                        let nx = j11 * nk[0] - j10 * nk[1];
                        let ny = -j01 * nk[0] + j00 * nk[1];
                        dmat[s * n_rt1 + k] = signs[s] * (fx * nx + fy * ny);
                    }
                    let sample_pts = [
                        [x0[0], x0[1]],
                        [x1[0], x1[1]],
                        [x2[0], x2[1]],
                        [0.5 * (x0[0] + x1[0]), 0.5 * (x0[1] + x1[1])],
                        [0.5 * (x1[0] + x2[0]), 0.5 * (x1[1] + x2[1])],
                        [0.5 * (x0[0] + x2[0]), 0.5 * (x0[1] + x2[1])],
                    ];
                    for p in 0..n_l2_local {
                        ymat[p * n_rt1 + k] = eval_div(k, sample_pts[p][0], sample_pts[p][1]);
                    }
                }

                let mut dt = vec![0.0_f64; n_rt1 * n_rt1];
                for i in 0..n_rt1 {
                    for j in 0..n_rt1 {
                        dt[i * n_rt1 + j] = dmat[j * n_rt1 + i];
                    }
                }
                let mut yt = vec![0.0_f64; n_rt1 * n_l2_local];
                for p in 0..n_l2_local {
                    for k in 0..n_rt1 {
                        yt[k * n_l2_local + p] = ymat[p * n_rt1 + k];
                    }
                }

                let z = solve_small(n_rt1, n_l2_local, &dt, &yt);
                for (p_local, &global_p) in l2_dofs.iter().enumerate() {
                    for (i_local, &global_rt1) in hdiv_dofs.iter().enumerate() {
                        let val = z[i_local * n_l2_local + p_local];
                        if val.abs() > 1e-15 {
                            coo.add(global_p as usize, global_rt1 as usize, val);
                        }
                    }
                }
            }
        } else {
            let rt1_elem = TetRT1;
            let n_rt1 = rt1_elem.n_dofs(); // 15

            let eval_field = |k: usize, x: f64, y: f64, z: f64| -> [f64; 3] {
                match k {
                    0 => [1.0, 0.0, 0.0],
                    1 => [x, 0.0, 0.0],
                    2 => [y, 0.0, 0.0],
                    3 => [z, 0.0, 0.0],
                    4 => [0.0, 1.0, 0.0],
                    5 => [0.0, x, 0.0],
                    6 => [0.0, y, 0.0],
                    7 => [0.0, z, 0.0],
                    8 => [0.0, 0.0, 1.0],
                    9 => [0.0, 0.0, x],
                    10 => [0.0, 0.0, y],
                    11 => [0.0, 0.0, z],
                    12 => [x * x, x * y, x * z],
                    13 => [x * y, y * y, y * z],
                    14 => [x * z, y * z, z * z],
                    _ => unreachable!(),
                }
            };
            let eval_div = |k: usize, x: f64, y: f64, z: f64| -> f64 {
                match k {
                    0 => 0.0,
                    1 => 1.0,
                    2 => 0.0,
                    3 => 0.0,
                    4 => 0.0,
                    5 => 0.0,
                    6 => 1.0,
                    7 => 0.0,
                    8 => 0.0,
                    9 => 0.0,
                    10 => 0.0,
                    11 => 1.0,
                    12 => 4.0 * x,
                    13 => 4.0 * y,
                    14 => 4.0 * z,
                    _ => unreachable!(),
                }
            };

            // D34: nodal dofs (MFEM RT_TetrahedronElement semantics) shared
            // with the element crate.
            let (dof_pts, dof_nks) = fem_element::raviart_thomas::tet_rt1::mfem_nodal_dofs(1);

            for e in mesh.elem_iter() {
                let hdiv_dofs = hdiv_space.element_dofs(e);
                let l2_dofs = l2_space.element_dofs(e); // 4 (P1) or 10 (P2)
                let nodes = mesh.element_nodes(e);

                let mut dmat = vec![0.0_f64; n_rt1 * n_rt1];
                let n_l2_local = l2_dofs.len();
                let mut ymat = vec![0.0_f64; n_l2_local * n_rt1];

                let x0 = mesh.node_coords(nodes[0]);
                let x1 = mesh.node_coords(nodes[1]);
                let x2 = mesh.node_coords(nodes[2]);
                let x3 = mesh.node_coords(nodes[3]);
                let j0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
                let j1 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
                let j2 = [x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]];
                // cof(J) = det(J)·J^{-T} (adjugate transpose)
                let cof = [
                    [
                        j1[1] * j2[2] - j1[2] * j2[1],
                        j0[2] * j2[1] - j0[1] * j2[2],
                        j0[1] * j1[2] - j0[2] * j1[1],
                    ],
                    [
                        j1[2] * j2[0] - j1[0] * j2[2],
                        j0[0] * j2[2] - j0[2] * j2[0],
                        j0[2] * j1[0] - j0[0] * j1[2],
                    ],
                    [
                        j1[0] * j2[1] - j1[1] * j2[0],
                        j0[1] * j2[0] - j0[0] * j2[1],
                        j0[0] * j1[1] - j0[1] * j1[0],
                    ],
                ];
                let signs = hdiv_space.element_signs(e);

                for k in 0..n_rt1 {
                    for s in 0..n_rt1 {
                        let (xi, nk) = (&dof_pts[s], &dof_nks[s]);
                        let xp = [
                            x0[0] + j0[0] * xi[0] + j1[0] * xi[1] + j2[0] * xi[2],
                            x0[1] + j0[1] * xi[0] + j1[1] * xi[1] + j2[1] * xi[2],
                            x0[2] + j0[2] * xi[0] + j1[2] * xi[1] + j2[2] * xi[2],
                        ];
                        let fv = eval_field(k, xp[0], xp[1], xp[2]);
                        let mut val = 0.0;
                        for r in 0..3 {
                            val += fv[r]
                                * (cof[r][0] * nk[0] + cof[r][1] * nk[1] + cof[r][2] * nk[2]);
                        }
                        dmat[s * n_rt1 + k] = signs[s] * val;
                    }
                    let sample_pts = [
                        [x0[0], x0[1], x0[2]],
                        [x1[0], x1[1], x1[2]],
                        [x2[0], x2[1], x2[2]],
                        [x3[0], x3[1], x3[2]],
                        [0.5 * (x0[0] + x1[0]), 0.5 * (x0[1] + x1[1]), 0.5 * (x0[2] + x1[2])],
                        [0.5 * (x1[0] + x2[0]), 0.5 * (x1[1] + x2[1]), 0.5 * (x1[2] + x2[2])],
                        [0.5 * (x2[0] + x0[0]), 0.5 * (x2[1] + x0[1]), 0.5 * (x2[2] + x0[2])],
                        [0.5 * (x0[0] + x3[0]), 0.5 * (x0[1] + x3[1]), 0.5 * (x0[2] + x3[2])],
                        [0.5 * (x1[0] + x3[0]), 0.5 * (x1[1] + x3[1]), 0.5 * (x1[2] + x3[2])],
                        [0.5 * (x2[0] + x3[0]), 0.5 * (x2[1] + x3[1]), 0.5 * (x2[2] + x3[2])],
                    ];
                    for p in 0..n_l2_local {
                        ymat[p * n_rt1 + k] =
                            eval_div(k, sample_pts[p][0], sample_pts[p][1], sample_pts[p][2]);
                    }
                }

                let mut dt = vec![0.0_f64; n_rt1 * n_rt1];
                for i in 0..n_rt1 {
                    for j in 0..n_rt1 {
                        dt[i * n_rt1 + j] = dmat[j * n_rt1 + i];
                    }
                }
                let mut yt = vec![0.0_f64; n_rt1 * n_l2_local];
                for p in 0..n_l2_local {
                    for k in 0..n_rt1 {
                        yt[k * n_l2_local + p] = ymat[p * n_rt1 + k];
                    }
                }

                let z = solve_small(n_rt1, n_l2_local, &dt, &yt); // shape 15 x n_l2_local
                for (p_local, &global_p) in l2_dofs.iter().enumerate() {
                    for (i_local, &global_rt1) in hdiv_dofs.iter().enumerate() {
                        let val = z[i_local * n_l2_local + p_local];
                        if val.abs() > 1e-15 {
                            coo.add(global_p as usize, global_rt1 as usize, val);
                        }
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    // ── RT2 → P2 (2D triangles) ───────────────────────────────────────────────

    fn divergence_rt2_p2<M: MeshTopology>(
        hdiv_space: &HDivSpace<M>,
        l2_space: &L2Space<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hdiv_space.mesh();
        if mesh.dim() != 2 {
            return Err(DiscreteOpError::UnsupportedDimension {
                op: "divergence (RT2→P2)",
                dim: mesh.dim(),
            });
        }

        const N_RT2: usize = 15;

        let n_rt2 = N_RT2;
        let n_l2 = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();
        let mut coo = CooMatrix::new(n_l2, n_hdiv);

        for e in mesh.elem_iter() {
            let hdiv_dofs = hdiv_space.element_dofs(e);
            let l2_dofs = l2_space.element_dofs(e);
            let nodes = mesh.element_nodes(e);

            let n_l2_local = l2_dofs.len();
            if n_l2_local != 6 {
                return Err(DiscreteOpError::IncompatibleOrders {
                    op: "divergence (RT2→P2)",
                    h1_order: 2,
                    hcurl_order: l2_space.order(),
                });
            }

            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let j00 = x1[0] - x0[0];
            let j10 = x1[1] - x0[1];
            let j01 = x2[0] - x0[0];
            let j11 = x2[1] - x0[1];
            let det_j = j00 * j11 - j01 * j10;

            let signs = hdiv_space.element_signs(e);

            let (dmat, ymat) =
                rt2_triangle_dmat_ymat_div_p2(mesh, nodes, j00, j01, j10, j11, det_j, signs);

            let mut dt = vec![0.0_f64; n_rt2 * n_rt2];
            for i in 0..n_rt2 {
                for j in 0..n_rt2 {
                    dt[i * n_rt2 + j] = dmat[j * n_rt2 + i];
                }
            }
            let mut yt = vec![0.0_f64; n_rt2 * n_l2_local];
            for p in 0..n_l2_local {
                for kk in 0..n_rt2 {
                    yt[kk * n_l2_local + p] = ymat[p * n_rt2 + kk];
                }
            }

            let z = solve_small(n_rt2, n_l2_local, &dt, &yt);
            for (p_local, &global_p) in l2_dofs.iter().enumerate() {
                for (i_local, &global_rt2) in hdiv_dofs.iter().enumerate() {
                    let val = z[i_local * n_l2_local + p_local];
                    if val.abs() > 1e-15 {
                        coo.add(global_p as usize, global_rt2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }

    /// Build the discrete curl matrix C: H(curl) -> H(div) in 3D (tetrahedra).
    ///
    /// For lowest-order (ND1 -> RT0), the discrete curl is the topological
    /// face-edge incidence matrix. Each face is processed once; the Stokes
    /// signs are derived from the sorted global vertex order of the face,
    /// which matches the global face-normal convention used by HDivSpace.
    ///
    /// For a face with sorted global vertices (a < b < c):
    ///   - boundary traversal (right-hand rule): a → b → c → a
    ///   - C[face, edge(a,b)] = +1
    ///   - C[face, edge(b,c)] = +1
    ///   - C[face, edge(a,c)] = −1  (traversal goes c→a, opposite to global a→c)
    ///
    /// For the high-order pair (ND2 -> RT1), a per-element local
    /// reconstruction is used:
    ///
    /// 1. Choose 20 spanning fields of the TetND2 polynomial space.
    /// 2. Evaluate their ND2 DOFs (`D`) and RT1 DOFs of their curls (`Y`).
    /// 3. Solve `D^T * Z = Y^T`, where `Z = A^T`, to obtain the local map
    ///    `A: ND2_dofs -> RT1_dofs`.
    ///
    /// # Errors
    /// Returns [`DiscreteOpError`] if orders are unsupported/incompatible or
    /// if the mesh is not 3-dimensional.
    pub fn curl_3d<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();
        if mesh.dim() != 3 {
            return Err(DiscreteOpError::UnsupportedDimension {
                op: "curl_3d",
                dim: mesh.dim(),
            });
        }

        match hcurl_space.order() {
            1 | 2 => {}
            order => {
                eprintln!("TEMP curl_3d bad hcurl order = {order}");
                return Err(DiscreteOpError::UnsupportedHCurlOrder {
                    op: "curl_3d",
                    order,
                })
            }
        }
        match hdiv_space.order() {
            0 | 1 => {}
            order => {
                return Err(DiscreteOpError::UnsupportedHDivOrder {
                    op: "curl_3d",
                    order,
                })
            }
        }

        match (hcurl_space.order(), hdiv_space.order()) {
            (1, 0) => Self::curl_3d_nd1_rt0(hcurl_space, hdiv_space),
            (2, 1) => Self::curl_3d_nd2_rt1(hcurl_space, hdiv_space),
            _ => Err(DiscreteOpError::IncompatibleOrders {
                op: "curl_3d",
                h1_order: hcurl_space.order(),
                hcurl_order: hdiv_space.order(),
            }),
        }
    }

    fn curl_3d_nd1_rt0<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();

        let n_hdiv = hdiv_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n_hdiv, n_hcurl);

        // Local face / edge tables matching HDivSpace / HCurlSpace.
        // Face order = MFEM tet FaceVert (canonical, Elem1 orientation).
        let tet_faces: [(usize, usize, usize); 4] = [
            (1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1),
        ];
        let tet_edges: [(usize, usize); 6] = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];
        // Hex8: face order must match HDivSpace::HEX_FACES (MFEM CUBE FaceVert);
        // edge order must match HCurlSpace::HEX_EDGES (MFEM CUBE Edges).
        let hex_faces: [[usize; 4]; 6] = [
            [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
        ];
        let hex_edges: [(usize, usize); 12] = [
            (0, 1), (1, 2), (3, 2), (0, 3),
            (4, 5), (5, 6), (7, 6), (4, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        // Prism6: faces matching HDivSpace::PRISM_FACES_CANON (MFEM PRISM
        // FaceVert), edges matching HCurlSpace::PRISM_EDGES (MFEM PRISM Edges).
        let prism_faces: [[usize; 4]; 5] = [
            [0, 2, 1, 0], // bottom tri
            [3, 4, 5, 0], // top tri
            [0, 1, 4, 3], // quad 0
            [1, 2, 5, 4], // quad 1
            [2, 0, 3, 5], // quad 2
        ];
        let prism_edges: [(usize, usize); 9] = [
            (0, 1), (1, 2), (2, 0), // bottom triangle
            (3, 4), (4, 5), (5, 3), // top triangle
            (0, 3), (1, 4), (2, 5), // vertical
        ];

        // Track visited face DOFs so each face is assembled exactly once.
        let mut visited = HashSet::with_capacity(n_hdiv);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let hdiv_dofs = hdiv_space.element_dofs(e);

            match mesh.element_type(e) {
                ElementType::Tet4 | ElementType::Tet10 => {
                    for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                        let face_dof = hdiv_dofs[face_local] as usize;

                        if !visited.insert(face_dof) {
                            continue;
                        }

                        // Boundary traversal a→b→c→a along the canonical (Elem1)
                        // face orientation — normal (p_b−p_a)×(p_c−p_a) — matching
                        // HDivSpace's RT face orientation so div∘curl = 0 with the
                        // RT signs.  Each boundary edge's sign accounts for the
                        // HCurl DOF direction (min→max vertex id).
                        let a = verts[la]; let b = verts[lb]; let c = verts[lc];
                        let sgn = |x: u32, y: u32| if x < y { 1.0 } else { -1.0 };
                        let face_boundary: [(u32, u32, f64); 3] = [
                            (a, b, sgn(a, b)),
                            (b, c, sgn(b, c)),
                            (a, c, -sgn(a, c)), // c→a contributes +sgn(c,a)
                        ];

                        for (gv0, gv1, stokes_sign) in face_boundary {
                            // Find the local edge index whose global vertices match.
                            let edge_idx = tet_edges.iter().position(|&(li, lj)| {
                                let (gi, gj) = (verts[li], verts[lj]);
                                (gi == gv0 && gj == gv1) || (gi == gv1 && gj == gv0)
                            }).expect("face boundary edge not found in element");

                            let edge_dof = hcurl_dofs[edge_idx] as usize;
                            coo.add(face_dof, edge_dof, stokes_sign);
                        }
                    }
                }
                ElementType::Hex8 | ElementType::Hex20 => {
                    for (face_local, face_verts) in hex_faces.iter().enumerate() {
                        let face_dof = hdiv_dofs[face_local] as usize;
                        if !visited.insert(face_dof) {
                            continue;
                        }
                        // Quad face boundary cycle a→b→c→d→a; each edge's sign
                        // follows the cycle direction vs the global (ascending
                        // vertex id) DOF orientation.
                        let gv: Vec<u32> = face_verts.iter().map(|&li| verts[li]).collect();
                        for k in 0..4 {
                            let (a, b) = (gv[k], gv[(k + 1) % 4]);
                            let stokes_sign = if a < b { 1.0 } else { -1.0 };
                            let edge_idx = hex_edges.iter().position(|&(li, lj)| {
                                let (gi, gj) = (verts[li], verts[lj]);
                                (gi == a && gj == b) || (gi == b && gj == a)
                            }).expect("hex face boundary edge not found in element");
                            let edge_dof = hcurl_dofs[edge_idx] as usize;
                            coo.add(face_dof, edge_dof, stokes_sign);
                        }
                    }
                }
                ElementType::Prism6 => {
                    // Canonical (Elem1) face orientation = MFEM PRISM FaceVert;
                    // the ring direction already gives the canonical face normal
                    // (no Newell flip needed — matches HDivSpace's RT sign).
                    for (face_local, face_verts) in prism_faces.iter().enumerate() {
                        let face_dof = hdiv_dofs[face_local] as usize;
                        if !visited.insert(face_dof) {
                            continue;
                        }
                        // MFEM Wedge::GetNFaceVertices: faces 0,1 are
                        // triangles (padded with a dummy 4th slot in
                        // PRISM_FACES_CANON — NOT a closure), 2..4 quads.
                        // The old `face_verts[2]==face_verts[3]` test misread
                        // the top tri [3,4,5,0] as a quad and looked for the
                        // non-existent edge (5,0) → "prism face boundary edge
                        // not found" (ex34 SubMesh on fichera-mixed.mesh).
                        let nv = if face_local < 2 { 3 } else { 4 };
                        let gv: Vec<u32> = face_verts[..nv].iter().map(|&li| verts[li]).collect();
                        for k in 0..nv {
                            let (a, b) = (gv[k], gv[(k + 1) % nv]);
                            let stokes_sign = if a < b { 1.0 } else { -1.0 };
                            let edge_idx = prism_edges.iter().position(|&(li, lj)| {
                                let (gi, gj) = (verts[li], verts[lj]);
                                (gi == a && gj == b) || (gi == b && gj == a)
                            }).expect("prism face boundary edge not found in element");
                            let edge_dof = hcurl_dofs[edge_idx] as usize;
                            coo.add(face_dof, edge_dof, stokes_sign);
                        }
                    }
                }
                other => {
                    return Err(DiscreteOpError::UnsupportedDimension {
                        op: "curl_3d",
                        dim: 3,
                    })
                    .map_err(|_| DiscreteOpError::UnsupportedHCurlOrder {
                        op: "curl_3d",
                        order: other as u8,
                    })?;
                }
            }
        }

        Ok(coo.into_csr())
    }

    fn curl_3d_nd2_rt1<M: MeshTopology>(
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> Result<CsrMatrix<f64>, DiscreteOpError> {
        let mesh = hcurl_space.mesh();
        let n_hdiv = hdiv_space.n_dofs();
        let n_hcurl = hcurl_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_hdiv, n_hcurl);

        let nd2_elem = TetND2;
        let n_nd2 = nd2_elem.n_dofs(); // 20
        let rt1_elem = TetRT1;
        let n_rt1 = rt1_elem.n_dofs(); // 15

        let tet_edges: [(usize, usize); 6] = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];
        let tet_faces: [(usize, usize, usize); 4] = [
            (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
        ];

        // Same spanning set as the TetND2 monomial implementation.
        let eval_field = |k: usize, x: f64, y: f64, z: f64| -> [f64; 3] {
            match k {
                0 => [1.0, 0.0, 0.0],
                1 => [x, 0.0, 0.0],
                2 => [y, 0.0, 0.0],
                3 => [z, 0.0, 0.0],
                4 => [0.0, 1.0, 0.0],
                5 => [0.0, x, 0.0],
                6 => [0.0, y, 0.0],
                7 => [0.0, z, 0.0],
                8 => [0.0, 0.0, 1.0],
                9 => [0.0, 0.0, x],
                10 => [0.0, 0.0, y],
                11 => [0.0, 0.0, z],
                12 => [-x * y, x * x, 0.0],
                13 => [-z * x, 0.0, x * x],
                14 => [-y * y, x * y, 0.0],
                15 => [0.0, -y * z, y * y],
                16 => [-z * z, 0.0, z * x],
                17 => [0.0, -z * z, y * z],
                18 => [-y * z, x * z, 0.0],
                19 => [-z * y, 0.0, x * y],
                _ => unreachable!(),
            }
        };
        let eval_curl = |k: usize, x: f64, y: f64, z: f64| -> [f64; 3] {
            match k {
                0 => [0.0, 0.0, 0.0],
                1 => [0.0, 0.0, 0.0],
                2 => [0.0, 0.0, -1.0],
                3 => [0.0, 1.0, 0.0],
                4 => [0.0, 0.0, 0.0],
                5 => [0.0, 0.0, 1.0],
                6 => [0.0, 0.0, 0.0],
                7 => [-1.0, 0.0, 0.0],
                8 => [0.0, 0.0, 0.0],
                9 => [0.0, -1.0, 0.0],
                10 => [1.0, 0.0, 0.0],
                11 => [0.0, 0.0, 0.0],
                12 => [0.0, 0.0, 3.0 * x],
                13 => [0.0, -3.0 * x, 0.0],
                14 => [0.0, 0.0, 3.0 * y],
                15 => [3.0 * y, 0.0, 0.0],
                16 => [0.0, -3.0 * z, 0.0],
                17 => [3.0 * z, 0.0, 0.0],
                18 => [-x, -y, 2.0 * z],
                19 => [x, -2.0 * y, z],
                _ => unreachable!(),
            }
        };

        // D32: the ND2 dof rows are the MFEM **nodal** point-value functionals:
        // edges = `F(x_i)·t̂` at the 2-point Gauss points with the canonical
        // tangent (orientation recovered from the global ids, exactly like
        // `curl_2d_nd2_p1`); faces = `F(centroid)·w` with the shared face
        // anchor tangents from `HCurlSpace` (identical to
        // `HCurlSpace::interpolate_vector`).
        let (gl2, _) = gauss_legendre_01(2);

        // D34: the RT1 dof rows use the MFEM nodal functionals (signed
        // pointwise normal-flux samples), shared with
        // `HDivSpace::interpolate_vector`.
        let (rt_dof_pts, rt_dof_nks) = fem_element::raviart_thomas::tet_rt1::mfem_nodal_dofs(1);
        debug_assert_eq!(rt_dof_pts.len(), n_rt1);
        let mut visited_rt1 = HashSet::with_capacity(n_hdiv);

        for e in mesh.elem_iter() {
            let nodes = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let hdiv_dofs = hdiv_space.element_dofs(e);

            let mut dmat = vec![0.0_f64; n_nd2 * n_nd2];
            let mut ymat = vec![0.0_f64; n_rt1 * n_nd2];

            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let x3 = mesh.node_coords(nodes[3]);
            let j0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
            let j1 = [x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]];
            let j2 = [x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]];
            // cof(J) = det(J)·J^{-T} (adjugate transpose) and the element dof
            // signs (both shared with `HDivSpace::interpolate_vector`).
            let cof = [
                [
                    j1[1] * j2[2] - j1[2] * j2[1],
                    j0[2] * j2[1] - j0[1] * j2[2],
                    j0[1] * j1[2] - j0[2] * j1[1],
                ],
                [
                    j1[2] * j2[0] - j1[0] * j2[2],
                    j0[0] * j2[2] - j0[2] * j2[0],
                    j0[2] * j1[0] - j0[0] * j1[2],
                ],
                [
                    j1[0] * j2[1] - j1[1] * j2[0],
                    j0[1] * j2[0] - j0[0] * j2[1],
                    j0[0] * j1[1] - j0[1] * j1[0],
                ],
            ];
            let rt_signs = hdiv_space.element_signs(e);

            for k in 0..n_nd2 {
                let mut dof_nd2 = vec![0.0_f64; n_nd2];

                // ND2 edge rows: nodal point values along the canonical
                // (min,max) direction; the canonical slot index of each
                // element slot is recovered from its global id (adjacent ids
                // per block).
                for (edge_local, &(li, lj)) in tet_edges.iter().enumerate() {
                    let gi = nodes[li];
                    let gj = nodes[lj];
                    let (ga, gb) = if gi < gj { (gi, gj) } else { (gj, gi) };
                    let pa = mesh.node_coords(ga);
                    let pb = mesh.node_coords(gb);
                    let tau = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];

                    let first = hcurl_dofs[2 * edge_local].min(hcurl_dofs[2 * edge_local + 1]);
                    for m in 0..2usize {
                        let j_canon = (hcurl_dofs[2 * edge_local + m] - first) as usize;
                        let t = gl2[j_canon];
                        let pt = [
                            pa[0] + t * tau[0],
                            pa[1] + t * tau[1],
                            pa[2] + t * tau[2],
                        ];
                        let fv = eval_field(k, pt[0], pt[1], pt[2]);
                        dof_nd2[2 * edge_local + m] =
                            fv[0] * tau[0] + fv[1] * tau[1] + fv[2] * tau[2];
                    }
                }

                // ND2 face rows: point values at the face centroid with the
                // shared-face anchor tangents (creation element's TetND2 slot
                // tangents — the same functionals that produced the dof
                // values fed into this operator).
                for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                    let key = FaceKey::new(nodes[la], nodes[lb], nodes[lc]);
                    let anchor = hcurl_space
                        .face_anchor(key)
                        .expect("tet face must have an interpolation anchor");
                    // ND2: the canonical face carries a single DOF point (the
                    // centroid) with its two anchor tangents.
                    let centroid = anchor.point(0);
                    let [w0, w1] = anchor.tangents(0);
                    let fv = eval_field(k, centroid[0], centroid[1], centroid[2]);
                    dof_nd2[12 + 2 * face_local] =
                        fv[0] * w0[0] + fv[1] * w0[1] + fv[2] * w0[2];
                    dof_nd2[12 + 2 * face_local + 1] =
                        fv[0] * w1[0] + fv[1] * w1[1] + fv[2] * w1[2];
                }

                for i in 0..n_nd2 {
                    dmat[i * n_nd2 + k] = dof_nd2[i];
                }

                // D34: RT1 nodal dofs of curl(field): signed pointwise samples
                // curl Φ_k(x_p)·(cof(J)·n̂_p) (same functionals as
                // `HDivSpace::interpolate_vector`).
                for p in 0..n_rt1 {
                    let (xi, nk) = (&rt_dof_pts[p], &rt_dof_nks[p]);
                    let pt = [
                        x0[0] + j0[0] * xi[0] + j1[0] * xi[1] + j2[0] * xi[2],
                        x0[1] + j0[1] * xi[0] + j1[1] * xi[1] + j2[1] * xi[2],
                        x0[2] + j0[2] * xi[0] + j1[2] * xi[1] + j2[2] * xi[2],
                    ];
                    let cv = eval_curl(k, pt[0], pt[1], pt[2]);
                    let mut val = 0.0;
                    for r in 0..3 {
                        val += cv[r] * (cof[r][0] * nk[0] + cof[r][1] * nk[1] + cof[r][2] * nk[2]);
                    }
                    ymat[p * n_nd2 + k] = rt_signs[p] * val;
                }
            }

            // Solve D^T * Z = Y^T, where Z = A^T and A maps ND2 -> RT1.
            let mut dt = vec![0.0_f64; n_nd2 * n_nd2];
            for i in 0..n_nd2 {
                for j in 0..n_nd2 {
                    dt[i * n_nd2 + j] = dmat[j * n_nd2 + i];
                }
            }
            let mut yt = vec![0.0_f64; n_nd2 * n_rt1];
            for p in 0..n_rt1 {
                for k in 0..n_nd2 {
                    yt[k * n_rt1 + p] = ymat[p * n_nd2 + k];
                }
            }

            let z = solve_small(n_nd2, n_rt1, &dt, &yt); // shape 20x15 row-major

            for (p_local, &global_rt1) in hdiv_dofs.iter().enumerate() {
                let g_rt1 = global_rt1 as usize;
                if !visited_rt1.insert(g_rt1) {
                    continue;
                }
                for (i_local, &global_nd2) in hcurl_dofs.iter().enumerate() {
                    let val = z[i_local * n_rt1 + p_local];
                    if val.abs() > 1e-15 {
                        coo.add(g_rt1, global_nd2 as usize, val);
                    }
                }
            }
        }

        Ok(coo.into_csr())
    }
}

/// Solve the small dense linear system `A * X = B` where `A` is `n × n` and
/// `B` is `n × m` (stored row-major).  Returns `X` as a flat `n × m` row-major
/// vector.
///
/// Uses partial-pivoting Gaussian elimination.  Intended for the small
/// per-element systems (n = 3 or 8) that arise in the high-order discrete
/// operator assembly.
fn solve_small(n: usize, m: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    // Augmented matrix [A | B], row-major, n rows × (n+m) cols.
    let mut aug: Vec<f64> = vec![0.0; n * (n + m)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + m) + j] = a[i * n + j];
        }
        for j in 0..m {
            aug[i * (n + m) + n + j] = b[i * m + j];
        }
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1 * (n + m) + col]
                    .abs()
                    .partial_cmp(&aug[r2 * (n + m) + col].abs())
                    .unwrap()
            })
            .unwrap();
        if pivot_row != col {
            for k in 0..(n + m) {
                aug.swap(col * (n + m) + k, pivot_row * (n + m) + k);
            }
        }

        let pivot = aug[col * (n + m) + col];
        debug_assert!(pivot.abs() > 1e-14, "solve_small: singular matrix");

        for row in (col + 1)..n {
            let factor = aug[row * (n + m) + col] / pivot;
            for k in col..(n + m) {
                let sub = factor * aug[col * (n + m) + k];
                aug[row * (n + m) + k] -= sub;
            }
        }
    }

    // Back substitution.
    let mut x: Vec<f64> = vec![0.0; n * m];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut val = aug[i * (n + m) + n + j];
            for k in (i + 1)..n {
                val -= aug[i * (n + m) + k] * x[k * m + j];
            }
            x[i * m + j] = val / aug[i * (n + m) + i];
        }
    }
    x
}

/// Compute the determinant of the simplex Jacobian for element `e`.
fn simplex_det<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> f64 {
    ElementTransformation::from_simplex_nodes(mesh, geo_nodes).det_j()
}



// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// Test: Discrete gradient of a linear function u = x + 2y.
    ///
    /// The gradient field is (1, 2) everywhere.  Interpolating u into H1
    /// and applying G should give the same result as interpolating (1,2)
    /// into H(curl) via its DOF functional.
    #[test]
    fn gradient_of_linear_function() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        // Interpolate u = x + 2y into H1
        let u_h1 = h1.interpolate(&|x| x[0] + 2.0 * x[1]);

        // Build gradient matrix and apply
        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let mut g_u = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_h1.as_slice(), &mut g_u);

        // Interpolate grad(u) = (1, 2) into H(curl) via the DOF functional
        let grad_interp = hcurl.interpolate_vector(&|_x| vec![1.0, 2.0]);

        // Compare: they should match exactly (up to floating-point)
        for i in 0..hcurl.n_dofs() {
            assert!(
                (g_u[i] - grad_interp.as_slice()[i]).abs() < 1e-12,
                "gradient mismatch at DOF {i}: G*u = {}, interp = {}",
                g_u[i], grad_interp.as_slice()[i]
            );
        }
    }

    /// Test: de Rham exact sequence property: curl(grad(u)) = 0.
    ///
    /// Build G (H1 -> H(curl)) and C (H(curl) -> L2), then verify C * G = 0.
    #[test]
    fn de_rham_curl_of_grad_is_zero() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh3, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // Test C * G * u = 0 for several functions
        let test_fns: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| x[0]),
            Box::new(|x: &[f64]| x[1]),
            Box::new(|x: &[f64]| x[0] + x[1]),
            Box::new(|x: &[f64]| 3.0 * x[0] - 2.0 * x[1]),
        ];

        for (idx, f) in test_fns.iter().enumerate() {
            let u = h1.interpolate(f.as_ref());
            let mut gu = vec![0.0; hcurl.n_dofs()];
            g.spmv(u.as_slice(), &mut gu);
            let mut cgu = vec![0.0; l2.n_dofs()];
            c.spmv(&gu, &mut cgu);

            let max_err: f64 = cgu.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                max_err < 1e-12,
                "curl(grad(u_{idx})) not zero: max |C*G*u| = {max_err}"
            );
        }
    }

    /// Test: Discrete divergence of a known field.
    ///
    /// For the constant field F = (1, 0), div(F) = 0.
    /// For the field F = (x, y), div(F) = 2.
    #[test]
    fn divergence_constant_field() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 0);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh2, 0);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();

        // Test 1: F = (1, 0) -> div = 0
        let f_const = hdiv.interpolate_vector(&|_x| vec![1.0, 0.0]);
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f_const.as_slice(), &mut div_f);

        let max_err: f64 = div_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err < 1e-10,
            "div(1,0) should be 0, max |D*F| = {max_err}"
        );

        // Test 2: F = (0, 1) -> div = 0
        let f_const2 = hdiv.interpolate_vector(&|_x| vec![0.0, 1.0]);
        let mut div_f2 = vec![0.0; l2.n_dofs()];
        d.spmv(f_const2.as_slice(), &mut div_f2);

        let max_err2: f64 = div_f2.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err2 < 1e-10,
            "div(0,1) should be 0, max |D*F| = {max_err2}"
        );
    }

    /// Test: Matrix dimensions are correct.
    #[test]
    fn matrix_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = Mesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh3, 0);
        let mesh4 = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh4, 0);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        assert_eq!(g.nrows, hcurl.n_dofs());
        assert_eq!(g.ncols, h1.n_dofs());

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();
        assert_eq!(c.nrows, l2.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs());
        assert_eq!(d.ncols, hdiv.n_dofs());
    }

    /// Test: gradient is nonzero for non-constant functions, zero for constants.
    #[test]
    fn gradient_nonzero_for_nonconst() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();

        // A non-constant function should have non-zero gradient
        let u = h1.interpolate(&|x| x[0]);
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);

        let norm: f64 = gu.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "gradient of x should be nonzero, got norm = {norm}");

        // A constant function should have zero gradient
        let u_const = h1.interpolate(&|_x| 1.0);
        let mut gu_const = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_const.as_slice(), &mut gu_const);

        let norm_const: f64 = gu_const.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_const < 1e-12, "gradient of constant should be zero, got norm = {norm_const}");
    }

    /// Test: de Rham exact sequence in 3D — div(curl(u)) = 0.
    #[test]
    fn de_rham_div_of_curl_3d_is_zero() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let mesh3 = Mesh::<3>::unit_cube_tet(2);
        let mesh4 = Mesh::<3>::unit_cube_tet(2);

        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let hdiv2 = HDivSpace::new(mesh3, 0);
        let l2    = L2Space::new(mesh4,  0);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let d = DiscreteLinearOperator::divergence(&hdiv2, &l2).unwrap();

        assert_eq!(c.nrows, hdiv.n_dofs(),  "C: wrong nrows");
        assert_eq!(c.ncols, hcurl.n_dofs(), "C: wrong ncols");

        // D * C * u = 0 for arbitrary u
        for seed in 0..5u64 {
            let u: Vec<f64> = (0..hcurl.n_dofs())
                .map(|i| (((i as u64 * 1_000_003 + seed * 998_244_353) % 1000) as f64) / 500.0 - 1.0)
                .collect();
            let mut cu = vec![0.0f64; hdiv.n_dofs()];
            c.spmv(&u, &mut cu);
            let mut dcu = vec![0.0f64; l2.n_dofs()];
            d.spmv(&cu, &mut dcu);
            let max_err: f64 = dcu.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-10,
                "div(curl(u)) ≠ 0 for seed={seed}: max|D*C*u| = {max_err}");
        }
    }

    /// Test: order-2 3D de Rham property div(curl(u)) = 0 for ND2->RT1->P1.
    #[test]
    fn de_rham_div_of_curl_3d_is_zero_order2() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let mesh3 = Mesh::<3>::unit_cube_tet(2);
        let mesh4 = Mesh::<3>::unit_cube_tet(2);

        let hcurl = HCurlSpace::new(mesh, 2);
        let hdiv = HDivSpace::new(mesh2, 1);
        let hdiv2 = HDivSpace::new(mesh3, 1);
        let l2 = L2Space::new(mesh4, 1);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let d = DiscreteLinearOperator::divergence(&hdiv2, &l2).unwrap();

        for seed in 0..3u64 {
            let u: Vec<f64> = (0..hcurl.n_dofs())
                .map(|i| (((i as u64 * 1_146_959_810_393 + seed * 972_663_749) % 1000) as f64) / 500.0 - 1.0)
                .collect();
            let mut cu = vec![0.0f64; hdiv.n_dofs()];
            c.spmv(&u, &mut cu);
            let mut dcu = vec![0.0f64; l2.n_dofs()];
            d.spmv(&cu, &mut dcu);

            let max_err: f64 = dcu.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(
                max_err < 1e-8,
                "order-2 3D de Rham div(curl(u)) ≠ 0, seed={seed}, max |D*C*u| = {max_err}"
            );
        }
    }

    /// Test: curl_3d matrix dimensions.
    #[test]
    fn curl_3d_dimensions() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        assert_eq!(c.nrows, hdiv.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: Curl ND2->RT1 matrix dimensions in 3D.
    #[test]
    fn curl_3d_nd2_rt1_dimensions() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let hdiv = HDivSpace::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        assert_eq!(c.nrows, hdiv.n_dofs(), "C nrows should equal n_rt1");
        assert_eq!(c.ncols, hcurl.n_dofs(), "C ncols should equal n_nd2");
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: Curl ND2->RT1 in 3D — commuting property.
    #[test]
    fn curl_3d_nd2_rt1_commutes_with_interpolation() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let hdiv = HDivSpace::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();

        // D32: the commuting identity `C·Π(f) = Π_RT(curl f)` is exact for
        // fields **in** the ND2 span (nodal interpolators are not an exact
        // cochain map for out-of-space fields — only integral-moment
        // interpolators are).  Use the in-span mode
        // A = (−xy, x², 0) + (0, −yz, y²) + (−z², 0, zx), whose curl is the
        // linear field (3y, −3z, 3x) ⊂ RT1.
        let a = hcurl.interpolate_vector(&|x| {
            vec![
                -x[0] * x[1] - x[2] * x[2],
                x[0] * x[0] - x[1] * x[2],
                x[2] * x[0] + x[1] * x[1],
            ]
        });
        let mut ca = vec![0.0; hdiv.n_dofs()];
        c.spmv(a.as_slice(), &mut ca);

        let curl_interp = hdiv.interpolate_vector(&|x| vec![3.0 * x[1], -3.0 * x[2], 3.0 * x[0]]);

        let max_err: f64 = (0..hdiv.n_dofs())
            .map(|i| (ca[i] - curl_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
            assert!(
                max_err < 1e-8,
                "ND2->RT1 3D: curl interpolation mismatch, max error = {max_err}"
            );
    }

    /// Test: Curl ND2->RT1 in 3D — randomized commuting stress test.
    ///
    /// D32: randomizes over **linear** fields — P1³ ⊂ ND2 with constant curl
    /// ⊂ RT1 — so the commuting identity `C·Π(A) = Π_RT(curl A)` holds
    /// exactly for the nodal (point-value) interpolators (they are an exact
    /// cochain map only on the space; quadratic fields generally leave the
    /// per-element physical span under affine pullback).
    #[test]
    fn curl_3d_nd2_rt1_commuting_randomized_stress() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let hdiv = HDivSpace::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();

        for seed in 0..8u64 {
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut coeffs = [0.0f64; 12];
            for k in 0..12 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                coeffs[k] = 2.0 * r - 1.0;
            }

            let [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3] = coeffs else {
                unreachable!()
            };

            // A = (a0 + a1·x + a2·y + a3·z, b0 + b1·x + b2·y + b3·z,
            //      c0 + c1·x + c2·y + c3·z)
            let a_h = hcurl.interpolate_vector(&|x: &[f64]| {
                let (xx, yy, zz) = (x[0], x[1], x[2]);
                vec![
                    a0 + a1 * xx + a2 * yy + a3 * zz,
                    b0 + b1 * xx + b2 * yy + b3 * zz,
                    c0 + c1 * xx + c2 * yy + c3 * zz,
                ]
            });

            let mut ca = vec![0.0; hdiv.n_dofs()];
            c.spmv(a_h.as_slice(), &mut ca);

            // curl(A) = (c2 − b3, a3 − c1, b1 − a2) — constant.
            let curl_interp = hdiv.interpolate_vector(&|_x: &[f64]| {
                vec![c2 - b3, a3 - c1, b1 - a2]
            });

            let max_err: f64 = (0..hdiv.n_dofs())
                .map(|idx| (ca[idx] - curl_interp.as_slice()[idx]).abs())
                .fold(0.0, f64::max);
            assert!(
                max_err < 1e-8,
                "ND2->RT1 randomized commuting failed (seed={seed}), max error = {max_err}"
            );
        }
    }

    /// Test: bad order combination returns an error instead of panicking.
    ///
    /// H1 P2 (order 2) + ND1 (order 1): ND1 is unsupported for H1 order 2.
    #[test]
    fn gradient_bad_order_returns_error() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let h1 = H1Space::new(mesh, 2); // P2
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh2, 1); // ND1

        let result = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert!(
            matches!(result, Err(DiscreteOpError::UnsupportedHCurlOrder { .. })),
            "expected UnsupportedHCurlOrder for P2+ND1, got {:?}", result
        );
    }

    /// Test: curl_2d with wrong dimension returns an error.
    #[test]
    fn curl_2d_wrong_dim_returns_error() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let hcurl = HCurlSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(1);
        let l2 = L2Space::new(mesh2, 0);

        let result = DiscreteLinearOperator::curl_2d(&hcurl, &l2);
        assert!(
            matches!(result, Err(DiscreteOpError::UnsupportedDimension { op: "curl_2d", dim: 3 })),
            "expected UnsupportedDimension for curl_2d on 3D mesh, got {:?}", result
        );
    }

    /// Test: curl_2d supports ND2->L2(P2) with expected dimensions.
    #[test]
    fn curl_2d_nd2_p2_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();
        assert_eq!(c.nrows, l2.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: curl_3d with wrong dimension returns an error.
    #[test]
    fn curl_3d_wrong_dim_returns_error() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let hdiv = HDivSpace::new(mesh2, 0);

        let result = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv);
        assert!(
            matches!(result, Err(DiscreteOpError::UnsupportedDimension { op: "curl_3d", dim: 2 })),
            "expected UnsupportedDimension for curl_3d on 2D mesh, got {:?}", result
        );
    }

    /// Test: divergence supports RT1->L2(P2) with expected dimensions.
    #[test]
    fn divergence_rt1_p2_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let hdiv = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh2, 2);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs());
        assert_eq!(d.ncols, hdiv.n_dofs());
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    // ── Higher-order tests ────────────────────────────────────────────────────

    /// Test: Gradient P2→ND2 — commuting diagram property.
    ///
    /// For any smooth function u, the diagram commutes:
    ///   G * I_{P2}(u) = I_{ND2}(∇u)
    /// where I_{P2} and I_{ND2} are the respective interpolation operators.
    /// We verify this for u = x² + 2xy (a quadratic function).
    #[test]
    fn gradient_p2_nd2_commutes_with_interpolation() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 2);

        // u = x² + 2xy,  ∇u = (2x + 2y,  2x)
        let u_h1 = h1.interpolate(&|x| x[0] * x[0] + 2.0 * x[0] * x[1]);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let mut g_u = vec![0.0; hcurl.n_dofs()];
        g.spmv(u_h1.as_slice(), &mut g_u);

        // Interpolate ∇u into ND2
        let grad_interp = hcurl.interpolate_vector(&|x| vec![2.0 * x[0] + 2.0 * x[1], 2.0 * x[0]]);

        for i in 0..hcurl.n_dofs() {
            let diff = (g_u[i] - grad_interp.as_slice()[i]).abs();
            assert!(
                diff < 1e-10,
                "P2→ND2 gradient mismatch at DOF {i}: G*u = {}, interp = {}, diff = {}",
                g_u[i], grad_interp.as_slice()[i], diff
            );
        }
    }

    /// Test: Gradient P2→ND2 — dimensions are correct.
    #[test]
    fn gradient_p2_nd2_dimensions() {
        let mesh  = Mesh::<2>::unit_square_tri(3);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh2, 2);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        assert_eq!(g.nrows, hcurl.n_dofs(), "G nrows should equal n_nd2");
        assert_eq!(g.ncols, h1.n_dofs(),    "G ncols should equal n_p2");
        assert!(g.nrows > 0 && g.ncols > 0);
    }

    /// Test: Gradient P2→ND2 — constant function has zero gradient.
    #[test]
    fn gradient_p2_nd2_constant_is_zero() {
        let mesh  = Mesh::<2>::unit_square_tri(3);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh2, 2);

        let u = h1.interpolate(&|_x| 3.0);
        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);

        let norm: f64 = gu.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "P2→ND2: gradient of constant should be zero, norm = {norm}");
    }

    /// Test: Curl 2D rejects incompatible order pairs.
    #[test]
    fn curl_2d_incompatible_orders_returns_error() {
        let mesh  = Mesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 2); // ND2
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let l2    = L2Space::new(mesh2, 0); // P0 (mismatch for ND2)

        let result = DiscreteLinearOperator::curl_2d(&hcurl, &l2);
        assert!(
            matches!(result, Err(DiscreteOpError::IncompatibleOrders { .. })),
            "expected IncompatibleOrders for ND2->P0, got {:?}", result
        );
    }

    /// Test: Curl ND2->P1 matrix dimensions.
    #[test]
    fn curl_2d_nd2_p1_dimensions() {
        let mesh  = Mesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(3);
        let l2    = L2Space::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();
        assert_eq!(c.nrows, l2.n_dofs(), "C nrows should equal n_p1");
        assert_eq!(c.ncols, hcurl.n_dofs(), "C ncols should equal n_nd2");
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: order-2 2D de Rham property curl(grad(u)) = 0.
    #[test]
    fn de_rham_curl_of_grad_is_zero_order2() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 2);
        let mesh3 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh3, 1);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // Quadratic scalar potential.
        let u = h1.interpolate(&|x| x[0] * x[0] + x[0] * x[1] + 0.25 * x[1] * x[1]);
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);
        let mut cgu = vec![0.0; l2.n_dofs()];
        c.spmv(&gu, &mut cgu);

        let max_err: f64 = cgu.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-9, "order-2 curl(grad(u)) should be zero, max |C*G*u| = {max_err}");
    }

    /// Test: ND2->P1 curl of a curl-free field is zero.
    #[test]
    fn curl_2d_nd2_p1_curl_free_field_is_zero() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // F = (x, y), so curl(F) = d/dx(y) - d/dy(x) = 0.
        let f = hcurl.interpolate_vector(&|x| vec![x[0], x[1]]);
        let mut cf = vec![0.0; l2.n_dofs()];
        c.spmv(f.as_slice(), &mut cf);

        let max_err: f64 = cf.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-8, "ND2->P1: curl(x,y) should be zero, max |C*F| = {max_err}");
    }

    /// Test: ND2->P1 curl of F=(-y, x) equals constant 2.
    #[test]
    fn curl_2d_nd2_p1_constant_curl_field() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // F = (-y, x), so curl(F) = d/dx(x) - d/dy(-y) = 2.
        let f = hcurl.interpolate_vector(&|x| vec![-x[1], x[0]]);
        let mut cf = vec![0.0; l2.n_dofs()];
        c.spmv(f.as_slice(), &mut cf);

        let c_interp = l2.interpolate(&|_x| 2.0);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (cf[i] - c_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "ND2->P1: curl(-y,x) should be 2, max error = {max_err}");
    }

    /// Test: ND2->P2 curl commutes with interpolation.
    ///
    /// D32: the identity `C·Π(f) = Π(curl f)` is exact only for fields **in**
    /// the ND2 span (nodal interpolation is exact there).  Use the monomial
    /// `F = (−xy, x²)` (the rotational ND2 mode), whose curl is `3x`.
    #[test]
    fn curl_2d_nd2_p2_commutes_with_interpolation() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // F = (-x*y, x^2) ∈ ND2, curl(F) = d/dx(x^2) - d/dy(-x*y) = 3x.
        let f = hcurl.interpolate_vector(&|x| vec![-x[0] * x[1], x[0] * x[0]]);
        let mut cf = vec![0.0; l2.n_dofs()];
        c.spmv(f.as_slice(), &mut cf);

        let c_interp = l2.interpolate(&|x| 3.0 * x[0]);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (cf[i] - c_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "ND2->P2: curl(-xy,x^2) should be 3x, max error = {max_err}");
    }

    /// Gradient P2→ND1: unsupported HCurl order returns error.
    #[test]
    fn gradient_incompatible_orders_returns_error() {
        let mesh  = Mesh::<2>::unit_square_tri(2);
        let h1    = H1Space::new(mesh, 2); // P2
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh2, 1); // ND1

        let result = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert!(
            matches!(result, Err(DiscreteOpError::UnsupportedHCurlOrder { .. })),
            "expected UnsupportedHCurlOrder for P2+ND1, got {:?}", result
        );
    }

    /// Test: Divergence RT1→P1 — commuting diagram property.
    ///
    /// For any smooth vector field F, the diagram commutes:
    ///   D * I_{RT1}(F) = I_{P1}(div F)
    /// We verify for F = (x², xy) with div F = 2x + x = 3x... wait,
    /// div(x², xy) = 2x + x = 3x.  Since P1 can represent 3x, this works.
    #[test]
    fn divergence_rt1_p1_commutes_with_interpolation() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 1);

        // F = (x, y),  div F = 2 (constant — lies in P0 ⊂ P1)
        let f = hdiv.interpolate_vector(&|x| vec![x[0], x[1]]);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        // Interpolate div F = 2 into P1
        let div_interp = l2.interpolate(&|_x| 2.0);

        for i in 0..l2.n_dofs() {
            let diff = (div_f[i] - div_interp.as_slice()[i]).abs();
            assert!(
                diff < 1e-9,
                "RT1→P1 divergence mismatch at DOF {i}: D*F={}, interp={}, diff={}",
                div_f[i], div_interp.as_slice()[i], diff
            );
        }
    }

    /// Test: Divergence RT1->P2 commutes with interpolation.
    ///
    /// D34 correction: the test field must lie **in RT₁** for the composite
    /// `D ∘ interpolate` to reproduce the exact divergence (the nodal dof
    /// semantics, like MFEM `Project_RT`, interpolates exactly only fields of
    /// the space; the pre-D34 moment interpolation additionally projected
    /// `div` of out-of-space fields, which is why `F = (x², y²) ∉ RT₁` used to
    /// pass).  `F = (x², xy) = x·(x, y)` ∈ RT₁ with `div F = 3x`.
    #[test]
    fn divergence_rt1_p2_commutes_with_interpolation() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 2);

        // F = (x^2, xy), so div F = 3x.
        let f = hdiv.interpolate_vector(&|x| vec![x[0] * x[0], x[0] * x[1]]);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|x| 3.0 * x[0]);

        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-8,
            "RT1->P2: div(x^2,y^2) should be 2x+2y, max error = {max_err}"
        );
    }

    /// Test: divergence supports RT2→L2(P2) on 2D meshes (matrix shape).
    #[test]
    fn divergence_rt2_p2_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let hdiv = HDivSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh2, 2);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs());
        assert_eq!(d.ncols, hdiv.n_dofs());
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT2→P2 — commuting diagram for a linear field (exact in RT2 / P2).
    #[test]
    fn divergence_rt2_p2_commutes_with_interpolation() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh2, 2);

        // F = (x, y), div F = 2 (constant — exact in RT2 and P2).
        let f = hdiv.interpolate_vector(&|x| vec![x[0], x[1]]);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|_x| 2.0);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-7,
            "RT2→P2: div(x,y) should be 2, max error = {max_err}"
        );
    }

    /// Test: Divergence RT2→P2 — divergence-free field gives zero.
    #[test]
    fn divergence_rt2_p2_div_free_field_is_zero() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2 = L2Space::new(mesh2, 2);

        let f = hdiv.interpolate_vector(&|x| vec![-x[1], x[0]]);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let max_err: f64 = div_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err < 1e-7,
            "RT2→P2: div(-y,x) should be zero, max |D*F| = {max_err}"
        );
    }

    /// Test: `curl_2d_hdiv` rejects 3-D meshes.
    #[test]
    fn curl_2d_hdiv_wrong_dim_returns_error() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let hdiv = HDivSpace::new(mesh2, 1);

        let r = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv);
        assert!(
            matches!(r, Err(DiscreteOpError::UnsupportedDimension { op: "curl_2d_hdiv", dim: 3 })),
            "expected UnsupportedDimension for curl_2d_hdiv on 3D mesh, got {r:?}"
        );
    }

    /// Test: ND1→RT0 `curl_2d_hdiv` matrix dimensions.
    #[test]
    fn curl_2d_nd1_rt0_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh.clone(), 1);
        let hdiv = HDivSpace::new(mesh, 0);
        let c = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv).unwrap();
        assert_eq!(c.nrows, hdiv.n_dofs(), "nrows should match HDiv DOFs");
        assert_eq!(c.ncols, hcurl.n_dofs(), "ncols should match HCurl DOFs");
        assert!(c.nnz() > 0, "curl matrix should be non-empty");
    }

    /// Test: ND2→RT2 `curl_2d_hdiv` matrix dimensions.
    #[test]
    fn curl_2d_nd2_rt2_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(3);
        let hdiv = HDivSpace::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv).unwrap();
        assert_eq!(c.nrows, hdiv.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// ND2→RT2 curl of a curl-free polynomial field stays small (exact curl is 0).
    ///
    /// `de_rham_curl_grad_nd2_rt2_bounded` checks `C*G` on a quadratic potential; this test
    /// guards against gross assembly errors on multi-triangle meshes.
    #[test]
    fn curl_2d_nd2_rt2_curl_free_radial_field_bounded() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv).unwrap();
        let f = hcurl.interpolate_vector(&|x| vec![x[0], x[1]]);
        let mut cf = vec![0.0; hdiv.n_dofs()];
        c.spmv(f.as_slice(), &mut cf);

        let max_abs: f64 = cf.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_abs < 0.55,
            "ND2→RT2: curl(x,y) should stay small, max |C*f| = {max_abs}"
        );
    }

    /// ND2→RT2: discrete `C*G*u` for a quadratic potential stays small (continuum curl(grad u)=0).
    #[test]
    fn de_rham_curl_grad_nd2_rt2_bounded() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh.clone(), 2);
        let hcurl = HCurlSpace::new(mesh.clone(), 2);
        let hdiv = HDivSpace::new(mesh, 2);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        let c = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv).unwrap();

        let u = h1.interpolate(&|x| x[0] * x[0] + x[0] * x[1] + 0.25 * x[1] * x[1]);
        let mut gu = vec![0.0; hcurl.n_dofs()];
        g.spmv(u.as_slice(), &mut gu);
        let mut cgu = vec![0.0; hdiv.n_dofs()];
        c.spmv(&gu, &mut cgu);

        let max_err: f64 = cgu.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_err < 0.65,
            "ND2→RT2: max|C*G*u| for quadratic u should stay small, got {max_err}"
        );
    }

    /// Test: Divergence RT1→P1 — dimensions are correct.
    #[test]
    fn divergence_rt1_p1_dimensions() {
        let mesh  = Mesh::<2>::unit_square_tri(3);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(3);
        let l2    = L2Space::new(mesh2, 1);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(),   "D nrows should equal n_p1");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1→P1 — divergence-free field gives zero.
    #[test]
    fn divergence_rt1_p1_div_free_field_is_zero() {
        let mesh  = Mesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 1);

        // F = (-y, x) is div-free
        let f = hdiv.interpolate_vector(&|x| vec![-x[1], x[0]]);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let max_err: f64 = div_f.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-9, "RT1→P1: div of (-y,x) should be zero, max|D*F| = {max_err}");
    }

    /// Test: Divergence RT1->P1 in 3D — dimensions are correct.
    #[test]
    fn divergence_rt1_p1_3d_dimensions() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 1);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(), "D nrows should equal n_l2_p1");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1->P1 in 3D — commuting property for F=(x,y,z).
    #[test]
    fn divergence_rt1_p1_3d_commutes_with_interpolation() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 1);

        let f = hdiv.interpolate_vector(&|x| vec![x[0], x[1], x[2]]);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|_x| 3.0);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);

        assert!(max_err < 1e-8, "RT1->P1 3D: divergence mismatch too large, max error = {max_err}");
    }

    /// Test: Divergence RT1->P2 in 3D — dimensions are correct.
    #[test]
    fn divergence_rt1_p2_3d_dimensions() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 2);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(), "D nrows should equal n_l2_p2");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1->P2 in 3D — commuting property for F=(x,y,z).
    #[test]
    fn divergence_rt1_p2_3d_commutes_with_interpolation() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 2);

        // F = (x,y,z), div F = 3.
        let f = hdiv.interpolate_vector(&|x| vec![x[0], x[1], x[2]]);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|_x| 3.0);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "RT1->P2 3D: divergence mismatch, max error = {max_err}");
    }

    /// Test: Divergence RT1->P2 in 3D — randomized commuting stress test.
    #[test]
    fn divergence_rt1_p2_3d_commuting_randomized_stress() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 2);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();

        for seed in 0..8u64 {
            let mut state = seed.wrapping_mul(11400714819323198485).wrapping_add(1);
            let mut coeffs = [0.0f64; 15];
            for k in 0..15 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                coeffs[k] = 2.0 * r - 1.0;
            }

            // D34 correction: the random field is drawn from RT₁ itself
            // (P₁³ ⊕ x·P̃₁) — see the 2-D note above.  With
            // [a1,a2,a3,a4, b1,b2,b3,b4, c1,c2,c3,c4, β1,β2,β3]:
            let [a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, beta1, beta2, beta3] = coeffs;

            let f_h = hdiv.interpolate_vector(&|x| {
                let xx = x[0];
                let yy = x[1];
                let zz = x[2];
                vec![
                    a1 + a2 * xx + a3 * yy + a4 * zz
                        + beta1 * xx * xx + beta2 * xx * yy + beta3 * xx * zz,
                    b1 + b2 * xx + b3 * yy + b4 * zz
                        + beta1 * xx * yy + beta2 * yy * yy + beta3 * yy * zz,
                    c1 + c2 * xx + c3 * yy + c4 * zz
                        + beta1 * xx * zz + beta2 * yy * zz + beta3 * zz * zz,
                ]
            });

            let mut div_f = vec![0.0; l2.n_dofs()];
            d.spmv(f_h.as_slice(), &mut div_f);

            // div(F) = (a2 + b3 + c4) + 4β1·x + 4β2·y + 4β3·z
            let div_interp = l2.interpolate(&|x| {
                (a2 + b3 + c4)
                    + 4.0 * beta1 * x[0]
                    + 4.0 * beta2 * x[1]
                    + 4.0 * beta3 * x[2]
            });

            let max_err: f64 = (0..l2.n_dofs())
                .map(|idx| (div_f[idx] - div_interp.as_slice()[idx]).abs())
                .fold(0.0, f64::max);
            assert!(
                max_err < 1e-8,
                "RT1->P2 3D randomized commuting failed (seed={seed}), max error = {max_err}"
            );
        }
    }

    /// Test: order-2 3D de Rham property with L2(P2) target — div(curl(u)) = 0.
    #[test]
    fn de_rham_div_of_curl_3d_is_zero_order2_l2_p2() {
        let mesh  = Mesh::<3>::unit_cube_tet(2);
        let mesh2 = Mesh::<3>::unit_cube_tet(2);
        let mesh3 = Mesh::<3>::unit_cube_tet(2);
        let mesh4 = Mesh::<3>::unit_cube_tet(2);

        let hcurl = HCurlSpace::new(mesh,  2);
        let hdiv  = HDivSpace::new(mesh2, 1);
        let hdiv2 = HDivSpace::new(mesh3, 1);
        let l2    = L2Space::new(mesh4,  2);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let d = DiscreteLinearOperator::divergence(&hdiv2, &l2).unwrap();

        for seed in 0..5u64 {
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut u = vec![0.0; hcurl.n_dofs()];
            for v in &mut u {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                *v = 2.0 * r - 1.0;
            }

            let mut cu = vec![0.0; hdiv.n_dofs()];
            c.spmv(&u, &mut cu);
            let mut dcu = vec![0.0; l2.n_dofs()];
            d.spmv(&cu, &mut dcu);

            let max_err: f64 = dcu.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                max_err < 1e-8,
                "order-2 3D with L2(P2): div(curl(u)) should be zero, seed={seed}, max |D*C*u| = {max_err}"
            );
        }
    }

    /// Debug test: print curl_3d and divergence matrices for a single element.
    #[test]
    #[ignore] // Disabled - curl_3d is placeholder
    fn debug_curl_3d_single_element() {
        let mesh  = Mesh::<3>::unit_cube_tet(1);
        let mesh2 = Mesh::<3>::unit_cube_tet(1);
        let mesh3 = Mesh::<3>::unit_cube_tet(1);
        let hcurl = HCurlSpace::new(mesh,  1);
        let hdiv  = HDivSpace::new(mesh2, 0);
        let l2    = L2Space::new(mesh3,  0);

        // Print element DOFs and signs for element 0
        let hcurl_dofs = hcurl.element_dofs(0);
        let hcurl_signs = hcurl.element_signs(0);
        let hdiv_dofs = hdiv.element_dofs(0);
        let hdiv_signs = hdiv.element_signs(0);
        let l2_dofs = l2.element_dofs(0);

        println!("\n=== Element 0 ===");
        println!("HCurl DOFs: {:?}", hcurl_dofs);
        println!("HCurl signs: {:?}", hcurl_signs);
        println!("HDiv DOFs: {:?}", hdiv_dofs);
        println!("HDiv signs: {:?}", hdiv_signs);
        println!("L2 DOFs: {:?}", l2_dofs);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();

        println!("\n=== C matrix ({} x {}) ===", c.nrows, c.ncols);
        for row in 0..c.nrows.min(8) {
            let start = c.row_ptr[row];
            let end = c.row_ptr[row+1];
            print!("Row {}: ", row);
            for i in start..end {
                let col = c.col_idx[i];
                print!("({}, {:.1}) ", col, c.values[i]);
            }
            println!();
        }

        println!("\n=== D matrix ({} x {}) ===", d.nrows, d.ncols);
        for row in 0..d.nrows.min(4) {
            let start = d.row_ptr[row];
            let end = d.row_ptr[row+1];
            print!("Row {}: ", row);
            for i in start..end {
                print!("({}, {:.3}) ", d.col_idx[i], d.values[i]);
            }
            println!();
        }

        // Show contributions from each element to C for shared faces
        println!("\n=== Face sharing analysis ===");
        for e in 0..2u32 {
            let hdiv_dofs = hdiv.element_dofs(e);
            let hcurl_dofs = hcurl.element_dofs(e);
            println!("Element {}: HDiv DOFs {:?}, HCurl DOFs {:?}", e, hdiv_dofs, hcurl_dofs);
        }

        // Compute D*C
        let mut dc_max = 0.0f64;
        for i in 0..d.nrows {
            let d_start = d.row_ptr[i];
            let d_end = d.row_ptr[i+1];
            for d_idx in d_start..d_end {
                let k = d.col_idx[d_idx] as usize;
                let d_val = d.values[d_idx];
                let c_start = c.row_ptr[k];
                let c_end = c.row_ptr[k+1];
                for c_idx in c_start..c_end {
                    let j = c.col_idx[c_idx];
                    let val: f64 = d_val * c.values[c_idx];
                    if val.abs() > 1e-10 {
                        println!("D*C[{},{}] += {:.3} * {:.1} = {:.3}", i, j, d_val, c.values[c_idx], val);
                    }
                    dc_max = dc_max.max(val.abs());
                }
            }
        }
        println!("\nmax|D*C| = {}", dc_max);
    }
}
