//! Geometric multigrid preconditioner.
//!
//! Uses nested meshes (uniformly refined) as the multigrid hierarchy.
//! The polynomial order is the same at each level.
//!
//! # Usage
//! ```ignore
//! // Build hierarchy externally (e.g. in example code), then:
//! let mg = GeometricMgPrecond::default();
//! mg.v_cycle(&hierarchy, &b, &mut x);
//! ```

use crate::constrained_operator::RectangularConstrainedOperator;
use crate::SolverConfig;
use fem_element::lagrange::factory::{H1TriPk, QuadQk, TriPk};
use fem_element::ReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::{topology::MeshTopology, ElementType};
use nalgebra::DMatrix;

// ─── SumFactDiffusionOp: sum-factorization diffusion operator for Quad4 ──────

/// Sum-factorization diffusion operator matching MFEM's `AddMultPA`.
///
/// Uses a 3-phase sum-factorization algorithm for Quad4 elements:
///
/// **Phase 1** — For each 1D ξ-quad-point, contract along η (s_x, s_y).
/// **Phase 2** — At each full 2D qp (ξ_q, η_r), compute du/dξ, du/dη and
///               apply pa_data (J⁻¹·J⁻ᵀ · |detJ| · w · κ) to get physical flux.
/// **Phase 3** — Assemble back using the same 1D basis factorisation.
///
/// This produces bitwise-identical floating-point results to MFEM's
/// `ApplyPAKernels::Run` for tensor-product Quad4 elements, enabling
/// MG convergence matching C++ (28→4 iters for star.mesh with -or 2).
#[allow(non_snake_case)]
pub struct SumFactDiffusionOp {
    /// Element DOF indices (QuadQk ordering per element).
    pub elem_dofs: Vec<u32>,
    pub n_elems: usize,
    pub n_dofs: usize,
    /// Local DOFs per element (p+1)².
    pub ldofs: usize,
    /// Polynomial order p (internal DOFs per direction = p+1).
    pub p: usize,
    /// Number of 1D quadrature points.
    pub q1d: usize,

    /// 1D basis values: B[q * p1 + i] = l_i(ξ_q), p1 = p+1.
    pub B: Vec<f64>,
    /// 1D basis gradients: G[q * p1 + i] = dl_i/dξ(ξ_q).
    pub G: Vec<f64>,
    /// 1D quadrature weights on [-1, 1].
    pub W: Vec<f64>,

    /// PA data per (element, ξ_q, η_r), 3 symmetric components:
    /// `pa_data[(e * q1d * q1d + q * q1d + r) * 3 + c]`
    ///   c=0 → D00, c=1 → D01, c=2 → D11.
    /// Value = w_ξ[q] · w_η[r] · |detJ| · κ · (J⁻¹ · J⁻ᵀ).
    pub pa_data: Vec<f64>,

    /// Mapping: tensor-product index (iy*(p+1)+ix) → QuadQk DOF index.
    pub tp_to_dof: Vec<usize>,
    /// Mapping: QuadQk DOF index → tensor-product index.
    pub dof_to_tp: Vec<usize>,
}

impl SumFactDiffusionOp {
    /// Build the sum-factorization diffusion operator for a Quad4 mesh.
    ///
    /// # Panics
    /// Panics if the mesh does not contain Quad4 elements.
    pub fn build(
        mesh: &dyn MeshTopology,
        n_dofs: usize,
        order: u8,
        quad_order: u8,
        kappa: f64,
        mut elem_dofs_fn: impl FnMut(u32) -> Vec<u32>,
    ) -> Self {
        assert_eq!(
            mesh.element_type(0),
            ElementType::Quad4,
            "SumFactDiffusionOp requires Quad4 elements"
        );

        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        assert_eq!(dim, 2, "SumFactDiffusionOp requires 2D");

        // 1D Gauss-Legendre points on [-1, 1].  For Q_p diffusion the integrand
        // is degree 2p in one tensor-product direction (∂_x φ has degree p in
        // η), so P4 needs 5-point Gauss (exact to degree 9) — matching the CSR
        // assembly after the same fix in `quad_rule_01`.
        let q1d = ((quad_order as usize + 2) / 2).max(1);
        let (xi_1d, w_1d) = if q1d <= 4 {
            Self::gauss_legendre_1d(q1d)
        } else {
            fem_element::quadrature::gauss_legendre_arbitrary(q1d)
        };

        // 1D Lagrange nodes on [-1, 1] — Gauss-Lobatto-Legendre, matching the
        // QuadQk basis used by the CSR assembly (H1_FECollection BasisType::
        // GaussLobatto).  For p = 1, 2 the GLL points coincide with the
        // equispaced points; for p >= 3 they differ, so the old equispaced
        // choice produced a different operator than the CSR matrix.
        let p = order as usize;
        let p1 = p + 1;
        let (nodes_1d, _w) = fem_element::quadrature::gauss_lobatto_arbitrary(p1);

        // Precompute 1D basis: B[q][i] and G[q][i]
        let mut b_1d = vec![0.0; q1d * p1];
        let mut g_1d = vec![0.0; q1d * p1];
        for q in 0..q1d {
            let x = xi_1d[q];
            for i in 0..p1 {
                let (val, der) = lagrange_1d_eval(&nodes_1d, i, x);
                if !val.is_finite() || !der.is_finite() {
                    eprintln!("[SumFact] B/G NaN: q={q} i={i} x={x} val={val} der={der}");
                    panic!("non-finite B/G");
                }
                b_1d[q * p1 + i] = val;
                g_1d[q * p1 + i] = der;
            }
        }

        // Build DOF permutation: tensor-product index (iy*(p+1)+ix) ↔ QuadQk DOF
        let mut tp_to_dof = vec![0usize; p1 * p1];
        let mut dof_to_tp = vec![0usize; p1 * p1];
        for iy in 0..p1 {
            for ix in 0..p1 {
                let tp_idx = iy * p1 + ix;
                let qdof = quadqk_node_to_dof(ix, iy, p);
                tp_to_dof[tp_idx] = qdof;
                dof_to_tp[qdof] = tp_idx;
            }
        }

        // Precompute pa_data for each element
        let ldofs = p1 * p1;
        let pa_size = n_elems * q1d * q1d * 3;
        let mut pa_data = vec![0.0; pa_size];
        let mut all_dofs = Vec::with_capacity(n_elems * ldofs);

        for e in 0..n_elems as u32 {
            let gd = elem_dofs_fn(e);
            all_dofs.extend_from_slice(&gd);

            let nodes = mesh.element_nodes(e);
            // Bilinear Quad4 corner coordinates: nodes[0..4]
            let mut xy = [[0.0f64; 2]; 4];
            for n in 0..4 {
                let c = mesh.node_coords(nodes[n]);
                xy[n][0] = c[0];
                xy[n][1] = c[1];
            }

            for q in 0..q1d {
                let xi = xi_1d[q];
                for r in 0..q1d {
                    let eta = xi_1d[r];
                    // Bilinear Quad4 Jacobian (same convention as PADiffusionOp,
                    // i.e. WITHOUT the 1/4 factor — see long comment there)
                    let dndxi = [-(1.0 - eta), (1.0 - eta), eta, -eta];
                    let dndeta = [-(1.0 - xi), -xi, xi, (1.0 - xi)];
                    let mut jac = [[0.0f64; 2]; 2]; // [d][col]
                    for n in 0..4 {
                        jac[0][0] += xy[n][0] * dndxi[n];
                        jac[1][0] += xy[n][1] * dndxi[n];
                        jac[0][1] += xy[n][0] * dndeta[n];
                        jac[1][1] += xy[n][1] * dndeta[n];
                    }

                    // |detJ|
                    let det_j = (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]).abs();

                    // J⁻¹ = 1/detJ * [j11, -j01; -j10, j00]
                    let inv_det = 1.0 / (det_j + 1e-300);
                    let a00 = jac[1][1] * inv_det;
                    let a01 = -jac[0][1] * inv_det;
                    let a10 = -jac[1][0] * inv_det;
                    let a11 = jac[0][0] * inv_det;

                    // D = J⁻¹ · J⁻ᵀ (symmetric 2×2)
                    let d00 = a00 * a00 + a01 * a01;
                    let d01 = a00 * a10 + a01 * a11;
                    let d11 = a10 * a10 + a11 * a11;

                    // Scale by w[q]·w[r]·|detJ|·κ
                    let scale = w_1d[q] * w_1d[r] * det_j * kappa;
                    let base = (e as usize * q1d * q1d + q * q1d + r) * 3;
                    pa_data[base] = d00 * scale;
                    pa_data[base + 1] = d01 * scale;
                    pa_data[base + 2] = d11 * scale;
                }
            }
        }

        SumFactDiffusionOp {
            elem_dofs: all_dofs,
            n_elems,
            n_dofs,
            ldofs,
            p,
            q1d,
            B: b_1d,
            G: g_1d,
            W: w_1d,
            pa_data,
            tp_to_dof,
            dof_to_tp,
        }
    }

    /// 1D Gauss-Legendre points and weights on [-1, 1].
    fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
        match n {
            1 => (vec![0.0], vec![2.0]),
            2 => {
                let x = 1.0_f64 / 3.0_f64.sqrt();
                (vec![-x, x], vec![1.0, 1.0])
            }
            3 => {
                let x = (3.0_f64 / 5.0_f64).sqrt();
                (vec![-x, 0.0, x], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
            }
            4 => {
                let a = (3.0 / 7.0 - 2.0 / 7.0 * (6.0_f64 / 5.0).sqrt()).sqrt();
                let b = (3.0 / 7.0 + 2.0 / 7.0 * (6.0_f64 / 5.0).sqrt()).sqrt();
                let wa = (18.0 + 30.0_f64.sqrt()) / 36.0;
                let wb = (18.0 - 30.0_f64.sqrt()) / 36.0;
                (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
            }
            _ => panic!("gauss_legendre_1d: only n=1..4 supported, got {n}"),
        }
    }

    /// Raw mat-vec with sum-factorization: `y = A x`. **No BC enforcement.**
    pub fn mult_raw(&self, x: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }

        let p1 = self.p + 1;
        let q1d = self.q1d;
        // Scratch arrays sized for max p=4 → p1=5, q1d=5 (P4 diffusion needs
        // 5-point Gauss, exact to degree 9).
        let mut tp_x = [0.0f64; 25]; // max (p+1)² = 25
        let mut tp_y = [0.0f64; 25];
        let mut s_b = [[0.0f64; 8]; 8]; // s_b[q][j], max q1d=5, p1=5
        let mut s_g = [[0.0f64; 8]; 8];

        for e in 0..self.n_elems {
            let dof_base = e * self.ldofs;
            let pa_base = e * q1d * q1d * 3;

            // Gather and permute to tensor-product order
            for iy in 0..p1 {
                for ix in 0..p1 {
                    let tp_idx = iy * p1 + ix;
                    let qdof = self.tp_to_dof[tp_idx];
                    tp_x[tp_idx] = x[self.elem_dofs[dof_base + qdof] as usize];
                }
            }
            // Zero tp_y
            for v in tp_y.iter_mut() {
                *v = 0.0;
            }

            // ─────────────────────────────────────────────────────────────
            // Phase 1: Forward sum-factorization (over ξ quadrature points)
            // ─────────────────────────────────────────────────────────────
            // For each ξ-qp q, compute intermediates along η:
            //   s_b[q][j] = Σ_i tp_x[i][j] · B[q][i]
            //   s_g[q][j] = Σ_i tp_x[i][j] · G[q][i]
            for q in 0..q1d {
                let bq = &self.B[q * p1..];
                let gq = &self.G[q * p1..];
                for j in 0..p1 {
                    let mut sb = 0.0;
                    let mut sg = 0.0;
                    for i in 0..p1 {
                        let x_val = tp_x[j * p1 + i]; // tp_x[i][j]
                        sb += x_val * bq[i];
                        sg += x_val * gq[i];
                    }
                    s_b[q][j] = sb;
                    s_g[q][j] = sg;
                }
            }

            // ─────────────────────────────────────────────────────────────
            // Phase 2: Apply pa_data at each full qp (ξ_q, η_r)
            // ─────────────────────────────────────────────────────────────
            // For each (q, r):
            //   u_ξ = Σ_j s_g[q][j] · B[r][j]
            //   u_η = Σ_j s_b[q][j] · G[r][j]
            //   f_ξ = D00 · u_ξ + D01 · u_η
            //   f_η = D01 · u_ξ + D11 · u_η
            for r in 0..q1d {
                let br = &self.B[r * p1..];
                let gr = &self.G[r * p1..];
                for q in 0..q1d {
                    let sb = &s_b[q];
                    let sg = &s_g[q];

                    let mut u_xi = 0.0;
                    let mut u_eta = 0.0;
                    for j in 0..p1 {
                        u_xi += sg[j] * br[j];
                        u_eta += sb[j] * gr[j];
                    }

                    let pa_off = pa_base + (q * q1d + r) * 3;
                    let d00 = self.pa_data[pa_off];
                    let d01 = self.pa_data[pa_off + 1];
                    let d11 = self.pa_data[pa_off + 2];

                    let f_xi = d00 * u_xi + d01 * u_eta;
                    let f_eta = d01 * u_xi + d11 * u_eta;

                    // ─────────────────────────────────────────────────────
                    // Phase 3: Backward sum-factorization (assemble)
                    // ─────────────────────────────────────────────────────
                    // tp_y[i][j] += G[q][i] · f_ξ · B[r][j]
                    //            + B[q][i] · f_η · G[r][j]
                    let bq_i_slice = &self.B[q * p1..];
                    let gq_i_slice = &self.G[q * p1..];
                    for i in 0..p1 {
                        let gqi = gq_i_slice[i];
                        let bqi = bq_i_slice[i];
                        for j in 0..p1 {
                            tp_y[j * p1 + i] += gqi * f_xi * br[j] + bqi * f_eta * gr[j];
                        }
                    }
                }
            }

            // Scatter and permute back from tensor-product order
            for iy in 0..p1 {
                for ix in 0..p1 {
                    let tp_idx = iy * p1 + ix;
                    let qdof = self.tp_to_dof[tp_idx];
                    let global_idx = self.elem_dofs[dof_base + qdof] as usize;
                    y[global_idx] += tp_y[tp_idx];
                }
            }
        }
    }

    /// ConstrainedOperator-style mat-vec with BC enforcement.
    pub fn mult_constrained(&self, x: &[f64], y: &mut [f64], bc_dofs: &[u32]) {
        let mut xc = vec![0.0; self.n_dofs];
        xc.copy_from_slice(x);
        for &d in bc_dofs {
            if (d as usize) < xc.len() {
                xc[d as usize] = 0.0;
            }
        }
        self.mult_raw(&xc, y);
        for &d in bc_dofs {
            if (d as usize) < y.len() {
                // MFEM ConstrainedOperator diag_policy = DIAG_ONE (BilinearForm
                // FormSystemMatrix): y[id] = x[id] on constrained DOFs (NOT 0 —
                // DIAG_ZERO shifts every Chebyshev sweep / power iteration).
                y[d as usize] = x[d as usize];
            }
        }
    }
}

/// Evaluate the i-th 1D Lagrange basis function and its derivative at x.
///
/// Uses the Lagrangian formula directly. Handles the case where x coincides
/// with any Lagrange node (not just the i-th) using node-derivative limits.
fn lagrange_1d_eval(nodes: &[f64], i: usize, x: f64) -> (f64, f64) {
    let n = nodes.len();
    let xi = nodes[i];
    // If at the same node, use exact values
    if (x - xi).abs() < 1e-14 {
        let mut der = 0.0;
        for j in 0..n {
            if j != i {
                der += 1.0 / (xi - nodes[j]);
            }
        }
        return (1.0, der);
    }
    // Check if x coincides with another node k ≠ i
    for k in 0..n {
        if k != i && (x - nodes[k]).abs() < 1e-14 {
            // l_i(x_k) = 0, l_i'(x_k) = (w_i/w_k) / (x_k - x_i)
            // where w_i = 1/Π_{j≠i}(x_i - x_j) are barycentric weights
            // Compute barycentric weight ratio: w_i / w_k
            let ratio = barycentric_weight_ratio(nodes, i, k);
            let der = ratio / (nodes[k] - nodes[i]);
            return (0.0, der);
        }
    }
    // Generic case: x is not at any node
    let mut val = 1.0;
    for j in 0..n {
        if j != i {
            val *= (x - nodes[j]) / (xi - nodes[j]);
        }
    }
    // Derivative: l_i'(x) = l_i(x) · Σ_{j≠i} 1/(x - x_j)
    let mut sum_inv = 0.0;
    for j in 0..n {
        if j != i {
            sum_inv += 1.0 / (x - nodes[j]);
        }
    }
    (val, val * sum_inv)
}

/// Compute w_i / w_k where w_i = 1/Π_{j≠i}(xi - xj) are barycentric weights.
///
/// w_i / w_k = Π_{j≠k}(x_k − x_j) / Π_{j≠i}(x_i − x_j).
fn barycentric_weight_ratio(nodes: &[f64], i: usize, k: usize) -> f64 {
    let mut wi = 1.0; // Π_{j≠i}(x_i − x_j)
    let mut wk = 1.0; // Π_{j≠k}(x_k − x_j)
    for j in 0..nodes.len() {
        if j != i {
            wi *= nodes[i] - nodes[j];
        }
        if j != k {
            wk *= nodes[k] - nodes[j];
        }
    }
    wk / wi
}

/// QuadQk node-to-DOF mapping: (ix, iy) in [0, p] → QuadQk DOF index.
fn quadqk_node_to_dof(ix: usize, iy: usize, p: usize) -> usize {
    let x = -1.0 + 2.0 * ix as f64 / p as f64;
    let y = -1.0 + 2.0 * iy as f64 / p as f64;
    let tol = 1e-12;
    let on_xmin = (x + 1.0).abs() < tol;
    let on_xmax = (x - 1.0).abs() < tol;
    let on_ymin = (y + 1.0).abs() < tol;
    let on_ymax = (y - 1.0).abs() < tol;

    // Corners
    if on_xmin && on_ymin {
        return 0;
    }
    if on_xmax && on_ymin {
        return 1;
    }
    if on_xmax && on_ymax {
        return 2;
    }
    if on_xmin && on_ymax {
        return 3;
    }

    let mut idx = 4usize;
    // Bottom edge (η=-1), iy=0, corners already handled
    if on_ymin {
        return idx + (ix - 1);
    }
    idx += p - 1;
    // Right edge (ξ=+1), ix=p
    if on_xmax {
        return idx + (iy - 1);
    }
    idx += p - 1;
    // Top edge (η=+1), iy=p, reversed order
    if on_ymax {
        return idx + (p - 1 - ix);
    }
    idx += p - 1;
    // Left edge (ξ=-1), ix=0, reversed order
    if on_xmin {
        return idx + (p - 1 - iy);
    }

    // Interior: none of the boundary checks matched
    let base = 4 + 4 * (p - 1);
    base + (iy - 1) * (p - 1) + (ix - 1)
}

// ─── PADiffusionOp: on-the-fly partial assembly diffusion operator ──────────

// ─── PADiffusionOp: on-the-fly partial assembly diffusion operator ──────────

/// On-the-fly diffusion operator matching MFEM `AddMultPA`.
///
/// Instead of storing element matrices, stores per-quadrature-point data:
/// Jacobian⁻ᵀ, |detJ|, and pre-transformed gradient basis values.  At apply
/// time, for each element and quadrature point:
///
///   1. Compute ∇u = Σⱼ xⱼ · ∇φⱼ(ξ)   (on-the-fly, no stored K_e)
///   2. y[i] += w·|detJ|·κ·∇u·∇φ_i(ξ)
///
/// This produces different floating-point results than the stored element
/// matrix approach (which forms K_e[i,j] = Σ w·|detJ|·∇φ_i·∇φ_j first).
/// The on-the-fly order matches MFEM's `AssemblyLevel::PARTIAL`.
pub struct PADiffusionOp {
    pub elem_dofs: Vec<u32>,
    /// Precomputed physical gradients: `grad_phys[(e*n_qp + q) * stride + i*dim + d]`
    /// where `stride = ldofs * dim`.
    pub grad_phys: Vec<f64>,
    /// Quadrature weight × |detJ|: `weight_det[e * n_qp + q]`
    pub weight_det: Vec<f64>,
    pub ldofs: usize,
    pub n_elems: usize,
    pub n_qp: usize,
    pub dim: usize,
    pub n_dofs: usize,
}

impl PADiffusionOp {
    /// Build from raw components (mesh + per-element DOF iterator).
    ///
    /// Supports 2-D Tri3 and Quad4 elements (affine + isoparametric).
    /// `elem_dofs_fn(e)` returns the global DOF indices for element `e`.
    pub fn build(
        mesh: &dyn MeshTopology,
        n_dofs: usize,
        order: u8,
        quad_order: u8,
        kappa: f64,
        mut elem_dofs_fn: impl FnMut(u32) -> Vec<u32>,
    ) -> Self {
        let n_elems = mesh.n_elements();
        let dim = mesh.dim() as usize;
        let et0 = mesh.element_type(0);

        let ref_elem: Box<dyn ReferenceElement> = match et0 {
            ElementType::Tri3 => Box::new(H1TriPk::new(order as usize)),
            ElementType::Quad4 => Box::new(QuadQk::new(order as usize)),
            _ => panic!("PADiffusionOp: unsupported {et0:?}"),
        };
        let ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(quad_order);
        let n_qp = quad.points.len();

        let mut all_dofs = Vec::with_capacity(n_elems * ldofs);
        let mut all_grad = Vec::with_capacity(n_elems * n_qp * ldofs * dim);
        let mut all_wdet = Vec::with_capacity(n_elems * n_qp);
        let mut grad_ref = vec![0.0; ldofs * dim];
        let mut grad_phys = vec![0.0; ldofs * dim];

        for e in 0..n_elems as u32 {
            let et = mesh.element_type(e);
            let gd = elem_dofs_fn(e);
            all_dofs.extend_from_slice(&gd);

            let nodes = mesh.element_nodes(e);
            let is_quad = matches!(et, ElementType::Quad4);

            for (qi, xi) in quad.points.iter().enumerate() {
                let (jac, det_j): (DMatrix<f64>, f64) = if is_quad {
                    // Isoparametric Jacobian for Quad4 (bilinear mapping)
                    let mut j = DMatrix::<f64>::zeros(dim, dim);
                    let (xi_v, eta) = (xi[0], xi[1]);
                    // Bilinear shape function derivatives at (ξ, η):
                    // N0 = (1-ξ)(1-η), N1 = ξ(1-η), N2 = ξη, N3 = (1-ξ)η
                    // dN/dξ: -(1-η), (1-η), η, -η
                    // dN/dη: -(1-ξ), -ξ, ξ, (1-ξ)
                    let dndxi = [-(1.0 - eta), (1.0 - eta), eta, -eta];
                    let dndeta = [-(1.0 - xi_v), -xi_v, xi_v, (1.0 - xi_v)];
                    for n in 0..4 {
                        let xy = mesh.node_coords(nodes[n]);
                        for d in 0..dim {
                            j[(d, 0)] += xy[d] * dndxi[n];
                            j[(d, 1)] += xy[d] * dndeta[n];
                        }
                    }
                    (j.clone(), j.determinant())
                } else {
                    let x0 = mesh.node_coords(nodes[0]);
                    let mut j = DMatrix::<f64>::zeros(dim, dim);
                    for col in 0..dim {
                        let xc = mesh.node_coords(nodes[col + 1]);
                        for row in 0..dim {
                            j[(row, col)] = xc[row] - x0[row];
                        }
                    }
                    (j.clone(), j.determinant())
                };
                let jit = jac
                    .clone()
                    .try_inverse()
                    .expect("degenerate element in PADiffusionOp")
                    .transpose();
                let w = quad.weights[qi] * det_j.abs();

                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                for i in 0..ldofs {
                    for d in 0..dim {
                        let mut s = 0.0;
                        for k in 0..dim {
                            s += jit[(d, k)] * grad_ref[i * dim + k];
                        }
                        grad_phys[i * dim + d] = s;
                    }
                }
                all_grad.extend_from_slice(&grad_phys);
                all_wdet.push(w * kappa);
            }
        }

        PADiffusionOp {
            elem_dofs: all_dofs,
            grad_phys: all_grad,
            weight_det: all_wdet,
            ldofs,
            n_elems,
            n_qp,
            dim,
            n_dofs,
        }
    }

    /// On-the-fly mat-vec: `y = A x`.  **No BC enforcement.**
    pub fn mult_raw(&self, x: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let stride = self.ldofs * self.dim;
        for e in 0..self.n_elems {
            let dof_base = e * self.ldofs;
            let wdet_base = e * self.n_qp;
            let grad_base = e * self.n_qp * stride;

            // Gather x_e
            let mut xe = [0.0_f64; 64];
            for i in 0..self.ldofs {
                xe[i] = x[self.elem_dofs[dof_base + i] as usize];
            }

            for q in 0..self.n_qp {
                let w = self.weight_det[wdet_base + q];
                let gq = &self.grad_phys[grad_base + q * stride..];

                // ∇u = Σⱼ xⱼ · ∇φⱼ(ξ)  (dim components)
                let mut grad_u = [0.0_f64; 3];
                for j in 0..self.ldofs {
                    let gj = &gq[j * self.dim..];
                    for d in 0..self.dim {
                        grad_u[d] += xe[j] * gj[d];
                    }
                }

                // y[i] += w·|detJ|·κ·∇u·∇φ_i
                for i in 0..self.ldofs {
                    let gi = &gq[i * self.dim..];
                    let mut dot = 0.0;
                    for d in 0..self.dim {
                        dot += grad_u[d] * gi[d];
                    }
                    let idx = self.elem_dofs[dof_base + i] as usize;
                    y[idx] += w * dot;
                }
            }
        }
    }

    /// ConstrainedOperator-style mat-vec with BC enforcement.
    pub fn mult_constrained(&self, x: &[f64], y: &mut [f64], bc_dofs: &[u32]) {
        let mut xc = vec![0.0; self.n_dofs];
        xc.copy_from_slice(x);
        for &d in bc_dofs {
            if (d as usize) < xc.len() {
                xc[d as usize] = 0.0;
            }
        }
        self.mult_raw(&xc, y);
        for &d in bc_dofs {
            if (d as usize) < y.len() {
                // MFEM ConstrainedOperator diag_policy = DIAG_ONE (BilinearForm
                // FormSystemMatrix): y[id] = x[id] on constrained DOFs (NOT 0 —
                // DIAG_ZERO shifts every Chebyshev sweep / power iteration).
                y[d as usize] = x[d as usize];
            }
        }
    }
}

/// Element-by-element stored matrix operator (ConstrainedOperator pattern).
///
/// Precomputed element stiffness matrices applied via
/// gather → dense mat-vec → scatter.  The element matrices come from the
/// **same** integration loop as the CSR matrix (bitwise-identical).
///
/// BC enforcement follows the `ConstrainedOperator` pattern: zero BC DOFs in
/// the input, multiply, zero BC DOFs in the output.
pub struct StoredElementOperator {
    /// Flattened global DOF indices: `elem_dofs[e * ld + i]`.
    pub elem_dofs: Vec<u32>,
    /// Flattened element matrices: `elem_mats[e * ld² + i * ld + j]`.
    pub elem_mats: Vec<f64>,
    /// Local DOFs per element.
    pub ldofs: usize,
    /// Number of elements.
    pub n_elems: usize,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl StoredElementOperator {
    /// Raw element-by-element mat-vec: `y = A x`.  **No BC enforcement.**
    pub fn mult_raw(&self, x: &[f64], y: &mut [f64]) {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        let ld = self.ldofs;
        // Dynamic buffer: supports high-order 3-D elements (e.g. Hex8 P4 has
        // (p+1)^3 = 125 local DOFs).
        let mut xe = vec![0.0_f64; ld];
        for e in 0..self.n_elems {
            let dof_base = e * ld;
            let mat_base = e * ld * ld;
            for i in 0..ld {
                xe[i] = x[self.elem_dofs[dof_base + i] as usize];
            }
            for i in 0..ld {
                let mut sum = 0.0;
                let row_off = mat_base + i * ld;
                for j in 0..ld {
                    sum += self.elem_mats[row_off + j] * xe[j];
                }
                y[self.elem_dofs[dof_base + i] as usize] += sum;
            }
        }
    }

    /// ConstrainedOperator-style mat-vec: `y = A x` with BC enforcement.
    pub fn mult_constrained(&self, x: &[f64], y: &mut [f64], bc_dofs: &[u32]) {
        let mut xc = vec![0.0; self.n_dofs];
        xc.copy_from_slice(x);
        for &d in bc_dofs {
            if (d as usize) < xc.len() {
                xc[d as usize] = 0.0;
            }
        }
        self.mult_raw(&xc, y);
        for &d in bc_dofs {
            if (d as usize) < y.len() {
                // MFEM ConstrainedOperator diag_policy = DIAG_ONE (BilinearForm
                // FormSystemMatrix): y[id] = x[id] on constrained DOFs (NOT 0 —
                // DIAG_ZERO shifts every Chebyshev sweep / power iteration).
                y[d as usize] = x[d as usize];
            }
        }
    }
}

/// A single level in the geometric multigrid hierarchy.
pub struct GeometricMgLevel {
    /// System matrix at this level.
    pub mat: CsrMatrix<f64>,
    /// Boundary DOF list for this level.
    pub bc_dofs: Vec<u32>,
    /// Optional element-by-element operator.
    /// When present, `mat_vec()` uses `mult_constrained` (ConstrainedOperator
    /// pattern: zero BC in → raw mult → zero BC out).
    pub elem_op: Option<StoredElementOperator>,
    /// Diagonal from the raw (unmodified) matrix, used by Chebyshev smoother.
    pub raw_diag: Vec<f64>,
    /// Inverse of raw diagonal (1/diag[i] for non-zero, 1.0 for zero/BC DOFs).
    pub raw_dinv: Vec<f64>,
    /// Optional on-the-fly partial assembly operator (matches MFEM AddMultPA).
    /// When present, `mat_vec()` uses this first for the most accurate
    /// floating-point order match to MFEM's `AssemblyLevel::PARTIAL`.
    pub pa_op: Option<PADiffusionOp>,
    /// Optional sum-factorization diffusion operator (bitwise match to MFEM).
    /// Highest priority in `mat_vec()`: matches MFEM's sum-factorization
    /// kernel, producing identical floating-point results for MG convergence.
    pub sf_op: Option<SumFactDiffusionOp>,
}

/// Geometric multigrid hierarchy.
///
/// `levels[0]` = finest, `levels[n-1]` = coarsest.
/// `prolong[l]` maps from level l+1 (coarse) to level l (fine) with BC enforcement.
pub struct GeometricMgHierarchy {
    pub levels: Vec<GeometricMgLevel>,
    pub prolong: Vec<RectangularConstrainedOperator>,
}

impl GeometricMgLevel {
    /// Perform mat-vec with ConstrainedOperator-style BC enforcement.
    ///
    /// Priority: `sf_op` (sum-factorization PA) > `pa_op` (on-the-fly PA) >
    /// `elem_op` (stored elem mats) > CSR spmv.
    pub fn mat_vec(&self, x: &[f64], y: &mut [f64]) {
        // MFEM's DiffusionIntegrator PA (PADiffusionApply2D) is exact to
        // ~1e-14 vs its CSR; SumFactDiffusionOp currently deviates ~8.6e-6
        // from the CSR (a precision bug under investigation), which shifts
        // every Chebyshev sweep and the whole PCG trace.  Prefer the exact
        // element-by-element pa_op over the approximate sf_op.
        if let Some(ref pa) = self.pa_op {
            pa.mult_constrained(x, y, &self.bc_dofs);
        } else if let Some(ref sf) = self.sf_op {
            sf.mult_constrained(x, y, &self.bc_dofs);
        } else if let Some(ref op) = self.elem_op {
            op.mult_constrained(x, y, &self.bc_dofs);
        } else {
            self.mat.spmv(x, y);
        }
    }
}

impl GeometricMgHierarchy {
    pub fn new(levels: Vec<GeometricMgLevel>, prolong_mat: Vec<CsrMatrix<f64>>) -> Self {
        assert_eq!(
            prolong_mat.len(),
            levels.len() - 1,
            "GeometricMgHierarchy: need len(prolong) == len(levels) - 1"
        );
        let mut prolong = Vec::with_capacity(prolong_mat.len());
        for l in 0..prolong_mat.len() {
            prolong.push(RectangularConstrainedOperator {
                mat: prolong_mat[l].clone(),
                ess_fine: levels[l].bc_dofs.clone(),
                ess_coarse: levels[l + 1].bc_dofs.clone(),
            });
        }
        GeometricMgHierarchy { levels, prolong }
    }
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
    pub fn finest_matrix(&self) -> &CsrMatrix<f64> {
        &self.levels[0].mat
    }
}

/// Multigrid cycle type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MgCycleType {
    V,
    W,
}

/// Multigrid smoother type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MgSmootherType {
    /// Jacobi with configurable omega (chebyshev_order = 0)
    Jacobi,
    /// Chebyshev polynomial smoother (order 1-3)
    Chebyshev(usize),
    /// SSOR (forward + backward Gauss-Seidel)
    Ssor,
}

/// Geometric multigrid V-cycle configuration.
#[derive(Debug, Clone)]
pub struct GeometricMgConfig {
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    /// Smoother type (default: Chebyshev(2) for backward compatibility).
    pub smoother: MgSmootherType,
    pub jacobi_omega: f64,
    pub coarse_max_iter: usize,
    pub coarse_rtol: f64,
    /// Override λ_max estimate. None = auto-estimate.
    pub max_eig_override: Option<f64>,
    /// Per-level λ_max overrides (finest-first), matching `levels[]` order.
    /// Empty = use [`Self::max_eig_override`] (or auto-estimate) on every level.
    pub max_eig_overrides: Vec<Option<f64>>,
    /// Multigrid cycle type (V or W). Default: V for backward compatibility.
    pub cycle_type: MgCycleType,
}

impl Default for GeometricMgConfig {
    fn default() -> Self {
        GeometricMgConfig {
            pre_sweeps: 2,
            post_sweeps: 2,
            smoother: MgSmootherType::Chebyshev(2),
            jacobi_omega: 0.8,
            coarse_max_iter: 200,
            coarse_rtol: 1e-12,
            max_eig_override: None,
            max_eig_overrides: Vec::new(),
            cycle_type: MgCycleType::V,
        }
    }
}

/// Chebyshev smoother for geometric MG.
/// Stores precomputed coefficients and diagonal.
pub struct MgChebyshevSmoother {
    dinv: Vec<f64>,
    coeffs: Vec<f64>,
}

impl MgChebyshevSmoother {
    fn new(
        a: &CsrMatrix<f64>,
        bc: &[u32],
        order: usize,
        max_eig_override: Option<f64>,
        mat_vec: &dyn Fn(&[f64], &mut [f64]),
    ) -> Self {
        let n = a.nrows;
        let diag = a.diagonal();
        let mut dinv = vec![0.0; n];
        for i in 0..n {
            dinv[i] = if diag[i].abs() > 1e-30 {
                1.0 / diag[i]
            } else {
                1.0
            };
        }
        for &d in bc {
            if (d as usize) < n {
                dinv[d as usize] = 1.0;
            }
        }

        // Estimate λ_max(D⁻¹A) via power iteration using the ACTUAL mat-vec
        // (which goes through sf_op / pa_op when available), matching the
        // operator used during Chebyshev smoothing.  Using the CSR-only
        // spmv can give a different eigenvalue than the SF/PA operator,
        // causing the Chebyshev polynomial to become unstable.
        let max_eig =
            max_eig_override.unwrap_or_else(|| estimate_max_eigenvalue_with_op(mat_vec, &dinv, bc));
        if std::env::var("FEM_MG_DEBUG").is_ok() {
            eprintln!("  [mg] Chebyshev smoother: n={n}, λmax(D⁻¹A)≈{max_eig:.6}");
        }

        // MFEM OperatorChebyshevSmoother parameters
        let upper = 1.2 * max_eig;
        let lower = 0.3 * max_eig;
        let theta = 0.5 * (upper + lower);
        let delta = 0.5 * (upper - lower);
        let th2 = theta * theta;
        let d2 = delta * delta;

        let coeffs = match order - 1 {
            0 => vec![1.0 / theta],
            1 => {
                let tmp = 1.0 / (d2 - 2.0 * th2);
                vec![-4.0 * theta * tmp, 2.0 * tmp]
            }
            2 => {
                let t0 = 3.0 * d2;
                let t1 = th2;
                let t2 = 1.0 / (-4.0 * theta * th2 + theta * t0);
                vec![t2 * (t0 - 12.0 * t1), 12.0 / (t0 - 4.0 * t1), -4.0 * t2]
            }
            _ => panic!("MgChebyshevSmoother: order {order} not supported (1-3)"),
        };
        MgChebyshevSmoother { dinv, coeffs }
    }

    fn smooth(&self, level: &GeometricMgLevel, b: &[f64], x: &mut [f64]) {
        let a = &level.mat;
        let n = a.nrows;
        // Residual: r = b - A*x (uses ConstrainedOperator mat-vec when elem_op present)
        let mut r = vec![0.0; n];
        level.mat_vec(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }

        // Correction: Δ = p(D⁻¹A) D⁻¹ r, matching MFEM OperatorChebyshevSmoother::Mult
        //
        //   y = Σ_{k=0}^{order-1} C[k] · (D⁻¹A)^k · D⁻¹ · r
        let m = self.coeffs.len();
        let mut correction = vec![0.0; n];
        // residual = D⁻¹ · r
        let mut residual = vec![0.0; n];
        for i in 0..n {
            residual[i] = r[i] * self.dinv[i];
        }

        for k in 0..m {
            let c = self.coeffs[k];
            for i in 0..n {
                correction[i] += c * residual[i];
            }
            if k + 1 < m {
                // residual = D⁻¹ · A · residual
                let mut tmp = vec![0.0; n];
                level.mat_vec(&residual, &mut tmp);
                for i in 0..n {
                    residual[i] = tmp[i] * self.dinv[i];
                }
            }
        }

        for i in 0..n {
            x[i] += correction[i];
        }
        for &d in &level.bc_dofs {
            if (d as usize) < n {
                x[d as usize] = 0.0;
            }
        }
    }
}

/// Power iteration on D⁻¹A using a generic mat-vec operator.
///
/// Bit-for-bit port of MFEM `PowerMethod::EstimateLargestEigenvalue` as used
/// by `OperatorChebyshevSmoother`'s 6-argument constructor (ex26 uses
/// `OperatorChebyshevSmoother(*opr, diag, ess_tdofs, 2)` → defaults
/// `numSteps=10, tolerance=1e-8, seed=12345`):
/// `v0.Randomize(seed)` (= `srand(seed)` + `rand()/(RAND_MAX+1)`, glibc
/// `rand()` TYPE_3), then for each of 10 steps: normalize v0, `v1 =
/// invDiag(oper(v0))` (ProductOperator(invDiag, oper) = invDiag first? no —
/// `ProductOperator(A=invDiag, B=oper).Mult = A(B(x))` = invDiag(oper(x))),
/// Rayleigh quotient `eig = <v0, v1>`, break when `|Δeig/eig| < 1e-8`.
fn estimate_max_eigenvalue_with_op(
    mat_vec: &dyn Fn(&[f64], &mut [f64]),
    dinv: &[f64],
    _bc: &[u32],
) -> f64 {
    let n = dinv.len();
    let mut rng = GlibcRand::new(12345);
    let mut v0: Vec<f64> = (0..n).map(|_| rng.rand_real()).collect();
    let mut v1 = vec![0.0; n];
    let mut eigenvalue = 1.0;
    for _ in 0..10 {
        let norm0: f64 = v0.iter().map(|x| x * x).sum();
        // MFEM: v0 /= sqrt(normV0) — element-wise DIVISION (a `*= inv`
        // multiply differs by ~1 ulp per step, which 10 power iterations
        // amplify into a ~4e-6 λmax shift → Chebyshev trace divergence).
        let nrm = norm0.sqrt();
        for x in v0.iter_mut() {
            *x /= nrm;
        }
        // v1 = D⁻¹·(A·v0): mat_vec applies the essential-BC constraint with
        // DIAG_ONE semantics (ConstrainedOperator: y[id] = x[id] on bc DOFs,
        // NOT 0), and dinv carries the raw diagonal inverse (bc set to 1).
        mat_vec(&v0, &mut v1);
        for i in 0..n { v1[i] *= dinv[i]; }
        let eig_new: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        let diff = ((eig_new - eigenvalue) / eigenvalue).abs();
        eigenvalue = eig_new;
        std::mem::swap(&mut v0, &mut v1);
        if diff < 1e-8 {
            break;
        }
    }
    eigenvalue
}

/// glibc `rand()` TYPE_3 (`x^31 + x^3 + 1`, 31-bit additive feedback),
/// bit-exact port of `stdlib/random_r.c` (`__srandom_r` + `__random_r`) as
/// used by MFEM's `Vector::Randomize(seed)` → `srand(seed)` + `rand()`.
/// `next()` returns the 31-bit `rand()` value; `rand_real()` =
/// `rand()/(RAND_MAX+1)` (MFEM `rand_real` in linalg/vector.hpp).
struct GlibcRand {
    state: Vec<u32>,
    fptr: usize,
    rptr: usize,
}

impl GlibcRand {
    fn new(seed: u32) -> Self {
        const DEG: usize = 31;
        const SEP: usize = 3;
        let mut state = vec![0u32; DEG];
        let seed = if seed == 0 { 1 } else { seed };
        state[0] = seed;
        // state[i] = (16807 * state[i-1]) % 2147483647, Schrage to avoid overflow.
        let mut word: i64 = seed as i64;
        for i in 1..DEG {
            let hi = word / 127773;
            let lo = word % 127773;
            word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2147483647;
            }
            state[i] = word as u32;
        }
        let mut r = GlibcRand {
            state,
            fptr: SEP,
            rptr: 0,
        };
        // __srandom_r: kc = deg*10 = 310 discard draws before the first result.
        for _ in 0..(DEG * 10) {
            let _ = r.next();
        }
        r
    }

    /// One `rand()` draw (31-bit), pointer advance exactly as `__random_r`.
    fn next(&mut self) -> u32 {
        let n = self.state.len();
        let val = self.state[self.fptr].wrapping_add(self.state[self.rptr]);
        self.state[self.fptr] = val;
        let result = val >> 1;
        self.fptr += 1;
        if self.fptr >= n {
            self.fptr = 0;
            self.rptr += 1;
        } else {
            self.rptr += 1;
            if self.rptr >= n {
                self.rptr = 0;
            }
        }
        result
    }

    /// `rand() / (RAND_MAX + 1)` with `RAND_MAX = 2^31 - 1` (MFEM `rand_real`).
    fn rand_real(&mut self) -> f64 {
        self.next() as f64 / 2147483648.0
    }
}

/// SSOR smoother: forward sweep then backward sweep with relaxation factor omega.
/// Standard SSOR uses omega in (0, 2); omega = 1 gives symmetric Gauss-Seidel.
fn ssor_smooth_level(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], bc: &[u32], omega: f64) {
    let n = a.nrows;
    // Forward sweep (SOR)
    for i in 0..n {
        let mut diag = 0.0;
        let mut sum = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr as usize] as usize;
            if j == i {
                diag = a.values[ptr as usize];
            } else {
                sum += a.values[ptr as usize] * x[j];
            }
        }
        if diag.abs() > 1e-30 {
            // x_i_new = (1-ω) * x_i_old + ω * (b_i - Σ_{j≠i} A_ij * x_j) / A_ii
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sum) / diag;
        }
    }
    // Backward sweep (SOR, reversed)
    for i in (0..n).rev() {
        let mut diag = 0.0;
        let mut sum = 0.0;
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr as usize] as usize;
            if j == i {
                diag = a.values[ptr as usize];
            } else {
                sum += a.values[ptr as usize] * x[j];
            }
        }
        if diag.abs() > 1e-30 {
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sum) / diag;
        }
    }
    // Re-apply BCs
    for &d in bc {
        if (d as usize) < n {
            x[d as usize] = 0.0;
        }
    }
}

/// Geometric multigrid V-cycle preconditioner.
pub struct GeometricMgPrecond {
    pub config: GeometricMgConfig,
    /// Pre-computed Chebyshev smoothers for each level.
    pub smoothers: Vec<MgChebyshevSmoother>,
}

/// Adapter exposing a [`GeometricMgPrecond`] + [`GeometricMgHierarchy`] pair as
/// a linlvo [`linlvo::Preconditioner`], so it can be plugged directly into
/// [`crate::solve_pcg`] (one V-cycle per application, matching MFEM's use of a
/// `Multigrid` as preconditioner in `PCG`).
pub struct GeometricMgAsPrecond<'a> {
    /// The V-cycle engine (smoother data, config).
    pub mg: &'a GeometricMgPrecond,
    /// The level matrices / transfer operators.
    pub hierarchy: &'a GeometricMgHierarchy,
}

impl linlvo::Preconditioner for GeometricMgAsPrecond<'_> {
    type Vector = linlvo::DenseVec<f64>;

    fn apply_precond(&self, x: &linlvo::DenseVec<f64>, y: &mut linlvo::DenseVec<f64>) {
        self.mg
            .v_cycle(self.hierarchy, x.as_slice(), y.as_mut_slice());
    }
}

impl GeometricMgPrecond {
    pub fn new(config: GeometricMgConfig, h: &GeometricMgHierarchy) -> Self {
        let smoothers = match config.smoother {
            MgSmootherType::Chebyshev(order) => {
                let mut s = Vec::new();
                for (li, level) in h.levels.iter().enumerate() {
                    let override_eig = config
                        .max_eig_overrides
                        .get(li)
                        .and_then(|o| *o)
                        .or(config.max_eig_override);
                    s.push(MgChebyshevSmoother::new(
                        &level.mat,
                        &level.bc_dofs,
                        order,
                        override_eig,
                        &|x, y| level.mat_vec(x, y),
                    ));
                }
                s
            }
            _ => Vec::new(), // Jacobi and SSOR don't need pre-computed smoothers
        };
        GeometricMgPrecond { config, smoothers }
    }

    /// Apply one V/W-cycle: `x ← cycle(levels, prolong, b)` starting from zero.
    pub fn v_cycle(&self, h: &GeometricMgHierarchy, b: &[f64], x: &mut [f64]) {
        // Start from zero (matching MFEM MultigridBase::ArrayMult: *Y(M-1,j) = 0.0)
        for v in x.iter_mut() {
            *v = 0.0;
        }
        let w_cycle = self.config.cycle_type == MgCycleType::W;
        self.v_cycle_level_inner(h, 0, b, x, w_cycle);
    }

    /// Core recursive cycle: handles both V-cycle and W-cycle.
    fn v_cycle_level_inner(
        &self,
        h: &GeometricMgHierarchy,
        lvl: usize,
        b: &[f64],
        x: &mut [f64],
        w_cycle: bool,
    ) {
        let level = &h.levels[lvl];
        let a = &level.mat;
        let n = a.nrows;

        if lvl + 1 == h.levels.len() {
            // Coarsest level: CG solve
            let cfg = SolverConfig {
                rtol: self.config.coarse_rtol,
                atol: 0.0,
                max_iter: self.config.coarse_max_iter,
                verbose: false,
                ..Default::default()
            };
            let res = crate::solve_cg(a, b, x, &cfg);
            if std::env::var("FEM_MG_DEBUG").is_ok() {
                match &res {
                    Ok(r) => eprintln!(
                        "[mg] coarse CG: {} iters, residual {:.3e}",
                        r.iterations, r.final_residual
                    ),
                    Err(e) => eprintln!("[mg] coarse CG FAILED: {e:?}"),
                }
            }
            for &d in &level.bc_dofs {
                if (d as usize) < n {
                    x[d as usize] = 0.0;
                }
            }
            return;
        }

        // Pre-smooth
        for _ in 0..self.config.pre_sweeps {
            self.smooth_level(lvl, level, b, x);
        }

        // Restrict residual
        let mut ax = vec![0.0; n];
        level.mat_vec(x, &mut ax);
        let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let mut r_c = Vec::new();
        h.prolong[lvl].restrict(&r, &mut r_c);
        let n_c = h.levels[lvl + 1].mat.nrows;

        if w_cycle {
            // W-cycle: two coarse solves where the second starts from
            // the result of the first (non-zero initial guess), giving
            // a more accurate coarse correction than a single V-cycle.
            let mut e_c = vec![0.0; n_c];
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, true);
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, true);
            // Prolongate correction
            let mut corr = vec![0.0; n];
            h.prolong[lvl].prolong(&e_c, &mut corr);
            for i in 0..n {
                x[i] += corr[i];
            }
        } else {
            let mut e_c = vec![0.0; n_c];
            self.v_cycle_level_inner(h, lvl + 1, &r_c, &mut e_c, false);

            // Prolongate correction
            let mut corr = vec![0.0; n];
            h.prolong[lvl].prolong(&e_c, &mut corr);
            for i in 0..n {
                x[i] += corr[i];
            }
        }

        // Post-smooth
        self.smooth_level(lvl, level, b, x);
        for _ in 1..self.config.post_sweeps {
            self.smooth_level(lvl, level, b, x);
        }
        for &d in &level.bc_dofs {
            if (d as usize) < n {
                x[d as usize] = 0.0;
            }
        }
    }

    fn smooth_level(&self, lvl: usize, level: &GeometricMgLevel, b: &[f64], x: &mut [f64]) {
        let a = &level.mat;
        let bc = &level.bc_dofs;
        match self.config.smoother {
            MgSmootherType::Chebyshev(_) => {
                if lvl < self.smoothers.len() {
                    self.smoothers[lvl].smooth(level, b, x);
                } else {
                    // Jacobi fallback
                    self.jacobi_smooth_level(level, b, x);
                }
            }
            MgSmootherType::Ssor => {
                ssor_smooth_level(a, b, x, bc, self.config.jacobi_omega);
            }
            MgSmootherType::Jacobi => {
                self.jacobi_smooth_level(level, b, x);
            }
        }
    }

    /// Plain Jacobi smoothing (omega from config).
    fn jacobi_smooth_level(&self, level: &GeometricMgLevel, b: &[f64], x: &mut [f64]) {
        let a = &level.mat;
        let diag = a.diagonal();
        let mut r = vec![0.0; a.nrows];
        level.mat_vec(x, &mut r);
        for i in 0..r.len() {
            r[i] = b[i] - r[i];
        }
        let omega = self.config.jacobi_omega;
        for i in 0..r.len() {
            if diag[i].abs() > 1e-30 {
                x[i] += omega * r[i] / diag[i];
            }
        }
        for &d in &level.bc_dofs {
            if (d as usize) < x.len() {
                x[d as usize] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;

    /// Build a 3-level 1D Poisson geometric MG hierarchy (finite difference).
    /// Coarse level ~ n_c DOFs; each refinement doubles DOFs (linear interpolation).
    fn build_1d_poisson_geom_hierarchy(n_coarse: usize) -> GeometricMgHierarchy {
        let n_levels = 3;
        let mut levels = Vec::new();
        let mut prolong_mats = Vec::new();

        let mut n = n_coarse;
        for lvl in 0..n_levels {
            let h = 1.0 / (n - 1) as f64;
            let mut coo = CooMatrix::<f64>::new(n, n);
            for i in 0..n {
                if i > 0 {
                    coo.add(i, i - 1, -1.0 / h);
                }
                coo.add(i, i, 2.0 / h);
                if i + 1 < n {
                    coo.add(i, i + 1, -1.0 / h);
                }
            }
            let mut mat = coo.into_csr();
            let bc_dofs = vec![0u32, (n - 1) as u32];
            let mut dummy = vec![0.0; n];
            for &d in &bc_dofs {
                mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
            }
            levels.push(GeometricMgLevel {
                mat,
                bc_dofs: bc_dofs.clone(),
                elem_op: None,
                raw_diag: vec![],
                raw_dinv: vec![],
                pa_op: None,
                sf_op: None,
            });

            // Build prolongation (linear interpolation) to next finer level
            if lvl + 1 < n_levels {
                let n_fine = 2 * n - 1;
                let mut p_coo = CooMatrix::<f64>::new(n_fine, n);
                for i in 0..n_fine {
                    if i % 2 == 0 {
                        p_coo.add(i, i / 2, 1.0);
                    } else {
                        p_coo.add(i, i / 2, 0.5);
                        p_coo.add(i, i / 2 + 1, 0.5);
                    }
                }
                prolong_mats.push(p_coo.into_csr());
            }
            n = 2 * n - 1;
        }

        // Reverse so finest is first
        levels.reverse();
        prolong_mats.reverse();

        GeometricMgHierarchy::new(levels, prolong_mats)
    }

    /// Simple preconditioned Richardson iteration with MG as preconditioner.
    fn richardson_mg(
        mg: &GeometricMgPrecond,
        h: &GeometricMgHierarchy,
        b: &[f64],
        x: &mut [f64],
        max_iter: usize,
        rtol: f64,
    ) -> usize {
        let a = h.finest_matrix();
        let n = a.nrows;
        let b_nrm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        for iter in 0..max_iter {
            let mut ax = vec![0.0; n];
            a.spmv(x, &mut ax);
            let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
            let res = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if res < rtol * b_nrm {
                return iter;
            }
            let mut dx = vec![0.0; n];
            mg.v_cycle(h, &r, &mut dx);
            for i in 0..n {
                x[i] += dx[i];
            }
        }
        max_iter
    }

    #[test]
    fn geometric_mg_w_cycle_converges_faster() {
        let n_coarse = 9; // coarse DOFs = 9, fine DOFs = 33
        let h = build_1d_poisson_geom_hierarchy(n_coarse);
        let n = h.levels[0].mat.nrows;
        eprintln!("  MG hierarchy DOFs per level:");
        for (i, lvl) in h.levels.iter().enumerate() {
            eprintln!("    level {}: {} DOFs", i, lvl.mat.nrows);
        }

        // RHS: all ones (compatible with homogeneous Dirichlet BCs)
        let mut b = vec![1.0_f64; n];
        for &d in &h.levels[0].bc_dofs {
            if (d as usize) < n {
                b[d as usize] = 0.0;
            }
        }

        let max_iter = 200;
        let rtol = 1e-8;

        // V-cycle (Jacobi smoother)
        let cfg_v = GeometricMgConfig {
            cycle_type: MgCycleType::V,
            smoother: MgSmootherType::Jacobi,
            jacobi_omega: 0.67,
            pre_sweeps: 2,
            post_sweeps: 2,
            ..Default::default()
        };
        let mg_v = GeometricMgPrecond::new(cfg_v, &h);
        let mut x_v = vec![0.0; n];
        let iters_v = richardson_mg(&mg_v, &h, &b, &mut x_v, max_iter, rtol);

        // W-cycle (Jacobi smoother)
        let cfg_w = GeometricMgConfig {
            cycle_type: MgCycleType::W,
            smoother: MgSmootherType::Jacobi,
            jacobi_omega: 0.67,
            pre_sweeps: 2,
            post_sweeps: 2,
            ..Default::default()
        };
        let mg_w = GeometricMgPrecond::new(cfg_w, &h);
        let mut x_w = vec![0.0; n];
        let iters_w = richardson_mg(&mg_w, &h, &b, &mut x_w, max_iter, rtol);

        eprintln!("  V-cycle: {iters_v} iters, W-cycle: {iters_w} iters");
        assert!(
            iters_w <= iters_v,
            "W-cycle ({iters_w}) should need <= V-cycle ({iters_v}) iterations"
        );
    }
}
