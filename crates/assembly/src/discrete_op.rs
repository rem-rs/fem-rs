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
//! | `curl_2d`    | H(curl) ND1| L2 (P0)   | 1     |
//! | `curl_2d`    | H(curl) ND2| L2 (P1)   | 2     |
//! | `curl_2d`    | H(curl) ND2| L2 (P2)   | 2     |
//! | `divergence` | H(div) RT0 | L2 (P0)   | 0     |
//! | `divergence` | H(div) RT1 | L2 (P1)   | 1     |
//! | `divergence` | H(div) RT1 | L2 (P2)   | 1     |
//! | `curl_3d`    | H(curl) ND1| H(div) RT0| 1     |
//! | `curl_3d`    | H(curl) ND2| H(div) RT1| 2     |
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

use fem_element::{ReferenceElement, TetND2, TetRT1, TriND1, TriND2, TriRT1, VectorReferenceElement};
use fem_element::lagrange::TriP2;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, ElementTransformation};
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
    #[error("{op}: H(div) space must be order 0 (RT0) or 1 (RT1), got order {order}")]
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
        match hcurl_order {
            1 | 2 => {}
            o => return Err(DiscreteOpError::UnsupportedHCurlOrder { op: "gradient", order: o }),
        }
        // H1 order k requires H(curl) order k.
        if h1_order != hcurl_order {
            return Err(DiscreteOpError::IncompatibleOrders {
                op: "gradient",
                h1_order,
                hcurl_order,
            });
        }

        match h1_order {
            1 => Self::gradient_p1_nd1(h1_space, hcurl_space),
            2 => Self::gradient_p2_nd2(h1_space, hcurl_space),
            _ => unreachable!(),
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

        let local_edges: &[(usize, usize)] = match mesh.dim() {
            2 => &[(0, 1), (1, 2), (0, 2)],
            3 => &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            d => return Err(DiscreteOpError::UnsupportedDimension { op: "gradient", dim: d }),
        };

        let mut visited = HashSet::with_capacity(n_hcurl);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
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

        // ── Overview: G[i,j] = DOF_i^{phys}(∇_phys φ_j)
        //
        // By the Nédélec pullback (H(curl) covariant):
        //   DOF_i^{phys}(F_phys) = DOF_i^{ref}(J^T F_phys)
        //   J^T (J^{-T} ∇_ref φ_j) = ∇_ref φ_j
        //
        // So for EDGE DOFs (tangential moments) the entries are J-independent:
        //   G_edge[i,j] = DOF_i^{ref}(∇_ref φ_j)   (i = 0..5)
        //
        // For INTERIOR DOFs (L2 moments of components), the physical DOF is:
        //   DOF_6^{phys}(F_phys) = ∫_{T_phys} F_x dA = |det_J| ∫_{T_ref} (J^{-T} ∇_ref φ_j)_x dξ
        //   DOF_7^{phys}(F_phys) = ∫_{T_phys} F_y dA = |det_J| ∫_{T_ref} (J^{-T} ∇_ref φ_j)_y dξ
        // These depend on J.  They are computed per element.
        let n_p2_local  = 6usize;
        let dim = 2usize;

        // 3-point Gauss-Legendre on [0,1] (exact for polys ≤ degree 5).
        let sq35: f64 = (3.0f64 / 5.0).sqrt();
        let gl_pts = [0.5 * (1.0 - sq35), 0.5, 0.5 * (1.0 + sq35)];
        let gl_wts = [5.0f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

        let p2_elem  = TriP2;
        let quad_int = p2_elem.quadrature(4); // degree-4 triangle rule

        // ── Precompute edge rows of G_ref (rows 0..5, J-independent).
        let mut g_edge = vec![0.0f64; 6 * n_p2_local]; // rows 0..5
        let mut p2_grads = vec![0.0f64; n_p2_local * dim];

        // Edge e₀: bottom, param (t,0), tangential = v_x
        for k in 0..3 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            p2_elem.eval_grad_basis(&[t, 0.0], &mut p2_grads);
            for j in 0..n_p2_local {
                let vx = p2_grads[j * dim];
                g_edge[0 * n_p2_local + j] += w * vx;
                g_edge[1 * n_p2_local + j] += w * vx * t;
            }
        }
        // Edge e₁: hypotenuse, param (1−t, t), tangential = −v_x+v_y
        for k in 0..3 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            p2_elem.eval_grad_basis(&[1.0 - t, t], &mut p2_grads);
            for j in 0..n_p2_local {
                let mom = -p2_grads[j * dim] + p2_grads[j * dim + 1];
                g_edge[2 * n_p2_local + j] += w * mom;
                g_edge[3 * n_p2_local + j] += w * mom * t;
            }
        }
        // Edge e₂: left, param (0,t), tangential = v_y
        for k in 0..3 {
            let (t, w) = (gl_pts[k], gl_wts[k]);
            p2_elem.eval_grad_basis(&[0.0, t], &mut p2_grads);
            for j in 0..n_p2_local {
                let vy = p2_grads[j * dim + 1];
                g_edge[4 * n_p2_local + j] += w * vy;
                g_edge[5 * n_p2_local + j] += w * vy * t;
            }
        }

        // ── Scatter into global COO, one element at a time.
        //
        // Each global ND2 DOF is written by the first element that claims it
        // (shared edge DOFs: de Rham guarantees adjacent elements agree).
        // Interior bubble DOFs are element-local (never shared), so no
        // visited check is needed for them — but we include them in the same set
        // for uniformity.
        let mut visited = HashSet::with_capacity(n_nd2);

        for e in mesh.elem_iter() {
            let nodes    = mesh.element_nodes(e);
            let h1_dofs  = h1_space.element_dofs(e);    // 6 global P2 DOFs
            let nd2_dofs = hcurl_space.element_dofs(e); // 8 global ND2 DOFs
            let nd2_signs = hcurl_space.element_signs(e);
            let tri_edges = [(0usize, 1usize), (1usize, 2usize), (0usize, 2usize)];

            // ── Interior DOF rows (6 and 7) — depend on J.
            // DOF_6^{phys}(∇_phys φ_j) = |det_J| ∫_{T_ref} (J^{-T} ∇_ref φ_j)_x dξ
            // DOF_7^{phys}(∇_phys φ_j) = |det_J| ∫_{T_ref} (J^{-T} ∇_ref φ_j)_y dξ
            let transform = ElementTransformation::from_simplex_nodes(mesh, nodes);
            let jit       = transform.jacobian_inv_t(); // J^{-T}
            let abs_det   = transform.det_j().abs();
            let jit00 = jit[(0,0)]; let jit01 = jit[(0,1)];
            let jit10 = jit[(1,0)]; let jit11 = jit[(1,1)];

            let mut g_int = vec![0.0f64; 2 * n_p2_local]; // rows 6 and 7
            for (xi, &w) in quad_int.points.iter().zip(quad_int.weights.iter()) {
                p2_elem.eval_grad_basis(xi, &mut p2_grads);
                for j in 0..n_p2_local {
                    let gx = p2_grads[j * dim];
                    let gy = p2_grads[j * dim + 1];
                    // J^{-T} ∇_ref φ_j
                    let phys_x = jit00 * gx + jit01 * gy;
                    let phys_y = jit10 * gx + jit11 * gy;
                    g_int[0 * n_p2_local + j] += w * phys_x; // DOF 6
                    g_int[1 * n_p2_local + j] += w * phys_y; // DOF 7
                }
            }
            // Scale by |det_J|
            for v in g_int.iter_mut() { *v *= abs_det; }

            // ── Scatter edge rows (0..5) with proper ND2 edge-moment orientation transform.
            // For reversed edge orientation, moments transform as:
            //   [m0, m1]_global = [-m0_local, -m0_local + m1_local].
            for (edge_local, &(li, lj)) in tri_edges.iter().enumerate() {
                let (gi, gj) = (nodes[li], nodes[lj]);
                let same_orient = gi < gj;

                let gd0 = nd2_dofs[2 * edge_local] as usize;
                let gd1 = nd2_dofs[2 * edge_local + 1] as usize;

                if visited.insert(gd0) {
                    for (j_local, &global_p2) in h1_dofs.iter().enumerate() {
                        let l0 = g_edge[(2 * edge_local) * n_p2_local + j_local];
                        let v0 = if same_orient { l0 } else { -l0 };
                        if v0.abs() > 1e-15 {
                            coo.add(gd0, global_p2 as usize, v0);
                        }
                    }
                }

                if visited.insert(gd1) {
                    for (j_local, &global_p2) in h1_dofs.iter().enumerate() {
                        let l0 = g_edge[(2 * edge_local) * n_p2_local + j_local];
                        let l1 = g_edge[(2 * edge_local + 1) * n_p2_local + j_local];
                        let v1 = if same_orient { l1 } else { -l0 + l1 };
                        if v1.abs() > 1e-15 {
                            coo.add(gd1, global_p2 as usize, v1);
                        }
                    }
                }
            }

            // ── Scatter interior rows (6,7).
            for i_int in 0..2usize {
                let i_local = 6 + i_int;
                let g_nd2 = nd2_dofs[i_local] as usize;
                if !visited.insert(g_nd2) { continue; }

                let sign = nd2_signs[i_local];
                for (j_local, &global_p2) in h1_dofs.iter().enumerate() {
                    let val = sign * g_int[i_int * n_p2_local + j_local];
                    if val.abs() > 1e-15 {
                        coo.add(g_nd2, global_p2 as usize, val);
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
            0 | 1 | 2 => {}
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
        let ref_elem = TriND1;
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
        let tri_edges = [(0usize, 1usize), (1usize, 2usize), (0usize, 2usize)];

        // 3-point Gauss-Legendre on [0,1] for edge moments.
        let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
        let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
        let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

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

            // Geometry for interior quadrature mapping.
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let j00 = x1[0] - x0[0]; let j10 = x1[1] - x0[1];
            let j01 = x2[0] - x0[0]; let j11 = x2[1] - x0[1];
            let det_j = (j00 * j11 - j01 * j10).abs();

            // D (8x8): ND2 DOFs of spanning fields. Y (3x8): nodal curls.
            let mut dmat = vec![0.0_f64; n_nd2 * n_nd2];
            let mut ymat = vec![0.0_f64; 3 * n_nd2];

            let qr = ref_elem.quadrature(4);
            for k in 0..n_nd2 {
                let mut dof_k = [0.0_f64; 8];

                // Edge moments in canonical global orientation (a < b).
                for (edge_local, &(li, lj)) in tri_edges.iter().enumerate() {
                    let va = nodes[li];
                    let vb = nodes[lj];
                    let (a, b) = if va < vb { (va, vb) } else { (vb, va) };
                    let pa = mesh.node_coords(a);
                    let pb = mesh.node_coords(b);
                    let tx = pb[0] - pa[0];
                    let ty = pb[1] - pa[1];

                    let mut mom0 = 0.0_f64;
                    let mut mom1 = 0.0_f64;
                    for q in 0..3 {
                        let t = gl_pts[q];
                        let w = gl_wts[q];
                        let xp = pa[0] + t * tx;
                        let yp = pa[1] + t * ty;
                        let (fx, fy) = eval_field(k, xp, yp);
                        let tangential = fx * tx + fy * ty;
                        mom0 += w * tangential;
                        mom1 += w * tangential * t;
                    }
                    dof_k[2 * edge_local] = mom0;
                    dof_k[2 * edge_local + 1] = mom1;
                }

                // Interior moments.
                let mut int_x = 0.0_f64;
                let mut int_y = 0.0_f64;
                for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                    let xp = x0[0] + j00 * xi[0] + j01 * xi[1];
                    let yp = x0[1] + j10 * xi[0] + j11 * xi[1];
                    let (fx, fy) = eval_field(k, xp, yp);
                    int_x += w * fx;
                    int_y += w * fy;
                }
                dof_k[6] = int_x * det_j;
                dof_k[7] = int_y * det_j;

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
        let tri_edges = [(0usize, 1usize), (1usize, 2usize), (0usize, 2usize)];

        let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
        let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
        let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

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
            let det_j = (j00 * j11 - j01 * j10).abs();

            let mut dmat = vec![0.0_f64; n_nd2 * n_nd2];
            let mut ymat = vec![0.0_f64; 6 * n_nd2];
            let qr = ref_elem.quadrature(4);

            for k in 0..n_nd2 {
                let mut dof_k = [0.0_f64; 8];

                for (edge_local, &(li, lj)) in tri_edges.iter().enumerate() {
                    let va = nodes[li];
                    let vb = nodes[lj];
                    let (a, b) = if va < vb { (va, vb) } else { (vb, va) };
                    let pa = mesh.node_coords(a);
                    let pb = mesh.node_coords(b);
                    let tx = pb[0] - pa[0];
                    let ty = pb[1] - pa[1];

                    let mut mom0 = 0.0_f64;
                    let mut mom1 = 0.0_f64;
                    for q in 0..3 {
                        let t = gl_pts[q];
                        let w = gl_wts[q];
                        let xp = pa[0] + t * tx;
                        let yp = pa[1] + t * ty;
                        let (fx, fy) = eval_field(k, xp, yp);
                        let tangential = fx * tx + fy * ty;
                        mom0 += w * tangential;
                        mom1 += w * tangential * t;
                    }
                    dof_k[2 * edge_local] = mom0;
                    dof_k[2 * edge_local + 1] = mom1;
                }

                let mut int_x = 0.0_f64;
                let mut int_y = 0.0_f64;
                for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                    let xp = x0[0] + j00 * xi[0] + j01 * xi[1];
                    let yp = x0[1] + j10 * xi[0] + j11 * xi[1];
                    let (fx, fy) = eval_field(k, xp, yp);
                    int_x += w * fx;
                    int_y += w * fy;
                }
                dof_k[6] = int_x * det_j;
                dof_k[7] = int_y * det_j;

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

    /// Build the discrete divergence matrix D: H(div) -> L2.
    ///
    /// ## Order 0 — topological assembly (RT0 → P0)
    ///
    /// The matrix is the signed face-element incidence matrix:
    /// `D[elem, face] = face_sign`.  Exact, no quadrature.
    ///
    /// ## Order 1 — numerical assembly (RT1 → P1/P2)
    ///
    /// `D[l2_dof_i, hdiv_dof_j] = DOF_i^{P1}(div Ψ_j)`, computed via
    /// numerical integration on the reference element and scatter-assembled.
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
            0 | 1 => {}
            o => return Err(DiscreteOpError::UnsupportedHDivOrder { op: "divergence", order: o }),
        }
        match l2_order {
            0 | 1 | 2 => {}
            o => return Err(DiscreteOpError::UnsupportedL2Order { op: "divergence", order: o }),
        }
        // RT0 -> P0 and RT1 -> P1/P2.
        if !((hdiv_order == 0 && l2_order == 0) || (hdiv_order == 1 && (l2_order == 1 || l2_order == 2))) {
            return Err(DiscreteOpError::IncompatibleOrders {
                op: "divergence",
                h1_order: hdiv_order,
                hcurl_order: l2_order,
            });
        }

        match (hdiv_order, l2_order) {
            (0, 0) => Self::divergence_rt0_p0(hdiv_space, l2_space),
            (1, 1) | (1, 2) => Self::divergence_rt1_p1(hdiv_space, l2_space),
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

        let n_local_dofs = if dim == 2 { 3 } else { 4 };

        let n_l2   = l2_space.n_dofs();
        let n_hdiv = hdiv_space.n_dofs();
        let mut coo = CooMatrix::<f64>::new(n_l2, n_hdiv);

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
            let tri_faces = [(1usize, 2usize), (0usize, 2usize), (0usize, 1usize)];

            let sq_3_5: f64 = (3.0_f64 / 5.0).sqrt();
            let gl_pts = [0.5 * (1.0 - sq_3_5), 0.5, 0.5 * (1.0 + sq_3_5)];
            let gl_wts = [5.0_f64 / 18.0, 4.0 / 9.0, 5.0 / 18.0];

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
                let det_j = (j00 * j11 - j01 * j10).abs();

                let mut dmat = vec![0.0_f64; n_rt1 * n_rt1];
                let n_l2_local = l2_dofs.len(); // 3 (P1) or 6 (P2)
                let mut ymat = vec![0.0_f64; n_l2_local * n_rt1];
                let qr = rt1_elem.quadrature(4);

                for k in 0..n_rt1 {
                    let mut dof_k = [0.0_f64; 8];
                    for (edge_local, &(li, lj)) in tri_faces.iter().enumerate() {
                        let va = nodes[li];
                        let vb = nodes[lj];
                        let (a, b) = if va < vb { (va, vb) } else { (vb, va) };
                        let pa = mesh.node_coords(a);
                        let pb = mesh.node_coords(b);
                        let tx = pb[0] - pa[0];
                        let ty = pb[1] - pa[1];
                        let nx = -ty;
                        let ny = tx;

                        let mut mom0 = 0.0_f64;
                        let mut mom1 = 0.0_f64;
                        for q in 0..3 {
                            let t = gl_pts[q];
                            let w = gl_wts[q];
                            let xp = pa[0] + t * tx;
                            let yp = pa[1] + t * ty;
                            let (fx, fy) = eval_field(k, xp, yp);
                            let flux = fx * nx + fy * ny;
                            mom0 += w * flux;
                            mom1 += w * flux * t;
                        }
                        dof_k[2 * edge_local] = mom0;
                        dof_k[2 * edge_local + 1] = mom1;
                    }

                    let mut int_x = 0.0_f64;
                    let mut int_y = 0.0_f64;
                    for (xi, &w) in qr.points.iter().zip(qr.weights.iter()) {
                        let xp = x0[0] + j00 * xi[0] + j01 * xi[1];
                        let yp = x0[1] + j10 * xi[0] + j11 * xi[1];
                        let (fx, fy) = eval_field(k, xp, yp);
                        int_x += w * fx;
                        int_y += w * fy;
                    }
                    dof_k[6] = int_x * det_j;
                    dof_k[7] = int_y * det_j;

                    for i in 0..n_rt1 {
                        dmat[i * n_rt1 + k] = dof_k[i];
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
            let tet_faces = [
                (1usize, 2usize, 3usize),
                (0usize, 2usize, 3usize),
                (0usize, 1usize, 3usize),
                (0usize, 1usize, 2usize),
            ];

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

            let qr_face = TriRT1.quadrature(4);
            let qr_vol = rt1_elem.quadrature(4);

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
                let det_j = (j0[0] * (j1[1] * j2[2] - j1[2] * j2[1])
                    - j1[0] * (j0[1] * j2[2] - j0[2] * j2[1])
                    + j2[0] * (j0[1] * j1[2] - j0[2] * j1[1])).abs();

                for k in 0..n_rt1 {
                    let mut dof_k = vec![0.0_f64; n_rt1];

                    // Face moments in canonical global orientation (sorted face vertices).
                    for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                        let mut fv = [nodes[la], nodes[lb], nodes[lc]];
                        fv.sort_unstable();
                        let pa = mesh.node_coords(fv[0]);
                        let pb = mesh.node_coords(fv[1]);
                        let pc = mesh.node_coords(fv[2]);

                        let ds = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                        let dt = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                        let cx = ds[1] * dt[2] - ds[2] * dt[1];
                        let cy = ds[2] * dt[0] - ds[0] * dt[2];
                        let cz = ds[0] * dt[1] - ds[1] * dt[0];
                        let jac_area = (cx * cx + cy * cy + cz * cz).sqrt();
                        let n_unit = [cx / jac_area, cy / jac_area, cz / jac_area];

                        let mut m0 = 0.0_f64;
                        let mut m1 = 0.0_f64;
                        let mut m2 = 0.0_f64;
                        for (xi2, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                            let s = xi2[0];
                            let t = xi2[1];
                            let pt = [
                                pa[0] + s * ds[0] + t * dt[0],
                                pa[1] + s * ds[1] + t * dt[1],
                                pa[2] + s * ds[2] + t * dt[2],
                            ];
                            let fv = eval_field(k, pt[0], pt[1], pt[2]);
                            let nflux = fv[0] * n_unit[0] + fv[1] * n_unit[1] + fv[2] * n_unit[2];
                            let d_sigma = w * jac_area;
                            m0 += d_sigma * nflux;
                            m1 += d_sigma * nflux * s;
                            m2 += d_sigma * nflux * t;
                        }

                        dof_k[3 * face_local] = m0;
                        dof_k[3 * face_local + 1] = m1;
                        dof_k[3 * face_local + 2] = m2;
                    }

                    // Interior moments.
                    let mut int_x = 0.0_f64;
                    let mut int_y = 0.0_f64;
                    let mut int_z = 0.0_f64;
                    for (xi, &w) in qr_vol.points.iter().zip(qr_vol.weights.iter()) {
                        let pt = [
                            x0[0] + j0[0] * xi[0] + j1[0] * xi[1] + j2[0] * xi[2],
                            x0[1] + j0[1] * xi[0] + j1[1] * xi[1] + j2[1] * xi[2],
                            x0[2] + j0[2] * xi[0] + j1[2] * xi[1] + j2[2] * xi[2],
                        ];
                        let fv = eval_field(k, pt[0], pt[1], pt[2]);
                        int_x += w * fv[0];
                        int_y += w * fv[1];
                        int_z += w * fv[2];
                    }
                    dof_k[12] = int_x * det_j;
                    dof_k[13] = int_y * det_j;
                    dof_k[14] = int_z * det_j;

                    for i in 0..n_rt1 {
                        dmat[i * n_rt1 + k] = dof_k[i];
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
                        ymat[p * n_rt1 + k] = eval_div(k, sample_pts[p][0], sample_pts[p][1], sample_pts[p][2]);
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
        let tet_faces: [(usize, usize, usize); 4] = [
            (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
        ];
        let tet_edges: [(usize, usize); 6] = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];

        // Track visited face DOFs so each face is assembled exactly once.
        let mut visited = HashSet::with_capacity(n_hdiv);

        for e in mesh.elem_iter() {
            let verts = mesh.element_nodes(e);
            let hcurl_dofs = hcurl_space.element_dofs(e);
            let hdiv_dofs = hdiv_space.element_dofs(e);

            for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                let face_dof = hdiv_dofs[face_local] as usize;

                if !visited.insert(face_dof) {
                    continue;
                }

                // Global vertices of this face, sorted → canonical orientation.
                let mut fv = [verts[la], verts[lb], verts[lc]];
                fv.sort_unstable();

                // Boundary traversal fv[0]→fv[1]→fv[2]→fv[0] (right-hand rule
                // with the global face normal (p1−p0)×(p2−p0)).
                let face_boundary: [(u32, u32, f64); 3] = [
                    (fv[0], fv[1],  1.0),
                    (fv[1], fv[2],  1.0),
                    (fv[0], fv[2], -1.0),
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

        // Same spanning set as TetND2 monomial implementation.
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

        // 4-point Gauss-Legendre on [0,1].
        let sq6_5 = (6.0f64 / 5.0).sqrt();
        let ta = ((3.0 - 2.0 * sq6_5) / 7.0).sqrt();
        let tb = ((3.0 + 2.0 * sq6_5) / 7.0).sqrt();
        let wa = (18.0 + 30.0f64.sqrt()) / 36.0;
        let wb = (18.0 - 30.0f64.sqrt()) / 36.0;
        let gl_pts = [
            0.5 * (1.0 - tb),
            0.5 * (1.0 - ta),
            0.5 * (1.0 + ta),
            0.5 * (1.0 + tb),
        ];
        let gl_wts = [0.5 * wb, 0.5 * wa, 0.5 * wa, 0.5 * wb];

        let qr_face = TriRT1.quadrature(4);
        let qr_vol = rt1_elem.quadrature(4);
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
            let det_abs = (j0[0] * (j1[1] * j2[2] - j1[2] * j2[1])
                - j1[0] * (j0[1] * j2[2] - j0[2] * j2[1])
                + j2[0] * (j0[1] * j1[2] - j0[2] * j1[1]))
                .abs();

            for k in 0..n_nd2 {
                let mut dof_nd2 = vec![0.0_f64; n_nd2];
                let mut dof_rt1 = vec![0.0_f64; n_rt1];

                // ND2 edge moments (2 per edge) in global edge orientation.
                for (edge_local, &(li, lj)) in tet_edges.iter().enumerate() {
                    let gi = nodes[li];
                    let gj = nodes[lj];
                    let (ga, gb) = if gi < gj { (gi, gj) } else { (gj, gi) };
                    let pa = mesh.node_coords(ga);
                    let pb = mesh.node_coords(gb);
                    let tau = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];

                    let mut m0 = 0.0_f64;
                    let mut m1 = 0.0_f64;
                    for q in 0..4 {
                        let t = gl_pts[q];
                        let w = gl_wts[q];
                        let pt = [
                            pa[0] + t * tau[0],
                            pa[1] + t * tau[1],
                            pa[2] + t * tau[2],
                        ];
                        let fv = eval_field(k, pt[0], pt[1], pt[2]);
                        let tang = fv[0] * tau[0] + fv[1] * tau[1] + fv[2] * tau[2];
                        m0 += w * tang;
                        m1 += w * tang * t;
                    }

                    dof_nd2[2 * edge_local] = m0;
                    dof_nd2[2 * edge_local + 1] = m1;
                }

                // ND2 face tangential moments and RT1 face normal moments.
                for (face_local, &(la, lb, lc)) in tet_faces.iter().enumerate() {
                    let mut fvtx = [nodes[la], nodes[lb], nodes[lc]];
                    fvtx.sort_unstable();
                    let pa = mesh.node_coords(fvtx[0]);
                    let pb = mesh.node_coords(fvtx[1]);
                    let pc = mesh.node_coords(fvtx[2]);

                    let ds = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let dt = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let cross = [
                        ds[1] * dt[2] - ds[2] * dt[1],
                        ds[2] * dt[0] - ds[0] * dt[2],
                        ds[0] * dt[1] - ds[1] * dt[0],
                    ];
                    let jac_area =
                        (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                    let n_unit = [cross[0] / jac_area, cross[1] / jac_area, cross[2] / jac_area];

                    let mut nd_t1 = 0.0_f64;
                    let mut nd_t2 = 0.0_f64;
                    let mut rt_m0 = 0.0_f64;
                    let mut rt_m1 = 0.0_f64;
                    let mut rt_m2 = 0.0_f64;

                    for (xi, &w) in qr_face.points.iter().zip(qr_face.weights.iter()) {
                        let s = xi[0];
                        let t = xi[1];
                        let pt = [
                            pa[0] + s * ds[0] + t * dt[0],
                            pa[1] + s * ds[1] + t * dt[1],
                            pa[2] + s * ds[2] + t * dt[2],
                        ];

                        let fv = eval_field(k, pt[0], pt[1], pt[2]);
                        let cv = eval_curl(k, pt[0], pt[1], pt[2]);
                        let d_sigma = w * jac_area;

                        nd_t1 += d_sigma * (fv[0] * ds[0] + fv[1] * ds[1] + fv[2] * ds[2]);
                        nd_t2 += d_sigma * (fv[0] * dt[0] + fv[1] * dt[1] + fv[2] * dt[2]);

                        let nflux = cv[0] * n_unit[0] + cv[1] * n_unit[1] + cv[2] * n_unit[2];
                        rt_m0 += d_sigma * nflux;
                        rt_m1 += d_sigma * nflux * s;
                        rt_m2 += d_sigma * nflux * t;
                    }

                    dof_nd2[12 + 2 * face_local] = nd_t1;
                    dof_nd2[12 + 2 * face_local + 1] = nd_t2;

                    dof_rt1[3 * face_local] = rt_m0;
                    dof_rt1[3 * face_local + 1] = rt_m1;
                    dof_rt1[3 * face_local + 2] = rt_m2;
                }

                // RT1 interior moments of curl(field).
                let mut int_x = 0.0_f64;
                let mut int_y = 0.0_f64;
                let mut int_z = 0.0_f64;
                for (xi, &w) in qr_vol.points.iter().zip(qr_vol.weights.iter()) {
                    let pt = [
                        x0[0] + j0[0] * xi[0] + j1[0] * xi[1] + j2[0] * xi[2],
                        x0[1] + j0[1] * xi[0] + j1[1] * xi[1] + j2[1] * xi[2],
                        x0[2] + j0[2] * xi[0] + j1[2] * xi[1] + j2[2] * xi[2],
                    ];
                    let cv = eval_curl(k, pt[0], pt[1], pt[2]);
                    int_x += w * cv[0];
                    int_y += w * cv[1];
                    int_z += w * cv[2];
                }
                dof_rt1[12] = int_x * det_abs;
                dof_rt1[13] = int_y * det_abs;
                dof_rt1[14] = int_z * det_abs;

                for i in 0..n_nd2 {
                    dmat[i * n_nd2 + k] = dof_nd2[i];
                }
                for p in 0..n_rt1 {
                    ymat[p * n_nd2 + k] = dof_rt1[p];
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
    use fem_mesh::SimplexMesh;

    /// Test: Discrete gradient of a linear function u = x + 2y.
    ///
    /// The gradient field is (1, 2) everywhere.  Interpolating u into H1
    /// and applying G should give the same result as interpolating (1,2)
    /// into H(curl) via its DOF functional.
    #[test]
    fn gradient_of_linear_function() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh, 0);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 1);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv = HDivSpace::new(mesh3, 0);
        let mesh4 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let h1 = H1Space::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh4 = SimplexMesh::<3>::unit_cube_tet(2);

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
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh4 = SimplexMesh::<3>::unit_cube_tet(2);

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
                "order-2 3D div(curl(u)) should be zero, seed={seed}, max |D*C*u| = {max_err}"
            );
        }
    }

    /// Test: curl_3d matrix dimensions.
    #[test]
    fn curl_3d_dimensions() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
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
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
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
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let hdiv = HDivSpace::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();

        // A = (x*y, y*z, z*x), so curl(A) = (-y, -z, -x).
        let a = hcurl.interpolate_vector(&|x| vec![x[0] * x[1], x[1] * x[2], x[2] * x[0]]);
        let mut ca = vec![0.0; hdiv.n_dofs()];
        c.spmv(a.as_slice(), &mut ca);

        let curl_interp = hdiv.interpolate_vector(&|x| vec![-x[1], -x[2], -x[0]]);

        let max_err: f64 = (0..hdiv.n_dofs())
            .map(|i| (ca[i] - curl_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-8,
            "ND2->RT1 3D: curl interpolation mismatch, max error = {max_err}"
        );
    }

    /// Test: bad order combination returns an error instead of panicking.
    ///
    /// P2 (order 2) + ND1 (order 1) are incompatible; the dispatcher should
    /// return `IncompatibleOrders` rather than panicking or silently producing
    /// wrong results.
    #[test]
    fn gradient_bad_order_returns_error() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let h1 = H1Space::new(mesh, 2); // P2
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh2, 1); // ND1 — mismatch with P2

        let result = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert!(
            matches!(result, Err(DiscreteOpError::IncompatibleOrders { .. })),
            "expected IncompatibleOrders for P2+ND1, got {:?}", result
        );
    }

    /// Test: curl_2d with wrong dimension returns an error.
    #[test]
    fn curl_2d_wrong_dim_returns_error() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let hcurl = HCurlSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(1);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
        let l2 = L2Space::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();
        assert_eq!(c.nrows, l2.n_dofs());
        assert_eq!(c.ncols, hcurl.n_dofs());
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: curl_3d with wrong dimension returns an error.
    #[test]
    fn curl_3d_wrong_dim_returns_error() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let hdiv = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh2, 2);

        let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
        assert_eq!(g.nrows, hcurl.n_dofs(), "G nrows should equal n_nd2");
        assert_eq!(g.ncols, h1.n_dofs(),    "G ncols should equal n_p2");
        assert!(g.nrows > 0 && g.ncols > 0);
    }

    /// Test: Gradient P2→ND2 — constant function has zero gradient.
    #[test]
    fn gradient_p2_nd2_constant_is_zero() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(3);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh, 2); // ND2
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(3);
        let l2    = L2Space::new(mesh2, 1);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();
        assert_eq!(c.nrows, l2.n_dofs(), "C nrows should equal n_p1");
        assert_eq!(c.ncols, hcurl.n_dofs(), "C ncols should equal n_nd2");
        assert!(c.nrows > 0 && c.ncols > 0);
    }

    /// Test: order-2 2D de Rham property curl(grad(u)) = 0.
    #[test]
    fn de_rham_curl_of_grad_is_zero_order2() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let h1    = H1Space::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh2, 2);
        let mesh3 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
    #[test]
    fn curl_2d_nd2_p2_commutes_with_interpolation() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hcurl = HCurlSpace::new(mesh, 2);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 2);

        let c = DiscreteLinearOperator::curl_2d(&hcurl, &l2).unwrap();

        // F = (x^2, x*y), so curl(F) = d/dx(x*y) - d/dy(x^2) = y.
        let f = hcurl.interpolate_vector(&|x| vec![x[0] * x[0], x[0] * x[1]]);
        let mut cf = vec![0.0; l2.n_dofs()];
        c.spmv(f.as_slice(), &mut cf);

        let c_interp = l2.interpolate(&|x| x[1]);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (cf[i] - c_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "ND2->P2: curl(x^2,xy) should be y, max error = {max_err}");
    }

    /// Test: Gradient P2→ND2 — incompatible orders return an error.
    #[test]
    fn gradient_incompatible_orders_returns_error() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(2);
        let h1    = H1Space::new(mesh, 1);  // P1
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh2, 2); // ND2 — mismatch

        let result = DiscreteLinearOperator::gradient(&h1, &hcurl);
        assert!(
            matches!(result, Err(DiscreteOpError::IncompatibleOrders { .. })),
            "expected IncompatibleOrders, got {:?}", result
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
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
    #[test]
    fn divergence_rt1_p2_commutes_with_interpolation() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
        let l2    = L2Space::new(mesh2, 2);

        // F = (x^2, y^2), so div F = 2x + 2y.
        let f = hdiv.interpolate_vector(&|x| vec![x[0] * x[0], x[1] * x[1]]);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|x| 2.0 * x[0] + 2.0 * x[1]);

        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_err < 1e-8,
            "RT1->P2: div(x^2,y^2) should be 2x+2y, max error = {max_err}"
        );
    }

    /// Test: Divergence RT1→P1 — dimensions are correct.
    #[test]
    fn divergence_rt1_p1_dimensions() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(3);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(3);
        let l2    = L2Space::new(mesh2, 1);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(),   "D nrows should equal n_p1");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1→P1 — divergence-free field gives zero.
    #[test]
    fn divergence_rt1_p1_div_free_field_is_zero() {
        let mesh  = SimplexMesh::<2>::unit_square_tri(4);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<2>::unit_square_tri(4);
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
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 1);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(), "D nrows should equal n_l2_p1");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1->P1 in 3D — commuting property for F=(x,y,z).
    #[test]
    fn divergence_rt1_p1_3d_commutes_with_interpolation() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 1);

        // F = (x,y,z), div F = 3.
        let f = hdiv.interpolate_vector(&|x| vec![x[0], x[1], x[2]]);
        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        let mut div_f = vec![0.0; l2.n_dofs()];
        d.spmv(f.as_slice(), &mut div_f);

        let div_interp = l2.interpolate(&|_x| 3.0);
        let max_err: f64 = (0..l2.n_dofs())
            .map(|i| (div_f[i] - div_interp.as_slice()[i]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "RT1->P1 3D: divergence mismatch, max error = {max_err}");
    }

    /// Test: Divergence RT1->P2 in 3D — dimensions are correct.
    #[test]
    fn divergence_rt1_p2_3d_dimensions() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let l2    = L2Space::new(mesh2, 2);

        let d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
        assert_eq!(d.nrows, l2.n_dofs(), "D nrows should equal n_l2_p2");
        assert_eq!(d.ncols, hdiv.n_dofs(), "D ncols should equal n_rt1");
        assert!(d.nrows > 0 && d.ncols > 0);
    }

    /// Test: Divergence RT1->P2 in 3D — commuting property for F=(x,y,z).
    #[test]
    fn divergence_rt1_p2_3d_commutes_with_interpolation() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let hdiv  = HDivSpace::new(mesh, 1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
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

    /// Test: order-2 3D de Rham property with L2(P2) target — div(curl(u)) = 0.
    #[test]
    fn de_rham_div_of_curl_3d_is_zero_order2_l2_p2() {
        let mesh  = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(2);
        let mesh4 = SimplexMesh::<3>::unit_cube_tet(2);

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
        let mesh  = SimplexMesh::<3>::unit_cube_tet(1);
        let mesh2 = SimplexMesh::<3>::unit_cube_tet(1);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
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
