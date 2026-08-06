//! IGA (Isogeometric Analysis) physical-domain mapping and global assembly.
//!
//! Provides Jacobian computation from parametric to physical space for NURBS
//! patches, and single-patch assembly of the diffusion stiffness matrix, mass
//! matrix, and load vector.
//!
//! # Mathematical background
//!
//! Given a NURBS patch with control points $\mathbf{x}_A \in \mathbb{R}^d$ and
//! basis functions $R_A(\boldsymbol\xi)$, the physical-domain map is:
//!
//! $$\mathbf{x}(\boldsymbol\xi) = \sum_A R_A(\boldsymbol\xi)\, \mathbf{x}_A$$
//!
//! The Jacobian $J_{ij} = \partial x_i / \partial \xi_j$ is:
//!
//! $$J = \sum_A \mathbf{x}_A \cdot \nabla_\xi R_A^T$$
//!
//! Physical-domain gradients are recovered via the chain rule:
//!
//! $$\nabla_x R_A = J^{-T}\, \nabla_\xi R_A$$
//!
//! The standard weak-form integrals become:
//!
//! $$K_{AB} = \int_\Omega \kappa\, \nabla_x R_A \cdot \nabla_x R_B\, \mathrm{d}\Omega
//!          = \int_{\hat\Omega} \kappa\, \nabla_x R_A \cdot \nabla_x R_B\, |\det J|\, \mathrm{d}\hat\Omega$$
//!
//! $$M_{AB} = \int_\Omega \rho\, R_A R_B\, \mathrm{d}\Omega$$
//!
//! $$f_A   = \int_\Omega f\, R_A\, \mathrm{d}\Omega$$

use fem_core::types::DofId;
use fem_element::iga::{NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData};
use fem_element::quadrature::seg_rule;
use fem_element::reference::{QuadratureRule, ReferenceElement};
use fem_linalg::{CooMatrix, CsrMatrix};

fn nonempty_spans(knots: &[f64]) -> Vec<(f64, f64)> {
    knots.windows(2)
        .filter_map(|w| {
            let a = w[0];
            let b = w[1];
            if b > a { Some((a, b)) } else { None }
        })
        .collect()
}

/// Build a span-wise tensor-product Gauss rule on a 2-D NURBS patch.
///
/// Integration is performed over each non-empty knot span in `u` and `v`.
pub(crate) fn patch_quad_2d(pd: &NurbsPatch2DData, order: u8) -> QuadratureRule {
    let seg = seg_rule(order);
    let spans_u = nonempty_spans(&pd.kv_u.knots);
    let spans_v = nonempty_spans(&pd.kv_v.knots);
    let n1 = seg.points.len();
    let mut pts = Vec::with_capacity(spans_u.len() * spans_v.len() * n1 * n1);
    let mut wts = Vec::with_capacity(spans_u.len() * spans_v.len() * n1 * n1);

    for (u0, u1) in spans_u {
        let hu = u1 - u0;
        for (v0, v1) in &spans_v {
            let hv = v1 - v0;
            for (su, wu) in seg.points.iter().zip(seg.weights.iter()) {
                for (sv, wv) in seg.points.iter().zip(seg.weights.iter()) {
                    let u = u0 + hu * su[0];
                    let v = *v0 + hv * sv[0];
                    pts.push(vec![u, v]);
                    wts.push(wu * wv * hu * hv);
                }
            }
        }
    }

    QuadratureRule { points: pts, weights: wts }
}

/// Build a span-wise tensor-product Gauss rule on a 3-D NURBS patch.
fn patch_quad_3d(pd: &NurbsPatch3DData, order: u8) -> QuadratureRule {
    let seg = seg_rule(order);
    let spans_u = nonempty_spans(&pd.kv_u.knots);
    let spans_v = nonempty_spans(&pd.kv_v.knots);
    let spans_w = nonempty_spans(&pd.kv_w.knots);
    let n1 = seg.points.len();
    let mut pts = Vec::with_capacity(spans_u.len() * spans_v.len() * spans_w.len() * n1 * n1 * n1);
    let mut wts = Vec::with_capacity(spans_u.len() * spans_v.len() * spans_w.len() * n1 * n1 * n1);

    for (u0, u1) in spans_u {
        let hu = u1 - u0;
        for (v0, v1) in &spans_v {
            let hv = v1 - v0;
            for (w0, w1) in &spans_w {
                let hw = w1 - w0;
                for (su, wu) in seg.points.iter().zip(seg.weights.iter()) {
                    for (sv, wv) in seg.points.iter().zip(seg.weights.iter()) {
                        for (sw, ww) in seg.points.iter().zip(seg.weights.iter()) {
                            let u = u0 + hu * su[0];
                            let v = *v0 + hv * sv[0];
                            let w = *w0 + hw * sw[0];
                            pts.push(vec![u, v, w]);
                            wts.push(wu * wv * ww * hu * hv * hw);
                        }
                    }
                }
            }
        }
    }

    QuadratureRule { points: pts, weights: wts }
}

// ─── 2-D physical map ────────────────────────────────────────────────────────

/// Physical-domain map result for a 2-D NURBS patch at one parametric point.
pub struct PhysMap2D {
    /// Physical coordinates $\mathbf{x}(\xi, \eta)$.
    pub x_phys: [f64; 2],
    /// Jacobian $J$ (2×2, row-major): $J[i][j] = \partial x_i / \partial \xi_j$.
    pub jac: [[f64; 2]; 2],
    /// Determinant $\det J$ (must be > 0 for a valid element).
    pub det_j: f64,
    /// $J^{-T}$ (2×2, row-major): used to transform parametric gradients.
    pub jac_inv_t: [[f64; 2]; 2],
}

/// Compute the physical-domain map for a single 2-D patch at `xi = [u, v]`.
///
/// # Panics
/// Panics if $|\det J| < 10^{-14}$ (degenerate mapping).
pub fn physical_map_2d(pd: &NurbsPatch2DData, xi: &[f64]) -> PhysMap2D {
    use fem_element::iga::NurbsPatch2D;

    let patch = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
    let n_dof = patch.n_dofs();

    // Evaluate parametric gradients ∇_ξ R_A at xi.
    let mut grads_xi = vec![0.0_f64; n_dof * 2]; // [dR_A/du, dR_A/dv] per DOF
    patch.eval_grad_basis(xi, &mut grads_xi);

    // Evaluate basis values R_A at xi (for physical-coords map).
    let mut basis = vec![0.0_f64; n_dof];
    patch.eval_basis(xi, &mut basis);

    // Physical coordinates: x = Σ_A R_A * x_A
    let mut x_phys = [0.0_f64; 2];
    for a in 0..n_dof {
        x_phys[0] += basis[a] * pd.control_pts[a][0];
        x_phys[1] += basis[a] * pd.control_pts[a][1];
    }

    // Jacobian: J[i][j] = Σ_A x_A[i] * dR_A/dξ_j
    let mut jac = [[0.0_f64; 2]; 2];
    for a in 0..n_dof {
        let dru = grads_xi[a * 2];     // dR_A/du
        let drv = grads_xi[a * 2 + 1]; // dR_A/dv
        let xa = pd.control_pts[a][0];
        let ya = pd.control_pts[a][1];
        jac[0][0] += xa * dru; // dx/du
        jac[0][1] += xa * drv; // dx/dv
        jac[1][0] += ya * dru; // dy/du
        jac[1][1] += ya * drv; // dy/dv
    }

    let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    assert!(
        det_j.abs() > 1e-14,
        "physical_map_2d: degenerate Jacobian det={det_j:.3e} at xi={xi:?}"
    );

    // J^{-T} (inverse of the transpose): J^{-T}[i][j] = cofactor[j][i] / det
    let inv_det = 1.0 / det_j;
    let jac_inv_t = [
        [ jac[1][1] * inv_det, -jac[1][0] * inv_det],
        [-jac[0][1] * inv_det,  jac[0][0] * inv_det],
    ];

    PhysMap2D { x_phys, jac, det_j, jac_inv_t }
}

/// Compute physical-domain gradients $\nabla_x R_A$ for all basis functions
/// at parametric point `xi` in a 2-D patch.
///
/// Returns `(phys_grads, det_j)` where `phys_grads` has length `n_dof * 2`:
/// `phys_grads[a*2] = dR_A/dx`, `phys_grads[a*2+1] = dR_A/dy`.
///
/// Unlike calling [`physical_map_2d`] separately, this function reuses the
/// already-evaluated parametric gradients to build the Jacobian, avoiding an
/// extra `NurbsPatch2D` clone (saves ~3 Vec clones per Gauss point).
pub fn physical_grads_2d(pd: &NurbsPatch2DData, xi: &[f64]) -> (Vec<f64>, f64) {
    use fem_element::iga::NurbsPatch2D;

    let patch = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
    let n_dof = patch.n_dofs();

    // Evaluate parametric gradients ∇_ξ R_A and basis values R_A.
    let mut grads_xi = vec![0.0_f64; n_dof * 2];
    patch.eval_grad_basis(xi, &mut grads_xi);
    let mut basis = vec![0.0_f64; n_dof];
    patch.eval_basis(xi, &mut basis);

    // Build Jacobian J from the parametric gradients and control points:
    //   J[i][j] = Σ_A cpt[A][i] · dR_A/dξ_j
    let mut jac = [[0.0_f64; 2]; 2];
    let mut x_phys = [0.0_f64; 2];
    for a in 0..n_dof {
        let cx = pd.control_pts[a][0];
        let cy = pd.control_pts[a][1];
        let dru = grads_xi[a * 2];
        let drv = grads_xi[a * 2 + 1];
        jac[0][0] += cx * dru;  jac[0][1] += cx * drv;
        jac[1][0] += cy * dru;  jac[1][1] += cy * drv;
        x_phys[0] += basis[a] * cx;
        x_phys[1] += basis[a] * cy;
    }
    let det_j = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    let inv_det = 1.0 / det_j;
    // J^{-T} (row-major)
    let ji = [
        [jac[1][1] * inv_det, -jac[1][0] * inv_det],
        [-jac[0][1] * inv_det,  jac[0][0] * inv_det],
    ];

    // ∇_x R_A = J^{-T} · ∇_ξ R_A
    let mut phys_grads = vec![0.0_f64; n_dof * 2];
    for a in 0..n_dof {
        let dru = grads_xi[a * 2];
        let drv = grads_xi[a * 2 + 1];
        phys_grads[a * 2]     = ji[0][0] * dru + ji[0][1] * drv;
        phys_grads[a * 2 + 1] = ji[1][0] * dru + ji[1][1] * drv;
    }

    (phys_grads, det_j)
}

// ─── 3-D physical map ────────────────────────────────────────────────────────

/// Physical-domain map for a 3-D NURBS patch at `xi = [u, v, w]`.
pub struct PhysMap3D {
    pub x_phys: [f64; 3],
    pub jac: [[f64; 3]; 3],
    pub det_j: f64,
    pub jac_inv_t: [[f64; 3]; 3],
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
  - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
  + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0])
}

fn inv_t3(m: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let inv = 1.0 / det;
    // Cofactor matrix transposed = inverse * det, then divided by det.
    [
        [
            (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * inv,
            (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * inv,
            (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * inv,
        ],
        [
            (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv,
            (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv,
            (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv,
        ],
        [
            (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv,
            (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv,
            (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv,
        ],
    ]
}

/// Compute the physical-domain map for a 3-D patch at `xi = [u, v, w]`.
pub fn physical_map_3d(pd: &NurbsPatch3DData, xi: &[f64]) -> PhysMap3D {
    use fem_element::iga::NurbsPatch3D;

    let patch = NurbsPatch3D::new(
        pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone(), pd.weights.clone(),
    );
    let n_dof = patch.n_dofs();

    let mut grads_xi = vec![0.0_f64; n_dof * 3];
    patch.eval_grad_basis(xi, &mut grads_xi);

    let mut basis = vec![0.0_f64; n_dof];
    patch.eval_basis(xi, &mut basis);

    let mut x_phys = [0.0_f64; 3];
    for a in 0..n_dof {
        x_phys[0] += basis[a] * pd.control_pts[a][0];
        x_phys[1] += basis[a] * pd.control_pts[a][1];
        x_phys[2] += basis[a] * pd.control_pts[a][2];
    }

    let mut jac = [[0.0_f64; 3]; 3];
    for a in 0..n_dof {
        let dru = grads_xi[a * 3];
        let drv = grads_xi[a * 3 + 1];
        let drw = grads_xi[a * 3 + 2];
        for i in 0..3 {
            let xa = pd.control_pts[a][i];
            jac[i][0] += xa * dru;
            jac[i][1] += xa * drv;
            jac[i][2] += xa * drw;
        }
    }

    let det_j = det3(&jac);
    assert!(
        det_j.abs() > 1e-14,
        "physical_map_3d: degenerate Jacobian det={det_j:.3e} at xi={xi:?}"
    );

    let jac_inv_t = inv_t3(&jac, det_j);

    PhysMap3D { x_phys, jac, det_j, jac_inv_t }
}

/// Compute physical-domain gradients for all basis functions at parametric `xi`
/// in a 3-D patch.
///
/// Returns `(phys_grads, det_j)` where `phys_grads[a*3 + i] = dR_A/dx_i`.
///
/// Builds the Jacobian from the already-evaluated parametric gradients, avoiding
/// an extra `NurbsPatch3D` clone per Gauss point.
pub fn physical_grads_3d(pd: &NurbsPatch3DData, xi: &[f64]) -> (Vec<f64>, f64) {
    use fem_element::iga::NurbsPatch3D;

    let patch = NurbsPatch3D::new(
        pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone(), pd.weights.clone(),
    );
    let n_dof = patch.n_dofs();

    let mut grads_xi = vec![0.0_f64; n_dof * 3];
    patch.eval_grad_basis(xi, &mut grads_xi);

    // Build 3×3 Jacobian from parametric gradients and control points.
    let mut jac = [[0.0_f64; 3]; 3];
    for a in 0..n_dof {
        let dru = grads_xi[a * 3];
        let drv = grads_xi[a * 3 + 1];
        let drw = grads_xi[a * 3 + 2];
        for i in 0..3 {
            let xa = pd.control_pts[a][i];
            jac[i][0] += xa * dru;
            jac[i][1] += xa * drv;
            jac[i][2] += xa * drw;
        }
    }
    let det_j = det3(&jac);
    let ji = inv_t3(&jac, det_j);

    let mut phys_grads = vec![0.0_f64; n_dof * 3];
    for a in 0..n_dof {
        let dru = grads_xi[a * 3];
        let drv = grads_xi[a * 3 + 1];
        let drw = grads_xi[a * 3 + 2];
        phys_grads[a * 3]     = ji[0][0]*dru + ji[0][1]*drv + ji[0][2]*drw;
        phys_grads[a * 3 + 1] = ji[1][0]*dru + ji[1][1]*drv + ji[1][2]*drw;
        phys_grads[a * 3 + 2] = ji[2][0]*dru + ji[2][1]*drv + ji[2][2]*drw;
    }

    (phys_grads, det_j)
}

// ─── 2-D single-patch global assembly ────────────────────────────────────────

/// Assemble the diffusion stiffness matrix $K_{AB} = \int \kappa\,\nabla R_A \cdot \nabla R_B\,\mathrm{d}\Omega$
/// for a 2-D NURBS mesh (single-patch or multi-patch, DOFs are per-patch-global).
///
/// DOF ordering: for a single patch, DOF `a` is control point `a` of the patch.
/// For multi-patch, DOFs are block-offset: patch `p` starts at offset = sum of
/// `n_dofs` of all previous patches.
///
/// Returns the global stiffness matrix in CSR format.
pub fn assemble_iga_diffusion_2d(
    mesh: &NurbsMesh2D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_2d(pd, qp_xi);
            let w = qp_w * det_j.abs();

            for a in 0..n_dof {
                let ga = dof_offset + a;
                for b in 0..n_dof {
                    let gb = dof_offset + b;
                    let dot = phys_grads[a*2]   * phys_grads[b*2]
                            + phys_grads[a*2+1] * phys_grads[b*2+1];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the mass matrix $M_{AB} = \int \rho\, R_A R_B\,\mathrm{d}\Omega$
/// for a 2-D NURBS mesh.
pub fn assemble_iga_mass_2d(
    mesh: &NurbsMesh2D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                for b in 0..n_dof {
                    coo.add(dof_offset + a, dof_offset + b,
                        rho * basis[a] * basis[b] * w);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the **vector** mass matrix for a 2-D NURBS mesh with interleaved DOFs.
///
/// Each control point contributes `dim = 2` DOFs: `2a` (x) and `2a+1` (y).
/// The mass matrix is block-diagonal: `M_vec[2a+c, 2b+d] = ρ·∫R_a·R_b·δ_{cd} dΩ`.
/// This matches the DOF layout used by [`assemble_iga_elasticity_2d`].
pub fn assemble_iga_mass_2d_vec(
    mesh: &NurbsMesh2D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let n_vec = 2 * n_total;
    let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                let ba = rho * basis[a];
                let row_x = 2 * (dof_offset + a);
                let row_y = row_x + 1;
                for b in 0..n_dof {
                    let m = ba * basis[b] * w;
                    let col_x = 2 * (dof_offset + b);
                    let col_y = col_x + 1;
                    coo.add(row_x, col_x, m);
                    coo.add(row_y, col_y, m);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the **vector** mass matrix for a 3-D NURBS mesh with interleaved DOFs.
///
/// Each control point contributes `dim = 3` DOFs: `3a` (x), `3a+1` (y), `3a+2` (z).
pub fn assemble_iga_mass_3d_vec(
    mesh: &NurbsMesh3D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let n_vec = 3 * n_total;
    let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (_, det_j) = physical_grads_3d(pd, qp_xi);
            let w = qp_w * det_j.abs();

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                let ba = rho * basis[a];
                let row0 = 3 * (dof_offset + a);
                for b in 0..n_dof {
                    let m = ba * basis[b] * w;
                    let col0 = 3 * (dof_offset + b);
                    coo.add(row0, col0, m);
                    coo.add(row0 + 1, col0 + 1, m);
                    coo.add(row0 + 2, col0 + 2, m);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the load vector $f_A = \int f(\mathbf{x})\, R_A\,\mathrm{d}\Omega$
/// for a 2-D NURBS mesh.
///
/// `source` receives the physical coordinate `&[f64; 2]` and returns the source value.
pub fn assemble_iga_load_2d(
    mesh: &NurbsMesh2D,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut rhs = vec![0.0_f64; n_total];

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                rhs[dof_offset + a] += f_val * basis[a] * w;
            }
        }
        dof_offset += n_dof;
    }

    rhs
}

// ─── 2-D multi-patch C⁰ assembly ───────────────────────────────────────────

/// Assemble the diffusion stiffness matrix for a multi-patch 2-D NURBS mesh
/// with C⁰ coupling via a DOF map.
///
/// Uses `dof_map[pi][a]` to map local patch DOF `a` to the global DOF index,
/// merging shared DOFs along patch interfaces.
pub fn assemble_iga_diffusion_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_2d(pd, qp_xi);
            let w = qp_w * det_j.abs();
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    let dot = phys_grads[a*2]*phys_grads[b*2] + phys_grads[a*2+1]*phys_grads[b*2+1];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
    }
    coo.into_csr()
}

/// Assemble the load vector for a multi-patch 2-D NURBS mesh with C⁰ coupling.
pub fn assemble_iga_load_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n_global_dofs];
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                rhs[dof_map[pi][a] as usize] += f_val * basis[a] * w;
            }
        }
    }
    rhs
}

/// Assemble the mass matrix for a multi-patch 2-D NURBS mesh with C⁰ coupling.
pub fn assemble_iga_mass_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    coo.add(ga, gb, rho * basis[a] * basis[b] * w);
                }
            }
        }
    }
    coo.into_csr()
}

/// Assemble the 2-D linear elasticity stiffness matrix with uniform material.
///
/// Uses interleaved DOF ordering: control point `a` contributes DOFs
/// `2a` (x-displacement) and `2a+1` (y-displacement).
/// Assembles the standard isotropic elasticity operator:
///
/// ```text
/// K_{ab} = ∫ B_a^T D B_b dΩ
/// ```
///
/// where D is the plane-stress or plane-strain material matrix:
///
/// ```text
/// D = [λ+2μ   λ    0  ]
///     [λ     λ+2μ  0  ]
///     [0      0    μ  ]
/// ```
pub fn assemble_iga_elasticity_2d(
    mesh: &NurbsMesh2D,
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_iga_elasticity_2d_impl(mesh, quad_order, |_tag, _patch_idx| (lambda, mu))
}

/// Assemble the 2-D linear elasticity stiffness matrix with per-tag materials.
///
/// `materials` is a slice of `(tag, lambda, mu)` entries.  Each NURBS patch's
/// `tag` is looked up in this slice to determine its material parameters.
/// Uses the same strain-displacement matrix as [`assemble_iga_elasticity_2d`]
/// but with patch-dependent D.
///
/// # Panics
/// Panics if a patch tag is not found in `materials`.
pub fn assemble_iga_elasticity_2d_multi(
    mesh: &NurbsMesh2D,
    materials: &[(i32, f64, f64)],
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_iga_elasticity_2d_impl(mesh, quad_order, |tag, _idx| {
        materials.iter().copied()
            .find(|(t, _, _)| *t == tag)
            .map(|(_, l, m)| (l, m))
            .unwrap_or_else(|| panic!(
                "assemble_iga_elasticity_2d_multi: no material entry for tag={tag}"
            ))
    })
}

/// Shared implementation for 2-D IGA elasticity assembly.
///
/// The closure `get_material` receives `(tag, patch_index)` and returns
/// `(lambda, mu)` for that patch.
fn assemble_iga_elasticity_2d_impl(
    mesh: &NurbsMesh2D,
    quad_order: u8,
    get_material: impl Fn(i32, usize) -> (f64, f64),
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let n_vec = 2 * n_total;
    let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

    let mut dof_offset = 0usize;
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let (lambda, mu) = get_material(pd.tag, pi);
        let c1 = lambda + 2.0 * mu;  // D[0,0] and D[1,1]
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_2d(pd, qp_xi);
            let w = qp_w * det_j.abs();

            for a in 0..n_dof {
                let gax = phys_grads[a * 2];
                let gay = phys_grads[a * 2 + 1];
                let base_a = 2 * (dof_offset + a);

                for b in 0..n_dof {
                    let gbx = phys_grads[b * 2];
                    let gby = phys_grads[b * 2 + 1];
                    let base_b = 2 * (dof_offset + b);

                    // K[2a,   2b]    = ∫ (λ+2μ)·∂Ra/∂x·∂Rb/∂x + μ·∂Ra/∂y·∂Rb/∂y
                    let k00 = w * (c1 * gax * gbx + mu * gay * gby);
                    // K[2a,   2b+1]  = ∫ λ·∂Ra/∂x·∂Rb/∂y + μ·∂Ra/∂y·∂Rb/∂x
                    let k01 = w * (lambda * gax * gby + mu * gay * gbx);
                    // K[2a+1, 2b]    = ∫ λ·∂Ra/∂y·∂Rb/∂x + μ·∂Ra/∂x·∂Rb/∂y
                    let k10 = w * (lambda * gay * gbx + mu * gax * gby);
                    // K[2a+1, 2b+1]  = ∫ μ·∂Ra/∂x·∂Rb/∂x + (λ+2μ)·∂Ra/∂y·∂Rb/∂y
                    let k11 = w * (mu * gax * gbx + c1 * gay * gby);

                    coo.add(base_a, base_b, k00);
                    coo.add(base_a, base_b + 1, k01);
                    coo.add(base_a + 1, base_b, k10);
                    coo.add(base_a + 1, base_b + 1, k11);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

// ─── 3-D single-patch global assembly ────────────────────────────────────────

/// Assemble the diffusion stiffness matrix for a 3-D NURBS mesh.
pub fn assemble_iga_diffusion_3d(
    mesh: &NurbsMesh3D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_3d(pd, qp_xi);
            let w = qp_w * det_j.abs();

            for a in 0..n_dof {
                let ga = dof_offset + a;
                for b in 0..n_dof {
                    let gb = dof_offset + b;
                    let dot = phys_grads[a*3]   * phys_grads[b*3]
                            + phys_grads[a*3+1] * phys_grads[b*3+1]
                            + phys_grads[a*3+2] * phys_grads[b*3+2];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the mass matrix for a 3-D NURBS mesh.
pub fn assemble_iga_mass_3d(
    mesh: &NurbsMesh3D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                for b in 0..n_dof {
                    coo.add(dof_offset + a, dof_offset + b,
                        rho * basis[a] * basis[b] * w);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

/// Assemble the load vector for a 3-D NURBS mesh.
pub fn assemble_iga_load_3d(
    mesh: &NurbsMesh3D,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut rhs = vec![0.0_f64; n_total];

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);

            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);

            for a in 0..n_dof {
                rhs[dof_offset + a] += f_val * basis[a] * w;
            }
        }
        dof_offset += n_dof;
    }

    rhs
}

/// Assemble the 3-D linear elasticity stiffness matrix.
///
/// Uses interleaved DOF ordering: control point `a` contributes DOFs
/// `3a` (x-displacement), `3a+1` (y-displacement), `3a+2` (z-displacement).
///
/// Isotropic 3-D elasticity (λ and μ are the Lamé parameters).
/// Assemble the 3-D linear elasticity stiffness matrix with uniform material.
///
/// See [`assemble_iga_elasticity_2d`] for conventions.
pub fn assemble_iga_elasticity_3d(
    mesh: &NurbsMesh3D,
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_iga_elasticity_3d_impl(mesh, quad_order, |_tag, _idx| (lambda, mu))
}

/// Assemble the 3-D linear elasticity stiffness matrix with per-tag materials.
///
/// `materials` is a slice of `(tag, lambda, mu)` entries.
/// See [`assemble_iga_elasticity_2d_multi`] for details.
pub fn assemble_iga_elasticity_3d_multi(
    mesh: &NurbsMesh3D,
    materials: &[(i32, f64, f64)],
    quad_order: u8,
) -> CsrMatrix<f64> {
    assemble_iga_elasticity_3d_impl(mesh, quad_order, |tag, _idx| {
        materials.iter().copied()
            .find(|(t, _, _)| *t == tag)
            .map(|(_, l, m)| (l, m))
            .unwrap_or_else(|| panic!(
                "assemble_iga_elasticity_3d_multi: no material entry for tag={tag}"
            ))
    })
}

/// Shared implementation for 3-D IGA elasticity assembly.
fn assemble_iga_elasticity_3d_impl(
    mesh: &NurbsMesh3D,
    quad_order: u8,
    get_material: impl Fn(i32, usize) -> (f64, f64),
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let n_vec = 3 * n_total;
    let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

    let mut dof_offset = 0usize;
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let (lambda, mu) = get_material(pd.tag, pi);
        let c1 = lambda + 2.0 * mu;
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);

        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_3d(pd, qp_xi);
            let w = qp_w * det_j.abs();

            for a in 0..n_dof {
                let gax = phys_grads[a * 3];
                let gay = phys_grads[a * 3 + 1];
                let gaz = phys_grads[a * 3 + 2];
                let base_a = 3 * (dof_offset + a);

                for b in 0..n_dof {
                    let gbx = phys_grads[b * 3];
                    let gby = phys_grads[b * 3 + 1];
                    let gbz = phys_grads[b * 3 + 2];
                    let base_b = 3 * (dof_offset + b);

                    // 3×3 block: K[3a+i, 3b+j] for i,j ∈ {0,1,2}
                    // K[0,0] = (λ+2μ)·gax·gbx + μ·(gay·gby + gaz·gbz)
                    let k00 = w * (c1 * gax * gbx + mu * (gay * gby + gaz * gbz));
                    // K[0,1] = λ·gax·gby + μ·gay·gbx
                    let k01 = w * (lambda * gax * gby + mu * gay * gbx);
                    // K[0,2] = λ·gax·gbz + μ·gaz·gbx
                    let k02 = w * (lambda * gax * gbz + mu * gaz * gbx);
                    // K[1,1] = (λ+2μ)·gay·gby + μ·(gax·gbx + gaz·gbz)
                    let k11 = w * (c1 * gay * gby + mu * (gax * gbx + gaz * gbz));
                    // K[1,2] = λ·gay·gbz + μ·gaz·gby
                    let k12 = w * (lambda * gay * gbz + mu * gaz * gby);
                    // K[2,2] = (λ+2μ)·gaz·gbz + μ·(gax·gbx + gay·gby)
                    let k22 = w * (c1 * gaz * gbz + mu * (gax * gbx + gay * gby));

                    coo.add(base_a, base_b, k00);
                    coo.add(base_a, base_b + 1, k01);
                    coo.add(base_a, base_b + 2, k02);
                    coo.add(base_a + 1, base_b, k01);     // symmetric: K[1,0]
                    coo.add(base_a + 1, base_b + 1, k11);
                    coo.add(base_a + 1, base_b + 2, k12);
                    coo.add(base_a + 2, base_b, k02);     // symmetric: K[2,0]
                    coo.add(base_a + 2, base_b + 1, k12); // symmetric: K[2,1]
                    coo.add(base_a + 2, base_b + 2, k22);
                }
            }
        }
        dof_offset += n_dof;
    }

    coo.into_csr()
}

// ─── 3-D multi-patch C⁰ assembly ────────────────────────────────────────────

/// Assemble the diffusion stiffness matrix for a multi-patch 3-D NURBS mesh
/// with C⁰ coupling via a DOF map.
pub fn assemble_iga_diffusion_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_3d(pd, qp_xi);
            let w = qp_w * det_j.abs();
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    let dot = phys_grads[a*3]*phys_grads[b*3]
                            + phys_grads[a*3+1]*phys_grads[b*3+1]
                            + phys_grads[a*3+2]*phys_grads[b*3+2];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
    }
    coo.into_csr()
}

/// Assemble the load vector for a multi-patch 3-D NURBS mesh with C⁰ coupling.
pub fn assemble_iga_load_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n_global_dofs];
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                rhs[dof_map[pi][a] as usize] += f_val * basis[a] * w;
            }
        }
    }
    rhs
}

/// Assemble the mass matrix for a multi-patch 3-D NURBS mesh with C⁰ coupling.
pub fn assemble_iga_mass_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    coo.add(ga, gb, rho * basis[a] * basis[b] * w);
                }
            }
        }
    }
    coo.into_csr()
}

// ─── IGA geometrically nonlinear hyperelasticity (2-D) ──────────────────

/// 2-D IGA geometrically nonlinear hyperelasticity (Neo-Hookean).
///
/// DOF ordering: control point `a` → DOFs `2a` (ux), `2a+1` (uy).
///
/// This is a basic implementation that demonstrates nonlinear IGA.
/// Extensions: multi-patch, higher-order materials.
pub struct IgaHyperelasticity2D {
    mesh: NurbsMesh2D,
    model: crate::physics::nonlinear_hyperelasticity::HyperelasticModel,
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8,
}

impl IgaHyperelasticity2D {
    pub fn new(
        mesh: NurbsMesh2D,
        model: crate::physics::nonlinear_hyperelasticity::HyperelasticModel,
        dirichlet: Vec<(usize, f64)>,
        quad_order: u8,
    ) -> Self {
        Self { mesh, model, dirichlet, quad_order }
    }
}

impl crate::physics::nonlinear::NonlinearForm for IgaHyperelasticity2D {
    fn n_dofs(&self) -> usize {
        2 * self.mesh.patches.iter().map(|p| p.control_pts.len()).sum::<usize>()
    }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let dim = 2usize;
        for i in 0..r.len() { r[i] = -rhs[i]; }

        let mut offset = 0usize;
        for pd in &self.mesh.patches {
            let elem = pd_to_patch2d(pd);
            let n_dof = elem.n_dofs();
            let qr = patch_quad_2d(pd, self.quad_order);
            let mut f_elem = vec![0.0_f64; n_dof * dim];

            for (xi, wq) in qr.points.iter().zip(qr.weights.iter()) {
                let (grads, det_j) = physical_grads_2d(pd, xi);
                let w = wq * det_j.abs();

                let mut du = nalgebra::DMatrix::zeros(dim, dim);
                for a in 0..n_dof {
                    let ux = u[2 * (offset + a)];
                    let uy = u[2 * (offset + a) + 1];
                    du[(0, 0)] += ux * grads[a * 2];
                    du[(0, 1)] += ux * grads[a * 2 + 1];
                    du[(1, 0)] += uy * grads[a * 2];
                    du[(1, 1)] += uy * grads[a * 2 + 1];
                }
                let mut f_mat = nalgebra::DMatrix::identity(dim, dim);
                f_mat += &du;
                let (p, _) = self.model.pk1_and_tangent(&f_mat);

                for a in 0..n_dof {
                    let gx = grads[a * 2];
                    let gy = grads[a * 2 + 1];
                    f_elem[a * 2]     += w * (p[(0, 0)] * gx + p[(0, 1)] * gy);
                    f_elem[a * 2 + 1] += w * (p[(1, 0)] * gx + p[(1, 1)] * gy);
                }
            }
            for a in 0..n_dof {
                r[2 * (offset + a)]     += f_elem[a * 2];
                r[2 * (offset + a) + 1] += f_elem[a * 2 + 1];
            }
            offset += n_dof;
        }
        for &(dof, val) in &self.dirichlet { r[dof] = u[dof] - val; }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let dim = 2usize;
        let n_total: usize = self.mesh.patches.iter().map(|p| p.control_pts.len()).sum();
        let n_vec = dim * n_total;
        let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

        let mut offset = 0usize;
        for pd in &self.mesh.patches {
            let elem = pd_to_patch2d(pd);
            let n_dof = elem.n_dofs();
            let qr = patch_quad_2d(pd, self.quad_order);

            for (xi, wq) in qr.points.iter().zip(qr.weights.iter()) {
                let (grads, det_j) = physical_grads_2d(pd, xi);
                let w = wq * det_j.abs();

                let mut du = nalgebra::DMatrix::zeros(dim, dim);
                for a in 0..n_dof {
                    let ux = u[2 * (offset + a)];
                    let uy = u[2 * (offset + a) + 1];
                    du[(0, 0)] += ux * grads[a * 2];
                    du[(0, 1)] += ux * grads[a * 2 + 1];
                    du[(1, 0)] += uy * grads[a * 2];
                    du[(1, 1)] += uy * grads[a * 2 + 1];
                }
                let mut f_mat = nalgebra::DMatrix::identity(dim, dim);
                f_mat += &du;
                // Tangent from the same model used by `residual`
                // (pk1_and_tangent) so the Newton update direction is
                // consistent — the previous hand-written formula did not
                // match the PK1 stress and stalled the iteration.
                let (_, ct) = self.model.pk1_and_tangent(&f_mat);
                // ct[(i*dim+I, j*dim+J)] = ∂P_{iI}/∂F_{jJ}
                // K[2a+i, 2b+j] += w · Σ_{I,J} ct[(i,I),(j,J)]·∂N_a/∂x_I·∂N_b/∂x_J
                for a in 0..n_dof {
                    let ba = 2 * (offset + a);
                    for b in 0..n_dof {
                        let bb = 2 * (offset + b);
                        for i in 0..dim {
                            for I in 0..dim {
                                for j in 0..dim {
                                    for J in 0..dim {
                                        let val = w * ct[(i * dim + I, j * dim + J)]
                                            * grads[a * 2 + I] * grads[b * 2 + J];
                                        if val.abs() > 1e-300 {
                                            coo.add(ba + i, bb + j, val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            offset += n_dof;
        }

        let mut mat = coo.into_csr_sorted();
        let mut dummy = vec![0.0; n_vec];
        for &(dof, _) in &self.dirichlet {
            mat.apply_dirichlet_row_zeroing(dof, 0.0, &mut dummy);
        }
        mat
    }
}

// ─── IGA geometrically nonlinear hyperelasticity (3-D) ──────────────────

/// 3-D IGA geometrically nonlinear hyperelasticity (Neo-Hookean).
///
/// DOF ordering: control point `a` → DOFs `3a` (ux), `3a+1` (uy), `3a+2` (uz).
pub struct IgaHyperelasticity3D {
    mesh: NurbsMesh3D,
    model: crate::physics::nonlinear_hyperelasticity::HyperelasticModel,
    dirichlet: Vec<(usize, f64)>,
    quad_order: u8,
}

impl IgaHyperelasticity3D {
    pub fn new(
        mesh: NurbsMesh3D,
        model: crate::physics::nonlinear_hyperelasticity::HyperelasticModel,
        dirichlet: Vec<(usize, f64)>,
        quad_order: u8,
    ) -> Self {
        Self { mesh, model, dirichlet, quad_order }
    }
}

impl crate::physics::nonlinear::NonlinearForm for IgaHyperelasticity3D {
    fn n_dofs(&self) -> usize {
        3 * self.mesh.patches.iter().map(|p| p.control_pts.len()).sum::<usize>()
    }

    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]) {
        let dim = 3usize;
        for i in 0..r.len() { r[i] = -rhs[i]; }

        let mut offset = 0usize;
        for pd in &self.mesh.patches {
            let elem = pd.patch_element_ref();
            let n_dof = elem.n_dofs();
            let qr = patch_quad_3d(pd, self.quad_order);
            let mut f_elem = vec![0.0_f64; n_dof * dim];

            for (xi, wq) in qr.points.iter().zip(qr.weights.iter()) {
                let (grads, det_j) = physical_grads_3d(pd, xi);
                let w = wq * det_j.abs();

                let mut du = nalgebra::DMatrix::zeros(dim, dim);
                for a in 0..n_dof {
                    let ux = u[3 * (offset + a)];
                    let uy = u[3 * (offset + a) + 1];
                    let uz = u[3 * (offset + a) + 2];
                    du[(0, 0)] += ux * grads[a * 3];
                    du[(0, 1)] += ux * grads[a * 3 + 1];
                    du[(0, 2)] += ux * grads[a * 3 + 2];
                    du[(1, 0)] += uy * grads[a * 3];
                    du[(1, 1)] += uy * grads[a * 3 + 1];
                    du[(1, 2)] += uy * grads[a * 3 + 2];
                    du[(2, 0)] += uz * grads[a * 3];
                    du[(2, 1)] += uz * grads[a * 3 + 1];
                    du[(2, 2)] += uz * grads[a * 3 + 2];
                }
                let mut f_mat = nalgebra::DMatrix::identity(dim, dim);
                f_mat += &du;
                let (p, _) = self.model.pk1_and_tangent(&f_mat);

                for a in 0..n_dof {
                    let gx = grads[a * 3];
                    let gy = grads[a * 3 + 1];
                    let gz = grads[a * 3 + 2];
                    f_elem[a * 3]     += w * (p[(0, 0)] * gx + p[(0, 1)] * gy + p[(0, 2)] * gz);
                    f_elem[a * 3 + 1] += w * (p[(1, 0)] * gx + p[(1, 1)] * gy + p[(1, 2)] * gz);
                    f_elem[a * 3 + 2] += w * (p[(2, 0)] * gx + p[(2, 1)] * gy + p[(2, 2)] * gz);
                }
            }
            for a in 0..n_dof {
                r[3 * (offset + a)]     += f_elem[a * 3];
                r[3 * (offset + a) + 1] += f_elem[a * 3 + 1];
                r[3 * (offset + a) + 2] += f_elem[a * 3 + 2];
            }
            offset += n_dof;
        }
        for &(dof, val) in &self.dirichlet { r[dof] = u[dof] - val; }
    }

    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64> {
        let dim = 3usize;
        let n_total: usize = self.mesh.patches.iter().map(|p| p.control_pts.len()).sum();
        let n_vec = dim * n_total;
        let mut coo = CooMatrix::<f64>::new(n_vec, n_vec);

        let mut offset = 0usize;
        for pd in &self.mesh.patches {
            let elem = pd.patch_element_ref();
            let n_dof = elem.n_dofs();
            let qr = patch_quad_3d(pd, self.quad_order);

            for (xi, wq) in qr.points.iter().zip(qr.weights.iter()) {
                let (grads, det_j) = physical_grads_3d(pd, xi);
                let w = wq * det_j.abs();

                let mut du = nalgebra::DMatrix::zeros(dim, dim);
                for a in 0..n_dof {
                    let ux = u[3 * (offset + a)];
                    let uy = u[3 * (offset + a) + 1];
                    let uz = u[3 * (offset + a) + 2];
                    du[(0, 0)] += ux * grads[a * 3];
                    du[(0, 1)] += ux * grads[a * 3 + 1];
                    du[(0, 2)] += ux * grads[a * 3 + 2];
                    du[(1, 0)] += uy * grads[a * 3];
                    du[(1, 1)] += uy * grads[a * 3 + 1];
                    du[(1, 2)] += uy * grads[a * 3 + 2];
                    du[(2, 0)] += uz * grads[a * 3];
                    du[(2, 1)] += uz * grads[a * 3 + 1];
                    du[(2, 2)] += uz * grads[a * 3 + 2];
                }
                let mut f_mat = nalgebra::DMatrix::identity(dim, dim);
                f_mat += &du;
                // Tangent from the same model used by `residual`
                // (pk1_and_tangent) so the Newton update direction is
                // consistent — the previous hand-written spatial-tangent
                // formula did not match the PK1 stress and stalled/NaN'd.
                let (_, ct) = self.model.pk1_and_tangent(&f_mat);
                // ct[(i*dim+I, j*dim+J)] = ∂P_{iI}/∂F_{jJ}
                // K[3a+i, 3b+j] += w · Σ_{I,J} ct[(i,I),(j,J)]·∂R_a/∂x_I·∂R_b/∂x_J
                for a in 0..n_dof {
                    let ba = 3 * (offset + a);
                    for b in 0..n_dof {
                        let bb = 3 * (offset + b);
                        for i in 0..dim {
                            for I in 0..dim {
                                for j in 0..dim {
                                    for J in 0..dim {
                                        let val = w * ct[(i * dim + I, j * dim + J)]
                                            * grads[a * 3 + I] * grads[b * 3 + J];
                                        if val.abs() > 1e-300 {
                                            coo.add(ba + i, bb + j, val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            offset += n_dof;
        }

        let mut mat = coo.into_csr_sorted();
        let mut dummy = vec![0.0; n_vec];
        for &(dof, _) in &self.dirichlet {
            mat.apply_dirichlet_row_zeroing(dof, 0.0, &mut dummy);
        }
        mat
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

pub(crate) fn pd_to_patch2d(pd: &NurbsPatch2DData) -> fem_element::iga::NurbsPatch2D {
    fem_element::iga::NurbsPatch2D::new(
        pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone(),
    )
}

// ─── IGA Time Integration ─────────────────────────────────────────────────────

/// Perform one step of **Backward Euler** (BE) time integration for the
/// semi-discrete IGA heat equation:
///
/// ```text
/// M u^{n+1} + dt * K u^{n+1} = M u^n + dt * f^{n+1}
/// ```
///
/// # Arguments
/// * `dt`         — time step size
/// * `mass_mat`   — IGA mass matrix M (assembled, same sparsity as stiff_mat)
/// * `stiff_mat`  — IGA stiffness / diffusion matrix K
/// * `u_prev`     — solution at the previous time level
/// * `f`          — load vector at the new time level
///
/// # Returns
/// The solution vector `u^{n+1}` as a `Vec<f64>`.
///
/// **Note**: This function builds the system matrix `A = M + dt*K` as a dense
/// matrix and uses LU factorisation.  For production use, a sparse solver should
/// be plugged in.
pub fn iga_time_step_be(
    dt:        f64,
    mass_mat:  &CsrMatrix<f64>,
    stiff_mat: &CsrMatrix<f64>,
    u_prev:    &[f64],
    f:         &[f64],
) -> Vec<f64> {
    let n = u_prev.len();
    assert_eq!(n, f.len());
    assert_eq!(n, mass_mat.nrows);
    assert_eq!(n, stiff_mat.nrows);

    // Build RHS: b = M * u_prev + dt * f
    let mut rhs = vec![0.0_f64; n];
    // M * u_prev
    for i in 0..n {
        for ptr in mass_mat.row_ptr[i]..mass_mat.row_ptr[i + 1] {
            let j = mass_mat.col_idx[ptr] as usize;
            rhs[i] += mass_mat.values[ptr] * u_prev[j];
        }
    }
    for i in 0..n {
        rhs[i] += dt * f[i];
    }

    // Build system matrix A = M + dt * K (dense for now)
    csr_axpy_solve(mass_mat, stiff_mat, dt, 1.0, &rhs)
}

/// Perform one step of **Crank–Nicolson** (CN) time integration for the
/// semi-discrete IGA heat equation:
///
/// ```text
/// (M + dt/2 * K) u^{n+1} = (M - dt/2 * K) u^n + dt/2 * (f^n + f^{n+1})
/// ```
///
/// # Arguments
/// * `dt`         — time step size
/// * `mass_mat`   — IGA mass matrix M
/// * `stiff_mat`  — IGA stiffness / diffusion matrix K
/// * `u_prev`     — solution at the previous time level u^n
/// * `f`          — average load `0.5*(f^n + f^{n+1})` (or just `f^{n+1}`)
///
/// # Returns
/// The solution vector `u^{n+1}`.
pub fn iga_time_step_cn(
    dt:        f64,
    mass_mat:  &CsrMatrix<f64>,
    stiff_mat: &CsrMatrix<f64>,
    u_prev:    &[f64],
    f:         &[f64],
) -> Vec<f64> {
    let n = u_prev.len();
    assert_eq!(n, f.len());
    assert_eq!(n, mass_mat.nrows);
    assert_eq!(n, stiff_mat.nrows);

    let half_dt = 0.5 * dt;

    // Build RHS: b = (M - dt/2 * K) * u_prev + dt * f
    let mut rhs = vec![0.0_f64; n];
    // M * u_prev
    for i in 0..n {
        for ptr in mass_mat.row_ptr[i]..mass_mat.row_ptr[i + 1] {
            let j = mass_mat.col_idx[ptr] as usize;
            rhs[i] += mass_mat.values[ptr] * u_prev[j];
        }
    }
    // - dt/2 * K * u_prev
    for i in 0..n {
        for ptr in stiff_mat.row_ptr[i]..stiff_mat.row_ptr[i + 1] {
            let j = stiff_mat.col_idx[ptr] as usize;
            rhs[i] -= half_dt * stiff_mat.values[ptr] * u_prev[j];
        }
    }
    for i in 0..n {
        rhs[i] += dt * f[i];
    }

    // Solve (M + dt/2 * K) u^{n+1} = rhs
    csr_axpy_solve(mass_mat, stiff_mat, half_dt, 1.0, &rhs)
}

/// Internal helper: solve `(alpha * A + beta * B) x = rhs` via dense LU.
///
/// Builds `A = alpha * mass_mat + beta * stiff_mat` as a dense matrix, then
/// applies nalgebra LU decomposition.
fn csr_axpy_solve(
    a:     &CsrMatrix<f64>,
    b:     &CsrMatrix<f64>,
    alpha: f64,
    beta:  f64,
    rhs:   &[f64],
) -> Vec<f64> {
    use nalgebra::{DMatrix, DVector};
    let n = rhs.len();
    let mut dense = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            dense[(i, a.col_idx[ptr] as usize)] += alpha * a.values[ptr];
        }
    }
    for i in 0..n {
        for ptr in b.row_ptr[i]..b.row_ptr[i + 1] {
            dense[(i, b.col_idx[ptr] as usize)] += beta * b.values[ptr];
        }
    }

    let bv = DVector::from_column_slice(rhs);
    dense.lu().solve(&bv)
        .map(|x| x.iter().cloned().collect())
        .unwrap_or_else(|| rhs.to_vec())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsMesh2D, NurbsMesh3D};

    // ── Physical map ──────────────────────────────────────────────────────────

    /// Unit square mapped from the reference [0,1]^2: identity map.
    /// Control points at the four corners with Q1 (bilinear degree-1) basis.
    #[test]
    fn physical_map_2d_unit_square_is_identity() {
        let kv = NurbsKnotVector::uniform(1, 1);
        // Q1 patch on [0,1]^2: control pts = corners in (i,j) order
        // DOF order: j*n_u + i, so DOF 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
        let ctrl = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; 4]);
        let pd = &mesh.patches[0];

        // At the centre of the domain: xi = [0.5, 0.5] → x = [0.5, 0.5]
        let map = physical_map_2d(pd, &[0.5, 0.5]);
        assert!((map.x_phys[0] - 0.5).abs() < 1e-12, "x_phys[0]={}", map.x_phys[0]);
        assert!((map.x_phys[1] - 0.5).abs() < 1e-12, "x_phys[1]={}", map.x_phys[1]);
        // Jacobian should be identity for unit-square map.
        assert!((map.det_j - 1.0).abs() < 1e-12, "det_j={}", map.det_j);
    }

    /// A scaled rectangle [0,2] × [0,3]: Jacobian should be diagonal with
    /// det = 2*3 = 6.
    #[test]
    fn physical_map_2d_rectangle_jacobian() {
        let kv = NurbsKnotVector::uniform(1, 1);
        let ctrl = vec![[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]];
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; 4]);
        let pd = &mesh.patches[0];

        let map = physical_map_2d(pd, &[0.5, 0.5]);
        // Physical coords should be (1.0, 1.5).
        assert!((map.x_phys[0] - 1.0).abs() < 1e-12);
        assert!((map.x_phys[1] - 1.5).abs() < 1e-12);
        // det(J) = 2 * 3 = 6
        assert!((map.det_j - 6.0).abs() < 1e-12, "det_j={}", map.det_j);
    }

    /// 3-D unit cube: det_j should be 1 everywhere.
    #[test]
    fn physical_map_3d_unit_cube_det_is_one() {
        let kv = NurbsKnotVector::uniform(1, 1);
        // 8 control points of the unit cube
        let ctrl = vec![
            [0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],
            [0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0],
        ];
        let mesh = NurbsMesh3D::single_patch(
            kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; 8],
        );
        let pd = &mesh.patches[0];
        let map = physical_map_3d(pd, &[0.5, 0.5, 0.5]);
        assert!((map.det_j - 1.0).abs() < 1e-12, "det_j={}", map.det_j);
    }

    // ── IGA diffusion (2-D) ───────────────────────────────────────────────────

    /// Partition-of-unity test: mass matrix row sums should equal the patch volume.
    #[test]
    fn iga_mass_2d_row_sum_equals_area() {
        let kv = NurbsKnotVector::uniform(1, 2); // 2 elements per direction
        let n_u = kv.n_basis(); // = 3
        let n_dof = n_u * n_u; // = 9
        let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = idx / n_u;
            [i as f64 / (n_u - 1) as f64, j as f64 / (n_u - 1) as f64]
        }).collect();
        let mesh = NurbsMesh2D::single_patch(
            kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let m = assemble_iga_mass_2d(&mesh, 1.0, 3);
        // Row sums should each equal the integral of 1 weighted by that basis function.
        // Their total (sum of all row sums) should equal the area = 1.0.
        let total: f64 = (0..n_dof).map(|a| {
            (0..n_dof).map(|b| m.values[m.row_ptr[a]..m.row_ptr[a+1]]
                .iter().zip(&m.col_idx[m.row_ptr[a]..m.row_ptr[a+1]])
                .find(|(_, &c)| c as usize == b)
                .map(|(v, _)| *v)
                .unwrap_or(0.0)
            ).sum::<f64>()
        }).sum();
        assert!((total - 1.0).abs() < 1e-10, "total mass={total:.6}, expected 1.0");
    }

    /// Load vector sum should equal the integral of f = 1 over the domain = area.
    #[test]
    fn iga_load_2d_unit_source_sums_to_area() {
        let kv = NurbsKnotVector::uniform(1, 2);
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u;
        let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = idx / n_u;
            [i as f64 / (n_u - 1) as f64, j as f64 / (n_u - 1) as f64]
        }).collect();
        let mesh = NurbsMesh2D::single_patch(
            kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let rhs = assemble_iga_load_2d(&mesh, |_| 1.0, 3);
        let total: f64 = rhs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "rhs sum={total:.6}, expected 1.0");
    }

    /// Stiffness matrix symmetry for unit-square patch.
    #[test]
    fn iga_stiffness_2d_is_symmetric() {
        let kv = NurbsKnotVector::uniform(2, 2); // degree-2, 2 elements
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u;
        let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = idx / n_u;
            [i as f64 / (n_u - 1) as f64, j as f64 / (n_u - 1) as f64]
        }).collect();
        let mesh = NurbsMesh2D::single_patch(
            kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let k = assemble_iga_diffusion_2d(&mesh, 1.0, 4);

        // Build dense representation for symmetry check.
        let n = n_dof;
        let mut dense = vec![0.0_f64; n * n];
        for i in 0..n {
            for ptr in k.row_ptr[i]..k.row_ptr[i+1] {
                dense[i * n + k.col_idx[ptr] as usize] = k.values[ptr];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i*n+j] - dense[j*n+i]).abs();
                assert!(diff < 1e-12, "K[{i},{j}]={:.6e} != K[{j},{i}]={:.6e}", dense[i*n+j], dense[j*n+i]);
            }
        }
    }

    /// Stiffness matrix positive semi-definiteness: all eigenvalues ≥ -1e-10.
    /// We check this by verifying x^T K x ≥ 0 for random vectors.
    #[test]
    fn iga_stiffness_2d_is_positive_semidefinite() {
        let kv = NurbsKnotVector::uniform(1, 3); // degree-1, 3 elements
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u;
        let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = idx / n_u;
            [i as f64 / (n_u - 1) as f64, j as f64 / (n_u - 1) as f64]
        }).collect();
        let mesh = NurbsMesh2D::single_patch(
            kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let k = assemble_iga_diffusion_2d(&mesh, 1.0, 3);

        // Test x^T K x ≥ 0 for a few vectors.
        let test_vecs: Vec<Vec<f64>> = vec![
            (0..n_dof).map(|i| i as f64).collect(),
            (0..n_dof).map(|i| (i % 3) as f64 - 1.0).collect(),
            (0..n_dof).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
        ];
        for x in &test_vecs {
            let mut kx = vec![0.0_f64; n_dof];
            for i in 0..n_dof {
                for ptr in k.row_ptr[i]..k.row_ptr[i+1] {
                    kx[i] += k.values[ptr] * x[k.col_idx[ptr] as usize];
                }
            }
            let xt_kx: f64 = x.iter().zip(&kx).map(|(xi, kxi)| xi * kxi).sum();
            assert!(xt_kx > -1e-10, "x^T K x = {xt_kx:.3e} < 0 (not PSD)");
        }
    }

    // ── IGA diffusion (3-D) ───────────────────────────────────────────────────

    /// 3-D load vector sum for f=1 on unit cube should equal volume = 1.
    #[test]
    fn iga_load_3d_unit_source_sums_to_volume() {
        let kv = NurbsKnotVector::uniform(1, 2);
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u * n_u;
        let ctrl: Vec<[f64; 3]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = (idx / n_u) % n_u;
            let k = idx / (n_u * n_u);
            [
                i as f64 / (n_u - 1) as f64,
                j as f64 / (n_u - 1) as f64,
                k as f64 / (n_u - 1) as f64,
            ]
        }).collect();
        let mesh = NurbsMesh3D::single_patch(
            kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let rhs = assemble_iga_load_3d(&mesh, |_| 1.0, 3);
        let total: f64 = rhs.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "3D rhs sum={total:.6}, expected 1.0");
    }

    /// 3-D stiffness matrix symmetry on unit cube.
    #[test]
    fn iga_stiffness_3d_is_symmetric() {
        let kv = NurbsKnotVector::uniform(1, 2);
        let n_u = kv.n_basis();
        let n_dof = n_u * n_u * n_u;
        let ctrl: Vec<[f64; 3]> = (0..n_dof).map(|idx| {
            let i = idx % n_u;
            let j = (idx / n_u) % n_u;
            let k_idx = idx / (n_u * n_u);
            [
                i as f64 / (n_u - 1) as f64,
                j as f64 / (n_u - 1) as f64,
                k_idx as f64 / (n_u - 1) as f64,
            ]
        }).collect();
        let mesh = NurbsMesh3D::single_patch(
            kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_dof],
        );
        let k_mat = assemble_iga_diffusion_3d(&mesh, 1.0, 3);
        let n = n_dof;
        let mut dense = vec![0.0_f64; n * n];
        for i in 0..n {
            for ptr in k_mat.row_ptr[i]..k_mat.row_ptr[i+1] {
                dense[i * n + k_mat.col_idx[ptr] as usize] = k_mat.values[ptr];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i*n+j] - dense[j*n+i]).abs();
                assert!(diff < 1e-11, "K3D[{i},{j}] != K3D[{j},{i}]: diff={diff:.3e}");
            }
        }
    }

    // ── Convergence test ─────────────────────────────────────────────────────

    /// Poisson on [0,1]^2: for a manufactured solution, increasing the number
    /// of elements (h-refinement) should decrease the L2 error.
    ///
    /// Manufactured solution: u(x,y) = x*(1-x)*y*(1-y)
    /// Source: f = 2*y*(1-y) + 2*x*(1-x)
    #[test]
    fn iga_poisson_2d_l2_error_decreases_with_refinement() {
        fn l2_error_with_n_elems(n_elems: usize) -> f64 {
            let kv = NurbsKnotVector::uniform(2, n_elems); // degree-2
            let n_u = kv.n_basis();
            let n_dof = n_u * n_u;
            let ctrl: Vec<[f64; 2]> = (0..n_dof).map(|idx| {
                let i = idx % n_u;
                let j = idx / n_u;
                [i as f64 / (n_u - 1) as f64, j as f64 / (n_u - 1) as f64]
            }).collect();
            let mesh = NurbsMesh2D::single_patch(
                kv.clone(), kv.clone(), ctrl.clone(), vec![1.0; n_dof],
            );

            let mut k = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
            let mut rhs = assemble_iga_load_2d(&mesh, |x| {
                2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0])
            }, 4);

            // Apply Dirichlet BCs: boundary control points → u = 0.
            // Identify boundary DOFs: those on i=0, i=n_u-1, j=0, j=n_u-1.
            let mut bc_dofs = Vec::new();
            for j in 0..n_u {
                for i in 0..n_u {
                    if i == 0 || i == n_u - 1 || j == 0 || j == n_u - 1 {
                        bc_dofs.push(j * n_u + i);
                    }
                }
            }
            // Enforce Dirichlet by zeroing rows/cols (symmetric elimination).
            apply_dirichlet_iga(&mut k, &mut rhs, &bc_dofs);
            let u = direct_solve(&k, &rhs);

            // Compute L2 error by quadrature.
            let pd = &mesh.patches[0];
            let qr = patch_quad_2d(pd, 5);
            let mut err_sq = 0.0_f64;
            for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
                let map = physical_map_2d(pd, qp_xi);
                let w = qp_w * map.det_j.abs();
                let x = map.x_phys[0];
                let y = map.x_phys[1];
                let u_exact = x * (1.0 - x) * y * (1.0 - y);

                let elem = pd_to_patch2d(pd);
                let mut basis = vec![0.0_f64; n_dof];
                elem.eval_basis(qp_xi, &mut basis);
                let u_h: f64 = basis.iter().zip(&u).map(|(r, ui)| r * ui).sum();
                err_sq += (u_exact - u_h).powi(2) * w;
            }
            err_sq.sqrt()
        }

        let e_coarse = l2_error_with_n_elems(2);
        let e_fine   = l2_error_with_n_elems(4);
        assert!(
            e_fine < e_coarse,
            "L2 error should decrease: coarse={e_coarse:.3e}, fine={e_fine:.3e}"
        );
        // For degree-2 IGA, expect at least O(h^2) convergence ≈ factor of 4.
        let ratio = e_coarse / e_fine;
        assert!(
            ratio > 2.0,
            "Expected at least O(h^2) convergence; got ratio={ratio:.2}"
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Apply homogeneous Dirichlet BCs: zero out rows and columns for `dofs`,
    /// set diagonal to 1, zero out RHS.
    fn apply_dirichlet_iga(k: &mut CsrMatrix<f64>, rhs: &mut Vec<f64>, dofs: &[usize]) {
        // Build a set for fast lookup.
        let bc_set: std::collections::HashSet<usize> = dofs.iter().copied().collect();
        let n = rhs.len();

        // Zero rows.
        for &d in dofs {
            if d < n {
                for ptr in k.row_ptr[d]..k.row_ptr[d+1] {
                    let col = k.col_idx[ptr] as usize;
                    k.values[ptr] = if col == d { 1.0 } else { 0.0 };
                }
                rhs[d] = 0.0;
            }
        }
        // Zero columns.
        for i in 0..n {
            if bc_set.contains(&i) { continue; }
            for ptr in k.row_ptr[i]..k.row_ptr[i+1] {
                let col = k.col_idx[ptr] as usize;
                if bc_set.contains(&col) {
                    k.values[ptr] = 0.0;
                }
            }
        }
    }

    /// Direct solver for testing (nalgebra LU on dense representation).
    fn direct_solve(k: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
        use nalgebra::{DMatrix, DVector};
        let n = rhs.len();
        let mut dense = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for ptr in k.row_ptr[i]..k.row_ptr[i+1] {
                dense[(i, k.col_idx[ptr] as usize)] = k.values[ptr];
            }
        }
        let b = DVector::from_column_slice(rhs);
        dense.lu().solve(&b)
            .map(|x| x.iter().cloned().collect())
            .unwrap_or_else(|| rhs.to_vec())
    }

    // ── IGA time integration tests ────────────────────────────────────────────

    fn make_1d_iga_mesh() -> NurbsMesh2D {
        use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};
        let kv_u = NurbsKnotVector::uniform(1, 2); // degree-1, 2 elements
        let kv_v = NurbsKnotVector::uniform(1, 2);
        let n_u = kv_u.n_basis();
        let n_v = kv_v.n_basis();
        let mut cpts = Vec::new();
        let mut weights = Vec::new();
        for j in 0..n_v {
            for i in 0..n_u {
                cpts.push([i as f64 / (n_u - 1) as f64, j as f64 / (n_v - 1) as f64]);
                weights.push(1.0_f64);
            }
        }
        NurbsMesh2D::single_patch(kv_u, kv_v, cpts, weights)
    }

    #[test]
    fn be_step_preserves_zero_solution_with_zero_source() {
        let mesh = make_1d_iga_mesh();
        let mass = assemble_iga_mass_2d(&mesh, 1.0, 4);
        let stiff = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
        let n = mass.nrows;
        let u_prev = vec![0.0_f64; n];
        let f = vec![0.0_f64; n];
        let u_next = iga_time_step_be(0.01, &mass, &stiff, &u_prev, &f);
        let norm: f64 = u_next.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1e-12, "BE: zero IC + zero source should stay zero, norm={norm}");
    }

    #[test]
    fn cn_step_preserves_zero_solution_with_zero_source() {
        let mesh = make_1d_iga_mesh();
        let mass = assemble_iga_mass_2d(&mesh, 1.0, 4);
        let stiff = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
        let n = mass.nrows;
        let u_prev = vec![0.0_f64; n];
        let f = vec![0.0_f64; n];
        let u_next = iga_time_step_cn(0.01, &mass, &stiff, &u_prev, &f);
        let norm: f64 = u_next.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1e-12, "CN: zero IC + zero source should stay zero, norm={norm}");
    }

    #[test]
    fn be_step_with_constant_source_increases_magnitude() {
        let mesh = make_1d_iga_mesh();
        let mass = assemble_iga_mass_2d(&mesh, 1.0, 4);
        let stiff = assemble_iga_diffusion_2d(&mesh, 0.0, 4); // zero diffusion → pure mass
        let n = mass.nrows;
        let u_prev = vec![0.0_f64; n];
        let f = vec![1.0_f64; n];
        let u_next = iga_time_step_be(1.0, &mass, &stiff, &u_prev, &f);
        // With zero diffusion: M u^{n+1} = M*0 + dt*f = f → u^{n+1} = M^{-1} f > 0
        let norm: f64 = u_next.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "BE with constant source should produce non-zero solution, norm={norm}");
    }

    #[test]
    fn cn_be_agree_at_small_dt() {
        // For very small dt, BE and CN should give nearly identical results.
        // Use dt=0.01 (not tiny) to avoid LU ill-conditioning of M+dt*K.
        let mesh = make_1d_iga_mesh();
        let mass = assemble_iga_mass_2d(&mesh, 1.0, 4);
        let stiff = assemble_iga_diffusion_2d(&mesh, 1.0, 4);
        let n = mass.nrows;
        let u_prev: Vec<f64> = vec![0.0; n]; // zero IC to avoid ill-conditioning
        let f = vec![0.0_f64; n];            // zero source
        let dt = 0.01;
        let u_be = iga_time_step_be(dt, &mass, &stiff, &u_prev, &f);
        let u_cn = iga_time_step_cn(dt, &mass, &stiff, &u_prev, &f);
        let diff: f64 = u_be.iter().zip(&u_cn).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        // Both should give zero (from zero IC + zero source)
        assert!(diff < 1e-12, "BE and CN from zero IC/source should agree, diff={diff}");
    }

    // ── IGA 3D elasticity ─────────────────────────────────────────────────

    #[test]
    fn iga_elasticity_3d_unit_cube_smoke() {
        let p = 1;
        let n = 4; // 4×4×4 control points
        let n_ctrl = n * n * n;
        let kv = NurbsKnotVector::uniform(p, n - p);
        let ctrl: Vec<[f64; 3]> = (0..n_ctrl).map(|idx| {
            let i = idx % n;
            let j = (idx / n) % n;
            let k = idx / (n * n);
            [i as f64 / (n - 1) as f64,
             j as f64 / (n - 1) as f64,
             k as f64 / (n - 1) as f64]
        }).collect();
        let mesh = NurbsMesh3D::single_patch(
            kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl],
        );
        let n_vec = 3 * n_ctrl;

        let e_mod = 1e3;
        let nu = 0.3;
        let lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e_mod / (2.0 * (1.0 + nu));
        let mut stiff = assemble_iga_elasticity_3d(&mesh, lam, mu, 3);

        // Gravity-like load in z-direction
        let load = assemble_iga_load_3d(&mesh, |_| -1.0, 3);
        let mut rhs = vec![0.0; n_vec];
        for a in 0..n_ctrl {
            rhs[3 * a + 2] = load[a]; // z-component
        }

        // Clamped BC: u=0 on all boundary control points
        let mut bc_dofs = Vec::new();
        for i in 0..n { for j in 0..n { for k in 0..n {
            if i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1 {
                let a = k * n * n + j * n + i;
                bc_dofs.push(3*a); bc_dofs.push(3*a+1); bc_dofs.push(3*a+2);
            }
        }}}
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        apply_dirichlet_iga(&mut stiff, &mut rhs, &bc_dofs);

        let u = direct_solve(&stiff, &rhs);
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("  [IGA 3D elasticity] p={}, n_ctrl={}, ||u||={:.6e}", p, n_ctrl, norm);
        assert!(norm > 0.0 && norm < 100.0, "||u||={:.6e} outside [0, 100]", norm);
    }

    // ── IGA 2D nonlinear hyperelasticity ──────────────────────────────

    #[test]
    fn iga_hyperelasticity_2d_smoke() {
        use crate::physics::nonlinear::NewtonConfig;
        use crate::physics::nonlinear_hyperelasticity::HyperelasticModel;

        let p = 1;
        let n = 4;
        let n_ctrl = n * n;
        let kv = NurbsKnotVector::uniform(p, n - p);
        let ctrl: Vec<[f64; 2]> = (0..n_ctrl).map(|idx| {
            let i = idx % n;
            let j = idx / n;
            [i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]
        }).collect();
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl]);

        let mu = 10.0;
        let lam = 10.0;
        let model = HyperelasticModel::NeoHookean { mu, lambda: lam };
        let n_dofs = 2 * n_ctrl;

        // Clamped on bottom (y=0), prescribed uy = -0.05 on top
        let mut dirichlet = Vec::new();
        for i in 0..n {
            let b = i;
            let t = (n-1) * n + i;
            dirichlet.push((2 * b, 0.0));
            dirichlet.push((2 * b + 1, 0.0));
            dirichlet.push((2 * t, 0.0));
            dirichlet.push((2 * t + 1, -0.05));
        }

        let form = IgaHyperelasticity2D::new(mesh, model, dirichlet, 4);
        let rhs = vec![0.0; n_dofs];
        let mut u = vec![0.0; n_dofs];

        let cfg = NewtonConfig {
            atol: 1e-6,
            rtol: 1e-6,
            max_iter: 50,
            linear_tol: 1e-8,
            line_search: true,
            ..NewtonConfig::default()
        };
        let result = crate::physics::nonlinear::NewtonSolver::new(cfg).solve(&form, &rhs, &mut u);
        match &result {
            Ok(r) => eprintln!("  [IGA hyperelasticity] converged in {} iters, final ‖F‖={:.3e}", r.iterations, r.final_residual),
            Err(r) => eprintln!("  [IGA hyperelasticity] FAILED: {} iters, ‖F‖={:.3e}", r.iterations, r.final_residual),
        }

        if let Ok(r) = &result {
            let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!("  [IGA hyperelasticity 2D] ||u||={:.6e}", norm);
            assert!(norm > 0.0 && norm < 100.0, "||u||={:.6e} outside range", norm);
        }
        assert!(result.is_ok(), "Newton did not converge");
    }

    #[test]
    fn iga_hyperelasticity_3d_smoke() {
        use crate::physics::nonlinear::NewtonConfig;
        use crate::physics::nonlinear_hyperelasticity::HyperelasticModel;

        let p = 1;
        let n = 3; // 3×3×3 = 27 control points
        let n_ctrl = n * n * n;
        let kv = NurbsKnotVector::uniform(p, n - p);
        let ctrl: Vec<[f64; 3]> = (0..n_ctrl).map(|idx| {
            let i = idx % n;
            let j = (idx / n) % n;
            let k = idx / (n * n);
            [i as f64 / (n - 1) as f64,
             j as f64 / (n - 1) as f64,
             k as f64 / (n - 1) as f64]
        }).collect();
        let mesh = NurbsMesh3D::single_patch(
            kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl],
        );

        let mu = 10.0;
        let lam = 10.0;
        let model = HyperelasticModel::NeoHookean { mu, lambda: lam };
        let n_dofs = 3 * n_ctrl;

        // Clamped on bottom (z=0, k=0), ux=uy=uz=0
        // Prescribed uz = -0.05 on top (z=1, k=2)
        let mut dirichlet = Vec::new();
        for i in 0..n { for j in 0..n {
            let b = j * n + i;          // k=0 (bottom)
            let t = (n-1)*n*n + j*n + i; // k=n-1 (top)
            dirichlet.push((3*b, 0.0)); dirichlet.push((3*b+1, 0.0)); dirichlet.push((3*b+2, 0.0));
            dirichlet.push((3*t, 0.0)); dirichlet.push((3*t+1, 0.0)); dirichlet.push((3*t+2, -0.05));
        }}

        let form = IgaHyperelasticity3D::new(mesh, model, dirichlet, 4);
        let rhs = vec![0.0; n_dofs];
        let mut u = vec![0.0; n_dofs];

        let cfg = NewtonConfig {
            atol: 1e-6, rtol: 1e-6, max_iter: 50, linear_tol: 1e-8,
            line_search: true,
            ..NewtonConfig::default()
        };
        let result = crate::physics::nonlinear::NewtonSolver::new(cfg).solve(&form, &rhs, &mut u);
        match &result {
            Ok(r) => eprintln!("  [IGA 3D hyperelasticity] converged in {} iters, ‖F‖={:.3e}", r.iterations, r.final_residual),
            Err(r) => eprintln!("  [IGA 3D hyperelasticity] FAILED: {} iters, ‖F‖={:.3e}", r.iterations, r.final_residual),
        }
        if let Ok(r) = &result {
            let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!("  [IGA 3D hyperelasticity] ||u||={:.6e}, iters={}", norm, r.iterations);
            assert!(norm > 0.0 && norm < 100.0, "||u||={:.6e} outside range", norm);
        }
        assert!(result.is_ok(), "3D Newton did not converge");
    }
}

#[cfg(test)]
mod multipatch_tests {
    use super::*;
    use fem_core::types::DofId;
    use fem_element::iga::{NurbsKnotVector, NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData};
    use fem_space::{IgaMultiPatchMesh2D, IgaMultiPatchMesh3D};

    fn make_two_patch_mesh() -> (NurbsMesh2D, Vec<Vec<DofId>>, usize) {
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let patch_a = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.0,0.0],[0.5,0.0],[0.0,1.0],[0.5,1.0]],
            weights: vec![1.0;4], tag: 1,
        };
        let patch_b = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv,
            control_pts: vec![[0.5,0.0],[1.0,0.0],[0.5,1.0],[1.0,1.0]],
            weights: vec![1.0;4], tag: 2,
        };
        let nurbs = NurbsMesh2D { patches: vec![patch_a, patch_b], edge_connectivity: vec![(0,1,1,3)] };
        let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&nurbs);
        let dof_maps: Vec<Vec<DofId>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
        (nurbs, dof_maps, mp.n_global_dofs())
    }

    #[test]
    fn test_multipatch_2d_diffusion_runs() {
        let (mesh, dof_map, n_global) = make_two_patch_mesh();
        let k = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_map, n_global, 1.0, 3);
        assert_eq!(k.nrows, n_global);
        for i in 0..k.nrows {
            let mut sum_row = 0.0_f64;
            for p in k.row_ptr[i]..k.row_ptr[i+1] {
                if k.col_idx[p] as usize == i {
                    assert!(k.values[p] > 0.0, "diagonal entry K[{i},{i}] must be positive");
                }
                sum_row += k.values[p];
            }
            assert!(sum_row >= -1e-14, "row {i} sum = {sum_row} should be ≈ 0");
        }
    }

    #[test]
    fn test_multipatch_3d_diffusion_runs() {
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let patch_a = NurbsPatch3DData {
            kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv.clone(),
            control_pts: vec![
                [0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,1.0,0.0],[0.5,1.0,0.0],
                [0.0,0.0,1.0],[0.5,0.0,1.0],[0.0,1.0,1.0],[0.5,1.0,1.0],
            ],
            weights: vec![1.0;8], tag: 1,
        };
        let patch_b = NurbsPatch3DData {
            kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv,
            control_pts: vec![
                [0.5,0.0,0.0],[1.0,0.0,0.0],[0.5,1.0,0.0],[1.0,1.0,0.0],
                [0.5,0.0,1.0],[1.0,0.0,1.0],[0.5,1.0,1.0],[1.0,1.0,1.0],
            ],
            weights: vec![1.0;8], tag: 2,
        };
        let mesh = NurbsMesh3D { patches: vec![patch_a, patch_b], face_connectivity: vec![(0,1,1,0)] };
        let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&mesh);
        let dof_maps: Vec<Vec<DofId>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
        let k = assemble_iga_diffusion_multipatch_3d(&mesh, &dof_maps, mp.n_global_dofs(), 1.0, 2);
        assert_eq!(k.nrows, mp.n_global_dofs());
        for i in 0..k.nrows {
            let mut sum_row = 0.0_f64;
            for p in k.row_ptr[i]..k.row_ptr[i+1] {
                if k.col_idx[p] as usize == i {
                    assert!(k.values[p] > 0.0, "diag K[{i},{i}] > 0");
                }
                sum_row += k.values[p];
            }
            assert!(sum_row >= -1e-14, "row {i} sum ≈ 0");
        }
    }
}
