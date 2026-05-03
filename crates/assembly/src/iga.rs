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

use fem_element::nurbs::{NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData};
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
fn patch_quad_2d(pd: &NurbsPatch2DData, order: u8) -> QuadratureRule {
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
    use fem_element::nurbs::NurbsPatch2D;

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
pub fn physical_grads_2d(pd: &NurbsPatch2DData, xi: &[f64]) -> (Vec<f64>, f64) {
    use fem_element::nurbs::NurbsPatch2D;

    let patch = NurbsPatch2D::new(pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone());
    let n_dof = patch.n_dofs();

    let mut grads_xi = vec![0.0_f64; n_dof * 2];
    patch.eval_grad_basis(xi, &mut grads_xi);

    let map = physical_map_2d(pd, xi);
    let ji = &map.jac_inv_t;

    // ∇_x R_A = J^{-T} * ∇_ξ R_A
    let mut phys_grads = vec![0.0_f64; n_dof * 2];
    for a in 0..n_dof {
        let dru = grads_xi[a * 2];
        let drv = grads_xi[a * 2 + 1];
        phys_grads[a * 2]     = ji[0][0] * dru + ji[0][1] * drv; // dR/dx
        phys_grads[a * 2 + 1] = ji[1][0] * dru + ji[1][1] * drv; // dR/dy
    }

    (phys_grads, map.det_j)
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
    use fem_element::nurbs::NurbsPatch3D;

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
pub fn physical_grads_3d(pd: &NurbsPatch3DData, xi: &[f64]) -> (Vec<f64>, f64) {
    use fem_element::nurbs::NurbsPatch3D;

    let patch = NurbsPatch3D::new(
        pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone(), pd.weights.clone(),
    );
    let n_dof = patch.n_dofs();

    let mut grads_xi = vec![0.0_f64; n_dof * 3];
    patch.eval_grad_basis(xi, &mut grads_xi);

    let map = physical_map_3d(pd, xi);
    let ji = &map.jac_inv_t;

    let mut phys_grads = vec![0.0_f64; n_dof * 3];
    for a in 0..n_dof {
        let dru = grads_xi[a * 3];
        let drv = grads_xi[a * 3 + 1];
        let drw = grads_xi[a * 3 + 2];
        phys_grads[a * 3]     = ji[0][0]*dru + ji[0][1]*drv + ji[0][2]*drw;
        phys_grads[a * 3 + 1] = ji[1][0]*dru + ji[1][1]*drv + ji[1][2]*drw;
        phys_grads[a * 3 + 2] = ji[2][0]*dru + ji[2][1]*drv + ji[2][2]*drw;
    }

    (phys_grads, map.det_j)
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn pd_to_patch2d(pd: &NurbsPatch2DData) -> fem_element::nurbs::NurbsPatch2D {
    fem_element::nurbs::NurbsPatch2D::new(
        pd.kv_u.clone(), pd.kv_v.clone(), pd.weights.clone(),
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::nurbs::{KnotVector, NurbsMesh2D, NurbsMesh3D};

    // ── Physical map ──────────────────────────────────────────────────────────

    /// Unit square mapped from the reference [0,1]^2: identity map.
    /// Control points at the four corners with Q1 (bilinear degree-1) basis.
    #[test]
    fn physical_map_2d_unit_square_is_identity() {
        let kv = KnotVector::uniform(1, 1);
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
        let kv = KnotVector::uniform(1, 1);
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
        let kv = KnotVector::uniform(1, 1);
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
        let kv = KnotVector::uniform(1, 2); // 2 elements per direction
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
        let kv = KnotVector::uniform(1, 2);
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
        let kv = KnotVector::uniform(2, 2); // degree-2, 2 elements
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
        let kv = KnotVector::uniform(1, 3); // degree-1, 3 elements
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
        let kv = KnotVector::uniform(1, 2);
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
        let kv = KnotVector::uniform(1, 2);
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
            let kv = KnotVector::uniform(2, n_elems); // degree-2
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
}
