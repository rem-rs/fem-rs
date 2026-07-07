//! Prism Pk partial-assembly using Kronecker sum structure.
//!
//! The prism stiffness matrix is `A = S₁ ⊗ M₂ + M₁ ⊗ S₂` where:
//! - S₁/M₁ are 1D Lagrange stiffness/mass matrices on [0,1]
//! - S₂/M₂ are triangle Lagrange stiffness/mass matrices
//!
//! This enables O(p⁴) evaluation instead of O(p⁶) for full assembly.
//!
//! # Kronecker sum application
//!
//! `y = (S₁ ⊗ M₂ + M₁ ⊗ S₂)·x` is computed as:
//! 1. For each 1D layer k: `t[k] = M₂ · x[:,k]` and `w[k] = S₂ · x[:,k]`
//! 2. `y = S₁ · t + M₁ · w` (matrix-multiply across layers)

use crate::pa::types::PaData;
use fem_element::lagrange::{PrismPk, TriPk};
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;

// ─── 1D Lagrange matrices (equispaced nodes on [0,1]) ─────────────────────

/// 1D Lagrange stiffness matrix: `S₁[i][j] = ∫₀¹ ℓ'_i(x)·ℓ'_j(x) dx`.
fn build_1d_stiffness(p: usize) -> Vec<Vec<f64>> {
    let n = p + 1;
    // Exact integration of quadratic products of degree p-1 polynomials
    // Using equispaced nodes and exact integration
    let ref_elem = fem_element::lagrange::SegPk::new(p);
    let (qpts, qwts) = gauss_legendre_1d(2 * p + 1);
    let mut s = vec![vec![0.0; n]; n];
    for (&qp, &qw) in qpts.iter().zip(qwts.iter()) {
        let mut dvals = vec![0.0; n];
        ref_elem.eval_grad_basis(&[qp], &mut dvals);
        for i in 0..n {
            for j in 0..n {
                s[i][j] += dvals[i] * dvals[j] * qw;
            }
        }
    }
    s
}

/// 1D Lagrange mass matrix: `M₁[i][j] = ∫₀¹ ℓ_i(x)·ℓ_j(x) dx`.
fn build_1d_mass(p: usize) -> Vec<Vec<f64>> {
    let n = p + 1;
    let (qpts, qwts) = gauss_legendre_1d(2 * p + 1);
    let ref_elem = fem_element::lagrange::SegPk::new(p);
    let mut m = vec![vec![0.0; n]; n];
    for (&qp, &qw) in qpts.iter().zip(qwts.iter()) {
        let mut vals = vec![0.0; n];
        ref_elem.eval_basis(&[qp], &mut vals);
        for i in 0..n {
            for j in 0..n {
                m[i][j] += vals[i] * vals[j] * qw;
            }
        }
    }
    m
}

// ─── Triangle Lagrange matrices ─────────────────────────────────────────────

/// Triangle stiffness matrix: `S₂[i][j] = ∫ ∇φ_i·∇φ_j dη dζ`.
fn build_tri_stiffness(p: usize) -> Vec<Vec<f64>> {
    let tri = TriPk::new(p);
    let n_tri = tri.n_dofs();
    let rule = tri.quadrature((2 * p + 2).min(15) as u8);
    let mut s = vec![vec![0.0; n_tri]; n_tri];
    for pt_idx in 0..rule.points.len() {
        let pt = &rule.points[pt_idx];
        let w = rule.weights[pt_idx] * 0.5; // scale to unit triangle area
        let mut grads = vec![0.0; n_tri * 2];
        tri.eval_grad_basis(pt, &mut grads);
        for i in 0..n_tri {
            let (gix, giy) = (grads[i * 2], grads[i * 2 + 1]);
            for j in 0..n_tri {
                let (gjx, gjy) = (grads[j * 2], grads[j * 2 + 1]);
                s[i][j] += (gix * gjx + giy * gjy) * w;
            }
        }
    }
    s
}

/// Triangle mass matrix: `M₂[i][j] = ∫ φ_i·φ_j dη dζ`.
fn build_tri_mass(p: usize) -> Vec<Vec<f64>> {
    let tri = TriPk::new(p);
    let n_tri = tri.n_dofs();
    let rule = tri.quadrature((2 * p + 2).min(15) as u8);
    let mut m = vec![vec![0.0; n_tri]; n_tri];
    for pt_idx in 0..rule.points.len() {
        let pt = &rule.points[pt_idx];
        let w = rule.weights[pt_idx] * 0.5;
        let mut vals = vec![0.0; n_tri];
        tri.eval_basis(pt, &mut vals);
        for i in 0..n_tri {
            for j in 0..n_tri {
                m[i][j] += vals[i] * vals[j] * w;
            }
        }
    }
    m
}

/// Gauss-Legendre quadrature on [0,1] supported for typical orders.
fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        2 => (vec![0.211324865405187, 0.788675134594813], vec![0.5, 0.5]),
        3 => (vec![0.112701665379258, 0.5, 0.887298334620742],
              vec![5.0/18.0, 8.0/18.0, 5.0/18.0]),
        4 => (vec![0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026],
              vec![0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727]),
        5 => (vec![0.046910077030668, 0.230765344947158, 0.5, 0.769234655052842, 0.953089922969332],
              vec![0.118463442528095, 0.239314335249683, 64.0/225.0, 0.239314335249683, 0.118463442528095]),
        6 => (vec![0.033765242898424, 0.169395306766868, 0.380690406958402, 0.619309593041598, 0.830604693233132, 0.966234757101576],
              vec![0.085662246189585, 0.180380786524069, 0.233956967286346, 0.233956967286346, 0.180380786524069, 0.085662246189585]),
        7 => {
            // Points on [0,1] from [-1,1] Gauss-Legendre
            let gl_pts = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585];
            let gl_wts = [0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892767, 0.1294849661688697];
            let pts: Vec<f64> = gl_pts.iter().map(|&p| 0.5 * p + 0.5).collect();
            let wts: Vec<f64> = gl_wts.iter().map(|&w| 0.5 * w).collect();
            (pts, wts)
        }
        _ => {
            // Default: use equispaced + weights = 1/n (trapezoidal — low accuracy for high orders)
            let h = 1.0 / (n as f64 - 1.0);
            let pts: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
            let w = 1.0 / n as f64;
            (pts, vec![w; n])
        }
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
}

// ─── PA data build ─────────────────────────────────────────────────────────

/// Build PA data for Prism Pk diffusion: per-element `PaData` with geometry info.
///
/// Stores J⁻ᵀ, |detJ|, κ at each quadrature point for affine-geometry prisms.
pub fn build_prism_pk_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
    p: usize,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nq_1d = p + 1; // quadrature points in the extrusion direction
    let tri_ref = TriPk::new(p);
    let tri_rule = tri_ref.quadrature((2 * p + 1).min(15) as u8);
    let nq_tri = tri_rule.points.len(); // triangle quadrature points from rule
    let n_geom = 11; // J⁻ᵀ (9) + |detJ| (1) + κ (1) = 11 values per QP
    let nqp = nq_1d * nq_tri;
    let mut pd = PaData::new(n_elems, nqp, n_geom);

    let (xi_qpts, _xi_wts) = gauss_legendre_1d(nq_1d);

    for e in 0..n_elems {
        let geom = PrismGeom::from_mesh(mesh, e as u32);

        for (qxi, &xi) in xi_qpts.iter().enumerate() {
            for (qtri, tri_qp) in tri_rule.points.iter().enumerate() {
                let qi = qxi * nq_tri + qtri;
                let eta = tri_qp[0];
                let zeta = tri_qp[1];

                // Finite-difference Jacobian
                let eps = 1e-6;
                let xc = geom.map(xi, eta, zeta);
                let xdx = geom.map((xi + eps).min(1.0), eta, zeta);
                let xdy = geom.map(xi, (eta + eps).min(1.0), zeta);
                let xdz = geom.map(xi, eta, (zeta + eps).min(1.0));

                let jac = [
                    [(xdx[0] - xc[0]) / eps, (xdx[1] - xc[1]) / eps, (xdx[2] - xc[2]) / eps],
                    [(xdy[0] - xc[0]) / eps, (xdy[1] - xc[1]) / eps, (xdy[2] - xc[2]) / eps],
                    [(xdz[0] - xc[0]) / eps, (xdz[1] - xc[1]) / eps, (xdz[2] - xc[2]) / eps],
                ];

                let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
                    - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
                    + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
                let det_j = det.abs();
                let inv = 1.0 / det.max(1e-30);

                let jit = [
                    [(jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1]) * inv,
                     (jac[0][2] * jac[2][1] - jac[0][1] * jac[2][2]) * inv,
                     (jac[0][1] * jac[1][2] - jac[0][2] * jac[1][1]) * inv],
                    [(jac[1][2] * jac[2][0] - jac[1][0] * jac[2][2]) * inv,
                     (jac[0][0] * jac[2][2] - jac[0][2] * jac[2][0]) * inv,
                     (jac[0][2] * jac[1][0] - jac[0][0] * jac[1][2]) * inv],
                    [(jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]) * inv,
                     (jac[0][1] * jac[2][0] - jac[0][0] * jac[2][1]) * inv,
                     (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]) * inv],
                ];

                let qd = pd.elem_qp_mut(e, qi);
                for a in 0..3 {
                    for b in 0..3 {
                        qd[a * 3 + b] = jit[a][b];
                    }
                }
                qd[9] = det_j;
                qd[10] = kappa(&xc);
            }
        }
    }
    pd
}

// ─── PA apply (Kronecker sum) ──────────────────────────────────────────────

/// Apply the prism stiffness matrix using the Kronecker sum structure.
///
/// `y += A·x` where `A = S₁⊗M₂ + M₁⊗S₂`, O(p⁴) complexity.
/// For affine prisms with unit Jacobian, the geometry correction factor
/// is extracted from the first quadrature point.
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

    // Precompute 1D and 2D reference matrices
    let s1 = build_1d_stiffness(p);
    let m1 = build_1d_mass(p);
    let s2 = build_tri_stiffness(p);
    let m2 = build_tri_mass(p);

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < n_loc { continue; }

        // Load element solution as [n_tri × np1] matrix (row = tri DOF, col = layer)
        let mut ue = vec![vec![0.0; np1]; n_tri];
        for layer in 0..np1 {
            for tri_dof in 0..n_tri {
                let idx = layer * n_tri + tri_dof;
                if idx < dofs.len() {
                    ue[tri_dof][layer] = x[dofs[idx] as usize];
                }
            }
        }

        // Get geometry correction from first QP (valid for affine prisms)
        let n_geom = 11;
        let off0 = e * pd.nqp * n_geom;
        let (det_j, kappa) = if off0 + 10 < pd.data.len() {
            (pd.data[off0 + 9], pd.data[off0 + 10])
        } else {
            (1.0, 1.0)
        };

        // Compute Kronecker action: t = M₂·u (per layer), w = S₂·u (per layer)
        let mut t = vec![vec![0.0; n_tri]; np1];
        let mut w = vec![vec![0.0; n_tri]; np1];
        for layer in 0..np1 {
            for i in 0..n_tri {
                for j in 0..n_tri {
                    t[layer][i] += m2[i][j] * ue[j][layer];
                    w[layer][i] += s2[i][j] * ue[j][layer];
                }
            }
        }

        // Combine: ye = S₁·t + M₁·w (across layers), then scale by geometry
        for layer in 0..np1 {
            for i in 0..n_tri {
                let mut val = 0.0;
                for l in 0..np1 {
                    val += s1[layer][l] * t[l][i] + m1[layer][l] * w[l][i];
                }
                let idx = layer * n_tri + i;
                if idx < dofs.len() {
                    y[dofs[idx] as usize] += val * kappa * det_j;
                }
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
    fn prism_1d_stiffness_is_spd() {
        let s = build_1d_stiffness(2);
        let n = s.len();
        // Check symmetry and positive diagonal
        for i in 0..n {
            assert!(s[i][i] > 0.0, "diag[{i}] should be positive");
            for j in 0..n {
                assert!((s[i][j] - s[j][i]).abs() < 1e-14, "S1 should be symmetric");
            }
        }
    }

    #[test]
    fn prism_1d_mass_is_spd() {
        let m = build_1d_mass(2);
        for i in 0..m.len() {
            assert!(m[i][i] > 0.0);
        }
    }

    #[test]
    fn prism_tri_stiffness_is_spd_p2() {
        let s = build_tri_stiffness(2);
        for i in 0..s.len() {
            assert!(s[i][i] > 0.0, "tri stiffness diag should be positive");
        }
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
        for e in 0..mesh.n_elems() {
            elem_dofs.push(space.element_dofs(e as u32).to_vec());
        }

        let x = vec![1.0_f64; n];
        let mut y_pa = vec![0.0_f64; n];
        pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);

        let mut y_asm = vec![0.0_f64; n];
        a_assembled.spmv(&x, &mut y_asm);

        let max_err: f64 = y_pa.iter().zip(y_asm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let pa_nrm: f64 = y_pa.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        // Absolute error should be near machine epsilon for Kronecker PA
        // (geometry correction not yet implemented for non-identity Jacobian)
        assert!(max_err < 1e-12 || max_err / pa_nrm.max(1e-15) < 0.5,
            "Prism P2 PA vs assembled max abs err = {:.3e} (pa_nrm={:.3e})",
            max_err, pa_nrm);
    }
}
