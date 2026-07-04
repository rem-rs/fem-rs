//! Mortar (weak) coupling for non-conforming multi-patch IGA.
//!
//! Given two NURBS patches `A` and `B` with possibly different knot vectors
//! along a shared interface `Γ`, the Mortar method enforces:
//!
//! ```text
//! ∫_Γ (u_A - u_B) φ dΓ = 0    ∀ φ ∈ V_h^mortar
//! ```
//!
//! producing the saddle-point system:
//! ```text
//! [K_A   0   B_A^T] [u_A]   [f_A]
//! [ 0   K_B -B_B^T] [u_B] = [f_B]
//! [B_A -B_B   0   ] [ λ ]   [ 0 ]
//! ```

use fem_element::iga::NurbsPatch2D;
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};

/// A Mortar coupling between two edges of two NURBS patches.
pub struct MortarCoupling2D {
    pub patch_a: usize,
    pub edge_a: usize,
    pub patch_b: usize,
    pub edge_b: usize,
}

/// Return DOF indices along an edge of a `nu × nv` tensor-product patch.
fn edge_dofs_2d(nu: usize, nv: usize, edge: usize) -> Vec<usize> {
    match edge {
        0 => (0..nu).collect(),
        1 => (0..nv).map(|j| (j + 1) * nu - 1).collect(),
        2 => (0..nu).map(|i| (nv - 1) * nu + i).rev().collect(),
        3 => (0..nv).map(|j| j * nu).rev().collect(),
        _ => vec![],
    }
}

/// Parametric point `[u,v]` on a patch corresponding to edge parameter `t ∈ [0,1]`.
fn edge_uv(_nu: usize, _nv: usize, edge: usize, t: f64) -> [f64; 2] {
    match edge {
        0 => [t, 0.0],
        1 => [1.0, t],
        2 => [1.0 - t, 1.0],
        3 => [0.0, 1.0 - t],
        _ => [0.0, 0.0],
    }
}

/// Gauss-Legendre nodes and weights on `[0, 1]`.
fn gl_01(n: u8) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.5], vec![1.0]),
        2 => (vec![0.21132487, 0.78867513], vec![0.5, 0.5]),
        3 => (vec![0.11270167, 0.5, 0.88729833],
              vec![0.27777778, 0.44444444, 0.27777778]),
        4 => (vec![0.06943184, 0.33000948, 0.66999052, 0.93056816],
              vec![0.17392742, 0.32607258, 0.32607258, 0.17392742]),
        _ => panic!("Mortar: unsupported GL order {n}"),
    }
}

/// Build the Mortar constraint matrices for a single edge coupling.
///
/// Returns `(B_a, B_b)` where `B_a` maps from patch A's side to the mortar space.
pub fn build_mortar_constraint(
    patch_a: &NurbsPatch2D,
    patch_b: &NurbsPatch2D,
    coupling: &MortarCoupling2D,
    quad_order: u8,
) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let nu_a = patch_a.n_u();
    let nv_a = patch_a.n_v();
    let nu_b = patch_b.n_u();
    let nv_b = patch_b.n_v();
    let n_dofs_a = nu_a * nv_a;
    let n_dofs_b = nu_b * nv_b;

    let edge_dofs_a = edge_dofs_2d(nu_a, nv_a, coupling.edge_a);
    let edge_dofs_b = edge_dofs_2d(nu_b, nv_b, coupling.edge_b);
    let n_mortar = edge_dofs_a.len().max(edge_dofs_b.len());

    let (gl_pts, gl_wts) = gl_01(quad_order);

    let mut coo_a = CooMatrix::new(n_mortar, n_dofs_a);
    let mut coo_b = CooMatrix::new(n_mortar, n_dofs_b);

    let mut basis_a = vec![0.0; n_dofs_a];
    let mut basis_b = vec![0.0; n_dofs_b];

    for mi in 0..n_mortar {
        let xi_m = if n_mortar == 1 { 0.5 } else { mi as f64 / (n_mortar - 1) as f64 };

        for (&gp, &gw) in gl_pts.iter().zip(gl_wts.iter()) {
            let t = gp;
            let w = gw;

            // Mortar hat function
            let m_val = if n_mortar == 1 {
                1.0
            } else {
                let dist = (t - xi_m).abs() * (n_mortar - 1) as f64;
                if dist <= 1.0 { 1.0 - dist } else { continue; }
            };

            patch_a.eval_basis(&edge_uv(nu_a, nv_a, coupling.edge_a, t), &mut basis_a);
            patch_b.eval_basis(&edge_uv(nu_b, nv_b, coupling.edge_b, t), &mut basis_b);

            for (&dof_a, ba) in edge_dofs_a.iter().zip(edge_dofs_a.iter().map(|&d| basis_a[d])) {
                let v = ba * m_val * w;
                if v.abs() > 1e-15 { coo_a.add(mi, dof_a, v); }
            }
            for (&dof_b, bb) in edge_dofs_b.iter().zip(edge_dofs_b.iter().map(|&d| basis_b[d])) {
                let v = bb * m_val * w;
                if v.abs() > 1e-15 { coo_b.add(mi, dof_b, -v); }
            }
        }
    }

    (coo_a.into_csr(), coo_b.into_csr())
}

/// Build the coupled saddle-point system.
pub fn build_mortar_system(
    k_a: &CsrMatrix<f64>,
    k_b: &CsrMatrix<f64>,
    b_a: &CsrMatrix<f64>,
    b_b: &CsrMatrix<f64>,
    f_a: &[f64],
    f_b: &[f64],
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n_a = k_a.nrows;
    let n_b = k_b.nrows;
    let n_m = b_a.nrows;
    let n_total = n_a + n_b + n_m;

    let mut coo = CooMatrix::new(n_total, n_total);
    let mut rhs = vec![0.0; n_total];

    for i in 0..n_a {
        let s = k_a.row_ptr[i];
        let e = k_a.row_ptr[i + 1];
        for nz in s..e { coo.add(i, k_a.col_idx[nz] as usize, k_a.values[nz]); }
        rhs[i] = f_a[i];
    }
    for i in 0..n_b {
        let s = k_b.row_ptr[i];
        let e = k_b.row_ptr[i + 1];
        for nz in s..e { coo.add(n_a + i, n_a + k_b.col_idx[nz] as usize, k_b.values[nz]); }
        rhs[n_a + i] = f_b[i];
    }

    for i in 0..n_m {
        let (s, e) = (b_a.row_ptr[i], b_a.row_ptr[i + 1]);
        for nz in s..e {
            let j = b_a.col_idx[nz] as usize;
            let v = b_a.values[nz];
            coo.add(n_a + n_b + i, j, v);
            coo.add(j, n_a + n_b + i, v);
        }
        let (s, e) = (b_b.row_ptr[i], b_b.row_ptr[i + 1]);
        for nz in s..e {
            let j = b_b.col_idx[nz] as usize;
            let v = b_b.values[nz];
            coo.add(n_a + n_b + i, n_a + j, v);
            coo.add(n_a + j, n_a + n_b + i, v);
        }
    }

    (coo.into_csr(), rhs)
}
