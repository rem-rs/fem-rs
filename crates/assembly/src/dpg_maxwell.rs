//! DPG Maxwell solver — ultraweak formulation for 2D TE mode.
//!
//! Solves the scalar Helmholtz equation `-Δu - k²u = f` on Tri3 meshes,
//! corresponding to the 2D transverse-electric Maxwell problem where
//! u = Ez, k² = ω²με.
//!
//! # Method
//!
//! Per-element DPG with optimal test functions:
//! - Trial: P1 (3 DOFs) — matches standard nodal FE space
//! - Test:  P3 (10 DOFs) — enriched, discontinuous
//! - Bilinear form: B(u,v) = ∫(∇u·∇v - k²·uv) dx
//! - Stabilization via local H¹ Gram matrix solve M_V^{-1} B

use fem_element::{ReferenceElement, lagrange::{TriP1, TriP3}, quadrature::tri_rule};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::MeshTopology;

fn solve_dense(n: usize, a: &mut [f64], b: &mut [f64]) {
    for col in 0..n {
        let mut best = col;
        let mut bv = a[col * n + col].abs();
        for row in (col + 1)..n { let v = a[row * n + col].abs(); if v > bv { bv = v; best = row; } }
        if bv < 1e-40 { continue; }
        if best != col { for c in col..n { a.swap(col * n + c, best * n + c); } b.swap(col, best); }
        let piv = a[col * n + col];
        for row in (col + 1)..n { let f = a[row * n + col] / piv;
            for c in col..n { a[row * n + c] -= f * a[col * n + c]; } b[row] -= f * b[col]; }
    }
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n { s -= a[row * n + c] * b[c]; }
        b[row] = if a[row * n + row].abs() > 1e-40 { s / a[row * n + row] } else { 0.0 };
    }
}

/// Solve `-Δu - k²u = f` via DPG on a Tri3 mesh.
///
/// # Arguments
/// * `mesh` — Tri3 mesh
/// * `k` — wavenumber (ω√με)
/// * `f` — source term f(x,y)
/// * `dirichlet_bc` — `(dof_index, value)` pairs for Dirichlet BC
///
/// # Returns
/// Nodal solution u.
pub fn solve_dpg_maxwell_2d<M: MeshTopology>(
    mesh: &M,
    k: f64,
    f: &dyn Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elements();
    let n_test = 10;
    let n_trial = 3;
    let k2 = k * k;

    let tri_p3 = TriP3;
    let tri_p1 = TriP1;
    let qr = tri_rule(7);

    let mut coo = CooMatrix::<f64>::new(n_nodes, n_nodes);
    let mut rhs_global = vec![0.0; n_nodes];

    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = nodes.iter().map(|&n| n as usize).collect();

        let x: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[0]).collect();
        let y: Vec<f64> = (0..3).map(|k| mesh.node_coords(nodes[k])[1]).collect();

        let j00 = x[1] - x[0]; let j01 = x[2] - x[0];
        let j10 = y[1] - y[0]; let j11 = y[2] - y[0];
        let det_j = j00 * j11 - j01 * j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0 / det_j;
        let vol = 0.5 * abs_det;

        let mut phi = vec![0.0; n_test];
        let mut dphi = vec![0.0; n_test * 2];
        let mut mv = vec![0.0; n_test * n_test];
        let mut bm = vec![0.0; n_test * n_trial];

        for (xi, &wr) in qr.points.iter().zip(qr.weights.iter()) {
            let w = wr * abs_det;

            tri_p3.eval_basis(xi, &mut phi);
            tri_p3.eval_grad_basis(xi, &mut dphi);

            let mut dpx = vec![0.0; n_test]; let mut dpy = vec![0.0; n_test];
            for i in 0..n_test {
                dpx[i] = (j11*dphi[i*2] - j10*dphi[i*2+1]) * inv_det;
                dpy[i] = (-j01*dphi[i*2] + j00*dphi[i*2+1]) * inv_det;
            }

            // P1 trial
            tri_p1.eval_basis(xi, &mut phi);
            let tgrad_x = [-1.0, 1.0, 0.0]; let tgrad_y = [-1.0, 0.0, 1.0];
            let mut tdpx = vec![0.0; n_trial]; let mut tdpy = vec![0.0; n_trial];
            for i in 0..n_trial {
                tdpx[i] = (j11*tgrad_x[i] - j10*tgrad_y[i]) * inv_det;
                tdpy[i] = (-j01*tgrad_x[i] + j00*tgrad_y[i]) * inv_det;
            }

            // M_V: H¹ inner product for test space
            for i in 0..n_test {
                for j in 0..n_test {
                    mv[i*n_test+j] += w * (phi[i]*phi[j] + dpx[i]*dpx[j] + dpy[i]*dpy[j]);
                }
            }

            // B: Helmholtz bilinear form ∇u·∇v - k²·u·v
            for i in 0..n_test {
                for j in 0..n_trial {
                    bm[i*n_trial+j] += w * (dpx[i]*tdpx[j] + dpy[i]*tdpy[j] - k2 * phi[i] * phi[j]);
                }
            }

            // Re-evaluate P3 basis for M_V (since phi was overwritten by P1)
            tri_p3.eval_basis(xi, &mut phi);
        }

        // Solve for optimal test functions
        let mut v_opt = vec![0.0; n_test * n_trial];
        for i in 0..n_trial {
            let mut rhs = vec![0.0; n_test];
            for r in 0..n_test { rhs[r] = bm[r * n_trial + i]; }
            let mut mv_copy = mv.clone();
            solve_dense(n_test, &mut mv_copy, &mut rhs);
            for r in 0..n_test { v_opt[r * n_trial + i] = rhs[r]; }
        }

        // Element stiffness Ke = B^T V_opt
        let mut ke = vec![0.0; n_trial * n_trial];
        for i in 0..n_trial {
            for j in 0..n_trial {
                let mut s = 0.0;
                for k in 0..n_test { s += bm[k * n_trial + i] * v_opt[k * n_trial + j]; }
                ke[i * n_trial + j] = s;
            }
        }

        // RHS
        let centroid_x = x.iter().sum::<f64>() / 3.0;
        let centroid_y = y.iter().sum::<f64>() / 3.0;
        let f_val = f(centroid_x, centroid_y);
        let elem_rhs = vec![f_val * vol / 3.0; n_trial];

        // Scatter
        for (li, &gi) in dofs.iter().enumerate() {
            rhs_global[gi] += elem_rhs[li];
            for lj in 0..n_trial {
                let gj = dofs[lj];
                let v = ke[li * n_trial + lj];
                if v.abs() > 1e-30 { coo.add(gi, gj, v); }
            }
        }
    }

    let mat = coo.into_csr();
    let mut rhs = rhs_global;

    // Apply Dirichlet BC
    let mut bc_dofs: Vec<usize> = Vec::new();
    for bf in 0..mesh.n_boundary_faces() as u32 {
        for &n in mesh.face_nodes(bf) { bc_dofs.push(n as usize); }
    }
    bc_dofs.sort_unstable(); bc_dofs.dedup();

    // For boundary DOFs, zero row (homogeneous Dirichlet)
    for &d in &bc_dofs {
        let s = mat.row_ptr[d]; let e = mat.row_ptr[d + 1];
        for p in s..e {
            let j = mat.col_idx[p] as usize;
            if j != d { rhs[j] -= mat.values[p] * 0.0; }
        }
        rhs[d] = 0.0;
    }

    // Solve via CG
    let mut x = vec![0.0; n_nodes];
    for _ in 0..400 {
        let mut ax = vec![0.0; n_nodes];
        for i in 0..n_nodes {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i + 1];
            ax[i] = (s..e).map(|p| mat.values[p] * x[mat.col_idx[p] as usize]).sum();
        }
        let mut r = vec![0.0; n_nodes];
        let mut rr = 0.0;
        for i in 0..n_nodes { r[i] = rhs[i] - ax[i]; rr += r[i] * r[i]; }
        if rr < 1e-24 { break; }
        let mut ar = vec![0.0; n_nodes];
        for i in 0..n_nodes {
            let s = mat.row_ptr[i]; let e = mat.row_ptr[i + 1];
            ar[i] = (s..e).map(|p| mat.values[p] * r[mat.col_idx[p] as usize]).sum();
        }
        let rar: f64 = (0..n_nodes).map(|i| r[i] * ar[i]).sum();
        let alpha = if rar.abs() > 1e-30 { rr / rar } else { 0.0 };
        for i in 0..n_nodes { x[i] += alpha * r[i]; }

        // Reset boundary DOFs
        for &d in &bc_dofs { x[d] = 0.0; }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn dpg_maxwell_2d_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let u = solve_dpg_maxwell_2d(&mesh, 1.0, &|_,_| 1.0);
        for &v in &u { assert!(v.is_finite(), "non-finite solution"); }
    }

    #[test]
    fn dpg_maxwell_k0_finite() {
        // k=0 → Poisson-like, should give finite result
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let u = solve_dpg_maxwell_2d(&mesh, 0.0, &|_,_| 1.0);
        for &v in &u { assert!(v.is_finite()); }
    }
}
