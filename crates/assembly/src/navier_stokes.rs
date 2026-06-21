//! Incompressible Navier–Stokes assembly utilities.
//!
//! Provides the building blocks for solving the incompressible Navier–Stokes
//! equations with Taylor–Hood elements (P2 velocity / P1 pressure).
//!
//! ## Usage pattern
//! ```rust,ignore
//! // 1. Assemble Stokes blocks using existing integrators
//! let A = Assembler::assemble_bilinear(&vel_space, &[&DiffusionIntegrator { kappa: nu }], 2);
//! let B = assemble_divergence_matrix(&vel_space, &pres_mesh, 2);
//! let rhs = assemble_ns_rhs(...);
//!
//! // 2. Picard iteration: (νA + C(uₖ))·Δv + Bᵀ·Δp = −r,  B·Δv = 0
//! let C = assemble_convection_matrix(&vel_space, &u_k, 2);
//!
//! // 3. Solve saddle-point system with StokesPrecond
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;
use crate::assembler::Assembler;
use crate::standard::{DiffusionIntegrator, ConvectionIntegrator};

/// Assemble the convection matrix C(u₀) for Oseen iteration:
/// C[i,j] = ∫ (u₀·∇φⱼ)·φᵢ dx  (nonlinear convection linearised at u₀)
pub fn assemble_convection_matrix<M: MeshTopology + Clone>(
    vel_space: &VectorH1Space<M>,
    u_0: &[f64],
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_vel = vel_space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_vel, n_vel);
    let mesh = vel_space.mesh();

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = super::dg_advection::ref_elem_vol(elem_type, vel_space.order());
        let n_ldofs = ref_elem.n_dofs();
        let n_vec = n_ldofs * mesh.dim() as usize;
        let quad = ref_elem.quadrature(quad_order);
        let dofs: Vec<usize> = vel_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = super::dg_advection::simplex_jac(mesh, nodes, mesh.dim() as usize);
        let jit = jac.try_inverse().unwrap().transpose();
        let dim = mesh.dim() as usize;

        let mut u_elem = vec![0.0_f64; n_vec];
        for (k, &dof) in dofs.iter().enumerate() { u_elem[k] = u_0[dof]; }

        let mut k_elem = vec![0.0_f64; n_vec * n_vec];
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut gref = vec![0.0_f64; n_ldofs * dim];
        let mut gphys = vec![0.0_f64; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);
            ref_elem.eval_grad_basis(xi, &mut gref);
            super::dg_advection::xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            // Interpolate u₀ at QP
            let mut u0_at_qp = [0.0_f64; 3];
            for k in 0..n_ldofs {
                for d in 0..dim { u0_at_qp[d] += u_elem[k * dim + d] * phi[k]; }
            }

            for k in 0..n_ldofs {
                for a in 0..dim {
                    let row = k * dim + a;
                    for l in 0..n_ldofs {
                        for b in 0..dim {
                            let col = l * dim + b;
                            // ∫ (u₀·∇φⱼ)·φᵢ = ∫ φᵢ·Σ_d u₀_d·∂φⱼ/∂x_d
                            let mut u_dot_grad = 0.0;
                            for d in 0..dim { u_dot_grad += u0_at_qp[d] * gphys[l * dim + d]; }
                            if a == b {
                                k_elem[row * n_vec + col] += w * phi[k] * u_dot_grad;
                            }
                        }
                    }
                }
            }
        }
        coo.add_element_matrix(&dofs, &k_elem);
    }
    coo.into_csr()
}

/// Compute the divergence operator matrix B: B[q,j] = -∫ ψ_q·(∇·φ_j) dx
pub fn assemble_divergence_matrix<M: MeshTopology + Clone>(
    vel_space: &VectorH1Space<M>,
    pres_mesh: &M,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let dim = pres_mesh.dim() as usize;
    let n_pres = pres_mesh.n_nodes();
    let n_vel = vel_space.n_dofs();
    let mut coo = CooMatrix::<f64>::new(n_pres, n_vel);

    for e in pres_mesh.elem_iter() {
        let elem_type = pres_mesh.element_type(e);
        let ref_v = super::dg_advection::ref_elem_vol(elem_type, vel_space.order());
        let ref_p = super::dg_advection::ref_elem_vol(elem_type, 1); // P1 pressure
        let n_v = ref_v.n_dofs();
        let n_p = ref_p.n_dofs();
        let quad = ref_v.quadrature(quad_order);
        let dofs_v: Vec<usize> = vel_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = pres_mesh.element_nodes(e);
        let (jac, det_j) = super::dg_advection::simplex_jac(pres_mesh, nodes, dim);
        let jit = jac.try_inverse().unwrap().transpose();

        let mut b_elem = vec![0.0_f64; n_p * n_v * dim];
        // 2D: b_elem[p, v*2+comp] = -∫ ψ_p · (∂φ_v/∂x_comp) dx
        // Actually B has rows = pressure DOF, cols = velocity DOF
        let mut b_mat = vec![0.0_f64; n_p * n_v * dim];
        let mut phi_v = vec![0.0_f64; n_v];
        let mut phi_p = vec![0.0_f64; n_p];
        let mut gref = vec![0.0_f64; n_v * dim];
        let mut gphys = vec![0.0_f64; n_v * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            ref_v.eval_basis(xi, &mut phi_v);
            ref_p.eval_basis(xi, &mut phi_p);
            ref_v.eval_grad_basis(xi, &mut gref);
            super::dg_advection::xform_grads(&jit, &gref, &mut gphys, n_v, dim);

            for p in 0..n_p {
                for v in 0..n_v {
                    for d in 0..dim {
                        // B[p, v*dim+d] = -∫ ψ_p · (∂φ_v/∂x_d) dx
                        let col_idx = p * n_v * dim + v * dim + d;
                        b_mat[col_idx] += w * phi_p[p] * gphys[v * dim + d];
                    }
                }
            }
        }

        // Scatter to global
        for p in 0..n_p {
            let pres_dof = p; // P1: pressure DOF = node index
            if pres_dof >= n_pres { continue; }
            for v in 0..n_v {
                for d in 0..dim {
                    let vel_dof = dofs_v[v * dim + d];
                    let idx = p * n_v * dim + v * dim + d;
                    coo.add(pres_dof as usize, vel_dof as usize, -b_mat[idx]);
                }
            }
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::vector_h1::VectorH1Space;

    #[test]
    fn divergence_matrix_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let vel_space = VectorH1Space::new(mesh.clone(), 1, 2);
        let b = assemble_divergence_matrix(&vel_space, &mesh, 2);
        let mut sum = 0.0;
        for i in 0..b.nrows.min(10) {
            for j in 0..b.ncols.min(10) { sum += b.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "Divergence matrix B should be non-zero");
    }

    #[test]
    fn convection_matrix_nonzero() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let vel_space = VectorH1Space::new(mesh.clone(), 1, 2);
        let n = vel_space.n_dofs();
        // Uniform flow u=(1,0)
        let mut u0 = vec![0.0_f64; n];
        for i in 0..n { u0[i] = if i % 2 == 0 { 1.0 } else { 0.0 }; }
        let c = assemble_convection_matrix(&vel_space, &u0, 2);
        let mut sum = 0.0;
        for i in 0..c.nrows.min(10) {
            for j in 0..c.ncols.min(10) { sum += c.get(i, j).abs(); }
        }
        assert!(sum > 0.0, "Convection matrix C should be non-zero");
    }
}
