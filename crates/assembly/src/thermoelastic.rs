//! Thermoelastic coupling utilities.
//!
//! Provides thermal expansion RHS assembly, heat equation assembly, and
//! a staggered thermoelastic solve driver.
//!
//! ## Governing equations
//! - Heat: `ρ·c·∂T/∂t - k·∇²T = Q`
//! - Elasticity: `∇·σ = f` with `σ = C:(ε - α·ΔT·I)`
//!
//! ## Staggered coupling
//! 1. Solve heat equation → temperature field T
//! 2. Compute thermal load `f_th = ∫ (3λ+2μ)·α·(T-T₀)·tr(B) dx`
//! 3. Solve elasticity K·u = f_th → displacement u

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_space::VectorH1Space;
use fem_solver::SolverConfig;

use crate::assembler::Assembler;
use crate::standard::{DiffusionIntegrator, ElasticityIntegrator};
use crate::dg_advection::{ref_elem_vol, simplex_jac, xform_grads};

/// Assemble the thermal expansion right-hand side vector.
///
/// `f_i = ∫ (3λ + 2μ)·α·(T - T₀)·(∇·φ_i) dx`
///
/// where `φ_i` is the i-th vector test function in the displacement space
/// (VectorH1Space with interleaved DOFs), `T` is the temperature field,
/// `T₀` is the stress-free reference temperature, `α` is the thermal
/// expansion coefficient, and `λ, μ` are the Lamé parameters.
///
/// Returns a vector of length `n_dofs` (same as the displacement space).
pub fn assemble_thermal_expansion_rhs<M: MeshTopology + Clone>(
    mesh: &M,
    disp_space: &VectorH1Space<M>,
    temp: &[f64],
    t_ref: f64,
    alpha: f64,
    lambda: f64,
    mu: f64,
    quad_order: u8,
) -> Vec<f64> {
    let dim = mesh.dim() as usize;
    let n_dofs = disp_space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    let pres_space = H1Space::new(mesh.clone(), 1);

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let order = disp_space.order();
        let re = ref_elem_vol(et, order);
        let n_ldofs = re.n_dofs();
        let n_vec = n_ldofs * dim;
        let quad = re.quadrature(quad_order);
        let dofs: Vec<usize> = disp_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let temp_dofs: Vec<usize> = pres_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        let jit = jac.try_inverse().unwrap().transpose();

        let mut f_elem = vec![0.0; n_vec];
        let mut phi = vec![0.0; n_ldofs];
        let mut gref = vec![0.0; n_ldofs * dim];
        let mut gphys = vec![0.0; n_ldofs * dim];

        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            // Interpolate T at QP
            let temp_qp: f64 = temp_dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| temp[d] * p).sum();
            let delta_t = temp_qp - t_ref;

            // Thermal expansion coefficient: β = α·(3λ + 2μ)
            let beta = alpha * (3.0 * lambda + 2.0 * mu);

            for k in 0..n_ldofs {
                // div(φ_k) = Σ_d gphys[k*dim + d]
                let div_phi = (0..dim).map(|d| gphys[k * dim + d]).sum::<f64>();
                let val = w * beta * delta_t * div_phi;
                for a in 0..dim {
                    f_elem[k * dim + a] += val;
                }
            }
        }

        for (i, &gi) in dofs.iter().enumerate() {
            rhs[gi] += f_elem[i];
        }
    }

    rhs
}

/// Assemble the steady heat conduction matrix and RHS.
///
/// Matrix: `K_ij = ∫ k·∇φ_i·∇φ_j dx`
/// RHS:    `f_i = ∫ φ_i·Q dx` (source)
///
/// Returns `(K, rhs)`. Only volumetric heat source is included;
/// Neumann BC must be added separately by modifying the RHS.
pub fn assemble_heat_system<M: MeshTopology + Clone>(
    mesh: &M,
    kappa: f64,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let k = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa }], quad_order,
    );

    (k, vec![0.0; n])
}

/// Solve a staggered thermoelastic problem.
///
/// 1. Solve the steady heat equation for temperature T.
/// 2. Compute the thermal expansion RHS from T.
/// 3. Solve linear elasticity with thermal load.
///
/// Returns `(T, u)`.
pub fn solve_thermoelastic_staggered<M: MeshTopology + Clone>(
    mesh: &M,
    kappa: f64,
    alpha: f64,
    lambda: f64,
    mu: f64,
    t_ref: f64,
    temp_bc_dofs: &[usize],
    temp_bc_vals: &[f64],
    disp_bc_dofs: &[usize],
    disp_bc_vals: &[f64],
    quad_order: u8,
) -> (Vec<f64>, Vec<f64>) {
    let dim = mesh.dim() as usize;
    let pres_space = H1Space::new(mesh.clone(), 1);
    let disp_space = VectorH1Space::new(mesh.clone(), 1, dim as u8);
    let n_p = pres_space.n_dofs();
    let n_u = disp_space.n_dofs();
    let cfg = SolverConfig {
        rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false,
        ..SolverConfig::default()
    };

    // 1. Solve heat equation
    let (mut k_t, mut rhs_t) = assemble_heat_system(mesh, kappa, quad_order);
    for (&dof, &val) in temp_bc_dofs.iter().zip(temp_bc_vals.iter()) {
        k_t.apply_dirichlet_symmetric(dof, val, &mut rhs_t);
    }
    let mut temp = vec![0.0; n_p];
    fem_solver::solve_pcg_jacobi(&k_t, &rhs_t, &mut temp, &cfg)
        .expect("heat solve failed");

    // 2. Assemble elasticity matrix
    let mut k_u = Assembler::assemble_bilinear(
        &disp_space, &[&ElasticityIntegrator { lambda, mu }], quad_order,
    );

    // 3. Compute thermal expansion RHS
    let mut rhs_u = assemble_thermal_expansion_rhs(
        mesh, &disp_space, &temp, t_ref, alpha, lambda, mu, quad_order,
    );

    // 4. Apply Dirichlet BC on displacement
    for (&dof, &val) in disp_bc_dofs.iter().zip(disp_bc_vals.iter()) {
        k_u.apply_dirichlet_symmetric(dof, val, &mut rhs_u);
    }

    // 5. Solve elasticity
    let mut u = vec![0.0; n_u];
    fem_solver::solve_pcg_jacobi(&k_u, &rhs_u, &mut u, &cfg)
        .expect("elasticity solve failed");

    (temp, u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn thermal_expansion_rhs_zero_at_reference_temp() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let quad_order = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let n_p = H1Space::new(mesh.clone(), 1).n_dofs();
        let temp = vec![0.0; n_p]; // T = T₀ = 0

        let rhs = assemble_thermal_expansion_rhs(
            &mesh, &disp_space, &temp, 0.0, 1.0e-5, 1.0, 0.5, quad_order,
        );

        let norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-14, "thermal RHS should be zero at reference temp, got {norm}");
    }

    #[test]
    fn thermal_expansion_rhs_nonzero_with_delta_t() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let quad_order = 2;
        let disp_space = VectorH1Space::new(mesh.clone(), 1, dim);
        let n_p = H1Space::new(mesh.clone(), 1).n_dofs();
        let temp = vec![100.0; n_p]; // uniform ΔT = 100

        let rhs = assemble_thermal_expansion_rhs(
            &mesh, &disp_space, &temp, 0.0, 1.0e-5, 1.0, 0.5, quad_order,
        );

        let norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0, "thermal RHS should be non-zero with ΔT > 0");
    }

    #[test]
    fn heat_system_assembles() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let (k, rhs) = assemble_heat_system(&mesh, 1.0, 2);
        assert_eq!(k.nrows, mesh.n_nodes());
        assert_eq!(rhs.len(), mesh.n_nodes());
        assert!(k.nnz() > 0);
    }

    #[test]
    fn thermoelastic_staggered_solves() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let kappa = 1.0;
        let alpha = 1.0e-5;
        let lambda = 121154.0;
        let mu = 80769.0;
        let t_ref = 0.0;
        let quad_order = 2;

        // Dirichlet BC: T = 100 on left (tag 4), T = 0 on right (tag 2)
        let pres_space = H1Space::new(mesh.clone(), 1);
        let temp_bc_dofs: Vec<usize> = (0..pres_space.n_dofs())
            .filter(|&d| {
                let x = pres_space.dof_manager().dof_coord(d as u32);
                x[0] < 0.01 || x[0] > 0.99
            })
            .collect();
        let temp_bc_vals: Vec<f64> = temp_bc_dofs.iter().map(|&d| {
            let x = pres_space.dof_manager().dof_coord(d as u32);
            if x[0] < 0.01 { 100.0 } else { 0.0 }
        }).collect();

        // Fix displacement at bottom
        let disp_space = VectorH1Space::new(mesh.clone(), 1, 2);
        let dim = 2;
        let disp_bc_dofs: Vec<usize> = (0..mesh.n_nodes() as u32)
            .filter(|&n| mesh.node_coords(n)[1] < 0.01)
            .flat_map(|n| {
                let idx = n as usize;
                vec![idx * dim, idx * dim + 1]
            })
            .collect();
        let disp_bc_vals = vec![0.0; disp_bc_dofs.len()];

        let (temp, u) = solve_thermoelastic_staggered(
            &mesh, kappa, alpha, lambda, mu, t_ref,
            &temp_bc_dofs, &temp_bc_vals,
            &disp_bc_dofs, &disp_bc_vals, quad_order,
        );

        assert_eq!(temp.len(), pres_space.n_dofs());
        assert_eq!(u.len(), disp_space.n_dofs());

        // Temperature should be between 0 and 100
        for &t in &temp {
            assert!(t >= -1.0 && t <= 101.0, "temperature out of range: {t}");
        }

        // Displacement should be non-zero (thermal expansion)
        let u_norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(u_norm > 0.0, "thermal expansion should produce non-zero displacement");
    }
}
