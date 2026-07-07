//! fluids_navier_transient — time-dependent 2D incompressible flow (Stokes → NS).
//!
//! BDF1 time stepping with Picard linearization on driven-cavity.
//! Taylor-Hood P2/P1. Uses scalar mass matrix applied component-wise.
//! Analogous to MFEM miniapp `fluids/navier`.
//!
//! Usage:
//!   cargo run --example fluids_navier_transient --release -- 16 100

use std::time::Instant;
use fem_assembly::{
    Assembler, standard::{VectorDiffusionIntegrator, MassIntegrator},
    physics::navier_stokes::{assemble_convection_matrix, assemble_divergence_matrix},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_gmres, SolverConfig, BlockSystem};
use fem_space::{H1Space, VectorH1Space, fe_space::FESpace};

/// Build block-diagonal mass matrix for VectorH1Space (2 components) from scalar mass.
fn build_vector_mass(n_scalar: usize, mesh: &Mesh<2>, quad_order: u8) -> CsrMatrix<f64> {
    let scalar_space = fem_space::H1Space::new(mesh.clone(), 2);
    let scalar_mass = Assembler::assemble_bilinear(
        &scalar_space, &[&MassIntegrator { rho: 1.0 }], quad_order);
    let mut coo = CooMatrix::new(n_scalar * 2, n_scalar * 2);
    for r in 0..n_scalar { for k in scalar_mass.row_ptr[r]..scalar_mass.row_ptr[r+1] {
        let c = scalar_mass.col_idx[k] as usize;
        let v = scalar_mass.values[k];
        coo.add(r, c, v);           // x-component
        coo.add(r + n_scalar, c + n_scalar, v);  // y-component
    }}
    coo.into_csr()
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let re: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100.0);
    let nu = 1.0 / re.max(1.0);
    let dt = 0.02;
    let steps = 50;
    let t0 = Instant::now();

    let mesh = Mesh::<2>::unit_square_tri(n);
    let vs = VectorH1Space::new(mesh.clone(), 2, 2);
    let ps = H1Space::new(mesh, 1);
    let n_u = vs.n_dofs(); let n_p = ps.n_dofs();
    let n_scalar = vs.n_scalar_dofs();

    let diff = VectorDiffusionIntegrator { kappa: nu };
    let a_diff = Assembler::assemble_bilinear(&vs, &[&diff], 5);
    let m_vec = build_vector_mass(n_scalar, vs.mesh(), 5);
    let b_mat = assemble_divergence_matrix(&vs, vs.mesh(), 5);
    let bt = b_mat.transpose();
    let zero_p = vec![0.0_f64; n_p];

    let mut u = vec![0.0_f64; n_u];
    let mut p = vec![0.0_f64; n_p];
    let cfg = SolverConfig { rtol: 1e-4, atol: 0.0, max_iter: 500, verbose: false, ..Default::default() };
    let inv_dt = 1.0 / dt;

    for step in 0..steps {
        let t = step as f64 * dt;
        // Convection at current velocity (omit for first step or low Re)
        let c_conv = assemble_convection_matrix(&vs, &u, 5);

        // Oseen operator: (1/dt)·M + ν·A_diff + C(u)
        let mut coo = CooMatrix::new(n_u, n_u);
        for r in 0..n_u { for k in m_vec.row_ptr[r]..m_vec.row_ptr[r+1] {
            coo.add(r, m_vec.col_idx[k] as usize, inv_dt * m_vec.values[k]);
        }}
        for r in 0..n_u { for k in a_diff.row_ptr[r]..a_diff.row_ptr[r+1] {
            coo.add(r, a_diff.col_idx[k] as usize, a_diff.values[k]);
        }}
        for r in 0..n_u { for k in c_conv.row_ptr[r]..c_conv.row_ptr[r+1] {
            coo.add(r, c_conv.col_idx[k] as usize, c_conv.values[k]);
        }}
        let mut a_oseen = coo.into_csr();

        // RHS = M·u_k / dt
        let mut rhs = vec![0.0_f64; n_u];
        for r in 0..n_u { for k in m_vec.row_ptr[r]..m_vec.row_ptr[r+1] {
            rhs[r] += inv_dt * m_vec.values[k] * u[m_vec.col_idx[k] as usize];
        }}

        // Dirichlet BC: top lid
        let ux_lid = if t < 0.5 { t / 0.5 } else { 1.0 };
        for &tag in &[1i32, 5, 6] {
            for d in fem_space::constraints::boundary_dofs(vs.mesh(), vs.scalar_dof_manager(), &[tag]) {
                a_oseen.apply_dirichlet_row_zeroing(d as usize, 0.0, &mut rhs);
                a_oseen.apply_dirichlet_row_zeroing(d as usize + n_scalar, 0.0, &mut rhs);
            }
        }
        for d in fem_space::constraints::boundary_dofs(vs.mesh(), vs.scalar_dof_manager(), &[2i32]) {
            a_oseen.apply_dirichlet_row_zeroing(d as usize, ux_lid, &mut rhs);
            a_oseen.apply_dirichlet_row_zeroing(d as usize + n_scalar, 0.0, &mut rhs);
        }

        let sys = BlockSystem { a: a_oseen, bt: bt.clone(), b: b_mat.clone(), c: None };
        let flat_mat = sys.to_flat_csr();
        let mut rhs_flat = vec![0.0_f64; n_u + n_p];
        rhs_flat[..n_u].copy_from_slice(&rhs);
        let mut x = vec![0.0_f64; n_u + n_p];
        let res = match solve_gmres(&flat_mat, &rhs_flat, &mut x, 50, &cfg) {
            Ok(r) => { u.copy_from_slice(&x[..n_u]); p.copy_from_slice(&x[n_u..]); r }
            Err(e) => { eprintln!("  GMRES failed at step {step}: {e}"); break; }
        };

        if step % 10 == 0 || step == steps - 1 {
            let ke: f64 = u.iter().map(|v| v * v).sum::<f64>() / n_u as f64;
            println!("  step {step:>3}  t={t:.3}  ke={ke:.4e}  iters={}", res.iterations);
        }
    }

    println!("=== fluids_navier_transient: driven cavity Re={re:.0} ===");
    println!("  Mesh: {n}x{n}, DOFs: vel={n_u} pres={n_p}");
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::{VectorDiffusionIntegrator, MassIntegrator}};
    use fem_assembly::physics::navier_stokes::assemble_divergence_matrix;
    use fem_linalg::CooMatrix;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, VectorH1Space, fe_space::FESpace};

    #[test]
    fn ns_transient_step_is_finite() {
        let mesh = Mesh::<2>::unit_square_tri(6);
        let vs = VectorH1Space::new(mesh.clone(), 2, 2);
        let ps = H1Space::new(mesh, 1);
        let nu = 0.01;
        let inv_dt = 100.0;
        let diff = VectorDiffusionIntegrator { kappa: nu };
        let a_diff = Assembler::assemble_bilinear(&vs, &[&diff], 5);

        // Build scalar mass, then extend to vector
        let scalar_mesh = Mesh::<2>::unit_square_tri(6);
        let scalar_space = fem_space::H1Space::new(scalar_mesh, 2);
        let sm = Assembler::assemble_bilinear(&scalar_space, &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }], 5);
        let n_s = vs.n_scalar_dofs();
        let mut coo = CooMatrix::new(vs.n_dofs(), vs.n_dofs());
        for r in 0..n_s { for k in sm.row_ptr[r]..sm.row_ptr[r+1] {
            let c = sm.col_idx[k] as usize; let v = inv_dt * sm.values[k];
            coo.add(r, c, v); coo.add(r + n_s, c + n_s, v);
        }}
        for r in 0..vs.n_dofs() { for k in a_diff.row_ptr[r]..a_diff.row_ptr[r+1] {
            coo.add(r, a_diff.col_idx[k] as usize, a_diff.values[k]);
        }}
        let a = coo.into_csr();
        let b = assemble_divergence_matrix(&vs, &ps, 5);
        let flat = fem_solver::BlockSystem { a, bt: b.transpose(), b, c: None }.to_flat_csr();
        let mut x = vec![0.0_f64; vs.n_dofs() + ps.n_dofs()];
        let r = fem_solver::solve_gmres(&flat, &vec![0.0_f64; vs.n_dofs() + ps.n_dofs()], &mut x, 50, &Default::default());
        assert!(r.is_ok());
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
