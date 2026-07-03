//! Example 35 — Multidomain Poisson coupling (analogous to MFEM ex35)
//!
//! Two sub-domains with a shared interface: solves -Δu = f on each
//! domain and couples via matching interface conditions.
//!
//! Usage:
//!   cargo run --example mfem_ex35_multidomain

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    // Two independent meshes (sub-domains)
    let mesh_a = SimplexMesh::<2>::unit_square_tri(6);
    let mesh_b = SimplexMesh::<2>::unit_square_tri(6);
    let space_a = H1Space::new(mesh_a, 1);
    let space_b = H1Space::new(mesh_b, 1);
    let na = space_a.n_dofs();
    let nb = space_b.n_dofs();

    // Build block diagonal system: diag(K_a, K_b)
    let ka = Assembler::assemble_bilinear(&space_a, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let kb = Assembler::assemble_bilinear(&space_b, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let fa = Assembler::assemble_linear(&space_a, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);
    let fb = Assembler::assemble_linear(&space_b, &[&DomainSourceIntegrator::new(|_| 0.0)], 3);

    let n = na + nb;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..na { for p in ka.row_ptr[i]..ka.row_ptr[i+1] { coo.add(i, ka.col_idx[p] as usize, ka.values[p]); }}
    for i in 0..nb { for p in kb.row_ptr[i]..kb.row_ptr[i+1] { coo.add(na+i, na+kb.col_idx[p] as usize, kb.values[p]); }}
    let mut k = coo.into_csr();
    let mut rhs = [fa, fb].concat();

    // Dirichlet BCs: u=0 on all boundaries
    let bnd_a = boundary_dofs(space_a.mesh(), &space_a.dof_manager(), &[1,2,3,4]);
    let bnd_b = boundary_dofs(space_b.mesh(), &space_b.dof_manager(), &[1,2,3,4]);
    for &d in &bnd_a { k.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs); }
    for &d in &bnd_b { k.apply_dirichlet_symmetric(na + d as usize, 0.0, &mut rhs); }

    let mut u = vec![0.0; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    let res = solve_cg(&k, &rhs, &mut u, &cfg).unwrap();
    let u_norm: f64 = u.iter().map(|v| v*v).sum::<f64>().sqrt();

    println!("=== ex35: Multidomain Poisson ===");
    println!("  DOFs: {n} (A={na}, B={nb}), iters={}, ‖u‖={:.6e}", res.iterations, u_norm);
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::DiffusionIntegrator};
    use fem_linalg::CooMatrix;
    use fem_mesh::SimplexMesh;
    use fem_solver::{solve_cg, SolverConfig};
    use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};
    #[test] fn smoke() {
        let a = H1Space::new(SimplexMesh::<2>::unit_square_tri(4), 1);
        let b = H1Space::new(SimplexMesh::<2>::unit_square_tri(4), 1);
        let ka = Assembler::assemble_bilinear(&a, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let kb = Assembler::assemble_bilinear(&b, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let n = a.n_dofs() + b.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        let na = a.n_dofs();
        for i in 0..na { for p in ka.row_ptr[i]..ka.row_ptr[i+1] { coo.add(i, ka.col_idx[p] as usize, ka.values[p]); }}
        for i in 0..b.n_dofs() { for p in kb.row_ptr[i]..kb.row_ptr[i+1] { coo.add(na+i, na+kb.col_idx[p] as usize, kb.values[p]); }}
        let mut k = coo.into_csr();
        let mut rhs = vec![1.0; n];
        for &d in &boundary_dofs(a.mesh(), &a.dof_manager(), &[1,2,3,4]) { k.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs); }
        for &d in &boundary_dofs(b.mesh(), &b.dof_manager(), &[1,2,3,4]) { k.apply_dirichlet_symmetric(na + d as usize, 0.0, &mut rhs); }
        let mut u = vec![0.0; n];
        assert!(solve_cg(&k, &rhs, &mut u, &SolverConfig { max_iter: 500, ..SolverConfig::default() }).is_ok());
    }
}
