//! Active-set solver for 3D frictionless contact (Poisson).
//!
//! Usage:
//!   cargo run --example contact_active_set_3d

use std::time::Instant;
use fem_assembly::Assembler;
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::SimplexMesh;
use fem_solver::active_set::solve_active_set_contact;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

fn main() {
    println!("=== Active-set 3D contact (Poisson) ===");
    let t0 = Instant::now();
    let mesh = SimplexMesh::<3>::unit_cube_tet(2);
    let n = mesh.n_nodes() as usize;
    let space = H1Space::new(mesh, 1);
    let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let rhs = vec![1.0; space.n_dofs()];
    let u = solve_active_set_contact(&k, &rhs, space.mesh(), &[1], &|_| 0.0, 50);
    let u_norm: f64 = u.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  DOFs: {}, ‖u‖ = {u_norm:.6e}", space.n_dofs());
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::Assembler;
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_mesh::SimplexMesh;
    use fem_solver::active_set::solve_active_set_contact;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;
    #[test] fn smoke() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let space = H1Space::new(mesh, 1);
        let k = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
        let u = solve_active_set_contact(&k, &vec![1.0; space.n_dofs()], space.mesh(), &[1], &|_| 0.0, 30);
        assert!(u.iter().all(|v| v.is_finite()));
    }
}
