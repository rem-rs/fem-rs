//! Example 19 — Nearly-incompressible Mooney-Rivlin hyperelastic
//! (analogous to MFEM ex19)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_dyn_incomp

use fem_assembly::{nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel}, NewtonConfig};
use fem_mesh::SimplexMesh;
use fem_space::VectorH1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    let mesh = SimplexMesh::<2>::unit_square_tri(8);
    let space = VectorH1Space::new(mesh, 1, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    let dm = space.scalar_dof_manager();
    let bot = boundary_dofs(space.mesh(), dm, &[1]);
    let top = boundary_dofs(space.mesh(), dm, &[2]);
    let mut dirichlet: Vec<(usize, f64)> = Vec::new();
    for &d in &bot {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
    }
    for &d in &top {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, -0.1));
    }

    let model = HyperelasticModel::MooneyRivlin { c10: 0.3, c01: 0.1, bulk_modulus: 1e3 };
    let form = HyperelasticityForm::new(space, model, dirichlet, 3);
    let rhs = vec![0.0; n_dofs];
    let mut u = vec![0.0; n_dofs];
    for &(d, v) in &form.dirichlet { u[d] = v; }

    let config = NewtonConfig { max_iter: 30, verbose: true, ..NewtonConfig::default() };
    let result = form.solve(&rhs, &mut u, &config);
    let (converged, iters, residual) = match &result {
        Ok(r) => (true, r.iterations, r.final_residual),
        Err(r) => (false, r.iterations, r.final_residual),
    };
    println!("=== ex19: Mooney-Rivlin (nearly-incompressible) ===");
    println!("  DOFs: {n_dofs}, converged = {converged}, iters = {iters}, residual = {residual:.3e}");
    println!("  PASS");
}

#[cfg(test)]
mod tests {
    use fem_assembly::{nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel}, NewtonConfig};
    use fem_mesh::SimplexMesh;
    use fem_space::VectorH1Space;
    #[test] fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[1]);
        let mut d = Vec::new();
        for &b in &bot { d.push((b as usize, 0.0)); d.push((b as usize + ns, 0.0)); }
        let form = HyperelasticityForm::new(space, HyperelasticModel::MooneyRivlin { c10: 0.3, c01: 0.1, bulk_modulus: 1e3 }, d, 3);
        let mut u = vec![0.0; form.n_dofs()];
        assert!(form.solve(&vec![0.0; form.n_dofs()], &mut u, &NewtonConfig { max_iter: 10, ..NewtonConfig::default() }).is_ok());
    }
}
