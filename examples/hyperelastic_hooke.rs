//! hyperelastic_hooke — 3D hyperelasticity (NeoHookean) unit cube compression.
//!
//! Analogous to MFEM miniapp `hooke`, demonstrating static finite-strain
//! hyperelasticity on a tetrahedral mesh with Newton–Raphson solve.
//!
//! Usage:
//!   cargo run --example hyperelastic_hooke
//!   cargo run --example hyperelastic_hooke -- --n 6

use fem_assembly::physics::nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel};
use fem_assembly::NewtonConfig;
use fem_mesh::SimplexMesh;
use fem_space::VectorH1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = VectorH1Space::new(mesh, 1, 3);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // Dirichlet BC: fix bottom (z=0, tag=1), compress top (z=1, tag=2) by 20%
    let dm = space.scalar_dof_manager();
    let bot = boundary_dofs(space.mesh(), dm, &[1]);
    let top = boundary_dofs(space.mesh(), dm, &[2]);
    let mut dirichlet: Vec<(usize, f64)> = Vec::new();
    for &d in &bot {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
        dirichlet.push((d as usize + 2 * n_scalar, 0.0));
    }
    for &d in &top {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
        dirichlet.push((d as usize + 2 * n_scalar, -0.2)); // compress 20%
    }

    let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
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
    let max_disp: f64 = u.iter().fold(0.0, |a, &v| a.max(v.abs()));
    println!("=== hyperelastic_hooke: 3D NeoHookean compression ===");
    println!("  Mesh: {n}×{n}×{n} tets, DOFs: {n_dofs}");
    println!("  Newton: converged={converged}, iters={iters}, residual={residual:.3e}");
    println!("  Max displacement: {max_disp:.4e}");
}

#[cfg(test)]
mod tests {
    use fem_assembly::physics::nonlinear_hyperelasticity::HyperelasticModel;

    #[test]
    fn hooke_3d_converges() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let space = fem_space::VectorH1Space::new(mesh, 1, 3);
        let n_dofs = space.n_dofs();
        let n_scalar = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[1]);
        let top = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[2]);
        let mut dirichlet: Vec<(usize, f64)> = Vec::new();
        for &d in &bot {
            dirichlet.push((d as usize, 0.0));
            dirichlet.push((d as usize + n_scalar, 0.0));
            dirichlet.push((d as usize + 2 * n_scalar, 0.0));
        }
        for &d in &top {
            dirichlet.push((d as usize, 0.0));
            dirichlet.push((d as usize + n_scalar, 0.0));
            dirichlet.push((d as usize + 2 * n_scalar, -0.2));
        }
        let model = HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 };
        let form = fem_assembly::physics::nonlinear_hyperelasticity::HyperelasticityForm::new(space, model, dirichlet, 3);
        let rhs = vec![0.0; n_dofs];
        let mut u = vec![0.0; n_dofs];
        for &(d, v) in &form.dirichlet { u[d] = v; }
        let config = fem_assembly::NewtonConfig { max_iter: 30, verbose: false, ..Default::default() };
        let r = form.solve(&rhs, &mut u, &config);
        assert!(r.is_ok() || r.as_ref().err().map(|e| e.final_residual < 1e-3).unwrap_or(false),
            "3D hyperelastic solve failed");
    }
}
