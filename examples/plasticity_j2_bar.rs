//! J2 small-strain plasticity form setup and residual computation.
//!
//! Demonstrates creating a J2 plasticity form with Dirichlet BCs,
//! prescribing a displacement field, and computing the residual.
//!
//! Usage:
//!   cargo run --example plasticity_j2_bar

use std::time::Instant;

use fem_assembly::nonlinear::NonlinearForm;
use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
use fem_assembly::Assembler;
use fem_assembly::standard::DomainSourceIntegrator;
use fem_mesh::SimplexMesh;
use fem_space::vector_h1::VectorH1Space;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    println!("=== J2 plasticity: 2D bar (plane strain) ===");
    let t0 = Instant::now();

    let mesh = SimplexMesh::<2>::unit_square_tri(4);
    let order: u8 = 1;
    let quad_order: u8 = 2;
    let space = VectorH1Space::new(mesh, order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();
    println!("  DOFs: {n_dofs} (scalar {n_scalar})");

    let cfg = PlasticConfig::j2(200_000.0, 0.3, 250.0, 2000.0);
    println!("  E={}, nu={}, sigma_y={}, H={}",
             cfg.E, cfg.nu, cfg.yield_stress, cfg.hardening_modulus);

    // Dirichlet BC: bottom fixed
    let dm = space.scalar_dof_manager();
    let bot_dofs: Vec<u32> = boundary_dofs(space.mesh(), dm, &[1]);
    let mut dirichlet = Vec::new();
    for &d in &bot_dofs {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
    }

    // Body force (small, elastic regime)
    let scalar_space = H1Space::new(space.mesh().clone(), order);
    let fy = Assembler::assemble_linear(
        &scalar_space, &[&DomainSourceIntegrator::new(|_: &[f64]| -10.0)], quad_order);
    let mut rhs = vec![0.0; n_dofs];
    for i in 0..n_scalar { rhs[i + n_scalar] = fy[i]; }

    // Create form and prescribe a small displacement
    let form = J2PlasticityForm::new(space, cfg, dirichlet, quad_order);
    let mut u = vec![0.0; n_dofs];
    for ni in 0..n_scalar {
        let y = (ni as f64) / (n_scalar as f64 - 1.0).max(1.0);
        u[ni + n_scalar] = -0.001 * y; // linear compression
    }

    // Compute residual
    let mut r = vec![0.0; n_dofs];
    form.residual(&u, &rhs, &mut r);
    let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  ‖residual‖ = {:.6e}", r_norm);
    assert!(r_norm.is_finite(), "residual must be finite");
    println!("  Done.");
    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::nonlinear::NonlinearForm;
    use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
    use fem_mesh::SimplexMesh;
    use fem_space::vector_h1::VectorH1Space;
    use fem_space::constraints::boundary_dofs;

    #[test]
    fn residual_is_finite() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot: Vec<u32> = boundary_dofs(space.mesh(), dm, &[1]);
        let mut dirichlet = Vec::new();
        for &d in &bot {
            dirichlet.push((d as usize, 0.0));
            dirichlet.push((d as usize + ns, 0.0));
        }
        let cfg = PlasticConfig::j2(200_000.0, 0.3, 250.0, 2000.0);
        let form = J2PlasticityForm::new(space, cfg, dirichlet, 2);
        let rhs = vec![0.0; n];

        let mut u = vec![0.0; n];
        for ni in 0..ns {
            let y = (ni as f64) / (ns as f64 - 1.0).max(1.0);
            u[ni + ns] = -0.001 * y;
        }

        let mut r = vec![0.0; n];
        form.residual(&u, &rhs, &mut r);
        assert!(r.iter().all(|v| v.is_finite()), "residual must be finite");
    }
}
