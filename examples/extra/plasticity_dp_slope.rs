//! Drucker-Prager plasticity: 2D plane-strain slope stability.
//!
//! Drucker-Prager is a pressure-sensitive yield model suitable for
//! geomaterials (soil, rock).  This example sets up the form, prescribes
//! a displacement field, and computes the residual.
//!
//! Usage:
//!   cargo run --example plasticity_dp_slope

use std::time::Instant;

use fem_assembly::physics::nonlinear::NonlinearForm;
use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
use fem_assembly::Assembler;
use fem_assembly::standard::DomainSourceIntegrator;
use fem_mesh::Mesh;
use fem_space::vector_h1::VectorH1Space;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    println!("=== Drucker-Prager plasticity: slope ===");
    let t0 = Instant::now();

    let mesh = Mesh::<2>::unit_square_tri(4);
    let order: u8 = 1;
    let quad_order: u8 = 2;
    let space = VectorH1Space::new(mesh, order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // Drucker-Prager: E, nu, cohesion, friction_angle_deg, hardening
    let cfg = PlasticConfig::drucker_prager(200_000.0, 0.3, 50.0, 30.0, 1000.0);
    println!("  Mohr-Coulomb approx: c={:.1}, φ={:.0}°, H={}",
             cfg.yield_stress, cfg.friction_angle.to_degrees(), cfg.hardening_modulus);

    let dm = space.scalar_dof_manager();
    let bot: Vec<u32> = boundary_dofs(space.mesh(), dm, &[1]);
    let mut dirichlet = Vec::new();
    for &d in &bot {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
    }

    let rhs = vec![0.0; n_dofs];
    let form = J2PlasticityForm::new(space, cfg, dirichlet, quad_order);
    let mut u = vec![0.0; n_dofs];
    for ni in 0..n_scalar {
        let y = (ni as f64) / (n_scalar as f64 - 1.0).max(1.0);
        u[ni + n_scalar] = -0.001 * y;
    }

    let mut r = vec![0.0; n_dofs];
    form.residual(&u, &rhs, &mut r);
    let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  ‖residual‖ = {rn:.6e}");
    println!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use fem_assembly::physics::nonlinear::NonlinearForm;
    use fem_assembly::plasticity::{J2PlasticityForm, PlasticConfig};
    use fem_mesh::Mesh;
    use fem_space::vector_h1::VectorH1Space;
    use fem_space::constraints::boundary_dofs;
    #[test] fn smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot: Vec<u32> = boundary_dofs(space.mesh(), dm, &[1]);
        let mut dirichlet = Vec::new();
        for &d in &bot { dirichlet.push((d as usize, 0.0)); dirichlet.push((d as usize + ns, 0.0)); }
        let cfg = PlasticConfig::drucker_prager(200_000.0, 0.3, 50.0, 30.0, 1000.0);
        let form = J2PlasticityForm::new(space, cfg, dirichlet, 2);
        let u = vec![0.0; form.n_dofs()];
        let mut r = vec![0.0; form.n_dofs()];
        form.residual(&u, &[0.0; 0], &mut r);
        assert!(r.iter().all(|v| v.is_finite()));
    }
}
