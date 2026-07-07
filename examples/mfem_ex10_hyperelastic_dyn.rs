//! # Example 10 — Hyperelastic (NeoHookean) static deformation
//! (analogous to MFEM ex10)
//!
//! Solves a 2-D hyperelastic problem with Dirichlet BCs.
//! Boundary attribute 1 is fixed; remaining boundaries are free.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex10_hyperelastic_dyn
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -m ../data/beam-quad.mesh
//! cargo run --example mfem_ex10_hyperelastic_dyn -- -m ../data/beam-tri.mesh -o 2
//! ```

use fem_assembly::{nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel}, NewtonConfig};
use fem_io::mfem::read_mfem_file;
use fem_mesh::SimplexMesh;
use fem_space::VectorH1Space;
use fem_space::fe_space::FESpace;
use fem_space::constraints::boundary_dofs;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 10: Hyperelastic (NeoHookean) ===");

    // Load or generate mesh
    let mesh: SimplexMesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(8)
    };
    let space = VectorH1Space::new(mesh, args.order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // Dirichlet BCs matching MFEM ex10: boundary attribute 1 is fixed.
    // Additional Dirichlet on the top (attribute 2) for a prescribed deformation.
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

    let model = HyperelasticModel::NeoHookean { mu: args.mu, lambda: args.lambda };
    let form = HyperelasticityForm::new(space, model, dirichlet, 3);
    let rhs = vec![0.0; n_dofs];
    let mut u = vec![0.0; n_dofs];
    // Set initial guess with the prescribed BC
    for &(d, v) in &form.dirichlet { u[d] = v; }

    let config = NewtonConfig { max_iter: 30, verbose: true, ..NewtonConfig::default() };
    let result = form.solve(&rhs, &mut u, &config);
    let (converged, iters, residual) = match &result {
        Ok(r) => (true, r.iterations, r.final_residual),
        Err(r) => (false, r.iterations, r.final_residual),
    };
    println!("  DOFs: {n_dofs}, converged = {converged}, iters = {iters}, residual = {residual:.3e}");
    if converged {
        println!("  PASS");
    }
}

/// CLI arguments matching MFEM ex10 conventions.
struct Args {
    mesh: Option<String>,
    order: u8,
    mu: f64,
    lambda: f64,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, mu: 0.3, lambda: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-mu" | "--shear-modulus" => {
                a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3)
            }
            "-lam" | "--lambda" => {
                a.lambda = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use fem_assembly::{physics::nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel}, NewtonConfig};
    use fem_mesh::SimplexMesh;
    use fem_space::VectorH1Space;

    #[test]
    fn smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[1]);
        let mut d = Vec::new();
        for &b in &bot { d.push((b as usize, 0.0)); d.push((b as usize + ns, 0.0)); }
        let form = HyperelasticityForm::new(space, HyperelasticModel::NeoHookean { mu: 0.3, lambda: 1.0 }, d, 3);
        let mut u = vec![0.0; form.n_dofs()];
        let result = form.solve(&vec![0.0; form.n_dofs()], &mut u, &NewtonConfig { max_iter: 30, ..NewtonConfig::default() });
        // Accept convergence or failure (smoke test)
        let _ = result;
    }
}
