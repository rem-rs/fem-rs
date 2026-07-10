//! Example 19 鈥?Incompressible neo-Hookean hyperelastic
//! (analogous to MFEM ex19)
//!
//! Solves for the equilibrium of a nearly-incompressible neo-Hookean
//! material under prescribed Dirichlet BCs.  Zero RHS: Newton's method
//! solves the equilibrium equation F(u) = 0.
//!
//! BCs (matching MFEM ex19):
//!   Boundary attribute 1: u = 0 (fixed)
//!   Boundary attribute 2: u_x = 0, u_y = 0.25路x (prescribed shear)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_dyn_incomp
//!   cargo run --example mfem_ex19_hyperelastic_dyn_incomp -- -m ../data/beam-tet.mesh -o 2
//!   cargo run --example mfem_ex19_hyperelastic_dyn_incomp -- -r 1 -mu 1.0

use fem_assembly::{
    nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel},
    NewtonConfig,
    LinearSolver,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_space::VectorH1Space;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 19: neo-Hookean (nearly-incompressible) ===");
    if !args.mesh.is_empty() {
        println!("  Mesh file: {}", args.mesh);
    }
    println!("  Order: {}, refine: {}, mu: {}", args.order, args.refine, args.mu);

    // 1. Load or generate mesh
    let base_mesh: Mesh<2> = if !args.mesh.is_empty() {
        let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(8)
    };
    let mesh = if args.refine > 0 {
        let mut m = base_mesh;
        for _ in 0..args.refine {
            m = refine_uniform(&m);
        }
        m
    } else {
        base_mesh
    };

    // 2. Finite element space 鈥?VectorH鹿 for displacement (P2 by default)
    let space = VectorH1Space::new(mesh, args.order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    // 3. Dirichlet BCs matching MFEM ex19:
    //    Boundary attribute 1 鈫?fixed (u = 0)
    //    Boundary attribute 2 鈫?u_y = 0.25路x, u_x = 0
    let dm = space.scalar_dof_manager();
    let attr1 = boundary_dofs(space.mesh(), dm, &[1]);
    let attr2 = boundary_dofs(space.mesh(), dm, &[2]);
    let mut dirichlet: Vec<(usize, f64)> = Vec::new();
    for &d in &attr1 {
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.0));
    }
    for &d in &attr2 {
        let x = dm.dof_coord(d)[0]; // x-coordinate of this DOF
        dirichlet.push((d as usize, 0.0));
        dirichlet.push((d as usize + n_scalar, 0.25 * x));
    }

    // 4. Neo-Hookean material (near-incompressible via large 位)
    let mu = args.mu;
    let lambda = 1.0e3_f64; // bulk penalty for near-incompressibility
    let model = HyperelasticModel::NeoHookean { mu, lambda };
    let form = HyperelasticityForm::new(space, model, dirichlet, 3);

    // 5. Zero RHS 鈥?Newton's method for equilibrium F(u) = 0
    let rhs = vec![0.0; n_dofs];
    let mut u = vec![0.0; n_dofs];
    // Initial guess: prescribed Dirichlet values
    for &(d, v) in &form.dirichlet {
        u[d] = v;
    }

    // 6. Newton solve
    let config = NewtonConfig {
        max_iter: 200,
        verbose: true,
        linear_solver: LinearSolver::SparseLu,
        ..NewtonConfig::default()
    };
    let result = form.solve(&rhs, &mut u, &config);
    let (converged, iters, residual) = match &result {
        Ok(r) => (true, r.iterations, r.final_residual),
        Err(r) => (false, r.iterations, r.final_residual),
    };
    println!("=== ex19: neo-Hookean (nearly-incompressible) ===");
    println!(
        "  DOFs: {n_dofs}, converged = {converged}, iters = {iters}, residual = {residual:.3e}"
    );
    if converged {
        println!("  PASS");
    }
}

/// CLI arguments matching MFEM ex19 conventions.
struct Args {
    mesh: String,
    order: u8,
    refine: usize,
    mu: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: String::new(),
        order: 2,
        refine: 0,
        mu: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_default(),
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
            }
            "-r" | "--refine" => {
                a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0)
            }
            "-mu" | "--shear-modulus" => {
                a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use fem_assembly::{
        nonlinear_hyperelasticity::{HyperelasticityForm, HyperelasticModel},
        NewtonConfig, NonlinearForm,
    LinearSolver,
    };
    use fem_mesh::Mesh;
    use fem_space::{VectorH1Space, FESpace};

    #[test]
    fn smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager();
        let bot = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[1]);
        let mut d = Vec::new();
        for &b in &bot {
            d.push((b as usize, 0.0));
            d.push((b as usize + ns, 0.0));
        }
        let form = HyperelasticityForm::new(
            space,
            HyperelasticModel::NeoHookean { mu: 1.0, lambda: 100.0 },
            d,
            3,
        );
        let mut u = vec![0.0; form.n_dofs()];
        let result = form.solve(
            &vec![0.0; form.n_dofs()],
            &mut u,
            &NewtonConfig { max_iter: 10, ..NewtonConfig::default() },
        );
        let _ = result;
    }
}




