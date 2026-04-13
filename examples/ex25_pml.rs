//! ex25_pml - baseline PML-like complex Helmholtz demo.
//!
//! Uses a spatially varying damping coefficient in boundary layers to mimic
//! absorbing behavior.

use fem_assembly::{
    ComplexAssembler, ComplexGridFunction,
    coefficient::PmlCoeff,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

fn main() {
    let args = parse_args();

    println!("=== ex25_pml: baseline complex PML-like damping ===");
    println!(
        "  n={}, omega={}, pml_thickness={}, sigma_max={}, wx={}, wy={}",
        args.n, args.omega, args.thickness, args.sigma_max, args.wx, args.wy
    );

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let n = space.n_dofs();

    let pml_sigma = PmlCoeff::new(
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        args.thickness,
        args.sigma_max,
    )
    .with_axis_weights(vec![args.wx, args.wy]);

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&MassIntegrator { rho: 1.0 }],
        &[&MassIntegrator { rho: pml_sigma }],
        args.omega,
        3,
    );

    let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);

    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let left: Vec<usize> = boundary_dofs(mesh_ref, dm, &[4]).into_iter().map(|d| d as usize).collect();
    let other: Vec<usize> = boundary_dofs(mesh_ref, dm, &[1, 2, 3]).into_iter().map(|d| d as usize).collect();

    sys.apply_dirichlet(&other, &vec![0.0; other.len()], &vec![0.0; other.len()], &mut rhs);
    sys.apply_dirichlet(&left, &vec![1.0; left.len()], &vec![0.0; left.len()], &mut rhs);

    let a = sys.to_flat_csr();
    let mut x = vec![0.0; 2 * n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 3000, verbose: false, ..Default::default() };
    let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES failed");

    let gf = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let amax = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let amin = amp.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  dofs={}, GMRES iters={}, res={:.3e}", n, res.iterations, res.final_residual);
    println!("  |u| range: [{:.4}, {:.4}]", amin, amax);
    assert!(res.converged, "PML baseline solve did not converge");
    println!("  PASS");
}

struct Args {
    n: usize,
    omega: f64,
    thickness: f64,
    sigma_max: f64,
    wx: f64,
    wy: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 12, omega: 2.0, thickness: 0.2, sigma_max: 1.0, wx: 1.0, wy: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("12".into()).parse().unwrap_or(12),
            "--omega" => a.omega = it.next().unwrap_or("2.0".into()).parse().unwrap_or(2.0),
            "--pml-thickness" => a.thickness = it.next().unwrap_or("0.2".into()).parse().unwrap_or(0.2),
            "--sigma-max" => a.sigma_max = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--wx" => a.wx = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            "--wy" => a.wy = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0),
            _ => {}
        }
    }
    a
}
