//! FEM solver benchmarks.
//!
//! Run with: `cargo test --release -p fem-solver -- bench_ --nocapture`
//!
//! Each benchmark constructs a matrix at the specified size, solves, and
//! reports wall-clock time.  Results are printed in CSV-like format for
//! easy collection.

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator, ConvectionIntegrator, DomainSourceIntegrator},
    coefficient::ConstantVectorCoeff,
};
use fem_mesh::SimplexMesh;
use fem_solver::*;
use fem_space::{H1Space, fe_space::FESpace, constraints::{apply_dirichlet, boundary_dofs}};

/// Solve Poisson with CG + Jacobi and report time.
fn bench_poisson_cg_jacobi(n: usize, label: &str) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let f = |x: &[f64]| 2.0 * std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin();
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], 3);
    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1,2,3,4]);
    let vals = vec![0.0; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..SolverConfig::default() };
    let start = Instant::now();
    let res = solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg).unwrap();
    let elapsed = start.elapsed();
    println!("BENCH,{label},{},{},{:.3},{},{}", space.n_dofs(), mat.nnz(), elapsed.as_secs_f64(), res.iterations, res.final_residual);
}

/// Solve Poisson with CG + ILU0.
fn bench_poisson_cg_ilu0(n: usize, label: &str) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let f = |x: &[f64]| 2.0 * std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin();
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], 3);
    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1,2,3,4]);
    let vals = vec![0.0; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..SolverConfig::default() };
    let start = Instant::now();
    let res = solve_pcg_ilu0(&mat, &rhs, &mut x, &cfg).unwrap();
    let elapsed = start.elapsed();
    println!("BENCH,{label},{},{},{:.3},{},{}", space.n_dofs(), mat.nnz(), elapsed.as_secs_f64(), res.iterations, res.final_residual);
}

/// Solve mass matrix with CG (near-optimal scaling).
fn bench_mass_cg(n: usize, label: &str) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
    let rhs = vec![1.0; space.n_dofs()];
    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-8, max_iter: 10000, verbose: false, ..SolverConfig::default() };
    let start = Instant::now();
    let res = solve_cg(&mat, &rhs, &mut x, &cfg).unwrap();
    let elapsed = start.elapsed();
    println!("BENCH,{label},{},{},{:.3},{},{}", space.n_dofs(), mat.nnz(), elapsed.as_secs_f64(), res.iterations, res.final_residual);
}

/// Solve non-symmetric convection-diffusion with GMRES + ILUT.
fn bench_convdiff_gmres_ilut(n: usize, label: &str) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1);
    let diff = DiffusionIntegrator { kappa: 0.01 };
    let conv = ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0]) };
    let mat = Assembler::assemble_bilinear(&space, &[&diff, &conv], 3);
    let f = |_: &[f64]| 1.0;
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], 3);

    let mut mat = mat;
    let bnd = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4]);
    let vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    let mut x = vec![0.0; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-6, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    let start = Instant::now();
    let res = solve_gmres_ilut(&mat, &rhs, &mut x, 30, 1e-4, 2, &cfg).unwrap();
    let elapsed = start.elapsed();
    println!("BENCH,{label},{},{},{:.3},{},{}", space.n_dofs(), mat.nnz(), elapsed.as_secs_f64(), res.iterations, res.final_residual);
}

macro_rules! bench_series {
    ($name:ident, $fn:ident, $label:expr, $sizes:expr) => {
        #[test]
        fn $name() {
            println!("BENCH,label,n_dofs,nnz,time_s,iters,residual");
            for &n in $sizes { $fn(n, $label); }
        }
    };
}

bench_series!(bench_cg_jacobi_8,     bench_poisson_cg_jacobi,  "cg_jacobi_8",    &[8, 16, 32, 64]);
bench_series!(bench_cg_jacobi_128,  bench_poisson_cg_jacobi,  "cg_jacobi_128",  &[128]);
bench_series!(bench_cg_ilu0_8,      bench_poisson_cg_ilu0,    "cg_ilu0_8",      &[8, 16, 32, 64]);
bench_series!(bench_mass_cg_8,      bench_mass_cg,            "mass_cg_8",      &[8, 16, 32, 64]);
bench_series!(bench_gmres_ilut_8,   bench_convdiff_gmres_ilut,"gmres_ilut_8",   &[8, 16, 32]);
