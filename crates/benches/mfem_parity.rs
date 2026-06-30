//! MFEM parity baseline — Poisson/Elasticity/Maxwell/Stokes performance.
//!
//! Run: cargo bench -p fem-benches --bench mfem_parity
//! Quick: FEM_BENCH_QUICK=1 cargo bench -p fem-benches --bench mfem_parity

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::{Duration, Instant};

use fem_mesh::SimplexMesh;
use fem_space::{H1Space, FESpace};
use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_solver::{SolverConfig, solve_pcg_jacobi};

fn quick_mode() -> bool {
    std::env::var("FEM_BENCH_QUICK").map(|v| v != "0").unwrap_or(false)
}

fn config() -> Criterion {
    if quick_mode() {
        Criterion::default().sample_size(5).warm_up_time(Duration::from_millis(100)).measurement_time(Duration::from_millis(500))
    } else {
        Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(3))
    }
}

fn bench_poisson(c: &mut Criterion) {
    for &n in &[4, 8, 16, 24] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let dofs = space.n_dofs();
        if dofs > 2_000_000 && quick_mode() { break; }

        c.bench_function(&format!("Poisson2D_P1_{dofs}dof"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let mesh = SimplexMesh::<2>::unit_square_tri(n);
                    let space = H1Space::new(mesh, 1);
                    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
                    let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 1.0)], 2);
                    let _ta = t0.elapsed();
                    let cfg = SolverConfig { rtol: 1e-6, max_iter: 10000, ..SolverConfig::default() };
                    let mut u = vec![0.0; space.n_dofs()];
                    let _ = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg);
                    total += t0.elapsed();
                }
                total
            })
        });
    }
}

criterion_group! {
    name = mfem_parity;
    config = config();
    targets = bench_poisson
}

criterion_main!(mfem_parity);
