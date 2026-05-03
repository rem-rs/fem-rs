use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_amg::{AmgConfig, AmgSolver};
use fem_solver::SolverConfig;

fn bench_amg_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_setup");

    for n in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("hierarchy", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(mesh, 1u8);

            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let csr = Assembler::assemble_bilinear(&space, &[&diffusion], 2);

            let config = AmgConfig::default();

            b.iter(|| {
                let solver = AmgSolver::setup(&csr, config.clone());
                black_box(solver);
            });
        });
    }

    group.finish();
}

fn bench_amg_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_solve");

    for n in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("vcycle", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(mesh, 1u8);

            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let csr = Assembler::assemble_bilinear(&space, &[&diffusion], 2);

            let config = AmgConfig::default();
            let solver = AmgSolver::setup(&csr, config);
            
            let n_dofs = space.n_dofs();
            let rhs = vec![1.0_f64; n_dofs];
            let mut x = vec![0.0_f64; n_dofs];
            let cfg = SolverConfig::default();

            b.iter(|| {
                let mut x_iter = x.clone();
                let _ = solver.solve(&csr, &rhs, &mut x_iter, &cfg);
                black_box(&x_iter);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_amg_setup, bench_amg_solve);
criterion_main!(benches);
