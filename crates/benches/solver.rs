use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_solver::{solve_pcg_jacobi, SolverConfig};

fn bench_pcg(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");

    for n in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("jacobi", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(mesh, 1u8);

            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);

            let n_dofs = space.n_dofs();
            let rhs = vec![1.0_f64; n_dofs];
            let cfg = SolverConfig::default();

            b.iter(|| {
                let mut x = vec![0.0_f64; n_dofs];
                let result = solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pcg);
criterion_main!(benches);

