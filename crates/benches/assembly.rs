use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_mesh::SimplexMesh;
use fem_space::H1Space;

fn bench_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");

    for n in [8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::new("poisson_p1", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(mesh, 1);
            let diffusion = DiffusionIntegrator { kappa: 1.0 };
            let source = DomainSourceIntegrator::new(|_| 1.0);

            b.iter(|| {
                let mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);
                let rhs = Assembler::assemble_linear(&space, &[&source], 2);
                black_box((mat, rhs));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_assembly);
criterion_main!(benches);
