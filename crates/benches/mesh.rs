use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_mesh::amr::refine_uniform;

fn bench_mesh_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_generation");

    for n in [16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("unit_square_tri", n), n, |b, n| {
            b.iter(|| {
                let mesh = SimplexMesh::<2>::unit_square_tri(*n);
                black_box(mesh);
            });
        });
    }

    group.finish();
}

fn bench_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("refinement");

    for n in [8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::new("uniform_2d", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);

            b.iter(|| {
                let refined = refine_uniform(&mesh);
                black_box(refined);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mesh_generation, bench_refinement);
criterion_main!(benches);
