use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_element::quadrature::gauss_triangle;
use fem_space::H1Space;
use fem_assembly::{Assembler, DiffusionIntegrator, DomainSourceIntegrator};

fn bench_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");

    for n in [8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::new("poisson_p1", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
            let qr = gauss_triangle(2);

            b.iter(|| {
                let mut assembler = Assembler::new(&space);
                assembler.add_domain(DiffusionIntegrator::new(1.0));
                assembler.add_domain_load(DomainSourceIntegrator::new(|_| 1.0));
                let (mat, rhs) = assembler.assemble_bilinear(&qr);
                black_box((mat, rhs));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_assembly);
criterion_main!(benches);
