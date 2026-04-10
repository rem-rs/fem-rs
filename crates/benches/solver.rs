use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_element::quadrature::gauss_triangle;
use fem_space::{H1Space, FESpace};
use fem_assembly::{Assembler, DiffusionIntegrator};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_linalg::Vector;

fn bench_pcg(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg");

    for n in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("jacobi", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
            let qr = gauss_triangle(2);

            let mut assembler = Assembler::new(&space);
            assembler.add_domain(DiffusionIntegrator::new(1.0));
            let (mat, _rhs) = assembler.assemble_bilinear(&qr);
            let mat = mat.into_csr();

            let rhs = Vector::from_vec(vec![1.0; space.n_dofs()]);
            let config = SolverConfig::default()
                .with_tol(1e-10)
                .with_max_iter(10000);

            b.iter(|| {
                let x0 = Vector::zeros(space.n_dofs());
                let result = solve_pcg_jacobi(&mat, &rhs, x0, config);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pcg);
criterion_main!(benches);
