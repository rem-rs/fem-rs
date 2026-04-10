use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fem_mesh::SimplexMesh;
use fem_element::quadrature::gauss_triangle;
use fem_space::{H1Space, FESpace};
use fem_assembly::{Assembler, DiffusionIntegrator};
use fem_amg::{AmgParams, CycleType, SmootherType, AmgSolver};
use fem_linalg::Vector;

fn bench_amg_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_setup");

    for n in [16, 32, 64].iter() {
        group.bench_with_input(BenchmarkId::new("hierarchy", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
            let qr = gauss_triangle(2);

            let mut assembler = Assembler::new(&space);
            assembler.add_domain(DiffusionIntegrator::new(1.0));
            let (mat, _) = assembler.assemble_bilinear(&qr);
            let csr = mat.into_csr();

            let params = AmgParams::default()
                .with_max_levels(10)
                .with_smoother(SmootherType::Jacobi(1.0));

            b.iter(|| {
                let solver = AmgSolver::new(params.clone());
                let hierarchy = solver.setup(&csr);
                black_box(hierarchy);
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
            let space = H1Space::new(&mesh, fem_element::lagrange::TriP1::new());
            let qr = gauss_triangle(2);

            let mut assembler = Assembler::new(&space);
            assembler.add_domain(DiffusionIntegrator::new(1.0));
            let (mat, _) = assembler.assemble_bilinear(&qr);
            let csr = mat.into_csr();

            let params = AmgParams::default()
                .with_max_levels(10)
                .with_cycle_type(CycleType::V)
                .with_smoother(SmootherType::Jacobi(1.0));

            let solver = AmgSolver::new(params);
            let hierarchy = solver.setup(&csr);
            let rhs = Vector::from_vec(vec![1.0; space.n_dofs()]);
            let x0 = Vector::zeros(space.n_dofs());

            b.iter(|| {
                let result = solver.solve(&hierarchy, &rhs, x0.clone());
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_amg_setup, bench_amg_solve);
criterion_main!(benches);
