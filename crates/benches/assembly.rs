use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList, TangentialMassIntegrator, VectorBoundaryAssembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, HCurlSpace, L2Space};

fn bench_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    // n=8..32: serial range; n=64..128: parallel benefit visible (~8k-32k elements)
    for n in [8, 16, 32, 64, 128].iter() {
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

    let mut vb_group = c.benchmark_group("assembly_hcurl_boundary");
    for n in [16, 32, 64].iter() {
        vb_group.bench_with_input(BenchmarkId::new("tangential_mass_nd1", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let space = HCurlSpace::new(mesh, 1);
            let integ = TangentialMassIntegrator { gamma: 1.0 };
            b.iter(|| {
                let mat =
                    VectorBoundaryAssembler::assemble_boundary_bilinear(&space, &[&integ], &[1, 2, 3, 4], 4);
                black_box(mat);
            });
        });
    }
    vb_group.finish();

    let mut dg_group = c.benchmark_group("assembly_dg_faces");
    for n in [16, 32, 48].iter() {
        dg_group.bench_with_input(BenchmarkId::new("sip_l2_p1", n), n, |b, n| {
            let mesh = SimplexMesh::<2>::unit_square_tri(*n);
            let ifl = InteriorFaceList::build(&mesh);
            let space = L2Space::new(mesh, 1);
            b.iter(|| {
                let mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, 20.0, 3);
                black_box(mat);
            });
        });
    }
    dg_group.finish();
}

criterion_group!(benches, bench_assembly);
criterion_main!(benches);
