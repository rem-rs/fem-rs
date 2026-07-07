use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList, TangentialMassIntegrator, VectorBoundaryAssembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_mesh::Mesh;
use fem_space::{H1Space, HCurlSpace, L2Space, fe_space::FESpace};

fn bench_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    for n in [8, 16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("poisson_p1", n), n, |b, n| {
            let mesh = Mesh::<2>::unit_square_tri(*n);
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
            let mesh = Mesh::<2>::unit_square_tri(*n);
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
            let mesh = Mesh::<2>::unit_square_tri(*n);
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

#[cfg(feature = "gpu")]
fn bench_assembly_gpu(c: &mut Criterion) {
    use fem_assembly::standard::DiffusionIntegrator;
    use fem_assembly::Assembler;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, fe_space::FESpace};

    // GPU assembly needs a real GPU — skip if GpuContext fails
    let Ok(gpu) = fem_linalg_gpu::GpuContext::new_sync() else { return; };

    let mut group = c.benchmark_group("assembly_gpu");
    for n in [32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("poisson_p1_gpu", n), n, |b, n| {
            let mesh = Mesh::<2>::unit_square_tri(*n);
            let space = H1Space::new(mesh, 1);
            let diffusion = DiffusionIntegrator { kappa: 1.0 };

            b.iter(|| {
                let mat = Assembler::assemble_bilinear_gpu(&space, &[&diffusion])
                    .expect("GPU assembly failed");
                black_box(mat);
            });
        });
    }
    group.finish();

    // Compare with CPU for largest size
    let mut cmp = c.benchmark_group("assembly_gpu_vs_cpu");
    for n in [64, 128].iter() {
        let mesh = Mesh::<2>::unit_square_tri(*n);
        let space = H1Space::new(mesh, 1);
        let diffusion = DiffusionIntegrator { kappa: 1.0 };

        cmp.bench_with_input(BenchmarkId::new("cpu", n), n, |b, _| {
            b.iter(|| {
                let mat = Assembler::assemble_bilinear(&space, &[&diffusion], 2);
                black_box(mat);
            });
        });
        cmp.bench_with_input(BenchmarkId::new("gpu", n), n, |b, _| {
            b.iter(|| {
                let mat = Assembler::assemble_bilinear_gpu(&space, &[&diffusion])
                    .expect("GPU assembly failed");
                black_box(mat);
            });
        });
    }
    cmp.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(benches_gpu, bench_assembly_gpu);

criterion_group!(benches, bench_assembly);

#[cfg(feature = "gpu")]
criterion_main!(benches, benches_gpu);
#[cfg(not(feature = "gpu"))]
criterion_main!(benches);
