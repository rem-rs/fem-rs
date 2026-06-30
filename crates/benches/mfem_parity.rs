//! MFEM parity baseline — Poisson/Elasticity/Maxwell/Stokes performance.
//!
//! Run: cargo bench -p fem-benches --bench mfem_parity
//! Quick: FEM_BENCH_QUICK=1 cargo bench -p fem-benches --bench mfem_parity

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::{Duration, Instant};

use fem_mesh::SimplexMesh;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;
use fem_space::vector_h1::VectorH1Space;
use fem_space::hcurl::HCurlSpace;
use fem_assembly::{
    Assembler, VectorAssembler, MixedAssembler,
    standard::{
        DiffusionIntegrator, DomainSourceIntegrator,
        VectorDiffusionIntegrator, VectorH1MassIntegrator,
        CurlCurlIntegrator, VectorMassIntegrator,
    },
    mixed::DivIntegrator,
};
use fem_solver::{SolverConfig, solve_pcg_jacobi};

fn quick_mode() -> bool {
    std::env::var("FEM_BENCH_QUICK").map(|v| v != "0").unwrap_or(false)
}

fn config() -> Criterion {
    if quick_mode() {
        Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(100)).measurement_time(Duration::from_millis(500))
    } else {
        Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(3))
    }
}

// ─── Poisson 2D H¹ P1 ────────────────────────────────────────────────────

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

// ─── Elasticity 2D VectorH¹ P1 ────────────────────────────────────────────

fn bench_elasticity(c: &mut Criterion) {
    for &n in &[4, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = VectorH1Space::new(mesh, 1, 2);
        let dofs = space.n_dofs();
        if dofs > 2_000_000 && quick_mode() { break; }

        c.bench_function(&format!("Elasticity2D_VectorH1P1_{dofs}dof"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let mesh = SimplexMesh::<2>::unit_square_tri(n);
                    let space = VectorH1Space::new(mesh, 1, 2);
                    let _mat = Assembler::assemble_bilinear(&space, &[&VectorDiffusionIntegrator { kappa: 1.0 }], 3);
                    total += t0.elapsed();
                }
                total
            })
        });
    }
}

// ─── Maxwell 2D HCurl ND1 ────────────────────────────────────────────────

fn bench_maxwell(c: &mut Criterion) {
    for &n in &[4, 8, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = HCurlSpace::new(mesh, 1);
        let dofs = space.n_dofs();
        if dofs > 2_000_000 && quick_mode() { break; }

        c.bench_function(&format!("Maxwell2D_ND1_{dofs}dof"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let mesh = SimplexMesh::<2>::unit_square_tri(n);
                    let space = HCurlSpace::new(mesh, 1);
                    // Curl-curl + mass (time-harmonic like)
                    let mat = VectorAssembler::assemble_bilinear(
                        &space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], 4);
                    total += t0.elapsed();
                }
                total
            })
        });
    }
}

// ─── Stokes 2D Mixed HDiv×L² ────────────────────────────────────────────

fn bench_stokes(c: &mut Criterion) {
    for &n in &[3, 6, 9] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let vel_space = VectorH1Space::new(mesh.clone(), 2, 2);
        let pre_space = H1Space::new(mesh, 1);
        let n_v = vel_space.n_dofs();
        let n_p = pre_space.n_dofs();
        let total = n_v + n_p;
        if total > 2_000_000 && quick_mode() { break; }

        c.bench_function(&format!("Stokes2D_Mixed_{}vel_{}pre", n_v, n_p), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let mesh = SimplexMesh::<2>::unit_square_tri(n);
                    let vel_space = VectorH1Space::new(mesh.clone(), 2, 2);
                    let pre_space = H1Space::new(mesh, 1);

                    let n_v = vel_space.n_dofs();
                    let n_p = pre_space.n_dofs();
                    let dim = n_v + n_p;

                    let diff = VectorDiffusionIntegrator { kappa: 1.0 };
                    let mass = VectorH1MassIntegrator { kappa: 1.0 };
                    let mat_a = Assembler::assemble_bilinear(&vel_space, &[&diff, &mass], 5);
                    let mat_b = MixedAssembler::assemble_bilinear(
                        &pre_space, &vel_space, &[&DivIntegrator], 4);
                    let _mat_bt = mat_b.transpose();
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
    targets = bench_poisson, bench_elasticity, bench_maxwell, bench_stokes
}

criterion_main!(mfem_parity);
