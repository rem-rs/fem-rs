//! Benchmarks for solver comparison: CG vs GMRES vs BiCGSTAB vs IDR vs p-MG.
//!
//! Run: `cargo bench -p fem-benches --bench solver_comparison`
//! Quick mode: `FEM_BENCH_QUICK=1 cargo bench -p fem-benches --bench solver_comparison`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::env;
use std::time::Duration;

use fem_linalg::CsrMatrix;
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_solver::{
    SolverConfig, solve_cg, solve_pcg_jacobi, solve_pcg_ilu0,
    solve_gmres, solve_gmres_jacobi, solve_gmres_ilu0,
    solve_bicgstab, solve_idrs, solve_tfqmr,
    solve_sparse_cholesky,
    p_multigrid::{
        build_pmg_hierarchy_1d_laplacian, PmgPrecond, solve_vcycle_pmg,
    },
};

fn quick_mode() -> bool {
    env::var("FEM_BENCH_QUICK").map(|v| v != "0").unwrap_or(false)
}

fn bench_config() -> Criterion {
    if quick_mode() {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(250))
    } else {
        Criterion::default().sample_size(30)
    }
}

fn poisson_2d(n: usize) -> (CsrMatrix<f64>, Vec<f64>) {
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh, 1u8);
    let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
    let rhs = vec![1.0_f64; space.n_dofs()];
    (mat, rhs)
}

// ─── CG variants ──────────────────────────────────────────────────────────

fn bench_cg_variants(c: &mut Criterion) {
    let sizes: &[usize] = if quick_mode() { &[8, 16] } else { &[8, 16, 32] };
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };

    let mut group = c.benchmark_group("solver_cg");
    for &n in sizes {
        let (a, rhs) = poisson_2d(n);
        let dofs = a.nrows;
        group.bench_with_input(BenchmarkId::new("plain", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_cg(&a, &rhs, &mut x, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("pcg_jacobi", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_pcg_jacobi(&a, &rhs, &mut x, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("pcg_ilu0", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_pcg_ilu0(&a, &rhs, &mut x, &cfg)).ok();
            });
        });
    }
    group.finish();
}

// ─── GMRES variants ─────────────────────────────────────────────────────────

fn bench_gmres_variants(c: &mut Criterion) {
    let sizes: &[usize] = if quick_mode() { &[8, 16] } else { &[8, 16, 32] };
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };

    let mut group = c.benchmark_group("solver_gmres");
    for &n in sizes {
        let (a, rhs) = poisson_2d(n);
        let dofs = a.nrows;
        group.bench_with_input(BenchmarkId::new("plain(30)", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_gmres(&a, &rhs, &mut x, 30, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("jacobi(30)", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_gmres_jacobi(&a, &rhs, &mut x, 30, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("ilu0(30)", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_gmres_ilu0(&a, &rhs, &mut x, 30, &cfg)).ok();
            });
        });
    }
    group.finish();
}

// ─── Non-symmetric solvers ─────────────────────────────────────────────────

fn bench_nonsymmetric_solvers(c: &mut Criterion) {
    let sizes: &[usize] = if quick_mode() { &[8] } else { &[8, 16] };
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };

    let mut group = c.benchmark_group("solver_nonsymmetric");
    for &n in sizes {
        let (a, rhs) = poisson_2d(n);
        let dofs = a.nrows;

        group.bench_with_input(BenchmarkId::new("bicgstab", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_bicgstab(&a, &rhs, &mut x, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("idr(4)", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_idrs(&a, &rhs, &mut x, 4, &cfg)).ok();
            });
        });
        group.bench_with_input(BenchmarkId::new("tfqmr", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_tfqmr(&a, &rhs, &mut x, &cfg)).ok();
            });
        });
    }
    group.finish();
}

// ─── Direct solvers ────────────────────────────────────────────────────────

fn bench_direct_solvers(c: &mut Criterion) {
    let sizes: &[usize] = if quick_mode() { &[8, 16] } else { &[8, 16, 32] };
    let mut group = c.benchmark_group("solver_direct");
    for &n in sizes {
        let (a, rhs) = poisson_2d(n);
        let dofs = a.nrows;
        group.bench_with_input(BenchmarkId::new("cholesky", dofs), &n, |b, _| {
            b.iter(|| {
                black_box(solve_sparse_cholesky(&a, &rhs)).ok();
            });
        });
    }
    group.finish();
}

// ─── p-multigrid (1D Laplacian) ────────────────────────────────────────────

fn bench_p_multigrid(c: &mut Criterion) {
    let sizes: &[usize] = if quick_mode() { &[32, 64] } else { &[32, 64, 128] };
    let pmax: u8 = 4;
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 1_000, verbose: false, ..SolverConfig::default() };

    let mut group = c.benchmark_group("p_multigrid");
    for &n in sizes {
        let hierarchy = build_pmg_hierarchy_1d_laplacian(n, pmax);
        let dofs = hierarchy.levels[0].nrows;
        let prec = PmgPrecond::default();

        // CG (no prec) baseline
        let rhs = assemble_rhs_1d(n, pmax);
        group.bench_with_input(BenchmarkId::new("cg_plain", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_cg(&hierarchy.levels[0], &rhs, &mut x, &cfg)).ok();
            });
        });

        // p-MG V-cycle
        group.bench_with_input(BenchmarkId::new("pmg_vcycle", dofs), &n, |b, _| {
            b.iter(|| {
                let mut x = vec![0.0f64; dofs];
                black_box(solve_vcycle_pmg(&hierarchy, &rhs, &mut x, &prec, &cfg)).ok();
            });
        });
    }
    group.finish();
}

fn assemble_rhs_1d(n_elem: usize, p_max: u8) -> Vec<f64> {
    use fem_element::{ReferenceElement, lagrange::LinePk};
    use fem_linalg::{CooMatrix, CsrMatrix};
    let p = p_max;
    let re = LinePk(p);
    let quad = re.quadrature((p as usize + 2) * 2);
    let h = 1.0 / n_elem as f64;
    let n_dpe = (p as usize + 1) as usize;
    let n_total = n_elem * n_dpe + 1;
    let pi = std::f64::consts::PI;

    let mut coo = CooMatrix::new(n_total, 1);
    let mut phi = vec![0.0; re.n_dofs()];
    for e in 0..n_elem {
        let x0 = e as f64 * h;
        let x1 = (e + 1) as f64 * h;
        let jac = h / 2.0;
        let base = e * n_dpe;
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * jac;
            let xp = 0.5 * (x0 + x1) + 0.5 * h * xi[0];
            let f = pi * pi * (pi * xp).sin();
            for i in 0..n_dpe {
                coo.add((base + i).min(n_total - 1), 0, w * f * phi[i]);
            }
        }
    }
    let rhs_csr = CsrMatrix::from_coo(&coo);
    let mut rhs = vec![0.0; n_total];
    for i in 0..n_total { rhs[i] = rhs_csr.get(i, 0).unwrap_or(0.0); }
    rhs[0] = 0.0;
    rhs[n_total - 1] = 0.0;
    rhs
}

criterion_group! {
    name = benches;
    config = bench_config();
    targets = bench_cg_variants, bench_gmres_variants, bench_nonsymmetric_solvers, bench_direct_solvers, bench_p_multigrid
}
criterion_main!(benches);
