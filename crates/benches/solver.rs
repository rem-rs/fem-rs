use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::GpuContext;
use fem_mesh::SimplexMesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_assembly::{Assembler, standard::DiffusionIntegrator};
use fem_solver::{
    SolverConfig, PrintLevel, solve_cg, solve_pcg_jacobi,
    cg_gpu::{CgGpuWorkspace, solve_cg_gpu, solve_cg_gpu_f32},
    gmres_gpu::{GmresGpuWorkspace, solve_gmres_gpu, solve_gmres_gpu_f32},
};
use std::time::Duration;

fn quick_bench_mode() -> bool {
    std::env::var("FEM_BENCH_QUICK").map(|value| value != "0").unwrap_or(false)
}

fn solver_criterion_config() -> Criterion {
    if quick_bench_mode() {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(250))
    } else {
        Criterion::default().sample_size(30)
    }
}

fn poisson_1d<T: Scalar>(n: usize) -> (CsrMatrix<T>, Vec<T>) {
    let nnz = 3 * n - 2;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for i in 0..n {
        row_ptr[i + 1] = row_ptr[i];
        if i > 0 {
            col_idx.push(i as u32 - 1);
            values.push(T::from_f64(-1.0));
            row_ptr[i + 1] += 1;
        }
        col_idx.push(i as u32);
        values.push(T::from_f64(2.0));
        row_ptr[i + 1] += 1;
        if i + 1 < n {
            col_idx.push(i as u32 + 1);
            values.push(T::from_f64(-1.0));
            row_ptr[i + 1] += 1;
        }
    }

    let a = CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values };
    let pi = std::f64::consts::PI;
    let x_exact: Vec<T> = (0..n)
        .map(|i| T::from_f64((pi * i as f64 / (n as f64 - 1.0)).sin()))
        .collect();
    let mut rhs = vec![T::zero(); n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut sum = T::zero();
        for k in start..end {
            sum += a.values[k] * x_exact[a.col_idx[k] as usize];
        }
        rhs[i] = sum;
    }
    (a, rhs)
}

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
            let cfg = SolverConfig {
                rtol: 1e-10,
                atol: 0.0,
                max_iter: 2000,
                verbose: false,
                print_level: PrintLevel::Silent,
            };

            b.iter(|| {
                let mut x = vec![0.0_f64; n_dofs];
                let result = solve_pcg_jacobi(&mat, &rhs, &mut x, &cfg);
                let _ = black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_cg_cpu_gpu(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let cfg_gpu = SolverConfig {
        rtol: 1e-4,
        atol: 1e-6,
        max_iter: 5000,
        verbose: false,
        print_level: PrintLevel::Silent,
    };
    let cfg_cpu = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        print_level: PrintLevel::Silent,
    };
    let sizes: &[usize] = if quick_bench_mode() { &[512, 2048] } else { &[512, 2048, 8192] };
    let mut group = c.benchmark_group("cg_compare");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            group.bench_with_input(BenchmarkId::new("cpu_f64", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f64; n];
                    let result = solve_cg(&a, &rhs, &mut x, &cfg_cpu);
                    let _ = black_box(result.expect("CPU CG f64 should converge"));
                });
            });
            group.bench_with_input(BenchmarkId::new("gpu_f64", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f64; n];
                    let result = solve_cg_gpu(&gpu, &a, &rhs, &mut x, &cfg_cpu);
                    let _ = black_box(result.expect("GPU CG f64 should converge"));
                });
            });
        }
    } else {
        for &n in sizes {
            let (a_cpu, rhs_cpu) = poisson_1d::<f64>(n);
            let (a_gpu, rhs_gpu) = poisson_1d::<f32>(n);
            let mut gpu_workspace = CgGpuWorkspace::new(&gpu, &a_gpu, &rhs_gpu);
            group.bench_with_input(BenchmarkId::new("cpu_f64_baseline", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f64; n];
                    let result = solve_cg(&a_cpu, &rhs_cpu, &mut x, &cfg_cpu);
                    let _ = black_box(result.expect("CPU CG f64 baseline should converge"));
                });
            });
            group.bench_with_input(BenchmarkId::new("gpu_f32", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f32; n];
                    let result = solve_cg_gpu_f32(&gpu, &a_gpu, &rhs_gpu, &mut x, &cfg_gpu);
                    let _ = black_box(result.expect("GPU CG f32 should converge"));
                });
            });
            group.bench_with_input(BenchmarkId::new("gpu_f32_reuse", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f32; n];
                    let result = gpu_workspace.solve(&gpu, &mut x, &cfg_gpu);
                    let _ = black_box(result.expect("GPU CG f32 reuse should converge"));
                });
            });
        }
    }

    group.finish();
}

fn bench_cg_gpu_setup(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[512, 2048] } else { &[512, 2048, 8192] };
    let mut group = c.benchmark_group("cg_setup");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            group.bench_with_input(BenchmarkId::new("gpu_f64_workspace", n), &n, |b, _| {
                b.iter(|| {
                    let workspace = CgGpuWorkspace::new(&gpu, &a, &rhs);
                    black_box(workspace);
                });
            });
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            group.bench_with_input(BenchmarkId::new("gpu_f32_workspace", n), &n, |b, _| {
                b.iter(|| {
                    let workspace = CgGpuWorkspace::new(&gpu, &a, &rhs);
                    black_box(workspace);
                });
            });
        }
    }

    group.finish();
}

fn bench_cg_fixed_iters(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[512, 2048] } else { &[512, 2048, 8192] };
    let fixed_iters: &[usize] = if quick_bench_mode() { &[8, 32] } else { &[8, 32, 128] };
    let mut group = c.benchmark_group("cg_fixed_iters");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            let mut workspace = CgGpuWorkspace::new(&gpu, &a, &rhs);
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f64_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let mut x = vec![0.0f64; n];
                        let residual = workspace.solve_fixed_iters(&gpu, &mut x, iters);
                        black_box((x, residual));
                    });
                });
            }
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            let mut workspace = CgGpuWorkspace::new(&gpu, &a, &rhs);
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f32_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let mut x = vec![0.0f32; n];
                        let residual = workspace.solve_fixed_iters(&gpu, &mut x, iters);
                        black_box((x, residual));
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_gmres_gpu_compare(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[64, 128] } else { &[64, 128, 256] };
    let mut group = c.benchmark_group("gmres_compare");

    if gpu.features.native_f64 {
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 1000,
            verbose: false,
            print_level: PrintLevel::Silent,
        };
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            group.bench_with_input(BenchmarkId::new("gpu_f64", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f64; n];
                    let result = solve_gmres_gpu(&gpu, &a, &rhs, &mut x, &cfg);
                    let _ = black_box(result.expect("GPU GMRES f64 should converge"));
                });
            });
            group.bench_with_input(BenchmarkId::new("gpu_f64_reuse", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f64; n];
                    let result = workspace.solve(&gpu, &mut x, &cfg);
                    let _ = black_box(result.expect("GPU GMRES f64 reuse should converge"));
                });
            });
        }
    } else {
        let cfg = SolverConfig {
            rtol: 1e-5,
            atol: 1e-6,
            max_iter: 1000,
            verbose: false,
            print_level: PrintLevel::Silent,
        };
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            group.bench_with_input(BenchmarkId::new("gpu_f32", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f32; n];
                    let result = solve_gmres_gpu_f32(&gpu, &a, &rhs, &mut x, &cfg);
                    let _ = black_box(result.expect("GPU GMRES f32 should converge"));
                });
            });
            group.bench_with_input(BenchmarkId::new("gpu_f32_reuse", n), &n, |b, _| {
                b.iter(|| {
                    let mut x = vec![0.0f32; n];
                    let result = workspace.solve(&gpu, &mut x, &cfg);
                    let _ = black_box(result.expect("GPU GMRES f32 reuse should converge"));
                });
            });
        }
    }

    group.finish();
}

fn bench_gmres_gpu_setup(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[64, 128] } else { &[64, 128, 256] };
    let mut group = c.benchmark_group("gmres_setup");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            group.bench_with_input(BenchmarkId::new("gpu_f64_workspace", n), &n, |b, _| {
                b.iter(|| {
                    let workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
                    black_box(workspace);
                });
            });
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            group.bench_with_input(BenchmarkId::new("gpu_f32_workspace", n), &n, |b, _| {
                b.iter(|| {
                    let workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
                    black_box(workspace);
                });
            });
        }
    }

    group.finish();
}

fn bench_gmres_fixed_iters(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[64, 128] } else { &[64, 128, 256] };
    let fixed_iters: &[usize] = if quick_bench_mode() { &[8, 16] } else { &[8, 16, 32] };
    let mut group = c.benchmark_group("gmres_fixed_iters");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f64; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f64_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f32; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f32_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_gmres_arnoldi_only_fixed_iters(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[64, 128] } else { &[64, 128, 256] };
    let fixed_iters: &[usize] = if quick_bench_mode() { &[8, 16] } else { &[8, 16, 32] };
    let mut group = c.benchmark_group("gmres_arnoldi_only_fixed_iters");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f64; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f64_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters_arnoldi_only(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f32; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f32_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters_arnoldi_only(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_gmres_spmv_only_fixed_iters(c: &mut Criterion) {
    let Ok(gpu) = GpuContext::new_sync() else {
        return;
    };
    let sizes: &[usize] = if quick_bench_mode() { &[64, 128] } else { &[64, 128, 256] };
    let fixed_iters: &[usize] = if quick_bench_mode() { &[8, 16] } else { &[8, 16, 32] };
    let mut group = c.benchmark_group("gmres_spmv_only_fixed_iters");

    if gpu.features.native_f64 {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f64>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f64; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f64_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters_spmv_only(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    } else {
        for &n in sizes {
            let (a, rhs) = poisson_1d::<f32>(n);
            let mut workspace = GmresGpuWorkspace::new(&gpu, &a, &rhs);
            let zero_x = vec![0.0f32; n];
            for &iters in fixed_iters {
                group.bench_with_input(BenchmarkId::new(format!("gpu_f32_reuse_{iters}iters"), n), &n, |b, _| {
                    b.iter(|| {
                        let residual = workspace.measure_fixed_iters_spmv_only(&gpu, &zero_x, iters);
                        black_box(residual);
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = solver_criterion_config();
    targets = bench_pcg, bench_cg_cpu_gpu, bench_cg_gpu_setup, bench_cg_fixed_iters, bench_gmres_gpu_compare, bench_gmres_gpu_setup, bench_gmres_fixed_iters, bench_gmres_arnoldi_only_fixed_iters, bench_gmres_spmv_only_fixed_iters
}
criterion_main!(benches);

