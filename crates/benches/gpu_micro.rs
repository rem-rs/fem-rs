use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use fem_core::Scalar;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_linalg_gpu::{DeviceBuffer, GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline, VectorOpsPipeline};
use std::time::Duration;

fn create_sparse_poisson_1d<T: Scalar>(n: usize) -> CsrMatrix<T> {
    let mut coo = CooMatrix::<T>::new(n, n);
    for i in 0..n {
        if i > 0 {
            coo.add(i, i - 1, T::from_f64(-1.0));
        }
        coo.add(i, i, T::from_f64(2.0));
        if i + 1 < n {
            coo.add(i, i + 1, T::from_f64(-1.0));
        }
    }
    coo.into_csr()
}

fn quick_bench_mode() -> bool {
    matches!(std::env::var("FEM_BENCH_QUICK").ok().as_deref(), Some("1" | "true" | "TRUE" | "yes" | "YES"))
}

fn gpu_micro_criterion_config() -> Criterion {
    if quick_bench_mode() {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(250))
    } else {
        Criterion::default().sample_size(30)
    }
}

fn try_gpu_context() -> Option<GpuContext> {
    match GpuContext::new_sync() {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            eprintln!("Skipping gpu_micro benches: {err}");
            None
        }
    }
}

fn create_sparse_poisson_2d<T: Scalar>(n: usize) -> CsrMatrix<T> {
    let mut coo = CooMatrix::<T>::new(n * n, n * n);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            let mut diag = 4.0_f64;

            if i > 0 {
                coo.add(k, k - n, T::from_f64(-1.0));
                diag -= 1.0;
            }
            if i + 1 < n {
                coo.add(k, k + n, T::from_f64(-1.0));
                diag -= 1.0;
            }
            if j > 0 {
                coo.add(k, k - 1, T::from_f64(-1.0));
                diag -= 1.0;
            }
            if j + 1 < n {
                coo.add(k, k + 1, T::from_f64(-1.0));
                diag -= 1.0;
            }

            coo.add(k, k, T::from_f64(diag));
        }
    }
    coo.into_csr()
}

fn bench_gpu_spmv_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, gpu: &GpuContext, sizes: &[usize], label_prefix: &str) {
    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    for n in sizes.iter().copied() {
        let cpu_mat = create_sparse_poisson_2d::<T>(n);
        let gpu_mat = GpuCsrMatrix::<T>::from_cpu(gpu, &cpu_mat);
        let x = GpuVector::from_slice(gpu, &vec![T::from_f64(1.0); n * n]);
        let y = GpuVector::<T>::zeros(gpu, (n * n) as u32);

        group.bench_function(format!("{}_{}x{}", label_prefix, n, n), |b| {
            b.iter(|| {
                pipeline.spmv(gpu, 1.0, black_box(&gpu_mat), black_box(&x), 0.0, black_box(&y));
            })
        });
    }
}

fn bench_gpu_norm2_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, gpu: &GpuContext, sizes: &[u32], label_prefix: &str) {
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);
    for len in sizes.iter().copied() {
        let host: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(1.0 + (i % 17) as f64 * 0.0625))
            .collect();
        let vec = GpuVector::from_slice(gpu, &host);

        group.bench_function(format!("{}_len_{}", label_prefix, len), |b| {
            b.iter(|| {
                let norm = pipeline.compute_norm2(gpu, black_box(&vec));
                black_box(norm)
            })
        });
    }
}

fn bench_gpu_dot_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, gpu: &GpuContext, sizes: &[u32], label_prefix: &str) {
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);
    for len in sizes.iter().copied() {
        let host_a: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(1.0 + (i % 17) as f64 * 0.0625))
            .collect();
        let host_b: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(0.5 + (i % 11) as f64 * 0.03125))
            .collect();
        let vec_a = GpuVector::from_slice(gpu, &host_a);
        let vec_b = GpuVector::from_slice(gpu, &host_b);
        let partial_count = (len + 255) / 256;
        let scratch = DeviceBuffer::with_staging(
            &gpu.device,
            partial_count as u64 * std::mem::size_of::<T>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "dot_tmp",
        );

        group.bench_function(format!("{}_len_{}", label_prefix, len), |b| {
            b.iter(|| {
                let dot = pipeline.dispatch_dot_readback(gpu, black_box(&vec_a), black_box(&vec_b), black_box(&scratch));
                black_box(dot)
            })
        });
    }
}

fn bench_gpu_orthogonalization_chain_for_type<T: Scalar>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    gpu: &GpuContext,
    counts: &[usize],
    len: u32,
    label_prefix: &str,
) {
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);
    let seed_w: Vec<T> = (0..len as usize)
        .map(|i| T::from_f64(1.0 + (i % 13) as f64 * 0.09375))
        .collect();
    let gpu_w = GpuVector::from_slice(gpu, &seed_w);
    let basis: Vec<GpuVector<T>> = counts
        .iter()
        .copied()
        .max()
        .map(|max_count| {
            (0..max_count)
                .map(|j| {
                    let host_basis: Vec<T> = (0..len as usize)
                        .map(|i| T::from_f64(0.25 + ((i + j) % 17) as f64 * 0.03125))
                        .collect();
                    GpuVector::from_slice(gpu, &host_basis)
                })
                .collect()
        })
        .unwrap_or_default();
    let partial_count = (len + 255) / 256;
    let scratch = DeviceBuffer::with_staging(
        &gpu.device,
        partial_count as u64 * std::mem::size_of::<T>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "orth_chain_tmp",
    );

    for &count in counts {
        group.bench_function(format!("{}_len_{}_count_{}", label_prefix, len, count), |b| {
            b.iter(|| {
                gpu_w.write_from_slice(gpu, &seed_w);
                for basis_vec in basis.iter().take(count) {
                    let dot = pipeline.dispatch_dot_readback(gpu, black_box(&gpu_w), black_box(basis_vec), black_box(&scratch));
                    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    pipeline.encode_axpy(gpu, &mut enc, -dot, basis_vec, 1.0, &gpu_w);
                    gpu.queue.submit(Some(enc.finish()));
                }
                let norm = pipeline.compute_norm2(gpu, black_box(&gpu_w));
                black_box(norm)
            })
        });
    }
}

fn bench_gpu_spmv_norm_chain_for_type<T: Scalar>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    gpu: &GpuContext,
    counts: &[usize],
    len: u32,
    label_prefix: &str,
) {
    let spmv = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    let vops = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);
    let cpu_mat = create_sparse_poisson_1d::<T>(len as usize);
    let gpu_mat = GpuCsrMatrix::<T>::from_cpu(gpu, &cpu_mat);
    let seed_basis: Vec<T> = (0..len as usize)
        .map(|i| T::from_f64(1.0 + (i % 7) as f64 * 0.125))
        .collect();
    let max_count = counts.iter().copied().max().unwrap_or(0);
    let basis: Vec<GpuVector<T>> = (0..=max_count)
        .map(|_| GpuVector::zeros(gpu, len))
        .collect();
    let gpu_w = GpuVector::zeros(gpu, len);

    for &count in counts {
        group.bench_function(format!("{}_len_{}_count_{}", label_prefix, len, count), |b| {
            b.iter(|| {
                basis[0].write_from_slice(gpu, &seed_basis);
                for jj in 0..count {
                    spmv.spmv(gpu, 1.0, black_box(&gpu_mat), black_box(&basis[jj]), 0.0, black_box(&gpu_w));
                    let w_norm = vops.compute_norm2(gpu, black_box(&gpu_w));
                    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    vops.encode_axpy(gpu, &mut enc, 1.0 / w_norm, &gpu_w, 0.0, &basis[jj + 1]);
                    gpu.queue.submit(Some(enc.finish()));
                }
                let norm = vops.compute_norm2(gpu, black_box(&basis[count]));
                black_box(norm)
            })
        });
    }
}

fn bench_cpu_spmv_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, sizes: &[usize], label_prefix: &str) {
    for n in sizes.iter().copied() {
        let mat = create_sparse_poisson_2d::<T>(n);
        let x = vec![T::from_f64(1.0); n * n];
        let mut y = vec![T::zero(); n * n];

        group.bench_function(format!("{}_{}x{}", label_prefix, n, n), |b| {
            b.iter(|| {
                mat.spmv(black_box(&x), black_box(&mut y));
            })
        });
    }
}

fn bench_cpu_norm2_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, sizes: &[u32], label_prefix: &str) {
    for len in sizes.iter().copied() {
        let host: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(1.0 + (i % 17) as f64 * 0.0625))
            .collect();

        group.bench_function(format!("{}_len_{}", label_prefix, len), |b| {
            b.iter(|| {
                let norm2 = host
                    .iter()
                    .copied()
                    .fold(T::zero(), |acc, value| acc + value * value)
                    .sqrt();
                black_box(norm2)
            })
        });
    }
}

fn bench_cpu_dot_for_type<T: Scalar>(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>, sizes: &[u32], label_prefix: &str) {
    for len in sizes.iter().copied() {
        let host_a: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(1.0 + (i % 17) as f64 * 0.0625))
            .collect();
        let host_b: Vec<T> = (0..len as usize)
            .map(|i| T::from_f64(0.5 + (i % 11) as f64 * 0.03125))
            .collect();

        group.bench_function(format!("{}_len_{}", label_prefix, len), |b| {
            b.iter(|| {
                let dot = host_a
                    .iter()
                    .zip(&host_b)
                    .fold(T::zero(), |acc, (&lhs, &rhs)| acc + lhs * rhs);
                black_box(dot)
            })
        });
    }
}

fn gpu_spmv_benchmark(c: &mut Criterion) {
    let Some(gpu) = try_gpu_context() else {
        return;
    };
    let mut group = c.benchmark_group("spmv_compare");
    let sizes: &[usize] = if quick_bench_mode() { &[128, 256, 512] } else { &[128, 256, 512, 1024] };
    if gpu.features.native_f64 {
        bench_cpu_spmv_for_type::<f64>(&mut group, sizes, "cpu_f64_poisson");
        bench_gpu_spmv_for_type::<f64>(&mut group, &gpu, sizes, "f64_poisson");
    } else {
        bench_cpu_spmv_for_type::<f32>(&mut group, sizes, "cpu_f32_poisson");
        bench_gpu_spmv_for_type::<f32>(&mut group, &gpu, sizes, "f32_poisson");
    }

    group.finish();
}

fn gpu_norm2_benchmark(c: &mut Criterion) {
    let Some(gpu) = try_gpu_context() else {
        return;
    };
    let mut group = c.benchmark_group("norm2_compare");
    let sizes: &[u32] = if quick_bench_mode() { &[1 << 14, 1 << 16, 1 << 18] } else { &[1 << 14, 1 << 16, 1 << 18, 1 << 20] };
    if gpu.features.native_f64 {
        bench_cpu_norm2_for_type::<f64>(&mut group, sizes, "cpu_f64");
        bench_gpu_norm2_for_type::<f64>(&mut group, &gpu, sizes, "f64");
    } else {
        bench_cpu_norm2_for_type::<f32>(&mut group, sizes, "cpu_f32");
        bench_gpu_norm2_for_type::<f32>(&mut group, &gpu, sizes, "f32");
    }

    group.finish();
}

fn gpu_dot_benchmark(c: &mut Criterion) {
    let Some(gpu) = try_gpu_context() else {
        return;
    };
    let mut group = c.benchmark_group("dot_compare");
    let sizes: &[u32] = if quick_bench_mode() { &[64, 128, 512, 1 << 14] } else { &[64, 128, 512, 1 << 14, 1 << 16] };
    if gpu.features.native_f64 {
        bench_cpu_dot_for_type::<f64>(&mut group, sizes, "cpu_f64");
        bench_gpu_dot_for_type::<f64>(&mut group, &gpu, sizes, "f64");
    } else {
        bench_cpu_dot_for_type::<f32>(&mut group, sizes, "cpu_f32");
        bench_gpu_dot_for_type::<f32>(&mut group, &gpu, sizes, "f32");
    }

    group.finish();
}

fn gpu_arnoldi_chain_benchmark(c: &mut Criterion) {
    let Some(gpu) = try_gpu_context() else {
        return;
    };
    let mut group = c.benchmark_group("arnoldi_chain_compare");
    let counts: &[usize] = if quick_bench_mode() { &[8, 16] } else { &[8, 16, 32] };
    let len = 64;
    if gpu.features.native_f64 {
        bench_gpu_orthogonalization_chain_for_type::<f64>(&mut group, &gpu, counts, len, "f64_orth");
        bench_gpu_spmv_norm_chain_for_type::<f64>(&mut group, &gpu, counts, len, "f64_spmv_norm");
    } else {
        bench_gpu_orthogonalization_chain_for_type::<f32>(&mut group, &gpu, counts, len, "f32_orth");
        bench_gpu_spmv_norm_chain_for_type::<f32>(&mut group, &gpu, counts, len, "f32_spmv_norm");
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = gpu_micro_criterion_config();
    targets = gpu_spmv_benchmark, gpu_norm2_benchmark, gpu_dot_benchmark, gpu_arnoldi_chain_benchmark
}
criterion_main!(benches);