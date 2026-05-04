//! Micro-benchmarks for critical inner-loop operations.
//!
//! These benchmarks isolate performance-sensitive code paths:
//! - SpMV: core sparse matrix-vector multiplication
//! - Assembly: element-wise accumulation into COO format
//! - Sorting: triplet sorting for COO→CSR conversion

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fem_linalg::{CooMatrix, CsrMatrix, Vector};

// ─────────────────────────────────────────────────────────────────────────────
// SpMV Micro-Benchmark
// ─────────────────────────────────────────────────────────────────────────────

fn create_sparse_poisson_2d(n: usize) -> CsrMatrix<f64> {
    // 2D Poisson stencil on n×n grid
    let mut coo = CooMatrix::new(n * n, n * n);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            let mut diag = 4.0_f64;
            
            if i > 0 { coo.add(k, k - n, -1.0); diag -= 1.0; }
            if i < n - 1 { coo.add(k, k + n, -1.0); diag -= 1.0; }
            if j > 0 { coo.add(k, k - 1, -1.0); diag -= 1.0; }
            if j < n - 1 { coo.add(k, k + 1, -1.0); diag -= 1.0; }
            
            coo.add(k, k, diag);
        }
    }
    coo.into_csr()
}

fn spmv_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");
    
    for n in [32, 64, 128, 256].iter() {
        let mat = black_box(create_sparse_poisson_2d(*n));
        let x = black_box(vec![1.0_f64; n * n]);
        let mut y = vec![0.0_f64; n * n];
        
        group.bench_with_input(
            format!("serial_poisson_{}x{}", n, n),
            n,
            |b, _| b.iter(|| mat.spmv(&x, &mut y))
        );
    }

    // Parallel SpMV path (Rayon): force threshold to 1 via env is not available
    // in bench context, so we use sizes large enough to exceed the default 128-row
    // threshold and exercise the parallel code path.
    #[cfg(feature = "parallel")]
    for n in [64, 128, 256].iter() {
        let mat = black_box(create_sparse_poisson_2d(*n));
        let x = black_box(vec![1.0_f64; n * n]);
        let mut y = vec![0.0_f64; n * n];

        group.bench_with_input(
            format!("parallel_poisson_{}x{}", n, n),
            n,
            |b, _| b.iter(|| mat.spmv(&x, &mut y))
        );
    }
    
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly Micro-Benchmark (COO accumulation)
// ─────────────────────────────────────────────────────────────────────────────

fn assembly_coo_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_coo");
    
    for n_elems in [100, 1000, 10000].iter() {
        let n_dofs = (*n_elems as f64).sqrt() as usize * 4;  // Typical ~4 DOFs per element
        let n_elem_dofs = 9;  // TriP2 elements
        
        group.bench_with_input(
            format!("accumulate_{}_elems", n_elems),
            n_elems,
            |b, &n_elems| {
                b.iter(|| {
                    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
                    coo.reserve(n_elems * n_elem_dofs * n_elem_dofs);
                    
                    for e in 0..n_elems {
                        let mut k_elem = vec![1.0_f64; n_elem_dofs * n_elem_dofs];
                        k_elem[4] = 2.0;  // Vary one entry
                        
                        let dofs: Vec<usize> = (0..n_elem_dofs)
                            .map(|i| (e + i) % n_dofs)
                            .collect();
                        
                        coo.add_element_matrix(&dofs, &k_elem);
                    }
                    black_box(coo)
                })
            }
        );
    }
    
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// COO→CSR Conversion Benchmark (sorting dominates)
// ─────────────────────────────────────────────────────────────────────────────

fn coo_to_csr_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("coo_to_csr");
    
    for n_elems in [1000, 10000, 100000].iter() {
        let n_dofs = (*n_elems as f64).sqrt() as usize * 4;
        let n_elem_dofs = 9;
        
        group.bench_with_input(
            format!("sort_and_convert_{}_nnz", n_elems),
            n_elems,
            |b, &n_elems| {
                b.iter_batched(
                    || {
                        let mut coo = CooMatrix::new(n_dofs, n_dofs);
                        coo.reserve(n_elems * n_elem_dofs * n_elem_dofs);
                        for e in 0..n_elems {
                            let k_elem = vec![1.0_f64; n_elem_dofs * n_elem_dofs];
                            let dofs: Vec<usize> = (0..n_elem_dofs)
                                .map(|i| (e + i) % n_dofs)
                                .collect();
                            coo.add_element_matrix(&dofs, &k_elem);
                        }
                        coo
                    },
                    |coo| {
                        let _csr = black_box(coo.into_csr());
                    },
                    criterion::BatchSize::SmallInput,
                )
            }
        );
    }
    
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorting Micro-Benchmark (COO triplet sorting)
// ─────────────────────────────────────────────────────────────────────────────

fn triplet_sorting_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("triplet_sort");
    
    for n_triplets in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            format!("sort_{}_triplets", n_triplets),
            n_triplets,
            |b, &n_triplets| {
                b.iter_batched(
                    || {
                        let mut rows = Vec::with_capacity(n_triplets);
                        let mut cols = Vec::with_capacity(n_triplets);
                        for i in 0..n_triplets {
                            rows.push(((i * 17) % 1000) as u32);
                            cols.push(((i * 31) % 1000) as u32);
                        }
                        let idx: Vec<usize> = (0..n_triplets).collect();
                        (rows, cols, idx)
                    },
                    |(rows, cols, mut idx)| {
                        idx.sort_unstable_by_key(|&i| (rows[i], cols[i]));
                        black_box(idx)
                    },
                    criterion::BatchSize::SmallInput,
                )
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = spmv_benchmark, assembly_coo_benchmark, coo_to_csr_benchmark,
              triplet_sorting_benchmark, spmm_benchmark
);
criterion_main!(benches);

// ─────────────────────────────────────────────────────────────────────────────
// SpGEMM Benchmark: csr_spmm vs. naive COO round-trip
// ─────────────────────────────────────────────────────────────────────────────

/// Build a tridiagonal CSR matrix of size `n×n` with bandwidth 1.
fn tridiag_csr(n: usize) -> CsrMatrix<f64> {
    let mut c = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        c.add(i, i, 2.0);
        if i + 1 < n {
            c.add(i, i + 1, -1.0);
            c.add(i + 1, i, -1.0);
        }
    }
    c.into_csr()
}

fn spmm_benchmark(c: &mut Criterion) {
    use fem_linalg::csr_spmm;

    let mut group = c.benchmark_group("csr_spmm");
    group.sample_size(20);

    for n in [64usize, 256, 1024, 4096].iter().copied() {
        let a = tridiag_csr(n);
        let b = tridiag_csr(n);

        group.bench_with_input(
            format!("spmm_tridiag_{}x{}", n, n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let c = csr_spmm(black_box(&a), black_box(&b));
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}
