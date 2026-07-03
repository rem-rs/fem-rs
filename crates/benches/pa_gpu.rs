//! PA (Partial Assembly) benchmarks: CPU PA vs GPU PA for Hex Q1 diffusion.
//!
//! Acceptance criterion: GPU PA ≥ 5× faster than CPU PA at 1M DOF.
//!
//! Usage:
//!   cargo bench -p fem-benches --bench pa_gpu
//!   FEM_BENCH_QUICK=1 cargo bench -p fem-benches --bench pa_gpu  (quick mode)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use fem_assembly::pa::{build_hex_q1_pa_data, pa_apply_hex_q1};
use fem_linalg_gpu::pa_apply::gpu_pa_apply_hex_q1;
use fem_linalg_gpu::GpuContext;
use fem_mesh::topology::MeshTopology;
use fem_mesh::SimplexMesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

// ─── help: PA data layout ──────────────────────────────────────────────────

/// Convert CPU PaData (f64) to GPU flat f32 buffer with 11 values per QP.
fn pa_data_to_gpu(pd: &fem_assembly::pa::types::PaData) -> Vec<f32> {
    pd.data.iter().map(|&v| v as f32).collect()
}

/// Convert Vec<Vec<u32>> element DOFs to flat [n_elems × 8] u32 for GPU.
fn dofs_to_flat(elem_dofs: &[Vec<u32>]) -> Vec<u32> {
    elem_dofs.iter().flat_map(|d| d.iter().copied()).collect()
}

// ─── setup: mesh + space + PA data + GPU context ──────────────────────────

struct BenchSetup {
    n_dofs: usize,
    // CPU
    pd: fem_assembly::pa::types::PaData,
    elem_dofs: Vec<Vec<u32>>,
    x_cpu: Vec<f64>,
    // GPU
    pa_f32: Vec<f32>,
    dofs_flat: Vec<u32>,
    x_gpu: Vec<f32>,
    gpu_ctx: Option<GpuContext>,
}

fn setup(n: usize) -> BenchSetup {
    let mesh = SimplexMesh::<3>::unit_cube_hex(n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();

    let pd = build_hex_q1_pa_data(space.mesh(), &|_| 1.0);
    let elem_dofs: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
        .map(|e| space.element_dofs(e).to_vec())
        .collect();

    // Random x
    let mut rng: u64 = 42;
    let x_cpu: Vec<f64> = (0..n_dofs)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect();

    // GPU buffers
    let pa_f32 = pa_data_to_gpu(&pd);
    let dofs_flat = dofs_to_flat(&elem_dofs);
    let x_gpu: Vec<f32> = x_cpu.iter().map(|&v| v as f32).collect();

    let gpu_ctx = GpuContext::new_sync().ok();

    BenchSetup { n_dofs, pd, elem_dofs, x_cpu, pa_f32, dofs_flat, x_gpu, gpu_ctx }
}

// ─── benchmark functions ───────────────────────────────────────────────────

fn bench_pa_gpu(c: &mut Criterion) {
    let sizes = [10usize, 20, 40, 80];
    // Corresponding DOFs: ~1.3K, ~9K, ~69K, ~531K

    // ── CPU PA ──────────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let s = setup(n);
            group.bench_with_input(
                BenchmarkId::new("hex_q1", format!("{}dof", s.n_dofs)),
                &s,
                |b, s| {
                    b.iter(|| {
                        let mut y = vec![0.0; s.n_dofs];
                        pa_apply_hex_q1(
                            black_box(&s.pd),
                            black_box(&s.elem_dofs),
                            black_box(&s.x_cpu),
                            black_box(y.as_mut_slice()),
                        );
                    })
                },
            );
        }
        group.finish();
    }

    // ── CPU PA build (one‑time setup cost) ──────────────────────────────
    {
        let mut group = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let mesh_ref = black_box(&mesh);
            group.bench_with_input(
                BenchmarkId::new("hex_q1", format!("{}elem", n * n * n)),
                &n,
                |b, _n| b.iter(|| build_hex_q1_pa_data(mesh_ref, &|_| 1.0)),
            );
        }
        group.finish();
    }

    // ── GPU PA apply (end‑to‑end) ───────────────────────────────────────
    {
        let mut group = c.benchmark_group("pa_gpu_apply");
        for &n in &sizes {
            let s = setup(n);
            let gpu = match &s.gpu_ctx {
                Some(ctx) => ctx,
                None => {
                    eprintln!("  SKIP GPU benchmark (no wgpu context) for n={n}");
                    continue;
                }
            };
            // Warm‑up: run once to compile shaders
            let mut y_warm = vec![0.0f32; s.n_dofs];
            gpu_pa_apply_hex_q1(gpu, &s.pa_f32, &s.dofs_flat, &s.x_gpu, &mut y_warm);

            group.bench_with_input(
                BenchmarkId::new("hex_q1", format!("{}dof", s.n_dofs)),
                &s,
                |b, s| {
                    let gpu = s.gpu_ctx.as_ref().unwrap();
                    let mut y = vec![0.0f32; s.n_dofs];
                    b.iter(|| {
                        gpu_pa_apply_hex_q1(
                            black_box(gpu),
                            black_box(&s.pa_f32),
                            black_box(&s.dofs_flat),
                            black_box(&s.x_gpu),
                            black_box(y.as_mut_slice()),
                        );
                    })
                },
            );
        }
        group.finish();
    }
}

fn quick_bench_mode() -> bool {
    matches!(
        std::env::var("FEM_BENCH_QUICK")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn pa_gpu_criterion_config() -> Criterion {
    if quick_bench_mode() {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(250))
    } else {
        Criterion::default().sample_size(30)
    }
}

criterion_group! {
    name = pa_gpu;
    config = pa_gpu_criterion_config();
    targets = bench_pa_gpu
}
criterion_main!(pa_gpu);
