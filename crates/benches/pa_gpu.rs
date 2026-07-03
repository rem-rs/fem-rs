//! CPU PA benchmarks for all operators: Hex Q1–Q4, Quad Q1–Q2, Tet4.
//!
//! Usage:
//!   cargo bench -p fem-benches --bench pa_gpu
//!   FEM_BENCH_QUICK=1 cargo bench -p fem-benches --bench pa_gpu

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use fem_assembly::pa::*;
use fem_mesh::topology::MeshTopology;
use fem_mesh::SimplexMesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

// ─── helpers ────────────────────────────────────────────────────────────────

fn random_x(n: usize) -> Vec<f64> {
    let mut rng: u64 = 42;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect()
}

/// For Q2/Q3/Q4 on hex: synthetic consecutive DOFs, one block per element.
fn flat_elem_dofs(n_elems: usize, ldof: usize) -> Vec<Vec<u32>> {
    (0..n_elems)
        .map(|e| ((e * ldof) as u32..((e + 1) * ldof) as u32).collect())
        .collect()
}

// ─── Hex Q1 ─────────────────────────────────────────────────────────────────

fn bench_hex_q1(c: &mut Criterion) {
    let sizes = [10usize, 20, 40, 80];

    // apply
    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let space = H1Space::new(mesh, 1);
            let nd = space.n_dofs();
            let pd = build_hex_q1_pa_data(space.mesh(), &|_| 1.0);
            let ed: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
                .map(|e| space.element_dofs(e).to_vec()).collect();
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("hex_q1", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_hex_q1(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    // build
    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            g.bench_with_input(BenchmarkId::new("hex_q1", n * n * n), &mesh,
                |b, m| b.iter(|| build_hex_q1_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Quad Q1 ────────────────────────────────────────────────────────────────

fn bench_quad_q1(c: &mut Criterion) {
    let sizes = [20usize, 40, 80, 160];

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<2>::unit_square_quad(n);
            let space = H1Space::new(mesh, 1);
            let nd = space.n_dofs();
            let pd = build_quad_q1_pa_data(space.mesh(), &|_| 1.0);
            let ed: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
                .map(|e| space.element_dofs(e).to_vec()).collect();
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("quad_q1", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_quad_q1(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<2>::unit_square_quad(n);
            g.bench_with_input(BenchmarkId::new("quad_q1", n * n), &mesh,
                |b, m| b.iter(|| build_quad_q1_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Quad Q2 ────────────────────────────────────────────────────────────────

fn bench_quad_q2(c: &mut Criterion) {
    let sizes = [10usize, 20, 40, 80];

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<2>::unit_square_quad(n);
            let space = H1Space::new(mesh, 2);
            let nd = space.n_dofs();
            let pd = build_quad_q2_pa_data(space.mesh(), &|_| 1.0);
            let ed: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
                .map(|e| space.element_dofs(e).to_vec()).collect();
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("quad_q2", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_quad_q2(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<2>::unit_square_quad(n);
            g.bench_with_input(BenchmarkId::new("quad_q2", n * n), &mesh,
                |b, m| b.iter(|| build_quad_q2_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Tet4 ───────────────────────────────────────────────────────────────────

fn bench_tet4(c: &mut Criterion) {
    let sizes = [2usize, 4, 6, 8];

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_tet(n);
            let space = H1Space::new(mesh, 1);
            let nd = space.n_dofs();
            let pd = build_tet4_pa_data(space.mesh(), &|_| 1.0);
            let ed: Vec<Vec<u32>> = (0..space.mesh().n_elements() as u32)
                .map(|e| space.element_dofs(e).to_vec()).collect();
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("tet4", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_tet4(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_tet(n);
            g.bench_with_input(BenchmarkId::new("tet4", n * n * n * 6), &mesh,
                |b, m| b.iter(|| build_tet4_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Hex Q2 (synthetic DOFs, 27 nodes per element) ─────────────────────────

fn bench_hex_q2(c: &mut Criterion) {
    let sizes = [2usize, 4, 8, 16];  // n³ elements

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let ne = mesh.n_elements();
            let nd = ne * 27;
            let pd = build_hex_q2_pa_data(&mesh, &|_| 1.0);
            let ed = flat_elem_dofs(ne, 27);
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("hex_q2", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_hex_q2(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            g.bench_with_input(BenchmarkId::new("hex_q2", n * n * n), &mesh,
                |b, m| b.iter(|| build_hex_q2_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Hex Q3 (synthetic DOFs, 64 nodes per element) ─────────────────────────

fn bench_hex_q3(c: &mut Criterion) {
    let sizes = [2usize, 4, 6];

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let ne = mesh.n_elements();
            let nd = ne * 64;
            let pd = build_hex_q3_pa_data(&mesh, &|_| 1.0);
            let ed = flat_elem_dofs(ne, 64);
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("hex_q3", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_hex_q3(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    // sum-factorized version
    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let ne = mesh.n_elements();
            let nd = ne * 64;
            let pd = build_hex_q3_pa_data(&mesh, &|_| 1.0);
            let ed = flat_elem_dofs(ne, 64);
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("hex_q3_sf", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_hex_q3_sf(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            g.bench_with_input(BenchmarkId::new("hex_q3", n * n * n), &mesh,
                |b, m| b.iter(|| build_hex_q3_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── Hex Q4 (synthetic DOFs, 125 nodes per element) ────────────────────────

fn bench_hex_q4(c: &mut Criterion) {
    let sizes = [2usize, 3];

    {
        let mut g = c.benchmark_group("pa_cpu_apply");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            let ne = mesh.n_elements();
            let nd = ne * 125;
            let pd = build_hex_q4_pa_data(&mesh, &|_| 1.0);
            let ed = flat_elem_dofs(ne, 125);
            let x = random_x(nd);
            g.bench_with_input(BenchmarkId::new("hex_q4", nd), &(pd, ed, x, nd),
                |b, (pd, ed, x, nd)| b.iter(|| {
                    let mut y = vec![0.0; *nd];
                    pa_apply_hex_q4(black_box(pd), black_box(ed), black_box(x), black_box(y.as_mut_slice()));
                })
            );
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group("pa_cpu_build");
        for &n in &sizes {
            let mesh = SimplexMesh::<3>::unit_cube_hex(n);
            g.bench_with_input(BenchmarkId::new("hex_q4", n * n * n), &mesh,
                |b, m| b.iter(|| build_hex_q4_pa_data(black_box(m), &|_| 1.0))
            );
        }
        g.finish();
    }
}

// ─── main entry ─────────────────────────────────────────────────────────────

fn bench_pa_all(c: &mut Criterion) {
    bench_hex_q1(c);
    bench_quad_q1(c);
    bench_quad_q2(c);
    bench_tet4(c);
    bench_hex_q2(c);
    bench_hex_q3(c);
    bench_hex_q4(c);
}

fn quick_bench_mode() -> bool {
    matches!(
        std::env::var("FEM_BENCH_QUICK").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn pa_criterion_config() -> Criterion {
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
    config = pa_criterion_config();
    targets = bench_pa_all
}
criterion_main!(pa_gpu);
