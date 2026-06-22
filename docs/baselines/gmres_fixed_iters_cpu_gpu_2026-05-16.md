# GMRES Fixed-Iters CPU/GPU Baseline (2026-05-16)

Purpose: record aligned fixed-work GMRES measurements after landing
`fixed_iters` support in `vendor/linger` and matching the CPU benchmark
semantics to the current `fem-rs` GPU benchmark path.

Method:
- Problem: 1D Poisson, `n = 64`
- GMRES restart: `30`
- Fixed inner work: `8` and `16` iterations
- CPU path: `vendor/linger` `bench_krylov` fixed-iters group
- GPU path: `crates/benches/solver.rs` `gmres_fixed_iters` group
- GPU runs used quick-mode Criterion settings (`FEM_BENCH_QUICK=1`)

## Exact benchmark IDs

CPU:
- `fixed_iters/GMRES_restart30_8iters/64`
- `fixed_iters/GMRES_restart30_16iters/64`

GPU:
- `gmres_fixed_iters/gpu_f32_reuse_8iters/64`
- `gmres_fixed_iters/gpu_f32_reuse_16iters/64`

## Measurements

| Path | Benchmark ID | Time range | Midpoint | Relative to CPU midpoint |
| --- | --- | --- | --- | --- |
| CPU | `fixed_iters/GMRES_restart30_8iters/64` | `14.498-14.740 us` | `14.614 us` | `1.0x` |
| GPU | `gmres_fixed_iters/gpu_f32_reuse_8iters/64` | `8.0917-8.4939 ms` | `8.3196 ms` | `569.3x` |
| CPU | `fixed_iters/GMRES_restart30_16iters/64` | `45.939-46.578 us` | `46.234 us` | `1.0x` |
| GPU | `gmres_fixed_iters/gpu_f32_reuse_16iters/64` | `25.531-32.976 ms` | `28.540 ms` | `617.3x` |

## Observations

- The aligned CPU/GPU comparison still shows a very large fixed-work gap:
  roughly `5.7e2x` at `8` iterations and `6.2e2x` at `16` iterations.
- The CPU path scales close to linearly across these two points:
  `46.234 us / 14.614 us ~= 3.16x` for `2x` more inner iterations. This is
  higher than ideal linear scaling but still remains in the same low-microsecond
  regime.
- The GPU path also grows with iteration count but remains in the tens of
  milliseconds: `28.540 ms / 8.3196 ms ~= 3.43x`.
- Setup is not the dominant explanation for this gap. The aligned CPU/GPU
  fixed-iters measurements reinforce the current hypothesis that per-iteration
  GPU synchronization/readback remains the controlling cost.
- Quick-mode GPU results remain noisy. The `16`-iteration run reported
  Criterion variance wide enough to show "no change" relative to a previous
  sample, so future comparisons should prefer isolated exact-ID reruns over
  mixed benchmark batches.

## Supporting microbenchmark evidence

Additional `gpu_micro` dot microbenchmarks were added to isolate the fixed cost
of GPU scalar readback at GMRES-sized vectors.

Exact benchmark IDs:

- `dot_compare/cpu_f32_len_64`
- `dot_compare/f32_len_64`

Measurements:

| Path | Benchmark ID | Time range | Midpoint |
| --- | --- | --- | --- |
| CPU | `dot_compare/cpu_f32_len_64` | `19.015-20.923 ns` | `19.842 ns` |
| GPU | `dot_compare/f32_len_64` | `209.53-255.11 us` | `230.34 us` |

Interpretation:

- A single GPU dot-readback at `len = 64` already costs about `0.23 ms`.
- GMRES(m) Arnoldi work for `8` fixed iterations performs roughly
  `1 + 2 + ... + 8 = 36` dot products, plus repeated norm/readback calls.
- `36 * 0.23 ms ~= 8.3 ms`, which is already on the same order as the measured
  `gmres_fixed_iters/gpu_f32_reuse_8iters/64` midpoint (`8.3196 ms`).
- This does not prove dot-readback is the *only* dominant cost, but it is now
  strong evidence that repeated scalar synchronization/readback is sufficient to
  explain most of the observed GMRES GPU runtime at these small sizes.

## Arnoldi-only decomposition

To separate Arnoldi inner work from the final `x += V y` update, the GPU
benchmark surface now includes an Arnoldi-only fixed-iters path that skips the
 backsolve/update stage while keeping the same fixed iteration count.

Exact benchmark IDs:

- `gmres_fixed_iters/gpu_f32_reuse_8iters/64`
- `gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_8iters/64`
- `gmres_fixed_iters/gpu_f32_reuse_16iters/64`
- `gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_16iters/64`

Measurements:

| Path | Benchmark ID | Time range | Midpoint |
| --- | --- | --- | --- |
| Full GMRES fixed-iters | `gmres_fixed_iters/gpu_f32_reuse_8iters/64` | `10.882-14.846 ms` | `12.874 ms` |
| Arnoldi-only | `gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_8iters/64` | `10.592-14.720 ms` | `12.657 ms` |
| Full GMRES fixed-iters | `gmres_fixed_iters/gpu_f32_reuse_16iters/64` | `32.974-43.333 ms` | `38.251 ms` |
| Arnoldi-only | `gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_16iters/64` | `30.396-36.496 ms` | `33.440 ms` |

Interpretation:

- At `8` iterations the two paths are effectively identical: the midpoint gap is
  `12.874 ms - 12.657 ms = 0.217 ms`, or about `1.7%` of the full runtime.
- At `16` iterations the Arnoldi-only path is still close to the full solve:
  `38.251 ms - 33.440 ms = 4.811 ms`, roughly `12.6%` of the full runtime.
- The final backsolve plus `x += V y` update is therefore not the controlling
  cost for these small GMRES runs. The dominant cost remains inside the Arnoldi
  loop, consistent with the dot-readback microbenchmark evidence above.

## Real-path segmented profile

To reduce Criterion noise and inspect the real solver path directly, a
`profile_fixed_iters` timing surface was added on `GmresGpuWorkspace` and run on
the same `n = 64`, `8`-iteration `f32` case.

Observed output:

| Segment | Time |
| --- | --- |
| total | `238.0824 ms` |
| residual | `11.174 ms` |
| basis seed | `1.1356 ms` |
| Arnoldi SpMV | `1.5556 ms` |
| Arnoldi orthogonalization | `194.2184 ms` |
| Arnoldi normalization | `28.922 ms` |
| solution update | `1.0582 ms` |
| finalization | `0 ns` |

Interpretation:

- This direct segmented timing agrees with the benchmark decomposition: the
  final solution update is negligible compared with Arnoldi inner work.
- In this run, Arnoldi orthogonalization alone accounts for about
  `194.2184 / 238.0824 ~= 81.6%` of total time.
- Arnoldi normalization is the next visible cost at about
  `28.922 / 238.0824 ~= 12.1%`.
- Residual setup, SpMV, basis seeding, and the final `x += V y` update are all
  secondary at this problem size.
- The current best-local optimization target is therefore the repeated
  orthogonalization readback/synchronization path, not solution update or outer
  residual cleanup.

## Reproduction commands

GPU:

```powershell
$env:FEM_BENCH_QUICK='1'
cargo bench -p fem-benches --bench solver gmres_fixed_iters/gpu_f32_reuse_8iters/64
cargo bench -p fem-benches --bench solver gmres_fixed_iters/gpu_f32_reuse_16iters/64
cargo bench -p fem-benches --bench solver gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_8iters/64
cargo bench -p fem-benches --bench solver gmres_arnoldi_only_fixed_iters/gpu_f32_reuse_16iters/64
cargo test -p fem-solver --features gpu --test gmres_gpu_test gmres_gpu_profile_fixed_iters_reports_segment_times -- --nocapture
```

CPU (standalone `linger` copy outside the root workspace):

```powershell
$src = 'C:/Users/lilu/works/fem-rs/vendor/linger'
$tmp = Join-Path $env:TEMP ('linger-compare-' + [guid]::NewGuid().ToString())
Copy-Item -Path $src -Destination $tmp -Recurse
Push-Location $tmp
cargo bench --bench bench_krylov fixed_iters/GMRES_restart30_8iters/64
cargo bench --bench bench_krylov fixed_iters/GMRES_restart30_16iters/64
```