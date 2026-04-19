# fem-rs vs deal.ii Quantitative KPI Benchmark Plan (2026-04-19)

## Scope

This document defines a measurable benchmark protocol for comparing fem-rs with deal.ii.
The goal is reproducible, command-backed evidence across correctness, scalability, robustness, and efficiency.

## Comparison Rules

- Use equivalent PDE and discretization order when possible.
- Run on the same machine/cluster, same core count, same compiler optimization level.
- Record complete runtime context: CPU model, memory, MPI runtime, compiler flags, mesh size.
- Separate warmup from measured runs.
- Repeat each case at least 5 times and report median and p95.

## KPI Matrix

| KPI ID | Dimension | Metric | Unit | Better | Acceptance Gate (fem-rs target) |
|---|---|---|---|---|---|
| K1 | Correctness | Relative L2 error vs analytical/reference solution | ratio | lower | <= 1e-6 for baseline anchors |
| K2 | Solver quality | Final relative residual | ratio | lower | <= 1e-8 |
| K3 | Solver efficiency | Krylov iterations to convergence | count | lower | within +25% of deal.ii baseline |
| K4 | Strong scaling | Parallel efficiency E_s = T1 / (P * Tp) | ratio | higher | >= 0.70 at P=16 |
| K5 | Weak scaling | Time growth at fixed DOF/rank | ratio | lower | <= 1.25x from 1 to 16 ranks |
| K6 | Memory efficiency | Peak RSS per DOF | bytes/DOF | lower | within +20% of deal.ii baseline |
| K7 | Robustness | Failure rate across stress set | ratio | lower | 0 failures in mandatory set |
| K8 | Startup overhead | Setup time / total time | ratio | lower | <= 0.40 for repeated solve workflow |
| K9 | Throughput | DOF solved per second | DOF/s | higher | within 0.8x of deal.ii baseline initially |
| K10 | Dev productivity | Time-to-add-new-case (engineering task) | hours | lower | tracked, trend non-increasing |

## Mandatory Case Set

### Case A: Scalar Poisson (serial + MPI)

- Physics: -Delta u = f on unit square
- Goal: correctness + scaling baseline
- fem-rs anchor:
  - examples/mfem_ex1_poisson.rs
  - examples/mfem_pex1_poisson.rs

### Case B: H(curl) Maxwell-like system

- Goal: vector-space robustness and solver behavior
- fem-rs anchor:
  - examples/mfem_ex3.rs
  - examples/mfem_pex3_maxwell.rs

### Case C: Mixed Darcy / saddle-point

- Goal: block system behavior and robustness
- fem-rs anchor:
  - examples/mfem_ex5_mixed_darcy.rs
  - examples/mfem_pex2_mixed_darcy.rs

## fem-rs Command Protocol

### 1. Correctness anchors

Run and capture pass/fail, residual, and runtime:

```powershell
cargo test -p fem-assembly transfer::tests::conservative_projection_3d_matches_global_integral -- --nocapture
cargo test -p fem-examples --example mfem_ex48_template_joule_heating --example mfem_ex49_template_fsi --example mfem_ex50_template_acoustics_structure --example mfem_ex51_template_em_thermal_stress --example mfem_ex52_template_reaction_flow_thermal
```

### 2. RAS/Krylov baseline (already available in fem-parallel)

```powershell
$env:RAS_BENCH_CSV = "output/ras_benchmark.csv"
cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

### 3. MPI scaling sweep template

Use rank set P={1,2,4,8,16} and fixed problem size for strong scaling.

```powershell
# Replace executable/args with your benchmark harness once selected.
# Example shape only:
# mpiexec -n 1  cargo run -p fem-examples --example mfem_pex1_poisson --release -- <args>
# mpiexec -n 2  cargo run -p fem-examples --example mfem_pex1_poisson --release -- <args>
# mpiexec -n 4  cargo run -p fem-examples --example mfem_pex1_poisson --release -- <args>
# mpiexec -n 8  cargo run -p fem-examples --example mfem_pex1_poisson --release -- <args>
# mpiexec -n 16 cargo run -p fem-examples --example mfem_pex1_poisson --release -- <args>
```

### 4. Weak scaling sweep template

Scale global DOF linearly with rank count to keep DOF/rank constant.

```powershell
# P=1 : mesh size N
# P=2 : mesh size ~sqrt(2)*N in 2D (or equivalent DOF doubling)
# P=4 : mesh size ~2*N in 2D (or equivalent DOF quadrupling)
# Continue similarly and record wall time, residual, iterations.
```

## deal.ii Side Collection Template

For each mandatory case (A/B/C), collect the same fields:

- mesh and DOF
- total time (median/p95)
- setup time
- solve time
- iterations
- final residual
- peak memory
- rank count

Note: this repository does not include deal.ii sources/binaries, so data must be imported from external runs.

## Result Table Template

| Case | Ranks | DOF | fem-rs time (ms) | deal.ii time (ms) | fem-rs iters | deal.ii iters | fem-rs residual | deal.ii residual | fem-rs peak RSS (MB) | deal.ii peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 |  |  |  |  |  |  |  |  |  |
| A | 2 |  |  |  |  |  |  |  |  |  |
| A | 4 |  |  |  |  |  |  |  |  |  |
| B | 1 |  |  |  |  |  |  |  |  |  |
| B | 2 |  |  |  |  |  |  |  |  |  |
| C | 1 |  |  |  |  |  |  |  |  |  |

## Derived Metrics

Compute after data collection:

- Strong-scaling efficiency:
  E_s(P) = T(1) / (P * T(P))
- Weak-scaling growth:
  G_w(P) = T(P) / T(1)
- Iteration inflation ratio:
  R_it = iters_fem_rs / iters_dealii
- Memory ratio:
  R_mem = mem_fem_rs / mem_dealii

## Current Read of Positioning

Based on current repository state:

- fem-rs likely strong in language safety, architecture clarity, and rapid feature iteration.
- deal.ii likely strong in ultra-large-scale production maturity and ecosystem depth.
- This KPI sheet is intended to turn that qualitative view into hard numbers.
