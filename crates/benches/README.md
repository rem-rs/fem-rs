# fem-rs Benchmarks

Performance benchmarks for core fem-rs operations using Criterion.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p fem-benches

# Run specific benchmark
cargo bench -p fem-benches --bench assembly
cargo bench -p fem-benches --bench solver
cargo bench -p fem-benches --bench amg
cargo bench -p fem-benches --bench mesh
cargo bench -p fem-benches --bench micro

# Compare serial vs parallel assembly paths
cargo bench -p fem-benches --bench assembly
cargo bench -p fem-benches --features parallel --bench assembly

# Reproducible serial/parallel compare (PowerShell)
pwsh scripts/run_assembly_parallel_bench.ps1 -Threads 8 -ParallelMinElems 64
pwsh scripts/run_assembly_parallel_bench.ps1 -CompileOnly
pwsh scripts/run_assembly_parallel_bench.ps1 -Threads 8 -ExportStamp 2026-04-25-parallel-pass1
pwsh scripts/run_assembly_parallel_bench.ps1 -Mode parallel -Filter assembly_dg_faces
pwsh scripts/run_assembly_parallel_bench.ps1 -Mode serial -Filter tangential_mass_nd1
pwsh scripts/run_spmv_parallel_bench.ps1 -Quick

# Run benchmarks without saving results (for quick testing)
cargo bench -p fem-benches -- --test
```

## Benchmark Categories

### Assembly (`assembly.rs`)
- Poisson matrix assembly with P1/P2 elements
- H(curl) tangential boundary mass assembly
- DG SIP matrix assembly (volume + interior/boundary faces)
- Sparsity pattern construction

### Solver (`solver.rs`)
- PCG with Jacobi preconditioner
- Convergence at different mesh resolutions

### AMG (`amg.rs`)
- AMG hierarchy setup time
- V-cycle solve performance

### Mesh (`mesh.rs`)
- Mesh generation (2D/3D)
- Uniform refinement

### Micro (`micro.rs`)
- SpMV micro-benchmarks for serial and parallel Poisson stencils
- COO accumulation, COO→CSR conversion, triplet sorting
- Quick CI-oriented mode via `FEM_BENCH_QUICK=1`

## Output

Benchmark results are saved to `target/criterion/` directory with HTML reports.
Open `target/criterion/report/index.html` in a browser to view detailed results.

## Reproducible Assembly Compare

Use `scripts/run_assembly_parallel_bench.ps1` to run serial and parallel assembly
benchmarks back-to-back with controlled environment:

- `-Threads` sets `RAYON_NUM_THREADS` (default: Rayon auto).
- `-ParallelMinElems` sets `FEM_ASSEMBLY_PARALLEL_MIN_ELEMS` (default: `64`).
- `-CompileOnly` uses `--no-run` for quick validation in CI/dev loops.
- `-ExportStamp` copies `target/criterion` to `target/criterion-history/<stamp>` for trend tracking.
- `-Mode` controls which run is executed: `all` (default), `serial`, or `parallel`.
- `-Filter` forwards a Criterion filter (e.g. `assembly_dg_faces`, `tangential_mass_nd1`).
