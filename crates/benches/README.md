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

# Run benchmarks without saving results (for quick testing)
cargo bench -p fem-benches -- --test
```

## Benchmark Categories

### Assembly (`assembly.rs`)
- Poisson matrix assembly with P1/P2 elements
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

## Output

Benchmark results are saved to `target/criterion/` directory with HTML reports.
Open `target/criterion/report/index.html` in a browser to view detailed results.
