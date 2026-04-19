# RAS DDM Status (2026-04-19)

## Scope

This note tracks the current Domain Decomposition rollout in `fem-parallel`.
The implemented method is Restricted Additive Schwarz (RAS) used as a
preconditioner for parallel Krylov solvers.

## Implemented

- RAS preconditioner API in `crates/parallel/src/par_ras.rs`.
- Solver entrypoints:
  - `par_solve_pcg_ras`
  - `par_solve_gmres_ras`
- Local solver kernels:
  - `DiagJacobi`
  - `Ilu0`
- HPC diagnostics:
  - `RasHpcDiagnostics`
  - `summarize_ras_hpc(&ParCsrMatrix)`
  - Verbose solver mode now logs rank-aggregated health stats
    (owned/ghost distribution, diag/offd nnz, CV imbalance indicators).
- Overlap support:
  - `overlap = 0`
  - `overlap = 1` (multiplicative two-stage overlap correction)
  - `overlap > 1` currently rejected with a clear error.

## Regression Coverage

Current RAS regression tests are in `crates/parallel/src/par_ras.rs` and include:

- Build-time overlap contract checks.
- Serial and 2-rank convergence checks for PCG + RAS.
- Serial and 2-rank convergence checks for GMRES + RAS.
- Diag and ILU0 local-kernel paths.
- Stability checks across overlap modes.

Run:

```bash
cargo test -p fem-parallel par_ras -- --nocapture
```

## Benchmark Entry

A benchmark-style ignored test is provided in:

- `crates/parallel/tests/ras_benchmark.rs`

Run explicitly:

```bash
cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

Optional CSV export:

PowerShell:

```powershell
$env:RAS_BENCH_CSV = "output/ras_benchmark.csv"
cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

Bash:

```bash
RAS_BENCH_CSV=output/ras_benchmark.csv cargo test -p fem-parallel ras_benchmark_report_two_ranks -- --ignored --nocapture
```

The benchmark currently reports:

- `pcg_ras_diag_ov0`
- `pcg_ras_diag_ov1`
- `pcg_ras_ilu0_ov0`
- `pcg_ras_ilu0_ov1`
- `gmres_ras_diag_ov0`
- `gmres_ras_diag_ov1`
- `gmres_ras_ilu0_ov0`
- `gmres_ras_ilu0_ov1`

Each benchmark row now includes HPC diagnostics columns:

- `ranks`
- `owned`
- `ghost`
- `nnz_diag`
- `nnz_offd`
- `owned_cv`
- `ghost_cv`

## Scaling Benchmark Entry (HPC Maturity)

The same benchmark file now includes an ignored scaling test for
`PCG + RAS(ILU0, overlap=1)`:

- `ras_scaling_report_pcg_ilu0_overlap1`

Run explicitly:

```bash
cargo test -p fem-parallel ras_scaling_report_pcg_ilu0_overlap1 -- --ignored --nocapture
```

Optional CSV export:

PowerShell:

```powershell
$env:RAS_SCALING_CSV = "output/ras_scaling.csv"
cargo test -p fem-parallel ras_scaling_report_pcg_ilu0_overlap1 -- --ignored --nocapture
```

Bash:

```bash
RAS_SCALING_CSV=output/ras_scaling.csv cargo test -p fem-parallel ras_scaling_report_pcg_ilu0_overlap1 -- --ignored --nocapture
```

Scaling CSV columns:

- `mode` (`strong` or `weak`)
- `ranks`
- `mesh_n`
- `dofs`
- `iterations`
- `final_residual`
- `time_ms`
- `strong_eff`
- `weak_growth`
- `owned`
- `ghost`
- `nnz_diag`
- `nnz_offd`
- `owned_cv`
- `ghost_cv`
- `score` (`pass`/`warn`/`fail`)

Scoring is computed automatically per row from:

- Mode KPI:
  - strong mode uses `strong_eff` (higher is better)
  - weak mode uses `weak_growth` (lower is better)
- Load-balance KPI:
  - `owned_cv` (lower is better)
  - `ghost_cv` (lower is better)

Thresholds can be configured via environment variables:

- `RAS_STRONG_EFF_WARN` (default `0.50`)
- `RAS_STRONG_EFF_FAIL` (default `0.30`)
- `RAS_WEAK_GROWTH_WARN` (default `3.00`)
- `RAS_WEAK_GROWTH_FAIL` (default `6.00`)
- `RAS_OWNED_CV_WARN` (default `0.20`)
- `RAS_OWNED_CV_FAIL` (default `0.35`)
- `RAS_GHOST_CV_WARN` (default `0.35`)
- `RAS_GHOST_CV_FAIL` (default `0.55`)

## Scaling Gate Automation

Local gate script:

- `scripts/check_ras_scaling_csv.ps1`

Example (score-only gate):

```powershell
./scripts/check_ras_scaling_csv.ps1 -CsvPath output/ras_scaling.csv
```

Example (warn is also fatal):

```powershell
./scripts/check_ras_scaling_csv.ps1 -CsvPath output/ras_scaling.csv -FailOnWarn
```

Example (with baseline trend compare):

```powershell
./scripts/check_ras_scaling_csv.ps1 -CsvPath output/ras_scaling.csv -BaselinePath output/ras_scaling_baseline.csv
```

Optional delta CSV output path:

```powershell
./scripts/check_ras_scaling_csv.ps1 -CsvPath output/ras_scaling.csv -BaselinePath output/ras_scaling_baseline.csv -DeltaOutPath output/ras_scaling_delta.csv
```

Example (make trend regressions fatal):

```powershell
./scripts/check_ras_scaling_csv.ps1 -CsvPath output/ras_scaling.csv -BaselinePath output/ras_scaling_baseline.csv -FailOnTrendRegression
```

CI workflow:

- `.github/workflows/hpc-scaling-gate.yml`
- Baseline CSV (default trend compare target):
  - `docs/baselines/ras_scaling_baseline.csv`

This workflow can run on schedule or manually (`workflow_dispatch`) and will:

1. Run `ras_scaling_report_pcg_ilu0_overlap1`
2. Gate on `score` (`fail` rows are fatal by default)
3. Compare against `docs/baselines/ras_scaling_baseline.csv` by default
  (or a custom baseline via workflow input)
4. Upload `output/ras_scaling.csv` as an artifact
5. Upload `output/ras_scaling_delta.csv` as an artifact
6. Publish `hpc_gate_summary` and `hpc_trend_summary` to GitHub Job Summary

`workflow_dispatch` supports:

- `fail_on_warn`
- `fail_on_trend_regression`
- `baseline_path`

## Multiphysics Template KPI Sweep

To track built-in multiphysics templates (ex48-ex52) with a single KPI CSV,
use:

- `scripts/run_template_kpi_sweep.ps1`

Local run example:

```powershell
./scripts/run_template_kpi_sweep.ps1 -OutputCsv output/template_kpi.csv -RunId local -Tag quick
```

This script runs:

- `mfem_ex48_template_joule_heating`
- `mfem_ex49_template_fsi`
- `mfem_ex50_template_acoustics_structure`
- `mfem_ex51_template_em_thermal_stress`
- `mfem_ex52_template_reaction_flow_thermal`

with reduced quick settings for CI-friendly runtime, and writes a unified CSV
through `FEM_TEMPLATE_KPI_CSV`.

Template KPI gate script:

- `scripts/check_template_kpi_csv.ps1`

Example (score + trend compare):

```powershell
./scripts/check_template_kpi_csv.ps1 -CsvPath output/template_kpi.csv -BaselinePath docs/baselines/template_kpi_baseline.csv -DeltaOutPath output/template_kpi_delta.csv
```

Example (warn/fail strict mode):

```powershell
./scripts/check_template_kpi_csv.ps1 -CsvPath output/template_kpi.csv -BaselinePath docs/baselines/template_kpi_baseline.csv -FailOnWarn -FailOnTrendRegression
```

Default baseline for template trend compare:

- `docs/baselines/template_kpi_baseline.csv`

`hpc-scaling-gate` now also executes this sweep and uploads:

- `output/template_kpi.csv`
- `output/template_kpi_delta.csv`

as `template-kpi-csv` artifact, plus a `template_kpi_summary` line in job
summary, plus `template_kpi_gate_summary` / `template_kpi_trend_summary`.

## Current Conclusion

At current mesh/problem settings, overlap=1 is generally beneficial, and
`Ilu0 + overlap=1` is the strongest tested variant for both PCG and GMRES.

## Next Engineering Steps

1. Replace overlap=1 MVP correction with explicit subdomain expansion and
   restriction/prolongation operators.
2. Add overlap-level communication diagnostics (bytes/exchange, halo edge counts).
3. Add larger-mesh benchmark points and produce scaling trend CSVs.
