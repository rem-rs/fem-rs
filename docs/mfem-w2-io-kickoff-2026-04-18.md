# MFEM Week-2 IO Kickoff Log (2026-04-18)

Date: 2026-04-18
Scope: Start execution of Week-2 IO parity plan (PM-001).

## Delivered in this kickoff

1. Added dedicated CI workflow for IO parity:
   - `.github/workflows/io-parity-hdf5.yml`
2. Workflow includes two backend lanes:
   - `partitioned` (pure-Rust default backend path)
   - `mpi` (pure-Rust default backend path)
3. Workflow command set covers:
   - crate tests for `fem-io-hdf5-parallel`
   - example test `mfem_ex43_hdf5_checkpoint`
   - runnable ex43 parity command with output HDF5/XDMF artifacts
4. Workflow uploads `.h5` and `.xdmf` artifacts for evidence retention.
5. Added flaky gate loop for partitioned backend:
   - ex43 command repeated 20 times in CI and fails job if any run fails.
6. Split IO gate into smoke/full tiers:
   - smoke: pull_request default tier (fast checks)
   - full: workflow_dispatch + nightly schedule tier (includes full test set and 20-repeat gate)

## PM-001 mapping

- Parity row: `PM-001` in `docs/mfem-parity-matrix-template.md`
- This kickoff satisfies:
  - W2-1 (Add CI job for MPI + HDF5 environment)
   - W2-4 (separate smoke vs full IO test tiers)
- PM-001 closure evidence (completed):
   - smoke run: https://github.com/rem-rs/fem-rs/actions/runs/24606857993
   - full run: https://github.com/rem-rs/fem-rs/actions/runs/24606858418
   - latest PR smoke re-validation: https://github.com/rem-rs/fem-rs/actions/runs/24606887939

## Local execution evidence (Actions unavailable)

1. Local smoke runner added:
   - `scripts/run_io_parity_hdf5.ps1`
2. Local report generated:
   - `docs/mfem-w2-io-local-report-2026-04-18.md`
3. Current local blocker:
   - Removed by switching PM-001 execution path to pure-Rust checkpoint backend in `fem-io-hdf5-parallel` default build.
   - ex43 runs in local default environment without native HDF5 installation.

4. Validation snapshot after pure-Rust route switch:
   - `cargo test -p fem-io-hdf5-parallel`: pass (6 passed, 0 failed)
   - `cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint`: pass (4 passed, 0 failed)
   - `scripts/run_io_parity_hdf5.ps1 -Mode full -Backend all -Repeat 20`: pass
     - repeat stability (partitioned): 0 failures / 20 runs
     - repeat stability (mpi): 0 failures / 5 runs
   - Execution evidence synced in `docs/mfem-w2-io-local-report-2026-04-18.md`.

## CI backfill template (Actions recovery)

- Use `docs/mfem-w2-io-ci-backfill-template.md` as the single source of CI run URLs and artifact links.
- After filling template URLs, update PM-001 row status and tracker entry in the same commit.

## Next execution items (Week 2)

1. PM-001 closure path is complete (smoke/full CI evidence + matrix/tracker backfill done).
2. Keep this workflow as the reusable gate for future IO-related PRs.
3. Re-run full tier by `workflow_dispatch` when ex43 or IO schema changes.

## CI implementation note (important)

1. The repository contains private submodules (`vendor/linger`), while `IO Parity HDF5` only needs `crates/io_hdf5_parallel` + `examples/mfem_ex43_hdf5_checkpoint.rs`.
2. To avoid checkout/build failures on runners without submodule access, the workflow now:
   - uses checkout with `submodules: false`
   - creates a temporary minimal workspace at runtime for PM-001 checks.
3. This keeps PM-001 gate deterministic and independent from private submodule availability.

### Script command template (CI recovery)

Precondition:

1. `.github/workflows/io-parity-hdf5.yml` must be present on the remote default branch, otherwise `gh workflow run` returns workflow not found.

1. Preview only (no file write):
   - `./scripts/complete_pm001_after_ci.ps1 -SmokeRunUrl <SMOKE_RUN_URL> -FullRunUrl <FULL_RUN_URL> -UpdatedBy <NAME> -AutoFillFromRunUrl -Preview`
2. Apply changes:
   - `./scripts/complete_pm001_after_ci.ps1 -SmokeRunUrl <SMOKE_RUN_URL> -FullRunUrl <FULL_RUN_URL> -UpdatedBy <NAME> -AutoFillFromRunUrl`
3. Optional explicit commit override:
   - `./scripts/complete_pm001_after_ci.ps1 -SmokeRunUrl <SMOKE_RUN_URL> -FullRunUrl <FULL_RUN_URL> -CommitSha <COMMIT_SHA> -UpdatedBy <NAME> -AutoFillFromRunUrl`
4. Placeholder dry-run only:
   - `./scripts/complete_pm001_after_ci.ps1 -SmokeRunUrl https://example/smoke -FullRunUrl https://example/full -UpdatedBy <NAME> -AutoFillFromRunUrl -AllowPlaceholderUrls -Preview`
5. Template-only apply (no status flip):
   - `./scripts/complete_pm001_after_ci.ps1 -SmokeRunUrl <SMOKE_RUN_URL> -FullRunUrl <FULL_RUN_URL> -UpdatedBy <NAME> -AutoFillFromRunUrl -NoStatusFlip`
6. One-shot CI dispatch + watch + backfill:
   - `./scripts/run_pm001_ci_and_backfill.ps1 -UpdatedBy <NAME>`
   - Optional staged mode: `./scripts/run_pm001_ci_and_backfill.ps1 -UpdatedBy <NAME> -NoStatusFlip`
