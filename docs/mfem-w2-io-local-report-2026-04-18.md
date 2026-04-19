# Week-2 IO Local Execution Report

Generated at: 2026-04-18 19:31:28 +08:00
Mode: full
Backends: partitioned, mpi
Repeat target: 20

## Passed
- crate test (partitioned, pure-rust backend)
- example tests (partitioned, pure-rust backend)
- example run (partitioned, pure-rust backend)
- repeat stability (partitioned): 0 failures / 20 runs
- crate test (mpi, pure-rust backend)
- example tests (mpi, pure-rust backend)
- example run (mpi, pure-rust backend)
- repeat stability (mpi): 0 failures / 5 runs

## Failed
- none

## Warnings
- none

## Notes
- This report is intended as PM-001 local evidence when GitHub Actions is unavailable.
- For formal closure, CI artifact links should still be added once Actions is available.
- This run uses the pure-Rust checkpoint backend path (no native HDF5 dependency required).
- CI URL/artifact backfill should be recorded in `docs/mfem-w2-io-ci-backfill-template.md`.
