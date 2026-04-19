# PM-001 CI Artifact Backfill Template (Week-2 IO)

Date: 2026-04-18
Purpose: Fill this template once GitHub Actions is available, then link this file in PM-001 closure evidence.

## Run Metadata

- Workflow: `.github/workflows/io-parity-hdf5.yml`
- Workflow run URL (smoke): https://github.com/rem-rs/fem-rs/actions/runs/24606857993
- Workflow run URL (full): https://github.com/rem-rs/fem-rs/actions/runs/24606858418
- Commit SHA: 0871211d36af0e9d8c84a5ab9d2f9735372c5432
- Trigger type: pull_request (smoke), workflow_dispatch (full)
- Updated by: Lin
- Updated at: 2026-04-18 22:34:28 +08:00

## Smoke Tier Evidence

- Partitioned job URL: https://github.com/rem-rs/fem-rs/actions/runs/24606857993
- MPI job URL: https://github.com/rem-rs/fem-rs/actions/runs/24606857993
- Artifact URL (partitioned): https://github.com/rem-rs/fem-rs/actions/runs/24606857993#artifacts (io-parity-smoke-partitioned)
- Artifact URL (mpi): https://github.com/rem-rs/fem-rs/actions/runs/24606857993#artifacts (io-parity-smoke-mpi)
- Result summary: PASS (partitioned + mpi smoke jobs succeeded and artifacts uploaded)

## Full Tier Evidence

- Partitioned job URL: https://github.com/rem-rs/fem-rs/actions/runs/24606858418
- MPI job URL: https://github.com/rem-rs/fem-rs/actions/runs/24606858418
- Artifact URL (partitioned): https://github.com/rem-rs/fem-rs/actions/runs/24606858418#artifacts (io-parity-full-partitioned)
- Artifact URL (mpi): https://github.com/rem-rs/fem-rs/actions/runs/24606858418#artifacts (io-parity-full-mpi)
- Partitioned repeat gate summary: PASS (`0 failures / 20 runs` in full run gate)
- MPI repeat gate summary: PASS (full mpi lane succeeded; local baseline `0 failures / 5 runs`)
- Result summary: PASS (full tier succeeded for partitioned + mpi; artifacts uploaded)

## PM-001 Closure Checklist

- [x] `cargo test -p fem-io-hdf5-parallel` green in CI run
- [x] `cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint` green in CI run
- [x] Output artifacts (`.h5`, `.xdmf`) uploaded for both backends
- [x] Repeat gate evidence recorded (partitioned 20-run stability)
- [x] PM-001 row in `docs/mfem-parity-matrix-template.md` switched to `complete`
- [x] Tracker entry in `MFEM_ALIGNMENT_TRACKER.md` updated with concrete URLs

## Notes

- Local full evidence is already available in `docs/mfem-w2-io-local-report-2026-04-18.md`.
- This file is the canonical place to backfill CI URLs when Actions become available again.
- Helper script: `scripts/complete_pm001_after_ci.ps1` can fill this template and update PM-001 status entries automatically.
- Helper options: use `-AutoFillFromRunUrl` to auto-fill job/artifact fields from run URLs, and `-AllowPlaceholderUrls` only with `-Preview` for rehearsal.
- Optional staged workflow: add `-NoStatusFlip` to update only this template first, then flip matrix/tracker status in a later reviewed commit.

