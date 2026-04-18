# PM-001 CI Artifact Backfill Template (Week-2 IO)

Date: 2026-04-18
Purpose: Fill this template once GitHub Actions is available, then link this file in PM-001 closure evidence.

## Run Metadata

- Workflow: `.github/workflows/io-parity-hdf5.yml`
- Workflow run URL (smoke): TODO
- Workflow run URL (full): TODO
- Commit SHA: TODO
- Trigger type: TODO (pull_request/workflow_dispatch/schedule)
- Updated by: TODO
- Updated at: TODO

## Smoke Tier Evidence

- Partitioned job URL: TODO
- MPI job URL: TODO
- Artifact URL (partitioned): TODO (`io-parity-smoke-partitioned`)
- Artifact URL (mpi): TODO (`io-parity-smoke-mpi`)
- Result summary: TODO

## Full Tier Evidence

- Partitioned job URL: TODO
- MPI job URL: TODO
- Artifact URL (partitioned): TODO (`io-parity-full-partitioned`)
- Artifact URL (mpi): TODO (`io-parity-full-mpi`)
- Partitioned repeat gate summary: TODO (`0 failures / 20 runs` expected)
- MPI repeat gate summary: TODO (`0 failures / 5 runs` local baseline)
- Result summary: TODO

## PM-001 Closure Checklist

- [ ] `cargo test -p fem-io-hdf5-parallel` green in CI run
- [ ] `cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint` green in CI run
- [ ] Output artifacts (`.h5`, `.xdmf`) uploaded for both backends
- [ ] Repeat gate evidence recorded (partitioned 20-run stability)
- [ ] PM-001 row in `docs/mfem-parity-matrix-template.md` switched to `complete`
- [ ] Tracker entry in `MFEM_ALIGNMENT_TRACKER.md` updated with concrete URLs

## Notes

- Local full evidence is already available in `docs/mfem-w2-io-local-report-2026-04-18.md`.
- This file is the canonical place to backfill CI URLs when Actions become available again.
- Helper script: `scripts/complete_pm001_after_ci.ps1` can fill this template and update PM-001 status entries automatically.
- Helper options: use `-AutoFillFromRunUrl` to auto-fill job/artifact fields from run URLs, and `-AllowPlaceholderUrls` only with `-Preview` for rehearsal.
- Optional staged workflow: add `-NoStatusFlip` to update only this template first, then flip matrix/tracker status in a later reviewed commit.
