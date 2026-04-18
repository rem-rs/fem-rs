# MFEM Parity Matrix Template

Date: 2026-04-18
Purpose: Use this matrix as a measurable parity gate instead of narrative status only.

## How to Use

1. One row per parity item (feature, workflow, or behavior).
2. Fill all threshold columns with numeric or binary criteria.
3. Link each row to concrete test commands and evidence artifacts.
4. Status can be marked complete only when all thresholds are met.

## Status Legend

- complete: fully meets thresholds and evidence exists
- partial: implemented but thresholds/evidence incomplete
- planned: not implemented yet
- out-of-scope: intentionally excluded from parity scope

## Parity Matrix

| Parity ID | MFEM Capability | fem-rs Module/Path | Priority | Status | Correctness Threshold | Robustness Threshold | Scale Threshold | Parallel Consistency Threshold | IO/Restart Threshold | Tests/Commands | Evidence Link | Owner | Target Date | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PM-001 | HDF5 checkpoint-restart end-to-end | examples/mfem_ex43_hdf5_checkpoint.rs | P0 | complete | restart field error <= 1e-12 | 0 flaky runs in 20 repeats | >= 2 rank partitions | rank-local/global values consistent | temporal XDMF path consistency | cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint | docs/mfem-w2-io-kickoff-2026-04-18.md + docs/mfem-w2-io-local-report-2026-04-18.md + docs/mfem-w2-io-ci-backfill-template.md + scripts/run_io_parity_hdf5.ps1 + .github/workflows/io-parity-hdf5.yml | Lin (IO) | 2026-04-29 | CI smoke/full evidence backfilled; closure criteria satisfied on pure-Rust default backend path. |
| PM-002 | Nonmatching conservative transfer 3D | crates/assembly/src/transfer.rs | P0 | complete | integral error after <= 1e-11 | no fallback panic in stress cases | tetra mesh level >= 4 | partition-invariant scalar checks | n/a | cargo test -p fem-assembly transfer::tests::conservative_projection_3d_matches_global_integral | docs/mfem-baseline-snapshot-2026-04-18.md (section: Conservative transfer 3D anchor, PASS) | Chen (Assembly) | 2026-04-22 | Keep as release anchor; rerun on solver/transfer touching PRs. |
| PM-003 | Multiphysics template adaptive sync | examples/mfem_ex48..52 | P1 | partial | all target tests pass | strict-fail branches covered | multi-step >= 4 stable | deterministic under same seed/config | n/a | cargo test -p fem-examples --example mfem_ex48_template_joule_heating --example mfem_ex49_template_fsi --example mfem_ex50_template_acoustics_structure --example mfem_ex51_template_em_thermal_stress --example mfem_ex52_template_reaction_flow_thermal | docs/mfem-baseline-snapshot-2026-04-18.md (section: ex48-ex52 anchor set, PASS) | Chen (Multiphysics) | 2026-05-13 | Add CI artifact links for strict-fail branch coverage before closing. |
| PM-004 | AMS/ADS/AIR parity hardening | crates/amg + crates/solver | P0 | partial | residual target met for benchmark set | fail rate < 1% on stress suite | DOF tiers S/M/L all pass | rank 2/4 parity in iteration trend | n/a | cargo bench -p fem-benches amg -- --save-baseline ams_ads_air_w3 | docs/mfem-baseline-snapshot-2026-04-18.md (baseline foundation; benchmark artifact link to be added after W3 harness run) | Wang (Solver) | 2026-05-06 | W3 target: publish baseline JSON/CSV artifact and fill CI link. |
| PM-005 | Netgen/Abaqus round-trip fidelity | crates/io | P1 | partial | topology/tag equality 100% | parser robust on mixed inputs | medium mesh round-trip <= 2x baseline time | deterministic output metadata | read-write-read invariants pass | TODO io round-trip commands | TODO | TODO | TODO | TODO |

## KPI Definitions (Recommended)

### Correctness

- Numerical error thresholds (L2/H1/integral/constraint residual).
- For exact-preservation cases, use near-machine tolerance checks.

### Robustness

- Flaky rate under repeated runs.
- Behavior under stress inputs (anisotropy, high contrast, nonsymmetry).

### Scale

- Define DOF tiers:
  - S: quick PR-scale
  - M: daily CI-scale
  - L: nightly or pre-release scale

### Parallel Consistency

- Same problem under different partitioning should preserve accepted invariants.
- Prefer invariant metrics (norms/sums/errors) over index-sensitive checksums.

### IO/Restart Consistency

- Restart from step k should match uninterrupted run from same state.
- Metadata consistency: time stamps, dataset paths, field lengths/offsets.

## Review Checklist for Each Row

- Thresholds are explicit and non-ambiguous.
- Command reproduces the evidence.
- Evidence link points to CI artifact or checked-in report.
- Owner and target date are set.
- Status matches measurable reality.

## Optional CSV Export Template

```csv
parity_id,mfem_capability,module_path,priority,status,correctness_threshold,robustness_threshold,scale_threshold,parallel_consistency_threshold,io_restart_threshold,tests_commands,evidence_link,owner,target_date,notes
PM-001,HDF5 checkpoint-restart end-to-end,examples/mfem_ex43_hdf5_checkpoint.rs,P0,partial,restart field error <= 1e-12,0 flaky runs in 20 repeats,>= 2 rank partitions,rank-local/global values consistent,temporal XDMF path consistency,"cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint",TODO,TODO,TODO,TODO
```

