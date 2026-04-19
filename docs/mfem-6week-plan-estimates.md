# MFEM Alignment 6-Week Plan with Person-Day Estimates

Date: 2026-04-18
Scope: Convert current "feature mostly present" status into measurable, repeatable parity delivery.

## Assumptions

- 1 person-day (PD) means one engineer focused for one working day.
- Team baseline: 2 to 3 engineers can work in parallel.
- Estimates include implementation, tests, and doc updates.
- Risk buffer is included per week as explicit PD.

## Named Team and Calendar Window

- Sprint window: 2026-04-20 to 2026-05-29 (6 weeks)
- Named owners used in this plan:
	- Lin: IO and data format owner
	- Wang: solver and benchmark owner
	- Chen: assembly and transfer owner
	- Zhao: QA and parity validation owner
	- Xu: CI and infrastructure owner

## Staffing Roster (Owner + Start Date + End Date + Task)

| ID | Owner | Start | End | Task | Estimate (PD) |
|---|---|---|---|---|---:|
| W1-1 | Zhao | 2026-04-20 | 2026-04-20 | Build parity KPI taxonomy (correctness/robustness/scale/parallel/IO) | 1.0 |
| W1-2 | Chen + Zhao | 2026-04-20 | 2026-04-21 | Freeze baseline commands and metric snapshots | 1.5 |
| W1-3 | Lin | 2026-04-21 | 2026-04-21 | Convert tracker statuses to measurable done criteria | 1.0 |
| W1-4 | Zhao | 2026-04-22 | 2026-04-22 | Add PR gating checklist and review rubric | 0.5 |
| W1-B | Shared | 2026-04-23 | 2026-04-23 | Scope clarification and unblock buffer | 1.0 |
| W2-1 | Xu | 2026-04-27 | 2026-04-28 | Add MPI + HDF5 CI job and environment checks | 1.5 |
| W2-2 | Lin | 2026-04-27 | 2026-04-28 | Harden restart consistency checks (step/time/offset/len/value) | 1.5 |
| W2-3 | Lin | 2026-04-29 | 2026-04-29 | Add XDMF dataset path and temporal sequence assertions | 1.0 |
| W2-4 | Xu + Zhao | 2026-04-30 | 2026-04-30 | Split smoke/full IO test tiers in CI | 1.0 |
| W2-B | Shared | 2026-05-01 | 2026-05-01 | Flaky stabilization and env contingency | 1.5 |
| W3-1 | Wang | 2026-05-04 | 2026-05-04 | Build AMS/ADS/AIR benchmark harness and schema | 1.5 |
| W3-2 | Wang | 2026-05-05 | 2026-05-05 | Add anisotropic and high-contrast stress cases | 1.5 |
| W3-3 | Wang | 2026-05-06 | 2026-05-06 | Add nonsymmetric convection-diffusion stress cases | 1.0 |
| W3-4 | Wang + Zhao | 2026-05-07 | 2026-05-07 | Capture convergence/iteration/failure-rate baselines | 1.0 |
| W3-B | Shared | 2026-05-08 | 2026-05-08 | Numerical tuning and retry policy buffer | 1.5 |
| W4-1 | Lin | 2026-05-11 | 2026-05-11 | Define canonicalized round-trip comparison strategy | 1.0 |
| W4-2 | Lin | 2026-05-12 | 2026-05-12 | Implement Netgen round-trip tests (uniform + mixed) | 1.5 |
| W4-3 | Lin | 2026-05-13 | 2026-05-13 | Implement Abaqus round-trip tests (C3D4/C3D8 + mixed) | 1.5 |
| W4-4 | Lin + Zhao | 2026-05-14 | 2026-05-14 | Add section/material/boundary tag preservation checks | 1.0 |
| W4-B | Shared | 2026-05-15 | 2026-05-15 | Format edge-case parser robustness buffer | 1.0 |
| W5-1 | Chen | 2026-05-18 | 2026-05-18 | Lock 3 long-running anchors and acceptance form | 0.5 |
| W5-2 | Xu + Zhao | 2026-05-19 | 2026-05-19 | Add quick/nightly split for anchor suite | 1.5 |
| W5-3 | Zhao | 2026-05-20 | 2026-05-20 | Add residual/error/runtime trend capture | 1.5 |
| W5-4 | Shared | 2026-05-21 | 2026-05-21 | Stabilize known flaky paths and retry policy | 1.0 |
| W5-B | Shared | 2026-05-22 | 2026-05-22 | CI runtime optimization and infra buffer | 1.0 |
| W6-1 | Chen | 2026-05-25 | 2026-05-26 | Produce parity closeout report (match/risk/gaps) | 1.5 |
| W6-2 | Lin + Wang | 2026-05-27 | 2026-05-27 | Backfill tracker evidence links and status updates | 1.0 |
| W6-3 | Zhao + Chen | 2026-05-28 | 2026-05-28 | Final matrix-threshold acceptance review | 1.0 |
| W6-4 | Chen | 2026-05-29 | 2026-05-29 | Define next-cycle backlog and owners | 0.5 |
| W6-B | Shared | 2026-05-29 | 2026-05-29 | Final blocker closure buffer | 1.0 |

## Delivery Goals

1. HDF5/XDMF checkpoint-restart parity reaches stable CI acceptance.
2. Native AMG route (AMS/ADS/AIR) moves from baseline to hardened parity behavior.
3. Netgen/Abaqus import-export reaches high-fidelity round-trip validation.
4. A unified parity acceptance matrix becomes the release gate.

## Week-by-Week Work Breakdown

### Week 1: Acceptance Matrix and Baseline Freeze

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W1-1 | Build parity KPI taxonomy (correctness, robustness, scale, parallel consistency, IO consistency) | Tech lead | 1.0 | KPI definitions |
| W1-2 | Freeze current baseline commands and metric snapshots | QA + module owners | 1.5 | baseline report |
| W1-3 | Convert tracker "partial/completed" to measurable done criteria | Tech lead | 1.0 | updated criteria table |
| W1-4 | Add quick checklist for PR gating | QA | 0.5 | checklist section |
| W1-B | Buffer for scope clarification | Shared | 1.0 | risk absorption |

Week 1 subtotal: 5.0 PD

### Week 2: Parallel IO End-to-End Gate (MPI + HDF5)

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W2-1 | Add CI job for MPI + HDF5 environment | Infra engineer | 1.5 | workflow job |
| W2-2 | Harden restart consistency tests (step/time/offset/len/value checks) | IO owner | 1.5 | stronger tests |
| W2-3 | Add XDMF dataset path and temporal sequence checks | IO owner | 1.0 | xdmf assertions |
| W2-4 | Separate smoke vs full IO test tiers | Infra + QA | 1.0 | tiered test plan |
| W2-B | Environment and flaky-test buffer | Shared | 1.5 | stability fixes |

Week 2 subtotal: 6.5 PD

### Week 3: Native AMG Hardening (AMS/ADS/AIR)

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W3-1 | Build unified benchmark harness and output schema | Solver owner | 1.5 | benchmark harness |
| W3-2 | Add anisotropic + high-contrast stress cases | Solver owner | 1.5 | test scenarios |
| W3-3 | Add nonsymmetric convection-diffusion stress cases | Solver owner | 1.0 | test scenarios |
| W3-4 | Capture convergence/iteration/failure-rate baselines | QA + solver owner | 1.0 | benchmark report |
| W3-B | Numerical tuning buffer | Shared | 1.5 | parameter stabilization |

Week 3 subtotal: 6.5 PD

### Week 4: Netgen/Abaqus Round-Trip Fidelity

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W4-1 | Define canonicalized comparison strategy (renumbering-safe) | Mesh/IO owner | 1.0 | compare utility spec |
| W4-2 | Implement round-trip tests for Netgen (uniform + mixed) | Mesh/IO owner | 1.5 | netgen tests |
| W4-3 | Implement round-trip tests for Abaqus (C3D4/C3D8 + mixed) | Mesh/IO owner | 1.5 | abaqus tests |
| W4-4 | Add section/material/boundary tag preservation checks | QA + IO owner | 1.0 | preservation tests |
| W4-B | Format edge-case buffer | Shared | 1.0 | parser robustness fixes |

Week 4 subtotal: 6.0 PD

### Week 5: Long-Running Anchor Regressions

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W5-1 | Select and formalize 3 anchor scenarios (Maxwell, transfer 3D, multiphysics templates) | Tech lead | 0.5 | anchor list |
| W5-2 | Add quick and nightly split for anchor suite | Infra + QA | 1.5 | CI split |
| W5-3 | Add trend capture for key metrics (residual/error/runtime) | QA | 1.5 | trend artifacts |
| W5-4 | Stabilize known flaky paths and retry policy | Shared | 1.0 | stable runs |
| W5-B | CI runtime and infra buffer | Shared | 1.0 | runtime optimization |

Week 5 subtotal: 5.5 PD

### Week 6: Closeout and Release-Grade Reporting

| ID | Task | Owner Suggestion | Estimate (PD) | Output |
|---|---|---|---:|---|
| W6-1 | Produce parity report (what matched, risks, remaining gaps) | Tech lead | 1.5 | parity report |
| W6-2 | Backfill tracker updates and evidence links | Module owners | 1.0 | updated tracker |
| W6-3 | Final acceptance review against matrix thresholds | QA + lead | 1.0 | acceptance signoff |
| W6-4 | Define next-cycle backlog with priority and ownership | Tech lead | 0.5 | next-cycle backlog |
| W6-B | Final buffer for blockers | Shared | 1.0 | issue closure |

Week 6 subtotal: 5.0 PD

## Total Effort

- Planned engineering effort: 34.5 PD
- Suggested contingency if team is new to this workflow: +15% (5.2 PD)
- Recommended total budget: 39.7 PD (round to 40 PD)

## Suggested Staffing Models

- 2 engineers full-time for 6 weeks: ~60 PD capacity, comfortable margin.
- 3 engineers part-time (~60% allocation): ~54 PD capacity, still healthy.
- 1 engineer solo: high risk for CI/infra coupling and review latency.

## Minimal Acceptance Criteria at End of Week 6

1. MPI+HDF5 end-to-end restart CI gate is green and non-flaky.
2. AMS/ADS/AIR have benchmarked parity baselines with tracked trends.
3. Netgen/Abaqus round-trip fidelity checks pass for core element/tag coverage.
4. Every parity item in active scope maps to a measurable criterion and test command.
5. Tracker status reflects code and test evidence only.

## Command Groups to Keep as Baseline Anchors

```bash
cargo test -p fem-assembly transfer::tests::conservative_projection_3d_matches_global_integral
cargo test -p fem-examples --example mfem_ex48_template_joule_heating --example mfem_ex49_template_fsi --example mfem_ex50_template_acoustics_structure --example mfem_ex51_template_em_thermal_stress --example mfem_ex52_template_reaction_flow_thermal
```

Optional IO/parallel anchors (environment-dependent):

```bash
cargo test -p fem-examples --example mfem_ex43_hdf5_checkpoint
cargo test -p fem-io-hdf5-parallel
```
