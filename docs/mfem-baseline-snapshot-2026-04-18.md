# MFEM Parity Baseline Snapshot (2026-04-18)

Captured at: 2026-04-18 19:02:20 +08:00
Workspace: C:/Users/lilu/works/fem-rs
Purpose: Initial measurable baseline tied to the parity matrix and 6-week plan.

## Command Results

### 1) Conservative transfer 3D anchor

Command:

```bash
cargo test -p fem-assembly transfer::tests::conservative_projection_3d_matches_global_integral
```

Result:

- Status: PASS
- Tests: 1 passed, 0 failed
- Duration: ~0.02s test body (build/test process completed normally)

### 2) Multiphysics template adaptive-sync anchor set (ex48-ex52)

Command:

```bash
cargo test -p fem-examples --example mfem_ex48_template_joule_heating --example mfem_ex49_template_fsi --example mfem_ex50_template_acoustics_structure --example mfem_ex51_template_em_thermal_stress --example mfem_ex52_template_reaction_flow_thermal
```

Result:

- Status: PASS
- ex48: 3 passed, 0 failed
- ex49: 2 passed, 0 failed
- ex50: 3 passed, 0 failed
- ex51: 2 passed, 0 failed
- ex52: 4 passed, 0 failed
- Combined: 14 passed, 0 failed

## Observed Warnings (Non-blocking)

- Existing workspace warnings remain in unrelated modules (unused imports/variables, mixed-script confusable warning).
- No failures introduced for the above parity anchors.

## Intended Use

- Use this snapshot as the baseline evidence for:
  - docs/mfem-parity-matrix-template.md
  - docs/mfem-6week-plan-estimates.md
- Compare future runs against this baseline when changing solver/assembly/multiphysics sync logic.
