# Post-Parity Expansion Roadmap (2026-05-15)

Purpose: define the execution model after core MFEM parity closure.

This document separates two tracks:

1. Parity maintenance
2. Beyond-MFEM expansion

The goal is to preserve a stable, defensible MFEM correspondence story while
continuing to grow user-facing capabilities on top of the now-closed core.

## 1. Operating Principle

MFEM correspondence in this repository is closed primarily by:

- feature coverage of the core FEM stack
- example and workflow parity
- compatible public APIs

It is not defined by delivering external solver FFI integrations. Solver names
such as `hypre`, `mumps`, and `mkl` are treated as compatibility contracts and
API correspondences. Native `linger` implementations are the delivery path.

## 2. Two-Track Model

### 2.1 Parity Maintenance Track

Scope:

- bug fixes in parity-covered functionality
- regression hardening
- performance hardening within existing parity features
- CI evidence backfill and acceptance-threshold maintenance
- doc corrections to preserve consistent parity claims

Out of scope:

- new feature families not required for MFEM correspondence
- external solver FFI delivery as a parity goal
- reclassifying product/workflow expansion as parity gaps

Closure rule:

- parity rows close only with code plus executable evidence
- parity rows do not reopen for engineering stretch goals once the agreed
  baseline is met

### 2.2 Beyond-MFEM Expansion Track

Scope:

- imported-mesh user workflows
- named-attribute driven configuration and solve/export pipelines
- multiphysics templates and coupled workflows
- large-scale engineering hardening beyond parity baseline
- workflow productization, demos, and domain-specific accelerators

Success metric:

- end-to-end user closure: import or build -> solve -> export or checkpoint ->
  regression coverage

## 3. Priority Order

### E1. Workflow Expansion

Focus:

- imported mesh configuration workflows
- named set or named attribute driven BC and material configuration
- result export bundles for VTK/HDF5/XDMF
- restart-capable user workflows

Why first:

- highest user leverage
- reuses already-closed mesh/IO/solver infrastructure
- lowest architectural risk

### E2. Multiphysics Productization

Focus:

- stabilize ex44-ex53 style template drivers
- convert template examples into configurable workflow entry points
- standardize sync metrics, exports, restart support, and smoke coverage

Why second:

- major user value on top of a mature core
- benefits directly from existing transfer, solver, and checkpoint primitives

### E3. Scale and Engineering Hardening

Focus:

- AMG robustness at larger scales
- distributed checkpoint throughput and stability
- broader CI feature matrix
- repeatability and long-run stability

Why third:

- important for deployment quality
- should not be allowed to blur parity closure

### E4. Higher-Fidelity Format Work

Focus:

- richer round-trip preservation for Abaqus/Netgen and imported-mesh metadata
- additional section or tag fidelity not required by current parity baseline

Why fourth:

- useful engineering work
- not required to maintain the current parity claim

## 4. Immediate Execution Policy

For any new feature proposal:

1. Decide whether it belongs to parity maintenance or Beyond-MFEM.
2. Name the user-facing workflow it enables.
3. Reuse an existing closed core path when possible.
4. Add at least one executable regression tied to the workflow.
5. Keep parity docs unchanged unless the feature changes a parity claim.

## 5. First Expansion Batch

### X1. Imported Mesh Workflow Consolidation

Goal:

- unify the current Gmsh, Abaqus, and Netgen user closures into a documented,
  example-backed workflow family

Candidate anchors:

- `examples/mfem_ex39_named_attributes.rs`
- `crates/io/tests/io_integration.rs`

Expected outputs:

- one documented configuration story for named sets, boundary masks, and
  material tags
- one stable export story spanning VTK and HDF5/XDMF
- one smoke suite grouping imported-mesh user closures

Initial progress (2026-05-15):

- `mfem_ex39_named_attributes` now exposes a unified imported-workflow entry
  surface via `--import-format {gmsh|abaqus|netgen}` plus `--input <path>`.
- Legacy `--abaqus` and `--netgen` entry points remain supported for
  compatibility.
- Parser coverage for the unified entry surface was added to the example test
  suite.
- The imported-workflow spec now documents the three-layer closure chain across
  example-level VTK export, imported-mesh HDF5/XDMF regressions, and
  checkpoint/restart anchors.

### X2. Multiphysics Template Hardening

Goal:

- standardize product-quality workflow behavior across ex44-ex53 style drivers

Candidate anchors:

- `examples/mfem_ex44_thermoelastic_coupled.rs`
- `examples/mfem_ex48_template_joule_heating.rs`
- `examples/mfem_ex49_template_fsi.rs`
- `examples/mfem_ex50_template_acoustics_structure.rs`
- `examples/mfem_ex51_template_em_thermal_stress.rs`
- `examples/mfem_ex52_template_reaction_flow_thermal.rs`

Expected outputs:

- common workflow conventions for restart, exported fields, and sweepable CLI
  knobs
- a tighter smoke acceptance set for template families

### X3. Expansion Governance

Goal:

- prevent future drift where extension work is misfiled as parity gaps

Expected outputs:

- parity docs remain stable unless a parity claim changes
- expansion items are tracked under Beyond-MFEM labels or roadmap entries

## 6. Immediate Next Actions

1. Keep `MFEM_MAPPING.md` and `MFEM_ALIGNMENT_TRACKER.md` as parity authorities.
2. Use this roadmap as the authority for post-parity prioritization.
3. Start new feature planning under E1 unless there is a clear reason to jump to
   E2 or E3.
4. Require every new expansion item to name its closed-core dependency chain.

## 7. Initial Candidate Task Queue

| ID | Track | Priority | Title | Anchor |
|---|---|---|---|---|
| EXP-001 | E1 | P0 | Imported mesh workflow consolidation | `examples/mfem_ex39_named_attributes.rs` |
| EXP-002 | E1 | P0 | Unified export/checkpoint user closure | `crates/io/tests/io_integration.rs` |
| EXP-003 | E2 | P1 | Template workflow conventions | `examples/mfem_ex44_thermoelastic_coupled.rs` |
| EXP-004 | E2 | P1 | Template restart/export smoke matrix | `examples/` + `.github/workflows/` |
| EXP-005 | E3 | P1 | Distributed checkpoint hardening | `crates/io_hdf5_parallel/` + `crates/parallel/` |

## 8. Authority and Update Rule

- Update this file when expansion priorities change.
- Update parity docs only when parity claims change.
- Do not move an item from Beyond-MFEM into parity tracking unless it is truly
  required to defend MFEM correspondence.

Progress note (2026-05-15):

- EXP-001 first milestone landed: unified imported-workflow entry surface in
  `mfem_ex39_named_attributes`.
- EXP-002 first milestone landed in documentation: the imported-mesh closure is
  now described as a coherent VTK -> HDF5/XDMF -> checkpoint progression.
- EXP-002 first code milestone landed: imported-workflow field names now share
  a common source in `fem-io` (`crates/io/src/imported_workflow.rs`), and both
  `mfem_ex39_named_attributes` and `crates/io/tests/io_integration.rs` consume
  that contract for core exported field identifiers.
- EXP-002 second code milestone landed: `fem-io` now also provides shared
  imported-workflow helper constructors for VTK and XDMF field objects, and the
  first imported workflow anchors consume those helpers instead of assembling
  field objects ad hoc.
- EXP-002 third code milestone landed: VTK field-list contracts are now shared
  in `fem-io` for imported boundary-mask, named-boundary, named-attribute
  solution, and Abaqus solution exports (`vtk_imported_mask_fields`,
  `vtk_named_boundary_fields`, `vtk_named_attribute_solution_fields`,
  `vtk_abaqus_solution_fields`, `vtk_nodal_workflow_fields`), and the imported
  workflow anchors in `mfem_ex39_named_attributes` plus the Tri6 `fem-io`
  integration workflow consume those helpers instead of assembling the
  corresponding VTK field lists ad hoc.
- EXP-002 fourth code milestone landed: `fem-io` now also provides a semantic
  imported-mask XDMF helper (`xdmf_imported_mask_workflow_fields`), and the
  Abaqus/Netgen imported workflow integration tests consume the shared
  imported-mask helpers on both the VTK and XDMF paths, making those contracts
  explicit regression anchors in `io_integration`.
- EXP-002 fifth code milestone landed: `fem-io` now also provides a semantic
  Abaqus solution-style XDMF helper (`xdmf_abaqus_solution_fields`), and the
  Abaqus imported workflow integration test validates the full shared
  `solution + fixed_mask + drive_mask + material_id` contract on both the VTK
  and XDMF paths instead of treating that workflow as a mask-only export.
- EXP-002 sixth code milestone landed: `fem-io` now also provides a semantic
  named-attribute solution XDMF helper (`xdmf_named_attribute_solution_fields`),
  and `io_integration` validates the full shared named-attribute solution
  contract across HDF5, VTK, and XDMF for `u + inlet/outlet/merged masks +
  material_id + kappa + source_strength + fluid_id + liner_id`.
- EXP-002 seventh code milestone landed: `fem-io` now also provides a semantic
  named-boundary XDMF helper (`xdmf_named_boundary_fields`), and
  `io_integration` validates the shared named-boundary contract across HDF5,
  VTK, and XDMF for `inlet_mask + outlet_mask + merged_boundary_mask +
  fluid_id`; the higher-level named-attribute solution helper now composes that
  lower-level boundary contract instead of duplicating it.
- EXP-002 eighth code milestone landed: the imported-workflow helpers were
  further decomposed into reusable point/cell sub-contracts for named-boundary
  and named-attribute exports (`vtk_named_boundary_point_fields`,
  `vtk_named_boundary_cell_fields`, `vtk_named_attribute_cell_fields`,
  `xdmf_named_boundary_point_fields`, `xdmf_named_boundary_cell_fields`,
  `xdmf_named_attribute_cell_fields`), so the higher-level solution helpers now
  act as thin composition layers rather than rebuilding those field groups.
- EXP-002 ninth code milestone landed: the Abaqus imported-workflow helpers are
  now decomposed into reusable point/cell sub-contracts as well
  (`vtk_abaqus_solution_point_fields`, `vtk_abaqus_solution_cell_fields`,
  `xdmf_abaqus_solution_point_fields`, `xdmf_abaqus_solution_cell_fields`), so
  the Abaqus solution helpers match the same composition pattern already used by
  named-boundary and named-attribute workflows.
