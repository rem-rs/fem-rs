# ex39 Imported Named-Config Workflows

This note records the user-facing file-based workflows added to `mfem_ex39_named_attributes`.

## Unified imported-workflow entry surface

The example now supports a consolidated imported-workflow CLI surface:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --import-format gmsh --input examples/meshes/your_mesh.msh --vtk output/ex39_gmsh_imported.vtu
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --import-format abaqus --input examples/meshes/named_sets_tet.inp --fixed-set FIXED --drive-set DRIVE --drive-value 1.0 --vtk output/ex39_abaqus_imported.vtu
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --import-format netgen --input examples/meshes/surface_tags_tet.vol --boundary-tag 5 --vtk output/ex39_netgen_imported.vtu
```

Notes:
- `--import-format` selects the imported workflow family: `gmsh`, `abaqus`, or `netgen`.
- `--input` provides the file path for the selected format.
- Legacy format-specific flags (`--abaqus`, `--netgen`) remain supported for compatibility.

## Gmsh named-set solve with material contrast

Run a two-material solve on the built-in Gmsh fixture and write VTK output:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --solve-poisson --merge-boundary --vtk output/ex39_named_solve.vtu
```

Expected VTK arrays:
- `u`
- `inlet_mask`
- `outlet_mask`
- `merged_boundary_mask`
- `material_id`
- `kappa`
- `source_strength`
- `fluid_id`
- `liner_id`

Behavior notes:
- `fluid` and `liner` are different named regions with distinct `kappa` values.
- The solve path also injects a region-driven source on `fluid`, assembled through the extracted named-region submesh and scattered back to the parent mesh.

## Abaqus file workflow

Run the imported named-set workflow against a real `.inp` file in the examples tree:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --abaqus examples/meshes/named_sets_tet.inp --fixed-set FIXED --drive-set DRIVE --drive-value 1.0 --vtk output/ex39_abaqus_imported.vtu
```

Equivalent unified form:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --import-format abaqus --input examples/meshes/named_sets_tet.inp --fixed-set FIXED --drive-set DRIVE --drive-value 1.0 --vtk output/ex39_abaqus_imported.vtu
```

Expected VTK arrays:
- `u`
- `fixed_mask`
- `drive_mask`
- `material_id`

## Netgen file workflow

Run the imported boundary-tag workflow against a real `.vol` file in the examples tree:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --netgen examples/meshes/surface_tags_tet.vol --boundary-tag 5 --vtk output/ex39_netgen_imported.vtu
```

Equivalent unified form:

```powershell
cargo run -p fem-examples --example mfem_ex39_named_attributes -- --import-format netgen --input examples/meshes/surface_tags_tet.vol --boundary-tag 5 --vtk output/ex39_netgen_imported.vtu
```

Expected VTK arrays:
- `boundary_mask`
- `material_id`

## Notes

- Run commands from the repository root.
- The example uses `GMRES` for the solve path because Dirichlet row-zeroing with nonzero values is not guaranteed to preserve an SPD system.
- The solve workflow is now both material-driven and source-driven: named regions affect `kappa` and the domain source term.
- The Abaqus and Netgen file paths above are intended as minimal, inspectable templates for user-provided imported meshes.
- Imported workflows now share a single selection surface, which is the first consolidation step for `EXP-001`.

## Unified export and checkpoint closure

The imported-mesh story is now organized as one closure chain, even though the
current repository still uses different anchors for different export layers.

### Layer 1: Example-level VTK closure

Use `mfem_ex39_named_attributes` when the user wants:

- imported mesh inspection
- named-set or boundary-tag driven configuration
- immediate VTK output for visual inspection

Primary anchors:

- `examples/mfem_ex39_named_attributes.rs`
- VTK arrays emitted by the commands above

### Layer 2: Imported-mesh HDF5/XDMF closure

Use the `fem-io` imported-workflow regressions when the user wants:

- imported mesh tag preservation through HDF5
- node-centered and cell-centered field exposure in XDMF
- executable evidence that imported metadata survives export

Primary anchors:

- `crates/io/tests/io_integration.rs`
- `hdf5_xdmf_imported_mesh_preserves_tags_and_result_metadata_workflow`
- `abaqus_named_sets_and_result_export_workflow`
- `netgen_surfaceelements_boundary_mask_and_result_export_workflow`

### Layer 3: Checkpoint/restart closure

Use `mfem_ex43_hdf5_checkpoint` when the user wants:

- step-indexed checkpoint persistence
- restart from stored state
- a command-backed checkpoint baseline instead of a one-shot export

Primary anchors:

- `examples/mfem_ex43_hdf5_checkpoint.rs`
- `docs/mfem-w2-io-local-report-2026-04-18.md`
- `docs/mfem-baseline-snapshot-2026-04-18.md`

### Recommended user progression

1. Start with `mfem_ex39_named_attributes` to validate imported sets, boundary
	masks, material tags, and solve/export intent.
2. Promote the same imported-mesh expectations to HDF5/XDMF via the imported
	workflow regressions in `crates/io/tests/io_integration.rs`.
3. Move to `mfem_ex43_hdf5_checkpoint` when the workflow needs restartable,
	time-stepped persistence rather than a single exported result.

Current status:

- VTK user closure is example-level.
- HDF5/XDMF user closure is regression-backed at the IO layer.
- Checkpoint/restart closure is example-backed through `mfem_ex43_hdf5_checkpoint`.
- `EXP-002` is the work of making this closure chain easier to consume as one
  coherent imported-mesh workflow family.
