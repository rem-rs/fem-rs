# ex39 Imported Named-Config Workflows

This note records the user-facing file-based workflows added to `mfem_ex39_named_attributes`.

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

Expected VTK arrays:
- `boundary_mask`
- `material_id`

## Notes

- Run commands from the repository root.
- The example uses `GMRES` for the solve path because Dirichlet row-zeroing with nonzero values is not guaranteed to preserve an SPD system.
- The solve workflow is now both material-driven and source-driven: named regions affect `kappa` and the domain source term.
- The Abaqus and Netgen file paths above are intended as minimal, inspectable templates for user-provided imported meshes.
