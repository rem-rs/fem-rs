# MFEM Reference Mesh Data

All mesh files from upstream MFEM `data/` directory.
Source: https://github.com/mfem/mfem/tree/master/data

## Files (58 total)

### 1D
`inline-segment.mesh`

### 2D-tri
`amr-quad.mesh` `beam-tri.mesh` `hexagon.mesh` `inline-tri.mesh`
`l-shape.mesh` `octahedron.mesh` `ref-prism.mesh` `square-mixed.mesh`
`star.mesh` `star-hilbert.mesh` `star-mixed.mesh` `star-mixed-p2.mesh`
`star-q3.mesh`

### 2D-quad
`beam-quad.mesh` `beam-quad-amr.mesh` `beam-quad-nurbs.mesh`
`inline-quad.mesh` `periodic-square.mesh` `square-disc.mesh`
`square-disc-nurbs.mesh` `square-disc-p3.mesh` `square-disc-surf.mesh`

### 2D-other
`compass.msh` `disc-nurbs.mesh` `escher.mesh` `escher-p2.mesh`
`klein-bottle.mesh` `mobius-strip.mesh` `periodic-hexagon.mesh`
`periodic-segment.mesh` `star-surf.mesh`

### 3D-tet
`beam-tet.mesh` `fichera.mesh` `fichera-amr.mesh` `fichera-mixed.mesh`
`fichera-mixed-p2.mesh` `fichera-q2.mesh` `fichera-q3.mesh`
`inline-tet.mesh`

### 3D-hex
`amr-hex.mesh` `beam-hex.mesh` `beam-hex-nurbs.mesh` `ball-nurbs.mesh`
`inline-hex.mesh` `nc3-nurbs.mesh` `periodic-cube.mesh`
`toroid-wedge.mesh`

### 3D-other
`beam-wedge.mesh` `inline-pyramid.mesh` `inline-wedge.mesh`
`pipe-nurbs.mesh`

### Periodic
`periodic-annulus-sector.msh` `periodic-cube.mesh` `periodic-cube.msh`
`periodic-hexagon.mesh` `periodic-segment.mesh` `periodic-square.mesh`
`periodic-square.msh` `periodic-torus-sector.msh`

### VTK
`fichera-q2.vtk` `square-disc-p2.vtk`

### GMSH (.msh)
`compass.msh` `periodic-annulus-sector.msh` `periodic-cube.msh`
`periodic-square.msh` `periodic-torus-sector.msh`

## Usage (from repo root)

```bash
cargo run --example mfem_ex1_poisson    -- -m data/star.mesh -o 2
cargo run --example mfem_ex2_elasticity -- -m data/beam-tri.mesh
cargo run --example mfem_ex9_dg_advection -- -m data/periodic-hexagon.mesh
cargo run --example mfem_ex6_flux_recovery -- -m data/fichera.mesh
```
