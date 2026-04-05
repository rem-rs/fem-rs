# fem-rs Coefficient Types & Patterns — Quick Reference

## 1. Current Coefficient Patterns

| Pattern | Example | File | Use Case |
|---------|---------|------|----------|
| **Uniform scalar** | `MassIntegrator { rho: f64 }` | `assembly/standard/mass.rs` | Constant material properties |
| **Spatially varying** | `PADiffusionOperator<S, K: Fn(&[f64])->f64>` | `assembly/partial.rs` | Position-dependent coefficients |
| **Per-element diagonal** | `LumpedMassOperator { diag: Vec<f64> }` | `assembly/partial.rs` | Explicit time-stepping |
| **Block partitioning** | `BlockVector { data, offsets }` | `linalg/block.rs` | Mixed/saddle-point systems |
| **Element tags** | `mesh.element_tag(elem): i32` | `mesh/simplex.rs` | Material/region labels |
| **Closure callbacks** | `dyn Fn(&[f64]) -> f64 + Send + Sync` | Various | Flexible user-defined functions |

---

## 2. Type Aliases (Existing in Core)

| Type | Definition | Location | Notes |
|------|-----------|----------|-------|
| `Scalar` | `trait: Float + NumAssign + ...` | `core/scalar.rs` | `f64` or `f32` |
| `NodeId` | `u32` | `core/types.rs` | Mesh vertex index |
| `ElemId` | `u32` | `core/types.rs` | Element (cell) index |
| `DofId` | `u32` | `core/types.rs` | DOF index |
| `FaceId` | `u32` | `core/types.rs` | Boundary face index |
| `Coord2` | `Point2<f64>` | `core/point.rs` | 2D coordinate |
| `Coord3` | `Point3<f64>` | `core/point.rs` | 3D coordinate |
| `BoundaryTag` | `i32` | `mesh/boundary.rs` | Physical group tag |

---

## 3. Missing Type Aliases (Opportunities)

| Alias | Proposed Definition | Purpose |
|-------|-------------------|---------|
| `ScalarFn` | `Box<dyn Fn(&[f64]) -> f64 + Send + Sync>` | Scalar coefficient functions |
| `VectorFn` | `Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>` | Vector coefficient functions |
| `MaterialResolver` | `Box<dyn Fn(i32, &[f64]) -> f64 + Send + Sync>` | Tag + position → coefficient |
| `GridFunction` | `struct { data: Vector<f64>, space: &FESpace }` | DOF coefficient vector + space |
| `ElementCoefficients` | `Vec<f64>` (one per element) | Per-element constant properties |

---

## 4. Integrator Architecture

### Trait Hierarchy
```
BilinearIntegrator          LinearIntegrator
└─ add_to_element_matrix    └─ add_to_element_vector

VectorBilinearIntegrator    VectorLinearIntegrator
└─ vector FE version        └─ vector FE version

BoundaryLinearIntegrator
└─ face integrals only
```

### Quadrature Point Data
```rust
pub struct QpData<'a> {
    pub n_dofs: usize,          // local DOFs
    pub dim: usize,             // spatial dimension
    pub weight: f64,            // quad weight × |det J|
    pub phi: &'a [f64],         // basis values [n_dofs]
    pub grad_phys: &[f64],      // gradients [n_dofs × dim]
    pub x_phys: &'a [f64],      // coordinates [dim]  ← use for coeff eval
}
```

---

## 5. Assembly Loop Pattern

```rust
for e in mesh.elem_iter() {
    let tag = mesh.element_tag(e);        // ← material/region label
    let nodes = mesh.element_nodes(e);
    
    for qp in quadrature_points {
        let x_phys = /* compute from Jacobian */;
        let qp_data = QpData { x_phys, weight, phi, grad_phys, ... };
        
        // All integrators receive the same qp_data
        for integrator in integrators {
            integrator.add_to_element_matrix(&qp_data, &mut k_elem);
        }
    }
}
```

**Key:** `x_phys` and `tag` available to each integrator!

---

## 6. Element Tag Usage Map

| Use | Current | File | Notes |
|-----|---------|------|-------|
| **Stored** | ✓ `SimplexMesh.elem_tags: Vec<i32>` | `mesh/simplex.rs` | One per element |
| **Accessed** | ✓ `MeshTopology.element_tag(elem)` | `mesh/topology.rs` | Trait method |
| **In assembly** | ✗ Not currently used | `assembly/assembler.rs` | Could add material map |
| **In boundary** | ✓ `face_tag` filtering | `assembly/assembler.rs` | For BC regions |

---

## 7. Crate Dependencies for Coefficients

| Crate | Purpose | Key Types | Depends On |
|-------|---------|-----------|-----------|
| `fem_core` | Foundation | `Scalar`, `NodeId`, `ElemId` | (none) |
| `fem_mesh` | Topology | `MeshTopology`, `BoundaryTag`, `elem_tag` | `core` |
| `fem_element` | Reference elements | `ReferenceElement` | `core` |
| `fem_space` | DOF maps | `FESpace`, `interpolate` | `core`, `mesh` |
| `fem_assembly` | Integrators | `BilinearIntegrator`, `QpData` | `core`, `mesh`, `space` |
| `fem_linalg` | Matrix/vectors | `CsrMatrix`, `BlockVector` | `core` |

---

## 8. Example: Adding Piecewise Constant Coefficient

### Current (workaround with closure):
```rust
let coeff_fn = |x: &[f64]| -> f64 {
    if x[0] < 0.5 { 1.0 } else { 2.0 }  // domain-based
};
let op = PADiffusionOperator::new(space, coeff_fn, quad_order);
```

### With element tag (future):
```rust
let coeff_map: HashMap<i32, f64> = [(1, 1.0), (2, 2.0)].into();
for e in mesh.elem_iter() {
    let tag = mesh.element_tag(e);
    let coeff = coeff_map[&tag];  // ← lookup
}
```

### With proposed GridFunction:
```rust
let coeff_vec = space.interpolate(&|x| get_coeff_at_point(x));
// coeff_vec.data[dof_i] = coefficient at DOF i
```

---

## 9. Files to Modify for Coefficient Support

| File | Current | Suggested Addition |
|------|---------|-------------------|
| `core/lib.rs` | Export core types | Export `ScalarFn`, `VectorFn` |
| `core/types.rs` (new?) | Index types | Add function type aliases |
| `mesh/simplex.rs` | `elem_tags` storage | Material map / resolver |
| `assembly/assembler.rs` | Uniform coefficients | Element-aware integrators |
| `assembly/partial.rs` | `PADiffusionOperator<K>` | Template for others |
| `linalg/block.rs` | Block vectors/matrices | Already present |

---

## 10. Key Code Locations (by grep)

| Search Term | Location | Count |
|-------------|----------|-------|
| `element_tag` | `mesh/topology.rs:31` | trait definition |
| `elem_tags` | `mesh/simplex.rs:20` | field definition |
| `element_tag(` | `assembly/dg.rs` (0 refs) | not used |
| `face_tag` | `assembly/assembler.rs` | used for filtering |
| `Fn(&[f64]) -> f64` | 3+ locations | scalar callbacks |
| `Fn(&[f64]) -> Vec` | 3+ locations | vector callbacks |

---

## 11. Summary Table: Coefficient Options

```
┌─ Uniform Scalar ─────────────────────────────────┐
│ pub struct MyIntegrator { pub coeff: f64 }       │
│ Usage: one value for all quadrature points      │
└─────────────────────────────────────────────────┘

┌─ Spatially Varying ──────────────────────────────┐
│ pub struct MyOp<K: Fn(&[f64])->f64> { k: K }    │
│ Usage: evaluate at each physical coordinate x   │
└─────────────────────────────────────────────────┘

┌─ Piecewise Constant (by tag) ────────────────────┐
│ let tag = mesh.element_tag(e);                  │
│ let coeff = coeff_map[&tag];                    │
│ Usage: one coefficient per material/region      │
└─────────────────────────────────────────────────┘

┌─ Per-DOF (GridFunction) ─────────────────────────┐
│ pub struct GridFunction { data: Vec<f64>, ... } │
│ Usage: interpolated coefficient at each node    │
└─────────────────────────────────────────────────┘
```

---

## 12. Design Recommendation

For a unified coefficient system, propose:

1. **Type aliases** (in `core/types.rs`):
   ```rust
   pub type ScalarFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
   pub type VectorFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
   pub type MaterialResolver = Box<dyn Fn(i32, &[f64]) -> f64 + Send + Sync>;
   ```

2. **GridFunction struct** (in `space/grid_function.rs`):
   ```rust
   pub struct GridFunction<S: FESpace> {
       pub data: Vector<f64>,
       space: S,
   }
   ```

3. **Enhanced integrators** (in `assembly/`):
   - Accept `ScalarFn` instead of bare closures
   - Support element tag lookups
   - Pass element context to integrators

4. **Material management** (in `mesh/`):
   - Registry of material properties
   - Tag → coefficient mapping

---

**Last Updated:** 2026-04-04  
**Scope:** Complete fem-rs codebase search  
**Status:** Planning phase ✓
