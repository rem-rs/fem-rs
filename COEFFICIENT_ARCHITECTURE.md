# Coefficient Architecture in fem-rs

## Current Type Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│                          FEM System                              │
└──────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌─────────────────┐      ┌──────────────────┐
            │   Coefficient   │      │  Assembly Loop   │
            │   Strategies    │      │                  │
            └─────────────────┘      └──────────────────┘
                    │                         │
        ┌───────────┼───────────┐            │
        ▼           ▼           ▼            │
    Uniform    Spatial    Per-Element       │
    Scalar     Varying    (by Tag)          │
        │           │           │            │
        │           │           │     ┌──────┴──────┐
        │           │           │     ▼             ▼
        │           │           │  QpData      Element
        │           │           │  {           Tag
        │           │           │    phi,      (i32)
        │           │           │    x_phys,
        │           │           │    ...
        │           │           │  }
        │           │           │
        └───────────┴───────────┴──→ BilinearIntegrator
                                   │
                                   ├─ MassIntegrator
                                   ├─ DiffusionIntegrator
                                   ├─ CurlCurlIntegrator
                                   └─ CustomIntegrator
```

---

## Storage Locations

### 1. Core Types (`fem_core`)
```
core/
├── scalar.rs
│   └── Scalar trait (f64, f32)
├── types.rs
│   ├── NodeId = u32
│   ├── ElemId = u32
│   ├── DofId = u32
│   ├── FaceId = u32
│   └── Rank = i32
└── point.rs
    ├── Coord2/3 = Point2/3<f64>
    └── Vec2/3, Mat2x2/3x3
```

### 2. Mesh Tags (`fem_mesh`)
```
mesh/
├── topology.rs
│   └── trait MeshTopology
│       ├── fn element_tag(&self, elem: ElemId) -> i32
│       └── fn face_tag(&self, face: FaceId) -> i32
├── simplex.rs
│   └── pub struct SimplexMesh<D>
│       ├── elem_tags: Vec<i32>  ← material IDs
│       └── face_tags: Vec<i32>  ← boundary condition IDs
└── boundary.rs
    └── pub type BoundaryTag = i32
```

### 3. Integrators (`fem_assembly`)
```
assembly/
├── integrator.rs
│   ├── pub struct QpData<'a>
│   │   ├── x_phys: &'a [f64]  ← use for coefficient eval
│   │   ├── phi, grad_phys, weight
│   │   └── dim, n_dofs
│   ├── pub trait BilinearIntegrator
│   │   └── fn add_to_element_matrix(&self, qp, k_elem)
│   └── pub trait LinearIntegrator
│       └── fn add_to_element_vector(&self, qp, f_elem)
│
├── standard/
│   ├── mass.rs
│   │   └── MassIntegrator { rho: f64 }
│   ├── diffusion.rs
│   │   └── DiffusionIntegrator { kappa: f64 }
│   ├── curl_curl.rs
│   │   └── CurlCurlIntegrator { mu: f64 }
│   └── vector_mass.rs
│       └── VectorMassIntegrator { alpha: f64 }
│
├── partial.rs
│   ├── PAMassOperator<S> { rho: f64, ... }
│   ├── PADiffusionOperator<S, K: Fn(&[f64])->f64>
│   │   └── kappa: K  ← spatially varying!
│   └── LumpedMassOperator { diag: Vec<f64> }
│
├── assembler.rs
│   ├── pub struct Assembler
│   └── pub fn assemble_bilinear(
│       space, integrators: &[&dyn BilinearIntegrator], ...)
│       │
│       ├─ for e in mesh.elem_iter()
│       │  let tag = mesh.element_tag(e);  ← available but unused!
│       │
│       └─ for qp in quadrature_points
│          ├─ compute x_phys
│          └─ for integrator in integrators
│             integrator.add_to_element_matrix(&qp_data, ...)
│
├── vector_integrator.rs
│   ├── pub struct VectorQpData<'a>
│   ├── pub trait VectorBilinearIntegrator
│   └── pub trait VectorLinearIntegrator
│
└── dg.rs
    └── DgAssembler::assemble_sip(..., kappa: f64, ...)
        └── uniform diffusion coefficient
```

### 4. Spaces (`fem_space`)
```
space/
├── fe_space.rs
│   └── pub trait FESpace
│       ├── fn interpolate(&self, f: &dyn Fn(&[f64])->f64) -> Vec
│       ├── fn element_dofs(&self, elem) -> &[DofId]
│       └── fn mesh(&self) -> &Self::Mesh
│
├── h1.rs
│   └── H1Space (continuous Lagrange)
│       └── fn interpolate(f: &dyn Fn(&[f64])->f64)
│
├── l2.rs
│   └── L2Space (discontinuous)
│
├── vector_h1.rs
│   └── VectorH1Space
│       ├── fn interpolate(f: &dyn Fn(&[f64])->f64)
│       └── fn interpolate_vec(f: &dyn Fn(&[f64])->Vec<f64>)
│
├── hcurl.rs
│   └── HCurlSpace (Nédélec)
│       └── fn interpolate_vector(f: &dyn Fn(&[f64])->Vec<f64>)
│
└── hdiv.rs
    └── HDivSpace (Raviart-Thomas)
        └── fn interpolate_vector(f: &dyn Fn(&[f64])->Vec<f64>)
```

### 5. Linear Algebra (`fem_linalg`)
```
linalg/
└── block.rs
    ├── pub struct BlockVector
    │   ├── data: Vec<f64>
    │   ├── offsets: Vec<usize>
    │   └── fn block(&self, i) -> &[f64]
    │
    └── pub struct BlockMatrix
        ├── blocks: Vec<Option<CsrMatrix<f64>>>
        └── fn spmv(&self, x, y)  ← matrix-vector product
```

---

## Data Flow: How Coefficients Are Used

### Path 1: Uniform Coefficient
```
User Code:
  let integr = MassIntegrator { rho: 1.5 };
  let mat = Assembler::assemble_bilinear(&space, &[&integr], 3);
                        │
                        ▼
Assembler Loop:
  for e in mesh.elem_iter() {
    for qp in quadrature_points {
      let qp_data = QpData { x_phys, weight, phi, ... };
      integr.add_to_element_matrix(&qp_data, &mut k_elem);
                │
                ▼
      Integrator Implementation:
        let w_rho = qp.weight * self.rho;  // ← use coefficient
        k_elem[i*n+j] += w_rho * qp.phi[i] * qp.phi[j];
    }
  }
```

### Path 2: Spatially-Varying Coefficient
```
User Code:
  let kappa = |x: &[f64]| { 1.0 + x[0]*x[0] };
  let op = PADiffusionOperator::new(space, kappa, 3);
                        │
                        ▼
Matrix-Free Loop:
  for e in mesh.elem_iter() {
    for qp in quadrature_points {
      let xp = phys_coords(x0, jac, xi);
      let kappa_qp = (self.kappa)(&xp);  // ← evaluate at qp
      // accumulate into y_elem with kappa_qp
    }
  }
```

### Path 3: Per-Element (by Tag) — NOT CURRENTLY USED
```
Proposed Pattern (Future):
  let coeff_map: HashMap<i32, f64> = [(1, 1.0), (2, 2.0)].into();
  
  Assembler Loop:
    for e in mesh.elem_iter() {
      let tag = mesh.element_tag(e);  // ← material ID
      let coeff = coeff_map[&tag];    // ← lookup coefficient
      
      for qp in quadrature_points {
        // use 'coeff' for this entire element
      }
    }
```

### Path 4: Per-DOF (GridFunction) — PROPOSED
```
User Code:
  let coeff_vec = space.interpolate(&|x| compute_coeff(x));
  // coeff_vec[i] = coefficient at DOF i
  
  Custom Integrator:
    pub struct CoefficientIntegrator<'a> {
      pub coeff: &'a [f64],
      pub dof_map: ...,
    }
    
    fn add_to_element_matrix(&self, qp: &QpData, k_elem: &mut [f64]) {
      for (local_i, global_i) in self.dof_map {
        let coeff_i = self.coeff[global_i];
        // use coeff_i
      }
    }
```

---

## Type Alias Opportunities

### Current Aliases
| Location | Alias | Type |
|----------|-------|------|
| `core/types.rs` | `NodeId` | `u32` |
| `core/types.rs` | `ElemId` | `u32` |
| `core/types.rs` | `DofId` | `u32` |
| `core/types.rs` | `FaceId` | `u32` |
| `core/point.rs` | `Coord2` | `Point2<f64>` |
| `core/point.rs` | `Coord3` | `Point3<f64>` |
| `mesh/boundary.rs` | `BoundaryTag` | `i32` |

### Proposed Additions
| Location | Alias | Type | Rationale |
|----------|-------|------|-----------|
| `core/types.rs` | `ScalarFn` | `Box<dyn Fn(&[f64]) -> f64 + Send + Sync>` | Common pattern |
| `core/types.rs` | `VectorFn` | `Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>` | H(curl), H(div) |
| `core/types.rs` | `MaterialResolver` | `Box<dyn Fn(i32, &[f64]) -> f64 + Send + Sync>` | Tag + position → coeff |
| `mesh/boundary.rs` | `ElementTag` | `i32` | Parallel to BoundaryTag |
| `space/mod.rs` | `CoeffVector` | `Vector<f64>` | For GridFunction data |

---

## Assembly Signature Enhancements

### Current
```rust
pub fn assemble_bilinear<S: FESpace>(
    space: &S,
    integrators: &[&dyn BilinearIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64>
```

### Enhanced (Proposed)
```rust
pub fn assemble_bilinear_with_materials<S: FESpace>(
    space: &S,
    integrators: &[&dyn BilinearIntegrator],
    material_map: &HashMap<ElementTag, f64>,  // or MaterialResolver
    quad_order: u8,
) -> CsrMatrix<f64>

// Alternative: pass material info to integrators via context
pub fn assemble_bilinear_contextual<S: FESpace>(
    space: &S,
    integrators: &[&dyn ContextAwareBilinearIntegrator],
    context: &AssemblyContext,  // contains material map, etc.
    quad_order: u8,
) -> CsrMatrix<f64>
```

---

## Summary: What Exists vs. What's Missing

| Feature | Status | Location/Note |
|---------|--------|---------------|
| Uniform scalar coefficients | ✅ Done | All standard integrators |
| Spatially varying coefficients | ✅ Done | `PADiffusionOperator<K: Fn>` |
| Element tags | ✅ Done | `mesh.element_tag(e): i32` |
| Per-element material lookup | ❌ Not Used | Tags exist but assembly ignores them |
| GridFunction (DOF vector) | ❌ Missing | Only interpolation available |
| ScalarFn type alias | ❌ Missing | Raw `dyn Fn` used instead |
| VectorFn type alias | ❌ Missing | Raw `dyn Fn` used instead |
| Material registry | ❌ Missing | User must manage maps |
| Context-aware integrators | ❌ Missing | QpData only, no element context |
| Per-DOF coefficients | ⚠️ Partial | Lumped mass exists; general case missing |

---

**Visualization Generated:** 2026-04-04  
**Codebase:** fem-rs complete search  
**Ready for:** Design phase implementation
