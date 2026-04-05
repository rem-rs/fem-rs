# Coefficient Types and Patterns in fem-rs — Complete Search Results

**Searched:** April 4, 2026  
**Scope:** All `crates/*/src/` Rust files

---

## 1. CORE CRATE STRUCTURE (`fem_core`)

### File: `crates/core/src/lib.rs`
**Modules:**
- `error` — `FemError` enum and `FemResult<T>` alias
- `point` — coordinate/matrix type aliases (nalgebra re-exports)
- `scalar` — floating-point scalar abstraction (`f32`/`f64`)
- `types` — index type aliases

### Error Types: `crates/core/src/error.rs`
```rust
pub enum FemError {
    Mesh(String),
    DofMapping { elem: usize, dof: usize },
    SolverDivergence(usize),
    DimMismatch { expected: usize, actual: usize },
    Io(#[from] std::io::Error),
    NegativeJacobian { elem: usize, det: f64 },
    NotImplemented(String),
}

pub type FemResult<T> = Result<T, FemError>;
```

### Scalar Trait: `crates/core/src/scalar.rs`
```rust
pub trait Scalar: Copy + Clone + Send + Sync + 'static
    + Debug + Display + Float + NumAssign + bytemuck::Pod {
    fn from_f64(v: f64) -> Self;
}
```
Implemented for: `f64` (default), `f32` (WASM/memory-critical)

### Index Type Aliases: `crates/core/src/types.rs`
```rust
pub type NodeId = u32;
pub type ElemId = u32;
pub type DofId = u32;
pub type FaceId = u32;
pub type Rank = i32;  // MPI process rank
```

### Coordinate & Matrix Aliases: `crates/core/src/point.rs`
```rust
pub type Coord2 = Point2<f64>;
pub type Coord3 = Point3<f64>;
pub type Vec2 = Vector2<f64>;
pub type Vec3 = Vector3<f64>;
pub type Mat2x2 = Matrix2<f64>;
pub type Mat3x3 = Matrix3<f64>;
```

---

## 2. COEFFICIENT PATTERNS IN ASSEMBLY

### Simple Scalar Coefficient Examples

#### `crates/assembly/src/standard/mass.rs` — MassIntegrator
```rust
pub struct MassIntegrator {
    /// Scalar density / reaction coefficient.
    pub rho: f64,
}

impl BilinearIntegrator for MassIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let w_rho = qp.weight * self.rho;  // ← uniform coefficient
        for i in 0..n { for j in 0..n {
            k_elem[i * n + j] += w_rho * qp.phi[i] * qp.phi[j];
        }}
    }
}
```

#### `crates/assembly/src/standard/diffusion.rs` — DiffusionIntegrator
```rust
pub struct DiffusionIntegrator {
    /// Scalar conductivity / diffusivity coefficient.
    pub kappa: f64,  // ← uniform coefficient
}

impl BilinearIntegrator for DiffusionIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let w_k = qp.weight * self.kappa;  // ← uniform scaling
        for i in 0..n { for j in 0..n {
            let dot = /* ∇φᵢ·∇φⱼ */;
            k_elem[i * n + j] += w_k * dot;
        }}
    }
}
```

#### `crates/assembly/src/standard/vector_mass.rs` — VectorMassIntegrator
```rust
pub struct VectorMassIntegrator {
    /// Scalar mass coefficient (α).
    pub alpha: f64,
}
```

#### `crates/assembly/src/standard/curl_curl.rs` — CurlCurlIntegrator
```rust
pub struct CurlCurlIntegrator {
    /// Permeability coefficient (μ).
    pub mu: f64,
}
```

### Spatially-Varying Coefficient Pattern

#### `crates/assembly/src/partial.rs` — PADiffusionOperator (Matrix-Free)
```rust
pub struct PADiffusionOperator<S: FESpace, K>
where
    K: Fn(&[f64]) -> f64 + Send + Sync,  // ← closure taking physical coords
{
    space:      S,
    kappa:      K,  // ← function of position
    quad_order: u8,
}

// Usage at each quadrature point:
let xp: Vec<f64> = /* compute physical coords */;
let kappa_qp = (self.kappa)(&xp);  // ← evaluate coefficient at qp
```

**Constructor variants:**
- `PADiffusionOperator::new(space, kappa_fn, quad_order)` — spatially varying
- `PADiffusionOperator::uniform(space, kappa_val, quad_order)` — constant (returns closure)

---

## 3. FUNCTION TYPES IN THE CODEBASE

### Fn Closures Found (via grep)

**Interpolation functions** — `crates/space/src/fe_space.rs`:
```rust
fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64>;
```
Scalar function: `f(x) → f64`

**Vector interpolation** — `crates/space/src/vector_h1.rs`:
```rust
pub fn interpolate_vec(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64>;
```
Vector function: `f(x) → Vec<f64>`

**ODE/Time-stepping RHS** — `crates/solver/src/ode.rs`:
```rust
pub trait TimeStepper: Send + Sync {
    fn step<F>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F)
    where F: Fn(f64, &[f64], &mut [f64]);  // ← (time, state) → derivative
}
```

**Implicit ODE Jacobian** — `crates/solver/src/ode.rs`:
```rust
pub trait ImplicitTimeStepper: Send + Sync {
    fn step_implicit<F, J>(&self, t: f64, dt: f64, u: &mut [f64], rhs: F, jac_fn: J)
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64]) -> CsrMatrix<f64>;  // ← Jacobian assembly
}
```

**Nonlinear solver** — `crates/assembly/src/nonlinear.rs`:
```rust
pub trait NonlinearForm: Send + Sync {
    fn residual(&self, u: &[f64], rhs: &[f64], r: &mut [f64]);
    fn jacobian(&self, u: &[f64]) -> CsrMatrix<f64>;
    fn n_dofs(&self) -> usize;
}
```

**DOF mapping closure** — `crates/assembly/src/assembler.rs`:
```rust
face_dofs: &dyn Fn(u32) -> Vec<DofId>,
```

**Nonlinear diffusion** — `crates/assembly/src/nonlinear.rs`:
```rust
kappa_prime: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
```

---

## 4. MESH TAGS / MATERIAL LABELS

### SimplexMesh Structure — `crates/mesh/src/simplex.rs`
```rust
pub struct SimplexMesh<const D: usize> {
    pub coords: Vec<f64>,           // node coordinates
    pub conn: Vec<NodeId>,          // element connectivity
    pub elem_tags: Vec<i32>,        // ← Physical group tag per element
    pub elem_type: ElementType,     // uniform element type
    pub face_conn: Vec<NodeId>,     // boundary face connectivity
    pub face_tags: Vec<BoundaryTag>, // ← Physical group tag per boundary face
    pub face_type: ElementType,
}
```

### MeshTopology Trait — `crates/mesh/src/topology.rs`
```rust
pub trait MeshTopology: Send + Sync {
    fn element_tag(&self, elem: ElemId) -> i32;  // ← material/domain label
    fn face_tag(&self, face: FaceId) -> i32;    // ← boundary condition label
    // ...
}
```

### BoundaryTag Type Alias — `crates/mesh/src/boundary.rs`
```rust
pub type BoundaryTag = i32;  // GMSH physical group number

pub struct PhysicalGroup {
    pub dim: u8,              // topological dimension
    pub tag: BoundaryTag,     // GMSH tag
    pub name: String,         // human-readable name
}
```

### Example: Unit Square Mesh Generator
```rust
pub fn unit_square_tri(n: usize) -> Self {
    // Boundary tag convention:
    // - 1: bottom (y = 0)
    // - 2: right  (x = 1)
    // - 3: top    (y = 1)
    // - 4: left   (x = 0)
}
```

### Assembly Usage of Tags — `crates/assembly/src/assembler.rs`
```rust
pub fn assemble_boundary_linear(..., tags: &[i32], ...) {
    for f in mesh.face_iter() {
        if !tags.contains(&mesh.face_tag(f)) { continue; }  // ← filter by tag
        // assemble boundary integral for this face
    }
}
```

---

## 5. PIECEWISE CONSTANT COEFFICIENT PATTERNS

### Current Approach: Per-Element Coefficients via Integrators

**Key pattern:** Integrators receive `QpData` containing:
- `x_phys: &[f64]` — physical coordinates of quadrature point
- `weight: f64` — integration weight × |det J|

**No current explicit "per-element constant" structure**, but can be implemented via:

1. **Closure-based** (like `PADiffusionOperator`):
   ```rust
   let elem_coeff = |x: &[f64]| -> f64 {
       if is_in_region_1(x) { 1.0 } else { 2.0 }
   };
   ```

2. **Element tag lookup** (unused in current assembly):
   ```rust
   for e in mesh.elem_iter() {
       let tag = mesh.element_tag(e);
       let coeff = coefficient_map[tag];  // ← piecewise constant
   }
   ```

3. **Lumped per-element properties** — `crates/assembly/src/partial.rs`:
   ```rust
   pub struct LumpedMassOperator<S: FESpace> {
       pub diag: Vec<f64>,  // one diagonal entry per DOF
   }
   ```

### DG Interior Penalty Assembly — `crates/assembly/src/dg.rs`
```rust
/// Scalar diffusion with uniform coefficient:
pub fn assemble_sip<S: FESpace>(
    space:      &S,
    ifl:        &InteriorFaceList,
    kappa:      f64,          // ← uniform
    sigma:      f64,          // penalty parameter
    quad_order: u8,
) -> CsrMatrix<f64> {
    // Volume terms, interior face terms, boundary face terms
    // All use the same uniform kappa
}
```

---

## 6. BLOCK MATRICES FOR MIXED/SADDLE-POINT SYSTEMS

### BlockVector — `crates/linalg/src/block.rs`
```rust
pub struct BlockVector {
    data:    Vec<f64>,      // flat coefficient vector
    offsets: Vec<usize>,    // block boundaries
}

impl BlockVector {
    pub fn block(&self, i: usize) -> &[f64] { ... }
    pub fn block_mut(&mut self, i: usize) -> &mut [f64] { ... }
}
```

**Use case:** Stokes/Navier-Stokes mixed systems with velocity/pressure blocks.

### BlockMatrix — `crates/linalg/src/block.rs`
```rust
pub struct BlockMatrix {
    pub row_sizes: Vec<usize>,
    pub col_sizes: Vec<usize>,
    blocks: Vec<Option<CsrMatrix<f64>>>,  // 2-D array of sparse blocks
}
```

---

## 7. INTEGRATOR TRAIT HIERARCHY

### Base Traits

#### QpData — `crates/assembly/src/integrator.rs`
```rust
pub struct QpData<'a> {
    pub n_dofs:    usize,           // local DOFs on this element
    pub dim:       usize,           // spatial dimension
    pub weight:    f64,             // quad weight × |det J|
    pub phi:       &'a [f64],       // basis values at qp
    pub grad_phys: &'a [f64],       // [n_dofs × dim] physical gradients
    pub x_phys:    &'a [f64],       // [dim] physical coordinates
}
```

#### BilinearIntegrator
```rust
pub trait BilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]);
}
```

#### LinearIntegrator
```rust
pub trait LinearIntegrator: Send + Sync {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]);
}
```

#### BoundaryLinearIntegrator
```rust
pub struct BdQpData<'a> {
    pub n_dofs:  usize,
    pub dim:     usize,
    pub weight:  f64,
    pub phi:     &'a [f64],
    pub x_phys:  &'a [f64],
    pub normal:  &'a [f64],  // ← outward unit normal
}

pub trait BoundaryLinearIntegrator: Send + Sync {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]);
}
```

### Vector Element Integrators

#### VectorQpData — `crates/assembly/src/vector_integrator.rs`
```rust
pub struct VectorQpData<'a> {
    pub n_dofs: usize,
    pub dim: usize,
    pub weight: f64,
    pub phi_vec: &'a [f64],     // [n_dofs × dim] vector basis values
    pub curl: &'a [f64],         // curl (2D: scalar, 3D: [n_dofs × 3])
    pub div: &'a [f64],          // [n_dofs] divergence
    pub x_phys: &'a [f64],
}
```

#### VectorBilinearIntegrator
```rust
pub trait VectorBilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]);
}
```

---

## 8. ASSEMBLY LOOP STRUCTURE

### Standard Assembly Pattern — `crates/assembly/src/assembler.rs`

```rust
pub struct Assembler;

impl Assembler {
    pub fn assemble_bilinear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],  // ← can pass multiple
        quad_order:  u8,
    ) -> CsrMatrix<f64> {
        let mesh = space.mesh();
        
        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let re = ref_elem_vol(elem_type, order);
            let quad = re.quadrature(quad_order);
            
            // Element-level Jacobian
            let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
            let j_inv_t = jac.try_inverse().unwrap().transpose();
            
            let mut k_elem = vec![0.0; n_elem_dofs * n_elem_dofs];
            
            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det_j.abs();
                
                // Evaluate basis and gradients
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);
                let xp = phys_coords(x0, &jac, xi, dim);
                
                let qp = QpData {
                    n_dofs: n_elem_dofs,
                    dim,
                    weight: w,
                    phi: &phi,
                    grad_phys: &grad_phys,
                    x_phys: &xp,
                };
                
                // ← All integrators see the same qp data
                for integ in integrators {
                    integ.add_to_element_matrix(&qp, &mut k_elem);
                }
            }
            
            coo.add_element_matrix(&global_dofs, &k_elem);
        }
        
        coo.into_csr()
    }
}
```

### Key Features for Coefficients:
1. **Physical coordinates passed to each integrator** — can use for evaluation
2. **Multiple integrators per element** — additive contribution
3. **Element tag available via `mesh.element_tag(e)`** — not currently used in loop
4. **Element Jacobian computed once per element** — not per-quadrature-point

---

## 9. FINITE ELEMENT SPACES

### FESpace Trait — `crates/space/src/fe_space.rs`
```rust
pub trait FESpace: Send + Sync {
    type Mesh: MeshTopology;
    
    fn mesh(&self) -> &Self::Mesh;
    fn n_dofs(&self) -> usize;
    fn element_dofs(&self, elem: u32) -> &[DofId];
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64>;
    fn space_type(&self) -> SpaceType;
    fn order(&self) -> u8;
    fn element_signs(&self, _elem: u32) -> Option<&[f64]> { None }
}

pub enum SpaceType {
    H1,           // continuous Lagrange
    L2,           // discontinuous (DG)
    VectorH1(u8), // vector-valued H¹
    HCurl,        // Nédélec edge elements
    HDiv,         // Raviart-Thomas face elements
}
```

### Implementations
- `H1Space` — continuous Lagrange (P1, P2)
- `L2Space` — discontinuous Lagrange (P0, P1 per element)
- `VectorH1Space` — [H¹]^d for elasticity/Stokes
- `HCurlSpace` — H(curl) Nédélec
- `HDivSpace` — H(div) Raviart-Thomas

---

## 10. SUMMARY: COEFFICIENT REPRESENTATION PATTERNS

### Pattern 1: Scalar Field Member (Uniform)
```rust
pub struct MassIntegrator {
    pub rho: f64,  // ← constant coefficient
}
```
**Use:** All standard integrators (Mass, Diffusion, etc.)

### Pattern 2: Closure/Function (Spatially Varying)
```rust
pub struct PADiffusionOperator<S, K>
where K: Fn(&[f64]) -> f64 + Send + Sync,
{
    kappa: K,
}
```
**Use:** Matrix-free operators, flexible coefficient evaluation

### Pattern 3: Element Tag Lookup (Piecewise Constant)
```rust
let tag = mesh.element_tag(elem);
let coeff = coefficient_map[&tag];  // ← map tag → coefficient
```
**Use:** Different material properties per region

### Pattern 4: Dense Diagonal/Vector (Per-DOF)
```rust
pub struct LumpedMassOperator<S> {
    pub diag: Vec<f64>,  // one entry per DOF
}
```
**Use:** Lumped mass, explicit time-stepping

### Pattern 5: Block Structure (Mixed Systems)
```rust
pub struct BlockVector {
    data: Vec<f64>,
    offsets: Vec<usize>,
}
```
**Use:** Stokes/mixed formulations with multiple physical unknowns

---

## 11. FILES WITH COEFFICIENT MENTIONS

(Verified with grep -rn "coeff|coefficient")

1. **`crates/assembly/src/dg.rs`** (line 57) — "diffusion coefficient (scalar, uniform)"
2. **`crates/assembly/src/standard/vector_mass.rs`** (line 18) — "Scalar mass coefficient (α)"
3. **`crates/assembly/src/standard/curl_curl.rs`** (line 19) — "Permeability coefficient (μ)"
4. **`crates/assembly/src/standard/diffusion.rs`** (line 21) — "Scalar conductivity / diffusivity coefficient"
5. **`crates/assembly/src/standard/mass.rs`** (line 21) — "Scalar density / reaction coefficient"
6. **`crates/assembly/src/partial.rs`** (line 74) — "density coefficient (uniform)"
7. **`crates/linalg/src/block.rs`** (line 3) — "coefficient vector into named blocks"
8. **`crates/space/src/fe_space.rs`** (line 44) — "Interpolate ... into a DOF coefficient vector"
9. **`crates/solver/src/ode.rs`** (line 136) — Runge–Kutta Butcher tableau coefficients

---

## 12. KEY OBSERVATIONS

1. **No explicit "GridFunction" type** — uses closures `Fn(&[f64]) -> f64` instead
2. **No "ScalarFn/VectorFn" type aliases** — only raw `dyn Fn` traits
3. **Per-element tags are stored but not used in current assembly** — can be exploited for piecewise coefficients
4. **Integrator pattern is extensible** — `BilinearIntegrator` trait allows custom implementations
5. **Matrix-free path supports spatially varying coefficients** — `PADiffusionOperator` with closure
6. **Standard assembled assembly assumes uniform coefficients** — integrators don't vary per-element
7. **Boundary integration filters by tag** — `assemble_boundary_linear` checks `face_tag`
8. **No nonlinear coefficient assembly yet** — only in `NonlinearForm` context with `NonlinearDiffusionForm`

---

## 13. RECOMMENDED TYPE DEFINITIONS (For Coefficient Support)

Based on the codebase patterns, consider:

```rust
// Scalar-valued coefficient functions
pub type ScalarFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

// Vector-valued coefficient functions
pub type VectorFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

// Per-element (material) coefficient resolver
pub type MaterialResolver = Box<dyn Fn(i32, &[f64]) -> f64 + Send + Sync>;
// (takes element tag + physical coords → coefficient)

// GridFunction for DOF vector representation
pub struct GridFunction {
    pub data: Vector<f64>,  // DOF coefficients
    pub space: /* FESpace type */,
}
```

