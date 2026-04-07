# fem-rs Assembly Crate: Complete Exploration

## Overview
The `fem-assembly` crate provides stateless assembly of bilinear and linear forms over finite element meshes. All public APIs are in three main files:
1. **assembler.rs** — `Assembler` (stateless assembly driver)
2. **integrator.rs** — trait definitions
3. **mixed.rs** — `MixedAssembler` for rectangular matrices
4. **standard/** — concrete integrators (diffusion, mass, elasticity, etc.)

---

## 1. Integrator Traits (integrator.rs)

### BilinearIntegrator
```rust
pub trait BilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]);
}
```
- **Contract**: ACCUMULATE into `k_elem` (row-major, `[n_dofs × n_dofs]`)
- Multiple integrators share the same matrix → must use `+=`
- Called once per quadrature point

### LinearIntegrator
```rust
pub trait LinearIntegrator: Send + Sync {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]);
}
```
- **Contract**: ACCUMULATE into `f_elem` (length `n_dofs`)
- Called once per quadrature point

### BoundaryLinearIntegrator
```rust
pub trait BoundaryLinearIntegrator: Send + Sync {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]);
}
```
- For boundary contributions (e.g. Neumann BCs)
- Works with face DOFs only

---

## 2. Quadrature Point Data

### QpData (volume quadrature)
```rust
pub struct QpData<'a> {
    pub n_dofs:    usize,        // Local DOF count on this element
    pub dim:       usize,        // Spatial dimension
    pub weight:    f64,          // w = quad_weight × |det J|
    pub phi:       &'a [f64],    // Basis values, length n_dofs
    pub grad_phys: &'a [f64],    // Physical gradients, row-major [n_dofs × dim]
    pub x_phys:    &'a [f64],    // Physical coords of QP, length dim
}
```

**Layout of grad_phys**:
```
grad_phys[i * dim + j] = ∂φᵢ/∂xⱼ    (physical-space gradient)
```

### BdQpData (boundary quadrature)
```rust
pub struct BdQpData<'a> {
    pub n_dofs:  usize,
    pub dim:     usize,
    pub weight:  f64,            // w = quad_weight × |J_face|
    pub phi:     &'a [f64],      // Face basis values
    pub x_phys:  &'a [f64],      // Physical coords on face
    pub normal:  &'a [f64],      // Outward unit normal, length dim
}
```

---

## 3. Assembler (assembler.rs)

### Structure
```rust
pub struct Assembler;

impl Assembler {
    pub fn assemble_bilinear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn BilinearIntegrator],
        quad_order:  u8,
    ) -> CsrMatrix<f64> { ... }

    pub fn assemble_linear<S: FESpace>(
        space:       &S,
        integrators: &[&dyn LinearIntegrator],
        quad_order:  u8,
    ) -> Vec<f64> { ... }

    pub fn assemble_boundary_linear(
        n_dofs:      usize,
        mesh:        &dyn MeshTopology,
        face_dofs:   &dyn Fn(u32) -> Vec<DofId>,
        order:       u8,
        integrators: &[&dyn BoundaryLinearIntegrator],
        tags:        &[i32],
        quad_order:  u8,
    ) -> Vec<f64> { ... }
}
```

### Key Methods

#### assemble_bilinear — Element Loop Pattern
1. Loop over all elements: `for e in mesh.elem_iter()`
2. For each element:
   - Get element type and reference element
   - Extract global DOF map: `space.element_dofs(e) → &[DofId]`
   - Get geometric nodes: `mesh.element_nodes(e)`
   - **Compute geometric Jacobian** from first `dim+1` vertices:
     ```
     J[i,j] = x_{j+1}[i] - x_0[i]
     ```
   - Allocate element matrix: `k_elem = vec![0.0; n_elem_dofs²]`
   - **Quadrature loop**: for each quadrature point:
     - Evaluate basis: `ref_elem.eval_basis(xi, &mut phi)`
     - Evaluate reference gradients: `ref_elem.eval_grad_basis(xi, &mut grad_ref)`
     - **Transform gradients to physical space**:
       ```
       grad_phys = J^{-T} @ grad_ref
       ```
     - Compute physical coords: `x_phys = x_0 + J @ xi`
     - For each integrator: `integ.add_to_element_matrix(&qp, &mut k_elem)`
   - **Scatter to global**: `coo.add_element_matrix(&global_dofs, &k_elem)`
3. Convert COO to CSR

#### assemble_linear — Nearly Identical
Same pattern as bilinear, except:
- Element vector: `f_elem = vec![0.0; n_ldofs]` (not n_elem_dofs!)
- Call: `integ.add_to_element_vector(&qp, &mut f_elem)`
- Scatter: `coo_add_element_vec(&global_dofs, &f_elem, &mut rhs)`

#### assemble_boundary_linear
- Loops over boundary faces: `for f in mesh.face_iter()`
- Filters by tag: `if !tags.contains(&mesh.face_tag(f)) { continue; }`
- For each face:
  - Get face DOFs via closure: `face_dofs(f) → Vec<DofId>`
  - Determine face element type (Line2, Tri3)
  - Get reference face element
  - Compute face Jacobian and outward normal
  - Quadrature loop with `BdQpData`
  - Scatter: `coo_add_element_vec(&global_dofs, &f_face, &mut rhs)`

---

## 4. Jacobian & Coordinate Transformations

### simplex_jacobian
```rust
fn simplex_jacobian<M: MeshTopology>(
    mesh: &M,
    geo_nodes: &[u32],    // first dim+1 are vertices
    dim: usize,
) -> (DMatrix<f64>, f64) // (J, det J)
```
- Builds Jacobian from first `dim+1` nodes (vertices only!)
- For a simplex: `J[i,j] = x_{j+1}[i] - x_0[i]`
- Returns both matrix and determinant

### transform_grads
```rust
fn transform_grads(
    j_inv_t: &DMatrix<f64>,      // J^{-T}
    grad_ref: &[f64],             // reference gradient
    grad_phys: &mut [f64],        // output
    n_ldofs: usize,               // number of basis functions
    dim: usize,
) {
    for i in 0..n_ldofs {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += j_inv_t[(j, k)] * grad_ref[i * dim + k];
            }
            grad_phys[i * dim + j] = s;
        }
    }
}
```
- Applies the chain rule: ∇_phys = J^{-T} ∇_ref
- Row-major layout preserved

### phys_coords
```rust
fn phys_coords(x0: &[f64], j: &DMatrix<f64>, xi: &[f64], dim: usize) -> Vec<f64>
```
- Maps reference point to physical: `x_phys = x_0 + J @ xi`

### face_jacobian_and_normal (2-D only)
```rust
fn face_jacobian_and_normal(
    mesh: &dyn MeshTopology,
    face_nodes: &[u32],
    dim: usize,
) -> (f64, Vec<f64>)  // (|J_face|, outward normal)
```
- For 2-D boundary edges only
- Edge length: `|J_face| = sqrt((x1-x0)²)`
- Outward normal: rotate tangent by -90° → `(dy, -dx) / |J_face|`

---

## 5. Reference Element Factory

```rust
fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement>
```
Supported:
- `(ElementType::Tri3, 1)` or `(ElementType::Tri6, 1)` → `TriP1`
- `(ElementType::Tri3, 2)` or `(ElementType::Tri6, 2)` → `TriP2`
- `(ElementType::Tet4, 1)` → `TetP1`

```rust
fn ref_elem_face(face_elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement>
```
Supported:
- `(ElementType::Line2, 1)` → `SegP1`
- `(ElementType::Line2, 2)` → `SegP2`
- `(ElementType::Tri3, 1)` → `TriP1`

---

## 6. Standard Integrators (standard/)

### DiffusionIntegrator
```rust
pub struct DiffusionIntegrator {
    pub kappa: f64,  // conductivity
}

// K_elem[i,j] += w · κ · (∇φᵢ · ∇φⱼ)
impl BilinearIntegrator for DiffusionIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n   = qp.n_dofs;
        let d   = qp.dim;
        let w_k = qp.weight * self.kappa;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += qp.grad_phys[i * d + k] * qp.grad_phys[j * d + k];
                }
                k_elem[i * n + j] += w_k * dot;
            }
        }
    }
}
```
- Computes: ∫_Ω κ ∇u · ∇v dx
- Stiffness matrix for Laplacian (κ=1)
- Symmetric, positive semi-definite
- Row sums ≈ 0 (Neumann compatibility)

### MassIntegrator
```rust
pub struct MassIntegrator {
    pub rho: f64,  // density/reaction
}

// M_elem[i,j] += w · ρ · φᵢ · φⱼ
impl BilinearIntegrator for MassIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n     = qp.n_dofs;
        let w_rho = qp.weight * self.rho;
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w_rho * qp.phi[i] * qp.phi[j];
            }
        }
    }
}
```
- Computes: ∫_Ω ρ u v dx
- L² mass matrix (ρ=1)
- Symmetric, positive definite

### DomainSourceIntegrator
```rust
pub struct DomainSourceIntegrator<F> where F: Fn(&[f64]) -> f64 + Send + Sync {
    f: F,
}

// f_elem[i] += w · f(x) · φᵢ
impl<F> LinearIntegrator for DomainSourceIntegrator<F> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let fval = (self.f)(qp.x_phys);
        let w_f  = qp.weight * fval;
        for i in 0..qp.n_dofs {
            f_elem[i] += w_f * qp.phi[i];
        }
    }
}
```
- Domain source: ∫_Ω f(x) v dx
- Function `f` receives physical coordinates

### NeumannIntegrator
```rust
pub struct NeumannIntegrator<F> where F: Fn(&[f64], &[f64]) -> f64 + Send + Sync {
    g: F,
}

// f_face[i] += w · g(x, n) · φᵢ
impl<F> BoundaryLinearIntegrator for NeumannIntegrator<F> {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let gval = (self.g)(qp.x_phys, qp.normal);
        let w_g  = qp.weight * gval;
        for i in 0..qp.n_dofs {
            f_face[i] += w_g * qp.phi[i];
        }
    }
}
```
- Neumann BC: ∫_Γ g(x,n) v ds
- Function receives physical coords AND outward unit normal

### ElasticityIntegrator
```rust
pub struct ElasticityIntegrator {
    pub lambda: f64,  // first Lamé parameter
    pub mu: f64,      // second Lamé parameter (shear)
}

// Assembles: ∫_Ω [ λ (∇·u)(∇·v) + 2μ ε(u):ε(v) ] dx
// where ε = ½(∇u + ∇u^T) is the symmetric strain
impl BilinearIntegrator for ElasticityIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) { ... }
}
```

**Key Points**:
- **DOF convention**: Interleaved (node-major): `[u_x(0), u_y(0), u_x(1), u_y(1), ...]`
- Global DOF for component c of node k: `k * dim + c`
- Scalar basis function index: `k` (node index, ignore components)
- Reference gradients are for the scalar basis (n_ldofs entries)
- For vector space: `n_dofs = n_ldofs * dim` (but grad_phys still has n_ldofs entries!)
- **Strain computation** uses:
  - div(φ^{k,a}) = ∂φ_k/∂x_a (the a'th component of gradient)
  - ε_{ij}^{k,a} = ½(δ_{ja} ∂φ_k/∂x_i + δ_{ia} ∂φ_k/∂x_j)
- Accumulates:
  ```
  K[(k,a),(l,b)] += w * [λ div_ka div_lb + 2μ ε:ε]
  ```

---

## 7. FESpace Trait (fem-space crate)

```rust
pub trait FESpace: Send + Sync {
    type Mesh: MeshTopology;
    
    fn mesh(&self) -> &Self::Mesh;
    fn n_dofs(&self) -> usize;
    fn element_dofs(&self, elem: u32) -> &[DofId];
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64>;
    fn space_type(&self) -> SpaceType;
    fn order(&self) -> u8;
    
    /// NEW: Orientation signs (±1.0) for each element's DOFs
    fn element_signs(&self, _elem: u32) -> Option<&[f64]> {
        None  // default: no signs
    }
}
```

### element_dofs
- Returns global DOF indices for an element
- **Length varies**:
  - H¹ scalar: length = n_ldofs
  - VectorH¹: length = n_ldofs × dim (interleaved)
  - HCurl: length = n_ldofs (one per edge)
  - HDiv: length = n_ldofs (one per face)
- **Ordering**: matches the reference element's DOF ordering

### element_signs
- **New method** (recently added)
- Returns `Some(&[f64])` with ±1.0 values
- **Only used by HCurl and HDiv** spaces (orientation-dependent)
- H¹ and L² always return `None`
- **Purpose**: For vector-valued elements, encodes whether local entity (edge/face) orientation matches global convention
- **Assembly usage**: TBD — not yet integrated into main Assembler!

---

## 8. HCurl Space Example (fem-space/src/hcurl.rs)

```rust
pub struct HCurlSpace<M: MeshTopology> {
    dofs_flat: Vec<DofId>,          // All global DOFs, element by element
    signs_flat: Vec<f64>,            // All ±1.0 signs, element by element
    dofs_per_elem: usize,            // 3 for tri, 6 for tet
    // ...
}

impl<M: MeshTopology> HCurlSpace<M> {
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = elem as usize * self.dofs_per_elem;
        &self.signs_flat[start..start + self.dofs_per_elem]
    }
}
```

**Sign Convention**:
- Global edge orientation: "from smaller to larger vertex index"
- Local edge traverses vertices in global direction → sign = +1
- Local edge traverses vertices in opposite direction → sign = -1
- **Purpose**: Ensures tangential continuity across elements

---

## 9. MixedAssembler (mixed.rs)

### Structure
```rust
pub struct MixedAssembler;

impl MixedAssembler {
    pub fn assemble_bilinear<SR, SC>(
        row_space:   &SR,
        col_space:   &SC,
        integrators: &[&dyn MixedBilinearIntegrator],
        quad_order:  u8,
    ) -> CsrMatrix<f64>
    where
        SR: FESpace,
        SC: FESpace,
    { ... }
}
```

### MixedBilinearIntegrator Trait
```rust
pub trait MixedBilinearIntegrator: Send + Sync {
    fn add_to_element_matrix(
        &self,
        qp_row: &QpData<'_>,  // row space basis data
        qp_col: &QpData<'_>,  // col space basis data
        m_elem: &mut [f64],   // n_row_dofs × n_col_dofs
    );
}
```

**Key Difference from Regular Assembler**:
- Assembles **rectangular** matrices: `n_rows = row_space.n_dofs()`, `n_cols = col_space.n_dofs()`
- Both spaces must be on the **same mesh**
- Integrators receive **two separate QpData**: one for each space
- Element loop still single (one element at a time, both spaces evaluated)

### Element Loop Pattern
```
for e in mesh.elem_iter():
    get ref_elem_row, ref_elem_col (possibly different orders)
    get global_rows, global_cols (from element_dofs of each space)
    compute Jacobian (shared by both spaces)
    allocate m_elem [n_elem_rows × n_elem_cols]
    
    for qp in quadrature:
        eval basis/grad for row space → phi_r, grad_phys_r
        eval basis/grad for col space → phi_c, grad_phys_c
        create qp_r, qp_c (separate QpData)
        for integ in integrators:
            integ.add_to_element_matrix(&qp_r, &qp_c, &mut m_elem)
    
    scatter m_elem into COO:
        for (ir, global_row) in global_rows.enumerate():
            for (ic, global_col) in global_cols.enumerate():
                coo.add(global_row, global_col, m_elem[ir * n_elem_cols + ic])

return coo.into_csr()
```

### Built-in Integrators

#### PressureDivIntegrator
```rust
pub struct PressureDivIntegrator;

// Computes: b(u, p) = -∫ p (∇·u) dx
// m_elem[j, col_ik] += -w · p_j · (∂u^{i,k}/∂x_k)
impl MixedBilinearIntegrator for PressureDivIntegrator {
    fn add_to_element_matrix(&self, qp_row, qp_col, m_elem) {
        let n_p   = qp_row.n_dofs;
        let n_u   = qp_col.n_dofs;
        let dim   = qp_col.dim;
        let w     = qp_col.weight;
        let n_nodes_u = n_u / dim;
        
        for j in 0..n_p {
            let pj = qp_row.phi[j];
            for k in 0..n_nodes_u {
                for c in 0..dim {
                    let col = k * dim + c;
                    let div_ukc = qp_col.grad_phys[k * dim + c];
                    m_elem[j * n_u + col] += -w * pj * div_ukc;
                }
            }
        }
    }
}
```
- Row space = pressure (scalar, typically L² or H¹)
- Col space = velocity (vector, typically VectorH¹)
- Velocity DOFs are **interleaved** by component
- Divergence = ∑_c ∂u_c/∂x_c (trace of gradient)
- For component c of node k: gradient is `grad_phys[k*dim + c]`

#### DivIntegrator
```rust
pub struct DivIntegrator;

// Same as PressureDivIntegrator but with positive sign:
// b(u, p) = ∫ p (∇·u) dx
```

---

## 10. Scattered-to-Global Pattern

### COO Accumulation (bilinear)
```rust
let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
// ...
for e in mesh.elem_iter():
    let global_dofs = space.element_dofs(e);
    let k_elem = /* ... computed element matrix ... */
    coo.add_element_matrix(&global_dofs, &k_elem);  // ← This does the scatter
```

**What add_element_matrix does**:
```rust
fn add_element_matrix(&mut self, dofs: &[usize], k_elem: &[f64]) {
    let n = dofs.len();
    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            self.add(gi, gj, k_elem[i * n + j]);  // ← COO.add sums duplicates
        }
    }
}
```

### Vector Accumulation (linear)
```rust
fn coo_add_element_vec(dofs: &[usize], f_elem: &[f64], rhs: &mut [f64]) {
    for (&d, &v) in dofs.iter().zip(f_elem.iter()) {
        rhs[d] += v;
    }
}
```

### Mixed Assembler Scatter
```rust
for (ir, &gr) in global_rows.iter().enumerate() {
    for (ic, &gc) in global_cols.iter().enumerate() {
        coo.add(gr, gc, m_elem[ir * n_elem_cols + ic]);
    }
}
```

---

## 11. Face DOF Helpers

### face_dofs_p1
```rust
pub fn face_dofs_p1(mesh: &dyn MeshTopology) -> impl Fn(u32) -> Vec<DofId> + '_ {
    move |f| mesh.face_nodes(f).iter().map(|&n| n as DofId).collect()
}
```
- Simple: just return face node indices
- For P1 H¹ and L² spaces

### face_dofs_p2
```rust
pub fn face_dofs_p2<S>(space: &S) -> impl Fn(u32) -> Vec<DofId> + '_
where S: FESpace, S::Mesh: MeshTopology {
    move |f| {
        let mesh = space.mesh();
        let fn_nodes = mesh.face_nodes(f);
        let (elem, _) = mesh.face_elements(f);
        let elem_nodes = mesh.element_nodes(elem);
        let elem_dofs  = space.element_dofs(elem);
        
        // Find local vertex positions
        let pos_a = elem_nodes.iter().position(|&n| n == fn_nodes[0]).unwrap();
        let pos_b = elem_nodes.iter().position(|&n| n == fn_nodes[1]).unwrap();
        
        let dof_a = elem_dofs[pos_a];
        let dof_b = elem_dofs[pos_b];
        let edge_dof = find_edge_dof(elem_nodes, elem_dofs, pos_a, pos_b);
        
        vec![dof_a, dof_b, edge_dof]
    }
}
```
- For P2 triangular elements (6 DOFs per triangle)
- Returns: 2 vertex DOFs + 1 edge midpoint DOF
- **Important**: Assumes face is owned by an element

### find_edge_dof
```rust
fn find_edge_dof(elem_nodes: &[u32], elem_dofs: &[DofId], pos_a: usize, pos_b: usize) -> DofId {
    let (lo, hi) = if pos_a < pos_b { (pos_a, pos_b) } else { (pos_b, pos_a) };
    let edge_local = match (lo, hi) {
        (0, 1) => 3,  // edge DOF index in TriP2 local DOF table
        (1, 2) => 4,
        (0, 2) => 5,
        _ => panic!("TriP2 only"),
    };
    elem_dofs[edge_local]
}
```

---

## 12. Interior Faces (interior_faces.rs)

### InteriorFaceList
```rust
pub struct InteriorFace {
    pub elem_left:  ElemId,
    pub elem_right: ElemId,
    pub face_nodes: Vec<NodeId>,
}

pub struct InteriorFaceList {
    pub faces: Vec<InteriorFace>,
}

impl InteriorFaceList {
    pub fn build<M: MeshTopology>(mesh: &M) -> Self { ... }
}
```

**Use case**: For DG assembly; mesh doesn't store interior faces, must enumerate from element connectivity.

---

## 13. Design Patterns & Key Insights

### 1. **Element Loop Inner Loop (Core Assembly Pattern)**
```rust
for e in mesh.elem_iter() {
    // Get element meta
    let ref_elem  = ref_elem_vol(elem_type, order);
    let global_dofs = space.element_dofs(e);  // ← KEY: space tells us DOF mapping
    let nodes = mesh.element_nodes(e);
    
    // Compute Jacobian (from first dim+1 vertices only)
    let (jac, det_j) = simplex_jacobian(mesh, nodes, dim);
    let j_inv_t = jac.try_inverse().transpose();
    
    // Allocate element matrix/vector
    let mut k_elem = vec![0.0; n_elem_dofs²];
    
    // Quadrature loop
    for (q, xi) in quad.points.iter().enumerate() {
        let w = quad.weights[q] * det_j.abs();
        
        // Evaluate reference element
        ref_elem.eval_basis(xi, &mut phi);
        ref_elem.eval_grad_basis(xi, &mut grad_ref);
        
        // Transform gradients
        transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);
        
        // Physical coords
        let xp = phys_coords(x0, &jac, xi, dim);
        
        // Create QpData
        let qp = QpData { n_dofs: n_elem_dofs, dim, weight: w, 
                          phi: &phi, grad_phys: &grad_phys, x_phys: &xp };
        
        // Call integrators
        for integ in integrators {
            integ.add_to_element_matrix(&qp, &mut k_elem);
        }
    }
    
    // Scatter to global
    coo.add_element_matrix(&global_dofs, &k_elem);
}
```

### 2. **Gradient Storage & Semantics**
- **Reference gradients**: `grad_ref[i*dim + j] = ∂φᵢ/∂ξⱼ` (reference space)
- **Physical gradients**: `grad_phys[i*dim + j] = ∂φᵢ/∂xⱼ` (physical space)
- Both are **row-major**, indexed by basis function first
- **Transformation**: apply `J^{-T}` to columns

### 3. **Vector Element Handling**
For interleaved DOF spaces (VectorH¹, etc.):
- `n_ldofs` = number of scalar basis functions (nodes)
- `n_dofs` = number of element DOFs = `n_ldofs * dim`
- `element_dofs()` returns `n_dofs` indices (all interleaved)
- Gradients still have `n_ldofs` entries (scalar basis)
- Integrator (e.g. ElasticityIntegrator) **reinterprets** gradients:
  - For component c of node k, global DOF = `k*dim + c`
  - Gradient of scalar basis k = `grad_phys[k*dim + j]`

### 4. **Integrator Accumulation Contract**
- **Must use `+=`**, not `=`
- Multiple integrators can share the same element matrix
- e.g., `Assembler::assemble_bilinear(&space, &[&diff, &mass], quad_order)`
  - Both `diff` and `mass` add to the same `k_elem`

### 5. **Spaces and element_dofs() Usage**
The Assembler **never** looks at `space.element_signs()`!
- `element_signs()` only works for HCurl/HDiv spaces
- It's **available** in the FESpace trait (as optional, returns None for H¹)
- But assembly code has **no integration** yet
- This is the architectural gap for HCurl/HDiv assembly!

---

## 14. Existing Implementations Not in Standard/

### DgAssembler (dg.rs, ~510 lines)
- Discontinuous Galerkin assembly (interior-penalty DG)
- More complex: uses `InteriorFaceList` for interior faces
- Separate integrators for interior face contributions

### NonlinearForm / NewtonSolver (nonlinear.rs, ~492 lines)
- Nonlinear form assembly (Jacobian)
- Newton's method solver

### Partial Assemblers (partial.rs)
- Matrix-free operators: `PAMassOperator`, `PADiffusionOperator`
- Lumped mass: `LumpedMassOperator`

---

## 15. How a CurlCurlIntegrator Would Fit

### Step 1: Define the Integrator
```rust
pub struct CurlCurlIntegrator {
    pub mu: f64,  // coefficient
}

impl BilinearIntegrator for CurlCurlIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        // qp is for an H(curl) space → basis/grad still from scalar reference element!
        // But we're on a vector-valued space, so:
        // - n_dofs = n_ldofs * dim (likely n_ldofs = 3 for TriND1, dim = 2)
        // - grad_phys has n_ldofs entries
        
        let n = qp.n_dofs;              // total DOFs (n_ldofs * dim)
        let dim = qp.dim;               // 2 or 3
        let w = qp.weight * self.mu;
        
        // We need CURL of vector basis functions!
        // Problem: We only have scalar gradients in qp.grad_phys
        
        // Solution A: Use VectorReferenceElement (if it's available)
        //   But assembler doesn't take vector reference elements yet!
        
        // Solution B: Compute curl from scalar gradients
        //   For H(curl) Nédélec elements:
        //   Φᵢ = [Φᵢ_x, Φᵢ_y]  (2-D)
        //   curl Φᵢ = ∂Φᵢ_y/∂x - ∂Φᵢ_x/∂y (scalar)
        //   But we need the components of Φ at the QP!
        
        // This is the bottleneck:
        // We can't construct the vector basis from scalar gradients alone.
        // We need either:
        // 1. A VectorReferenceElement that gives eval_curl() directly
        // 2. The actual vector basis values at the QP
    }
}
```

### The Problem
The current Assembler **only provides**:
1. `phi` — scalar basis values
2. `grad_phys` — scalar basis physical gradients
3. No vector basis values or curl values!

For H(curl) elements, we need:
- Vector basis: **Φᵢ(x)** (2 or 3 components)
- Curl: **∇ × Φᵢ(x)** (scalar in 2-D, vector in 3-D)

### Solution Options

**Option 1: Extend Assembler to support VectorReferenceElement**
```rust
// New signature for vector spaces:
pub fn assemble_bilinear_vector<S: FESpace>(
    space:       &S,
    integrators: &[&dyn BilinearIntegratorVector],  // new trait
    quad_order:  u8,
) -> CsrMatrix<f64> {
    // Query space.vector_ref_elem() instead of ref_elem_vol()
    // Call vref_elem.eval_basis_vec(xi, &mut vec_phi)
    // Call vref_elem.eval_curl(xi, &mut curl_vals)
    // Pass VectorQpData with these
}
```

**Option 2: Use element_signs() in Assembler**
```rust
// In the main assembly loop:
let signs = space.element_signs(e);
// If Some(s), scale basis values by signs before integrators

// Integrators then receive scaled basis that ensures continuity
```

This is simpler but only handles sign correction, not curl computation.

**Option 3: Integrators construct curl from gradients themselves**
(Not recommended — repeating work across all integrators)

---

## 16. Crate Dependencies

```
fem-assembly → fem-element (ReferenceElement, VectorReferenceElement, lagrange elements, nedelec)
            → fem-mesh       (SimplexMesh, MeshTopology)
            → fem-space      (FESpace, H1Space, VectorH1Space, HCurlSpace)
            → fem-linalg     (CooMatrix, CsrMatrix, Vector)
            → nalgebra       (DMatrix)
```

**Note**: fem-element **defines VectorReferenceElement** but assembler doesn't use it yet!

---

## 17. Testing & Conventions

### Unit Tests
- **DiffusionIntegrator**: row sums ≈ 0, symmetry
- **MassIntegrator**: symmetry, ∫1 dx = domain area
- **ElasticityIntegrator**: symmetry, rigid body (row sums ≈ 0)

### Integration Tests (poisson.rs)
- Solves -Δu = f on unit square
- Convergence tests: P1 (O(h²)), P2 (O(h³))
- Patch test: linear function exactly represented

### Key Conventions
- **DOF ordering**: matches ReferenceElement
- **Interleaved for vectors**: [u_x(0), u_y(0), u_x(1), u_y(1), ...]
- **Global edges for HCurl**: min→max vertex index
- **Sign convention**: ±1 encodes entity orientation

---

## Summary Table

| Component | Type | Signature | Purpose |
|-----------|------|-----------|---------|
| `Assembler::assemble_bilinear` | fn | `(space, integrators, quad_order) → CsrMatrix` | Main assembly driver |
| `Assembler::assemble_linear` | fn | `(space, integrators, quad_order) → Vec<f64>` | RHS assembly |
| `Assembler::assemble_boundary_linear` | fn | `(n_dofs, mesh, face_dofs, order, integrators, tags, quad_order) → Vec<f64>` | Boundary conditions |
| `BilinearIntegrator` | trait | `add_to_element_matrix(&qp, &mut k_elem)` | Bilinear form per QP |
| `LinearIntegrator` | trait | `add_to_element_vector(&qp, &mut f_elem)` | Linear form per QP |
| `QpData` | struct | volume quadrature data | n_dofs, dim, weight, phi, grad_phys, x_phys |
| `DiffusionIntegrator` | struct | κ | ∫κ∇u·∇v |
| `MassIntegrator` | struct | ρ | ∫ρuv |
| `ElasticityIntegrator` | struct | λ, μ | ∫[λ(∇·u)(∇·v) + 2μ ε:ε] |
| `MixedAssembler::assemble_bilinear` | fn | `(row_space, col_space, integrators, quad_order) → CsrMatrix` | Rectangular assembly |
| `MixedBilinearIntegrator` | trait | `add_to_element_matrix(&qp_row, &qp_col, &mut m_elem)` | Mixed form per QP |
| `PressureDivIntegrator` | struct | — | -∫p(∇·u) |
| `DivIntegrator` | struct | — | ∫p(∇·u) |
| `FESpace::element_dofs` | fn | `(elem) → &[DofId]` | Global DOF map per element |
| `FESpace::element_signs` | fn | `(elem) → Option<&[f64]>` | Orientation ±1 per DOF (HCurl/HDiv) |
| `face_dofs_p1` | fn | `(mesh) → Fn(face) → Vec<DofId>` | P1 face DOF closure |
| `face_dofs_p2` | fn | `(space) → Fn(face) → Vec<DofId>` | P2 face DOF closure |

