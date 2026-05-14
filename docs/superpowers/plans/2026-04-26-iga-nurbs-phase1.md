# IGA/NURBS Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal end-to-end NURBS/IGA path in fem-rs for 1D and 2D single-patch Poisson workflows.

**Architecture:** Implement pure basis math in `fem-element`, expose space/DOF mapping in `fem-space`, and add focused IGA assembly entry points in `fem-assembly`. Deliver in two validated slices (1D then 2D), each with tests and runnable examples.

**Tech Stack:** Rust workspace crates (`fem-element`, `fem-space`, `fem-assembly`, `examples`), existing COO/CSR linalg path, existing solver path.

---

## File Structure (planned)

- Create: `crates/element/src/iga/mod.rs` (knot vector + B-spline/NURBS evaluation)
- Modify: `crates/element/src/lib.rs` (export `iga` module)
- Create: `crates/space/src/iga.rs` (`IgaSpace1D`, `IgaSpace2D`)
- Modify: `crates/space/src/lib.rs` (export IGA spaces)
- Create: `crates/assembly/src/iga_assembler.rs` (IGA mass/diffusion/source assembly)
- Modify: `crates/assembly/src/lib.rs` (export IGA assembler API)
- Create: `examples/mfem_ex_iga_poisson_1d.rs`
- Create: `examples/mfem_ex_iga_poisson_2d_patch.rs`
- Modify: `examples/Cargo.toml` (register examples)
- Modify: `MFEM_MAPPING.md` (NURBS/IGA status update from out-of-scope to partial)

---

### Task 1: Element-level IGA kernel (`fem-element`)

**Files:**
- Create: `crates/element/src/iga/mod.rs`
- Modify: `crates/element/src/lib.rs`
- Test: inline `#[cfg(test)]` in `crates/element/src/iga/mod.rs`

- [ ] **Step 1: Write failing basis tests**

```rust
#[test]
fn bspline_partition_of_unity_p2() {
    let knots = KnotVector::new_clamped(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
    let basis = BsplineBasis::new(2, knots).unwrap();
    let x = 0.37;
    let vals = basis.nonzero_values(x).unwrap();
    let s: f64 = vals.iter().map(|(_, v)| *v).sum();
    assert!((s - 1.0).abs() < 1.0e-12);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fem-element bspline_partition_of_unity_p2 -- --nocapture`  
Expected: FAIL due to missing `iga` module/types.

- [ ] **Step 3: Implement minimal IGA kernel**

```rust
pub struct KnotVector { pub knots: Vec<f64> }
pub struct BsplineBasis { degree: usize, knots: KnotVector }
pub struct NurbsBasis { bspline: BsplineBasis, weights: Vec<f64> }
// include: find_span, active_basis_indices, nonzero_values, nonzero_derivatives
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test -p fem-element iga -- --nocapture`  
Expected: PASS for partition/local-support/derivative/NURBS-unity tests.

- [ ] **Step 5: Commit**

```bash
git add crates/element/src/iga/mod.rs crates/element/src/lib.rs
git commit -m "feat(element): add core iga knot and basis evaluation"
```

### Task 2: Space layer (`fem-space`) for 1D/2D single patch

**Files:**
- Create: `crates/space/src/iga.rs`
- Modify: `crates/space/src/lib.rs`
- Test: inline `#[cfg(test)]` in `crates/space/src/iga.rs`

- [ ] **Step 1: Write failing DOF mapping tests**

```rust
#[test]
fn iga_space_1d_dof_count_matches_control_points() {
    let s = IgaSpace1D::new_uniform_clamped(2, 6).unwrap();
    assert_eq!(s.n_dofs(), 6);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fem-space iga_space_1d_dof_count_matches_control_points -- --nocapture`  
Expected: FAIL with unresolved `IgaSpace1D`.

- [ ] **Step 3: Implement minimal spaces**

```rust
pub struct IgaSpace1D { /* degree, knots, ctrl_points, weights */ }
pub struct IgaSpace2D { /* p,q, knots_u, knots_v, ctrl grid, weights */ }
impl IgaSpace1D { pub fn span_elements(&self) -> Vec<Span1D> { ... } }
impl IgaSpace2D { pub fn span_elements(&self) -> Vec<Span2D> { ... } }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test -p fem-space iga -- --nocapture`  
Expected: PASS for DOF count, boundary DOF extraction, span-to-global map tests.

- [ ] **Step 5: Commit**

```bash
git add crates/space/src/iga.rs crates/space/src/lib.rs
git commit -m "feat(space): add iga spaces for 1d and 2d single patch"
```

### Task 3: 1D IGA assembly slice (`fem-assembly`)

**Files:**
- Create: `crates/assembly/src/iga_assembler.rs`
- Modify: `crates/assembly/src/lib.rs`
- Test: `crates/assembly/src/iga_assembler.rs` inline `#[cfg(test)]`

- [ ] **Step 1: Write failing 1D assembly tests**

```rust
#[test]
fn iga_1d_diffusion_matrix_is_symmetric() {
    let (k, _f) = assemble_poisson_1d_demo_system().unwrap();
    assert!(k.is_symmetric(1.0e-12));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fem-assembly iga_1d_diffusion_matrix_is_symmetric -- --nocapture`  
Expected: FAIL due to missing IGA assembler.

- [ ] **Step 3: Implement minimal 1D assembly**

```rust
pub fn assemble_bilinear_diffusion_iga_1d(space: &IgaSpace1D, quad_order: u8) -> CsrMatrix<f64> { ... }
pub fn assemble_bilinear_mass_iga_1d(space: &IgaSpace1D, quad_order: u8) -> CsrMatrix<f64> { ... }
pub fn assemble_linear_source_iga_1d(space: &IgaSpace1D, f: impl Fn(f64)->f64, quad_order: u8) -> Vec<f64> { ... }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test -p fem-assembly iga_1d -- --nocapture`  
Expected: PASS for symmetry, shape, positive diagonal tests.

- [ ] **Step 5: Commit**

```bash
git add crates/assembly/src/iga_assembler.rs crates/assembly/src/lib.rs
git commit -m "feat(assembly): add 1d iga mass and diffusion assembly"
```

### Task 4: 1D end-to-end example

**Files:**
- Create: `examples/mfem_ex_iga_poisson_1d.rs`
- Modify: `examples/Cargo.toml`
- Test: run command

- [ ] **Step 1: Write failing example skeleton**

```rust
fn main() -> FemResult<()> {
    // build IgaSpace1D, assemble K/f, apply Dirichlet, solve
    Ok(())
}
```

- [ ] **Step 2: Run example compile to verify it fails**

Run: `cargo run -p fem-examples --example mfem_ex_iga_poisson_1d`  
Expected: FAIL until IGA APIs are wired.

- [ ] **Step 3: Implement complete 1D example**

```rust
let space = IgaSpace1D::new_uniform_clamped(2, 16)?;
let k = assemble_bilinear_diffusion_iga_1d(&space, 4);
let f = assemble_linear_source_iga_1d(&space, |x| x.sin(), 4);
// apply bc and solve with existing solver
```

- [ ] **Step 4: Run example to verify it passes**

Run: `cargo run -p fem-examples --example mfem_ex_iga_poisson_1d`  
Expected: PASS and print residual/error metric.

- [ ] **Step 5: Commit**

```bash
git add examples/mfem_ex_iga_poisson_1d.rs examples/Cargo.toml
git commit -m "feat(examples): add 1d iga poisson end-to-end example"
```

### Task 5: 2D tensor-product IGA assembly + example

**Files:**
- Modify: `crates/assembly/src/iga_assembler.rs`
- Create: `examples/mfem_ex_iga_poisson_2d_patch.rs`
- Modify: `examples/Cargo.toml`
- Test: assembly tests and example run

- [ ] **Step 1: Write failing 2D tests**

```rust
#[test]
fn iga_2d_diffusion_shape_matches_ndofs() {
    let space = IgaSpace2D::new_uniform_clamped(2, 2, 8, 8).unwrap();
    let k = assemble_bilinear_diffusion_iga_2d(&space, 4).unwrap();
    assert_eq!(k.nrows(), space.n_dofs());
    assert_eq!(k.ncols(), space.n_dofs());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fem-assembly iga_2d_diffusion_shape_matches_ndofs -- --nocapture`  
Expected: FAIL before 2D assembly support.

- [ ] **Step 3: Implement 2D assembly and example**

```rust
pub fn assemble_bilinear_diffusion_iga_2d(space: &IgaSpace2D, quad_order: u8) -> FemResult<CsrMatrix<f64>> { ... }
pub fn assemble_linear_source_iga_2d(space: &IgaSpace2D, f: impl Fn([f64;2])->f64, quad_order: u8) -> FemResult<Vec<f64>> { ... }
```

- [ ] **Step 4: Run tests and examples**

Run:
- `cargo test -p fem-assembly iga_2d -- --nocapture`
- `cargo run -p fem-examples --example mfem_ex_iga_poisson_2d_patch`  
Expected: PASS with finite error metric.

- [ ] **Step 5: Commit**

```bash
git add crates/assembly/src/iga_assembler.rs examples/mfem_ex_iga_poisson_2d_patch.rs examples/Cargo.toml
git commit -m "feat(iga): add 2d single-patch iga poisson workflow"
```

### Task 6: Docs and mapping update

**Files:**
- Modify: `MFEM_MAPPING.md`
- Optional: crate README touch points if needed

- [ ] **Step 1: Write failing documentation check**

```text
Manual check: NURBS/IGA line must no longer be marked out-of-scope.
```

- [ ] **Step 2: Verify current docs state**

Run: `rg "NURBS|IGA|out-of-scope" MFEM_MAPPING.md`  
Expected: shows current out-of-scope marker.

- [ ] **Step 3: Update docs**

```markdown
NURBS/IGA | partial | Phase 1: 1D + 2D single-patch Poisson workflow
```

- [ ] **Step 4: Verify docs change**

Run: `rg "NURBS|IGA" MFEM_MAPPING.md`  
Expected: reflects partial/phase-1 status.

- [ ] **Step 5: Commit**

```bash
git add MFEM_MAPPING.md
git commit -m "docs(mapping): track iga phase-1 as partial support"
```

---

## Self-review

- Spec coverage: goal/scope, architecture, error handling, testing, incremental delivery, and acceptance criteria are all mapped to concrete tasks.
- Placeholder scan: no `TODO`/`TBD` placeholders in tasks.
- Type consistency: names `IgaSpace1D`, `IgaSpace2D`, and `assemble_*_iga_*` are used consistently across tasks.

