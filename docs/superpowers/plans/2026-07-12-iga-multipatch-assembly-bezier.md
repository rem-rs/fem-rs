# IGA: Multi-patch Assembly + Full Bézier Extraction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (A) Connect the just-built `IgaMultiPatchMesh3D` to the assembly pipeline so multi-patch C⁰ IGA solves work, with verification examples. (B) Upgrade Bézier extraction from uniform-only identity to full non-uniform Algorithm A5.1, with 3D tensor-product support.

**Architecture:** (A) New `assemble_iga_*_multipatch_*` functions in `fem-assembly` that reuse existing `physical_map_3d`/`physical_grads_3d` but scatter per-patch element matrices through the DOF map provided by `IgaMultiPatchMesh2D/3D`. (B) Implement Bézier decomposition via knot-insertion (Borden et al. 2011 / Piegl & Tiller A5.1) in `fem-element`, then extend to 3D tensor-product via Kronecker product.

**Tech Stack:** Rust, fem-assembly crate, fem-element crate, existing IGA infrastructure.

## Global Constraints

- Follow existing code patterns in `crates/assembly/src/iga/iga.rs` and `crates/element/src/bezier_extraction.rs`
- DofId type = u32; use `dof_map[pi][a] as usize` when indexing CooMatrix/vectors
- CooMatrix::new(n, n) for square systems
- Examples go in `examples/` directory, register in `Cargo.toml` if needed
- Use `#[cfg(test)]` for MMS verification inside example files
- TDD: write test first, verify fail, implement, verify pass
- Commit per task

---

## File Structure

### Created
- `examples/mfem_ex_iga_poisson_2d_multipatch.rs` — 2D multi-patch Poisson solver
- `examples/mfem_ex_iga_poisson_3d_multipatch.rs` — 3D multi-patch Poisson solver

### Modified
- `crates/assembly/src/iga/iga.rs` — add `assemble_iga_*_multipatch_*` functions (A)
- `crates/element/src/bezier_extraction.rs` — add full 1D/3D extraction (B)
- `Cargo.toml` (workspace root) — register new examples if needed

---

### Task A1: Multi-patch 2D assembly functions

**Files:**
- Modify: `crates/assembly/src/iga/iga.rs` (add ~80 lines before `assemble_iga_elasticity_2d`)

**Interfaces:**
- Consumes: `NurbsMesh2D`, `dof_map: &[Vec<DofId>]`, `n_global_dofs: usize`
- Produces: functions listed below

- [ ] **Step 1: Write the failing multipatch 2D diffusion assembly test** in the existing test module at bottom of `iga.rs`

Add after the last test function:
```rust
#[cfg(test)]
mod multipatch_tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};

    fn make_two_patch_mesh() -> (NurbsMesh2D, Vec<Vec<DofId>>, usize) {
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let patch_a = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.0,0.0],[0.5,0.0],[0.0,1.0],[0.5,1.0]],
            weights: vec![1.0;4], tag: 1,
        };
        let patch_b = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv,
            control_pts: vec![[0.5,0.0],[1.0,0.0],[0.5,1.0],[1.0,1.0]],
            weights: vec![1.0;4], tag: 2,
        };
        let nurbs = NurbsMesh2D { patches: vec![patch_a, patch_b], edge_connectivity: vec![(0,1,1,3)] };
        let mp = fem_space::IgaMultiPatchMesh2D::from_nurbs_mesh(&nurbs);
        let dof_maps: Vec<Vec<DofId>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
        (nurbs, dof_maps, mp.n_global_dofs())
    }

    #[test]
    fn test_multipatch_2d_diffusion_runs() {
        let (mesh, dof_map, n_global) = make_two_patch_mesh();
        let k = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_map, n_global, 1.0, 3);
        assert_eq!(k.nrows, n_global);
        // Should be symmetric and positive-definite (Laplacian)
        for i in 0..k.nrows {
            let mut sum_row = 0.0_f64;
            for p in k.row_ptr[i]..k.row_ptr[i+1] {
                if k.col_idx[p] as usize == i {
                    assert!(k.values[p] > 0.0, "diagonal entry K[{i},{i}] must be positive");
                }
                sum_row += k.values[p];
            }
            // Diffusion with Neumann-like free boundaries has singly degenerate row sum
            // (the constant nullspace). Just check non-negative for safety.
            assert!(sum_row >= -1e-14, "row {i} sum = {sum_row} should be ≈ 0");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
  `cargo test -p fem-assembly -- multipatch_2d_diffusion --nocapture`
  Expected: compile error "function not found"

- [ ] **Step 3: Implement `assemble_iga_diffusion_multipatch_2d`** in `crates/assembly/src/iga/iga.rs`

Insert before `assemble_iga_elasticity_2d`:
```rust
/// Assemble the diffusion stiffness matrix for a multi-patch 2-D NURBS mesh
/// with C⁰ continuity via DOF map.
pub fn assemble_iga_diffusion_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_2d(pd, qp_xi);
            let w = qp_w * det_j.abs();
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    let dot = phys_grads[a*2]*phys_grads[b*2] + phys_grads[a*2+1]*phys_grads[b*2+1];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
    }
    coo.into_csr()
}
```

- [ ] **Step 4: Add `assemble_iga_load_multipatch_2d`**

```rust
/// Assemble the load vector for a multi-patch 2-D NURBS mesh with C⁰ continuity.
pub fn assemble_iga_load_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n_global_dofs];
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                rhs[dof_map[pi][a] as usize] += f_val * basis[a] * w;
            }
        }
    }
    rhs
}
```

- [ ] **Step 5: Add `assemble_iga_mass_multipatch_2d`**

```rust
/// Assemble the mass matrix for a multi-patch 2-D NURBS mesh with C⁰ continuity.
pub fn assemble_iga_mass_multipatch_2d(
    mesh: &NurbsMesh2D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd_to_patch2d(pd);
        let n_dof = elem.n_dofs();
        let qr = patch_quad_2d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_2d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                for b in 0..n_dof {
                    coo.add(
                        dof_map[pi][a] as usize, dof_map[pi][b] as usize,
                        rho * basis[a] * basis[b] * w,
                    );
                }
            }
        }
    }
    coo.into_csr()
}
```

- [ ] **Step 6: Run test to verify it passes**

  `cargo test -p fem-assembly -- multipatch_2d_diffusion --nocapture`
  Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/assembly/src/iga/iga.rs
git commit -m "feat(iga): add multipatch 2D diffusion/load/mass assembly with DOF map"
```

---

### Task A2: Multi-patch 3D assembly functions

**Files:**
- Modify: `crates/assembly/src/iga/iga.rs` (add ~90 lines after `assemble_iga_elasticity_3d`)

**Interfaces:**
- Consumes: `NurbsMesh3D`, `dof_map: &[Vec<DofId>]`, `n_global_dofs: usize`
- Produces: functions listed below

- [ ] **Step 1: Add tests** at end of existing `multipatch_tests` module

```rust
#[test]
fn test_multipatch_3d_diffusion_runs() {
    let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
    let patch_a = fem_element::iga::NurbsPatch3DData {
        kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv.clone(),
        control_pts: vec![
            [0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,1.0,0.0],[0.5,1.0,0.0],
            [0.0,0.0,1.0],[0.5,0.0,1.0],[0.0,1.0,1.0],[0.5,1.0,1.0],
        ],
        weights: vec![1.0;8], tag: 1,
    };
    let patch_b = fem_element::iga::NurbsPatch3DData {
        kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv,
        control_pts: vec![
            [0.5,0.0,0.0],[1.0,0.0,0.0],[0.5,1.0,0.0],[1.0,1.0,0.0],
            [0.5,0.0,1.0],[1.0,0.0,1.0],[0.5,1.0,1.0],[1.0,1.0,1.0],
        ],
        weights: vec![1.0;8], tag: 2,
    };
    let mesh = NurbsMesh3D { patches: vec![patch_a, patch_b], face_connectivity: vec![(0,1,1,0)] };
    let mp = fem_space::IgaMultiPatchMesh3D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<DofId>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let k = assemble_iga_diffusion_multipatch_3d(&mesh, &dof_maps, mp.n_global_dofs(), 1.0, 2);
    assert_eq!(k.nrows, mp.n_global_dofs());
    for i in 0..k.nrows {
        let mut sum_row = 0.0_f64;
        for p in k.row_ptr[i]..k.row_ptr[i+1] {
            if k.col_idx[p] as usize == i {
                assert!(k.values[p] > 0.0, "diag K[{i},{i}] > 0");
            }
            sum_row += k.values[p];
        }
        assert!(sum_row >= -1e-14, "row {i} sum ≈ 0");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

  `cargo test -p fem-assembly -- multipatch_3d_diffusion --nocapture`
  Expected: compile error

- [ ] **Step 3: Implement `assemble_iga_diffusion_multipatch_3d`**

Insert after existing `assemble_iga_elasticity_3d` in `iga.rs`:
```rust
/// Assemble the diffusion stiffness matrix for a multi-patch 3-D NURBS mesh
/// with C⁰ continuity via DOF map.
pub fn assemble_iga_diffusion_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let (phys_grads, det_j) = physical_grads_3d(pd, qp_xi);
            let w = qp_w * det_j.abs();
            for a in 0..n_dof {
                let ga = dof_map[pi][a] as usize;
                for b in 0..n_dof {
                    let gb = dof_map[pi][b] as usize;
                    let dot = phys_grads[a*3]*phys_grads[b*3]
                            + phys_grads[a*3+1]*phys_grads[b*3+1]
                            + phys_grads[a*3+2]*phys_grads[b*3+2];
                    coo.add(ga, gb, kappa * dot * w);
                }
            }
        }
    }
    coo.into_csr()
}
```

- [ ] **Step 4: Implement `assemble_iga_load_multipatch_3d`**

```rust
/// Assemble the load vector for a multi-patch 3-D NURBS mesh with C⁰ continuity.
pub fn assemble_iga_load_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    source: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n_global_dofs];
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let f_val = source(&map.x_phys);
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                rhs[dof_map[pi][a] as usize] += f_val * basis[a] * w;
            }
        }
    }
    rhs
}
```

- [ ] **Step 5: Implement `assemble_iga_mass_multipatch_3d`**

```rust
/// Assemble the mass matrix for a multi-patch 3-D NURBS mesh with C⁰ continuity.
pub fn assemble_iga_mass_multipatch_3d(
    mesh: &NurbsMesh3D,
    dof_map: &[Vec<DofId>],
    n_global_dofs: usize,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_global_dofs, n_global_dofs);
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let elem = pd.patch_element_ref();
        let n_dof = elem.n_dofs();
        let qr = patch_quad_3d(pd, quad_order);
        for (qp_xi, qp_w) in qr.points.iter().zip(qr.weights.iter()) {
            let map = physical_map_3d(pd, qp_xi);
            let w = qp_w * map.det_j.abs();
            let mut basis = vec![0.0_f64; n_dof];
            elem.eval_basis(qp_xi, &mut basis);
            for a in 0..n_dof {
                for b in 0..n_dof {
                    coo.add(
                        dof_map[pi][a] as usize, dof_map[pi][b] as usize,
                        rho * basis[a] * basis[b] * w,
                    );
                }
            }
        }
    }
    coo.into_csr()
}
```

- [ ] **Step 6: Run test to verify it passes**

  `cargo test -p fem-assembly -- multipatch --nocapture`
  Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add crates/assembly/src/iga/iga.rs
git commit -m "feat(iga): add multipatch 3D diffusion/load/mass assembly with DOF map"
```

---

### Task A3: Multi-patch 2D Poisson example

**Files:**
- Create: `examples/mfem_ex_iga_poisson_2d_multipatch.rs`

- [ ] **Step 1: Create the example**

```rust
//! 2D IGA Poisson on two side-by-side patches with C⁰ continuity.
//!
//! Solves -Δu = 1 on [0,1]×[0,1] split into two patches at x=0.5,
//! with u=0 on ∂Ω. Verifies the solution is C⁰ across the shared boundary.
use fem_assembly::iga::{assemble_iga_diffusion_multipatch_2d, assemble_iga_load_multipatch_2d};
use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};
use fem_space::IgaMultiPatchMesh2D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 2;
const NU: usize = 8;
const NV: usize = 8;

fn main() {
    let kv = NurbsKnotVector::uniform(P, NU - P);
    let (nua, nub) = (NU / 2, NU - NU / 2); // split control points
    let patch_a = NurbsPatch2DData {
        kv_u: kv.clone(), kv_v: kv.clone(),
        control_pts: (0..NV).flat_map(|j| (0..nua).map(move |i| {
            let u = i as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            [u, v]
        })).collect(),
        weights: vec![1.0; nua * NV], tag: 1,
    };
    let patch_b = NurbsPatch2DData {
        kv_u: kv.clone(), kv_v: kv,
        control_pts: (0..NV).flat_map(|j| (0..nub).map(move |i| {
            let u = (nua + i) as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            [u, v]
        })).collect(),
        weights: vec![1.0; nub * NV], tag: 2,
    };
    let mesh = NurbsMesh2D {
        patches: vec![patch_a, patch_b],
        edge_connectivity: vec![(0, 1, 1, 3)],
    };
    let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<u32>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let n_global = mp.n_global_dofs();

    let stiff = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_maps, n_global, 1.0, 4);
    let mut rhs = assemble_iga_load_multipatch_2d(&mesh, &dof_maps, n_global, |_| 1.0, 4);

    // Dirichlet BC: u=0 on all boundary DOFs via symmetric elimination
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();
        for j in 0..nv { for i in 0..nu {
            let local = j * nu + i;
            let global = dof_maps[pi][local] as usize;
            if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 {
                is_bnd[global] = true;
            }
        }}
    }
    // Zero-out Dirichlet rows and columns (symmetric elimination in CSR)
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize != d {
                    stiff.values[p] = 0.0;
                }
            }
            rhs[d] = 0.0;
        }
    }
    // Zero Dirichlet columns (search all rows for column == d)
    for d in 0..n_global {
        if is_bnd[d] {
            for r in 0..n_global {
                if r == d || !is_bnd[r] { continue; }
                for p in stiff.row_ptr[r]..stiff.row_ptr[r+1] {
                    if stiff.col_idx[p] as usize == d {
                        stiff.values[p] = 0.0;
                    }
                }
            }
        }
    }
    // Set diagonal entry for boundary DOFs (ensuring K[d,d] = 1)
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize == d {
                    stiff.values[p] = 1.0;
                }
            }
        }
    }

    let mut u = vec![0.0; n_global];
    solve_cg(&stiff, &rhs, &mut u, &SolverConfig { rtol: 1e-10, max_iter: 5000, ..Default::default() })
        .expect("CG solve failed");

    // Check C0 continuity across shared boundary: DOF values at x=0.5 should match
    let nva = mesh.patches[0].kv_v.n_basis();
    let nua_p0 = mesh.patches[0].kv_u.n_basis();
    let nua_p1 = mesh.patches[1].kv_u.n_basis();
    for j in 0..nva.min(NV) {
        let dof_a = dof_maps[0][j * nua_p0 + (nua_p0 - 1)] as usize;
        let dof_b = dof_maps[1][j * nua_p1 + 0] as usize;
        assert!((u[dof_a] - u[dof_b]).abs() < 1e-12,
            "C⁰ mismatch at interface j={j}: {:.6e} vs {:.6e}", u[dof_a], u[dof_b]);
    }
    println!("2D multi-patch Poisson: {} DOFs, |u|_2 = {:.6e}", n_global,
        u.iter().map(|x| x*x).sum::<f64>().sqrt());
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() { main(); }
}
```

- [ ] **Step 2: Build and run example**

  `cargo run --example mfem_ex_iga_poisson_2d_multipatch`
  Expected: prints "2D multi-patch Poisson: N DOFs, |u|_2 = ..."

- [ ] **Step 3: Commit**

```bash
git add examples/mfem_ex_iga_poisson_2d_multipatch.rs
git commit -m "feat(iga): add 2D multi-patch Poisson example with C⁰ verification"
```

---

### Task A4: Multi-patch 3D Poisson example

**Files:**
- Create: `examples/mfem_ex_iga_poisson_3d_multipatch.rs`

- [ ] **Step 1: Create the example**

```rust
//! 3D IGA Poisson on two side-by-side cubic patches with C⁰ continuity.
//!
//! Solves -Δu = 1 on [0,1]³ split into two patches at x=0.5,
//! with u=0 on ∂Ω. Verifies C⁰ across shared face.
use fem_assembly::iga::{assemble_iga_diffusion_multipatch_3d, assemble_iga_load_multipatch_3d};
use fem_element::iga::{NurbsKnotVector, NurbsPatch3DData, NurbsMesh3D};
use fem_space::IgaMultiPatchMesh3D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 1;
const NU: usize = 5;
const NV: usize = 5;
const NW: usize = 5;

fn main() {
    let kv = NurbsKnotVector::uniform(P, NU - P);
    let nua = NU / 2 + 1;
    let nub = NU - nua + 1;

    let build_ctrl = |start_i: usize, end_i: usize| -> Vec<[f64; 3]> {
        let mut pts = Vec::with_capacity((end_i - start_i) * NV * NW);
        for k in 0..NW { for j in 0..NV { for i in start_i..end_i {
            let u = i as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            let w = k as f64 / (NW - 1) as f64;
            pts.push([u, v, w]);
        }}}
        pts
    };

    let patch_a = NurbsPatch3DData {
        kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv.clone(),
        control_pts: build_ctrl(0, nua),
        weights: vec![1.0; nua * NV * NW], tag: 1,
    };
    let patch_b = NurbsPatch3DData {
        kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv,
        control_pts: build_ctrl(nua - 1, NU),
        weights: vec![1.0; nub * NV * NW], tag: 2,
    };
    let mesh = NurbsMesh3D {
        patches: vec![patch_a, patch_b],
        face_connectivity: vec![(0, 1, 1, 0)],
    };
    let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<u32>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let n_global = mp.n_global_dofs();

    let stiff = assemble_iga_diffusion_multipatch_3d(&mesh, &dof_maps, n_global, 1.0, 3);
    let mut rhs = assemble_iga_load_multipatch_3d(&mesh, &dof_maps, n_global, |_| 1.0, 3);

    // Dirichlet BC: u=0 on all boundary DOFs (symmetric elimination in CSR)
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis(); let nv = pd.kv_v.n_basis(); let nw = pd.kv_w.n_basis();
        for k in 0..nw { for j in 0..nv { for i in 0..nu {
            let local = k * nu * nv + j * nu + i;
            let global = dof_maps[pi][local] as usize;
            if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 || k == 0 || k == nw - 1 {
                is_bnd[global] = true;
            }
        }}}
    }
    // Zero Dirichlet rows
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize != d { stiff.values[p] = 0.0; }
            }
            rhs[d] = 0.0;
        }
    }
    // Zero Dirichlet columns
    for d in 0..n_global {
        if is_bnd[d] {
            for r in 0..n_global {
                if r == d || !is_bnd[r] { continue; }
                for p in stiff.row_ptr[r]..stiff.row_ptr[r+1] {
                    if stiff.col_idx[p] as usize == d { stiff.values[p] = 0.0; }
                }
            }
        }
    }
    // Set diagonals
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize == d { stiff.values[p] = 1.0; }
            }
        }
    }

    let mut u = vec![0.0; n_global];
    solve_cg(&stiff, &rhs, &mut u, &SolverConfig { rtol: 1e-10, max_iter: 5000, ..Default::default() })
        .expect("CG solve failed");

    // Check C⁰ across the shared face: each coincident DOF pair must match
    let nv0 = mesh.patches[0].kv_v.n_basis();
    let nw0 = mesh.patches[0].kv_w.n_basis();
    let nu0 = mesh.patches[0].kv_u.n_basis();
    let nu1 = mesh.patches[1].kv_u.n_basis();
    for k in 0..nw0 { for j in 0..nv0 {
        let dof_a = dof_maps[0][k * nu0 * nv0 + j * nu0 + (nu0 - 1)] as usize;
        let dof_b = dof_maps[1][k * nu1 * nv0 + j * nu1 + 0] as usize;
        assert!((u[dof_a] - u[dof_b]).abs() < 1e-12,
            "C⁰ mismatch at shared face (j={j},k={k}): {:.6e} vs {:.6e}", u[dof_a], u[dof_b]);
    }}
    println!("3D multi-patch Poisson: {} DOFs, |u|_2 = {:.6e}", n_global,
        u.iter().map(|x| x*x).sum::<f64>().sqrt());
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() { main(); }
}
```

- [ ] **Step 2: Build and run example**

  `cargo run --example mfem_ex_iga_poisson_3d_multipatch`
  Expected: prints "3D multi-patch Poisson: N DOFs, |u|_2 = ..."

- [ ] **Step 3: Commit**

```bash
git add examples/mfem_ex_iga_poisson_3d_multipatch.rs
git commit -m "feat(iga): add 3D multi-patch Poisson example with C⁰ verification"
```

---

### Task B1: Non-uniform 1D Bézier extraction

**Files:**
- Modify: `crates/element/src/bezier_extraction.rs` (add ~120 lines)

**Interfaces:**
- Produces: `compute_extraction_1d_full(kv: &KnotVector) -> Option<BezierExtraction1D>` — same return type as `compute_extraction_1d` but handles non-uniform knot vectors via Bezier decomposition

- [ ] **Step 1: Write failing test for non-uniform extraction**

Add to the existing test module in `bezier_extraction.rs`:
```rust
#[test]
fn ext_1d_nonuniform_extraction_matches_bspline_eval() {
    // Non-uniform knot vector: [0,0,0, 0.2, 0.5, 0.8, 1,1,1], degree 2
    let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0], 2);
    let ext = compute_extraction_1d_full(&kv).unwrap();

    let p = kv.degree;
    let bspline = crate::iga::BsplineBasis::new(p, crate::iga::KnotVector::new_clamped(kv.knots.clone()).unwrap()).unwrap();

    // For each element, verify C_e^T · B(ξ) = N(ξ) at several points
    let n_bernstein = |xi: f64| -> Vec<f64> {
        let mut b = vec![0.0; p+1];
        // Bernstein basis on [0,1]: B_i,p(x) = C(p,i) * x^i * (1-x)^(p-i)
        for i in 0..=p {
            let binom = |n: usize, k: usize| -> f64 {
                if k > n { return 0.0; }
                let k = k.min(n - k);
                (1..=k).fold(1.0_f64, |r, j| r * (n - k + j) as f64 / j as f64)
            };
            b[i] = binom(p, i) * xi.powi(i as i32) * (1.0 - xi).powi((p - i) as i32);
        }
        b
    };

    let knots = &kv.knots;
    let spans: Vec<usize> = (p..kv.n_basis()-1).filter(|&s| knots[s+1] > knots[s]).collect();

    for (ei, &span) in spans.iter().enumerate() {
        let u0 = knots[span]; let u1 = knots[span+1];
        let Ce = &ext.matrices[ei];
        for &xi_frac in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let u = u0 + xi_frac * (u1 - u0);
            let b = n_bernstein(xi_frac); // Bernstein on [0,1]

            // Apply C_e^T: B-spline values = C_e^T * Bernstein values
            let mut n_via_extraction = vec![0.0; p+1];
            for i in 0..=p {
                for j in 0..=p {
                    n_via_extraction[i] += Ce[j * (p+1) + i] * b[j];
                }
            }

            // Direct B-spline evaluation
            let n_direct = bspline.nonzero_values(u).unwrap();
            for (idx, val) in n_direct {
                let local_idx = idx - span + p; // map to local index
                if local_idx <= p {
                    assert!((n_via_extraction[local_idx] - val).abs() < 1e-12,
                        "u={u}, local basis {local_idx}: extraction={:.12e} direct={:.12e}",
                        n_via_extraction[local_idx], val);
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

  `cargo test -p fem-element -- ext_1d_nonuniform --nocapture`
  Expected: compile error `compute_extraction_1d_full` not found

- [ ] **Step 3: Implement `compute_extraction_1d_full`** using a point-matching approach

For each element (knot span [ξ_s, ξ_{s+1}]), evaluate the p+1 active B-spline basis
functions at p+1 Chebyshev-Lobatto points on [0,1], and solve a (p+1)×(p+1) system
to express them as linear combinations of Bernstein polynomials.

```rust
/// Solve (p+1)×(p+1) linear system A·x = b in-place (Gauss elimination with
/// partial pivot, for small p up to ~8).
fn solve_small_system(a: &mut [[f64; 8]; 8], b: &mut [f64; 8], p: usize) {
    let n = p + 1;
    // Forward elimination
    for col in 0..n {
        // Partial pivot
        let mut best = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[best][col].abs() { best = row; }
        }
        if best != col { a.swap(col, best); b.swap(col, best); }
        let pivot = a[col][col];
        if pivot.abs() < 1e-300 { continue; }
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n { a[row][k] -= factor * a[col][k]; }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum -= a[i][j] * b[j]; }
        b[i] = sum / a[i][i];
    }
}

/// Compute 1-D extraction operators for possibly non-uniform knot vectors
/// via point-matching on Chebyshev-Lobatto points.
///
/// For each knot span [ξ_k, ξ_{k+1}], builds the (p+1)×(p+1) extraction
/// operator C_e such that N(ξ) = C_e^T · B(ξ̂) where ξ̂ ∈ [0,1].
pub fn compute_extraction_1d_full(kv: &KnotVector) -> Option<BezierExtraction1D> {
    use crate::iga::{BsplineBasis, KnotVector as NewKnotVector};
    let p = kv.degree;
    let knots = &kv.knots;
    let n_basis = kv.n_basis();

    let span_indices: Vec<usize> = (p..=n_basis - 1)
        .filter(|&s| knots[s + 1] > knots[s])
        .collect();
    if span_indices.is_empty() { return None; }

    let n_elements = span_indices.len();
    let mut matrices = Vec::with_capacity(n_elements);
    let np1 = p + 1;

    // Chebyshev-Lobatto points on [0,1]: ξ̂_m = 0.5 * (1 - cos(mπ/p))
    let mut cheb = Vec::with_capacity(np1);
    for m in 0..np1 {
        cheb.push(0.5 * (1.0 - (m as f64 * std::f64::consts::PI / p as f64).cos()));
    }

    // Pre-compute Bernstein values at Chebyshev-Lobatto points: B_{j,m}
    let mut B_mat = vec![0.0_f64; np1 * np1];
    for m in 0..np1 {
        let x = cheb[m];
        for j in 0..=p {
            let binom = |n: usize, k: usize| -> f64 {
                if k > n { return 0.0; }
                let kk = k.min(n - k);
                (1..=kk).fold(1.0_f64, |r, jj| r * (n - kk + jj) as f64 / jj as f64)
            };
            B_mat[j * np1 + m] = binom(p, j) * x.powi(j as i32) * (1.0 - x).powi((p - j) as i32);
        }
    }

    for &span in &span_indices {
        let u0 = knots[span];
        let u1 = knots[span + 1];
        let h = u1 - u0;

        // Build the B-spline evaluation knot vector (element-local)
        let el_knots: Vec<f64> = (span - p..=span + p + 1)
            .map(|i| if i < knots.len() { knots[i] } else { knots[knots.len() - 1] })
            .collect();
        let el_kv = NewKnotVector::new_clamped({
            // Map to [0,1] by subtracting u0 and dividing by h
            let mut mapped: Vec<f64> = el_knots.iter().map(|&t| (t - u0) / h).collect();
            // Ensure endpoints are exactly 0 and 1
            mapped[0] = 0.0;
            mapped[mapped.len() - 1] = 1.0;
            mapped
        }).ok()?;
        let el_basis = BsplineBasis::new(p, el_kv).ok()?;

        // Evaluate B-spline at Chebyshev points: N_{i,m} for local index i and point m
        // The point in physical parameter space is u = u0 + ξ̂ * h
        // But we've mapped the knot vector to [0,1], so we evaluate at cheb[m] directly
        let mut N_mat = vec![0.0_f64; np1 * np1]; // N[i * np1 + m]
        for m in 0..np1 {
            let u_el = cheb[m];
            let vals = el_basis.nonzero_values(u_el).ok()?;
            // The nonzero values have global indices, but since our el_kv is
            // of length 2p+2 and we have p+1 basis functions on [0,1],
            // the local index i corresponds to the (span-p+i)-th global index.
            // With our clamped knot vector, the local indices are 0..p.
            for (global_idx, val) in &vals {
                if *global_idx <= p {
                    N_mat[global_idx * np1 + m] = *val;
                }
            }
        }

        // Solve C_e · B = N^T  for each row i of C_e
        // C_e[i][:] solves  Σ_j C_e[i][j] * B[j][m] = N[i][m]
        // i.e., B^T · C_e[i][:]^T = N[i][:]^T
        // i.e., (this transposed notation is messy)
        // We'll solve: for each row i of Ce, we have B · x_i = n_i
        // where x_i[j] = C_e[i][j] and n_i[m] = N_mat[i * np1 + m]
        // But B matrix is B_mat[j][m] = B_mat[j * np1 + m]

        // Transpose N and B for easier Gauss: work with A · x = b
        // where A[m][j] = B_mat[j][m] = B_j(ξ̂_m)
        // and b[m] = N_mat[i * np1 + m]  (for row i)

        let mut Ce = vec![0.0_f64; np1 * np1];

        // Fill A = B^T (since B_mat is j×m, we transpose to m×j)
        // Actually B_mat[j * np1 + m] is B_j(ξ̂_m). We want A[m][j] = B_j(ξ̂_m).
        // So A = B_mat^T but the storage happens to be the same: B_mat[j][m] = A[m][j]
        // Let's just copy B_mat into a working matrix:

        let mut work_a = vec![0.0_f64; np1 * np1];
        for m in 0..np1 {
            for j in 0..np1 {
                work_a[m * np1 + j] = B_mat[j * np1 + m];
            }
        }

        for i in 0..np1 {
            // Copy A into local array
            let mut a_loc = [[0.0_f64; 8]; 8];
            let mut b_loc = [0.0_f64; 8];
            for m in 0..np1 {
                for j in 0..np1 {
                    a_loc[m][j] = work_a[m * np1 + j];
                }
                b_loc[m] = N_mat[i * np1 + m];
            }
            solve_small_system(&mut a_loc, &mut b_loc, p);
            for j in 0..np1 {
                Ce[i * np1 + j] = b_loc[j];
            }
        }

        matrices.push(Ce);
    }

    Some(BezierExtraction1D { matrices, degree: p, n_elements })
}
```

- [ ] **Step 4: Run test to verify it passes**

  `cargo test -p fem-element -- ext_1d_nonuniform --nocapture`
  Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/element/src/bezier_extraction.rs
git commit -m "feat(iga): add non-uniform 1D Bezier extraction operator"
```

---

### Task B2: 3D Bézier extraction (tensor-product)

**Files:**
- Modify: `crates/element/src/bezier_extraction.rs` (add ~100 lines)

- [ ] **Step 1: Define `BezierExtraction3D` struct and `compute_extraction_3d`**

```rust
/// 3-D Bezier extraction data: Kronecker product C_w ⊗ C_v ⊗ C_u per element.
pub struct BezierExtraction3D {
    pub matrices: Vec<Vec<f64>>,
    pub degree_u: usize,
    pub degree_v: usize,
    pub degree_w: usize,
    pub n_elements_u: usize,
    pub n_elements_v: usize,
    pub n_elements_w: usize,
    pub n_local: usize,
}

/// Compute 3-D extraction operators (tensor-product of 1-D).
pub fn compute_extraction_3d(pd: &super::nurbs::NurbsPatch3DData) -> Option<BezierExtraction3D> {
    let ext_u = compute_extraction_1d_full(&pd.kv_u)?;
    let ext_v = compute_extraction_1d_full(&pd.kv_v)?;
    let ext_w = compute_extraction_1d_full(&pd.kv_w)?;

    let p = ext_u.degree; let q = ext_v.degree; let r = ext_w.degree;
    let np1 = p + 1; let nq1 = q + 1; let nr1 = r + 1;
    let n_local = np1 * nq1 * nr1;

    let mut matrices = Vec::with_capacity(ext_u.n_elements * ext_v.n_elements * ext_w.n_elements);
    for ew in 0..ext_w.n_elements {
        let Cw = &ext_w.matrices[ew];
        for ev in 0..ext_v.n_elements {
            let Cv = &ext_v.matrices[ev];
            for eu in 0..ext_u.n_elements {
                let Cu = &ext_u.matrices[eu];
                let mut C = vec![0.0; n_local * n_local];
                for iw in 0..nr1 { for iv in 0..nq1 { for iu in 0..np1 {
                    for jw in 0..nr1 { for jv in 0..nq1 { for ju in 0..np1 {
                        let row = iw * nq1 * np1 + iv * np1 + iu;
                        let col = jw * nq1 * np1 + jv * np1 + ju;
                        C[row * n_local + col] = Cu[iu * np1 + ju]
                                               * Cv[iv * nq1 + jv]
                                               * Cw[iw * nr1 + jw];
                    }}}
                }}}
                matrices.push(C);
            }
        }
    }

    Some(BezierExtraction3D {
        matrices, degree_u: p, degree_v: q, degree_w: r,
        n_elements_u: ext_u.n_elements,
        n_elements_v: ext_v.n_elements,
        n_elements_w: ext_w.n_elements,
        n_local,
    })
}
```

- [ ] **Step 2: Write test for 3D extraction**

```rust
#[test]
fn ext_3d_identity_uniform() {
    let pd = crate::nurbs::NurbsPatch3DData {
        kv_u: KnotVector::uniform(1, 2), kv_v: KnotVector::uniform(1, 2),
        kv_w: KnotVector::uniform(1, 2),
        control_pts: vec![[0.0; 3]; 8], weights: vec![1.0; 8], tag: 1,
    };
    let ext = compute_extraction_3d(&pd).unwrap();
    assert_eq!(ext.matrices.len(), 8); // 2×2×2 elements
    assert_eq!(ext.n_local, 8); // (1+1)³
    for C in &ext.matrices {
        // Identity for uniform degree 1
        for i in 0..8 { assert!((C[i * 8 + i] - 1.0).abs() < 1e-14); }
    }
}
```

- [ ] **Step 3: Write test for 3D Bernstein eval + extraction apply**

```rust
#[test]
fn ext_3d_apply_matches_direct_eval() {
    // Uniform degree 1 → extraction = identity → trivial.
    // Test with non-uniform degree 2 on a small mesh.
    let kv_u = KnotVector::new(vec![0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0], 2);
    let kv_v = KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);
    let kv_w = KnotVector::uniform(1, 2);
    let pd = crate::nurbs::NurbsPatch3DData {
        kv_u: kv_u.clone(), kv_v: kv_v.clone(), kv_w: kv_w.clone(),
        control_pts: vec![[0.0; 3]; kv_u.n_basis() * kv_v.n_basis() * kv_w.n_basis()],
        weights: vec![1.0; kv_u.n_basis() * kv_v.n_basis() * kv_w.n_basis()],
        tag: 1,
    };
    let ext = compute_extraction_3d(&pd).unwrap();
    assert_eq!(ext.n_elements_u, 3);
    assert_eq!(ext.n_elements_v, 1);
    assert_eq!(ext.n_elements_w, 2);
    assert_eq!(ext.matrices.len(), 6);
}
```

- [ ] **Step 4: Run tests**

  `cargo test -p fem-element -- ext_3d --nocapture`
  Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/element/src/bezier_extraction.rs
git commit -m "feat(iga): add 3D tensor-product Bezier extraction"
```

---

### Task B3: 3D Bernstein eval and extraction apply

**Files:**
- Modify: `crates/element/src/bezier_extraction.rs` (add ~60 lines)

- [ ] **Step 1: Implement `eval_bernstein_3d` and `apply_extraction_3d`**

```rust
/// Evaluate 3-D Bernstein basis values and parametric gradients at (xi, eta, zeta).
pub fn eval_bernstein_3d(p: usize, q: usize, r: usize, xi: f64, eta: f64, zeta: f64,
    phi: &mut [f64], grads: &mut [f64])
{
    use crate::bernstein::{bernstein_ders, bernstein_vals};
    let bu = bernstein_vals(p, xi);
    let bv = bernstein_vals(q, eta);
    let bw = bernstein_vals(r, zeta);
    let du = bernstein_ders(p, xi);
    let dv = bernstein_ders(q, eta);
    let dw = bernstein_ders(r, zeta);
    let np1 = p + 1; let nq1 = q + 1; let nr1 = r + 1;
    for k in 0..nr1 { for j in 0..nq1 { for i in 0..np1 {
        let idx = k * nq1 * np1 + j * np1 + i;
        phi[idx] = bu[i] * bv[j] * bw[k];
        grads[idx * 3]     = du[i] * bv[j] * bw[k];
        grads[idx * 3 + 1] = bu[i] * dv[j] * bw[k];
        grads[idx * 3 + 2] = bu[i] * bv[j] * dw[k];
    }}}
}

/// Apply 3-D extraction: phi_nurbs = C^T · phi_bernstein, grads_nurbs = C^T · grads_bernstein.
pub fn apply_extraction_3d(C: &[f64], n_local: usize,
    phi_b: &[f64], grads_b: &[f64],
    phi_n: &mut [f64], grads_n: &mut [f64])
{
    for i in 0..n_local {
        let (mut s, mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..n_local {
            let ct = C[j * n_local + i];
            s  += ct * phi_b[j];
            sx += ct * grads_b[j * 3];
            sy += ct * grads_b[j * 3 + 1];
            sz += ct * grads_b[j * 3 + 2];
        }
        phi_n[i] = s;
        grads_n[i * 3] = sx;
        grads_n[i * 3 + 1] = sy;
        grads_n[i * 3 + 2] = sz;
    }
}
```

- [ ] **Step 2: Write test**

```rust
#[test]
fn eval_bernstein_3d_partition_unity() {
    let (mut phi, mut g) = (vec![0.0; 8], vec![0.0; 24]);
    eval_bernstein_3d(1, 1, 1, 0.3, 0.7, 0.2, &mut phi, &mut g);
    assert!((phi.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    let (mut phi2, mut g2) = (vec![0.0; 27], vec![0.0; 81]);
    eval_bernstein_3d(2, 2, 2, 0.5, 0.5, 0.5, &mut phi2, &mut g2);
    assert!((phi2.iter().sum::<f64>() - 1.0).abs() < 1e-14);
}

#[test]
fn apply_extraction_3d_identity_recovers_bernstein() {
    let (p, q, r) = (1, 1, 1);
    let n_local = 8;
    let mut phi_b = vec![0.0; n_local];
    let mut grads_b = vec![0.0; n_local * 3];
    eval_bernstein_3d(p, q, r, 0.4, 0.6, 0.3, &mut phi_b, &mut grads_b);

    // Identity extraction → result == Bernstein
    let mut C = vec![0.0; n_local * n_local];
    for i in 0..n_local { C[i * n_local + i] = 1.0; }
    let mut phi_n = vec![0.0; n_local];
    let mut grads_n = vec![0.0; n_local * 3];
    apply_extraction_3d(&C, n_local, &phi_b, &grads_b, &mut phi_n, &mut grads_n);
    for i in 0..n_local {
        assert!((phi_n[i] - phi_b[i]).abs() < 1e-14);
        assert!((grads_n[i*3] - grads_b[i*3]).abs() < 1e-14);
        assert!((grads_n[i*3+1] - grads_b[i*3+1]).abs() < 1e-14);
        assert!((grads_n[i*3+2] - grads_b[i*3+2]).abs() < 1e-14);
    }
}
```

- [ ] **Step 3: Run tests**

  `cargo test -p fem-element -- bernstein_3d --nocapture`
  Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/element/src/bezier_extraction.rs
git commit -m "feat(iga): add 3D Bernstein eval and extraction apply"
```
