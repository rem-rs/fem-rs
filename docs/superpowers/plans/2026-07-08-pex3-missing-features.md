# pex3 Missing Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 missing features to `examples/mfem_pex3_maxwell_cavity.rs`: L² error, higher order, file output, GLVis.

**Architecture:** All features are independent additions to the existing pex3 example, no new crates or external dependencies. The example grows from ~125 lines to ~280 lines.

**Tech Stack:** fem-rs (Rust), fem-io GlVisSocket, fem-parallel ThreadLauncher

**Global Constraints:**
- All changes contained to `examples/mfem_pex3_maxwell_cavity.rs` and `crates/io/src/glvis.rs`
- No new dependencies
- Follow existing code patterns (CLI parsing style, solver setup style)
- All 4 features must work together (not mutually exclusive flags)

---

### Task 1: Higher-order Nédélec elements (`-o` flag)

**Files:**
- Modify: `examples/mfem_pex3_maxwell_cavity.rs`

**Interfaces:**
- Consumes: `HCurlSpace::new(mesh, order)` — already supports order parameter
- Produces: CLI `--order` / `-o` flag, variable quadrature order

- [ ] **Step 1: Add `order` CLI argument**

In the CLI parsing section, add order parameter:

```rust
let mut order = 1u8;
```

In the match block:
```rust
"-o" | "--order" => { i += 1; order = args[i].parse().unwrap_or(1); }
```

- [ ] **Step 2: Compute quadrature order from element order**

```rust
let quad_order = order as u8 * 2 + 2;
```

- [ ] **Step 3: Pass `order` and `quad_order` to space and assembly**

Change the `HCurlSpace::new(lm, 1)` call:
```rust
let ps = ParallelFESpace::new_for_edge_space(HCurlSpace::new(lm, order), &pm, comm.clone());
```

Change `3` (hardcoded quad_order) to `quad_order` in both `assemble_bilinear` and `assemble_linear` calls.

- [ ] **Step 4: Display order in output**

Add to rank 0 output:
```rust
if comm.rank() == 0 {
    println!("Options: order={order} quad_order={quad_order}");
    println!("Number of finite element unknowns: {n_global}");
}
```

- [ ] **Step 5: Test with order 2**

Run: `cargo run --example mfem_pex3_maxwell_cavity -- --ranks 2 --n 8 -o 2`
Expected: PCG converges, DOF count higher than order 1 (more edge DOFs per element).

- [ ] **Step 6: Commit**

```bash
git add examples/mfem_pex3_maxwell_cavity.rs
git commit -m "feat(pex3): add -o/--order flag for higher-order Nédélec elements"
```

---

### Task 2: Per-rank file output

**Files:**
- Modify: `examples/mfem_pex3_maxwell_cavity.rs`

**Interfaces:**
- Consumes: `fem_io::mfem::write_mfem` for mesh output, `write!` for solution output
- Produces: `mesh.000000`–`mesh.00000N` and `sol.000000`–`sol.00000N` per rank

- [ ] **Step 1: Import write_mfem**

```rust
use fem_io::mfem::write_mfem;
```

- [ ] **Step 2: Write mesh and solution files per rank**

After the solve, inside the ThreadLauncher closure:

```rust
// 14. Save the refined mesh and solution per rank (matching MFEM pex3 format).
{
    let mesh_name = format!("mesh.{:06}", comm.rank());
    let sol_name = format!("sol.{:06}", comm.rank());
    let mut mesh_f = std::fs::File::create(&mesh_name)
        .expect("cannot create mesh file");
    write_mfem(&mut mesh_f, ps.local_space().mesh(), None)
        .expect("mesh write failed");
    let mut sol_f = std::fs::File::create(&sol_name)
        .expect("cannot create sol file");
    // Write solution in MFEM format (one value per line)
    for &v in &u.data[..u.n_owned()] {
        writeln!(sol_f, "{:.14e}", v).expect("sol write failed");
    }
}
if comm.rank() == 0 {
    eprintln!("  Wrote mesh.XXXXXX and sol.XXXXXX per rank");
}
```

- [ ] **Step 3: Verify output files exist**

Run: `cargo run --example mfem_pex3_maxwell_cavity -- --ranks 2 --n 8`
Check: `ls mesh.000000 mesh.000001 sol.000000 sol.000001`
Expected: Files exist and are non-empty.

- [ ] **Step 4: Commit**

```bash
git add examples/mfem_pex3_maxwell_cavity.rs
git commit -m "feat(pex3): add per-rank mesh and solution file output"
```

---

### Task 3: GLVis parallel visualization

**Files:**
- Modify: `crates/io/src/glvis.rs` — add `send_parallel_solution_2d_vector` method
- Modify: `examples/mfem_pex3_maxwell_cavity.rs` — add GLVis visualization code

**Interfaces:**
- Consumes: `GlVisSocket::connect`, `GlVisSocket::send_solution_2d_vector` internals
- Produces: `GlVisSocket::send_parallel_solution_2d_vector(n_ranks, rank, mesh, vx, vy, name)`

- [ ] **Step 1: Add parallel GLVis method to GlVisSocket**

In `crates/io/src/glvis.rs`, add before the `send_solution_2d_vector` method:

```rust
/// Send a 2-D vector field solution to GLVis in parallel mode.
///
/// Prefixes the standard stream with the parallel header:
/// `parallel <n_ranks> <my_rank>` so GLVis combines solutions from all ranks.
pub fn send_parallel_solution_2d_vector(
    &mut self,
    n_ranks: usize,
    my_rank: usize,
    mesh: &Mesh<2>,
    field_x: &[f64],
    field_y: &[f64],
    field_name: &str,
) -> io::Result<()> {
    use std::fmt::Write;
    // Parallel header: "parallel <np> <rank>\n"
    let mut header = String::new();
    write!(header, "parallel {} {}", n_ranks, my_rank)?;
    // The standard send_solution_2d_vector writes "solution\n" internally.
    // We need to send the header before it.
    writeln!(self.stream, "{}", header)?;
    self.send_solution_2d_vector(mesh, field_x, field_y, field_name)
}
```

- [ ] **Step 2: Add `--no-vis` / `--visualization` CLI flag to pex3**

In CLI parsing:
```rust
let mut visualization = true;
```

In match block:
```rust
"-vis" | "--visualization" => { visualization = true; }
"-no-vis" | "--no-visualization" => { visualization = false; }
```

- [ ] **Step 3: Import GlVisSocket**

```rust
use fem_io::glvis::GlVisSocket;
```

- [ ] **Step 4: Add GLVis visualization code**

After solver output (still inside the ThreadLauncher closure):

```rust
if visualization {
    // Project H(Curl) solution to vertex-based nodal field for GLVis
    let ref_elem = TriND1;
    let n_ldofs = ref_elem.n_dofs();
    let n_nodes = ps.local_space().mesh().n_nodes() as usize;

    let ref_verts: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let mut sum_x = vec![0.0_f64; n_nodes];
    let mut sum_y = vec![0.0_f64; n_nodes];
    let mut count = vec![0u32; n_nodes];
    let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

    let lm = ps.local_space().mesh();
    let n_owned_elems = pm.partition().n_local_elems;

    for e in (0..n_owned_elems as u32).map(|i| i as fem_mesh::ElemId) {
        let nodes = lm.element_nodes(e);
        let dofs: Vec<usize> = ps.local_space().element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let signs = ps.local_space().element_signs(e);

        let x0 = lm.node_coords(nodes[0]);
        let x1 = lm.node_coords(nodes[1]);
        let x2 = lm.node_coords(nodes[2]);

        let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1];
        let det_j = j00 * j11 - j01 * j10;
        let inv_det = 1.0 / det_j;
        let jit00 =  j11 * inv_det; let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det; let jit11 =  j00 * inv_det;

        for vi in 0..3 {
            ref_elem.eval_basis_vec(&ref_verts[vi], &mut ref_phi);

            let mut eh_x = 0.0_f64;
            let mut eh_y = 0.0_f64;
            for i in 0..n_ldofs {
                let px = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let py = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh_x += signs[i] * u.data[dofs[i]] * px;
                eh_y += signs[i] * u.data[dofs[i]] * py;
            }

            let nid = nodes[vi] as usize;
            sum_x[nid] += eh_x;
            sum_y[nid] += eh_y;
            count[nid] += 1;
        }
    }

    let mut e_node_x = vec![0.0_f64; n_nodes];
    let mut e_node_y = vec![0.0_f64; n_nodes];
    for i in 0..n_nodes {
        if count[i] > 0 {
            let inv = 1.0 / count[i] as f64;
            e_node_x[i] = sum_x[i] * inv;
            e_node_y[i] = sum_y[i] * inv;
        }
    }

    let n_ranks = pm.comm().n_ranks();
    let my_rank = pm.comm().rank();
    match GlVisSocket::connect("localhost", 19916) {
        Ok(mut vis) => {
            if n_ranks > 1 {
                vis.send_parallel_solution_2d_vector(
                    n_ranks, my_rank, lm, &e_node_x, &e_node_y, "E",
                ).ok();
            } else {
                vis.send_solution_2d_vector(lm, &e_node_x, &e_node_y, "E").ok();
            }
        }
        Err(e) => {
            if comm.rank() == 0 {
                eprintln!("  GLVis not available: {e}");
            }
        }
    }
}
```

- [ ] **Step 5: Test GLVis code compiles**

Run: `cargo check --example mfem_pex3_maxwell_cavity`
Expected: Compiles without errors.

- [ ] **Step 6: Commit**

```bash
git add crates/io/src/glvis.rs examples/mfem_pex3_maxwell_cavity.rs
git commit -m "feat(pex3): add parallel GLVis visualization"
```

---

### Task 4: Parallel L² error computation

**Files:**
- Modify: `examples/mfem_pex3_maxwell_cavity.rs`

**Interfaces:**
- Consumes: `pm.partition().n_local_elems` (owned element count), existing `l2_error_hcurl_exact` logic
- Produces: printed `|| E_h - E ||_{{L^2}}` value on rank 0

- [ ] **Step 1: Add L² error computation code after solve**

Inside the ThreadLauncher closure, after the PCG solve:

```rust
// Compute L² error on owned elements (ghost elements excluded via n_local_elems).
let lm = ps.local_space().mesh();
let n_owned_elems = pm.partition().n_local_elems;
let elem_type = lm.element_type(0); // assume uniform element type

let elem_err2 = match elem_type {
    ElementType::Tri3 => {
        let ref_elem = TriND1;
        compute_hcurl_l2_error_sq::<_, 2>(
            lm, ps.local_space(), &u.data, ref_elem,
            |x| exact_e(x, kappa), n_owned_elems,
        )
    }
    ElementType::Quad4 => {
        let ref_elem = QuadND1;
        compute_hcurl_l2_error_sq::<_, 2>(
            lm, ps.local_space(), &u.data, ref_elem,
            |x| exact_e(x, kappa), n_owned_elems,
        )
    }
    _ => {
        if comm.rank() == 0 {
            eprintln!("  L² error not implemented for element type {elem_type:?}");
        }
        0.0
    }
};

// Allreduce to get global squared error.
let global_err2 = comm.allreduce_sum_f64(elem_err2);
if comm.rank() == 0 {
    let l2_err = global_err2.sqrt();
    println!("\n|| E_h - E ||_{{L^2}} = {l2_err:.14e}\n");
}
```

- [ ] **Step 2: Add helper function for L² error quadrature**

Outside the ThreadLauncher closure (in the module scope of `fn main` or after it), add a generic helper:

```rust
/// Compute the squared L² error on owned elements for H(Curl) spaces.
///
/// Only integrates over the first `n_elems` elements (owned, no ghosts).
/// Returns the element-level integral, caller must allreduce.
fn compute_hcurl_l2_error_sq<R: VectorReferenceElement, const D: usize>(
    mesh: &Mesh<2>,
    space: &HCurlSpace<Mesh<2>>,
    uh: &[f64],
    ref_elem: R,
    exact: impl Fn(&[f64]) -> [f64; D],
    n_elems: usize,
) -> f64 {
    let quad = ref_elem.quadrature(6);
    let n_ldofs = ref_elem.n_dofs();
    let mut ref_phi = vec![0.0; n_ldofs * 2];
    let mut err2 = 0.0_f64;

    for e in 0..n_elems as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();
        let inv_det = 1.0 / (j00 * j11 - j01 * j10);
        let (jit00, jit01) = ( j11 * inv_det, -j10 * inv_det);
        let (jit10, jit11) = (-j01 * inv_det,  j00 * inv_det);

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + j00 * xi[0] + j01 * xi[1],
                x0[1] + j10 * xi[0] + j11 * xi[1],
            ];
            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            let mut eh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let phi_x = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let phi_y = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh[0] += signs[i] * uh[dofs[i]] * phi_x;
                eh[1] += signs[i] * uh[dofs[i]] * phi_y;
            }
            let e_exact = exact(&xp);
            let dx = eh[0] - e_exact[0];
            let dy = eh[1] - e_exact[1];
            err2 += w * (dx * dx + dy * dy);
        }
    }
    err2
}
```

- [ ] **Step 3: Import needed types**

Add to imports:
```rust
use fem_element::{VectorReferenceElement, nedelec::{TriND1, QuadND1}};
```

- [ ] **Step 4: Test L² error output**

Run: `cargo run --example mfem_pex3_maxwell_cavity -- --ranks 2 --n 8`
Expected: Shows `|| E_h - E ||_{{L^2}} = ...` with a reasonable value (~0.5-1.0 for ND1 on coarse mesh).

- [ ] **Step 5: Verify L² error matches serial ex3**

Run both:
- `cargo test --example mfem_ex3_maxwell_cavity ex3_regression_baseline -- --nocapture`
- `cargo run --example mfem_pex3_maxwell_cavity -- --ranks 1 --n 8`

The serial L² error and parallel L² error (1 rank) should match within floating-point tolerance.

- [ ] **Step 6: Commit**

```bash
git add examples/mfem_pex3_maxwell_cavity.rs
git commit -m "feat(pex3): add parallel L² error computation"
```
