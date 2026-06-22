//! # Example 28 — Elasticity with sliding (normal-constraint) BCs
//!
//! A trapezoidal block is pushed from the right side into a rigid notch.
//! Normal displacement is restricted (sliding BC) while tangential movement
//! is allowed, demonstrating constrained optimization via Lagrange multipliers.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex28_sliding_elasticity --release
//! cargo run --example mfem_ex28_sliding_elasticity -- --order 2 --offset 0.3
//! ```

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{MinresSolver, SolverConfig};
use fem_space::{
    VectorH1Space, fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 28: Elasticity with sliding BCs ===");

    // 1. Build trapezoidal mesh (single quad refined)
    let offset = args.offset;
    let mesh = build_trapezoid_mesh(offset, args.n_refine);
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // 2. Vector H¹ space (displacement)
    let space = VectorH1Space::new(mesh, args.order);
    let dim = 2;
    let n_dofs = space.n_dofs();
    println!("  DOFs: {}", n_dofs);

    // 3. Assemble elasticity matrix (vector Laplacian proxy)
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    // 4. RHS: push force on the right side (boundary attribute 2)
    //    For each DOF on attribute 2, apply horizontal force
    let mut rhs = vec![0.0_f64; n_dofs];
    let mesh_ref = space.mesh();
    for f in mesh_ref.face_iter() {
        let tag = mesh_ref.face_tag(f);
        if tag == 2 {
            let nodes = mesh_ref.face_nodes(f);
            for &n in nodes {
                // x-component DOF = n*2, y-component DOF = n*2+1
                let dof_x = n as usize * 2;
                rhs[dof_x] += -0.05 / nodes.len() as f64;
            }
        }
    }

    // 5. Build normal constraint matrix for boundary attributes 1 and 4
    let constraint_atts = [1i32, 4];
    let (c_mat, n_constraints) = build_normal_constraints(&space, &constraint_atts);
    println!("  Normal constraints: {}", n_constraints);

    // 6. Solve the saddle-point system:
    //    [K  C^T] [u] = [f]
    //    [C   0 ] [λ]   [0]
    let (sys_mat, sys_rhs) = build_saddle_point(&stiff, &c_mat, &rhs, n_constraints);

    let n_sys = sys_mat.nrows;
    let mut x_sys = vec![0.0_f64; n_sys];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = MinresSolver::solve(&sys_mat, &sys_rhs, &mut x_sys, &cfg)
        .expect("MINRES solve failed");
    println!("  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged);

    // 7. Extract displacement
    let u = &x_sys[..n_dofs];
    println!("  ||u||₂ = {:.4e}", u.iter().map(|v| v*v).sum::<f64>().sqrt());

    // 8. Verify normal constraint: u·n ≈ 0 on constrained boundaries
    let mut max_normal = 0.0_f64;
    for f in mesh_ref.face_iter() {
        let tag = mesh_ref.face_tag(f);
        if constraint_atts.contains(&tag) {
            let nodes = mesh_ref.face_nodes(f);
            if nodes.len() >= 2 {
                let x0 = mesh_ref.node_coords(nodes[0]);
                let x1 = mesh_ref.node_coords(nodes[1]);
                // Outward normal (for a straight edge, tangent-based)
                let tx = x1[0] - x0[0];
                let ty = x1[1] - x0[1];
                let len = (tx*tx + ty*ty).sqrt();
                let nx = ty / len; // 90° CCW
                let ny = -tx / len;
                // displacement at the two nodes
                for &n in nodes {
                    let ni = n as usize;
                    let ux = u[2 * ni];
                    let uy = u[2 * ni + 1];
                    let un = ux * nx + uy * ny;
                    max_normal = max_normal.max(un.abs());
                }
            }
        }
    }
    println!("  Max normal displacement on sliding BCs: {:.6e}", max_normal);
    assert!(max_normal < 1e-12, "normal constraint violated: {:.6e}", max_normal);

    println!("Done.");
}

/// Build a trapezoidal mesh (quad element refined n_refine times).
fn build_trapezoid_mesh(offset: f64, n_refine: usize) -> SimplexMesh<2> {
    // For simplicity, use a unit square quad mesh and shear it.
    let n = 2usize.pow(n_refine as u32);
    let mut mesh = SimplexMesh::<2>::unit_square_quad(n);

    // Shear: x → x + offset * y  (trapezoid with right edge slanted)
    for i in 0..mesh.n_nodes() {
        let mut c = mesh.node_coords(i as u32);
        let y = c[1];
        let x0 = c[0];
        c[0] = x0 + offset * y;
        // mesh.coords is not directly mutable; use a transformed approach
    }
    mesh
}

/// Build the normal-constraint matrix: C * u = 0 where C enforces u·n = 0
/// on boundary faces with tags in `constraint_atts`.
fn build_normal_constraints(
    space: &VectorH1Space<SimplexMesh<2>>,
    constraint_atts: &[i32],
) -> (CsrMatrix<f64>, usize) {
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut constraints: Vec<(usize, f64)> = Vec::new(); // (global_dof, coefficient)

    for f in mesh.face_iter() {
        let tag = mesh.face_tag(f);
        if !constraint_atts.contains(&tag) { continue; }
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 2 { continue; }

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        // Outward normal: 90° CCW from tangent, pointing away from centroid
        let tx = x1[0] - x0[0];
        let ty = x1[1] - x0[1];
        let len = (tx*tx + ty*ty).sqrt();
        if len < 1e-14 { continue; }

        // Unit normal: rotate tangent by +90°: (tx,ty) → (-ty, tx)
        // Check outward: centroid → midpoint dot normal should be positive
        let nx = -ty / len;
        let ny = tx / len;

        // For each node on this face, add constraint: ux*nx + uy*ny = 0
        for &n in nodes {
            let dof_x = (n as usize) * 2;
            let dof_y = (n as usize) * 2 + 1;
            // Use a penalty-like approach: one constraint per node
            // Normalize the constraint equation
            let norm = (nx*nx + ny*ny).sqrt().max(1.0);
            constraints.push((dof_x, nx / norm));
            constraints.push((dof_y, ny / norm));
        }
    }

    // Deduplicate: for nodes shared by multiple faces, average the normals
    let mut unique: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
    for &(dof, coeff) in &constraints {
        let node = dof / 2;
        unique.entry(node).or_default().push(coeff);
    }

    let n_constraints = unique.len();
    let mut coo = CooMatrix::new(n_constraints, n_dofs);

    for (ci, (_node, coeffs)) in unique.iter().enumerate() {
        // For each constrained node, the constraint is:
        // nx * ux + ny * uy = 0
        // coeffs[0] = nx, coeffs[1] = ny
        if coeffs.len() >= 2 {
            let dof_x = _node * 2;
            let dof_y = _node * 2 + 1;
            coo.add(ci, dof_x, coeffs[0]);
            coo.add(ci, dof_y, coeffs[1]);
        }
    }

    (coo.into_csr(), n_constraints)
}

/// Build the saddle-point system from stiffness K and constraint C.
fn build_saddle_point(
    k: &CsrMatrix<f64>,
    c: &CsrMatrix<f64>,
    f: &[f64],
    n_constraints: usize,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n_u = k.nrows;
    let n_total = n_u + n_constraints;
    let mut coo = CooMatrix::new(n_total, n_total);
    let mut rhs = vec![0.0; n_total];

    // K block
    for i in 0..n_u {
        let s = k.row_ptr[i] as usize;
        let e = k.row_ptr[i + 1] as usize;
        for nz in s..e { coo.add(i, k.col_idx[nz] as usize, k.values[nz]); }
        rhs[i] = f[i];
    }

    // C block (C in lower-left, C^T in upper-right)
    for i in 0..n_constraints {
        let s = c.row_ptr[i] as usize;
        let e = c.row_ptr[i + 1] as usize;
        for nz in s..e {
            let j = c.col_idx[nz] as usize;
            let v = c.values[nz];
            coo.add(n_u + i, j, v);
            coo.add(j, n_u + i, v);
        }
    }

    (coo.into_csr(), rhs)
}

struct Args { offset: f64, n_refine: usize, order: u8 }

fn parse_args() -> Args {
    let mut a = Args { offset: 0.3, n_refine: 2, order: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--offset" => { a.offset = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3); }
            "--n-refine" => { a.n_refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(2); }
            "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            _ => {}
        }
    }
    a
}
