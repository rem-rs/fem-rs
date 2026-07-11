//! # Example 26 — Geometric Multigrid for Poisson  [1:1 translation of MFEM ex26]
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BCs using
//! a geometric multigrid preconditioner.
//!
//! Demonstrates a hierarchy of H¹ discretisation spaces on uniformly refined
//! meshes, with Jacobi smoothing and CG on the coarsest level.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m data/fichera.mesh
//! ```

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    GeometricMgLevel, GeometricMgHierarchy, GeometricMgConfig, GeometricMgPrecond,
};
use fem_space::{
    H1Space, fe_space::FESpace, constraints::boundary_dofs,
};

fn main() {
    let args = parse_args();

    // 1. Parse CLI (done above)

    // 2. Read mesh
    let mesh: Mesh<2> = match &args.mesh {
        Some(path) => {
            let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
            mfem.mesh2d.expect("2D mesh required")
        }
        None => Mesh::<2>::unit_square_tri(args.n),
    };

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("built-in"));
    println!("   --geometric-refinements {}", args.geometric_refs);
    println!("   --order-refinements {}", args.order_refs);
    println!("   --no-visualization");

    // 3. Auto-refine to target ≤5000 elements (matching C++ ex26)
    let dim = 2;
    let coarse_mesh = {
        let ne = mesh.n_elements();
        let refs = if ne > 0 {
            ((5000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        } else { 0 };
        let mut m = mesh;
        for _ in 0..refs { m = fem_mesh::refine_uniform(&m); }
        m
    };

    // 4. Build mesh hierarchy (geometric refinements)
    let mut meshes = vec![coarse_mesh];
    let n_geom = args.geometric_refs;
    for _ in 0..n_geom {
        let fine = fem_mesh::refine_uniform(meshes.last().unwrap());
        meshes.push(fine);
    }

    // 5. Build FE space hierarchy: P1 on each mesh
    //    Then order-refine: create P2, P4, … on the finest mesh
    let n_order = args.order_refs;

    // We build: level 0 = finest (highest order on finest mesh)
    //           level n = coarsest (P1 on coarsest mesh)
    // Order: geometric levels (P1 on each mesh) + order levels (P{2^k} on finest mesh)
    //
    // For 1:1 with C++ ex26, the hierarchy is:
    //   - Start with P1 on coarsest mesh
    //   - For each geometric refinement: refine mesh, keep P1
    //   - For each order refinement: keep finest mesh, double order

    // Collect all spaces: coarsest to finest
    let mut spaces: Vec<H1Space<Mesh<2>>> = Vec::new();
    let mut mesh_ords: Vec<u8> = Vec::new();

    // Geometric levels: P1 on each geometrically refined mesh
    for m in &meshes {
        spaces.push(H1Space::new(m.clone(), 1));
        mesh_ords.push(1);
    }

    // Order-refined levels: increasing order on the FINEST mesh
    let finest_mesh_ref = meshes.last().unwrap().clone();
    for k in 1..=n_order {
        let order = 1u8 << k; // 2^k
        spaces.push(H1Space::new(finest_mesh_ref.clone(), order));
        mesh_ords.push(order);
    }

    // The finest space is the last one
    let n_spaces = spaces.len();

    println!("Number of finite element unknowns: {}", spaces.last().unwrap().n_dofs());

    // 6. Set up RHS: linear form (1, phi_i)
    let fine_space = spaces.last().unwrap();
    let n_dofs = fine_space.n_dofs();
    let mut rhs = Assembler::assemble_linear(fine_space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);
    // Zero RHS at BC DOFs (matching MFEM FormLinearSystem for homogeneous Dirichlet BCs)
    let bc_fine = boundary_dofs(fine_space.mesh(), fine_space.dof_manager(), &fine_space.mesh().unique_boundary_tags());
    for &d in &bc_fine { if (d as usize) < n_dofs { rhs[d as usize] = 0.0; } }
    let mut x = vec![0.0; n_dofs];

    // 7. Build hierarchy matrices and prolongation operators
    let mut levels: Vec<GeometricMgLevel> = Vec::new();
    let mut prolong: Vec<CsrMatrix<f64>> = Vec::new();

    let boundary_tags: Vec<i32> = fine_space.mesh().unique_boundary_tags();

    for i in 0..n_spaces {
        let space = &spaces[i];
        let qo = (2 * space.order() + 1).max(3) as u8;
        let mut mat = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let bc = boundary_dofs(space.mesh(), space.dof_manager(), &boundary_tags);
        // Symmetric BC elimination (matching MFEM FormSystemMatrix for homogeneous BCs)
        let mut dummy = vec![0.0; mat.nrows];
        for &d in &bc { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy); }
        levels.push(GeometricMgLevel { mat, bc_dofs: bc });
    }

    // Build prolongation: levels[coarse] → levels[fine] where coarse < fine index
    for i in 0..n_spaces - 1 {
        // spaces[i] is coarser, spaces[i+1] is finer
        let p = build_prolongation(&spaces[i], &spaces[i + 1]);
        prolong.push(p);
    }

    // MG hierarchy: levels[0]=finest, levels[n-1]=coarsest
    // prolong[0] maps from levels[1] (coarser) to levels[0] (finer)
    levels.reverse();
    prolong.reverse();
    // Rebuild prolongation since it's from coarse→fine and we reversed
    // Actually the correct ordering is: prolong[l] maps from level l+1 to level l
    // After reverse: levels[0]=finest, levels[n-1]=coarsest
    // prolong[0] should map from levels[1] to levels[0]
    // So we just reversed the prolongation too, which is correct

    let hierarchy = GeometricMgHierarchy::new(levels, prolong);
    println!("Size of linear system: {}", hierarchy.finest_matrix().nrows);
    println!("  Levels: {}", hierarchy.n_levels());

    // 8. Solve with PCG + geometric multigrid preconditioner
    let mg_config = GeometricMgConfig {
        pre_sweeps: 1, post_sweeps: 1, chebyshev_order: 2,
        jacobi_omega: 0.8, coarse_max_iter: 500, coarse_rtol: 1e-14,
    };
    let mg_precond = GeometricMgPrecond::new(mg_config, &hierarchy);

    // Implement PCG with MG preconditioner manually (matching C++ PCG(*A, M, B, X, ...))
    pcg_with_mg(&hierarchy, &mg_precond, &rhs, &mut x);

    // 9. Compute L2 error (if exact solution known)
    let l2_err = compute_l2_error(&spaces.last().unwrap(), &x);
    println!("  L2 error = {:.6e}", l2_err);
}

// ─── PCG with geometric MG preconditioner ────────────────────────────────────

fn pcg_with_mg(h: &GeometricMgHierarchy, mg: &GeometricMgPrecond,
               b: &[f64], x: &mut [f64]) {
    let a = h.finest_matrix();
    let n = a.nrows;
    let rtol = 1e-12;
    let max_iter = 2000usize;

    // r = b - A*x
    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    a.spmv(x, &mut ax);
    for i in 0..n { r[i] = b[i] - ax[i]; }
    let b_norm = b.iter().map(|v| v*v).sum::<f64>().sqrt().max(1e-30);
    let rhs_norm = b.iter().map(|v| v*v).sum::<f64>().sqrt();

    let mut p = r.clone();  // search direction
    let mut z = vec![0.0; n];  // preconditioned residual
    let mut old_rho = r.iter().map(|v| v*v).sum::<f64>();

    if old_rho.sqrt() / b_norm < rtol { return; }

    for iter in 1..=max_iter {
        // Compute Ap = A * p
        a.spmv(&p, &mut ax);

        // alpha = (r, z) / (p, Ap)
        let pap: f64 = (0..n).map(|i| p[i] * ax[i]).sum();
        if pap.abs() < 1e-30 { break; }
        let alpha = old_rho / pap;

        // x = x + alpha * p
        // r = r - alpha * Ap
        for i in 0..n { x[i] += alpha * p[i]; }
        for i in 0..n { r[i] -= alpha * ax[i]; }

        // Check convergence
        let r_norm = r.iter().map(|v| v*v).sum::<f64>().sqrt();
        println!("   Iteration : {:3}  (B r, r) = {:.5}", iter, r_norm / rhs_norm);

        if r_norm / b_norm < rtol {
            println!("Average reduction factor = {:.5}",
                     (r_norm / rhs_norm).powf(1.0 / iter as f64));
            return;
        }

        // z = M^{-1} * r  (MG V-cycle)
        z.copy_from_slice(&r);
        mg.v_cycle(h, &r, &mut z);

        let new_rho = (0..n).map(|i| r[i] * z[i]).sum::<f64>();
        let beta = new_rho / old_rho;
        old_rho = new_rho;

        // p = z + beta * p
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
    }
    println!("  PCG did not converge in {max_iter} iterations");
}

// ─── Build prolongation between geometrically/order-refined spaces ────────────

fn build_prolongation(coarse: &H1Space<Mesh<2>>, fine: &H1Space<Mesh<2>>) -> CsrMatrix<f64> {
    let n_fine = fine.n_dofs();
    let n_coarse = coarse.n_dofs();
    let mut coo = CooMatrix::new(n_fine, n_coarse);

    let c_order = coarse.order();
    let f_order = fine.order();
    let same_mesh = coarse.mesh().n_elements() == fine.mesh().n_elements();

    if same_mesh && f_order > c_order {
        // p-refinement: same mesh, higher order on fine.
        // Evaluate each coarse basis function at every fine DOF coordinate.
        let mesh: &Mesh<2> = coarse.mesh();
        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let c_ref = ref_elem_for(et, c_order);
            let f_ref = ref_elem_for(et, f_order);
            let c_dofs: Vec<usize> = coarse.element_dofs(e).iter().map(|&d| d as usize).collect();
            let f_dofs: Vec<usize> = fine.element_dofs(e).iter().map(|&d| d as usize).collect();
            let f_coords = f_ref.dof_coords();
            let n_c = c_dofs.len();
            let mut phi = vec![0.0; n_c];

            for (li, &fg) in f_dofs.iter().enumerate() {
                let xi = &f_coords[li];
                c_ref.eval_basis(xi, &mut phi);
                for (ci, &cg) in c_dofs.iter().enumerate() {
                    if phi[ci].abs() > 1e-14 {
                        coo.add(fg, cg, phi[ci]);
                    }
                }
            }
        }
    } else if !same_mesh && f_order == c_order {
        // Geometric refinement: finer mesh, same order.
        // Fine mesh has more elements; each coarse DOF maps to one or more fine DOFs.
        // Geometric refinement: coordinate matching works for same-order spaces
        let dm_c = coarse.dof_manager();
        let dm_f = fine.dof_manager();
        for fi in 0..n_fine {
            let fx = dm_f.dof_coord(fi as u32);
            for ci in 0..n_coarse {
                let cx = dm_c.dof_coord(ci as u32);
                let d2 = (fx[0]-cx[0]).powi(2) + (fx[1]-cx[1]).powi(2);
                if d2 < 1e-20 { coo.add(fi, ci, 1.0); break; }
            }
        }
    } else {
        panic!("build_prolongation: unsupported (c_order={}, f_order={}, same_mesh={})",
               c_order, f_order, same_mesh);
    }
    coo.into_csr()
}


fn ref_elem_for(et: fem_mesh::element_type::ElementType, order: u8) -> Box<dyn fem_element::ReferenceElement> {
    use fem_element::lagrange::*;

    use fem_mesh::element_type::ElementType as ET;
    match (et, order) {
        (ET::Tri3, 1) => Box::new(TriP1),
        (ET::Tri3, 2) => Box::new(TriP2),
        (ET::Quad4, 1) => Box::new(QuadQ1),
        (ET::Quad4, 2) => Box::new(QuadQ2),
        (ET::Quad4, 4) => Box::new(QuadQ4),
        _ => panic!("ref_elem: ({et:?}, order={order})"),
    }
}

// ─── L2 error computation ────────────────────────────────────────────────────

fn compute_l2_error(space: &H1Space<Mesh<2>>, uh: &[f64]) -> f64 {
    use fem_element::{lagrange::TriP2, ReferenceElement};
    let mesh = space.mesh();
    let re = TriP2;
    let qr = re.quadrature(5);
    let mut err2 = 0.0_f64;
    let mut phi = vec![0.0_f64; re.n_dofs()];
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for (qi, xi) in qr.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = qr.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let v: f64 = phi.iter().zip(dofs.iter()).map(|(&v, &d)| v * uh[d]).sum();
            let exact = exact_solution(&xp);
            err2 += w * (v - exact).powi(2);
        }
    }
    err2.sqrt()
}

fn exact_solution(x: &[f64]) -> f64 {
    use std::f64::consts::PI;
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    geometric_refs: usize,
    order_refs: usize,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 10, geometric_refs: 0, order_refs: 2 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|s| s.parse().ok()).unwrap_or(10),
            "-gr" | "--geometric-refinements" => {
                a.geometric_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(0)
            }
            "-or" | "--order-refinements" => {
                a.order_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(2)
            }
            _ => {}
        }
    }
    a
}
