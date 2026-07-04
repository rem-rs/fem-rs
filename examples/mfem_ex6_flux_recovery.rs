//! # MFEM Example 6 — Flux Recovery (Derivatives of the Solution)
//!
//! Solves the Poisson problem `−Δu = f` with Dirichlet BC, recovers the flux
//! `σ = ∇u` via area-weighted nodal gradient averaging (Zienkiewicz–Zhu), and
//! writes both the scalar solution and vector flux field to a VTU file.
//!
//! Reference: `mfem/ex6.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex6_flux_recovery [refinements=3] [order=2]
//! ```

use std::f64::consts::PI;
use std::time::Instant;

use fem_assembly::{
    Assembler,
    postprocess::{compute_h1_error, recover_gradient_nodal},
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_element::lagrange::{TriP1, TriP2, TriP3};
use fem_element::ReferenceElement;
use fem_io::{DataArray, VtkWriter};
use fem_mesh::{SimplexMesh, topology::MeshTopology, element_type::ElementType};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let refinements: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let order: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    println!("=== MFEM Example 6: Flux Recovery ===");
    println!("  refinements = {refinements}, order = {order}");
    let t0 = Instant::now();

    // ─── 1. Mesh ─────────────────────────────────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(refinements);
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ─── 2. H¹ space ─────────────────────────────────────────────────────────
    let space = H1Space::new(mesh, order);
    let n = space.n_dofs();
    println!("  H1Space: {n} DOFs, order {order}");

    // ─── 3. Assemble stiffness + RHS ─────────────────────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let quad = order * 2 + 1;

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

    // ─── 4. Dirichlet BC ─────────────────────────────────────────────────────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_vals: Vec<f64> = bnd.iter().map(|&dof| {
        let x = dm.dof_coord(dof);
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }).collect();
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
    println!("  Dirichlet BC on {} DOFs", bnd.len());

    // ─── 5. Solve ────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("PCG solve");
    println!("  Solve: {} iterations, final residual = {:.3e}", res.iterations, res.final_residual);

    // ─── 6. Error norms ──────────────────────────────────────────────────────
    let exact = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let grad_exact = |x: &[f64]| vec![
        PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
        PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
    ];

    let l2 = l2_error(&space, &u, exact);
    println!("  ‖u − u_h‖_L²   = {l2:.6e}");

    let h1s = compute_h1_error(&space, &u, grad_exact, order * 3);
    println!("  |u − u_h|_H¹  = {h1s:.6e}");

    // ─── 7. Flux recovery (ZZ nodal gradient averaging) ──────────────────────
    let grad_nodal = recover_gradient_nodal(&space, &u);
    let nv = space.mesh().n_nodes();
    let mut grad_flat = Vec::with_capacity(nv * 2);
    for node in 0..nv {
        grad_flat.push(grad_nodal[0][node]);
        grad_flat.push(grad_nodal[1][node]);
    }

    // L² error of the recovered flux (nodal RMS).
    let flux_err: f64 = (0..nv)
        .map(|node| {
            let x = space.mesh().node_coords(node as u32);
            let ex = PI * (PI * x[0]).cos() * (PI * x[1]).sin();
            let ey = PI * (PI * x[0]).sin() * (PI * x[1]).cos();
            let dx = grad_nodal[0][node] - ex;
            let dy = grad_nodal[1][node] - ey;
            dx * dx + dy * dy
        })
        .sum::<f64>()
        .sqrt()
        / nv as f64;
    println!("  ‖σ − σ_h‖_L²/n = {flux_err:.6e}");

    // ─── 8. Write VTU (interpolate DOF solution to mesh nodes) ──────────────
    let mesh = space.mesh();
    let nv = mesh.n_nodes();
    let mut u_at_nodes = vec![0.0_f64; nv];
    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_tri(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);
        // Interpolate DOFs onto vertex nodes of each element.
        for (vi, &node) in nodes.iter().enumerate() {
            let mut xi = vec![0.0_f64; 2];
            if vi < 3 {
                // Reference triangle vertices: (0,0), (1,0), (0,1)
                xi = match vi { 0 => vec![0.0, 0.0], 1 => vec![1.0, 0.0], _ => vec![0.0, 1.0] };
            }
            let mut basis = vec![0.0; n_ldofs];
            ref_elem.eval_basis(&xi, &mut basis);
            let mut val = 0.0;
            for i in 0..n_ldofs {
                val += basis[i] * u[elem_dofs[i] as usize];
            }
            u_at_nodes[node as usize] = val;
        }
    }

    let fname = format!("ex6_flux_r{refinements}_p{order}.vtu");
    let mut writer = VtkWriter::new(mesh);
    writer
        .add_point_data(DataArray::scalars("u", u_at_nodes))
        .add_point_data(DataArray::vectors("flux", 2, grad_flat));
    writer.write_file(&fname).expect("write VTU");
    println!("  Output: {fname}");

    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

// ─── L² error helper ─────────────────────────────────────────────────────────

fn ref_elem_tri(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        _ => panic!("unsupported triangle order {order}"),
    }
}

fn l2_error<S: FESpace>(
    space: &S,
    dofs: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    let mesh = space.mesh();
    let _dim = mesh.dim() as usize;
    let order = space.order();
    let mut err2 = 0.0;

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_tri(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);

        // Jacobian for triangle
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                   - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs();

        let q = ref_elem.quadrature(order * 3);
        let mut basis = vec![0.0; n_ldofs];

        for (qi, xi) in q.points.iter().enumerate() {
            let w = q.weights[qi] * det_j;

            // Physical point: x = x0 + J * xi
            let x_phys = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];

            ref_elem.eval_basis(xi, &mut basis);
            let mut uh = 0.0;
            for i in 0..n_ldofs {
                uh += basis[i] * dofs[elem_dofs[i] as usize];
            }
            let ue = exact(&x_phys);
            err2 += w * (uh - ue).powi(2);
        }
    }
    err2.sqrt()
}
