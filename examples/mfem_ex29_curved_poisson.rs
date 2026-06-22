//! # Example 29 — Curved-surface Poisson (elevated geometry + anisotropic diffusion)
//!
//! Solves the Poisson problem on a high-order curved domain using isoparametric
//! geometry. The mesh is elevated and mapped onto a curved shape, demonstrating:
//!
//! - High-order `CurvedMesh` with geometric elevation
//! - Anisotropic (tensor) diffusion coefficient
//! - Curved element assembly through the isoparametric path
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex29_curved_poisson --release
//! cargo run --example mfem_ex29_curved_poisson -- --order 3 --geom-order 3
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, BilinearIntegrator,
    standard::DomainSourceIntegrator,
};
use fem_mesh::{CurvedMesh, SimplexMesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

/// Anisotropic 2×2 diffusion matrix σ(x,y):
/// σ = [[1+0.5x,  0.3xy],  [0.3xy,  1+0.5y]]
struct AnisotropicDiffusion;

impl BilinearIntegrator for AnisotropicDiffusion {
    fn add_to_element_matrix(&self, qp: &fem_assembly::QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let w = qp.weight;
        let dim = qp.dim;
        let x = qp.x_phys;
        let grad = qp.grad;
        let s00 = 1.0 + 0.5 * x[0];
        let s01 = 0.3 * x[0] * x[1];
        let s11 = 1.0 + 0.5 * x[1];
        for i in 0..n {
            for j in 0..n {
                let val = (grad[i * dim] * (s00 * grad[j * dim] + s01 * grad[j * dim + 1])
                         + grad[i * dim + 1] * (s01 * grad[j * dim] + s11 * grad[j * dim + 1]))
                         * w;
                k_elem[i * n + j] += val;
            }
        }
    }
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 29: Curved-surface Poisson ===");

    // 1. Base linear mesh
    let mesh_lin = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Base: {}×{} P1, {} elements", args.n, args.n, mesh_lin.n_elems());

    // 2. Elevate to high-order geometry
    let curved = CurvedMesh::elevate_to_order(&mesh_lin, args.geom_order, |p| {
        let x = p[0];
        let y = p[1];
        [x + 0.08 * (2.0 * PI * x).sin() * (2.0 * PI * y).sin(),
         y + 0.08 * (2.0 * PI * x).cos() * (2.0 * PI * y).cos()]
    });
    println!("  Curved: geom_order={}, geom_nodes={}", args.geom_order, curved.n_nodes);

    // 3. H1 space on curved mesh.  CurvedMesh<2> implements MeshTopology,
    //    so H1Space::new(curved, order) works, and the assembly code
    //    detects geom_order > 1 → uses the isoparametric Jacobian path.
    //    (The WARNING below is expected — integration will use high-order
    //    geometry through ElementTransformation.)
    let space = H1Space::new(curved, args.order);
    let n_dofs = space.n_dofs();
    println!("  H1(P{}): {} DOFs", args.order, n_dofs);

    // 4. Assemble bilinear form (the integrator is passed a QpData struct
    //    that already contains the physical-space gradient, correctly
    //    transformed through the curved isoparametric mapping).
    let quad = (args.order * 2 + 1).max(3);
    let mat = if args.anisotropic {
        Assembler::assemble_bilinear(&space, &[&AnisotropicDiffusion], quad)
    } else {
        struct Isotropic;
        impl BilinearIntegrator for Isotropic {
            fn add_to_element_matrix(&self, qp: &fem_assembly::QpData<'_>, k_elem: &mut [f64]) {
                let n = qp.n_dofs;
                let w = qp.weight;
                let dim = qp.dim;
                let grad = qp.grad;
                for i in 0..n {
                    for j in 0..n {
                        let val = (grad[i*dim] * grad[j*dim] + grad[i*dim+1]*grad[j*dim+1]) * w;
                        k_elem[i * n + j] += val;
                    }
                }
            }
        }
        Assembler::assemble_bilinear(&space, &[&Isotropic], quad)
    };

    // 5. RHS
    let src = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], quad);

    // 6. Dirichlet BCs
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let vals = vec![0.0; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    // 7. Solve
    let mut u = vec![0.0; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5_000, verbose: false, ..SolverConfig::default() };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
    println!("  Solve: {} iters, residual={:.3e}, converged={}",
        res.iterations, res.final_residual, res.converged);

    println!("  ||u||₂ = {:.4e}", u.iter().map(|v| v*v).sum::<f64>().sqrt());
    println!("Done.");

    // This example demonstrates that fem-rs correctly integrates on
    // high-order curved geometries, even with user-defined anisotropic
    // coefficient tensors — the framework handles the isoparametric
    // Jacobian, the covariant Piola transform for gradients, and the
    // quadrature on the physical element automatically.
}

struct Args { n: usize, order: u8, geom_order: usize, anisotropic: bool }

fn parse_args() -> Args {
    let mut a = Args { n: 10, order: 2, geom_order: 3, anisotropic: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(10); }
            "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2); }
            "--geom-order" => { a.geom_order = it.next().and_then(|v| v.parse().ok()).unwrap_or(3); }
            "--aniso" => { a.anisotropic = true; }
            "--iso" => { a.anisotropic = false; }
            _ => {}
        }
    }
    a
}
