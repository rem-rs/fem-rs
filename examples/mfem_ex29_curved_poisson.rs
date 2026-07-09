//! # Example 29 — Curved-surface Poisson (analogous to MFEM ex29)
//!
//! Solves the Poisson problem on a 2-D curved surface (elevated geometry)
//! with anisotropic diffusion:
//!
//! ```text
//!   −∇·(σ ∇u) = 1    in Ω
//!          u = 0     on ∂Ω
//! ```
//!
//! where σ is a user-defined 2×2 anisotropic tensor.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex29_curved_poisson
//! cargo run --example mfem_ex29_curved_poisson -- -m ../data/star.mesh --order 3
//! ```

use fem_assembly::{
    Assembler, BilinearIntegrator,
    standard::DomainSourceIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{CurvedMesh, Mesh};
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

/// Anisotropic 2×2 diffusion matrix σ(x,y):
/// σ₁₁ = 1 + 0.5x,  σ₁₂ = 0.3xy,  σ₂₂ = 1 + 0.5y
struct AnisotropicDiffusion;

impl BilinearIntegrator for AnisotropicDiffusion {
    fn add_to_element_matrix(&self, qp: &fem_assembly::QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let w = qp.weight;
        let dim = qp.dim;
        let x = qp.x_phys;
        let grad = qp.grad_phys;
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
    println!("=== Example 29: Curved-surface Poisson (MFEM ex29) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    } else {
        println!("  Mesh: {}×{}", args.n, args.n);
    }
    println!(
        "  Order: P{}, geom_order: {}, anisotropic: {}",
        args.order, args.geom_order, args.anisotropic
    );

    // Load or generate base mesh
    let mesh_lin: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    // Elevate to high-order geometry with sinusoidal deformation
    let curved = CurvedMesh::elevate_to_order(&mesh_lin, args.geom_order, |p| {
        let x = p[0];
        let y = p[1];
        [
            x + 0.08 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin(),
            y + 0.08 * (2.0 * std::f64::consts::PI * x).cos() * (2.0 * std::f64::consts::PI * y).cos(),
        ]
    });
    println!("  Curved: geom_order={}, geom_nodes={}", args.geom_order, curved.n_nodes);

    let space = H1Space::new(curved, args.order);
    let n_dofs = space.n_dofs();
    println!("  DOFs: {n_dofs}");

    // Assemble bilinear form
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
                let grad = qp.grad_phys;
                for i in 0..n {
                    for j in 0..n {
                        let val = (grad[i * dim] * grad[j * dim]
                            + grad[i * dim + 1] * grad[j * dim + 1])
                            * w;
                        k_elem[i * n + j] += val;
                    }
                }
            }
        }
        Assembler::assemble_bilinear(&space, &[&Isotropic], quad)
    };

    // RHS: f = 1 (matching MFEM ex29)
    let src = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], quad);

    // Homogeneous Dirichlet BC on all boundaries
    let dm = space.dof_manager();
    let mesh_ref = space.mesh();
    let all_tags: Vec<i32> = (0..mesh_ref.n_boundary_faces() as u32)
        .map(|f| mesh_ref.face_tag(f))
        .collect::<std::collections::HashSet<i32>>()
        .into_iter().collect();
    let bnd = boundary_dofs(space.mesh(), dm, &all_tags);
    let vals = vec![0.0; bnd.len()];
    let mut mat = mat;
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    // Solve
    let mut u = vec![0.0; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("solver failed");
    println!(
        "  Solve: {} iters, residual={:.3e}, converged={}",
        res.iterations, res.final_residual, res.converged
    );
    println!("  ‖u‖₂ = {:.6e}", u.iter().map(|v| v * v).sum::<f64>().sqrt());
    println!("Done.");
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    geom_order: usize,
    anisotropic: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 10,
        order: 2,
        geom_order: 3,
        anisotropic: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().unwrap_or("10".into()).parse().unwrap_or(10),
            "-o" | "--order" => a.order = it.next().unwrap_or("2".into()).parse().unwrap_or(2),
            "--geom-order" => a.geom_order = it.next().unwrap_or("3".into()).parse().unwrap_or(3),
            "--aniso" => a.anisotropic = true,
            "--iso" => a.anisotropic = false,
            _ => {}
        }
    }
    a
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex29_curved_poisson_converges() {
        let mesh_lin = Mesh::<2>::unit_square_tri(6);
        let curved = CurvedMesh::elevate_to_order(&mesh_lin, 2, |p| {
            let x = p[0];
            let y = p[1];
            [
                x + 0.05 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin(),
                y + 0.05 * (2.0 * std::f64::consts::PI * x).cos() * (2.0 * std::f64::consts::PI * y).cos(),
            ]
        });
        let space = H1Space::new(curved, 1);
        let n_dofs = space.n_dofs();
        let mat = Assembler::assemble_bilinear(&space, &[&AnisotropicDiffusion], 3);
        let src = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

        let dm = space.dof_manager();
        let mesh_ref = space.mesh();
        let all_tags: Vec<i32> = (0..mesh_ref.n_boundary_faces() as u32)
            .map(|f| mesh_ref.face_tag(f))
            .collect::<std::collections::HashSet<i32>>()
            .into_iter().collect();
        let bnd = boundary_dofs(space.mesh(), dm, &all_tags);
        let vals = vec![0.0; bnd.len()];
        let mut mat = mat;
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

        let mut u = vec![0.0; n_dofs];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).unwrap();
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-8);
    }
}
