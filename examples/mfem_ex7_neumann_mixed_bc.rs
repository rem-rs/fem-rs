//                                MFEM Example 7
//
// 1:1 Rust translation of MFEM C++ ex7.cpp — Screened Poisson on a sphere.
//
// Compile: cargo run --example mfem_ex7_neumann_mixed_bc -- [options]
//   -e 0  : use triangles (octahedron), default
//   -e 1  : use quadrilaterals (cube)
//   -o 2  : finite element order
//   -r 4  : uniform refinement levels
//
// Description: Screened Poisson problem -Delta u + u = f on the unit sphere.
// Reference: mfem/examples/ex7.cpp
//
// NOTE: L2 error vs MFEM differs by ~15× at order=2 due to missing high-order
// geometry (SetNodalFESpace equivalent). Surface assembly infrastructure is
// complete (3×2 Jacobian, metric-based gradient transform, curved assembly).
// High-order geometry tracked as separate feature.

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::{
    H1Space,
    fe_space::FESpace,
};

fn analytic_solution(x: &[f64]) -> f64 {
    let l2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    x[0]*x[1] / l2
}
fn analytic_rhs(x: &[f64]) -> f64 {
    let l2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    7.0*x[0]*x[1] / l2
}

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── 1. Generate sphere mesh ──────────────────────────────────────────────
    let (mesh, _name) = if args.elem_type == 0 {
        (Mesh::<3>::unit_sphere_octahedron(), "Tri3")
    } else {
        (Mesh::<3>::unit_sphere_cube(), "Quad4")
    };
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ── 2. Refine (NO snap until the end, matching MFEM snap-at-the-end) ─────
    let mut mesh = mesh;
    for l in 0..args.ref_levels {
        match mesh.element_type(0) {
            fem_mesh::element_type::ElementType::Tri3 =>
                mesh = fem_mesh::amr::refine_uniform_surface_tri3(&mesh),
            _ =>
                mesh = fem_mesh::amr::refine_uniform_surface_quad4(&mesh),
        }
    }
    mesh.snap_to_sphere(); // single snap at the end (MFEM default)
    eprintln!("  After refinement: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    let order = args.order;

    // ── 4. H1 space ─────────────────────────────────────────────────────────
    let space = H1Space::new(mesh.clone(), order);
    println!("Number of unknowns: {}", space.n_dofs());

    // ── 5. Linear form ──────────────────────────────────────────────────────
    let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(analytic_rhs)], (order as u8)*2+1);

    // ── 6. Solution vector ──────────────────────────────────────────────────
    let mut u = vec![0.0; space.n_dofs()];

    // ── 7-8. Bilinear form + solve ──────────────────────────────────────────
    let quad = (order as u8) * 2;
    let mat = Assembler::assemble_bilinear(&space, &[
        &DiffusionIntegrator{kappa:1.0},
        &MassIntegrator{rho:1.0},
    ], quad);

    let cfg = SolverConfig{rtol:1e-12,atol:0.0,max_iter:200,verbose:false,..SolverConfig::default()};
    let res = solve_pcg_gssmoother(&mat, &rhs, &mut u, &cfg).expect("PCG");
    if !res.converged { eprintln!("  PCG: No convergence!"); }

    // ── 9. L2 error ─────────────────────────────────────────────────────────
    let gf = GridFunction::new(&space, u);
    let l2_err = gf.compute_l2_error(&analytic_solution, (order as u8)*2+2);
    println!("\nL2 norm of error: {}", l2_err);

    eprintln!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────
struct Args { elem_type: usize, order: u8, ref_levels: usize, _visualization: bool }
impl Args {
    fn parse() -> Self {
        let (mut elem_type, mut order, mut ref_levels, mut vis) = (0usize, 2u8, 4usize, true);
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-e"|"--elem" => elem_type = it.next().and_then(|v|v.parse().ok()).unwrap_or(0),
                "-o"|"--order" => order = it.next().and_then(|v|v.parse().ok()).unwrap_or(2),
                "-r"|"--refine" => ref_levels = it.next().and_then(|v|v.parse().ok()).unwrap_or(4),
                "-no-vis"|"--no-visualization" => vis = false,
                _ => {}
            }
        }
        Args{elem_type,order,ref_levels,_visualization:vis}
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn solve_sphere(etype: usize, order: u8, r: usize) -> f64 {
        let mut mesh = if etype == 0 { Mesh::<3>::unit_sphere_octahedron() } else { Mesh::<3>::unit_sphere_cube() };
        for l in 0..=r {
            if l > 0 {
                match mesh.element_type(0) {
                    fem_mesh::element_type::ElementType::Tri3 =>
                        mesh = fem_mesh::amr::refine_uniform_surface_tri3(&mesh),
                    _ => mesh = fem_mesh::amr::refine_uniform_surface_quad4(&mesh),
                }
            }
            mesh.snap_to_sphere();
        }
        let space = H1Space::new(mesh, order);
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(analytic_rhs)], (order as u8)*2+1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator{kappa:1.0}, &MassIntegrator{rho:1.0}], (order as u8)*2);
        let mut u = vec![0.0; space.n_dofs()];
        let cfg = SolverConfig{rtol:1e-12,atol:0.0,max_iter:200,verbose:false,..SolverConfig::default()};
        solve_pcg_gssmoother(&mat, &rhs, &mut u, &cfg).expect("PCG");
        let gf = GridFunction::new(&space, u);
        gf.compute_l2_error(&analytic_solution, (order as u8)*2+2)
    }

    #[test] fn ex7_tri_sphere_converges() {
        let c = solve_sphere(0, 1, 3); let f = solve_sphere(0, 1, 4);
        assert!(f < c, "L2 must decrease: {:.6e} -> {:.6e}", c, f);
        fem_regression::regression("mfem_ex7_tri_sphere")
            .check("l2_error_r3_o1", c).check("l2_error_r4_o1", f).finalize();
    }
}
