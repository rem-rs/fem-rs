//! # Example 28 — Sliding Elasticity (analogous to MFEM ex28)
//!
//! Linear elasticity on a trapezoid with sliding (normal-constraint) BC:
//!
//! ```text
//!   −∇·σ(u) = 0         in Ω
//!    u = 0               on boundary 1 (bottom, fixed)
//!    u_y = 0             on boundary 4 (left, sliding)
//!    σ·n = (f_x, 0)      on boundary 2 (right, push force)
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex28_sliding_elasticity
//! cargo run --example mfem_ex28_sliding_elasticity -- --offset 0.3 --order 2
//! ```

use fem_assembly::{
    Assembler,
    standard::{ElasticityIntegrator, NeumannIntegrator},
};
use fem_mesh::{
    MeshTopology, SimplexMesh,
    element_type::ElementType,
    refine_uniform,
};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    VectorH1Space, H1Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

fn main() {
    let args = parse_args();
    println!("=== Example 28: Sliding Elasticity (MFEM ex28) ===");
    println!("  Offset: {:.3}, Order: {}, Refinements: {}",
             args.offset, args.order, args.ref_levels);

    // Build trapezoid mesh
    let mesh = build_trapezoid_mesh(args.offset);

    // Refine uniformly
    let mut mesh = mesh;
    for _ in 0..args.ref_levels {
        mesh = refine_uniform(&mesh);
    }

    let scalar_mesh = mesh.clone();
    let space = VectorH1Space::new(mesh, args.order, 2);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();

    println!("  Mesh: {} nodes, {} elements",
             space.mesh().n_nodes(), space.mesh().n_elems());
    println!("  DOFs: {n_dofs}");

    // Lamé parameters
    let lam = 0.5;
    let mu_val = 0.5;

    let elast = ElasticityIntegrator {
        lambda: lam,
        mu: mu_val,
        plane_stress: false,
    };
    let mut mat = Assembler::assemble_bilinear(&space, &[&elast], args.order as u8 * 2 + 1);

    // Push force on boundary attribute 2 (right): f_x = -5.0e-2
    let mut rhs = vec![0.0_f64; n_dofs];
    {
        let scalar_space = H1Space::new(scalar_mesh, args.order);
        let n_sc = scalar_space.n_dofs();
        let face_dofs = |f: u32| -> Vec<u32> {
            let nodes = scalar_space.mesh().face_nodes(f);
            nodes.iter().copied().collect()
        };
        let push = NeumannIntegrator::new(move |_x: &[f64], _n: &[f64]| -5.0e-2);
        let f_rhs = Assembler::assemble_boundary_linear(
            n_sc, scalar_space.mesh(), &face_dofs, args.order,
            &[&push], &[2], args.order as u8 * 2 + 1,
        );
        for (i, &v) in f_rhs.iter().enumerate() {
            rhs[i] += v; // x-component
        }
    }

    // BCs: fix all components on boundary 1 (bottom);
    //       fix y-component only on boundary 4 (left, sliding-like normal constraint).
    let scalar_dm = space.scalar_dof_manager();
    let bnd1 = boundary_dofs(space.mesh(), scalar_dm, &[1]);
    let bnd4 = boundary_dofs(space.mesh(), scalar_dm, &[4]);

    let mut clamped: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for &d in &bnd1 {
        clamped.push(d); vals.push(0.0);
        clamped.push(d + n_scalar as u32); vals.push(0.0);
    }
    for &d in &bnd4 {
        clamped.push(d + n_scalar as u32); vals.push(0.0);
    }
    apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

    // Solve
    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig {
        rtol: 1e-8, atol: 0.0, max_iter: 10_000, verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("elasticity solve failed");

    let ux = &u[..n_scalar];
    let uy = &u[n_scalar..];
    let ux_max = ux.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
    let uy_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.abs().max(b.abs()));

    println!("  Solve: {} iters, residual={:.3e}, converged={}",
             res.iterations, res.final_residual, res.converged);
    println!("  max|u_x| = {:.4e}, max|u_y| = {:.4e}", ux_max, uy_max);
    println!("Done.");
}

fn build_trapezoid_mesh(offset: f64) -> SimplexMesh<2> {
    assert!(offset < 0.9, "offset too large");
    // Triangle mesh for trapezoid: split quad (0,1,3,2) into two triangles
    // Vertices: 0=(0,0), 1=(1,0), 2=(offset,1), 3=(1,1)
    let coords = vec![
        0.0, 0.0,
        1.0, 0.0,
        offset, 1.0,
        1.0, 1.0,
    ];
    let conn = vec![
        0u32, 1, 3,
        0u32, 3, 2,
    ];
    let elem_tags = vec![1, 1];
    let face_conn = vec![
        0u32, 1, // bottom, attr 1
        1u32, 3, // right,  attr 2
        2u32, 3, // top,    attr 3
        0u32, 2, // left,   attr 4
    ];
    let face_tags = vec![1, 2, 3, 4];
    SimplexMesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        face_conn, face_tags, ElementType::Line2,
    )
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    offset: f64,
    order: u8,
    ref_levels: usize,
}

fn parse_args() -> Args {
    let mut a = Args { offset: 0.3, order: 1, ref_levels: 3 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--offset" => a.offset = it.next().unwrap_or("0.3".into()).parse().unwrap_or(0.3),
            "-o" | "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-r" | "--refine" | "--ref-levels" => {
                a.ref_levels = it.next().unwrap_or("3".into()).parse().unwrap_or(3)
            }
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
    fn ex28_sliding_elasticity_converges() {
        let mesh = build_trapezoid_mesh(0.3);
        let mesh = refine_uniform(&refine_uniform(&mesh));
        let space = VectorH1Space::new(mesh, 1, 2);
        let n_dofs = space.n_dofs();
        let n_sc = space.n_scalar_dofs();

        let elast = ElasticityIntegrator { lambda: 0.5, mu: 0.5, plane_stress: false };
        let mut mat = Assembler::assemble_bilinear(&space, &[&elast], 3);
        let mut rhs = vec![0.0_f64; n_dofs];

        let scalar_dm = space.scalar_dof_manager();
        let bnd1 = boundary_dofs(space.mesh(), scalar_dm, &[1]);
        let bnd4 = boundary_dofs(space.mesh(), scalar_dm, &[4]);
        let mut clamped: Vec<u32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for &d in &bnd1 {
            clamped.push(d); vals.push(0.0);
            clamped.push(d + n_sc as u32); vals.push(0.0);
        }
        for &d in &bnd4 {
            clamped.push(d + n_sc as u32); vals.push(0.0);
        }
        apply_dirichlet(&mut mat, &mut rhs, &clamped, &vals);

        let mut u = vec![0.0_f64; n_dofs];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10000, verbose: false, ..Default::default() };
        let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).unwrap();
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-6);
    }
}
