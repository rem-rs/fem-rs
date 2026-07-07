//! # MFEM Example 6 — Flux Recovery (Derivatives of the Solution)
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BC, recovers
//! the flux `σ = ∇u` via area-weighted nodal gradient averaging
//! (Zienkiewicz–Zhu), and writes both the scalar solution and vector flux field
//! to a VTU file.
//!
//! Reference: `mfem/ex6.cpp`  (AMR Poisson with `f = 1`)
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex6_flux_recovery
//! cargo run --example mfem_ex6_flux_recovery -- -m ../data/star.mesh -o 2
//! ```

use std::time::Instant;

use fem_assembly::{
    Assembler,
    postprocess::recover_gradient_nodal,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_element::lagrange::{TriP1, TriP2, TriP3};
use fem_element::ReferenceElement;
use fem_io::{DataArray, VtkWriter};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, topology::MeshTopology, element_type::ElementType};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    println!("=== MFEM Example 6: Flux Recovery ===");

    // ─── 1. Mesh (from file or unit-square) ──────────────────────────────────
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        println!("  Mesh file: {path}");
        read_mfem_file(path)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    } else {
        let r = args.refinements.unwrap_or(3);
        println!("  Unit-square tri mesh, refinements = {r}");
        Mesh::<2>::unit_square_tri(r)
    };
    println!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ─── 2. H¹ space ─────────────────────────────────────────────────────────
    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  H1Space: {n} DOFs, order {}", args.order);

    // ─── 3. Assemble stiffness + RHS  (matching MFEM ex6: f = 1) ──────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let quad = args.order as u8 * 2 + 1;

    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);

    // ─── 4. Homogeneous Dirichlet BC on all boundaries ───────────────────────
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
    println!("  Dirichlet BC on {} DOFs", bnd.len());

    // ─── 5. Solve ────────────────────────────────────────────────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 5_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("PCG solve");
    println!("  Solve: {} iterations, final residual = {:.3e}", res.iterations, res.final_residual);

    let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  ||u||_2 = {u_norm:.6e}");

    // ─── 6. Flux recovery (ZZ nodal gradient averaging) ──────────────────────
    let grad_nodal = recover_gradient_nodal(&space, &u);
    let nv = space.mesh().n_nodes();
    let mut grad_flat = Vec::with_capacity(nv * 2);
    for node in 0..nv {
        grad_flat.push(grad_nodal[0][node]);
        grad_flat.push(grad_nodal[1][node]);
    }

    // ─── 7. Write VTU (interpolate DOF solution to mesh nodes) ───────────────
    let mesh = space.mesh();
    let nv = mesh.n_nodes();
    let mut u_at_nodes = vec![0.0_f64; nv];
    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_tri(elem_type, args.order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs = space.element_dofs(e);
        let nodes = mesh.element_nodes(e);
        for (vi, &node) in nodes.iter().enumerate() {
            let mut xi = vec![0.0_f64; 2];
            if vi < 3 {
                xi = match vi {
                    0 => vec![0.0, 0.0],
                    1 => vec![1.0, 0.0],
                    _ => vec![0.0, 1.0],
                };
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

    let fname = if let Some(ref path) = args.mesh {
        let stem = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed("mesh"));
        format!("ex6_{stem}_p{}.vtu", args.order)
    } else {
        let r = args.refinements.unwrap_or(3);
        format!("ex6_flux_r{r}_p{}.vtu", args.order)
    };

    let mut writer = VtkWriter::new(mesh);
    writer
        .add_point_data(DataArray::scalars("u", u_at_nodes))
        .add_point_data(DataArray::vectors("flux", 2, grad_flat));
    writer.write_file(&fname).expect("write VTU");
    println!("  Output: {fname}");

    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

// ─── Element reference helper ──────────────────────────────────────────────

fn ref_elem_tri(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        _ => panic!("unsupported triangle order {order}"),
    }
}

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    refinements: Option<usize>,
    order: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        refinements: None,
        order: 2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
            "-r" | "--refinements" => {
                a.refinements = it.next().and_then(|s| s.parse().ok());
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2);
            }
            _ => {}
        }
    }
    a
}

// ─── Tests (MMS exact-solution verification) ───────────────────────────────

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use fem_assembly::postprocess::compute_h1_error;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::Assembler;
    use fem_mesh::Mesh;
    use fem_mesh::topology::MeshTopology;
    use fem_solver::{solve_pcg_jacobi, SolverConfig};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    use super::ref_elem_tri;

    /// L² error against an exact function.
    fn l2_error<S: FESpace>(
        space: &S,
        dofs: &[f64],
        exact: impl Fn(&[f64]) -> f64,
    ) -> f64 {
        use fem_element::ReferenceElement;
        let mesh = space.mesh();
        let mut err2 = 0.0;
        for e in mesh.elem_iter() {
            let elem_type = mesh.element_type(e);
            let ref_elem = ref_elem_tri(elem_type, order);
            let n_ldofs = ref_elem.n_dofs();
            let elem_dofs = space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let x0 = mesh.node_coords(nodes[0]);
            let x1 = mesh.node_coords(nodes[1]);
            let x2 = mesh.node_coords(nodes[2]);
            let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                - (x1[1] - x0[1]) * (x2[0] - x0[0]))
            .abs();
            let q = ref_elem.quadrature(order * 3u8);
            let mut basis = vec![0.0; n_ldofs];
            for (qi, xi) in q.points.iter().enumerate() {
                let w = q.weights[qi] * det_j;
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

    /// Manufactured exact solution: u = sin(πx) sin(πy)
    fn exact(x: &[f64]) -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// RHS for the manufactured solution: -Δu = 2π² sin(πx) sin(πy)
    fn rhs_mms(x: &[f64]) -> f64 {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    }

    /// Exact gradient: ∇u
    fn grad_exact(x: &[f64]) -> Vec<f64> {
        vec![
            PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
            PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
        ]
    }

    fn solve_mms(n: usize, order: u8) -> (Vec<f64>, H1Space<Mesh<2>>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, order);
        let ndofs = space.n_dofs();
        let quad = order as u8 * 2 + 1;

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let src = DomainSourceIntegrator::new(rhs_mms);
        let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad);
        let mut rhs = Assembler::assemble_linear(&space, &[&src], quad);

        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals: Vec<f64> = bnd.iter().map(|&dof| {
            let x = dm.dof_coord(dof);
            exact(&x)
        }).collect();
        apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

        let mut u = vec![0.0_f64; ndofs];
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 5_000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("PCG solve");
        (u, space)
    }

    #[test]
    fn ex6_mms_l2_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let err_c = l2_error(&sp_c, &u_c, exact);
        let err_f = l2_error(&sp_f, &u_f, exact);
        eprintln!("  L2 coarse={:.6e} fine={:.6e}", err_c, err_f);
        assert!(err_f < err_c, "L2 error must decrease on refinement");
        let rate = (err_f / err_c).ln() / (32.0_f64 / 16.0_f64).ln();
        assert!(rate < -1.8, "L2 convergence rate {:.2} too slow", rate);
    }

    #[test]
    fn ex6_mms_h1_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let h1_c = compute_h1_error(&sp_c, &u_c, grad_exact, 6);
        let h1_f = compute_h1_error(&sp_f, &u_f, grad_exact, 6);
        eprintln!("  H1 coarse={:.6e} fine={:.6e}", h1_c, h1_f);
        assert!(h1_f < h1_c, "H1 error must decrease on refinement");
    }

    #[test]
    fn ex6_mms_flux_recovery_error_small() {
        let (u, space) = solve_mms(32, 2);
        let grad_nodal =
            fem_assembly::postprocess::recover_gradient_nodal(&space, &u);
        let nv = space.mesh().n_nodes();
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
        eprintln!("  Flux error (nodal RMS) = {flux_err:.6e}");
        assert!(flux_err < 1e-2, "flux recovery error too large: {flux_err:.6e}");
    }
}
