//! Example 4 — H(div) Diffusion / Darcy (1:1 with MFEM ex4)
//!
//! Solves the second-order definite H(div) problem:
//!   -∇(α ∇·F) + β F = f    in Ω
//!              F·n = 0     on ∂Ω
//!
//! with a manufactured source derived from the exact solution
//! `F = (cos(κx)sin(κy), cos(κy)sin(κx))` where `κ = π·freq`.
//! Discretisation uses Raviart-Thomas H(div) elements.

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{GradDivIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::{solve_pcg, GSSmoother, SolverConfig};
use fem_space::{
    HDivSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs_hdiv, form_linear_system},
};

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("../data/star.mesh"));
    println!("   --order {}", args.order);
    if args.set_bc {
        println!("   --impose-bc");
    } else {
        println!("   --dont-impose-bc");
    }
    println!("   --frequency {}", args.freq);
    println!("   --no-static-condensation");
    println!("   --no-hybridization");
    println!("   --no-partial-assembly");
    println!("   --no-element-assembly");
    println!("   --device cpu");
    println!("   --no-visualization");

    // 2. Device setup — skipped (no GPU backend).
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    // 3. Read the mesh from the given mesh file.
    //    MFEM ex4: `Mesh *mesh = new Mesh(mesh_file, 1, 1);` — refine=1 on load.
    let mesh_path = args.mesh.as_deref().unwrap_or("../data/star.mesh");
    let mfem = read_mfem_file(mesh_path).expect("failed to read MFEM mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("MFEM mesh must be 2D");
    let dim = 2;

    // 4. Uniform refinement: choose levels so the final mesh has ≤ 25 000 elements.
    let ref_levels =
        ((25000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels {
            m = refine_uniform(&m);
        }
        m
    } else {
        mesh
    };

    // 5. H(div) Raviart-Thomas finite element space of order (args.order - 1).
    //    MFEM's RT_FECollection(order-1, dim) → RT0 for order=1, RT1 for order=2.
    let rt_order = if args.order >= 1 { args.order - 1 } else { 0 };

    let space = HDivSpace::new(mesh, rt_order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    // 6. Essential (Dirichlet) boundary DOFs — all external boundaries.
    //    BC: F·n = <projected exact normal component> (or 0 when -no-bc).
    let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_bdr = if args.set_bc && !all_tags.is_empty() {
        boundary_dofs_hdiv(space.mesh(), &space, &all_tags)
    } else {
        vec![]
    };
    let kappa = args.freq * PI;

    // 7. Right-hand side: b(v) = ∫ f·v dx  where
    //    f = (1+2κ²)(cos(κx)sin(κy), cos(κy)sin(κx)).
    //    MFEM: VectorFEDomainLFIntegrator default order = 2*order for RT.
    let source = MaxwellHSource { kappa };
    let quad_order = if args.order > 0 { 2 * args.order as u8 } else { 2 };
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

    // 8. Solution vector x — zero initial guess (will be set by Dirichlet below).

    // 9. Stiffness matrix: a(u, v) = ∫ α (∇·u)(∇·v) + β u·v dx.
    //    MFEM: DivDivIntegrator order = max(2*order-2, 0), VectorFEMassIntegrator order = 2*order.
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    let mut mat = VectorAssembler::assemble_bilinear(
        &space, &[&grad_div, &vec_mass], quad_order,
    );

    // 10. Form the linear system.
    // MFEM: x.ProjectCoefficient(F); a.FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
    // Standard path (no hybridization, no static condensation).
    let mut x = vec![0.0_f64; n_dofs];
    if !ess_bdr.is_empty() {
        let x_exact = space.interpolate_vector(&|p| {
            let k = kappa;
            vec![(k * p[0]).cos() * (k * p[1]).sin(),
                 (k * p[1]).cos() * (k * p[0]).sin()]
        });
        for &d in &ess_bdr {
            x[d as usize] = x_exact[d as usize];
        }
        let bv: Vec<f64> = ess_bdr.iter().map(|&d| x_exact[d as usize]).collect();
        form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bv);
    }
    let n_sys = n_dofs;
    println!("Size of linear system: {n_sys}");

    // 11. Solve: PCG with symmetric Gauss-Seidel preconditioner.
    // MFEM ex4: PCG(*A, M, B, X, 1, 10000, 1e-20, 0.0)
    // print_level=1, max_iter=10000, rtol=sqrt(1e-20)=1e-10, atol=0.0
    let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&mat);
    let precond = GSSmoother::from_csr(&linlvo_mat).expect("SSOR setup failed");
    let cfg = SolverConfig {
        rtol: 1e-10,
        max_iter: 10000,
        verbose: true,
        ..SolverConfig::default()
    };
    let _result = solve_pcg(&mat, &rhs, &mut x, &precond, 1e-10, 10000, true)
        .expect("solver failed");

    // 13. Compute and print the L² norm of the error.
    // MFEM: x.ComputeL2Error(F) — uses contravariant Piola for H(div).
    let l2_err = compute_hdiv_l2_error(&space, &x, &|xi| {
        let k = kappa;
        [(k * xi[0]).cos() * (k * xi[1]).sin(),
         (k * xi[1]).cos() * (k * xi[0]).sin()]
    });
    println!("\n|| F_h - F ||_{{L^2}} = {}", fem_solver::fmt_g(l2_err));

    // 14. Save the refined mesh and solution (matches MFEM ex4 output files).
    //     MFEM: ofstream precision(8); mesh->Print(mesh_ofs); x.Save(sol_ofs);
    //     → FiniteElementCollection: RT_2D_P0 (RT_FECollection(order-1)).
    {
        write_mfem_file("refined.mesh", space.mesh()).expect("mesh write failed");
        write_mfem_gf_file("sol.gf", dim, &x, "RT", args.order.saturating_sub(1), 1, 8)
            .expect("sol write failed");
    }
}

// ─── Source term (VectorLinearIntegrator) ────────────────────────────────────
//
//   f = (1 + 2κ²) · (cos(κx)sin(κy), cos(κy)sin(κx))

struct MaxwellHSource {
    kappa: f64,
}

impl VectorLinearIntegrator for MaxwellHSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let k = self.kappa;
        let c = 1.0 + 2.0 * k * k;
        let fx = c * (k * x[0]).cos() * (k * x[1]).sin();
        let fy = c * (k * x[1]).cos() * (k * x[0]).sin();
        for i in 0..qp.n_dofs {
            f[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
        }
    }
}

// ─── Exact solution ──────────────────────────────────────────────────────────
//
//   F = (cos(κx)sin(κy), cos(κy)sin(κx))

fn exact_f(x: &[f64], kappa: f64) -> [f64; 2] {
    let k = kappa;
    [(k * x[0]).cos() * (k * x[1]).sin(),
     (k * x[1]).cos() * (k * x[0]).sin()]
}

// ─── H(div) L² error (contravariant Piola) ──────────────────────────────────

fn compute_hdiv_l2_error<F>(space: &HDivSpace<Mesh<2>>, u: &[f64], ex: &F) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    use fem_element::raviart_thomas::TriRTk;
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::ElementTransformation;

    let mut e2 = 0.0;
    let ref_elem = TriRTk::new(0);
    let n_ldofs = ref_elem.n_dofs();
    let q = ref_elem.quadrature(6);
    let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

    for e in space.mesh().elem_iter() {
        let nodes = space.mesh().elem_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let tr = ElementTransformation::from_simplex_nodes(space.mesh(), nodes);
        let jac = tr.jacobian();
        let det_j = tr.det_j();
        let inv_det = 1.0 / det_j;

        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let w = q.weights[qi] * det_j.abs();

            // Contravariant Piola: φ_phys = (1/det(J)) · J · φ_ref
            let mut fh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) * inv_det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) * inv_det;
                fh[0] += u[dofs[i]] * px;
                fh[1] += u[dofs[i]] * py;
            }

            let xp = [
                tr.map_to_physical(xi)[0],
                tr.map_to_physical(xi)[1],
            ];
            let exact = ex(&xp);
            e2 += w * ((fh[0] - exact[0]).powi(2) + (fh[1] - exact[1]).powi(2));
        }
    }
    e2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    order: u8,
    set_bc: bool,
    freq: f64,
    static_cond: bool,
    hybridization: bool,
    pa: bool,
    ea: bool,
    device: Option<String>,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        order: 1,
        set_bc: true,
        freq: 1.0,
        static_cond: false,
        hybridization: false,
        pa: false,
        ea: false,
        device: None,
        visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-bc" | "--impose-bc" => { a.set_bc = true; }
            "-no-bc" | "--dont-impose-bc" => { a.set_bc = false; }
            "-f" | "--frequency" => { a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "-sc" | "--static-condensation" => { a.static_cond = true; }
            "-no-sc" | "--no-static-condensation" => { a.static_cond = false; }
            "-hb" | "--hybridization" => { a.hybridization = true; }
            "-no-hb" | "--no-hybridization" => { a.hybridization = false; }
            "-pa" | "--partial-assembly" => { a.pa = true; }
            "-no-pa" | "--no-partial-assembly" => { a.pa = false; }
            "-ea" | "--element-assembly" => { a.ea = true; }
            "-no-ea" | "--no-element-assembly" => { a.ea = false; }
            "-d" | "--device" => { a.device = it.next(); }
            "-vis" | "--visualization" => { a.visualization = true; }
            "-no-vis" | "--no-visualization" => { a.visualization = false; }
            _ => {}
        }
    }
    a
}
