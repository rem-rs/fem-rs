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
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_pcg, GSSmoother};
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
    // `solve_pcg` mirrors MFEM's legacy `PCG()` helper, which takes the RAW
    // RTOLERANCE literal (1e-20) and itself applies `SetRelTol(sqrt(1e-20))`;
    // its criterion is `(B r, r) <= 1e-20 · (B r0, r0)`.  Passing the
    // pre-sqrt'ed 1e-10 here loosened the stop by 10 orders of magnitude
    // (D634: 287-iteration premature "convergence", ‖F−F_h‖ 27× C++).
    let linlvo_mat = fem_linalg::fem_to_linlvo_csr(&mat);
    let precond = GSSmoother::from_csr(&linlvo_mat).expect("SSOR setup failed");
    let _result = solve_pcg(&mat, &rhs, &mut x, &precond, 1e-20, 10000, true)
        .expect("solver failed");

    // 13. Compute and print the L² norm of the error.
    // MFEM: x.ComputeL2Error(F) — the vector overload that evaluates the field
    // through `GetVectorValues` → `CalcVShape_RT`, i.e. the contravariant
    // Piola map, with the default rule `IntRules.Get(geom, 2·GetOrder()+3)`
    // (GetOrder() = p+1 for RT).  D639: this used to be an example-local
    // reconstruction hardcoded to `TriRTk::new(0)` + triangle quadrature +
    // the first 3 element DOFs — garbage on star.mesh, which is a *quad*
    // mesh (20 squares; RT0-quad = 4 edge DOFs/element, bilinear geometry);
    // the printed value was therefore solution-insensitive (0.432497 vs
    // 0.432509 for two different solutions) and 27× the C++ truth.  The
    // core routine picks the per-element reference element/geometry exactly
    // like the assembler, and its default quadrature order implements
    // MFEM's 2·(p+1)+3 convention.
    use fem_assembly::hdiv_error::compute_hdiv_l2_error;
    let l2_err = compute_hdiv_l2_error(&space, &x, &|p: &[f64]| {
        let k = kappa;
        vec![(k * p[0]).cos() * (k * p[1]).sin(),
             (k * p[1]).cos() * (k * p[0]).sin()]
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
