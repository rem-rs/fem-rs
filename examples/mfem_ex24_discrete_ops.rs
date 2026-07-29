//! # Example 24 — Mixed Discrete Operators  [1:1 translation of MFEM ex24]
//!
//! Projects gradient / curl / divergence operators via mixed FE formulations.
//! Three problem types:
//!
//! ```text
//!   0 (grad): ∇p       for p ∈ H¹        → E ∈ H(curl)  (2D + 3D)
//!   1 (curl): curl v   for v ∈ H(curl)   → E ∈ H(div)   (3D)
//!   2 (div):  div v    for v ∈ H(div)    → f ∈ L²       (2D + 3D)
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex24_discrete_ops
//! cargo run --example mfem_ex24_discrete_ops -- -m data/star.mesh -p 2 -o 2
//! cargo run --example mfem_ex24_discrete_ops -- -m data/beam-tet.mesh -p 1
//! cargo run --example mfem_ex24_discrete_ops -- -p 0 -vis
//! ```

#![allow(non_snake_case)]

use std::f64::consts::PI;
use fem_assembly::{
    DiscreteLinearOperator, VectorAssembler,
    mixed::{assemble_hcurl_h1_gradient, assemble_hdiv_l2_mixed, HDivL2DivIntegrator},
    postproc::grid_function::{
        compute_l2_error_hcurl, compute_l2_error_hdiv, compute_l2_error_l2,
        project_hdiv_coefficient_3d,
    },
    project_coefficient,
    project_hdiv_coefficient_2d,
    standard::{DomainSourceIntegrator, MassIntegrator, VectorDomainLFIntegrator, VectorMassIntegrator},
    postproc::coefficient::FnVectorCoeff,
    Assembler,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_mesh::{refine_uniform, ElementType, topology::MeshTopology, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{fe_space::FESpace, H1Space, HCurlSpace, HDivSpace, L2Space};

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: u8,
    prob: u8,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args { mesh_file: String::new(), order: 1, prob: 0, visualization: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex24 [-m mesh] [-o order] [-p prob] [-vis]");
                std::process::exit(0);
            }
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => a.prob = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}

// ─── Exact solution functions (matching C++ ex24) ────────────────────────────

fn p_exact(x: &[f64]) -> f64 {
    if x.len() == 3 {
        x[0].sin() * x[1].sin() * x[2].sin()
    } else {
        x[0].sin() * x[1].sin()
    }
}

fn gradp_exact(x: &[f64]) -> Vec<f64> {
    let mut g = if x.len() == 3 {
        vec![
            x[0].cos() * x[1].sin() * x[2].sin(),
            x[0].sin() * x[1].cos() * x[2].sin(),
            x[0].sin() * x[1].sin() * x[2].cos(),
        ]
    } else {
        vec![
            x[0].cos() * x[1].sin(),
            x[0].sin() * x[1].cos(),
        ]
    };
    while g.len() < x.len() { g.push(0.0); }
    g
}

fn div_gradp_exact(x: &[f64]) -> f64 {
    if x.len() == 3 {
        -3.0 * x[0].sin() * x[1].sin() * x[2].sin()
    } else {
        -2.0 * x[0].sin() * x[1].sin()
    }
}

fn v_exact(x: &[f64]) -> Vec<f64> {
    let kappa = PI;
    let mut v = if x.len() == 3 {
        vec![
            (kappa * x[1]).sin(),
            (kappa * x[2]).sin(),
            (kappa * x[0]).sin(),
        ]
    } else {
        vec![
            (kappa * x[1]).sin(),
            (kappa * x[0]).sin(),
        ]
    };
    while v.len() < x.len() { v.push(0.0); }
    v
}

fn curlv_exact(x: &[f64]) -> Vec<f64> {
    let kappa = PI;
    vec![
        -kappa * (kappa * x[2]).cos(),
        -kappa * (kappa * x[0]).cos(),
        -kappa * (kappa * x[1]).cos(),
    ]
}

// ─── Main dispatch ────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Read mesh
    let (mesh2d, mesh3d) = if args.mesh_file.is_empty() {
        (Some(Mesh::<2>::unit_square_tri(4)), None)
    } else {
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
        (mfem.mesh2d, mfem.mesh3d)
    };

    // Auto-refine (target ≤ 50000 elements, matching C++ ex24)
    let ref_levels = |ne: usize, dim: usize| -> usize {
        if ne == 0 { 0 } else {
            ((50_000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        }
    };

    println!("Options used:");
    println!("   --mesh {}", if args.mesh_file.is_empty() { "built-in" } else { &args.mesh_file });
    println!("   --order {}", args.order);
    println!("   --problem-type {}", args.prob);
    if args.visualization { println!("   --visualization"); }

    match args.prob {
        0 => {
            if let Some(mut m) = mesh2d {
                let l = ref_levels(m.n_elements(), 2);
                for _ in 0..l { m = refine_uniform(&m); }
                solve_grad_2d(&m, args.order, args.visualization);
            } else if let Some(mut m) = mesh3d {
                let l = ref_levels(m.n_elements(), 3);
                for _ in 0..l { m = fem_mesh::refine_uniform_3d(&m); }
                solve_grad_3d(&m, args.order, args.visualization);
            }
        }
        1 => {
            if let Some(mut m) = mesh3d {
                let l = ref_levels(m.n_elements(), 3);
                for _ in 0..l { m = fem_mesh::refine_uniform_3d(&m); }
                solve_curl_3d(&m, args.order, args.visualization);
            } else {
                eprintln!("Problem 1 (curl) requires a 3D mesh");
            }
        }
        2 => {
            if let Some(mut m) = mesh2d {
                let l = ref_levels(m.n_elements(), 2);
                for _ in 0..l { m = refine_uniform(&m); }
                solve_div_2d(&m, args.order, args.visualization);
            } else if let Some(mut m) = mesh3d {
                let l = ref_levels(m.n_elements(), 3);
                for _ in 0..l { m = fem_mesh::refine_uniform_3d(&m); }
                solve_div_3d(&m, args.order, args.visualization);
            }
        }
        _ => eprintln!("Unrecognized problem type: {}", args.prob),
    }
}

// ─── Problem 0: Grad — ∇p: H¹→H(curl) (2D) ──────────────────────────────────

fn solve_grad_2d(mesh: &Mesh<2>, order: u8, vis: bool) {
    let dim = 2;
    let qo = (2 * order + 1).max(3) as u8;
    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);
    println!("Number of Nedelec finite element unknowns: {}", nd.n_dofs());
    println!("Number of H1 finite element unknowns: {}", h1.n_dofs());

    // Project p = sin(x)*sin(y) onto H1
    let p = project_coefficient(&h1, &|x: &[f64]| p_exact(x), qo);

    // (a) Mixed form: M·E = B·p
    let b = assemble_hcurl_h1_gradient(&nd, &h1, qo);
    let mut rhs = vec![0.0; nd.n_dofs()];
    b.spmv(&p, &mut rhs);
    let mass = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut e_sol = vec![0.0; nd.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_jacobi(&mass, &rhs, &mut e_sol, &cfg).expect("PCG solve failed");

    // (b) DLO gradient interpolant
    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("grad DLO");
    let mut e_interp = vec![0.0; nd.n_dofs()];
    g.spmv(&p, &mut e_interp);
    let interp_norm: f64 = e_interp.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  DLO interpolant norm = {:.6e}", interp_norm);

    // (c) Exact L² projection of grad p onto H(curl)
    let rhs_ex = VectorAssembler::assemble_linear(&nd, &[
        &VectorDomainLFIntegrator {
            f: FnVectorCoeff(Box::new(|x: &[f64], out: &mut [f64]| {
                let g = gradp_exact(x);
                out[0] = g[0]; out[1] = g[1];
            })),
        }
    ], qo);
    let mut e_ex = vec![0.0; nd.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut e_ex, &cfg).expect("exact PCG");

    // L² errors (higher quadrature for accuracy)
    let err_qo = (2 * order + 4).max(5) as u8;
    let gradp = |x: &[f64]| gradp_exact(x);
    let e1 = compute_l2_error_hcurl(&e_sol, &nd, &gradp, err_qo);
    let e2 = compute_l2_error_hcurl(&e_interp, &nd, &gradp, err_qo);
    let e3 = compute_l2_error_hcurl(&e_ex, &nd, &gradp, err_qo);
    println!("\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): || E_h - grad p ||_{{L_2}} = {:.8}\n", e1);
    println!(" Gradient interpolant E_h = grad p_h in H(curl): || E_h - grad p ||_{{L_2}} = {:.8}\n", e2);
    println!(" Projection E_h of exact grad p in H(curl): || E_h - grad p ||_{{L_2}} = {:.8}\n", e3);

    // Output with precision(8) matching C++
    write_mfem_file("refined.mesh", mesh).expect("write mesh");
    write_mfem_gf_file("sol.gf", dim, &e_sol, "H1", 1, 1, 8).expect("write sol.gf");
    println!("\nWrote refined.mesh and sol.gf");
    if vis {
        println!("  glvis -m refined.mesh -g sol.gf");
    }
}

// ─── Problem 0: Grad (3D) — ∇p: H¹→H(curl) ──────────────────────────────────

fn solve_grad_3d(mesh: &Mesh<3>, order: u8, vis: bool) {
    let dim = 3;
    let qo = (2 * order + 1).max(3) as u8;
    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);
    println!("Number of Nedelec finite element unknowns: {}", nd.n_dofs());
    println!("Number of H1 finite element unknowns: {}", h1.n_dofs());

    // Project p = sin(x)*sin(y)*sin(z) onto H1
    let p = project_coefficient(&h1, &|x: &[f64]| p_exact(x), qo);

    // (a) Mixed form: M·E = B·p
    // Note: assemble_hcurl_h1_gradient uses 2D-specific code (2×2 J⁻ᵀ).
    // For 3D we assemble the mixed matrix inline.
    let mesh_ref = nd.mesh();
    let b_3d = {
        use fem_linalg::CooMatrix;
        use fem_element::{vec_ref_elem, VecFamily};
        let mut coo = CooMatrix::new(nd.n_dofs(), h1.n_dofs());
        for e in mesh_ref.elem_iter() {
            let et = mesh_ref.element_type(e);
            let h1_ref = et.ref_elem(order);
            let n_h1 = h1_ref.n_dofs();
            let nd_ref = vec_ref_elem(VecFamily::Nedelec, et.to_elem_type(), order);
            let n_nd = nd_ref.n_dofs();
            let signs = nd.element_signs(e);
            let global_h1: Vec<usize> = h1.element_dofs(e).iter().map(|&d| d as usize).collect();
            let global_nd: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();
            let quad = h1_ref.quadrature(qo);
            let nodes = mesh_ref.element_nodes(e);
            let mut me = vec![0.0; global_nd.len() * global_h1.len()];
            let mut gr = vec![0.0; n_h1 * dim];
            let mut gp = vec![0.0; n_h1 * dim];
            let mut nd_basis = vec![0.0; n_nd * dim];
            let use_iso = !matches!(et, ElementType::Tri3 | ElementType::Tet4 | ElementType::Line2);
            let geo_elem = if use_iso { Some(et.ref_elem(mesh_ref.geom_order().max(1))) } else { None };
            for (qi, xi) in quad.points.iter().enumerate() {
                let (w, jit): (f64, nalgebra::DMatrix<f64>) = if use_iso {
                    let ge = geo_elem.as_ref().unwrap();
                    let (jac, det, _xp) = fem_assembly::isoparametric_jacobian(mesh_ref, &nodes, ge.as_ref(), xi, dim);
                    (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose())
                } else {
                    let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh_ref, nodes);
                    (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone())
                };
                h1_ref.eval_grad_basis(xi, &mut gr);
                fem_mesh::xform_grads(&jit, &gr, &mut gp, n_h1, dim);
                nd_ref.eval_basis_vec(xi, &mut nd_basis);
                for j in 0..global_h1.len() {
                    let g = &gp[j * dim..][..dim];
                    for i in 0..global_nd.len() {
                        let s = signs[i] as f64;
                        // HCurl covariant Piola: φ_phys = J^{-T} · φ̂_ref
                        let mut psi = [0.0; 3];
                        for c in 0..dim {
                            let mut sum = 0.0;
                            for k in 0..dim { sum += jit[(c, k)] * nd_basis[i * dim + k]; }
                            psi[c] = s * sum;
                        }
                        let mut dot = 0.0;
                        for c in 0..dim { dot += g[c] * psi[c]; }
                        me[i * global_h1.len() + j] += w * dot;
                    }
                }
            }
            for (ir, &r) in global_nd.iter().enumerate() {
                for (ic, &c) in global_h1.iter().enumerate() {
                    let v = me[ir * global_h1.len() + ic];
                    if v != 0.0 { coo.add(r, c, v); }
                }
            }
        }
        coo.into_csr()
    };
    let mut rhs = vec![0.0; nd.n_dofs()];
    b_3d.spmv(&p, &mut rhs);
    let mass = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut e_sol = vec![0.0; nd.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_jacobi(&mass, &rhs, &mut e_sol, &cfg).expect("PCG solve failed");

    // (b) DLO gradient interpolant
    let g = DiscreteLinearOperator::gradient(&h1, &nd);
    let mut e_interp = vec![0.0; nd.n_dofs()];
    if let Ok(ref g) = g {
        g.spmv(&p, &mut e_interp);
        let interp_norm: f64 = e_interp.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("  DLO interpolant norm = {:.6e}", interp_norm);
    }

    // (c) Exact L² projection of grad p onto H(curl) in 3D
    let rhs_ex = VectorAssembler::assemble_linear(&nd, &[
        &VectorDomainLFIntegrator {
            f: FnVectorCoeff(Box::new(|x: &[f64], out: &mut [f64]| {
                let g = gradp_exact(x);
                for i in 0..3 { out[i] = g[i]; }
            })),
        }
    ], qo);
    let mut e_ex = vec![0.0; nd.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut e_ex, &cfg).expect("exact PCG");

    // L² errors
    let err_qo = (2 * order + 4).max(5) as u8;
    let gradp = |x: &[f64]| gradp_exact(x);
    let e1 = compute_l2_error_hcurl(&e_sol, &nd, &gradp, err_qo);
    let e3 = compute_l2_error_hcurl(&e_ex, &nd, &gradp, err_qo);
    println!("\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): || E_h - grad p ||_{{L_2}} = {:.8}\n", e1);
    println!(" Projection E_h of exact grad p in H(curl): || E_h - grad p ||_{{L_2}} = {:.8}\n", e3);

    // Output
    write_mfem_file_3d("refined.mesh", mesh).expect("write mesh 3d");
    write_mfem_gf_file("sol.gf", dim, &e_sol, "H1", 1, 1, 8).expect("write sol.gf");
    println!("\nWrote refined.mesh and sol.gf");
    if vis {
        println!("  glvis -m refined.mesh -g sol.gf");
    }
}

// ─── Problem 1: Curl (3D) — curl v: H(curl)→H(div) ──────────────────────────

fn solve_curl_3d(mesh: &Mesh<3>, order: u8, vis: bool) {
    let dim = 3;
    let qo = (2 * order + 1).max(3) as u8;
    let nd_order = order;
    let rt_order = if order > 0 { order - 1 } else { 0 };

    let nd = HCurlSpace::new(mesh.clone(), nd_order);
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    println!("Number of Nedelec finite element unknowns: {}", nd.n_dofs());
    println!("Number of Raviart-Thomas finite element unknowns: {}", rt.n_dofs());

    // Interpolate v = (sin(κy), sin(κz), sin(κx)) onto H(curl)
    let v = nd.interpolate_vector(&|x: &[f64]| v_exact(x)).into_vec();

    // (a) Mixed form: M·w = C·v
    let c = fem_assembly::mixed::assemble_hcurl_hdiv_weak_curl(&nd, &rt, qo, 1.0);
    let mut rhs = vec![0.0; rt.n_dofs()];
    c.spmv(&v, &mut rhs);
    let mass = VectorAssembler::assemble_bilinear(&rt, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut w_sol = vec![0.0; rt.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_jacobi(&mass, &rhs, &mut w_sol, &cfg).expect("PCG");

    // (b) DLO curl interpolant
    let curl_dlo = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d DLO");
    let mut w_interp = vec![0.0; rt.n_dofs()];
    curl_dlo.spmv(&v, &mut w_interp);

    // (c) Exact L² projection of curl v into H(div)
    struct CurlVExact;
    impl fem_assembly::postproc::coefficient::VectorCoeff for CurlVExact {
        fn eval(&self, ctx: &fem_assembly::postproc::coefficient::CoeffCtx<'_>, out: &mut [f64]) {
            let cv = curlv_exact(ctx.x);
            for i in 0..3 { out[i] = cv[i]; }
        }
    }
    let rhs_ex = VectorAssembler::assemble_linear(&rt, &[&VectorDomainLFIntegrator { f: CurlVExact }], qo);
    let mut w_ex = vec![0.0; rt.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut w_ex, &cfg).expect("exact PCG");

    // L² errors
    let err_qo = (2 * order + 4).max(5) as u8;
    let curlv_fn = |x: &[f64]| curlv_exact(x);
    let e1 = compute_l2_error_hdiv(&w_sol, &rt, &curlv_fn, err_qo);
    let e2 = compute_l2_error_hdiv(&w_interp, &rt, &curlv_fn, err_qo);
    let e3 = compute_l2_error_hdiv(&w_ex, &rt, &curlv_fn, err_qo);
    println!("\n Solution of (E_h,w) = (curl v_h,w) for E_h and w in H(div): || E_h - curl v ||_{{L_2}} = {:.8}\n", e1);
    println!(" Curl interpolant E_h = curl v_h in H(div): || E_h - curl v ||_{{L_2}} = {:.8}\n", e2);
    println!(" Projection E_h of exact curl v in H(div): || E_h - curl v ||_{{L_2}} = {:.8}\n", e3);

    // Output
    write_mfem_file_3d("refined.mesh", mesh).expect("write mesh 3d");
    write_mfem_gf_file("sol.gf", dim, &w_sol, "H1", 1, 1, 8).expect("write sol.gf");
    println!("\nWrote refined.mesh and sol.gf");
    if vis {
        println!("  glvis -m refined.mesh -g sol.gf");
    }
}

// ─── Problem 2: Div — div v: H(div)→L² (2D) ─────────────────────────────────

fn solve_div_2d(mesh: &Mesh<2>, order: u8, vis: bool) {
    let dim = 2;
    let qo = (2 * order + 1).max(3) as u8;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let l2_p = if rt_order > 0 { rt_order } else { 0 };
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2 = L2Space::new(mesh.clone(), l2_p);
    println!("Number of Raviart-Thomas finite element unknowns: {}", rt.n_dofs());
    println!("Number of L2 finite element unknowns: {}", l2.n_dofs());

    // Project grad p onto H(div)
    let v = project_hdiv_coefficient_2d(
        &rt,
        &|x: &[f64], out: &mut [f64]| {
            let gp = gradp_exact(x);
            out[0] = gp[0]; out[1] = gp[1];
        },
        qo,
    );

    // (a) Mixed form: M·f = D·v
    let d = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo);
    let mut rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut rhs);
    let mass = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
    let mut f_sol = vec![0.0; l2.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_jacobi(&mass, &rhs, &mut f_sol, &cfg).expect("PCG");

    // (b) DLO divergence interpolant
    let mut f_interp = vec![0.0; l2.n_dofs()];
    let mut interp_rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut interp_rhs);
    solve_pcg_jacobi(&mass, &interp_rhs, &mut f_interp, &cfg).expect("interp mass solve");

    // (c) Exact L² projection of div(grad p) into L²
    let rhs_ex = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(div_gradp_exact)], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut f_ex, &cfg).expect("exact PCG");

    // L² errors
    let err_qo = (2 * order + 6).max(7) as u8;
    let e1 = compute_l2_error_l2(&f_sol, &l2, &div_gradp_exact, err_qo);
    let e2 = compute_l2_error_l2(&f_interp, &l2, &div_gradp_exact, err_qo);
    let e3 = compute_l2_error_l2(&f_ex, &l2, &div_gradp_exact, err_qo);
    println!("\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e1);
    println!(" Divergence interpolant f_h = div v_h in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e2);
    println!(" Projection f_h of exact div v in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e3);

    // Output
    write_mfem_file("refined.mesh", mesh).expect("write mesh");
    write_mfem_gf_file("sol.gf", dim, &f_sol, "H1", 1, 1, 8).expect("write sol.gf");
    println!("\nWrote refined.mesh and sol.gf");
    if vis {
        println!("  glvis -m refined.mesh -g sol.gf");
    }
}

// ─── Problem 2: Div (3D) — div v: H(div)→L² ─────────────────────────────────

fn solve_div_3d(mesh: &Mesh<3>, order: u8, vis: bool) {
    let dim = 3;
    let qo = (2 * order + 1).max(3) as u8;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let l2_p = if rt_order > 0 { rt_order } else { 0 };
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2 = L2Space::new(mesh.clone(), l2_p);
    println!("Number of Raviart-Thomas finite element unknowns: {}", rt.n_dofs());
    println!("Number of L2 finite element unknowns: {}", l2.n_dofs());

    // Project grad p onto H(div) in 3D
    let v = project_hdiv_coefficient_3d(
        &rt,
        &|x: &[f64], out: &mut [f64]| {
            let gp = gradp_exact(x);
            for i in 0..3 { out[i] = gp[i]; }
        },
        qo,
    );

    // (a) Mixed form: M·f = D·v
    let d = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo);
    let mut rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut rhs);
    let mass = Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], qo);
    let mut f_sol = vec![0.0; l2.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_jacobi(&mass, &rhs, &mut f_sol, &cfg).expect("PCG");

    // (b) DLO divergence interpolant
    let mut f_interp = vec![0.0; l2.n_dofs()];
    let mut interp_rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut interp_rhs);
    solve_pcg_jacobi(&mass, &interp_rhs, &mut f_interp, &cfg).expect("interp mass solve");

    // (c) Exact L² projection of div(grad p) into L²
    let rhs_ex = Assembler::assemble_linear(&l2, &[&DomainSourceIntegrator::new(div_gradp_exact)], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_jacobi(&mass, &rhs_ex, &mut f_ex, &cfg).expect("exact PCG");

    // L² errors
    let err_qo = (2 * order + 6).max(7) as u8;
    let e1 = compute_l2_error_l2(&f_sol, &l2, &div_gradp_exact, err_qo);
    let e2 = compute_l2_error_l2(&f_interp, &l2, &div_gradp_exact, err_qo);
    let e3 = compute_l2_error_l2(&f_ex, &l2, &div_gradp_exact, err_qo);
    println!("\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e1);
    println!(" Divergence interpolant f_h = div v_h in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e2);
    println!(" Projection f_h of exact div v in L_2: || f_h - div v ||_{{L_2}} = {:.8}\n", e3);

    // Output
    write_mfem_file_3d("refined.mesh", mesh).expect("write mesh 3d");
    write_mfem_gf_file("sol.gf", dim, &f_sol, "H1", 1, 1, 8).expect("write sol.gf");
    println!("\nWrote refined.mesh and sol.gf");
    if vis {
        println!("  glvis -m refined.mesh -g sol.gf");
    }
}
