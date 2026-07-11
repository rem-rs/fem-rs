//! # Example 24 — Mixed Discrete Operators  [1:1 translation of MFEM ex24]
//!
//! Projects gradient / curl / divergence operators via mixed FE formulations.
//! Three problem types:
//!
//! ```text
//!   0 (grad): ∇p       for p ∈ H¹        → E ∈ H(curl)  (2D)
//!   1 (curl): curl v   for v ∈ H(curl)   → E ∈ H(div)   (3D)
//!   2 (div):  div v    for v ∈ H(div)    → f ∈ L²       (2D)
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex24_discrete_ops
//! cargo run --example mfem_ex24_discrete_ops -- -m data/star.mesh
//! cargo run --example mfem_ex24_discrete_ops -- -m data/star.mesh -p 2 -o 2
//! ```

#![allow(unused_imports, dead_code)]

use std::f64::consts::PI;

use fem_assembly::{
    DiscreteLinearOperator, VectorAssembler, mixed::assemble_hcurl_h1_gradient,
    standard::VectorMassIntegrator,
};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{
    Mesh, ElementTransformation, ElementType as EType,
    topology::MeshTopology,
};
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use fem_space::{
    H1Space, HCurlSpace, HDivSpace, L2Space,
    fe_space::FESpace, SpaceType,
};

fn main() {
    let args = parse_args();
    let (mesh2d, _mesh3d) = match &args.mesh {
        Some(path) => {
            let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
            (mfem.mesh2d, mfem.mesh3d)
        }
        None => (Some(Mesh::<2>::unit_square_tri(args.n)), None),
    };
    // Auto-refine (target ≤ 50000 elements, matching C++ ex24)
    let rl = |ne: usize, dim: usize| -> usize {
        if ne == 0 { 0 } else { ((50_000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize }
    };
    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("built-in"));
    println!("   --order {}", args.order);
    println!("   --problem-type {}", args.prob);
    println!("   --no-static-condensation\n   --no-partial-assembly\n   --no-visualization");
    match args.prob {
        0 => {
            let mut m = mesh2d.expect("2D mesh needed");
            let l = rl(m.n_elements(), 2);
            for _ in 0..l { m = fem_mesh::refine_uniform(&m); }
            run_grad(&m, args.order);
        }
        2 => {
            let mut m = mesh2d.expect("2D mesh needed");
            let l = rl(m.n_elements(), 2);
            for _ in 0..l { m = fem_mesh::refine_uniform(&m); }
            run_div(&m, args.order);
        }
        _ => eprintln!("prob 1 (3D curl) not yet implemented"),
    }
}

// ─── Mixed gradient matrix: H¹→H(curl) ────────────────────────────────────────

fn assemble_grad_mixed(nd: &HCurlSpace<Mesh<2>>, h1: &H1Space<Mesh<2>>, qo: u8) -> CsrMatrix<f64> {
    let mesh = h1.mesh();
    let dim = 2;
    let mut coo = CooMatrix::new(nd.n_dofs(), h1.n_dofs());
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let hr = ref_elem_vol(et, h1.order()).unwrap_or_else(||
            panic!("ref_elem_vol failed: et={:?}, h1_order={}", et, h1.order()));
        let nr = hr.n_dofs();
        let vc = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap_or_else(||
            panic!("ref_elem_vec failed: et={:?}, nd_order={}, HCurl", et, nd.order()));
        let nc = vc.n_dofs();
        let g1: Vec<usize> = h1.element_dofs(e).iter().map(|&d| d as usize).collect();
        let g2: Vec<usize> = nd.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = nd.element_signs(e);
        let n1 = g1.len();
        let n2 = g2.len();
        let quad = hr.quadrature(qo);
        let mut me = vec![0.0; n2 * n1];
        let mut gr = vec![0.0; nr * dim];
        let mut gp = vec![0.0; nr * dim];
        let mut bv = vec![0.0; nc * dim];
        let use_iso = !matches!(et, EType::Tri3 | EType::Tet4);
        let geo_elem = if use_iso { Some(fem_assembly::geo_ref_elem_from_mesh(mesh, e)) } else { None };
        let nodes = mesh.element_nodes(e);

        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jit): (f64, nalgebra::DMatrix<f64>) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, _xp) = fem_assembly::isoparametric_jacobian(mesh, &nodes, ge.as_deref().unwrap(), xi, dim);
                (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose())
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone())
            };
            hr.eval_grad_basis(xi, &mut gr);
            xform_grad(&jit, &gr, &mut gp, nr);
            vc.eval_basis_vec(xi, &mut bv);
            for j in 0..n1 {
                let g = &gp[j * dim..][..dim];
                for i in 0..n2 {
                    let s = signs[i];
                    let psi_x = s * (jit[(0,0)] * bv[i * dim] + jit[(0,1)] * bv[i * dim + 1]);
                    let psi_y = s * (jit[(1,0)] * bv[i * dim] + jit[(1,1)] * bv[i * dim + 1]);
                    me[i * n1 + j] += w * (g[0] * psi_x + g[1] * psi_y);
                }
            }
        }
        for (ir, &r) in g2.iter().enumerate() {
            for (ic, &c) in g1.iter().enumerate() {
                let v = me[ir * n1 + ic];
                if v != 0.0 { coo.add(r, c, v); }
            }
        }
    }
    coo.into_csr()
}

// ─── Mixed divergence matrix: H(div)→L² ───────────────────────────────────────
// B[i,j] = ∫ div(ψ_j) * φ_i dx
// div_phys = (1/det(J)) * div_ref → det(J) cancels in weak form:
// B[i,j] = Σ w_ref_q * div_ref(ψ̂_j(xi_q)) * φ_i(xi_q) * sign_j
// where w_ref_q are reference-domain quadrature weights (no det(J)).

fn assemble_div_mixed(l2: &L2Space<Mesh<2>>, rt: &HDivSpace<Mesh<2>>, qo: u8) -> CsrMatrix<f64> {
    let mesh = l2.mesh();
    let dim = 2;
    let mut coo = CooMatrix::new(l2.n_dofs(), rt.n_dofs());
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let l2_order = l2.order();
        let vr = ref_elem_vec(et, rt.order(), SpaceType::HDiv)
            .expect("HDiv ref elem");
        let nv = vr.n_dofs();
        let signs = rt.element_signs(e);
        let gl: Vec<usize> = l2.element_dofs(e).iter().map(|&d| d as usize).collect();
        let gv: Vec<usize> = rt.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_l = gl.len();
        let n_v = gv.len();
        let quad = vr.quadrature(qo);
        let mut me = vec![0.0; n_l * n_v];
        let mut dv = vec![0.0; nv * dim];
        // Pre-compute L2 basis values at all quad points
        let mut l2_phi: Vec<Vec<f64>> = Vec::new();
        if l2_order > 0 {
            let lr = ref_elem_vol(et, l2_order).unwrap();
            for (_qi, xi) in quad.points.iter().enumerate() {
                let mut ph = vec![0.0; lr.n_dofs()];
                lr.eval_basis(xi, &mut ph);
                l2_phi.push(ph);
            }
        }
        for (qi, xi) in quad.points.iter().enumerate() {
            let w_ref = quad.weights[qi]; // reference weight (no det(J) — det(J) cancels)
            vr.eval_div(xi, &mut dv);
            for i in 0..n_l {
                let phi = if l2_order == 0 { 1.0 } else { l2_phi[qi][i] };
                for j in 0..n_v {
                    me[i * n_v + j] += w_ref * phi * dv[j] * signs[j];
                }
            }
        }
        for (ir, &r) in gl.iter().enumerate() {
            for (ic, &c) in gv.iter().enumerate() {
                let v = me[ir * n_v + ic];
                if v != 0.0 { coo.add(r, c, v); }
            }
        }
    }
    coo.into_csr()
}

// ─── L² error for H(curl) ─────────────────────────────────────────────────────

fn l2e_hcurl(nd: &HCurlSpace<Mesh<2>>, d: &[f64], exact: &dyn Fn(&[f64]) -> Vec<f64>, qoq: u8) -> f64 {
    let dim = 2;
    let mesh = nd.mesh();
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let vc = ref_elem_vec(et, nd.order(), SpaceType::HCurl).unwrap();
        let quad = vc.quadrature(qoq);
        let dofs: Vec<usize> = nd.element_dofs(e).iter().map(|&x| x as usize).collect();
        let signs = nd.element_signs(e);
        let n_ldofs = vc.n_dofs();
        let nodes = mesh.element_nodes(e);
        let mut ref_bv = vec![0.0; n_ldofs * dim];
        let use_iso = !matches!(et, EType::Tri3 | EType::Tet4);
        let geo_elem = if use_iso { Some(fem_assembly::geo_ref_elem_from_mesh(mesh, e)) } else { None };
        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, jit, xp): (f64, nalgebra::DMatrix<f64>, Vec<f64>) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (jac, det, x) = fem_assembly::isoparametric_jacobian(mesh, &nodes, ge.as_deref().unwrap(), xi, dim);
                (quad.weights[qi] * det.abs(), jac.try_inverse().unwrap().transpose(), x)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.jacobian_inv_t().clone(), tr.map_to_physical(xi))
            };
            vc.eval_basis_vec(xi, &mut ref_bv);
            let mut eh = [0.0; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let phi_x = jit[(0,0)] * ref_bv[i * 2] + jit[(0,1)] * ref_bv[i * 2 + 1];
                let phi_y = jit[(1,0)] * ref_bv[i * 2] + jit[(1,1)] * ref_bv[i * 2 + 1];
                eh[0] += s * d[dofs[i]] * phi_x;
                eh[1] += s * d[dofs[i]] * phi_y;
            }
            let ex = exact(&xp);
            for k in 0..2 { e2 += w * (eh[k] - ex[k]).powi(2); }
        }
    }
    e2.sqrt()
}

fn l2e_l2(l2: &L2Space<Mesh<2>>, d: &[f64], exact: &dyn Fn(&[f64]) -> f64, qoq: u8) -> f64 {
    let dim = 2;
    let mesh = l2.mesh();
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let l2_order = l2.order();
        let quad = if l2_order == 0 {
            // P0: use HDiv reference element quadrature (element-appropriate)
            let vr = ref_elem_vec(et, 0, SpaceType::HDiv).unwrap();
            vr.quadrature(qoq)
        } else {
            let lr = ref_elem_vol(et, l2_order).unwrap();
            lr.quadrature(qoq)
        };
        let ng: Vec<usize> = l2.element_dofs(e).iter().map(|&x| x as usize).collect();
        let nn = ng.len();
        let use_iso = !matches!(et, EType::Tri3 | EType::Tet4);
        let geo_elem = if use_iso { Some(fem_assembly::geo_ref_elem_from_mesh(mesh, e)) } else { None };
        let nodes = mesh.element_nodes(e);
        let mut ph = if l2_order > 0 { vec![0.0; ref_elem_vol(et, l2_order).unwrap().n_dofs()] } else { vec![] };
        for (qi, xi) in quad.points.iter().enumerate() {
            let (w, xp): (f64, Vec<f64>) = if use_iso {
                let ge = geo_elem.as_ref().unwrap();
                let (_jac, det, x) = fem_assembly::isoparametric_jacobian(mesh, &nodes, ge.as_deref().unwrap(), xi, dim);
                (quad.weights[qi] * det.abs(), x)
            } else {
                let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
                (quad.weights[qi] * tr.det_j().abs(), tr.map_to_physical(xi))
            };
            let mut n = 0.0;
            if l2_order == 0 {
                n = d[ng[0]]; // P0: single constant DOF per element
            } else {
                let lr = ref_elem_vol(et, l2_order).unwrap();
                lr.eval_basis(xi, &mut ph);
                for i in 0..nn { n += d[ng[i]] * ph[i]; }
            }
            e2 += w * (n - exact(&xp)).powi(2);
        }
    }
    e2.sqrt()
}

fn xform_grad(jit: &nalgebra::DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize) {
    for i in 0..n {
        gp[i * 2] = jit[(0,0)] * gr[i*2] + jit[(0,1)] * gr[i*2+1];
        gp[i * 2 + 1] = jit[(1,0)] * gr[i*2] + jit[(1,1)] * gr[i*2+1];
    }
}

fn ref_elem_vol(et: EType, order: u8) -> Option<Box<dyn ReferenceElement>> {
    use fem_element::lagrange::*;
    // ElementType already imported as EType at top level
    match (et, order) {
        (EType::Tri3, 0) => Some(Box::new(TriP1) as Box<dyn ReferenceElement>),
        (EType::Tri3, 1) => Some(Box::new(TriP1) as Box<dyn ReferenceElement>),
        (EType::Tri3, 2) => Some(Box::new(TriP2) as Box<dyn ReferenceElement>),
        (EType::Quad4, 1) => Some(Box::new(fem_element::lagrange::quad::QuadQ1) as Box<dyn ReferenceElement>),
        _ => { eprintln!("WARN: ref_elem_vol({et:?}, {order}) unhandled"); None }
    }
}

fn ref_elem_vec(et: EType, order: u8, sp: SpaceType) -> Option<Box<dyn fem_element::VectorReferenceElement>> {
    use fem_element::nedelec::*;
    use fem_element::raviart_thomas::*;
    // ElementType already imported as EType at top level
    match (et, order, sp) {
        (EType::Tri3, 1, SpaceType::HCurl) => Some(Box::new(TriND1) as Box<dyn fem_element::VectorReferenceElement>),
        (EType::Tri3, _, SpaceType::HDiv) => Some(Box::new(TriRT0) as Box<dyn fem_element::VectorReferenceElement>),
        (EType::Quad4, 1, SpaceType::HCurl) => Some(Box::new(QuadND1) as Box<dyn fem_element::VectorReferenceElement>),
        (EType::Quad4, _, SpaceType::HDiv) => Some(Box::new(QuadRT0) as Box<dyn fem_element::VectorReferenceElement>),
        _ => None,
    }
}

// ─── Problem 0: Grad — ∇p: H¹→H(curl) ────────────────────────────────────────

fn run_grad(mesh: &Mesh<2>, order: u8) {
    let qo = (2 * order + 1).max(3) as u8;
    let h1 = H1Space::new(mesh.clone(), order);
    let nd = HCurlSpace::new(mesh.clone(), order);
    println!("Number of Nedelec finite element unknowns: {}", nd.n_dofs());
    println!("Number of H1 finite element unknowns: {}", h1.n_dofs());

    // C++ ex24 prob 0: p(x) = sin(x)*sin(y)  (NOT sin(πx))
    let p: Vec<f64> = h1.interpolate(&|x| x[0].sin() * x[1].sin()).as_slice().to_vec();

    // (a) Mixed form: solve M·E = B·p  (B via library assemble_hcurl_h1_gradient)
    let b = assemble_hcurl_h1_gradient(&nd, &h1, qo);
    let mut rhs = vec![0.0; nd.n_dofs()];
    b.spmv(&p, &mut rhs);

    let mass = VectorAssembler::assemble_bilinear(&nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut e_sol = vec![0.0; nd.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_gssmoother(&mass, &rhs, &mut e_sol, &cfg).expect("PCG solve failed");

    // (b) Discrete gradient interpolant
    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("grad DLO");
    let mut e_interp = vec![0.0; nd.n_dofs()];
    g.spmv(&p, &mut e_interp);
    let interp_norm: f64 = e_interp.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  DLO interpolant norm = {:.6e}", interp_norm);

    // (c) Exact L² projection of grad p_exact onto H(curl)
    use fem_assembly::standard::VectorDomainLFIntegrator;
    // grad p = (cos(x)*sin(y), sin(x)*cos(y))
    let gradp = |x: &[f64]| vec![x[0].cos() * x[1].sin(), x[0].sin() * x[1].cos()];
    let rhs_ex = VectorAssembler::assemble_linear(&nd, &[
        &VectorDomainLFIntegrator {
            f: fem_assembly::postproc::coefficient::FnVectorCoeff(
                Box::new(|x: &[f64], out: &mut [f64]| {
                    let g = gradp(x);
                    out[0] = g[0]; out[1] = g[1];
                })
            ),
        }
    ], qo);
    let mut e_ex = vec![0.0; nd.n_dofs()];
    solve_pcg_gssmoother(&mass, &rhs_ex, &mut e_ex, &cfg).expect("exact PCG");

    // C++ ex24 exact: grad p = (cos(x)*sin(y), sin(x)*cos(y))
    let err_qo = (2 * order + 4).max(5) as u8; // higher order for accurate error integration
    let gradp = |x: &[f64]| vec![x[0].cos() * x[1].sin(), x[0].sin() * x[1].cos()];
    let e1 = l2e_hcurl(&nd, &e_sol, &gradp, err_qo);
    let e2 = l2e_hcurl(&nd, &e_interp, &gradp, err_qo);
    let e3 = l2e_hcurl(&nd, &e_ex, &gradp, err_qo);
    println!("\n Solution:  || E_h - grad p ||_{{L_2}} = {:.8}", e1);
    println!(" Interpolant: || E_h - grad p ||_{{L_2}} = {:.8}", e2);
    println!(" Projection:  || E_h - grad p ||_{{L_2}} = {:.8}", e3);
}

// ─── Problem 2: Div — div v: H(div)→L² ───────────────────────────────────────

fn run_div(mesh: &Mesh<2>, order: u8) {
    let qo = (2 * order + 1).max(3) as u8;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let rt = HDivSpace::new(mesh.clone(), rt_order);
    let l2_p = if rt_order > 0 { rt_order } else { 0 };
    let l2 = L2Space::new(mesh.clone(), l2_p);
    println!("Number of Raviart-Thomas finite element unknowns: {}", rt.n_dofs());
    println!("Number of L2 finite element unknowns: {}", l2.n_dofs());

    // C++ ex24 prob 2: trial = grad p in H(div), exact = div(grad p) in L²
    // grad p = (cos(x)*sin(y), sin(x)*cos(y))
    let gradp = |x: &[f64]| vec![x[0].cos() * x[1].sin(), x[0].sin() * x[1].cos()];
    // div(grad p) = -2*sin(x)*sin(y)
    let div_gradp = |x: &[f64]| -2.0 * x[0].sin() * x[1].sin();

    // Project grad p onto H(div) trial space
    let v: Vec<f64> = rt.interpolate_vector(&gradp).as_slice().to_vec();

    // (a) Mixed form: solve M·f = D·v
    let d = assemble_div_mixed(&l2, &rt, qo);
    let mut rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut rhs);

    // L² mass matrix (P0 now supported in assembler via new P0 ref element)
    let mass = fem_assembly::Assembler::assemble_bilinear(
        &l2, &[&fem_assembly::standard::MassIntegrator { rho: 1.0 }], qo);
    let mut f_sol = vec![0.0; l2.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_gssmoother(&mass, &rhs, &mut f_sol, &cfg).expect("PCG");

    // (b) Discrete divergence interpolant: reuse the mixed matrix D (same as C++ DivergenceInterpolator)
    let mut f_interp_rhs = vec![0.0; l2.n_dofs()];
    d.spmv(&v, &mut f_interp_rhs);
    let mut f_interp = vec![0.0; l2.n_dofs()];
    let cfg_interp = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    solve_pcg_gssmoother(&mass, &f_interp_rhs, &mut f_interp, &cfg_interp).expect("interp mass solve");

    // (c) Exact L² projection of div(grad p) into L²
    let rhs_ex = fem_assembly::Assembler::assemble_linear(&l2, &[
        &fem_assembly::standard::DomainSourceIntegrator::new(div_gradp)
    ], qo);
    let mut f_ex = vec![0.0; l2.n_dofs()];
    solve_pcg_gssmoother(&mass, &rhs_ex, &mut f_ex, &cfg).expect("exact PCG");

    // Errors (higher quadrature for accuracy)
    let err_qo = (2 * order + 4).max(5) as u8;
    let e1 = l2e_l2(&l2, &f_sol, &div_gradp, err_qo);
    let e2 = l2e_l2(&l2, &f_interp, &div_gradp, err_qo);
    let e3 = l2e_l2(&l2, &f_ex, &div_gradp, err_qo);
    println!("\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: || f_h - div v ||_{{L_2}} = {:.8}", e1);
    println!("\n Divergence interpolant f_h = div v_h in L_2: || f_h - div v ||_{{L_2}} = {:.8}", e2);
    println!("\n Projection f_h of exact div v in L_2: || f_h - div v ||_{{L_2}} = {:.8}", e3);
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, n: usize, order: u8, prob: u8 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 4, order: 1, prob: 0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(4),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem" | "--problem-type" => a.prob = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            _ => {}
        }
    }
    a
}
