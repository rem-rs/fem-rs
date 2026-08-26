//! # Example 3 — Maxwell Electromagnetic Diffusion  (1:1 with MFEM ex3)
//!
//! Solves ∇×(∇×E) + E = f with non-homogeneous Dirichlet BC.
//! Supports both 2D and 3D (matching MFEM ex3.cpp `dim` dispatch).
//!
//! Default: data/beam-tet.mesh (3D, 32016 unknowns). Use `-m data/star.mesh` for 2D.

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler, DiscreteLinearOperator,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_element::VectorReferenceElement;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_gf_file};
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{solve_pcg, SolverConfig};
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs_hcurl, form_linear_system},
};

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("(built-in beam-tet)"));
    println!("   --order {}", args.order);
    println!("   --frequency {}", args.freq);

    let mfem = if let Some(ref path) = args.mesh {
        read_mfem_file(path).expect("failed to read MFEM mesh")
    } else {
        read_mfem_file("data/beam-tet.mesh").expect("failed to read data/beam-tet.mesh")
    };

    if let Some(mesh3d) = mfem.mesh3d {
        solve_3d(&args, mesh3d);
    } else if let Some(mesh2d) = mfem.mesh2d {
        solve_2d(&args, mesh2d);
    } else {
        panic!("MFEM file must contain a 2D or 3D mesh");
    }
}

// ─── 2D ─────────────────────────────────────────────────────────────────────

fn solve_2d(args: &Args, mut mesh: Mesh<2>) {
    let dim = 2;
    let n_ref = ((50000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize;
    for _ in 0..n_ref { mesh = refine_uniform(&mesh); }

    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    let tags = space.mesh().unique_boundary_tags();
    let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);
    eprintln!("  Boundary DOFs: {} / {}", ess_bdr.len(), n_dofs);

    let kappa = args.freq * PI;
    let qo = args.order as u8 * 2 + 2;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&Src2D { kappa }], qo);
    let u_proj = project_2d(&space, kappa);
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    let mut mat = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);

    solve_report_2d(&space, &mut x, &rhs, args, kappa);
    write_mfem_file("refined.mesh", space.mesh()).unwrap();
    write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order, dim, 14).unwrap();
    eprintln!("  Wrote refined.mesh and sol.gf");
}

struct Src2D { kappa: f64 }
impl VectorLinearIntegrator for Src2D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs { f[i] += qp.weight * (qp.phi_vec[i*2]*fx + qp.phi_vec[i*2+1]*fy); }
    }
}

fn exact_2d(x: &[f64], k: f64) -> [f64; 2] { [(k*x[1]).sin(), (k*x[0]).sin()] }

fn project_2d(space: &HCurlSpace<Mesh<2>>, k: f64) -> Vec<f64> {
    space.interpolate_vector(&|x| exact_2d(x, k).to_vec()).into_vec()
}

fn l2_err_2d<F: Fn(&[f64]) -> [f64; 2]>(mesh: &Mesh<2>, sp: &HCurlSpace<Mesh<2>>, u: &[f64], ex: F) -> f64 {
    use fem_element::nedelec::TriND1;
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        let r = TriND1; let n = r.n_dofs(); let q = r.quadrature(6);
        let mut p = vec![0.0; n*2];
        let d: Vec<usize> = sp.element_dofs(e).iter().map(|&x| x as usize).collect();
        let s = sp.element_signs(e);
        let nd = mesh.element_nodes(e);
        let x0 = mesh.node_coords(nd[0]); let x1 = mesh.node_coords(nd[1]); let x2 = mesh.node_coords(nd[2]);
        let dj = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
        let ij = 1.0 / dj.max(1e-30);
        let (j00,j01,j10,j11) = (x1[1]*ij, -x1[0]*ij, -x2[1]*ij, x2[0]*ij);
        for (qi, xi) in q.points.iter().enumerate() {
            r.eval_basis_vec(xi, &mut p);
            let w = q.weights[qi] * dj;
            let mut uh = [0.0; 2];
            for a in 0..n {
                let sa = s[a];
                uh[0] += sa * u[d[a]] * (j00*p[a*2] + j01*p[a*2+1]);
                uh[1] += sa * u[d[a]] * (j10*p[a*2] + j11*p[a*2+1]);
            }
            let xp = [(1.0-xi[0]-xi[1])*x0[0]+xi[0]*x1[0]+xi[1]*x2[0], (1.0-xi[0]-xi[1])*x0[1]+xi[0]*x1[1]+xi[1]*x2[1]];
            let e = ex(&xp);
            e2 += w * ((uh[0]-e[0]).powi(2) + (uh[1]-e[1]).powi(2));
        }
    }
    e2.sqrt()
}

fn solve_report_2d(sp: &HCurlSpace<Mesh<2>>, x: &mut [f64], b: &[f64], a: &Args, k: f64) {
    let qo = a.order as u8 * 2 + 2;
    let mut mat = VectorAssembler::assemble_bilinear(sp, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], qo);
    if a.no_ams {
        let precond = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&mat)).unwrap();
        let r = solve_pcg(&mat, b, x, &precond, 1e-12, 2000, true).unwrap();
        println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    } else {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(&fem_space::H1Space::new(sp.mesh().clone(), 1), sp).unwrap();
        let r = solve_pcg_ams(&mat, &ftl(&g), b, x, &AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true, ..SolverConfig::default() },
            ams_cfg: AmsConfig::default(),
        }).unwrap();
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    }
    println!("\n|| E_h - E ||_{{L^2}} = {:.14e}\n", l2_err_2d(sp.mesh(), sp, x, |xi| exact_2d(xi, k)));
}

// ─── 3D ─────────────────────────────────────────────────────────────────────

fn solve_3d(args: &Args, mut mesh: Mesh<3>) {
    let dim = 3;
    let n_ref = ((50000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize;
    for _ in 0..n_ref { mesh = fem_mesh::amr::refine_uniform_3d(&mesh); }

    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    let tags = space.mesh().unique_boundary_tags();
    let ess_bdr = boundary_dofs_hcurl(space.mesh(), &space, &tags);
    eprintln!("  Boundary DOFs: {} / {}", ess_bdr.len(), n_dofs);

    let kappa = args.freq * PI;
    let qo = args.order as u8 * 2 + 2;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&Src3D { kappa }], qo);
    let u_proj = project_3d(&space, kappa);
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    let mut mat = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], qo);
    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);

    solve_report_3d(&space, &mut x, &rhs, args, kappa);
    write_mfem_file_3d("refined.mesh", space.mesh()).unwrap();
    write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order, dim, 14).unwrap();
    eprintln!("  Wrote refined.mesh and sol.gf");
}

struct Src3D { kappa: f64 }
impl VectorLinearIntegrator for Src3D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[2]).sin();
        let fz = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs { f[i] += qp.weight * (qp.phi_vec[i*3]*fx + qp.phi_vec[i*3+1]*fy + qp.phi_vec[i*3+2]*fz); }
    }
}

fn exact_3d(x: &[f64], k: f64) -> [f64; 3] { [(k*x[1]).sin(), (k*x[2]).sin(), (k*x[0]).sin()] }

fn project_3d(space: &HCurlSpace<Mesh<3>>, k: f64) -> Vec<f64> {
    space.interpolate_vector(&|x| exact_3d(x, k).to_vec()).into_vec()
}

fn jac_3d(mesh: &Mesh<3>, e: u32, xi: &[f64]) -> (nalgebra::DMatrix<f64>, [f64; 3]) {
    let n = mesh.element_nodes(e);
    let x0 = mesh.node_coords(n[0]); let x1 = mesh.node_coords(n[1]);
    let x2 = mesh.node_coords(n[2]); let x3 = mesh.node_coords(n[3]);
    let (a,b,c) = (xi[0], xi[1], xi[2]);
    let j = nalgebra::dmatrix![
        -x0[0]+x1[0], -x0[0]+x2[0], -x0[0]+x3[0];
        -x0[1]+x1[1], -x0[1]+x2[1], -x0[1]+x3[1];
        -x0[2]+x1[2], -x0[2]+x2[2], -x0[2]+x3[2]
    ];
    let xp = [
        (1.0-a-b-c)*x0[0]+a*x1[0]+b*x2[0]+c*x3[0],
        (1.0-a-b-c)*x0[1]+a*x1[1]+b*x2[1]+c*x3[1],
        (1.0-a-b-c)*x0[2]+a*x1[2]+b*x2[2]+c*x3[2],
    ];
    (j, xp)
}

fn l2_err_3d<F: Fn(&[f64]) -> [f64; 3]>(mesh: &Mesh<3>, sp: &HCurlSpace<Mesh<3>>, u: &[f64], ex: F) -> f64 {
    use fem_element::nedelec::{TetND1, HexND1};
    let mut e2 = 0.0;
    for e in mesh.elem_iter() {
        let r: &dyn VectorReferenceElement = match mesh.element_type(e) {
            fem_mesh::element_type::ElementType::Hex8 => &HexND1,
            _ => &TetND1,
        };
        let n = r.n_dofs(); let q = r.quadrature(6);
        let mut p = vec![0.0; n*3];
        let d: Vec<usize> = sp.element_dofs(e).iter().map(|&x| x as usize).collect();
        let s = sp.element_signs(e);
        for (qi, xi) in q.points.iter().enumerate() {
            r.eval_basis_vec(xi, &mut p);
            let (j, xp) = jac_3d(mesh, e, xi);
            let w = q.weights[qi] * j.determinant().abs();
            let jt = j.try_inverse().unwrap_or_default().transpose();
            let mut uh = [0.0; 3];
            for a in 0..n {
                let sa = s[a];
                for c in 0..3 {
                    let mut v = 0.0;
                    for k in 0..3 { v += jt[(c,k)] * p[a*3+k]; }
                    uh[c] += sa * u[d[a]] * v;
                }
            }
            let ex = ex(&xp);
            e2 += w * ((uh[0]-ex[0]).powi(2) + (uh[1]-ex[1]).powi(2) + (uh[2]-ex[2]).powi(2));
        }
    }
    e2.sqrt()
}

fn solve_report_3d(sp: &HCurlSpace<Mesh<3>>, x: &mut [f64], b: &[f64], a: &Args, k: f64) {
    let qo = a.order as u8 * 2 + 2;
    let mut mat = VectorAssembler::assemble_bilinear(sp, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], qo);
    if a.no_ams {
        let precond = fem_solver::GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&mat)).unwrap();
        let r = solve_pcg(&mat, b, x, &precond, 1e-12, 2000, true).unwrap();
        println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    } else {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(&fem_space::H1Space::new(sp.mesh().clone(), 1), sp).unwrap();
        let r = solve_pcg_ams(&mat, &ftl(&g), b, x, &AmsSolverConfig {
            inner_cfg: SolverConfig { rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true, ..SolverConfig::default() },
            ams_cfg: AmsConfig::default(),
        }).unwrap();
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}", r.iterations, r.final_residual);
    }
    println!("\n|| E_h - E ||_{{L^2}} = {:.14e}\n", l2_err_3d(sp.mesh(), sp, x, |xi| exact_3d(xi, k)));
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Args { mesh: Option<String>, order: u8, freq: f64, no_ams: bool, vis: bool }

fn parse_args() -> Args {
    let mut a = Args { order: 1, freq: 1.0, ..Args::default() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-f" | "--frequency" => { a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "-no-ams" => { a.no_ams = true; }
            "-vis" => { a.vis = true; }
            "-no-vis" => { a.vis = false; }
            _ => {}
        }
    }
    a
}
