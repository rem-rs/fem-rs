//! # Example 3 — Maxwell Electromagnetic Diffusion  (1:1 with MFEM ex3)
//!
//! Solves the second-order definite Maxwell problem:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω
//!      n×(∇×E) = 0    on ∂Ω  (natural)
//! ```
//!
//! 3D version (matching MFEM ex3 default): `beam-tet.mesh`, ND1 (TetND1)
//! H(curl) space, muinv = sigma = 1, κ = π·freq.
//!
//! C++ exact solution (MFEM ex3.cpp `E_exact`):
//!   3D:  E = (sin(κy), sin(κz), sin(κx))
//!   2D:  E = (sin(κy), sin(κx))
//!
//! f = (1+κ²)·E ( manufactured source ).
//!
//! Non-homogeneous Dirichlet BC (n×E projected onto boundary edges).
//! Solver: PCG + AMS (default, matches MFEM ex3) or PCG + GSSmoother (-no-ams).

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler, DiscreteLinearOperator,
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_element::VectorReferenceElement;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_mesh::{refine_uniform, Mesh};
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

    // 3. Read mesh (default beam-tet.mesh — MFEM ex3 default).
    let mesh: Mesh<3> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh3d.expect("MFEM mesh must be 3D")
    } else {
        let mfem = read_mfem_file("data/beam-tet.mesh")
            .expect("failed to read data/beam-tet.mesh");
        mfem.mesh3d.expect("beam-tet.mesh must be 3-D")
    };
    let dim = 3;

    // 4. Refine: ≤ 50 000 elements.
    let ref_levels =
        ((50000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize;
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels { m = refine_uniform(&m); }
        m
    } else { mesh };

    // 5. H(curl) space (TetND1 for 3D order 1).
    let space = HCurlSpace::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("\nNumber of finite element unknowns: {n_dofs}");

    // 6. Essential BC DOFs — all boundaries.
    let all_tags: Vec<i32> = space.mesh().unique_boundary_tags();
    let ess_bdr = if all_tags.is_empty() {
        vec![]
    } else {
        boundary_dofs_hcurl(space.mesh(), &space, &all_tags)
    };
    eprintln!("  Boundary DOFs: {} / {}", ess_bdr.len(), n_dofs);

    // 7. RHS: f = (1+κ²)·E_exact (3D).
    let kappa = args.freq * PI;
    let source = MaxwellSource3D { kappa };
    let quad_order = args.order as u8 * 2 + 2;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&source], quad_order);

    // 8. Project exact solution → initial guess + BC values.
    let u_proj = project_hcurl_exact_3d(&space, kappa, quad_order);
    let bc_vals: Vec<f64> = ess_bdr.iter().map(|&d| u_proj[d as usize]).collect();

    // 9. Stiffness: a(u,v) = ∫ (∇×u)·(∇×v) + u·v dx.
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    let mut mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl, &vec_mass], quad_order,
    );

    // 10. Form linear system.
    print!("Assembling: matrix ... ");
    let mut x = u_proj.clone();
    form_linear_system(&mut mat, &mut rhs, &mut x, &ess_bdr, &bc_vals);
    println!("done.");
    println!("Size of linear system: {}", n_dofs);

    // 11. Solve: PCG + AMS (default) or PCG + GSSmoother (-no-ams).
    if args.no_ams {
        let linlvo_sys = fem_linalg::fem_to_linlvo_csr(&mat);
        let precond = fem_solver::GSSmoother::from_csr(&linlvo_sys)
            .expect("GSSmoother setup");
        let result = solve_pcg(&mat, &rhs, &mut x, &precond, 1e-12, 2000, true)
            .expect("PCG+GSSmoother solve failed");
        println!("PCG+GSSmoother: {} iters, ||r||/||b|| = {:.3e}",
            result.iterations, result.final_residual);
    } else {
        use fem_solver::{solve_pcg_ams, AmsSolverConfig, AmsConfig};
        use fem_linalg::fem_to_linlvo_csr as ftl;
        let g = DiscreteLinearOperator::gradient(
            &fem_space::H1Space::new(space.mesh().clone(), 1),
            &space,
        ).expect("gradient assembly failed");
        let g_linlvo = ftl(&g);
        let result = solve_pcg_ams(&mat, &g_linlvo, &rhs, &mut x, &AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-12, atol: 1e-20, max_iter: 2000, verbose: true,
                ..SolverConfig::default()
            },
            ams_cfg: AmsConfig::default(),
        }).expect("PCG+AMS solve failed");
        println!("PCG+AMS: {} iters, ||r||/||b|| = {:.3e}",
            result.iterations, result.final_residual);
    }

    // 12. L² error (3D).
    let l2_err = l2_error_hcurl_exact_3d(
        space.mesh(), &space, &x, |x| exact_e_3d(x, kappa),
    );
    println!("\n|| E_h - E ||_{L^2} = {l2_err:.14e}\n");

    // 13. Save refined mesh + sol.gf (matching MFEM ex3 output).
    {
        write_mfem_file("refined.mesh", space.mesh()).expect("mesh write failed");
        write_mfem_gf_file("sol.gf", dim, &x, "ND", args.order, dim, 14).expect("sol write failed");
        eprintln!("  Wrote refined.mesh and sol.gf");
    }

    // 14. GLVis visualization.
    if args.visualization {
        match send_to_glvis_3d(space.mesh(), &space, &x, "E") {
            Ok(_) => eprintln!("  Sent solution to GLVis (localhost:19916)"),
            Err(e) => eprintln!("  GLVis not available: {e}"),
        }
    }
}

// ─── 3D source term: f = (1+κ²)·E_exact,  E = (sin κy, sin κz, sin κx) ────────

struct MaxwellSource3D { kappa: f64 }

impl VectorLinearIntegrator for MaxwellSource3D {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        let c = 1.0 + self.kappa * self.kappa;
        let fx = c * (self.kappa * x[1]).sin();
        let fy = c * (self.kappa * x[2]).sin();
        let fz = c * (self.kappa * x[0]).sin();
        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 3] * fx + qp.phi_vec[i * 3 + 1] * fy + qp.phi_vec[i * 3 + 2] * fz;
            f_elem[i] += qp.weight * dot;
        }
    }
}

// ─── 3D exact solution: E = (sin κy, sin κz, sin κx) ─────────────────────────

fn exact_e_3d(x: &[f64], kappa: f64) -> [f64; 3] {
    [(kappa * x[1]).sin(), (kappa * x[2]).sin(), (kappa * x[0]).sin()]
}

fn project_hcurl_exact_3d(
    space: &HCurlSpace<Mesh<3>>,
    kappa: f64,
    _quad_order: u8,
) -> Vec<f64> {
    space.interpolate_vector(&|x| exact_e_3d(x, kappa).to_vec()).into_vec()
}

// ─── 3D L² error (TetND1 / HexND1) ──────────────────────────────────────────

fn l2_error_hcurl_exact_3d<F>(mesh: &Mesh<3>, space: &HCurlSpace<Mesh<3>>,
                              uh: &[f64], exact: F) -> f64
where F: Fn(&[f64]) -> [f64; 3],
{
    use fem_element::nedelec::{TetND1, HexND1};
    let mut err2 = 0.0_f64;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re: &dyn VectorReferenceElement = match et {
            fem_mesh::element_type::ElementType::Hex8 => &HexND1,
            _ => &TetND1,
        };
        let nld = re.n_dofs();
        let q = re.quadrature(6);
        let mut phi = vec![0.0; nld * 3];
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (j, xp) = element_jacobian_at(mesh, e, xi, 3);
            let det = j.determinant();
            let w = q.weights[qi] * det.abs();
            let jit = j.try_inverse().unwrap_or_default().transpose();
            let mut uh_phys = [0.0_f64; 3];
            for a in 0..nld {
                let s = signs[a];
                for c in 0..3 {
                    let mut comp = 0.0_f64;
                    for k in 0..3 { comp += jit[(c, k)] * phi[a * 3 + k]; }
                    uh_phys[c] += s * uh[ed[a]] * comp;
                }
            }
            let ex = exact(&[xp[0], xp[1], xp[2]]);
            err2 += w * ((uh_phys[0] - ex[0]).powi(2)
                        + (uh_phys[1] - ex[1]).powi(2)
                        + (uh_phys[2] - ex[2]).powi(2));
        }
    }
    err2.sqrt()
}

// ─── 3D GLVis helper ─────────────────────────────────────────────────────────

fn send_to_glvis_3d(
    mesh: &Mesh<3>,
    space: &HCurlSpace<Mesh<3>>,
    u: &[f64],
    field_name: &str,
) -> std::io::Result<()> {
    use fem_element::nedelec::TetND1;
    let n_nodes = mesh.n_nodes() as usize;
    let re = TetND1;
    let n_ldofs = re.n_dofs();
    let mut sum_x = vec![0.0_f64; n_nodes];
    let mut sum_y = vec![0.0_f64; n_nodes];
    let mut sum_z = vec![0.0_f64; n_nodes];
    let mut count = vec![0u32; n_nodes];
    let mut phi = vec![0.0_f64; n_ldofs * 3];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let tet_verts: [[f64; 3]; 4] = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ];
        for vi in 0..4 {
            re.eval_basis_vec(&tet_verts[vi], &mut phi);
            let (j, _xp) = element_jacobian_at(mesh, e, &tet_verts[vi], 3);
            let jit = j.try_inverse().unwrap_or_default().transpose();
            let mut ex = 0.0_f64; let mut ey = 0.0_f64; let mut ez = 0.0_f64;
            for a in 0..n_ldofs {
                let s = signs[a];
                ex += s * u[dofs[a]] * (jit[(0, 0)] * phi[a*3] + jit[(0, 1)] * phi[a*3+1] + jit[(0, 2)] * phi[a*3+2]);
                ey += s * u[dofs[a]] * (jit[(1, 0)] * phi[a*3] + jit[(1, 1)] * phi[a*3+1] + jit[(1, 2)] * phi[a*3+2]);
                ez += s * u[dofs[a]] * (jit[(2, 0)] * phi[a*3] + jit[(2, 1)] * phi[a*3+1] + jit[(2, 2)] * phi[a*3+2]);
            }
            let nid = nodes[vi] as usize;
            sum_x[nid] += ex; sum_y[nid] += ey; sum_z[nid] += ez;
            count[nid] += 1;
        }
    }
    let mut vx = vec![0.0_f64; n_nodes];
    let mut vy = vec![0.0_f64; n_nodes];
    let mut vz = vec![0.0_f64; n_nodes];
    for i in 0..n_nodes {
        if count[i] > 0 {
            let inv = 1.0 / count[i] as f64;
            vx[i] = sum_x[i] * inv; vy[i] = sum_y[i] * inv; vz[i] = sum_z[i] * inv;
        }
    }
    let mut sock = fem_io::glvis::GlVisSocket::connect("localhost", 19916)?;
    sock.send_solution_3d_vector(mesh, &vx, &vy, &vz, field_name)?;
    Ok(())
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Args {
    mesh:          Option<String>,
    order:         u8,
    visualization: bool,
    freq:          f64,
    no_ams:        bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None, order: 1, visualization: true, freq: 1.0, no_ams: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-f" | "--frequency" => {
                a.freq = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            }
            "-no-ams" | "--no-ams" => { a.no_ams = true; }
            "-vis" | "--visualization" => { a.visualization = true; }
            "-no-vis" | "--no-visualization" => { a.visualization = false; }
            _ => {}
        }
    }
    a
}

// ─── Re-exports used by this file ────────────────────────────────────────────

use fem_linalg::CsrMatrix;
use fem_mesh::MeshTopology;
use fem_element::VectorReferenceElement as _;

// ─── Jacobian helper (local to this example) ────────────────────────────────

fn element_jacobian_at(mesh: &Mesh<3>, e: u32, xi: &[f64], _dim: usize)
    -> (nalgebra::DMatrix<f64>, [f64; 3])
{
    let nodes = mesh.element_nodes(e);
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let x3 = mesh.node_coords(nodes[3]);
    let (xi0, xi1, xi2) = (xi[0], xi[1], xi[2]);
    // ∂x/∂ξ columns:  ∂X/∂ξᵢ = Σ nodes[k] · ∂N_k/∂ξᵢ
    // Tet P1: N = [1-ξ-η-ζ, ξ, η, ζ]
    let j00 = -x0[0] + x1[0]; let j01 = -x0[0] + x2[0]; let j02 = -x0[0] + x3[0];
    let j10 = -x0[1] + x1[1]; let j11 = -x0[1] + x2[1]; let j12 = -x0[1] + x3[1];
    let j20 = -x0[2] + x1[2]; let j21 = -x0[2] + x2[2]; let j22 = -x0[2] + x3[2];
    let j = nalgebra::dmatrix![j00,j01,j02; j10,j11,j12; j20,j21,j22];
    let xp = [
        (1.0 - xi0 - xi1 - xi2) * x0[0] + xi0 * x1[0] + xi1 * x2[0] + xi2 * x3[0],
        (1.0 - xi0 - xi1 - xi2) * x0[1] + xi0 * x1[1] + xi1 * x2[1] + xi2 * x3[1],
        (1.0 - xi0 - xi1 - xi2) * x0[2] + xi0 * x1[2] + xi1 * x2[2] + xi2 * x3[2],
    ];
    (j, xp)
}
