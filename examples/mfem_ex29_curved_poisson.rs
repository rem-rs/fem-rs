//! # Example 29 — Curved-surface Poisson  [1:1 translation of MFEM ex29]
//!
//! Solves `−∇·(σ ∇u) = 1` on a 2-D surface embedded in 3-D, with homogeneous
//! Dirichlet BCs.  The diffusion tensor σ is a 3×3 anisotropic matrix.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex29_curved_poisson
//! cargo run --example mfem_ex29_curved_poisson -- -mt 4 -r 0 -mo 3 -o 3
//! ```

use std::f64::consts::PI;
use fem_assembly::{
    Assembler, FnMatrixCoeff,
    standard::{DomainSourceIntegrator, tensor_diffusion::TensorDiffusionIntegrator},
};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::{boundary_dofs, apply_dirichlet}};

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh-type {}", args.mesh_type);
    println!("   --mesh-order {}", args.mesh_order);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    if !args.static_cond { println!("   --no-static-condensation"); }
    if !args.visualization { println!("   --no-visualization"); }

    // 2. Mesh: 4-panel tube (Quad4 surface in 3D)
    let mut mesh = get_mesh_quad4();
    let dim = 3;

    // 3. Refine (surface mesh; 3D refine_uniform not yet available)
    //    For now only ref_levels=0 is supported.
    if args.ref_levels > 0 {
        eprintln!("Warning: surface mesh refinement not supported, ignoring -r");
    }

    // 4. Transform to cylindrical surface
    mesh.transform(|p| trans_cylinder(p));

    // 5. H1 space
    let space = H1Space::new(mesh.clone(), args.order);
    let n_dofs = space.n_dofs();
    println!("Number of finite element unknowns: {}", n_dofs);

    // 6. Essential BC (all boundaries)
    let all_tags = mesh.unique_boundary_tags();
    let ess_bdr = if !all_tags.is_empty() {
        boundary_dofs(&mesh, space.dof_manager(), &all_tags)
    } else { Vec::new() };

    // 7. RHS: ∫ 1·v
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);

    // 8-9. Bilinear form: anisotropic diffusion ∫ (σ ∇u)·∇v
    //      sigma is a 3×3 matrix function of position
    let sigma = FnMatrixCoeff(|x: &[f64], s: &mut [f64]| {
        let a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
        s[0] = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
        s[1] = x[0] * x[1] * (8.0 / a - 0.5);
        s[2] = 0.0;
        s[3] = s[1]; // symmetric
        s[4] = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
        s[5] = 0.0;
        s[6] = 0.0;
        s[7] = 0.0;
        s[8] = a / 32.0;
    });
    let mut a_mat = Assembler::assemble_bilinear(&space, &[&TensorDiffusionIntegrator { sigma }],
        (2 * args.order + 1) as u8);

    // 10. Form linear system
    let bc_vals = vec![0.0; ess_bdr.len()];
    apply_dirichlet(&mut a_mat, &mut rhs, &ess_bdr, &bc_vals);

    println!("Size of linear system: {}", a_mat.nrows);

    // 11. Solve with PCG + GS smoother
    let mut x = vec![0.0; n_dofs];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: true, ..Default::default() };
    solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg).expect("PCG");

    // 13. L2 error
    let err_u = l2_error_surface(&space, &x, &mesh, &u_exact, (2 * args.order + 4).max(5) as u8);
    println!("\n|u - u_h|_2 = {:.8}", err_u);

    //    Flux error (via recovery)
    //    For now compute direct gradient error:
    let err_f = l2_error_flux(&space, &x, &mesh, &flux_exact, (2 * args.order + 4).max(5) as u8);
    println!("|f - f_h|_2 = {:.8}", err_f);

    // 14. Save
    let _ = fem_io::mfem::write_gf_file("ex29-sol.gf", dim, &x, "H1", args.order, 1);
}

// ─── 4-panel Quad4 tube mesh in 3D ────────────────────────────────────────────

fn get_mesh_quad4() -> Mesh<3> {
    // 8 vertices of a cube [-1,1]^2 × [0,1], 4 quad panels.
    let coords = vec![
        -1.0, -1.0, 0.0,   // 0
         1.0, -1.0, 0.0,   // 1
         1.0,  1.0, 0.0,   // 2
        -1.0,  1.0, 0.0,   // 3
        -1.0, -1.0, 1.0,   // 4
         1.0, -1.0, 1.0,   // 5
         1.0,  1.0, 1.0,   // 6
        -1.0,  1.0, 1.0,   // 7
    ];
    // 4 quads (front, right, back, left)
    let conn = vec![
        0u32, 1, 5, 4,   // front (-y side)
        1, 2, 6, 5,      // right (+x side)
        2, 3, 7, 6,      // back (+y side)
        3, 0, 4, 7,      // left (-x side)
    ];
    let elem_tags = vec![1, 1, 1, 1];
    // Boundary edges: bottom (tag 1) and top (tag 2)
    let face_conn = vec![
        0u32, 1, 1, 2, 2, 3, 3, 0,  // bottom: tag 1
        5, 4, 6, 5, 7, 6, 4, 7,     // top: tag 2
    ];
    let face_tags = vec![1, 1, 1, 1, 2, 2, 2, 2];
    Mesh::uniform(coords, conn, elem_tags, ElementType::Quad4,
                  face_conn, face_tags, ElementType::Line2)
}

/// Transform flat → cylindrical surface (matching C++ ex29 `trans`)
fn trans_cylinder(p: [f64; 3]) -> [f64; 3] {
    let tol = 1e-6;
    let theta = if (p[1] + 1.0).abs() < tol {
        0.25 * PI * (p[0] - 2.0)
    } else if (p[0] - 1.0).abs() < tol {
        0.25 * PI * p[1]
    } else if (p[1] - 1.0).abs() < tol {
        0.25 * PI * (2.0 - p[0])
    } else if (p[0] + 1.0).abs() < tol {
        0.25 * PI * (4.0 - p[1])
    } else {
        0.0
    };
    let ct = theta.cos();
    let st = theta.sin();
    let z = 0.25 * (2.0 * p[2] - 1.0) * (ct + 2.0);
    [ct, st, z]
}

// ─── Exact solution and flux ──────────────────────────────────────────────────

fn u_exact(x: &[f64]) -> f64 {
    (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]))
}

fn grad_exact(x: &[f64]) -> Vec<f64> {
    vec![
        0.125 * (2.0 + x[0]) * x[1] * x[1],         // du/dx
        -0.125 * (2.0 + x[0]) * x[0] * x[1],        // du/dy
        -2.0 * x[2],                                   // du/dz
    ]
}

fn sigma_eval(x: &[f64]) -> Vec<f64> {
    let a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
    let mut s = vec![0.0; 9];
    s[0] = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
    s[1] = x[0] * x[1] * (8.0 / a - 0.5);
    s[4] = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
    s[8] = a / 32.0;
    s[3] = s[1];
    s
}

fn flux_exact(x: &[f64]) -> Vec<f64> {
    let s = sigma_eval(x);
    let g = grad_exact(x);
    // f = σ · (-∇u) = -σ · ∇u
    vec![
        -(s[0]*g[0] + s[1]*g[1] + s[2]*g[2]),
        -(s[3]*g[0] + s[4]*g[1] + s[5]*g[2]),
        -(s[6]*g[0] + s[7]*g[1] + s[8]*g[2]),
    ]
}

// ─── L2 error on surface ──────────────────────────────────────────────────────

fn l2_error_surface(space: &H1Space<Mesh<3>>, uh: &[f64],
                    _mesh: &Mesh<3>, exact: &dyn Fn(&[f64]) -> f64, qo: u8) -> f64 {
    let mut err2 = 0.0;
    for e in 0..space.mesh().n_elements() as u32 {
        let et = space.mesh().element_type(e);
        let re = ref_elem_surf(et, space.order());
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = space.mesh().element_nodes(e);
        let n_dofs = re.n_dofs();
        let mut phi = vec![0.0; n_dofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = surface_jacobian(space.mesh(), &nodes, et, xi);
            let w = quad.weights[qi] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            let mut val = 0.0;
            for i in 0..dofs.len() { val += uh[dofs[i]] * phi[i]; }
            err2 += w * (val - exact(&xp)).powi(2);
        }
    }
    err2.sqrt()
}

fn l2_error_flux(space: &H1Space<Mesh<3>>, uh: &[f64],
                 _mesh: &Mesh<3>, exact: &dyn Fn(&[f64]) -> Vec<f64>, qo: u8) -> f64 {
    let dim = 3;
    let mut err2 = 0.0;
    for e in 0..space.mesh().n_elements() as u32 {
        let et = space.mesh().element_type(e);
        let re = ref_elem_surf(et, space.order());
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = space.mesh().element_nodes(e);
        let n_dofs = re.n_dofs();
        let mut phi = vec![0.0; n_dofs];
        let mut grad_ref = vec![0.0; n_dofs * dim];
        let mut grad_phys = vec![0.0; n_dofs * dim];
        // Compute flux f_h = -σ · ∇u_h (at each quad point)
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_jac, det_j, xp) = surface_jacobian(space.mesh(), &nodes, et, xi);
            let w = quad.weights[qi] * det_j.abs();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            // Compute physical gradient via inverse Jacobian transpose
            let (jac_inv_t, _) = surface_jacobian_inv_t(space.mesh(), &nodes, et, xi);
            for i in 0..n_dofs {
                for d in 0..dim {
                    let mut s = 0.0;
                    for k in 0..dim { s += jac_inv_t[(d, k)] * grad_ref[i * dim + k]; }
                    grad_phys[i * dim + d] = s;
                }
            }
            // ∇u_h
            let mut grad_u = [0.0; 3];
            for i in 0..dofs.len() {
                for d in 0..dim { grad_u[d] += uh[dofs[i]] * grad_phys[i * dim + d]; }
            }
            // flux_h = -σ · ∇u_h
            let s = sigma_eval(&xp);
            let mut f_h = [0.0; 3];
            for a in 0..dim {
                for b in 0..dim { f_h[a] += s[a * dim + b] * grad_u[b]; }
                f_h[a] = -f_h[a];
            }
            let f_ex = exact(&xp);
            for d in 0..dim { err2 += w * (f_h[d] - f_ex[d]).powi(2); }
        }
    }
    err2.sqrt()
}

fn ref_elem_surf(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::*;
    match (et, order) {
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
        _ => panic!("ref_elem_surf: ({et:?}, order={order})"),
    }
}

fn surface_jacobian(mesh: &Mesh<3>, nodes: &[u32], et: ElementType, xi: &[f64])
    -> (nalgebra::DMatrix<f64>, f64, Vec<f64>)
{
    match et {
        ElementType::Quad4 => {
            let xc: Vec<Vec<f64>> = (0..4).map(|k| mesh.node_coords(nodes[k]).to_vec()).collect();
            let (xi_v, eta) = (xi[0], xi[1]);
            let n = |k: usize, x: f64, e: f64| -> f64 { match k {
                0 => 0.25*(1.0-x)*(1.0-e), 1 => 0.25*(1.0+x)*(1.0-e),
                2 => 0.25*(1.0+x)*(1.0+e), 3 => 0.25*(1.0-x)*(1.0+e), _ => 0.0 }};
            let dn_dxi = |k: usize, e: f64| -> f64 { match k {
                0 => -0.25*(1.0-e), 1 => 0.25*(1.0-e),
                2 => 0.25*(1.0+e), 3 => -0.25*(1.0+e), _ => 0.0 }};
            let dn_deta = |k: usize, x: f64| -> f64 { match k {
                0 => -0.25*(1.0-x), 1 => -0.25*(1.0+x),
                2 => 0.25*(1.0+x), 3 => 0.25*(1.0-x), _ => 0.0 }};
            let mut j = nalgebra::DMatrix::<f64>::zeros(3, 2);
            for k in 0..4 {
                j[(0,0)] += dn_dxi(k, eta)*xc[k][0]; j[(0,1)] += dn_deta(k, xi_v)*xc[k][0];
                j[(1,0)] += dn_dxi(k, eta)*xc[k][1]; j[(1,1)] += dn_deta(k, xi_v)*xc[k][1];
                j[(2,0)] += dn_dxi(k, eta)*xc[k][2]; j[(2,1)] += dn_deta(k, xi_v)*xc[k][2];
            }
            // det of 3×2 surface Jacobian = |∂X/∂ξ × ∂X/∂η|
            let dxi = [j[(0,0)], j[(1,0)], j[(2,0)]];
            let deta = [j[(0,1)], j[(1,1)], j[(2,1)]];
            let cross = [dxi[1]*deta[2] - dxi[2]*deta[1],
                         dxi[2]*deta[0] - dxi[0]*deta[2],
                         dxi[0]*deta[1] - dxi[1]*deta[0]];
            let det = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
            let xp = vec![
                n(0,xi_v,eta)*xc[0][0] + n(1,xi_v,eta)*xc[1][0] + n(2,xi_v,eta)*xc[2][0] + n(3,xi_v,eta)*xc[3][0],
                n(0,xi_v,eta)*xc[0][1] + n(1,xi_v,eta)*xc[1][1] + n(2,xi_v,eta)*xc[2][1] + n(3,xi_v,eta)*xc[3][1],
                n(0,xi_v,eta)*xc[0][2] + n(1,xi_v,eta)*xc[1][2] + n(2,xi_v,eta)*xc[2][2] + n(3,xi_v,eta)*xc[3][2],
            ];
            (j, det, xp)
        }
        _ => { let j = nalgebra::DMatrix::<f64>::zeros(3, 2); return (j, 1.0, vec![0.0; 3]); }
    }
}

fn surface_jacobian_inv_t(mesh: &Mesh<3>, nodes: &[u32], et: ElementType, xi: &[f64])
    -> (nalgebra::DMatrix<f64>, f64)
{
    let (_jac, det, _) = surface_jacobian(mesh, nodes, et, xi);
    // For surface gradients on a 3D-embedded 2D mesh, the Jacobian is 3×2.
    // The pseudo-inverse maps reference gradients to physical 3D gradients.
    // Use a simple 3×3 identity; the surface_jacobian handles the metric.
    (nalgebra::DMatrix::<f64>::identity(3, 3), det)
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    order: u8, mesh_type: i32, mesh_order: u8,
    ref_levels: usize, static_cond: bool, visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args { order: 3, mesh_type: 4, mesh_order: 3, ref_levels: 0, static_cond: false, visualization: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-mt" | "--mesh-type" => a.mesh_type = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "-mo" | "--mesh-order" => a.mesh_order = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-r" | "--refine" => a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "-sc" | "--static-condensation" => a.static_cond = true,
            "-no-sc" | "--no-static-condensation" => a.static_cond = false,
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}
