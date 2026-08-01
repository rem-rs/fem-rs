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
use fem_assembly::assembler::Assembler;
use fem_assembly::postproc::coefficient::FnMatrixCoeff;
use fem_assembly::standard::{DomainSourceIntegrator, TensorDiffusionIntegrator};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh-type {}", args.mesh_type);
    println!("   --mesh-order {}", args.mesh_order);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    if !args.static_cond { println!("   --no-static-condensation"); }
    if !args.visualization { println!("   --no-visualization"); }

    // 2. Mesh: 4-panel Quad4 tube in 3D
    let mesh = get_mesh_quad4();
    // 3. Refine (only ref_levels=0 supported for surface meshes)
    if args.ref_levels > 0 { eprintln!("Warning: surface refinement not supported"); }

    // 4. Transform to cylindrical surface with isoparametric geometry
    let mesh_order = args.mesh_order;
    let mut mesh = mesh;
    mesh.set_curvature(mesh_order);
    mesh.transform(|p| trans_cylinder(p));

    println!("  Geometry order = {}", mesh_order);
    println!("  Mesh nodes     = {} (vertices) + {} (geom)",
             mesh.n_nodes(), mesh.n_geom_nodes().saturating_sub(mesh.n_nodes()));

    // 5. H1 space
    let order = args.order;
    let space = H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    println!("Number of finite element unknowns: {}", n_dofs);

    // 6. Essential BCs (Dirichlet on all boundaries)
    let all_tags = mesh.unique_boundary_tags();
    let ess_bdr = if !all_tags.is_empty() {
        boundary_dofs(&mesh, space.dof_manager(), &all_tags)
    } else { Vec::new() };

    // 7-9. Assemble with the core-library surface integrator path.
    // The assembler's is_surface branch computes the true 3-D tangential
    // gradient ∇_surf φ = J·G⁻¹·∇_ref φ and the 3×3 anisotropic σ is applied
    // by TensorDiffusionIntegrator (1:1 with MFEM ex29's DiffusionIntegrator).
    let qo = (2 * order + 1).max(3 + 3) as u8; // extra quadrature for curved geometry
    let sigma = FnMatrixCoeff(|x: &[f64], s: &mut [f64]| {
        let mut s9 = [0.0; 9];
        sigma_func(x, &mut s9);
        s.copy_from_slice(&s9);
    });
    let diff = TensorDiffusionIntegrator { sigma };
    let src = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
    let mut a_mat = Assembler::assemble_bilinear(&space, &[&diff], qo);
    let mut rhs = Assembler::assemble_linear(&space, &[&src], qo);

    // 10. Dirichlet BCs
    for &d in &ess_bdr {
        let mut dummy = vec![0.0; n_dofs];
        a_mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
        // Fix diagonal for CG
        if let Some(k) = a_mat.find_entry(d as usize, d as usize) {
            a_mat.values[k] = 1.0;
        }
        rhs[d as usize] = 0.0;
    }

    println!("Size of linear system: {}", a_mat.nrows);

    // 11. PCG + GS smoother
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: true, ..Default::default() };
    let mut x = vec![0.0; n_dofs];
    solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg).expect("PCG");

    // 13. L2 error (MFEM ComputeL2Error uses intorder = 2*order + 3)
    let err_qo = (2 * order + 3) as u8; // = 9 for order 3 (matches C++ 5-point Gauss)
    let err_u = l2_error_surface(&mesh, &space, &x, &u_exact, err_qo);
    println!("\n|u - u_h|_2 = {:.8}", err_u);

    let err_f = l2_error_flux_gf(&mesh, &space, &compute_flux_projected(&mesh, &space, &x, order), &flux_exact, err_qo);
    println!("|f - f_h|_2 = {:.8}", err_f);
}

// ─── 4-panel Quad4 surface mesh ───────────────────────────────────────────────

fn get_mesh_quad4() -> Mesh<3> {
    let coords = vec![
        -1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0,  1.0, 0.0,  -1.0,  1.0, 0.0,
        -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0,  1.0, 1.0,  -1.0,  1.0, 1.0,
    ];
    let conn = vec![0u32,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7];
    let elem_tags = vec![1; 4];
    let face_conn = vec![0u32,1,1,2,2,3,3,0,5,4,6,5,7,6,4,7];
    let face_tags = vec![1,1,1,1,2,2,2,2];
    Mesh::uniform(coords, conn, elem_tags, ElementType::Quad4,
                  face_conn, face_tags, ElementType::Line2)
}

fn trans_cylinder(p: [f64; 3]) -> [f64; 3] {
    let tol = 1e-6;
    let theta = if (p[1] + 1.0).abs() < tol { 0.25*PI*(p[0] - 2.0) }
    else if (p[0] - 1.0).abs() < tol { 0.25*PI*p[1] }
    else if (p[1] - 1.0).abs() < tol { 0.25*PI*(2.0 - p[0]) }
    else if (p[0] + 1.0).abs() < tol { 0.25*PI*(4.0 - p[1]) }
    else { 0.0 };
    let (ct, st) = (theta.cos(), theta.sin());
    [ct, st, 0.25*(2.0*p[2] - 1.0)*(ct + 2.0)]
}

// ─── Surface Jacobian — isoparametric via geometry nodes ──────────────────────
//
// Uses the mesh's high-order geometry nodes (created via SetCurvature) with
// QuadQk basis functions to compute the 3×2 surface Jacobian.

fn surface_jacobian(mesh: &Mesh<3>, e: u32, _et: ElementType, xi: &[f64])
    -> (nalgebra::DMatrix<f64>, f64, Vec<f64>)
{
    use fem_element::lagrange::factory::QuadQk;
    use fem_element::lagrange::QuadQ1;
    use fem_element::ReferenceElement;

    let geom_order = mesh.geom_order();
    let (nodes, n_dofs, quad): (&[NodeId], usize, Box<dyn ReferenceElement>) = if geom_order > 1 {
        let n = mesh.geometry_nodes(e);
        let q: Box<dyn ReferenceElement> = Box::new(QuadQk::new(geom_order as usize));
        let len = n.len();
        (n, len, q)
    } else {
        // Linear geometry: use regular element nodes with Q1 basis
        let n = mesh.elem_nodes(e);
        let q: Box<dyn ReferenceElement> = Box::new(QuadQ1);
        (n, 4, q)
    };

    let mut phi = vec![0.0; n_dofs];
    let mut grad_ref = vec![0.0; n_dofs * 2];
    // QuadQk expects [0,1]²; QuadQ1 expects [-1,1]².
    // For geom_order <= 1, map xi from [0,1]→[-1,1] before passing to QuadQ1.
    if geom_order > 1 {
        quad.eval_basis(xi, &mut phi);
        quad.eval_grad_basis(xi, &mut grad_ref);
    } else {
        // Map point to [-1,1]² for QuadQ1
        let xi_mapped = [2.0 * xi[0] - 1.0, 2.0 * xi[1] - 1.0];
        quad.eval_basis(&xi_mapped, &mut phi);
        quad.eval_grad_basis(&xi_mapped, &mut grad_ref);
        // Chain rule: d/dx ∈ [0,1] = 2 · d/dξ ∈ [-1,1]
        for g in grad_ref.iter_mut() { *g *= 2.0; }
    }

    // Helper: get node coords from geometry (if available) or regular coords
    use fem_core::NodeId;
    let get_coords = |gid: NodeId| -> [f64; 3] {
        if geom_order > 1 {
            let c = mesh.geom_coords_of(gid);
            [c[0], c[1], c[2]]
        } else {
            mesh.coords_of(gid)
        }
    };

    // Physical position
    let mut xp = [0.0; 3];
    for k in 0..n_dofs {
        let xk = get_coords(nodes[k]);
        for d in 0..3 { xp[d] += xk[d] * phi[k]; }
    }

    // 3×2 Jacobian
    let mut j = nalgebra::DMatrix::<f64>::zeros(3, 2);
    for k in 0..n_dofs {
        let xk = get_coords(nodes[k]);
        j[(0,0)] += xk[0] * grad_ref[k*2];
        j[(1,0)] += xk[1] * grad_ref[k*2];
        j[(2,0)] += xk[2] * grad_ref[k*2];
        j[(0,1)] += xk[0] * grad_ref[k*2+1];
        j[(1,1)] += xk[1] * grad_ref[k*2+1];
        j[(2,1)] += xk[2] * grad_ref[k*2+1];
    }

    let (j00,j01,j10,j11,j20,j21) = (j[(0,0)],j[(0,1)],j[(1,0)],j[(1,1)],j[(2,0)],j[(2,1)]);
    let g00 = j00*j00 + j10*j10 + j20*j20;
    let g01 = j00*j01 + j10*j11 + j20*j21;
    let g11 = j01*j01 + j11*j11 + j21*j21;
    let det = (g00*g11 - g01*g01).sqrt();

    (j, det, xp.to_vec())
}

fn ref_elem_for(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Quad4, _) => Box::new(fem_element::lagrange::factory::QuadQk::new(order as usize)),
        _ => panic!("ref_elem: ({et:?}, order={order})"),
    }
}

// ─── Exact solution and flux ──────────────────────────────────────────────────

fn u_exact(x: &[f64]) -> f64 {
    (0.25*(2.0+x[0]) - x[2]) * (x[2] + 0.25*(2.0+x[0]))
}

fn du_exact(x: &[f64]) -> Vec<f64> {
    vec![0.125*(2.0+x[0])*x[1]*x[1], -0.125*(2.0+x[0])*x[0]*x[1], -2.0*x[2]]
}

fn sigma_func(x: &[f64], s: &mut [f64; 9]) {
    let a = 17.0 - 2.0*x[0]*(1.0+x[0]);
    s[0] = 0.5 + x[0]*x[0]*(8.0/a - 0.5);
    s[1] = x[0]*x[1]*(8.0/a - 0.5); s[3] = s[1];
    s[4] = 0.5*x[0]*x[0] + 8.0*x[1]*x[1]/a;
    s[8] = a/32.0;
    // s[2]=s[5]=s[6]=s[7]=0 already from initialization
}

fn flux_exact(x: &[f64]) -> Vec<f64> {
    let mut s = [0.0; 9];
    sigma_func(x, &mut s);
    let g = du_exact(x);
    vec![-(s[0]*g[0]+s[1]*g[1]), -(s[3]*g[0]+s[4]*g[1]), -s[8]*g[2]]
}

// ─── L2 error on surface ──────────────────────────────────────────────────────

fn l2_error_surface(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, uh: &[f64],
                    exact: &dyn Fn(&[f64]) -> f64, qo: u8) -> f64 {
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, space.order());
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let _nodes = mesh.element_nodes(e);
        let n_ldofs = re.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, xp) = surface_jacobian(mesh, e, et, xi);
            let w = quad.weights[qi] * det_j;
            re.eval_basis(xi, &mut phi);
            let val: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| uh[d] * p).sum();
            err2 += w * (val - exact(&xp)).powi(2);
        }
    }
    err2.sqrt()
}

// ─── Flux recovery: MFEM ComputeFlux equivalent ──────────────────────────────
//
// MFEM's GridFunction::ComputeFlux(integrator, flux) evaluates −σ·∇u_h at the
// flux-space FE nodes (the same GLL dof positions as u_h), sums element
// contributions into the global flux grid function and averages by visit
// count (SumFluxAndCount).  The flux grid function is then interpolated for
// the L2 error, exactly like C++ ex29's |f - f_h|_2.

/// Compute the averaged flux grid function `-σ·∇u_h` (layout [dof][comp],
/// byNODES = block, same as MFEM H1 with 3 vdims).
fn compute_flux_projected(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, uh: &[f64],
                          order: u8) -> Vec<f64> {
    let n_dofs = space.n_dofs();
    let re = ref_elem_for(ElementType::Quad4, order);
    let n_ldofs = re.n_dofs();
    let flux_dof_coords = re.dof_coords(); // GLL positions on [0,1]²
    let mut flux = vec![0.0; n_dofs * 3];
    let mut count = vec![0u32; n_dofs * 3];
    let mut phi = vec![0.0; n_ldofs];
    let mut grad_ref = vec![0.0; n_ldofs * 2];

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let g = mesh.geometry_nodes(e);
        for k in 0..n_ldofs {
            // Evaluate −σ(x)·∇u_h at flux dof k (physical position of geometry node k)
            let xi = &flux_dof_coords[k];
            let (j, det_j, xp) = surface_jacobian(mesh, e, et, xi);
            let _ = det_j; // pointwise evaluation: measure not needed
            let (j00, j01, j10, j11, j20, j21) =
                (j[(0,0)], j[(0,1)], j[(1,0)], j[(1,1)], j[(2,0)], j[(2,1)]);
            let g00 = j00*j00 + j10*j10 + j20*j20;
            let g01 = j00*j01 + j10*j11 + j20*j21;
            let g11 = j01*j01 + j11*j11 + j21*j21;
            let det_g = g00*g11 - g01*g01;
            let (gi00, gi01, gi11) = (g11/det_g, -g01/det_g, g00/det_g);

            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            let mut gu = [0.0; 3];
            for i in 0..n_ldofs {
                let gr = &grad_ref[i*2..i*2+2];
                let t0 = gi00*gr[0] + gi01*gr[1];
                let t1 = gi01*gr[0] + gi11*gr[1];
                gu[0] += uh[dofs[i]] * (j00*t0 + j01*t1);
                gu[1] += uh[dofs[i]] * (j10*t0 + j11*t1);
                gu[2] += uh[dofs[i]] * (j20*t0 + j21*t1);
            }
            let mut s9 = [0.0; 9];
            sigma_func(&xp, &mut s9);
            let base = dofs[k] * 3;
            for c in 0..3 {
                flux[base + c] -= s9[c*3]*gu[0] + s9[c*3+1]*gu[1] + s9[c*3+2]*gu[2];
                count[base + c] += 1;
            }
        }
    }
    for i in 0..flux.len() {
        if count[i] > 0 { flux[i] /= count[i] as f64; }
    }
    flux
}

/// L2 error of an averaged flux grid function vs the exact flux (MFEM
/// ComputeL2Error: intorder = 2*order + 3, interpolating the flux GF).
fn l2_error_flux_gf(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, flux: &[f64],
                    exact: &dyn Fn(&[f64]) -> Vec<f64>, qo: u8) -> f64 {
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, space.order());
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_ldofs = re.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, xp) = surface_jacobian(mesh, e, et, xi);
            let w = quad.weights[qi] * det_j;
            re.eval_basis(xi, &mut phi);
            let mut fh = [0.0; 3];
            for i in 0..dofs.len() {
                for d in 0..3 { fh[d] += flux[dofs[i]*3 + d] * phi[i]; }
            }
            let fe = exact(&xp);
            for d in 0..3 { err2 += w * (fh[d] - fe[d]).powi(2); }
        }
    }
    err2.sqrt()
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args { order: u8, mesh_type: i32, mesh_order: u8, ref_levels: usize,
              static_cond: bool, visualization: bool }

fn parse_args() -> Args {
    let mut a = Args { order: 3, mesh_type: 4, mesh_order: 3, ref_levels: 0,
                       static_cond: false, visualization: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o"|"--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-mt"|"--mesh-type" => a.mesh_type = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "-mo"|"--mesh-order" => a.mesh_order = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "-r"|"--refine" => a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "-sc"|"--static-condensation" => a.static_cond = true,
            "-no-sc"|"--no-static-condensation" => a.static_cond = false,
            "-vis"|"--visualization" => a.visualization = true,
            "-no-vis"|"--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}
