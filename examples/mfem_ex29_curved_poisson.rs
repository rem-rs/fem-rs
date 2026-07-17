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

#![allow(dead_code)]
use std::f64::consts::PI;
use fem_element::ReferenceElement;
use fem_linalg::CooMatrix;
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

    // 7-9. Assemble system manually (surface mesh needs 3×2 Jacobian)
    let qo = (2 * order + 1).max(3 + 3) as u8; // extra quadrature for curved geometry
    let mut a_coo = CooMatrix::new(n_dofs, n_dofs);
    let mut rhs = vec![0.0; n_dofs];

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, order);
        let n_ldofs = re.n_dofs();
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let ng = dofs.len();
        let _nodes = mesh.element_nodes(e);

        let mut me = vec![0.0; ng * ng];
        let mut fe = vec![0.0; ng];
        let mut phi = vec![0.0; n_ldofs];
        let mut grad_ref = vec![0.0; n_ldofs * 2]; // 2 reference coordinates
        let mut grad_phys = vec![0.0; n_ldofs * 3]; // 3 physical coordinates

        for (qi, xi) in quad.points.iter().enumerate() {
            // Isoparametric surface Jacobian: 3×2 matrix and determinant
            let (j, det_j, xp) = surface_jacobian(&mesh, e, et, xi);
            let w = quad.weights[qi] * det_j;
            let (j00, j01, j10, j11, j20, j21) =
                (j[(0,0)], j[(0,1)], j[(1,0)], j[(1,1)], j[(2,0)], j[(2,1)]);

            // Covariant metric: g = J^T * J (2×2)
            let g00 = j00*j00 + j10*j10 + j20*j20;
            let g01 = j00*j01 + j10*j11 + j20*j21;
            let g11 = j01*j01 + j11*j11 + j21*j21;
            let det_g = g00*g11 - g01*g01;
            // Inverse metric (for contravariant gradient transformation)
            let gi00 = g11 / det_g;
            let gi01 = -g01 / det_g;
            let gi11 = g00 / det_g;

            // Map reference gradient to surface gradient: ∇_phys = J * g⁻¹ * J^T * ∇_ref
            // First compute physical gradient contribution per DOF
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            for i in 0..ng {
                let gi = &grad_ref[i*2..i*2+2];
                // ∇_surf φ_i = J * g⁻¹ * ∇_ref φ_i  (3 components)
                // First compute g⁻¹ * ∇_ref (2 components)
                let t0 = gi00 * gi[0] + gi01 * gi[1];
                let t1 = gi01 * gi[0] + gi11 * gi[1];
                // Then J * result (3 components)
                grad_phys[i*3]   = j00*t0 + j01*t1;
                grad_phys[i*3+1] = j10*t0 + j11*t1;
                grad_phys[i*3+2] = j20*t0 + j21*t1;
            }

            // RHS contribution: ∫ 1·v = w * φ_i (sum_i)
            for i in 0..ng { fe[i] += w * phi[i]; }

            // Stiffness: ∫ σ ∇φ_i · ∇φ_j  (3×3 sigma)
            let mut sigma9 = [0.0; 9];
            sigma_func(&xp, &mut sigma9);
            for i in 0..ng {
                for j in 0..ng {
                    let mut val = 0.0;
                    for a in 0..3 {
                        let ga = grad_phys[i*3 + a];
                        if ga == 0.0 { continue; }
                        for b in 0..3 {
                            val += ga * sigma9[a*3 + b] * grad_phys[j*3 + b];
                        }
                    }
                    me[i*ng + j] += w * val;
                }
            }
        }

        // (geometry handled by surface_jacobian)

        for (ir, &r) in dofs.iter().enumerate() {
            rhs[r] += fe[ir];
            for (ic, &c) in dofs.iter().enumerate() {
                let v = me[ir*ng + ic];
                if v != 0.0 { a_coo.add(r, c, v); }
            }
        }
    }

    let mut a_mat = a_coo.into_csr();

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

    // Compute surface area for verification
    let mut total_area = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, order);
        let quad = re.quadrature(qo);
        let _nodes = mesh.element_nodes(e);
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, _) = surface_jacobian(&mesh, e, et, xi);
            total_area += quad.weights[qi] * det_j;
        }
    }
    println!("  Surface area = {:.6e}", total_area);

    // 11. PCG + GS smoother
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200, verbose: true, ..Default::default() };
    let mut x = vec![0.0; n_dofs];
    solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg).expect("PCG");

    // Debug: assemble L2 mass matrix and compute L2 projection error of exact solution
    let mut m_coo = CooMatrix::new(n_dofs, n_dofs);
    let mut rhs_proj = vec![0.0; n_dofs];
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, order);
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let _nodes = mesh.element_nodes(e);
        let ng = dofs.len();
        let n_ldofs = re.n_dofs();
        let mut me = vec![0.0; ng*ng];
        let mut fe = vec![0.0; ng];
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, xp) = surface_jacobian(&mesh, e, et, xi);
            let w = quad.weights[qi] * det_j;
            re.eval_basis(xi, &mut phi);
            let ev = u_exact(&xp);
            for i in 0..ng {
                fe[i] += w * ev * phi[i];
                for j in 0..ng { me[i*ng + j] += w * phi[i] * phi[j]; }
            }
        }
        for (ir, &r) in dofs.iter().enumerate() {
            rhs_proj[r] += fe[ir];
            for (ic, &c) in dofs.iter().enumerate() {
                let v = me[ir*ng + ic];
                if v != 0.0 { m_coo.add(r, c, v); }
            }
        }
    }
    let mut m_mat = m_coo.into_csr();
    for &d in &ess_bdr {
        let mut dummy = vec![0.0; n_dofs];
        m_mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
        if let Some(k) = m_mat.find_entry(d as usize, d as usize) { m_mat.values[k] = 1.0; }
        rhs_proj[d as usize] = 0.0;
    }
    let mut u_proj = vec![0.0; n_dofs];
    solve_pcg_gssmoother(&m_mat, &rhs_proj, &mut u_proj, &cfg).expect("PCG proj");
    let err_proj = l2_error_surface(&mesh, &space, &u_proj, &u_exact, (2*order+4).max(5) as u8);
    println!("  L2 projection error = {:.8}", err_proj);
    println!("  (Zero means exact solution is representable in P3 on this mesh)");

    // Verify stiffness matrix: A * u_exact should equal rhs (u_exact solves PDE exactly)
    let dm = space.dof_manager();
    let mut u_ex = vec![0.0; n_dofs];
    for d in 0..n_dofs { u_ex[d] = u_exact(dm.dof_coord(d as u32)); }
    for &d in &ess_bdr { u_ex[d as usize] = 0.0; }
    let mut auex = vec![0.0; n_dofs];
    a_mat.spmv(&u_ex, &mut auex);
    let res_stiff: f64 = (0..n_dofs).map(|i| (rhs[i]-auex[i]).powi(2)).sum::<f64>().sqrt();
    let b_n: f64 = rhs.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||b - A*u_exact|| = {:.6e} (rel {:.6e})", res_stiff, res_stiff / b_n.max(1e-30));

    // Debug A*u_exact vs rhs for key DOFs
    println!("\n  RHS vs A*u_exact (first 12 DOFs):");
    for d in 0..12.min(n_dofs) {
        println!("    dof[{d:2}] rhs={:.6e} (A*uex)[{d}]={:.6e} diff={:.6e}",
                 rhs[d], auex[d], rhs[d]-auex[d]);
    }

    // 13. L2 error (high quadrature for accuracy)
    let err_qo = (2 * order + 6).max(8) as u8; // high quadrature for curved geometry
    let err_u = l2_error_surface(&mesh, &space, &x, &u_exact, err_qo);
    println!("\n|u - u_h|_2 = {:.8}", err_u);

    let err_f = l2_error_flux(&mesh, &space, &x, &flux_exact, err_qo);
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
    quad.eval_basis(xi, &mut phi);
    quad.eval_grad_basis(xi, &mut grad_ref);

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
    use fem_element::lagrange::quad::*;
    match (et, order) {
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
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

// ─── Assemble L2 projection RHS: ∫ u_exact · φ_i dS ───────────────────────────

fn assemble_rhs_projection(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>,
                           exact: &dyn Fn(&[f64]) -> f64, qo: u8) -> Vec<f64> {
    let n = space.n_dofs();
    let mut rhs = vec![0.0; n];
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
            let ev = exact(&xp);
            for i in 0..dofs.len() { rhs[dofs[i]] += w * ev * phi[i]; }
        }
    }
    rhs
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

fn l2_error_flux(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, uh: &[f64],
                 exact: &dyn Fn(&[f64]) -> Vec<f64>, qo: u8) -> f64 {
    let dim = 3;
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re = ref_elem_for(et, space.order());
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let _nodes = mesh.element_nodes(e);
        let n_ldofs = re.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        let mut grad_ref = vec![0.0; n_ldofs*2];
        let mut grad_phys = vec![0.0; n_ldofs*dim];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (j, det_j, xp) = surface_jacobian(mesh, e, et, xi);
            let w = quad.weights[qi] * det_j;
            let (j00,j01,j10,j11,j20,j21) = (j[(0,0)],j[(0,1)],j[(1,0)],j[(1,1)],j[(2,0)],j[(2,1)]);

            // Metric + inverse
            let g00 = j00*j00+j10*j10+j20*j20;
            let g01 = j00*j01+j10*j11+j20*j21;
            let g11 = j01*j01+j11*j11+j21*j21;
            let det_g = g00*g11 - g01*g01;
            let (gi00, gi01, gi11) = (g11/det_g, -g01/det_g, g00/det_g);

            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            for i in 0..n_ldofs {
                let gr = &grad_ref[i*2..i*2+2];
                let t0 = gi00*gr[0] + gi01*gr[1];
                let t1 = gi01*gr[0] + gi11*gr[1];
                grad_phys[i*dim]   = j00*t0 + j01*t1;
                grad_phys[i*dim+1] = j10*t0 + j11*t1;
                grad_phys[i*dim+2] = j20*t0 + j21*t1;
            }

            // ∇u_h
            let mut g = [0.0; 3];
            for i in 0..n_ldofs {
                for d in 0..dim { g[d] += uh[dofs[i]] * grad_phys[i*dim+d]; }
            }

            // flux_h = -σ · ∇u_h
            let mut s = [0.0; 9];
            sigma_func(&xp, &mut s);
            let mut fh = [0.0; 3];
            for a in 0..dim {
                for b in 0..dim { fh[a] -= s[a*dim+b] * g[b]; }
            }
            let fe = exact(&xp);
            for d in 0..dim { err2 += w * (fh[d] - fe[d]).powi(2); }
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
