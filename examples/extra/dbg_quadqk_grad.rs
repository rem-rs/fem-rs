//! Debug: verify QuadQk gradient vs QuadQ1 on a flat 2D mesh.
//! Tests that A * u_exact = rhs for the standard Laplacian on a unit square.
#![allow(dead_code)]

use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_element::{ReferenceElement, lagrange::QuadQ3};
use fem_linalg::CooMatrix;
use fem_mesh::{Mesh, topology::MeshTopology, ElementTransformation};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

fn main() {
    // Flat unit square, 1 Quad4 element
    let mesh = Mesh::<2>::unit_square_quad(1);
    let order = 3u8;
    let space = H1Space::new(mesh.clone(), order);
    let n_dofs = space.n_dofs();
    println!("DOFs: {}", n_dofs);

    // Exact solution: u = sin(πx)sin(πy) (homogeneous Dirichlet)
    let u_ex = |x: &[f64]| (std::f64::consts::PI*x[0]).sin() * (std::f64::consts::PI*x[1]).sin();
    let f_src = |x: &[f64]| 2.0*std::f64::consts::PI*std::f64::consts::PI * u_ex(x); // -Δu = 2π²u

    // Assemble using standard Assembler (2D, not surface)
    let mut a_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f_src)], 5);

    // Apply Dirichlet BCs (all boundaries)
    let all_tags = mesh.unique_boundary_tags();
    let ess_bdr = boundary_dofs(&mesh, space.dof_manager(), &all_tags);
    let mut rhs2 = rhs.clone();
    for &d in &ess_bdr {
        let mut dummy = vec![0.0; n_dofs];
        a_mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
        if let Some(k) = a_mat.find_entry(d as usize, d as usize) { a_mat.values[k] = 1.0; }
        rhs2[d as usize] = 0.0;
    }

    // Solve
    let mut x = vec![0.0; n_dofs];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 500, verbose: false, ..Default::default() };
    solve_cg(&a_mat, &rhs2, &mut x, &cfg).expect("CG");

    // Compute exact at DOFs
    let dm = space.dof_manager();
    let mut u_ex_vec = vec![0.0; n_dofs];
    for d in 0..n_dofs { u_ex_vec[d] = u_ex(dm.dof_coord(d as u32)); }
    for &d in &ess_bdr { u_ex_vec[d as usize] = 0.0; }

    // Check A*u_exact vs rhs
    let mut auex = vec![0.0; n_dofs];
    a_mat.spmv(&u_ex_vec, &mut auex);
    let res: f64 = (0..n_dofs).map(|i| (rhs2[i]-auex[i]).powi(2)).sum::<f64>().sqrt();
    let b_n: f64 = rhs2.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("Std assembly: ||b - A*u_exact|| = {:.6e} (rel {:.6e})", res, res/b_n.max(1e-30));

    // L2 error
    let mut err2 = 0.0;
    for e in mesh.elem_iter() {
        let re = QuadQ3;
        let quad = re.quadrature(6);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let mut phi = vec![0.0; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            let tr = ElementTransformation::from_simplex_nodes(&mesh, nodes);
            let w = quad.weights[qi] * tr.det_j().abs();
            let xp = tr.map_to_physical(xi);
            re.eval_basis(xi, &mut phi);
            let val: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| x[d] * p).sum();
            err2 += w * (val - u_ex(&xp)).powi(2);
        }
    }
    println!("Std assembly: L2 error = {:.6e}", err2.sqrt());

    // Now test with MANUAL assembly using isoparametric Jacobian (Quad4 needs this)
    let mut m_coo = CooMatrix::new(n_dofs, n_dofs);
    let mut r_manual = vec![0.0; n_dofs];
    for e in mesh.elem_iter() {
        let re = QuadQ3;
        let qo = 5u8;
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let ng = dofs.len();
        let nodes = mesh.element_nodes(e);
        let nld = re.n_dofs();
        let mut me = vec![0.0; ng*ng];
        let mut fe = vec![0.0; ng];
        let mut phi = vec![0.0; nld];
        let mut gr = vec![0.0; nld*2];
        let mut gp = vec![0.0; nld*2];
        // Use isoparametric Jacobian for Quad4 (not from_simplex_nodes!)
        let geo_elem = fem_element::lagrange::QuadQ1;
        for (qi, xi) in quad.points.iter().enumerate() {
            let (jac, det, xp_vec): (nalgebra::DMatrix<f64>, f64, Vec<f64>) =
                fem_assembly::isoparametric_jacobian(&mesh, &nodes.to_vec(), &geo_elem, xi, 2);
            let w = quad.weights[qi] * det.abs();
            let jit = jac.try_inverse().unwrap().transpose();
            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut gr);
            // Transform reference gradients to physical
            for i in 0..ng {
                let g0 = jit[(0,0)]*gr[i*2] + jit[(0,1)]*gr[i*2+1];
                let g1 = jit[(1,0)]*gr[i*2] + jit[(1,1)]*gr[i*2+1];
                gp[i*2] = g0; gp[i*2+1] = g1;
            }
            let fv = f_src(&xp_vec);
            for i in 0..ng {
                fe[i] += w * fv * phi[i];
                for j in 0..ng {
                    me[i*ng+j] += w * (gp[i*2]*gp[j*2] + gp[i*2+1]*gp[j*2+1]);
                }
            }
        }
        for (ir, &r) in dofs.iter().enumerate() {
            r_manual[r] += fe[ir];
            for (ic, &c) in dofs.iter().enumerate() {
                let v = me[ir*ng+ic];
                if v != 0.0 { m_coo.add(r, c, v); }
            }
        }
    }
    let mut a_manual = m_coo.into_csr();
    // BCs
    for &d in &ess_bdr {
        let mut dummy = vec![0.0; n_dofs];
        a_manual.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy);
        if let Some(k) = a_manual.find_entry(d as usize, d as usize) { a_manual.values[k] = 1.0; }
        r_manual[d as usize] = 0.0;
    }
    // Solve
    let mut xm = vec![0.0; n_dofs];
    solve_cg(&a_manual, &r_manual, &mut xm, &cfg).expect("CG manual");
    // Check
    let mut amuex = vec![0.0; n_dofs];
    a_manual.spmv(&u_ex_vec, &mut amuex);
    let res_m: f64 = (0..n_dofs).map(|i| (r_manual[i]-amuex[i]).powi(2)).sum::<f64>().sqrt();
    println!("Manual assembly: ||b - A*u_exact|| = {:.6e}", res_m);

    // Compare solution
    let diff: f64 = (0..n_dofs).map(|i| (x[i]-xm[i]).powi(2)).sum::<f64>().sqrt();
    println!("Solution diff: ||x_std - x_manual|| = {:.6e}", diff);
}
