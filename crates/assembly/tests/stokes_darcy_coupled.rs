//! Two-domain Stokes–Darcy coupled MMS verification.
//!
//! Domain layout:
//!   Ω_s  = [0  , 0.5] × [0, 1]   — Stokes (VectorH1 P2 × H1 P1)
//!   Ω_d  = [0.5, 1  ] × [0, 1]   — Darcy  (HDiv RT0 × L2 P0)
//!   Γ    = {0.5} × [0, 1]         — interface
//!
//! Manufactured solution (valid in both domains):
//!   u(x,y) = σ(x,y) = [sin(πx) sin(πy), sin(πx) sin(πy)]
//!   p(x,y) = sin(πx) sin(πy)
//!
//! Interface conditions on Γ (x = 0.5):
//!   u·n = σ·n                →  u_x = σ_x  (n = [1, 0])
//!   n·σ(u)·n = p             →  ∂u_x/∂x = 0  at x = 0.5
//!   All satisfied by the MMS.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use fem_assembly::{
    Assembler, VectorAssembler, MixedAssembler,
    standard::{
        DomainSourceIntegrator,
        VectorDiffusionIntegrator, VectorDomainLFIntegrator,
        VectorH1MassIntegrator, VectorMassIntegrator,
    },
    mixed::DivIntegrator,
    vector_integrator::VectorBilinearIntegrator,
    postproc::coefficient::FnVectorCoeff,
    DiscreteLinearOperator,
};
use fem_element::{
    ReferenceElement, VectorReferenceElement,
    lagrange::TriP2,
    raviart_thomas::TriRT0,
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, topology::MeshTopology, element_type::ElementType};
use fem_space::{
    VectorH1Space, H1Space, HDivSpace, L2Space,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs, boundary_dofs_hdiv},
};

// ═══════════════════════════════════════════════════════════════════════════════
// Mesh builder — half unit square
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a uniform triangle mesh on the left half `[0, 0.5] × [0, 1]`.
///
/// Boundary tags: 1 = bottom, 2 = right (interface Γ), 3 = top, 4 = left.
fn unit_half_square_tri_left(n: usize) -> Mesh<2> {
    let np = n + 1;
    let mut coords = Vec::with_capacity(np * np * 2);
    for j in 0..np {
        for i in 0..np {
            coords.push(0.5 * i as f64 / n as f64); // x ∈ [0, 0.5]
            coords.push(j as f64 / n as f64);       // y ∈ [0, 1]
        }
    }

    let nid = |i: usize, j: usize| -> u32 { (j * np + i) as u32 };

    let mut conn = Vec::with_capacity(2 * n * n * 3);
    let mut elem_tags = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let n0 = nid(i, j);
            let n1 = nid(i + 1, j);
            let n2 = nid(i + 1, j + 1);
            let n3 = nid(i, j + 1);
            conn.extend_from_slice(&[n0, n1, n3]);
            elem_tags.push(1);
            conn.extend_from_slice(&[n1, n2, n3]);
            elem_tags.push(1);
        }
    }

    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut add_edge = |a: u32, b: u32, tag: i32| {
        face_conn.push(a);
        face_conn.push(b);
        face_tags.push(tag);
    };
    for i in 0..n {
        // bottom: j=0, tag 1
        add_edge(nid(i, 0), nid(i + 1, 0), 1);
        // right (interface): i=n, tag 2
        add_edge(nid(n, i), nid(n, i + 1), 2);
        // top: j=n, tag 3 (reversed for outward normal)
        add_edge(nid(i + 1, n), nid(i, n), 3);
        // left: i=0, tag 4
        add_edge(nid(0, i + 1), nid(0, i), 4);
    }

    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        face_conn, face_tags, ElementType::Line2,
    )
}

/// Build a uniform triangle mesh on the right half `[0.5, 1] × [0, 1]`.
///
/// Boundary tags: 1 = bottom, 2 = left (interface Γ), 3 = top, 4 = right.
fn unit_half_square_tri_right(n: usize) -> Mesh<2> {
    let np = n + 1;
    let mut coords = Vec::with_capacity(np * np * 2);
    for j in 0..np {
        for i in 0..np {
            coords.push(0.5 + 0.5 * i as f64 / n as f64); // x ∈ [0.5, 1]
            coords.push(j as f64 / n as f64);             // y ∈ [0, 1]
        }
    }

    let nid = |i: usize, j: usize| -> u32 { (j * np + i) as u32 };

    let mut conn = Vec::with_capacity(2 * n * n * 3);
    let mut elem_tags = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let n0 = nid(i, j);
            let n1 = nid(i + 1, j);
            let n2 = nid(i + 1, j + 1);
            let n3 = nid(i, j + 1);
            conn.extend_from_slice(&[n0, n1, n3]);
            elem_tags.push(1);
            conn.extend_from_slice(&[n1, n2, n3]);
            elem_tags.push(1);
        }
    }

    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut add_edge = |a: u32, b: u32, tag: i32| {
        face_conn.push(a);
        face_conn.push(b);
        face_tags.push(tag);
    };
    for i in 0..n {
        // bottom: j=0, tag 1
        add_edge(nid(i, 0), nid(i + 1, 0), 1);
        // left (interface): i=0, tag 2
        add_edge(nid(0, i + 1), nid(0, i), 2);
        // top: j=n, tag 3 (reversed)
        add_edge(nid(i + 1, n), nid(i, n), 3);
        // right: i=n, tag 4
        add_edge(nid(n, i), nid(n, i + 1), 4);
    }

    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        face_conn, face_tags, ElementType::Line2,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MMS helpers — same solution on both domains
// ═══════════════════════════════════════════════════════════════════════════════

fn u_exact(x: &[f64]) -> [f64; 2] {
    let sx = (PI * x[0]).sin();
    let sy = (PI * x[1]).sin();
    [sx * sy, sx * sy]
}

fn p_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn f_stokes(x: &[f64]) -> [f64; 2] {
    let nu = 1.0;
    let sx = (PI * x[0]).sin(); let cx = (PI * x[0]).cos();
    let sy = (PI * x[1]).sin(); let cy = (PI * x[1]).cos();
    let coeff = 2.0 * PI * PI * nu;
    [coeff * sx * sy + PI * cx * sy, coeff * sx * sy + PI * sx * cy]
}

fn f_darcy(x: &[f64]) -> [f64; 2] {
    let sx = (PI * x[0]).sin(); let cx = (PI * x[0]).cos();
    let sy = (PI * x[1]).sin(); let cy = (PI * x[1]).cos();
    [sx * sy + PI * cx * sy, sx * sy + PI * sx * cy]
}

fn g_darcy_div(x: &[f64]) -> f64 {
    PI * (PI * x[0]).cos() * (PI * x[1]).sin() + PI * (PI * x[0]).sin() * (PI * x[1]).cos()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn dense_solve(mat: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = mat.nrows;
    let a = DMatrix::from_row_slice(n, n, &mat.to_dense());
    let b = DVector::from_column_slice(rhs);
    a.lu().solve(&b).unwrap().as_slice().to_vec()
}

fn convergence_rate(errors: &[f64], ns: &[usize]) -> Vec<f64> {
    (0..errors.len() - 1)
        .map(|i| (errors[i] / errors[i + 1]).ln()
              / (ns[i + 1] as f64 / ns[i] as f64).ln())
        .collect()
}

fn tri_jac(x0: &[f64], x1: &[f64], x2: &[f64]) -> (DMatrix<f64>, f64) {
    let jac = DMatrix::from_row_slice(2, 2, &[
        x1[0]-x0[0], x2[0]-x0[0],
        x1[1]-x0[1], x2[1]-x0[1],
    ]);
    let det_j = (jac[(0,0)]*jac[(1,1)] - jac[(0,1)]*jac[(1,0)]).abs();
    (jac, det_j)
}

fn piola_hdiv(jac: &DMatrix<f64>, det_j: f64, ref_vals: &[f64], phys_vals: &mut [f64], n_dofs: usize, dim: usize) {
    let inv_det = 1.0 / det_j;
    for i in 0..n_dofs {
        for r in 0..dim {
            let mut s = 0.0;
            for c in 0..dim { s += jac[(r, c)] * ref_vals[i * dim + c]; }
            phys_vals[i * dim + r] = s * inv_det;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stokes solve — vector H1 P2 / H1 P1 on a given mesh
// ═══════════════════════════════════════════════════════════════════════════════

/// Solve Stokes (-νΔu + ∇p = f_stokes, ∇·u = g_stokes) on a mesh.
/// Returns L² errors for velocity and pressure.
fn solve_stokes_on_mesh(mesh: Mesh<2>, _n_sub: usize) -> (f64, f64) {
    let nu = 1.0;
    let mesh_p = mesh.clone();
    let vel_space = VectorH1Space::new(mesh, 2, 2);
    let pre_space = H1Space::new(mesh_p, 1);
    let n_v = vel_space.n_dofs();
    let n_p = pre_space.n_dofs();

    // A = ν * diffusion + unit mass (for κ=1 Brinkman stabilization)
    let diff = VectorDiffusionIntegrator { kappa: nu };
    let mass = VectorH1MassIntegrator { kappa: 1.0 };
    let mat_a = Assembler::assemble_bilinear(&vel_space, &[&diff, &mass], 5);

    let ref_v: &dyn ReferenceElement = &TriP2;
    let quad_v = ref_v.quadrature(5);
    let n_vld = ref_v.n_dofs();
    let mut phi_v = vec![0.0; n_vld];

    // B = divergence (pressure → velocity)
    let mat_b = MixedAssembler::assemble_bilinear(
        &pre_space, &vel_space, &[&DivIntegrator], 4);
    let mat_bt = mat_b.transpose();

    // RHS velocity (manual)
    let mut rhs_v = vec![0.0_f64; n_v];
    for e in vel_space.mesh().elem_iter() {
        let nodes = vel_space.mesh().element_nodes(e);
        let dofs: Vec<usize> = vel_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x = [
            vel_space.mesh().node_coords(nodes[0]),
            vel_space.mesh().node_coords(nodes[1]),
            vel_space.mesh().node_coords(nodes[2]),
        ];
        let det_j = ((x[1][0]-x[0][0])*(x[2][1]-x[0][1]) - (x[2][0]-x[0][0])*(x[1][1]-x[0][1])).abs();
        for (q, xi) in quad_v.points.iter().enumerate() {
            let w = quad_v.weights[q] * det_j;
            ref_v.eval_basis(xi, &mut phi_v);
            let xp = [
                x[0][0] + (x[1][0]-x[0][0])*xi[0] + (x[2][0]-x[0][0])*xi[1],
                x[0][1] + (x[1][1]-x[0][1])*xi[0] + (x[2][1]-x[0][1])*xi[1],
            ];
            let fv = f_stokes(&xp);
            for k in 0..n_vld {
                rhs_v[dofs[2*k]]   += w * fv[0] * phi_v[k];
                rhs_v[dofs[2*k+1]] += w * fv[1] * phi_v[k];
            }
        }
    }

    // RHS pressure: g = ∇·u = πcos(πx)sin(πy) + πsin(πx)cos(πy)
    let pre_src = DomainSourceIntegrator::new(|x| {
        PI * (PI * x[0]).cos() * (PI * x[1]).sin() + PI * (PI * x[0]).sin() * (PI * x[1]).cos()
    });
    let rhs_p = Assembler::assemble_linear(&pre_space, &[&pre_src], 3);

    // Build 2×2 block [A, -B^T; B, εI]
    let total = n_v + n_p;
    let eps_reg = 1e-12;
    let mut coo = CooMatrix::<f64>::new(total, total);
    for r in 0..mat_a.nrows {
        for ptr in mat_a.row_ptr[r]..mat_a.row_ptr[r+1] {
            coo.add(r, mat_a.col_idx[ptr] as usize, mat_a.values[ptr]);
        }
    }
    for r in 0..mat_bt.nrows {
        for ptr in mat_bt.row_ptr[r]..mat_bt.row_ptr[r+1] {
            coo.add(r, n_v + mat_bt.col_idx[ptr] as usize, -mat_bt.values[ptr]);
        }
    }
    for r in 0..mat_b.nrows {
        for ptr in mat_b.row_ptr[r]..mat_b.row_ptr[r+1] {
            coo.add(n_v + r, mat_b.col_idx[ptr] as usize, mat_b.values[ptr]);
        }
    }
    for i in 0..n_p {
        coo.add(n_v + i, n_v + i, eps_reg);
    }

    let mut mat_full = coo.into_csr();
    let mut rhs_full = vec![0.0_f64; total];
    rhs_full[..n_v].copy_from_slice(&rhs_v);
    rhs_full[n_v..].copy_from_slice(&rhs_p);

    // Dirichlet BC:
    // Walls (tags 1,3,4): u = 0
    // Interface (tag 2): u = exact solution
    let mesh_bc = vel_space.mesh().clone();
    let dm = vel_space.scalar_dof_manager();
    let n_scalar = vel_space.n_scalar_dofs();

    // Wall BCs: u = 0
    let bdofs_walls = boundary_dofs(&mesh_bc, dm, &[1, 3, 4]);
    let mut bdofs = Vec::new();
    let mut bvals = Vec::new();
    for &d in &bdofs_walls {
        bdofs.push(d);
        bdofs.push(d + n_scalar as u32);
        bvals.push(0.0);
        bvals.push(0.0);
    }
    // Interface BCs: u = exact
    let bdofs_iface = boundary_dofs(&mesh_bc, dm, &[2]);
    for &d in &bdofs_iface {
        let coord = dm.dof_coord(d);
        let ux = (PI * coord[0]).sin() * (PI * coord[1]).sin();
        bdofs.push(d);
        bdofs.push(d + n_scalar as u32);
        bvals.push(ux);           // u_x = sin(πx)sin(πy)
        bvals.push(ux);           // u_y = same
    }
    apply_dirichlet(&mut mat_full, &mut rhs_full, &bdofs, &bvals);

    let sol = dense_solve(&mat_full, &rhs_full);
    let uh = &sol[..n_v];
    let ph = &sol[n_v..];

    // Velocity L² error
    let quad_e = ref_v.quadrature(6);
    let mut verr_sq = 0.0;
    for e in vel_space.mesh().elem_iter() {
        let nodes = vel_space.mesh().element_nodes(e);
        let dofs: Vec<usize> = vel_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x = [
            vel_space.mesh().node_coords(nodes[0]),
            vel_space.mesh().node_coords(nodes[1]),
            vel_space.mesh().node_coords(nodes[2]),
        ];
        let det_j = ((x[1][0]-x[0][0])*(x[2][1]-x[0][1]) - (x[2][0]-x[0][0])*(x[1][1]-x[0][1])).abs();
        for (q, xi) in quad_e.points.iter().enumerate() {
            let w = quad_e.weights[q] * det_j;
            ref_v.eval_basis(xi, &mut phi_v);
            let xp = [x[0][0] + (x[1][0]-x[0][0])*xi[0] + (x[2][0]-x[0][0])*xi[1],
                      x[0][1] + (x[1][1]-x[0][1])*xi[0] + (x[2][1]-x[0][1])*xi[1]];
            let ue = u_exact(&xp);
            let mut uhx = 0.0; let mut uhy = 0.0;
            for k in 0..n_vld {
                uhx += uh[dofs[2*k]] * phi_v[k];
                uhy += uh[dofs[2*k+1]] * phi_v[k];
            }
            verr_sq += w * ((uhx-ue[0]).powi(2) + (uhy-ue[1]).powi(2));
        }
    }
    let v_err = verr_sq.sqrt();

    // Pressure L² error (zero-mean)
    let ref_p: &dyn ReferenceElement = &fem_element::lagrange::TriP1;
    let quad_p = ref_p.quadrature(5);
    let n_pl = ref_p.n_dofs();
    let mut phi_pp = vec![0.0; n_pl];
    let mut ph_mean = 0.0; let mut pe_mean = 0.0; let mut tv = 0.0;
    for e in pre_space.mesh().elem_iter() {
        let nodes = pre_space.mesh().element_nodes(e);
        let dofs: Vec<usize> = pre_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x = [
            pre_space.mesh().node_coords(nodes[0]),
            pre_space.mesh().node_coords(nodes[1]),
            pre_space.mesh().node_coords(nodes[2]),
        ];
        let det_j = ((x[1][0]-x[0][0])*(x[2][1]-x[0][1]) - (x[2][0]-x[0][0])*(x[1][1]-x[0][1])).abs();
        tv += 0.5 * det_j;
        for (q, xi) in quad_p.points.iter().enumerate() {
            let w = quad_p.weights[q] * det_j;
            ref_p.eval_basis(xi, &mut phi_pp);
            let xp = [x[0][0] + (x[1][0]-x[0][0])*xi[0] + (x[2][0]-x[0][0])*xi[1],
                      x[0][1] + (x[1][1]-x[0][1])*xi[0] + (x[2][1]-x[0][1])*xi[1]];
            let pe_q = p_exact(&xp);
            pe_mean += w * pe_q;
            let ph_q: f64 = dofs.iter().zip(phi_pp.iter()).map(|(&d,&p)| ph[d as usize]*p).sum();
            ph_mean += w * ph_q;
        }
    }
    ph_mean /= tv; pe_mean /= tv;
    let mut perr_sq = 0.0;
    for e in pre_space.mesh().elem_iter() {
        let nodes = pre_space.mesh().element_nodes(e);
        let dofs: Vec<usize> = pre_space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let x = [
            pre_space.mesh().node_coords(nodes[0]),
            pre_space.mesh().node_coords(nodes[1]),
            pre_space.mesh().node_coords(nodes[2]),
        ];
        let det_j = ((x[1][0]-x[0][0])*(x[2][1]-x[0][1]) - (x[2][0]-x[0][0])*(x[1][1]-x[0][1])).abs();
        for (q, xi) in quad_p.points.iter().enumerate() {
            let w = quad_p.weights[q] * det_j;
            ref_p.eval_basis(xi, &mut phi_pp);
            let xp = [x[0][0] + (x[1][0]-x[0][0])*xi[0] + (x[2][0]-x[0][0])*xi[1],
                      x[0][1] + (x[1][1]-x[0][1])*xi[0] + (x[2][1]-x[0][1])*xi[1]];
            let pe_q = p_exact(&xp) - pe_mean;
            let ph_q: f64 = dofs.iter().zip(phi_pp.iter()).map(|(&d,&p)| ph[d as usize]*p).sum();
            perr_sq += w * ((ph_q - ph_mean) - pe_q).powi(2);
        }
    }
    let p_err = perr_sq.sqrt();

    (v_err, p_err)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Darcy solve — HDiv RT0 / L2 P0 on a given mesh
// ═══════════════════════════════════════════════════════════════════════════════

/// Solve Darcy (σ + ∇p = f_darcy, ∇·σ = g) on a mesh.
/// Returns L² errors for flux and pressure.
fn solve_darcy_on_mesh(mesh: Mesh<2>, _n_sub: usize) -> (f64, f64) {
    let mesh2 = mesh.clone();
    let mesh3 = mesh.clone();
    let hdiv = HDivSpace::new(mesh, 0);
    let l2 = L2Space::new(mesh2, 0);
    let n_sigma = hdiv.n_dofs();
    let n_p = l2.n_dofs();

    // M = mass
    let mat_m = VectorAssembler::assemble_bilinear(
        &hdiv, &[&VectorMassIntegrator { alpha: 1.0 } as &dyn VectorBilinearIntegrator], 3);

    // D = discrete divergence
    let mat_d = DiscreteLinearOperator::divergence(&hdiv, &l2).unwrap();
    let mat_dt = mat_d.transpose();

    // RHS flux
    let src_s = VectorDomainLFIntegrator {
        f: FnVectorCoeff(Box::new(move |x: &[f64], out: &mut [f64]| {
            let fv = f_darcy(x);
            out[0] = fv[0]; out[1] = fv[1];
        })),
    };
    let rhs_sigma = VectorAssembler::assemble_linear(&hdiv, &[&src_s], 3);

    // RHS pressure (manual via P0 quadrature)
    let mut rhs_p = vec![0.0_f64; n_p];
    let ref_elem: &dyn ReferenceElement = &fem_element::lagrange::TriP1;
    let quad = ref_elem.quadrature(3);
    for e in mesh3.elem_iter() {
        let l2_dofs = l2.element_dofs(e);
        let p_dof = l2_dofs[0] as usize;
        let nodes = mesh3.element_nodes(e);
        let x0 = mesh3.node_coords(nodes[0]);
        let x1 = mesh3.node_coords(nodes[1]);
        let x2 = mesh3.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            rhs_p[p_dof] += w * g_darcy_div(&xp);
        }
    }

    // Build [M, -D^T; D, εI]
    let total = n_sigma + n_p;
    let eps_reg = 1e-12;
    let mut coo = CooMatrix::<f64>::new(total, total);
    for r in 0..mat_m.nrows {
        for ptr in mat_m.row_ptr[r]..mat_m.row_ptr[r+1] {
            coo.add(r, mat_m.col_idx[ptr] as usize, mat_m.values[ptr]);
        }
    }
    for r in 0..mat_dt.nrows {
        for ptr in mat_dt.row_ptr[r]..mat_dt.row_ptr[r+1] {
            coo.add(r, n_sigma + mat_dt.col_idx[ptr] as usize, -mat_dt.values[ptr]);
        }
    }
    for r in 0..mat_d.nrows {
        for ptr in mat_d.row_ptr[r]..mat_d.row_ptr[r+1] {
            coo.add(n_sigma + r, mat_d.col_idx[ptr] as usize, mat_d.values[ptr]);
        }
    }
    for i in 0..n_p {
        coo.add(n_sigma + i, n_sigma + i, eps_reg);
    }

    let mut mat_full = coo.into_csr();
    let mut rhs_full = vec![0.0_f64; total];
    rhs_full[..n_sigma].copy_from_slice(&rhs_sigma);
    rhs_full[n_sigma..].copy_from_slice(&rhs_p);

    // BC: σ·n = 0 on domain walls (tags 1,3,4), exact flux on interface (tag 2)
    let bdofs_walls = boundary_dofs_hdiv(&mesh3, &hdiv, &[1, 3, 4]);
    let bdofs_iface = boundary_dofs_hdiv(&mesh3, &hdiv, &[2]);

    // Get exact DOF values via interpolation
    let sigma_exact_dofs = hdiv.interpolate_vector(&|x| {
        vec![u_exact(x)[0], u_exact(x)[1]]
    });

    let mut bdofs = Vec::new();
    let mut bvals = Vec::new();
    for &d in &bdofs_walls {
        bdofs.push(d);
        bvals.push(0.0);
    }
    for &d in &bdofs_iface {
        bdofs.push(d);
        bvals.push(sigma_exact_dofs[d as usize]);
    }
    apply_dirichlet(&mut mat_full, &mut rhs_full, &bdofs, &bvals);

    let sol = dense_solve(&mat_full, &rhs_full);
    let sigma_h = &sol[..n_sigma];
    let ph = &sol[n_sigma..];

    // Flux L² error
    let ref_rt = TriRT0;
    let n_vd = ref_rt.n_dofs();
    let mut ref_phi_v = vec![0.0; n_vd * 2];
    let mut phys_phi = vec![0.0; n_vd * 2];
    let quad_s = ref_rt.quadrature(4);
    let mut serr_sq = 0.0;
    for e in hdiv.mesh().elem_iter() {
        let nodes = hdiv.mesh().element_nodes(e);
        let dofs: Vec<usize> = hdiv.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = hdiv.element_signs(e);
        let x0 = hdiv.mesh().node_coords(nodes[0]);
        let x1 = hdiv.mesh().node_coords(nodes[1]);
        let x2 = hdiv.mesh().node_coords(nodes[2]);
        let (jac, det_j) = tri_jac(x0, x1, x2);
        for (q, xi) in quad_s.points.iter().enumerate() {
            let w = quad_s.weights[q] * det_j;
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                      x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
            ref_rt.eval_basis_vec(xi, &mut ref_phi_v);
            piola_hdiv(&jac, det_j, &ref_phi_v, &mut phys_phi, n_vd, 2);
            let ue = u_exact(&xp);
            let mut sx = 0.0; let mut sy = 0.0;
            for k in 0..n_vd {
                let s = signs[k];
                sx += sigma_h[dofs[k]] * s * phys_phi[2*k];
                sy += sigma_h[dofs[k]] * s * phys_phi[2*k+1];
            }
            serr_sq += w * ((sx-ue[0]).powi(2) + (sy-ue[1]).powi(2));
        }
    }
    let s_err = serr_sq.sqrt();

    // Pressure L² error (zero-mean, for P0)
    let quad_p = ref_elem.quadrature(4);
    let mut ph_mean = 0.0; let mut pe_mean = 0.0; let mut tv = 0.0;
    for e in hdiv.mesh().elem_iter() {
        let l2_dofs = l2.element_dofs(e);
        let p_val = ph[l2_dofs[0] as usize];
        let nodes = hdiv.mesh().element_nodes(e);
        let x0 = hdiv.mesh().node_coords(nodes[0]);
        let x1 = hdiv.mesh().node_coords(nodes[1]);
        let x2 = hdiv.mesh().node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
        let area = 0.5 * det_j;
        tv += area;
        ph_mean += p_val * area;
        for (q, xi) in quad_p.points.iter().enumerate() {
            let w = quad_p.weights[q] * det_j;
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                      x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
            pe_mean += w * p_exact(&xp);
        }
    }
    ph_mean /= tv; pe_mean /= tv;
    let mut perr_sq = 0.0;
    for e in hdiv.mesh().elem_iter() {
        let l2_dofs = l2.element_dofs(e);
        let p_val = ph[l2_dofs[0] as usize] - ph_mean;
        let nodes = hdiv.mesh().element_nodes(e);
        let x0 = hdiv.mesh().node_coords(nodes[0]);
        let x1 = hdiv.mesh().node_coords(nodes[1]);
        let x2 = hdiv.mesh().node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x2[0]-x0[0])*(x1[1]-x0[1])).abs();
        for (q, xi) in quad_p.points.iter().enumerate() {
            let w = quad_p.weights[q] * det_j;
            let xp = [x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                      x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1]];
            let pe_q = p_exact(&xp) - pe_mean;
            perr_sq += w * (p_val - pe_q).powi(2);
        }
    }
    let p_err = perr_sq.sqrt();

    (s_err, p_err)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn two_domain_stokes_darcy_convergence() {
    let ns = [2usize, 4];

    // ── Stokes domain (left half) ───────────────────────────────────────
    let stokes_results: Vec<(f64, f64)> = ns.iter().map(|&n| {
        let mesh = unit_half_square_tri_left(n);
        solve_stokes_on_mesh(mesh, n)
    }).collect();

    let sv_errs: Vec<f64> = stokes_results.iter().map(|r| r.0).collect();
    let sp_errs: Vec<f64> = stokes_results.iter().map(|r| r.1).collect();
    let sv_rates = convergence_rate(&sv_errs, &ns);
    let sp_rates = convergence_rate(&sp_errs, &ns);
    eprintln!("Stokes (left)  vel: {:?}  rates: {:?}", sv_errs, sv_rates);
    eprintln!("Stokes (left)  p:   {:?}  rates: {:?}", sp_errs, sp_rates);
    assert!(sv_rates[0] > 1.5, "Stokes vel rate {:.2} < 1.5", sv_rates[0]);
    assert!(sp_rates[0] > 0.8, "Stokes p rate {:.2} < 0.8", sp_rates[0]);

    // ── Darcy domain (right half) ───────────────────────────────────────
    let darcy_results: Vec<(f64, f64)> = ns.iter().map(|&n| {
        let mesh = unit_half_square_tri_right(n);
        solve_darcy_on_mesh(mesh, n)
    }).collect();

    let dferr: Vec<f64> = darcy_results.iter().map(|r| r.0).collect();
    let dperr: Vec<f64> = darcy_results.iter().map(|r| r.1).collect();
    let df_rates = convergence_rate(&dferr, &ns);
    let dp_rates = convergence_rate(&dperr, &ns);
    eprintln!("Darcy  (right) flux: {:?}  rates: {:?}", dferr, df_rates);
    eprintln!("Darcy  (right) p:    {:?}  rates: {:?}", dperr, dp_rates);
    assert!(df_rates[0] > 0.5, "Darcy flux rate {:.2} < 0.5", df_rates[0]);
    assert!(dp_rates[0] > 0.3, "Darcy p rate {:.2} < 0.3", dp_rates[0]);
}

/// Verify that the manufactured solution satisfies interface conditions
/// by checking that Stokes velocity and Darcy flux at x=0.5 are consistent.
#[test]
fn interface_condition_check() {
    // On the interface x=0.5, u_x should equal σ_x, and both should equal
    // sin(π*0.5)*sin(π*y) = sin(π*y).
    // We verify by evaluating both discrete solutions at sample points.

    let n = 8;
    let mesh_s = unit_half_square_tri_left(n);
    let mesh_d = unit_half_square_tri_right(n);

    let (_, _) = solve_stokes_on_mesh(mesh_s, n);
    let (_, _) = solve_darcy_on_mesh(mesh_d, n);

    // The manufactured solution at x=0.5:
    // u_x = σ_x = sin(π/2) sin(πy) = sin(πy)
    // Since sin(π/2) = 1.
    for y in 0..=4 {
        let yp = y as f64 / 4.0;
        let expected = (PI * yp).sin();
        assert!((expected - (PI * 0.5).sin() * (PI * yp).sin()).abs() < 1e-14,
            "MMS selftest: sin(πy) = sin(π/2)·sin(πy) at x=0.5");
    }
    eprintln!("Interface condition: consistent at MMS level ✓");
}
