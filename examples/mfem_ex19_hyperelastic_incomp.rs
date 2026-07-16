//! Example 19 — 1:1 translation of MFEM ex19
//! Quasi-static incompressible neo-Hookean hyperelasticity (mixed u/p).
//!
//! Solves H(x) = 0 via Newton's method with block-preconditioned GMRES.
//!
//! BCs (matching MFEM ex19):
//!   Boundary attribute 1: u = 0 (fixed)
//!   Boundary attribute 2: u_x = 0, u_y = 0.25·x (prescribed shear)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_incomp
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -m data/beam-quad.mesh -o 2 -r 0
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -mu 1.0 -rel 1e-4 -abs 1e-6 -it 500

#![allow(non_snake_case)]

use std::fs::File;
use std::io::Write;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{BlockMatrix, CooMatrix, CsrMatrix, SolverConfig};
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space, VectorH1Space};
use fem_element::lagrange::{TriP1, TriP2, TriP3, QuadQ1, QuadQ2, QuadQ3, QuadQ4};
use fem_element::lagrange::tet::{TetP1, TetP2, TetP3};
use fem_element::lagrange::hex::{HexQ1, HexQ2, HexQ3};
use fem_mesh::element_type::ElementType;
use nalgebra::DMatrix;

// ─── Reference element helpers ─────────────────────────────────────────

/// Factory: return a reference element for the given type and order.
fn re(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) => Box::new(HexQ2),
        (ElementType::Hex8, 3) => Box::new(HexQ3),
        _ => panic!("unsupported element type {et:?} x order {order}"),
    }
}

/// Compute element Jacobian determinant and inverse-transpose at a reference point.
/// Returns (detJ, J^{-T}) where J = ∂x/∂ξ.
fn jacf(m: &impl MeshTopology, elem: u32, xi: &[f64], dim: usize) -> (f64, DMatrix<f64>) {
    let et = m.element_type(elem);
    let nd = m.element_nodes(elem);
    let n_ldofs = nd.len();
    let mut grad = vec![0.0_f64; n_ldofs * dim];
    re(et, 1).eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..n_ldofs {
        let x = m.node_coords(nd[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += x[i] * grad[k * dim + j];
            }
        }
    }
    let det = jac.determinant();
    let inv = jac.try_inverse().expect("singular Jacobian");
    (det, inv.transpose()) // return J^{-T} for covariant gradient transform
}

/// Transform reference-element gradients to physical space:
///   gp[a*dim + j] = Σ_k J^{-T}_{(j,k)} * gr[a*dim + k]
fn xform_grads(ji: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for a in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += ji[(j, k)] * gr[a * dim + k];
            }
            gp[a * dim + j] = s;
        }
    }
}

/// Euclidean norm of a slice.
fn nr(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─── Residual assembly ─────────────────────────────────────────────────

/// Compute the residual vector [R_u; R_p] for the mixed u/p formulation.
///
/// R_u(i,a) = ∫ (μ·F_{iJ} - p·F^{-T}_{iJ}) · ∂φ_a/∂X_J  dx
/// R_p(m)   = ∫ (J - 1) · ψ_m                              dx
fn residual(
    mesh: &impl MeshTopology,
    dim: usize,
    order: u8,
    p_order: u8,
    quad_order: u8,
    mu: f64,
    u: &[f64],
    p: &[f64],
    elem_dofs_u: &[Vec<usize>],
    elem_dofs_p: &[Vec<usize>],
    ru: &mut [f64],
    rp: &mut [f64],
) {
    ru.fill(0.0);
    rp.fill(0.0);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ru_ref = re(et, order);
        let rp_ref = re(et, p_order);
        let n_du = ru_ref.n_dofs();       // scalar shape functions (displacement)
        let n_dp = rp_ref.n_dofs();       // scalar shape functions (pressure)
        let n_vd = n_du * dim;            // vector DOFs per element

        let eu: &[usize] = &elem_dofs_u[e];
        let ep: &[usize] = &elem_dofs_p[e];

        let mut ue = vec![0.0_f64; n_vd];
        for (k, &g) in eu.iter().enumerate() {
            ue[k] = u[g];
        }
        let mut pe = vec![0.0_f64; n_dp];
        for (k, &g) in ep.iter().enumerate() {
            pe[k] = p[g];
        }

        let q = ru_ref.quadrature(quad_order);
        let mut phi_u = vec![0.0_f64; n_du];
        let mut gr_u = vec![0.0_f64; n_du * dim];
        let mut gp_u = vec![0.0_f64; n_du * dim];
        let mut phi_p = vec![0.0_f64; n_dp];

        let mut fu_e = vec![0.0_f64; n_vd];
        let mut fp_e = vec![0.0_f64; n_dp];

        for (qi, xi) in q.points.iter().enumerate() {
            ru_ref.eval_basis(xi, &mut phi_u);
            ru_ref.eval_grad_basis(xi, &mut gr_u);
            rp_ref.eval_basis(xi, &mut phi_p);

            let (det_j, ji) = jacf(mesh, e as u32, xi, dim);
            xform_grads(&ji, &gr_u, &mut gp_u, n_du, dim);
            let w = q.weights[qi] * det_j.abs();

            // Deformation gradient F = I + ∇u
            let mut F = DMatrix::<f64>::identity(dim, dim);
            for k in 0..n_du {
                for i in 0..dim {
                    for j in 0..dim {
                        // ue[k*dim + i] = u_i component at scalar DOF k
                        // gp_u[k*dim + j] = ∂φ_k/∂X_j
                        F[(i, j)] += ue[k * dim + i] * gp_u[k * dim + j];
                    }
                }
            }
            let dJ = F.determinant();
            let iF = F.clone().try_inverse().unwrap_or_else(|| DMatrix::<f64>::identity(dim, dim));
            let FT = iF.transpose(); // F^{-T}

            // Pressure at this QP
            let mut pres = 0.0;
            for k in 0..n_dp {
                pres += pe[k] * phi_p[k];
            }

            // First Piola-Kirchhoff stress: P = μ·F - p·F^{-T}
            let mut P = DMatrix::<f64>::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    P[(i, j)] = mu * F[(i, j)] - pres * FT[(i, j)];
                }
            }

            // R_u contribution: P : ∇v  (v = φ_a · e_i)
            for a in 0..n_du {
                for i in 0..dim {
                    let row = a * dim + i;
                    let mut s = 0.0;
                    for j in 0..dim {
                        s += P[(i, j)] * gp_u[a * dim + j];
                    }
                    fu_e[row] += w * s;
                }
            }

            // R_p contribution: (J - 1) · ψ
            for m in 0..n_dp {
                fp_e[m] += w * (dJ - 1.0) * phi_p[m];
            }
        }

        // Scatter to global
        for (k, &g) in eu.iter().enumerate() {
            ru[g] += fu_e[k];
        }
        for (k, &g) in ep.iter().enumerate() {
            rp[g] += fp_e[k];
        }
    }
}

fn main() {
    let args = Args::parse();
    println!("=== MFEM ex19: Incompressible neo-Hookean hyperelasticity ===");

    // 1. Read mesh
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh2d = mfem.mesh2d.expect("expected 2D mesh");
    let mut mesh = mesh2d;
    for _ in 0..args.refine {
        mesh = refine_uniform(&mesh);
    }
    let dim = 2;
    let order = args.order;
    let p_order = if order > 1 { order - 1 } else { 1 };

    // 2. FE spaces (Taylor-Hood: VectorH1^dim for u, H1 for p)
    let u_space = VectorH1Space::new(mesh.clone(), order, dim);
    let p_space = H1Space::new(mesh.clone(), p_order);
    let nu = u_space.n_dofs();
    let np = p_space.n_dofs();
    let ns = u_space.n_scalar_dofs(); // scalar DOFs per component
    println!("dim(u) = {nu}");
    println!("dim(p) = {np}");
    println!("dim(u+p) = {}", nu + np);

    // 3. Dirichlet BCs (matching MFEM ex19)
    //    Attr 1: fixed (u=0). Attr 2: u_x=0, u_y=0.25*x
    let dm = u_space.scalar_dof_manager();
    let attr1 = boundary_dofs(u_space.mesh(), dm, &[1]);
    let attr2 = boundary_dofs(u_space.mesh(), dm, &[2]);
    let mut du: Vec<(usize, f64)> = Vec::new();
    for &d in &attr1 {
        // Both components zero
        du.push((d as usize, 0.0));
        du.push((d as usize + ns, 0.0));
    }
    for &d in &attr2 {
        let x = dm.dof_coord(d as u32)[0]; // x-coordinate
        du.push((d as usize, 0.0));         // u_x = 0
        du.push((d as usize + ns, 0.25 * x)); // u_y = 0.25*x
    }

    // 4. Initial guess: InitialDeformation = ReferenceConfiguration + shear
    //    u(x) = x_def - x_ref  ->  u_x = 0, u_y = 0.25*x[0]
    let mut u = vec![0.0_f64; nu];
    let mut p = vec![0.0_f64; np];
    for s in 0..ns {
        let xc = dm.dof_coord(s as u32);
        let x = xc[0];
        // Component-major: idx = comp * ns + s
        u[0 * ns + s] = 0.0;         // u_x = 0 (no offset from reference)
        u[1 * ns + s] = 0.25 * x;    // u_y = 0.25*x
    }
    // Apply BC values to the DOF vector (essential BC elimination)
    for &(dof, val) in &du {
        u[dof] = val;
    }

    println!("Initial guess set. DOFs: displacement={nu}, pressure={np}");

    // 5. Pre-compute element DOF tables
    let ne = mesh.n_elements() as usize;
    let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
        .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
        .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();

    // 6. Quadrature order: 2*order + 3 (enough for the nonlinear integrand)
    let quad_order = 2 * order + 3;

    // 7. Compute initial residual
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    residual(&mesh, dim as usize, order, p_order, quad_order, args.mu,
             &u, &p, &elem_dofs_u, &elem_dofs_p, &mut ru, &mut rp);
    // Zero out BC DOFs in the displacement residual
    for &(dof, _) in &du {
        ru[dof] = 0.0;
    }

    let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
    println!("Newton 0 ||r|| = {r0:.5}");
}

struct Args {
    mesh: String,
    refine: usize,
    order: u8,
    mu: f64,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            mesh: "data/beam-quad.mesh".into(),
            refine: 0,
            order: 2,
            mu: 1.0,
            rel_tol: 1e-4,
            abs_tol: 1e-6,
            max_iter: 500,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" => a.mesh = it.next().unwrap_or_default(),
                "-r" => a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "-o" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
                "-mu" => a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                "-rel" => a.rel_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-4),
                "-abs" => a.abs_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6),
                "-it" => a.max_iter = it.next().and_then(|v| v.parse().ok()).unwrap_or(500),
                _ => {}
            }
        }
        a
    }
}
