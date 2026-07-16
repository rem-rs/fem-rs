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

    // --- STUB: Newton and output will be added in subsequent tasks ---
    println!("Skeleton complete \u{2014} Newton solver not yet implemented.");
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
