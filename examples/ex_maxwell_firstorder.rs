//! # Example: First-order E-B Maxwell (staggered leapfrog)
//!
//! Solves the first-order full-wave Maxwell system in the 2-D TM mode:
//!
//! ```text
//!   ε ∂E/∂t = curl(μ⁻¹ B) − σ E + J    E ∈ H(curl)
//!     ∂B/∂t = −curl E                   B ∈ L2 (scalar)
//! ```
//!
//! on the unit square Ω = [0,1]² with **PEC** boundary condition `n×E = 0`
//! on all four walls.
//!
//! ## Manufactured solution (ε=μ=1, σ=0, J=0)
//!
//! ```text
//!   E(x, t) = sin(π t) · (sin(πy), sin(πx))
//!   B(x, t) = cos(π t) · (cos(πx) − cos(πy))
//! ```
//!
//! This is an exact cavity mode with angular frequency ω = π.
//! The source is identically zero (no driving term needed).
//!
//! ## Time integration
//!
//! Staggered (Yee) leapfrog:
//! ```text
//!   B^{n+½} = B^{n-½} − dt · M_B⁻¹ · C · E^n
//!   E^{n+1} = E^n     + dt · M_E⁻¹ · (1/μ) Cᵀ · B^{n+½}
//! ```
//!
//! The conserved energy is `(1/2) ||E||²_{ε M_E}  +  (1/2μ) ||B||²_{M_B}`.
//!
//! ## Usage
//! ```
//! cargo run --example ex_maxwell_firstorder
//! cargo run --example ex_maxwell_firstorder -- --n 16 --dt 0.005 --t-end 2.0
//! ```

use std::f64::consts::PI;

use fem_examples::FirstOrderMaxwellOp;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::SolverConfig;
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();

    println!("=== fem-rs: First-order Maxwell (staggered leapfrog) ===");
    println!(
        "  Mesh: {}×{}, ε={:.2}, μ={:.2}, σ={:.2}",
        args.n, args.n, args.eps, args.mu, args.sigma
    );
    println!(
        "  dt={:.4}, T={:.2}, steps={}",
        args.dt, args.t_end, args.n_steps()
    );

    // ─── 1. Build first-order Maxwell operator ────────────────────────────────
    let op = FirstOrderMaxwellOp::new_unit_square(args.n, args.eps, args.mu, args.sigma);
    println!(
        "  E DOFs (H(curl)): {}    B DOFs (L2): {}",
        op.n_e, op.n_b
    );

    let cfg = SolverConfig {
        rtol: 1e-10, atol: 0.0, max_iter: 1000, verbose: false,
        ..SolverConfig::default()
    };

    // ─── 2. Initial conditions (t = 0) ───────────────────────────────────────
    // Exact solution: E(x, 0) = sin(0)·(…) = 0
    let mut e = vec![0.0_f64; op.n_e];

    // B(x, 0) = cos(0)·(cos πx − cos πy) = cos πx − cos πy
    // Project onto L2 P0: evaluate at each triangle's centroid.
    let mesh   = SimplexMesh::<2>::unit_square_tri(args.n);
    let l2     = L2Space::new(mesh.clone(), 0);
    let mut b  = project_b0_onto_l2(&l2, args.mu);

    let force = vec![0.0_f64; op.n_e];

    // ─── 3. Initial energy ───────────────────────────────────────────────────
    let e_init = op.compute_energy(&e, &b);
    println!("\n  Step     t         Energy       RelErrEnergy");
    println!("  ──────────────────────────────────────────────");
    println!("  {step:5}  {t:8.4}  {en:12.6e}  {rel:10.4e}",
        step=0, t=0.0, en=e_init, rel=0.0);

    // ─── 4. Leapfrog time loop ────────────────────────────────────────────────
    let dt      = args.dt;
    let nsteps  = args.n_steps();
    let mut t   = 0.0;
    let print_every = (nsteps / 10).max(1);

    for step in 0..nsteps {
        // B^{n+½}
        op.b_half_step(dt, &e, &mut b);
        // E^{n+1}
        e = op.e_full_step(dt, &e, &b, &force, &cfg);
        t += dt;

        if (step + 1) % print_every == 0 || step + 1 == nsteps {
            let en  = op.compute_energy(&e, &b);
            let rel = (en - e_init).abs() / e_init.max(1e-30);
            println!("  {step:5}  {t:8.4}  {en:12.6e}  {rel:10.4e}",
                step = step + 1);
        }
    }

    // ─── 5. L2 error of B at final time ─────────────────────────────────────
    let b_err = l2_error_b(&l2, &b, t, args.mu);
    let b_norm = l2_norm_b_exact(&l2, t, args.mu);
    println!("\n  L2 error of B at t={t:.4}: {b_err:.4e}  (relative: {:.4e})",
        b_err / b_norm.max(1e-30));

    println!("\nDone.");
}

// ─── Project B(x,0) = (1/μ) · (cos πx − cos πy) onto L2 P0 ─────────────────

fn project_b0_onto_l2<M: MeshTopology>(l2: &L2Space<M>, _mu: f64) -> Vec<f64> {
    let mesh = l2.mesh();
    let n_elems = mesh.n_elements();
    let mut b = vec![0.0_f64; l2.n_dofs()];

    for e in mesh.elem_iter() {
        let nodes  = mesh.element_nodes(e);
        let dofs   = l2.element_dofs(e);

        // Compute centroid
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let n_nodes = nodes.len();
        for &ni in nodes {
            let coords = mesh.node_coords(ni);
            cx += coords[0];
            cy += coords[1];
        }
        cx /= n_nodes as f64;
        cy /= n_nodes as f64;

        // B(x, 0) = cos(πx) − cos(πy)   (exact, μ-independent; no 1/μ here
        // because the weak equation has the factor in the E equation's RHS)
        let bval = (PI * cx).cos() - (PI * cy).cos();
        for &d in dofs {
            b[d as usize] = bval;
        }
    }
    let _ = n_elems; // suppress unused warning
    b
}

// ─── Compute ||B_h − B_exact(t)||_{L2} ──────────────────────────────────────

fn l2_error_b<M: MeshTopology>(l2: &L2Space<M>, b_h: &[f64], t: f64, _mu: f64) -> f64 {
    let mesh = l2.mesh();
    let mut err2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs  = l2.element_dofs(e);

        // Element measure (area for 2-D triangles)
        let a = mesh.node_coords(nodes[0]);
        let b = mesh.node_coords(nodes[1]);
        let c = mesh.node_coords(nodes[2]);
        let area = 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])).abs();

        // Centroid
        let cx = (a[0] + b[0] + c[0]) / 3.0;
        let cy = (a[1] + b[1] + c[1]) / 3.0;

        // Exact B(x, t) = cos(πt) · (cos πx − cos πy)
        let b_exact = (PI * t).cos() * ((PI * cx).cos() - (PI * cy).cos());

        // P0 approximation: constant on the element
        let b_approx = b_h[dofs[0] as usize];

        err2 += area * (b_approx - b_exact).powi(2);
    }
    err2.sqrt()
}

fn l2_norm_b_exact<M: MeshTopology>(l2: &L2Space<M>, t: f64, _mu: f64) -> f64 {
    let mesh = l2.mesh();
    let mut norm2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let a = mesh.node_coords(nodes[0]);
        let b = mesh.node_coords(nodes[1]);
        let c = mesh.node_coords(nodes[2]);
        let area = 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])).abs();
        let cx = (a[0] + b[0] + c[0]) / 3.0;
        let cy = (a[1] + b[1] + c[1]) / 3.0;
        let b_exact = (PI * t).cos() * ((PI * cx).cos() - (PI * cy).cos());
        norm2 += area * b_exact.powi(2);
    }
    norm2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    n:     usize,
    eps:   f64,
    mu:    f64,
    sigma: f64,
    dt:    f64,
    t_end: f64,
}

impl Args {
    fn n_steps(&self) -> usize {
        (self.t_end / self.dt).ceil() as usize
    }
}

fn parse_args() -> Args {
    let mut args = Args {
        n:     8,
        eps:   1.0,
        mu:    1.0,
        sigma: 0.0,
        dt:    0.01,
        t_end: 1.0,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--n"     => args.n     = iter.next().unwrap().parse().unwrap(),
            "--eps"   => args.eps   = iter.next().unwrap().parse().unwrap(),
            "--mu"    => args.mu    = iter.next().unwrap().parse().unwrap(),
            "--sigma" => args.sigma = iter.next().unwrap().parse().unwrap(),
            "--dt"    => args.dt    = iter.next().unwrap().parse().unwrap(),
            "--t-end" => args.t_end = iter.next().unwrap().parse().unwrap(),
            other => eprintln!("unknown flag: {other}"),
        }
    }
    args
}
