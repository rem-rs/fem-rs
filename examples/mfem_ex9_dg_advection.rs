//! # MFEM Example 9 — DG Advection (upwind flux, explicit RK)
//!
//! Reference: `mfem/ex9.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex9_dg_advection -- -m data/periodic-square.mesh -p 0 -r 2 -o 3 -dt 0.005 -no-vis
//! ```

use std::time::Instant;
use fem_assembly::dg::dg_advection::{
    DGAdvectionIntegrator, assemble_dg_interior_faces,
};
use fem_assembly::interior_faces::InteriorFaceList;
use fem_assembly::Assembler;
use fem_assembly::postproc::coefficient::{CoeffCtx, VectorCoeff};
use fem_assembly::standard::MassIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{ode::Rk4, TimeStepper};
use fem_space::{L2Space, fe_space::FESpace};

// ─── Lumped mass ────────────────────────────────────────────────────────────

fn lumped_diag(mass: &CsrMatrix<f64>) -> Vec<f64> {
    let n = mass.nrows;
    let mut d = vec![0.0; n];
    for i in 0..n { for k in mass.row_ptr[i]..mass.row_ptr[i+1] { d[i] += mass.values[k]; } }
    d
}

// ─── Complementary error function (approx) ──────────────────────────────────

fn erfc(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.5 * a);
    let erf = 1.0 - t * ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * (-a * a).exp();
    if x >= 0.0 { erf } else { -erf }
}

// ─── Velocity coefficient ────────────────────────────────────────────────────

struct VelCoeff { problem: usize, bb_min: Vec<f64>, bb_max: Vec<f64> }

impl VectorCoeff for VelCoeff {
    fn eval(&self, ctx: &CoeffCtx<'_>, out: &mut [f64]) {
        let xr: Vec<f64> = (0..ctx.dim).map(|i| {
            2.0 * (ctx.x[i] - 0.5 * (self.bb_min[i] + self.bb_max[i])) / (self.bb_max[i] - self.bb_min[i])
        }).collect();
        let v = match self.problem {
            0 => vec![(2.0_f64 / 3.0).sqrt(), (1.0_f64 / 3.0).sqrt()],
            _ => { let w = std::f64::consts::PI / 2.0; vec![w * xr[1], -w * xr[0]] }
        };
        out[..v.len()].copy_from_slice(&v);
    }
}

// ─── Initial condition ───────────────────────────────────────────────────────

fn u0(x: &[f64], problem: usize, bb_min: &[f64], bb_max: &[f64]) -> f64 {
    #[allow(non_snake_case)]
    let X: Vec<f64> = (0..x.len()).map(|i| {
        2.0 * (x[i] - 0.5 * (bb_min[i] + bb_max[i])) / (bb_max[i] - bb_min[i])
    }).collect();
    match problem {
        0 | 1 => {
            // Translation & rotation: erfc-smoothed Gaussian bump
            let (rx, ry, cx, cy, w) = (0.45, 0.25, 0.0, -0.2, 10.0);
            erfc(w * (X[0] - cx - rx)) * erfc(-w * (X[0] - cx + rx))
          * erfc(w * (X[1] - cy - ry)) * erfc(-w * (X[1] - cy + ry)) / 16.0
        }
        _ => unimplemented!("problem {} not yet implemented", problem),
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let wall = Instant::now();

    // 1. Read mesh
    let mesh: Mesh<2> = {
        eprintln!("  Mesh file: {}", args.mesh);
        read_mfem_file(&args.mesh).expect("read mesh").mesh2d.expect("2D")
    };
    let mut bb_min = vec![f64::MAX; 2];
    let mut bb_max = vec![f64::MIN; 2];
    for n in 0..mesh.n_nodes() as u32 {
        let c = mesh.node_coords(n);
        for d in 0..2 { bb_min[d] = bb_min[d].min(c[d]); bb_max[d] = bb_max[d].max(c[d]); }
    }
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());
    eprintln!("  Box: [{:.3},{:.3}]x[{:.3},{:.3}]", bb_min[0], bb_max[0], bb_min[1], bb_max[1]);

    // 2. Refine
    let mesh = if args.refine > 0 {
        let mut m = mesh;
        for _ in 0..args.refine { m = refine_uniform(&m); }
        eprintln!("  Refined: {} nodes, {} elements", m.n_nodes(), m.n_elems());
        m
    } else { mesh };

    // 3. DG space (L2)
    let space = L2Space::new(mesh, args.order);
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {}", n_dofs);
    let quad = args.order * 2 + 1;

    // 4. Lumped mass
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], quad);
    let mass_diag = lumped_diag(&mass);

    // 5. DG advection operator
    let vel = VelCoeff { problem: args.problem, bb_min: bb_min.clone(), bb_max: bb_max.clone() };
    let dg_adv = DGAdvectionIntegrator { velocity: vel };
    let k_vol = Assembler::assemble_bilinear(&space, &[&dg_adv], quad);
    let ifl = InteriorFaceList::build(space.mesh());
    let mut coo = CooMatrix::new(n_dofs, n_dofs);
    for i in 0..n_dofs {
        for k in k_vol.row_ptr[i]..k_vol.row_ptr[i+1] {
            coo.add(i, k_vol.col_idx[k] as usize, k_vol.values[k]);
        }
    }
    assemble_dg_interior_faces(&mut coo, space.mesh(), &space, &ifl, args.order, quad, &dg_adv);
    let k_adv: CsrMatrix<f64> = coo.into_csr();
    eprintln!("  Adv op: {} nnz", k_adv.nnz());
    let rhs_bc = vec![0.0; n_dofs];

    // 6. Initial condition (convert Vector<f64> to Vec<f64>)
    let u_vec = space.interpolate(&|x| u0(x, args.problem, &bb_min, &bb_max));
    let mut u: Vec<f64> = u_vec.as_slice().to_vec();

    // 7. Time integration (RK4)
    // RHS: du/dt = -M_lump⁻¹ · (K_adv · u + rhs_bc)
    let k_adv_ref = &k_adv;
    let mass_diag_ref = &mass_diag;
    let rhs_bc_ref = &rhs_bc;
    let n = n_dofs;
    let rk4 = Rk4;
    let dt = args.dt.min(args.t_final);
    let n_steps = (args.t_final / dt).ceil() as usize;
    let mut t = 0.0;

    for step in 0..n_steps {
        let dt_actual = dt.min(args.t_final - t);
        rk4.step(t, dt_actual, &mut u, |_t, u, dudt| {
            let mut tmp = rhs_bc_ref.clone();
            k_adv_ref.spmv(u, &mut tmp);
            for i in 0..n { dudt[i] = -tmp[i] / mass_diag_ref[i]; }
        });
        t += dt_actual;
        if (step + 1) % 200 == 0 || step + 1 == n_steps {
            let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mn = u.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!("  t={:.4}  min={:.4e}  max={:.4e}", t, mn, mx);
        }
        if t >= args.t_final - 1e-14 { break; }
    }

    // 8. Final diagnostics
    let mut mu = vec![0.0; n_dofs];
    mass.spmv(&u, &mut mu);
    let l2 = u.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>().sqrt();
    let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("\nL2 norm of u: {:.10e}", l2);
    println!("max norm of u: {:.10e}", mx);
    eprintln!("  Wall: {:.3}s", wall.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Args { mesh: String, problem: usize, refine: usize, order: u8, dt: f64, t_final: f64, no_vis: bool }

impl Args {
    fn parse() -> Self {
        let mut mesh = "data/periodic-square.mesh".to_string();
        let mut p = 0usize; let mut r = 2usize; let mut o = 3u8;
        let mut dt = 0.005; let mut tf = 10.0; let mut nv = false;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() { match a.as_str() {
            "-m"|"--mesh" => { mesh = it.next().unwrap_or(mesh); }
            "-p"|"--problem" => { p = it.next().and_then(|s| s.parse().ok()).unwrap_or(0); }
            "-r"|"--refine" => { r = it.next().and_then(|s| s.parse().ok()).unwrap_or(2); }
            "-o"|"--order" => { o = it.next().and_then(|s| s.parse().ok()).unwrap_or(3); }
            "-dt"|"--time-step" => { dt = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.005); }
            "-tf"|"--final-time" => { tf = it.next().and_then(|s| s.parse().ok()).unwrap_or(10.0); }
            "-no-vis"|"--no-visualization" => { nv = true; }
            _ => {}
        }}
        Args { mesh, problem: p, refine: r, order: o, dt, t_final: tf, no_vis: nv }
    }
}
