//! # MFEM Example 9 — DG Advection
//!
//! Reference: `mfem/ex9.cpp`
//! `du/dt + v·∇u = 0` using DG with upwind flux.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex9_dg_advection -- -m data/square-disc.mesh -p 0 -no-vis
//! ```

use std::time::Instant;
use fem_assembly::interior_faces::InteriorFaceList;
use fem_assembly::Assembler;
use fem_assembly::standard::MassIntegrator;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology, amr::refine_uniform};
use fem_solver::{ode::Rk4, TimeStepper};
use fem_space::{L2Space, fe_space::FESpace};

// Lumped mass diagonal
fn lump(m: &CsrMatrix<f64>) -> Vec<f64> {
    (0..m.nrows).map(|i| { let mut s = 0.0; for k in m.row_ptr[i]..m.row_ptr[i+1] { s += m.values[k]; } s }).collect()
}

fn erfc(x: f64) -> f64 {
    let a = x.abs(); let t = 1.0 / (1.0 + 0.5 * a);
    let e = 1.0 - t * ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t - 0.284496736)*t + 0.254829592) * (-a*a).exp();
    if x >= 0.0 { e } else { -e }
}

fn u0(x: &[f64], problem: usize) -> f64 {
    if problem == 0 {
        // Gaussian
        (-40.0 * ((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2))).exp()
    } else {
        let (rx, ry, cx, cy, w) = (0.45, 0.25, 0.0, -0.2, 10.0);
        erfc(w*(x[0]-cx-rx))*erfc(-w*(x[0]-cx+rx))*erfc(w*(x[1]-cy-ry))*erfc(-w*(x[1]-cy+ry)) / 16.0
    }
}

fn main() {
    let args = Args::parse();
    let wall = Instant::now();

    // 1. Mesh
    let mesh: Mesh<2> = {
        eprintln!("  Mesh: {}", args.mesh);
        read_mfem_file(&args.mesh).expect("read mesh").mesh2d.expect("2D")
    };
    let mesh = if args.refine > 0 {
        let mut m = mesh;
        for _ in 0..args.refine { m = refine_uniform(&m); }
        eprintln!("  Refined: {} elems", m.n_elems());
        m
    } else { mesh };

    // 2. DG space + mass
    let space = L2Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("Number of unknowns: {}", n);
    let q = args.order * 2 + 1;
    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], q);
    let md = lump(&m);

    // 3. Assemble advection operator K where K*u = ∫ v ∇·(b u) dΩ
    //    Using DG weak form with upwind flux.
    //    For each element: ∫ v ∇·(b u) = -∫ (b·∇v) u + ∫_∂K v (b·n) u
    //    Interior faces use upwind: ∫_F [[v]] (vn_pos * u⁻ + vn_neg * u⁺)
    let b = [1.0_f64, 0.0_f64]; // velocity
    let mut coo = CooMatrix::new(n, n);

    // Per-element assembly (P1 triangle)
    for e in 0..space.mesh().n_elems() as u32 {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let _n = dofs.len(); // 3 for P1
        let nd = space.mesh().element_nodes(e);
        let (x0, x1, x2) = (
            space.mesh().node_coords(nd[0]), space.mesh().node_coords(nd[1]),
            space.mesh().node_coords(nd[2]));

        // Jacobian: J = [x1-x0, x2-x0]
        let (j00, j01) = (x1[0]-x0[0], x2[0]-x0[0]);
        let (j10, j11) = (x1[1]-x0[1], x2[1]-x0[1]);
        let det_j = j00*j11 - j01*j10;
        let abs_det = det_j.abs();
        let inv_det = 1.0/det_j;
        let area = 0.5 * abs_det;

        // P1 reference gradients: ∇φ₀=(-1,-1), ∇φ₁=(1,0), ∇φ₂=(0,1)
        let gref = [[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];
        // Physical gradients: ∇φ = J^{-T} * ∇φ_ref
        let mut gphys = [[0.0_f64; 2]; 3];
        for i in 0..3 {
            gphys[i][0] = (j11*gref[i][0] - j10*gref[i][1]) * inv_det;
            gphys[i][1] = (-j01*gref[i][0] + j00*gref[i][1]) * inv_det;
        }

        // Volume integral: weak form -∫ (b·∇v) u
        for i in 0..3 {
            let b_dot_grad_i = b[0]*gphys[i][0] + b[1]*gphys[i][1];
            for j in 0..3 {
                coo.add(dofs[i], dofs[j], -area * b_dot_grad_i * (1.0/3.0));
            }
        }
    }

    // Interior face upwind flux
    let ifl = InteriorFaceList::build(space.mesh());
    for face in &ifl.faces {
        let el = face.elem_left;
        let er = face.elem_right;
        let fnodes = &face.face_nodes;

        // Face geometry
        let xa = space.mesh().node_coords(fnodes[0]);
        let xb = space.mesh().node_coords(fnodes[1]);
        let dx = xb[0]-xa[0]; let dy = xb[1]-xa[1];
        let h = (dx*dx+dy*dy).sqrt();
        let nx = dy/h; let ny = -dx/h; // outward from left

        // Ensure outward normal from left element
        let el_nd = space.mesh().element_nodes(el);
        let cx = space.mesh().node_coords(el_nd[0]);
        let mx = (xa[0]+xb[0])/2.0; let my = (xa[1]+xb[1])/2.0;
        let ex = mx - cx[0]; let ey = my - cx[1];
        let (nx, ny) = if nx*ex + ny*ey < 0.0 { (-nx, -ny) } else { (nx, ny) };

        let vn = b[0]*nx + b[1]*ny;
        let vn_pos = vn.max(0.0);
        let vn_neg = vn.min(0.0);

        let dofs_l: Vec<usize> = space.element_dofs(el).iter().map(|&d| d as usize).collect();
        let dofs_r: Vec<usize> = space.element_dofs(er).iter().map(|&d| d as usize).collect();
        let nl = dofs_l.len(); let nr = dofs_r.len();

        // One-point quadrature at face midpoint: weight = h
        // φ values at midpoint for P1:
        // For the left element: φ values at the midpoint
        // We approximate using the average of the two face vertex values
        // Actually for P1, each face vertex belongs to exactly one element
        // At the face midpoint, the basis functions from the left element
        // are aligned with the face

        // Compute φ values at face midpoint for both elements
        let nodes_l = space.mesh().element_nodes(el);
        let nodes_r = space.mesh().element_nodes(er);
        let fe_l = [nodes_l[0], nodes_l[1], nodes_l[2]];
        let fe_r = [nodes_r[0], nodes_r[1], nodes_r[2]];

        // Which vertex indices of each element are on this face?
        // For P1, basis function at midpoint is 0.5 for the two face vertices
        let mut phi_l = vec![0.0; nl];
        let mut phi_r = vec![0.0; nr];
        // Left: find which of the 3 vertices are the face endpoints
        for i in 0..3 {
            if fe_l[i] == fnodes[0] || fe_l[i] == fnodes[1] {
                phi_l[i] = 0.5;
            }
        }
        for i in 0..3 {
            if fe_r[i] == fnodes[0] || fe_r[i] == fnodes[1] {
                phi_r[i] = 0.5;
            }
        }

        let w_f = h;
        // Upwind face flux: ∫_F [[v]] (vn_pos * u⁻ + vn_neg * u⁺)
        // K_ll: +φ⁻ · vn_pos · φ⁻
        // K_lr: +φ⁻ · vn_neg · φ⁺
        // K_rl: -φ⁺ · vn_pos · φ⁻
        // K_rr: -φ⁺ · vn_neg · φ⁺
        for i in 0..nl { for j in 0..nl {
            coo.add(dofs_l[i], dofs_l[j], w_f * phi_l[i] * vn_pos * phi_l[j]);
        }}
        for i in 0..nl { for j in 0..nr {
            coo.add(dofs_l[i], dofs_r[j], w_f * phi_l[i] * vn_neg * phi_r[j]);
        }}
        for i in 0..nr { for j in 0..nl {
            coo.add(dofs_r[i], dofs_l[j], -w_f * phi_r[i] * vn_pos * phi_l[j]);
        }}
        for i in 0..nr { for j in 0..nr {
            coo.add(dofs_r[i], dofs_r[j], -w_f * phi_r[i] * vn_neg * phi_r[j]);
        }}
    }

    let k_adv: CsrMatrix<f64> = coo.into_csr();
    eprintln!("  K_adv nnz: {}", k_adv.nnz());

    // Verify: K_adv * I should be small (mass conservation)
    let ones = vec![1.0; n];
    let mut k_ones = vec![0.0; n];
    k_adv.spmv(&ones, &mut k_ones);
    let max_k_ones = k_ones.iter().cloned().fold(0.0_f64, f64::max);
    let sum_k_ones: f64 = k_ones.iter().sum();
    eprintln!("  K_adv * I: max={:.4e}, sum={:.4e}", max_k_ones, sum_k_ones);

    // 4. Initial condition
    let iv = space.interpolate(&|x| u0(x, args.problem));
    let mut u: Vec<f64> = iv.as_slice().to_vec();
    let mx0 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("  Initial max: {:.4e}", mx0);

    // 5. Time integration: du/dt = -M^{-1} * K_adv * u
    let (k_ref, md_ref) = (&k_adv, &md);
    let rk4 = Rk4;
    let dt = args.dt.min(args.t_final);
    let n_steps = (args.t_final / dt).ceil() as usize;
    let mut t = 0.0;

    for step in 0..n_steps {
        let dta = dt.min(args.t_final - t);
        rk4.step(t, dta, &mut u, |_t, u, dudt| {
            let mut tmp = vec![0.0; n];
            k_ref.spmv(u, &mut tmp);
            for i in 0..n { dudt[i] = -tmp[i] / md_ref[i]; }
        });
        t += dta;
        if (step+1) % 200 == 0 || step+1 == n_steps {
            let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mn = u.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!("  t={:.4}  min={:.4e}  max={:.4e}", t, mn, mx);
        }
        if t >= args.t_final - 1e-14 { break; }
    }

    // 6. Final diagnostics
    let mut mu = vec![0.0; n];
    m.spmv(&u, &mut mu);
    let l2 = u.iter().zip(mu.iter()).map(|(a,b)| a*b).sum::<f64>().sqrt();
    let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("\nL2 norm of u: {:.10e}", l2);
    println!("max norm of u: {:.10e}", mx);
    eprintln!("  Wall: {:.3}s", wall.elapsed().as_secs_f64());
}

#[allow(dead_code)]
struct Args { mesh: String, problem: usize, refine: usize, order: u8, dt: f64, t_final: f64, no_vis: bool }
impl Args {
    fn parse() -> Self {
        let mut mesh = "data/square-disc.mesh".to_string();
        let (mut p, mut r, mut o, mut dt, mut tf) = (0usize, 2usize, 1u8, 0.001_f64, 0.5_f64);
        let mut nv = false;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() { match a.as_str() {
            "-m"|"--mesh" => { mesh = it.next().unwrap_or(mesh); }
            "-p"|"--problem" => { p = it.next().and_then(|s| s.parse().ok()).unwrap_or(0); }
            "-r"|"--refine" => { r = it.next().and_then(|s| s.parse().ok()).unwrap_or(2); }
            "-o"|"--order" => { o = it.next().and_then(|s| s.parse().ok()).unwrap_or(1); }
            "-dt"|"--time-step" => { dt = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.001); }
            "-tf"|"--final-time" => { tf = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.5); }
            "-no-vis"|"--no-visualization" => { nv = true; }
            _ => {}
        }}
        Args { mesh, problem: p, refine: r, order: o as u8, dt, t_final: tf, no_vis: nv }
    }
}
