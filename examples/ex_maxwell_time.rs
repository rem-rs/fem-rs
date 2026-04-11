//! # Example: Time-domain Maxwell (Newmark-β)
//!
//! Solves the second-order time-domain Maxwell equation for the electric field:
//!
//! ```text
//!   ε ∂²E/∂t² + σ ∂E/∂t + curl(μ⁻¹ curl E) = J(t)    in Ω = [0,1]²
//!                                       n×E = 0          on ∂Ω
//! ```
//!
//! with the manufactured solution:
//!
//! ```text
//!   E(x, t) = sin(ω t) · (sin(πy), sin(πx))
//!   ω = π   (angular frequency)
//! ```
//!
//! which satisfies:
//! ```text
//!   curl curl E = (π² sin(πy), π² sin(πx))
//!   J(t) = ε(-ω² sin(ωt)) E₀ + σ(ω cos(ωt)) E₀ + sin(ωt) curl curl E₀
//!         where E₀ = (sin(πy), sin(πx))
//! ```
//!
//! The Newmark-β average acceleration scheme (β=1/4, γ=1/2) is used —
//! unconditionally stable and second-order accurate in time.
//!
//! ## Usage
//! ```
//! cargo run --example ex_maxwell_time
//! cargo run --example ex_maxwell_time -- --n 16 --dt 0.01 --t-end 1.0
//! cargo run --example ex_maxwell_time -- --n 8 --dt 0.05 --sigma 0.1
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_element::nedelec::TriND1;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{
    SolverConfig, solve_cg,
    ode::{Newmark, NewmarkState},
};
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs_hcurl},
};

// ─── Physical constants ───────────────────────────────────────────────────────

const OMEGA: f64 = PI; // angular frequency of manufactured solution

fn main() {
    let args = parse_args();

    println!("=== fem-rs: Time-domain Maxwell (Newmark-β) ===");
    println!("  Mesh: {}×{}, ε={:.2}, μ={:.2}, σ={:.2}", args.n, args.n, args.eps, args.mu, args.sigma);
    println!("  dt={:.4}, T={:.2}, steps={}", args.dt, args.t_end, args.n_steps());

    // ─── 1. Mesh + H(curl) space ─────────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = HCurlSpace::new(mesh, 1);
    let n_dof = space.n_dofs();
    println!("  Edge DOFs: {n_dof}");

    // ─── 2. Identify boundary DOFs (n×E = 0) ─────────────────────────────────
    let bnd_dofs: Vec<u32> = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4]);

    // ─── 3. Assemble ε·M (mass), σ·M (damping), K (curl-curl) ───────────────
    let mass_integ   = VectorMassIntegrator { alpha: args.eps };
    let damp_integ   = VectorMassIntegrator { alpha: args.sigma };
    let stiff_integ  = CurlCurlIntegrator   { mu: 1.0 / args.mu };

    let mut mass  = VectorAssembler::assemble_bilinear(&space, &[&mass_integ],  4);
    let mut damp  = VectorAssembler::assemble_bilinear(&space, &[&damp_integ],  4);
    let mut stiff = VectorAssembler::assemble_bilinear(&space, &[&stiff_integ], 4);

    // Apply Dirichlet BCs to all three matrices (zero out rows + diag=1).
    let zero_vals = vec![0.0_f64; bnd_dofs.len()];
    let mut dummy_rhs = vec![0.0_f64; n_dof];
    apply_dirichlet(&mut mass,  &mut dummy_rhs, &bnd_dofs, &zero_vals);
    apply_dirichlet(&mut damp,  &mut dummy_rhs, &bnd_dofs, &zero_vals);
    apply_dirichlet(&mut stiff, &mut dummy_rhs, &bnd_dofs, &zero_vals);

    // ─── 4. Build Newmark effective stiffness: K_eff = ε·M/(β dt²) + σ·M/(γ·dt) + K
    //   (see Newmark::step docs; we let Newmark handle this internally)
    let newmark = Newmark::default(); // β=0.25, γ=0.5

    // ─── 5. Initial conditions: E(0) = 0, Ė(0) = ω E₀ ──────────────────────
    let mut u = vec![0.0_f64; n_dof]; // E(0) = 0
    let init_vel = project_exact_vel(&space, 0.0);
    let f0 = assemble_force(&space, &args, 0.0);
    let mut state = NewmarkState::init_from(init_vel, &mass, &stiff, &u, &f0);

    // ─── 6. Time loop ─────────────────────────────────────────────────────────
    let dt       = args.dt;
    let n_steps  = args.n_steps();
    let mut t    = 0.0;

    let mut l2_err_last = 0.0;

    let print_every = (n_steps / 5).max(1);

    for step in 0..n_steps {
        t += dt;

        // Assemble force at t_{n+1}.
        let mut force = assemble_force(&space, &args, t);
        // Apply BC to force.
        for &d in &bnd_dofs { force[d as usize] = 0.0; }

        // Newmark step: ε M ü + σ M u̇ + K u = f(t)
        // We add damping to the effective system manually here by augmenting
        // stiffness: K_aug = K + σ/(γ dt) M  and effective force.
        // However, fem-rs Newmark::step handles only undamped M ü + K u = f.
        // For the damped case we reformulate as a first-order system manually.
        newmark_damped_step(
            &mass, &damp, &stiff, &force, dt, &newmark,
            &mut u, &mut state, &bnd_dofs,
        );

        if step % print_every == print_every - 1 || step == n_steps - 1 {
            l2_err_last = l2_error_hcurl(&space, &u, t);
            println!("  t={t:.3}  L² err={l2_err_last:.3e}");
        }
    }

    println!("  Final t={t:.3}, L² error = {l2_err_last:.4e}");
    println!("  (Expected O(h) in space + O(dt²) in time for ND1 + Newmark-β)");
}

// ─── Damped Newmark step: ε M ü + σ M u̇ + K u = f ──────────────────────────
//
// Classic Newmark average acceleration for the damped second-order system.
// K_eff = ε·M / (β dt²) + σ·M·γ/(β·dt) + K
// f_eff = f_{n+1} + M/（β dt²) [u_n + dt v_n + (0.5-β)dt² a_n]
//                  + σ M [γ/(β dt) u_n + (γ/β - 1) v_n + dt(γ/(2β)-1) a_n]

fn newmark_damped_step(
    mass:  &fem_linalg::CsrMatrix<f64>,
    damp:  &fem_linalg::CsrMatrix<f64>,
    stiff: &fem_linalg::CsrMatrix<f64>,
    force: &[f64],
    dt:    f64,
    nm:    &Newmark,
    u:     &mut [f64],
    st:    &mut NewmarkState,
    bnd:   &[u32],
) {
    let n   = u.len();
    let b   = nm.beta;
    let g   = nm.gamma;
    let dt2 = dt * dt;

    // Newmark predictors.
    let c_m = 1.0 / (b * dt2);
    let c_d = g / (b * dt);

    // Effective stiffness: K_eff v = K_eff * (coeff for u_{n+1})
    // We build K_eff = (1/(β dt²)) M + (γ/(β dt)) C + K
    // as a CsrMatrix combination.
    let n_dofs = n;
    let mut k_eff_dense = vec![0.0_f64; n_dofs * n_dofs];
    // Add mass term.
    add_scaled_dense(mass, c_m, &mut k_eff_dense, n_dofs);
    // Add damping term.
    add_scaled_dense(damp, c_d, &mut k_eff_dense, n_dofs);
    // Add stiffness.
    add_scaled_dense(stiff, 1.0, &mut k_eff_dense, n_dofs);

    // Effective force.
    let mut f_eff = force.to_vec();

    // Contribution from mass predictor.
    let mut m_pred = vec![0.0_f64; n];
    for i in 0..n {
        m_pred[i] = c_m * (u[i] + dt * st.vel[i] + (0.5 - b) * dt2 * st.acc[i]);
    }
    let mut tmp = vec![0.0_f64; n];
    mass.spmv(&m_pred, &mut tmp);
    for i in 0..n { f_eff[i] += tmp[i]; }

    // Contribution from damping predictor.
    let mut d_pred = vec![0.0_f64; n];
    for i in 0..n {
        d_pred[i] = c_d * u[i]
            + (g / b - 1.0) * st.vel[i]
            + dt * (g / (2.0 * b) - 1.0) * st.acc[i];
    }
    damp.spmv(&d_pred, &mut tmp);
    for i in 0..n { f_eff[i] += tmp[i]; }

    // Enforce Dirichlet BCs on K_eff and f_eff.
    for &d in bnd {
        let di = d as usize;
        // Zero row and column.
        for j in 0..n_dofs {
            k_eff_dense[di * n_dofs + j] = 0.0;
            k_eff_dense[j * n_dofs + di] = 0.0;
        }
        k_eff_dense[di * n_dofs + di] = 1.0;
        f_eff[di] = 0.0;
    }

    // Solve K_eff u_{n+1} = f_eff using dense LU (small systems in examples).
    let k_csr = dense_to_csr(&k_eff_dense, n_dofs);
    let mut u_new = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-14, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    solve_cg(&k_csr, &f_eff, &mut u_new, &cfg).expect("Newmark solve failed");

    // Update acceleration and velocity.
    let mut acc_new = vec![0.0_f64; n];
    for i in 0..n {
        acc_new[i] = c_m * (u_new[i] - u[i] - dt * st.vel[i]) - (0.5 / b - 1.0) * st.acc[i];
    }
    let mut vel_new = vec![0.0_f64; n];
    for i in 0..n {
        vel_new[i] = st.vel[i] + dt * ((1.0 - g) * st.acc[i] + g * acc_new[i]);
    }

    // Commit.
    u.copy_from_slice(&u_new);
    st.vel.copy_from_slice(&vel_new);
    st.acc.copy_from_slice(&acc_new);
}

fn add_scaled_dense(mat: &fem_linalg::CsrMatrix<f64>, scale: f64, out: &mut [f64], n: usize) {
    let dense = mat.to_dense();
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] += scale * dense[i * n + j];
        }
    }
}

fn dense_to_csr(dense: &[f64], n: usize) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = dense[i * n + j];
            if v.abs() > 1e-300 {
                coo.add(i, j, v);
            }
        }
    }
    coo.into_csr()
}

// ─── Force vector: J(x,t) = ε(-ω² sin(ωt))E₀ + σ(ω cos(ωt))E₀ + sin(ωt) curl curl E₀ ────

struct MaxwellTimeForce {
    t:     f64,
    eps:   f64,
    sigma: f64,
    mu:    f64,
}

impl VectorLinearIntegrator for MaxwellTimeForce {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        // E₀ = (sin(πy), sin(πx)), curl curl E₀ = (π² sin(πy), π² sin(πx))
        let e0x = (PI * x[1]).sin();
        let e0y = (PI * x[0]).sin();
        let curl2_e0x = PI * PI * e0x;
        let curl2_e0y = PI * PI * e0y;

        let sin_wt = (OMEGA * self.t).sin();
        let cos_wt = (OMEGA * self.t).cos();
        let w2 = OMEGA * OMEGA;

        // J = ε(-ω²)sin(ωt) E₀ + σ ω cos(ωt) E₀ + sin(ωt)/μ curl curl E₀
        let jx = self.eps * (-w2) * sin_wt * e0x
               + self.sigma * OMEGA * cos_wt * e0x
               + sin_wt / self.mu * curl2_e0x;
        let jy = self.eps * (-w2) * sin_wt * e0y
               + self.sigma * OMEGA * cos_wt * e0y
               + sin_wt / self.mu * curl2_e0y;

        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * jx + qp.phi_vec[i * 2 + 1] * jy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

fn assemble_force(space: &HCurlSpace<SimplexMesh<2>>, args: &Args, t: f64) -> Vec<f64> {
    let integ = MaxwellTimeForce { t, eps: args.eps, sigma: args.sigma, mu: args.mu };
    VectorAssembler::assemble_linear(space, &[&integ], 4)
}

// ─── Initial velocity: Ė(0) = ω E₀ ─────────────────────────────────────────

fn project_exact_vel(space: &HCurlSpace<SimplexMesh<2>>, t: f64) -> Vec<f64> {
    // Ė = ω cos(ωt) E₀  →  at t=0: Ė(0) = ω E₀
    let cos_wt = (OMEGA * t).cos();
    let vel_fn = |x: &[f64]| vec![
        OMEGA * cos_wt * (PI * x[1]).sin(),
        OMEGA * cos_wt * (PI * x[0]).sin(),
    ];
    space.interpolate_vector(&vel_fn).as_slice().to_vec()
}

// ─── L² error ────────────────────────────────────────────────────────────────

fn l2_error_hcurl(space: &HCurlSpace<SimplexMesh<2>>, uh: &[f64], t: f64) -> f64 {
    let mesh      = space.mesh();
    let ref_elem  = TriND1;
    let quad      = ref_elem.quadrature(6);
    let n_ldofs   = ref_elem.n_dofs();
    let sin_wt    = (OMEGA * t).sin();

    let mut err2 = 0.0_f64;
    let mut ref_phi = vec![0.0; n_ldofs * 2];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();
        let inv_det = 1.0 / (j00 * j11 - j01 * j10);
        let jit00 =  j11 * inv_det; let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det; let jit11 =  j00 * inv_det;

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j;
            let xp = [x0[0] + j00*xi[0] + j01*xi[1], x0[1] + j10*xi[0] + j11*xi[1]];

            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            let mut eh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let phi_x = jit00 * ref_phi[i*2] + jit01 * ref_phi[i*2+1];
                let phi_y = jit10 * ref_phi[i*2] + jit11 * ref_phi[i*2+1];
                eh[0] += s * uh[dofs[i]] * phi_x;
                eh[1] += s * uh[dofs[i]] * phi_y;
            }

            // Exact: E = sin(ωt) * (sin(πy), sin(πx))
            let ex = sin_wt * (PI * xp[1]).sin();
            let ey = sin_wt * (PI * xp[0]).sin();

            let dx = eh[0] - ex; let dy = eh[1] - ey;
            err2 += w * (dx*dx + dy*dy);
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, dt: f64, t_end: f64, eps: f64, mu: f64, sigma: f64 }

impl Args {
    fn n_steps(&self) -> usize { (self.t_end / self.dt).round() as usize }
}

fn parse_args() -> Args {
    let mut a = Args { n: 16, dt: 0.05, t_end: 1.0, eps: 1.0, mu: 1.0, sigma: 0.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"      => { a.n      = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--dt"     => { a.dt     = it.next().unwrap_or("0.05".into()).parse().unwrap_or(0.05); }
            "--t-end"  => { a.t_end  = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--eps"    => { a.eps    = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--mu"     => { a.mu     = it.next().unwrap_or("1.0".into()).parse().unwrap_or(1.0); }
            "--sigma"  => { a.sigma  = it.next().unwrap_or("0.0".into()).parse().unwrap_or(0.0); }
            _ => {}
        }
    }
    a
}
