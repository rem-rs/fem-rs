use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator},
};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_cg, SolverConfig, Newmark, NewmarkState, Rk4, TimeStepper};
use fem_space::{
    H1Space, FESpace,
    constraints::{apply_dirichlet, boundary_dofs},
};

/// Exact solution: u(x,y,t) = cos(πct) sin(πx) sin(πy)
fn u_exact(x: &[f64], t: f64, c: f64) -> f64 {
    (PI * c * t).cos() * (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 23: Wave Equation (d²u/dt² = c² ∇²u) ===");

    // 1. Mesh and space
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  Mesh: {}×{}, P{}, {} DOFs", args.n, args.n, args.order, n);

    // 2. Assemble stiffness and mass matrices
    let diff_coeff = args.c * args.c;
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: diff_coeff }], 2 * args.order + 1);
    let mass   = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 2 * args.order + 1);
    let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 0.0)], 2 * args.order + 1);

    // 3. Dirichlet BCs: u=0 on ∂Ω
    let bdofs = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4]);
    let bvals = vec![0.0; bdofs.len()];

    let mut stiff_bc = stiff.clone();
    let mut mass_bc = mass.clone();
    let mut rhs_bc = rhs.clone();
    apply_dirichlet(&mut stiff_bc, &mut rhs_bc, &bdofs, &bvals);
    // Also apply BC to mass for implicit solve — zero Dirichlet rows
    // (Alternative: keep mass as-is and only constrain RHS)
    let mut rhs_bc_mass = rhs.clone();
    apply_dirichlet(&mut mass_bc, &mut rhs_bc_mass, &bdofs, &bvals);

    // 4. Initial condition: u(x,y,0) = sin(πx) sin(πy)
    let mut u = vec![0.0; n];
    for dof in 0..n as u32 {
        let coord = space.dof_manager().dof_coord(dof);
        u[dof as usize] = (PI * coord[0]).sin() * (PI * coord[1]).sin();
    }
    // Apply BC to initial condition
    for &d in &bdofs { u[d as usize] = 0.0; }
    let _v0 = vec![0.0; n]; // zero initial velocity

    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };

    match args.scheme.as_str() {
        "newmark" => {
            println!("  Scheme: Newmark-beta (β=1/4, γ=1/2)");
            println!("  dt={}, T={}", args.dt, args.t_final);

            let newmark = Newmark::default();
            let mut state = NewmarkState::new(n);
            // Initial acceleration: a₀ = M⁻¹(-K u₀)
            let mut ku = vec![0.0; n];
            stiff.spmv(&u, &mut ku);
            let mut rhs_a = vec![0.0; n];
            for i in 0..n { rhs_a[i] = -ku[i]; }
            for &d in &bdofs { rhs_a[d as usize] = 0.0; }
            let mut a0 = vec![0.0; n];
            solve_cg(&mass_bc, &rhs_a, &mut a0, &cfg).unwrap();
            state.acc.copy_from_slice(&a0);

            let mut u_hist = u.clone();
            let t_end = args.t_final;
            let dt = args.dt;
            let n_steps = (t_end / dt).round() as usize;
            let t_final = n_steps as f64 * dt;

            for _step in 0..n_steps {
                newmark.step(&mass_bc, &stiff_bc, &rhs_bc, dt, &mut u_hist, &mut state, &[]);
            }

            let exact = u_exact(&[0.25, 0.25], t_final, args.c);
            println!("  u(0.25,0.25) ≈ {:.6e}, exact = {:.6e}", u_hist[n / 4], exact);
            println!("Done.");
        }
        "rk4" => {
            println!("  Scheme: RK4 (explicit, for the equivalent first-order system)");
            println!("  dt={}, T={}", args.dt, args.t_final);

            // Convert to first-order: [u̇, v̇] = [v, -M⁻¹Ku]
            // where v = du/dt
            // We use a mass lumping approach for explicit integration

            // Build lumped mass (row sum for explicit)
            let mut mlump = vec![0.0; n];
            for i in 0..n {
                for ptr in mass.row_ptr[i]..mass.row_ptr[i + 1] {
                    if mass.col_idx[ptr] as usize == i {
                        mlump[i] = mass.values[ptr];
                    }
                }
            }

            let uu = u.clone();
            let vv = vec![0.0; n]; // velocity
            let t_end = args.t_final;
            let dt = args.dt;
            let n_steps = (t_end / dt).round() as usize;
            let t_final = n_steps as f64 * dt;

            // Prepare: y = [u; v] (size 2n)
            let mut y = Vec::with_capacity(2 * n);
            y.extend_from_slice(&uu);
            y.extend_from_slice(&vv);

            let rhs_fn = |_t: f64, y: &[f64], dydt: &mut [f64]| {
                let n = y.len() / 2;
                // du/dt = v
                for i in 0..n { dydt[i] = y[n + i]; }
                // dv/dt = -M⁻¹Ku
                let mut ku = vec![0.0; n];
                stiff.spmv(&y[..n], &mut ku);
                for i in 0..n {
                    if mlump[i].abs() > 1e-30 {
                        dydt[n + i] = -ku[i] / mlump[i];
                    } else {
                        dydt[n + i] = 0.0;
                    }
                }
                // Apply BC: zero on boundary
                for &d in &bdofs {
                    dydt[d as usize] = 0.0;
                    dydt[n + d as usize] = 0.0;
                }
            };

            let rk4 = Rk4;
            let mut t = 0.0;
            for _ in 0..n_steps {
                let h = dt.min(t_end - t);
                rk4.step(t, h, &mut y, &rhs_fn);
                t += h;
            }

            let u_final: Vec<f64> = y[..n].to_vec();
            let exact = u_exact(&[0.25, 0.25], t_final, args.c);
            println!("  u(0.25,0.25) ≈ {:.6e}, exact = {:.6e}", u_final[n / 4], exact);
            println!("Done.");
        }
        _ => {
            eprintln!("Unknown scheme '{}' — use 'newmark' or 'rk4'", args.scheme);
        }
    }
}

struct Args {
    n: usize,
    order: u8,
    c: f64,
    dt: f64,
    t_final: f64,
    scheme: String,
}

fn parse_args() -> Args {
    let mut a = Args { n: 20, order: 1, c: 1.0, dt: 0.001, t_final: 0.5, scheme: "newmark".to_string() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"       => { a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(20); }
            "--order"   => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "--c"       => { a.c = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
            "--dt"      => { a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.001); }
            "--T"       => { a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5); }
            "--scheme"  => { a.scheme = it.next().unwrap_or_else(|| "newmark".to_string()); }
            _           => {}
        }
    }
    a
}
