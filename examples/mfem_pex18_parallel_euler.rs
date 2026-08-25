//!
//! Parallel Euler (pex18).
//!
//! 2D compressible Euler equations (DG, RK4) on a periodic square mesh.
//! Strategy: rank 0 assembles full system, does time integration, broadcasts.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex18_parallel_euler
//! cargo run --release --example mfem_pex18_parallel_euler -- --ranks 4
//! ```

use fem_assembly::dg::{DgHyperbolicConservationLaws, EulerFlux, RusanovFlux};
use fem_element::lagrange::tri::{TriP1, TriP2, TriP3};
use fem_element::lagrange::QuadL2GL;
use fem_element::reference::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::refine_uniform;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;
use fem_solver::ode::explicit::Rk4;
use fem_solver::ode::traits::TimeStepper;

fn make_ref_elem(mesh: &dyn MeshTopology, order: u8) -> Box<dyn ReferenceElement> {
    if mesh.element_type(0) == ElementType::Quad4 {
        Box::new(QuadL2GL::new(order as usize))
    } else {
        match order {
            1 => Box::new(TriP1),
            2 => Box::new(TriP2),
            3 => Box::new(TriP3),
            _ => Box::new(TriP1),
        }
    }
}

struct Args {
    mesh: String,
    problem: i32,
    refine: usize,
    order: u8,
    t_final: f64,
    dt: f64,
    cfl: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/periodic-square.mesh".into(),
        problem: 1,
        refine: 1,
        order: 1,
        t_final: 2.0,
        dt: -0.01,
        cfl: 0.3,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next().unwrap_or(a.mesh.clone()); }
            "-p" | "--problem" => { a.problem = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-r" | "--refine" => { a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            "-tf" | "--t-final" => { a.t_final = it.next().and_then(|v| v.parse().ok()).unwrap_or(2.0); }
            "-dt" | "--time-step" => { a.dt = it.next().and_then(|v| v.parse().ok()).unwrap_or(-0.01); }
            "-c" | "--cfl-number" => { a.cfl = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3); }
            _ => {}
        }
    }
    a
}

fn euler_initial_condition(problem: i32, x: &[f64], gamma: f64) -> Vec<f64> {
    match problem {
        1 => {
            let radius = 0.2;
            let minf = 0.5;
            let beta = 1.0 / 5.0;
            let xc = 0.5;
            let yc = 0.5;
            let dx = x[0] - xc;
            let dy = x[1] - yc;
            let r2 = dx * dx + dy * dy;
            let phi = (1.0 - r2 / (radius * radius)).exp();
            let u = minf * (1.0 - beta * dy / radius * phi);
            let v = minf * beta * dx / radius * phi;
            let t = 1.0 - (gamma - 1.0) * minf * minf * beta * beta / (8.0 * gamma * std::f64::consts::PI * std::f64::consts::PI) * phi;
            let rho = t.powf(1.0 / (gamma - 1.0));
            let p = rho.powf(gamma);
            let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
            vec![rho, rho * u, rho * v, e]
        }
        2 => {
            let radius = 0.2;
            let minf = 0.05;
            let beta = 1.0 / 50.0;
            let xc = 0.5;
            let yc = 0.5;
            let dx = x[0] - xc;
            let dy = x[1] - yc;
            let r2 = dx * dx + dy * dy;
            let phi = (1.0 - r2 / (radius * radius)).exp();
            let u = minf * (1.0 - beta * dy / radius * phi);
            let v = minf * beta * dx / radius * phi;
            let t = 1.0 - (gamma - 1.0) * minf * minf * beta * beta / (8.0 * gamma * std::f64::consts::PI * std::f64::consts::PI) * phi;
            let rho = t.powf(1.0 / (gamma - 1.0));
            let p = rho.powf(gamma);
            let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
            vec![rho, rho * u, rho * v, e]
        }
        3 => {
            let rho = 1.0 + 0.2 * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin();
            let u = 1.0;
            let v = 0.5;
            let p = 1.0;
            let e = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
            vec![rho, rho * u, rho * v, e]
        }
        _ => vec![1.0, 0.0, 0.0, 2.5],
    }
}

fn project_initial(u0: &dyn Fn(&[f64]) -> Vec<f64>, mesh: &dyn MeshTopology, order: u8, n_eq: usize) -> Vec<f64> {
    let ref_elem = make_ref_elem(mesh, order);
    let dp = ref_elem.n_dofs();
    let n_elems = mesh.n_elements();
    let mut sol = vec![0.0; n_elems * dp * n_eq];
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let nc = nodes.len();
        let mut centroid = vec![0.0; 2];
        for &n in nodes {
            let c = mesh.node_coords(n);
            centroid[0] += c[0] / nc as f64;
            centroid[1] += c[1] / nc as f64;
        }
        let u = u0(&centroid);
        for i in 0..dp {
            for eq in 0..n_eq {
                sol[e * dp * n_eq + i * n_eq + eq] = u[eq];
            }
        }
    }
    sol
}

fn compute_h_min(mesh: &dyn MeshTopology) -> f64 {
    let mut h_min = f64::INFINITY;
    for e in 0..mesh.n_elements() {
        let nodes = mesh.element_nodes(e as u32);
        let mut max_dist = 0.0f64;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let ci = mesh.node_coords(nodes[i]);
                let cj = mesh.node_coords(nodes[j]);
                let d = ((ci[0] - cj[0]).powi(2) + (ci[1] - cj[1]).powi(2)).sqrt();
                max_dist = max_dist.max(d);
            }
        }
        h_min = h_min.min(max_dist);
    }
    h_min
}

fn main() {
    let args = parse_args();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex18: Parallel Euler (DG, RK4) ===");
    println!("  Workers: {}, Problem: {}, Refine: {}, Order: {}", n_workers, args.problem, args.refine, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, f64)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // Solve: rank 0 assembles full system, does time integration, broadcasts.
        let (n_dofs, sol_final) = if rank == 0 {
            let mfem = read_mfem_file(&args.mesh).expect("Failed to read mesh");
            let mut mesh = mfem.mesh2d.expect("2D mesh required");
            for _ in 0..args.refine {
                mesh = refine_uniform(&mesh);
            }

            let dim = 2;
            let n_eq = dim + 2;
            let order = args.order;

            let flux = RusanovFlux { inner: EulerFlux { gamma: 1.4 } };
            let euler_op = DgHyperbolicConservationLaws::new(&mesh, order, Box::new(flux), true);
            let n_dofs = euler_op.n_dofs();

            let u0 = |x: &[f64]| euler_initial_condition(args.problem, x, 1.4);
            let mut sol = project_initial(&u0, &mesh, order, n_eq);

            let h_min = compute_h_min(&mesh);
            let mut dt = if args.dt > 0.0 {
                args.dt
            } else {
                let mut tmp = vec![0.0; n_dofs];
                euler_op.mult(&sol, &mut tmp);
                let max_cs = euler_op.max_char_speed();
                args.cfl * h_min / max_cs / (2.0 * order as f64 + 1.0)
            };

            let mut t = 0.0;
            let mut ti = 0usize;
            let rk4 = Rk4;
            loop {
                let dta = dt.min(args.t_final - t);
                let rhs = |_: f64, u: &[f64], dudt: &mut [f64]| euler_op.mult(u, dudt);
                rk4.step(t, dta, &mut sol, &rhs);
                t += dta;
                ti += 1;
                if args.dt <= 0.0 {
                    let max_cs = euler_op.max_char_speed();
                    dt = args.cfl * h_min / max_cs / (2.0 * order as f64 + 1.0);
                }
                if t >= args.t_final - 1e-8 * dt {
                    break;
                }
            }

            (n_dofs, sol)
        } else {
            (0, vec![])
        };

        // Broadcast n_dofs first.
        let mut n_bytes = if rank == 0 {
            (n_dofs as u64).to_le_bytes().to_vec()
        } else {
            vec![0u8; 8]
        };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_dofs: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        // Broadcast solution to all ranks.
        let mut sol_bytes = if rank == 0 {
            sol_final.iter().flat_map(|&v| v.to_le_bytes()).collect::<Vec<u8>>()
        } else {
            vec![0u8; n_dofs * 8]
        };
        comm.broadcast_bytes(0, &mut sol_bytes);
        let _sol: Vec<f64> = sol_bytes.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();

        if rank == 0 {
            *result_slot.lock().unwrap() = Some((n_dofs, 0.0, n_dofs as f64));
        }
    });

    let (n_dofs, _, _) = result.lock().unwrap().take().unwrap_or((0, 0.0, 0.0));
    println!("Number of unknowns: {}", n_dofs);
    println!("=== Done ===");
}
