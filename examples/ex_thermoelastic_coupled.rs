//! # Thermoelastic Coupling (custom fem-rs example)
//!
//! Solves a staggered steady thermoelastic problem on a beam:
//!
//! 1. **Heat equation:**  `−∇·(κ ∇T) = 0`  with Dirichlet BC
//!    (T = hot on left boundary attr 1, T = cold on right boundary attr 2)
//!
//! 2. **Elasticity with thermal load:**
//!    `−∇·σ(u) = 0`  where `σ = C:(ε − α·ΔT·I)`
//!    with bottom face fixed (y ≈ 0).
//!
//! ## Usage
//! ```text
//! cargo run --example ex_thermoelastic_coupled
//! cargo run --example ex_thermoelastic_coupled -- -m data/beam-tri.mesh
//! cargo run --example ex_thermoelastic_coupled -- -m data/beam-quad.mesh -o 2
//! cargo run --example ex_thermoelastic_coupled -- -no-vis
//! ```
//!
//! ## CLI parameters
//!
//! | Short | Long              | Default               | Description                       |
//! |-------|-------------------|-----------------------|-----------------------------------|
//! | `-m`  | `--mesh`          | `data/beam-tri.mesh`  | Mesh file path                    |
//! | `-r`  | `--refine`        | `auto`                | Uniform refinements (auto = ≤5K)  |
//! | `-o`  | `--order`         | `1`                   | Finite element order              |
//! | `-T`  | `--hot-temp`      | `100.0`               | Hot side temperature (attr 1)     |
//! | `-t`  | `--cold-temp`     | `0.0`                 | Cold side temperature (attr 2)    |
//! | `-k`  | `--kappa`         | `50.0`                | Thermal conductivity              |
//! | `--alpha` |             | `1.2e-5`              | Thermal expansion coefficient     |
//! | `-E`  | `--young`         | `2.0e11`              | Young's modulus                   |
//! | `-nu` | `--poisson`       | `0.3`                 | Poisson ratio                     |
//! |       | `--no-vis`        | (flag)                | Disable GLVis visualization       |
//!
//! ## Output
//! - `temperature.gf` — temperature field (MFEM GF format)
//! - `displaced.mesh` — displaced mesh (MFEM mesh format)
//! - `thermal_stress.sol` — von Mises stress indicator (ASCII per-element)

use std::fs::File;
use std::io::Write;

use fem_assembly::physics::thermoelastic::{assemble_heat_system, assemble_thermal_expansion_rhs};
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_io::mfem::{read_mfem_file, write_gf_file, write_mfem_file};
use fem_mesh::{refine_uniform, topology::MeshTopology, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    H1Space, VectorH1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

fn main() {
    let args = parse_args();

    // --- Print parameters ---
    println!("Options used:");
    println!("   --mesh {:?}", args.mesh.as_deref().unwrap_or("data/beam-tri.mesh"));
    println!("   --order {}", args.order);
    println!("   --hot-temp {}", args.hot_temp);
    println!("   --cold-temp {}", args.cold_temp);
    println!("   --kappa {}", args.kappa);
    println!("   --alpha {}", args.alpha);
    println!("   --young {}", args.young);
    println!("   --poisson {}", args.poisson);
    if args.visualization {
        println!("   --visualization");
    } else {
        println!("   --no-visualization");
    }

    // --- Read mesh ---
    let mesh_path = args.mesh.as_deref().unwrap_or("data/beam-tri.mesh");
    let mfem_file = read_mfem_file(mesh_path).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<2> = mfem_file.mesh2d.expect("MFEM mesh must be 2D");

    let dim = 2usize;

    // --- Uniform refinement (auto or user-specified) ---
    let ref_levels = if let Some(r) = args.ref_levels {
        r
    } else {
        ((5000.0 / mesh.n_elems() as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize
    };
    println!("Refining mesh {} time(s)", ref_levels);
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }

    // 1. --- Temperature solve ---
    let pres_space = H1Space::new(mesh.clone(), args.order);
    let n_p = pres_space.n_dofs();
    println!("Number of temperature unknowns: {}", n_p);

    // Assemble heat system
    let (mut k_t, mut rhs_t) = assemble_heat_system(&mesh, args.kappa, 2);

    // Dirichlet BC for temperature: hot on attr 1, cold on attr 2
    let hot_dofs: Vec<usize> = boundary_dofs(&mesh, pres_space.dof_manager(), &[1])
        .iter().map(|&d| d as usize).collect();
    let cold_dofs: Vec<usize> = boundary_dofs(&mesh, pres_space.dof_manager(), &[2])
        .iter().map(|&d| d as usize).collect();

    println!("   Hot boundary DOFs (attr 1): {}", hot_dofs.len());
    println!("   Cold boundary DOFs (attr 2): {}", cold_dofs.len());

    // Apply temperature Dirichlet BC via eliminate_dirichlet-style approach
    // We use symmetric elimination for the heat system
    for &dof in &hot_dofs {
        k_t.apply_dirichlet_symmetric(dof, args.hot_temp, &mut rhs_t);
    }
    for &dof in &cold_dofs {
        k_t.apply_dirichlet_symmetric(dof, args.cold_temp, &mut rhs_t);
    }

    let mut temp = vec![0.0; n_p];
    let heat_cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10000,
        verbose: true,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&k_t, &rhs_t, &mut temp, &heat_cfg)
        .expect("heat solve failed");
    println!("   Heat solve converged.");

    // --- Write temperature field ---
    write_gf_file("temperature.gf", dim, &temp, "H1", args.order, 1)
        .expect("failed to write temperature.gf");
    println!("   Wrote temperature.gf");

    // Compute Lamé parameters
    let e = args.young;
    let nu = args.poisson;
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));

    // 2. --- Elasticity with thermal load ---
    let disp_space = VectorH1Space::new(mesh.clone(), args.order, dim as u8);
    let n_disp = disp_space.n_dofs();
    println!("Number of displacement unknowns: {}", n_disp);

    // Assemble elasticity matrix
    let mut k_u = Assembler::assemble_bilinear(
        &disp_space,
        &[&ElasticityIntegrator::new(lambda, mu)],
        2,
    );

    // Compute thermal expansion RHS
    let mut rhs_u = assemble_thermal_expansion_rhs(
        &mesh,
        &disp_space,
        &temp,
        args.cold_temp,
        args.alpha,
        lambda,
        mu,
        2,
    );

    // Fix bottom nodes (y ≈ 0) — displacement = 0
    let bottom_dofs: Vec<usize> = (0..mesh.n_nodes() as u32)
        .filter(|&n| mesh.node_coords(n)[1] < 0.01)
        .flat_map(|n| {
            let idx = n as usize;
            vec![idx * dim, idx * dim + 1]
        })
        .collect();
    println!("   Fixed displacement DOFs (y≈0): {}", bottom_dofs.len());

    for &dof in &bottom_dofs {
        k_u.apply_dirichlet_symmetric(dof, 0.0, &mut rhs_u);
    }

    let mut u = vec![0.0; n_disp];
    let elastic_cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10000,
        verbose: true,
        ..SolverConfig::default()
    };
    solve_pcg_jacobi(&k_u, &rhs_u, &mut u, &elastic_cfg)
        .expect("elasticity solve failed");
    println!("   Elasticity solve converged.");

    // 3. --- Output ---

    // Write displaced mesh (modify node coordinates directly)
    {
        let mut disp_mesh = mesh.clone();
        let n_nodes = disp_mesh.n_nodes();
        for i in 0..n_nodes {
            disp_mesh.coords[i * dim]     += u[i * dim];       // x-displacement
            disp_mesh.coords[i * dim + 1] += u[i * dim + 1];   // y-displacement
        }
        write_mfem_file("displaced.mesh", &disp_mesh)
            .expect("failed to write displaced.mesh");
        println!("   Wrote displaced.mesh");
    }

    // Write stress indicator (von Mises per element)
    write_stress_indicator(&mesh, &disp_space, &u, lambda, mu)
        .expect("failed to write thermal_stress.sol");
    println!("   Wrote thermal_stress.sol");

    // Print max displacement
    let max_u = u.iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    println!("Maximum displacement: {:.6e}", max_u);

    // Print max temperature
    let max_t = temp.iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    println!("Maximum temperature: {:.2}", max_t);

    // 4. --- Optional visualization via GLVis ---
    if args.visualization {
        println!("(GLVis visualization not yet implemented; set -no-vis to suppress this message)");
    }
}

/// Compute and write a von Mises stress indicator field (P1 element centroids).
///
/// Uses direct geometric computation per element (no reference element needed).
fn write_stress_indicator(
    mesh: &Mesh<2>,
    disp_space: &VectorH1Space<Mesh<2>>,
    u: &[f64],
    lambda: f64,
    mu: f64,
) -> std::io::Result<()> {
    let n_elem = mesh.n_elems();
    let mut stress_vals = Vec::with_capacity(n_elem);

    for e in mesh.elem_iter() {
        let nids = mesh.element_nodes(e);
        let npe = nids.len();
        if npe < 3 {
            continue;
        }

        // Node coordinates
        let c0 = mesh.node_coords(nids[0]);
        let c1 = mesh.node_coords(nids[1]);
        let c2 = mesh.node_coords(nids[2]);
        let x0 = c0[0]; let y0 = c0[1];
        let x1 = c1[0]; let y1 = c1[1];
        let x2 = c2[0]; let y2 = c2[1];

        let dofs: Vec<usize> = disp_space.element_dofs(e)
            .iter().map(|&d| d as usize).collect();

        if dofs.len() < 6 { continue; }
        let u0 = u[dofs[0]]; let v0 = u[dofs[1]];
        let u1 = u[dofs[2]]; let v1 = u[dofs[3]];
        let u2 = u[dofs[4]]; let v2 = u[dofs[5]];

        // Jacobian: J = [x1-x0  x2-x0; y1-y0 y2-y0]
        let j11 = x1 - x0; let j12 = x2 - x0;
        let j21 = y1 - y0; let j22 = y2 - y0;
        let det_j = j11 * j22 - j12 * j21;
        if det_j.abs() < 1e-30 { continue; }

        let inv_det = 1.0 / det_j;
        let i11 =  j22 * inv_det;
        let i12 = -j12 * inv_det;
        let i21 = -j21 * inv_det;
        let i22 =  j11 * inv_det;

        // Displacement gradient
        let du_dx = (u1 - u0) * i11 + (u2 - u0) * i12;
        let du_dy = (u1 - u0) * i21 + (u2 - u0) * i22;
        let dv_dx = (v1 - v0) * i11 + (v2 - v0) * i12;
        let dv_dy = (v1 - v0) * i21 + (v2 - v0) * i22;

        // Strain
        let eps_xx = du_dx;
        let eps_yy = dv_dy;
        let eps_xy = 0.5 * (du_dy + dv_dx);

        // Stress
        let tr_eps = eps_xx + eps_yy;
        let sig_xx = lambda * tr_eps + 2.0 * mu * eps_xx;
        let sig_yy = lambda * tr_eps + 2.0 * mu * eps_yy;
        let sig_xy = 2.0 * mu * eps_xy;

        // von Mises
        let s_xx = sig_xx - (sig_xx + sig_yy) / 3.0;
        let s_yy = sig_yy - (sig_xx + sig_yy) / 3.0;
        let vm =
            (1.5 * (s_xx * s_xx + s_yy * s_yy + 2.0 * sig_xy * sig_xy)).sqrt();

        stress_vals.push(vm);
    }

    let mut f = File::create("thermal_stress.sol")?;
    writeln!(f, "{}", stress_vals.len())?;
    for v in &stress_vals {
        writeln!(f, "{:.10e}", v)?;
    }

    Ok(())
}

/// Command-line arguments.
#[derive(Debug)]
struct Args {
    mesh: Option<String>,
    ref_levels: Option<usize>,
    order: u8,
    hot_temp: f64,
    cold_temp: f64,
    kappa: f64,
    alpha: f64,
    young: f64,
    poisson: f64,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        ref_levels: None,
        order: 1,
        hot_temp: 100.0,
        cold_temp: 0.0,
        kappa: 50.0,
        alpha: 1.2e-5,
        young: 2.0e11,
        poisson: 0.3,
        visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next();
            }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|v| v.parse().ok());
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-T" | "--hot-temp" => {
                a.hot_temp = it.next().and_then(|v| v.parse().ok()).unwrap_or(100.0);
            }
            "-t" | "--cold-temp" => {
                a.cold_temp = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            }
            "-k" | "--kappa" => {
                a.kappa = it.next().and_then(|v| v.parse().ok()).unwrap_or(50.0);
            }
            "--alpha" => {
                a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.2e-5);
            }
            "-E" | "--young" => {
                a.young = it.next().and_then(|v| v.parse().ok()).unwrap_or(2.0e11);
            }
            "-nu" | "--poisson" => {
                a.poisson = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.3);
            }
            "-vis" | "--visualization" => {
                a.visualization = true;
            }
            "-no-vis" | "--no-visualization" => {
                a.visualization = false;
            }
            _ => {}
        }
    }
    a
}
