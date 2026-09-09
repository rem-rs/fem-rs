//! Extrapolation Miniapp: PDE-based extrapolation
//! (MFEM `miniapps/shifted/extrapolate.cpp`, serial `-np 1` 1:1 port).
//!
//! Extrapolates a finite element function from the elements where the level
//! set is positive into the rest of the domain by solving a sequence of
//! advection problems (Aslam / Bochkov–Gibou schemes via
//! `fem_assembly::dist_solver::Extrapolator`).
//!
//! Deviations from the C++ miniapp: ParaView output is not available in the
//! serial port; the C++ miniapp always writes ParaView files at the end, so
//! `-vis` (the C++ default) prints a notice and exits with code 3.
//!
//! Sample runs:
//!   cargo run --release --example shifted_extrapolate -- -rs 4 -p 0 -ed 1 -no-vis
//!   cargo run --release --example shifted_extrapolate -- -rs 4 -p 1 -ed 2 -no-vis
//!   cargo run --release --example shifted_extrapolate -- -m data/inline-hex.mesh -p 1 -ed 1 -rs 1 -no-vis

use fem_assembly::dist_solver::{AdvectionMode, Extrapolator, XtrapType};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_space::L2Space;

/// Level set of the known domain (extrapolate.cpp `domainLS`).
fn domain_ls(coord: &[f64], problem: i32) -> f64 {
    // Map from [0,1] to [-1,1].
    let dim = coord.len();
    let x = coord[0] * 2.0 - 1.0;
    let y = if dim > 1 { coord[1] * 2.0 - 1.0 } else { 0.0 };
    let z = if dim > 2 { coord[2] * 2.0 - 1.0 } else { 0.0 };

    match problem {
        0 => {
            // Sphere.
            0.75 - (x * x + y * y + z * z + 1e-12).sqrt()
        }
        1 => {
            // Star.
            assert!(dim > 1, "Problem 1 is not applicable to 1D.");
            0.60 - (x * x + y * y + z * z + 1e-12).sqrt()
                + 0.25 * (y * y * y * y * y + 5.0 * x * x * x * x * y - 10.0 * x * x * y * y * y)
                    / (x * x + y * y + z * z + 1e-12).powf(2.5)
                    * (0.5 * std::f64::consts::PI * z / 0.6).cos()
        }
        _ => panic!("Bad option for --problem!"),
    }
}

/// Input function (extrapolate.cpp `solution0`).
fn solution0(coord: &[f64]) -> f64 {
    // Map from [0,1] to [-1,1].
    let dim = coord.len();
    let x = coord[0] * 2.0 - 1.0 + 0.25;
    let y = if dim > 1 { coord[1] * 2.0 - 1.0 } else { 0.0 };
    let z = if dim > 2 { coord[2] * 2.0 - 1.0 } else { 0.0 };
    (std::f64::consts::PI * x).cos()
        * (std::f64::consts::PI * y).cos()
        * (std::f64::consts::PI * z).cos()
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut mesh_file = "data/inline-quad.mesh".to_string();
    let mut rs_levels = 2_i32;
    let mut ex_type = 0_i32; // Aslam
    let mut dg_mode = 0_i32; // HO
    let mut ex_degree = 1_i32;
    let mut order = 2_i32;
    let mut distance = 0.35_f64;
    let mut problem = 0_i32;
    let mut vis_on = true;

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        i += 1;
        let mut next = || {
            let v = argv.get(i).cloned().unwrap_or_default();
            i += 1;
            v
        };
        match arg {
            "-m" | "--mesh" => mesh_file = next(),
            "-rs" | "--refine-serial" => rs_levels = next().parse().unwrap_or(rs_levels),
            "-et" | "--extrap-type" => ex_type = next().parse().unwrap_or(ex_type),
            "-dg" | "--dg-mode" => dg_mode = next().parse().unwrap_or(dg_mode),
            "-ed" | "--extrap-degree" => ex_degree = next().parse().unwrap_or(ex_degree),
            "-o" | "--order" => order = next().parse().unwrap_or(order),
            "-d" | "--distance" => distance = next().parse().unwrap_or(distance),
            "-p" | "--problem" => problem = next().parse().unwrap_or(problem),
            "-vis" | "--visualization" => vis_on = true,
            "-no-vis" | "--no-visualization" => vis_on = false,
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }
    if vis_on {
        println!("ParaView output is not available in this serial port.");
        println!("Re-run with -no-vis to disable visualization output.");
        std::process::exit(3);
    }

    let mfem = read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("Failed to read MFEM mesh '{}': {e}", mesh_file));
    if let Some(mesh) = mfem.mesh2d {
        run::<2>(mesh, problem, rs_levels.max(0) as usize, order.max(0) as u8, ex_type, dg_mode, ex_degree, distance);
    } else {
        let mesh = mfem.mesh3d.expect("no mesh in file");
        run::<3>(mesh, problem, rs_levels.max(0) as usize, order.max(0) as u8, ex_type, dg_mode, ex_degree, distance);
    }
}

/// Mesh refinement helper uniform over the dimension.
trait RefineMany: Clone {
    fn refine_many(&self, levels: usize) -> Self;
}
impl RefineMany for Mesh<2> {
    fn refine_many(&self, levels: usize) -> Self {
        let mut m = self.clone();
        for _ in 0..levels {
            m = refine_uniform(&m);
        }
        m
    }
}
impl RefineMany for Mesh<3> {
    fn refine_many(&self, levels: usize) -> Self {
        let mut m = self.clone();
        for _ in 0..levels {
            m = refine_uniform_3d(&m);
        }
        m
    }
}

fn run<const D: usize>(
    mesh: Mesh<D>,
    problem: i32,
    rs_levels: usize,
    order: u8,
    ex_type: i32,
    dg_mode: i32,
    ex_degree: i32,
    distance: f64,
) where
    Mesh<D>: RefineMany,
{
    // Refine the mesh and distribute (serial: no distribution).
    let mesh = mesh.refine_many(rs_levels);

    // Input function, L2-projected (MFEM u.ProjectCoefficient(u0_coeff)).
    let pfes_l2 = L2Space::new(mesh.clone(), order);
    let u = GridFunction::from_projection(
        &pfes_l2,
        &solution0,
        2 * order as u8 + 1,
    );

    // Extrapolate (MFEM Extrapolator::Extrapolate).
    let xtrap = Extrapolator {
        xtrap_type: if ex_type == 0 {
            XtrapType::Aslam
        } else {
            XtrapType::Bochkov
        },
        advection_mode: if dg_mode == 0 {
            AdvectionMode::Ho
        } else {
            AdvectionMode::Lo
        },
        xtrap_degree: ex_degree,
        visualization: false,
        vis_steps: 50,
    };
    let ls = move |coord: &[f64]| domain_ls(coord, problem);
    let mut ux_dofs = vec![0.0_f64; pfes_l2.n_dofs()];
    xtrap.extrapolate(&ls, &u, distance, &mut ux_dofs);
    let ux = GridFunction::new(&pfes_l2, ux_dofs.clone());

    // PrintNorm: discrete l1 norm of the DOF vector.
    let l1_dof: f64 = ux_dofs.iter().map(|v| v.abs()).sum();
    println!("Solution l1 norm: {l1_dof:.12}");

    // PrintIntegral: L1 norm ∫|u| (ComputeL1Error against zero).
    let qo = 2 * order as u8 + 1;
    let l1_int = ux.compute_l1_error(&|_x: &[f64]| 0.0, qo);
    println!("Solution L1 norm: {l1_int:.12}");

    // Global errors against the input (MFEM: ux.ComputeL1Error(u_exact_coeff)
    // with the exact function being the *input* grid function).
    let exact = |x: &[f64]| u.get_value(x).unwrap_or(0.0);
    let err_l1 = ux.compute_l1_error(&exact, qo);
    let err_l2 = ux.compute_l2_error(&exact, qo);
    println!("Global L1 error: {err_l1:.12}");
    println!("Global L2 error: {err_l2:.12}");

    // Local errors in the cut elements.
    let (loc_l1, loc_l2, loc_li) = xtrap.compute_local_errors(&ls, &u, &ux);
    println!("Local  L1 error: {loc_l1:.12}");
    println!("Local  L2 error: {loc_l2:.12}");
    println!("Local  Li error: {loc_li:.12}");
}
