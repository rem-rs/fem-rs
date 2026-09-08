//! Gaussian Random Fields of Matérn Covariance for Imperfect Materials
//! (MFEM `miniapps/spde/generate_random_field.cpp`, serial 1:1 port).
//!
//! 1:1 with the C++ miniapp except where noted (serial mesh refinement only,
//! ParaView export as a single VTU, no GLVis socket, `-cbi` boundary-error
//! verification cut). Run with `-no-rs` for a reproducible field.
//!
//! Compile with: cargo run --release --example generate_random_field --
//!   -m data/ref-cube.mesh -no-vis -no-rs

mod material_metrics;
mod spde_solver;
mod transformation;
mod util;
mod visualizer;

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

use material_metrics::{MaterialTopology, OctetTrussTopology, ParticleTopology};
use spde_solver::{spde_seed, Boundary, SpdeSolver};
use util::{fill_with_random_numbers, fill_with_random_rotations};
use visualizer::Visualizer;

enum TopologicalSupport {
    Particles,
    OctetTruss,
}

/// Uniform refinement dispatch for the serial mesh (the C++ refines the serial
/// mesh `num_refs` times and the ParMesh `num_parallel_refs` times; on one
/// rank both apply to the same mesh).
trait RefineOnce: Clone {
    fn refine_once(&self) -> Self;
}

impl RefineOnce for Mesh<2> {
    fn refine_once(&self) -> Self {
        refine_uniform(self)
    }
}

impl RefineOnce for Mesh<3> {
    fn refine_once(&self) -> Self {
        refine_uniform_3d(self)
    }
}

struct Args {
    mesh_file: String,
    order: i32,
    num_refs: i32,
    num_parallel_refs: i32,
    number_of_particles: i32,
    topological_support: i32,
    nu: f64,
    tau: f64,
    l: [f64; 3],
    e: [f64; 3],
    pl: [f64; 3],
    uniform_min: f64,
    uniform_max: f64,
    offset: f64,
    scale: f64,
    level_set_threshold: f64,
    paraview_export: bool,
    glvis_export: bool,
    uniform_rf: bool,
    random_seed: bool,
    compute_boundary_integrals: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/ref-cube.mesh".to_string(),
        order: 1,
        num_refs: 3,
        num_parallel_refs: 3,
        number_of_particles: 3,
        topological_support: 1, // kOctetTruss
        nu: 2.0,
        tau: 0.08,
        l: [0.02, 0.02, 0.02],
        e: [0.0, 0.0, 0.0],
        pl: [1.0, 1.0, 1.0],
        uniform_min: 0.0,
        uniform_max: 1.0,
        offset: 0.0,
        scale: 0.01,
        level_set_threshold: 0.0,
        paraview_export: true,
        glvis_export: true,
        uniform_rf: false,
        random_seed: true,
        compute_boundary_integrals: false,
    };

    let mut it = std::env::args().skip(1);
    let need = |it: &mut std::iter::Skip<std::env::Args>, opt: &str| -> String {
        it.next().unwrap_or_else(|| panic!("missing value for {opt}"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = need(&mut it, "-m"),
            "-o" | "--order" => a.order = need(&mut it, "-o").parse().unwrap(),
            "-r" | "--refs" => a.num_refs = need(&mut it, "-r").parse().unwrap(),
            "-rp" | "--refs-parallel" => a.num_parallel_refs = need(&mut it, "-rp").parse().unwrap(),
            "-top" | "--topology" => a.topological_support = need(&mut it, "-top").parse().unwrap(),
            "-nu" | "--nu" => a.nu = need(&mut it, "-nu").parse().unwrap(),
            "-t" | "--tau" => a.tau = need(&mut it, "-t").parse().unwrap(),
            "-l1" | "--l1" => a.l[0] = need(&mut it, "-l1").parse().unwrap(),
            "-l2" | "--l2" => a.l[1] = need(&mut it, "-l2").parse().unwrap(),
            "-l3" | "--l3" => a.l[2] = need(&mut it, "-l3").parse().unwrap(),
            "-e1" | "--e1" => a.e[0] = need(&mut it, "-e1").parse().unwrap(),
            "-e2" | "--e2" => a.e[1] = need(&mut it, "-e2").parse().unwrap(),
            "-e3" | "--e3" => a.e[2] = need(&mut it, "-e3").parse().unwrap(),
            "-pl1" | "--pl1" => a.pl[0] = need(&mut it, "-pl1").parse().unwrap(),
            "-pl2" | "--pl2" => a.pl[1] = need(&mut it, "-pl2").parse().unwrap(),
            "-pl3" | "--pl3" => a.pl[2] = need(&mut it, "-pl3").parse().unwrap(),
            "-umin" | "--uniform-min" => a.uniform_min = need(&mut it, "-umin").parse().unwrap(),
            "-umax" | "--uniform-max" => a.uniform_max = need(&mut it, "-umax").parse().unwrap(),
            "-off" | "--offset" => a.offset = need(&mut it, "-off").parse().unwrap(),
            "-s" | "--scale" => a.scale = need(&mut it, "-s").parse().unwrap(),
            "-lst" | "--level-set-threshold" => {
                a.level_set_threshold = need(&mut it, "-lst").parse().unwrap()
            }
            "-n" | "--number-of-particles" => {
                a.number_of_particles = need(&mut it, "-n").parse().unwrap()
            }
            "-pvis" | "--paraview-visualization" => a.paraview_export = true,
            "-no-pvis" | "--no-paraview-visualization" => a.paraview_export = false,
            "-vis" | "--visualization" => a.glvis_export = true,
            "-no-vis" | "--no-visualization" => a.glvis_export = false,
            "-urf" | "--uniform-rf" => a.uniform_rf = true,
            "-no-urf" | "--no-uniform-rf" => a.uniform_rf = false,
            "-rs" | "--random-seed" => a.random_seed = true,
            "-no-rs" | "--no-random-seed" => a.random_seed = false,
            "-cbi" | "--compute-boundary-integrals" => a.compute_boundary_integrals = true,
            "-no-cbi" | "--no-compute-boundary-integrals" => a.compute_boundary_integrals = false,
            other => {
                eprintln!("unknown option: {other}");
                std::process::exit(1);
            }
        }
    }

    if a.compute_boundary_integrals {
        // IntegrateBC relies on FaceElementTransformations (boundary quadrature
        // with element-side gradients), which the fem-rs boundary machinery
        // does not provide yet.
        eprintln!(
            "generate_random_field (Rust port): -cbi/--compute-boundary-integrals is not supported yet."
        );
        std::process::exit(3);
    }
    a
}

fn main() {
    let args = parse_args();

    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh file");
    if mfem.mesh3d.is_some() {
        run::<3>(mfem.mesh3d.unwrap(), &args);
    } else if mfem.mesh2d.is_some() {
        run::<2>(mfem.mesh2d.unwrap(), &args);
    } else {
        panic!("could not read a 2D or 3D mesh from {}", args.mesh_file);
    }
}

fn run<const D: usize>(mesh0: Mesh<D>, args: &Args)
where
    Mesh<D>: RefineOnce,
{
    let is_3d = D == 3;

    // 3. Refine the mesh to increase the resolution.
    let mut mesh = mesh0;
    for _ in 0..args.num_refs {
        mesh = mesh.refine_once();
    }
    for _ in 0..args.num_parallel_refs {
        mesh = mesh.refine_once();
    }

    // 4. Define a finite element space on the mesh.
    let space = H1Space::new(mesh.clone(), args.order as u8);
    let size = space.n_dofs();
    let boundary = spde_solver::unique_boundary_tags(&mesh);
    println!("Number of finite element unknowns: {size}");
    print!("Boundary attributes: ");
    for t in &boundary {
        print!("{t} ");
    }
    println!();

    // ========================================================================
    // II. Generate topological support
    // ========================================================================
    let n_dofs = space.n_dofs();
    let mut v = vec![0.0_f64; n_dofs];

    if is_3d {
        // II.1 Define the metric for the topological support.
        let mdm: Box<dyn MaterialTopology> = if args.topological_support == 1 {
            Box::new(OctetTrussTopology::new())
        } else if args.topological_support == 0 {
            // Create the same random particles on all processes: drawn once
            // here (serial = single rank, no broadcast needed).
            let np = args.number_of_particles as usize;
            let mut random_positions = vec![0.0_f64; 3 * np];
            let mut random_rotations = vec![0.0_f64; 9 * np];
            fill_with_random_numbers(&mut random_positions, 0.2, 0.8);
            fill_with_random_rotations(&mut random_rotations);
            Box::new(ParticleTopology::new(
                args.pl[0], args.pl[1], args.pl[2], &random_positions, &random_rotations,
            ))
        } else {
            println!("Error: Selected topological support not valid.");
            std::process::exit(1);
        };

        // II.2-II.3 Project tau - metric(x) onto the space (MFEM
        // ProjectCoefficient = nodal interpolation for H1).
        let tau = args.tau;
        let v_dofs = space.interpolate(&|x: &[f64]| {
            tau - mdm.compute_metric(&[x[0], x[1], x[2]])
        });
        v.copy_from_slice(v_dofs.as_slice());
    }

    // ========================================================================
    // III. Generate random imperfections via fractional PDE
    // ========================================================================
    let mut bc = Boundary::new();
    bc.print_info();
    bc.verify_defined_boundaries(&mesh);

    let mut solver = SpdeSolver::new(
        args.nu, &bc, &space, args.l[0], args.l[1], args.l[2], args.e[0], args.e[1], args.e[2],
    );
    let seed = spde_seed(args.random_seed);
    let mut u = vec![0.0_f64; n_dofs];
    solver.generate_random_field(&space, seed, &mut u);

    // ========================================================================
    // III. Combine topological support and random field
    // ========================================================================
    if args.uniform_rf {
        transformation::uniform_grf_transform(&mut u, args.uniform_min, args.uniform_max);
    }
    if args.scale != 1.0 {
        transformation::scale_transform(&mut u, args.scale);
    }
    if args.offset != 0.0 {
        transformation::offset_transform(&mut u, args.offset);
    }
    let mut w = vec![0.0_f64; n_dofs]; // Noisy material field.
    for i in 0..n_dofs {
        w[i] = u[i] + v[i];
    }
    let mut level_set = w.clone(); // Level set field.
    transformation::level_set_transform(&mut level_set, args.level_set_threshold);

    // ========================================================================
    // IV. Export visualization to ParaView and GLVis
    // ========================================================================
    let vis = Visualizer::new(
        &mesh,
        args.order as u8,
        &u,
        &v,
        &w,
        &level_set,
        is_3d,
    );
    if args.paraview_export {
        vis.export_to_para_view().expect("ParaView export failed");
    }
    if args.glvis_export {
        vis.send_to_gl_vis();
    }
}

#[cfg(test)]
mod white_noise_reference_tests {
    use super::*;

    /// End-to-end bit comparison against the serial C++ MFEM reference
    /// (GCC 13, MFEM 4.9): `LinearForm` + `WhiteGaussianNoiseDomainLFIntegrator`
    /// with `seed = 2147483647` on a 2×2 quad mesh and a 2×1×1 hex mesh.
    /// The meshes are the exact files printed by the C++ harness.
    fn reference_mesh() -> &'static str {
        "MFEM mesh v1.0

dimension
2

elements
4
1 3 0 1 4 3
1 3 3 4 7 6
1 3 4 5 8 7
1 3 1 2 5 4

boundary
8
1 1 0 1
1 1 1 2
3 1 7 6
3 1 8 7
4 1 3 0
4 1 6 3
2 1 2 5
2 1 5 8

vertices
9
2
0 0
0.5 0
1 0
0 0.5
0.5 0.5
1 0.5
0 1
0.5 1
1 1
"
    }

    fn reference_mesh_3d() -> &'static str {
        "MFEM mesh v1.0

dimension
3

elements
2
1 5 0 1 4 3 6 7 10 9
1 5 1 2 5 4 7 8 11 10

boundary
10
1 3 0 3 4 1
1 3 1 4 5 2
6 3 6 7 10 9
6 3 7 8 11 10
5 3 0 6 9 3
3 3 2 5 11 8
2 3 0 1 7 6
2 3 1 2 8 7
4 3 3 9 10 4
4 3 4 10 11 5

vertices
12
3
0 0 0
0.5 0 0
1 0 0
0 1 0
0.5 1 0
1 1 0
0 0 1
0.5 0 1
1 0 1
0 1 1
0.5 1 1
1 1 1
"
    }

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn white_noise_rhs_matches_cpp_2x2_quads() {
        let path = write_temp("spde_ref_quad.mesh", reference_mesh());
        let mfem = read_mfem_file(path.to_str().unwrap()).unwrap();
        let mesh: Mesh<2> = mfem.mesh2d.unwrap();
        let space = H1Space::new(mesh, 1);
        let mut integ = WhiteGaussianNoise::new(2147483647);
        let b = fem_assembly::Assembler::assemble_white_gaussian_noise(&space, &mut integ);
        let expected: [f64; 9] = [
            -0.020327630690266153,
            0.0031279669371782337,
            0.19815546338621942,
            -0.089633173391301113,
            0.23637376401052229,
            0.14813814997822342,
            -0.060631881410217588,
            0.29833555544216983,
            0.34042706584569393,
        ];
        for (got, want) in b.iter().zip(expected.iter()) {
            assert_eq!(got, want, "quad white noise RHS mismatch");
        }
    }

    #[test]
    fn white_noise_rhs_matches_cpp_2x1x1_hexes() {
        let path = write_temp("spde_ref_hex.mesh", reference_mesh_3d());
        let mfem = read_mfem_file(path.to_str().unwrap()).unwrap();
        let mesh: Mesh<3> = mfem.mesh3d.unwrap();
        let space = H1Space::new(mesh, 1);
        let mut integ = WhiteGaussianNoise::new(2147483647);
        let b = fem_assembly::Assembler::assemble_white_gaussian_noise(&space, &mut integ);
        let expected: [f64; 12] = [
            -0.016597440956963822,
            -0.073438816621871281,
            0.055123975415474501,
            -0.077712581902797284,
            0.20706134798024683,
            0.277957535318267,
            -0.0043778750812392053,
            0.16158209078904667,
            0.16767905959729565,
            -0.08172950545266211,
            0.16981706059705365,
            0.19598949499377527,
        ];
        // Shared dofs accumulate element contributions from 64-point hex
        // mass rules whose quadrature-point ordering differs from MFEM, so a
        // few entries deviate by 1–2 ulps (rel ~2e-16) — a floating-point
        // ordering difference, quantified here (tol = 1e-14 relative).
        for (i, (got, want)) in b.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-14 * want.abs().max(1e-300);
            assert!(
                (got - want).abs() <= tol,
                "hex white noise RHS mismatch at dof {i}: {got} vs {want}"
            );
        }
    }

    use fem_assembly::standard::WhiteGaussianNoiseDomainLFIntegrator as WhiteGaussianNoise;
}
