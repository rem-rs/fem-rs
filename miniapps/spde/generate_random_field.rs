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

/// MFEM `args.PrintOptions(cout)` (generate_random_field.cpp:149-152), via
/// `OptionsParser::PrintOptions` (optparser.cpp:331-360): the option echo in
/// `AddOption` order, `   --long-name value` per entry; ENABLE pairs print the
/// long_name whose value is true.  Doubles go through `fem_solver::fmt_g`
/// (`ostream <<` is `%g` at the default precision 6).
fn print_options(a: &Args) {
    use fem_solver::fmt_g;
    println!("Options used:");
    println!("   --mesh {}", a.mesh_file);
    println!("   --order {}", a.order);
    println!("   --refs {}", a.num_refs);
    println!("   --refs-parallel {}", a.num_parallel_refs);
    println!("   --topology {}", a.topological_support);
    println!("   --nu {}", fmt_g(a.nu));
    println!("   --tau {}", fmt_g(a.tau));
    println!("   --l1 {}", fmt_g(a.l[0]));
    println!("   --l2 {}", fmt_g(a.l[1]));
    println!("   --l3 {}", fmt_g(a.l[2]));
    println!("   --e1 {}", fmt_g(a.e[0]));
    println!("   --e2 {}", fmt_g(a.e[1]));
    println!("   --e3 {}", fmt_g(a.e[2]));
    println!("   --pl1 {}", fmt_g(a.pl[0]));
    println!("   --pl2 {}", fmt_g(a.pl[1]));
    println!("   --pl3 {}", fmt_g(a.pl[2]));
    println!("   --uniform-min {}", fmt_g(a.uniform_min));
    println!("   --uniform-max {}", fmt_g(a.uniform_max));
    println!("   --offset {}", fmt_g(a.offset));
    println!("   --scale {}", fmt_g(a.scale));
    println!("   --level-set-threshold {}", fmt_g(a.level_set_threshold));
    println!("   --number-of-particles {}", a.number_of_particles);
    println!(
        "   {}",
        if a.paraview_export {
            "--paraview-visualization"
        } else {
            "--no-paraview-visualization"
        }
    );
    println!(
        "   {}",
        if a.glvis_export { "--visualization" } else { "--no-visualization" }
    );
    println!(
        "   {}",
        if a.uniform_rf { "--uniform-rf" } else { "--no-uniform-rf" }
    );
    println!(
        "   {}",
        if a.random_seed { "--random-seed" } else { "--no-random-seed" }
    );
    println!(
        "   {}",
        if a.compute_boundary_integrals {
            "--compute-boundary-integrals"
        } else {
            "--no-compute-boundary-integrals"
        }
    );
}

fn main() {
    let args = parse_args();

    // MFEM echoes the parsed options (`args.PrintOptions(cout)`,
    // generate_random_field.cpp:149-152) before reading the mesh.
    print_options(&args);

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
    // MFEM `boundary.Print(cout, 6)` (general/array.cpp:24-38): items
    // separated by " ", '\n' after every `width` items or at the end (no
    // trailing space before the final newline).
    for (i, t) in boundary.iter().enumerate() {
        print!("{t}");
        if (i + 1) % 6 == 0 || i + 1 == boundary.len() {
            println!();
        } else {
            print!(" ");
        }
    }

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
    let bc = Boundary::new();
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
    fn white_noise_rhs_matches_cpp_2x2_quads() {        let path = write_temp("spde_ref_quad.mesh", reference_mesh());
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

    /// End-to-end solved-field comparison against the serial C++ MFEM
    /// reference (`mpirun -np 1`, GCC, hypre BoomerAMG CG; probe dumps
    /// embedded below, round-105): ref-square refined 3× (81 dofs, nu = 4,
    /// anisotropic Θ, seed 2147483647 via `-no-rs`) and ref-cube refined
    /// 2× (125 dofs, nu = 2, octet truss, same fixed seed).
    ///
    /// The white-noise RHS is bit-exact (tests above, same seed chain); the
    /// solved field differs only through the linear-solve path: fem-rs AMG-CG
    /// stops on its own (preconditioned-residual) criterion around a true
    /// residual of 1e-7..1e-6 (8–11 iterations) where hypre BoomerAMG CG
    /// iterates the full rtol 1e-12 (kernel debt D976), so the composed
    /// 13-solve fractional chain agrees to the measured 3.1e-6 max relative
    /// deviation (square, round-105).  Pins hold 1e-5.
    /// MFEM stock `data/ref-square.mesh` (unit square, 1 quad) and
    /// `data/ref-cube.mesh` (unit cube, 1 hex), verbatim — embedded because
    /// `data/` is git-ignored (the tracked stock meshes are the exception).
    fn ref_square_mesh() -> &'static str {
        "MFEM mesh v1.0

dimension
2

elements
1
1 3 0 1 2 3

boundary
4
1 1 0 1
2 1 1 2
3 1 2 3
4 1 3 0

vertices
4
2
0 0
1 0
1 1
0 1
"
    }

    fn ref_cube_mesh() -> &'static str {
        "MFEM mesh v1.0

dimension
3

elements
1
1 5 0 1 2 3 4 5 6 7

boundary
6
1 3 3 2 1 0
2 3 0 1 5 4
3 3 1 2 6 5
4 3 2 3 7 6
5 3 3 0 4 7
6 3 4 5 6 7

vertices
8
3
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
"
    }

    /// Full 81-dof C++ probe dump (serial `grf_probe -m data/ref-square.mesh
    /// -r 3 -rp 0 -nu 4 -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 -t 0.08 -top 1
    /// -no-rs -no-vis -no-pvis`, %.17g of `u` right after
    /// `GenerateRandomField`, round-105).
    fn cpp_field_square81() -> [f64; 81] {
        [
            -6.78849641837103168e-01,
        1.84010049771581558e-01,
        3.66009467559550306e+00,
        -2.75475636106941835e+00,
        1.07193422287158135e+00,
        -2.23664098668094846e+00,
        -1.23458742133698096e+00,
        -8.15179380834383238e-01,
        -1.59609792529443906e-01,
        4.99803835162539012e-01,
        8.19792338378218854e-01,
        8.71287320125005871e-01,
        -6.50425965460365685e-01,
        6.27230257973367067e-01,
        1.18322593808475152e-01,
        6.46690435938322428e-01,
        3.47625004550899175e-01,
        -7.12002629111050056e-01,
        -7.88365663260918503e-01,
        -5.98878327641700570e-01,
        1.06977785666427169e-02,
        1.44973592047473510e+00,
        4.21596973692518751e-01,
        -2.26224971288728716e+00,
        7.25396062881868531e-01,
        -1.20555063822695230e+00,
        -1.34270791338385265e+00,
        4.04823138454060760e-01,
        8.18785085749947239e-01,
        5.70479491641040171e-01,
        -1.65465942919774500e+00,
        1.30194456097007993e+00,
        -1.01033741298102031e+00,
        -7.77076116154447050e-02,
        2.49714710832830311e-01,
        -1.01790762927155365e+00,
        6.74021943268180879e-01,
        2.34559877108774684e+00,
        -7.71530076580412283e-01,
        4.35320108779989501e-01,
        9.58958022172134994e-01,
        4.47952380848599840e-01,
        9.02168730278783831e-02,
        6.02730988013579938e-01,
        -3.18659307058727692e-01,
        6.79534565260084600e-01,
        7.79878794857223001e-01,
        9.08753171084057287e-01,
        -1.18698736571027275e+00,
        1.17818110706183019e-01,
        4.20794907505839602e-02,
        -1.14213940922442067e-01,
        -3.01173921690041713e+00,
        -2.38710831071047513e+00,
        -1.91487312060814985e-01,
        6.40602022469419663e-01,
        2.04091094587541422e+00,
        3.69641809315105652e-01,
        -1.04798336360221911e+00,
        -5.53668337915022901e-01,
        -6.74363937918700973e-01,
        -2.82291893998011112e+00,
        -1.88994130958119610e+00,
        -1.45599324072166558e+00,
        7.09925062738768431e-01,
        7.60418088748624377e-01,
        -1.36607997490739419e+00,
        3.08775932864188085e-01,
        2.09949177170705104e-01,
        -1.32294404048383329e+00,
        1.61482488519002126e-01,
        9.32475741113806711e-01,
        -9.25447926846057756e-02,
        -1.87401872607031061e-01,
        1.14706960884503628e+00,
        1.18186626187761712e-01,
        5.10833441458647813e-02,
        2.40285045709950018e-01,
        -7.55808108072491658e-01,
        -4.57446804111107053e-01,
        1.22988070399146235e-01,
        ]
    }

    /// First 8 dofs of the C++ cube probe (`-m data/ref-cube.mesh -r 2 -rp 0
    /// -no-rs -no-vis -no-pvis`, round-105).
    fn cpp_field_cube8() -> [f64; 8] {
        [
            1.53210264866688872e-01,
        -1.20655384778545494e+00,
        -7.61328053681114697e-02,
        -6.38856754286150297e-01,
        -4.18592176384814607e-01,
        -1.62414243267806707e-01,
        1.16969406779251120e-01,
        -1.10810154577703202e+00,
        ]
    }

    fn solve_fixed_seed_square_r3() -> Vec<f64> {
        let path = write_temp("spde_ref_square_r105.mesh", ref_square_mesh());
        let mfem = read_mfem_file(path.to_str().unwrap()).unwrap();
        let mut mesh: Mesh<2> = mfem.mesh2d.unwrap();
        for _ in 0..3 {
            mesh = refine_uniform(&mesh);
        }
        let space = H1Space::new(mesh.clone(), 1);
        let bc = Boundary::new();
        let mut solver = SpdeSolver::new(4.0, &bc, &space, 0.09, 0.03, 0.05, 0.0, 0.0, 0.0);
        let mut u = vec![0.0_f64; space.n_dofs()];
        solver.generate_random_field(&space, spde_seed(false), &mut u);
        u
    }

    fn solve_fixed_seed_cube_r2() -> Vec<f64> {
        let path = write_temp("spde_ref_cube_r105.mesh", ref_cube_mesh());
        let mfem = read_mfem_file(path.to_str().unwrap()).unwrap();
        let mut mesh: Mesh<3> = mfem.mesh3d.unwrap();
        for _ in 0..2 {
            mesh = refine_uniform_3d(&mesh);
        }
        let space = H1Space::new(mesh.clone(), 1);
        let bc = Boundary::new();
        let mut solver = SpdeSolver::new(2.0, &bc, &space, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0);
        let mut u = vec![0.0_f64; space.n_dofs()];
        solver.generate_random_field(&space, spde_seed(false), &mut u);
        u
    }

    /// Full-vector field pin (engineering tolerance, see the module note).
    #[test]
    fn spde_field_full_vector_matches_cpp_square_r3() {
        let u = solve_fixed_seed_square_r3();
        assert_eq!(u.len(), 81, "ref-square r3 dof count");
        let refs = cpp_field_square81();
        let mut max_rel = 0.0_f64;
        for (got, want) in u.iter().zip(refs.iter()) {
            max_rel = max_rel.max((got - want).abs() / want.abs().max(1.0));
        }
        assert!(
            max_rel < 1e-5,
            "square field max relative deviation {max_rel:.3e} exceeds 1e-5"
        );
    }

    #[test]
    fn spde_field_first_dofs_match_cpp_cube_r2() {
        let u = solve_fixed_seed_cube_r2();
        assert_eq!(u.len(), 125, "ref-cube r2 dof count");
        for (i, want) in cpp_field_cube8().iter().enumerate() {
            let got = u[i];
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "cube field dof {i}: {got:.17e} vs {want:.17e}"
            );
        }
    }

    use fem_assembly::standard::WhiteGaussianNoiseDomainLFIntegrator as WhiteGaussianNoise;
}
